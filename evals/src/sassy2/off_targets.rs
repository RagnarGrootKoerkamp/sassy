use crate::sassy1::edlib_bench::sim_data::Alphabet;
use crate::sassy2::bench;
use needletail::parse_fastx_file;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct Config {
    guide_file: String,
    genome_file: String,
    min_benchtime: f64,
    warmup_iterations: usize,
    output_file: String,
    ks: Vec<usize>,
}

fn load_guide_sequences(path: &str) -> Vec<Vec<u8>> {
    let mut reader = parse_fastx_file(path).unwrap();
    let mut sequences = Vec::new();
    while let Some(record) = reader.next() {
        sequences.push(record.unwrap().seq().to_ascii_uppercase());
    }
    sequences
}

fn load_first_chromosome(path: &str) -> Option<Vec<u8>> {
    let mut reader = parse_fastx_file(path).unwrap();
    while let Some(record) = reader.next() {
        if let Ok(rec) = record {
            return Some(rec.seq().to_ascii_uppercase());
        }
    }
    None
}

fn guides_same_length(guides: &[Vec<u8>]) -> bool {
    guides
        .first()
        .map(|g| g.len())
        .map(|len| guides.iter().all(|g| g.len() == len))
        .unwrap_or(true)
}

pub fn run(config_path: &str) {
    let toml_str = fs::read_to_string(config_path).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    println!("Running off-target search benchmark");
    println!("Config: {:?}", config_path);
    println!("Guides: {}", config.guide_file);
    println!("Genome: first chromosome from {}", config.genome_file);
    println!("K values: {:?}", config.ks);
    println!(
        "Warmup: {}, min_benchtime: {} s",
        config.warmup_iterations, config.min_benchtime
    );
    println!("Output: {}", config.output_file);
    println!();

    let guides = load_guide_sequences(&config.guide_file);
    let chromosome = load_first_chromosome(&config.genome_file).expect("could not load genome");

    println!(
        "Loaded {} guides, chromosome length {}",
        guides.len(),
        chromosome.len()
    );

    if guides.is_empty() {
        panic!("No guide sequences loaded from {}", config.guide_file);
    }
    if chromosome.is_empty() {
        panic!("Empty chromosome from {}", config.genome_file);
    }

    let total_bytes = chromosome.len();
    let same_length = guides_same_length(&guides);
    if !same_length {
        println!("  Note: guides have mixed lengths; tiling benchmark skipped (zeros written).");
    }

    let query_len = guides.first().map(|g| g.len()).unwrap_or(0);
    let mut csv = bench::BenchCsv::new(&config.output_file).unwrap();

    for &k in &config.ks {
        println!("Benchmarking k={}", k);

        let suite = bench::benchmark_tools(
            &guides,
            &[chromosome.clone()],
            k,
            config.warmup_iterations,
            config.min_benchtime,
            1,
            &Alphabet::Iupac,
            false,
        );

        println!("{}", &suite);

        csv.write_row(
            guides.len(),
            chromosome.len(),
            query_len,
            k,
            &suite,
            total_bytes,
        )
        .unwrap();
    }

    println!(
        "\nOff-target benchmark complete. Results written to {}",
        config.output_file
    );
}
