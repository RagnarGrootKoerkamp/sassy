use crate::sassy1::edlib_bench::sim_data::Alphabet;
use crate::sassy2::bench;
use needletail::parse_fastx_file;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct Config {
    guide_file: String,
    genome_file: String,
    iterations: usize,
    warmup_iterations: usize,
    output_file: String,
    ks: Vec<usize>,
    run_edlib: Option<bool>,
    edlib_alphabet: Option<String>,
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

    let run_edlib = config.run_edlib.unwrap_or(false);
    let edlib_alphabet = config.edlib_alphabet.as_deref().unwrap_or("iupac");

    println!("Running off-target search benchmark");
    println!("Config: {:?}", config_path);
    println!("Guides: {}", config.guide_file);
    println!("Genome: first chromosome from {}", config.genome_file);
    println!("K values: {:?}", config.ks);
    println!(
        "Warmup: {}, iterations: {}",
        config.warmup_iterations, config.iterations
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

    let total_bytes = chromosome.len() * guides.len();
    let same_length = guides_same_length(&guides);
    if !same_length {
        println!("  Note: guides have mixed lengths; tiling benchmark skipped (zeros written).");
    }

    let query_len = guides.first().map(|g| g.len()).unwrap_or(0);
    let mut csv = bench::BenchCsv::new(&config.output_file).unwrap();

    for &k in &config.ks {
        println!("Benchmarking k={}", k);

        let search_results = bench::benchmark_individual_search(
            &guides,
            &chromosome,
            k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
            "Benchmarking",
            false,
        );

        let tiling_results = bench::benchmark_tiling(
            &guides,
            &chromosome,
            k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
            "Benchmarking",
            false,
        );

        let edlib_results = if run_edlib {
            let alphabet = match edlib_alphabet {
                "dna" => Alphabet::Dna,
                "iupac" => Alphabet::Iupac,
                _ => panic!("Unsupported edlib alphabet: {}", edlib_alphabet),
            };
            bench::benchmark_edlib(
                &guides,
                &chromosome,
                k,
                total_bytes,
                config.warmup_iterations,
                config.iterations,
                &alphabet,
                "Benchmarking",
                false,
            )
        } else {
            bench::BenchmarkResults::empty()
        };

        csv.write_row(
            guides.len(),
            chromosome.len(),
            query_len,
            k,
            &search_results,
            &tiling_results,
            &edlib_results,
            total_bytes,
        )
        .unwrap();

        println!(
            "  Results: search={:.2}ms [{:.2}, {:.2}] ({:.3} GB/s), tiling={:.2}ms ({:.3} GB/s), edlib={:.2}ms [{:.2}, {:.2}] ({:.3} GB/s) | n_matches: search={}, tiling={}, edlib={}",
            search_results.median,
            search_results.ci_lower,
            search_results.ci_upper,
            search_results.throughput_gbps,
            tiling_results.median,
            tiling_results.throughput_gbps,
            edlib_results.median,
            edlib_results.ci_lower,
            edlib_results.ci_upper,
            edlib_results.throughput_gbps,
            search_results.n_matches,
            tiling_results.n_matches,
            edlib_results.n_matches
        );
    }

    println!(
        "\nOff-target benchmark complete. Results written to {}",
        config.output_file
    );
}
