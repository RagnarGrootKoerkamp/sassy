use crate::benchsuite::sim_data::Alphabet;
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
    /// K values: integers (e.g. 3) or fraction of pattern length (e.g. 0.05 = 5%, rounded up).
    ks: Vec<f64>,
    threads: usize,
    #[serde(default)]
    tools: Option<Vec<bench::BenchTool>>,
}

fn load_guide_sequences(path: &str) -> Vec<Vec<u8>> {
    let mut reader = parse_fastx_file(path).unwrap();
    let mut sequences = Vec::new();
    while let Some(record) = reader.next() {
        sequences.push(record.unwrap().seq().to_ascii_uppercase());
    }
    sequences
}

fn load_chromosomes(path: &str) -> Vec<Vec<u8>> {
    let mut reader = parse_fastx_file(path).unwrap();
    let mut chromosomes = Vec::new();
    while let Some(record) = reader.next() {
        chromosomes.push(record.unwrap().seq().to_ascii_uppercase());
    }
    chromosomes
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
    println!("Genome: {}", config.genome_file);
    println!("K values: {:?}", config.ks);
    println!(
        "Warmup: {}, min_benchtime: {} s",
        config.warmup_iterations, config.min_benchtime
    );
    println!("Output: {}", config.output_file);
    println!();

    let guides = load_guide_sequences(&config.guide_file);
    let chromosomes = load_chromosomes(&config.genome_file);

    let total_text_len: usize = chromosomes.iter().map(|c| c.len()).sum();
    let total_bytes = total_text_len;

    println!(
        "Loaded {} guides, {} chromosomes (total {} bp)",
        guides.len(),
        chromosomes.len(),
        total_text_len
    );

    if guides.is_empty() {
        panic!("No guide sequences loaded from {}", config.guide_file);
    }
    if chromosomes.is_empty() {
        panic!("No chromosomes loaded from {}", config.genome_file);
    }
    let same_length = guides_same_length(&guides);
    if !same_length {
        println!("  Note: guides have mixed lengths; tiling benchmark skipped (zeros written).");
    }

    let query_len = guides.first().map(|g| g.len()).unwrap_or(0);
    let mut csv = bench::BenchCsv::new(&config.output_file).unwrap();

    for &k_config in &config.ks {
        let k = bench::resolve_k(k_config, query_len);

        println!("Benchmarking k={} (resolved from config {})", k, k_config);

        let suite = bench::benchmark_tools(
            &guides,
            &chromosomes,
            k,
            config.warmup_iterations,
            config.min_benchtime,
            config.threads,
            &Alphabet::Iupac,
            false,
            config.tools.as_deref().unwrap_or(&bench::DEFAULT_TOOLS),
            false,
        );

        println!("{}", &suite);

        csv.write_row(
            guides.len(),
            total_text_len,
            query_len,
            &format!("{}", k_config),
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
