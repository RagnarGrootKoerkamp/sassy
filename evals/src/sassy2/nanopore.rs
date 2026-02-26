use crate::benchsuite::bench;
use crate::benchsuite::sim_data::Alphabet;
use needletail::parse_fastx_file;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct Config {
    read_file: String,
    /// K values: integers (e.g. 3) or fraction of pattern length (e.g. 0.05 = 5%, rounded up).
    ks: Vec<f64>,
    min_benchtime: f64,
    warmup_iterations: usize,
    output_file: String,
    barcode_file: String,
    /// Number of threads for parallel many-texts benchmarks (1 = sequential).
    #[serde(default = "default_threads")]
    threads: usize,
    #[serde(default)]
    tools: Option<Vec<bench::BenchTool>>,
}

fn default_threads() -> usize {
    1
}

fn read_barcodes(barcode_file: &str) -> Vec<Vec<u8>> {
    let mut reader = parse_fastx_file(barcode_file).unwrap();
    let mut barcodes = Vec::new();
    while let Some(record) = reader.next() {
        barcodes.push(record.unwrap().seq().to_ascii_uppercase());
    }
    barcodes
}

fn load_reads(read_file: &str) -> Vec<Vec<u8>> {
    let mut reader = parse_fastx_file(read_file).unwrap();
    let mut reads = Vec::new();
    while let Some(record) = reader.next() {
        reads.push(record.unwrap().seq().to_ascii_uppercase());
    }
    reads
}

pub fn run(config_path: &str) {
    let toml_str = fs::read_to_string(config_path).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    println!("Running nanopore benchmark");
    println!("Config: {:?}", config_path);
    println!("Reads: {}", config.read_file);
    println!("Barcodes: {}", config.barcode_file);
    println!("Output: {}", config.output_file);
    println!("K values: {:?}", config.ks);
    println!(
        "Warmup: {}, min_benchtime: {} s, threads: {}",
        config.warmup_iterations, config.min_benchtime, config.threads
    );
    println!();

    let barcodes = read_barcodes(&config.barcode_file);
    let reads = load_reads(&config.read_file);

    let total_text_len: usize = reads.iter().map(|r| r.len()).sum();
    let total_bytes = total_text_len;
    let query_len: usize = barcodes.first().map(|b| b.len()).unwrap_or(0);

    println!(
        "Loaded {} reads (total {} bp), {} barcodes",
        reads.len(),
        total_text_len,
        barcodes.len()
    );

    if reads.is_empty() {
        panic!("No reads loaded from {}", config.read_file);
    }
    if barcodes.is_empty() {
        panic!("No barcodes loaded from {}", config.barcode_file);
    }

    let mut csv = bench::BenchCsv::new(&config.output_file).unwrap();

    for &k_config in &config.ks {
        let k = bench::resolve_k(k_config, query_len);

        println!("Benchmarking k={} (resolved from config {})", k, k_config);

        let suite = bench::benchmark_tools(
            &barcodes,
            &reads,
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
            barcodes.len(),
            total_text_len,
            query_len,
            &format!("{}", k_config),
            &suite,
            total_bytes,
        )
        .unwrap();
    }

    println!(
        "\nNanopore benchmark complete. Results written to {}",
        config.output_file
    );
}
