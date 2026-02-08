use crate::sassy1::edlib_bench::sim_data::Alphabet;
use crate::sassy2::bench;
use needletail::parse_fastx_file;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct Config {
    read_file: String,
    ks: Vec<usize>,
    iterations: usize,
    warmup_iterations: usize,
    output_file: String,
    run_edlib: bool,
    edlib_alphabet: String,
    barcode_file: String,
    /// Number of threads for parallel many-texts benchmarks (1 = sequential).
    #[serde(default = "default_threads")]
    threads: usize,
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

    let run_edlib = config.run_edlib;
    let edlib_alphabet = config.edlib_alphabet.as_str();

    println!("Running nanopore benchmark");
    println!("Config: {:?}", config_path);
    println!("Reads: {}", config.read_file);
    println!("Barcodes: {}", config.barcode_file);
    println!("Output: {}", config.output_file);
    println!("K values: {:?}", config.ks);
    println!(
        "Warmup: {}, iterations: {}, threads: {}",
        config.warmup_iterations, config.iterations, config.threads
    );
    println!();

    let barcodes = read_barcodes(&config.barcode_file);
    let reads = load_reads(&config.read_file);

    let total_text_len: usize = reads.iter().map(|r| r.len()).sum();
    let total_bytes = reads
        .iter()
        .map(|r| r.len() * barcodes.len())
        .sum::<usize>();
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

    for &k in &config.ks {
        println!("Benchmarking k={}", k);

        let search_results = bench::benchmark_individual_search_many_texts(
            &barcodes,
            &reads,
            k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
            config.threads,
            "nanopore",
            false,
        );

        let tiling_results = bench::benchmark_tiling_many_texts(
            &barcodes,
            &reads,
            k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
            config.threads,
            "nanopore",
            false,
        );

        let edlib_results = if run_edlib {
            let alphabet = match edlib_alphabet {
                "dna" => Alphabet::Dna,
                "iupac" => Alphabet::Iupac,
                _ => panic!("Unsupported edlib alphabet: {}", edlib_alphabet),
            };
            bench::benchmark_edlib_many_texts(
                &barcodes,
                &reads,
                k,
                total_bytes,
                config.warmup_iterations,
                config.iterations,
                config.threads,
                &alphabet,
                "nanopore",
                false,
            )
        } else {
            bench::BenchmarkResults::empty()
        };

        csv.write_row(
            barcodes.len(),
            total_text_len,
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
        "\nNanopore benchmark complete. Results written to {}",
        config.output_file
    );
}
