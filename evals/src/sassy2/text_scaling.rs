use rand::Rng;
use serde::Deserialize;
use std::fs;

use crate::sassy1::edlib_bench::sim_data::Alphabet;
use crate::sassy2::bench;

#[derive(Deserialize)]
struct Config {
    query_len: usize,
    text_len: usize,
    k: usize,
    num_queries_list: Vec<usize>,
    iterations: usize,
    warmup_iterations: usize,
    output_file: String,
    run_edlib: bool,
    edlib_alphabet: String,
}

pub fn run(config_path: &str) {
    let toml_str = fs::read_to_string(config_path).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    println!("Running text scaling benchmark");
    println!("Config: {:?}", config_path);
    println!("Output: {}", config.output_file);
    println!("Warmup iterations: {}", config.warmup_iterations);
    println!("Measurement iterations: {}", config.iterations);
    println!();

    // Generate text
    let mut rng = rand::rng();
    let text: Vec<u8> = (0..config.text_len)
        .map(|_| b"ACGT"[rng.random_range(0..4)])
        .collect();

    let mut csv = bench::BenchCsv::new(&config.output_file).unwrap();

    for &num_queries in &config.num_queries_list {
        println!("Benchmarking q={}", num_queries);

        // Generate queries
        let queries: Vec<Vec<u8>> = (0..num_queries)
            .map(|_| {
                (0..config.query_len)
                    .map(|_| b"ACGT"[rng.random_range(0..4)])
                    .collect()
            })
            .collect();

        let total_bytes = config.text_len * num_queries;

        // Run benchmarks (with IPC measurement for text_scaling)
        let search_results = bench::benchmark_individual_search(
            &queries,
            &text,
            config.k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
            "Warming up",
            true,
        );

        let tiling_results = bench::benchmark_tiling(
            &queries,
            &text,
            config.k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
            "Warming up",
            true,
        );

        let edlib_results = if config.run_edlib {
            let alphabet = match config.edlib_alphabet.as_str() {
                "dna" => Alphabet::Dna,
                "iupac" => Alphabet::Iupac,
                _ => panic!("Unsupported edlib alphabet: {}", config.edlib_alphabet),
            };

            bench::benchmark_edlib(
                &queries,
                &text,
                config.k,
                total_bytes,
                config.warmup_iterations,
                config.iterations,
                &alphabet,
                "Warming up",
                true,
            )
        } else {
            bench::BenchmarkResults::empty()
        };

        csv.write_row(
            num_queries,
            config.text_len,
            config.query_len,
            config.k,
            &search_results,
            &tiling_results,
            &edlib_results,
            total_bytes,
        )
        .unwrap();

        println!(
            "  Results: search={:.2}ms [{:.2}, {:.2}], tiling={:.2}ms [{:.2}, {:.2}] | n_matches: search={}, tiling={}, edlib={}",
            search_results.median,
            search_results.ci_lower,
            search_results.ci_upper,
            tiling_results.median,
            tiling_results.ci_lower,
            tiling_results.ci_upper,
            search_results.n_matches,
            tiling_results.n_matches,
            edlib_results.n_matches
        );
    }

    println!(
        "\nBenchmark complete. Results written to {}",
        config.output_file
    );
}
