use rand::Rng;
use serde::Deserialize;
use std::fs;

use crate::sassy1::edlib_bench::sim_data::Alphabet;
use crate::sassy2::bench;

#[derive(Deserialize)]
struct Config {
    target_lens: Vec<usize>,
    query_lens: Vec<usize>,
    ks: Vec<usize>,
    iterations: usize,
    warmup_iterations: usize,
    n_queries: usize,
    output_file: String,
    run_edlib: bool,
    edlib_alphabet: String,
}

pub fn run(config_path: &str) {
    let toml_str = fs::read_to_string(config_path).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    println!("Running pattern scaling benchmark");
    println!("Config: {:?}", config_path);
    println!("Output: {}", config.output_file);
    println!("Target lengths: {:?}", config.target_lens);
    println!("Query lengths: {:?}", config.query_lens);
    println!("K values: {:?}", config.ks);
    println!("Queries per benchmark: {}", config.n_queries);
    println!("Warmup iterations: {}", config.warmup_iterations);
    println!("Measurement iterations: {}", config.iterations);
    println!();

    let mut csv = bench::BenchCsv::new(&config.output_file).unwrap();

    let mut rng = rand::rng();

    // Test all combinations
    for &query_len in &config.query_lens {
        for &target_len in &config.target_lens {
            for &k in &config.ks {
                if k >= query_len {
                    continue;
                }

                println!("Testing q_len={}, t_len={}, k={}", query_len, target_len, k);

                // Generate text
                let text: Vec<u8> = (0..target_len)
                    .map(|_| b"ACGT"[rng.random_range(0..4)])
                    .collect();

                // Generate queries
                let queries: Vec<Vec<u8>> = (0..config.n_queries)
                    .map(|_| {
                        (0..query_len)
                            .map(|_| b"ACGT"[rng.random_range(0..4)])
                            .collect()
                    })
                    .collect();

                let total_bytes = target_len * config.n_queries;

                // Run benchmarks
                let search_results = bench::benchmark_individual_search(
                    &queries,
                    &text,
                    k,
                    total_bytes,
                    config.warmup_iterations,
                    config.iterations,
                    "Benchmarking",
                    false,
                );

                let patterns_results = bench::benchmark_patterns(
                    &queries,
                    &text,
                    k,
                    total_bytes,
                    config.warmup_iterations,
                    config.iterations,
                    "Benchmarking",
                    false,
                );

                let tiling_results = bench::benchmark_tiling(
                    &queries,
                    &text,
                    k,
                    total_bytes,
                    config.warmup_iterations,
                    config.iterations,
                    "Benchmarking",
                    false,
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
                    config.n_queries,
                    target_len,
                    query_len,
                    k,
                    &search_results,
                    &tiling_results,
                    &edlib_results,
                    total_bytes,
                )
                .unwrap();

                println!(
                    "  Results: search={:.2}ms [{:.2}, {:.2}] ({:.3}GB/s), patterns={:.2}ms [{:.2}, {:.2}] ({:.3}GB/s), tiling={:.2}ms [{:.2}, {:.2}] ({:.3}GB/s) | n_matches: search={}, patterns={}, tiling={}",
                    search_results.median,
                    search_results.ci_lower,
                    search_results.ci_upper,
                    search_results.throughput_gbps,
                    patterns_results.median,
                    patterns_results.ci_lower,
                    patterns_results.ci_upper,
                    patterns_results.throughput_gbps,
                    tiling_results.median,
                    tiling_results.ci_lower,
                    tiling_results.ci_upper,
                    tiling_results.throughput_gbps,
                    search_results.n_matches,
                    patterns_results.n_matches,
                    tiling_results.n_matches
                );
            }
        }
    }

    println!(
        "\nPattern scaling benchmark complete. Results written to {}",
        config.output_file
    );
}
