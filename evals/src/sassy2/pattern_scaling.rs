use rand::RngExt;
use serde::Deserialize;
use std::fs;

use crate::sassy1::edlib_bench::sim_data::Alphabet;
use crate::sassy2::bench;

#[derive(Deserialize)]
struct Config {
    target_lens: Vec<usize>,
    query_lens: Vec<usize>,
    ks: Vec<usize>,
    min_benchtime: f64,
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
    println!("Min bench time: {} s", config.min_benchtime);
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

                let suite = bench::benchmark_tools(
                    &queries,
                    &[text],
                    k,
                    config.warmup_iterations,
                    config.min_benchtime,
                    1,
                    &Alphabet::Iupac,
                    true,
                );

                println!("{}", &suite);

                csv.write_row(
                    config.n_queries,
                    target_len,
                    query_len,
                    k,
                    &suite,
                    total_bytes,
                )
                .unwrap();
            }
        }
    }

    println!(
        "\nPattern scaling benchmark complete. Results written to {}",
        config.output_file
    );
}
