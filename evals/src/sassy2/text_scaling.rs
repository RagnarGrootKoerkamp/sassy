use rand::RngExt;
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
    min_benchtime: f64,
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
    println!("Min bench time: {} s", config.min_benchtime);
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

        let suite = bench::benchmark_tools(
            &queries,
            &[text.clone()],
            config.k,
            config.warmup_iterations,
            config.min_benchtime,
            1,
            &Alphabet::Iupac,
            true,
        );

        println!("{}", &suite);

        csv.write_row(
            num_queries,
            config.text_len,
            config.query_len,
            config.k,
            &suite,
            total_bytes,
        )
        .unwrap();
    }

    println!(
        "\nBenchmark complete. Results written to {}",
        config.output_file
    );
}
