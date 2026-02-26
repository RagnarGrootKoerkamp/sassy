use rand::RngExt;
use serde::Deserialize;
use std::fs;

use crate::benchsuite::sim_data::Alphabet;
use crate::sassy2::bench;

#[derive(Deserialize)]
struct Config {
    target_lens: Vec<usize>,
    query_lens: Vec<usize>,
    /// K values: integers (e.g. 3, 20) or fraction of pattern length (e.g. 0.05 = 5%, rounded up).
    ks: Vec<f64>,
    min_benchtime: f64,
    warmup_iterations: usize,
    n_queries: usize,
    output_file: String,
    run_edlib: bool,
    edlib_alphabet: String,
    #[serde(default)]
    tools: Option<Vec<bench::BenchTool>>,
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
            for &k_config in &config.ks {
                let k = bench::resolve_k(k_config, query_len);
                if k >= query_len {
                    continue;
                }
                // println!(
                //     "Testing q_len={}, t_len={}, k={} (config k={})",
                //     query_len, target_len, k, k_config
                // );

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
                    config.tools.as_deref().unwrap_or(&bench::DEFAULT_TOOLS),
                    false,
                );

                println!("{}", &suite);

                csv.write_row(
                    config.n_queries,
                    target_len,
                    query_len,
                    &format!("{}", k_config),
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
