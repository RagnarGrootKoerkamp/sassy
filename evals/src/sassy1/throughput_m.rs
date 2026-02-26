use crate::benchsuite::bench;
use crate::benchsuite::sim_data::Alphabet;
use crate::benchsuite::sim_data::generate_random_sequence;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct Config {
    target_len: usize,
    query_lens: Vec<usize>,
    /// K values: integers (e.g. 3, 20) or fraction of pattern length (e.g. 0.05 = 5%, rounded up).
    ks: Vec<f64>,
    /// Number of query sequences per (query_len, k) mostly important for `m` benches
    #[serde(default)]
    n_queries: Option<usize>,
    /// Number of reference/text sequences per (query_len, k) mostly important for `n` benches
    #[serde(default)]
    n_refs: Option<usize>,
    min_benchtime: f64,
    warmup_iterations: usize,
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
    println!("Target length: {}", config.target_len);
    println!("Query lengths: {:?}", config.query_lens);
    println!("K values: {:?}", config.ks);
    let n_queries = config.n_queries.unwrap_or(1);
    let n_refs = config.n_refs.unwrap_or(1);
    println!("n_queries: {}, n_refs: {}", n_queries, n_refs);
    println!("Warmup iterations: {}", config.warmup_iterations);
    println!("Min bench time: {} s", config.min_benchtime);
    println!();

    let mut csv = bench::BenchCsv::new(&config.output_file).unwrap();
    let target_len = config.target_len;

    for &query_len in &config.query_lens {
        for &k_config in &config.ks {
            let k = bench::resolve_k(k_config, query_len);
            if k >= query_len {
                continue;
            }

            if query_len <= 3 * k || query_len > target_len {
                continue;
            }

            println!(
                "Testing q_len={}, t_len={}, k={} (config k={})",
                query_len, target_len, k, k_config
            );

            // Generate n_queries queries and n_refs texts for more stable throughput
            // if we bench with `m` on x-axis we increase n_queries, if we bench with `n` on x-axis we increase n_refs
            // cause different targets won't really affect the result but choosing, by chance, a matching or similar
            // pattern will (or vice versa)
            let queries: Vec<Vec<u8>> = (0..n_queries)
                .map(|_| generate_random_sequence(query_len, &Alphabet::Dna, None))
                .collect();
            let texts: Vec<Vec<u8>> = (0..n_refs)
                .map(|_| generate_random_sequence(target_len, &Alphabet::Dna, None))
                .collect();

            let text_refs: Vec<&[u8]> = texts.iter().map(Vec::as_ref).collect();

            let total_bytes = target_len * n_refs * n_queries;

            let suite = bench::benchmark_tools(
                &queries,
                &text_refs,
                k,
                config.warmup_iterations,
                config.min_benchtime,
                1,
                &Alphabet::Iupac,
                false,
                config.tools.as_deref().unwrap_or(&bench::DEFAULT_TOOLS),
                false,
            );

            println!("{}", &suite);

            csv.write_row(
                n_queries,
                target_len,
                query_len,
                &format!("{}", k_config),
                &suite,
                total_bytes,
            )
            .unwrap();
        }
    }

    println!(
        "\nPattern scaling benchmark complete. Results written to {}",
        config.output_file
    );
}
