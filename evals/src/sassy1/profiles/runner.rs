use rand::Rng;
use sassy::Searcher;
use serde::Deserialize;
use std::fs;
use std::time::Instant;

#[derive(Deserialize)]
struct Config {
    q_lens: Vec<usize>,
    t_lens: Vec<usize>,
    k: usize,
    // alpha: f32,
    iterations: Option<usize>,
}

fn generate_random_dna_sequence(len: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    (0..len)
        .map(|_| {
            let n = rng.random_range(0..4);
            match n {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            }
        })
        .collect()
}

fn benchmark_profile<P>(query: &[u8], text: &[u8], k: usize, iterations: usize) -> f64
where
    P: sassy::profiles::Profile,
{
    // No overhang, no rc?
    let mut searcher = Searcher::<P>::new_fwd();

    // Warm up
    for _ in 0..10 {
        let _matches = searcher.search(query, &text, k);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _matches = searcher.search(query, &text, k);
    }
    let duration = start.elapsed();

    // Calculate throughput in GB/s
    let total_bytes_processed = text.len() * iterations;
    let duration_secs = duration.as_secs_f64();
    let throughput_gbps = (total_bytes_processed as f64) / (duration_secs * 1_000_000_000.0);

    throughput_gbps
}

pub fn run(config: &str) {
    let toml_str = fs::read_to_string(config).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();
    let iterations = config.iterations.unwrap_or(500);

    println!("Benchmarking with {} iterations per test", iterations);
    println!("k: {}", config.k);
    println!();

    // Print header
    println!(
        "{:<8} {:<8} {:<12} {:<12} {:<12} {:<12}",
        "Q_Len", "T_Len", "DNA_GB/s", "IUPAC_GB/s", "ASCII_GB/s", "Text_GB"
    );

    for (q_len, t_len) in config.q_lens.iter().zip(config.t_lens.iter()) {
        let query = generate_random_dna_sequence(*q_len);
        let text = generate_random_dna_sequence(*t_len);

        // Benchmark each profile type
        let dna_throughput =
            benchmark_profile::<sassy::profiles::Dna>(&query, &text, config.k, iterations);

        let iupac_throughput =
            benchmark_profile::<sassy::profiles::Iupac>(&query, &text, config.k, iterations);

        let ascii_throughput =
            benchmark_profile::<sassy::profiles::Ascii>(&query, &text, config.k, iterations);

        let text_gb = text.len() as f64 / 1_000_000_000.0;

        println!(
            "{:<8} {:<8} {:<12.3} {:<12.3} {:<12.3} {:<12.6}",
            q_len, t_len, dna_throughput, iupac_throughput, ascii_throughput, text_gb
        );
    }
}
