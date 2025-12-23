use rand::Rng;
use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::process::Command;
use std::time::Instant;
use tempfile::NamedTempFile;

#[derive(Deserialize)]
struct Config {
    q_lens: Vec<usize>,
    t_lens: Vec<usize>,
    k: usize,
    iterations: Option<usize>,
    sassy_path: String,
}

fn generate_random_ascii_sequence(len: usize) -> Vec<u8> {
    // lets test random DNA
    let mut rng = rand::rng();
    (0..len)
        .map(|_| rng.random_range(0..4)) // DNA characters
        .map(|c| match c {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            3 => b'T',
            _ => unreachable!(),
        })
        .collect()
    // let mut rng = rand::thread_rng();
    // (0..len)
    //     .map(|_| rng.gen_range(32..127)) // Printable ASCII characters
    //     .collect()
}

fn benchmark_sassy_executable(
    query: &[u8],
    text: &[u8],
    k: usize,
    iterations: usize,
    sassy_path: &str,
) -> (f64, usize) {
    // Create a temporary file for the text
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file
        .write_all(text)
        .expect("Failed to write text to temp file");
    let temp_path = temp_file.path();

    // Warm up
    for _ in 0..10 {
        let _output = Command::new(sassy_path)
            .arg("search")
            .arg("--alphabet")
            .arg("ascii")
            .arg(String::from_utf8_lossy(query).to_string())
            .arg(k.to_string())
            .arg(temp_path)
            .output();
    }

    // Benchmark
    let start = Instant::now();
    let mut total_matches = 0;
    for _ in 0..iterations {
        let output = Command::new(sassy_path)
            .arg("search")
            .arg("--alphabet")
            .arg("dna")
            .arg(String::from_utf8_lossy(query).to_string())
            .arg(k.to_string())
            .arg(temp_path)
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                // Count lines in output to get number of matches
                let output_str = String::from_utf8_lossy(&output.stdout);
                total_matches += output_str.lines().count();
            }
        } else {
            eprintln!("sassy failed");
            return (0.0, 0);
        }
    }
    let duration = start.elapsed();

    // Calculate throughput in GB/s
    let total_bytes_processed = text.len() * iterations;
    let duration_secs = duration.as_secs_f64();
    let throughput_gbps = (total_bytes_processed as f64) / (duration_secs * 1_000_000_000.0);

    (throughput_gbps, total_matches)
}

fn benchmark_tre_agrep(query: &[u8], text: &[u8], k: usize, iterations: usize) -> (f64, usize) {
    // Create a temporary file for the text
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file
        .write_all(text)
        .expect("Failed to write text to temp file");
    let temp_path = temp_file.path();

    // Warm up
    for _ in 0..10 {
        let _output = Command::new("tre-agrep")
            .arg("-E")
            .arg(k.to_string())
            .arg("-D")
            .arg("1")
            .arg("-I")
            .arg("1")
            .arg("-S")
            .arg("1")
            .arg("-w")
            .arg(String::from_utf8_lossy(query).to_string())
            .arg(temp_path)
            .output();
    }

    // Benchmark
    let start = Instant::now();
    let mut total_matches = 0;
    for _ in 0..iterations {
        let output = Command::new("tre-agrep")
            .arg("-E")
            .arg(k.to_string())
            .arg("-D")
            .arg("1")
            .arg("-I")
            .arg("1")
            .arg("-S")
            .arg("1")
            .arg("-w")
            .arg(String::from_utf8_lossy(query).to_string())
            .arg(temp_path)
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                // Count lines in output to get number of matches
                let output_str = String::from_utf8_lossy(&output.stdout);
                total_matches += output_str.lines().count();
            }
        } else {
            eprintln!("tre-agrep failed");
            return (0.0, 0);
        }
    }
    let duration = start.elapsed();

    // Calculate throughput in GB/s
    let total_bytes_processed = text.len() * iterations;
    let duration_secs = duration.as_secs_f64();
    let throughput_gbps = (total_bytes_processed as f64) / (duration_secs * 1_000_000_000.0);

    (throughput_gbps, total_matches)
}

pub fn run(config: &str) {
    let toml_str = fs::read_to_string(config).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();
    let iterations = config.iterations.unwrap_or(500);

    println!(
        "Benchmarking sassy executable vs tre-agrep with {} iterations per test",
        iterations
    );
    println!("k: {}", config.k);
    println!("sassy path: {}", config.sassy_path);
    println!();

    // Print header
    println!(
        "{:<8} {:<8} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Q_Len", "T_Len", "Sassy_GB/s", "Agrep_GB/s", "Speedup", "Sassy_Matches", "Agrep_Matches"
    );

    for (q_len, t_len) in config.q_lens.iter().zip(config.t_lens.iter()) {
        let query = generate_random_ascii_sequence(*q_len);
        let mut text = generate_random_ascii_sequence(*t_len);

        // Insert query random position in text
        let mut rng = rand::rng();
        let pos = rng.random_range(0..text.len());
        text.splice(pos..pos, query.clone());

        // Benchmark sassy executable
        let (sassy_throughput, sassy_matches) =
            benchmark_sassy_executable(&query, &text, config.k, iterations, &config.sassy_path);

        // Benchmark tre-agrep
        let (agrep_throughput, agrep_matches) =
            benchmark_tre_agrep(&query, &text, config.k, iterations);

        let speedup = if agrep_throughput > 0.0 {
            sassy_throughput / agrep_throughput
        } else {
            0.0
        };

        println!(
            "{:<8} {:<8} {:<12.3} {:<12.3} {:<12.3}x {:<12} {:<12}",
            q_len, t_len, sassy_throughput, agrep_throughput, speedup, sassy_matches, agrep_matches
        );
    }
}
