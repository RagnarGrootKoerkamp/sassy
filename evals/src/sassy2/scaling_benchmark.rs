use rand::Rng;
use sassy::Searcher;
use sassy::profiles::Iupac;
use serde::Deserialize;
use std::fs;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use crate::sassy1::edlib_bench::edlib::{get_edlib_config, run_edlib};
use crate::sassy1::edlib_bench::sim_data::Alphabet;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use perfcnt::AbstractPerfCounter;

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

struct BenchmarkResults {
    median: f64,
    mean: f64,
    std_dev: f64,
    ci_lower: f64,
    ci_upper: f64,
    throughput_gbps: f64,
    ci_lower_throughput_gbps: f64,
    ci_upper_throughput_gbps: f64,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn measure_ipc<F: FnOnce()>(f: F) -> f64 {
    let mut instr = perfcnt::linux::PerfCounterBuilderLinux::from_hardware_event(
        perfcnt::linux::HardwareEventType::Instructions,
    )
    .finish()
    .expect("Could not create instruction counter");

    let mut cycles = perfcnt::linux::PerfCounterBuilderLinux::from_hardware_event(
        perfcnt::linux::HardwareEventType::CPUCycles,
    )
    .finish()
    .expect("Could not create cycle counter");

    instr.start().unwrap();
    cycles.start().unwrap();
    f();
    instr.stop().unwrap();
    cycles.stop().unwrap();

    let instructions = instr.read().unwrap();
    let cycles = cycles.read().unwrap();

    if cycles == 0 {
        0.0
    } else {
        instructions as f64 / cycles as f64
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn measure_ipc<F: FnOnce()>(f: F) -> f64 {
    // Perf counters aren't available/portable here (e.g. macOS / Apple Silicon).
    // Execute the workload once, then return a sentinel IPC value.
    f();
    0.0
}

fn benchmark_with_stats<F>(mut f: F, warmup: usize, iterations: usize) -> (f64, f64, f64, f64, f64)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..warmup {
        f();
    }

    // Collect timings
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_nanos() as f64 / 1_000_000.0);
    }

    // Calculate statistics
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std_dev = variance.sqrt();

    // Calculate 95% confidence interval using normal approximation
    // CI = mean ± 1.96 * (std_dev / sqrt(n))
    let n = iterations as f64;
    let standard_error = std_dev / n.sqrt();
    let z_score = 1.96; // 95% confidence interval
    let margin = z_score * standard_error;
    let ci_lower = mean - margin;
    let ci_upper = mean + margin;

    (median, mean, std_dev, ci_lower, ci_upper)
}

fn calculate_throughput(total_bytes: usize, median_ms: f64) -> f64 {
    (total_bytes as f64 / (median_ms / 1000.0)) / 1_000_000_000.0
}

fn benchmark_individual_search(
    queries: &[Vec<u8>],
    text: &[u8],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
) -> BenchmarkResults {
    println!("  Benchmarking sassy search...");

    let mut searcher = Searcher::<Iupac>::new_fwd();

    let (median, mean, std_dev, ci_lower, ci_upper) = benchmark_with_stats(
        || {
            for query in queries {
                black_box(query);
                let _matches = searcher.search(query, text, k);
                black_box(_matches);
            }
        },
        warmup,
        iterations,
    );

    let throughput_gbps = calculate_throughput(total_bytes, median);
    // Calculate throughput CI bounds from time CI bounds
    // Note: higher time = lower throughput, so ci_upper gives lower throughput bound
    let ci_lower_throughput_gbps = calculate_throughput(total_bytes, ci_upper);
    let ci_upper_throughput_gbps = calculate_throughput(total_bytes, ci_lower);

    BenchmarkResults {
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        throughput_gbps,
        ci_lower_throughput_gbps,
        ci_upper_throughput_gbps,
    }
}

fn benchmark_patterns(
    queries: &[Vec<u8>],
    text: &[u8],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
) -> BenchmarkResults {
    println!("  Benchmarking sassy patterns...");

    let mut searcher = Searcher::<Iupac>::new_fwd();

    let (median, mean, std_dev, ci_lower, ci_upper) = benchmark_with_stats(
        || {
            let _matches = searcher.search_patterns(queries, text, k);
            black_box(_matches);
        },
        warmup,
        iterations,
    );

    let throughput_gbps = calculate_throughput(total_bytes, median);
    // Calculate throughput CI bounds from time CI bounds
    // Note: higher time = lower throughput, so ci_upper gives lower throughput bound
    let ci_lower_throughput_gbps = calculate_throughput(total_bytes, ci_upper);
    let ci_upper_throughput_gbps = calculate_throughput(total_bytes, ci_lower);

    BenchmarkResults {
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        throughput_gbps,
        ci_lower_throughput_gbps,
        ci_upper_throughput_gbps,
    }
}

fn benchmark_tiling(
    queries: &[Vec<u8>],
    text: &[u8],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
) -> BenchmarkResults {
    println!("  Benchmarking sassy tiling...");

    let mut searcher = Searcher::<Iupac>::new_fwd();
    let encoded = searcher.encode_patterns(queries);

    let (median, mean, std_dev, ci_lower, ci_upper) = benchmark_with_stats(
        || {
            let _matches = searcher.search_encoded_patterns(&encoded, text, k);
            black_box(_matches);
        },
        warmup,
        iterations,
    );

    let throughput_gbps = calculate_throughput(total_bytes, median);
    // Calculate throughput CI bounds from time CI bounds
    // Note: higher time = lower throughput, so ci_upper gives lower throughput bound
    let ci_lower_throughput_gbps = calculate_throughput(total_bytes, ci_upper);
    let ci_upper_throughput_gbps = calculate_throughput(total_bytes, ci_lower);

    BenchmarkResults {
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        throughput_gbps,
        ci_lower_throughput_gbps,
        ci_upper_throughput_gbps,
    }
}

fn benchmark_edlib(
    queries: &[Vec<u8>],
    text: &[u8],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    alphabet: &Alphabet,
) -> BenchmarkResults {
    println!("  Benchmarking edlib...");

    let edlib_config = get_edlib_config(k as i32, alphabet);

    let (median, mean, std_dev, ci_lower, ci_upper) = benchmark_with_stats(
        || {
            for query in queries {
                black_box(query);
                let _result = run_edlib(query, text, &edlib_config);
                black_box(_result);
            }
        },
        warmup,
        iterations,
    );

    let throughput_gbps = calculate_throughput(total_bytes, median);
    // Calculate throughput CI bounds from time CI bounds
    // Note: higher time = lower throughput, so ci_upper gives lower throughput bound
    let ci_lower_throughput_gbps = calculate_throughput(total_bytes, ci_upper);
    let ci_upper_throughput_gbps = calculate_throughput(total_bytes, ci_lower);

    BenchmarkResults {
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        throughput_gbps,
        ci_lower_throughput_gbps,
        ci_upper_throughput_gbps,
    }
}

fn write_csv_header(file: &mut File) {
    writeln!(
        file,
        "query_len,target_len,k,algorithm,median_ms,mean_ms,std_ms,ci_lower_ms,ci_upper_ms,throughput_gbps,ci_lower_throughput_gbps,ci_upper_throughput_gbps,total_bytes"
    )
    .unwrap();
}

fn write_csv_row(
    file: &mut File,
    query_len: usize,
    target_len: usize,
    k: usize,
    algorithm: &str,
    results: &BenchmarkResults,
    total_bytes: usize,
) {
    writeln!(
        file,
        "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
        query_len,
        target_len,
        k,
        algorithm,
        results.median,
        results.mean,
        results.std_dev,
        results.ci_lower,
        results.ci_upper,
        results.throughput_gbps,
        results.ci_lower_throughput_gbps,
        results.ci_upper_throughput_gbps,
        total_bytes
    )
    .unwrap();
}

pub fn run(config_path: &str) {
    let toml_str = fs::read_to_string(config_path).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    println!("Running scaling throughput benchmark");
    println!("Config: {:?}", config_path);
    println!("Output: {}", config.output_file);
    println!("Target lengths: {:?}", config.target_lens);
    println!("Query lengths: {:?}", config.query_lens);
    println!("K values: {:?}", config.ks);
    println!("Queries per benchmark: {}", config.n_queries);
    println!("Warmup iterations: {}", config.warmup_iterations);
    println!("Measurement iterations: {}", config.iterations);
    println!();

    let mut file = File::create(&config.output_file).unwrap();
    write_csv_header(&mut file);

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
                let search_results = benchmark_individual_search(
                    &queries,
                    &text,
                    k,
                    total_bytes,
                    config.warmup_iterations,
                    config.iterations,
                );

                let patterns_results = benchmark_patterns(
                    &queries,
                    &text,
                    k,
                    total_bytes,
                    config.warmup_iterations,
                    config.iterations,
                );

                let tiling_results = benchmark_tiling(
                    &queries,
                    &text,
                    k,
                    total_bytes,
                    config.warmup_iterations,
                    config.iterations,
                );

                // Write sassy results
                write_csv_row(
                    &mut file,
                    query_len,
                    target_len,
                    k,
                    "sassy_search",
                    &search_results,
                    total_bytes,
                );

                write_csv_row(
                    &mut file,
                    query_len,
                    target_len,
                    k,
                    "sassy_patterns",
                    &patterns_results,
                    total_bytes,
                );

                write_csv_row(
                    &mut file,
                    query_len,
                    target_len,
                    k,
                    "sassy_tiling",
                    &tiling_results,
                    total_bytes,
                );

                // Run and write edlib results if enabled
                if config.run_edlib {
                    let alphabet = match config.edlib_alphabet.as_str() {
                        "dna" => Alphabet::Dna,
                        "iupac" => Alphabet::Iupac,
                        _ => panic!("Unsupported edlib alphabet: {}", config.edlib_alphabet),
                    };

                    let edlib_results = benchmark_edlib(
                        &queries,
                        &text,
                        k,
                        total_bytes,
                        config.warmup_iterations,
                        config.iterations,
                        &alphabet,
                    );

                    write_csv_row(
                        &mut file,
                        query_len,
                        target_len,
                        k,
                        "edlib",
                        &edlib_results,
                        total_bytes,
                    );
                }

                println!(
                    "  Results: search={:.2}ms [{:.2}, {:.2}] ({:.3}GB/s), patterns={:.2}ms [{:.2}, {:.2}] ({:.3}GB/s), tiling={:.2}ms [{:.2}, {:.2}] ({:.3}GB/s)",
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
                    tiling_results.throughput_gbps
                );
            }
        }
    }

    println!(
        "\nScaling benchmark complete. Results written to {}",
        config.output_file
    );
}
