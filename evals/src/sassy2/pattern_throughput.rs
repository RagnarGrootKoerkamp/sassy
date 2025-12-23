use perfcnt::AbstractPerfCounter;
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

struct BenchmarkResults {
    median: f64,
    mean: f64,
    std_dev: f64,
    ipc: f64,
    throughput_gbps: f64,
}

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

fn benchmark_with_stats<F>(mut f: F, warmup: usize, iterations: usize) -> (f64, f64, f64)
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

    (median, mean, std_dev)
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
    println!("  Warming up search...");

    // Declare searcher outside timing loop
    let mut searcher = Searcher::<Iupac>::new_fwd();

    let (median, mean, std_dev) = benchmark_with_stats(
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

    // IPC
    let ipc = measure_ipc(|| {
        for query in queries {
            black_box(query);
            let _matches = searcher.search(query, text, k);
            black_box(_matches);
        }
    });

    BenchmarkResults {
        median,
        mean,
        std_dev,
        ipc,
        throughput_gbps,
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
    println!("  Warming up patterns...");

    // Declare searcher outside timing loop
    let mut searcher = Searcher::<Iupac>::new_fwd();

    let (median, mean, std_dev) = benchmark_with_stats(
        || {
            let _matches = searcher.search_patterns(queries, text, k);
            black_box(_matches);
        },
        warmup,
        iterations,
    );

    let throughput_gbps = calculate_throughput(total_bytes, median);

    // IPC
    let ipc = measure_ipc(|| {
        let _matches = searcher.search_patterns(queries, text, k);
        black_box(_matches);
    });

    BenchmarkResults {
        median,
        mean,
        std_dev,
        ipc,
        throughput_gbps,
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
    println!("  Warming up tiling...");

    // Declare searcher and encode patterns outside timing loop
    let mut searcher = Searcher::<Iupac>::new_fwd();
    let encoded = searcher.encode_patterns(queries);

    let (median, mean, std_dev) = benchmark_with_stats(
        || {
            let _matches = searcher.search_encoded_patterns(&encoded, text, k);
            black_box(_matches);
        },
        warmup,
        iterations,
    );

    let throughput_gbps = calculate_throughput(total_bytes, median);

    // IPC
    let ipc = measure_ipc(|| {
        let _matches = searcher.search_encoded_patterns(&encoded, text, k);
        black_box(_matches);
    });

    BenchmarkResults {
        median,
        mean,
        std_dev,
        ipc,
        throughput_gbps,
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
    println!("  Warming up edlib...");

    // Prepare config outside timing loop
    let edlib_config = get_edlib_config(k as i32, alphabet);

    let (median, mean, std_dev) = benchmark_with_stats(
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

    // IPC
    let ipc = measure_ipc(|| {
        for query in queries {
            black_box(query);
            let _result = run_edlib(query, text, &edlib_config);
            black_box(_result);
        }
    });

    BenchmarkResults {
        median,
        mean,
        std_dev,
        ipc,
        throughput_gbps,
    }
}

fn write_csv_header(file: &mut File) {
    writeln!(
        file,
        "num_queries,search_median_ms,search_mean_ms,search_std_ms,patterns_median_ms,patterns_mean_ms,patterns_std_ms,tiling_median_ms,tiling_mean_ms,tiling_std_ms,edlib_median_ms,edlib_mean_ms,edlib_std_ms,search_ipc,patterns_ipc,tiling_ipc,edlib_ipc,search_throughput_gbps,patterns_throughput_gbps,tiling_throughput_gbps,edlib_throughput_gbps,throughput_bytes"
    ).unwrap();
}

fn write_csv_row(
    file: &mut File,
    num_queries: usize,
    search: &BenchmarkResults,
    patterns: &BenchmarkResults,
    tiling: &BenchmarkResults,
    edlib: &BenchmarkResults,
    total_bytes: usize,
) {
    writeln!(
        file,
        "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{}",
        num_queries,
        search.median, search.mean, search.std_dev,
        patterns.median, patterns.mean, patterns.std_dev,
        tiling.median, tiling.mean, tiling.std_dev,
        edlib.median, edlib.mean, edlib.std_dev,
        search.ipc, patterns.ipc, tiling.ipc, edlib.ipc,
        search.throughput_gbps, patterns.throughput_gbps, tiling.throughput_gbps, edlib.throughput_gbps,
        total_bytes
    ).unwrap();
}

pub fn run(config_path: &str) {
    let toml_str = fs::read_to_string(config_path).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    println!("Running pattern throughput benchmark");
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

    // Open output file and write header
    let mut file = File::create(&config.output_file).unwrap();
    write_csv_header(&mut file);

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

        // Run benchmarks
        let search_results = benchmark_individual_search(
            &queries,
            &text,
            config.k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
        );

        let patterns_results = benchmark_patterns(
            &queries,
            &text,
            config.k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
        );

        let tiling_results = benchmark_tiling(
            &queries,
            &text,
            config.k,
            total_bytes,
            config.warmup_iterations,
            config.iterations,
        );

        let edlib_results = if config.run_edlib {
            let alphabet = match config.edlib_alphabet.as_str() {
                "dna" => Alphabet::Dna,
                "iupac" => Alphabet::Iupac,
                _ => panic!("Unsupported edlib alphabet: {}", config.edlib_alphabet),
            };

            benchmark_edlib(
                &queries,
                &text,
                config.k,
                total_bytes,
                config.warmup_iterations,
                config.iterations,
                &alphabet,
            )
        } else {
            BenchmarkResults {
                median: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                ipc: 0.0,
                throughput_gbps: 0.0,
            }
        };

        // Write results to CSV
        write_csv_row(
            &mut file,
            num_queries,
            &search_results,
            &patterns_results,
            &tiling_results,
            &edlib_results,
            total_bytes,
        );

        println!(
            "  Results: search={:.2}±{:.2}ms, patterns={:.2}±{:.2}ms, tiling={:.2}±{:.2}ms",
            search_results.median,
            search_results.std_dev,
            patterns_results.median,
            patterns_results.std_dev,
            tiling_results.median,
            tiling_results.std_dev
        );
    }

    println!(
        "\nBenchmark complete. Results written to {}",
        config.output_file
    );
}
