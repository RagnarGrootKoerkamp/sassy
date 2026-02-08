use sassy::Searcher;
use sassy::profiles::Iupac;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::sassy1::edlib_bench::edlib::{get_edlib_config, run_edlib};
use crate::sassy1::edlib_bench::sim_data::Alphabet;
use rayon::prelude::*;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use perfcnt::AbstractPerfCounter;

const NO_IPC: f64 = 0.0;
const USE_RC: bool = false; // Only applies to sassy1/2

#[derive(Clone, Debug)]
pub struct BenchmarkResults {
    pub median: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub throughput_gbps: f64,
    pub ci_lower_throughput_gbps: f64,
    pub ci_upper_throughput_gbps: f64,
    pub n_matches: usize,
    /// Instructions per cycle (median); Some(0.0) on platforms without perf counters.
    pub ipc: Option<f64>,
}

impl BenchmarkResults {
    pub fn empty() -> Self {
        Self {
            median: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            throughput_gbps: 0.0,
            ci_lower_throughput_gbps: 0.0,
            ci_upper_throughput_gbps: 0.0,
            n_matches: 0,
            ipc: None,
        }
    }
}

fn benchmark_with_stats<F>(f: &mut F, warmup: usize, iterations: usize) -> (f64, f64, f64, f64, f64)
where
    F: FnMut(),
{
    for _ in 0..warmup {
        f();
    }

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_nanos() as f64 / 1_000_000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std_dev = variance.sqrt();

    let n = iterations as f64;
    let standard_error = std_dev / n.sqrt();
    let z_score = 1.96;
    let margin = z_score * standard_error;
    let ci_lower = mean - margin;
    let ci_upper = mean + margin;

    (median, mean, std_dev, ci_lower, ci_upper)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn measure_ipc_median<F: FnMut()>(f: &mut F, iterations: usize) -> f64 {
    let mut ipc_values = Vec::with_capacity(iterations);

    for _ in 0..iterations {
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
            ipc_values.push(0.0);
        } else {
            ipc_values.push(instructions as f64 / cycles as f64);
        }
    }

    ipc_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ipc_values[ipc_values.len() / 2]
}

// If it's not linux we don't measure IPC and just return 0.0
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn measure_ipc_median<F: FnMut()>(f: &mut F, _iterations: usize) -> f64 {
    f();
    NO_IPC
}

fn calculate_throughput(total_bytes: usize, median_ms: f64) -> f64 {
    (total_bytes as f64 / (median_ms / 1000.0)) / 1_000_000_000.0
}

fn finish_results(
    median: f64,
    mean: f64,
    std_dev: f64,
    ci_lower: f64,
    ci_upper: f64,
    total_bytes: usize,
    n_matches: usize,
    ipc: Option<f64>,
) -> BenchmarkResults {
    let throughput_gbps = calculate_throughput(total_bytes, median);
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
        n_matches,
        ipc,
    }
}

pub fn benchmark_individual_search(
    queries: &[Vec<u8>],
    text: &[u8],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    log_label: &str,
    measure_ipc: bool,
) -> BenchmarkResults {
    println!("  {} sassy search...", log_label);

    let mut searcher = if USE_RC {
        Searcher::<Iupac>::new_rc()
    } else {
        Searcher::<Iupac>::new_fwd()
    };

    let mut do_work = || {
        for query in queries {
            black_box(query);
            let _matches = searcher.search(query, text, k);
            black_box(_matches);
        }
    };

    let (median, mean, std_dev, ci_lower, ci_upper) =
        benchmark_with_stats(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = {
        let mut total = 0usize;
        for query in queries {
            let matches = searcher.search(query, text, k);
            total += matches.len();
        }
        total
    };

    finish_results(
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        total_bytes,
        n_matches,
        ipc,
    )
}

pub fn benchmark_patterns(
    queries: &[Vec<u8>],
    text: &[u8],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    log_label: &str,
    measure_ipc: bool,
) -> BenchmarkResults {
    println!("  {} sassy patterns...", log_label);

    let mut searcher = if USE_RC {
        Searcher::<Iupac>::new_rc()
    } else {
        Searcher::<Iupac>::new_fwd()
    };

    let mut do_work = || {
        let _matches = searcher.search_patterns(queries, text, k);
        black_box(_matches);
    };

    let (median, mean, std_dev, ci_lower, ci_upper) =
        benchmark_with_stats(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = searcher.search_patterns(queries, text, k).len();

    finish_results(
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        total_bytes,
        n_matches,
        ipc,
    )
}

pub fn benchmark_tiling(
    queries: &[Vec<u8>],
    text: &[u8],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    log_label: &str,
    measure_ipc: bool,
) -> BenchmarkResults {
    println!("  {} sassy tiling...", log_label);

    let mut searcher = if USE_RC {
        Searcher::<Iupac>::new_rc()
    } else {
        Searcher::<Iupac>::new_fwd()
    };

    let encoded = searcher.encode_patterns(queries);
    let mut do_work = || {
        let _matches = searcher.search_encoded_patterns(&encoded, text, k);
        black_box(_matches);
    };

    let (median, mean, std_dev, ci_lower, ci_upper) =
        benchmark_with_stats(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = searcher.search_encoded_patterns(&encoded, text, k).len();

    finish_results(
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        total_bytes,
        n_matches,
        ipc,
    )
}

pub fn benchmark_edlib(
    queries: &[Vec<u8>],
    text: &[u8],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    alphabet: &Alphabet,
    log_label: &str,
    measure_ipc: bool,
) -> BenchmarkResults {
    println!("  {} edlib...", log_label);

    let edlib_config = get_edlib_config(k as i32, alphabet);
    let mut do_work = || {
        for query in queries {
            black_box(query);
            let _result = run_edlib(query, text, &edlib_config);
            black_box(_result);
        }
    };

    let (median, mean, std_dev, ci_lower, ci_upper) =
        benchmark_with_stats(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = queries
        .iter()
        .filter(|q| {
            let result = run_edlib(q, text, &edlib_config);
            result.editDistance <= k as i32 && result.editDistance != -1
        })
        .count();

    finish_results(
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        total_bytes,
        n_matches,
        ipc,
    )
}

pub fn benchmark_individual_search_many_texts(
    queries: &[Vec<u8>],
    texts: &[impl AsRef<[u8]> + Send + Sync],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    threads: usize,
    log_label: &str,
    measure_ipc: bool,
) -> BenchmarkResults {
    println!(
        "  {} sassy search ({} texts, {} threads)...",
        log_label,
        texts.len(),
        threads.max(1)
    );

    let mut searcher = if USE_RC {
        Searcher::<Iupac>::new_rc()
    } else {
        Searcher::<Iupac>::new_fwd()
    };

    let mut do_work = || {
        if threads <= 1 {
            for text in texts {
                let t = text.as_ref();
                for query in queries {
                    let _matches = searcher.search(query, t, k);
                    black_box(_matches);
                }
            }
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let chunk_size = (texts.len() + threads - 1) / threads;
            pool.install(|| {
                texts.par_chunks(chunk_size).for_each(|chunk| {
                    let mut worker = if USE_RC {
                        Searcher::<Iupac>::new_rc()
                    } else {
                        Searcher::<Iupac>::new_fwd()
                    };
                    for text in chunk {
                        let t = text.as_ref();
                        for query in queries {
                            let _matches = worker.search(query, t, k);
                            black_box(_matches);
                        }
                    }
                });
            });
        }
    };

    let (median, mean, std_dev, ci_lower, ci_upper) =
        benchmark_with_stats(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = texts.iter().fold(0usize, |acc, text| {
        acc + queries
            .iter()
            .map(|q| searcher.search(q, text.as_ref(), k).len())
            .sum::<usize>()
    });

    finish_results(
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        total_bytes,
        n_matches,
        ipc,
    )
}

pub fn benchmark_tiling_many_texts(
    queries: &[Vec<u8>],
    texts: &[impl AsRef<[u8]> + Send + Sync],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    threads: usize,
    log_label: &str,
    measure_ipc: bool,
) -> BenchmarkResults {
    println!(
        "  {} sassy tiling ({} texts, {} threads)...",
        log_label,
        texts.len(),
        threads.max(1)
    );

    let mut searcher = if USE_RC {
        Searcher::<Iupac>::new_rc()
    } else {
        Searcher::<Iupac>::new_fwd()
    };

    let encoded = Arc::new(searcher.encode_patterns(queries));
    let mut do_work = || {
        if threads <= 1 {
            for text in texts {
                let _matches = searcher.search_encoded_patterns(&*encoded, text.as_ref(), k);
                black_box(_matches);
            }
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let chunk_size = (texts.len() + threads - 1) / threads;
            pool.install(|| {
                texts.par_chunks(chunk_size).for_each(|chunk| {
                    let encoded = Arc::clone(&encoded);
                    let mut worker = if USE_RC {
                        Searcher::<Iupac>::new_rc()
                    } else {
                        Searcher::<Iupac>::new_fwd()
                    };
                    for text in chunk {
                        let _matches = worker.search_encoded_patterns(&*encoded, text.as_ref(), k);
                        black_box(_matches);
                    }
                });
            });
        }
    };

    let (median, mean, std_dev, ci_lower, ci_upper) =
        benchmark_with_stats(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = texts
        .iter()
        .map(|t| {
            searcher
                .search_encoded_patterns(&*encoded, t.as_ref(), k)
                .len()
        })
        .sum();

    finish_results(
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        total_bytes,
        n_matches,
        ipc,
    )
}

pub fn benchmark_edlib_many_texts(
    queries: &[Vec<u8>],
    texts: &[impl AsRef<[u8]> + Send + Sync],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    threads: usize,
    alphabet: &Alphabet,
    log_label: &str,
    measure_ipc: bool,
) -> BenchmarkResults {
    println!(
        "  {} edlib ({} texts, {} threads)...",
        log_label,
        texts.len(),
        threads.max(1)
    );

    let edlib_config = get_edlib_config(k as i32, alphabet);
    let mut do_work = || {
        if threads <= 1 {
            for text in texts {
                let t = text.as_ref();
                for query in queries {
                    let _result = run_edlib(query, t, &edlib_config);
                    black_box(_result);
                }
            }
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let chunk_size = (texts.len() + threads - 1) / threads;
            pool.install(|| {
                texts.par_chunks(chunk_size).for_each(|chunk| {
                    let worker_config = get_edlib_config(k as i32, alphabet);
                    for text in chunk {
                        let t = text.as_ref();
                        for query in queries {
                            let _result = run_edlib(query, t, &worker_config);
                            black_box(_result);
                        }
                    }
                });
            });
        }
    };

    let (median, mean, std_dev, ci_lower, ci_upper) =
        benchmark_with_stats(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = texts.iter().fold(0usize, |acc, text| {
        acc + queries
            .iter()
            .filter(|q| {
                let result = run_edlib(q, text.as_ref(), &edlib_config);
                result.editDistance <= k as i32 && result.editDistance != -1
            })
            .count()
    });

    finish_results(
        median,
        mean,
        std_dev,
        ci_lower,
        ci_upper,
        total_bytes,
        n_matches,
        ipc,
    )
}

pub const BENCH_CSV_HEADER: &str = "search_median_ms,search_mean_ms,search_std_ms,search_ci_lower_ms,search_ci_upper_ms,search_n_matches,tiling_median_ms,tiling_mean_ms,tiling_std_ms,tiling_ci_lower_ms,tiling_ci_upper_ms,tiling_n_matches,edlib_median_ms,edlib_mean_ms,edlib_std_ms,edlib_ci_lower_ms,edlib_ci_upper_ms,edlib_n_matches,search_ipc,tiling_ipc,edlib_ipc,search_throughput_gbps,search_ci_lower_throughput_gbps,search_ci_upper_throughput_gbps,tiling_throughput_gbps,tiling_ci_lower_throughput_gbps,tiling_ci_upper_throughput_gbps,edlib_throughput_gbps,edlib_ci_lower_throughput_gbps,edlib_ci_upper_throughput_gbps,throughput_bytes";

pub struct BenchCsv {
    file: File,
}

impl BenchCsv {
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut file = File::create(path)?;
        writeln!(
            file,
            "num_queries,target_len,query_len,k,{}",
            BENCH_CSV_HEADER
        )?;
        Ok(Self { file })
    }

    pub fn write_row(
        &mut self,
        num_queries: usize,
        target_len: usize,
        query_len: usize,
        k: usize,
        search: &BenchmarkResults,
        tiling: &BenchmarkResults,
        edlib: &BenchmarkResults,
        total_bytes: usize,
    ) -> std::io::Result<()> {
        writeln!(
            self.file,
            "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.2},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
            num_queries,
            target_len,
            query_len,
            k,
            search.median,
            search.mean,
            search.std_dev,
            search.ci_lower,
            search.ci_upper,
            search.n_matches,
            tiling.median,
            tiling.mean,
            tiling.std_dev,
            tiling.ci_lower,
            tiling.ci_upper,
            tiling.n_matches,
            edlib.median,
            edlib.mean,
            edlib.std_dev,
            edlib.ci_lower,
            edlib.ci_upper,
            edlib.n_matches,
            search.ipc.unwrap_or(NO_IPC),
            tiling.ipc.unwrap_or(NO_IPC),
            edlib.ipc.unwrap_or(NO_IPC),
            search.throughput_gbps,
            search.ci_lower_throughput_gbps,
            search.ci_upper_throughput_gbps,
            tiling.throughput_gbps,
            tiling.ci_lower_throughput_gbps,
            tiling.ci_upper_throughput_gbps,
            edlib.throughput_gbps,
            edlib.ci_lower_throughput_gbps,
            edlib.ci_upper_throughput_gbps,
            total_bytes
        )
    }
}
