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

fn benchmark_with_stats_and_results<F, R>(
    f: &mut F,
    warmup: usize,
    iterations: usize,
) -> ((f64, f64, f64, f64, f64), Vec<R>)
where
    F: FnMut() -> R,
{
    for _ in 0..warmup {
        f();
    }

    let mut times = Vec::with_capacity(iterations);
    let mut results = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let result = f();
        times.push(start.elapsed().as_nanos() as f64 / 1_000_000.0);
        results.push(result);
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

    ((median, mean, std_dev, ci_lower, ci_upper), results)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn measure_ipc_median<F, R>(f: &mut F, iterations: usize) -> f64
where
    F: FnMut() -> R,
{
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
fn measure_ipc_median<F, R>(f: &mut F, _iterations: usize) -> f64
where
    F: FnMut() -> R,
{
    f();
    NO_IPC
}

fn p_chunks<T: AsRef<[u8]>>(texts: &[T], boundaries: &[usize]) {
    let total: usize = texts.iter().map(|t| t.as_ref().len()).sum();
    let mut out = String::new();
    let mut total_reads = 0;
    for (i, w) in boundaries.windows(2).enumerate() {
        let chunk_len: usize = texts[w[0]..w[1]].iter().map(|t| t.as_ref().len()).sum();
        let perc = 100.0 * chunk_len as f64 / total as f64;
        let n_reads = w[1] - w[0];
        total_reads += n_reads;
        out.push_str(&format!(
            "chunk {} ({:.1}% nts) {} reads  ",
            i, perc, n_reads
        ));
    }
    assert_eq!(total_reads, texts.len());
    println!("{}", out.trim_end());
}

fn balance_by_text_len<T: AsRef<[u8]>>(texts: &[T], threads: usize) -> Vec<usize> {
    let n = texts.len();
    if n == 0 {
        return vec![];
    }

    // At least 1 at most n
    let num_chunks = threads.clamp(1, n);

    // Only 1, we just return boundary for entire text slice
    if num_chunks == 1 {
        return vec![0, n];
    }

    let mut cum = Vec::with_capacity(n + 1);
    cum.push(0usize);
    for t in texts {
        cum.push(cum.last().unwrap() + t.as_ref().len());
    }
    let total = *cum.last().unwrap();
    assert_eq!(total, texts.iter().map(|t| t.as_ref().len()).sum());
    assert!(total > 0);

    let mut boundaries = Vec::with_capacity(num_chunks + 1);
    boundaries.push(0);

    for k in 1..num_chunks {
        let target = (total * k) / num_chunks;
        // We find the index where the cumulative length is greater than or equal to our target
        // length
        let idx = match cum.binary_search_by(|&c| c.cmp(&target)) {
            Ok(i) => i,
            Err(i) => i,
        };
        boundaries.push(idx);
    }

    boundaries.push(n);
    p_chunks(texts, &boundaries);
    boundaries
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

    let boundaries = if threads > 1 {
        balance_by_text_len(texts, threads)
    } else {
        vec![]
    };

    let mut do_work = || {
        if threads <= 1 {
            let mut count = 0usize;
            for text in texts {
                let t = text.as_ref();
                for query in queries {
                    count += searcher.search(query, t, k).len();
                }
            }
            count
        } else {
            let num_chunks = boundaries.len().saturating_sub(1);
            if num_chunks == 0 {
                return 0usize;
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                (0..num_chunks)
                    .into_par_iter()
                    .map(|j| {
                        let chunk = &texts[boundaries[j]..boundaries[j + 1]];
                        let mut worker = if USE_RC {
                            Searcher::<Iupac>::new_rc()
                        } else {
                            Searcher::<Iupac>::new_fwd()
                        };
                        let mut count = 0usize;
                        for text in chunk {
                            let t = text.as_ref();
                            for query in queries {
                                count += worker.search(query, t, k).len();
                            }
                        }
                        count
                    })
                    .sum()
            })
        }
    };

    let ((median, mean, std_dev, ci_lower, ci_upper), match_counts) =
        benchmark_with_stats_and_results(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = match_counts.first().copied().unwrap_or(0);

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
    let boundaries = if threads > 1 {
        balance_by_text_len(texts, threads)
    } else {
        vec![]
    };

    let mut do_work = || {
        if threads <= 1 {
            let mut count = 0usize;
            for text in texts {
                count += searcher
                    .search_encoded_patterns(&*encoded, text.as_ref(), k)
                    .len();
            }
            count
        } else {
            let num_chunks = boundaries.len().saturating_sub(1);
            if num_chunks == 0 {
                return 0usize;
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                (0..num_chunks)
                    .into_par_iter()
                    .map(|j| {
                        let chunk = &texts[boundaries[j]..boundaries[j + 1]];
                        let encoded = Arc::clone(&encoded);
                        let mut worker = if USE_RC {
                            Searcher::<Iupac>::new_rc()
                        } else {
                            Searcher::<Iupac>::new_fwd()
                        };
                        let mut count = 0usize;
                        for text in chunk {
                            count += worker
                                .search_encoded_patterns(&*encoded, text.as_ref(), k)
                                .len();
                        }
                        count
                    })
                    .sum()
            })
        }
    };

    let ((median, mean, std_dev, ci_lower, ci_upper), match_counts) =
        benchmark_with_stats_and_results(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = match_counts.first().copied().unwrap_or(0);

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

    let edlib_config = Arc::new(get_edlib_config(k as i32, alphabet));
    let boundaries = if threads > 1 {
        balance_by_text_len(texts, threads)
    } else {
        vec![]
    };

    let mut do_work = || {
        if threads <= 1 {
            let mut count = 0usize;
            for text in texts {
                let t = text.as_ref();
                for query in queries {
                    let result = run_edlib(query, t, &*edlib_config);
                    if result.editDistance <= k as i32 && result.editDistance != -1 {
                        // Might have multiple end points with same lowest k, so get
                        // len(end locs)
                        count += result.endLocations.unwrap_or_default().len() as usize;
                    }
                }
            }
            count
        } else {
            let num_chunks = boundaries.len().saturating_sub(1);
            if num_chunks == 0 {
                return 0usize;
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                (0..num_chunks)
                    .into_par_iter()
                    .map(|j| {
                        let chunk = &texts[boundaries[j]..boundaries[j + 1]];
                        let config = Arc::clone(&edlib_config);
                        let mut count = 0usize;
                        for text in chunk {
                            let t = text.as_ref();
                            for query in queries {
                                let result = run_edlib(query, t, &*config);
                                if result.editDistance <= k as i32 && result.editDistance != -1 {
                                    count += result.endLocations.unwrap_or_default().len() as usize;
                                }
                            }
                        }
                        count
                    })
                    .sum()
            })
        }
    };

    let ((median, mean, std_dev, ci_lower, ci_upper), match_counts) =
        benchmark_with_stats_and_results(&mut do_work, warmup, iterations);

    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = match_counts.first().copied().unwrap_or(0);

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
