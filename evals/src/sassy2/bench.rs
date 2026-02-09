use edlib_rs::*;
use sassy::EncodedPatterns;
use sassy::Searcher;
use sassy::profiles::Iupac;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::sassy1::edlib_bench::edlib::{get_edlib_config, run_edlib};
use crate::sassy1::edlib_bench::sim_data::Alphabet;
use rayon::prelude::*;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use perfcnt::AbstractPerfCounter;

const NO_IPC: f64 = 0.0; // If IPC is not measured, we set it to 0.0
const USE_RC: bool = false; // Only applies to sassy 1/2, just for testing not used via config
const MIN_PATTERNS_PER_CHUNK: usize = 32; // For amd we do want to have at least 32 to pipeline two blocks
const MIN_TEXT_BYTES_PER_CHUNK: usize = 1_000_000; // 1MB

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
    pub ipc: Option<f64>,
}

impl fmt::Display for BenchmarkResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:8.2}ms  {:6.2} GB/s  {:>10} matches",
            self.median, self.throughput_gbps, self.n_matches
        )?;
        if let Some(ipc) = self.ipc
            && ipc > 0.0
        {
            write!(f, "  IPC: {:.2}", ipc)?;
        }
        Ok(())
    }
}

pub struct BenchmarkSuite {
    pub search: BenchmarkResults,
    pub tiling: BenchmarkResults,
    pub edlib: BenchmarkResults,
}

impl fmt::Display for BenchmarkSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{:─<80}", "")?;
        writeln!(f, "Benchmark Results")?;
        writeln!(f, "{:─<80}", "")?;
        writeln!(
            f,
            "{:<20} {:>10}  {:>12}  {:>12}",
            "Tool", "Time", "Throughput", "Matches"
        )?;
        writeln!(f, "{:─<80}", "")?;
        writeln!(f, "{:<20} {}", "Sassy Search", self.search)?;
        writeln!(f, "{:<20} {}", "Sassy Tiling", self.tiling)?;
        writeln!(f, "{:<20} {}", "Edlib", self.edlib)?;
        writeln!(f, "{:─<80}", "")?;
        Ok(())
    }
}

impl fmt::Debug for BenchmarkSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Runs a "work" function and gets median timings
fn benchmark_with_stats_and_results<F, R>(
    f: &mut F,
    warmup: usize,
    iterations: usize,
) -> ((f64, f64, f64, f64, f64), Vec<R>)
where
    F: FnMut() -> R,
{
    // Do warmup if needed
    for _ in 0..warmup {
        f();
    }

    // Run `iterations` times and get the median, mean, variance etc.
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

/// Get the median IPC, as instructions / cycles
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

/// If not linux, we can't use perfcnt we just 0.0
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn measure_ipc_median<F, R>(f: &mut F, _iterations: usize) -> f64
where
    F: FnMut() -> R,
{
    f();
    NO_IPC
}

fn calculate_throughput(total_bytes: usize, median_ms: f64) -> f64 {
    (total_bytes as f64 / (median_ms / 1000.0)) / 1_000_000_000.0
}

/// (chunk_idx, pattern_start, pattern_end, text_start, text_end)
type WorkChunk = (usize, usize, usize, usize, usize);

/// Create work chunks based on pattern batches and text accumulation
fn create_work_chunks<T: AsRef<[u8]>>(num_patterns: usize, texts: &[T]) -> Vec<WorkChunk> {
    if num_patterns == 0 || texts.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut pattern_start = 0;
    let mut chunk_idx = 0;

    // Iterate over pattern batches (MIN_PATTERNS_PER_CHUNK)
    while pattern_start < num_patterns {
        let pattern_end = (pattern_start + MIN_PATTERNS_PER_CHUNK).min(num_patterns);

        let mut text_start = 0;
        while text_start < texts.len() {
            let mut accumulated_bytes = 0;
            let mut text_end = text_start;

            // Accumulate texts until we hit MIN_TEXT_BYTES_PER_CHUNK or are done
            while text_end < texts.len() && accumulated_bytes < MIN_TEXT_BYTES_PER_CHUNK {
                accumulated_bytes += texts[text_end].as_ref().len();
                text_end += 1;
            }

            // We always have to move forward here
            if text_end == text_start {
                text_end = text_start + 1;
            }

            chunks.push((chunk_idx, pattern_start, pattern_end, text_start, text_end));
            text_start = text_end;
        }

        pattern_start = pattern_end;
        chunk_idx += 1;
    }

    chunks
}

/// Trait for all tools to bench (prepare, search, new_worker)
trait SearchTool: Send + Sync {
    type PreparedData: Send + Sync;

    /// Prepare/setup any data structures needed (like encoding patterns)
    fn prepare(&self, queries: &[Vec<u8>]) -> Self::PreparedData;

    /// Create a new worker instance for parallel execution
    fn new_worker(&self) -> Self;

    /// Execute search on pattern slice and text, return match count
    /// chunk_idx indicates which pattern chunk this is (for accessing pre-encoded data)
    fn search(
        &mut self,
        prepared: &Self::PreparedData,
        chunk_idx: usize,
        pattern_slice: &[Vec<u8>],
        text: &[u8],
        k: usize,
    ) -> usize;
}

/// Sassy v1 - text tilling
struct SassyV1;

impl SearchTool for SassyV1 {
    type PreparedData = ();

    fn prepare(&self, _queries: &[Vec<u8>]) -> Self::PreparedData {}

    fn new_worker(&self) -> Self {
        SassyV1
    }

    #[inline(always)]
    fn search(
        &mut self,
        _prepared: &Self::PreparedData,
        _chunk_idx: usize,
        pattern_slice: &[Vec<u8>],
        text: &[u8],
        k: usize,
    ) -> usize {
        let mut searcher = if USE_RC {
            Searcher::<Iupac>::new_rc()
        } else {
            Searcher::<Iupac>::new_fwd()
        };
        let mut count = 0;
        for pattern in pattern_slice {
            count += searcher.search(pattern, text, k).len();
        }
        count
    }
}

/// Sassy V2 - pattern tilling pre encoded patterns
struct SassyV2;

impl SearchTool for SassyV2 {
    type PreparedData = Vec<EncodedPatterns<Iupac>>;

    fn prepare(&self, queries: &[Vec<u8>]) -> Self::PreparedData {
        // Pre-encode patterns for each pattern chunk
        let mut encoded_chunks = Vec::new();
        let mut pattern_start = 0;

        while pattern_start < queries.len() {
            let pattern_end = (pattern_start + MIN_PATTERNS_PER_CHUNK).min(queries.len());
            let pattern_slice = &queries[pattern_start..pattern_end];

            let mut searcher = if USE_RC {
                Searcher::<Iupac>::new_rc()
            } else {
                Searcher::<Iupac>::new_fwd()
            };
            encoded_chunks.push(searcher.encode_patterns(pattern_slice));

            pattern_start = pattern_end;
        }

        encoded_chunks
    }

    fn new_worker(&self) -> Self {
        SassyV2
    }

    #[inline(always)]
    fn search(
        &mut self,
        prepared: &Self::PreparedData,
        chunk_idx: usize,
        _pattern_slice: &[Vec<u8>],
        text: &[u8],
        k: usize,
    ) -> usize {
        let mut searcher = if USE_RC {
            Searcher::<Iupac>::new_rc()
        } else {
            Searcher::<Iupac>::new_fwd()
        };

        // Use the pre-encoded patterns for this chunk
        searcher
            .search_encoded_patterns(&prepared[chunk_idx], text, k)
            .len()
    }
}

/// Edlib
struct Edlib {
    alphabet: Alphabet,
}

impl SearchTool for Edlib {
    type PreparedData = EdlibAlignConfigRs<'static>;

    fn prepare(&self, _queries: &[Vec<u8>]) -> Self::PreparedData {
        get_edlib_config(0, &self.alphabet) // k will be set in search
    }

    fn new_worker(&self) -> Self {
        Edlib {
            alphabet: self.alphabet,
        }
    }

    #[inline(always)]
    fn search(
        &mut self,
        prepared: &Self::PreparedData,
        _chunk_idx: usize,
        pattern_slice: &[Vec<u8>],
        text: &[u8],
        k: usize,
    ) -> usize {
        let mut config = *prepared;
        config.k = k as i32;

        let mut count = 0;
        for query in pattern_slice {
            let result = run_edlib(query, text, &config);
            if result.editDistance <= k as i32 && result.editDistance != -1 {
                count += result.endLocations.unwrap_or_default().len();
            }
        }
        count
    }
}

/// Here we do all the benching given any Tool that impl SearchTool
fn benchmark_tool<Tool, T>(
    tool: Tool,
    queries: &[Vec<u8>],
    texts: &[T],
    k: usize,
    total_bytes: usize,
    warmup: usize,
    iterations: usize,
    threads: usize,
    log_label: &str,
    measure_ipc: bool,
) -> BenchmarkResults
where
    Tool: SearchTool,
    T: AsRef<[u8]> + Send + Sync,
{
    println!(
        "  {} ({} texts, {} threads)...",
        log_label,
        texts.len(),
        threads.max(1)
    );

    let prepared = Arc::new(tool.prepare(queries));

    let work_chunks = if threads > 1 {
        create_work_chunks(queries.len(), texts)
    } else {
        vec![]
    };

    let mut do_work = || {
        if threads <= 1 {
            let mut worker = tool.new_worker();
            let mut count = 0usize;
            for text in texts {
                count += worker.search(&*prepared, 0, queries, text.as_ref(), k);
            }
            count
        } else {
            if work_chunks.is_empty() {
                return 0usize;
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                work_chunks
                    .par_iter()
                    .map(|&(chunk_idx, pat_start, pat_end, text_start, text_end)| {
                        let mut worker = tool.new_worker();
                        let prepared = Arc::clone(&prepared);
                        let mut count = 0usize;
                        let pattern_slice = &queries[pat_start..pat_end];
                        for text in &texts[text_start..text_end] {
                            count += worker.search(
                                &*prepared,
                                chunk_idx,
                                pattern_slice,
                                text.as_ref(),
                                k,
                            );
                        }
                        count
                    })
                    .sum()
            })
        }
    };

    let ((median, mean, std_dev, ci_lower, ci_upper), match_counts) =
        benchmark_with_stats_and_results(&mut do_work, warmup, iterations);

    // not used for the expensive benches (Nanopore, off-target) anyway so we can just run this for the
    // `cheap` benches separate
    let ipc = if measure_ipc {
        Some(measure_ipc_median(&mut do_work, iterations))
    } else {
        None
    };

    let n_matches = match_counts.first().copied().unwrap_or(0);

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

/// Call this do to a single bench run across sassy v1, sassy v2, and edlib
#[allow(clippy::too_many_arguments)]
pub fn benchmark_tools<T>(
    queries: &[Vec<u8>],
    texts: &[T],
    k: usize,
    warmup: usize,
    iterations: usize,
    threads: usize,
    alphabet: &Alphabet,
    measure_ipc: bool,
) -> BenchmarkSuite
where
    T: AsRef<[u8]> + Send + Sync,
{
    // We consider the total throughput the text * number of patterns as if
    // each pattern entirely scans the text (like in v1, and edlib, but not in v2 really)
    let text_bytes: usize = texts.iter().map(|t| t.as_ref().len()).sum();
    let total_bytes = text_bytes * queries.len();

    println!(
        "Benchmarking with {} queries, {} texts ({} text bytes, {} total throughput bytes), k={}",
        queries.len(),
        texts.len(),
        text_bytes,
        total_bytes,
        k
    );

    let search = benchmark_tool(
        SassyV1,
        queries,
        texts,
        k,
        total_bytes,
        warmup,
        iterations,
        threads,
        "Sassy V1",
        measure_ipc,
    );

    let tiling = benchmark_tool(
        SassyV2,
        queries,
        texts,
        k,
        total_bytes,
        warmup,
        iterations,
        threads,
        "Sassy V2",
        measure_ipc,
    );

    let edlib = benchmark_tool(
        Edlib {
            alphabet: *alphabet,
        },
        queries,
        texts,
        k,
        total_bytes,
        warmup,
        iterations,
        threads,
        "Edlib",
        measure_ipc,
    );

    BenchmarkSuite {
        search,
        tiling,
        edlib,
    }
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

    // fixme: this is aaa lot of columns
    pub fn write_row(
        &mut self,
        num_queries: usize,
        target_len: usize,
        query_len: usize,
        k: usize,
        suite: &BenchmarkSuite,
        total_bytes: usize,
    ) -> std::io::Result<()> {
        let search = &suite.search;
        let tiling = &suite.tiling;
        let edlib = &suite.edlib;

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
