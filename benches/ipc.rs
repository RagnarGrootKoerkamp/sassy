use rand::random_range;
use sassy::Searcher;
use sassy::profiles::Iupac;
use std::hint::black_box;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use perfcnt::AbstractPerfCounter;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use perfcnt::linux::{HardwareEventType, PerfCounterBuilderLinux};

#[derive(Debug, Default)]
struct PerfStats {
    instructions: Option<u64>,
    cycles: Option<u64>,
    branch_instructions: Option<u64>,
    branch_misses: Option<u64>,
    cache_references: Option<u64>,
    cache_misses: Option<u64>,
}

impl PerfStats {
    fn ipc(&self) -> Option<f64> {
        match (self.instructions, self.cycles) {
            (Some(i), Some(c)) if c > 0 => Some(i as f64 / c as f64),
            _ => None,
        }
    }
    fn branch_miss_rate(&self) -> Option<f64> {
        match (self.branch_misses, self.branch_instructions) {
            (Some(m), Some(bi)) if bi > 0 => Some(m as f64 / bi as f64),
            _ => None,
        }
    }
    fn cache_miss_rate(&self) -> Option<f64> {
        match (self.cache_misses, self.cache_references) {
            (Some(mm), Some(cr)) if cr > 0 => Some(mm as f64 / cr as f64),
            _ => None,
        }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn try_make(event: HardwareEventType) -> Option<Box<dyn AbstractPerfCounter>> {
    PerfCounterBuilderLinux::from_hardware_event(event)
        .finish()
        .map(|c| Box::new(c) as Box<dyn AbstractPerfCounter>)
        .ok()
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn measure_perf<F: FnMut()>(mut f: F) -> PerfStats {
    let mut instr = try_make(HardwareEventType::Instructions);
    let mut cycles = try_make(HardwareEventType::CPUCycles);
    let mut branch_instructions = try_make(HardwareEventType::BranchInstructions);
    let mut branch_misses = try_make(HardwareEventType::BranchMisses);
    let mut cache_references = try_make(HardwareEventType::CacheReferences);
    let mut cache_misses = try_make(HardwareEventType::CacheMisses);

    if let Some(c) = instr.as_mut() {
        c.start().ok();
    }
    if let Some(c) = cycles.as_mut() {
        c.start().ok();
    }
    if let Some(c) = branch_instructions.as_mut() {
        c.start().ok();
    }
    if let Some(c) = branch_misses.as_mut() {
        c.start().ok();
    }
    if let Some(c) = cache_references.as_mut() {
        c.start().ok();
    }
    if let Some(c) = cache_misses.as_mut() {
        c.start().ok();
    }

    f();

    if let Some(c) = instr.as_mut() {
        c.stop().ok();
    }
    if let Some(c) = cycles.as_mut() {
        c.stop().ok();
    }
    if let Some(c) = branch_instructions.as_mut() {
        c.stop().ok();
    }
    if let Some(c) = branch_misses.as_mut() {
        c.stop().ok();
    }
    if let Some(c) = cache_references.as_mut() {
        c.stop().ok();
    }
    if let Some(c) = cache_misses.as_mut() {
        c.stop().ok();
    }

    let read_opt = |c: Option<Box<dyn AbstractPerfCounter>>| -> Option<u64> {
        c.and_then(|mut cc| cc.read().ok())
    };

    PerfStats {
        instructions: read_opt(instr),
        cycles: read_opt(cycles),
        branch_instructions: read_opt(branch_instructions),
        branch_misses: read_opt(branch_misses),
        cache_references: read_opt(cache_references),
        cache_misses: read_opt(cache_misses),
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn measure_perf<F: FnMut()>(mut f: F) -> PerfStats {
    // Perf counters aren't available/portable here (e.g. macOS / Apple Silicon).
    f();
    PerfStats::default()
}

fn main() {
    let query_len = 23;
    let text_len = 100_000;
    let k = 3;

    let text: Vec<u8> = (0..text_len).map(|_| b"ACGT"[random_range(0..4)]).collect();

    let num_queries_list: Vec<usize> = (0..=7)
        .map(|i| 1usize << i)
        .chain(std::iter::once(96))
        .collect();

    println!(
        "{:<6} | {:^20} | {:^25} | {:^25}",
        "NumQ", "IPC", "Branch Miss %", "Cache Miss %"
    );
    println!(
        "{:<6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}",
        "", "S", "P", "P2", "S", "P", "P2", "S", "P", "P2"
    );
    println!("{}", "-".repeat(80));

    for &num_queries in &num_queries_list {
        // build queries
        let queries: Vec<Vec<u8>> = (0..num_queries)
            .map(|_| {
                (0..query_len)
                    .map(|_| b"ACGT"[random_range(0..4)])
                    .collect()
            })
            .collect();

        let reps = 4;

        let ipc_search = measure_perf(|| {
            for _ in 0..reps {
                let mut s = Searcher::<Iupac>::new_fwd();
                for q in &queries {
                    black_box(s.search(q, &text, k));
                }
            }
        });

        let ipc_patterns = measure_perf(|| {
            for _ in 0..reps {
                let mut s = Searcher::<Iupac>::new_fwd();
                black_box(s.search_patterns(&queries, &text, k));
            }
        });

        let mut s_enc = Searcher::<Iupac>::new_fwd();
        let encoded = s_enc.encode_patterns(&queries);
        let ipc_tiling = measure_perf(|| {
            for _ in 0..reps {
                black_box(s_enc.search_encoded_patterns(&encoded, &text, k));
            }
        });

        let fmt_ipc = |ps: &PerfStats| ps.ipc().map_or("-".into(), |v| format!("{:.2}", v));

        println!(
            "{:<6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}",
            num_queries,
            fmt_ipc(&ipc_search),
            fmt_ipc(&ipc_patterns),
            fmt_ipc(&ipc_tiling),
            ipc_search
                .branch_miss_rate()
                .map_or("-".into(), |v| format!("{:.2}%", v)),
            ipc_patterns
                .branch_miss_rate()
                .map_or("-".into(), |v| format!("{:.2}%", v)),
            ipc_tiling
                .branch_miss_rate()
                .map_or("-".into(), |v| format!("{:.2}%", v)),
            ipc_search
                .cache_miss_rate()
                .map_or("-".into(), |v| format!("{:.2}%", v)),
            ipc_patterns
                .cache_miss_rate()
                .map_or("-".into(), |v| format!("{:.2}%", v)),
            ipc_tiling
                .cache_miss_rate()
                .map_or("-".into(), |v| format!("{:.2}%", v)),
        );
    }
}
