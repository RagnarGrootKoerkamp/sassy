use perfcnt::AbstractPerfCounter;
use perfcnt::linux::{HardwareEventType, PerfCounterBuilderLinux};
use rand::random_range;
use sassy::Searcher;
use sassy::profiles::Iupac;
use std::hint::black_box;

// Measure IPC (Instructions per CPU cycle)
fn measure_ipc<F: FnOnce()>(f: F) -> f64 {
    let mut instr = PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::Instructions)
        .finish()
        .expect("Could not create instruction counter");
    let mut cycles = PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::CPUCycles)
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
        "{:<12} | {:<20} | {:<20} | {:<20}",
        "Num Queries", "Single-query IPC", "Pattern IPC", "Encoded Tiling IPC"
    );
    println!("{}", "-".repeat(80));

    for &num_queries in &num_queries_list {
        let queries: Vec<Vec<u8>> = (0..num_queries)
            .map(|_| {
                (0..query_len)
                    .map(|_| b"ACGT"[random_range(0..4)])
                    .collect()
            })
            .collect();

        // Warmup
        for _ in 0..3 {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            for query in &queries {
                black_box(searcher.search(query, &text, k));
            }
        }

        let ipc_search = measure_ipc(|| {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            for query in &queries {
                black_box(searcher.search(query, &text, k));
            }
        });

        // Warmup
        for _ in 0..3 {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            black_box(searcher.search_patterns(&queries, &text, k));
        }

        let ipc_patterns = measure_ipc(|| {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            black_box(searcher.search_patterns(&queries, &text, k));
        });

        // Encode patterns once
        let mut searcher = Searcher::<Iupac>::new_fwd();
        let encoded = searcher.encode_patterns(&queries);

        // Warmup
        for _ in 0..3 {
            black_box(searcher.search_encoded_patterns(&encoded, &text, k));
        }

        let ipc_tiling = measure_ipc(|| {
            black_box(searcher.search_encoded_patterns(&encoded, &text, k));
        });

        println!(
            "{:<12} | {:<20.2} | {:<20.2} | {:<20.2}",
            num_queries, ipc_search, ipc_patterns, ipc_tiling
        );
    }
}
