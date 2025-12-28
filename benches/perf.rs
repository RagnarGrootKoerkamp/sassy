use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::random_range;
use sassy::Searcher;
use sassy::profiles::Iupac;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use perfcnt::AbstractPerfCounter;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use perfcnt::linux::{HardwareEventType, PerfCounterBuilderLinux};

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
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

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn measure_ipc<F: FnOnce()>(f: F) -> f64 {
    // Perf counters aren't available/portable here (e.g. macOS / Apple Silicon).
    f();
    0.0
}

fn bench_searchers(c: &mut Criterion) {
    let query_len = 23;
    let text_len = 100_000;
    let k = 3;

    let text: Vec<u8> = (0..text_len).map(|_| b"ACGT"[random_range(0..4)]).collect();

    let mut num_queries_list: Vec<usize> = (0..=7)
        .map(|i| 1usize << i)
        .chain(std::iter::once(96))
        .collect();

    num_queries_list.sort_unstable();
    num_queries_list.dedup();

    let mut group = c.benchmark_group("search_performance");
    group.sample_size(10);

    for &num_queries in &num_queries_list {
        let queries: Vec<Vec<u8>> = (0..num_queries)
            .map(|_| {
                (0..query_len)
                    .map(|_| b"ACGT"[random_range(0..4)])
                    .collect()
            })
            .collect();

        group.throughput(Throughput::Bytes((text_len * num_queries) as u64));

        group.bench_function(format!("sassy_search/q={}", num_queries), |b| {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            b.iter(|| {
                for query in &queries {
                    black_box(searcher.search(query, &text, k));
                }
            });
        });
        let ipc1 = measure_ipc(|| {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            for query in &queries {
                black_box(searcher.search(query, &text, k));
            }
        });

        group.bench_function(format!("sassy_search_patterns/q={}", num_queries), |b| {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            b.iter(|| {
                black_box(searcher.search_patterns(&queries, &text, k));
            });
        });
        let ipc2 = measure_ipc(|| {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            black_box(searcher.search_patterns(&queries, &text, k));
        });

        let mut searcher = Searcher::<Iupac>::new_fwd();
        let encoded = searcher.encode_patterns(&queries);

        group.bench_function(format!("sassy_pattern_tiling/q={}", num_queries), |b| {
            b.iter(|| {
                black_box(searcher.search_encoded_patterns(&encoded, &text, k));
            });
        });
        let ipc3 = measure_ipc(|| {
            black_box(searcher.search_encoded_patterns(&encoded, &text, k));
        });

        println!("\nIPC Stats for q={}:", num_queries);
        println!(
            "Search: {:.2} | Patterns: {:.2} | Tiling: {:.2}",
            ipc1, ipc2, ipc3
        );
        println!("---------------------------\n");
    }

    group.finish();
}

criterion_group!(benches, bench_searchers);
criterion_main!(benches);
