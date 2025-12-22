use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::random_range;
use sassy::Searcher;
use sassy::profiles::Iupac;

fn bench_searchers(c: &mut Criterion) {
    let query_len = 32;
    let text_len = 100_000;
    let k = 0;

    let text: Vec<u8> = (0..text_len).map(|_| b"ACGT"[random_range(0..4)]).collect();

    let num_queries_list = vec![1, 10, 100, 1_000, 10_000];

    let mut group = c.benchmark_group("search_performance");
    group.sample_size(10); // Otherwise bench takes forever

    for &num_queries in &num_queries_list {
        let queries: Vec<Vec<u8>> = (0..num_queries)
            .map(|_| {
                (0..query_len)
                    .map(|_| b"ACGT"[random_range(0..4)])
                    .collect()
            })
            .collect();

        let total_bytes = text_len * num_queries;
        group.throughput(criterion::Throughput::Bytes(total_bytes as u64));

        {
            let mut searcher = Searcher::<Iupac>::new_fwd();

            group.bench_function(format!("sassy_search/q={}", num_queries), |b| {
                b.iter(|| {
                    for query in &queries {
                        let m = searcher.search(query, &text, k);
                        black_box(m);
                    }
                });
            });
        }

        {
            let mut searcher = Searcher::<Iupac>::new_fwd();

            group.bench_function(format!("sassy_search_patterns/q={}", num_queries), |b| {
                b.iter(|| {
                    let r = searcher.search_patterns(&queries, &text, k);
                    black_box(r);
                });
            });
        }

        {
            let mut searcher = Searcher::<Iupac>::new_fwd();
            let encoded = searcher.encode_patterns(&queries);

            group.bench_function(format!("sassy_pattern_tiling/q={}", num_queries), |b| {
                b.iter(|| {
                    let r = searcher.search_encoded_patterns(&encoded, &text, k);
                    black_box(r);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_searchers);
criterion_main!(benches);
