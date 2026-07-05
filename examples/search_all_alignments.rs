//! A quick example on how to use `search_all_alignments`, mostly to benchmark its performance.
use std::path::Path;

use sassy::{Searcher, profiles::Iupac};

fn main() {
    let text = read_path(Path::new(&"human-genome.fa"));
    let pattern = b"ACTCTACGACTACTCAGATA";
    let k = 3;

    let time = std::time::Instant::now();
    let alignments = Searcher::<Iupac>::new_rc()
        .with_max_n_frac(0.5)
        .search_all_alignments(pattern, &text, k);
    eprintln!("Search time: {:?}", time.elapsed());
    eprintln!("end pos: {}", alignments.len());
    eprintln!(
        "total alignments: {}",
        alignments.iter().map(|a| a.len()).sum::<usize>()
    );
}

fn read_path(patterns_path: &Path) -> Vec<u8> {
    let mut text = vec![];
    let mut reader = needletail::parse_fastx_file(patterns_path).unwrap();
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        text.extend_from_slice(&*record.seq());
    }
    text
}
