use clap::Parser;
use rand::Rng;
use sassy::{Searcher, profiles::Iupac};

#[derive(clap::Parser)]
struct Args {
    /// text len
    #[clap(short)]
    n: usize,
    /// pattern len
    #[clap(short)]
    m: usize,
    /// edits
    #[clap(short)]
    k: usize,
    /// #patterns
    #[clap(short)]
    p: usize,
}

fn main() {
    env_logger::init();

    let Args { n, m, k, p } = Args::parse();

    let mut rng = rand::rng();
    let text: Vec<u8> = (0..n).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();
    let patterns: Vec<Vec<u8>> = (0..p)
        .map(|_| (0..m).map(|_| b"ACGT"[rng.random_range(0..4)]).collect())
        .collect();

    {
        let start = std::time::Instant::now();
        let mut cnt = 0;
        let mut searcher = Searcher::<Iupac>::new_rc();
        for pattern in &patterns {
            let matches = searcher.search(pattern, &text, k);
            // eprintln!("matches: {matches:?}");
            cnt += matches.len();
        }
        eprintln!("Single    {cnt}: {:?}", start.elapsed());
    }
    {
        let start = std::time::Instant::now();
        let mut cnt = 0;
        let mut searcher = Searcher::<Iupac>::new_rc();
        let matches = searcher.search_patterns(&patterns, &text, k, None);
        // eprintln!("matches: {matches:?}");
        cnt += matches.len();
        eprintln!("Patterns: {cnt}: {:?}", start.elapsed());
    }
}
