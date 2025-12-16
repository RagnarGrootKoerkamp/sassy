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
    /// #texts
    #[clap(short)]
    t: usize,

    #[clap(long)]
    pattern: bool,
    #[clap(long)]
    text: bool,
}

fn main() {
    env_logger::init();

    let Args {
        n,
        m,
        k,
        p,
        t,
        pattern,
        text,
    } = Args::parse();

    let mut rng = rand::rng();
    let texts: Vec<Vec<u8>> = (0..t)
        .map(|_| (0..n).map(|_| b"ACGT"[rng.random_range(0..4)]).collect())
        .collect();
    let texts: Vec<&[u8]> = texts.iter().map(|t| t.as_slice()).collect();
    let patterns: Vec<Vec<u8>> = (0..p)
        .map(|_| (0..m).map(|_| b"ACGT"[rng.random_range(0..4)]).collect())
        .collect();

    if !pattern && !text {
        let mut searcher = Searcher::<Iupac>::new_rc();
        for _ in 0..2 {
            let start = std::time::Instant::now();
            let mut cnt = 0;
            for pattern in &patterns {
                for text in &texts {
                    let matches = searcher.search(pattern, &text, k);
                    cnt += matches.len();
                }
            }
            eprintln!("Single    {cnt}: {:?}", start.elapsed());
        }
    }
    if !pattern {
        let mut searcher = Searcher::<Iupac>::new_rc();
        for _ in 0..2 {
            let start = std::time::Instant::now();
            let mut cnt = 0;
            for pattern in &patterns {
                let matches = searcher.search_texts(&pattern, &texts, k);
                cnt += matches.len();
            }
            eprintln!("Text      {cnt}: {:?}", start.elapsed());
        }
    }
    if !text {
        let mut searcher = Searcher::<Iupac>::new_rc();
        for _ in 0..2 {
            let start = std::time::Instant::now();
            let mut cnt = 0;
            for text in &texts {
                let matches = searcher.search_patterns(&patterns, &text, k);
                cnt += matches.len();
            }
            eprintln!("Patterns: {cnt}: {:?}", start.elapsed());
        }
    }
}
