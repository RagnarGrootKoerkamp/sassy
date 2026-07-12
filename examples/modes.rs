use std::sync::atomic::Ordering::Relaxed;

use clap::Parser;
use rand::RngExt;
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
    #[clap(long)]
    gpu: bool,
}

const GPU_REPEATS: usize = 4;
const REPEATS: usize = 1;

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
        gpu,
    } = Args::parse();

    let mut rng = rand::rng();
    let texts: Vec<Vec<u8>> = (0..t)
        .map(|_| (0..n).map(|_| b"ACGT"[rng.random_range(0..4)]).collect())
        .collect();
    let patterns: Vec<Vec<u8>> = (0..p)
        .map(|_| (0..m).map(|_| b"ACGT"[rng.random_range(0..4)]).collect())
        .collect();

    let nn = n * t;
    let mm = m * p;
    let pat_bp = p * t * n;

    {
        sassy::USE_WGPU.store(true, Relaxed);
        let mut searcher = Searcher::<Iupac>::new_fwd().without_trace();
        for _ in 0..GPU_REPEATS {
            let start = std::time::Instant::now();
            let mut cnt = 0;
            for text in &texts {
                let pats = searcher.encode_patterns(&patterns);
                let matches = searcher.search_encoded_patterns(&pats, text, k);
                cnt += matches.len();
            }
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "GPU       {cnt}: {:.3?} s {:.3?} ns/pat/bp",
                elapsed,
                elapsed / pat_bp as f64 * 1e9
            );
        }
    }
    {
        sassy::USE_WGPU.store(false, Relaxed);
        let mut searcher = Searcher::<Iupac>::new_fwd().without_trace();
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            let mut cnt = 0;
            for text in &texts {
                let pats = searcher.encode_patterns(&patterns);
                let matches = searcher.search_encoded_patterns(&pats, text, k);
                cnt += matches.len();
            }
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "v2        {cnt}: {:.3?} s {:.3?} ns/pat/bp",
                elapsed,
                elapsed / pat_bp as f64 * 1e9
            );
        }
    }

    {
        let mut searcher = Searcher::<Iupac>::new_fwd().without_trace();
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            let mut cnt = 0;
            for pattern in &patterns {
                for text in &texts {
                    let matches = searcher.search(pattern, &text, k);
                    cnt += matches.len();
                }
            }
            eprintln!("Single    {cnt}: {:.1?}", start.elapsed());
        }
    }
    {
        let mut searcher = Searcher::<Iupac>::new_fwd().without_trace();
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            let mut cnt = 0;
            for pattern in &patterns {
                let matches = searcher.search_texts(&pattern, &texts, k);
                cnt += matches.len();
            }
            eprintln!("Text      {cnt}: {:.1?}", start.elapsed());
        }
    }
    {
        let mut searcher = Searcher::<Iupac>::new_fwd().without_trace();
        for _ in 0..REPEATS {
            let start = std::time::Instant::now();
            let mut cnt = 0;
            for text in &texts {
                let matches = searcher.search_patterns(&patterns, &text, k);
                cnt += matches.len();
            }
            eprintln!("Patterns: {cnt}: {:.1?}", start.elapsed());
        }
    }
}
