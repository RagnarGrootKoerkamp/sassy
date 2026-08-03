//! Given a fastx file with equal length (150bp) reads, search them with up to 7 errors across a list of .fna.gz files.
//! This matches the IndelFreeAligner RefSeq bench.

use clap::Parser;
use rand::RngExt;
use rayon::prelude::*;
use sassy::{Searcher, profiles::Iupac};
use std::{
    path::{Path, PathBuf},
    sync::atomic::AtomicU64,
};

#[derive(clap::Parser)]
struct Args {
    /// Text files to search
    text_paths: Vec<PathBuf>,

    /// max edits
    #[clap(short)]
    k: usize,

    // Params for generating random patterns
    /// pattern len
    #[clap(short)]
    m: usize,
    /// #patterns
    #[clap(short)]
    p: usize,

    /// #patterns
    #[clap(short = 'j', default_value_t = 64)]
    threads: usize,
}

fn main() {
    let Args {
        k,
        text_paths,
        m,
        p,
        threads,
    } = Args::parse();

    // let reads = read_path(&patterns_path);

    let mut rng = rand::rng();
    let patterns: Vec<Vec<u8>> = (0..p)
        .map(|_| (0..m).map(|_| b"ACGT"[rng.random_range(0..4)]).collect())
        .collect();

    let searcher = Searcher::<Iupac>::new_rc().without_trace();
    let encoded_patterns = searcher.encode_patterns(&patterns);

    let reading = AtomicU64::new(0);
    let searching = AtomicU64::new(0);
    let input_bp = AtomicU64::new(0);
    let files_done = AtomicU64::new(0);
    let total_matches = AtomicU64::new(0);

    // set parallellism
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    text_paths
        .par_iter()
        // clone searcher per batch of work
        .for_each_with(searcher, |searcher, text_path| {
            let mut num_matches = 0;
            let start = std::time::Instant::now();
            let texts = read_path(text_path);
            let mid = std::time::Instant::now();
            for text in &texts {
                let matches = searcher.search_all_encoded_patterns(&encoded_patterns, text, k);
                num_matches += matches.len();
            }
            let end = std::time::Instant::now();
            reading.fetch_add(
                (mid-start).as_millis() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            searching.fetch_add(
                (end-mid).as_millis() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            let input_bp_for_file: u64 = texts.iter().map(|t| t.len() as u64).sum();
            let input_bp = input_bp.fetch_add(input_bp_for_file, std::sync::atomic::Ordering::Relaxed);
            let done = files_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let total_matches = total_matches.fetch_add(num_matches as u64, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "Done {:>4}/{:>4} [{:>8.3} Gbp / {:7.3} M matches]: {:>8.3} Gbp {:>9} matches, reading: {:>7.3}s, searching: {:>7.3}s",
                done + 1,
                text_paths.len(),
                input_bp as f64 / 1_000_000_000.0,
                total_matches as f64 / 1_000_000.0,
                input_bp_for_file as f64 / 1_000_000_000.0,
                num_matches,
                reading.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0,
                searching.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0,
            );
            if input_bp > 105_000_000_000 {
                eprintln!("Stopping early after 105 Gbp");
                std::process::exit(0);
            }
        });
    eprintln!("Total matches: {}", total_matches.into_inner());
}

fn read_path(patterns_path: &Path) -> Vec<Vec<u8>> {
    let mut records = vec![];
    let Ok(mut reader) = needletail::parse_fastx_file(patterns_path) else {
        return vec![];
    };
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        records.push(record.seq().to_vec());
    }
    records
}
