//! Given a fastx file with short, equal-length barcodes of length <= 64, find all pairwise
//! matches with *semi-global* distance <= k.
//! Or, when a 2nd path is given, find all occurrences of the patterns in the texts.

use clap::Parser;
use sassy::{Searcher, profiles::Iupac};
use std::path::{Path, PathBuf};

#[derive(clap::Parser)]
struct Args {
    #[clap(short)]
    k: usize,

    patterns: PathBuf,
    texts: Option<PathBuf>,
}

fn main() {
    let Args {
        k,
        patterns: patterns_path,
        texts: texts_path,
    } = Args::parse();

    let barcodes = read_path(&patterns_path);

    let texts = read_path(&texts_path.unwrap_or(patterns_path));

    let mut searcher = Searcher::<Iupac>::new_fwd();
    let encoded_barcodes = searcher.encode_patterns(&barcodes);

    let mut num_matches = 0;
    for text in &texts {
        let matches = searcher.search_encoded_patterns(&encoded_barcodes, text, k);
        num_matches += matches.len();
    }
    eprintln!("Total matches: {}", num_matches);
}

fn read_path(patterns_path: &Path) -> Vec<Vec<u8>> {
    let mut records = vec![];
    let mut reader = needletail::parse_fastx_file(patterns_path).unwrap();
    while let Some(record) = reader.next() {
        let record = record.unwrap();
        records.push(record.seq().to_vec());
    }
    records
}
