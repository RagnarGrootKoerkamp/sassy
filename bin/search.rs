use crate::input_iterator::{InputIterator, PatternRecord};
use needletail::parse_fastx_file;
use sassy::CachedRev;
use sassy::{
    Match, RcSearchAble, Searcher, Strand,
    profiles::{Ascii, Dna, Iupac, Profile},
};
use std::fs::File;
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Mutex,
};

#[derive(clap::Parser, Clone)]
pub struct SearchArgs {
    // Named args
    /// Pattern to search for (cannot be used with -f).
    #[arg(short = 'p', long, conflicts_with = "pattern_fasta")]
    pattern: Option<String>,

    /// Fasta file with multiple patterns (cannot be used with -p).
    #[arg(short = 'f', long, conflicts_with = "pattern")]
    pattern_fasta: Option<String>,

    /// Report matches up to (and including) this distance threshold.
    #[arg(short)]
    k: usize,

    // Flags
    /// The alphabet to use. DNA=ACTG, or IUPAC=ACTG+NYR...
    #[arg(long, short = 'a', value_enum)]
    alphabet: Alphabet,

    /// Disable reverse complement search (enabled by default for DNA and IUPAC).
    #[arg(long)]
    no_rc: bool,

    // Optional
    /// Number of threads to use. All CPUs by default.
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Output file, otherwise stdout
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    // Positional
    /// Fasta file to search. May be gzipped.
    path: PathBuf,
}

#[derive(clap::ValueEnum, Default, Clone, Copy, PartialEq)]
pub enum Alphabet {
    Ascii,
    #[default]
    Dna,
    Iupac,
}

fn get_patterns(args: &SearchArgs) -> Vec<PatternRecord> {
    if let Some(p) = &args.pattern {
        // Single inline pattern; give it a dummy id
        vec![PatternRecord {
            id: "pattern".to_string(),
            seq: p.as_bytes().to_vec(),
        }]
    } else if let Some(pattern_fasta) = &args.pattern_fasta {
        // Pull all sequences and ids from the FASTA file
        let mut reader = parse_fastx_file(pattern_fasta).expect("valid path/file");
        let mut patterns = Vec::new();
        while let Some(record) = reader.next() {
            let seqrec = record.expect("invalid record");
            let id = String::from_utf8(seqrec.id().to_vec()).unwrap();
            patterns.push(PatternRecord {
                id,
                seq: seqrec.seq().into_owned(),
            });
        }
        patterns
    } else {
        panic!("No --pattern or --pattern-fasta provided!");
    }
}

fn get_output_writer(args: &SearchArgs) -> Box<dyn Write + Send> {
    if let Some(output_path) = &args.output {
        Box::new(BufWriter::new(File::create(output_path).unwrap())) as Box<dyn Write + Send>
    } else {
        Box::new(std::io::stdout()) as Box<dyn Write + Send>
    }
}

fn as_output_line(
    m: &Match,
    pat_id: &str,
    text_id: &str,
    cached_text: &CachedRev<impl AsRef<[u8]>>,
    alphabet: &Alphabet,
) -> String {
    let cost = m.cost;
    let start = m.text_start;
    let end = m.text_end;
    let slice = &cached_text.text.as_ref()[start..end];

    // If we match reverse complement, reverse complement the slice to make it easier to read
    let slice_str = if m.strand == Strand::Rc {
        match alphabet {
            Alphabet::Dna => {
                String::from_utf8_lossy(&<Dna as Profile>::reverse_complement(slice)).into_owned()
            }
            Alphabet::Iupac => {
                String::from_utf8_lossy(&<Iupac as Profile>::reverse_complement(slice)).into_owned()
            }
            Alphabet::Ascii => unreachable!("no rc for ascii"), // Guarded against above
        }
    } else {
        String::from_utf8_lossy(slice).into_owned()
    };

    let cigar = m.cigar.to_string();
    let strand = match m.strand {
        Strand::Fwd => "+",
        Strand::Rc => "-",
    };
    format!("{pat_id}\t{text_id}\t{cost}\t{strand}\t{start}\t{end}\t{slice_str}\t{cigar}\n")
}

// To create a fixed type, cost of match should be neglible
enum SearchWrapper {
    Ascii(Box<Searcher<Ascii>>),
    Dna(Box<Searcher<Dna>>),
    Iupac(Box<Searcher<Iupac>>),
}

impl SearchWrapper {
    fn search<I: RcSearchAble>(&mut self, pattern: &[u8], input: &I, k: usize) -> Vec<Match> {
        match self {
            SearchWrapper::Ascii(s) => s.search(pattern, input, k),
            SearchWrapper::Dna(s) => s.search(pattern, input, k),
            SearchWrapper::Iupac(s) => s.search(pattern, input, k),
        }
    }
}

pub fn search(args: &SearchArgs) {
    // Get queries based on `pattern` or `pattern_fasta`
    let patterns = get_patterns(args);
    assert!(!patterns.is_empty(), "No pattern sequences found");

    // Create output writer, stdout by default (or user provided), and share it safely across threads
    let output_writer = &Mutex::new(get_output_writer(args));

    // Write header
    let header = format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        "pat_id", "text_id", "cost", "strand", "start", "end", "match_region", "cigar"
    );
    output_writer
        .lock()
        .unwrap()
        .write_all(header.as_bytes())
        .unwrap();

    let k = args.k;
    let rc = (args.alphabet == Alphabet::Dna || args.alphabet == Alphabet::Iupac) && !args.no_rc;

    let num_threads = args.threads.unwrap_or_else(num_cpus::get);
    let task_iterator = &InputIterator::new(&args.path, &patterns, None, rc);
    std::thread::scope(|s| {
        for _ in 0..num_threads {
            s.spawn(move || {
                // Each thread has own searcher here
                let mut searcher: SearchWrapper = match args.alphabet {
                    Alphabet::Ascii => {
                        SearchWrapper::Ascii(Box::new(Searcher::<Ascii>::new(false, None)))
                    }
                    Alphabet::Dna => SearchWrapper::Dna(Box::new(Searcher::<Dna>::new(rc, None))),
                    Alphabet::Iupac => {
                        SearchWrapper::Iupac(Box::new(Searcher::<Iupac>::new(rc, None)))
                    }
                };

                while let Some(batch) = task_iterator.next_batch() {
                    for item in batch {
                        let matches = searcher.search(&item.pattern.seq, &item.text.seq, k);
                        let mut writer_guard = output_writer.lock().unwrap();
                        for m in matches {
                            let line = as_output_line(
                                &m,
                                &item.pattern.id,
                                &item.text.id,
                                &item.text.seq,
                                &args.alphabet,
                            );
                            writer_guard.write_all(line.as_bytes()).unwrap();
                        }
                        drop(writer_guard);
                    }
                }
            });
        }
    });
}

#[cfg(test)]
mod test {

    use super::*;
    use rand::Rng;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn random_dna_string(len: usize) -> Vec<u8> {
        let mut rng = rand::rng();
        (0..len).map(|_| b"ACGT"[rng.random_range(0..4)]).collect()
    }

    fn rc(dna: &[u8]) -> Vec<u8> {
        dna.iter()
            .rev()
            .map(|c| match c {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                b'T' => b'A',
                _ => panic!("Invalid DNA character"),
            })
            .collect()
    }

    #[test]
    fn end_to_end_search() {
        eprintln!("WARNING: Run this test with -- --nocapture to see the output");
        // Create file at data/test.fasta, with two fwd, one reverse complement match
        // insert at 10, 50, and 100
        let dna = random_dna_string(1000);
        let mut text = dna.clone();
        let query = b"TAGCTAGAC";
        text.splice(10..10, query.iter().copied());
        text.splice(50..50, query.iter().copied());
        text.splice(100..100, rc(query).iter().copied());
        // Write to file
        let mut tmp_file = NamedTempFile::new().expect("Failed to create tmp file");
        writeln!(tmp_file, ">test").unwrap();
        writeln!(tmp_file, "{}", String::from_utf8_lossy(&text)).unwrap();

        let mut args: SearchArgs = SearchArgs {
            pattern: Some(String::from_utf8(query.to_vec()).unwrap()),
            k: 0,
            path: tmp_file.path().to_path_buf(),
            alphabet: Alphabet::Dna,
            no_rc: true,
            threads: Some(1),
            pattern_fasta: None,
            output: None,
        };

        // FIXME: capture output and assert or write to tmp_file for easy check
        // anyway for now run with -- --nocapture and check for 10,50,100
        println!("Search without RC");
        search(&args);

        println!("Search with RC");
        args.no_rc = false;
        search(&args);

        drop(tmp_file)
    }
}
