use crate::input_iterator::{InputIterator, PatternRecord};
use sassy::{
    RcSearchAble, Searcher, Strand,
    profiles::{Iupac, Profile},
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Mutex,
};

#[derive(clap::Parser)]
pub struct CrisprArgs {
    // Named args
    /// Path to file with guide sequences (including PAM)
    #[arg(long, short = 'g')]
    guide: String,

    /// Report matches up to (and including) this distance threshold (excluding PAM).
    #[arg(short, long)]
    k: usize,

    // optional
    /// Output file, otherwise stdout.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Allow at most max_n_frac of N bases in the target sequence. Values must be in the
    /// range [0. 1].  A value of 0 will allow only hits where the target sequence contains no
    /// Ns. A value of 0.1-0.2 will allow for matches that include a small number of Ns.
    #[arg(long)]
    max_n_frac: f32,

    /// Number of threads to use. All CPUs by default.
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    // Flags
    /// Allow edits in PAM sequence.
    #[arg(long)]
    allow_pam_edits: bool,

    /// Disable reverse complement search.
    #[arg(long)]
    no_rc: bool,

    /// Fasta file to search. May be gzipped.
    path: PathBuf,
}

fn get_output_writer(args: &CrisprArgs) -> Box<dyn Write + Send> {
    if let Some(output_path) = &args.output {
        Box::new(BufWriter::new(File::create(output_path).unwrap())) as Box<dyn Write + Send>
    } else {
        Box::new(std::io::stdout()) as Box<dyn Write + Send>
    }
}

fn check_n_frac(max_n_frac: f32, match_slice: &[u8]) -> bool {
    let n_count = match_slice
        .iter()
        .filter(|c| (**c & 0xDF) == b'N') // Convert to uppercase check against N
        .count() as f32;
    let n_frac = n_count / match_slice.len() as f32;
    n_frac <= max_n_frac
}

fn print_and_check_params(args: &CrisprArgs, guide_sequences: &[Vec<u8>]) -> (String, f32) {
    // Check if n frac is within valid range
    if !(0.0..=1.0).contains(&args.max_n_frac) {
        eprintln!("[N-chars] Error: max_n_frac must be between 0 and 1.0");
        std::process::exit(1);
    }

    // If no guide sequences throw error
    if guide_sequences.is_empty() {
        eprintln!(
            "[PAM] Error: No guide sequences provided, please check your input file (one guide sequence per line)"
        );
        std::process::exit(1);
    }

    // We have at least one guide sequence, extract the PAM sequence
    let pam = if !guide_sequences.is_empty() {
        let guide = &guide_sequences[0];
        let pam = &guide[guide.len() - 3..];
        println!("[PAM] Sequence: [{}]", String::from_utf8_lossy(pam));
        println!(
            "[PAM] If the above PAM is incorrect, please make sure that the guide sequence ENDs with the PAM-sequence, i.e. XXXXXGGN (not it's reverse complement)"
        );
        pam
    } else {
        unreachable!("No guide sequences provided");
    };

    // If we have multiple guide sequences, ensure they have the same PAM sequence.
    // not per se a requirement for the code to work but now we define that as a fixed PAM in the closure below
    if guide_sequences.len() > 1 {
        for guide_sequence in guide_sequences {
            let guide_pam = &guide_sequence[guide_sequence.len() - 3..];
            if pam != guide_pam {
                eprintln!(
                    "[PAM] One of the guide sequences has a PAM different than the provided PAM"
                );
                eprintln!(
                    "[PAM] provided PAM {}, detected PAM {}",
                    String::from_utf8_lossy(pam),
                    String::from_utf8_lossy(guide_pam)
                );
                std::process::exit(1);
            }
        }
    }

    println!("[PAM] PAM used to filter: {}", String::from_utf8_lossy(pam));
    println!("[PAM] Edits in PAM are allowed: {}", args.allow_pam_edits);
    println!(
        "[N-chars] Allowing up to {}% N characters",
        args.max_n_frac * 100.0
    );

    (String::from_utf8_lossy(pam).into_owned(), args.max_n_frac)
}

pub fn read_guide_sequences(path: &str) -> Vec<Vec<u8>> {
    let file = File::open(path).expect("Failed to open guide file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|l| l.unwrap().as_bytes().to_vec())
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|seq| !seq.is_empty())
        .collect()
}

pub fn matching_seq<P: Profile>(seq1: &[u8], seq2: &[u8]) -> bool {
    for (c1, c2) in seq1.iter().zip(seq2.iter()) {
        if !P::is_match(*c1, *c2) {
            return false;
        }
    }
    true
}

pub fn crispr(args: &CrisprArgs) {
    let guide_sequences = read_guide_sequences(&args.guide);
    println!("[GUIDES] Found {} guides", guide_sequences.len());

    if guide_sequences.is_empty() {
        return;
    }

    // Read the first record from the FASTA file for benchmarking
    let writer = &Mutex::new(get_output_writer(args));

    // Write header
    let header = format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        "guide", "text_id", "cost", "strand", "start", "end", "match_region", "cigar"
    );
    writer.lock().unwrap().write_all(header.as_bytes()).unwrap();

    let (pam, max_n_frac) = print_and_check_params(args, &guide_sequences);
    let pam = pam.as_bytes();
    let pam_compl = Iupac::complement(pam);
    let pam_compl = pam_compl.as_slice();

    let total_found = AtomicUsize::new(0);

    let num_threads = args.threads.unwrap_or_else(num_cpus::get);
    println!("[Threads] Using {num_threads} threads");

    // Build queries for RecordIterator (one per guide sequence)
    let queries: Vec<PatternRecord> = guide_sequences
        .iter()
        .enumerate()
        .map(|(i, seq)| PatternRecord {
            id: format!("guide_{}", i + 1),
            seq: seq.clone(),
        })
        .collect();

    // Shared iterator that pairs each query with every FASTA record in a batched fashion
    let task_iter = InputIterator::new(&args.path, &queries, None, true);

    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..num_threads {
            scope.spawn(|| {
                // Searcher, IUPAC and always reverse complement
                let mut searcher = Searcher::<Iupac>::new(!args.no_rc, None);

                let filter_fn = |_q: &[u8], text_up_to_end: &[u8], strand: Strand| {
                    let pam_slice = &text_up_to_end[text_up_to_end.len() - 3..];
                    if strand == Strand::Fwd {
                        matching_seq::<Iupac>(pam_slice, pam)
                    } else {
                        matching_seq::<Iupac>(pam_slice, pam_compl)
                    }
                };

                while let Some(batch) = task_iter.next_batch() {
                    for item in batch {
                        let guide_sequence = &item.pattern.seq;
                        let guide_string = String::from_utf8_lossy(guide_sequence);
                        let id_text = &item.text;

                        let id = &id_text.id;
                        let text = &id_text.seq;

                        let matches = if !args.allow_pam_edits {
                            searcher.search_with_fn(guide_sequence, text, args.k, true, filter_fn)
                        } else {
                            searcher.search_all(guide_sequence, text, args.k)
                        };

                        total_found.fetch_add(matches.len(), Ordering::Relaxed);

                        let mut writer_guard = writer.lock().unwrap();

                        for m in matches {
                            let start = m.text_start;
                            let end = m.text_end;
                            let text = text.text();
                            let slice = &text.as_ref()[start..end];

                            // Check if satisfies user max N cut off
                            let n_ok = if max_n_frac < 100.0 {
                                check_n_frac(max_n_frac, slice)
                            } else {
                                true
                            };

                            if !n_ok {
                                continue;
                            }

                            total_found.fetch_add(1, Ordering::Relaxed);

                            let match_region = if m.strand == Strand::Rc {
                                let rc = <Iupac as Profile>::reverse_complement(slice);
                                String::from_utf8_lossy(&rc).into_owned()
                            } else {
                                String::from_utf8_lossy(slice).into_owned()
                            };
                            let cost = m.cost;
                            let cigar = m.cigar.to_string();
                            let strand = match m.strand {
                                Strand::Fwd => "+",
                                Strand::Rc => "-",
                            };
                            writeln!(
                                writer_guard,
                                "{guide_string}\t{id}\t{cost}\t{strand}\t{start}\t{end}\t{match_region}\t{cigar}"
                            )
                            .unwrap();
                        }
                        drop(writer_guard);
                    }
                }
            });
        }
    });

    println!("\nSummary");
    println!(
        "  Total targets found:   {}",
        total_found.load(Ordering::Relaxed)
    );
    println!("  Time taken: {:?}", start.elapsed());
}
