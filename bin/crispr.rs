use crate::input_iterator::{BufferedInputIterator, InputIterator, PatternRecord};
use sassy::{
    CachedRev, EncodedPatterns, Match, RcSearchAble, Searcher, Strand,
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

    /// The length of the PAM.
    #[arg(long, default_value_t = 3)]
    pam_length: usize,

    // Flags
    /// Allow edits in PAM sequence.
    #[arg(long)]
    allow_pam_edits: bool,

    /// Disable reverse complement search.
    #[arg(long)]
    no_rc: bool,

    /// Use the v2 encoded search. All guides must have the same length.
    #[arg(long)]
    v2: bool,

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
        let pam = &guide[guide.len() - args.pam_length..];
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
            let guide_pam = &guide_sequence[guide_sequence.len() - args.pam_length..];
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

    let guide_len = guide_sequences[0].len();
    let equal_length = guide_sequences.iter().all(|guide| guide.len() == guide_len);
    let v2 = args.v2 || (equal_length && guide_len <= 64);
    assert!(
        !v2 || equal_length,
        "All guide sequences must have the same length for --v2"
    );
    if v2 && !args.v2 {
        eprintln!("All guides have equal length; using the --v2 backend.");
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
    let paths = vec![args.path.clone()];
    let input_batch_iter = InputIterator::new(
        &paths,
        &queries,
        None,
        None,
        !args.no_rc,
        guide_sequences.iter().map(Vec::len).max().unwrap_or(0) + args.k,
        false,
    );
    let input_reader = &BufferedInputIterator::new(num_threads);
    let start = Instant::now();
    std::thread::scope(|s| {
        input_reader.start_reader(s, &input_batch_iter);

        let total_found = &total_found;
        for _ in 0..num_threads {
            s.spawn(move || {
                let mut searcher = if args.no_rc {
                    Searcher::<Iupac>::new_fwd().with_max_n_frac(max_n_frac)
                } else {
                    Searcher::<Iupac>::new_rc().with_max_n_frac(max_n_frac)
                };

                let matches_pam = |text_up_to_end: &[u8], strand: Strand| {
                    if text_up_to_end.len() < args.pam_length {
                        return false;
                    }
                    let pam_slice = &text_up_to_end[text_up_to_end.len() - args.pam_length..];
                    if strand == Strand::Fwd {
                        matching_seq::<Iupac>(pam_slice, pam)
                    } else {
                        matching_seq::<Iupac>(pam_slice, pam_compl)
                    }
                };

                while let Some((_batch_id, batch)) = input_reader.next_batch() {
                    let encoded = v2.then(|| {
                        let patterns: Vec<_> = batch.1.iter().map(|p| p.seq.clone()).collect();
                        searcher.encode_patterns(&patterns)
                    });
                    for text in &*batch.2 {
                        let sequence = &text.seq;
                        let mut v2_matches = encoded.as_ref().map(|encoded| {
                            search_v2(
                                &mut searcher, encoded, sequence,
                                batch.1.len(), args.k,
                                |prefix, strand| args.allow_pam_edits || matches_pam(prefix, strand),
                            )
                        });
                        for (idx, pattern) in batch.1.iter().enumerate() {
                            let guide_sequence = &pattern.seq;
                            let guide_string = String::from_utf8_lossy(guide_sequence);
                            let id_text = &text;

                            let id = &id_text.id;

                            let mut matches = if let Some(matches) = &mut v2_matches {
                                std::mem::take(&mut matches[idx])
                            } else if !args.allow_pam_edits {
                                searcher.search_with_fn(guide_sequence, sequence, args.k, true, |_, prefix, strand| {
                                    matches_pam(prefix, strand)
                                })
                            } else {
                                searcher.search_all(guide_sequence, sequence, args.k)
                            };
                            matches.retain(|m| text.owns_match(m));
                            if matches.is_empty() {
                                continue;
                            }

                            total_found.fetch_add(matches.len(), Ordering::Relaxed);

                            let mut writer_guard = writer.lock().unwrap();

                            for m in matches {
                                let start = m.text_start;
                                let end = m.text_end;
                                let text_bytes = sequence.text();
                                let slice = &text_bytes.as_ref()[start..end];

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
                                    "{guide_string}\t{id}\t{cost}\t{strand}\t{}\t{}\t{match_region}\t{cigar}",
                                    text.record_offset + start,
                                    text.record_offset + end,
                                )
                                    .unwrap();
                            }
                            drop(writer_guard);
                        }
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

pub(super) fn search_v2(
    searcher: &mut Searcher<Iupac>,
    encoded: &EncodedPatterns<Iupac>,
    sequence: &CachedRev<Vec<u8>>,
    num_guides: usize,
    k: usize,
    accepts_endpoint: impl Fn(&[u8], Strand) -> bool,
) -> Vec<Vec<Match>> {
    let mut matches = vec![Vec::new(); num_guides];
    let len = sequence.text.len();
    if len == 0 {
        return matches;
    }

    for m in searcher.search_all_encoded_patterns(encoded, sequence, k) {
        let prefix = match m.strand {
            Strand::Fwd => &sequence.text[..m.text_end],
            Strand::Rc => &sequence.rev.as_ref().unwrap()[..len - m.text_start],
        };
        if accepts_endpoint(prefix, m.strand) {
            matches[m.pattern_idx].push(m.clone());
        }
    }
    matches
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    fn mock_guide_txt_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let g1_txt = format!("TAGCATCAGCTACGNGG\n");
        file.write_all(g1_txt.as_bytes()).unwrap();
        file
    }

    fn mock_target_fasta() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let t1_fasta = format!(">exact\nTAGCATCAGCTACGAGG\n");
        let t1_pam_mutated_fasta = format!(">pam_mutated\nTAGCATCAGCTACGACG\n"); // C instead of G in PAM
        let t1_n_frac_failure_fasta = format!(">n_frac\nTNNNATCAGCTACGAGG\n"); // 3 N's out of 17 total bases
        file.write_all(t1_fasta.as_bytes()).unwrap();
        file.write_all(t1_pam_mutated_fasta.as_bytes()).unwrap();
        file.write_all(t1_n_frac_failure_fasta.as_bytes()).unwrap();
        file
    }

    #[derive(Default)]
    struct Counts {
        exact: usize,
        pam_mutated: usize,
        n_frac: usize,
    }

    fn get_counts_from_output(tmp_path: &PathBuf) -> Counts {
        let output = fs::read_to_string(tmp_path).unwrap();
        let mut counts = Counts::default();
        for line in output.lines().skip(1) {
            let parts: Vec<&str> = line.split('\t').collect();
            let target = parts[1];
            match target {
                "exact" => counts.exact += 1,
                "pam_mutated" => counts.pam_mutated += 1,
                "n_frac" => counts.n_frac += 1,
                _ => panic!("Unknown target: {target}"),
            }
        }
        counts
    }

    #[test]
    fn test_crispr() {
        let tmp_output = NamedTempFile::new().unwrap();
        let guide_file = mock_guide_txt_file();
        let target_file: NamedTempFile = mock_target_fasta();
        let mut args = CrisprArgs {
            guide: guide_file.path().to_string_lossy().to_string(),
            k: 1,
            output: Some(tmp_output.path().to_path_buf()),
            max_n_frac: 1.0,
            threads: None,
            pam_length: 3,
            allow_pam_edits: true,
            no_rc: false,
            v2: false,
            path: target_file.path().to_path_buf(),
        };
        crispr(&args);
        let found_counts = get_counts_from_output(&tmp_output.path().to_path_buf());
        assert_eq!(found_counts.exact, 2);
        assert_eq!(found_counts.pam_mutated, 1);
        assert_eq!(found_counts.n_frac, 2);

        // If we now disable PAM mutations we should have zero for pam_mutated
        // exact also changes as now it cant move on the last char anymore
        args.allow_pam_edits = false;
        crispr(&args);
        let found_counts = get_counts_from_output(&tmp_output.path().to_path_buf());
        assert_eq!(found_counts.exact, 1);
        assert_eq!(found_counts.pam_mutated, 0);
        assert_eq!(found_counts.n_frac, 1);

        // We have 3 N's / 17 = 0.176..
        let expected_treshold = (3.0 / 17.0) as f32;
        // Slightly above threshold should still pass
        args.max_n_frac = expected_treshold + 0.01;
        crispr(&args);
        let found_counts = get_counts_from_output(&tmp_output.path().to_path_buf());
        assert_eq!(found_counts.n_frac, 1);
        // Setting slightly below the threshold should drop the match
        args.max_n_frac = expected_treshold - 0.01;
        crispr(&args);
        let found_counts = get_counts_from_output(&tmp_output.path().to_path_buf());
        assert_eq!(found_counts.n_frac, 0);
    }

    #[test]
    fn v2_crispr_maps_reversed_text_hits_back_to_forward_coordinates() {
        let mut guide_file = NamedTempFile::new().unwrap();
        let guide = "TAGCATCAGCTACGNGG";
        let second_guide = "CTTCTTCTTCTTCTNGG";
        writeln!(guide_file, "{guide}\n{second_guide}").unwrap();
        let region = "TAGCATCAGCTACGAGG";
        let reverse_region = Iupac::reverse_complement(region.as_bytes());
        let reverse_region = std::str::from_utf8(&reverse_region).unwrap();
        let mut target_file = NamedTempFile::new().unwrap();
        writeln!(
            target_file,
            ">forward\n{region}\n>reverse\nAAAAA{reverse_region}TT"
        )
        .unwrap();
        let output = NamedTempFile::new().unwrap();

        let mut args = CrisprArgs {
            guide: guide_file.path().to_string_lossy().into_owned(),
            k: 0,
            output: Some(output.path().to_path_buf()),
            max_n_frac: 0.0,
            threads: Some(1),
            pam_length: 3,
            allow_pam_edits: false,
            no_rc: false,
            v2: true,
            path: target_file.path().to_path_buf(),
        };
        crispr(&args);
        let results = fs::read_to_string(output.path()).unwrap();
        let len = region.len();
        let forward = format!("{guide}\tforward\t0\t+\t0\t{len}\t{region}\t{len}=");
        let reverse = format!("{guide}\treverse\t0\t-\t5\t{}\t{region}\t{len}=", 5 + len);
        assert!(results.lines().any(|line| line == forward));
        assert!(results.lines().any(|line| line == reverse));

        args.no_rc = true;
        crispr(&args);
        let results = fs::read_to_string(output.path()).unwrap();
        assert!(results.lines().any(|line| line == forward));
        assert!(results.lines().skip(1).all(|line| !line.contains("\t-\t")));
    }
}
