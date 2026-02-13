use std::{
    collections::VecDeque,
    fs::File,
    io::{BufRead, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use colored::Colorize;
use either::Either;
use needletail::parse_fastx_file;
use sassy::{
    Match, Searcher, Strand,
    pretty_print::{PrettyPrintDirection, PrettyPrintStyle},
    profiles::{Dna, Iupac, Profile},
};

use crate::{
    crispr::check_n_frac,
    input_iterator::{InputIterator, PatternRecord, TextBatch, TextRecord},
};

// TODO: Support ASCII alphabet.
#[derive(clap::ValueEnum, Default, Clone, Copy, PartialEq)]
enum Alphabet {
    Dna,
    #[default]
    Iupac,
}

#[derive(clap::Parser, Clone)]
pub struct BaseArgs {
    // Named args
    /// Pattern to search for (cannot be used with -f).
    #[arg(
        short = 'p',
        long,
        conflicts_with = "pattern_fasta",
        conflicts_with = "pattern_file"
    )]
    pattern: Option<String>,

    /// Search each line of the file (cannot be used with -p).
    #[arg(
        short = 'l',
        long,
        conflicts_with = "pattern",
        conflicts_with = "pattern_fasta"
    )]
    pattern_file: Option<PathBuf>,

    /// Search each record in the file (cannot be used with -p).
    #[arg(
        short = 'f',
        long,
        conflicts_with = "pattern",
        conflicts_with = "pattern_file"
    )]
    pattern_fasta: Option<PathBuf>,

    /// Number of patterns to process inside each batch. Default 64.
    #[arg(long, requires = "pattern_file")]
    pattern_batch_size: Option<usize>,

    /// Report matches up to (and including) this distance threshold.
    #[arg(short)]
    k: usize,

    /// The alphabet to use. DNA=ACTG, or default IUPAC=ACTG+NYR....
    ///
    /// ASCII is not yet supported.
    #[arg(
        long,
        short = 'a',
        default_value_t = Alphabet::Iupac,
        value_enum
    )]
    alphabet: Alphabet,

    /// Enable overhang alignment.
    #[arg(long)]
    overhang: Option<f32>,

    /// Disable reverse complement search.
    #[arg(long)]
    no_rc: bool,

    /// Allow at most max_n_frac of N bases in the target sequence. Values must be in the
    /// range [0, 1].  A value of 0 will allow only hits where the target sequence contains no
    /// Ns. A value of 0.1-0.2 will allow for matches that include a small number of Ns.
    #[arg(long, default_value_t = 0.2)]
    max_n_frac: f32,

    /// Number of threads to use. All CPUs by default.
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Only report non-matching records. Only applies to `filter` output.
    #[arg(short = 'v', long)]
    invert: bool,

    // Positional
    /// Input Fastx files. May be gzipped.
    paths: Vec<PathBuf>,
}

#[derive(clap::Parser, Clone)]
pub struct GrepArgs {
    #[command(flatten)]
    base: BaseArgs,

    /// Number of characters (or lines, for ASCII) to print before/after each match.
    #[arg(short = 'C', long, default_value_t = 20)]
    context: usize,

    /// TSV output file to write all matches. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    search: Option<PathBuf>,

    /// Filtered output file. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    filter: Option<PathBuf>,
}

#[derive(clap::Parser, Clone)]
pub struct SearchArgs {
    #[command(flatten)]
    base: BaseArgs,

    /// Filtered output file. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    filter: Option<PathBuf>,
}

#[derive(clap::Parser, Clone)]
pub struct FilterArgs {
    #[command(flatten)]
    base: BaseArgs,

    /// TSV output file to write all matches. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    search: Option<PathBuf>,
}

struct Args {
    base: BaseArgs,

    context: usize,

    grep: bool,
    search: Option<PathBuf>,
    filter: Option<PathBuf>,
}

impl GrepArgs {
    pub fn run(self) {
        let GrepArgs {
            base,
            context,
            search,
            filter,
        } = self;
        Args {
            base,
            context,
            grep: true,
            search,
            filter,
        }
        .run()
    }
}

impl SearchArgs {
    pub fn run(self) {
        let SearchArgs { base, filter } = self;
        Args {
            base,
            context: 0,
            grep: false,
            search: Some(PathBuf::from("")),
            filter,
        }
        .run()
    }
}

impl FilterArgs {
    pub fn run(self) {
        let FilterArgs { base, search } = self;
        Args {
            base,
            context: 0,
            grep: false,
            search,
            filter: Some(PathBuf::from("")),
        }
        .run()
    }
}

impl Args {
    fn run(mut self) {
        if self.base.invert && self.filter.is_none() {
            eprintln!(
                "{}",
                "Warning: --invert/-v has no effect without --filter."
                    .red()
                    .bold()
            );
        }

        if self.base.paths.is_empty() {
            self.base.paths = vec![PathBuf::from("")];
        }
        let args = &self;

        let patterns = args.get_patterns();
        assert!(!patterns.is_empty(), "No pattern sequences found");

        let k = args.base.k;
        let rc = (args.base.alphabet == Alphabet::Dna || args.base.alphabet == Alphabet::Iupac)
            && !args.base.no_rc;

        let num_threads = args.base.threads.unwrap_or_else(num_cpus::get);
        let task_iterator = &InputIterator::new(
            &args.base.paths,
            &patterns,
            None,
            args.base.pattern_batch_size,
            rc,
        );

        let output = Mutex::new((
            0,
            VecDeque::<Option<Vec<(&Path, (TextBatch, usize), Vec<(&PatternRecord, Match)>)>>>::new(
            ),
        ));
        let global_histogram = Mutex::new(vec![0usize; k + 1]);

        // Grep writer is stderr
        // Ensure colours are always used.
        colored::control::set_override(true);

        let mut filter_to_stdout = false;
        let mut matches_to_stdout = false;

        // Filter writer
        let filter_writer = self.filter.as_ref().map(|path| {
            let writer = if path != "" && path != "-" {
                Box::new(std::fs::File::create(path).expect("create output file"))
                    as Box<dyn std::io::Write + Send>
            } else {
                filter_to_stdout = true;
                Box::new(std::io::stdout()) as Box<dyn std::io::Write + Send>
            };
            Mutex::new(writer)
        });

        // Match writer
        let match_writer = args.search.as_ref().map(|path| {
            let mut writer = if path != "" && path != "-" {
                Box::new(std::fs::File::create(path).expect("create matches file"))
                    as Box<dyn std::io::Write + Send>
            } else {
                matches_to_stdout = true;
                Box::new(std::io::stdout()) as Box<dyn std::io::Write + Send>
            };

            if matches_to_stdout && filter_to_stdout {
                eprintln!(
                    "{}",
                    "NOTE: Writing both filtered records *and* matching locations to stdout."
                        .red()
                        .bold()
                );
            }

            let header = format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                "pat_id", "text_id", "cost", "strand", "start", "end", "match_region", "cigar"
            );
            write!(writer, "{header}").unwrap();

            Mutex::new(writer)
        });

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let output = &output;
                let global_histogram = &global_histogram;
                let filter_writer = filter_writer.as_ref();
                let match_writer = match_writer.as_ref();
                s.spawn(move || {
                    // Each thread has own searcher here
                    let mut searcher = match &args.base.alphabet {
                        Alphabet::Dna => Either::Left(Searcher::<Dna>::new(rc, args.base.overhang)),
                        Alphabet::Iupac => {
                            Either::Right(Searcher::<Iupac>::new(rc, args.base.overhang))
                        }
                    };

                    let mut results: Vec<(
                        &Path,
                        (TextBatch, usize),
                        Vec<(&PatternRecord, Match)>,
                    )> = vec![];
                    let mut local_histogram = vec![0usize; k + 1];

                    while let Some((batch_id, batch)) = task_iterator.next_batch() {
                        let path = batch.0;
                        for (i, text) in batch.2.iter().enumerate() {
                            let mut batch_matches = vec![];
                            for pattern in batch.1 {
                                let record_matches = match &mut searcher {
                                    Either::Left(s) => s.search(&pattern.seq, &text.seq, k),
                                    Either::Right(s) => s.search(&pattern.seq, &text.seq, k),
                                };
                                batch_matches.extend(
                                    record_matches
                                        .into_iter()
                                        .filter(|m| {
                                            check_n_frac(
                                                args.base.max_n_frac,
                                                &text.seq.text[m.text_start..m.text_end],
                                            )
                                        })
                                        .map(|m| (pattern, m)),
                                );
                            }

                            if batch_matches.is_empty() {
                                continue;
                            }

                            for (_pat, m) in &batch_matches {
                                local_histogram[m.cost as usize] += 1;
                            }
                            results.push((path, (batch.2.clone(), i), batch_matches));
                        }

                        let (next_batch_id, output_buf) = &mut *output.lock().unwrap();
                        let mut filter_writer = filter_writer.map(|w| w.lock().unwrap());
                        let mut match_writer = match_writer.map(|w| w.lock().unwrap());

                        // Push to buffer.
                        let idx = batch_id - *next_batch_id;
                        if output_buf.len() <= idx {
                            output_buf.resize_with(idx + 1, || None);
                        }
                        assert!(output_buf[idx].is_none());
                        output_buf[idx] = Some(results);
                        results = vec![];

                        // Print pending results once ready.
                        while let Some(front) = output_buf.front()
                            && front.is_some()
                        {
                            *next_batch_id += 1;
                            let front_results = output_buf.pop_front().unwrap().unwrap();
                            for (path, text, matches) in front_results {
                                args.output(
                                    path,
                                    &text.0[text.1],
                                    matches,
                                    &mut filter_writer,
                                    &mut match_writer,
                                );
                            }
                        }
                    }

                    // Merge local histogram into global histogram.
                    let mut global_hist = global_histogram.lock().unwrap();
                    for dist in 0..=k {
                        global_hist[dist] += local_histogram[dist];
                    }
                });
            }
        });

        let global_hist = global_histogram.into_inner().unwrap();
        eprint!(
            "\nStatistics: total {}, ",
            global_hist.iter().sum::<usize>().to_string().bold()
        );
        for (dist, &count) in global_hist.iter().enumerate() {
            if count > 0 {
                eprint!(
                    "dist {} => {}, ",
                    dist.to_string().bold(),
                    count.to_string().bold()
                );
            }
        }
        eprintln!();

        assert!(output.into_inner().unwrap().1.is_empty());
    }

    /// Output all matches for a single text record.
    fn output(
        &self,
        path: &Path,
        text: &TextRecord,
        mut matches: Vec<(&PatternRecord, Match)>,
        filter_writer: &mut Option<std::sync::MutexGuard<'_, Box<dyn Write + Send>>>,
        match_writer: &mut Option<std::sync::MutexGuard<'_, Box<dyn Write + Send>>>,
    ) {
        matches.sort_by_key(|m| m.1.text_start);
        // 1. Print filter output.
        if self.filter.is_some() {
            let writer = &mut **filter_writer.as_mut().unwrap();
            if !self.base.invert && !matches.is_empty() {
                self.print_matching_record(&text, writer);
            }
            if self.base.invert && matches.is_empty() {
                self.print_matching_record(&text, writer);
            }
        }

        match (self.grep, self.search.is_some()) {
            // 2. If grep, interleave with search output if needed.
            (true, _) => self.print_matches_for_record(
                path,
                &text,
                &mut matches,
                match_writer.as_deref_mut(),
            ),
            // 3. Just write the search output.
            (false, true) => {
                for (pattern, m) in &matches {
                    self.print_match_tsv(pattern, &text, m, match_writer.as_deref_mut().unwrap());
                }
            }
            (false, false) => {}
        };
    }

    fn get_patterns(&self) -> Vec<PatternRecord> {
        if let Some(p) = &self.base.pattern {
            // Single inline pattern; give it a dummy id
            return vec![PatternRecord {
                id: "pattern".to_string(),
                seq: p.as_bytes().to_vec(),
            }];
        }
        if let Some(pattern_file) = &self.base.pattern_file {
            // Read all lines of the file.
            let file = File::open(pattern_file).expect("valid pattern file");
            let reader = std::io::BufReader::new(file);

            let mut patterns = Vec::new();
            for line in reader.lines() {
                patterns.push(PatternRecord {
                    id: format!("{}", patterns.len() + 1),
                    seq: line.unwrap().into(),
                });
            }
            return patterns;
        }
        if let Some(pattern_fasta) = &self.base.pattern_fasta {
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
            return patterns;
        }
        panic!("No --pattern, --pattern-file, or --pattern-fasta provided!");
    }

    fn print_matching_record(&self, text: &TextRecord, writer: &mut (dyn std::io::Write + Send)) {
        if !text.quality.is_empty() {
            writeln!(writer, "@{}", text.id).unwrap();
            writeln!(writer, "{}", String::from_utf8_lossy(&text.seq.text)).unwrap();
            writeln!(writer, "+").unwrap();
            writeln!(writer, "{}", String::from_utf8_lossy(&text.quality)).unwrap();
        } else {
            writeln!(writer, ">{}", text.id).unwrap();
            writeln!(writer, "{}", String::from_utf8_lossy(&text.seq.text)).unwrap();
        }
    }

    fn print_matches_for_record(
        &self,
        path: &Path,
        text: &TextRecord,
        matches: &Vec<(&PatternRecord, Match)>,
        mut match_writer: Option<&mut impl std::io::Write>,
    ) {
        if matches.is_empty() {
            return;
        }
        eprintln!(
            "{}",
            format!(
                "{}>{}",
                path.display().to_string().cyan().bold(),
                text.id.bold()
            )
            .bold()
        );
        for (pattern, m) in matches {
            if let Some(match_writer) = match_writer.as_mut() {
                self.print_match_tsv(pattern, text, m, match_writer);
            }
            let s = m.pretty_print(
                Some(&pattern.id),
                &pattern.seq,
                &text.seq.text,
                PrettyPrintDirection::Text,
                self.context,
                PrettyPrintStyle::Full,
            );
            eprintln!("{s}");
        }
    }

    fn print_match_tsv(
        &self,
        pattern: &PatternRecord,
        text: &TextRecord,
        m: &Match,
        writer: &mut impl std::io::Write,
    ) {
        let cost = m.cost;
        let start = m.text_start;
        let end = m.text_end;
        let slice = &text.seq.text[start..end];

        // If we match reverse complement, reverse complement the slice to make it easier to read
        let slice_str = if m.strand == Strand::Rc {
            match self.base.alphabet {
                Alphabet::Dna => {
                    String::from_utf8_lossy(&<Dna as Profile>::reverse_complement(slice))
                        .into_owned()
                }
                Alphabet::Iupac => {
                    String::from_utf8_lossy(&<Iupac as Profile>::reverse_complement(slice))
                        .into_owned()
                }
            }
        } else {
            String::from_utf8_lossy(slice).into_owned()
        };

        let pat_id = &pattern.id;
        let text_id = &text.id;
        let cigar = m.cigar.to_string();
        let strand = match m.strand {
            Strand::Fwd => "+",
            Strand::Rc => "-",
        };
        writeln!(
            writer,
            "{pat_id}\t{text_id}\t{cost}\t{strand}\t{start}\t{end}\t{slice_str}\t{cigar}"
        )
        .unwrap();
    }
}

#[cfg(test)]
mod test {
    use sassy::profiles::{Dna, Profile};

    #[test]
    fn amplicon_crash() {
        let pattern = b"TGTACTTCGTTCAGTTACGTATTGCTAAGGTTAACTACTTACGAAGCTGAGGGACTGCCAGCACCTACCATCTTGGACTGAGATCTTTCATTTTACCGTCACCACCACGAATTCGTCTGGTAGCTCTTCGGTAGTAGCCAATTTGGTCATCTGGACTGCTATTGGTGTTAATTGGAACGCCTTGTCCTCGAGGGAATTTAAGGTCTTCCTTGCCATGTTGAGTGAGAGCGGTGAACCAAGACGCAGTATTATTGGGTAAACCTTGGGGCCGACGTTGTTTTGATCGCGCCCACTGCGTTCTCCATTCTGGTTACTGCCAGTTGAATCTGAGGTCCACCAAACGTAATGCGGGGTGCTTTCGCTGATTTTGGGGTCCATTATCGAACATTTTAGTTTGTTCGTTTAGATGAAATCTAAAACAACACGAACGTCATGATACTCTAAAAAGTCTTCATAGAACGAACAACGCACAGAGTGCTAGCAGTCCCTCAGCTTCGTAAGTAT";
        let text = b"CAGTTACGTATTGCTAAGGTTAACTACTTACGAAGCTGAGGGACTGCCAGCACCTACCATCTTGGACTGAGATCTTTCATTTTACCGTCACCACCACGAATTCGTCTGGTAGCTCTTCGGTAGTAGCCACCGGTCATCTGGACTGCTATTGGTGTTAATTGGAACGCCTTGTCCTCGAGGGAATTTAAGGTCTTCCTTGCCATGTTGAGTGAGAGCGGTGAACCAAGACGCAGTATTATTGGGTAAACCTTGGGGCCGACGTTGTTTTGATCGCGCCCCCACTGCGTTCTCCATTCTGGTTACTGCCAGTTGAATCTGAGGGTCCACCAAACGTAATGCGGGGTGCATTTCGCTGATTTTGGGGTCCATTATCAGACATTTTAGTTTACCTGTTTAGATGAAATCTAAAACAACACGAACGTCATGATACTCTAAAAAGTCTTCATAGAACGAACAACGCACAGGTGCTGGCAGTCCCCAGCTTCGTAAGTAGTTAACCTTAGCAATACGTAACTGAACGAAGCATAA";
        let text = &Dna::reverse_complement(text);

        let mut searcher = sassy::Searcher::<sassy::profiles::Iupac>::new_rc_with_overhang(0.5);
        let matches = searcher.search(pattern, text, 40);
        matches.iter().for_each(|m| {
            eprintln!("match: {:?}", m.without_cigar());
            eprintln!("cigar: {}", m.cigar.to_string());
            m.pretty_print(
                None,
                pattern,
                text,
                sassy::pretty_print::PrettyPrintDirection::Text,
                0,
                sassy::pretty_print::PrettyPrintStyle::Full,
            );
        });
    }
}
