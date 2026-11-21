use std::{
    collections::VecDeque,
    fs::File,
    io::{BufRead, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use colored::Colorize;
use helicase::{FastxParser, HelicaseParser};
use pa_types::{Cigar, CigarOp};
use sassy::{
    Match, Searcher, Strand,
    pretty_print::{PrettyPrintDirection, PrettyPrintStyle},
    profiles::{Ascii, Dna, Iupac, Profile},
};

use crate::input_iterator::{
    BufferedInputIterator, FASTX_CONFIG, InputIterator, PatternRecord, TextBatch, TextRecord,
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

    /// Number of patterns to process inside each batch. Default 32.
    #[arg(long, requires = "pattern_file")]
    pattern_batch_size: Option<usize>,

    /// Report matches up to (and including) this distance threshold.
    #[arg(short)]
    k: usize,

    /// The alphabet to use. DNA=ACTG, or default IUPAC=ACTG+NYR....
    ///
    /// Use `agrep` for ascii.
    #[arg(
        long,
        short = 'a',
        default_value_t = Alphabet::Iupac,
        value_enum
    )]
    alphabet: Alphabet,

    /// Cost per base of overhang alignment in 0..1, where the pattern extends beyond the text. Default disabled.
    #[arg(long)]
    overhang: Option<f32>,

    /// Disable reverse complement search.
    #[arg(long)]
    no_rc: bool,

    /// Allow at most max_n_frac of N bases in the target sequence.
    ///
    /// Values must be in the
    /// range [0, 1].  A value of 0 will allow only hits where the target sequence contains no
    /// Ns. A value of 0.1-0.2 will allow for matches that include a small number of Ns.
    #[arg(long, default_value_t = 0.2)]
    max_n_frac: f32,

    /// Use Sassy v2 encoded search (faster for many equal-length patterns).
    ///
    /// With N filtering, v2 can pick different local minima than v1.
    #[arg(long)]
    v2: bool,

    /// Number of threads to use. All CPUs by default.
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Only report non-matching records. Only applies to `filter` output.
    #[arg(short = 'v', long)]
    invert: bool,

    /// SAM-compatible output: print the `match_region` and `cigar` in text direction.
    ///
    /// (By default, sassy outputs these in the _pattern_ direction, and reverse complements the `match_region` and `cigar` for rc matches.)
    #[arg(long)]
    sam: bool,

    // Positional
    /// Input Fastx files. May be gzipped.
    paths: Vec<PathBuf>,
}

#[derive(clap::Parser, Clone)]
pub struct GrepArgs {
    #[command(flatten)]
    base: BaseArgs,

    /// Number of characters to print before/after each match.
    #[arg(short = 'C', long, default_value_t = 20)]
    context: usize,

    /// Optional .tsv output path for matches. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    matches: Option<PathBuf>,

    /// Optional .fa output path for matching records. Empty or "-" for stdout.
    #[arg(short='o', long, default_missing_value = "-", num_args(0..=1))]
    output: Option<PathBuf>,
}

/// Thin mirror of `GrepArgs` for lightweight `agrep` subcommand.
#[derive(clap::Parser, Clone)]
pub struct AGrepArgs {
    /// The pattern to search for.
    pattern: String,

    /// Report matches up to (and including) this distance threshold.
    k: usize,

    /// Number of lines to print before/after each match.
    #[arg(short = 'C', long, default_value_t = 0)]
    context: usize,

    /// Input files. Empty or "-" for stdin.
    paths: Vec<PathBuf>,
}

#[derive(clap::Parser, Clone)]
pub struct SearchArgs {
    #[command(flatten)]
    base: BaseArgs,

    /// Optional .tsv output path for matches. Default stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    matches: Option<PathBuf>,

    /// Optional .fa output path for matching records. Empty or "-" for stdout.
    #[arg(short='o', long, default_missing_value = "-", num_args(0..=1))]
    output: Option<PathBuf>,

    /// Report all qualifying endpoints and require a matching terminal PAM.
    #[arg(long)]
    crispr: bool,

    /// Number of bases in the terminal PAM.
    #[arg(long, requires = "crispr")]
    pam_length: Option<usize>,

    /// Allow edits in the terminal PAM.
    #[arg(long, requires = "crispr")]
    allow_pam_edits: bool,
}

#[derive(clap::Parser, Clone)]
pub struct FilterArgs {
    #[command(flatten)]
    base: BaseArgs,

    /// Optional .tsv output path for matches. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    matches: Option<PathBuf>,

    /// Optional .fa output path for matching records. Default stdout.
    #[arg(short='o', long, default_missing_value = "-", num_args(0..=1))]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy)]
pub(super) struct CrisprMode {
    pub(super) pam_length: usize,
    pub(super) allow_pam_edits: bool,
    pub(super) legacy_tsv: bool,
}

impl CrisprMode {
    fn needs_pam(self) -> bool {
        !self.allow_pam_edits && self.pam_length != 0
    }

    pub(super) fn accepts_endpoint<P: Profile>(self, pattern: &[u8], prefix: &[u8]) -> bool {
        prefix.len() >= self.pam_length
            && P::is_match_slice(
                &pattern[pattern.len() - self.pam_length..],
                &prefix[prefix.len() - self.pam_length..],
            )
    }

    pub(super) fn accepts_match(self, pattern: &PatternRecord, m: &Match) -> bool {
        // CIGARs are in pattern orientation on both strands. Require the PAM
        // to be covered by an uninterrupted run of terminal matches.
        !self.needs_pam()
            || (m.pattern_end == pattern.seq.len()
                && m.cigar.ops.last().is_some_and(|op| {
                    op.op == CigarOp::Match && op.cnt as usize >= self.pam_length
                }))
    }
}

struct Args {
    base: BaseArgs,

    context: usize,

    grep: bool,
    crispr: Option<CrisprMode>,
    search: Option<PathBuf>,
    filter: Option<PathBuf>,
}

impl GrepArgs {
    pub fn run(self) {
        let GrepArgs {
            base,
            context,
            matches: search,
            output: filter,
        } = self;
        Args {
            base,
            context,
            grep: true,
            crispr: None,
            search,
            filter,
        }
        .run()
    }
}

impl AGrepArgs {
    pub fn run(self) {
        let pattern = self.pattern.as_bytes();

        let num_threads = num_cpus::get();

        let output = Mutex::new((
            0,
            VecDeque::<Option<Vec<(&PathBuf, Vec<u8>, Vec<Match>)>>>::new(),
        ));
        let global_histogram = Mutex::new(vec![0usize; self.k + 1]);

        // Grep writer is stderr
        // Ensure colours are always used.
        colored::control::set_override(true);

        let path_iter = &Mutex::new(self.paths.iter().enumerate());

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let output = &output;
                let global_histogram = &global_histogram;
                s.spawn(move || {
                    // Each thread has own searcher here
                    let mut searcher = Searcher::<Ascii>::new_fwd();

                    let mut results = vec![];
                    let mut local_histogram = vec![0usize; self.k + 1];

                    while let Some((path_id, path)) = path_iter.lock().unwrap().next() {
                        let mut text = Vec::new();
                        if path.to_str() == Some("-") || path.to_str() == Some("") {
                            std::io::stdin().read_to_end(&mut text).unwrap();
                        } else {
                            File::open(path)
                                .unwrap()
                                .read_to_end(&mut text)
                                .expect("valid input file");
                        };

                        let matches = searcher.search(pattern, &text, self.k);

                        if matches.is_empty() {
                            continue;
                        }
                        results.push((path, text, matches));

                        for (_path, _text, matches) in &results {
                            for m in matches {
                                local_histogram[m.cost as usize] += 1;
                            }
                        }

                        let (next_path_id, output_buf) = &mut *output.lock().unwrap();

                        // Push to buffer.
                        let idx = path_id - *next_path_id;
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
                            *next_path_id += 1;
                            let front_results = output_buf.pop_front().unwrap().unwrap();
                            for (path, text, mut matches) in front_results {
                                // Write the matches.
                                matches.sort_by_key(|m| m.text_start);
                                // Print matches for path.
                                eprintln!(
                                    "{}",
                                    format!("{}:", path.display().to_string().cyan().bold(),)
                                        .bold()
                                );
                                for m in matches {
                                    let s = m.pretty_print(
                                        Some(""),
                                        pattern,
                                        &text,
                                        PrettyPrintDirection::Text,
                                        self.context,
                                        PrettyPrintStyle::Line,
                                    );
                                    eprintln!("{s}");
                                    if self.context > 0 {
                                        eprintln!("{}", "---".cyan());
                                    }
                                }
                            }
                        }
                    }

                    // Merge local histogram into global histogram.
                    let mut global_hist = global_histogram.lock().unwrap();
                    for dist in 0..=self.k {
                        global_hist[dist] += local_histogram[dist];
                    }
                });
            }
        });

        print_statistics(&global_histogram.into_inner().unwrap());
        assert!(output.into_inner().unwrap().1.is_empty());
    }
}

fn print_statistics(hist: &[usize]) {
    eprintln!(
        "\nStatistics: total {}",
        hist.iter().sum::<usize>().to_string().bold()
    );

    let max_cnt = hist.iter().max().unwrap_or(&0);
    let digits = max_cnt.to_string().len();

    eprint!("dist: ");
    for i in 0..hist.len() {
        eprint!("{:>digits$} ", i.to_string().bold());
    }
    eprintln!();
    eprint!("cnt:  ");
    for count in hist {
        eprint!("{:>digits$} ", count.to_string().bold());
    }
    eprintln!();
}

impl SearchArgs {
    pub fn run(self) {
        Args {
            base: self.base,
            context: 0,
            grep: false,
            crispr: self.crispr.then_some(CrisprMode {
                pam_length: self.pam_length.unwrap_or(3),
                allow_pam_edits: self.allow_pam_edits,
                legacy_tsv: false,
            }),
            search: self.matches.or(Some(PathBuf::new())),
            filter: self.output,
        }
        .run()
    }

    pub(super) fn run_crispr(args: &crate::crispr::CrisprArgs) {
        Args {
            base: BaseArgs {
                pattern: None,
                pattern_file: Some(PathBuf::from(&args.guide)),
                pattern_fasta: None,
                pattern_batch_size: None,
                k: args.k,
                alphabet: Alphabet::Iupac,
                overhang: None,
                no_rc: args.no_rc,
                max_n_frac: args.max_n_frac,
                v2: args.v2,
                threads: args.threads,
                invert: false,
                sam: false,
                paths: vec![args.path.clone()],
            },
            context: 0,
            grep: false,
            crispr: Some(CrisprMode {
                pam_length: args.pam_length,
                allow_pam_edits: args.allow_pam_edits,
                legacy_tsv: true,
            }),
            search: Some(args.output.clone().unwrap_or_default()),
            filter: None,
        }
        .run()
    }
}

impl FilterArgs {
    pub fn run(self) {
        let FilterArgs {
            base,
            matches,
            output,
        } = self;
        Args {
            base,
            context: 0,
            grep: false,
            crispr: None,
            search: matches,
            filter: output.or(Some(PathBuf::from(""))),
        }
        .run()
    }
}

fn run_batch_v2<'a, P: Profile>(
    searcher: &mut Searcher<P>,
    //fixme: simpler return?
    results: &mut Vec<(
        &'a Path,
        (TextBatch, usize),
        Vec<(&'a PatternRecord, Match)>,
    )>,
    path: &'a Path,
    patterns: &'a [PatternRecord],
    text_batch: &TextBatch,
    k: usize,
    crispr: Option<CrisprMode>,
) {
    let patterns_vec: Vec<Vec<u8>> = patterns.iter().map(|p| p.seq.clone()).collect();
    let encoded = searcher.encode_patterns(&patterns_vec);
    for (i, text) in text_batch.iter().enumerate() {
        if text.seq.text.is_empty() && crispr.is_some() {
            continue;
        }
        let record_matches = search_v2(searcher, &encoded, text, k, crispr);
        let batch_matches: Vec<_> = record_matches
            .iter()
            .filter(|m| text.owns_match(m))
            .filter_map(|m| {
                let pattern = &patterns[m.pattern_idx];
                crispr
                    .is_none_or(|mode| mode.accepts_match(pattern, m))
                    .then(|| (pattern, m.clone()))
            })
            .collect();
        if batch_matches.is_empty() {
            continue;
        }
        results.push((path, (text_batch.clone(), i), batch_matches));
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
        if patterns.is_empty() && args.crispr.is_some_and(|mode| mode.legacy_tsv) {
            return;
        }
        assert!(!patterns.is_empty(), "No pattern sequences found");

        if let Some(mode) = args.crispr {
            assert!(
                (0.0..=1.0).contains(&args.base.max_n_frac),
                "max_n_frac must be between 0 and 1"
            );
            assert!(
                patterns
                    .iter()
                    .all(|pattern| pattern.seq.len() >= mode.pam_length),
                "All guides must be at least as long as the PAM"
            );
            if mode.legacy_tsv {
                let pam = &patterns[0].seq[patterns[0].seq.len() - mode.pam_length..];
                assert!(
                    patterns.iter().all(|pattern| pattern.seq.ends_with(pam)),
                    "All guides must have the same PAM"
                );
            }
        }
        let auto_v2 = if args.crispr.is_some() {
            equal_v2_pattern_lengths(&patterns)
        } else {
            should_suggest_v2(&patterns)
        };
        let v2 = args.base.v2 || auto_v2;
        if !args.base.v2 && auto_v2 {
            eprintln!("All patterns have equal length; using the --v2 backend.");
        }

        let k = args.base.k;
        let rc = (args.base.alphabet == Alphabet::Dna || args.base.alphabet == Alphabet::Iupac)
            && !args.base.no_rc;

        let num_threads = args.base.threads.unwrap_or_else(num_cpus::get);
        let input_iterator = &InputIterator::new(
            &args.base.paths,
            &patterns,
            None,
            args.base.pattern_batch_size,
            rc,
            patterns.iter().map(|p| p.seq.len()).max().unwrap_or(0) + k,
            self.filter.is_some(),
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
                Box::new(BufWriter::new(File::create(path).expect("create matches file")))
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

            let header = if args.crispr.is_some_and(|mode| mode.legacy_tsv) {
                "guide\ttext_id\tcost\tstrand\tstart\tend\tmatch_region\tcigar"
            } else {
                "pat_id\ttext_id\tcost\tstrand\tstart\tend\tmatch_region\tcigar\tpattern_start\tpattern_end\tpattern_match_region"
            };
            writeln!(writer, "{header}").unwrap();

            Mutex::new(writer)
        });

        let input_reader = &BufferedInputIterator::new(num_threads);

        std::thread::scope(|s| {
            input_reader.start_reader(s, input_iterator);

            for _ in 0..num_threads {
                let output = &output;
                let global_histogram = &global_histogram;
                let filter_writer = filter_writer.as_ref();
                let match_writer = match_writer.as_ref();
                s.spawn(move || {
                    enum SearcherType {
                        Dna(Searcher<Dna>),
                        Iupac(Searcher<Iupac>),
                    }

                    // Each thread has own searcher here
                    let mut searcher = match &args.base.alphabet {
                        Alphabet::Dna => {
                            SearcherType::Dna(Searcher::<Dna>::new(rc, args.base.overhang))
                        }
                        // Only make sense to set n_frac for IUPAC profile?
                        Alphabet::Iupac => SearcherType::Iupac(
                            Searcher::<Iupac>::new(rc, args.base.overhang)
                                .with_max_n_frac(args.base.max_n_frac),
                        ),
                    };

                    let mut results = vec![];
                    let mut local_histogram = vec![0usize; k + 1];

                    while let Some((batch_id, batch)) = input_reader.next_batch() {
                        let path = batch.0;

                        if v2 {
                            match &mut searcher {
                                SearcherType::Dna(s) => run_batch_v2(
                                    s,
                                    &mut results,
                                    path,
                                    batch.1,
                                    &batch.2,
                                    k,
                                    args.crispr,
                                ),
                                SearcherType::Iupac(s) => run_batch_v2(
                                    s,
                                    &mut results,
                                    path,
                                    batch.1,
                                    &batch.2,
                                    k,
                                    args.crispr,
                                ),
                            }
                        } else {
                            for (i, text) in batch.2.iter().enumerate() {
                                let mut batch_matches = vec![];
                                for pattern in batch.1 {
                                    let record_matches = match &mut searcher {
                                        SearcherType::Dna(s) => {
                                            search_v1(s, pattern, text, k, args.crispr)
                                        }
                                        SearcherType::Iupac(s) => {
                                            search_v1(s, pattern, text, k, args.crispr)
                                        }
                                    };
                                    batch_matches.extend(
                                        record_matches
                                            .into_iter()
                                            .filter(|m| text.owns_match(m))
                                            .filter(|m| {
                                                args.crispr.is_none_or(|mode| {
                                                    mode.accepts_match(pattern, m)
                                                })
                                            })
                                            .map(|m| (pattern, m)),
                                    );
                                }

                                // Don't skip when outputting non-matching records.
                                if !self.base.invert && batch_matches.is_empty() {
                                    continue;
                                }
                                results.push((path, (batch.2.clone(), i), batch_matches));
                            }
                        }

                        for (_, _, matches) in &results {
                            for (_, m) in matches {
                                local_histogram[m.cost as usize] += 1;
                            }
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

        if let Some(writer) = match_writer {
            writer.into_inner().unwrap().flush().unwrap();
        }
        print_statistics(&global_histogram.into_inner().unwrap());
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
                self.print_matching_record(text, writer);
            }
            if self.base.invert && matches.is_empty() {
                self.print_matching_record(text, writer);
            }
        }

        match (self.grep, self.search.is_some()) {
            // 2. If grep, interleave with search output if needed.
            (true, _) => {
                self.print_matches_for_record(path, text, &matches, match_writer.as_deref_mut())
            }
            // 3. Just write the search output.
            (false, true) => {
                for (pattern, m) in &matches {
                    self.print_match_tsv(pattern, text, m, match_writer.as_deref_mut().unwrap());
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
                let seq = line.unwrap().into_bytes();
                if self.crispr.is_some() && seq.is_empty() {
                    continue;
                }
                patterns.push(PatternRecord {
                    id: format!("{}", patterns.len() + 1),
                    seq,
                });
            }
            return patterns;
        }
        if let Some(pattern_fasta) = &self.base.pattern_fasta {
            let mut reader =
                FastxParser::<FASTX_CONFIG>::from_file(pattern_fasta).expect("valid path/file");
            let mut patterns = Vec::new();
            while let Some(record) = reader.next() {
                record.expect("invalid record");
                patterns.push(PatternRecord {
                    id: String::from_utf8(reader.get_header_owned()).unwrap(),
                    seq: reader.get_dna_string_owned(),
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
        let match_region = self.format_match_region(slice, m.strand);
        let match_region = String::from_utf8_lossy(&match_region);
        let pattern_start = m.pattern_start;
        let pattern_end = m.pattern_end;
        let pattern_match_region =
            String::from_utf8_lossy(&pattern.seq[pattern_start..pattern_end]);

        let legacy = self.crispr.is_some_and(|mode| mode.legacy_tsv);
        let guide;
        let pat_id = if legacy {
            guide = String::from_utf8_lossy(&pattern.seq);
            guide.as_ref()
        } else {
            &pattern.id
        };
        let text_id = &text.id;
        let cigar = self.format_cigar(&m.cigar, m.strand);
        let strand = match m.strand {
            Strand::Fwd => "+",
            Strand::Rc => "-",
        };
        write!(
            writer,
            "{pat_id}\t{text_id}\t{cost}\t{strand}\t{}\t{}\t{match_region}\t{cigar}",
            text.record_offset + start,
            text.record_offset + end,
        )
        .unwrap();
        if legacy {
            writeln!(writer).unwrap();
        } else {
            writeln!(
                writer,
                "\t{pattern_start}\t{pattern_end}\t{pattern_match_region}"
            )
            .unwrap();
        }
    }

    fn format_match_region(&self, slice: &[u8], strand: Strand) -> Vec<u8> {
        if strand == Strand::Rc && !self.base.sam {
            match self.base.alphabet {
                Alphabet::Dna => <Dna as Profile>::reverse_complement(slice),
                Alphabet::Iupac => <Iupac as Profile>::reverse_complement(slice),
            }
        } else {
            slice.to_vec()
        }
    }

    fn format_cigar(&self, cigar: &Cigar, strand: Strand) -> String {
        if strand == Strand::Rc && self.base.sam {
            let mut cigar = cigar.clone();
            cigar.reverse();
            cigar.to_string()
        } else {
            cigar.to_string()
        }
    }
}

pub(super) fn search_v1<P: Profile>(
    searcher: &mut Searcher<P>,
    pattern: &PatternRecord,
    text: &TextRecord,
    k: usize,
    crispr: Option<CrisprMode>,
) -> Vec<Match> {
    match crispr {
        Some(mode) if mode.needs_pam() => {
            searcher.search_with_fn(&pattern.seq, &text.seq, k, true, |pattern, prefix, _| {
                mode.accepts_endpoint::<P>(pattern, prefix)
            })
        }
        Some(_) => searcher.search_all(&pattern.seq, &text.seq, k),
        None => searcher.search(&pattern.seq, &text.seq, k),
    }
}

pub(super) fn search_v2<'a, P: Profile>(
    searcher: &'a mut Searcher<P>,
    encoded: &sassy::EncodedPatterns<P>,
    text: &TextRecord,
    k: usize,
    crispr: Option<CrisprMode>,
) -> &'a [Match] {
    match crispr {
        Some(mode) if mode.needs_pam() => searcher.search_all_encoded_patterns_with_fn(
            encoded,
            &text.seq,
            k,
            |pattern, prefix| mode.accepts_endpoint::<P>(pattern, prefix),
        ),
        Some(_) => searcher.search_all_encoded_patterns(encoded, &text.seq, k),
        None => searcher.search_encoded_patterns(encoded, &text.seq, k),
    }
}

fn should_suggest_v2(patterns: &[PatternRecord]) -> bool {
    patterns.len() > 8 && equal_v2_pattern_lengths(patterns)
}

fn equal_v2_pattern_lengths(patterns: &[PatternRecord]) -> bool {
    let Some(first) = patterns.first() else {
        return false;
    };
    first.seq.len() <= 64 && patterns.iter().all(|p| p.seq.len() == first.seq.len())
}

#[cfg(test)]
mod test {
    use clap::Parser;
    use pa_types::Cigar;
    use sassy::profiles::{Dna, Profile};
    use std::io::Write;
    use tempfile::NamedTempFile;

    use super::{Args, FilterArgs, GrepArgs, SearchArgs};
    use sassy::Strand;

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

    #[test]
    fn sam_output() {
        // Grep and search support --sam
        let args = GrepArgs::try_parse_from(["grep", "-p", "ACGT", "-k", "1", "--sam"]).unwrap();
        assert!(args.base.sam);
        let args =
            SearchArgs::try_parse_from(["search", "-p", "ACGT", "-k", "1", "--sam"]).unwrap();
        assert!(args.base.sam);

        let args_sam = test_args(true);
        let args_dft = test_args(false);

        // In SAM mode, RC matches are still in text direction.
        let slice = b"AAGT";
        assert_eq!(
            args_dft.format_match_region(slice, Strand::Rc),
            Dna::reverse_complement(slice)
        );
        assert_eq!(
            args_sam.format_match_region(slice, Strand::Rc),
            b"AAGT".to_vec()
        );

        // In SAM mode, RC cigars are still in text direction.
        let cigar = Cigar::from_string("2=1X3D");
        assert_eq!(args_dft.format_cigar(&cigar, Strand::Rc), "2=1X3D");
        assert_eq!(args_sam.format_cigar(&cigar, Strand::Rc), "3D1X2=");
        assert_eq!(args_sam.format_cigar(&cigar, Strand::Fwd), "2=1X3D");
        assert_eq!(args_dft.format_cigar(&cigar, Strand::Fwd), "2=1X3D");
    }

    fn test_args(sam: bool) -> Args {
        let mut argv = vec!["grep", "-p", "ACGT", "-k", "1"];
        if sam {
            argv.push("--sam");
        }
        let GrepArgs {
            base,
            context,
            matches: search,
            output: filter,
        } = GrepArgs::try_parse_from(argv).unwrap();
        Args {
            base,
            context,
            grep: true,
            crispr: None,
            search,
            filter,
        }
    }
}
