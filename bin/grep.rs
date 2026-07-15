use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    io::{BufRead, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use colored::Colorize;
use needletail::parse_fastx_file;
use noodles::{
    bam::{self, io::reader::header::sam_header},
    bgzf,
    sam::{
        self,
        alignment::{RecordBuf, io::Write as AlignmentWrite},
    },
};
use pa_types::Cigar;
use sassy::{
    Match, Searcher, Strand,
    pretty_print::{PrettyPrintDirection, PrettyPrintStyle},
    profiles::{Ascii, Dna, Iupac, Profile},
};

use crate::{
    input_iterator::{InputIterator, PatternRecord, TextBatch, TextRecord},
    sam::{SamColumn, is_alignment_path, is_bam_path},
};

// TODO: Support ASCII alphabet.
#[derive(clap::ValueEnum, Default, Clone, Copy, PartialEq)]
pub(crate) enum Alphabet {
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
    /// Use `agrep` for ascii.
    #[arg(
        long,
        short = 'a',
        default_value_t = Alphabet::Iupac,
        value_enum
    )]
    alphabet: Alphabet,

    /// Cost per base of overhang alignment, where the pattern extends beyond the text in [0,1]. Default disabled.
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

    /// Use Sassy v2 encoded search (faster for many equal-length patterns).
    /// Note: results can differ slightly from V1 due to the difference in reverse complement searching.
    #[arg(long)]
    v2: bool,

    /// Number of threads to use. All CPUs by default.
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Only report non-matching records. Only applies to `filter` output.
    #[arg(short = 'v', long)]
    invert: bool,

    /// SAM-compatible output: print the `match_region` and `cigar` in text direction.
    /// (By default, sassy outputs these in the _pattern_ direction, and reverse complements the `match_region` and `cigar` for rc matches.)
    #[arg(long)]
    sam: bool,

    // Positional
    /// Input FASTX or BAM files. FASTX may be gzipped; BAM is selected by a `.bam` extension.
    paths: Vec<PathBuf>,
}

#[derive(clap::Parser, Clone)]
pub struct GrepArgs {
    #[command(flatten)]
    base: BaseArgs,

    /// Number of characters to print before/after each match.
    #[arg(short = 'C', long, default_value_t = 20)]
    context: usize,

    /// TSV output file to write all matches. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    search: Option<PathBuf>,

    /// Filtered output file. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    filter: Option<PathBuf>,

    /// Extra SAM fields to append to TSV output for SAM/BAM input.
    #[arg(long, value_name = "FIELDS")]
    more_columns: Option<String>,
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

    /// Filtered output file. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    filter: Option<PathBuf>,

    /// Extra SAM fields to append to TSV output for SAM/BAM input.
    #[arg(long, value_name = "FIELDS")]
    more_columns: Option<String>,
}

#[derive(clap::Parser, Clone)]
pub struct FilterArgs {
    #[command(flatten)]
    base: BaseArgs,

    /// TSV output file to write all matches. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    search: Option<PathBuf>,

    /// Add Sassy match data as local-use SAM/BAM tags when filtering alignment input.
    #[arg(long)]
    annotate: bool,
}

struct Args {
    base: BaseArgs,

    context: usize,

    grep: bool,
    search: Option<PathBuf>,
    filter: Option<PathBuf>,
    more_columns: Vec<SamColumn>,
    annotate: bool,
}

enum FilterWriter {
    Fastx(Box<dyn Write + Send>),
    Bam {
        writer: bam::io::Writer<bgzf::io::Writer<Box<dyn Write + Send>>>,
        header: std::sync::Arc<sam::Header>,
    },
    Sam {
        writer: sam::io::Writer<Box<dyn Write + Send>>,
        header: Arc<sam::Header>,
    },
}

impl GrepArgs {
    pub fn run(self) {
        let GrepArgs {
            base,
            context,
            search,
            filter,
            more_columns,
        } = self;
        Args {
            base,
            context,
            grep: true,
            search,
            filter,
            more_columns: SamColumn::parse_list(more_columns.as_deref()),
            annotate: false,
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
        let SearchArgs {
            base,
            filter,
            more_columns,
        } = self;
        Args {
            base,
            context: 0,
            grep: false,
            search: Some(PathBuf::from("")),
            filter,
            more_columns: SamColumn::parse_list(more_columns.as_deref()),
            annotate: false,
        }
        .run()
    }
}

impl FilterArgs {
    pub fn run(self) {
        let FilterArgs {
            base,
            search,
            annotate,
        } = self;
        Args {
            base,
            context: 0,
            grep: false,
            search,
            filter: Some(PathBuf::from("")),
            more_columns: Vec::new(),
            annotate,
        }
        .run()
    }
}

fn run_batch_v2<'a, P: Profile>(
    searcher: &mut Searcher<P>,
    results: &mut Vec<(
        &'a Path,
        (TextBatch, usize),
        Vec<(&'a PatternRecord, Match)>,
    )>,
    path: &'a Path,
    patterns: &'a [PatternRecord],
    text_batch: &TextBatch,
    k: usize,
) {
    let patterns_vec: Vec<Vec<u8>> = patterns.iter().map(|p| p.seq.clone()).collect();
    let encoded = searcher.encode_patterns(&patterns_vec);
    for (i, text) in text_batch.iter().enumerate() {
        let record_matches = searcher.search_encoded_patterns(&encoded, &text.seq().text, k);
        let batch_matches: Vec<_> = record_matches
            .iter()
            .cloned()
            .map(|m| {
                let pattern = &patterns[m.pattern_idx];
                (pattern, m)
            })
            .collect();
        if batch_matches.is_empty() {
            continue;
        }
        results.push((path, (text_batch.clone(), i), batch_matches));
    }
}

impl Args {
    fn validate_args(&mut self) {
        if self.base.invert && self.filter.is_none() {
            eprintln!(
                "{}",
                "Warning: --invert/-v has no effect without --filter."
                    .red()
                    .bold()
            );
        }

        match self.base.paths.len() {
            0 => {
                // No argument. Read from stdin.
                self.base.paths = vec![PathBuf::from("")];
            }

            1 => {
                // One input file. Can be fast or alignment
                let path = &self.base.paths[0];
                if is_alignment_path(path) {
                    if self.base.sam {
                        panic!(
                            "`--sam` does not apply to SAM/BAM input to avoid confusion. Remove `--sam`"
                        );
                    } else {
                        if !self.more_columns.is_empty() {
                            eprintln!(
                                "`--more-columns` applies to SAM/BAM inputs. Ignored for fastx input."
                            );
                        }
                    }
                }
            }

            _ => {
                // Multiple inputs. Does not support multiple alignment file input.
                if self.base.paths.iter().any(|path| is_alignment_path(path)) {
                    panic!("Only fastx files are supported as multiple inputs.")
                }
            }
        }
    }
    fn run(mut self) {
        self.validate_args();

        let args = &self;

        let patterns = args.get_patterns();
        assert!(!patterns.is_empty(), "No pattern sequences found");

        let k = args.base.k;
        let rc = (args.base.alphabet == Alphabet::Dna || args.base.alphabet == Alphabet::Iupac)
            && !args.base.no_rc;

        let num_threads = args.base.threads.unwrap_or_else(num_cpus::get);
        let (task_iterator, sam_header) = InputIterator::new(
            &args.base.paths,
            &patterns,
            None,
            args.base.pattern_batch_size,
            rc,
        );
        let task_iterator = &task_iterator;

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

            if let Some(header) = sam_header.clone() {
                // Input is an alignment file
                if is_bam_path(path) {
                    let mut bam_writer = bam::io::Writer::new(writer);
                    bam_writer.write_header(&header).expect("write BAM header");
                    Mutex::new(FilterWriter::Bam {
                        writer: bam_writer,
                        header,
                    })
                } else {
                    let mut sam_writer = sam::io::Writer::new(writer);
                    sam_writer.write_header(&header).expect("write SAM header");
                    Mutex::new(FilterWriter::Sam {
                        writer: sam_writer,
                        header,
                    })
                }
            } else {
                Mutex::new(FilterWriter::Fastx(writer))
            }
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

            let mut columns = vec![
                "pat_id",
                "text_id",
                "cost",
                "strand",
                "start",
                "end",
                "match_region",
                "cigar",
            ];
            columns.extend(args.more_columns.iter().map(|column| column.name()));
            let header = format!("{}\n", columns.join("\t")); // TODO: data columns
            write!(writer, "{header}").unwrap();

            Mutex::new(writer)
        });

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let output = &output;
                let global_histogram = &global_histogram;
                let filter_writer = filter_writer.as_ref();
                let match_writer = match_writer.as_ref();
                let sam_header = sam_header.clone();
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

                    while let Some((batch_id, batch)) = task_iterator.next_batch() {
                        let path = batch.0;

                        if args.base.v2 {
                            match &mut searcher {
                                SearcherType::Dna(s) => {
                                    run_batch_v2(s, &mut results, path, batch.1, &batch.2, k)
                                }
                                SearcherType::Iupac(s) => {
                                    run_batch_v2(s, &mut results, path, batch.1, &batch.2, k)
                                }
                            }
                        } else {
                            for (i, text) in batch.2.iter().enumerate() {
                                let mut batch_matches = vec![];
                                for pattern in batch.1 {
                                    let record_matches = match &mut searcher {
                                        SearcherType::Dna(s) => {
                                            s.search(&pattern.seq, text.seq(), k)
                                        }
                                        SearcherType::Iupac(s) => {
                                            s.search(&pattern.seq, text.seq(), k)
                                        }
                                    };
                                    batch_matches
                                        .extend(record_matches.into_iter().map(|m| (pattern, m)));
                                }

                                if batch_matches.is_empty() {
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
                                    sam_header.as_deref(),
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

        if let Some(filter_writer) = filter_writer {
            let mut filter_writer = filter_writer.into_inner().unwrap();
            if let FilterWriter::Bam { writer, .. } = &mut filter_writer {
                writer.try_finish().expect("finish BAM output");
            }
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
        filter_writer: &mut Option<std::sync::MutexGuard<'_, FilterWriter>>,
        match_writer: &mut Option<std::sync::MutexGuard<'_, Box<dyn Write + Send>>>,
        header: Option<&noodles::sam::Header>,
    ) {
        matches.sort_by_key(|m| m.1.text_start);
        // 1. Print filter output.
        if self.filter.is_some() {
            let writer = &mut **filter_writer.as_mut().unwrap();

            if !self.base.invert && !matches.is_empty() | self.base.invert && matches.is_empty() {}
            match (text, writer) {
                (TextRecord::Fastx { .. }, FilterWriter::Fastx(writer)) => {
                    self.print_matching_record(text, writer);
                }
                (TextRecord::Sam { record_buf, .. }, FilterWriter::Sam { writer, header }) => {
                    self.print_matching_alignment_record(record_buf, &matches, header, writer);
                }
                (TextRecord::Sam { record_buf, .. }, FilterWriter::Bam { writer, header }) => {
                    self.print_matching_alignment_record(record_buf, &matches, header, writer);
                }
                _ => {
                    // Should not happen.
                }
            }
        }

        match (self.grep, self.search.is_some()) {
            // 2. If grep, interleave with search output if needed.
            (true, _) => self.print_matches_for_record(
                path,
                header,
                text,
                &matches,
                match_writer.as_deref_mut(),
            ),
            // 3. Just write the search output.
            (false, true) => {
                for (pattern, m) in &matches {
                    self.print_match_tsv(
                        pattern,
                        header,
                        text,
                        m,
                        match_writer.as_deref_mut().unwrap(),
                    );
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
        if !text.quality().is_empty() {
            writeln!(writer, "@{}", text.id()).unwrap();
            writeln!(writer, "{}", String::from_utf8_lossy(&text.seq().text)).unwrap();
            writeln!(writer, "+").unwrap();
            writeln!(writer, "{}", String::from_utf8_lossy(&text.quality())).unwrap();
        } else {
            writeln!(writer, ">{}", text.id()).unwrap();
            writeln!(writer, "{}", String::from_utf8_lossy(&text.seq().text)).unwrap();
        }
    }

    fn print_matching_alignment_record(
        &self,
        record_buf: &RecordBuf,
        matches: &[(&PatternRecord, Match)],
        header: &Arc<noodles::sam::Header>,
        writer: &mut (dyn sam::alignment::io::Write),
    ) {
        let mut record = record_buf.clone();
        if self.annotate && !matches.is_empty() {
            crate::sam::annotate_bam_record(
                &mut record,
                matches,
                self.base.alphabet,
                self.base.sam,
            );
        }
        writer
            .write_alignment_record(header, &record)
            .expect("write alignment record");
    }

    fn print_matches_for_record(
        &self,
        path: &Path,
        header: Option<&sam::Header>,
        text: &TextRecord,
        matches: &Vec<(&PatternRecord, Match)>,
        mut match_writer: Option<&mut impl std::io::Write>,
    ) {
        if matches.is_empty() {
            return;
        }
        let additional_columns = if !self.more_columns.is_empty()
            && let TextRecord::Sam { record_buf, .. } = text
        {
            let fields = crate::sam::sam_fields(
                header.expect("BAM header metadata is unavailable for this output record"),
                record_buf,
            );
            format!(
                "\t{}",
                self.more_columns
                    .iter()
                    .map(|column| format!("{}={}", column.name(), column.value(&fields)))
                    .collect::<Vec<_>>()
                    .join("\t")
            )
        } else {
            String::new()
        };
        eprintln!(
            "{}",
            format!(
                "{}>{}{additional_columns}",
                path.display().to_string().cyan().bold(),
                text.id().bold()
            )
            .bold()
        );
        for (pattern, m) in matches {
            if let Some(match_writer) = match_writer.as_mut() {
                self.print_match_tsv(pattern, header, text, m, match_writer);
            }
            let s = m.pretty_print(
                Some(&pattern.id),
                &pattern.seq,
                &text.seq().text,
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
        header: Option<&sam::Header>,
        text: &TextRecord,
        m: &Match,
        writer: &mut impl std::io::Write,
    ) {
        let cost = m.cost;
        let start = m.text_start;
        let end = m.text_end;
        let slice = &text.seq().text[start..end];
        let match_region = self.format_match_region(slice, m.strand);
        let match_region = String::from_utf8_lossy(&match_region);

        let pat_id = &pattern.id;
        let text_id = text.id();
        let cigar = self.format_cigar(&m.cigar, m.strand);
        let strand = match m.strand {
            Strand::Fwd => "+",
            Strand::Rc => "-",
        };
        write!(
            writer,
            "{pat_id}\t{text_id}\t{cost}\t{strand}\t{start}\t{end}\t{match_region}\t{cigar}"
        )
        .unwrap();
        if !self.more_columns.is_empty()
            && let TextRecord::Sam { record_buf, .. } = text
            && let Some(header) = header
        {
            let fields = crate::sam::sam_fields(header, record_buf);
            for column in &self.more_columns {
                write!(writer, "\t{}", column.value(&fields)).unwrap();
            }
        }
        writeln!(writer).unwrap();
    }

    fn format_match_region(&self, slice: &[u8], strand: Strand) -> Vec<u8> {
        crate::sam::format_match_region(self.base.alphabet, self.base.sam, slice, strand)
    }

    fn format_cigar(&self, cigar: &Cigar, strand: Strand) -> String {
        crate::sam::format_cigar(self.base.sam, cigar, strand)
    }
}

#[cfg(test)]
mod test {
    use clap::Parser;
    use pa_types::Cigar;
    use sassy::profiles::{Dna, Profile};

    use super::{Args, GrepArgs, SearchArgs};
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
            search,
            filter,
            more_columns: _,
        } = GrepArgs::try_parse_from(argv).unwrap();
        Args {
            base,
            context,
            grep: true,
            search,
            filter,
            more_columns: Vec::new(),
            annotate: false,
        }
    }
}
