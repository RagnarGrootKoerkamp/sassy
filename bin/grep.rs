use std::{
    collections::VecDeque,
    fmt::Write as _,
    fs::File,
    io::{BufRead, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use colored::Colorize;
use either::Either;
use needletail::parse_fastx_file;
use pa_types::{Cigar, CigarElem, CigarOp};
use sassy::{
    Match, Searcher, Strand,
    profiles::{Dna, Iupac, Profile},
};

use crate::{
    crispr::check_n_frac,
    input_iterator::{InputIterator, PatternRecord, TextRecord},
};

// TODO: Support ASCII alphabet.
#[derive(clap::ValueEnum, Default, Clone, Copy, PartialEq)]
enum Alphabet {
    Dna,
    #[default]
    Iupac,
}

#[derive(clap::Parser, Clone)]
pub struct GrepArgs {
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
        short = 'f',
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

    /// Only report non-matching records. Implies `--filter`.
    #[arg(short = 'v', long)]
    invert: bool,

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

    /// Disable reverse complement search (enabled by default for DNA and IUPAC).
    #[arg(long)]
    no_rc: bool,

    /// Allow at most max_n_frac of N bases in the target sequence. Values must be in the
    /// range [0. 1].  A value of 0 will allow only hits where the target sequence contains no
    /// Ns. A value of 0.1-0.2 will allow for matches that include a small number of Ns.
    #[arg(long, default_value_t = 0.2)]
    max_n_frac: f32,

    /// Number of characters (or lines, for ASCII) to print before/after each match.
    #[arg(short = 'C', long, default_value_t = 20)]
    context: usize,

    /// Number of threads to use. All CPUs by default.
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Force full-record output, even when no output files are given. Aliased to `sassy filter`.
    #[arg(skip)]
    filter: bool,

    /// Only output TSV of matches, hide coloured grep output.
    #[arg(skip)]
    search: bool,

    /// Grep mode. Only disabled when calling `sassy search`.
    #[arg(skip)]
    grep: bool,

    /// TSV output file to write all matches. Empty or "-" for stdout.
    #[arg(long, default_missing_value = "-", num_args(0..=1))]
    matches: Option<PathBuf>,

    /// Filtered output file, otherwise stdout. Must be a directory when multiple input paths are given.
    ///
    /// use "-" for explicit stdout.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    // Positional
    /// Fasta files to search. May be gzipped.
    /// TODO: Support searching multiple files.
    paths: Vec<PathBuf>,
}

impl GrepArgs {
    pub fn grep(mut self) {
        self.grep = true;
        self.run();
    }
    pub fn search(mut self) {
        self.search = true;
        self.run();
    }
    pub fn filter(mut self) {
        self.filter = true;
        self.run();
    }

    fn set_mode(&mut self) {
        // Enable filter mode when output is given.
        if self.output.is_some() {
            self.filter = true;
        }
        if self.invert {
            self.filter = true;
        }

        // Enable search mode when matches is given.
        if self.matches.is_some() {
            self.search = true;
        }
        if self.search && self.matches.is_none() {
            self.matches = Some(PathBuf::from("-"));
        }
    }

    fn run(mut self) {
        self.set_mode();
        if self.paths.is_empty() {
            self.paths = vec![PathBuf::from("")];
        }
        let args = &self;

        let patterns = args.get_patterns();
        assert!(!patterns.is_empty(), "No pattern sequences found");

        let k = args.k;
        let rc =
            (args.alphabet == Alphabet::Dna || args.alphabet == Alphabet::Iupac) && !args.no_rc;

        let num_threads = args.threads.unwrap_or_else(num_cpus::get);
        let task_iterator =
            &InputIterator::new(&args.paths, &patterns, Some(100 * 1024 * 1024), rc);

        let output = Mutex::new((
            0,
            VecDeque::<Option<Vec<(&Path, TextRecord, Vec<(&PatternRecord, Match)>)>>>::new(),
        ));
        let global_histogram = Mutex::new(vec![0usize; k + 1]);

        // Grep writer is stderr
        // Ensure colours are always used.
        colored::control::set_override(true);

        let mut filter_to_stdout = false;
        let mut matches_to_stdout = false;

        // Filter writer
        let filter_writer = self.filter.then(|| {
            let writer = if let Some(output) = &args.output
                && output != ""
                && output != "-"
            {
                let path = PathBuf::from(output);
                Box::new(std::fs::File::create(path).expect("create output file"))
                    as Box<dyn std::io::Write + Send>
            } else {
                filter_to_stdout = true;
                Box::new(std::io::stdout()) as Box<dyn std::io::Write + Send>
            };
            Mutex::new(writer)
        });

        // Match writer
        let match_writer = args.search.then(|| {
            let mut writer = if let Some(output) = &args.matches
                && output != ""
                && output != "-"
            {
                let path = PathBuf::from(output);
                Box::new(std::fs::File::create(path).expect("create matches file"))
                    as Box<dyn std::io::Write + Send>
            } else {
                matches_to_stdout = true;
                Box::new(std::io::stdout()) as Box<dyn std::io::Write + Send>
            };

            if matches_to_stdout && filter_to_stdout {
                panic!("Cannot write both filtered records and matches to stdout");
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
                    let mut searcher = match &args.alphabet {
                        Alphabet::Dna => Either::Left(Searcher::<Dna>::new(rc, args.overhang)),
                        Alphabet::Iupac => Either::Right(Searcher::<Iupac>::new(rc, args.overhang)),
                    };

                    let mut results: Vec<(&Path, TextRecord, Vec<(&PatternRecord, Match)>)> =
                        vec![];
                    let mut local_histogram = vec![0usize; k + 1];

                    while let Some((batch_id, batch)) = task_iterator.next_batch() {
                        let path = batch.0;
                        for text in batch.2 {
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
                                                args.max_n_frac,
                                                &text.seq.text[m.text_start..m.text_end],
                                            )
                                        })
                                        .map(|m| (pattern, m)),
                                );
                            }
                            for (_pat, m) in &batch_matches {
                                local_histogram[m.cost as usize] += 1;
                            }
                            results.push((path, text, batch_matches));
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
                                    text,
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
        eprint!("\nStatistics: ");
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
        text: TextRecord,
        mut matches: Vec<(&PatternRecord, Match)>,
        filter_writer: &mut Option<std::sync::MutexGuard<'_, Box<dyn Write + Send>>>,
        match_writer: &mut Option<std::sync::MutexGuard<'_, Box<dyn Write + Send>>>,
    ) {
        matches.sort_by_key(|m| m.1.text_start);
        // 1. Print filter output.
        if self.filter {
            let writer = &mut **filter_writer.as_mut().unwrap();
            if !self.invert && !matches.is_empty() {
                self.print_matching_record(&text, writer);
            }
            if self.invert && matches.is_empty() {
                self.print_matching_record(&text, writer);
            }
        }

        match (self.grep, self.search) {
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
        if let Some(p) = &self.pattern {
            // Single inline pattern; give it a dummy id
            return vec![PatternRecord {
                id: "pattern".to_string(),
                seq: p.as_bytes().to_vec(),
            }];
        }
        if let Some(pattern_file) = &self.pattern_file {
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
        if let Some(pattern_fasta) = &self.pattern_fasta {
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
            self.pretty_print_match_line(pattern, text, m);
        }
    }

    /// Pretty print the context of a match.
    fn pretty_print_match_line(&self, pattern: &PatternRecord, text: &TextRecord, m: &Match) {
        let rc_pattern;
        let mut cigar = m.cigar.clone();
        let matching_pattern = match m.strand {
            Strand::Fwd => {
                if m.pattern_start > 0 {
                    eprintln!("insert DEL at start");
                    cigar
                        .ops
                        .insert(0, CigarElem::new(CigarOp::Del, m.pattern_start as i32));
                }
                if m.pattern_end < pattern.seq.len() {
                    cigar.ops.push(CigarElem::new(
                        CigarOp::Del,
                        (pattern.seq.len() - m.pattern_end) as i32,
                    ));
                }
                &pattern.seq
            }
            Strand::Rc => {
                rc_pattern = Iupac::reverse_complement(&pattern.seq);
                cigar.reverse();
                let rc_start = pattern.seq.len() - m.pattern_end;
                let rc_end = pattern.seq.len() - m.pattern_start;
                if m.pattern_start > 0 {
                    cigar
                        .ops
                        .insert(0, CigarElem::new(CigarOp::Del, rc_start as i32));
                }
                if rc_end < pattern.seq.len() {
                    cigar.ops.push(CigarElem::new(
                        CigarOp::Del,
                        (pattern.seq.len() - rc_end) as i32,
                    ));
                }
                &rc_pattern[rc_start..rc_end]
            }
        };
        let (matching_text, mut suffix) = text.seq.text.split_at(m.text_end);
        let (mut prefix, matching_text) = matching_text.split_at(m.text_start);

        let mut prefix_skip = 0;
        if prefix.len() > self.context {
            prefix_skip = prefix.len() - self.context;
            prefix = &prefix[prefix_skip..];
        };
        fn format_skip(skip: usize, prefix: bool) -> String {
            if skip > 0 {
                if prefix {
                    format!("{:>9} bp + ", skip)
                } else {
                    format!(" + {:>9} bp", skip)
                }
            } else {
                format!(" {:>9}     ", "")
            }
        }
        let prefix_skip = format_skip(prefix_skip, true);
        let (match_len, match_string) = pretty_print_match(matching_pattern, matching_text, &cigar);

        let suffix_skip =
            (suffix.len() + match_len - matching_pattern.len()) as isize - self.context as isize;
        if suffix_skip > 0 {
            suffix = &suffix[..suffix.len() - suffix_skip as usize];
        };
        let suffix_padding = (-suffix_skip.min(0)) as usize;
        let suffix_skip = format_skip(suffix_skip.max(0) as usize, false);

        let strand = match m.strand {
            Strand::Fwd => "+",
            Strand::Rc => "-",
        };

        let context = self.context;
        eprintln!(
            "{} ({}) {} | {}{:>context$}{match_string}{}{:>suffix_padding$}{} @ {}\n",
            pattern.id,
            strand.bold(),
            format!("{:>2}", m.cost).bold(),
            prefix_skip.dimmed(),
            String::from_utf8_lossy(prefix),
            String::from_utf8_lossy(suffix),
            "",
            suffix_skip.dimmed(),
            format!("{:<19}", format!("{}-{}", m.text_start, m.text_end)).dimmed(),
        );
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
            match self.alphabet {
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

/// Pretty print a match with differences highlighted in color.
fn pretty_print_match(pattern: &[u8], text: &[u8], cigar: &Cigar) -> (usize, String) {
    let mut out = String::new();
    let mut len = 0;
    for pair in cigar.to_char_pairs(pattern, text) {
        len += 1;
        match pair {
            pa_types::CigarOpChars::Match(c) => {
                write!(out, "{}", (c as char).to_string().green().bold()).unwrap()
            }
            pa_types::CigarOpChars::Sub(_old, new) => {
                write!(out, "{}", (new as char).to_string().yellow().bold()).unwrap()
            }
            pa_types::CigarOpChars::Del(c) => {
                write!(out, "{}", (c as char).to_string().red().bold()).unwrap()
            }
            pa_types::CigarOpChars::Ins(c) => {
                write!(out, "{}", (c as char).to_string().cyan().bold()).unwrap()
            }
        }
    }
    (len, out)
}
