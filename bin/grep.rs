use std::{
    collections::VecDeque,
    fmt::Write,
    path::{Path, PathBuf},
    sync::Mutex,
};

use colored_text::Colorize;
use pa_types::{Cigar, CigarElem, CigarOp};
use sassy::{
    Match, Searcher, Strand,
    profiles::{Ascii, Dna, Iupac, Profile},
};

use crate::{
    crispr::check_n_frac,
    input_iterator::{InputIterator, PatternRecord, TextRecord},
    search::{Alphabet, SearchWrapper, get_patterns},
};

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
        // default_missing_value = "Alphabet::Ascii",
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
    #[arg(short = 'C', long)]
    context: Option<usize>,

    /// Number of threads to use. All CPUs by default.
    #[arg(short = 'j', long)]
    threads: Option<usize>,

    /// Force full-record output, even when no output files are given. Aliased to `sassy filter`.
    #[arg(long)]
    filter: bool,

    /// Output file, otherwise stdout. Must be a directory when multiple input paths are given.
    ///
    /// - for explicit stdout.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    // Positional
    /// Fasta files to search. May be gzipped.
    /// TODO: Support searching multiple files.
    paths: Vec<PathBuf>,
}

impl GrepArgs {
    pub fn filter(mut self) {
        self.filter = true;
        self.grep();
    }

    fn set_mode(&mut self) {
        if self.output.is_some() {
            self.filter = true;
        }
        if self.invert {
            self.filter = true;
        }
    }

    fn set_context(&mut self) {
        if self.context.is_none() {
            if self.alphabet == Alphabet::Ascii {
                self.context = Some(0);
            } else {
                self.context = Some(20);
            }
        }
    }

    pub fn grep(mut self) {
        self.set_mode();
        self.set_context();
        if self.paths.is_empty() {
            self.paths = vec![PathBuf::from("")];
        }
        let args = &self;

        if args.alphabet == Alphabet::Ascii {
            unimplemented!(
                "ASCII alphabet is not yet supported. Please comment on https://github.com/RagnarGrootKoerkamp/sassy/issues/35 if you're interested."
            );
        }

        let patterns = get_patterns(&args.pattern, &args.pattern_file, &args.pattern_fasta);
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

        let writer = if let Some(output) = &args.output
            && output != "-"
        {
            let path = PathBuf::from(output);
            Box::new(std::fs::File::create(path).expect("create output file"))
                as Box<dyn std::io::Write + Send>
        } else {
            Box::new(std::io::stdout()) as Box<dyn std::io::Write + Send>
        };
        let writer = Mutex::new(writer);

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let output = &output;
                let global_histogram = &global_histogram;
                let writer = &writer;
                s.spawn(move || {
                    // Each thread has own searcher here
                    let mut searcher: SearchWrapper = match &args.alphabet {
                        Alphabet::Ascii => SearchWrapper::Ascii(Box::new(Searcher::<Ascii>::new(
                            false,
                            args.overhang,
                        ))),
                        Alphabet::Dna => {
                            SearchWrapper::Dna(Box::new(Searcher::<Dna>::new(rc, args.overhang)))
                        }
                        Alphabet::Iupac => SearchWrapper::Iupac(Box::new(Searcher::<Iupac>::new(
                            rc,
                            args.overhang,
                        ))),
                    };

                    let mut results: Vec<(&Path, TextRecord, Vec<(&PatternRecord, Match)>)> =
                        vec![];
                    let mut local_histogram = vec![0usize; k + 1];

                    while let Some((batch_id, batch)) = task_iterator.next_batch() {
                        let path = batch.0;
                        for text in batch.2 {
                            let mut matches = vec![];
                            for pattern in batch.1 {
                                matches.extend(
                                    searcher
                                        .search(&pattern.seq, &text.seq, k)
                                        .into_iter()
                                        .filter(|m| {
                                            // Check max-n-frac.
                                            check_n_frac(
                                                args.max_n_frac,
                                                &text.seq.text[m.text_start..m.text_end],
                                            )
                                        })
                                        .map(|m| (pattern, m)),
                                );
                            }
                            for (_pat, m) in &matches {
                                local_histogram[m.cost as usize] += 1;
                            }
                            results.push((path, text, matches));
                        }

                        let (next_batch_id, output_buf) = &mut *output.lock().unwrap();
                        let mut writer = if self.filter {
                            Some(writer.lock().unwrap())
                        } else {
                            None
                        };

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
                            for (path, text, mut matches) in front_results {
                                if self.filter {
                                    let writer = &mut **writer.as_mut().unwrap();
                                    if !self.invert && !matches.is_empty() {
                                        args.print_matching_record(&text, writer);
                                    }
                                    if self.invert && matches.is_empty() {
                                        args.print_matching_record(&text, writer);
                                    }
                                } else {
                                    args.print_matches_for_record(path, &text, &mut matches);
                                }
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
                eprint!("dist {} => {}, ", dist.bold(), count.bold());
            }
        }
        eprintln!();

        assert!(output.into_inner().unwrap().1.is_empty());
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
        matches: &mut Vec<(&PatternRecord, Match)>,
    ) {
        if matches.is_empty() {
            return;
        }
        eprintln!(
            "{}",
            format!("{}>{}", path.display().cyan().bold(), text.id.bold()).bold()
        );
        matches.sort_by_key(|m| m.1.text_start);
        for (pattern, match_record) in matches {
            let line = self.pretty_print_match_line(pattern, text, match_record);
            eprint!("{}", line);
        }
    }

    /// Pretty print the context of a match.
    fn pretty_print_match_line(
        &self,
        pattern: &PatternRecord,
        text: &TextRecord,
        m: &mut Match,
    ) -> String {
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

        let context = self.context.unwrap();
        let mut prefix_skip = 0;
        if prefix.len() > context {
            prefix_skip = prefix.len() - context;
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
            (suffix.len() + match_len - matching_pattern.len()) as isize - context as isize;
        if suffix_skip > 0 {
            suffix = &suffix[..suffix.len() - suffix_skip as usize];
        };
        let suffix_padding = (-suffix_skip.min(0)) as usize;
        let suffix_skip = format_skip(suffix_skip.max(0) as usize, false);

        let strand = match m.strand {
            Strand::Fwd => "+",
            Strand::Rc => "-",
        };

        format!(
            "{} ({}) {} | {}{:>context$}{match_string}{}{:>suffix_padding$}{} @ {}\n",
            pattern.id,
            strand.bold(),
            format!("{:>2}", m.cost).bold(),
            prefix_skip.dim(),
            String::from_utf8_lossy(prefix),
            String::from_utf8_lossy(suffix),
            "",
            suffix_skip.dim(),
            format!("{:<19}", format!("{}-{}", m.text_start, m.text_end)).dim(),
        )
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
                write!(out, "{}", (c as char).green().bold()).unwrap()
            }
            pa_types::CigarOpChars::Sub(_old, new) => {
                write!(out, "{}", (new as char).yellow().bold()).unwrap()
            }
            pa_types::CigarOpChars::Del(c) => write!(out, "{}", (c as char).red().bold()).unwrap(),
            pa_types::CigarOpChars::Ins(c) => write!(out, "{}", (c as char).cyan().bold()).unwrap(),
        }
    }
    (len, out)
}
