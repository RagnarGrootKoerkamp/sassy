use std::{collections::VecDeque, fmt::Write, path::PathBuf, sync::Mutex};

use colored_text::Colorize;
use pa_types::Cigar;
use sassy::{
    Match, Searcher, Strand,
    profiles::{Ascii, Dna, Iupac, Profile},
};

use crate::{
    input_iterator::{InputIterator, PatternRecord, Task, TextRecord},
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

    /// The alphabet to use. DNA=ACTG, or default IUPAC=ACTG+NYR... -a for ASCII.
    /// TODO: Infer alphabet from input file types.
    #[arg(
        long,
        short = 'a',
        default_value_t = Alphabet::Iupac,
        default_missing_value = "Alphabet::Ascii",
        value_enum
    )]
    alphabet: Alphabet,

    /// Enable overhang alignment.
    #[arg(long)]
    overhang: Option<f32>,

    /// Disable reverse complement search (enabled by default for DNA and IUPAC).
    #[arg(long)]
    no_rc: bool,

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
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    // Positional
    /// Fasta files to search. May be gzipped.
    /// TODO: Support searching multiple files.
    path: PathBuf,
}

impl GrepArgs {
    pub fn filter(mut self) {
        self.filter = true;
        self.grep();
    }
    pub fn grep(mut self) {
        self.set_mode();
        self.set_context();
        assert!(!self.filter);
        let args = &self;

        // 1. iterate over files
        // 2. iterate over records in file
        // 3. iterate over patterns

        // Copied from search::search

        let patterns = get_patterns(&args.pattern, &args.pattern_file, &args.pattern_fasta);
        assert!(!patterns.is_empty(), "No pattern sequences found");

        let k = args.k;
        let rc =
            (args.alphabet == Alphabet::Dna || args.alphabet == Alphabet::Iupac) && !args.no_rc;

        let num_threads = args.threads.unwrap_or_else(num_cpus::get);
        let task_iterator = &InputIterator::new(&args.path, &patterns, Some(100 * 1024 * 1024), rc);

        let output = Mutex::new((0, VecDeque::<Option<Vec<(Task<'_>, Vec<Match>)>>>::new()));
        let global_histogram = Mutex::new(vec![0usize; k + 1]);

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let output = &output;
                let global_histogram = &global_histogram;
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

                    let mut results: Vec<(Task<'_>, Vec<Match>)> = vec![];
                    let mut local_histogram = vec![0usize; k + 1];

                    while let Some((batch_id, batch)) = task_iterator.next_batch() {
                        for item in batch {
                            let matches = searcher.search(&item.pattern.seq, &item.text.seq, k);
                            for m in &matches {
                                local_histogram[m.cost as usize] += 1;
                            }
                            results.push((item, matches));
                        }

                        // Wait until next_batch_id equals batch_id.
                        let (next_batch_id, output_buf) = &mut *output.lock().unwrap();

                        // Push to buffer.
                        let idx = batch_id - *next_batch_id;
                        if output_buf.len() <= idx {
                            output_buf.resize(idx + 1, None);
                        }
                        assert!(output_buf[idx].is_none());
                        output_buf[idx] = Some(results);
                        results = vec![];

                        // Print pending results.
                        while let Some(front) = output_buf.front()
                            && front.is_some()
                        {
                            *next_batch_id += 1;
                            let front_results = output_buf.pop_front().unwrap().unwrap();
                            for (item, mut matches) in front_results {
                                args.print_matches_for_record(
                                    &item.pattern,
                                    &item.text,
                                    &mut matches,
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
        eprintln!("\nStatistics:");
        eprintln!("Dist  {:>10}", "Count");
        for (dist, &count) in global_hist.iter().enumerate() {
            eprintln!("{:>4}: {:>10}", dist, count);
        }

        assert!(output.into_inner().unwrap().1.is_empty());
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

    fn print_matches_for_record(
        &self,
        pattern: &PatternRecord,
        text: &TextRecord,
        matches: &mut Vec<Match>,
    ) {
        if matches.is_empty() {
            return;
        }
        println!("{}", format!(">{}", text.id).bold());
        matches.sort_by_key(|m| m.text_start);
        for match_record in matches {
            let line = self.pretty_print_match_line(pattern, text, match_record);
            print!("{}", line);
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
        let matching_pattern = match m.strand {
            Strand::Fwd => &pattern.seq[m.pattern_start..m.pattern_end],
            Strand::Rc => {
                rc_pattern = Iupac::reverse_complement(&pattern.seq);
                m.cigar.reverse();
                &rc_pattern[pattern.seq.len() - m.pattern_end..pattern.seq.len() - m.pattern_start]
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
        let (match_len, match_string) =
            pretty_print_match(matching_pattern, matching_text, &m.cigar);

        let mut suffix_skip = 0;
        if (suffix.len() + match_len).saturating_sub(matching_pattern.len()) > context {
            suffix_skip = suffix.len() + match_len - matching_pattern.len() - context;
            suffix = &suffix[..suffix.len() - suffix_skip];
        };
        let suffix_skip = format_skip(suffix_skip, false);

        format!(
            "{} {} d={} @ {}-{}\n{}{}{}{}{}\n",
            match m.strand {
                Strand::Fwd => "fwd",
                Strand::Rc => "rc ",
            }
            .bold(),
            pattern.id,
            m.cost.bold(),
            m.text_start.dim(),
            m.text_end.dim(),
            prefix_skip.dim(),
            String::from_utf8_lossy(prefix),
            match_string,
            String::from_utf8_lossy(suffix),
            suffix_skip.dim()
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
