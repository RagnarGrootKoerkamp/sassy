use colored::Colorize;
use pa_types::{Cigar, CigarElem, CigarOp, CigarOpChars};
use std::fmt::Write;

use crate::{
    Match, Strand,
    profiles::{Iupac, Profile},
};

/// Pretty print the differences between two strings.
///
/// Returns length and colourized string.
pub fn pretty_print_match(pattern: &[u8], text: &[u8], cigar: &Cigar) -> (usize, String) {
    if cigar.ops.is_empty() {
        return (0, String::new());
    }

    let mut out = String::new();
    let mut len = 0;
    // might work flipping args here that swaps ins/del in pa-types
    // but that's odd
    let char_pairs = cigar.to_char_pairs(text, pattern);
    let prefix_del = match cigar.ops[0] {
        CigarElem {
            op: CigarOp::Ins,
            cnt,
        } => cnt as usize,
        _ => 0,
    };
    let suffix_del = match cigar.ops.last().unwrap() {
        CigarElem {
            op: CigarOp::Ins,
            cnt,
        } => *cnt as usize,
        _ => 0,
    };
    for (i, &pair) in char_pairs.iter().enumerate() {
        len += 1;

        let is_overhang = (i < prefix_del) || (i >= char_pairs.len() - suffix_del);

        match pair {
            CigarOpChars::Match(c) => {
                // matches are not bold
                write!(out, "{}", (c as char).to_string().green()).unwrap()
            }
            CigarOpChars::Sub(_old, new) => {
                write!(out, "{}", (new as char).to_string().yellow().bold()).unwrap()
            }
            CigarOpChars::Del(c) => {
                write!(out, "{}", (c as char).to_string().red().bold()).unwrap()
            }
            CigarOpChars::Ins(c) => {
                // leading and trailing deletions are not bold
                if is_overhang {
                    write!(out, "{}", (c as char).to_string().cyan()).unwrap()
                } else {
                    write!(out, "{}", (c as char).to_string().cyan().bold()).unwrap()
                }
            }
        }
    }
    (len, out)
}

/// The direction to print reverse complement matches.
pub enum PrettyPrintDirection {
    /// Print all matches in the direction of the pattern.
    Pattern,
    /// Print all matches as they appeared in the original text file/input.
    Text,
}

pub enum PrettyPrintStyle {
    Compact,
    Full,
}

impl Match {
    /// Pretty print a `Match` between `pattern` and `text`.
    ///
    /// Returns length and colourized string.
    pub fn pretty_print(
        &self,
        pattern_id: Option<&str>,
        mut _pattern: &[u8],
        mut _text: &[u8],
        dir: PrettyPrintDirection,
        context: usize,
        style: PrettyPrintStyle,
    ) -> String {
        // make a forward version of `self`.
        let mut m = self.clone();
        let rc_pat;
        let rc_text;
        let mut pattern = _pattern;
        let mut text = _text;
        match (self.strand, dir) {
            (Strand::Rc, PrettyPrintDirection::Pattern) => {
                rc_text = Iupac::reverse_complement(text);
                text = &rc_text;
                (m.text_start, m.text_end) =
                    (text.len() - self.text_end, text.len() - self.text_start);
            }
            (Strand::Rc, PrettyPrintDirection::Text) => {
                rc_pat = Iupac::reverse_complement(pattern);
                pattern = &rc_pat;
                (m.pattern_start, m.pattern_end) = (
                    pattern.len() - self.pattern_end,
                    pattern.len() - self.pattern_start,
                );
                m.cigar.reverse();
            }
            (Strand::Fwd, _) => {}
        }

        // Modify the cigar for overhang
        {
            if m.pattern_start > 0 {
                m.cigar
                    .ops
                    .insert(0, CigarElem::new(CigarOp::Ins, m.pattern_start as i32));
            }
            if m.pattern_end < pattern.len() {
                m.cigar.ops.push(CigarElem::new(
                    CigarOp::Ins,
                    (pattern.len() - m.pattern_end) as i32,
                ));
            }
        }

        // Extract the part of the text that matches.
        let (matching_text, mut suffix) = text.split_at(m.text_end);
        let (mut prefix, matching_text) = matching_text.split_at(m.text_start);

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
        let (match_len, match_string) = pretty_print_match(pattern, matching_text, &m.cigar);

        let suffix_skip = (suffix.len() + match_len - pattern.len()) as isize - context as isize;
        if suffix_skip > 0 {
            suffix = &suffix[..suffix.len().saturating_sub(suffix_skip as usize)];
        };
        let suffix_padding = (-suffix_skip.min(0)) as usize;
        let suffix_skip = format_skip(suffix_skip.max(0) as usize, false);

        let strand = match m.strand {
            Strand::Fwd => "+",
            Strand::Rc => "-",
        };

        match style {
            PrettyPrintStyle::Full =>
            // The core matching text.
            // Note: we pass the full pattern, since we already updated the cigar for overhangs.
            // pretty_print_match(&pattern, &text[m.text_start..m.text_end], &m.cigar)
            {
                format!(
                    "{} ({}) {} | {}{:>context$}{}{}{:>suffix_padding$}{} @ {}",
                    pattern_id.unwrap_or(""),
                    strand.bold(),
                    format!("{:>2}", m.cost).bold(),
                    prefix_skip.dimmed(),
                    String::from_utf8_lossy(prefix),
                    match_string,
                    String::from_utf8_lossy(suffix),
                    "",
                    suffix_skip.dimmed(),
                    format!("{:<19}", format!("{}-{}", m.text_start, m.text_end)).dimmed(),
                )
            }
            PrettyPrintStyle::Compact => format!(
                "{} {} | {:>context$}{}{}",
                strand.bold(),
                format!("{:>2}", m.cost).bold(),
                String::from_utf8_lossy(prefix),
                match_string,
                String::from_utf8_lossy(suffix),
            ),
        }
    }
}
