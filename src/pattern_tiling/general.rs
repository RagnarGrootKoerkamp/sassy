use crate::n_filter::traced_satisfy_n_frac;
use crate::pattern_tiling::backend::SimdBackend;
use crate::pattern_tiling::minima::TracePostProcess;
use crate::pattern_tiling::search::{HitRange, Myers};
use crate::pattern_tiling::tqueries::TQueries;
use crate::pattern_tiling::trace::TraceBuffer;
use crate::pattern_tiling::trace::trace_batch_ranges;
use crate::profiles::Profile;
use crate::search::{Match, RcSearchAble, Strand};

// If AVX-512 is available, we want to use the 512-bit backends instead of the default 256-bit ones.
// This allows processing twice as many patterns in parallel.

pub use crate::pattern_tiling::backend::*;

// Even with macros to generate most of the code this is still A LOT to look at
macro_rules! dispatch_encoded {
    ($self:ident, $encoded:expr, |$searcher:ident, $tq:ident| $body:expr) => {
        match $encoded {
            EncodedPatterns::U8($tq) => {
                let $searcher = &mut $self.searcher_u8;
                $body
            }
            EncodedPatterns::U16 { full: $tq, .. } => {
                let $searcher = &mut $self.searcher_u16;
                $body
            }
            EncodedPatterns::U32 { full: $tq, .. } => {
                let $searcher = &mut $self.searcher_u32;
                $body
            }
            EncodedPatterns::U64 { full: $tq, .. } => {
                let $searcher = &mut $self.searcher_u64;
                $body
            }
            EncodedPatterns::Both { .. } => unreachable!("search one strand at a time"),
        }
    };
}

macro_rules! run_hierarchical {
    ($self:ident, $full_tq:expr, $suffix_opt:expr, $prefilter_searcher:ident, $full_searcher:ident, $text:expr, $k:expr, $post:expr, $max_n_frac:expr, $out:expr, $suffix_name:expr) => {{
        let suffix_tq = $suffix_opt
            .as_ref()
            .expect(concat!($suffix_name, " suffix should be pre-encoded"));
        hierarchical_search(
            &mut $self.$prefilter_searcher,
            suffix_tq,
            &mut $self.$full_searcher,
            $full_tq,
            $text,
            $k,
            $post,
            $max_n_frac,
            $out,
            &mut $self.trace_buffer,
        )
    }};
}

#[allow(clippy::too_many_arguments)]
pub fn hierarchical_search<S, F, P: Profile>(
    suffix_searcher: &mut Myers<S, P>,
    suffix_tqueries: &TQueries<S, P>,
    full_searcher: &mut Myers<F, P>,
    full_tqueries: &TQueries<F, P>,
    text: &[u8],
    k: u32,
    post: TracePostProcess<'_>,
    max_n_frac: Option<f32>,
    out: &mut Vec<Match>,
    trace_buffer: &mut TraceBuffer,
) where
    S: SimdBackend,
    F: SimdBackend,
{
    let pattern_len = full_tqueries.pattern_length;

    let suffix_ranges = suffix_searcher.search_ranges(suffix_tqueries, text, k, true, true);
    let filtered_ranges;
    let suffix_ranges = if let Some(max_n_frac) = max_n_frac {
        filtered_ranges =
            filter_endpoint_ranges(suffix_ranges, text, pattern_len, k as usize, max_n_frac);
        &filtered_ranges
    } else {
        suffix_ranges
    };

    full_searcher.ensure_capacity(full_tqueries.n_simd_blocks, full_tqueries.n_queries);
    full_searcher.search_prep(
        full_tqueries.n_simd_blocks,
        full_tqueries.n_queries,
        full_tqueries.pattern_length,
        full_searcher.alpha_pattern,
    );

    for chunk_start in (0..suffix_ranges.len()).step_by(F::LANES) {
        let chunk_end: usize = (chunk_start + F::LANES).min(suffix_ranges.len());
        let batch_slice = &suffix_ranges[chunk_start..chunk_end];

        trace_batch_ranges(
            full_searcher,
            full_tqueries,
            text,
            batch_slice,
            k,
            post,
            max_n_frac,
            Some(full_searcher.alpha),
            full_searcher.max_overhang,
            trace_buffer,
        );
        out.append(&mut trace_buffer.filtered_alignments);
    }
}

fn filter_endpoint_ranges(
    ranges: &[HitRange],
    text: &[u8],
    pattern_len: usize,
    k: usize,
    max_n_frac: f32,
) -> Vec<HitRange> {
    assert!(max_n_frac < 1.0, "max_n_frac must reject all-N matches");
    let lookbehind = pattern_len + k;
    let mut filtered = Vec::new();
    for range in ranges {
        let scan_start = (range.start.max(0) as usize).min(text.len());
        let scan_end = ((range.end + 1).max(0) as usize).min(text.len());
        let mut start = range.start;
        let mut offset = scan_start;
        while offset < scan_end {
            let Some(first_n) = memchr::memchr2(b'N', b'n', &text[offset..scan_end]) else {
                break;
            };
            let run_start = offset + first_n;
            let run_end = text[run_start..scan_end]
                .iter()
                .position(|&b| b | (b'N' ^ b'n') != b'N' | (b'N' ^ b'n'))
                .map_or(scan_end, |len| run_start + len);
            offset = run_end;

            // Keep the first pattern_len + k inclusive positions of the run as a buffer.
            let gap_start = run_start + lookbehind;
            if gap_start < run_end {
                let gap_start = gap_start as isize;
                if start < gap_start {
                    filtered.push(HitRange {
                        pattern_idx: range.pattern_idx,
                        start,
                        end: gap_start - 1,
                    });
                }
                start = run_end as isize;
            }
        }
        if start <= range.end {
            filtered.push(HitRange {
                pattern_idx: range.pattern_idx,
                start,
                end: range.end,
            });
        }
    }
    filtered
}

#[inline(always)]
fn trace_ranges_backend<S: SimdBackend, P: Profile>(
    searcher: &mut Myers<S, P>,
    tqueries: &TQueries<S, P>,
    text: &[u8],
    ranges: &[HitRange],
    k: u32,
    post: TracePostProcess<'_>,
    max_n_frac: Option<f32>,
    out: &mut Vec<Match>,
    trace_buffer: &mut TraceBuffer,
) {
    for chunk in ranges.chunks(S::LANES) {
        trace_batch_ranges(
            searcher,
            tqueries,
            text,
            chunk,
            k,
            post,
            max_n_frac,
            Some(searcher.alpha),
            searcher.max_overhang,
            trace_buffer,
        );
        out.append(&mut trace_buffer.filtered_alignments);
    }
}

#[derive(Debug, Clone)]
pub(crate) enum EncodedPatterns<P: Profile> {
    U8(TQueries<U8Backend, P>),
    /// Independent encodings for the forward and reversed text.
    Both {
        forward: Box<Self>,
        reverse: Box<Self>,
    },
    U16 {
        full: TQueries<U16Backend, P>,
        suffix_u8: Option<Box<TQueries<U8Backend, P>>>,
    },
    U32 {
        full: TQueries<U32Backend, P>,
        suffix_u16: Option<Box<TQueries<U16Backend, P>>>,
        suffix_u8: Option<Box<TQueries<U8Backend, P>>>,
    },
    U64 {
        full: TQueries<U64Backend, P>,
        suffix_u16: Option<Box<TQueries<U16Backend, P>>>,
        suffix_u32: Option<Box<TQueries<U32Backend, P>>>,
        suffix_u8: Option<Box<TQueries<U8Backend, P>>>,
    },
}

enum PrefilterBackend {
    U8,
    U16,
    U32,
}

pub struct Searcher<P: Profile> {
    searcher_u8: Myers<U8Backend, P>,
    searcher_u16: Myers<U16Backend, P>,
    searcher_u32: Myers<U32Backend, P>,
    searcher_u64: Myers<U64Backend, P>,
    suffix_searcher_u8: Myers<U8Backend, P>,
    suffix_searcher_u16: Myers<U16Backend, P>,
    suffix_searcher_u32: Myers<U32Backend, P>,
    trace_buffer: TraceBuffer,
}

impl<P: Profile> Clone for Searcher<P> {
    fn clone(&self) -> Self {
        Self {
            searcher_u8: Myers::new(Some(self.searcher_u8.alpha)),
            searcher_u16: Myers::new(Some(self.searcher_u16.alpha)),
            searcher_u32: Myers::new(Some(self.searcher_u32.alpha)),
            searcher_u64: Myers::new(Some(self.searcher_u64.alpha)),
            suffix_searcher_u8: Myers::new(Some(self.suffix_searcher_u8.alpha)),
            suffix_searcher_u16: Myers::new(Some(self.suffix_searcher_u16.alpha)),
            suffix_searcher_u32: Myers::new(Some(self.suffix_searcher_u32.alpha)),
            trace_buffer: TraceBuffer::new(64),
        }
    }
}

impl<P: Profile> Searcher<P> {
    pub fn new(alpha: Option<f32>) -> Self {
        Self {
            searcher_u8: Myers::new(alpha),
            searcher_u16: Myers::new(alpha),
            searcher_u32: Myers::new(alpha),
            searcher_u64: Myers::new(alpha),
            suffix_searcher_u8: Myers::new(alpha),
            suffix_searcher_u16: Myers::new(alpha),
            suffix_searcher_u32: Myers::new(alpha),
            trace_buffer: TraceBuffer::new(64),
        }
    }

    pub fn encode(&self, queries: &[impl AsRef<[u8]>], include_rc: bool) -> EncodedPatterns<P> {
        let forward = Self::encode_strand(queries);
        if include_rc {
            // As in v1, complement the patterns and reuse the reversed text.
            let complements: Vec<_> = queries.iter().map(|q| P::complement(q.as_ref())).collect();
            EncodedPatterns::Both {
                forward: Box::new(forward),
                reverse: Box::new(Self::encode_strand(&complements)),
            }
        } else {
            forward
        }
    }

    fn encode_strand(queries: &[impl AsRef<[u8]>]) -> EncodedPatterns<P> {
        //Fixme: neat to encode all possible suffixes so every k is supported when
        // search is called, it is wasteful when user never searches different k's
        if queries.is_empty() {
            panic!("No queries provided");
        }

        let max_pattern_length = queries.iter().map(|q| q.as_ref().len()).max().unwrap();

        // all queries should be the same max length
        assert!(
            queries
                .iter()
                .all(|q| q.as_ref().len() == max_pattern_length),
            "All pattern must have the same length"
        );

        if max_pattern_length <= U8Backend::LIMB_BITS {
            EncodedPatterns::U8(TQueries::<U8Backend, P>::new(queries))
        } else if max_pattern_length <= U16Backend::LIMB_BITS {
            let full = TQueries::<U16Backend, P>::new(queries);
            EncodedPatterns::U16 {
                suffix_u8: Some(Box::new(full.reduce_to_suffix::<U8Backend, P>())),
                full,
            }
        } else if max_pattern_length <= U32Backend::LIMB_BITS {
            let full = TQueries::new(queries);
            EncodedPatterns::U32 {
                suffix_u16: Some(Box::new(full.reduce_to_suffix::<U16Backend, P>())),
                suffix_u8: Some(Box::new(full.reduce_to_suffix::<U8Backend, P>())),
                full,
            }
        } else if max_pattern_length <= U64Backend::LIMB_BITS {
            let full = TQueries::new(queries);
            EncodedPatterns::U64 {
                suffix_u16: Some(Box::new(full.reduce_to_suffix::<U16Backend, P>())),
                suffix_u32: Some(Box::new(full.reduce_to_suffix::<U32Backend, P>())),
                suffix_u8: Some(Box::new(full.reduce_to_suffix::<U8Backend, P>())),
                full,
            }
        } else {
            panic!(
                "pattern length {} exceeds maximum supported length {}",
                max_pattern_length,
                U64Backend::LIMB_BITS
            );
        }
    }

    fn should_use_hierarchical(
        encoded_queries: &EncodedPatterns<P>,
        k: u32,
        use_hierarchical: Option<bool>,
    ) -> Option<PrefilterBackend> {
        if use_hierarchical != Some(true) {
            return None;
        }
        // Based on emperical benchmarks
        match encoded_queries {
            EncodedPatterns::U8(_) => None,
            EncodedPatterns::U16 { .. } if k == 0 => Some(PrefilterBackend::U8),
            EncodedPatterns::U32 { .. } if k == 0 => Some(PrefilterBackend::U8),
            EncodedPatterns::U32 { .. } if k < 4 => Some(PrefilterBackend::U16),
            EncodedPatterns::U64 { .. } if k == 0 => Some(PrefilterBackend::U8),
            EncodedPatterns::U64 { .. } if k < 4 => Some(PrefilterBackend::U16),
            EncodedPatterns::U64 { .. } if k < 8 => Some(PrefilterBackend::U32),
            _ => None,
        }
    }

    #[rustfmt::skip]
    fn hierarchical_search_with_prefilter(
        &mut self,
        full_queries: &EncodedPatterns<P>,
        text: &[u8],
        k: u32,
        prefilter: PrefilterBackend,
        post: TracePostProcess<'_>,
        max_n_frac: Option<f32>,
        out: &mut Vec<Match>,
    ) {
        match (prefilter, full_queries) {
            (PrefilterBackend::U8, EncodedPatterns::U16 { full, suffix_u8 }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u16, text, k, post, max_n_frac, out, "U8"),
            (PrefilterBackend::U8, EncodedPatterns::U32 { full, suffix_u8, .. }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u32, text, k, post, max_n_frac, out, "U8"),
            (PrefilterBackend::U8, EncodedPatterns::U64 { full, suffix_u8, .. }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u64, text, k, post, max_n_frac, out, "U8"),
            (PrefilterBackend::U16, EncodedPatterns::U32 { full, suffix_u16, .. }) => run_hierarchical!(self, full, suffix_u16, suffix_searcher_u16, searcher_u32, text, k, post, max_n_frac, out, "U16"),
            (PrefilterBackend::U16, EncodedPatterns::U64 { full, suffix_u16, .. }) => run_hierarchical!(self, full, suffix_u16, suffix_searcher_u16, searcher_u64, text, k, post, max_n_frac, out, "U16"),
            (PrefilterBackend::U32, EncodedPatterns::U64 { full, suffix_u32, .. }) => run_hierarchical!(self, full, suffix_u32, suffix_searcher_u32, searcher_u64, text, k, post, max_n_frac, out, "U32"),
            _ => panic!("Invalid prefilter backend combination"),
        }
    }

    #[cfg(test)]
    pub fn search<I: RcSearchAble + ?Sized>(
        &mut self,
        encoded_queries: &EncodedPatterns<P>,
        text: &I,
        k: u32,
        max_n_frac: Option<f32>,
    ) -> Vec<Match> {
        self.search_with_options(
            encoded_queries,
            text,
            k,
            Some(true),
            TracePostProcess::LocalMinima,
            max_n_frac,
        )
    }

    #[cfg(test)]
    pub fn search_all<I: RcSearchAble + ?Sized>(
        &mut self,
        encoded_queries: &EncodedPatterns<P>,
        text: &I,
        k: u32,
        max_n_frac: Option<f32>,
    ) -> Vec<Match> {
        self.search_with_options(
            encoded_queries,
            text,
            k,
            Some(true),
            TracePostProcess::All(None),
            max_n_frac,
        )
    }

    #[cfg(test)]
    pub fn search_with_options<I: RcSearchAble + ?Sized>(
        &mut self,
        encoded_queries: &EncodedPatterns<P>,
        text: &I,
        k: u32,
        use_hierarchical: Option<bool>,
        post: TracePostProcess<'_>,
        max_n_frac: Option<f32>,
    ) -> Vec<Match> {
        let mut matches = Vec::new();
        self.search_record(
            encoded_queries,
            text,
            k,
            use_hierarchical,
            post,
            max_n_frac,
            &mut matches,
        );
        matches
    }

    pub(crate) fn search_record<I: RcSearchAble + ?Sized>(
        &mut self,
        encoded_queries: &EncodedPatterns<P>,
        text: &I,
        k: u32,
        use_hierarchical: Option<bool>,
        post: TracePostProcess<'_>,
        max_n_frac: Option<f32>,
        out: &mut Vec<Match>,
    ) {
        let first = out.len();
        let forward_text = text.text();
        let forward_text = forward_text.as_ref();
        if forward_text.is_empty() {
            return;
        }
        if let EncodedPatterns::Both { forward, reverse } = encoded_queries {
            self.search_strand(
                forward,
                forward_text,
                k,
                use_hierarchical,
                post,
                max_n_frac,
                out,
            );
            let start = out.len();
            let reversed_text = text.rev_text();
            self.search_strand(
                reverse,
                reversed_text.as_ref(),
                k,
                use_hierarchical,
                post,
                max_n_frac,
                out,
            );
            for m in &mut out[start..] {
                (m.text_start, m.text_end) = (
                    forward_text.len() - m.text_end,
                    forward_text.len() - m.text_start,
                );
                m.strand = Strand::Rc;
            }
        } else {
            self.search_strand(
                encoded_queries,
                forward_text,
                k,
                use_hierarchical,
                post,
                max_n_frac,
                out,
            );
        }
        if let Some(max_n_frac) = max_n_frac {
            // Compact this record's matches without revisiting earlier records.
            let mut write = first;
            for read in first..out.len() {
                if traced_satisfy_n_frac(&out[read], forward_text, max_n_frac) {
                    out.swap(write, read);
                    write += 1;
                }
            }
            out.truncate(write);
        }
    }

    fn search_strand(
        &mut self,
        encoded_queries: &EncodedPatterns<P>,
        text: &[u8],
        k: u32,
        use_hierarchical: Option<bool>,
        post: TracePostProcess<'_>,
        max_n_frac: Option<f32>,
        out: &mut Vec<Match>,
    ) {
        if let Some(prefilter) = Self::should_use_hierarchical(encoded_queries, k, use_hierarchical)
        {
            self.hierarchical_search_with_prefilter(
                encoded_queries,
                text,
                k,
                prefilter,
                post,
                max_n_frac,
                out,
            );
        } else {
            dispatch_encoded!(self, encoded_queries, |searcher, tq| {
                // Find matching ranges
                let ranges = searcher.search_ranges(tq, text, k, true, true);
                let ranges = if let Some(max_n_frac) = max_n_frac {
                    filter_endpoint_ranges(ranges, text, tq.pattern_length, k as usize, max_n_frac)
                } else {
                    ranges.to_vec()
                };
                // Trace each of the ranges, appending to the output.
                trace_ranges_backend(
                    searcher,
                    tq,
                    text,
                    ranges.as_slice(),
                    k,
                    post,
                    max_n_frac,
                    out,
                    &mut self.trace_buffer,
                );
            });
        }
    }
}

impl<P: Profile> Default for Searcher<P> {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::Iupac;

    #[test]
    fn endpoint_ranges_split_only_at_all_n_windows() {
        let text = b"AAAAAAAAAAAAAAAANNNNnnnnNNNNnnnnAAAAAAAA";
        let ranges = [
            HitRange {
                pattern_idx: 0,
                start: 0,
                end: text.len() as isize - 1,
            },
            HitRange {
                pattern_idx: 1,
                start: 25,
                end: 31,
            },
        ];
        let filtered = filter_endpoint_ranges(&ranges, text, 4, 1, 0.0);
        let bounds: Vec<_> = filtered.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(bounds, [(0, 20), (32, 39), (25, 29)]);
    }

    #[test]
    fn hierarchical_search_preserves_matches_immediately_outside_a_long_n_run() {
        // An all-A pattern ties with shifted endpoints inside the N run.
        let pattern = b"AAAAAAAAAAAAAAAAAAAA";
        let mut text = vec![b'N'; 100_000];
        text[..pattern.len()].copy_from_slice(pattern);
        text.extend_from_slice(pattern);

        let mut searcher = Searcher::<Iupac>::new(None);
        let encoded = searcher.encode(&[pattern.to_vec()], false);
        let matches = searcher.search(&encoded, &text, 0, Some(0.0));
        let spans: Vec<_> = matches.iter().map(|m| m.text_start..m.text_end).collect();
        assert_eq!(spans, [0..pattern.len(), 100_000..text.len()]);
    }

    #[test]
    fn hierarchical_search_preserves_matches_overlapping_both_edges_of_an_n_run() {
        let pattern = b"ACGTACGTACGTACGTACGT";
        let overlap = 2;
        let mut text = pattern[..pattern.len() - overlap].to_vec();
        text.resize(text.len() + 256, b'N');
        let right_start = text.len() - overlap;
        text.extend_from_slice(&pattern[overlap..]);

        let max_n_frac = Some(overlap as f32 / pattern.len() as f32);
        let mut searcher = Searcher::<Iupac>::new(None);
        let encoded = searcher.encode(&[pattern.to_vec()], false);
        let expected = [0..pattern.len(), right_start..text.len()];
        let matches = searcher.search(&encoded, &text, 0, max_n_frac);
        let spans: Vec<_> = matches.iter().map(|m| m.text_start..m.text_end).collect();
        assert_eq!(spans, expected);

        let matches = searcher.search_all(&encoded, &text, 0, max_n_frac);
        let spans: Vec<_> = matches.iter().map(|m| m.text_start..m.text_end).collect();
        assert_eq!(spans, expected);
    }

    #[test]
    fn all_matches_exclude_only_the_all_n_window_at_high_n_fraction() {
        let mut searcher = Searcher::<Iupac>::new(None);
        let encoded = searcher.encode(&[b"AAAA".to_vec()], false);
        let text = b"AAAANNNN";
        let max_n_frac = Some(0.9);

        let matches = searcher.search_all(&encoded, text, 0, max_n_frac);
        let spans: Vec<_> = matches.iter().map(|m| m.text_start..m.text_end).collect();
        assert_eq!(spans, [0..4, 1..5, 2..6, 3..7]);

        let matches = searcher.search(&encoded, text, 0, max_n_frac);
        let spans: Vec<_> = matches.iter().map(|m| m.text_start..m.text_end).collect();
        assert_eq!(spans, [3..7]);
    }

    #[test]
    fn hierarchical_search_rejects_all_n_matches_at_a_permissive_threshold() {
        let mut searcher = Searcher::<Iupac>::new(None);
        let encoded = searcher.encode(&[vec![b'A'; 20]], false);
        // This threshold can admit an all-N U16 suffix, but the full match is all Ns.
        let matches = searcher.search_all(&encoded, &[b'N'; 40], 3, Some(13.0 / 23.0));
        assert!(matches.is_empty());
    }
}
