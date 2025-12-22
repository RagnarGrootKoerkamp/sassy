use crate::pattern_tiling::backend::{SimdBackend, U8, U16, U32, U64};
use crate::pattern_tiling::search::{HitRange, Myers};
use crate::pattern_tiling::tqueries::TQueries;
use crate::pattern_tiling::trace::{TracePostProcess, trace_batch_ranges};
use crate::search::Match;
use pa_types::Cost;

// Even with macros to generate most of the code this is still A LOT to look at
macro_rules! dispatch_encoded {
    ($self:ident, $encoded:expr, |$searcher:ident, $tq:ident| $body:expr) => {
        match $encoded {
            EncodedQueries::U8($tq) => {
                let $searcher = &mut $self.searcher_u8;
                $body
            }
            EncodedQueries::U16 { full: $tq, .. } => {
                let $searcher = &mut $self.searcher_u16;
                $body
            }
            EncodedQueries::U32 { full: $tq, .. } => {
                let $searcher = &mut $self.searcher_u32;
                $body
            }
            EncodedQueries::U64 { full: $tq, .. } => {
                let $searcher = &mut $self.searcher_u64;
                $body
            }
        }
    };
}

macro_rules! run_hierarchical {
    ($self:ident, $full_tq:expr, $suffix_opt:expr, $prefilter_searcher:ident, $full_searcher:ident, $text:expr, $k:expr, $post:expr, $suffix_name:expr) => {{
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
            &mut $self.alignments_buf,
            &mut $self.batch_buf,
        )
    }};
}

#[allow(clippy::too_many_arguments)]
pub fn hierarchical_search<S, F>(
    suffix_searcher: &mut Myers<S>,
    suffix_tqueries: &TQueries<S>,
    full_searcher: &mut Myers<F>,
    full_tqueries: &TQueries<F>,
    text: &[u8],
    k: u32,
    post: TracePostProcess,
    out: &mut Vec<Match>,
    batch_buf: &mut Vec<Match>,
) where
    S: SimdBackend,
    F: SimdBackend,
{
    let suffix_ranges = suffix_searcher.search_ranges(suffix_tqueries, text, k, true, true);

    full_searcher.ensure_capacity(full_tqueries.n_simd_blocks, full_tqueries.n_queries);
    full_searcher.search_prep(
        full_tqueries.n_simd_blocks,
        full_tqueries.n_queries,
        full_tqueries.pattern_length,
        full_searcher.alpha_pattern,
    );

    out.clear();
    batch_buf.clear();

    for chunk_start in (0..suffix_ranges.len()).step_by(F::LANES) {
        let chunk_end: usize = (chunk_start + F::LANES).min(suffix_ranges.len());
        let batch_slice = &suffix_ranges[chunk_start..chunk_end];

        batch_buf.clear();
        trace_batch_ranges(
            full_searcher,
            full_tqueries,
            text,
            batch_slice,
            k,
            post,
            batch_buf,
            Some(full_searcher.alpha),
            full_searcher.max_overhang,
        );

        for aln in batch_buf.drain(..) {
            if aln.cost <= k as Cost {
                out.push(aln);
            }
        }
    }
}

#[inline(always)]
fn trace_ranges_backend<S: SimdBackend>(
    searcher: &mut Myers<S>,
    tqueries: &TQueries<S>,
    text: &[u8],
    ranges: &[HitRange],
    k: u32,
    post: TracePostProcess,
    out: &mut Vec<Match>,
) {
    for chunk in ranges.chunks(S::LANES) {
        trace_batch_ranges(
            searcher,
            tqueries,
            text,
            chunk,
            k,
            post,
            out,
            Some(searcher.alpha),
            searcher.max_overhang,
        );
    }
}

#[derive(Debug, Clone)]
pub enum EncodedQueries {
    U8(TQueries<U8>),
    U16 {
        full: TQueries<U16>,
        suffix_u8: Option<Box<TQueries<U8>>>,
    },
    U32 {
        full: TQueries<U32>,
        suffix_u16: Option<Box<TQueries<U16>>>,
        suffix_u8: Option<Box<TQueries<U8>>>,
    },
    U64 {
        full: TQueries<U64>,
        suffix_u16: Option<Box<TQueries<U16>>>,
        suffix_u32: Option<Box<TQueries<U32>>>,
        suffix_u8: Option<Box<TQueries<U8>>>,
    },
}

impl EncodedQueries {
    pub fn max_pattern_length(&self) -> usize {
        match self {
            EncodedQueries::U8(_) => U8::LIMB_BITS,
            EncodedQueries::U16 { .. } => U16::LIMB_BITS,
            EncodedQueries::U32 { .. } => U32::LIMB_BITS,
            EncodedQueries::U64 { .. } => U64::LIMB_BITS,
        }
    }

    pub fn n_queries(&self) -> usize {
        match self {
            EncodedQueries::U8(tq) => tq.n_queries,
            EncodedQueries::U16 { full, .. } => full.n_queries,
            EncodedQueries::U32 { full, .. } => full.n_queries,
            EncodedQueries::U64 { full, .. } => full.n_queries,
        }
    }

    pub fn suffix_u16(&self) -> Option<&TQueries<U16>> {
        match self {
            EncodedQueries::U8(_) | EncodedQueries::U16 { .. } => None,
            EncodedQueries::U32 { suffix_u16, .. } => suffix_u16.as_deref(),
            EncodedQueries::U64 { suffix_u16, .. } => suffix_u16.as_deref(),
        }
    }

    pub fn suffix_u32(&self) -> Option<&TQueries<U32>> {
        match self {
            EncodedQueries::U8(_) | EncodedQueries::U16 { .. } | EncodedQueries::U32 { .. } => None,
            EncodedQueries::U64 { suffix_u32, .. } => suffix_u32.as_deref(),
        }
    }

    pub fn suffix_u8(&self) -> Option<&TQueries<U8>> {
        match self {
            EncodedQueries::U8(_) => None,
            EncodedQueries::U16 { suffix_u8, .. } => suffix_u8.as_deref(),
            EncodedQueries::U32 { suffix_u8, .. } => suffix_u8.as_deref(),
            EncodedQueries::U64 { suffix_u8, .. } => suffix_u8.as_deref(),
        }
    }
}

enum PrefilterBackend {
    U8,
    U16,
    U32,
}

pub struct Searcher {
    searcher_u8: Myers<U8>,
    searcher_u16: Myers<U16>,
    searcher_u32: Myers<U32>,
    searcher_u64: Myers<U64>,
    suffix_searcher_u8: Myers<U8>,
    suffix_searcher_u16: Myers<U16>,
    suffix_searcher_u32: Myers<U32>,
    alignments_buf: Vec<Match>,
    batch_buf: Vec<Match>,
}

impl Clone for Searcher {
    fn clone(&self) -> Self {
        Self {
            searcher_u8: Myers::new(Some(self.searcher_u8.alpha)),
            searcher_u16: Myers::new(Some(self.searcher_u16.alpha)),
            searcher_u32: Myers::new(Some(self.searcher_u32.alpha)),
            searcher_u64: Myers::new(Some(self.searcher_u64.alpha)),
            suffix_searcher_u8: Myers::new(Some(self.suffix_searcher_u8.alpha)),
            suffix_searcher_u16: Myers::new(Some(self.suffix_searcher_u16.alpha)),
            suffix_searcher_u32: Myers::new(Some(self.suffix_searcher_u32.alpha)),
            alignments_buf: Vec::new(),
            batch_buf: Vec::new(),
        }
    }
}

impl Searcher {
    pub fn new(alpha: Option<f32>) -> Self {
        Self {
            searcher_u8: Myers::new(alpha),
            searcher_u16: Myers::new(alpha),
            searcher_u32: Myers::new(alpha),
            searcher_u64: Myers::new(alpha),
            suffix_searcher_u8: Myers::new(alpha),
            suffix_searcher_u16: Myers::new(alpha),
            suffix_searcher_u32: Myers::new(alpha),
            alignments_buf: Vec::new(),
            batch_buf: Vec::new(),
        }
    }

    pub fn encode(&self, queries: &[Vec<u8>], include_rc: bool) -> EncodedQueries {
        if queries.is_empty() {
            panic!("No queries provided");
        }

        let max_pattern_length = queries.iter().map(|q| q.len()).max().unwrap();

        // all queries should be the same max length
        assert!(queries.iter().all(|q| q.len() == max_pattern_length));

        if max_pattern_length <= U8::LIMB_BITS {
            EncodedQueries::U8(TQueries::new(queries, include_rc))
        } else if max_pattern_length <= U16::LIMB_BITS {
            let full = TQueries::new(queries, include_rc);
            EncodedQueries::U16 {
                suffix_u8: Some(Box::new(full.reduce_to_suffix::<U8>())),
                full,
            }
        } else if max_pattern_length <= U32::LIMB_BITS {
            let full = TQueries::new(queries, include_rc);
            EncodedQueries::U32 {
                suffix_u16: Some(Box::new(full.reduce_to_suffix::<U16>())),
                suffix_u8: Some(Box::new(full.reduce_to_suffix::<U8>())),
                full,
            }
        } else if max_pattern_length <= U64::LIMB_BITS {
            let full = TQueries::new(queries, include_rc);
            EncodedQueries::U64 {
                suffix_u16: Some(Box::new(full.reduce_to_suffix::<U16>())),
                suffix_u32: Some(Box::new(full.reduce_to_suffix::<U32>())),
                suffix_u8: Some(Box::new(full.reduce_to_suffix::<U8>())),
                full,
            }
        } else {
            panic!(
                "pattern length {} exceeds maximum supported length {}",
                max_pattern_length,
                U64::LIMB_BITS
            );
        }
    }

    fn should_use_hierarchical(
        encoded_queries: &EncodedQueries,
        k: u32,
        use_hierarchical: Option<bool>,
    ) -> Option<PrefilterBackend> {
        if use_hierarchical != Some(true) {
            return None;
        }
        // Based on emperical benchmarks
        match encoded_queries {
            EncodedQueries::U8(_) => None,
            EncodedQueries::U16 { .. } if k == 0 => Some(PrefilterBackend::U8),
            EncodedQueries::U32 { .. } if k == 0 => Some(PrefilterBackend::U8),
            EncodedQueries::U32 { .. } if k < 4 => Some(PrefilterBackend::U16),
            EncodedQueries::U64 { .. } if k == 0 => Some(PrefilterBackend::U8),
            EncodedQueries::U64 { .. } if k < 4 => Some(PrefilterBackend::U16),
            EncodedQueries::U64 { .. } if k < 8 => Some(PrefilterBackend::U32),
            _ => None,
        }
    }

    #[rustfmt::skip]
    fn hierarchical_search_with_prefilter(
        &mut self,
        full_queries: &EncodedQueries,
        text: &[u8],
        k: u32,
        prefilter: PrefilterBackend,
        post: TracePostProcess,
    ) -> &[Match] {
        match (prefilter, full_queries) {
            (PrefilterBackend::U8, EncodedQueries::U16 { full, suffix_u8 }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u16, text, k, post, "U8"),
            (PrefilterBackend::U8, EncodedQueries::U32 { full, suffix_u8, .. }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u32, text, k, post, "U8"),
            (PrefilterBackend::U8, EncodedQueries::U64 { full, suffix_u8, .. }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u64, text, k, post, "U8"),
            (PrefilterBackend::U16, EncodedQueries::U32 { full, suffix_u16, .. }) => run_hierarchical!(self, full, suffix_u16, suffix_searcher_u16, searcher_u32, text, k, post, "U16"),
            (PrefilterBackend::U16, EncodedQueries::U64 { full, suffix_u16, .. }) => run_hierarchical!(self, full, suffix_u16, suffix_searcher_u16, searcher_u64, text, k, post, "U16"),
            (PrefilterBackend::U32, EncodedQueries::U64 { full, suffix_u32, .. }) => run_hierarchical!(self, full, suffix_u32, suffix_searcher_u32, searcher_u64, text, k, post, "U32"),
            _ => panic!("Invalid prefilter backend combination"),
        }
        self.alignments_buf.as_slice()
    }

    pub fn search(&mut self, encoded_queries: &EncodedQueries, text: &[u8], k: u32) -> &[Match] {
        self.search_with_options(
            encoded_queries,
            text,
            k,
            Some(true),
            TracePostProcess::LocalMinima,
        )
    }

    pub fn search_all(
        &mut self,
        encoded_queries: &EncodedQueries,
        text: &[u8],
        k: u32,
    ) -> &[Match] {
        self.search_with_options(encoded_queries, text, k, Some(true), TracePostProcess::All)
    }

    pub fn search_with_options(
        &mut self,
        encoded_queries: &EncodedQueries,
        text: &[u8],
        k: u32,
        use_hierarchical: Option<bool>,
        post: TracePostProcess,
    ) -> &[Match] {
        if let Some(prefilter) = Self::should_use_hierarchical(encoded_queries, k, use_hierarchical)
        {
            self.hierarchical_search_with_prefilter(encoded_queries, text, k, prefilter, post)
        } else {
            dispatch_encoded!(self, encoded_queries, |searcher, tq| {
                // Find matching ranges
                let ranges: Vec<HitRange> =
                    searcher.search_ranges(tq, text, k, true, true).to_vec();
                self.alignments_buf.clear();
                // Trace each of the ranges
                trace_ranges_backend(
                    searcher,
                    tq,
                    text,
                    &ranges,
                    k,
                    post,
                    &mut self.alignments_buf,
                );
                self.alignments_buf.as_slice()
            })
        }
    }
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new(None)
    }
}
