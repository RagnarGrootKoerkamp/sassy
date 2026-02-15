use crate::pattern_tiling::backend::SimdBackend;
use crate::pattern_tiling::minima::TracePostProcess;
use crate::pattern_tiling::search::{HitRange, Myers};
use crate::pattern_tiling::tqueries::TQueries;
use crate::pattern_tiling::trace::TraceBuffer;
use crate::pattern_tiling::trace::trace_batch_ranges;
use crate::profiles::Profile;
use crate::search::Match;

// If AVX-512 is available, we want to use the 512-bit backends instead of the default 256-bit ones.
// This allows processing twice as many patterns in parallel.

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
pub type U8Backend = crate::pattern_tiling::backend::U8;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub type U8Backend = crate::pattern_tiling::backend::U8_512;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
pub type U16Backend = crate::pattern_tiling::backend::U16;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub type U16Backend = crate::pattern_tiling::backend::U16_512;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub type U32Backend = crate::pattern_tiling::backend::U32;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type U32Backend = crate::pattern_tiling::backend::U32_512;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub type U64Backend = crate::pattern_tiling::backend::U64;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type U64Backend = crate::pattern_tiling::backend::U64_512;

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
    post: TracePostProcess,
    out: &mut Vec<Match>,
    trace_buffer: &mut TraceBuffer,
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
            Some(full_searcher.alpha),
            full_searcher.max_overhang,
            trace_buffer,
        );
        out.extend_from_slice(&trace_buffer.filtered_alignments);
    }
}

#[inline(always)]
fn trace_ranges_backend<S: SimdBackend, P: Profile>(
    searcher: &mut Myers<S, P>,
    tqueries: &TQueries<S, P>,
    text: &[u8],
    ranges: &[HitRange],
    k: u32,
    post: TracePostProcess,
    out: &mut Vec<Match>,
    trace_buffer: &mut TraceBuffer,
) {
    out.clear();
    for chunk in ranges.chunks(S::LANES) {
        trace_batch_ranges(
            searcher,
            tqueries,
            text,
            chunk,
            k,
            post,
            Some(searcher.alpha),
            searcher.max_overhang,
            trace_buffer,
        );
        out.extend_from_slice(&trace_buffer.filtered_alignments);
    }
}

#[derive(Debug, Clone)]
pub enum EncodedPatterns<P: Profile> {
    U8(TQueries<U8Backend, P>),
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

impl<P: Profile> EncodedPatterns<P> {
    pub fn max_pattern_length(&self) -> usize {
        match self {
            EncodedPatterns::U8(_) => U8Backend::LIMB_BITS,
            EncodedPatterns::U16 { .. } => U16Backend::LIMB_BITS,
            EncodedPatterns::U32 { .. } => U32Backend::LIMB_BITS,
            EncodedPatterns::U64 { .. } => U64Backend::LIMB_BITS,
        }
    }

    pub fn n_queries(&self) -> usize {
        match self {
            EncodedPatterns::U8(tq) => tq.n_queries,
            EncodedPatterns::U16 { full, .. } => full.n_queries,
            EncodedPatterns::U32 { full, .. } => full.n_queries,
            EncodedPatterns::U64 { full, .. } => full.n_queries,
        }
    }

    pub fn suffix_u16(&self) -> Option<&TQueries<U16Backend, P>> {
        match self {
            EncodedPatterns::U8(_) | EncodedPatterns::U16 { .. } => None,
            EncodedPatterns::U32 { suffix_u16, .. } => suffix_u16.as_deref(),
            EncodedPatterns::U64 { suffix_u16, .. } => suffix_u16.as_deref(),
        }
    }

    pub fn suffix_u32(&self) -> Option<&TQueries<U32Backend, P>> {
        match self {
            EncodedPatterns::U8(_) | EncodedPatterns::U16 { .. } | EncodedPatterns::U32 { .. } => {
                None
            }
            EncodedPatterns::U64 { suffix_u32, .. } => suffix_u32.as_deref(),
        }
    }

    pub fn suffix_u8(&self) -> Option<&TQueries<U8Backend, P>> {
        match self {
            EncodedPatterns::U8(_) => None,
            EncodedPatterns::U16 { suffix_u8, .. } => suffix_u8.as_deref(),
            EncodedPatterns::U32 { suffix_u8, .. } => suffix_u8.as_deref(),
            EncodedPatterns::U64 { suffix_u8, .. } => suffix_u8.as_deref(),
        }
    }
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
    alignments_buf: Vec<Match>,
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
            alignments_buf: Vec::new(),
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
            alignments_buf: Vec::new(),
            trace_buffer: TraceBuffer::new(64),
        }
    }

    pub fn encode(&self, queries: &[Vec<u8>], include_rc: bool) -> EncodedPatterns<P> {
        //Fixme: neat to encode all possible suffixes so every k is supported when
        // search is called, it is wasteful when user never searches different k's
        if queries.is_empty() {
            panic!("No queries provided");
        }

        let max_pattern_length = queries.iter().map(|q| q.len()).max().unwrap();

        // all queries should be the same max length
        assert!(queries.iter().all(|q| q.len() == max_pattern_length));

        if max_pattern_length <= U8Backend::LIMB_BITS {
            EncodedPatterns::U8(TQueries::<U8Backend, P>::new(queries, include_rc))
        } else if max_pattern_length <= U16Backend::LIMB_BITS {
            let full = TQueries::<U16Backend, P>::new(queries, include_rc);
            EncodedPatterns::U16 {
                suffix_u8: Some(Box::new(full.reduce_to_suffix::<U8Backend, P>())),
                full,
            }
        } else if max_pattern_length <= U32Backend::LIMB_BITS {
            let full = TQueries::new(queries, include_rc);
            EncodedPatterns::U32 {
                suffix_u16: Some(Box::new(full.reduce_to_suffix::<U16Backend, P>())),
                suffix_u8: Some(Box::new(full.reduce_to_suffix::<U8Backend, P>())),
                full,
            }
        } else if max_pattern_length <= U64Backend::LIMB_BITS {
            let full = TQueries::new(queries, include_rc);
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
        post: TracePostProcess,
    ) -> &[Match] {
        match (prefilter, full_queries) {
            (PrefilterBackend::U8, EncodedPatterns::U16 { full, suffix_u8 }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u16, text, k, post, "U8"),
            (PrefilterBackend::U8, EncodedPatterns::U32 { full, suffix_u8, .. }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u32, text, k, post, "U8"),
            (PrefilterBackend::U8, EncodedPatterns::U64 { full, suffix_u8, .. }) => run_hierarchical!(self, full, suffix_u8, suffix_searcher_u8, searcher_u64, text, k, post, "U8"),
            (PrefilterBackend::U16, EncodedPatterns::U32 { full, suffix_u16, .. }) => run_hierarchical!(self, full, suffix_u16, suffix_searcher_u16, searcher_u32, text, k, post, "U16"),
            (PrefilterBackend::U16, EncodedPatterns::U64 { full, suffix_u16, .. }) => run_hierarchical!(self, full, suffix_u16, suffix_searcher_u16, searcher_u64, text, k, post, "U16"),
            (PrefilterBackend::U32, EncodedPatterns::U64 { full, suffix_u32, .. }) => run_hierarchical!(self, full, suffix_u32, suffix_searcher_u32, searcher_u64, text, k, post, "U32"),
            _ => panic!("Invalid prefilter backend combination"),
        }
        self.alignments_buf.as_slice()
    }

    pub fn search(
        &mut self,
        encoded_queries: &EncodedPatterns<P>,
        text: &[u8],
        k: u32,
    ) -> &[Match] {
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
        encoded_queries: &EncodedPatterns<P>,
        text: &[u8],
        k: u32,
    ) -> &[Match] {
        self.search_with_options(encoded_queries, text, k, Some(true), TracePostProcess::All)
    }

    pub fn search_with_options(
        &mut self,
        encoded_queries: &EncodedPatterns<P>,
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
                // Trace each of the ranges, accumulating into alignments_buf
                trace_ranges_backend(
                    searcher,
                    tq,
                    text,
                    ranges.as_slice(),
                    k,
                    post,
                    &mut self.alignments_buf,
                    &mut self.trace_buffer,
                );
                self.alignments_buf.as_slice()
            })
        }
    }
}

impl<P: Profile> Default for Searcher<P> {
    fn default() -> Self {
        Self::new(None)
    }
}
