use std::cmp::Reverse;
use std::fmt::Debug;
use std::hint::assert_unchecked;

use crate::delta_encoding::H;
use crate::minima::prefix_min;
use crate::profiles::Profile;
use crate::trace::{CostMatrix, fill, get_trace, simd_fill, simd_fill_multipattern};
use crate::{LANES, S};
use crate::{bitpacking::compute_block_simd, delta_encoding::V};
use pa_types::{Cigar, CigarOp, Cost, Pos};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// A match of the pattern against the text.
///
/// All indices are 0-based.
///
/// The `pattern_start` and `pattern_end` are usually just `0` and `m` to cover the entire pattern,
/// unless overhang alignments are enabled.
///
/// For matches against the reverse complement text (when `match.strand = Strand::Rc`),
/// `text_start` and `text_end` are indices into the _forward_ text, as given by the user.
/// Thus, the pattern will match `rc(&text[text_start..text_end])`.
/// In this case, the CIGAR tells the differences between `pattern` and `rc(&text[text_start..text_end])`.
/// In the CIGAR, `I` represents a character in the text that is not in the pattern,
/// and `D` represents a character in the pattern that is not in the text.
#[derive(derivative::Derivative, Clone, PartialEq, Eq)]
#[derivative(PartialOrd, Ord)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct Match {
    /// index of the pattern (when multiple patterns are given)
    pub pattern_idx: usize,
    /// index of the text (when multiple texts are given)
    pub text_idx: usize,
    /// 0-based start position in text.
    pub text_start: usize,
    /// 0-based exclusive end position in text.
    pub text_end: usize,
    /// 0-based start position in pattern. 0 unless left-overhanging alignment.
    pub pattern_start: usize,
    /// 0-based exclusive end position in pattern. `m=|pattern|`, unless right-overhanging alignment.
    pub pattern_end: usize,
    /// Cost of the alignment.
    pub cost: Cost,
    /// Strand of the match.
    pub strand: Strand,
    /// CIGAR representation of the alignment.
    ///
    /// The CIGAR should always be read in the direction of the input pattern.
    /// `=`: match
    /// `X`: mismatch
    /// `I`: character in text but not in pattern.
    /// `D`: character in pattern but not in text.
    #[derivative(PartialOrd = "ignore")]
    #[derivative(Ord = "ignore")]
    pub cigar: Cigar,
}

impl Debug for Match {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Match")
            .field("pattern_idx", &self.pattern_idx)
            .field("text_idx", &self.text_idx)
            .field("text_start", &self.text_start)
            .field("text_end", &self.text_end)
            .field("pattern_start", &self.pattern_start)
            .field("pattern_end", &self.pattern_end)
            .field("cost", &self.cost)
            .field("strand", &self.strand)
            .field("cigar", &self.cigar.to_string())
            .finish()
    }
}

impl Match {
    /// Convert the match to a list of (pattern pos, text pos) positions.
    pub fn to_path(&self) -> Vec<Pos> {
        let (path_start_text, sign) = if self.strand == Strand::Rc {
            (self.text_end - 1, -1) // exclusive end
        } else {
            (self.text_start, 1)
        };
        let mut pos = Pos(self.pattern_start as i32, path_start_text as i32);
        let mut path = vec![pos];
        for op in &self.cigar.ops {
            for _ in 0..op.cnt {
                pos += match op.op {
                    CigarOp::Match | CigarOp::Sub => Pos(1, sign),
                    CigarOp::Del => Pos(1, 0),
                    CigarOp::Ins => Pos(0, sign),
                };
                path.push(pos);
            }
        }
        path.pop();
        path
    }

    /// Drop the cigar from the match. Convenient for debug printing.
    pub fn without_cigar(&self) -> Match {
        Match {
            cigar: Cigar::default(),
            ..*self
        }
    }
}

/// Strand of a match. If Rc, pattern matches the reverse complement text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Strand {
    Fwd,
    Rc,
}

/// A trait for sequences that can cache their reverse.
pub trait RcSearchAble {
    /// The forward text
    fn text(&self) -> impl AsRef<[u8]>;
    /// The reverse text
    fn rev_text(&self) -> impl AsRef<[u8]>;
}

/// Any text can be reversed on the fly.
impl<T: ?Sized> RcSearchAble for T
where
    T: AsRef<[u8]>,
{
    fn text(&self) -> impl AsRef<[u8]> {
        self.as_ref()
    }
    fn rev_text(&'_ self) -> impl AsRef<[u8]> {
        self.as_ref().iter().rev().copied().collect::<Vec<_>>()
    }
}

/// Struct that computes the reverse once on construction.
#[derive(Debug)]
pub struct CachedRev<T: AsRef<[u8]>> {
    pub text: T,
    pub rev: Option<Vec<u8>>,
}

impl<T: AsRef<[u8]>> CachedRev<T> {
    pub fn new(text: T, build_rev: bool) -> Self {
        let rev = build_rev.then(|| text.as_ref().iter().rev().copied().collect());
        CachedRev { text, rev }
    }
    pub fn initialize_rev(&mut self) {
        self.rev = Some(self.text.as_ref().iter().rev().copied().collect());
    }
}

impl<T: AsRef<[u8]>> RcSearchAble for CachedRev<T> {
    fn text(&self) -> impl AsRef<[u8]> {
        &self.text
    }
    fn rev_text(&self) -> impl AsRef<[u8]> {
        self.rev.as_ref().unwrap()
    }
}

#[derive(Clone)]
struct LaneState<P: Profile> {
    decreasing: bool,
    text_slice: [u8; 64],
    text_profile: P::B,
    matches: Vec<(usize, Cost)>,
    chunk_offset: usize,
    /// Index of last computed text position for this lane.
    lane_end: usize,
}

impl<P: Profile> LaneState<P> {
    fn new(text_profile: P::B, chunk_offset: usize) -> Self {
        Self {
            decreasing: false,
            text_slice: [0; 64],
            text_profile,
            matches: Vec::new(),
            chunk_offset,
            lane_end: 0,
        }
    }

    #[inline(always)]
    fn update_and_encode(&mut self, text: &[u8], i: usize, profiler: &P, overhang: bool) {
        let start = self.chunk_offset * 64 + 64 * i;
        self.lane_end = start + 64;
        if start + 64 <= text.len() {
            self.text_slice.copy_from_slice(&text[start..start + 64]);
        } else {
            // Pad with N, so that costs at the end are diagonally preserved.
            self.text_slice.fill(if overhang { b'N' } else { b'X' });
            if start <= text.len() {
                let slice = &text[start..];
                self.text_slice[..slice.len()].copy_from_slice(slice);
            }
        }
        profiler.encode_ref(&self.text_slice, &mut self.text_profile);
    }
}

/// The main entry point for searching.
///
/// Construct using one of the `new_*` methods.
/// Then call `search` (giving local minima) or `search_all` (giving all minima).
///
/// Supports:
/// - `Ascii`/`Dna`/`Iupac` profiles.
/// - Searching only forward or also the reverse complement text.
/// - Overhang cost, for `Iupac` profile.
///
/// This object caches internal buffers, so that reusing it avoids allocations.
///
/// See the library documentation for examples.
#[derive(Clone)]
pub struct Searcher<P: Profile> {
    // Config
    /// Search reverse complement text?
    rc: bool,
    /// overhang cost
    /// If set, must satisfy `0.0 <= alpha <= 1.0`
    alpha: Option<f32>,
    /// If set, only the best match of each (pattern, text) pair is returned.
    only_best_match: bool,
    /// If set, matches are returned without trace and starting position.
    without_trace: bool,
    /// If set, overhang can be at most this long.
    max_overhang: Option<usize>,

    // Internal caches
    cost_matrices: [CostMatrix; LANES],
    hp: Vec<S>,
    hm: Vec<S>,
    lanes: [LaneState<P>; LANES],

    matches: Vec<Match>,

    _phantom: std::marker::PhantomData<P>,
}

#[derive(Clone, Copy)]
enum MultiText<'t> {
    One(&'t [u8]),
    Multi(&'t [&'t [u8]]),
}

impl<'t> MultiText<'t> {
    fn one(t: &'t [u8]) -> Self {
        MultiText::One(t)
    }

    fn get_lane(&self, lane: usize) -> Option<&'t [u8]> {
        match self {
            MultiText::One(t) => Some(t),
            MultiText::Multi(ts) => ts.as_ref().get(lane).copied(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum MultiPattern<'t> {
    One(&'t [u8]),
    /// All patterns must have the same length.
    Multi(&'t [&'t [u8]]),
}

impl<'t> MultiPattern<'t> {
    pub fn one(t: &'t [u8]) -> Self {
        MultiPattern::One(t)
    }

    pub fn get_lane(&self, lane: usize) -> Option<&'t [u8]> {
        match self {
            MultiPattern::One(t) => Some(t),
            MultiPattern::Multi(ts) => ts.get(lane).copied(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            MultiPattern::One(t) => t.len(),
            MultiPattern::Multi(ts) => ts[0].len(),
        }
    }
}

#[derive(Clone, Copy)]
enum MultiRcText<'x, I: RcSearchAble + ?Sized> {
    One(&'x I),
    Multi(&'x [&'x I]),
}

impl<'x, I: RcSearchAble + ?Sized> MultiRcText<'x, I> {
    fn one(t: &'x I) -> Self {
        MultiRcText::One(t)
    }
}

impl<'x, I: RcSearchAble + ?Sized> MultiRcText<'x, I> {
    fn get_lane(&self, lane: usize) -> Option<&I> {
        match self {
            MultiRcText::One(t) => Some(t),
            MultiRcText::Multi(ts) => ts.as_ref().get(lane).copied(),
        }
    }
}

/// The algorithms to use for [`Searcher::search_many`].
pub enum SearchMode {
    /// Do one (pattern, text) search at a time.
    /// SIMD lanes correspond to chunks of each text.
    ///
    /// Use when both patterns and texts have mixed lengths.
    /// Not ideal when texts are short (roughly <500) because of overlapping text chunks.
    Single,
    /// When all patterns have similar/equal length, use one pattern per SIMD lane.
    /// Avoids overlapping text chunks and usually faster.
    ///
    /// Consider sorting patterns by length beforehand.
    /// Ideal when all patterns have length <32.
    BatchPatterns,
    /// Search one pattern against multiple texts at a time, with one SIMD lane per text.
    ///
    /// Ideal when texts have similar lengths.
    /// Possibly has large overhead for texts of highly differing lengths.
    /// Consider sorting texts by length beforehand.
    BatchTexts,
}

impl<P: Profile> Searcher<P> {
    // The number of rows (pattern chars) we *at least*
    // mainly to avoid branching
    const CHECK_AT_LEAST_ROWS: usize = 8;

    /// Forward search only.
    pub fn new_fwd() -> Self {
        Self::new(false, None)
    }

    /// Forward and reverse complement search.
    pub fn new_rc() -> Self {
        Self::new(true, None)
    }

    fn _overhang_check(alpha: f32) {
        if !P::supports_overhang() {
            panic!(
                "Overhang is not supported for {:?}",
                std::any::type_name::<P>()
            );
        }
        if !(0.0..=1.0).contains(&alpha) {
            panic!("Alpha must be in range 0.0 <= alpha <= 1.0");
        }
    }

    /// Forward search with overhang cost `0<=alpha<=1`.
    pub fn new_fwd_with_overhang(alpha: f32) -> Self {
        Self::_overhang_check(alpha);
        Self::new(false, Some(alpha))
    }

    /// Forward and reverse complement search with overhang cost `0<=alpha<=1`.
    pub fn new_rc_with_overhang(alpha: f32) -> Self {
        Self::_overhang_check(alpha);
        Self::new(true, Some(alpha))
    }

    /// Set overhang cost `0<=alpha<=1`.
    pub fn with_overhang(mut self, alpha: f32) -> Self {
        Self::_overhang_check(alpha);
        self.alpha = Some(alpha);
        self
    }

    /// Set overhang cost `0<=alpha<=1`.
    pub fn with_max_overhang(mut self, max_overhang: Option<usize>) -> Self {
        self.max_overhang = max_overhang;
        self
    }

    /// Only return the best match.
    pub fn only_best_match(mut self) -> Self {
        self.only_best_match = true;
        self
    }

    /// Return matches without trace and starting point.
    pub fn without_trace(mut self) -> Self {
        self.without_trace = true;
        self
    }

    /// Default: return matches with trace.
    ///
    /// Only here to negate `without_trace`.
    pub fn with_trace(mut self) -> Self {
        self.without_trace = false;
        self
    }
    pub fn set_trace(&mut self, trace: bool) {
        self.without_trace = !trace;
    }

    /// Create a new `Searcher`.
    pub fn new(rc: bool, alpha: Option<f32>) -> Self {
        Self {
            alpha,
            rc,
            only_best_match: false,
            without_trace: false,
            max_overhang: None,
            cost_matrices: std::array::from_fn(|_| CostMatrix::default()),
            hp: vec![],
            hm: vec![],
            lanes: std::array::from_fn(|_| LaneState::new(P::alloc_out(), 0)),
            matches: vec![],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a match for each *rightmost local minimum* end position with score <=k.
    ///
    /// This avoids reporting matches that completely overlap apart from a few characters at the ends.
    ///
    /// Searches the forward text, and optionally the reverse complement of the text.
    pub fn search<I: RcSearchAble + ?Sized>(
        &mut self,
        pattern: &[u8],
        text: &I,
        k: usize,
    ) -> Vec<Match> {
        self.matches.clear();
        self.search_handle_rc(
            MultiPattern::one(pattern),
            MultiRcText::one(text),
            k,
            false,
            None::<fn(&[u8], &[u8], Strand) -> bool>,
        );
        std::mem::take(&mut self.matches)
    }

    /// Search each given pattern in each given text, using the algorithms given by `mode` (see [`SearchMode`]).
    ///
    /// Does multithreading using `rayon`.
    /// Returns a vector of (pattern index, text index, match).
    pub fn search_many<PAT: AsRef<[u8]> + Sync, I: RcSearchAble + ?Sized + Sync>(
        &mut self,
        patterns: &[PAT],
        texts: &[&I],
        k: usize,
        num_threads: usize,
        mode: SearchMode,
    ) -> Vec<Match> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap()
            .install(|| match mode {
                SearchMode::Single => map_collect_cartesian_product(
                    patterns,
                    texts,
                    self,
                    |searcher, pattern, text, pi, ti| {
                        let mut matches = searcher.search(pattern.as_ref(), *text, k);
                        matches.iter_mut().for_each(move |m| {
                            m.pattern_idx = pi;
                            m.text_idx = ti;
                        });
                        matches
                    },
                ),
                SearchMode::BatchPatterns => {
                    let pattern_batches: Vec<&[PAT]> = patterns.chunks(LANES).collect();

                    map_collect_cartesian_product(
                        &pattern_batches,
                        texts,
                        self,
                        |searcher, pattern_batch, text, pbi, ti| {
                            let mut matches = searcher.search_patterns(pattern_batch, *text, k);
                            matches.iter_mut().for_each(move |m| {
                                m.pattern_idx += pbi * LANES;
                                m.text_idx = ti;
                            });
                            matches
                        },
                    )
                }

                SearchMode::BatchTexts => {
                    let text_batches: Vec<&[&I]> = texts.chunks(LANES).collect();

                    map_collect_cartesian_product(
                        &patterns,
                        &text_batches,
                        self,
                        |searcher, pattern, text_batch, pi, tbi| {
                            let mut matches =
                                searcher.search_texts(pattern.as_ref(), *text_batch, k);
                            matches.iter_mut().for_each(move |m| {
                                m.pattern_idx = pi;
                                m.text_idx += tbi * LANES;
                            });
                            matches
                        },
                    )
                }
            })
    }

    /// Search multiple similar-length texts in chunks of `LANES` at a time.
    ///
    /// Use this instead of `for text in texts { searcher.search(pattern, text, k) }`
    /// when the texts are short (below 1 to 10 kbp) and have similar length.
    ///
    /// Consider sorting the texts by length beforehand.
    ///
    /// Returns a vector of (text index, match).
    ///
    /// Use `early_break_below` to return the best values as soon as a match is at least that good.
    pub fn search_texts<I: RcSearchAble + ?Sized>(
        &mut self,
        pattern: &[u8],
        texts: &[&I],
        k: usize,
    ) -> Vec<Match> {
        self.matches.clear();
        for (i, chunk) in texts.chunks(LANES).enumerate() {
            let chunk_matches = self.search_handle_rc(
                MultiPattern::one(pattern),
                MultiRcText::Multi(chunk),
                k,
                false,
                None::<fn(&[u8], &[u8], Strand) -> bool>,
            );
            chunk_matches.iter_mut().for_each(|m| {
                m.text_idx += i * LANES;
            });
        }

        std::mem::take(&mut self.matches)
    }

    /// Search multiple equals-length patterns in chunks of `LANES` at a time.
    ///
    /// Use this instead of `for pattern in patterns { searcher.search(pattern, text, k) }`
    /// when the patterns have similar length.
    ///
    /// Consider sorting the patterns by length beforehand.
    ///
    /// Returns a vector of (pattern index, match).
    ///
    /// Use `early_break_below` to return the best values as soon as a match is at least that good.
    pub fn search_patterns<I: RcSearchAble + ?Sized, PT: AsRef<[u8]>>(
        &mut self,
        patterns: &[PT],
        text: &I,
        k: usize,
    ) -> Vec<Match> {
        assert!(
            patterns
                .iter()
                .all(|p| p.as_ref().len() == patterns[0].as_ref().len()),
            "All patterns passed to search_patterns must have the same length"
        );

        self.matches.clear();
        for (i, chunk) in patterns.chunks(LANES).enumerate() {
            let slice_chunk: [&[u8]; LANES] =
                std::array::from_fn(|lane| chunk.get(lane).map(|p| p.as_ref()).unwrap_or(&[]));
            let chunk_matches = self.search_handle_rc(
                MultiPattern::Multi(&slice_chunk[..chunk.len()]),
                MultiRcText::One(text),
                k,
                false,
                None::<fn(&[u8], &[u8], Strand) -> bool>,
            );
            chunk_matches.iter_mut().for_each(|m| {
                m.pattern_idx += i * LANES;
            });
        }

        std::mem::take(&mut self.matches)
    }

    /// Returns a match for *all* end positions with score <=k.
    ///
    /// Searches the forward text, and optionally the reverse complement of the text.
    /// Only use this instead of [`search`] if you know what you are doing,
    /// which typically means there is some postprocessing step to filter overlapping matches.
    pub fn search_all<I: RcSearchAble + ?Sized>(
        &mut self,
        pattern: &[u8],
        input: &I,
        k: usize,
    ) -> Vec<Match> {
        self.matches.clear();
        self.search_handle_rc(
            MultiPattern::one(pattern),
            MultiRcText::one(input),
            k,
            true,
            None::<fn(&[u8], &[u8], Strand) -> bool>,
        );
        std::mem::take(&mut self.matches)
    }

    /// Returns matches for *all* end positions where `end_filter_fn` returns true.
    ///
    /// The extra `usize` field is always 0 here and is only present for
    /// consistency with the mult-search functions.
    ///
    /// Used in CRISPR search to filter for only those end positions where the
    /// PAM (last three characters) matches exactly.
    ///
    /// `filter_fn` is passed the pattern, the (rc) text up to the end position, and the strand.
    /// Note that due to the implementation, for rc searches,
    /// both the pattern and text are complemented from what you would expect.
    pub fn search_with_fn<I: RcSearchAble + ?Sized>(
        &mut self,
        pattern: &[u8],
        text: &I,
        k: usize,
        all_minima: bool,
        filter_fn: impl Fn(&[u8], &[u8], Strand) -> bool,
    ) -> Vec<Match> {
        self.matches.clear();
        self.search_handle_rc(
            MultiPattern::one(pattern),
            MultiRcText::one(text),
            k,
            all_minima,
            Some(filter_fn),
        );
        std::mem::take(&mut self.matches)
    }

    /// Appends results to `self.idx_matches`.
    fn search_handle_rc<I: RcSearchAble + ?Sized>(
        &mut self,
        pattern: MultiPattern,
        text: MultiRcText<I>,
        k: usize,
        all_minima: bool,
        filter_fn: Option<impl Fn(&[u8], &[u8], Strand) -> bool>,
    ) -> &mut [Match] {
        let old_len = self.matches.len();
        // FIXME: This stuff is so ugly :/
        let fwd_text;
        let fwd_texts: Vec<_>;
        let fwd_slices: Vec<_>;
        let fwd_input = match text {
            MultiRcText::One(t) => {
                fwd_text = t.text();
                MultiText::one(fwd_text.as_ref())
            }
            MultiRcText::Multi(ts) => {
                fwd_texts = ts.iter().map(|i| i.text()).collect();
                fwd_slices = fwd_texts.iter().map(|t| t.as_ref()).collect();
                MultiText::Multi(fwd_slices.as_slice())
            }
        };
        self.search_one_strand(pattern, fwd_input, k, all_minima, &filter_fn, Strand::Fwd);

        if self.rc {
            // FIXME: This stuff is so ugly :/
            let rev_text;
            let rev_texts: Vec<_>;
            let rev_slices: Vec<_>;
            let rev_input = match text {
                MultiRcText::One(t) => {
                    rev_text = t.rev_text();
                    MultiText::one(rev_text.as_ref())
                }
                MultiRcText::Multi(ts) => {
                    rev_texts = ts.iter().map(|i| i.rev_text()).collect();
                    rev_slices = rev_texts.iter().map(|t| t.as_ref()).collect();
                    MultiText::Multi(rev_slices.as_slice())
                }
            };
            let _p;
            let _ps: [_; LANES];
            let _ps_slices: [_; LANES];
            let complement_pattern = match &pattern {
                MultiPattern::One(p) => {
                    _p = P::complement(p);
                    MultiPattern::one(&_p)
                }
                MultiPattern::Multi(ps) => {
                    _ps = std::array::from_fn(|i| {
                        if let Some(p) = ps.get(i) {
                            P::complement(p)
                        } else {
                            vec![]
                        }
                    });
                    _ps_slices = std::array::from_fn(|i| _ps[i].as_slice());

                    MultiPattern::Multi(&_ps_slices[..ps.len()])
                }
            };
            let without_trace = self.without_trace;
            let rc_matches = self.search_one_strand(
                complement_pattern,
                rev_input,
                k,
                all_minima,
                &filter_fn,
                Strand::Rc,
            );
            rc_matches.iter_mut().for_each(|m| {
                m.strand = Strand::Rc;
                // Also adjust start and end positions to original text orientation
                let rc_start = m.text_start;
                let rc_end = m.text_end;
                let t = text.get_lane(m.text_idx).unwrap();
                let len = t.text().as_ref().len();
                m.text_start = len - rc_end;
                if without_trace {
                    m.text_end = usize::MAX;
                } else {
                    m.text_end = len - rc_start;
                }
                // NOTE: We keep the cigar in the direction of the pattern.
                // Thus, passing text or rc(text) gives the same CIGAR.
                // m.cigar.ops.reverse();
            });
        }
        &mut self.matches[old_len..]
    }

    /// Appends results to `self.idx_matches`, and returns the new slice.
    fn search_one_strand<'t>(
        &mut self,
        pattern: MultiPattern<'t>,
        text: MultiText<'t>,
        k: usize,
        all_minima: bool,
        filter_fn: &Option<impl Fn(&[u8], &[u8], Strand) -> bool>,
        strand: Strand,
    ) -> &mut [Match] {
        self.search_positions_bounded(pattern, text, k as Cost, all_minima);
        // If there is a filter fn, filter end positions based on function before processing matches
        if let Some(filter_fn) = filter_fn {
            for (l, lane) in self.lanes.iter_mut().enumerate() {
                let Some(t) = text.get_lane(l) else {
                    break;
                };
                if let Some(p) = pattern.get_lane(l) {
                    lane.matches.retain(|(end_pos, _)| {
                        let text_till_end = &t[..*end_pos];
                        filter_fn(p, text_till_end, strand)
                    });
                }
            }
        }
        self.process_matches(pattern, text, k as Cost)
    }

    /// Check if any value in the lane is <= k.
    #[inline(always)]
    fn min_in_lane(&self, v: V, lane: usize, dist_to_start_of_lane: &S) -> Cost {
        // Get the current cost state for this lane

        // Calculate the minimum possible cost in this lane
        // This is the best case scenario - if even this minimum is > k,
        // then no matches are possible in this lane
        prefix_min(v.0, v.1).0 as Cost + dist_to_start_of_lane.as_array()[lane] as Cost
    }

    /// Check if any value in any lane is <= k.
    #[inline(always)]
    fn check_lanes(
        &self,
        vp: &S,
        vm: &S,
        dist_to_start_of_lane: &S,
        k: Cost,
        j: usize,
    ) -> Option<usize> {
        for lane in 0..LANES {
            let v = V(vp.as_array()[lane], vm.as_array()[lane]);
            let min_in_lane = self.min_in_lane(v, lane, dist_to_start_of_lane);
            if min_in_lane <= k {
                // Promising lane, we "estimate" how many rows more we need to check
                // as the difference between the minimum in the lane and the maximum edits
                // we can afford.
                let rows_needed = (k - min_in_lane) as usize;
                let new_end = j + Self::CHECK_AT_LEAST_ROWS.max(rows_needed);
                return Some(new_end);
            }
        }

        // No lanes are promising
        None
    }

    pub fn search_positions_bounded<'t>(
        &mut self,
        pattern: MultiPattern<'t>,
        text: MultiText<'t>,
        k: Cost,
        all_minima: bool,
    ) {
        match pattern {
            MultiPattern::One(p) => {
                let (profiler, pattern_profile) = P::encode_pattern(p);
                let get_eq = |j: usize, lanes: &[LaneState<P>; LANES]| {
                    let pattern_char = unsafe { pattern_profile.get_unchecked(j) };
                    S::from(std::array::from_fn(|lane| {
                        P::eq(pattern_char, &lanes[lane].text_profile)
                    }))
                };
                self.search_internal(pattern, profiler, get_eq, text, k, all_minima, false);
            }
            MultiPattern::Multi(ps) => {
                let (profiler, pattern_profiles) = P::encode_patterns(ps);
                let get_eq = |j: usize, lanes: &[LaneState<P>; LANES]| {
                    S::from(std::array::from_fn(|lane| {
                        P::eq(&pattern_profiles[j][lane], &lanes[lane].text_profile)
                    }))
                };
                self.search_internal(pattern, profiler, get_eq, text, k, all_minima, true);
            }
        }
    }

    #[inline(always)]
    fn search_prep(
        &mut self,
        pattern: &MultiPattern,
        text: &MultiText,
        k: Cost,
        is_multi_pattern: bool,
    ) -> (usize, usize, usize) {
        let pattern_len = pattern.len();
        let is_single_text = matches!(text, MultiText::One(_));

        let max_overlap_blocks = if is_single_text && !is_multi_pattern {
            (pattern_len + k as usize).div_ceil(64)
        } else {
            0
        };

        let text_padding = self.alpha.map_or(0, |alpha| {
            pattern_len
                .min(((k as f32) / alpha).ceil() as usize)
                .min(self.max_overhang.unwrap_or(usize::MAX))
        });

        let blocks_per_chunk = match text {
            MultiText::One(t) => {
                let base = (t.len() + text_padding).div_ceil(64);
                if is_multi_pattern {
                    base
                } else {
                    base.saturating_sub(max_overlap_blocks).div_ceil(LANES)
                }
            }
            MultiText::Multi(ts) => ts
                .iter()
                .map(|t| t.len() + text_padding)
                .max()
                .unwrap_or(0)
                .div_ceil(64),
        };

        let chunk_offset_blocks = if is_single_text && !is_multi_pattern {
            blocks_per_chunk
        } else {
            0
        };

        for lane in 0..LANES {
            self.lanes[lane].chunk_offset = lane * chunk_offset_blocks;
            self.lanes[lane].lane_end = (lane + 1) * chunk_offset_blocks * 64;
            self.lanes[lane].matches.clear();
            self.lanes[lane].decreasing = true;
        }

        self.hp.clear();
        self.hm.clear();
        self.hp.resize(pattern_len, S::splat(1));
        self.hm.resize(pattern_len, S::splat(0));

        if is_multi_pattern || !is_single_text {
            init_deltas_for_overshoot_all_lanes(&mut self.hp, self.alpha, self.max_overhang);
        } else {
            init_deltas_for_overshoot(&mut self.hp, self.alpha, self.max_overhang);
        }

        (blocks_per_chunk, max_overlap_blocks, pattern_len)
    }

    #[inline(always)]
    fn search_internal<'t, F>(
        &mut self,
        pattern: MultiPattern<'t>,
        profiler: P,
        mut get_eq: F,
        text: MultiText<'t>,
        k: Cost,
        all_minima: bool,
        is_multi_pattern: bool,
    ) where
        F: FnMut(usize, &[LaneState<P>; LANES]) -> S,
    {
        // Terminology:
        // - chunk: roughly 1/4th of the input text, with small overlaps.
        // - block: 64 bytes of text.
        // - lane: a u64 of a SIMD vec.

        // The pattern will match a pattern of length at most pattern.len() + k.
        // We round that up to a multiple of 64 to find the number of blocks overlap between chunks.
        let is_single_text = matches!(text, MultiText::One(_));
        let (blocks_per_chunk, max_overlap_blocks, pattern_len) =
            self.search_prep(&pattern, &text, k, is_multi_pattern);

        let mut prev_max_j = 0;
        let mut prev_end_last_below = 0;

        'text_chunk: for i in 0..blocks_per_chunk + max_overlap_blocks {
            let (mut vp, mut vm) = (S::splat(0), S::splat(0));

            for lane in 0..LANES {
                if let Some(t) = text.get_lane(lane) {
                    self.lanes[lane].update_and_encode(t, i, &profiler, self.alpha.is_some());
                }
            }

            let (mut dist_to_start, mut dist_to_end) = (S::splat(0), S::splat(0));
            let mut cur_end_last_below = 0;

            // Iterate over pattern chars (rows in the DP matrix)
            for j in 0..pattern_len {
                // These are needed to prevent bound checks on `self.hp[j]`.
                // I'm not quite sure why those aren't optimized away automatically.
                unsafe {
                    assert_unchecked(j < self.hp.len());
                    assert_unchecked(j < self.hm.len());
                }
                let hp: &mut wide::u64x4 = &mut self.hp[j];
                let hm = &mut self.hm[j];

                dist_to_start += *hp;
                dist_to_start -= *hm;

                let eq = get_eq(j, &self.lanes);
                compute_block_simd(hp, hm, &mut vp, &mut vm, eq);

                // Early termination check: If we've moved past the last row that had any
                // promising matches (cost <= k), we can potentially skip ahead or terminate
                'check: {
                    dist_to_end += *hp;
                    dist_to_end -= *hm;

                    // Check if any lane has cost <= k (ie <k+1) at the current row
                    let cmp = dist_to_end.simd_lt(S::splat(k as u64 + 1));
                    if cmp != wide::u64x4::splat(0) {
                        // Track the highest row where we found any promising matches

                        cur_end_last_below = j;
                    }

                    // Only do early termination checks if we've moved past the last promising row
                    if j > prev_end_last_below {
                        // Check if any lane has a minimum cost that could lead to matches <= k
                        if let Some(new_end) = self.check_lanes(&vp, &vm, &dist_to_start, k, j) {
                            // Found a promising lane - update our tracking and continue
                            prev_end_last_below = new_end;
                            break 'check;
                        }
                        // No lanes have promising matches - we can skip ahead
                        self.reset_rows(j + 1, prev_max_j);
                        prev_end_last_below = cur_end_last_below.max(Self::CHECK_AT_LEAST_ROWS);
                        prev_max_j = j;

                        // Early termination: if we're in overlap region and too far from text end
                        if self.should_terminate_early(i, blocks_per_chunk, j, k) {
                            break 'text_chunk;
                        }
                        continue 'text_chunk;
                    }
                }
            }

            // Save positions with cost <= k directly after processing each row
            for lane in 0..LANES {
                let v = V::from(vp.as_array()[lane], vm.as_array()[lane]);
                let cost = dist_to_start.as_array()[lane] as Cost;

                // `check_lanes` only happens at most every 8 rows,
                // so for short patterns there's a good chance things are bad by now.
                if self.min_in_lane(v, lane, &dist_to_start) <= k {
                    // Both pattern and text should exist, guaranteed?
                    if let (Some(t), Some(_p)) = (text.get_lane(lane), pattern.get_lane(lane)) {
                        let base_pos = self.lanes[lane].chunk_offset * 64 + 64 * i;
                        self.find_minima_with_overhang(
                            v,
                            cost,
                            k,
                            t.len(),
                            pattern_len,
                            base_pos,
                            lane,
                            all_minima,
                        );
                    }
                }
            }
            prev_end_last_below = cur_end_last_below.max(Self::CHECK_AT_LEAST_ROWS);
            prev_max_j = pattern_len - 1;
        }

        self.reset_rows(0, prev_max_j);

        // If text was chunked we have to prune
        if is_single_text && !is_multi_pattern {
            self.prune_lane_overlaps();
        }
    }

    #[inline(always)]
    fn prune_lane_overlaps(&mut self) {
        if log::log_enabled!(log::Level::Trace) {
            self.lanes[0].matches.iter().for_each(|&(end_pos, cost)| {
                log::trace!("lane 0 KEEP {end_pos} {cost}");
            });
        }

        let cur_lane_end = self.lanes[0].lane_end;
        self.lanes[0].matches.retain(|&(end_pos, cost)| {
            if end_pos >= cur_lane_end {
                log::trace!("lane 0 drop {end_pos} {cost} because it's >= lane end {cur_lane_end}");
            }
            end_pos < cur_lane_end
        });

        for lane in 1..LANES {
            let prev_lane_end = self.lanes[lane - 1].lane_end;
            let cur_lane_end = self.lanes[lane].lane_end;
            log::trace!("End of lane {}: {prev_lane_end}", lane - 1);
            self.lanes[lane].matches.retain(|&(end_pos, cost)| {
                let too_early = end_pos < prev_lane_end;
                let too_late = end_pos >= cur_lane_end && lane != LANES - 1;

                if too_early {
                    log::trace!(
                        "lane {lane} drop {end_pos} {cost} because it's before {prev_lane_end}"
                    );
                } else if too_late {
                    log::trace!(
                        "lane {lane} drop {end_pos} {cost} because it's >= lane end {cur_lane_end}"
                    );
                } else {
                    log::trace!("lane {lane} KEEP {end_pos} {cost}");
                }

                !too_early && !too_late
            });
        }
    }

    /// Reset rows that are no longer needed for future computations
    #[inline(always)]
    fn reset_rows(&mut self, from_row: usize, to_row: usize) {
        for j2 in from_row..=to_row {
            self.hp[j2] = S::splat(1);
            self.hm[j2] = S::splat(0);
        }
    }

    /// Check if we should terminate early based on position and distance from text end
    #[inline(always)]
    fn should_terminate_early(
        &self,
        current_block: usize,
        blocks_per_chunk: usize,
        current_row: usize,
        k: Cost,
    ) -> bool {
        // Only consider early termination in the overlap region (after main chunks)
        if current_block < blocks_per_chunk {
            return false;
        }

        // Calculate how far we are from the end of the text
        let distance_from_end =
            (64 * (current_block - blocks_per_chunk)).saturating_sub(current_row);

        // If we're too far from the end, no matches are possible
        distance_from_end > k as usize
    }

    #[inline(always)]
    fn add_overshoot_cost(&self, cost: Cost, pos: usize, text_len: usize) -> Cost {
        let overshoot = pos.saturating_sub(text_len);
        let overshoot_cost = if self.alpha.is_none() || overshoot == 0 {
            0
        } else {
            (self.alpha.unwrap() * overshoot as f32).floor() as Cost
        };
        cost + overshoot_cost
    }

    #[inline(always)]
    fn find_minima_with_overhang(
        &mut self,
        v: V,
        cur_cost: Cost,
        k: Cost,
        text_len: usize,
        pattern_len: usize,
        base_pos: usize,
        lane: usize,
        all_minima: bool,
    ) {
        let (p, m) = v.pm();
        let max_pos = if self.alpha.is_some() {
            text_len
                + pattern_len
                    .min(((k as f32) / self.alpha.unwrap()).ceil() as usize)
                    .min(self.max_overhang.unwrap_or(usize::MAX))
        } else {
            text_len
        };

        let mut cost = cur_cost;
        let mut prev_cost = self.add_overshoot_cost(cur_cost, base_pos, text_len);
        let mut prev_pos = base_pos;

        if base_pos >= max_pos {
            return;
        }

        for bit in 1..=64 {
            let pos: usize = base_pos + bit;
            if pos > max_pos {
                break;
            }

            cost += ((p >> (bit - 1)) & 1) as Cost;
            cost -= ((m >> (bit - 1)) & 1) as Cost;

            let total_cost = self.add_overshoot_cost(cost, pos, text_len);

            if all_minima {
                if total_cost <= k {
                    log::debug!("MATCH: lane {lane} push {prev_pos} {prev_cost}");
                    self.lanes[lane].matches.push((pos, total_cost));
                }
            } else {
                log::trace!("lane {lane}      {pos} {total_cost}");
                // Local minima
                // Check how costs are changing
                let costs_are_equal = total_cost == prev_cost;
                let costs_are_increasing = total_cost > prev_cost;
                let costs_are_decreasing = total_cost < prev_cost;

                // Found a local minimum if we were decreasing and now costs are increasing
                if self.lanes[lane].decreasing && costs_are_increasing && prev_cost <= k {
                    log::debug!("MATCH: lane {lane} push {prev_pos} {prev_cost}");
                    self.lanes[lane].matches.push((prev_pos, prev_cost));
                }

                // Update decreasing state:
                // - If costs are decreasing, we're in a decreasing sequence
                // - If costs are equal, keep the previous state
                // - If costs are increasing, we're not decreasing
                self.lanes[lane].decreasing =
                    costs_are_decreasing || (self.lanes[lane].decreasing && costs_are_equal);
            }

            prev_cost = total_cost;
            prev_pos = pos;
        }
        if !all_minima && prev_pos == max_pos && self.lanes[lane].decreasing && prev_cost <= k {
            log::debug!("lane {lane} push {prev_pos} {prev_cost} <last>");
            self.lanes[lane].matches.push((prev_pos, prev_cost));
        }
    }

    /// Appends results to `self.idx_matches`, and returns the new slice.
    fn process_matches<'t>(
        &mut self,
        pattern: MultiPattern<'t>,
        text: MultiText<'t>,
        k: Cost,
    ) -> &mut [Match] {
        let old_len = self.matches.len();
        let fill_len = pattern.len() + k as usize;

        let multi_text = matches!(text, MultiText::Multi(_));
        let multi_pattern = matches!(pattern, MultiPattern::Multi(_));
        assert!(
            !(multi_text && multi_pattern),
            "process_matches does not support both
        multiple patterns and multiple text at the same time"
        );

        // Collect slices to process in batches
        let mut batch = MatchBatch::new();

        if self.only_best_match {
            // only return 1 best match per (pattern, text) pair
            let mut best = [(Cost::MAX, Reverse(0), 0, [].as_slice(), [].as_slice()); LANES];
            for lane in 0..LANES {
                for &(end_pos, cost) in &self.lanes[lane].matches {
                    let offset = end_pos.saturating_sub(fill_len);
                    let Some(t) = text.get_lane(lane) else {
                        break;
                    };
                    let slice = &t[offset..end_pos.min(t.len())];

                    // rightmost match with minimal cost
                    best[lane] = best[lane].min((
                        cost,
                        Reverse(end_pos),
                        offset,
                        pattern.get_lane(lane).unwrap(), // We always should have a pattern here
                        slice,
                    ));
                }
            }

            let mut add_match =
                |pattern_idx,
                 text_idx,
                 t: &'t [u8],
                 best: (i32, Reverse<usize>, usize, &'t [u8], &'t [u8])| {
                    let (cost, Reverse(end_pos), offset, pattern, slice) = best;
                    if self.without_trace {
                        self.matches.push(Match {
                            pattern_idx,
                            text_idx,
                            text_start: usize::MAX,
                            text_end: end_pos.min(t.len()),
                            pattern_start: usize::MAX,
                            pattern_end: pattern.len() - end_pos.saturating_sub(t.len()),
                            cost,
                            strand: Strand::Fwd,
                            cigar: Cigar::default(),
                        });
                    } else {
                        batch.add(pattern_idx, text_idx, pattern, slice, offset, end_pos, cost);
                    }
                };

            if !multi_text && !multi_pattern {
                // the lanes are text chunks, minimize over them
                let best = *best.iter().min().unwrap();
                if best.0 != Cost::MAX {
                    add_match(0, 0, text.get_lane(0).unwrap(), best);
                }
            } else {
                // return a minimum per lane
                for lane in 0..LANES {
                    if best[lane].0 != Cost::MAX {
                        add_match(
                            if multi_pattern { lane } else { 0 },
                            if multi_text { lane } else { 0 },
                            text.get_lane(lane).unwrap(),
                            best[lane],
                        );
                    }
                }
            }
        } else {
            for lane in 0..LANES {
                // FIXME: Fix pattern_idx & text_idx
                let Some(t) = text.get_lane(lane) else {
                    break;
                };
                for &(end_pos, cost) in &self.lanes[lane].matches {
                    let offset = end_pos.saturating_sub(fill_len);
                    let slice = &t[offset..end_pos.min(t.len())];

                    batch.add(
                        if multi_pattern { lane } else { 0 },
                        if multi_text { lane } else { 0 },
                        pattern.get_lane(lane).unwrap(), // guaranteed?
                        slice,
                        offset,
                        end_pos,
                        cost,
                    );

                    if batch.is_full() {
                        batch.process::<P>(
                            fill_len,
                            &mut self.cost_matrices,
                            self.alpha,
                            self.max_overhang,
                            k,
                            &mut self.matches,
                        );
                        batch.clear();
                    }
                }
            }
        }

        if !batch.is_empty() {
            batch.process::<P>(
                fill_len,
                &mut self.cost_matrices,
                self.alpha,
                self.max_overhang,
                k,
                &mut self.matches,
            );
        }
        &mut self.matches[old_len..]
    }
}

fn map_collect_cartesian_product<
    X: Sync,
    Y: Sync,
    O: Send,
    C: Clone + Sync,
    F: Fn(&mut C, &X, &Y, usize, usize) -> Vec<O> + Sync,
>(
    xs: &[X],
    ys: &[Y],
    context: &C,
    f: F,
) -> Vec<O> {
    let xn = xs.len();
    let yn = ys.len();
    let f = &f;

    (0..xn * yn)
        .into_par_iter()
        .map_init(
            || context.clone(),
            move |context, index| {
                let xi = index / yn;
                let yi = index % yn;
                let x = &xs[xi];
                let y = &ys[yi];
                f(context, x, y, xi, yi)
            },
        )
        .flatten_iter()
        .collect()
}

struct MatchBatch<'a> {
    pattern_idx: [usize; LANES],
    text_idx: [usize; LANES],
    patterns: [&'a [u8]; LANES],
    text_slices: [&'a [u8]; LANES],
    offsets: [usize; LANES],
    ends: [usize; LANES],
    expected_costs: [Cost; LANES],
    count: usize,
}

impl<'a> MatchBatch<'a> {
    fn new() -> Self {
        Self {
            pattern_idx: [0; LANES],
            text_idx: [0; LANES],
            patterns: [b""; LANES],
            text_slices: [b""; LANES],
            offsets: [0; LANES],
            ends: [0; LANES],
            expected_costs: [0; LANES],
            count: 0,
        }
    }

    fn add(
        &mut self,
        pattern_idx: usize,
        text_idx: usize,
        pattern: &'a [u8],
        slice: &'a [u8],
        offset: usize,
        end: usize,
        cost: Cost,
    ) {
        self.pattern_idx[self.count] = pattern_idx;
        self.text_idx[self.count] = text_idx;
        self.patterns[self.count] = pattern;
        self.text_slices[self.count] = slice;
        self.offsets[self.count] = offset;
        self.ends[self.count] = end;
        self.expected_costs[self.count] = cost;
        self.count += 1;
    }

    fn is_full(&self) -> bool {
        self.count == LANES
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn clear(&mut self) {
        self.count = 0;
        // Don't have to clear all data as add will keep track of count
        // which process uses to make sure it only uses filled data
    }

    /// Returns pairs of (lane, Match).
    fn process<P: Profile>(
        &self,
        fill_len: usize,
        cost_matrices: &mut [CostMatrix; LANES],
        alpha: Option<f32>,
        max_overhang: Option<usize>,
        k: Cost,
        out: &mut Vec<Match>,
    ) {
        if self.count > 1 {
            let equal_patterns = self
                .patterns
                .iter()
                .take(self.count)
                .all(|p| *p == self.patterns[0]);
            if equal_patterns {
                simd_fill::<P>(
                    &self.patterns[0],
                    &self.text_slices[..self.count],
                    fill_len,
                    cost_matrices,
                    alpha,
                    max_overhang,
                );
            } else {
                simd_fill_multipattern::<P>(
                    &self.patterns[..self.count],
                    &self.text_slices[..self.count],
                    fill_len,
                    cost_matrices,
                    alpha,
                    max_overhang,
                );
            }
        } else {
            fill::<P>(
                &self.patterns[0],
                self.text_slices[0],
                fill_len,
                &mut cost_matrices[0],
                alpha,
                max_overhang,
            );
        }

        for i in 0..self.count {
            let mut m = get_trace::<P>(
                &self.patterns[i],
                self.offsets[i],
                self.ends[i],
                self.text_slices[i],
                &cost_matrices[i],
                alpha,
                max_overhang,
            );
            m.pattern_idx = self.pattern_idx[i];
            m.text_idx = self.text_idx[i];

            // Check if get_trace cost is same as expected end position cost
            assert!(
                m.cost <= self.expected_costs[i],
                "Match has unexpected cost {} > {}: {m:?}",
                m.cost,
                self.expected_costs[i],
            );

            // Make sure it's also <=k
            assert!(
                m.cost <= k,
                "Match exceeds k after traceback: m.cost={}, k={}",
                m.cost,
                k,
            );

            out.push(m);
        }
    }
}

/// Assumes hp and hm are already the right size, hm=0 and hp=1.
/// Then sets hp according to the given alpha, if needed.
#[inline(always)]
pub(crate) fn init_deltas_for_overshoot_scalar(
    h: &mut [H],
    alpha: Option<f32>,
    max_overhang: Option<usize>,
) {
    if let Some(alpha) = alpha {
        for i in 0..h.len().min(max_overhang.unwrap_or(usize::MAX)) {
            // Alternate 0 and 1 costs at very left of the matrix.
            // (Note: not at start of later chunks.)
            // FIXME: floor, round, or ceil?
            h[i].0 =
                (((i + 1) as f32) * alpha).floor() as u64 - ((i as f32) * alpha).floor() as u64;
        }
    }
}

/// Assumes hp and hm are already the right size, hm=0 and hp=1.
/// Then sets hp according to the given alpha, if needed.
#[inline(always)]
pub(crate) fn init_deltas_for_overshoot(
    hp: &mut [S],
    alpha: Option<f32>,
    max_overhang: Option<usize>,
) {
    if let Some(alpha) = alpha {
        for i in 0..hp.len().min(max_overhang.unwrap_or(usize::MAX)) {
            // Alternate 0 and 1 costs at very left of the matrix.
            // (Note: not at start of later chunks.)
            // FIXME: floor, round, or ceil?
            hp[i].as_mut_array()[0] =
                (((i + 1) as f32) * alpha).floor() as u64 - ((i as f32) * alpha).floor() as u64;
        }
    }
}

/// Assumes hp and hm are already the right size, hm=0 and hp=1.
/// Then sets hp according to the given alpha, if needed.
#[inline(always)]
pub(crate) fn init_deltas_for_overshoot_all_lanes(
    hp: &mut [S],
    alpha: Option<f32>,
    max_overhang: Option<usize>,
) {
    if let Some(alpha) = alpha {
        for i in 0..hp.len().min(max_overhang.unwrap_or(usize::MAX)) {
            // Alternate 0 and 1 costs at very left of the matrix.
            // (Note: not at start of later chunks.)
            // FIXME: floor, round, or ceil?
            let bit =
                (((i + 1) as f32) * alpha).floor() as u64 - ((i as f32) * alpha).floor() as u64;
            hp[i].as_mut_array().fill(bit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::{Dna, Iupac};
    use itertools::Itertools;
    use rand::random_range;

    #[test]
    fn overhang_test() {
        let pattern = b"CTTAAGCACTACCGGCTAAT";
        let text = b"AGTCGTCCTTTGCGAGCTCGGACATCTCCAGGCGAACCTGCAAGTTTTAATGTTCCCACAGTCCCTCATATGTTCTGAATTTCGTGATGTTTGTTTACCG";
        let mut s = Searcher::<Iupac>::new_fwd_with_overhang(0.0);
        let _matches = s.search_all(pattern, text, 100);
    }

    #[test]
    #[should_panic]
    fn overhang_test_panic_for_dna() {
        Searcher::<Dna>::new_fwd_with_overhang(0.0);
    }

    #[test]
    fn overshoot() {
        let pattern = b"CCCTTTCCCGGG";
        let text = b"AAAAAAAAACCCTTT".as_slice();
        let mut s = Searcher::<Iupac>::new_fwd();
        s.alpha = Some(0.5);
        s.search_positions_bounded(MultiPattern::One(pattern), MultiText::One(text), 10, true);
        for l in s.lanes {
            println!("Matches: {:?}", l.matches);
        }
    }

    #[test]
    fn overshoot_test_prefix_trace() {
        let pattern = b"CCCTTTCCCGGG";
        let text = b"AAAAAAAAACCCTTT";
        let mut s = Searcher::<Iupac>::new_fwd();
        s.alpha = Some(0.5);
        s.search_all(pattern, text, 10);
        // First not error
    }

    #[test]
    fn overshoot_simple_prefix() {
        /*
        AAAAGGGG
            ||||
            GGGGTTTTTTTTTTTTTTTTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        01234567---
            0123---
        */
        let prefix = "AAAAGGGG";
        let text = "GGGGTTTTTTTTTTTTTTTT";
        let mut s = Searcher::<Iupac>::new_fwd();
        s.alpha = Some(0.5);
        s.search_positions_bounded(
            MultiPattern::One(prefix.as_bytes()),
            MultiText::One(text.as_bytes()),
            2,
            true,
        );
        let expected_idx = 3;
        let expected_edits = 2 as Cost;
        let m = s.lanes[0]
            .matches
            .iter()
            .find(|m| m.0 == expected_idx && m.1 <= expected_edits);
        assert!(m.is_some());
    }

    #[test]
    fn overshoot_simple_suffix() {
        /*
                            GGGGAAAA
                            ||||
            TTTTTTTTTTTTTTTTGGGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
            0123456789-123456789-12345
                                   ^23,24,25, etc
        */
        let prefix = "GGGGAAAA";
        let text = "TTTTTTTTTTTTTTTTGGGG";
        let mut s = Searcher::<Iupac>::new_fwd();
        s.alpha = Some(0.5);
        s.search_positions_bounded(
            MultiPattern::One(prefix.as_bytes()),
            MultiText::One(text.as_bytes()),
            2,
            true,
        );
        let expected_idx = 24;
        let expected_edits = 2 as Cost;
        let m = s.lanes[0]
            .matches
            .iter()
            .find(|m| m.0 == expected_idx && m.1 <= expected_edits);
        assert!(m.is_some());
    }

    #[test]
    fn overshoot_simple_suffix_local_minima() {
        /*
                            GGGGAAAA
                            ||||
            TTTTTTTTTTTTTTTTGGGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
            0123456789-123456789-12345
                                   ^23,24,25, etc
        */
        let prefix = b"GGGGAAAA";
        let text = b"TTTTTTTTTTTTTTTTGGGG";
        let mut s = Searcher::<Iupac>::new_fwd();
        s.alpha = Some(0.5);
        let matches = s.search(prefix, text, 4);
        let expected_edits = 2 as Cost;
        for m in matches.iter() {
            println!("Match: {:?}", m.without_cigar());
        }
        let m = matches
            .iter()
            .find(|m| m.text_end == 20 && m.pattern_end == 3 && m.cost == expected_edits);
        assert!(m.is_some());
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn overshoot_test_prefix_and_suffix() {
        /*
              AAAAGGGG                 AAAAGGGG
                  ||||                 ||||
                  GGGGGAAAAA     GGGGGAAAAANNNN
                  0123456789     0123456789-123
                     ^ 3                      ^ 13
        */
        let contained = "AAAAGGGG";
        let text = "GGGGGAAAAA";
        let mut s = Searcher::<Iupac>::new_fwd();
        s.alpha = Some(0.5);
        s.search_positions_bounded(
            MultiPattern::One(contained.as_bytes()),
            MultiText::One(text.as_bytes()),
            2,
            true,
        );
        let expected_indices = [3, 13];
        let expected_edits = [2, 2];
        let mut found = [false, false];
        for m in s.lanes[0].matches.iter() {
            for j in 0..expected_indices.len() {
                if m.0 == expected_indices[j] && m.1 == expected_edits[j] {
                    found[j] = true;
                }
            }
        }
        assert!(found[0]);
        assert!(found[1]);
    }

    #[test]
    fn test_case1() {
        let pattern = b"AGATGTGTCC";
        let text = b"GAGAGATAACCGTGCGCTCACTGTTACAGTTTATGTGTCGAATTCTTTTAGGGCTGGTCACTGCCCATGCGTAGGAATGAATAGCGGTTGCGGATAACTAAGCAGTGCTTGTCGCTATTAAAGTTAGACCCCGCTCCCACCTTGCCACTCCTAGATGTCGACGAGATTCTCACCGCCAAGGGATCAGGCATATCATAGTAGCTGGCGCAGCCCGTCGTTTAAGGAGGCCCATATACTAATTCAAACGATGGGGTGGGCACATCCCCTAGAAGCACTTGTCCCTTGAGTCACGACTGATGCGTGGTCTCTCGCTAAATGTTCCGGCCTCTCGGACATTTAAAGGGTGGCATGTGACCATGGAGGATTAGTGAATGAGAGGTGTCCGCCTTTGTTCGCCAGAACTCTTATAGCGTAGGGGAGTGTACTCACCGCGAACCCGTATCAGCAATCTTGTCAGTGGCTCCTGACTCAAACATCGATGCGCTGCACATGGCCTTAGAATGAAGCAGCCCCTTTCTATTGTGGCCGGGCTGATTCTTTGTTTTGTTGAATGGTCGGGCCTGTCTGCCTTTTCCTAGTGTTGAAACTCCGAACCGCATGAACTGCGTTGCTAGCGAGCTATCACTGGACTGGCCGGGGGACGAAAGTTCGCGGGACCCACTACCCGCGCCCAGAAGACCACACTAGGGAGAAGGATTCTATCGGCATAGCCGTC";
        let mut searcher = Searcher::<Dna>::new_rc();
        let matches = searcher.search(pattern, &text, 2);
        println!("Matches: {:?}", matches);
    }

    #[test]
    fn no_extra_matches() {
        let edits = 6;
        let expected_idx = 277;
        let pattern = b"TAAGCAGAAGGGAGGTATAAAGTCTGTCAGCGGTGCTTAAG";
        let text = b"ACCGTAACCGCTTGGTACCATCCGGCCAGTCGCTCGTTGCGCCCCACTATCGGGATCGACGCGCAGTAATTAAACACCACCCACGCCACGAGGTAGAACGAGAGCGGGGGGCTAGCAAATAATAGTGAGAGTGCGTTCAAAGGGTCTTTCGTAACCTCAGCGGGCGGGTACGGGGGAAATATCGCACCAATTTTGGAGATGCGATTAGCTCAGCGTAACGCGAATTCCCTATAACTTGCCTAGTGTGTGTGAATGGACAATTCGTTTTACAGTTTCAAGGTAGCAGAAGGGCAGGATAAGTCTGTCGCGGTGCTTAAGGCTTTCCATCCATGTTGCCCCCTACATGAATCGGATCGCCAGCCAGAATATCACATGGTTCCAAAAGTTGCAAGCTTCCCCGTACCGCTACTTCACCTCACGCCAGAGGCCTATCGCCGCTCGGCCGTTCCGTTTTGGGGAAGAATCTGCCTGTTCTCGTCACAAGCTTCTTAGTCCTTCCACCATGGTGCTGTTACTCATGCCATCAAATATTCGAGCTCTTGCCTAGGGGGGTTATACCTGTGCGATAGATACACCCCCTATGACCGTAGGTAGAGAGCCTATTTTCAACGTGTCGATCGTTTAATGACACCAACTCCCGGTGTCGAGGTCCCCAAGTTTCGTAGATCTACTGAGCGGGGGAATATTTGACGGTAAGGCATCGCTTGTAGGATCGTATCGCGACGGTAGATACCCATAAGCGTTGCTAACCTGCCAATAACTGTCTCGCGATCCCAATTTAGCACAAGTCGGTGGCCTTGATAAGGCTAACCAGTTTCGCACCGCTTCCGTTCCATTTTACGATCTACCGCTCGGATGGATCCGAAATACCGAGGTAGTAATATCAACACGTACCCAATGTCC";
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, &text, edits);
        let m = matches
            .iter()
            .find(|m| m.text_start.abs_diff(expected_idx) <= edits);
        assert!(m.is_some());
    }

    fn random_dna_string(len: usize) -> Vec<u8> {
        (0..len).map(|_| b"ACGT"[random_range(0..4)]).collect()
    }

    // Just for profiling
    #[test]
    #[ignore = "for profiling only"]
    fn random_big_search() {
        let mut total_matches = 0;
        for _ in 0..1000 {
            let pattern = random_dna_string(random_range(10..100));
            let text = random_dna_string(1_000_000);
            let mut searcher = Searcher::<Dna>::new_fwd();
            let matches = searcher.search(&pattern, &text, 5);
            total_matches += matches.len();
        }
        println!("total matches: {total_matches}");
    }

    #[test]
    fn test_fwd_rc_search() {
        let pattern = b"ATCGATCA";
        let rc = Dna::reverse_complement(pattern);
        let text = [b"GGGGGGGG".as_ref(), &rc, b"GGGGGGGG"].concat();
        let mut searcher = Searcher::<Dna>::new_rc();
        let matches = searcher.search(pattern, &text, 0);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text_start, 8);
        assert_eq!(matches[0].text_end, 8 + pattern.len());
        // Now disableing rc search should yield no matches
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, &text, 0);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_filter_fn_simple() {
        let pattern = b"ATCGATCA";
        let mut text = vec![b'G'; 100];

        // Insert match once before 10 and once after 10
        text.splice(10..10, pattern.iter().copied());
        text.splice(50..50, pattern.iter().copied());
        let end_filter = |q: &[u8], text: &[u8], _strand: Strand| text.len() > 10 + q.len();
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search_with_fn(pattern, &text, 0, false, end_filter);
        assert_eq!(matches.len(), 1); // First match *ending* at 10 should be discarded
        assert_eq!(matches[0].text_start, 50);

        // Sanity check, run the same without filter
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, &text, 0);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].text_start, 10);
        assert_eq!(matches[1].text_start, 50);
    }

    fn complement(text: &[u8]) -> Vec<u8> {
        let mut complement = text.to_vec();
        complement.iter_mut().for_each(|c| {
            *c = match *c {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                b'T' => b'A',
                _ => *c,
            }
        });
        complement
    }

    #[test]
    fn test_filter_fn_rc() {
        let pattern_fwd = b"ATCGATCA";
        let pattern_rc = Dna::reverse_complement(pattern_fwd);
        let mut text = vec![b'G'; 100];

        // Insert match once before 10 and once after 10
        text.splice(10..10, pattern_fwd.iter().copied()); // FWD
        text.splice(50..50, pattern_rc.iter().copied()); // RC

        let end_filter = |q: &[u8], text: &[u8], strand: Strand| match strand {
            Strand::Fwd => text[text.len() - q.len()..] == *pattern_fwd,
            Strand::Rc => {
                complement(&text[text.len() - q.len()..]) == *pattern_fwd // NOTE complement call
            }
        };

        let mut searcher = Searcher::<Dna>::new_rc();
        let matches = searcher.search_with_fn(pattern_fwd, &text, 0, false, end_filter);
        assert_eq!(matches.len(), 2); // Both matches should be found
        assert_eq!(matches[0].text_start, 10);
        assert_eq!(matches[1].text_start, 50);
    }

    #[test]
    fn search_fuzz() {
        let mut pattern_lens = (10..20)
            .chain((0..10).map(|_| random_range(10..100)))
            .chain((0..10).map(|_| random_range(100..1000)))
            .collect::<Vec<_>>();

        let mut text_lens = (10..20)
            .chain((0..10).map(|_| random_range(10..100)))
            .chain((0..10).map(|_| random_range(100..1000)))
            .chain((0..10).map(|_| random_range(1000..10000)))
            .collect::<Vec<_>>();

        pattern_lens.sort();
        text_lens.sort();

        // Create single searcher for all tests to check proper resetting of internal states
        let mut searcher = Searcher::<Dna>::new_fwd();
        let mut rc_searcher = Searcher::<Dna>::new_rc();

        for pattern_len in pattern_lens {
            for t in text_lens.clone() {
                println!("q {pattern_len} t {t}");
                let pattern = (0..pattern_len)
                    .map(|_| b"ACGT"[random_range(0..4)])
                    .collect::<Vec<_>>();
                let mut text = (0..t)
                    .map(|_| b"ACGT"[random_range(0..4)])
                    .collect::<Vec<_>>();

                let edits = random_range(0..pattern_len / 3);
                let mut p_mutated = pattern.clone();
                for _ in 0..edits {
                    let tp = random_range(0..3);
                    match tp {
                        0 => {
                            // insert
                            let idx = random_range(0..=p_mutated.len());
                            p_mutated.insert(idx, b"ACGT"[random_range(0..4)]);
                        }
                        1 => {
                            // del
                            let idx = random_range(0..p_mutated.len());
                            p_mutated.remove(idx);
                        }
                        2 => {
                            // replace
                            let idx = random_range(0..p_mutated.len());
                            p_mutated[idx] = b"ACGT"[random_range(0..4)];
                        }
                        _ => panic!(),
                    }
                }

                fn show(x: &[u8]) -> &str {
                    str::from_utf8(x).unwrap()
                }
                eprintln!("\n");
                eprintln!("edits {edits}");
                eprintln!("Search pattern p={pattern_len} {}", show(&pattern));
                eprintln!("Inserted pattern {}", show(&p_mutated));

                if p_mutated.len() > text.len() {
                    continue;
                }

                let idx = random_range(0..=text.len().saturating_sub(p_mutated.len()));
                eprintln!("text len {}", text.len());
                eprintln!("planted idx {idx}");
                let expected_idx = (idx + p_mutated.len()).saturating_sub(pattern_len);
                eprintln!("expected idx {expected_idx}");

                text.splice(idx..idx + p_mutated.len(), p_mutated);
                eprintln!("text {}", show(&text));

                // Just fwd
                let matches = searcher.search(&pattern, &text, edits);
                eprintln!("matches {matches:?}");
                let m = matches
                    .iter()
                    .find(|m| m.text_start.abs_diff(expected_idx) <= edits);
                assert!(m.is_some());

                // Also rc search, should still find the same match
                let matches = rc_searcher.search(&pattern, &text, edits);

                eprintln!("matches {matches:?}");
                let m = matches
                    .iter()
                    .find(|m| m.text_start.abs_diff(expected_idx) <= edits);
                assert!(m.is_some());

                // search_texts should give the same result
                let multi_matches = searcher.search_texts(&pattern, &[&text, &text], edits);

                eprintln!("multi matches {multi_matches:?}");
                let multi_matches = multi_matches
                    .into_iter()
                    .filter(|m| m.text_idx == 1)
                    .collect::<Vec<_>>();
                let m = multi_matches
                    .iter()
                    .find(|m| m.text_start.abs_diff(expected_idx) <= edits);
                assert!(m.is_some());
            }
        }
    }

    #[test]
    // #[ignore = "for plotting only"]
    fn print_matches() {
        let pattern = b"GCCGT";
        let text = b"AGCGCGTA";
        let k = 1;
        let mut searcher = Searcher::<Dna>::new_rc();
        let matches = searcher.search_all(pattern, text, k);
        let match_local = searcher.search(pattern, text, k);
        //   println!("matches: {:?}", matches);
        println!("fwd matches (ALL): {}", matches.len());
        for m in matches {
            println!("m: {:?}", m.without_cigar());
        }
        println!("local matches: {}", match_local.len());
        for m in match_local {
            println!("m: {:?}", m.without_cigar());
        }
        let pattern_rev = pattern.iter().rev().copied().collect::<Vec<_>>();
        let text_rev = text.iter().rev().copied().collect::<Vec<_>>();
        let matches_rev = searcher.search_all(&pattern_rev, &text_rev, k);
        let match_rev_local = searcher.search(&pattern_rev, &text_rev, k);
        println!("rev matches (ALL): {}", matches_rev.len());
        for m in matches_rev {
            println!("m: {:?}", m.without_cigar());
        }
        println!("rev local matches: {}", match_rev_local.len());
        for m in match_rev_local {
            println!("m: {:?}", m.without_cigar());
        }
    }

    #[test]
    fn test_fixed_matches() {
        let pattern = b"ATCGATCA";
        let mut text = vec![b'G'; 1000]; // Create a text of 1000 G's

        // Insert 5 matches at fixed positions
        let positions = [50, 150, 250, 350, 450, 800];
        let mut expected_matches = Vec::new();

        for &pos in &positions {
            // Insert the pattern at each position
            text.splice(pos..pos + pattern.len(), pattern.iter().copied());

            // Record expected match position
            let expected_idx = (pos + pattern.len()).saturating_sub(pattern.len());
            expected_matches.push(expected_idx);
        }

        // Test forward search
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search_all(pattern, &text, 1);

        // Verify all matches are found
        for expected_idx in expected_matches {
            let found = matches.iter().any(|m| m.text_start == expected_idx);
            assert!(found, "Expected match at {} not found", expected_idx);
        }

        for m in matches {
            println!("match: {:?}", m);
        }
    }

    #[test]
    fn overhang_trace_fuzz() {
        // env_logger::init();

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::iter::repeat_with;

        let mut rng = StdRng::seed_from_u64(42);
        let mut searcher = Searcher::<Iupac>::new_fwd();
        searcher.alpha = Some(0.5);

        fn rand_dna_w_seed(len: usize, rng: &mut StdRng) -> Vec<u8> {
            repeat_with(|| {
                let n = rng.random_range(0..4);
                match n {
                    0 => b'A',
                    1 => b'C',
                    2 => b'G',
                    _ => b'T',
                }
            })
            .take(len)
            .collect()
        }

        let mut skipped = 0;
        let iter = 10000;

        for _ in 0..iter {
            eprintln!("\n\n============================\n\n");

            // Random pattern (short for testing)
            let pattern_len = rng.random_range(1..=100);
            let pattern = rand_dna_w_seed(pattern_len, &mut rng);

            // Random text (short for testing)
            let text_len = rng.random_range(1..=1000);
            let mut text = rand_dna_w_seed(text_len, &mut rng);

            // generate overlap at the prefix and suffix of the text
            let prefix_overlap = rng.random_range(1..=pattern_len.min(text_len));
            let suffix_overlap = rng.random_range(1..=pattern_len.min(text_len));

            // Ensure there's at least one character spacing between prefix and suffix
            if prefix_overlap + suffix_overlap >= text_len {
                skipped += 1;
                continue;
            }

            text.splice(
                0..prefix_overlap,
                pattern[pattern_len - prefix_overlap..].iter().copied(),
            );

            let expected_prefix_cost =
                ((pattern.len() as f32 - prefix_overlap as f32) * 0.5).floor();
            let expected_prefix_end_pos = prefix_overlap;

            // suffix overlap means we insert "suffix_overlap" start of the pattern at the end of the text
            text.splice(
                text_len - suffix_overlap..text_len,
                pattern[..suffix_overlap].iter().copied(),
            );
            let expected_suffix_cost =
                ((pattern.len() as f32 - suffix_overlap as f32) * 0.5).floor();
            let expected_suffix_end_pos = text_len;

            eprintln!("Q: {}", String::from_utf8_lossy(&pattern));
            eprintln!("T: {}", String::from_utf8_lossy(&text));
            eprintln!("pattern len {pattern_len}");
            eprintln!("Text len {text_len}");
            eprintln!("[prefix] overlap {prefix_overlap}");
            eprintln!("[suffix] overlap {suffix_overlap}");
            eprintln!("[prefix] expected_cost {expected_prefix_cost}");
            eprintln!("[prefix] expected_end_pos {expected_prefix_end_pos}");
            eprintln!("[suffix] expected_cost {expected_suffix_cost}");
            eprintln!("[suffix] expected_end_pos {expected_suffix_end_pos}");
            eprintln!("--------------------------------");

            // Allow all k for now but later should be k
            let matches = searcher.search_all(&pattern, &text, pattern_len);
            // Check if matches are found with expected cost at expected positions
            let mut found = [false, false];
            let expected_locs = [expected_prefix_end_pos, expected_suffix_end_pos];
            let expected_costs = [expected_prefix_cost, expected_suffix_cost];
            for m in matches {
                println!(
                    "m: {}-{} {}-{} {}",
                    m.text_start, m.text_end, m.pattern_start, m.pattern_end, m.cost
                );
                for i in 0..expected_locs.len() {
                    if m.text_end == expected_locs[i] && m.cost == expected_costs[i] as Cost {
                        found[i] = true;
                    }
                }
            }
            assert!(found[0], "Expected prefix overlap not found");
            assert!(found[1], "Expected suffix overlap not found");
        }
        eprintln!("Passed: {} (skipped: {})", iter - skipped, skipped);
    }

    #[test]
    fn test_pattern_trace_path_0_edits() {
        /*
           Q:     ATGC
           T: GGGGATGCGGG
              0123456789*
        */
        let pattern = b"ATGC";
        let text = b"GGGGATGCGGG";
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, text, 0);
        let path = matches[0].to_path();
        assert_eq!(path, vec![Pos(0, 4), Pos(1, 5), Pos(2, 6), Pos(3, 7)]);
        // Ends are exclusive
        assert_eq!(matches[0].pattern_end, path.last().unwrap().0 as usize + 1);
        assert_eq!(matches[0].text_end, path.last().unwrap().1 as usize + 1);
    }

    #[test]
    fn test_pattern_trace_path_0_edits_rc() {
        let pattern = b"TCCGGAT".to_vec(); // ATGCCGA
        let text = b"GGGGGGGGATGCGGAAAA";
        //                              0123456789*1234567
        //                                      ||||-||
        //                                      ATGCGGA
        let mut searcher = Searcher::<Dna>::new_rc();
        let matches = searcher.search(&pattern, &text, 1);
        let path: Vec<Pos> = matches[0].to_path();
        for i in 0..4 {
            let Pos(q_pos, r_pos) = path[i];
            assert_eq!(
                pattern[q_pos as usize] as char,
                Dna::reverse_complement(&text[r_pos as usize..r_pos as usize + 1])[0] as char
            );
        }
    }

    #[test]
    fn test_pattern_trace_path_1_edits() {
        let pattern = b"ATGC";
        let text = b"GGGGATTGCGGG";
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, text, 1);
        let path = matches[0].to_path();
        assert_eq!(
            path,
            vec![Pos(0, 4), Pos(1, 5), Pos(1, 6), Pos(2, 7), Pos(3, 8)]
        );
        // Ends are exclusive
        assert_eq!(matches[0].pattern_end, path.last().unwrap().0 as usize + 1);
        assert_eq!(matches[0].text_end, path.last().unwrap().1 as usize + 1);
    }

    #[test]
    fn test_pattern_trace_path_with_overhang_prefix() {
        let pattern = b"ATCGATCG";
        let text = b"ATCGGGGGGGGGG"; // half of pattern removed at start
        let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(0.5);
        searcher.alpha = Some(0.5);
        let matches = searcher.search(pattern, text, 2);
        let path = matches[0].to_path();
        // This "skips" the first 4 character of the pattern as they are in the overhang
        assert_eq!(path, vec![Pos(4, 0), Pos(5, 1), Pos(6, 2), Pos(7, 3)]);
        // Ends are exclusive
        assert_eq!(matches[0].pattern_end, path.last().unwrap().0 as usize + 1);
        assert_eq!(matches[0].text_end, path.last().unwrap().1 as usize + 1);
    }

    #[test]
    fn test_pattern_trace_path_with_overhang_suffix() {
        let pattern = b"ATCGATCG";
        let text = b"GGGGGGGATCG"; // half of pattern removed at end
        let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(0.5);
        searcher.alpha = Some(0.5);
        let matches = searcher.search(pattern, text, 2);
        println!("matches: {:?}", matches);
        let path = matches[0].to_path();
        assert_eq!(path, vec![Pos(0, 7), Pos(1, 8), Pos(2, 9), Pos(3, 10)]);
        // Ends are exclusive
        assert_eq!(matches[0].pattern_end, path.last().unwrap().0 as usize + 1);
        assert_eq!(matches[0].text_end, path.last().unwrap().1 as usize + 1);
    }

    #[test]
    fn test_random_patterns_60_range_fuzz() {
        use rand::Rng;
        let mut rng = rand::rng();
        let mut i = 0;

        for _ in 0..10000 {
            let mut searcher = Searcher::<Iupac>::new_rc_with_overhang(0.4);

            // Generate random pattern of length 126
            let pattern: Vec<u8> = (0..126).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();

            // Generate random text length between 62-90
            let text_len = rng.random_range(62..91);
            let text: Vec<u8> = (0..text_len)
                .map(|_| b"ACGT"[rng.random_range(0..4)])
                .collect();

            // Use k as half of pattern length
            let k = pattern.len() / 2;

            let matches = searcher.search(&pattern, &text, k);

            // Print every 1000 iterations
            i += 1;
            println!(
                "Iteration {}: Q={}, T={}, k={}, matches={}\npattern: {}\nText: {}",
                i,
                pattern.len(),
                text.len(),
                k,
                matches.len(),
                String::from_utf8_lossy(&pattern),
                String::from_utf8_lossy(&text)
            );

            // Verify matches
            for m in matches {
                assert!(
                    m.cost <= k as Cost,
                    "Match has cost {} > {}: {m:?}\npattern: {}\nText: {}\n",
                    m.cost,
                    k,
                    String::from_utf8_lossy(&pattern),
                    String::from_utf8_lossy(&text)
                );
            }
        }
    }

    #[test]
    fn test_case3() {
        let pattern = b"GTCTTTCATTCTCTCATCATAATCTCTAATACGACACATTGTACATCTGCTTGCGAGCCGGTGTAGCGCCGTCCTGTTATTTCAAGGCTATAATTACGAATTCAATTCCTCCTCTTCCAAAACACG";
        let text = b"AGTGATATCTCAAGGGGCCCTATTGGAAGGAAAGCCGCGATGGGTTCAACGTCAAGTGGATCATTCGATATTCATTAGCCCAACAGAAAC";
        let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(0.4);
        let matches = searcher.search(pattern, &text, 63);
        println!("Matches: {:?}", matches);
    }

    #[test]
    fn test_case4() {
        let expected_end_pos = 1;
        let expected_cost = 1;
        let pattern = b"ATC";
        let text = b"CGGGGGG";
        let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(0.5);
        let matches = searcher.search(pattern, &text, pattern.len());
        let all_matches = searcher.search_all(pattern, &text, pattern.len());

        println!("[ALL MATCHES]");
        let mut all_found = false;
        for m in all_matches {
            println!("\t{}-{} c: {}", m.text_start, m.text_end, m.cost);
            if m.text_end == expected_end_pos && m.cost == expected_cost {
                all_found = true;
                break;
            }
        }

        println!("[LOCAL MATCHES]");
        let mut local_found = false;
        for m in matches {
            println!("\t{}-{} c: {}", m.text_start, m.text_end, m.cost);
            if m.text_end == expected_end_pos && m.cost == expected_cost {
                local_found = true;
                break;
            }
        }

        assert!(
            all_found,
            "No ALL match found ending at {expected_end_pos} with cost {expected_cost}"
        );
        assert!(
            local_found,
            "No LOCAL match found ending at {expected_end_pos} with cost {expected_cost}"
        );
    }

    #[test]
    fn test_match_exact_at_end() {
        let pattern = b"ATAC".to_vec();
        let text = b"CCCCCCATAC";
        let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(0.5);
        let matches = searcher.search(&pattern, &text, 0);
        println!("Matches: {:?}", matches);
        let all_matches = searcher.search_all(&pattern, &text, 0);
        println!("All matches: {:?}", all_matches)
    }

    #[test]
    fn fwd_rc_test_simple() {
        let pattern = b"ATCATGCTAGC".to_vec();
        let text = b"GGGGGGGGGGATCATGCTAGCGGGGGGGGGGG".to_vec();
        let rc = Iupac::reverse_complement(&pattern);

        let mut searcher = Searcher::<Iupac>::new_rc_with_overhang(0.5);
        let fwd_matches = searcher.search(&pattern, &text, 0);
        let rc_matches = searcher.search(&rc, &text, 0);

        assert_eq!(
            fwd_matches.len(),
            rc_matches.len(),
            "Simple test: Forward and RC searches should find the same number of matches"
        );

        for fwd_match in fwd_matches.iter() {
            let matching_rc = rc_matches.iter().find(|rc_match| {
                rc_match.text_start == fwd_match.text_start
                    && rc_match.text_end == fwd_match.text_end
                    && rc_match.cost == fwd_match.cost
            });
            assert!(
                matching_rc.is_some(),
                "No matching RC match found for forward match: {:?}",
                fwd_match.without_cigar()
            );
        }
    }

    #[test]
    fn fwd_rc_test() {
        let fwd = b"TGAAGCGGCGCACGAAAAACGCGAAAGCGTTTCACGATAAATGCGAAAAC";
        let rc = Iupac::reverse_complement(fwd);

        let text = b"TGTTATATTTCCCTGTACTTCGTTCCAGTTATTTTTATGCAAAAAACCGGTGTTTAACCACCACTGCCATGTATCAAAGTACGGTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCAACAGGAAAACTATTTAGCTACGATCAGAGCATCTATCGACTCTATCGACT".to_vec();
        //                                                                                 ^                                                 ^ c=20
        //                                                                                                       ^                                                ^ c = 0

        println!("TEXT LEN: {}", text.len());
        println!("FWD LEN: {}", fwd.len());

        let mut searcher = Searcher::<Iupac>::new_rc(); // (0.5);
        let fwd_matches = searcher.search(fwd, &text, 20);
        let rc_matches = searcher.search(&rc, &text, 20);

        // Print matches for debugging
        println!("Forward matches:");
        for m in fwd_matches.iter() {
            println!("  {:?}", m.without_cigar());
            let matching_slice = String::from_utf8_lossy(&text[m.text_start..m.text_end]);
            println!("\tM slice: {}", matching_slice);
        }
        println!("\nReverse complement matches:");
        for m in rc_matches.iter() {
            println!("  {:?}", m.without_cigar());
            let matching_slice = String::from_utf8_lossy(&text[m.text_start..m.text_end]);
            println!("\tM slice: {}", matching_slice);
        }

        assert_eq!(
            fwd_matches.len(),
            rc_matches.len(),
            "Forward and reverse complement searches should find the same number of matches"
        );

        // For each fwd, there should be matching rc (just strand difference)
        for fwd_match in fwd_matches.iter() {
            let matching_rc = rc_matches.iter().find(|rc_match| {
                rc_match.text_start == fwd_match.text_start
                    && rc_match.text_end == fwd_match.text_end
                    && rc_match.cost == fwd_match.cost
            });
            assert!(
                matching_rc.is_some(),
                "No matching RC match found for forward match: {:?}",
                fwd_match.without_cigar()
            );
        }
    }

    #[test]
    #[ignore = "expected fail; planed match is part of another extending local minima"]
    fn search_bug() {
        /*
        edits 2
        pattern q=11 AGCTAGCTCTC
        pattern    GCTAGCTGCTC
        text len 8843
        planted idx 151
        expected idx 152
        AGCTAGCT-CTC
        -|||||||-|||
        -GCTAGCTGCTC
        */

        let pattern = b"AGCTAGCTCTC";
        let text = b"TATCCGGAAAAGAGCTTTAACAGTAAGTGCTTGTAGTACTATACGAATCTAATGGTGCGTCTTGTCCAATATGTTATATGCAGGTACTTAGTCTTCCCAATGTGTCTTAAAGTCTAGGCACATCTTTCTACTACAGCGAATGAACCGCGAATGCTAGCTGCTCTTAACGCCTTAAAGGATCTACTATATTTGGGGTTTGCTTAGACCGCCTTGCCGAGCATAATTAGTTCTAAATTCAGCGACCACTATTCCCCCGACAGGGTCAACCCAACTTAGCAAACTGTCATTCTATTTCTTGGAATGCAAGATCGGTACAT";
        //  ==============
        //      ^        ^ match, c = 2
        //    ^         ^ match, c = 2
        //  ^          ^ planted, c = 2 < only right reported

        let edits = 2;
        let expected_idx = 452;
        let mut searcher = Searcher::<Dna>::new_fwd();

        // NOTE: does pass with search_all as all in minima plateau are then reported
        let matches = searcher.search(pattern, &text, edits);
        for fw_m in matches.iter() {
            println!("fw_m: {:?}", fw_m.without_cigar());
            let (text_start, text_end) = (fw_m.text_start, fw_m.text_end);
            println!(
                "Text slice: {}",
                String::from_utf8_lossy(&text[text_start..text_end])
            );
        }

        let m = matches
            .iter()
            .find(|m| m.text_start.abs_diff(expected_idx) <= edits);
        assert!(m.is_some(), "Fwd searcher failed");
    }

    #[test]
    fn search_bug_2() {
        /*
        edits 2
        edits 1
        pattern q=12 TACACAGTCAAG
        pattern TACGACAGTCAAG
        text len 560
        planted idx 435
        expected idx 436
        matches []
        */

        let pattern = b"TACACAGTCAAG";
        let text = b"GAAGTGTCACGACTGTAGGATTGTTCGTTTGTGTGGTCATATTAAGAATATGCGTCCTGGCATTTACTCCGCAATATGATAACCCACTAACGCCTGGCTAAACTAATAAAATTCTTGCGTATGCCAGTGGGTATTGTCCACCTCACTCCTGAGTCTACGCGCGACCAATAACTTAGTTACGAACTTCCGGAACACATATTACCAGAAAAAGCGCACGATGTTACGTATCGTTATGGGCAGCCTCCGTAACCCCGTCTCTAGGGTTTCGCCCTTCGTAGTCCTAACACCCCCTGATTTTTTAATACAGACGGACGCTCTCCAAAGTCCGCTGACTAGTTTCCTAATACTCTCTTTGTCATATAACACCCTCGTTTTCGACAGGCCATCTAGAATTTTATGGATCCTTAGGGTATTCAGGGCGGTCAAATCTAGCCTTACGACAGTCAAGTCACATGTGAATACTCCTTCTTCCACGGACGTCTTTATAAATTCCCCCTATTGCCTCTCACTAGGGGTTTCCATGGGGCTTGATCGCACAATAGGAATGTCTAGGAGGCAAG";

        let edits = 1;
        let expected_idx = 436;
        let mut searcher = Searcher::<Dna>::new_fwd();

        // NOTE: does pass with search_all as all in minima plateau are then reported
        let matches = searcher.search(pattern, &text, edits);
        for fw_m in matches.iter() {
            println!("fw_m: {:?}", fw_m.without_cigar());
            let (text_start, text_end) = (fw_m.text_start, fw_m.text_end);
            println!(
                "Text slice: {}",
                String::from_utf8_lossy(&text[text_start..text_end])
            );
        }

        let m = matches
            .iter()
            .find(|m| m.text_start.abs_diff(expected_idx) <= edits);
        assert!(m.is_some(), "Fwd searcher failed");
    }

    #[test]
    fn search_bug_3() {
        /*
        edits 18
        pattern q=61 CGATCGGAATCTCTTTGTTCATGATCCAAAGCCCAGCCATCAGCCCGAACGGTGGTTCGCG
        pattern TGATCGAATCTTTTTTTTTGTACTCCAAAGCCCTCATCAGCTCCGACAGTGGTTCGCG
        text len 64
        planted idx 6
        expected idx 3
        text ACAGGGTGATCGAATCTTTTTTTTTGTACTCCAAAGCCCTCATCAGCTCCGACAGTGGTTCGCG
        matches []
        */

        let pattern = b"CGATCGGAATCTCTTTGTTCATGATCCAAAGCCCAGCCATCAGCCCGAACGGTGGTTCGCG";
        let text = b"ACAGGGTGATCGAATCTTTTTTTTTGTACTCCAAAGCCCTCATCAGCTCCGACAGTGGTTCGCG";

        let edits = 18;
        let expected_idx = 3;
        let mut searcher = Searcher::<Dna>::new_fwd();

        // NOTE: does pass with search_all as all in minima plateau are then reported
        let matches = searcher.search(pattern, &text, edits);
        for fw_m in matches.iter() {
            println!("fw_m: {:?}", fw_m.without_cigar());
            let (text_start, text_end) = (fw_m.text_start, fw_m.text_end);
            println!(
                "Text slice: {}",
                String::from_utf8_lossy(&text[text_start..text_end])
            );
        }

        let m = matches
            .iter()
            .find(|m| m.text_start.abs_diff(expected_idx) <= edits);
        assert!(m.is_some(), "Fwd searcher failed");
    }

    #[test]
    fn original_rc_bug() {
        let fwd = b"TGAAGCGGCGCACGAAAAACGCGAAAGCGTTTCACGATAAATGCGAAAACNNNNNNNNNNNNNNNNNNNNNNNNGGTTAAACACCCAAGCAGCAATACGTAACTGAACGAAGTACAGGAAAAAAAA";
        let rc: Vec<u8> = Iupac::reverse_complement(fwd);

        let text = b"TGTTATATTTCCCTGTACTTCGTTCCAGTTATTTTTATGCAAAAAACCGGTGTTTAACCACCACTGCCATGTATCAAAGTACGGTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCAACAGGAAAACTATTTTCTGCAG".to_vec();

        println!("Q len: {}", fwd.len());
        println!("T len: {}", text.len());

        println!("FWD");
        let mut searcher = Searcher::<Iupac>::new_rc();
        let matches = searcher.search(fwd, &text, 44);
        for m in matches.iter() {
            println!("fwd: {:?}", m.without_cigar());
        }

        println!("\nRC");
        let matches = searcher.search(&rc, &text, 44);
        for m in matches.iter() {
            println!("rc: {:?}", m.without_cigar());
        }
    }

    #[test]
    #[ignore = "Cigar is invariant under rc text, not rc pattern"]
    fn test_cigar_invariant_under_rc_pattern() {
        let pattern = b"AAAAAAA";
        let text = "GGGGAATAAAAGGG"; // 2 match, 1 sub, 4 match
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, &text, 1);
        let fwd_cigar = matches[0].cigar.to_string();
        // Now enabling rc search, and reverse complementing pattern should yield same cigar
        let mut searcher = Searcher::<Dna>::new_rc();
        let pattern_rc = Iupac::reverse_complement(pattern);
        let matches = searcher.search(&pattern_rc, &text, 1);
        let rc_cigar = matches[0].cigar.to_string();
        println!("FWD: {}", fwd_cigar);
        println!("RC: {}", rc_cigar);
        assert_eq!(fwd_cigar, rc_cigar);
    }

    #[test]
    fn test_cigar_invariant_under_rc_text() {
        let pattern = b"AAAAAAA";
        let text = b"GGGGAATAAAAGGG"; // 2 match, 1 sub, 4 match
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, &text, 1);
        let fwd_cigar = matches[0].cigar.to_string();
        // Now enabling rc search, and reverse complementing pattern should yield same cigar
        let mut searcher = Searcher::<Dna>::new_rc();
        let text_rc = Iupac::reverse_complement(text);
        let matches = searcher.search(pattern, &text_rc, 1);
        let rc_cigar = matches[0].cigar.to_string();
        println!("FWD: {}", fwd_cigar);
        println!("RC: {}", rc_cigar);
        assert_eq!(fwd_cigar, rc_cigar);
    }

    #[test]
    #[ignore = "Cigar is invariant under rc text, not rc pattern and text"]
    fn test_cigar_invariant_under_rc_pat_and_text() {
        let pattern = b"AAAAAAA";
        let text = b"GGGGAATAAAAGGG"; // 2 match, 1 sub, 4 match
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, &text, 1);
        let fwd_cigar = matches[0].cigar.to_string();
        // Now enabling rc search, and reverse complementing pattern should yield same cigar
        let mut searcher = Searcher::<Dna>::new_rc();
        let pattern_rc = Iupac::reverse_complement(pattern);
        let text_rc = Iupac::reverse_complement(text);
        let matches = searcher.search(&pattern_rc, &text_rc, 1);
        let rc_cigar = matches[0].cigar.to_string();
        println!("FWD: {}", fwd_cigar);
        println!("RC: {}", rc_cigar);
        assert_eq!(fwd_cigar, rc_cigar);
    }

    #[test]
    #[ignore = "expected fail; local minima flip, see search all results"]
    fn test_cigar_rc_at_overhang_beging() {
        let pattern = b"TTTTAAAAAA";
        let text: &'static str = "AAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGG"; // 5 matches
        let pattern_rc = Iupac::reverse_complement(pattern);

        println!("[MANUAL Reversing]");
        println!("- RC(q):\t{:?}", String::from_utf8_lossy(&pattern_rc));
        println!(
            "- compl(RC(q)):\t{:?}",
            String::from_utf8_lossy(&Iupac::complement(&pattern_rc))
        );
        let mut reversed_text = text.as_bytes().to_vec();
        reversed_text.reverse();
        println!(
            "- Rev(text):\t{:?}",
            String::from_utf8_lossy(&reversed_text)
        );
        //                    TTTTAAAAAA, 6 matches, 4 * 0.5 = 2 cost overhang

        let mut searcher = Searcher::<Iupac>::new_rc_with_overhang(0.5);

        let fwd_matches = searcher.search(pattern, &text, 2);
        let rc_matches = searcher.search(&pattern_rc, &text, 2);

        let fwd_matches_all = searcher.search_all(pattern, &text, 2);
        let rc_matches_all = searcher.search_all(&pattern_rc, &text, 2);

        for m in fwd_matches_all.iter() {
            println!("fwd: {:?}", m);
        }
        for m in rc_matches_all.iter() {
            println!("rc: {:?}", m);
        }
        let fwd_cigar = fwd_matches[0].cigar.to_string();
        let rc_cigar = rc_matches[0].cigar.to_string(); // Should also be 6 matches (prints above)
        assert_eq!(fwd_matches.len(), 1);
        assert_eq!(fwd_matches.len(), rc_matches.len());
        assert_eq!(fwd_cigar, rc_cigar);
    }

    #[test]
    fn test_cigar_rc_at_overhang_end() {
        let pattern = b"TTTTAAA";
        let pattern_rc = Iupac::reverse_complement(pattern);
        let text = b"GGGGGGGGGTTTTAAA"; // 2 match, 1 sub, 4 match
        let mut searcher = Searcher::<Iupac>::new_rc_with_overhang(0.5);
        // Fwd search
        let matches = searcher.search(pattern, &text, 1);
        let fwd_cigar = matches[0].cigar.to_string();
        println!(
            "start - end: {} - {}",
            matches[0].text_start, matches[0].text_end
        );
        println!("FWD: {}", fwd_cigar);
        // RC search
        let matches = searcher.search(&pattern_rc, &text, 1);
        let rc_cigar = matches[0].cigar.to_string();
        println!(
            "start - end: {} - {}",
            matches[0].text_start, matches[0].text_end
        );
        println!("RC: {}", rc_cigar);
    }

    #[test]
    fn real_data_bug() {
        let pattern = b"TTTTTTTTCCTGTACTTCGTTCAGTTACGTATTGCTGCTTGGGTGTTTAACCNNNNNNNNNNNNNNNNNNNNNNNNGTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA";
        let text = b"TTATGTATACCTTGGCATTGAAGCCGATATTGACAACTAGGCACAGCGAGTCTTGGTTGTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCTGCCGCTTCACTGGCATTGATTGAAAATCTGCAACGCGAAGATTTGACACCAATCGAAGAAGCAGAAGCCTATGAGCGCTTGCTTGCGTTTACAAGACATCACGCAGAAGTGTTAGCTCGTAAGCTCGGACGTAGTCAATCGACGATTGCTAACAAATTGCGTTTGCTTCGATTGCCAACGGATGTCCGGGGAAACGTGAAGCAACGCAAATAACGGAGCGTCATGCCCGTGCGTTATTGCCGCTCAAGGATGAAGCGCTACAAGTAACGGTACTCGCTGAAATTCTGGAACGGGAATGGAACGTCAAGGAGACGGAGCGCCGGGTGGAACGATTGATGACACCACAGCCACCGAAGAAAAAACGTCATAAGAGCTTTGCTCGGGATACACGGATTGCGTTAAATACCCTTCGCGATTCCGTCGATATGATCGAGCAAACCGGATTGACGATTGAAAAAGAAGAAGTCGATTGTGAAGAATATGTAGAGGTGCGGATTCGCATCGTGAAGGCACGTCCGGAATAAGCGGTCGTGCCTTCCGCTACGTTTAGGAGAGAAGGCAAGTGAACGAATTACCTGTTCGGGCAATTAGCCAATCCGACTCAACCCCGGAAACGATTCAATGAAAAAGCATTAGAGACTCGCCCAATCACTCGTTCGGCACGGGATTAGTCGGACCATCGTCGTCCGACCATGTGATGGCTATTATGAAATCATCGCCGGCGAACGACGGTATCAAGCAGCGAGTCGCGCAGGATTCGAACGTGTACCGGTCCTCGTCGTCGAAGACGACGAGACACGCGTGATGGAGCTCGCTTTGATCGAAAACATCCAACGGGCGGATTTATCCGCGATTGGAAGAGGCGATGGCGTATGCGGAGATGATTCGAGGAATTCGGTATCACGCAAGCAGAGCTTGCGCAGCGTGTCGCAGAAAGTCGTTCGCACATCACGAACAGTCTTGGGTTACTACAATTGCCGTTACTCGTTCAACAAGCGGTCATAGATAGCGTCTATCGATGGGACATGCCCGCGCGCTCCTGTCGCTGAAACATCCAAAGAAGATGAACAGATGGCAGAGGGGGCATGGCGGAGAACTGGAACGTCCGTCAGTCAGTTCAGGCCACACTCTGGGAACGTAAGGAAGCCGCCCGTCCGCAACAAGCGACCGCTGTTCAATTCGTCGAGGAATCACTTCGCGAAAAATACGGGGCGACCGTTCGGATTAAACAAGGAAAACAAGCAGGGAAACTCGAGATCGATTTTATAGACGAAGACGACCTCAATCGGTTGCTCGACTTGTTATTACCTGAATCGGATCACTAAAAAGAAGCGATCCGGGCGACGGTCCGCTCTTTTGCTTACATCGAGCGTGGCGTGAAAGAAATCGTCGTCCGTGTGGATTCGGCGGAAGCTCGCATCAAGTAAGTGGAAGCTGTTCGCGACGAGTTCCGCTAGTTCATAAACAAGGTGTAACCGTGTGTTCTGGAGAATCATCATCTCCATGAATCCACTGACGTTGACGACAGCCTTAATGTTGAAGTCACCGACCGCAGGTAACTCTTTTTGGACGCCGGCGCCGGGTGCGAGTGGACCTTCCGAGAAAAAGGCATGTCCGACACTTGAAAGGCGACC";
        let mut searcher = Searcher::<Iupac>::new_rc_with_overhang(0.5);
        let matches = searcher.search(pattern, &text, 45);
        for m in matches.iter() {
            println!("m: {:?}", m.without_cigar());
        }
    }

    #[test]
    fn test_simple_ascii() {
        use crate::profiles::Ascii;
        let pattern = b"hello";
        let text = b"heeloo world";
        let mut searcher = Searcher::<Ascii>::new_fwd();
        let matches = searcher.search(pattern, &text, 1);
        for m in matches.iter() {
            println!("m: {:?}", m.without_cigar());
        }
    }

    #[test]
    fn test_reported_start_end() {
        let pattern = b"AGTCGACTAC";
        let pattern_rc = Iupac::reverse_complement(pattern);

        let mutated_ins = b"AGTGACTTC";
        let mutated_ins_rc = Iupac::reverse_complement(mutated_ins);

        let mut text = b"GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG".to_vec();
        text.splice(20..20, mutated_ins_rc.iter().copied());
        text.splice(50..50, mutated_ins.iter().copied());

        // Fwd search
        println!("Fwd search");
        let mut searcher = Searcher::<Iupac>::new_fwd();
        let matches = searcher.search(pattern, &text, 2);

        for m in matches {
            let start = m.text_start;
            let end = m.text_end;
            let m_text = &text[start..end];
            println!(
                "m_text: {}",
                String::from_utf8_lossy(Iupac::reverse_complement(m_text).as_ref())
            );
        }

        // Rc search
        println!("Rc search");
        let mut searcher = Searcher::<Iupac>::new_rc();
        let matches = searcher.search(&pattern_rc, &text, 2);
        for m in matches {
            let start = m.text_start;
            let end = m.text_end;
            let m_text = &text[start..end];
            println!(
                "m_text: {}",
                String::from_utf8_lossy(Iupac::reverse_complement(m_text).as_ref())
            );
        }
    }

    #[test]
    fn test_searchable_slice() {
        let q = b"ATG";
        let t = b"ATGCTACA";
        let t_ref = t.as_slice();
        let mut searcher = Searcher::<Iupac>::new_rc();
        let matches = searcher.search(q, &t_ref, 0);
        assert_eq!(matches.is_empty(), false);
    }

    #[test]
    #[ignore = "Example of rc(q) vs t != compl(q) vs rev(t)"]
    fn diff_rc_method() {
        let pattern = b"AATGCTCCAATGGATGTCACTGCAAGCTCTT".to_vec();
        let text = b"ATAGAGAGCTTGCAGTGACATCCATTGGAGCATTGCG";
        let pattern_rc = Iupac::reverse_complement(pattern.as_slice());
        println!("pattern: {:?}", String::from_utf8_lossy(&pattern));
        println!(
            "pattern rc: {:?}",
            String::from_utf8_lossy(pattern_rc.as_slice())
        );
        println!("Sub text: {:?}", String::from_utf8_lossy(text));
        let mut searcher = Searcher::<Iupac>::new_rc();
        let matches = searcher.search_all(&pattern, &text, 3);
        for m in matches {
            println!(
                "m start {} end {} cost {}, cigar: {}",
                m.text_start,
                m.text_end,
                m.cost,
                m.cigar.to_string()
            );
        }

        println!("Using RC as pattern fwd only");
        let mut searcher = Searcher::<Iupac>::new_fwd();
        let matches = searcher.search_all(&pattern_rc, &text, 3);
        for m in matches {
            println!(
                "m start {} end {} cost {}, cigar: {}",
                m.text_start,
                m.text_end,
                m.cost,
                m.cigar.to_string()
            );
        }
    }

    #[test]
    fn diff_rc_result() {
        let text = b"ACCAGATTGCTGGTGCTGCTTGTCCAGGGTTTGTGTAACCTTTTAACCTTCGTCGGCAGCGTCAGATGTGTATAAGAGACAGTACCTGGTTGATCCTGCCAGTAGTCATATGCTTGTCTCAAGATTAAGCCATGCATGTCTAAGTATAAACAAATTCATACTGTGAAACTGCGAATGGCTCATTAAATCAGTTATAGTTTATTTGATGGTTTCTTGCTACATGGATAACTGTGGTAATTCTAGAGCTAATACATGCTGAAAAGCCCCGACTTCTGGAAGGGGTGTATTTATTAGATAAAAAACCAATGACTTCGGTCTTCTTGGTGATTCATAATAACTTCTCGAATCGCATGGCCTCGCGCCGGCGATGCTTCATTCAAATATCTGCCCTATCAACTTTCGATGGTAGGATAGAGGCCTACCATGGTATCAACGGGTAACGGGAATTAGGGTTCGATTCCGGAGAGGGAGCCTAGAAACGGCTACCACATCCAAGGAAGGCAGCAGGCGCGCAAATTACCCAATCCCGACACGGGGA";
        let text_rc = Iupac::reverse_complement(text);
        let q = b"AATGTACTTCGTTCAGTTACGTATTGCTGGTGCTGNNNNNNNNNNNNNNNNNNNNNNNNTTAACCTTCGTCGGCAGCGTCAGATGTGTATAAGAGACAGTACCTGGTTGATYCTG";
        let mut searcher = Searcher::<Iupac>::new_rc_with_overhang(0.5);

        println!("Forward matches");
        let matches = searcher.search(q, &text, 12);
        for m in matches.iter() {
            println!("m: {:?}", m.without_cigar());
            let (m_start, m_end) = (m.text_start, m.text_end);
            let m_text = &text[m_start..m_end];
            let path = m.to_path();
            println!("Cigar: {}", m.cigar.to_string());
            for pos in path.iter() {
                let q_pos = pos.0;
                let r_pos = pos.1;
                let q_char = q[q_pos as usize];
                let r_char = text[r_pos as usize];
                println!(
                    "q_pos: {}, r_pos: {}, q_char: {}, r_char: {}",
                    q_pos, r_pos, q_char as char, r_char as char
                );
            }
            println!("m_text: {}", String::from_utf8_lossy(m_text));
        }

        println!("Reverse matches");
        let matches = searcher.search(q, &text_rc, 12);
        for m in matches.iter() {
            let (m_start, m_end) = (m.text_start, m.text_end);
            let m_text = &text_rc[m_start..m_end];
            println!("Match text: {}", String::from_utf8_lossy(m_text));
            let path = m.to_path();
            for pos in path.iter() {
                let q_pos = pos.0;
                let r_pos = pos.1;
                let q_char = q[q_pos as usize];
                let r_char = text_rc[r_pos as usize];
                println!(
                    "q_pos: {}, r_pos: {}, q_char: {}, r_char: {}",
                    q_pos, r_pos, q_char as char, r_char as char
                );
            }
            println!("Cigar: {}", m.cigar.to_string());
            println!("m_text: {}", String::from_utf8_lossy(m_text));
        }
    }

    #[test]
    fn not_rev_invariant() {
        let pattern = b"GCC";
        let text = b"AGCGCTA";
        let mut searcher = Searcher::<Dna>::new_fwd();
        let matches = searcher.search(pattern, &text, 1);
        let pattern_rev = pattern.iter().rev().copied().collect::<Vec<_>>();
        let text_rev = text.iter().rev().copied().collect::<Vec<_>>();
        let matches_rev = searcher.search(&pattern_rev, &text_rev, 1);
        assert!(
            matches.len() != matches_rev.len(),
            "error: fwd matches {} vs rev {}",
            matches.len(),
            matches_rev.len()
        );
    }

    #[test]
    fn search_slice() {
        let text = b"ACCAGATTGC";
        let q = b"AATACAC";
        let mut searcher = Searcher::<Iupac>::new_rc_with_overhang(0.5);
        let _matches = searcher.search(q, text, 1);
        let _matches = searcher.search(q, &text, 1);
        let _matches = searcher.search(q, &&text, 1);
        let _matches = searcher.search(&*q, text, 1);
        let q = q.as_slice();
        let _matches = searcher.search(q, text, 1);
        let _matches = searcher.search(&q, text, 1);
        let _matches = searcher.search(&&q, text, 1);
        let text = text.as_slice();
        let _matches = searcher.search(q, text, 1);
        let _matches = searcher.search(q, &text, 1);
        let _matches = searcher.search(q, &&text, 1);
    }

    #[test]
    fn double_match_search_all() {
        let q = b"CAGTC".to_vec();
        let t = b"CGTGATAAAAAAGCAACGTCAGATAAATCATAGGCTGTAACCAAAACAAAACGGGAGTG".to_vec();
        let k = 3;
        let alpha = 0.5;
        let mut sassy_searcher = Searcher::<Iupac>::new_fwd_with_overhang(alpha);
        let matches = sassy_searcher.search_all(&q, &t, k as usize);
        for m in matches {
            println!("m: {:?}", m.without_cigar());
        }
    }

    #[test]
    fn search_many_fuzz() {
        env_logger::init();

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let mut searcher = Searcher::<Iupac>::new_fwd();

        let iter = 10000;

        for _ in 0..iter {
            searcher.rc = rng.random_bool(0.5);
            searcher.alpha = if rng.random_bool(0.5) {
                None
            } else {
                Some(0.5)
            };
            let num_patterns = rng.random_range(1..=100);
            let num_texts = rng.random_range(1..=100);
            eprintln!("num patterns {num_patterns}");
            eprintln!("num texts {num_texts}");

            let pattern_len = rng.random_range(1..=100);
            eprintln!("pattern len {pattern_len}");

            let patterns = (0..num_patterns)
                .map(|_| {
                    (0..pattern_len)
                        .map(|_| b"ACGT"[random_range(0..4)])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let patterns = patterns.iter().map(|p| p.as_slice()).collect_vec();
            // for pattern in &patterns {
            //     eprintln!("Pattern: {:?}", String::from_utf8_lossy(pattern));
            // }

            let texts = (0..num_texts)
                .map(|_| {
                    let text_len = rng.random_range(2..=1000);
                    (0..text_len)
                        .map(|_| b"ACGT"[random_range(0..4)])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let texts = texts.iter().map(|t| t.as_slice()).collect_vec();
            // for text in &texts {
            //     eprintln!("Text: {:?}", String::from_utf8_lossy(text));
            // }

            eprintln!("alpha {:?}", searcher.alpha);
            let k = rng.random_range(0..=pattern_len * 4 / 10);
            eprintln!("k {k}");
            let num_threads = rng.random_range(1..=4);
            eprintln!("threads {num_threads}");

            let mut matches_single =
                searcher.search_many(&patterns, &texts, k, num_threads, SearchMode::Single);
            let mut matches_batch_texts =
                searcher.search_many(&patterns, &texts, k, num_threads, SearchMode::BatchTexts);
            let mut matches_batch_patterns =
                searcher.search_many(&patterns, &texts, k, num_threads, SearchMode::BatchPatterns);
            matches_single.sort();
            matches_batch_texts.sort();
            matches_batch_patterns.sort();

            eprintln!("single: {}", matches_single.len());
            eprintln!("batch_text: {}", matches_batch_texts.len());
            eprintln!("batch_patterns: {}", matches_batch_patterns.len());

            eprintln!(
                "single dup {}",
                matches_single
                    .iter()
                    .tuple_windows()
                    .filter(|(x, y)| x == y)
                    .inspect(|(m, _)| eprintln!(
                        "Duplicate match in text len {}\n{m:?}",
                        texts[m.text_idx].len()
                    ))
                    .count()
            );
            eprintln!(
                "batch_text dup {}",
                matches_batch_texts
                    .iter()
                    .tuple_windows()
                    .filter(|(x, y)| x == y)
                    .inspect(|(m, _)| eprintln!(
                        "Duplicate match in text len {}\n{m:?}",
                        texts[m.text_idx].len()
                    ))
                    .count()
            );
            eprintln!(
                "batch_patterns dup {}",
                matches_batch_patterns
                    .iter()
                    .tuple_windows()
                    .filter(|(x, y)| x == y)
                    .inspect(|(m, _)| eprintln!(
                        "Duplicate match in text len {}\n{m:?}",
                        texts[m.text_idx].len()
                    ))
                    .count()
            );

            if matches_batch_patterns != matches_single {
                eprintln!("MATCHES BATCH PATTERNS\n{matches_batch_patterns:?}");
                eprintln!("MATCHES SINGLE\n{matches_single:?}");
                for m in &matches_batch_patterns {
                    if !matches_single.contains(m) {
                        eprintln!("pattern len {}", patterns[m.pattern_idx].len());
                        eprintln!("text len {}", texts[m.text_idx].len());
                        panic!("Pattern match\n{m:?}\nwas not found in single search");
                    }
                }
                for m in &matches_single {
                    if !matches_batch_patterns.contains(m) {
                        eprintln!("pattern len {}", patterns[m.pattern_idx].len());
                        eprintln!("text len {}", texts[m.text_idx].len());
                        panic!("Single match\n{m:?}\nwas not found in pattern search");
                    }
                }
            }
            eprintln!();

            assert_eq!(matches_batch_patterns, matches_batch_texts);
            assert_eq!(matches_batch_texts, matches_single);
        }
    }
}
