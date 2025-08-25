use crate::delta_encoding::H;
use crate::minima::prefix_min;
use crate::profiles::Profile;
use crate::trace::{CostMatrix, fill, get_trace, simd_fill};
use crate::{LANES, S};
use crate::{bitpacking::compute_block_simd, delta_encoding::V};
use pa_types::{Cigar, CigarOp, Cost, Pos};
use std::simd::cmp::SimdPartialOrd;

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
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct Match {
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
    pub cigar: Cigar,
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
    #[cfg(test)]
    pub(crate) fn without_cigar(&self) -> Match {
        Match {
            text_start: self.text_start,
            text_end: self.text_end,
            pattern_start: self.pattern_start,
            pattern_end: self.pattern_end,
            cost: self.cost,
            strand: self.strand,
            cigar: Cigar::default(),
        }
    }
}

/// Strand of a match. If Rc, pattern matches the reverse complement text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
impl<T> RcSearchAble for T
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

    fn update_and_encode(&mut self, text: &[u8], i: usize, profiler: &P) {
        let start = self.chunk_offset * 64 + 64 * i;
        self.lane_end = start + 64;
        if start + 64 <= text.len() {
            self.text_slice.copy_from_slice(&text[start..start + 64]);
        } else {
            // Pad with N, so that costs at the end are diagonally preserved.
            self.text_slice.fill(b'N');
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
pub struct Searcher<P: Profile> {
    // Config
    /// Search reverse complement text?
    rc: bool,
    /// overhang cost
    /// If set, must satisfy `0.0 <= alpha <= 1.0`
    alpha: Option<f32>,

    // Internal caches
    cost_matrices: [CostMatrix; LANES],
    hp: Vec<S>,
    hm: Vec<S>,
    lanes: [LaneState<P>; LANES],

    _phantom: std::marker::PhantomData<P>,
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

    /// Create a new `Searcher`.
    pub fn new(rc: bool, alpha: Option<f32>) -> Self {
        Self {
            rc,
            _phantom: std::marker::PhantomData,
            cost_matrices: std::array::from_fn(|_| CostMatrix::default()),
            hp: Vec::new(),
            hm: Vec::new(),
            lanes: std::array::from_fn(|_| LaneState::new(P::alloc_out(), 0)),
            alpha,
        }
    }

    /// Returns a match for each *rightmost local minimum* end position with score <=k.
    ///
    /// Searches the forward text, and optionally the reverse complement of the text.
    pub fn search<I: RcSearchAble>(&mut self, pattern: &[u8], input: &I, k: usize) -> Vec<Match> {
        self.search_handle_rc(
            pattern,
            input,
            k,
            false,
            None::<fn(&[u8], &[u8], Strand) -> bool>,
        )
    }

    /// Returns a match for *all* end positions with score <=k.
    ///
    /// Searches the forward text, and optionally the reverse complement of the text.
    pub fn search_all<I: RcSearchAble>(
        &mut self,
        pattern: &[u8],
        input: &I,
        k: usize,
    ) -> Vec<Match> {
        self.search_handle_rc(
            pattern,
            input,
            k,
            true,
            None::<fn(&[u8], &[u8], Strand) -> bool>,
        )
    }

    /// Returns matches for *all* end positions where `end_filter_fn` returns true.
    ///
    /// Used in CRISPR search to filter for only those end positions where the
    /// PAM (last three characters) matches exactly.
    ///
    /// `filter_fn` is passed the pattern, the (rc) text up to the end position, and the strand.
    /// Note that due to the implementation, for rc searches,
    /// both the pattern and text are complemented from what you would expect.
    pub fn search_with_fn<I: RcSearchAble>(
        &mut self,
        pattern: &[u8],
        input: &I,
        k: usize,
        all_minima: bool,
        filter_fn: impl Fn(&[u8], &[u8], Strand) -> bool,
    ) -> Vec<Match> {
        self.search_handle_rc(pattern, input, k, all_minima, Some(filter_fn))
    }

    fn search_handle_rc<I: RcSearchAble>(
        &mut self,
        pattern: &[u8],
        input: &I,
        k: usize,
        all_minima: bool,
        filter_fn: Option<impl Fn(&[u8], &[u8], Strand) -> bool>,
    ) -> Vec<Match> {
        let mut matches = self.search_one_strand(
            pattern,
            input.text().as_ref(),
            k,
            all_minima,
            &filter_fn,
            Strand::Fwd,
        );
        if self.rc {
            let rc_matches = self.search_one_strand(
                &P::complement(pattern),
                input.rev_text().as_ref(),
                k,
                all_minima,
                &filter_fn,
                Strand::Rc,
            );
            matches.extend(rc_matches.into_iter().map(|mut m| {
                m.strand = Strand::Rc;
                // Also adjust start and end positions to original text orientation
                let rc_start = m.text_start;
                let rc_end = m.text_end;
                m.text_start = input.text().as_ref().len() - rc_end;
                m.text_end = input.text().as_ref().len() - rc_start;
                // NOTE: We keep the cigar in the direction of the pattern.
                // Thus, passing text or rc(text) gives the same CIGAR.
                // m.cigar.ops.reverse();
                m
            }));
        }
        matches
    }

    fn search_one_strand(
        &mut self,
        pattern: &[u8],
        text: &[u8],
        k: usize,
        all_minima: bool,
        filter_fn: &Option<impl Fn(&[u8], &[u8], Strand) -> bool>,
        strand: Strand,
    ) -> Vec<Match> {
        self.search_positions_bounded(pattern, text, k as Cost, all_minima);
        // If there is a filter fn, filter end positions based on function before processing matches
        if let Some(filter_fn) = filter_fn {
            self.lanes.iter_mut().for_each(|lane| {
                lane.matches.retain(|(end_pos, _)| {
                    let text_till_end = &text[..*end_pos];
                    filter_fn(pattern, text_till_end, strand)
                });
            });
        }
        self.process_matches(pattern, text, k as Cost)
    }

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
            // Get the current cost state for this lane
            let v = V(vp.as_array()[lane], vm.as_array()[lane]);

            // Calculate the minimum possible cost in this lane
            // This is the best case scenario - if even this minimum is > k,
            // then no matches are possible in this lane
            let min_in_lane =
                prefix_min(v.0, v.1).0 as Cost + dist_to_start_of_lane.as_array()[lane] as Cost;

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

    fn search_positions_bounded(&mut self, pattern: &[u8], text: &[u8], k: Cost, all_minima: bool) {
        let (profiler, pattern_profile) = P::encode_pattern(pattern);

        // Terminology:
        // - chunk: roughly 1/4th of the input text, with small overlaps.
        // - block: 64 bytes of text.
        // - lane: a u64 of a SIMD vec.

        // The pattern will match a pattern of length at most pattern.len() + k.
        // We round that up to a multiple of 64 to find the number of blocks overlap between chunks.
        let max_overlap_blocks = (pattern.len() + k as usize).next_multiple_of(64) / 64;
        let overlap_blocks = 0;

        // Total number of blocks to be processed, including overlaps.
        let text_blocks = if self.alpha.is_some() {
            // When allowing overlaps, for simplicity we 'extend' the text a bit more with N.
            (text.len() + pattern.len()).div_ceil(64)
        } else {
            text.len().div_ceil(64)
        };
        let total_blocks = text_blocks + (LANES - 1) * overlap_blocks;
        let blocks_per_chunk = total_blocks.div_ceil(LANES);
        // Length of each of the four chunks.
        let chunk_offset = blocks_per_chunk.saturating_sub(overlap_blocks);

        // Update chunk offsets
        for lane in 0..LANES {
            self.lanes[lane].chunk_offset = lane * chunk_offset;
        }

        // Clear matches in each lane
        for lane in 0..LANES {
            self.lanes[lane].matches.clear();
            self.lanes[lane].decreasing = true;
        }

        // State tracking for early termination optimization:
        // - prev_max_j: tracks the highest pattern row we've computed so far
        // - prev_end_last_below: tracks the highest row where any lane had cost <= k
        let mut prev_max_j = 0;
        let mut prev_end_last_below = 0;

        self.hp.clear();
        self.hm.clear();
        self.hp.resize(pattern.len(), S::splat(1));
        self.hm.resize(pattern.len(), S::splat(0));

        init_deltas_for_overshoot(&mut self.hp, self.alpha);

        'text_chunk: for i in 0..blocks_per_chunk + max_overlap_blocks {
            let mut vp = S::splat(0);
            let mut vm = S::splat(0);

            // Update text slices and profiles
            for lane in 0..LANES {
                self.lanes[lane].update_and_encode(text, i, &profiler);
            }

            let mut dist_to_start_of_lane = S::splat(0);
            let mut dist_to_end_of_lane = S::splat(0);
            let mut cur_end_last_below = 0;

            // Iterate over pattern chars (rows in the DP matrix)
            for j in 0..pattern.len() {
                dist_to_start_of_lane += self.hp[j];
                dist_to_start_of_lane -= self.hm[j];

                let pattern_char = unsafe { pattern_profile.get_unchecked(j) };
                let eq: std::simd::Simd<u64, 4> = S::from(std::array::from_fn(|lane| {
                    P::eq(pattern_char, &self.lanes[lane].text_profile)
                }));

                compute_block_simd(&mut self.hp[j], &mut self.hm[j], &mut vp, &mut vm, eq);

                // Early termination check: If we've moved past the last row that had any
                // promising matches (cost <= k), we can potentially skip ahead or terminate
                'check: {
                    dist_to_end_of_lane += self.hp[j];
                    dist_to_end_of_lane -= self.hm[j];

                    // Check if any lane has cost <= k at the current row
                    let cmp = dist_to_end_of_lane.simd_le(S::splat(k as u64));
                    let bitmask = cmp.to_bitmask();
                    let end_leq_k = bitmask != 0;

                    // Track the highest row where we found any promising matches
                    cur_end_last_below = if end_leq_k { j } else { cur_end_last_below };

                    // Only do early termination checks if we've moved past the last promising row
                    if j > prev_end_last_below {
                        // Check if any lane has a minimum cost that could lead to matches <= k
                        if let Some(new_end) =
                            self.check_lanes(&vp, &vm, &dist_to_start_of_lane, k, j)
                        {
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
                        //  println!("Skipping {j}");
                        continue 'text_chunk;
                    }
                }
            }

            // Save positions with cost <= k directly after processing each row
            for lane in 0..LANES {
                let v = V::from(vp[lane], vm[lane]);
                let base_pos = self.lanes[lane].chunk_offset * 64 + 64 * i;
                let cost = dist_to_start_of_lane.as_array()[lane] as Cost;

                self.find_minima_with_overhang(
                    v,
                    cost,
                    k,
                    text.len(),
                    pattern.len(),
                    base_pos,
                    lane,
                    all_minima,
                );
            }

            prev_end_last_below = cur_end_last_below.max(Self::CHECK_AT_LEAST_ROWS);
            prev_max_j = pattern.len() - 1;
        }

        // Clean up any remaining rows that weren't reset
        self.reset_rows(0, prev_max_j);

        // Prune matches in overlapping regions.
        if log::log_enabled!(log::Level::Trace) {
            self.lanes[0].matches.iter().for_each(|&(end_pos, cost)| {
                log::trace!("lane 0 KEEP {end_pos} {cost}");
            });
        }
        for lane in 1..LANES {
            let prev_lane_end = self.lanes[lane - 1].lane_end;
            log::trace!("End of lane {}: {prev_lane_end}", lane - 1);
            log::trace!(
                "Last match of lane {}: {:?}",
                lane - 1,
                self.lanes[lane - 1].matches.last()
            );
            self.lanes[lane].matches.retain(|&(end_pos, cost)| {
                if end_pos < prev_lane_end {
                    log::trace!(
                        "lane {lane} drop {end_pos} {cost} because it's before {prev_lane_end}"
                    );
                } else {
                    log::trace!("lane {lane} KEEP {end_pos} {cost}");
                }
                // Keep matches that end after the previous lane's end position
                // Note that `prev_lane_end` itself is handled by the current lane.
                end_pos >= prev_lane_end
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
        let overshoot_cost = (self.alpha.unwrap_or(0.0) * overshoot as f32).floor() as Cost;
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
            text_len + pattern_len
        } else {
            text_len
        };

        let mut cost = cur_cost;
        let mut prev_cost = self.add_overshoot_cost(cur_cost, base_pos, text_len);
        let mut prev_pos = base_pos;

        if base_pos >= max_pos {
            if base_pos == max_pos {
                if self.lanes[lane].decreasing && prev_cost <= k {
                    log::trace!("lane {lane} push {prev_pos} {prev_cost} <last>");
                    self.lanes[lane].matches.push((prev_pos, prev_cost));
                }
            }

            return;
        }

        for bit in 1..=64 {
            cost += ((p >> (bit - 1)) & 1) as Cost;
            cost -= ((m >> (bit - 1)) & 1) as Cost;

            let pos: usize = base_pos + bit;
            if pos > max_pos {
                if !all_minima && self.lanes[lane].decreasing && prev_cost <= k {
                    log::trace!("lane {lane} push {prev_pos} {prev_cost} <last>");
                    self.lanes[lane].matches.push((prev_pos, prev_cost));
                }
                break;
            }

            let total_cost = self.add_overshoot_cost(cost, pos, text_len);

            if all_minima {
                if total_cost <= k {
                    log::trace!("lane {lane} push {prev_pos} {prev_cost}");
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
                    log::trace!("lane {lane} push {prev_pos} {prev_cost}");
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
    }

    fn process_matches(&mut self, pattern: &[u8], text: &[u8], k: Cost) -> Vec<Match> {
        let mut traces = Vec::new();
        let fill_len = pattern.len() + k as usize;

        for m in &mut self.cost_matrices {
            m.alpha = self.alpha;
        }

        // Collect slices to process in batches
        let mut batch = MatchBatch::new();

        for lane in 0..LANES {
            for &(end_pos, cost) in &self.lanes[lane].matches {
                let offset = end_pos.saturating_sub(fill_len);
                let slice = &text[offset..end_pos.min(text.len())];

                batch.add(slice, offset, end_pos, cost);

                if batch.is_full() {
                    traces.extend(batch.process::<P>(
                        pattern,
                        fill_len,
                        &mut self.cost_matrices,
                        self.alpha,
                        k,
                    ));
                    batch.clear();
                }
            }
        }

        if !batch.is_empty() {
            traces.extend(batch.process::<P>(
                pattern,
                fill_len,
                &mut self.cost_matrices,
                self.alpha,
                k,
            ));
        }

        traces
    }
}

struct MatchBatch<'a> {
    slices: [&'a [u8]; LANES],
    offsets: [usize; LANES],
    ends: [usize; LANES],
    expected_costs: [Cost; LANES],
    count: usize,
}

impl<'a> MatchBatch<'a> {
    fn new() -> Self {
        Self {
            slices: [b""; LANES],
            offsets: [0; LANES],
            ends: [0; LANES],
            expected_costs: [0; LANES],
            count: 0,
        }
    }

    fn add(&mut self, slice: &'a [u8], offset: usize, end: usize, cost: Cost) {
        self.slices[self.count] = slice;
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

    fn process<P: Profile>(
        &self,
        pattern: &[u8],
        fill_len: usize,
        cost_matrices: &mut [CostMatrix; LANES],
        alpha: Option<f32>,
        k: Cost,
    ) -> Vec<Match> {
        if self.count > 1 {
            simd_fill::<P>(
                pattern,
                &self.slices[..self.count],
                fill_len,
                cost_matrices,
                alpha,
            );
        } else {
            fill::<P>(
                pattern,
                self.slices[0],
                fill_len,
                &mut cost_matrices[0],
                alpha,
            );
        }

        let mut results = Vec::with_capacity(self.count);

        for i in 0..self.count {
            let m = get_trace::<P>(
                pattern,
                self.offsets[i],
                self.ends[i],
                self.slices[i],
                &cost_matrices[i],
                alpha,
            );

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

            results.push(m);
        }

        results
    }
}

/// Assumes hp and hm are already the right size, hm=0 and hp=1.
/// Then sets hp according to the given alpha, if needed.
pub(crate) fn init_deltas_for_overshoot_scalar(h: &mut [H], alpha: Option<f32>) {
    if let Some(alpha) = alpha {
        for i in 0..h.len() {
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
pub(crate) fn init_deltas_for_overshoot(hp: &mut [S], alpha: Option<f32>) {
    if let Some(alpha) = alpha {
        for i in 0..hp.len() {
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
pub(crate) fn init_deltas_for_overshoot_all_lanes(hp: &mut [S], alpha: Option<f32>) {
    if let Some(alpha) = alpha {
        for i in 0..hp.len() {
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
        let text = b"AAAAAAAAACCCTTT";
        let mut s = Searcher::<Iupac>::new_fwd();
        s.alpha = Some(0.5);
        s.search_positions_bounded(pattern, text, 10, true);
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
        s.search_positions_bounded(prefix.as_bytes(), text.as_bytes(), 2, true);
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
        s.search_positions_bounded(prefix.as_bytes(), text.as_bytes(), 2, true);
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
        s.search_positions_bounded(contained.as_bytes(), text.as_bytes(), 2, true);
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
        let matches = Searcher::<Dna>::new_rc().search(pattern, &text, 2);
        println!("Matches: {:?}", matches);
    }

    #[test]
    fn no_extra_matches() {
        let edits = 6;
        let expected_idx = 277;
        let pattern = b"TAAGCAGAAGGGAGGTATAAAGTCTGTCAGCGGTGCTTAAG";
        let text = b"ACCGTAACCGCTTGGTACCATCCGGCCAGTCGCTCGTTGCGCCCCACTATCGGGATCGACGCGCAGTAATTAAACACCACCCACGCCACGAGGTAGAACGAGAGCGGGGGGCTAGCAAATAATAGTGAGAGTGCGTTCAAAGGGTCTTTCGTAACCTCAGCGGGCGGGTACGGGGGAAATATCGCACCAATTTTGGAGATGCGATTAGCTCAGCGTAACGCGAATTCCCTATAACTTGCCTAGTGTGTGTGAATGGACAATTCGTTTTACAGTTTCAAGGTAGCAGAAGGGCAGGATAAGTCTGTCGCGGTGCTTAAGGCTTTCCATCCATGTTGCCCCCTACATGAATCGGATCGCCAGCCAGAATATCACATGGTTCCAAAAGTTGCAAGCTTCCCCGTACCGCTACTTCACCTCACGCCAGAGGCCTATCGCCGCTCGGCCGTTCCGTTTTGGGGAAGAATCTGCCTGTTCTCGTCACAAGCTTCTTAGTCCTTCCACCATGGTGCTGTTACTCATGCCATCAAATATTCGAGCTCTTGCCTAGGGGGGTTATACCTGTGCGATAGATACACCCCCTATGACCGTAGGTAGAGAGCCTATTTTCAACGTGTCGATCGTTTAATGACACCAACTCCCGGTGTCGAGGTCCCCAAGTTTCGTAGATCTACTGAGCGGGGGAATATTTGACGGTAAGGCATCGCTTGTAGGATCGTATCGCGACGGTAGATACCCATAAGCGTTGCTAACCTGCCAATAACTGTCTCGCGATCCCAATTTAGCACAAGTCGGTGGCCTTGATAAGGCTAACCAGTTTCGCACCGCTTCCGTTCCATTTTACGATCTACCGCTCGGATGGATCCGAAATACCGAGGTAGTAATATCAACACGTACCCAATGTCC";
        let matches = Searcher::<Dna>::new_fwd().search(pattern, &text, edits);
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
            let matches = Searcher::<Dna>::new_fwd().search(&pattern, &text, 5);
            total_matches += matches.len();
        }
        println!("total matches: {total_matches}");
    }

    #[test]
    fn test_fwd_rc_search() {
        let pattern = b"ATCGATCA";
        let rc = Dna::reverse_complement(pattern);
        let text = [b"GGGGGGGG".as_ref(), &rc, b"GGGGGGGG"].concat();
        let matches = Searcher::<Dna>::new_rc().search(pattern, &text, 0);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text_start, 8);
        assert_eq!(matches[0].text_end, 8 + pattern.len());
        // Now disableing rc search should yield no matches
        let matches = Searcher::<Dna>::new_fwd().search(pattern, &text, 0);
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
        let matches =
            Searcher::<Dna>::new_fwd().search_with_fn(pattern, &text, 0, false, end_filter);
        assert_eq!(matches.len(), 1); // First match *ending* at 10 should be discarded
        assert_eq!(matches[0].text_start, 50);

        // Sanity check, run the same without filter
        let matches = Searcher::<Dna>::new_fwd().search(pattern, &text, 0);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].text_start, 10);
        assert_eq!(matches[1].text_start, 50);
    }

    fn rc(text: &[u8]) -> Vec<u8> {
        let mut rc = text.to_vec();
        rc.reverse();
        rc.iter_mut().for_each(|c| {
            *c = match *c {
                b'A' => b'T',
                b'C' => b'G',
                b'G' => b'C',
                b'T' => b'A',
                _ => *c,
            }
        });
        rc
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
        let pattern_rc = rc(pattern_fwd);
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

        let matches =
            Searcher::<Dna>::new_rc().search_with_fn(pattern_fwd, &text, 0, false, end_filter);
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
            }
        }
    }

    #[test]
    // #[ignore = "for plotting only"]
    fn print_matches() {
        let pattern = b"GCCGT";
        let text = b"AGCGCGTA";
        let k = 1;
        let matches = Searcher::<Dna>::new_rc().search_all(pattern, text, k);
        let match_local = Searcher::<Dna>::new_rc().search(pattern, text, k);
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
        let matches_rev = Searcher::<Dna>::new_rc().search_all(&pattern_rev, &text_rev, k);
        let match_rev_local = Searcher::<Dna>::new_rc().search(&pattern_rev, &text_rev, k);
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
        let iter = 1000;

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
                rc(&text[r_pos as usize..r_pos as usize + 1])[0] as char
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
}

/*
q: AAGGTTACACAAACCCTGGACAAG

GAAGGCAGCAGGCGCGCAAATTAC
CTTGTCCAGGGTTTGTGTAACCTT

*/
