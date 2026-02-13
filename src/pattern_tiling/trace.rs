use crate::pattern_tiling::backend::SimdBackend;
use crate::pattern_tiling::minima::{TracePostProcess, process_minima};
use crate::pattern_tiling::search::{HitRange, Myers};
use crate::pattern_tiling::tqueries::TQueries;
use crate::profiles::Profile;
use crate::search::Match;
use crate::search::Strand;
use crate::search::get_overhang_steps;
use pa_types::{Cigar, CigarOp, Cost};

pub struct PatternHistory<S: Copy> {
    pub steps: Vec<SimdHistoryStep<S>>,
}

impl<S: Copy> Default for PatternHistory<S> {
    fn default() -> Self {
        Self { steps: Vec::new() }
    }
}

pub struct SimdHistoryStep<S: Copy> {
    pub vp: S,
    pub vn: S,
    pub eq: S,
}

pub struct TraceBuffer {
    pub pattern_indices: Vec<usize>,
    pub approx_slices: Vec<(isize, isize)>,
    pub range_bounds: Vec<(isize, isize)>,
    pub per_range_alignments: Vec<Vec<Match>>,
    pub filtered_alignments: Vec<Match>,
    pub filled_till: usize,
}

impl TraceBuffer {
    pub fn new(lanes: usize) -> Self {
        Self {
            pattern_indices: vec![0; lanes],
            approx_slices: vec![(0isize, 0isize); lanes],
            range_bounds: vec![(0isize, 0isize); lanes],
            per_range_alignments: vec![Vec::new(); lanes],
            filtered_alignments: Vec::with_capacity(10),
            filled_till: 0,
        }
    }

    #[inline(always)]
    pub fn clear_alns(&mut self) {
        for aln in self.per_range_alignments.iter_mut() {
            aln.clear();
        }
        self.filtered_alignments.clear();
        self.filled_till = 0;
    }

    #[inline(always)]
    pub fn populate(&mut self, ranges: &[HitRange], left_buffer: usize) {
        if self.pattern_indices.len() < ranges.len() {
            self.pattern_indices.resize(ranges.len(), 0);
            self.approx_slices.resize(ranges.len(), (0isize, 0isize));
            self.range_bounds.resize(ranges.len(), (0isize, 0isize));
            self.per_range_alignments.resize(ranges.len(), Vec::new());
        }
        for (i, r) in ranges.iter().enumerate() {
            self.pattern_indices[i] = r.pattern_idx;
            // Text slice where the match for sure is present (prevent out of bounds)
            self.approx_slices[i] = (r.start.saturating_sub(left_buffer as isize).max(0), r.end);
            // The original range bounds (limited to 0 in case prefix matches, i.e. -1 only really if entire pattern is before text)
            self.range_bounds[i] = (r.start.max(0), r.end);
            // The alignments we find per range
            self.per_range_alignments[i].clear();
        }
        self.filled_till = ranges.len();
    }
}

#[inline(always)]
fn handle_prefix_overhangs<B: SimdBackend, P: Profile>(
    ranges: &[HitRange],
    t_queries: &TQueries<B, P>,
    searcher: &Myers<B, P>,
    buffer: &mut TraceBuffer,
) {
    // Process only ranges with prefix overhang (start == -1)
    ranges
        .iter()
        .enumerate()
        .filter(|(_, r)| r.start == -1)
        .for_each(|(i, r)| {
            let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.pattern_length));
            let masked_alpha = searcher.alpha_pattern & length_mask;
            let cost = masked_alpha.count_ones();

            let mut cigar = Cigar::default();
            if searcher.alpha == 1.0 {
                for _ in 0..t_queries.pattern_length {
                    cigar.push(CigarOp::Del);
                }
            }

            let strand = if r.pattern_idx >= t_queries.n_original_queries {
                crate::search::Strand::Rc
            } else {
                crate::search::Strand::Fwd
            };

            buffer.per_range_alignments[i].push(Match {
                pattern_idx: r.pattern_idx % t_queries.n_original_queries,
                text_idx: 0,
                text_start: usize::MAX,
                text_end: usize::MAX,
                pattern_start: t_queries.pattern_length,
                pattern_end: t_queries.pattern_length,
                cost: cost as Cost,
                strand,
                cigar,
            });
        });
}

#[inline(always)]
fn handle_suffix_overhangs<B: SimdBackend, P: Profile>(
    searcher: &mut Myers<B, P>,
    last_bit_shift: u32,
    last_bit_mask: B::Simd,
    batch_size: usize,
    overhang_steps: usize,
) {
    let all_ones = B::splat_all_ones();
    let blocks_ptr = searcher.blocks.as_mut_ptr();

    for _i in 0..overhang_steps {
        unsafe {
            let block = &mut *blocks_ptr;
            let (vp_out, vn_out, _cost_out) = Myers::<B, P>::myers_step(
                block.vp,
                block.vn,
                block.cost,
                all_ones, // eq = 1111..
                all_ones,
                last_bit_shift,
                last_bit_mask,
            );
            let eq_arr = B::to_array(all_ones);
            for lane in 0..batch_size {
                let eq_scalar = eq_arr.as_ref()[lane];
                searcher.history[lane].steps.push(SimdHistoryStep {
                    vp: vp_out,
                    vn: vn_out,
                    eq: B::splat_scalar(eq_scalar),
                });
            }
            block.vp = vp_out;
            block.vn = vn_out;
        }
    }
}

/// Trace alignments for hit ranges using a single SIMD forward pass per range.
pub fn trace_batch_ranges<B: SimdBackend, P: Profile>(
    searcher: &mut Myers<B, P>,
    t_queries: &TQueries<B, P>,
    text: &[u8],
    ranges: &[HitRange],
    k: u32,
    post: TracePostProcess,
    alpha: Option<f32>,
    max_overhang: Option<usize>,
    buffer: &mut TraceBuffer,
) {
    assert!(ranges.len() <= B::LANES, "Batch size must be <= LANES");

    if ranges.is_empty() {
        return;
    }

    // How far (at most) we have to move to the left to capture the full alignment
    let left_buffer = t_queries.pattern_length + k as usize;
    buffer.clear_alns();
    buffer.populate(ranges, left_buffer);
    let batch_size = buffer.filled_till;

    // Handle prefix overhangs (basically just r_start == -1 only which is full prefix overhang)
    // this is prevented in search with the get_max_overhang as entire overhang is
    // quite meaningless anyway, but kept here for now
    if searcher.alpha_pattern != !0 {
        handle_prefix_overhangs(ranges, t_queries, searcher, buffer);
    }

    // We only have a single block with B::LANES items
    searcher.ensure_capacity(1, buffer.filled_till);

    // Prep allocs for single block search
    let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.pattern_length));
    searcher.search_prep(
        1,
        t_queries.n_queries,
        t_queries.pattern_length,
        searcher.alpha_pattern & length_mask,
    );

    // Prep history for single block search
    // fixme: lets move from searcher to traceBuffer
    for i in 0..buffer.filled_till {
        searcher.history[i].steps.clear();
        searcher.history[i].steps.reserve(left_buffer);
    }

    let last_bit_shift = (t_queries.pattern_length - 1) as u32;
    let last_bit_mask = B::splat_one() << last_bit_shift;
    let all_ones = B::splat_all_ones();
    let zero_scalar = B::scalar_from_i64(0);
    let one_mask = <B as SimdBackend>::mask_word_to_scalar(!0u64);

    let blocks_ptr = searcher.blocks.as_mut_ptr();

    // Calculate max_len based on range ends
    let mut max_len = 0;
    for slice in buffer.approx_slices.iter().take(batch_size) {
        let len = (slice.1 - slice.0 + 1) as usize;
        if len > max_len {
            max_len = len;
        }
    }

    let overhang_steps = get_overhang_steps(
        t_queries.pattern_length,
        k as usize,
        alpha.unwrap_or(1.0),
        max_overhang,
    );

    for i in 0..max_len {
        unsafe {
            let block = &mut *blocks_ptr;
            let mut eq_arr = B::LaneArray::default();
            let mut keep_mask_arr = B::LaneArray::default();

            let eq_slice = eq_arr.as_mut();
            let keep_slice = keep_mask_arr.as_mut();

            for lane in 0..batch_size {
                let q_idx = buffer.pattern_indices[lane];
                let start = buffer.approx_slices[lane].0;
                let abs_pos = (i as isize) + start;
                if abs_pos >= 0 && (abs_pos as usize) < text.len() {
                    let cur_char = text[abs_pos as usize];
                    let enc = P::encode_char(cur_char) as usize;
                    eq_slice[lane] = B::mask_word_to_scalar(t_queries.peq_masks[enc][q_idx]);
                    keep_slice[lane] = one_mask;
                } else {
                    eq_slice[lane] = zero_scalar;
                    keep_slice[lane] = zero_scalar;
                }
            }

            let eq = B::from_array(eq_arr);
            let keep_mask = B::from_array(keep_mask_arr);
            let freeze_mask = all_ones ^ keep_mask;

            let (vp_new, vn_new, cost_new) = Myers::<B, P>::myers_step(
                block.vp,
                block.vn,
                block.cost,
                eq,
                all_ones,
                last_bit_shift,
                last_bit_mask,
            );

            let vp_masked = (vp_new & keep_mask) | (block.vp & freeze_mask);
            let vn_masked = (vn_new & keep_mask) | (block.vn & freeze_mask);
            let cost_masked = (cost_new & keep_mask) | (block.cost & freeze_mask);
            let freeze_arr = B::to_array(freeze_mask);

            let eq_arr = B::to_array(eq);
            for lane in 0..batch_size {
                let is_frozen = B::scalar_to_u64(freeze_arr.as_ref()[lane]) != 0;
                if !is_frozen {
                    let eq_scalar = eq_arr.as_ref()[lane];
                    searcher.history[lane].steps.push(SimdHistoryStep {
                        vp: vp_masked,
                        vn: vn_masked,
                        eq: B::splat_scalar(eq_scalar),
                    });
                }
            }

            block.vp = vp_masked;
            block.vn = vn_masked;
            block.cost = cost_masked;
        }
    }

    if alpha.is_some() {
        handle_suffix_overhangs(
            searcher,
            last_bit_shift,
            last_bit_mask,
            batch_size,
            overhang_steps,
        );
    }

    for lane in 0..batch_size {
        let (r_start, r_end) = buffer.range_bounds[lane];
        let approx_start = buffer.approx_slices[lane].0;
        for pos in r_start..=r_end {
            let step_idx = pos - approx_start;

            // cost_at_endpoint (matrix + overhang adjustment)
            let cost_at_endpoint = get_cost_at::<B, P>(
                searcher,
                lane,
                step_idx,
                t_queries.pattern_length as isize - 1,
                approx_start,
                text.len(),
            );

            if cost_at_endpoint <= k as isize {
                let aln = traceback_single(
                    searcher,
                    lane,
                    buffer.pattern_indices[lane],
                    t_queries,
                    (approx_start, pos),
                    text.len(),
                );
                buffer.per_range_alignments[lane].push(aln);
            }
        }
    }

    for lane in 0..batch_size {
        process_minima(
            post,
            text.len(),
            &mut buffer.per_range_alignments[lane],
            &mut buffer.filtered_alignments,
        );
    }
}

#[inline(always)]
pub fn clamp_range_to_text_bounds(start: usize, end: usize, upper_bound: usize) -> (usize, usize) {
    // If both outside the text we bound to 0,0 in line with sassy reporting
    if start == usize::MAX && end == usize::MAX {
        return (0, 0);
    }
    if start > upper_bound && end > upper_bound {
        (upper_bound + 1, upper_bound + 1)
    } else {
        (start.max(0), end.min(upper_bound) + 1) // +1 to report inclusive ends like sassy
    }
}

#[inline(always)]
fn extract_simd_lane<B: SimdBackend>(simd_val: B::Simd, lane: usize) -> u64 {
    let arr = B::to_array(simd_val);
    B::scalar_to_u64(arr.as_ref()[lane])
}

fn get_cost_at<B: SimdBackend, P: Profile>(
    searcher: &Myers<B, P>,
    lane_idx: usize,
    step_idx: isize,
    pattern_pos_idx: isize,
    approx_start: isize,
    text_len: usize,
) -> isize {
    let mut cost = if step_idx < 0 {
        // Handle Prefix Overhang
        let mask = if pattern_pos_idx >= 63 {
            !0u64
        } else {
            (1u64 << (pattern_pos_idx + 1)) - 1
        };
        (searcher.alpha_pattern & mask).count_ones() as isize
    } else {
        // Handle Matrix Cost
        let step_data = &searcher.history[lane_idx].steps[step_idx as usize];
        let vp_bits = extract_simd_lane::<B>(step_data.vp, lane_idx);
        let vn_bits = extract_simd_lane::<B>(step_data.vn, lane_idx);

        let mask = if pattern_pos_idx >= 63 {
            !0u64
        } else {
            (1u64 << (pattern_pos_idx + 1)) - 1
        };

        let pos = (vp_bits & mask).count_ones() as isize;
        let neg = (vn_bits & mask).count_ones() as isize;
        pos - neg
    };

    // Apply Suffix Overhang Correction
    let abs_pos = approx_start + step_idx;
    let max_valid_text_pos = text_len.saturating_sub(1) as isize;

    if abs_pos > max_valid_text_pos && searcher.alpha != 1.0 {
        let overshoot = (abs_pos - max_valid_text_pos) as usize;
        let correction = (overshoot as f32 * searcher.alpha).floor() as isize;
        cost += correction;
    }

    cost
}

fn traceback_single<B: SimdBackend, P: Profile>(
    searcher: &Myers<B, P>,
    lane_idx: usize,
    original_pattern_idx: usize,
    t_queries: &TQueries<B, P>,
    slice: (isize, isize),
    text_len: usize,
) -> Match {
    let history = &searcher.history[lane_idx];
    let steps = &history.steps;
    let pattern_len = t_queries.pattern_length as isize;

    let max_step = slice.1 - slice.0;
    let end_pos = slice.1 as usize;
    let mut curr_step = max_step;
    let mut pattern_pos = pattern_len - 1;
    let mut cigar: Cigar = Cigar::default();

    let max_valid_text_pos = text_len - 1;
    let pattern_end = if end_pos >= max_valid_text_pos && searcher.alpha != 1.0 {
        let overshoot = end_pos - max_valid_text_pos;
        pattern_len as usize - overshoot
    } else {
        pattern_len as usize
    };

    let mut pattern_start = pattern_len as usize; // Will be updated as we go back

    // Helper closure to keep the calls concise
    let get_cost = |s: isize, p: isize| get_cost_at(searcher, lane_idx, s, p, slice.0, text_len);

    while curr_step >= 0 && pattern_pos >= 0 {
        let curr_cost = get_cost(curr_step, pattern_pos);
        let diag_cost = get_cost(curr_step - 1, pattern_pos - 1);
        let up_cost = get_cost(curr_step, pattern_pos - 1);
        let left_cost = get_cost(curr_step - 1, pattern_pos);

        let step = &steps[curr_step as usize];
        let eq_bits = extract_simd_lane::<B>(step.eq, lane_idx);
        let is_match = (eq_bits & (1u64 << pattern_pos)) != 0;
        let match_cost = if is_match { 0 } else { 1 };

        if slice.0 + curr_step > (text_len as isize - 1) {
            pattern_pos -= 1;
            curr_step -= 1;
            pattern_start = (pattern_pos + 1).max(0) as usize;
        } else if curr_cost == diag_cost + match_cost && is_match {
            cigar.push(CigarOp::Match);
            curr_step -= 1;
            pattern_pos -= 1;
            pattern_start = (pattern_pos + 1).max(0) as usize;
        } else if curr_cost == left_cost + 1 {
            cigar.push(CigarOp::Ins);
            curr_step -= 1;
        } else if curr_cost == diag_cost + match_cost && !is_match {
            cigar.push(CigarOp::Sub);
            curr_step -= 1;
            pattern_pos -= 1;
            pattern_start = (pattern_pos + 1).max(0) as usize;
        } else if curr_cost == up_cost + 1 {
            cigar.push(CigarOp::Del);
            pattern_pos -= 1;
            pattern_start = (pattern_pos + 1).max(0) as usize;
        } else {
            panic!("Invalid traceback step reached :(");
        }
    }

    // Handle prefix overhang, but we only add it as operations if overhang was disabled
    if pattern_pos >= 0 && searcher.alpha == 1.0 {
        let overhang_len = (pattern_pos + 1) as usize;
        for _ in 0..overhang_len {
            cigar.push(CigarOp::Del);
        }
        pattern_start = 0;
    }

    let slice_start_offset = (curr_step + 1).max(0);
    let text_start = (slice.0 + slice_start_offset).max(0);

    let final_cost = get_cost_at(
        searcher,
        lane_idx,
        max_step,
        pattern_len - 1,
        slice.0,
        text_len,
    );

    cigar.reverse();

    Match {
        pattern_idx: original_pattern_idx % t_queries.n_original_queries,
        text_idx: 0,
        text_start: text_start as usize,
        text_end: slice.1 as usize,
        pattern_start,
        pattern_end,
        cost: final_cost as Cost,
        strand: if original_pattern_idx >= t_queries.n_original_queries {
            Strand::Rc
        } else {
            Strand::Fwd
        },
        cigar,
    }
}
