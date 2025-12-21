use crate::pattern_tilling::backend::SimdBackend;
use crate::pattern_tilling::search::{HitRange, Myers};
use crate::pattern_tilling::tqueries::TQueries;
use crate::profiles::iupac::get_encoded;
use crate::search::Match;
use crate::search::Strand;
use crate::search::get_overhang_steps;
use pa_types::{Cigar, CigarOp, Cost};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePostProcess {
    All,
    LocalMinima,
}

pub struct QueryHistory<S: Copy> {
    pub steps: Vec<SimdHistoryStep<S>>,
}

impl<S: Copy> Default for QueryHistory<S> {
    fn default() -> Self {
        Self { steps: Vec::new() }
    }
}

pub struct SimdHistoryStep<S: Copy> {
    pub vp: S,
    pub vn: S,
    pub eq: S,
}

/// Trace alignments for hit ranges using a single SIMD forward pass per range.
pub fn trace_batch_ranges<B: SimdBackend>(
    searcher: &mut Myers<B>,
    t_queries: &TQueries<B>,
    text: &[u8],
    ranges: &[HitRange],
    k: u32,
    post: TracePostProcess,
    output: &mut Vec<Match>,
    alpha: Option<f32>,
    max_overhang: Option<usize>,
) {
    assert!(ranges.len() <= B::LANES, "Batch size must be <= LANES");

    // fixme: get rid of alloc here
    let normal_ranges: Vec<HitRange> = ranges
        .iter()
        .map(|r| HitRange {
            start: r.start.max(0),
            ..*r
        })
        .collect();
    if normal_ranges.is_empty() {
        return;
    }

    let left_buffer = t_queries.query_length + k as usize;

    let mut query_indices = vec![0; B::LANES];
    let mut approx_slices = vec![(0isize, 0isize); B::LANES];
    let mut range_bounds = vec![(0isize, 0isize); B::LANES];
    let mut per_range_alignments: Vec<Vec<Match>> = vec![Vec::new(); B::LANES];
    let batch_size = normal_ranges.len();
    for (i, r) in normal_ranges.iter().enumerate() {
        query_indices[i] = r.query_idx;
        let start = r.start.saturating_sub(left_buffer as isize).max(0);
        approx_slices[i] = (start, r.end);
        range_bounds[i] = (r.start, r.end);
        per_range_alignments[i].clear();
    }

    // Handle prefix overhang ranges (-1) directly
    for (i, r) in ranges.iter().enumerate() {
        if r.start == -1 {
            let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
            let masked_alpha = searcher.alpha_pattern & length_mask;
            let cost = masked_alpha.count_ones();

            let mut cigar = Cigar::default();
            if searcher.alpha == 1.0 {
                for _ in 0..t_queries.query_length {
                    cigar.push(CigarOp::Del);
                }
            }

            per_range_alignments[i].push(Match {
                pattern_idx: r.query_idx % t_queries.n_original_queries,
                text_idx: 0,
                text_start: usize::MAX, // previously -1, but does not match sassy struct, so using usize::MAX (+ additional fix in trace then)
                text_end: usize::MAX,
                pattern_start: t_queries.query_length,
                pattern_end: t_queries.query_length,
                cost: cost as Cost,
                strand: if r.query_idx >= t_queries.n_original_queries {
                    crate::search::Strand::Rc
                } else {
                    crate::search::Strand::Fwd
                },
                cigar,
            });
        }
    }

    let num_blocks = 1;
    searcher.ensure_capacity(num_blocks, batch_size);

    let expected_steps = t_queries.query_length + k as usize;
    for i in 0..batch_size {
        searcher.history[i].steps.clear();
        searcher.history[i].steps.reserve(expected_steps);
    }

    let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.query_length));
    searcher.search_prep(
        num_blocks,
        t_queries.n_queries,
        t_queries.query_length,
        searcher.alpha_pattern & length_mask,
    );

    let last_bit_shift = (t_queries.query_length - 1) as u32;
    let last_bit_mask = B::splat_one() << last_bit_shift;
    let all_ones = B::splat_all_ones();
    let zero_scalar = B::scalar_from_i64(0);
    let one_mask = <B as SimdBackend>::mask_word_to_scalar(!0u64);

    let blocks_ptr = searcher.blocks.as_mut_ptr();

    // Calculate max_len based on range ends
    let mut max_len = 0;
    for slice in approx_slices.iter().take(batch_size) {
        let len = (slice.1 - slice.0 + 1) as usize;
        if len > max_len {
            max_len = len;
        }
    }
    if searcher.alpha_pattern != !0 {
        max_len += get_overhang_steps(
            t_queries.query_length,
            k as usize,
            alpha.unwrap_or(1.0),
            max_overhang,
        );
    }

    for i in 0..max_len {
        unsafe {
            let block = &mut *blocks_ptr;
            let mut eq_arr = B::LaneArray::default();
            let mut keep_mask_arr = B::LaneArray::default();

            let eq_slice = eq_arr.as_mut();
            let keep_slice = keep_mask_arr.as_mut();

            for lane in 0..batch_size {
                let q_idx = query_indices[lane];
                let start = approx_slices[lane].0;
                let abs_pos = (i as isize) + start;
                if abs_pos >= 0 && (abs_pos as usize) < text.len() {
                    let cur_char = text[abs_pos as usize];
                    let enc = get_encoded(cur_char) as usize;
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

            let (vp_new, vn_new, cost_new) = Myers::<B>::myers_step(
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

    if searcher.alpha_pattern != !0 {
        let eq = all_ones;
        let blocks_ptr = searcher.blocks.as_mut_ptr();
        for _i in 0..t_queries.query_length {
            unsafe {
                let block = &mut *blocks_ptr;
                let (vp_out, vn_out, _cost_out) = Myers::<B>::myers_step(
                    block.vp,
                    block.vn,
                    block.cost,
                    eq,
                    all_ones,
                    last_bit_shift,
                    last_bit_mask,
                );
                let eq_arr = B::to_array(eq);
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

    for lane in 0..batch_size {
        let (r_start, r_end) = range_bounds[lane];
        let approx_start = approx_slices[lane].0;
        for pos in r_start..=r_end {
            let aln = traceback_single(
                searcher,
                lane,
                query_indices[lane],
                t_queries,
                (approx_start, pos),
                text.len(),
            );
            if aln.cost <= k as Cost {
                per_range_alignments[lane].push(aln);
            }
        }
    }

    for (lane, r) in ranges.iter().enumerate() {
        post_process_alignments(post, text.len(), r, &mut per_range_alignments[lane], output);
    }
}

#[inline(always)]
fn clamp_range_to_text_bounds(start: usize, end: usize, upper_bound: usize) -> (usize, usize) {
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

pub fn post_process_alignments(
    post: TracePostProcess,
    text_len: usize,
    _range: &HitRange,
    alignments: &mut Vec<Match>,
    output: &mut Vec<Match>,
) {
    if alignments.is_empty() {
        return;
    }

    let upper_bound = text_len.saturating_sub(1);

    // Now we use sassy's Match struct all positions are usize
    // while we used -1 to indicate starting before the text
    // this now becuase usize::MAX, but we need -1 to see if for example a match at 0 and -1/usize::MAX are adjacent
    // fixme: this is a bit hacky
    let bound = |end: usize| -> isize { if end == usize::MAX { -1 } else { end as isize } };

    match post {
        TracePostProcess::All => {
            for aln in alignments.drain(..) {
                let mut clamped = aln;
                let (start, end) =
                    clamp_range_to_text_bounds(clamped.text_start, clamped.text_end, upper_bound);
                clamped.text_start = start;
                clamped.text_end = end;
                output.push(clamped);
            }
        }
        TracePostProcess::LocalMinima => {
            let mut pattern_tillingma_indices = Vec::new();
            let mut prev_cost = alignments[0].cost;
            let mut prev_idx = 0usize;
            let mut prev_end = bound(alignments[0].text_end);
            let mut last_trend: i8 = 2; // 2 = none, -1 = down/flat, 0 = flat, 1 = up

            for (idx, aln) in alignments.iter().enumerate().skip(1) {
                let cost = aln.cost;

                // A suffix range does not per se translate to consecutive positions
                // in the full match range
                let aln_end = bound(aln.text_end); // Since we now use usize::MAX for left overhang, which is really -1 so 
                // it's adjacent to 0
                if aln_end - prev_end > 1 {
                    pattern_tillingma_indices.push(prev_idx);
                    last_trend = 2;
                    prev_cost = cost;
                    prev_idx = idx;
                    prev_end = aln_end;
                    continue;
                }

                let increasing = cost > prev_cost;
                let decreasing = cost < prev_cost;
                let equal = cost == prev_cost;

                if increasing && last_trend != 1 {
                    pattern_tillingma_indices.push(prev_idx);
                    last_trend = 1;
                } else if decreasing {
                    last_trend = -1;
                } else if equal && last_trend == 2 {
                    last_trend = 0;
                }

                prev_cost = cost;
                prev_idx = idx;
                prev_end = aln_end;
            }

            if last_trend != 1 {
                pattern_tillingma_indices.push(prev_idx);
            }

            for idx in pattern_tillingma_indices {
                let aln = &alignments[idx];
                let mut chosen = aln.clone();
                let (start, end) =
                    clamp_range_to_text_bounds(chosen.text_start, chosen.text_end, upper_bound);
                chosen.text_start = start;
                chosen.text_end = end;
                output.push(chosen);
            }

            alignments.clear();
        }
    }
}

#[inline(always)]
fn extract_simd_lane<B: SimdBackend>(simd_val: B::Simd, lane: usize) -> u64 {
    let arr = B::to_array(simd_val);
    B::scalar_to_u64(arr.as_ref()[lane])
}

fn get_cost_at<B: SimdBackend>(
    searcher: &Myers<B>,
    lane_idx: usize,
    step_idx: isize,
    query_pos_idx: isize,
) -> isize {
    if step_idx < 0 {
        let mask = if query_pos_idx >= 63 {
            !0u64
        } else {
            (1u64 << (query_pos_idx + 1)) - 1
        };
        return (searcher.alpha_pattern & mask).count_ones() as isize;
    }

    let step_data = &searcher.history[lane_idx].steps[step_idx as usize];
    let vp_bits = extract_simd_lane::<B>(step_data.vp, lane_idx);
    let vn_bits = extract_simd_lane::<B>(step_data.vn, lane_idx);

    let mask = if query_pos_idx >= 63 {
        !0u64
    } else {
        (1u64 << (query_pos_idx + 1)) - 1
    };

    let pos = (vp_bits & mask).count_ones() as isize;
    let neg = (vn_bits & mask).count_ones() as isize;

    pos - neg
}

fn traceback_single<B: SimdBackend>(
    searcher: &Myers<B>,
    lane_idx: usize,
    original_query_idx: usize,
    t_queries: &TQueries<B>,
    slice: (isize, isize),
    text_len: usize,
) -> Match {
    let history = &searcher.history[lane_idx];
    let steps = &history.steps;
    let query_len = t_queries.query_length as isize;

    let max_step = slice.1 - slice.0;
    let end_pos = slice.1 as usize;
    let mut curr_step = max_step;
    let mut query_pos = query_len - 1;
    let mut cigar: Cigar = Cigar::default();

    let mut cost_correction = 0;

    let max_valid_text_pos = text_len - 1;
    let pattern_end = if end_pos >= max_valid_text_pos && searcher.alpha != 1.0 {
        let overshoot = end_pos - max_valid_text_pos;
        query_len as usize - overshoot
    } else {
        query_len as usize
    };

    let mut pattern_start = query_len as usize; // Will be updated as we go back

    while curr_step >= 0 && query_pos >= 0 {
        let curr_cost = get_cost_at(searcher, lane_idx, curr_step, query_pos);

        let diag_cost = get_cost_at(searcher, lane_idx, curr_step - 1, query_pos - 1);
        let up_cost = get_cost_at(searcher, lane_idx, curr_step, query_pos - 1);
        let left_cost = get_cost_at(searcher, lane_idx, curr_step - 1, query_pos);

        let step = &steps[curr_step as usize];
        let eq_bits = extract_simd_lane::<B>(step.eq, lane_idx);
        let is_match = (eq_bits & (1u64 << query_pos)) != 0;
        let match_cost = if is_match { 0 } else { 1 };

        if slice.0 + curr_step > (text_len as isize - 1) {
            query_pos -= 1;
            curr_step -= 1;
            cost_correction += 1;
            pattern_start = (query_pos + 1).max(0) as usize;
        } else if curr_cost == diag_cost + match_cost && is_match {
            cigar.push(CigarOp::Match);
            curr_step -= 1;
            query_pos -= 1;
            pattern_start = (query_pos + 1).max(0) as usize;
        } else if curr_cost == left_cost + 1 {
            cigar.push(CigarOp::Ins);
            curr_step -= 1;
        } else if curr_cost == diag_cost + match_cost && !is_match {
            cigar.push(CigarOp::Sub);
            curr_step -= 1;
            query_pos -= 1;
            pattern_start = (query_pos + 1).max(0) as usize;
        } else if curr_cost == up_cost + 1 {
            cigar.push(CigarOp::Del);
            query_pos -= 1;
            pattern_start = (query_pos + 1).max(0) as usize;
        } else {
            panic!("Invalid traceback step reached :(");
        }
    }

    // Handle prefix overhang, but we only add it as operations if overhang was disabled
    if query_pos >= 0 && searcher.alpha == 1.0 {
        let overhang_len = (query_pos + 1) as usize;
        for _ in 0..overhang_len {
            cigar.push(CigarOp::Del);
        }
        pattern_start = 0;
    }

    let slice_start_offset = (curr_step + 1).max(0);
    let text_start = (slice.0 + slice_start_offset).max(0);

    let mut final_raw_cost = get_cost_at(searcher, lane_idx, max_step, query_len - 1);

    if cost_correction > 0 {
        final_raw_cost += (cost_correction as f32 * searcher.alpha).floor() as isize;
    }

    cigar.reverse();

    Match {
        pattern_idx: original_query_idx % t_queries.n_original_queries,
        text_idx: 0,
        text_start: text_start as usize,
        text_end: slice.1 as usize,
        pattern_start,
        pattern_end,
        cost: final_raw_cost as Cost,
        strand: if original_query_idx >= t_queries.n_original_queries {
            Strand::Rc
        } else {
            Strand::Fwd
        },
        cigar,
    }
}
