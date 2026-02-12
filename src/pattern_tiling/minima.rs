use crate::pattern_tiling::trace::clamp_range_to_text_bounds;
use crate::search::Match;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePostProcess {
    All,
    LocalMinima,
}

pub fn process_minima(
    post: TracePostProcess,
    text_len: usize,
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
            let mut pattern_tilingma_indices = Vec::new();
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
                    if last_trend != 1 {
                        pattern_tilingma_indices.push(prev_idx);
                    };
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
                    pattern_tilingma_indices.push(prev_idx);
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
                pattern_tilingma_indices.push(prev_idx);
            }

            for idx in pattern_tilingma_indices {
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
