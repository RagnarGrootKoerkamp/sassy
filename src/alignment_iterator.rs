//! Compute _all_ alignments of cost `k`.

use pa_types::{Cigar, CigarOp, Cost, Pos};

use crate::{
    Match, Searcher, Strand,
    profiles::Profile,
    trace::{CostLookup, CostMatrix, fill},
};

#[cfg_attr(
    feature = "python",
    pyo3::pyclass(eq, from_py_object, module = "sassy")
)]
#[derive(PartialEq, Clone)]
pub enum Continuation {
    /// Continue exploring the subtree.
    Continue,
    /// Prune the subtree of alignments ending in the current one.
    Prune,
    /// Found a sufficiently good alignment for the current end pos, go to the next one.
    Break,
}

/// (match_is_complete, (partial) match) -> continuation.
pub trait Callback: FnMut(bool, &Match) -> Continuation {}

impl<F: FnMut(bool, &Match) -> Continuation> Callback for F {}

impl<P: Profile> Searcher<P> {
    /// Iterate over _all_ alignments of cost up to `k`.
    ///
    /// First run `search_all` to get all endpoints.
    /// Then pass those to `all_alignments` to get an iterator over _all_ alignments.
    ///
    /// If `partial_matches` is `true`, the callback is called for *every* visited DFS state.
    ///
    /// If `prune_suboptimal` is `true`, path for which some part can be replaced by exact matches are skipped.
    /// E.g., if `====` is an option, this will skip over `=I=D=`, and similarly, this will prefer `===...` over `=I=...`.
    #[allow(clippy::too_many_arguments)]
    pub fn iterate_all_alignments(
        &self,
        pattern: &[u8],
        text: &[u8],
        k: usize,
        matches: &[Match],
        partial_matches: bool,
        prune_suboptimal: bool,
        callback: &mut impl Callback,
    ) {
        let width = k + pattern.len();

        // for now, assume fwd matches
        for m in matches {
            assert_eq!(
                m.strand,
                crate::Strand::Fwd,
                "Tracing all alignments for RC matches is not yet implemented."
            );
        }
        assert_eq!(
            self.alpha, None,
            "Tracing all alignments with overhang is not yet implemented."
        );

        // 1. Group matches as long as they are close
        let mut ranges = vec![];
        if let [fm, tail @ ..] = matches {
            let mut first_end = fm.text_end.saturating_sub(width);
            let mut last_end = fm.text_end;
            for m in tail {
                if m.text_end <= last_end + width {
                    last_end = m.text_end;
                } else {
                    ranges.push(first_end..last_end);
                    first_end = m.text_end.saturating_sub(width);
                    last_end = m.text_end;
                }
            }
            ranges.push(first_end..last_end);
        }

        // Reusable cost matrix.
        let cm = &mut CostMatrix::default();

        // Clear reused state
        let m = &mut Match {
            pattern_idx: 0,
            text_idx: 0,
            text_start: 0,
            text_end: 0,
            pattern_start: pattern.len(),
            pattern_end: pattern.len(),
            cost: 0,
            strand: Strand::Fwd,
            cigar: Cigar::default(),
        };

        let last_row_in_diagonal = &mut vec![];

        // 2. iterate over the ranges
        for range in ranges {
            // 3. compute the matrix for this range
            fill::<P>(
                pattern,
                &text[range.clone()],
                range.len(),
                cm,
                self.alpha,
                self.max_overhang,
            );

            last_row_in_diagonal.resize(range.len() + pattern.len() + 1, pattern.len() as _);
            last_row_in_diagonal.fill(pattern.len() as _);

            for text_end in range.start..=range.end {
                // This end_pos is > k.
                if cm.get(text_end - range.start, pattern.len()) > k as Cost {
                    continue;
                }

                // 6. Backtrack to get all cost <=k alignments ending at this position.
                // TODO: Order returned paths by cost.

                m.pattern_start = pattern.len();
                m.pattern_end = pattern.len();
                m.text_start = text_end;
                m.text_end = text_end;
                m.cost = 0;
                m.cigar = Cigar::default();

                let mut context = Context {
                    pattern,
                    text,
                    range_start: range.start,
                    cm,
                    m,
                    k,
                    partial_matches,
                    prune_suboptimal,
                    callback,
                    last_row_in_diagonal,
                };
                context.dfs::<P>();
            }
        }
    }
}

struct Context<'s, C: Callback> {
    pattern: &'s [u8],
    text: &'s [u8],
    range_start: usize,
    cm: &'s CostMatrix,
    m: &'s mut Match,
    k: usize,
    partial_matches: bool,
    prune_suboptimal: bool,
    callback: &'s mut C,
    last_row_in_diagonal: &'s mut Vec<pa_types::I>,
}

impl<'s, C: Callback> Context<'s, C> {
    fn dfs<P: Profile>(&mut self) -> Continuation {
        let full_match = self.m.pattern_start == 0;
        if full_match || self.partial_matches {
            self.m.cigar.reverse();
            let continuation = (self.callback)(full_match, self.m);
            self.m.cigar.reverse();
            match continuation {
                Continuation::Continue => {}
                Continuation::Prune => return Continuation::Continue,
                Continuation::Break => return Continuation::Break,
            }
        }

        let min_pos = Pos(self.range_start as _, 0);
        let pos = Pos(self.m.text_start as _, self.m.pattern_start as _);

        let mut edges = arrayvec::ArrayVec::<(CigarOp, Cost), 3>::new();

        for mut op in [CigarOp::Match, CigarOp::Del, CigarOp::Ins] {
            // Don't allow leading or trailing deletions
            if op == CigarOp::Del && (self.m.pattern_start == 0 || self.m.pattern_start == self.pattern.len()) {
                continue;
            }
            // Filter in-range edges.
            let new_pos = pos - op.delta();
            if new_pos.0 < min_pos.0 || new_pos.1 < min_pos.1 {
                // matrix OOB (either coordinate out of range)
                continue;
            }
            // Replate Match by Sub if needed
            if op == CigarOp::Match && !P::is_match(self.text[new_pos.0 as usize], self.pattern[new_pos.1 as usize])
            {
                op = CigarOp::Sub;
            }
            // Min total cost of path through this edge.
            let total_cost = self.m.cost
                + op.edit_cost()
                + self
                    .cm
                    .get(new_pos.0 as usize - self.range_start, new_pos.1 as _);
            // Skip edge if total cost is > k.
            if total_cost > self.k as Cost {
                continue;
            }

            if self.prune_suboptimal {
                // We may not *leave* a diagonal if it can be extended by exact matches to the top of the matrix.
                if op == CigarOp::Ins || op == CigarOp::Del {
                    let pat_slice = &self.pattern[..pos.1 as usize];
                    let text_slice = &self.text[(pos.0 - pos.1).max(0) as usize..pos.0 as usize];
                    if P::is_match_slice(pat_slice, text_slice) {
                        continue;
                    }
                }

                // We may not *enter* a diagonal if it was reachable by exact matches from either:
                // - the bottom of the matrix, or
                // - the last time we were in this diagonal.
                if op == CigarOp::Ins || op == CigarOp::Del {
                    // The last (most recent) row we visited in the `new_pos` diagonal.
                    // Defaults to `pattern.len()` for bottom of the matrix.
                    let last_in_diag = self.last_row_in_diagonal[new_pos.0 as usize
                        + self.pattern.len()
                        - self.range_start
                        - new_pos.1 as usize];
                    let pat_slice = &self.pattern[new_pos.1 as usize..last_in_diag as usize];
                    let text_end = new_pos.0 as usize + pat_slice.len();
                    if text_end <= self.text.len() {
                        let text_slice = &self.text[new_pos.0 as usize..text_end];
                        if P::is_match_slice(pat_slice, text_slice) {
                            continue;
                        }
                    }
                }
            }

            edges.push((op, total_cost));
        }

        // Stable sort edges by total cost, preferring match/sub in case of ties.
        edges.sort_by_key(|(_, cost)| *cost);

        for (op, _cost) in edges {
            let delta = op.delta();
            let new_pos = pos - delta;

            // update last_in_diagonal
            let diagonal =
                new_pos.0 as usize + self.pattern.len() - self.range_start - new_pos.1 as usize;
            let old_last_in_diag = self.last_row_in_diagonal[diagonal];
            self.last_row_in_diagonal[diagonal] = new_pos.1;

            // Update the match
            self.m.text_start -= delta.0 as usize;
            self.m.pattern_start -= delta.1 as usize;
            self.m.cost += op.edit_cost();
            self.m.cigar.push(op);
            // Recurse!
            let continuation = self.dfs::<P>();
            // Revert the match
            self.m.text_start += delta.0 as usize;
            self.m.pattern_start += delta.1 as usize;
            self.m.cost -= op.edit_cost();
            assert_eq!(self.m.cigar.pop_op(), Some(op));

            // Revert last_in_diagonal
            self.last_row_in_diagonal[diagonal] = old_last_in_diag;

            match continuation {
                Continuation::Continue => {}
                Continuation::Prune => unreachable!(),
                Continuation::Break => return Continuation::Break,
            }
        }

        Continuation::Continue
    }
}
