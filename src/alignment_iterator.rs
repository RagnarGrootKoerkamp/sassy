//! Compute _all_ alignments of cost `k`.

use pa_types::{Cigar, CigarOp, Cost, Pos};

use crate::{
    Match, Searcher, Strand,
    profiles::Profile,
    trace::{CostLookup, CostMatrix, fill},
};

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
    pub fn iterate_all_alignments(
        &self,
        pattern: &[u8],
        text: &[u8],
        k: usize,
        matches: &[Match],
        partial_matches: bool,
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
        let mut cm = CostMatrix::default();

        // Clear reused state
        let mut m = Match {
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

        m.cost = 0;

        // 2. iterate over the ranges
        for range in ranges {
            // 3. compute the matrix for this range
            fill::<P>(
                pattern,
                &text[range.clone()],
                range.len(),
                &mut cm,
                self.alpha,
                self.max_overhang,
            );

            for text_end in range.clone() {
                // This end_pos is > k.
                if cm.get(text_end - range.start, pattern.len()) > k as Cost {
                    continue;
                }

                // 6. Backtrack to get all cost <=k alignments ending at this position.
                // TODO: Filter out clearly suboptimal paths.
                // TODO: Order returned paths by cost.

                m.pattern_end = pattern.len();
                m.text_end = text_end;

                let mut context = Context {
                    pattern,
                    text,
                    range_start: range.start,
                    cm: &cm,
                    m: &mut m,
                    k,
                    partial_matches,
                    callback,
                };
                context.dfs();
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
    callback: &'s mut C,
}

impl<'s, C: Callback> Context<'s, C> {
    fn dfs(&mut self) -> Continuation {
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

        let mut edges = vec![];
        for mut op in [CigarOp::Match, CigarOp::Del, CigarOp::Ins] {
            // Filter in-range edges.
            if !(min_pos + op.delta() <= pos) {
                continue;
            }
            let new_pos = pos - op.delta();
            // Replate Match by Sub if needed
            if op == CigarOp::Match
                && self.text[new_pos.0 as usize] != self.pattern[new_pos.1 as usize]
            {
                op = CigarOp::Sub;
            }
            // Min total cost of path through this edge.
            let total_cost = self.m.cost
                + op.edit_cost()
                + self
                    .cm
                    .get(new_pos.0 as usize - self.range_start, new_pos.1 as _);
            // Push edge if total cost is <= k.
            if total_cost <= self.k as Cost {
                edges.push((op, total_cost));
            }
        }

        // Stable sort edges by total cost, preferring match/sub in case of ties.
        edges.sort_by_key(|(_, cost)| *cost);

        for (op, _cost) in edges {
            let delta = op.delta();
            // Update the match
            self.m.text_start -= delta.0 as usize;
            self.m.pattern_start -= delta.1 as usize;
            self.m.cost += op.edit_cost();
            self.m.cigar.push(op);
            // Recurse!
            let continuation = self.dfs();
            // Revert the match
            assert_eq!(self.m.cigar.pop_op(), Some(op));
            self.m.cost -= op.edit_cost();
            self.m.pattern_start += delta.1 as usize;
            self.m.text_start += delta.0 as usize;

            match continuation {
                Continuation::Continue => {}
                Continuation::Prune => unreachable!(),
                Continuation::Break => return Continuation::Break,
            }
        }

        return Continuation::Continue;
    }
}
