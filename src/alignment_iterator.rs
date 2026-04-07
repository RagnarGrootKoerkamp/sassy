//! Compute _all_ alignments of cost `k`.

use pa_types::{Cigar, CigarOp, Cost, Pos};

use crate::{
    Match, RcSearchAble, Searcher, Strand,
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

fn net_insertions_since_last_match(cigar: &Cigar) -> i32 {
    let mut net = 0i32;
    for elem in cigar.ops.iter().rev() {
        match elem.op {
            CigarOp::Match => break,
            CigarOp::Ins   => net += elem.cnt,
            CigarOp::Del   => net -= elem.cnt,
            CigarOp::Sub   => {} // ignore
        }
    }
    net
}

/// (match_is_complete, (partial) match) -> continuation.
pub trait Callback: FnMut(bool, &mut Match) -> Continuation {}

impl<F: FnMut(bool, &mut Match) -> Continuation> Callback for F {}

impl<P: Profile> Searcher<P> {
    /// Iterate over _all_ alignments of cost up to `k`.
    ///
    /// `matches` may contain both `Strand::Fwd` and `Strand::Rc` entries.
    /// Fwd matches are traced on the forward text; RC matches are traced on the
    /// reversed text using the complemented pattern, then translated back to
    /// forward-text coordinates before the callback fires.
    ///
    /// If `partial_matches` is `true`, the callback is called for *every* visited DFS state.
    ///
    /// If `prune_suboptimal` is `true`, path for which some part can be replaced by exact matches are skipped.
    /// E.g., if `====` is an option, this will skip over `=I=D=`, and similarly, this will prefer `===...` over `=I=...`.
    #[allow(clippy::too_many_arguments)]
    pub fn iterate_all_alignments<I: RcSearchAble + ?Sized>(
        &self,
        pattern: &[u8],
        text: &I,
        k: usize,
        matches: &mut [Match],
        partial_matches: bool,
        prune_suboptimal: bool,
        callback: &mut impl Callback,
    ) {
        assert_eq!(self.alpha, None, "Tracing all alignments with overhang is not yet implemented.");

        let fwd_text = text.text();
        let fwd_text = fwd_text.as_ref();

        // `search_all` returns Fwd matches first then Rc — find the split point.
        let split = matches.partition_point(|m| m.strand == Strand::Fwd);
        let (fwd, rc) = matches.split_at_mut(split);

        // --- Forward strand ---
        if !fwd.is_empty() {
            self.iterate_one_strand(pattern, fwd_text, k, fwd, partial_matches, prune_suboptimal, callback, None);
        }

        // --- Reverse-complement strand ---
        if !rc.is_empty() {
            let fwd_len = fwd_text.len();
            let rev_text = text.rev_text();
            let rev_text = rev_text.as_ref();
            let comp_pattern = P::complement(pattern);

            // Wrap callback: translate DFS results (rev-text coords) back to fwd-text coords.
            let mut rc_callback = |complete: bool, m: &mut Match| -> Continuation {
                let orig_start  = m.text_start;
                let orig_end    = m.text_end;
                let orig_strand = m.strand;
                m.text_start = fwd_len - orig_end;
                m.text_end   = fwd_len - orig_start;
                m.strand     = Strand::Rc;
                let result = callback(complete, m);
                m.text_start = orig_start;
                m.text_end   = orig_end;
                m.strand     = orig_strand;
                result
            };

            // RC matches are in fwd coords; pass flip=Some(fwd_len) so iterate_one_strand
            // derives rev-text endpoints on the fly without mutating the slice.
            self.iterate_one_strand(&comp_pattern, rev_text, k, rc,
                                    partial_matches, prune_suboptimal, &mut rc_callback,
                                    Some(fwd_len));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn iterate_one_strand(
        &self,
        pattern: &[u8],
        text: &[u8],
        k: usize,
        matches: &[Match],
        partial_matches: bool,
        prune_suboptimal: bool,
        callback: &mut impl Callback,
        flip: Option<usize>,
    ) {
        let width = k + pattern.len();

        // When flip=Some(fwd_len), matches are in fwd coords; derive the rev-text endpoint
        // as fwd_len - text_start (RC matches are sorted by text_start DESC in fwd coords,
        // so fwd_len - text_start is ascending — the order iterate_one_strand requires).
        let eff_end = |m: &Match| match flip {
            None          => m.text_end,
            Some(fwd_len) => fwd_len - m.text_start,
        };
        debug_assert!(matches.windows(2).all(|w| eff_end(&w[0]) <= eff_end(&w[1])));

        // 1. Group matches as long as they are close
        let mut ranges = vec![];
        if let [fm, tail @ ..] = matches {
            let mut first_end = eff_end(fm).saturating_sub(width);
            let mut last_end = eff_end(fm);
            for m in tail {
                if eff_end(m) <= last_end + width {
                    last_end = eff_end(m);
                } else {
                    ranges.push(first_end..last_end);
                    first_end = eff_end(m).saturating_sub(width);
                    last_end = eff_end(m);
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

                // We may not have both inserted and deleted bases since the last match
                // NB: This forces taking a diagonal path (subs) wherever possible
                let net_ins = net_insertions_since_last_match(&self.m.cigar);
                if (op == CigarOp::Ins && net_ins < 0) || (op == CigarOp::Del && net_ins > 0)  {
                    continue;
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

#[cfg(test)]
mod tests {
    use super::net_insertions_since_last_match;
    use pa_types::{Cigar, CigarOp};
    use CigarOp::{Del as D, Ins as I, Match as M, Sub as X};

    fn cigar(ops: &[pa_types::CigarOp]) -> Cigar {
        let mut c = Cigar::default();
        for &op in ops {
            c.push(op);
        }
        c
    }

    #[test]
    fn net_insertions_since_last_match_cases() {
        let cases: &[(&[pa_types::CigarOp], i32)] = &[
            (&[],                  0),  // empty cigar
            (&[M],                 0),  // match resets immediately
            (&[I, I, I],           3),  // all insertions, no anchor
            (&[D, D],             -2),  // all deletions, no anchor
            (&[M, I, I],           2),  // two I's after match
            (&[M, D, D],          -2),  // two D's after match
            (&[M, I, I, D],        1),  // net 2I − 1D
            (&[M, I, I, D, D],     0),  // balanced
            (&[I, X, D],           0),  // X is transparent: +1 − 1
            (&[M, I, X, D],        0),  // X between I and D, after anchor
            (&[M, X, X, I],        1),  // X's skipped, only I counted
            (&[I, I, M, D, D],    -2),  // stop at M, only trailing DD visible
            (&[M, D, M, I, I],     2),  // stop at second M, see II
        ];
        for &(ops, expected) in cases {
            assert_eq!(
                net_insertions_since_last_match(&cigar(ops)),
                expected,
                "ops = {ops:?}"
            );
        }
    }
}
