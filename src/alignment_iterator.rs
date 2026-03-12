//! Compute _all_ alignments of cost `k`.

use itertools::Itertools;
use pa_types::{Cigar, CigarOp, Cost};

use crate::{
    Match, Searcher, Strand,
    profiles::Profile,
    trace::{CostLookup, CostMatrix, fill},
};

impl<P: Profile> Searcher<P> {
    /// Iterate over _all_ alignments of cost up to `k`.
    ///
    /// First run `search_all` to get all endpoints.
    /// Then pass those to `all_alignments` to get an iterator over _all_ alignments.
    fn iterate_all_alignments(
        &self,
        pattern: &[u8],
        text: &[u8],
        k: usize,
        matches: &[Match],
    ) -> impl Iterator<Item = &Match> {
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
        // (TODO? inline this into the main iterator?)
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

        let mut cm = CostMatrix::default();
        let mut m = Match {
            pattern_idx: 0,
            text_idx: 0,
            text_start: 0,
            text_end: 0,
            pattern_start: 0,
            pattern_end: 0,
            cost: 0,
            strand: Strand::Fwd,
            cigar: Cigar::default(),
        };
        let mut stack = vec![];

        // 2. iterate over the ranges
        ranges.into_iter().flat_map(|range| {
            // 3. compute the matrix for this range
            fill(
                pattern,
                &text[range],
                range.len(),
                &mut cm,
                self.alpha,
                self.max_overhang,
            );

            // 4. Iterate over end positions in the range.
            (0..range.len())
                // 5. Filter end positions with cost <= k.
                .filter_map(|end| {
                    if cm.get(end, pattern.len()) <= k as Cost {
                        Some(end)
                    } else {
                        None
                    }
                })
                .flat_map(move |end| {
                    // 6. Backtrack to get all cost <=k alignments ending at this position.
                    // TODO: Filter out clearly suboptimal paths.
                    // TODO: Order returned paths by cost.

                    // Clear reused state
                    stack.clear();
                    m.text_end = end;
                    m.pattern_end = pattern.len();
                    m.cost = 0;

                    // Call the DFS function until it returns None.
                    std::iter::from_fn(move || {
                        while let Some(state) = stack.last_mut() {
                            // FIXME: Continue here.
                        }
                        None
                    })
                })
        })
    }
}
