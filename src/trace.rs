use crate::bitpacking::compute_block;
use crate::delta_encoding::H;
use crate::delta_encoding::V;
use crate::profiles::Profile;
use crate::search::init_deltas_for_overshoot_all_lanes;
use crate::search::init_deltas_for_overshoot_scalar;
use pa_types::Cigar;
use pa_types::CigarOp;
use pa_types::Cost;
use pa_types::I;

use crate::LANES;
use crate::S;
use crate::bitpacking::compute_block_simd;
use crate::search::{Match, Strand};
use std::array::from_fn;

pub trait CostLookup {
    /// Get the DP cost at text position `i`, pattern position `j`.
    /// i=0 and j=0 are are "outside" the text/pattern
    fn get(&self, i: usize, j: usize) -> Cost;
}

#[derive(Debug, Clone, Default)]
pub struct CostMatrix {
    /// Query length.
    q: usize,
    deltas: Vec<V>,
    pub alpha: Option<f32>,
    pub max_overhang: Option<usize>,
}

impl CostLookup for CostMatrix {
    /// i: text idx
    /// j: query idx
    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> Cost {
        let mut s = if let Some(alpha) = self.alpha {
            overhang_cost(j, alpha, self.max_overhang)
        } else {
            j as Cost
        };
        for idx in (j..j + i / 64 * (self.q + 1)).step_by(self.q + 1) {
            s += self.deltas[idx].value();
        }
        if !i.is_multiple_of(64) {
            s += self.deltas[j + i / 64 * (self.q + 1)].value_of_prefix(i as I % 64);
        }
        s
    }
}

/// Compute the overhang (free-end) cost for `j` unconsumed pattern characters
/// at the text boundary, given overhang rate `alpha` and optional cap `max_overhang`.
fn overhang_cost(j: usize, alpha: f32, max_overhang: Option<usize>) -> Cost {
    if let Some(mo) = max_overhang {
        (j.min(mo) as f32 * alpha).floor() as Cost + j.saturating_sub(mo) as Cost
    } else {
        (j as f32 * alpha).floor() as Cost
    }
}

// ---------------------------------------------------------------------------
// All-alignments DFS iterator
// ---------------------------------------------------------------------------

/// Which edit operation to try next during DFS backtracking.
///
/// Operations are tried in the order Match -> Sub -> Del -> Ins,
/// matching the greedy preference order of `get_trace`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TraceOp {
    Match,
    Sub,
    Del,
    Ins,
}

impl TraceOp {
    /// Return the next operation to try, or `None` if all have been exhausted.
    fn next(self) -> Option<Self> {
        match self {
            TraceOp::Match => Some(TraceOp::Sub),
            TraceOp::Sub => Some(TraceOp::Del),
            TraceOp::Del => Some(TraceOp::Ins),
            TraceOp::Ins => None,
        }
    }
}

/// One frame on the DFS stack.
///
/// Stores the coordinates `(i, j)` of the current cell, the next operation
/// to try, and where `cigar_ops` should be truncated when we backtrack to this
/// frame to try the next operation (erasing ops pushed by exhausted subtrees).
///
/// `next_op` is the *only* progress variable: because operations are always
/// tried in the fixed order Match->Sub->Del->Ins, a single field encodes which
/// subset has already been tried. The path taken to reach this cell is
/// already recorded in the shared `cigar_ops` Vec, so no additional bookkeeping
/// is needed.
struct TraceFrame {
    i: usize,
    j: usize,
    /// Remaining edit budget: starts at `max_cost` for the root frame and
    /// decrements by 1 for each Sub/Del/Ins step. A Match step leaves it
    /// unchanged. When the base case `j == 0` is reached, the alignment cost
    /// is `max_cost - edit_budget`.
    edit_budget: Cost,
    /// Next operation to try in the DFS.
    next_op: Option<TraceOp>,
    /// Restore `cigar_ops` to this length when re-entering this frame.
    ///
    /// Each frame is entered from its parent by pushing exactly one `CigarOp`
    /// onto the shared `cigar_ops` vec, so `cigar_len_at_entry` equals the
    /// length of `cigar_ops` after that push. When a subtree is exhausted and
    /// we backtrack to try the next op, truncating to `cigar_len_at_entry`
    /// removes precisely the ops that subtree pushed — no more, no less. A
    /// single length value suffices because the shared vec grows monotonically
    /// within a subtree and is only shortened on backtrack.
    cigar_len_at_entry: usize,
}

/// Lazy iterator over all valid alignments (CIGARs / start positions) for a
/// single matched end position.
///
/// Implements [`Iterator<Item = Match>`] directly — use it like any standard iterator.
///
/// All yielded `Match` objects share the same strand. For forward alignments
/// they also share `text_end` (start positions vary); for RC alignments they
/// share `text_start` in forward coordinates (end positions vary, because the
/// fixed RC end maps to the forward start).

pub struct AllAlignmentsAtPos {
    matrix: CostMatrix,
    pattern: Vec<u8>,
    text: Vec<u8>,
    text_offset: usize,
    // Stores P::is_match resolved at construction time rather than making the
    // struct generic over P: Profile. AllAlignmentsAtPos must be non-generic
    // because PyO3 #[pyclass] types cannot have type parameters.
    is_match_fn: fn(u8, u8) -> bool,
    strand: Strand,
    /// Optimal edit cost at the end position; used for right-overshoot arithmetic.
    optimal_cost: Cost,
    /// Maximum total cost of any yielded `Match`: `min(optimal_cost + margin, k)`.
    max_cost: Cost,
    /// Exclusive end of pattern; may be < pattern.len() due to right overshoot.
    pattern_end: usize,
    /// `Some(fwd_len)` for RC alignments: used to flip coordinates when building a match.
    fwd_text_len: Option<usize>,
    stack: Vec<TraceFrame>,
    cigar_ops: Vec<CigarOp>,
}

impl AllAlignmentsAtPos {
    /// Return the optimal (minimum) alignment cost at this end position.
    pub fn optimal_cost(&self) -> Cost {
        self.optimal_cost
    }

    /// Returns `true` if `pattern[j]` and `text[i]` are considered matching
    /// under the alphabet's scoring function.
    fn chars_match(&self, i: usize, j: usize) -> bool {
        (self.is_match_fn)(self.pattern[j], self.text[i])
    }

    /// Push a child frame onto the DFS stack for the given transition.
    /// Appends `op` to the shared cigar buffer first, then records the
    /// resulting buffer length as the frame's rollback point.
    fn push_child_frame(&mut self, i: usize, j: usize, edit_budget: Cost, op: CigarOp) {
        self.cigar_ops.push(op);
        self.stack.push(TraceFrame {
            i,
            j,
            edit_budget,
            next_op: Some(TraceOp::Match),
            cigar_len_at_entry: self.cigar_ops.len(),
        });
    }

    /// Build a `Match` from the current `cigar_ops` state.
    ///
    /// `i` is the text position at the start of the alignment (within the slice).
    /// `pattern_start` the pattern position at the start of the alignment which
    /// is usually 0 except for left-overshoot matches.
    /// `cost` is the alignment cost.
    fn build_match(&self, i: usize, pattern_start: usize, cost: Cost) -> Match {
        let mut cigar = Cigar::default();
        // cigar_ops is in reverse order (end->start); iterate in reverse to
        // produce forward-order output without a separate reverse pass.
        for &op in self.cigar_ops.iter().rev() {
            cigar.push(op);
        }

        let slice_end = self.text_offset + self.text.len();
        let (text_start, text_end) = if let Some(fwd_len) = self.fwd_text_len {
            // RC: flip from reversed-text coords to forward-text coords.
            let rc_start = self.text_offset + i;
            (fwd_len - slice_end, fwd_len - rc_start)
        } else {
            (self.text_offset + i, slice_end)
        };

        Match {
            pattern_idx: 0,
            text_idx: 0,
            cost,
            text_start,
            text_end,
            pattern_start,
            pattern_end: self.pattern_end,
            strand: self.strand,
            cigar,
        }
    }
}

impl Iterator for AllAlignmentsAtPos {
    type Item = Match;

    /// Perform the DFS and return the next match.
    fn next(&mut self) -> Option<Match> {
        loop {
            let frame = self.stack.last_mut()?;

            // Restore cigar to the state when this frame was entered.
            // This removes ops pushed by previously exhausted subtrees.
            self.cigar_ops.truncate(frame.cigar_len_at_entry);

            // ── Base case: j == 0 ────────────────────────────────────────────
            if frame.j == 0 {
                let i = frame.i;
                let remaining = frame.edit_budget;
                self.stack.pop();
                let cost = self.max_cost - remaining;
                return Some(self.build_match(i, 0, cost));
            }

            // ── Base case: i == 0 with left overshoot ───────────────────────
            // When alpha is None, i == 0 is not a terminal — only Ins ops are
            // valid (Match/Sub/Del require i > 0). Fall through to op-checking.
            if frame.i == 0
                && let Some(alpha) = self.matrix.alpha
            {
                let j = frame.j;
                let remaining = frame.edit_budget;
                self.stack.pop();
                let oc = overhang_cost(j, alpha, self.matrix.max_overhang);
                if oc <= remaining {
                    let cost = (self.max_cost - remaining) + oc;
                    return Some(self.build_match(0, j, cost));
                }
                continue;
            }

            // ── Advance to next op (or pop if exhausted) ────────────────────
            let op = match frame.next_op {
                None => {
                    self.stack.pop();
                    continue;
                }
                Some(op) => op,
            };
            // Advance before validity check so the next iteration tries the
            // subsequent op even if this one is invalid.
            frame.next_op = op.next();

            let (i, j, edit_budget) = (frame.i, frame.j, frame.edit_budget);

            // ── Check validity and push child frame ──────────────────────────
            //
            // Each arm checks `matrix.get(i-1, j-1)` (or the Del/Ins predecessor)
            // against `edit_budget`: the stored value is the DP minimum cost from
            // that predecessor to any valid start, so the check guarantees a valid
            // completion exists within the remaining budget.
            //
            // With margin=0, edit_budget == matrix.get(current) on every valid
            // path (DP monotonicity ensures this), so `<=` degenerates to `==`
            // and the behaviour is identical to the DFS over optimal alignments.
            match op {
                TraceOp::Match => {
                    if i > 0
                        && self.matrix.get(i - 1, j - 1) <= edit_budget
                        && self.chars_match(i - 1, j - 1)
                    {
                        self.push_child_frame(i - 1, j - 1, edit_budget, CigarOp::Match);
                    }
                }
                TraceOp::Sub => {
                    // Explicit !chars_match guard: in the optimal case this is
                    // implicitly impossible (dp[i-1][j-1] >= dp[i][j] when
                    // chars match. In allowing suboptimal matches and traversals
                    // in the cost matrix, we must enforce a mismatch to avoid
                    // conflating matches for substitutions.
                    if i > 0
                        && edit_budget >= 1
                        && self.matrix.get(i - 1, j - 1) < edit_budget
                        && !self.chars_match(i - 1, j - 1)
                    {
                        self.push_child_frame(i - 1, j - 1, edit_budget - 1, CigarOp::Sub);
                    }
                }
                TraceOp::Del => {
                    if i > 0 && edit_budget >= 1 && self.matrix.get(i - 1, j) < edit_budget {
                        self.push_child_frame(i - 1, j, edit_budget - 1, CigarOp::Del);
                    }
                }
                TraceOp::Ins => {
                    if j > 0 && edit_budget >= 1 && self.matrix.get(i, j - 1) < edit_budget {
                        self.push_child_frame(i, j - 1, edit_budget - 1, CigarOp::Ins);
                    }
                }
            }
            // If invalid: loop again — truncate+advance next op at top of loop.
        }
    }
}

/// Fill the cost matrix for `text_slice` and return an [`AllAlignmentsAtPos`]
/// ready for depth-first traversal. Mirrors [`get_trace`].
///
/// `text_slice` is the window `text[text_offset .. text_offset + text_slice.len()]`.
/// `end_pos` is the match end in the full text; it may exceed
/// `text_offset + text_slice.len()` when the pattern overshoots the right edge.
/// `fill_len` is the number of text columns to fill (`pattern.len() + k`).
///
/// `alpha` and `max_overhang` are the overhang parameters from the [`Searcher`]
/// configuration. `strand` and `fwd_text_len` are stored in every [`Match`]
/// yielded by the iterator; set `fwd_text_len` to `Some(fwd_len)` for RC
/// alignments so that coordinates are flipped back to forward-text space.
///
/// `margin` controls sub-optimal enumeration: the iterator yields all alignments
/// with cost <= `optimal_cost + margin`. Pass `0` to enumerate only optimal
/// alignments. The budget is clamped to `k` so the DFS never accesses matrix
/// cells outside the filled window.
///
/// Yielded [`Match`] objects always have `pattern_idx = 0` and `text_idx = 0`.
/// This function is designed for single-pattern, single-text search only.
#[allow(clippy::too_many_arguments)]
pub fn all_alignments_at_position<P: Profile>(
    pattern: &[u8],
    text_slice: &[u8],
    text_offset: usize,
    end_pos: usize,
    k: Cost,
    alpha: Option<f32>,
    max_overhang: Option<usize>,
    strand: Strand,
    fwd_text_len: Option<usize>,
    margin: Cost,
) -> AllAlignmentsAtPos {
    let fill_len = pattern.len() + k as usize;
    let mut matrix = CostMatrix::default();
    fill::<P>(
        pattern,
        text_slice,
        fill_len,
        &mut matrix,
        alpha,
        max_overhang,
    );

    let mut i = end_pos - text_offset;
    let mut j = pattern.len();
    let mut pattern_end = pattern.len();
    let mut total_cost = matrix.get(i, j);

    // Right-overshoot adjustment — mirrors get_trace:287-299.
    if i > text_slice.len() {
        let overshoot = i - text_slice.len();
        pattern_end -= overshoot;
        let overshoot_cost = (overshoot as f32 * alpha.unwrap()).floor() as Cost;
        total_cost += overshoot_cost;
        i -= overshoot;
        j -= overshoot;
    }

    // Clamp total_budget to k so the DFS never accesses cells outside the
    // filled matrix window (fill_len = pattern.len() + k).
    let total_budget = (total_cost + margin).min(k);

    let init_frame = TraceFrame {
        i,
        j,
        edit_budget: total_budget,
        next_op: Some(TraceOp::Match),
        cigar_len_at_entry: 0,
    };

    // NB: There are two optimizations that may be worth considering in the
    // in the future to make the memory allocations in a sequence of returned
    // iterators more efficient:
    // 1. Instead of reallocating a CostMatrix at each aligned position, an
    //    implementation could reuse a preallocated cost-matrix buffer,
    //    initialized once by the caller.
    // 2. Insted of holding (typically) small pattern and text byte vectors,
    //    a pure rust implementation would hold a slice or, again, a pre-allocated
    //    text and pattern buffer. Below, a Vec<u8> is used because the returned
    //    struct exposed in the PyO3 interface can only hold 'static lifetimes.
    //    Matched text and pattern slices are typically small so the
    //    implementation below is tolerable, for now.
    AllAlignmentsAtPos {
        matrix,
        pattern: pattern.to_vec(),
        text: text_slice.to_vec(),
        text_offset,
        is_match_fn: P::is_match,
        strand,
        optimal_cost: total_cost,
        max_cost: total_budget,
        pattern_end,
        fwd_text_len,
        stack: vec![init_frame],
        cigar_ops: Vec::new(),
    }
}

/// Compute the full n*m matrix corresponding to the query * text alignment.
pub fn fill<P: Profile>(
    query: &[u8],
    text: &[u8],
    len: usize,
    m: &mut CostMatrix,
    alpha: Option<f32>,
    max_overhang: Option<usize>,
) {
    if alpha.is_some() && !P::supports_overhang() {
        panic!(
            "Overhang is not supported for {:?}",
            std::any::type_name::<P>()
        );
    }
    m.alpha = alpha;
    m.max_overhang = max_overhang;
    m.q = query.len();
    m.deltas.clear();
    m.deltas.reserve((m.q + 1) * len.div_ceil(64));
    let (profiler, query_profile) = P::encode_pattern(query);
    let mut h = vec![H(1, 0); query.len()];

    init_deltas_for_overshoot_scalar(&mut h, alpha, max_overhang);

    let mut text_profile = P::alloc_out();

    let num_chunks = len.div_ceil(64);

    // Process chunks of 64 chars, that end exactly at the end of the text.
    for i in 0..num_chunks {
        let mut slice: [u8; 64] = [b'N'; 64];
        let block = text.get(64 * i..).unwrap_or_default();
        let block = block.get(..64).unwrap_or(block);
        slice[..block.len()].copy_from_slice(block);
        profiler.encode_ref(&slice, &mut text_profile);

        let mut v = V::zero();

        m.deltas.push(v);
        for j in 0..query.len() {
            compute_block::<P>(&mut h[j], &mut v, &query_profile[j], &text_profile);
            m.deltas.push(v);
        }
    }
}

// FIXME: DEDUP
pub fn simd_fill<P: Profile>(
    pattern: &[u8],
    texts: &[&[u8]],
    max_len: usize,
    m: &mut [CostMatrix; LANES],
    alpha: Option<f32>,
    max_overhang: Option<usize>,
) {
    assert!(texts.len() <= LANES);
    if alpha.is_some() && !P::supports_overhang() {
        panic!(
            "Overhang is not supported for {:?}",
            std::any::type_name::<P>()
        );
    }
    let lanes = texts.len();

    let (profiler, pattern_profile) = P::encode_pattern(pattern);
    let num_chunks = max_len.div_ceil(64);

    for m in &mut *m {
        m.alpha = alpha;
        m.max_overhang = max_overhang;
        m.q = pattern.len();
        m.deltas.clear();
        m.deltas.reserve((m.q + 1) * num_chunks);
    }

    let mut hp: Vec<S> = Vec::with_capacity(pattern.len());
    let mut hm: Vec<S> = Vec::with_capacity(pattern.len());
    hp.resize(pattern.len(), S::splat(1));
    hm.resize(pattern.len(), S::splat(0));

    // NOTE: It's OK to always fill the left with 010101, even if it's not
    // actually the left of the text, because in that case the left column can't
    // be included in the alignment anyway. (The text has length q+k in that case.)
    init_deltas_for_overshoot_all_lanes(&mut hp, alpha, max_overhang);

    let mut text_profile: [_; LANES] = from_fn(|_| P::alloc_out());

    for i in 0..num_chunks {
        for lane in 0..lanes {
            let mut slice = [b'N'; 64];
            let block = texts[lane].get(64 * i..).unwrap_or_default();
            let block = block.get(..64).unwrap_or(block);
            slice[..block.len()].copy_from_slice(block);
            profiler.encode_ref(&slice, &mut text_profile[lane]);
        }
        let mut vp = S::splat(0);
        let mut vm = S::splat(0);
        for lane in 0..lanes {
            let v = V::from(vp.as_array()[lane], vm.as_array()[lane]);
            m[lane].deltas.push(v);
        }
        // FIXME: for large queries, use the SIMD within this single block, rather than spreading it thin over LANES 'matches' when there is only a single candidate match.
        for j in 0..pattern.len() {
            let eq = from_fn(|lane| P::eq(&pattern_profile[j], &text_profile[lane])).into();
            compute_block_simd(&mut hp[j], &mut hm[j], &mut vp, &mut vm, eq);
            for lane in 0..lanes {
                let v = V::from(vp.as_array()[lane], vm.as_array()[lane]);
                m[lane].deltas.push(v);
            }
        }
    }

    for lane in 0..lanes {
        assert_eq!(m[lane].deltas.len(), num_chunks * (m[lane].q + 1));
    }
}

// FIXME: DEDUP
pub fn simd_fill_multipattern<P: Profile>(
    patterns: &[&[u8]],
    texts: &[&[u8]],
    max_len: usize,
    m: &mut [CostMatrix; LANES],
    alpha: Option<f32>,
    max_overhang: Option<usize>,
) {
    assert!(texts.len() <= LANES);
    if alpha.is_some() && !P::supports_overhang() {
        panic!(
            "Overhang is not supported for {:?}",
            std::any::type_name::<P>()
        );
    }
    let lanes = texts.len();

    let (profiler, pattern_profiles) = P::encode_patterns(patterns);
    let pattern = &patterns[0];
    let num_chunks = max_len.div_ceil(64);

    log::debug!("max len {max_len} num_chunks {num_chunks}");

    for m in &mut *m {
        m.alpha = alpha;
        m.max_overhang = max_overhang;
        m.q = pattern.len();
        m.deltas.clear();
        m.deltas.reserve((m.q + 1) * num_chunks);
    }

    let mut hp: Vec<S> = Vec::with_capacity(pattern.len());
    let mut hm: Vec<S> = Vec::with_capacity(pattern.len());
    hp.resize(pattern.len(), S::splat(1));
    hm.resize(pattern.len(), S::splat(0));

    // NOTE: It's OK to always fill the left with 010101, even if it's not
    // actually the left of the text, because in that case the left column can't
    // be included in the alignment anyway. (The text has length q+k in that case.)
    init_deltas_for_overshoot_all_lanes(&mut hp, alpha, max_overhang);

    let mut text_profile: [_; LANES] = from_fn(|_| P::alloc_out());

    for i in 0..num_chunks {
        for lane in 0..lanes {
            let mut slice = [b'N'; 64];
            let block = texts[lane].get(64 * i..).unwrap_or_default();
            let block = block.get(..64).unwrap_or(block);
            slice[..block.len()].copy_from_slice(block);
            profiler.encode_ref(&slice, &mut text_profile[lane]);
        }
        let mut vp = S::splat(0);
        let mut vm = S::splat(0);
        for lane in 0..lanes {
            let v = V::from(vp.as_array()[lane], vm.as_array()[lane]);
            m[lane].deltas.push(v);
        }
        // FIXME: for large queries, use the SIMD within this single block, rather than spreading it thin over LANES 'matches' when there is only a single candidate match.
        for j in 0..pattern.len() {
            let eq = from_fn(|lane| P::eq(&pattern_profiles[j][lane], &text_profile[lane])).into();
            compute_block_simd(&mut hp[j], &mut hm[j], &mut vp, &mut vm, eq);
            for lane in 0..lanes {
                let v = V::from(vp.as_array()[lane], vm.as_array()[lane]);
                m[lane].deltas.push(v);
            }
        }
    }

    for lane in 0..lanes {
        assert_eq!(m[lane].deltas.len(), num_chunks * (m[lane].q + 1));
    }
}

/*
Op  BAM  Consumes query  Consumes reference
--  ---  --------------  ------------------
M   0    yes             yes
I   1    yes             no
D   2    no              yes
N   3    no              yes
S   4    yes             no
H   5    no              no
P   6    no              no
=   7    yes             yes
X   8    yes             yes

NOTE: query = pattern, reference = text.
In this context mostly means:
    i+1 , j+1  -> Match/Sub
    i+1 , j    -> Ins
    i   , j+1  -> Del
*/
pub fn get_trace<P: Profile>(
    pattern: &[u8],
    text_offset: usize,
    end_pos: usize,
    text: &[u8],
    m: &impl CostLookup,
    alpha: Option<f32>,
    max_overhang: Option<usize>,
) -> Match {
    let mut trace = Vec::new();
    let mut j = pattern.len();
    let mut i = end_pos - text_offset;

    let cost = |j: usize, i: usize| -> Cost { m.get(i, j) };

    log::debug!("Trace ({j}, {i}) end pos {end_pos} offset {text_offset}");
    // remaining dist to (i,j)
    let mut g = cost(j, i);
    let mut total_cost = g;
    log::debug!("Initial cost at ({j}, {i}) is {g}");

    let mut cigar = Cigar::default();

    let mut pattern_start = 0;
    let mut pattern_end = pattern.len();

    // Overshoot at end.
    if i > text.len() {
        let overshoot = i - text.len();
        pattern_end -= overshoot;
        let overshoot_cost = (overshoot as f32 * alpha.unwrap()).floor() as Cost;

        total_cost += overshoot_cost;
        i -= overshoot;
        j -= overshoot;
        log::debug!("Trace from ({j}, {i}) for total cost {total_cost}");
        log::debug!("Right overshoot {overshoot} for cost {overshoot_cost}");
    } else {
        log::debug!("Trace from ({j}, {i}) for total cost {total_cost}");
    }

    loop {
        // log::debug!("({i}, {j}) {g}");
        trace.push((j, text_offset + i));

        if j == 0 {
            break;
        }

        if i == 0
            && let Some(alpha) = alpha
        {
            pattern_start = j;
            // Overshoot at start.
            g -= overhang_cost(j, alpha, max_overhang);
            break;
        }

        // Match
        if i > 0 && cost(j - 1, i - 1) == g && P::is_match(pattern[j - 1], text[i - 1]) {
            cigar.push(pa_types::CigarOp::Match);
            j -= 1;
            i -= 1;
            continue;
        }
        // We make some kind of mutation.
        g -= 1;

        // Mismatch.
        if i > 0 && cost(j - 1, i - 1) == g {
            cigar.push(pa_types::CigarOp::Sub);
            j -= 1;
            i -= 1;
            continue;
        }
        // Consumes i = text/ref (not j) = Del
        if i > 0 && cost(j, i - 1) == g {
            cigar.push(pa_types::CigarOp::Del);
            i -= 1;
            continue;
        }
        // Consumes j = query/pattern (not i) = Ins
        if cost(j - 1, i) == g {
            cigar.push(pa_types::CigarOp::Ins);
            j -= 1;
            continue;
        }

        if !P::valid_seq(&[pattern[j - 1]]) {
            panic!(
                "Trace failed, because the query contains non-{:?} character {} at position {}. (Use `profiles::Iupac` instead of `profiles::Dna`.)",
                std::any::type_name::<P>(),
                pattern[j - 1] as char,
                j - 1
            );
        }
        if !P::valid_seq(&[text[i - 1]]) {
            panic!(
                "Trace failed, because the text contains non-{:?} character {} at position {}. (Use `profiles::Iupac` instead of `profiles::Dna`.)",
                std::any::type_name::<P>(),
                text[i - 1] as char,
                i - 1
            );
        }

        panic!(
            "Trace failed! No ancestor found of {j} {i} at distance {}",
            g + 1
        );
    }

    assert_eq!(g, 0, "Remaining cost after the trace must be 0.");

    // Reverse the cigar, because the trace goes from end to start.
    cigar.reverse();

    Match {
        pattern_idx: 0,
        text_idx: 0,
        cost: total_cost,
        text_start: text_offset + i,
        text_end: text_offset + text.len(),
        pattern_start,
        pattern_end,
        strand: Strand::Fwd,
        cigar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::{Dna, Iupac};

    #[test]
    fn test_traceback() {
        let query = b"ATTTTCCCGGGGATTTT".as_slice();
        let text2: &[u8] = b"ATTTTGGGGATTTT".as_slice();

        let mut cost_matrix = Default::default();
        fill::<Dna>(query, text2, text2.len(), &mut cost_matrix, None, None);

        let trace = get_trace::<Dna>(query, 0, text2.len(), text2, &cost_matrix, None, None);
        println!("Trace: {:?}", trace);
    }

    #[test]
    fn test_all_alignments_simple() {
        // Perfect match: exactly one alignment.
        // "ACGT" vs "ACGT" at k=0, text_start=0, text_end=4, cost=0, cigar=4=
        let pattern = b"ACGT".as_slice();
        let text = b"ACGT".as_slice();

        let group = all_alignments_at_position::<Dna>(
            pattern,
            text,
            0,
            text.len(),
            0,
            None,
            None,
            Strand::Fwd,
            None,
            0,
        );
        let all: Vec<_> = group.collect();

        assert_eq!(all.len(), 1, "expected exactly one alignment");
        assert_eq!(all[0].text_start, 0);
        assert_eq!(all[0].text_end, 4);
        assert_eq!(all[0].cost, 0);
        assert_eq!(all[0].cigar.to_string(), "4=");
    }

    #[test]
    fn test_all_alignments_with_errors() {
        // "ACGT" vs "AACGT" at k=1.
        // The minimum cost at text_end=5 is 0: text[1:5]="ACGT" is an exact match.
        // There is exactly one alignment: text_start=1, cigar=4=, cost=0.
        let pattern = b"ACGT".as_slice();
        let text = b"AACGT".as_slice();

        let group = all_alignments_at_position::<Dna>(
            pattern,
            text,
            0,
            text.len(),
            1,
            None,
            None,
            Strand::Fwd,
            None,
            0,
        );
        let all: Vec<_> = group.collect();

        assert_eq!(all.len(), 1, "expected exactly one alignment");
        assert_eq!(all[0].text_start, 1);
        assert_eq!(all[0].text_end, 5);
        assert_eq!(all[0].cost, 0);
        assert_eq!(all[0].cigar.to_string(), "4=");
    }

    #[test]
    fn test_all_alignments_multiple_paths() {
        // "AT" vs "ACT" at k=1. All three cost-1 alignments land at text_end=3:
        //  Del  : text[0:3]="ACT", cigar "1=1D1="
        //  Sub  : text[1:3]="CT",  cigar "1X1="
        //  Ins  : text[2:3]="T",   cigar "1I1="
        let pattern = b"AT".as_slice();
        let text = b"ACT".as_slice();

        let group = all_alignments_at_position::<Dna>(
            pattern,
            text,
            0,
            text.len(),
            1,
            None,
            None,
            Strand::Fwd,
            None,
            0,
        );
        let mut all: Vec<_> = group.collect();
        // Sort by text_start for deterministic comparison.
        all.sort_by_key(|m| m.text_start);

        assert_eq!(all.len(), 3, "expected exactly three alignments");

        assert_eq!(all[0].text_start, 0);
        assert_eq!(all[0].text_end, 3);
        assert_eq!(all[0].cost, 1);
        assert_eq!(all[0].cigar.to_string(), "1=1D1=");

        assert_eq!(all[1].text_start, 1);
        assert_eq!(all[1].text_end, 3);
        assert_eq!(all[1].cost, 1);
        assert_eq!(all[1].cigar.to_string(), "1X1=");

        assert_eq!(all[2].text_start, 2);
        assert_eq!(all[2].text_end, 3);
        assert_eq!(all[2].cost, 1);
        assert_eq!(all[2].cigar.to_string(), "1I1=");
    }

    #[test]
    fn test_margin_zero_matches_optimal() {
        // "AT" vs "ACT", k=1, margin=0.
        // The three cost-1 alignments at text_end=3 in DFS order
        // (Match subtree first, then Del, then Ins at the root level):
        //   1X1=   : sub A>C, match T=T         (text_start=1)
        //   1=1D1= : match A=A, del C, match T=T (text_start=0)
        //   1I1=   : insert A, match T=T          (text_start=2)
        let pattern = b"AT".as_slice();
        let text = b"ACT".as_slice();

        let group = all_alignments_at_position::<Dna>(
            pattern,
            text,
            0,
            text.len(),
            1,
            None,
            None,
            Strand::Fwd,
            None,
            0,
        );
        let all: Vec<_> = group.collect();

        assert_eq!(all.len(), 3);
        assert_eq!(all[0].text_start, 1);
        assert_eq!(all[0].text_end, 3);
        assert_eq!(all[0].cost, 1);
        assert_eq!(all[0].cigar.to_string(), "1X1=");
        assert_eq!(all[1].text_start, 0);
        assert_eq!(all[1].text_end, 3);
        assert_eq!(all[1].cost, 1);
        assert_eq!(all[1].cigar.to_string(), "1=1D1=");
        assert_eq!(all[2].text_start, 2);
        assert_eq!(all[2].text_end, 3);
        assert_eq!(all[2].cost, 1);
        assert_eq!(all[2].cigar.to_string(), "1I1=");
    }

    #[test]
    fn test_margin_one_yields_suboptimal() {
        // "AT" vs "ACT", k=2, margin=1.
        // Optimal cost is 1; budget = min(1+1, 2) = 2.
        // DFS yields the 3 cost-1 alignments first, then 4 additional cost-2
        // alignments reachable within the extra budget:
        //   cost=1: 1X1= (start=1), 1=1D1= (start=0), 1I1= (start=2)
        //   cost=2: 1I1D1= (start=1), 1=1X1D (start=0), 1X1I (start=2), 2I (start=3)
        let pattern = b"AT".as_slice();
        let text = b"ACT".as_slice();

        let group = all_alignments_at_position::<Dna>(
            pattern,
            text,
            0,
            text.len(),
            2,
            None,
            None,
            Strand::Fwd,
            None,
            1,
        );
        let all: Vec<_> = group.collect();

        // The DFS interleaves cost-1 and cost-2 results as branches open up;
        // they are not grouped by cost. Order mirrors the Match→Sub→Del→Ins
        // traversal and the extra branches the larger budget unlocks.
        assert_eq!(all.len(), 7);
        assert_eq!(all[0].text_start, 1);
        assert_eq!(all[0].cost, 1);
        assert_eq!(all[0].cigar.to_string(), "1X1=");
        assert_eq!(all[1].text_start, 0);
        assert_eq!(all[1].cost, 1);
        assert_eq!(all[1].cigar.to_string(), "1=1D1=");
        assert_eq!(all[2].text_start, 1);
        assert_eq!(all[2].cost, 2);
        assert_eq!(all[2].cigar.to_string(), "1I1D1=");
        assert_eq!(all[3].text_start, 2);
        assert_eq!(all[3].cost, 1);
        assert_eq!(all[3].cigar.to_string(), "1I1=");
        assert_eq!(all[4].text_start, 0);
        assert_eq!(all[4].cost, 2);
        assert_eq!(all[4].cigar.to_string(), "1=1X1D");
        assert_eq!(all[5].text_start, 2);
        assert_eq!(all[5].cost, 2);
        assert_eq!(all[5].cigar.to_string(), "1X1I");
        assert_eq!(all[6].text_start, 3);
        assert_eq!(all[6].cost, 2);
        assert_eq!(all[6].cigar.to_string(), "2I");
    }

    #[test]
    fn test_margin_clamps_to_k() {
        // k=1, margin=99: budget is clamped to k=1, so the result is exactly
        // the same 3 alignments in the same order as margin=0.
        let pattern = b"AT".as_slice();
        let text = b"ACT".as_slice();

        let group = all_alignments_at_position::<Dna>(
            pattern,
            text,
            0,
            text.len(),
            1,
            None,
            None,
            Strand::Fwd,
            None,
            99,
        );
        let all: Vec<_> = group.collect();

        assert_eq!(all.len(), 3);
        assert_eq!(all[0].text_start, 1);
        assert_eq!(all[0].cost, 1);
        assert_eq!(all[0].cigar.to_string(), "1X1=");
        assert_eq!(all[1].text_start, 0);
        assert_eq!(all[1].cost, 1);
        assert_eq!(all[1].cigar.to_string(), "1=1D1=");
        assert_eq!(all[2].text_start, 2);
        assert_eq!(all[2].cost, 1);
        assert_eq!(all[2].cigar.to_string(), "1I1=");
    }

    #[test]
    fn test_all_alignments_left_overshoot() {
        // Pattern "ACGT" aligned to text "GT" with alpha=0.0 (left overhang free, k=0).
        // The only valid alignment matches pattern[2:4]="GT" to text[0:2]="GT", with
        // pattern[0:2]="AC" hanging off the left at zero cost.
        // Expected: one alignment, pattern_start=2, text_start=0, text_end=2, cost=0, cigar="2=".
        let pattern = b"ACGT".as_slice();
        let text = b"GT".as_slice();

        let mut group = all_alignments_at_position::<Iupac>(
            pattern,
            text,
            0,
            text.len(),
            0,
            Some(0.0),
            None,
            Strand::Fwd,
            None,
            0,
        );

        let m = group.next().expect("expected an alignment");
        assert_eq!(group.next(), None, "expected exactly one alignment");
        assert_eq!(m.pattern_start, 2);
        assert_eq!(m.pattern_end, 4);
        assert_eq!(m.text_start, 0);
        assert_eq!(m.text_end, 2);
        assert_eq!(m.cost, 0);
        assert_eq!(m.cigar.to_string(), "2=");
    }

    #[test]
    fn test_optimal_cost_getter() {
        let pattern = b"ACGT".as_slice();
        let text = b"ACGT".as_slice();

        let group = all_alignments_at_position::<Dna>(
            pattern,
            text,
            0,
            text.len(),
            0,
            None,
            None,
            Strand::Fwd,
            None,
            0,
        );
        assert_eq!(group.optimal_cost(), 0);
    }

    #[test]
    fn test_traceback_simd() {
        let query = b"ATTTTCCCGGGGATTTT".as_slice();
        let text1 = b"ATTTTCCCGGGGATTTT".as_slice();
        let text2 = b"ATTTTGGGGATTTT".as_slice();
        let text3 = b"TGGGGATTTT".as_slice();
        let text4 = b"TTTTTTTTTTATTTTGGGGATTTT".as_slice();

        let mut cost_matrix = Default::default();
        simd_fill::<Dna>(
            query,
            &[text1, text2, text3, text4],
            text4.len(),
            &mut cost_matrix,
            None,
            None,
        );
        let _trace = get_trace::<Dna>(query, 0, text1.len(), text1, &cost_matrix[0], None, None);
        let _trace = get_trace::<Dna>(query, 0, text2.len(), text2, &cost_matrix[1], None, None);
        let _trace = get_trace::<Dna>(query, 0, text3.len(), text3, &cost_matrix[2], None, None);
        let trace = get_trace::<Dna>(query, 0, text4.len(), text4, &cost_matrix[3], None, None);
        println!("Trace: {:?}", trace);
    }
}
// let text1 = b"ATCGACTAGC".as_slice();

// let text3 = b"CTAGC".as_slice();
// let text4 = b"TGGC".as_slice();

// let col_costs = fill(query, text2);
// for c in col_costs {
//     println!("col: {:?}", c);
//     let (p, m) = c.deltas[0].pm();
//     println!("p: {:064b} \nm: {:064b}\n\n", p, m);
// }

// let col_costs = simd_fill::<Iupac>(&query, [&text2, &text2, &text2, &text2]);
// let c = &col_costs[0];
// for col in c {
//     let (p, m) = col.deltas[0].pm();
//     // println!("(p, m): {:?}", (p, m));
//     // print the binary 0100101
//     println!("p: {:064b} \nm: {:064b}\n\n", p, m);
// }

// // Simd
// let col_costs = simd_fill::<Dna>(&query, [&text2, &text2, &text2, &text2]);

// for lane in 0..LANES {
//     //println!("\nCol costs for lane {}\n{:?}", lane, col_costs[lane]);
//     let trace = get_trace(&col_costs[lane]);
//     println!("Trace {}: {:?}", lane, trace);
// }

// #[test]
// fn test_and_block_boundary() {
//     let query = b"ACGTGGA";
//     let mut text = [b'G'; 128];
//     text[64 - 3..64 + 4].copy_from_slice(query);
//     let col_costs = fill(query, &text[..64 + 4]);
//     let trace = get_trace(&col_costs);
//     println!("Trace 1: {:?}", trace); // FIXME: This is wrong when crossing block boundary
// }
/*
query:   ATTTTCCCGGGGATTTT
text2: ...GGATTTTCCGGATTTT
 */
