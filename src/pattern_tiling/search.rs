use crate::pattern_tiling::backend::SimdBackend;
use crate::pattern_tiling::tqueries::TQueries;
use crate::pattern_tiling::trace::PatternHistory;
use crate::profiles::Profile;
use crate::search::get_overhang_steps;
use std::marker::PhantomData;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_prefetch;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch_read<T>(ptr: *const T) {
    unsafe { _mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0) };
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn prefetch_read<T>(_ptr: *const T) {}

#[derive(Clone, Copy, Default)]
pub(crate) struct BlockState<S: Copy> {
    pub(crate) vp: S,
    pub(crate) vn: S,
    pub(crate) cost: S,
    pub(crate) active_mask: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HitRange {
    pub pattern_idx: usize,
    pub start: isize,
    pub end: isize,
}

pub struct Myers<B: SimdBackend, P: Profile> {
    pub(crate) blocks: Vec<BlockState<B::Simd>>,
    pub(crate) positions: Vec<Vec<usize>>,
    pub(crate) alpha_pattern: u64,
    pub(crate) alpha: f32,
    pub(crate) history: Vec<PatternHistory<B::Simd>>,
    pub(crate) hit_ranges: Vec<HitRange>,
    // Per-pattern open-range start position; isize::MIN means no active range
    pub(crate) max_overhang: Option<usize>,
    pub(crate) active_ranges: Vec<isize>,
    _marker: PhantomData<P>,
}

impl<B: SimdBackend, P: Profile> Default for Myers<B, P> {
    fn default() -> Self {
        Self::new(None)
    }
}

impl<B: SimdBackend, P: Profile> Myers<B, P> {
    pub fn new(alpha: Option<f32>) -> Self {
        let alpha_val = alpha.unwrap_or(1.0);
        Self {
            blocks: Vec::new(),
            positions: Vec::new(),
            alpha_pattern: Self::generate_alpha_mask(alpha_val, 64, None),
            alpha: alpha_val,
            history: Vec::new(),
            hit_ranges: Vec::new(),
            active_ranges: Vec::new(),
            max_overhang: None,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) fn ensure_capacity(&mut self, num_blocks: usize, total_queries: usize) {
        if self.blocks.len() < num_blocks {
            let all_ones = B::splat_all_ones();
            let zero = B::splat_zero();
            self.blocks.resize(
                num_blocks,
                BlockState {
                    vp: all_ones,
                    vn: zero,
                    cost: zero,
                    active_mask: 0,
                },
            );
        }
        if self.positions.len() < total_queries {
            self.positions.resize(total_queries, Vec::new());
        }
        if self.history.len() < total_queries {
            self.history
                .resize_with(total_queries, PatternHistory::default);
        }
        if self.active_ranges.len() < total_queries {
            self.active_ranges.resize(total_queries, isize::MIN);
        }
    }

    #[inline(always)]
    pub(crate) fn search_prep(
        &mut self,
        num_blocks: usize,
        n_queries: usize,
        pattern_length: usize,
        alpha_pattern: u64,
    ) {
        let length_mask = (!0u64) >> (64usize.saturating_sub(pattern_length));
        let masked_alpha = alpha_pattern & length_mask;
        let initial_cost = B::splat_scalar(B::scalar_from_i64(masked_alpha.count_ones() as i64));
        let alpha_simd = B::splat_scalar(B::mask_word_to_scalar(alpha_pattern));
        let zero = B::splat_zero();

        for block in self.blocks.iter_mut().take(num_blocks) {
            block.vp = alpha_simd;
            block.vn = zero;
            block.cost = initial_cost;
            block.active_mask = 0;
        }

        self.active_ranges[..n_queries].fill(isize::MIN);
        self.hit_ranges.clear();
    }

    #[inline(always)]
    fn finalize_ranges(&mut self, n_queries: usize, num_blocks: usize, final_pos: isize) {
        for block_i in 0..num_blocks {
            let block = &mut self.blocks[block_i];
            let mut mask = std::mem::replace(&mut block.active_mask, 0);

            if mask == 0 {
                continue;
            }

            let base_idx = block_i * B::LANES;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                let qidx = base_idx + lane;

                if qidx < n_queries {
                    let start = self.active_ranges[qidx];
                    self.hit_ranges.push(HitRange {
                        pattern_idx: qidx,
                        start,
                        end: final_pos,
                    });
                }
                mask &= mask - 1;
            }
        }
    }

    #[inline(always)]
    pub(crate) fn myers_step(
        vp: B::Simd,
        vn: B::Simd,
        cost: B::Simd,
        eq: B::Simd,
        all_ones: B::Simd,
        last_bit_shift: u32,
        last_bit_mask: B::Simd,
    ) -> (B::Simd, B::Simd, B::Simd) {
        let eq_and_pv = eq & vp;
        let xh = ((eq_and_pv + vp) ^ vp) | eq;
        let mh = vp & xh;
        let ph = vn | (all_ones ^ (xh | vp));

        let ph_shifted = ph << 1;
        let mh_shifted = mh << 1;

        let xv = eq | vn;
        let vp_out = mh_shifted | (all_ones ^ (xv | ph_shifted));
        let vn_out = ph_shifted & xv;

        let ph_bit = (ph & last_bit_mask) >> last_bit_shift;
        let mh_bit = (mh & last_bit_mask) >> last_bit_shift;

        let cost_out = (cost + ph_bit) - mh_bit;

        (vp_out, vn_out, cost_out)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn handle_range_mask_change(
        &mut self,
        block: &mut BlockState<B::Simd>,
        change: u64,
        prev_mask: u64,
        hit_mask: u64,
        base_pattern_idx: usize,
        current_pos: isize,
        n_queries: usize,
        start_ptr: *mut isize,
    ) {
        if change == 0 {
            return;
        }

        let mut leaving = change & prev_mask;
        while leaving != 0 {
            let lane = leaving.trailing_zeros() as usize;
            let qidx = base_pattern_idx + lane;
            if qidx < n_queries {
                let start = unsafe { *start_ptr.add(qidx) };
                self.hit_ranges.push(HitRange {
                    pattern_idx: qidx,
                    start,
                    end: current_pos - 1,
                });
            }
            leaving &= leaving - 1;
        }

        let mut entering = change & hit_mask;
        while entering != 0 {
            let lane = entering.trailing_zeros() as usize;
            let qidx = base_pattern_idx + lane;
            if qidx < n_queries {
                unsafe { *start_ptr.add(qidx) = current_pos };
            }
            entering &= entering - 1;
        }

        block.active_mask = hit_mask;
    }

    #[inline(always)]
    fn handle_suffix_overhang(
        &mut self,
        t_queries: &TQueries<B, P>,
        text: &[u8],
        k: u32,
        is_end_block: bool,
    ) {
        if self.alpha_pattern == !0 || !is_end_block {
            return;
        }

        let num_blocks = t_queries.n_simd_blocks;
        let eq: <B as SimdBackend>::Simd = B::splat_all_ones();

        let steps_needed = get_overhang_steps(
            t_queries.pattern_length,
            k as usize,
            self.alpha,
            self.max_overhang,
        );

        let blocks_ptr = self.blocks.as_mut_ptr();
        let mut current_text_pos = text.len();
        let all_ones = B::splat_all_ones();
        let last_bit_shift = (t_queries.pattern_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        let start_ptr = self.active_ranges.as_mut_ptr();

        let n_queries = t_queries.n_queries;

        for i in 0..steps_needed {
            for block_i in 0..num_blocks {
                unsafe {
                    let block = &mut *blocks_ptr.add(block_i);
                    let (vp_out, vn_out, cost_out) = Self::myers_step(
                        block.vp,
                        block.vn,
                        block.cost,
                        eq,
                        all_ones,
                        last_bit_shift,
                        last_bit_mask,
                    );
                    block.vp = vp_out;
                    block.vn = vn_out;
                    block.cost = cost_out;

                    let mut hit_mask: u64 = 0;

                    let cost_arr = B::to_array(cost_out);
                    for (lane_idx, cost) in cost_arr.as_ref().iter().enumerate() {
                        let c_val = B::scalar_to_u64(*cost);
                        let new_cost = c_val as f32 + (self.alpha * (i + 1) as f32);
                        if (new_cost.floor() as u32) <= k {
                            hit_mask |= 1u64 << lane_idx;
                        }
                    }

                    let prev_mask = block.active_mask;
                    let change = hit_mask ^ prev_mask;
                    let current_pos = current_text_pos as isize;
                    let base_pattern_idx = block_i * B::LANES;
                    self.handle_range_mask_change(
                        block,
                        change,
                        prev_mask,
                        hit_mask,
                        base_pattern_idx,
                        current_pos,
                        n_queries,
                        start_ptr,
                    );
                }
            }
            current_text_pos += 1;
        }
    }

    #[inline(always)]
    fn handle_full_prefix_matches(&mut self, t_queries: &TQueries<B, P>, k: u32) {
        let mask = if t_queries.pattern_length >= 64 {
            !0u64
        } else {
            (1u64 << t_queries.pattern_length) - 1
        };

        if self.alpha_pattern == !0 || (self.alpha_pattern & mask).count_ones() > k {
            return;
        }

        let n_queries = t_queries.n_queries;
        self.active_ranges[..n_queries].fill(-1);

        for block_i in 0..t_queries.n_simd_blocks {
            let valid_lanes = (n_queries - block_i * B::LANES).min(B::LANES);
            let mask_bits = (1u64 << valid_lanes) - 1;
            self.blocks[block_i].active_mask = mask_bits;
        }
    }

    #[inline(always)]
    pub fn search_ranges(
        &mut self,
        t_queries: &TQueries<B, P>,
        text: &[u8],
        k: u32,
        is_start_block: bool,
        is_end_block: bool,
    ) -> &[HitRange] {
        let num_blocks = t_queries.n_simd_blocks;
        let length_mask = (!0u64) >> (64usize.saturating_sub(t_queries.pattern_length));
        let alpha_pattern = self.alpha_pattern & length_mask;

        self.ensure_capacity(num_blocks, t_queries.n_queries);

        self.search_prep(
            num_blocks,
            t_queries.n_queries,
            t_queries.pattern_length,
            alpha_pattern,
        );
        self.hit_ranges.reserve(t_queries.n_queries);

        let k_simd = B::splat_scalar(B::scalar_from_i64(k as i64));
        let last_bit_shift = (t_queries.pattern_length - 1) as u32;
        let last_bit_mask = B::splat_one() << last_bit_shift;
        let all_ones = B::splat_all_ones();

        let peqs_ptr: *const <B as SimdBackend>::Simd = t_queries.peqs.as_ptr();
        let blocks_ptr = self.blocks.as_mut_ptr();
        let n_queries = t_queries.n_queries;
        let start_ptr = self.active_ranges.as_mut_ptr();

        let text_ptr = text.as_ptr();
        let text_len = text.len();

        if self.alpha_pattern != !0 && is_start_block {
            self.handle_full_prefix_matches(t_queries, k);
        }

        for idx in 0..text_len {
            let c = unsafe { *text_ptr.add(idx) };
            let encoded = P::encode_char(c) as usize;
            let peq_base = unsafe { peqs_ptr.add(encoded * num_blocks) };

            // Prefetch next peq
            if idx + 1 < text_len {
                let next_c = unsafe { *text_ptr.add(idx + 1) };
                let next_encoded = P::encode_char(next_c) as usize;
                unsafe { prefetch_read(peqs_ptr.add(next_encoded * num_blocks)) };
            }

            for block_i in 0..num_blocks {
                unsafe {
                    let block = &mut *blocks_ptr.add(block_i);

                    let prev_mask = block.active_mask;

                    let eq = *peq_base.add(block_i);

                    let (vp_out, vn_out, cost_out) = Self::myers_step(
                        block.vp,
                        block.vn,
                        block.cost,
                        eq,
                        all_ones,
                        last_bit_shift,
                        last_bit_mask,
                    );
                    block.vp = vp_out;
                    block.vn = vn_out;
                    block.cost = cost_out;

                    // Compute hits, <=k cost
                    let gt_mask = B::simd_gt(cost_out, k_simd);
                    let hit_bits = if gt_mask != all_ones {
                        B::lanes_with_zero(gt_mask)
                    } else {
                        0
                    };

                    let change = hit_bits ^ prev_mask;

                    let current_pos = idx as isize;
                    let base_pattern_idx = block_i * B::LANES;

                    block.active_mask = hit_bits;

                    self.handle_range_mask_change(
                        block,
                        change,
                        prev_mask,
                        hit_bits,
                        base_pattern_idx,
                        current_pos,
                        n_queries,
                        start_ptr,
                    );
                }
            }
        }

        // Handle suffix overhang
        self.handle_suffix_overhang(t_queries, text, k, is_end_block);

        //fixme: fix final pos based on alpha
        let final_pos = if self.alpha_pattern != !0 && is_end_block {
            (text_len + t_queries.pattern_length).saturating_sub(1) as isize
        } else if text_len == 0 {
            -1
        } else {
            (text_len - 1) as isize
        };

        self.finalize_ranges(t_queries.n_queries, num_blocks, final_pos);

        &self.hit_ranges
    }

    // Similar to sassy
    #[inline(always)]
    fn generate_alpha_mask(alpha: f32, length: usize, max_overhang: Option<usize>) -> u64 {
        let mut mask = 0u64;
        let limit = length.min(max_overhang.unwrap_or(usize::MAX));
        for i in 0..limit {
            let val = ((i + 1) as f32 * alpha).floor() as u64 - (i as f32 * alpha).floor() as u64;
            if val >= 1 {
                mask |= 1 << i;
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use rand::random_range;

    use super::*;
    use crate::Searcher as SassySearcher;
    use crate::pattern_tiling::backend::SimdBackend;
    use crate::pattern_tiling::general::Searcher;
    use crate::pattern_tiling::trace::{TracePostProcess, trace_batch_ranges};
    use crate::profiles::Iupac;
    use crate::profiles::Profile;
    use crate::search::Match;
    use std::collections::{HashMap, HashSet};

    #[cfg(test)]
    type TestBackend = crate::pattern_tiling::backend::U64;

    const ITER: usize = 1_000_000;

    fn trace_all_for_test<B: SimdBackend>(
        searcher: &mut Myers<B, Iupac>,
        t_queries: &TQueries<B, Iupac>,
        text: &[u8],
        k: u32,
        post: TracePostProcess,
    ) -> Vec<Match> {
        let ranges: Vec<HitRange> = searcher
            .search_ranges(t_queries, text, k, true, true)
            .to_vec();
        let mut out = Vec::new();

        for r in ranges.iter() {
            println!("Range: {} {} {}", r.start, r.end, r.pattern_idx);
        }
        trace_batch_ranges(
            searcher, t_queries, text, &ranges, k, post, &mut out, None, None,
        );
        out
    }

    #[test]
    fn test_trace_all_hits_integration() {
        let mut searcher = Myers::<TestBackend, Iupac>::new(None);

        let queries = vec![b"ACGT".to_vec(), b"TGCA".to_vec()];
        let t_queries = TQueries::<TestBackend, Iupac>::new(&queries, false);
        let text = b"AAACGTTTGCAAA";
        //                           ^    ^
        //                      0123456789-12
        let k = 0;

        let alignments =
            trace_all_for_test(&mut searcher, &t_queries, text, k, TracePostProcess::All);

        for a in alignments.iter() {
            println!(
                "Alignment: {} {} {:?} {}",
                a.text_start, a.text_end, a.strand, a.cost
            );
        }

        assert_eq!(alignments.len(), 2, "Should find and trace 2 matches");

        let aln0 = alignments.iter().find(|a| a.pattern_idx == 0).unwrap();
        assert_eq!(aln0.cost, 0);
        assert_eq!(aln0.text_start, 2);
        assert_eq!(aln0.text_end, 6);

        let aln1 = alignments.iter().find(|a| a.pattern_idx == 1).unwrap();
        assert_eq!(aln1.cost, 0);
        assert_eq!(aln1.text_start, 7);
        assert_eq!(aln1.text_end, 11);
    }

    #[test]
    fn test_alpha_overhang() {
        let mut searcher = Myers::<TestBackend, Iupac>::new(Some(0.5));

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend, Iupac>::new(&queries, false);

        let text = b"AC";
        let k = 2;

        let hits = searcher.search_ranges(&t_queries, text, k, false, false);

        assert!(!hits.is_empty(), "Should find match with suffix overhang");
    }

    #[test]
    fn test_prefix_overhang() {
        let mut searcher = Myers::<TestBackend, Iupac>::new(Some(0.5));

        let queries = vec![b"AAAGT".to_vec()];
        let t_queries = TQueries::<TestBackend, Iupac>::new(&queries, false);
        let text = b"GTCCCCCCCCC";
        let k = 2;

        let hits = trace_all_for_test(&mut searcher, &t_queries, text, k, TracePostProcess::All);
        assert!(!hits.is_empty(), "Should find match with prefix overhang");
    }

    #[test]
    fn test_no_matches() {
        let mut searcher = Myers::<TestBackend, Iupac>::new(None);

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend, Iupac>::new(&queries, false);
        let text = b"TTTTTTTT";
        let k = 1;

        let hits = searcher.search_ranges(&t_queries, text, k, false, false);

        assert_eq!(hits.len(), 0, "Should find no matches");
    }

    #[test]
    fn test_range() {
        let mut searcher = Myers::<TestBackend, Iupac>::new(None);

        let queries = vec![b"ACGT".to_vec()];
        let t_queries = TQueries::<TestBackend, Iupac>::new(&queries, false);
        //                       ACGT
        let text = b"AAACGTAAA";
        //                     012345678
        let k = 1;

        let hits = searcher.search_ranges(&t_queries, text, k, false, false);
        for h in hits {
            println!("Hit: {:?}", h);
        }
        assert_eq!(hits[0].start, 4);
        assert_eq!(hits[0].end, 6);
    }

    #[test]
    fn test_batch_size_edge_case() {
        let mut searcher = Myers::<TestBackend, Iupac>::new(None);

        let n_queries = TestBackend::LANES;
        let queries: Vec<Vec<u8>> = (0..n_queries)
            .map(|i| vec![b'A', b'C', b'G', b'T'][i % 4])
            .map(|c| vec![c; 4])
            .collect();

        let t_queries = TQueries::<TestBackend, Iupac>::new(&queries, false);
        let text = b"AAAACCCCGGGGTTTT";
        let k = 2;

        let alignments =
            trace_all_for_test(&mut searcher, &t_queries, text, k, TracePostProcess::All);

        assert!(!alignments.is_empty(), "Should find some matches");
    }

    #[allow(dead_code)]
    fn apply_edits(seq: &[u8], k: u32) -> Vec<u8> {
        let mut res = seq.to_vec();
        const BASES: &[u8] = b"ACGT";

        for _ in 0..k {
            let op = if res.len() < 2 { 1 } else { random_range(0..3) };

            match op {
                0 => {
                    let idx = random_range(0..res.len());
                    let old_base = res[idx];
                    let mut new_base = BASES[random_range(0..4)];
                    while new_base == old_base {
                        new_base = BASES[random_range(0..4)];
                    }
                    res[idx] = new_base;
                }
                1 => {
                    let idx = random_range(0..=res.len());
                    let new_base = BASES[random_range(0..4)];
                    res.insert(idx, new_base);
                }
                2 => {
                    let idx = random_range(0..res.len());
                    res.remove(idx);
                }
                _ => unreachable!(),
            }
        }
        res
    }

    #[allow(dead_code)]
    fn random_dna_seq(l: usize) -> Vec<u8> {
        use rand::random_range;
        const DNA: &[u8; 4] = b"ACGT";
        let mut dna = Vec::with_capacity(l);
        for _ in 0..l {
            let idx = random_range(0..4);
            dna.push(DNA[idx]);
        }
        dna
    }

    fn fuzz_against_sassy_batch(
        alpha: Option<f32>,
        include_rc: bool,
        use_hierarchical: bool,
        filter_local_pattern_tilingma: bool,
    ) {
        let mut sassy_searcher = if let Some(a) = alpha {
            SassySearcher::<Iupac>::new_fwd_with_overhang(a)
        } else {
            SassySearcher::<Iupac>::new_fwd()
        };
        let mut pattern_tiling_searcher = Searcher::<Iupac>::new(alpha);

        for _ in 0..ITER {
            let k = random_range(0..4);
            let q_len = random_range(5..60);
            let text_len = random_range(10..60);
            let batch_size = random_range(1..=25);

            // Test in detail these?
            // if text_len < q_len || (k as f32 * 0.5).floor() as usize >= q_len {
            //     continue;
            // }

            let mut text = random_dna_seq(text_len);
            let queries: Vec<Vec<u8>> = (0..batch_size).map(|_| random_dna_seq(q_len)).collect();

            if let Some(pattern) = queries.first() {
                let mutated = apply_edits(pattern, k / 2);
                let text_end = text.len().saturating_sub(mutated.len());
                let prefix = &mutated[..mutated.len() / 2];
                // Make sure we don't splice beyond text length
                let splice_end = (text_end + prefix.len()).min(text.len());
                text.splice(
                    text_end..splice_end,
                    prefix[..splice_end - text_end].iter().copied(),
                );
            }

            let pattern_tiling_encoded = pattern_tiling_searcher.encode(&queries, include_rc);
            let matches = pattern_tiling_searcher.search_with_options(
                &pattern_tiling_encoded,
                &text,
                k as u32,
                if use_hierarchical { Some(true) } else { None },
                if filter_local_pattern_tilingma {
                    crate::pattern_tiling::trace::TracePostProcess::LocalMinima
                } else {
                    crate::pattern_tiling::trace::TracePostProcess::All
                },
            );

            let mut pattern_tiling_map: HashMap<usize, Vec<Match>> = HashMap::new();
            for m in matches {
                pattern_tiling_map
                    .entry(m.pattern_idx)
                    .or_default()
                    .push(m.clone());
            }
            pattern_tiling_map.values_mut().for_each(|matches| {
                matches.sort_unstable_by_key(|m| {
                    (
                        m.text_start,
                        m.text_end,
                        m.cost,
                        m.strand,
                        m.cigar.to_string(),
                    )
                });
            });

            let mut sassy_map: HashMap<usize, Vec<Match>> = HashMap::new();

            for (idx, pattern) in queries.iter().enumerate() {
                let fwd_matches = if filter_local_pattern_tilingma {
                    sassy_searcher.search(pattern, &text, k as usize)
                } else {
                    sassy_searcher.search_all(pattern, &text, k as usize)
                };

                let mut all_matches = fwd_matches;

                if include_rc {
                    let pattern_rc = crate::profiles::iupac::reverse_complement(pattern);
                    let mut rc_matches = if filter_local_pattern_tilingma {
                        sassy_searcher.search(&pattern_rc, &text, k as usize)
                    } else {
                        sassy_searcher.search_all(&pattern_rc, &text, k as usize)
                    };
                    rc_matches.iter_mut().for_each(|m| {
                        m.strand = crate::Strand::Rc;
                    });
                    all_matches.extend(rc_matches);
                }

                all_matches.sort_unstable_by_key(|m| {
                    (
                        m.text_start,
                        m.text_end,
                        m.cost,
                        m.strand,
                        m.cigar.to_string(),
                    )
                });

                // Fix pattern idx
                all_matches.iter_mut().for_each(|m| {
                    m.pattern_idx = idx;
                });

                if !all_matches.is_empty() {
                    sassy_map.insert(idx, all_matches);
                }
            }

            let sassy_pattern_ids: HashSet<_> = sassy_map.keys().copied().collect();
            let pattern_tiling_pattern_ids: HashSet<_> =
                pattern_tiling_map.keys().copied().collect();

            let pooled_keys: HashSet<_> = sassy_pattern_ids
                .union(&pattern_tiling_pattern_ids)
                .copied()
                .collect();

            for i in pooled_keys {
                let sassy_val = sassy_map.get(&i).cloned().unwrap_or_default();
                let pattern_tiling_val = pattern_tiling_map.get(&i).cloned().unwrap_or_default();

                if sassy_val != pattern_tiling_val {
                    eprintln!("Mismatch for pattern {}", i);
                    eprintln!("Text: {:?}", String::from_utf8_lossy(&text));
                    eprintln!("pattern: {:?}", String::from_utf8_lossy(&queries[i]));
                    eprintln!(
                        "k: {}, pattern length: {}, text length: {}",
                        k,
                        queries[i].len(),
                        text.len()
                    );
                    eprintln!();

                    eprintln!("Sassy matches ({}):", sassy_val.len());
                    for m in &sassy_val {
                        eprintln!(
                            "  pattern_start: {}, pattern_end: {}, text_start: {}, text_end: {}, cigar: {}, cost: {}, strand: {:?}",
                            m.pattern_start,
                            m.pattern_end,
                            m.text_start,
                            m.text_end,
                            m.cigar.to_string(),
                            m.cost,
                            m.strand
                        );
                    }
                    eprintln!();

                    eprintln!("Pattern_tiling matches ({}):", pattern_tiling_val.len());
                    for m in &pattern_tiling_val {
                        eprintln!(
                            "  pattern_start: {}, pattern_end: {}, text_start: {}, text_end: {}, cigar: {}, cost: {}, strand: {:?}",
                            m.pattern_start,
                            m.pattern_end,
                            m.text_start,
                            m.text_end,
                            m.cigar.to_string(),
                            m.cost,
                            m.strand
                        );
                    }
                    eprintln!();

                    panic!("Match mismatch detected");
                }
            }
        }
    }

    #[allow(dead_code)]
    fn search_sassy_pattern_local_pattern_tilingma(
        searcher: &mut SassySearcher<Iupac>,
        pattern: &[u8],
        text: &[u8],
        k: usize,
    ) -> Vec<Match> {
        searcher.search(pattern, &text, k)
    }

    #[allow(dead_code)]
    fn search_sassy_pattern(
        searcher: &mut SassySearcher<Iupac>,
        pattern: &[u8],
        text: &[u8],
        k: usize,
    ) -> Vec<Match> {
        searcher.search_all(pattern, &text, k)
    }

    #[test]
    // #[ignore]
    fn fuzz_against_sassy_alll() {
        // This is just sassy search all in pattern_tiling
        fuzz_against_sassy_batch(Some(0.5), true, true, false);
    }

    #[test]
    fn fuzz_against_sassy_general_local() {
        // This is sassy search in pattern_tiling
        fuzz_against_sassy_batch(Some(0.5), true, true, true);
    }

    #[test]
    fn fuzz_against_sassy_general_local_no_alpha() {
        // This is sassy search in pattern_tiling (no overhang)
        fuzz_against_sassy_batch(None, true, true, true);
    }

    #[test]
    fn fuzz_against_sassy_general_local_no_alpha_no_rc() {
        // This is sassy search in pattern_tiling (no overhang)
        fuzz_against_sassy_batch(None, false, true, true);
    }

    #[test]
    fn pattern_tiling_trace_bug() {
        let q = b"GTCCGAC";
        let q_rc = crate::profiles::iupac::reverse_complement(q);
        let t = b"AAACGAAGTCCTTAGACTGACTTGGCACCAGTATACTCACTTTTTTGTCTCC";

        let k = 1;
        let mut searcher = Myers::<TestBackend, Iupac>::new(None);
        let pattern_transposed = TQueries::<TestBackend, Iupac>::new(&[q.to_vec()], true);
        let pattern_tiling_matches = trace_all_for_test(
            &mut searcher,
            &pattern_transposed,
            t,
            k as u32,
            TracePostProcess::All,
        );
        for m in pattern_tiling_matches {
            println!(
                "pattern_tiling edits: {} cigar: {} end: {} strand: {:?}",
                m.cost,
                m.cigar.to_string(),
                m.text_end,
                m.strand
            );
        }

        let mut sassy_searcher = SassySearcher::<Iupac>::new_fwd();
        let sassy_matches = sassy_searcher.search_all(q, &t, k as usize);
        for m in sassy_matches {
            println!(
                "Sassy edits: {} cigar: {} end: {} strand: {:?}",
                m.cost,
                m.cigar.to_string(),
                m.text_end,
                m.strand
            );
        }
        let sassy_matches_rc = sassy_searcher.search_all(&q_rc, &t, k as usize);
        for m in sassy_matches_rc {
            println!(
                "Sassy RC edits: {} cigar: {} end: {} strand: {:?}",
                m.cost,
                m.cigar.to_string(),
                m.text_end,
                m.strand
            );
        }
    }

    #[test]
    fn pattern_tiling_test() {
        let q = b"TTCGCTAAAGCGGATTTTC";
        let t = b"TGGTTGTGTCTGAGGTGTTCTCTTGCGTGTTGCTAAAGCGGATCTCT";
        println!("Text len: {}", t.len());
        let k = 3;
        let mut sassy_searcher = SassySearcher::<Iupac>::new_fwd_with_overhang(0.5);
        let matches = sassy_searcher.search(q, t, k);
        for m in matches {
            println!(
                "Sassy Match: {} {} {:?} {} {}",
                m.text_start,
                m.text_end,
                m.strand,
                m.cost,
                m.cigar.to_string()
            );
        }

        let mut pattern_tiling_searcher = Searcher::<Iupac>::new(Some(0.5));
        let queries = vec![q.to_vec()];
        let encoded = pattern_tiling_searcher.encode(&queries, false);
        let pattern_tiling_matches = pattern_tiling_searcher.search_with_options(
            &encoded,
            t,
            k as u32,
            Some(true),
            crate::pattern_tiling::trace::TracePostProcess::LocalMinima,
        );
        for m in pattern_tiling_matches.iter() {
            println!(
                "pattern_tiling match: {} {} {:?} {} {}",
                m.text_start,
                m.text_end,
                m.strand,
                m.cost,
                m.cigar.to_string()
            );
        }
    }

    #[test]
    fn test_performance_large_text() {
        let mut searcher = Searcher::<Iupac>::new(None);

        let text = random_dna_seq(1_000_000);

        let pattern = random_dna_seq(32);
        let queries = vec![pattern];

        let encoded = searcher.encode(&queries, false);

        let k = 4;
        let matches = searcher.search_with_options(
            &encoded,
            &text,
            k,
            Some(true),
            crate::pattern_tiling::trace::TracePostProcess::All,
        );

        // Just verify we got some results (or none, both are valid)
        println!("Found {} matches in 1M text", matches.len());
    }

    #[test]
    fn profile_search_patterns_flipped() {
        let mut searcher = Searcher::<Iupac>::new(None);

        let text = random_dna_seq(100_000);

        let queries: Vec<Vec<u8>> = (0..1000).map(|_| random_dna_seq(32)).collect();

        let encoded = searcher.encode(&queries, false);

        let k = 3;
        let matches = searcher.search_with_options(
            &encoded,
            &text,
            k,
            Some(true),
            crate::pattern_tiling::trace::TracePostProcess::All,
        );

        println!(
            "Found {} matches in 100K text with 8 queries of length 64",
            matches.len()
        );
    }

    #[test]
    fn test_sassy_bug() {
        /*
         k = 2
         alpha = 0.5
         ...TCTTAGCTCGGCTT
                         |
                         TGGTT (sassy)
                          TGGTT (pattern_tiling) (5 * 0.5 = 2.5 = 2)


        */
        let t = b"CTGGGTTTAGTTAATTAACAGTGACCACCGAAACAATCTGCATGGAAGAG".to_vec();
        //                    0123456789012345
        //                              TGGTT
        println!("Text len: {}", t.len());
        let p = b"AGTAACC".to_vec();
        let k = 3;
        use crate::Searcher as SassySearcher;
        let mut searcher = SassySearcher::<Iupac>::new_fwd_with_overhang(0.5);
        let mut matches = searcher.search_all(&p, &t, k as usize);

        // Sort the Sassy matches by start, end, edits
        matches.sort_by(|a, b| {
            let a = a.without_cigar();
            let b = b.without_cigar();
            (a.text_start, a.text_end, a.cost).cmp(&(b.text_start, b.text_end, b.cost))
        });
        println!("Sassy matches: {}", matches.len());

        for m in &matches {
            println!(
                "Sassy match: {} {} {:?} {} {}",
                m.text_start,
                m.text_end,
                m.strand,
                m.cost,
                m.cigar.to_string()
            );
        }

        use crate::pattern_tiling::general::Searcher as pattern_tilingSearcher;
        let mut pattern_tiling_searcher = pattern_tilingSearcher::<Iupac>::new(Some(0.5));
        let encoded = pattern_tiling_searcher.encode(&[p.to_vec()], false);
        let pattern_tiling_matches = pattern_tiling_searcher.search_with_options(
            &encoded,
            &t,
            k as u32,
            Some(false),
            crate::pattern_tiling::trace::TracePostProcess::All,
        );

        // Sort the pattern_tiling matches by start, end, edits
        let mut pattern_tiling_matches_sorted = pattern_tiling_matches.to_vec();
        pattern_tiling_matches_sorted.sort_by(|a, b| {
            (a.text_start, a.text_end, a.cost).cmp(&(b.text_start, b.text_end, b.cost))
        });
        println!(
            "pattern_tiling matches: {}",
            pattern_tiling_matches_sorted.len()
        );

        for m in pattern_tiling_matches_sorted.iter() {
            println!(
                "m: {} {} edits {} cigar {}",
                m.text_start,
                m.text_end,
                m.cost,
                m.cigar.to_string()
            );
        }
    }

    #[test]
    fn mini_trace_bug() {
        /*

                left: [(8, 14, 1, "4=1D2="), (8, 16, 2, "3=1I2=1I1=")]
                right: [(8, 14, 1, "4=1D2="), (8, 16, 2, "3=1I2=1I1=1D")]
        */
        let q = b"CCGTCTC".to_vec();
        let t = b"GCACAAAGCCGTTCAT".to_vec();
        let k = 2;
        let mut searcher = Searcher::<Iupac>::new(Some(0.5));
        let encoded = searcher.encode(&[q.to_vec()], false);
        let matches = searcher.search_with_options(
            &encoded,
            &t,
            k as u32,
            Some(true),
            crate::pattern_tiling::trace::TracePostProcess::All,
        );
        for m in matches {
            println!(
                "mini trace bug match: {} {} {:?} {} {}",
                m.text_start,
                m.text_end,
                m.strand,
                m.cost,
                m.cigar.to_string()
            );
        }

        // Same but now with sassy
        let mut sassy_searcher = SassySearcher::<Iupac>::new_fwd_with_overhang(0.5);
        let sassy_matches = sassy_searcher.search_all(&q, &t, k as usize);
        for m in sassy_matches {
            println!(
                "sassy match: {} {} {:?} {} {}",
                m.text_start,
                m.text_end,
                m.strand,
                m.cost,
                m.cigar.to_string()
            );
        }
    }

    #[test]
    fn prefix_bug_using_usize() {
        let p = b"AAATTTGGCTATAGTCT".to_vec();
        //   let p = crate::profiles::iupac::reverse_complement(&p);
        let t = b"TGGTCAATTTGGCTATTCTCT".to_vec();
        let k = 3;
        let mut searcher = Searcher::<Iupac>::new(Some(0.5));
        let encoded = searcher.encode(&[p.to_vec()], false);
        let matches = searcher.search_with_options(
            &encoded,
            &t,
            k as u32,
            Some(true),
            crate::pattern_tiling::trace::TracePostProcess::All,
        );
        let mut sorted_matches = matches.to_vec();
        sorted_matches
            .sort_unstable_by_key(|m| (m.text_start, m.text_end, m.strand, m.cigar.to_string()));
        for m in sorted_matches {
            println!(
                "mini match: text_start: {} text_end: {} pattern_start: {} pattern_end: {} {:?} {} {}",
                m.text_start,
                m.text_end,
                m.pattern_start,
                m.pattern_end,
                m.strand,
                m.cost,
                m.cigar.to_string(),
            );
        }
        // Same but now with sassy
        //let p_rc = crate::profiles::iupac::reverse_complement(&p);
        let mut sassy_searcher = SassySearcher::<Iupac>::new_fwd_with_overhang(0.5);
        let mut sassy_matches = sassy_searcher.search_all(&p, &t, k as usize);
        sassy_matches
            .sort_unstable_by_key(|m| (m.text_start, m.text_end, m.strand, m.cigar.to_string()));
        for m in sassy_matches {
            println!(
                "sassy match: text_start: {} text_end: {} pattern_start: {} pattern_end: {} {:?} {} {}",
                m.text_start,
                m.text_end,
                m.pattern_start,
                m.pattern_end,
                m.strand,
                m.cost,
                m.cigar.to_string()
            );
        }
    }
}
