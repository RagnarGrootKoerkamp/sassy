use crate::pattern_tiling::backend::SimdBackend;
use crate::profiles::{Profile, iupac::reverse_complement};
use std::marker::PhantomData;

pub(crate) const IUPAC_MASKS: usize = 16;

#[derive(Debug, Clone, Default)]
pub struct TQueries<B: SimdBackend, P: Profile> {
    /// Holds [position][transposition_block_index]
    pub pattern_length: usize,
    pub n_queries: usize,
    pub n_original_queries: usize,
    /// Number of SIMD vectors needed to hold all queries (for Peqs)
    pub n_simd_blocks: usize,
    pub peq_masks: [Vec<u64>; IUPAC_MASKS],
    pub peqs: Vec<B::Simd>,
    // Not sure if we should keep it here, but might be handy with
    // get_pattern_seq(i) to make sure it aligns with the indices of `scan` result bools
    pub queries: Vec<Vec<u8>>,
    _marker: PhantomData<(B, P)>,
}

#[inline(always)]
fn build_flat_peqs<B: SimdBackend>(
    peq_masks: &[Vec<u64>; IUPAC_MASKS],
    nq: usize,
    n_simd_blocks: usize,
) -> Vec<B::Simd> {
    let mut peqs = Vec::with_capacity(IUPAC_MASKS * n_simd_blocks);

    for mask_vec in peq_masks.iter() {
        for block_idx in 0..n_simd_blocks {
            let base = block_idx * B::LANES;
            let limit = (nq - base).min(B::LANES);

            let mut lane = B::LaneArray::default();
            let lane_slice = lane.as_mut();

            for lane_idx in 0..limit {
                lane_slice[lane_idx] = B::mask_word_to_scalar(mask_vec[base + lane_idx]);
            }
            peqs.push(B::from_array(lane));
        }
    }
    peqs
}

#[allow(clippy::needless_range_loop)]
impl<B: SimdBackend, P: Profile> TQueries<B, P> {
    pub fn new(queries: &[Vec<u8>], include_rc: bool) -> Self {
        if P::N_CHARS > IUPAC_MASKS {
            panic!("Please raise issue on GitHub if you are using this for ASCII profile");
        }

        assert!(!queries.is_empty(), "No queries provided");
        let pattern_length = queries[0].len();
        assert!(
            pattern_length > 0 && pattern_length <= B::LIMB_BITS,
            "Invalid pattern length {} (must be <= {})",
            pattern_length,
            B::LIMB_BITS
        );
        assert!(
            queries.iter().all(|q| q.len() == pattern_length),
            "All queries must have the same length"
        );

        let n_original_queries = queries.len();
        let est_capacity = n_original_queries * (1 + include_rc as usize);
        let mut all_queries = Vec::with_capacity(est_capacity);
        all_queries.extend_from_slice(queries);
        if include_rc {
            for q in queries {
                let q_rc = reverse_complement(q);
                all_queries.push(q_rc);
            }
        }

        let n_queries = all_queries.len();
        let n_transposition_blocks = n_queries.div_ceil(B::LIMB_BITS);
        let n_simd_blocks = n_queries.div_ceil(B::LANES);

        let mut peq_masks: [Vec<u64>; IUPAC_MASKS] = std::array::from_fn(|_| vec![0u64; n_queries]);

        let mut temp_block_buffer = vec![0u8; B::LIMB_BITS];

        for pos in 0..pattern_length {
            for block_idx in 0..n_transposition_blocks {
                let start_q = block_idx * B::LIMB_BITS;
                let end_q = (start_q + B::LIMB_BITS).min(n_queries);
                let count = end_q - start_q;

                temp_block_buffer.fill(0);

                // iter queries
                for i in 0..count {
                    let qi = start_q + i;
                    let encoded = P::encode_char(all_queries[qi][pos]);
                    temp_block_buffer[i] = encoded;
                    if encoded != 0 {
                        // Skipping X
                        let bit = 1u64 << pos;
                        for (mask_idx, mask_vec) in peq_masks.iter_mut().enumerate().skip(1) {
                            if (encoded & (mask_idx as u8)) != 0 {
                                mask_vec[qi] |= bit;
                            }
                        }
                    }
                }
            }
        }

        let peqs = build_flat_peqs::<B>(&peq_masks, n_queries, n_simd_blocks);

        Self {
            pattern_length,
            n_queries,
            n_original_queries,
            n_simd_blocks,
            peq_masks,
            peqs,
            queries: all_queries,
            _marker: PhantomData,
        }
    }

    pub fn get_pattern_seq(&self, pattern_idx: usize) -> &[u8] {
        &self.queries[pattern_idx]
    }

    pub fn reduce_to_suffix<NB: SimdBackend, NP: Profile>(&self) -> TQueries<NB, NP> {
        let suffix_len = NB::LIMB_BITS;
        let suffix_queries: Vec<Vec<u8>> = self
            .queries
            .iter()
            .map(|q| {
                let start = q.len().saturating_sub(suffix_len);
                q[start..].to_vec()
            })
            .collect();
        TQueries::<NB, NP>::new(&suffix_queries, false)
    }
}
