use crate::bitpacking::compute_block;
use crate::delta_encoding::H;
use crate::delta_encoding::V;
use crate::profiles::Profile;
use crate::search::init_deltas_for_overshoot_all_lanes;
use crate::search::init_deltas_for_overshoot_scalar;
use pa_types::Cigar;
use pa_types::Cost;
use pa_types::I;

use crate::LANES;
use crate::S;
use crate::bitpacking::compute_block_simd;
use crate::search::{Match, Strand};
use std::array::from_fn;

#[derive(Debug, Clone, Default)]
pub struct CostMatrix {
    /// Query length.
    q: usize,
    deltas: Vec<V>,
    pub(crate) alpha: Option<f32>,
}

impl CostMatrix {
    /// i: text idx
    /// j: query idx
    fn get(&self, i: usize, j: usize) -> Cost {
        let mut s = if let Some(alpha) = self.alpha {
            (j as f32 * alpha).floor() as Cost
        } else {
            j as Cost
        };
        for idx in (j..j + i / 64 * (self.q + 1)).step_by(self.q + 1) {
            s += self.deltas[idx].value();
        }
        if i % 64 != 0 {
            s += self.deltas[j + i / 64 * (self.q + 1)].value_of_prefix(i as I % 64);
        }
        s
    }
}

/// Compute the full n*m matrix corresponding to the query * text alignment.
/// TODO: SIMD variant that takes 1 query, and LANES text slices of the same length.
pub fn fill<P: Profile>(
    query: &[u8],
    text: &[u8],
    len: usize,
    m: &mut CostMatrix,
    alpha: Option<f32>,
) {
    m.q = query.len();
    m.deltas.clear();
    m.deltas.reserve((m.q + 1) * len.div_ceil(64));
    let (profiler, query_profile) = P::encode_pattern(query);
    let mut h = vec![H(1, 0); query.len()];

    init_deltas_for_overshoot_scalar(&mut h, alpha);

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

pub fn simd_fill<P: Profile>(
    query: &[u8],
    texts: &[&[u8]],
    max_len: usize,
    m: &mut [CostMatrix; LANES],
    alpha: Option<f32>,
) {
    assert!(texts.len() <= LANES);
    let lanes = texts.len();

    let (profiler, query_profile) = P::encode_pattern(query);
    let num_chunks = max_len.div_ceil(64);

    for m in &mut *m {
        m.q = query.len();
        m.deltas.clear();
        m.deltas.reserve((m.q + 1) * num_chunks);
    }

    let mut hp: Vec<S> = Vec::with_capacity(query.len());
    let mut hm: Vec<S> = Vec::with_capacity(query.len());
    hp.resize(query.len(), S::splat(1));
    hm.resize(query.len(), S::splat(0));

    // NOTE: It's OK to always fill the left with 010101, even if it's not
    // actually the left of the text, because in that case the left column can't
    // be included in the alignment anyway. (The text has length q+k in that case.)
    init_deltas_for_overshoot_all_lanes(&mut hp, alpha);

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
            let v = V::from(vp[lane], vm[lane]);
            m[lane].deltas.push(v);
        }
        // FIXME: for large queries, use the SIMD within this single block, rather than spreading it thin over LANES 'matches' when there is only a single candidate match.
        for j in 0..query.len() {
            let eq = from_fn(|lane| P::eq(&query_profile[j], &text_profile[lane])).into();
            compute_block_simd(&mut hp[j], &mut hm[j], &mut vp, &mut vm, eq);
            for lane in 0..lanes {
                let v = V::from(vp[lane], vm[lane]);
                m[lane].deltas.push(v);
            }
        }
    }

    for lane in 0..lanes {
        assert_eq!(m[lane].deltas.len(), num_chunks * (m[lane].q + 1));
    }
}

pub fn get_trace<P: Profile>(
    query: &[u8],
    text_offset: usize,
    end_pos: usize,
    text: &[u8],
    m: &CostMatrix,
    alpha: Option<f32>,
) -> Match {
    let mut trace = Vec::new();
    let mut j = query.len();
    let mut i = end_pos - text_offset;

    let cost = |j: usize, i: usize| -> Cost { m.get(i, j) };

    // remaining dist to (i,j)
    let mut g = cost(j, i);
    let mut total_cost = g;

    let mut cigar = Cigar::default();

    let mut pattern_start = 0;
    let mut pattern_end = query.len();

    // Overshoot at end.
    if i > text.len() {
        let overshoot = i - text.len();
        pattern_end -= overshoot;
        let overshoot_cost = (overshoot as f32 * alpha.unwrap()).floor() as Cost;

        total_cost += overshoot_cost;
        i -= overshoot;
        j -= overshoot;
        log::trace!("Trace from ({j}, {i}) for total cost {total_cost}");
        log::trace!("Right overshoot {overshoot} for cost {overshoot_cost}");
    } else {
        log::trace!("Trace from ({j}, {i}) for total cost {total_cost}");
    }

    loop {
        // eprintln!("({i}, {j}) {g}");
        trace.push((j, text_offset + i));

        if j == 0 {
            break;
        }

        if i == 0
            && let Some(alpha) = alpha
        {
            let overshoot = j;
            pattern_start = overshoot;
            // Overshoot at start.
            let overshoot_cost = (overshoot as f32 * alpha).floor() as Cost;
            g -= overshoot_cost;
            break;
        }

        // Match
        if i > 0 && cost(j - 1, i - 1) == g && P::is_match(query[j - 1], text[i - 1]) {
            cigar.push(pa_types::CigarOp::Match);
            j -= 1;
            i -= 1;
            continue;
        }
        // We make some kind of mutation.
        g -= 1;

        // Insert text char.
        if i > 0 && cost(j, i - 1) == g {
            cigar.push(pa_types::CigarOp::Ins);
            i -= 1;
            continue;
        }
        // Mismatch.
        if i > 0 && cost(j - 1, i - 1) == g {
            cigar.push(pa_types::CigarOp::Sub);
            j -= 1;
            i -= 1;
            continue;
        }
        // Delete query char.
        if cost(j - 1, i) == g {
            cigar.push(pa_types::CigarOp::Del);
            j -= 1;
            continue;
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
    use crate::profiles::Dna;

    #[test]
    fn test_traceback() {
        let query = b"ATTTTCCCGGGGATTTT".as_slice();
        let text2: &[u8] = b"ATTTTGGGGATTTT".as_slice();

        let mut cost_matrix = Default::default();
        fill::<Dna>(query, text2, text2.len(), &mut cost_matrix, None);

        let trace = get_trace::<Dna>(query, 0, text2.len(), text2, &cost_matrix, None);
        println!("Trace: {:?}", trace);
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
        );
        let _trace = get_trace::<Dna>(query, 0, text1.len(), text1, &cost_matrix[0], None);
        let _trace = get_trace::<Dna>(query, 0, text2.len(), text2, &cost_matrix[1], None);
        let _trace = get_trace::<Dna>(query, 0, text3.len(), text3, &cost_matrix[2], None);
        let trace = get_trace::<Dna>(query, 0, text4.len(), text4, &cost_matrix[3], None);
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
