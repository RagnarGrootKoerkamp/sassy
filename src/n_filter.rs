use crate::Match;

#[inline(always)]
fn count_ns(text: &[u8], start_pos: usize, end_pos: usize, max_n_frac: f32) -> bool {
    // Overhang cases:
    // - `end_pos` can be >= `text.len()`; we do not consider these `N`s.
    // - suffix start can fall outside the text if the match is entirely in padding
    if start_pos >= text.len() {
        return true;
    }
    let slice = &text[start_pos..end_pos];
    // To prevent a divide by zero error
    if slice.is_empty() {
        return true;
    }
    let n_count = slice
        .iter()
        .filter(|&&c| c.eq_ignore_ascii_case(&b'N'))
        .count();
    n_count as f32 <= max_n_frac * (slice.len() as f32)
}

/// Returns `true` if `m` has an alignment that may satisfy `max_n_frac`.
///
/// Since this is not based on a traced alignment we cannot guarantee the traced
/// version would indeed satisfy the threshold, but we can lower-bound filter it.
pub(crate) fn satisfy_n_endpoint_filter(
    end_pos: usize,
    text: &[u8],
    pattern_len: usize,
    k: usize,
    max_n_frac: f32,
) -> bool {
    let suffix_len = pattern_len.saturating_sub(k);
    let start_pos = end_pos.saturating_sub(suffix_len);
    let end_pos = end_pos.min(text.len());
    // Original code uses pattern_len + k as denominator, but in case of overhang
    // this is invalid since the actual aligned slice *can* be shorter than that.
    count_ns(text, start_pos, end_pos, max_n_frac)
}

/// Returns `true` if `m` has an alignment satisfying `max_n_frac`.
///
/// For traced matches we know the text slice so we just have to count the N's
/// and check if the N-fraction is less than or equal to `max_n_frac`.
pub(crate) fn traced_satisfy_n_frac(m: &Match, text: &[u8], max_n_frac: f32) -> bool {
    count_ns(text, m.text_start, m.text_end, max_n_frac)
}

#[cfg(test)]
mod tests {
    use crate::Searcher;
    use crate::profiles::Iupac;

    #[test]
    fn n_filter_full_overhang_match() {
        let pattern = b"AAAA";
        let text = b"GGGGGG";
        let k = 2; // using alpha 0.5 = 2 * 0.5 = 4 chars 
        let alpha = 0.5;
        // max frac should leave overhang "N's"
        let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(alpha).with_max_n_frac(0.0);
        let matches = searcher.search_all(pattern, text, k);
        /*
               NNNNGGGGGGNNNN (pat_len + k = 4 + 2 = 6 * N)
               AAAA     AAAA
                AAAA     AAAA
        */
        assert_eq!(matches.len(), 4);
    }

    #[test]
    fn n_filter_complex_example() {
        let pattern = b"ACGTACGTACGT";
        let text = b"NNNNNNNNNNNNNAAAAAAAAAAAAAAAAAANNNNNNNGTACGT";
        /*
         NNNNNNNNNNNNNAAAAAAAAAAAAAAAAAANNNNNNNGTACGT
        ACGTACGTACGT    11
         ACGTACGTACGT   12               ACGTACGTACGT (44, e=0)
          ACGTACGTACGT  13              ACGTACGTACGT (43, e=1)
           ACGTACGTACGT 14

        of these, only 43
        */
        let k = 1;
        let mut searcher = Searcher::<Iupac>::new_fwd();
        let no_n_filter_matches = searcher.search_all(pattern, text, k);
        let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(0.5).with_max_n_frac(0.5);
        let n_filter_matches = searcher.search_all(pattern, text, k);
        assert_eq!(no_n_filter_matches.len(), 6); // [11, 12, 13, 14, 43, 44]
        assert_eq!(n_filter_matches.len(), 1); // [44]
        assert_eq!(n_filter_matches[0].text_end, 44);
    }

    #[test]
    fn n_filter_fuzz_case() {
        let pattern = b"GGGACN".to_vec();
        let text = b"GAGGGCCA".to_vec();
        let k = 3;
        let max_n_frac = 0.13340974;
        let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(0.5);

        // No n filter
        let matches_with_n_filter = searcher.search_all(&pattern, &text, k);

        // N filter of 0.13..
        searcher.set_max_n_frac(max_n_frac);
        let matches_without_n_filter = searcher.search_all(&pattern, &text, k);

        assert_eq!(matches_with_n_filter.len(), matches_without_n_filter.len());
    }

    #[test]
    fn fuzz_not_crashing_with_max_n_frac() {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let bases = b"NACGT";
        let max_n_frac = rng.random_range(0.0..=1.0);
        let alpha = 0.5;

        for _ in 0..100_000 {
            let plen = rng.random_range(4..=20);
            let tlen = rng.random_range(plen..=plen + 10);
            let k = rng.random_range(0..=3usize);

            let pattern: Vec<u8> = (0..plen)
                .map(|_| bases[rng.random_range(0..4usize)])
                .collect();
            let text: Vec<u8> = (0..tlen)
                .map(|_| bases[rng.random_range(0..4usize)])
                .collect();
            eprintln!("Pattern: {}", String::from_utf8_lossy(&pattern));
            eprintln!("Text: {}", String::from_utf8_lossy(&text));
            eprintln!("k: {}", k);
            eprintln!("max N Fraction: {}", max_n_frac);
            eprintln!("--------------------------------");
            let mut searcher =
                Searcher::<Iupac>::new_rc_with_overhang(alpha).with_max_n_frac(max_n_frac);
            let matches_v1 = std::hint::black_box(searcher.search_all(&pattern, &text, k));

            // This is also handled in other fuzz test already
            let encoded = searcher.encode_patterns(&[pattern.clone()]);
            let matches_v2 =
                std::hint::black_box(searcher.search_all_encoded_patterns(&encoded, &text, k));

            eprintln!("matches v1");
            for m in matches_v1.iter() {
                eprintln!("Match: {:?}", m);
            }
            eprintln!("matches v2");
            for m in matches_v2.iter() {
                eprintln!("Match: {:?}", m);
            }

            assert_eq!(matches_v1.len(), matches_v2.len());
        }
    }
}
