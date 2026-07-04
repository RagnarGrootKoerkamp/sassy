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
        let mut searcher = Searcher::<Iupac>::new_fwd().with_max_n_frac(0.5);
        let n_filter_matches = searcher.search_all(pattern, text, k);
        assert_eq!(no_n_filter_matches.len(), 6); // [11, 12, 13, 14, 43, 44]
        assert_eq!(n_filter_matches.len(), 1); // [44]
        assert_eq!(n_filter_matches[0].text_end, 44);
    }
}
