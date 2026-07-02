use crate::Match;
use crate::Strand;

#[inline(always)]
fn count_ns(slice: &[u8]) -> usize {
    slice
        .iter()
        .filter(|&&c| c.eq_ignore_ascii_case(&b'N'))
        .count()
}

/// Returns `true` if `m` could possibly produce an alignment satisfying `max_n_frac`.
///
/// Uses the mandatory-suffix lower bound: any alignment of cost <= k must cover at least
/// `mandatory_suffix = text[end - pattern_length + k: end]`, so the N-fraction is
/// bounded below by `count_ns(mandatory_suffix) / (pattern_len + k)``.
fn untraced_satisfy_n_frac(
    m: &Match,
    text: &[u8],
    pattern_len: usize,
    k: usize,
    max_n_frac: f32,
) -> bool {
    let suffix_len = pattern_len.saturating_sub(k);
    let (start, end) = match m.strand {
        Strand::Fwd => (m.text_end.saturating_sub(suffix_len), m.text_end),
        Strand::Rc => (m.text_start, (m.text_start + suffix_len).min(text.len())),
    };
    let n_count = count_ns(&text[start..end]);
    n_count as f32 <= max_n_frac * (pattern_len + k) as f32
}

/// Returns `true` if `m` has an alignment satisfying `max_n_frac`.
///
/// For traced matches we know the text slice so we just have to count the N's
/// and check if the N-fraction is less than or equal to `max_n_frac`.
fn traced_satisfy_n_frac(m: &Match, text: &[u8], max_n_frac: f32) -> bool {
    let slice = &text[m.text_start..m.text_end];
    let n_count = count_ns(slice) as f32;
    n_count / slice.len() as f32 <= max_n_frac
}
