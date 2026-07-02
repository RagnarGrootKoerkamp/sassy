use crate::Match;
use crate::Strand;

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
    let (start, end) = match m.strand {
        Strand::Fwd => {
            let end = m.text_end;
            (end.saturating_sub(pattern_len.saturating_sub(k)), end)
        }
        Strand::Rc => {
            let start = m.text_start;
            (
                start,
                (start + pattern_len.saturating_sub(k)).min(text.len()),
            )
        }
    };
    let n_count = text[start..end]
        .iter()
        .filter(|&&c| c.eq_ignore_ascii_case(&b'N'))
        .count();
    n_count as f32 <= max_n_frac * (pattern_len + k) as f32
}

/// Returns `true` if `m` has an alignment satisfying `max_n_frac`.
///
/// For traced matches we know the text slice so we just have to count the N's
/// and check if the N-fraction is less than or equal to `max_n_frac`.
fn traced_satisfy_n_frac(m: &Match, text: &[u8], max_n_frac: f32) -> bool {
    let slice = &text[m.text_start..m.text_end];
    let n_count = slice
        .iter()
        .filter(|&&c| c.eq_ignore_ascii_case(&b'N'))
        .count() as f32;
    n_count / slice.len() as f32 <= max_n_frac
}
