use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Alphabet {
    Dna,
    Iupac,
    Ascii,
}

/// Generate random data with inserted target matches
pub fn generate_query_and_text_with_matches(
    ql: usize,
    tl: usize,
    num: usize,
    min_edits: usize,
    max_edits: usize,
    alphabet: &Alphabet,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<(usize, usize)>) {
    let mut rng = rand::rng();
    let query = generate_random_sequence(ql, alphabet, None);

    // Get the original text, where we insert NUM queries
    let mut text_base = generate_random_sequence(tl, alphabet, None);

    let mut locs = Vec::new();
    for _ in 0..num {
        let m = mutate_sequence(&query, min_edits, max_edits);
        if m.len() > text_base.len() {
            continue;
        }
        let max_start = text_base.len() - m.len();
        for _ in 0..10 {
            let start = rng.random_range(0..=max_start);
            let end = start + m.len();
            if locs.iter().all(|&(s, e)| end <= s || start >= e) {
                text_base.splice(start..end, m.iter().cloned());
                locs.push((start, end));
                break;
            }
        }
    }

    // Insert one query extra, at a random location keep trying until we find a location that doesn't overlap with any of the existing matches
    let mut text_with_insert = text_base.clone();
    // let max_start = text_base.len() - query.len();
    loop {
        //let start = rng.random_range(0..=max_start);
        //let end = start + query.len();
        let start = text_base.len() - query.len();
        let end = text_base.len();

        if locs.iter().all(|&(s, e)| end <= s || start >= e) {
            text_with_insert.splice(start..end, query.iter().cloned());
            break;
        }
    }

    (query, text_base, text_with_insert, locs)
}

/// Generate random dna sequence of length "length" with alphabet
fn generate_random_sequence(length: usize, alphabet: &Alphabet, bias: Option<&str>) -> Vec<u8> {
    let mut rng = rand::rng();
    // If there is bias just repeat the bias character length times
    if let Some(bias) = bias {
        return vec![bias.as_bytes()[0]; length];
    }
    match alphabet {
        Alphabet::Dna => (0..length)
            .map(|_| b"ACGT"[rng.random_range(0..4)])
            .collect(),
        Alphabet::Iupac => (0..length)
            .map(|_| b"ACGTURYSWKMBDHVNX"[rng.random_range(0..16)])
            .collect(),
        Alphabet::Ascii => (0..length)
            .map(|_| rng.random_range(0..256) as u8)
            .collect(),
    }
}

/// Mutate sequence with at most max_edits
fn mutate_sequence(sequence: &[u8], min_edits: usize, max_edits: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    let mut seq = sequence.to_vec();
    for _ in 0..rng.random_range(min_edits..=max_edits) {
        let idx = rng.random_range(0..seq.len());
        match rng.random_range(0..3) {
            0 => {
                let current = seq[idx];
                let mut new_char;
                // Keep trying until we get a different character
                loop {
                    new_char = b"ACGT"[rng.random_range(0..4)];
                    if new_char != current {
                        break;
                    }
                }
                seq[idx] = new_char;
            }
            1 if seq.len() > 1 => {
                seq.remove(idx);
            }
            2 => seq.insert(idx, b"ACGT"[rng.random_range(0..4)]),
            _ => {}
        }
    }
    seq
}

#[cfg(test)]
#[allow(unused)]
mod sim_data_tests {

    use super::*;

    fn naive_edit_dist(q: &[u8], t: &[u8]) -> usize {
        let m = q.len();
        let n = t.len();

        // Create a matrix to store the edit distances
        let mut dp = vec![vec![0; n + 1]; m + 1];

        // Initialize the first row and column
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        // Fill the dp matrix
        for i in 1..=m {
            for j in 1..=n {
                let cost = if q[i - 1] == t[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1) // deletion
                    .min(dp[i][j - 1] + 1) // insertion
                    .min(dp[i - 1][j - 1] + cost); // substitution
            }
        }

        // Return the edit distance
        dp[m][n]
    }

    #[test]
    fn test_random_data_single_match_no_edits() {
        let (q, t, t_plus_one_q, locs) =
            generate_query_and_text_with_matches(10, 100, 1, 0, 0, &Alphabet::Dna);
        let (s, e) = locs[0];
        assert_eq!(q, t[s..e]);
    }

    #[test]
    fn test_random_data_single_match_1_edit() {
        let (q, t, t_plus_one_q, locs) =
            generate_query_and_text_with_matches(10, 100, 1, 1, 1, &Alphabet::Dna);
        let (s, e) = locs[0];
        assert_ne!(q, t[s..e]);
        // Get actual edits using edlib wrapper
        let e = naive_edit_dist(&q, &t[s..e]);
        assert_eq!(e, 1);
    }

    #[test]
    fn test_random_two_matches() {
        let (q, t, t_plus_one_q, locs) =
            generate_query_and_text_with_matches(10, 100, 2, 1, 1, &Alphabet::Dna);
        assert_eq!(locs.len(), 2);
        for loc in locs {
            let (s, e) = loc;
            let e = naive_edit_dist(&q, &t[s..e]);
            assert_eq!(e, 1);
        }
    }
}
