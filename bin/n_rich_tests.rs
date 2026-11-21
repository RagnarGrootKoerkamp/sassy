use crate::grep::{CrisprMode, search_v1, search_v2};
use crate::input_iterator::{InputIterator, PatternRecord, TextRecord};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sassy::{
    Match, Searcher, Strand,
    profiles::{Iupac, Profile},
};
use std::io::Write;
use tempfile::NamedTempFile;

const MAX_N_FRACTIONS: [f32; 7] = [0.0, 0.05, 0.1, 0.2, 0.45, 0.9, 0.99];

struct Case {
    pattern: PatternRecord,
    text: Vec<u8>,
    k: usize,
    chunk_size: usize,
    max_n_frac: f32,
    forward_start: usize,
    reverse_start: usize,
}

impl Case {
    fn new(rng: &mut StdRng, len: usize, k: usize) -> Self {
        let mut pattern: Vec<_> = (0..len).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();
        // A/T ends on the two strands and C/G flanks isolate the planted
        // zero-edit hits from the long N runs.
        pattern[0] = b'A';
        pattern[len - 1] = b'A';
        let flank = b"CG"[rng.random_range(0..2)];
        let overlap = len + k;
        let chunk_size = 2 * overlap + rng.random_range(1..=31);
        let forward_start = chunk_size - overlap - len / 2;
        let mut text = vec![flank; forward_start];
        text.extend_from_slice(&pattern);
        let mut reverse_start = 0;
        let min_run_len = chunk_size.max(64);
        for run in 0..rng.random_range(2..=4) {
            text.resize(text.len() + overlap + rng.random_range(1..=25), flank);
            let run_len = rng.random_range(min_run_len..=4 * min_run_len);
            text.extend((0..run_len).map(|_| if rng.random_bool(0.5) { b'N' } else { b'n' }));
            text.resize(text.len() + overlap + rng.random_range(1..=25), flank);
            if run == 0 {
                reverse_start = text.len();
                text.extend(Iupac::reverse_complement(&pattern));
            } else if run == 1 && len > 2 {
                // A separate hit whose interior has Ns can pass or fail the
                // exact fraction check independently of the long N runs.
                let mut masked = pattern.clone();
                let start = rng.random_range(1..len - 1);
                let end = rng.random_range(start + 1..len);
                masked[start..end].fill(b'N');
                text.extend(masked);
                text.resize(text.len() + overlap, flank);
            }
        }
        Self {
            pattern: PatternRecord {
                id: "pattern".into(),
                seq: pattern,
            },
            text,
            k,
            chunk_size,
            max_n_frac: MAX_N_FRACTIONS[rng.random_range(0..MAX_N_FRACTIONS.len())],
            forward_start,
            reverse_start,
        }
    }

    fn searcher(&self, rc: bool) -> Searcher<Iupac> {
        Searcher::new(rc, None).with_max_n_frac(self.max_n_frac)
    }

    fn check(
        &self,
        case_id: usize,
        mode: &str,
        mut search: impl FnMut(&TextRecord) -> (Vec<Match>, Vec<Match>),
    ) {
        let len = self.pattern.seq.len();
        let context = format!(
            "{mode}, case {case_id}, length {len}, k {}, chunk {}, max_n_frac {}",
            self.k, self.chunk_size, self.max_n_frac,
        );
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, ">n-rich").unwrap();
        file.write_all(&self.text).unwrap();
        writeln!(file).unwrap();

        let paths = vec![file.path().to_path_buf()];
        let input = InputIterator::new_with_chunk_size(
            &paths,
            std::slice::from_ref(&self.pattern),
            None,
            None,
            true,
            len + self.k,
            self.chunk_size,
            false,
        );
        let planted = [
            (Strand::Fwd, self.forward_start),
            (Strand::Rc, self.reverse_start),
        ];
        let mut seen = [false; 2];
        while let Some((_, (_, _, texts))) = input.next_batch() {
            for text in texts.iter() {
                let (v1, v2) = search(text);
                let offset = text.record_offset;
                let owned = |matches: Vec<Match>| {
                    let mut owned: Vec<_> = matches
                        .into_iter()
                        .filter(|m| text.owns_match(m))
                        .map(|m| {
                            // Match equality deliberately excludes the CIGAR.
                            let cigar = m.cigar.to_string();
                            (m, cigar)
                        })
                        .collect();
                    owned.sort_unstable();
                    owned
                };
                let v1 = owned(v1);
                assert_eq!(v1, owned(v2), "{context}, record offset {offset}");
                for (seen, (strand, start)) in seen.iter_mut().zip(planted) {
                    *seen |= v1.iter().any(|(m, _)| {
                        m.strand == strand
                            && offset + m.text_start == start
                            && offset + m.text_end == start + len
                            && m.cost == 0
                    });
                }
            }
        }
        assert!(
            seen.into_iter().all(|hit| hit),
            "missing planted hit {seen:?}: {context}"
        );
    }
}

fn check_search(case: &Case, id: usize, local_minima: bool) {
    let pattern = &case.pattern.seq;
    let k = case.k;
    let mut v1 = case.searcher(true);
    let mut v2 = case.searcher(true);
    let patterns = [pattern.as_slice()];
    let mode = if local_minima {
        "local minima"
    } else {
        "all endpoints"
    };
    case.check(id, mode, |text| {
        if local_minima {
            (
                v1.search(pattern, &text.seq, k),
                v2.search_v2(&patterns, std::slice::from_ref(text), k),
            )
        } else {
            (
                v1.search_all(pattern, &text.seq, k),
                v2.search_all_v2(&patterns, std::slice::from_ref(text), k),
            )
        }
    });
}

#[test]
fn fuzz_n_rich_search_backends_on_both_strands() {
    let mut rng = StdRng::seed_from_u64(0x6e_7269_6368);
    for id in 0..192 {
        let case = Case::new(&mut rng, id % 64 + 1, 0);
        check_search(&case, id, true);
        check_search(&case, id, false);
    }

    let mut rng = StdRng::seed_from_u64(0x61_6e63_686f_72);
    for len in 2..=64 {
        let k = rng.random_range(1..=3.min(len - 1));
        check_search(&Case::new(&mut rng, len, k), len, false);
    }
}

#[test]
fn fuzz_n_rich_crispr_backends_on_both_strands() {
    let mut rng = StdRng::seed_from_u64(0x63_7269_7370_72);
    for case_id in 0..192 {
        let len = case_id % 64 + 1;
        let k = rng.random_range(0..=3.min(len - 1));
        let case = Case::new(&mut rng, len, k);
        let guide = &case.pattern.seq;
        let pam_len = rng.random_range(1..=3.min(len));
        let pam = &guide[len - pam_len..];
        let reverse_pam = Iupac::complement(pam);
        let mut v1 = case.searcher(true);
        let mut v2 = case.searcher(true);
        let patterns = [guide.as_slice()];
        for allow_pam_edits in [false, true] {
            let pam_filter = CrisprMode {
                pam_length: pam_len,
                allow_pam_edits,
                legacy_tsv: false,
            };
            let mode = if allow_pam_edits {
                "editable PAM"
            } else {
                "checked PAM"
            };
            case.check(case_id, mode, |text| {
                let mut a = v1.search_all(guide, &text.seq, k);
                let mut b = v2.search_all_v2(&patterns, std::slice::from_ref(text), k);
                let filter = |m: &Match| {
                    let prefix = match m.strand {
                        Strand::Fwd => &text.seq.text[..m.text_end],
                        Strand::Rc => {
                            &text.seq.rev.as_ref().unwrap()[..text.seq.text.len() - m.text_start]
                        }
                    };
                    let expected = if m.strand == Strand::Fwd {
                        pam
                    } else {
                        &reverse_pam
                    };
                    let matching_pam = prefix.len() >= pam_len
                        && Iupac::is_match_slice(expected, &prefix[prefix.len() - pam_len..]);
                    let accepted = pam_filter.accepts_match(&case.pattern, m);
                    assert_eq!(
                        accepted,
                        allow_pam_edits || matching_pam,
                        "PAM length {pam_len}: {m:?}"
                    );
                    accepted
                };
                a.retain(filter);
                b.retain(filter);
                let filtered_v1 = search_v1(&mut v1, &case.pattern, text, k, Some(pam_filter));
                let filtered_v2 = search_v2(
                    &mut v2,
                    &patterns,
                    std::slice::from_ref(text),
                    k,
                    Some(pam_filter),
                );
                assert_eq!(filtered_v1, a, "v1, case {case_id}, {mode}");
                assert_eq!(filtered_v2, b, "v2, case {case_id}, {mode}");
                (filtered_v1, filtered_v2)
            });
        }
    }
}

#[test]
fn n_filter_changes_local_minimum_selection_between_backends() {
    let pattern = b"AAAA".to_vec();
    let text = b"AAAANNNN";
    let mut searcher = Searcher::<Iupac>::new_fwd().with_max_n_frac(0.9);
    let patterns = [pattern.as_slice()];
    assert!(searcher.search(&pattern, text, 0).is_empty());
    let v2 = searcher.search_v2(&patterns, &[text], 0);
    assert_eq!(v2.len(), 1);
    assert_eq!(v2[0].text_start..v2[0].text_end, 3..7);
}

#[test]
fn reverse_search_uses_the_same_anchors_in_both_backends() {
    let pattern = b"AAAA".to_vec();
    let text = b"TTTTT";
    let mut searcher = Searcher::<Iupac>::new_rc().with_max_n_frac(0.5);
    let patterns = [pattern.as_slice()];
    let spans = |matches: &[Match]| {
        let mut spans: Vec<_> = matches
            .iter()
            .map(|m| (m.text_start, m.text_end, m.cost))
            .collect();
        spans.sort_unstable();
        spans
    };
    assert_eq!(spans(&searcher.search(&pattern, text, 0)), [(0, 4, 0)]);
    assert_eq!(
        spans(&searcher.search_v2(&patterns, &[text], 0)),
        [(0, 4, 0)]
    );
    assert_eq!(
        spans(&searcher.search_all(&pattern, text, 1)),
        [(0, 4, 0), (1, 5, 0), (2, 5, 1)]
    );
    assert_eq!(
        spans(&searcher.search_all_v2(&patterns, &[text], 1)),
        [(0, 4, 0), (1, 5, 0), (2, 5, 1)]
    );
}
