use crate::crispr::{matching_seq, search_v2};
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

#[test]
fn fuzz_n_rich_search_backends_on_both_strands() {
    let mut rng = StdRng::seed_from_u64(0x6e_7269_6368);
    for case_id in 0..192 {
        // Isolated exact hits agree even with opposite reverse-strand anchors.
        let case = Case::new(&mut rng, case_id % 64 + 1, 0);
        let pattern = &case.pattern.seq;
        let mut v1 = case.searcher(true);
        let mut v2 = case.searcher(true);
        let encoded = v2.encode_patterns(std::slice::from_ref(pattern));
        case.check(case_id, "local minima", |text| {
            (
                v1.search(pattern, &text.seq, 0),
                v2.search_encoded_patterns(&encoded, &text.seq.text, 0)
                    .to_vec(),
            )
        });
        case.check(case_id, "all endpoints", |text| {
            (
                v1.search_all(pattern, &text.seq, 0),
                v2.search_all_encoded_patterns(&encoded, &text.seq.text, 0)
                    .to_vec(),
            )
        });
    }
}

#[test]
fn fuzz_n_rich_search_all_with_matching_anchors() {
    let mut rng = StdRng::seed_from_u64(0x61_6e63_686f_72);
    for len in 2..=64 {
        let k = rng.random_range(1..=3.min(len - 1));
        let case = Case::new(&mut rng, len, k);
        let pattern = &case.pattern.seq;
        let reverse = Iupac::reverse_complement(pattern);
        let mut v1 = case.searcher(false);
        let mut v2 = case.searcher(true);
        let encoded = v2.encode_patterns(std::slice::from_ref(pattern));
        case.check(len, "all endpoints with matching anchors", |text| {
            let mut expected = v1.search_all(pattern, &text.seq, k);
            // Give v1 the same forward-text end anchor as ordinary v2 search.
            for mut m in v1.search_all(&reverse, &text.seq, k) {
                m.strand = Strand::Rc;
                (m.pattern_start, m.pattern_end) = (len - m.pattern_end, len - m.pattern_start);
                m.cigar.reverse();
                expected.push(m);
            }
            (
                expected,
                v2.search_all_encoded_patterns(&encoded, &text.seq.text, k)
                    .to_vec(),
            )
        });
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
        let mut v2 = case.searcher(false);
        let forward = v2.encode_patterns(std::slice::from_ref(guide));
        let reverse = v2.encode_patterns(&[Iupac::complement(guide)]);
        for allow_pam_edits in [false, true] {
            let accepts_endpoint = |prefix: &[u8], strand| {
                let expected = if strand == Strand::Fwd {
                    pam
                } else {
                    &reverse_pam
                };
                allow_pam_edits
                    || (prefix.len() >= pam_len
                        && matching_seq::<Iupac>(&prefix[prefix.len() - pam_len..], expected))
            };
            let mode = if allow_pam_edits {
                "editable PAM"
            } else {
                "checked PAM"
            };
            case.check(case_id, mode, |text| {
                let a = if allow_pam_edits {
                    v1.search_all(guide, &text.seq, k)
                } else {
                    v1.search_with_fn(guide, &text.seq, k, true, |_, prefix, strand| {
                        accepts_endpoint(prefix, strand)
                    })
                };
                let [b] = search_v2(
                    &mut v2,
                    &forward,
                    Some(&reverse),
                    &text.seq,
                    1,
                    k,
                    accepts_endpoint,
                )
                .try_into()
                .unwrap();
                (a, b)
            });
        }
    }
}

#[test]
fn n_filter_changes_local_minimum_selection_between_backends() {
    let pattern = b"AAAA".to_vec();
    let text = b"AAAANNNN";
    let mut searcher = Searcher::<Iupac>::new_fwd().with_max_n_frac(0.9);
    let encoded = searcher.encode_patterns(std::slice::from_ref(&pattern));
    assert!(searcher.search(&pattern, text, 0).is_empty());
    let v2 = searcher.search_encoded_patterns(&encoded, text, 0);
    assert_eq!(v2.len(), 1);
    assert_eq!(v2[0].text_start..v2[0].text_end, 3..7);
}

#[test]
fn reverse_search_uses_different_anchors_between_backends() {
    let pattern = b"AAAA".to_vec();
    let text = b"TTTTT";
    let mut searcher = Searcher::<Iupac>::new_rc().with_max_n_frac(0.5);
    let encoded = searcher.encode_patterns(std::slice::from_ref(&pattern));
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
        spans(searcher.search_encoded_patterns(&encoded, text, 0)),
        [(1, 5, 0)]
    );
    assert_eq!(
        spans(&searcher.search_all(&pattern, text, 1)),
        [(0, 4, 0), (1, 5, 0), (2, 5, 1)]
    );
    assert_eq!(
        spans(searcher.search_all_encoded_patterns(&encoded, text, 1)),
        [(0, 3, 1), (0, 4, 0), (1, 5, 0)]
    );
}
