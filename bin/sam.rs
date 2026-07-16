use noodles::sam::{
    self,
    alignment::{
        RecordBuf, io::Write as AlignmentWrite, record::data::field::Tag,
        record_buf::data::field::Value,
    },
};
use pa_types::Cigar;
use sassy::{
    Match, Strand,
    profiles::{Dna, Iupac, Profile},
};
use std::path::Path;

use crate::{grep::Alphabet, input_iterator::PatternRecord};

/// Returns whether `path` selects BAM input using the CLI's `.bam` extension convention.
pub fn is_bam_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("bam"))
}

/// Returns whether `path` selects SAM input using a `.sam` extension.
pub fn is_sam_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("sam"))
}

/// Returns whether `path` is a BAM or SAM alignment input.
pub fn is_alignment_path(path: &Path) -> bool {
    is_bam_path(path) || is_sam_path(path)
}

const QNAME: &str = "QNAME";
const FLAG: &str = "FLAG";
const RNAME: &str = "RNAME";
const POS: &str = "POS";
const MAPQ: &str = "MAPQ";
const CIGAR: &str = "CIGAR";
const RNEXT: &str = "RNEXT";
const PNEXT: &str = "PNEXT";
const TLEN: &str = "TLEN";
const SEQ: &str = "SEQ";
const QUAL: &str = "QUAL";
const DATA: &str = "DATA";

const VALID_ALIGNMENT_COLUMNS: [&str; 12] = [
    QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, DATA,
];

/// Adds Sassy's local-use BAM tags to a matching record.
///
/// `sN` stores the match count. `sC`, `sS`, `sB`, and `sE` are parallel numeric arrays for
/// cost, strand (`0` forward, `1` reverse-complement), zero-based start, and exclusive end.
/// `sP`, `sR`, and `sG` are percent-encoded comma-separated lists of pattern IDs, matched
/// regions, and Sassy CIGARs in the same match order.
pub fn annotate_bam_record(
    record: &mut RecordBuf,
    matches: &[(&PatternRecord, Match)],
    alphabet: Alphabet,
    sam_output: bool,
) {
    let match_regions = matches
        .iter()
        .map(|(_, m)| {
            percent_encode(&format_match_region(
                alphabet,
                sam_output,
                &record.sequence().as_ref()[m.text_start..m.text_end],
                m.strand,
            ))
        })
        .collect::<Vec<_>>()
        .join(",");
    let data = record.data_mut();
    for tag in [b"sN", b"sC", b"sS", b"sB", b"sE", b"sP", b"sR", b"sG"] {
        data.remove(&Tag::from(*tag));
    }

    let to_u32 = |value: usize, name: &str| {
        u32::try_from(value).unwrap_or_else(|_| panic!("{name} exceeds BAM auxiliary tag range"))
    };
    data.insert(
        Tag::from(*b"sN"),
        Value::from(to_u32(matches.len(), "match count")),
    );
    data.insert(
        Tag::from(*b"sC"),
        Value::from(
            matches
                .iter()
                .map(|(_, m)| u32::try_from(m.cost).expect("match cost must be non-negative"))
                .collect::<Vec<_>>(),
        ),
    );
    data.insert(
        Tag::from(*b"sS"),
        Value::from(
            matches
                .iter()
                .map(|(_, m)| if m.strand == Strand::Fwd { 0u8 } else { 1u8 })
                .collect::<Vec<_>>(),
        ),
    );
    data.insert(
        Tag::from(*b"sB"),
        Value::from(
            matches
                .iter()
                .map(|(_, m)| to_u32(m.text_start, "match start"))
                .collect::<Vec<_>>(),
        ),
    );
    data.insert(
        Tag::from(*b"sE"),
        Value::from(
            matches
                .iter()
                .map(|(_, m)| to_u32(m.text_end, "match end"))
                .collect::<Vec<_>>(),
        ),
    );
    data.insert(
        Tag::from(*b"sP"),
        Value::from(
            matches
                .iter()
                .map(|(pattern, _)| percent_encode(&pattern.id))
                .collect::<Vec<_>>()
                .join(","),
        ),
    );
    data.insert(Tag::from(*b"sR"), Value::from(match_regions));
    data.insert(
        Tag::from(*b"sG"),
        Value::from(
            matches
                .iter()
                .map(|(_, m)| {
                    percent_encode(format_cigar(sam_output, &m.cigar, m.strand).as_bytes())
                })
                .collect::<Vec<_>>()
                .join(","),
        ),
    );
}

/// Formats a matching region in Sassy's default or SAM-compatible orientation.
pub fn format_match_region(
    alphabet: Alphabet,
    sam_output: bool,
    slice: &[u8],
    strand: Strand,
) -> Vec<u8> {
    if strand == Strand::Rc && !sam_output {
        match alphabet {
            Alphabet::Dna => <Dna as Profile>::reverse_complement(slice),
            Alphabet::Iupac => <Iupac as Profile>::reverse_complement(slice),
        }
    } else {
        slice.to_vec()
    }
}

/// Formats a Sassy CIGAR, reversing reverse-complement matches for SAM-compatible output.
pub fn format_cigar(sam_output: bool, cigar: &Cigar, strand: Strand) -> String {
    if strand == Strand::Rc && sam_output {
        let mut cigar = cigar.clone();
        cigar.reverse();
        cigar.to_string()
    } else {
        cigar.to_string()
    }
}

fn percent_encode(value: impl AsRef<[u8]>) -> String {
    value
        .as_ref()
        .iter()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => vec![*byte],
            byte => format!("%{byte:02X}").into_bytes(),
        })
        .map(char::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use noodles::sam::alignment::record_buf::data::field::Value;
    use pa_types::{CigarElem, CigarOp};

    fn record() -> (sam::Header, RecordBuf) {
        let data = b"@HD\tVN:1.6\n@SQ\tSN:chr20\tLN:1000\nr1\t0\tchr20\t1\t60\t4M\t*\t0\t0\tACGT\t!!!!\tNM:i:0\n";
        let mut reader = sam::io::Reader::new(&data[..]);
        let header = reader.read_header().unwrap();
        let record = reader.record_bufs(&header).next().unwrap().unwrap();
        (header, record)
    }

    fn match_record(
        pattern_idx: usize,
        start: usize,
        end: usize,
        cost: i32,
        strand: Strand,
    ) -> Match {
        Match {
            pattern_idx,
            text_idx: 0,
            text_start: start,
            text_end: end,
            pattern_start: 0,
            pattern_end: end - start,
            cost,
            strand,
            cigar: Cigar {
                ops: vec![CigarElem::new(CigarOp::Match, (end - start) as _)],
            },
        }
    }

    #[test]
    fn sam_column_parses_named_columns_and_all() {
        assert!(matches!(SamColumn::parse_list(None).as_slice(), []));
        assert_eq!(
            SamColumn::parse_list(Some("qname,FLAG,Pos"))
                .iter()
                .map(|column| column.name())
                .collect::<Vec<_>>(),
            ["QNAME", "FLAG", "POS"]
        );
        assert_eq!(
            SamColumn::parse_list(Some("all"))
                .iter()
                .map(|column| column.name())
                .collect::<Vec<_>>(),
            [
                "QNAME", "FLAG", "RNAME", "POS", "MAPQ", "CIGAR", "RNEXT", "PNEXT", "TLEN", "SEQ",
                "QUAL", "DATA",
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Invalid `--more-columns` field")]
    fn sam_column_rejects_unknown_field() {
        SamColumn::parse_list(Some("QNAME,NM"));
    }

    #[test]
    fn sam_fields_formats_mandatory_columns() {
        let (header, record) = record();
        let fields = sam_fields(&header, &record);
        assert_eq!(
            fields,
            [
                "r1", "0", "chr20", "1", "60", "4M", "*", "0", "0", "ACGT", "!!!!", "NM:i:0",
            ]
        );
        assert_eq!(SamColumn::Data.value(&fields), "NM:i:0");
    }

    #[test]
    fn annotate_bam_record_writes_and_replaces_sassy_tags() {
        let (_, mut record) = record();
        record.data_mut().insert(Tag::from(*b"sN"), Value::from(99));
        let patterns = [
            PatternRecord {
                id: "foo/bar".into(),
                seq: b"AC".to_vec(),
            },
            PatternRecord {
                id: "rev".into(),
                seq: b"AC".to_vec(),
            },
        ];
        let matches = [
            (&patterns[0], match_record(0, 0, 2, 0, Strand::Fwd)),
            (&patterns[1], match_record(1, 2, 4, 1, Strand::Rc)),
        ];

        annotate_bam_record(&mut record, &matches, Alphabet::Dna, false);

        let data = record.data();
        assert_eq!(data.get(&Tag::from(*b"sN")), Some(&Value::from(2)));
        assert_eq!(
            data.get(&Tag::from(*b"sC")),
            Some(&Value::from(vec![0u32, 1]))
        );
        assert_eq!(
            data.get(&Tag::from(*b"sS")),
            Some(&Value::from(vec![0u8, 1]))
        );
        assert_eq!(
            data.get(&Tag::from(*b"sB")),
            Some(&Value::from(vec![0u32, 2]))
        );
        assert_eq!(
            data.get(&Tag::from(*b"sE")),
            Some(&Value::from(vec![2u32, 4]))
        );
        assert_eq!(
            data.get(&Tag::from(*b"sP")),
            Some(&Value::from("foo%2Fbar,rev"))
        );
        assert_eq!(data.get(&Tag::from(*b"sR")), Some(&Value::from("AC,AC")));
        assert_eq!(
            data.get(&Tag::from(*b"sG")),
            Some(&Value::from("2%3D,2%3D"))
        );
    }
}
