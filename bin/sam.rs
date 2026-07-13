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

use crate::{grep::Alphabet, input_iterator::PatternRecord};

#[derive(Clone, Copy)]
/// A SAM field that can be appended to Sassy TSV output.
pub(crate) enum SamColumn {
    Qname,
    Flag,
    Rname,
    Pos,
    Mapq,
    Cigar,
    Rnext,
    Pnext,
    Tlen,
    Seq,
    Qual,
    /// All optional SAM alignment fields in their standard `TAG:TYPE:VALUE` representation.
    Data,
}

impl SamColumn {
    const ALL: [Self; 12] = [
        Self::Qname,
        Self::Flag,
        Self::Rname,
        Self::Pos,
        Self::Mapq,
        Self::Cigar,
        Self::Rnext,
        Self::Pnext,
        Self::Tlen,
        Self::Seq,
        Self::Qual,
        Self::Data,
    ];

    /// Parses the `--more-columns` value, accepting `all` or mandatory SAM field names.
    pub fn parse_list(value: Option<&str>) -> Vec<Self> {
        let Some(value) = value else {
            return Vec::new();
        };
        let fields: Vec<_> = value.split(',').map(str::trim).collect();
        assert!(
            !fields.is_empty() && fields.iter().all(|field| !field.is_empty()),
            "--more-columns must be a non-empty comma-separated list"
        );
        if fields.len() == 1 && fields[0].eq_ignore_ascii_case("all") {
            return Self::ALL.to_vec();
        }
        assert!(
            !fields.iter().any(|field| field.eq_ignore_ascii_case("all")),
            "`--more-columns=all` cannot be combined with individual SAM fields"
        );
        fields
            .into_iter()
            .map(|field| match field.to_ascii_uppercase().as_str() {
                "QNAME" => Self::Qname,
                "FLAG" => Self::Flag,
                "RNAME" => Self::Rname,
                "POS" => Self::Pos,
                "MAPQ" => Self::Mapq,
                "CIGAR" => Self::Cigar,
                "RNEXT" => Self::Rnext,
                "PNEXT" => Self::Pnext,
                "TLEN" => Self::Tlen,
                "SEQ" => Self::Seq,
                "QUAL" => Self::Qual,
                "DATA" => Self::Data,
                _ => panic!(
                    "Invalid `--more-columns` field `{field}`. Use `all` or one of QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, DATA."
                ),
            })
            .collect()
    }

    /// Returns the SAM specification field name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Qname => "QNAME",
            Self::Flag => "FLAG",
            Self::Rname => "RNAME",
            Self::Pos => "POS",
            Self::Mapq => "MAPQ",
            Self::Cigar => "CIGAR",
            Self::Rnext => "RNEXT",
            Self::Pnext => "PNEXT",
            Self::Tlen => "TLEN",
            Self::Seq => "SEQ",
            Self::Qual => "QUAL",
            Self::Data => "DATA",
        }
    }

    /// Returns this column's standard SAM representation from a serialized alignment line.
    ///
    /// `DATA` joins all optional fields with tabs, matching the SAM alignment-line format
    /// produced by Noodles' writer.
    pub fn value(self, fields: &[String]) -> String {
        match self {
            Self::Qname => fields[0].clone(),
            Self::Flag => fields[1].clone(),
            Self::Rname => fields[2].clone(),
            Self::Pos => fields[3].clone(),
            Self::Mapq => fields[4].clone(),
            Self::Cigar => fields[5].clone(),
            Self::Rnext => fields[6].clone(),
            Self::Pnext => fields[7].clone(),
            Self::Tlen => fields[8].clone(),
            Self::Seq => fields[9].clone(),
            Self::Qual => fields[10].clone(),
            Self::Data => fields[11..].join("\t"),
        }
    }
}

/// Formats a BAM record as SAM and returns its mandatory fields plus any auxiliary fields.
///
/// Noodles uses `header` to resolve BAM reference IDs to the textual `RNAME` and `RNEXT` fields.
pub fn sam_fields(header: &sam::Header, record_buf: &RecordBuf) -> Vec<String> {
    let mut writer = sam::io::Writer::new(Vec::new());
    writer
        .write_alignment_record(header, record_buf)
        .expect("format SAM record");
    let line = String::from_utf8(writer.into_inner()).expect("SAM record is UTF-8");
    let fields: Vec<_> = line.trim_end().split('\t').map(str::to_owned).collect();
    assert!(
        fields.len() >= 11,
        "SAM writer returned fewer than 11 mandatory fields"
    );
    fields
}

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
