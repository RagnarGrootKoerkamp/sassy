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
