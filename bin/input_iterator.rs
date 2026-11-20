use helicase::{Config, FastxParser, HelicaseParser, ParallelFastxParser, ParserOptions};
use sassy::{CachedRev, Match};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    mpsc::{Receiver, SyncSender, sync_channel},
};
use std::thread::Scope;

/// Each batch of text records will be at most this size if possible.
const DEFAULT_BATCH_BYTES: usize = 1024 * 1024; // 1 MiB
const MAX_TEXT_CHUNK_BYTES: usize = 8 * 1024 * 1024;
/// Each batch of patterns will be at most this size.
const DEFAULT_BATCH_PATTERNS: usize = 32;

pub(super) const FASTX_CONFIG: Config = ParserOptions::default().config();
const FASTX_WITH_QUALITY: Config = ParserOptions::default().compute_quality().config();

/// Type alias for fasta record IDs.
pub type ID = String;

/// A search pattern, with ID from fasta file.
#[derive(Clone, Debug)]
pub struct PatternRecord {
    pub id: ID,
    pub seq: Vec<u8>,
}

/// A text to be searched, with ID from fasta file.
/// It overlaps `overlap` bases with both a preceding and following text chunk.
#[derive(Debug)]
pub struct TextRecord {
    pub id: ID,
    pub seq: CachedRev<Vec<u8>>,
    pub quality: Vec<u8>,
    /// Offset of this chunk in the original FASTA record.
    pub record_offset: usize,
    /// First position in this chunk belonging to its non-overlapping region.
    pub first_real_position: usize,
    /// Last exclusive endpoint owned by this chunk, or the chunk length if last.
    pub last_real_position: usize,
}

impl TextRecord {
    /// A match is owned by the chunk containing its rightmost forward-text base.
    pub fn owns_match(&self, m: &Match) -> bool {
        // The first chunk also owns matches ending at position zero.
        (self.record_offset == 0 || m.text_end > self.first_real_position)
            && m.text_end <= self.last_real_position
    }
}

/// A batch of text records of around 1MB by default.
pub type TextBatch = Arc<Vec<TextRecord>>;

/// A batch of alignment tasks, with total text size around `DEFAULT_BATCH_BYTES`.
/// This avoids lock contention of sending too small items across threads.
///
/// Each task represents searching _all_ patterns against _all_ text records.
pub type TaskBatch<'a> = (&'a Path, &'a [PatternRecord], TextBatch);

/// Input batches produced by one reader thread and consumed by search workers.
pub struct BufferedInputIterator<'a> {
    sender: Mutex<Option<SyncSender<(usize, TaskBatch<'a>)>>>,
    receiver: Mutex<Receiver<(usize, TaskBatch<'a>)>>,
}

impl<'a> BufferedInputIterator<'a> {
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = sync_channel(capacity);
        Self {
            sender: Mutex::new(Some(sender)),
            receiver: Mutex::new(receiver),
        }
    }

    pub fn start_reader<'scope, 'env>(
        &self,
        scope: &'scope Scope<'scope, 'env>,
        input: &'scope InputIterator<'a>,
    ) where
        'a: 'scope,
    {
        let sender = self
            .sender
            .lock()
            .unwrap()
            .take()
            .expect("buffered input reader already started");
        scope.spawn(move || {
            while let Some(batch) = input.next_batch() {
                if sender.send(batch).is_err() {
                    break;
                }
            }
        });
    }

    pub fn next_batch(&self) -> Option<(usize, TaskBatch<'a>)> {
        self.receiver.lock().unwrap().recv().ok()
    }
}

struct RecordState {
    /// The current fasta reader.
    reader: TextReader,
    /// The last batch of text records.
    text_batch: Arc<Vec<TextRecord>>,
    /// The next file index.
    file_idx: usize,
    /// The next pattern-batch index.
    pat_idx: usize,
    /// The next batch id.
    batch_id: usize,
    pending_record: Option<PendingRecord>,
}

/// The current record, when it's being emitted in chunks.
struct PendingRecord {
    id: ID,
    seq: Vec<u8>,
    quality: Vec<u8>,
    next_chunk_start: usize,
}

/// Thread-safe iterator giving *batches* of (pattern, text) pairs.
/// Each batch searches at least `batch_byte_limit` bytes of text.
///
/// Created using `TaskIterator::new` from a list of patterns and a path to a Fasta file to be searched.
pub struct InputIterator<'a> {
    patterns: &'a [PatternRecord],
    paths: &'a Vec<PathBuf>,
    state: Mutex<RecordState>,
    batch_byte_limit: usize,
    batch_pattern_limit: usize,
    rev: bool,
    overlap: usize,
    max_text_chunk_bytes: usize,
    filter_mode: bool,
}

/// Helicase-based parallel input reader.
struct TextReader {
    batches: Receiver<Vec<PendingRecord>>,
    records: std::vec::IntoIter<PendingRecord>,
    worker: Option<std::thread::JoinHandle<Result<(), helicase::parser::Error>>>,
}

impl TextReader {
    fn open(path: &Path, read_quality: bool) -> Self {
        if read_quality {
            Self::open_parser::<FASTX_WITH_QUALITY, _>(path, |parser| {
                parser.get_quality_owned().unwrap_or_default()
            })
        } else {
            Self::open_parser::<FASTX_CONFIG, _>(path, |_| Vec::new())
        }
    }

    fn open_parser<const CONFIG: Config, Q>(path: &Path, quality: Q) -> Self
    where
        Q: Fn(&mut FastxParser<'_, CONFIG>) -> Vec<u8> + Send + Sync + 'static,
    {
        let workers =
            std::thread::available_parallelism().map_or(1, |cpus| (cpus.get() / 4).max(1));
        let parser = if path == Path::new("") || path == Path::new("-") {
            ParallelFastxParser::<CONFIG>::from_reader(io::stdin())
        } else if path.is_file() {
            ParallelFastxParser::<CONFIG>::from_file_mmap(path).unwrap()
        } else {
            ParallelFastxParser::<CONFIG>::from_reader(std::fs::File::open(path).unwrap())
        }
        .threads(workers);
        let (sender, batches) = sync_channel(2);
        let worker = std::thread::spawn(move || {
            let send = |records| {
                sender
                    .send(records)
                    .map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))
            };
            parser.for_each(|mut parser| {
                let mut records = Vec::new();
                let mut bytes = 0;
                while let Some(event) = parser.next() {
                    event?;
                    let seq = parser.get_dna_string_owned();
                    bytes += seq.len().max(1);
                    records.push(PendingRecord {
                        id: String::from_utf8(parser.get_header_owned()).unwrap(),
                        seq,
                        quality: quality(&mut parser),
                        next_chunk_start: 0,
                    });
                    if bytes >= DEFAULT_BATCH_BYTES {
                        send(std::mem::take(&mut records))?;
                        bytes = 0;
                    }
                }
                if !records.is_empty() {
                    send(records)?;
                }
                Ok(())
            })
        });
        Self {
            batches,
            records: Vec::new().into_iter(),
            worker: Some(worker),
        }
    }

    fn next_record(&mut self) -> Option<PendingRecord> {
        loop {
            if let Some(record) = self.records.next() {
                return Some(record);
            }
            match self.batches.recv() {
                Ok(records) => self.records = records.into_iter(),
                Err(_) => {
                    if let Some(worker) = self.worker.take() {
                        worker.join().unwrap().unwrap();
                    }
                    return None;
                }
            }
        }
    }
}

impl<'a> InputIterator<'a> {
    /// Create a new iterator over `fasta_path`, going through `patterns`.
    /// `max_batch_bytes` (default 1MB) controls the target batch size.
    /// `max_batch_patterns` (default 32) controls how many patterns are in a batch.
    /// `overlap` is the maximum reference span of a match, including edits.
    /// `filter_mode` keeps records whole and reads FASTQ qualities for output.
    pub fn new(
        paths: &'a Vec<PathBuf>,
        patterns: &'a [PatternRecord],
        max_batch_bytes: Option<usize>,
        max_batch_patterns: Option<usize>,
        rev: bool,
        overlap: usize,
        filter_mode: bool,
    ) -> Self {
        let chunk_size = if filter_mode {
            usize::MAX
        } else {
            MAX_TEXT_CHUNK_BYTES
        };
        Self::new_with_chunk_size(
            paths,
            patterns,
            max_batch_bytes,
            max_batch_patterns,
            rev,
            overlap,
            chunk_size,
            filter_mode,
        )
    }

    /// Construct an iterator with an explicit maximum size for each text chunk.
    /// Use `usize::MAX` to disable chunking.
    pub fn new_with_chunk_size(
        paths: &'a Vec<PathBuf>,
        patterns: &'a [PatternRecord],
        max_batch_bytes: Option<usize>,
        max_batch_patterns: Option<usize>,
        rev: bool,
        overlap: usize,
        max_text_chunk_bytes: usize,
        filter_mode: bool,
    ) -> Self {
        assert!(max_text_chunk_bytes > overlap.saturating_mul(2));
        let reader = TextReader::open(&paths[0], filter_mode);
        // Just empty state when we create the iterator
        let batch_pattern_limit = max_batch_patterns.unwrap_or(DEFAULT_BATCH_PATTERNS);
        let state = RecordState {
            reader,
            text_batch: Arc::new(Vec::new()),
            file_idx: 0,
            pat_idx: patterns.len() / batch_pattern_limit + 2,
            batch_id: 0,
            pending_record: None,
        };
        Self {
            patterns,
            paths,
            state: Mutex::new(state),
            batch_byte_limit: max_batch_bytes.unwrap_or(DEFAULT_BATCH_BYTES),
            batch_pattern_limit,
            rev,
            overlap,
            max_text_chunk_bytes,
            filter_mode,
        }
    }

    /// Get the next batch, or returns None when done.
    pub fn next_batch(&self) -> Option<(usize, TaskBatch<'a>)> {
        let mut state = self.state.lock().unwrap();
        let batch_id = state.batch_id;
        state.batch_id += 1;
        // Read the next batch of text.
        if state.pat_idx * self.batch_pattern_limit >= self.patterns.len() {
            if state.file_idx >= self.paths.len() {
                log::debug!("No more files to read, returning None");
                return None;
            }

            let mut text_batch = Vec::new();
            let mut bytes_in_batch = 0usize;

            // Effectively this gets a record, add all patterns, then tries
            // to push another text record, if possible. This way texts
            // are only 'read' from the Fasta file once.

            'outer: loop {
                // Make sure we have a current record, just so we can unwrap
                let current_record = loop {
                    if let Some(pending) = state.pending_record.as_mut() {
                        if pending.seq.len() <= self.max_text_chunk_bytes {
                            let pending = state.pending_record.take().unwrap();
                            let record_len = pending.seq.len();
                            break TextRecord {
                                id: pending.id,
                                seq: CachedRev::new(pending.seq, false),
                                quality: pending.quality,
                                record_offset: 0,
                                first_real_position: 0,
                                last_real_position: record_len,
                            };
                        }

                        let offset = pending.next_chunk_start;
                        let end = offset
                            .saturating_add(self.max_text_chunk_bytes)
                            .min(pending.seq.len());
                        let is_first_chunk = offset == 0;
                        let is_last_chunk = end == pending.seq.len();
                        let first_real_position = if is_first_chunk { 0 } else { self.overlap };
                        let chunk_len = end - offset;
                        let last_real_position = if is_last_chunk {
                            chunk_len
                        } else {
                            chunk_len - self.overlap
                        };
                        let chunk = pending.seq[offset..end].to_vec();
                        let quality = if pending.quality.is_empty() {
                            Vec::new()
                        } else {
                            pending.quality[offset..end].to_vec()
                        };
                        let text_record = TextRecord {
                            id: pending.id.clone(),
                            seq: CachedRev::new(chunk, false),
                            quality,
                            record_offset: offset,
                            first_real_position,
                            last_real_position,
                        };
                        if is_last_chunk {
                            state.pending_record = None;
                        } else {
                            pending.next_chunk_start = end - 2 * self.overlap;
                        }
                        break text_record;
                    }

                    match state.reader.next_record() {
                        Some(record) => {
                            state.pending_record = Some(record);
                        }
                        None => {
                            // Return last batch for the current file.
                            if !text_batch.is_empty() {
                                break 'outer;
                            }
                            // Reached end of reader, now we advance to next file (or quit when no files left)
                            state.file_idx += 1;
                            let end_of_files = state.file_idx >= self.paths.len();
                            if end_of_files {
                                log::debug!("No more files to read, returning None");
                                return None;
                            }
                            // Start reading next file.
                            state.reader =
                                TextReader::open(&self.paths[state.file_idx], self.filter_mode);
                            continue;
                        }
                    }
                };

                // We get the ref to the current record we have available
                let record_len = current_record.seq.text.len();
                bytes_in_batch += record_len;

                log::trace!(
                    "Push record of len {record_len:>5} total len {bytes_in_batch:>8} limit {}",
                    self.batch_byte_limit
                );

                // Add next pattern
                text_batch.push(current_record);

                // When exceeding batch limit, return current batch.
                if bytes_in_batch >= self.batch_byte_limit {
                    log::debug!("New batch of {} kB", bytes_in_batch / 1024);

                    break;
                }
            }

            if self.rev {
                for text_record in &mut text_batch {
                    text_record.seq.initialize_rev();
                }
            }

            state.text_batch = Arc::new(text_batch);
            state.pat_idx = 0;
        }

        // Hand out the next batch of patterns for the current text batch.
        let start = state.pat_idx * self.batch_pattern_limit;
        let end = (start + self.batch_pattern_limit).min(self.patterns.len());
        state.pat_idx += 1;
        log::debug!(
            "Batch {batch_id:>3}: {} seqs {} patterns",
            state.text_batch.len(),
            end - start,
        );
        Some((
            batch_id,
            (
                &self.paths[state.file_idx],
                &self.patterns[start..end],
                state.text_batch.clone(),
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use sassy::{Searcher, profiles::Dna};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parallel_reader_preserves_fastq_records_and_quality() {
        let mut file = NamedTempFile::new().unwrap();
        let sequence = "ACGT".repeat(64);
        let qualities: Vec<_> = (0..128)
            .map(|idx| {
                char::from(b'!' + (idx % 35) as u8)
                    .to_string()
                    .repeat(sequence.len())
            })
            .collect();
        // Helicase uses the serial parser for files smaller than 32 KiB.
        for (idx, quality) in qualities.iter().enumerate() {
            writeln!(file, "@read {idx}\n{sequence}\n+\n{quality}").unwrap();
        }

        for read_quality in [false, true] {
            let mut reader = TextReader::open(file.path(), read_quality);
            let mut records: Vec<_> = std::iter::from_fn(|| reader.next_record()).collect();
            records.sort_by_key(|record| record.id[5..].parse::<usize>().unwrap());
            assert_eq!(records.len(), qualities.len());
            for (idx, (record, quality)) in records.iter().zip(&qualities).enumerate() {
                assert_eq!(record.id, format!("read {idx}"));
                assert_eq!(record.seq, sequence.as_bytes());
                let expected = if read_quality {
                    quality.as_bytes()
                } else {
                    &[]
                };
                assert_eq!(record.quality, expected);
            }
        }
    }

    fn random_dan_seq(len: usize) -> Vec<u8> {
        let mut rng = rand::rng();
        let mut seq = Vec::new();
        let bases = b"ACGT";
        for _ in 0..len {
            seq.push(bases[rng.random_range(0..bases.len())]);
        }
        seq
    }

    #[test]
    fn test_record_iterator() {
        // Create 100 different random sequences with length of 100-1000
        let mut rng = rand::rng();
        let mut seqs = Vec::new();
        for _ in 0..100 {
            seqs.push(random_dan_seq(rng.random_range(100..1000)));
        }

        // Create a temporary file to write the fasta file to
        let mut file = NamedTempFile::new().unwrap();
        for (i, seq) in seqs.into_iter().enumerate() {
            file.write_all(format!(">seq_{}\n{}\n", i, String::from_utf8(seq).unwrap()).as_bytes())
                .unwrap();
        }
        file.flush().unwrap();

        // Create 10 different random patterns
        let mut patterns = Vec::new();
        for i in 0..10 {
            patterns.push(PatternRecord {
                id: format!("pattern_{}", i),
                seq: random_dan_seq(rng.random_range(250..1000)),
            });
        }

        // Create the iterator
        let paths = vec![file.path().to_path_buf()];
        let overlap = patterns.iter().map(|p| p.seq.len()).max().unwrap_or(0);
        let iter = InputIterator::new(&paths, &patterns, Some(500), None, true, overlap, false);

        // Pull 10 batches
        let mut batch_id = 0;
        while let Some(batch) = iter.next_batch() {
            batch_id += 1;
            // Get unique texts, and then their length sum
            let unique_texts = batch
                .1
                .2
                .iter()
                .map(|item| item.seq.text.clone())
                .collect::<std::collections::HashSet<_>>();
            let text_len = unique_texts.iter().map(|text| text.len()).sum::<usize>();
            let n_patterns = batch
                .1
                .1
                .iter()
                .map(|item| item.id.clone())
                .collect::<std::collections::HashSet<_>>()
                .len();
            let n_texts = unique_texts.len();
            println!(
                "Batch {batch_id} (tot_size: {text_len}, n_texts: {n_texts}): {n_patterns} patterns"
            );
        }
        drop(file);
    }

    #[test]
    fn fuzz_short_chunks_match_unchunked_search() {
        fn search(
            path: &Path,
            pattern: &PatternRecord,
            k: usize,
            chunk_size: usize,
            overlap: usize,
        ) -> Vec<(usize, usize, i32, u8, String)> {
            let paths = vec![path.to_path_buf()];
            let patterns = std::slice::from_ref(pattern);
            let iter = InputIterator::new_with_chunk_size(
                &paths, patterns, None, None, true, overlap, chunk_size, false,
            );
            let mut searcher = Searcher::<Dna>::new_rc();
            let mut found = Vec::new();
            while let Some((_, (_, _, texts))) = iter.next_batch() {
                for text in texts.iter() {
                    let mut matches = searcher.search(&pattern.seq, &text.seq, k);
                    matches.retain(|m| text.owns_match(m));
                    found.extend(matches.into_iter().map(|m| {
                        (
                            text.record_offset + m.text_start,
                            text.record_offset + m.text_end,
                            m.cost,
                            m.strand as u8,
                            m.cigar.to_string(),
                        )
                    }));
                }
            }
            found.sort();
            found
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a55_79);
        for case in 0..100 {
            let pattern_len = rng.random_range(5..=19);
            let pattern_seq: Vec<u8> = (0..pattern_len)
                .map(|_| *b"ACGT".get(rng.random_range(0..4)).unwrap())
                .collect();
            let k = rng.random_range(0..=2);
            let overlap = pattern_len + k;
            let chunk_size = rng.random_range((2 * overlap + 1)..=(2 * overlap + 30));
            let mut sequence: Vec<u8> = (0..250)
                .map(|_| *b"ACGT".get(rng.random_range(0..4)).unwrap())
                .collect();
            let pos = rng.random_range(0..=(sequence.len() - pattern_len));
            sequence[pos..pos + pattern_len].copy_from_slice(&pattern_seq);

            let mut file = NamedTempFile::new().unwrap();
            writeln!(file, ">case_{case}").unwrap();
            file.write_all(&sequence).unwrap();
            writeln!(file).unwrap();
            file.flush().unwrap();

            let pattern = PatternRecord {
                id: "p".into(),
                seq: pattern_seq,
            };
            let chunked = search(file.path(), &pattern, k, chunk_size, overlap);
            let unchunked = search(file.path(), &pattern, k, usize::MAX, overlap);
            assert_eq!(
                chunked, unchunked,
                "fuzz case {case}, chunk size {chunk_size}"
            );
        }

        let pattern = PatternRecord {
            id: "boundary".into(),
            seq: b"ACGCATCACAGCACATCAG".to_vec(),
        };
        let reverse_complement = b"CTGATGTGCTGTGATGCGT";
        let mut sequence: Vec<u8> = (0..500).map(|i| b"ACGT"[i % 4]).collect();
        for (position, value) in [
            (28, pattern.seq.as_slice()),
            (49, pattern.seq.as_slice()),
            (77, reverse_complement.as_slice()),
            (101, pattern.seq.as_slice()),
            (149, reverse_complement.as_slice()),
            (199, pattern.seq.as_slice()),
            (248, reverse_complement.as_slice()),
            (299, pattern.seq.as_slice()),
            (347, reverse_complement.as_slice()),
            (401, pattern.seq.as_slice()),
            (450, reverse_complement.as_slice()),
        ] {
            sequence[position..position + value.len()].copy_from_slice(value);
        }
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, ">boundary").unwrap();
        file.write_all(&sequence).unwrap();
        writeln!(file).unwrap();
        file.flush().unwrap();
        let overlap = pattern.seq.len() + 2;
        assert_eq!(
            search(file.path(), &pattern, 2, 50, overlap),
            search(file.path(), &pattern, 2, usize::MAX, overlap),
            "fixed chunk-boundary regression"
        );
    }

    #[test]
    fn chunk_ownership_uses_forward_text_end_for_both_strands() {
        let text = TextRecord {
            id: "record".into(),
            seq: CachedRev::new(vec![b'A'; 50], false),
            quality: Vec::new(),
            record_offset: 30,
            first_real_position: 20,
            last_real_position: 40,
        };
        let m = |strand, start, end, pattern_start, pattern_end| Match {
            pattern_idx: 0,
            text_idx: 0,
            text_start: start,
            text_end: end,
            pattern_start,
            pattern_end,
            cost: 0,
            strand,
            cigar: pa_types::Cigar::default(),
        };

        assert!(!text.owns_match(&m(sassy::Strand::Fwd, 10, 20, 0, 10)));
        assert!(text.owns_match(&m(sassy::Strand::Fwd, 10, 21, 0, 10)));
        assert!(!text.owns_match(&m(sassy::Strand::Rc, 19, 20, 0, 10)));
        assert!(text.owns_match(&m(sassy::Strand::Rc, 19, 21, 0, 10)));
        assert!(text.owns_match(&m(sassy::Strand::Fwd, 30, 40, 0, 10)));
        assert!(!text.owns_match(&m(sassy::Strand::Fwd, 30, 41, 0, 10)));

        assert!(!text.owns_match(&m(sassy::Strand::Fwd, 40, 50, 0, 9)));
        assert!(!text.owns_match(&m(sassy::Strand::Rc, 40, 50, 1, 10)));
    }
}
