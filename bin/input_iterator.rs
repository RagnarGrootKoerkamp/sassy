use needletail::{FastxReader, parse_fastx_file, parse_fastx_stdin};
use sassy::{CachedRev, Match};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex}; //Todo: could use parking_lot mutex - faster

/// Each batch of text records will be at most this size if possible.
const DEFAULT_BATCH_BYTES: usize = 1024 * 1024; // 1 MiB
const MAX_TEXT_CHUNK_BYTES: usize = 1024 * 1024;
/// Each batch of patterns will be at most this size.
const DEFAULT_BATCH_PATTERNS: usize = 64;

/// Type alias for fasta record IDs.
pub type ID = String;

/// A search pattern, with ID from fasta file.
#[derive(Clone, Debug)]
pub struct PatternRecord {
    pub id: ID,
    pub seq: Vec<u8>,
}

/// A text to be searched, with ID from fasta file.
/// TODO: Reduce the number of allocations here.
#[derive(Debug)]
pub struct TextRecord {
    pub id: ID,
    pub seq: CachedRev<Vec<u8>>,
    pub quality: Vec<u8>,
    /// Offset of this chunk in the original FASTA record.
    pub record_offset: usize,
    /// First position in this chunk belonging to its non-overlapping region.
    pub first_real_position: usize,
    pub is_first_chunk: bool,
    pub is_last_chunk: bool,
}

impl TextRecord {
    /// A match is owned by the chunk that contains its last character.
    pub fn owns_match(&self, m: &Match, pattern: &PatternRecord) -> bool {
        let text_len = self.seq.text.len();
        // `text_end` is exclusive, so a match ending exactly at the first real
        // position contains no real base from this chunk and belongs to the prior one.
        if !self.is_first_chunk && m.text_end <= self.first_real_position {
            return false;
        }
        // Right overhang is represented by a match ending at the text boundary
        // without consuming the full pattern.
        if !self.is_last_chunk && m.text_end >= text_len && m.pattern_end < pattern.seq.len() {
            return false;
        }
        true
    }
}

/// A batch of text records of around 1MB by default.
pub type TextBatch = Arc<Vec<TextRecord>>;

/// A batch of alignment tasks, with total text size around `DEFAULT_BATCH_BYTES`.
/// This avoids lock contention of sending too small items across threads.
///
/// Each task represents searching _all_ patterns against _all_ text records.
pub type TaskBatch<'a> = (&'a Path, &'a [PatternRecord], TextBatch);

struct RecordState {
    /// The current fasta reader.
    reader: Box<dyn FastxReader + Send>,
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
}

fn parse_file(path: &PathBuf) -> Box<dyn FastxReader> {
    if path == Path::new("") || path == Path::new("-") {
        parse_fastx_stdin().unwrap()
    } else {
        parse_fastx_file(path).unwrap()
    }
}

impl<'a> InputIterator<'a> {
    /// Create a new iterator over `fasta_path`, going through `patterns`.
    /// `max_batch_bytes` (default 1MB) controls the target batch size.
    /// `max_batch_patterns` (default 64) controls how many patterns are in a batch.
    /// `overlap` is the maximum reference span of a match, including edits.
    pub fn new(
        paths: &'a Vec<PathBuf>,
        patterns: &'a [PatternRecord],
        max_batch_bytes: Option<usize>,
        max_batch_patterns: Option<usize>,
        rev: bool,
        overlap: usize,
        chunk_long_records: bool,
    ) -> Self {
        let max_text_chunk_bytes = if chunk_long_records {
            assert!(
                overlap < MAX_TEXT_CHUNK_BYTES,
                "match overlap must be smaller than the 1 MiB text chunk size"
            );
            MAX_TEXT_CHUNK_BYTES
        } else {
            usize::MAX
        };
        let reader = parse_file(&paths[0]);
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
                            break TextRecord {
                                id: pending.id,
                                seq: CachedRev::new(pending.seq, false),
                                quality: pending.quality,
                                record_offset: 0,
                                first_real_position: 0,
                                is_first_chunk: true,
                                is_last_chunk: true,
                            };
                        }

                        let offset = pending.next_chunk_start;
                        let end = offset
                            .saturating_add(self.max_text_chunk_bytes)
                            .min(pending.seq.len());
                        let is_first_chunk = offset == 0;
                        let is_last_chunk = end == pending.seq.len();
                        let first_real_position = if is_first_chunk { 0 } else { self.overlap };
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
                            is_first_chunk,
                            is_last_chunk,
                        };
                        if is_last_chunk {
                            state.pending_record = None;
                        } else {
                            pending.next_chunk_start = end - self.overlap;
                        }
                        break text_record;
                    }

                    match state.reader.next() {
                        Some(Ok(rec)) => {
                            state.pending_record = Some(PendingRecord {
                                id: String::from_utf8(rec.id().to_vec()).unwrap(),
                                seq: rec.seq().into_owned(),
                                quality: rec.qual().unwrap_or(&[]).to_vec(),
                                next_chunk_start: 0,
                            });
                        }
                        Some(Err(e)) => panic!("Error reading FASTA record: {e}"),
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
                            state.reader = parse_file(&self.paths[state.file_idx]);
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
    use rand::RngExt;
    use std::io::Write;
    use tempfile::NamedTempFile;

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
        let iter = InputIterator::new(&paths, &patterns, Some(500), None, true, overlap, true);

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
}
