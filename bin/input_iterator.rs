use needletail::{FastxReader, parse_fastx_file};
use sassy::CachedRev;
use std::path::Path;
use std::sync::{Arc, Mutex}; //Todo: could use parking_lot mutex - faster

/// Each batch of text records will be at most this size if possible.
const DEFAULT_BATCH_BYTES: usize = 256 * 1024; // 256 KB

/// Type alias for fasta record IDs.
pub type ID = String;

/// A search pattern, with ID from fasta file.
#[derive(Clone, Debug)]
pub struct PatternRecord {
    pub id: ID,
    pub seq: Vec<u8>,
}

/// A text to be searched, with ID from fasta file.
#[derive(Debug)]
pub struct TextRecord {
    pub id: ID,
    pub seq: CachedRev<Vec<u8>>,
}

/// A single alignment task, consisting of a pattern and text.
#[derive(Clone, Debug)]
pub struct Task<'a> {
    pub pattern: &'a PatternRecord,
    pub text: Arc<TextRecord>,
}

/// A batch of alignment tasks, with total text size around `DEFAULT_BATCH_BYTES`.
/// This avoids lock contention of sending too small items across threads.
pub type TaskBatch<'a> = Vec<Task<'a>>;

struct RecordState {
    /// The fasta reader.
    reader: Box<dyn FastxReader + Send>,
    /// Current text record, that can be send to multiple threads.
    current_record: Option<Arc<TextRecord>>,
    /// Index of the next pattern for the current record.
    next_pattern_idx: usize,
}

/// Thread-safe iterator giving *batches* of (pattern, text) pairs.
/// Each batch searches at least `batch_byte_limit` bytes of text.
///
/// Created using `TaskIterator::new` from a list of patterns and a path to a Fasta file to be searched.
pub struct InputIterator<'a> {
    patterns: &'a [PatternRecord],
    state: Mutex<RecordState>,
    batch_byte_limit: usize,
    rev: bool,
}

impl<'a> InputIterator<'a> {
    /// Create a new iterator over `fasta_path`, going through `patterns`.
    /// `max_batch_bytes` controls how many texts are bundled together.
    pub fn new<P: AsRef<Path>>(
        fasta_path: P,
        patterns: &'a [PatternRecord],
        max_batch_bytes: Option<usize>,
        rev: bool,
    ) -> Self {
        let reader = parse_fastx_file(fasta_path).expect("valid fasta");
        // Just empty state when we create the iterator
        let state = RecordState {
            reader,
            next_pattern_idx: 0,
            current_record: None,
        };
        Self {
            patterns,
            state: Mutex::new(state),
            batch_byte_limit: max_batch_bytes.unwrap_or(DEFAULT_BATCH_BYTES),
            rev,
        }
    }

    /// Get the next batch, or returns None when done.
    pub fn next_batch(&self) -> Option<TaskBatch<'a>> {
        let mut state = self.state.lock().unwrap();
        let mut batch: TaskBatch<'a> = Vec::new();
        let mut bytes_in_batch = 0usize;

        // Effectively this gets a record, add all patterns, then tries
        // to push another text record, if possible. This way texts
        // are only 'read' from the Fasta file once.

        loop {
            // Make sure we have a current record, just so we can unwrap
            if state.current_record.is_none() {
                match state.reader.next() {
                    Some(Ok(rec)) => {
                        let id = String::from_utf8(rec.id().to_vec()).unwrap().to_string();
                        let seq = rec.seq().into_owned();
                        let static_text = CachedRev::new(seq, self.rev);
                        state.current_record = Some(Arc::new(TextRecord {
                            id,
                            seq: static_text,
                        }));
                        state.next_pattern_idx = 0; // fresh record -> pattern index reset
                    }
                    Some(Err(e)) => panic!("Error reading FASTA record: {e}"),
                    None => {
                        // Done, reached end
                        return if batch.is_empty() { None } else { Some(batch) };
                    }
                }
            }

            // We get the ref to the current record we have available
            let record = state.current_record.as_ref().unwrap().clone();
            let record_len = record.seq.text.len();

            // If no space left for next record, we return current batch
            if !batch.is_empty() && bytes_in_batch + record_len > self.batch_byte_limit {
                break; // return current batch, keep state for next call
            }

            // Add next pattern
            let pattern = &self.patterns[state.next_pattern_idx];
            let task = Task {
                pattern,
                text: record,
            };
            batch.push(task);
            bytes_in_batch += record_len;

            // Advance pattern index, but if we dont have more patterns we reset to empty
            // record so we pull new sequence in next batch (line 74)
            state.next_pattern_idx += 1;
            if state.next_pattern_idx == self.patterns.len() {
                state.current_record = None;
            }
        }

        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
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
        let iter = InputIterator::new(file.path(), &patterns, Some(500), true);

        // Pull 10 batches
        let mut batch_id = 0;
        while let Some(batch) = iter.next_batch() {
            batch_id += 1;
            // Get unique texts, and then their length sum
            let unique_texts = batch
                .iter()
                .map(|item| item.text.seq.text.clone())
                .collect::<std::collections::HashSet<_>>();
            let text_len = unique_texts.iter().map(|text| text.len()).sum::<usize>();
            let n_patterns = batch
                .iter()
                .map(|item| item.pattern.id.clone())
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
