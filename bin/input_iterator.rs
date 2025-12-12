use paraseq::fastx::RefRecord;
use paraseq::prelude::ParallelReader;
use sassy::CachedRev;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize; //Todo: could use parking_lot mutex - faster

/// Each batch of text records will be at most this size if possible.
const DEFAULT_BATCH_SIZE: usize = 1024 * 1024; // 1 MiB

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
}

/// Thread-safe iterator giving *batches* of (pattern, text) pairs.
/// Each batch searches at least `batch_byte_limit` bytes of text.
///
/// Created using `TaskIterator::new` from a list of patterns and a path to a Fasta file to be searched.
pub struct InputIterator<'a> {
    patterns: &'a [PatternRecord],
    paths: &'a Vec<PathBuf>,
    batch_size: usize,

    batch_id: AtomicUsize,
}

impl<'a> InputIterator<'a> {
    /// Create a new iterator over `fasta_path`, going through `patterns`.
    /// `max_batch_bytes` controls the amount of text to be searched per batch.
    pub fn new(
        paths: &'a Vec<PathBuf>,
        patterns: &'a [PatternRecord],
        batch_size: Option<usize>,
    ) -> Self {
        Self {
            patterns,
            paths,
            batch_id: AtomicUsize::new(0),
            batch_size: batch_size.unwrap_or(DEFAULT_BATCH_SIZE),
        }
    }

    /// Get the next batch, or returns None when done.
    pub fn process(
        &mut self,
        num_threads: usize,
        f: impl FnMut(usize, &'a Path, &'a [PatternRecord], &mut dyn Iterator<Item = RefRecord>)
        + Clone
        + Send,
    ) {
        for path in self.paths {
            let mut reader = paraseq::fastx::Reader::from_path(path).unwrap();
            reader.update_batch_size_in_bp(self.batch_size).unwrap();
            let batch_id = &self.batch_id;
            let patterns = &self.patterns;
            let mut f = f.clone();
            let mut processor = move |record_batch: &mut dyn Iterator<Item = RefRecord>| {
                let batch_id = batch_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                f(batch_id, path, patterns, record_batch);
                Ok(())
            };

            reader
                .process_parallel(&mut processor, num_threads)
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paraseq::Record;
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
        let paths = vec![file.path().to_path_buf()];
        let mut input_iterator = InputIterator::new(&paths, &patterns, Some(500));

        input_iterator.process(1, |batch_id, _path, patterns, record_iter| {
            // Get unique texts, and then their length sum
            let unique_texts = record_iter
                .map(|record| record.seq().into_owned())
                .collect::<std::collections::HashSet<_>>();
            let text_len = unique_texts.iter().map(|text| text.len()).sum::<usize>();
            let n_patterns = patterns
                .iter()
                .map(|item| item.id.clone())
                .collect::<std::collections::HashSet<_>>()
                .len();
            let n_texts = unique_texts.len();
            println!(
                "Batch {batch_id} (tot_size: {text_len}, n_texts: {n_texts}): {n_patterns} patterns"
            );
        })
    }
}
