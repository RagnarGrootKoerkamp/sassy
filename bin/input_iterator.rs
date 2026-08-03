use crate::grep::{is_bam_path, is_sam_path};
use needletail::{FastxReader, parse_fastx_file, parse_fastx_stdin};
use noodles::{
    bam,
    sam::{self, alignment::RecordBuf},
};
use sassy::CachedRev;
use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{Arc, Mutex}, // TODO: could use parking_lot mutex - faster
};

/// Each batch of text records will be at most this size if possible.
const DEFAULT_BATCH_BYTES: usize = 1024 * 1024; // 1 MiB
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

#[derive(Debug)]
pub enum TextRecord {
    Fastx {
        id: ID,
        seq: CachedRev<Vec<u8>>,
        quality: Vec<u8>,
    },

    Sam {
        id: ID,
        seq: CachedRev<Vec<u8>>,
        record_buf: RecordBuf,
    },
}

impl TextRecord {
    pub fn id(&self) -> &str {
        match self {
            Self::Fastx { id, .. } | Self::Sam { id, .. } => id,
        }
    }

    pub fn seq(&self) -> &CachedRev<Vec<u8>> {
        match self {
            Self::Fastx { seq, .. } | Self::Sam { seq, .. } => seq,
        }
    }

    pub fn seq_mut(&mut self) -> &mut CachedRev<Vec<u8>> {
        match self {
            Self::Fastx { seq, .. } | Self::Sam { seq, .. } => seq,
        }
    }

    pub fn quality(&self) -> &[u8] {
        match self {
            Self::Fastx { quality, .. } => quality,
            Self::Sam { record_buf, .. } => record_buf.quality_scores().as_ref(),
        }
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
    pub(super) reader: InputReader,
    /// The last batch of text records.
    text_batch: Arc<Vec<TextRecord>>,
    /// The next file index.
    file_idx: usize,
    /// The next pattern-batch index.
    pat_idx: usize,
    /// The next batch id.
    batch_id: usize,
}

enum InputReader {
    Fastx(Box<dyn FastxReader + Send>),
    Bam {
        reader: bam::io::Reader<noodles::bgzf::io::Reader<File>>,
        header: Arc<noodles::sam::Header>,
    },
    Sam {
        reader: sam::io::Reader<BufReader<File>>,
        header: Arc<noodles::sam::Header>,
    },
}

impl InputReader {
    fn next_record(&mut self) -> Option<TextRecord> {
        match self {
            Self::Fastx(reader) => match reader.next() {
                Some(Ok(rec)) => Some(TextRecord::Fastx {
                    id: String::from_utf8_lossy(rec.id()).into_owned(),
                    seq: CachedRev::new(rec.seq().into_owned(), false),
                    quality: rec.qual().unwrap_or(&[]).to_vec(),
                }),
                Some(Err(e)) => panic!("Error reading FASTX record: {e}"),
                None => None,
            },
            Self::Bam { reader, header } => {
                let mut record_buf = RecordBuf::default();
                let block_size = reader
                    .read_record_buf(header, &mut record_buf)
                    .expect("Error reading BAM record");
                if block_size == 0 {
                    return None;
                }

                let id = record_buf
                    .name()
                    .map(|name| String::from_utf8_lossy(name.as_ref()).into_owned())
                    .unwrap_or_else(|| "*".to_string());
                let seq = CachedRev::new(record_buf.sequence().as_ref().to_vec(), false);
                Some(TextRecord::Sam {
                    id,
                    seq,
                    record_buf,
                })
            }
            Self::Sam { reader, header } => {
                let mut record_buf = RecordBuf::default();
                let bytes_read = reader
                    .read_record_buf(header, &mut record_buf)
                    .expect("Failed to read a SAM alignment record");
                if bytes_read == 0 {
                    return None;
                }

                let id = record_buf
                    .name()
                    .map(|name| String::from_utf8_lossy(name.as_ref()).into_owned())
                    .unwrap_or_else(|| "*".to_string());
                let seq = CachedRev::new(record_buf.sequence().as_ref().to_vec(), false);
                Some(TextRecord::Sam {
                    id,
                    seq,
                    record_buf,
                })
            }
        }
    }
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
}

fn parse_file(path: &PathBuf) -> (InputReader, Option<Arc<noodles::sam::Header>>) {
    if is_bam_path(path) {
        let mut reader = bam::io::Reader::new(
            File::open(path)
                .unwrap_or_else(|e| panic!("Failed to open BAM input `{}`: {e}", path.display())),
        );
        let header = Arc::new(reader.read_header().unwrap_or_else(|e| {
            panic!("Failed to read BAM header from `{}`: {e}", path.display())
        }));
        (
            InputReader::Bam {
                reader,
                header: header.clone(),
            },
            Some(header),
        )
    } else if is_sam_path(path) {
        let mut reader =
            sam::io::Reader::new(BufReader::new(File::open(path).unwrap_or_else(|e| {
                panic!("Failed to open SAM input `{}`: {e}", path.display())
            })));
        let header = Arc::new(reader.read_header().unwrap_or_else(|e| {
            panic!("Failed to read SAM header from `{}`: {e}", path.display())
        }));
        (
            InputReader::Sam {
                reader,
                header: header.clone(),
            },
            Some(header),
        )
    } else if path == Path::new("") || path == Path::new("-") {
        (InputReader::Fastx(parse_fastx_stdin().unwrap()), None)
    } else {
        (InputReader::Fastx(parse_fastx_file(path).unwrap()), None)
    }
}

impl<'a> InputIterator<'a> {
    /// Create a new iterator over `fasta_path`, going through `patterns`.
    /// `max_batch_bytes` (default 1MB) controls how much input text is in a batch.
    /// `max_batch_patterns` (default 64) controls how many patterns are in a batch.
    pub fn new(
        paths: &'a Vec<PathBuf>,
        patterns: &'a [PatternRecord],
        max_batch_bytes: Option<usize>,
        max_batch_patterns: Option<usize>,
        rev: bool,
    ) -> (Self, Option<Arc<noodles::sam::Header>>) {
        let (reader, header) = parse_file(&paths[0]);
        // Just empty state when we create the iterator
        let batch_pattern_limit = max_batch_patterns.unwrap_or(DEFAULT_BATCH_PATTERNS);
        let state = RecordState {
            reader,
            text_batch: Arc::new(Vec::new()),
            file_idx: 0,
            pat_idx: patterns.len() / batch_pattern_limit + 2,
            batch_id: 0,
        };
        (
            Self {
                patterns,
                paths,
                state: Mutex::new(state),
                batch_byte_limit: max_batch_bytes.unwrap_or(DEFAULT_BATCH_BYTES),
                batch_pattern_limit,
                rev,
            },
            header,
        )
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
                    match state.reader.next_record() {
                        Some(record) => break record,
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
                            state.reader = parse_file(&self.paths[state.file_idx]).0;
                            continue;
                        }
                    }
                };

                // We get the ref to the current record we have available
                let record_len = current_record.seq().text.len();
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
                    text_record.seq_mut().initialize_rev();
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
        let (iter, _) = InputIterator::new(&paths, &patterns, Some(500), None, true);

        // Pull 10 batches
        let mut batch_id = 0;
        while let Some(batch) = iter.next_batch() {
            batch_id += 1;
            // Get unique texts, and then their length sum
            let unique_texts = batch
                .1
                .2
                .iter()
                .map(|item| item.seq().text.clone())
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
