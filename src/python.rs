use crate::profiles::{Ascii, Dna, Iupac};
use crate::search::{self, Match, Strand};
use crate::trace;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "sassy")]
fn sassy(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_class::<Searcher>()?;
    m.add_class::<Match>()?;
    m.add_class::<PyAllAlignmentsAtPosIter>()?;
    Ok(())
}

/// Lazy Python iterator over all alignments at a single matched end position.
///
/// Obtained from `Searcher.search_all_alignments()`. Implements the Python iterator
/// protocol (`__iter__` / `__next__`).
///
/// All yielded `Match` objects share the same `text_end`; they differ in
/// `text_start`, `cigar`, and (when `margin > 0`) `cost`.
/// Break early to avoid enumerating exponentially many paths.
#[pyclass(name = "AllAlignmentsAtPosIter")]
pub struct PyAllAlignmentsAtPosIter {
    inner: trace::AllAlignmentsAtPos,
}

#[pymethods]
impl PyAllAlignmentsAtPosIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Match> {
        self.inner.next()
    }

    #[doc = "Return the optimal (minimum) alignment cost at this end position."]
    fn optimal_cost(&self) -> i32 {
        self.inner.optimal_cost()
    }
}

enum SearcherType {
    Ascii(search::Searcher<Ascii>),
    Dna(search::Searcher<Dna>),
    Iupac(search::Searcher<Iupac>),
}

#[pyclass]
#[doc = "A reusable searcher object for fast sequence search."]
pub struct Searcher {
    searcher: SearcherType,
}

#[pyfunction]
fn features() {
    crate::test_cpu_features();
    crate::test_throughput();
}

#[pymethods]
impl Searcher {
    #[new]
    #[pyo3(signature = (alphabet, rc=true, alpha=None))]
    fn new(alphabet: &str, rc: bool, alpha: Option<f64>) -> PyResult<Self> {
        let searcher = match alphabet.to_lowercase().as_str() {
            "ascii" => {
                let s = search::Searcher::<Ascii>::new(false, alpha.map(|a| a as f32));
                SearcherType::Ascii(s)
            }
            "dna" => {
                let s = search::Searcher::<Dna>::new(rc, alpha.map(|a| a as f32));
                SearcherType::Dna(s)
            }
            "iupac" => {
                let s = search::Searcher::<Iupac>::new(rc, alpha.map(|a| a as f32));
                SearcherType::Iupac(s)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported alphabet: {}",
                    alphabet
                )));
            }
        };

        Ok(Searcher { searcher })
    }

    #[pyo3(signature = (pattern, text, k))]
    #[doc = "Search for a pattern in a text. Returns a list of Match."]
    fn search(
        &mut self,
        pattern: &Bound<'_, PyBytes>,
        text: &Bound<'_, PyBytes>,
        k: usize,
    ) -> Vec<Match> {
        // We don't let control go back to Python while we hold the slices.
        let pattern = pattern.as_bytes();
        let text = text.as_bytes();
        match &mut self.searcher {
            SearcherType::Ascii(searcher) => searcher.search(&pattern, &text, k),
            SearcherType::Dna(searcher) => searcher.search(&pattern, &text, k),
            SearcherType::Iupac(searcher) => searcher.search(&pattern, &text, k),
        }
    }

    #[pyo3(signature = (patterns, texts, k, threads, mode))]
    #[doc = "Search multiple patterns in multiple texts, using multiple threads. Returns a list of Matches."]
    fn search_many(
        &mut self,
        patterns: Vec<Bound<'_, PyBytes>>,
        texts: Vec<Bound<'_, PyBytes>>,
        k: usize,
        threads: usize,
        mode: &Bound<'_, PyString>,
    ) -> Vec<Match> {
        let patterns: Vec<&[u8]> = patterns.iter().map(|p| p.as_bytes()).collect();
        let texts: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
        // We don't let control go back to Python while we hold the slices.
        let mode = match mode.to_string_lossy().as_ref() {
            "single" => search::SearchMode::Single,
            "batch_patterns" => search::SearchMode::BatchPatterns,
            "batch_texts" => search::SearchMode::BatchTexts,
            _ => panic!(
                "Unsupported search mode. Must be one of 'single', 'batch_patterns', or 'batch_texts'"
            ),
        };
        match &mut self.searcher {
            SearcherType::Ascii(searcher) => {
                searcher.search_many(&patterns, &texts, k, threads, mode)
            }
            SearcherType::Dna(searcher) => {
                searcher.search_many(&patterns, &texts, k, threads, mode)
            }
            SearcherType::Iupac(searcher) => {
                searcher.search_many(&patterns, &texts, k, threads, mode)
            }
        }
    }

    #[pyo3(signature = (pattern, text, k, margin=0))]
    #[doc = "Search for all end positions with score <= k, returning a lazy iterator per end position. Each iterator yields alignments with cost <= optimal_cost + margin. Break early from the lazy iterator(s) to avoid exponential enumeration."]
    fn search_all_alignments(
        &mut self,
        pattern: &Bound<'_, PyBytes>,
        text: &Bound<'_, PyBytes>,
        k: usize,
        margin: usize,
    ) -> Vec<PyAllAlignmentsAtPosIter> {
        let pattern = pattern.as_bytes();
        let text = text.as_bytes();
        let groups = match &mut self.searcher {
            SearcherType::Ascii(searcher) => {
                searcher.search_all_alignments(pattern, text, k, margin)
            }
            SearcherType::Dna(searcher) => searcher.search_all_alignments(pattern, text, k, margin),
            SearcherType::Iupac(searcher) => {
                searcher.search_all_alignments(pattern, text, k, margin)
            }
        };
        groups
            .into_iter()
            .map(|g| PyAllAlignmentsAtPosIter { inner: g })
            .collect()
    }

    #[pyo3(signature = (pattern, text, k))]
    #[doc = "Search for a pattern in a text. Returns a list of Matches for *all* end positions with score <=k."]
    fn search_all(
        &mut self,
        pattern: &Bound<'_, PyBytes>,
        text: &Bound<'_, PyBytes>,
        k: usize,
    ) -> Vec<Match> {
        // We don't let control go back to Python while we hold the slices.
        let pattern = pattern.as_bytes();
        let text = text.as_bytes();
        match &mut self.searcher {
            SearcherType::Ascii(searcher) => searcher.search_all(&pattern, &text, k),
            SearcherType::Dna(searcher) => searcher.search_all(&pattern, &text, k),
            SearcherType::Iupac(searcher) => searcher.search_all(&pattern, &text, k),
        }
    }
}

#[pymethods]
impl Match {
    #[getter]
    fn pattern_idx(&self) -> usize {
        self.pattern_idx
    }

    #[getter]
    fn text_idx(&self) -> usize {
        self.text_idx
    }

    #[getter]
    fn text_start(&self) -> usize {
        self.text_start
    }

    #[getter]
    fn text_end(&self) -> usize {
        self.text_end
    }

    #[getter]
    fn pattern_start(&self) -> usize {
        self.pattern_start
    }

    #[getter]
    fn pattern_end(&self) -> usize {
        self.pattern_end
    }

    #[getter]
    fn cost(&self) -> i32 {
        self.cost
    }

    #[getter]
    fn strand(&self) -> &'static str {
        match self.strand {
            Strand::Fwd => "+",
            Strand::Rc => "-",
        }
    }

    #[getter]
    fn cigar(&self) -> String {
        self.cigar.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Match pattern_start={} text_start={} pattern_end={} text_end={} cost={} strand='{}' cigar='{}'>",
            self.pattern_start,
            self.text_start,
            self.pattern_end,
            self.text_end,
            self.cost,
            match self.strand {
                Strand::Fwd => "+",
                Strand::Rc => "-",
            },
            self.cigar.to_string()
        )
    }
}
