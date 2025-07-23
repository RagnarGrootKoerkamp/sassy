use crate::profiles::{Ascii, Dna, Iupac};
use crate::search::{self, Match, Strand};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "sassy")]
fn sassy(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_class::<Searcher>()?;
    Ok(())
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
}

#[pymethods]
impl Match {
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
