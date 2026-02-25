use crate::sassy1::edlib_bench::sim_data::Alphabet;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct GridConfig {
    pub query_lengths: Vec<usize>,
    pub text_lengths: Vec<usize>,
    pub k: Vec<f32>,
    pub matches: Vec<f32>,
    pub bench_iter: Vec<usize>,
    pub alphabet: Vec<Alphabet>,
    pub profile: Vec<String>,
    pub rc: Vec<String>,
    pub edlib: bool,
    pub verbose: bool,
}

#[derive(Clone, Debug)]
pub struct ParamSet<'a> {
    pub query_length: usize,
    pub text_length: usize,
    pub k: usize,
    pub matches: usize,
    pub max_edits: usize,
    pub bench_iter: usize,
    pub alphabet: Alphabet,
    pub profile: &'a str,
    pub rc: &'a str,
    pub edlib: bool,
    pub verbose: bool,
}

impl GridConfig {
    pub fn output_file(&self) -> String {
        format!(
            "data/match_frac_{}_k_{}.csv",
            self.matches
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("_"),
            self.k
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("_")
        )
    }

    /// Returns an iterator over all parameter combinations as tuples.
    pub fn all_combinations<'a>(&'a self) -> impl Iterator<Item = ParamSet<'a>> + 'a {
        self.query_lengths.iter().flat_map(move |&ql| {
            self.text_lengths.iter().flat_map(move |&tl| {
                self.k.iter().flat_map(move |&k| {
                    self.matches.iter().flat_map(move |&mf| {
                        self.bench_iter.iter().flat_map(move |&bi| {
                            self.alphabet.iter().flat_map(move |&a| {
                                self.profile
                                    .iter()
                                    .flat_map(move |p| {
                                        let k = if k < 1.0 {
                                            (k * ql as f32).round() as usize
                                        } else {
                                            k as usize
                                        };
                                        let matches = if mf < 1.0 {
                                            (mf * tl as f32).round() as usize
                                        } else {
                                            mf as usize
                                        };
                                        // Only allow matching profile/alphabet pairs
                                        self.rc.iter().map(move |rc| ParamSet {
                                            query_length: ql,
                                            text_length: tl,
                                            k,
                                            matches,
                                            max_edits: k,
                                            bench_iter: bi,
                                            alphabet: a,
                                            profile: p.as_str(),
                                            rc: rc.as_str(),
                                            edlib: self.edlib,
                                            verbose: self.verbose,
                                        })
                                    })
                                    // Avoid cases with accidental matches.
                                    .filter(|param| param.query_length > 3 * param.k)
                                    .filter(|param| param.text_length > param.query_length)
                                    .filter(|param| {
                                        (param.alphabet == Alphabet::Dna && param.profile == "dna")
                                            || (param.alphabet == Alphabet::Dna
                                                && param.profile == "iupac")
                                            || (param.alphabet == Alphabet::Iupac
                                                && param.profile == "iupac")
                                            || (param.alphabet == Alphabet::Ascii
                                                && param.profile == "ascii")
                                    })
                            })
                        })
                    })
                })
            })
        })
    }
}

pub fn read_grid(path: &str) -> Result<GridConfig, Box<dyn std::error::Error>> {
    let toml_str = fs::read_to_string(path)?;
    let config: GridConfig = toml::from_str(&toml_str)?;
    Ok(config)
}
