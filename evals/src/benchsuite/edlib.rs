use crate::benchsuite::sim_data::Alphabet;
use ::std::os::raw::c_char;
use edlib_rs::edlib_sys::*;
use edlib_rs::*;
use once_cell::sync::Lazy;
use sassy::profiles::*;

static EQUALITY_PAIRS: Lazy<Vec<EdlibEqualityPairRs>> = Lazy::new(build_equality_pairs);

pub fn get_edlib_config(k: i32, alphabet: &Alphabet) -> EdlibAlignConfigRs<'static> {
    let mut config = EdlibAlignConfigRs::default();
    config.mode = EdlibAlignModeRs::EDLIB_MODE_HW;
    if alphabet == &Alphabet::Iupac {
        //println!("[EDLIB] Added iupac alphabet");
        config.additionalequalities = &EQUALITY_PAIRS;
    }
    config.k = k;
    config.task = EdlibAlignTaskRs::EDLIB_TASK_PATH;
    config
}

fn build_equality_pairs() -> Vec<EdlibEqualityPairRs> {
    let codes = b"ACGTURYSWKMBDHVNX";
    let mut pairs = Vec::new();
    for &a in codes.iter() {
        for &b in codes.iter() {
            if Iupac::is_match(a, b) {
                // both upper
                pairs.push(EdlibEqualityPairRs {
                    first: a as c_char,
                    second: b as c_char,
                });
                // both lower
                pairs.push(EdlibEqualityPairRs {
                    first: a.to_ascii_lowercase() as c_char,
                    second: b.to_ascii_lowercase() as c_char,
                });
                // first upper, second lower
                pairs.push(EdlibEqualityPairRs {
                    first: a.to_ascii_lowercase() as c_char,
                    second: b as c_char,
                });
                // first lower, second upper
                pairs.push(EdlibEqualityPairRs {
                    first: a as c_char,
                    second: b.to_ascii_lowercase() as c_char,
                });
            }
        }
    }
    pairs
}

pub fn run_edlib(
    query: &[u8],
    target: &[u8],
    edlib_config: &EdlibAlignConfigRs,
) -> EdlibAlignResultRs {
    let edlib_result = edlibAlignRs(query, target, edlib_config);
    assert_eq!(edlib_result.status, EDLIB_STATUS_OK);
    edlib_result
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_edlib_wrapper() {
        let query = b"ATG";
        let target = b"CCCATGCCC";
        let config = get_edlib_config(1, &Alphabet::Dna);
        let r = run_edlib(query, target, &config);
        assert_eq!(r.editDistance, 0);
    }

    #[test]
    fn test_edlib_iupac() {
        let query = b"NTG";
        let target = b"CCCATGCCC";
        let dna_config = get_edlib_config(1, &Alphabet::Dna);
        let iupac_config = get_edlib_config(1, &Alphabet::Iupac);
        let dna_res = run_edlib(query, target, &dna_config);
        let iupac_res = run_edlib(query, target, &iupac_config);
        assert_eq!(dna_res.editDistance, 1);
        assert_eq!(iupac_res.editDistance, 0);
    }
}
