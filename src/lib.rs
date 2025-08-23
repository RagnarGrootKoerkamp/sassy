#![feature(portable_simd)]
//! # Sassy: fast approximate string matching
//!
//! Sassy is a library for searching approximate matches of short patterns/queries in longer texts.
//! It supports ASCII and DNA, and works best for patterns of length up to 1000.
//!
//! The main entrypoint is the [`Searcher`] object.
//! This can be configured with the alphabet ([`profiles::Ascii`], [`profiles::Dna`], or [`profiles::Iupac`]),
//! whether to search the reverse complement ([`Searcher::new_fwd`], [`Searcher::new_rc`]),
//! and optionally with an _overhang cost_ for IUPAC profiles ([`Searcher::new_fwd_with_overhang`]).
//!
//! Given a [`Searcher`], you can search call [`Searcher::search`] with a pattern, text, and maximum edit distance `k`.
//! This will return a vector of [`Match`] objects, that each contain the substring of text they match, the corresponding `cost`,
//! and the `cigar` string that describes the alignment.
//!
//! ## `search` vs `search_all`
//!
//! By default, [`Searcher::search`] will only return matches that end in a rightmost local minimum.
//! [`Searcher::search_all`], on the other hand, always returns matches in all end-positions with cost <= `k`.
//!
//! See the paper (linked on [GitHub](https://github.com/RagnarGrootKoerkamp/sassy)) for details.
//!
//! ## Example
//!
//! Usage example:
//! ```
//! use sassy::{
//!     CachedRev, Match, Searcher, Strand,
//!     profiles::{Dna, Iupac},
//! };
//!
//! // --- Test data ---
//! let pattern = b"ATCG";
//! let text = b"CCCATCACCC";
//! let k = 1;
//!
//! // --- FWD only search ---
//! /*
//!     CCCATCACCC (text)
//!        |||x
//!        ATCG    (pattern)
//! */
//! let mut searcher = Searcher::<Dna>::new_fwd();
//! let matches = searcher.search(pattern, &text, k);
//!
//! assert_eq!(matches.len(), 1);
//! let fwd_match = matches[0].clone();
//! assert_eq!(fwd_match.pattern_start, 0);
//! assert_eq!(fwd_match.pattern_end, 4);
//! assert_eq!(fwd_match.text_start, 3);
//! assert_eq!(fwd_match.text_end, 7);
//! assert_eq!(fwd_match.cost, 1);
//! assert_eq!(fwd_match.strand, Strand::Fwd);
//! assert_eq!(fwd_match.cigar.to_string(), "3=1X");
//!
//! // --- FWD + RC search ---
//! /*
//!     CCCATCACCC (text)       GGGTGATGGG (text - rc)
//!        |||x                      ||x|
//!        ATCG    (pattern)         ATCG
//! */
//! let mut searcher = Searcher::<Dna>::new_rc();
//! // We can cache the reverse text if we do multiple pattern
//! // searches in `rc` mode
//! let cached_text = CachedRev::new(text, true);
//! let matches = searcher.search(pattern, &cached_text, k);
//!
//! // Gives two matches, of which one the previous forward match
//! assert_eq!(matches.len(), 2);
//! assert_eq!(matches[0], fwd_match);
//! let rc_match = matches[1].clone();
//! assert_eq!(rc_match.pattern_start, 0);
//! assert_eq!(rc_match.pattern_end, 4);
//! assert_eq!(rc_match.text_start, 1);
//! assert_eq!(rc_match.text_end, 5);
//! assert_eq!(rc_match.cost, 1);
//! assert_eq!(rc_match.strand, Strand::Rc);
//! assert_eq!(rc_match.cigar.to_string(), "2=1X1=");
//!
//! // --- FWD + RC search with overhang ---
//! /*
//!               GTXXXNNN     (text)
//!               ||   |||
//!             ACGT   ACGT    (pattern)
//! */
//! let pattern = b"ACGT";
//! let text = b"GTXXXNNN";
//! let alpha = 0.5;
//! let k = 1;
//! let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(alpha);
//! let matches = searcher.search(pattern, &text, k);
//!
//! assert_eq!(matches[0].pattern_start, 2);
//! assert_eq!(matches[0].pattern_end, 4);
//! assert_eq!(matches[0].text_start, 0);
//! assert_eq!(matches[0].text_end, 2);
//! assert_eq!(matches[0].cost, 1);
//! assert_eq!(matches[0].strand, Strand::Fwd);
//! assert_eq!(matches[0].cigar.to_string(), "2=");
//!
//! assert_eq!(matches[1].pattern_start, 0);
//! assert_eq!(matches[1].pattern_end, 3);
//! assert_eq!(matches[1].text_start, 5);
//! assert_eq!(matches[1].text_end, 8);
//! assert_eq!(matches[1].cost, 0);
//! assert_eq!(matches[1].strand, Strand::Fwd);
//! assert_eq!(matches[1].cigar.to_string(), "3=");
//! ```
#[cfg(not(any(
    doc,
    debug_assertions,
    target_feature = "avx2",
    target_feature = "neon",
    feature = "scalar"
)))]
compile_error!(
    "Sassy uses AVX2 or NEON SIMD instructions. Compile using `-C target-cpu=native` to get the expected performance. Silence this error using the `scalar` feature."
);

// INTERNAL MODS
mod bitpacking;
mod delta_encoding;
mod minima;
mod search;
mod trace;

// (PARTIALLY) PUBLIC MODS

pub mod profiles;

pub use search::CachedRev;
pub use search::Match;
pub use search::RcSearchAble;
pub use search::Searcher;
pub use search::Strand;

// BINDINGS

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "c")]
mod c;

// TYPEDEFS

use std::simd::Simd;

#[cfg(feature = "avx512")]
const LANES: usize = 8;
#[cfg(not(feature = "avx512"))]
const LANES: usize = 4;

type S = Simd<u64, LANES>;

// TESTS

/// Print info on CPU features and speed of searching.
pub fn test_cpu_features() {
    eprintln!("CPU features during compilation and runtime:");
    #[cfg(target_arch = "x86_64")]
    {
        eprintln!("Target architecture: x86_64");

        let sse = if is_x86_feature_detected!("sse") {
            "+"
        } else {
            "-"
        };
        #[cfg(target_feature = "sse")]
        eprintln!("SSE  + {sse}");
        #[cfg(not(target_feature = "sse"))]
        eprintln!("SSE  - {sse}");

        let avx2 = if is_x86_feature_detected!("avx2") {
            "+"
        } else {
            "-"
        };
        #[cfg(target_feature = "avx")]
        eprintln!("AVX2 + {avx2}");
        #[cfg(not(target_feature = "avx"))]
        eprintln!("AVX2 - {avx2}");

        let bmi2 = if is_x86_feature_detected!("bmi2") {
            "+"
        } else {
            "-"
        };
        #[cfg(target_feature = "bmi2")]
        eprintln!("BMI2 + {bmi2}");
        #[cfg(not(target_feature = "bmi2"))]
        eprintln!("BMI2 - {bmi2}");
    }
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;

        eprintln!("Target architecture: aarch64 currently unsupported");

        let neon = if is_aarch64_feature_detected!("neon") {
            "+"
        } else {
            "-"
        };
        #[cfg(target_feature = "neon")]
        eprintln!("NEON + {neon}");
        #[cfg(not(target_feature = "neon"))]
        eprintln!("NEON - {neon}");
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        eprintln!("Unsupported target architecture");
    }
}

/// Print throughput in GB/s for random pattern and text.
pub fn test_throughput() {
    eprintln!("Running a little test: aligning a 23bp pattern against 100kb text, with 1 error.");
    eprintln!("With AVX2, this is typically around 2GB/s. Without, closer to 1.3GB/s.");
    eprintln!("If you see 0.02GB/s, that means you're on a debug rather than release build.");

    use rand::Rng;
    let n = 100000;
    let m = 23;
    let k = 1;

    let mut rng = rand::rng();
    let text: Vec<u8> = (0..n).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();
    let pattern: Vec<u8> = (0..m).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();

    let mut searcher = Searcher::<profiles::Dna>::new(false, None);
    let start = std::time::Instant::now();
    let _matches = searcher.search(&pattern, &text, k);
    let duration = start.elapsed();
    eprintln!(
        "Search throughput in GB/s: {}",
        text.len() as f32 / duration.as_secs_f32() / 1_000_000_000.0
    );
}

#[cfg(test)]
mod test {
    #[test]
    fn test_cpu_features() {
        super::test_cpu_features();
    }
    #[test]
    fn test_throughput() {
        super::test_throughput();
    }
}
