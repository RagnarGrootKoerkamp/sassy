#![feature(portable_simd)]
//! # Sassy: fast approximate string matching
//!
//! Usage example:
//! ```
//! use sassy::{Searcher, Match, profiles::{Dna, Iupac}, Strand};
//!
//!
//! let pattern = b"ATCG"; // ATCG dist=1
//! let text = b"AAAATCGGGGGGCATGGG";
//! let k = 1;
//! // rc = CGAT         rc: CAT dist=1
//!
//! let mut searcher = Searcher::<Dna>::new_rc();
//! let matches = searcher.search(pattern, &text, k);
//!
//! assert_eq!(matches.len(), 3);
//!
//! assert_eq!(matches[0].text_start, 3);
//! assert_eq!(matches[0].text_end, 7);
//! assert_eq!(matches[0].cost, 0);
//! assert_eq!(matches[0].strand, Strand::Fwd);
//! assert_eq!(matches[0].cigar.to_string(), "4=");
//!
//! assert_eq!(matches[1].text_start, 13);
//! assert_eq!(matches[1].text_end, 17);
//! assert_eq!(matches[1].cost, 1);
//! assert_eq!(matches[1].strand, Strand::Fwd);
//! assert_eq!(matches[1].cigar.to_string(), "2=X=");
//!
//! assert_eq!(matches[2].text_start, 12);
//! assert_eq!(matches[2].text_end, 15);
//! assert_eq!(matches[2].cost, 1);
//! assert_eq!(matches[2].strand, Strand::Rc);
//! assert_eq!(matches[2].cigar.to_string(), "2=D=");
//!
//! // Search with overhang and IUPAC.
//!
//! let pattern = b"ACGT";
//! let text =      b"GTXXXNNN";
//! //                     ACGT
//! let alpha = 0.5;
//! let k = 1;
//!
//! let mut searcher = Searcher::<Iupac>::new_fwd_with_overhang(alpha);
//! let matches = searcher.search(pattern, &text, k);
//!
//! eprintln!("{:?}", matches);
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
#![cfg_attr(
    not(any(doc, all(target_feature = "avx2", target_feature = "bmi2",))),
    deprecated(
        note = "Sassy currently requires x86-64 with AVX2 and BMI2 instructions. Compile using `-C target-cpu=x64-64-v3`."
    )
)]

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
