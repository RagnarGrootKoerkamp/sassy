mod ascii;
mod dna;
pub(crate) mod iupac;

pub use ascii::{Ascii, CaseInsensitiveAscii, CaseSensitiveAscii};
pub use dna::Dna;
pub use iupac::Iupac;

use std::ops::{Index, IndexMut};

use crate::LANES;

pub trait Profile: Clone + std::fmt::Debug + Sync {
    /// Encoding for a single character in the pattern.
    type A: Sync;
    /// Encoding for 64 characters in the text.
    type B: Index<usize, Output = u64> + IndexMut<usize, Output = u64> + Copy + Sync;
    /// Total number of character in the alphabet.
    const N_CHARS: usize;
    fn encode_pattern(a: &[u8]) -> (Self, Vec<Self::A>);
    fn encode_patterns(_a: &[&[u8]]) -> (Self, Vec<[Self::A; LANES]>) {
        unimplemented!(
            "Profile::encode_patterns not implemented for {:?}",
            std::any::type_name::<Self>()
        );
    }
    /// Encode a character to map from 0..N_CHARS.
    fn encode_char(c: u8) -> u8;
    fn encode_ref(&self, b: &[u8; 64], out: &mut Self::B);
    /// Given the encoding of an `a` and the encoding for 64 `b`s,
    /// return a bitmask of which characters of `b` equal the corresponding character of `a`.
    fn eq(ca: &Self::A, cb: &Self::B) -> u64;
    /// Index into the encoded-reference profile (`Self::B`) for pattern character
    /// `ca`. This is the same index `eq` reads internally; exposing it lets callers
    /// build a profile transposed across SIMD lanes and keyed by character, so the
    /// per-row lookup becomes a single vector load instead of a per-lane gather.
    fn eq_idx(ca: &Self::A) -> usize;
    /// Allocate a buffer of at most n_bases in search (and reuse)
    fn alloc_out() -> Self::B;
    fn n_bases(&self) -> usize;
    /// Verify whether a sequence matching the profile characters
    fn valid_seq(seq: &[u8]) -> bool;
    /// Return true if the two characters are a match according to profile
    fn is_match(char1: u8, char2: u8) -> bool;
    /// Return true if every position in `pattern` matches the corresponding
    /// position in `text` according to this profile (e.g. IUPAC ambiguity,
    /// case-insensitivity).
    fn is_match_slice(pattern: &[u8], text: &[u8]) -> bool {
        pattern.len() == text.len()
            && pattern
                .iter()
                .zip(text)
                .all(|(&p, &t)| Self::is_match(p, t))
    }
    /// Reverse-complement the input string.
    fn reverse_complement(_query: &[u8]) -> Vec<u8> {
        unimplemented!(
            "Profile::reverse_complement not implemented for {:?}",
            std::any::type_name::<Self>()
        );
    }
    fn complement(_query: &[u8]) -> Vec<u8> {
        unimplemented!(
            "Profile::reverse_complement not implemented for {:?}",
            std::any::type_name::<Self>()
        );
    }
    fn supports_overhang() -> bool {
        unimplemented!("Profile does not support overhang");
    }
}
