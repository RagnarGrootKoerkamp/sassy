mod ascii;
mod dna;
mod iupac;

pub use ascii::{Ascii, CaseInsensitiveAscii, CaseSensitiveAscii};
pub use dna::Dna;
pub use iupac::Iupac;
use wide::u8x32;

use std::ops::{Index, IndexMut};

pub trait Profile: Clone + std::fmt::Debug {
    /// Encoding for a single character in the pattern.
    type A;
    /// Encoding for 64 characters in the text.
    type B: Index<usize, Output = u64> + IndexMut<usize, Output = u64> + Copy;
    fn encode_pattern(a: &[u8]) -> (Self, Vec<Self::A>);
    fn encode_ref(&self, b: &[u8; 64], out: &mut Self::B);
    /// Given the encoding of an `a` and the encoding for 64 `b`s,
    /// return a bitmask of which characters of `b` equal the corresponding character of `a`.
    fn eq(ca: &Self::A, cb: &Self::B) -> u64;
    /// Allocate a buffer of at most n_bases in search (and reuse)
    fn alloc_out() -> Self::B;
    fn n_bases(&self) -> usize;
    /// Verify whether a sequence matching the profile characters
    fn valid_seq(seq: &[u8]) -> bool;
    /// Return true if the two characters are a match according to profile
    fn is_match(char1: u8, char2: u8) -> bool;
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

// Simd helpers for stuff missing from wide.
// FIXME: Upstream into `wide`.
#[inline(always)]
fn u8x32_gt(a: u8x32, b: u8x32) -> u8x32 {
    unsafe {
        use std::mem::transmute;
        use wide::i8x32;
        let a: i8x32 = transmute(a);
        let b: i8x32 = transmute(b);
        let mask = i8x32::splat(1 << 7);
        transmute(wide::CmpGt::simd_gt(a ^ mask, b ^ mask))
    }
}

// FIXME: Upstream into `wide`.
#[inline(always)]
fn u8x32_shr(a: u8x32, shift: u8) -> u8x32 {
    unsafe {
        use std::mem::transmute;
        transmute(transmute::<_, wide::u16x16>(a) >> shift)
    }
}
