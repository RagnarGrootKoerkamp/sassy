use crate::{
    LANES,
    profiles::{Profile, u8x32_gt, u8x32_shr},
};
use std::mem::transmute;
use wide::{CmpEq, u8x32};

/// IUPAC alphabet: ACGT + NYR...
///
/// <https://www.bioinformatics.org/sms/iupac.html>
#[derive(Clone, Debug)]
pub struct Iupac {
    bases: Vec<u8>,
}

impl Profile for Iupac {
    type A = usize;
    type B = [u64; 16];

    fn encode_pattern(a: &[u8]) -> (Self, Vec<Self::A>) {
        if !Self::valid_seq(a) {
            panic!(
                "Pattern is not valid IUPAC: {:?}",
                String::from_utf8_lossy(a)
            );
        }

        let mut bases = vec![b'A', b'C', b'T', b'G'];
        let mut query_profile = Vec::with_capacity(a.len());
        for &c in a {
            let c = c & !0x20; // to uppercase
            if !bases.contains(&c) {
                bases.push(c);
            }
            query_profile.push(bases.iter().position(|&x| x == c).unwrap());
        }
        (Iupac { bases }, query_profile)
    }
    fn encode_patterns(a: &[&[u8]]) -> (Self, Vec<[Self::A; LANES]>) {
        assert!(!a.is_empty());
        assert!(a.len() <= LANES);
        let len = a[0].len();
        for p in a {
            if !Self::valid_seq(p) {
                panic!(
                    "Pattern is not valid IUPAC: {:?}",
                    String::from_utf8_lossy(p)
                );
            }
            assert_eq!(p.len(), len);
        }

        let mut bases = vec![b'A', b'C', b'T', b'G'];
        let mut query_profile = vec![[0; LANES]; len];
        for i in 0..a[0].len() {
            for lane in 0..LANES {
                let c = if lane < a.len() { a[lane][i] } else { b'X' };
                let c = c & !0x20; // to uppercase
                if !bases.contains(&c) {
                    bases.push(c);
                }
                query_profile[i][lane] = bases.iter().position(|&x| x == c).unwrap();
            }
        }
        (Iupac { bases }, query_profile)
    }

    /// NOTE: `out` should be initialized using `self.alloc_out()`.
    #[inline(always)]
    fn encode_ref(&self, b: &[u8; 64], out: &mut Self::B) {
        assert!(self.bases.len() <= out.len());
        let extra_bases: &[u8] = &self.bases[4..];
        unsafe {
            let zero = u8x32::splat(0);
            let mask4 = u8x32::splat(0x0F);
            let mask5 = u8x32::splat(0x1F);

            let chunk0 = u8x32::from(&b[0..32]);
            let chunk1 = u8x32::from(&b[32..64]);

            let low4_0 = chunk0 & mask4;
            let low4_1 = chunk1 & mask4;
            let idx5_0 = chunk0 & mask5;
            let idx5_1 = chunk1 & mask5;

            let is_hi_0 = u8x32_gt(idx5_0, u8x32::splat(15));
            let is_hi_1 = u8x32_gt(idx5_1, u8x32::splat(15));

            let tbl256 = u8x32::from(transmute::<_, [u8; 32]>([PACKED_NIBBLES, PACKED_NIBBLES]));

            let shuffled0 = half_shuffle(tbl256, low4_0);
            let shuffled1 = half_shuffle(tbl256, low4_1);

            let lo_nib0 = shuffled0 & mask4;
            let lo_nib1 = shuffled1 & mask4;

            // Shift as u16 because AVX2 does not have u8 shifts.
            // This 'leaks' bits into high half of each byte, but we only ever
            // read the low half via `m=get_encoded(base)`, so that's ok.
            let hi_nib0 = u8x32_shr(shuffled0, 4);
            let hi_nib1 = u8x32_shr(shuffled1, 4);

            let nib0 = is_hi_0.blend(hi_nib0, lo_nib0);
            let nib1 = is_hi_1.blend(hi_nib1, lo_nib1);

            for (i, &base) in [b'A', b'C', b'T', b'G'].iter().enumerate() {
                let m = u8x32::splat(get_encoded(base));

                let match0 = !(nib0 & m).simd_eq(zero);
                let match1 = !(nib1 & m).simd_eq(zero);

                let low = match0.to_bitmask() as u32 as u64;
                let high = match1.to_bitmask() as u32 as u64;

                *out.get_unchecked_mut(i) = (high << 32) | low;
            }

            for (i, &base) in extra_bases.iter().enumerate() {
                let m = u8x32::splat(get_encoded(base));

                let match0 = !(nib0 & m).simd_eq(zero);
                let match1 = !(nib1 & m).simd_eq(zero);

                let low = match0.to_bitmask() as u32 as u64;
                let high = match1.to_bitmask() as u32 as u64;

                *out.get_unchecked_mut(i + 4) = (high << 32) | low;
            }
        }
    }

    #[inline(always)]
    fn eq(ca: &usize, cb: &[u64; 16]) -> u64 {
        unsafe { *cb.get_unchecked(*ca) }
    }

    #[inline(always)]
    fn is_match(char1: u8, char2: u8) -> bool {
        (get_encoded(char1) & get_encoded(char2)) > 0
    }

    #[inline(always)]
    fn alloc_out() -> Self::B {
        [0; 16] //FIXME: is this always valid?
    }

    #[inline(always)]
    fn n_bases(&self) -> usize {
        self.bases.len()
    }

    #[inline(always)]
    fn valid_seq(seq: &[u8]) -> bool {
        const LANES: usize = 32;
        let len = seq.len();
        let mut i = 0;
        unsafe {
            let mask4 = u8x32::splat(0x0F);
            let high4 = u8x32::splat(0xF0);
            let tbl256 = u8x32::from(transmute::<_, [u8; 32]>([
                PACKED_NIBBLES_INDICATOR,
                PACKED_NIBBLES_INDICATOR,
            ]));
            while i + LANES <= len {
                let chunk = u8x32::from(&seq[i..i + LANES]);
                let upper = chunk & u8x32::splat(!0x20);

                // Check if > '@' (64) (=b'A'-1) and < 128.
                let in_range =
                    u8x32_gt(upper, u8x32::splat(64)) & u8x32_gt(u8x32::splat(128), upper);
                if !in_range.all() {
                    return false;
                }

                let idx5 = upper & u8x32::splat(0x1F);
                let low4 = idx5 & mask4;
                let is_hi = u8x32_gt(idx5, u8x32::splat(15));
                let shuffled = half_shuffle(tbl256, low4);
                let lo_nib = shuffled & mask4;
                let hi_nib = shuffled & high4;
                let nib = is_hi.blend(hi_nib, lo_nib);

                // nibbles are 0 for IUPAC.
                if nib.simd_eq(u8x32::splat(0)).any() {
                    return false;
                }

                i += LANES;
            }
        }

        // Scalar for rest
        while i < len {
            let c = seq[i] & !0x20;
            if c <= b'@' || c >= b'Z' || IUPAC_CODE[(c & 0x1F) as usize] == 255 {
                return false;
            }
            i += 1;
        }

        true
    }

    #[inline(always)]
    fn reverse_complement(seq: &[u8]) -> Vec<u8> {
        seq.iter().rev().map(|&c| RC[c as usize]).collect()
    }

    // TODO: Implement this using SIMD
    #[inline(always)]
    fn complement(seq: &[u8]) -> Vec<u8> {
        seq.iter().map(|&c| RC[c as usize]).collect()
    }

    #[inline(always)]
    fn supports_overhang() -> bool {
        true
    }
}

/// Do a shuffle within each half of table.
/// Matching `__mm256_shuffle_epi8`.
#[inline(always)]
fn half_shuffle(table: u8x32, idx: u8x32) -> u8x32 {
    table.swizzle_half_relaxed(idx)
}

const RC: [u8; 256] = {
    let mut rc = [0; 256];
    let mut i = 0;
    while i < 256 {
        rc[i] = i as u8;
        i += 1;
    }
    // Standard bases
    rc[b'A' as usize] = b'T';
    rc[b'C' as usize] = b'G';
    rc[b'T' as usize] = b'A';
    rc[b'G' as usize] = b'C';
    rc[b'a' as usize] = b't';
    rc[b'c' as usize] = b'g';
    rc[b't' as usize] = b'a';
    rc[b'g' as usize] = b'c';
    // IUPAC ambiguity codes
    rc[b'R' as usize] = b'Y'; // A|G -> T|C
    rc[b'Y' as usize] = b'R'; // C|T -> G|A
    rc[b'S' as usize] = b'S'; // G|C -> C|G
    rc[b'W' as usize] = b'W'; // A|T -> T|A
    rc[b'K' as usize] = b'M'; // G|T -> C|A
    rc[b'M' as usize] = b'K'; // A|C -> T|G
    rc[b'B' as usize] = b'V'; // C|G|T -> G|C|A
    rc[b'D' as usize] = b'H'; // A|G|T -> T|C|A
    rc[b'H' as usize] = b'D'; // A|C|T -> T|G|A
    rc[b'V' as usize] = b'B'; // A|C|G -> T|G|C
    rc[b'N' as usize] = b'N'; // A|C|G|T -> T|G|C|A
    rc[b'X' as usize] = b'X';
    // Lowercase versions
    rc[b'r' as usize] = b'y';
    rc[b'y' as usize] = b'r';
    rc[b's' as usize] = b's';
    rc[b'w' as usize] = b'w';
    rc[b'k' as usize] = b'm';
    rc[b'm' as usize] = b'k';
    rc[b'b' as usize] = b'v';
    rc[b'd' as usize] = b'h';
    rc[b'h' as usize] = b'd';
    rc[b'v' as usize] = b'b';
    rc[b'n' as usize] = b'n';
    rc[b'x' as usize] = b'x';
    rc
};

#[rustfmt::skip]
const IUPAC_CODE: [u8; 32] = {
    let mut t = [255u8; 32];
    // Standard bases
    // Map ACGT -> [0,1,3,2], like packed_seq does.
    const A: u8 = 1 << 0;
    const C: u8 = 1 << 1;
    const T: u8 = 1 << 2;
    const G: u8 = 1 << 3;

    // Map common chars.
    // Lower case has the same last 5 bits as upper case.
    // (Thanks ASCII :)
    t[b'A' as usize & 0x1F] = A;
    t[b'C' as usize & 0x1F] = C;
    t[b'T' as usize & 0x1F] = T;
    t[b'U' as usize & 0x1F] = T;
    t[b'G' as usize & 0x1F] = G;
    t[b'N' as usize & 0x1F] = A|C|T|G;
    
    // IUPAC ambiguity codes
    // https://www.bioinformatics.org/sms/iupac.html
    t[b'R' as usize & 0x1F] = A|G;
    t[b'Y' as usize & 0x1F] = C|T;
    t[b'S' as usize & 0x1F] = G|C;
    t[b'W' as usize & 0x1F] = A|T;
    t[b'K' as usize & 0x1F] = G|T;
    t[b'M' as usize & 0x1F] = A|C;
    t[b'B' as usize & 0x1F] = C|G|T;
    t[b'D' as usize & 0x1F] = A|G|T;
    t[b'H' as usize & 0x1F] = A|C|T;
    t[b'V' as usize & 0x1F] = A|C|G;
    
    // Gap/unknown
    t[b'X' as usize & 0x1F] = 0;
    
    t
};

#[inline(always)]
pub fn get_encoded(c: u8) -> u8 {
    IUPAC_CODE[(c & 0x1F) as usize]
}

const PACKED_NIBBLES: [u8; 16] = {
    let mut p = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        let lo = IUPAC_CODE[i] & 0x0F;
        let hi = IUPAC_CODE[i + 16] & 0x0F;
        // packed 8 bit of low nibbles(0-3) and high nibbles(4-7)
        p[i] = (hi << 4) | lo;
        i += 1;
    }
    p
};

/// Nibbles are 1111 for IUPAC chars, and 0000 for non-IUPAC chars.
const PACKED_NIBBLES_INDICATOR: [u8; 16] = {
    let mut p = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        let lo = if IUPAC_CODE[i] < 255 { 0b1111 } else { 0 };
        let hi = if IUPAC_CODE[i + 16] < 255 { 0b1111 } else { 0 };
        // packed 8 bit of low nibbles(0-3) and high nibbles(4-7)
        p[i] = (hi << 4) | lo;
        i += 1;
    }
    p
};

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_iupac_is_match() {
        assert!(Iupac::is_match(b'a', b'A'));
        assert!(Iupac::is_match(b'C', b'C'));
        assert!(Iupac::is_match(b'T', b't'));
        assert!(Iupac::is_match(b'G', b'G'));
        assert!(Iupac::is_match(b'y', b'Y'));
        assert!(Iupac::is_match(b'A', b'N'));
        assert!(Iupac::is_match(b'C', b'Y'));
    }

    fn get_match_positions_u64(result: &[u64]) -> Vec<Vec<usize>> {
        result
            .iter()
            .filter_map(|&base_result| {
                if base_result == 0 {
                    None
                } else {
                    let positions: Vec<usize> = (0..64)
                        .filter(|&pos| (base_result & (1 << pos)) != 0)
                        .collect();
                    Some(positions)
                }
            })
            .collect()
    }

    #[test]
    fn test_just_atgc() {
        let mut seq = [b'g'; 64];
        seq[0] = b'a';
        seq[1] = b'y'; // C or T
        let profiler = Iupac::encode_pattern(b"").0;
        let mut result = Iupac::alloc_out();
        profiler.encode_ref(&seq, &mut result);
        let positions = get_match_positions_u64(&result);
        let a_positions = positions[0].clone();
        let c_positions = positions[1].clone();
        let t_positions = positions[2].clone();
        let g_positions = positions[3].clone();
        assert_eq!(a_positions, vec![0]);
        assert_eq!(t_positions, vec![1]);
        assert_eq!(g_positions, (2..64).collect::<Vec<_>>());
        assert_eq!(c_positions, vec![1]);
    }

    #[test]
    fn test_extra_bases_ny() {
        let mut seq = [b'g'; 64];
        seq[0] = b'a'; // Does not match Y
        seq[1] = b'y'; // Matches Y
        seq[2] = b'C'; // Matches Y
        let profiler = Iupac::encode_pattern(b"NY").0;
        let mut result = Iupac::alloc_out();
        profiler.encode_ref(&seq, &mut result);
        let positions = get_match_positions_u64(&result);
        let n_positions = positions[4].clone();
        let y_positions = positions[5].clone();
        // N matches all positions
        assert_eq!(n_positions, (0..64).collect::<Vec<_>>());
        // Y matches 1,2
        assert_eq!(y_positions, vec![1, 2]);
    }

    #[test]
    fn test_just_atgc_64() {
        let mut seq = [b'g'; 64];
        seq[0] = b'a';
        seq[1] = b'y'; // C or T
        seq[34] = b'y'; // C or T
        let profiler = Iupac::encode_pattern(b"").0;
        let mut result = Iupac::alloc_out();
        profiler.encode_ref(&seq, &mut result);
        let positions = get_match_positions_u64(&result);
        let a_positions = positions[0].clone();
        let c_positions = positions[1].clone();
        let t_positions = positions[2].clone();
        let g_positions = positions[3].clone();
        assert_eq!(a_positions, vec![0]);
        assert_eq!(t_positions, vec![1, 34]);
        assert_eq!(
            g_positions,
            [
                &(2..34).collect::<Vec<_>>()[..], // 34 not inclusive
                &(35..64).collect::<Vec<_>>()[..]
            ]
            .concat()
        );
        assert_eq!(c_positions, vec![1, 34]);
    }

    #[test]
    fn test_extra_bases_ny_64() {
        let mut seq = [b'g'; 64];
        seq[0] = b'a'; // Does not match Y
        seq[1] = b'y'; // Matches Y
        seq[2] = b'C'; // Matches Y
        seq[50] = b'y'; // Matches Y
        seq[63] = b'y'; // Matches Y
        let profiler = Iupac::encode_pattern(b"NY").0;
        let mut result = Iupac::alloc_out();
        profiler.encode_ref(&seq, &mut result);
        let positions = get_match_positions_u64(&result);
        let n_positions = positions[4].clone();
        let y_positions = positions[5].clone();
        // N matches all positions
        assert_eq!(n_positions, (0..64).collect::<Vec<_>>());
        assert_eq!(y_positions, vec![1, 2, 50, 63]);
    }

    #[test]
    fn test_iupac_u64_case_insensitive() {
        let mut seq = [b'G'; 64];
        seq[0] = b'a';
        seq[1] = b'A';
        seq[3] = b'r';
        seq[4] = b'W';
        let profiler = Iupac::encode_pattern(b"").0;
        let mut result = Iupac::alloc_out();
        profiler.encode_ref(&seq, &mut result);
        let positions = get_match_positions_u64(&result);
        assert_eq!(positions[0], vec![0, 1, 3, 4]);
    }

    #[test]
    fn test_iupac_valid_seq_all() {
        // long enough to trigger the SIMD path.
        let all_codes = b"ACTUGNRYSWKMBDHVXACTUGNRYSWKMBDHVX";
        for &c in all_codes {
            assert!(Iupac::valid_seq(&[c]));
            assert!(Iupac::valid_seq(&[c.to_ascii_lowercase()]));
        }
        assert!(Iupac::valid_seq(all_codes));
        assert!(Iupac::valid_seq(&all_codes.to_ascii_lowercase()));
        let mut mixed_codes = all_codes.to_vec();
        mixed_codes.extend_from_slice(&all_codes.to_ascii_lowercase());
        assert!(Iupac::valid_seq(&mixed_codes));
        assert!(Iupac::valid_seq(&all_codes.to_ascii_lowercase()));
        // Mixed case should also be valid
        assert!(Iupac::valid_seq(b"AaCcTtUuGgNnRrYySsWwKkMmBbDdHhVvXx"));
        // Adding one bad character should make it invalid
        assert!(!Iupac::valid_seq(b"_aCcTtUuGgNnRrYySsWwKkMmBbDdHhVvXx"));
        assert!(!Iupac::valid_seq(b"AaCcTtUuGgNnRrYySsWwKkMmBbDdH_VvXx"));
        assert!(!Iupac::valid_seq(b"AaCcTtUuGgN@RrYySsWwKkMmBbDdHhVvXx"));
        assert!(!Iupac::valid_seq(b"AaEcTtUuGgNnRrYySsWwKkMmBbDdHhVvXx"));
        assert!(!Iupac::valid_seq(b"AaCeTtUuGgNnRrYySsWwKkMmBbDdHhVvXx"));
    }

    #[test]
    fn test_iupac_different_lengths() {
        let valid_codes = b"ACTUGNRYSWKMBDHVX";
        for len in [1, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
            let seq = valid_codes
                .iter()
                .cycle()
                .take(len)
                .copied()
                .collect::<Vec<_>>();
            assert!(Iupac::valid_seq(&seq), "Failed at length {}", len);
        }
    }

    #[test]
    fn test_iupac_valid_seq_empty() {
        assert!(Iupac::valid_seq(b"")); // Not sure if this should be valid or not
    }

    #[test]
    fn test_invalid_iupac_codes() {
        // Test invalid characters
        let invalid_cases = [
            // Below 'A'
            b"@CGT", b"?CGT", b"1CGT", b" CGT", // Above 'X'
            b"ACGZ", b"ACG[", b"ACG{", b"ACG~",
            // Control characters, \n, \t, \r, etc
            b"ACG\n", b"ACG\t", b"ACG\r", b"\0CGT",
        ];

        for case in invalid_cases {
            assert!(!Iupac::valid_seq(case));
        }
    }

    #[test]
    fn test_alloc_out() {
        let pattern = b"actgryswkmbdhvnx";
        assert!(Iupac::valid_seq(pattern));
        // Fill first 16 with pattern, rest with 'a'
        let mut text = [b'a'; 64];
        text[..16].copy_from_slice(pattern);
        let (profiler, _) = Iupac::encode_pattern(pattern);
        let mut out = Iupac::alloc_out();
        profiler.encode_ref(&text, &mut out);
    }

    #[test]
    fn test_iupac_boundary_chars() {
        // Test exact boundaries
        assert!(!Iupac::valid_seq(b"@")); // 64 - invalid
        assert!(Iupac::valid_seq(b"A")); // 65 - valid
        assert!(Iupac::valid_seq(b"X")); // 88 - valid
        assert!(Iupac::valid_seq(b"Y")); // 89 - valid
        assert!(!Iupac::valid_seq(b"Z")); // 90 - invalid

        // Same but in 32 bytes to trigger SIMD
        let mut seq = b"ACGT".repeat(8); // 32 bytes
        seq[31] = b'Y';
        assert!(Iupac::valid_seq(&seq));
        seq[31] = b'Z';
        assert!(!Iupac::valid_seq(&seq));
    }
}
