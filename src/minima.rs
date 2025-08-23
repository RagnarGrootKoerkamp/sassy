/// For each byte: (min_cost, end_cost)
/// Each 1 in a byte indicates -1.
/// Each 0 in a byte indicates +1.
#[cfg(target_feature = "bmi2")]
const PACKED_TABLE: [(i8, i8); 256] = {
    let mut table = [(0, 0); 256];

    let mut i = 0;
    while i < 256 {
        let mut min = 0;
        let mut cur = 0;
        let mut j = 0;
        while j < 8 {
            let bit = (i >> j) & 1;
            let delta = if bit == 1 { -1 } else { 1 };
            cur += delta;
            if cur < min {
                min = cur;
            }
            j += 1;
        }
        table[i] = (min, cur);
        i += 1;
    }

    table
};

/// Each byte has 4 deltas, with the low 4-bit nibble indicating +1 deltas and
/// the high nibble indicating -1 deltas.
#[cfg(not(target_feature = "bmi2"))]
const NIBBLE_TABLE: [(i8, i8); 256] = {
    let mut table = [(0, 0); 256];

    let mut i = 0;
    while i < 256 {
        let mut min = 0;
        let mut cur = 0;
        let mut j = 0;
        let pos = i & 15;
        let neg = i >> 4;
        while j < 4 {
            let pos_bit = (pos >> j) & 1;
            let neg_bit = (neg >> j) & 1;
            let delta = if pos_bit == 1 { 1 } else { 0 } - if neg_bit == 1 { 1 } else { 0 };
            cur += delta;
            if cur < min {
                min = cur;
            }
            j += 1;
        }
        table[i] = (min, cur);
        i += 1;
    }

    table
};

/// Compute any prefix min <= k over 8 bytes via SIMD vectorized DP approach.
#[cfg(target_feature = "bmi2")]
#[inline(always)]
pub fn prefix_min(p: u64, m: u64) -> (i8, i8) {
    // extract only the relevant chars
    let delta = p | m;
    let num_p = p.count_ones();
    let num_m = m.count_ones();
    let deltas = unsafe { std::arch::x86_64::_pext_u64(m, delta) };
    let mut min = 0;
    let mut cur = 0;
    for i in 0..8 {
        let byte = (deltas >> (i * 8)) as u8;
        let (min_cost, end_cost) = PACKED_TABLE[byte as usize];
        min = min.min(cur + min_cost);
        cur += end_cost;
    }
    (min, num_p as i8 - num_m as i8)
}

#[cfg(not(target_feature = "bmi2"))]
#[inline]
pub fn prefix_min(p: u64, m: u64) -> (i8, i8) {
    let mut min = 0;
    let mut cur = 0;
    for i in 0..16 {
        let byte = ((p >> (i * 4)) & 15) as u8 | (m >> (i * 4) << 4) as u8;
        let (min_cost, end_cost) = NIBBLE_TABLE[byte as usize];
        min = min.min(cur + min_cost);
        cur += end_cost;
    }

    (min, cur)
}
