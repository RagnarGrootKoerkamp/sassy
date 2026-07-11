#![no_std]
#![feature(abi_ptx, core_intrinsics)]

use core::intrinsics;

/// Represents a hit range for a pattern match
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HitRange {
    pub pattern_idx: u32,
    pub start: i32,
    pub end: i32,
}

/// Type alias for pattern encoding (shared with host)
pub type PeqValue = u64;

/// Core Myers bit-parallel algorithm step (scalar version for GPU)
///
/// This is the same algorithm as the SIMD version, but operates on scalar u64 values.
/// Each GPU thread executes this independently for its assigned pattern.
#[inline(always)]
pub fn myers_step_scalar(
    vp: u64,
    vn: u64,
    cost: u64,
    eq: u64,
    all_ones: u64,
    last_bit_shift: u32,
    last_bit_mask: u64,
) -> (u64, u64, u64) {
    let eq_and_pv = eq & vp;
    let xh = ((eq_and_pv.wrapping_add(vp)) ^ vp) | eq;
    let mh = vp & xh;
    let ph = vn | (all_ones ^ (xh | vp));

    let ph_shifted = ph << 1;
    let mh_shifted = mh << 1;

    let xv = eq | vn;
    let vp_out = mh_shifted | (all_ones ^ (xv | ph_shifted));
    let vn_out = ph_shifted & xv;

    let ph_bit = (ph & last_bit_mask) >> last_bit_shift;
    let mh_bit = (mh & last_bit_mask) >> last_bit_shift;

    let cost_out = cost.wrapping_add(ph_bit).wrapping_sub(mh_bit);

    (vp_out, vn_out, cost_out)
}

// NVVM intrinsics for thread indexing
extern "C" {
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.x"]
    fn block_idx_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"]
    fn thread_idx_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.x"]
    fn block_dim_x() -> i32;
}

/// Main CUDA kernel for Myers bit-parallel search
/// 
/// Each thread processes one pattern against the shared text.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn myers_search_kernel(
    // Input: text to search in
    text_ptr: *const u8,
    text_len: usize,
    
    // Input: pre-computed pattern equality vectors (PEQs)
    // Layout: peqs[pattern_idx * 256 + char_value] = bit vector for that char
    peqs_ptr: *const PeqValue,
    
    // Input: pattern parameters
    pattern_length: usize,
    num_patterns: usize,
    
    // Input: search parameters
    k: u32,                 // Maximum allowed cost
    alpha_pattern: u64,     // Alpha mask for overhang
    last_bit_shift: u32,    // Shift amount for last bit
    
    // Output: array of hit ranges (pre-allocated)
    hit_ranges_ptr: *mut HitRange,
    max_hits: usize,
    
    // Output: atomic counter for number of hits found
    hit_count_ptr: *mut u32,
) {
    let tid = thread::index_1d() as usize;
    
    if tid >= num_patterns {
        return;
    }
    
    // Each thread processes one pattern
    let pattern_peqs_offset = tid * 256;
    let last_bit_mask = 1u64 << last_bit_shift;
    
    // Initialize Myers state
    let length_mask = (!0u64) >> (64usize.saturating_sub(pattern_length));
    let masked_alpha = alpha_pattern & length_mask;
    let mut vp = alpha_pattern;
    let mut vn = 0u64;
    let mut cost = masked_alpha.count_ones() as u64;
    
    let all_ones = !0u64;
    
    // Track active range
    let mut is_active = false;
    let mut range_start = -1i32;
    
    // Process each position in text
    for pos in 0..text_len {
        // Read character (coalesced across warp for adjacent threads)
        let c = *text_ptr.add(pos);
        
        // Look up pre-computed equality vector for this char
        let eq = *peqs_ptr.add(pattern_peqs_offset + c as usize);
        
        // Execute Myers step
        let (vp_new, vn_new, cost_new) = myers_step_scalar(
            vp,
            vn,
            cost,
            eq,
            all_ones,
            last_bit_shift,
            last_bit_mask,
        );
        
        vp = vp_new;
        vn = vn_new;
        cost = cost_new;
        
        // Check if cost is within threshold
        let was_active = is_active;
        is_active = cost <= k as u64;
        
        // Handle range transitions
        if is_active && !was_active {
            // Range starting
            range_start = pos as i32;
        } else if !is_active && was_active {
            // Range ending - emit hit
            // Use atomic to get unique slot
            let idx = atomics::fetch_add(hit_count_ptr, 1, Ordering::Relaxed);
            
            if idx < max_hits as u32 {
                let hit = HitRange {
                    pattern_idx: tid as u32,
                    start: range_start,
                    end: (pos - 1) as i32,
                };
                *hit_ranges_ptr.add(idx as usize) = hit;
            }
        }
    }
    
    // Finalize: if still active at end, emit final range
    if is_active {
        let idx = atomics::fetch_add(hit_count_ptr, 1, Ordering::Relaxed);
        
        if idx < max_hits as u32 {
            let hit = HitRange {
                pattern_idx: tid as u32,
                start: range_start,
                end: (text_len - 1) as i32,
            };
            *hit_ranges_ptr.add(idx as usize) = hit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_myers_step_scalar() {
        // Test case: simple match
        let vp = 0b1111111111111111u64;
        let vn = 0u64;
        let cost = 0u64;
        let eq = 0b1111111111111111u64; // All match
        let all_ones = !0u64;
        let last_bit_shift = 15;
        let last_bit_mask = 1u64 << last_bit_shift;

        let (vp_out, vn_out, cost_out) =
            myers_step_scalar(vp, vn, cost, eq, all_ones, last_bit_shift, last_bit_mask);

        // Cost should remain 0 for perfect match
        assert_eq!(cost_out, 0);
    }

    #[test]
    fn test_myers_step_mismatch() {
        // Test case: single mismatch
        let vp = 0b1111111111111111u64;
        let vn = 0u64;
        let cost = 0u64;
        let eq = 0b1111111111111110u64; // One mismatch at position 0
        let all_ones = !0u64;
        let last_bit_shift = 15;
        let last_bit_mask = 1u64 << last_bit_shift;

        let (vp_out, vn_out, cost_out) =
            myers_step_scalar(vp, vn, cost, eq, all_ones, last_bit_shift, last_bit_mask);

        // Cost should increase
        assert!(cost_out > cost);
    }
}
