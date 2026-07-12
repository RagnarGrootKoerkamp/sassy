//! GPU Compute kernel for Myers bit-parallel approximate string matching
//!
//! This kernel can be compiled to SPIR-V for use with WGPU (Vulkan, Metal, DX12, etc.)

#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(target_arch = "spirv")]
use spirv_std::{glam::UVec3, spirv};

/// Workgroup size - number of threads per workgroup
/// Each thread processes one pattern
pub const WORKGROUP_SIZE: u32 = 256;

type T = u32;

/// Represents a hit range for a pattern match
/// Must be repr(C) and Pod for GPU/CPU sharing
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct HitRange {
    pub pattern_idx: u32,
    pub start: i32,
    pub end: i32,
    pub _padding: u32, // Align to 16 bytes for GPU
}

/// Parameters passed via push constants
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct MyersParams {
    pub text_len: u32,
    pub pattern_length: u32,
    pub num_patterns: u32,
    pub k: u32, // Maximum allowed cost
    pub alpha_pattern: T,
    pub last_bit_shift: u32,
    pub max_hits: u32,
}

/// Type alias for pattern equality vectors
pub type PeqValue = T;

/// Core Myers bit-parallel algorithm step (scalar version for GPU)
///
/// This is the same algorithm as the SIMD version, but operates on scalar T values.
/// Each GPU thread executes this independently for its assigned pattern.
#[inline(always)]
fn myers_step_scalar(
    vp: T,
    vn: T,
    cost: T,
    eq: T,
    all_ones: T,
    last_bit_shift: u32,
    last_bit_mask: T,
) -> (T, T, T) {
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

/// Count the number of one bits in a T (popcount)
/// SPIR-V doesn't have T::count_ones in core, so we implement it
#[allow(unused)]
#[inline]
fn count_ones(mut x: T) -> u32 {
    let mut count = 0u32;
    while x != 0 {
        x &= x - 1; // Clear the lowest set bit
        count += 1;
    }
    count
}

/// Main CUDA/SPIR-V compute kernel for Myers bit-parallel search
///
/// Each thread processes one pattern against the shared text.
///
/// Memory layout:
/// - binding 0: text (read-only)
/// - binding 1: pattern equality vectors (PEQs) - [num_patterns * 256]T
/// - binding 2: output hit ranges (read-write)
/// - binding 3: atomic hit counter (read-write)
#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(256)))] // Match WORKGROUP_SIZE
pub fn myers_search_kernel(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] text: &[u8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] peqs: &[T],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hit_ranges: &mut [HitRange],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] hit_count: &mut [u32],
    #[spirv(push_constant)] params: &MyersParams,
) {
    let thread_id = global_id.x as usize;

    // Early exit if thread ID exceeds number of patterns
    if thread_id >= params.num_patterns as usize {
        return;
    }

    // Each thread processes one pattern
    let pattern_peqs_offset = thread_id * 256;
    let last_bit_mask = (1 as T) << params.last_bit_shift;

    // Initialize Myers state
    // `saturating_sub` is not supported on the SPIR-V target, so compute the
    // shift amount with a branchless equivalent.
    let pattern_length = params.pattern_length as usize;
    let shift_amount = if pattern_length >= 64 {
        0
    } else {
        64 - pattern_length
    };
    let length_mask = (!0 as T) >> shift_amount;
    let masked_alpha = params.alpha_pattern & length_mask;
    let mut vp = params.alpha_pattern;
    let mut vn: T = 0;
    let mut cost = count_ones(masked_alpha) as T;

    let all_ones: T = !0;

    // Track active range
    let mut is_active = false;
    let mut range_start = -1i32;

    // Process each position in text
    let text_len = params.text_len as usize;
    for pos in 0..text_len {
        // Read character
        let c = unsafe { *text.get_unchecked(pos) };

        // Look up pre-computed equality vector for this char
        let eq = unsafe { *peqs.get_unchecked(pattern_peqs_offset + c as usize) };

        // Execute Myers step
        let (vp_new, vn_new, cost_new) = myers_step_scalar(
            vp,
            vn,
            cost,
            eq,
            all_ones,
            params.last_bit_shift,
            last_bit_mask,
        );

        vp = vp_new;
        vn = vn_new;
        cost = cost_new;

        // Check if cost is within threshold
        let was_active = is_active;
        is_active = cost <= params.k as T;

        // Handle range transitions
        if is_active && !was_active {
            // Range starting
            range_start = pos as i32;
        } else if !is_active && was_active {
            // Range ending - emit hit
            // Use atomic to get unique slot
            #[cfg(target_arch = "spirv")]
            {
                use spirv_std::arch::atomic_i_add;
                use spirv_std::memory::{Scope, Semantics};
                let idx = unsafe {
                    atomic_i_add::<
                        u32,
                        { Scope::QueueFamily as u32 },
                        { Semantics::UNIFORM_MEMORY.bits() },
                    >(&mut hit_count[0], 1)
                };

                if idx < params.max_hits {
                    hit_ranges[idx as usize] = HitRange {
                        pattern_idx: thread_id as u32,
                        start: range_start,
                        end: (pos - 1) as i32,
                        _padding: 0,
                    };
                }
            }
        }
    }

    // Finalize: if still active at end, emit final range
    if is_active {
        #[cfg(target_arch = "spirv")]
        {
            use spirv_std::arch::atomic_i_add;
            use spirv_std::memory::{Scope, Semantics};
            let idx = unsafe {
                atomic_i_add::<
                    u32,
                    { Scope::QueueFamily as u32 },
                    { Semantics::UNIFORM_MEMORY.bits() },
                >(&mut hit_count[0], 1)
            };

            if idx < params.max_hits {
                hit_ranges[idx as usize] = HitRange {
                    pattern_idx: thread_id as u32,
                    start: range_start,
                    end: (text_len - 1) as i32,
                    _padding: 0,
                };
            }
        }
    }
}

// CPU version for testing
#[cfg(not(target_arch = "spirv"))]
pub fn myers_search_kernel_cpu(
    text: &[u8],
    peqs: &[T],
    hit_ranges: &mut [HitRange],
    hit_count: &mut u32,
    params: &MyersParams,
    thread_id: usize,
) {
    if thread_id >= params.num_patterns as usize {
        return;
    }

    let pattern_peqs_offset = thread_id * 256;
    let last_bit_mask = (1 as T) << params.last_bit_shift;

    let length_mask = (!0 as T) >> (64usize.saturating_sub(params.pattern_length as usize));
    let masked_alpha = params.alpha_pattern & length_mask;
    let mut vp = params.alpha_pattern;
    let mut vn: T = 0;
    let mut cost: T = masked_alpha.count_ones() as T;

    let all_ones: T = !0;

    let mut is_active = false;
    let mut range_start = -1i32;

    for pos in 0..params.text_len as usize {
        let c = text[pos];
        let eq = peqs[pattern_peqs_offset + c as usize];

        let (vp_new, vn_new, cost_new) = myers_step_scalar(
            vp,
            vn,
            cost,
            eq,
            all_ones,
            params.last_bit_shift,
            last_bit_mask,
        );

        vp = vp_new;
        vn = vn_new;
        cost = cost_new;

        let was_active = is_active;
        is_active = cost <= params.k as T;

        if is_active && !was_active {
            range_start = pos as i32;
        } else if !is_active && was_active {
            let idx = *hit_count;
            *hit_count += 1;

            if idx < params.max_hits {
                hit_ranges[idx as usize] = HitRange {
                    pattern_idx: thread_id as u32,
                    start: range_start,
                    end: (pos - 1) as i32,
                    _padding: 0,
                };
            }
        }
    }

    if is_active {
        let idx = *hit_count;
        *hit_count += 1;

        if idx < params.max_hits {
            hit_ranges[idx as usize] = HitRange {
                pattern_idx: thread_id as u32,
                start: range_start,
                end: (params.text_len - 1) as i32,
                _padding: 0,
            };
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
    fn test_count_ones() {
        assert_eq!(count_ones(0), 0);
        assert_eq!(count_ones(1), 1);
        assert_eq!(count_ones(0b1010), 2);
        assert_eq!(count_ones(0b11111111), 8);
        assert_eq!(count_ones(!0u64), 64);
    }
}
