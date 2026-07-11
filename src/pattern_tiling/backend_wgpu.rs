use crate::pattern_tiling::backend::SimdBackend;

/// WGPU backend for GPU execution
///
/// Like the CUDA backend, each GPU thread handles a single scalar `u64` value
/// (one pattern per thread), so the "SIMD" operations here are just plain
/// scalar arithmetic/bitwise operations. This type exists purely so the shared
/// `SimdBackend` abstraction can be reused for documentation/testing on the
/// host; the actual GPU-side kernel logic lives in the `gpu_kernel` crate,
/// which is compiled to SPIR-V independently.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuBackend;

impl SimdBackend for WgpuBackend {
    // Each GPU thread processes a scalar value.
    type Simd = u64;
    type Scalar = u64;
    type Array = [u64; 1];

    // Each thread is effectively 1 "lane".
    const LANES: usize = 1;
    const LIMB_BITS: usize = 64;

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word
    }

    #[inline(always)]
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value as u64
    }

    #[inline(always)]
    fn default_array() -> Self::Array {
        [0]
    }

    #[inline(always)]
    fn from_array(arr: Self::Array) -> Self::Simd {
        arr[0]
    }

    #[inline(always)]
    fn to_array(vec: Self::Simd) -> Self::Array {
        [vec]
    }

    #[inline(always)]
    fn simd_eq(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        if lhs == rhs {
            !0u64
        } else {
            0u64
        }
    }

    #[inline(always)]
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        // Return all-ones or all-zeros like SIMD comparison.
        if lhs > rhs {
            !0u64
        } else {
            0u64
        }
    }

    #[inline(always)]
    fn scalar_to_u64(value: Self::Scalar) -> u64 {
        value
    }

    #[inline(always)]
    fn splat_all_ones() -> Self::Simd {
        !0u64
    }

    #[inline(always)]
    fn splat_zero() -> Self::Simd {
        0u64
    }

    #[inline(always)]
    fn splat_one() -> Self::Simd {
        1u64
    }

    #[inline(always)]
    fn splat_scalar(value: Self::Scalar) -> Self::Simd {
        value
    }

    #[inline(always)]
    fn lanes_with_zero(vec: Self::Simd) -> u64 {
        // For scalar, either 0 (if non-zero) or 1 (if zero).
        if vec == 0 {
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_backend_basic() {
        let a = WgpuBackend::splat_scalar(42);
        let b = WgpuBackend::splat_scalar(10);

        assert_eq!(a, 42);
        assert_eq!(b, 10);

        let sum = a.wrapping_add(b);
        assert_eq!(sum, 52);

        let cmp = WgpuBackend::simd_gt(a, b);
        assert_eq!(cmp, !0u64); // All ones for true
    }

    #[test]
    fn test_wgpu_backend_bitwise() {
        let a = WgpuBackend::splat_scalar(0b1010);
        let b = WgpuBackend::splat_scalar(0b1100);

        assert_eq!(a & b, 0b1000);
        assert_eq!(a | b, 0b1110);
        assert_eq!(a ^ b, 0b0110);
    }

    #[test]
    fn test_wgpu_backend_shifts() {
        let a = WgpuBackend::splat_scalar(0b1010);

        assert_eq!(a << 1, 0b10100);
        assert_eq!(a >> 1, 0b0101);
    }
}
