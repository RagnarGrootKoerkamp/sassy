//! Implement our own wrapper trait around SIMD types

use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Shl, Shr, Sub};
use wide::{u8x32, u16x16, u32x8, u64x4};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use wide::{u16x32, u32x16, u64x8};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
use std::arch::x86_64::*;

pub trait SimdBackend: Copy + 'static + Send + Sync + Default + std::fmt::Debug {
    type Simd: Copy
        + Add<Output = Self::Simd>
        + Sub<Output = Self::Simd>
        + BitAnd<Output = Self::Simd>
        + BitOr<Output = Self::Simd>
        + BitAndAssign<Self::Simd>
        + BitOrAssign<Self::Simd>
        + BitXor<Output = Self::Simd>
        + Shl<i32, Output = Self::Simd>
        + Shr<i32, Output = Self::Simd>
        + Shl<u32, Output = Self::Simd>
        + Shr<u32, Output = Self::Simd>
        + PartialEq;

    type Scalar: Copy + PartialEq + std::fmt::Debug;
    type Array: AsRef<[Self::Scalar]> + AsMut<[Self::Scalar]> + Copy;

    const LANES: usize;
    const LIMB_BITS: usize;

    fn mask_word_to_scalar(word: u64) -> Self::Scalar;
    fn scalar_from_i64(value: i64) -> Self::Scalar;
    fn default_array() -> Self::Array;
    fn from_array(arr: Self::Array) -> Self::Simd;
    fn to_array(vec: Self::Simd) -> Self::Array;
    fn simd_eq(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd;
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd;
    fn scalar_to_u64(value: Self::Scalar) -> u64;
    fn splat_all_ones() -> Self::Simd;
    fn splat_zero() -> Self::Simd;
    fn splat_one() -> Self::Simd;
    fn splat_scalar(value: Self::Scalar) -> Self::Simd;
    fn lanes_with_zero(vec: Self::Simd) -> u64;
}

macro_rules! impl_wide_backend {
    ($name:ident, $simd_ty:ty, $scalar:ty, $lanes:expr, $bits:expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl SimdBackend for $name {
            type Simd = $simd_ty;
            type Scalar = $scalar;
            type Array = [$scalar; $lanes];

            const LANES: usize = $lanes;
            const LIMB_BITS: usize = $bits;

            #[inline(always)]
            fn mask_word_to_scalar(word: u64) -> Self::Scalar {
                word as $scalar
            }
            #[inline(always)]
            fn scalar_from_i64(value: i64) -> Self::Scalar {
                value as $scalar
            }
            #[inline(always)]
            fn default_array() -> Self::Array {
                [0; $lanes]
            }
            #[inline(always)]
            fn from_array(arr: Self::Array) -> Self::Simd {
                <$simd_ty>::new(arr.into())
            }
            #[inline(always)]
            fn to_array(vec: Self::Simd) -> Self::Array {
                vec.to_array()
            }
            #[inline(always)]
            fn splat_all_ones() -> Self::Simd {
                <$simd_ty>::splat(!0 as $scalar)
            }
            #[inline(always)]
            fn splat_zero() -> Self::Simd {
                <$simd_ty>::splat(0)
            }
            #[inline(always)]
            fn splat_one() -> Self::Simd {
                <$simd_ty>::splat(1)
            }
            #[inline(always)]
            fn splat_scalar(value: Self::Scalar) -> Self::Simd {
                <$simd_ty>::splat(value)
            }
            #[inline(always)]
            fn scalar_to_u64(value: Self::Scalar) -> u64 {
                value as u64
            }
            #[inline(always)]
            fn simd_eq(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
                lhs.simd_eq(rhs)
            }
            #[inline(always)]
            fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
                lhs.simd_gt(rhs)
            }
            #[inline(always)]
            fn lanes_with_zero(vec: Self::Simd) -> u64 {
                vec.simd_eq(<$simd_ty>::splat(0)).to_bitmask() as u64
            }
        }
    };
}

// Define Backends
impl_wide_backend!(U64, u64x4, u64, 4, 64);
impl_wide_backend!(U32, u32x8, u32, 8, 32);
impl_wide_backend!(U16, u16x16, u16, 16, 16);
impl_wide_backend!(U8, u8x32, u8, 32, 8);

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl_wide_backend!(U64_512, u64x8, u64, 8, 64);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl_wide_backend!(U32_512, u32x16, u32, 16, 32);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl_wide_backend!(U16_512, u16x32, u16, 32, 16);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl_wide_backend!(U8_512, u8x64, u8, 64, 8);
