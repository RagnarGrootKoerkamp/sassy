use std::convert::TryInto;
use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Shl, Shr, Sub};
use wide::{CmpEq, CmpGt, i8x32, u8x32, u16x16, u32x8, u64x4};

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
        + CmpEq<Output = Self::Simd>
        + PartialEq;

    type Scalar: Copy + PartialEq + std::fmt::Debug;
    type LaneArray: AsRef<[Self::Scalar]> + AsMut<[Self::Scalar]> + Copy + Default;
    type PatternBlock: Copy + std::fmt::Debug;

    const LANES: usize;
    const LIMB_BITS: usize;

    fn mask_word_to_scalar(word: u64) -> Self::Scalar;
    fn scalar_from_i64(value: i64) -> Self::Scalar;
    fn from_array(arr: Self::LaneArray) -> Self::Simd;
    fn to_array(vec: Self::Simd) -> Self::LaneArray;
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd;
    fn to_pattern_block(slice: &[u8]) -> Self::PatternBlock;
    fn scalar_to_u64(value: Self::Scalar) -> u64;
    fn splat_all_ones() -> Self::Simd;
    fn splat_zero() -> Self::Simd;
    fn splat_one() -> Self::Simd;
    fn splat_scalar(value: Self::Scalar) -> Self::Simd;
    fn lanes_with_zero(vec: Self::Simd) -> u64;
}

macro_rules! impl_wide_backend {
    ($name:ident, $simd_ty:ty, $scalar:ty, $lanes:expr, $bits:expr, $pattern:ty) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl SimdBackend for $name {
            type Simd = $simd_ty;
            type Scalar = $scalar;
            type LaneArray = [$scalar; $lanes];
            type PatternBlock = $pattern;

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
            fn from_array(arr: Self::LaneArray) -> Self::Simd {
                <$simd_ty>::new(arr)
            }
            #[inline(always)]
            fn to_array(vec: Self::Simd) -> Self::LaneArray {
                vec.to_array()
            }
            #[inline(always)]
            fn to_pattern_block(slice: &[u8]) -> Self::PatternBlock {
                slice.try_into().expect("Invalid slice size")
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

macro_rules! impl_u8_wrapper {
    ($name:ident, $wide_ty:ty, $signed_ty:ty, $lanes:expr) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        #[repr(transparent)]
        pub struct $name(pub $wide_ty);

        impl Add for $name {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }
        impl BitAnd for $name {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }
        impl BitOr for $name {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
        impl BitXor for $name {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }
        impl BitAndAssign for $name {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }
        impl BitOrAssign for $name {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }
        impl CmpEq for $name {
            type Output = Self;
            #[inline(always)]
            fn simd_eq(self, rhs: Self) -> Self {
                Self(self.0.simd_eq(rhs.0))
            }
        }
        impl Shl<u32> for $name {
            type Output = Self;
            #[inline(always)]
            fn shl(self, rhs: u32) -> Self {
                if rhs >= 8 {
                    return Self::splat(0);
                }
                let mut arr = self.to_array();
                for byte in &mut arr {
                    *byte = byte.wrapping_shl(rhs);
                }
                Self(<$wide_ty>::new(arr))
            }
        }
        impl Shr<u32> for $name {
            type Output = Self;
            #[inline(always)]
            fn shr(self, rhs: u32) -> Self {
                if rhs >= 8 {
                    return Self::splat(0);
                }
                let mut arr = self.to_array();
                for byte in &mut arr {
                    *byte = byte.wrapping_shr(rhs);
                }
                Self(<$wide_ty>::new(arr))
            }
        }
        impl Shl<i32> for $name {
            type Output = Self;
            #[inline(always)]
            fn shl(self, rhs: i32) -> Self {
                self << (rhs as u32)
            }
        }
        impl Shr<i32> for $name {
            type Output = Self;
            #[inline(always)]
            fn shr(self, rhs: i32) -> Self {
                self >> (rhs as u32)
            }
        }

        impl $name {
            #[inline(always)]
            pub fn new(arr: [u8; $lanes]) -> Self {
                Self(<$wide_ty>::new(arr))
            }
            #[inline(always)]
            pub fn to_array(self) -> [u8; $lanes] {
                self.0.to_array()
            }
            #[inline(always)]
            pub fn splat(val: u8) -> Self {
                Self(<$wide_ty>::splat(val))
            }
            #[inline(always)]
            pub fn simd_gt(self, rhs: Self) -> Self {
                use std::mem::transmute;
                let mask = <$signed_ty>::splat(1 << 7);
                let a: $signed_ty = unsafe { transmute(self.0) };
                let b: $signed_ty = unsafe { transmute(rhs.0) };
                Self(unsafe { transmute(CmpGt::simd_gt(a ^ mask, b ^ mask)) })
            }
            #[inline(always)]
            pub fn to_bitmask(self) -> u64 {
                self.0.to_bitmask() as u64
            }
        }
    };
}

impl_u8_wrapper!(WrapperU8x32, u8x32, i8x32, 32);

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Avx512U8(pub __m512i);

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Default for Avx512U8 {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(0)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl PartialEq for Avx512U8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let mut a = [0u64; 8];
            let mut b = [0u64; 8];
            _mm512_storeu_si512(a.as_mut_ptr() as *mut _, self.0);
            _mm512_storeu_si512(b.as_mut_ptr() as *mut _, other.0);
            a == b
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Avx512U8 {
    #[inline(always)]
    pub fn new(arr: [u8; 64]) -> Self {
        unsafe { Self(_mm512_loadu_si512(arr.as_ptr() as *const _)) }
    }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 64] {
        unsafe {
            let mut arr = [0u8; 64];
            _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0);
            arr
        }
    }

    #[inline(always)]
    pub fn splat(val: u8) -> Self {
        unsafe { Self(_mm512_set1_epi8(val as i8)) }
    }

    #[inline(always)]
    pub fn simd_gt(self, rhs: Self) -> Self {
        unsafe {
            let mask = _mm512_cmp_epu8_mask(self.0, rhs.0, 6); // is this safe to assume 6?
            Self(_mm512_maskz_set1_epi8(mask, -1))
        }
    }

    #[inline(always)]
    pub fn to_bitmask(self) -> u64 {
        unsafe { _mm512_movepi8_mask(self.0) as u64 }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Add for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_add_epi8(self.0, rhs.0)) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Sub for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_sub_epi8(self.0, rhs.0)) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl BitAnd for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, rhs.0)) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl BitOr for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_or_si512(self.0, rhs.0)) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl BitXor for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { Self(_mm512_xor_si512(self.0, rhs.0)) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl BitAndAssign for Avx512U8 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm512_and_si512(self.0, rhs.0) };
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl BitOrAssign for Avx512U8 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm512_or_si512(self.0, rhs.0) };
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Shl<u32> for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        if rhs >= 8 {
            return Self::splat(0);
        }
        let mut arr = self.to_array();
        for byte in &mut arr {
            *byte = byte.wrapping_shl(rhs);
        }
        Self::new(arr)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Shr<u32> for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        if rhs >= 8 {
            return Self::splat(0);
        }
        let mut arr = self.to_array();
        for byte in &mut arr {
            *byte = byte.wrapping_shr(rhs);
        }
        Self::new(arr)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Shl<i32> for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: i32) -> Self {
        self << (rhs as u32)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl Shr<i32> for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        self >> (rhs as u32)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl CmpEq for Avx512U8 {
    type Output = Self;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self {
        unsafe {
            let mask = _mm512_cmpeq_epu8_mask(self.0, rhs.0);
            Self(_mm512_maskz_set1_epi8(mask, -1))
        }
    }
}

// Define Backends
impl_wide_backend!(I32x8Backend, u32x8, u32, 8, 32, u8x32);
impl_wide_backend!(I64x4Backend, u64x4, u64, 4, 64, [u8; 64]);
impl_wide_backend!(I16x16Backend, u16x16, u16, 16, 16, [u8; 16]);
impl_wide_backend!(I8x32Backend, WrapperU8x32, u8, 32, 8, [u8; 8]);

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl_wide_backend!(I32x16Backend, u32x16, u32, 16, 32, [u8; 64]);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl_wide_backend!(I64x8Backend, u64x8, u64, 8, 64, [u8; 64]);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl_wide_backend!(I8x64Backend, Avx512U8, u8, 64, 8, [u8; 8]);
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl_wide_backend!(I16x32Backend, u16x32, u16, 32, 16, [u8; 64]);

// --- Aliases ---
pub type U8 = I8x32Backend;
pub type U32 = I32x8Backend;
pub type U64 = I64x4Backend;
pub type U16 = I16x16Backend;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub type U8_512 = I8x64Backend;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type U32_512 = I32x16Backend;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type U64_512 = I64x8Backend;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
pub type U16_512 = I16x32Backend;
