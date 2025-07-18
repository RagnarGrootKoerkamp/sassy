#![allow(unused)] // TODO: Drop this

use pa_types::{Cost, I};

type VBase = u64;

#[derive(Clone, Default, Copy, PartialEq, Eq, Debug)]
pub struct V(pub VBase, pub VBase);

impl V {
    #[inline(always)]
    pub fn zero() -> Self {
        V(0, 0)
    }
    #[inline(always)]
    pub fn one() -> Self {
        V(<VBase>::MAX, 0)
    }
    #[inline(always)]
    pub fn from(p: VBase, m: VBase) -> Self {
        V(p, m)
    }
    #[inline(always)]
    pub fn value(&self) -> Cost {
        self.0.count_ones() as Cost - self.1.count_ones() as Cost
    }
    /// Value of the first `j` bits.
    /// NOTE: Requires `0 <= j < VBase::BITS`.
    #[inline(always)]
    pub fn value_of_prefix(&self, j: I) -> Cost {
        debug_assert!(0 <= j && j < <VBase>::BITS as I);
        let mask = (1 << j) - 1;
        (self.0 & mask).count_ones() as Cost - (self.1 & mask).count_ones() as Cost
    }
    /// Value of the last `j` bits.
    /// NOTE: Requires `j > 0`.
    #[inline(always)]
    pub fn value_of_suffix(&self, j: I) -> Cost {
        debug_assert!(0 < j && j <= <VBase>::BITS as I);
        let mask = !(((1 as VBase) << (<VBase>::BITS as I - j)).wrapping_sub(1));
        (self.0 & mask).count_ones() as Cost - (self.1 & mask).count_ones() as Cost
    }
    #[inline(always)]
    pub fn pm(&self) -> (VBase, VBase) {
        (self.0, self.1)
    }
    #[inline(always)]
    pub fn p(&self) -> VBase {
        self.0
    }
    #[inline(always)]
    pub fn m(&self) -> VBase {
        self.1
    }
    pub fn value_to(v: &[Self], j: I) -> Cost {
        let mut s = 0;
        for vj in &v[0..j as usize / 64] {
            s += vj.value();
        }
        if j % 64 != 0 {
            s += v[j as usize / 64].value_of_prefix(j % 64);
        }
        s
    }
    pub fn value_from(v: &[Self], j: I) -> Cost {
        let mut s = 0;
        if j % 64 != 0 {
            s += v[j as usize / 64].value_of_suffix(64 - j % 64);
        }
        for vj in &v[(j as usize).div_ceil(64)..] {
            s += vj.value();
        }
        s
    }
}

type HBase = u64;

#[derive(Clone, Copy)]
pub struct H(pub HBase, pub HBase);

impl H {
    #[inline(always)]
    pub fn zero() -> Self {
        H(0, 0)
    }
    #[inline(always)]
    pub fn one() -> Self {
        H(1, 0)
    }
    #[inline(always)]
    pub fn from(p: HBase, m: HBase) -> Self {
        H(p as HBase, m as HBase)
    }
    #[inline(always)]
    pub fn value(&self) -> Cost {
        self.0 as Cost - self.1 as Cost
    }
    #[inline(always)]
    pub fn p(&self) -> HBase {
        self.0 as HBase
    }
    #[inline(always)]
    pub fn m(&self) -> HBase {
        self.1 as HBase
    }
    #[inline(always)]
    pub fn pm(&self) -> (HBase, HBase) {
        (self.p(), self.m())
    }
}
