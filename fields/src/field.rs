use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ops::DivAssign;

use num_bigint::BigUint;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{from_integer_types, QuotientMap};

pub trait Field:
    From<Self>
    + Default
    + Copy
    + Clone
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + 'static
    + Eq
    + Hash
    + Send
    + Sync
    + Debug
    + Display
    + Serialize
    + DeserializeOwned
{
    const ZERO: Self;

    const ONE: Self;

    const TWO: Self;

    const NEG_ONE: Self;

    const GENERATOR: Self;

    #[must_use]
    #[inline(always)]
    fn from_bool(b: bool) -> Self {
        if b {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

    #[must_use]
    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    #[must_use]
    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::ONE
    }

    #[must_use]
    #[inline(always)]
    fn double(&self) -> Self {
        *self + *self
    }

    #[must_use]
    #[inline(always)]
    fn square(&self) -> Self {
        *self * *self
    }

    #[must_use]
    #[inline(always)]
    fn cube(&self) -> Self {
        self.square() * *self
    }

    fn exp_u64(&self, exp: u64) -> Self {
        let mut result = Self::ONE;
        let mut base = *self;

        for j in 0..bits_u64(exp) {
            if (exp >> j) & 1 == 1 {
                result *= base;
            }
            base = base.square();
        }

        return result;

        pub fn bits_u64(n: u64) -> usize {
            (64 - n.leading_zeros()) as usize
        }
    }

    #[must_use]
    #[inline(always)]
    fn exp_const<const EXP: u64>(&self) -> Self {
        match EXP {
            0 => Self::ONE,
            1 => *self,
            2 => self.square(),
            3 => self.cube(),
            4 => self.square().square(),
            5 => *self * self.square().square(),
            6 => self.square().cube(),
            7 => {
                let sq = self.square();
                let cb = sq * *self;
                let qrt = sq.square();
                cb * qrt
            }
            _ => self.exp_u64(EXP),
        }
    }

    #[must_use]
    #[inline(always)]
    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    #[must_use]
    fn try_inverse(&self) -> Option<Self>;

    #[must_use]
    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }
}

pub trait PrimeField: Field + Ord {
    #[must_use]
    fn as_canonical_biguint(&self) -> BigUint;
}

pub trait PrimeField64:
    PrimeField
    + QuotientMap<u8>
    + QuotientMap<u16>
    + QuotientMap<u32>
    + QuotientMap<u64>
    + QuotientMap<usize>
    + QuotientMap<i8>
    + QuotientMap<i16>
    + QuotientMap<i32>
    + QuotientMap<i64>
    + QuotientMap<isize>
{
    const ORDER_U64: u64;

    #[must_use]
    fn as_canonical_u64(&self) -> u64;

    #[must_use]
    #[inline(always)]
    fn to_unique_u64(&self) -> u64 {
        self.as_canonical_u64()
    }

    from_integer_types!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
}

pub trait ExtensionField<Base: Field>:
    Field
    + Add<Base, Output = Self>
    + AddAssign<Base>
    + Sub<Base, Output = Self>
    + SubAssign<Base>
    + Mul<Base, Output = Self>
    + MulAssign<Base>
    + Div<Base, Output = Self>
    + DivAssign<Base>
{
    const NON_RESIDUE: u64;

    fn as_basis_coefficients_slice(&self) -> &[Base];

    fn from_basis_singleton(value: Base) -> Self;

    fn from_basis_coefficients_slice(value: &[Base]) -> Self;

    fn from_basis_coefficients_fn(f: impl Fn(usize) -> Base) -> Self;
}
