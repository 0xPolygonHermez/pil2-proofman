use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub trait Field:
    From<Self>
    + Default
    + Clone
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + 'static
    + Copy
    + Div<Self, Output = Self>
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

    fn from_bool(b: bool) -> Self;

    fn from_u64(int: u64) -> Self;

    fn double(&self) -> Self;

    fn square(&self) -> Self;

    fn inverse(&self) -> Self;

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
}

pub trait PrimeField: Field + Ord {
    #[must_use]
    fn as_canonical_biguint(&self) -> BigUint;
}

pub trait PrimeField64: PrimeField {
    const ORDER_U64: u64;

    #[must_use]
    fn as_canonical_u64(&self) -> u64;

    #[must_use]
    #[inline(always)]
    fn to_unique_u64(&self) -> u64 {
        self.as_canonical_u64()
    }
}
