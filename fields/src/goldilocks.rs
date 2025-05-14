use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use std::ops::DivAssign;

use num_bigint::BigUint;
use p3_goldilocks::Goldilocks as P3Goldilocks;
use p3_field::Field as P3Field;
use p3_field::PrimeField64 as P3PrimeField64;
use p3_field::PrimeCharacteristicRing as P3PrimeCharacteristicRing;

use rand::distr::{Distribution, StandardUniform};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{Field, PrimeField, PrimeField64};

const P: u64 = 0xFFFF_FFFF_0000_0001;

#[derive(Copy, Clone, Default, Serialize, Deserialize)]
pub struct Goldilocks {
    pub(crate) value: P3Goldilocks,
}

impl Goldilocks {
    pub(crate) const fn new(value: P3Goldilocks) -> Self {
        Self { value }
    }
}

impl PartialEq for Goldilocks {
    fn eq(&self, other: &Self) -> bool {
        self.value.as_canonical_u64() == other.value.as_canonical_u64()
    }
}

impl Eq for Goldilocks {}

impl Hash for Goldilocks {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.as_canonical_u64());
    }
}

impl Ord for Goldilocks {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u64().cmp(&other.as_canonical_u64())
    }
}

impl PartialOrd for Goldilocks {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Goldilocks {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value, f)
    }
}

impl Debug for Goldilocks {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl Field for Goldilocks {
    const ZERO: Self = Self::new(P3Goldilocks::ZERO);
    const ONE: Self = Self::new(P3Goldilocks::ONE);
    const TWO: Self = Self::new(P3Goldilocks::TWO);
    const NEG_ONE: Self = Self::new(P3Goldilocks::NEG_ONE);
    const GENERATOR: Self = Self::new(P3Goldilocks::GENERATOR);

    #[must_use]
    #[inline(always)]
    fn from_u64(int: u64) -> Self {
        Self::new(P3PrimeCharacteristicRing::from_u64(int))
    }

    #[must_use]
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Some(Self::new(self.value.inverse()))
    }
}

impl PrimeField for Goldilocks {
    fn as_canonical_biguint(&self) -> BigUint {
        P3PrimeField64::as_canonical_u64(&self.value).into()
    }
}

impl PrimeField64 for Goldilocks {
    const ORDER_U64: u64 = P;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        P3Goldilocks::as_canonical_u64(&self.value)
    }
}

impl Add for Goldilocks {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}

impl AddAssign for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value = self.value + rhs.value;
    }
}

impl Sub for Goldilocks {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value - rhs.value)
    }
}

impl SubAssign for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value = self.value - rhs.value;
    }
}

impl Neg for Goldilocks {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(P3Goldilocks::neg(self.value))
    }
}

impl Mul for Goldilocks {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.value * rhs.value)
    }
}

impl MulAssign for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.value = self.value * rhs.value;
    }
}

impl Div for Goldilocks {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        Self::new(P3Goldilocks::div(self.value, rhs.value))
    }
}

impl DivAssign for Goldilocks {
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Distribution<Goldilocks> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Goldilocks {
        loop {
            let next_u64 = rng.next_u64();
            let is_canonical = next_u64 < Goldilocks::ORDER_U64;
            if is_canonical {
                return Goldilocks::from_u64(next_u64);
            }
        }
    }
}
