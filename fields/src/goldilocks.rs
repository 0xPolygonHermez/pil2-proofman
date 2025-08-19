use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use std::ops::DivAssign;

use num_bigint::BigUint;

#[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
use proofman_starks_lib_c::{
    goldilocks_add_assign_ffi, goldilocks_add_ffi, goldilocks_div_assign_ffi, goldilocks_div_ffi, goldilocks_inv_ffi,
    goldilocks_mul_assign_ffi, goldilocks_mul_ffi, goldilocks_neg_ffi, goldilocks_sub_assign_ffi, goldilocks_sub_ffi,
};
use serde::{Deserialize, Serialize};

use crate::{quotient_map_small_int, Field, PrimeField, PrimeField64, QuotientMap};

#[derive(Copy, Clone, Default, Serialize, Deserialize)]
pub struct Goldilocks(u64);

#[inline(always)]
pub fn branch_hint() {
    // NOTE: These are the currently supported assembly architectures. See the
    // [nightly reference](https://doc.rust-lang.org/nightly/reference/inline-assembly.html) for
    // the most up-to-date list.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!("", options(nomem, nostack, preserves_flags));
    }
}

impl Goldilocks {
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    const NEG_ORDER: u64 = 0xFFFF_FFFF;
    pub const SHIFT: u64 = 7;

    pub const W: [u64; 33] = [
        1,
        18446744069414584320,
        281474976710656,
        16777216,
        4096,
        64,
        8,
        2198989700608,
        4404853092538523347,
        6434636298004421797,
        4255134452441852017,
        9113133275150391358,
        4355325209153869931,
        4308460244895131701,
        7126024226993609386,
        1873558160482552414,
        8167150655112846419,
        5718075921287398682,
        3411401055030829696,
        8982441859486529725,
        1971462654193939361,
        6553637399136210105,
        8124823329697072476,
        5936499541590631774,
        2709866199236980323,
        8877499657461974390,
        3757607247483852735,
        4969973714567017225,
        2147253751702802259,
        2530564950562219707,
        1905180297017055339,
        3524815499551269279,
        7277203076849721926,
    ];

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn add_internal(a: u64, b: u64) -> u64 {
        let (sum, over) = a.overflowing_add(b);
        let (mut sum, over) = sum.overflowing_add(u64::from(over) * Self::NEG_ORDER);
        if over {
            branch_hint();
            sum += Self::NEG_ORDER;
        }
        sum
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn sub_internal(a: u64, b: u64) -> u64 {
        let (diff, under) = a.overflowing_sub(b);
        let (mut diff, under) = diff.overflowing_sub(u64::from(under) * Self::NEG_ORDER);
        if under {
            branch_hint();
            diff -= Self::NEG_ORDER;
        }
        diff
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn neg_internal(a: u64) -> u64 {
        Self::sub_internal(0, a)
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn mul_internal(a: u64, b: u64) -> u64 {
        let res = u128::from(a) * u128::from(b);
        let rl = res as u64;
        let rh = (res >> 64) as u64;

        let rhh = rh >> 32;
        let rhl = rh & Self::NEG_ORDER;

        let (mut aux1, borrow) = rl.overflowing_sub(rhh);
        if borrow {
            branch_hint();
            aux1 -= Self::NEG_ORDER;
        }

        let aux = rhl * Self::NEG_ORDER;

        let (result, carry) = aux1.overflowing_add(aux);
        result + Self::NEG_ORDER * u64::from(carry)
    }
}

impl PartialEq for Goldilocks {
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u64() == other.as_canonical_u64()
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
        Display::fmt(&self.0, f)
    }
}

impl Debug for Goldilocks {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

quotient_map_small_int!(Goldilocks, u64, [u8, u16, u32]);
quotient_map_small_int!(Goldilocks, i64, [i8, i16, i32]);

impl QuotientMap<u64> for Goldilocks {
    #[inline]
    fn from_int(int: u64) -> Self {
        Self::new(int)
    }

    #[inline]
    fn from_canonical_checked(int: u64) -> Option<Self> {
        (int < Self::ORDER_U64).then(|| Self::new(int))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: u64) -> Self {
        Self::new(int)
    }
}

impl QuotientMap<i64> for Goldilocks {
    #[inline]
    fn from_int(int: i64) -> Self {
        if int >= 0 {
            Self::new(int as u64)
        } else {
            Self::new(Self::ORDER_U64.wrapping_add_signed(int))
        }
    }

    #[inline]
    fn from_canonical_checked(int: i64) -> Option<Self> {
        const POS_BOUND: i64 = (Goldilocks::P >> 1) as i64;
        const NEG_BOUND: i64 = -POS_BOUND;
        match int {
            0..=POS_BOUND => Some(Self::new(int as u64)),
            NEG_BOUND..0 => Some(Self::new(Self::ORDER_U64.wrapping_add_signed(int))),
            _ => None,
        }
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: i64) -> Self {
        Self::from_int(int)
    }
}

impl Field for Goldilocks {
    const ZERO: Self = Self::new(0);
    const ONE: Self = Self::new(1);
    const TWO: Self = Self::new(2);
    const NEG_ONE: Self = Self::new(Self::ORDER_U64 - 1);
    const GENERATOR: Self = Self::new(7);

    fn inverse(&self) -> Self {
        let mut t: u64 = 0;
        let mut r = Self::P;
        let mut newt = 1;
        let mut newr = self.0;

        let mut q: u64;
        let mut aux1: u64;
        let mut aux2: u64;

        while newr != 0 {
            q = r / newr;
            aux1 = t;
            aux2 = newt;
            t = aux2;
            newt = Self::sub_internal(aux1, Self::mul_internal(q, aux2));

            aux1 = r;
            aux2 = newr;
            r = aux2;

            newr = Self::sub_internal(aux1, Self::mul_internal(q, aux2));
        }

        Goldilocks(t)
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            Some(unsafe { Self::new(goldilocks_inv_ffi(&self.0)) })
        }

        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            Some(Self::new(self.inverse().0))
        }
    }
}

impl PrimeField for Goldilocks {
    fn as_canonical_biguint(&self) -> BigUint {
        self.0.into()
    }
}

impl PrimeField64 for Goldilocks {
    const ORDER_U64: u64 = Self::P;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        let mut c = self.0;
        if c >= Self::ORDER_U64 {
            c -= Self::ORDER_U64;
        }
        c
    }
}

impl Add for Goldilocks {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            Self(unsafe { goldilocks_add_ffi(&self.0, &rhs.0) })
        }

        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            Self(Self::add_internal(self.0, rhs.0))
        }
    }
}

impl AddAssign for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            unsafe { goldilocks_add_assign_ffi(&mut self.0, &self.0, &rhs.0) }
        }
        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            self.0 = Self::add_internal(self.0, rhs.0);
        }
    }
}

impl Sub for Goldilocks {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            Self(unsafe { goldilocks_sub_ffi(&self.0, &rhs.0) })
        }

        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            Self(Self::sub_internal(self.0, rhs.0))
        }
    }
}

impl SubAssign for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            unsafe { goldilocks_sub_assign_ffi(&mut self.0, &self.0, &rhs.0) }
        }
        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            self.0 = Self::sub_internal(self.0, rhs.0);
        }
    }
}

impl Neg for Goldilocks {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            Self(unsafe { goldilocks_neg_ffi(&self.0) })
        }
        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            Self(Self::neg_internal(self.0))
        }
    }
}

impl Mul for Goldilocks {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            Self(unsafe { goldilocks_mul_ffi(&self.0, &rhs.0) })
        }

        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            Self(Self::mul_internal(self.0, rhs.0))
        }
    }
}

impl MulAssign for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            unsafe { goldilocks_mul_assign_ffi(&mut self.0, &self.0, &rhs.0) }
        }

        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            self.0 = Self::mul_internal(self.0, rhs.0)
        }
    }
}

impl Div for Goldilocks {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            Self(unsafe { goldilocks_div_ffi(&self.0, &rhs.0) })
        }

        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            Self(Self::mul_internal(self.0, rhs.inverse().0))
        }
    }
}

impl DivAssign for Goldilocks {
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div_assign(&mut self, rhs: Self) {
        #[cfg(all(target_arch = "x86_64", not(feature = "verify")))]
        {
            unsafe { goldilocks_div_assign_ffi(&mut self.0, &self.0, &rhs.0) }
        }
        #[cfg(any(not(target_arch = "x86_64"), feature = "verify"))]
        {
            self.0 = Self::mul_internal(self.0, rhs.inverse().0)
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

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

    #[test]
    pub fn test_add_neg_sub_mul()
    where
        StandardUniform: Distribution<Goldilocks>,
    {
        let mut rng = rand::rng();
        let x = rng.random::<Goldilocks>();
        let y = rng.random::<Goldilocks>();
        let z = rng.random::<Goldilocks>();
        assert_eq!(Goldilocks::ONE + Goldilocks::NEG_ONE, Goldilocks::ZERO);
        assert_eq!(x + (-x), Goldilocks::ZERO);
        assert_eq!(Goldilocks::ONE + Goldilocks::ONE, Goldilocks::TWO);
        assert_eq!(-x, Goldilocks::ZERO - x);
        assert_eq!(x + x, x * Goldilocks::TWO);
        assert_eq!(x * Goldilocks::TWO, x.double());
        // assert_eq!(x, x.halve() * Goldilocks::TWO);
        assert_eq!(x * (-x), -x.square());
        assert_eq!(x + y, y + x);
        assert_eq!(x * Goldilocks::ZERO, Goldilocks::ZERO);
        assert_eq!(x * Goldilocks::ONE, x);
        assert_eq!(x * y, y * x);
        assert_eq!(x * (y * z), (x * y) * z);
        assert_eq!(x - (y + z), (x - y) - z);
        assert_eq!((x + y) - z, x + (y - z));
        assert_eq!(x * (y + z), x * y + x * z);
        // assert_eq!(x + y + z + x + y + z, [x, x, y, y, z, z].into_iter().sum());
    }
    #[test]
    pub fn test_inv_div()
    where
        StandardUniform: Distribution<Goldilocks>,
    {
        let mut rng = rand::rng();
        let x = rng.random::<Goldilocks>();
        let y = rng.random::<Goldilocks>();
        let z = rng.random::<Goldilocks>();
        assert_eq!(x * x.inverse(), Goldilocks::ONE);
        assert_eq!(x.inverse() * x, Goldilocks::ONE);
        assert_eq!(x.square().inverse(), x.inverse().square());
        assert_eq!((x / y) * y, x);
        assert_eq!(x / (y * z), (x / y) / z);
        assert_eq!((x * y) / z, x * (y / z));
    }

    #[test]
    pub fn test_inverse()
    where
        StandardUniform: Distribution<Goldilocks>,
    {
        assert_eq!(None, Goldilocks::ZERO.try_inverse());

        assert_eq!(Some(Goldilocks::ONE), Goldilocks::ONE.try_inverse());

        let mut rng = rand::rng();
        for _ in 0..1000 {
            let x = rng.random::<Goldilocks>();
            if !x.is_zero() && !x.is_one() {
                let z = x.inverse();
                assert_ne!(x, z);
                assert_eq!(x * z, Goldilocks::ONE);
            }
        }
    }

    #[test]
    pub fn test_from_int()
    where
        StandardUniform: Distribution<Goldilocks>,
    {
        assert_eq!(Goldilocks::from_int(0), Goldilocks::ZERO);
        assert_eq!(Goldilocks::from_int(1), Goldilocks::ONE);
        let field_two = Goldilocks::ONE + Goldilocks::ONE;
        assert_eq!(Goldilocks::from_int(2), field_two);
        assert_eq!(Goldilocks::from_int(2), Goldilocks::TWO);
        let field_three = field_two + Goldilocks::ONE;
        assert_eq!(Goldilocks::from_int(3), field_three);
        assert_eq!(Goldilocks::from_int(3), Goldilocks::TWO + Goldilocks::ONE);
        let field_six = field_two * field_three;
        assert_eq!(Goldilocks::from_int(6), field_six);
        let field_36 = field_six * field_six;
        assert_eq!(Goldilocks::from_int(36), field_36);
        let field_108 = field_36 * field_three;
        assert_eq!(Goldilocks::from_int(108), field_108);

        let field_neg_one = -Goldilocks::ONE;
        assert_eq!(Goldilocks::from_int(-1), field_neg_one);
        assert_eq!(Goldilocks::from_int(-1), Goldilocks::NEG_ONE);
        let field_neg_two = field_neg_one + field_neg_one;
        assert_eq!(Goldilocks::from_int(-2), field_neg_two);
        let field_neg_four = field_neg_two + field_neg_two;
        assert_eq!(Goldilocks::from_int(-4), field_neg_four);
        let field_neg_six = field_neg_two + field_neg_four;
        assert_eq!(Goldilocks::from_int(-6), field_neg_six);
        let field_neg_24 = -field_neg_six * field_neg_four;
        assert_eq!(Goldilocks::from_int(-24), field_neg_24);

        let mut rng = rand::rng();
        for _ in 0..1000 {
            let x = rng.random::<Goldilocks>();
            assert_eq!(x, Goldilocks::from_int(x.as_canonical_u64()));
            assert_eq!(x, Goldilocks::from_canonical_checked(x.as_canonical_u64()).unwrap());
            assert_eq!(x, unsafe { Goldilocks::from_canonical_unchecked(x.as_canonical_u64()) });
        }
    }

    #[test]
    fn test_as_canonical_u64() {
        let mut rng = rand::rng();
        let x: u64 = F::from_u64(rng.random_range(0..=(1 << 63) - 1));
        let x_mod_order = x % Goldilocks::ORDER_U64;

        assert_eq!(Goldilocks::ZERO.as_canonical_u64(), 0);
        assert_eq!(Goldilocks::ONE.as_canonical_u64(), 1);
        assert_eq!(Goldilocks::TWO.as_canonical_u64(), 2 % Goldilocks::ORDER_U64);
        assert_eq!(Goldilocks::NEG_ONE.as_canonical_u64(), Goldilocks::ORDER_U64 - 1);

        assert_eq!(Goldilocks::from_int(Goldilocks::ORDER_U64).as_canonical_u64(), 0);
        assert_eq!(Goldilocks::from_int(x).as_canonical_u64(), x_mod_order);
        assert_eq!(unsafe { Goldilocks::from_canonical_unchecked(x_mod_order).as_canonical_u64() }, x_mod_order);
    }

    #[test]
    fn test_as_unique_u64() {
        assert_ne!(Goldilocks::ZERO.to_unique_u64(), Goldilocks::ONE.to_unique_u64());
        assert_ne!(Goldilocks::ZERO.to_unique_u64(), Goldilocks::NEG_ONE.to_unique_u64());
        assert_eq!(Goldilocks::from_int(Goldilocks::ORDER_U64).to_unique_u64(), Goldilocks::ZERO.to_unique_u64());
    }

    #[test]
    fn generate_from_large_u_int_tests() {
        // Check some wraparound cases:
        // Note that for unsigned integers, from_canonical_checked returns
        // None when the input is bigger or equal to the field order.
        // Similarly, from_canonical_unchecked may also return invalid results in these cases.
        let field_order = Goldilocks::ORDER_U64;

        // On the other hand, everything should work fine for field_order - 1 and (field_order + 1)/2.
        assert_eq!(Goldilocks::from_int(field_order - 1), -Goldilocks::ONE);

        let half = (field_order + 1) >> 1;
        let field_half = (Goldilocks::ONE + Goldilocks::ONE).inverse();
        assert_eq!(Goldilocks::from_int(half), field_half);

        // We check that from_canonical_checked returns None for large enough values
        // but from_int is still correct.
        assert_eq!(Goldilocks::from_int(field_order), Goldilocks::ZERO);
        assert_eq!(Goldilocks::from_canonical_checked(field_order), None);
        assert_eq!(Goldilocks::from_int(field_order + 1), Goldilocks::ONE);
        assert_eq!(Goldilocks::from_canonical_checked(field_order + 1), None);
        assert_eq!(Goldilocks::from_canonical_checked(u64::MAX), None);
    }

    #[test]
    fn generate_from_large_i_int_tests() {
        // Check some wraparound cases:
        // Note that for unsigned integers, from_canonical_checked returns
        // None when |input| is bigger than (field order - 1)/2 and from_canonical_unchecked
        // may also return invalid results in these cases.
        let neg_half = (Goldilocks::ORDER_U64 >> 1) as i64;
        let half_as_neg_rep = -neg_half;

        let field_half = (Goldilocks::ONE + Goldilocks::ONE).inverse();
        let field_neg_half = field_half - Goldilocks::ONE;

        assert_eq!(Goldilocks::from_int(half_as_neg_rep), field_half);
        assert_eq!(Goldilocks::from_int(neg_half), field_neg_half);

        // We check that from_canonical_checked returns None for large enough values but
        // from_int is still correct.
        let half = neg_half + 1;
        assert_eq!(Goldilocks::from_int(half), field_half);
        assert_eq!(Goldilocks::from_canonical_checked(half), None);
        assert_eq!(Goldilocks::from_int(-half), field_neg_half);
        assert_eq!(Goldilocks::from_canonical_checked(-half), None);
        assert_eq!(Goldilocks::from_canonical_checked(i64::MAX), None);
        assert_eq!(Goldilocks::from_canonical_checked(i64::MIN), None);
    }
}
