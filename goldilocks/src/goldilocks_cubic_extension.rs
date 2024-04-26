use std::{
    fmt::Display,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use p3_field::*;
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use num_bigint::BigUint;

use rand::distributions::{Distribution, Standard};

use crate::field_to_array;

pub trait Extendable<const D: usize>: Field {
    fn w() -> Self;

    // // DTH_ROOT = W^((n - 1)/D).
    // // n is the order of base field.
    // // Only works when exists k such that n = kD + 1.
    // fn dth_root() -> Self;

    fn ext_generator() -> [Self; D];
}

/// Cubic extension field
/// Represents a cubic extension of a Goldilocks prime field 0xFFFFFFFF00000001
///
/// The extension element is defined by this irreducible polynomial x^3 - x -1
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct CubicExtension<AF> {
    pub(crate) value: [AF; 3],
}

impl<AF: AbstractField> CubicExtension<AF> {
    pub fn value(&self) -> &[AF; 3] {
        &self.value
    }

    pub fn set_zero(destination: &mut Self) {
        destination.value = [AF::zero(), AF::zero(), AF::zero()];
    }

    pub fn set_one(destination: &mut Self) {
        destination.value = [AF::one(), AF::zero(), AF::zero()];
    }
}

impl Extendable<3> for Goldilocks {
    fn w() -> Self {
        Self::from_canonical_u64(7)
    }

    fn ext_generator() -> [Self; 3] {
        [Self::from_canonical_u64(1), Self::from_canonical_u64(0), Self::from_canonical_u64(0)]
    }
}

impl<AF: AbstractField> Default for CubicExtension<AF> {
    fn default() -> Self {
        Self { value: [AF::zero(), AF::zero(), AF::zero()] }
    }
}

impl<AF: AbstractField> From<AF> for CubicExtension<AF> {
    fn from(value: AF) -> Self {
        Self { value: [value, AF::zero(), AF::zero()] }
    }
}

// impl<F: Extendable<3>> Packable for CubicExtension<F> {}

impl<F: Extendable<3>> ExtensionField<F> for CubicExtension<F> {
    // type ExtensionPacking = CubicExtension<F::Packing>;
}

impl<AF> AbstractField for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    type F = CubicExtension<AF::F>;

    fn zero() -> Self {
        Self { value: [AF::zero(), AF::zero(), AF::zero()] }
    }

    fn one() -> Self {
        Self { value: [AF::one(), AF::zero(), AF::zero()] }
    }

    fn two() -> Self {
        Self { value: [AF::two(), AF::zero(), AF::zero()] }
    }

    fn neg_one() -> Self {
        Self { value: [AF::neg_one(), AF::zero(), AF::zero()] }
    }

    fn from_f(f: Self::F) -> Self {
        Self { value: f.value.map(AF::from_f) }
    }

    fn from_bool(b: bool) -> Self {
        AF::from_bool(b).into()
    }

    fn from_canonical_u8(n: u8) -> Self {
        AF::from_canonical_u8(n).into()
    }

    fn from_canonical_u16(n: u16) -> Self {
        AF::from_canonical_u16(n).into()
    }

    fn from_canonical_u32(n: u32) -> Self {
        AF::from_canonical_u32(n).into()
    }

    fn from_canonical_u64(n: u64) -> Self {
        AF::from_canonical_u64(n).into()
    }

    fn from_canonical_usize(n: usize) -> Self {
        AF::from_canonical_usize(n).into()
    }

    fn from_wrapped_u32(n: u32) -> Self {
        AF::from_wrapped_u32(n).into()
    }

    fn from_wrapped_u64(n: u64) -> Self {
        AF::from_wrapped_u64(n).into()
    }

    fn generator() -> Self {
        Self { value: AF::F::ext_generator().map(AF::from_f) }
    }

    #[inline(always)]
    fn square(&self) -> Self {
        Self { value: cubic_square(&self.value, AF::F::w()).to_vec().try_into().unwrap() }
    }
}

impl<F: Extendable<3>> Field for CubicExtension<F> {
    type Packing = Self;

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        Some(Self::from_base_slice(&cubic_inv(&self.value, F::w())))
    }

    // fn halve(&self) -> Self {
    //     Self { value: self.value.map(|x| x.halve()) }
    // }

    // fn order() -> BigUint {
    //     F::order().pow(3 as u32)
    // }
}

impl<F: Extendable<3>> Display for CubicExtension<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "CubicExtension(0)");
        }

        //TODO!!!!!!!
        write!(f, "CubicExtension({} + {}w + {}w^2)", self.value[0], self.value[1], self.value[2])
    }
}

impl<AF> Neg for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self { value: self.value.map(AF::neg) }
    }
}

impl<AF> Add for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r += rhs_val;
        }
        Self { value: res }
    }
}

impl<AF> Add<AF> for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: AF) -> Self::Output {
        let mut res = self.value;
        res[0] += rhs;
        Self { value: res }
    }
}

impl<AF> AddAssign for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<AF> AddAssign<AF> for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    fn add_assign(&mut self, rhs: AF) {
        *self = self.clone() + rhs;
    }
}

impl<AF> Sum for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = Self { value: field_to_array::<AF, 3>(AF::zero()) };
        iter.fold(zero, |acc, x| acc + x)
    }
}

impl<AF> Sub for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r -= rhs_val;
        }
        Self { value: res }
    }
}

impl<AF> Sub<AF> for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: AF) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl<AF> SubAssign for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<AF> SubAssign<AF> for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: AF) {
        *self = self.clone() - rhs;
    }
}

impl<AF> Mul for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let w = AF::F::w();

        Self { value: cubic_mul(&a, &b, w).to_vec().try_into().unwrap() }
    }
}

impl<AF> Mul<AF> for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: AF) -> Self {
        Self { value: self.value.map(|x| x * rhs.clone()) }
    }
}

impl<AF> Product for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let one = Self { value: field_to_array::<AF, 3>(AF::one()) };
        iter.fold(one, |acc, x| acc * x)
    }
}

impl<AF> MulAssign for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<AF> MulAssign<AF> for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    fn mul_assign(&mut self, rhs: AF) {
        *self = self.clone() * rhs;
    }
}

impl<F> Div for CubicExtension<F>
where
    F: Extendable<3>,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F> DivAssign for CubicExtension<F>
where
    F: Extendable<3>,
{
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<AF> AbstractExtensionField<AF> for CubicExtension<AF>
where
    AF: AbstractField,
    AF::F: Extendable<3>,
{
    const D: usize = 3;

    fn from_base(b: AF) -> Self {
        Self { value: field_to_array(b) }
    }

    fn from_base_slice(bs: &[AF]) -> Self {
        Self { value: bs.to_vec().try_into().expect("slice has wrong length") }
    }

    #[inline]
    fn from_base_fn<F: FnMut(usize) -> AF>(f: F) -> Self {
        Self { value: core::array::from_fn(f) }
    }

    fn as_base_slice(&self) -> &[AF] {
        &self.value
    }
}

impl<AF: Extendable<3>> Distribution<CubicExtension<AF>> for Standard
where
    Standard: Distribution<AF>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CubicExtension<AF> {
        let mut res = [AF::zero(); 3];
        for r in res.iter_mut() {
            *r = Standard.sample(rng);
        }
        CubicExtension::<AF>::from_base_slice(&res)
    }
}

// From Plonky3 binomial_extension.rs

// Section 11.3.6b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn cubic_inv<F: Field>(a: &[F], w: F) -> [F; 3] {
    let a0_square = a[0].square();
    let a1_square = a[1].square();
    let a2_w = w * a[2];
    let a0_a1 = a[0] * a[1];

    // scalar = (a0^3+wa1^3+w^2a2^3-3wa0a1a2)^-1
    let scalar = (a0_square * a[0] + w * a[1] * a1_square + a2_w.square() * a[2]
        - (F::one() + F::two()) * a2_w * a0_a1)
        .inverse();

    //scalar*[a0^2-wa1a2, wa2^2-a0a1, a1^2-a0a2]
    [scalar * (a0_square - a[1] * a2_w), scalar * (a2_w * a[2] - a0_a1), scalar * (a1_square - a[0] * a[2])]
}

/// karatsuba multiplication for cubic extension field
#[inline]
fn cubic_mul<AF: AbstractField>(a: &[AF], b: &[AF], w: AF::F) -> [AF; 3] {
    let a0_b0 = a[0].clone() * b[0].clone();
    let a1_b1 = a[1].clone() * b[1].clone();
    let a2_b2 = a[2].clone() * b[2].clone();

    let c0 = a0_b0.clone()
        + ((a[1].clone() + a[2].clone()) * (b[1].clone() + b[2].clone()) - a1_b1.clone() - a2_b2.clone())
            * AF::from_f(w);
    let c1 = (a[0].clone() + a[1].clone()) * (b[0].clone() + b[1].clone()) - a0_b0.clone() - a1_b1.clone()
        + a2_b2.clone() * AF::from_f(w);
    let c2 = (a[0].clone() + a[2].clone()) * (b[0].clone() + b[2].clone()) - a0_b0 - a2_b2 + a1_b1;

    [c0, c1, c2]
}

/// Section 11.3.6a in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn cubic_square<AF: AbstractField>(a: &[AF], w: AF::F) -> [AF; 3] {
    let w_a2 = a[2].clone() * AF::from_f(w);

    let c0 = a[0].square() + (a[1].clone() * w_a2.clone()).double();
    let c1 = w_a2 * a[2].clone() + (a[0].clone() * a[1].clone()).double();
    let c2 = a[1].square() + (a[0].clone() * a[2].clone()).double();

    [c0, c1, c2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_goldilocks::Goldilocks;
    use p3_field::AbstractField;
    use rand::Rng;
    use rand::distributions::{Distribution, Standard};

    type F = Goldilocks;
    type EF = CubicExtension<F>;

    #[test]
    fn test_cubic_extension_creation() {
        let mut cubic = EF::default();
        assert_eq!(cubic.value, [F::from_canonical_u64(0), F::from_canonical_u64(0), F::from_canonical_u64(0)]);
        assert_eq!(cubic.value(), &[F::from_canonical_u64(0), F::from_canonical_u64(0), F::from_canonical_u64(0)]);

        CubicExtension::<F>::set_one(&mut cubic);
        assert_eq!(cubic.value, [F::from_canonical_u64(1), F::from_canonical_u64(0), F::from_canonical_u64(0)]);

        CubicExtension::<F>::set_zero(&mut cubic);
        assert_eq!(cubic.value, [F::from_canonical_u64(0), F::from_canonical_u64(0), F::from_canonical_u64(0)]);
    }

    #[test]
    fn test_cubic_extension_from() {
        // Create a random Golfilocks value from random u64
        let random: u64 = Rng::gen(&mut rand::thread_rng());
        let gl_number = F::from_canonical_u64(random);
        let cubic = EF::from(gl_number);
        assert_eq!(cubic.value, [F::from_canonical_u64(random), F::from_canonical_u64(0), F::from_canonical_u64(0)]);
    }

    #[test]
    fn test_cubic_extension_abstract_field() {
        let zero = EF::zero();

        assert_eq!(zero, EF::from_canonical_u64(0));
    }

    #[allow(clippy::eq_op)]
    #[test]
    fn test_add_neg_sub_mul()
    where
        Standard: Distribution<EF>,
    {
        let mut rng = rand::thread_rng();
        let x = rng.gen::<EF>();
        let y = rng.gen::<EF>();
        let z = rng.gen::<EF>();
        assert_eq!(x + (-x), EF::zero());
        assert_eq!(-x, EF::zero() - x);
        assert_eq!(x + x, x * EF::two());
        // assert_eq!(x, x.halve() * F::two());
        assert_eq!(x * (-x), -x.square());
        assert_eq!(x + y, y + x);
        assert_eq!(x * y, y * x);
        assert_eq!(x * (y * z), (x * y) * z);
        assert_eq!(x - (y + z), (x - y) - z);
        assert_eq!((x + y) - z, x + (y - z));
        assert_eq!(x * (y + z), x * y + x * z);
        assert_eq!(x + y + z + x + y + z, [x, x, y, y, z, z].iter().cloned().sum());
    }

    #[test]
    pub fn test_inv_div()
    where
        Standard: Distribution<EF>,
    {
        let mut rng = rand::thread_rng();
        let x = rng.gen::<EF>();
        let y = rng.gen::<EF>();
        let z = rng.gen::<EF>();
        assert_eq!(x * x.inverse(), EF::one());
        assert_eq!(x.inverse() * x, EF::one());
        assert_eq!(x.square().inverse(), x.inverse().square());
        assert_eq!((x / y) * y, x);
        assert_eq!(x / (y * z), (x / y) / z);
        assert_eq!((x * y) / z, x * (y / z));

        println!("x: {}", x);
        println!("x.inverse(): {}", x.inverse());
        println!("x * x.inverse(): {}", x * x.inverse());
    }

    #[test]
    pub fn test_inverse()
    where
        Standard: Distribution<EF>,
    {
        assert_eq!(None, EF::zero().try_inverse());

        assert_eq!(Some(EF::one()), EF::one().try_inverse());

        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x = rng.gen::<F>();
            if !x.is_zero() && !x.is_one() {
                let z = x.inverse();
                assert_ne!(x, z);
                assert_eq!(x * z, F::one());
            }
        }
    }
}