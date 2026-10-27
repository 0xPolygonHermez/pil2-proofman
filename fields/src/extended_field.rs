use crate::{DotAcc, Field, Goldilocks};
use core::array;
use core::fmt::{Display, Formatter, Result};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CubicExtensionField<F: Display> {
    pub value: [F; 3],
}

impl<F: Display> Index<usize> for CubicExtensionField<F> {
    type Output = F;

    #[inline]
    fn index(&self, idx: usize) -> &F {
        &self.value[idx]
    }
}

impl<F: Display> IndexMut<usize> for CubicExtensionField<F> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut F {
        &mut self.value[idx]
    }
}

impl<F: Display> Display for CubicExtensionField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        // Format the elements in the array as [a, b, c]
        write!(f, "[{}, {}, {}]", self.value[0], self.value[1], self.value[2])
    }
}

impl<F: Field> CubicExtensionField<F> {
    pub fn zero() -> Self {
        Self { value: field_to_array(F::ZERO) }
    }
    pub fn one() -> Self {
        Self { value: field_to_array(F::ONE) }
    }
    pub fn two() -> Self {
        Self { value: field_to_array(F::TWO) }
    }
    pub fn neg_one() -> Self {
        Self { value: field_to_array(F::NEG_ONE) }
    }

    pub fn is_zero(&self) -> bool {
        self.value.iter().all(|&x| x.is_zero())
    }

    #[inline(always)]
    pub fn square(&self) -> Self {
        Self { value: cubic_square(&self.value) }
    }

    #[inline]
    pub fn pow(&self, mut exp: u64) -> Self {
        // result = 1
        let mut result = Self::one();
        // temp = self
        let mut base = *self;
        // while there are bits left in exp
        while exp > 0 {
            // if the low bit is set, multiply it into result
            if exp & 1 == 1 {
                result *= base;
            }
            // square the base each round
            base = base.square();
            // shift off the bit we just processed
            exp >>= 1;
        }
        result
    }

    pub fn inverse(&self) -> Self {
        Self { value: cubic_inv(&self.value) }
    }

    pub fn from_array(arr: &[F]) -> Self {
        // Ensure the array has the correct size
        debug_assert!(arr.len() == 3, "Array must have length 3");

        let mut value: [F; 3] = Default::default();
        value.copy_from_slice(arr);

        Self { value }
    }

    #[inline]
    pub fn sub_from_scalar(self, scalar: F) -> Self {
        // destructure for clarity
        let [v0, v1, v2] = self.value;
        let zero = F::ZERO;
        CubicExtensionField { value: [scalar - v0, zero - v1, zero - v2] }
    }
}

impl<F: Field> Add for CubicExtensionField<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r += rhs_val;
        }
        Self { value: res }
    }
}

impl<F: Field> Add<F> for CubicExtensionField<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        let mut res = self.value;
        res[0] += rhs;
        Self { value: res }
    }
}

impl<F: Field> AddAssign for CubicExtensionField<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> AddAssign<F> for CubicExtensionField<F> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: Field> Sum for CubicExtensionField<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = Self { value: field_to_array(F::ZERO) };
        iter.fold(zero, |acc, x| acc + x)
    }
}

impl<F: Field> Sub for CubicExtensionField<F> {
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

impl<F: Field> Sub<F> for CubicExtensionField<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl<F: Field> SubAssign for CubicExtensionField<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> SubAssign<F> for CubicExtensionField<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: Field> Mul for CubicExtensionField<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        Self { value: cubic_mul(&a, &b) }
    }
}

impl<F: Field> Mul<F> for CubicExtensionField<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self { value: self.value.map(|x| x * rhs) }
    }
}

impl<F: Field> Product for CubicExtensionField<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let one = Self { value: field_to_array(F::ONE) };
        iter.fold(one, |acc, x| acc * x)
    }
}

impl<F: Field> MulAssign for CubicExtensionField<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> MulAssign<F> for CubicExtensionField<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: Field> Neg for CubicExtensionField<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self { value: self.value.map(F::neg) }
    }
}

impl<F: Field> Div for CubicExtensionField<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        let a = self.value;
        let b_inv = cubic_inv(&rhs.value);
        Self { value: cubic_mul(&a, &b_inv) }
    }
}

impl<F: Field> DivAssign for CubicExtensionField<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Extend a field `F` element `x` to an array of length 3
/// by filling zeros.
pub fn field_to_array<F: Field>(x: F) -> [F; 3] {
    let mut arr = array::from_fn(|_| F::ZERO);
    arr[0] = x;
    arr
}

#[inline]
fn cubic_square<F: Field>(a: &[F]) -> [F; 3] {
    let c0 = a[0].square() + (a[2] * a[1]).double();
    let c1 = a[2].square() + (a[0] * a[1]).double() + (a[1] * a[2]).double();
    let c2 = a[1].square() + (a[0] * a[2]).double() + a[2].square();

    [c0, c1, c2]
}

/// `a * b` under `x^3 = x + 1`, as the `(a index, b index)` products of each output coordinate.
/// The one place the relation lives: `cubic_mul` and [`cubic_mul_acc`] both read it.
const CUBIC_C0: [(usize, usize); 3] = [(0, 0), (2, 1), (1, 2)];
const CUBIC_C1: [(usize, usize); 5] = [(1, 0), (0, 1), (2, 1), (1, 2), (2, 2)];
const CUBIC_C2: [(usize, usize); 4] = [(2, 0), (1, 1), (0, 2), (2, 2)];

fn gather<F: Copy, const N: usize>(terms: &[(usize, usize); N], a: &[F], b: &[F]) -> ([F; N], [F; N]) {
    (terms.map(|(i, _)| a[i]), terms.map(|(_, j)| b[j]))
}

/// `acc += a * b`, the reduction deferred to `DotAcc::finish` with whatever else is summed in.
pub fn cubic_mul_acc(acc: &mut [DotAcc; 3], a: &[Goldilocks; 3], b: &[Goldilocks; 3]) {
    for &(i, j) in &CUBIC_C0 {
        acc[0].add(a[i], b[j]);
    }
    for &(i, j) in &CUBIC_C1 {
        acc[1].add(a[i], b[j]);
    }
    for &(i, j) in &CUBIC_C2 {
        acc[2].add(a[i], b[j]);
    }
}

#[inline]
fn cubic_mul<F: Field>(a: &[F], b: &[F]) -> [F; 3] {
    let (a0, b0) = gather(&CUBIC_C0, a, b);
    let (a1, b1) = gather(&CUBIC_C1, a, b);
    let (a2, b2) = gather(&CUBIC_C2, a, b);
    let c0 = F::dot(a0, b0);
    let c1 = F::dot(a1, b1);
    let c2 = F::dot(a2, b2);

    [c0, c1, c2]
}

fn cubic_inv<F: Field>(a: &[F]) -> [F; 3] {
    let aa = a[0].square();
    let ac = a[0] * a[2];
    let ba = a[1] * a[0];
    let bb = a[1].square();
    let bc = a[1] * a[2];
    let cc = a[2].square();

    let aaa = aa * a[0];
    let aac = aa * a[2];
    let abc = ba * a[2];
    let abb = ba * a[1];
    let acc = ac * a[2];
    let bbb = bb * a[1];
    let bcc = bc * a[2];
    let ccc = cc * a[2];

    let t = abc + abc + abc + abb - aaa - aac - aac - acc - bbb + bcc - ccc;

    let t_inv = t.inverse();
    let i0 = (bc + bb - aa - ac - ac - cc) * t_inv;
    let i1 = (ba - cc) * t_inv;
    let i2 = (ac + cc - bb) * t_inv;

    [i0, i1, i2]
}
