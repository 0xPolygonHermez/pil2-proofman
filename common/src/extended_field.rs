use p3_field::{AbstractField, Field};
use core::array;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ExtensionField<F> {
    value: [F; 3],
}

impl<F: AbstractField + Copy> ExtensionField<F> {
    pub fn zero() -> Self {
        Self {
            value: field_to_array::<F>(F::zero()),
        }
    }
    pub fn one() -> Self {
        Self {
            value: field_to_array::<F>(F::one()),
        }
    }
    pub fn two() -> Self {
        Self {
            value: field_to_array::<F>(F::two()),
        }
    }
    pub fn neg_one() -> Self {
        Self {
            value: field_to_array::<F>(F::neg_one()),
        }
    }

    #[inline(always)]
    pub fn square(&self) -> Self {
        Self {
            value: cubic_square(&self.value)
                .to_vec()
                .try_into()
                .unwrap(),
        }
    }

    pub fn from_array(arr: &[F]) -> Self {
        // Ensure the array has the correct size
        assert!(arr.len() == 3, "Array must have length 3");

        let mut value: [F; 3] = Default::default();
        value.copy_from_slice(arr);

        Self { value}
    }
}

impl<F:AbstractField> Add for ExtensionField<F> {
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

impl<F:AbstractField> Add<F> for ExtensionField<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        let mut res = self.value;
        res[0] += rhs;
        Self { value: res }
    }
}

impl<F: AbstractField> AddAssign for ExtensionField<F>
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F: AbstractField> AddAssign<F> for ExtensionField<F>
{
    fn add_assign(&mut self, rhs: F) {
        *self = self.clone() + rhs;
    }
}

impl<F: AbstractField> Sum for ExtensionField<F>
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = Self {
            value: field_to_array::<F>(F::zero()),
        };
        iter.fold(zero, |acc, x| acc + x)
    }
}


impl<F: AbstractField> Sub for ExtensionField<F>
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

impl<F: AbstractField> Sub<F> for ExtensionField<F>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl<F: AbstractField> SubAssign for ExtensionField<F>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F: AbstractField> SubAssign<F> for ExtensionField<F>
{
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = self.clone() - rhs;
    }
}


impl<F: AbstractField> Mul for ExtensionField<F>
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        Self {
            value: cubic_mul(&a, &b)
                .to_vec()
                .try_into()
                .unwrap(),
        }
    }
}

impl<F: AbstractField> Mul<F> for ExtensionField<F>
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self {
            value: self.value.map(|x| x * rhs.clone()),
        }
    }
}

impl<F: AbstractField> Product for ExtensionField<F>
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let one = Self {
            value: field_to_array::<F>(F::one()),
        };
        iter.fold(one, |acc, x| acc * x)
    }
}

impl<F: AbstractField> MulAssign for ExtensionField<F>
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F: AbstractField> MulAssign<F> for ExtensionField<F>
{
    fn mul_assign(&mut self, rhs: F) {
        *self = self.clone() * rhs;
    }
}

impl<F: AbstractField> Neg for ExtensionField<F>
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.value.map(F::neg),
        }
    }
}

impl<F: AbstractField + Field> Div for ExtensionField<F>
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        let a = self.value;
        let b_inv = cubic_inv(&rhs.value);
        Self {
            value: cubic_mul(&a, &b_inv)
            .to_vec()
            .try_into()
            .unwrap(),
        }
    }
}

impl<F: AbstractField + Field> DivAssign for ExtensionField<F>
{
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Extend a field `F` element `x` to an array of length 3
/// by filling zeros.
pub fn field_to_array<F: AbstractField>(x: F) -> [F; 3] {
    let mut arr = array::from_fn(|_| F::zero());
    arr[0] = x;
    arr
}

#[inline]
fn cubic_square<F: AbstractField>(a: &[F]) -> [F; 3] {
    let c0 = a[0].square() + (a[2].clone()*a[1].clone()).double();
    let c1 = a[2].square() + (a[0].clone() * a[1].clone()).double() + (a[1].clone() * a[2].clone()).double();
    let c2 = a[1].square() + (a[0].clone() * a[2].clone()).double() + a[2].square();

    [c0, c1, c2]
}

#[inline]
fn cubic_mul<F: AbstractField>(a: &[F], b: &[F]) -> [F; 3] {
    let c0 = a[0].clone()*b[0].clone() + a[2].clone()*b[1].clone() + a[1].clone()*b[2].clone();
    let c1 = a[1].clone()*b[0].clone() + a[0].clone()*b[1].clone() + a[2].clone()*b[1].clone() + a[1].clone()*b[2].clone() + a[2].clone()*b[2].clone();
    let c2 = a[2].clone()*b[0].clone() + a[1].clone()*b[1].clone() + a[0].clone()*b[2].clone() + a[2].clone()*b[2].clone();

    [c0, c1, c2]
}


fn cubic_inv<F: AbstractField + Field>(a: &[F]) -> [F; 3] {

    let aa = a[0].square();
    let ac = a[0].clone() * a[2].clone();
    let ba = a[1].clone() * a[0].clone();
    let bb = a[1].square();
    let bc = a[1].clone() * a[2].clone();
    let cc = a[2].square();
        
    let aaa = aa.clone() * a[0].clone();
    let aac = aa.clone() * a[2].clone();
    let abc = ba.clone() * a[2].clone();
    let abb = ba.clone() * a[1].clone();
    let acc = ac.clone() * a[2].clone();
    let bbb = bb.clone() * a[1].clone();
    let bcc = bc.clone() * a[2].clone();
    let ccc = cc.clone() * a[2].clone();
    
    let t = abc.clone() + abc.clone() + abc.clone() + abb.clone() - aaa.clone() - aac.clone() - aac.clone() - acc.clone() - bbb.clone() + bcc.clone() - ccc.clone();
    
    let i0 = (bc.clone() + bb.clone() - aa.clone() - ac.clone() - ac.clone() - cc.clone()) * t.inverse();
    let i1 = (ba.clone() - cc.clone()) * t.inverse();
    let i2 = (ac.clone() + cc.clone() - bb.clone()) * t.inverse();

    [i0, i1, i2]
}