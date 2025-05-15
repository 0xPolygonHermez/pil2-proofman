use std::{
    array,
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::{PrimeField64, Field, ExtensionField, Goldilocks};

/// Field Fp⁵ = F\[X\]/(X⁵-3) with generator X + 2
pub type GoldilocksQuinticExtension = QuinticExtension<Goldilocks>;

#[derive(Debug, Copy, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct QuinticExtension<F> {
    pub(crate) value: [F; 5],
}

impl<F: Field> Default for QuinticExtension<F> {
    fn default() -> Self {
        Self { value: array::from_fn(|_| F::ZERO) }
    }
}

impl<F: Field> Display for QuinticExtension<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{}*X^4 + {}*X^3 + {}*X^2 + {}*X + {}",
            self.value[4], self.value[3], self.value[2], self.value[1], self.value[0]
        )
    }
}

impl<F: PrimeField64> QuinticExtension<F> {
    pub(crate) const fn new(value: [F; 5]) -> Self {
        Self { value }
    }

    fn inv(&self) -> Self {
        // Inverse is computed as
        //    1       xʳ⁻¹
        //  ----- = -------
        //    x       xʳ
        // where r = p⁴ + p³ + p² + p + 1

        // Compute xʳ⁻¹ = self^(p⁴ + p³ + p² + p)
        let exp = self.exp_fifth_cyclotomic_minus_one();

        // Compute xʳ = self^(p⁴ + p³ + p² + p + 1)
        let a = self.as_basis_coefficients_slice();
        let b = exp.as_basis_coefficients_slice();
        let exp_cyclo =
            a[0] * b[0] + F::from_u64(Self::NON_RESIDUE) * (a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1]);

        // Compute xʳ⁻¹ / xʳ
        let inv_cyclo = exp_cyclo.inverse();
        exp * inv_cyclo
    }
}

impl<F: PrimeField64> ExtensionField<F> for QuinticExtension<F> {
    const NON_RESIDUE: u64 = 3;

    fn as_basis_coefficients_slice(&self) -> &[F] {
        &self.value
    }

    fn from_basis_singleton(value: F) -> Self {
        Self { value: [value, F::ZERO, F::ZERO, F::ZERO, F::ZERO] }
    }

    fn from_basis_coefficients_slice(value: &[F]) -> Self {
        assert_eq!(value.len(), 5);

        Self { value: [value[0], value[1], value[2], value[3], value[4]] }
    }

    fn from_basis_coefficients_fn(f: impl Fn(usize) -> F) -> Self {
        let mut value = [F::ZERO; 5];
        for (i, v) in value.iter_mut().enumerate() {
            *v = f(i);
        }
        Self { value }
    }
}

impl<F: PrimeField64> Field for QuinticExtension<F> {
    const ZERO: Self = Self::new([F::ZERO; 5]);
    const ONE: Self = Self::new([F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
    const TWO: Self = Self::new([F::TWO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
    const NEG_ONE: Self = Self::new([F::NEG_ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
    const GENERATOR: Self = Self::new([F::TWO, F::ONE, F::ZERO, F::ZERO, F::ZERO]);

    #[inline(always)]
    fn square(&self) -> Self {
        let a = self.value;
        let non_residue = F::from_u64(Self::NON_RESIDUE);
        let non_residue_double = non_residue.double();

        let c0 = a[0].square() + non_residue_double * (a[1] * a[4] + a[2] * a[3]);
        let c1 = non_residue * a[3].square() + non_residue_double * (a[2] * a[4]) + (a[0] * a[1]).double();
        let c2 = a[1].square() + non_residue_double * (a[3] * a[4]) + (a[0] * a[2]).double();
        let c3 = non_residue * a[4].square() + (a[1] * a[2] + a[0] * a[3]).double();
        let c4 = a[2].square() + (a[0] * a[4] + a[1] * a[3]).double();

        Self::new([c0, c1, c2, c3, c4])
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Some(self.inv())
    }
}

// Traits implementation for QuinticExtension
impl<F: Field> Add for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { value: array::from_fn(|i| self.value[i] + rhs.value[i]) }
    }
}

impl<F: Field> Add<F> for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        let mut value = self.value;
        value[0] += rhs;
        Self { value }
    }
}

impl<F: Field> AddAssign for QuinticExtension<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> AddAssign<F> for QuinticExtension<F> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: Field> Sub for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { value: array::from_fn(|i| self.value[i] - rhs.value[i]) }
    }
}

impl<F: Field> Sub<F> for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        let mut value = self.value;
        value[0] -= rhs;
        Self { value }
    }
}

impl<F: Field> SubAssign for QuinticExtension<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> SubAssign<F> for QuinticExtension<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: Field> Neg for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self { value: array::from_fn(|i| -self.value[i]) }
    }
}

impl<F: PrimeField64> Mul for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let non_residue = F::from_u64(Self::NON_RESIDUE);

        let c0 = a[0] * b[0] + non_residue * (a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1]);
        let c1 = a[0] * b[1] + a[1] * b[0] + non_residue * (a[2] * b[4] + a[3] * b[3] + a[4] * b[2]);
        let c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + non_residue * (a[3] * b[4] + a[4] * b[3]);
        let c3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0] + non_residue * (a[4] * b[4]);
        let c4 = a[0] * b[4] + a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + a[4] * b[0];

        Self::new([c0, c1, c2, c3, c4])
    }
}

impl<F: Field> Mul<F> for QuinticExtension<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self { value: array::from_fn(|i| self.value[i] * rhs) }
    }
}

impl<F: PrimeField64> MulAssign for QuinticExtension<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> MulAssign<F> for QuinticExtension<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: PrimeField64> Div for QuinticExtension<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl<F: Field> Div<F> for QuinticExtension<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: F) -> Self {
        self * rhs.inverse()
    }
}

impl<F: PrimeField64> DivAssign for QuinticExtension<F> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: Field> DivAssign<F> for QuinticExtension<F> {
    #[inline]
    fn div_assign(&mut self, rhs: F) {
        *self = *self / rhs;
    }
}

/// Methods for computing the square root in the QuinticExtension field
/// as described in [Elliptic Curves over Goldilocks](https://hackmd.io/CxJrIhv-SP65W3GWS_J5bw?view#Extension-Field-Selection),
/// which is inspired by [Curve ecGFp5](https://github.com/pornin/ecgfp5/tree/main)
pub trait SquaringFp5<F: PrimeField64> {
    /// Constants for the first Frobenius operator
    const GAMMAS1: [u64; 5] = [1, 1041288259238279555, 15820824984080659046, 211587555138949697, 1373043270956696022];

    /// Constants for the second Frobenius operator
    const GAMMAS2: [u64; 5] = [1, 15820824984080659046, 1373043270956696022, 1041288259238279555, 211587555138949697];

    /// Compute the first Frobenius operator: self^p
    fn first_frobenius(&self) -> Self;

    /// Compute the second Frobenius operator: self^p²
    fn second_frobenius(&self) -> Self;

    /// Compute the fifth cyclotomic exponentiation: self^(p⁴ + p³ + p² + p)
    fn exp_fifth_cyclotomic_minus_one(&self) -> Self;

    /// Compute the fifth cyclotomic exponentiation: self^(p⁴ + p³ + p² + p + 1)
    fn exp_fifth_cyclotomic(&self) -> F;

    /// Check if the element is a square in Fp, assumes x != 0
    fn is_square_base(x: &F) -> bool;

    /// Compute the square root of the element in Fp, assumes x is an square and x != 0,1
    fn sqrt_base(x: &F) -> F;

    /// Check if the element is a square in Fp⁵ and returns the fith cyclotomic exponentiation, assumes self != 0
    fn is_square(&self) -> (F, bool);

    /// Compute the square root of the element in Fp⁵
    fn sqrt(&self) -> Option<Self>
    where
        Self: Sized;

    /// Compute the sign of a field element
    fn sign0(&self) -> bool;
}

impl<F: PrimeField64> SquaringFp5<F> for QuinticExtension<F> {
    fn first_frobenius(&self) -> Self {
        let a = self.as_basis_coefficients_slice();
        Self::from_basis_coefficients_fn(|i| F::from_u64(Self::GAMMAS1[i]) * a[i])
    }

    fn second_frobenius(&self) -> Self {
        let a = self.as_basis_coefficients_slice();
        Self::from_basis_coefficients_fn(|i| F::from_u64(Self::GAMMAS2[i]) * a[i])
    }

    fn exp_fifth_cyclotomic_minus_one(&self) -> Self {
        let t0 = self.first_frobenius() * self.second_frobenius(); // self^(p² + p)
        let t1 = t0.second_frobenius(); // self^(p⁴ + p³)
        t0 * t1 // self^(p⁴ + p³ + p² + p)
    }

    fn exp_fifth_cyclotomic(&self) -> F {
        let exp = self.exp_fifth_cyclotomic_minus_one(); // self^(p⁴ + p³ + p² + p)

        let a = self.as_basis_coefficients_slice();
        let b = exp.as_basis_coefficients_slice();
        a[0] * b[0] + F::from_u64(Self::NON_RESIDUE) * (a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1])
        // self^(p⁴ + p³ + p² + p + 1)
    }

    fn is_square_base(x: &F) -> bool {
        // (p-1)/2 = 2^63 - 2^31 -> x^((p-1)/2) = x^(2^63) / x^(2^31)
        let exp_31 = x.exp_power_of_2(31);
        let exp_63 = exp_31.exp_power_of_2(32);
        let symbol = exp_63 / exp_31;
        symbol == F::ONE
    }

    fn sqrt_base(x: &F) -> F {
        // We use the Cipolla's algorithm as implemented here https://github.com/Plonky3/Plonky3/pull/439/files
        // The reason to choose Cipolla's algorithm is that it outperforms Tonelli-Shanks when S·(S-1) > 8m + 20,
        // where S is the largest power of two dividing p-1 and m is the number of bits in p
        // In this case we have: S = 32 and m = 64, so S·(S-1) = 992 > 8*64 + 20 = 532
        let x = *x;

        // 1] Compute a ∈ Fp such that a² - x is not a square
        let g = F::GENERATOR;
        let mut a = F::ONE;
        let mut nonresidue = a - x;
        while Self::is_square_base(&nonresidue) {
            a *= g;
            nonresidue = a.square() - x;
        }

        // 2] Compute (a + sqrt(a² - x))^((p+1)/2)
        let mut result = CipollaExtension::new(a, F::ONE);
        result = result.exp(nonresidue);

        result.real
    }

    fn is_square(&self) -> (F, bool) {
        // Compute a = self^(p⁴ + p³ + p² + p + 1), a ∈ Fp
        let pow_fifth_cyclo = self.exp_fifth_cyclotomic();

        // Checks whether a is a square in Fp
        (pow_fifth_cyclo, Self::is_square_base(&pow_fifth_cyclo))
    }

    fn sqrt(&self) -> Option<Self> {
        // We compute the square root using the identity:
        //      1     p⁴ + p³ + p² + p + 1       p+1          p+1
        //     --- + ----------------------  = (-----)·p³ + (-----)·p + 1
        //      2              2                  2            2

        // sqrt(0) = 0 and sqrt(1) = 1
        if self.is_zero() || self.is_one() {
            return Some(*self);
        }

        let (exp_fifth_cyclo, is_square) = self.is_square();

        // If it's not a square, there is no square root
        if !is_square {
            return None;
        }

        // First Part: Compute the square root of self^-(p⁴ + p³ + p² + p + 1) ∈ Fp
        let x = Self::sqrt_base(&exp_fifth_cyclo.inverse());

        // Second Part: Compute self^(((p+1)/2)p³ + ((p+1)/2)p + 1)

        // 1] Compute self^((p+1)/2). Notice (p+1)/2 = 2^63 - 2^31 + 1
        let pow_31 = self.exp_power_of_2(31);
        let pow_63 = pow_31.exp_power_of_2(32);
        let pow = *self * pow_63 / pow_31;

        // 2] Compute the rest using Frobenius
        let mut pow_frob = pow.first_frobenius(); // self^(((p+1)/2)p)
        let mut y = pow_frob;
        pow_frob = pow_frob.second_frobenius(); // self^(((p+1)/2)p³)
        y *= pow_frob; // self^(((p+1)/2)p³ + ((p+1)/2)p)
        y *= *self; // self^(((p+1)/2)p³ + ((p+1)/2)p + 1)

        Some(y * x)
    }

    fn sign0(&self) -> bool {
        let e_coeffs = self.as_basis_coefficients_slice();
        let mut result = false;
        let mut zero = true;
        for coeff in e_coeffs.iter() {
            let sign_i = (coeff.as_canonical_u64() & 1) == 1;
            let zero_i = coeff.is_zero();
            result = result || (zero && sign_i);
            zero = zero && zero_i;
        }

        result
    }
}

/// Extension field for Cipolla's algorithm, adapted from [Plonky3 PR #439](https://github.com/Plonky3/Plonky3/pull/439)
/// Cipolla extension is defined as Fp\[sqrt(a² - n)\], where a² - n is a non-residue in Fp
#[derive(Clone, Copy, Debug)]
struct CipollaExtension<F: Field> {
    real: F,
    imag: F,
}

impl<F: Field> CipollaExtension<F> {
    fn new(real: F, imag: F) -> Self {
        Self { real, imag }
    }

    fn mul(&self, other: Self, nonresidue: F) -> Self {
        Self::new(
            self.real * other.real + nonresidue * self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }

    fn square(&self, nonresidue: F) -> Self {
        let real = self.real.square() + nonresidue * self.imag.square();
        let imag = F::TWO * self.real * self.imag;
        Self::new(real, imag)
    }

    fn div(&self, other: Self, nonresidue: F) -> Self {
        let denom = other.real.square() - nonresidue * other.imag.square();
        let real = (self.real * other.real - nonresidue * self.imag * other.imag) / denom;
        let imag = (self.imag * other.real - self.real * other.imag) / denom;
        Self::new(real, imag)
    }

    fn exp_power_of_2(&self, power_log: usize, nonresidue: F) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square(nonresidue);
        }
        res
    }

    // Computes exponentiation by (p+1)/2 = 2^63 - 2^31 + 1
    fn exp(&self, nonresidue: F) -> Self {
        let pow_31 = self.exp_power_of_2(31, nonresidue);
        let pow_63 = pow_31.exp_power_of_2(32, nonresidue);
        let pow = pow_63.div(pow_31, nonresidue);
        self.mul(pow, nonresidue)
    }
}

impl<F: PrimeField64> Distribution<QuinticExtension<F>> for StandardUniform
where
    StandardUniform: Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> QuinticExtension<F> {
        QuinticExtension::new(array::from_fn(|_| self.sample(rng)))
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, StandardUniform},
        rng,
    };

    use super::*;

    #[test]
    fn test_inv() {
        let mut rng = rng();
        for _ in 0..1000 {
            let x: QuinticExtension<Goldilocks> = StandardUniform.sample(&mut rng);
            let inv = x.inv();
            assert_eq!(x * inv, QuinticExtension::ONE);
        }
    }

    #[test]
    fn test_is_square() {
        let g: QuinticExtension<Goldilocks> = QuinticExtension::GENERATOR;
        let mut x: QuinticExtension<Goldilocks> = QuinticExtension::ONE;
        for i in 0..1000 {
            let (_, is_square) = x.is_square();
            assert_eq!(is_square, i % 2 == 0);
            x *= g;
        }
    }

    #[test]
    fn test_sqrt() {
        // Test edge cases
        let zero_sqrt: Option<QuinticExtension<Goldilocks>> = QuinticExtension::ZERO.sqrt();
        assert_eq!(zero_sqrt, Some(QuinticExtension::ZERO));

        let one_sqrt: Option<QuinticExtension<Goldilocks>> = QuinticExtension::ONE.sqrt();
        assert_eq!(one_sqrt, Some(QuinticExtension::ONE));

        // Test a non-square
        let g: QuinticExtension<Goldilocks> = QuinticExtension::GENERATOR;
        let g_sqrt = g.sqrt();
        assert_eq!(g_sqrt, None);

        // Test random elements
        let mut rng = rng();
        for _ in 0..1000 {
            let x: QuinticExtension<Goldilocks> = StandardUniform.sample(&mut rng);
            let x_sq = x.square();
            let x_sqrt = x_sq.sqrt().unwrap();
            assert_eq!(x_sqrt * x_sqrt, x_sq);
        }
    }
}
