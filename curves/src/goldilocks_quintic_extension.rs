use p3_goldilocks::Goldilocks;
use p3_field::{BasedVectorSpace, Field, PrimeField64, PrimeCharacteristicRing};
use p3_field::extension::BinomialExtensionField;

/// Field Fp⁵ = F\[X\]/(X⁵-3) with generator X + 2
pub type GoldilocksQuinticExtension = BinomialExtensionField<Goldilocks, 5>;

/// Methods for computing the square root in the GoldilocksQuinticExtension field
/// as described in [Elliptic Curves over Goldilocks](https://hackmd.io/CxJrIhv-SP65W3GWS_J5bw?view#Extension-Field-Selection),
/// which is inspired by [Curve ecGFp5](https://github.com/pornin/ecgfp5/tree/main)
pub trait SquaringFp5 {
    /// Constants for the first Frobenius operator
    const GAMMAS1: [u64; 5] = [1, 1041288259238279555, 15820824984080659046, 211587555138949697, 1373043270956696022];

    /// Constants for the second Frobenius operator
    const GAMMAS2: [u64; 5] = [1, 15820824984080659046, 1373043270956696022, 1041288259238279555, 211587555138949697];

    /// Compute the first Frobenius operator: self^p
    fn first_frobenius(&self) -> Self;

    /// Compute the second Frobenius operator: self^p²
    fn second_frobenius(&self) -> Self;

    /// Compute the fifth cyclotomic exponentiation: self^(p⁴ + p³ + p² + p + 1)
    fn exp_fifth_cyclotomic(&self) -> Goldilocks;

    /// Check if the element is a square in Fp, assumes x != 0
    fn is_square_base(x: &Goldilocks) -> bool;

    /// Compute the square root of the element in Fp, assumes x is an square and x != 0,1
    fn sqrt_base(x: &Goldilocks) -> Goldilocks;

    /// Check if the element is a square in Fp⁵ and returns the fith cyclotomic exponentiation, assumes self != 0
    fn is_square(&self) -> (Goldilocks, bool);

    /// Compute the square root of the element in Fp⁵
    fn sqrt(&self) -> Option<Self>
    where
        Self: Sized;

    /// Compute the sign of a field element
    fn sign0(&self) -> bool;
}

impl SquaringFp5 for GoldilocksQuinticExtension {
    fn first_frobenius(&self) -> Self {
        let a: &[Goldilocks] = self.as_basis_coefficients_slice();
        Self::from_basis_coefficients_fn(|i| Goldilocks::from_u64(Self::GAMMAS1[i]) * a[i])
    }

    fn second_frobenius(&self) -> Self {
        let a: &[Goldilocks] = self.as_basis_coefficients_slice();
        Self::from_basis_coefficients_fn(|i| Goldilocks::from_u64(Self::GAMMAS2[i]) * a[i])
    }

    fn exp_fifth_cyclotomic(&self) -> Goldilocks {
        let t0 = self.first_frobenius() * self.second_frobenius(); // self^(p² + p)
        let t1 = t0.second_frobenius(); // self^(p⁴ + p³)
        let t2 = t0 * t1; // self^(p⁴ + p³ + p² + p)

        let a: &[Goldilocks] = self.as_basis_coefficients_slice();
        let b: &[Goldilocks] = t2.as_basis_coefficients_slice();
        // self^(p⁴ + p³ + p² + p + 1)
        a[0] * b[0] + Goldilocks::from_u8(3) * (a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1])
    }

    fn is_square_base(x: &Goldilocks) -> bool {
        // (p-1)/2 = 2^63 - 2^31 -> x^((p-1)/2) = x^(2^63) / x^(2^31)
        let exp_31 = x.exp_power_of_2(31);
        let exp_63 = exp_31.exp_power_of_2(32);
        let symbol = exp_63 / exp_31;
        symbol == Goldilocks::ONE
    }

    fn sqrt_base(x: &Goldilocks) -> Goldilocks {
        // We use the Cipolla's algorithm as implemented here https://github.com/Plonky3/Plonky3/pull/439/files
        // The reason to choose Cipolla's algorithm is that it outperforms Tonelli-Shanks when S·(S-1) > 8m + 20,
        // where S is the largest power of two dividing p-1 and m is the number of bits in p
        // In this case we have: S = 32 and m = 64, so S·(S-1) = 992 > 8*64 + 20 = 532
        let x = *x;

        // 1] Compute a ∈ Fp such that a² - x is not a square
        let g = Goldilocks::GENERATOR;
        let mut a = Goldilocks::ONE;
        let mut nonresidue = a - x;
        while Self::is_square_base(&nonresidue) {
            a *= g;
            nonresidue = a.square() - x;
        }

        // 2] Compute (a + sqrt(a² - x))^((p+1)/2)
        let mut result = CipollaExtension::new(a, Goldilocks::ONE);
        result = result.exp(nonresidue);

        result.real
    }

    fn is_square(&self) -> (Goldilocks, bool) {
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
        let e_coeffs: &[Goldilocks] = self.as_basis_coefficients_slice();
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

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, StandardUniform},
        rng,
    };

    use super::*;

    #[test]
    fn test_is_square() {
        let g = GoldilocksQuinticExtension::GENERATOR;
        let mut x = GoldilocksQuinticExtension::ONE;
        for i in 0..1000 {
            let (_, is_square) = x.is_square();
            assert_eq!(is_square, i % 2 == 0);
            x *= g;
        }
    }

    #[test]
    fn test_sqrt() {
        // Test edge cases
        let zero_sqrt = GoldilocksQuinticExtension::ZERO.sqrt();
        assert_eq!(zero_sqrt, Some(GoldilocksQuinticExtension::ZERO));

        let one_sqrt = GoldilocksQuinticExtension::ONE.sqrt();
        assert_eq!(one_sqrt, Some(GoldilocksQuinticExtension::ONE));

        // Test a non-square
        let g = GoldilocksQuinticExtension::GENERATOR;
        let g_sqrt = g.sqrt();
        assert_eq!(g_sqrt, None);

        // Test random elements
        let mut rng = rng();
        for _ in 0..1000 {
            let x: GoldilocksQuinticExtension = StandardUniform.sample(&mut rng);
            let x_sq = x.square();
            let x_sqrt = x_sq.sqrt().unwrap();
            assert_eq!(x_sqrt * x_sqrt, x_sq);
        }
    }
}
