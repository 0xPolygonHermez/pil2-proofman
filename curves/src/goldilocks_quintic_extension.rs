use num_bigint::BigUint;

use p3_goldilocks::Goldilocks;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing};
use p3_field::extension::BinomialExtensionField;

// Field Fp⁵ = F[X]/(X⁵-3) with fixed generator X + 2
pub type GoldilocksQuinticExtension = BinomialExtensionField<Goldilocks, 5>;

// Specific methods for computing square root of a goldilocks quintic extension
// as described in https://hackmd.io/CxJrIhv-SP65W3GWS_J5bw?view#Extension-Field-Selection
// which is inspired by https://github.com/succinctlabs/sp1/blob/dev/crates/stark/src/septic_extension.rs
pub trait Squaring {
    const EXP: u64 = 0x7FFF_FFFF_8000_0000; // (p-1)/2

    fn gammas1(i: usize) -> Self;

    fn gammas2(i: usize) -> Self;

    fn first_frobenius(&self) -> Self;

    fn second_frobenius(&self) -> Self;

    fn exp_fifth_cyclotomic(&self) -> Self;

    fn is_square(&self) -> (Goldilocks, bool);

    fn sqrt(&self) -> Option<Self>
    where
        Self: Sized;
}

impl Squaring for GoldilocksQuinticExtension {
    fn gammas1(index: usize) -> Self {
        // ```sage
        // p = 2**64 - 2**32 + 1
        // F = GF(p)
        //
        // for i in range(1,5):
        //     gamma1i = F(3)^(i*(p-1)/5)
        //     print(f"gamma1{i} = {gamma1i}")
        // ```
        match index {
            0 => Self::ONE,
            1 => GoldilocksQuinticExtension::from_u64(1041288259238279555),
            2 => GoldilocksQuinticExtension::from_u64(15820824984080659046),
            3 => GoldilocksQuinticExtension::from_u64(211587555138949697),
            4 => GoldilocksQuinticExtension::from_u64(1373043270956696022),
            _ => panic!("Invalid index for gammas1: {}", index),
        }
    }

    fn gammas2(index: usize) -> Self {
        // ```sage
        // p = 2**64 - 2**32 + 1
        // F = GF(p)
        //
        // for i in range(1,5):
        //     gamma1i = F(3)^(i*(p^2-1)/5)
        //     print(f"gamma2{i} = {gamma1i^2}")
        // ```
        match index {
            0 => Self::ONE,
            1 => GoldilocksQuinticExtension::from_u64(15820824984080659046),
            2 => GoldilocksQuinticExtension::from_u64(1373043270956696022),
            3 => GoldilocksQuinticExtension::from_u64(1041288259238279555),
            4 => GoldilocksQuinticExtension::from_u64(211587555138949697),
            _ => panic!("Invalid index for gammas1: {}", index),
        }
    }

    fn first_frobenius(&self) -> Self {
        let a: &[Goldilocks] = self.as_basis_coefficients_slice();
        let mut result = Self::ZERO;
        for i in 0..5 {
            result += Self::gammas1(i) * a[i];
        }
        result
    }

    fn second_frobenius(&self) -> Self {
        let a: &[Goldilocks] = self.as_basis_coefficients_slice();
        let mut result = Self::ZERO;
        for i in 0..5 {
            result += Self::gammas2(i) * a[i];
        }
        result
    }

    fn exp_fifth_cyclotomic(&self) -> Self {
        let t0 = self.first_frobenius() * self.second_frobenius();
        let t1 = t0.second_frobenius();
        *self * t0 * t1
    }

    // Computes self^((p⁵ - 1)/2), assumes self != 0
    fn is_square(&self) -> (Goldilocks, bool) {
        // Compute a = self^(p⁴ + p³ + p² + p + 1), a ∈ Fp
        let pow_fifth_cyclo: Goldilocks =
            self.exp_fifth_cyclotomic().as_base().expect("This should be a base field element");

        // Compute a^((p - 1)/2)
        let pow = pow_fifth_cyclo.exp_u64(Self::EXP);

        // Check if self^((p⁵ - 1)/2) == 1
        (pow_fifth_cyclo, pow == Goldilocks::ONE)
    }

    // We compute the square root using the identity:
    //      1     p⁴ + p³ + p² + p + 1       p+1          p+1
    //     --- + ----------------------  = (-----)·p³ + (-----)·p + 1
    //      2              2                  2            2
    fn sqrt(&self) -> Option<Self> {
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
        // ================================================================
        // We use the Cipolla's algorithm as implemented here https://github.com/Plonky3/Plonky3/pull/439/files
        // The reason to choose Cipolla's algorithm is that it outperforms Tonelli-Shanks when S·(S-1) > 8m + 20,
        // where S is the largest power of two dividing p-1 and m is the number of bits in p
        // In this case we have: S = 32 and m = 64, so S·(S-1) = 992 > 8*64 + 20 = 532
        let n = exp_fifth_cyclo.inverse();

        // 1] Compute a ∈ Fp such that a² - n is not a square
        let g = Goldilocks::GENERATOR;
        let mut a = Goldilocks::ONE;
        let mut nonresidue = a - n;
        while nonresidue.exp_u64(Self::EXP) == Goldilocks::ONE {
            a *= g;
            nonresidue = a.square() - n;
        }

        // 2] Compute (a + sqrt(a² - n))^((p+1)/2)
        let exp = BigUint::from(Self::EXP + 1);
        let mut x = CipollaExtension::new(a, Goldilocks::ONE);
        x = x.pow(&exp, nonresidue);

        // Second Part: Compute self^(((p+1)/2)p³ + ((p+1)/2)p + 1)
        // ================================================================
        // 1] Compute self^((p+1)/2)
        let pow = self.exp_u64(Self::EXP + 1);

        // 2] Compute the rest using Frobenius
        let mut pow_frob = pow.first_frobenius(); // self^(((p+1)/2)p)
        let mut result = pow_frob;
        pow_frob = pow_frob.second_frobenius(); // self^(((p+1)/2)p³)
        result *= pow_frob; // self^(((p+1)/2)p³ + ((p+1)/2)p)
        result *= *self; // self^(((p+1)/2)p³ + ((p+1)/2)p + 1)

        Some(result * x.real)
    }
}

// Extension field for Cipolla's algorithm, taken from https://github.com/Plonky3/Plonky3/pull/439/files
// adapted to the new Plonky3 API
#[derive(Clone, Copy, Debug)]
struct CipollaExtension<F: Field> {
    real: F,
    imag: F,
}

impl<F: Field> CipollaExtension<F> {
    fn new(real: F, imag: F) -> Self {
        Self { real, imag }
    }

    fn one() -> Self {
        Self::new(F::ONE, F::ZERO)
    }

    fn mul(&self, other: Self, nonresidue: F) -> Self {
        Self::new(
            self.real * other.real + nonresidue * self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }

    fn pow(&self, exp: &BigUint, nonresidue: F) -> Self {
        let mut result = Self::one();
        let mut base = *self;
        let bits = exp.bits();

        for i in 0..bits {
            if exp.bit(i) {
                result = result.mul(base, nonresidue);
            }
            base = base.mul(base, nonresidue);
        }
        result
    }
}

// TODO: Add some tests
