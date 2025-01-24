use num_bigint::BigUint;
use num_traits::{One, Zero, Num};

/// Represents the base field and the field extension F3g.
#[derive(Debug, Clone)]
pub struct F3g {
    pub p: BigUint, // Prime modulus
    pub zero: BigUint,
    pub one: BigUint,
    pub nqr: BigUint, // Non-quadratic residue
    pub shift: BigUint,
    pub shift_inv: BigUint,
    pub half: BigUint,
    pub negone: BigUint,
    pub k: BigUint, // Generator of the multiplicative group
    pub s: u32,
    pub t: BigUint,
    pub n8: usize,
    pub n32: usize,
    pub n64: usize,
    pub m: usize, // Field extension degree
    pub bit_length: u32,
}

/// Represents elements in the F3g extension field (x^3 - x - 1).
#[derive(Debug, Clone, PartialEq)]
pub struct F3gElement {
    pub coeffs: [BigUint; 3], // Coefficients for x^0, x^1, x^2
}

impl F3g {
    /// Constructor for the F3g field.
    pub fn new() -> Self {
        let p = BigUint::from_str_radix("FFFFFFFF00000001", 16).unwrap();
        let one = BigUint::one();
        let zero = BigUint::zero();
        let nqr = BigUint::from(7u64);
        let shift = BigUint::from(7u64);
        let shift_inv = F3g::mod_inv(&shift, &p).unwrap();
        let half = &p >> 1u32;
        let negone = &p - &one;
        let k = BigUint::from(12275445934081160404u64);
        let t = (&p - &one) >> 32;

        let bit_length = p.bits() as u32;

        Self { p, zero, one, nqr, shift, shift_inv, half, negone, k, s: 32, t, n8: 8, n32: 2, n64: 1, m: 3, bit_length }
    }

    /// Modular addition for field elements.
    pub fn add(&self, a: &F3gElement, b: &F3gElement) -> F3gElement {
        F3gElement {
            coeffs: [
                (a.coeffs[0].clone() + b.coeffs[0].clone()) % &self.p,
                (a.coeffs[1].clone() + b.coeffs[1].clone()) % &self.p,
                (a.coeffs[2].clone() + b.coeffs[2].clone()) % &self.p,
            ],
        }
    }

    /// Modular subtraction for field elements.
    pub fn sub(&self, a: &F3gElement, b: &F3gElement) -> F3gElement {
        F3gElement {
            coeffs: [
                self.sub_scalar(&a.coeffs[0], &b.coeffs[0]),
                self.sub_scalar(&a.coeffs[1], &b.coeffs[1]),
                self.sub_scalar(&a.coeffs[2], &b.coeffs[2]),
            ],
        }
    }

    fn sub_scalar(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a >= b {
            a - b
        } else {
            &self.p - (b - a)
        }
    }

    /// Modular negation for field elements.
    pub fn neg(&self, a: &F3gElement) -> F3gElement {
        F3gElement {
            coeffs: [self.neg_scalar(&a.coeffs[0]), self.neg_scalar(&a.coeffs[1]), self.neg_scalar(&a.coeffs[2])],
        }
    }

    fn neg_scalar(&self, a: &BigUint) -> BigUint {
        if a.is_zero() {
            a.clone()
        } else {
            &self.p - a
        }
    }

    /// Modular multiplication for field elements.
    pub fn mul(&self, a: &F3gElement, b: &F3gElement) -> F3gElement {
        let p = &self.p;

        let a0 = &a.coeffs[0];
        let a1 = &a.coeffs[1];
        let a2 = &a.coeffs[2];

        let b0 = &b.coeffs[0];
        let b1 = &b.coeffs[1];
        let b2 = &b.coeffs[2];

        let c0 = (a0 * b0 + a1 * b2 + a2 * b1) % p;
        let c1 = (a0 * b1 + a1 * b0 + a2 * b2) % p;
        let c2 = (a0 * b2 + a1 * b1 + a2 * b0) % p;

        F3gElement { coeffs: [c0, c1, c2] }
    }

    /// Modular inversion for field elements.
    pub fn inv(&self, a: &F3gElement) -> F3gElement {
        // Use an appropriate inversion algorithm for extension fields.
        unimplemented!()
    }

    /// Modular scalar inversion using the Extended Euclidean Algorithm.
    pub fn mod_inv(a: &BigUint, m: &BigUint) -> Option<BigUint> {
        let mut t = BigUint::zero();
        let mut new_t = BigUint::one();
        let mut r = m.clone();
        let mut new_r = a.clone();

        while !new_r.is_zero() {
            let quotient = &r / &new_r;
            let tmp_r = r.clone();
            let tmp_t = t.clone();
            r = new_r.clone();
            new_r = &tmp_r - &quotient * &new_r;
            t = new_t.clone();
            new_t = &tmp_t - &quotient * &new_t;
        }

        if r > BigUint::one() {
            return None; // No inverse exists
        }

        if t < BigUint::zero() {
            t += m;
        }
        Some(t)
    }
}

impl F3gElement {
    /// Create a zero element in the extension field.
    pub fn zero() -> Self {
        F3gElement { coeffs: [BigUint::zero(), BigUint::zero(), BigUint::zero()] }
    }

    /// Create a one element in the extension field.
    pub fn one() -> Self {
        F3gElement { coeffs: [BigUint::one(), BigUint::zero(), BigUint::zero()] }
    }
}
