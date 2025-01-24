use num_bigint::BigUint;
use num_traits::{One, Zero, Num};
use rand::Rng;

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
    pub m: usize,
    pub bit_length: u32,
    pub sqrt_fn: Option<fn(&F3g, &BigUint) -> Option<BigUint>>, // Square root function
}

impl Default for F3g {
    fn default() -> Self {
        Self::new()
    }
}

impl F3g {
    /// Constructor for the F3g field
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

        // Calculate the bit length of the modulus
        let bit_length = p.bits() as u32;

        Self {
            p,
            zero,
            one,
            nqr,
            shift,
            shift_inv,
            half,
            negone,
            k,
            s: 32,
            t,
            n8: 8,
            n32: 2,
            n64: 1,
            m: 3,
            bit_length,
            sqrt_fn: None, // Initialize without a square root function
        }
    }

    /// Modular addition
    pub fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        (a + b) % &self.p
    }

    /// Modular subtraction
    pub fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a >= b {
            a - b
        } else {
            &self.p - (b - a)
        }
    }

    /// Modular negation
    pub fn neg(&self, a: &BigUint) -> BigUint {
        if a.is_zero() {
            a.clone()
        } else {
            &self.p - a
        }
    }

    /// Modular multiplication
    pub fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        (a * b) % &self.p
    }

    /// Modular division
    pub fn div(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let inv_b = F3g::mod_inv(b, &self.p).expect("Division by zero");
        self.mul(a, &inv_b)
    }

    /// Modular exponentiation
    pub fn exp(&self, base: &BigUint, exp: &BigUint) -> BigUint {
        base.modpow(exp, &self.p)
    }

    /// Generate a random element in the field
    pub fn random(&self) -> BigUint {
        let mut rng = rand::thread_rng();
        let mut bytes = vec![0u8; (self.bit_length / 8) as usize];
        rng.fill(&mut bytes[..]);
        BigUint::from_bytes_be(&bytes) % &self.p
    }

    /// Modular inversion using the Extended Euclidean Algorithm
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

    /// Square a number
    pub fn square(&self, a: &BigUint) -> BigUint {
        self.mul(a, a)
    }

    /// Compute the square root of a number if supported
    pub fn sqrt(&self, a: &BigUint) -> Option<BigUint> {
        if let Some(sqrt_fn) = self.sqrt_fn {
            sqrt_fn(self, a)
        } else {
            None
        }
    }

    /// Set the square root function based on the field configuration
    pub fn set_sqrt_fn(&mut self, sqrt_fn: fn(&F3g, &BigUint) -> Option<BigUint>) {
        self.sqrt_fn = Some(sqrt_fn);
    }
}
