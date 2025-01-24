use num_bigint::BigUint;
use num_traits::{Num, One, Zero};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct F3g {
    pub p: BigUint,
    pub zero: BigUint,
    pub one: BigUint,
    pub nqr: BigUint,
    pub shift: BigUint,
    pub shift_inv: BigUint,
    pub half: BigUint,
    pub negone: BigUint,
    pub k: BigUint,
    pub s: u32,
    pub t: BigUint,
    pub n8: usize,
    pub n32: usize,
    pub n64: usize,
    pub m: usize,
    pub bit_length: u32,
    pub w: Vec<BigUint>,          // Precomputed roots of unity
    pub wi: Vec<BigUint>,         // Precomputed inverse roots of unity
    pub roots: Vec<Vec<BigUint>>, // Cached roots for FFT
}

impl Default for F3g {
    fn default() -> Self {
        Self::new()
    }
}

impl F3g {
    pub fn new() -> Self {
        let p = BigUint::from_str_radix("FFFFFFFF00000001", 16).unwrap();
        let zero = BigUint::zero();
        let one = BigUint::one();
        let nqr = BigUint::from(7u64);
        let shift = BigUint::from(7u64);
        let shift_inv = Self::mod_inv(&shift, &p).unwrap();
        let half = &p >> 1u32;
        let negone = &p - &one;
        let k = BigUint::from(12275445934081160404u64);
        let s = 32;
        let t = (&p - &one) >> s;
        let n8 = 8;
        let n32 = 2;
        let n64 = 1;
        let m = 3;
        let bit_length = p.bits();

        let mut field = F3g {
            p,
            zero,
            one,
            nqr,
            shift,
            shift_inv,
            half,
            negone,
            k,
            s,
            t,
            n8,
            n32,
            n64,
            m,
            bit_length: bit_length as u32,
            w: Vec::new(),
            wi: Vec::new(),
            roots: Vec::new(),
        };

        field.compute_roots();
        field
    }

    /// Adds two numbers in the field
    pub fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        (a + b) % &self.p
    }

    /// Subtracts two numbers in the field
    pub fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint {
        if a >= b {
            a - b
        } else {
            &self.p - (b - a)
        }
    }

    /// Negates a number in the field
    pub fn neg(&self, a: &BigUint) -> BigUint {
        if a.is_zero() {
            a.clone()
        } else {
            &self.p - a
        }
    }

    /// Multiplies two numbers in the field
    pub fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        (a * b) % &self.p
    }

    /// Squares a number in the field
    pub fn square(&self, a: &BigUint) -> BigUint {
        self.mul(a, a)
    }

    /// Computes the modular inverse of a number in the field
    pub fn inv(&self, a: &BigUint) -> BigUint {
        a.modpow(&(&self.p - BigUint::from(2u64)), &self.p)
    }

    /// Divides two numbers in the field
    pub fn div(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let b_inv = self.inv(b);
        self.mul(a, &b_inv)
    }

    /// Checks if two numbers are equal in the field
    pub fn eq(&self, a: &BigUint, b: &BigUint) -> bool {
        a == b
    }

    /// Checks if a number is zero in the field
    pub fn is_zero(&self, a: &BigUint) -> bool {
        a.is_zero()
    }

    /// Exponentiates a number in the field
    pub fn exp(&self, base: &BigUint, exp: &BigUint) -> BigUint {
        base.modpow(exp, &self.p)
    }

    /// Generates a random element in the field
    pub fn random(&self) -> BigUint {
        let mut rng = rand::thread_rng();
        let mut bytes = vec![0u8; (self.bit_length / 8) as usize];
        rng.fill(&mut bytes[..]);
        BigUint::from_bytes_be(&bytes) % &self.p
    }

    /// Multiplies a number by a scalar in the field
    pub fn mul_scalar(&self, a: &BigUint, scalar: u64) -> BigUint {
        self.mul(a, &BigUint::from(scalar))
    }

    /// Converts a number to a string in the specified base
    pub fn to_string(&self, a: &BigUint, base: u32) -> String {
        a.to_str_radix(base)
    }

    /// Computes precomputed roots of unity and their inverses
    fn compute_roots(&mut self) {
        let mut nqr = self.one.clone();
        while self.exp(&nqr, &self.half) == self.one {
            nqr = self.add(&nqr, &self.one);
        }

        self.w = vec![self.zero.clone(); (self.s + 1) as usize];
        self.wi = vec![self.zero.clone(); (self.s + 1) as usize];

        self.w[self.s as usize] = self.exp(&nqr, &self.t);
        self.wi[self.s as usize] = self.inv(&self.w[self.s as usize]);

        for i in (0..self.s).rev() {
            self.w[i as usize] = self.square(&self.w[i as usize + 1]);
            self.wi[i as usize] = self.square(&self.wi[i as usize + 1]);
        }

        self.roots = vec![vec![]; (self.s + 1) as usize];
    }

    /// Precomputes roots for FFT
    pub fn set_roots(&mut self, n: usize) {
        for i in (0..=n).rev() {
            if self.roots[i].is_empty() {
                let mut r = self.one.clone();
                let nroots = 1 << i;
                self.roots[i] = vec![self.zero.clone(); nroots];
                for j in 0..nroots {
                    self.roots[i][j] = r.clone();
                    r = self.mul(&r, &self.w[i]);
                }
            }
        }
    }

    /// Computes bit-reversed index
    pub fn bit_reverse(x: usize, n_bits: usize) -> usize {
        let mut x = x;
        let mut result = 0;
        for _ in 0..n_bits {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }

    /// Computes the modular inverse using the Extended Euclidean Algorithm
    fn mod_inv(a: &BigUint, m: &BigUint) -> Option<BigUint> {
        let mut t = BigUint::zero();
        let mut new_t = BigUint::one();
        let mut r = m.clone();
        let mut new_r = a.clone();

        while !new_r.is_zero() {
            let quotient = &r / &new_r;
            let tmp_r = r.clone();
            let tmp_t = t.clone();
            r = new_r.clone();
            new_r = tmp_r - &quotient * &new_r;
            t = new_t.clone();
            new_t = tmp_t - &quotient * &new_t;
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
