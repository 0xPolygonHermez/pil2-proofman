//! The security bounds in exact arithmetic, rounded against the prover: every irrational (√ρ) is
//! replaced by a rational on the weak side, so the bits reported never exceed the true ones.
//! This is the one implementation of each bound formula the check uses.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

/// An exact fraction `num/den`, `den > 0`, not reduced (the sizes here stay small).
#[derive(Clone, Debug)]
pub struct Rational {
    num: BigInt,
    den: BigInt,
}

impl Rational {
    pub fn new(num: impl Into<BigInt>, den: impl Into<BigInt>) -> Self {
        let (mut num, mut den) = (num.into(), den.into());
        assert!(!den.is_zero(), "zero denominator");
        if den.is_negative() {
            num = -num;
            den = -den;
        }
        Self { num, den }
    }

    pub fn int(v: impl Into<BigInt>) -> Self {
        Self { num: v.into(), den: BigInt::one() }
    }

    /// 2^e, any integer e.
    pub fn pow2(e: i64) -> Self {
        if e >= 0 {
            Self::int(BigInt::one() << e as usize)
        } else {
            Self::new(1, BigInt::one() << (-e) as usize)
        }
    }

    /// The exact value of a finite f64.
    pub fn from_f64(v: f64) -> Self {
        assert!(v.is_finite(), "non-finite f64");
        let bits = v.to_bits();
        let sign = if bits >> 63 == 1 { -1 } else { 1 };
        let exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = (bits & ((1u64 << 52) - 1)) as i64;
        let (mant, e) = if exp == 0 { (frac, -1074) } else { (frac | (1 << 52), exp - 1075) };
        Self::int(sign * mant) * Self::pow2(e)
    }

    pub fn pow(&self, e: u64) -> Self {
        Self { num: num_traits::pow(self.num.clone(), e as usize), den: num_traits::pow(self.den.clone(), e as usize) }
    }

    /// ⌈self⌉ for a non-negative value.
    pub fn ceil_u64(&self) -> u64 {
        assert!(!self.num.is_negative());
        let q = (&self.num + &self.den - BigInt::one()) / &self.den;
        q.to_u64().expect("ceil overflows u64")
    }

    /// ⌊self⌋ for a non-negative value.
    pub fn floor_u32(&self) -> u32 {
        (&self.num / &self.den).to_u32().expect("floor overflows u32")
    }

    /// self as an f64, for display only.
    pub fn to_f64(&self) -> f64 {
        let scale = 1u64 << 20;
        (&self.num * BigInt::from(scale) / &self.den).to_f64().unwrap_or(f64::NAN) / scale as f64
    }

    /// The largest b/`BITS_DEN` with `self ≤ 2^(−b/BITS_DEN)`: bits of security of the error
    /// `self`, rounded down. Exact (no logarithm is evaluated).
    pub fn security_bits(&self) -> Rational {
        assert!(self.num.sign() == Sign::Plus, "error must be positive");
        let d = BITS_DEN as u64;
        // self^d ≤ 2^-b  <=>  num^d · 2^b ≤ den^d.
        let pw = self.pow(d);
        let n = BigUint::try_from(pw.num).unwrap();
        let m = BigUint::try_from(pw.den).unwrap();
        if n > m {
            return Rational::int(0);
        }
        let mut b = (m.bits() as i64) - (n.bits() as i64) + 1;
        while b > 0 && (&n << (b as usize)) > m {
            b -= 1;
        }
        Rational::new(b.max(0), d as i64)
    }
}

/// Bits are reported on a grid of 1/`BITS_DEN`.
pub const BITS_DEN: i64 = 1024;

impl std::ops::Add for Rational {
    type Output = Rational;
    fn add(self, o: Rational) -> Rational {
        Rational { num: &self.num * &o.den + &o.num * &self.den, den: self.den * o.den }
    }
}
impl std::ops::Sub for Rational {
    type Output = Rational;
    fn sub(self, o: Rational) -> Rational {
        Rational { num: &self.num * &o.den - &o.num * &self.den, den: self.den * o.den }
    }
}
impl std::ops::Mul for Rational {
    type Output = Rational;
    fn mul(self, o: Rational) -> Rational {
        Rational { num: self.num * o.num, den: self.den * o.den }
    }
}
impl std::ops::Div for Rational {
    type Output = Rational;
    fn div(self, o: Rational) -> Rational {
        assert!(!o.num.is_zero(), "division by zero");
        Rational::new(self.num * o.den, self.den * o.num)
    }
}
impl PartialEq for Rational {
    fn eq(&self, o: &Rational) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Rational {}
impl PartialOrd for Rational {
    fn partial_cmp(&self, o: &Rational) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Rational {
    fn cmp(&self, o: &Rational) -> Ordering {
        (&self.num * &o.den).cmp(&(&o.num * &self.den))
    }
}
impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}", self.to_f64())
    }
}

fn r(v: i64) -> Rational {
    Rational::int(v)
}

/// A rational at least √2 (√2 = 1.41421356237...): the one place √ρ is rounded, upward.
fn sqrt2_upper() -> Rational {
    Rational::new(1_414_213_563i64, 1_000_000_000i64)
}

/// A rational ≥ 2^(μ/2) (that is 1/√ρ for ρ = 2^-μ), exact for even μ.
pub fn inv_sqrt_rate_upper(mu: u32) -> Rational {
    let half = Rational::pow2((mu / 2) as i64);
    if mu.is_multiple_of(2) {
        half
    } else {
        half * sqrt2_upper()
    }
}

/// A rational ≥ 2^(-μ/2) = √ρ: 2^(-(μ+1)/2)·√2 for odd μ.
pub fn sqrt_rate_upper(mu: u32) -> Rational {
    if mu.is_multiple_of(2) {
        Rational::pow2(-((mu / 2) as i64))
    } else {
        Rational::pow2(-(mu.div_ceil(2) as i64)) * sqrt2_upper()
    }
}

/// The decoding regime with its exact gap. `eta` is the JBR gap γ = η; `alpha` the UDR widening.
#[derive(Clone, Debug)]
pub enum ExactRegime {
    Jbr { eta: Rational },
    Udr { alpha: Rational },
}

/// The extension-field size p^3 for Goldilocks p, exactly.
pub fn goldilocks_cubic_field_size() -> Rational {
    let p = (BigInt::one() << 64usize) - (BigInt::one() << 32usize) + BigInt::one();
    Rational::int(&p * &p * &p)
}

impl ExactRegime {
    pub fn name(&self) -> &'static str {
        match self {
            ExactRegime::Jbr { .. } => "JBR",
            ExactRegime::Udr { .. } => "UDR",
        }
    }

    /// Whether the gap is admissible at rate 2^-mu (rounded against the prover).
    pub fn admits(&self, mu: u32) -> bool {
        let rho = Rational::pow2(-(mu as i64));
        match self {
            // η < (1 − √ρ)²/2, with √ρ rounded up.
            ExactRegime::Jbr { eta } => {
                let s = sqrt_rate_upper(mu);
                if s >= r(1) {
                    return false;
                }
                let d = r(1) - s;
                *eta < d.clone() * d / r(2)
            }
            ExactRegime::Udr { alpha } => self.udr_gap(&rho, alpha) < (r(1) - rho) / r(6),
        }
    }

    fn udr_gap(&self, rho: &Rational, alpha: &Rational) -> Rational {
        (r(1) - rho.clone()) / r(20) * (r(1) + alpha.clone())
    }

    /// 1 − δ, an upper bound, where δ is the proximity parameter at rate 2^-mu.
    pub fn one_minus_delta(&self, mu: u32) -> Rational {
        match self {
            ExactRegime::Jbr { eta } => sqrt_rate_upper(mu) + eta.clone(),
            ExactRegime::Udr { alpha } => {
                let rho = Rational::pow2(-(mu as i64));
                r(1) - (r(1) - rho.clone()) / r(2) + self.udr_gap(&rho, alpha)
            }
        }
    }

    /// Upper bound on the list size ℓ at rate 2^-mu.
    pub fn list_size(&self, mu: u32) -> u64 {
        match self {
            ExactRegime::Jbr { eta } => (inv_sqrt_rate_upper(mu) / (r(2) * eta.clone())).ceil_u64(),
            ExactRegime::Udr { .. } => 1,
        }
    }

    /// BCHKS25 error for independent coefficients, code of dimension `dim` at rate 2^-mu:
    /// Thm 4.2 (JBR, with the factor 2 in m the paper's statement omits) or Cor. 1.4 (UDR).
    pub fn error_linear(&self, mu: u32, dim: u64, q: &Rational) -> Rational {
        let rho = Rational::pow2(-(mu as i64));
        let n = Rational::int(dim) * Rational::pow2(mu as i64);
        match self {
            ExactRegime::Jbr { eta } => {
                // m = max(⌈√ρ/(2η)⌉, 3): the least integer with (2ηm)² ≥ ρ.
                let x = rho.clone() / (r(4) * eta.clone() * eta.clone());
                let mut m = BigInt::from(x.ceil_u64()).sqrt();
                while Rational::int(m.clone() * m.clone()) < x {
                    m += 1;
                }
                let m = m.max(BigInt::from(3));
                let big_m = Rational::int(m) + Rational::new(1, 2);
                let inv_sqrt = inv_sqrt_rate_upper(mu);
                // ((2M⁵ + 3Mγρ)·n / (3ρ√ρ) + M/√ρ) / |F|, with 1/ρ = 2^mu.
                let poly = r(2) * big_m.pow(5) + r(3) * big_m.clone() * eta.clone() * rho;
                let first = poly * n * Rational::pow2(mu as i64) * inv_sqrt.clone() / r(3);
                (first + big_m * inv_sqrt) / q.clone()
            }
            ExactRegime::Udr { alpha } => {
                // (δ'·n + 1)/|F| with δ' = (1−ρ)/2 − γ.
                let pp = (r(1) - rho.clone()) / r(2) - self.udr_gap(&rho, alpha);
                (pp * n + r(1)) / q.clone()
            }
        }
    }

    /// Error for coefficients 1, r, …, r^(batch−1): (batch − 1) · error_linear.
    pub fn error_powers(&self, mu: u32, dim: u64, batch: u64, q: &Rational) -> Rational {
        self.error_linear(mu, dim, q) * Rational::int(batch.saturating_sub(1))
    }
}

/// ε · 2^-g · 2^64/p: the verifier accepts a Goldilocks element below 2^(64-g).
pub fn with_grinding(eps: Rational, g: u32) -> Rational {
    eps * Rational::pow2(64 - g as i64) / Rational::int(0xFFFF_FFFF_0000_0001u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_is_exact() {
        assert_eq!(Rational::from_f64(0.5), Rational::new(1, 2));
        assert!(Rational::from_f64(0.1) * Rational::int(10) != Rational::int(1));
    }

    #[test]
    fn security_bits_floor_on_grid() {
        // 2^-10 -> exactly 10 bits; slightly above -> just under.
        assert_eq!(Rational::pow2(-10).security_bits(), Rational::int(10));
        let above = Rational::pow2(-10) * Rational::new(1001, 1000);
        assert!(above.security_bits() < Rational::int(10));
        assert!(above.security_bits() >= Rational::new(79, 8));
    }
}
