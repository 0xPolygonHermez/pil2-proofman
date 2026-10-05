//! `Fr` arithmetic with `num-bigint`, for the oracle only: slow, and simple enough to trust.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::BigUint;

use crate::error::{invalid, PilfflonkResult};
use crate::field::{r, FrBytes};
use crate::global_info::MAX_NBITS;

/// An element of `Fr`, always reduced (`< r`).
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Fr(BigUint);

impl Fr {
    pub fn zero() -> Self {
        Self(BigUint::default())
    }

    pub fn one() -> Self {
        Self(BigUint::from(1u32))
    }

    pub fn from_u64(value: u64) -> Self {
        Self(BigUint::from(value) % r())
    }

    /// `value mod r`.
    pub fn reduce(value: &BigUint) -> Self {
        Self(value % r())
    }

    /// A value the pilout writes: big-endian bytes, the empty string for 0. Refuses a value that
    /// is not below `r`: a pilout over BN128 has none, and reducing it would hide a compiler bug.
    pub fn from_pilout_bytes(bytes: &[u8]) -> PilfflonkResult<Self> {
        let value = BigUint::from_bytes_be(bytes);
        if value < *r() {
            Ok(Self(value))
        } else {
            invalid!("the pilout has {value}, which is not below r")
        }
    }

    pub fn to_bytes(&self) -> FrBytes {
        let digits = self.0.to_bytes_le();
        let mut bytes = [0u8; 32];
        bytes[..digits.len()].copy_from_slice(&digits);
        // Reduced, so below r.
        FrBytes::from_le_bytes(bytes).unwrap_or_default()
    }

    pub fn as_biguint(&self) -> &BigUint {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0 == BigUint::default()
    }

    pub fn pow(&self, exponent: &BigUint) -> Self {
        Self(self.0.modpow(exponent, r()))
    }

    pub fn pow_u64(&self, exponent: u64) -> Self {
        self.pow(&BigUint::from(exponent))
    }

    /// `1/self`, by Fermat: `self^(r - 2)`.
    pub fn inv(&self) -> PilfflonkResult<Self> {
        if self.is_zero() {
            return invalid!("0 has no inverse");
        }
        Ok(self.pow(&(r() - 2u32)))
    }
}

impl From<FrBytes> for Fr {
    fn from(value: FrBytes) -> Self {
        Self(BigUint::from_bytes_le(&value.to_le_bytes()))
    }
}

impl From<&FrBytes> for Fr {
    fn from(value: &FrBytes) -> Self {
        Self::from(*value)
    }
}

impl Add for &Fr {
    type Output = Fr;

    fn add(self, rhs: &Fr) -> Fr {
        Fr((&self.0 + &rhs.0) % r())
    }
}

impl Sub for &Fr {
    type Output = Fr;

    fn sub(self, rhs: &Fr) -> Fr {
        Fr((&self.0 + r() - &rhs.0) % r())
    }
}

impl Mul for &Fr {
    type Output = Fr;

    fn mul(self, rhs: &Fr) -> Fr {
        Fr((&self.0 * &rhs.0) % r())
    }
}

impl Neg for &Fr {
    type Output = Fr;

    fn neg(self) -> Fr {
        &Fr::zero() - self
    }
}

impl fmt::Display for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fr({})", self.0)
    }
}

/// `ω_N = 5^((r - 1)/N)` for `N = 2^n_bits`: the generator of the `H` of ffiasm's FFT and of
/// ffjavascript's `Fr.w[n_bits]`, 5 being the smallest quadratic non-residue mod `r`. Row `j` of a
/// trace is the point `ω_N^j`.
pub fn omega(n_bits: u32) -> PilfflonkResult<Fr> {
    if u64::from(n_bits) > MAX_NBITS {
        return invalid!("there is no root of unity of order 2^{n_bits}: r - 1 = 2^{MAX_NBITS} · odd");
    }
    Ok(Fr::from_u64(5).pow(&((r() - 1u32) >> n_bits)))
}

/// The inverses of `values`, with one inversion (Montgomery's trick). Refuses a 0.
pub fn batch_inverse(values: &[Fr]) -> PilfflonkResult<Vec<Fr>> {
    let mut prefix = Vec::with_capacity(values.len());
    let mut acc = Fr::one();
    for v in values {
        prefix.push(acc.clone());
        acc = &acc * v;
    }
    let mut inv = acc.inv()?;
    let mut out = vec![Fr::zero(); values.len()];
    for (i, v) in values.iter().enumerate().rev() {
        out[i] = &inv * &prefix[i];
        inv = &inv * v;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ffjavascript 0.3.1's `Fr.w[8]` and `Fr.w[28]` for BN128 (`buildBn128`), printed with
    /// `Fr.toString`.
    const W8: &str = "3478517300119284901893091970156912948790432420133812234316178878452092729974";
    const W28: &str = "19103219067921713944291392827692070036145651957329286315305642004821462161904";

    fn dec(s: &str) -> Fr {
        Fr::reduce(&BigUint::parse_bytes(s.as_bytes(), 10).unwrap())
    }

    #[test]
    fn omega_is_ffjavascripts_root_of_unity() {
        assert_eq!(omega(8).unwrap(), dec(W8));
        assert_eq!(omega(28).unwrap(), dec(W28));
        assert!(omega(29).is_err());
        for n_bits in [1, 4, 8] {
            let w = omega(n_bits).unwrap();
            let n = 1u64 << n_bits;
            assert_eq!(w.pow_u64(n), Fr::one());
            assert_eq!(w.pow_u64(n / 2), -&Fr::one(), "primitive: ω^(N/2) = -1");
        }
        assert_eq!(omega(0).unwrap(), Fr::one());
    }

    #[test]
    fn arithmetic_is_mod_r() {
        let r_minus_1 = -&Fr::one();
        assert_eq!(&r_minus_1 + &Fr::one(), Fr::zero());
        assert_eq!(&Fr::zero() - &Fr::one(), r_minus_1);
        assert_eq!(&r_minus_1 * &r_minus_1, Fr::one());
        let x = Fr::from_u64(12345);
        assert_eq!(&x * &x.inv().unwrap(), Fr::one());
        assert!(Fr::zero().inv().is_err());
        assert_eq!(Fr::from(x.to_bytes()), x);
        assert_eq!(r_minus_1.to_bytes().to_decimal(), (r() - 1u32).to_string());
    }

    #[test]
    fn pilout_values_are_big_endian_and_below_r() {
        assert_eq!(Fr::from_pilout_bytes(&[]).unwrap(), Fr::zero());
        assert_eq!(Fr::from_pilout_bytes(&[1, 0]).unwrap(), Fr::from_u64(256));
        assert!(Fr::from_pilout_bytes(&r().to_bytes_be()).is_err());
        assert_eq!(Fr::from_pilout_bytes(&(r() - 1u32).to_bytes_be()).unwrap(), -&Fr::one());
    }

    #[test]
    fn batch_inverse_inverts_each() {
        let values: Vec<Fr> = (1..10).map(Fr::from_u64).collect();
        let inverses = batch_inverse(&values).unwrap();
        for (v, i) in values.iter().zip(&inverses) {
            assert_eq!(v * i, Fr::one());
        }
        assert!(batch_inverse(&[Fr::one(), Fr::zero()]).is_err());
        assert!(batch_inverse(&[]).unwrap().is_empty());
    }
}
