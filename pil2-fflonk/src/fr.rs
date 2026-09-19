//! Arithmetic in the BN254 scalar field.
//!
//! `BigUint` rather than a Montgomery representation: the verifier touches a
//! few hundred field elements, so the constant factor is irrelevant next to the
//! pairing, and staying with the type the transcript already uses avoids both a
//! conversion layer and a new dependency. The prover, where the constant factor
//! does matter, stays in C++ over ffiasm.
//!
//! Every operation reduces, so a value of this module's making is always a
//! canonical representative. Inputs parsed from a proof are not -- use
//! [`from_decimal`], which rejects anything out of range, rather than reducing
//! silently: a proof carrying `r + 1` where it means `1` is malformed, and
//! accepting it would let one proof have two encodings.

use anyhow::{Context, Result, bail};
use num_bigint::BigUint;
use num_traits::{One, Num, Zero};

use crate::transcript::fr_modulus;

/// `a + b`
pub fn add(a: &BigUint, b: &BigUint) -> BigUint {
    (a + b) % fr_modulus()
}

/// `a - b`, wrapping rather than underflowing.
pub fn sub(a: &BigUint, b: &BigUint) -> BigUint {
    let r = fr_modulus();
    (a + &r - (b % &r)) % r
}

/// `a * b`
pub fn mul(a: &BigUint, b: &BigUint) -> BigUint {
    (a * b) % fr_modulus()
}

/// `a ^ e`
pub fn pow(a: &BigUint, e: u64) -> BigUint {
    a.modpow(&BigUint::from(e), &fr_modulus())
}

/// `a ^ -1`, by Fermat. Errors on zero rather than returning a wrong answer:
/// every inversion the verifier performs is of a quantity the protocol
/// guarantees to be non-zero, so a zero here means a malformed proof or a
/// challenge collision, both of which must stop verification.
pub fn inv(a: &BigUint) -> Result<BigUint> {
    if a.is_zero() {
        bail!("inverse of zero in Fr");
    }
    let r = fr_modulus();
    Ok(a.modpow(&(&r - BigUint::from(2u32)), &r))
}

/// `a / b`
pub fn div(a: &BigUint, b: &BigUint) -> Result<BigUint> {
    Ok(mul(a, &inv(b)?))
}

/// Parse a canonical decimal field element, as proofs and keys carry them.
pub fn from_decimal(s: &str) -> Result<BigUint> {
    let v = BigUint::from_str_radix(s, 10).with_context(|| format!("parsing field element {s:?}"))?;
    if v >= fr_modulus() {
        bail!("field element {s:?} is not reduced");
    }
    Ok(v)
}

/// `∏ (x - root)`, the zerofier of `roots` evaluated at `x`.
///
/// Repeated roots contribute a factor each. That is deliberate: `computeZT`
/// builds its zerofier from the concatenation of every `f_i`'s roots without
/// deduplicating, and the verifier has to agree with the prover on the
/// polynomial, not merely on where it vanishes.
pub fn zerofier_at(roots: &[BigUint], x: &BigUint) -> BigUint {
    roots.iter().fold(BigUint::one(), |acc, root| mul(&acc, &sub(x, root)))
}

/// Interpolate the points `(xs[i], ys[i])` and evaluate the result at `x`.
///
/// Lagrange in barycentric-free form: the verifier only ever needs the value at
/// the challenge, never the coefficients, so interpolating explicitly would be
/// wasted work. `xs` must be distinct -- within one `f_i` they are distinct
/// powers of a root of unity, so a collision means a malformed key.
pub fn lagrange_eval(xs: &[BigUint], ys: &[BigUint], x: &BigUint) -> Result<BigUint> {
    if xs.len() != ys.len() {
        bail!("interpolation has {} points but {} values", xs.len(), ys.len());
    }

    let mut acc = BigUint::zero();
    for (i, xi) in xs.iter().enumerate() {
        // An exact hit needs no interpolation, and would divide by zero below.
        if xi == x {
            return Ok(ys[i].clone());
        }

        let mut term = ys[i].clone();
        for (j, xj) in xs.iter().enumerate() {
            if i == j {
                continue;
            }
            if xi == xj {
                bail!("interpolation points {i} and {j} coincide");
            }
            term = mul(&term, &div(&sub(x, xj), &sub(xi, xj))?);
        }
        acc = add(&acc, &term);
    }

    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(v: u64) -> BigUint {
        BigUint::from(v)
    }

    /// Evaluate a polynomial given by ascending coefficients, by Horner. Only
    /// used to generate interpolation vectors; the verifier never holds a
    /// polynomial in coefficient form.
    fn eval(coefficients: &[BigUint], x: &BigUint) -> BigUint {
        let mut acc = BigUint::zero();
        for c in coefficients.iter().rev() {
            acc = add(&mul(&acc, x), c);
        }
        acc
    }

    #[test]
    fn sub_wraps_instead_of_underflowing() {
        assert_eq!(sub(&n(3), &n(5)), &fr_modulus() - BigUint::from(2u32));
        assert_eq!(add(&sub(&n(3), &n(5)), &n(2)), BigUint::zero());
    }

    #[test]
    fn operations_reduce() {
        let big = &fr_modulus() - BigUint::one();
        assert!(add(&big, &big) < fr_modulus());
        assert!(mul(&big, &big) < fr_modulus());
        assert_eq!(add(&big, &BigUint::one()), BigUint::zero());
    }

    #[test]
    fn inv_round_trips() {
        for v in [1u64, 2, 3, 1 << 40] {
            assert_eq!(mul(&n(v), &inv(&n(v)).unwrap()), BigUint::one());
        }
    }

    #[test]
    fn inv_of_zero_is_an_error() {
        assert!(inv(&BigUint::zero()).is_err());
        assert!(div(&n(1), &BigUint::zero()).is_err());
    }

    /// An unreduced element is rejected rather than folded, so a proof cannot
    /// carry the same value two ways.
    #[test]
    fn from_decimal_rejects_an_unreduced_element() {
        assert!(from_decimal("0").is_ok());
        assert!(from_decimal(&(fr_modulus() - BigUint::one()).to_str_radix(10)).is_ok());
        assert!(from_decimal(&fr_modulus().to_str_radix(10)).is_err());
        assert!(from_decimal(&(fr_modulus() + BigUint::one()).to_str_radix(10)).is_err());
        assert!(from_decimal("-1").is_err());
        assert!(from_decimal("").is_err());
    }

    #[test]
    fn eval_is_horner() {
        // 1 + 2x + 3x^2 at x = 10 is 321.
        let p = [n(1), n(2), n(3)];
        assert_eq!(eval(&p, &n(10)), n(321));
        assert_eq!(eval(&[], &n(10)), BigUint::zero());
        assert_eq!(eval(&[n(7)], &n(10)), n(7));
    }

    #[test]
    fn zerofier_vanishes_exactly_on_its_roots() {
        let roots = [n(1), n(2), n(3)];
        for r in &roots {
            assert!(zerofier_at(&roots, r).is_zero());
        }
        assert!(!zerofier_at(&roots, &n(4)).is_zero());
        // (4-1)(4-2)(4-3) = 6
        assert_eq!(zerofier_at(&roots, &n(4)), n(6));
        assert_eq!(zerofier_at(&[], &n(4)), BigUint::one());
    }

    /// A repeated root contributes twice. The prover's ZT is built over a
    /// multiset, so a verifier that deduplicated would evaluate a different
    /// polynomial and reject every honest proof.
    #[test]
    fn zerofier_counts_repeated_roots() {
        assert_eq!(zerofier_at(&[n(2), n(2)], &n(5)), n(9));
        assert_ne!(zerofier_at(&[n(2), n(2)], &n(5)), zerofier_at(&[n(2)], &n(5)));
    }

    #[test]
    fn lagrange_reproduces_its_own_points() {
        let xs = [n(1), n(2), n(3)];
        let ys = [n(10), n(20), n(35)];
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert_eq!(&lagrange_eval(&xs, &ys, x).unwrap(), y);
        }
    }

    /// The interpolant of points taken from a known polynomial must be that
    /// polynomial: 3x^2 + 2x + 1 sampled at three points, checked at a fourth.
    #[test]
    fn lagrange_recovers_a_known_polynomial() {
        let p = [n(1), n(2), n(3)];
        let xs = [n(5), n(6), n(7)];
        let ys: Vec<BigUint> = xs.iter().map(|x| eval(&p, x)).collect();
        assert_eq!(lagrange_eval(&xs, &ys, &n(11)).unwrap(), eval(&p, &n(11)));
    }

    #[test]
    fn lagrange_rejects_mismatched_or_repeated_points() {
        assert!(lagrange_eval(&[n(1), n(2)], &[n(1)], &n(3)).is_err());
        assert!(lagrange_eval(&[n(1), n(1)], &[n(1), n(2)], &n(3)).is_err());
    }
}
