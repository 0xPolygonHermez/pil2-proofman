//! Multilinear-extension (MLE) utilities over the Boolean hypercube.
//!
//! Convention used throughout the crate: a table `t` of length `2^n` represents
//! the multilinear `t̃(X_1, …, X_n)` with `t̃(b) = t[index(b)]` where **variable
//! `X_1` is the least-significant bit** of the index.

use fields::{CubicExtensionField, Field, Goldilocks};

/// The challenge/folding field: cubic extension of Goldilocks (~192 bits).
pub type Ext = CubicExtensionField<Goldilocks>;

#[inline]
pub fn ext_from_base(v: Goldilocks) -> Ext {
    Ext { value: [v, Goldilocks::ZERO, Goldilocks::ZERO] }
}

pub fn to_ext_vec(vals: &[Goldilocks]) -> Vec<Ext> {
    vals.iter().map(|&v| ext_from_base(v)).collect()
}

/// Bind the first (least-significant) variable of an MLE table to `r`,
/// halving it in place: `t'[i] = t[2i] + r·(t[2i+1] − t[2i])`.
pub fn fold_mle(evals: &mut Vec<Ext>, r: Ext) {
    debug_assert!(evals.len().is_power_of_two() && evals.len() >= 2);
    let half = evals.len() / 2;
    for i in 0..half {
        let e0 = evals[2 * i];
        let e1 = evals[2 * i + 1];
        evals[i] = e0 + (e1 - e0) * r;
    }
    evals.truncate(half);
}

/// Evaluate the MLE of `evals` at `point` (LSB-first variable order).
pub fn mle_eval(evals: &[Ext], point: &[Ext]) -> Ext {
    assert_eq!(evals.len(), 1usize << point.len());
    let mut t = evals.to_vec();
    for &r in point {
        fold_mle(&mut t, r);
    }
    t[0]
}

/// Evaluate the MLE of a base-field table at an extension point.
pub fn mle_eval_base(evals: &[Goldilocks], point: &[Ext]) -> Ext {
    mle_eval(&to_ext_vec(evals), point)
}

/// In-place Möbius transform: hypercube evaluations → multilinear monomial
/// coefficients. After this, `t[i]` is the coefficient of `Π_{j: bit_j(i)=1} X_j`.
pub fn values_to_coeffs<T>(vals: &mut [T])
where
    T: Copy + core::ops::SubAssign,
{
    let n = vals.len();
    debug_assert!(n.is_power_of_two());
    let mut step = 1;
    while step < n {
        let mut base = 0;
        while base < n {
            for i in base..base + step {
                let hi = vals[i];
                vals[i + step] -= hi;
            }
            base += 2 * step;
        }
        step <<= 1;
    }
}

/// Inverse of [`values_to_coeffs`]: monomial coefficients → hypercube evaluations.
pub fn coeffs_to_values<T>(coeffs: &mut [T])
where
    T: Copy + core::ops::AddAssign,
{
    let n = coeffs.len();
    debug_assert!(n.is_power_of_two());
    let mut step = 1;
    while step < n {
        let mut base = 0;
        while base < n {
            for i in base..base + step {
                let lo = coeffs[i];
                coeffs[i + step] += lo;
            }
            base += 2 * step;
        }
        step <<= 1;
    }
}

/// Bind the first variable of a multilinear given in *coefficient* form:
/// `c'[t] = c[2t] + r·c[2t+1]` (the coefficient-side counterpart of
/// [`fold_mle`]; also exactly what one FRI fold does to the codeword).
pub fn fold_coeffs(coeffs: &mut Vec<Ext>, r: Ext) {
    debug_assert!(coeffs.len().is_power_of_two() && coeffs.len() >= 2);
    let half = coeffs.len() / 2;
    for i in 0..half {
        coeffs[i] = coeffs[2 * i] + r * coeffs[2 * i + 1];
    }
    coeffs.truncate(half);
}

/// Evaluate a multilinear given in *coefficient* form at `point`
/// (LSB-first): `Σ_i c[i] · Π_{j: bit_j(i)=1} point[j]`.
pub fn monomial_eval(coeffs: &[Ext], point: &[Ext]) -> Ext {
    assert_eq!(coeffs.len(), 1usize << point.len());
    let mut t = coeffs.to_vec();
    for &r in point {
        fold_coeffs(&mut t, r);
    }
    t[0]
}

/// Inner product `Σ_i base[i] · table[i]` of a base-field column with an
/// extension-field kernel table. This is how the prover computes claimed
/// weighted-sum values `v = Σ_b w(b)·K(b)`.
pub fn dot_base_ext(base: &[Goldilocks], table: &[Ext]) -> Ext {
    debug_assert_eq!(base.len(), table.len());
    let mut acc = Ext::zero();
    for (b, t) in base.iter().zip(table.iter()) {
        acc += *t * *b;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq::eq_evals;
    use rand::{rng, RngExt};

    pub(crate) fn random_ext() -> Ext {
        let mut r = rng();
        Ext::from_array(&[
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
        ])
    }

    use fields::PrimeField64;

    #[test]
    fn mle_eval_agrees_with_eq_inner_product() {
        let n = 5;
        let evals: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        let point: Vec<Ext> = (0..n).map(|_| random_ext()).collect();

        let direct = mle_eval(&evals, &point);
        let eq = eq_evals(&point);
        let via_eq: Ext = evals.iter().zip(eq.iter()).map(|(a, b)| *a * *b).sum();
        assert_eq!(direct, via_eq);
    }

    #[test]
    fn mobius_roundtrip_and_monomial_eval() {
        let n = 5;
        let evals: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        let point: Vec<Ext> = (0..n).map(|_| random_ext()).collect();

        let mut coeffs = evals.clone();
        values_to_coeffs(&mut coeffs);

        // Monomial-basis evaluation of the coefficients == MLE of the values.
        assert_eq!(monomial_eval(&coeffs, &point), mle_eval(&evals, &point));

        // fold_coeffs and fold_mle both bind X_1: they must commute with Möbius.
        let r = random_ext();
        let mut folded_coeffs = coeffs.clone();
        fold_coeffs(&mut folded_coeffs, r);
        let mut folded_vals = evals.clone();
        fold_mle(&mut folded_vals, r);
        let mut expect = folded_vals.clone();
        values_to_coeffs(&mut expect);
        assert_eq!(folded_coeffs, expect);

        // Roundtrip
        let mut back = coeffs.clone();
        coeffs_to_values(&mut back);
        assert_eq!(back, evals);
    }

    #[test]
    fn mle_eval_interpolates_hypercube() {
        // At a Boolean point the MLE must return the table entry.
        let n = 4;
        let evals: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        for idx in [0usize, 1, 7, 15] {
            let point: Vec<Ext> = (0..n).map(|j| if (idx >> j) & 1 == 1 { Ext::one() } else { Ext::zero() }).collect();
            assert_eq!(mle_eval(&evals, &point), evals[idx]);
        }
    }
}
