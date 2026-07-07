//! Reed–Solomon encoding.

use crate::hypercube::{values_to_coeffs, Ext};
use fields::{coset_lde, Field, Goldilocks};

/// RS-encode one base-field column (given as hypercube values). Returns the
/// `2^(n_bits + log_blowup)` codeword values in natural evaluation order.
pub fn encode_column(col: &[Goldilocks], log_blowup: usize) -> Vec<Goldilocks> {
    assert!(col.len().is_power_of_two());
    let n_bits = col.len().trailing_zeros() as usize;
    let mut coeffs = col.to_vec();
    values_to_coeffs(&mut coeffs);
    coset_lde(&coeffs, n_bits, log_blowup, 1)
}

/// Evaluate the univariate polynomial with extension-field coefficients
/// `coeffs` at a base-field point `x`.
pub fn eval_ext_poly_at_base(coeffs: &[Ext], x: Goldilocks) -> Ext {
    // Use the Horner method
    let mut acc = Ext::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

/// The `j`-th point of the level-`level` folding domain: `shift^(2^level) · w^j`
/// where `w` generates the group of size `2^bits` (`bits` = current domain bits).
pub fn domain_point(n0_bits: usize, level: usize, j: u64) -> Goldilocks {
    let bits = n0_bits - level;
    let shift = Goldilocks::new(Goldilocks::SHIFT).exp_power_of_2(level);
    let w = Goldilocks::new(Goldilocks::W[bits]);
    shift * w.exp_u64(j)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::{fold_coeffs, fold_mle, to_ext_vec, values_to_coeffs};
    use fields::PrimeField64;
    use rand::{rng, RngExt};

    fn random_col(n: usize) -> Vec<Goldilocks> {
        let mut r = rng();
        (0..n).map(|_| Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64)).collect()
    }

    fn random_ext() -> Ext {
        let c = random_col(3);
        Ext::from_array(&c)
    }

    /// One FRI fold of codeword values must equal the codeword of the
    /// first-variable-bound multilinear — the identity the whole Basefold
    /// construction rests on. Checked both in coefficient form (`fold_coeffs`)
    /// and value form (`fold_mle` + Möbius).
    #[test]
    fn codeword_fold_is_coefficient_fold() {
        let n_bits = 4;
        let log_blowup = 2;
        let n0_bits = n_bits + log_blowup;
        let col = random_col(1 << n_bits);
        let codeword = to_ext_vec(&encode_column(&col, log_blowup));
        let r = random_ext();

        // Fold codeword values: pair (j, j+half), x = domain_point(level 0, j)
        let n0 = 1usize << n0_bits;
        let half = n0 / 2;
        let two_inv = Goldilocks::TWO.inverse();
        let folded_values: Vec<Ext> = (0..half)
            .map(|j| {
                let x_inv = domain_point(n0_bits, 0, j as u64).inverse();
                let a = codeword[j];
                let b = codeword[j + half];
                (a + b) * two_inv + r * ((a - b) * (two_inv * x_inv))
            })
            .collect();

        // Bind X_1 = r in coefficient form and evaluate naively on the level-1 domain.
        let mut coeffs = to_ext_vec(&col);
        values_to_coeffs(&mut coeffs);
        fold_coeffs(&mut coeffs, r);
        for (j, folded) in folded_values.iter().enumerate() {
            let x = domain_point(n0_bits, 1, j as u64);
            assert_eq!(*folded, eval_ext_poly_at_base(&coeffs, x), "position {j}");
        }

        // Cross-check: the same coefficients arise from fold_mle + Möbius.
        let mut vals = to_ext_vec(&col);
        fold_mle(&mut vals, r);
        values_to_coeffs(&mut vals);
        assert_eq!(vals, coeffs);
    }
}
