//! Reed–Solomon encoding.

use crate::hypercube::Ext;
use fields::{coset_lde, Field, Goldilocks};

/// RS-encode one base-field column (given as hypercube values). Returns the
/// `2^(n_bits + log_blowup)` codeword values in natural evaluation order.
///
/// Uses the **value-to-coefficient identification**: the
/// hypercube values are fed directly to the RS encoder as the coefficients of
/// the univariate `g(x) = Σ_i f̂_MLE(bin(i))·xⁱ`.
pub fn encode_column(col: &[Goldilocks], log_blowup: usize) -> Vec<Goldilocks> {
    assert!(col.len().is_power_of_two());
    let n_bits = col.len().trailing_zeros() as usize;
    coset_lde(col, n_bits, log_blowup, 1)
}

/// RS-encode a whole matrix of equal-length base-field columns, returning one
/// codeword per column, via a single batched call into the C++ AVX/threaded NTT.
///
/// Interleaves the columns row-major into canonical `u64`, runs the C++ coset
/// NTT once (`num_cols` batched), then de-interleaves the extended codewords.
pub fn encode_columns(columns: &[&[Goldilocks]], log_blowup: usize) -> Vec<Vec<Goldilocks>> {
    use fields::PrimeField64;
    assert!(!columns.is_empty());
    let n = columns[0].len();
    assert!(n.is_power_of_two() && columns.iter().all(|c| c.len() == n));
    let ncols = columns.len();
    let n_ext = n << log_blowup;

    let mut input = vec![0u64; n * ncols];
    for (c, col) in columns.iter().enumerate() {
        for (j, &v) in col.iter().enumerate() {
            input[j * ncols + c] = v.as_canonical_u64();
        }
    }
    let mut out = vec![0u64; n_ext * ncols];
    proofman_starks_lib_c::ntt_coset_lde_c(out.as_mut_ptr(), input.as_ptr(), ncols as u64, n as u64, n_ext as u64);

    (0..ncols).map(|c| (0..n_ext).map(|i| Goldilocks::new(out[i * ncols + c])).collect()).collect()
}

/// RS-encode an **extension-field** column (given as `Ext` hypercube values /
/// univariate coefficients). Value-to-coefficient coset LDE is `F`-linear and
/// `Ext = F³`, so this is the limb-wise base-field encode recombined: identical
/// to encoding the `Ext` polynomial directly. Used by WHIR's STIR re-encode of
/// a folded (extension) oracle.
pub fn encode_column_ext(vals: &[Ext], log_blowup: usize) -> Vec<Ext> {
    let limb0: Vec<Goldilocks> = vals.iter().map(|v| v.value[0]).collect();
    let limb1: Vec<Goldilocks> = vals.iter().map(|v| v.value[1]).collect();
    let limb2: Vec<Goldilocks> = vals.iter().map(|v| v.value[2]).collect();
    let enc = encode_columns(&[&limb0, &limb1, &limb2], log_blowup);
    let n_ext = enc[0].len();
    (0..n_ext).map(|i| Ext::from_array(&[enc[0][i], enc[1][i], enc[2][i]])).collect()
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
    use crate::hypercube::fold_mle;
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

    /// The FFI batched coset LDE must be byte-identical to the pure-Rust
    /// per-column `encode_column` — the codeword layout the verifier's
    /// `domain_point`/`fold_codeword`/query phase all assume.
    #[test]
    fn ffi_ntt_matches_coset_lde() {
        for n_bits in [3usize, 6, 10] {
            for log_blowup in [1usize, 2] {
                let cols: Vec<Vec<Goldilocks>> = (0..5).map(|_| random_col(1 << n_bits)).collect();
                let refs: Vec<&[Goldilocks]> = cols.iter().map(|c| c.as_slice()).collect();
                let ffi = encode_columns(&refs, log_blowup);
                for (c, col) in cols.iter().enumerate() {
                    assert_eq!(ffi[c], encode_column(col, log_blowup), "n_bits={n_bits} blowup={log_blowup} col={c}");
                }
            }
        }
    }

    /// `encode_column_ext` must equal the limb-wise base-field encode: for a
    /// base column lifted to `Ext`, it equals `encode_column` lifted to `Ext`.
    #[test]
    fn encode_column_ext_is_limbwise() {
        for n_bits in [3usize, 6] {
            for log_blowup in [1usize, 2] {
                let cols: [Vec<Goldilocks>; 3] =
                    [random_col(1 << n_bits), random_col(1 << n_bits), random_col(1 << n_bits)];
                let vals: Vec<Ext> =
                    (0..(1 << n_bits)).map(|i| Ext::from_array(&[cols[0][i], cols[1][i], cols[2][i]])).collect();
                let ext_enc = encode_column_ext(&vals, log_blowup);
                let e0 = encode_column(&cols[0], log_blowup);
                let e1 = encode_column(&cols[1], log_blowup);
                let e2 = encode_column(&cols[2], log_blowup);
                for i in 0..ext_enc.len() {
                    assert_eq!(ext_enc[i], Ext::from_array(&[e0[i], e1[i], e2[i]]), "n={n_bits} bl={log_blowup} i={i}");
                }
            }
        }
    }

    /// One FRI fold of the codeword must equal the codeword of the
    /// first-variable-bound multilinear.
    #[test]
    fn codeword_fold_is_mle_fold() {
        let n_bits = 4;
        let log_blowup = 2;
        let n0_bits = n_bits + log_blowup;
        let col = random_col(1 << n_bits);
        let codeword = Ext::from_base_batch(&encode_column(&col, log_blowup));
        let r = random_ext();

        // Fold codeword values: pair (j, j+half), x = domain_point(level 0, j),
        // using the value-to-coefficient fold `(1−r)·even + r·odd`.
        let n0 = 1usize << n0_bits;
        let half = n0 / 2;
        let two_inv = Goldilocks::TWO.inverse();
        let folded_values: Vec<Ext> = (0..half)
            .map(|j| {
                let x_inv = domain_point(n0_bits, 0, j as u64).inverse();
                let a = codeword[j];
                let b = codeword[j + half];
                (Ext::ONE - r) * ((a + b) * two_inv) + r * ((a - b) * (two_inv * x_inv))
            })
            .collect();

        // Bind X_1 = r on the values; those bound values are directly the
        // coefficients of the level-1 codeword's univariate.
        let mut vals = Ext::from_base_batch(&col);
        fold_mle(&mut vals, r);
        for (j, folded) in folded_values.iter().enumerate() {
            let x = domain_point(n0_bits, 1, j as u64);
            assert_eq!(*folded, eval_ext_poly_at_base(&vals, x), "position {j}");
        }
    }
}
