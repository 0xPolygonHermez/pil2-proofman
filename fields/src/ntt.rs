//! Forward NTT, coset LDE and batch inversion over Goldilocks.

use alloc::vec;
use alloc::vec::Vec;

use crate::{Field, Goldilocks};

fn bit_reverse(mut x: u32, bits: usize) -> u32 {
    x = ((x >> 1) & 0x5555_5555) | ((x & 0x5555_5555) << 1);
    x = ((x >> 2) & 0x3333_3333) | ((x & 0x3333_3333) << 2);
    x = ((x >> 4) & 0x0F0F_0F0F) | ((x & 0x0F0F_0F0F) << 4);
    x = ((x >> 8) & 0x00FF_00FF) | ((x & 0x00FF_00FF) << 8);
    x = x.rotate_left(16);
    x >> (32 - bits)
}

/// In-place forward NTT of `n_cols` interleaved columns of length `1 << n_bits`.
///
/// Input: coefficients in natural order. Output: evaluations in natural order,
/// `out[i] = p(w^i)` with `w = Goldilocks::W[n_bits]`. Inverse of
/// [`intt_tiny`](crate::intt_tiny).
pub fn ntt_tiny(data: &mut [Goldilocks], n_bits: usize, n_cols: usize) {
    let n = 1 << n_bits;
    debug_assert_eq!(data.len(), n * n_cols);

    let mut vals = vec![Goldilocks::ZERO; n * n_cols];
    for i in 0..n {
        let r = bit_reverse(i as u32, n_bits) as usize;
        for c in 0..n_cols {
            vals[r * n_cols + c] = data[i * n_cols + c];
        }
    }

    for stage in 0..n_bits {
        let m = 1 << (stage + 1);
        let half_m = m >> 1;
        let omega = Goldilocks::new(Goldilocks::W[stage + 1]);
        let mut twiddles = Vec::with_capacity(half_m);
        twiddles.push(Goldilocks::ONE);
        for j in 1..half_m {
            twiddles.push(twiddles[j - 1] * omega);
        }

        for k in (0..n).step_by(m) {
            for (j, w) in twiddles.iter().enumerate().take(half_m) {
                for c in 0..n_cols {
                    let idx1 = (k + j) * n_cols + c;
                    let idx2 = (k + j + half_m) * n_cols + c;
                    let u = vals[idx1];
                    let t = vals[idx2] * *w;
                    vals[idx1] = u + t;
                    vals[idx2] = u - t;
                }
            }
        }
    }

    data.copy_from_slice(&vals);
}

// TODO: IT IS NECESSARY TO GO TO A COSET?????

/// Low-degree extension onto a multiplicative coset.
///
/// Takes `n_cols` interleaved columns of `1 << n_bits` coefficients (natural
/// order) and returns their evaluations over the coset
/// `SHIFT * H_ext`, where `H_ext` has size `1 << (n_bits + log_blowup)`:
/// `out[i * n_cols + c] = p_c(SHIFT * w_ext^i)`.
pub fn coset_lde(coeffs: &[Goldilocks], n_bits: usize, log_blowup: usize, n_cols: usize) -> Vec<Goldilocks> {
    let n = 1 << n_bits;
    let n_ext = n << log_blowup;
    debug_assert_eq!(coeffs.len(), n * n_cols);

    let shift = Goldilocks::new(Goldilocks::SHIFT);
    let mut vals = vec![Goldilocks::ZERO; n_ext * n_cols];
    // Scale coefficient j by SHIFT^j so the plain NTT evaluates on the coset.
    let mut shift_pow = Goldilocks::ONE;
    for i in 0..n {
        for c in 0..n_cols {
            vals[i * n_cols + c] = coeffs[i * n_cols + c] * shift_pow;
        }
        shift_pow *= shift;
    }

    ntt_tiny(&mut vals, n_bits + log_blowup, n_cols);
    vals
}

/// Batch inversion via Montgomery's trick: one field inversion plus `3(n-1)`
/// multiplications. Panics in debug mode if any element is zero.
pub fn batch_inverse<F: Field>(values: &[F]) -> Vec<F> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut prefix = Vec::with_capacity(values.len());
    let mut acc = F::ONE;
    for v in values {
        debug_assert!(!v.is_zero(), "batch_inverse: zero element");
        prefix.push(acc);
        acc *= *v;
    }

    let mut inv = acc.inverse();
    let mut result = vec![F::ZERO; values.len()];
    for i in (0..values.len()).rev() {
        result[i] = prefix[i] * inv;
        inv *= values[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{intt_tiny, PrimeField64};
    use rand::{rng, RngExt};

    fn random_vals(n: usize) -> Vec<Goldilocks> {
        let mut r = rng();
        (0..n).map(|_| Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64)).collect()
    }

    /// Horner evaluation of a single-column coefficient vector.
    fn eval_naive(coeffs: &[Goldilocks], x: Goldilocks) -> Goldilocks {
        coeffs.iter().rev().fold(Goldilocks::ZERO, |acc, c| acc * x + *c)
    }

    #[test]
    fn ntt_roundtrip() {
        for n_bits in [1, 3, 6] {
            for n_cols in [1, 3] {
                let coeffs = random_vals((1 << n_bits) * n_cols);
                let mut data = coeffs.clone();
                ntt_tiny(&mut data, n_bits, n_cols);
                intt_tiny(&mut data, n_bits, n_cols);
                assert_eq!(data, coeffs);
            }
        }
    }

    #[test]
    fn ntt_matches_naive_evaluation() {
        let n_bits = 4;
        let n = 1 << n_bits;
        let coeffs = random_vals(n);
        let mut evals = coeffs.clone();
        ntt_tiny(&mut evals, n_bits, 1);

        let w = Goldilocks::new(Goldilocks::W[n_bits]);
        let mut x = Goldilocks::ONE;
        for i in 0..n {
            assert_eq!(evals[i], eval_naive(&coeffs, x), "mismatch at index {i}");
            x *= w;
        }
    }

    #[test]
    fn coset_lde_matches_naive_evaluation() {
        let n_bits = 3;
        let log_blowup = 2;
        let n = 1 << n_bits;
        let n_ext = n << log_blowup;
        let n_cols = 2;
        let coeffs = random_vals(n * n_cols);
        let lde = coset_lde(&coeffs, n_bits, log_blowup, n_cols);

        let w_ext = Goldilocks::new(Goldilocks::W[n_bits + log_blowup]);
        let shift = Goldilocks::new(Goldilocks::SHIFT);
        for c in 0..n_cols {
            let col: Vec<Goldilocks> = (0..n).map(|i| coeffs[i * n_cols + c]).collect();
            let mut x = shift;
            for i in 0..n_ext {
                assert_eq!(lde[i * n_cols + c], eval_naive(&col, x), "mismatch at row {i}, col {c}");
                x *= w_ext;
            }
        }
    }

    #[test]
    fn batch_inverse_matches_inverse() {
        let vals = random_vals(17);
        let invs = batch_inverse(&vals);
        for (v, inv) in vals.iter().zip(invs.iter()) {
            assert_eq!(*v * *inv, Goldilocks::ONE);
        }
        assert!(batch_inverse::<Goldilocks>(&[]).is_empty());
    }
}
