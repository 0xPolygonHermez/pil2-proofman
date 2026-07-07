//! The equality kernel `eq` and the rotation kernel `rot_s`.
//!
//! `eq(x, y) = Π_j (x_j·y_j + (1−x_j)(1−y_j))` is the Lagrange kernel of the
//! hypercube.
//!
//! The rotation kernel handles "arbitrary offset" accesses, the
//! shifted column `w^{→s}(b) = w((b + s) mod 2^n)` satisfies
//!
//! `w̃^{→s}(λ) = Σ_y w((y + s) mod 2^n) · eq(y, λ) = Σ_y w(y) · K_s(y)`,
//!
//! with `K_s(y) = eq((y − s) mod 2^n, λ)`,
//! so a shifted-column evaluation claim is a weighted-sum claim on the *base*
//! column with the `eq(·, λ)` table cyclically rotated.

use crate::hypercube::Ext;

/// Evaluates the equality kernel on points `a` and `b`: `eq(a,b)`.
pub fn eq_eval(a: &[Ext], b: &[Ext]) -> Ext {
    debug_assert_eq!(a.len(), b.len());

    // eq(a, b) = Π_j (a_j·b_j + (1−a_j)(1−b_j))
    let one = Ext::ONE;
    let mut acc = one;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc *= x * y + (one - x) * (one - y);
    }
    acc
}

/// Tensor expansion of the equality kernel: returns `t[i] = eq(bits(i), point)`
/// for all `i ∈ [0, 2^n)`.
pub fn eq_evals(point: &[Ext]) -> Vec<Ext> {
    let n = point.len();
    let mut t = Vec::with_capacity(1 << n);
    t.push(Ext::ONE);
    for (j, &r) in point.iter().enumerate() {
        let len = 1 << j;
        t.resize(2 * len, Ext::ZERO);
        for i in (0..len).rev() {
            let v = t[i];
            let hi = v * r;
            t[i + len] = hi;
            t[i] = v - hi;
        }
    }
    t
}

/// Cyclically rotate a kernel table by offset `s`.
pub fn rotate_table<T: Copy>(table: &[T], s: i64) -> Vec<T> {
    let n = table.len();
    debug_assert!(n.is_power_of_two());

    // out[y] = table[(y − s) mod len]
    let sh = s.rem_euclid(n as i64) as usize;
    (0..n).map(|y| table[(y + n - sh) & (n - 1)]).collect()
}

/// Evaluate the MLE of the rotation kernel `y ↦ eq((y − s) mod 2^n, lambda)`
/// at an arbitrary point `z`.
///
/// Works for arbitrary offsets, positive or negative; `s = 0` degenerates to
/// `eq(y, lambda)`.
pub fn rot_kernel_eval(s: i64, lambda: &[Ext], z: &[Ext]) -> Ext {
    // Writing x = y − s, the output eq((y − s) mod 2^n, lambda) equals
    //      Σ_x eq(x, lambda) · eq((x + s) mod 2^n, z).
    // The sum factorizes bit-by-bit through the carry chain of x + s: process the
    // bits LSB→MSB keeping one accumulator per carry state (y_j = x_j ⊕ s_j ⊕ c,
    // c' = maj(x_j, s_j, c)); the wrap-around (mod 2^n) discards the final carry,
    // so both end states are summed.

    let n = lambda.len();
    debug_assert_eq!(z.len(), n);
    let s_mod = (s.rem_euclid(1i64 << n)) as u64;
    let one = Ext::ONE;

    // acc[c] = partial sum over the low bits, for carry `c` into the next bit.
    let mut acc = [one, Ext::ZERO];
    for j in 0..n {
        let sj = (s_mod >> j) & 1;
        let mut next = [Ext::ZERO, Ext::ZERO];
        for (c, &a) in acc.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for xj in 0..2u64 {
                let yj = xj ^ sj ^ (c as u64);
                let carry_out = ((xj & sj) | ((c as u64) & (xj ^ sj))) as usize;
                let wx = if xj == 1 { lambda[j] } else { one - lambda[j] };
                let wy = if yj == 1 { z[j] } else { one - z[j] };
                next[carry_out] += a * wx * wy;
            }
        }
        acc = next;
    }
    acc[0] + acc[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::{mle_eval, to_ext_vec, boolean_point};
    use fields::{Goldilocks, PrimeField64};
    use rand::{rng, RngExt};

    fn random_ext() -> Ext {
        let mut r = rng();
        Ext::from_array(&[
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
        ])
    }

    fn random_base_col(n: usize) -> Vec<Goldilocks> {
        let mut r = rng();
        (0..(1 << n)).map(|_| Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64)).collect()
    }

    #[test]
    fn eq_evals_matches_eq_eval() {
        let n = 4;
        let point: Vec<Ext> = (0..n).map(|_| random_ext()).collect();
        let table = eq_evals(&point);
        for idx in 0..(1usize << n) {
            let b = boolean_point(idx as u64, n);
            assert_eq!(table[idx], eq_eval(&b, &point));
        }
    }

    #[test]
    fn rotated_eq_table_evaluates_shifted_column() {
        // Σ_y w(y)·rot_s_table[y] must equal w̃^{→s}(λ) = MLE of the shifted column.
        let n = 4;
        let len = 1usize << n;
        let col = random_base_col(n);
        let lambda: Vec<Ext> = (0..n).map(|_| random_ext()).collect();

        for s in [1i64, 2, 3, -1, 5, 0] {
            let kernel = rotate_table(&eq_evals(&lambda), s);
            let claimed: Ext = col.iter().zip(kernel.iter()).map(|(&w, &k)| k * w).sum();

            let shifted: Vec<Goldilocks> =
                (0..len).map(|i| col[(i as i64 + s).rem_euclid(len as i64) as usize]).collect();
            assert_eq!(claimed, mle_eval(&to_ext_vec(&shifted), &lambda), "offset {s}");
        }
    }

    #[test]
    fn rot_kernel_eval_matches_table_mle() {
        // The O(n) carry DP must agree with brute-force MLE of the rotated table.
        let n = 4;
        let lambda: Vec<Ext> = (0..n).map(|_| random_ext()).collect();
        let z: Vec<Ext> = (0..n).map(|_| random_ext()).collect();

        for s in [0i64, 1, 2, 7, -3, 15] {
            let table = rotate_table(&eq_evals(&lambda), s);
            assert_eq!(rot_kernel_eval(s, &lambda, &z), mle_eval(&table, &z), "offset {s}");
        }
    }

    #[test]
    fn rot_kernel_zero_offset_is_eq() {
        let n = 5;
        let lambda: Vec<Ext> = (0..n).map(|_| random_ext()).collect();
        let z: Vec<Ext> = (0..n).map(|_| random_ext()).collect();
        assert_eq!(rot_kernel_eval(0, &lambda, &z), eq_eval(&lambda, &z));
    }
}
