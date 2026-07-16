//! The equality kernel `eq` and the rotation kernel `rot_s`.
//!
//! `eq(x, y) = Π_j (x_j·y_j + (1−x_j)(1−y_j))` is the Lagrange kernel of the
//! hypercube.
//!
//! The rotation kernel handles "arbitrary offset" accesses, the
//! shifted column `w^{(s)}(b) = w((b + s) mod 2^n)` satisfies
//!
//! `w̃^{(s)}(λ) = Σ_y w((y + s) mod 2^n) · eq(y, λ) = Σ_y w(y) · K_s(y)`,
//!
//! with `K_s(y) = eq((y − s) mod 2^n, λ)`,
//! so a shifted-column evaluation claim is a weighted-sum claim on the *base*
//! column with the `eq(·, λ)` table cyclically rotated.

use crate::hypercube::Ext;
use fields::{Field, Goldilocks};

/// Elements of the multiplicative subgroup `D ⊂ F*` of order `2^ell`, as base
/// field: `D = { ω^0, …, ω^{2^ell − 1} }` with `ω = W[ell]` a primitive
/// `2^ell`-th root of unity. Used by the univariate-skip (hyperprism `D × H^n`)
/// zerocheck; see `docs/multilinear-univariate-skip.md`.
pub fn d_subgroup(ell: usize) -> Vec<Goldilocks> {
    let omega = Goldilocks::new(Goldilocks::W[ell]);
    let mut d = Vec::with_capacity(1 << ell);
    let mut cur = Goldilocks::ONE;
    for _ in 0..(1usize << ell) {
        d.push(cur);
        cur *= omega;
    }
    d
}

#[inline]
fn pow_2exp(mut z: Ext, ell: usize) -> Ext {
    for _ in 0..ell {
        z = z * z;
    }
    z
}

/// Lagrange basis of `D` evaluated at `z`: returns `[L_d(z)]_{d∈D}` with
/// `L_d(z) = d·(z^N − 1) / (N·(z − d))`, `N = 2^ell`. If `z ∈ D` (i.e.
/// `z^N = 1`), returns the corresponding unit vector. `Σ_d f(d)·L_d(z)` is the
/// unique degree-`<N` interpolant of `f: D → F` evaluated at `z`.
pub fn lagrange_d(ell: usize, z: Ext) -> Vec<Ext> {
    let n = 1usize << ell;
    let d = d_subgroup(ell);
    let num = pow_2exp(z, ell) - Ext::ONE; // z^N − 1
    if num.is_zero() {
        // z is a 2^ell-th root of unity — which for Goldilocks all lie in D — so
        // the interpolation is the unit vector at that point.
        let mut e = vec![Ext::ZERO; n];
        let pos = d.iter().position(|&di| z == Ext::from_base(di)).expect("root of unity must be in D");
        e[pos] = Ext::ONE;
        return e;
    }
    let n_inv = Goldilocks::new(n as u64).inverse();
    let factor = num * n_inv; // (z^N − 1)/N
    d.iter().map(|&di| factor * (z - di).inverse() * di).collect()
}

/// Prover-side table of the plain univariate-skip kernel over `{0,1}^m`
/// (`m = ell + lambda_x.len()`): `K_0(b) = L_{b_P}(γ)·eq(b_X, λ_X)`, where the
/// low `ell` bits `b_P` index the subgroup `D` (Lagrange in `γ`) and the high
/// bits `b_X` the hypercube (eq in `λ_X`). Shifted reads use
/// [`rotate_table`]`(K_0, s)`. Option B (multilinear commitment) univariate skip;
/// see `docs/multilinear-univariate-skip.md`.
pub fn skip_kernel_table(ell: usize, gamma: Ext, lambda_x: &[Ext]) -> Vec<Ext> {
    let lag = lagrange_d(ell, gamma);
    let eqx = eq_evals(lambda_x);
    let mut t = Vec::with_capacity((1 << ell) * eqx.len());
    for &ex in &eqx {
        for &lp in &lag {
            t.push(lp * ex);
        }
    }
    t
}

/// MLE of the shifted skip kernel `b ↦ K_0((b − s) mod 2^m)` at `z`
/// (`z.len() = ell + lambda_x.len()`). Decomposes the shift into the P-block
/// (Lagrange in `γ`, hypercube `eq` over `z_P`) and the carry into the X-block
/// (the hypercube rotation kernel over `z_X`) — the multilinear-P-basis analog
/// of SWIRL 2.5.1. `s = 0` gives the plain kernel `K_0`; `ell = 0` degenerates
/// to the pure-hypercube [`rot_kernel_eval`].
pub fn skip_kernel_eval(ell: usize, s: i64, gamma: Ext, lambda_x: &[Ext], z: &[Ext]) -> Ext {
    debug_assert_eq!(z.len(), ell + lambda_x.len());
    let np = 1usize << ell;
    let m = ell + lambda_x.len();
    let lag = lagrange_d(ell, gamma);
    let eq_zp = eq_evals(&z[..ell]);
    let z_x = &z[ell..];

    let s_mod = s.rem_euclid(1i64 << m) as usize;
    let s_p = s_mod & (np - 1);
    let s_x = (s_mod >> ell) as i64;
    // The X-block shift is `s_x` (no P-carry) or `s_x + 1` (P-block overflows).
    let rot0 = rot_kernel_eval(s_x, lambda_x, z_x);
    let rot1 = rot_kernel_eval(s_x + 1, lambda_x, z_x);

    let mut acc = Ext::ZERO;
    for (p, &lp) in lag.iter().enumerate() {
        let full = p + s_p;
        let low_new = full & (np - 1);
        let rot = if full >> ell == 0 { rot0 } else { rot1 };
        acc += lp * eq_zp[low_new] * rot;
    }
    acc
}

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
    use crate::hypercube::{mle_eval, boolean_point};
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
        for (idx, &t) in table.iter().enumerate() {
            let b = boolean_point(idx as u64, n);
            assert_eq!(t, eq_eval(&b, &point));
        }
    }

    #[test]
    fn rotated_eq_table_evaluates_shifted_column() {
        // Σ_y w(y)·rot_s_table[y] must equal w̃^{(s)}(λ) = MLE of the shifted column.
        let n = 4;
        let len = 1usize << n;
        let col = random_base_col(n);
        let lambda: Vec<Ext> = (0..n).map(|_| random_ext()).collect();

        for s in [1i64, 2, 3, -1, 5, 0] {
            let kernel = rotate_table(&eq_evals(&lambda), s);
            let claimed: Ext = col.iter().zip(kernel.iter()).map(|(&w, &k)| k * w).sum();

            let shifted: Vec<Goldilocks> =
                (0..len).map(|i| col[(i as i64 + s).rem_euclid(len as i64) as usize]).collect();
            assert_eq!(claimed, mle_eval(&Ext::from_base_batch(&shifted), &lambda), "offset {s}");
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

    // --- Univariate-skip (hyperprism D × H^n) kernels ---

    #[test]
    fn lagrange_d_is_delta_on_subgroup() {
        for ell in 0..=3 {
            let d = d_subgroup(ell);
            assert_eq!(d.len(), 1 << ell);
            // Each element has order dividing 2^ell (so d^{2^ell} = 1).
            for &di in &d {
                assert_eq!(super::pow_2exp(Ext::from_base(di), ell), Ext::ONE);
            }
            // L_{d_i}(d_j) = δ_{ij}.
            for (i, &di) in d.iter().enumerate() {
                let l = lagrange_d(ell, Ext::from_base(di));
                for (j, &v) in l.iter().enumerate() {
                    assert_eq!(v, if i == j { Ext::ONE } else { Ext::ZERO }, "ell={ell} i={i} j={j}");
                }
            }
        }
    }

    // --- Option-B (multilinear-commitment) univariate-skip kernel ---

    /// `skip_kernel_eval` must equal the MLE of the rotated `skip_kernel_table`
    /// (the ground-truth definition), across skip lengths, suffix sizes, and
    /// offsets — including offsets that carry out of the P-block into the X-block.
    #[test]
    fn skip_kernel_eval_matches_table_mle() {
        for ell in 0..=3usize {
            for nx in 0..=3usize {
                let m = ell + nx;
                let gamma = random_ext();
                let lambda_x: Vec<Ext> = (0..nx).map(|_| random_ext()).collect();
                let z: Vec<Ext> = (0..m).map(|_| random_ext()).collect();
                let k0 = skip_kernel_table(ell, gamma, &lambda_x);
                assert_eq!(k0.len(), 1 << m);
                for s in [0i64, 1, 2, 3, 5, -1, (1 << m) + 1] {
                    let ks = rotate_table(&k0, s);
                    let brute = mle_eval(&ks, &z);
                    assert_eq!(skip_kernel_eval(ell, s, gamma, &lambda_x, &z), brute, "ell={ell} nx={nx} s={s}");
                }
            }
        }
    }

    /// `ell = 0` reduces to the pure-hypercube rotation kernel (current behaviour).
    #[test]
    fn skip_kernel_eval_ell0_is_hypercube_rot() {
        let nx = 4;
        let gamma = random_ext();
        let lambda_x: Vec<Ext> = (0..nx).map(|_| random_ext()).collect();
        let z: Vec<Ext> = (0..nx).map(|_| random_ext()).collect();
        for s in [0i64, 1, 3] {
            assert_eq!(skip_kernel_eval(0, s, gamma, &lambda_x, &z), rot_kernel_eval(s, &lambda_x, &z));
        }
    }
}
