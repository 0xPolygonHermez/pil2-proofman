//! The sumcheck PIOP: Given f ∈ F[Z,X₁,...,Xₘ] be a polynomial of bounded individual degree,
//! prove that
//!
//! `∑_{x ∈ D_m} f(x) = 0`.
//!
//! where `D_m = D x {0,1}^m` is a hyperprism with base `D`, a smooth multiplicative
//! subgroup of F, and `m` a positive integer.

use crate::error::MlError;
use crate::hypercube::{fold_mle, Ext};
use fields::{Field, Goldilocks, PrimeField64};

/// A prover-side oracle producing sumcheck round polynomials.
pub trait SumcheckOracle {
    /// Number of remaining rounds.
    fn num_rounds(&self) -> usize;
    /// Degree bound `d` of every round polynomial.
    fn round_degree(&self) -> usize;
    /// Evaluations of the current round polynomial at `0, 1, …, d`.
    fn round_evals(&self) -> Vec<Ext>;
    /// Bind the current variable to `r`.
    fn bind(&mut self, r: Ext);
}

/// Oracle for `Σ_z a(z)·b(z)` with `a`, `b` multilinear.
pub struct ProductOracle {
    pub a: Vec<Ext>,
    pub b: Vec<Ext>,
}

impl ProductOracle {
    pub fn new(a: Vec<Ext>, b: Vec<Ext>) -> Self {
        assert_eq!(a.len(), b.len());
        assert!(a.len().is_power_of_two());
        Self { a, b }
    }

    pub fn current_sum(&self) -> Ext {
        self.a.iter().zip(self.b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

impl SumcheckOracle for ProductOracle {
    fn num_rounds(&self) -> usize {
        self.a.len().trailing_zeros() as usize
    }

    fn round_degree(&self) -> usize {
        2
    }

    fn round_evals(&self) -> Vec<Ext> {
        let half = self.a.len() / 2;
        // Parallel reduction over pairs; each chunk keeps private accumulators.
        let partials = crate::par::map_chunks(half, |start, end| {
            let (mut g0, mut g1, mut g2) = (Ext::ZERO, Ext::ZERO, Ext::ZERO);
            for i in start..end {
                let a0 = self.a[2 * i];
                let a1 = self.a[2 * i + 1];
                let b0 = self.b[2 * i];
                let b1 = self.b[2 * i + 1];
                g0 += a0 * b0;
                g1 += a1 * b1;
                // value at X = 2 by linearity: v(2) = 2·v(1) − v(0)
                let a2 = a1.double() - a0;
                let b2 = b1.double() - b0;
                g2 += a2 * b2;
            }
            [g0, g1, g2]
        });
        let mut g = [Ext::ZERO; 3];
        for p in partials {
            for (gx, px) in g.iter_mut().zip(p.iter()) {
                *gx += *px;
            }
        }
        g.to_vec()
    }

    fn bind(&mut self, r: Ext) {
        // Each fold streams sequentially (memory-bound); the two run in parallel.
        let Self { a, b } = self;
        crate::par::join(|| fold_mle(a, r), || fold_mle(b, r));
    }
}

/// Oracle for `Σ_z a(z)·eq(z, ω)` with `a` multilinear and the second factor the
/// equality kernel at random `ω`.
///
/// Exploits the tensor structure of `eq`: at round `i`,
/// `eq((r₁,...,rᵢ₋₁,X, x), ω) = cᵢ · eq₁(X, ωᵢ) · eq(x, (ωᵢ₊₁, ... , ωₙ))` with
/// `cᵢ = eq((r₁,...,rᵢ₋₁), (ω₁,...,ωᵢ₋₁))`, so the round polynomial factors as
/// `gᵢ(X) = cᵢ · eq₁(X, ωᵢ) · hᵢ(X)` where
///     `hᵢ(X) = Σ_x a(r₁,...,rᵢ₋₁,X, x)·eq(x, (ωᵢ₊₁, ... , ωₙ))`
/// is *linear*.
pub struct EqProductOracle {
    pub a: Vec<Ext>,
    /// `suffix[i] = [eq(x, (ωᵢ₊₁, ... , ωₙ))]_x`, length `2^{n−(i+1)}`.
    suffix: Vec<Vec<Ext>>,
    round: usize,
}

impl EqProductOracle {
    pub fn new(a: Vec<Ext>, omega: Vec<Ext>) -> Self {
        assert!(a.len().is_power_of_two());
        let n = omega.len();
        assert_eq!(a.len(), 1usize << n);

        // Build every suffix `eq(x, (ωᵢ₊₁, ... , ωₙ))` table in one backwards tensor sweep.
        let mut suffix: Vec<Vec<Ext>> = Vec::with_capacity(n);
        if n > 0 {
            suffix.push(vec![Ext::ONE]); // t = n−1: empty suffix
            for t in (0..n - 1).rev() {
                let l = omega[t + 1];
                let prev = suffix.last().unwrap();
                let mut cur = Vec::with_capacity(prev.len() * 2);
                for &v in prev {
                    let hi = v * l;
                    cur.push(v - hi);
                    cur.push(hi);
                }
                suffix.push(cur);
            }
            suffix.reverse();
        }
        Self { a, suffix, round: 0 }
    }

    /// The *linear* factor of the current round polynomial.
    fn h_evals(&self) -> Ext {
        // h(X) = Σ_x a(r₁,...,rᵢ₋₁,X, x)·eq(x, (ωᵢ₊₁, ... , ωₙ))
        // Compute h(0)
        let table = &self.suffix[self.round];
        debug_assert_eq!(table.len(), self.a.len() / 2);
        let mut h0 = Ext::ZERO;
        for (i, &e) in table.iter().enumerate() {
            h0 += self.a[2 * i] * e;
        }
        h0
    }
}

impl SumcheckOracle for EqProductOracle {
    fn num_rounds(&self) -> usize {
        self.a.len().trailing_zeros() as usize
    }

    fn round_degree(&self) -> usize {
        1
    }

    fn round_evals(&self) -> Vec<Ext> {
        let h0 = self.h_evals();
        vec![h0]
    }

    fn bind(&mut self, r: Ext) {
        // Compute the next round polynomial
        fold_mle(&mut self.a, r);
        self.round += 1;
    }
}

/// Verifier side of one generic sumcheck round
pub fn verifier_sumcheck_round(claim: Ext, round_evals: &[Ext], r: Ext, round: usize) -> Result<Ext, MlError> {
    if round_evals.is_empty() {
        return Err(MlError::Malformed(format!("round {round}: round polynomial too short")));
    }

    // The prover sends the round polynomial `gᵢ`` evaluated at `1, …, d`.
    // Compute the "missing" evaluation at `0` by the sumcheck relation
    //  gᵢ₋₁(rᵢ₋₁) = gᵢ(0) + gᵢ(1) ==> gᵢ(0) = gᵢ₋₁(rᵢ₋₁) − gᵢ(1)
    let g0 = claim - round_evals[0];

    // Compute the next claim `gᵢ(rᵢ)` by Lagrange interpolation at `r`.
    Ok(interpolate_at(&[g0].iter().chain(round_evals.iter()).copied().collect::<Vec<_>>(), r))
}

/// Verifier side of one [`EqProductOracle`] sumcheck round
pub fn eq_product_verifier_sumcheck_round(claim: Ext, h0: Ext, l: Ext, r: Ext) -> Result<Ext, MlError> {
    // gᵢ(0) + gᵢ(1) = eq((r₁,...,rᵢ₋₁), (ω₁,...,ωᵢ₋₁))·eq₁(0, ωᵢ)·hᵢ(0) +
    //                 eq((r₁,...,rᵢ₋₁), (ω₁,...,ωᵢ₋₁))·eq₁(1, ωᵢ)·hᵢ(1) = prefix·[(1−ωᵢ)·hᵢ(0) + ωᵢ·hᵢ(1)].
    // gᵢ₋₁(rᵢ₋₁) = eq((r₁,...,rᵢ₋₂), (ω₁,...,ωᵢ₋₂))·eq₁(rᵢ₋₁, ωᵢ)·hᵢ₋₁(rᵢ₋₁) = prefix·hᵢ₋₁(rᵢ₋₁)
    // ==> hᵢ(1) = 1/(ωᵢ) * (hᵢ₋₁(rᵢ₋₁) - (1−ωᵢ)·hᵢ(0))
    //
    // Note: As ωᵢ is random, we can assume w.h.p it is nonzero and invertible.

    let h_prev = claim;
    let h1 = (h_prev - (Ext::ONE - l) * h0) * l.inverse();
    let round_evals = [h0, h1];
    let next_claim = interpolate_at(&round_evals, r);

    Ok(next_claim)
}

/// Evaluate at `x` the univariate polynomial of degree `≤ d` given by its evaluations at
/// the integer nodes `0, 1, …, d` (Lagrange interpolation).
pub fn interpolate_at(evals: &[Ext], x: Ext) -> Ext {
    let d = evals.len() - 1;

    // prefix[i] = Π_{j<i} (x − j), suffix[i] = Π_{j>i} (x − j)
    let mut prefix = vec![Ext::ONE; d + 2];
    for j in 0..=d {
        prefix[j + 1] = prefix[j] * (x - Goldilocks::from_u64(j as u64));
    }
    let mut suffix = vec![Ext::ONE; d + 2];
    for j in (0..=d).rev() {
        suffix[j] = suffix[j + 1] * (x - Goldilocks::from_u64(j as u64));
    }

    // den_i = Π_{j≠i} (i − j) = (−1)^(d−i) · i! · (d−i)!
    let mut result = Ext::ZERO;
    for i in 0..=d {
        let mut den = Goldilocks::ONE;
        for j in 1..=i {
            den *= Goldilocks::from_u64(j as u64);
        }
        for j in 1..=(d - i) {
            den *= Goldilocks::from_u64(j as u64);
        }
        if (d - i) % 2 == 1 {
            den = -den;
        }
        result += evals[i] * prefix[i] * suffix[i + 1] * den.inverse();
    }
    result
}

/// Evaluate at `x` the polynomial of degree `< nodes.len()` interpolating the
/// points `(nodes[i], values[i])`, via direct Lagrange. Nodes must be distinct;
/// `x` may coincide with a node (returns that value, no division by zero).
/// `O(n²)`; used to resample the univariate-skip round polynomial from cheap
/// evaluation nodes onto the canonical `0..deg` grid.
pub fn lagrange_eval(nodes: &[Ext], values: &[Ext], x: Ext) -> Ext {
    let n = nodes.len();
    let mut acc = Ext::ZERO;
    for i in 0..n {
        let mut num = values[i];
        let mut den = Ext::ONE;
        for (j, &nj) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }
            num *= x - nj;
            den *= nodes[i] - nj;
        }
        acc += num * den.inverse();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::PrimeField64;
    use rand::{rng, RngExt};

    fn random_ext() -> Ext {
        let mut r = rng();
        Ext::from_array(&[
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
        ])
    }

    #[test]
    fn interpolate_at_recovers_polynomial() {
        // p(x) = c0 + c1 x + c2 x²
        let c: Vec<Ext> = (0..3).map(|_| random_ext()).collect();
        let p = |x: Ext| c[0] + c[1] * x + c[2] * x * x;
        let evals: Vec<Ext> = (0..3u64)
            .map(|t| p(Ext::from_array(&[Goldilocks::from_u64(t), Goldilocks::ZERO, Goldilocks::ZERO])))
            .collect();
        let x = random_ext();
        assert_eq!(interpolate_at(&evals, x), p(x));
    }

    #[test]
    fn product_sumcheck_roundtrip() {
        let n = 6;
        let a: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        let b: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        let mut oracle = ProductOracle::new(a.clone(), b.clone());
        let mut claim = oracle.current_sum();

        let mut point = Vec::new();
        for round in 0..n {
            // Tweak 1: the prover omits g(0); send the tail g(1..=d).
            let evals = oracle.round_evals();
            let r = random_ext();
            claim = verifier_sumcheck_round(claim, &evals[1..], r, round).expect("round check");
            oracle.bind(r);
            point.push(r);
        }
        // Final: claim == ã(point)·b̃(point)
        let av = crate::hypercube::mle_eval(&a, &point);
        let bv = crate::hypercube::mle_eval(&b, &point);
        assert_eq!(claim, av * bv);
    }

    #[test]
    fn tampered_round_poly_breaks_final_claim() {
        // With Tweak 1 the verifier recovers g(0) from g(0)+g(1)=claim rather
        // than checking it, so a bad round message no longer errors in-round —
        // it must instead surface as a wrong final claim (caught downstream).
        let n = 3;
        let a: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        let b: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        let mut oracle = ProductOracle::new(a.clone(), b.clone());
        let mut claim = oracle.current_sum();

        let mut point = Vec::new();
        for round in 0..n {
            let evals = oracle.round_evals();
            let mut sent = evals[1..].to_vec();
            if round == 0 {
                sent[0] += Ext::ONE; // tamper g(1) of the first round
            }
            let r = random_ext();
            claim = verifier_sumcheck_round(claim, &sent, r, round).expect("round runs");
            oracle.bind(r);
            point.push(r);
        }
        let av = crate::hypercube::mle_eval(&a, &point);
        let bv = crate::hypercube::mle_eval(&b, &point);
        assert_ne!(claim, av * bv, "a tampered round message must break the final claim");
    }
}
