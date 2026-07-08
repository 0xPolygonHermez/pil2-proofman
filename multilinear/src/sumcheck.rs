//! Generic sumcheck: prover-side round oracles and verifier-side round checks.

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

/// Oracle for `Σ_b a(b)·b'(b)` with `a`, `b'` multilinear.
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
        let (mut g0, mut g1, mut g2) = (Ext::ZERO, Ext::ZERO, Ext::ZERO);
        for i in 0..half {
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
        vec![g0, g1, g2]
    }

    fn bind(&mut self, r: Ext) {
        fold_mle(&mut self.a, r);
        fold_mle(&mut self.b, r);
    }
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

/// One verifier-side sumcheck round: check `g(0) + g(1) == claim`, then return
/// the next claim `g(r)`.
pub fn verify_sumcheck_round(claim: Ext, round_evals: &[Ext], r: Ext, round: usize) -> Result<Ext, MlError> {
    if round_evals.len() < 2 {
        return Err(MlError::Malformed(format!("round {round}: round polynomial too short")));
    }
    if round_evals[0] + round_evals[1] != claim {
        return Err(MlError::SumcheckRound { round });
    }
    Ok(interpolate_at(round_evals, r))
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
            let evals = oracle.round_evals();
            let r = random_ext();
            claim = verify_sumcheck_round(claim, &evals, r, round).expect("round check");
            oracle.bind(r);
            point.push(r);
        }
        // Final: claim == ã(point)·b̃(point)
        let av = crate::hypercube::mle_eval(&a, &point);
        let bv = crate::hypercube::mle_eval(&b, &point);
        assert_eq!(claim, av * bv);
    }

    #[test]
    fn tampered_round_poly_rejected() {
        let n = 3;
        let a: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        let b: Vec<Ext> = (0..(1 << n)).map(|_| random_ext()).collect();
        let oracle = ProductOracle::new(a, b);
        let claim = oracle.current_sum();
        let mut evals = oracle.round_evals();
        evals[0] += Ext::ONE;
        assert!(verify_sumcheck_round(claim, &evals, random_ext(), 0).is_err());
    }
}
