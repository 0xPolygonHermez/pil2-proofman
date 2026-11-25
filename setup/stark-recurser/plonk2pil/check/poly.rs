//! Sparse polynomials of degree <= 2 over copy-class variables, in a canonical form.

use proofman_fields::Field;

/// The unit "variable": `(ONE, ONE)` is the constant monomial, `(v, ONE)` the linear one.
pub const ONE: u32 = u32::MAX;

/// A monomial `(a, b)` with `a <= b`; see [`ONE`].
pub type Mono = (u32, u32);

pub fn mono(a: u32, b: u32) -> Mono {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// `Σ coef·mono`, sorted by monomial, merged, without zero coefficients.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Poly<F>(pub Vec<(Mono, F)>);

/// A linear combination `Σ coef·var` (`ONE` = the constant term), sorted and merged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lin<F>(pub Vec<(u32, F)>);

fn merge<K: Ord + Copy, F: Field>(mut t: Vec<(K, F)>) -> Vec<(K, F)> {
    t.sort_by_key(|x| x.0);
    let mut out: Vec<(K, F)> = Vec::with_capacity(t.len());
    for (k, c) in t {
        match out.last_mut() {
            Some((lk, lc)) if *lk == k => *lc += c,
            _ => out.push((k, c)),
        }
    }
    out.retain(|(_, c)| !c.is_zero());
    out
}

impl<F: Field> Lin<F> {
    pub fn new(terms: Vec<(u32, F)>) -> Self {
        Lin(merge(terms))
    }
    pub fn var(v: u32) -> Self {
        Lin(vec![(v, F::ONE)])
    }
}

impl<F: Field> Poly<F> {
    pub fn new(terms: Vec<(Mono, F)>) -> Self {
        Poly(merge(terms))
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// `k·l`, as monomials `(v, ONE)`.
    pub fn push_lin(out: &mut Vec<(Mono, F)>, k: F, l: &Lin<F>) {
        out.extend(l.0.iter().map(|&(v, c)| ((v, ONE), k * c)));
    }

    /// `k·l·r`.
    pub fn push_prod(out: &mut Vec<(Mono, F)>, k: F, l: &Lin<F>, r: &Lin<F>) {
        for &(a, ca) in &l.0 {
            for &(b, cb) in &r.0 {
                out.push((mono(a, b), k * ca * cb));
            }
        }
    }
}

/// Scales every poly of `ps` to a leading coefficient of 1 (one batched inversion), so `λ·p` and `p` compare equal.
pub fn normalize<F: Field>(ps: &mut [Poly<F>]) {
    let mut prefix = Vec::with_capacity(ps.len());
    let mut acc = F::ONE;
    for p in ps.iter() {
        prefix.push(acc);
        acc *= p.0.first().map_or(F::ONE, |t| t.1);
    }
    let mut inv = acc.inverse();
    for (p, pre) in ps.iter_mut().zip(prefix).rev() {
        let Some(&(_, lead)) = p.0.first() else { continue };
        let s = inv * pre;
        inv *= lead;
        p.0.iter_mut().for_each(|t| t.1 *= s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_fields::Goldilocks;

    fn g(v: u64) -> Goldilocks {
        Goldilocks::new(v)
    }

    #[test]
    fn scalar_multiples_normalize_equal() {
        let p = Poly::new(vec![((1, ONE), g(3)), ((ONE, ONE), g(6)), ((1, 2), g(9))]);
        let q = Poly::new(vec![((1, ONE), g(5)), ((ONE, ONE), g(10)), ((1, 2), g(15))]);
        let r = Poly::new(vec![((1, ONE), g(5)), ((ONE, ONE), g(11)), ((1, 2), g(15))]);
        let mut v = vec![p, Poly(vec![]), q, r];
        normalize(&mut v);
        assert!(v[0] == v[2] && v[0] != v[3] && v[1].is_zero());
    }

    #[test]
    fn terms_merge_and_cancel() {
        let p = Poly::new(vec![((1, ONE), g(3)), ((1, ONE), -g(3)), ((2, ONE), g(1))]);
        assert_eq!(p.0, vec![((2, ONE), g(1))]);
    }
}
