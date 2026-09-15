//! The opening sets: which points each `f_i` is opened at.
//!
//! Mirrors `ShPlonkProver::calculateRoots`. An `f_i` packs `nPols` polynomials
//! into one, interleaved by coefficient:
//!
//! ```text
//! f(X) = Σ_j X^j · pol_j(X^nPols)
//! ```
//!
//! so evaluating the packed polynomial at any `nPols`-th root of `xi·w^p`
//! recovers a linear combination of the individual `pol_j(xi·w^p)`. Opening
//! `f_i` therefore means opening it at all `nPols` such roots, and the opening
//! set for one point is a coset of the `nPols`-th roots of unity.
//!
//! The roots come from the seed rather than from `xi`: `root = w{n}_{p}d{n} ·
//! xiSeed^(powerW/nPols)`, which avoids taking an `nPols`-th root of `xi` --
//! the verifier cannot do that, but it can raise the seed to a smaller power,
//! since `xi = xiSeed^powerW` by construction. That is the whole reason the
//! protocol carries a seed and derives `xi` from it.

use anyhow::{Context, Result, bail};
use num_bigint::BigUint;

use crate::fr;
use crate::setup::{ShPlonkPol, ShPlonkSetup};

/// One `f_i`'s opening set, in the prover's order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningSet {
    /// Which `f_i` this belongs to.
    pub index: u32,
    /// The roots, grouped by opening point: `nPols` for each, in the order
    /// `calculateRoots` writes them.
    pub roots: Vec<BigUint>,
}

impl OpeningSet {
    /// The roots belonging to the `k`-th opening point.
    pub fn slot(&self, k: usize, n_pols: usize) -> &[BigUint] {
        &self.roots[k * n_pols..(k + 1) * n_pols]
    }
}

/// The name of the omega generating the `n`-th roots of unity.
pub fn omega_key(n_pols: usize) -> String {
    format!("w{n_pols}")
}

/// The name of the omega that starts the coset for opening point `p`.
///
/// This is an `nPols`-th root of `w^p`, precomputed in the key because the
/// verifier cannot take roots itself.
pub fn coset_key(n_pols: usize, opening_point: u32) -> String {
    format!("w{n_pols}_{opening_point}d{n_pols}")
}

fn omega(setup: &ShPlonkSetup, key: &str) -> Result<BigUint> {
    let raw = setup.omegas.get(key).with_context(|| format!("verification key is missing the omega {key:?}"))?;
    fr::from_decimal(raw).with_context(|| format!("omega {key:?}"))
}

/// The roots for one `f_i`.
pub fn roots_for(setup: &ShPlonkSetup, f: &ShPlonkPol, xi_seed: &BigUint) -> Result<OpeningSet> {
    let n_pols = f.pols.len();
    if n_pols == 0 {
        bail!("f{} packs no polynomials", f.index);
    }

    // The exponent below is integer division, so a group whose size does not
    // divide powerW would silently open at the wrong points.
    if !(setup.power_w as usize).is_multiple_of(n_pols) {
        bail!("f{} packs {n_pols} polynomials, which does not divide powerW = {}", f.index, setup.power_w);
    }

    let step = omega(setup, &omega_key(n_pols))?;
    let seed_power = fr::pow(xi_seed, setup.power_w as u64 / n_pols as u64);

    let mut roots = Vec::with_capacity(n_pols * f.opening_points.len());
    for &point in &f.opening_points {
        // Point 0 opens at xi itself, whose coset starts at 1.
        let start = if point == 0 { BigUint::from(1u32) } else { omega(setup, &coset_key(n_pols, point))? };

        let mut root = fr::mul(&start, &seed_power);
        for _ in 0..n_pols {
            roots.push(root.clone());
            root = fr::mul(&root, &step);
        }
    }

    Ok(OpeningSet { index: f.index, roots })
}

/// The roots for every `f_i`, in index order.
pub fn all_roots(setup: &ShPlonkSetup, xi_seed: &BigUint) -> Result<Vec<OpeningSet>> {
    setup.f.iter().map(|f| roots_for(setup, f, xi_seed)).collect()
}

/// Every root of every `f_i`, concatenated in the prover's order.
///
/// This is the multiset `computeZT` vanishes on. Distinct `f_i` that pack the
/// same number of polynomials at the same opening points share their roots, so
/// this sequence repeats values -- it is not the underlying set.
pub fn flattened(sets: &[OpeningSet]) -> Vec<BigUint> {
    sets.iter().flat_map(|s| s.roots.iter().cloned()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{self, Reference};
    use std::collections::HashSet;

    fn real() -> Reference {
        reference::load()
    }

    /// The external check: the roots must be the prover's, element for element
    /// and in its order. Every invariant below would also hold for a permuted
    /// or differently-seeded set, so this is the test that pins the rule.
    #[test]
    fn matches_the_reference_roots() {
        let r = real();
        let sets = all_roots(&r.setup, &r.xi_seed).unwrap();

        assert_eq!(sets.len(), r.roots.len());
        for (set, want) in sets.iter().zip(&r.roots) {
            assert_eq!(&set.roots, want, "f{} roots differ from the prover", set.index);
        }
    }

    /// The defining property: every root of a group is an `nPols`-th root of
    /// `xi*w^p`. This is what makes the packed evaluation recoverable, and it
    /// ties the roots back to the challenge without trusting the key's omegas.
    #[test]
    fn every_root_is_an_nth_root_of_xi_times_w_to_the_point() {
        let r = real();
        let w = omega(&r.setup, "w").unwrap();

        for (f, set) in r.setup.f.iter().zip(all_roots(&r.setup, &r.xi_seed).unwrap()) {
            let n = f.pols.len();
            for (k, &point) in f.opening_points.iter().enumerate() {
                let want = fr::mul(&r.xi, &fr::pow(&w, point as u64));
                for (j, root) in set.slot(k, n).iter().enumerate() {
                    assert_eq!(fr::pow(root, n as u64), want, "f{} point {point} root {j}", f.index);
                }
            }
        }
    }

    /// Within one opening set the roots must be distinct, or the interpolation
    /// that recovers `R_i` is not determined.
    #[test]
    fn roots_within_an_opening_set_are_distinct() {
        let r = real();
        for set in all_roots(&r.setup, &r.xi_seed).unwrap() {
            let distinct: HashSet<_> = set.roots.iter().collect();
            assert_eq!(distinct.len(), set.roots.len(), "f{} repeats a root", set.index);
        }
    }

    /// xi is recoverable from the seed the roots were built from, which is the
    /// consistency the protocol relies on.
    #[test]
    fn the_seed_raised_to_power_w_is_xi() {
        let r = real();
        assert_eq!(fr::pow(&r.xi_seed, r.setup.power_w as u64), r.xi);
    }

    /// One root per (polynomial, opening point) slot: the opening set has
    /// exactly the size of the evaluation list it will interpolate.
    #[test]
    fn there_is_one_root_per_evaluation_slot() {
        let r = real();
        let sets = all_roots(&r.setup, &r.xi_seed).unwrap();

        for (f, set) in r.setup.f.iter().zip(&sets) {
            assert_eq!(set.roots.len(), f.pols.len() * f.opening_points.len(), "f{}", f.index);
        }
        assert_eq!(flattened(&sets).len(), 42);
    }

    /// Groups that pack the same number of polynomials at the same points open
    /// at the same roots, so the flattened list is a multiset. ZT is built from
    /// it verbatim, so this repetition is part of the polynomial the verifier
    /// must agree on rather than an artefact to remove.
    #[test]
    fn the_flattened_roots_are_a_multiset_not_a_set() {
        let r = real();
        let all = flattened(&all_roots(&r.setup, &r.xi_seed).unwrap());
        let distinct: HashSet<_> = all.iter().collect();

        assert_eq!(all.len(), 42);
        assert_eq!(distinct.len(), 25, "f3, f4 and f5 share one set of cube roots");
    }

    /// A different seed gives entirely different roots -- they are challenge
    /// dependent, not fixed by the key.
    #[test]
    fn roots_follow_the_seed() {
        let r = real();
        let a = flattened(&all_roots(&r.setup, &r.xi_seed).unwrap());
        let b = flattened(&all_roots(&r.setup, &BigUint::from(3u32)).unwrap());
        assert!(a.iter().zip(&b).all(|(x, y)| x != y));
    }

    #[test]
    fn rejects_a_group_size_that_does_not_divide_power_w() {
        let mut r = real();
        r.setup.f[0].pols.push("extra".into()); // 6 -> 7, and 7 does not divide 12
        assert!(all_roots(&r.setup, &r.xi_seed).is_err());
    }

    #[test]
    fn rejects_a_key_missing_an_omega() {
        let mut r = real();
        r.setup.omegas.remove("w6");
        assert!(all_roots(&r.setup, &r.xi_seed).is_err());
    }

    /// Opening point 0 starts its coset at 1 rather than at a key omega, so a
    /// key that omits `w{n}_0d{n}` is still usable.
    #[test]
    fn point_zero_needs_no_coset_omega() {
        let mut r = real();
        r.setup.omegas.remove("w6_0d6");
        assert!(roots_for(&r.setup, &r.setup.f[0].clone(), &r.xi_seed).is_ok());
    }
}
