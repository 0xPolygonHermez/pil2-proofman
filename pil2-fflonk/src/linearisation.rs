//! The scalars of the SHPLONK linearisation.
//!
//! Mirrors `ShPlonkProver::computeL`, which builds
//!
//! ```text
//! L(X) = Σ_i preL[i] · (f_i(X) - R_i(y)) - Z_T(y) · W(X)
//! ```
//!
//! and divides by `X - y`. The verifier never forms `L`; it needs the same
//! scalars to assemble the pairing check, and only those are computed here.
//! Everything in this module is in `Fr`, so it is testable without any curve
//! arithmetic -- the group elements enter only at the final pairing.
//!
//! One input is not derivable here. `f8` packs the quotient `Q`, whose
//! evaluation the proof deliberately omits: the verifier reconstructs it from
//! the constraint identity, which needs the AIR's expressions rather than the
//! opening scheme. [`resolve_evaluations`] therefore takes the reconstructed
//! values as a parameter, keeping the layers separate instead of guessing.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use num_bigint::BigUint;
use num_traits::One;

use crate::fr;
use crate::proof::{ShPlonkProof, evaluation_key};
use crate::roots::OpeningSet;
use crate::setup::{ShPlonkPol, ShPlonkSetup};
use crate::verifier::non_committed_pols;

/// Every evaluation the opening needs, keyed as [`evaluation_key`] does.
pub type Evaluations = BTreeMap<String, BigUint>;

/// Combine the proof's evaluations with the ones the verifier reconstructs.
///
/// `derived` supplies the polynomials the proof omits -- in the reference, the
/// single entry `Q`. Supplying one the proof already carries is an error rather
/// than an override: it would let a caller silently replace a claimed opening.
pub fn resolve_evaluations(setup: &ShPlonkSetup, proof: &ShPlonkProof, derived: &Evaluations) -> Result<Evaluations> {
    let omitted = non_committed_pols(setup);
    let mut out = Evaluations::new();

    for f in &setup.f {
        for &point in &f.opening_points {
            for pol in &f.pols {
                let key = evaluation_key(pol, point);
                if out.contains_key(&key) {
                    continue;
                }

                let value = if omitted.contains(&pol.as_str()) {
                    derived
                        .get(&key)
                        .cloned()
                        .with_context(|| format!("{key:?} is not in the proof and was not reconstructed"))?
                } else {
                    let raw =
                        proof.evaluations.get(&key).with_context(|| format!("proof is missing evaluation {key:?}"))?;
                    fr::from_decimal(raw).with_context(|| format!("evaluation {key:?}"))?
                };

                out.insert(key, value);
            }
        }
    }

    for key in derived.keys() {
        if proof.evaluations.contains_key(key) {
            bail!("{key:?} was reconstructed but the proof also supplies it");
        }
        if !out.contains_key(key) {
            bail!("{key:?} was reconstructed but no f_i opens it");
        }
    }

    Ok(out)
}

/// The packed polynomial's value at one of its roots.
///
/// `f_i(X) = Σ_j X^j · pol_j(X^nPols)`, and a root of the opening point `p`
/// satisfies `root^nPols = xi·w^p`, so the inner evaluations are exactly the
/// openings the proof claims at that point. The packing is what lets one
/// commitment answer `nPols` opening claims.
pub fn f_at_root(f: &ShPlonkPol, evals: &Evaluations, root: &BigUint, opening_point: u32) -> Result<BigUint> {
    let mut acc = BigUint::from(0u32);
    let mut power = BigUint::one();

    for pol in &f.pols {
        let key = evaluation_key(pol, opening_point);
        let value = evals.get(&key).with_context(|| format!("no evaluation for {key:?}"))?;
        acc = fr::add(&acc, &fr::mul(value, &power));
        power = fr::mul(&power, root);
    }

    Ok(acc)
}

/// `R_i(y)`: interpolate `f_i` through its opening set, then evaluate at `y`.
///
/// The prover interpolates the polynomial's true values; the verifier
/// interpolates the *claimed* ones. They agree exactly when the claims are
/// honest, which is what the pairing then checks.
pub fn r_at(f: &ShPlonkPol, set: &OpeningSet, evals: &Evaluations, y: &BigUint) -> Result<BigUint> {
    let n = f.pols.len();
    let mut xs = Vec::with_capacity(set.roots.len());
    let mut ys = Vec::with_capacity(set.roots.len());

    for (k, &point) in f.opening_points.iter().enumerate() {
        for root in set.slot(k, n) {
            xs.push(root.clone());
            ys.push(f_at_root(f, evals, root, point)?);
        }
    }

    fr::lagrange_eval(&xs, &ys, y).with_context(|| format!("interpolating R{}", f.index))
}

/// The scalars `computeL` needs, all evaluated at the challenge `y`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linearisation {
    /// `Z_{S_i}(y)`, one per `f_i`.
    pub z_s: Vec<BigUint>,
    /// `Z_T(y)`, the zerofier over every root.
    pub z_t: BigUint,
    /// `preL[i] = alpha^i · ∏_{j≠i} Z_{S_j}(y)`.
    pub pre_l: Vec<BigUint>,
    /// `R_i(y)`.
    pub r: Vec<BigUint>,
}

impl Linearisation {
    /// `Σ_i preL[i] · R_i(y)`, the scalar the pairing's `E` term commits to.
    pub fn e_scalar(&self) -> BigUint {
        self.pre_l.iter().zip(&self.r).fold(BigUint::from(0u32), |acc, (p, r)| fr::add(&acc, &fr::mul(p, r)))
    }
}

/// Compute every linearisation scalar.
pub fn linearise(
    setup: &ShPlonkSetup,
    sets: &[OpeningSet],
    evals: &Evaluations,
    alpha: &BigUint,
    y: &BigUint,
) -> Result<Linearisation> {
    if setup.f.len() != sets.len() {
        bail!("{} opening sets for {} combined polynomials", sets.len(), setup.f.len());
    }

    let z_s: Vec<BigUint> = sets.iter().map(|s| fr::zerofier_at(&s.roots, y)).collect();

    // Z_T is the zerofier over the concatenation of every root, which is
    // exactly the product of the per-group zerofiers.
    let z_t = z_s.iter().fold(BigUint::one(), |acc, z| fr::mul(&acc, z));

    // preL[i] = alpha^i * prod_{j != i} z_s[j], built as a product rather than
    // as z_t / z_s[i]: a challenge y that happens to land on a root makes one
    // z_s[i] zero, and the quotient form would divide by it.
    let mut pre_l = Vec::with_capacity(z_s.len());
    let mut alpha_i = BigUint::one();
    for i in 0..z_s.len() {
        let mut acc = alpha_i.clone();
        for (j, z) in z_s.iter().enumerate() {
            if i != j {
                acc = fr::mul(&acc, z);
            }
        }
        pre_l.push(acc);
        alpha_i = fr::mul(&alpha_i, alpha);
    }

    let r = setup.f.iter().zip(sets).map(|(f, set)| r_at(f, set, evals, y)).collect::<Result<Vec<_>>>()?;

    Ok(Linearisation { z_s, z_t, pre_l, r })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{self, Reference};
    use crate::roots::all_roots;
    use num_traits::Zero;

    fn real() -> Reference {
        reference::load()
    }

    fn resolved(r: &Reference) -> Evaluations {
        resolve_evaluations(&r.setup, &r.proof, &r.derived()).unwrap()
    }

    fn lin(r: &Reference, evals: &Evaluations) -> Linearisation {
        let sets = all_roots(&r.setup, &r.xi_seed).unwrap();
        linearise(&r.setup, &sets, evals, &r.alpha, &r.y).unwrap()
    }

    /// The external check: every scalar must equal the prover's. Given the
    /// quotient evaluation the AIR layer will reconstruct, the whole Fr side of
    /// the opening is reproduced exactly.
    #[test]
    fn matches_the_reference_scalars() {
        let r = real();
        let l = lin(&r, &resolved(&r));

        assert_eq!(l.z_s, r.z_s, "Z_S(y)");
        assert_eq!(l.z_t, r.z_t, "Z_T(y)");
        assert_eq!(l.pre_l, r.pre_l, "preL");
        assert_eq!(l.r, r.r_at_y, "R_i(y)");
    }

    /// The packing convention, checked against the prover's own evaluations of
    /// the combined polynomials at their roots. A wrong slot order would still
    /// interpolate consistently, so only this catches it.
    #[test]
    fn the_packed_values_match_the_reference() {
        let r = real();
        let evals = resolved(&r);
        let sets = all_roots(&r.setup, &r.xi_seed).unwrap();

        for ((f, set), want) in r.setup.f.iter().zip(&sets).zip(&r.f_at_roots) {
            let n = f.pols.len();
            let mut got = Vec::new();
            for (k, &point) in f.opening_points.iter().enumerate() {
                for root in set.slot(k, n) {
                    got.push(f_at_root(f, &evals, root, point).unwrap());
                }
            }
            assert_eq!(&got, want, "f{} evaluated at its roots", f.index);
        }
    }

    /// The quotient's evaluation is the one input the opening scheme cannot
    /// derive. Feeding a wrong one changes the scalars, which is what makes the
    /// AIR layer's reconstruction load-bearing rather than decorative.
    #[test]
    fn the_reconstructed_quotient_is_load_bearing() {
        let r = real();
        let mut wrong = r.derived();
        *wrong.get_mut("Q").unwrap() = BigUint::from(1u32);

        let evals = resolve_evaluations(&r.setup, &r.proof, &wrong).unwrap();
        assert_ne!(lin(&r, &evals).r, r.r_at_y);
    }

    #[test]
    fn resolves_every_opening_slot() {
        let r = real();
        let evals = resolved(&r);
        // 42 slots, but f3/f4/f5 name distinct polynomials, so the keys are
        // distinct too: 41 from the proof plus the reconstructed Q.
        assert_eq!(evals.len(), 41 + 1);
        assert_eq!(evals["Q"], r.quotient_evaluation);
        assert!(!r.proof.evaluations.contains_key("Q"));
        // inv/invZh are hints, not openings, so they are not resolved here.
        assert!(!evals.contains_key("inv"));
    }

    #[test]
    fn requires_the_omitted_evaluation_to_be_supplied() {
        let r = real();
        let err = resolve_evaluations(&r.setup, &r.proof, &Evaluations::new()).unwrap_err().to_string();
        assert!(err.contains("Q"), "{err}");
    }

    /// A caller must not be able to override an opening the proof already
    /// claims -- that would substitute a value the pairing never sees.
    #[test]
    fn rejects_a_reconstruction_that_overrides_the_proof() {
        let r = real();
        let mut d = r.derived();
        d.insert("Global.L1".into(), BigUint::from(1u32));
        assert!(resolve_evaluations(&r.setup, &r.proof, &d).is_err());
    }

    #[test]
    fn rejects_a_reconstruction_nothing_opens() {
        let r = real();
        let mut d = r.derived();
        d.insert("NotAPolynomial".into(), BigUint::from(1u32));
        assert!(resolve_evaluations(&r.setup, &r.proof, &d).is_err());
    }

    /// Z_T is the zerofier over the concatenation, so it factors exactly into
    /// the per-group zerofiers.
    #[test]
    fn z_t_is_the_product_of_the_group_zerofiers() {
        let r = real();
        let l = lin(&r, &resolved(&r));

        let product = l.z_s.iter().fold(BigUint::one(), |a, z| fr::mul(&a, z));
        assert_eq!(l.z_t, product);

        // And it really is the zerofier of the flattened multiset.
        let all = crate::roots::flattened(&all_roots(&r.setup, &r.xi_seed).unwrap());
        assert_eq!(l.z_t, fr::zerofier_at(&all, &r.y));
    }

    /// The relation that defines preL. Checking it as a product avoids
    /// restating the loop that computes it.
    #[test]
    fn pre_l_times_its_own_zerofier_is_alpha_i_times_z_t() {
        let r = real();
        let l = lin(&r, &resolved(&r));

        let mut alpha_i = BigUint::one();
        for i in 0..l.pre_l.len() {
            assert_eq!(fr::mul(&l.pre_l[i], &l.z_s[i]), fr::mul(&alpha_i, &l.z_t), "i = {i}");
            alpha_i = fr::mul(&alpha_i, &r.alpha);
        }
    }

    /// R_i interpolates the packed values, so it must reproduce them at the
    /// roots. This is what ties the claimed openings to the committed f_i.
    #[test]
    fn r_reproduces_the_packed_values_at_every_root() {
        let r = real();
        let evals = resolved(&r);
        let sets = all_roots(&r.setup, &r.xi_seed).unwrap();

        for (f, set) in r.setup.f.iter().zip(&sets) {
            let n = f.pols.len();
            for (k, &point) in f.opening_points.iter().enumerate() {
                for root in set.slot(k, n) {
                    let want = f_at_root(f, &evals, root, point).unwrap();
                    assert_eq!(r_at(f, set, &evals, root).unwrap(), want, "f{} at a root", f.index);
                }
            }
        }
    }

    /// A group opened at one point with one polynomial has a constant R.
    #[test]
    fn a_single_slot_group_interpolates_to_its_own_value() {
        let r = real();
        let evals = resolved(&r);
        let sets = all_roots(&r.setup, &r.xi_seed).unwrap();

        let i = r.setup.f.iter().position(|f| f.pols.len() == 1 && f.opening_points.len() == 1).unwrap();
        assert_eq!(r.setup.f[i].pols, vec!["Q".to_string()]);
        assert_eq!(r_at(&r.setup.f[i], &sets[i], &evals, &r.y).unwrap(), evals["Q"]);
    }

    /// Every claimed evaluation reaches the scalars: changing one must move
    /// R_i(y), or that opening would go unchecked by the pairing.
    #[test]
    fn every_evaluation_reaches_the_linearisation() {
        let r = real();
        let evals = resolved(&r);
        let baseline = lin(&r, &evals);

        for key in ["Global.L1", "Plookup.Bw", "Im28", "Plookup.Z0w", "Q"] {
            let mut tampered = evals.clone();
            *tampered.get_mut(key).unwrap() = fr::add(&evals[key], &BigUint::one());
            assert_ne!(lin(&r, &tampered).r, baseline.r, "{key} did not move any R_i");
        }
    }

    #[test]
    fn e_scalar_is_the_weighted_sum_of_the_r_values() {
        let r = real();
        let l = lin(&r, &resolved(&r));

        let mut want = BigUint::zero();
        for (p, v) in l.pre_l.iter().zip(&l.r) {
            want = fr::add(&want, &fr::mul(p, v));
        }
        assert_eq!(l.e_scalar(), want);
    }

    #[test]
    fn rejects_a_mismatched_number_of_opening_sets() {
        let r = real();
        let evals = resolved(&r);
        let sets = all_roots(&r.setup, &r.xi_seed).unwrap();
        assert!(linearise(&r.setup, &sets[..3], &evals, &r.alpha, &r.y).is_err());
    }
}
