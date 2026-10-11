//! The SHPLONK pairing check, assembled but not executed.
//!
//! The prover's `computeL` builds
//!
//! ```text
//! L(X) = Σ_i preL[i]·(f_i(X) - R_i(y)) - Z_T(y)·W(X)
//! ```
//!
//! and `computeWp` divides it by `X - y`, which only succeeds if `L(y) = 0` --
//! that is, if every claimed opening is consistent. The verifier cannot divide
//! polynomials it does not have, so it checks the same statement in the
//! exponent. From `L(X) = (X - y)·Wp(X)`:
//!
//! ```text
//! e(A, [1]₂) = e([Wp]₁, [x]₂),   A = Σ_i preL[i]·[f_i]₁ - (Σ_i preL[i]·R_i(y))·G₁
//!                                    - Z_T(y)·[W]₁ + y·[Wp]₁
//! ```
//!
//! The `y·[Wp]` term is what moves the `-y` off the G2 side, so the verifier
//! pairs against `[x]₂` from the key rather than needing `[x - y]₂`.
//!
//! This module stops at `A`'s definition: it produces the scalars and points,
//! leaving the multi-scalar multiplication and the pairing to a backend. That
//! split is not just tidiness -- everything here is exact and testable with
//! integer arithmetic, while the part left out needs a curve library.

use anyhow::{Context, Result};
use num_bigint::BigUint;

use crate::curve::{G1Affine, G2Affine, g1_generator};
use crate::fr;
use crate::linearisation::Linearisation;
use crate::proof::{W_KEY, WP_KEY, commitment_key};
use crate::setup::ShPlonkSetup;
use crate::verifier::{Challenges, commitment_for};
use crate::ShPlonkProof;

/// One `scalar · point` contribution to `A`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    /// Reduced scalar in `Fr`. Subtractions are folded in as `r - s`, so the
    /// whole of `A` is a single multi-scalar multiplication.
    pub scalar: BigUint,
    pub point: G1Affine,
    /// What this term is, for error messages.
    pub label: String,
}

/// The assembled check: `e(Σ a, [1]₂) == e(b, x2)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairingCheck {
    pub a: Vec<Term>,
    pub b: G1Affine,
    pub x2: G2Affine,
}

/// Build the pairing check from the recomputed challenges and scalars.
///
/// Commitments come from [`commitment_for`], so the constant-only `f_i` are
/// taken from the key rather than from the proof.
pub fn assemble(
    setup: &ShPlonkSetup,
    proof: &ShPlonkProof,
    lin: &Linearisation,
    challenges: &Challenges,
) -> Result<PairingCheck> {
    let mut a = Vec::with_capacity(setup.f.len() + 3);

    // F: the committed combined polynomials, weighted by preL.
    for (f, scalar) in setup.f.iter().zip(&lin.pre_l) {
        let key = commitment_key(f.index);
        let point = commitment_for(setup, proof, f.index)?;
        a.push(Term { scalar: scalar.clone(), point: G1Affine::parse_checked(point, &key)?, label: key });
    }

    // -E: the claimed openings, collapsed onto the generator. Negated so the
    // whole sum is one MSM rather than an MSM and a subtraction.
    a.push(Term { scalar: fr::sub(&BigUint::from(0u32), &lin.e_scalar()), point: g1_generator(), label: "-E".into() });

    // -J: the batched opening proof, weighted by the full zerofier.
    let w = proof.polynomials.get(W_KEY).context("proof is missing the W commitment")?;
    a.push(Term {
        scalar: fr::sub(&BigUint::from(0u32), &lin.z_t),
        point: G1Affine::parse_checked(w, W_KEY)?,
        label: "-J".into(),
    });

    // +y·Wp, which turns [x - y]₂ into [x]₂ on the other side.
    let wp = proof.polynomials.get(WP_KEY).context("proof is missing the Wp commitment")?;
    let wp = G1Affine::parse_checked(wp, WP_KEY)?;
    a.push(Term { scalar: challenges.y.clone(), point: wp.clone(), label: "y*Wp".into() });

    Ok(PairingCheck { a, b: wp, x2: G2Affine::from_json(&setup.x2).context("verification key X_2")? })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linearisation::{linearise, resolve_evaluations};
    use crate::reference::{self, Reference};
    use crate::roots::all_roots;
    use crate::verifier::recompute_challenges;

    fn built() -> (Reference, PairingCheck) {
        let r = reference::load();
        let challenges = recompute_challenges(r.previous_challenge.clone(), &r.setup, &r.proof).unwrap();
        let evals = resolve_evaluations(&r.setup, &r.proof, &r.derived()).unwrap();
        let sets = all_roots(&r.setup, &challenges.xi_seed).unwrap();
        let lin = linearise(&r.setup, &sets, &evals, &challenges.alpha, &challenges.y).unwrap();
        let check = assemble(&r.setup, &r.proof, &lin, &challenges).unwrap();
        (r, check)
    }

    /// Every commitment the key or proof supplies appears exactly once, plus
    /// the three terms the equation adds.
    #[test]
    fn every_commitment_enters_the_check() {
        let (r, c) = built();
        assert_eq!(c.a.len(), r.setup.f.len() + 3);

        let labels: Vec<&str> = c.a.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(&labels[..r.setup.f.len()], &["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]);
        assert_eq!(&labels[r.setup.f.len()..], &["-E", "-J", "y*Wp"]);
    }

    /// The weights on the f_i are exactly preL, and the last three terms are
    /// the ones the equation prescribes.
    #[test]
    fn the_scalars_are_the_linearisation_scalars() {
        let (r, c) = built();
        let challenges = recompute_challenges(r.previous_challenge.clone(), &r.setup, &r.proof).unwrap();
        let evals = resolve_evaluations(&r.setup, &r.proof, &r.derived()).unwrap();
        let sets = all_roots(&r.setup, &challenges.xi_seed).unwrap();
        let lin = linearise(&r.setup, &sets, &evals, &challenges.alpha, &challenges.y).unwrap();

        for (term, want) in c.a.iter().zip(&lin.pre_l) {
            assert_eq!(&term.scalar, want, "{}", term.label);
        }

        let n = r.setup.f.len();
        // A negated scalar plus the value it negates is zero in Fr.
        assert_eq!(fr::add(&c.a[n].scalar, &lin.e_scalar()), BigUint::from(0u32));
        assert_eq!(fr::add(&c.a[n + 1].scalar, &lin.z_t), BigUint::from(0u32));
        assert_eq!(c.a[n + 2].scalar, challenges.y);
    }

    /// Terms are `scalar · point` with the scalar reduced and the point real:
    /// what an MSM backend is entitled to assume.
    #[test]
    fn the_terms_are_well_formed_msm_inputs() {
        let (_, c) = built();
        for t in &c.a {
            assert!(t.scalar < crate::transcript::fr_modulus(), "{} is not reduced", t.label);
            assert!(t.point.is_on_curve(), "{} is not on the curve", t.label);
        }
        assert!(c.b.is_on_curve());
    }

    /// The constant-only f_i are taken from the key. Substituting them in the
    /// proof must not change the point the check uses.
    #[test]
    fn constant_commitments_are_taken_from_the_key() {
        let (r, honest) = built();

        let mut proof = r.proof.clone();
        proof.polynomials.get_mut("f0").unwrap()[0] = "7".to_string();

        let challenges = recompute_challenges(r.previous_challenge.clone(), &r.setup, &r.proof).unwrap();
        let evals = resolve_evaluations(&r.setup, &proof, &r.derived()).unwrap();
        let sets = all_roots(&r.setup, &challenges.xi_seed).unwrap();
        let lin = linearise(&r.setup, &sets, &evals, &challenges.alpha, &challenges.y).unwrap();
        let tampered = assemble(&r.setup, &proof, &lin, &challenges).unwrap();

        assert_eq!(tampered.a[0].point, honest.a[0].point, "f0 followed the proof instead of the key");
    }

    /// The G2 side is the key's X_2, not the generator: pairing against the
    /// generator would check `L = Wp`, which is not the statement.
    #[test]
    fn the_g2_side_is_the_keys_x2() {
        let (r, c) = built();
        assert_eq!(c.x2, G2Affine::from_json(&r.setup.x2).unwrap());
        assert_ne!(c.x2, crate::curve::g2_generator());
    }

    #[test]
    fn rejects_a_proof_with_an_off_curve_commitment() {
        let r = reference::load();
        let challenges = recompute_challenges(r.previous_challenge.clone(), &r.setup, &r.proof).unwrap();
        let evals = resolve_evaluations(&r.setup, &r.proof, &r.derived()).unwrap();
        let sets = all_roots(&r.setup, &challenges.xi_seed).unwrap();
        let lin = linearise(&r.setup, &sets, &evals, &challenges.alpha, &challenges.y).unwrap();

        let mut proof = r.proof.clone();
        proof.polynomials.get_mut("f8").unwrap()[1] = "3".to_string();
        assert!(assemble(&r.setup, &proof, &lin, &challenges).is_err());
    }
}
