//! The verifier, end to end, up to the pairing.
//!
//! Everything a verifier derives from a proof, in dependency order:
//!
//! ```text
//! publics + commitments ──► AIR challenges ──► xiSeed ──► xi ──► cExp(xi) ──► Q(xi)
//!                                                 └─────► alpha ──► y           │
//!                                                           │                   ▼
//!                                                           └──► R_i(y), Z_T(y), preL
//!                                                                              │
//!                                                                              ▼
//!                                                                       pairing check
//! ```
//!
//! Nothing here is taken from the proof except commitments and claimed
//! openings: every challenge is recomputed -- the AIR's own chain included --
//! the quotient is reconstructed from the constraint system, and the constant
//! commitments come from the key.
//!
//! The last step is not executed. [`prepare`] returns the assembled
//! [`PairingCheck`]; running it needs a curve library, which this crate does
//! not yet depend on. Until then a caller can check everything that precedes
//! it, which is the whole of the Fiat-Shamir and constraint reasoning.

use anyhow::{Context, Result};
use num_bigint::BigUint;

use crate::air::{self, ChallengeSchedule};
use crate::linearisation::{Linearisation, linearise, resolve_evaluations};
use crate::pairing::{PairingCheck, assemble};
use crate::proof::ShPlonkProof;
use crate::roots::{OpeningSet, all_roots};
use crate::setup::ShPlonkSetup;
use crate::verifier::{Challenges, non_committed_pols, recompute_challenges};
use crate::verifier_code::{Inputs, VerifierCode};

/// What the AIR contributes, as opposed to the opening scheme.
///
/// The challenges are no longer among these: they are derived from the proof
/// and the public inputs, so nothing about the transcript is taken on trust.
pub struct AirInputs<'a> {
    /// `log2` of the trace length, for the vanishing polynomial.
    pub n_bits: u32,
    pub publics: &'a [BigUint],
    /// How many challenges each stage draws. A property of the AIR.
    pub schedule: &'a ChallengeSchedule,
}

/// A verified derivation, with the pairing left to execute.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// The AIR's own challenges, derived from the commitments and publics.
    pub air_challenges: Vec<BigUint>,
    pub challenges: Challenges,
    /// `Q(xi)`, reconstructed rather than read.
    pub quotient: BigUint,
    pub roots: Vec<OpeningSet>,
    pub linearisation: Linearisation,
    /// The remaining obligation: `e(Σ a, [1]₂) == e(b, x2)`.
    pub check: PairingCheck,
}

/// Derive everything a verifier can without curve arithmetic.
///
/// Returns an error the moment anything fails to reconcile, so a caller that
/// gets a [`Prepared`] back knows every step up to the pairing agreed.
pub fn prepare(
    setup: &ShPlonkSetup,
    proof: &ShPlonkProof,
    code: &VerifierCode,
    air_inputs: &AirInputs,
) -> Result<Prepared> {
    // The AIR's chain first: it binds the publics and every stage commitment,
    // and its last challenge is what seeds the opening.
    let air = air::air_challenges(setup, proof, air_inputs.publics, air_inputs.schedule)?;
    let previous = air.last().cloned().context("the AIR schedule draws no challenges")?;

    // Then the opening's, which also runs the structural and
    // constant-commitment checks.
    let challenges = recompute_challenges(previous, setup, proof)?;

    // Reconstruct the quotient from the constraint system. The proof omits it
    // precisely so that it cannot be claimed.
    let evals = code.evaluations(&setup.pols_map, proof).context("reading the openings the constraints need")?;
    let c_exp = code
        .evaluate(&Inputs { evals: &evals, challenges: &air, publics: air_inputs.publics, x: &challenges.xi })
        .context("evaluating the constraint expression at xi")?;

    let quotient = air::quotient_at(&c_exp, &air::inv_zh(proof)?, &challenges.xi, air_inputs.n_bits)?;

    // Hand it to the opening scheme as the evaluation the proof left out.
    let mut derived = crate::linearisation::Evaluations::new();
    for pol in non_committed_pols(setup) {
        derived.insert(pol.to_string(), quotient.clone());
    }
    let openings = resolve_evaluations(setup, proof, &derived)?;

    let roots = all_roots(setup, &challenges.xi_seed)?;
    let linearisation = linearise(setup, &roots, &openings, &challenges.alpha, &challenges.y)?;
    let check = assemble(setup, proof, &linearisation, &challenges)?;

    Ok(Prepared { air_challenges: air, challenges, quotient, roots, linearisation, check })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{self, Reference};

    const INFO: &str = include_str!("../tests/fixtures/pilfflonk.verifierinfo.json");

    fn run(r: &Reference, proof: &ShPlonkProof) -> Result<Prepared> {
        run_with(r, proof, &r.publics)
    }

    fn run_with(r: &Reference, proof: &ShPlonkProof, publics: &[BigUint]) -> Result<Prepared> {
        let info: serde_json::Value = serde_json::from_str(INFO).unwrap();
        let code = VerifierCode::from_json(&info).unwrap();
        let n_bits = info["pilPower"].as_u64().unwrap() as u32;

        prepare(&r.setup, proof, &code, &AirInputs { n_bits, publics, schedule: &ChallengeSchedule(vec![2, 2, 1]) })
    }

    /// The whole derivation against the prover's own values. Only the public
    /// inputs and the AIR's shape are given; everything else -- every
    /// challenge, the quotient, every opening scalar -- is recomputed and must
    /// agree.
    #[test]
    fn derives_everything_the_prover_computed() {
        let r = reference::load();
        let p = run(&r, &r.proof).unwrap();

        assert_eq!(p.air_challenges, r.air_challenges, "the AIR chain");
        assert_eq!(p.challenges.xi_seed, r.xi_seed);
        assert_eq!(p.challenges.xi, r.xi);
        assert_eq!(p.challenges.alpha, r.alpha);
        assert_eq!(p.challenges.y, r.y);

        assert_eq!(p.quotient, r.quotient_evaluation, "reconstructed quotient");

        for (set, want) in p.roots.iter().zip(&r.roots) {
            assert_eq!(&set.roots, want);
        }
        assert_eq!(p.linearisation.z_s, r.z_s);
        assert_eq!(p.linearisation.z_t, r.z_t);
        assert_eq!(p.linearisation.pre_l, r.pre_l);
        assert_eq!(p.linearisation.r, r.r_at_y);

        assert_eq!(p.check.a.len(), r.setup.f.len() + 3);
    }

    /// The quotient is reconstructed, not read: the proof carries no value for
    /// it, and the reconstruction is what ties the opening to the constraints.
    #[test]
    fn the_quotient_is_not_in_the_proof() {
        let r = reference::load();
        assert!(!r.proof.evaluations.contains_key("Q"));
        assert_eq!(run(&r, &r.proof).unwrap().quotient, r.quotient_evaluation);
    }

    /// Tampering with any claimed opening changes the derivation. Each of
    /// these reaches the result by a different route -- through the constraint
    /// expression, through alpha, or through both.
    #[test]
    fn every_tampered_opening_changes_the_derivation() {
        let r = reference::load();
        let honest = run(&r, &r.proof).unwrap();

        for key in ["Global.L1", "Plookup.Bw", "Im28", "Plookup.Z0w"] {
            let mut proof = r.proof.clone();
            let bumped =
                crate::fr::add(&crate::fr::from_decimal(&proof.evaluations[key]).unwrap(), &BigUint::from(1u32));
            *proof.evaluations.get_mut(key).unwrap() = bumped.to_str_radix(10);

            let tampered = run(&r, &proof).unwrap();
            assert!(
                tampered.challenges.alpha != honest.challenges.alpha
                    || tampered.quotient != honest.quotient
                    || tampered.linearisation.r != honest.linearisation.r,
                "{key} changed nothing"
            );
        }
    }

    /// A forged invZh is caught before the opening scheme sees it.
    #[test]
    fn rejects_a_forged_vanishing_inverse() {
        let r = reference::load();
        let mut proof = r.proof.clone();
        proof.evaluations.insert("invZh".into(), "1".into());

        let err = format!("{:#}", run(&r, &proof).unwrap_err());
        assert!(err.contains("invZh"), "{err}");
    }

    /// The public inputs are part of the statement, so the same proof must not
    /// verify against a different one.
    ///
    /// Since the publics seed the challenge chain, changing one moves `xi`, and
    /// the proof's `invZh` -- computed for the honest `xi` -- stops being the
    /// inverse it claims to be. The proof is rejected rather than merely
    /// producing a different quotient, which is the stronger outcome: it fails
    /// before any opening is considered.
    #[test]
    fn a_proof_does_not_verify_against_different_publics() {
        let r = reference::load();
        let honest = run(&r, &r.proof).unwrap();

        let mut publics = r.publics.clone();
        publics[0] = crate::fr::add(&publics[0], &BigUint::from(1u32));

        let err = format!("{:#}", run_with(&r, &r.proof, &publics).unwrap_err());
        assert!(err.contains("invZh"), "{err}");

        // The rejection is a consequence of the chain moving, not a
        // coincidence: every challenge differs.
        let moved = air::air_challenges(&r.setup, &r.proof, &publics, &ChallengeSchedule(vec![2, 2, 1])).unwrap();
        assert!(moved.iter().zip(&honest.air_challenges).all(|(a, b)| a != b));
    }

    /// The deterministic proof verifies too.
    ///
    /// It is produced with blinding zeroed, so it is reproducible and its
    /// per-stage commitments can be recomputed from the trace -- which is what
    /// makes the prover testable. This confirms it is a genuine proof and not
    /// merely a reproducible artefact: every challenge, the quotient and every
    /// opening scalar reconcile, exactly as for the blinded one.
    #[test]
    fn the_deterministic_proof_verifies() {
        let r = reference::load_deterministic();
        let p = run(&r, &r.proof).unwrap();

        assert_eq!(p.air_challenges, r.air_challenges);
        assert_eq!(p.challenges.xi, r.xi);
        assert_eq!(p.challenges.alpha, r.alpha);
        assert_eq!(p.challenges.y, r.y);
        assert_eq!(p.quotient, r.quotient_evaluation);
        assert_eq!(p.linearisation.r, r.r_at_y);
        assert_eq!(p.linearisation.z_t, r.z_t);
    }

    /// Blinding is what separates the two, and it changes the commitments
    /// without changing what is being proved: same key, same publics, and the
    /// verifier accepts both.
    #[test]
    fn blinding_changes_the_proof_but_not_its_validity() {
        let blinded = reference::load();
        let plain = reference::load_deterministic();

        assert_eq!(blinded.publics, plain.publics, "the same statement");
        assert_ne!(blinded.proof.polynomials["f8"], plain.proof.polynomials["f8"]);
        assert_ne!(blinded.xi, plain.xi, "different commitments give different challenges");

        assert!(run(&blinded, &blinded.proof).is_ok());
        assert!(run(&plain, &plain.proof).is_ok());
    }

    /// A proof that fails a structural check never reaches the arithmetic.
    #[test]
    fn rejects_a_malformed_proof() {
        let r = reference::load();
        let mut proof = r.proof.clone();
        proof.polynomials.remove("Wp");
        assert!(run(&r, &proof).is_err());
    }
}
