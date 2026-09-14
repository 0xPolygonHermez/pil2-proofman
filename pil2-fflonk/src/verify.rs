//! The verifier, end to end, up to the pairing.
//!
//! Everything a verifier derives from a proof, in dependency order:
//!
//! ```text
//! previous challenge ──► xiSeed ──► xi ──► cExp(xi) ──► Q(xi)
//!                           └─────► alpha ──► y            │
//!                                     │                    ▼
//!                                     └──────────► R_i(y), Z_T(y), preL
//!                                                          │
//!                                                          ▼
//!                                                   pairing check
//! ```
//!
//! Nothing here is taken from the proof except commitments and claimed
//! openings: every challenge is recomputed, the quotient is reconstructed from
//! the constraint system, and the constant commitments come from the key.
//!
//! The last step is not executed. [`prepare`] returns the assembled
//! [`PairingCheck`]; running it needs a curve library, which this crate does
//! not yet depend on. Until then a caller can check everything that precedes
//! it, which is the whole of the Fiat-Shamir and constraint reasoning.

use anyhow::{Context, Result};
use num_bigint::BigUint;

use crate::air;
use crate::linearisation::{Linearisation, linearise, resolve_evaluations};
use crate::pairing::{PairingCheck, assemble};
use crate::proof::ShPlonkProof;
use crate::roots::{OpeningSet, all_roots};
use crate::setup::ShPlonkSetup;
use crate::verifier::{Challenges, non_committed_pols, recompute_challenges};
use crate::verifier_code::{Inputs, VerifierCode};

/// What the AIR contributes, as opposed to the opening scheme.
pub struct AirInputs<'a> {
    /// `log2` of the trace length, for the vanishing polynomial.
    pub n_bits: u32,
    /// The AIR's challenges, in the order the protocol draws them.
    pub challenges: &'a [BigUint],
    pub publics: &'a [BigUint],
    /// The last challenge of the phase before the opening, which seeds `xi`.
    pub previous_challenge: BigUint,
}

/// A verified derivation, with the pairing left to execute.
#[derive(Clone, Debug)]
pub struct Prepared {
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
    // Challenges first: everything downstream depends on them, and this also
    // runs the structural and constant-commitment checks.
    let challenges = recompute_challenges(air_inputs.previous_challenge.clone(), setup, proof)?;

    // Reconstruct the quotient from the constraint system. The proof omits it
    // precisely so that it cannot be claimed.
    let evals = code.evaluations(&setup.pols_map, proof).context("reading the openings the constraints need")?;
    let c_exp = code
        .evaluate(&Inputs {
            evals: &evals,
            challenges: air_inputs.challenges,
            publics: air_inputs.publics,
            x: &challenges.xi,
        })
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

    Ok(Prepared { challenges, quotient, roots, linearisation, check })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{self, Reference};

    const INFO: &str = include_str!("../tests/fixtures/pilfflonk.verifierinfo.json");

    fn run(r: &Reference, proof: &ShPlonkProof) -> Result<Prepared> {
        let info: serde_json::Value = serde_json::from_str(INFO).unwrap();
        let code = VerifierCode::from_json(&info).unwrap();
        let n_bits = info["pilPower"].as_u64().unwrap() as u32;

        prepare(
            &r.setup,
            proof,
            &code,
            &AirInputs {
                n_bits,
                challenges: &r.air_challenges,
                publics: &r.publics,
                previous_challenge: r.previous_challenge.clone(),
            },
        )
    }

    /// The whole derivation against the prover's own values. Only the
    /// previous challenge and the public inputs are given; everything else --
    /// challenges, the quotient, every opening scalar -- is recomputed and
    /// must agree.
    #[test]
    fn derives_everything_the_prover_computed() {
        let r = reference::load();
        let p = run(&r, &r.proof).unwrap();

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

    /// The public inputs are part of the statement: proving a different claim
    /// must not reuse the same proof.
    #[test]
    fn the_public_inputs_reach_the_quotient() {
        let r = reference::load();
        let honest = run(&r, &r.proof).unwrap();

        let mut publics = r.publics.clone();
        publics[0] = crate::fr::add(&publics[0], &BigUint::from(1u32));

        let info: serde_json::Value = serde_json::from_str(INFO).unwrap();
        let code = VerifierCode::from_json(&info).unwrap();
        let other = prepare(
            &r.setup,
            &r.proof,
            &code,
            &AirInputs {
                n_bits: info["pilPower"].as_u64().unwrap() as u32,
                challenges: &r.air_challenges,
                publics: &publics,
                previous_challenge: r.previous_challenge.clone(),
            },
        )
        .unwrap();

        assert_ne!(other.quotient, honest.quotient);
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
