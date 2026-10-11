//! Reconstructing the quotient evaluation the opening scheme omits.
//!
//! The AIR's constraints hold on the trace domain exactly when the combined
//! constraint expression vanishes there, which is to say when it is divisible
//! by the domain's vanishing polynomial:
//!
//! ```text
//! cExp(X) = Q(X) · Z_H(X),    Z_H(X) = X^N - 1
//! ```
//!
//! So `Q(xi) = cExp(xi) / Z_H(xi)`. The prover sends the inverse rather than
//! the quotient, and the verifier recovers the value it needs with one
//! multiplication instead of a field inversion.
//!
//! That inverse is a prover-supplied hint, so it is checked here rather than
//! trusted: a prover free to choose `invZh` could scale the quotient to
//! whatever value made its opening consistent, which is the whole constraint
//! system. Checking costs one multiplication and one exponentiation.
//!
//! This is the one input [`crate::linearisation`] cannot derive on its own,
//! and it is why that module takes reconstructed evaluations as a parameter.

use anyhow::{Result, bail};
use num_bigint::BigUint;
use num_traits::One;

use crate::fr;
use crate::proof::{INV_ZH_KEY, ShPlonkProof, commitment_key};
use crate::setup::ShPlonkSetup;
use crate::transcript::Transcript;

/// How many challenges are drawn after each stage's commitments are absorbed.
///
/// Entry `i` is for stage `i + 1`; stage 0 holds the constants, which are
/// absorbed with the public inputs before any challenge is drawn. The reference
/// AIR draws `[2, 2, 1]` -- alpha and beta, then gamma and delta, then `a` --
/// but the shape belongs to the AIR rather than to the opening scheme, so it is
/// a parameter rather than a constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeSchedule(pub Vec<usize>);

/// The AIR's challenge chain, in the order the protocol draws it.
///
/// Mirrors `PilFflonkProver`'s use of the transcript across `stage1`..`stage4`.
/// The chain has one uniform rule that is easy to miss in the source, where it
/// is spread over four functions: **every challenge resets the transcript and
/// reseeds it with itself**. So a stage's commitments are absorbed on top of
/// the previous challenge, never on top of an accumulated history.
///
/// The seed is built once, from the constant commitments and the public inputs,
/// which is what binds the statement to every challenge that follows.
pub fn air_challenges(
    setup: &ShPlonkSetup,
    proof: &ShPlonkProof,
    publics: &[BigUint],
    schedule: &ChallengeSchedule,
) -> Result<Vec<BigUint>> {
    // The schedule has to cover every stage between the constants and the
    // quotient. A short one would skip a stage's commitments and still return
    // a plausible list of challenges -- the wrong ones -- which downstream
    // shows up only as an unrelated-looking rejection.
    let quotient = crate::verifier::quotient_stage(setup);
    let expected = quotient.saturating_sub(1) as usize;
    if schedule.0.len() != expected {
        bail!(
            "the challenge schedule covers {} stages, but the AIR has {} between the constants and the quotient",
            schedule.0.len(),
            expected
        );
    }

    let mut t = Transcript::new();

    // Stage 0: the constant polynomials, then the publics.
    absorb_stage(&mut t, setup, proof, 0)?;
    for p in publics {
        t.add_scalar(p.clone());
    }

    let mut challenges = Vec::new();
    for (i, &count) in schedule.0.iter().enumerate() {
        absorb_stage(&mut t, setup, proof, i as u32 + 1)?;

        for _ in 0..count {
            let c = t.get_challenge();
            t.reset();
            t.add_scalar(c.clone());
            challenges.push(c);
        }
    }

    Ok(challenges)
}

/// Absorb every `f_i` belonging to one stage, in index order.
fn absorb_stage(t: &mut Transcript, setup: &ShPlonkSetup, proof: &ShPlonkProof, stage: u32) -> Result<()> {
    for f in &setup.f {
        if f.stages.first().map(|s| s.stage) != Some(stage) {
            continue;
        }
        let key = commitment_key(f.index);
        // The key's value wins for the constant stage, so a prover cannot
        // steer the challenges by substituting a commitment it does not own.
        let point = crate::verifier::commitment_for(setup, proof, f.index)?;
        t.add_commitment_json(point).map_err(|e| e.context(format!("absorbing {key}")))?;
    }
    Ok(())
}

/// `Z_H(x) = x^N - 1` for a domain of `2^n_bits` rows.
pub fn zh_at(x: &BigUint, n_bits: u32) -> BigUint {
    fr::sub(&x.modpow(&(BigUint::one() << n_bits), &crate::transcript::fr_modulus()), &BigUint::one())
}

/// Check the prover's claimed `1 / Z_H(xi)`.
///
/// A zero `Z_H(xi)` would mean the challenge landed on the trace domain, where
/// the quotient is not defined. That is negligibly unlikely with an honest
/// transcript, so meeting it means something is wrong rather than unlucky.
pub fn check_inv_zh(inv_zh: &BigUint, xi: &BigUint, n_bits: u32) -> Result<()> {
    let zh = zh_at(xi, n_bits);
    if zh == BigUint::from(0u32) {
        bail!("xi is a root of Z_H: the challenge landed on the trace domain");
    }
    if fr::mul(inv_zh, &zh) != BigUint::one() {
        bail!("the proof's invZh is not the inverse of Z_H(xi)");
    }
    Ok(())
}

/// `Q(xi) = cExp(xi) · invZh`, after checking the hint.
pub fn quotient_at(c_exp: &BigUint, inv_zh: &BigUint, xi: &BigUint, n_bits: u32) -> Result<BigUint> {
    check_inv_zh(inv_zh, xi, n_bits)?;
    Ok(fr::mul(c_exp, inv_zh))
}

/// Read `invZh` from a proof.
pub fn inv_zh(proof: &ShPlonkProof) -> Result<BigUint> {
    let raw = match proof.evaluations.get(INV_ZH_KEY) {
        Some(v) => v,
        None => bail!("proof is missing the {INV_ZH_KEY} evaluation"),
    };
    fr::from_decimal(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference;
    use crate::verifier_code::{Inputs, VerifierCode};

    const INFO: &str = include_str!("../tests/fixtures/pilfflonk.verifierinfo.json");

    fn n_bits() -> u32 {
        let info: serde_json::Value = serde_json::from_str(INFO).unwrap();
        info["pilPower"].as_u64().unwrap() as u32
    }

    /// The whole reconstruction, end to end, against the quotient the prover
    /// actually committed to. This is the last input the opening scheme needed
    /// that it could not derive for itself.
    #[test]
    fn reconstructs_the_provers_quotient_evaluation() {
        let r = reference::load();
        let info: serde_json::Value = serde_json::from_str(INFO).unwrap();
        let code = VerifierCode::from_json(&info).unwrap();

        let evals = code.evaluations(&r.setup.pols_map, &r.proof).unwrap();
        let c_exp = code
            .evaluate(&Inputs { evals: &evals, challenges: &r.air_challenges, publics: &r.publics, x: &r.xi })
            .unwrap();

        let got = quotient_at(&c_exp, &inv_zh(&r.proof).unwrap(), &r.xi, n_bits()).unwrap();
        assert_eq!(got, r.quotient_evaluation);
    }

    /// The prover's invZh really is the inverse of the vanishing polynomial at
    /// xi, for the domain the key declares.
    #[test]
    fn the_proofs_inv_zh_is_the_true_inverse() {
        let r = reference::load();
        assert_eq!(n_bits(), 8, "the reference AIR has 256 rows");
        check_inv_zh(&inv_zh(&r.proof).unwrap(), &r.xi, n_bits()).unwrap();
    }

    /// A forged inverse is rejected. Without this check a prover could pick
    /// invZh to make any constraint value produce whatever quotient its
    /// opening claimed.
    #[test]
    fn rejects_a_forged_inverse() {
        let r = reference::load();
        let honest = inv_zh(&r.proof).unwrap();

        for forged in [BigUint::from(1u32), fr::add(&honest, &BigUint::one()), BigUint::from(0u32)] {
            assert!(check_inv_zh(&forged, &r.xi, n_bits()).is_err());
            assert!(quotient_at(&BigUint::from(7u32), &forged, &r.xi, n_bits()).is_err());
        }
    }

    /// The domain size is part of the statement: the same proof read against a
    /// different N gives a different Z_H, and the hint no longer checks out.
    #[test]
    fn the_domain_size_is_bound_by_the_check() {
        let r = reference::load();
        let honest = inv_zh(&r.proof).unwrap();
        assert!(check_inv_zh(&honest, &r.xi, n_bits() + 1).is_err());
    }

    #[test]
    fn zh_vanishes_on_the_trace_domain() {
        let r = reference::load();
        let w = fr::from_decimal(&r.setup.omegas["w"]).unwrap();

        // w generates the trace domain, so every power of it is a root.
        for k in [0u64, 1, 2, 37, 255] {
            assert_eq!(zh_at(&fr::pow(&w, k), 8), BigUint::from(0u32), "w^{k}");
        }
        // And the challenge is not one of them.
        assert_ne!(zh_at(&r.xi, 8), BigUint::from(0u32));
    }

    /// A challenge on the domain is refused rather than divided by zero.
    #[test]
    fn a_challenge_on_the_domain_is_rejected() {
        let r = reference::load();
        let w = fr::from_decimal(&r.setup.omegas["w"]).unwrap();
        let err = check_inv_zh(&BigUint::one(), &w, 8).unwrap_err().to_string();
        assert!(err.contains("root of Z_H"), "{err}");
    }

    /// The reference AIR's schedule: alpha and beta, gamma and delta, then a.
    fn reference_schedule() -> ChallengeSchedule {
        ChallengeSchedule(vec![2, 2, 1])
    }

    /// A schedule that does not cover every stage is rejected.
    ///
    /// Without this it succeeds and returns a shorter list of plausible-looking
    /// challenges, having silently never absorbed the missing stage's
    /// commitments -- and the failure then surfaces as an unrelated rejection
    /// much further downstream.
    #[test]
    fn rejects_a_schedule_that_does_not_cover_every_stage() {
        let r = reference::load();
        for bad in [vec![2, 2], vec![2, 2, 1, 1], vec![], vec![5]] {
            let err = air_challenges(&r.setup, &r.proof, &r.publics, &ChallengeSchedule(bad.clone()))
                .unwrap_err()
                .to_string();
            assert!(err.contains("schedule covers"), "{bad:?}: {err}");
        }
    }

    /// The count is what the AIR's stages require, not an arbitrary choice.
    #[test]
    fn the_schedule_length_follows_from_the_stages() {
        let r = reference::load();
        assert_eq!(crate::verifier::quotient_stage(&r.setup), 4);
        assert_eq!(reference_schedule().0.len(), 3, "stages 1, 2 and 3 lie between constants and quotient");
    }

    /// The whole AIR chain, against the challenges the prover logged. These
    /// feed every later stage, so getting the chain wrong invalidates
    /// everything downstream -- including the quotient and the opening.
    #[test]
    fn reproduces_the_provers_air_challenges() {
        let r = reference::load();
        let got = air_challenges(&r.setup, &r.proof, &r.publics, &reference_schedule()).unwrap();
        assert_eq!(got, r.air_challenges);
    }

    /// The chain ends where the opening scheme begins: the last AIR challenge
    /// is what seeds the xi seed.
    #[test]
    fn the_last_air_challenge_seeds_the_opening() {
        let r = reference::load();
        let got = air_challenges(&r.setup, &r.proof, &r.publics, &reference_schedule()).unwrap();
        assert_eq!(got.last().unwrap(), &r.previous_challenge);

        let seed = crate::verifier::challenge_xi_seed(got.last().unwrap().clone(), &r.setup, &r.proof).unwrap();
        assert_eq!(seed, r.xi_seed);
    }

    /// The publics are absorbed into the seed, so a different statement gives
    /// different challenges throughout.
    #[test]
    fn the_publics_bind_every_challenge() {
        let r = reference::load();
        let honest = air_challenges(&r.setup, &r.proof, &r.publics, &reference_schedule()).unwrap();

        let mut publics = r.publics.clone();
        publics[0] = fr::add(&publics[0], &BigUint::one());
        let other = air_challenges(&r.setup, &r.proof, &publics, &reference_schedule()).unwrap();

        assert!(honest.iter().zip(&other).all(|(a, b)| a != b), "a challenge survived a changed public");
    }

    /// Each stage's commitments bind the challenges drawn after them, and only
    /// those -- which is what makes the chain sequential rather than a single
    /// hash of everything.
    #[test]
    fn a_stage_binds_only_the_challenges_after_it() {
        let r = reference::load();
        let honest = air_challenges(&r.setup, &r.proof, &r.publics, &reference_schedule()).unwrap();

        // f6 is the only stage-2 polynomial, absorbed before gamma (index 2).
        let mut proof = r.proof.clone();
        proof.polynomials.get_mut("f6").unwrap()[0] = "7".to_string();
        let tampered = air_challenges(&r.setup, &proof, &r.publics, &reference_schedule()).unwrap();

        assert_eq!(&tampered[..2], &honest[..2], "alpha and beta precede stage 2");
        assert!(tampered[2..].iter().zip(&honest[2..]).all(|(a, b)| a != b), "gamma onward must change");
    }

    /// Every challenge reseeds the transcript with itself, so the chain is a
    /// sequence of single-item hashes rather than a growing accumulation. A
    /// verifier that accumulated would agree on the first and diverge after.
    #[test]
    fn each_challenge_reseeds_the_transcript() {
        let r = reference::load();
        let got = air_challenges(&r.setup, &r.proof, &r.publics, &reference_schedule()).unwrap();

        // beta follows alpha with nothing else absorbed, so it is exactly the
        // hash of alpha alone.
        let mut t = Transcript::new();
        t.add_scalar(got[0].clone());
        assert_eq!(t.get_challenge(), got[1]);

        // and delta follows gamma the same way.
        let mut t = Transcript::new();
        t.add_scalar(got[2].clone());
        assert_eq!(t.get_challenge(), got[3]);
    }

    #[test]
    fn reports_a_proof_without_the_hint() {
        let r = reference::load();
        let mut proof = r.proof.clone();
        proof.evaluations.remove(INV_ZH_KEY);
        assert!(inv_zh(&proof).is_err());
    }
}
