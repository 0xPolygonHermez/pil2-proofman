//! Recomputing the SHPLONK challenges from a proof.
//!
//! A verifier must derive every challenge itself; taking any of them from the
//! proof would let a prover choose them. Each function here mirrors its
//! counterpart in `shplonk.cpp` and is checked against challenges logged by a
//! real proving run.
//!
//! The pairing check is not here yet -- this covers the Fiat-Shamir half, which
//! is the part that can be validated without G1/G2 arithmetic.

use anyhow::{Context, Result, bail};
use num_bigint::BigUint;
use num_traits::Num;

use crate::proof::{ShPlonkProof, UNEVALUATED_POL, W_KEY, commitment_key, evaluation_key};
use crate::setup::ShPlonkSetup;
use crate::transcript::{Transcript, fr_modulus};

/// The stage carrying the quotient: the highest any `f_i` declares.
pub fn quotient_stage(setup: &ShPlonkSetup) -> u32 {
    setup.f.iter().filter_map(|f| f.stages.first().map(|s| s.stage)).max().unwrap_or(0)
}

/// Whether an `f_i`'s commitment enters the xi-seed transcript.
///
/// Only those in the quotient stage. Everything earlier is already bound: each
/// stage's commitments went into the transcript that produced the challenge for
/// the stage after it, and that chain ends in the scalar seeded here.
///
/// Note this is *not* `ShPlonkProver::computeChallengeXiSeed`, which includes
/// every non-constant `f_i`. That method is unused in this flow -- `open()`
/// receives the seed as a parameter, and `pilfflonk_prover.cpp` computes it
/// with the narrower rule above.
pub fn enters_xi_seed_transcript(f: &crate::setup::ShPlonkPol, quotient_stage: u32) -> bool {
    f.stages.first().map(|s| s.stage == quotient_stage).unwrap_or(false)
}

/// Recompute the xi seed.
///
/// `previous_challenge` is the last challenge of the preceding protocol phase,
/// which the AIR's own transcript produces; it is an input here rather than
/// something the SHPLONK layer can derive.
pub fn challenge_xi_seed(previous_challenge: BigUint, setup: &ShPlonkSetup, proof: &ShPlonkProof) -> Result<BigUint> {
    let mut t = Transcript::new();
    t.add_scalar(previous_challenge);

    let last = quotient_stage(setup);
    for f in &setup.f {
        if !enters_xi_seed_transcript(f, last) {
            continue;
        }
        let key = commitment_key(f.index);
        let point = proof
            .polynomials
            .get(&key)
            .with_context(|| format!("proof is missing the commitment for {key} needed by the transcript"))?;
        t.add_commitment_json(point)?;
    }

    Ok(t.get_challenge())
}

/// The polynomials whose evaluations are *not* sent, and so do not enter the
/// alpha transcript.
///
/// From `pilfflonk_prover.cpp`: `Q` is omitted exactly when the quotient stage
/// holds a single polynomial. Split across several (`Q0`, `Q1`, ...) each piece
/// is committed and evaluated normally, so nothing is omitted.
pub fn non_committed_pols(setup: &ShPlonkSetup) -> Vec<&str> {
    let last = quotient_stage(setup);
    let mut pols = setup
        .f
        .iter()
        .filter(|f| f.stages.first().map(|s| s.stage == last).unwrap_or(false))
        .flat_map(|f| f.pols.iter());

    match (pols.next(), pols.next()) {
        (Some(only), None) if only == UNEVALUATED_POL => vec![UNEVALUATED_POL],
        _ => Vec::new(),
    }
}

/// Recompute the challenge that batches the openings.
///
/// From `computeChallengeAlpha`: reset, add the xi seed, then add every
/// evaluation the proof commits to, and squeeze.
///
/// The order is part of the encoding and is not the proof's own key order --
/// it is `f_i`, then opening point, then polynomial, so that a `BTreeMap`'s
/// alphabetical iteration would give a different challenge. `inv` and `invZh`
/// are prover-supplied inverse hints rather than opening claims, and are
/// excluded by construction: no `f_i` names them.
pub fn challenge_alpha(xi_seed: BigUint, setup: &ShPlonkSetup, proof: &ShPlonkProof) -> Result<BigUint> {
    let skip = non_committed_pols(setup);

    let mut t = Transcript::new();
    t.add_scalar(xi_seed);

    for f in &setup.f {
        for &point in &f.opening_points {
            for pol in &f.pols {
                if skip.contains(&pol.as_str()) {
                    continue;
                }
                let key = evaluation_key(pol, point);
                let value = proof
                    .evaluations
                    .get(&key)
                    .with_context(|| format!("proof is missing evaluation {key:?}, needed by the transcript"))?;
                let value =
                    BigUint::from_str_radix(value, 10).with_context(|| format!("parsing evaluation {key:?}"))?;
                t.add_scalar(value);
            }
        }
    }

    Ok(t.get_challenge())
}

/// `xi = xiSeed ^ powerW`.
pub fn challenge_xi(xi_seed: &BigUint, power_w: u32) -> BigUint {
    xi_seed.modpow(&BigUint::from(power_w), &fr_modulus())
}

/// Recompute the challenge that opens W.
///
/// From `computeChallengeY`: reset, add alpha, add the W commitment, squeeze.
pub fn challenge_y(alpha: BigUint, proof: &ShPlonkProof) -> Result<BigUint> {
    let w = proof.polynomials.get(W_KEY).context("proof is missing the W commitment")?;
    let mut t = Transcript::new();
    t.add_scalar(alpha);
    t.add_commitment_json(w)?;
    Ok(t.get_challenge())
}

/// The commitment a verifier must use for one `f_i`.
///
/// Combined polynomials built only from constant polynomials are fixed by the
/// setup, and the key carries their commitments. The proof carries a copy of
/// every commitment including those, but a verifier that read them from the
/// proof would let a prover pick its own constant polynomials -- which is to
/// say, prove a different AIR. The key's value wins wherever it exists.
pub fn commitment_for<'a>(setup: &'a ShPlonkSetup, proof: &'a ShPlonkProof, index: u32) -> Result<&'a Vec<String>> {
    let key = commitment_key(index);
    if let Some(fixed) = setup.f_commitments.get(&key) {
        return Ok(fixed);
    }
    proof.polynomials.get(&key).with_context(|| format!("proof is missing the commitment for {key}"))
}

/// Reject a proof that disagrees with the key about a fixed commitment.
///
/// [`commitment_for`] already prevents such a proof from being verified against
/// the wrong point, so this changes no outcome -- it turns a proof that would
/// fail obscurely at the pairing into one that fails where the reason is
/// visible.
pub fn check_constant_commitments(setup: &ShPlonkSetup, proof: &ShPlonkProof) -> Result<()> {
    for (key, fixed) in &setup.f_commitments {
        if let Some(claimed) = proof.polynomials.get(key) {
            if claimed != fixed {
                bail!("proof's {key} commitment differs from the one fixed by the verification key");
            }
        }
    }
    Ok(())
}

/// Check the proof's shape and recompute the challenges that follow from it.
///
/// Structural validation first, so a malformed proof fails before any
/// arithmetic is spent on it.
pub fn recompute_challenges(
    previous_challenge: BigUint,
    setup: &ShPlonkSetup,
    proof: &ShPlonkProof,
) -> Result<Challenges> {
    proof.validate_against(setup)?;
    check_constant_commitments(setup, proof)?;

    let xi_seed = challenge_xi_seed(previous_challenge, setup, proof)?;
    let xi = challenge_xi(&xi_seed, setup.power_w);
    let alpha = challenge_alpha(xi_seed.clone(), setup, proof)?;
    let y = challenge_y(alpha.clone(), proof)?;

    let r = fr_modulus();
    if xi_seed >= r || xi >= r || alpha >= r || y >= r {
        bail!("a recomputed challenge is not reduced");
    }

    Ok(Challenges { xi_seed, xi, alpha, y })
}

/// The challenges a verifier derives, in the order the protocol produces them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Challenges {
    pub xi_seed: BigUint,
    pub xi: BigUint,
    pub alpha: BigUint,
    pub y: BigUint,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fr;
    use crate::reference::{self, Reference};

    fn real() -> Reference {
        reference::load()
    }

    /// Only the quotient stage enters the seed. In the reference that is f8
    /// alone -- earlier stages are already bound through the challenge chain.
    #[test]
    fn only_the_quotient_stage_enters_the_transcript() {
        let r = real();
        let setup = r.setup.clone();
        let last = quotient_stage(&setup);
        assert_eq!(last, 4);

        let included: Vec<u32> =
            setup.f.iter().filter(|f| enters_xi_seed_transcript(f, last)).map(|f| f.index).collect();
        assert_eq!(included, vec![8]);
    }

    /// Recomputed from the proof's own commitments, matching the prover.
    #[test]
    fn reproduces_the_provers_xi_seed() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let seed = challenge_xi_seed(r.previous_challenge.clone(), &setup, &proof).unwrap();
        assert_eq!(seed, r.xi_seed.clone());
    }

    #[test]
    fn reproduces_the_provers_xi() {
        let r = real();
        let setup = r.setup.clone();
        assert_eq!(setup.power_w, 12);
        assert_eq!(challenge_xi(&r.xi_seed.clone(), setup.power_w), r.xi.clone());
    }

    #[test]
    fn reproduces_the_provers_y() {
        let r = real();
        let proof = r.proof.clone();
        assert_eq!(challenge_y(r.alpha.clone(), &proof).unwrap(), r.y.clone());
    }

    /// The reference omits Q alone: its stage holds one polynomial, so the
    /// verifier reconstructs that single evaluation rather than reading it.
    #[test]
    fn only_the_quotient_is_left_out_of_the_alpha_transcript() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        assert_eq!(non_committed_pols(&setup), vec!["Q"]);

        let skip = non_committed_pols(&setup);
        let hashed: Vec<String> = setup
            .f
            .iter()
            .flat_map(|f| f.opening_points.iter().flat_map(move |&p| f.pols.iter().map(move |pol| (pol.clone(), p))))
            .filter(|(pol, _)| !skip.contains(&pol.as_str()))
            .map(|(pol, p)| evaluation_key(&pol, p))
            .collect();

        assert_eq!(hashed.len(), 41, "42 opening slots, less Q");
        // Every hashed name is an evaluation the proof actually carries.
        for key in &hashed {
            assert!(proof.evaluations.contains_key(key), "{key} is hashed but not in the proof");
        }
        // The two the proof carries beyond them are inv/invZh, which are
        // inverse hints rather than opening claims.
        assert_eq!(proof.evaluations.len(), hashed.len() + 2);
    }

    /// A split quotient leaves nothing out: each piece is committed normally.
    #[test]
    fn a_split_quotient_omits_nothing() {
        let r = real();
        let mut setup = r.setup.clone();
        let last = quotient_stage(&setup);
        let f = setup.f.iter_mut().find(|f| f.stages[0].stage == last).unwrap();
        f.pols = vec!["Q0".into(), "Q1".into()];

        assert!(non_committed_pols(&setup).is_empty());
    }

    #[test]
    fn reproduces_the_provers_alpha() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let alpha = challenge_alpha(r.xi_seed.clone(), &setup, &proof).unwrap();
        assert_eq!(alpha, r.alpha.clone());
    }

    /// Alpha binds every evaluation it hashes: changing any one must change it,
    /// or a prover could substitute an opening after alpha was drawn.
    #[test]
    fn alpha_binds_every_hashed_evaluation() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let baseline = challenge_alpha(r.xi_seed.clone(), &setup, &proof).unwrap();

        for key in ["Global.L1", "Plookup.Bw", "Im28", "Plookup.Z0w"] {
            let mut tampered = proof.clone();
            *tampered.evaluations.get_mut(key).unwrap() = "3".to_string();
            assert_ne!(
                challenge_alpha(r.xi_seed.clone(), &setup, &tampered).unwrap(),
                baseline,
                "changing {key} left alpha unchanged"
            );
        }
    }

    /// The order is the setup's, not the proof map's. Hashing the same values
    /// alphabetically -- the order a BTreeMap iterates in -- must not agree, or
    /// the ordering would be unenforced and a prover could permute openings.
    #[test]
    fn alpha_depends_on_the_setup_order_not_the_map_order() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let expected = challenge_alpha(r.xi_seed.clone(), &setup, &proof).unwrap();

        let skip = non_committed_pols(&setup);
        let mut t = Transcript::new();
        t.add_scalar(r.xi_seed.clone());
        for (key, value) in &proof.evaluations {
            if key == "inv" || key == "invZh" || skip.contains(&key.as_str()) {
                continue;
            }
            t.add_scalar(fr::from_decimal(value).unwrap());
        }

        assert_ne!(t.get_challenge(), expected);
    }

    #[test]
    fn alpha_binds_the_xi_seed() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let a = challenge_alpha(r.xi_seed.clone(), &setup, &proof).unwrap();
        let b = challenge_alpha(BigUint::from(1u32), &setup, &proof).unwrap();
        assert_ne!(a, b);
    }

    /// All four together, straight from the fixtures: the whole Fiat-Shamir
    /// chain, from the preceding phase's challenge to Y, with nothing taken
    /// from the proof but commitments and evaluations.
    #[test]
    fn recomputes_every_challenge_from_the_proof() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let c = recompute_challenges(r.previous_challenge.clone(), &setup, &proof).unwrap();

        assert_eq!(c.xi_seed, r.xi_seed.clone());
        assert_eq!(c.xi, r.xi.clone());
        assert_eq!(c.alpha, r.alpha.clone());
        assert_eq!(c.y, r.y.clone());
    }

    /// The seed binds the quotient commitment: altering it must change the
    /// seed, or a prover could swap it after the challenge was drawn.
    #[test]
    fn seed_binds_every_included_commitment() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let baseline = challenge_xi_seed(r.previous_challenge.clone(), &setup, &proof).unwrap();

        let last = quotient_stage(&setup);
        let included = setup.f.iter().filter(|f| enters_xi_seed_transcript(f, last)).map(|f| f.index);

        for index in included {
            let mut tampered = proof.clone();
            let key = commitment_key(index);
            let point = tampered.polynomials.get_mut(&key).unwrap();
            point[0] = "7".to_string();
            let seed = challenge_xi_seed(r.previous_challenge.clone(), &setup, &tampered).unwrap();
            assert_ne!(seed, baseline, "changing f{index} left the seed unchanged");
        }
    }

    /// And it binds the preceding phase, so a proof cannot be replayed under a
    /// different AIR transcript.
    #[test]
    fn seed_binds_the_previous_challenge() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let a = challenge_xi_seed(r.previous_challenge.clone(), &setup, &proof).unwrap();
        let b = challenge_xi_seed(BigUint::from(1u32), &setup, &proof).unwrap();
        assert_ne!(a, b);
    }

    /// Y binds W: it is the point at which W's own opening is checked.
    #[test]
    fn y_binds_the_w_commitment() {
        let r = real();
        let proof = r.proof.clone();
        let baseline = challenge_y(r.alpha.clone(), &proof).unwrap();

        let mut tampered = proof.clone();
        tampered.polynomials.get_mut(W_KEY).unwrap()[1] = "9".to_string();
        assert_ne!(challenge_y(r.alpha.clone(), &tampered).unwrap(), baseline);
    }

    #[test]
    fn rejects_a_proof_missing_a_transcript_commitment() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let mut broken = proof.clone();
        broken.polynomials.remove("f8");
        assert!(challenge_xi_seed(r.previous_challenge.clone(), &setup, &broken).is_err());
    }

    /// A commitment the seed does not read must not be required to derive it.
    /// The stage-2 and stage-3 f_i are still checked by `validate_against` and
    /// by the pairing, just not by this transcript.
    #[test]
    fn a_commitment_outside_the_quotient_stage_is_not_needed_for_the_seed() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let baseline = challenge_xi_seed(r.previous_challenge.clone(), &setup, &proof).unwrap();

        let mut without = proof.clone();
        without.polynomials.remove("f4");
        assert_eq!(challenge_xi_seed(r.previous_challenge.clone(), &setup, &without).unwrap(), baseline);
    }

    /// The key fixes the constant-only commitments, and the reference proof
    /// agrees with it -- so reading from the key changes nothing for an honest
    /// proof, which is what makes it safe to prefer.
    #[test]
    fn constant_commitments_come_from_the_key() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());

        assert_eq!(setup.f_commitments.keys().collect::<Vec<_>>(), vec!["f0", "f1"]);
        // The key fixes exactly the f_i that are built from constants alone.
        let constant: Vec<String> =
            setup.f.iter().filter(|f| f.stages.iter().all(|s| s.stage == 0)).map(|f| commitment_key(f.index)).collect();
        assert_eq!(constant, vec!["f0".to_string(), "f1".to_string()]);

        for f in &setup.f {
            let used = commitment_for(&setup, &proof, f.index).unwrap();
            assert_eq!(used, &proof.polynomials[&commitment_key(f.index)]);
        }
        assert!(check_constant_commitments(&setup, &proof).is_ok());
    }

    /// A prover that substitutes a constant commitment is choosing its own
    /// constant polynomials, so the verifier must not follow it there.
    #[test]
    fn a_substituted_constant_commitment_is_ignored_and_rejected() {
        let r = real();
        let (setup, mut proof) = (r.setup.clone(), r.proof.clone());
        let honest = setup.f_commitments["f0"].clone();

        proof.polynomials.get_mut("f0").unwrap()[0] = "7".to_string();

        // The key's value is still the one used ...
        assert_eq!(commitment_for(&setup, &proof, 0).unwrap(), &honest);
        // ... and the disagreement is reported rather than passed over.
        assert!(check_constant_commitments(&setup, &proof).is_err());
        assert!(recompute_challenges(r.previous_challenge.clone(), &setup, &proof).is_err());
    }

    /// A commitment the key does not fix comes from the proof, since that is
    /// the only place it exists.
    #[test]
    fn a_non_constant_commitment_comes_from_the_proof() {
        let r = real();
        let (setup, mut proof) = (r.setup.clone(), r.proof.clone());

        proof.polynomials.get_mut("f8").unwrap()[0] = "7".to_string();
        assert_eq!(commitment_for(&setup, &proof, 8).unwrap()[0], "7");
        assert!(check_constant_commitments(&setup, &proof).is_ok());
    }

    #[test]
    fn recompute_rejects_a_structurally_invalid_proof() {
        let r = real();
        let (setup, proof) = (r.setup.clone(), r.proof.clone());
        let mut broken = proof.clone();
        broken.evaluations.remove("inv");
        assert!(recompute_challenges(r.previous_challenge.clone(), &setup, &broken).is_err());
    }
}
