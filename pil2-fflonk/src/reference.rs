//! Vectors logged from a single run of pil-fflonk's C++ prover.
//!
//! Self-consistency is not evidence: an implementation can satisfy every
//! internal invariant and still disagree with the prover, which is exactly what
//! happened when the xi-seed rule was first mirrored from the wrong function.
//! These vectors are the external check.
//!
//! They come from one run and only fit each other. The prover blinds with fresh
//! randomness on every invocation, so pairing these challenges with a different
//! proof, or that proof with these, produces confident nonsense. Re-capture the
//! whole set together or not at all.
//!
//! Captured by logging `rootsMap`, the interpolation inputs, and the scalars in
//! `computeL` from `shplonk.cpp`.

use num_bigint::BigUint;
use serde_json::Value;

use crate::fr;
use crate::proof::ShPlonkProof;
use crate::setup::ShPlonkSetup;

const VKEY: &str = include_str!("../tests/fixtures/pilfflonk.vkey");
const PROOF: &str = include_str!("../tests/fixtures/pilfflonk.proof.json");
const REFERENCE: &str = include_str!("../tests/fixtures/pilfflonk.reference.json");

pub struct Reference {
    pub setup: ShPlonkSetup,
    pub proof: ShPlonkProof,
    pub previous_challenge: BigUint,
    pub xi_seed: BigUint,
    pub xi: BigUint,
    pub alpha: BigUint,
    pub y: BigUint,
    /// `Q(xi)`, which the proof omits. [`crate::verify::prepare`] reconstructs
    /// it from the constraint system; keeping the prover's value here lets the
    /// opening scheme be tested on its own, and gives the reconstruction
    /// something to be checked against.
    pub quotient_evaluation: BigUint,
    /// The AIR's own challenges, in the order the protocol draws them.
    pub air_challenges: Vec<BigUint>,
    pub publics: Vec<BigUint>,
    pub roots: Vec<Vec<BigUint>>,
    pub f_at_roots: Vec<Vec<BigUint>>,
    pub z_s: Vec<BigUint>,
    pub z_t: BigUint,
    pub pre_l: Vec<BigUint>,
    pub r_at_y: Vec<BigUint>,
}

fn scalar(v: &Value, key: &str) -> BigUint {
    fr::from_decimal(v[key].as_str().unwrap_or_else(|| panic!("reference is missing {key}")))
        .unwrap_or_else(|e| panic!("reference {key}: {e}"))
}

fn vector(v: &Value, key: &str) -> Vec<BigUint> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("reference {key} is not an array"))
        .iter()
        .map(|s| fr::from_decimal(s.as_str().unwrap()).unwrap())
        .collect()
}

fn matrix(v: &Value, key: &str) -> Vec<Vec<BigUint>> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("reference {key} is not an array"))
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|s| fr::from_decimal(s.as_str().unwrap()).unwrap()).collect())
        .collect()
}

/// Load the fixtures. Cheap enough to call per test.
pub fn load() -> Reference {
    let setup = ShPlonkSetup::from_vkey_json(&serde_json::from_str(VKEY).unwrap()).unwrap();
    let proof = ShPlonkProof::from_json(&serde_json::from_str(PROOF).unwrap()).unwrap();
    let v: Value = serde_json::from_str(REFERENCE).unwrap();

    Reference {
        setup,
        proof,
        previous_challenge: scalar(&v, "previousChallenge"),
        xi_seed: scalar(&v, "xiSeed"),
        xi: scalar(&v, "xi"),
        alpha: scalar(&v, "alpha"),
        y: scalar(&v, "y"),
        quotient_evaluation: scalar(&v, "quotientEvaluation"),
        air_challenges: vector(&v, "airChallenges"),
        publics: vector(&v, "publics"),
        roots: matrix(&v, "roots"),
        f_at_roots: matrix(&v, "fAtRoots"),
        z_s: vector(&v, "zS"),
        z_t: scalar(&v, "zT"),
        pre_l: vector(&v, "preL"),
        r_at_y: vector(&v, "rAtY"),
    }
}

impl Reference {
    /// The quotient evaluation, in the shape [`resolve_evaluations`] wants.
    ///
    /// [`resolve_evaluations`]: crate::linearisation::resolve_evaluations
    pub fn derived(&self) -> crate::linearisation::Evaluations {
        let mut d = crate::linearisation::Evaluations::new();
        for pol in crate::verifier::non_committed_pols(&self.setup) {
            // Every omitted polynomial in the reference is opened at point 0.
            d.insert(pol.to_string(), self.quotient_evaluation.clone());
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture must describe the key it is paired with, or a test could
    /// pass by comparing two things that are both wrong.
    #[test]
    fn the_vectors_match_the_shape_of_the_key() {
        let r = load();
        let n = r.setup.f.len();

        assert_eq!(r.roots.len(), n);
        assert_eq!(r.f_at_roots.len(), n);
        assert_eq!(r.z_s.len(), n);
        assert_eq!(r.pre_l.len(), n);
        assert_eq!(r.r_at_y.len(), n);

        for (f, roots) in r.setup.f.iter().zip(&r.roots) {
            assert_eq!(roots.len(), f.pols.len() * f.opening_points.len(), "f{}", f.index);
        }
        for (roots, values) in r.roots.iter().zip(&r.f_at_roots) {
            assert_eq!(roots.len(), values.len());
        }
    }

    /// The challenges in the fixture are internally consistent, which catches a
    /// half-updated capture.
    #[test]
    fn the_captured_challenges_agree_with_each_other() {
        let r = load();
        assert_eq!(fr::pow(&r.xi_seed, r.setup.power_w as u64), r.xi);
        // The seed's transcript is fed the last AIR challenge.
        assert_eq!(r.previous_challenge, r.air_challenges[4]);
    }
}
