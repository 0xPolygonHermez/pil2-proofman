//! The pil2-fflonk proof: the JSON the prover emits and the verifier consumes.
//!
//! Shape, as produced by `ShPlonkProver::toJson`:
//!
//! ```json
//! {
//!   "polynomials": { "f0": [x, y, "1"], ..., "W": [...], "Wp": [...] },
//!   "evaluations":  { "<pol><wPower>": "<decimal>", ..., "inv": "...", "invZh": "..." },
//!   "protocol": "pilfflonk",
//!   "curve": "bn128"
//! }
//! ```
//!
//! Commitments are affine G1 points written as three decimal strings, the third
//! always `"1"`. Evaluations are field elements in decimal.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use crate::setup::ShPlonkSetup;

/// The `protocol` field of a proof.
///
/// Note this is *not* the vkey's value: the verification key says
/// `"pil-fflonk"` and the proof says `"pilfflonk"`. The inconsistency is in the
/// original artifacts, and both spellings are kept so existing files stay
/// readable. Use [`crate::setup::PROTOCOL`] for the vkey.
pub const PROOF_PROTOCOL: &str = "pilfflonk";
pub const CURVE: &str = "bn128";

/// The commitment for the batched opening polynomial W.
pub const W_KEY: &str = "W";
/// The commitment for W', the polynomial that proves W's opening.
pub const WP_KEY: &str = "Wp";
/// The batched Montgomery inverse the verifier reuses.
pub const INV_KEY: &str = "inv";
/// The inverse of the vanishing polynomial at xi, supplied by the prover.
pub const INV_ZH_KEY: &str = "invZh";

/// The quotient polynomial is committed but its evaluation is deliberately
/// absent: the prover erases it, since the verifier reconstructs it from the
/// constraint identity rather than trusting a claimed value.
pub const UNEVALUATED_POL: &str = "Q";

/// An affine G1 point as the proof carries it: `[x, y, "1"]`, decimal strings.
pub type G1Json = Vec<String>;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ShPlonkProof {
    /// Commitments, keyed `f0`, `f1`, ..., plus `W` and `Wp`.
    pub polynomials: BTreeMap<String, G1Json>,
    /// Evaluations, keyed by [`evaluation_key`], plus `inv` and `invZh`.
    pub evaluations: BTreeMap<String, String>,
}

/// The key an evaluation is stored under.
///
/// The opening point is encoded as a suffix rather than a separate field:
/// point 0 has no suffix, point 1 is `w`, and point *n* is `w<n>`. Mirrors the
/// `wPower` construction in `shplonk.cpp`.
pub fn evaluation_key(pol_name: &str, opening_point: u32) -> String {
    match opening_point {
        0 => pol_name.to_string(),
        1 => format!("{pol_name}w"),
        n => format!("{pol_name}w{n}"),
    }
}

/// The key an `f_i`'s commitment is stored under.
pub fn commitment_key(index: u32) -> String {
    format!("f{index}")
}

impl ShPlonkProof {
    pub fn to_json(&self) -> Value {
        let mut polynomials = Map::new();
        for (key, point) in &self.polynomials {
            polynomials.insert(key.clone(), json!(point));
        }

        let mut evaluations = Map::new();
        for (key, value) in &self.evaluations {
            evaluations.insert(key.clone(), json!(value));
        }

        let mut out = Map::new();
        out.insert("polynomials".into(), Value::Object(polynomials));
        out.insert("evaluations".into(), Value::Object(evaluations));
        out.insert("protocol".into(), json!(PROOF_PROTOCOL));
        out.insert("curve".into(), json!(CURVE));
        Value::Object(out)
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let obj = value.as_object().context("proof is not a JSON object")?;

        let protocol = obj.get("protocol").and_then(Value::as_str).unwrap_or_default();
        if protocol != PROOF_PROTOCOL {
            bail!("unexpected proof protocol {protocol:?}, want {PROOF_PROTOCOL:?}");
        }
        let curve = obj.get("curve").and_then(Value::as_str).unwrap_or_default();
        if curve != CURVE {
            bail!("unexpected curve {curve:?}, want {CURVE:?}");
        }

        let polynomials = obj
            .get("polynomials")
            .and_then(Value::as_object)
            .context("proof is missing \"polynomials\"")?
            .iter()
            .map(|(k, v)| {
                let point: G1Json =
                    serde_json::from_value(v.clone()).with_context(|| format!("parsing commitment {k:?}"))?;
                if point.len() != 3 {
                    bail!("commitment {k:?} has {} coordinates, want 3", point.len());
                }
                Ok((k.clone(), point))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        let evaluations = obj
            .get("evaluations")
            .and_then(Value::as_object)
            .context("proof is missing \"evaluations\"")?
            .iter()
            .map(|(k, v)| {
                let s = v.as_str().with_context(|| format!("evaluation {k:?} is not a string"))?;
                Ok((k.clone(), s.to_string()))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        Ok(Self { polynomials, evaluations })
    }

    /// Check the proof carries exactly what this setup calls for.
    ///
    /// Purely structural -- it says nothing about whether the proof is *valid*,
    /// only that it is well formed for the setup, so a malformed proof fails
    /// here rather than midway through a pairing check.
    pub fn validate_against(&self, setup: &ShPlonkSetup) -> Result<()> {
        for f in &setup.f {
            let key = commitment_key(f.index);
            let point = self
                .polynomials
                .get(&key)
                .with_context(|| format!("proof is missing the commitment for f{}", f.index))?;
            if point.len() != 3 {
                bail!("commitment {key:?} has {} coordinates, want 3", point.len());
            }
        }

        for key in [W_KEY, WP_KEY] {
            if !self.polynomials.contains_key(key) {
                bail!("proof is missing the {key:?} commitment");
            }
        }

        let expected_commitments = setup.f.len() + 2;
        if self.polynomials.len() != expected_commitments {
            bail!(
                "proof carries {} commitments, want {} ({} f_i plus W and Wp)",
                self.polynomials.len(),
                expected_commitments,
                setup.f.len()
            );
        }

        for f in &setup.f {
            for &point in &f.opening_points {
                for pol in &f.pols {
                    if pol == UNEVALUATED_POL {
                        continue;
                    }
                    let key = evaluation_key(pol, point);
                    if !self.evaluations.contains_key(&key) {
                        bail!("proof is missing evaluation {key:?} (f{}, opening point {point})", f.index);
                    }
                }
            }
        }

        for key in [INV_KEY, INV_ZH_KEY] {
            if !self.evaluations.contains_key(key) {
                bail!("proof is missing the {key:?} evaluation");
            }
        }

        // The quotient's evaluation must be absent, not merely unused: a proof
        // supplying one is claiming a value the verifier is supposed to derive.
        if self.evaluations.contains_key(UNEVALUATED_POL) {
            bail!("proof supplies an evaluation for {UNEVALUATED_POL:?}, which the verifier must reconstruct");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VKEY: &str = include_str!("../tests/fixtures/pilfflonk.vkey");

    fn real_setup() -> ShPlonkSetup {
        ShPlonkSetup::from_vkey_json(&serde_json::from_str(VKEY).unwrap()).unwrap()
    }

    /// A structurally complete proof for the given setup.
    fn proof_for(setup: &ShPlonkSetup) -> ShPlonkProof {
        let point = || vec!["1".to_string(), "2".to_string(), "1".to_string()];

        let mut polynomials = BTreeMap::new();
        for f in &setup.f {
            polynomials.insert(commitment_key(f.index), point());
        }
        polynomials.insert(W_KEY.to_string(), point());
        polynomials.insert(WP_KEY.to_string(), point());

        let mut evaluations = BTreeMap::new();
        for f in &setup.f {
            for &op in &f.opening_points {
                for pol in &f.pols {
                    if pol == UNEVALUATED_POL {
                        continue;
                    }
                    evaluations.insert(evaluation_key(pol, op), "7".to_string());
                }
            }
        }
        evaluations.insert(INV_KEY.to_string(), "3".to_string());
        evaluations.insert(INV_ZH_KEY.to_string(), "5".to_string());

        ShPlonkProof { polynomials, evaluations }
    }

    #[test]
    fn evaluation_keys_encode_the_opening_point_as_a_suffix() {
        assert_eq!(evaluation_key("A.x", 0), "A.x");
        assert_eq!(evaluation_key("A.x", 1), "A.xw");
        assert_eq!(evaluation_key("A.x", 2), "A.xw2");
        assert_eq!(evaluation_key("A.x", 17), "A.xw17");
    }

    #[test]
    fn round_trips_through_json() {
        let proof = proof_for(&real_setup());
        let parsed = ShPlonkProof::from_json(&proof.to_json()).unwrap();
        assert_eq!(parsed, proof);
    }

    #[test]
    fn accepts_a_well_formed_proof_for_the_real_setup() {
        let setup = real_setup();
        let proof = proof_for(&setup);

        proof.validate_against(&setup).unwrap();

        // 9 f_i plus W and Wp.
        assert_eq!(proof.polynomials.len(), 11);
    }

    #[test]
    fn rejects_a_missing_commitment() {
        let setup = real_setup();
        let mut proof = proof_for(&setup);
        proof.polynomials.remove("f3");

        let err = proof.validate_against(&setup).unwrap_err().to_string();
        assert!(err.contains("commitment for f3"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_a_missing_opening_polynomial() {
        let setup = real_setup();
        for key in [W_KEY, WP_KEY] {
            let mut proof = proof_for(&setup);
            proof.polynomials.remove(key);
            let err = proof.validate_against(&setup).unwrap_err().to_string();
            assert!(err.contains(key), "unexpected error for {key}: {err}");
        }
    }

    #[test]
    fn rejects_a_missing_evaluation() {
        let setup = real_setup();
        let mut proof = proof_for(&setup);

        // Drop one real (polynomial, opening point) evaluation.
        let victim = evaluation_key(&setup.f[0].pols[0], setup.f[0].opening_points[0]);
        proof.evaluations.remove(&victim);

        let err = proof.validate_against(&setup).unwrap_err().to_string();
        assert!(err.contains(&victim), "unexpected error: {err}");
    }

    #[test]
    fn rejects_extra_commitments() {
        let setup = real_setup();
        let mut proof = proof_for(&setup);
        proof.polynomials.insert("f99".to_string(), vec!["1".into(), "1".into(), "1".into()]);

        assert!(proof.validate_against(&setup).is_err());
    }

    /// Q is committed inside its f_i but must not carry an evaluation -- the
    /// prover erases it and the verifier reconstructs it.
    #[test]
    fn rejects_a_supplied_quotient_evaluation() {
        let setup = real_setup();
        let mut proof = proof_for(&setup);

        // The real setup packs Q alone in the last f_i.
        assert!(setup.f.iter().any(|f| f.pols.iter().any(|p| p == UNEVALUATED_POL)));

        proof.evaluations.insert(UNEVALUATED_POL.to_string(), "1".to_string());
        let err = proof.validate_against(&setup).unwrap_err().to_string();
        assert!(err.contains("must reconstruct"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_a_foreign_protocol() {
        let proof = proof_for(&real_setup());
        let mut v = proof.to_json();
        v["protocol"] = json!("fflonk");
        assert!(ShPlonkProof::from_json(&v).is_err());
    }

    #[test]
    fn rejects_a_malformed_commitment() {
        let proof = proof_for(&real_setup());
        let mut v = proof.to_json();
        v["polynomials"]["f0"] = json!(["1", "2"]);
        let err = ShPlonkProof::from_json(&v).unwrap_err().to_string();
        assert!(err.contains("coordinates"), "unexpected error: {err}");
    }

    /// The proof and the vkey disagree on the protocol string. Pinned so the
    /// discrepancy is deliberate rather than something a later edit "fixes"
    /// and breaks compatibility with existing artifacts.
    #[test]
    fn proof_and_vkey_protocol_strings_differ() {
        assert_eq!(PROOF_PROTOCOL, "pilfflonk");
        assert_eq!(crate::setup::PROTOCOL, "pil-fflonk");
        assert_ne!(PROOF_PROTOCOL, crate::setup::PROTOCOL);
    }
}

#[cfg(test)]
mod real_proof_tests {
    use super::*;
    use crate::solidity::ProofLayout;

    const VKEY: &str = include_str!("../tests/fixtures/pilfflonk.vkey");
    const PROOF: &str = include_str!("../tests/fixtures/pilfflonk.proof.json");

    fn real() -> (ShPlonkSetup, ShPlonkProof) {
        let setup = ShPlonkSetup::from_vkey_json(&serde_json::from_str(VKEY).unwrap()).unwrap();
        let proof = ShPlonkProof::from_json(&serde_json::from_str(PROOF).unwrap()).unwrap();
        (setup, proof)
    }

    /// A proof produced by pil-fflonk's own prover, against the setup this
    /// crate parses from the verification key. Nothing here is synthesised.
    #[test]
    fn validates_a_real_proof() {
        let (setup, proof) = real();
        proof.validate_against(&setup).expect("real proof must satisfy the validator");

        assert_eq!(proof.polynomials.len(), 11, "9 f_i plus W and Wp");
        assert_eq!(proof.evaluations.len(), 43);
    }

    /// The layout is derived from the verification key alone. That it names
    /// exactly the evaluations a real proof carries -- no more, no fewer -- is
    /// what makes it safe to drive calldata encoding from the vkey.
    #[test]
    fn layout_matches_a_real_proof_exactly() {
        let (setup, proof) = real();
        let layout = ProofLayout::from_setup(&setup);

        let predicted: std::collections::BTreeSet<&String> = layout.evaluations.iter().collect();
        let actual: std::collections::BTreeSet<&String> = proof.evaluations.keys().collect();
        assert_eq!(predicted, actual, "layout and proof disagree on the evaluation set");

        let predicted_c: std::collections::BTreeSet<&String> = layout.commitments.iter().collect();
        let actual_c: std::collections::BTreeSet<&String> = proof.polynomials.keys().collect();
        assert_eq!(predicted_c, actual_c, "layout and proof disagree on the commitment set");

        // And the encoder can flatten it without a missing entry.
        let words = proof.to_calldata_words(&layout).unwrap();
        assert_eq!(words.len(), layout.word_count());
        assert_eq!(words.len(), 11 * 2 + 43);
    }

    /// The prover erases Q's evaluation; the validator requires that.
    #[test]
    fn real_proof_omits_the_quotient_evaluation() {
        let (_, proof) = real();
        assert!(!proof.evaluations.contains_key(UNEVALUATED_POL));
        assert!(proof.polynomials.contains_key("f8"), "but Q's f_i is still committed");
    }
}
