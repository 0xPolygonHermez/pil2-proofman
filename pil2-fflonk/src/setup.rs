//! The SHPLONK setup: which committed polynomials are packed into each combined
//! polynomial `f_i`, at which opening points, and the roots of unity the opening
//! is built from.
//!
//! This is the Rust counterpart of `cpp/shplonk_setup.hpp`, and it is what the
//! verification key serialises. pil-fflonk produced the same JSON from its pil1
//! `.zkey`; pil2 will derive it from `StarkInfo`, but the on-disk shape is kept
//! identical so the existing verifier artifacts stay readable.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

pub const PROTOCOL: &str = "pil-fflonk";
pub const CURVE: &str = "bn128";

/// One committed polynomial's slot within a stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShPlonkStagePol {
    pub name: String,
    pub degree: u64,
}

/// The polynomials a given stage contributes to an `f_i`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShPlonkStage {
    pub stage: u32,
    pub pols: Vec<ShPlonkStagePol>,
}

/// A combined polynomial `f_i`: the polynomials packed behind one commitment,
/// and the points it is opened at.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShPlonkPol {
    pub index: u32,
    pub degree: u64,
    #[serde(rename = "openingPoints")]
    pub opening_points: Vec<u32>,
    pub pols: Vec<String>,
    pub stages: Vec<ShPlonkStage>,
}

/// Everything the SHPLONK opening and its verifier need from setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShPlonkSetup {
    pub n_publics: u32,
    pub max_q_degree: u32,
    pub power: u32,
    pub power_w: u32,
    /// The combined polynomials, in index order.
    pub f: Vec<ShPlonkPol>,
    /// Roots of unity, keyed as the verifier expects them: `w`, `w2`,
    /// `w2_1d2`, ... Serialised flat at the top level of the vkey.
    pub omegas: BTreeMap<String, String>,
    /// Precomputed commitments to the `f_i` built only from constant
    /// polynomials, keyed `f0`, `f1`, ... Each is a G1 point `[x, y, z]`.
    pub f_commitments: BTreeMap<String, Vec<String>>,
    /// The G2 element the pairing check uses.
    pub x2: Vec<Vec<String>>,
    /// Polynomial index to name, grouped by kind (`cm`, `const`, ...).
    pub pols_map: BTreeMap<String, BTreeMap<String, String>>,
}

impl ShPlonkSetup {
    /// Serialise to the verification-key JSON.
    ///
    /// The layout is flat: omegas and `f_i` commitments sit alongside the fixed
    /// keys rather than nested, which is what pil-fflonk's verifier reads.
    pub fn to_vkey_json(&self) -> Value {
        let mut out = Map::new();

        out.insert("protocol".into(), json!(PROTOCOL));
        out.insert("curve".into(), json!(CURVE));
        out.insert("nPublics".into(), json!(self.n_publics));
        out.insert("maxQDegree".into(), json!(self.max_q_degree));
        out.insert("power".into(), json!(self.power));
        out.insert("powerW".into(), json!(self.power_w));
        out.insert("f".into(), serde_json::to_value(&self.f).expect("f serialises"));

        for (name, value) in &self.omegas {
            out.insert(name.clone(), json!(value));
        }

        out.insert("X_2".into(), json!(self.x2));

        for (name, point) in &self.f_commitments {
            out.insert(name.clone(), json!(point));
        }

        out.insert("polsMap".into(), json!(self.pols_map));

        Value::Object(out)
    }

    /// Parse a verification-key JSON back into a setup.
    ///
    /// Keys are classified by shape, since omegas and commitments are both
    /// dynamic: `f<digits>` is a commitment, anything else starting with `w`
    /// is an omega.
    pub fn from_vkey_json(value: &Value) -> Result<Self> {
        let obj = value.as_object().context("vkey is not a JSON object")?;

        let protocol = obj.get("protocol").and_then(Value::as_str).unwrap_or_default();
        if protocol != PROTOCOL {
            bail!("unexpected protocol {protocol:?}, want {PROTOCOL:?}");
        }
        let curve = obj.get("curve").and_then(Value::as_str).unwrap_or_default();
        if curve != CURVE {
            bail!("unexpected curve {curve:?}, want {CURVE:?}");
        }

        let u32_at = |key: &str| -> Result<u32> {
            obj.get(key)
                .and_then(Value::as_u64)
                .map(|v| v as u32)
                .with_context(|| format!("vkey is missing integer field {key:?}"))
        };

        let f: Vec<ShPlonkPol> =
            serde_json::from_value(obj.get("f").context("vkey is missing \"f\"")?.clone()).context("parsing \"f\"")?;

        let x2: Vec<Vec<String>> = serde_json::from_value(obj.get("X_2").context("vkey is missing \"X_2\"")?.clone())
            .context("parsing \"X_2\"")?;

        let pols_map: BTreeMap<String, BTreeMap<String, String>> = match obj.get("polsMap") {
            Some(v) => serde_json::from_value(v.clone()).context("parsing \"polsMap\"")?,
            None => BTreeMap::new(),
        };

        let mut omegas = BTreeMap::new();
        let mut f_commitments = BTreeMap::new();

        for (key, val) in obj {
            match key.as_str() {
                "protocol" | "curve" | "nPublics" | "maxQDegree" | "power" | "powerW" | "f" | "X_2" | "polsMap" => {}
                k if is_commitment_key(k) => {
                    let point: Vec<String> =
                        serde_json::from_value(val.clone()).with_context(|| format!("parsing commitment {k:?}"))?;
                    f_commitments.insert(k.to_string(), point);
                }
                k if k.starts_with('w') => {
                    let omega = val.as_str().with_context(|| format!("omega {k:?} is not a string"))?;
                    omegas.insert(k.to_string(), omega.to_string());
                }
                k => bail!("unrecognised vkey field {k:?}"),
            }
        }

        Ok(Self {
            n_publics: u32_at("nPublics")?,
            max_q_degree: u32_at("maxQDegree")?,
            power: u32_at("power")?,
            power_w: u32_at("powerW")?,
            f,
            omegas,
            f_commitments,
            x2,
            pols_map,
        })
    }
}

/// `f` followed by at least one digit and nothing else -- distinguishes the
/// `f0`, `f1`, ... commitments from the `f` array itself.
fn is_commitment_key(key: &str) -> bool {
    let Some(rest) = key.strip_prefix('f') else { return false };
    !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VKEY: &str = include_str!("../tests/fixtures/pilfflonk.vkey");

    fn fixture() -> Value {
        serde_json::from_str(VKEY).expect("fixture parses")
    }

    #[test]
    fn parses_the_pil_fflonk_vkey() {
        let setup = ShPlonkSetup::from_vkey_json(&fixture()).unwrap();

        assert_eq!(setup.n_publics, 3);
        assert_eq!(setup.max_q_degree, 0);
        assert_eq!(setup.power, 8);
        assert_eq!(setup.power_w, 12);
        assert_eq!(setup.f.len(), 9);

        // Omegas are flat in the JSON but collected here; "w" plus the
        // per-arity roots and their shifted variants.
        assert!(setup.omegas.contains_key("w"));
        assert_eq!(setup.omegas["w1"], "1");
        assert!(setup.omegas.contains_key("w2_1d2"));

        // Only the constant-only f_i carry precomputed commitments.
        assert_eq!(setup.f_commitments.keys().collect::<Vec<_>>(), vec!["f0", "f1"]);
        assert_eq!(setup.f_commitments["f0"].len(), 3, "G1 point is [x, y, z]");

        // G2 element for the pairing check.
        assert_eq!(setup.x2.len(), 3);
        assert!(setup.x2.iter().all(|c| c.len() == 2), "each G2 coordinate is an Fq2 pair");
    }

    #[test]
    fn f_entries_describe_their_packing() {
        let setup = ShPlonkSetup::from_vkey_json(&fixture()).unwrap();

        // f0 packs six constant polynomials, all from stage 0, opened at one point.
        let f0 = &setup.f[0];
        assert_eq!(f0.index, 0);
        assert_eq!(f0.opening_points, vec![0]);
        assert_eq!(f0.pols.len(), 6);
        assert_eq!(f0.stages.len(), 1);
        assert_eq!(f0.stages[0].stage, 0);
        assert_eq!(f0.stages[0].pols.len(), 6);

        // f8 is the quotient polynomial alone, in the last stage.
        let f8 = &setup.f[8];
        assert_eq!(f8.pols, vec!["Q"]);
        assert_eq!(f8.stages[0].stage, 4);
        assert_eq!(f8.degree, 780);

        // Every f_i's declared degree must cover its largest component: a
        // component of degree d at slot j occupies index d * nPols + j.
        for f in &setup.f {
            let n_pols = f.pols.len() as u64;
            for stage in &f.stages {
                for (j, pol) in stage.pols.iter().enumerate() {
                    let combined = pol.degree * n_pols + j as u64;
                    assert!(
                        combined <= f.degree + n_pols,
                        "f{} declares degree {} but component {} needs {}",
                        f.index,
                        f.degree,
                        pol.name,
                        combined
                    );
                }
            }
        }
    }

    /// The emitter must reproduce the verifier's format exactly, or existing
    /// verification keys stop being readable.
    #[test]
    fn vkey_round_trips_byte_for_byte() {
        let original = fixture();
        let setup = ShPlonkSetup::from_vkey_json(&original).unwrap();
        let emitted = setup.to_vkey_json();

        assert_eq!(emitted, original, "emitted vkey differs from the pil-fflonk fixture");
    }

    #[test]
    fn round_trips_through_the_model_twice() {
        let setup = ShPlonkSetup::from_vkey_json(&fixture()).unwrap();
        let again = ShPlonkSetup::from_vkey_json(&setup.to_vkey_json()).unwrap();
        assert_eq!(setup, again);
    }

    #[test]
    fn rejects_a_foreign_protocol() {
        let mut v = fixture();
        v["protocol"] = json!("groth16");
        assert!(ShPlonkSetup::from_vkey_json(&v).is_err());
    }

    #[test]
    fn rejects_an_unrecognised_field() {
        let mut v = fixture();
        v.as_object_mut().unwrap().insert("surprise".into(), json!(1));
        // Silently dropping unknown keys would make the round-trip lossy.
        assert!(ShPlonkSetup::from_vkey_json(&v).is_err());
    }

    #[test]
    fn commitment_keys_are_distinguished_from_the_f_array() {
        assert!(is_commitment_key("f0"));
        assert!(is_commitment_key("f12"));
        assert!(!is_commitment_key("f"), "the f array is not a commitment");
        assert!(!is_commitment_key("foo"));
        assert!(!is_commitment_key("w2"));
    }
}
