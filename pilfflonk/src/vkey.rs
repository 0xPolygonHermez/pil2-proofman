//! `pilfflonk.vkey.json` (A.6, §4.2.5): the self-contained verification key, and its digest.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{Error as _, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::error::{invalid, PilfflonkResult};
use crate::field::{Digest, G1Affine, G2Affine};
use crate::global_info::{FORMAT_VERSION, MAX_NBITS};
use crate::json::{canonical_json, serialize_sorted, JsonFile};
use crate::layout::{q_pieces, Layout, LayoutCheck};
use crate::pilfflonk_info::{Boundary, EvMapEntry, PilfflonkInfo};
use crate::q_verifier::{check_q_verifier, QVerifierShape};
use crate::tag::{Curve, Protocol};
use crate::verkey::AirVerkey;

/// What the digest's preimage starts with (A.6).
pub const DIGEST_DOMAIN: &[u8] = b"pilfflonk-v1";

/// `pilfflonk.vkey.json`: everything the verifier needs, and nothing else, as snarkjs's
/// `verification_key.json` (A.6). Big integers and points are decimal strings.
///
/// Format version 1 holds one AIR (the v1 scope, D2): its evMap, layout, degrees and code are
/// the AIR's, and `power` is its `nBits`.
///
/// It also holds the AIR's `boundaries`, which A.6's list leaves out: the `qVerifier` code refers
/// to the zerofiers `Z_D(ξ)` by their index in them (its `Zi` operands), and the verifier cannot
/// compute `Q(ξ)` without them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Vkey {
    pub protocol: Protocol,
    pub curve: Curve,
    pub format_version: u64,
    pub n_public: u64,
    /// `nBits` of the AIR.
    pub power: u64,
    /// The least common multiple of the layout's `k`: `ξ = xiSeed^powerW` (A.2, rule 5).
    pub power_w: u64,
    /// `[τ]₂`.
    #[serde(rename = "X_2")]
    pub x_2: G2Affine,
    /// The challenges of each stage, as the globalInfo's `numChallenges`: one entry per stage, and
    /// none of stage 1 (A.4 squeezes no challenge before its commitments).
    pub num_challenges: Vec<u64>,
    pub ev_map: Vec<EvMapEntry>,
    pub layout: Layout,
    /// The AIR's boundaries: `everyRow` first, whose `Zi` is `1/Z_H` (A.6), and each `everyFrame`
    /// leaving a row.
    pub boundaries: Vec<Boundary>,
    /// The commitments of the fixed `f_i`, the AIR's verkey: keys `f0`, `f1`, … as in snarkjs's
    /// and pil-fflonk's vkeys, `f<i>` for layout entry `i`, which in a proof of one AIR is also its
    /// global index (A.5). The verifier takes them from here and never from the proof (A.5).
    #[serde(flatten)]
    pub fixed_commitments: FixedCommitments,
    pub q_deg: u64,
    pub max_q_degree: u64,
    /// The `qVerifier` of `<air>.verifierinfo.json`, as `pil-info` writes it (the STARK's format),
    /// copied as it is. This crate does not run it, but checks that the verifier can
    /// (`crate::q_verifier`). Written with its keys sorted.
    #[serde(serialize_with = "serialize_sorted")]
    pub q_verifier: Value,
    /// `keccak256("pilfflonk-v1" ‖ canonical(vkey without digest))` (A.6).
    pub digest: Digest,
}

/// The fixed commitments of the vkey, `f0 … f<n-1>` at its top level.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FixedCommitments(pub Vec<G1Affine>);

impl Serialize for FixedCommitments {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (i, point) in self.0.iter().enumerate() {
            map.serialize_entry(&format!("f{i}"), point)?;
        }
        map.end()
    }
}

/// The index of a key `f<i>`, `i` in decimal without leading zeros.
fn fixed_index(key: &str) -> Option<usize> {
    let digits = key.strip_prefix('f')?;
    let canonical =
        !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) && (digits == "0" || !digits.starts_with('0'));
    if canonical {
        digits.parse().ok()
    } else {
        None
    }
}

impl<'de> Deserialize<'de> for FixedCommitments {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FixedVisitor;

        impl<'de> Visitor<'de> for FixedVisitor {
            type Value = FixedCommitments;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("the fixed commitments f0, f1, … of a vkey")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut points = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    // Flattened: every key the vkey does not name comes here.
                    let Some(i) = fixed_index(&key) else {
                        return Err(A::Error::custom(format!("unknown vkey field {key:?}")));
                    };
                    if points.insert(i, map.next_value::<G1Affine>()?).is_some() {
                        return Err(A::Error::custom(format!("vkey field {key:?} is there twice")));
                    }
                }
                // f0 … f<n-1>, none missing: the i-th key in order is f<i>.
                match points.keys().enumerate().find(|(position, i)| position != *i) {
                    Some((missing, _)) => {
                        Err(A::Error::custom(format!("the vkey has fixed commitments but not f{missing}")))
                    }
                    None => Ok(FixedCommitments(points.into_values().collect())),
                }
            }
        }

        deserializer.deserialize_map(FixedVisitor)
    }
}

impl Vkey {
    /// The vkey of the AIR `info` describes, before its digest is set (it is zero until `seal`).
    pub fn new(
        info: &PilfflonkInfo,
        n_public: u64,
        num_challenges: Vec<u64>,
        x_2: G2Affine,
        verkey: &AirVerkey,
        q_verifier: Value,
    ) -> PilfflonkResult<Self> {
        let vkey = Vkey {
            protocol: Protocol,
            curve: Curve,
            format_version: FORMAT_VERSION,
            n_public,
            power: info.n_bits,
            power_w: info.layout.power_w()?,
            x_2,
            num_challenges,
            ev_map: info.ev_map.clone(),
            layout: info.layout.clone(),
            boundaries: info.boundaries.clone(),
            fixed_commitments: FixedCommitments(verkey.0.clone()),
            q_deg: info.q_deg,
            max_q_degree: info.max_q_degree,
            q_verifier,
            digest: Digest::default(),
        };
        vkey.validate()?;
        Ok(vkey)
    }

    /// `"pilfflonk-v1" ‖ canonical(vkey without digest)`: what the digest hashes (A.6).
    pub fn digest_preimage(&self) -> PilfflonkResult<Vec<u8>> {
        let mut value = serde_json::to_value(self)?;
        if let Value::Object(map) = &mut value {
            map.remove("digest");
        }
        let mut preimage = DIGEST_DOMAIN.to_vec();
        preimage.extend_from_slice(canonical_json(&value)?.as_bytes());
        Ok(preimage)
    }

    /// The vkey with its digest set, `keccak256` the hash (Keccak-256, not SHA3-256).
    pub fn seal(mut self, keccak256: impl FnOnce(&[u8]) -> [u8; 32]) -> PilfflonkResult<Self> {
        self.digest = Digest(keccak256(&self.digest_preimage()?));
        Ok(self)
    }

    /// Whether `digest` is the digest of the rest of the vkey.
    pub fn digest_matches(&self, keccak256: impl FnOnce(&[u8]) -> [u8; 32]) -> PilfflonkResult<bool> {
        Ok(Digest(keccak256(&self.digest_preimage()?)) == self.digest)
    }
}

impl JsonFile for Vkey {
    fn validate(&self) -> PilfflonkResult<()> {
        if self.format_version != FORMAT_VERSION {
            return invalid!(
                "formatVersion {} is not supported; this build reads {FORMAT_VERSION}",
                self.format_version
            );
        }
        if self.power > MAX_NBITS {
            return invalid!("power {} is above {MAX_NBITS}", self.power);
        }
        let q_stage = self.layout.0.last().map_or(0, |f| f.stage);
        if q_stage == 0 {
            return invalid!("the layout has no f for Q");
        }
        self.layout.check(&LayoutCheck {
            n_bits: self.power,
            q_stage,
            q_pieces: q_pieces(self.q_deg, self.max_q_degree),
            ev_map: &self.ev_map,
            pol_maps: None,
        })?;
        if self.power_w != self.layout.power_w()? {
            return invalid!(
                "powerW is {} and the least common multiple of the layout's k {}",
                self.power_w,
                self.layout.power_w()?
            );
        }
        if self.fixed_commitments.0.len() != self.layout.n_fixed() {
            return invalid!(
                "the vkey has {} fixed commitments and its layout {} fixed f",
                self.fixed_commitments.0.len(),
                self.layout.n_fixed()
            );
        }

        // What the verifier needs to replay A.4 and compute Q(ξ), as it checks it
        // (pilfflonk/js/src/vkey.js, fromObjectVk).
        let n_stages = q_stage - 1;
        if self.num_challenges.len() as u64 != n_stages {
            return invalid!("numChallenges has {} stages, and the layout {n_stages}", self.num_challenges.len());
        }
        if self.num_challenges.first() != Some(&0) {
            return invalid!(
                "numChallenges {:?} must start with 0: A.4 squeezes no challenge of stage 1",
                self.num_challenges
            );
        }
        if self.boundaries.first() != Some(&Boundary::EveryRow) {
            return invalid!("boundaries[0] must be everyRow, whose Zi is 1/Z_H (A.6)");
        }
        let n_rows = 1u64 << self.power;
        for (i, b) in self.boundaries.iter().enumerate() {
            if self.boundaries[..i].contains(b) {
                return invalid!("boundary {b:?} is there twice");
            }
            if let Boundary::EveryFrame { offset_min, offset_max } = *b {
                if offset_min.checked_add(offset_max).is_none_or(|excluded| excluded >= n_rows) {
                    return invalid!(
                        "boundaries[{i}] is everyFrame {{{offset_min}, {offset_max}}}: no row of {n_rows} is left"
                    );
                }
            }
        }
        check_q_verifier(
            &self.q_verifier,
            &QVerifierShape {
                n_evaluations: self.ev_map.len(),
                n_public: self.n_public,
                n_boundaries: self.boundaries.len(),
                num_challenges: &self.num_challenges,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_keys_are_f_and_an_index_without_leading_zeros() {
        assert_eq!(fixed_index("f0"), Some(0));
        assert_eq!(fixed_index("f12"), Some(12));
        for key in ["f", "f01", "g0", "f-1", "f1a", "F1", "qDeg"] {
            assert_eq!(fixed_index(key), None, "{key}");
        }
    }
}
