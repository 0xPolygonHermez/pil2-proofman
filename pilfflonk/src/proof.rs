//! The proof (A.6, D7): its bytes, its JSON view in snarkjs's style, and the publics.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{invalid, PilfflonkError, PilfflonkResult};
use crate::field::{FqBytes, FrBytes, G1Affine, FIELD_BYTES, G1_BYTES};
use crate::global_info::PilfflonkGlobalInfo;
use crate::json::JsonFile;
use crate::layout::{q_pieces, LayoutEntry};
use crate::names::{column_name, commitment_name, evaluation_name, Scope, INV, INV_ZH, W, WP};
use crate::pilfflonk_info::{PilfflonkInfo, PolType};
use crate::tag::{Curve, Protocol};
use crate::vkey::Vkey;

/// The file name of the JSON view of the proof.
pub const PROOF_FILE: &str = "proof.json";

/// The file name of the publics.
pub const PUBLICS_FILE: &str = "publics.json";

/// The extension of a file that holds a proof's bytes (A.6), as `proof.bin`: [`Proof::read`].
pub const PROOF_BYTES_EXTENSION: &str = "bin";

/// A proof. Its bytes (`to_bytes`), as `gen_final_snark_proof` gives the FFLONK's, are in this
/// order, each point `x‖y` and each coordinate or scalar 32 bytes big-endian (A.6):
///
/// 1. `commitments`: the non-fixed `f`, in the global order of A.5; the fixed ones are the
///    vkey's, never the proof's (C.3.1);
/// 2. `w` and `wp`: SHPLONK's `W` and `W'`;
/// 3. `evaluations`: those of the fixed columns of each AIR, then those of the other columns of
///    each instance, each in the order of its AIR's evMap, then the `Q_i(ξ)` of each instance if
///    `Q` is split, in the order of its layout (A.4, step 4);
/// 4. `air_values` (of each instance, in the order of its AIR's `airValuesMap`), `airgroup_values`
///    (of each airgroup with an instance, in the order of its `airgroupValuesMap`) and
///    `proof_values` (in the order of the globalInfo's `proofValuesMap`);
/// 5. `inv` and `inv_zh`, as pil-fflonk.
///
/// The bytes alone do not say where one part ends: `ProofNames` does, from the AIRs of the proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub commitments: Vec<G1Affine>,
    pub w: G1Affine,
    pub wp: G1Affine,
    pub evaluations: Vec<FrBytes>,
    pub air_values: Vec<FrBytes>,
    pub airgroup_values: Vec<FrBytes>,
    pub proof_values: Vec<FrBytes>,
    pub inv: FrBytes,
    pub inv_zh: FrBytes,
}

/// How many values of each kind a proof holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProofShape {
    pub n_commitments: usize,
    pub n_evaluations: usize,
    pub n_air_values: usize,
    pub n_airgroup_values: usize,
    pub n_proof_values: usize,
}

impl ProofShape {
    /// The length of the proof's bytes.
    pub fn byte_len(&self) -> usize {
        let scalars = self.n_evaluations + self.n_air_values + self.n_airgroup_values + self.n_proof_values + 2;
        (self.n_commitments + 2) * G1_BYTES + scalars * FIELD_BYTES
    }
}

impl Proof {
    pub fn shape(&self) -> ProofShape {
        ProofShape {
            n_commitments: self.commitments.len(),
            n_evaluations: self.evaluations.len(),
            n_air_values: self.air_values.len(),
            n_airgroup_values: self.airgroup_values.len(),
            n_proof_values: self.proof_values.len(),
        }
    }

    fn scalars(&self) -> impl Iterator<Item = &FrBytes> {
        self.evaluations
            .iter()
            .chain(&self.air_values)
            .chain(&self.airgroup_values)
            .chain(&self.proof_values)
            .chain([&self.inv, &self.inv_zh])
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.shape().byte_len());
        for point in self.commitments.iter().chain([&self.w, &self.wp]) {
            bytes.extend_from_slice(&point.to_be_bytes());
        }
        for scalar in self.scalars() {
            bytes.extend_from_slice(&scalar.to_be_bytes());
        }
        bytes
    }

    /// Reads the bytes of a proof of this shape. Every coordinate must be below `q` and every
    /// scalar below `r`.
    pub fn from_bytes(bytes: &[u8], shape: &ProofShape) -> PilfflonkResult<Self> {
        if bytes.len() != shape.byte_len() {
            return invalid!("a proof of this shape has {} bytes, not {}", shape.byte_len(), bytes.len());
        }
        let mut reader = ByteReader { bytes };
        let commitments = reader.points(shape.n_commitments)?;
        let w = reader.point()?;
        let wp = reader.point()?;
        let evaluations = reader.scalars(shape.n_evaluations)?;
        let air_values = reader.scalars(shape.n_air_values)?;
        let airgroup_values = reader.scalars(shape.n_airgroup_values)?;
        let proof_values = reader.scalars(shape.n_proof_values)?;
        let inv = reader.scalar()?;
        let inv_zh = reader.scalar()?;
        Ok(Proof { commitments, w, wp, evaluations, air_values, airgroup_values, proof_values, inv, inv_zh })
    }

    /// Reads a proof with the values of `names` from the file at `path`: its bytes (A.6,
    /// [`Proof::from_bytes`]) if the file name ends in [`PROOF_BYTES_EXTENSION`], and its JSON view,
    /// `proof.json` ([`Proof::from_json`]), otherwise.
    pub fn read(path: &Path, names: &ProofNames) -> PilfflonkResult<Self> {
        if path.extension().is_some_and(|extension| extension == PROOF_BYTES_EXTENSION) {
            let bytes = fs::read(path).map_err(|source| PilfflonkError::Io { path: path.to_path_buf(), source })?;
            Proof::from_bytes(&bytes, &names.shape()).map_err(|e| e.in_file(path))
        } else {
            Proof::from_json(&ProofJson::read(path)?, names).map_err(|e| e.in_file(path))
        }
    }

    /// The JSON view of the proof, with the names of `names` (see `crate::names`).
    pub fn to_json(&self, names: &ProofNames) -> PilfflonkResult<ProofJson> {
        if self.shape() != names.shape() {
            return invalid!(
                "a proof of shape {:?} does not have the names of one of shape {:?}",
                self.shape(),
                names.shape()
            );
        }
        let mut polynomials = BTreeMap::new();
        let named_points = names.commitments.iter().map(String::as_str).chain([W, WP]);
        for (name, point) in named_points.zip(self.commitments.iter().chain([&self.w, &self.wp])) {
            if point.is_infinity() {
                return invalid!("{name} is the point at infinity, which the JSON view cannot write as [x, y, \"1\"]");
            }
            polynomials.insert(name.to_string(), SnarkjsG1(*point));
        }
        let evaluations = names.scalars().map(str::to_string).zip(self.scalars().copied()).collect();
        Ok(ProofJson { protocol: Protocol, curve: Curve, polynomials, evaluations })
    }

    /// The proof a JSON view holds: it must have exactly the names of `names`.
    pub fn from_json(json: &ProofJson, names: &ProofNames) -> PilfflonkResult<Self> {
        let expected_points: BTreeSet<&str> = names.commitments.iter().map(String::as_str).chain([W, WP]).collect();
        let expected_scalars: BTreeSet<&str> = names.scalars().collect();
        if !json.polynomials.keys().map(String::as_str).eq(expected_points.iter().copied()) {
            return invalid!("the proof's polynomials are not {expected_points:?}");
        }
        if !json.evaluations.keys().map(String::as_str).eq(expected_scalars.iter().copied()) {
            return invalid!("the proof's evaluations do not have the names of its AIRs");
        }
        // Every name is there: the checks above.
        let point = |name: &str| json.polynomials.get(name).map(|p| p.0).unwrap_or_default();
        let scalar = |name: &str| json.evaluations.get(name).copied().unwrap_or_default();
        let scalars = |list: &[String]| list.iter().map(|name| scalar(name)).collect();
        Ok(Proof {
            commitments: names.commitments.iter().map(|name| point(name)).collect(),
            w: point(W),
            wp: point(WP),
            evaluations: scalars(&names.evaluations),
            air_values: scalars(&names.air_values),
            airgroup_values: scalars(&names.airgroup_values),
            proof_values: scalars(&names.proof_values),
            inv: scalar(INV),
            inv_zh: scalar(INV_ZH),
        })
    }
}

/// Reads the values of a proof's bytes in order. `from_bytes` checks the length first, so the
/// bytes never run out.
struct ByteReader<'a> {
    bytes: &'a [u8],
}

impl ByteReader<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        let (head, rest) = self.bytes.split_at(N.min(self.bytes.len()));
        out[..head.len()].copy_from_slice(head);
        self.bytes = rest;
        out
    }

    fn point(&mut self) -> PilfflonkResult<G1Affine> {
        G1Affine::from_be_bytes(&self.take::<G1_BYTES>())
    }

    fn scalar(&mut self) -> PilfflonkResult<FrBytes> {
        FrBytes::from_be_bytes(self.take::<FIELD_BYTES>())
    }

    fn points(&mut self, n: usize) -> PilfflonkResult<Vec<G1Affine>> {
        (0..n).map(|_| self.point()).collect()
    }

    fn scalars(&mut self, n: usize) -> PilfflonkResult<Vec<FrBytes>> {
        (0..n).map(|_| self.scalar()).collect()
    }
}

/// The names of the values of a proof in its JSON view (`crate::names`), in the order of its
/// bytes. They fix the proof's shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofNames {
    commitments: Vec<String>,
    evaluations: Vec<String>,
    air_values: Vec<String>,
    airgroup_values: Vec<String>,
    proof_values: Vec<String>,
}

impl ProofNames {
    /// The names of a proof of these instances, each given by the pilfflonkinfo of its AIR as
    /// read (and so validated), in canonical order: by `(airgroupId, airId)`, and then by
    /// instance. Refuses two values of the same name.
    pub fn new(global_info: &PilfflonkGlobalInfo, instances: &[&PilfflonkInfo]) -> PilfflonkResult<Self> {
        if instances.is_empty() {
            return invalid!("a proof has one instance at least");
        }
        let n = instances.len();

        // The AIRs with an instance, and the index of each instance within its AIR.
        let mut airs: Vec<&PilfflonkInfo> = Vec::new();
        let mut instance_scopes = Vec::with_capacity(n);
        let mut index_in_air = 0;
        for info in instances {
            let air = global_info.air(info.airgroup_id, info.air_id)?;
            if air.name != info.name {
                return invalid!(
                    "air {}/{} is {} in the globalInfo, not {}",
                    info.airgroup_id,
                    info.air_id,
                    air.name,
                    info.name
                );
            }
            let key = (info.airgroup_id, info.air_id);
            match airs.last().map(|last| (last.airgroup_id, last.air_id)) {
                Some(last) if last == key => index_in_air += 1,
                Some(last) if last > key => {
                    return invalid!("the instances are not in canonical order: air {key:?} after {last:?}");
                }
                _ => {
                    airs.push(info);
                    index_in_air = 0;
                }
            }
            instance_scopes.push(Scope::Instance {
                airgroup: info.airgroup_id,
                air: info.air_id,
                instance: index_in_air,
            });
        }
        let prefixed = |scope: Scope, name: String| format!("{}{name}", scope.prefix(n));

        let mut global_index = airs.iter().map(|air| air.layout.n_fixed() as u64).sum::<u64>();
        let mut commitments = Vec::new();
        for info in instances {
            for _ in info.layout.0.iter().filter(|f| f.stage != 0) {
                commitments.push(commitment_name(global_index));
                global_index += 1;
            }
        }

        let column = |info: &PilfflonkInfo, pol_type: PolType, id: u64| -> PilfflonkResult<String> {
            match info.pol(pol_type, id) {
                Some(pol) => Ok(column_name(&pol.name, &pol.lengths)),
                None => invalid!("air {} has no {} {id}", info.name, pol_type.as_str()),
            }
        };
        let mut evaluations = Vec::new();
        for air in &airs {
            let scope = Scope::Air { airgroup: air.airgroup_id, air: air.air_id };
            for e in air.ev_map.iter().filter(|e| e.pol_type == PolType::Const) {
                evaluations.push(prefixed(scope, evaluation_name(&column(air, e.pol_type, e.id)?, e.prime)));
            }
        }
        for (info, scope) in instances.iter().zip(&instance_scopes) {
            for e in info.ev_map.iter().filter(|e| e.pol_type == PolType::Cm) {
                evaluations.push(prefixed(*scope, evaluation_name(&column(info, e.pol_type, e.id)?, e.prime)));
            }
        }
        for (info, scope) in instances.iter().zip(&instance_scopes) {
            if q_pieces(info.q_deg, info.max_q_degree) > 1 {
                for f in info.layout.0.iter().filter(|f| f.stage == info.q_stage()) {
                    for pol in &f.pols {
                        evaluations.push(prefixed(*scope, column(info, PolType::Cm, pol.id)?));
                    }
                }
            }
        }

        let mut air_values = Vec::new();
        for (info, scope) in instances.iter().zip(&instance_scopes) {
            for v in &info.air_values_map {
                air_values.push(prefixed(*scope, column_name(&v.name, &v.lengths)));
            }
        }
        let mut airgroup_values = Vec::new();
        let mut airgroups_done = BTreeSet::new();
        for air in &airs {
            if airgroups_done.insert(air.airgroup_id) {
                let scope = Scope::Airgroup { airgroup: air.airgroup_id };
                for v in &air.airgroup_values_map {
                    airgroup_values.push(prefixed(scope, column_name(&v.name, &v.lengths)));
                }
            }
        }
        let proof_values = global_info.proof_values_map.iter().map(|v| column_name(&v.name, &v.lengths)).collect();

        ProofNames { commitments, evaluations, air_values, airgroup_values, proof_values }.unique()
    }

    /// The names of a proof of `vkey`: those [`ProofNames::new`] gives the proof of one instance of
    /// its AIR, the one a vkey of format 1 describes (D2), from what the vkey holds, as the JS
    /// verifier names them (`vkey.js`, `fromObjectVk`). The columns are named by the layout
    /// ([`LayoutPol::name`](crate::LayoutPol)), the `f` by their index in it, and a proof of
    /// format 1 has no air, airgroup or proof values. Refuses two values of the same name.
    pub fn of_vkey(vkey: &Vkey) -> PilfflonkResult<Self> {
        let layout = &vkey.layout.0;
        let n_fixed = vkey.layout.n_fixed();
        let q_stage = layout.last().map_or(0, |f| f.stage);
        let commitments = (n_fixed..layout.len()).map(|g| commitment_name(g as u64)).collect();

        // The fixed columns are in the f of stage 0, the committed ones in the others (A.2).
        let column = |pol_type: PolType, id: u64| -> PilfflonkResult<&str> {
            let packs = |f: &&LayoutEntry| (f.stage == 0) == (pol_type == PolType::Const);
            match layout.iter().filter(packs).flat_map(|f| &f.pols).find(|p| p.id == id) {
                Some(pol) => Ok(&pol.name),
                None => invalid!("the evMap has {} {id}, which no f of the vkey's layout packs", pol_type.as_str()),
            }
        };
        let mut evaluations = Vec::with_capacity(vkey.ev_map.len());
        for pol_type in [PolType::Const, PolType::Cm] {
            for e in vkey.ev_map.iter().filter(|e| e.pol_type == pol_type) {
                evaluations.push(evaluation_name(column(e.pol_type, e.id)?, e.prime));
            }
        }
        if q_pieces(vkey.q_deg, vkey.max_q_degree) > 1 {
            let pieces = layout.iter().filter(|f| f.stage == q_stage).flat_map(|f| &f.pols);
            evaluations.extend(pieces.map(|pol| pol.name.clone()));
        }
        let none = Vec::new;
        ProofNames { commitments, evaluations, air_values: none(), airgroup_values: none(), proof_values: none() }
            .unique()
    }

    /// Refuses names of which two values share one: the JSON view would lose a value.
    fn unique(self) -> PilfflonkResult<Self> {
        let mut seen = BTreeSet::new();
        for name in self.scalars() {
            if !seen.insert(name) {
                return invalid!("two values of the proof are named {name:?} in its JSON view (see crate::names)");
            }
        }
        Ok(self)
    }

    pub fn shape(&self) -> ProofShape {
        ProofShape {
            n_commitments: self.commitments.len(),
            n_evaluations: self.evaluations.len(),
            n_air_values: self.air_values.len(),
            n_airgroup_values: self.airgroup_values.len(),
            n_proof_values: self.proof_values.len(),
        }
    }

    pub fn commitments(&self) -> &[String] {
        &self.commitments
    }

    pub fn evaluations(&self) -> &[String] {
        &self.evaluations
    }

    pub fn air_values(&self) -> &[String] {
        &self.air_values
    }

    pub fn airgroup_values(&self) -> &[String] {
        &self.airgroup_values
    }

    pub fn proof_values(&self) -> &[String] {
        &self.proof_values
    }

    /// The names of the `evaluations` object, in the order of the bytes.
    fn scalars(&self) -> impl Iterator<Item = &str> {
        self.evaluations
            .iter()
            .chain(&self.air_values)
            .chain(&self.airgroup_values)
            .chain(&self.proof_values)
            .map(String::as_str)
            .chain([INV, INV_ZH])
    }
}

/// The JSON view of a proof, `proof.json`: `{"protocol": "pilfflonk", "curve": "bn128",
/// "polynomials": {name: [x, y, "1"]}, "evaluations": {name: value}}` (A.6, D7), as
/// `snark_proof_to_json` writes the FFLONK's and pil-fflonk wrote its own. Keys in sorted order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofJson {
    pub protocol: Protocol,
    pub curve: Curve,
    pub polynomials: BTreeMap<String, SnarkjsG1>,
    pub evaluations: BTreeMap<String, FrBytes>,
}

impl JsonFile for ProofJson {
    fn validate(&self) -> PilfflonkResult<()> {
        Ok(())
    }
}

/// A G1 point as snarkjs writes one in a proof, projective with `z = 1`: `["x", "y", "1"]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnarkjsG1(pub G1Affine);

impl Serialize for SnarkjsG1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.0.x, &self.0.y, "1").serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SnarkjsG1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (x, y, z) = <(FqBytes, FqBytes, String)>::deserialize(deserializer)?;
        if z != "1" {
            return Err(D::Error::custom(format!("a proof's point is [x, y, \"1\"], and this one has z = {z:?}")));
        }
        Ok(SnarkjsG1(G1Affine { x, y }))
    }
}

/// `publics.json`: the publics, as decimal strings in the order of the globalInfo's `publicsMap`
/// (A.6), as pil-fflonk and the final wrap write them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Publics(pub Vec<FrBytes>);

impl JsonFile for Publics {
    fn validate(&self) -> PilfflonkResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snarkjs_points_have_z_1() {
        let p = SnarkjsG1(G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(2) });
        assert_eq!(serde_json::to_string(&p).unwrap(), r#"["1","2","1"]"#);
        assert_eq!(serde_json::from_str::<SnarkjsG1>(r#"["1","2","1"]"#).unwrap(), p);
        assert!(serde_json::from_str::<SnarkjsG1>(r#"["1","2","0"]"#).is_err());
        assert!(serde_json::from_str::<SnarkjsG1>(r#"["1","2"]"#).is_err());
    }

    #[test]
    fn publics_are_an_array_of_decimal_strings() {
        let publics = Publics(vec![FrBytes::from_u64(1), FrBytes::from_u64(2)]);
        let text = publics.to_json_string().unwrap();
        assert_eq!(text, "[\n \"1\",\n \"2\"\n]");
        assert_eq!(Publics::from_json_str(&text).unwrap(), publics);
    }

    #[test]
    fn the_byte_length_follows_a6() {
        let shape =
            ProofShape { n_commitments: 3, n_evaluations: 5, n_air_values: 1, n_airgroup_values: 0, n_proof_values: 2 };
        assert_eq!(shape.byte_len(), (3 + 2) * 64 + (5 + 1 + 2 + 2) * 32);
    }
}
