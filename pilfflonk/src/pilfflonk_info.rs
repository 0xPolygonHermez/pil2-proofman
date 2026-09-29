//! `<air>.pilfflonkinfo.json` (A.6): the part of the STARK's `starkinfo.json` pilfflonk keeps, and
//! the layout of the AIR's `f_i`. `pil2-stark/src/pilfflonk/pilfflonk_info.{hpp,cpp}` reads it for
//! the prover.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{invalid, PilfflonkResult};
use crate::global_info::MAX_NBITS;
use crate::json::JsonFile;
use crate::layout::{q_pieces, Layout, LayoutCheck};

/// `<air>.pilfflonkinfo.json`. Its fields keep the meaning they have in `starkinfo.json`, with
/// every dimension 1: BN254 has no extension field.
///
/// There is no `starkStruct`, nothing of FRI, no custom commits (the setup refuses them, P5) and
/// no publics or proof values (they are the globalInfo's). `nBits`, which the STARK keeps in
/// `starkStruct`, is here; `name`, `airgroupId` and `airId` say which AIR it is, as in the
/// starkinfo.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PilfflonkInfo {
    pub name: String,
    pub airgroup_id: u64,
    pub air_id: u64,
    /// `N = 2^nBits` rows.
    pub n_bits: u64,
    pub n_stages: u64,
    pub n_constants: u64,
    /// The committed columns of stages `1 … nStages`, and the pieces of `Q` as stage `nStages + 1`.
    pub cm_pols_map: Vec<PolMapEntry>,
    /// The fixed columns, stage 0.
    pub const_pols_map: Vec<PolMapEntry>,
    pub challenges_map: Vec<ChallengeMapEntry>,
    pub air_values_map: Vec<NameStageEntry>,
    pub airgroup_values_map: Vec<NameStageEntry>,
    /// The width of each section: `const` and `cm1 … cm<nStages+1>`.
    pub map_sections_n: BTreeMap<String, u64>,
    /// Every offset some column is opened at, increasing.
    pub opening_points: Vec<i64>,
    pub boundaries: Vec<Boundary>,
    /// The evaluations of the proof, in the order the proof has them (A.6): each `(column,
    /// offset)` the layout opens, but `Q`'s.
    pub ev_map: Vec<EvMapEntry>,
    pub q_deg: u64,
    /// Always 1.
    pub q_dim: u64,
    /// 0 when `Q` is not split (A.1).
    pub max_q_degree: u64,
    /// The expression of the constraint polynomial.
    pub c_exp_id: u64,
    pub layout: Layout,
}

/// A column: an entry of `cmPolsMap` or `constPolsMap`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolMapEntry {
    pub stage: u64,
    pub name: String,
    /// Always 1.
    pub dim: u64,
    /// Its index in the map.
    pub pols_map_id: u64,
    /// Its index among the columns of its stage.
    pub stage_id: u64,
    /// Its indices, if it is an element of an array column.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lengths: Vec<u64>,
    /// An intermediate polynomial of the setup, `expId` its expression (spec §4.2.3).
    #[serde(default, skip_serializing_if = "is_false")]
    pub im_pol: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exp_id: Option<u64>,
    /// Its position in its section.
    pub stage_pos: u64,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ChallengeMapEntry {
    pub name: String,
    pub stage: u64,
    /// Always 1.
    pub dim: u64,
    pub stage_id: u64,
}

/// A name and a stage: an entry of `airValuesMap`, `airgroupValuesMap`, or of the globalInfo's
/// `publicsMap` and `proofValuesMap`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NameStageEntry {
    pub name: String,
    pub stage: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lengths: Vec<u64>,
}

/// The kind of column an evaluation is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolType {
    /// `cmPolsMap`.
    Cm,
    /// `constPolsMap`.
    Const,
}

impl PolType {
    pub fn as_str(self) -> &'static str {
        match self {
            PolType::Cm => "cm",
            PolType::Const => "const",
        }
    }
}

/// An evaluation: column `id` of `type` at `ξ·ω^prime`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvMapEntry {
    #[serde(rename = "type")]
    pub pol_type: PolType,
    pub id: u64,
    pub prime: i64,
    /// The position of `prime` in `openingPoints`.
    pub opening_pos: u64,
}

/// The domain of a constraint (A.1). JSON: `{"name": "everyRow"}`, `{"name": "firstRow"}`,
/// `{"name": "lastRow"}` and `{"name": "everyFrame", "offsetMin": a, "offsetMax": b}`, as in the
/// starkinfo.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "BoundaryRepr", into = "BoundaryRepr")]
pub enum Boundary {
    EveryRow,
    FirstRow,
    LastRow,
    /// Every row but the first `offset_min` and the last `offset_max`.
    EveryFrame {
        offset_min: u64,
        offset_max: u64,
    },
}

/// A boundary as JSON has it. serde's internally tagged enums let a unit variant carry any other
/// field, so the strict form goes through this.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BoundaryRepr {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    offset_min: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    offset_max: Option<u64>,
}

impl TryFrom<BoundaryRepr> for Boundary {
    type Error = String;

    fn try_from(repr: BoundaryRepr) -> Result<Self, String> {
        match (repr.name.as_str(), repr.offset_min, repr.offset_max) {
            ("everyRow", None, None) => Ok(Boundary::EveryRow),
            ("firstRow", None, None) => Ok(Boundary::FirstRow),
            ("lastRow", None, None) => Ok(Boundary::LastRow),
            ("everyFrame", Some(offset_min), Some(offset_max)) => Ok(Boundary::EveryFrame { offset_min, offset_max }),
            (name, _, _) => Err(format!(
                "{name:?} is not a boundary: everyRow, firstRow and lastRow have no offsets, and everyFrame has \
                 offsetMin and offsetMax"
            )),
        }
    }
}

impl From<Boundary> for BoundaryRepr {
    fn from(boundary: Boundary) -> Self {
        let (name, offset_min, offset_max) = match boundary {
            Boundary::EveryRow => ("everyRow", None, None),
            Boundary::FirstRow => ("firstRow", None, None),
            Boundary::LastRow => ("lastRow", None, None),
            Boundary::EveryFrame { offset_min, offset_max } => ("everyFrame", Some(offset_min), Some(offset_max)),
        };
        BoundaryRepr { name: name.to_string(), offset_min, offset_max }
    }
}

impl PilfflonkInfo {
    /// The column an evaluation is of.
    pub fn pol(&self, pol_type: PolType, id: u64) -> Option<&PolMapEntry> {
        let map = match pol_type {
            PolType::Cm => &self.cm_pols_map,
            PolType::Const => &self.const_pols_map,
        };
        usize::try_from(id).ok().and_then(|id| map.get(id))
    }

    /// The stage of `Q`.
    pub fn q_stage(&self) -> u64 {
        self.n_stages.saturating_add(1)
    }

    fn check_pol_map(&self, pol_type: PolType) -> PilfflonkResult<()> {
        let (map, what) = match pol_type {
            PolType::Cm => (&self.cm_pols_map, "cmPolsMap"),
            PolType::Const => (&self.const_pols_map, "constPolsMap"),
        };
        let mut per_stage: BTreeMap<u64, (BTreeSet<u64>, BTreeSet<u64>)> = BTreeMap::new();
        for (i, p) in map.iter().enumerate() {
            let stages_ok = match pol_type {
                PolType::Const => p.stage == 0,
                PolType::Cm => (1..=self.q_stage()).contains(&p.stage),
            };
            if !stages_ok {
                return invalid!("{what}[{i}] ({}) is of stage {}, which is not one of its map", p.name, p.stage);
            }
            if p.dim != 1 || p.pols_map_id != i as u64 {
                return invalid!("{what}[{i}] ({}) must have dim 1 and polsMapId {i}", p.name);
            }
            if p.im_pol != p.exp_id.is_some() {
                return invalid!("{what}[{i}] ({}) must have an expId if and only if it is an imPol", p.name);
            }
            let (ids, positions) = per_stage.entry(p.stage).or_default();
            ids.insert(p.stage_id);
            positions.insert(p.stage_pos);
        }
        // Every column is one element wide: in each stage stageId and stagePos number the columns.
        for (stage, (ids, positions)) in &per_stage {
            let n = map.iter().filter(|p| p.stage == *stage).count() as u64;
            let numbered = |set: &BTreeSet<u64>| set.len() as u64 == n && set.last().is_none_or(|last| *last == n - 1);
            if !numbered(ids) || !numbered(positions) {
                return invalid!("the stageId and the stagePos of stage {stage} of {what} must each be 0 … {}", n - 1);
            }
            let section = if *stage == 0 { "const".to_string() } else { format!("cm{stage}") };
            if self.map_sections_n.get(&section) != Some(&n) {
                return invalid!("mapSectionsN.{section} must be {n}, the columns of stage {stage}");
            }
        }
        Ok(())
    }
}

impl JsonFile for PilfflonkInfo {
    fn validate(&self) -> PilfflonkResult<()> {
        if self.n_bits > MAX_NBITS {
            return invalid!("nBits is {}, above {MAX_NBITS}", self.n_bits);
        }
        if self.n_stages == 0 {
            return invalid!("nStages is 0: an AIR has one stage at least");
        }
        if self.q_dim != 1 {
            return invalid!("qDim is {}, and BN254 has no extension field: it is 1", self.q_dim);
        }
        if self.n_constants != self.const_pols_map.len() as u64 {
            return invalid!("nConstants is {} but constPolsMap has {}", self.n_constants, self.const_pols_map.len());
        }

        // A section per stage, and "const": this also bounds nStages by the size of the file.
        if self.n_stages.checked_add(2) != Some(self.map_sections_n.len() as u64) {
            return invalid!(
                "mapSectionsN has {} sections and nStages is {}",
                self.map_sections_n.len(),
                self.n_stages
            );
        }
        let mut sections: BTreeSet<String> = (1..=self.q_stage()).map(|s| format!("cm{s}")).collect();
        sections.insert("const".to_string());
        if !self.map_sections_n.keys().cloned().eq(sections.iter().cloned()) {
            return invalid!("mapSectionsN must have exactly the sections {sections:?}");
        }
        if self.map_sections_n.get("const") != Some(&self.n_constants) {
            return invalid!("mapSectionsN.const must be nConstants, {}", self.n_constants);
        }
        self.check_pol_map(PolType::Const)?;
        self.check_pol_map(PolType::Cm)?;
        for stage in 1..=self.q_stage() {
            if !self.cm_pols_map.iter().any(|p| p.stage == stage)
                && self.map_sections_n.get(&format!("cm{stage}")) != Some(&0)
            {
                return invalid!("mapSectionsN.cm{stage} must be 0: stage {stage} has no columns");
            }
        }

        for c in &self.challenges_map {
            if c.dim != 1 || !(1..=self.n_stages + 2).contains(&c.stage) {
                return invalid!("challenge {} must have dim 1 and a stage from 1 to nStages + 2", c.name);
            }
        }
        for v in self.air_values_map.iter().chain(&self.airgroup_values_map) {
            if !(1..=self.n_stages).contains(&v.stage) {
                return invalid!("value {} must have a stage from 1 to nStages", v.name);
            }
        }
        if self.opening_points.windows(2).any(|w| w[0] >= w[1]) {
            return invalid!("openingPoints {:?} must be increasing", self.opening_points);
        }
        for (i, b) in self.boundaries.iter().enumerate() {
            if self.boundaries[..i].contains(b) {
                return invalid!("boundary {b:?} is there twice");
            }
        }

        for (i, e) in self.ev_map.iter().enumerate() {
            let Some(pol) = self.pol(e.pol_type, e.id) else {
                return invalid!("evMap[{i}] is {} {}, which is not in the pol map", e.pol_type.as_str(), e.id);
            };
            if pol.stage == self.q_stage() {
                return invalid!("evMap[{i}] is a piece of Q, which the verifier computes (A.1)");
            }
            let at_position = usize::try_from(e.opening_pos).ok().and_then(|p| self.opening_points.get(p));
            if at_position != Some(&e.prime) {
                return invalid!(
                    "evMap[{i}] has openingPos {}, and openingPoints has not {} there",
                    e.opening_pos,
                    e.prime
                );
            }
        }

        self.layout.check(&LayoutCheck {
            n_bits: self.n_bits,
            q_stage: self.q_stage(),
            q_pieces: q_pieces(self.q_deg, self.max_q_degree),
            ev_map: &self.ev_map,
            pol_maps: Some((&self.const_pols_map, &self.cm_pols_map)),
        })?;
        let q_columns = self.cm_pols_map.iter().filter(|p| p.stage == self.q_stage()).count() as u64;
        if q_columns != q_pieces(self.q_deg, self.max_q_degree) {
            return invalid!(
                "cmPolsMap has {q_columns} pieces of Q, and Q is made of {}",
                q_pieces(self.q_deg, self.max_q_degree)
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries_are_tagged_by_name() {
        let every_frame = Boundary::EveryFrame { offset_min: 1, offset_max: 2 };
        assert_eq!(serde_json::to_string(&Boundary::EveryRow).unwrap(), r#"{"name":"everyRow"}"#);
        assert_eq!(
            serde_json::to_string(&every_frame).unwrap(),
            r#"{"name":"everyFrame","offsetMin":1,"offsetMax":2}"#
        );
        assert_eq!(serde_json::from_str::<Boundary>(r#"{"name":"lastRow"}"#).unwrap(), Boundary::LastRow);
        assert!(serde_json::from_str::<Boundary>(r#"{"name":"lastRow","offsetMin":1}"#).is_err());
        assert!(serde_json::from_str::<Boundary>(r#"{"name":"everyFrame","offsetMin":1}"#).is_err());
        assert!(serde_json::from_str::<Boundary>(r#"{"name":"someRow"}"#).is_err());
    }

    #[test]
    fn optional_fields_are_left_out_when_absent() {
        let plain = PolMapEntry {
            stage: 1,
            name: "a".into(),
            dim: 1,
            pols_map_id: 0,
            stage_id: 0,
            lengths: vec![],
            im_pol: false,
            exp_id: None,
            stage_pos: 0,
        };
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            r#"{"stage":1,"name":"a","dim":1,"polsMapId":0,"stageId":0,"stagePos":0}"#
        );
        let im = PolMapEntry { lengths: vec![2], im_pol: true, exp_id: Some(5), ..plain };
        let json = serde_json::to_string(&im).unwrap();
        assert_eq!(
            json,
            r#"{"stage":1,"name":"a","dim":1,"polsMapId":0,"stageId":0,"lengths":[2],"imPol":true,"expId":5,"stagePos":0}"#
        );
        assert_eq!(serde_json::from_str::<PolMapEntry>(&json).unwrap(), im);
        assert!(serde_json::from_str::<EvMapEntry>(r#"{"type":"custom","id":0,"prime":0,"openingPos":0}"#).is_err());
        assert!(serde_json::from_str::<EvMapEntry>(r#"{"type":"cm","id":0,"prime":0,"openingPos":0,"commitId":0}"#)
            .is_err());
    }
}
