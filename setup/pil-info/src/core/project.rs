//! Pure projection of the layout, openings and checked im choice into the starkinfo fields.

use serde::{Serialize, Serializer};

use crate::core::im::{Checked, ImChoice};
use crate::core::ir::{Boundary, ColType, ConstraintSystem, CustomCommit, Lengths, NamedValue};
use crate::core::layout::{ChallengeEntry, Layout, PolMap};
use crate::core::openings::{fri_batch_size, EvEntry, Openings};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarkParams {
    pub n_bits: u32,
    pub n_bits_ext: u32,
    /// The arena id of Q (`Roles::c_exp`).
    pub c_exp_id: u32,
}

/// What a verifier reads (stark struct sizes, maps that resolve the evMap, Merkle widths): a bug here costs soundness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierView {
    pub n_stages: u32,
    pub n_constants: u32,
    pub n_publics: u32,
    pub n_bits: u32,
    pub n_bits_ext: u32,
    pub cm_pols_map: Vec<PolMap>,
    pub const_pols_map: Vec<PolMap>,
    pub custom_pols_map: Vec<Vec<PolMap>>,
    /// Commitment widths per section (the verifier's Merkle leaves).
    pub map_sections_n: Vec<(String, u32)>,
    pub challenges_map: Vec<ChallengeEntry>,
    pub publics_map: Vec<NamedValue>,
    pub proof_values_map: Vec<NamedValue>,
    pub air_values_map: Vec<NamedValue>,
    pub airgroup_values_map: Vec<NamedValue>,
    pub ev_map: Vec<EvEntry>,
    pub opening_points: Vec<i64>,
    /// Distinct constraint boundaries in first-appearance order; `[EveryRow]` under Deep.
    pub boundaries: Vec<Boundary>,
    pub q_deg: u32,
    pub q_dim: u32,
    /// Pilout constraints plus one `im_e - e` per promoted expression.
    pub n_constraints: u32,
    pub fri_batch_size: u32,
    /// Tags the qVerifier code; the verifier reads it.
    pub c_exp_id: u32,
    pub custom_commits: Vec<CustomCommit>,
    /// `(aggType, stage)` of each of the airgroup's values.
    pub airgroup_values: Vec<(i32, u32)>,
}

/// Read by the prover only: a bug in ProverLayout costs completeness only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverLayout {
    /// Expression id of the k-th im column.
    pub im_exp_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarkInfoCore {
    pub verifier: VerifierView,
    pub prover: ProverLayout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectError {
    Mismatch { what: String },
}

/// Copies and counts only: no decision is taken here.
pub fn stark_info(
    cs: &ConstraintSystem,
    im: &ImChoice,
    chk: &Checked,
    lay: &Layout,
    ops: &Openings,
    p: &StarkParams,
) -> Result<StarkInfoCore, ProjectError> {
    if cs.airgroup_agg_types.len() != cs.airgroup_values.len() {
        return Err(ProjectError::Mismatch { what: "airgroup values and their aggTypes".into() });
    }
    // distinct kinds in first-appearance order over the constraints then the im roots (everyRow), as today
    let roots = cs.constraints.iter().map(|c| &c.boundary);
    let im_roots = im.exprs.iter().map(|_| &Boundary::EveryRow);
    let mut boundaries: Vec<Boundary> = Vec::new();
    for b in roots.chain(im_roots) {
        if !boundaries.iter().any(|x| std::mem::discriminant(x) == std::mem::discriminant(b)) {
            boundaries.push(b.clone());
        }
    }
    if boundaries.is_empty() {
        boundaries.push(Boundary::EveryRow);
    }
    let verifier = VerifierView {
        n_stages: cs.n_stages,
        // counts below the ingestion limit
        n_constants: cs.fixed.len() as u32,
        n_publics: cs.publics.len() as u32,
        n_bits: p.n_bits,
        n_bits_ext: p.n_bits_ext,
        cm_pols_map: lay.cm_pols_map.clone(),
        const_pols_map: lay.const_pols_map.clone(),
        custom_pols_map: lay.custom_pols_map.clone(),
        map_sections_n: lay.map_sections_n.clone(),
        challenges_map: lay.challenges_map.clone(),
        publics_map: lay.publics_map.clone(),
        proof_values_map: lay.proof_values_map.clone(),
        air_values_map: lay.air_values_map.clone(),
        airgroup_values_map: lay.airgroup_values_map.clone(),
        ev_map: ops.ev_map.clone(),
        opening_points: ops.opening_points.clone(),
        boundaries,
        q_deg: chk.q_deg,
        // today's qDim is the extension degree even when the Q pieces are base
        q_dim: cs.field.ext_dim() as u32,
        n_constraints: (cs.constraints.len() + im.exprs.len()) as u32,
        fri_batch_size: fri_batch_size(ops),
        c_exp_id: p.c_exp_id,
        custom_commits: cs.custom_commits.clone(),
        airgroup_values: cs.airgroup_agg_types.iter().zip(&cs.airgroup_values).map(|(a, v)| (*a, v.stage)).collect(),
    };
    Ok(StarkInfoCore { verifier, prover: ProverLayout { im_exp_ids: im.exprs.clone() } })
}

/// The starkinfo file in its golden key order; the caller supplies the stark struct `S` and the security `X`.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StarkInfoOutput<S, X> {
    /// Always `KEY_VERSION`; the provers refuse any other.
    #[serde(serialize_with = "key_version")]
    pub key_version: (),
    pub name: String,
    pub cm_pols_map: Vec<PolEntry>,
    pub const_pols_map: Vec<PolEntry>,
    pub challenges_map: Vec<ChallengeEntry>,
    pub publics_map: Vec<NamedValue>,
    pub proof_values_map: Vec<NamedValue>,
    pub airgroup_values_map: Vec<NamedValue>,
    pub air_values_map: Vec<NamedValue>,
    #[serde(serialize_with = "as_map")]
    pub map_sections_n: Vec<(String, u32)>,
    pub air_id: usize,
    pub airgroup_id: usize,
    /// Basic airs only: what their transcript and stage-1 contribution absorb.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub circuit_type: Option<u64>,
    pub n_constants: u32,
    pub n_publics: u32,
    pub air_group_values: Vec<AirGroupValueEntry>,
    pub n_stages: u32,
    pub custom_commits: Vec<CustomCommit>,
    pub custom_commits_map: Vec<Vec<PolEntry>>,
    pub stark_struct: S,
    pub boundaries: Vec<BoundaryEntry>,
    pub opening_points: Vec<i64>,
    pub c_exp_id: u32,
    pub q_dim: u32,
    pub q_deg: u32,
    pub n_constraints: u32,
    pub n_commitments_stage1: u32,
    pub ev_map: Vec<EvMapEntry>,
    pub fri_batch_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security: Option<X>,
    /// The proof travels as a Merkle multiproof; absent on flat keys.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub multiproof: bool,
}

/// A cmPolsMap/constPolsMap/customCommitsMap entry; a Q piece writes stagePos before stageId (golden order).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PolEntry {
    pub stage: u32,
    pub name: String,
    pub dim: u32,
    pub pols_map_id: usize,
    #[serde(rename = "stagePos", skip_serializing_if = "Option::is_none")]
    pub q_stage_pos: Option<u32>,
    pub stage_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lengths: Lengths,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub im_pol: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_pos: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AirGroupValueEntry {
    pub agg_type: i32,
    pub stage: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct BoundaryEntry {
    pub name: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_min: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_max: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvMapEntry {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub id: u32,
    pub prime: i64,
    pub opening_pos: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_id: Option<u32>,
}

/// The key format: bumped when the same pilout gives a different key (2: canonical expression and im order).
pub const KEY_VERSION: u32 = 2;

fn key_version<S: Serializer>(_: &(), s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u32(KEY_VERSION)
}

/// `mapSectionsN` keeps its insertion order.
fn as_map<S: Serializer>(v: &[(String, u32)], s: S) -> Result<S::Ok, S::Error> {
    s.collect_map(v.iter().map(|(k, n)| (k, n)))
}

/// The starkinfo of `core` with the caller's `stark_struct` and `security`.
pub fn starkinfo_output<S, X>(
    core: &StarkInfoCore,
    air_name: &str,
    airgroup_id: usize,
    air_id: usize,
    stark_struct: S,
    security: Option<X>,
) -> Result<StarkInfoOutput<S, X>, ProjectError> {
    let v = &core.verifier;
    let boundary = |b: &Boundary| {
        let (offset_min, offset_max) = match b {
            Boundary::EveryFrame { min, max } => (Some(*min), Some(*max)),
            _ => (None, None),
        };
        BoundaryEntry { name: b.name(), offset_min, offset_max }
    };
    let ev = |e: &EvEntry| EvMapEntry {
        kind: match e.pol.col {
            ColType::Witness => "cm",
            ColType::Fixed => "const",
            ColType::Custom => "custom",
        },
        id: e.pol.id,
        prime: e.pol.offset,
        opening_pos: e.opening_pos,
        commit_id: e.pol.commit_id,
    };
    let entry = |i: usize, p: &PolMap| PolEntry {
        stage: p.stage,
        name: p.name.clone(),
        dim: p.dim,
        pols_map_id: i,
        stage_id: p.stage_id,
        ..Default::default()
    };
    let const_pols =
        v.const_pols_map.iter().enumerate().map(|(i, p)| PolEntry { lengths: p.lengths.clone(), ..entry(i, p) });
    // a custom entry is named by the column's own name (after the last `.`)
    let custom_pol = |(i, p): (usize, &PolMap)| PolEntry {
        name: p.name.rsplit_once('.').map_or(p.name.as_str(), |(_, n)| n).into(),
        stage_pos: Some(p.stage_pos),
        ..entry(i, p)
    };
    Ok(StarkInfoOutput {
        key_version: (),
        name: air_name.into(),
        cm_pols_map: cm_pols(core, entry)?,
        const_pols_map: const_pols.collect(),
        challenges_map: v.challenges_map.clone(),
        publics_map: v.publics_map.clone(),
        proof_values_map: v.proof_values_map.clone(),
        airgroup_values_map: v.airgroup_values_map.clone(),
        air_values_map: v.air_values_map.clone(),
        map_sections_n: v.map_sections_n.clone(),
        air_id,
        airgroup_id,
        circuit_type: None,
        n_constants: v.n_constants,
        n_publics: v.n_publics,
        air_group_values: v
            .airgroup_values
            .iter()
            .map(|&(agg_type, stage)| AirGroupValueEntry { agg_type, stage })
            .collect(),
        n_stages: v.n_stages,
        custom_commits: v.custom_commits.clone(),
        custom_commits_map: v.custom_pols_map.iter().map(|m| m.iter().enumerate().map(custom_pol).collect()).collect(),
        stark_struct,
        boundaries: v.boundaries.iter().map(boundary).collect(),
        opening_points: v.opening_points.clone(),
        c_exp_id: v.c_exp_id,
        q_dim: v.q_dim,
        q_deg: v.q_deg,
        n_constraints: v.n_constraints,
        n_commitments_stage1: 0,
        ev_map: v.ev_map.iter().map(ev).collect(),
        fri_batch_size: v.fri_batch_size,
        security,
        multiproof: false,
    })
}

/// cmPolsMap: a Q piece carries its stagePos first; the others lengths, imPol/expId, then stagePos.
fn cm_pols(core: &StarkInfoCore, entry: impl Fn(usize, &PolMap) -> PolEntry) -> Result<Vec<PolEntry>, ProjectError> {
    let v = &core.verifier;
    let n_im = v.cm_pols_map.iter().filter(|p| p.im_pol).count();
    if n_im != core.prover.im_exp_ids.len() {
        return Err(ProjectError::Mismatch { what: "im columns and im expressions".into() });
    }
    let mut im_exp_ids = core.prover.im_exp_ids.iter().copied();
    let q_stage = v.n_stages + 1;
    let pol = |(i, p): (usize, &PolMap)| {
        if p.stage == q_stage {
            return PolEntry { q_stage_pos: Some(p.stage_pos), ..entry(i, p) };
        }
        let exp_id = if p.im_pol { im_exp_ids.next() } else { None };
        let im_pol = exp_id.map(|_| true);
        PolEntry { lengths: p.lengths.clone(), im_pol, exp_id, stage_pos: Some(p.stage_pos), ..entry(i, p) }
    };
    Ok(v.cm_pols_map.iter().enumerate().map(pol).collect())
}
