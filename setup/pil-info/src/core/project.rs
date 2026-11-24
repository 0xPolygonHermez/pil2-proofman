//! Pure projection of the layout, openings and checked im choice into the starkinfo fields.

use crate::core::im::{Checked, ImChoice};
use crate::core::ir::{Boundary, ConstraintSystem, CustomCommit, NamedValue};
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
