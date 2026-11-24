//! pilfflonk's setup of one air on the core: `from_pilout(Opening::Shplonk)`, `search_im` + `im_order`,
//! `check_im` for qDeg, `core::{layout,openings}` + `emit::prover`.

use indexmap::IndexMap;
use pil2_pilout::pilout as pb;

use crate::cfg::{Opening, PilInfoCfg, PrepareOptions};
use crate::core::from_pilout::{from_pilout, PiloutError};
use crate::core::im::{check_im, im_order, ImChoice};
use crate::core::ir::Boundary;
use crate::core::layout::{layout, quotient_dim, LayoutParams};
use crate::core::openings::openings;
use crate::emit::prover::{emit_prover, to_arena, EmitParams};
use crate::error::{PilInfoError, Result};
use crate::pil::gen_code::PilCodeResult;
use crate::search::im::search_im;
use crate::types::pilout_info::{CustomCommitInfo, SetupResult};

/// What pilfflonk's key is made of.
pub struct PilInfoResult {
    pub setup: SetupResult,
    pub pil_code: PilCodeResult,
    /// Intermediate polynomial info: (base_field, extended_field) expression strings.
    pub im_pols_info: (Vec<String>, Vec<String>),
    /// Constraint polynomial expression ID.
    pub c_exp_id: usize,
    /// Polynomial Q degree.
    pub q_deg: i64,
    /// In `boundaryId` order.
    pub boundaries: Vec<Boundary>,
}

/// The setup of air (`airgroup_id`, `air_id`) of `pilout` for `cfg`, which must open with SHPLONK.
pub fn shplonk_info(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    cfg: &PilInfoCfg,
    options: &PrepareOptions,
) -> Result<PilInfoResult> {
    if cfg.opening != Opening::Shplonk {
        return Err(PilInfoError::DeepPasses);
    }
    let failed = |step: &str, e: &dyn std::fmt::Debug| PilInfoError::InvalidPilout(format!("{step}: {e:?}"));
    let cs = from_pilout(pilout, airgroup_id, air_id, &cfg.field, cfg.opening).map_err(|e| match e {
        PiloutError::UnknownAir { .. } => PilInfoError::NoSuchAir { airgroup_id, air_id },
        e => PilInfoError::InvalidPilout(e.to_string()),
    })?;
    let air_name = pilout.air_groups[airgroup_id].airs[air_id].name.clone().unwrap_or_default();

    let q_dim = quotient_dim(&cs, &ImChoice { exprs: Vec::new() }).map_err(|e| failed("quotient dim", &e))?;
    let set = search_im(&cs, &cfg.degree_policy, q_dim as usize)?;
    let im = im_order(&cs, &set).map_err(|e| failed("im order", &e))?;
    let max_deg = cfg.degree_policy.max_constraint_degree() as u32;
    let chk = check_im(&cs, &im, max_deg, false).map_err(|e| failed("im check", &e))?;
    let lay_params = LayoutParams {
        im_pols_stages: options.im_pols_stages,
        deep_challenges: false,
        q_deg: chk.q_deg,
        air_name: air_name.clone(),
    };
    let lay = layout(&cs, &im, &lay_params).map_err(|e| failed("layout", &e))?;
    let exprs = to_arena(&cs, &im, &lay)?;
    let c_exp_id = exprs.roles.c_exp;
    let ops = openings(&cs, &im, &lay).map_err(|e| failed("openings", &e))?;
    let e = emit_prover(&cs, &im, &lay, &ops, &chk, exprs, &EmitParams { air_id, airgroup_id })?;

    // no root reads a column: qDeg −1, which pilfflonk refuses
    let top = chk.root_degrees.iter().chain(&chk.im_degrees).max().copied().unwrap_or(0);
    let q_deg = if top == 0 { -1 } else { i64::from(chk.q_deg) };
    let m = e.maps;
    let n_commitments_stage1 = m.cm.iter().filter(|p| p.stage == Some(1) && !p.im_pol).count();
    let symbols = m.symbols();
    let setup = SetupResult {
        name: air_name,
        air_id,
        airgroup_id,
        pil_power: cs.n_rows_bits,
        n_stages: cs.n_stages as usize,
        n_constants: m.fixed.len(),
        n_publics: m.publics.len(),
        n_commitments: m.cm.len(),
        cm_pols_map: m.cm,
        const_pols_map: m.fixed,
        challenges_map: m.challenges,
        publics_map: m.publics,
        proof_values_map: m.proof_values,
        airgroup_values_map: m.airgroup_values,
        air_values_map: m.air_values,
        map_sections_n: lay.map_sections_n.iter().map(|(k, n)| (k.clone(), *n as usize)).collect::<IndexMap<_, _>>(),
        custom_commits: (cs.custom_commits.iter())
            .map(|c| CustomCommitInfo {
                name: c.name.clone(),
                stage_widths: c.stage_widths.clone(),
                public_values: c.public_values.clone(),
            })
            .collect(),
        custom_commits_map: m.custom,
        air_group_values: (cs.airgroup_values.iter().zip(&cs.airgroup_agg_types))
            .map(|(v, a)| pb::AirGroupValue { agg_type: *a, stage: v.stage })
            .collect(),
        expressions: e.arena,
        constraints: e.constraints,
        symbols,
        hints: e.hints,
        n_commitments_stage1,
        im_pols_info: e.im_pols_info.clone(),
        opening_points: ops.opening_points,
    };
    Ok(PilInfoResult {
        setup,
        pil_code: e.code,
        im_pols_info: e.im_pols_info,
        c_exp_id,
        q_deg,
        boundaries: cs.boundaries(),
    })
}
