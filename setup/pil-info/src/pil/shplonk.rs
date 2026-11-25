//! pilfflonk's setup of one air on the core: `pipeline::run` under SHPLONK, then `emit::prover`.

use pil2_pilout::pilout as pb;

use crate::cfg::{Opening, PilInfoCfg, PrepareOptions};
use crate::core::ir::Boundary;
use crate::core::openings::EvEntry;
use crate::emit::prover::{emit_prover, EmitParams};
use crate::error::{PilInfoError, Result};
use crate::pil::gen_code::PilCodeResult;
use crate::types::pilout_info::ShplonkAir;

/// What pilfflonk's key is made of.
pub struct ShplonkResult {
    pub setup: ShplonkAir,
    pub pil_code: PilCodeResult,
    /// Constraint polynomial expression ID.
    pub c_exp_id: usize,
    /// Polynomial Q degree.
    pub q_deg: i64,
    /// In `boundaryId` order.
    pub boundaries: Vec<Boundary>,
    /// The core's evMap, which the qVerifier's `eval` operands index.
    pub ev_map: Vec<EvEntry>,
}

/// The setup of air (`airgroup_id`, `air_id`) of `pilout` for `cfg`, which must open with SHPLONK.
pub fn shplonk_info(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    cfg: &PilInfoCfg,
    options: &PrepareOptions,
) -> Result<ShplonkResult> {
    if cfg.opening != Opening::Shplonk {
        return Err(PilInfoError::DeepPasses);
    }
    let crate::pipeline::CorePass { cs, im, chk, lay, exprs, ops, air_name } =
        crate::pipeline::run(pilout, airgroup_id, air_id, cfg, options.im_pols_stages)?;
    let c_exp_id = exprs.roles.c_exp;
    let e = emit_prover(&cs, &im, &lay, &ops, exprs, &EmitParams { air_id, airgroup_id })?;

    // no root reads a column: qDeg −1, which pilfflonk refuses
    let top = chk.root_degrees.iter().chain(&chk.im_degrees).max().copied().unwrap_or(0);
    let q_deg = if top == 0 { -1 } else { i64::from(chk.q_deg) };
    let m = e.maps;
    let setup = ShplonkAir {
        name: air_name,
        pil_power: cs.n_rows_bits,
        n_stages: cs.n_stages as usize,
        cm_pols_map: m.cm,
        const_pols_map: m.fixed,
        challenges_map: m.challenges,
        airgroup_values_map: m.airgroup_values,
        air_values_map: m.air_values,
        constraints: e.constraints,
        opening_points: ops.opening_points.clone(),
    };
    Ok(ShplonkResult { setup, pil_code: e.code, c_exp_id, q_deg, boundaries: cs.boundaries(), ev_map: ops.ev_map })
}
