//! One air through the core: system, im search and check, layout, arena, openings; the STARK and pilfflonk share it.

use pil2_pilout::pilout as pb;

use crate::cfg::{DegreePolicy, Opening, PilInfoCfg};
use crate::core::from_pilout::{from_pilout, PiloutError};
use crate::core::im::{check_im, im_order, Checked, ImChoice};
use crate::core::ir::ConstraintSystem;
use crate::core::layout::{layout, quotient_dim, Layout, LayoutParams};
use crate::core::openings::{openings, Openings};
use crate::emit::prover::{to_arena, ProverExprs};
use crate::error::{PilInfoError, Result};
use crate::search::im::search_im;

pub struct CorePass {
    pub cs: ConstraintSystem,
    pub im: ImChoice,
    pub chk: Checked,
    pub lay: Layout,
    pub exprs: ProverExprs,
    pub ops: Openings,
}

/// The system of air (`airgroup_id`, `air_id`) and its im choice, checked against `cfg`'s degree bound.
pub fn search(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    cfg: &PilInfoCfg,
) -> Result<(ConstraintSystem, ImChoice, Checked)> {
    let cs = from_pilout(pilout, airgroup_id, air_id, &cfg.field, cfg.opening).map_err(|e| match e {
        PiloutError::UnknownAir { .. } => PilInfoError::NoSuchAir { airgroup_id, air_id },
        e => PilInfoError::InvalidPilout(e.to_string()),
    })?;
    let q_dim = quotient_dim(&cs, &ImChoice { exprs: Vec::new() }).map_err(|e| failed("quotient dim", &e))?;
    let set = search_im(&cs, &cfg.degree_policy, q_dim as usize)?;
    let im = im_order(&cs, &set).map_err(|e| failed("im order", &e))?;
    let stark_min_q = matches!(cfg.degree_policy, DegreePolicy::FromBlowup { .. });
    let max_deg = cfg.degree_policy.max_constraint_degree() as u32;
    let chk = check_im(&cs, &im, max_deg, stark_min_q).map_err(|e| failed("im check", &e))?;
    Ok((cs, im, chk))
}

/// `search`, then the layout, the arena and the openings.
pub fn run(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    cfg: &PilInfoCfg,
    im_pols_stages: bool,
) -> Result<CorePass> {
    let (cs, im, chk) = search(pilout, airgroup_id, air_id, cfg)?;
    let params = LayoutParams {
        im_pols_stages,
        deep_challenges: cfg.opening == Opening::Deep,
        q_deg: chk.q_deg,
        air_name: cs.name.clone(),
    };
    let lay = layout(&cs, &im, &params).map_err(|e| failed("layout", &e))?;
    let exprs = to_arena(&cs, &im, &lay)?;
    let ops = openings(&cs, &im, &lay).map_err(|e| failed("openings", &e))?;
    Ok(CorePass { cs, im, chk, lay, exprs, ops })
}

fn failed(step: &str, e: &dyn std::fmt::Debug) -> PilInfoError {
    PilInfoError::InvalidPilout(format!("{step}: {e:?}"))
}
