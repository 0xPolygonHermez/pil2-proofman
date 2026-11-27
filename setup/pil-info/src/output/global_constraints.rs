//! Build `pilout.globalConstraints.json`: the global hints. The global constraints are a DAG in globalInfo
//! (`core::dag::build::build_global_dag`).

use pil2_pilout::pilout as pb;
use serde_json::json;

use crate::cfg::FieldCfg;
use crate::core::global::{global_from_pilout, GlobalSystem};
use crate::emit::arena;
use crate::error::{PilInfoError, Result};
use crate::pil::gen_code::add_hints_info;

/// `global_constraints_json` of the pilout's global system, computing over `field`.
pub fn build_global_constraints_json(pilout: &pb::PilOut, field: &FieldCfg) -> Result<serde_json::Value> {
    let gs = global_from_pilout(pilout, field).map_err(|e| PilInfoError::InvalidPilout(e.to_string()))?;
    global_constraints_json(&gs)
}

/// The globalConstraints JSON of `gs` (its hints); fails on what the passes cannot process.
pub fn global_constraints_json(gs: &GlobalSystem) -> Result<serde_json::Value> {
    if gs.hints.is_empty() {
        return Ok(json!({"hints": []}));
    }
    let leaf = arena::global_leaf(gs);
    let mut expressions = arena::from_ir(&gs.exprs, &leaf)?;
    let hints = arena::hints(&gs.hints, &gs.exprs, &leaf, &mut expressions)?;
    let hints = add_hints_info(&[], &mut expressions, &hints, None)?;
    Ok(json!({"hints": hints}))
}
