//! Build `pilout.globalConstraints.json`: the code of the global constraints and the global hints.

use pil2_pilout::pilout as pb;
use serde_json::json;

use crate::cfg::FieldCfg;
use crate::core::global::{global_from_pilout, GlobalSystem};
use crate::emit::arena;
use crate::error::{PilInfoError, Result};
use crate::expr::helpers::add_info_expressions;
use crate::pil::codegen::{build_code, pil_code_gen, CodeGenCtx};
use crate::pil::gen_code::add_hints_info;

/// `global_constraints_json` of the pilout's global system, computing over `field`.
pub fn build_global_constraints_json(pilout: &pb::PilOut, field: &FieldCfg) -> Result<serde_json::Value> {
    let gs = global_from_pilout(pilout, field).map_err(|e| PilInfoError::InvalidPilout(e.to_string()))?;
    global_constraints_json(&gs)
}

/// The globalConstraints JSON of `gs`; fails on what the passes cannot process.
pub fn global_constraints_json(gs: &GlobalSystem) -> Result<serde_json::Value> {
    if gs.constraints.is_empty() && gs.hints.is_empty() {
        return Ok(json!({"constraints": [], "hints": []}));
    }
    let leaf = arena::global_leaf(gs);
    let mut expressions = arena::from_ir(&gs.exprs, &leaf)?;
    let hints = arena::hints(&gs.hints, &gs.exprs, &leaf, &mut expressions)?;
    for c in &gs.constraints {
        add_info_expressions(&mut expressions, c.expr as usize, &gs.field)?;
    }

    let n_stages = gs.num_challenges.len().max(1);
    let mut ctx = CodeGenCtx::new(0, 0, n_stages, "n");
    let mut constraints = Vec::new();
    for c in &gs.constraints {
        pil_code_gen(&mut ctx, &[], &expressions, c.expr as usize, 0)?;
        let block = build_code(&mut ctx)?;
        ctx.tmp_used = block.tmp_used;
        constraints.push(json!({
            "tmpUsed": block.tmp_used,
            "code": block.code,
            "boundary": "finalProof",
            "line": c.label,
        }));
    }
    let hints = add_hints_info(&[], &mut expressions, &hints, None)?;
    Ok(json!({"constraints": constraints, "hints": hints}))
}
