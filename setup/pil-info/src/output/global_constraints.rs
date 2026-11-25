//! Build `pilout.globalConstraints.json`: the code of the global constraints and the global hints.

use pil2_pilout::pilout as pb;
use serde_json::json;

use crate::cfg::FieldCfg;
use crate::core::global::global_from_pilout;
use crate::emit::arena;
use crate::error::{PilInfoError, Result};
use crate::expr::helpers::add_info_expressions;
use crate::output::expressions_info::{code_entries_to_json, hint_value_to_json};
use crate::pil::codegen::{build_code, pil_code_gen, CodeGenCtx};
use crate::pil::gen_code::add_hints_info;

/// Build the globalConstraints JSON from pilout data, computing over `field`.
///
/// Fails on what `global_from_pilout` refuses and on what the passes cannot process.
pub fn build_global_constraints_json(pilout: &pb::PilOut, field: &FieldCfg) -> Result<serde_json::Value> {
    let gs = global_from_pilout(pilout, field).map_err(|e| PilInfoError::InvalidPilout(e.to_string()))?;
    if gs.constraints.is_empty() && gs.hints.is_empty() {
        return Ok(json!({"constraints": [], "hints": []}));
    }
    let leaf = arena::global_leaf(&gs);
    let mut expressions = arena::from_ir(&gs.exprs, &leaf)?;
    let hints = arena::hints(&gs.hints, &gs.exprs, &leaf, &mut expressions)?;
    let roots = gs.constraints.iter().map(|c| match c.e {
        crate::core::ir::Expr::Ref(id) => id as usize,
        _ => unreachable!("a global constraint is a reference"),
    });
    let roots: Vec<usize> = roots.collect();
    for &e in &roots {
        add_info_expressions(&mut expressions, e, field)?;
    }

    let n_stages = gs.num_challenges.len().max(1);
    let mut ctx = CodeGenCtx::new(0, 0, n_stages, "n");
    let mut constraints_json = Vec::new();
    for (c, &e) in gs.constraints.iter().zip(&roots) {
        pil_code_gen(&mut ctx, &[], &expressions, e, 0)?;
        let block = build_code(&mut ctx)?;
        ctx.tmp_used = block.tmp_used;
        constraints_json.push(json!({
            "tmpUsed": block.tmp_used,
            "code": code_entries_to_json(&block.code),
            "boundary": "finalProof",
            "line": c.label,
        }));
    }

    let hints_json: Vec<serde_json::Value> = add_hints_info(&[], &mut expressions, &hints, None)?
        .iter()
        .map(|h| {
            let fields = h.fields.iter().map(
                |f| json!({"name": f.name, "values": f.values.iter().map(hint_value_to_json).collect::<Vec<_>>()}),
            );
            json!({"name": h.name, "fields": fields.collect::<Vec<_>>()})
        })
        .collect();

    Ok(json!({"constraints": constraints_json, "hints": hints_json}))
}
