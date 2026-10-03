//! Build `pilout.globalConstraints.json`: the code of the global constraints and the global hints.

use pil2_pilout::pilout as pb;
use serde_json::json;

use crate::cfg::FieldCfg;
use crate::error::Result;
use crate::output::expressions_info::{code_entries_to_json, hint_value_to_json};

/// Build the globalConstraints JSON from pilout data, computing over `field`.
///
/// Fails on a global constraint or expression that refers to nothing, on a hint field without a
/// value and on what the passes cannot process.
pub fn build_global_constraints_json(pilout: &pb::PilOut, field: &FieldCfg) -> Result<serde_json::Value> {
    use crate::pil::codegen::{build_code, pil_code_gen, CodeGenCtx};
    use crate::expr::helpers::add_info_expressions;
    use crate::types::pilout_info::{
        check_constraint_expressions, format_global_constraints, format_global_expressions, format_global_hints,
        format_global_symbols, SymbolInfo,
    };
    use crate::expr::print::PrintCtx;

    // If no global constraints exist, return empty
    if pilout.constraints.is_empty() && pilout.hints.iter().all(|h| h.air_group_id.is_some() || h.air_id.is_some()) {
        return Ok(json!({"constraints": [], "hints": []}));
    }

    let mut expressions =
        format_global_expressions(&pilout.expressions, &pilout.num_challenges, &pilout.air_groups, field)?;

    let constraints = format_global_constraints(&pilout.constraints);
    check_constraint_expressions(&constraints, pilout.expressions.len(), "the global constraints")?;
    let symbols = format_global_symbols(&pilout.symbols, &pilout.num_challenges, field);

    for constraint in &constraints {
        add_info_expressions(&mut expressions, constraint.e, field)?;
    }

    let publics_map: Vec<SymbolInfo> = symbols.iter().filter(|s| s.sym_type == "public").cloned().collect();
    let challenges_map: Vec<SymbolInfo> = symbols.iter().filter(|s| s.sym_type == "challenge").cloned().collect();
    let airgroup_values_map: Vec<SymbolInfo> =
        symbols.iter().filter(|s| s.sym_type == "airgroupvalue").cloned().collect();
    let proof_values_map: Vec<SymbolInfo> = symbols.iter().filter(|s| s.sym_type == "proofvalue").cloned().collect();

    // Build sorted maps indexed by ID for PrintCtx
    let max_public_id = publics_map.iter().filter_map(|s| s.id).max().unwrap_or(0);
    let mut publics_by_id = vec![
        SymbolInfo {
            name: String::new(),
            sym_type: "public".to_string(),
            stage: Some(1),
            dim: 1,
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        max_public_id + 1
    ];
    for s in &publics_map {
        if let Some(id) = s.id {
            if id < publics_by_id.len() {
                publics_by_id[id] = s.clone();
            }
        }
    }

    let max_challenge_id = challenges_map.iter().filter_map(|s| s.id).max().unwrap_or(0);
    let mut challenges_by_id = vec![
        SymbolInfo {
            name: String::new(),
            sym_type: "challenge".to_string(),
            stage: Some(1),
            dim: field.ext_dim(),
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        max_challenge_id + 1
    ];
    for s in &challenges_map {
        if let Some(id) = s.id {
            if id < challenges_by_id.len() {
                challenges_by_id[id] = s.clone();
            }
        }
    }

    let max_agv_id = airgroup_values_map.iter().filter_map(|s| s.id).max().unwrap_or(0);
    let mut agv_by_id = vec![
        SymbolInfo {
            name: String::new(),
            sym_type: "airgroupvalue".to_string(),
            stage: None,
            dim: field.ext_dim(),
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        max_agv_id + 1
    ];
    for s in &airgroup_values_map {
        if let Some(id) = s.id {
            if id < agv_by_id.len() {
                agv_by_id[id] = s.clone();
            }
        }
    }

    let max_pv_id = proof_values_map.iter().filter_map(|s| s.id).max().unwrap_or(0);
    let mut pv_by_id = vec![
        SymbolInfo {
            name: String::new(),
            sym_type: "proofvalue".to_string(),
            stage: Some(1),
            dim: 1,
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        max_pv_id + 1
    ];
    for s in &proof_values_map {
        if let Some(id) = s.id {
            if id < pv_by_id.len() {
                pv_by_id[id] = s.clone();
            }
        }
    }

    let empty_sym_vec: Vec<SymbolInfo> = Vec::new();
    let empty_custom_commits: Vec<Vec<SymbolInfo>> = Vec::new();
    let print_ctx = PrintCtx {
        cm_pols_map: &empty_sym_vec,
        const_pols_map: &empty_sym_vec,
        custom_commits_map: &empty_custom_commits,
        publics_map: &publics_by_id,
        challenges_map: &challenges_by_id,
        air_values_map: &empty_sym_vec,
        airgroup_values_map: &agv_by_id,
        proof_values_map: &pv_by_id,
    };

    let n_stages = if !pilout.num_challenges.is_empty() { pilout.num_challenges.len() } else { 1 };

    let mut ctx = CodeGenCtx::new(0, 0, n_stages, "n", false, Vec::new(), field);

    let mut constraints_json = Vec::new();

    for constraint in &constraints {
        pil_code_gen(&mut ctx, &symbols, &expressions, constraint.e, 0)?;
        let block = build_code(&mut ctx)?;

        ctx.tmp_used = block.tmp_used;

        let line = constraint.line.clone().unwrap_or_default();

        let mut obj = serde_json::Map::new();
        obj.insert("tmpUsed".to_string(), json!(block.tmp_used));
        obj.insert("code".to_string(), code_entries_to_json(&block.code));
        obj.insert("boundary".to_string(), json!(constraint.boundary));
        obj.insert("line".to_string(), json!(line));
        constraints_json.push(serde_json::Value::Object(obj));
    }

    let hints = format_global_hints(pilout, &mut expressions, field)?;

    let processed_hints = process_global_hints(&mut expressions, &hints, Some(&print_ctx));

    let hints_json: Vec<serde_json::Value> = processed_hints
        .iter()
        .map(|h| {
            json!({
                "name": h.name,
                "fields": h.fields.iter().map(|f| {
                    json!({
                        "name": f.name,
                        "values": f.values.iter().map(|v| {
                            hint_value_to_json(v)
                        }).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<serde_json::Value>>(),
            })
        })
        .collect();

    Ok(json!({
        "constraints": constraints_json,
        "hints": hints_json,
    }))
}

/// Process global hints into flat hint field values.
fn process_global_hints(
    expressions: &mut Vec<crate::expr::expression::Expression>,
    hints: &[crate::types::pilout_info::HintInfo],
    print_ctx: Option<&crate::expr::print::PrintCtx>,
) -> Vec<crate::pil::gen_code::ProcessedHint> {
    use crate::pil::gen_code::{ProcessedHint, ProcessedHintFieldEntry};

    let mut result = Vec::new();

    for hint in hints {
        let mut processed_fields = Vec::new();

        for field in &hint.fields {
            let flat_values = process_global_hint_values(&field.values, expressions, &[], print_ctx);

            let mut entry = ProcessedHintFieldEntry { name: field.name.clone(), values: flat_values };

            if field.lengths.is_none() {
                if let Some(first) = entry.values.first_mut() {
                    first.pos = Vec::new();
                }
            }

            processed_fields.push(entry);
        }

        result.push(ProcessedHint { name: hint.name.clone(), fields: processed_fields });
    }

    result
}

/// Recursively flatten global hint field values.
fn process_global_hint_values(
    values: &[crate::types::pilout_info::HintFieldValue],
    expressions: &mut Vec<crate::expr::expression::Expression>,
    pos: &[usize],
    print_ctx: Option<&crate::expr::print::PrintCtx>,
) -> Vec<crate::pil::gen_code::ProcessedHintField> {
    use crate::types::pilout_info::HintFieldValue;

    let mut result = Vec::new();

    for (j, field) in values.iter().enumerate() {
        let mut current_pos: Vec<usize> = pos.to_vec();
        current_pos.push(j);

        match field {
            HintFieldValue::Array(arr) => {
                let inner = process_global_hint_values(arr, expressions, &current_pos, print_ctx);
                result.extend(inner);
            }
            HintFieldValue::Single(expr) => {
                let processed = process_global_single_hint_field(expr, expressions, &current_pos, print_ctx);
                result.push(processed);
            }
        }
    }

    result
}

/// Process a single global hint field value.
fn process_global_single_hint_field(
    expr: &crate::expr::expression::Expression,
    #[allow(clippy::ptr_arg)] expressions: &mut Vec<crate::expr::expression::Expression>,
    pos: &[usize],
    print_ctx: Option<&crate::expr::print::PrintCtx>,
) -> crate::pil::gen_code::ProcessedHintField {
    use crate::pil::gen_code::ProcessedHintField;

    match expr.op.as_str() {
        "exp" => {
            let ref_id = expr.id.unwrap_or(0);
            let dim = expressions.get(ref_id).map_or(expr.dim.max(1), |e| e.dim);

            if let Some(ctx) = print_ctx {
                if ref_id < expressions.len() {
                    crate::expr::print::print_expression(ctx, expressions, ref_id, false);
                }
            }

            ProcessedHintField {
                op: "tmp".to_string(),
                id: Some(ref_id),
                dim: Some(dim),
                pos: pos.to_vec(),
                stage: None,
                stage_id: None,
                value: None,
                row_offset: None,
                row_offset_index: None,
                commit_id: None,
                airgroup_id: None,
            }
        }
        "challenge" | "public" | "airgroupvalue" | "airvalue" | "number" | "string" | "proofvalue" => {
            ProcessedHintField {
                op: expr.op.clone(),
                id: expr.id,
                dim: Some(expr.dim),
                pos: pos.to_vec(),
                stage: Some(expr.stage),
                stage_id: expr.stage_id,
                value: expr.value.clone(),
                row_offset: None,
                row_offset_index: None,
                commit_id: None,
                airgroup_id: expr.airgroup_id,
            }
        }
        _ => ProcessedHintField {
            op: expr.op.clone(),
            id: expr.id,
            dim: Some(expr.dim),
            pos: pos.to_vec(),
            stage: Some(expr.stage),
            stage_id: expr.stage_id,
            value: expr.value.clone(),
            row_offset: None,
            row_offset_index: None,
            commit_id: None,
            airgroup_id: expr.airgroup_id,
        },
    }
}
