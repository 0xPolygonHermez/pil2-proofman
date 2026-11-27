//! Orchestrates DAG code generation for all stages.
//!
//! Produces code blocks for:
//! - Expression computations (witness polynomials, intermediate polynomials)
//! - Constraint polynomial (Q stage)
//! - Hint computations

use crate::core::dag::format::ConstraintDag;
use crate::cfg::FieldCfg;
use crate::error::{PilInfoError, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

use crate::pil::codegen::{build_code, pil_code_gen, CalcEntry, CodeGenCtx};
use crate::expr::expression::Expression;
use crate::types::pilout_info::{ConstraintInfo, HintFieldValue, HintInfo, SymbolInfo};
use crate::expr::print::PrintCtx;

/// Build a HashMap from (exp_id, air_id, airgroup_id) -> symbol index
/// for witness symbols. Used by fix_commit_pol for O(1) lookups.
fn build_witness_index(
    symbols: &[SymbolInfo],
    air_id: usize,
    airgroup_id: usize,
) -> Arc<HashMap<(usize, usize, usize), usize>> {
    let mut index = HashMap::new();
    for (i, s) in symbols.iter().enumerate() {
        if s.sym_type == "witness" && s.air_id == Some(air_id) && s.airgroup_id == Some(airgroup_id) {
            if let Some(exp_id) = s.exp_id {
                index.insert((exp_id, air_id, airgroup_id), i);
            }
        }
    }
    Arc::new(index)
}
use crate::types::output::CodeRef;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Destination metadata for an expression that computes an intermediate polynomial.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExprDest {
    pub op: String,
    pub stage: usize,
    pub stage_id: usize,
    pub id: usize,
}

/// Extended code block that carries per-expression metadata.
/// Field declaration order matches JSON output order (tmpUsed, code, expId, stage, dest, line).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpressionCodeEntry {
    pub tmp_used: usize,
    pub code: Vec<crate::types::output::CodeEntry>,
    pub exp_id: usize,
    pub stage: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dest: Option<ExprDest>,
    pub line: String,
}

/// A constraint code block with boundary and debug metadata.
/// Field declaration order matches JSON output order.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintCodeEntry {
    pub tmp_used: usize,
    pub code: Vec<crate::types::output::CodeEntry>,
    pub boundary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<String>,
    pub im_pol: usize,
    pub stage: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_min: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_max: Option<u32>,
}

/// A processed hint field value (leaf node).
/// Uses custom Serialize because field order depends on the op type.
#[derive(Debug, Clone)]
pub struct ProcessedHintField {
    pub op: String,
    pub id: Option<usize>,
    pub dim: Option<usize>,
    pub pos: Vec<usize>,
    pub stage: Option<usize>,
    pub stage_id: Option<usize>,
    pub value: Option<String>,
    pub row_offset: Option<i64>,
    pub row_offset_index: Option<isize>,
    pub commit_id: Option<usize>,
    pub airgroup_id: Option<usize>,
}

impl Serialize for ProcessedHintField {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(None)?;
        map.serialize_entry("op", &self.op)?;
        match self.op.as_str() {
            "string" => {
                if let Some(ref val) = self.value {
                    map.serialize_entry("string", val)?;
                }
                map.serialize_entry("pos", &self.pos)?;
            }
            "number" => {
                if let Some(ref val) = self.value {
                    map.serialize_entry("value", val)?;
                }
                map.serialize_entry("pos", &self.pos)?;
            }
            "tmp" => {
                if let Some(id) = self.id {
                    map.serialize_entry("id", &id)?;
                }
                if let Some(dim) = self.dim {
                    map.serialize_entry("dim", &dim)?;
                }
                map.serialize_entry("pos", &self.pos)?;
            }
            "cm" | "custom" | "const" => {
                if let Some(id) = self.id {
                    map.serialize_entry("id", &id)?;
                }
                if let Some(sid) = self.stage_id {
                    map.serialize_entry("stageId", &sid)?;
                }
                if let Some(ro) = self.row_offset {
                    map.serialize_entry("rowOffset", &ro)?;
                }
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                if let Some(dim) = self.dim {
                    map.serialize_entry("dim", &dim)?;
                }
                if let Some(cid) = self.commit_id {
                    map.serialize_entry("commitId", &cid)?;
                }
                if let Some(roi) = self.row_offset_index {
                    map.serialize_entry("rowOffsetIndex", &roi)?;
                }
                map.serialize_entry("pos", &self.pos)?;
            }
            "challenge" => {
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                if let Some(sid) = self.stage_id {
                    map.serialize_entry("stageId", &sid)?;
                }
                if let Some(id) = self.id {
                    map.serialize_entry("id", &id)?;
                }
                if let Some(dim) = self.dim {
                    map.serialize_entry("dim", &dim)?;
                }
                map.serialize_entry("pos", &self.pos)?;
            }
            "airgroupvalue" => {
                if let Some(id) = self.id {
                    map.serialize_entry("id", &id)?;
                }
                if let Some(agid) = self.airgroup_id {
                    map.serialize_entry("airgroupId", &agid)?;
                }
                if let Some(dim) = self.dim {
                    map.serialize_entry("dim", &dim)?;
                }
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                map.serialize_entry("pos", &self.pos)?;
            }
            "public" => {
                if let Some(id) = self.id {
                    map.serialize_entry("id", &id)?;
                }
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                map.serialize_entry("pos", &self.pos)?;
            }
            _ => {
                // airvalue, proofvalue, and any others
                if let Some(id) = self.id {
                    map.serialize_entry("id", &id)?;
                }
                if let Some(stage) = self.stage {
                    map.serialize_entry("stage", &stage)?;
                }
                if let Some(dim) = self.dim {
                    map.serialize_entry("dim", &dim)?;
                }
                map.serialize_entry("pos", &self.pos)?;
            }
        }
        map.end()
    }
}

/// A processed hint field with name and flat values.
#[derive(Debug, Clone, Serialize)]
pub struct ProcessedHintFieldEntry {
    pub name: String,
    pub values: Vec<ProcessedHintField>,
}

/// A processed hint.
#[derive(Debug, Clone, Serialize)]
pub struct ProcessedHint {
    pub name: String,
    pub fields: Vec<ProcessedHintFieldEntry>,
}

/// The verifierinfo: the STARK's constraint DAG, or pilfflonk's qVerifier code (exactly one of them).
#[derive(Debug, Clone, Default)]
pub struct VerifierInfo {
    /// pilfflonk only (`Opening::Shplonk`): the lowered DAG `emit_prover` sets.
    pub q_verifier: Option<ExpressionCodeEntry>,
    /// STARK only: the DAG `stark_setup` attaches.
    pub constraint_dag: Option<ConstraintDag>,
}

impl Serialize for VerifierInfo {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let n = self.q_verifier.is_some() as usize + self.constraint_dag.is_some() as usize;
        let mut map = s.serialize_map(Some(n))?;
        if let Some(q) = &self.q_verifier {
            map.serialize_entry("qVerifier", &QVerifierView(q))?;
        }
        if let Some(dag) = &self.constraint_dag {
            map.serialize_entry("constraintDag", dag)?;
        }
        map.end()
    }
}

struct QVerifierView<'a>(&'a ExpressionCodeEntry);
impl Serialize for QVerifierView<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(3))?;
        map.serialize_entry("tmpUsed", &self.0.tmp_used)?;
        map.serialize_entry("code", &self.0.code)?;
        map.serialize_entry("line", "")?;
        map.end()
    }
}

/// Expression code blocks plus metadata.
/// Field declaration order matches JSON output order (hintsInfo, expressionsCode, constraints).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpressionsInfo {
    pub hints_info: Vec<ProcessedHint>,
    pub expressions_code: Vec<ExpressionCodeEntry>,
    pub constraints: Vec<ConstraintCodeEntry>,
}

/// Top-level result of `generate_pil_code`.
#[derive(Debug, Clone)]
pub struct PilCodeResult {
    pub expressions_info: ExpressionsInfo,
    pub verifier_info: VerifierInfo,
}

// ---------------------------------------------------------------------------
// Res: a view of the setup parameters needed for code generation
// ---------------------------------------------------------------------------

/// Parameters extracted from the setup result needed for code generation.
/// This avoids passing the entire setup/prepare result.
pub struct CodeGenParams {
    pub air_id: usize,
    pub airgroup_id: usize,
    pub n_stages: usize,
    pub c_exp_id: usize,
    pub q_dim: usize,
    pub opening_points: Vec<i64>,
    /// The field the code computes over.
    pub field: FieldCfg,
}

// ---------------------------------------------------------------------------
// generate_pil_code
// ---------------------------------------------------------------------------

/// Orchestrate code generation for all stages.
///
/// Mirrors JS `generatePilCode(res, symbols, constraints, expressions, hints)`. The DEEP challenges and
/// the batching size are not the generator's: the STARK takes them from the core's layout and openings.
/// The verifierinfo is the caller's: `stark_setup` attaches the DAG, `emit_prover` pilfflonk's qVerifier.
pub fn generate_pil_code(
    params: &CodeGenParams,
    symbols: &[SymbolInfo],
    constraints: &[ConstraintInfo],
    expressions: &mut [Expression],
    hints: &[HintInfo],
    print_ctx: Option<&PrintCtx>,
) -> Result<PilCodeResult> {
    // Pre-compute witness symbol index for O(1) lookups in fix_commit_pol
    let witness_index = build_witness_index(symbols, params.air_id, params.airgroup_id);

    let hints_info = add_hints_info(&params.opening_points, expressions, hints, print_ctx)?;

    let expressions_code = generate_expressions_code(params, symbols, expressions, &witness_index)?;

    let constraints_code = generate_constraints_debug_code(params, symbols, constraints, expressions, &witness_index)?;

    Ok(PilCodeResult {
        expressions_info: ExpressionsInfo { hints_info, expressions_code, constraints: constraints_code },
        verifier_info: VerifierInfo::default(),
    })
}

// ---------------------------------------------------------------------------
// generateExpressionsCode
// ---------------------------------------------------------------------------

/// Generate code blocks for all kept/imPol/cExp expressions.
///
/// Mirrors JS `generateExpressionsCode(res, symbols, expressions)`.
fn generate_expressions_code(
    params: &CodeGenParams,
    symbols: &[SymbolInfo],
    expressions: &[Expression],
    witness_index: &Arc<HashMap<(usize, usize, usize), usize>>,
) -> Result<Vec<ExpressionCodeEntry>> {
    let mut result = Vec::new();

    for j in 0..expressions.len() {
        let exp = &expressions[j];
        let dominated = !exp.keep.unwrap_or(false) && !exp.im_pol && j != params.c_exp_id;
        if dominated {
            continue;
        }

        let dom = if j == params.c_exp_id { "ext" } else { "n" };

        let mut ctx = CodeGenCtx::new(params.air_id, params.airgroup_id, exp.stage, dom);
        ctx.witness_by_exp_id = Arc::clone(witness_index);

        if j == params.c_exp_id {
            // Pre-mark imPol expressions as calculated (cm=true) for all opening points
            for sym in symbols.iter() {
                if !sym.im_pol {
                    continue;
                }
                if let Some(exp_id) = sym.exp_id {
                    let inner = ctx.calculated.entry(exp_id).or_default();
                    for &op in &params.opening_points {
                        inner.insert(op, CalcEntry { cm: true, tmp_id: None });
                    }
                }
            }
        }

        // Determine destination for imPol expressions
        let expr_dest = if exp.im_pol {
            symbols.iter().find(|s| s.exp_id == Some(j)).map(|s| ExprDest {
                op: "cm".to_string(),
                stage: s.stage.unwrap_or(0),
                stage_id: s.stage_id.unwrap_or(0),
                id: s.pol_id.unwrap_or(0),
            })
        } else {
            None
        };

        pil_code_gen(&mut ctx, symbols, expressions, j, 0)?;
        let mut block = build_code(&mut ctx)?;

        if j == params.c_exp_id {
            if let Some(last) = block.code.last_mut() {
                last.dest = CodeRef {
                    ref_type: "q".to_string(),
                    id: 0,
                    dim: params.q_dim,
                    prime: None,
                    value: None,
                    stage: None,
                    stage_id: None,
                    commit_id: None,
                    boundary_id: None,
                    airgroup_id: None,
                    exp_id: None,
                };
            }
        }

        // Stages beyond the Q stage (nStages + 1) are reported as 0, as the JS setup did.
        let entry_stage = if exp.stage > params.n_stages + 1 { 0 } else { exp.stage };

        // Copy the cached line from the expression (set by printExpressions
        // during hint processing or map phase), or empty string.
        // Mirrors JS: `expInfo.line = exp.line || ""`
        let line = exp.line.clone().unwrap_or_default();

        result.push(ExpressionCodeEntry {
            tmp_used: block.tmp_used,
            code: block.code,
            exp_id: j,
            stage: entry_stage,
            dest: expr_dest,
            line,
        });
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// generateConstraintsDebugCode
// ---------------------------------------------------------------------------

/// Generate debug code blocks for each constraint.
///
/// Mirrors JS `generateConstraintsDebugCode(res, symbols, constraints, expressions)`.
fn generate_constraints_debug_code(
    params: &CodeGenParams,
    symbols: &[SymbolInfo],
    constraints: &[ConstraintInfo],
    expressions: &[Expression],
    witness_index: &Arc<HashMap<(usize, usize, usize), usize>>,
) -> Result<Vec<ConstraintCodeEntry>> {
    let mut result = Vec::new();

    for constraint in constraints {
        let mut ctx = CodeGenCtx::new(params.air_id, params.airgroup_id, params.n_stages, "n");
        ctx.witness_by_exp_id = Arc::clone(witness_index);

        // Pre-mark imPol expressions as calculated
        for sym in symbols.iter() {
            if !sym.im_pol {
                continue;
            }
            if let Some(exp_id) = sym.exp_id {
                let inner = ctx.calculated.entry(exp_id).or_default();
                for &op in &params.opening_points {
                    inner.insert(op, CalcEntry { cm: true, tmp_id: None });
                }
            }
        }

        pil_code_gen(&mut ctx, symbols, expressions, constraint.e, 0)?;
        let block = build_code(&mut ctx)?;

        let stage =
            if constraint.stage == Some(0) || constraint.stage.is_none() { 1 } else { constraint.stage.unwrap_or(1) };

        let mut entry = ConstraintCodeEntry {
            tmp_used: block.tmp_used,
            code: block.code,
            boundary: constraint.boundary.clone(),
            line: constraint.line.clone(),
            im_pol: if constraint.im_pol { 1 } else { 0 },
            stage,
            offset_min: None,
            offset_max: None,
        };

        if constraint.boundary == "everyFrame" {
            entry.offset_min = constraint.offset_min;
            entry.offset_max = constraint.offset_max;
        }

        result.push(entry);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// addHintsInfo
// ---------------------------------------------------------------------------

/// Process hints into flat hint field values.
///
/// Mirrors JS `addHintsInfo(res, expressions, hints, global)`; the global hints open at no point.
pub(crate) fn add_hints_info(
    opening_points: &[i64],
    expressions: &mut [Expression],
    hints: &[HintInfo],
    print_ctx: Option<&PrintCtx>,
) -> Result<Vec<ProcessedHint>> {
    let mut result = Vec::new();

    for hint in hints {
        let mut processed_fields = Vec::new();

        for field in &hint.fields {
            let flat_values = process_hint_field_values(&field.values, opening_points, expressions, &[], print_ctx)?;

            let mut entry = ProcessedHintFieldEntry { name: field.name.clone(), values: flat_values };

            // If no lengths, set first value's pos to empty
            if field.lengths.is_none() {
                if let Some(first) = entry.values.first_mut() {
                    first.pos = Vec::new();
                }
            }

            processed_fields.push(entry);
        }

        result.push(ProcessedHint { name: hint.name.clone(), fields: processed_fields });
    }

    Ok(result)
}

/// Recursively flatten hint field values.
fn process_hint_field_values(
    values: &[HintFieldValue],
    opening_points: &[i64],
    expressions: &mut [Expression],
    pos: &[usize],
    print_ctx: Option<&PrintCtx>,
) -> Result<Vec<ProcessedHintField>> {
    let mut result = Vec::new();

    for (j, field) in values.iter().enumerate() {
        let mut current_pos: Vec<usize> = pos.to_vec();
        current_pos.push(j);

        match field {
            HintFieldValue::Array(arr) => {
                let inner = process_hint_field_values(arr, opening_points, expressions, &current_pos, print_ctx)?;
                result.extend(inner);
            }
            HintFieldValue::Single(expr) => {
                let processed = process_single_hint_field(expr, opening_points, expressions, &current_pos, print_ctx)?;
                result.push(processed);
            }
        }
    }

    Ok(result)
}

/// Process a single (leaf) hint field expression.
fn process_single_hint_field(
    expr: &Expression,
    opening_points: &[i64],
    expressions: &mut [Expression],
    pos: &[usize],
    print_ctx: Option<&PrintCtx>,
) -> Result<ProcessedHintField> {
    let processed = match expr.op.as_str() {
        "exp" => {
            let ref_id = expr.id.unwrap_or(0);
            let dim = expressions.get(ref_id).map_or(expr.dim.max(1), |e| e.dim);

            // Set the line on the expression (mirrors JS:
            // expressions[field.id].line = printExpressions(...))
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
        "cm" | "custom" | "const" => {
            let row_offset = expr.row_offset.unwrap_or(0);
            let prime_index = opening_points.iter().position(|&p| p == row_offset).map(|p| p as isize).unwrap_or(-1);
            ProcessedHintField {
                op: expr.op.clone(),
                id: expr.id,
                dim: Some(expr.dim),
                pos: pos.to_vec(),
                stage: Some(expr.stage),
                stage_id: expr.stage_id,
                value: None,
                row_offset: expr.row_offset,
                row_offset_index: Some(prime_index),
                commit_id: expr.commit_id,
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
        _ => return Err(PilInfoError::UnknownHintOp(expr.op.clone())),
    };
    Ok(processed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::expression::Expression;
    use crate::types::pilout_info::HintFieldEntry;

    fn make_number(val: &str) -> Expression {
        Expression { op: "number".to_string(), value: Some(val.to_string()), dim: 1, ..Default::default() }
    }

    fn make_cm(id: usize, stage: usize) -> Expression {
        Expression { op: "cm".to_string(), id: Some(id), dim: 1, stage, row_offset: Some(0), ..Default::default() }
    }

    fn make_add(lhs: usize, rhs: usize) -> Expression {
        use crate::expr::expression::ExprChild;
        Expression {
            op: "add".to_string(),
            values: vec![ExprChild::Id(lhs), ExprChild::Id(rhs)],
            dim: 1,
            ..Default::default()
        }
    }

    fn make_params() -> CodeGenParams {
        CodeGenParams {
            air_id: 0,
            airgroup_id: 0,
            n_stages: 1,
            c_exp_id: 2,
            q_dim: 1,
            opening_points: vec![0],
            field: FieldCfg::goldilocks(),
        }
    }

    #[test]
    fn test_generate_expressions_code_basic() {
        // Build a minimal expression set:
        // [0] = number("1"), [1] = cm(0), [2] = add(0,1) with keep=true
        let expressions = vec![make_number("1"), make_cm(0, 1), {
            let mut e = make_add(0, 1);
            e.keep = Some(true);
            e.stage = 1;
            e
        }];

        let symbols: Vec<SymbolInfo> = Vec::new();
        let params = CodeGenParams {
            air_id: 0,
            airgroup_id: 0,
            n_stages: 1,
            c_exp_id: 999, // not matching any expr
            q_dim: 1,
            opening_points: vec![0],
            field: FieldCfg::goldilocks(),
        };

        let wi = build_witness_index(&symbols, 0, 0);
        let code = generate_expressions_code(&params, &symbols, &expressions, &wi).unwrap();
        // Only expression[2] has keep=true, so we should get 1 entry
        assert_eq!(code.len(), 1);
        assert_eq!(code[0].exp_id, 2);
        assert!(!code[0].code.is_empty());
    }

    #[test]
    fn test_add_hints_info_basic() {
        let mut expressions = vec![make_number("5"), make_cm(0, 1)];
        let params = make_params();

        let hints = vec![HintInfo {
            name: "test_hint".to_string(),
            fields: vec![HintFieldEntry {
                name: "field1".to_string(),
                values: vec![HintFieldValue::Single(Box::new(Expression {
                    op: "number".to_string(),
                    value: Some("42".to_string()),
                    dim: 1,
                    ..Default::default()
                }))],
                lengths: None,
            }],
        }];

        let result = add_hints_info(&params.opening_points, &mut expressions, &hints, None).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "test_hint");
        assert_eq!(result[0].fields.len(), 1);
        assert_eq!(result[0].fields[0].name, "field1");
        // When lengths is None, first value's pos should be empty
        assert!(result[0].fields[0].values[0].pos.is_empty());
    }

    #[test]
    fn test_constraints_debug_code() {
        let expressions = vec![make_number("1"), make_cm(0, 1), make_add(0, 1)];
        let symbols: Vec<SymbolInfo> = Vec::new();
        let constraints = vec![ConstraintInfo {
            boundary: "everyRow".to_string(),
            e: 2,
            line: Some("test".to_string()),
            offset_min: None,
            offset_max: None,
            stage: Some(1),
            im_pol: false,
        }];
        let params = CodeGenParams {
            air_id: 0,
            airgroup_id: 0,
            n_stages: 1,
            c_exp_id: 2,
            q_dim: 1,
            opening_points: vec![0],
            field: FieldCfg::goldilocks(),
        };

        let wi2 = build_witness_index(&symbols, 0, 0);
        let result = generate_constraints_debug_code(&params, &symbols, &constraints, &expressions, &wi2).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].boundary, "everyRow");
        assert_eq!(result[0].stage, 1);
        assert!(!result[0].code.is_empty());
    }

    #[test]
    fn test_process_hint_field_nested() {
        let mut expressions = vec![make_number("1")];
        let params = make_params();

        let values = vec![HintFieldValue::Array(Box::new(vec![
            HintFieldValue::Single(Box::new(Expression {
                op: "number".to_string(),
                value: Some("1".to_string()),
                dim: 1,
                ..Default::default()
            })),
            HintFieldValue::Single(Box::new(Expression {
                op: "number".to_string(),
                value: Some("2".to_string()),
                dim: 1,
                ..Default::default()
            })),
        ]))];

        let result = process_hint_field_values(&values, &params.opening_points, &mut expressions, &[], None).unwrap();
        // Should flatten to 2 entries
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].pos, vec![0, 0]);
        assert_eq!(result[1].pos, vec![0, 1]);
    }

    /// Was `panic!("Invalid hint op")`.
    #[test]
    fn a_hint_field_of_an_unknown_op_is_an_error() {
        let zi = Expression { op: "Zi".to_string(), ..Default::default() };
        let err = process_single_hint_field(&zi, &make_params().opening_points, &mut [], &[0], None).unwrap_err();
        assert!(matches!(&err, PilInfoError::UnknownHintOp(op) if op == "Zi"), "{err}");
    }
}
