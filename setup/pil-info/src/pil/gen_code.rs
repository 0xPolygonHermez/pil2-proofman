//! Orchestrates DAG code generation for all stages.
//!
//! Produces code blocks for:
//! - Expression computations (witness polynomials, intermediate polynomials)
//! - Constraint polynomial (Q stage)
//! - FRI polynomial (the FRI opening only)
//! - Verifier evaluations
//! - Hint computations

use crate::cfg::{FieldCfg, Opening};
use crate::error::{PilInfoError, Result};
use crate::pil::cse;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

use crate::pil::codegen::{build_code, pil_code_gen, rebuild_ev_map_index, CalcEntry, CodeGenCtx, EvMapRef};
use crate::expr::expression::Expression;
use crate::pil::fri_poly::{self, ChallengeMapEntry};
use crate::expr::helpers::{add_info_expressions_symbols, EvMapItem};
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

/// Verifier code blocks.
/// Uses custom Serialize because qVerifier and queryVerifier emit different subsets of fields.
#[derive(Debug, Clone)]
pub struct VerifierInfo {
    pub q_verifier: ExpressionCodeEntry,
    /// The FRI polynomial's code, evaluated at each query: `None` unless the opening is FRI.
    pub query_verifier: Option<ExpressionCodeEntry>,
}

impl Serialize for VerifierInfo {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(1 + usize::from(self.query_verifier.is_some())))?;
        map.serialize_entry("qVerifier", &QVerifierView(&self.q_verifier))?;
        if let Some(query_verifier) = &self.query_verifier {
            map.serialize_entry("queryVerifier", &QueryVerifierView(query_verifier))?;
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

struct QueryVerifierView<'a>(&'a ExpressionCodeEntry);
impl Serialize for QueryVerifierView<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(5))?;
        map.serialize_entry("tmpUsed", &self.0.tmp_used)?;
        map.serialize_entry("code", &self.0.code)?;
        map.serialize_entry("expId", &self.0.exp_id)?;
        map.serialize_entry("stage", &self.0.stage)?;
        map.serialize_entry("line", &self.0.line)?;
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
    /// The evaluation map built during verifier code generation.
    pub ev_map: Vec<EvMapRef>,
    /// The FRI polynomial expression ID (may differ from c_exp_id): `None` unless the opening is
    /// FRI.
    pub fri_exp_id: Option<usize>,
    /// Updated challenges map (with FRI challenges appended; empty unless the opening is FRI).
    pub challenges_map: Vec<ChallengeMapEntry>,
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
    /// Set by `generate_pil_code`: the FRI polynomial's expression, `None` unless the opening is FRI.
    pub fri_exp_id: Option<usize>,
    pub q_deg: usize,
    pub q_dim: usize,
    pub opening_points: Vec<i64>,
    pub cm_pols_map: Vec<SymbolInfo>,
    pub custom_commits_count: usize,
    /// The field the code computes over.
    pub field: FieldCfg,
    /// How the committed polynomials are opened: FRI adds the FRI polynomial and the quotient's
    /// evaluations.
    pub opening: Opening,
}

// ---------------------------------------------------------------------------
// generate_pil_code
// ---------------------------------------------------------------------------

/// Orchestrate code generation for all stages.
///
/// Mirrors JS `generatePilCode(res, symbols, constraints, expressions, hints, debug)`. The opening
/// hook: only `Opening::Fri` generates the FRI polynomial, sets `fri_exp_id` and builds the
/// `queryVerifier`.
pub fn generate_pil_code(
    params: &mut CodeGenParams,
    symbols: &mut Vec<SymbolInfo>,
    constraints: &[ConstraintInfo],
    expressions: &mut Vec<Expression>,
    hints: &[HintInfo],
    debug: bool,
    print_ctx: Option<&PrintCtx>,
) -> Result<PilCodeResult> {
    let mut ev_map_items: Vec<EvMapRef> = Vec::new();
    let mut challenges_map: Vec<ChallengeMapEntry> = Vec::new();

    // Pre-compute witness symbol index for O(1) lookups in fix_commit_pol
    let witness_index = build_witness_index(symbols, params.air_id, params.airgroup_id);

    // In non-debug mode: generate verifier code, then, for FRI, the FRI polynomial
    let q_verifier = if !debug {
        let qv = generate_constraint_polynomial_verifier_code(
            params,
            symbols,
            expressions,
            &mut ev_map_items,
            &witness_index,
        )?;

        params.fri_exp_id = match params.opening {
            Opening::Fri => {
                // Generate FRI polynomial (mirrors JS: generateFRIPolynomial(res, symbols, expressions))
                let ev_map_for_fri: Vec<EvMapItem> = ev_map_items
                    .iter()
                    .map(|e| EvMapItem {
                        entry_type: e.entry_type.clone(),
                        id: e.id,
                        prime: e.prime,
                        commit_id: e.commit_id,
                    })
                    .collect();

                let fri_result = fri_poly::generate_fri_polynomial(
                    params.n_stages,
                    expressions,
                    symbols,
                    &ev_map_for_fri,
                    &params.opening_points,
                    &mut challenges_map,
                    &params.field,
                )?;
                Some(fri_result.fri_exp_id)
            }
            Opening::Shplonk => None,
        };

        qv
    } else {
        // Debug mode builds no FRI polynomial; for FRI, cExpId stands in for it (as in the JS setup).
        params.fri_exp_id = match params.opening {
            Opening::Fri => Some(params.c_exp_id),
            Opening::Shplonk => None,
        };
        ExpressionCodeEntry { tmp_used: 0, code: Vec::new(), exp_id: 0, stage: 0, dest: None, line: String::new() }
    };

    let hints_info = add_hints_info(params, expressions, hints, false, print_ctx)?;

    let mut expressions_code = generate_expressions_code(params, symbols, expressions, &witness_index)?;

    // Build query_verifier from the FRI expression code entry.
    // In JS, `find` returns a reference, so modifying the found element also
    // modifies the `expressionsCode` array. We replicate this by modifying
    // the entry in-place in `expressions_code` first, then cloning.
    let query_verifier = match params.fri_exp_id {
        None => None,
        Some(fri_exp_id) => {
            let fri_entry_idx = expressions_code
                .iter()
                .position(|e| e.exp_id == fri_exp_id)
                .ok_or(PilInfoError::FriCodeMissing(fri_exp_id))?;

            // Overwrite last dest to be a tmp with the extension's dim (in-place)
            let fri_entry = &mut expressions_code[fri_entry_idx];
            if let Some(last) = fri_entry.code.last_mut() {
                last.dest = CodeRef {
                    ref_type: "tmp".to_string(),
                    id: fri_entry.tmp_used - 1,
                    dim: params.field.ext_dim(),
                    prime: None,
                    value: None,
                    stage: None,
                    stage_id: None,
                    commit_id: None,
                    opening: None,
                    boundary_id: None,
                    airgroup_id: None,
                    exp_id: None,
                };
            }

            Some(fri_entry.clone())
        }
    };

    let constraints_code = generate_constraints_debug_code(params, symbols, constraints, expressions, &witness_index)?;

    let fri_exp_id = params.fri_exp_id;

    // Verifier entries only: these feed the Rust/C++ verifiers and the circom
    // recursion circuit, none of which is the prover's cExp.
    let mut q_verifier = q_verifier;
    let mut query_verifier = query_verifier;
    let verifier_entries = std::iter::once(("qVerifier", &mut q_verifier))
        .chain(query_verifier.as_mut().map(|entry| ("queryVerifier", entry)));
    for (name, entry) in verifier_entries {
        let result = cse::cse_code(&entry.code);
        tracing::debug!("CSE {name}: {} -> {} ops", entry.code.len(), result.code.len());
        entry.code = result.code;
    }

    Ok(PilCodeResult {
        expressions_info: ExpressionsInfo { hints_info, expressions_code, constraints: constraints_code },
        verifier_info: VerifierInfo { q_verifier, query_verifier },
        ev_map: ev_map_items,
        fri_exp_id,
        challenges_map,
    })
}

// ---------------------------------------------------------------------------
// generateExpressionsCode
// ---------------------------------------------------------------------------

/// Generate code blocks for all kept/imPol/cExp/friExp expressions.
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
        let is_fri_exp = params.fri_exp_id == Some(j);
        let dominated = !exp.keep.unwrap_or(false) && !exp.im_pol && j != params.c_exp_id && !is_fri_exp;
        if dominated {
            continue;
        }

        let dom = if j == params.c_exp_id || is_fri_exp { "ext" } else { "n" };

        let mut ctx =
            CodeGenCtx::new(params.air_id, params.airgroup_id, exp.stage, dom, false, Vec::new(), &params.field);
        ctx.witness_by_exp_id = Arc::clone(witness_index);

        if is_fri_exp {
            ctx.opening_points = params.opening_points.clone();
        }

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
                    opening: None,
                    boundary_id: None,
                    airgroup_id: None,
                    exp_id: None,
                };
            }
        }

        if is_fri_exp {
            if let Some(last) = block.code.last_mut() {
                last.dest = CodeRef {
                    ref_type: "f".to_string(),
                    id: 0,
                    dim: params.field.ext_dim(),
                    prime: None,
                    value: None,
                    stage: None,
                    stage_id: None,
                    commit_id: None,
                    opening: None,
                    boundary_id: None,
                    airgroup_id: None,
                    exp_id: None,
                };
            }
        }

        // Match JS `expInfo.stage = exp.stage || 0`:
        // In JS, the FRI expression ends up with stage=NaN (due to eval/xDivXSubXi
        // nodes lacking a stage property), and NaN||0 gives 0.  Replicate this by
        // clamping stages beyond nStages+1 (the Q stage) to 0.
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
        let mut ctx =
            CodeGenCtx::new(params.air_id, params.airgroup_id, params.n_stages, "n", false, Vec::new(), &params.field);
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
// generateConstraintPolynomialVerifierCode
// ---------------------------------------------------------------------------

/// Generate verifier code for the constraint polynomial.
///
/// Mirrors JS `generateConstraintPolynomialVerifierCode(res, verifierInfo, symbols, expressions)`.
fn generate_constraint_polynomial_verifier_code(
    params: &CodeGenParams,
    symbols: &[SymbolInfo],
    expressions: &[Expression],
    ev_map_out: &mut Vec<EvMapRef>,
    witness_index: &Arc<HashMap<(usize, usize, usize), usize>>,
) -> Result<ExpressionCodeEntry> {
    let mut ctx = CodeGenCtx::new(
        params.air_id,
        params.airgroup_id,
        params.n_stages + 1,
        "n",
        true,
        params.opening_points.clone(),
        &params.field,
    );
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

    // Build the evaluation map from expression symbols
    let mut evals: Vec<EvMapItem> = Vec::new();
    let mut explored = vec![false; expressions.len()];
    add_info_expressions_symbols(&mut evals, expressions, params.c_exp_id, &mut explored);

    for eval_item in &evals {
        let prime = eval_item.prime;
        let opening_pos = params.opening_points.iter().position(|&p| p == prime).unwrap_or(0);
        let mut rf = EvMapRef {
            entry_type: eval_item.entry_type.clone(),
            id: eval_item.id,
            prime,
            opening_pos,
            commit_id: None,
        };
        if eval_item.entry_type == "custom" {
            rf.commit_id = eval_item.commit_id;
        }
        ctx.ev_map.push(rf);
    }

    // Add Q polynomial columns to ev_map. FRI opens the quotient's pieces and the verifier checks
    // their evaluations against Q(xi); with SHPLONK the verifier computes Q(xi) itself
    // (pilfflonk/docs/protocol.md#constraint-polynomial).
    if params.opening == Opening::Fri {
        let q_index = params
            .cm_pols_map
            .iter()
            .position(|p| p.stage == Some(params.n_stages + 1) && p.stage_id == Some(0))
            .unwrap_or(0);
        let opening_pos = params.opening_points.iter().position(|&p| p == 0).unwrap_or(0);
        for i in 0..params.q_deg {
            ctx.ev_map.push(EvMapRef {
                entry_type: "cm".to_string(),
                id: q_index + i,
                prime: 0,
                opening_pos,
                commit_id: None,
            });
        }
    }

    // Sort ev_map by (openingPos, reverse type order, id, prime)
    let custom_commits_count = params.custom_commits_count;
    ctx.ev_map.sort_by(|a, b| {
        let a_type_key = type_sort_key(&a.entry_type, a.commit_id, custom_commits_count);
        let b_type_key = type_sort_key(&b.entry_type, b.commit_id, custom_commits_count);

        a.opening_pos
            .cmp(&b.opening_pos)
            .then(b_type_key.cmp(&a_type_key))
            .then(a.id.cmp(&b.id))
            .then(a.prime.cmp(&b.prime))
    });

    // Build the hash index after sorting so fix_eval can do O(1) lookups
    rebuild_ev_map_index(&mut ctx);

    pil_code_gen(&mut ctx, symbols, expressions, params.c_exp_id, 0)?;
    let block = build_code(&mut ctx)?;

    *ev_map_out = ctx.ev_map;

    Ok(ExpressionCodeEntry {
        tmp_used: block.tmp_used,
        code: block.code,
        exp_id: params.c_exp_id,
        stage: 0,
        dest: None,
        line: String::new(),
    })
}

/// Compute a sort key for ev_map type ordering.
/// cm=0, const=1, custom{i}=i+2
fn type_sort_key(entry_type: &str, commit_id: Option<usize>, _custom_count: usize) -> usize {
    match entry_type {
        "cm" => 0,
        "const" => 1,
        _ => {
            // custom type: key is commit_id + 2
            commit_id.unwrap_or(0) + 2
        }
    }
}

// ---------------------------------------------------------------------------
// addHintsInfo
// ---------------------------------------------------------------------------

/// Process hints into flat hint field values.
///
/// Mirrors JS `addHintsInfo(res, expressions, hints, global)`.
fn add_hints_info(
    params: &CodeGenParams,
    expressions: &mut Vec<Expression>,
    hints: &[HintInfo],
    _global: bool,
    print_ctx: Option<&PrintCtx>,
) -> Result<Vec<ProcessedHint>> {
    let mut result = Vec::new();

    for hint in hints {
        let mut processed_fields = Vec::new();

        for field in &hint.fields {
            let flat_values = process_hint_field_values(&field.values, params, expressions, &[], print_ctx)?;

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
    params: &CodeGenParams,
    expressions: &mut Vec<Expression>,
    pos: &[usize],
    print_ctx: Option<&PrintCtx>,
) -> Result<Vec<ProcessedHintField>> {
    let mut result = Vec::new();

    for (j, field) in values.iter().enumerate() {
        let mut current_pos: Vec<usize> = pos.to_vec();
        current_pos.push(j);

        match field {
            HintFieldValue::Array(arr) => {
                let inner = process_hint_field_values(arr, params, expressions, &current_pos, print_ctx)?;
                result.extend(inner);
            }
            HintFieldValue::Single(expr) => {
                let processed = process_single_hint_field(expr, params, expressions, &current_pos, print_ctx)?;
                result.push(processed);
            }
        }
    }

    Ok(result)
}

/// Process a single (leaf) hint field expression.
fn process_single_hint_field(
    expr: &Expression,
    params: &CodeGenParams,
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
            let prime_index =
                params.opening_points.iter().position(|&p| p == row_offset).map(|p| p as isize).unwrap_or(-1);
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
            fri_exp_id: Some(3),
            q_deg: 1,
            q_dim: 1,
            opening_points: vec![0],
            cm_pols_map: vec![SymbolInfo { stage: Some(2), stage_id: Some(0), ..Default::default() }],
            custom_commits_count: 0,
            field: FieldCfg::goldilocks(),
            opening: Opening::Fri,
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
            fri_exp_id: Some(998),
            q_deg: 1,
            q_dim: 1,
            opening_points: vec![0],
            cm_pols_map: Vec::new(),
            custom_commits_count: 0,
            field: FieldCfg::goldilocks(),
            opening: Opening::Fri,
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

        let result = add_hints_info(&params, &mut expressions, &hints, false, None).unwrap();
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
            fri_exp_id: Some(999),
            q_deg: 1,
            q_dim: 1,
            opening_points: vec![0],
            cm_pols_map: Vec::new(),
            custom_commits_count: 0,
            field: FieldCfg::goldilocks(),
            opening: Opening::Fri,
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

        let result = process_hint_field_values(&values, &params, &mut expressions, &[], None).unwrap();
        // Should flatten to 2 entries
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].pos, vec![0, 0]);
        assert_eq!(result[1].pos, vec![0, 1]);
    }

    /// Was `panic!("Invalid hint op")`.
    #[test]
    fn a_hint_field_of_an_unknown_op_is_an_error() {
        let zi = Expression { op: "Zi".to_string(), ..Default::default() };
        let err = process_single_hint_field(&zi, &make_params(), &mut [], &[0], None).unwrap_err();
        assert!(matches!(&err, PilInfoError::UnknownHintOp(op) if op == "Zi"), "{err}");
    }
}
