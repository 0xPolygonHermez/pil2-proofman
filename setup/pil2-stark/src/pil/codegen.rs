//! Expression DAG code generation.
//!
//! Walks an expression tree, emitting add/sub/mul/copy operations into a
//! linear code buffer and resolving expression references to temporaries.

use std::collections::HashMap;

use crate::expr::expression::Expression;
use crate::types::pilout_info::{SymbolInfo, FIELD_EXTENSION};
use crate::types::output::{CodeEntry, CodeRef};

// ---------------------------------------------------------------------------
// CodeGenCtx: mutable state threaded through code generation
// ---------------------------------------------------------------------------

/// Calculated entry: tracks whether an expression at (expId, prime) was
/// already emitted and whether it was promoted to a committed polynomial.
#[derive(Debug, Clone)]
pub struct CalcEntry {
    pub cm: bool,
    pub tmp_id: Option<usize>,
}

/// Context carried through `pil_code_gen` / `eval_exp` / `build_code`.
#[derive(Debug)]
pub struct CodeGenCtx {
    pub air_id: usize,
    pub airgroup_id: usize,
    pub stage: usize,
    pub dom: String,
    pub verifier_evaluations: bool,
    pub opening_points: Vec<i64>,
    pub ev_map: Vec<EvMapRef>,
    /// Pre-computed index: (entry_type, id, opening_pos, commit_id) -> first ev_map index.
    /// Enables O(1) lookup in `fix_eval` instead of O(|ev_map|) linear scan.
    /// The commit_id is needed to disambiguate custom commit entries that share the
    /// same (type, id, opening_pos) but belong to different custom commits.
    pub ev_map_index: HashMap<(String, usize, usize, Option<usize>), usize>,

    pub tmp_used: usize,
    pub code: Vec<CodeEntry>,
    /// calculated[expId][prime] -> CalcEntry
    pub calculated: HashMap<usize, HashMap<i64, CalcEntry>>,
    /// Used by `build_code` to resolve expression refs to temporaries.
    /// Keyed by prime (can be negative, e.g. -1) then by expression id.
    exp_map: HashMap<i64, HashMap<usize, usize>>,
    /// Pre-computed symbol index: (exp_id, air_id, airgroup_id) -> symbol index
    /// for witness imPol lookups. Shared across all codegen calls via reference;
    /// stored as owned only when context is self-contained (tests).
    pub witness_by_exp_id: std::sync::Arc<HashMap<(usize, usize, usize), usize>>,
}

/// An entry in the evaluation map built during verifier code generation.
#[derive(Debug, Clone)]
pub struct EvMapRef {
    pub entry_type: String,
    pub id: usize,
    pub prime: i64,
    pub opening_pos: usize,
    pub commit_id: Option<usize>,
}

impl CodeGenCtx {
    /// Create a new context for expression code generation.
    pub fn new(
        air_id: usize,
        airgroup_id: usize,
        stage: usize,
        dom: &str,
        verifier_evaluations: bool,
        opening_points: Vec<i64>,
        ev_map: Vec<EvMapRef>,
    ) -> Self {
        Self {
            air_id,
            airgroup_id,
            stage,
            dom: dom.to_string(),
            verifier_evaluations,
            opening_points,
            ev_map,
            ev_map_index: HashMap::new(),
            tmp_used: 0,
            code: Vec::new(),
            calculated: HashMap::new(),
            exp_map: HashMap::new(),
            witness_by_exp_id: std::sync::Arc::new(HashMap::new()),
        }
    }
}

/// Rebuild the `ev_map_index` from the current `ev_map` contents.
///
/// Must be called after `ev_map` is fully populated and sorted so that
/// `fix_eval` can perform O(1) lookups instead of O(|ev_map|) linear scans.
pub fn rebuild_ev_map_index(ctx: &mut CodeGenCtx) {
    ctx.ev_map_index.clear();
    for (i, e) in ctx.ev_map.iter().enumerate() {
        let key = (e.entry_type.clone(), e.id, e.opening_pos, e.commit_id);
        ctx.ev_map_index.entry(key).or_insert(i); // first match wins, like JS findIndex
    }
}

// ---------------------------------------------------------------------------
// Helper: construct a CodeRef
// ---------------------------------------------------------------------------

fn make_ref(ref_type: &str, id: usize, dim: usize) -> CodeRef {
    CodeRef {
        ref_type: ref_type.to_string(),
        id,
        dim,
        prime: None,
        value: None,
        stage: None,
        stage_id: None,
        commit_id: None,
        opening: None,
        boundary_id: None,
        airgroup_id: None,
        exp_id: None,
    }
}

// ---------------------------------------------------------------------------
// pilCodeGen
// ---------------------------------------------------------------------------

/// Generate evaluation code for a single expression.
///
/// Mirrors JS `pilCodeGen(ctx, symbols, expressions, expId, prime)`.
pub fn pil_code_gen(
    ctx: &mut CodeGenCtx,
    symbols: &[SymbolInfo],
    expressions: &[Expression],
    exp_id: usize,
    prime: i64,
) {
    if let Some(inner) = ctx.calculated.get(&exp_id) {
        if inner.contains_key(&prime) {
            return;
        }
    }

    calculate_deps(ctx, symbols, expressions, &expressions[exp_id], prime);

    let e = &expressions[exp_id];

    // Create a sub-context for this expression.
    // Share `calculated` and `witness_by_exp_id` (Arc) to avoid expensive clones.
    // Only clone code/ev_map/exp_map which are small or need isolation.
    let mut code_ctx = CodeGenCtx {
        air_id: ctx.air_id,
        airgroup_id: ctx.airgroup_id,
        stage: ctx.stage,
        dom: ctx.dom.clone(),
        verifier_evaluations: ctx.verifier_evaluations,
        opening_points: ctx.opening_points.clone(),
        ev_map: std::mem::take(&mut ctx.ev_map), // move instead of clone
        ev_map_index: std::mem::take(&mut ctx.ev_map_index),
        tmp_used: ctx.tmp_used,
        code: Vec::new(),
        calculated: std::mem::take(&mut ctx.calculated),
        exp_map: HashMap::new(),
        witness_by_exp_id: std::sync::Arc::clone(&ctx.witness_by_exp_id),
    };

    let ret_ref = eval_exp(&mut code_ctx, symbols, expressions, e, prime);

    let mut r = CodeRef {
        ref_type: "exp".to_string(),
        id: exp_id,
        dim: e.dim,
        prime: Some(prime),
        value: None,
        stage: None,
        stage_id: None,
        commit_id: None,
        opening: None,
        boundary_id: None,
        airgroup_id: None,
        exp_id: None,
    };

    if ret_ref.ref_type == "tmp" {
        fix_commit_pol(&mut r, &code_ctx, symbols);
        let last_idx = code_ctx.code.len() - 1;
        code_ctx.code[last_idx].dest = r.clone();
        if r.ref_type == "cm" {
            code_ctx.tmp_used -= 1;
        }
    } else {
        fix_commit_pol(&mut r, &code_ctx, symbols);
        code_ctx.code.push(CodeEntry { op: "copy".to_string(), dest: r.clone(), src: vec![ret_ref] });
    }

    ctx.code.extend(code_ctx.code);
    // Restore moved fields
    ctx.calculated = code_ctx.calculated;
    ctx.ev_map = code_ctx.ev_map;
    ctx.ev_map_index = code_ctx.ev_map_index;

    ctx.calculated.entry(exp_id).or_default().insert(prime, CalcEntry { cm: false, tmp_id: Some(code_ctx.tmp_used) });

    if code_ctx.tmp_used > ctx.tmp_used {
        ctx.tmp_used = code_ctx.tmp_used;
    }
}

// ---------------------------------------------------------------------------
// evalExp
// ---------------------------------------------------------------------------

/// Recursively evaluate an expression node, emitting code operations.
///
/// Mirrors JS `evalExp(ctx, symbols, expressions, exp, prime)`.
fn eval_exp(
    ctx: &mut CodeGenCtx,
    symbols: &[SymbolInfo],
    expressions: &[Expression],
    exp: &Expression,
    prime: i64,
) -> CodeRef {
    match exp.op.as_str() {
        "add" | "sub" | "mul" => {
            let mut values = Vec::new();
            for child in &exp.values {
                let child_expr = child.resolve(expressions);
                values.push(eval_exp(ctx, symbols, expressions, child_expr, prime));
            }
            let max_dim = values.iter().map(|v| v.dim).max().unwrap_or(1);
            let r = make_ref("tmp", ctx.tmp_used, max_dim);
            ctx.tmp_used += 1;

            ctx.code.push(CodeEntry { op: exp.op.clone(), dest: r.clone(), src: values });

            r
        }
        "cm" | "const" | "custom" => build_column_ref(ctx, symbols, expressions, exp, prime, false),
        "exp" => {
            let ref_id = exp.id.unwrap_or(0);
            let ref_expr = &expressions[ref_id];
            if matches!(ref_expr.op.as_str(), "cm" | "const" | "custom") {
                build_column_ref(ctx, symbols, expressions, exp, prime, true)
            } else {
                let p = exp.row_offset.unwrap_or(prime);
                let mut r = CodeRef {
                    ref_type: "exp".to_string(),
                    id: ref_id,
                    dim: exp.dim,
                    prime: Some(p),
                    value: None,
                    stage: None,
                    stage_id: None,
                    commit_id: None,
                    opening: None,
                    boundary_id: None,
                    airgroup_id: None,
                    exp_id: Some(ref_id),
                };
                fix_commit_pol(&mut r, ctx, symbols);
                r
            }
        }
        "challenge" => CodeRef {
            ref_type: "challenge".to_string(),
            id: exp.id.unwrap_or(0),
            dim: exp.dim,
            prime: None,
            value: None,
            stage: Some(exp.stage),
            stage_id: exp.stage_id,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        },
        "public" => CodeRef {
            ref_type: "public".to_string(),
            id: exp.id.unwrap_or(0),
            dim: 1,
            prime: None,
            value: None,
            stage: None,
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        },
        "proofvalue" => CodeRef {
            ref_type: "proofvalue".to_string(),
            id: exp.id.unwrap_or(0),
            dim: exp.dim,
            prime: None,
            value: None,
            stage: Some(exp.stage),
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        },
        "number" => CodeRef {
            ref_type: "number".to_string(),
            id: 0,
            dim: 1,
            prime: None,
            value: exp.value.as_ref().map(|v| v.to_string()),
            stage: None,
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        },
        "eval" => CodeRef {
            ref_type: "eval".to_string(),
            id: exp.id.unwrap_or(0),
            dim: exp.dim,
            prime: None,
            value: None,
            stage: None,
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        },
        "airgroupvalue" | "airvalue" => CodeRef {
            ref_type: exp.op.clone(),
            id: exp.id.unwrap_or(0),
            dim: exp.dim,
            prime: None,
            value: None,
            stage: Some(exp.stage),
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: exp.airgroup_id,
            exp_id: None,
        },
        "xDivXSubXi" => CodeRef {
            ref_type: "xDivXSubXi".to_string(),
            id: exp.id.unwrap_or(0),
            dim: FIELD_EXTENSION,
            prime: None,
            value: None,
            stage: None,
            stage_id: None,
            commit_id: None,
            opening: exp.opening,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        },
        "Zi" => CodeRef {
            ref_type: "Zi".to_string(),
            id: 0,
            dim: 1,
            prime: None,
            value: None,
            stage: None,
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: exp.boundary_id,
            airgroup_id: None,
            exp_id: None,
        },
        _ => panic!("Invalid op: {}", exp.op),
    }
}

/// Build a CodeRef for column-like ops (cm, const, custom), optionally via
/// an `exp` indirection.
fn build_column_ref(
    ctx: &mut CodeGenCtx,
    symbols: &[SymbolInfo],
    expressions: &[Expression],
    exp: &Expression,
    prime: i64,
    via_exp: bool,
) -> CodeRef {
    let expr = if via_exp {
        let ref_id = exp.id.unwrap_or(0);
        &expressions[ref_id]
    } else {
        exp
    };
    let p = expr.row_offset.unwrap_or(prime);
    let mut r = CodeRef {
        ref_type: expr.op.clone(),
        id: expr.id.unwrap_or(0),
        dim: expr.dim,
        prime: Some(p),
        value: None,
        stage: None,
        stage_id: None,
        commit_id: expr.commit_id,
        opening: None,
        boundary_id: None,
        airgroup_id: None,
        exp_id: None,
    };
    if ctx.verifier_evaluations {
        fix_eval(&mut r, ctx, symbols);
    }
    r
}

// ---------------------------------------------------------------------------
// calculateDeps
// ---------------------------------------------------------------------------

/// Ensure dependencies are computed before the current expression.
fn calculate_deps(
    ctx: &mut CodeGenCtx,
    symbols: &[SymbolInfo],
    expressions: &[Expression],
    exp: &Expression,
    prime: i64,
) {
    match exp.op.as_str() {
        "exp" => {
            let p = exp.row_offset.unwrap_or(prime);
            let ref_id = exp.id.unwrap_or(0);
            pil_code_gen(ctx, symbols, expressions, ref_id, p);
        }
        "add" | "sub" | "mul" => {
            for child in &exp.values {
                let child_expr = child.resolve(expressions);
                calculate_deps(ctx, symbols, expressions, child_expr, prime);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// fixCommitPol
// ---------------------------------------------------------------------------

/// If an expression reference corresponds to a witness symbol that is an
/// intermediate polynomial and meets the promotion criteria, rewrite the
/// reference to point to the committed polynomial column.
fn fix_commit_pol(r: &mut CodeRef, ctx: &CodeGenCtx, symbols: &[SymbolInfo]) {
    // Use pre-computed index for O(1) symbol lookup instead of O(N) linear scan
    let key = (r.id, ctx.air_id, ctx.airgroup_id);
    let symbol = match ctx.witness_by_exp_id.get(&key) {
        Some(&idx) => &symbols[idx],
        None => return,
    };

    let r_prime = r.prime.unwrap_or(0);
    let calc = ctx.calculated.get(&r.id).and_then(|m| m.get(&r_prime));

    if symbol.im_pol && (ctx.dom == "ext" || (symbol.stage.unwrap_or(0) <= ctx.stage && calc.is_some_and(|c| c.cm))) {
        r.ref_type = "cm".to_string();
        r.id = symbol.pol_id.unwrap_or(0);
        r.dim = symbol.dim;
        if ctx.verifier_evaluations {
            fix_eval(r, ctx, symbols);
        }
    } else if !ctx.verifier_evaluations && ctx.dom == "n" && calc.is_some_and(|c| c.cm) {
        r.ref_type = "cm".to_string();
        r.id = symbol.pol_id.unwrap_or(0);
        r.dim = symbol.dim;
    }
}

// ---------------------------------------------------------------------------
// fixEval
// ---------------------------------------------------------------------------

/// Rewrite a column reference into an eval reference using the ev_map index.
///
/// Uses the pre-built `ev_map_index` HashMap for O(1) lookup instead of
/// scanning `ev_map` linearly, which was the source of a >20 min stall on
/// Poseidon2 (31k code entries x 182 ev_map entries).
fn fix_eval(r: &mut CodeRef, ctx: &CodeGenCtx, _symbols: &[SymbolInfo]) {
    let prime = r.prime.unwrap_or(0);
    let opening_pos = match ctx.opening_points.iter().position(|&p| p == prime) {
        Some(pos) => pos,
        None => return, // unknown opening point: do not remap
    };
    let key = (r.ref_type.clone(), r.id, opening_pos, r.commit_id);
    if let Some(&idx) = ctx.ev_map_index.get(&key) {
        r.prime = None;
        r.id = idx;
        r.ref_type = "eval".to_string();
        r.dim = FIELD_EXTENSION;
    }
}

// ---------------------------------------------------------------------------
// fixExpression
// ---------------------------------------------------------------------------

/// Resolve an expression reference to a temporary id in `build_code`.
fn fix_expression(r: &mut CodeRef, ctx: &mut CodeGenCtx) {
    let prime = r.prime.unwrap_or(0);
    // Use a HashMap keyed by prime instead of a Vec indexed by usize, since
    // prime can be negative (e.g. opening point -1).
    let entry = ctx.exp_map.entry(prime).or_default().entry(r.id).or_insert_with(|| {
        let id = ctx.tmp_used;
        ctx.tmp_used += 1;
        id
    });

    r.ref_type = "tmp".to_string();
    r.id = *entry;
    // JS fixExpression does NOT delete r.prime or r.expId, so we keep them.
}

// ---------------------------------------------------------------------------
// fixDimensionsVerifier
// ---------------------------------------------------------------------------

/// Recompute dimensions for verifier code where all destinations are tmps.
fn fix_dimensions_verifier(code: &mut [CodeEntry]) {
    let mut tmp_dim: Vec<usize> = Vec::new();

    for entry in code.iter_mut() {
        assert!(matches!(entry.op.as_str(), "add" | "sub" | "mul" | "copy"), "Invalid op: {}", entry.op,);
        assert_eq!(entry.dest.ref_type, "tmp", "Invalid dest type: {}", entry.dest.ref_type);

        let new_dim = entry.src.iter().map(|s| get_dim(s, &tmp_dim)).max().unwrap_or(1);

        // Ensure tmp_dim is large enough
        let dest_id = entry.dest.id;
        if tmp_dim.len() <= dest_id {
            tmp_dim.resize(dest_id + 1, 0);
        }
        tmp_dim[dest_id] = new_dim;
        entry.dest.dim = new_dim;

        // Update source dims in place
        for s in entry.src.iter_mut() {
            let d = get_dim(s, &tmp_dim);
            s.dim = d;
        }
    }
}

fn get_dim(r: &CodeRef, tmp_dim: &[usize]) -> usize {
    if r.ref_type == "tmp" {
        tmp_dim.get(r.id).copied().unwrap_or(0)
    } else if r.ref_type == "Zi" {
        FIELD_EXTENSION
    } else {
        r.dim
    }
}

// ---------------------------------------------------------------------------
// buildCode
// ---------------------------------------------------------------------------

/// Finalize generated code: resolve expression references to temporaries,
/// optionally fix verifier dimensions, and return the code block.
///
/// Resets the context for re-use.
pub fn build_code(ctx: &mut CodeGenCtx) -> CodeBlock {
    ctx.exp_map.clear();

    for i in 0..ctx.code.len() {
        let src_len = ctx.code[i].src.len();
        for j in 0..src_len {
            if ctx.code[i].src[j].ref_type == "exp" {
                // We need to work around the borrow checker by extracting,
                // mutating, and re-inserting.
                let mut src_ref = ctx.code[i].src[j].clone();
                fix_expression(&mut src_ref, ctx);
                ctx.code[i].src[j] = src_ref;
            }
        }
        if ctx.code[i].dest.ref_type == "exp" {
            let mut dest_ref = ctx.code[i].dest.clone();
            fix_expression(&mut dest_ref, ctx);
            ctx.code[i].dest = dest_ref;
        }
    }

    if ctx.verifier_evaluations {
        fix_dimensions_verifier(&mut ctx.code);
    }

    let code = CodeBlock { tmp_used: ctx.tmp_used, code: std::mem::take(&mut ctx.code) };

    ctx.calculated.clear();
    ctx.tmp_used = 0;

    code
}

// ---------------------------------------------------------------------------
// CodeBlock: the output of build_code
// ---------------------------------------------------------------------------

/// A finalized code block with its temporary count.
#[derive(Debug, Clone)]
pub struct CodeBlock {
    pub tmp_used: usize,
    pub code: Vec<CodeEntry>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::expression::{ExprChild, Expression};

    fn make_number(val: &str) -> Expression {
        Expression { op: "number".to_string(), value: Some(val.to_string()), dim: 1, ..Default::default() }
    }

    fn make_cm(id: usize) -> Expression {
        Expression { op: "cm".to_string(), id: Some(id), dim: 1, stage: 1, row_offset: Some(0), ..Default::default() }
    }

    fn make_add(lhs: usize, rhs: usize) -> Expression {
        Expression {
            op: "add".to_string(),
            values: vec![ExprChild::Id(lhs), ExprChild::Id(rhs)],
            dim: 1,
            ..Default::default()
        }
    }

    fn make_mul(lhs: usize, rhs: usize) -> Expression {
        Expression {
            op: "mul".to_string(),
            values: vec![ExprChild::Id(lhs), ExprChild::Id(rhs)],
            dim: 1,
            ..Default::default()
        }
    }

    fn make_exp_ref(ref_id: usize, dim: usize) -> Expression {
        Expression { op: "exp".to_string(), id: Some(ref_id), dim, ..Default::default() }
    }

    fn new_ctx() -> CodeGenCtx {
        CodeGenCtx::new(0, 0, 1, "n", false, vec![0], Vec::new())
    }

    #[test]
    fn test_simple_add() {
        // expressions[0] = number("5")
        // expressions[1] = cm(0)
        // expressions[2] = add(0, 1)
        let expressions = vec![make_number("5"), make_cm(0), make_add(0, 1)];
        let symbols: Vec<SymbolInfo> = Vec::new();
        let mut ctx = new_ctx();

        pil_code_gen(&mut ctx, &symbols, &expressions, 2, 0);
        let block = build_code(&mut ctx);

        // Should have at least one code entry
        assert!(!block.code.is_empty());
        // The operation should be "add"
        assert_eq!(block.code[0].op, "add");
    }

    #[test]
    fn test_mul_expression() {
        let expressions = vec![make_number("3"), make_cm(0), make_mul(0, 1)];
        let symbols: Vec<SymbolInfo> = Vec::new();
        let mut ctx = new_ctx();

        pil_code_gen(&mut ctx, &symbols, &expressions, 2, 0);
        let block = build_code(&mut ctx);

        assert!(!block.code.is_empty());
        assert_eq!(block.code[0].op, "mul");
    }

    #[test]
    fn test_nested_expression_with_dep() {
        // expressions[0] = number("1")
        // expressions[1] = cm(0)
        // expressions[2] = add(0, 1)       -- inner expression
        // expressions[3] = exp(ref=2)       -- reference to inner
        // expressions[4] = number("2")
        // expressions[5] = mul(3, 4)        -- outer: mul(exp_ref(2), 2)
        let expressions =
            vec![make_number("1"), make_cm(0), make_add(0, 1), make_exp_ref(2, 1), make_number("2"), make_mul(3, 4)];
        let symbols: Vec<SymbolInfo> = Vec::new();
        let mut ctx = new_ctx();

        pil_code_gen(&mut ctx, &symbols, &expressions, 5, 0);
        let block = build_code(&mut ctx);

        // Should have code for the inner add + the outer mul
        assert!(block.code.len() >= 2);
    }

    #[test]
    fn test_build_code_resets_ctx() {
        let expressions = vec![make_number("1"), make_cm(0), make_add(0, 1)];
        let symbols: Vec<SymbolInfo> = Vec::new();
        let mut ctx = new_ctx();

        pil_code_gen(&mut ctx, &symbols, &expressions, 2, 0);
        let _ = build_code(&mut ctx);

        // After build_code, context should be reset
        assert!(ctx.code.is_empty());
        assert_eq!(ctx.tmp_used, 0);
        assert!(ctx.calculated.is_empty());
    }

    #[test]
    fn test_number_ref() {
        let expressions = vec![make_number("42")];
        let symbols: Vec<SymbolInfo> = Vec::new();
        let mut ctx = new_ctx();

        pil_code_gen(&mut ctx, &symbols, &expressions, 0, 0);
        let block = build_code(&mut ctx);

        assert!(!block.code.is_empty());
        // The last entry should be a copy with source being a number
        let last = block.code.last().unwrap();
        assert_eq!(last.op, "copy");
        assert_eq!(last.src[0].ref_type, "number");
        assert_eq!(last.src[0].value.as_deref(), Some("42"));
    }

    // -----------------------------------------------------------------------
    // fix_eval tests
    // -----------------------------------------------------------------------

    /// Helper: build a CodeGenCtx with verifier_evaluations=true, a populated
    /// ev_map and the corresponding ev_map_index.
    fn verifier_ctx_with_ev_map(opening_points: Vec<i64>, ev_map: Vec<EvMapRef>) -> CodeGenCtx {
        let mut ctx = CodeGenCtx::new(0, 0, 1, "n", true, opening_points, ev_map);
        rebuild_ev_map_index(&mut ctx);
        ctx
    }

    fn make_code_ref(ref_type: &str, id: usize, prime: Option<i64>) -> CodeRef {
        CodeRef {
            ref_type: ref_type.to_string(),
            id,
            dim: 1,
            prime,
            value: None,
            stage: None,
            stage_id: None,
            commit_id: None,
            opening: None,
            boundary_id: None,
            airgroup_id: None,
            exp_id: None,
        }
    }

    #[test]
    fn test_fix_eval_finds_matching_eval() {
        let ev_map = vec![
            EvMapRef { entry_type: "cm".into(), id: 0, prime: 0, opening_pos: 0, commit_id: None },
            EvMapRef { entry_type: "const".into(), id: 5, prime: 0, opening_pos: 0, commit_id: None },
        ];
        let ctx = verifier_ctx_with_ev_map(vec![0], ev_map);
        let symbols: Vec<SymbolInfo> = Vec::new();

        // Look up "const" id=5 at prime=0 -> opening_pos=0 -> should match index 1
        let mut r = make_code_ref("const", 5, Some(0));
        fix_eval(&mut r, &ctx, &symbols);

        assert_eq!(r.ref_type, "eval");
        assert_eq!(r.id, 1);
        assert_eq!(r.dim, FIELD_EXTENSION);
        assert!(r.prime.is_none());
    }

    #[test]
    fn test_fix_eval_missing_opening_point_returns_early() {
        let ev_map = vec![EvMapRef { entry_type: "cm".into(), id: 0, prime: 0, opening_pos: 0, commit_id: None }];
        // opening_points only contains 0; prime=99 is not present
        let ctx = verifier_ctx_with_ev_map(vec![0], ev_map);
        let symbols: Vec<SymbolInfo> = Vec::new();

        let mut r = make_code_ref("cm", 0, Some(99));
        fix_eval(&mut r, &ctx, &symbols);

        // Should be unchanged because prime=99 is not in opening_points
        assert_eq!(r.ref_type, "cm");
        assert_eq!(r.id, 0);
        assert_eq!(r.prime, Some(99));
    }

    #[test]
    fn test_fix_eval_missing_eval_entry_no_change() {
        let ev_map = vec![EvMapRef { entry_type: "cm".into(), id: 0, prime: 0, opening_pos: 0, commit_id: None }];
        let ctx = verifier_ctx_with_ev_map(vec![0], ev_map);
        let symbols: Vec<SymbolInfo> = Vec::new();

        // "const" id=99 does not exist in ev_map
        let mut r = make_code_ref("const", 99, Some(0));
        fix_eval(&mut r, &ctx, &symbols);

        assert_eq!(r.ref_type, "const");
        assert_eq!(r.id, 99);
    }

    #[test]
    fn test_fix_eval_first_match_wins() {
        // Two entries with the same (entry_type, id, opening_pos); the first
        // one inserted should win (replicating JS Array.findIndex semantics).
        let ev_map = vec![
            EvMapRef { entry_type: "cm".into(), id: 3, prime: 0, opening_pos: 0, commit_id: None },
            EvMapRef { entry_type: "cm".into(), id: 3, prime: 0, opening_pos: 0, commit_id: None },
        ];
        let ctx = verifier_ctx_with_ev_map(vec![0], ev_map);
        let symbols: Vec<SymbolInfo> = Vec::new();

        let mut r = make_code_ref("cm", 3, Some(0));
        fix_eval(&mut r, &ctx, &symbols);

        assert_eq!(r.ref_type, "eval");
        assert_eq!(r.id, 0); // first entry at index 0, not index 1
    }
}
