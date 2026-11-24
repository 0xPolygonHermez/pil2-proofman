//! Expression DAG code generation.
//!
//! Walks an expression tree, emitting add/sub/mul/copy operations into a
//! linear code buffer and resolving expression references to temporaries.

use std::collections::HashMap;

use crate::error::{PilInfoError, Result};
use crate::expr::expression::Expression;
use crate::types::pilout_info::SymbolInfo;
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

impl CodeGenCtx {
    /// Create a new context for prover expression code generation.
    pub fn new(air_id: usize, airgroup_id: usize, stage: usize, dom: &str) -> Self {
        Self {
            air_id,
            airgroup_id,
            stage,
            dom: dom.to_string(),
            tmp_used: 0,
            code: Vec::new(),
            calculated: HashMap::new(),
            exp_map: HashMap::new(),
            witness_by_exp_id: std::sync::Arc::new(HashMap::new()),
        }
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
/// Mirrors JS `pilCodeGen(ctx, symbols, expressions, expId, prime)`. Fails on an op it does not
/// know.
pub fn pil_code_gen(
    ctx: &mut CodeGenCtx,
    symbols: &[SymbolInfo],
    expressions: &[Expression],
    exp_id: usize,
    prime: i64,
) -> Result<()> {
    if let Some(inner) = ctx.calculated.get(&exp_id) {
        if inner.contains_key(&prime) {
            return Ok(());
        }
    }

    calculate_deps(ctx, symbols, expressions, &expressions[exp_id], prime)?;

    let e = &expressions[exp_id];

    // Create a sub-context for this expression.
    // Share `calculated` and `witness_by_exp_id` (Arc) to avoid expensive clones.
    // Only code/exp_map are the sub-context's own.
    let mut code_ctx = CodeGenCtx {
        air_id: ctx.air_id,
        airgroup_id: ctx.airgroup_id,
        stage: ctx.stage,
        dom: ctx.dom.clone(),
        tmp_used: ctx.tmp_used,
        code: Vec::new(),
        calculated: std::mem::take(&mut ctx.calculated),
        exp_map: HashMap::new(),
        witness_by_exp_id: std::sync::Arc::clone(&ctx.witness_by_exp_id),
    };

    let ret_ref = eval_exp(&mut code_ctx, symbols, expressions, e, prime)?;

    let mut r = CodeRef {
        ref_type: "exp".to_string(),
        id: exp_id,
        dim: e.dim,
        prime: Some(prime),
        value: None,
        stage: None,
        stage_id: None,
        commit_id: None,
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

    ctx.calculated.entry(exp_id).or_default().insert(prime, CalcEntry { cm: false, tmp_id: Some(code_ctx.tmp_used) });

    if code_ctx.tmp_used > ctx.tmp_used {
        ctx.tmp_used = code_ctx.tmp_used;
    }
    Ok(())
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
) -> Result<CodeRef> {
    let r = match exp.op.as_str() {
        "add" | "sub" | "mul" => {
            let mut values = Vec::new();
            for child in &exp.values {
                let child_expr = child.resolve(expressions);
                values.push(eval_exp(ctx, symbols, expressions, child_expr, prime)?);
            }
            let max_dim = values.iter().map(|v| v.dim).max().unwrap_or(1);
            let r = make_ref("tmp", ctx.tmp_used, max_dim);
            ctx.tmp_used += 1;

            ctx.code.push(CodeEntry { op: exp.op.clone(), dest: r.clone(), src: values });

            r
        }
        "cm" | "const" | "custom" => build_column_ref(expressions, exp, prime, false),
        "exp" => {
            let ref_id = exp.id.unwrap_or(0);
            let ref_expr = &expressions[ref_id];
            if matches!(ref_expr.op.as_str(), "cm" | "const" | "custom") {
                build_column_ref(expressions, exp, prime, true)
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
            boundary_id: None,
            airgroup_id: exp.airgroup_id,
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
            boundary_id: exp.boundary_id,
            airgroup_id: None,
            exp_id: None,
        },
        _ => return Err(PilInfoError::UnknownOp { pass: "code generation", op: exp.op.clone() }),
    };
    Ok(r)
}

/// Build a CodeRef for column-like ops (cm, const, custom), optionally via
/// an `exp` indirection.
fn build_column_ref(expressions: &[Expression], exp: &Expression, prime: i64, via_exp: bool) -> CodeRef {
    let expr = if via_exp {
        let ref_id = exp.id.unwrap_or(0);
        &expressions[ref_id]
    } else {
        exp
    };
    let p = expr.row_offset.unwrap_or(prime);
    CodeRef {
        ref_type: expr.op.clone(),
        id: expr.id.unwrap_or(0),
        dim: expr.dim,
        prime: Some(p),
        value: None,
        stage: None,
        stage_id: None,
        commit_id: expr.commit_id,
        boundary_id: None,
        airgroup_id: None,
        exp_id: None,
    }
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
) -> Result<()> {
    match exp.op.as_str() {
        "exp" => {
            let p = exp.row_offset.unwrap_or(prime);
            let ref_id = exp.id.unwrap_or(0);
            pil_code_gen(ctx, symbols, expressions, ref_id, p)?;
        }
        "add" | "sub" | "mul" => {
            for child in &exp.values {
                let child_expr = child.resolve(expressions);
                calculate_deps(ctx, symbols, expressions, child_expr, prime)?;
            }
        }
        _ => {}
    }
    Ok(())
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

    let cm = calc.is_some_and(|c| c.cm);
    let promoted = symbol.im_pol && (ctx.dom == "ext" || (symbol.stage.unwrap_or(0) <= ctx.stage && cm));
    if promoted || (ctx.dom == "n" && cm) {
        r.ref_type = "cm".to_string();
        r.id = symbol.pol_id.unwrap_or(0);
        r.dim = symbol.dim;
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
// buildCode
// ---------------------------------------------------------------------------

/// Finalize generated code: resolve expression references to temporaries and return the code block.
///
/// Resets the context for re-use.
pub fn build_code(ctx: &mut CodeGenCtx) -> Result<CodeBlock> {
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

    let code = CodeBlock { tmp_used: ctx.tmp_used, code: std::mem::take(&mut ctx.code) };

    ctx.calculated.clear();
    ctx.tmp_used = 0;

    Ok(code)
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
        CodeGenCtx::new(0, 0, 1, "n")
    }

    #[test]
    fn test_simple_add() {
        // expressions[0] = number("5")
        // expressions[1] = cm(0)
        // expressions[2] = add(0, 1)
        let expressions = vec![make_number("5"), make_cm(0), make_add(0, 1)];
        let symbols: Vec<SymbolInfo> = Vec::new();
        let mut ctx = new_ctx();

        pil_code_gen(&mut ctx, &symbols, &expressions, 2, 0).unwrap();
        let block = build_code(&mut ctx).unwrap();

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

        pil_code_gen(&mut ctx, &symbols, &expressions, 2, 0).unwrap();
        let block = build_code(&mut ctx).unwrap();

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

        pil_code_gen(&mut ctx, &symbols, &expressions, 5, 0).unwrap();
        let block = build_code(&mut ctx).unwrap();

        // Should have code for the inner add + the outer mul
        assert!(block.code.len() >= 2);
    }

    #[test]
    fn test_build_code_resets_ctx() {
        let expressions = vec![make_number("1"), make_cm(0), make_add(0, 1)];
        let symbols: Vec<SymbolInfo> = Vec::new();
        let mut ctx = new_ctx();

        pil_code_gen(&mut ctx, &symbols, &expressions, 2, 0).unwrap();
        let _ = build_code(&mut ctx).unwrap();

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

        pil_code_gen(&mut ctx, &symbols, &expressions, 0, 0).unwrap();
        let block = build_code(&mut ctx).unwrap();

        assert!(!block.code.is_empty());
        // The last entry should be a copy with source being a number
        let last = block.code.last().unwrap();
        assert_eq!(last.op, "copy");
        assert_eq!(last.src[0].ref_type, "number");
        assert_eq!(last.src[0].value.as_deref(), Some("42"));
    }

    /// Was `panic!("Invalid op")`.
    #[test]
    fn code_of_an_unknown_op_is_an_error() {
        let exprs = vec![Expression { op: "bogus".to_string(), ..Default::default() }];
        let err = pil_code_gen(&mut new_ctx(), &[], &exprs, 0, 0).unwrap_err();
        assert_eq!(err.to_string(), "code generation: unknown expression op `bogus`");
    }
}
