//! Pretty-printer for expression trees.
//!
//! The printer converts an expression tree into a human-readable string using
//! symbol names, operator notation, etc.  Results are cached on `Expression.line`
//! ONLY for `exp` type nodes encountered during recursion (matching the JS
//! pattern where only `exp.op === "exp"` nodes get `exp.line` cached).

use crate::expr::expression::{ExprChild, Expression};
use crate::types::pilout_info::SymbolInfo;

/// Context for the pretty-printer, providing access to the various symbol maps.
pub struct PrintCtx<'a> {
    pub cm_pols_map: &'a [SymbolInfo],
    pub const_pols_map: &'a [SymbolInfo],
    pub custom_commits_map: &'a [Vec<SymbolInfo>],
    pub publics_map: &'a [SymbolInfo],
    pub challenges_map: &'a [SymbolInfo],
    pub air_values_map: &'a [SymbolInfo],
    pub airgroup_values_map: &'a [SymbolInfo],
    pub proof_values_map: &'a [SymbolInfo],
}

/// Prints arena entry `exp_idx` and caches the line on it (the JS hint `expressions[id].line = printExpressions(..)`).
pub fn print_expression(ctx: &PrintCtx, expressions: &mut [Expression], exp_idx: usize, is_constraint: bool) -> String {
    let result = print_expr_inner(ctx, expressions, exp_idx, is_constraint);
    expressions[exp_idx].line = Some(result.clone());
    result
}

/// Nodes a stats line may walk; a runaway walk stalls setup.
const PRINT_NODE_BUDGET: usize = 50_000;

/// Whether `expr` (an `exp` read through the arena) walks at most `budget` nodes.
fn fits_print_budget(expr: &Expression, expressions: &[Expression], budget: &mut usize) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    if expr.op == "exp" {
        return expr.id.and_then(|id| expressions.get(id)).is_none_or(|e| fits_print_budget(e, expressions, budget));
    }
    expr.values.iter().all(|c| fits_print_budget(c.resolve(expressions), expressions, budget))
}

/// Prints arena entry `exp_idx` (uncached) within `PRINT_NODE_BUDGET` nodes, else a placeholder naming `what`.
pub fn print_bounded(
    ctx: &PrintCtx,
    expressions: &mut [Expression],
    exp_idx: usize,
    is_constraint: bool,
    what: &str,
) -> String {
    let mut budget = PRINT_NODE_BUDGET;
    if fits_print_budget(&expressions[exp_idx], expressions, &mut budget) {
        print_expr_inner(ctx, expressions, exp_idx, is_constraint)
    } else {
        format!("<{what} too large to print: >{PRINT_NODE_BUDGET} nodes>")
    }
}

/// Prints arena entry `idx`; only an `exp` entry caches its line (as the JS `exp.line`).
fn print_expr_inner(ctx: &PrintCtx, expressions: &mut [Expression], idx: usize, is_constraint: bool) -> String {
    if expressions[idx].op != "exp" {
        // a copy, since printing caches lines in `expressions`
        let e = expressions[idx].clone();
        return print_node(ctx, &e, expressions, is_constraint);
    }
    if let Some(line) = &expressions[idx].line {
        return line.clone();
    }
    let line = print_expr_inner(ctx, expressions, expressions[idx].id.unwrap_or(0), is_constraint);
    expressions[idx].line = Some(line.clone());
    line
}

/// Prints `expr`, an arena entry's copy or an inline child.
fn print_node(ctx: &PrintCtx, expr: &Expression, expressions: &mut [Expression], is_constraint: bool) -> String {
    let id = expr.id.unwrap_or(0);
    let named = |map: &[SymbolInfo], what: &str| map.get(id).map_or_else(|| format!("{what}_{id}"), |s| s.name.clone());
    let child = |i: usize, expressions: &mut [Expression]| match &expr.values[i] {
        ExprChild::Id(id) => print_expr_inner(ctx, expressions, *id, is_constraint),
        ExprChild::Inline(e) => print_node(ctx, e, expressions, is_constraint),
    };
    let binary = |symbol: &str, expressions: &mut [Expression]| {
        let lhs = child(0, expressions);
        format!("({lhs}{symbol}{})", child(1, expressions))
    };
    match expr.op.as_str() {
        "exp" => print_expr_inner(ctx, expressions, id, is_constraint),
        "add" => binary(" + ", expressions),
        "sub" => binary(" - ", expressions),
        "mul" => binary(" * ", expressions),
        "neg" => child(0, expressions),
        "number" => expr.value.clone().unwrap_or_else(|| "0".to_string()),
        "const" | "cm" | "custom" => print_col_ref(ctx, expr, expressions, is_constraint),
        "public" => named(ctx.publics_map, "public"),
        "airvalue" => named(ctx.air_values_map, "airvalue"),
        "airgroupvalue" => named(ctx.airgroup_values_map, "airgroupvalue"),
        "challenge" => named(ctx.challenges_map, "challenge"),
        "proofvalue" => named(ctx.proof_values_map, "proofvalue"),
        "Zi" => "zh".to_string(),
        other => format!("unknown_op_{other}"),
    }
}

/// Prints a column reference (const/cm/custom); an im column outside a constraint prints its expression.
fn print_col_ref(ctx: &PrintCtx, expr: &Expression, expressions: &mut [Expression], is_constraint: bool) -> String {
    let id = expr.id.unwrap_or(0);
    let row_offset = expr.row_offset.unwrap_or(0);
    let col = match expr.op.as_str() {
        "const" => ctx.const_pols_map.get(id),
        "cm" => ctx.cm_pols_map.get(id),
        _ => ctx.custom_commits_map.get(expr.commit_id.unwrap_or(0)).and_then(|m| m.get(id)),
    };
    let Some(col) = col else { return format!("{}_{id}", expr.op) };
    if col.im_pol && !is_constraint {
        if let Some(col_exp_id) = col.exp_id {
            return print_expr_inner(ctx, expressions, col_exp_id, false);
        }
    }
    let mut name = col.name.clone();
    for len in col.lengths.iter().flatten() {
        name.push_str(&format!("[{len}]"));
    }
    // in a constraint, an im column is numbered by the im columns before it
    if col.im_pol {
        let prior_count = ctx.cm_pols_map.iter().take(id).filter(|w| w.im_pol).count();
        name.push_str(&prior_count.to_string());
    }
    match row_offset {
        1 => name.push('\''),
        o if o > 1 => name.push_str(&format!("'{o}")),
        -1 => name = format!("'{name}"),
        o if o < -1 => name = format!("{}'{name}", -o),
        _ => {}
    }
    name
}
