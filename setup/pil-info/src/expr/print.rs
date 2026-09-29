//! Pretty-printer for expression trees.
//!
//! The printer converts an expression tree into a human-readable string using
//! symbol names, operator notation, etc.  Results are cached on `Expression.line`
//! ONLY for `exp` type nodes encountered during recursion (matching the JS
//! pattern where only `exp.op === "exp"` nodes get `exp.line` cached).

use crate::expr::expression::Expression;
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

/// Pretty-print an expression tree and cache the result on `expressions[exp_idx].line`.
///
/// This is the top-level entry point used by hint processing code:
///   `expressions[field.id].line = printExpressions(res, expressions[field.id], expressions)`
///
/// It prints the expression and explicitly sets `expressions[exp_idx].line`.
pub fn print_expression(ctx: &PrintCtx, expressions: &mut [Expression], exp_idx: usize, is_constraint: bool) -> String {
    let result = print_expr_inner(ctx, expressions, exp_idx, is_constraint);
    // Explicitly cache (mirrors JS: `expressions[field.id].line = printExpressions(...)`)
    expressions[exp_idx].line = Some(result.clone());
    result
}

/// Print an expression tree without caching on the top-level expression.
///
/// Used by map.rs for constraint lines and imPolsInfo where the JS code
/// does NOT set `expressions[...].line` (only the constraint.line or a
/// local variable receives the result).
pub fn print_expression_no_cache(
    ctx: &PrintCtx,
    expressions: &mut [Expression],
    exp_idx: usize,
    is_constraint: bool,
) -> String {
    print_expr_inner(ctx, expressions, exp_idx, is_constraint)
}

/// Inner recursive printer.
///
/// Caches ONLY on `exp` type nodes (matching JS where `if(exp.op === "exp")
/// { if(!exp.line) exp.line = ...; return exp.line; }`).
fn print_expr_inner(ctx: &PrintCtx, expressions: &mut [Expression], idx: usize, is_constraint: bool) -> String {
    // Read fields from expression (clone what we need to avoid borrow issues)
    let op = expressions[idx].op.clone();
    let values_snapshot: Vec<_> = expressions[idx]
        .values
        .iter()
        .map(|c| match c {
            crate::expr::expression::ExprChild::Id(id) => Some(*id),
            crate::expr::expression::ExprChild::Inline(_) => None,
        })
        .collect();

    match op.as_str() {
        "exp" => {
            // Check cache on the expression this `exp` node references
            let ref_id = expressions[idx].id.unwrap_or(0);
            if let Some(ref line) = expressions[idx].line {
                return line.clone();
            }
            // Recurse into the referenced expression
            let line = print_expr_inner(ctx, expressions, ref_id, is_constraint);
            // Cache on this exp node (matching JS: exp.line = ...)
            expressions[idx].line = Some(line.clone());
            line
        }

        "add" | "mul" | "sub" => {
            let lhs_id = values_snapshot.first().copied().flatten();
            let rhs_id = values_snapshot.get(1).copied().flatten();

            let lhs_str = if let Some(id) = lhs_id {
                print_expr_inner(ctx, expressions, id, is_constraint)
            } else {
                let inline = match &expressions[idx].values[0] {
                    crate::expr::expression::ExprChild::Inline(e) => (**e).clone(),
                    _ => unreachable!(),
                };
                print_inline_expr(ctx, &inline, expressions, is_constraint)
            };

            let rhs_str = if let Some(id) = rhs_id {
                print_expr_inner(ctx, expressions, id, is_constraint)
            } else {
                let inline = match &expressions[idx].values[1] {
                    crate::expr::expression::ExprChild::Inline(e) => (**e).clone(),
                    _ => unreachable!(),
                };
                print_inline_expr(ctx, &inline, expressions, is_constraint)
            };

            let op_str = match op.as_str() {
                "add" => " + ",
                "sub" => " - ",
                "mul" => " * ",
                _ => unreachable!(),
            };
            format!("({}{}{})", lhs_str, op_str, rhs_str)
        }

        "neg" => {
            let child_id = values_snapshot.first().copied().flatten();
            if let Some(id) = child_id {
                print_expr_inner(ctx, expressions, id, is_constraint)
            } else {
                let inline = match &expressions[idx].values[0] {
                    crate::expr::expression::ExprChild::Inline(e) => (**e).clone(),
                    _ => unreachable!(),
                };
                print_inline_expr(ctx, &inline, expressions, is_constraint)
            }
        }

        "number" => expressions[idx].value.clone().unwrap_or_else(|| "0".to_string()),

        "const" | "cm" | "custom" => print_col_ref(ctx, expressions, idx, is_constraint),

        "public" => {
            let id = expressions[idx].id.unwrap_or(0);
            if id < ctx.publics_map.len() {
                ctx.publics_map[id].name.clone()
            } else {
                format!("public_{}", id)
            }
        }

        "airvalue" => {
            let id = expressions[idx].id.unwrap_or(0);
            if id < ctx.air_values_map.len() {
                ctx.air_values_map[id].name.clone()
            } else {
                format!("airvalue_{}", id)
            }
        }

        "airgroupvalue" => {
            let id = expressions[idx].id.unwrap_or(0);
            if id < ctx.airgroup_values_map.len() {
                ctx.airgroup_values_map[id].name.clone()
            } else {
                format!("airgroupvalue_{}", id)
            }
        }

        "challenge" => {
            let id = expressions[idx].id.unwrap_or(0);
            if id < ctx.challenges_map.len() {
                ctx.challenges_map[id].name.clone()
            } else {
                format!("challenge_{}", id)
            }
        }

        "Zi" => "zh".to_string(),

        "proofvalue" => {
            let id = expressions[idx].id.unwrap_or(0);
            if id < ctx.proof_values_map.len() {
                ctx.proof_values_map[id].name.clone()
            } else {
                format!("proofvalue_{}", id)
            }
        }

        other => {
            format!("unknown_op_{}", other)
        }
    }
}

/// Print a column reference (const/cm/custom). Handles imPol expansion.
fn print_col_ref(ctx: &PrintCtx, expressions: &mut [Expression], idx: usize, is_constraint: bool) -> String {
    let op = expressions[idx].op.clone();
    let exp_id = expressions[idx].id.unwrap_or(0);
    let row_offset = expressions[idx].row_offset.unwrap_or(0);
    let commit_id = expressions[idx].commit_id;

    let col = match op.as_str() {
        "const" => {
            if exp_id < ctx.const_pols_map.len() {
                Some(ctx.const_pols_map[exp_id].clone())
            } else {
                None
            }
        }
        "cm" => {
            if exp_id < ctx.cm_pols_map.len() {
                Some(ctx.cm_pols_map[exp_id].clone())
            } else {
                None
            }
        }
        "custom" => {
            let cid = commit_id.unwrap_or(0);
            if cid < ctx.custom_commits_map.len() && exp_id < ctx.custom_commits_map[cid].len() {
                Some(ctx.custom_commits_map[cid][exp_id].clone())
            } else {
                None
            }
        }
        _ => None,
    };

    let col = match col {
        Some(c) => c,
        None => return format!("{}_{}", op, exp_id),
    };

    // If imPol and not in constraint context, expand the underlying expression
    if col.im_pol && !is_constraint {
        if let Some(col_exp_id) = col.exp_id {
            return print_expr_inner(ctx, expressions, col_exp_id, false);
        }
    }

    let mut name = col.name.clone();

    // Append array indices if lengths present
    if let Some(ref lengths) = col.lengths {
        for len in lengths {
            name.push_str(&format!("[{}]", len));
        }
    }

    // For imPol in constraint context, append the count of prior imPol entries
    if col.im_pol {
        let prior_count = ctx.cm_pols_map.iter().enumerate().filter(|(i, w)| *i < exp_id && w.im_pol).count();
        name.push_str(&prior_count.to_string());
    }

    // Append row offset notation
    if row_offset > 0 {
        name.push('\'');
        if row_offset > 1 {
            name.push_str(&row_offset.to_string());
        }
    } else if row_offset < 0 {
        let abs_offset = row_offset.abs();
        if abs_offset > 1 {
            name = format!("{}'{}", abs_offset, name);
        } else {
            name = format!("'{}", name);
        }
    }

    name
}

/// Print an inline expression (not in the arena).
fn print_inline_expr(ctx: &PrintCtx, expr: &Expression, expressions: &mut [Expression], is_constraint: bool) -> String {
    match expr.op.as_str() {
        "exp" => {
            let ref_id = expr.id.unwrap_or(0);
            print_expr_inner(ctx, expressions, ref_id, is_constraint)
        }
        "add" | "mul" | "sub" => {
            let lhs = match &expr.values[0] {
                crate::expr::expression::ExprChild::Id(id) => print_expr_inner(ctx, expressions, *id, is_constraint),
                crate::expr::expression::ExprChild::Inline(e) => print_inline_expr(ctx, e, expressions, is_constraint),
            };
            let rhs = match &expr.values[1] {
                crate::expr::expression::ExprChild::Id(id) => print_expr_inner(ctx, expressions, *id, is_constraint),
                crate::expr::expression::ExprChild::Inline(e) => print_inline_expr(ctx, e, expressions, is_constraint),
            };
            let op_str = match expr.op.as_str() {
                "add" => " + ",
                "sub" => " - ",
                "mul" => " * ",
                _ => unreachable!(),
            };
            format!("({}{}{})", lhs, op_str, rhs)
        }
        "neg" => match &expr.values[0] {
            crate::expr::expression::ExprChild::Id(id) => print_expr_inner(ctx, expressions, *id, is_constraint),
            crate::expr::expression::ExprChild::Inline(e) => print_inline_expr(ctx, e, expressions, is_constraint),
        },
        "number" => expr.value.clone().unwrap_or_else(|| "0".to_string()),
        "cm" | "const" | "custom" => {
            let id = expr.id.unwrap_or(0);
            let row_offset = expr.row_offset.unwrap_or(0);
            let commit_id = expr.commit_id;

            let col = match expr.op.as_str() {
                "const" => {
                    if id < ctx.const_pols_map.len() {
                        Some(&ctx.const_pols_map[id])
                    } else {
                        None
                    }
                }
                "cm" => {
                    if id < ctx.cm_pols_map.len() {
                        Some(&ctx.cm_pols_map[id])
                    } else {
                        None
                    }
                }
                "custom" => {
                    let cid = commit_id.unwrap_or(0);
                    if cid < ctx.custom_commits_map.len() && id < ctx.custom_commits_map[cid].len() {
                        Some(&ctx.custom_commits_map[cid][id])
                    } else {
                        None
                    }
                }
                _ => None,
            };

            let col = match col {
                Some(c) => c,
                None => return format!("{}_{}", expr.op, id),
            };

            if col.im_pol && !is_constraint {
                if let Some(col_exp_id) = col.exp_id {
                    return print_expr_inner(ctx, expressions, col_exp_id, false);
                }
            }

            let mut name = col.name.clone();
            if let Some(ref lengths) = col.lengths {
                for len in lengths {
                    name.push_str(&format!("[{}]", len));
                }
            }
            if col.im_pol {
                let prior_count = ctx.cm_pols_map.iter().enumerate().filter(|(i, w)| *i < id && w.im_pol).count();
                name.push_str(&prior_count.to_string());
            }
            if row_offset > 0 {
                name.push('\'');
                if row_offset > 1 {
                    name.push_str(&row_offset.to_string());
                }
            } else if row_offset < 0 {
                let abs_offset = row_offset.abs();
                if abs_offset > 1 {
                    name = format!("{}'{}", abs_offset, name);
                } else {
                    name = format!("'{}", name);
                }
            }
            name
        }
        "challenge" => {
            let id = expr.id.unwrap_or(0);
            if id < ctx.challenges_map.len() {
                ctx.challenges_map[id].name.clone()
            } else {
                format!("challenge_{}", id)
            }
        }
        "public" => {
            let id = expr.id.unwrap_or(0);
            if id < ctx.publics_map.len() {
                ctx.publics_map[id].name.clone()
            } else {
                format!("public_{}", id)
            }
        }
        "airvalue" => {
            let id = expr.id.unwrap_or(0);
            if id < ctx.air_values_map.len() {
                ctx.air_values_map[id].name.clone()
            } else {
                format!("airvalue_{}", id)
            }
        }
        "airgroupvalue" => {
            let id = expr.id.unwrap_or(0);
            if id < ctx.airgroup_values_map.len() {
                ctx.airgroup_values_map[id].name.clone()
            } else {
                format!("airgroupvalue_{}", id)
            }
        }
        "proofvalue" => {
            let id = expr.id.unwrap_or(0);
            if id < ctx.proof_values_map.len() {
                ctx.proof_values_map[id].name.clone()
            } else {
                format!("proofvalue_{}", id)
            }
        }
        "Zi" => "zh".to_string(),
        _ => format!("unknown_{}", expr.op),
    }
}
