use crate::expr::expression::{ExprChild, Expression};
use crate::types::pilout_info::FIELD_EXTENSION;

/// Goldilocks field negation of 1: p - 1 = 2^64 - 2^32 + 1 - 1 = 18446744069414584320
const NEG_ONE: &str = "18446744069414584320";

// ---------------------------------------------------------------------------
// add_info_expressions
// ---------------------------------------------------------------------------

/// Recursively walk expression trees computing `exp_deg`, `dim`, `stage`,
/// and `rows_offsets`. Mirrors the JS `addInfoExpressions` from helpers.js.
///
/// The `expressions` slice must be the flat expression array (indexed by ID).
/// This function mutates expressions in-place.
///
/// For arena-indexed expressions, call with the arena index. For expressions
/// with inline children, the inline children are processed recursively.
pub fn add_info_expressions(expressions: &mut Vec<Expression>, idx: usize) {
    // Guard: already processed (mirrors JS `if("expDeg" in exp) return;`)
    if expressions[idx].info_computed {
        return;
    }

    let op = expressions[idx].op.clone();

    match op.as_str() {
        "exp" => {
            let ref_id = expressions[idx].id.unwrap_or(0);
            add_info_expressions(expressions, ref_id);

            let exp_deg = expressions[ref_id].exp_deg;
            let rows_offsets = expressions[ref_id].rows_offsets.clone();
            let ref_dim = expressions[ref_id].dim;
            let ref_stage = expressions[ref_id].stage;

            expressions[idx].exp_deg = exp_deg;
            expressions[idx].rows_offsets = rows_offsets;
            if expressions[idx].dim <= 1 {
                expressions[idx].dim = ref_dim;
            }
            if expressions[idx].stage == 0 {
                expressions[idx].stage = ref_stage;
            }
        }
        "cm" | "custom" | "const" => {
            expressions[idx].exp_deg = 1;
            if expressions[idx].stage == 0 || op == "const" {
                expressions[idx].stage = if op == "cm" { 1 } else { 0 };
            }
            if expressions[idx].dim == 0 || expressions[idx].dim == 1 {
                expressions[idx].dim = 1;
            }
            if let Some(row_off) = expressions[idx].row_offset {
                expressions[idx].rows_offsets = vec![row_off];
            }
        }
        "Zi" => {
            let boundary = expressions[idx].boundary.clone();
            if boundary.as_deref() != Some("everyRow") {
                expressions[idx].exp_deg = 1;
            } else {
                // everyRow Zi
                expressions[idx].exp_deg = 0;
                expressions[idx].stage = 0;
                if expressions[idx].dim <= 1 {
                    expressions[idx].dim = 1;
                }
            }
        }
        "xDivXSubXi" => {
            expressions[idx].exp_deg = 1;
        }
        "challenge" | "eval" => {
            expressions[idx].exp_deg = 0;
            expressions[idx].dim = FIELD_EXTENSION;
        }
        "airgroupvalue" | "proofvalue" => {
            expressions[idx].exp_deg = 0;
            if expressions[idx].dim <= 1 {
                expressions[idx].dim = if expressions[idx].stage != 1 { FIELD_EXTENSION } else { 1 };
            }
        }
        "airvalue" => {
            expressions[idx].exp_deg = 0;
            if expressions[idx].dim <= 1 {
                expressions[idx].dim = if expressions[idx].stage != 1 { FIELD_EXTENSION } else { 1 };
            }
        }
        "public" => {
            expressions[idx].exp_deg = 0;
            expressions[idx].stage = 1;
            if expressions[idx].dim <= 1 {
                expressions[idx].dim = 1;
            }
        }
        "number" => {
            expressions[idx].exp_deg = 0;
            expressions[idx].stage = 0;
            if expressions[idx].dim <= 1 {
                expressions[idx].dim = 1;
            }
        }
        "add" | "sub" | "mul" | "neg" => {
            handle_binary_or_neg(expressions, idx);
        }
        _ => {
            panic!("Exp op not defined: {}", op);
        }
    }
    expressions[idx].info_computed = true;
}

/// Compute info (exp_deg, dim, stage, rows_offsets) on a standalone
/// Expression that is NOT in the arena. Inline children are processed
/// recursively; `exp` references are resolved via the arena.
pub fn add_info_expression_inline(expressions: &mut Vec<Expression>, expr: &mut Expression) {
    if expr.info_computed {
        return;
    }

    let op = expr.op.clone();

    match op.as_str() {
        "exp" => {
            let ref_id = expr.id.unwrap_or(0);
            add_info_expressions(expressions, ref_id);

            expr.exp_deg = expressions[ref_id].exp_deg;
            expr.rows_offsets = expressions[ref_id].rows_offsets.clone();
            if expr.dim <= 1 {
                expr.dim = expressions[ref_id].dim;
            }
            if expr.stage == 0 {
                expr.stage = expressions[ref_id].stage;
            }
        }
        "cm" | "custom" | "const" => {
            expr.exp_deg = 1;
            if expr.stage == 0 || op == "const" {
                expr.stage = if op == "cm" { 1 } else { 0 };
            }
            if expr.dim == 0 || expr.dim == 1 {
                expr.dim = 1;
            }
            if let Some(row_off) = expr.row_offset {
                expr.rows_offsets = vec![row_off];
            }
        }
        "Zi" => {
            if expr.boundary.as_deref() != Some("everyRow") {
                expr.exp_deg = 1;
            } else {
                expr.exp_deg = 0;
                expr.stage = 0;
                if expr.dim <= 1 {
                    expr.dim = 1;
                }
            }
        }
        "xDivXSubXi" => {
            expr.exp_deg = 1;
        }
        "challenge" | "eval" => {
            expr.exp_deg = 0;
            expr.dim = FIELD_EXTENSION;
        }
        "airgroupvalue" | "proofvalue" => {
            expr.exp_deg = 0;
            if expr.dim <= 1 {
                expr.dim = if expr.stage != 1 { FIELD_EXTENSION } else { 1 };
            }
        }
        "airvalue" => {
            expr.exp_deg = 0;
            if expr.dim <= 1 {
                expr.dim = if expr.stage != 1 { FIELD_EXTENSION } else { 1 };
            }
        }
        "public" => {
            expr.exp_deg = 0;
            expr.stage = 1;
            if expr.dim <= 1 {
                expr.dim = 1;
            }
        }
        "number" => {
            expr.exp_deg = 0;
            expr.stage = 0;
            if expr.dim <= 1 {
                expr.dim = 1;
            }
        }
        "add" | "sub" | "mul" | "neg" => {
            handle_binary_or_neg_inline(expressions, expr);
        }
        _ => {
            panic!("Exp op not defined: {}", op);
        }
    }
    expr.info_computed = true;
}

/// Handle add/sub/mul/neg operations in add_info_expressions.
/// Mirrors the JS logic including neg->mul transformation and
/// zero-constant optimizations.
fn handle_binary_or_neg(expressions: &mut Vec<Expression>, idx: usize) {
    let op = expressions[idx].op.clone();

    // neg -> mul by NEG_ONE  (inline, NOT pushed to arena — matches JS behaviour)
    if op == "neg" {
        expressions[idx].op = "mul".to_string();
        let neg_one = Expression {
            op: "number".to_string(),
            value: Some(NEG_ONE.to_string()),
            exp_deg: 0,
            stage: 0,
            dim: 1,
            ..Default::default()
        };
        let existing_values = expressions[idx].values.clone();
        expressions[idx].values = vec![ExprChild::Inline(Box::new(neg_one)), existing_values[0].clone()];
    }

    let current_op = expressions[idx].op.clone();

    // For add where LHS is number(0), convert to mul with LHS=1
    if current_op == "add" {
        let lhs_is_zero = {
            let lhs = expressions[idx].values[0].resolve(expressions);
            lhs.op == "number" && lhs.value.as_deref() == Some("0")
        };
        if lhs_is_zero {
            expressions[idx].op = "mul".to_string();
            // Extract child kind first to avoid overlapping borrows
            let child = expressions[idx].values[0].clone();
            match child {
                ExprChild::Id(id) => {
                    expressions[id].value = Some("1".to_string());
                }
                ExprChild::Inline(mut expr) => {
                    expr.value = Some("1".to_string());
                    expressions[idx].values[0] = ExprChild::Inline(expr);
                }
            }
        }
    }

    // For add/sub where RHS is number(0), convert to mul with RHS=1
    let current_op = expressions[idx].op.clone();
    if current_op == "add" || current_op == "sub" {
        let rhs_is_zero = {
            let rhs = expressions[idx].values[1].resolve(expressions);
            rhs.op == "number" && rhs.value.as_deref() == Some("0")
        };
        if rhs_is_zero {
            expressions[idx].op = "mul".to_string();
            let child = expressions[idx].values[1].clone();
            match child {
                ExprChild::Id(id) => {
                    expressions[id].value = Some("1".to_string());
                }
                ExprChild::Inline(mut expr) => {
                    expr.value = Some("1".to_string());
                    expressions[idx].values[1] = ExprChild::Inline(expr);
                }
            }
        }
    }

    // Process children: for Id children, recurse via arena; for Inline, recurse inline
    for i in 0..expressions[idx].values.len() {
        match expressions[idx].values[i].clone() {
            ExprChild::Id(child_id) => {
                add_info_expressions(expressions, child_id);
            }
            ExprChild::Inline(mut child) => {
                add_info_expression_inline(expressions, &mut child);
                expressions[idx].values[i] = ExprChild::Inline(child);
            }
        }
    }

    // Read child info
    let (lhs_deg, lhs_dim, lhs_stage, lhs_offsets) = {
        let lhs = expressions[idx].values[0].resolve(expressions);
        (lhs.exp_deg, lhs.dim, lhs.stage, if lhs.rows_offsets.is_empty() { vec![0] } else { lhs.rows_offsets.clone() })
    };
    let (rhs_deg, rhs_dim, rhs_stage, rhs_offsets) = {
        let rhs = expressions[idx].values[1].resolve(expressions);
        (rhs.exp_deg, rhs.dim, rhs.stage, if rhs.rows_offsets.is_empty() { vec![0] } else { rhs.rows_offsets.clone() })
    };

    let final_op = expressions[idx].op.clone();
    expressions[idx].exp_deg = if final_op == "mul" { lhs_deg + rhs_deg } else { lhs_deg.max(rhs_deg) };

    expressions[idx].dim = lhs_dim.max(rhs_dim);
    expressions[idx].stage = lhs_stage.max(rhs_stage);

    // Merge rows_offsets
    let mut merged: Vec<i64> = lhs_offsets;
    for o in rhs_offsets {
        if !merged.contains(&o) {
            merged.push(o);
        }
    }
    expressions[idx].rows_offsets = merged;
}

/// Handle add/sub/mul/neg for inline expressions (not in the arena).
fn handle_binary_or_neg_inline(expressions: &mut Vec<Expression>, expr: &mut Expression) {
    let op = expr.op.clone();

    // neg -> mul by NEG_ONE
    if op == "neg" {
        expr.op = "mul".to_string();
        let neg_one = Expression {
            op: "number".to_string(),
            value: Some(NEG_ONE.to_string()),
            exp_deg: 0,
            stage: 0,
            dim: 1,
            ..Default::default()
        };
        let existing_values = expr.values.clone();
        expr.values = vec![ExprChild::Inline(Box::new(neg_one)), existing_values[0].clone()];
    }

    let current_op = expr.op.clone();
    if current_op == "add" {
        let lhs_is_zero = {
            let lhs = expr.values[0].resolve(expressions);
            lhs.op == "number" && lhs.value.as_deref() == Some("0")
        };
        if lhs_is_zero {
            expr.op = "mul".to_string();
            match &mut expr.values[0] {
                ExprChild::Id(id) => {
                    expressions[*id].value = Some("1".to_string());
                }
                ExprChild::Inline(e) => {
                    e.value = Some("1".to_string());
                }
            }
        }
    }

    let current_op = expr.op.clone();
    if current_op == "add" || current_op == "sub" {
        let rhs_is_zero = {
            let rhs = expr.values[1].resolve(expressions);
            rhs.op == "number" && rhs.value.as_deref() == Some("0")
        };
        if rhs_is_zero {
            expr.op = "mul".to_string();
            match &mut expr.values[1] {
                ExprChild::Id(id) => {
                    expressions[*id].value = Some("1".to_string());
                }
                ExprChild::Inline(e) => {
                    e.value = Some("1".to_string());
                }
            }
        }
    }

    // Process children
    for i in 0..expr.values.len() {
        match expr.values[i].clone() {
            ExprChild::Id(child_id) => {
                add_info_expressions(expressions, child_id);
            }
            ExprChild::Inline(mut child) => {
                add_info_expression_inline(expressions, &mut child);
                expr.values[i] = ExprChild::Inline(child);
            }
        }
    }

    // Read child info
    let (lhs_deg, lhs_dim, lhs_stage, lhs_offsets) = {
        let lhs = expr.values[0].resolve(expressions);
        (lhs.exp_deg, lhs.dim, lhs.stage, if lhs.rows_offsets.is_empty() { vec![0] } else { lhs.rows_offsets.clone() })
    };
    let (rhs_deg, rhs_dim, rhs_stage, rhs_offsets) = {
        let rhs = expr.values[1].resolve(expressions);
        (rhs.exp_deg, rhs.dim, rhs.stage, if rhs.rows_offsets.is_empty() { vec![0] } else { rhs.rows_offsets.clone() })
    };

    let final_op = expr.op.clone();
    expr.exp_deg = if final_op == "mul" { lhs_deg + rhs_deg } else { lhs_deg.max(rhs_deg) };

    expr.dim = lhs_dim.max(rhs_dim);
    expr.stage = lhs_stage.max(rhs_stage);

    let mut merged: Vec<i64> = lhs_offsets;
    for o in rhs_offsets {
        if !merged.contains(&o) {
            merged.push(o);
        }
    }
    expr.rows_offsets = merged;
}

// ---------------------------------------------------------------------------
// get_exp_dim
// ---------------------------------------------------------------------------

/// Get the field dimension of an expression, mirroring JS `getExpDim`.
pub fn get_exp_dim(expressions: &[Expression], exp_id: usize) -> usize {
    get_exp_dim_inner(expressions, &expressions[exp_id])
}

fn get_exp_dim_inner(expressions: &[Expression], exp: &Expression) -> usize {
    if exp.dim > 0 && exp.op != "add" && exp.op != "sub" && exp.op != "mul" {
        return exp.dim;
    }

    match exp.op.as_str() {
        "add" | "sub" | "mul" => {
            let mut max_dim = 0;
            for child in &exp.values {
                let child_expr = child.resolve(expressions);
                let child_dim = get_exp_dim_inner(expressions, child_expr);
                if child_dim > max_dim {
                    max_dim = child_dim;
                }
            }
            max_dim
        }
        "exp" => {
            let id = exp.id.unwrap_or(0);
            get_exp_dim_inner(expressions, &expressions[id])
        }
        "cm" | "custom" => {
            if exp.dim > 0 {
                exp.dim
            } else {
                1
            }
        }
        "const" | "number" | "public" | "Zi" => 1,
        "challenge" | "eval" | "xDivXSubXi" => FIELD_EXTENSION,
        _ => panic!("Exp op not defined: {}", exp.op),
    }
}

/// Track which columns (cm, const, custom) are referenced by an expression tree.
/// Mirrors JS `addInfoExpressionsSymbols`.
///
/// `ev_map` accumulates unique (type, id, prime, commitId) entries.
#[derive(Debug, Clone, PartialEq)]
pub struct EvMapItem {
    pub entry_type: String,
    pub id: usize,
    pub prime: i64,
    pub commit_id: Option<usize>,
}

pub fn add_info_expressions_symbols(
    ev_map: &mut Vec<EvMapItem>,
    expressions: &[Expression],
    idx: usize,
    explored: &mut Vec<bool>,
) {
    if explored[idx] {
        return;
    }

    let exp = &expressions[idx];
    let op = exp.op.clone();

    match op.as_str() {
        "exp" => {
            let ref_id = exp.id.unwrap_or(0);
            add_info_expressions_symbols(ev_map, expressions, ref_id, explored);
            explored[idx] = true;
        }
        "cm" | "const" | "custom" => {
            let id = exp.id.unwrap_or(0);
            let prime = exp.row_offset.unwrap_or(0);
            let commit_id = exp.commit_id;

            let new_item = EvMapItem { entry_type: op.clone(), id, prime, commit_id };

            let already_present = ev_map.iter().any(|item| {
                item.entry_type == new_item.entry_type
                    && item.id == new_item.id
                    && item.prime == new_item.prime
                    && (new_item.commit_id.is_none() || item.commit_id == new_item.commit_id)
            });

            if !already_present {
                ev_map.push(new_item);
            }
        }
        "add" | "sub" | "mul" | "neg" => {
            // Collect child arena IDs; for inline children, recurse into them
            let values = expressions[idx].values.clone();
            for child in &values {
                match child {
                    ExprChild::Id(child_id) => {
                        add_info_expressions_symbols(ev_map, expressions, *child_id, explored);
                    }
                    ExprChild::Inline(child_expr) => {
                        add_info_expressions_symbols_inline(ev_map, expressions, child_expr, explored);
                    }
                }
            }
            explored[idx] = true;
        }
        _ => {}
    }
}

/// Walk inline expression trees for symbol collection.
/// Shares the `explored` vec from the parent to avoid re-exploring arena expressions.
fn add_info_expressions_symbols_inline(
    ev_map: &mut Vec<EvMapItem>,
    expressions: &[Expression],
    expr: &Expression,
    explored: &mut Vec<bool>,
) {
    match expr.op.as_str() {
        "exp" => {
            let ref_id = expr.id.unwrap_or(0);
            add_info_expressions_symbols(ev_map, expressions, ref_id, explored);
        }
        "cm" | "const" | "custom" => {
            let id = expr.id.unwrap_or(0);
            let prime = expr.row_offset.unwrap_or(0);
            let commit_id = expr.commit_id;

            let new_item = EvMapItem { entry_type: expr.op.clone(), id, prime, commit_id };

            let already_present = ev_map.iter().any(|item| {
                item.entry_type == new_item.entry_type
                    && item.id == new_item.id
                    && item.prime == new_item.prime
                    && (new_item.commit_id.is_none() || item.commit_id == new_item.commit_id)
            });

            if !already_present {
                ev_map.push(new_item);
            }
        }
        "add" | "sub" | "mul" | "neg" => {
            for child in &expr.values {
                match child {
                    ExprChild::Id(child_id) => {
                        add_info_expressions_symbols(ev_map, expressions, *child_id, explored);
                    }
                    ExprChild::Inline(child_expr) => {
                        add_info_expressions_symbols_inline(ev_map, expressions, child_expr, explored);
                    }
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_number(val: &str) -> Expression {
        Expression { op: "number".to_string(), value: Some(val.to_string()), ..Default::default() }
    }

    fn make_cm(id: usize, stage: usize) -> Expression {
        Expression { op: "cm".to_string(), id: Some(id), stage, dim: 1, row_offset: Some(0), ..Default::default() }
    }

    #[test]
    fn test_add_info_number() {
        let mut exprs = vec![make_number("42")];
        add_info_expressions(&mut exprs, 0);
        assert_eq!(exprs[0].exp_deg, 0);
        assert_eq!(exprs[0].dim, 1);
        assert_eq!(exprs[0].stage, 0);
    }

    #[test]
    fn test_add_info_cm() {
        let mut exprs = vec![make_cm(0, 1)];
        add_info_expressions(&mut exprs, 0);
        assert_eq!(exprs[0].exp_deg, 1);
        assert_eq!(exprs[0].dim, 1);
        assert_eq!(exprs[0].stage, 1);
    }

    #[test]
    fn test_add_info_mul() {
        // mul(cm, cm) => exp_deg = 2
        let cm1 = make_cm(0, 1);
        let cm2 = make_cm(1, 1);
        let mul_expr = Expression {
            op: "mul".to_string(),
            values: vec![ExprChild::Id(0), ExprChild::Id(1)],
            ..Default::default()
        };
        let mut exprs = vec![cm1, cm2, mul_expr];
        add_info_expressions(&mut exprs, 2);
        assert_eq!(exprs[2].exp_deg, 2);
        assert_eq!(exprs[2].dim, 1);
        assert_eq!(exprs[2].stage, 1);
    }

    #[test]
    fn test_get_exp_dim_number() {
        let exprs =
            vec![Expression { op: "number".to_string(), value: Some("1".to_string()), dim: 1, ..Default::default() }];
        assert_eq!(get_exp_dim(&exprs, 0), 1);
    }

    #[test]
    fn test_get_exp_dim_challenge() {
        let exprs = vec![Expression { op: "challenge".to_string(), dim: FIELD_EXTENSION, ..Default::default() }];
        assert_eq!(get_exp_dim(&exprs, 0), FIELD_EXTENSION);
    }

    #[test]
    fn test_ev_map_dedup() {
        let exprs = vec![
            make_cm(0, 1),
            make_cm(0, 1),
            Expression {
                op: "add".to_string(),
                values: vec![ExprChild::Id(0), ExprChild::Id(1)],
                ..Default::default()
            },
        ];
        let mut ev_map = Vec::new();
        let mut explored = vec![false; exprs.len()];
        add_info_expressions_symbols(&mut ev_map, &exprs, 2, &mut explored);
        // Both children reference cm id=0 prime=0, so dedup should give 1 entry
        assert_eq!(ev_map.len(), 1);
    }
}
