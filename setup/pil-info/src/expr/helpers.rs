use crate::cfg::FieldCfg;
use crate::error::{PilInfoError, Result};
use crate::expr::expression::{ExprChild, Expression};

// ---------------------------------------------------------------------------
// add_info_expressions
// ---------------------------------------------------------------------------

/// Computes `exp_deg`, `dim`, `stage` and `rows_offsets` of arena entry `idx` and what it reaches, in place (the JS
/// `addInfoExpressions`). Challenges, evaluations and the values past stage 1 take the field's extension dimension,
/// and a `neg` becomes a multiplication by `modulus − 1`. Fails on an op it does not know.
pub fn add_info_expressions(expressions: &mut Vec<Expression>, idx: usize, field: &FieldCfg) -> Result<()> {
    if expressions[idx].info_computed {
        return Ok(());
    }
    // the entry is processed as an inline expression; topological, so nothing it reaches reads it back
    let mut e = std::mem::take(&mut expressions[idx]);
    let r = add_info_expression_inline(expressions, &mut e, field);
    expressions[idx] = e;
    r
}

/// `add_info_expressions` of an expression outside the arena; its `exp` references are resolved in the arena.
pub fn add_info_expression_inline(
    expressions: &mut Vec<Expression>,
    expr: &mut Expression,
    field: &FieldCfg,
) -> Result<()> {
    if expr.info_computed {
        return Ok(());
    }
    let op = expr.op.clone();
    let value_dim = |e: &Expression| if e.stage != 1 { field.ext_dim() } else { 1 };
    match op.as_str() {
        "exp" => {
            let ref_id = exp_ref(expr)?;
            add_info_expressions(expressions, ref_id, field)?;
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
            if expr.dim == 0 {
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
                expr.dim = expr.dim.max(1);
            }
        }
        "challenge" | "eval" => {
            expr.exp_deg = 0;
            expr.dim = field.ext_dim();
        }
        "airgroupvalue" | "proofvalue" | "airvalue" => {
            expr.exp_deg = 0;
            if expr.dim <= 1 {
                expr.dim = value_dim(expr);
            }
        }
        "public" | "number" => {
            expr.exp_deg = 0;
            expr.stage = usize::from(op == "public");
            expr.dim = expr.dim.max(1);
        }
        "add" | "sub" | "mul" | "neg" => handle_binary_or_neg(expressions, expr, field)?,
        _ => return Err(PilInfoError::UnknownOp { pass: "add_info_expressions", op }),
    }
    expr.info_computed = true;
    Ok(())
}

/// The arena id an `exp` node reads.
fn exp_ref(expr: &Expression) -> Result<usize> {
    expr.id.ok_or_else(|| PilInfoError::InvalidPilout("an expression reference without an id".into()))
}

/// add/sub/mul/neg (the JS logic): neg becomes `mul(modulus − 1, x)`, an add/sub of a 0 operand a mul by 1.
fn handle_binary_or_neg(expressions: &mut Vec<Expression>, expr: &mut Expression, field: &FieldCfg) -> Result<()> {
    if expr.op == "neg" {
        expr.op = "mul".to_string();
        let neg_one = Expression {
            op: "number".to_string(),
            value: Some(field.neg_one().to_string()),
            dim: 1,
            ..Default::default()
        };
        expr.values.insert(0, ExprChild::Inline(Box::new(neg_one)));
        expr.values.truncate(2);
    }
    // add(0, x) and add/sub(x, 0) become mul with that operand set to 1 (the lhs first, as the JS)
    for (i, ops) in [(0, &["add"][..]), (1, &["add", "sub"][..])] {
        let zero = |c: &ExprChild| {
            let e = c.resolve(expressions);
            e.op == "number" && e.value.as_deref() == Some("0")
        };
        if ops.contains(&expr.op.as_str()) && zero(&expr.values[i]) {
            expr.op = "mul".to_string();
            match &mut expr.values[i] {
                ExprChild::Id(id) => expressions[*id].value = Some("1".to_string()),
                ExprChild::Inline(e) => e.value = Some("1".to_string()),
            }
        }
    }
    for child in expr.values.iter_mut() {
        match child {
            ExprChild::Id(id) => add_info_expressions(expressions, *id, field)?,
            ExprChild::Inline(e) => add_info_expression_inline(expressions, e, field)?,
        }
    }
    let info = |c: &ExprChild| {
        let e = c.resolve(expressions);
        let offsets = if e.rows_offsets.is_empty() { vec![0] } else { e.rows_offsets.clone() };
        (e.exp_deg, e.dim, e.stage, offsets)
    };
    let ((lhs_deg, lhs_dim, lhs_stage, mut merged), (rhs_deg, rhs_dim, rhs_stage, rhs_offsets)) =
        (info(&expr.values[0]), info(&expr.values[1]));
    expr.exp_deg = if expr.op == "mul" { lhs_deg + rhs_deg } else { lhs_deg.max(rhs_deg) };
    expr.dim = lhs_dim.max(rhs_dim);
    expr.stage = lhs_stage.max(rhs_stage);
    for o in rhs_offsets {
        if !merged.contains(&o) {
            merged.push(o);
        }
    }
    expr.rows_offsets = merged;
    Ok(())
}

// ---------------------------------------------------------------------------
// get_exp_dim
// ---------------------------------------------------------------------------

/// Get the field dimension of an expression, mirroring JS `getExpDim`. Fails on an op it does not
/// know.
pub fn get_exp_dim(expressions: &[Expression], exp_id: usize, field: &FieldCfg) -> Result<usize> {
    get_exp_dim_inner(expressions, &expressions[exp_id], field)
}

fn get_exp_dim_inner(expressions: &[Expression], exp: &Expression, field: &FieldCfg) -> Result<usize> {
    if exp.dim > 0 && exp.op != "add" && exp.op != "sub" && exp.op != "mul" {
        return Ok(exp.dim);
    }

    let dim = match exp.op.as_str() {
        "add" | "sub" | "mul" => {
            let mut max_dim = 0;
            for child in &exp.values {
                let child_expr = child.resolve(expressions);
                let child_dim = get_exp_dim_inner(expressions, child_expr, field)?;
                if child_dim > max_dim {
                    max_dim = child_dim;
                }
            }
            max_dim
        }
        "exp" => get_exp_dim_inner(expressions, &expressions[exp_ref(exp)?], field)?,
        "cm" | "custom" => {
            if exp.dim > 0 {
                exp.dim
            } else {
                1
            }
        }
        "const" | "number" | "public" | "Zi" => 1,
        "challenge" | "eval" => field.ext_dim(),
        _ => return Err(PilInfoError::UnknownOp { pass: "get_exp_dim", op: exp.op.clone() }),
    };
    Ok(dim)
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
        add_info_expressions(&mut exprs, 0, &FieldCfg::goldilocks()).unwrap();
        assert_eq!(exprs[0].exp_deg, 0);
        assert_eq!(exprs[0].dim, 1);
        assert_eq!(exprs[0].stage, 0);
    }

    #[test]
    fn test_add_info_cm() {
        let mut exprs = vec![make_cm(0, 1)];
        add_info_expressions(&mut exprs, 0, &FieldCfg::goldilocks()).unwrap();
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
        add_info_expressions(&mut exprs, 2, &FieldCfg::goldilocks()).unwrap();
        assert_eq!(exprs[2].exp_deg, 2);
        assert_eq!(exprs[2].dim, 1);
        assert_eq!(exprs[2].stage, 1);
    }

    #[test]
    fn test_get_exp_dim_number() {
        let exprs =
            vec![Expression { op: "number".to_string(), value: Some("1".to_string()), dim: 1, ..Default::default() }];
        assert_eq!(get_exp_dim(&exprs, 0, &FieldCfg::goldilocks()).unwrap(), 1);
    }

    #[test]
    fn test_get_exp_dim_challenge() {
        let exprs = vec![Expression { op: "challenge".to_string(), dim: 3, ..Default::default() }];
        assert_eq!(get_exp_dim(&exprs, 0, &FieldCfg::goldilocks()).unwrap(), 3);
    }

    /// A challenge whose dimension is not yet known takes the field's extension dimension.
    #[test]
    fn test_get_exp_dim_challenge_follows_the_field() {
        let exprs = vec![Expression { op: "challenge".to_string(), dim: 0, ..Default::default() }];
        assert_eq!(get_exp_dim(&exprs, 0, &FieldCfg::goldilocks()).unwrap(), 3);
        assert_eq!(get_exp_dim(&exprs, 0, &FieldCfg::bn128()).unwrap(), 1);
    }

    /// `challenge * cm2` (a stage-2 column): extension-valued over Goldilocks, base-valued over BN128.
    #[test]
    fn test_add_info_dims_follow_the_field() {
        for (field, dim) in [(FieldCfg::goldilocks(), 3), (FieldCfg::bn128(), 1)] {
            let mut exprs = vec![
                Expression { op: "challenge".to_string(), stage: 2, ..Default::default() },
                Expression { op: "airvalue".to_string(), stage: 2, ..Default::default() },
                Expression {
                    op: "mul".to_string(),
                    values: vec![ExprChild::Id(0), ExprChild::Id(1)],
                    ..Default::default()
                },
            ];
            add_info_expressions(&mut exprs, 2, &field).unwrap();
            assert_eq!(exprs[0].dim, dim);
            assert_eq!(exprs[1].dim, dim);
            assert_eq!(exprs[2].dim, dim);
        }
    }

    /// `neg(x)` becomes `mul(modulus − 1, x)`, in the arena and inline.
    #[test]
    fn test_neg_is_a_multiplication_by_modulus_minus_one() {
        for field in [FieldCfg::goldilocks(), FieldCfg::bn128()] {
            let mut exprs = vec![
                make_cm(0, 1),
                Expression { op: "neg".to_string(), values: vec![ExprChild::Id(0)], ..Default::default() },
            ];
            add_info_expressions(&mut exprs, 1, &field).unwrap();
            assert_eq!(exprs[1].op, "mul");
            let lhs = exprs[1].values[0].resolve(&exprs);
            assert_eq!(lhs.op, "number");
            assert_eq!(lhs.value.as_deref(), Some(field.neg_one()));

            let mut inline = Expression { op: "neg".to_string(), values: vec![ExprChild::Id(0)], ..Default::default() };
            add_info_expression_inline(&mut exprs, &mut inline, &field).unwrap();
            assert_eq!(inline.op, "mul");
            assert_eq!(inline.values[0].resolve(&exprs).value.as_deref(), Some(field.neg_one()));
        }
    }

    /// Was `panic!("Exp op not defined")`.
    #[test]
    fn an_unknown_op_is_an_error() {
        let bogus = || Expression { op: "bogus".to_string(), ..Default::default() };
        let mut exprs = vec![bogus()];
        let err = add_info_expressions(&mut exprs, 0, &FieldCfg::goldilocks()).unwrap_err();
        assert!(matches!(&err, PilInfoError::UnknownOp { pass: "add_info_expressions", op } if op == "bogus"), "{err}");
        let err = add_info_expression_inline(&mut exprs, &mut bogus(), &FieldCfg::goldilocks()).unwrap_err();
        assert_eq!(err.to_string(), "add_info_expressions: unknown expression op `bogus`");
        let err = get_exp_dim(&[Expression { dim: 0, ..bogus() }], 0, &FieldCfg::bn128()).unwrap_err();
        assert_eq!(err.to_string(), "get_exp_dim: unknown expression op `bogus`");
    }
}
