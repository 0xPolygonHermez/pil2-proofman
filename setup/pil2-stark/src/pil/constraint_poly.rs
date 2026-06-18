use crate::expr::expression::{ExprChild, Expression};
use crate::expr::helpers::{add_info_expression_inline, get_exp_dim};
use crate::types::pilout_info::{ConstraintInfo, SymbolInfo, FIELD_EXTENSION};

/// Result of constraint polynomial generation, attached to the setup result.
#[derive(Debug, Clone)]
pub struct ConstraintPolyResult {
    /// Expression index of the accumulated constraint polynomial.
    pub c_exp_id: usize,
    /// Field dimension of the constraint polynomial.
    pub q_dim: usize,
    /// Maximum constraint degree (before intermediate polynomial optimization).
    pub initial_q_degree: i64,
}

/// Boundary entry for the result.
#[derive(Debug, Clone)]
pub struct Boundary {
    pub name: String,
    pub offset_min: Option<u32>,
    pub offset_max: Option<u32>,
}

/// Generate the constraint polynomial by combining all constraints with a
/// random linear combination challenge (`std_vc`).
///
/// In JS, helper nodes (challenge, exp refs, zi) are inline objects, and only
/// composite expressions are pushed to the expressions array. We match that
/// by using `ExprChild::Inline` for helper nodes and only pushing the final
/// composite mul/add expressions.
pub fn generate_constraint_polynomial(
    n_stages: usize,
    expressions: &mut Vec<Expression>,
    symbols: &mut Vec<SymbolInfo>,
    constraints: &[ConstraintInfo],
    boundaries: &mut Vec<Boundary>,
) -> ConstraintPolyResult {
    let dim = FIELD_EXTENSION;
    let stage = n_stages + 1;

    // Create std_vc challenge
    let vc_id = symbols.iter().filter(|s| s.sym_type == "challenge" && s.stage.is_some_and(|st| st < stage)).count();

    symbols.push(SymbolInfo {
        sym_type: "challenge".to_string(),
        name: "std_vc".to_string(),
        stage: Some(stage),
        dim,
        stage_id: Some(0),
        id: Some(vc_id),
        pol_id: None,
        air_id: None,
        airgroup_id: None,
        commit_id: None,
        lengths: None,
        idx: None,
        stage_pos: None,
        im_pol: false,
        exp_id: None,
    });

    // Build the vc expression node INLINE (not pushed to arena, matching JS)
    let vc_expr = Expression {
        op: "challenge".to_string(),
        stage,
        dim,
        stage_id: Some(0),
        id: Some(vc_id),
        value: Some("std_vc".to_string()),
        exp_deg: 0,
        ..Default::default()
    };

    let mut c_exp_id: Option<usize> = None;

    for (i, constraint) in constraints.iter().enumerate() {
        let boundary = &constraint.boundary;
        if !["everyRow", "firstRow", "lastRow", "everyFrame"].contains(&boundary.as_str()) {
            panic!("Boundary {} not supported", boundary);
        }

        // Build inline expression reference to the constraint's expression
        let e = Expression {
            op: "exp".to_string(),
            id: Some(constraint.e),
            row_offset: Some(0),
            stage,
            ..Default::default()
        };

        let constraint_id;

        if boundary == "everyFrame" {
            let boundary_id =
                find_or_add_boundary_every_frame(boundaries, constraint.offset_min, constraint.offset_max);
            let zi = Expression {
                op: "Zi".to_string(),
                boundary_id: Some(boundary_id),
                boundary: Some("everyFrame".to_string()),
                ..Default::default()
            };

            // mul(e, zi) - one push, inline children (matches JS)
            let mul_expr = Expression {
                op: "mul".to_string(),
                values: vec![ExprChild::Inline(Box::new(e)), ExprChild::Inline(Box::new(zi))],
                ..Default::default()
            };
            expressions.push(mul_expr);
            constraint_id = expressions.len() - 1;
        } else if boundary != "everyRow" {
            let boundary_id = find_or_add_boundary(boundaries, boundary);
            let zi = Expression {
                op: "Zi".to_string(),
                boundary_id: Some(boundary_id),
                boundary: Some(boundary.clone()),
                ..Default::default()
            };

            // mul(e, zi) - one push, inline children (matches JS)
            let mul_expr = Expression {
                op: "mul".to_string(),
                values: vec![ExprChild::Inline(Box::new(e)), ExprChild::Inline(Box::new(zi))],
                ..Default::default()
            };
            expressions.push(mul_expr);
            constraint_id = expressions.len() - 1;
        } else {
            // everyRow: use the original constraint expression directly
            constraint_id = constraint.e;
        }

        if i == 0 {
            c_exp_id = Some(constraint_id);
        } else {
            let prev_c_exp_id = c_exp_id.unwrap();

            // weightedConstraint = mul(vc, exp(prev_c_exp_id))
            // All children are inline, ONE push (matches JS)
            let prev_exp_ref = Expression {
                op: "exp".to_string(),
                id: Some(prev_c_exp_id),
                row_offset: Some(0),
                stage,
                ..Default::default()
            };
            let mut weighted = Expression {
                op: "mul".to_string(),
                values: vec![ExprChild::Inline(Box::new(vc_expr.clone())), ExprChild::Inline(Box::new(prev_exp_ref))],
                ..Default::default()
            };
            add_info_expression_inline(expressions, &mut weighted);
            expressions.push(weighted);
            let weighted_id = expressions.len() - 1;

            // accumulatedConstraints = add(exp(weighted_id), exp(constraint_id))
            // All children are inline, ONE push (matches JS)
            let weighted_ref = Expression {
                op: "exp".to_string(),
                id: Some(weighted_id),
                row_offset: Some(0),
                stage,
                ..Default::default()
            };
            let constraint_ref = Expression {
                op: "exp".to_string(),
                id: Some(constraint_id),
                row_offset: Some(0),
                stage,
                ..Default::default()
            };
            let mut accumulated = Expression {
                op: "add".to_string(),
                values: vec![ExprChild::Inline(Box::new(weighted_ref)), ExprChild::Inline(Box::new(constraint_ref))],
                ..Default::default()
            };
            add_info_expression_inline(expressions, &mut accumulated);
            expressions.push(accumulated);
            let accumulated_id = expressions.len() - 1;

            c_exp_id = Some(accumulated_id);
        }
    }

    let c_exp_id = match c_exp_id {
        Some(id) => id,
        None => {
            // Zero-constraint AIRs (e.g., VirtualTable lookup tables) have no
            // constraint polynomial. Return a dummy expression with dim=1.
            let dummy =
                Expression { op: "number".to_string(), value: Some("0".to_string()), dim: 1, ..Default::default() };
            let c_exp_id = expressions.len();
            expressions.push(dummy);
            return ConstraintPolyResult { c_exp_id, q_dim: 1, initial_q_degree: 0 };
        }
    };
    let q_dim = get_exp_dim(expressions, c_exp_id);

    // Create std_xi challenge for evaluation
    let xi_id =
        symbols.iter().filter(|s| s.sym_type == "challenge" && s.stage.is_some_and(|st| st < stage + 1)).count();

    symbols.push(SymbolInfo {
        sym_type: "challenge".to_string(),
        name: "std_xi".to_string(),
        stage: Some(stage + 1),
        dim: FIELD_EXTENSION,
        stage_id: Some(0),
        id: Some(xi_id),
        pol_id: None,
        air_id: None,
        airgroup_id: None,
        commit_id: None,
        lengths: None,
        idx: None,
        stage_pos: None,
        im_pol: false,
        exp_id: None,
    });

    // Use pre-computed exp_deg when available (set by add_info_expressions).
    // Fall back to local calculation for tests where add_info_expressions hasn't run.
    let initial_q_degree = if expressions[c_exp_id].exp_deg > 0 {
        expressions[c_exp_id].exp_deg
    } else {
        calculate_exp_deg(expressions, c_exp_id, &[], true)
    };

    tracing::info!("The maximum constraint degree is {} (without intermediate polynomials)", initial_q_degree);

    ConstraintPolyResult { c_exp_id, q_dim, initial_q_degree }
}

/// Calculate the degree of an expression tree.
/// Mirrors `calculateExpDeg` from `imPolynomials.js`.
pub fn calculate_exp_deg(expressions: &[Expression], idx: usize, im_exps: &[usize], _cache_values: bool) -> i64 {
    let exp = &expressions[idx];
    match exp.op.as_str() {
        "exp" => {
            let ref_id = exp.id.unwrap_or(0);
            if im_exps.contains(&ref_id) {
                return 1;
            }
            calculate_exp_deg(expressions, ref_id, im_exps, _cache_values)
        }
        "const" | "cm" | "custom" => 1,
        "Zi" => {
            if exp.boundary.as_deref() == Some("everyRow") {
                0
            } else {
                1
            }
        }
        "number" | "public" | "challenge" | "eval" | "airgroupvalue" | "airvalue" | "proofvalue" => 0,
        "neg" => {
            let child = exp.values[0].resolve(expressions);
            match &exp.values[0] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, _cache_values),
                ExprChild::Inline(_) => calculate_exp_deg_inline(expressions, child, im_exps, _cache_values),
            }
        }
        "add" | "sub" | "mul" => {
            let lhs_deg = match &exp.values[0] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, _cache_values),
                ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, _cache_values),
            };
            let rhs_deg = match &exp.values[1] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, _cache_values),
                ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, _cache_values),
            };
            if exp.op == "mul" {
                lhs_deg + rhs_deg
            } else {
                lhs_deg.max(rhs_deg)
            }
        }
        _ => panic!("Exp op not defined: {}", exp.op),
    }
}

/// Calculate degree for an inline (non-arena) expression.
fn calculate_exp_deg_inline(
    expressions: &[Expression],
    exp: &Expression,
    im_exps: &[usize],
    cache_values: bool,
) -> i64 {
    match exp.op.as_str() {
        "exp" => {
            let ref_id = exp.id.unwrap_or(0);
            if im_exps.contains(&ref_id) {
                return 1;
            }
            calculate_exp_deg(expressions, ref_id, im_exps, cache_values)
        }
        "const" | "cm" | "custom" => 1,
        "Zi" => {
            if exp.boundary.as_deref() == Some("everyRow") {
                0
            } else {
                1
            }
        }
        "number" | "public" | "challenge" | "eval" | "airgroupvalue" | "airvalue" | "proofvalue" => 0,
        "neg" => match &exp.values[0] {
            ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, cache_values),
            ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, cache_values),
        },
        "add" | "sub" | "mul" => {
            let lhs_deg = match &exp.values[0] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, cache_values),
                ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, cache_values),
            };
            let rhs_deg = match &exp.values[1] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, cache_values),
                ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, cache_values),
            };
            if exp.op == "mul" {
                lhs_deg + rhs_deg
            } else {
                lhs_deg.max(rhs_deg)
            }
        }
        _ => panic!("Exp op not defined: {}", exp.op),
    }
}

/// Find or add a boundary by name. Returns the index.
fn find_or_add_boundary(boundaries: &mut Vec<Boundary>, name: &str) -> usize {
    if let Some(idx) = boundaries.iter().position(|b| b.name == name) {
        return idx;
    }
    boundaries.push(Boundary { name: name.to_string(), offset_min: None, offset_max: None });
    boundaries.len() - 1
}

/// Find or add an everyFrame boundary with matching offset_min/offset_max. Returns the index.
fn find_or_add_boundary_every_frame(
    boundaries: &mut Vec<Boundary>,
    offset_min: Option<u32>,
    offset_max: Option<u32>,
) -> usize {
    if let Some(idx) = boundaries
        .iter()
        .position(|b| b.name == "everyFrame" && b.offset_min == offset_min && b.offset_max == offset_max)
    {
        return idx;
    }
    boundaries.push(Boundary { name: "everyFrame".to_string(), offset_min, offset_max });
    boundaries.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_cm(id: usize, stage: usize) -> Expression {
        Expression { op: "cm".to_string(), id: Some(id), stage, dim: 1, row_offset: Some(0), ..Default::default() }
    }

    fn make_mul(lhs: usize, rhs: usize) -> Expression {
        Expression { op: "mul".to_string(), values: vec![ExprChild::Id(lhs), ExprChild::Id(rhs)], ..Default::default() }
    }

    #[test]
    fn test_calculate_exp_deg_leaf() {
        let exprs = vec![Expression { op: "number".to_string(), value: Some("42".to_string()), ..Default::default() }];
        assert_eq!(calculate_exp_deg(&exprs, 0, &[], false), 0);
    }

    #[test]
    fn test_calculate_exp_deg_cm() {
        let exprs = vec![make_cm(0, 1)];
        assert_eq!(calculate_exp_deg(&exprs, 0, &[], false), 1);
    }

    #[test]
    fn test_calculate_exp_deg_mul() {
        // mul(cm0, cm1) -> degree 2
        let exprs = vec![make_cm(0, 1), make_cm(1, 1), make_mul(0, 1)];
        assert_eq!(calculate_exp_deg(&exprs, 2, &[], false), 2);
    }

    #[test]
    fn test_generate_constraint_polynomial_single() {
        // Single everyRow constraint: c_exp_id should point to the exp ref
        let cm = make_cm(0, 1);
        let mut expressions = vec![cm]; // expression 0 is a cm

        let mut symbols = Vec::new();
        let constraints = vec![ConstraintInfo {
            boundary: "everyRow".to_string(),
            e: 0,
            line: None,
            offset_min: None,
            offset_max: None,
            stage: None,
            im_pol: false,
        }];
        let mut boundaries = vec![Boundary { name: "everyRow".to_string(), offset_min: None, offset_max: None }];

        let result = generate_constraint_polynomial(1, &mut expressions, &mut symbols, &constraints, &mut boundaries);

        // With one everyRow constraint, c_exp_id is the original expression
        assert_eq!(result.c_exp_id, 0);
        assert_eq!(result.q_dim, 1);
        assert_eq!(result.initial_q_degree, 1);

        // Should have added std_vc and std_xi
        let challenge_names: Vec<&str> =
            symbols.iter().filter(|s| s.sym_type == "challenge").map(|s| s.name.as_str()).collect();
        assert!(challenge_names.contains(&"std_vc"));
        assert!(challenge_names.contains(&"std_xi"));
    }

    #[test]
    fn test_generate_constraint_polynomial_two() {
        // Two everyRow constraints
        let cm0 = make_cm(0, 1);
        let cm1 = make_cm(1, 1);
        let mut expressions = vec![cm0, cm1];

        let mut symbols = Vec::new();
        let constraints = vec![
            ConstraintInfo {
                boundary: "everyRow".to_string(),
                e: 0,
                line: None,
                offset_min: None,
                offset_max: None,
                stage: None,
                im_pol: false,
            },
            ConstraintInfo {
                boundary: "everyRow".to_string(),
                e: 1,
                line: None,
                offset_min: None,
                offset_max: None,
                stage: None,
                im_pol: false,
            },
        ];
        let mut boundaries = vec![Boundary { name: "everyRow".to_string(), offset_min: None, offset_max: None }];

        let result = generate_constraint_polynomial(1, &mut expressions, &mut symbols, &constraints, &mut boundaries);

        // With two constraints: 2 original + 1 weighted + 1 accumulated = 4 total
        assert_eq!(result.c_exp_id, 3);
        assert!(result.q_dim >= 1);
    }

    #[test]
    fn test_boundary_firstrow() {
        let cm0 = make_cm(0, 1);
        let mut expressions = vec![cm0];
        let mut symbols = Vec::new();
        let constraints = vec![ConstraintInfo {
            boundary: "firstRow".to_string(),
            e: 0,
            line: None,
            offset_min: None,
            offset_max: None,
            stage: None,
            im_pol: false,
        }];
        let mut boundaries = vec![Boundary { name: "everyRow".to_string(), offset_min: None, offset_max: None }];

        let result = generate_constraint_polynomial(1, &mut expressions, &mut symbols, &constraints, &mut boundaries);

        // Should have added "firstRow" boundary
        assert!(boundaries.iter().any(|b| b.name == "firstRow"));
        // Degree should be 2 (cm * Zi)
        assert_eq!(result.initial_q_degree, 2);
    }
}
