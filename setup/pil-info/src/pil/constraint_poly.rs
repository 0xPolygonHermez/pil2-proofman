use std::collections::HashMap;

use crate::cfg::FieldCfg;
use crate::error::{PilInfoError, Result};
use crate::expr::expression::{mul_deg, ExprChild, Expression};
use crate::expr::helpers::{add_info_expression_inline, add_info_expressions, get_exp_dim};
use crate::cfg::Opening;
pub use crate::pil::domain::Boundary;
use crate::pil::domain::{normalize, Domain, MAX_DOMAIN_FACTORS};
use crate::types::pilout_info::{ConstraintInfo, SymbolInfo};

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

/// Normalizes each constraint's domain (`name_of(i)` names it); the STARK caps its factors, pilfflonk refuses it.
pub fn normalize_constraint_domains(
    constraints: &mut [ConstraintInfo],
    n_bits: u32,
    opening: Opening,
    name_of: impl Fn(usize) -> String,
) -> Result<()> {
    // constraints share few domains: normalize each distinct one once
    let mut normalized: HashMap<Domain, Boundary> = HashMap::new();
    for (i, c) in constraints.iter_mut().enumerate() {
        let Boundary::Domain(declared) = &c.boundary else { continue };
        let b = match normalized.get(declared) {
            Some(b) => b.clone(),
            None => {
                let name = name_of(i);
                let b = normalize(declared, n_bits, &name)?;
                if let Boundary::Domain(d) = &b {
                    match opening {
                        Opening::Deep if d.offsets.len() > MAX_DOMAIN_FACTORS => {
                            return Err(PilInfoError::InvalidDomain(format!(
                                "{name}: the domain needs {} vanishing factors, above the limit of {MAX_DOMAIN_FACTORS}; use a fixed selector column",
                                d.offsets.len()
                            )));
                        }
                        Opening::Deep => {}
                        Opening::Shplonk => return Err(PilInfoError::DomainConstraintsUnsupported),
                    }
                }
                normalized.insert(declared.clone(), b.clone());
                b
            }
        };
        c.boundary = b;
    }
    Ok(())
}

/// Generate the constraint polynomial by combining all constraints with a
/// random linear combination challenge (`std_vc`).
///
/// In JS, helper nodes (challenge, exp refs, zi) are inline objects, and only
/// composite expressions are pushed to the expressions array. We match that
/// by using `ExprChild::Inline` for helper nodes and only pushing the final
/// composite mul/add expressions.
#[allow(clippy::too_many_arguments)]
pub fn generate_constraint_polynomial(
    n_stages: usize,
    n_bits: u32,
    expressions: &mut Vec<Expression>,
    symbols: &mut Vec<SymbolInfo>,
    constraints: &[ConstraintInfo],
    boundaries: &mut Vec<Boundary>,
    field: &FieldCfg,
) -> Result<ConstraintPolyResult> {
    let dim = field.ext_dim();
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

    for constraint in constraints {
        // normalized by `normalize_constraint_domains`
        let boundary = &constraint.boundary;

        let constraint_id = if *boundary == Boundary::EveryRow {
            constraint.e
        } else {
            // Build inline expression reference to the constraint's expression
            let e = Expression {
                op: "exp".to_string(),
                id: Some(constraint.e),
                row_offset: Some(0),
                stage,
                ..Default::default()
            };
            let zi = Expression {
                op: "Zi".to_string(),
                boundary_id: Some(find_or_add_boundary(boundaries, boundary)),
                boundary: Some(boundary.name().to_string()),
                zi_domain: match boundary {
                    Boundary::Domain(d) => Some((d.clone(), n_bits)),
                    Boundary::EveryRow => None,
                },
                ..Default::default()
            };

            // mul(e, zi) - one push, inline children (matches JS)
            let mul_expr = Expression {
                op: "mul".to_string(),
                values: vec![ExprChild::Inline(Box::new(e)), ExprChild::Inline(Box::new(zi))],
                ..Default::default()
            };
            expressions.push(mul_expr);
            let id = expressions.len() - 1;
            // A lone constraint is never reached through the folding below, so its degree is set here.
            add_info_expressions(expressions, id, field)?;
            id
        };

        if let Some(prev_c_exp_id) = c_exp_id {
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
            add_info_expression_inline(expressions, &mut weighted, field)?;
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
            add_info_expression_inline(expressions, &mut accumulated, field)?;
            expressions.push(accumulated);
            let accumulated_id = expressions.len() - 1;

            c_exp_id = Some(accumulated_id);
        } else {
            c_exp_id = Some(constraint_id);
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
            return Ok(ConstraintPolyResult { c_exp_id, q_dim: 1, initial_q_degree: 0 });
        }
    };
    let q_dim = get_exp_dim(expressions, c_exp_id, field)?;

    // Create std_xi challenge for evaluation
    let xi_id =
        symbols.iter().filter(|s| s.sym_type == "challenge" && s.stage.is_some_and(|st| st < stage + 1)).count();

    symbols.push(SymbolInfo {
        sym_type: "challenge".to_string(),
        name: "std_xi".to_string(),
        stage: Some(stage + 1),
        dim: field.ext_dim(),
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
        calculate_exp_deg(expressions, c_exp_id, &[], true)?
    };

    tracing::info!("The maximum constraint degree is {} (without intermediate polynomials)", initial_q_degree);

    Ok(ConstraintPolyResult { c_exp_id, q_dim, initial_q_degree })
}

/// Calculate the degree of an expression tree.
/// Mirrors `calculateExpDeg` from `imPolynomials.js`. Fails on an op it does not know.
pub fn calculate_exp_deg(
    expressions: &[Expression],
    idx: usize,
    im_exps: &[usize],
    _cache_values: bool,
) -> Result<i64> {
    let exp = &expressions[idx];
    let deg = match exp.op.as_str() {
        "exp" => {
            let ref_id = exp.id.unwrap_or(0);
            if im_exps.contains(&ref_id) {
                return Ok(1);
            }
            calculate_exp_deg(expressions, ref_id, im_exps, _cache_values)?
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
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, _cache_values)?,
                ExprChild::Inline(_) => calculate_exp_deg_inline(expressions, child, im_exps, _cache_values)?,
            }
        }
        "add" | "sub" | "mul" => {
            let lhs_deg = match &exp.values[0] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, _cache_values)?,
                ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, _cache_values)?,
            };
            let rhs_deg = match &exp.values[1] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, _cache_values)?,
                ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, _cache_values)?,
            };
            if exp.op == "mul" {
                mul_deg(lhs_deg, exp.values[1].resolve(expressions), rhs_deg)
            } else {
                lhs_deg.max(rhs_deg)
            }
        }
        _ => return Err(PilInfoError::UnknownOp { pass: "calculate_exp_deg", op: exp.op.clone() }),
    };
    Ok(deg)
}

/// Calculate degree for an inline (non-arena) expression.
fn calculate_exp_deg_inline(
    expressions: &[Expression],
    exp: &Expression,
    im_exps: &[usize],
    cache_values: bool,
) -> Result<i64> {
    let deg = match exp.op.as_str() {
        "exp" => {
            let ref_id = exp.id.unwrap_or(0);
            if im_exps.contains(&ref_id) {
                return Ok(1);
            }
            calculate_exp_deg(expressions, ref_id, im_exps, cache_values)?
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
            ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, cache_values)?,
            ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, cache_values)?,
        },
        "add" | "sub" | "mul" => {
            let lhs_deg = match &exp.values[0] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, cache_values)?,
                ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, cache_values)?,
            };
            let rhs_deg = match &exp.values[1] {
                ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, cache_values)?,
                ExprChild::Inline(e) => calculate_exp_deg_inline(expressions, e, im_exps, cache_values)?,
            };
            if exp.op == "mul" {
                mul_deg(lhs_deg, exp.values[1].resolve(expressions), rhs_deg)
            } else {
                lhs_deg.max(rhs_deg)
            }
        }
        _ => return Err(PilInfoError::UnknownOp { pass: "calculate_exp_deg", op: exp.op.clone() }),
    };
    Ok(deg)
}

/// Index of `boundary` in `boundaries`, appending it if new.
fn find_or_add_boundary(boundaries: &mut Vec<Boundary>, boundary: &Boundary) -> usize {
    boundaries.iter().position(|b| b == boundary).unwrap_or_else(|| {
        boundaries.push(boundary.clone());
        boundaries.len() - 1
    })
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
    fn only_the_stark_accepts_domain_constraints() {
        let domain = |offsets: Vec<u32>| {
            vec![ConstraintInfo {
                boundary: Boundary::Domain(Domain { cycle_bits: 3, offsets, complement: false }),
                e: 0,
                line: None,
                stage: None,
                im_pol: false,
            }]
        };
        let mut cs = domain(vec![0]);
        normalize_constraint_domains(&mut cs, 3, Opening::Deep, |_| "T".into()).unwrap();
        let err = normalize_constraint_domains(&mut domain(vec![0]), 3, Opening::Shplonk, |_| "T".into());
        assert_eq!(err.unwrap_err().to_string(), "domain constraints are not supported by pilfflonk");
        // a domain of every row is everyRow, for pilfflonk too
        let mut all = domain((0..8).collect());
        normalize_constraint_domains(&mut all, 3, Opening::Shplonk, |_| "T".into()).unwrap();
        assert_eq!(all[0].boundary, Boundary::EveryRow);
    }

    #[test]
    fn test_calculate_exp_deg_leaf() {
        let exprs = vec![Expression { op: "number".to_string(), value: Some("42".to_string()), ..Default::default() }];
        assert_eq!(calculate_exp_deg(&exprs, 0, &[], false).unwrap(), 0);
    }

    #[test]
    fn test_calculate_exp_deg_cm() {
        let exprs = vec![make_cm(0, 1)];
        assert_eq!(calculate_exp_deg(&exprs, 0, &[], false).unwrap(), 1);
    }

    #[test]
    fn test_calculate_exp_deg_mul() {
        // mul(cm0, cm1) -> degree 2
        let exprs = vec![make_cm(0, 1), make_cm(1, 1), make_mul(0, 1)];
        assert_eq!(calculate_exp_deg(&exprs, 2, &[], false).unwrap(), 2);
    }

    #[test]
    fn test_generate_constraint_polynomial_single() {
        // Single everyRow constraint: c_exp_id should point to the exp ref
        let cm = make_cm(0, 1);
        let mut expressions = vec![cm]; // expression 0 is a cm

        let mut symbols = Vec::new();
        let constraints =
            vec![ConstraintInfo { boundary: Boundary::EveryRow, e: 0, line: None, stage: None, im_pol: false }];
        let mut boundaries = vec![Boundary::EveryRow];

        let result = generate_constraint_polynomial(
            1,
            8,
            &mut expressions,
            &mut symbols,
            &constraints,
            &mut boundaries,
            &FieldCfg::goldilocks(),
        )
        .unwrap();

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
            ConstraintInfo { boundary: Boundary::EveryRow, e: 0, line: None, stage: None, im_pol: false },
            ConstraintInfo { boundary: Boundary::EveryRow, e: 1, line: None, stage: None, im_pol: false },
        ];
        let mut boundaries = vec![Boundary::EveryRow];

        let result = generate_constraint_polynomial(
            1,
            8,
            &mut expressions,
            &mut symbols,
            &constraints,
            &mut boundaries,
            &FieldCfg::goldilocks(),
        )
        .unwrap();

        // With two constraints: 2 original + 1 weighted + 1 accumulated = 4 total
        assert_eq!(result.c_exp_id, 3);
        assert!(result.q_dim >= 1);
    }

    /// The folding challenge, and so the constraint polynomial, lives in the field's extension:
    /// dimension 3 over Goldilocks, 1 over BN128.
    #[test]
    fn test_folding_follows_the_field() {
        for (field, dim) in [(FieldCfg::goldilocks(), 3), (FieldCfg::bn128(), 1)] {
            let mut expressions = vec![make_cm(0, 1), make_cm(1, 1)];
            let mut symbols = Vec::new();
            let every_row =
                |e| ConstraintInfo { boundary: Boundary::EveryRow, e, line: None, stage: None, im_pol: false };
            let constraints = vec![every_row(0), every_row(1)];
            let mut boundaries = vec![Boundary::EveryRow];

            let result = generate_constraint_polynomial(
                1,
                8,
                &mut expressions,
                &mut symbols,
                &constraints,
                &mut boundaries,
                &field,
            )
            .unwrap();

            assert_eq!(result.q_dim, dim);
            for name in ["std_vc", "std_xi"] {
                let challenge = symbols.iter().find(|s| s.name == name).expect("challenge added");
                assert_eq!(challenge.dim, dim, "{name}");
            }
        }
    }

    #[test]
    fn test_boundary_first_row_domain() {
        let cm0 = make_cm(0, 1);
        let mut expressions = vec![cm0];
        let mut symbols = Vec::new();
        let constraints = vec![ConstraintInfo {
            boundary: Boundary::Domain(Domain { cycle_bits: 8, offsets: vec![0], complement: false }),
            e: 0,
            line: None,
            stage: None,
            im_pol: false,
        }];
        let mut boundaries = vec![Boundary::EveryRow];

        let result = generate_constraint_polynomial(
            1,
            8,
            &mut expressions,
            &mut symbols,
            &constraints,
            &mut boundaries,
            &FieldCfg::goldilocks(),
        )
        .unwrap();

        let first = Domain { cycle_bits: 8, offsets: vec![0], complement: false };
        assert_eq!(boundaries, vec![Boundary::EveryRow, Boundary::Domain(first)]);
        // Degree should be 2 (cm * Zi): one row + degree 1 never exceeds N
        assert_eq!(result.initial_q_degree, 2);
    }

    #[test]
    fn a_stark_domain_with_too_many_factors_is_refused() {
        let offsets: Vec<u32> = (0..33).map(|i| i * 7).collect();
        let mut constraints = vec![ConstraintInfo {
            boundary: Boundary::Domain(Domain { cycle_bits: 8, offsets, complement: false }),
            e: 0,
            line: None,
            stage: None,
            im_pol: false,
        }];
        let e = normalize_constraint_domains(&mut constraints, 10, Opening::Deep, |_| "T".into()).unwrap_err();
        assert!(e.to_string().contains("fixed selector column"), "{e}");
    }
}
