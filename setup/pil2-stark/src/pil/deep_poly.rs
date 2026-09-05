use std::collections::BTreeMap;

use crate::expr::expression::{ExprChild, Expression};
use crate::expr::helpers::EvMapItem;
use crate::types::pilout_info::{SymbolInfo, FIELD_EXTENSION};

/// Result of DEEP polynomial generation.
#[derive(Debug, Clone)]
pub struct DeepPolyResult {
    /// Expression index of the DEEP polynomial in the expression arena.
    pub deep_exp_id: usize,
}

/// Generate the DEEP polynomial expression — the initial polynomial f₀ the
/// low-degree test (FRI or STIR) is run on.
///
/// In the JS version, all intermediate nodes are built inline and only ONE
/// `expressions.push(friExp)` happens at the end. We match that by building
/// the entire FRI tree with inline `ExprChild::Inline` children and pushing
/// only the final composite expression.
pub fn generate_deep_polynomial(
    n_stages: usize,
    expressions: &mut Vec<Expression>,
    symbols: &mut Vec<SymbolInfo>,
    ev_map: &[EvMapItem],
    opening_points: &[i64],
    challenges_map: &mut Vec<ChallengeMapEntry>,
) -> DeepPolyResult {
    let stage = n_stages + 3;

    // Create std_vf1 challenge
    let vf1_id = symbols.iter().filter(|s| s.sym_type == "challenge" && s.stage.is_some_and(|st| st < stage)).count();
    let vf2_id = vf1_id + 1;

    let vf1_symbol = SymbolInfo {
        sym_type: "challenge".to_string(),
        name: "std_vf1".to_string(),
        stage: Some(stage),
        dim: FIELD_EXTENSION,
        stage_id: Some(0),
        id: Some(vf1_id),
        pol_id: None,
        air_id: None,
        airgroup_id: None,
        commit_id: None,
        lengths: None,
        idx: None,
        stage_pos: None,
        im_pol: false,
        exp_id: None,
    };
    let vf2_symbol = SymbolInfo {
        sym_type: "challenge".to_string(),
        name: "std_vf2".to_string(),
        stage: Some(stage),
        dim: FIELD_EXTENSION,
        stage_id: Some(1),
        id: Some(vf2_id),
        pol_id: None,
        air_id: None,
        airgroup_id: None,
        commit_id: None,
        lengths: None,
        idx: None,
        stage_pos: None,
        im_pol: false,
        exp_id: None,
    };

    symbols.push(vf1_symbol.clone());
    symbols.push(vf2_symbol.clone());

    // Update challenges map
    extend_challenges_map(challenges_map, vf1_id, &vf1_symbol);
    extend_challenges_map(challenges_map, vf2_id, &vf2_symbol);

    // Build vf1 and vf2 as inline expression nodes (NOT pushed to arena)
    let vf1_expr = Expression {
        op: "challenge".to_string(),
        value: Some("std_vf1".to_string()),
        stage,
        dim: FIELD_EXTENSION,
        stage_id: Some(0),
        id: Some(vf1_id),
        ..Default::default()
    };

    let vf2_expr = Expression {
        op: "challenge".to_string(),
        value: Some("std_vf2".to_string()),
        stage,
        dim: FIELD_EXTENSION,
        stage_id: Some(1),
        id: Some(vf2_id),
        ..Default::default()
    };

    // Build per-opening-point FRI sub-expressions as inline trees
    let mut fri_exps: BTreeMap<i64, Expression> = BTreeMap::new();

    for (i, ev) in ev_map.iter().enumerate() {
        let symbol = find_symbol_for_ev(symbols, ev);
        let col_expr = build_column_expr(ev, &symbol);

        let eval_expr = Expression { op: "eval".to_string(), id: Some(i), dim: FIELD_EXTENSION, ..Default::default() };

        // sub(column, eval(i))
        let sub_expr = Expression {
            op: "sub".to_string(),
            values: vec![ExprChild::Inline(Box::new(col_expr)), ExprChild::Inline(Box::new(eval_expr))],
            ..Default::default()
        };

        if let Some(existing) = fri_exps.remove(&ev.prime) {
            // acc = add(mul(existing, vf2), sub)
            let mul_expr = Expression {
                op: "mul".to_string(),
                values: vec![ExprChild::Inline(Box::new(existing)), ExprChild::Inline(Box::new(vf2_expr.clone()))],
                ..Default::default()
            };

            let add_expr = Expression {
                op: "add".to_string(),
                values: vec![ExprChild::Inline(Box::new(mul_expr)), ExprChild::Inline(Box::new(sub_expr))],
                ..Default::default()
            };

            fri_exps.insert(ev.prime, add_expr);
        } else {
            fri_exps.insert(ev.prime, sub_expr);
        }
    }

    // Combine per-opening-point expressions with xDivXSubXi and vf1
    let mut fri_exp: Option<Expression> = None;

    for (i, &opening) in opening_points.iter().enumerate() {
        // Opening points from kept/hint expressions may not have ev_map
        // entries (and thus no fri_exps entry). Skip those; they are only
        // needed by prover code, not the FRI verifier polynomial.
        let opening_expr = match fri_exps.remove(&opening) {
            Some(e) => e,
            None => continue,
        };

        let xdiv_expr =
            Expression { op: "xDivXSubXi".to_string(), opening: Some(opening), id: Some(i), ..Default::default() };

        // mul(fri_exps[opening], xDivXSubXi)
        let mul_expr = Expression {
            op: "mul".to_string(),
            values: vec![ExprChild::Inline(Box::new(opening_expr)), ExprChild::Inline(Box::new(xdiv_expr))],
            ..Default::default()
        };

        if let Some(prev_fri) = fri_exp.take() {
            // add(mul(vf1, prev_fri), current)
            let vf1_mul = Expression {
                op: "mul".to_string(),
                values: vec![ExprChild::Inline(Box::new(vf1_expr.clone())), ExprChild::Inline(Box::new(prev_fri))],
                ..Default::default()
            };

            let add_expr = Expression {
                op: "add".to_string(),
                values: vec![ExprChild::Inline(Box::new(vf1_mul)), ExprChild::Inline(Box::new(mul_expr))],
                ..Default::default()
            };
            fri_exp = Some(add_expr);
        } else {
            fri_exp = Some(mul_expr);
        }
    }

    // Push only the final FRI expression (matches JS: one expressions.push)
    let mut fri_final = fri_exp.expect("At least one opening point required");
    let fri_final_id = expressions.len();

    // Set dim and stage on the final expression
    fri_final.dim = get_exp_dim_inline(expressions, &fri_final);
    fri_final.stage = n_stages + 2;

    expressions.push(fri_final);

    DeepPolyResult { deep_exp_id: fri_final_id }
}

/// Get dimension for an inline expression tree.
fn get_exp_dim_inline(expressions: &[Expression], exp: &Expression) -> usize {
    if exp.dim > 0 && exp.op != "add" && exp.op != "sub" && exp.op != "mul" {
        return exp.dim;
    }
    match exp.op.as_str() {
        "add" | "sub" | "mul" => {
            let mut max_dim = 0;
            for child in &exp.values {
                let child_expr = child.resolve(expressions);
                let d = get_exp_dim_inline(expressions, child_expr);
                if d > max_dim {
                    max_dim = d;
                }
            }
            max_dim
        }
        "exp" => {
            let id = exp.id.unwrap_or(0);
            get_exp_dim_inline(expressions, &expressions[id])
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

/// Entry in the challenges map (matches JSON output format).
#[derive(Debug, Clone)]
pub struct ChallengeMapEntry {
    pub name: String,
    pub stage: usize,
    pub dim: usize,
    pub stage_id: usize,
}

/// Extend the challenges_map to accommodate an entry at a given index.
fn extend_challenges_map(challenges_map: &mut Vec<ChallengeMapEntry>, id: usize, symbol: &SymbolInfo) {
    while challenges_map.len() <= id {
        challenges_map.push(ChallengeMapEntry { name: String::new(), stage: 0, dim: 0, stage_id: 0 });
    }
    challenges_map[id] = ChallengeMapEntry {
        name: symbol.name.clone(),
        stage: symbol.stage.unwrap_or(0),
        dim: symbol.dim,
        stage_id: symbol.stage_id.unwrap_or(0),
    };
}

/// Find the symbol matching an evaluation map entry.
fn find_symbol_for_ev(symbols: &[SymbolInfo], ev: &EvMapItem) -> SymbolInfo {
    let sym_type_target = match ev.entry_type.as_str() {
        "const" => "fixed",
        "cm" => "witness",
        "custom" => "custom",
        other => panic!("Unknown ev type: {}", other),
    };

    symbols
        .iter()
        .find(|s| {
            s.pol_id == Some(ev.id)
                && s.sym_type == sym_type_target
                && (ev.entry_type != "custom" || s.commit_id == ev.commit_id)
        })
        .unwrap_or_else(|| panic!("Symbol not found for ev type={} id={}", ev.entry_type, ev.id))
        .clone()
}

/// Build a column expression node for an evaluation entry.
fn build_column_expr(ev: &EvMapItem, symbol: &SymbolInfo) -> Expression {
    Expression {
        op: ev.entry_type.clone(),
        id: Some(ev.id),
        row_offset: Some(0),
        stage: symbol.stage.unwrap_or(0),
        dim: symbol.dim,
        commit_id: symbol.commit_id,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_witness_symbol(name: &str, pol_id: usize, stage: usize) -> SymbolInfo {
        SymbolInfo {
            name: name.to_string(),
            sym_type: "witness".to_string(),
            stage: Some(stage),
            dim: 1,
            pol_id: Some(pol_id),
            stage_id: Some(0),
            id: Some(pol_id),
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        }
    }

    fn make_fixed_symbol(name: &str, pol_id: usize) -> SymbolInfo {
        SymbolInfo {
            name: name.to_string(),
            sym_type: "fixed".to_string(),
            stage: Some(0),
            dim: 1,
            pol_id: Some(pol_id),
            stage_id: Some(0),
            id: Some(pol_id),
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        }
    }

    #[test]
    fn test_generate_fri_polynomial_basic() {
        let mut expressions: Vec<Expression> = Vec::new();
        let mut symbols = vec![make_witness_symbol("w0", 0, 1), make_fixed_symbol("f0", 0)];
        let ev_map = vec![
            EvMapItem { entry_type: "cm".to_string(), id: 0, prime: 0, commit_id: None },
            EvMapItem { entry_type: "const".to_string(), id: 0, prime: 0, commit_id: None },
        ];
        let opening_points = vec![0];
        let mut challenges_map = Vec::new();

        let result =
            generate_deep_polynomial(1, &mut expressions, &mut symbols, &ev_map, &opening_points, &mut challenges_map);

        // deep_exp_id is the index in the expressions arena; when the arena
        // starts empty (as in this test) the first pushed expression gets id 0.
        assert_eq!(result.deep_exp_id, 0);
        assert!(!expressions.is_empty());

        // Should have added std_vf1 and std_vf2
        let challenge_names: Vec<&str> =
            symbols.iter().filter(|s| s.sym_type == "challenge").map(|s| s.name.as_str()).collect();
        assert!(challenge_names.contains(&"std_vf1"));
        assert!(challenge_names.contains(&"std_vf2"));
        assert_eq!(challenges_map.len(), challenge_names.len());
    }

    #[test]
    fn test_generate_fri_polynomial_multiple_openings() {
        let mut expressions: Vec<Expression> = Vec::new();
        let mut symbols = vec![make_witness_symbol("w0", 0, 1), make_witness_symbol("w1", 1, 1)];
        let ev_map = vec![
            EvMapItem { entry_type: "cm".to_string(), id: 0, prime: 0, commit_id: None },
            EvMapItem { entry_type: "cm".to_string(), id: 1, prime: 1, commit_id: None },
        ];
        let opening_points = vec![0, 1];
        let mut challenges_map = Vec::new();

        let result =
            generate_deep_polynomial(1, &mut expressions, &mut symbols, &ev_map, &opening_points, &mut challenges_map);

        // deep_exp_id is a valid index into the expressions arena
        assert!(result.deep_exp_id < expressions.len());
        assert!(!expressions.is_empty());
    }
}
