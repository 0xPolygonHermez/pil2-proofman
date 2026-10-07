use std::collections::{BTreeMap, HashMap};

use crate::expr::expression::{ExprChild, Expression};
use crate::expr::helpers::EvMapItem;
use crate::types::pilout_info::{SymbolInfo, FIELD_EXTENSION};

/// Result of FRI polynomial generation.
#[derive(Debug, Clone)]
pub struct FriPolyResult {
    /// Expression index of the FRI polynomial in the expression arena.
    pub fri_exp_id: usize,
    /// Power-batching size matching the FRI polynomial's degree in (vf1, vf2).
    pub batch_size: usize,
}

/// Generate the FRI polynomial expression.
///
/// In the JS version, all intermediate nodes are built inline and only ONE
/// `expressions.push(friExp)` happens at the end. We match that by building
/// the entire FRI tree with inline `ExprChild::Inline` children and pushing
/// only the final composite expression.
pub fn generate_fri_polynomial(
    n_stages: usize,
    expressions: &mut Vec<Expression>,
    symbols: &mut Vec<SymbolInfo>,
    ev_map: &[EvMapItem],
    opening_points: &[i64],
    challenges_map: &mut Vec<ChallengeMapEntry>,
) -> FriPolyResult {
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

    // fri = SUM_G (SUM_{c in G} b_c p_c) (SUM_{o in G} A_o) - SUM_o A_o K_o, with G the polynomials sharing one
    // set of openings, A_o = vf1^e_o xDivXSubXi_o, b_c = vf2^c and K_o = SUM_{c in o} b_c e_(o,c): the coefficient
    // of (o, c) is the distinct monomial vf1^e_o vf2^c, c = first appearance in ev_map. b_c, vf1^e_o and K_o only
    // depend on challenges and evals, so a verifier computes them once for all queries.
    let mut col_index: HashMap<(String, Option<usize>, usize), usize> = HashMap::new();
    let mut col_evals: Vec<Vec<(i64, usize)>> = Vec::new(); // (opening, ev index)
    for (i, ev) in ev_map.iter().enumerate() {
        let c = *col_index.entry((ev.entry_type.clone(), ev.commit_id, ev.id)).or_insert_with(|| {
            col_evals.push(Vec::new());
            col_evals.len() - 1
        });
        assert!(
            col_evals[c].iter().all(|&(o, _)| o != ev.prime),
            "FRI: {} {} opened twice at {}",
            ev.entry_type,
            ev.id,
            ev.prime
        );
        col_evals[c].push((ev.prime, i));
    }

    let present: Vec<(usize, i64)> =
        opening_points.iter().copied().enumerate().filter(|&(_, op)| ev_map.iter().any(|ev| ev.prime == op)).collect();
    assert!(!present.is_empty(), "At least one opening point required");
    let n_open = present.len();
    // Degree in (vf1, vf2), for the security calculator.
    let batch_size = ev_map.len().max(n_open + col_evals.len() - 1);

    let op = |name: &str, a: Expression, b: Expression| Expression {
        op: name.to_string(),
        values: vec![ExprChild::Inline(Box::new(a)), ExprChild::Inline(Box::new(b))],
        ..Default::default()
    };
    let one = Expression { op: "number".to_string(), value: Some("1".to_string()), dim: 1, ..Default::default() };
    // Shared through the arena so it is computed once. Leaves stay inline: a shared leaf would become a copy,
    // which the packed evaluators do not implement.
    let mut share = |e: Expression| {
        if !matches!(e.op.as_str(), "add" | "sub" | "mul") {
            return e;
        }
        expressions.push(Expression { dim: FIELD_EXTENSION, stage: n_stages + 2, ..e });
        Expression {
            op: "exp".to_string(),
            id: Some(expressions.len() - 1),
            dim: FIELD_EXTENSION,
            ..Default::default()
        }
    };
    let mul = |a: &Expression, b: Expression| if a.op == "number" { b } else { op("mul", a.clone(), b) };
    let sum = |terms: Vec<Expression>| terms.into_iter().reduce(|acc, t| op("add", acc, t)).expect("non-empty sum");

    let mut vf2_pow = vec![one.clone()];
    for c in 1..col_evals.len() {
        vf2_pow.push(if c == 1 {
            vf2_expr.clone()
        } else {
            share(op("mul", vf2_pow[c - 1].clone(), vf2_expr.clone()))
        });
    }
    let mut vf1_pow = vec![one.clone()];
    for k in 1..n_open {
        vf1_pow.push(if k == 1 {
            vf1_expr.clone()
        } else {
            share(op("mul", vf1_pow[k - 1].clone(), vf1_expr.clone()))
        });
    }
    let rank: HashMap<i64, usize> = present.iter().enumerate().map(|(r, &(_, opening))| (opening, r)).collect();
    let a_of: Vec<Expression> = present
        .iter()
        .enumerate()
        .map(|(r, &(i, opening))| {
            let xdiv =
                Expression { op: "xDivXSubXi".to_string(), opening: Some(opening), id: Some(i), ..Default::default() };
            share(mul(&vf1_pow[n_open - 1 - r], xdiv))
        })
        .collect();

    let mut k_terms: Vec<Vec<Expression>> = vec![Vec::new(); n_open];
    let mut groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for (c, evals) in col_evals.iter().enumerate() {
        for &(opening, i) in evals {
            let eval = Expression { op: "eval".to_string(), id: Some(i), dim: FIELD_EXTENSION, ..Default::default() };
            k_terms[rank[&opening]].push(mul(&vf2_pow[c], eval));
        }
        let mut ops: Vec<usize> = evals.iter().map(|&(opening, _)| rank[&opening]).collect();
        ops.sort_unstable();
        groups.entry(ops).or_default().push(c);
    }

    let mut terms = Vec::new();
    for (ops, cols) in &groups {
        let s = sum(cols
            .iter()
            .map(|&c| {
                let ev = &ev_map[col_evals[c][0].1];
                mul(&vf2_pow[c], build_column_expr(ev, &find_symbol_for_ev(symbols, ev)))
            })
            .collect());
        terms.push(op("mul", s, sum(ops.iter().map(|&r| a_of[r].clone()).collect())));
    }
    let mut fri_exp = sum(terms);
    for (r, k) in k_terms.into_iter().enumerate() {
        fri_exp = op("sub", fri_exp, op("mul", a_of[r].clone(), share(sum(k))));
    }

    // Push only the final FRI expression (matches JS: one expressions.push)
    let mut fri_final = fri_exp;
    let fri_final_id = expressions.len();

    // Set dim and stage on the final expression
    fri_final.dim = get_exp_dim_inline(expressions, &fri_final);
    fri_final.stage = n_stages + 2;

    expressions.push(fri_final);

    FriPolyResult { fri_exp_id: fri_final_id, batch_size }
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
            generate_fri_polynomial(1, &mut expressions, &mut symbols, &ev_map, &opening_points, &mut challenges_map);

        assert_eq!(result.fri_exp_id, expressions.len() - 1);
        assert_eq!(result.batch_size, 2);
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
            generate_fri_polynomial(1, &mut expressions, &mut symbols, &ev_map, &opening_points, &mut challenges_map);

        // fri_exp_id is a valid index into the expressions arena
        assert!(result.fri_exp_id < expressions.len());
        assert!(!expressions.is_empty());
        // Disjoint openings: degree 1 + 1 exceeds the 2-entry ev_map's power-batching degree.
        assert_eq!(result.batch_size, 3);
    }

    #[test]
    #[should_panic(expected = "opened twice")]
    fn test_generate_fri_polynomial_rejects_duplicate_evaluation() {
        let mut expressions: Vec<Expression> = Vec::new();
        let mut symbols = vec![make_witness_symbol("w0", 0, 1)];
        let ev_map = vec![
            EvMapItem { entry_type: "cm".to_string(), id: 0, prime: 0, commit_id: None },
            EvMapItem { entry_type: "cm".to_string(), id: 0, prime: 0, commit_id: None },
        ];
        let mut challenges_map = Vec::new();
        generate_fri_polynomial(1, &mut expressions, &mut symbols, &ev_map, &[0], &mut challenges_map);
    }
}
