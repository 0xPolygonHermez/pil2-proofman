use std::collections::{BTreeSet, HashMap};

use tracing::info;

use crate::expr::expression::{ExprChild, Expression};
use crate::expr::helpers::{add_info_expression_inline, get_exp_dim};
use crate::types::pilout_info::{ConstraintInfo, SymbolInfo, FIELD_EXTENSION};

// ---------------------------------------------------------------------------
// calculateExpDeg
// ---------------------------------------------------------------------------

/// Calculate the polynomial degree of an expression, treating any expression
/// whose id appears in `im_exps` as degree 1 (it will become a committed
/// intermediate polynomial).
///
/// When `cache_values` is true, computed degrees are stashed in `degree_cache`
/// and reused on subsequent calls with the same expression index.
pub fn calculate_exp_deg(
    expressions: &[Expression],
    exp_idx: usize,
    im_exps: &[usize],
    cache_values: bool,
    degree_cache: &mut HashMap<usize, i64>,
) -> i64 {
    if cache_values {
        if let Some(&cached) = degree_cache.get(&exp_idx) {
            return cached;
        }
    }
    let deg = calc_deg_inner(expressions, exp_idx, im_exps, cache_values, degree_cache);
    if cache_values {
        degree_cache.insert(exp_idx, deg);
    }
    deg
}

fn calc_deg_inner(
    expressions: &[Expression],
    idx: usize,
    im_exps: &[usize],
    cache_values: bool,
    degree_cache: &mut HashMap<usize, i64>,
) -> i64 {
    let exp = &expressions[idx];
    calc_deg_expr(expressions, exp, im_exps, cache_values, degree_cache)
}

/// Calculate degree for an expression (may be arena-based or inline).
fn calc_deg_expr(
    expressions: &[Expression],
    exp: &Expression,
    im_exps: &[usize],
    cache_values: bool,
    degree_cache: &mut HashMap<usize, i64>,
) -> i64 {
    match exp.op.as_str() {
        "exp" => {
            let id = exp.id.unwrap_or(0);
            if im_exps.contains(&id) {
                return 1;
            }
            calculate_exp_deg(expressions, id, im_exps, cache_values, degree_cache)
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
        "neg" => calc_deg_child(expressions, &exp.values[0], im_exps, cache_values, degree_cache),
        "add" | "sub" => {
            let lhs = calc_deg_child(expressions, &exp.values[0], im_exps, cache_values, degree_cache);
            let rhs = calc_deg_child(expressions, &exp.values[1], im_exps, cache_values, degree_cache);
            lhs.max(rhs)
        }
        "mul" => {
            let lhs = calc_deg_child(expressions, &exp.values[0], im_exps, cache_values, degree_cache);
            let rhs = calc_deg_child(expressions, &exp.values[1], im_exps, cache_values, degree_cache);
            lhs + rhs
        }
        other => panic!("Exp op not defined: {}", other),
    }
}

/// Calculate degree for a child (either arena index or inline).
fn calc_deg_child(
    expressions: &[Expression],
    child: &ExprChild,
    im_exps: &[usize],
    cache_values: bool,
    degree_cache: &mut HashMap<usize, i64>,
) -> i64 {
    match child {
        ExprChild::Id(id) => calculate_exp_deg(expressions, *id, im_exps, cache_values, degree_cache),
        ExprChild::Inline(expr) => calc_deg_expr(expressions, expr, im_exps, cache_values, degree_cache),
    }
}

// ---------------------------------------------------------------------------
// calculateIntermediatePolynomials  (greedy search)
// ---------------------------------------------------------------------------

/// Result of the greedy intermediate polynomial search.
pub struct ImPolsResult {
    /// Expression IDs that should become intermediate polynomials.
    pub im_exps: Vec<usize>,
    /// The Q polynomial degree (cExp polynomial degree minus 1).
    pub q_deg: i64,
}

/// Greedy search over constraint degrees 2..=max_q_deg to find the split
/// that minimizes added base-field columns.
///
/// Returns the optimal `(im_exps, q_deg)`.
pub fn calculate_intermediate_polynomials(
    expressions: &[Expression],
    c_exp_id: usize,
    max_q_deg: usize,
    q_dim: usize,
) -> ImPolsResult {
    info!("-------------------- POSSIBLE DEGREES ----------------------");
    let blowup = if max_q_deg > 1 { (max_q_deg as f64 - 1.0).log2() } else { 0.0 };
    info!("Considering degrees between 2 and {} (blowup factor: {:.0})", max_q_deg, blowup);
    info!("------------------------------------------------------------");

    // Shared memo across degree iterations — the cache key includes `max_deg`,
    // so results for different degrees don't collide, but sub-problems that
    // happen to share `(idx, max_deg, im_pols)` are reused.
    let mut memo: HashMap<MemoKey, MemoVal> = HashMap::new();

    let mut best: Option<ImPolsResult> = None;
    let mut best_added_cols: i64 = 0;

    for d in 2..=max_q_deg.max(2) {
        if d > 2 {
            info!("------------------------------------------------------------");
        }

        // A degree can be infeasible (e.g. an inline composite that cannot be
        // promoted to an intermediate): skip it, it says nothing about the
        // remaining (larger) degrees.
        let Some((im_exps_p, q_deg_p)) = calculate_im_pols(expressions, c_exp_id, d, &mut memo) else {
            info!("Max constraint degree: {}", d);
            info!("Infeasible: no valid intermediate split exists at this degree");
            continue;
        };
        let new_added = calculate_added_cols(d, expressions, &im_exps_p, q_deg_p, q_dim);
        let no_intermediates = im_exps_p.is_empty();

        // On ties, keep the lowest feasible degree.
        if best.is_none() || new_added < best_added_cols {
            best_added_cols = new_added;
            best = Some(ImPolsResult { im_exps: im_exps_p, q_deg: q_deg_p });
        }

        // Once a degree needs no intermediates, every larger degree returns the
        // identical (empty, same q_deg) result: nothing left to improve.
        if no_intermediates {
            break;
        }
    }

    best.unwrap_or_else(|| {
        panic!("No feasible intermediate-polynomial split found for constraint degrees 2..={}", max_q_deg.max(2))
    })
}

fn calculate_added_cols(
    max_deg: usize,
    expressions: &[Expression],
    im_exps: &[usize],
    q_deg: i64,
    q_dim: usize,
) -> i64 {
    let q_cols = q_deg * q_dim as i64;
    let mut im_cols: i64 = 0;
    for &exp_id in im_exps {
        im_cols += expressions[exp_id].dim as i64;
    }
    let added_cols = q_cols + im_cols;
    info!("Max constraint degree: {}", max_deg);
    info!("Number of intermediate polynomials: {}", im_exps.len());
    info!("Polynomial Q degree: {}", q_deg);
    info!(
        "Number of columns added in the basefield: {} (Polynomial Q columns: {} + Intermediate polynomials columns: {})",
        added_cols, q_cols, im_cols
    );
    added_cols
}

// ---------------------------------------------------------------------------
// Inner recursive search (mirrors JS `_calculateImPols`)
// ---------------------------------------------------------------------------

/// Memoization key: (expression_id, max_deg, current im_pol IDs as an ordered set).
/// `BTreeSet<usize>` keeps the key canonical without an explicit sort step, and
/// turns the per-call `contains` check from O(N) (Vec) into O(log N).
type MemoKey = (usize, usize, BTreeSet<usize>);
/// Memoization value: (Option<im_pols>, degree).
/// `None` means the search failed for this sub-tree.
type MemoVal = (Option<BTreeSet<usize>>, i64);

fn calculate_im_pols(
    expressions: &[Expression],
    root_id: usize,
    max_deg: usize,
    memo: &mut HashMap<MemoKey, MemoVal>,
) -> Option<(Vec<usize>, i64)> {
    let absolute_max = max_deg;
    let mut abs_max_d: i64 = 0;

    let (result_pols, rd) =
        calc_im_pols_inner(expressions, root_id, &BTreeSet::new(), max_deg, absolute_max, &mut abs_max_d, memo);

    result_pols.map(|pols| {
        let final_deg = rd.max(abs_max_d) - 1;
        (pols.into_iter().collect(), final_deg)
    })
}

/// Recursive core. Returns `(Option<im_pols>, degree)`.
/// `None` in the first position means the sub-problem is infeasible.
fn calc_im_pols_inner(
    expressions: &[Expression],
    idx: usize,
    im_pols: &BTreeSet<usize>,
    max_deg: usize,
    absolute_max: usize,
    abs_max_d: &mut i64,
    memo: &mut HashMap<MemoKey, MemoVal>,
) -> (Option<BTreeSet<usize>>, i64) {
    let memo_key = (idx, max_deg, im_pols.clone());
    if let Some(cached) = memo.get(&memo_key) {
        return cached.clone();
    }

    let exp = &expressions[idx];
    let result = calc_im_pols_expr(expressions, exp, im_pols, max_deg, absolute_max, abs_max_d, memo);
    memo.insert(memo_key, result.clone());
    result
}

/// Inner recursive search that works on any expression (arena or inline).
#[allow(clippy::too_many_arguments)]
fn calc_im_pols_expr(
    expressions: &[Expression],
    exp: &Expression,
    im_pols: &BTreeSet<usize>,
    max_deg: usize,
    absolute_max: usize,
    abs_max_d: &mut i64,
    memo: &mut HashMap<MemoKey, MemoVal>,
) -> (Option<BTreeSet<usize>>, i64) {
    let op = exp.op.as_str();

    match op {
        "add" | "sub" => {
            let mut md: i64 = 0;
            let mut current_pols = im_pols.clone();
            for child in &exp.values {
                let (child_pols, d) = match child {
                    ExprChild::Id(id) => {
                        calc_im_pols_inner(expressions, *id, &current_pols, max_deg, absolute_max, abs_max_d, memo)
                    }
                    ExprChild::Inline(e) => {
                        calc_im_pols_expr(expressions, e, &current_pols, max_deg, absolute_max, abs_max_d, memo)
                    }
                };
                match child_pols {
                    None => return (None, -1),
                    Some(p) => {
                        current_pols = p;
                        if d > md {
                            md = d;
                        }
                    }
                }
            }
            (Some(current_pols), md)
        }
        "mul" => {
            // If either child is a non-composite degree-0 node, skip it
            let lhs_expr = exp.values[0].resolve(expressions);
            if !["add", "mul", "sub", "exp"].contains(&lhs_expr.op.as_str()) && lhs_expr.exp_deg == 0 {
                return match &exp.values[1] {
                    ExprChild::Id(id) => {
                        calc_im_pols_inner(expressions, *id, im_pols, max_deg, absolute_max, abs_max_d, memo)
                    }
                    ExprChild::Inline(e) => {
                        calc_im_pols_expr(expressions, e, im_pols, max_deg, absolute_max, abs_max_d, memo)
                    }
                };
            }
            let rhs_expr = exp.values[1].resolve(expressions);
            if !["add", "mul", "sub", "exp"].contains(&rhs_expr.op.as_str()) && rhs_expr.exp_deg == 0 {
                return match &exp.values[0] {
                    ExprChild::Id(id) => {
                        calc_im_pols_inner(expressions, *id, im_pols, max_deg, absolute_max, abs_max_d, memo)
                    }
                    ExprChild::Inline(e) => {
                        calc_im_pols_expr(expressions, e, im_pols, max_deg, absolute_max, abs_max_d, memo)
                    }
                };
            }

            let max_deg_here = exp.exp_deg as usize;
            if max_deg_here <= max_deg {
                return (Some(im_pols.clone()), max_deg_here as i64);
            }

            let mut eb: Option<BTreeSet<usize>> = None;
            let mut ed: i64 = -1;

            for l in 0..=max_deg {
                let r = max_deg - l;
                let (e1, d1) = match &exp.values[0] {
                    ExprChild::Id(id) => {
                        calc_im_pols_inner(expressions, *id, im_pols, l, absolute_max, abs_max_d, memo)
                    }
                    ExprChild::Inline(e) => {
                        calc_im_pols_expr(expressions, e, im_pols, l, absolute_max, abs_max_d, memo)
                    }
                };
                let Some(e1_set) = e1 else {
                    continue;
                };
                let (e2, d2) = match &exp.values[1] {
                    ExprChild::Id(id) => {
                        calc_im_pols_inner(expressions, *id, &e1_set, r, absolute_max, abs_max_d, memo)
                    }
                    ExprChild::Inline(e) => {
                        calc_im_pols_expr(expressions, e, &e1_set, r, absolute_max, abs_max_d, memo)
                    }
                };
                if let Some(e2_set) = e2 {
                    let e2_len = e2_set.len();
                    let should_replace = match &eb {
                        None => true,
                        Some(prev) => e2_len < prev.len(),
                    };
                    if should_replace {
                        eb = Some(e2_set);
                        ed = d1 + d2;
                    }
                    // Cannot do better than the starting set
                    if e2_len == im_pols.len() {
                        return (eb, ed);
                    }
                }
            }
            (eb, ed)
        }
        "exp" => {
            if max_deg < 1 {
                return (None, -1);
            }
            let id = exp.id.unwrap_or(0);
            if im_pols.contains(&id) {
                return (Some(im_pols.clone()), 1);
            }

            // calc_im_pols_inner handles memoization at its own entry; the
            // outer memo lookup and the redundant post-recurse memo write that
            // used to live here both reused the same key, so we drop them.
            let (e, d) = calc_im_pols_inner(expressions, id, im_pols, absolute_max, absolute_max, abs_max_d, memo);

            match e {
                None => (None, -1),
                Some(e_set) => {
                    if d > max_deg as i64 {
                        if d > *abs_max_d {
                            *abs_max_d = d;
                        }
                        let mut new_pols = e_set;
                        new_pols.insert(id);
                        (Some(new_pols), 1)
                    } else {
                        (Some(e_set), d)
                    }
                }
            }
        }
        _ => {
            // Leaf nodes: number, cm, const, challenge, etc.
            if exp.exp_deg == 0 {
                (Some(im_pols.clone()), 0)
            } else if max_deg < 1 {
                (None, -1)
            } else {
                (Some(im_pols.clone()), 1)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// addIntermediatePolynomials
// ---------------------------------------------------------------------------

/// Add intermediate polynomial witness columns and Q polynomial columns.
///
/// `c_exp_id` is the constraint expression ID, updated in-place.
/// Returns the final `(q_deg, q_dim, c_exp_id)` to be stored in the output.
///
/// Matches JS `addIntermediatePolynomials` from imPolynomials.js:
/// helper nodes (challenge, exp refs, cm, zi) are inline children,
/// only composite expressions are pushed to the arena.
#[allow(clippy::too_many_arguments)]
pub fn add_im_polynomials(
    expressions: &mut Vec<Expression>,
    constraints: &mut Vec<ConstraintInfo>,
    symbols: &mut Vec<SymbolInfo>,
    name: &str,
    air_id: usize,
    airgroup_id: usize,
    n_stages: usize,
    n_commitments: &mut usize,
    c_exp_id: &mut usize,
    im_exps: &[usize],
    q_deg: i64,
    im_pols_stages: bool,
    boundaries: &[(String, Option<i64>, Option<i64>)],
) -> usize {
    let dim = FIELD_EXTENSION;
    let stage = n_stages + 1;

    // Count existing challenges before this stage for vc_id
    let vc_id = symbols.iter().filter(|s| s.sym_type == "challenge" && s.stage.is_some_and(|st| st < stage)).count();

    // Create virtual challenge node INLINE (not pushed to arena)
    let vc_expr = Expression {
        op: "challenge".to_string(),
        id: Some(vc_id),
        dim,
        stage,
        stage_id: Some(0),
        exp_deg: 0,
        ..Default::default()
    };

    for &exp_id in im_exps {
        let stage_im = if im_pols_stages { expressions[exp_id].stage } else { n_stages };

        let stage_id = symbols.iter().filter(|s| s.sym_type == "witness" && s.stage == Some(stage_im)).count();

        let exp_dim = get_exp_dim(expressions, exp_id);

        let pol_id = *n_commitments;
        *n_commitments += 1;

        symbols.push(SymbolInfo {
            sym_type: "witness".to_string(),
            name: format!("{}.ImPol", name),
            id: Some(exp_id),
            pol_id: Some(pol_id),
            stage: Some(stage_im),
            stage_id: Some(stage_id),
            dim: exp_dim,
            air_id: Some(air_id),
            airgroup_id: Some(airgroup_id),
            im_pol: true,
            exp_id: Some(exp_id),
            ..Default::default()
        });

        expressions[exp_id].im_pol = true;
        expressions[exp_id].pol_id = Some(pol_id);
        expressions[exp_id].stage = stage_im;

        // Create sub-constraint: cm - imExpr (matches JS inline pattern)
        // JS: e = { op: "sub", values: [E.cm(...), Object.assign({}, expressions[imExps[i]])] }
        let cm_node = Expression {
            op: "cm".to_string(),
            id: Some(pol_id),
            row_offset: Some(0),
            stage: stage_im,
            dim: exp_dim,
            ..Default::default()
        };

        // Copy of the im expression (Object.assign in JS)
        let im_expr_copy = expressions[exp_id].clone();

        let mut sub_expr = Expression {
            op: "sub".to_string(),
            values: vec![ExprChild::Inline(Box::new(cm_node)), ExprChild::Inline(Box::new(im_expr_copy))],
            ..Default::default()
        };
        add_info_expression_inline(expressions, &mut sub_expr);
        expressions.push(sub_expr);
        let constraint_id = expressions.len() - 1;

        constraints.push(ConstraintInfo {
            e: constraint_id,
            boundary: "everyRow".to_string(),
            line: Some(format!("{}.ImPol", name)),
            stage: Some(expressions[exp_id].stage),
            offset_min: None,
            offset_max: None,
            im_pol: false,
        });

        // Weighted constraint: mul(vc, exp(cExpId)) - ONE push
        let c_exp_ref =
            Expression { op: "exp".to_string(), id: Some(*c_exp_id), row_offset: Some(0), stage, ..Default::default() };

        let mut weighted = Expression {
            op: "mul".to_string(),
            values: vec![ExprChild::Inline(Box::new(vc_expr.clone())), ExprChild::Inline(Box::new(c_exp_ref))],
            ..Default::default()
        };
        add_info_expression_inline(expressions, &mut weighted);
        expressions.push(weighted);
        let weighted_id = expressions.len() - 1;

        // Accumulated: add(exp(weighted_id), exp(constraint_id)) - ONE push
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

        let mut accum = Expression {
            op: "add".to_string(),
            values: vec![ExprChild::Inline(Box::new(weighted_ref)), ExprChild::Inline(Box::new(constraint_ref))],
            ..Default::default()
        };
        add_info_expression_inline(expressions, &mut accum);
        expressions.push(accum);
        *c_exp_id = expressions.len() - 1;
    }

    // Q polynomial: cExp * zi(everyRow)
    // JS: let q = E.mul(expressions[res.cExpId], E.zi(...));
    // JS clones the cExp expression inline and uses inline zi - ONE push
    let every_row_idx = boundaries.iter().position(|(bname, _, _)| bname == "everyRow").unwrap_or(0);

    let c_exp_copy = expressions[*c_exp_id].clone();
    let zi_node = Expression {
        op: "Zi".to_string(),
        boundary_id: Some(every_row_idx),
        boundary: Some("everyRow".to_string()),
        ..Default::default()
    };

    let mut q_expr = Expression {
        op: "mul".to_string(),
        values: vec![ExprChild::Inline(Box::new(c_exp_copy)), ExprChild::Inline(Box::new(zi_node))],
        ..Default::default()
    };
    add_info_expression_inline(expressions, &mut q_expr);
    expressions.push(q_expr);
    // JS does: res.cExpId++ after push, which means cExpId = expressions.length - 1
    *c_exp_id = expressions.len() - 1;

    let c_exp_dim = get_exp_dim(expressions, *c_exp_id);
    expressions[*c_exp_id].dim = c_exp_dim;

    let q_dim = c_exp_dim;

    // Create Q polynomial witness symbols
    for i in 0..q_deg {
        let index = *n_commitments;
        *n_commitments += 1;
        symbols.push(SymbolInfo {
            sym_type: "witness".to_string(),
            name: format!("Q{}", i),
            pol_id: Some(index),
            stage: Some(stage),
            dim: q_dim,
            air_id: Some(air_id),
            airgroup_id: Some(airgroup_id),
            ..Default::default()
        });
    }

    q_dim
}

// ---------------------------------------------------------------------------
// Default for SymbolInfo (needed for the `..Default::default()` above)
// ---------------------------------------------------------------------------

impl Default for SymbolInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            sym_type: String::new(),
            stage: None,
            dim: 1,
            id: None,
            pol_id: None,
            stage_id: None,
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::expression::Expression;

    /// Helper: create a number expression
    fn make_number(val: &str) -> Expression {
        Expression { op: "number".to_string(), value: Some(val.to_string()), exp_deg: 0, dim: 1, ..Default::default() }
    }

    /// Helper: create a committed polynomial (witness) node
    fn make_cm(id: usize, stage: usize) -> Expression {
        Expression {
            op: "cm".to_string(),
            id: Some(id),
            stage,
            dim: 1,
            exp_deg: 1,
            row_offset: Some(0),
            ..Default::default()
        }
    }

    /// Helper: create an "exp" reference node
    fn make_exp_ref(id: usize) -> Expression {
        Expression { op: "exp".to_string(), id: Some(id), ..Default::default() }
    }

    /// Helper: create a mul node
    fn make_mul(lhs: usize, rhs: usize, deg: i64) -> Expression {
        Expression {
            op: "mul".to_string(),
            values: vec![ExprChild::Id(lhs), ExprChild::Id(rhs)],
            exp_deg: deg,
            ..Default::default()
        }
    }

    // -----------------------------------------------------------------------
    // calculate_exp_deg tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_calc_exp_deg_leaf() {
        let exprs = vec![make_cm(0, 1)];
        let mut cache = HashMap::new();
        assert_eq!(calculate_exp_deg(&exprs, 0, &[], false, &mut cache), 1);
    }

    #[test]
    fn test_calc_exp_deg_mul() {
        // exprs[0] = cm, exprs[1] = cm, exprs[2] = mul(0,1)
        let exprs = vec![make_cm(0, 1), make_cm(1, 1), make_mul(0, 1, 2)];
        let mut cache = HashMap::new();
        assert_eq!(calculate_exp_deg(&exprs, 2, &[], false, &mut cache), 2);
    }

    #[test]
    fn test_calc_exp_deg_with_im_pol() {
        let exprs = vec![
            make_mul(0, 0, 2), // placeholder, not used directly
            make_cm(0, 1),
            make_cm(1, 1),
            make_mul(1, 2, 2), // exprs[3] = cm*cm, deg 2
            make_exp_ref(3),   // exprs[4] = exp ref to 3
        ];
        let mut cache = HashMap::new();
        // Without imPol
        assert_eq!(calculate_exp_deg(&exprs, 4, &[], false, &mut cache), 2);
        // With imPol on expr 3
        assert_eq!(calculate_exp_deg(&exprs, 4, &[3], false, &mut cache), 1);
    }

    #[test]
    fn test_calc_exp_deg_number() {
        let exprs = vec![make_number("42")];
        let mut cache = HashMap::new();
        assert_eq!(calculate_exp_deg(&exprs, 0, &[], false, &mut cache), 0);
    }

    // -----------------------------------------------------------------------
    // calculate_intermediate_polynomials tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_im_pols_needed() {
        let exprs = vec![
            make_cm(0, 1),     // 0
            make_cm(1, 1),     // 1
            make_mul(0, 1, 2), // 2: deg 2
        ];
        let result = calculate_intermediate_polynomials(&exprs, 2, 3, 1);
        assert!(
            result.im_exps.is_empty(),
            "No intermediate polynomials should be needed for degree-2 expr with maxQDeg=3"
        );
        assert_eq!(result.q_deg, 1); // deg 2 - 1
    }

    #[test]
    fn test_im_pols_needed_for_high_degree() {
        let mut exprs = vec![
            make_cm(0, 1),     // 0
            make_cm(1, 1),     // 1
            make_mul(0, 1, 2), // 2: cm_a * cm_b, deg 2
            make_exp_ref(2),   // 3: ref to expr 2, deg 2
            make_cm(2, 1),     // 4
            make_cm(3, 1),     // 5
            make_mul(4, 5, 2), // 6: cm_c * cm_d, deg 2
            make_exp_ref(6),   // 7: ref to expr 6, deg 2
            make_mul(3, 7, 4), // 8: (cm_a*cm_b) * (cm_c*cm_d), deg 4
        ];
        exprs[3].exp_deg = 2;
        exprs[7].exp_deg = 2;

        let result = calculate_intermediate_polynomials(&exprs, 8, 2, 1);
        assert!(
            !result.im_exps.is_empty(),
            "Intermediate polynomials should be needed for degree-4 expr with maxQDeg=2"
        );
    }

    // -----------------------------------------------------------------------
    // add_im_polynomials tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_im_pols_creates_q_symbols() {
        let mut expressions = vec![
            make_cm(0, 1),     // 0
            make_cm(1, 1),     // 1
            make_mul(0, 1, 2), // 2: cExp
        ];
        let mut constraints = Vec::new();
        let mut symbols = Vec::new();
        let mut n_commitments: usize = 2;
        let mut c_exp_id: usize = 2;
        let boundaries = vec![("everyRow".to_string(), None, None)];

        let q_dim = add_im_polynomials(
            &mut expressions,
            &mut constraints,
            &mut symbols,
            "test_air",
            0,
            0,
            1,
            &mut n_commitments,
            &mut c_exp_id,
            &[],
            1, // q_deg
            false,
            &boundaries,
        );

        // Should have added 1 Q polynomial witness symbol
        let q_symbols: Vec<_> = symbols.iter().filter(|s| s.name.starts_with("Q")).collect();
        assert_eq!(q_symbols.len(), 1);
        assert_eq!(q_symbols[0].sym_type, "witness");
        assert!(q_dim >= 1);
    }

    #[test]
    fn test_add_im_pols_with_im_exp() {
        let mut expressions = vec![
            make_cm(0, 1),     // 0: cm_a
            make_cm(1, 1),     // 1: cm_b
            make_mul(0, 1, 2), // 2: cm_a * cm_b (will be im_pol)
            make_cm(2, 1),     // 3: cm_c
            {
                // 4: exp ref to expr 2
                let mut e = make_exp_ref(2);
                e.exp_deg = 2;
                e
            },
            make_mul(4, 3, 3), // 5: ref(cm_a*cm_b) * cm_c, deg 3 (cExp)
        ];
        let mut constraints = Vec::new();
        let mut symbols = Vec::new();
        let mut n_commitments: usize = 3;
        let mut c_exp_id: usize = 5;
        let boundaries = vec![("everyRow".to_string(), None, None)];

        let _q_dim = add_im_polynomials(
            &mut expressions,
            &mut constraints,
            &mut symbols,
            "test_air",
            0,
            0,
            1,
            &mut n_commitments,
            &mut c_exp_id,
            &[2], // expr 2 is the intermediate polynomial
            1,
            false,
            &boundaries,
        );

        // Should have: 1 ImPol witness + 1 Q witness = 2 symbols
        let im_symbols: Vec<_> = symbols.iter().filter(|s| s.name.contains("ImPol")).collect();
        assert_eq!(im_symbols.len(), 1);
        assert_eq!(im_symbols[0].sym_type, "witness");

        let q_symbols: Vec<_> = symbols.iter().filter(|s| s.name.starts_with("Q")).collect();
        assert_eq!(q_symbols.len(), 1);

        // Should have added one constraint for the im polynomial
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].boundary, "everyRow");

        // Expression for im_pol should be marked
        assert!(expressions[2].im_pol);
        assert!(expressions[2].pol_id.is_some());
    }
}
