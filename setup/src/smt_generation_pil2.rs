use z3::{
    ast::{Ast, Bool, Int},
    Context, Optimize, SatResult, Symbol,
};
use std::collections::HashSet;

/// Declares the `needed_variables` constraint in the solver.
pub fn declare_minimize_keeps<'ctx>(number_intermediates: usize, solver: &Optimize<'ctx>, ctx: &'ctx Context) {
    let mut aux = Int::from_i64(ctx, 0);
    for s in 0..number_intermediates {
        let k_exp = Bool::new_const(ctx, format!("k_exp_{}", s));
        aux = &aux + &k_exp.ite(&Int::from_i64(ctx, 1), &Int::from_i64(ctx, 0));
    }
    solver.assert(&Int::new_const(ctx, "needed_variables")._eq(&aux));
}

/// Declares the keep variables constraints
pub fn declare_keep_variables<'ctx>(number_intermediates: usize, n: i64, solver: &Optimize<'ctx>, ctx: &'ctx Context) {
    for s in 0..number_intermediates {
        let d_exp = Int::new_const(ctx, format!("d_exp_{}", s));
        solver.assert(&d_exp.ge(&Int::from_i64(ctx, 0)));
        solver.assert(&d_exp.le(&Int::from_i64(ctx, n)));
    }
}

/// Generates constraints for expression declaration
pub fn generate_expression_declaration<'ctx>(
    tree: &serde_json::Value,
    expressions: &[serde_json::Value],
    zero_expressions: &HashSet<String>,
    one_expressions: &HashSet<String>,
    position: usize,
    n: i64,
    solver: &Optimize<'ctx>,
    ctx: &'ctx Context,
) {
    let possible_degrees =
        get_degrees_tree(tree, zero_expressions, one_expressions, &format!("aux_{}", position), n, solver, ctx);

    let d_exp = Int::new_const(ctx, format!("d_exp_{}", position));
    let k_exp = Bool::new_const(ctx, format!("k_exp_{}", position));

    solver.assert(&k_exp.implies(&d_exp._eq(&Int::from_i64(ctx, 1))));
    solver.assert(&k_exp.not().implies(&d_exp._eq(&possible_degrees)));

    solver.assert_soft(&k_exp.not(), 1, Some(Symbol::String("degree".into())));

    if let Some(dim) = expressions[position].get("dim").and_then(|v| v.as_i64()) {
        solver.assert_soft(&k_exp.not(), dim, Some(Symbol::String("dimension".into())));
    } else {
        solver.assert(&k_exp.not());
    }
}

pub fn get_degrees_tree<'ctx>(
    tree: &serde_json::Value,
    zero_expressions: &HashSet<String>,
    one_expressions: &HashSet<String>,
    prefix: &str,
    n: i64,
    solver: &Optimize<'ctx>,
    ctx: &'ctx Context,
) -> Int<'ctx> {
    if tree.is_array() {
        let prefix_var = Int::new_const(ctx, prefix);
        solver.assert(&prefix_var.le(&Int::from_i64(ctx, n)));

        let mut or_condition = Bool::from_bool(ctx, false);
        for (i, e) in tree.as_array().unwrap().iter().enumerate() {
            let new_degree =
                get_degrees_tree(e, zero_expressions, one_expressions, &format!("{}_{}", prefix, i), n, solver, ctx);
            solver.assert(&prefix_var.ge(&new_degree));
            or_condition = Bool::or(ctx, &[&or_condition, &prefix_var.le(&new_degree)]);
        }
        return prefix_var;
    } else if tree.is_array() && tree.as_array().unwrap().len() == 2 {
        let left_degree =
            get_degrees_tree(&tree[0], zero_expressions, one_expressions, &format!("{}_0", prefix), n, solver, ctx);
        let right_degree =
            get_degrees_tree(&tree[1], zero_expressions, one_expressions, &format!("{}_1", prefix), n, solver, ctx);

        // Fix: Convert the formatted string to `&str`
        let total_degree_var = Int::new_const(ctx, format!("{}_total_degree", prefix).as_str());

        solver.assert(&total_degree_var.le(&Int::from_i64(ctx, n)));
        solver.assert(&total_degree_var._eq(&(left_degree + right_degree)));
        return total_degree_var;
    } else if let Some(val) = tree.as_i64() {
        return Int::from_i64(ctx, val);
    } else {
        let tree_str = tree.as_str().unwrap().to_string();
        if zero_expressions.contains(&tree_str) {
            return Int::from_i64(ctx, 0);
        } else if one_expressions.contains(&tree_str) {
            return Int::from_i64(ctx, 1);
        } else {
            return Int::new_const(ctx, tree_str.as_str());
        }
    }
}

/// Retrieves the minimal set of expressions needed
pub fn get_minimal_expressions<'ctx>(
    number_intermediates: usize,
    solver: &Optimize<'ctx>,
    ctx: &'ctx Context,
) -> HashSet<usize> {
    let mut should_keep = HashSet::new();
    if solver.check(&[]) == SatResult::Sat {
        let model = solver.get_model().unwrap();
        for s in 0..number_intermediates {
            if model.eval(&Bool::new_const(ctx, format!("k_exp_{}", s)), true).unwrap().as_bool().unwrap() {
                should_keep.insert(s);
            }
        }
        println!("Number of needed variables: {}", should_keep.len());
    }
    should_keep
}
