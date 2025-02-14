use crate::smt_generation_pil2::{declare_keep_variables, generate_expression_declaration, get_minimal_expressions};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use z3::{Config, Context, Optimize};

/// Computes the mixed degrees of three sets of degrees.
pub fn mix_degrees(deg_1: &[Value], deg_2: &[Value], deg_3: &[Value]) -> Vec<Value> {
    let mut result = deg_3.to_vec();
    for i in deg_1 {
        for j in deg_2 {
            result.push(json!((i, j)));
        }
    }
    result
}

/// Finds all used expressions recursively.
pub fn get_used_expressions(expr: &Value, expressions: &[Value], used_expressions: &mut HashSet<usize>) {
    let type_union = ["add", "sub", "mul"].iter().cloned().collect::<HashSet<_>>();

    if let Some(op) = expr.get("op").and_then(|v| v.as_str()) {
        if type_union.contains(op) {
            if let Some(values) = expr.get("values").and_then(|v| v.as_array()) {
                for e in values {
                    get_used_expressions(e, expressions, used_expressions);
                }
            }
        } else if op == "exp" {
            if let Some(id) = expr.get("id").and_then(|v| v.as_u64()) {
                let id_usize = id as usize;
                if !used_expressions.contains(&id_usize) {
                    used_expressions.insert(id_usize);
                    get_used_expressions(&expressions[id_usize], expressions, used_expressions);
                }
            }
        }
    }
}

/// Parses a PIL expression to determine possible max values.
pub fn parse_expression_pil(expression: &Value) -> Value {
    let type_add = ["add", "sub", "neg"].iter().cloned().collect::<HashSet<_>>();

    if let Some(op) = expression.get("op").and_then(|v| v.as_str()) {
        if type_add.contains(op) {
            let mut result = Vec::new();
            if let Some(values) = expression.get("values").and_then(|v| v.as_array()) {
                for e in values {
                    result.push(parse_expression_pil(e));
                }
            }
            return json!(result);
        } else if op == "mul" {
            let values = expression.get("values").and_then(|v| v.as_array()).unwrap();
            return json!((parse_expression_pil(&values[0]), parse_expression_pil(&values[1])));
        } else if op == "exp" {
            return json!(format!("exp_{}", expression["id"]));
        } else if op == "challenge" {
            return json!(0);
        }
    }
    json!(expression["expDeg"].as_i64().unwrap())
}

/// **Library function replacing the main function**
pub fn process_pil_data(input_json: &str) -> String {
    let data: Value = serde_json::from_str(input_json).expect("Failed to parse JSON");

    let expressions = data["expressions"].as_array().unwrap().clone();
    let c_exp_id = data["cExpId"].as_u64().unwrap() as usize;
    let degree = data["maxDeg"].as_i64().unwrap();
    let q_dim = data["qDim"].as_i64().unwrap();

    let mut used_expressions = HashSet::new();
    used_expressions.insert(c_exp_id);
    get_used_expressions(&expressions[c_exp_id], &expressions, &mut used_expressions);

    let zero_expressions = HashSet::new();
    let one_expressions = HashSet::new();
    let mut trees = HashMap::new();

    for (i, e) in expressions.iter().enumerate() {
        if used_expressions.contains(&i) {
            let tree = parse_expression_pil(e);
            let new_tree = tree.clone();
            trees.insert(i, new_tree);
        }
    }

    let mut min_vars = expressions.len();
    let mut min_value = -1;
    let mut optimal_degree = -1;
    let mut used_variables = HashSet::new();
    let mut possible_degree = 2;

    println!("-------------------- POSSIBLE DEGREES ----------------------");
    println!(
        "** Considering degrees between 2 and {} (blowup factor: {}) **",
        degree,
        (degree as f64 - 1.0).log2() as i64
    );
    println!("------------------------------------------------------------");

    // Create Z3 context
    let config = Config::new();
    let ctx = Context::new(&config);

    while min_vars != 0 && possible_degree <= degree {
        println!("------------------------------------------------------------");
        let solver = Optimize::new(&ctx);

        declare_keep_variables(expressions.len(), possible_degree, &solver, &ctx);

        for (index, value) in &trees {
            generate_expression_declaration(
                value,
                &expressions,
                &zero_expressions,
                &one_expressions,
                *index,
                possible_degree,
                &solver,
                &ctx,
            );
        }

        let new_used_variables = get_minimal_expressions(expressions.len(), &solver, &ctx);
        let added_basefield_cols = q_dim * possible_degree
            + new_used_variables.iter().map(|v| expressions[*v]["dim"].as_i64().unwrap()).sum::<i64>();

        if min_value == -1 || added_basefield_cols < min_value {
            min_value = added_basefield_cols;
            used_variables = new_used_variables.clone();
            optimal_degree = possible_degree - 1;
        }
        if new_used_variables.len() < min_vars {
            min_vars = new_used_variables.len();
        }

        possible_degree += 1;
    }

    let solution = json!({
        "newExpressions": expressions,
        "imExps": used_variables.into_iter().collect::<Vec<_>>(),
        "qDeg": optimal_degree
    });

    solution.to_string()
}
