use serde_json::{json, Value};
use std::collections::HashMap;

use crate::helpers::get_exp_dim;
use crate::add_intermediate_pols::{calculate_exp_deg, ExpressionOps};

/// Generates the constraint polynomial and updates the provided resources, expressions, symbols, and constraints.
pub fn generate_constraint_polynomial(
    res: &mut HashMap<String, Value>,
    expressions: &mut Vec<Value>,
    symbols: &mut Vec<HashMap<String, Value>>,
    constraints: &[Value],
) {
    let stage = res["nStages"].as_u64().unwrap_or(0) as usize + 1;
    let e = ExpressionOps::new(stage, 3);

    // Add "std_vc" challenge to symbols
    let vc_id = symbols
        .iter()
        .filter(|s| s.get("type") == Some(&json!("challenge")) && s["stage"].as_u64().unwrap_or(0) < stage as u64)
        .count();

    symbols.push(HashMap::from([
        ("type".to_string(), json!("challenge")),
        ("name".to_string(), json!("std_vc")),
        ("stage".to_string(), json!(stage)),
        ("dim".to_string(), json!(3)),
        ("stageId".to_string(), json!(0)),
        ("id".to_string(), json!(vc_id)),
    ]));

    // FIX: Corrected `challenge` method call (added `dim` and `stage_id`)
    let vc = e.challenge("std_vc", stage, 3, 0, vc_id);

    // Retrieve and store cExpId before modifying res
    let c_exp_id = expressions.len();
    res.insert("cExpId".to_string(), json!(c_exp_id));

    // Ensure "boundaries" exists in `res`
    let mut boundaries = res.entry("boundaries".to_string()).or_insert_with(|| json!([])).take(); // Take ownership to avoid mutable borrow conflicts

    let boundaries_array = boundaries.as_array_mut().expect("Expected 'boundaries' to be an array");

    // Process constraints
    for constraint in constraints {
        let boundary = constraint["boundary"].as_str().expect("Boundary must be a string");

        if !["everyRow", "firstRow", "lastRow", "everyFrame"].contains(&boundary) {
            panic!("Boundary {} not supported", boundary);
        }

        // FIX: Removed the incorrect call to `e.exp` (no `exp` method exists)
        let mut expr = json!({
            "op": "exp",
            "e": constraint["e"].as_u64().unwrap_or(0) as usize,
            "stage": 0
        });

        // Handle boundary-specific logic
        if boundary == "everyFrame" {
            let boundary_id = boundaries_array
                .iter()
                .position(|b| {
                    b["name"] == json!("everyFrame")
                        && b["offsetMin"] == constraint["offsetMin"]
                        && b["offsetMax"] == constraint["offsetMax"]
                })
                .unwrap_or_else(|| {
                    boundaries_array.push(json!({
                        "name": "everyFrame",
                        "offsetMin": constraint["offsetMin"],
                        "offsetMax": constraint["offsetMax"]
                    }));
                    boundaries_array.len() - 1
                });

            expr = e.mul(expr, e.zi(boundary_id));
        } else if boundary != "everyRow" {
            let boundary_id = boundaries_array.iter().position(|b| b["name"] == json!(boundary)).unwrap_or_else(|| {
                boundaries_array.push(json!({ "name": boundary }));
                boundaries_array.len() - 1
            });

            expr = e.mul(expr, e.zi(boundary_id));
        }

        // Update or push the constraint polynomial expression
        if expressions.len() == c_exp_id {
            expressions.push(expr);
        } else {
            expressions[c_exp_id] = e.add(e.mul(vc.clone(), expressions[c_exp_id].clone()), expr);
        }
    }

    // Restore "boundaries" back into res
    res.insert("boundaries".to_string(), boundaries);

    // Update qDim with the dimensionality of the constraint polynomial
    res.insert("qDim".to_string(), json!(get_exp_dim(expressions, c_exp_id)));

    // Add "std_xi" challenge to symbols
    let xi_id = symbols
        .iter()
        .filter(|s| s.get("type") == Some(&json!("challenge")) && s["stage"].as_u64().unwrap_or(0) < stage as u64 + 1)
        .count();

    symbols.push(HashMap::from([
        ("type".to_string(), json!("challenge")),
        ("name".to_string(), json!("std_xi")),
        ("stage".to_string(), json!(stage + 1)),
        ("dim".to_string(), json!(3)),
        ("stageId".to_string(), json!(0)),
        ("id".to_string(), json!(xi_id)),
    ]));

    // FIX: Corrected `calculate_exp_deg` (pass `c_exp_id`, not `&expressions[c_exp_id]`)
    let initial_q_degree = calculate_exp_deg(expressions, c_exp_id, &[], false);

    println!("The maximum constraint degree is {} (without intermediate polynomials)", initial_q_degree);
}
