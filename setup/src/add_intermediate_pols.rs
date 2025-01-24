use serde_json::{json, Value};
use std::collections::HashMap;

/// Struct for managing expressions and operations
pub struct ExpressionOps;

impl ExpressionOps {
    pub fn new() -> Self {
        Self
    }

    pub fn challenge(&self, _name: &str, _stage: usize, _dim: usize, _value: u32, _id: usize) -> Value {
        json!({
            "op": "challenge",
            "stage": _stage,
            "dim": _dim,
            "value": _value,
            "id": _id
        })
    }

    pub fn add(&self, lhs: Value, rhs: Value) -> Value {
        json!({ "op": "add", "values": [lhs, rhs] })
    }

    pub fn mul(&self, lhs: Value, rhs: Value) -> Value {
        json!({ "op": "mul", "values": [lhs, rhs] })
    }

    pub fn cm(&self, id: usize, _value: usize, _stage: usize, _dim: usize) -> Value {
        json!({
            "op": "cm",
            "id": id,
            "value": _value,
            "stage": _stage,
            "dim": _dim
        })
    }

    pub fn zi(&self, boundary_index: usize) -> Value {
        json!({
            "op": "Zi",
            "boundary": boundary_index
        })
    }
}

/// Adds intermediate polynomials based on provided expressions and constraints
pub fn add_intermediate_polynomials(
    res: &mut HashMap<String, Value>,
    expressions: &mut Vec<Value>,
    constraints: &mut Vec<Value>,
    symbols: &mut Vec<HashMap<String, Value>>,
    im_exps: &[usize],
    q_deg: usize,
) {
    let e = ExpressionOps::new();

    println!("--------------------- SELECTED DEGREE ----------------------");
    println!("Constraints maximum degree: {}", q_deg + 1);
    println!("Number of intermediate polynomials required: {}", im_exps.len());

    res.insert("qDeg".to_string(), json!(q_deg));

    let stage = res["nStages"].as_u64().unwrap_or(0) as usize + 1;

    let vc_id = symbols
        .iter()
        .filter(|s| s.get("type") == Some(&json!("challenge")) && s["stage"].as_u64().unwrap_or(0) < stage as u64)
        .count();

    let vc = e.challenge("std_vc", stage, 3, 0, vc_id);
    let max_deg_expr =
        calculate_exp_deg(expressions, &expressions[res["cExpId"].as_u64().unwrap_or(0) as usize], im_exps);

    if max_deg_expr > q_deg + 1 {
        panic!(
            "The maximum degree of the constraint expression has a higher degree ({}) than the maximum allowed degree ({})",
            max_deg_expr,
            q_deg + 1
        );
    }

    for &exp_id in im_exps {
        let im_pol_deg = calculate_exp_deg(expressions, &expressions[exp_id], im_exps);
        if im_pol_deg > q_deg + 1 {
            panic!(
                "Intermediate polynomial with id: {} has a higher degree ({}) than the maximum allowed degree ({})",
                exp_id,
                im_pol_deg,
                q_deg + 1
            );
        }
    }

    for &exp_id in im_exps {
        let stage_im = res
            .get("imPolsStages")
            .map(|_| expressions[exp_id]["stage"].as_u64().unwrap_or(0) as usize)
            .unwrap_or(res["nStages"].as_u64().unwrap_or(0) as usize);

        let stage_id = symbols
            .iter()
            .filter(|s| s.get("type") == Some(&json!("witness")) && s["stage"] == json!(stage_im))
            .count();

        let dim = get_exp_dim(expressions, exp_id);

        symbols.push(HashMap::from([
            ("type".to_string(), json!("witness")),
            ("name".to_string(), json!(format!("{}.ImPol", res["name"]))),
            ("expId".to_string(), json!(exp_id)),
            ("polId".to_string(), json!(res["nCommitments"].as_u64().unwrap_or(0) as usize)),
            ("stage".to_string(), json!(stage_im)),
            ("stageId".to_string(), json!(stage_id)),
            ("dim".to_string(), json!(dim)),
            ("imPol".to_string(), json!(true)),
            ("airId".to_string(), res["airId"].clone()),
            ("airgroupId".to_string(), res["airgroupId"].clone()),
        ]));

        expressions[exp_id]["imPol"] = json!(true);
        expressions[exp_id]["polId"] = json!(res["nCommitments"].as_u64().unwrap_or(0) as usize);
        expressions[exp_id]["stage"] = json!(stage_im);

        let intermediate_expr = json!({
            "op": "sub",
            "values": [
                e.cm(res["nCommitments"].as_u64().unwrap_or(0) as usize, 0, stage_im, dim),
                expressions[exp_id].clone()
            ]
        });

        expressions.push(intermediate_expr.clone());
        add_info_expressions(expressions, &intermediate_expr);

        constraints.push(json!({
            "e": expressions.len() - 1,
            "boundary": "everyRow",
            "filename": format!("{}.ImPol", res["name"]),
            "stage": expressions[exp_id]["stage"]
        }));

        let updated_expr = e.add(
            e.mul(vc.clone(), expressions[res["cExpId"].as_u64().unwrap_or(0) as usize].clone()),
            intermediate_expr,
        );

        expressions[res["cExpId"].as_u64().unwrap_or(0) as usize] = updated_expr;
    }

    let final_expr = e.mul(
        expressions[res["cExpId"].as_u64().unwrap_or(0) as usize].clone(),
        e.zi(res["boundaries"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .position(|b| b["name"] == json!("everyRow"))
            .unwrap_or(0)),
    );

    expressions[res["cExpId"].as_u64().unwrap_or(0) as usize] = final_expr;
    expressions[res["cExpId"].as_u64().unwrap_or(0) as usize]["stage"] =
        json!(res["nStages"].as_u64().unwrap_or(0) + 1);

    let c_exp_dim = get_exp_dim(expressions, res["cExpId"].as_u64().unwrap_or(0) as usize);
    expressions[res["cExpId"].as_u64().unwrap_or(0) as usize]["dim"] = json!(c_exp_dim);

    res.insert("qDim".to_string(), json!(c_exp_dim));

    for i in 0..q_deg {
        let index = res["nCommitments"].as_u64().unwrap_or(0) as usize + i;
        symbols.push(HashMap::from([
            ("type".to_string(), json!("witness")),
            ("name".to_string(), json!(format!("Q{}", i))),
            ("polId".to_string(), json!(index)),
            ("stage".to_string(), json!(stage)),
            ("dim".to_string(), json!(res["qDim"])),
            ("airId".to_string(), res["airId"].clone()),
            ("airgroupId".to_string(), res["airgroupId"].clone()),
        ]));
        e.cm(index, 0, stage, c_exp_dim);
    }
}

/// Calculates the maximum degree of an expression.
pub fn calculate_exp_deg(expressions: &[Value], exp: &Value, im_exps: &[usize]) -> usize {
    match exp.get("op").and_then(|v| v.as_str()) {
        Some("exp") => {
            if im_exps.contains(&(exp["id"].as_u64().unwrap_or(0) as usize)) {
                1
            } else {
                calculate_exp_deg(expressions, &expressions[exp["id"].as_u64().unwrap_or(0) as usize], im_exps)
            }
        }
        Some("mul") => {
            let lhs = calculate_exp_deg(expressions, &exp["values"][0], im_exps);
            let rhs = calculate_exp_deg(expressions, &exp["values"][1], im_exps);
            lhs + rhs
        }
        Some("add") | Some("sub") => {
            let lhs = calculate_exp_deg(expressions, &exp["values"][0], im_exps);
            let rhs = calculate_exp_deg(expressions, &exp["values"][1], im_exps);
            usize::max(lhs, rhs)
        }
        _ => 0,
    }
}

/// Updates metadata for expressions.
fn add_info_expressions(expressions: &mut [Value], expr: &Value) {
    if let Some(degree) = expr["degree"].as_u64() {
        expressions.last_mut().unwrap()["degree"] = json!(degree);
    }
}

/// Gets the dimensionality of an expression.
fn get_exp_dim(expressions: &[Value], exp_id: usize) -> usize {
    expressions[exp_id]["dim"].as_u64().unwrap_or(1) as usize
}
