use serde_json::{json, Value};
use std::collections::HashMap;

/// Struct for managing expressions and operations
pub struct ExpressionOps {
    pub stage: usize,
    pub dim: usize,
}

impl ExpressionOps {
    /// Creates a new `ExpressionOps` instance with the default stage and dimension.
    pub fn new(stage: usize, dim: usize) -> Self {
        Self { stage, dim }
    }

    /// Creates an exponential operation
    pub fn exp(&self, id: usize, value: usize) -> Value {
        json!({
            "op": "exp",
            "id": id,
            "value": value,
            "stage": self.stage
        })
    }

    /// Creates a `Zi` operation for a given boundary
    pub fn zi(&self, boundary_id: usize) -> Value {
        json!({
            "op": "Zi",
            "boundary": boundary_id,
            "stage": self.stage
        })
    }

    pub fn cm(&self, id: usize, value: usize) -> Value {
        json!({
            "op": "cm",
            "id": id,
            "value": value,
            "stage": self.stage,
            "dim": self.dim
        })
    }

    /// Adds two expressions together
    pub fn add(&self, lhs: Value, rhs: Value) -> Value {
        json!({ "op": "add", "values": [lhs, rhs] })
    }

    /// Multiplies two expressions
    pub fn mul(&self, lhs: Value, rhs: Value) -> Value {
        json!({ "op": "mul", "values": [lhs, rhs] })
    }

    /// Creates a challenge expression
    pub fn challenge(&self, name: &str, stage_id: usize, id: usize) -> Value {
        json!({
            "op": "challenge",
            "name": name,
            "stage": self.stage,
            "dim": self.dim,
            "stageId": stage_id,
            "id": id
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
    let stage = res["nStages"].as_u64().unwrap_or(0) as usize + 1;
    let e = ExpressionOps::new(stage, 3);

    println!("--------------------- SELECTED DEGREE ----------------------");
    println!("Constraints maximum degree: {}", q_deg + 1);
    println!("Number of intermediate polynomials required: {}", im_exps.len());

    res.insert("qDeg".to_string(), json!(q_deg));

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

    let vc = e.challenge("std_vc", 0, vc_id);
    let max_deg_expr =
        calculate_exp_deg(expressions, &expressions[res["cExpId"].as_u64().unwrap_or(0) as usize], im_exps);

    if max_deg_expr > q_deg + 1 {
        panic!(
            "The maximum degree of the constraint expression has a higher degree ({}) than the maximum allowed degree ({})",
            max_deg_expr, q_deg + 1
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
        let stage_im = res["nStages"].as_u64().unwrap_or(0) as usize;
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

        let intermediate_expr =
            e.add(e.cm(res["nCommitments"].as_u64().unwrap_or(0) as usize, 0), expressions[exp_id].clone());

        expressions.push(intermediate_expr.clone());
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
