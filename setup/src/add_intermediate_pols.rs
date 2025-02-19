use std::collections::HashMap;
use serde_json::{json, Value};

use crate::helpers::get_exp_dim;

/// Struct for managing expressions and operations
#[derive(Debug, Clone)]
pub struct ExpressionOps {
    pub stage: usize,
    pub dim: usize,
}

impl ExpressionOps {
    /// Creates a new `ExpressionOps` instance with the given stage and dimension.
    pub fn new(stage: usize, dim: usize) -> Self {
        Self { stage, dim }
    }

    /// Adds two expressions together
    pub fn add(&self, a: Value, b: Value) -> Value {
        if a.is_null() {
            return b;
        }
        if b.is_null() {
            return a;
        }
        json!({ "op": "add", "values": [a, b] })
    }

    /// Subtracts one expression from another
    pub fn sub(&self, a: Value, b: Value) -> Value {
        if a.is_null() {
            return b;
        }
        if b.is_null() {
            return a;
        }
        json!({ "op": "sub", "values": [a, b] })
    }

    /// Multiplies two expressions
    pub fn mul(&self, a: Value, b: Value) -> Value {
        if a.is_null() {
            return b;
        }
        if b.is_null() {
            return a;
        }
        json!({ "op": "mul", "values": [a, b] })
    }

    /// Creates an exponential (`exp`) operation
    pub fn exp(&self, id: usize, row_offset: usize, stage: usize) -> Value {
        json!({
            "op": "exp",
            "id": id,
            "rowOffset": row_offset,
            "stage": stage
        })
    }

    /// Creates a column memory (cm) operation
    pub fn cm(&self, id: usize, row_offset: usize, stage: Option<usize>, dim: usize) -> Value {
        let stage = stage.unwrap_or_else(|| panic!("Stage not defined for cm {}", id));
        json!({
            "op": "cm",
            "id": id,
            "stage": stage,
            "dim": dim,
            "rowOffset": row_offset
        })
    }

    /// Creates a custom operation
    pub fn custom(&self, id: usize, row_offset: usize, stage: Option<usize>, dim: usize, commit_id: usize) -> Value {
        let stage = stage.unwrap_or_else(|| panic!("Stage not defined for custom {}", id));
        json!({
            "op": "custom",
            "id": id,
            "stage": stage,
            "dim": dim,
            "rowOffset": row_offset,
            "commitId": commit_id
        })
    }

    /// Creates a challenge expression
    pub fn challenge(&self, name: &str, stage: usize, dim: usize, stage_id: usize, id: usize) -> Value {
        json!({
            "op": "challenge",
            "name": name,
            "stageId": stage_id,
            "id": id,
            "stage": stage,
            "dim": dim
        })
    }

    /// Creates a `q` operation
    pub fn q(&self, q_dim: usize) -> Value {
        json!({
            "op": "q",
            "id": 0,
            "dim": q_dim
        })
    }

    /// Creates an `f` operation
    pub fn f(&self) -> Value {
        json!({
            "op": "f",
            "id": 0,
            "dim": 3
        })
    }

    /// Creates a `const` operation
    pub fn const_(&self, id: usize, row_offset: usize, stage: usize, dim: usize) -> Value {
        if stage != 0 {
            panic!("Const must be declared in stage 0");
        }
        json!({
            "op": "const",
            "id": id,
            "rowOffset": row_offset,
            "dim": dim,
            "stage": stage
        })
    }

    /// Creates a `number` operation
    pub fn number(&self, n: f64) -> Value {
        json!({
            "op": "number",
            "value": n.to_string()
        })
    }

    /// Creates an `eval` operation
    pub fn eval(&self, id: usize, dim: usize) -> Value {
        json!({
            "op": "eval",
            "id": id,
            "dim": dim
        })
    }

    /// Creates an `xDivXSubXi` operation
    pub fn x_div_x_sub_xi(&self, opening: usize, id: usize) -> Value {
        json!({
            "op": "xDivXSubXi",
            "opening": opening,
            "id": id
        })
    }

    /// Creates a `Zi` operation
    pub fn zi(&self, boundary_id: usize) -> Value {
        json!({
            "op": "Zi",
            "boundaryId": boundary_id
        })
    }

    /// Creates an `x` operation
    pub fn x(&self) -> Value {
        json!({
            "op": "x"
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
    q_deg: i64,
) {
    let stage = res["nStages"].as_u64().unwrap_or(0) as usize + 1;
    let e = ExpressionOps::new(stage, 3);

    println!("--------------------- SELECTED DEGREE ----------------------");
    println!("Constraints maximum degree: {}", q_deg + 1);
    println!("Number of intermediate polynomials required: {}", im_exps.len());

    res.insert("qDeg".to_string(), json!(q_deg));

    let vc_id = symbols
        .iter()
        .filter(|s| {
            println!("🔍 Checking symbol: {:?}", s);
            s.get("type") == Some(&json!("challenge")) && s["stage"].as_u64().unwrap_or(0) < stage as u64
        })
        .count();

    let vc = e.challenge("std_vc", stage, 3, 0, vc_id);

    let max_deg_expr = calculate_exp_deg(
        expressions,
        res["cExpId"].as_u64().unwrap_or(0) as usize,
        im_exps,
        false, // cache_values
    );

    if max_deg_expr as i64 > q_deg + 1 {
        panic!(
            "The maximum degree of the constraint expression has a higher degree ({}) than the maximum allowed degree ({})",
            max_deg_expr, q_deg + 1
        );
    }

    for &exp_id in im_exps {
        let im_pol_deg = calculate_exp_deg(expressions, exp_id, im_exps, false);
        if im_pol_deg as i64 > q_deg + 1 {
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
            ("airGroupId".to_string(), res["airGroupId"].clone()),
        ]));

        let intermediate_expr = e.sub(
            e.cm(res["nCommitments"].as_u64().unwrap_or(0) as usize, 0, Some(stage_im), dim),
            expressions[exp_id].clone(),
        );

        expressions.push(intermediate_expr.clone());

        constraints.push(json!({
            "e": expressions.len() - 1,
            "boundary": "everyRow",
            "filename": format!("{}.ImPol", res["name"]),
            "stage": expressions[exp_id]["stage"]
        }));

        expressions[res["cExpId"].as_u64().unwrap_or(0) as usize] = e.add(
            e.mul(vc.clone(), expressions[res["cExpId"].as_u64().unwrap_or(0) as usize].clone()),
            intermediate_expr,
        );
    }

    let every_row_index = res["boundaries"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .position(|b| b["name"] == json!("everyRow"))
        .unwrap_or_else(|| panic!("Boundary 'everyRow' not found"));

    expressions[res["cExpId"].as_u64().unwrap_or(0) as usize] =
        e.mul(expressions[res["cExpId"].as_u64().unwrap_or(0) as usize].clone(), e.zi(every_row_index));

    res.insert("qDim".to_string(), json!(get_exp_dim(expressions, res["cExpId"].as_u64().unwrap_or(0) as usize)));

    for i in 0..res["qDeg"].as_u64().unwrap_or(0) as usize {
        let index = res["nCommitments"].as_u64().unwrap_or(0) as usize;
        res.insert("nCommitments".to_string(), json!(index + 1));

        symbols.push(HashMap::from([
            ("type".to_string(), json!("witness")),
            ("name".to_string(), json!(format!("Q{}", i))),
            ("polId".to_string(), json!(index)),
            ("stage".to_string(), json!(stage)),
            ("dim".to_string(), json!(res["qDim"])),
        ]));

        expressions.push(e.cm(index, 0, Some(stage), res["qDim"].as_u64().unwrap_or(0) as usize));
    }
}

use std::collections::HashSet;

/// Public function: Keeps the existing API
pub fn calculate_exp_deg(expressions: &mut [Value], exp_id: usize, im_exps: &[usize], cache_values: bool) -> usize {
    let mut visited = HashSet::new();
    calculate_exp_deg_helper(expressions, exp_id, im_exps, cache_values, &mut visited)
}

/// Recursive helper function: Does the actual work
fn calculate_exp_deg_helper(
    expressions: &mut [Value],
    exp_id: usize,
    im_exps: &[usize],
    cache_values: bool,
    visited: &mut HashSet<usize>,
) -> usize {
    if visited.contains(&exp_id) {
        println!("⚠️ Possible cycle detected at exp_id {}. Returning 1 instead of 0.", exp_id);
        return 1;
    }

    visited.insert(exp_id);
    println!("🔍 Entering calculate_exp_deg for exp_id: {}", exp_id);

    // Clone the expression to avoid borrowing issues
    let exp = expressions.get(exp_id).expect("Invalid exp_id index").clone();
    println!("📜 Processing exp_id {}: {:?}", exp_id, exp);

    if cache_values {
        if let Some(degree) = exp.get("degree_").and_then(|v| v.as_u64()) {
            println!("✅ Using cached degree {} for exp_id {}", degree, exp_id);
            visited.remove(&exp_id);
            return degree as usize;
        }
    }

    let deg = match exp.get("op").and_then(|v| v.as_str()) {
        Some("exp") => {
            let id = exp.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            println!("🌀 Recursing into 'exp' op for id {}", id);
            if im_exps.contains(&id) {
                println!("✅ id {} found in im_exps, returning 1", id);
                1
            } else {
                calculate_exp_deg_helper(expressions, id, im_exps, cache_values, visited)
            }
        }
        Some("x") | Some("const") | Some("cm") | Some("custom") => {
            println!("📌 Returning 1 for constant-like op: {}", exp["op"]);
            1
        }
        Some("Zi") => {
            if exp.get("boundary") == Some(&json!("everyRow")) {
                println!("📌 Returning 0 for 'Zi' op with 'everyRow' boundary");
                0
            } else {
                println!("📌 Returning 1 for 'Zi' op with other boundary");
                1
            }
        }
        Some("number")
        | Some("public")
        | Some("challenge")
        | Some("eval")
        | Some("airgroupvalue")
        | Some("airvalue") => {
            println!("📌 Returning 0 for known zero-degree op: {}", exp["op"]);
            0
        }
        Some("neg") => {
            if let Some(values) = exp.get("values").and_then(|v| v.as_array()) {
                if let Some(first) = values.first() {
                    let id = first.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    println!("🌀 Recursing into 'neg' op for id {}", id);
                    calculate_exp_deg_helper(expressions, id, im_exps, cache_values, visited)
                } else {
                    println!("⚠️ 'neg' op missing values, returning 0");
                    0
                }
            } else {
                println!("⚠️ 'neg' op has no values array, returning 0");
                0
            }
        }
        Some("add") | Some("sub") | Some("mul") => {
            let empty_vec = vec![]; // Fix borrowing issue
            let values = exp.get("values").and_then(|v| v.as_array()).unwrap_or(&empty_vec);

            if values.len() < 2 {
                println!("⚠️ Binary op '{}' missing operands, returning 0", exp["op"]);
                return 0;
            }

            let lhs_id = values[0].get("id").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let rhs_id = values[1].get("id").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

            println!("🌀 Processing '{}' op: exp_id={} -> lhs_id={}, rhs_id={}", exp["op"], exp_id, lhs_id, rhs_id);

            let lhs_deg = calculate_exp_deg_helper(expressions, lhs_id, im_exps, cache_values, visited);
            let rhs_deg = calculate_exp_deg_helper(expressions, rhs_id, im_exps, cache_values, visited);

            if exp["op"] == "mul" {
                println!("📌 Returning {} + {} for 'mul' op", lhs_deg, rhs_deg);
                lhs_deg + rhs_deg
            } else {
                println!("📌 Returning max({}, {}) for '{}' op", lhs_deg, rhs_deg, exp["op"]);
                lhs_deg.max(rhs_deg)
            }
        }
        _ => panic!("❌ Exp op not defined: {:?}", exp),
    };

    if cache_values {
        println!("📝 Caching degree {} for exp_id {}", deg, exp_id);
        expressions[exp_id]["degree_"] = json!(deg);
    }

    visited.remove(&exp_id);
    println!("✅ Leaving calculate_exp_deg for exp_id {} with degree {}", exp_id, deg);
    deg
}
