use serde_json::{json, Value};
use std::collections::HashMap;

/// Gets the dimensionality of an expression by its ID.
pub fn get_exp_dim(expressions: &[Value], exp_id: usize) -> usize {
    fn _get_exp_dim(exp: &Value, expressions: &[Value]) -> usize {
        match exp.get("dim") {
            Some(dim) if !dim.is_null() => dim.as_u64().unwrap_or(1) as usize,
            _ => match exp.get("op").and_then(|op| op.as_str()) {
                Some("add") | Some("sub") | Some("mul") => exp["values"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| _get_exp_dim(v, expressions))
                    .max()
                    .unwrap_or(1),
                Some("exp") => {
                    let id = exp["id"].as_u64().unwrap_or(0) as usize;
                    let dim = _get_exp_dim(&expressions[id], expressions);
                    dim
                }
                Some("cm") | Some("custom") => exp.get("dim").and_then(|d| d.as_u64()).unwrap_or(1) as usize,
                Some("const") | Some("number") | Some("public") | Some("x") | Some("Zi") => 1,
                Some("challenge") | Some("eval") | Some("xDivXSubXi") => 3,
                _ => panic!("Exp op not defined: {}", exp.get("op").unwrap_or(&json!("unknown"))),
            },
        }
    }

    _get_exp_dim(&expressions[exp_id], expressions)
}

/// Adds metadata information to an expression.
pub fn add_info_expressions(expressions: &mut [Value], exp_id: usize) {
    if expressions[exp_id].get("expDeg").is_some() {
        return;
    }

    if let Some(next) = expressions[exp_id].get("next") {
        let row_offset = if next.as_bool().unwrap_or(false) { 1 } else { 0 };
        expressions[exp_id]["rowOffset"] = json!(row_offset);
        expressions[exp_id].as_object_mut().unwrap().remove("next");
    }

    // Extract `op` and handle logic outside the borrow scope
    let op = expressions[exp_id].get("op").and_then(|op| op.as_str()).unwrap_or("").to_string();

    match op.as_str() {
        "exp" => {
            let id = expressions[exp_id]["id"].as_u64().unwrap_or(0) as usize;
            add_info_expressions(expressions, id);

            let cloned_id_data = expressions[id].clone();
            let exp = &mut expressions[exp_id];

            exp["expDeg"] = cloned_id_data["expDeg"].clone();
            exp["rowsOffsets"] = cloned_id_data["rowsOffsets"].clone();
            if exp.get("dim").is_none() {
                exp["dim"] = cloned_id_data["dim"].clone();
            }
            if exp.get("stage").is_none() {
                exp["stage"] = cloned_id_data["stage"].clone();
            }
            if ["cm", "const", "custom"].contains(&cloned_id_data["op"].as_str().unwrap_or("")) {
                *exp = cloned_id_data;
            }
        }
        "x" | "cm" | "custom" | "const" | "Zi" if expressions[exp_id].get("boundary") != Some(&json!("everyRow")) => {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(1);
            if exp["stage"].is_null() || exp["op"] == json!("const") {
                exp["stage"] = json!(if exp["op"] == json!("cm") { 1 } else { 0 });
            }
            if exp.get("dim").is_none() {
                exp["dim"] = json!(1);
            }
            if let Some(row_offset) = exp.get("rowOffset") {
                exp["rowsOffsets"] = json!([row_offset.clone()]);
            }
        }
        "add" | "sub" | "mul" | "neg" => {
            let lhs_id = expressions[exp_id]["values"][0]["id"].as_u64().unwrap_or(0) as usize;
            let rhs_id = expressions[exp_id]["values"][1]["id"].as_u64().unwrap_or(0) as usize;

            add_info_expressions(expressions, lhs_id);
            add_info_expressions(expressions, rhs_id);

            // Clone data outside of borrow scope
            let lhs = expressions[lhs_id].clone();
            let rhs = expressions[rhs_id].clone();

            let lhs_deg = lhs["expDeg"].as_u64().unwrap_or(0);
            let rhs_deg = rhs["expDeg"].as_u64().unwrap_or(0);
            let exp_deg = if op == "mul" { lhs_deg + rhs_deg } else { lhs_deg.max(rhs_deg) };

            let lhs_dim = lhs["dim"].as_u64().unwrap_or(1);
            let rhs_dim = rhs["dim"].as_u64().unwrap_or(1);
            let dim = lhs_dim.max(rhs_dim);

            let lhs_stage = lhs["stage"].as_u64().unwrap_or(0);
            let rhs_stage = rhs["stage"].as_u64().unwrap_or(0);
            let stage = lhs_stage.max(rhs_stage);

            let lhs_offsets = lhs["rowsOffsets"].as_array().cloned().unwrap_or_default();
            let rhs_offsets = rhs["rowsOffsets"].as_array().cloned().unwrap_or_default();
            let combined_offsets: Vec<_> = lhs_offsets.into_iter().chain(rhs_offsets).collect();

            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(exp_deg);
            exp["dim"] = json!(dim);
            exp["stage"] = json!(stage);
            exp["rowsOffsets"] = json!(combined_offsets);
        }
        _ => panic!("Exp op not defined: {}", expressions[exp_id].get("op").unwrap_or(&json!("unknown"))),
    }
}

/// Adds symbol-related metadata to expressions.
pub fn add_info_expressions_symbols(symbols: &[HashMap<String, Value>], expressions: &mut [Value], exp_id: usize) {
    if expressions[exp_id].get("symbols").is_some() {
        return;
    }

    let op = expressions[exp_id].get("op").and_then(|op| op.as_str()).unwrap_or("").to_string();

    match op.as_str() {
        "exp" => {
            let id = expressions[exp_id]["id"].as_u64().unwrap_or(0) as usize;
            add_info_expressions_symbols(symbols, expressions, id);

            let cloned_id_data = expressions[id].clone();
            expressions[exp_id]["symbols"] = cloned_id_data["symbols"].clone();
        }
        "cm" | "const" | "custom" => {
            if expressions[exp_id].get("symbols").is_none() {
                expressions[exp_id]["symbols"] = json!([{
                    "op": op,
                    "stage": expressions[exp_id]["stage"],
                    "id": expressions[exp_id]["id"],
                    "rowsOffsets": expressions[exp_id]["rowsOffsets"]
                }]);
            }
        }
        _ => {}
    }
}
