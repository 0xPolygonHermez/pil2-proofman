use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::Write;

/// Prints a formatted expression recursively.
pub fn print_expressions(
    res: &HashMap<String, Value>,
    exp: &Value,
    expressions: &[Value],
    is_constraint: bool,
) -> String {
    match exp.get("op").and_then(Value::as_str) {
        Some("exp") => {
            if exp.get("line").is_none() {
                let id = exp["id"].as_u64().expect("Missing 'id' in exp") as usize;
                let line = print_expressions(res, &expressions[id], expressions, is_constraint);
                return line;
            }
            exp["line"].as_str().unwrap_or("").to_string()
        }
        Some("add") | Some("mul") | Some("sub") => {
            let lhs = print_expressions(res, &exp["values"][0], expressions, is_constraint);
            let rhs = print_expressions(res, &exp["values"][1], expressions, is_constraint);
            let op = match exp["op"].as_str().unwrap() {
                "add" => " + ",
                "sub" => " - ",
                "mul" => " * ",
                _ => unreachable!(),
            };
            format!("({lhs}{op}{rhs})")
        }
        Some("neg") => {
            let value = print_expressions(res, &exp["values"][0], expressions, is_constraint);
            format!("-{}", value)
        }
        Some("number") => exp["value"].as_str().unwrap_or("").to_string(),
        Some("const") | Some("cm") | Some("custom") => {
            let id = exp["id"].as_u64().expect("Missing 'id' in exp") as usize;
            let col = match exp["op"].as_str().unwrap() {
                "const" => &res["constPolsMap"][id],
                "cm" => &res["cmPolsMap"][id],
                "custom" => {
                    let commit_id = exp["commitId"].as_u64().expect("Missing 'commitId' in custom") as usize;
                    &res["customCommitsMap"][commit_id][id]
                }
                _ => unreachable!(),
            };

            if col.get("imPol").and_then(Value::as_bool).unwrap_or(false) && !is_constraint {
                let exp_id = col["expId"].as_u64().expect("Missing 'expId' in col") as usize;
                return print_expressions(res, &expressions[exp_id], expressions, false);
            }

            let mut name = col["name"].as_str().unwrap_or("").to_string();
            if let Some(lengths) = col.get("lengths").and_then(Value::as_array) {
                lengths.iter().fold(&mut name, |name, len| {
                    let _ = write!(name, "[{}]", len);
                    name
                });
            }

            if col.get("imPol").and_then(Value::as_bool).unwrap_or(false) {
                let count = res["cmPolsMap"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .take(id)
                    .filter(|w| w.get("imPol").and_then(Value::as_bool).unwrap_or(false))
                    .count();
                name.push_str(&count.to_string());
            }

            if let Some(row_offset) = exp.get("rowOffset").and_then(Value::as_i64) {
                match row_offset.cmp(&0) {
                    std::cmp::Ordering::Greater => {
                        name.push('\'');
                        if row_offset > 1 {
                            name.push_str(&row_offset.to_string());
                        }
                    }
                    std::cmp::Ordering::Less => {
                        name = format!("'{}{}", row_offset.abs(), name);
                    }
                    std::cmp::Ordering::Equal => {} // Do nothing when row_offset == 0
                }
            }
            name
        }
        Some("public") => {
            res["publicsMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap_or("").to_string()
        }
        Some("airvalue") => {
            res["airValuesMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap_or("").to_string()
        }
        Some("airgroupvalue") => {
            res["airgroupValuesMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap_or("").to_string()
        }
        Some("challenge") => {
            res["challengesMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap_or("").to_string()
        }
        Some("x") => "x".to_string(),
        Some("Zi") => "zh".to_string(),
        _ => panic!("Unknown op: {:?}", exp["op"]),
    }
}

// MATCHES JS
/// Gets the dimension of the expression at `exp_id`, mutably caching it
/// in expressions[exp_id]["dim"] when `op == "exp"`.
pub fn get_exp_dim(expressions: &mut [Value], exp_id: usize) -> usize {
    // 1) Temporarily take out the expression from the array to avoid aliasing
    let mut exp = std::mem::take(&mut expressions[exp_id]);

    // 2) Compute dimension
    let dim = get_exp_dim_inner(&mut exp, expressions);

    // 3) Put the expression back
    expressions[exp_id] = exp;

    dim
}

fn get_exp_dim_inner(exp: &mut Value, expressions: &mut [Value]) -> usize {
    // If exp["dim"] is already defined and non-null, just return it.
    if let Some(dim_val) = exp.get("dim") {
        if !dim_val.is_null() {
            return dim_val.as_u64().unwrap_or(1) as usize;
        }
    }

    // Otherwise, match on exp["op"]
    match exp.get("op").and_then(|op| op.as_str()) {
        // ---------------- add / sub / mul ------------------
        // In JS, we do `Math.max(...exp.values.map(...))` without caching in `exp.dim`.
        Some("add") | Some("sub") | Some("mul") => {
            let values = exp["values"].as_array_mut().expect("Expected 'values' to be an array");
            let max_dim = values.iter_mut().map(|child| get_exp_dim_inner(child, expressions)).max().unwrap_or(1);
            max_dim
        }

        // ---------------- exp ------------------
        // In JS: we do `exp.dim = _getExpDim(expressions[exp.id])` then return `exp.dim`.
        Some("exp") => {
            let child_id = exp["id"].as_u64().unwrap_or(0) as usize;

            // Temporarily remove the child from the array
            let mut child_exp = std::mem::take(&mut expressions[child_id]);

            // Recursively compute child's dimension
            let child_dim = get_exp_dim_inner(&mut child_exp, expressions);

            // Put child back
            expressions[child_id] = child_exp;

            // Cache child's dim into exp["dim"]
            exp["dim"] = Value::from(child_dim);

            child_dim
        }

        // ---------------- cm / custom ------------------
        // JS: returns `exp.dim || 1` but does not store it back
        Some("cm") | Some("custom") => exp.get("dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize,

        // ---------------- const / number / public / x / Zi => 1 ----------------
        Some("const") | Some("number") | Some("public") | Some("x") | Some("Zi") => 1,

        // ---------------- challenge / eval / xDivXSubXi => 3 ----------------
        Some("challenge") | Some("eval") | Some("xDivXSubXi") => 3,

        // ---------------- unknown op => panic ----------------
        other => panic!("Exp op not defined: {:?}", other),
    }
}

/// Adds metadata information to an expression.
pub fn add_info_expressions(expressions: &mut [Value], exp_id: usize) {
    if expressions[exp_id].get("expDeg").is_some() {
        return;
    }

    // Handle "next" field
    if let Some(next) = expressions[exp_id].get("next") {
        let row_offset = if next.as_bool().unwrap_or(false) { 1 } else { 0 };
        expressions[exp_id]["rowOffset"] = json!(row_offset);
        expressions[exp_id].as_object_mut().unwrap().remove("next");
    }

    let op = expressions[exp_id].get("op").and_then(|op| op.as_str()).unwrap_or("").to_string();

    match op.as_str() {
        "exp" => {
            let id = expressions[exp_id]["id"].as_u64().unwrap_or(0) as usize;

            // Prevent self-reference processing
            if id == exp_id {
                return;
            }

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
        "xDivXSubXi" => {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(1);
        }
        "challenge" | "eval" => {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(0);
            exp["dim"] = json!(3);
        }
        "airgroupvalue" | "proofvalue" => {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(0);
            exp["dim"] = json!(3);
        }
        "airvalue" => {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(0);
            if exp.get("dim").is_none() {
                exp["dim"] = json!(if exp["stage"] != json!(1) { 3 } else { 1 });
            }
        }
        "public" => {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(0);
            exp["stage"] = json!(1);
            if exp.get("dim").is_none() {
                exp["dim"] = json!(1);
            }
        }
        "number" | "Zi" if expressions[exp_id].get("boundary") == Some(&json!("everyRow")) => {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(0);
            exp["stage"] = json!(0);
            if exp.get("dim").is_none() {
                exp["dim"] = json!(1);
            }
        }
        "add" | "sub" | "mul" | "neg" => {
            let empty = vec![];
            let values = expressions[exp_id]["values"].as_array().unwrap_or(&empty);

            if values.is_empty() {
                panic!("Operation `{}` has no values!", op);
            }

            let lhs_id = values[0]["id"].as_u64().unwrap_or(0) as usize;
            let rhs_id = values.get(1).map_or(0, |v| v["id"].as_u64().unwrap_or(0)) as usize;

            // Prevent infinite recursion due to self-referencing expressions
            if lhs_id == exp_id || rhs_id == exp_id {
                return;
            }

            add_info_expressions(expressions, lhs_id);
            add_info_expressions(expressions, rhs_id);

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

    let op = expressions[exp_id].get("op").and_then(Value::as_str).unwrap_or("");

    match op {
        "exp" => {
            let id = expressions[exp_id]["id"].as_u64().unwrap_or(0) as usize;
            add_info_expressions_symbols(symbols, expressions, id);

            let cloned_id_data = expressions[id].clone();
            expressions[exp_id]["symbols"] = cloned_id_data["symbols"].clone();

            if cloned_id_data.get("imPol").and_then(Value::as_bool).unwrap_or(false) {
                if let Some(exp_sym) = symbols.iter().find(|s| {
                    s.get("type").and_then(Value::as_str) == Some("witness") && s.get("expId") == Some(&json!(id))
                }) {
                    let mut existing_symbols = expressions[exp_id]["symbols"].as_array().cloned().unwrap_or_default();

                    if !existing_symbols.iter().any(|s| {
                        s.get("op").and_then(Value::as_str) == Some("cm")
                            && s.get("stage") == exp_sym.get("stage")
                            && s.get("stageId") == exp_sym.get("stageId")
                            && s.get("id") == exp_sym.get("polId")
                    }) {
                        existing_symbols.push(json!({
                            "op": "cm",
                            "stage": exp_sym["stage"],
                            "stageId": exp_sym["stageId"],
                            "id": exp_sym["polId"],
                            "rowsOffsets": [0]
                        }));
                    }
                    expressions[exp_id]["symbols"] = json!(existing_symbols);
                }
            }
        }
        "cm" | "const" | "custom" => {
            if expressions[exp_id].get("symbols").is_none() {
                let mut symbol_obj = json!({
                    "op": op,
                    "stage": expressions[exp_id]["stage"],
                    "id": expressions[exp_id]["id"],
                    "rowsOffsets": expressions[exp_id]["rowsOffsets"]
                });

                if op == "cm" {
                    if expressions[exp_id]["stageId"].is_null() {
                        if let Some(sym) = symbols.iter().find(|s| {
                            s.get("type").and_then(Value::as_str) == Some("witness")
                                && s.get("polId") == expressions[exp_id].get("id")
                        }) {
                            symbol_obj["stageId"] = sym["stageId"].clone();
                        }
                    }
                } else if op == "custom" {
                    symbol_obj["commitId"] = expressions[exp_id]["commitId"].clone();
                }

                expressions[exp_id]["symbols"] = json!([symbol_obj]);
            }
        }
        "add" | "sub" | "mul" | "neg" => {
            let empty_vec = Vec::new(); // Persistent empty Vec
            let values_array = expressions[exp_id]["values"].as_array().unwrap_or(&empty_vec);
            let values = values_array.to_vec(); // Clone the array to avoid borrow conflicts

            if values.is_empty() {
                return;
            }

            let lhs = &values[0];
            let rhs = values.get(1);

            add_info_expressions_symbols(symbols, expressions, lhs["id"].as_u64().unwrap_or(0) as usize);
            if let Some(rhs_value) = rhs {
                add_info_expressions_symbols(symbols, expressions, rhs_value["id"].as_u64().unwrap_or(0) as usize);
            }

            let lhs_symbols = extract_symbols(symbols, lhs);
            let rhs_symbols = rhs.map_or(vec![], |rhs_val| extract_symbols(symbols, rhs_val));

            let mut unique_symbols = merge_symbols(lhs_symbols, rhs_symbols);

            // Sort by stage, op, and id/stageId
            unique_symbols.sort_by(|a, b| {
                a["stage"]
                    .as_u64()
                    .cmp(&b["stage"].as_u64())
                    .then_with(|| a["op"].as_str().cmp(&b["op"].as_str()).reverse())
                    .then_with(|| {
                        if ["const", "airgroupvalue", "airvalue", "public"].contains(&a["op"].as_str().unwrap_or("")) {
                            a["id"].as_u64().cmp(&b["id"].as_u64())
                        } else {
                            a["stageId"].as_u64().cmp(&b["stageId"].as_u64())
                        }
                    })
            });

            expressions[exp_id]["symbols"] = json!(unique_symbols);
        }
        _ => {}
    }
}

/// Merges two symbol lists, ensuring uniqueness.
fn merge_symbols(lhs_symbols: Vec<Value>, rhs_symbols: Vec<Value>) -> Vec<Value> {
    let mut unique_symbols = lhs_symbols;

    for rhs_symbol in rhs_symbols {
        if let Some(existing_symbol) = unique_symbols
            .iter_mut()
            .find(|s| s["op"] == rhs_symbol["op"] && s["id"] == rhs_symbol["id"] && s["stage"] == rhs_symbol["stage"])
        {
            if let (Some(existing_offsets), Some(rhs_offsets)) =
                (existing_symbol["rowsOffsets"].as_array_mut(), rhs_symbol["rowsOffsets"].as_array())
            {
                let mut all_offsets: Vec<u64> = existing_offsets
                    .iter()
                    .filter_map(Value::as_u64)
                    .chain(rhs_offsets.iter().filter_map(Value::as_u64))
                    .collect();
                all_offsets.sort_unstable();
                all_offsets.dedup();
                *existing_offsets = all_offsets.into_iter().map(Value::from).collect();
            }
        } else {
            unique_symbols.push(rhs_symbol);
        }
    }

    unique_symbols
}

/// Extracts symbols from an expression.
fn extract_symbols(symbols: &[HashMap<String, Value>], exp: &Value) -> Vec<Value> {
    let mut result = vec![];

    let op = exp["op"].as_str().unwrap_or("");

    match op {
        "cm" | "challenge" => {
            let mut sym_obj = json!({
                "op": op,
                "stage": exp["stage"],
                "id": exp["id"],
            });

            if exp["stageId"].is_null() {
                if let Some(sym) = symbols.iter().find(|s| {
                    s.get("type").and_then(Value::as_str) == Some("witness") && s.get("polId") == exp.get("id")
                }) {
                    sym_obj["stageId"] = sym["stageId"].clone();
                }
            } else {
                sym_obj["stageId"] = exp["stageId"].clone();
            }

            if op == "cm" {
                sym_obj["rowsOffsets"] = exp["rowsOffsets"].clone();
            }

            result.push(sym_obj);
        }
        "const" => {
            result.push(json!({
                "op": op,
                "stage": exp["stage"],
                "id": exp["id"],
                "rowsOffsets": exp["rowsOffsets"],
            }));
        }
        "custom" => {
            result.push(json!({
                "op": op,
                "stage": exp["stage"],
                "id": exp["id"],
                "stageId": exp["stageId"],
                "commitId": exp["commitId"],
                "rowsOffsets": exp["rowsOffsets"],
            }));
        }
        "public" | "airgroupvalue" | "airvalue" => {
            result.push(json!({
                "op": op,
                "stage": exp["stage"],
                "id": exp["id"],
            }));
        }
        _ => {
            if let Some(symbols) = exp["symbols"].as_array() {
                result.extend(symbols.clone());
            }
        }
    }

    result
}
