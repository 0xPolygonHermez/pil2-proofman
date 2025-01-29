use serde_json::{json, Value};
use std::collections::HashMap;

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
                name.push_str(&lengths.iter().map(|len| format!("[{}]", len)).collect::<String>());
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
                if row_offset > 0 {
                    name.push('\'');
                    if row_offset > 1 {
                        name.push_str(&row_offset.to_string());
                    }
                } else if row_offset < 0 {
                    name = format!("'{}{}", row_offset.abs(), name);
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
                    
                    _get_exp_dim(&expressions[id], expressions)
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
