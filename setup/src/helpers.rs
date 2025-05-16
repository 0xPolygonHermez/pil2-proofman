use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
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

// MATCHES JS
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

/// Recursively fills in `expDeg`, `dim`, `stage` and `rowsOffsets`
/// on `expressions[exp_id]`, matching the JS `addInfoExpressions`.
pub fn add_info_expressions(expressions: &mut [Value], exp_id: usize) {
    // 1) Fast exit if already has expDeg
    if expressions[exp_id].get("expDeg").is_some() {
        return;
    }

    // 2) Handle "next" → rowOffset
    if let Some(next) = expressions[exp_id].get("next") {
        let ro = if next.as_bool().unwrap_or(false) { 1 } else { 0 };
        expressions[exp_id]["rowOffset"] = json!(ro);
        expressions[exp_id].as_object_mut().unwrap().remove("next");
    }

    // 3) Pull out op *by value* so we can reborrow later
    let op = expressions[exp_id].get("op").and_then(Value::as_str).unwrap_or("").to_string();

    // ---- exp branch ----
    if op == "exp" {
        let child_id = expressions[exp_id]["id"].as_u64().unwrap() as usize;
        if child_id != exp_id {
            // recurse first
            add_info_expressions(expressions, child_id);

            // now clone the child so we can write back safely
            let child = expressions[child_id].clone();
            let child_deg = child["expDeg"].clone();
            let child_rows = child["rowsOffsets"].clone();
            let child_dim = child.get("dim").cloned();
            let child_stage = child.get("stage").cloned();
            let child_op = child["op"].as_str().unwrap_or("").to_string();

            // a tiny mutable borrow for writing
            {
                let exp = &mut expressions[exp_id];
                exp["expDeg"] = child_deg;
                exp["rowsOffsets"] = child_rows;
                if exp.get("dim").is_none() {
                    exp["dim"] = child_dim.unwrap();
                }
                if exp.get("stage").is_none() {
                    exp["stage"] = child_stage.unwrap();
                }
                // if leaf‐type, wholesale replace
                if ["cm", "const", "custom"].contains(&child_op.as_str()) {
                    *exp = child;
                }
            }
        }
        return;
    }

    // ---- leaf‐type branch ----
    if ["x", "cm", "custom", "const"].contains(&op.as_str())
        || (op == "Zi" && expressions[exp_id].get("boundary").and_then(Value::as_str) != Some("everyRow"))
    {
        // write all at once
        let stage = if op == "cm" { 1 } else { 0 };
        let ro_opt = expressions[exp_id].get("rowOffset").cloned();
        {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(1);
            if exp.get("stage").is_none() || op == "const" {
                exp["stage"] = json!(stage);
            }
            if exp.get("dim").is_none() {
                exp["dim"] = json!(1);
            }
            if let Some(ro) = ro_opt {
                exp["rowsOffsets"] = json!([ro]);
            }
        }
        return;
    }

    // ---- xDivXSubXi ----
    if op == "xDivXSubXi" {
        expressions[exp_id]["expDeg"] = json!(1);
        return;
    }

    // ---- challenge/eval ----
    if ["challenge", "eval"].contains(&op.as_str()) {
        expressions[exp_id]["expDeg"] = json!(0);
        expressions[exp_id]["dim"] = json!(3);
        return;
    }

    // ---- airgroupvalue/proofvalue ----
    if ["airgroupvalue", "proofvalue"].contains(&op.as_str()) {
        let stage = expressions[exp_id].get("stage").and_then(Value::as_u64).unwrap_or(0);
        expressions[exp_id]["expDeg"] = json!(0);
        expressions[exp_id]["dim"] = json!((stage != 1) as u8 * 2 + 1); // 3 if stage!=1 else 1
        return;
    }

    // ---- airvalue ----
    if op == "airvalue" {
        let stage = expressions[exp_id].get("stage").and_then(Value::as_u64).unwrap_or(0);
        {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(0);
            if exp.get("dim").is_none() {
                exp["dim"] = json!((stage != 1) as u8 * 2 + 1);
            }
        }
        return;
    }

    // ---- public ----
    if op == "public" {
        {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(0);
            exp["stage"] = json!(1);
            if exp.get("dim").is_none() {
                exp["dim"] = json!(1);
            }
        }
        return;
    }

    // ---- number or Zi @ everyRow ----
    if op == "number" || (op == "Zi" && expressions[exp_id].get("boundary").and_then(Value::as_str) == Some("everyRow"))
    {
        {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(0);
            exp["stage"] = json!(0);
            if exp.get("dim").is_none() {
                exp["dim"] = json!(1);
            }
        }
        return;
    }

    // ---- add | sub | mul | neg ----
    if ["add", "sub", "mul", "neg"].contains(&op.as_str()) {
        // 1) Clone the `values` array so we don't hold any borrow on `expressions`
        let vals_clone: Vec<Value> = expressions[exp_id].get("values").and_then(Value::as_array).unwrap().clone();

        // 2) Handle `neg` → rewrite to mul
        if op == "neg" {
            let original = vals_clone[0].clone();
            {
                let exp = &mut expressions[exp_id];
                exp["op"] = json!("mul");
                exp["values"] = json!([
                    { "op":"number", "value":"-1", "expDeg":0, "stage":0, "dim":1 },
                    original
                ]);
            }
        }

        // 3) Zero-fold: turn add 0 or sub 0 into mul 1
        if op == "add" {
            // use our clone to inspect the old values
            if vals_clone[0]["op"] == "number" && vals_clone[0]["value"] == "0" {
                let exp = &mut expressions[exp_id];
                exp["op"] = json!("mul");
                exp["values"][0]["value"] = json!("1");
            }
            if vals_clone[1]["op"] == "number" && vals_clone[1]["value"] == "0" {
                let exp = &mut expressions[exp_id];
                exp["op"] = json!("mul");
                exp["values"][1]["value"] = json!("1");
            }
        }

        // 4) Now do the recursive add/sub/mul logic without any overlapping borrows…
        let lhs_id = vals_clone[0]["id"].as_u64().unwrap() as usize;
        let rhs_id = vals_clone[1]["id"].as_u64().unwrap() as usize;
        if lhs_id != exp_id {
            add_info_expressions(expressions, lhs_id);
        }
        if rhs_id != exp_id {
            add_info_expressions(expressions, rhs_id);
        }
        let lhs = &expressions[lhs_id];
        let rhs = &expressions[rhs_id];

        let lhs_deg = lhs["expDeg"].as_u64().unwrap();
        let rhs_deg = rhs["expDeg"].as_u64().unwrap();
        let exp_deg = if expressions[exp_id]["op"] == json!("mul") { lhs_deg + rhs_deg } else { lhs_deg.max(rhs_deg) };

        let lhs_dim = lhs["dim"].as_u64().unwrap();
        let rhs_dim = rhs["dim"].as_u64().unwrap();
        let dim = lhs_dim.max(rhs_dim);

        let lhs_st = lhs["stage"].as_u64().unwrap();
        let rhs_st = rhs["stage"].as_u64().unwrap();
        let stage = lhs_st.max(rhs_st);

        // union rowsOffsets
        let mut set = std::collections::HashSet::new();
        for v in lhs["rowsOffsets"].as_array().unwrap() {
            set.insert(v.as_u64().unwrap());
        }
        for v in rhs["rowsOffsets"].as_array().unwrap() {
            set.insert(v.as_u64().unwrap());
        }
        let rows: Vec<_> = set.into_iter().collect();

        // 5) Single short mutable borrow to write back all fields
        {
            let exp = &mut expressions[exp_id];
            exp["expDeg"] = json!(exp_deg);
            exp["dim"] = json!(dim);
            exp["stage"] = json!(stage);
            exp["rowsOffsets"] = json!(rows);
        }
        return;
    }

    panic!("Exp op not defined: {}", op);
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
