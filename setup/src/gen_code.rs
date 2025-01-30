use serde_json::{json, Value};
use std::collections::HashMap;

/// Context struct for handling expression transformation
#[derive(Debug, Clone, Default)]
pub struct CodeGenContext {
    pub stage: usize,
    pub calculated: HashMap<usize, HashMap<i64, Value>>,
    pub symbols_used: Vec<Value>,
    pub tmp_used: usize,
    pub code: Vec<Value>,
    pub dom: String,
    pub air_id: Value,
    pub airgroup_id: Value,
    pub opening_points: Vec<i64>,
    pub verifier_evaluations: bool,
    pub ev_map: Vec<Value>,
    pub exp_map: HashMap<i64, HashMap<usize, usize>>, // Tracks expression remapping
}

/// Generates expressions code while keeping structure as close as possible to the original JS implementation.
pub fn generate_expressions_code(res: &Value, symbols: &[Value], expressions: &[Value]) -> Vec<Value> {
    let mut expressions_code = Vec::new();

    for (j, exp) in expressions.iter().enumerate() {
        if !exp["keep"].as_bool().unwrap_or(false)
            && !exp["imPol"].as_bool().unwrap_or(false)
            && ![res["cExpId"].as_u64().unwrap_or(u64::MAX), res["friExpId"].as_u64().unwrap_or(u64::MAX)]
                .contains(&(j as u64))
        {
            continue;
        }

        let dom = if j as u64 == res["cExpId"].as_u64().unwrap_or(u64::MAX)
            || j as u64 == res["friExpId"].as_u64().unwrap_or(u64::MAX)
        {
            "ext".to_string()
        } else {
            "n".to_string()
        };

        let mut ctx = CodeGenContext {
            stage: exp["stage"].as_u64().unwrap_or(0) as usize,
            calculated: HashMap::new(),
            symbols_used: Vec::new(),
            tmp_used: 0,
            code: Vec::new(),
            dom,
            air_id: res["airId"].clone(),
            airgroup_id: res["airgroupId"].clone(),
            opening_points: vec![],
            verifier_evaluations: false,
            ev_map: Vec::new(),
            exp_map: HashMap::new(),
        };

        if j as u64 == res["friExpId"].as_u64().unwrap_or(u64::MAX) {
            ctx.opening_points =
                res["openingPoints"].as_array().unwrap_or(&vec![]).iter().filter_map(|v| v.as_i64()).collect();
        }

        if j as u64 == res["cExpId"].as_u64().unwrap_or(u64::MAX) {
            for symbol in symbols {
                if symbol["imPol"].as_bool().unwrap_or(false) {
                    let exp_id = symbol["expId"].as_u64().unwrap_or(0) as usize;
                    ctx.calculated.insert(exp_id, HashMap::new());
                    for opening_point in &ctx.opening_points {
                        ctx.calculated.get_mut(&exp_id).unwrap().insert(*opening_point, json!({ "used": true }));
                    }
                }
            }
        }

        pil_code_gen(&mut ctx, symbols, expressions, j, 0);
        let mut exp_info = build_code(&mut ctx);

        if j as u64 == res["cExpId"].as_u64().unwrap_or(u64::MAX) {
            if let Some(last) = exp_info["code"].as_array_mut().and_then(|c| c.last_mut()) {
                last["dest"] = json!({ "type": "q", "id": 0, "dim": res["qDim"] });
            }
        }

        if j as u64 == res["friExpId"].as_u64().unwrap_or(u64::MAX) {
            if let Some(last) = exp_info["code"].as_array_mut().and_then(|c| c.last_mut()) {
                last["dest"] = json!({ "type": "f", "id": 0, "dim": 3 });
            }
        }

        exp_info["expId"] = json!(j);
        exp_info["stage"] = exp["stage"].clone();
        exp_info["dest"] = json!(null);
        exp_info["line"] = exp["line"].clone();

        expressions_code.push(exp_info);
    }

    expressions_code
}

/// Converts pilCodeGen logic to Rust
pub fn pil_code_gen(ctx: &mut CodeGenContext, symbols: &[Value], expressions: &[Value], exp_id: usize, prime: i64) {
    if ctx.calculated.get(&exp_id).and_then(|c| c.get(&prime)).is_some() {
        return;
    }

    let exp = &expressions[exp_id];
    calculate_deps(ctx, symbols, expressions, exp, prime);

    let mut code_ctx = CodeGenContext { exp_map: HashMap::new(), ..ctx.clone() };

    let ret_ref = eval_exp(&mut code_ctx, symbols, expressions, exp, prime);
    let mut r = json!({ "type": "exp", "prime": prime, "id": exp_id, "dim": exp["dim"] });

    if ret_ref["type"] == "tmp" {
        fix_commit_pol(&mut r, &mut code_ctx, symbols);
        if let Some(last) = code_ctx.code.last_mut() {
            last["dest"] = r.clone();
        }
        if r["type"] == "cm" {
            code_ctx.tmp_used -= 1;
        }
    } else {
        fix_commit_pol(&mut r, &mut code_ctx, symbols);
        code_ctx.code.push(json!({ "op": "copy", "dest": r, "src": [ret_ref] }));
    }

    ctx.code.extend(code_ctx.code);

    ctx.calculated.entry(exp_id).or_default().insert(prime, json!({ "cm": false, "tmpId": code_ctx.tmp_used }));

    if code_ctx.tmp_used > ctx.tmp_used {
        ctx.tmp_used = code_ctx.tmp_used;
    }
}

/// Evaluates an expression recursively
fn eval_exp(ctx: &mut CodeGenContext, symbols: &[Value], expressions: &[Value], exp: &Value, prime: i64) -> Value {
    let prime = exp.get("rowOffset").and_then(Value::as_i64).unwrap_or(prime);

    match exp["op"].as_str().unwrap_or("") {
        "add" | "sub" | "mul" => {
            let values: Vec<Value> = exp["values"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| eval_exp(ctx, symbols, expressions, v, prime))
                .collect();

            let max_dim = values.iter().map(|v| v["dim"].as_u64().unwrap_or(1)).max().unwrap_or(1);

            let r = json!({ "type": "tmp", "id": ctx.tmp_used, "dim": max_dim });
            ctx.tmp_used += 1;
            ctx.code.push(json!({ "op": exp["op"], "dest": r, "src": values }));
            r
        }
        "cm" | "const" | "custom" => {
            let mut r = json!({
                "type": exp["op"],
                "id": exp["id"],
                "prime": prime,
                "dim": exp["dim"]
            });

            if exp["op"] == "custom" {
                r["commitId"] = exp["commitId"].clone();
            }
            if ctx.verifier_evaluations {
                fix_eval(&mut r, ctx);
            }
            r
        }
        "exp" => {
            if let Some(op) = expressions.get(exp["id"].as_u64().unwrap_or(0) as usize).and_then(|e| e["op"].as_str()) {
                if ["cm", "const", "custom"].contains(&op) {
                    let expr = &expressions[exp["id"].as_u64().unwrap_or(0) as usize];
                    let mut r = json!({
                        "type": expr["op"],
                        "id": expr["id"],
                        "prime": prime,
                        "dim": expr["dim"]
                    });

                    if expr["op"] == "custom" {
                        r["commitId"] = expr["commitId"].clone();
                    }
                    if ctx.verifier_evaluations {
                        fix_eval(&mut r, ctx);
                    }
                    return r;
                }
            }

            let mut r = json!({
                "type": "exp",
                "expId": exp["id"],
                "id": exp["id"],
                "prime": prime,
                "dim": exp["dim"]
            });

            fix_commit_pol(&mut r, ctx, symbols);
            r
        }
        "challenge" => json!({
            "type": "challenge",
            "id": exp["id"],
            "stageId": exp["stageId"],
            "dim": exp["dim"],
            "stage": exp["stage"]
        }),
        "public" => json!({
            "type": "public",
            "id": exp["id"],
            "dim": 1
        }),
        "proofvalue" => json!({
            "type": "proofvalue",
            "id": exp["id"],
            "dim": 3
        }),
        "number" => {
            let num = exp["value"].as_str().unwrap_or("0").parse::<i128>().unwrap_or(0);
            let num = if num < 0 { num + 0xFFFFFFFF00000001 } else { num };
            json!({ "type": "number", "value": num.to_string(), "dim": 1 })
        }
        "eval" => json!({
            "type": "eval",
            "id": exp["id"],
            "dim": exp["dim"]
        }),
        "airgroupvalue" | "airvalue" => json!({
            "type": exp["op"],
            "id": exp["id"],
            "dim": exp["dim"],
            "airgroupId": exp["airgroupId"]
        }),
        "xDivXSubXi" => json!({
            "type": "xDivXSubXi",
            "id": exp["id"],
            "opening": exp["opening"],
            "dim": 3
        }),
        "Zi" => json!({
            "type": "Zi",
            "boundaryId": exp["boundaryId"],
            "dim": 1
        }),
        "x" => json!({
            "type": "x",
            "dim": 1
        }),
        _ => panic!("Invalid op: {}", exp["op"]),
    }
}

/// Calculates dependencies recursively
pub fn calculate_deps(ctx: &mut CodeGenContext, symbols: &[Value], expressions: &[Value], exp: &Value, prime: i64) {
    if exp["op"] == "exp" {
        let p = exp["rowOffset"].as_i64().unwrap_or(prime);
        pil_code_gen(ctx, symbols, expressions, exp["id"].as_u64().unwrap_or(0) as usize, p);
    } else if ["add", "sub", "mul"].contains(&exp["op"].as_str().unwrap_or("")) {
        if let Some(values) = exp["values"].as_array() {
            for v in values {
                calculate_deps(ctx, symbols, expressions, v, prime);
            }
        }
    }
}

/// Fixes the mapping of computed polynomials
pub fn fix_commit_pol(r: &mut Value, ctx: &mut CodeGenContext, symbols: &[Value]) {
    if let Some(symbol) = symbols.iter().find(|s| {
        s["type"] == "witness"
            && s["expId"] == r["id"]
            && s["airId"] == ctx.air_id
            && s["airgroupId"] == ctx.airgroup_id
    }) {
        if symbol["imPol"].as_bool().unwrap_or(false)
            && (ctx.dom == "ext"
                || (symbol["stage"].as_u64().unwrap_or(u64::MAX) <= ctx.stage as u64
                    && ctx
                        .calculated
                        .get(&(r["id"].as_u64().unwrap_or(0) as usize))
                        .and_then(|c| c.get(&r["prime"].as_i64().unwrap_or(0)))
                        .map(|v| v["cm"].as_bool().unwrap_or(false))
                        .unwrap_or(false)))
        {
            r["type"] = json!("cm");
            r["id"] = symbol["polId"].clone();
            r["dim"] = symbol["dim"].clone();
            if ctx.verifier_evaluations {
                fix_eval(r, ctx);
            }
        } else if !ctx.verifier_evaluations
            && ctx.dom == "n"
            && ctx
                .calculated
                .get(&(r["id"].as_u64().unwrap_or(0) as usize))
                .and_then(|c| c.get(&r["prime"].as_i64().unwrap_or(0)))
                .map(|v| v["cm"].as_bool().unwrap_or(false))
                .unwrap_or(false)
        {
            r["type"] = json!("cm");
            r["id"] = symbol["polId"].clone();
            r["dim"] = symbol["dim"].clone();
        }
    }
}

/// Builds the final code representation from context
pub fn build_code(ctx: &mut CodeGenContext) -> Value {
    // Extract exp_map and tmp_used to avoid multiple mutable borrows
    let mut exp_map = std::mem::take(&mut ctx.exp_map);
    let mut tmp_used = ctx.tmp_used;

    // Fix expression mappings
    for instr in &mut ctx.code {
        if let Some(src) = instr["src"].as_array_mut() {
            for src_val in src.iter_mut() {
                fix_expression(src_val, &mut exp_map, &mut tmp_used);
            }
        }
        fix_expression(&mut instr["dest"], &mut exp_map, &mut tmp_used);
    }

    // Restore exp_map and tmp_used to the context
    ctx.exp_map = exp_map;
    ctx.tmp_used = tmp_used;

    if ctx.verifier_evaluations {
        fix_dimensions_verifier(ctx);
    }

    // Construct the final code output
    let mut code = json!({
        "tmpUsed": ctx.tmp_used,
        "code": ctx.code
    });

    if !ctx.symbols_used.is_empty() {
        ctx.symbols_used.sort_by(|s1, s2| {
            let order = |s: &Value| match s["op"].as_str().unwrap_or("") {
                "const" => 0,
                "cm" => 1,
                "tmp" => 2,
                _ => 3,
            };

            let o1 = order(s1);
            let o2 = order(s2);
            if o1 != o2 {
                o1.cmp(&o2)
            } else {
                s1["stage"].as_u64().cmp(&s2["stage"].as_u64()).then_with(|| s1["id"].as_u64().cmp(&s2["id"].as_u64()))
            }
        });

        code["symbolsUsed"] = json!(ctx.symbols_used);
    }

    // Reset context state
    ctx.code.clear();
    ctx.calculated.clear();
    ctx.symbols_used.clear();
    ctx.tmp_used = 0;

    code
}

pub fn fix_expression(r: &mut Value, exp_map: &mut HashMap<i64, HashMap<usize, usize>>, tmp_used: &mut usize) {
    let prime = r["prime"].as_i64().unwrap_or(0);
    let entry = exp_map.entry(prime).or_default();
    let id = r["id"].as_u64().unwrap_or(0) as usize;

    if let std::collections::hash_map::Entry::Vacant(e) = entry.entry(id) {
        e.insert(*tmp_used);
        *tmp_used += 1;
    }

    r["type"] = json!("tmp");
    r["id"] = json!(entry[&id]);
}

pub fn fix_dimensions_verifier(ctx: &mut CodeGenContext) {
    let mut tmp_dim: HashMap<usize, usize> = HashMap::new();

    for code in &mut ctx.code {
        let op = code["op"].as_str().unwrap_or_default();

        // Ensure only valid operations are processed
        if !["add", "sub", "mul", "copy"].contains(&op) {
            panic!("Invalid op: {}", op);
        }

        // Ensure destination type is "tmp"
        if code["dest"]["type"] != "tmp" {
            panic!("Invalid dest type: {}", code["dest"]["type"]);
        }

        // Compute new dimension
        let new_dim = code["src"].as_array().unwrap_or(&vec![]).iter().map(|s| get_dim(s, &tmp_dim)).max().unwrap_or(1);

        let dest_id = code["dest"]["id"].as_u64().unwrap_or(0) as usize;
        tmp_dim.insert(dest_id, new_dim);
        code["dest"]["dim"] = json!(new_dim);
    }
}

/// Helper function to determine the dimension of a reference
pub fn get_dim(r: &Value, tmp_dim: &HashMap<usize, usize>) -> usize {
    let id = r["id"].as_u64().unwrap_or(0) as usize;

    if r["type"] == "tmp" {
        *tmp_dim.get(&id).unwrap_or(&1) // Default dimension is 1
    } else if r["type"] == "Zi" || r["type"] == "x" {
        3 // Special case for "Zi" and "x"
    } else {
        r["dim"].as_u64().unwrap_or(1) as usize // Default to 1 if not specified
    }
}

pub fn fix_eval(r: &mut Value, ctx: &mut CodeGenContext) {
    let prime = r["prime"].as_i64().unwrap_or(0);
    let opening_pos = ctx.opening_points.iter().position(|&p| p == prime).unwrap_or(0);

    if let Some(eval_index) = ctx.ev_map.iter().position(|e| {
        e["type"] == r["type"] && e["id"] == r["id"] && e["openingPos"].as_u64().unwrap_or(0) as usize == opening_pos
    }) {
        r["id"] = json!(eval_index);
        r["type"] = json!("eval");
        r["dim"] = json!(3);
    }
}

/// Generates debugging code for constraints
pub fn generate_constraints_debug_code(
    res: &Value,
    symbols: &[Value],
    constraints: &[Value],
    expressions: &[Value],
) -> Vec<Value> {
    let mut constraints_code = Vec::new();

    for constraint in constraints {
        let mut ctx = CodeGenContext {
            stage: constraint["stage"].as_u64().unwrap_or(0) as usize,
            calculated: HashMap::new(),
            symbols_used: Vec::new(),
            tmp_used: 0,
            code: Vec::new(),
            dom: "n".to_string(),
            air_id: res["airId"].clone(),
            airgroup_id: res["airgroupId"].clone(),
            ..Default::default()
        };

        let expr = &expressions[constraint["e"].as_u64().unwrap_or(0) as usize];

        if let Some(symbols) = expr["symbols"].as_array() {
            for symbol_used in symbols {
                if !ctx.symbols_used.iter().any(|s| {
                    s["op"] == symbol_used["op"] && s["stage"] == symbol_used["stage"] && s["id"] == symbol_used["id"]
                }) {
                    ctx.symbols_used.push(symbol_used.clone());
                }
            }
        }

        pil_code_gen(&mut ctx, symbols, expressions, constraint["e"].as_u64().unwrap_or(0) as usize, 0);
        let mut constraint_code = build_code(&mut ctx);

        constraint_code["boundary"] = constraint["boundary"].clone();
        constraint_code["line"] = constraint["line"].clone();
        constraint_code["imPol"] = json!(constraint["imPol"].as_bool().unwrap_or(false) as u8);
        constraint_code["stage"] = json!(if constraint["stage"].as_u64().unwrap_or(0) == 0 {
            1
        } else {
            constraint["stage"].as_u64().unwrap_or(0)
        });

        if constraint["boundary"] == json!("everyFrame") {
            constraint_code["offsetMin"] = constraint["offsetMin"].clone();
            constraint_code["offsetMax"] = constraint["offsetMax"].clone();
        }

        constraints_code.push(constraint_code);
    }

    constraints_code
}

/// Generates constraint polynomial verifier code
pub fn generate_constraint_polynomial_verifier_code(
    res: &mut Value,
    verifier_info: &mut Value,
    symbols: &[Value],
    expressions: &[Value],
) {
    let mut ctx = CodeGenContext {
        stage: res["nStages"].as_u64().unwrap_or(0) as usize + 1,
        calculated: HashMap::new(),
        tmp_used: 0,
        code: Vec::new(),
        ev_map: Vec::new(),
        dom: "n".to_string(),
        air_id: res["airId"].clone(),
        airgroup_id: res["airgroupId"].clone(),
        opening_points: res["openingPoints"].as_array().unwrap_or(&vec![]).iter().filter_map(|v| v.as_i64()).collect(),
        symbols_used: Vec::new(),
        verifier_evaluations: true,
        ..Default::default()
    };

    for symbol in symbols {
        if symbol["imPol"].as_bool().unwrap_or(false) {
            let exp_id = symbol["expId"].as_u64().unwrap_or(0) as usize;
            ctx.calculated.entry(exp_id).or_default();
            for &opening_point in &ctx.opening_points {
                ctx.calculated.get_mut(&exp_id).unwrap().insert(opening_point, json!({ "cm": true }));
            }
        }
    }

    if let Some(exp_symbols) = expressions[res["cExpId"].as_u64().unwrap_or(0) as usize]["symbols"].as_array() {
        for symbol_used in exp_symbols {
            if !ctx.symbols_used.iter().any(|s| {
                s["op"] == symbol_used["op"] && s["stage"] == symbol_used["stage"] && s["id"] == symbol_used["id"]
            }) {
                ctx.symbols_used.push(symbol_used.clone());
            }

            if ["cm", "const", "custom"].contains(&symbol_used["op"].as_str().unwrap_or("")) {
                if let Some(rows_offsets) = symbol_used["rowsOffsets"].as_array() {
                    for row_offset in rows_offsets {
                        let prime = row_offset.as_i64().unwrap_or(0);
                        let opening_pos = res["openingPoints"]
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .position(|p| p == row_offset)
                            .unwrap_or(0);

                        let mut rf = json!({
                            "type": symbol_used["op"],
                            "id": symbol_used["id"],
                            "prime": prime,
                            "openingPos": opening_pos
                        });

                        if symbol_used["op"] == json!("custom") {
                            rf["commitId"] = symbol_used["commitId"].clone();
                        }

                        ctx.ev_map.push(rf);
                    }
                }
            }
        }
    }

    let q_index = res["cmPolsMap"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .position(|p| p["stage"] == json!(res["nStages"].as_u64().unwrap_or(0) + 1) && p["stageId"] == json!(0))
        .unwrap_or(0);

    let opening_pos =
        res["openingPoints"].as_array().unwrap_or(&vec![]).iter().position(|p| p == &json!(0)).unwrap_or(0);

    for i in 0..res["qDeg"].as_u64().unwrap_or(0) {
        ctx.ev_map.push(json!({
            "type": "cm",
            "id": q_index + i as usize,
            "prime": 0,
            "openingPos": opening_pos
        }));
    }

    let mut type_order = HashMap::from([("cm".to_string(), 0), ("const".to_string(), 1)]);

    for (i, _) in res["customCommits"].as_array().unwrap_or(&vec![]).iter().enumerate() {
        type_order.insert(format!("custom{}", i), i + 2);
    }

    ctx.ev_map.sort_by(|a, b| {
        let a_type = a["type"].as_str().unwrap_or("");
        let b_type = b["type"].as_str().unwrap_or("");

        let a_order = type_order.get(a_type).copied().unwrap_or(usize::MAX);
        let b_order = type_order.get(b_type).copied().unwrap_or(usize::MAX);

        if a_order != b_order {
            b_order.cmp(&a_order)
        } else if a["id"].as_u64().unwrap_or(0) != b["id"].as_u64().unwrap_or(0) {
            a["id"].as_u64().unwrap_or(0).cmp(&b["id"].as_u64().unwrap_or(0))
        } else {
            a["prime"].as_i64().unwrap_or(0).cmp(&b["prime"].as_i64().unwrap_or(0))
        }
    });

    pil_code_gen(&mut ctx, symbols, expressions, res["cExpId"].as_u64().unwrap_or(0) as usize, 0);
    let mut verifier_code = build_code(&mut ctx);

    verifier_code["line"] = json!("");
    verifier_info["qVerifier"] = verifier_code;
    res["evMap"] = json!(ctx.ev_map);
}
