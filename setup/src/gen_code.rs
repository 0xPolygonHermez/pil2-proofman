use serde_json::{json, Value};
use std::collections::HashMap;

/// Context struct for handling expression transformation
#[derive(Debug, Clone)]
struct CodeGenContext {
    stage: usize,
    calculated: HashMap<usize, HashMap<i64, Value>>,
    symbols_used: Vec<Value>,
    tmp_used: usize,
    code: Vec<Value>,
    dom: String,
    air_id: Value,
    airgroup_id: Value,
    opening_points: Vec<i64>,
    verifier_evaluations: bool,
    ev_map: Vec<Value>,
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

        let mut expr_dest = None;
        if exp["imPol"].as_bool().unwrap_or(false) {
            if let Some(symbol_dest) = symbols.iter().find(|s| s["expId"] == json!(j)) {
                expr_dest = Some(json!({
                    "op": "cm",
                    "stage": symbol_dest["stage"],
                    "stageId": symbol_dest["stageId"],
                    "id": symbol_dest["polId"]
                }));
            }
        }

        if let Some(exp_symbols) = exp["symbols"].as_array() {
            for symbol in exp_symbols {
                if !ctx.symbols_used.iter().any(|s| s == symbol) {
                    ctx.symbols_used.push(symbol.clone());
                }
            }
        }

        pil_code_gen(&mut ctx, symbols, expressions, j, 0);
        let mut exp_info = build_code(&ctx);

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
        exp_info["dest"] = expr_dest.unwrap_or(json!(null));
        exp_info["line"] = exp["line"].clone();

        expressions_code.push(exp_info);
    }

    expressions_code
}

/// Generates constraint polynomial verifier code.
pub fn generate_constraint_polynomial_verifier_code(
    res: &mut Value, // Make `res` mutable
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
    };

    for symbol in symbols {
        if symbol["imPol"].as_bool().unwrap_or(false) {
            let exp_id = symbol["expId"].as_u64().unwrap_or(0) as usize;
            ctx.calculated.insert(exp_id, HashMap::new());
            for opening_point in &ctx.opening_points {
                ctx.calculated.get_mut(&exp_id).unwrap().insert(*opening_point, json!({ "cm": true }));
            }
        }
    }

    pil_code_gen(&mut ctx, symbols, expressions, res["cExpId"].as_u64().unwrap_or(0) as usize, 0);
    verifier_info["qVerifier"] = build_code(&ctx);
    verifier_info["qVerifier"]["line"] = json!("");

    res["evMap"] = json!(ctx.ev_map);
}

/// Placeholder for pilCodeGen function.
fn pil_code_gen(_ctx: &mut CodeGenContext, _symbols: &[Value], _expressions: &[Value], _exp_id: usize, _depth: usize) {
    // Implementation needed
}

/// Placeholder for buildCode function.
fn build_code(_ctx: &CodeGenContext) -> Value {
    json!({ "code": [] }) // Stub implementation
}
