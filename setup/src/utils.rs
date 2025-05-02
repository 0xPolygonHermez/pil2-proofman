use pilout::pilout::{AirGroupValue, GlobalConstraint, PilOut, SymbolType};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value, Map};
use std::collections::{HashMap, HashSet};
use crate::airout::AirOut;
use crate::{
    gen_code::{build_code, pil_code_gen, CodeGenContext},
    gen_pil_code::add_hints_info,
    helpers::add_info_expressions,
    setup::StarkStruct,
};
use itertools::Itertools; // Needed for sorting

/// Enum representing Pilout types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiloutType {
    FixedCol = 1,
    WitnessCol = 3,
    ProofValue = 4,
    AirgroupValue = 5,
    PublicValue = 6,
    Challenge = 8,
    AirValue = 9,
    CustomCol = 10,
}

impl PiloutType {
    /// Converts an integer to a `PiloutType`
    pub fn from_u64(value: u64) -> Option<Self> {
        match value {
            1 => Some(Self::FixedCol),
            3 => Some(Self::WitnessCol),
            4 => Some(Self::ProofValue),
            5 => Some(Self::AirgroupValue),
            6 => Some(Self::PublicValue),
            8 => Some(Self::Challenge),
            9 => Some(Self::AirValue),
            10 => Some(Self::CustomCol),
            _ => None,
        }
    }
}

/// Exact analogue of JS `formatExpressions`.
///
/// * `pilout`       – full pil-out JSON (requires `.expressions`).  
/// * `save_symbols` – if `true`, we collect symbols by passing the same
///                    mutable vector into every `format_expression` call.  
/// * `global`       – same flag as in the JS code.
///
/// Returns a JSON object:  
/// `{ "expressions": [...] }`  **or**  `{ "expressions": [...], "symbols": [...] }`.
pub fn format_expressions(pilout: &Value, save_symbols: bool, global: bool) -> Value {
    // Symbols collected during formatting
    let mut symbols: Vec<HashMap<String, Value>> = Vec::new();

    // Map each raw expression through `format_expression`
    let expressions: Vec<Value> = pilout["expressions"]
        .as_array()
        .expect("pilout.expressions must be an array")
        .iter()
        .map(|raw| format_expression(pilout, &mut symbols, raw, save_symbols, global))
        .collect();

    // --- build the return object -------------------------------------------
    if !save_symbols {
        json!({ "expressions": expressions })
    } else {
        // convert each HashMap → serde_json::Map before Value::Object
        let symbols_val: Vec<Value> = symbols
            .into_iter()
            .map(|hm| {
                // hm is HashMap<String,Value>; turn it into Map<String,Value>
                let map: serde_json::Map<String, Value> = hm.into_iter().collect();
                Value::Object(map)
            })
            .collect();

        json!({
            "expressions": expressions,
            "symbols": symbols_val
        })
    }
}

/// Convert raw `pilout.symbols` into canonical symbol objects, 1-for-1 with the
/// JS `formatSymbols`.
///
/// * `pilout` – full pil-out JSON (must contain `.symbols`, `.airGroupValues`, `.airValues`, …)  
/// * `global` – same flag as JS (`true` when formatting the global symbol table)
///
/// Returns `Vec<Value>` (each element is a symbol object).
/// Canonicalise `pilout.symbols` – faithful port of JS `formatSymbols`.
pub fn format_symbols(pilout: &Value, global: bool) -> Vec<Value> {
    const FIXED_COL: &str = "fixedCol";
    const WITNESS_COL: &str = "witnessCol";
    const CUSTOM_COL: &str = "customCol";
    const PROOF_VALUE: &str = "proofValue";
    const CHALLENGE: &str = "challenge";
    const PUBLIC_VALUE: &str = "publicValue";
    const AIRGROUP_VALUE: &str = "airGroupValue";
    const AIR_VALUE: &str = "airValue";

    // ――― 1. optional filter when `global == true` ―――
    let filtered: Vec<&Value> = pilout["symbols"]
        .as_array()
        .expect("pilout.symbols must be array")
        .iter()
        .filter(|s| {
            if !global {
                return true;
            }
            let t = s["type"].as_str().unwrap();
            !(t == AIR_VALUE || t == CUSTOM_COL || t == FIXED_COL || t == WITNESS_COL)
        })
        .collect();

    // ――― 2. flat-map into canonical symbols ―――
    let mut out: Vec<Value> = Vec::new();

    for s in filtered {
        let t = s["type"].as_str().unwrap();

        match t {
            // ════════════════════════════════════════════════════════════════
            // fixed / witness / custom columns
            // ════════════════════════════════════════════════════════════════
            FIXED_COL | WITNESS_COL | CUSTOM_COL => {
                if t == CUSTOM_COL && s["stage"].as_u64().unwrap() != 0 {
                    panic!("Invalid stage {} for a custom commit", s["stage"]);
                }

                let dim = if [0, 1].contains(&s["stage"].as_u64().unwrap()) { 1 } else { 3 };
                let typ_out = if t == FIXED_COL {
                    "fixed"
                } else if t == CUSTOM_COL {
                    "custom"
                } else {
                    "witness"
                };

                // count previous polynomials of same family
                let previous: Vec<&Value> = pilout["symbols"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|si| {
                        si["type"] == json!(t)
                            && si["airId"] == s["airId"]
                            && si["airGroupId"] == s["airGroupId"]
                            && (si["stage"].as_u64().unwrap() < s["stage"].as_u64().unwrap()
                                || (si["stage"] == s["stage"]
                                    && si["id"].as_u64().unwrap() < s["id"].as_u64().unwrap()))
                            && (t != CUSTOM_COL || si["commitId"] == s["commitId"])
                    })
                    .collect();

                let mut pol_id = 0u64;
                for p in previous {
                    if p.get("dim").is_none() {
                        pol_id += 1;
                    } else {
                        let prod =
                            p["lengths"].as_array().unwrap().iter().map(|l| l.as_u64().unwrap()).product::<u64>();
                        pol_id += prod;
                    }
                }

                if s.get("dim").is_none() {
                    // ― single-slot symbol
                    let mut m = Map::new();
                    m.insert("name".into(), s["name"].clone());
                    m.insert("stage".into(), s["stage"].clone());
                    m.insert("type".into(), json!(typ_out));
                    m.insert("polId".into(), json!(pol_id));
                    m.insert("stageId".into(), s["id"].clone());
                    m.insert("dim".into(), json!(dim));
                    m.insert("airId".into(), s["airId"].clone());
                    m.insert("airgroupId".into(), s["airGroupId"].clone());
                    if t == CUSTOM_COL {
                        m.insert("commitId".into(), s["commitId"].clone());
                    }
                    out.push(Value::Object(m));
                } else {
                    // ― multi-array expansion
                    let mut tmp: Vec<HashMap<String, Value>> = Vec::new();
                    generate_multi_array_symbols(
                        &mut tmp,
                        &[],
                        s,
                        typ_out,
                        s["stage"].as_u64().unwrap(),
                        dim,
                        pol_id,
                        0,
                    );
                    for h in tmp {
                        out.push(Value::Object(h.into_iter().collect()));
                    }
                }
            }

            // ════════════════════════════════════════════════════════════════
            // proofValue
            // ════════════════════════════════════════════════════════════════
            PROOF_VALUE => {
                let dim = if s["stage"].as_u64().unwrap() == 1 { 1 } else { 3 };
                if s.get("dim").is_none() {
                    out.push(json!({
                        "name":  s["name"],
                        "type":  "proofvalue",
                        "stage": s["stage"],
                        "dim":   dim,
                        "id":    s["id"]
                    }));
                } else {
                    let mut tmp = Vec::new();
                    generate_multi_array_symbols(
                        &mut tmp,
                        &[],
                        s,
                        "proofvalue",
                        s["stage"].as_u64().unwrap(),
                        dim,
                        s["id"].as_u64().unwrap(),
                        0,
                    );
                    for h in tmp {
                        out.push(Value::Object(h.into_iter().collect()));
                    }
                }
            }

            // ════════════════════════════════════════════════════════════════
            // challenge
            // ════════════════════════════════════════════════════════════════
            CHALLENGE => {
                let id = pilout["symbols"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|si| {
                        si["type"] == CHALLENGE
                            && (si["stage"].as_u64().unwrap() < s["stage"].as_u64().unwrap()
                                || (si["stage"] == s["stage"]
                                    && si["id"].as_u64().unwrap() < s["id"].as_u64().unwrap()))
                    })
                    .count() as u64;

                out.push(json!({
                    "name":    s["name"],
                    "type":    "challenge",
                    "stageId": s["id"],
                    "id":      id,
                    "stage":   s["stage"],
                    "dim":     3
                }));
            }

            // ════════════════════════════════════════════════════════════════
            // publicValue
            // ════════════════════════════════════════════════════════════════
            PUBLIC_VALUE => {
                if s.get("dim").is_none() {
                    out.push(json!({
                        "name":  s["name"],
                        "stage": 1,
                        "type":  "public",
                        "dim":   1,
                        "id":    s["id"]
                    }));
                } else {
                    let mut tmp = Vec::new();
                    generate_multi_array_symbols(&mut tmp, &[], s, "public", 1, 1, s["id"].as_u64().unwrap(), 0);
                    for h in tmp {
                        out.push(Value::Object(h.into_iter().collect()));
                    }
                }
            }

            // ════════════════════════════════════════════════════════════════
            // airGroupValue
            // ════════════════════════════════════════════════════════════════
            AIRGROUP_VALUE => {
                let stage_val = if !global {
                    pilout["airGroupValues"][s["id"].as_u64().unwrap() as usize]["stage"].clone()
                } else {
                    Value::Null
                };

                if s.get("dim").is_none() {
                    let mut m = json!({
                        "name": s["name"],
                        "type": "airgroupvalue",
                        "id":   s["id"],
                        "airgroupId": s["airGroupId"],
                        "dim":  3
                    });
                    if !stage_val.is_null() {
                        m["stage"] = stage_val;
                    }
                    out.push(m);
                } else {
                    let mut tmp = Vec::new();
                    generate_multi_array_symbols(
                        &mut tmp,
                        &[],
                        s,
                        "airgroupvalue",
                        stage_val.as_u64().unwrap_or(0),
                        3,
                        s["id"].as_u64().unwrap(),
                        0,
                    );
                    for h in tmp {
                        out.push(Value::Object(h.into_iter().collect()));
                    }
                }
            }

            // ════════════════════════════════════════════════════════════════
            // airValue
            // ════════════════════════════════════════════════════════════════
            AIR_VALUE => {
                let stage_val = pilout["airValues"][s["id"].as_u64().unwrap() as usize]["stage"].as_u64().unwrap();
                let dim = if stage_val == 1 { 1 } else { 3 };

                if s.get("dim").is_none() {
                    out.push(json!({
                        "name":  s["name"],
                        "type":  "airvalue",
                        "id":    s["id"],
                        "airgroupId": s["airGroupId"],
                        "stage": stage_val,
                        "dim":   dim
                    }));
                } else {
                    let mut tmp = Vec::new();
                    generate_multi_array_symbols(
                        &mut tmp,
                        &[],
                        s,
                        "airvalue",
                        stage_val,
                        dim,
                        s["id"].as_u64().unwrap(),
                        0,
                    );
                    for h in tmp {
                        out.push(Value::Object(h.into_iter().collect()));
                    }
                }
            }

            // ════════════════════════════════════════════════════════════════
            _ => panic!("Invalid type {}", t),
        }
    }

    out
}

/// Recursively expands a multi-dimensional array symbol, identical to the JS
/// `generateMultiArraySymbols`.
///
/// * `symbols` – mutable vector that receives the generated symbol objects  
/// * `indexes` – current index prefix (depth)  
/// * `sym`     – the template symbol (`sym.name`, `sym.lengths`, `sym.id`, …)  
/// * `typ`     – target `"type"` field ( `"witness"`, `"fixed"` … )  
/// * `stage`   – stage number copied to each symbol  
/// * `dim`     – dimension copied to each symbol  
/// * `pol_id`  – starting polynomial id (JS `polId`)  
/// * `shift`   – running shift counter, **returned** after all children added
///
/// Returns the updated `shift` just like in the JS code.
pub fn generate_multi_array_symbols(
    symbols: &mut Vec<HashMap<String, Value>>,
    indexes: &[usize],
    sym: &Value,
    typ: &str,
    stage: u64,
    dim: u64,
    pol_id: u64,
    mut shift: u64,
) -> u64 {
    let lengths = sym["lengths"].as_array().expect("sym.lengths array required");

    // ----------------------------------------------------------------------
    // Base case: depth == lengths.len()  →  produce one concrete symbol
    // ----------------------------------------------------------------------
    if indexes.len() == lengths.len() {
        let mut s = HashMap::from([
            ("name".into(), sym["name"].clone()),
            ("lengths".into(), json!(indexes)),
            ("idx".into(), json!(shift)),
            ("type".into(), json!(typ)),
            ("polId".into(), json!(pol_id + shift)),
            ("id".into(), json!(pol_id + shift)),
            ("stageId".into(), json!(sym["id"].as_u64().unwrap() + shift)),
            ("stage".into(), json!(stage)),
            ("dim".into(), json!(dim)),
        ]);

        if let Some(v) = sym.get("airId") {
            s.insert("airId".into(), v.clone());
        }
        if let Some(v) = sym.get("airGroupId") {
            s.insert("airgroupId".into(), v.clone());
        }
        if let Some(v) = sym.get("commitId") {
            s.insert("commitId".into(), v.clone());
        }

        symbols.push(s);
        return shift + 1; // JS: return shift+1
    }

    // ----------------------------------------------------------------------
    // Recursive case: iterate 0 .. lengths[current_depth]
    // ----------------------------------------------------------------------
    let depth = indexes.len();
    let cur_len_value = lengths[depth].as_u64().unwrap() as usize;

    for i in 0..cur_len_value {
        // build new prefix indexes + [i]
        let mut new_prefix = indexes.to_vec();
        new_prefix.push(i);

        shift = generate_multi_array_symbols(symbols, &new_prefix, sym, typ, stage, dim, pol_id, shift);
    }

    shift
}

/// Exact analogue of JS `formatHints`.
///
/// * `pilout`     – full pil-out JSON.
/// * `raw_hints`  – slice/array of raw hint objects from pil-out.
/// * `symbols`    – mutable global symbols array (Vec<HashMap>) so `process_hint_field` can add entries.
/// * `expressions`– mutable expressions so `process_hint_field` can mark `.keep = true`.
/// * `save_symbols` / `global` – same flags as in the JS.
///
/// Returns a `Vec<Value>` (JSON array) of formatted hints.
pub fn format_hints(
    pilout: &Value,
    raw_hints: &[Value],
    symbols: &mut Vec<HashMap<String, Value>>,
    expressions: &mut Vec<Value>,
    save_symbols: bool,
    global: bool,
) -> Vec<Value> {
    let mut hints: Vec<Value> = Vec::new();

    for raw_hint in raw_hints {
        let mut hint_obj = json!({ "name": raw_hint["name"] });
        let mut fields_arr: Vec<Value> = Vec::new();

        // JS:  const fields = rawHints[i].hintFields[0].hintFieldArray.hintFields;
        let fields = &raw_hint["hintFields"][0]["hintFieldArray"]["hintFields"];
        for field in fields.as_array().expect("hintFields must be array") {
            let field_name = field["name"].as_str().unwrap();

            // processHintField(...)
            let res = process_hint_field(field, pilout, symbols, expressions, save_symbols, global);

            // JS: if(!lengths) { values: [values], lengths: undefined } else {values,lengths}
            if res.get("lengths").is_none() {
                fields_arr.push(json!({
                    "name": field_name,
                    "values": [ res["values"].clone() ]
                }));
            } else {
                fields_arr.push(json!({
                    "name": field_name,
                    "values": res["values"].clone(),
                    "lengths": res["lengths"].clone()
                }));
            }
        }

        hint_obj["fields"] = Value::Array(fields_arr);
        hints.push(hint_obj);
    }

    hints
}

/// Formats constraints from `pilout`, mimicking the original JavaScript function.
pub fn format_constraints(pilout: &Value) -> Vec<Value> {
    let mut constraints = Vec::new();

    if let Some(pilout_constraints) = pilout["constraints"].as_array() {
        for constraint_obj in pilout_constraints {
            if let Some(constraint_inner) = constraint_obj.get("constraint") {
                let valid_boundaries = ["everyRow", "firstRow", "lastRow", "everyFrame"];

                if let Some(boundary) = valid_boundaries.iter().find_map(|&key| constraint_inner.get(key).map(|_| key))
                {
                    let constraint_data = &constraint_inner[boundary];

                    let mut constraint = json!({
                        "boundary": boundary, // Now correctly sets "everyRow", etc.
                        "e": constraint_data["expressionIdx"]["idx"],
                        "line": constraint_data["debugLine"]
                    });

                    if boundary == "everyFrame" {
                        constraint["offsetMin"] = constraint_data["offsetMin"].clone();
                        constraint["offsetMax"] = constraint_data["offsetMax"].clone();
                    }

                    constraints.push(constraint);
                } else {
                    println!(
                        "⚠️ Warning: Constraint boundary not recognized inside 'constraint': {:?}",
                        constraint_inner
                    );
                }
            } else {
                println!("⚠️ Warning: Constraint object does not contain a 'constraint' key: {:?}", constraint_obj);
            }
        }
    }

    constraints
}

/// Exact analogue of JS `printExpressions`.
pub fn print_expressions(
    res: &mut serde_json::Map<String, Value>,
    exp: &Value,
    expressions: &mut [Value],
    is_constraint: bool,
) -> String {
    // ------------- if exp.op is "exp"  (cached printing) -----------------
    if exp["op"] == "exp" {
        let id = exp["id"].as_u64().unwrap() as usize;

        if expressions[id].get("line").is_none() {
            let line = print_expressions(res, &expressions[id].clone(), expressions, is_constraint);
            expressions[id]["line"] = json!(line);
        }
        return expressions[id]["line"].as_str().unwrap().to_owned();
    }

    // op dispatch
    let op = exp["op"].as_str().unwrap();
    match op {
        // ----------------------- add / sub / mul -------------------------
        "add" | "sub" | "mul" => {
            let lhs = print_expressions(res, &exp["values"][0], expressions, is_constraint);
            let rhs = print_expressions(res, &exp["values"][1], expressions, is_constraint);
            let op_str = match op {
                "add" => " + ",
                "sub" => " - ",
                _ => " * ",
            };
            return format!("({lhs}{op_str}{rhs})");
        }

        // ----------------------- neg -------------------------------------
        "neg" => {
            return print_expressions(res, &exp["values"][0], expressions, is_constraint);
        }

        // ----------------------- number ----------------------------------
        "number" => {
            return exp["value"].as_str().unwrap().to_string();
        }

        // ---------------- const / cm / custom ----------------------------
        "const" | "cm" | "custom" => {
            let id = exp["id"].as_u64().unwrap() as usize;
            let col = match op {
                "const" => &res["constPolsMap"][id],
                "cm" => &res["cmPolsMap"][id],
                "custom" => {
                    let commit_id = exp["commitId"].as_u64().unwrap() as usize;
                    &res["customCommitsMap"][commit_id][id]
                }
                _ => unreachable!(),
            };

            // if it is imPol and we're *not* printing constraint, inline its expression
            if col["imPol"].as_bool().unwrap_or(false) && !is_constraint {
                let exp_id = col["expId"].as_u64().unwrap() as usize;
                return print_expressions(res, &expressions[exp_id].clone(), expressions, false);
            }

            // base name
            let mut name = col["name"].as_str().unwrap().to_string();

            // append `[len]` decorators
            if let Some(lens) = col.get("lengths").and_then(Value::as_array) {
                for l in lens {
                    name.push_str(&format!("[{}]", l));
                }
            }

            // append imPol index suffix
            if col["imPol"].as_bool().unwrap_or(false) {
                let count = res["cmPolsMap"].as_array().unwrap()[..id]
                    .iter()
                    .filter(|v| v["imPol"].as_bool().unwrap_or(false))
                    .count();
                name.push_str(&count.to_string());
            }

            // rowOffset postfix / prefix
            if let Some(row_off) = exp.get("rowOffset").and_then(Value::as_i64) {
                if row_off > 0 {
                    name.push('\'');
                    if row_off > 1 {
                        name.push_str(&row_off.to_string());
                    }
                } else if row_off < 0 {
                    name = format!("'{}{}", (-row_off).to_string(), name);
                }
            }
            return name;
        }

        // --------------------- public / airvalue / … ---------------------
        "public" => {
            return res["publicsMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap().to_owned()
        }
        "airvalue" => {
            return res["airValuesMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap().to_owned()
        }
        "airgroupvalue" => {
            return res["airgroupValuesMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap().to_owned()
        }
        "challenge" => {
            return res["challengesMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap().to_owned()
        }
        "proofvalue" => {
            return res["proofValuesMap"][exp["id"].as_u64().unwrap() as usize]["name"].as_str().unwrap().to_owned()
        }

        // --------------------- primitives x / Zi -------------------------
        "x" => return "x".to_string(),
        "Zi" => return "zh".to_string(),

        // --------------------- default -----------------------------------
        _ => panic!("Unknown op: {}", op),
    }
}

/// Mirrors JS `processHintField`.
///
/// Returns a JSON object:
/// `{ "values": <Value or Vec>, "lengths": <Vec> }`  (lengths omitted if none)
pub fn process_hint_field(
    hint_field: &Value,
    pilout: &Value,
    symbols: &mut Vec<HashMap<String, Value>>,
    expressions: &mut Vec<Value>,
    save_symbols: bool,
    global: bool,
) -> Value {
    // ----------------------------------------------------------------------
    // 1)  Branch: hintFieldArray  (recursive case)
    // ----------------------------------------------------------------------
    if let Some(arr) = hint_field.get("hintFieldArray") {
        let fields = &arr["hintFields"];
        let fields_arr = fields.as_array().expect("hintFields must be array");

        let mut result_fields: Vec<Value> = Vec::new();
        let mut lengths: Vec<usize> = Vec::new();

        for (_, sub_field) in fields_arr.iter().enumerate() {
            let sub_res = process_hint_field(sub_field, pilout, symbols, expressions, save_symbols, global);
            let sub_vals = &sub_res["values"];

            result_fields.push(sub_vals.clone());

            // push the first-level length once
            if lengths.is_empty() {
                lengths.push(fields_arr.len());
            }

            // propagate nested lengths (if any)
            if let Some(sub_lens) = sub_res.get("lengths").and_then(Value::as_array) {
                for (k, slen_val) in sub_lens.iter().enumerate() {
                    let slen = slen_val.as_u64().unwrap() as usize;
                    if lengths.len() <= k + 1 {
                        lengths.resize(k + 2, 0);
                    }
                    if lengths[k + 1] == 0 {
                        lengths[k + 1] = slen;
                    }
                }
            }
        }

        // Build return object
        let mut obj = json!({ "values": result_fields });
        if !lengths.is_empty() {
            obj["lengths"] = json!(lengths);
        }
        return obj;
    }

    // ----------------------------------------------------------------------
    // 2)  Leaf case  (operand or stringValue)
    // ----------------------------------------------------------------------
    let value: Value = if hint_field.get("operand").is_some() {
        let v = format_expression(pilout, symbols, &hint_field["operand"], save_symbols, global);
        // mark expressions[id].keep = true  if v.op=="exp"
        if v["op"] == "exp" {
            let id = v["id"].as_u64().unwrap() as usize;
            if id < expressions.len() {
                expressions[id]["keep"] = json!(true);
            }
        }
        v
    } else if let Some(sv) = hint_field.get("stringValue") {
        json!({ "op": "string", "string": sv })
    } else {
        panic!("Unknown hint field");
    };

    json!({ "values": value })
}

/// Rust port of JS `formatExpression`.
///
/// * `pilout`  – full pil-out JSON.
/// * `symbols` – global symbol vector (HashMap form).
/// * `exp`     – raw expression subtree.
/// * `save_symbols` – if `true`, we call `add_symbol` whenever we create a
///                    column / value node.
/// * `global`        – same flag as the JS code.
///
/// Returns a freshly-allocated canonical expression `Value`.
pub fn format_expression(
    pilout: &Value,
    symbols: &mut Vec<HashMap<String, Value>>,
    exp: &Value,
    save_symbols: bool,
    global: bool,
) -> Value {
    // ──────────────────────────────────────────────────────────────────────
    // fast-path: expression already formatted
    // ──────────────────────────────────────────────────────────────────────
    if exp.get("op").is_some() {
        return exp.clone();
    }

    // grab the single key (same as `Object.keys(exp)[0]` in JS)
    let (raw_op, raw_body) = exp.as_object().and_then(|m| m.iter().next()).expect("expression object is empty");

    let mut store = false; // should we call add_symbol()?
    let mut formatted: Value; // final value to return

    match raw_op.as_str() {
        /* ───────────────────────────────────────── expression reference */
        "expression" => {
            let idx = raw_body["idx"].as_u64().expect("expression.idx") as usize;
            let pil_expr = &pilout["expressions"][idx];
            let exp_op = pil_expr.as_object().unwrap().keys().next().unwrap();

            /* replicate the JS “skip to lhs” micro-optimisation */
            let skip_to_lhs = exp_op != "mul" && exp_op != "neg" && {
                let lhs_key = pil_expr[exp_op]["lhs"].as_object().unwrap().keys().next().unwrap();
                let rhs_key = pil_expr[exp_op]["rhs"].as_object().unwrap().keys().next().unwrap();
                lhs_key != "expression" && rhs_key == "constant" && {
                    // constant RHS must be zero
                    let c_val = &pil_expr[exp_op]["rhs"]["constant"]["value"];
                    let zero = match c_val {
                        Value::String(s) => buf_to_u128(s.as_bytes()),
                        Value::Array(a) => {
                            let bytes: Vec<u8> = a.iter().map(|v| v.as_u64().unwrap() as u8).collect();
                            buf_to_u128(&bytes)
                        }
                        _ => 1, // non-zero fallback ⇒ no skip
                    };
                    zero == 0
                }
            };

            if skip_to_lhs {
                return format_expression(pilout, symbols, &pil_expr[exp_op]["lhs"], save_symbols, global);
            }
            formatted = json!({ "op": "exp", "id": idx });
        }

        /* ───────────────────────────────────────── binary operators */
        "add" | "mul" | "sub" => {
            let lhs = format_expression(pilout, symbols, &raw_body["lhs"], save_symbols, global);
            let rhs = format_expression(pilout, symbols, &raw_body["rhs"], save_symbols, global);
            formatted = json!({ "op": raw_op, "values": [lhs, rhs] });
        }

        /* ───────────────────────────────────────── unary negation */
        "neg" => {
            let val = format_expression(pilout, symbols, &raw_body["value"], save_symbols, global);
            formatted = json!({ "op": "neg", "values": [val] });
        }

        /* ───────────────────────────────────────── numeric constant */
        "constant" => {
            let num = &raw_body["value"];
            let val_u128 = if num.is_string() {
                buf_to_u128(num.as_str().unwrap().as_bytes())
            } else if num.is_array() {
                let bytes: Vec<u8> = num.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
                buf_to_u128(&bytes)
            } else {
                panic!("Unexpected constant.value type: {num:?}");
            };
            formatted = json!({ "op": "number", "value": val_u128.to_string() });
        }

        /* ───────────────────────────────────────── witness / custom columns */
        "witnessCol" | "customCol" => {
            let (typ, commit_id_opt) = if raw_op == "witnessCol" {
                ("cm", None)
            } else {
                ("custom", Some(raw_body["commitId"].as_u64().unwrap()))
            };

            let stage_widths = if raw_op == "witnessCol" {
                pilout["stageWidths"].as_array().unwrap()
            } else {
                let cid = commit_id_opt.unwrap() as usize;
                pilout["customCommits"][cid]["stageWidths"].as_array().unwrap()
            };

            let stage = raw_body["stage"].as_u64().unwrap();
            let stage_idx = raw_body["colIdx"].as_u64().unwrap();
            let row_offset = raw_body["rowOffset"].as_i64().unwrap();
            let prior_sum: u64 = stage_widths
                .iter()
                .take(stage as usize - 1) // safe: saturating_sub implicit from range
                .map(|v| v.as_u64().unwrap())
                .sum();
            let id = prior_sum + stage_idx;
            let dim = if stage == 1 { 1 } else { 3 };

            formatted = json!({
                "op":       typ,
                "id":       id,
                "stageId":  stage_idx,
                "rowOffset":row_offset,
                "stage":    stage,
                "dim":      dim,
                "airGroupId": raw_body.get("airGroupId").cloned().unwrap_or(json!(0)),
                "airId":      raw_body.get("airId").cloned().unwrap_or(json!(0))
            });
            if let Some(cid) = commit_id_opt {
                formatted["commitId"] = json!(cid);
            }
            store = true;
        }

        /* ───────────────────────────────────────── fixed column */
        "fixedCol" => {
            formatted = json!({
                "op":  "const",
                "id":  raw_body["idx"],
                "rowOffset": raw_body["rowOffset"],
                "stage": 0,
                "dim":  1,
                "airGroupId": raw_body.get("airGroupId").cloned().unwrap_or(json!(0)),
                "airId":      raw_body.get("airId").cloned().unwrap_or(json!(0))
            });
            store = true;
        }

        /* ───────────────────────────────────────── public value */
        "publicValue" => {
            formatted = json!({ "op": "public", "id": raw_body["idx"], "stage": 1 });
            store = true;
        }

        /* ───────────────────────────────────────── proof value */
        "proofValue" => {
            let stage = raw_body["stage"].as_u64().unwrap();
            let dim = if stage == 1 { 1 } else { 3 };
            formatted = json!({ "op": "proofvalue", "id": raw_body["idx"], "stage": stage, "dim": dim });
            store = true;
        }

        /* ───────────────────────────────────────── air value */
        "airValue" => {
            let idx = raw_body["idx"].as_u64().unwrap();
            let stage = pilout["airValues"][idx as usize]["stage"].as_u64().unwrap();
            let dim = if stage == 1 { 1 } else { 3 };
            formatted = json!({ "op": "airvalue", "id": idx, "stage": stage, "dim": dim });
            store = true;
        }

        /* ───────────────────────────────────────── air-group value */
        "airGroupValue" => {
            let idx = raw_body["idx"].as_u64().unwrap();
            let airgroup_id = raw_body["airGroupId"].as_u64().unwrap();
            let stage = if !global {
                pilout["airGroupValues"][idx as usize]["stage"].as_u64().unwrap()
            } else {
                let grp = &pilout["airGroups"][airgroup_id as usize];
                grp["airGroupValues"][idx as usize]["stage"].as_u64().unwrap()
            };
            let dim = if stage == 1 { 1 } else { 3 };
            formatted = json!({
                "op": "airgroupvalue",
                "id": idx,
                "airGroupId": airgroup_id,
                "dim": dim,
                "stage": stage
            });
            store = true;
        }

        /* ───────────────────────────────────────── challenge (value) */
        "challenge" => {
            let idx = raw_body["idx"].as_u64().unwrap();
            let stage = raw_body["stage"].as_u64().unwrap();
            let prior: u64 = pilout["numChallenges"]
                .as_array()
                .unwrap()
                .iter()
                .take(stage as usize - 1)
                .map(|v| v.as_u64().unwrap())
                .sum();
            let id = prior + idx;
            formatted = json!({ "op": "challenge", "stage": stage, "stageId": idx, "id": id });
            store = true;
        }

        /* ───────────────────────────────────────── hint AST passthrough */
        "operation" => {
            formatted = format_expression(pilout, symbols, &exp["operation"], save_symbols, global);
        }
        "operand" => {
            formatted = format_expression(pilout, symbols, &exp["operand"], save_symbols, global);
        }

        /* ───────────────────────────────────────── unknown */
        _ => panic!("Unknown op: {raw_op} in {exp:?}"),
    }

    /* add symbol if required */
    if save_symbols && store {
        add_symbol(pilout, symbols, &formatted, global);
    }

    formatted
}

/// Convert a big-endian byte slice (≤ 16 bytes) to a `u128`.
///
/// Panics if the slice is longer than 16 bytes — the JS version could grow
/// arbitrarily large (`BigInt`); using `u128` limits us to 128 bits.
pub fn buf_to_u128(buf: &[u8]) -> u128 {
    assert!(buf.len() <= 16, "buffer too long ({} bytes) for u128 representation", buf.len());

    let mut value: u128 = 0;
    for &byte in buf {
        value = (value << 8) | byte as u128;
    }
    value
}

/// Push a missing entry into `symbols`, mirroring the JS `addSymbol`.
///
/// * `pilout`   – parsed PIL-out JSON (needs `.name`, `.airId`, `.airgroupId`, `.airGroupValues`).
/// * `symbols`  – mutable list of symbol objects (`Vec<HashMap<String,Value>>`).
/// * `exp`      – the expression to analyse (must contain at least `"op"` plus id / stage / stageId).
/// * `global`   – JS flag deciding whether air-group symbols get `airId` / `airgroupId` fields.
///
/// The function mutates `symbols` in-place; it returns `()`.
pub fn add_symbol(pilout: &Value, symbols: &mut Vec<HashMap<String, Value>>, exp: &Value, global: bool) {
    let op = exp["op"].as_str().expect("exp.op missing");
    let air_id = exp.get("airId").and_then(Value::as_u64).unwrap_or(0);
    let airgroup_id = exp.get("airGroupId").and_then(Value::as_u64).unwrap_or(0);
    let pil_name = pilout["name"].as_str().unwrap_or("");
    let stage = exp.get("stage").and_then(Value::as_u64).unwrap_or(0);
    let stage_id = exp.get("stageId").and_then(Value::as_u64).unwrap_or(0);
    let id = exp.get("id").and_then(Value::as_u64).unwrap_or(0);

    match op {
        // ------------------------------------------------------------------ public
        "public" => {
            if symbols.iter().all(|s| !(s["type"] == "public" && s["id"] == json!(id))) {
                symbols.push(HashMap::from([
                    ("type".into(), json!("public")),
                    ("dim".into(), json!(1)),
                    ("id".into(), json!(id)),
                    ("name".into(), json!(format!("{}.public_{}", pil_name, id))),
                ]));
            }
        }

        // ------------------------------------------------------------------ challenge
        "challenge" => {
            if symbols
                .iter()
                .all(|s| !(s["type"] == "challenge" && s["stage"] == json!(stage) && s["stageId"] == json!(stage_id)))
            {
                let prior = symbols
                    .iter()
                    .filter(|s| {
                        s["type"] == "challenge"
                            && (s["stage"].as_u64().unwrap() < stage
                                || (s["stage"] == json!(stage) && s["stageId"].as_u64().unwrap() < stage_id))
                    })
                    .count();
                symbols.push(HashMap::from([
                    ("type".into(), json!("challenge")),
                    ("stageId".into(), json!(stage_id)),
                    ("stage".into(), json!(stage)),
                    ("id".into(), json!(prior)), // JS “id = …length”
                    ("dim".into(), json!(3)),
                    ("name".into(), json!(format!("{}.challenge_{}_{}", pil_name, stage, stage_id))),
                ]));
            }
        }

        // ------------------------------------------------------------------ const  -> fixed
        "const" => {
            if symbols.iter().all(|s| {
                !(s["type"] == "fixed"
                    && s["airId"] == json!(air_id)
                    && s["airgroupId"] == json!(airgroup_id)
                    && s["stage"] == json!(stage)
                    && s["stageId"] == json!(id))
            }) {
                symbols.push(HashMap::from([
                    ("type".into(), json!("fixed")),
                    ("polId".into(), json!(id)),
                    ("stageId".into(), json!(id)),
                    ("stage".into(), json!(stage)),
                    ("dim".into(), json!(1)),
                    ("name".into(), json!(format!("{}.fixed_{}", pil_name, id))),
                    ("airId".into(), json!(air_id)),
                    ("airgroupId".into(), json!(airgroup_id)),
                ]));
            }
        }

        // ------------------------------------------------------------------ cm     -> witness
        "cm" => {
            if symbols.iter().all(|s| {
                !(s["type"] == "witness"
                    && s["airId"] == json!(air_id)
                    && s["airgroupId"] == json!(airgroup_id)
                    && s["stage"] == json!(stage)
                    && s["stageId"] == json!(stage_id))
            }) {
                let dim = if stage == 1 { 1 } else { 3 };
                symbols.push(HashMap::from([
                    ("type".into(), json!("witness")),
                    ("polId".into(), json!(id)),
                    ("stageId".into(), json!(stage_id)),
                    ("stage".into(), json!(stage)),
                    ("dim".into(), json!(dim)),
                    ("name".into(), json!(format!("{}.witness_{}_{}", pil_name, stage, stage_id))),
                    ("airId".into(), json!(air_id)),
                    ("airgroupId".into(), json!(airgroup_id)),
                ]));
            }
        }

        // ------------------------------------------------------------------ airgroupvalue
        "airgroupvalue" => {
            if symbols.iter().all(|s| {
                !(s["type"] == "airgroupvalue"
                    && s["id"] == json!(id)
                    && s["airId"] == json!(air_id)
                    && s["airgroupId"] == json!(airgroup_id))
            }) {
                let mut sym = HashMap::from([
                    ("type".into(), json!("airgroupvalue")),
                    ("id".into(), json!(id)),
                    ("name".into(), json!(format!("{}.airgroupvalue_{}", pil_name, id))),
                    ("stage".into(), pilout["airGroupValues"][id as usize]["stage"].clone()),
                    ("dim".into(), json!(3)),
                ]);
                if !global {
                    sym.insert("airId".into(), json!(air_id));
                    sym.insert("airgroupId".into(), json!(airgroup_id));
                }
                symbols.push(sym);
            }
        }

        // ------------------------------------------------------------------ airvalue
        "airvalue" => {
            if symbols.iter().all(|s| {
                !(s["type"] == "airvalue"
                    && s["id"] == json!(id)
                    && s["airId"] == json!(air_id)
                    && s["airgroupId"] == json!(airgroup_id))
            }) {
                let dim = if stage == 1 { 1 } else { 3 };
                symbols.push(HashMap::from([
                    ("type".into(), json!("airvalue")),
                    ("dim".into(), json!(dim)),
                    ("id".into(), json!(id)),
                    ("stage".into(), json!(stage)),
                    ("name".into(), json!(format!("{}.airvalue_{}", pil_name, id))),
                    ("airId".into(), json!(air_id)),
                    ("airgroupId".into(), json!(airgroup_id)),
                ]));
            }
        }

        // ------------------------------------------------------------------ proofvalue
        "proofvalue" => {
            if symbols.iter().all(|s| !(s["type"] == "proofvalue" && s["id"] == json!(id))) {
                let dim = if stage == 1 { 1 } else { 3 };
                symbols.push(HashMap::from([
                    ("type".into(), json!("proofvalue")),
                    ("name".into(), json!(format!("{}.proofvalue_{}", pil_name, id))),
                    ("dim".into(), json!(dim)),
                    ("id".into(), json!(id)),
                ]));
            }
        }

        // ------------------------------------------------------------------ unknown op
        _ => panic!("Unknown operation {}", op),
    }
}

/// Computes log2 of a given value using bitwise operations, similar to the JS implementation.
pub fn log2(mut v: u32) -> u32 {
    let mut r = 0;
    if (v & 0xFFFF0000) != 0 {
        v &= 0xFFFF0000;
        r |= 16;
    }
    if (v & 0xFF00FF00) != 0 {
        v &= 0xFF00FF00;
        r |= 8;
    }
    if (v & 0xF0F0F0F0) != 0 {
        v &= 0xF0F0F0F0;
        r |= 4;
    }
    if (v & 0xCCCCCCCC) != 0 {
        v &= 0xCCCCCCCC;
        r |= 2;
    }
    if (v & 0xAAAAAAAA) != 0 {
        r |= 1;
    }
    r
}

/// Computes a sequence of `ks` values based on field multiplication.
/// `Fr` is a struct that implements field arithmetic with a multiplication method `mul()`.
pub fn get_ks<F: Fn(f64, f64) -> f64>(fr_mul: F, n: usize, k: f64) -> Vec<f64> {
    let mut ks = vec![k];
    for i in 1..n {
        ks.push(fr_mul(ks[i - 1], ks[0]));
    }
    ks
}

/// Metadata structure for an AIR system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRMetadata {
    pub name: String,
    pub num_rows: usize,
}

/// FRI step information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FRIStep {
    pub n_bits: usize,
}

/// Proof value mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofValueMetadata {
    pub name: String,
    pub id: usize,
}

/// VADCOP information structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadcopInfo {
    pub name: String,
    pub airs: Vec<Vec<AIRMetadata>>,
    pub air_groups: Vec<String>,
    pub agg_types: Vec<Vec<Value>>, // Using `Value` to match JSON expectations
    pub steps_fri: Vec<FRIStep>,
    pub n_publics: usize,
    pub num_challenges: Vec<usize>,
    pub num_proof_values: usize,
    pub proof_values_map: Vec<ProofValueMetadata>,
}

/// Convert `AirGroupValue` to JSON manually
fn air_group_value_to_json(value: &AirGroupValue) -> Value {
    json!({
        "agg_type": value.agg_type,
        "stage": value.stage
    })
}

/// Convert `GlobalConstraint` to JSON manually
fn global_constraint_to_json(constraint: &GlobalConstraint) -> Value {
    json!({
        "expression_idx": constraint.expression_idx.as_ref().map(|e| json!({ "idx": e.idx })),
        "debug_line": constraint.debug_line.clone().unwrap_or_default()
    })
}

/// Convert `PilOut` to a `HashMap<String, Value>` for JSON serialization.
fn pilout_to_json(pilout: &PilOut) -> HashMap<String, Value> {
    let mut json_map = HashMap::new();

    json_map.insert("name".to_string(), json!(pilout.name.clone().unwrap_or_default()));
    json_map.insert("base_field".to_string(), json!(pilout.base_field));

    // Convert `AirGroup`
    let air_groups_json: Vec<Value> = pilout
        .air_groups
        .iter()
        .map(|group| {
            json!({
                "name": group.name.clone().unwrap_or_default(),
                "air_group_values": group.air_group_values.iter().map(air_group_value_to_json).collect::<Vec<_>>(),
                "airs": group.airs.len() // Storing just the count to reduce size
            })
        })
        .collect();
    json_map.insert("air_groups".to_string(), json!(air_groups_json));

    json_map.insert("num_challenges".to_string(), json!(pilout.num_challenges));
    json_map.insert("num_proof_values".to_string(), json!(pilout.num_proof_values));
    json_map.insert("num_public_values".to_string(), json!(pilout.num_public_values));

    // Convert `GlobalConstraint`
    let constraints_json: Vec<Value> = pilout.constraints.iter().map(global_constraint_to_json).collect();
    json_map.insert("constraints".to_string(), json!(constraints_json));

    json_map
}

/// Extracts metadata from `AirOut` and STARK structures.
/// Returns `(VadcopInfo, HashMap<String, Value>)` exactly like the JS.
pub fn set_airout_info(airout: &AirOut, stark_structs: &[StarkStruct]) -> (VadcopInfo, HashMap<String, Value>) {
    /* ─────────────────────────── PilOut → JSON ────────────────────────── */
    let pilout = airout.pilout(); // &PilOut
    let pilout_json_hash = pilout_to_json(pilout); // HashMap<String, Value>
                                                   // Turn it into a `Value::Object` so we can treat it like plain JS
    let pilout_json_val = Value::Object(pilout_json_hash.clone().into_iter().collect::<serde_json::Map<_, _>>());

    /* ─────────────────────────── VadcopInfo ───────────────────────────── */
    let mut vadcop_info = VadcopInfo {
        name: pilout.name.clone().unwrap_or_else(|| "default".into()),
        airs: vec![vec![]; pilout.air_groups.len()],
        air_groups: vec![],
        agg_types: vec![vec![]; pilout.air_groups.len()],
        steps_fri: vec![],
        n_publics: pilout.num_public_values as usize,
        num_challenges: pilout.num_challenges.iter().map(|&x| x as usize).collect(),
        num_proof_values: pilout.num_proof_values.len(),
        proof_values_map: vec![],
    };

    for (ag_id, ag) in pilout.air_groups.iter().enumerate() {
        vadcop_info.air_groups.push(ag.name.clone().unwrap_or_else(|| format!("AirGroup {ag_id}")));
        vadcop_info.agg_types[ag_id] = ag.air_group_values.iter().map(air_group_value_to_json).collect();

        vadcop_info.airs[ag_id] = ag
            .airs
            .iter()
            .map(|air| AIRMetadata {
                name: air.name.clone().unwrap_or_else(|| "Unnamed Air".into()),
                num_rows: air.num_rows.unwrap_or(0) as usize,
            })
            .collect();
    }

    /* -------------- gather & sort all distinct FRI steps --------------- */
    let final_step = stark_structs
        .first()
        .and_then(|s| s.steps.last())
        .map(|s| s.n_bits)
        .expect("StarkStruct must contain at least one step");

    let mut steps_fri: HashSet<usize> = HashSet::new();
    for s in stark_structs {
        for st in &s.steps {
            steps_fri.insert(st.n_bits);
        }
        if s.steps.last().map(|st| st.n_bits) != Some(final_step) {
            panic!("All FRI step chains must end at the same nBits");
        }
    }
    vadcop_info.steps_fri = steps_fri.into_iter().sorted_by(|a, b| b.cmp(a)).map(|n_bits| FRIStep { n_bits }).collect();

    /* ---------------------- proof-values map --------------------------- */
    vadcop_info.proof_values_map = pilout
        .symbols
        .iter()
        .filter(|s| s.r#type == SymbolType::ProofValue as i32)
        .map(|p| ProofValueMetadata { name: p.name.clone(), id: p.id as usize })
        .collect();

    /* ──────── global constraints + hints (same as JS helper) ─────────── */
    let mut res_obj: HashMap<String, Value> = HashMap::new();
    let global_constraints_val = get_global_constraints_info(&mut res_obj, &pilout_json_val);

    // we promised a `HashMap<String, Value>` here
    let global_constraints_map: HashMap<String, Value> =
        global_constraints_val.as_object().unwrap().clone().into_iter().collect();

    (vadcop_info, global_constraints_map)
}

/// Rust counterpart of JS `getGlobalConstraintsInfo`.
///
/// Returns a JSON object `{ "constraints": [...], "hints": [...] }`.
/// Rust counterpart of JS `getGlobalConstraintsInfo`.
/// Returns a JSON object `{ constraints: [...], hints: [...] }`.
pub fn get_global_constraints_info(res: &mut HashMap<String, Value>, pilout: &Value) -> Value {
    /* ───────────────────── 0. will we build symbols? ─────────────────── */
    let save_symbols = pilout.get("symbols").is_none();

    let mut expressions: Vec<Value> = Vec::new();
    let mut symbols_hash: Vec<HashMap<String, Value>> = Vec::new();
    let mut constraints_code: Vec<Value> = Vec::new();
    let mut hints_code: Vec<Value> = Vec::new();

    /* ───────────────────── 1. global CONSTRAINTS ─────────────────────── */
    if let Some(raw_constraints) = pilout.get("constraints").and_then(Value::as_array) {
        /* build lightweight JS-style objects {e,boundary,line} */
        let constraints: Vec<Value> = raw_constraints
            .iter()
            .map(|c| {
                json!({
                    "e":        c["expressionIdx"]["idx"],
                    "boundary": "finalProof",
                    "line":     c["debugLine"]
                })
            })
            .collect();

        /* expressions + symbols (GLOBAL flag == true) */
        if save_symbols {
            let out = format_expressions(pilout, true, true);
            expressions = out["expressions"].as_array().unwrap().clone();
            symbols_hash = out["symbols"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_object().unwrap().clone().into_iter().collect())
                .collect();
        } else {
            let out = format_expressions(pilout, false, true);
            expressions = out["expressions"].as_array().unwrap().clone();
            symbols_hash = format_symbols(pilout, true)
                .into_iter()
                .map(|v| v.as_object().unwrap().clone().into_iter().collect())
                .collect();
        }
        let symbols_val: Vec<Value> =
            symbols_hash.iter().map(|h| Value::Object(h.clone().into_iter().collect())).collect();

        /* ensure sub-expressions have `.keep` / symbol info */
        for c in &constraints {
            let e_idx = c["e"].as_u64().unwrap() as usize;
            add_info_expressions(&mut expressions, e_idx);
        }

        /* code-gen context – faithful to JS */
        let mut ctx = CodeGenContext {
            calculated: HashMap::new(),
            tmp_used: 0,
            code: Vec::new(),
            dom: "n".into(),
            ..Default::default()
        };

        for c in &constraints {
            let e_idx = c["e"].as_u64().unwrap() as usize;
            pil_code_gen(&mut ctx, &symbols_val, &expressions, e_idx, 0);

            let mut code = build_code(&mut ctx);
            ctx.tmp_used = code["tmpUsed"].as_u64().unwrap() as usize;

            code["boundary"] = c["boundary"].clone();
            code["line"] = c["line"].clone();
            constraints_code.push(code);
        }
    }

    /* ────────────────────── 2. global HINTS ──────────────────────────── */
    if let Some(all_hints) = pilout.get("hints").and_then(Value::as_array) {
        let global_hints: Vec<Value> =
            all_hints.iter().filter(|h| h.get("airId").is_none() && h.get("airGroupId").is_none()).cloned().collect();

        if !global_hints.is_empty() {
            let hints_formatted =
                format_hints(pilout, &global_hints, &mut symbols_hash, &mut expressions, save_symbols, true);
            hints_code = add_hints_info(res, &mut expressions, &hints_formatted);
        }
    }

    /* ────────────────────── 3. final object ──────────────────────────── */
    json!({
        "constraints": constraints_code,
        "hints":       hints_code
    })
}

use num_bigint::BigUint;
use num_traits::Zero;

/// Convert a JSON array of bytes (big-endian) into a big integer,
/// returning the result as a JSON string.
///
/// Example input:
/// ```json
/// [ 1, 2, 3, 4, 5 ]
/// ```
/// Output: `"43287193605"`  (decimal BigInt as string)
pub fn buf2bint(buf_json: &Value) -> Value {
    // 1. Validate & copy into Vec<u8>
    let bytes: Vec<u8> = buf_json
        .as_array()
        .expect("buffer must be a JSON array")
        .iter()
        .map(|v| v.as_u64().expect("buffer entries must be 0-255") as u8)
        .collect();

    // 2. Re-implement the JS algorithm
    let mut value = BigUint::zero();
    let mut offset = 0usize;

    // read 8-byte chunks
    while bytes.len() - offset >= 8 {
        let chunk = u64::from_be_bytes(bytes[offset..offset + 8].try_into().unwrap());
        value = (value << 64u8) + BigUint::from(chunk);
        offset += 8;
    }
    // read 4-byte chunks
    while bytes.len() - offset >= 4 {
        let chunk = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
        value = (value << 32u8) + BigUint::from(chunk);
        offset += 4;
    }
    // read 2-byte chunks
    while bytes.len() - offset >= 2 {
        let chunk = u16::from_be_bytes(bytes[offset..offset + 2].try_into().unwrap());
        value = (value << 16u8) + BigUint::from(chunk);
        offset += 2;
    }
    // read remaining single bytes
    while bytes.len() - offset >= 1 {
        let chunk = bytes[offset];
        value = (value << 8u8) + BigUint::from(chunk);
        offset += 1;
    }

    // 3. Return as JSON value (string form)
    json!(value.to_str_radix(10))
}
