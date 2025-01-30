use serde::{Deserialize, Serialize};
use serde_json::{json, Value, Map};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

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

/// Formats expressions from Pilout, returning formatted expressions and optionally symbols.
pub fn format_expressions(pilout: &HashMap<String, Value>, save_symbols: bool, global: bool) -> HashMap<String, Value> {
    let mut symbols = Vec::new();
    let expressions: Vec<Value> = pilout["expressions"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|e| format_expression(e, pilout, &mut symbols, save_symbols, global))
        .collect();

    let mut result = Map::new();
    result.insert("expressions".to_string(), json!(expressions));
    if save_symbols {
        result.insert("symbols".to_string(), json!(symbols));
    }

    result.into_iter().collect()
}

/// Formats the symbols in `pilout`, filtering and transforming them accordingly.
pub fn format_symbols(pilout: &HashMap<String, Value>, global: bool) -> Vec<Value> {
    let empty_vec = Vec::new();
    let pil_symbols = pilout.get("symbols").and_then(|s| s.as_array()).unwrap_or(&empty_vec);
    let mut symbols = Vec::new();

    for s in pil_symbols {
        let s_type = s["type"].as_u64().expect("Missing type field");

        if s_type == 10 {
            let stage = s["stage"].as_u64().unwrap_or(0);
            if stage != 0 {
                panic!("Invalid stage {} for a custom commit", stage);
            }
        }

        if matches!(s_type, 1 | 3 | 10) {
            let dim = match s["stage"].as_u64().unwrap_or(0) {
                0 | 1 => 1,
                _ => 3,
            };

            let type_str = match s_type {
                1 => "fixed",
                10 => "custom",
                _ => "witness",
            };

            let previous_pols = pil_symbols
                .iter()
                .filter(|si| {
                    si["type"] == s["type"]
                        && si["airId"] == s["airId"]
                        && si["airGroupId"] == s["airGroupId"]
                        && (si["stage"].as_u64().unwrap_or(0) < s["stage"].as_u64().unwrap_or(0)
                            || (si["stage"] == s["stage"]
                                && si["id"].as_u64().unwrap_or(0) < s["id"].as_u64().unwrap_or(0)))
                        && (s_type != 10 || s["commitId"] == si["commitId"])
                })
                .collect::<Vec<_>>();

            let mut pol_id = 0;
            for pol in &previous_pols {
                if pol.get("dim").is_none() {
                    pol_id += 1;
                } else {
                    let lengths = pol["lengths"].as_array().unwrap_or(&empty_vec);
                    pol_id += lengths.iter().map(|l| l.as_u64().unwrap_or(1)).product::<u64>() as usize;
                }
            }

            if s.get("dim").is_none() {
                let stage_id = s["id"].as_u64().unwrap_or(0) as usize;
                let mut symbol = json!({
                    "name": s["name"],
                    "stage": s["stage"],
                    "type": type_str,
                    "polId": pol_id,
                    "stageId": stage_id,
                    "dim": dim,
                    "airId": s["airId"],
                    "airgroupId": s["airGroupId"],
                });

                if s_type == 10 {
                    symbol["commitId"] = s["commitId"].clone();
                }
                symbols.push(symbol);
            } else {
                let mut multi_array_symbols = Vec::new();
                generate_multi_array_symbols(&mut multi_array_symbols, &[], s, type_str, dim, pol_id, 0);
                symbols.extend(multi_array_symbols);
            }
        } else if s_type == 4 {
            symbols.push(json!({
                "name": s["name"],
                "type": "proofValue",
                "id": s["id"]
            }));
        } else if s_type == 8 {
            let id = pil_symbols.iter().filter(|si| si["type"].as_u64() == Some(8)).count();

            symbols.push(json!({
                "name": s["name"],
                "type": "challenge",
                "stageId": s["id"],
                "id": id,
                "stage": s["stage"],
                "dim": 3
            }));
        } else if s_type == 6 {
            if s.get("dim").is_none() {
                symbols.push(json!({
                    "name": s["name"],
                    "stage": 1,
                    "type": "public",
                    "dim": 1,
                    "id": s["id"]
                }));
            } else {
                let mut multi_array_symbols = Vec::new();
                generate_multi_array_symbols(
                    &mut multi_array_symbols,
                    &[],
                    s,
                    "public",
                    1,
                    s["id"].as_u64().unwrap_or(0) as usize,
                    0,
                );
                symbols.extend(multi_array_symbols);
            }
        } else if s_type == 5 {
            let mut airgroup_value = json!({
                "name": s["name"],
                "type": "airgroupvalue",
                "id": s["id"],
                "airgroupId": s["airGroupId"],
                "dim": 3
            });

            if !global {
                let id = s["id"].as_u64().unwrap_or(0) as usize;
                if let Some(airgroup_values) = pilout["airGroupValues"].as_array() {
                    if id < airgroup_values.len() {
                        airgroup_value["stage"] = airgroup_values[id]["stage"].clone();
                    }
                }
            }
            symbols.push(airgroup_value);
        } else if s_type == 9 {
            let mut air_value = json!({
                "name": s["name"],
                "type": "airvalue",
                "id": s["id"],
                "airgroupId": s["airGroupId"]
            });

            if !global {
                let id = s["id"].as_u64().unwrap_or(0) as usize;
                if let Some(air_values) = pilout["airValues"].as_array() {
                    if id < air_values.len() {
                        let stage = air_values[id]["stage"].as_u64().unwrap_or(1);
                        air_value["stage"] = json!(stage);
                        air_value["dim"] = json!(if stage != 1 { 3 } else { 1 });
                    }
                }
            }
            symbols.push(air_value);
        } else {
            panic!("Invalid type {}", s_type);
        }
    }

    symbols
}

/// Generates multi-dimensional array symbols for pilout processing.
fn generate_multi_array_symbols(
    output: &mut Vec<Value>,
    current_indices: &[usize],
    s: &Value,
    type_str: &str,
    dim: usize,
    pol_id: usize,
    depth: usize,
) {
    let v = vec![];
    let lengths = s["lengths"].as_array().unwrap_or(&v);
    if depth == lengths.len() {
        output.push(json!({
            "name": s["name"],
            "stage": s["stage"],
            "type": type_str,
            "polId": pol_id,
            "stageId": s["id"],
            "dim": dim,
            "airId": s["airId"],
            "airgroupId": s["airGroupId"]
        }));
        return;
    }

    for i in 0..lengths[depth].as_u64().unwrap_or(1) {
        let mut new_indices = current_indices.to_vec();
        new_indices.push(i as usize);
        generate_multi_array_symbols(output, &new_indices, s, type_str, dim, pol_id, depth + 1);
    }
}

/// Formats raw hints by processing fields recursively.
pub fn format_hints(
    pilout: &HashMap<String, Value>,
    raw_hints: &[Value],
    symbols: &mut Vec<Value>,
    expressions: &mut Vec<Value>,
    save_symbols: bool,
    _global: bool,
) -> Vec<Value> {
    raw_hints
        .iter()
        .map(|hint| {
            let fields = hint["hintFields"][0]["hintFieldArray"]["hintFields"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|field| {
                    let (values, lengths) = process_hint_field(field, pilout, symbols, expressions, save_symbols);
                    json!({ "name": field["name"], "values": values, "lengths": lengths })
                })
                .collect::<Vec<_>>();

            json!({ "name": hint["name"], "fields": fields })
        })
        .collect()
}

/// Formats constraints from `pilout`, mimicking the original JavaScript function.
pub fn format_constraints(pilout: &Value) -> Vec<Value> {
    let mut constraints = Vec::new();

    if let Some(pilout_constraints) = pilout["constraints"].as_array() {
        for constraint_obj in pilout_constraints {
            if let Some((boundary, constraint_data)) = constraint_obj.as_object().unwrap().iter().next() {
                let mut constraint = json!({
                    "boundary": boundary,
                    "e": constraint_data["expressionIdx"]["idx"],
                    "line": constraint_data["debugLine"]
                });

                if boundary == "everyFrame" {
                    constraint["offsetMin"] = constraint_data["offsetMin"].clone();
                    constraint["offsetMax"] = constraint_data["offsetMax"].clone();
                }

                constraints.push(constraint);
            }
        }
    }

    constraints
}

/// Prints a formatted expression from the given data.
pub fn print_expressions(
    res: &HashMap<String, Value>,
    exp: &Value,
    expressions: &[Value],
    is_constraint: bool,
) -> String {
    match exp["op"].as_str() {
        Some("exp") => {
            if exp.get("line").is_none() {
                let id = exp["id"].as_u64().unwrap() as usize;
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
            let id = exp["id"].as_u64().unwrap() as usize;
            let col = if exp["op"] == "const" {
                &res["constPolsMap"][id]
            } else if exp["op"] == "cm" {
                &res["cmPolsMap"][id]
            } else {
                let commit_id = exp["commitId"].as_u64().unwrap() as usize;
                &res["customCommitsMap"][commit_id][id]
            };

            let mut name = col["name"].as_str().unwrap_or("").to_string();

            if let Some(lengths) = col.get("lengths").and_then(Value::as_array) {
                lengths.iter().for_each(|len| {
                    write!(name, "[{}]", len).unwrap();
                });
            }

            if col["imPol"].as_bool().unwrap_or(false) && !is_constraint {
                let exp_id = col["expId"].as_u64().unwrap() as usize;
                return print_expressions(res, &expressions[exp_id], expressions, false);
            }

            if let Some(row_offset) = exp.get("rowOffset").and_then(|v| v.as_i64()) {
                match row_offset.cmp(&0) {
                    Ordering::Greater => {
                        name.push('\'');
                        if row_offset > 1 {
                            name.push_str(&row_offset.to_string());
                        }
                    }
                    Ordering::Less => {
                        name = format!("'{}{}", row_offset.abs(), name);
                    }
                    Ordering::Equal => {} // Do nothing for zero
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

/// Recursively processes a hint field, extracting values and lengths.
pub fn process_hint_field(
    hint_field: &Value,
    pilout: &HashMap<String, Value>,
    symbols: &mut Vec<Value>,
    expressions: &mut Vec<Value>,
    save_symbols: bool,
) -> (Value, Option<Vec<usize>>) {
    if let Some(hint_field_array) = hint_field.get("hintFieldArray") {
        let fields = hint_field_array["hintFields"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|f| process_hint_field(f, pilout, symbols, expressions, save_symbols))
            .collect::<Vec<_>>();

        let values: Vec<Value> = fields.iter().map(|(v, _)| v.clone()).collect();
        let mut lengths = Vec::new();
        if let Some(first_len) = fields.first().and_then(|(_, l)| l.clone()) {
            lengths.push(first_len.len());
            lengths.extend(first_len);
        }

        (json!(values), Some(lengths))
    } else {
        let value = if let Some(operand) = hint_field.get("operand") {
            let formatted_expr = format_expression(operand, pilout, symbols, save_symbols, false);
            if formatted_expr["op"] == json!("exp") {
                expressions[formatted_expr["id"].as_u64().unwrap_or(0) as usize]["keep"] = json!(true);
            }
            formatted_expr
        } else if let Some(string_value) = hint_field.get("stringValue") {
            json!({ "op": "string", "string": string_value })
        } else {
            panic!("Unknown hint field");
        };

        (value, None)
    }
}

/// Formats an individual expression from Pilout.
pub fn format_expression(
    exp: &Value,
    pilout: &HashMap<String, Value>,
    symbols: &mut Vec<Value>,
    save_symbols: bool,
    _global: bool,
) -> Value {
    if exp.get("op").is_some() {
        return exp.clone();
    }

    let op = exp.as_object().unwrap().keys().next().unwrap().clone();
    let mut store = false;
    let formatted_exp = match op.as_str() {
        "expression" => {
            let id = exp[op]["idx"].as_u64().unwrap_or(0) as usize;
            json!({ "op": "exp", "id": id })
        }
        "add" | "mul" | "sub" => json!({
            "op": op,
            "values": [
                format_expression(&exp[&op]["lhs"], pilout, symbols, save_symbols, false),
                format_expression(&exp[&op]["rhs"], pilout, symbols, save_symbols, false)
            ]
        }),
        "neg" => json!({
            "op": op,
            "values": [format_expression(&exp[&op]["value"], pilout, symbols, save_symbols, false)]
        }),
        "constant" => json!({
            "op": "number",
            "value": buf2bint(&exp[op]["value"]).to_string()
        }),
        "publicValue" => {
            store = true;
            json!({ "op": "public", "id": exp[op]["idx"], "stage": 1 })
        }
        "proofValue" => {
            store = true;
            json!({ "op": "proofvalue", "id": exp[op]["idx"] })
        }
        _ => panic!("Unknown op: {}", op),
    };

    if save_symbols && store {
        add_symbol(pilout, symbols, &formatted_exp);
    }

    formatted_exp
}

/// Converts a buffer to a big integer.
pub fn buf2bint(buf: &Value) -> u128 {
    let empty_vec = Vec::new(); // Store the empty vector to extend its lifetime
    let buf_bytes = buf.as_array().unwrap_or(&empty_vec);

    let mut value = 0u128;
    for byte in buf_bytes {
        value = (value << 8) | byte.as_u64().unwrap_or(0) as u128;
    }

    value
}

/// Adds a symbol to the list of symbols.
pub fn add_symbol(pilout: &HashMap<String, Value>, symbols: &mut Vec<Value>, exp: &Value) {
    let name = format!("{}.{}", pilout["name"].as_str().unwrap_or("unknown"), exp["op"].as_str().unwrap_or("unknown"));
    symbols.push(json!({
        "name": name,
        "type": exp["op"],
        "id": exp["id"]
    }));
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

/// Represents metadata for an AIR system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRMetadata {
    pub name: String,
    pub num_rows: usize,
}

/// Represents an FRI step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FRIStep {
    pub n_bits: usize,
}

/// Represents a proof value mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofValue {
    pub name: String,
    pub id: usize,
}

/// Contains extracted AIR metadata and FRI settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadcopInfo {
    pub name: String,
    pub airs: Vec<Vec<AIRMetadata>>,
    pub air_groups: Vec<String>,
    pub agg_types: Vec<Vec<Value>>, // Holds aggregated values per air group
    pub steps_fri: Vec<FRIStep>,
    pub n_publics: usize,
    pub num_challenges: Vec<usize>,
    pub num_proof_values: usize,
    pub proof_values_map: Vec<ProofValue>,
}

/// Extracts metadata from the AIR system and STARK structures.
/// This function replicates the `setAiroutInfo` logic from JavaScript.
pub fn set_airout_info(airout: &AirOut, stark_structs: &[StarkStruct]) -> (VadcopInfo, HashMap<String, Value>) {
    let mut vadcop_info = VadcopInfo {
        name: "default".to_string(), // Adjusted to a default string (if `AirOut` has no name)
        airs: vec![vec![]; airout.air_groups.len()],
        air_groups: vec![],
        agg_types: vec![vec![]; airout.air_groups.len()],
        steps_fri: vec![],
        n_publics: 0,            // Adjusted since `num_public_values` is missing in `AirOut`
        num_challenges: vec![0], // Adjusted since `num_challenges` is missing
        num_proof_values: 0,     // Adjusted since `num_proof_values` is missing
        proof_values_map: vec![],
    };

    for airgroup in airout.air_groups.iter() {
        let airgroup_id = airgroup.airgroup_id;
        vadcop_info.air_groups.push(format!("AirGroup {}", airgroup_id)); // Placeholder name if missing

        vadcop_info.airs[airgroup_id] =
            airgroup.airs.iter().map(|air| AIRMetadata { name: air.name.clone(), num_rows: air.num_rows }).collect();
    }

    // Extract the final step FRI from the first StarkStruct
    let final_step = stark_structs
        .first()
        .and_then(|s| s.steps.last())
        .map(|s| s.n_bits)
        .unwrap_or_else(|| panic!("StarkStruct must contain at least one step"));

    let mut steps_fri: HashSet<usize> = HashSet::new();

    for stark_struct in stark_structs {
        for step in &stark_struct.steps {
            steps_fri.insert(step.n_bits);
        }
        if stark_struct.steps.last().map(|s| s.n_bits) != Some(final_step) {
            panic!("All FRI steps for different air groups must end at the same nBits");
        }
    }

    vadcop_info.steps_fri = steps_fri
        .into_iter()
        .sorted_by(|a, b| b.cmp(a)) // Sort in descending order
        .map(|n_bits| FRIStep { n_bits })
        .collect();

    // Extract global constraints
    let global_constraints = HashMap::new(); // Placeholder as `AirOut` lacks constraints field

    (vadcop_info, global_constraints)
}

use crate::{
    gen_code::{build_code, pil_code_gen, CodeGenContext},
    gen_pil_code::add_hints_info,
    helpers::add_info_expressions,
    setup::{AirOut, StarkStruct},
};

use itertools::Itertools; // Needed for sorting

/// Extracts global constraints information from a given `pilout` JSON structure.
/// This function replicates the `getGlobalConstraintsInfo` logic from JavaScript.
///
/// # Arguments
/// * `pilout` - A reference to the `pilout` JSON structure.
/// * `save_symbols` - A boolean indicating whether to save symbols.
///
/// # Returns
/// A `HashMap` containing `constraints` and `hints` extracted from pilout.
pub fn get_global_constraints_info(pilout: &HashMap<String, Value>, save_symbols: bool) -> HashMap<String, Value> {
    let mut constraints_code = Vec::new();
    let mut hints_code = Vec::new();
    let mut expressions = Vec::new();
    let mut symbols = Vec::new();

    // Check for constraints
    if let Some(constraints) = pilout.get("constraints").and_then(|c| c.as_array()) {
        let formatted_constraints: Vec<HashMap<String, Value>> = constraints
            .iter()
            .filter_map(|c| {
                Some(HashMap::from([
                    ("e".to_string(), json!(c["expressionIdx"]["idx"].as_u64()?)),
                    ("boundary".to_string(), json!("finalProof")),
                    ("line".to_string(), c["debugLine"].clone()),
                ]))
            })
            .collect();

        // Fetch formatted expressions and symbols
        let expr_result = format_expressions(pilout, save_symbols, true);
        expressions = expr_result["expressions"].as_array().unwrap_or(&vec![]).to_vec();

        if save_symbols {
            symbols = expr_result["symbols"].as_array().unwrap_or(&vec![]).to_vec();
        } else {
            symbols = format_symbols(pilout, true);
        }

        // Add expression info
        for constraint in &formatted_constraints {
            if let Some(e_idx) = constraint.get("e").and_then(|e| e.as_u64()) {
                add_info_expressions(&mut expressions, e_idx as usize);
            }
        }

        let mut ctx = CodeGenContext {
            stage: 0,
            calculated: HashMap::new(),
            symbols_used: Vec::new(),
            tmp_used: 0,
            code: Vec::new(),
            dom: "n".to_string(),
            air_id: json!(null),
            airgroup_id: json!(null),
            opening_points: Vec::new(),
            verifier_evaluations: false,
            ev_map: Vec::new(),
            exp_map: HashMap::new(),
        };

        // Generate constraint code
        for constraint in &formatted_constraints {
            if let Some(e_idx) = constraint.get("e").and_then(|e| e.as_u64()) {
                pil_code_gen(&mut ctx, &symbols, &expressions, e_idx as usize, 0);
                let mut code = build_code(&mut ctx);
                ctx.tmp_used = code["tmpUsed"].as_u64().unwrap_or(0) as usize;
                code["boundary"] = constraint["boundary"].clone();
                code["line"] = constraint["line"].clone();
                constraints_code.push(code);
            }
        }
    }

    // Handle global hints
    if let Some(global_hints) = pilout.get("hints").and_then(|h| h.as_array()).map(|hints| {
        hints
            .iter()
            .filter(|h| h.get("airId").is_none() && h.get("airgroupId").is_none())
            .cloned()
            .collect::<Vec<Value>>()
    }) {
        let hints = format_hints(pilout, &global_hints, &mut symbols, &mut expressions, save_symbols, true);
        let mut res = HashMap::new();
        hints_code = add_hints_info(&mut res, &mut expressions, &hints, &mut HashMap::new());
    }

    HashMap::from([("constraints".to_string(), json!(constraints_code)), ("hints".to_string(), json!(hints_code))])
}
