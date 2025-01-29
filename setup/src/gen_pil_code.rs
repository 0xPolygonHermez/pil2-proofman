use serde_json::{json, Value};
use std::collections::HashMap;

use crate::fri_poly::generate_fri_polynomial;
use crate::helpers::print_expressions;
use crate::helpers::{add_info_expressions, add_info_expressions_symbols};
use crate::gen_code::{
    generate_constraint_polynomial_verifier_code, generate_constraints_debug_code, generate_expressions_code,
};
use crate::fri_poly::{Res, Symbol};

pub fn generate_pil_code(
    res: &mut Value,
    symbols: &mut Vec<HashMap<String, Value>>,
    constraints: &[Value],
    expressions: &mut Vec<Value>,
    hints: &[Value],
    debug: bool,
) -> HashMap<String, Value> {
    let mut expressions_info = HashMap::new();
    let mut verifier_info = json!({}); // Ensure it's a JSON object

    for i in 0..expressions.len() {
        add_info_expressions_symbols(symbols, expressions, i);
    }

    if !debug {
        let symbols_as_values: Vec<Value> = symbols.iter().map(|s| json!(s)).collect();

        generate_constraint_polynomial_verifier_code(res, &mut verifier_info, &symbols_as_values, expressions);

        // Convert `res` into `Res` for `generate_fri_polynomial`
        let mut res_struct: Res = serde_json::from_value(res.clone()).expect("Failed to deserialize res");

        // Convert `symbols` to `Vec<Symbol>`
        let mut symbols_struct: Vec<Symbol> =
            symbols.iter().map(|s| serde_json::from_value(json!(s)).expect("Failed to deserialize symbol")).collect();

        generate_fri_polynomial(&mut res_struct, &mut symbols_struct, expressions);

        // Convert `Res` back to `Value`
        *res = serde_json::to_value(res_struct).expect("Failed to serialize res back");

        // Convert `symbols_struct` back to `Vec<HashMap<String, Value>>`
        *symbols = symbols_struct
            .iter()
            .map(|s| serde_json::to_value(s).expect("Failed to serialize symbol"))
            .map(|s| s.as_object().unwrap().clone().into_iter().collect())
            .collect();

        let fri_exp_id = res["friExpId"].as_u64().expect("Missing friExpId") as usize;
        add_info_expressions(expressions, fri_exp_id);
        add_info_expressions_symbols(symbols, expressions, fri_exp_id);
    }

    // Convert `res_map` to a HashMap
    let mut res_hashmap: HashMap<String, Value> = res.as_object().unwrap().clone().into_iter().collect();
    let hints_info = add_hints_info(&mut res_hashmap, expressions, hints, &mut HashMap::new());

    // Convert back and update `res`
    *res = json!(res_hashmap);
    expressions_info.insert("hintsInfo".to_string(), json!(hints_info));

    let symbols_as_values: Vec<Value> = symbols.iter().map(|s| json!(s)).collect();
    let expressions_code = generate_expressions_code(res, &symbols_as_values, expressions);
    expressions_info.insert("expressionsCode".to_string(), json!(expressions_code));

    let fri_exp_id = res["friExpId"].as_u64().expect("Missing friExpId") as usize;
    let mut query_verifier =
        expressions_code.iter().find(|e| e["expId"] == json!(fri_exp_id)).cloned().expect("Query verifier not found");

    let tmp_used = query_verifier["tmpUsed"].as_u64().expect("Missing tmpUsed");

    let last_code_entry = query_verifier["code"]
        .as_array_mut()
        .expect("Invalid query verifier structure")
        .last_mut()
        .expect("Query verifier has no instructions");

    last_code_entry["dest"] = json!({
        "type": "tmp",
        "id": tmp_used - 1,
        "dim": 3
    });

    verifier_info["queryVerifier"] = json!(query_verifier);

    let constraints_debug_code = generate_constraints_debug_code(res, &symbols_as_values, constraints, expressions);
    expressions_info.insert("constraints".to_string(), json!(constraints_debug_code));

    json!({
        "expressionsInfo": expressions_info,
        "verifierInfo": verifier_info
    })
    .as_object()
    .expect("Expected object")
    .clone()
    .into_iter()
    .collect()
}

/// Adds hints info, processing hint fields recursively.
/// Flattens a JSON array recursively, ensuring deeply nested arrays are extracted properly.
fn flatten_json_array(value: &Value) -> Vec<Value> {
    if let Some(arr) = value.as_array() {
        arr.iter()
            .flat_map(flatten_json_array) // Recursively flatten arrays
            .collect()
    } else {
        vec![value.clone()]
    }
}

/// Adds hints info, processing hint fields recursively.
pub fn add_hints_info(
    res: &mut HashMap<String, Value>,
    expressions: &mut Vec<Value>,
    hints: &[Value],
    global: &mut HashMap<String, Value>,
) -> Vec<Value> {
    let mut hints_info = Vec::new();

    for hint in hints {
        let mut hint_fields = Vec::new();

        if let Some(fields) = hint["fields"].as_array() {
            for field in fields {
                let processed_values = process_hint_field_value(&field["values"], res, expressions, global, vec![]);
                let flattened_values = flatten_json_array(&processed_values); // ✅ Fix: Flatten manually

                let mut hint_field = json!({
                    "name": field["name"],
                    "values": flattened_values
                });

                // Ensure `pos: []` if `lengths` is missing
                if field.get("lengths").is_none() {
                    if let Some(values) = hint_field["values"].as_array_mut() {
                        if let Some(first_value) = values.get_mut(0) {
                            first_value["pos"] = json!([]);
                        }
                    }
                }

                hint_fields.push(hint_field);
            }
        }

        hints_info.push(json!({
            "name": hint["name"],
            "fields": hint_fields
        }));
    }

    res.remove("hints");

    hints_info
}

/// Processes a hint field value recursively, matching the JavaScript behavior.
pub fn process_hint_field_value(
    values: &Value,
    res: &mut HashMap<String, Value>,
    expressions: &mut Vec<Value>,
    global: &mut HashMap<String, Value>,
    pos: Vec<usize>,
) -> Value {
    let mut processed_fields = Vec::new();

    if let Some(array) = values.as_array() {
        for (j, field) in array.iter().enumerate() {
            let mut current_pos = pos.clone();
            current_pos.push(j);

            if field.is_array() {
                processed_fields.push(process_hint_field_value(field, res, expressions, global, current_pos));
            } else if let Some(op) = field.get("op").and_then(|v| v.as_str()) {
                let processed_field = match op {
                    "exp" => {
                        let id = field["id"].as_u64().expect("Invalid id") as usize;
                        let formatted_expr = print_expressions(res, &expressions[id], expressions, false);
                        expressions[id]["line"] = json!(formatted_expr);
                        json!({
                            "op": "tmp",
                            "id": id,
                            "dim": expressions[id]["dim"],
                            "pos": current_pos
                        })
                    }
                    "cm" | "custom" | "const" => {
                        let row_offset = field["rowOffset"].as_u64().unwrap_or(0); // Ensure it's `u64`
                        let prime_index = res
                            .get("openingPoints")
                            .and_then(|v| v.as_array())
                            .and_then(|arr| arr.iter().position(|p| p.as_u64() == Some(row_offset)));

                        let mut new_field = field.clone();
                        new_field["rowOffsetIndex"] = json!(prime_index.unwrap_or(usize::MAX));
                        new_field["pos"] = json!(current_pos);
                        new_field
                    }
                    "challenge" | "public" | "airgroupvalue" | "airvalue" | "number" | "string" => {
                        let mut new_field = field.clone();
                        new_field["pos"] = json!(current_pos);
                        new_field
                    }
                    _ => panic!("Invalid hint op: {}", op),
                };

                processed_fields.push(processed_field);
            } else {
                panic!("Invalid field structure: {:?}", field);
            }
        }
    }

    json!(processed_fields)
}
