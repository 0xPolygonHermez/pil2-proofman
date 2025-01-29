use serde_json::{json, Value};
use std::collections::HashMap;

use crate::helpers::print_expressions;

/// Adds hints info, processing hint fields recursively.
/// Flattens a JSON array recursively, ensuring deeply nested arrays are extracted properly.
fn flatten_json_array(value: &Value) -> Vec<Value> {
    if let Some(arr) = value.as_array() {
        arr.iter()
            .flat_map(|v| flatten_json_array(v)) // Recursively flatten arrays
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
