use serde_json::{json, Value};
use std::collections::HashMap;

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
                        expressions[id]["line"] = print_expressions(res, &expressions[id], expressions);
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

/// Dummy function to simulate `printExpressions`
/// This should be implemented based on how expressions are printed in the PIL code.
fn print_expressions(_res: &mut HashMap<String, Value>, _expr: &Value, _expressions: &Vec<Value>) -> Value {
    json!("") // Placeholder: Replace with the actual implementation
}
