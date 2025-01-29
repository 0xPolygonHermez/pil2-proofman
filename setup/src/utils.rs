use serde_json::{json, Value, Map};
use std::collections::HashMap;

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

            if let Some(lengths) = col.get("lengths").and_then(|l| l.as_array()) {
                name.push_str(&lengths.iter().map(|len| format!("[{}]", len)).collect::<String>());
            }

            if col["imPol"].as_bool().unwrap_or(false) && !is_constraint {
                let exp_id = col["expId"].as_u64().unwrap() as usize;
                return print_expressions(res, &expressions[exp_id], expressions, false);
            }

            if let Some(row_offset) = exp.get("rowOffset").and_then(|v| v.as_i64()) {
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

/// Recursively processes a hint field, extracting values and lengths.
fn process_hint_field(
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
fn format_expression(
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
fn buf2bint(buf: &Value) -> u128 {
    let empty_vec = Vec::new(); // Store the empty vector to extend its lifetime
    let buf_bytes = buf.as_array().unwrap_or(&empty_vec);

    let mut value = 0u128;
    for byte in buf_bytes {
        value = (value << 8) | byte.as_u64().unwrap_or(0) as u128;
    }

    value
}

/// Adds a symbol to the list of symbols.
fn add_symbol(pilout: &HashMap<String, Value>, symbols: &mut Vec<Value>, exp: &Value) {
    let name = format!("{}.{}", pilout["name"].as_str().unwrap_or("unknown"), exp["op"].as_str().unwrap_or("unknown"));
    symbols.push(json!({
        "name": name,
        "type": exp["op"],
        "id": exp["id"]
    }));
}
