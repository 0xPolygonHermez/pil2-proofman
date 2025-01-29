use serde_json::{Map, Value};
use std::collections::HashMap;
use crate::utils::print_expressions;

pub fn map(
    res: &mut HashMap<String, Value>,
    symbols: &[Value],
    expressions: &[Value],
    constraints: &mut [Value],
    options: &Value,
) {
    map_symbols(res, symbols);
    set_stage_info_symbols(res, symbols);

    for constraint in constraints.iter_mut() {
        if let Some(filename) = constraint["filename"].as_str() {
            if filename == format!("{}.ImPol", res["name"].as_str().unwrap()) {
                constraint["imPol"] = Value::Bool(true);
                if !options["recursion"].as_bool().unwrap_or(false) {
                    constraint["line"] = Value::String(print_expressions(
                        res,
                        &expressions[constraint["e"].as_u64().unwrap() as usize],
                        expressions,
                        true,
                    ));
                } else {
                    constraint["line"] = Value::String(String::new());
                }
            }
        }
        if let Some(line) = constraint["line"].as_str() {
            constraint["line"] = Value::String(format!("{} == 0", line));
        }
    }

    println!("----------------- INTERMEDIATE POLYNOMIALS -----------------");

    res.entry("imPolsInfo".to_string()).or_insert_with(|| {
        let mut im_pols_info = Map::new();
        im_pols_info.insert("baseField".to_string(), Value::Array(vec![]));
        im_pols_info.insert("extendedField".to_string(), Value::Array(vec![]));
        Value::Object(im_pols_info)
    });

    let im_pols: Vec<Value> = res
        .get("cmPolsMap")
        .and_then(|v| v.as_array())
        .unwrap_or(&vec![])
        .iter()
        .filter(|i| i.get("imPol").and_then(Value::as_bool).unwrap_or(false))
        .cloned()
        .collect();

    for (i, im_pol) in im_pols.iter().enumerate() {
        if !options["recursion"].as_bool().unwrap_or(false) {
            let im_pol_expression =
                print_expressions(res, &expressions[im_pol["expId"].as_u64().unwrap() as usize], expressions, false);
            if i > 0 {
                println!("------------------------------------------------------------");
            }
            println!("Intermediate polynomial {} columns: {}", i, im_pol["dim"]);
            println!("{}", im_pol_expression);

            if let Some(im_pols_info) = res.get_mut("imPolsInfo").and_then(Value::as_object_mut) {
                if im_pol["dim"].as_u64().unwrap() == 1 {
                    if let Some(base_field) = im_pols_info.get_mut("baseField").and_then(Value::as_array_mut) {
                        base_field.push(Value::String(im_pol_expression));
                    }
                } else if let Some(extended_field) = im_pols_info.get_mut("extendedField").and_then(Value::as_array_mut)
                {
                    extended_field.push(Value::String(im_pol_expression));
                }
            }
        }
    }

    let cm1_count = res
        .get("cmPolsMap")
        .and_then(|v| v.as_array())
        .unwrap_or(&vec![])
        .iter()
        .filter(|p| {
            p.get("stage").map(|s| s == "cm1").unwrap_or(false)
                && !p.get("imPol").and_then(Value::as_bool).unwrap_or(false)
        })
        .count();

    res.insert("nCommitmentsStage1".to_string(), Value::Number(cm1_count.into()));
}

fn map_symbols(res: &mut HashMap<String, Value>, symbols: &[Value]) {
    for symbol in symbols.iter() {
        let symbol_type = symbol["type"].as_str().unwrap_or("");
        match symbol_type {
            "fixed" | "witness" | "custom" => {
                add_pol(res, symbol);
            }
            "challenge" => {
                res.entry("challengesMap".to_string())
                    .or_insert_with(|| Value::Object(Map::new()))
                    .as_object_mut()
                    .unwrap()
                    .insert(symbol["id"].to_string(), symbol.clone());
            }
            "public" => {
                res.entry("publicsMap".to_string())
                    .or_insert_with(|| Value::Object(Map::new()))
                    .as_object_mut()
                    .unwrap()
                    .insert(symbol["id"].to_string(), symbol.clone());
            }
            "airgroupvalue" => {
                res.entry("airgroupValuesMap".to_string())
                    .or_insert_with(|| Value::Object(Map::new()))
                    .as_object_mut()
                    .unwrap()
                    .insert(symbol["id"].to_string(), symbol.clone());
            }
            "airvalue" => {
                res.entry("airValuesMap".to_string())
                    .or_insert_with(|| Value::Object(Map::new()))
                    .as_object_mut()
                    .unwrap()
                    .insert(symbol["id"].to_string(), symbol.clone());
            }
            _ => {}
        }
    }
}

fn add_pol(res: &mut HashMap<String, Value>, symbol: &Value) {
    let ref_map = match symbol["type"].as_str().unwrap() {
        "fixed" => "constPolsMap",
        "witness" => "cmPolsMap",
        "custom" => "customCommitsMap",
        _ => panic!("Invalid symbol type"),
    };

    let pos = symbol["polId"].as_u64().unwrap() as usize;
    let mut entry = Map::from_iter(vec![
        ("stage".to_string(), symbol["stage"].clone()),
        ("name".to_string(), symbol["name"].clone()),
        ("dim".to_string(), symbol["dim"].clone()),
        ("polsMapId".to_string(), Value::Number(pos.into())),
    ]);

    if let Some(stage_id) = symbol["stageId"].as_u64() {
        entry.insert("stageId".to_string(), Value::Number(stage_id.into()));
    }

    if let Some(lengths) = symbol["lengths"].as_array() {
        entry.insert("lengths".to_string(), Value::Array(lengths.clone()));
    }

    if let Some(im_pol) = symbol["imPol"].as_bool() {
        entry.insert("imPol".to_string(), Value::Bool(im_pol));
        entry.insert("expId".to_string(), symbol["expId"].clone());
    }

    res.entry(ref_map.to_string())
        .or_insert_with(|| Value::Array(vec![]))
        .as_array_mut()
        .unwrap()
        .push(Value::Object(entry));
}

fn set_stage_info_symbols(res: &mut HashMap<String, Value>, symbols: &[Value]) {
    let q_stage = res["nStages"].as_u64().unwrap_or(0) + 1;

    for symbol in symbols.iter() {
        let symbol_type = symbol["type"].as_str().unwrap_or("");
        if !["fixed", "witness", "custom"].contains(&symbol_type) {
            continue;
        }

        if symbol_type == "witness" || symbol_type == "custom" {
            let pols_map_opt = if symbol_type == "witness" {
                res.get_mut("cmPolsMap").and_then(Value::as_array_mut)
            } else {
                let commit_id = symbol["commitId"].as_u64().unwrap();
                res.get_mut("customCommitsMap")
                    .and_then(|m| m.get_mut(commit_id.to_string()))
                    .and_then(Value::as_array_mut)
            };

            if let Some(pols_map) = pols_map_opt {
                let pol_id = symbol["polId"].as_u64().unwrap() as usize;

                let stage_pos: u64 = pols_map
                    .iter()
                    .take(pol_id)
                    .filter(|p| p["stage"] == symbol["stage"])
                    .map(|p| p["dim"].as_u64().unwrap_or(0))
                    .sum();

                let stage_id = if symbol["stageId"].is_null() {
                    if symbol["stage"] == Value::Number(q_stage.into()) {
                        pols_map.iter().take(pol_id).filter(|p| p["stage"] == symbol["stage"]).count()
                    } else {
                        pols_map.iter().position(|p| p["name"] == symbol["name"]).unwrap_or(0)
                    }
                } else {
                    symbol["stageId"].as_u64().unwrap() as usize
                };

                if let Some(pols_entry) = pols_map.get_mut(pol_id) {
                    pols_entry["stagePos"] = Value::Number(stage_pos.into());
                    pols_entry["stageId"] = Value::Number(stage_id.into());
                }
            }
        }
    }
}
