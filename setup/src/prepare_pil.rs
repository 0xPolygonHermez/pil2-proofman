use crate::{
    gen_constraint_pol::generate_constraint_polynomial, gen_pil1_pols::generate_pil1_polynomials,
    get_pilout_info::get_pilout_info, helpers::add_info_expressions,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

pub fn prepare_pil(
    f: fn(f64, f64) -> f64,
    pil: &mut Value,
    stark_struct: &Value,
    pil2: bool,
    options: &HashMap<String, Value>,
) -> HashMap<String, Value> {
    let keys = pil.as_object().unwrap().keys().cloned().collect::<Vec<String>>();
    println!("prepare_pil keys: {:?}", keys);
    let mut res = HashMap::new();
    res.insert("name".to_string(), pil["name"].clone());
    res.insert("imPolsStages".to_string(), options.get("imPolsStages").unwrap_or(&json!(false)).clone());

    res.insert("cmPolsMap".to_string(), json!([]));
    res.insert("constPolsMap".to_string(), json!([]));
    res.insert("challengesMap".to_string(), json!([]));
    res.insert("publicsMap".to_string(), json!([]));
    res.insert("airgroupValuesMap".to_string(), json!([]));
    res.insert("airValuesMap".to_string(), json!([]));
    res.insert("pil2".to_string(), json!(pil2));

    res.insert("mapSectionsN".to_string(), json!({ "const": 0 }));
    res.insert("pilPower".to_string(), pil["pilPower"].clone());

    let mut expressions;
    let symbols;
    let mut constraints;
    let hints;

    if let Some(expressions_array) = pil["expressions"].as_array_mut() {
        for exp in expressions_array.iter_mut() {
            exp["stage"] = json!(1);
        }
    }

    if true {
        // pil2
        let pil_map: HashMap<String, Value> =
            pil.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        let pil_info = get_pilout_info(&mut res, &pil_map);
        expressions = pil_info["expressions"].clone();
        symbols = pil_info["symbols"].clone();
        hints = pil_info["hints"].clone();
        constraints = pil_info["constraints"].clone();
    } else {
        let pil1_info = generate_pil1_polynomials(f, 1.0, &mut res, pil, Some(options));
        expressions = pil1_info["expressions"].clone();
        symbols = pil1_info["symbols"].clone();
        hints = pil1_info["hints"].clone();
        constraints = pil1_info["constraints"].clone();
    }

    let n_stages = res.get("nStages").and_then(|v| v.as_u64()).unwrap_or(0);
    if let Some(map_sections) = res.get_mut("mapSectionsN").and_then(|v| v.as_object_mut()) {
        for s in 1..=n_stages + 1 {
            map_sections.insert(format!("cm{}", s), json!(0));
        }
    }

    if !options.get("debug").unwrap_or(&json!(false)).as_bool().unwrap() {
        res.insert("starkStruct".to_string(), stark_struct.clone());

        println!("starkStruct: {:?}", stark_struct);
        if stark_struct["nBits"] != res["pilPower"] {
            panic!(
                "starkStruct and pilfile have degree mismatch (airId: {} airGroupId: {} starkStruct:{} pilfile:{})",
                pil["airId"], pil["airGroupId"], stark_struct["nBits"], res["pilPower"]
            );
        }

        if stark_struct["nBitsExt"] != stark_struct["steps"][0]["nBits"] {
            panic!(
                "starkStruct.nBitsExt and first step of starkStruct have a mismatch (nBitsExt:{} pil:{})",
                stark_struct["nBitsExt"], stark_struct["steps"][0]["nBits"]
            );
        }
    } else {
        res.insert("starkStruct".to_string(), json!({ "nBits": res["pilPower"] }));
    }

    println!("entering critical section");
    if let Some(constraints_array) = constraints.as_array_mut() {
        let mut stage_updates = Vec::new(); // Store updates instead of modifying in-place

        for constraint in constraints_array.iter() {
            let constraint_exp_id = constraint["e"].as_u64().unwrap() as usize;
            if let Some(expressions_array) = expressions.as_array_mut() {
                add_info_expressions(expressions_array, constraint_exp_id);
                stage_updates.push((constraint_exp_id, expressions_array[constraint_exp_id]["stage"].clone()));
            }
        }

        // Apply collected updates
        for (idx, stage) in stage_updates {
            constraints_array[idx]["stage"] = stage;
        }
    }
    println!("finished critical section");
    // end bad section

    let mut missing_symbol_indices = Vec::new();
    if let Some(expressions_array) = expressions.as_array() {
        for (index, exp) in expressions_array.iter().enumerate() {
            if exp.get("symbols").is_none() {
                missing_symbol_indices.push(index);
            }
        }
    }

    if let Some(expressions_array) = expressions.as_array_mut() {
        for index in missing_symbol_indices {
            add_info_expressions(expressions_array, index);
        }
    }

    res.insert("boundaries".to_string(), json!([{ "name": "everyRow" }]));

    let mut opening_points: HashSet<i64> = HashSet::new();
    if let Some(constraints_array) = constraints.as_array() {
        for constraint in constraints_array {
            let constraint_exp_id = constraint["e"].as_u64().unwrap() as usize;
            if let Some(offsets) = expressions[constraint_exp_id]["rowsOffsets"].as_array() {
                for offset in offsets.iter() {
                    if let Some(num) = offset.as_i64() {
                        opening_points.insert(num);
                    }
                }
            }
        }
    }

    let mut opening_points_vec: Vec<i64> = opening_points.into_iter().collect();
    opening_points_vec.sort_unstable();
    res.insert("openingPoints".to_string(), json!(opening_points_vec));

    // **Fix: Convert `symbols` from `Vec<Value>` to `Vec<HashMap<String, Value>>`**
    let symbols_vec: Vec<HashMap<String, Value>> = symbols
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect::<HashMap<String, Value>>())
        .collect();

    generate_constraint_polynomial(
        &mut res,
        expressions.as_array_mut().unwrap(),
        &mut symbols_vec.clone(),
        constraints.as_array().unwrap(),
    );

    serde_json::from_value(json!({
        "res": res,
        "expressions": expressions,
        "constraints": constraints,
        "symbols": symbols,
        "hints": hints
    }))
    .unwrap()
}
