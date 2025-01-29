use crate::{
    gen_constraint_pol::generate_constraint_polynomial, gen_pil1_pols::generate_pil1_polynomials,
    get_pilout_info::get_pilout_info, helpers::add_info_expressions,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// Prepares the PIL (Polynomial Identity Language) structure.
/// Mirrors the original JavaScript implementation.
pub fn prepare_pil(
    f: fn(f64, f64) -> f64, // Field multiplication function
    pil: &Value,
    stark_struct: &Value,
    pil2: bool,
    options: &HashMap<String, Value>,
) -> HashMap<String, Value> {
    let mut res = HashMap::new();
    res.insert("name".to_string(), pil["name"].clone());
    res.insert("imPolsStages".to_string(), options.get("imPolsStages").unwrap_or(&json!(false)).clone());

    // Initialize mappings
    res.insert("cmPolsMap".to_string(), json!([]));
    res.insert("constPolsMap".to_string(), json!([]));
    res.insert("challengesMap".to_string(), json!([]));
    res.insert("publicsMap".to_string(), json!([]));
    res.insert("airgroupValuesMap".to_string(), json!([]));
    res.insert("airValuesMap".to_string(), json!([]));
    res.insert("pil2".to_string(), json!(pil2));

    let mut map_sections_n = HashMap::new();
    map_sections_n.insert("const".to_string(), json!(0));

    let mut expressions;
    let mut symbols;
    let mut constraints;
    let hints; // Removed `mut`, since it's not modified

    // Ensure all expressions have `stage = 1`
    let mut pil_expressions = pil["expressions"].as_array().unwrap().clone();
    for exp in &mut pil_expressions {
        exp["stage"] = json!(1);
    }

    if pil2 {
        let mut pil_hashmap: HashMap<String, Value> = serde_json::from_value(pil.clone()).unwrap();
        let mut res_mut = res.clone();
        let pil_info = get_pilout_info(&mut res_mut, &mut pil_hashmap);
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

    // Set up section counts
    if let Some(n_stages) = res.get("nStages").and_then(|v| v.as_u64()) {
        for s in 1..=n_stages + 1 {
            map_sections_n.insert(format!("cm{}", s), json!(0));
        }
    }

    // Handle stark struct consistency checks
    if !options.get("debug").unwrap_or(&json!(false)).as_bool().unwrap() {
        res.insert("starkStruct".to_string(), stark_struct.clone());

        if stark_struct["nBits"] != res["pilPower"] {
            panic!(
                "starkStruct and pilfile have degree mismatch (airId: {} airgroupId: {} starkStruct:{} pilfile:{})",
                pil["airId"], pil["airgroupId"], stark_struct["nBits"], res["pilPower"]
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

    // Process constraints
    if let Some(constraints_array) = constraints.as_array_mut() {
        for constraint in constraints_array.iter_mut() {
            let constraint_exp_id = constraint["e"].as_u64().unwrap() as usize;
            if let Some(expressions_array) = expressions.as_array_mut() {
                add_info_expressions(expressions_array, constraint_exp_id);
                constraint["stage"] = expressions_array[constraint_exp_id]["stage"].clone();
            }
        }
    }

    // **Fix Borrowing Issue in expressions Processing**
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

    // Collect unique opening points
    let mut opening_points: HashSet<i64> = HashSet::new();
    if let Some(constraints_array) = constraints.as_array() {
        for constraint in constraints_array.iter() {
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
    opening_points_vec.sort();
    res.insert("openingPoints".to_string(), json!(opening_points_vec));

    // **Fixed `generate_constraint_polynomial` Call**
    if let (Some(expressions_array), Some(symbols_array)) = (expressions.as_array_mut(), symbols.as_array_mut()) {
        let mut parsed_symbols: Vec<HashMap<String, Value>> =
            symbols_array.iter_mut().map(|s| serde_json::from_value(s.clone()).unwrap()).collect();

        generate_constraint_polynomial(
            &mut res,
            expressions_array,
            &mut parsed_symbols,
            constraints.as_array().unwrap(),
        );
    }

    serde_json::from_value(json!({
        "res": res,
        "expressions": expressions,
        "constraints": constraints,
        "symbols": symbols,
        "hints": hints
    }))
    .unwrap()
}
