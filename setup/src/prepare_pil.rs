use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

use crate::{
    gen_constraint_pol::generate_constraint_polynomial, gen_pil1_pols::generate_pil1_polynomials,
    get_pilout_info::get_pilout_info, helpers::add_info_expressions,
};

/// Rust port of JS `preparePil` (commented-out degree checks remain omitted).
///
/// Returns `{ res, expressions, constraints, symbols, hints }`.
pub fn prepare_pil(
    f: fn(f64, f64) -> f64,
    pil: &mut Value,
    stark_struct: &Value,
    pil2: bool,
    options: &HashMap<String, Value>,
) -> Value {
    // ───────── 1. build `res` skeleton ────────────────────────────────────────
    let mut res: HashMap<String, Value> = HashMap::new();
    res.insert("name".into(), pil["name"].clone());
    res.insert("imPolsStages".into(), options.get("imPolsStages").cloned().unwrap_or(json!(false)));
    for k in [
        "cmPolsMap",
        "constPolsMap",
        "challengesMap",
        "publicsMap",
        "proofValuesMap",
        "airgroupValuesMap",
        "airValuesMap",
    ] {
        res.insert(k.into(), json!([]));
    }
    res.insert("pil2".into(), json!(pil2));
    res.insert("mapSectionsN".into(), json!({ "const": 0 }));
    if let Some(pp) = pil.get("pilPower") {
        res.insert("pilPower".into(), pp.clone());
    }

    // ───────── 2. mark every input-expression as stage 1 ──────────────────────
    if let Some(arr) = pil["expressions"].as_array_mut() {
        for e in arr {
            e["stage"] = json!(1);
        }
    }

    // ───────── 3. load helper info (pil-2 Value vs pil-1 HashMap) ─────────────
    let (mut expressions, symbols_val, hints, mut constraints_val) = if pil2 {
        // ❶ get_pilout_info → Value::Object
        let mut info_val = get_pilout_info(&mut res, pil); // Value
        let obj = info_val.as_object_mut().expect("get_pilout_info must return an object");

        (
            obj.remove("expressions").expect("expressions"),
            obj.remove("symbols").expect("symbols"),
            obj.remove("hints").unwrap_or(json!([])),
            obj.remove("constraints").expect("constraints"),
        )
    } else {
        // ❷ generate_pil1_polynomials → HashMap<String, Value>
        let mut info_map = generate_pil1_polynomials(f, 1.0, &mut res, pil, Some(options)); // HashMap

        (
            info_map.remove("expressions").expect("expressions"),
            info_map.remove("symbols").expect("symbols"),
            info_map.remove("hints").unwrap_or(json!([])),
            info_map.remove("constraints").expect("constraints"),
        )
    };

    // ───────── 4. fill res.mapSectionsN["cmN"]=0, N=1..=nStages+1 ─────────────
    let n_stages = res.get("nStages").and_then(Value::as_u64).unwrap_or(0);
    if let Some(ms) = res.get_mut("mapSectionsN").and_then(Value::as_object_mut) {
        for s in 1..=n_stages + 1 {
            ms.insert(format!("cm{}", s), json!(0));
        }
    }

    // ───────── 5. starkStruct sanity checks (unless debug) ────────────────────
    let debug = options.get("debug").and_then(Value::as_bool).unwrap_or(false);
    if !debug {
        res.insert("starkStruct".into(), stark_struct.clone());

        if stark_struct["nBits"] != res["pilPower"] {
            panic!(
                "nBits mismatch (airId:{} airGroupId:{} starkStruct:{} pilfile:{})",
                pil["airId"], pil["airGroupId"], stark_struct["nBits"], res["pilPower"]
            );
        }
        if stark_struct["nBitsExt"] != stark_struct["steps"][0]["nBits"] {
            panic!("nBitsExt mismatch ({} vs {})", stark_struct["nBitsExt"], stark_struct["steps"][0]["nBits"]);
        }
    } else {
        res.insert("starkStruct".into(), json!({ "nBits": res["pilPower"] }));
    }

    // ───────── 6. addInfoExpressions per constraint & set stage ───────────────
    if let (Some(con_arr), Some(expr_arr)) = (constraints_val.as_array_mut(), expressions.as_array_mut()) {
        for c in con_arr {
            let e_idx = c["e"].as_u64().unwrap() as usize;
            add_info_expressions(expr_arr, e_idx);
            c["stage"] = expr_arr[e_idx]["stage"].clone();
        }
    }

    // ───────── 7. ensure each expression has symbols ──────────────────────────
    if let Some(expr_arr) = expressions.as_array_mut() {
        for idx in 0..expr_arr.len() {
            if expr_arr[idx].get("symbols").is_none() {
                add_info_expressions(expr_arr, idx);
            }
        }
    }

    // ───────── 8. boundaries + openingPoints (seed 0) ─────────────────────────
    res.insert("boundaries".into(), json!([{ "name": "everyRow" }]));

    let mut opens: HashSet<i64> = HashSet::from([0]);
    if let (Some(con_arr), Some(expr_arr)) = (constraints_val.as_array(), expressions.as_array()) {
        for c in con_arr {
            let e_idx = c["e"].as_u64().unwrap() as usize;
            for off in expr_arr[e_idx]["rowsOffsets"].as_array().unwrap() {
                opens.insert(off.as_i64().unwrap());
            }
        }
    }
    let mut opens_vec: Vec<i64> = opens.into_iter().collect();
    opens_vec.sort_unstable();
    res.insert("openingPoints".into(), json!(opens_vec));

    // ───────── 9. symbols Vec<Value> → Vec<HashMap<…>> ────────────────────────
    let mut symbols: Vec<HashMap<String, Value>> = symbols_val
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .collect();

    // ─────────10. generateConstraintPolynomial(res, …) ────────────────────────
    let expr_vec = expressions.as_array_mut().unwrap();
    let con_vec = constraints_val.as_array_mut().unwrap();

    generate_constraint_polynomial(&mut res, expr_vec, &mut symbols, con_vec);

    // ─────────11. final return identical to JS ────────────────────────────────
    json!({
        "res":         res,
        "expressions": expressions,
        "constraints": constraints_val,
        "symbols":     symbols,
        "hints":       hints
    })
}
