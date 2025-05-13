use crate::{
    gen_constraint_pol::generate_constraint_polynomial, gen_pil1_pols::generate_pil1_polynomials,
    get_pilout_info::get_pilout_info, helpers::add_info_expressions,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

// 100% fixed
pub fn prepare_pil(
    f: fn(f64, f64) -> f64,
    pil: &mut Value,
    stark_struct: &Value,
    pil2: bool,
    options: &HashMap<String, Value>,
) -> HashMap<String, Value> {
    //tracing::info!("pil.constraints {:#?}", pil["constraints"]);
    /* ──────────────────────── header & trivial fields ─────────────────────── */
    let mut res = HashMap::<String, Value>::new();
    res.insert("name".into(), pil["name"].clone());
    res.insert("imPolsStages".into(), options.get("imPolsStages").cloned().unwrap_or(json!(false)));

    // JS starts *all* of these as empty **arrays**.
    res.insert("cmPolsMap".into(), json!([]));
    res.insert("constPolsMap".into(), json!([]));
    res.insert("challengesMap".into(), json!([]));
    res.insert("publicsMap".into(), json!([]));
    res.insert("proofValuesMap".into(), json!([])); // missing before
    res.insert("airgroupValuesMap".into(), json!([]));
    res.insert("airValuesMap".into(), json!([]));

    res.insert("pil2".into(), json!(pil2));
    res.insert("mapSectionsN".into(), json!({ "const": 0u64 }));
    res.insert("pilPower".into(), pil["pilPower"].clone());

    /* ──────────────────────── normalise top-level stages ───────────────────── */
    if let Some(exprs) = pil["expressions"].as_array_mut() {
        for e in exprs {
            e["stage"] = json!(1);
        }
    }

    /* ──────────────────────── load pil info (pil2 / pil1) ─────────────────── */
    let (mut expressions, symbols_json, hints, mut constraints) = if pil2 {
        let pil_map: HashMap<_, _> = pil.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let info = get_pilout_info(&mut res, &pil_map);
        (info["expressions"].clone(), info["symbols"].clone(), info["hints"].clone(), info["constraints"].clone())
    } else {
        let info = generate_pil1_polynomials(f, 1.0, &mut res, pil, Some(options));
        (info["expressions"].clone(), info["symbols"].clone(), info["hints"].clone(), info["constraints"].clone())
    };

    /* ──────────────────────── fill mapSectionsN.cm? ───────────────────────── */
    let n_stages = res.get("nStages").and_then(Value::as_u64).unwrap_or(0);
    if let Some(map_sect) = res.get_mut("mapSectionsN").and_then(Value::as_object_mut) {
        for s in 1..=n_stages + 1 {
            map_sect.insert(format!("cm{s}"), json!(0u64));
        }
    }

    /* ──────────────────────── stark-struct sanity checks ──────────────────── */
    let debug = options.get("debug").and_then(Value::as_bool).unwrap_or(false);
    if !debug {
        res.insert("starkStruct".into(), stark_struct.clone());

        if stark_struct["nBits"] != res["pilPower"] {
            panic!(
                "starkStruct / pil degree mismatch (airId:{}, airGroupId:{}  starkStruct:{}  pilfile:{})",
                pil["airId"], pil["airGroupId"], stark_struct["nBits"], res["pilPower"]
            );
        }
        if stark_struct["nBitsExt"] != stark_struct["steps"][0]["nBits"] {
            panic!(
                "starkStruct.nBitsExt mismatch (nBitsExt:{}  firstStep:{})",
                stark_struct["nBitsExt"], stark_struct["steps"][0]["nBits"]
            );
        }
    } else {
        res.insert("starkStruct".into(), json!({ "nBits": res["pilPower"] }));
    }

    /* ──────────────────────── enrich constraints / expressions ────────────── */
    // 1. propagate stage info from expression to constraint
    if let (Some(cons_arr), Some(expr_arr)) = (constraints.as_array_mut(), expressions.as_array_mut()) {
        for c in cons_arr.iter_mut() {
            let eid = c["e"].as_u64().unwrap() as usize;
            add_info_expressions(expr_arr, eid);
            c["stage"] = expr_arr[eid]["stage"].clone();
        }

        // 2. make sure every expression has symbols
        if let Some(expr_arr) = expressions.as_array_mut() {
            for idx in 0..expr_arr.len() {
                if expr_arr[idx].get("symbols").is_none() {
                    add_info_expressions(expr_arr, idx);
                }
            }
        }
    }

    /* ──────────────────────── boundaries & openingPoints ──────────────────── */
    res.insert("boundaries".into(), json!([{ "name": "everyRow" }]));

    let mut opening_points: HashSet<i64> = HashSet::new();
    opening_points.insert(0); // JS seeds with 0
    for c in constraints.as_array().unwrap() {
        let eid = c["e"].as_u64().unwrap() as usize;
        if let Some(rows) = expressions[eid]["rowsOffsets"].as_array() {
            for r in rows {
                if let Some(i) = r.as_i64() {
                    opening_points.insert(i);
                }
            }
        }
    }
    let mut opening_vec: Vec<_> = opening_points.into_iter().collect();
    opening_vec.sort_unstable();
    res.insert("openingPoints".into(), json!(opening_vec));

    /* ──────────────────────── convert symbols JSON → Vec<HashMap> ─────────── */
    let mut symbols_vec: Vec<HashMap<String, Value>> = symbols_json
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .collect();

    /* ──────────────────────── generate constraint polynomial ──────────────── */
    generate_constraint_polynomial(
        &mut res,
        expressions.as_array_mut().unwrap(),
        &mut symbols_vec, // ← no clone, share mutations
        constraints.as_array().unwrap(),
    );
    let symbols_json = serde_json::to_value(&symbols_vec).unwrap();

    /* ──────────────────────── return exactly the JS shape ─────────────────── */
    serde_json::from_value(json!({
        "res":         res,
        "expressions": expressions,
        "constraints": constraints,
        "symbols":     symbols_json,
        "hints":       hints
    }))
    .unwrap()
}
