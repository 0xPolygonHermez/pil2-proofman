use crate::{
    add_intermediate_pols::add_intermediate_polynomials, calculate_im_pols::process_pil_data,
    gen_pil_code::generate_pil_code, im_pols::calculate_intermediate_polynomials, mapping::map,
    prepare_pil::prepare_pil,
};
use serde_json::{json, Value};
use std::collections::HashMap;
// use std::fs;
// use std::path::Path;
// use std::process::Command;
// use tempfile::NamedTempFile;

/// Translates `pilInfo` function from JavaScript to Rust.
pub async fn pil_info(
    f: fn(f64, f64) -> f64,
    pil: &Value,
    pil2: bool,
    stark_struct: &Value,
    options: HashMap<String, Value>,
) -> HashMap<String, Value> {
    let mut pil_clone = pil.clone();
    // tracing::info!("pil.constraints {:#?}", pil_clone["constraints"]);
    let info_pil = prepare_pil(f, &mut pil_clone, stark_struct, pil2, &options);

    let mut expressions = info_pil["expressions"].as_array().unwrap().clone();
    let mut constraints = info_pil["constraints"].as_array().unwrap().clone();
    let hints = info_pil["hints"].as_array().unwrap().clone();
    let mut symbols: Vec<HashMap<String, Value>> = info_pil["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_object().unwrap().clone().into_iter().collect())
        .collect();
    let mut res: HashMap<String, Value> = info_pil["res"].as_object().unwrap().clone().into_iter().collect();

    let max_deg = (1
        << (res["starkStruct"]["nBitsExt"].as_u64().unwrap() as usize
            - res["starkStruct"]["nBits"].as_u64().unwrap() as usize))
        + 1;

    if !options.get("debug").unwrap_or(&json!(false)).as_bool().unwrap()
        || !options.get("skipImPols").unwrap_or(&json!(false)).as_bool().unwrap()
    {
        let im_info: serde_json::Value;

        if options.get("optImPols").unwrap_or(&json!(false)).as_bool().unwrap() {
            let info_pil_json = json!({
                "maxDeg": max_deg,
                "cExpId": res["cExpId"],
                "qDim": res["qDim"],
                "infoPil": info_pil,
                "expressions": expressions,
            });

            // Call our Rust function instead of invoking Python
            let im_info_str = process_pil_data(&info_pil_json.to_string());
            im_info = serde_json::from_str(&im_info_str).expect("Failed to parse JSON");

            expressions = im_info["newExpressions"].as_array().unwrap().clone();
        } else {
            im_info = calculate_intermediate_polynomials(
                &mut expressions,
                res["cExpId"].as_u64().unwrap() as usize,
                max_deg,
                res["qDim"].as_u64().unwrap() as usize,
            );
        }
        let im_exps: Vec<usize> =
            im_info["imExps"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();

        add_intermediate_polynomials(
            &mut res,
            &mut expressions,
            &mut constraints,
            &mut symbols,
            &im_exps,
            im_info["qDeg"].as_i64().unwrap(),
        );
    }

    map(
        &mut res,
        &symbols.iter().map(|s| json!(s)).collect::<Vec<Value>>(),
        &expressions,
        &mut constraints,
        &json!(options),
    );

    let mut res_value = json!(res);

    let pil_code = generate_pil_code(
        &mut res_value,
        &mut symbols,
        &constraints,
        &mut expressions,
        &hints,
        options.get("debug").unwrap_or(&json!(false)).as_bool().unwrap(),
    );

    let expressions_info = pil_code["expressionsInfo"].clone();
    let verifier_info = pil_code["verifierInfo"].clone();

    let mut n_cols = HashMap::new();
    let mut summary = String::new();

    println!("------------------------- AIR INFO -------------------------");

    let _n_columns_base_field = 0;
    let _n_columns = 0;
    summary.push_str(&format!(
        "nBits: {} | blowUpFactor: {} | maxConstraintDegree: {} ",
        res["starkStruct"]["nBits"],
        res["starkStruct"]["nBitsExt"].as_u64().unwrap() - res["starkStruct"]["nBits"].as_u64().unwrap(),
        res["qDeg"].as_u64().unwrap() + 1
    ));

    for i in 1..=res["nStages"].as_u64().unwrap() as usize + 1 {
        let stage_debug =
            if i == res["nStages"].as_u64().unwrap() as usize + 1 { "Q".to_string() } else { format!("{}", i) };
        let stage_name = format!("cm{}", i);
        let n_cols_stage = res["cmPolsMap"].as_array().unwrap().iter().filter(|p| p["stage"] == json!(i)).count();
        n_cols.insert(stage_name.clone(), n_cols_stage);

        summary.push_str(&format!("| Stage{}: {} ", stage_debug, res["mapSectionsN"][&stage_name].as_u64().unwrap()));
    }

    let final_output: HashMap<String, Value> = json!({
        "pilInfo": res_value,
        "expressionsInfo": expressions_info,
        "verifierInfo": verifier_info,
        "stats": {
            "summary": summary,
            "intermediatePolynomials": res_value["imPolsInfo"]
        }
    })
    .as_object()
    .unwrap()
    .iter()
    .map(|(k, v)| (k.clone(), v.clone()))
    .collect();

    final_output
}
