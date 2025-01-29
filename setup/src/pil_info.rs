use crate::{
    add_intermediate_pols::add_intermediate_polynomials, gen_pil_code::generate_pil_code, mapping::map,
    prepare_pil::prepare_pil,
};
use serde_json::{json, Value, Map};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::NamedTempFile;

/// Translates `pilInfo` function from JavaScript to Rust.
pub async fn pil_info(
    f: fn(f64, f64) -> f64,
    pil: &Value,
    pil2: bool,
    stark_struct: &Value,
    options: HashMap<String, Value>,
) -> HashMap<String, Value> {
    let mut pil_clone = pil.clone();
    let mut info_pil = prepare_pil(f, &mut pil_clone, stark_struct, pil2, &options);

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

    let mut new_expressions = expressions.clone();
    let max_deg = (1
        << (res["starkStruct"]["nBitsExt"].as_u64().unwrap() as usize
            - res["starkStruct"]["nBits"].as_u64().unwrap() as usize))
        + 1;

    if !options.get("debug").unwrap_or(&json!(false)).as_bool().unwrap()
        || !options.get("skipImPols").unwrap_or(&json!(false)).as_bool().unwrap()
    {
        let mut im_info: Value;

        if options.get("optImPols").unwrap_or(&json!(false)).as_bool().unwrap() {
            let info_pil_file = NamedTempFile::new().expect("Failed to create temp file");
            let im_pols_file = NamedTempFile::new().expect("Failed to create temp file");

            let max_deg = (1
                << (stark_struct["nBitsExt"].as_u64().unwrap() as usize
                    - stark_struct["nBits"].as_u64().unwrap() as usize))
                + 1;

            let info_pil_json = json!({
                "maxDeg": max_deg,
                "cExpId": res["cExpId"],
                "qDim": res["qDim"],
                "infoPil": info_pil
            });

            fs::write(info_pil_file.path(), info_pil_json.to_string()).expect("Failed to write temp file");

            let calculate_im_pols_path = Path::new("./imPolsCalculation/calculateImPols.py");

            let output = Command::new("python3")
                .arg(calculate_im_pols_path)
                .arg(info_pil_file.path())
                .arg(im_pols_file.path())
                .output()
                .expect("Failed to execute Python script");

            println!("{}", String::from_utf8_lossy(&output.stdout));

            im_info = serde_json::from_str(
                &fs::read_to_string(im_pols_file.path()).expect("Failed to read intermediate polynomials file"),
            )
            .expect("Failed to parse JSON");

            let _ = fs::remove_file(info_pil_file.path());
            let _ = fs::remove_file(im_pols_file.path());
        } else {
            unimplemented!("we didn't translate the python code, shouldn't need it");
        }

        new_expressions = im_info["newExpressions"].as_array().unwrap().clone();

        let im_exps: Vec<usize> =
            im_info["imExps"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();

        add_intermediate_polynomials(
            &mut res,
            &mut new_expressions,
            &mut constraints,
            &mut symbols,
            &im_exps,
            im_info["qDeg"].as_u64().unwrap() as usize,
        );
    }

    map(
        &mut res,
        &symbols.iter().map(|s| json!(s)).collect::<Vec<Value>>(),
        &expressions,
        &mut constraints,
        &json!(options),
    );

    let pil_code = generate_pil_code(
        &mut res,
        &mut symbols,
        &constraints,
        &mut new_expressions,
        &hints,
        options.get("debug").unwrap_or(&json!(false)).as_bool().unwrap(),
    );

    let expressions_info = pil_code["expressionsInfo"].clone();
    let verifier_info = pil_code["verifierInfo"].clone();

    let mut n_cols = HashMap::new();
    let mut summary = String::new();

    println!("------------------------- AIR INFO -------------------------");

    let mut n_columns_base_field = 0;
    let mut n_columns = 0;
    summary.push_str(&format!(
        "nBits: {} | blowUpFactor: {} | maxConstraintDegree: {} ",
        res["starkStruct"]["nBits"],
        res["starkStruct"]["nBitsExt"].as_u64().unwrap() - res["starkStruct"]["nBits"].as_u64().unwrap(),
        res["qDeg"].as_u64().unwrap() + 1
    ));

    for i in 1..=res["nStages"].as_u64().unwrap() as usize + 1 {
        let stage = i;
        let stage_debug =
            if i == res["nStages"].as_u64().unwrap() as usize + 1 { "Q".to_string() } else { format!("{}", stage) };
        let stage_name = format!("cm{}", stage);
        let n_cols_stage = res["cmPolsMap"].as_array().unwrap().iter().filter(|p| p["stage"] == json!(stage)).count();
        n_cols.insert(stage_name.clone(), n_cols_stage);
        let n_cols_base_field = res["mapSectionsN"][&stage_name].as_u64().unwrap() as usize;
        let im_pols: Vec<&Value> = res["cmPolsMap"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["stage"] == json!(stage) && p["imPol"].as_bool().unwrap_or(false))
            .collect();

        if i == res["nStages"].as_u64().unwrap() as usize + 1
            || (i < res["nStages"].as_u64().unwrap() as usize && !res["imPolsStages"].is_null())
        {
            println!(
                "Columns stage {}: {} -> Columns in the basefield: {}",
                stage_debug, n_cols_stage, n_cols_base_field
            );
        } else {
            println!(
                "Columns stage {}: {} ({} intermediate polynomials) -> Columns in the basefield: {} ({} from intermediate polynomials)",
                stage_debug,
                n_cols_stage,
                im_pols.len(),
                n_cols_base_field,
                im_pols.iter().map(|p| p["dim"].as_u64().unwrap_or(0)).sum::<u64>()
            );
        }

        summary.push_str(&format!(
            "| Stage{}: {} ",
            if i == res["nStages"].as_u64().unwrap() as usize + 1 { "Q".to_string() } else { i.to_string() },
            n_cols_base_field
        ));
        n_columns += n_cols_stage;
        n_columns_base_field += n_cols_base_field;
    }

    let final_output: HashMap<String, Value> = json!({
        "pilInfo": res,
        "expressionsInfo": expressions_info,
        "verifierInfo": verifier_info,
        "stats": {
            "summary": summary,
            "intermediatePolynomials": res["imPolsInfo"]
        }
    })
    .as_object()
    .unwrap()
    .iter()
    .map(|(k, v)| (k.clone(), v.clone()))
    .collect();

    final_output
}
