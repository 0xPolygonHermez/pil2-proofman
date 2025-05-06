use crate::utils::{format_expressions, format_constraints, format_symbols, format_hints};
use serde_json::{json, Value};
use std::collections::HashMap;

// 100% correct
/// Extracts and processes PIL information from `pilout`, mirroring JavaScript logic.
pub fn get_pilout_info(res: &mut HashMap<String, Value>, pilout: &HashMap<String, Value>) -> HashMap<String, Value> {
    res.insert("airId".to_string(), pilout["airId"].clone());
    res.insert("airGroupId".to_string(), pilout["airGroupId"].clone());

    let constraints = format_constraints(&json!(pilout));

    let save_symbols = !pilout.contains_key("symbols");
    let (expressions, mut symbols) = if save_symbols {
        let e = format_expressions(pilout, true, false);
        (e["expressions"].clone(), e["symbols"].clone())
    } else {
        let e = format_expressions(pilout, false, false);
        (e["expressions"].clone(), json!(format_symbols(pilout, false)))
    };

    // Filter symbols
    symbols = json!(symbols
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| {
            let symbol_type = s["type"].as_str().unwrap_or("");
            symbol_type != "witness" && symbol_type != "fixed"
                || (s["airId"] == res["airId"] && s["airGroupId"] == res["airGroupId"])
        })
        .cloned()
        .collect::<Vec<Value>>());

    let air_group_values = pilout.get("airGroupValues").cloned().unwrap_or(json!([]));

    res.insert(
        "pilPower".to_string(),
        json!(pilout["numRows"].as_u64().map(|n| (n as f64).log2()).unwrap_or(0.0).round() as u32),
    );

    let witness_symbols = symbols
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["type"] == "witness" && s["airId"] == res["airId"] && s["airGroupId"] == res["airGroupId"])
        .count();

    let fixed_symbols = symbols
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["type"] == "fixed" && s["airId"] == res["airId"] && s["airGroupId"] == res["airGroupId"])
        .count();

    let public_symbols = symbols.as_array().unwrap().iter().filter(|s| s["type"] == "public").count();

    res.insert("nCommitments".to_string(), json!(witness_symbols));
    res.insert("nConstants".to_string(), json!(fixed_symbols));
    res.insert("nPublics".to_string(), json!(public_symbols));
    res.insert("airGroupValues".to_string(), air_group_values.clone());

    // --- nStages -------------------------------------------------------------
    // • If pilout.numChallenges exists ➜ its length.
    // • Otherwise ➜ the maximum stage appearing in symbols (no “+ 1” like before).
    let n_stages_val = if let Some(arr) = pilout.get("numChallenges").and_then(Value::as_array) {
        arr.len()
    } else {
        symbols.as_array().and_then(|syms| syms.iter().map(|s| s["stage"].as_u64().unwrap_or(0)).max()).unwrap_or(0)
            as usize
    };

    res.insert("nStages".to_string(), json!(n_stages_val));

    let mut pilout = pilout.clone();
    if !pilout.contains_key("hints") {
        pilout.insert("hints".to_string(), json!([]));
    }
    let air_hints = pilout["hints"]
        .as_array()
        .map(|hints| {
            hints
                .iter()
                .filter(|h| h["airId"] == res["airId"] && h["airGroupId"] == res["airGroupId"])
                .cloned()
                .collect::<Vec<Value>>()
        })
        .unwrap_or_default();

    let mut symbols_vec = symbols.as_array().unwrap().clone();
    let mut expressions_vec = expressions.as_array().unwrap().clone();

    let hints = format_hints(&pilout, &air_hints, &mut symbols_vec, &mut expressions_vec, save_symbols, false);

    res.insert("customCommits".to_string(), pilout.get("customCommits").cloned().unwrap_or(json!([])));

    let mut custom_commits_map = vec![];

    if let Some(commits) = res.remove("customCommits").and_then(|v| v.as_array().cloned()) {
        for commit in commits.iter() {
            let mut commit_map = vec![];
            if let Some(stage_widths) = commit["stageWidths"].as_array() {
                for (j, width) in stage_widths.iter().enumerate() {
                    if width.as_u64().unwrap_or(0) > 0 {
                        res.entry("mapSectionsN".to_string())
                            .or_insert_with(|| json!(HashMap::<String, Value>::new()))
                            .as_object_mut()
                            .unwrap()
                            .insert(format!("{}{}", commit["name"], j), json!(0));
                    }
                    commit_map.push(json!({}));
                }
            }
            custom_commits_map.push(json!(commit_map));
        }
        res.insert("customCommits".to_string(), json!(commits)); // Put it back after processing
    }

    res.insert("customCommitsMap".to_string(), json!(custom_commits_map));

    json!({
        "expressions": expressions,
        "hints": hints,
        "constraints": constraints,
        "symbols": symbols
    })
    .as_object()
    .unwrap()
    .clone()
    .into_iter()
    .map(|(k, v)| (k.clone(), v.clone())) // Convert `serde_json::Map` to `HashMap<String, Value>`
    .collect()
}

use crate::utils::buf2bint;
use std::fs;
use std::path::Path;

/// Processes fixed polynomials from PIL2 and saves them to a file.
pub fn get_fixed_pols_pil2(
    files_dir: &str,
    pil: &HashMap<String, Value>,
    cnst_pols: &mut HashMap<String, Vec<Value>>,
) -> std::io::Result<()> {
    // Clone the def_array to avoid immutable + mutable borrow conflict
    let def_array = cnst_pols.get("$$defArray").cloned().unwrap_or_default();

    for def in &def_array {
        let id = def["id"].as_u64().expect("Missing `id` in def array");
        let deg = def["polDeg"].as_u64().expect("Missing `polDeg` in def array") as usize;

        let fixed_cols = pil
            .get("fixedCols")
            .and_then(|cols| cols.as_array())
            .expect("Missing `fixedCols` in PIL2 file")
            .get(id as usize)
            .expect("Invalid fixed column index");

        if let Some(values) = fixed_cols.get("values").and_then(|v| v.as_array()) {
            let const_pol = cnst_pols.entry(id.to_string()).or_insert_with(|| vec![json!(0); deg]);

            for (j, value) in values.iter().enumerate().take(deg) {
                const_pol[j] = json!(buf2bint(value));
            }
        }
    }

    let pil_name = pil["name"].as_str().expect("Missing `name` in PIL file");
    let file_path = Path::new(files_dir).join(format!("{}.const", pil_name));

    let json_data = serde_json::to_string_pretty(&cnst_pols)?;
    fs::write(file_path, json_data)?;

    Ok(())
}
