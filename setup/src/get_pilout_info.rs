use crate::utils::{buf2bint, format_constraints, format_expressions, format_hints, format_symbols};
use serde_json::{json, Value};
use std::{collections::HashMap, fs, path::Path};

/// Extracts and processes PIL information from `pilout`, mirroring JavaScript logic.
/// Rust equivalent of JS `getPiloutInfo(res, pilout)`
///
/// *Everything* is 1-to-1 with the JS except that Rust returns a single
/// `serde_json::Value` object containing `{ expressions, hints, constraints, symbols }`.
/// Rust equivalent of JS `getPiloutInfo(res, pilout)`
pub fn get_pilout_info(res: &mut HashMap<String, Value>, pilout: &Value) -> Value {
    // ─── 1.  copy AIR identifiers ────────────────────────────────────────────
    res.insert("airId".into(), pilout["airId"].clone());
    res.insert("airgroupId".into(), pilout["airgroupId"].clone());

    // ─── 2.  constraints (always needed) ─────────────────────────────────────
    let constraints = format_constraints(pilout); // Vec<Value>

    // ─── 3.  expressions (+ maybe symbols) ───────────────────────────────────
    let save_symbols = pilout.get("symbols").is_none();
    let expr_pkg = format_expressions(pilout, save_symbols, /*global=*/ false);

    // Convert expressions → Vec<Value>
    let mut expressions_vec: Vec<Value> =
        expr_pkg["expressions"].as_array().expect("expressions must be array").clone();

    // Collect symbols as Vec<Value>
    let mut symbols_val: Vec<Value> = if save_symbols {
        expr_pkg["symbols"].as_array().expect("symbols").clone()
    } else {
        // format_symbols now needs the `global` flag
        format_symbols(pilout, /*global=*/ false)
    };

    // ─── 4.  filter symbols like the JS does ─────────────────────────────────
    symbols_val.retain(|s| {
        let t = s["type"].as_str().unwrap_or("");
        !["witness", "fixed"].contains(&t) || (s["airId"] == res["airId"] && s["airgroupId"] == res["airgroupId"])
    });

    // ─── 5.  derive counters on `res` ────────────────────────────────────────
    res.insert("pilPower".into(), json!((pilout["numRows"].as_u64().unwrap() as f64).log2()));
    res.insert("nCommitments".into(), json!(symbols_val.iter().filter(|s| s["type"] == "witness").count()));
    res.insert("nConstants".into(), json!(symbols_val.iter().filter(|s| s["type"] == "fixed").count()));
    res.insert("nPublics".into(), json!(symbols_val.iter().filter(|s| s["type"] == "public").count()));
    res.insert("airGroupValues".into(), pilout.get("airGroupValues").cloned().unwrap_or(json!([])));

    // nStages: either explicit numChallenges or max stage among symbols
    if let Some(ch) = pilout.get("numChallenges") {
        res.insert("nStages".into(), json!(ch.as_array().unwrap().len()));
    } else {
        let max_stage = symbols_val.iter().map(|s| s["stage"].as_u64().unwrap_or(0)).max().unwrap_or(0);
        res.insert("nStages".into(), json!(max_stage));
    }

    // ─── 6.  hints ───────────────────────────────────────────────────────────
    let air_hints: Vec<Value> = pilout
        .get("hints")
        .and_then(Value::as_array)
        .map(|v| {
            v.iter().filter(|h| h["airId"] == res["airId"] && h["airGroupId"] == res["airgroupId"]).cloned().collect()
        })
        .unwrap_or_default();

    // Convert symbols_val → Vec<HashMap<..>> for format_hints
    let mut symbols_hash: Vec<HashMap<String, Value>> =
        symbols_val.iter().map(|v| v.as_object().unwrap().clone().into_iter().collect()).collect();

    // format_hints needs &mut Vec<HashMap<..>>, &mut Vec<Value>, save_symbols, global
    let hints =
        format_hints(pilout, &air_hints, &mut symbols_hash, &mut expressions_vec, save_symbols, /*global=*/ false);

    // symbols_hash → back to Vec<Value> so the returned JSON matches JS
    symbols_val = symbols_hash.into_iter().map(|hm| Value::Object(hm.into_iter().collect())).collect();

    // ─── 7.  customCommits bookkeeping (identical to JS) ─────────────────────
    let custom_commits = pilout.get("customCommits").cloned().unwrap_or(json!([]));
    res.insert("customCommits".into(), custom_commits.clone());

    // Fill mapSectionsN & customCommitsMap
    let mut cc_map: Vec<Value> = vec![];
    if let Some(arr) = custom_commits.as_array() {
        for cc in arr {
            let mut stages = vec![];
            for (j, w) in cc["stageWidths"].as_array().unwrap().iter().enumerate() {
                if w.as_u64().unwrap() > 0 {
                    res.entry("mapSectionsN".into())
                        .or_insert_with(|| json!({}))
                        .as_object_mut()
                        .unwrap()
                        .insert(format!("{}{}", cc["name"], j), json!(0));
                }
                stages.push(json!(null));
            }
            cc_map.push(json!(stages));
        }
    }
    res.insert("customCommitsMap".into(), json!(cc_map));

    // ─── 8.  final object exactly like the JS returns ────────────────────────
    json!({
        "expressions": Value::Array(expressions_vec),
        "hints":       hints,
        "constraints": constraints,
        "symbols":     Value::Array(symbols_val)
    })
}

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
