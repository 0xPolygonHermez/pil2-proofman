use crate::utils::{format_expressions, format_constraints, format_symbols, format_hints};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Extracts and processes PIL information from `pilout`, mirroring JavaScript logic.
pub fn get_pilout_info(res: &mut HashMap<String, Value>, pilout: &HashMap<String, Value>) -> HashMap<String, Value> {
    res.insert("airId".to_string(), pilout["airId"].clone());
    res.insert("airgroupId".to_string(), pilout["airGroupId"].clone());

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
                || (s["airId"] == res["airId"] && s["airgroupId"] == res["airgroupId"])
        })
        .cloned()
        .collect::<Vec<Value>>());

    let air_group_values = pilout.get("airGroupValues").cloned().unwrap_or(json!([]));

    res.insert("pilPower".to_string(), json!(pilout["numRows"].as_u64().map(|n| (n as f64).log2()).unwrap_or(0.0)));

    let witness_symbols = symbols
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["type"] == "witness" && s["airId"] == res["airId"] && s["airgroupId"] == res["airgroupId"])
        .count();

    let fixed_symbols = symbols
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["type"] == "fixed" && s["airId"] == res["airId"] && s["airgroupId"] == res["airgroupId"])
        .count();

    let public_symbols = symbols.as_array().unwrap().iter().filter(|s| s["type"] == "public").count();

    res.insert("nCommitments".to_string(), json!(witness_symbols));
    res.insert("nConstants".to_string(), json!(fixed_symbols));
    res.insert("nPublics".to_string(), json!(public_symbols));
    res.insert("airGroupValues".to_string(), air_group_values.clone());

    let num_challenges = pilout.get("numChallenges").and_then(|v| v.as_array());

    res.insert(
        "nStages".to_string(),
        json!(num_challenges.map(|v| v.len()).unwrap_or_else(|| {
            let max_stage =
                symbols.as_array().unwrap().iter().map(|s| s["stage"].as_u64().unwrap_or(0)).max().unwrap_or(0);
            (max_stage as usize) + 1 // Ensure `numChallenges.length` matches JavaScript logic
        })),
    );

    let air_hints = pilout["hints"]
        .as_array()
        .map(|hints| {
            hints
                .iter()
                .filter(|h| h["airId"] == res["airId"] && h["airGroupId"] == res["airgroupId"])
                .cloned()
                .collect::<Vec<Value>>()
        })
        .unwrap_or_else(Vec::new);

    let mut symbols_vec = symbols.as_array().unwrap().clone();
    let mut expressions_vec = expressions.as_array().unwrap().clone();

    let hints = format_hints(pilout, &air_hints, &mut symbols_vec, &mut expressions_vec, save_symbols, false);

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
