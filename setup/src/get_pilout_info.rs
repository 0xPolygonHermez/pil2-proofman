use crate::utils::{format_expressions, format_constraints, format_symbols, format_hints, buf2bint};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Extracts and processes PIL information from `pilout`, mirroring JavaScript logic.
pub fn get_pilout_info(res: &mut HashMap<String, Value>, pilout: &Value) -> HashMap<String, Value> {
    res.insert("airId".to_string(), pilout["airId"].clone());
    res.insert("airgroupId".to_string(), pilout["airGroupId"].clone());

    let constraints = format_constraints(pilout);

    let save_symbols = pilout.get("symbols").is_none();
    let (mut expressions, mut symbols) = if save_symbols {
        let e = format_expressions(pilout, true);
        (e["expressions"].clone(), e["symbols"].clone())
    } else {
        let e = format_expressions(pilout, false);
        (e["expressions"].clone(), format_symbols(pilout))
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

    let witness_symbols: Vec<&Value> = symbols
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["type"] == "witness" && s["airId"] == res["airId"] && s["airgroupId"] == res["airgroupId"])
        .collect();

    let fixed_symbols: Vec<&Value> = symbols
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["type"] == "fixed" && s["airId"] == res["airId"] && s["airgroupId"] == res["airgroupId"])
        .collect();

    let public_symbols: Vec<&Value> = symbols.as_array().unwrap().iter().filter(|s| s["type"] == "public").collect();

    res.insert("nCommitments".to_string(), json!(witness_symbols.len()));
    res.insert("nConstants".to_string(), json!(fixed_symbols.len()));
    res.insert("nPublics".to_string(), json!(public_symbols.len()));
    res.insert("airGroupValues".to_string(), air_group_values.clone());

    let num_challenges = pilout.get("numChallenges").and_then(|v| v.as_array());

    res.insert(
        "nStages".to_string(),
        json!(num_challenges.map(|v| v.len()).unwrap_or_else(|| {
            let max_stage =
                symbols.as_array().unwrap().iter().map(|s| s["stage"].as_u64().unwrap_or(0)).max().unwrap_or(0);
            max_stage as usize
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

    let hints = format_hints(pilout, &json!(air_hints), &symbols, &expressions, save_symbols);

    res.insert("customCommits".to_string(), pilout["customCommits"].clone().unwrap_or(json!([])));
    let mut custom_commits_map = vec![];

    if let Some(commits) = res["customCommits"].as_array() {
        for (i, commit) in commits.iter().enumerate() {
            let mut commit_map = vec![];
            if let Some(stage_widths) = commit["stageWidths"].as_array() {
                for (j, width) in stage_widths.iter().enumerate() {
                    if width.as_u64().unwrap_or(0) > 0 {
                        res.entry("mapSectionsN".to_string())
                            .or_insert(json!(HashMap::new()))
                            .as_object_mut()
                            .unwrap()
                            .insert(format!("{}{}", commit["name"], j), json!(0));
                    }
                    commit_map.push(json!({}));
                }
            }
            custom_commits_map.push(json!(commit_map));
        }
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
}
