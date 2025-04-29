use crate::helpers::print_expressions;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// Mirrors the JS `map` helper, mutating `res`, `constraints`, etc.
pub fn map(
    res: &mut HashMap<String, Value>,
    symbols: &[Value],
    expressions: &[Value],
    constraints: &mut [Value],
    options: &Value,
) {
    /* ───────────────── 1. symbols → res.*Maps ───────────────── */
    map_symbols(res, symbols);
    set_stage_info_symbols(res, symbols);

    /* ───────────────── 2. decorate constraints ───────────────── */
    for cons in constraints.iter_mut() {
        if cons.get("filename") == Some(&json!(format!("{}.ImPol", res["name"].as_str().unwrap()))) {
            cons["imPol"] = json!(true);

            if !options["recursion"].as_bool().unwrap_or(false) {
                let e_idx = cons["e"].as_u64().unwrap() as usize;
                cons["line"] = json!(print_expressions(res, &expressions[e_idx], expressions, true));
            } else {
                cons["line"] = json!(""); // recursion path: blank line
            }
        }
        if let Some(line) = cons["line"].as_str() {
            cons["line"] = json!(format!("{line} == 0"));
        }
    }

    /* ───────────────── 3. intermediate polys info ────────────── */
    // ensure the holder exists once
    res.entry("imPolsInfo".into()).or_insert_with(|| json!({ "baseField": [], "extendedField": [] }));

    /*  gather OWNED copies, then immutable borrow ends before we mutate `res` */
    let im_pols: Vec<Value> = res
        .get("cmPolsMap")
        .and_then(Value::as_array)
        .map_or(Vec::new(), |arr| arr.iter().filter(|v| v.get("imPol") == Some(&json!(true))).cloned().collect());

    println!("----------------- INTERMEDIATE POLYNOMIALS -----------------");

    for (idx, im) in im_pols.iter().enumerate() {
        if options["recursion"].as_bool().unwrap_or(false) {
            continue; // recursion => skip printing / storing text
        }

        let exp_id = im["expId"].as_u64().unwrap() as usize;
        let text = print_expressions(res, &expressions[exp_id], expressions, false);

        if idx > 0 {
            println!("------------------------------------------------------------");
        }
        println!("Intermediate polynomial {idx} columns: {}", im["dim"]);
        println!("{text}");

        // push into the correct bucket
        let bucket = if im["dim"] == json!(1) { "baseField" } else { "extendedField" };
        if let Some(arr) = res
            .get_mut("imPolsInfo")
            .and_then(Value::as_object_mut)
            .and_then(|obj| obj.get_mut(bucket))
            .and_then(Value::as_array_mut)
        {
            arr.push(json!(text));
        }
    }

    /* ───────────────── 4. nCommitmentsStage1 counter ─────────── */
    let cm1_cnt = res.get("cmPolsMap").and_then(Value::as_array).map_or(0, |arr| {
        arr.iter().filter(|p| p.get("stage") == Some(&json!("cm1")) && p.get("imPol") != Some(&json!(true))).count()
    });

    res.insert("nCommitmentsStage1".into(), json!(cm1_cnt));
}

// ─────────────────────────────────────────────────────────────────────────────
// below: helpers that mimic the JS helpers 1-for-1
// ─────────────────────────────────────────────────────────────────────────────

fn map_symbols(res: &mut HashMap<String, Value>, symbols: &[Value]) {
    for sym in symbols {
        match sym["type"].as_str().unwrap_or("") {
            "fixed" | "witness" | "custom" => add_pol(res, sym),
            "challenge" => {
                res.entry("challengesMap".into()).or_insert_with(|| json!({})).as_object_mut().unwrap().insert(
                    sym["id"].to_string(),
                    json!({ "name": sym["name"], "stage": sym["stage"], "dim": sym["dim"], "stageId": sym["stageId"] }),
                );
            }
            "public" => {
                res.entry("publicsMap".into())
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .unwrap()
                    .insert(
                        sym["id"].to_string(),
                        json!({ "name": sym["name"], "stage": sym["stage"], "lengths": sym.get("lengths").cloned().unwrap_or(json!(null)) }),
                    );
            }
            "airgroupvalue" => {
                res.entry("airgroupValuesMap".into())
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .unwrap()
                    .insert(
                        sym["id"].to_string(),
                        json!({ "name": sym["name"], "stage": sym["stage"], "lengths": sym.get("lengths").cloned().unwrap_or(json!(null)) }),
                    );
            }
            "airvalue" => {
                res.entry("airValuesMap".into())
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .unwrap()
                    .insert(
                        sym["id"].to_string(),
                        json!({ "name": sym["name"], "stage": sym["stage"], "lengths": sym.get("lengths").cloned().unwrap_or(json!(null)) }),
                    );
            }
            "proofvalue" => {
                res.entry("proofValuesMap".into())
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .unwrap()
                    .insert(
                        sym["id"].to_string(),
                        json!({ "name": sym["name"], "stage": sym["stage"], "lengths": sym.get("lengths").cloned().unwrap_or(json!(null)) }),
                    );
            }
            _ => {}
        }
    }
}

/// Insert a fixed/witness/custom symbol into the appropriate *PolsMap,
/// replicating the side-effects on `mapSectionsN`.
fn add_pol(res: &mut HashMap<String, Value>, sym: &Value) {
    // Which top-level vector are we inserting into?
    let (ref_key, stage, dim, pos) = match sym["type"].as_str().unwrap() {
        "fixed" => ("constPolsMap", 0, sym["dim"].as_u64().unwrap(), sym["polId"].as_u64().unwrap() as usize),
        "witness" => (
            "cmPolsMap",
            sym["stage"].as_u64().unwrap(),
            sym["dim"].as_u64().unwrap(),
            sym["polId"].as_u64().unwrap() as usize,
        ),
        "custom" => (
            "customCommitsMap",
            sym["stage"].as_u64().unwrap(),
            sym["dim"].as_u64().unwrap(),
            sym["polId"].as_u64().unwrap() as usize,
        ),
        _ => unreachable!(),
    };

    // Resolve the array we are inserting into (for custom we go one level deeper).
    let target_arr: &mut Vec<Value> = if sym["type"] == json!("custom") {
        // res.customCommitsMap[commitId]
        let cid = sym["commitId"].as_u64().unwrap() as usize;
        res.entry("customCommitsMap".into()).or_insert_with(|| json!([]));
        let outer = res.get_mut("customCommitsMap").unwrap().as_array_mut().unwrap();
        if outer.len() <= cid {
            outer.resize(cid + 1, json!([]));
        }
        outer[cid].as_array_mut().unwrap()
    } else {
        // constPolsMap or cmPolsMap
        res.entry(ref_key.into()).or_insert_with(|| json!([]));
        res.get_mut(ref_key).unwrap().as_array_mut().unwrap()
    };

    // Ensure correct length and insert
    if target_arr.len() <= pos {
        target_arr.resize(pos + 1, Value::Null);
    }

    let mut entry = Map::from_iter(vec![
        ("stage".into(), sym["stage"].clone()),
        ("name".into(), sym["name"].clone()),
        ("dim".into(), sym["dim"].clone()),
        ("polsMapId".into(), json!(pos)),
    ]);

    // stageId for fixed is always its polId
    if sym["type"] == json!("fixed") {
        entry.insert("stageId".into(), json!(sym["polId"]));
    } else if !sym["stageId"].is_null() {
        entry.insert("stageId".into(), sym["stageId"].clone());
    }

    if let Some(l) = sym.get("lengths") {
        entry.insert("lengths".into(), l.clone());
    }
    if sym["imPol"].as_bool().unwrap_or(false) {
        entry.insert("imPol".into(), json!(true));
        entry.insert("expId".into(), sym["expId"].clone());
    }

    target_arr[pos] = Value::Object(entry);

    // Update mapSectionsN counters
    let sec_key = match sym["type"].as_str().unwrap() {
        "fixed" => "const".to_string(),
        "witness" => format!("cm{}", stage),
        "custom" => {
            let cid = sym["commitId"].as_u64().unwrap() as usize;
            let name = res["customCommits"][cid]["name"].as_str().unwrap();
            format!("{}{}", name, stage)
        }
        _ => unreachable!(),
    };
    let ms = res.entry("mapSectionsN".into()).or_insert_with(|| json!({})).as_object_mut().unwrap();
    *ms.entry(sec_key).or_insert(json!(0)) = json!(ms.get(sec_key.as_str()).and_then(Value::as_u64).unwrap_or(0) + dim);
}

/// Mirrors JS `setStageInfoSymbols`.
fn set_stage_info_symbols(res: &mut HashMap<String, Value>, symbols: &[Value]) {
    let q_stage = res["nStages"].as_u64().unwrap_or(0) + 1;

    for sym in symbols {
        let typ = sym["type"].as_str().unwrap_or("");
        if !["fixed", "witness", "custom"].contains(&typ) {
            continue;
        }

        if typ == "witness" || typ == "custom" {
            // find the vector where this pol lives
            let (pols_map, pol_id) = if typ == "witness" {
                (res.get_mut("cmPolsMap").unwrap().as_array_mut().unwrap(), sym["polId"].as_u64().unwrap() as usize)
            } else {
                let cid = sym["commitId"].as_u64().unwrap() as usize;
                (
                    res.get_mut("customCommitsMap").unwrap().as_array_mut().unwrap()[cid].as_array_mut().unwrap(),
                    sym["polId"].as_u64().unwrap() as usize,
                )
            };

            // previous pols in same stage
            let prev: Vec<&Value> = pols_map.iter().take(pol_id).filter(|p| p["stage"] == sym["stage"]).collect();

            let stage_pos: u64 = prev.iter().map(|p| p["dim"].as_u64().unwrap()).sum();

            let stage_id = if sym["stageId"].is_null() {
                if sym["stage"] == json!(q_stage) {
                    prev.len() as u64
                } else {
                    pols_map
                        .iter()
                        .filter(|p| p["stage"] == sym["stage"])
                        .position(|p| p["name"] == sym["name"])
                        .unwrap_or(0) as u64
                }
            } else {
                sym["stageId"].as_u64().unwrap()
            };

            if let Some(p_entry) = pols_map.get_mut(pol_id) {
                p_entry["stagePos"] = json!(stage_pos);
                p_entry["stageId"] = json!(stage_id);
            }
        }
    }
}
