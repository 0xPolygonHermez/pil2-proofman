use crate::{
    grand_product_permutation_utils::{grand_product_permutation, init_challenges_permutation},
    grand_product_utils::{grand_product_connection, init_challenges_connection},
    grand_productp_lookup::{grand_product_plookup, init_challenges_plookup},
};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Generates library polynomials for PIL.
pub fn generate_libs_polynomials(
    fr_mul: fn(f64, f64) -> f64, // Function for field arithmetic
    k: f64,                      // Initial field element
    res: &mut HashMap<String, Value>,
    pil: &mut HashMap<String, Value>,
    symbols: &mut Vec<Value>,
    hints: &mut Vec<Value>,
) {
    // Initialize `nCm2` and `nCm3`
    pil.insert("nCm2".to_string(), json!(0));
    pil.insert("nCm3".to_string(), json!(0));

    let airgroup_id = res["airgroupId"].as_u64().unwrap();
    let air_id = res["airId"].as_u64().unwrap();

    let mut challenge_sets: Vec<Vec<Value>> = Vec::new();

    if pil.get("plookupIdentities").is_some_and(|v| !v.as_array().unwrap().is_empty()) {
        grand_product_plookup(pil, symbols, hints, airgroup_id, air_id);
        challenge_sets.push(init_challenges_plookup());
    }

    if pil.get("permutationIdentities").map_or(false, |v| !v.as_array().unwrap().is_empty()) {
        grand_product_permutation(pil, symbols, hints, airgroup_id, air_id);
        challenge_sets.push(init_challenges_permutation());
    }

    if pil.get("connectionIdentities").map_or(false, |v| !v.as_array().unwrap().is_empty()) {
        grand_product_connection(pil, symbols, hints, airgroup_id, air_id, fr_mul, k);
        challenge_sets.push(init_challenges_connection());
    }

    // Process challenge sets
    for challenges in challenge_sets {
        calculate_challenges(symbols, &challenges);
    }
}

/// Calculates challenge values and assigns proper IDs.
fn calculate_challenges(symbols: &mut Vec<Value>, challenges: &[Value]) {
    for challenge in challenges {
        if !symbols.iter().any(|c| {
            c["type"] == "challenge" && c["stage"] == challenge["stage"] && c["stageId"] == challenge["stageId"]
        }) {
            if let Some(challenge_obj) = challenge.as_object() {
                let mut challenge_clone = challenge_obj.clone();
                challenge_clone.insert("type".to_string(), json!("challenge"));
                symbols.push(Value::Object(challenge_clone));
            }
        }
    }

    let symbols_challenges: Vec<Value> = symbols.iter().filter(|s| s["type"] == "challenge").cloned().collect();

    for ch in symbols.iter_mut().filter(|s| s["type"] == "challenge") {
        let id = symbols_challenges
            .iter()
            .filter(|c| {
                c["stage"].as_u64().unwrap() < ch["stage"].as_u64().unwrap()
                    || (c["stage"] == ch["stage"] && c["stageId"].as_u64().unwrap() < ch["stageId"].as_u64().unwrap())
            })
            .count();
        ch.as_object_mut().unwrap().insert("id".to_string(), json!(id));
    }
}
