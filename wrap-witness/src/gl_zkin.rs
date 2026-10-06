//! The zkin of the final circuit of a blake3 key, from its recursivef proof: a Goldilocks STARK, whose
//! proof's words are in the order of pil2-stark's `pointer2json`, and whose circuit inputs are named
//! as `define_stark_inputs` names those of a Goldilocks proof (`s0_last_mt_levels*`, not
//! `pointer2json`'s `s0_last_levels*`).

use serde_json::{json, Map, Value};

use crate::error::{WrapWitnessError, WrapWitnessResult};

/// The zkin of the proof `proof` (the words after its publics) of publics `publics`, of the STARK
/// `stark_info`, or why the words are not those of a proof of it: too few or too many, or a
/// starkinfo with what the final circuit of a blake3 key has not (air values, airgroup values,
/// custom commits).
pub fn gl_proof_zkin(stark_info: &Value, proof: &[u64], publics: &[u64]) -> WrapWitnessResult<Value> {
    let bad = |e: String| WrapWitnessError::Mismatch(format!("the recursivef proof's zkin: {e}"));
    let num = |v: &Value, what: &str| v.as_u64().ok_or_else(|| bad(format!("the starkinfo has no {what}")));
    for key in ["airgroupValuesMap", "airValuesMap", "customCommits"] {
        if stark_info.get(key).and_then(Value::as_array).is_some_and(|a| !a.is_empty()) {
            return Err(bad(format!("the starkinfo has {key}, which the final circuit has not")));
        }
    }
    let ss = &stark_info["starkStruct"];
    let arity = num(&ss["merkleTreeArity"], "merkleTreeArity")?;
    let llv = ss["lastLevelVerification"].as_u64().unwrap_or(0);
    let n_queries = num(&ss["nQueries"], "nQueries")? as usize;
    let steps: Vec<u64> = ss["steps"]
        .as_array()
        .ok_or_else(|| bad("the starkinfo has no steps".into()))?
        .iter()
        .map(|s| num(&s["nBits"], "steps' nBits"))
        .collect::<WrapWitnessResult<_>>()?;
    let n_stages = num(&stark_info["nStages"], "nStages")?;
    let n_evals = stark_info["evMap"].as_array().map_or(0, Vec::len);
    let n_constants = num(&stark_info["nConstants"], "nConstants")? as usize;

    // Sibling levels below the last `llv`, which the proof gives whole.
    let levels = |n_bits: u64| -> usize {
        if n_bits == 0 {
            return 0;
        }
        let l = n_bits.div_ceil(arity.ilog2() as u64);
        l.saturating_sub(llv) as usize
    };
    let per_level = (arity as usize - 1) * 4;
    let last_nodes = if llv > 0 { arity.pow(llv as u32) as usize } else { 0 };

    let mut words = proof.iter();
    let mut take = |n: usize| -> WrapWitnessResult<Value> {
        let taken: Vec<Value> = words.by_ref().take(n).map(|w| Value::String(w.to_string())).collect();
        if taken.len() < n {
            return Err(bad(format!("the proof has {} words, fewer than its starkinfo's", proof.len())));
        }
        Ok(Value::Array(taken))
    };
    let many = |count: usize, n: usize, take: &mut dyn FnMut(usize) -> WrapWitnessResult<Value>| {
        (0..count).map(|_| take(n)).collect::<WrapWitnessResult<Vec<_>>>().map(Value::Array)
    };

    // The proof has every section; the circuit declares no publics, no empty stage, no nonce
    // without grinding.
    let mut zkin = Map::new();
    if !publics.is_empty() {
        zkin.insert("publics".into(), json!(publics.iter().map(u64::to_string).collect::<Vec<_>>()));
    }
    for s in 1..=n_stages + 1 {
        zkin.insert(format!("root{s}"), take(4)?);
    }
    zkin.insert("evals".into(), many(n_evals, 3, &mut take)?);
    let s0_levels = levels(steps[0]);
    let siblings = |take: &mut dyn FnMut(usize) -> WrapWitnessResult<Value>, n_levels: usize| {
        (0..n_queries)
            .map(|_| (0..n_levels).map(|_| take(per_level)).collect::<WrapWitnessResult<Vec<_>>>().map(Value::Array))
            .collect::<WrapWitnessResult<Vec<_>>>()
            .map(Value::Array)
    };
    zkin.insert("s0_valsC".into(), many(n_queries, n_constants, &mut take)?);
    zkin.insert("s0_siblingsC".into(), siblings(&mut take, s0_levels)?);
    if llv > 0 {
        zkin.insert("s0_last_mt_levelsC".into(), many(last_nodes, 4, &mut take)?);
    }
    for stage in 1..=n_stages + 1 {
        let width = stark_info["mapSectionsN"][format!("cm{stage}")].as_u64().unwrap_or(0) as usize;
        let section = [
            (format!("s0_vals{stage}"), many(n_queries, width, &mut take)?),
            (format!("s0_siblings{stage}"), siblings(&mut take, s0_levels)?),
        ];
        let last = if llv > 0 { Some(many(last_nodes, 4, &mut take)?) } else { None };
        if width > 0 || stage == n_stages + 1 {
            zkin.extend(section);
            if let Some(last) = last {
                zkin.insert(format!("s0_last_mt_levels{stage}"), last);
            }
        }
    }
    for step in 1..steps.len() {
        zkin.insert(format!("s{step}_root"), take(4)?);
    }
    for step in 1..steps.len() {
        let width = (1usize << (steps[step - 1] - steps[step])) * 3;
        zkin.insert(format!("s{step}_vals"), many(n_queries, width, &mut take)?);
        zkin.insert(format!("s{step}_siblings"), siblings(&mut take, levels(steps[step]))?);
        if llv > 0 {
            zkin.insert(format!("s{step}_last_mt_levels"), many(last_nodes, 4, &mut take)?);
        }
    }
    zkin.insert("finalPol".into(), many(1 << steps[steps.len() - 1], 3, &mut take)?);
    let nonce = take(1)?;
    if ss["powBits"].as_u64().unwrap_or(0) > 0 {
        zkin.insert("nonce".into(), nonce[0].clone());
    }
    let rest = words.count();
    if rest != 0 {
        return Err(bad(format!("the proof has {rest} words more than its starkinfo's")));
    }
    Ok(Value::Object(zkin))
}
