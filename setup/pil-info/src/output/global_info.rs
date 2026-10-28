//! The part of `pilout.globalInfo.json` that does not depend on the backend: the global proof
//! values and publics maps.

use pil2_pilout::pilout::{self as pb, SymbolType};
use serde_json::json;

/// Build the `proofValuesMap` array from pilout symbols (sorted by id).
pub fn build_global_proof_values_map(symbols: &[pb::Symbol]) -> Vec<serde_json::Value> {
    let mut entries: Vec<(u32, serde_json::Value)> = Vec::new();

    for s in symbols {
        if s.r#type != SymbolType::ProofValue as i32 {
            continue;
        }
        let stage = s.stage.unwrap_or(1);
        if s.dim == 0 {
            entries.push((s.id, json!({"name": s.name, "stage": stage})));
        } else {
            let total: u32 = s.lengths.iter().product::<u32>().max(1);
            for offset in 0..total {
                entries.push((s.id + offset, json!({"name": s.name, "stage": stage})));
            }
        }
    }

    entries.sort_by_key(|(id, _)| *id);
    entries.into_iter().map(|(_, v)| v).collect()
}

/// Build the `publicsMap` array from pilout symbols (sorted by id).
pub fn build_global_publics_map(symbols: &[pb::Symbol]) -> Vec<serde_json::Value> {
    let mut entries: Vec<(u32, serde_json::Value)> = Vec::new();

    for s in symbols {
        if s.r#type != SymbolType::PublicValue as i32 {
            continue;
        }
        if s.dim == 0 || s.lengths.is_empty() {
            entries.push((s.id, json!({"name": s.name, "stage": 1})));
        } else {
            expand_public_array_entries(&mut entries, s, &[], 0);
        }
    }

    entries.sort_by_key(|(id, _)| *id);
    entries.into_iter().map(|(_, v)| v).collect()
}

/// Recursively expand a multi-dimensional public array symbol into individual entries.
fn expand_public_array_entries(
    entries: &mut Vec<(u32, serde_json::Value)>,
    sym: &pb::Symbol,
    indexes: &[u32],
    shift: u32,
) -> u32 {
    if indexes.len() == sym.lengths.len() {
        let idx_vec: Vec<serde_json::Value> = indexes.iter().map(|&i| serde_json::Value::from(i)).collect();
        entries.push((sym.id + shift, json!({"name": sym.name, "stage": 1, "lengths": idx_vec})));
        return shift + 1;
    }

    let len = sym.lengths[indexes.len()];
    let mut current_shift = shift;
    for i in 0..len {
        let mut new_indexes = indexes.to_vec();
        new_indexes.push(i);
        current_shift = expand_public_array_entries(entries, sym, &new_indexes, current_shift);
    }
    current_shift
}
