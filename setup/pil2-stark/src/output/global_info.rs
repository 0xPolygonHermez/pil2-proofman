//! Build and write the global proving-key files:
//!   - `pilout.globalInfo.json`
//!   - `pilout.globalConstraints.json`
//!   - `pilout.globalConstraints.bin`
//!
//! Also provides `write_bin_files_native` for the test suite (reads
//! expressionsinfo/verifierinfo JSON from disk and writes binary files).

use std::fs;
use std::path::Path;

use anyhow::Result;
use pil2_pilout::pilout::{self as pb, SymbolType};
use proofman_starks_lib_c::GOLDILOCKS_MERKLE_TREE_ARITY;
use serde_json::json;

use crate::types::stark_struct::StarkStructsConfig;
use crate::output::stark_info::{code_entries_to_json, hint_value_to_json};

/// Build the `globalInfo` JSON value in memory (does not write to disk).
pub(crate) fn build_global_info_json(
    pilout: &pb::PilOut,
    pilout_name: &str,
    settings_map: &StarkStructsConfig,
    hash: &str,
) -> serde_json::Value {
    let mut airs = Vec::new();
    let mut air_groups = Vec::new();
    let mut agg_types = Vec::new();

    for airgroup in &pilout.air_groups {
        let ag_name = airgroup.name.clone().unwrap_or_else(|| "unnamed".to_string());
        air_groups.push(ag_name.clone());

        let agv: Vec<serde_json::Value> =
            airgroup.air_group_values.iter().map(|v| json!({"aggType": v.agg_type, "stage": v.stage})).collect();
        agg_types.push(agv);

        let mut air_list = Vec::new();
        for air in &airgroup.airs {
            let a_name = air.name.clone().unwrap_or_else(|| "unnamed".to_string());
            let has_compressor = settings_map.has_compressor(&ag_name, &a_name);
            let mut entry = json!({
                "name": a_name,
                "num_rows": air.num_rows.unwrap_or(0),
            });
            if has_compressor {
                entry.as_object_mut().unwrap().insert("hasCompressor".to_string(), json!(true));
            }
            air_list.push(entry);
        }
        airs.push(serde_json::Value::Array(air_list));
    }

    let num_challenges: Vec<u32> =
        if pilout.num_challenges.is_empty() { vec![0] } else { pilout.num_challenges.clone() };

    let proof_values_map = build_global_proof_values_map(&pilout.symbols);
    let publics_map = build_global_publics_map(&pilout.symbols);

    let transcript_arity: u64 =
        if hash == "blake3" { 2 } else { GOLDILOCKS_MERKLE_TREE_ARITY };

    json!({
        "name": pilout_name,
        "airs": airs,
        "air_groups": air_groups,
        "aggTypes": agg_types,
        "curve": "None",
        "latticeSize": 368,
        "transcriptArity": transcript_arity,
        "nPublics": pilout.num_public_values,
        "numChallenges": num_challenges,
        "numProofValues": pilout.num_proof_values,
        "proofValuesMap": proof_values_map,
        "publicsMap": publics_map,
        "hash": hash,
    })
}

/// Write only `pilout.globalInfo.json` into `<build_dir>/provingKey/`.
pub(crate) fn write_global_info_json(
    pilout: &pb::PilOut,
    pilout_name: &str,
    build_dir: &str,
    settings_map: &StarkStructsConfig,
    hash: &str,
) -> Result<()> {
    let proving_key_dir = Path::new(build_dir).join("provingKey");
    fs::create_dir_all(&proving_key_dir)?;
    let global_info = build_global_info_json(pilout, pilout_name, settings_map, hash);
    let global_info_str = crate::output::json::to_json_string(&global_info)?;
    fs::write(proving_key_dir.join("pilout.globalInfo.json"), &global_info_str)?;
    Ok(())
}

/// Write `pilout.globalConstraints.json` and `pilout.globalConstraints.bin`
/// into `<build_dir>/provingKey/`. Does not write `globalInfo.json`.
pub(crate) fn write_global_constraints(
    pilout: &pb::PilOut,
    pilout_name: &str,
    build_dir: &str,
    settings_map: &StarkStructsConfig,
) -> Result<()> {
    let proving_key_dir = Path::new(build_dir).join("provingKey");
    fs::create_dir_all(&proving_key_dir)?;

    // Only proofValuesMap/aggTypes are read below; hash is irrelevant here.
    let global_info =
        build_global_info_json(pilout, pilout_name, settings_map, proofman_common::hash_family::DEFAULT_HASH_ID);

    let global_constraints = build_global_constraints_json(pilout)?;
    let gc_str = crate::output::json::to_json_string(&global_constraints)?;
    fs::write(proving_key_dir.join("pilout.globalConstraints.json"), &gc_str)?;

    {
        use crate::output::global_constraints::write_global_constraints_bin_file;
        use crate::io::parser_args::{GlobalInfo as ParserGlobalInfo, ProofValueEntry};
        use crate::types::stark_info::GlobalConstraintsInfo;

        let proof_values_map_json =
            global_info.get("proofValuesMap").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let pvm: Vec<ProofValueEntry> = proof_values_map_json
            .iter()
            .map(|entry| ProofValueEntry { stage: entry.get("stage").and_then(|s| s.as_u64()).unwrap_or(1) })
            .collect();

        let agg_types_json = global_info.get("aggTypes").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let agg_types: Vec<Vec<u64>> = agg_types_json
            .iter()
            .map(|ag| {
                ag.as_array()
                    .map(|arr| arr.iter().map(|v| v.get("aggType").and_then(|a| a.as_u64()).unwrap_or(0)).collect())
                    .unwrap_or_default()
            })
            .collect();

        let gi = ParserGlobalInfo { proof_values_map: pvm, agg_types };
        let gci = GlobalConstraintsInfo::from_json(&global_constraints)?;
        let bin_path = proving_key_dir.join("pilout.globalConstraints.bin");
        write_global_constraints_bin_file(&gi, &gci, bin_path.to_str().unwrap_or(""))?;
    }

    Ok(())
}

/// Write all three output files (convenience wrapper, used by non-recursive path and tests).
pub(crate) fn write_global_info(
    pilout: &pb::PilOut,
    pilout_name: &str,
    build_dir: &str,
    settings_map: &StarkStructsConfig,
    hash: &str,
) -> Result<()> {
    write_global_constraints(pilout, pilout_name, build_dir, settings_map)?;
    write_global_info_json(pilout, pilout_name, build_dir, settings_map, hash)?;
    tracing::info!("Global info and constraints written");
    Ok(())
}

/// Build the `proofValuesMap` array from pilout symbols (sorted by id).
fn build_global_proof_values_map(symbols: &[pb::Symbol]) -> Vec<serde_json::Value> {
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
fn build_global_publics_map(symbols: &[pb::Symbol]) -> Vec<serde_json::Value> {
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

/// Build the globalConstraints JSON from pilout data.
fn build_global_constraints_json(pilout: &pb::PilOut) -> Result<serde_json::Value> {
    use crate::pil::codegen::{build_code, pil_code_gen, CodeGenCtx};
    use crate::pil::gen_code::CodeGenParams;
    use crate::expr::helpers::add_info_expressions;
    use crate::types::pilout_info::{
        format_global_constraints, format_global_expressions, format_global_hints, format_global_symbols, SymbolInfo,
        FIELD_EXTENSION,
    };
    use crate::expr::print::PrintCtx;

    // If no global constraints exist, return empty
    if pilout.constraints.is_empty() && pilout.hints.iter().all(|h| h.air_group_id.is_some() || h.air_id.is_some()) {
        return Ok(json!({"constraints": [], "hints": []}));
    }

    let mut expressions = format_global_expressions(&pilout.expressions, &pilout.num_challenges, &pilout.air_groups);

    let constraints = format_global_constraints(&pilout.constraints);
    let symbols = format_global_symbols(&pilout.symbols, &pilout.num_challenges);

    for constraint in &constraints {
        add_info_expressions(&mut expressions, constraint.e);
    }

    let publics_map: Vec<SymbolInfo> = symbols.iter().filter(|s| s.sym_type == "public").cloned().collect();
    let challenges_map: Vec<SymbolInfo> = symbols.iter().filter(|s| s.sym_type == "challenge").cloned().collect();
    let airgroup_values_map: Vec<SymbolInfo> =
        symbols.iter().filter(|s| s.sym_type == "airgroupvalue").cloned().collect();
    let proof_values_map: Vec<SymbolInfo> = symbols.iter().filter(|s| s.sym_type == "proofvalue").cloned().collect();

    // Build sorted maps indexed by ID for PrintCtx
    let max_public_id = publics_map.iter().filter_map(|s| s.id).max().unwrap_or(0);
    let mut publics_by_id = vec![
        SymbolInfo {
            name: String::new(),
            sym_type: "public".to_string(),
            stage: Some(1),
            dim: 1,
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        max_public_id + 1
    ];
    for s in &publics_map {
        if let Some(id) = s.id {
            if id < publics_by_id.len() {
                publics_by_id[id] = s.clone();
            }
        }
    }

    let max_challenge_id = challenges_map.iter().filter_map(|s| s.id).max().unwrap_or(0);
    let mut challenges_by_id = vec![
        SymbolInfo {
            name: String::new(),
            sym_type: "challenge".to_string(),
            stage: Some(1),
            dim: FIELD_EXTENSION,
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        max_challenge_id + 1
    ];
    for s in &challenges_map {
        if let Some(id) = s.id {
            if id < challenges_by_id.len() {
                challenges_by_id[id] = s.clone();
            }
        }
    }

    let max_agv_id = airgroup_values_map.iter().filter_map(|s| s.id).max().unwrap_or(0);
    let mut agv_by_id = vec![
        SymbolInfo {
            name: String::new(),
            sym_type: "airgroupvalue".to_string(),
            stage: None,
            dim: FIELD_EXTENSION,
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        max_agv_id + 1
    ];
    for s in &airgroup_values_map {
        if let Some(id) = s.id {
            if id < agv_by_id.len() {
                agv_by_id[id] = s.clone();
            }
        }
    }

    let max_pv_id = proof_values_map.iter().filter_map(|s| s.id).max().unwrap_or(0);
    let mut pv_by_id = vec![
        SymbolInfo {
            name: String::new(),
            sym_type: "proofvalue".to_string(),
            stage: Some(1),
            dim: 1,
            id: None,
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        max_pv_id + 1
    ];
    for s in &proof_values_map {
        if let Some(id) = s.id {
            if id < pv_by_id.len() {
                pv_by_id[id] = s.clone();
            }
        }
    }

    let empty_sym_vec: Vec<SymbolInfo> = Vec::new();
    let empty_custom_commits: Vec<Vec<SymbolInfo>> = Vec::new();
    let print_ctx = PrintCtx {
        cm_pols_map: &empty_sym_vec,
        const_pols_map: &empty_sym_vec,
        custom_commits_map: &empty_custom_commits,
        publics_map: &publics_by_id,
        challenges_map: &challenges_by_id,
        air_values_map: &empty_sym_vec,
        airgroup_values_map: &agv_by_id,
        proof_values_map: &pv_by_id,
    };

    let n_stages = if !pilout.num_challenges.is_empty() { pilout.num_challenges.len() } else { 1 };

    let mut ctx = CodeGenCtx::new(0, 0, n_stages, "n", false, Vec::new(), Vec::new());

    let mut constraints_json = Vec::new();

    for constraint in &constraints {
        pil_code_gen(&mut ctx, &symbols, &expressions, constraint.e, 0);
        let block = build_code(&mut ctx);

        ctx.tmp_used = block.tmp_used;

        let line = constraint.line.clone().unwrap_or_default();

        let mut obj = serde_json::Map::new();
        obj.insert("tmpUsed".to_string(), json!(block.tmp_used));
        obj.insert("code".to_string(), code_entries_to_json(&block.code));
        obj.insert("boundary".to_string(), json!(constraint.boundary));
        obj.insert("line".to_string(), json!(line));
        constraints_json.push(serde_json::Value::Object(obj));
    }

    let hints = format_global_hints(pilout, &mut expressions);

    let global_params = CodeGenParams {
        air_id: 0,
        airgroup_id: 0,
        n_stages,
        c_exp_id: 0,
        fri_exp_id: 0,
        q_deg: 0,
        q_dim: FIELD_EXTENSION,
        opening_points: Vec::new(),
        cm_pols_map: Vec::new(),
        custom_commits_count: 0,
    };

    let processed_hints = process_global_hints(&global_params, &mut expressions, &hints, Some(&print_ctx));

    let hints_json: Vec<serde_json::Value> = processed_hints
        .iter()
        .map(|h| {
            json!({
                "name": h.name,
                "fields": h.fields.iter().map(|f| {
                    json!({
                        "name": f.name,
                        "values": f.values.iter().map(|v| {
                            hint_value_to_json(v)
                        }).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<serde_json::Value>>(),
            })
        })
        .collect();

    Ok(json!({
        "constraints": constraints_json,
        "hints": hints_json,
    }))
}

/// Process global hints into flat hint field values.
fn process_global_hints(
    params: &crate::pil::gen_code::CodeGenParams,
    expressions: &mut Vec<crate::expr::expression::Expression>,
    hints: &[crate::types::pilout_info::HintInfo],
    print_ctx: Option<&crate::expr::print::PrintCtx>,
) -> Vec<crate::pil::gen_code::ProcessedHint> {
    use crate::pil::gen_code::{ProcessedHint, ProcessedHintFieldEntry};

    let mut result = Vec::new();

    for hint in hints {
        let mut processed_fields = Vec::new();

        for field in &hint.fields {
            let flat_values = process_global_hint_values(&field.values, params, expressions, &[], print_ctx);

            let mut entry = ProcessedHintFieldEntry { name: field.name.clone(), values: flat_values };

            if field.lengths.is_none() {
                if let Some(first) = entry.values.first_mut() {
                    first.pos = Vec::new();
                }
            }

            processed_fields.push(entry);
        }

        result.push(ProcessedHint { name: hint.name.clone(), fields: processed_fields });
    }

    result
}

/// Recursively flatten global hint field values.
fn process_global_hint_values(
    values: &[crate::types::pilout_info::HintFieldValue],
    params: &crate::pil::gen_code::CodeGenParams,
    expressions: &mut Vec<crate::expr::expression::Expression>,
    pos: &[usize],
    print_ctx: Option<&crate::expr::print::PrintCtx>,
) -> Vec<crate::pil::gen_code::ProcessedHintField> {
    use crate::types::pilout_info::HintFieldValue;

    let mut result = Vec::new();

    for (j, field) in values.iter().enumerate() {
        let mut current_pos: Vec<usize> = pos.to_vec();
        current_pos.push(j);

        match field {
            HintFieldValue::Array(arr) => {
                let inner = process_global_hint_values(arr, params, expressions, &current_pos, print_ctx);
                result.extend(inner);
            }
            HintFieldValue::Single(expr) => {
                let processed = process_global_single_hint_field(expr, params, expressions, &current_pos, print_ctx);
                result.push(processed);
            }
        }
    }

    result
}

/// Process a single global hint field value.
fn process_global_single_hint_field(
    expr: &crate::expr::expression::Expression,
    _params: &crate::pil::gen_code::CodeGenParams,
    #[allow(clippy::ptr_arg)] expressions: &mut Vec<crate::expr::expression::Expression>,
    pos: &[usize],
    print_ctx: Option<&crate::expr::print::PrintCtx>,
) -> crate::pil::gen_code::ProcessedHintField {
    use crate::pil::gen_code::ProcessedHintField;

    match expr.op.as_str() {
        "exp" => {
            let ref_id = expr.id.unwrap_or(0);
            let dim = expressions.get(ref_id).map_or(expr.dim.max(1), |e| e.dim);

            if let Some(ctx) = print_ctx {
                if ref_id < expressions.len() {
                    crate::expr::print::print_expression(ctx, expressions, ref_id, false);
                }
            }

            ProcessedHintField {
                op: "tmp".to_string(),
                id: Some(ref_id),
                dim: Some(dim),
                pos: pos.to_vec(),
                stage: None,
                stage_id: None,
                value: None,
                row_offset: None,
                row_offset_index: None,
                commit_id: None,
                airgroup_id: None,
            }
        }
        "challenge" | "public" | "airgroupvalue" | "airvalue" | "number" | "string" | "proofvalue" => {
            ProcessedHintField {
                op: expr.op.clone(),
                id: expr.id,
                dim: Some(expr.dim),
                pos: pos.to_vec(),
                stage: Some(expr.stage),
                stage_id: expr.stage_id,
                value: expr.value.clone(),
                row_offset: None,
                row_offset_index: None,
                commit_id: None,
                airgroup_id: expr.airgroup_id,
            }
        }
        _ => ProcessedHintField {
            op: expr.op.clone(),
            id: expr.id,
            dim: Some(expr.dim),
            pos: pos.to_vec(),
            stage: Some(expr.stage),
            stage_id: expr.stage_id,
            value: expr.value.clone(),
            row_offset: None,
            row_offset_index: None,
            commit_id: None,
            airgroup_id: expr.airgroup_id,
        },
    }
}

/// Read expressionsinfo and verifierinfo JSON files from disk and write binary outputs.
///
/// Used by the golden-reference test suite. In production the binary files are
/// written directly from in-memory structs via `write_bin_files_from_pil_code`.
#[cfg(test)]
pub(crate) fn write_bin_files_native(
    starkinfo_path: &Path,
    expressionsinfo_path: &Path,
    verifierinfo_path: &Path,
    bin_output: &Path,
    verifier_bin_output: &Path,
) -> Result<()> {
    use anyhow::Context;
    use crate::types::stark_info::{ExpressionsInfo, StarkInfo, VerifierInfo};

    let si_data =
        fs::read_to_string(starkinfo_path).with_context(|| format!("Cannot read starkinfo: {:?}", starkinfo_path))?;
    let si_json: serde_json::Value = serde_json::from_str(&si_data)?;
    let stark_info = StarkInfo::from_json(&si_json)?;

    let ei_data = fs::read_to_string(expressionsinfo_path)
        .with_context(|| format!("Cannot read expressionsinfo: {:?}", expressionsinfo_path))?;
    let ei_json: serde_json::Value = serde_json::from_str(&ei_data)?;
    let ei = ExpressionsInfo::from_json(&ei_json)?;

    let vi_data = fs::read_to_string(verifierinfo_path)
        .with_context(|| format!("Cannot read verifierinfo: {:?}", verifierinfo_path))?;
    let vi_json: serde_json::Value = serde_json::from_str(&vi_data)?;
    let vi = VerifierInfo::from_json(&vi_json)?;

    crate::io::bin_file::write_expressions_bin_file(bin_output.to_str().unwrap_or(""), &stark_info, &ei)?;
    crate::io::bin_file::write_verifier_expressions_bin_file(
        verifier_bin_output.to_str().unwrap_or(""),
        &stark_info,
        &vi,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn test_global_info_has_compressor() {
        let pilout_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../pil/zisk.pilout");
        if !std::path::Path::new(pilout_path).exists() {
            eprintln!("Skipping test_global_info_has_compressor: pilout not found");
            return;
        }

        let pilout_data = std::fs::read(pilout_path).unwrap();
        let pilout = pb::PilOut::decode(pilout_data.as_slice()).unwrap();
        let pilout_name = pilout.name.clone().unwrap_or_else(|| "pilout".to_string());

        let settings_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../state-machines/starkstructs.json");
        let settings_map: StarkStructsConfig = if std::path::Path::new(settings_path).exists() {
            let data = std::fs::read_to_string(settings_path).unwrap();
            StarkStructsConfig::from_json_str(&data).unwrap()
        } else {
            StarkStructsConfig::default()
        };

        let build_dir = "/tmp/r39_test_global_info";
        let _ = std::fs::remove_dir_all(build_dir);
        write_global_info(&pilout, &pilout_name, build_dir, &settings_map, "Poseidon2").unwrap();

        let gi_str = std::fs::read_to_string(format!("{}/provingKey/pilout.globalInfo.json", build_dir)).unwrap();
        let gi: serde_json::Value = serde_json::from_str(&gi_str).unwrap();

        let airs = gi.get("airs").unwrap().as_array().unwrap();
        assert!(!airs.is_empty());
        let first_group = airs[0].as_array().unwrap();
        let mut found_has_compressor = false;
        for air in first_group {
            let name = air.get("name").unwrap().as_str().unwrap();
            if name == "Keccakf" || name == "Sha256f" || name == "ArithEq" || name == "ArithEq384" {
                assert!(air.get("hasCompressor").is_some());
                assert!(air.get("hasCompressor").unwrap().as_bool().unwrap());
                found_has_compressor = true;
            }
            if name == "Main" || name == "Mem" {
                assert!(air.get("hasCompressor").is_none());
            }
        }
        assert!(found_has_compressor);

        let gc_str =
            std::fs::read_to_string(format!("{}/provingKey/pilout.globalConstraints.json", build_dir)).unwrap();
        let gc: serde_json::Value = serde_json::from_str(&gc_str).unwrap();
        let constraints = gc.get("constraints").unwrap().as_array().unwrap();
        assert!(!constraints.is_empty());
        let c0 = &constraints[0];
        assert!(c0.get("tmpUsed").is_some());
        assert!(c0.get("code").is_some());
        assert_eq!(c0.get("boundary").unwrap().as_str().unwrap(), "finalProof");
        assert!(!gc.get("hints").unwrap().as_array().unwrap().is_empty());
        assert!(std::path::Path::new(&format!("{}/provingKey/pilout.globalConstraints.bin", build_dir)).exists());

        let _ = std::fs::remove_dir_all(build_dir);
    }

    #[test]
    fn test_bin_file_byte_identical_to_golden() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let dir = base.join("golden_reference/zisk/Zisk/airs/Dma/air");
        let si = dir.join("Dma.starkinfo.json");
        let golden_bin = dir.join("Dma.bin");
        if !si.exists() || !golden_bin.exists() {
            eprintln!("Skipping: golden Dma files not found");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pil2_bin_regression_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let out_bin = tmp.join("Dma.bin");
        let out_vbin = tmp.join("Dma.verifier.bin");
        write_bin_files_native(
            &si,
            &dir.join("Dma.expressionsinfo.json"),
            &dir.join("Dma.verifierinfo.json"),
            &out_bin,
            &out_vbin,
        )
        .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden.len(), actual.len(), "Dma.bin size mismatch");
        assert_eq!(golden, actual, "Dma.bin content mismatch");
        assert_eq!(
            std::fs::read(dir.join("Dma.verifier.bin")).unwrap(),
            std::fs::read(&out_vbin).unwrap(),
            "Dma.verifier.bin mismatch"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_binary_bin_byte_identical_to_golden() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let dir = base.join("golden_reference/zisk/Zisk/airs/Binary/air");
        let si = dir.join("Binary.starkinfo.json");
        let golden_bin = dir.join("Binary.bin");
        if !si.exists() || !golden_bin.exists() {
            eprintln!("Skipping: golden Binary files not found");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pil2_binary_bin_regression_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let out_bin = tmp.join("Binary.bin");
        let out_vbin = tmp.join("Binary.verifier.bin");
        write_bin_files_native(
            &si,
            &dir.join("Binary.expressionsinfo.json"),
            &dir.join("Binary.verifierinfo.json"),
            &out_bin,
            &out_vbin,
        )
        .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden.len(), actual.len(), "Binary.bin size mismatch");
        if golden != actual {
            let pos = golden.iter().zip(actual.iter()).position(|(a, b)| a != b).unwrap_or(0);
            panic!("Binary.bin mismatch at byte {} (golden={:#x} actual={:#x})", pos, golden[pos], actual[pos]);
        }
        assert_eq!(std::fs::read(dir.join("Binary.verifier.bin")).unwrap(), std::fs::read(&out_vbin).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_arith_bin_byte_identical_to_golden() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let dir = base.join("golden_reference/zisk/Zisk/airs/Arith/air");
        let si = dir.join("Arith.starkinfo.json");
        let golden_bin = dir.join("Arith.bin");
        if !si.exists() || !golden_bin.exists() {
            eprintln!("Skipping: golden Arith files not found");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pil2_arith_bin_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let out_bin = tmp.join("Arith.bin");
        let out_vbin = tmp.join("Arith.verifier.bin");
        write_bin_files_native(
            &si,
            &dir.join("Arith.expressionsinfo.json"),
            &dir.join("Arith.verifierinfo.json"),
            &out_bin,
            &out_vbin,
        )
        .expect("write_bin_files_native failed");
        let golden = std::fs::read(&golden_bin).unwrap();
        let actual = std::fs::read(&out_bin).unwrap();
        assert_eq!(golden, actual, "Arith.bin mismatch");
        assert_eq!(std::fs::read(dir.join("Arith.verifier.bin")).unwrap(), std::fs::read(&out_vbin).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
