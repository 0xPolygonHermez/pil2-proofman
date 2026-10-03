//! Build the starkinfo JSON output structure (expressionsinfo and verifierinfo are built by
//! `pil_info::output::expressions_info`).
//!
//! per-AIR child-process pipeline in `setup_cmd.rs` and from the recursive/final
//! setup modules.

use serde_json::json;

use crate::pil::gen_code::PilCodeResult;
use crate::types::pilout_info::SetupResult;
use crate::types::FIELD_EXTENSION;
use crate::types::security;
use crate::types::stark_struct::StarkStruct;
use crate::types::output::{
    BoundaryOutput, ChallengeMapEntryOutput, EvMapEntry, NameStageEntry, PolMapEntry, PublicMapEntry, SecurityInfo,
    StarkInfoOutput, StarkStructOutput, StepOutput,
};

/// Build the StarkInfoOutput for JSON serialization from internal types.
#[allow(clippy::too_many_arguments)]
pub fn build_starkinfo_output(
    setup: &SetupResult,
    stark_struct: &StarkStruct,
    pil_code: &PilCodeResult,
    opening_points: &[i64],
    fri: &security::pcs::Fri,
    airgroup_id: usize,
    air_id: usize,
    air_name: &str,
    c_exp_id: usize,
    fri_exp_id: usize,
    q_deg: i64,
) -> StarkInfoOutput {
    let steps: Vec<StepOutput> = stark_struct.steps.iter().map(|s| StepOutput { n_bits: s.n_bits }).collect();

    let fri_security = fri.security_params();
    let stark_struct_out = StarkStructOutput {
        n_bits: stark_struct.n_bits,
        merkle_tree_arity: stark_struct.merkle_tree_arity,
        transcript_arity: stark_struct.transcript_arity,
        merkle_tree_custom: stark_struct.merkle_tree_custom,
        last_level_verification: stark_struct.last_level_verification,
        pow_bits: fri_security.grinding_bits_query as usize,
        hash_commits: stark_struct.hash_commits,
        n_bits_ext: stark_struct.n_bits_ext,
        verification_hash_type: stark_struct.verification_hash_type.clone(),
        steps,
        n_queries: fri_security.n_queries as usize,
    };

    let boundaries: Vec<BoundaryOutput> = {
        let mut seen = Vec::new();
        let mut result = Vec::new();
        for c in &setup.constraints {
            if !seen.contains(&c.boundary) {
                seen.push(c.boundary.clone());
                let b = BoundaryOutput {
                    name: c.boundary.clone(),
                    offset_min: c.offset_min.map(|v| v as i64),
                    offset_max: c.offset_max.map(|v| v as i64),
                };
                result.push(b);
            }
        }
        if result.is_empty() {
            result.push(BoundaryOutput { name: "everyRow".to_string(), offset_min: None, offset_max: None });
        }
        result
    };

    // Build evMap from verifier code context
    let ev_map: Vec<EvMapEntry> = pil_code
        .ev_map
        .iter()
        .map(|e| EvMapEntry {
            entry_type: e.entry_type.clone(),
            id: e.id,
            prime: e.prime,
            opening_pos: e.opening_pos,
            commit_id: e.commit_id,
        })
        .collect();

    let n_stages = setup.n_stages;

    // Build cmPolsMap as flat array matching golden schema.
    // Field order differs between Q-stage entries and regular entries.
    // JS map.js builds objects in this order:
    //   setSymbolSections: {stage, name, dim, polsMapId, [stageId], [lengths], [imPol, expId]}
    //   setStageInfoSymbols: adds stagePos, then stageId if not already set
    // Result for regular entries: stage, name, dim, polsMapId, stageId, [lengths], stagePos, [imPol, expId]
    // Result for Q-stage entries: stage, name, dim, polsMapId, stagePos, stageId
    let q_stage = n_stages + 1;
    let cm_pols_map: Vec<serde_json::Value> = setup
        .cm_pols_map
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let stage = p.stage.unwrap_or(0);
            let mut obj = serde_json::Map::new();
            obj.insert("stage".to_string(), json!(stage));
            obj.insert("name".to_string(), json!(p.name));
            obj.insert("dim".to_string(), json!(p.dim));
            obj.insert("polsMapId".to_string(), json!(i));
            if stage == q_stage {
                // Q-stage: stagePos before stageId
                obj.insert("stagePos".to_string(), json!(p.stage_pos.unwrap_or(0)));
                obj.insert("stageId".to_string(), json!(p.stage_id.unwrap_or(0)));
            } else {
                // Regular: stageId first
                obj.insert("stageId".to_string(), json!(p.stage_id.unwrap_or(0)));
                // Then lengths (if present)
                if let Some(ref lengths) = p.lengths {
                    obj.insert("lengths".to_string(), json!(lengths));
                }
                // imPol and expId come before stagePos (matches JS insertion
                // order: addPol sets imPol/expId, then setStageInfoSymbols
                // appends stagePos)
                if p.im_pol {
                    obj.insert("imPol".to_string(), json!(true));
                    if let Some(eid) = p.exp_id {
                        obj.insert("expId".to_string(), json!(eid));
                    }
                }
                obj.insert("stagePos".to_string(), json!(p.stage_pos.unwrap_or(0)));
            }
            serde_json::Value::Object(obj)
        })
        .collect();

    let const_pols_map: Vec<PolMapEntry> = setup
        .const_pols_map
        .iter()
        .enumerate()
        .map(|(i, p)| PolMapEntry {
            stage: 0,
            name: p.name.clone(),
            dim: p.dim,
            pols_map_id: i,
            stage_id: p.stage_id.unwrap_or(0),
            lengths: p.lengths.clone(),
            stage_pos: None,
            im_pol: None,
            exp_id: None,
        })
        .collect();

    // Build mapSectionsN as a serde_json::Map to preserve order
    let mut map_sections_n = serde_json::Map::new();
    for (key, &val) in &setup.map_sections_n {
        map_sections_n.insert(key.clone(), json!(val));
    }

    // Build custom commits JSON
    let custom_commits_json: Vec<serde_json::Value> = setup
        .custom_commits
        .iter()
        .map(|cc| {
            let public_values: Vec<serde_json::Value> =
                cc.public_values.iter().map(|&idx| json!({"idx": idx})).collect();
            json!({
                "name": cc.name,
                "publicValues": public_values,
                "stageWidths": cc.stage_widths,
            })
        })
        .collect();

    // Build custom commits map (array of arrays, one per custom commit)
    let custom_commits_map: Vec<serde_json::Value> = setup
        .custom_commits_map
        .iter()
        .map(|cc_entries| {
            let entries: Vec<serde_json::Value> = cc_entries
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let mut obj = serde_json::Map::new();
                    obj.insert("stage".to_string(), json!(p.stage.unwrap_or(0)));
                    // Strip namespace prefix (e.g. "Rom.line" -> "line")
                    let short_name = p.name.rsplit('.').next().unwrap_or(&p.name);
                    obj.insert("name".to_string(), json!(short_name));
                    obj.insert("dim".to_string(), json!(p.dim));
                    obj.insert("polsMapId".to_string(), json!(i));
                    obj.insert("stageId".to_string(), json!(p.stage_id.unwrap_or(0)));
                    obj.insert("stagePos".to_string(), json!(p.stage_pos.unwrap_or(0)));
                    serde_json::Value::Object(obj)
                })
                .collect();
            json!(entries)
        })
        .collect();

    // Build challengesMap from setup + FRI challenges.
    let mut challenges_map: Vec<ChallengeMapEntryOutput> = setup
        .challenges_map
        .iter()
        .map(|s| ChallengeMapEntryOutput {
            name: s.name.clone(),
            stage: s.stage.unwrap_or(0),
            dim: s.dim,
            stage_id: s.stage_id.unwrap_or(0),
        })
        .collect();
    // Merge FRI challenges (std_vf1, std_vf2) by index position.
    for (i, ch) in pil_code.challenges_map.iter().enumerate() {
        if ch.name.is_empty() {
            continue;
        }
        let entry =
            ChallengeMapEntryOutput { name: ch.name.clone(), stage: ch.stage, dim: ch.dim, stage_id: ch.stage_id };
        while challenges_map.len() <= i {
            challenges_map.push(ChallengeMapEntryOutput::default());
        }
        challenges_map[i] = entry;
    }
    while challenges_map.last().is_some_and(|e| e.name.is_empty()) {
        challenges_map.pop();
    }
    challenges_map.retain(|e| !e.name.is_empty());

    // Build publicsMap
    let publics_map: Vec<PublicMapEntry> = setup
        .publics_map
        .iter()
        .map(|s| PublicMapEntry { name: s.name.clone(), stage: s.stage.unwrap_or(0), lengths: s.lengths.clone() })
        .collect();

    // Build proofValuesMap
    let proof_values_map: Vec<NameStageEntry> = setup
        .proof_values_map
        .iter()
        .map(|s| NameStageEntry { name: s.name.clone(), stage: s.stage.unwrap_or(0), lengths: s.lengths.clone() })
        .collect();

    // Build airgroupValuesMap
    let airgroup_values_map: Vec<NameStageEntry> = setup
        .airgroup_values_map
        .iter()
        .map(|s| NameStageEntry { name: s.name.clone(), stage: s.stage.unwrap_or(0), lengths: s.lengths.clone() })
        .collect();

    // Build airValuesMap
    let air_values_map: Vec<NameStageEntry> = setup
        .air_values_map
        .iter()
        .map(|s| NameStageEntry { name: s.name.clone(), stage: s.stage.unwrap_or(0), lengths: s.lengths.clone() })
        .collect();

    // Build airGroupValues
    let air_group_values: Vec<serde_json::Value> =
        setup.air_group_values.iter().map(|v| json!({"aggType": v.agg_type, "stage": v.stage})).collect();

    // nCommitmentsStage1: in JS this compares stage === "cm1" (number vs string),
    // which always evaluates to 0. Golden confirms nCommitmentsStage1=0.
    let n_commitments_stage1 = 0;

    StarkInfoOutput {
        name: air_name.to_string(),
        cm_pols_map,
        const_pols_map,
        challenges_map,
        publics_map,
        proof_values_map,
        airgroup_values_map,
        air_values_map,
        map_sections_n,
        air_id,
        airgroup_id,
        n_constants: setup.n_constants,
        n_publics: setup.n_publics,
        air_group_values,
        n_stages,
        custom_commits: custom_commits_json,
        custom_commits_map,
        stark_struct: stark_struct_out,
        boundaries,
        opening_points: opening_points.to_vec(),
        c_exp_id,
        q_dim: FIELD_EXTENSION,
        q_deg: q_deg.max(1) as usize,
        n_constraints: setup.constraints.len(),
        n_commitments_stage1,
        ev_map,
        fri_exp_id,
        security: Some(SecurityInfo {
            proximity_gap: fri.proximity_gap(),
            proximity_parameter: fri.proximity_parameter(),
            regime: "JBR".to_string(),
        }),
    }
}

pub fn collect_opening_points(setup: &SetupResult) -> Vec<i64> {
    setup.opening_points.clone()
}

pub fn compute_log_folding_factors(stark_struct: &StarkStruct) -> Vec<u32> {
    let steps = &stark_struct.steps;
    let mut factors = Vec::new();
    for i in 0..steps.len() - 1 {
        factors.push((steps[i].n_bits - steps[i + 1].n_bits) as u32);
    }
    factors
}
