//! Build the various JSON output structures for starkinfo, expressionsinfo, and verifierinfo.
//!
//! per-AIR child-process pipeline in `setup_cmd.rs` and from the recursive/final
//! setup modules.

use serde_json::json;

use crate::pil::gen_code::{PilCodeResult, ProcessedHintField};
use crate::types::pilout_info::{SetupResult, FIELD_EXTENSION};
use crate::types::security;
use crate::types::stark_struct::StarkStruct;
use crate::types::output::{
    BoundaryOutput, ChallengeMapEntryOutput, CodeEntry, CodeRef, EvMapEntry, NameStageEntry, PolMapEntry,
    PublicMapEntry, SecurityInfo, StarkInfoOutput, StarkStructOutput, StepOutput,
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

/// Build the expressionsinfo JSON structure.
pub fn build_expressions_info_json(info: &crate::pil::gen_code::ExpressionsInfo) -> serde_json::Value {
    let expressions_code: Vec<serde_json::Value> = info
        .expressions_code
        .iter()
        .map(|e| {
            let mut obj = serde_json::Map::new();
            obj.insert("tmpUsed".to_string(), json!(e.tmp_used));
            obj.insert("code".to_string(), code_entries_to_json(&e.code));
            obj.insert("expId".to_string(), json!(e.exp_id));
            obj.insert("stage".to_string(), json!(e.stage));
            if let Some(ref dest) = e.dest {
                obj.insert(
                    "dest".to_string(),
                    json!({
                        "op": dest.op,
                        "stage": dest.stage,
                        "stageId": dest.stage_id,
                        "id": dest.id,
                    }),
                );
            }
            obj.insert("line".to_string(), json!(e.line));
            serde_json::Value::Object(obj)
        })
        .collect();

    let constraints: Vec<serde_json::Value> = info
        .constraints
        .iter()
        .map(|c| {
            let mut obj = serde_json::Map::new();
            obj.insert("tmpUsed".to_string(), json!(c.tmp_used));
            obj.insert("code".to_string(), code_entries_to_json(&c.code));
            obj.insert("boundary".to_string(), json!(c.boundary));
            if let Some(ref line) = c.line {
                obj.insert("line".to_string(), json!(line));
            }
            obj.insert("imPol".to_string(), json!(c.im_pol));
            obj.insert("stage".to_string(), json!(c.stage));
            if let Some(omin) = c.offset_min {
                obj.insert("offsetMin".to_string(), json!(omin));
            }
            if let Some(omax) = c.offset_max {
                obj.insert("offsetMax".to_string(), json!(omax));
            }
            serde_json::Value::Object(obj)
        })
        .collect();

    let hints_info: Vec<serde_json::Value> = info
        .hints_info
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

    // Build with explicit ordering matching golden: hintsInfo, expressionsCode, constraints
    let mut result = serde_json::Map::new();
    result.insert("hintsInfo".to_string(), serde_json::Value::Array(hints_info));
    result.insert("expressionsCode".to_string(), serde_json::Value::Array(expressions_code));
    result.insert("constraints".to_string(), serde_json::Value::Array(constraints));
    serde_json::Value::Object(result)
}

/// Build the verifierinfo JSON structure.
pub fn build_verifier_info_json(info: &crate::pil::gen_code::VerifierInfo) -> serde_json::Value {
    // qVerifier: {tmpUsed, code, line: ""}
    let mut qv = serde_json::Map::new();
    qv.insert("tmpUsed".to_string(), json!(info.q_verifier.tmp_used));
    qv.insert("code".to_string(), code_entries_to_json(&info.q_verifier.code));
    qv.insert("line".to_string(), json!(""));

    // queryVerifier: {tmpUsed, code, expId, stage, line}
    let mut qr = serde_json::Map::new();
    qr.insert("tmpUsed".to_string(), json!(info.query_verifier.tmp_used));
    qr.insert("code".to_string(), code_entries_to_json(&info.query_verifier.code));
    qr.insert("expId".to_string(), json!(info.query_verifier.exp_id));
    qr.insert("stage".to_string(), json!(info.query_verifier.stage));
    qr.insert("line".to_string(), json!(info.query_verifier.line));

    let mut result = serde_json::Map::new();
    result.insert("qVerifier".to_string(), serde_json::Value::Object(qv));
    result.insert("queryVerifier".to_string(), serde_json::Value::Object(qr));
    serde_json::Value::Object(result)
}

/// Serialize a hint field value to JSON matching golden field order per op type.
///
/// Field order per op type (matching JS object spread behavior):
///   string:         op, string, pos
///   number:         op, value, pos
///   tmp:            op, id, dim, pos
///   cm/custom/const: op, id, stageId, rowOffset, stage, dim, [commitId,] rowOffsetIndex, pos
///   challenge/public/airgroupvalue/airvalue/proofvalue:
///                   op, id, dim, stage, [stageId,] [airgroupId,] pos
pub(crate) fn hint_value_to_json(v: &ProcessedHintField) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("op".to_string(), json!(v.op));
    match v.op.as_str() {
        "string" => {
            if let Some(ref val) = v.value {
                obj.insert("string".to_string(), json!(val));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "number" => {
            if let Some(ref val) = v.value {
                obj.insert("value".to_string(), json!(val));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "tmp" => {
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "cm" | "custom" | "const" => {
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(sid) = v.stage_id {
                obj.insert("stageId".to_string(), json!(sid));
            }
            if let Some(ro) = v.row_offset {
                obj.insert("rowOffset".to_string(), json!(ro));
            }
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            if let Some(cid) = v.commit_id {
                obj.insert("commitId".to_string(), json!(cid));
            }
            if let Some(roi) = v.row_offset_index {
                obj.insert("rowOffsetIndex".to_string(), json!(roi));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "challenge" => {
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            if let Some(sid) = v.stage_id {
                obj.insert("stageId".to_string(), json!(sid));
            }
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "airgroupvalue" => {
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(agid) = v.airgroup_id {
                obj.insert("airgroupId".to_string(), json!(agid));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        "public" => {
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
        _ => {
            // airvalue, proofvalue
            if let Some(id) = v.id {
                obj.insert("id".to_string(), json!(id));
            }
            if let Some(stage) = v.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            if let Some(dim) = v.dim {
                obj.insert("dim".to_string(), json!(dim));
            }
            obj.insert("pos".to_string(), json!(v.pos));
        }
    }
    serde_json::Value::Object(obj)
}

pub fn code_entries_to_json(entries: &[CodeEntry]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| {
            json!({
                "op": e.op,
                "dest": code_ref_to_json(&e.dest),
                "src": e.src.iter().map(code_ref_to_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::Value::Array(arr)
}

/// Serialize a code ref to JSON matching the golden field order per type.
///
/// Each type has its own property order matching the JS object construction
/// order in codegen.js. The patterns are:
///   tmp (from binary op):  type, id, dim
///   tmp (from exp ref):    type, [expId,] id, prime, dim
///   cm/const/custom:       type, id, prime, dim, [commitId]
///   eval:                  type, id, dim
///   challenge:             type, id, stageId, dim, stage
///   public:                type, id, dim
///   proofvalue:            type, id, stage, dim
///   number:                type, value, dim
///   airgroupvalue/airvalue: type, id, stage, dim, [airgroupId]
///   xDivXSubXi:            type, id, opening, dim
///   Zi:                    type, boundaryId, dim
pub fn code_ref_to_json(r: &CodeRef) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".to_string(), json!(r.ref_type));
    match r.ref_type.as_str() {
        "tmp" => {
            // Three tmp patterns based on JS object construction order:
            //   1. Binary op dest:       type, id, dim (no prime, no expId)
            //   2. pilCodeGen wrapper:    type, prime, id, dim (prime present, no expId)
            //   3. evalExp exp ref:       type, expId, id, prime, dim
            if let Some(eid) = r.exp_id {
                obj.insert("expId".to_string(), json!(eid));
                obj.insert("id".to_string(), json!(r.id));
                if let Some(prime) = r.prime {
                    obj.insert("prime".to_string(), json!(prime));
                }
            } else if let Some(prime) = r.prime {
                obj.insert("prime".to_string(), json!(prime));
                obj.insert("id".to_string(), json!(r.id));
            } else {
                obj.insert("id".to_string(), json!(r.id));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "cm" | "const" | "custom" => {
            // For ImPol cm refs, expId comes before id (matching JS object
            // property insertion order from evalExp where r starts as
            // {type: "exp", expId, id, ...} then fixCommitPol changes type/id)
            if let Some(eid) = r.exp_id {
                obj.insert("expId".to_string(), json!(eid));
            }
            obj.insert("id".to_string(), json!(r.id));
            obj.insert("prime".to_string(), json!(r.prime.unwrap_or(0)));
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(cid) = r.commit_id {
                obj.insert("commitId".to_string(), json!(cid));
            }
        }
        "number" => {
            if let Some(ref value) = r.value {
                obj.insert("value".to_string(), json!(value));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "challenge" => {
            obj.insert("id".to_string(), json!(r.id));
            if let Some(sid) = r.stage_id {
                obj.insert("stageId".to_string(), json!(sid));
            }
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(stage) = r.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
        }
        "eval" => {
            if let Some(eid) = r.exp_id {
                obj.insert("expId".to_string(), json!(eid));
            }
            obj.insert("id".to_string(), json!(r.id));
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(cid) = r.commit_id {
                obj.insert("commitId".to_string(), json!(cid));
            }
        }
        "public" => {
            obj.insert("id".to_string(), json!(r.id));
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "proofvalue" => {
            obj.insert("id".to_string(), json!(r.id));
            if let Some(stage) = r.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "airgroupvalue" | "airvalue" => {
            obj.insert("id".to_string(), json!(r.id));
            if let Some(stage) = r.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(agid) = r.airgroup_id {
                obj.insert("airgroupId".to_string(), json!(agid));
            }
        }
        "xDivXSubXi" => {
            obj.insert("id".to_string(), json!(r.id));
            if let Some(opening) = r.opening {
                obj.insert("opening".to_string(), json!(opening));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        "Zi" => {
            if let Some(bid) = r.boundary_id {
                obj.insert("boundaryId".to_string(), json!(bid));
            }
            obj.insert("dim".to_string(), json!(r.dim));
        }
        _ => {
            obj.insert("id".to_string(), json!(r.id));
            obj.insert("dim".to_string(), json!(r.dim));
            if let Some(prime) = r.prime {
                obj.insert("prime".to_string(), json!(prime));
            }
            if let Some(ref value) = r.value {
                obj.insert("value".to_string(), json!(value));
            }
            if let Some(stage) = r.stage {
                obj.insert("stage".to_string(), json!(stage));
            }
        }
    }
    serde_json::Value::Object(obj)
}
