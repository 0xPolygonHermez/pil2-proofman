//! Build the starkinfo JSON output structure (expressionsinfo and verifierinfo are built by
//! `pil_info::output::expressions_info`).
//!
//! per-AIR child-process pipeline in `setup_cmd.rs` and from the recursive/final
//! setup modules.

use serde_json::json;

use crate::pil::gen_code::PilCodeResult;
use pil_info::pil::constraint_poly::Boundary;
use crate::types::pilout_info::SetupResult;
use crate::types::FIELD_EXTENSION;
use crate::types::security;
use crate::types::security::pcs::{Batching, LowDegreeTest, Pcs};
use crate::types::security::regimes::DecodingRegime;
use crate::types::stark_struct::{FriStruct, StarkStruct, StirStruct};
use crate::types::output::{
    ChallengeMapEntryOutput, EvMapEntry, NameStageEntry, PolMapEntry, PublicMapEntry, SecurityInfo, StarkInfoOutput,
    StarkStructOutput,
};

/// Build the StarkInfoOutput for JSON serialization from internal types.
#[allow(clippy::too_many_arguments)]
pub fn build_starkinfo_output(
    setup: &SetupResult,
    stark_struct: &StarkStruct,
    pil_code: &PilCodeResult,
    opening_points: &[i64],
    ldt: &LowDegreeTest,
    airgroup_id: usize,
    air_id: usize,
    circuit_type: Option<u64>,
    air_name: &str,
    c_exp_id: usize,
    deep_exp_id: usize,
    q_deg: i64,
    boundaries: &[Boundary],
) -> StarkInfoOutput {
    // The query counts and grinding bits are the solved values, not the settings.
    let low_degree_test = match (ldt, &stark_struct.low_degree_test) {
        (LowDegreeTest::Fri(fri), crate::types::stark_struct::LowDegreeTest::Fri(schedule)) => {
            let sec = fri.security_params();
            let solved = FriStruct {
                num_queries: sec.n_queries as usize,
                grinding_bits_queries: sec.grinding_bits_query as usize,
                ..schedule.clone()
            };
            crate::types::stark_struct::LowDegreeTest::Fri(solved)
        }
        (LowDegreeTest::Stir(stir), crate::types::stark_struct::LowDegreeTest::Stir(schedule)) => {
            let sec = stir.security_params();
            let solved = StirStruct {
                num_queries: sec.num_queries.iter().map(|&t| t as usize).collect(),
                grinding_bits_queries: sec.grinding_bits_queries.iter().map(|&g| g as usize).collect(),
                ..schedule.clone()
            };
            crate::types::stark_struct::LowDegreeTest::Stir(solved)
        }
        (ldt, schedule) => panic!(
            "solved low-degree test {} does not match the stark struct's {:?}",
            ldt.identifier(),
            schedule.kind()
        ),
    };
    let stark_struct_out = StarkStructOutput {
        n_bits: stark_struct.n_bits,
        merkle_tree_arity: stark_struct.merkle_tree_arity,
        transcript_arity: stark_struct.transcript_arity,
        merkle_tree_custom: stark_struct.merkle_tree_custom,
        last_level_verification: stark_struct.last_level_verification,
        hash_commits: stark_struct.hash_commits,
        n_bits_ext: stark_struct.n_bits_ext,
        verification_hash_type: stark_struct.verification_hash_type.clone(),
        low_degree_test,
    };

    let boundaries = boundaries.to_vec();

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
        circuit_type,
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
        deep_exp_id,
        fri_batch_size: pil_code.fri_batch_size,
        security: Some(SecurityInfo {
            proximity_gap: ldt.proximity_gap(),
            proximity_parameter: ldt.proximity_parameter(),
            regime: ldt.regime_identifier().to_string(),
        }),
        multiproof: false,
    }
}

pub fn collect_opening_points(setup: &SetupResult) -> Vec<i64> {
    setup.opening_points.clone()
}

pub fn compute_log_folding_factors(stark_struct: &StarkStruct) -> Vec<u32> {
    stark_struct.low_degree_test.log_folding_factors()
}

/// Solve the security parameters (query counts, grinding) of the low-degree
/// test the stark struct selects, for a DEEP polynomial batching `batch_size`
/// openings. The 128-bit JBR target and the geometry (rate, tree arity, pow
/// bits) come from the stark struct, as the FRI-only callers always did.
pub fn solve_low_degree_test(stark_struct: &StarkStruct, batch_size: u64) -> LowDegreeTest {
    let field_size = security::goldilocks_safe_extension_field_size();
    let rate = 1.0 / (1u64 << (stark_struct.n_bits_ext - stark_struct.n_bits)) as f64;
    let log_folding_factors = compute_log_folding_factors(stark_struct);
    match &stark_struct.low_degree_test {
        crate::types::stark_struct::LowDegreeTest::Fri(fri) => {
            LowDegreeTest::Fri(security::pcs::Fri::new(security::pcs::FriConfig {
                field_size,
                trace_length: 1u32 << stark_struct.n_bits,
                rate,
                batch_size,
                batching: Batching::Powers,
                log_folding_factors,
                max_grinding_bits_query: fri.grinding_bits_queries as u64,
                use_max_grinding_bits_query: true,
                // The prover grinds only before the queries are drawn.
                max_grinding_bits_non_query: 0,
                tree_arity: stark_struct.merkle_tree_arity as u64,
                hash_size_bits: 256,
                target_security_bits: 128,
                regime: DecodingRegime::Jbr,
            }))
        }
        crate::types::stark_struct::LowDegreeTest::Stir(stir) => {
            let solved = security::pcs::Stir::new(security::pcs::StirConfig {
                field_size,
                trace_length: 1u32 << stark_struct.n_bits,
                rate,
                batch_size,
                batching: Batching::Powers,
                log_folding_factors,
                max_grinding_bits_queries: stir.grinding_bits_queries.iter().map(|&g| g as u64).collect(),
                use_max_grinding_bits_query: true,
                // The prover grinds only before each round's queries are drawn.
                max_grinding_bits_non_query: 0,
                tree_arity: stark_struct.merkle_tree_arity as u64,
                hash_size_bits: 256,
                base_field_bits: 64,
                target_security_bits: 128,
                regime: DecodingRegime::Jbr,
            });
            // The GPU prover sizes every query buffer for round 1's count.
            let t = &solved.security_params().num_queries;
            if let Some(i) = t.iter().position(|&ti| ti > t[0]) {
                panic!(
                    "STIR schedule needs {} queries in round {i} but {} in round 0 (queries {t:?}, grinding {:?}); \
                     every round must need at most round 0's queries: raise the later rounds' grindingBitsQueries",
                    t[i], t[0], stir.grinding_bits_queries
                );
            }
            LowDegreeTest::Stir(solved)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::stark_struct::{generate_stark_struct, LowDegreeTest as Ldt, LowDegreeTestKind, StarkSettings};

    fn stir(n_bits: usize, blowup: usize, fold: usize, grinding: &[usize]) -> StarkStruct {
        let settings = StarkSettings {
            low_degree_test: Some(LowDegreeTestKind::Stir),
            initial_blowup_factor: Some(blowup),
            initial_folding_factor: Some(fold),
            ..Default::default()
        };
        let mut ss = generate_stark_struct(&settings, n_bits, "blake3");
        if let Ldt::Stir(s) = &mut ss.low_degree_test {
            s.grinding_bits_queries =
                grinding.iter().copied().chain(std::iter::repeat(0)).take(s.folding_factors.len()).collect();
        }
        ss
    }

    #[test]
    fn basic_air_schedule_needs_no_more_queries_after_round_0() {
        solve_low_degree_test(&stir(22, 1, 4, &[24]), 200);
    }

    /// The setup writes a basic air's circuitType and proofman reads the same value back.
    #[test]
    fn starkinfo_output_carries_circuit_type() {
        // proofman requires a security section, which the default leaves out.
        let security = Some(SecurityInfo { proximity_gap: 0.0, proximity_parameter: 0.0, regime: "JBR".into() });
        let with = StarkInfoOutput { circuit_type: Some(4), security: security.clone(), ..Default::default() };
        let json = crate::output::json::to_json_string(&with).unwrap();
        assert!(json.contains("\"circuitType\": 4") || json.contains("\"circuitType\":4"), "{json}");
        assert_eq!(proofman_common::StarkInfo::from_json(&json).circuit_type, Some(4));

        let without = crate::output::json::to_json_string(&StarkInfoOutput { security, ..Default::default() }).unwrap();
        assert!(!without.contains("circuitType"), "{without}");
        assert_eq!(proofman_common::StarkInfo::from_json(&without).circuit_type, None);
    }

    /// The GPU sizes its query buffers for round 0; a schedule that needs more later fails at setup.
    #[test]
    #[should_panic(expected = "every round must need at most round 0's queries")]
    fn later_round_needing_more_queries_is_refused() {
        solve_low_degree_test(&stir(20, 4, 2, &[30]), 200);
    }
}
