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
use crate::types::security::pcs::{Batching, LowDegreeTest, Pcs};
use crate::types::security::regimes::DecodingRegime;
use crate::types::stark_struct::{FriStruct, StarkStruct, StirStruct};
use crate::types::output::{
    BoundaryOutput, ChallengeMapEntryOutput, EvMapEntry, NameStageEntry, PolMapEntry, PublicMapEntry, SecurityInfo,
    StarkInfoOutput, StarkStructOutput,
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
    q_deg: i64,
) -> StarkInfoOutput {
    let stark_struct_out = stark_struct_output(stark_struct, ldt);

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

    // challengesMap: the setup's, then the DEEP challenges (std_vf1, std_vf2).
    let challenges_map: Vec<ChallengeMapEntryOutput> = setup
        .challenges_map
        .iter()
        .filter(|s| !s.name.is_empty())
        .map(|s| ChallengeMapEntryOutput {
            name: s.name.clone(),
            stage: s.stage.unwrap_or(0),
            dim: s.dim,
            stage_id: s.stage_id.unwrap_or(0),
        })
        .chain(pil_code.challenges_map.iter().map(|ch| ChallengeMapEntryOutput {
            name: ch.name.clone(),
            stage: ch.stage,
            dim: ch.dim,
            stage_id: ch.stage_id,
        }))
        .collect();

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
        fri_batch_size: pil_code.fri_batch_size,
        security: Some(SecurityInfo {
            proximity_gap: ldt.proximity_gap(),
            proximity_parameter: ldt.proximity_parameter(),
            regime: ldt.regime_identifier().to_string(),
        }),
        multiproof: false,
    }
}

/// As `build_starkinfo_output`, from the thin core's projection (the old one stays until Task 8).
pub fn build_starkinfo_output_from_core(
    core: &pil_info::core::project::StarkInfoCore,
    stark_struct: &StarkStruct,
    ldt: &LowDegreeTest,
    airgroup_id: usize,
    air_id: usize,
    circuit_type: Option<u64>,
    air_name: &str,
) -> anyhow::Result<StarkInfoOutput> {
    use pil_info::core::ir::{Boundary, ColType, NamedValue};

    let v = &core.verifier;
    let named = |m: &[NamedValue]| -> Vec<NameStageEntry> {
        m.iter()
            .map(|e| NameStageEntry { name: e.name.clone(), stage: e.stage as usize, lengths: lens(&e.lengths) })
            .collect()
    };
    let publics_map = v
        .publics_map
        .iter()
        .map(|e| PublicMapEntry { name: e.name.clone(), stage: e.stage as usize, lengths: lens(&e.lengths) })
        .collect();
    let challenges_map = v
        .challenges_map
        .iter()
        .map(|c| ChallengeMapEntryOutput {
            name: c.name.clone(),
            stage: c.stage as usize,
            dim: c.dim as usize,
            stage_id: c.stage_id as usize,
        })
        .collect();
    let boundaries = v
        .boundaries
        .iter()
        .map(|b| {
            let (name, min, max) = match b {
                Boundary::EveryRow => ("everyRow", None, None),
                Boundary::FirstRow => ("firstRow", None, None),
                Boundary::LastRow => ("lastRow", None, None),
                Boundary::EveryFrame { min, max } => ("everyFrame", Some(*min as i64), Some(*max as i64)),
            };
            BoundaryOutput { name: name.to_string(), offset_min: min, offset_max: max }
        })
        .collect();
    let ev_map = v
        .ev_map
        .iter()
        .map(|e| EvMapEntry {
            entry_type: match e.pol.col {
                ColType::Witness => "cm",
                ColType::Fixed => "const",
                ColType::Custom => "custom",
            }
            .to_string(),
            id: e.pol.id as usize,
            prime: e.pol.offset,
            opening_pos: e.opening_pos as usize,
            commit_id: e.pol.commit_id.map(|c| c as usize),
        })
        .collect();
    let air_group_values =
        v.airgroup_values.iter().map(|(agg, stage)| json!({"aggType": agg, "stage": stage})).collect();

    Ok(StarkInfoOutput {
        name: air_name.to_string(),
        cm_pols_map: cm_pols_json(core)?,
        const_pols_map: const_pols_json(&v.const_pols_map),
        challenges_map,
        publics_map,
        proof_values_map: named(&v.proof_values_map),
        airgroup_values_map: named(&v.airgroup_values_map),
        air_values_map: named(&v.air_values_map),
        map_sections_n: v.map_sections_n.iter().map(|(k, v)| (k.clone(), json!(v))).collect(),
        air_id,
        airgroup_id,
        circuit_type,
        n_constants: v.n_constants as usize,
        n_publics: v.n_publics as usize,
        air_group_values,
        n_stages: v.n_stages as usize,
        custom_commits: custom_commits_json(&v.custom_commits),
        custom_commits_map: custom_pols_json(&v.custom_pols_map),
        stark_struct: stark_struct_output(stark_struct, ldt),
        boundaries,
        opening_points: v.opening_points.clone(),
        c_exp_id: v.c_exp_id as usize,
        q_dim: v.q_dim as usize,
        q_deg: v.q_deg as usize,
        n_constraints: v.n_constraints as usize,
        n_commitments_stage1: 0,
        ev_map,
        fri_batch_size: v.fri_batch_size as usize,
        security: Some(SecurityInfo {
            proximity_gap: ldt.proximity_gap(),
            proximity_parameter: ldt.proximity_parameter(),
            regime: ldt.regime_identifier().to_string(),
        }),
        multiproof: false,
    })
}

fn lens(l: &Option<Vec<u32>>) -> Option<Vec<usize>> {
    l.as_ref().map(|v| v.iter().map(|x| *x as usize).collect())
}

/// cmPolsMap in the golden key order: Q pieces put stagePos before stageId, the others after lengths/imPol/expId.
fn cm_pols_json(core: &pil_info::core::project::StarkInfoCore) -> anyhow::Result<Vec<serde_json::Value>> {
    let v = &core.verifier;
    let n_im = v.cm_pols_map.iter().filter(|p| p.im_pol).count();
    anyhow::ensure!(
        n_im == core.prover.im_exp_ids.len(),
        "{n_im} im columns for {} im expressions",
        core.prover.im_exp_ids.len()
    );
    let mut im_exp_ids = core.prover.im_exp_ids.iter();
    let q_stage = v.n_stages + 1;
    let pols = v.cm_pols_map.iter().enumerate().map(|(i, p)| {
        let mut obj = serde_json::Map::new();
        obj.insert("stage".into(), json!(p.stage));
        obj.insert("name".into(), json!(p.name));
        obj.insert("dim".into(), json!(p.dim));
        obj.insert("polsMapId".into(), json!(i));
        if p.stage == q_stage {
            obj.insert("stagePos".into(), json!(p.stage_pos));
            obj.insert("stageId".into(), json!(p.stage_id));
        } else {
            obj.insert("stageId".into(), json!(p.stage_id));
            if let Some(l) = lens(&p.lengths) {
                obj.insert("lengths".into(), json!(l));
            }
            if p.im_pol {
                obj.insert("imPol".into(), json!(true));
                obj.insert("expId".into(), json!(im_exp_ids.next()));
            }
            obj.insert("stagePos".into(), json!(p.stage_pos));
        }
        serde_json::Value::Object(obj)
    });
    Ok(pols.collect())
}

fn const_pols_json(m: &[pil_info::core::layout::PolMap]) -> Vec<PolMapEntry> {
    m.iter()
        .enumerate()
        .map(|(i, p)| PolMapEntry {
            stage: 0,
            name: p.name.clone(),
            dim: p.dim as usize,
            pols_map_id: i,
            stage_id: p.stage_id as usize,
            lengths: lens(&p.lengths),
            stage_pos: None,
            im_pol: None,
            exp_id: None,
        })
        .collect()
}

fn custom_commits_json(c: &[pil_info::core::ir::CustomCommit]) -> Vec<serde_json::Value> {
    c.iter()
        .map(|c| {
            let public_values: Vec<serde_json::Value> = c.public_values.iter().map(|i| json!({"idx": i})).collect();
            json!({"name": c.name, "publicValues": public_values, "stageWidths": c.stage_widths})
        })
        .collect()
}

fn custom_pols_json(m: &[Vec<pil_info::core::layout::PolMap>]) -> Vec<serde_json::Value> {
    m.iter()
        .map(|m| {
            let entries: Vec<serde_json::Value> = m
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let mut obj = serde_json::Map::new();
                    obj.insert("stage".into(), json!(p.stage));
                    obj.insert("name".into(), json!(p.name.rsplit('.').next().unwrap_or(&p.name)));
                    obj.insert("dim".into(), json!(p.dim));
                    obj.insert("polsMapId".into(), json!(i));
                    obj.insert("stageId".into(), json!(p.stage_id));
                    obj.insert("stagePos".into(), json!(p.stage_pos));
                    serde_json::Value::Object(obj)
                })
                .collect();
            json!(entries)
        })
        .collect()
}

/// The stark struct with the solved query counts and grinding bits.
fn stark_struct_output(stark_struct: &StarkStruct, ldt: &LowDegreeTest) -> StarkStructOutput {
    use crate::types::stark_struct::LowDegreeTest as Schedule;
    let low_degree_test = match (ldt, &stark_struct.low_degree_test) {
        (LowDegreeTest::Fri(fri), Schedule::Fri(schedule)) => {
            let sec = fri.security_params();
            Schedule::Fri(FriStruct {
                num_queries: sec.n_queries as usize,
                grinding_bits_queries: sec.grinding_bits_query as usize,
                ..schedule.clone()
            })
        }
        (LowDegreeTest::Stir(stir), Schedule::Stir(schedule)) => {
            let sec = stir.security_params();
            Schedule::Stir(StirStruct {
                num_queries: sec.num_queries.iter().map(|&t| t as usize).collect(),
                grinding_bits_queries: sec.grinding_bits_queries.iter().map(|&g| g as usize).collect(),
                ..schedule.clone()
            })
        }
        (ldt, schedule) => panic!(
            "solved low-degree test {} does not match the stark struct's {:?}",
            ldt.identifier(),
            schedule.kind()
        ),
    };
    StarkStructOutput {
        n_bits: stark_struct.n_bits,
        merkle_tree_arity: stark_struct.merkle_tree_arity,
        transcript_arity: stark_struct.transcript_arity,
        merkle_tree_custom: stark_struct.merkle_tree_custom,
        last_level_verification: stark_struct.last_level_verification,
        hash_commits: stark_struct.hash_commits,
        n_bits_ext: stark_struct.n_bits_ext,
        verification_hash_type: stark_struct.verification_hash_type.clone(),
        low_degree_test,
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
    let ldt = solve_low_degree_test_unchecked(stark_struct, batch_size);
    match security::check::check_low_degree_test(stark_struct, &ldt, batch_size) {
        Ok(report) => tracing::info!("{} (2^{} domain, {})", report.summary(), stark_struct.n_bits_ext, report.config),
        Err(e) => panic!("low-degree test fails the exact security check: {e}"),
    }
    ldt
}

fn solve_low_degree_test_unchecked(stark_struct: &StarkStruct, batch_size: u64) -> LowDegreeTest {
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

    /// The core's starkinfo is the old one byte for byte on random pilouts, with and without `--impols`.
    #[test]
    fn starkinfo_matches_old_oracle() {
        use crate::output::json::to_json_string;
        use crate::output::test_support::{core_starkinfo, full_random_in};

        for ips in [false, true] {
            let runs = full_random_in(10, ips);
            assert_eq!(runs.len(), 10, "not enough feasible random pilouts");
            for (k, f) in runs.iter().enumerate() {
                let new = core_starkinfo(&f.run, &f.stark_struct, &f.ldt, 0, 0, "Synthetic");
                assert_eq!(to_json_string(&new).unwrap(), to_json_string(&f.old_out).unwrap(), "pilout {k} {ips}");
            }
        }
    }

    /// Byte-equal starkinfo on every air of every example pilout (2^18 rows), the hints' points from the roles.
    #[test]
    fn starkinfo_matches_old_oracle_on_real_pilouts() {
        use crate::output::json::to_json_string;
        use crate::output::test_support::common::{old_run, real_pilouts};
        use crate::output::test_support::{core_starkinfo, solved, N_BITS};

        let mut compared = 0;
        for (f, mut p) in real_pilouts() {
            for g in 0..p.air_groups.len() {
                for a in 0..p.air_groups[g].airs.len() {
                    p.air_groups[g].airs[a].num_rows = Some(1 << N_BITS);
                    let found = (1..=3).find_map(|bits: usize| old_run(&p, g, a, bits).map(|r| (r, solved(bits))));
                    let Some((run, (ss, ldt))) = found else { continue };
                    let old = &run.old;
                    let name = old.setup.name.clone();
                    let ct = Some(proofman_common::circuit_type_of(0, 1));
                    let old_out = build_starkinfo_output(
                        &old.setup,
                        ss,
                        &old.pil_code,
                        &collect_opening_points(&old.setup),
                        ldt,
                        g,
                        a,
                        ct,
                        &name,
                        old.c_exp_id,
                        old.q_deg,
                    );
                    let new = core_starkinfo(&run, ss, ldt, g, a, &name);
                    assert_eq!(to_json_string(&new).unwrap(), to_json_string(&old_out).unwrap(), "{f:?} {g}/{a}");
                    compared += 1;
                }
            }
        }
        assert!(compared >= 40, "only {compared} airs compared");
    }
}
