//! Build the starkinfo JSON output structure from the core's projection (expressionsinfo and
//! verifierinfo are built by `pil_info::output::expressions_info`).

use serde_json::json;

use crate::types::security;
use crate::types::security::pcs::{Batching, LowDegreeTest, Pcs};
use crate::types::security::regimes::DecodingRegime;
use crate::types::stark_struct::{FriStruct, StarkStruct, StirStruct};
use crate::types::output::{
    BoundaryOutput, ChallengeMapEntryOutput, EvMapEntry, NameStageEntry, PolMapEntry, PublicMapEntry, SecurityInfo,
    StarkInfoOutput, StarkStructOutput,
};

/// The starkinfo JSON output: the thin core's projection plus the solved low-degree test.
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
}
