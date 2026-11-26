//! The starkinfo output: the core's verifier view with the solved stark struct and security.

use anyhow::{bail, Result};
use pil_info::core::ir::{Boundary, ColType, CustomCommit, Lengths, NamedValue};
use pil_info::core::layout::{ChallengeEntry, PolMap};
use pil_info::core::openings::EvEntry;
use pil_info::core::project::VerifierView;
use proofman_util::KEY_VERSION;
use serde::{Serialize, Serializer};

use crate::types::output::{SecurityInfo, StarkStructOutput};
use crate::types::security;
use crate::types::security::pcs::{Batching, LowDegreeTest, Pcs};
use crate::types::security::regimes::DecodingRegime;
use crate::types::stark_struct::{FriStruct, StarkStruct, StirStruct};

/// The starkinfo output: the core's projection plus the solved low-degree test, as key (`airgroup`, `air`).
pub fn build_starkinfo_output(
    v: &VerifierView,
    stark_struct: &StarkStruct,
    ldt: &LowDegreeTest,
    ids: (usize, usize),
    air_name: &str,
) -> Result<StarkInfoOutput> {
    let security = SecurityInfo {
        proximity_gap: ldt.proximity_gap(),
        proximity_parameter: ldt.proximity_parameter(),
        regime: ldt.regime_identifier().to_string(),
    };
    Ok(starkinfo_output(v, air_name, ids, stark_struct_output(stark_struct, ldt)?, security))
}

/// The stark struct with the solved query counts and grinding bits.
fn stark_struct_output(stark_struct: &StarkStruct, ldt: &LowDegreeTest) -> Result<StarkStructOutput> {
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
        (ldt, schedule) => {
            bail!("solved low-degree test {} does not match the stark struct's {:?}", ldt.identifier(), schedule.kind())
        }
    };
    Ok(StarkStructOutput {
        n_bits: stark_struct.n_bits,
        merkle_tree_arity: stark_struct.merkle_tree_arity,
        transcript_arity: stark_struct.transcript_arity,
        merkle_tree_custom: stark_struct.merkle_tree_custom,
        last_level_verification: stark_struct.last_level_verification,
        hash_commits: stark_struct.hash_commits,
        n_bits_ext: stark_struct.n_bits_ext,
        verification_hash_type: stark_struct.verification_hash_type.clone(),
        low_degree_test,
    })
}

/// Solve the security parameters (query counts, grinding) of the low-degree
/// test the stark struct selects, for a DEEP polynomial batching `batch_size`
/// openings. The 128-bit JBR target and the geometry (rate, tree arity, pow
/// bits) come from the stark struct, as the FRI-only callers always did.
/// The gate covers the low-degree test only: the DEEP-ALI terms (~qDeg, nConstraints over |GL³| ≈ 2^192) are negligible.
pub fn solve_low_degree_test(stark_struct: &StarkStruct, batch_size: u64) -> Result<LowDegreeTest> {
    let ldt = solve_low_degree_test_unchecked(stark_struct, batch_size)?;
    match security::check::check_low_degree_test(stark_struct, &ldt, batch_size) {
        Ok(report) => tracing::info!("{} (2^{} domain, {})", report.summary(), stark_struct.n_bits_ext, report.config),
        Err(e) => bail!("low-degree test fails the exact security check: {e}"),
    }
    Ok(ldt)
}

fn solve_low_degree_test_unchecked(stark_struct: &StarkStruct, batch_size: u64) -> Result<LowDegreeTest> {
    let field_size = security::goldilocks_safe_extension_field_size();
    let rate = 1.0 / (1u64 << (stark_struct.n_bits_ext - stark_struct.n_bits)) as f64;
    let log_folding_factors = stark_struct.low_degree_test.log_folding_factors();
    Ok(match &stark_struct.low_degree_test {
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
                bail!(
                    "STIR schedule needs {} queries in round {i} but {} in round 0 (queries {t:?}, grinding {:?}); \
                     every round must need at most round 0's queries: raise the later rounds' grindingBitsQueries",
                    t[i],
                    t[0],
                    stir.grinding_bits_queries
                );
            }
            LowDegreeTest::Stir(solved)
        }
    })
}

/// The starkinfo file in its golden key order.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StarkInfoOutput {
    /// Always `KEY_VERSION`; the provers refuse any other.
    #[serde(serialize_with = "key_version")]
    pub key_version: (),
    pub name: String,
    pub cm_pols_map: Vec<PolEntry>,
    pub const_pols_map: Vec<PolEntry>,
    pub challenges_map: Vec<ChallengeEntry>,
    pub publics_map: Vec<NamedValue>,
    pub proof_values_map: Vec<NamedValue>,
    pub airgroup_values_map: Vec<NamedValue>,
    pub air_values_map: Vec<NamedValue>,
    #[serde(serialize_with = "as_map")]
    pub map_sections_n: Vec<(String, u32)>,
    pub air_id: usize,
    pub airgroup_id: usize,
    /// Basic airs only: what their transcript and stage-1 contribution absorb.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub circuit_type: Option<u64>,
    pub n_constants: u32,
    pub n_publics: u32,
    pub air_group_values: Vec<AirGroupValueEntry>,
    pub n_stages: u32,
    pub custom_commits: Vec<CustomCommit>,
    pub custom_commits_map: Vec<Vec<PolEntry>>,
    pub stark_struct: StarkStructOutput,
    pub boundaries: Vec<BoundaryEntry>,
    pub opening_points: Vec<i64>,
    pub c_exp_id: u32,
    pub q_dim: u32,
    pub q_deg: u32,
    pub n_constraints: u32,
    pub n_commitments_stage1: u32,
    pub ev_map: Vec<EvMapEntry>,
    pub fri_batch_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security: Option<SecurityInfo>,
    /// The proof travels as a Merkle multiproof; absent on flat keys.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub multiproof: bool,
}

/// A cmPolsMap/constPolsMap/customCommitsMap entry; a Q piece writes stagePos before stageId (golden order).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PolEntry {
    pub stage: u32,
    pub name: String,
    pub dim: u32,
    pub pols_map_id: usize,
    #[serde(rename = "stagePos", skip_serializing_if = "Option::is_none")]
    pub q_stage_pos: Option<u32>,
    pub stage_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lengths: Lengths,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub im_pol: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_pos: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AirGroupValueEntry {
    pub agg_type: i32,
    pub stage: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct BoundaryEntry {
    pub name: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_min: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_max: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvMapEntry {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub id: u32,
    pub prime: i64,
    pub opening_pos: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_id: Option<u32>,
}

fn key_version<S: Serializer>(_: &(), s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u32(KEY_VERSION)
}

/// `mapSectionsN` keeps its insertion order.
fn as_map<S: Serializer>(v: &[(String, u32)], s: S) -> Result<S::Ok, S::Error> {
    s.collect_map(v.iter().map(|(k, n)| (k, n)))
}

/// The starkinfo of `v` with its `stark_struct` and `security`.
fn starkinfo_output(
    v: &VerifierView,
    air_name: &str,
    (airgroup_id, air_id): (usize, usize),
    stark_struct: StarkStructOutput,
    security: SecurityInfo,
) -> StarkInfoOutput {
    let boundary = |b: &Boundary| {
        let (offset_min, offset_max) = match b {
            Boundary::EveryFrame { min, max } => (Some(*min), Some(*max)),
            _ => (None, None),
        };
        BoundaryEntry { name: b.name(), offset_min, offset_max }
    };
    let ev = |e: &EvEntry| EvMapEntry {
        kind: match e.pol.col {
            ColType::Witness => "cm",
            ColType::Fixed => "const",
            ColType::Custom => "custom",
        },
        id: e.pol.id,
        prime: e.pol.offset,
        opening_pos: e.opening_pos,
        commit_id: e.pol.commit_id,
    };
    let entry = |i: usize, p: &PolMap| PolEntry {
        stage: p.stage,
        name: p.name.clone(),
        dim: p.dim,
        pols_map_id: i,
        stage_id: p.stage_id,
        ..Default::default()
    };
    let const_pols =
        v.const_pols_map.iter().enumerate().map(|(i, p)| PolEntry { lengths: p.lengths.clone(), ..entry(i, p) });
    // a custom entry is named by the column's own name (after the last `.`)
    let custom_pol = |(i, p): (usize, &PolMap)| PolEntry {
        name: p.name.rsplit_once('.').map_or(p.name.as_str(), |(_, n)| n).into(),
        stage_pos: Some(p.stage_pos),
        ..entry(i, p)
    };
    StarkInfoOutput {
        key_version: (),
        name: air_name.into(),
        cm_pols_map: cm_pols(v, entry),
        const_pols_map: const_pols.collect(),
        challenges_map: v.challenges_map.clone(),
        publics_map: v.publics_map.clone(),
        proof_values_map: v.proof_values_map.clone(),
        airgroup_values_map: v.airgroup_values_map.clone(),
        air_values_map: v.air_values_map.clone(),
        map_sections_n: v.map_sections_n.clone(),
        air_id,
        airgroup_id,
        circuit_type: v.circuit_type,
        n_constants: v.n_constants,
        n_publics: v.n_publics,
        air_group_values: (v.airgroup_values_map.iter())
            .map(|a| AirGroupValueEntry { agg_type: a.agg_type as i32, stage: a.stage })
            .collect(),
        n_stages: v.n_stages,
        custom_commits: v.custom_commits.clone(),
        custom_commits_map: v.custom_pols_map.iter().map(|m| m.iter().enumerate().map(custom_pol).collect()).collect(),
        stark_struct,
        boundaries: v.boundaries.iter().map(boundary).collect(),
        opening_points: v.opening_points.clone(),
        c_exp_id: v.c_exp_id,
        q_dim: v.q_dim,
        q_deg: v.q_deg,
        n_constraints: v.n_constraints,
        n_commitments_stage1: 0,
        ev_map: v.ev_map.iter().map(ev).collect(),
        fri_batch_size: v.fri_batch_size,
        security: Some(security),
        multiproof: v.multiproof,
    }
}

/// cmPolsMap: a Q piece carries its stagePos first; the others lengths, imPol/expId, then stagePos.
fn cm_pols(v: &VerifierView, entry: impl Fn(usize, &PolMap) -> PolEntry) -> Vec<PolEntry> {
    let q_stage = v.n_stages + 1;
    let pol = |(i, p): (usize, &PolMap)| {
        if p.stage == q_stage {
            return PolEntry { q_stage_pos: Some(p.stage_pos), ..entry(i, p) };
        }
        let im_pol = p.exp_id.map(|_| true);
        PolEntry { lengths: p.lengths.clone(), im_pol, exp_id: p.exp_id, stage_pos: Some(p.stage_pos), ..entry(i, p) }
    };
    v.cm_pols_map.iter().enumerate().map(pol).collect()
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
        let mut ss = generate_stark_struct(&settings, n_bits, proofman_common::hash_family::HashFamily::Blake3);
        if let Ldt::Stir(s) = &mut ss.low_degree_test {
            s.grinding_bits_queries =
                grinding.iter().copied().chain(std::iter::repeat(0)).take(s.folding_factors.len()).collect();
        }
        ss
    }

    #[test]
    fn basic_air_schedule_needs_no_more_queries_after_round_0() {
        solve_low_degree_test(&stir(22, 1, 4, &[24]), 200).unwrap();
    }

    /// The setup writes a basic air's circuitType and proofman reads the same value back.
    #[test]
    fn starkinfo_output_carries_circuit_type() {
        // proofman requires a security section, which the default leaves out.
        let security = Some(SecurityInfo { proximity_gap: 0.0, proximity_parameter: 0.0, regime: "JBR".into() });
        let with = StarkInfoOutput { circuit_type: Some(4), security: security.clone(), ..Default::default() };
        let json = crate::output::json::to_json_string(&with).unwrap();
        assert!(json.contains("\"circuitType\": 4") || json.contains("\"circuitType\":4"), "{json}");
        assert_eq!(proofman_common::StarkInfo::try_from_json(&json).unwrap().circuit_type, Some(4));

        let without = crate::output::json::to_json_string(&StarkInfoOutput { security, ..Default::default() }).unwrap();
        assert!(!without.contains("circuitType"), "{without}");
        assert_eq!(proofman_common::StarkInfo::try_from_json(&without).unwrap().circuit_type, None);
    }

    /// The GPU sizes its query buffers for round 0; a schedule that needs more later fails at setup.
    #[test]
    fn later_round_needing_more_queries_is_refused() {
        let err = solve_low_degree_test(&stir(20, 4, 2, &[30]), 200).unwrap_err().to_string();
        assert!(err.contains("every round must need at most round 0's queries"), "{err}");
    }
}
