//! The starkinfo output: the core's `StarkInfoOutput` with the solved stark struct and security.

use crate::types::security;
use crate::types::security::pcs::{Batching, LowDegreeTest, Pcs};
use crate::types::security::regimes::DecodingRegime;
use crate::types::stark_struct::{FriStruct, StarkStruct, StirStruct};
use crate::types::output::{SecurityInfo, StarkInfoOutput, StarkStructOutput};

/// The starkinfo output: the core's projection plus the solved low-degree test.
pub fn build_starkinfo_output_from_core(
    core: &pil_info::core::project::StarkInfoCore,
    stark_struct: &StarkStruct,
    ldt: &LowDegreeTest,
    airgroup_id: usize,
    air_id: usize,
    air_name: &str,
) -> anyhow::Result<StarkInfoOutput> {
    let security = SecurityInfo {
        proximity_gap: ldt.proximity_gap(),
        proximity_parameter: ldt.proximity_parameter(),
        regime: ldt.regime_identifier().to_string(),
    };
    let ss = stark_struct_output(stark_struct, ldt);
    pil_info::core::project::starkinfo_output(core, air_name, airgroup_id, air_id, ss, Some(security))
        .map_err(|e| anyhow::anyhow!("starkinfo: {e:?}"))
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

/// Solve the security parameters (query counts, grinding) of the low-degree
/// test the stark struct selects, for a DEEP polynomial batching `batch_size`
/// openings. The 128-bit JBR target and the geometry (rate, tree arity, pow
/// bits) come from the stark struct, as the FRI-only callers always did.
/// The gate covers the low-degree test only: the DEEP-ALI terms (~qDeg, nConstraints over |GL³| ≈ 2^192) are negligible.
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
    let log_folding_factors = stark_struct.low_degree_test.log_folding_factors();
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
