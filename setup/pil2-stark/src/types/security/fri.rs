use rug::float::Round;
use rug::ops::Pow;
use rug::Float;

use super::regime::{DecodingRegime, RegimeParams, JBR, UDR};
use super::{calculate_query_num_hashes, hpf, security_bits_from_error, PREC};

// ---------------------------------------------------------------------------
// FRI Security Calculator
// ---------------------------------------------------------------------------

struct FRISecurityCalculator {
    n_queries: u64,
    n_grinding_bits: u64,
    proximity_parameter: Float,
    proximity_gap: Float,
    target_security_bits: u64,
}

/// Parameters for FRI security calculation (public API input).
pub struct FRISecurityParams {
    pub field_size: Float,
    pub dimension: u64,
    pub rate: f64,
    pub n_opening_points: u64,
    pub n_functions: u64,
    pub folding_factors: Vec<u64>,
    pub max_grinding_bits: u64,
    pub use_max_grinding_bits: bool,
    pub tree_arity: u64,
    pub target_security_bits: u64,
}

fn calculate_optimal_query_params(
    regime: &dyn DecodingRegime,
    folding_factors: &[u64],
    target_security_bits: u64,
    max_grinding_bits: u64,
    use_max_grinding_bits: bool,
    tree_arity: u64,
    codeword_length: f64,
) -> FRISecurityCalculator {
    let pp = regime.proximity_parameter();
    let gap = regime.gap();

    // Single query error = 1 - proximityParameter
    let single_query_error = Float::with_val(PREC, hpf(1) - &pp);
    let bits_per_query = -single_query_error.to_f64_round(Round::Nearest).log2();

    // Cost per query (in hash operations)
    let hashes_per_query = calculate_query_num_hashes(tree_arity, codeword_length, folding_factors);

    // Find max efficient grinding: 2^g < hashesPerQuery => g < log2(hashesPerQuery)
    let max_efficient_grinding = hashes_per_query.log2().floor() as u64;
    let n_grinding_bits =
        if use_max_grinding_bits { max_grinding_bits } else { max_efficient_grinding.min(max_grinding_bits) };

    let needed_from_queries = target_security_bits as f64 - n_grinding_bits as f64;
    // JS: Math.ceil(neededFromQueries / bitsPerQuery)
    let n_queries = if needed_from_queries > 0.0 {
        (needed_from_queries / bits_per_query).ceil() as u64
    } else {
        1 // Need at least 1 query
    };

    FRISecurityCalculator {
        n_queries,
        n_grinding_bits,
        proximity_parameter: pp,
        proximity_gap: gap,
        target_security_bits,
    }
}

fn meets_security_target(
    calc: &FRISecurityCalculator,
    regime: &dyn DecodingRegime,
    n_functions: u64,
    folding_factors: &[u64],
) -> bool {
    let total_bits = calculate_total_security_bits(calc, regime, n_functions, folding_factors);
    total_bits >= calc.target_security_bits as i64
}

fn calculate_total_security_bits(
    calc: &FRISecurityCalculator,
    regime: &dyn DecodingRegime,
    n_functions: u64,
    folding_factors: &[u64],
) -> i64 {
    let total_error = calculate_total_error(calc, regime, n_functions, folding_factors);
    security_bits_from_error(&total_error)
}

fn calculate_total_error(
    calc: &FRISecurityCalculator,
    regime: &dyn DecodingRegime,
    n_functions: u64,
    folding_factors: &[u64],
) -> Float {
    let batch_commit_error = calculate_batch_commit_error(regime, n_functions, folding_factors);
    let query_error = calculate_query_phase_error(calc);
    batch_commit_error.max(&query_error)
}

fn calculate_batch_commit_error(regime: &dyn DecodingRegime, n_functions: u64, folding_factors: &[u64]) -> Float {
    let batch_error = regime.calculate_powers_error(n_functions);

    let mut commit_error = hpf(0);
    for ff in folding_factors.iter() {
        let round_error = regime.calculate_powers_error(*ff);
        if round_error > commit_error {
            commit_error = round_error;
        }
    }

    batch_error.max(&commit_error)
}

fn calculate_query_phase_error(calc: &FRISecurityCalculator) -> Float {
    let two_pow = Float::with_val(PREC, hpf(2).pow(calc.n_grinding_bits as u32));
    let grinding_error = Float::with_val(PREC, hpf(1) / &two_pow);
    let single_query_error = Float::with_val(PREC, hpf(1) - &calc.proximity_parameter);
    let query_error = Float::with_val(PREC, single_query_error.pow(calc.n_queries as u32));
    Float::with_val(PREC, &query_error * &grinding_error)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of optimal FRI query parameter computation.
#[derive(Debug, Clone)]
pub struct FRIQueryResult {
    pub n_queries: u64,
    pub n_grinding_bits: u64,
    pub proximity_parameter: f64,
    pub proximity_gap: f64,
}

/// Port of `getOptimalFRIQueryParams(regime, params)` from security.js.
///
/// Computes the optimal `(nQueries, nGrindingBits)` pair for the given FRI
/// parameters, matching the JS output exactly.
pub fn get_optimal_fri_query_params(regime_name: &str, params: &FRISecurityParams) -> FRIQueryResult {
    let mut alpha: f64 = 0.0;

    loop {
        let rp =
            RegimeParams::new(params.field_size.clone(), params.dimension, params.rate, alpha, params.n_opening_points);

        let codeword_length = rp.codeword_length.to_f64_round(Round::Nearest);

        match regime_name {
            "JBR" => {
                let regime = JBR::new(&rp);
                let calc = calculate_optimal_query_params(
                    &regime,
                    &params.folding_factors,
                    params.target_security_bits,
                    params.max_grinding_bits,
                    params.use_max_grinding_bits,
                    params.tree_arity,
                    codeword_length,
                );
                if meets_security_target(&calc, &regime, params.n_functions, &params.folding_factors) {
                    return FRIQueryResult {
                        n_queries: calc.n_queries,
                        n_grinding_bits: calc.n_grinding_bits,
                        proximity_parameter: calc.proximity_parameter.to_f64_round(Round::Nearest),
                        proximity_gap: calc.proximity_gap.to_f64_round(Round::Nearest),
                    };
                }
            }
            "UDR" => {
                let regime = UDR::new(&rp);
                let calc = calculate_optimal_query_params(
                    &regime,
                    &params.folding_factors,
                    params.target_security_bits,
                    params.max_grinding_bits,
                    params.use_max_grinding_bits,
                    params.tree_arity,
                    codeword_length,
                );
                if meets_security_target(&calc, &regime, params.n_functions, &params.folding_factors) {
                    return FRIQueryResult {
                        n_queries: calc.n_queries,
                        n_grinding_bits: calc.n_grinding_bits,
                        proximity_parameter: calc.proximity_parameter.to_f64_round(Round::Nearest),
                        proximity_gap: calc.proximity_gap.to_f64_round(Round::Nearest),
                    };
                }
            }
            _ => panic!("Unknown decoding regime: {regime_name}. Supported: JBR, UDR"),
        };

        // Security not met -- widen the gap by increasing alpha
        alpha += 0.1;

        // Safety valve (should not be needed in practice)
        assert!(alpha < 100.0, "Alpha loop did not converge");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::security::goldilocks_cube_field_size;

    /// recursivef (GL→BN128 bridge) query count as a function of grinding bits.
    /// Real recursivef params from a generated recursivef.starkinfo.json:
    /// nBits=15, blowup=6 (rate 2^-6), 4 opening points, evMap len 145,
    /// FRI steps [21,18,15,12,9,7] (folding bit-drops [3,3,3,3,2]), arity 4, 128-bit target.
    /// The production setup pins powBits=19 (snark_setup.rs), giving 37 queries.
    #[test]
    fn test_recursivef_queries_vs_grinding_bits() {
        let queries = |max_grinding_bits: u64| {
            let p = FRISecurityParams {
                field_size: goldilocks_cube_field_size(),
                dimension: 1u64 << 15,
                rate: 1.0 / (1u64 << 6) as f64,
                n_opening_points: 4,
                n_functions: 145,
                folding_factors: vec![3, 3, 3, 3, 2],
                max_grinding_bits,
                use_max_grinding_bits: true,
                tree_arity: 4,
                target_security_bits: 128,
            };
            get_optimal_fri_query_params("JBR", &p).n_queries
        };

        // 19 is the production value (pinned in snark_setup.rs).
        assert_eq!(queries(17), 38);
        assert_eq!(queries(19), 37);
        assert_eq!(queries(20), 37);
    }

    /// Golden reference from running the JS security.js example with
    /// `getOptimalFRIQueryParams("JBR", params)`.
    #[test]
    fn test_golden_jbr_optimal_params() {
        let field_size = goldilocks_cube_field_size();

        let params = FRISecurityParams {
            field_size,
            dimension: 1 << 17,
            rate: 0.5,
            n_opening_points: 26,
            n_functions: 4065,
            folding_factors: vec![4, 4, 4],
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };

        let result = get_optimal_fri_query_params("JBR", &params);

        assert_eq!(result.n_queries, 219, "nQueries mismatch");
        assert_eq!(result.n_grinding_bits, 22, "nGrindingBits mismatch");

        // proximity_gap should be approximately 0.00733333...
        let expected_gap: f64 = 0.007_333_333_333_333_333;
        assert!(
            (result.proximity_gap - expected_gap).abs() < 1e-12,
            "proximity_gap mismatch: got {}, expected ~{}",
            result.proximity_gap,
            expected_gap
        );
    }

    /// Test that alpha=0 JBR (used by createSecurityCalculator) gives the expected values.
    #[test]
    fn test_jbr_alpha0_nqueries() {
        let field_size = goldilocks_cube_field_size();

        let rp = RegimeParams::new(field_size, 1 << 17, 0.5, 0.0, 26);
        let regime = JBR::new(&rp);
        let codeword_length = rp.codeword_length.to_f64_round(Round::Nearest);

        let calc = calculate_optimal_query_params(&regime, &[4, 4, 4], 128, 22, true, 4, codeword_length);

        // At alpha=0, nQueries=215 but total security is only 122 bits (batch limited).
        assert_eq!(calc.n_queries, 215, "nQueries at alpha=0 should be 215");
        assert_eq!(calc.n_grinding_bits, 22);
    }

    /// Test UDR with the same params (from JS output: nQueries=289, nGrindingBits=22).
    #[test]
    fn test_udr_optimal_params() {
        let field_size = goldilocks_cube_field_size();

        let params = FRISecurityParams {
            field_size,
            dimension: 1 << 17,
            rate: 0.5,
            n_opening_points: 26,
            n_functions: 4065,
            folding_factors: vec![4, 4, 4],
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };

        let result = get_optimal_fri_query_params("UDR", &params);

        assert_eq!(result.n_queries, 289, "UDR nQueries mismatch");
        assert_eq!(result.n_grinding_bits, 22, "UDR nGrindingBits mismatch");
    }

    /// Test with different folding factors (foldingFactors: [4, 3, 3]).
    #[test]
    fn test_jbr_different_folding() {
        let field_size = goldilocks_cube_field_size();

        let params = FRISecurityParams {
            field_size,
            dimension: 1 << 17,
            rate: 0.5,
            n_opening_points: 26,
            n_functions: 4065,
            folding_factors: vec![4, 3, 3],
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };

        let result = get_optimal_fri_query_params("JBR", &params);

        // Verify the result is reasonable (security met)
        assert!(result.n_queries > 0);
        assert!(result.n_grinding_bits <= 22);
    }

    /// Test with rate = 1/4 (blowupFactor=4 -> nBitsExt = nBits + 2).
    #[test]
    fn test_jbr_rate_quarter() {
        let field_size = goldilocks_cube_field_size();

        let params = FRISecurityParams {
            field_size,
            dimension: 1 << 17,
            rate: 0.25,
            n_opening_points: 26,
            n_functions: 4065,
            folding_factors: vec![4, 4, 4],
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };

        let result = get_optimal_fri_query_params("JBR", &params);

        // With a lower rate, we should need fewer queries
        assert!(result.n_queries > 0);
        assert!(result.n_queries < 219, "lower rate should yield fewer queries, got {}", result.n_queries);
    }

    /// Test the exact compressor parameters to match the JS golden output (nQueries=110).
    #[test]
    fn test_compressor_params() {
        let field_size = goldilocks_cube_field_size();

        let params = FRISecurityParams {
            field_size,
            dimension: 1 << 18,
            rate: 0.25,
            n_opening_points: 6,
            n_functions: 198,
            folding_factors: vec![3, 3, 3, 3, 3],
            max_grinding_bits: 20,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };

        let result = get_optimal_fri_query_params("JBR", &params);

        eprintln!("nQueries={}, gap={}, pp={}", result.n_queries, result.proximity_gap, result.proximity_parameter);
        assert_eq!(result.n_queries, 110, "nQueries mismatch: got {}", result.n_queries);
        assert_eq!(result.n_grinding_bits, 20);
    }

    /// Test SpecifiedRanges params — JS gives nQueries=228.
    #[test]
    fn test_specified_ranges_params() {
        let field_size = goldilocks_cube_field_size();

        let params = FRISecurityParams {
            field_size,
            dimension: 1 << 8,
            rate: 0.5,
            n_opening_points: 3,
            n_functions: 9,
            folding_factors: vec![3],
            max_grinding_bits: 16,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };

        let result = get_optimal_fri_query_params("JBR", &params);

        eprintln!("nQueries={}, gap={}, pp={}", result.n_queries, result.proximity_gap, result.proximity_parameter);
        assert_eq!(result.n_queries, 228, "nQueries mismatch: got {}", result.n_queries);
        assert_eq!(result.n_grinding_bits, 16);
    }
}
