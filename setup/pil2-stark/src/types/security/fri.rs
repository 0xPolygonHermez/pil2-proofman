use rug::{Float, float::Round, ops::Pow};

use super::{
    common::{hpf, security_bits_from_error},
    regimes::{DecodingRegime, CodeParams},
};

/// Parameters for FRI.
pub struct FriParams {
    /// Base-2 logarithm of the field size `|F|`.
    pub field_size_bits: Float,
    /// The dimension of the code.
    pub dimension: u64,
    /// The rate of the code.
    pub rate: f64,
    /// The number of opening points in the FRI protocol.
    pub n_opening_points: u64,
    /// The number of functions to batch together in the FRI protocol.
    pub batch_size: u64,
    /// The folding factors for FRI.
    pub folding_factors: Vec<u64>,
    /// The maximum number of grinding bits allowed.
    pub max_grinding_bits: u64,
    /// Whether to use the maximum number of grinding bits.
    pub use_max_grinding_bits: bool,
    /// The arity of the Merkle tree used in FRI.
    pub tree_arity: u64,
    /// The target security level in bits.
    pub target_security_bits: u64,
}

/// Result of optimal FRI query parameter computation.
#[derive(Debug, Clone)]
pub struct FriQueryResult {
    pub n_queries: u64,
    pub n_grinding_bits: u64,
    pub proximity_parameter: f64,
    pub proximity_gap: f64,
}

impl FriParams {
    /// Optimal FRI query count and grinding for a decoding regime.
    pub fn get_optimal_query_params(&self, regime: &DecodingRegime) -> FriQueryResult {
        let mut alpha: f64 = 0.0;

        loop {
            let code_params =
                CodeParams::new(&self.field_size_bits, self.dimension, self.rate, alpha, self.n_opening_points);

            let (n_queries, n_grinding_bits) = self.optimal_calc(regime, &code_params);

            if self.meets_security_target(regime, &code_params, n_queries, n_grinding_bits) {
                return FriQueryResult {
                    n_queries,
                    n_grinding_bits,
                    proximity_parameter: regime.proximity_parameter(&code_params).to_f64_round(Round::Nearest),
                    proximity_gap: regime.gap(&code_params).to_f64_round(Round::Nearest),
                };
            }

            // Security not met -- widen the gap by increasing alpha.
            alpha += 0.1;
            assert!(alpha < 100.0, "Alpha loop did not converge");
        }
    }

    /// The query/grinding split for a fixed regime: bits-per-query from the
    /// proximity parameter, grinding capped at what's efficient (or forced to
    /// the max), queries sized to cover the remaining target bits.
    fn optimal_calc(&self, regime: &DecodingRegime, cp: &CodeParams) -> (u64, u64) {
        let pp = regime.proximity_parameter(cp);
        let code_length = cp.length.to_f64_round(Round::Nearest);

        // Single query error = 1 - proximityParameter
        let single_query_error = hpf(1) - &pp;
        let bits_per_query = -single_query_error.to_f64_round(Round::Nearest).log2();

        // Cost per query (in hash operations)
        let hashes_per_query = calculate_query_num_hashes(self.tree_arity, code_length, &self.folding_factors);

        // Find max efficient grinding: 2^g < hashesPerQuery => g < log2(hashesPerQuery)
        let max_efficient_grinding = hashes_per_query.log2().floor() as u64;
        let n_grinding_bits = if self.use_max_grinding_bits {
            self.max_grinding_bits
        } else {
            max_efficient_grinding.min(self.max_grinding_bits)
        };

        let needed_from_queries = self.target_security_bits as f64 - n_grinding_bits as f64;
        let n_queries = if needed_from_queries > 0.0 {
            (needed_from_queries / bits_per_query).ceil() as u64
        } else {
            1 // Need at least 1 query
        };

        (n_queries, n_grinding_bits)
    }

    /// Whether the full FRI soundness meets the configured target.
    fn meets_security_target(
        &self,
        regime: &DecodingRegime,
        cp: &CodeParams,
        n_queries: u64,
        n_grinding_bits: u64,
    ) -> bool {
        let total_error = self.total_error(regime, cp, n_queries, n_grinding_bits);
        let security_bits = security_bits_from_error(&total_error);
        security_bits >= self.target_security_bits as i64
    }

    /// Total FRI error.
    fn total_error(&self, regime: &DecodingRegime, cp: &CodeParams, n_queries: u64, n_grinding_bits: u64) -> Float {
        let batch_commit_error = self.batch_commit_error(regime, cp);
        let query_phase_error = Self::query_phase_error(regime, cp, n_queries, n_grinding_bits);
        batch_commit_error.max(&query_phase_error)
    }

    /// Batch/commit error.
    fn batch_commit_error(&self, regime: &DecodingRegime, cp: &CodeParams) -> Float {
        let batch_error = regime.calculate_powers_error(cp, self.batch_size);

        let mut commit_error = hpf(0);
        for ff in self.folding_factors.iter() {
            let round_error = regime.calculate_powers_error(cp, *ff);
            if round_error > commit_error {
                commit_error = round_error;
            }
        }

        batch_error.max(&commit_error)
    }

    /// The query-phase error.
    fn query_phase_error(regime: &DecodingRegime, cp: &CodeParams, n_queries: u64, grinding_bits: u64) -> Float {
        let pp = regime.proximity_parameter(cp);

        // query error: (1 − γ)^t
        let single_query_error = hpf(1) - &pp;
        let query_error = single_query_error.pow(n_queries);

        // grinding error: 2^{−g}
        let two_pow = hpf(2).pow(grinding_bits);
        let grinding_error = hpf(1) / two_pow;

        query_error * grinding_error
    }
}

fn calculate_mtp_hashes(tree_arity: u64, n_leafs: f64) -> f64 {
    (tree_arity as f64 - 1.0) * (n_leafs.log2() / (tree_arity as f64).log2()).ceil()
}

fn calculate_query_num_hashes(tree_arity: u64, length: f64, folding_factors: &[u64]) -> f64 {
    if folding_factors.is_empty() {
        return 0.0;
    }
    let mut acc_folding_factor: f64 = 1.0;
    let mut total_hashes: f64 = 0.0;
    for &ff in &folding_factors[..folding_factors.len() - 1] {
        let n_leafs = length / acc_folding_factor;
        total_hashes += ff as f64 * calculate_mtp_hashes(tree_arity, n_leafs);
        acc_folding_factor *= ff as f64;
    }
    let n_leafs_input = length;
    total_hashes += folding_factors[0] as f64 * calculate_mtp_hashes(tree_arity, n_leafs_input);
    total_hashes
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::goldilocks_safe_extension_field_size_bits;

    /// recursivef (GL→BN128 bridge) query count as a function of grinding bits.
    /// Real recursivef params: nBits=15, blowup=6 (rate 2^-6), 4 opening points,
    /// evMap len 145, FRI folding bit-drops [3,3,3,3,2], arity 4, 128-bit target.
    /// The production setup pins powBits=19 (snark_setup.rs), giving 37 queries.
    #[test]
    fn test_recursivef_queries_vs_grinding_bits() {
        let queries = |max_grinding_bits: u64| {
            let p = FriParams {
                field_size_bits: goldilocks_safe_extension_field_size_bits(),
                dimension: 1u64 << 15,
                rate: 1.0 / (1u64 << 6) as f64,
                n_opening_points: 4,
                batch_size: 145,
                folding_factors: vec![3, 3, 3, 3, 2],
                max_grinding_bits,
                use_max_grinding_bits: true,
                tree_arity: 4,
                target_security_bits: 128,
            };
            p.get_optimal_query_params(&DecodingRegime::Jbr).n_queries
        };
        assert_eq!(queries(17), 38);
        assert_eq!(queries(19), 37);
        assert_eq!(queries(20), 37);
    }

    /// Golden reference for the shared (soundcalc-corrected) JBR regime.
    ///
    /// The original JS `security.js` example gave `n_queries = 219` (gap
    /// `0.007333`, alpha widened to 1.2). Since FRI now shares the soundcalc
    /// decoding regime, the multiplicity uses `m = ⌈√ρ/(2η)⌉` (the factor-2 that
    /// BCHKS25 Thm 4.2 omits is a confirmed typo — see soundcalc's
    /// `johnson_bound.py`) and the linear error includes the `(m+½)/√ρ` term.
    /// The tighter error needs only a single alpha widening (gap `1.1/300`), so
    /// the corrected reference is `n_queries = 216`, gap `0.003667`.
    #[test]
    fn test_golden_jbr_optimal_params() {
        let params = FriParams {
            field_size_bits: goldilocks_safe_extension_field_size_bits(),
            dimension: 1 << 17,
            rate: 0.5,
            n_opening_points: 26,
            batch_size: 4065,
            folding_factors: vec![4, 4, 4],
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };
        let result = params.get_optimal_query_params(&DecodingRegime::Jbr);
        assert_eq!(result.n_queries, 216, "nQueries mismatch");
        assert_eq!(result.n_grinding_bits, 22, "nGrindingBits mismatch");
        let expected_gap: f64 = 0.003_666_666_666_666_666;
        assert!(
            (result.proximity_gap - expected_gap).abs() < 1e-12,
            "proximity_gap mismatch: {}",
            result.proximity_gap
        );
    }

    /// alpha=0 JBR gives nQueries=215 (batch-limited to 122 bits total).
    /// Bypasses the alpha-widening loop by building the regime at alpha=0
    /// directly and calling the query/grinding split.
    #[test]
    fn test_jbr_alpha0_nqueries() {
        let params = FriParams {
            field_size_bits: goldilocks_safe_extension_field_size_bits(),
            dimension: 1 << 17,
            rate: 0.5,
            n_opening_points: 26,
            batch_size: 4065,
            folding_factors: vec![4, 4, 4],
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };
        let rp = CodeParams::new(&params.field_size_bits, params.dimension, params.rate, 0.0, params.n_opening_points);
        let length = rp.length.to_f64_round(Round::Nearest);
        let (n_queries, n_grinding_bits) = params.optimal_calc(&DecodingRegime::Jbr, &rp);
        assert_eq!(n_queries, 215, "nQueries at alpha=0 should be 215");
        assert_eq!(n_grinding_bits, 22);
    }

    /// UDR with the same params (JS output: nQueries=289).
    #[test]
    fn test_udr_optimal_params() {
        let params = FriParams {
            field_size_bits: goldilocks_safe_extension_field_size_bits(),
            dimension: 1 << 17,
            rate: 0.5,
            n_opening_points: 26,
            batch_size: 4065,
            folding_factors: vec![4, 4, 4],
            max_grinding_bits: 22,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };
        let result = params.get_optimal_query_params(&DecodingRegime::Udr);
        assert_eq!(result.n_queries, 289, "UDR nQueries mismatch");
        assert_eq!(result.n_grinding_bits, 22, "UDR nGrindingBits mismatch");
    }

    /// The field-size helper produces ~2^191.
    #[test]
    fn test_goldilocks_cube_field_size() {
        let log2_fs = goldilocks_safe_extension_field_size_bits().to_f64();
        assert!((191.0..192.0).contains(&log2_fs), "log2(fieldSize) should be ~191, got {log2_fs}");
    }

    /// Compressor params — JS golden nQueries=110.
    #[test]
    fn test_compressor_params() {
        let params = FriParams {
            field_size_bits: goldilocks_safe_extension_field_size_bits(),
            dimension: 1 << 18,
            rate: 0.25,
            n_opening_points: 6,
            batch_size: 198,
            folding_factors: vec![3, 3, 3, 3, 3],
            max_grinding_bits: 20,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };
        let result = params.get_optimal_query_params(&DecodingRegime::Jbr);
        assert_eq!(result.n_queries, 110, "nQueries mismatch: got {}", result.n_queries);
        assert_eq!(result.n_grinding_bits, 20);
    }

    /// SpecifiedRanges params — JS golden nQueries=228.
    #[test]
    fn test_specified_ranges_params() {
        let params = FriParams {
            field_size_bits: goldilocks_safe_extension_field_size_bits(),
            dimension: 1 << 8,
            rate: 0.5,
            n_opening_points: 3,
            batch_size: 9,
            folding_factors: vec![3],
            max_grinding_bits: 16,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
        };
        let result = params.get_optimal_query_params(&DecodingRegime::Jbr);
        assert_eq!(result.n_queries, 228, "nQueries mismatch: got {}", result.n_queries);
        assert_eq!(result.n_grinding_bits, 16);
    }
}
