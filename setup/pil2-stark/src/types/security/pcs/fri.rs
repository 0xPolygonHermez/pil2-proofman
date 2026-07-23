use super::super::regimes::{DecodingRegime, ProximityGapsRegime};
use super::types::{Batching, Pcs, apply_grinding, bits_of_security_from_error, bits_of_security_from_log2_error};

/// Configuration for the FRI PCS: the *free* parameters, i.e. everything
/// known upfront. Query counts and grinding bits are deduced from these at
/// construction time (see [`FriSecurityParams`]).
#[derive(Clone, Debug)]
pub struct FriConfig {
    /// Field size |F|.
    pub field_size: f64,
    /// Domain size before low-degree extension (trace length). Must be 2^h.
    pub trace_length: u32,
    /// The code rate ρ. Must be an exact power of two 2^-k.
    pub rate: f64,
    /// Number of functions in batched FRI.
    pub batch_size: u64,
    /// Batching strategy (see [`Batching`]).
    pub batching: Batching,
    /// FRI folding factors, one per round. Each must be a power of two.
    pub folding_factors: Vec<u32>,
    /// Domain size at which folding stops.
    pub early_stop_degree: u32,
    /// The maximum number of grinding bits allowed.
    pub max_grinding_bits: u64,
    /// Whether to use the maximum number of grinding bits.
    pub use_max_grinding_bits: bool,
    /// The arity of the Merkle tree used in FRI.
    pub tree_arity: u64,
    /// The target security level in bits.
    pub target_security_bits: u64,
    /// The decoding regime kind. The gap-widening factor `alpha` is deduced.
    pub regime: DecodingRegime,
}

/// Security parameters *deduced* from a [`FriConfig`]: the query/grinding
/// split that reaches the configured target.
#[derive(Clone, Debug)]
pub struct FriSecurityParams {
    /// The number of queries to make in the query phase.
    pub n_queries: u64,
    /// Grinding needed by the batching step to reach the target.
    pub grinding_bits_batching: u32,
    /// Grinding needed per folding round to reach the target.
    pub grinding_bits_folding: Vec<u32>,
    /// Query-phase grinding (pow bits).
    pub grinding_bits_query: u32,
}

/// FRI Polynomial Commitment Scheme.
///
/// Built in two phases: [`Fri::new`] validates the structural (free)
/// parameters, then *solves* for the security parameters — per-phase
/// grinding deficits, the query grinding budget, the query count, and the
/// regime's gap-widening factor `alpha` — so that every phase meets
/// `target_security_bits`. After construction the instance is self-contained.
#[derive(Clone, Debug)]
pub struct Fri {
    cfg: FriConfig,
    /// k = -log2(ρ), exact.
    log_inv_rate: u32,
    /// h = log2(trace_length), exact.
    log_trace: u32,
    /// Domain size after low-degree extension: D = trace_length << k, exact.
    domain_size: u64,
    /// Number of FRI folding rounds.
    num_rounds: usize,
    /// dimension of the (partially folded) code entering commit round i,
    /// i.e. trace_length / Π_{j<=i} folding_factors[j]. Precomputed and
    /// validated to be exact.
    round_dimensions: Vec<u32>,
    /// Deduced security parameters (query/grinding split).
    sec_params: FriSecurityParams,
    /// Deduced gap-widening factor for the regime.
    alpha: f64,
}

impl Fri {
    pub fn new(cfg: FriConfig) -> Self {
        let num_rounds = cfg.folding_factors.len();
        assert!(num_rounds > 0, "FRI must have at least one folding round (folding_factors non-empty)");

        // ρ = 2^-k.
        let k = (-cfg.rate.log2()).round();
        assert!(k >= 1.0 && k <= 32.0 && cfg.rate == f64::exp2(-k), "ρ must be an exact power of two");
        let log_inv_rate = k as u32;

        // trace_length = 2^h.
        assert!(cfg.trace_length.is_power_of_two(), "trace_length must be a power of two");
        let log_trace = cfg.trace_length.trailing_zeros();

        // Domain size n = trace_length / ρ = trace_length << k.
        assert!(log_trace + log_inv_rate < 64, "domain size overflowed u64");
        let domain_size = (cfg.trace_length as u64) << log_inv_rate;

        // Folding schedule validation, plus precomputation of per-round code dimensions.
        let mut n = domain_size;
        let mut dim = cfg.trace_length;
        let mut round_dimensions = Vec::with_capacity(num_rounds);
        for (round, &factor) in cfg.folding_factors.iter().enumerate() {
            assert!(factor.is_power_of_two(), "folding factor must be a power of two");
            assert!(
                n % factor as u64 == 0 && dim % factor == 0,
                "folding overflows domain: round {}, remaining {}, factor {}",
                round,
                n,
                factor
            );
            n /= factor as u64;
            dim /= factor;
            round_dimensions.push(dim);
        }
        assert!(
            n == cfg.early_stop_degree as u64,
            "early stop mismatch: after {} rounds, reached {}, expected {}",
            num_rounds,
            n,
            cfg.early_stop_degree
        );

        // Phase 1 done (structure). Phase 2: solve for the security params.
        let mut fri = Self {
            cfg,
            log_inv_rate,
            log_trace,
            domain_size,
            num_rounds,
            round_dimensions,
            sec_params: FriSecurityParams {
                n_queries: 0,
                grinding_bits_batching: 0,
                grinding_bits_folding: vec![0; num_rounds],
                grinding_bits_query: 0,
            },
            alpha: 0.0,
        };
        let (sec_params, alpha) = fri.solve();
        fri.sec_params = sec_params;
        fri.alpha = alpha;
        fri
    }

    /// Search for the smallest gap-widening `alpha` whose query/grinding
    /// split meets the target on every phase.
    fn solve(&self) -> (FriSecurityParams, f64) {
        let mut alpha: f64 = 0.0;
        loop {
            let regime = self.cfg.regime.instantiate(self.cfg.field_size, alpha);
            let sec_params = self.compute_security_params(regime.as_ref());
            if self.meets_security_target(regime.as_ref(), &sec_params) {
                return (sec_params, alpha);
            }

            // Security not met -- widen the gap by increasing alpha.
            alpha += 0.1;
            assert!(alpha < 100.0, "Alpha loop did not converge");
        }
    }

    /// The query/grinding split for a fixed regime: bits-per-query from the
    /// proximity parameter, grinding capped at what's efficient (or forced to
    /// the max), queries sized to cover the remaining target bits. The
    /// non-query phases get whatever grinding they are missing to the target.
    fn compute_security_params(&self, regime: &dyn ProximityGapsRegime) -> FriSecurityParams {
        let target = self.cfg.target_security_bits;

        // Non-query stages: grinding fills whatever is missing to the target.
        let deficit = |error: f64| target.saturating_sub(bits_of_security_from_error(error) as u64) as u32;

        // Batching grinding. With a single function there is no batching
        // step (its error is exactly 0), so no grinding is needed either.
        let grinding_bits_batching = if self.cfg.batch_size > 1 { deficit(self.batching_error(regime, 0)) } else { 0 };

        // Folding grinding.
        let grinding_bits_folding =
            (0..self.num_rounds).map(|round| deficit(self.commit_phase_error(round, regime, 0))).collect();

        // Bits contributed by a single query: -log2(1 − γ).
        let bits_per_query = -(1.0 - regime.proximity_parameter(&self.cfg.rate)).log2();

        // Find max efficient grinding: 2^g < hashesPerQuery => g < log2(hashesPerQuery).
        let max_efficient_grinding = self.query_num_hashes().log2().floor() as u64;
        let grinding_bits_query = if self.cfg.use_max_grinding_bits {
            self.cfg.max_grinding_bits
        } else {
            max_efficient_grinding.min(self.cfg.max_grinding_bits)
        };

        // Compute the number of queries needed to reach the target security level.
        let needed_from_queries = target as f64 - grinding_bits_query as f64;
        let n_queries = if needed_from_queries > 0.0 {
            (needed_from_queries / bits_per_query).ceil() as u64
        } else {
            1 // Need at least 1 query
        };

        FriSecurityParams {
            n_queries,
            grinding_bits_batching,
            grinding_bits_folding,
            grinding_bits_query: grinding_bits_query as u32,
        }
    }

    /// Whether every phase meets the configured target.
    fn meets_security_target(&self, regime: &dyn ProximityGapsRegime, sec_params: &FriSecurityParams) -> bool {
        self.security_levels_with(regime, sec_params)
            .into_iter()
            .all(|(_, bits)| bits as u64 >= self.cfg.target_security_bits)
    }

    /// Approximate verifier hash count per query, used to bound the grinding
    /// bits worth spending (grinding beyond the per-query cost is wasteful).
    /// Per round the verifier opens one coset of `factor` leaves plus a
    /// Merkle path in that round's tree.
    fn query_num_hashes(&self) -> f64 {
        let mut n_leafs = self.domain_size as f64;
        let mut total = 0.0;
        for &factor in &self.cfg.folding_factors {
            n_leafs /= factor as f64;
            total += factor as f64 + merkle_path_hashes(self.cfg.tree_arity, n_leafs);
        }
        total
    }

    /// PCS-specific security levels, in bits, phase by phase.
    /// Ordered: batching, commit rounds 1..=n, query phase.
    pub fn security_levels(&self) -> Vec<(String, u32)> {
        let regime = self.regime();
        self.security_levels_with(regime.as_ref(), &self.sec_params)
    }

    fn security_levels_with(
        &self,
        regime: &dyn ProximityGapsRegime,
        sec_params: &FriSecurityParams,
    ) -> Vec<(String, u32)> {
        let mut bits = Vec::with_capacity(self.num_rounds + 2);

        // Batching phase; absent when there is only one function to commit.
        if self.cfg.batch_size > 1 {
            bits.push((
                "batching".to_string(),
                bits_of_security_from_error(self.batching_error(regime, sec_params.grinding_bits_batching)),
            ));
        }

        // Commit phase: one entry per folding round.
        for i in 0..self.num_rounds {
            bits.push((
                format!("commit round {}", i + 1),
                bits_of_security_from_error(self.commit_phase_error(i, regime, sec_params.grinding_bits_folding[i])),
            ));
        }

        // Query phase.
        bits.push((
            "query phase".to_string(),
            bits_of_security_from_log2_error(self.query_phase_error(
                regime,
                sec_params.n_queries,
                sec_params.grinding_bits_query,
            )),
        ));

        bits
    }

    /// Error from the batching step; depends on the batching strategy.
    fn batching_error(&self, regime: &dyn ProximityGapsRegime, grinding_bits: u32) -> f64 {
        let rate = self.cfg.rate;
        let dimension = self.cfg.trace_length;

        let epsilon = match self.cfg.batching {
            Batching::Powers => regime.error_powers(&rate, dimension, self.cfg.batch_size),
            Batching::Multilinear => regime.error_multilinear(&rate, dimension, self.cfg.batch_size),
            Batching::Affine => regime.error_linear(&rate, dimension),
        };

        apply_grinding(epsilon, grinding_bits)
    }

    /// Error from round `round` (0-indexed) of the commit phase.
    ///
    /// Folding by factor m is batching m functions over the folded domain
    /// with powers of the folding challenge, hence `error_powers`.
    fn commit_phase_error(&self, round: usize, regime: &dyn ProximityGapsRegime, grinding_bits: u32) -> f64 {
        let rate = self.cfg.rate;
        let dimension = self.round_dimensions[round];
        let epsilon = regime.error_powers(&rate, dimension, self.cfg.folding_factors[round] as u64);
        apply_grinding(epsilon, grinding_bits)
    }

    /// log2 of the error from the FRI query phase. Computed in log2 space:
    /// (1 − γ)^t can underflow f64 for large query counts.
    fn query_phase_error(&self, regime: &dyn ProximityGapsRegime, n_queries: u64, grinding_bits: u32) -> f64 {
        // query error: (1 − γ)^t, where γ is the proximity parameter and
        // t the number of queries.
        let pp = regime.proximity_parameter(&self.cfg.rate);
        debug_assert!(pp > 0.0 && pp < 1.0, "proximity parameter {pp} outside (0,1)");

        n_queries as f64 * (1.0 - pp).log2() - grinding_bits as f64
    }

    /// The solved regime.
    pub fn regime(&self) -> Box<dyn ProximityGapsRegime> {
        self.cfg.regime.instantiate(self.cfg.field_size, self.alpha)
    }

    pub fn regime_identifier(&self) -> &'static str {
        self.regime().identifier()
    }

    /// The gap η of the solved regime at this code's rate.
    pub fn proximity_gap(&self) -> f64 {
        self.regime().gap(&self.cfg.rate)
    }

    /// The proximity parameter γ of the solved regime at this code's rate.
    pub fn proximity_parameter(&self) -> f64 {
        self.regime().proximity_parameter(&self.cfg.rate)
    }

    /// The deduced security parameters (query count, grinding split).
    pub fn security_params(&self) -> &FriSecurityParams {
        &self.sec_params
    }

    /// The deduced gap-widening factor.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn rate(&self) -> f64 {
        self.cfg.rate
    }

    pub fn log_inv_rate(&self) -> u32 {
        self.log_inv_rate
    }

    pub fn dimension(&self) -> u32 {
        self.cfg.trace_length
    }

    pub fn trace_length(&self) -> u32 {
        self.cfg.trace_length
    }

    pub fn domain_size(&self) -> u64 {
        self.domain_size
    }

    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    pub fn config(&self) -> &FriConfig {
        &self.cfg
    }

    /// Description of the parameters of the PCS (Markdown code block).
    pub fn parameter_summary(&self) -> String {
        let params: Vec<(&str, String)> = vec![
            ("rho", self.cfg.rate.to_string()),
            ("k = -log2(rho)", self.log_inv_rate.to_string()),
            ("trace_length", self.cfg.trace_length.to_string()),
            ("h = log2(trace_length)", self.log_trace.to_string()),
            ("domain_size D = trace_length / rho", self.domain_size.to_string()),
            ("batch_size", self.cfg.batch_size.to_string()),
            ("batching", self.cfg.batching.to_string()),
            ("FRI_folding_factors", format!("{:?}", self.cfg.folding_factors)),
            ("FRI_early_stop_degree", self.cfg.early_stop_degree.to_string()),
            ("FRI_rounds_n", self.num_rounds.to_string()),
            ("regime", format!("{:?} (alpha = {})", self.cfg.regime, self.alpha)),
            ("target_security_bits", self.cfg.target_security_bits.to_string()),
            ("n_queries (deduced)", self.sec_params.n_queries.to_string()),
            ("grinding_query (deduced)", self.sec_params.grinding_bits_query.to_string()),
            ("grinding_batching (deduced)", self.sec_params.grinding_bits_batching.to_string()),
            ("grinding_folding (deduced)", format!("{:?}", self.sec_params.grinding_bits_folding)),
        ];

        let key_width = params.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        let mut out = String::from("\n```\n");
        for (k, v) in &params {
            out.push_str(&format!("  {k:<key_width$} : {v}\n"));
        }
        out.push_str("```");
        out
    }
}

/// Hashes to verify one Merkle path in a tree of `n_leafs` leaves.
fn merkle_path_hashes(tree_arity: u64, n_leafs: f64) -> f64 {
    (tree_arity as f64 - 1.0) * (n_leafs.log2() / (tree_arity as f64).log2()).ceil()
}

impl Pcs for Fri {
    fn identifier(&self) -> &'static str {
        "FRI"
    }

    fn security_levels(&self) -> Vec<(String, u32)> {
        Fri::security_levels(self)
    }

    fn rate(&self) -> f64 {
        Fri::rate(self)
    }

    fn dimension(&self) -> u32 {
        Fri::dimension(self)
    }

    fn parameter_summary(&self) -> String {
        Fri::parameter_summary(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::goldilocks_safe_extension_field_size;
    use super::*;

    fn test_config(
        trace_length: u32,
        rate: f64,
        batch_size: u64,
        folding_bits: Vec<u32>,
        max_grinding_bits: u64,
    ) -> FriConfig {
        // Domain = trace / rate, folded by Π 2^bits.
        let log_inv_rate = (-rate.log2()).round() as u32;
        let total_fold_bits: u32 = folding_bits.iter().sum();
        let early_stop_degree = (((trace_length as u64) << log_inv_rate) >> total_fold_bits) as u32;
        FriConfig {
            field_size: goldilocks_safe_extension_field_size(),
            trace_length,
            rate,
            batch_size,
            batching: Batching::Powers,
            folding_factors: folding_bits.iter().map(|&b| 1u32 << b).collect(),
            early_stop_degree,
            max_grinding_bits,
            use_max_grinding_bits: true,
            tree_arity: 4,
            target_security_bits: 128,
            regime: DecodingRegime::Jbr,
        }
    }

    /// recursivef (GL→BN128 bridge) query count as a function of the query
    /// grinding budget. Real recursivef params: nBits=15, blowup=6 (rate 2^-6),
    /// evMap len 145, FRI folding factors [3,3,3,3,2] (bits), arity 4, 128-bit
    /// target. The production setup pins powBits=19 (snark_setup.rs), giving
    /// 37 queries.
    #[test]
    fn test_recursivef_queries_vs_grinding_bits() {
        let queries = |max_grinding_bits: u64| {
            let fri =
                Fri::new(test_config(1 << 15, 1.0 / (1u64 << 6) as f64, 145, vec![3, 3, 3, 3, 2], max_grinding_bits));
            let sec = fri.security_params();
            assert_eq!(sec.grinding_bits_batching, 0);
            assert_eq!(sec.grinding_bits_folding, vec![0; 5]);
            sec.n_queries
        };
        assert_eq!(queries(17), 38);
        assert_eq!(queries(19), 37);
        assert_eq!(queries(20), 37);
    }

    /// Golden reference cross-checked against soundcalc's formulas with the
    /// pil2 gap η = 1/300: 215 queries; the batching step falls 1 bit short of
    /// the 128-bit target and gets it back as grinding.
    #[test]
    fn test_golden_jbr_optimal_params() {
        let fri = Fri::new(test_config(1 << 17, 0.5, 4065, vec![4, 4, 4], 22));
        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 215, "nQueries mismatch");
        assert_eq!(sec.grinding_bits_query, 22);
        assert_eq!(sec.grinding_bits_batching, 1, "batching deficit should be 1 bit");
        assert_eq!(sec.grinding_bits_folding, vec![0; 3]);
        assert_eq!(fri.alpha(), 0.0);
    }

    /// UDR with the same params.
    #[test]
    fn test_udr_optimal_params() {
        let cfg = FriConfig { regime: DecodingRegime::Udr, ..test_config(1 << 17, 0.5, 4065, vec![4, 4, 4], 22) };
        let fri = Fri::new(cfg);
        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 289, "UDR nQueries mismatch");
        assert_eq!(sec.grinding_bits_query, 22);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![0; 3]);
    }

    /// Compressor params.
    #[test]
    fn test_compressor_params() {
        let fri = Fri::new(test_config(1 << 18, 0.25, 198, vec![3, 3, 3, 3, 3], 20));
        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 110, "nQueries mismatch: got {}", sec.n_queries);
        assert_eq!(sec.grinding_bits_query, 20);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![0; 5]);
    }

    /// SpecifiedRanges params.
    #[test]
    fn test_specified_ranges_params() {
        let fri = Fri::new(test_config(1 << 8, 0.5, 9, vec![3], 16));
        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 228, "nQueries mismatch: got {}", sec.n_queries);
        assert_eq!(sec.grinding_bits_query, 16);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![0]);
    }

    /// Every component reported by `security_levels` meets the target.
    #[test]
    fn test_security_levels_meet_target() {
        let fri = Fri::new(test_config(1 << 17, 0.5, 4065, vec![4, 4, 4], 22));
        for (name, bits) in fri.security_levels() {
            assert!(bits >= 128, "component {name} gives {bits} bits (< 128)");
        }
    }
}
