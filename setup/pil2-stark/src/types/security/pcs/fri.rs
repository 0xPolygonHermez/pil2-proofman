use super::super::regimes::{DecodingRegime, ProximityGapsRegime};
use super::types::{Batching, Pcs, apply_grinding, bits_of_security_from_error, security_from_error, merkle_path_hashes};

/// Configuration for the FRI PCS.
#[derive(Clone, Debug)]
pub struct FriConfig {
    /// Field size |F|.
    pub field_size: f64,
    /// The decoding regime kind. The gap-widening factor `alpha` is deduced.
    pub regime: DecodingRegime,
    /// Domain size before low-degree extension. Must be 2^h.
    pub trace_length: u32,
    /// The code rate ρ. Must be an exact power of two 2^-k.
    pub rate: f64,
    /// Batching strategy (see [`Batching`]).
    pub batching: Batching,
    /// Number of polynomials batched.
    pub batch_size: u64,
    /// FRI folding factors, one per round. Each must be a power of two.
    pub folding_factors: Vec<u32>,
    /// The maximum number of grinding bits allowed.
    pub max_grinding_bits_query: u64,
    /// Whether to use the maximum number of grinding bits.
    pub use_max_grinding_bits_query: bool,
    /// The arity of the Merkle tree used in FRI.
    pub tree_arity: u64,
    /// The target security level in bits.
    pub target_security_bits: u64,
}

/// Security parameters *deduced* from a [`FriConfig`].
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
#[derive(Clone, Debug)]
pub struct Fri {
    cfg: FriConfig,
    /// k = -log2(ρ), exact.
    log_inv_rate: u32,
    /// h = log2(trace_length), exact.
    log_trace: u32,
    /// Domain size after low-degree extension: D = trace_length << k, exact.
    domain_size: u32,
    /// Number of FRI folding rounds.
    num_rounds: usize,
    /// dimension of the (partially folded) code entering commit round i,
    /// i.e. trace_length / Π_{j<=i} folding_factors[j].
    round_dimensions: Vec<u32>,
    /// Domain size at which folding stops.
    early_stop_degree: u32,
    /// Security parameters.
    sec_params: FriSecurityParams,
    /// Gap-widening factor for the regime.
    alpha: f64,
}

impl Fri {
    pub fn new(cfg: FriConfig) -> Self {
        let mut fri = Self::validate(cfg);
        let (sec_params, alpha) = fri.solve();
        fri.sec_params = sec_params;
        fri.alpha = alpha;
        fri
    }

    /// Construct with an externally fixed schedule.
    pub fn with_security_params(cfg: FriConfig, sec_params: FriSecurityParams) -> Self {
        let mut fri = Self::validate(cfg);
        let m = fri.num_rounds;

        assert_eq!(sec_params.grinding_bits_folding.len(), m, "Expected one folding grinding vector per iteration");

        fri.sec_params = sec_params;
        fri
    }

    /// Structural validation shared by both constructors.
    fn validate(cfg: FriConfig) -> Self {
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
        assert!(log_trace + log_inv_rate < 32, "domain size overflowed u32");
        let domain_size = (cfg.trace_length as u32) << log_inv_rate;

        // Folding schedule validation, plus precomputation of per-round code dimensions.
        let mut n = domain_size;
        let mut dim = cfg.trace_length;
        let mut round_dimensions = Vec::with_capacity(num_rounds);
        for (round, &factor) in cfg.folding_factors.iter().enumerate() {
            assert!(factor.is_power_of_two(), "folding factor must be a power of two");
            assert!(
                n % factor == 0 && dim % factor == 0,
                "folding overflows domain: round {}, remaining {}, factor {}",
                round,
                n,
                factor
            );
            n /= factor;
            dim /= factor;
            round_dimensions.push(dim);
        }

        let empty = FriSecurityParams {
            n_queries: 0,
            grinding_bits_batching: 0,
            grinding_bits_folding: vec![0; num_rounds],
            grinding_bits_query: 0,
        };

        Self {
            cfg,
            log_inv_rate,
            log_trace,
            domain_size,
            num_rounds,
            round_dimensions,
            early_stop_degree: n,
            sec_params: empty,
            alpha: 0.0,
        }
    }

    /// Search for the smallest gap-widening `alpha` whose query schedule
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

    /// Compute the security parameters (query count, grinding split) for a given regime.
    fn compute_security_params(&self, regime: &dyn ProximityGapsRegime) -> FriSecurityParams {
        let target = self.cfg.target_security_bits;

        // Non-query stages: grinding fills whatever is missing to the target.
        let deficit = |error: f64| target.saturating_sub(bits_of_security_from_error(error) as u64) as u32;

        // Batching grinding.
        let grinding_bits_batching = if self.cfg.batch_size > 1 { deficit(self.batching_error(regime, 0)) } else { 0 };

        // Folding grinding.
        let grinding_bits_folding =
            (0..self.num_rounds).map(|round| deficit(self.commit_phase_error(round, regime, 0))).collect();

        // Security contributed by a single query.
        let single_query_error = self.query_phase_error(regime, 1, 0);
        let security_per_query = security_from_error(single_query_error);

        // Hash count per query
        let hash_per_query = self.query_num_hashes();

        // Find max efficient grinding: 2^g < hashesPerQuery => g < log2(hashesPerQuery).
        let max_efficient_grinding = hash_per_query.log2().floor() as u64;
        let grinding_bits_query = if self.cfg.use_max_grinding_bits_query {
            self.cfg.max_grinding_bits_query
        } else {
            max_efficient_grinding.min(self.cfg.max_grinding_bits_query)
        };

        // Compute the number of queries needed to reach the target security level.
        let bits_needed = target as f64 - grinding_bits_query as f64;
        let n_queries = if bits_needed > 0.0 {
            (bits_needed / security_per_query).ceil() as u64
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
                format!("fold(i={})", i),
                bits_of_security_from_error(self.commit_phase_error(i, regime, sec_params.grinding_bits_folding[i])),
            ));
        }

        // Query phase.
        bits.push((
            "query".to_string(),
            bits_of_security_from_error(self.query_phase_error(
                regime,
                sec_params.n_queries,
                sec_params.grinding_bits_query,
            )),
        ));

        bits
    }

    /// Error from the batching step.
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

    /// Error from round `round` of the commit phase.
    fn commit_phase_error(&self, round: usize, regime: &dyn ProximityGapsRegime, grinding_bits: u32) -> f64 {
        let rate = self.cfg.rate;
        let dimension = self.round_dimensions[round];
        let epsilon = regime.error_powers(&rate, dimension, self.cfg.folding_factors[round] as u64);
        apply_grinding(epsilon, grinding_bits)
    }

    /// Error from the FRI query phase.
    fn query_phase_error(&self, regime: &dyn ProximityGapsRegime, n_queries: u64, grinding_bits: u32) -> f64 {
        // query error: (1 − γ)^t, where γ is the proximity parameter and
        // t the number of queries.
        let pp = regime.proximity_parameter(&self.cfg.rate);
        let error = (1.0 - pp).powi(n_queries as i32);
        apply_grinding(error, grinding_bits)
    }

    /// The solved regime.
    pub fn regime(&self) -> Box<dyn ProximityGapsRegime> {
        self.cfg.regime.instantiate(self.cfg.field_size, self.alpha)
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

    pub fn domain_size(&self) -> u32 {
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
            ("FRI_early_stop_degree", self.early_stop_degree.to_string()),
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
        max_grinding_bits_query: u64,
    ) -> FriConfig {
        FriConfig {
            field_size: goldilocks_safe_extension_field_size(),
            trace_length,
            rate,
            batching: Batching::Powers,
            batch_size,
            folding_factors: folding_bits.iter().map(|&b| 1u32 << b).collect(),
            max_grinding_bits_query,
            use_max_grinding_bits_query: true,
            tree_arity: 4,
            target_security_bits: 128,
            regime: DecodingRegime::Jbr,
        }
    }

    #[test]
    fn test_main_params() {
        let fri = Fri::new(test_config(1 << 22, 0.5, 61, vec![3, 3, 3, 3, 3, 3], 16));
        assert_eq!(fri.alpha(), 0.0);

        let levels = fri.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 128);
        assert_eq!(level("fold(i=0)"), 134);
        assert_eq!(level("fold(i=1)"), 137);
        assert_eq!(level("fold(i=2)"), 140);
        assert_eq!(level("fold(i=3)"), 143);
        assert_eq!(level("fold(i=4)"), 146);
        assert_eq!(level("fold(i=5)"), 149);
        assert_eq!(level("query"), 128);
        assert_eq!(fri.total_security_bits(), 128);

        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 228, "nQueries mismatch: got {}", sec.n_queries);
        assert_eq!(sec.grinding_bits_query, 16);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![0; 6]);
    }

    #[test]
    fn test_dma_params() {
        let fri = Fri::new(test_config(1 << 21, 0.5, 46, vec![3, 3, 3, 3, 3, 2], 16));
        assert_eq!(fri.alpha(), 0.0);

        let levels = fri.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 129);
        assert_eq!(level("fold(i=0)"), 135);
        assert_eq!(level("fold(i=1)"), 138);
        assert_eq!(level("fold(i=2)"), 141);
        assert_eq!(level("fold(i=3)"), 144);
        assert_eq!(level("fold(i=4)"), 147);
        assert_eq!(level("fold(i=5)"), 150);
        assert_eq!(level("query"), 128);
        assert_eq!(fri.total_security_bits(), 128);

        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 228, "nQueries mismatch: got {}", sec.n_queries);
        assert_eq!(sec.grinding_bits_query, 16);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![0; 6]);
    }

    #[test]
    fn test_keccakf_params() {
        let fri = Fri::new(test_config(1 << 17, 0.5, 4065, vec![3, 3, 3, 3], 23));
        assert_eq!(fri.alpha(), 0.0);

        let levels = fri.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 128);
        assert_eq!(level("fold(i=0)"), 139);
        assert_eq!(level("fold(i=1)"), 142);
        assert_eq!(level("fold(i=2)"), 145);
        assert_eq!(level("fold(i=3)"), 148);
        assert_eq!(level("query"), 128);
        assert_eq!(fri.total_security_bits(), 128);

        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 213, "nQueries mismatch: got {}", sec.n_queries);
        assert_eq!(sec.grinding_bits_query, 23);
        assert_eq!(sec.grinding_bits_batching, 1);
        assert_eq!(sec.grinding_bits_folding, vec![0; 4]);
    }

    #[test]
    fn test_poseidon2_params() {
        let fri = Fri::new(test_config(1 << 17, 0.25, 182, vec![3, 3, 3, 3, 2], 16));
        assert_eq!(fri.alpha(), 0.0);

        let levels = fri.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 131);
        assert_eq!(level("fold(i=0)"), 139);
        assert_eq!(level("fold(i=1)"), 142);
        assert_eq!(level("fold(i=2)"), 145);
        assert_eq!(level("fold(i=3)"), 148);
        assert_eq!(level("fold(i=4)"), 151);
        assert_eq!(level("query"), 128);
        assert_eq!(fri.total_security_bits(), 128);

        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 114, "nQueries mismatch: got {}", sec.n_queries);
        assert_eq!(sec.grinding_bits_query, 16);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![0; 5]);
    }

    #[test]
    fn test_recursive2_params() {
        let fri = Fri::new(test_config(1 << 17, 0.125, 145, vec![3, 3, 3, 3, 3], 20));
        assert_eq!(fri.alpha(), 0.0);

        let levels = fri.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 132);
        assert_eq!(level("fold(i=0)"), 139);
        assert_eq!(level("fold(i=1)"), 142);
        assert_eq!(level("fold(i=2)"), 145);
        assert_eq!(level("fold(i=3)"), 148);
        assert_eq!(level("fold(i=4)"), 151);
        assert_eq!(level("query"), 128);
        assert_eq!(fri.total_security_bits(), 128);

        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 73, "nQueries mismatch: got {}", sec.n_queries);
        assert_eq!(sec.grinding_bits_query, 20);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![0; 5]);
    }

    #[test]
    fn test_final_params() {
        let fri = Fri::new(test_config(1 << 16, 0.03125, 139, vec![4, 4, 4, 4], 22));
        assert_eq!(fri.alpha(), 0.0);

        let levels = fri.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 133);
        assert_eq!(level("fold(i=0)"), 140);
        assert_eq!(level("fold(i=1)"), 144);
        assert_eq!(level("fold(i=2)"), 148);
        assert_eq!(level("fold(i=3)"), 152);
        assert_eq!(level("query"), 128);
        assert_eq!(fri.total_security_bits(), 128);

        let sec = fri.security_params();
        assert_eq!(sec.n_queries, 43, "nQueries mismatch: got {}", sec.n_queries);
        assert_eq!(sec.grinding_bits_query, 22);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![0; 4]);
    }
}
