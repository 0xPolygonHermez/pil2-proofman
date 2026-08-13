use super::super::regimes::{DecodingRegime, ProximityGapsRegime};
use super::types::{
    Batching, Pcs, apply_grinding, bits_of_security_from_error, coset_opening_hashes, merkle_opening_size_bits,
    security_from_error,
};

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
    /// Per-round folding factors `kᵢ`, in bits: round `i` folds by `2^kᵢ`.
    pub log_folding_factors: Vec<u32>,
    /// The maximum number of grinding bits allowed.
    pub max_grinding_bits_query: u64,
    /// Whether to use the maximum number of grinding bits.
    pub use_max_grinding_bits_query: bool,
    /// The arity of the Merkle tree used in FRI.
    pub tree_arity: u64,
    /// The output length of the Merkle-tree hash, in bits.
    pub hash_size_bits: u64,
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
    /// Number of FRI folding rounds.
    num_rounds: usize,
    /// `log2(2^{mᵢ})` — per-round log-dimensions (length `rounds + 1`):
    /// entry i is the log-dimension of the code entering round i.
    log_round_dimensions: Vec<u32>,
    /// Dimension (final polynomial length) at which folding stops.
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
        // `num_rounds == 0` is legal: the folding phase is empty and the prover
        // sends the whole (trace-degree) polynomial in clear, so the query phase
        // degenerates to a direct low-degree test against it. Small airs whose
        // 2^n_bits_ext is already at or below the final degree take this path.
        let num_rounds = cfg.log_folding_factors.len();

        // ρ = 2^-k.
        let k = (-cfg.rate.log2()).round();
        assert!((1.0..=32.0).contains(&k) && cfg.rate == f64::exp2(-k), "ρ must be an exact power of two");
        let log_inv_rate = k as u32;

        // trace_length = 2^h.
        assert!(cfg.trace_length.is_power_of_two(), "trace_length must be a power of two");
        let log_trace = cfg.trace_length.trailing_zeros();

        // Domain size n = trace_length / ρ = trace_length << k.
        assert!(log_trace + log_inv_rate < 32, "domain size overflowed u32");

        // Folding schedule validation, plus the per-round log-dimensions mᵢ.
        //
        // Recurrence:
        //   m_{i+1} = m_i - k_i   (folding by 2^{k_i}; the rate is unchanged)
        let mut log_round_dimensions = Vec::with_capacity(num_rounds + 1);
        log_round_dimensions.push(log_trace);
        for (round, &k_i) in cfg.log_folding_factors.iter().enumerate() {
            assert!(k_i >= 1, "Every folding factor must be >= 1");
            assert!(
                k_i <= log_round_dimensions[round],
                "folding overflows dimension: round {}, remaining 2^{}, folding 2^{}",
                round,
                log_round_dimensions[round],
                k_i
            );
            log_round_dimensions.push(log_round_dimensions[round] - k_i);
        }
        let early_stop_degree = 1u32 << log_round_dimensions[num_rounds];

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
            num_rounds,
            log_round_dimensions,
            early_stop_degree,
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
            (0..self.num_rounds).map(|round| deficit(self.fold_error(regime, round, 0))).collect();

        // Security contributed by a single query.
        let single_query_error = self.query_error(regime, 1, 0);
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

    /// Approximate verifier hash count per query: one coset opening in each
    /// round's tree.
    fn query_num_hashes(&self) -> f64 {
        if self.num_rounds == 0 {
            // No folding: the query still opens the initial oracle, as a single
            // leaf (coset size 2^0) on the full evaluation domain.
            return coset_opening_hashes(self.log_trace + self.log_inv_rate, 0, self.cfg.tree_arity);
        }
        (0..self.num_rounds)
            .map(|round| {
                let k = self.cfg.log_folding_factors[round];
                let log_domain = self.log_round_dimensions[round] + self.log_inv_rate;
                coset_opening_hashes(log_domain, k, self.cfg.tree_arity)
            })
            .sum()
    }

    /// Total Merkle openings in the query phase: each query opens one coset
    /// in every round's tree. With no folding rounds there is still the initial
    /// oracle to open, hence the `max(1)`.
    pub fn num_merkle_openings(&self) -> u64 {
        self.sec_params.n_queries * self.num_rounds.max(1) as u64
    }

    /// Approximate verifier hashes spent on the query phase.
    pub fn total_query_hashes(&self) -> f64 {
        self.sec_params.n_queries as f64 * self.query_num_hashes()
    }

    /// Estimated worst-case proof size in bits
    /// : one root per tree, per query one opening
    /// per tree, and the final polynomial in clear. The initial batched
    /// oracle is one tree whose leaves hold all `batch_size` values at a
    /// position (extension-field elements, conservatively).
    pub fn proof_size_bits(&self) -> u64 {
        let ext_bits = self.cfg.field_size.log2().round() as u64;
        let hash = self.cfg.hash_size_bits;
        let t = self.sec_params.n_queries as f64;
        let mut size = 0.0;

        // Initial round: root + per query one leaf with the batch_size functions.
        let log_domain = self.log_trace + self.log_inv_rate;
        size += hash as f64;
        size += t * merkle_opening_size_bits(
            f64::exp2(log_domain as f64),
            self.cfg.batch_size,
            ext_bits,
            self.cfg.tree_arity,
            hash,
        );

        // Folding rounds: root + per query one coset leaf of 2^{kᵢ} values.
        for (round, &k) in self.cfg.log_folding_factors.iter().enumerate() {
            let log_domain = self.log_round_dimensions[round] + self.log_inv_rate;
            let n_leafs = f64::exp2((log_domain - k) as f64);
            size += hash as f64;
            size += t * merkle_opening_size_bits(n_leafs, 1u64 << k, ext_bits, self.cfg.tree_arity, hash);
        }

        // Final polynomial in clear.
        size += (self.early_stop_degree as u64 * ext_bits) as f64;

        size.round() as u64
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

        // Batching step.
        if self.cfg.batch_size > 1 {
            let grinding_bits = sec_params.grinding_bits_batching;
            let error = self.batching_error(regime, grinding_bits);
            bits.push(("batching".to_string(), bits_of_security_from_error(error)));
        }

        // Folding step.
        for i in 0..self.num_rounds {
            let grinding_bits = sec_params.grinding_bits_folding[i];
            let error = self.fold_error(regime, i, grinding_bits);
            bits.push((format!("fold(i={})", i), bits_of_security_from_error(error)));
        }

        // Query step.
        let grinding_bits = sec_params.grinding_bits_query;
        let error = self.query_error(regime, sec_params.n_queries, grinding_bits);
        bits.push(("query".to_string(), bits_of_security_from_error(error)));

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

    /// Error from the folding step at round i.
    fn fold_error(&self, regime: &dyn ProximityGapsRegime, round: usize, grinding_bits: u32) -> f64 {
        let rate = self.cfg.rate;
        let dimension = 1u32 << self.log_round_dimensions[round + 1];
        let epsilon = regime.error_powers(&rate, dimension, 1u64 << self.cfg.log_folding_factors[round]);
        apply_grinding(epsilon, grinding_bits)
    }

    /// Error from the FRI query phase.
    fn query_error(&self, regime: &dyn ProximityGapsRegime, n_queries: u64, grinding_bits: u32) -> f64 {
        // query error: (1 − γ)^t, where γ is the proximity parameter and
        // t the number of queries.
        let pp = regime.proximity_parameter(&self.cfg.rate);
        let error = (1.0 - pp).powi(n_queries as i32);
        apply_grinding(error, grinding_bits)
    }

    pub fn config(&self) -> &FriConfig {
        &self.cfg
    }

    /// The solved regime.
    pub fn regime(&self) -> Box<dyn ProximityGapsRegime> {
        self.cfg.regime.instantiate(self.cfg.field_size, self.alpha)
    }

    /// The deduced security parameters (query count, grinding split).
    pub fn security_params(&self) -> &FriSecurityParams {
        &self.sec_params
    }

    /// Raise the query count above the security-optimal value, e.g. because a
    /// caller sizes a circuit by its query count. Never lowers it, so the
    /// solved security floor is preserved and soundness only ever strengthens.
    /// Returns whether the count actually changed.
    pub fn raise_n_queries(&mut self, n_queries: u64) -> bool {
        if n_queries > self.sec_params.n_queries {
            self.sec_params.n_queries = n_queries;
            return true;
        }
        false
    }

    /// The deduced gap-widening factor.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn proximity_gap(&self) -> f64 {
        self.regime().gap(&self.cfg.rate)
    }

    pub fn proximity_parameter(&self) -> f64 {
        self.regime().proximity_parameter(&self.cfg.rate)
    }

    /// Description of the parameters of the PCS (Markdown code block).
    pub fn parameter_summary(&self) -> String {
        let trace_length = "2^".to_string() + &self.log_trace.to_string();
        let rate = "1/2^".to_string() + &self.log_inv_rate.to_string() + " = " + &self.cfg.rate.to_string();
        let domain_size = "2^".to_string() + &(self.log_trace + self.log_inv_rate).to_string();
        let folding_factors = "[".to_string()
            + &self
                .cfg
                .log_folding_factors
                .iter()
                .map(|&k| "2^".to_string() + &k.to_string())
                .collect::<Vec<_>>()
                .join(", ")
            + "]";
        let early_stop_degree = "2^".to_string() + &(self.early_stop_degree as f64).log2().round().to_string();
        let params: Vec<(&str, String)> = vec![
            ("Target Security Bits", self.cfg.target_security_bits.to_string()),
            ("Regime", format!("{:?} (𝛼 = {})", self.cfg.regime, self.alpha)),
            ("Trace Length", trace_length),
            ("Rate", rate),
            ("Domain Size", domain_size),
            ("Batch Size", self.cfg.batch_size.to_string()),
            ("Batching", self.cfg.batching.to_string()),
            ("Rounds", self.num_rounds.to_string()),
            ("Folding Factors", folding_factors),
            ("Early Stop Degree", early_stop_degree),
            ("N Queries", self.sec_params.n_queries.to_string()),
            ("Grinding Bits Batching", self.sec_params.grinding_bits_batching.to_string()),
            ("Grinding Bits Folding", format!("{:?}", self.sec_params.grinding_bits_folding)),
            ("Grinding Bits Query", self.sec_params.grinding_bits_query.to_string()),
            (
                "N Merkle Openings",
                format!(
                    "{} per round x {} rounds = {}",
                    self.sec_params.n_queries,
                    self.num_rounds,
                    self.num_merkle_openings()
                ),
            ),
            ("Total Query Hashes (approx)", format!("{:.0}", self.total_query_hashes())),
            ("Proof Size (worst case)", format!("{:.0} KiB", self.proof_size_bits() as f64 / 8192.0)),
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

    fn parameter_summary(&self) -> String {
        Fri::parameter_summary(self)
    }

    fn num_merkle_openings(&self) -> u64 {
        Fri::num_merkle_openings(self)
    }

    fn total_query_hashes(&self) -> f64 {
        Fri::total_query_hashes(self)
    }

    fn proof_size_bits(&self) -> u64 {
        Fri::proof_size_bits(self)
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
            log_folding_factors: folding_bits,
            max_grinding_bits_query,
            use_max_grinding_bits_query: true,
            tree_arity: 4,
            hash_size_bits: 256,
            target_security_bits: 128,
            regime: DecodingRegime::Jbr,
        }
    }

    #[test]
    fn test_main_params() {
        let fri = Fri::new(test_config(1 << 22, 0.5, 61, vec![3, 3, 3, 3, 3, 3], 16));

        assert_eq!(fri.num_merkle_openings(), 1368);
        assert_eq!(fri.total_query_hashes().round(), 37_620.0);

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

        assert_eq!(fri.num_merkle_openings(), 1368);
        assert_eq!(fri.total_query_hashes().round(), 35_340.0);

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

        assert_eq!(fri.num_merkle_openings(), 852);
        assert_eq!(fri.total_query_hashes().round(), 20_874.0);

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

        assert_eq!(fri.num_merkle_openings(), 570);
        assert_eq!(fri.total_query_hashes().round(), 13_338.0);

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

        assert_eq!(fri.num_merkle_openings(), 365);
        assert_eq!(fri.total_query_hashes().round(), 9_271.0);

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

        assert_eq!(fri.num_merkle_openings(), 172);
        assert_eq!(fri.total_query_hashes().round(), 5_848.0);

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

    /// Tiny airs (e.g. the `Connection2`/`ConnectionNew` test airs, 2^4 rows with
    /// blowup 1) generate a starkStruct with a single step, so the folding phase
    /// is empty. The query phase then tests the codeword directly against the
    /// final polynomial, which is sent in clear.
    #[test]
    fn test_no_folding_rounds() {
        let fri = Fri::new(test_config(1 << 4, 0.5, 14, vec![], 20));

        let levels = fri.security_levels();
        assert!(levels.iter().all(|(k, _)| k != "fold(i=0)"), "no folding phase expected: {levels:?}");
        assert_eq!(levels.iter().find(|(k, _)| k == "query").unwrap().1, 128);
        assert_eq!(fri.total_security_bits(), 128);

        let sec = fri.security_params();
        assert!(sec.n_queries > 0);
        assert_eq!(sec.grinding_bits_query, 20);
        assert!(sec.grinding_bits_folding.is_empty());

        // Each query still opens the single committed oracle.
        assert_eq!(fri.num_merkle_openings(), sec.n_queries);
        assert!(fri.total_query_hashes() > 0.0);
        assert!(fri.proof_size_bits() > 0);
    }

    /// The nQueries override must survive into the reported security params and
    /// must never weaken the solved security floor.
    #[test]
    fn test_raise_n_queries() {
        let mut fri = Fri::new(test_config(1 << 16, 0.03125, 139, vec![4, 4, 4, 4], 22));
        let floor = fri.security_params().n_queries;

        assert!(!fri.raise_n_queries(floor - 1), "must not lower below the floor");
        assert_eq!(fri.security_params().n_queries, floor);

        assert!(fri.raise_n_queries(floor + 10));
        assert_eq!(fri.security_params().n_queries, floor + 10);
        assert!(fri.total_security_bits() >= 128);
    }
}
