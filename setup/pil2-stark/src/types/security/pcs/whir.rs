use crate::types::security::pcs::types::apply_grinding;

use super::super::regimes::{DecodingRegime, ProximityGapsRegime};
use super::types::{
    Batching, Pcs, bits_of_security_from_error, coset_opening_hashes, merkle_opening_size_bits, security_from_error,
};

/// Configuration for the WHIR PCS.
#[derive(Clone, Debug)]
pub struct WhirConfig {
    /// Field size |F|.
    pub field_size: f64,
    /// Domain size before low-degree extension. Must be 2^h.
    pub trace_length: u32,
    /// The code rate ρ. Must be an exact power of two 2^-k.
    pub rate: f64,
    /// Batching strategy (see [`Batching`]).
    pub batching: Batching,
    /// Number of constraints batched.
    pub batch_size: u64,
    /// Per-iteration folding factors `kᵢ`, in bits: iteration `i` folds
    /// `2^kᵢ` variables.
    pub log_folding_factors: Vec<u32>,
    /// Constraint degree `d = max(d*, 3)`, `d* = 1 + deg_Z(w̃) + max_i{deg_{X_i}(w̃)}`.
    pub constraint_degree: u64,
    /// The maximum number of grinding bits allowed.
    pub max_grinding_bits_query: u64,
    /// Whether to use the maximum number of grinding bits.
    pub use_max_grinding_bits_query: bool,
    /// The arity of the Merkle trees used in WHIR.
    pub tree_arity: u64,
    /// The output length of the Merkle-tree hash, in bits.
    pub hash_size_bits: u64,
    /// Base-field element size in bits. The initial oracle f₀ is over the
    /// base field; folded oracles, sumcheck polynomials and OOD replies take
    /// full extension elements (log2(field_size) bits).
    pub base_field_bits: u64,
    /// The target security level in bits.
    pub target_security_bits: u64,
    /// The decoding regime kind. The gap-widening factor `alpha` is deduced.
    pub regime: DecodingRegime,
}

/// Security parameters *deduced* from a [`WhirConfig`].
#[derive(Clone, Debug)]
pub struct WhirSecurityParams {
    /// Per-iteration query counts `tᵢ` (length `M`).
    pub num_queries: Vec<u64>,
    /// Per-iteration OOD sample counts `wᵢ` (length `M-1`).
    pub num_ood_samples: Vec<u64>,
    /// Grinding bits for the batching phase.
    pub grinding_bits_batching: u32,
    /// Grinding bits per folding round, `[iteration][round]` (length `M`, each `kᵢ`).
    pub grinding_bits_folding: Vec<Vec<u32>>,
    /// Grinding bits per iteration's query phase (length `M`).
    pub grinding_bits_queries: Vec<u32>,
    /// Grinding bits per iteration's OOD phase (length `M-1`).
    pub grinding_bits_ood: Vec<u32>,
}

/// WHIR Polynomial Commitment Scheme.
#[derive(Clone, Debug)]
pub struct Whir {
    cfg: WhirConfig,
    /// `log2(1/ρᵢ)` — per-iteration inverse rates (length `M+1`).
    log_inv_rates: Vec<u32>,
    /// h = log2(trace_length).
    log_trace: u32,
    /// Number of iterations `M`.
    num_rounds: usize,
    /// log2(2^{mᵢ})` — per-iteration log-dimensions (length `M+1`).
    log_round_dimensions: Vec<u32>,
    /// Domain size at which folding stops.
    early_stop_degree: u32,
    /// Security parameters.
    sec_params: WhirSecurityParams,
    /// Gap-widening factor for the regime.
    alpha: f64,
}

impl Whir {
    pub fn new(cfg: WhirConfig) -> Self {
        let mut whir = Self::validate(cfg);
        let (sec_params, alpha) = whir.solve();
        whir.sec_params = sec_params;
        whir.alpha = alpha;
        whir
    }

    /// Construct with an externally fixed schedule.
    pub fn with_security_params(cfg: WhirConfig, sec_params: WhirSecurityParams) -> Self {
        let mut whir = Self::validate(cfg);
        let m = whir.num_rounds;

        assert_eq!(sec_params.num_queries.len(), m, "Expected one query count per iteration");
        assert_eq!(sec_params.num_ood_samples.len(), m - 1, "Expected M-1 OOD sample counts");
        assert_eq!(sec_params.grinding_bits_queries.len(), m, "Expected one query grinding entry per iteration");
        assert_eq!(sec_params.grinding_bits_ood.len(), m - 1, "Expected M-1 OOD grinding entries");
        assert_eq!(sec_params.grinding_bits_folding.len(), m, "Expected one folding grinding vector per iteration");
        for (i, g) in sec_params.grinding_bits_folding.iter().enumerate() {
            // Each iteration has a sumcheck phase with exactly kᵢ rounds.
            assert_eq!(
                g.len(),
                whir.cfg.log_folding_factors[i] as usize,
                "Iteration {i}: expected one folding grinding entry per sumcheck round"
            );
        }

        whir.sec_params = sec_params;
        whir
    }

    /// Structural validation shared by both constructors.
    fn validate(cfg: WhirConfig) -> Self {
        let num_rounds = cfg.log_folding_factors.len();

        // ρ = 2^-k.
        let k = -cfg.rate.log2();
        assert!((1.0..=32.0).contains(&k) && cfg.rate == f64::exp2(-k), "ρ must be an exact power of two");
        let log_inv_rate = k as u32;

        // trace_length = 2^h.
        let log_trace = cfg.trace_length.trailing_zeros();
        assert!(
            (1..=32).contains(&log_trace) && cfg.trace_length == 1 << log_trace,
            "trace_length must be an exact power of two"
        );

        // Domain size n = trace_length / ρ = trace_length << k.
        assert!(log_trace + log_inv_rate < 32, "domain size overflowed u32");

        // d = max(d*, 3), where d* = 1 + deg_Z(w̃) + max_i{deg_{X_i}(w̃)}.
        assert!(cfg.constraint_degree >= 3, "Constraint degree must be >= 3");
        assert!(cfg.batch_size >= 1, "Batch size must be at least 1");
        assert!(num_rounds >= 1, "Must have at least 1 iteration");
        assert!(cfg.log_folding_factors.iter().all(|&k| k >= 1), "Every folding factor must be >= 1 to reduce degree");

        // Ensure the final polynomial does not end up with a negative number
        // of variables: m₀ >= Σᵢ kᵢ.
        let total_reduction: u32 = cfg.log_folding_factors.iter().sum();
        assert!(
            total_reduction <= log_trace,
            "Reducing {} variables by {total_reduction} (sum of folding factors) leaves a negative number of variables",
            log_trace,
        );

        // Compute the per-iteration log-degree m_i and log-inverse-rate μᵢ.
        //
        // Recurrence:
        //   m_{i+1}  = m_i - k_i        (folding by 2^{k_i})
        //   μ_{i+1}  = μ_i + (k_i - 1)  (domain halves; degree drops by 2^{k_i})
        let mut log_round_dimensions = Vec::with_capacity(num_rounds + 1);
        let mut log_inv_rates = Vec::with_capacity(num_rounds + 1);
        log_round_dimensions.push(log_trace);
        log_inv_rates.push(log_inv_rate);
        for i in 0..num_rounds {
            let k_i = cfg.log_folding_factors[i];
            log_round_dimensions.push(log_round_dimensions[i] - k_i);
            log_inv_rates.push(log_inv_rates[i] + (k_i - 1));
        }
        let early_stop_degree = 1 << log_round_dimensions[num_rounds];

        let empty = WhirSecurityParams {
            num_queries: vec![0; num_rounds],
            num_ood_samples: vec![0; num_rounds - 1],
            grinding_bits_batching: 0,
            grinding_bits_folding: cfg.log_folding_factors.iter().map(|&k| vec![0u32; k as usize]).collect(),
            grinding_bits_queries: vec![0; num_rounds],
            grinding_bits_ood: vec![0; num_rounds - 1],
        };

        Self {
            cfg,
            log_inv_rates,
            log_trace,
            num_rounds,
            log_round_dimensions,
            early_stop_degree,
            sec_params: empty,
            alpha: 0.0,
        }
    }

    /// Search for the smallest gap-widening `alpha` whose query schedule
    /// meets the target on every component.
    fn solve(&self) -> (WhirSecurityParams, f64) {
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

    /// Compute the security parameters for a given regime.
    fn compute_security_params(&self, regime: &dyn ProximityGapsRegime) -> WhirSecurityParams {
        let m = self.num_rounds;
        let target = self.cfg.target_security_bits;

        // Non-query stages: grinding fills whatever is missing to the target.
        let deficit = |error: f64| target.saturating_sub(bits_of_security_from_error(error) as u64) as u32;

        // Batching grinding
        let grinding_bits_batching = if self.cfg.batch_size > 1 { deficit(self.batch_error(regime, 0)) } else { 0 };

        // Folding grinding.
        let mut grinding_bits_folding = Vec::with_capacity(m);
        for i in 0..m {
            let len = self.cfg.log_folding_factors[i];
            let mut bits = vec![0u32; len as usize];
            for s in 1..=len {
                bits[(s - 1) as usize] = deficit(self.fold_error(regime, i, s, 0));
            }
            grinding_bits_folding.push(bits);
        }

        // OOD grinding.
        let mut grinding_bits_ood = vec![0u32; m - 1];
        for i in 1..m {
            grinding_bits_ood[i - 1] = deficit(self.ood_error(regime, i, 1, 0));
        }

        // Query grinding.
        let mut num_queries = Vec::with_capacity(m);
        let mut grinding_bits_queries = Vec::with_capacity(m);
        for i in 0..m {
            // Error is of the form (1 − δ_{i-1})^{t_{i-1}} + ℓ_{i,0}·(t_{i-1}+1)/|F|

            // Security contributed by (1 − δ_{i-1})^{t_{i-1}}
            let single_query_error = self.final_error(regime, 1, 0);
            let security_per_query = security_from_error(single_query_error);

            // Find max efficient grinding: 2^g < hashesPerQuery => g < log2(hashesPerQuery).
            let hash_per_query = self.query_num_hashes(i);
            let max_efficient_grinding = hash_per_query.log2().floor() as u64;
            let g = if self.cfg.use_max_grinding_bits_query {
                self.cfg.max_grinding_bits_query
            } else {
                max_efficient_grinding.min(self.cfg.max_grinding_bits_query)
            } as u32;

            // Compute the number of queries needed to reach the target security level.
            let needed_from_queries = target as f64 - g as f64;
            let mut t = if needed_from_queries > 0.0 {
                (needed_from_queries / security_per_query).ceil() as u64
            } else {
                1 // Need at least 1 query
            };

            // Now, adjust for ℓ_{i,0}·(t_{i-1}+1)/|F| when needed
            if i + 1 < m {
                let mut error = self.shift_error(regime, i + 1, t, g);
                while (bits_of_security_from_error(error) as u64) < target {
                    let next_error = self.shift_error(regime, i + 1, t + 1, g);
                    if next_error >= error {
                        break;
                    }
                    t += 1;
                    error = next_error;
                }
            }

            num_queries.push(t);
            grinding_bits_queries.push(g);
        }

        WhirSecurityParams {
            num_queries,
            num_ood_samples: vec![1; m - 1],
            grinding_bits_batching,
            grinding_bits_folding,
            grinding_bits_queries,
            grinding_bits_ood,
        }
    }

    /// Approximate verifier hash count per query: one coset opening in the
    /// iteration's tree.
    fn query_num_hashes(&self, iteration: usize) -> f64 {
        let k = self.cfg.log_folding_factors[iteration];
        let log_domain = self.log_round_dimensions[iteration] + self.log_inv_rates[iteration];
        coset_opening_hashes(log_domain, k, self.cfg.tree_arity)
    }

    /// Total Merkle openings in the query phase: each query opens one coset
    /// in its own iteration's tree only.
    pub fn num_merkle_openings(&self) -> u64 {
        self.sec_params.num_queries.iter().sum()
    }

    /// Approximate verifier hashes spent on the query phases (upper bound:
    /// shared path prefixes across queries are not deduplicated).
    pub fn total_query_hashes(&self) -> f64 {
        self.sec_params.num_queries.iter().enumerate().map(|(i, &t)| t as f64 * self.query_num_hashes(i)).sum()
    }

    /// Estimated worst-case proof size in bits: per-iteration roots, sumcheck polynomials and
    /// OOD replies, the decision-phase openings, and the final polynomial in
    /// clear.
    pub fn proof_size_bits(&self) -> u64 {
        let ext_bits = self.cfg.field_size.log2().round() as u64;
        let hash = self.cfg.hash_size_bits;
        let d = self.cfg.constraint_degree;
        let mut size = 0.0;

        // Initial commitment root.
        size += hash as f64;

        // Sumcheck: kᵢ univariate polynomials per iteration, each of degree
        // < d, sent as d − 1 evaluations (Gruen 2024 §3.1: the verifier
        // recovers h(1) from the running claim h(0) + h(1)).
        for i in 0..self.num_rounds {
            size += (self.cfg.log_folding_factors[i] as u64 * (d - 1) * ext_bits) as f64;
        }

        // Inner iterations: one root and the OOD replies each.
        for i in 1..self.num_rounds {
            size += hash as f64;
            size += (self.sec_params.num_ood_samples[i - 1] * ext_bits) as f64;
        }

        // Final polynomial in clear: 2^{m_M} coefficients.
        size += (self.early_stop_degree as u64 * ext_bits) as f64;

        // Decision phase: tᵢ coset openings in iteration i's tree. Iteration
        // 0's leaves hold all batch_size columns over the base field; later
        // iterations one extension-field coset of 2^{kᵢ} values.
        for i in 0..self.num_rounds {
            let k = self.cfg.log_folding_factors[i];
            let log_domain = self.log_round_dimensions[i] + self.log_inv_rates[i];
            let n_leafs = f64::exp2((log_domain - k) as f64);
            let (tuple_size, element_bits) = if i == 0 {
                ((1u64 << k) * self.cfg.batch_size, self.cfg.base_field_bits)
            } else {
                (1u64 << k, ext_bits)
            };
            size += self.sec_params.num_queries[i] as f64
                * merkle_opening_size_bits(n_leafs, tuple_size, element_bits, self.cfg.tree_arity, hash);
        }

        size.round() as u64
    }

    /// Whether every component meets the configured target.
    fn meets_security_target(&self, regime: &dyn ProximityGapsRegime, sec_params: &WhirSecurityParams) -> bool {
        self.security_levels_with(regime, sec_params)
            .into_iter()
            .all(|(_, bits)| bits as u64 >= self.cfg.target_security_bits)
    }

    /// Per-component PCS security levels (bits), keyed by component name.
    pub fn security_levels(&self) -> Vec<(String, u32)> {
        let regime = self.regime();
        self.security_levels_with(regime.as_ref(), &self.sec_params)
    }

    fn security_levels_with(&self, regime: &dyn ProximityGapsRegime, sec: &WhirSecurityParams) -> Vec<(String, u32)> {
        let m = self.num_rounds;
        let mut out = Vec::new();

        // Batching step.
        if self.cfg.batch_size > 1 {
            let grinding_bits = sec.grinding_bits_batching;
            let batch_error = self.batch_error(regime, grinding_bits);
            out.push(("batching".to_string(), bits_of_security_from_error(batch_error)));
        }

        // Initial iteration (i=0): only folding (sumcheck).
        for s in 1..=self.cfg.log_folding_factors[0] {
            let grinding_bits = sec.grinding_bits_folding[0][(s - 1) as usize];
            let error = self.fold_error(regime, 0, s, grinding_bits);
            out.push((format!("fold(i=0,s={s})"), bits_of_security_from_error(error)));
        }

        // Main loop (i=1,...,M-1): OOD, shift, and folding errors.
        for i in 1..m {
            // OOD error
            let n_samples = sec.num_ood_samples[i - 1];
            let grinding_bits = sec.grinding_bits_ood[i - 1];
            let ood_error = self.ood_error(regime, i, n_samples, grinding_bits);
            out.push((format!("ood(i={i})"), bits_of_security_from_error(ood_error)));

            // Shift error.
            let n_queries = sec.num_queries[i - 1];
            let grinding_bits = sec.grinding_bits_queries[i - 1];
            let shift_error = self.shift_error(regime, i, n_queries, grinding_bits);
            out.push((format!("shift(i={i})"), bits_of_security_from_error(shift_error)));

            // SumCheck folding errors.
            for s in 1..=self.cfg.log_folding_factors[i] {
                let grinding_bits = sec.grinding_bits_folding[i][(s - 1) as usize];
                let error = self.fold_error(regime, i, s, grinding_bits);
                out.push((format!("fold(i={i},s={s})"), bits_of_security_from_error(error)));
            }
        }

        // Final error.
        let n_queries = sec.num_queries[m - 1];
        let grinding_bits = sec.grinding_bits_queries[m - 1];
        let final_error = self.final_error(regime, n_queries, grinding_bits);
        out.push(("final".to_string(), bits_of_security_from_error(final_error)));

        out
    }

    /// Error from the batching step.
    fn batch_error(&self, regime: &dyn ProximityGapsRegime, grinding_bits: u32) -> f64 {
        let (rate, dimension) = self.code_for(0, 0);
        let epsilon = match self.cfg.batching {
            Batching::Powers => regime.error_powers(&rate, dimension, self.cfg.batch_size),
            Batching::Multilinear => regime.error_multilinear(&rate, dimension, self.cfg.batch_size),
            Batching::Affine => regime.error_linear(&rate, dimension),
        };
        apply_grinding(epsilon, grinding_bits)
    }

    /// Error from the folding step at iteration i, round s.
    fn fold_error(&self, regime: &dyn ProximityGapsRegime, iteration: usize, round: u32, grinding_bits: u32) -> f64 {
        assert!(iteration < self.num_rounds, "Iteration index out of bounds");
        let ff = self.cfg.log_folding_factors[iteration];
        assert!(
            round >= 1 && round <= ff,
            "Round index out of bounds for iteration {iteration}: got {round}, expected 1..={ff}"
        );

        // The error has two terms: ε^fold_{i,s} = d·ℓ_{i,s-1}/|F| + err_powers(C^{i,s}, 2)

        // The first term is d·ℓ_{i,s-1}/|F|, where ℓ_{i,s-1} is the max list
        // size at the previous round's code.
        let list_size = self.list_size_for(regime, iteration, round - 1);
        // let first = (self.cfg.constraint_degree as f64 * list_size).log2() - self.cfg.field_size.log2();
        let first = self.cfg.constraint_degree as f64 * list_size / self.cfg.field_size;

        // The second term is the batching error for powers coefficients at
        // the current round's code: folding one variable combines 2 halves.
        let (rate, dimension) = self.code_for(iteration, round);
        let second = regime.error_powers(&rate, dimension, 2);

        let epsilon = first + second;
        apply_grinding(epsilon, grinding_bits)
    }

    /// Error from the OOD step at iteration i.
    fn ood_error(&self, regime: &dyn ProximityGapsRegime, iteration: usize, n_samples: u64, grinding_bits: u32) -> f64 {
        assert!(iteration >= 1 && iteration < self.num_rounds, "Iteration index out of bounds");

        // ε^out_i = ℓ_{i,0}² · (2^{mᵢ} / (2|F|))^w.
        let w = n_samples;
        let list_size = self.list_size_for(regime, iteration, 0);
        let numerator = 2.0f64.powi(self.log_round_dimensions[iteration] as i32);
        let denominator = 2.0 * self.cfg.field_size;

        let epsilon = list_size * list_size * (numerator / denominator).powi(w as i32);
        apply_grinding(epsilon, grinding_bits)
    }

    /// Error from the shift step at iteration i.
    fn shift_error(
        &self,
        regime: &dyn ProximityGapsRegime,
        iteration: usize,
        n_queries: u64,
        grinding_bits: u32,
    ) -> f64 {
        assert!(iteration >= 1 && iteration < self.num_rounds, "Iteration index out of bounds");

        // ε^shift_i = (1 − δ_{i-1})^{t_{i-1}} + ℓ_{i,0}·(t_{i-1}+1)/|F|
        let t = n_queries;

        // First term is (1 − δ_{i-1})^{t_{i-1}}
        let pp = self.proximity_parameter_for(regime, iteration - 1);
        let first = (1.0 - pp).powi(t as i32);

        // Second term is ℓ_{i,0}·(t_{i-1}+1)/|F|
        let list_size = self.list_size_for(regime, iteration, 0);
        let second = list_size * (t + 1) as f64 / self.cfg.field_size;

        // The dominant first term is the previous iteration's query error,
        // so the previous iteration's query grinding is what strengthens it.
        let epsilon = first + second;
        apply_grinding(epsilon, grinding_bits)
    }

    /// Error from the final step at iteration i.
    fn final_error(&self, regime: &dyn ProximityGapsRegime, n_queries: u64, grinding_bits: u32) -> f64 {
        // (1 − δ)^t
        let t = n_queries;
        let pp = self.proximity_parameter_for(regime, self.num_rounds - 1);
        let epsilon = (1.0 - pp).powi(t as i32);
        apply_grinding(epsilon, grinding_bits)
    }

    /// The code `C_RS^{i,s} = RS[F, L_i^{(2^s)}, m_i − s]` as a
    /// `(rate, dimension)` pair: rate `2^{-μᵢ}`, dimension `2^{mᵢ−s}`.
    fn code_for(&self, iteration: usize, round: u32) -> (f64, u32) {
        assert!(iteration < self.num_rounds, "Iteration index out of bounds");
        assert!(
            round <= self.cfg.log_folding_factors[iteration],
            "Round index out of bounds for iteration {iteration}: got {round}, expected 0..={}",
            self.cfg.log_folding_factors[iteration]
        );

        let log_dimension = self.log_round_dimensions[iteration] as i64 - round as i64;
        assert!(log_dimension >= 0, "Log dimension cannot be negative");
        let rate = f64::exp2(-(self.log_inv_rates[iteration] as f64));
        (rate, 1u32 << log_dimension)
    }

    fn list_size_for(&self, regime: &dyn ProximityGapsRegime, iteration: usize, round: u32) -> f64 {
        let (rate, dimension) = self.code_for(iteration, round);
        regime.max_list_size(&rate, dimension) as f64
    }

    /// The proximity parameter δᵢ: the minimum over iteration i's rounds.
    fn proximity_parameter_for(&self, regime: &dyn ProximityGapsRegime, iteration: usize) -> f64 {
        assert!(iteration < self.num_rounds, "Iteration index out of bounds");

        let mut min_pp = 1.0f64;
        for s in 0..=self.cfg.log_folding_factors[iteration] {
            let (rate, _) = self.code_for(iteration, s);
            min_pp = min_pp.min(regime.proximity_parameter(&rate));
        }
        min_pp
    }

    pub fn config(&self) -> &WhirConfig {
        &self.cfg
    }

    /// The solved regime.
    pub fn regime(&self) -> Box<dyn ProximityGapsRegime> {
        self.cfg.regime.instantiate(self.cfg.field_size, self.alpha)
    }

    /// The deduced (or pinned) security parameters.
    pub fn security_params(&self) -> &WhirSecurityParams {
        &self.sec_params
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
        let rate = "1/2^".to_string() + &self.log_inv_rates[0].to_string() + " = " + &self.cfg.rate.to_string();
        let domain_size = "2^".to_string() + &(self.log_trace + self.log_inv_rates[0]).to_string();
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
            ("N Queries", format!("{:?}", self.sec_params.num_queries)),
            ("N OOD Samples", format!("{:?}", self.sec_params.num_ood_samples)),
            ("Grinding Bits Batching", self.sec_params.grinding_bits_batching.to_string()),
            ("Grinding Bits Folding", format!("{:?}", self.sec_params.grinding_bits_folding)),
            ("Grinding Bits OOD", format!("{:?}", self.sec_params.grinding_bits_ood)),
            ("Grinding Bits Queries", format!("{:?}", self.sec_params.grinding_bits_queries)),
            (
                "N Merkle Openings",
                format!(
                    "{} = {}",
                    self.sec_params.num_queries.iter().map(u64::to_string).collect::<Vec<_>>().join(" + "),
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

/// Security contributed by a single WHIR query at rate `2^-log_inv_rate`
/// under a regime with no gap widening (`alpha = 0`). Fractional — do not
/// floor. Used to size uniform query schedules externally (see `ml_params`).
pub fn whir_security_per_query(field_size: f64, log_inv_rate: u32, regime: DecodingRegime) -> f64 {
    let r = regime.instantiate(field_size, 0.0);
    let rate = f64::exp2(-(log_inv_rate as f64));
    -(1.0 - r.proximity_parameter(&rate)).log2()
}

impl Pcs for Whir {
    fn identifier(&self) -> &'static str {
        "WHIR"
    }

    fn security_levels(&self) -> Vec<(String, u32)> {
        Whir::security_levels(self)
    }

    fn parameter_summary(&self) -> String {
        Whir::parameter_summary(self)
    }

    fn num_merkle_openings(&self) -> u64 {
        Whir::num_merkle_openings(self)
    }

    fn total_query_hashes(&self) -> f64 {
        Whir::total_query_hashes(self)
    }

    fn proof_size_bits(&self) -> u64 {
        Whir::proof_size_bits(self)
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
        folding_factors: Vec<u32>,
        max_grinding_bits_query: u64,
    ) -> WhirConfig {
        WhirConfig {
            field_size: goldilocks_safe_extension_field_size(),
            trace_length,
            rate,
            batching: Batching::Powers,
            batch_size,
            log_folding_factors: folding_factors,
            // w̃(Z,X) = Z·eq(X,z)
            // d* = 1 + deg_Z(w̃) + max_i{deg_{X_i}(w̃)} = 1 + 1 + 1 = 3
            constraint_degree: 3,
            max_grinding_bits_query,
            use_max_grinding_bits_query: true,
            tree_arity: 4,
            hash_size_bits: 256,
            base_field_bits: 64,
            target_security_bits: 128,
            regime: DecodingRegime::Jbr,
        }
    }

    #[test]
    fn test_main_params() {
        let whir = Whir::new(test_config(1 << 22, 0.5, 61, vec![3, 3, 3, 3, 3, 3], 16));

        assert_eq!(whir.num_merkle_openings(), 431);
        assert_eq!(whir.total_query_hashes().round(), 15_853.0);

        assert_eq!(whir.alpha(), 0.0);

        let levels = whir.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 128);
        assert_eq!(level("fold(i=0,s=1)"), 135);
        assert_eq!(level("fold(i=0,s=2)"), 136);
        assert_eq!(level("fold(i=0,s=3)"), 137);
        assert_eq!(level("ood(i=1)"), 156);
        assert_eq!(level("shift(i=1)"), 128);
        assert_eq!(level("fold(i=1,s=1)"), 138);
        assert_eq!(level("fold(i=1,s=2)"), 139);
        assert_eq!(level("fold(i=1,s=3)"), 140);
        assert_eq!(level("ood(i=2)"), 157);
        assert_eq!(level("shift(i=2)"), 128);
        assert_eq!(level("fold(i=2,s=1)"), 141);
        assert_eq!(level("fold(i=2,s=2)"), 142);
        assert_eq!(level("fold(i=2,s=3)"), 143);
        assert_eq!(level("ood(i=3)"), 158);
        assert_eq!(level("shift(i=3)"), 129);
        assert_eq!(level("fold(i=3,s=1)"), 143);
        assert_eq!(level("fold(i=3,s=2)"), 144);
        assert_eq!(level("fold(i=3,s=3)"), 145);
        assert_eq!(level("ood(i=4)"), 159);
        assert_eq!(level("shift(i=4)"), 129);
        assert_eq!(level("fold(i=4,s=1)"), 146);
        assert_eq!(level("fold(i=4,s=2)"), 147);
        assert_eq!(level("fold(i=4,s=3)"), 148);
        assert_eq!(level("ood(i=5)"), 160);
        assert_eq!(level("shift(i=5)"), 130);
        assert_eq!(level("fold(i=5,s=1)"), 148);
        assert_eq!(level("fold(i=5,s=2)"), 149);
        assert_eq!(level("fold(i=5,s=3)"), 150);
        assert_eq!(level("final"), 132);
        assert_eq!(whir.total_security_bits(), 128);

        let sec = whir.security_params();
        assert_eq!(
            sec.grinding_bits_batching, 0,
            "grinding_bits_batching mismatch: got {}",
            sec.grinding_bits_batching
        );

        let num_queries = &sec.num_queries;
        assert!(num_queries.len() == 6, "Expected 6 iterations, got {}", num_queries.len());
        assert_eq!(num_queries[0], 228, "nQueries mismatch: got {}", num_queries[0]);
        assert_eq!(num_queries[1], 76, "nQueries mismatch: got {}", num_queries[1]);
        assert_eq!(num_queries[2], 46, "nQueries mismatch: got {}", num_queries[2]);
        assert_eq!(num_queries[3], 33, "nQueries mismatch: got {}", num_queries[3]);
        assert_eq!(num_queries[4], 26, "nQueries mismatch: got {}", num_queries[4]);
        assert_eq!(num_queries[5], 22, "nQueries mismatch: got {}", num_queries[5]);

        let num_ood_samples = &sec.num_ood_samples;
        assert!(num_ood_samples.len() == 5, "Expected 5 iterations, got {}", num_ood_samples.len());
        assert_eq!(num_ood_samples[0], 1, "nOOD mismatch: got {}", num_ood_samples[0]);
        assert_eq!(num_ood_samples[1], 1, "nOOD mismatch: got {}", num_ood_samples[1]);
        assert_eq!(num_ood_samples[2], 1, "nOOD mismatch: got {}", num_ood_samples[2]);
        assert_eq!(num_ood_samples[3], 1, "nOOD mismatch: got {}", num_ood_samples[3]);
        assert_eq!(num_ood_samples[4], 1, "nOOD mismatch: got {}", num_ood_samples[4]);

        let grinding_bits_folding = &sec.grinding_bits_folding;
        assert!(grinding_bits_folding.len() == 6, "Expected 6 iterations, got {}", grinding_bits_folding.len());
        for (i, g) in grinding_bits_folding.iter().enumerate() {
            assert!(g.len() == 3, "Expected 3 folding rounds for iteration {i}, got {}", g.len());
            assert_eq!(g[0], 0, "grinding_bits_folding mismatch for iteration {i}, round 1: got {}", g[0]);
            assert_eq!(g[1], 0, "grinding_bits_folding mismatch for iteration {i}, round 2: got {}", g[1]);
            assert_eq!(g[2], 0, "grinding_bits_folding mismatch for iteration {i}, round 3: got {}", g[2]);
        }

        let grinding_bits_queries = &sec.grinding_bits_queries;
        assert!(grinding_bits_queries.len() == 6, "Expected 6 iterations, got {}", grinding_bits_queries.len());
        assert_eq!(
            grinding_bits_queries[0], 16,
            "grinding_bits_queries mismatch for iteration 0: got {}",
            grinding_bits_queries[0]
        );
        assert_eq!(
            grinding_bits_queries[1], 16,
            "grinding_bits_queries mismatch for iteration 1: got {}",
            grinding_bits_queries[1]
        );
        assert_eq!(
            grinding_bits_queries[2], 16,
            "grinding_bits_queries mismatch for iteration 2: got {}",
            grinding_bits_queries[2]
        );
        assert_eq!(
            grinding_bits_queries[3], 16,
            "grinding_bits_queries mismatch for iteration 3: got {}",
            grinding_bits_queries[3]
        );
        assert_eq!(
            grinding_bits_queries[4], 16,
            "grinding_bits_queries mismatch for iteration 4: got {}",
            grinding_bits_queries[4]
        );
        assert_eq!(
            grinding_bits_queries[5], 16,
            "grinding_bits_queries mismatch for iteration 5: got {}",
            grinding_bits_queries[5]
        );

        let grinding_bits_ood = &sec.grinding_bits_ood;
        assert!(grinding_bits_ood.len() == 5, "Expected 5 iterations, got {}", grinding_bits_ood.len());
        assert_eq!(grinding_bits_ood[0], 0, "grinding_bits_ood mismatch for iteration 1: got {}", grinding_bits_ood[0]);
        assert_eq!(grinding_bits_ood[1], 0, "grinding_bits_ood mismatch for iteration 2: got {}", grinding_bits_ood[1]);
        assert_eq!(grinding_bits_ood[2], 0, "grinding_bits_ood mismatch for iteration 3: got {}", grinding_bits_ood[2]);
        assert_eq!(grinding_bits_ood[3], 0, "grinding_bits_ood mismatch for iteration 4: got {}", grinding_bits_ood[3]);
        assert_eq!(grinding_bits_ood[4], 0, "grinding_bits_ood mismatch for iteration 5: got {}", grinding_bits_ood[4]);
    }

    #[test]
    fn test_dma_params() {
        let whir = Whir::new(test_config(1 << 21, 0.5, 46, vec![3, 3, 3, 3, 3, 2], 16));

        assert_eq!(whir.num_merkle_openings(), 431);
        assert_eq!(whir.total_query_hashes().round(), 15_438.0);

        assert_eq!(whir.alpha(), 0.0);

        let levels = whir.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 129);
        assert_eq!(level("fold(i=0,s=1)"), 136);
        assert_eq!(level("fold(i=0,s=2)"), 137);
        assert_eq!(level("fold(i=0,s=3)"), 138);
        assert_eq!(level("ood(i=1)"), 157);
        assert_eq!(level("shift(i=1)"), 128);
        assert_eq!(level("fold(i=1,s=1)"), 139);
        assert_eq!(level("fold(i=1,s=2)"), 140);
        assert_eq!(level("fold(i=1,s=3)"), 141);
        assert_eq!(level("ood(i=2)"), 158);
        assert_eq!(level("shift(i=2)"), 128);
        assert_eq!(level("fold(i=2,s=1)"), 142);
        assert_eq!(level("fold(i=2,s=2)"), 143);
        assert_eq!(level("fold(i=2,s=3)"), 144);
        assert_eq!(level("ood(i=3)"), 159);
        assert_eq!(level("shift(i=3)"), 129);
        assert_eq!(level("fold(i=3,s=1)"), 144);
        assert_eq!(level("fold(i=3,s=2)"), 145);
        assert_eq!(level("fold(i=3,s=3)"), 146);
        assert_eq!(level("ood(i=4)"), 160);
        assert_eq!(level("shift(i=4)"), 129);
        assert_eq!(level("fold(i=4,s=1)"), 147);
        assert_eq!(level("fold(i=4,s=2)"), 148);
        assert_eq!(level("fold(i=4,s=3)"), 149);
        assert_eq!(level("ood(i=5)"), 161);
        assert_eq!(level("shift(i=5)"), 130);
        assert_eq!(level("fold(i=5,s=1)"), 149);
        assert_eq!(level("fold(i=5,s=2)"), 150);
        assert_eq!(level("final"), 132);
        assert_eq!(whir.total_security_bits(), 128);

        let sec = whir.security_params();
        assert_eq!(
            sec.grinding_bits_batching, 0,
            "grinding_bits_batching mismatch: got {}",
            sec.grinding_bits_batching
        );

        let num_queries = &sec.num_queries;
        assert!(num_queries.len() == 6, "Expected 6 iterations, got {}", num_queries.len());
        assert_eq!(num_queries[0], 228, "nQueries mismatch: got {}", num_queries[0]);
        assert_eq!(num_queries[1], 76, "nQueries mismatch: got {}", num_queries[1]);
        assert_eq!(num_queries[2], 46, "nQueries mismatch: got {}", num_queries[2]);
        assert_eq!(num_queries[3], 33, "nQueries mismatch: got {}", num_queries[3]);
        assert_eq!(num_queries[4], 26, "nQueries mismatch: got {}", num_queries[4]);
        assert_eq!(num_queries[5], 22, "nQueries mismatch: got {}", num_queries[5]);

        let num_ood_samples = &sec.num_ood_samples;
        assert!(num_ood_samples.len() == 5, "Expected 5 iterations, got {}", num_ood_samples.len());
        assert_eq!(num_ood_samples[0], 1, "nOOD mismatch: got {}", num_ood_samples[0]);
        assert_eq!(num_ood_samples[1], 1, "nOOD mismatch: got {}", num_ood_samples[1]);
        assert_eq!(num_ood_samples[2], 1, "nOOD mismatch: got {}", num_ood_samples[2]);
        assert_eq!(num_ood_samples[3], 1, "nOOD mismatch: got {}", num_ood_samples[3]);
        assert_eq!(num_ood_samples[4], 1, "nOOD mismatch: got {}", num_ood_samples[4]);

        let grinding_bits_folding = &sec.grinding_bits_folding;
        assert!(grinding_bits_folding.len() == 6, "Expected 6 iterations, got {}", grinding_bits_folding.len());
        for (i, g) in grinding_bits_folding.iter().enumerate() {
            for (s, bits) in g.iter().enumerate() {
                assert_eq!(*bits, 0, "grinding_bits_folding mismatch for iteration {i}, round {}: got {bits}", s + 1);
            }
        }

        let grinding_bits_queries = &sec.grinding_bits_queries;
        assert!(grinding_bits_queries.len() == 6, "Expected 6 iterations, got {}", grinding_bits_queries.len());
        assert_eq!(
            grinding_bits_queries[0], 16,
            "grinding_bits_queries mismatch for iteration 0: got {}",
            grinding_bits_queries[0]
        );
        assert_eq!(
            grinding_bits_queries[1], 16,
            "grinding_bits_queries mismatch for iteration 1: got {}",
            grinding_bits_queries[1]
        );
        assert_eq!(
            grinding_bits_queries[2], 16,
            "grinding_bits_queries mismatch for iteration 2: got {}",
            grinding_bits_queries[2]
        );
        assert_eq!(
            grinding_bits_queries[3], 16,
            "grinding_bits_queries mismatch for iteration 3: got {}",
            grinding_bits_queries[3]
        );
        assert_eq!(
            grinding_bits_queries[4], 16,
            "grinding_bits_queries mismatch for iteration 4: got {}",
            grinding_bits_queries[4]
        );
        assert_eq!(
            grinding_bits_queries[5], 16,
            "grinding_bits_queries mismatch for iteration 5: got {}",
            grinding_bits_queries[5]
        );

        let grinding_bits_ood = &sec.grinding_bits_ood;
        assert!(grinding_bits_ood.len() == 5, "Expected 5 iterations, got {}", grinding_bits_ood.len());
        assert_eq!(grinding_bits_ood[0], 0, "grinding_bits_ood mismatch for iteration 1: got {}", grinding_bits_ood[0]);
        assert_eq!(grinding_bits_ood[1], 0, "grinding_bits_ood mismatch for iteration 2: got {}", grinding_bits_ood[1]);
        assert_eq!(grinding_bits_ood[2], 0, "grinding_bits_ood mismatch for iteration 3: got {}", grinding_bits_ood[2]);
        assert_eq!(grinding_bits_ood[3], 0, "grinding_bits_ood mismatch for iteration 4: got {}", grinding_bits_ood[3]);
        assert_eq!(grinding_bits_ood[4], 0, "grinding_bits_ood mismatch for iteration 5: got {}", grinding_bits_ood[4]);
    }

    #[test]
    fn test_keccakf_params() {
        let whir = Whir::new(test_config(1 << 17, 0.5, 4065, vec![3, 3, 3, 3], 23));

        assert_eq!(whir.num_merkle_openings(), 358);
        assert_eq!(whir.total_query_hashes().round(), 10_928.0);

        assert_eq!(whir.alpha(), 0.0);

        let levels = whir.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 128);
        assert_eq!(level("fold(i=0,s=1)"), 140);
        assert_eq!(level("fold(i=0,s=2)"), 141);
        assert_eq!(level("fold(i=0,s=3)"), 142);
        assert_eq!(level("ood(i=1)"), 161);
        assert_eq!(level("shift(i=1)"), 128);
        assert_eq!(level("fold(i=1,s=1)"), 143);
        assert_eq!(level("fold(i=1,s=2)"), 144);
        assert_eq!(level("fold(i=1,s=3)"), 145);
        assert_eq!(level("ood(i=2)"), 162);
        assert_eq!(level("shift(i=2)"), 128);
        assert_eq!(level("fold(i=2,s=1)"), 146);
        assert_eq!(level("fold(i=2,s=2)"), 147);
        assert_eq!(level("fold(i=2,s=3)"), 148);
        assert_eq!(level("ood(i=3)"), 163);
        assert_eq!(level("shift(i=3)"), 129);
        assert_eq!(level("fold(i=3,s=1)"), 148);
        assert_eq!(level("fold(i=3,s=2)"), 149);
        assert_eq!(level("fold(i=3,s=3)"), 150);
        assert_eq!(level("final"), 129);
        assert_eq!(whir.total_security_bits(), 128);

        let sec = whir.security_params();
        assert_eq!(sec.num_queries, vec![213, 71, 43, 31]);
        assert_eq!(sec.num_ood_samples, vec![1; 3]);
        assert_eq!(sec.grinding_bits_batching, 1, "batching deficit should be 1 bit");
        assert_eq!(sec.grinding_bits_folding, vec![vec![0; 3]; 4]);
        assert_eq!(sec.grinding_bits_queries, vec![23; 4]);
        assert_eq!(sec.grinding_bits_ood, vec![0; 3]);
    }

    #[test]
    fn test_poseidon2_params() {
        let whir = Whir::new(test_config(1 << 17, 0.25, 182, vec![3, 3, 3, 3, 2], 16));

        assert_eq!(whir.num_merkle_openings(), 262);
        assert_eq!(whir.total_query_hashes().round(), 8_015.0);

        assert_eq!(whir.alpha(), 0.0);

        let levels = whir.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 131);
        assert_eq!(level("fold(i=0,s=1)"), 140);
        assert_eq!(level("fold(i=0,s=2)"), 141);
        assert_eq!(level("fold(i=0,s=3)"), 142);
        assert_eq!(level("ood(i=1)"), 160);
        assert_eq!(level("shift(i=1)"), 128);
        assert_eq!(level("fold(i=1,s=1)"), 143);
        assert_eq!(level("fold(i=1,s=2)"), 144);
        assert_eq!(level("fold(i=1,s=3)"), 145);
        assert_eq!(level("ood(i=2)"), 161);
        assert_eq!(level("shift(i=2)"), 128);
        assert_eq!(level("fold(i=2,s=1)"), 146);
        assert_eq!(level("fold(i=2,s=2)"), 147);
        assert_eq!(level("fold(i=2,s=3)"), 148);
        assert_eq!(level("ood(i=3)"), 162);
        assert_eq!(level("shift(i=3)"), 128);
        assert_eq!(level("fold(i=3,s=1)"), 148);
        assert_eq!(level("fold(i=3,s=2)"), 149);
        assert_eq!(level("fold(i=3,s=3)"), 150);
        assert_eq!(level("ood(i=4)"), 163);
        assert_eq!(level("shift(i=4)"), 129);
        assert_eq!(level("fold(i=4,s=1)"), 151);
        assert_eq!(level("fold(i=4,s=2)"), 152);
        assert_eq!(level("final"), 132);
        assert_eq!(whir.total_security_bits(), 128);

        let sec = whir.security_params();
        assert_eq!(
            sec.grinding_bits_batching, 0,
            "grinding_bits_batching mismatch: got {}",
            sec.grinding_bits_batching
        );

        let num_queries = &sec.num_queries;
        assert!(num_queries.len() == 5, "Expected 5 iterations, got {}", num_queries.len());
        assert_eq!(num_queries[0], 114, "nQueries mismatch: got {}", num_queries[0]);
        assert_eq!(num_queries[1], 57, "nQueries mismatch: got {}", num_queries[1]);
        assert_eq!(num_queries[2], 38, "nQueries mismatch: got {}", num_queries[2]);
        assert_eq!(num_queries[3], 29, "nQueries mismatch: got {}", num_queries[3]);
        assert_eq!(num_queries[4], 24, "nQueries mismatch: got {}", num_queries[4]);

        let num_ood_samples = &sec.num_ood_samples;
        assert!(num_ood_samples.len() == 4, "Expected 4 iterations, got {}", num_ood_samples.len());
        assert_eq!(num_ood_samples[0], 1, "nOOD mismatch: got {}", num_ood_samples[0]);
        assert_eq!(num_ood_samples[1], 1, "nOOD mismatch: got {}", num_ood_samples[1]);
        assert_eq!(num_ood_samples[2], 1, "nOOD mismatch: got {}", num_ood_samples[2]);
        assert_eq!(num_ood_samples[3], 1, "nOOD mismatch: got {}", num_ood_samples[3]);

        let grinding_bits_folding = &sec.grinding_bits_folding;
        assert!(grinding_bits_folding.len() == 5, "Expected 5 iterations, got {}", grinding_bits_folding.len());
        for (i, g) in grinding_bits_folding.iter().enumerate() {
            for (s, bits) in g.iter().enumerate() {
                assert_eq!(*bits, 0, "grinding_bits_folding mismatch for iteration {i}, round {}: got {bits}", s + 1);
            }
        }

        let grinding_bits_queries = &sec.grinding_bits_queries;
        assert!(grinding_bits_queries.len() == 5, "Expected 5 iterations, got {}", grinding_bits_queries.len());
        assert_eq!(
            grinding_bits_queries[0], 16,
            "grinding_bits_queries mismatch for iteration 0: got {}",
            grinding_bits_queries[0]
        );
        assert_eq!(
            grinding_bits_queries[1], 16,
            "grinding_bits_queries mismatch for iteration 1: got {}",
            grinding_bits_queries[1]
        );
        assert_eq!(
            grinding_bits_queries[2], 16,
            "grinding_bits_queries mismatch for iteration 2: got {}",
            grinding_bits_queries[2]
        );
        assert_eq!(
            grinding_bits_queries[3], 16,
            "grinding_bits_queries mismatch for iteration 3: got {}",
            grinding_bits_queries[3]
        );
        assert_eq!(
            grinding_bits_queries[4], 16,
            "grinding_bits_queries mismatch for iteration 4: got {}",
            grinding_bits_queries[4]
        );

        let grinding_bits_ood = &sec.grinding_bits_ood;
        assert!(grinding_bits_ood.len() == 4, "Expected 4 iterations, got {}", grinding_bits_ood.len());
        assert_eq!(grinding_bits_ood[0], 0, "grinding_bits_ood mismatch for iteration 1: got {}", grinding_bits_ood[0]);
        assert_eq!(grinding_bits_ood[1], 0, "grinding_bits_ood mismatch for iteration 2: got {}", grinding_bits_ood[1]);
        assert_eq!(grinding_bits_ood[2], 0, "grinding_bits_ood mismatch for iteration 3: got {}", grinding_bits_ood[2]);
        assert_eq!(grinding_bits_ood[3], 0, "grinding_bits_ood mismatch for iteration 4: got {}", grinding_bits_ood[3]);
    }

    #[test]
    fn test_recursive2_params() {
        let whir = Whir::new(test_config(1 << 17, 0.125, 145, vec![3, 3, 3, 3, 3], 20));

        assert_eq!(whir.num_merkle_openings(), 195);
        assert_eq!(whir.total_query_hashes().round(), 6_321.0);

        assert_eq!(whir.alpha(), 0.0);

        let levels = whir.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 132);
        assert_eq!(level("fold(i=0,s=1)"), 140);
        assert_eq!(level("fold(i=0,s=2)"), 141);
        assert_eq!(level("fold(i=0,s=3)"), 142);
        assert_eq!(level("ood(i=1)"), 159);
        assert_eq!(level("shift(i=1)"), 128);
        assert_eq!(level("fold(i=1,s=1)"), 143);
        assert_eq!(level("fold(i=1,s=2)"), 144);
        assert_eq!(level("fold(i=1,s=3)"), 145);
        assert_eq!(level("ood(i=2)"), 160);
        assert_eq!(level("shift(i=2)"), 128);
        assert_eq!(level("fold(i=2,s=1)"), 145);
        assert_eq!(level("fold(i=2,s=2)"), 146);
        assert_eq!(level("fold(i=2,s=3)"), 147);
        assert_eq!(level("ood(i=3)"), 161);
        assert_eq!(level("shift(i=3)"), 130);
        assert_eq!(level("fold(i=3,s=1)"), 148);
        assert_eq!(level("fold(i=3,s=2)"), 149);
        assert_eq!(level("fold(i=3,s=3)"), 150);
        assert_eq!(level("ood(i=4)"), 162);
        assert_eq!(level("shift(i=4)"), 129);
        assert_eq!(level("fold(i=4,s=1)"), 150);
        assert_eq!(level("fold(i=4,s=2)"), 151);
        assert_eq!(level("fold(i=4,s=3)"), 152);
        assert_eq!(level("final"), 131);
        assert_eq!(whir.total_security_bits(), 128);

        let sec = whir.security_params();
        assert_eq!(sec.num_queries, vec![73, 44, 32, 25, 21]);
        assert_eq!(sec.num_ood_samples, vec![1; 4]);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![vec![0; 3]; 5]);
        assert_eq!(sec.grinding_bits_queries, vec![20; 5]);
        assert_eq!(sec.grinding_bits_ood, vec![0; 4]);
    }

    #[test]
    fn test_final_params() {
        let whir = Whir::new(test_config(1 << 16, 0.03125, 139, vec![4, 4, 4, 4], 22));

        assert_eq!(whir.num_merkle_openings(), 109);
        assert_eq!(whir.total_query_hashes().round(), 4_438.0);

        assert_eq!(whir.alpha(), 0.0);

        let levels = whir.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 133);
        assert_eq!(level("fold(i=0,s=1)"), 141);
        assert_eq!(level("fold(i=0,s=2)"), 142);
        assert_eq!(level("fold(i=0,s=3)"), 143);
        assert_eq!(level("fold(i=0,s=4)"), 144);
        assert_eq!(level("ood(i=1)"), 158);
        assert_eq!(level("shift(i=1)"), 128);
        assert_eq!(level("fold(i=1,s=1)"), 144);
        assert_eq!(level("fold(i=1,s=2)"), 145);
        assert_eq!(level("fold(i=1,s=3)"), 146);
        assert_eq!(level("fold(i=1,s=4)"), 147);
        assert_eq!(level("ood(i=2)"), 159);
        assert_eq!(level("shift(i=2)"), 131);
        assert_eq!(level("fold(i=2,s=1)"), 147);
        assert_eq!(level("fold(i=2,s=2)"), 148);
        assert_eq!(level("fold(i=2,s=3)"), 149);
        assert_eq!(level("fold(i=2,s=4)"), 150);
        assert_eq!(level("ood(i=3)"), 160);
        assert_eq!(level("shift(i=3)"), 133);
        assert_eq!(level("fold(i=3,s=1)"), 145);
        assert_eq!(level("fold(i=3,s=2)"), 146);
        assert_eq!(level("fold(i=3,s=3)"), 147);
        assert_eq!(level("fold(i=3,s=4)"), 148);
        assert_eq!(level("final"), 132);
        assert_eq!(whir.total_security_bits(), 128);

        let sec = whir.security_params();
        assert_eq!(sec.num_queries, vec![43, 28, 21, 17]);
        assert_eq!(sec.num_ood_samples, vec![1; 3]);
        assert_eq!(sec.grinding_bits_batching, 0);
        assert_eq!(sec.grinding_bits_folding, vec![vec![0; 4]; 4]);
        assert_eq!(sec.grinding_bits_queries, vec![22; 4]);
        assert_eq!(sec.grinding_bits_ood, vec![0; 3]);
    }
}
