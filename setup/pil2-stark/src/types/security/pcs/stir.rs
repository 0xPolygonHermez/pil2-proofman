use crate::types::security::pcs::types::apply_grinding;

use super::super::regimes::{DecodingRegime, ProximityGapsRegime};
use super::types::{
    Batching, Pcs, bits_of_security_from_error, coset_opening_hashes, merkle_opening_size_bits, security_from_error,
};

/// Configuration for the STIR PCS (Arnon–Chiesa–Fenzi–Yogev, ePrint 2024/390).
#[derive(Clone, Debug)]
pub struct StirConfig {
    /// Field size |F|.
    pub field_size: f64,
    /// Domain size before low-degree extension. Must be 2^h.
    pub trace_length: u32,
    /// The code rate ρ. Must be an exact power of two 2^-k.
    pub rate: f64,
    /// Batching strategy (see [`Batching`]).
    pub batching: Batching,
    /// Number of polynomials batched.
    pub batch_size: u64,
    /// Per-iteration folding factors `kᵢ`, in bits: iteration `i` folds by `2^kᵢ`.
    pub log_folding_factors: Vec<u32>,
    /// The grinding budget for each iteration's query message (length `M`).
    /// Grinding and queries buy the same bits, so a per-round budget shifts
    /// prover proof-of-work to the rounds where queries are expensive.
    pub max_grinding_bits_queries: Vec<u64>,
    /// Whether to use the maximum number of grinding bits.
    pub use_max_grinding_bits_query: bool,
    /// The arity of the Merkle trees used in STIR.
    pub tree_arity: u64,
    /// The output length of the Merkle-tree hash, in bits.
    pub hash_size_bits: u64,
    /// Base-field element size in bits. The initial oracle f₀ is over the
    /// base field; folded oracles and OOD replies take full extension elements.
    pub base_field_bits: u64,
    /// The target security level in bits.
    pub target_security_bits: u64,
    /// The decoding regime kind. The gap-widening factor `alpha` is deduced.
    pub regime: DecodingRegime,
}

/// Security parameters *deduced* from a [`StirConfig`].
#[derive(Clone, Debug)]
pub struct StirSecurityParams {
    /// Per-iteration query counts `tᵢ` (length `M`).
    pub num_queries: Vec<u64>,
    /// Per-iteration OOD sample counts `sᵢ` (length `M-1`).
    pub num_ood_samples: Vec<u64>,
    /// Grinding bits for the batching phase.
    pub grinding_bits_batching: u32,
    /// Grinding bits for the initial fold of `f₀`.
    pub grinding_bits_folding: u32,
    /// Grinding bits per iteration's query/shift phase (length `M`).
    pub grinding_bits_queries: Vec<u32>,
    /// Grinding bits per iteration's OOD phase (length `M-1`).
    pub grinding_bits_ood: Vec<u32>,
}

/// STIR Polynomial Commitment Scheme.
#[derive(Clone, Debug)]
pub struct Stir {
    cfg: StirConfig,
    /// `log2(1/ρᵢ)` — per-iteration inverse rates (length `M+1`).
    log_inv_rates: Vec<u32>,
    /// h = log2(trace_length).
    log_trace: u32,
    /// Number of iterations `M`.
    num_rounds: usize,
    /// `mᵢ` — per-iteration log-dimensions (length `M+1`).
    log_round_dimensions: Vec<u32>,
    /// Degree of the final polynomial sent in clear.
    early_stop_degree: u32,
    /// Security parameters.
    sec_params: StirSecurityParams,
    /// Gap-widening factor for the regime.
    alpha: f64,
}

impl Stir {
    pub fn new(cfg: StirConfig) -> Self {
        let mut stir = Self::validate(cfg);
        let (sec_params, alpha) = stir.solve();
        stir.validate_query_counts(&sec_params);
        stir.sec_params = sec_params;
        stir.alpha = alpha;
        stir
    }

    /// Construct with an externally fixed schedule.
    pub fn with_security_params(cfg: StirConfig, sec_params: StirSecurityParams) -> Self {
        let mut stir = Self::validate(cfg);
        let m = stir.num_rounds;

        assert_eq!(sec_params.num_queries.len(), m, "Expected one query count per iteration");
        assert_eq!(sec_params.num_ood_samples.len(), m - 1, "Expected M-1 OOD sample counts");
        assert_eq!(sec_params.grinding_bits_queries.len(), m, "Expected one query grinding entry per iteration");
        assert_eq!(sec_params.grinding_bits_ood.len(), m - 1, "Expected M-1 OOD grinding entries");

        stir.validate_query_counts(&sec_params);
        stir.sec_params = sec_params;
        stir
    }

    /// STIR is only meaningful while the quotient keeps a positive degree bound: every quotient
    /// round i = 1..M-1 divides by ∏_{a∈G_i}(X − a) with |G_i| = t_{i-1} + s_i, so it needs
    /// |G_i| < d_i.
    fn validate_query_counts(&self, sec_params: &StirSecurityParams) {
        for i in 1..self.num_rounds {
            let d_i = 1u64 << self.log_round_dimensions[i];
            let n_g = sec_params.num_queries[i - 1] + sec_params.num_ood_samples[i - 1];
            assert!(
                n_g < d_i,
                "STIR schedule is invalid for this air: iteration {i} quotients over |G| = {n_g} \
                 points (t_{} = {} queries + {} out-of-domain samples) but the degree folds down \
                 to d_{i} = {d_i}. The trace is too small for STIR at this security target — \
                 raise finalDegree, lower foldingFactor, or keep FRI for this air.",
                i - 1,
                sec_params.num_queries[i - 1],
                sec_params.num_ood_samples[i - 1],
            );
        }
    }

    /// Structural validation shared by both constructors.
    fn validate(cfg: StirConfig) -> Self {
        let num_rounds = cfg.log_folding_factors.len();

        assert_eq!(cfg.max_grinding_bits_queries.len(), num_rounds, "Expected one query grinding budget per iteration");
        for &g in &cfg.max_grinding_bits_queries {
            assert!(g < 64, "A grinding budget of {g} bits does not fit the 64-bit proof-of-work check");
        }

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

        assert!(cfg.batch_size >= 1, "Batch size must be at least 1");
        assert!(num_rounds >= 1, "Must have at least 1 iteration");
        assert!(cfg.log_folding_factors.iter().all(|&k| k >= 1), "Every folding factor must be >= 1 to reduce degree");

        // The final polynomial must have a non-negative log-degree: m₀ >= Σᵢ kᵢ.
        let total_reduction: u32 = cfg.log_folding_factors.iter().sum();
        assert!(
            total_reduction <= log_trace,
            "Reducing log-degree {} by {total_reduction} (sum of folding factors) leaves a negative log-degree",
            log_trace,
        );

        // Per-iteration log-degree mᵢ and log-inverse-rate μᵢ.
        //
        // Recurrence:
        //   m_{i+1}  = m_i - k_i        (folding by 2^{k_i})
        //   μ_{i+1}  = μ_i + (k_i - 1)  (domain halves; degree drops by 2^{k_i})
        let mut log_round_dimensions = Vec::with_capacity(num_rounds + 1);
        let mut log_inv_rates = Vec::with_capacity(num_rounds + 1);
        log_round_dimensions.push(log_trace);
        log_inv_rates.push(log_inv_rate);
        for (i, &k_i) in cfg.log_folding_factors.iter().enumerate() {
            log_round_dimensions.push(log_round_dimensions[i] - k_i);
            log_inv_rates.push(log_inv_rates[i] + (k_i - 1));
        }
        let early_stop_degree = 1 << log_round_dimensions[num_rounds];

        let empty = StirSecurityParams {
            num_queries: vec![0; num_rounds],
            num_ood_samples: vec![0; num_rounds - 1],
            grinding_bits_batching: 0,
            grinding_bits_folding: 0,
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
    fn solve(&self) -> (StirSecurityParams, f64) {
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
    fn compute_security_params(&self, regime: &dyn ProximityGapsRegime) -> StirSecurityParams {
        let m = self.num_rounds;
        let target = self.cfg.target_security_bits;
        let num_ood_samples = vec![1u64; m - 1];

        // Non-query stages: grinding fills whatever is missing to the target.
        let deficit = |error: f64| target.saturating_sub(bits_of_security_from_error(error) as u64) as u32;

        // Batching grinding.
        let grinding_bits_batching = if self.cfg.batch_size > 1 { deficit(self.batch_error(regime, 0)) } else { 0 };

        // Initial fold grinding.
        let grinding_bits_folding = deficit(self.fold_error(regime, 0));

        // OOD grinding.
        let grinding_bits_ood: Vec<u32> =
            (1..m).map(|i| deficit(self.ood_error(regime, i, num_ood_samples[i - 1], 0))).collect();

        // Query counts and grinding, iteration by iteration.
        let mut num_queries = Vec::with_capacity(m);
        let mut grinding_bits_queries = Vec::with_capacity(m);
        for i in 0..m {
            // The queries into fᵢ contribute (1 − δᵢ)^{tᵢ}: to the shift error
            // of iteration i+1, or to the final error when i = M−1.
            let pp = self.proximity_parameter_for(regime, i);
            let security_per_query = security_from_error(1.0 - pp);

            // Find max efficient grinding: 2^g < hashesPerQuery => g < log2(hashesPerQuery).
            let hash_per_query = self.query_num_hashes(i);
            let max_efficient_grinding = hash_per_query.log2().floor() as u64;
            let g = if self.cfg.use_max_grinding_bits_query {
                self.cfg.max_grinding_bits_queries[i]
            } else {
                max_efficient_grinding.min(self.cfg.max_grinding_bits_queries[i])
            } as u32;

            // Queries needed for the (1 − δᵢ)^{tᵢ} term alone.
            let needed_from_queries = target as f64 - g as f64;
            let mut t = if needed_from_queries > 0.0 {
                (needed_from_queries / security_per_query).ceil() as u64
            } else {
                1 // Need at least 1 query
            };

            // Now account for the other shift terms, err*(C_{i+1}, tᵢ + s) and
            // the fold of f_{i+1}, which the query count does not shrink (the
            // first even grows with t). Add queries while it still helps.
            if let Some(&s) = num_ood_samples.get(i) {
                let mut error = self.shift_error(regime, i + 1, t, s, g);
                while (bits_of_security_from_error(error) as u64) < target {
                    let next_error = self.shift_error(regime, i + 1, t + 1, s, g);
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

        StirSecurityParams {
            num_queries,
            num_ood_samples,
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

    /// Estimated worst-case proof size in bits: per-iteration roots and OOD
    /// replies, the decision-phase openings, and the final polynomial in
    /// clear. Unlike WHIR there are no sumcheck polynomials.
    pub fn proof_size_bits(&self) -> u64 {
        let ext_bits = self.cfg.field_size.log2().round() as u64;
        let hash = self.cfg.hash_size_bits;
        let mut size = 0.0;

        // Initial commitment root.
        size += hash as f64;

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
    fn meets_security_target(&self, regime: &dyn ProximityGapsRegime, sec_params: &StirSecurityParams) -> bool {
        self.security_levels_with(regime, sec_params)
            .into_iter()
            .all(|(_, bits)| bits as u64 >= self.cfg.target_security_bits)
    }

    /// Per-component PCS security levels (bits), keyed by component name.
    /// One entry per round of the round-by-round analysis (STIR, Theorem 5.3).
    pub fn security_levels(&self) -> Vec<(String, u32)> {
        let regime = self.regime();
        self.security_levels_with(regime.as_ref(), &self.sec_params)
    }

    fn security_levels_with(&self, regime: &dyn ProximityGapsRegime, sec: &StirSecurityParams) -> Vec<(String, u32)> {
        let m = self.num_rounds;
        let mut out = Vec::new();

        // Batching step.
        if self.cfg.batch_size > 1 {
            let batch_error = self.batch_error(regime, sec.grinding_bits_batching);
            out.push(("batching".to_string(), bits_of_security_from_error(batch_error)));
        }

        // Initial fold of f₀: the only fold that is its own round.
        let fold_error = apply_grinding(self.fold_error(regime, 0), sec.grinding_bits_folding);
        out.push(("fold(i=0)".to_string(), bits_of_security_from_error(fold_error)));

        // Main loop (i=1,...,M-1): OOD and shift errors (the latter includes
        // the degree correction and the fold of fᵢ).
        for i in 1..m {
            let n_samples = sec.num_ood_samples[i - 1];
            let ood_error = self.ood_error(regime, i, n_samples, sec.grinding_bits_ood[i - 1]);
            out.push((format!("ood(i={i})"), bits_of_security_from_error(ood_error)));

            let shift_error =
                self.shift_error(regime, i, sec.num_queries[i - 1], n_samples, sec.grinding_bits_queries[i - 1]);
            out.push((format!("shift(i={i})"), bits_of_security_from_error(shift_error)));
        }

        // Final error.
        let final_error = self.final_error(regime, sec.num_queries[m - 1], sec.grinding_bits_queries[m - 1]);
        out.push(("final".to_string(), bits_of_security_from_error(final_error)));

        out
    }

    /// Error from the batching step, on the code C₀.
    fn batch_error(&self, regime: &dyn ProximityGapsRegime, grinding_bits: u32) -> f64 {
        let (rate, dimension) = self.code_for(0, false);
        let epsilon = match self.cfg.batching {
            Batching::Powers => regime.error_powers(&rate, dimension, self.cfg.batch_size),
            Batching::Multilinear => regime.error_multilinear(&rate, dimension, self.cfg.batch_size),
            Batching::Affine => regime.error_linear(&rate, dimension),
        };
        apply_grinding(epsilon, grinding_bits)
    }

    /// Error of folding fᵢ by 2^{kᵢ} with the powers 1, r, …, r^{2^{kᵢ}−1}
    /// of a single challenge: err*(RS[F, Lᵢ^{(2^{kᵢ})}, 2^{mᵢ−kᵢ}], 2^{kᵢ}, δᵢ).
    /// No grinding is applied here; the caller does it.
    fn fold_error(&self, regime: &dyn ProximityGapsRegime, iteration: usize) -> f64 {
        let (rate, dimension) = self.code_for(iteration, true);
        regime.error_powers(&rate, dimension, 1u64 << self.cfg.log_folding_factors[iteration])
    }

    /// Out-of-domain error for fᵢ, i = 1..M-1:
    ///   (ℓᵢ² / 2) · (2^{mᵢ} / (|F| − |Lᵢ|))^{sᵢ}
    /// The OOD point is sampled from F \ Lᵢ, hence the denominator. For
    /// sᵢ = 1 this is the paper's term; for sᵢ > 1 only the collision
    /// probability is raised to sᵢ (as in stir-whir-scripts).
    fn ood_error(&self, regime: &dyn ProximityGapsRegime, iteration: usize, n_samples: u64, grinding_bits: u32) -> f64 {
        assert!(iteration >= 1 && iteration < self.num_rounds, "Iteration index out of bounds");

        let list_size = self.list_size_for(regime, iteration);
        let degree = f64::exp2(self.log_round_dimensions[iteration] as f64);
        let domain_size = f64::exp2((self.log_round_dimensions[iteration] + self.log_inv_rates[iteration]) as f64);

        let epsilon =
            list_size * list_size / 2.0 * (degree / (self.cfg.field_size - domain_size)).powi(n_samples as i32);
        apply_grinding(epsilon, grinding_bits)
    }

    /// Shift error of iteration i = 1..M-1, three terms:
    ///   1. (1 − δ_{i−1})^{t_{i−1}}: the queries into f_{i−1} miss the disagreement;
    ///   2. err*(Cᵢ, t_{i−1} + sᵢ, δᵢ): degree correction, combining the
    ///      in-domain and out-of-domain quotient answers with powers of a challenge;
    ///   3. err*(folded Cᵢ, 2^{kᵢ}, δᵢ): folding fᵢ for the next iteration.
    ///
    /// All three share one verifier message, hence one grinding value.
    fn shift_error(
        &self,
        regime: &dyn ProximityGapsRegime,
        iteration: usize,
        n_queries: u64,
        n_samples: u64,
        grinding_bits: u32,
    ) -> f64 {
        assert!(iteration >= 1 && iteration < self.num_rounds, "Iteration index out of bounds");

        let pp = self.proximity_parameter_for(regime, iteration - 1);
        let first = (1.0 - pp).powi(n_queries as i32);

        let (rate, dimension) = self.code_for(iteration, false);
        let second = regime.error_powers(&rate, dimension, n_queries + n_samples);

        let third = self.fold_error(regime, iteration);

        apply_grinding(first + second + third, grinding_bits)
    }

    /// Final error (1 − δ_{M−1})^{t_{M−1}}.
    fn final_error(&self, regime: &dyn ProximityGapsRegime, n_queries: u64, grinding_bits: u32) -> f64 {
        let pp = self.proximity_parameter_for(regime, self.num_rounds - 1);
        let epsilon = (1.0 - pp).powi(n_queries as i32);
        apply_grinding(epsilon, grinding_bits)
    }

    /// The code of iteration i as a `(rate, dimension)` pair:
    /// - `folded = false`: Cᵢ = RS[F, Lᵢ, 2^{mᵢ}], the code fᵢ is checked against;
    /// - `folded = true`: RS[F, Lᵢ^{(2^{kᵢ})}, 2^{mᵢ−kᵢ}], the code of Fold(fᵢ, kᵢ, r).
    ///
    /// Both have rate 2^{−μᵢ}, since degree and domain shrink by the same factor.
    fn code_for(&self, iteration: usize, folded: bool) -> (f64, u32) {
        assert!(iteration < self.num_rounds, "Iteration index out of bounds");
        let mut log_dimension = self.log_round_dimensions[iteration];
        if folded {
            log_dimension -= self.cfg.log_folding_factors[iteration];
        }
        let rate = f64::exp2(-(self.log_inv_rates[iteration] as f64));
        (rate, 1u32 << log_dimension)
    }

    /// ℓᵢ such that Cᵢ is (δᵢ, ℓᵢ)-list decodable.
    fn list_size_for(&self, regime: &dyn ProximityGapsRegime, iteration: usize) -> f64 {
        let (rate, dimension) = self.code_for(iteration, false);
        regime.max_list_size(&rate, dimension) as f64
    }

    /// The proximity parameter δᵢ: the minimum over Cᵢ and its folded code
    /// (the theorem needs δᵢ below the regime's bound for both).
    fn proximity_parameter_for(&self, regime: &dyn ProximityGapsRegime, iteration: usize) -> f64 {
        let (rate, _) = self.code_for(iteration, false);
        let (folded_rate, _) = self.code_for(iteration, true);
        regime.proximity_parameter(&rate).min(regime.proximity_parameter(&folded_rate))
    }

    pub fn config(&self) -> &StirConfig {
        &self.cfg
    }

    /// The solved regime.
    pub fn regime(&self) -> Box<dyn ProximityGapsRegime> {
        self.cfg.regime.instantiate(self.cfg.field_size, self.alpha)
    }

    /// The deduced (or pinned) security parameters.
    pub fn security_params(&self) -> &StirSecurityParams {
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
            ("Grinding Bits Folding", self.sec_params.grinding_bits_folding.to_string()),
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

impl Pcs for Stir {
    fn identifier(&self) -> &'static str {
        "STIR"
    }

    fn security_levels(&self) -> Vec<(String, u32)> {
        Stir::security_levels(self)
    }

    fn parameter_summary(&self) -> String {
        Stir::parameter_summary(self)
    }

    fn num_merkle_openings(&self) -> u64 {
        Stir::num_merkle_openings(self)
    }

    fn total_query_hashes(&self) -> f64 {
        Stir::total_query_hashes(self)
    }

    fn proof_size_bits(&self) -> u64 {
        Stir::proof_size_bits(self)
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
    ) -> StirConfig {
        let max_grinding_bits_queries = vec![max_grinding_bits_query; folding_factors.len()];
        StirConfig {
            field_size: goldilocks_safe_extension_field_size(),
            trace_length,
            rate,
            batching: Batching::Powers,
            batch_size,
            log_folding_factors: folding_factors,
            max_grinding_bits_queries,
            use_max_grinding_bits_query: true,
            tree_arity: 4,
            hash_size_bits: 256,
            base_field_bits: 64,
            target_security_bits: 128,
            regime: DecodingRegime::Jbr,
        }
    }

    fn assert_all_components_reach_target(stir: &Stir) {
        let levels = stir.security_levels();
        for (name, bits) in &levels {
            assert!(*bits >= 128, "{name} reaches only {bits} bits: {levels:?}");
        }
        assert_eq!(stir.total_security_bits(), 128);
    }

    #[test]
    fn test_main_params() {
        let stir = Stir::new(test_config(1 << 22, 0.5, 61, vec![3, 3, 3, 3, 3, 3], 16));
        assert_eq!(stir.alpha(), 0.0);
        assert_all_components_reach_target(&stir);

        let levels = stir.security_levels();
        let names: Vec<&str> = levels.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "batching",
                "fold(i=0)",
                "ood(i=1)",
                "shift(i=1)",
                "ood(i=2)",
                "shift(i=2)",
                "ood(i=3)",
                "shift(i=3)",
                "ood(i=4)",
                "shift(i=4)",
                "ood(i=5)",
                "shift(i=5)",
                "final"
            ]
        );

        let sec = stir.security_params();
        assert_eq!(sec.num_ood_samples, vec![1; 5]);
        assert_eq!(sec.grinding_bits_queries, vec![16; 6]);
        assert_eq!(sec.grinding_bits_ood, vec![0; 5]);
        // The final polynomial is queried at t_{M-1} points against a code of rate
        // 2^-(1+5*2), so t_{M-1} is well below the initial 228.
        assert_eq!(sec.num_queries[0], 228);
        assert!(
            sec.num_queries.windows(2).all(|w| w[0] >= w[1]),
            "query counts should not increase: {:?}",
            sec.num_queries
        );
        assert_eq!(stir.num_merkle_openings(), sec.num_queries.iter().sum::<u64>());
    }

    #[test]
    fn test_dma_params() {
        let stir = Stir::new(test_config(1 << 21, 0.5, 46, vec![3, 3, 3, 3, 3, 2], 16));
        assert_eq!(stir.alpha(), 0.0);
        assert_all_components_reach_target(&stir);
        assert_eq!(stir.security_params().num_queries[0], 228);
    }

    #[test]
    fn test_keccakf_params() {
        let stir = Stir::new(test_config(1 << 17, 0.5, 4065, vec![3, 3, 3, 3], 23));
        assert_eq!(stir.alpha(), 0.0);
        assert_all_components_reach_target(&stir);

        let sec = stir.security_params();
        assert_eq!(sec.num_queries[0], 213);
        assert_eq!(sec.grinding_bits_batching, 1, "batching deficit should be 1 bit");
        assert_eq!(sec.grinding_bits_queries, vec![23; 4]);
    }

    #[test]
    fn test_poseidon2_params() {
        let stir = Stir::new(test_config(1 << 17, 0.25, 182, vec![3, 3, 3, 3, 2], 16));
        assert_eq!(stir.alpha(), 0.0);
        assert_all_components_reach_target(&stir);
        assert_eq!(stir.security_params().num_queries[0], 114);
    }

    #[test]
    fn test_recursive2_params() {
        let stir = Stir::new(test_config(1 << 17, 0.125, 145, vec![3, 3, 3, 3, 3], 20));
        assert_eq!(stir.alpha(), 0.0);
        assert_all_components_reach_target(&stir);
        assert_eq!(stir.security_params().num_queries[0], 73);
    }

    #[test]
    fn test_final_params() {
        let stir = Stir::new(test_config(1 << 16, 0.03125, 139, vec![4, 4, 4], 22));
        assert_eq!(stir.alpha(), 0.0);
        assert_all_components_reach_target(&stir);
        assert_eq!(stir.security_params().num_queries[0], 43);
    }

    /// STIR has no sumcheck, so for the same geometry its proof must be no
    /// larger than WHIR's when both use the same query schedule.
    #[test]
    fn test_proof_size_vs_whir() {
        use super::super::whir::{Whir, WhirConfig, WhirSecurityParams};

        let stir = Stir::new(test_config(1 << 22, 0.5, 61, vec![3, 3, 3, 3, 3, 3], 16));
        let sec = stir.security_params();
        let m = sec.num_queries.len();

        let whir = Whir::with_security_params(
            WhirConfig {
                field_size: goldilocks_safe_extension_field_size(),
                trace_length: 1 << 22,
                rate: 0.5,
                batching: Batching::Powers,
                batch_size: 61,
                log_folding_factors: vec![3; 6],
                constraint_degree: 3,
                max_grinding_bits_query: 16,
                use_max_grinding_bits_query: true,
                tree_arity: 4,
                hash_size_bits: 256,
                base_field_bits: 64,
                target_security_bits: 128,
                regime: DecodingRegime::Jbr,
            },
            WhirSecurityParams {
                num_queries: sec.num_queries.clone(),
                num_ood_samples: sec.num_ood_samples.clone(),
                grinding_bits_batching: 0,
                grinding_bits_folding: vec![vec![0; 3]; m],
                grinding_bits_queries: sec.grinding_bits_queries.clone(),
                grinding_bits_ood: sec.grinding_bits_ood.clone(),
            },
        );

        assert!(stir.proof_size_bits() < whir.proof_size_bits());
        assert_eq!(stir.num_merkle_openings(), whir.num_merkle_openings());
        assert_eq!(stir.total_query_hashes(), whir.total_query_hashes());
    }

    /// The pinned-schedule constructor must reproduce the solved instance.
    #[test]
    fn test_with_security_params_roundtrip() {
        let cfg = test_config(1 << 17, 0.25, 182, vec![3, 3, 3, 3, 2], 16);
        let solved = Stir::new(cfg.clone());
        let pinned = Stir::with_security_params(cfg, solved.security_params().clone());
        assert_eq!(pinned.security_levels(), solved.security_levels());
        assert_eq!(pinned.proof_size_bits(), solved.proof_size_bits());
    }

    /// An air too small for its query schedule must be rejected at setup time, not left for the
    /// prover to trip over: with d_0 = 2^8 at 128 bits the solver wants hundreds of queries, and
    /// one fold drops the degree below them.
    #[test]
    #[should_panic(expected = "too small for STIR")]
    fn schedules_that_fold_below_the_query_count_are_rejected() {
        Stir::new(test_config(1 << 8, 0.5, 10, vec![3, 3], 16));
    }

    /// Per-round grinding budgets: each round's query count comes from its own
    /// budget, since grinding and queries buy the same bits. Zeroing one round's
    /// budget must raise that round's query count and leave the others alone.
    #[test]
    fn per_round_grinding_budgets_trade_against_that_rounds_queries() {
        let uniform = Stir::new(test_config(1 << 17, 0.25, 182, vec![3, 3, 3, 3, 2], 16));

        let mut cfg = test_config(1 << 17, 0.25, 182, vec![3, 3, 3, 3, 2], 16);
        cfg.max_grinding_bits_queries[2] = 0;
        let skewed = Stir::new(cfg);

        let t_uniform = &uniform.security_params().num_queries;
        let t_skewed = &skewed.security_params().num_queries;
        assert!(
            t_skewed[2] > t_uniform[2],
            "round 2 lost its grinding, so it must query more: {t_skewed:?} vs {t_uniform:?}"
        );
        for i in [0, 1, 3, 4] {
            assert_eq!(t_skewed[i], t_uniform[i], "round {i} kept its budget, so its query count must not move");
        }
        assert_eq!(skewed.security_params().grinding_bits_queries[2], 0);
        assert_all_components_reach_target(&skewed);
    }

    /// The budgets are per round, so their count must match the schedule.
    #[test]
    #[should_panic(expected = "one query grinding budget per iteration")]
    fn budget_count_must_match_the_schedule() {
        let mut cfg = test_config(1 << 17, 0.25, 182, vec![3, 3, 3, 3, 2], 16);
        cfg.max_grinding_bits_queries.pop();
        Stir::new(cfg);
    }
}
