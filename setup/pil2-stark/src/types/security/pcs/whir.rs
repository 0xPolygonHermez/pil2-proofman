use super::super::regimes::{DecodingRegime, ProximityGapsRegime};
use super::types::{Batching, Pcs, bits_of_security_from_log2_error, log2_add};

/// Configuration for the WHIR PCS: the *free* parameters, known upfront.
/// Per-iteration query counts, OOD sample counts and grinding bits are
/// deduced at construction time (see [`WhirSecurityParams`]).
#[derive(Clone, Debug)]
pub struct WhirConfig {
    /// Field size |F|.
    pub field_size: f64,
    /// `m₀` — number of variables of the initial multilinear.
    pub num_variables: u32,
    /// `log2(1/ρ₀)` — initial inverse rate.
    pub log_inv_rate: u32,
    /// Per-iteration folding factors `kᵢ`, in bits: iteration `i` folds
    /// `2^kᵢ` variables.
    pub folding_factors: Vec<u32>,
    /// Number of polynomials batched.
    pub batch_size: u64,
    /// Batching strategy (see [`Batching`]).
    pub batching: Batching,
    /// Constraint degree `d = max(d*, 3)`, `d* = 1 + deg_Z(w̃) + max_i{deg_{X_i}(w̃)}`.
    pub constraint_degree: u64,
    /// Query-phase grinding budget (pow bits), uniform across iterations.
    pub grinding_bits: u32,
    /// The target security level in bits.
    pub target_security_bits: u64,
    /// The decoding regime kind. The gap-widening factor `alpha` is deduced.
    pub regime: DecodingRegime,
}

/// Security parameters *deduced* from a [`WhirConfig`] (or pinned externally
/// via [`Whir::with_security_params`]).
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
///
/// Like [`super::fri::Fri`], built in two phases: structural validation,
/// then a solve for the per-iteration query counts and the regime's `alpha`.
/// Use [`Whir::with_security_params`] to audit an externally fixed schedule
/// instead of solving for one.
#[derive(Clone, Debug)]
pub struct Whir {
    cfg: WhirConfig,
    /// `mᵢ` — number of variables entering iteration i (length `M+1`).
    log_dimensions: Vec<u32>,
    /// `log2(1/ρᵢ)` — per-iteration inverse rates (length `M+1`).
    log_inv_rates: Vec<u32>,
    /// Number of iterations `M`.
    num_iterations: usize,
    /// Deduced (or pinned) security parameters.
    sec_params: WhirSecurityParams,
    /// Deduced gap-widening factor for the regime.
    alpha: f64,
}

impl Whir {
    /// Construct and *solve*: per-iteration query counts sized to the target
    /// at each iteration's rate, one OOD sample per inner iteration, the
    /// configured query grinding, and the smallest `alpha` meeting the target.
    pub fn new(cfg: WhirConfig) -> Self {
        let mut whir = Self::validate(cfg);
        let (sec_params, alpha) = whir.solve();
        whir.sec_params = sec_params;
        whir.alpha = alpha;
        whir
    }

    /// Construct with an externally fixed schedule (no solving, `alpha = 0`).
    /// Use `security_levels`/`total_security_bits` to audit it.
    pub fn with_security_params(cfg: WhirConfig, sec_params: WhirSecurityParams) -> Self {
        let mut whir = Self::validate(cfg);
        let m = whir.num_iterations;

        assert_eq!(sec_params.num_queries.len(), m, "Expected one query count per iteration");
        assert_eq!(sec_params.num_ood_samples.len(), m - 1, "Expected M-1 OOD sample counts");
        assert_eq!(sec_params.grinding_bits_queries.len(), m, "Expected one query grinding entry per iteration");
        assert_eq!(sec_params.grinding_bits_ood.len(), m - 1, "Expected M-1 OOD grinding entries");
        assert_eq!(sec_params.grinding_bits_folding.len(), m, "Expected one folding grinding vector per iteration");
        for (i, g) in sec_params.grinding_bits_folding.iter().enumerate() {
            // Each iteration has a sumcheck phase with exactly kᵢ rounds.
            assert_eq!(
                g.len(),
                whir.cfg.folding_factors[i] as usize,
                "Iteration {i}: expected one folding grinding entry per sumcheck round"
            );
        }

        whir.sec_params = sec_params;
        whir
    }

    /// Structural validation shared by both constructors.
    fn validate(cfg: WhirConfig) -> Self {
        let m = cfg.folding_factors.len();

        // d = max(d*, 3), where d* = 1 + deg_Z(w̃) + max_i{deg_{X_i}(w̃)}.
        assert!(cfg.constraint_degree >= 3, "Constraint degree must be >= 3");
        assert!(cfg.batch_size >= 1, "Batch size must be at least 1");
        assert!(cfg.log_inv_rate >= 1, "Log inverse rate must be > 0 (rate < 1.0)");
        assert!(cfg.num_variables <= 31, "num_variables must fit a u32 dimension");
        assert!(m >= 1, "Must have at least 1 iteration");
        assert!(cfg.folding_factors.iter().all(|&k| k >= 1), "Every folding factor must be >= 1 to reduce degree");

        // Ensure the final polynomial does not end up with a negative number
        // of variables: m₀ >= Σᵢ kᵢ.
        let total_reduction: u32 = cfg.folding_factors.iter().sum();
        assert!(
            total_reduction <= cfg.num_variables,
            "Reducing {} variables by {total_reduction} (sum of folding factors) leaves a negative number of variables",
            cfg.num_variables,
        );

        // Compute the per-iteration log-degree m_i and log-inverse-rate μᵢ.
        //
        // Recurrence:
        //   m_{i+1}  = m_i - k_i        (folding by 2^{k_i})
        //   μ_{i+1}  = μ_i + (k_i - 1)  (domain halves; degree drops by 2^{k_i})
        let mut log_dimensions = Vec::with_capacity(m + 1);
        let mut log_inv_rates = Vec::with_capacity(m + 1);
        log_dimensions.push(cfg.num_variables);
        log_inv_rates.push(cfg.log_inv_rate);
        for i in 0..m {
            let k_i = cfg.folding_factors[i];
            log_dimensions.push(log_dimensions[i] - k_i);
            log_inv_rates.push(log_inv_rates[i] + (k_i - 1));
        }

        let empty = WhirSecurityParams {
            num_queries: vec![0; m],
            num_ood_samples: vec![0; m - 1],
            grinding_bits_batching: 0,
            grinding_bits_folding: cfg.folding_factors.iter().map(|&k| vec![0u32; k as usize]).collect(),
            grinding_bits_queries: vec![0; m],
            grinding_bits_ood: vec![0; m - 1],
        };

        Self { cfg, log_dimensions, log_inv_rates, num_iterations: m, sec_params: empty, alpha: 0.0 }
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

    /// Per-iteration query split: tᵢ = ⌈(target − g) / −log2(1 − δᵢ)⌉, at
    /// iteration i's rate 2^{-μᵢ}. Later iterations have lower rates (larger
    /// δᵢ), so they need fewer queries.
    fn compute_security_params(&self, regime: &dyn ProximityGapsRegime) -> WhirSecurityParams {
        let m = self.num_iterations;

        let mut num_queries = Vec::with_capacity(m);
        for i in 0..m {
            let rate = f64::exp2(-(self.log_inv_rates[i] as f64));
            let bits_per_query = -(1.0 - regime.proximity_parameter(&rate)).log2();

            let needed_from_queries = self.cfg.target_security_bits as f64 - self.cfg.grinding_bits as f64;
            let t = if needed_from_queries > 0.0 {
                (needed_from_queries / bits_per_query).ceil() as u64
            } else {
                1 // Need at least 1 query
            };
            num_queries.push(t);
        }

        WhirSecurityParams {
            num_queries,
            num_ood_samples: vec![1; m - 1],
            grinding_bits_batching: 0,
            grinding_bits_folding: self.cfg.folding_factors.iter().map(|&k| vec![0u32; k as usize]).collect(),
            grinding_bits_queries: vec![self.cfg.grinding_bits; m],
            grinding_bits_ood: vec![0u32; m - 1],
        }
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
        let m = self.num_iterations;
        let mut out = Vec::new();

        // Batching step.
        if self.cfg.batch_size > 1 {
            out.push(("batching".to_string(), bits_of_security_from_log2_error(self.batch_error(regime, sec))));
        }

        // Initial iteration (i=0): only folding (sumcheck), no OOD/Shift.
        for s in 1..=self.cfg.folding_factors[0] {
            out.push((
                format!("fold(i=0,s={s})"),
                bits_of_security_from_log2_error(self.fold_error(regime, sec, 0, s)),
            ));
        }

        // Main loop (i = 1 to M-1): OOD, shift, and folding errors.
        for i in 1..m {
            out.push((format!("OOD(i={i})"), bits_of_security_from_log2_error(self.ood_error(regime, sec, i))));
            out.push((format!("Shift(i={i})"), bits_of_security_from_log2_error(self.shift_error(regime, sec, i))));

            // SumCheck folding errors for this iteration.
            for s in 1..=self.cfg.folding_factors[i] {
                out.push((
                    format!("fold(i={i},s={s})"),
                    bits_of_security_from_log2_error(self.fold_error(regime, sec, i, s)),
                ));
            }
        }

        // Final error.
        out.push(("fin".to_string(), bits_of_security_from_log2_error(self.query_error(regime, sec, m - 1))));

        out
    }

    // The error functions below return log2(ε), NOT ε: the query terms
    // (1 − δ)^t can drop far below f64's smallest subnormal (uniform query
    // schedules spend hundreds of high-value queries in late iterations), so
    // the arithmetic stays in log2 space throughout. Grinding is a plain
    // subtraction there.

    /// log2 of the batching error; depends on the batching strategy.
    fn batch_error(&self, regime: &dyn ProximityGapsRegime, sec: &WhirSecurityParams) -> f64 {
        let (rate, dimension) = self.code_for(0, 0);
        let epsilon = match self.cfg.batching {
            Batching::Powers => regime.error_powers(&rate, dimension, self.cfg.batch_size),
            Batching::Multilinear => regime.error_multilinear(&rate, dimension, self.cfg.batch_size),
            Batching::Affine => regime.error_linear(&rate, dimension),
        };
        epsilon.log2() - sec.grinding_bits_batching as f64
    }

    /// log2 of the folding error for iteration i and round s (1-indexed round).
    fn fold_error(
        &self,
        regime: &dyn ProximityGapsRegime,
        sec: &WhirSecurityParams,
        iteration: usize,
        round: u32,
    ) -> f64 {
        assert!(iteration < self.num_iterations, "Iteration index out of bounds");
        let ff = self.cfg.folding_factors[iteration];
        assert!(
            round >= 1 && round <= ff,
            "Round index out of bounds for iteration {iteration}: got {round}, expected 1..={ff}"
        );

        // The error has two terms: ε^fold_{i,s} = d·ℓ_{i,s-1}/|F| + err_powers(C^{i,s}, 2)

        // The first term is d·ℓ_{i,s-1}/|F|, where ℓ_{i,s-1} is the max list
        // size at the previous round's code.
        let list_size = self.list_size_for(regime, iteration, round - 1);
        let first = (self.cfg.constraint_degree as f64 * list_size).log2() - self.cfg.field_size.log2();

        // The second term is the batching error for powers coefficients at
        // the current round's code.
        let (rate, dimension) = self.code_for(iteration, round);
        let second = regime.error_powers(&rate, dimension, 2).log2();

        log2_add(first, second) - sec.grinding_bits_folding[iteration][(round - 1) as usize] as f64
    }

    /// log2 of the OOD error for iteration i.
    fn ood_error(&self, regime: &dyn ProximityGapsRegime, sec: &WhirSecurityParams, iteration: usize) -> f64 {
        assert!(iteration >= 1 && iteration < self.num_iterations, "Iteration index out of bounds");

        // ε^out_i = ℓ_{i,0}² · (2^{mᵢ} / (2|F|))^w.
        let list_size = self.list_size_for(regime, iteration, 0);
        let num_ood_samples = sec.num_ood_samples[iteration - 1];
        let log2_sample = self.log_dimensions[iteration] as f64 - 1.0 - self.cfg.field_size.log2();
        let epsilon = 2.0 * list_size.log2() + num_ood_samples as f64 * log2_sample;

        epsilon - sec.grinding_bits_ood[iteration - 1] as f64
    }

    /// log2 of the shift error for iteration i.
    fn shift_error(&self, regime: &dyn ProximityGapsRegime, sec: &WhirSecurityParams, iteration: usize) -> f64 {
        assert!(iteration >= 1 && iteration < self.num_iterations, "Iteration index out of bounds");

        // ε^shift_i = (1 − δ_{i-1})^{t_{i-1}} + ℓ_{i,0}·(t_{i-1}+1)/|F|
        let t = sec.num_queries[iteration - 1];

        // First term is (1 − δ_{i-1})^{t_{i-1}}
        let pp = self.proximity_parameter_for(regime, iteration - 1);
        let first = t as f64 * (1.0 - pp).log2();

        // Second term is ℓ_{i,0}·(t_{i-1}+1)/|F|
        let list_size = self.list_size_for(regime, iteration, 0);
        let second = (list_size * (t + 1) as f64).log2() - self.cfg.field_size.log2();

        log2_add(first, second) - sec.grinding_bits_queries[iteration - 1] as f64
    }

    /// log2 of the query error for iteration i.
    fn query_error(&self, regime: &dyn ProximityGapsRegime, sec: &WhirSecurityParams, iteration: usize) -> f64 {
        assert!(iteration < self.num_iterations, "Iteration index out of bounds");

        // (1 − δᵢ)^{tᵢ}
        let t = sec.num_queries[iteration];
        let pp = self.proximity_parameter_for(regime, iteration);
        t as f64 * (1.0 - pp).log2() - sec.grinding_bits_queries[iteration] as f64
    }

    /// The code `C_RS^{i,s} = RS[F, L_i^{(2^s)}, m_i − s]` as a
    /// `(rate, dimension)` pair: rate `2^{-μᵢ}`, dimension `2^{mᵢ−s}`.
    fn code_for(&self, iteration: usize, round: u32) -> (f64, u32) {
        assert!(iteration < self.num_iterations, "Iteration index out of bounds");
        assert!(
            round <= self.cfg.folding_factors[iteration],
            "Round index out of bounds for iteration {iteration}: got {round}, expected 0..={}",
            self.cfg.folding_factors[iteration]
        );

        let log_dimension = self.log_dimensions[iteration] as i64 - round as i64;
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
        assert!(iteration < self.num_iterations, "Iteration index out of bounds");

        let mut min_pp = 1.0f64;
        for s in 0..=self.cfg.folding_factors[iteration] {
            let (rate, _) = self.code_for(iteration, s);
            min_pp = min_pp.min(regime.proximity_parameter(&rate));
        }
        min_pp
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

    pub fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    pub fn config(&self) -> &WhirConfig {
        &self.cfg
    }

    /// Description of the parameters of the PCS (Markdown code block).
    pub fn parameter_summary(&self) -> String {
        let params: Vec<(&str, String)> = vec![
            ("num_variables m0", self.cfg.num_variables.to_string()),
            ("log_inv_rate mu0", self.cfg.log_inv_rate.to_string()),
            ("folding_factors (bits)", format!("{:?}", self.cfg.folding_factors)),
            ("batch_size", self.cfg.batch_size.to_string()),
            ("batching", self.cfg.batching.to_string()),
            ("constraint_degree", self.cfg.constraint_degree.to_string()),
            ("regime", format!("{:?} (alpha = {})", self.cfg.regime, self.alpha)),
            ("target_security_bits", self.cfg.target_security_bits.to_string()),
            ("num_queries (deduced)", format!("{:?}", self.sec_params.num_queries)),
            ("num_ood_samples (deduced)", format!("{:?}", self.sec_params.num_ood_samples)),
            ("grinding_queries (deduced)", format!("{:?}", self.sec_params.grinding_bits_queries)),
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

/// Bits of security contributed by a single WHIR query at rate `2^-log_inv_rate`
/// under a regime with no gap widening (`alpha = 0`). Used to size uniform
/// query schedules externally (see `ml_params`).
pub fn whir_query_bits(field_size: f64, log_inv_rate: u32, regime: DecodingRegime) -> f64 {
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

    fn rate(&self) -> f64 {
        f64::exp2(-(self.cfg.log_inv_rate as f64))
    }

    fn dimension(&self) -> u32 {
        1u32 << self.cfg.num_variables
    }

    fn parameter_summary(&self) -> String {
        Whir::parameter_summary(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::goldilocks_safe_extension_field_size;
    use super::*;

    fn riscv_config(regime: DecodingRegime) -> WhirConfig {
        WhirConfig {
            field_size: goldilocks_safe_extension_field_size(),
            num_variables: 22,
            log_inv_rate: 4,
            folding_factors: vec![4, 4, 4, 4, 4],
            batch_size: 200,
            batching: Batching::Powers,
            constraint_degree: 8,
            grinding_bits: 0,
            target_security_bits: 116,
            regime,
        }
    }

    /// The riscv schedule as pinned by the prover (externally fixed).
    fn whir_riscv_params(regime: DecodingRegime) -> Whir {
        Whir::with_security_params(
            riscv_config(regime),
            WhirSecurityParams {
                num_queries: vec![55, 31, 22, 17, 14],
                num_ood_samples: vec![1, 1, 1, 1],
                grinding_bits_batching: 21,
                grinding_bits_folding: vec![
                    vec![16, 14, 12, 10],
                    vec![14, 12, 10, 8],
                    vec![17, 15, 13, 11],
                    vec![19, 17, 15, 13],
                    vec![22, 20, 18, 16],
                ],
                grinding_bits_queries: vec![22, 22, 20, 19, 17],
                grinding_bits_ood: vec![0, 0, 0, 0],
            },
        )
    }

    #[test]
    fn whir_security_levels_are_sane() {
        for kind in [DecodingRegime::Jbr, DecodingRegime::Udr] {
            let whir = whir_riscv_params(kind);
            let levels = whir.security_levels();
            // Expected component set: batching + fin + per-iteration OOD/Shift/fold.
            assert!(levels.iter().any(|(k, _)| k == "batching"));
            assert!(levels.iter().any(|(k, _)| k == "fin"));
            assert!(levels.iter().any(|(k, _)| k == "OOD(i=1)"));
            // Every component must give positive security.
            for (name, bits) in &levels {
                assert!(*bits > 0, "{kind:?} component {name} gave {bits} bits");
            }
            assert!(whir.total_security_bits() > 0);
        }
    }

    /// Golden levels cross-checked against soundcalc's formulas with the pil2
    /// gap (η = 1/300 for JBR, δ/20 for UDR). In both regimes the binding
    /// component is `fin`.
    #[test]
    fn whir_golden_levels() {
        let whir = whir_riscv_params(DecodingRegime::Jbr);
        let levels = whir.security_levels();
        let level = |name: &str| levels.iter().find(|(k, _)| k == name).unwrap().1;
        assert_eq!(level("batching"), 147);
        assert_eq!(level("fold(i=0,s=1)"), 151);
        assert_eq!(level("OOD(i=1)"), 153);
        assert_eq!(level("Shift(i=1)"), 130);
        assert_eq!(level("fin"), 116);
        assert_eq!(whir.total_security_bits(), 116);

        assert_eq!(whir_riscv_params(DecodingRegime::Udr).total_security_bits(), 29);
    }

    /// The solver meets its target and sizes queries per iteration:
    /// later iterations have lower rates (larger δ), so fewer queries.
    #[test]
    fn whir_optimal_query_params() {
        let whir = Whir::new(WhirConfig { target_security_bits: 100, ..riscv_config(DecodingRegime::Jbr) });
        assert!(whir.total_security_bits() >= 100);
        let queries = &whir.security_params().num_queries;
        assert_eq!(queries.len(), 5);
        assert!(queries.windows(2).all(|w| w[0] >= w[1]), "queries must not increase: {queries:?}");
    }
}