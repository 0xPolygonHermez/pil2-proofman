use rug::{Float, ops::Pow};

use super::common::{apply_grinding, hpf, security_bits_from_error};
use super::regimes::{CodeParams, DecodingRegime};

/// Parameters for WHIR.
#[derive(Debug, Clone)]
pub struct WhirParams {
    /// Base-2 logarithm of the field size `|F|` (exact to the calculator's precision).
    pub field_size_bits: Float,
    /// `log2(1/ρ₀)` — initial inverse rate.
    pub log_inv_rate: u32,
    /// `m₀` — initial number of variables (log2 of the degree/message size).
    pub log_degree: u32,
    /// Per-iteration folding factors `kᵢ` (length `M`).
    pub folding_factors: Vec<u32>,
    /// Number of polynomials batched into `Φ`.
    pub batch_size: u64,
    /// Powers batching (`true`) vs affine/linear (`false`).
    pub power_batching: bool,
    /// Constraint degree `d` (≥ 3 per Construction 5.1).
    pub constraint_degree: u64,
    /// Per-iteration query counts `tᵢ` (length `M`).
    pub num_queries: Vec<u64>,
    /// Per-iteration OOD sample counts `wᵢ` (length `M-1`).
    pub num_ood_samples: Vec<u64>,
    /// Grinding bits for the batching phase.
    pub grinding_batching_phase: u32,
    /// Grinding bits per folding round, `[iteration][round]` (length `M`, each `kᵢ`).
    pub grinding_bits_folding: Vec<Vec<u32>>,
    /// Grinding bits per iteration's query phase (length `M`).
    pub grinding_bits_queries: Vec<u32>,
    /// Grinding bits per iteration's OOD phase (length `M-1`).
    pub grinding_bits_ood: Vec<u32>,
}

impl WhirParams {
    fn num_iterations(&self) -> usize {
        self.folding_factors.len()
    }

    /// `log_degrees[i] = mᵢ` and `log_inv_rates[i] = μᵢ` (length `M+1`). The
    /// domain halves once per iteration while the degree drops by `2^kᵢ`, so
    /// `μ_{i+1} = μᵢ + (kᵢ − 1)` (rate `ρ_{i+1} = 2^{1−kᵢ}·ρᵢ`).
    fn schedules(&self) -> (Vec<i64>, Vec<i64>) {
        let mut m = vec![self.log_degree as i64];
        let mut mu = vec![self.log_inv_rate as i64];
        for &k in &self.folding_factors {
            m.push(m.last().unwrap() - k as i64);
            mu.push(mu.last().unwrap() + (k as i64 - 1));
        }
        (m, mu)
    }

    /// The code `C_RS^{i,s} = RS[F, L_i^{(2^s)}, m_i − s]` as `CodeParams`
    /// (rate `2^{-μᵢ}`, dimension `2^{mᵢ−s}`). `alpha = 0` (no FRI gap widening).
    fn code(&self, m: &[i64], mu: &[i64], iteration: usize, round: u32) -> CodeParams {
        let rate = 2f64.powi(-(mu[iteration] as i32));
        let dimension = 1u64 << (m[iteration] - round as i64);
        CodeParams::new(&self.field_size_bits, dimension, rate, 0.0, 1)
    }

    fn eps_batching(&self, m: &[i64], mu: &[i64], regime: DecodingRegime) -> Float {
        let cp = self.code(m, mu, 0, 0);
        let e = if self.power_batching {
            regime.calculate_powers_error(&cp, self.batch_size)
        } else {
            regime.calculate_linear_error(&cp)
        };
        apply_grinding(&e, self.grinding_batching_phase)
    }

    fn eps_fold(&self, m: &[i64], mu: &[i64], iteration: usize, round: u32, regime: DecodingRegime) -> Float {
        // d·ℓ_{i,s-1}/|F|  +  err_powers(C^{i,s}, 2)
        let list = regime.max_list_size(&self.code(m, mu, iteration, round - 1));
        let cp = self.code(m, mu, iteration, round);
        let first = hpf(self.constraint_degree) * &list / &cp.field_size;
        let err_powers = regime.calculate_powers_error(&cp, 2);
        let e = first + err_powers;
        apply_grinding(&e, self.grinding_bits_folding[iteration][(round - 1) as usize])
    }

    fn eps_out(&self, m: &[i64], mu: &[i64], iteration: usize, regime: DecodingRegime) -> Float {
        // ℓ_{i,0}² · (2^{mᵢ} / (2|F|))^w
        let cp = self.code(m, mu, iteration, 0);
        let list = regime.max_list_size(&cp);
        let mi = m[iteration];
        let w = self.num_ood_samples[iteration - 1];
        let two_mi = hpf(hpf(2).pow(mi as u32));
        let base = two_mi / (hpf(2) * &cp.field_size);
        let e = list.clone() * &list * hpf(base.pow(w as u32));
        apply_grinding(&e, self.grinding_bits_ood[iteration - 1])
    }

    fn eps_shift(&self, m: &[i64], mu: &[i64], iteration: usize, regime: DecodingRegime) -> Float {
        // (1 − δ_{i-1})^t  +  ℓ_{i,0}·(t+1)/|F|
        let t = self.num_queries[iteration - 1];
        let delta = regime.proximity_parameter(&self.code(m, mu, iteration - 1, 0));
        let cp = self.code(m, mu, iteration, 0);
        let list = regime.max_list_size(&cp);
        let term1 = hpf((hpf(1) - &delta).pow(t as u32));
        let term2 = list * hpf(t + 1) / &cp.field_size;
        let e = term1 + term2;
        apply_grinding(&e, self.grinding_bits_queries[iteration - 1])
    }

    fn eps_query(&self, m: &[i64], mu: &[i64], iteration: usize, regime: DecodingRegime) -> Float {
        // (1 − δᵢ)^t
        let t = self.num_queries[iteration];
        let delta = regime.proximity_parameter(&self.code(m, mu, iteration, 0));
        let e = hpf((hpf(1) - &delta).pow(t as u32));
        apply_grinding(&e, self.grinding_bits_queries[iteration])
    }

    /// Per-component PCS security levels (bits), keyed by component name — the
    /// analog of soundcalc's `WHIR.get_pcs_security_levels`.
    pub fn pcs_security_levels(&self, regime: DecodingRegime) -> Vec<(String, i64)> {
        let (m, mu) = self.schedules();
        let mm = self.num_iterations();
        let mut out = Vec::new();

        if self.batch_size > 1 {
            out.push(("batching".to_string(), security_bits_from_error(&self.eps_batching(&m, &mu, regime))));
        }
        for s in 1..=self.folding_factors[0] {
            out.push((format!("fold(i=0,s={s})"), security_bits_from_error(&self.eps_fold(&m, &mu, 0, s, regime))));
        }
        for i in 1..mm {
            out.push((format!("OOD(i={i})"), security_bits_from_error(&self.eps_out(&m, &mu, i, regime))));
            out.push((format!("Shift(i={i})"), security_bits_from_error(&self.eps_shift(&m, &mu, i, regime))));
            for s in 1..=self.folding_factors[i] {
                out.push((
                    format!("fold(i={i},s={s})"),
                    security_bits_from_error(&self.eps_fold(&m, &mu, i, s, regime)),
                ));
            }
        }
        out.push(("fin".to_string(), security_bits_from_error(&self.eps_query(&m, &mu, mm - 1, regime))));
        out
    }

    /// Minimum PCS security over all components (bits) — the total PCS soundness.
    pub fn min_pcs_bits(&self, regime: DecodingRegime) -> i64 {
        self.pcs_security_levels(regime).into_iter().map(|(_, b)| b).min().unwrap_or(0)
    }
}

/// Bits of soundness per WHIR query, `-log2(1 - δ)`, at a code of inverse rate
/// `2^log_inv_rate`. Iteration 0 (the highest rate) gives the smallest `δ` and
/// so the query-hungriest bound, which sets a uniform per-block query count:
/// `n_queries = ceil(target_bits / whir_query_bits(...))`.
pub fn whir_query_bits(field_size_bits: &Float, log_inv_rate: u32, regime: DecodingRegime) -> f64 {
    let rate = 2f64.powi(-(log_inv_rate as i32));
    let cp = CodeParams::new(field_size_bits, 2, rate, 0.0, 1);
    let delta = regime.proximity_parameter(&cp);
    let one_minus = hpf(1) - &delta;
    -one_minus.to_f64().log2()
}

/// LogUp-GKR sumcheck soundness error `½·(n+m)·(3(n+m)+1)/|F|`, with
/// `2^n` = alphabet size (`rows_L + rows_T`) and `m = log2(num_lookups)`
/// (soundcalc `lookups/gkr.py`). Returns bits of security.
pub fn logup_gkr_error(field_size_bits: &Float, alphabet_size: f64, num_lookups: f64) -> i64 {
    let n = alphabet_size.log2();
    let mm = num_lookups.log2();
    let nm = n + mm;
    let numer = 0.5 * nm * (3.0 * nm + 1.0);
    // bits = -log2(numer / |F|) = field_size_bits - log2(numer)
    let bits = field_size_bits.to_f64() - numer.log2();
    bits.floor() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::goldilocks_safe_extension_field_size_bits;

    fn whir_riscv_params() -> WhirParams {
        WhirParams {
            field_size_bits: goldilocks_safe_extension_field_size_bits(),
            log_inv_rate: 4,
            log_degree: 22,
            folding_factors: vec![4, 4, 4, 4, 4],
            batch_size: 200,
            power_batching: true,
            constraint_degree: 8,
            num_queries: vec![55, 31, 22, 17, 14],
            num_ood_samples: vec![1, 1, 1, 1],
            grinding_batching_phase: 21,
            grinding_bits_folding: vec![
                vec![16, 14, 12, 10],
                vec![14, 12, 10, 8],
                vec![17, 15, 13, 11],
                vec![19, 17, 15, 13],
                vec![22, 20, 18, 16],
            ],
            grinding_bits_queries: vec![22, 22, 20, 19, 17],
            grinding_bits_ood: vec![0, 0, 0, 0],
        }
    }

    #[test]
    fn whir_security_levels_are_sane() {
        for regime in [DecodingRegime::Jbr, DecodingRegime::Udr] {
            let levels = whir_riscv_params().pcs_security_levels(regime);
            // Expected component set: batching + fin + per-iteration OOD/Shift/fold.
            assert!(levels.iter().any(|(k, _)| k == "batching"));
            assert!(levels.iter().any(|(k, _)| k == "fin"));
            assert!(levels.iter().any(|(k, _)| k == "OOD(i=1)"));
            // Every component must give positive security.
            for (name, bits) in &levels {
                assert!(*bits > 0, "{regime:?} component {name} gave {bits} bits");
            }
            assert!(whir_riscv_params().min_pcs_bits(regime) > 0);
        }
    }
}
