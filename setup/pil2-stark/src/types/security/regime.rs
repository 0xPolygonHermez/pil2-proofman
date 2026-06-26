use rug::ops::Pow;
use rug::Float;

use super::{hpf, hpf_from_f64, truncate_decimal_places, PREC};

// ---------------------------------------------------------------------------
// Decoding regimes
// ---------------------------------------------------------------------------

pub(super) struct RegimeParams {
    pub(super) field_size: Float,
    pub(super) dimension: Float,
    pub(super) rate: Float,
    pub(super) codeword_length: Float,
    pub(super) augmented_rate: Float,
    pub(super) alpha: f64,
}

impl RegimeParams {
    pub(super) fn new(field_size: Float, dimension: u64, rate: f64, alpha: f64, n_opening_points: u64) -> Self {
        let dim_f = hpf(dimension);
        let rate_f = hpf_from_f64(rate);
        let codeword_length = Float::with_val(PREC, &dim_f / &rate_f);
        let augmented_rate = {
            let dim_plus_open = Float::with_val(PREC, &dim_f + n_opening_points);
            let numer = Float::with_val(PREC, &rate_f * &dim_plus_open);
            Float::with_val(PREC, numer / &dim_f)
        };
        RegimeParams { field_size, dimension: dim_f, rate: rate_f, codeword_length, augmented_rate, alpha }
    }
}

// ---------------------------------------------------------------------------
// JBR (Johnson Bound Regime)
// ---------------------------------------------------------------------------

#[allow(clippy::upper_case_acronyms)]
pub(super) struct JBR<'a> {
    pub(super) params: &'a RegimeParams,
}

impl<'a> JBR<'a> {
    pub(super) fn new(params: &'a RegimeParams) -> Self {
        JBR { params }
    }

    fn sqrt_rate(&self) -> Float {
        self.params.rate.clone().sqrt()
    }

    fn max_decoding_radius(&self) -> Float {
        let sr = self.sqrt_rate();
        Float::with_val(PREC, hpf(1) - &sr)
    }

    fn min_decoding_radius(&self) -> Float {
        let one_minus_rate = Float::with_val(PREC, hpf(1) - &self.params.rate);
        Float::with_val(PREC, one_minus_rate / hpf(2))
    }

    fn gap(&self) -> Float {
        let base_correction = Float::with_val(PREC, hpf(1) / hpf(300));
        // Match JS Decimal.js behavior: convert alpha via string repr to avoid
        // f64 binary approximation errors (e.g. 1.6_f64 is not exactly 1.6).
        let alpha_plus_one = {
            let alpha_str = format!("{}", 1.0 + self.params.alpha);
            Float::with_val(PREC, Float::parse(&alpha_str).unwrap())
        };
        let raw = Float::with_val(PREC, &base_correction * &alpha_plus_one);
        let gap = truncate_decimal_places(&raw, 20);
        // Assert: minDecodingRadius < maxDecodingRadius - gap
        let max_minus_gap = Float::with_val(PREC, &self.max_decoding_radius() - &gap);
        assert!(
            self.min_decoding_radius() < max_minus_gap,
            "Gap must keep minDecodingRadius < maxDecodingRadius - gap in JBR"
        );
        gap
    }

    fn proximity_parameter(&self) -> Float {
        Float::with_val(PREC, &self.max_decoding_radius() - &self.gap())
    }

    // Inherent form retained for parity with the JS reference; the live path
    // uses the `DecodingRegime::max_list_size` trait method.
    #[allow(dead_code)]
    pub(super) fn max_list_size(&self) -> Float {
        let sqrt_aug_rate = self.params.augmented_rate.clone().sqrt();
        let two_gap = Float::with_val(PREC, hpf(2) * &self.gap());
        let denom = Float::with_val(PREC, &two_gap * &sqrt_aug_rate);
        Float::with_val(PREC, hpf(1) / denom)
    }

    fn multiplicity(&self) -> Float {
        let m_raw = Float::with_val(PREC, &self.sqrt_rate() / &self.gap());
        let m_ceil = m_raw.ceil();
        let three = hpf(3);
        if m_ceil > three {
            m_ceil
        } else {
            three
        }
    }

    fn calculate_linear_error(&self) -> Float {
        let n = Float::with_val(PREC, &self.params.dimension / &self.params.rate);
        let m = self.multiplicity();

        let m_shifted = Float::with_val(PREC, &m + 0.5_f64);
        let m5 = Float::with_val(PREC, m_shifted.clone().pow(5));
        let term1 = Float::with_val(PREC, &m5 * hpf(2));
        let m_times_3 = Float::with_val(PREC, &m_shifted * hpf(3));
        let term2 = Float::with_val(PREC, &m_times_3 * &self.params.rate);
        let sum_terms = Float::with_val(PREC, &term1 + &term2);
        let numerator = Float::with_val(PREC, &sum_terms * &n);

        let three_rate = Float::with_val(PREC, hpf(3) * &self.params.rate);
        let three_rate_sqrt = Float::with_val(PREC, &three_rate * &self.sqrt_rate());
        let denominator = Float::with_val(PREC, &three_rate_sqrt * &self.params.field_size);

        Float::with_val(PREC, numerator / denominator)
    }

    fn calculate_powers_error(&self, n_functions: u64) -> Float {
        let linear = self.calculate_linear_error();
        Float::with_val(PREC, &linear * (n_functions - 1))
    }
}

// ---------------------------------------------------------------------------
// UDR (Unique Decoding Regime)
// ---------------------------------------------------------------------------

#[allow(clippy::upper_case_acronyms)]
pub(super) struct UDR<'a> {
    pub(super) params: &'a RegimeParams,
}

impl<'a> UDR<'a> {
    pub(super) fn new(params: &'a RegimeParams) -> Self {
        UDR { params }
    }

    fn max_decoding_radius(&self) -> Float {
        let one_minus_rate = Float::with_val(PREC, hpf(1) - &self.params.rate);
        Float::with_val(PREC, one_minus_rate / hpf(2))
    }

    fn gap(&self) -> Float {
        // In the JS source, getOptimalFRIQueryParams wraps fieldSize as a
        // Decimal before passing it into the UDR constructor.  The UDR
        // proximity-parameter getter then compares `this.fieldSize >= 1n << 150n`
        // (Decimal vs BigInt), which always evaluates to false in JS.
        // The correction therefore always takes the `rate / 20` branch.
        // We replicate this exact behavior for output-identical results.
        Float::with_val(PREC, &self.params.rate / hpf(20))
    }

    fn proximity_parameter(&self) -> Float {
        let correction = self.gap();
        let pp = Float::with_val(PREC, &self.max_decoding_radius() - &correction);
        assert!(pp > 0.0, "Proximity parameter must be positive in UDR");
        pp
    }

    /// In the unique-decoding regime the list size is 1 (unique codeword).
    /// Inherent form retained for parity; live path uses the trait method.
    #[allow(dead_code)]
    fn max_list_size(&self) -> Float {
        hpf(1)
    }

    fn calculate_linear_error(&self) -> Float {
        Float::with_val(PREC, &self.params.codeword_length / &self.params.field_size)
    }

    fn calculate_powers_error(&self, n_functions: u64) -> Float {
        let linear = self.calculate_linear_error();
        Float::with_val(PREC, &linear * (n_functions - 1))
    }
}

/// Trait to abstract over JBR and UDR for FRI calculation.
pub(super) trait DecodingRegime {
    fn proximity_parameter(&self) -> Float;
    fn gap(&self) -> Float;
    fn calculate_powers_error(&self, n_functions: u64) -> Float;
    fn max_list_size(&self) -> Float;
    /// Idealised decoding radius `delta = 1 - sqrt(rho)` (JBR) or `(1-rho)/2`
    /// (UDR), with no gap correction. STIR's repetition formula is derived
    /// against this radius.
    fn max_decoding_radius(&self) -> Float;
}

impl<'a> DecodingRegime for JBR<'a> {
    fn proximity_parameter(&self) -> Float {
        self.proximity_parameter()
    }
    fn gap(&self) -> Float {
        self.gap()
    }
    fn calculate_powers_error(&self, n_functions: u64) -> Float {
        self.calculate_powers_error(n_functions)
    }
    fn max_list_size(&self) -> Float {
        let sqrt_aug_rate = self.params.augmented_rate.clone().sqrt();
        let two_gap = Float::with_val(PREC, hpf(2) * &self.gap());
        let denom = Float::with_val(PREC, &two_gap * &sqrt_aug_rate);
        Float::with_val(PREC, hpf(1) / denom)
    }
    fn max_decoding_radius(&self) -> Float {
        // JBR: 1 - sqrt(rate)
        let sr = self.params.rate.clone().sqrt();
        Float::with_val(PREC, hpf(1) - &sr)
    }
}

impl<'a> DecodingRegime for UDR<'a> {
    fn proximity_parameter(&self) -> Float {
        self.proximity_parameter()
    }
    fn gap(&self) -> Float {
        self.gap()
    }
    fn calculate_powers_error(&self, n_functions: u64) -> Float {
        self.calculate_powers_error(n_functions)
    }
    fn max_list_size(&self) -> Float {
        hpf(1)
    }
    fn max_decoding_radius(&self) -> Float {
        // UDR: (1 - rate)/2
        let one_minus_rate = Float::with_val(PREC, hpf(1) - &self.params.rate);
        Float::with_val(PREC, one_minus_rate / hpf(2))
    }
}

/// Build a boxed decoding regime by name for the given round parameters.
#[allow(dead_code)]
pub(super) fn build_regime<'a>(rp: &'a RegimeParams, regime_name: &str) -> Box<dyn DecodingRegime + 'a> {
    match regime_name {
        "JBR" => Box::new(JBR::new(rp)),
        "UDR" => Box::new(UDR::new(rp)),
        _ => panic!("Unknown decoding regime: {regime_name}. Supported: JBR, UDR"),
    }
}
