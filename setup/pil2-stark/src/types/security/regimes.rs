use rug::{Float, ops::Pow};

use super::common::{field_size_from_bits, hpf, truncate_decimal_places};

// ---------------------------------------------------------------------------
// Code Parameters
// ---------------------------------------------------------------------------

pub(crate) struct CodeParams {
    /// Field size |F|.
    pub(crate) field_size: Float,
    #[allow(unused)]
    /// Dimension k of the code.
    pub(crate) dimension: Float,
    /// Length n of the code.
    pub(crate) length: Float,
    /// Code rate ρ = k/n.
    pub(crate) rate: Float,
    /// Minimum distance δ = 1 - ρ.
    pub(crate) minimum_distance: Float,
    /// Augmented rate ρ' = ρ * (k + n_opening_points) / k.
    pub(crate) augmented_rate: Float,
    /// Alpha parameter for the code, used in security calculations.
    pub(crate) alpha: f64,
}

impl CodeParams {
    pub(crate) fn new(field_size_bits: &Float, dimension: u64, rate: f64, alpha: f64, n_opening_points: u64) -> Self {
        let field_size = field_size_from_bits(field_size_bits);
        let dim_f = hpf(dimension);
        let rate_f = hpf(rate);
        let length = dim_f.clone() / &rate_f;
        let minimum_distance = hpf(1) - &rate_f;
        let augmented_rate = {
            let dim_plus_open = &dim_f + hpf(n_opening_points);
            let numer = rate_f.clone() * &dim_plus_open;
            numer / &dim_f
        };
        CodeParams { field_size, dimension: dim_f, length, rate: rate_f, minimum_distance, augmented_rate, alpha }
    }

    /// The square root of the code rate ρ.
    fn sqrt_rate(&self) -> Float {
        self.rate.clone().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Decoding Regimes
// ---------------------------------------------------------------------------

/// A Reed–Solomon decoding regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodingRegime {
    /// Johnson bound regime (JBR).
    Jbr,
    /// Unique decoding regime (UDR).
    Udr,
}

impl DecodingRegime {
    // --- regime-specific ---

    /// The decoding radius: the largest fraction of errors that can be corrected.
    pub(crate) fn decoding_radius(&self, cp: &CodeParams) -> Float {
        match self {
            // 1 - √ρ, where ρ is the code rate (= 1 - √(1-δ)).
            DecodingRegime::Jbr => hpf(1) - cp.sqrt_rate(),
            // δ/2, where δ is the minimum distance of the code.
            DecodingRegime::Udr => cp.minimum_distance.clone() / hpf(2),
        }
    }

    /// The gap parameter η between the maximum decoding radius and the proximity parameter.
    pub(crate) fn gap(&self, cp: &CodeParams) -> Float {
        match self {
            DecodingRegime::Jbr => {
                // Something small that makes γ ∈ [δ/2, 1 - √(1-δ) - η] (see [BCHKS25] Corollary 1.4),
                // where δ = 1 - ρ is the minimum distance of the code.
                let base_correction = hpf(1) / hpf(300);
                let alpha_str = format!("{}", 1.0 + cp.alpha);
                let alpha_plus_one = hpf(Float::parse(&alpha_str).unwrap());
                let raw = base_correction * &alpha_plus_one;
                let gap = truncate_decimal_places(&raw, 20);

                // γ ∈ [δ/2, 1 - √(1-δ) - η] <=> 1 - √(1-δ) - η > δ/2 <=> η < 1 - √(1-δ) - δ/2
                let delta_half = &cp.minimum_distance / hpf(2);
                assert!(
                    gap < self.decoding_radius(cp) - delta_half,
                    "Gap must be smaller than 1 - √(1-δ) - δ/2 in JBR"
                );
                gap
            }
            DecodingRegime::Udr => {
                // Something small that makes γ ∈ [δ/3, δ/2 - η] (see [BCHKS25] Corollary 1.4).
                let gap = &cp.rate / hpf(20);

                // γ ∈ [δ/3, δ/2 - η] <=> δ/2 - η > δ/3 <=> η < δ/6
                assert!(gap < cp.minimum_distance.clone() / hpf(6), "Gap must be smaller than minimum δ/6 in UDR");
                gap
            }
        }
    }

    /// The (mutual) correlated-agreement error for a single linear combination.
    pub(crate) fn calculate_linear_error(&self, cp: &CodeParams) -> Float {
        match self {
            DecodingRegime::Jbr => {
                // Theorem 4.2 from [BCHKS25].
                let n = &cp.length;
                let m = jbr_multiplicity(self, cp);
                let rate = &cp.rate;
                let sqrt_rate = cp.sqrt_rate();
                let pp = self.proximity_parameter(cp);

                let m_shifted = m + hpf(0.5);

                // First fraction: (2·(m + 1/2)⁵ + 3·(m + 1/2)·γ·ρ)·n / (3·ρ·√ρ)
                let numerator = (hpf(2) * m_shifted.clone().pow(5) + hpf(3) * &m_shifted * pp * rate) * n;
                let denominator = hpf(3) * rate * &sqrt_rate;
                let first_fraction = numerator / denominator;

                // Second fraction: (m + 1/2) / √ρ
                let second_fraction = m_shifted / &sqrt_rate;

                (first_fraction + second_fraction) / &cp.field_size
            }
            DecodingRegime::Udr => {
                // Obtained from [BCHKS25] Corollary 1.4.
                let pp = self.proximity_parameter(cp);
                let error = pp * &cp.length + hpf(1);
                error / &cp.field_size
            }
        }
    }

    /// Upper bound on the list size ℓ at this code's decoding radius.
    pub(crate) fn max_list_size(&self, cp: &CodeParams) -> Float {
        match self {
            // Reed–Solomon at radius 1 − √ρ − η is (ℓ, ·)-list decodable with
            // ℓ = 1 / (2·η·√ρ)  [BCHKS25].
            DecodingRegime::Jbr => {
                let sqrt_aug_rate = cp.augmented_rate.clone().sqrt();
                let two_gap = hpf(2) * &self.gap(cp);
                hpf(1) / (two_gap * &sqrt_aug_rate)
            }
            // Unique decoding: at most one codeword in the ball.
            DecodingRegime::Udr => hpf(1),
        }
    }

    // --- shared, common to every regime ---

    /// The proximity parameter γ = decoding_radius − gap.
    pub(crate) fn proximity_parameter(&self, cp: &CodeParams) -> Float {
        let pp = self.decoding_radius(cp) - self.gap(cp);
        assert!(pp > 0.0, "Proximity parameter must be positive");
        pp
    }

    /// Batching error for powers coefficients `c_i = γ^i`: `linear · (n − 1)`.
    pub(crate) fn calculate_powers_error(&self, cp: &CodeParams, n_functions: u64) -> Float {
        self.calculate_linear_error(cp) * (n_functions - 1)
    }
}

/// The multiplicity parameter `m` of the Guruswami–Sudan list decoder (JBR):
/// `m = max(⌈√ρ / (2η)⌉, 3)` [BCHKS25] Theorem 4.2. The factor of 2 that the
/// statement of Theorem 4.2 omits is a confirmed typo (see soundcalc).
fn jbr_multiplicity(regime: &DecodingRegime, cp: &CodeParams) -> Float {
    let two_gap = hpf(2) * &regime.gap(cp);
    let m_ceil = (cp.sqrt_rate() / &two_gap).ceil();
    let three = hpf(3);
    if m_ceil > three {
        m_ceil
    } else {
        three
    }
}
