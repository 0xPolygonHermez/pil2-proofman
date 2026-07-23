/// Which decoding regime to instantiate. This is the *kind* of regime, known
/// upfront; the concrete [`ProximityGapsRegime`] also needs the gap-widening
/// factor `alpha`, which is deduced (searched over) by the PCS solvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodingRegime {
    /// Johnson bound regime (JBR).
    Jbr,
    /// Unique decoding regime (UDR).
    Udr,
}

impl DecodingRegime {
    /// Instantiate a concrete regime. `alpha` widens the JBR gap and is
    /// ignored by UDR (whose gap is fixed at δ/20).
    pub fn instantiate(&self, field_size: f64, alpha: f64) -> Box<dyn ProximityGapsRegime> {
        match self {
            DecodingRegime::Jbr => Box::new(JohnsonBoundRegime::new(field_size, alpha)),
            DecodingRegime::Udr => Box::new(UniqueDecodingRegime::new(field_size)),
        }
    }
}

/// A regime for proximity gaps or (mutual) correlated agreement.
/// We only consider Reed-Solomon codes here, of dimension k, size n, and rate k/n.
pub trait ProximityGapsRegime {
    /// Returns the name of the regime.
    fn identifier(&self) -> &'static str;

    /// The field size over which the code is defined.
    fn field_size(&self) -> &f64;

    /// The decoding radius: the largest fraction of errors that can be corrected.
    fn decoding_radius(&self, rate: &f64) -> f64;

    /// The gap between the decoding radius and the proximity parameter.
    fn gap(&self, rate: &f64) -> f64;

    /// Returns the maximum delta for this regime, based on the rate
    /// and the dimension of the code.
    fn proximity_parameter(&self, rate: &f64) -> f64 {
        let pp = self.decoding_radius(rate) - self.gap(rate);
        assert!(pp > 0.0, "Proximity parameter must be positive");
        pp
    }

    /// Returns an upper bound on the list size for this regime.
    fn max_list_size(&self, rate: &f64, dimension: u32) -> u64;

    /// Upper bound on the MCA error for independent coefficients
    /// 1, r_1, ..., r_{batch_size-1}
    /// (batching over affine spaces, BCIKS20 Thm 1.6).
    fn error_linear(&self, rate: &f64, dimension: u32) -> f64;

    /// Upper bound on the MCA error for a random linear combination with
    /// coefficients r^0, r^1, ..., r^{batch_size-1}
    /// (batching over parameterized curves, BCIKS20 Thm 6.2 ).
    fn error_powers(&self, rate: &f64, dimension: u32, batch_size: u64) -> f64 {
        self.error_linear(rate, dimension) * (batch_size as f64 - 1.0)
    }

    /// Upper bound on the MCA error for coefficients eq(r, 0), ..., eq(r, batch_size-1)
    /// (multilinear batching, BCHKS25 §4.1; compare Thms 1.5 and 1.6).
    fn error_multilinear(&self, rate: &f64, dimension: u32, batch_size: u64) -> f64 {
        self.error_linear(rate, dimension) * (batch_size as f64).log2().ceil()
    }
}

pub struct UniqueDecodingRegime {
    /// Field size |F|.
    field_size: f64,
}

impl UniqueDecodingRegime {
    pub fn new(field_size: f64) -> Self {
        Self { field_size }
    }
}

impl ProximityGapsRegime for UniqueDecodingRegime {
    fn identifier(&self) -> &'static str {
        "UDR"
    }

    fn field_size(&self) -> &f64 {
        &self.field_size
    }

    fn decoding_radius(&self, rate: &f64) -> f64 {
        // δ/2, where δ = 1 - ρ is the minimum distance of the code.
        (1.0 - rate) / 2.0
    }

    fn gap(&self, rate: &f64) -> f64 {
        // Something small that makes γ ∈ [δ/3, δ/2 - η] (see [BCHKS25] Corollary 1.4).
        let minimum_distance = 1.0 - rate;
        let gap = minimum_distance / 20.0;

        // γ ∈ [δ/3, δ/2 - η] <=> δ/2 - η > δ/3 <=> η < δ/6
        assert!(gap < minimum_distance / 6.0, "Gap must be smaller than δ/6 in UDR");
        gap
    }

    fn max_list_size(&self, _rate: &f64, _dimension: u32) -> u64 {
        // Unique decoding: at most one codeword in the ball.
        1
    }

    fn error_linear(&self, rate: &f64, dimension: u32) -> f64 {
        // [BCHKS25] Corollary 1.4 with γ = δ/2 − η.
        let pp = self.proximity_parameter(rate);
        let n = dimension as f64 / rate;
        (pp * n + 1.0) / &self.field_size
    }
}

/// Johnson bound regime.
pub struct JohnsonBoundRegime {
    /// Field size |F|.
    field_size: f64,
    /// Gap-widening factor.
    alpha: f64,
}

impl JohnsonBoundRegime {
    pub fn new(field_size: f64, alpha: f64) -> Self {
        Self { field_size, alpha }
    }

    /// The multiplicity parameter `m` of the Guruswami–Sudan list decoder (JBR):
    /// `m = max(⌈√ρ / (2η)⌉, 3)` [BCHKS25] Theorem 4.2. The factor of 2 that the
    /// statement of Theorem 4.2 omits is a confirmed typo.
    fn jbr_multiplicity(&self, rate: &f64) -> u64 {
        let two_gap = 2.0 * self.gap(rate);
        let m_ceil = (rate.clone().sqrt() / two_gap).ceil() as u64;
        m_ceil.max(3)
    }
}

impl ProximityGapsRegime for JohnsonBoundRegime {
    fn identifier(&self) -> &'static str {
        "JBR"
    }

    fn field_size(&self) -> &f64 {
        &self.field_size
    }

    fn decoding_radius(&self, rate: &f64) -> f64 {
        // 1 - √ρ, where ρ is the code rate.
        1.0 - rate.sqrt()
    }

    fn gap(&self, rate: &f64) -> f64 {
        // Something small that makes γ ∈ [δ/2, 1 - √(1-δ) - η] (see [BCHKS25] Corollary 1.4),
        // where δ = 1 - ρ is the minimum distance of the code.
        let base_correction = 1.0 / 300.0;
        let gap = base_correction * (1.0 + self.alpha);

        // γ ∈ [δ/2, 1 - √(1-δ) - η] <=> 1 - √(1-δ) - η > δ/2 <=> η < 1 - √(1-δ) - δ/2
        let minimum_distance = 1.0 - rate;
        let delta_half = minimum_distance / 2.0;
        assert!(
            gap < self.decoding_radius(rate) - delta_half,
            "Gap must be smaller than 1 - √(1-δ) - δ/2 in JBR"
        );
        gap
    }

    fn max_list_size(&self, rate: &f64, _dimension: u32) -> u64 {
        // Reed–Solomon at radius 1 − √ρ − η is (ℓ, ·)-list decodable with
        // ℓ = 1 / (2·η·√ρ) [BCHKS25].
        let two_gap = 2.0 * self.gap(rate);
        (1.0 / (two_gap * rate.sqrt())).ceil() as u64
    }

    fn error_linear(&self, rate: &f64, dimension: u32) -> f64 {
        // Theorem 4.2 from [BCHKS25].
        let sqrt_rate = rate.sqrt();
        let m = self.jbr_multiplicity(rate);
        let m_shifted = m as f64 + 0.5;

        // First fraction: (2·(m + 1/2)⁵ + 3·(m + 1/2)·γ·ρ)·n / (3·ρ·√ρ)
        let n = dimension as f64 / rate;
        let numerator = (2.0 * m_shifted.powi(5) + 3.0 * m_shifted * self.gap(rate) * rate) * n;
        let denominator = 3.0 * rate * sqrt_rate;
        let first_fraction = numerator / denominator;

        // Second fraction: (m + 1/2) / √ρ
        let second_fraction = m_shifted / sqrt_rate;

        (first_fraction + second_fraction) / &self.field_size
    }
}
