use std::fmt;

/// Polynomial Commitment Scheme.
///
/// A PCS is constructed from its *free* parameters (field, rate, folding
/// schedule, target security, regime kind) and deduces the rest — query
/// counts, grinding bits, gap widening — at construction time. Hence the
/// methods below take no regime: the instance already owns a fully solved
/// parameterization.
pub trait Pcs {
    /// Returns the name of the PCS.
    fn identifier(&self) -> &'static str;

    /// PCS-specific security levels, phase by phase.
    /// Entries are (descriptive label, bits of security).
    fn security_levels(&self) -> Vec<(String, u32)>;

    /// The minimum over all security levels.
    fn total_security_bits(&self) -> u32 {
        self.security_levels().into_iter().map(|(_, b)| b).min().unwrap_or(0)
    }

    // /// Estimated proof size in bits.
    // fn proof_size_bits(&self) -> u64;

    /// The code dimension k.
    fn dimension(&self) -> u32;

    /// The code rate ρ.
    fn rate(&self) -> f64;

    /// Description of the parameters of the PCS.
    fn parameter_summary(&self) -> String;
}

/// How the coefficients c_i are chosen when combining `batch_size`
/// polynomials f_i into f_batch = Σ c_i · f_i.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Batching {
    /// c_i = γ^i for a random challenge γ.
    /// "Batching over parameterized curves", BCIKS20 Thm 6.2.
    /// Error depends on batch_size (ℓ in BCIKS20).
    Powers,
    /// c_0 = 1, c_i = r_i for independent random r_i.
    /// "Batching over affine spaces", BCIKS20 Thm 1.6.
    /// Error independent of batch_size.
    Affine,
    /// c_i = eq(r, i). Multilinear batching, BCHKS25 §4.1.
    /// Only usable with multilinear PCSs.
    Multilinear,
}

impl fmt::Display for Batching {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Batching::Powers => write!(f, "Powers"),
            Batching::Affine => write!(f, "Affine"),
            Batching::Multilinear => write!(f, "Multilinear"),
        }
    }
}

/// Bits of security from an error probability.
pub fn bits_of_security_from_error(epsilon: f64) -> u32 {
    debug_assert!(epsilon > 0.0 && epsilon.is_finite(), "invalid error {epsilon}");
    (-epsilon.log2()).floor().max(0.0) as u32
}

/// Bits of security from log2 of an error probability. Use this (and
/// log2-space arithmetic) whenever the error itself can underflow `f64`,
/// e.g. `(1-δ)^t` with many high-value queries.
pub fn bits_of_security_from_log2_error(log2_epsilon: f64) -> u32 {
    debug_assert!(log2_epsilon.is_finite(), "invalid log2 error {log2_epsilon}");
    (-log2_epsilon).floor().max(0.0) as u32
}

/// log2(2^a + 2^b) for log2-space `a`, `b` (log-sum-exp).
pub fn log2_add(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (1.0 + f64::exp2(lo - hi)).log2()
}

/// ε ↦ ε · 2^-g. exp2 of an integer is exact, so no precision is lost.
pub fn apply_grinding(epsilon: f64, grinding_bits: u32) -> f64 {
    epsilon * f64::exp2(-(grinding_bits as f64))
}