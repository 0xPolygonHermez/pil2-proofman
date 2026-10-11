//! SHPLONK: the pairing-based polynomial commitment scheme pil2-fflonk opens with.
//!
//! Structurally unlike [`Fri`](super::Fri) and [`Whir`](super::Whir). Those are
//! hash-based: security comes from query counts, grinding and proximity gaps,
//! and the proof grows with the trace. SHPLONK (BDFG20) batches KZG openings,
//! so it makes no queries, opens no Merkle paths, and emits a proof whose size
//! depends only on how many combined polynomials and opening points the setup
//! declares -- not on the trace length.
//!
//! Its security therefore decomposes into two unrelated parts:
//!
//! * **Computational.** KZG binding rests on q-SDH in the AGM over the pairing
//!   curve, so it is capped by the curve, not by any parameter chosen here.
//!   This is a hard ceiling: no amount of tuning raises it.
//! * **Statistical.** Two Fiat-Shamir challenges are sampled from the scalar
//!   field -- `alpha`, batching the combined polynomials, and `y`, the point
//!   at which the batched identity is checked. Each contributes a
//!   Schwartz-Zippel term over |Fr|.
//!
//! The reported total is the minimum, which for BN254 is the curve term. That
//! is the point of implementing this trait: it makes the soundness table say
//! plainly that a SHPLONK tail caps the pipeline near the curve's security
//! level, however many bits the STARK layers achieve.

/// Bits of security from an error probability. Mirrors the helper of the same
/// name in pil2-stark's `pcs::types`, kept local so this crate does not depend
/// on the STARK setup crate -- see the module docs.
fn bits_of_security_from_error(epsilon: f64) -> u32 {
    debug_assert!(epsilon >= 0.0 && epsilon.is_finite(), "invalid error {epsilon}");
    if epsilon == 0.0 {
        return 1074;
    }
    (-epsilon.log2()).floor().max(0.0) as u32
}

/// Security of the pairing curve's q-SDH / discrete-log problem, in bits.
///
/// BN254 is **not** a 128-bit curve. The exTNFS improvements to the number
/// field sieve (Kim--Barbulescu, CRYPTO'16) cut its embedding-degree-12
/// discrete-log security to roughly this level, and it is the value the
/// Ethereum precompiles are still specified around.
pub const BN254_SECURITY_BITS: u32 = 100;

/// Free parameters of a SHPLONK instance.
#[derive(Clone, Debug)]
pub struct ShPlonkConfig {
    /// Scalar field size |Fr|.
    pub field_size: f64,
    /// Highest degree among the combined polynomials f_i.
    pub max_degree: u64,
    /// Number of combined polynomials f_i, each carrying one commitment.
    pub num_combined_polys: u64,
    /// Distinct opening points across every f_i.
    pub num_opening_points: u64,
    /// Total (polynomial, opening point) pairs -- one field element each in
    /// the proof, and the count `alpha` has to separate.
    pub num_evaluations: u64,
    /// q-SDH security of the pairing curve, in bits. See [`BN254_SECURITY_BITS`].
    pub curve_security_bits: u32,
    /// Size of one G1 point in the proof, in bits.
    pub g1_point_bits: u64,
    /// Size of one scalar field element, in bits.
    pub field_element_bits: u64,
    /// Target security level in bits.
    pub target_security_bits: u64,
}

/// A solved SHPLONK parameterization.
#[derive(Clone, Debug)]
pub struct ShPlonk {
    cfg: ShPlonkConfig,
}

impl ShPlonk {
    pub fn new(cfg: ShPlonkConfig) -> Self {
        debug_assert!(cfg.field_size > 1.0, "field size must exceed 1");
        Self { cfg }
    }

    pub fn config(&self) -> &ShPlonkConfig {
        &self.cfg
    }

    /// Soundness error of the batching challenge `alpha`.
    ///
    /// The prover commits before `alpha` is drawn, so a batched identity that
    /// is false in any component survives only if `alpha` lands on a root of
    /// the difference: at most `num_evaluations / |Fr|` by Schwartz-Zippel.
    fn batching_error(&self) -> f64 {
        self.cfg.num_evaluations as f64 / self.cfg.field_size
    }

    /// Soundness error of the evaluation challenge `y`.
    ///
    /// W and W' are checked at a single random point, so a false identity
    /// survives with probability at most `deg / |Fr|`, where `deg` is bounded
    /// by the largest combined polynomial plus its opening set.
    fn opening_error(&self) -> f64 {
        (self.cfg.max_degree + self.cfg.num_opening_points) as f64 / self.cfg.field_size
    }

    pub fn security_levels(&self) -> Vec<(String, u32)> {
        vec![
            // The computational ceiling, listed first because it dominates.
            ("curve (q-SDH)".to_string(), self.cfg.curve_security_bits),
            ("batching (alpha)".to_string(), bits_of_security_from_error(self.batching_error())),
            ("opening (y)".to_string(), bits_of_security_from_error(self.opening_error())),
        ]
    }

    /// SHPLONK opens no Merkle trees.
    pub fn num_merkle_openings(&self) -> u64 {
        0
    }

    /// SHPLONK makes no queries, so the verifier spends no hashes on them.
    /// Its verifier cost is pairings, which this trait does not model.
    pub fn total_query_hashes(&self) -> f64 {
        0.0
    }

    /// One commitment per combined polynomial, plus W and W', plus one field
    /// element per (polynomial, opening point) evaluation. Independent of the
    /// trace length -- the property that makes this a viable final wrap.
    pub fn proof_size_bits(&self) -> u64 {
        let commitments = self.cfg.num_combined_polys + 2;
        commitments * self.cfg.g1_point_bits + self.cfg.num_evaluations * self.cfg.field_element_bits
    }

    /// The minimum over all security levels.
    pub fn total_security_bits(&self) -> u32 {
        self.security_levels().into_iter().map(|(_, b)| b).min().unwrap_or(0)
    }

    /// Name of the scheme, for the soundness table.
    pub fn identifier(&self) -> &'static str {
        "SHPLONK"
    }

    pub fn parameter_summary(&self) -> String {
        let params: Vec<(&str, String)> = vec![
            ("Target Security Bits", self.cfg.target_security_bits.to_string()),
            ("Curve Security Bits", self.cfg.curve_security_bits.to_string()),
            ("Field Size", format!("2^{:.0}", self.cfg.field_size.log2())),
            ("Max Degree", self.cfg.max_degree.to_string()),
            ("Combined Polynomials", self.cfg.num_combined_polys.to_string()),
            ("Opening Points", self.cfg.num_opening_points.to_string()),
            ("Evaluations", self.cfg.num_evaluations.to_string()),
            ("Proof Size", format!("{} bytes", self.proof_size_bits().div_ceil(8))),
        ];

        params.iter().map(|(k, v)| format!("  {k}: {v}")).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// |Fr| for BN254.
    fn bn254_fr_size() -> f64 {
        // r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        (2.0f64).powi(254) * 1.2071
    }

    fn test_config(max_degree: u64, num_combined_polys: u64, num_evaluations: u64) -> ShPlonkConfig {
        ShPlonkConfig {
            field_size: bn254_fr_size(),
            max_degree,
            num_combined_polys,
            num_opening_points: 2,
            num_evaluations,
            curve_security_bits: BN254_SECURITY_BITS,
            g1_point_bits: 512,
            field_element_bits: 256,
            target_security_bits: 100,
        }
    }

    #[test]
    fn curve_security_is_the_ceiling() {
        let shplonk = ShPlonk::new(test_config(1 << 20, 4, 40));

        // Both Schwartz-Zippel terms sit far above the curve, so the curve is
        // what the total reports. This is the whole point: tuning degrees or
        // opening counts cannot lift a BN254 SHPLONK past the curve.
        let levels = shplonk.security_levels();
        assert_eq!(levels[0], ("curve (q-SDH)".to_string(), BN254_SECURITY_BITS));
        assert!(levels[1].1 > BN254_SECURITY_BITS, "batching term should not be the binding one");
        assert!(levels[2].1 > BN254_SECURITY_BITS, "opening term should not be the binding one");

        assert_eq!(shplonk.total_security_bits(), BN254_SECURITY_BITS);
    }

    #[test]
    fn no_queries_and_no_merkle_openings() {
        let shplonk = ShPlonk::new(test_config(1 << 20, 4, 40));
        assert_eq!(shplonk.num_merkle_openings(), 0);
        assert_eq!(shplonk.total_query_hashes(), 0.0);
    }

    #[test]
    fn proof_size_is_independent_of_degree() {
        let small = ShPlonk::new(test_config(1 << 10, 4, 40));
        let large = ShPlonk::new(test_config(1 << 24, 4, 40));

        // Unlike FRI, growing the trace by 2^14 does not grow the proof at all.
        assert_eq!(small.proof_size_bits(), large.proof_size_bits());

        // 6 G1 points (4 f_i + W + W') at 512 bits, 40 evaluations at 256.
        assert_eq!(small.proof_size_bits(), 6 * 512 + 40 * 256);
    }

    #[test]
    fn proof_size_grows_with_openings_not_trace() {
        let few = ShPlonk::new(test_config(1 << 20, 4, 40));
        let many = ShPlonk::new(test_config(1 << 20, 8, 80));
        assert!(many.proof_size_bits() > few.proof_size_bits());
    }

    #[test]
    fn schwartz_zippel_terms_degrade_with_a_small_field() {
        // A 64-bit field would make the opening challenge, not the curve, the
        // binding term -- the check that these terms are actually computed and
        // not hardcoded.
        let mut cfg = test_config(1 << 20, 4, 40);
        cfg.field_size = (2.0f64).powi(64);
        let shplonk = ShPlonk::new(cfg);

        let levels = shplonk.security_levels();
        // 64 - log2(2^20 + 2) = 43.999..., and bits_of_security_from_error
        // floors, so 43 rather than a bare 44.
        assert_eq!(levels[2].1, 43);
        assert_eq!(shplonk.total_security_bits(), 43);
    }

    #[test]
    fn identifier_is_reported() {
        let shplonk = ShPlonk::new(test_config(1 << 20, 4, 40));
        assert_eq!(shplonk.identifier(), "SHPLONK");
    }
}
