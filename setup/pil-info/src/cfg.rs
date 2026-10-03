//! What the passes need to know about the backend they run for
//! (pilfflonk/docs/README.md#setup-pilfflonk): the field, how the constraint degree is bounded and
//! how the committed polynomials are opened.
//!
//! The STARK runs the passes over Goldilocks with its cubic extension and opens with FRI; pilfflonk
//! runs them over the BN254 scalar field, with no extension, and opens with SHPLONK. This is a
//! run-time value, not a cargo feature, because `proofman-setup` hosts both setups in one binary.

use num_bigint::BigUint;

/// Goldilocks: `p = 2^64 − 2^32 + 1`.
const GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// The BN254 scalar field `r` (the order of G1), big-endian:
/// `0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001`.
const BN254_R_BE: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d, 0x28, 0x33, 0xe8,
    0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
];

/// The largest constraint degree pilfflonk's search tries unless told otherwise, as pil-stark
/// (pilfflonk/docs/protocol.md#degree-search).
pub const DEFAULT_MAX_CONSTRAINT_DEGREE: usize = 9;

/// The field the passes compute over.
///
/// The fields are private so that `neg_one` always matches `modulus`; the two constructors are the
/// only fields the passes support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldCfg {
    modulus: BigUint,
    ext_dim: usize,
    neg_one: String,
}

impl FieldCfg {
    /// Goldilocks, whose challenges, evaluations and every value of a stage past the first live in
    /// its cubic extension.
    pub fn goldilocks() -> Self {
        Self::new(BigUint::from(GOLDILOCKS_MODULUS), 3)
    }

    /// The BN254 scalar field, large enough to need no extension: every value has dimension 1.
    pub fn bn254() -> Self {
        Self::new(BigUint::from_bytes_be(&BN254_R_BE), 1)
    }

    fn new(modulus: BigUint, ext_dim: usize) -> Self {
        let neg_one = (&modulus - 1u32).to_string();
        Self { modulus, ext_dim, neg_one }
    }

    /// The field's characteristic.
    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// The dimension of the extension field the challenges and evaluations live in: 1 when the
    /// base field needs no extension.
    pub fn ext_dim(&self) -> usize {
        self.ext_dim
    }

    /// `modulus − 1` in decimal: a `neg` becomes a multiplication by this constant.
    pub fn neg_one(&self) -> &str {
        &self.neg_one
    }
}

/// How the degree search that picks the intermediate polynomials is bounded and scored.
///
/// Both policies run the same search: for every maximum degree `d` from 2 up to the bound, find the
/// intermediate polynomials that bring the constraint polynomial down to degree `d`, and keep the
/// cheapest `d` (the lowest one on ties).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegreePolicy {
    /// The STARK's: the quotient is committed in pieces of degree `N` over a domain
    /// `2^blowup_bits` times larger (`blowup_bits = nBitsExt − nBits`), so the bound is
    /// `2^blowup_bits + 1`, and the cost of a degree is the base-field columns it adds.
    FromBlowup { blowup_bits: usize },
    /// pilfflonk's (pilfflonk/docs/protocol.md#degree-search): the bound is `max`, and the cost of
    /// a degree is `nImPols + qDeg`.
    Search { max: usize },
}

impl DegreePolicy {
    /// The largest constraint degree the search tries.
    pub fn max_constraint_degree(&self) -> usize {
        match *self {
            DegreePolicy::FromBlowup { blowup_bits } => (1usize << blowup_bits) + 1,
            DegreePolicy::Search { max } => max,
        }
    }

    /// The cost of a candidate degree: `n_im_pols` intermediate polynomials adding `im_cols`
    /// base-field columns, and a quotient of degree `q_deg` and dimension `q_dim`.
    pub(crate) fn cost(&self, n_im_pols: usize, im_cols: usize, q_deg: i64, q_dim: usize) -> i64 {
        match self {
            DegreePolicy::FromBlowup { .. } => q_deg * q_dim as i64 + im_cols as i64,
            DegreePolicy::Search { .. } => n_im_pols as i64 + q_deg,
        }
    }
}

/// How the committed polynomials are opened, which decides what code generation adds after the
/// constraint polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opening {
    /// The STARK's: the FRI polynomial (`friExpId`, its challenges `std_vf1`/`std_vf2` and the
    /// `queryVerifier`), and the quotient's pieces in the `evMap`, whose evaluations the verifier
    /// receives and checks against `Q(ξ)`.
    Fri,
    /// pilfflonk's: the prover opens the committed polynomials with SHPLONK outside these passes,
    /// and the verifier computes `Q(ξ)` rather than receiving it
    /// (pilfflonk/docs/protocol.md#constraint-polynomial), so there is neither a FRI polynomial nor
    /// a quotient in the `evMap`.
    Shplonk,
}

/// The configuration of the passes for one backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PilInfoCfg {
    pub field: FieldCfg,
    pub degree_policy: DegreePolicy,
    pub opening: Opening,
}

impl PilInfoCfg {
    /// The STARK's configuration for an AIR whose extended domain is `2^blowup_bits` times its
    /// trace (`blowup_bits = nBitsExt − nBits`).
    pub fn goldilocks(blowup_bits: usize) -> Self {
        Self {
            field: FieldCfg::goldilocks(),
            degree_policy: DegreePolicy::FromBlowup { blowup_bits },
            opening: Opening::Fri,
        }
    }

    /// pilfflonk's configuration, with the default search bound; `--max-constraint-degree`
    /// replaces `degree_policy`.
    pub fn bn254() -> Self {
        Self {
            field: FieldCfg::bn254(),
            degree_policy: DegreePolicy::Search { max: DEFAULT_MAX_CONSTRAINT_DEGREE },
            opening: Opening::Shplonk,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldilocks_is_the_starks_field() {
        let field = FieldCfg::goldilocks();
        assert_eq!(field.modulus().to_string(), "18446744069414584321");
        assert_eq!(field.ext_dim(), 3);
        // The constant the passes wrote for a `neg` before they were parameterised.
        assert_eq!(field.neg_one(), "18446744069414584320");
    }

    #[test]
    fn bn254_is_the_scalar_field_without_extension() {
        let field = FieldCfg::bn254();
        assert_eq!(
            field.modulus().to_string(),
            "21888242871839275222246405745257275088548364400416034343698204186575808495617"
        );
        assert_eq!(field.ext_dim(), 1);
        assert_eq!(field.neg_one(), "21888242871839275222246405745257275088548364400416034343698204186575808495616");
    }

    #[test]
    fn from_blowup_bound_is_the_blowup_plus_one() {
        // The bound pil2-stark-setup computed as (1 << (nBitsExt - nBits)) + 1.
        assert_eq!(DegreePolicy::FromBlowup { blowup_bits: 1 }.max_constraint_degree(), 3);
        assert_eq!(DegreePolicy::FromBlowup { blowup_bits: 2 }.max_constraint_degree(), 5);
        assert_eq!(DegreePolicy::FromBlowup { blowup_bits: 3 }.max_constraint_degree(), 9);
    }

    #[test]
    fn search_bound_is_its_max() {
        assert_eq!(DegreePolicy::Search { max: 4 }.max_constraint_degree(), 4);
        assert_eq!(PilInfoCfg::bn254().degree_policy.max_constraint_degree(), DEFAULT_MAX_CONSTRAINT_DEGREE);
    }

    #[test]
    fn costs_follow_each_policy() {
        // Two extension-field im pols (6 columns) and a quotient of degree 2 in the extension.
        assert_eq!(DegreePolicy::FromBlowup { blowup_bits: 1 }.cost(2, 6, 2, 3), 12);
        // pilfflonk counts polynomials, not columns.
        assert_eq!(DegreePolicy::Search { max: 9 }.cost(2, 2, 2, 1), 4);
    }

    #[test]
    fn constructors_pair_each_field_with_its_opening() {
        let stark = PilInfoCfg::goldilocks(2);
        assert_eq!(stark.field, FieldCfg::goldilocks());
        assert_eq!(stark.degree_policy, DegreePolicy::FromBlowup { blowup_bits: 2 });
        assert_eq!(stark.opening, Opening::Fri);

        let fflonk = PilInfoCfg::bn254();
        assert_eq!(fflonk.field, FieldCfg::bn254());
        assert_eq!(fflonk.opening, Opening::Shplonk);
    }
}
