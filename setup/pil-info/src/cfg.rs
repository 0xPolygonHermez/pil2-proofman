//! What the passes need to know about the backend they run for: the field and how the constraint
//! degree is bounded.
//!
//! The STARK runs the passes over Goldilocks with its cubic extension; a field large enough to need
//! no extension, such as the BN128 scalar field, makes every value of dimension 1. This is a
//! run-time value, not a cargo feature, so that one binary can host setups over several fields.

use num_bigint::BigUint;

/// Goldilocks: `p = 2^64 − 2^32 + 1`.
const GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// The BN128 scalar field `r` (the order of G1), big-endian:
/// `0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001`.
const BN128_R_BE: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d, 0x28, 0x33, 0xe8,
    0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
];

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

    /// The BN128 scalar field, large enough to need no extension: every value has dimension 1.
    pub fn bn128() -> Self {
        Self::new(BigUint::from_bytes_be(&BN128_R_BE), 1)
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

/// The configuration of the passes for one backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PilInfoCfg {
    pub field: FieldCfg,
    /// The largest constraint degree the search for intermediate polynomials tries.
    pub max_constraint_degree: usize,
}

impl PilInfoCfg {
    /// The STARK's configuration for an AIR whose extended domain is `2^blowup_bits` times its
    /// trace (`blowup_bits = nBitsExt − nBits`): the quotient is committed in pieces of degree `N`
    /// over that domain, so the constraint degree is bounded by `2^blowup_bits + 1`.
    pub fn goldilocks(blowup_bits: usize) -> Self {
        Self { field: FieldCfg::goldilocks(), max_constraint_degree: (1usize << blowup_bits) + 1 }
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
    fn bn128_is_the_scalar_field_without_extension() {
        let field = FieldCfg::bn128();
        assert_eq!(
            field.modulus().to_string(),
            "21888242871839275222246405745257275088548364400416034343698204186575808495617"
        );
        assert_eq!(field.ext_dim(), 1);
        assert_eq!(field.neg_one(), "21888242871839275222246405745257275088548364400416034343698204186575808495616");
    }

    #[test]
    fn the_starks_bound_is_the_blowup_plus_one() {
        // The bound pil2-stark-setup computed as (1 << (nBitsExt - nBits)) + 1.
        assert_eq!(PilInfoCfg::goldilocks(1).max_constraint_degree, 3);
        assert_eq!(PilInfoCfg::goldilocks(2).max_constraint_degree, 5);
        assert_eq!(PilInfoCfg::goldilocks(3).max_constraint_degree, 9);
        assert_eq!(PilInfoCfg::goldilocks(2).field, FieldCfg::goldilocks());
    }
}
