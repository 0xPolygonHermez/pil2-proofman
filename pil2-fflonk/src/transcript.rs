//! The Fiat-Shamir transcript, as the prover computes it.
//!
//! Mirrors `Keccak256Transcript` in pil2-stark's rapidsnark. The verifier has
//! to reproduce every challenge bit-for-bit from the proof alone, so the
//! encoding is pinned here against challenges taken from a real proving run.
//!
//! Encoding, from `Keccak256Transcript::getChallenge`:
//!
//! * a field element is 32 bytes big-endian;
//! * a G1 point is converted to affine and written as `x ‖ y`, 32 bytes
//!   big-endian each, with the point at infinity contributing nothing;
//! * the concatenation is hashed and the digest read back big-endian as a
//!   field element, reduced modulo r.
//!
//! The hash is **legacy Keccak-256, not SHA3-256**. `keccak_wrapper.cpp` calls
//! `Keccak(1088, 512, ..., 0x01, ...)`: rate and capacity give the 256-bit
//! variant, and the `0x01` suffix is the original padding. SHA3 uses `0x06`,
//! and substituting it produces a perfectly well-formed but entirely different
//! challenge.

use anyhow::{Context, Result, bail};
use num_bigint::BigUint;
use num_traits::Num;
use sha3::{Digest, Keccak256};

/// The BN254 scalar field modulus.
pub const FR_MODULUS: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

/// Bytes per serialised field element.
pub const FR_BYTES: usize = 32;

pub fn fr_modulus() -> BigUint {
    BigUint::from_str_radix(FR_MODULUS, 10).expect("modulus parses")
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Item {
    Scalar(BigUint),
    /// Affine G1 point. `None` is the point at infinity.
    Commitment(Option<(BigUint, BigUint)>),
}

/// Accumulates transcript items and squeezes challenges from them.
#[derive(Clone, Debug, Default)]
pub struct Transcript {
    items: Vec<Item>,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the transcript. The prover resets before each challenge rather
    /// than accumulating across them, so a verifier that keeps appending will
    /// diverge after the first.
    pub fn reset(&mut self) {
        self.items.clear();
    }

    pub fn add_scalar(&mut self, value: BigUint) {
        self.items.push(Item::Scalar(value));
    }

    /// Add an affine commitment as `x ‖ y`.
    pub fn add_commitment(&mut self, x: BigUint, y: BigUint) {
        self.items.push(Item::Commitment(Some((x, y))));
    }

    /// Add the point at infinity, which contributes no bytes.
    pub fn add_infinity(&mut self) {
        self.items.push(Item::Commitment(None));
    }

    /// Add a commitment written as the proof carries it: three decimal
    /// strings, the third of which must be "1" for an affine point.
    pub fn add_commitment_json(&mut self, point: &[String]) -> Result<()> {
        if point.len() != 3 {
            bail!("commitment has {} coordinates, want 3", point.len());
        }
        if point[2] != "1" {
            bail!("commitment is not affine (z = {:?})", point[2]);
        }
        let x = BigUint::from_str_radix(&point[0], 10).context("parsing commitment x")?;
        let y = BigUint::from_str_radix(&point[1], 10).context("parsing commitment y")?;
        self.add_commitment(x, y);
        Ok(())
    }

    /// The bytes that will be hashed.
    fn serialise(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for item in &self.items {
            match item {
                Item::Scalar(v) => out.extend_from_slice(&to_rpr_be(v)),
                Item::Commitment(Some((x, y))) => {
                    out.extend_from_slice(&to_rpr_be(x));
                    out.extend_from_slice(&to_rpr_be(y));
                }
                // toRprBE returns 0 for the point at infinity, contributing
                // nothing to the hashed length.
                Item::Commitment(None) => {}
            }
        }
        out
    }

    /// Squeeze a challenge: Keccak-256 over the serialisation, read back
    /// big-endian and reduced modulo r.
    pub fn get_challenge(&self) -> BigUint {
        let digest = Keccak256::digest(self.serialise());
        BigUint::from_bytes_be(&digest) % fr_modulus()
    }
}

/// A field element as 32 bytes big-endian, left-padded.
fn to_rpr_be(v: &BigUint) -> [u8; FR_BYTES] {
    let bytes = v.to_bytes_be();
    let mut out = [0u8; FR_BYTES];
    let start = FR_BYTES.saturating_sub(bytes.len());
    out[start..].copy_from_slice(&bytes[bytes.len().saturating_sub(FR_BYTES)..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &str) -> BigUint {
        BigUint::from_str_radix(s, 10).unwrap()
    }

    // Challenges taken from a real run of pil-fflonk's prover on the
    // checked-in fixtures. computeChallengeY is self-contained -- it resets,
    // adds alpha, adds the W commitment, and squeezes -- so it pins the whole
    // encoding: element order, big-endian layout, the Keccak variant and the
    // reduction.
    const ALPHA: &str = "14560641611632097331351172125449579633774312004368406670544341241766131118492";
    const W_X: &str = "12419649577883164498958593584067611959324832377388732561925666137961148910692";
    const W_Y: &str = "21476078643738536497282674355688501033753144678841213640203400378618544673473";
    const EXPECTED_Y: &str = "10213856114127628084316690059097516037569626100727307492391249673580661585742";

    #[test]
    fn reproduces_the_provers_challenge_y() {
        let mut t = Transcript::new();
        t.add_scalar(n(ALPHA));
        t.add_commitment(n(W_X), n(W_Y));

        assert_eq!(t.get_challenge(), n(EXPECTED_Y), "transcript does not match the prover");
    }

    /// The prover resets before each challenge. A verifier that accumulates
    /// would agree on the first challenge and diverge on every later one, so
    /// the reset is part of the protocol rather than housekeeping.
    #[test]
    fn reset_clears_previous_items() {
        let mut t = Transcript::new();
        t.add_scalar(n("12345"));
        t.reset();
        t.add_scalar(n(ALPHA));
        t.add_commitment(n(W_X), n(W_Y));

        assert_eq!(t.get_challenge(), n(EXPECTED_Y));
    }

    /// Order is part of the encoding: the same items appended the other way
    /// round must not give the same challenge.
    #[test]
    fn order_matters() {
        let mut forward = Transcript::new();
        forward.add_scalar(n(ALPHA));
        forward.add_commitment(n(W_X), n(W_Y));

        let mut reversed = Transcript::new();
        reversed.add_commitment(n(W_X), n(W_Y));
        reversed.add_scalar(n(ALPHA));

        assert_ne!(forward.get_challenge(), reversed.get_challenge());
    }

    #[test]
    fn serialisation_widths_are_fixed() {
        let mut t = Transcript::new();
        t.add_scalar(n("1"));
        assert_eq!(t.serialise().len(), 32, "a field element is 32 bytes however small");

        let mut t = Transcript::new();
        t.add_commitment(n("1"), n("2"));
        assert_eq!(t.serialise().len(), 64, "a commitment is x and y, 32 bytes each");

        // Small values must be left-padded, not truncated or right-aligned.
        let mut t = Transcript::new();
        t.add_scalar(n("1"));
        let bytes = t.serialise();
        assert_eq!(bytes[31], 1);
        assert!(bytes[..31].iter().all(|&b| b == 0));
    }

    /// The point at infinity contributes no bytes, matching toRprBE returning
    /// zero for it.
    #[test]
    fn infinity_contributes_nothing() {
        let mut t = Transcript::new();
        t.add_scalar(n("7"));
        t.add_infinity();
        assert_eq!(t.serialise().len(), 32);
    }

    #[test]
    fn accepts_a_commitment_in_proof_form() {
        let point = vec![W_X.to_string(), W_Y.to_string(), "1".to_string()];
        let mut t = Transcript::new();
        t.add_scalar(n(ALPHA));
        t.add_commitment_json(&point).unwrap();
        assert_eq!(t.get_challenge(), n(EXPECTED_Y));
    }

    #[test]
    fn rejects_a_non_affine_commitment() {
        let point = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let mut t = Transcript::new();
        assert!(t.add_commitment_json(&point).is_err());
    }

    /// A challenge is always a field element.
    #[test]
    fn challenge_is_reduced() {
        let mut t = Transcript::new();
        t.add_scalar(n("999"));
        assert!(t.get_challenge() < fr_modulus());
    }
}
