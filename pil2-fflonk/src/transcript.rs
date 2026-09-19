//! The Fiat-Shamir transcript.
//!
//! A thin adapter over rapidsnark's `Keccak256Transcript`, reached through
//! `proofman-fflonk-lib-c`. The encoding is not reimplemented here: it is
//! legacy Keccak-256 with `0x01` padding rather than SHA3's `0x06`, over
//! fixed-width big-endian values, and a second implementation would have to
//! match those exactly -- silently producing well-formed but different
//! challenges if it did not.
//!
//! What this layer adds is only the conversion between the `BigUint` the
//! verifier works in and the canonical big-endian bytes the C++ takes.

use anyhow::{Context, Result, bail};
use num_bigint::BigUint;
use num_traits::Num;

/// The BN254 scalar field modulus.
pub const FR_MODULUS: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

/// Bytes per serialised field element.
pub const FR_BYTES: usize = 32;

pub fn fr_modulus() -> BigUint {
    BigUint::from_str_radix(FR_MODULUS, 10).expect("modulus parses")
}

/// Accumulates transcript items and squeezes challenges from them.
pub struct Transcript {
    inner: proofman_fflonk_lib_c::Transcript,
}

impl Transcript {
    pub fn new() -> Self {
        Self::try_new().expect("the C++ transcript allocates")
    }

    pub fn try_new() -> Result<Self> {
        Ok(Transcript { inner: proofman_fflonk_lib_c::Transcript::new().map_err(anyhow::Error::msg)? })
    }

    /// Clear the transcript. The prover resets before each challenge rather
    /// than accumulating across them, so a verifier that keeps appending will
    /// diverge after the first.
    pub fn reset(&mut self) {
        self.inner.reset().expect("reset cannot fail on a live transcript");
    }

    pub fn add_scalar(&mut self, value: BigUint) {
        self.inner.add_scalar(&to_rpr_be(&value)).expect("adding a scalar cannot fail");
    }

    /// Add an affine commitment as `x ‖ y`.
    pub fn add_commitment(&mut self, x: BigUint, y: BigUint) {
        let mut xy = [0u8; 2 * FR_BYTES];
        xy[..FR_BYTES].copy_from_slice(&to_rpr_be(&x));
        xy[FR_BYTES..].copy_from_slice(&to_rpr_be(&y));
        self.inner.add_commitment(&xy).expect("adding a commitment cannot fail");
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

    /// Squeeze a challenge.
    pub fn get_challenge(&mut self) -> BigUint {
        BigUint::from_bytes_be(&self.inner.challenge().expect("squeezing cannot fail"))
    }
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
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

    // From a real run of pil-fflonk's prover: alpha and the W commitment give
    // challenge Y. Kept here as well as in the wrapper crate because this is
    // the layer the verifier actually calls.
    const ALPHA: &str = "14560641611632097331351172125449579633774312004368406670544341241766131118492";
    const W_X: &str = "12419649577883164498958593584067611959324832377388732561925666137961148910692";
    const W_Y: &str = "21476078643738536497282674355688501033753144678841213640203400378618544673473";
    const EXPECTED_Y: &str = "10213856114127628084316690059097516037569626100727307492391249673580661585742";

    fn n(s: &str) -> BigUint {
        BigUint::from_str_radix(s, 10).unwrap()
    }

    #[test]
    fn reproduces_the_provers_challenge_y() {
        let mut t = Transcript::new();
        t.add_scalar(n(ALPHA));
        t.add_commitment(n(W_X), n(W_Y));

        assert_eq!(t.get_challenge(), n(EXPECTED_Y), "transcript does not match the prover");
    }

    #[test]
    fn reset_clears_previous_items() {
        let mut t = Transcript::new();
        t.add_scalar(n("12345"));
        t.reset();
        t.add_scalar(n(ALPHA));
        t.add_commitment(n(W_X), n(W_Y));

        assert_eq!(t.get_challenge(), n(EXPECTED_Y));
    }

    /// Order is part of the encoding.
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
