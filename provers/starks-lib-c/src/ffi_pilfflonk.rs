//! Safe wrappers over the pilfflonk C API: every fallible call returns a `Result`.

use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_int, c_void};
use std::ptr::NonNull;

include!("../bindings_pilfflonk.rs");

/// Size of a BN254 scalar at the C API: a canonical (< r) little-endian integer.
pub const PILFFLONK_FR_BYTES: usize = 32;

/// Size of a G1 point at the C API: affine `x‖y`, each coordinate a canonical (< q) little-endian
/// integer of 32 bytes.
pub const PILFFLONK_G1_BYTES: usize = 64;

/// The status code of a failed pilfflonk call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PilFflonkErrorKind {
    InvalidArgument,
    NonCanonical,
    Internal,
    InvalidPoint,
    /// A status these bindings do not know: they are out of sync with `pilfflonk_api.hpp`.
    Unknown(i32),
}

/// A failed pilfflonk call: its status and the library's description of the failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PilFflonkError {
    pub kind: PilFflonkErrorKind,
    pub message: String,
}

impl fmt::Display for PilFflonkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:?})", self.message, self.kind)
    }
}

impl std::error::Error for PilFflonkError {}

/// The failure of the pilfflonk call just made on this thread, with the library's description.
fn last_error(kind: PilFflonkErrorKind) -> PilFflonkError {
    // SAFETY: the C side returns a NUL-terminated string, never NULL, that the next pilfflonk call
    // on this thread overwrites; it is copied before that can happen.
    let message = unsafe { CStr::from_ptr(pilfflonk_last_error()) }.to_string_lossy().into_owned();
    PilFflonkError { kind, message }
}

/// Maps the status of the pilfflonk call just made on this thread to a `Result`.
fn check_status(status: c_int) -> Result<(), PilFflonkError> {
    let kind = match status {
        PILFFLONK_OK => return Ok(()),
        PILFFLONK_ERR_INVALID_ARGUMENT => PilFflonkErrorKind::InvalidArgument,
        PILFFLONK_ERR_NON_CANONICAL => PilFflonkErrorKind::NonCanonical,
        PILFFLONK_ERR_INTERNAL => PilFflonkErrorKind::Internal,
        PILFFLONK_ERR_INVALID_POINT => PilFflonkErrorKind::InvalidPoint,
        other => PilFflonkErrorKind::Unknown(other),
    };
    Err(last_error(kind))
}

/// Checks that `scalar`, read as a little-endian integer, is below the BN254 scalar modulus r.
pub fn pilfflonk_fr_check_canonical_c(scalar: &[u8; PILFFLONK_FR_BYTES]) -> Result<(), PilFflonkError> {
    // SAFETY: `scalar` points to the 32 bytes the function reads.
    check_status(unsafe { pilfflonk_fr_check_canonical(scalar.as_ptr()) })
}

/// The Fiat-Shamir transcript of a proof (spec A.4), owned by the C++ side: rapidsnark's
/// `Keccak256Transcript`, driven as the existing FFLONK prover drives it.
///
/// Absorbing only appends elements; [`squeeze`](Self::squeeze) hashes everything absorbed since
/// the previous squeeze and seeds the next round with the challenge. A refused absorb leaves the
/// transcript as it was.
#[derive(Debug)]
pub struct PilFflonkTranscript {
    handle: NonNull<c_void>,
}

impl PilFflonkTranscript {
    /// A new, empty transcript.
    pub fn new() -> Result<Self, PilFflonkError> {
        // SAFETY: no arguments; the result is either NULL or a handle this value then owns.
        let handle = unsafe { pilfflonk_transcript_new() };
        // Creating a transcript can only fail inside the library (out of memory).
        NonNull::new(handle).map(|handle| Self { handle }).ok_or_else(|| last_error(PilFflonkErrorKind::Internal))
    }

    /// Absorbs scalars: canonical (< r) little-endian integers.
    pub fn absorb_fr(&mut self, scalars: &[[u8; PILFFLONK_FR_BYTES]]) -> Result<(), PilFflonkError> {
        self.absorb(scalars.as_flattened(), scalars.len(), PILFFLONK_TRANSCRIPT_FR)
    }

    /// Absorbs affine G1 points on the curve. The C API refuses the point at infinity and points
    /// with a coordinate below 2^192, which `Keccak256Transcript` does not hash as A.4 encodes them.
    pub fn absorb_g1(&mut self, points: &[[u8; PILFFLONK_G1_BYTES]]) -> Result<(), PilFflonkError> {
        self.absorb(points.as_flattened(), points.len(), PILFFLONK_TRANSCRIPT_G1)
    }

    fn absorb(&mut self, bytes: &[u8], n: usize, kind: u32) -> Result<(), PilFflonkError> {
        // SAFETY: `bytes` holds the `n` elements of `kind` that the call reads, and the handle is live.
        check_status(unsafe { pilfflonk_transcript_absorb(self.handle.as_ptr(), bytes.as_ptr(), n as u64, kind) })
    }

    /// The challenge over everything absorbed since the previous squeeze, as a canonical
    /// little-endian scalar. Fails on a transcript to which nothing has been absorbed yet.
    pub fn squeeze(&mut self) -> Result<[u8; PILFFLONK_FR_BYTES], PilFflonkError> {
        let mut challenge = [0u8; PILFFLONK_FR_BYTES];
        // SAFETY: `challenge` has the 32 bytes the call writes, and the handle is live.
        check_status(unsafe { pilfflonk_transcript_squeeze(self.handle.as_ptr(), challenge.as_mut_ptr()) })?;
        Ok(challenge)
    }
}

impl Drop for PilFflonkTranscript {
    fn drop(&mut self) {
        // SAFETY: the handle came from `pilfflonk_transcript_new` and is released only here.
        unsafe { pilfflonk_transcript_free(self.handle.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// r in little-endian 64-bit limbs, written out independently of the C++ side.
    const R_LIMBS: [u64; 4] = [0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];

    fn scalar(limbs: [u64; 4]) -> [u8; PILFFLONK_FR_BYTES] {
        let mut bytes = [0u8; PILFFLONK_FR_BYTES];
        for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn accepts_canonical_scalars() {
        let r_minus_one = [R_LIMBS[0] - 1, R_LIMBS[1], R_LIMBS[2], R_LIMBS[3]];
        for limbs in [[0; 4], [1, 0, 0, 0], r_minus_one] {
            assert_eq!(pilfflonk_fr_check_canonical_c(&scalar(limbs)), Ok(()), "{limbs:x?}");
        }
    }

    #[test]
    fn rejects_non_canonical_scalars() {
        for limbs in [R_LIMBS, [u64::MAX; 4]] {
            let err = pilfflonk_fr_check_canonical_c(&scalar(limbs)).unwrap_err();
            assert_eq!(err.kind, PilFflonkErrorKind::NonCanonical, "{limbs:x?}");
            assert!(err.message.contains("pilfflonk_fr_check_canonical"), "{err}");
        }
    }

    /// 32 little-endian bytes from a 64-digit big-endian hex string.
    fn from_hex(hex: &str) -> [u8; PILFFLONK_FR_BYTES] {
        assert_eq!(hex.len(), 2 * PILFFLONK_FR_BYTES, "{hex}");
        let mut bytes = [0u8; PILFFLONK_FR_BYTES];
        for (i, byte) in bytes.iter_mut().rev().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        bytes
    }

    /// An affine point `x‖y` from big-endian hex coordinates.
    fn point(x: &str, y: &str) -> [u8; PILFFLONK_G1_BYTES] {
        let mut bytes = [0u8; PILFFLONK_G1_BYTES];
        bytes[..PILFFLONK_FR_BYTES].copy_from_slice(&from_hex(x));
        bytes[PILFFLONK_FR_BYTES..].copy_from_slice(&from_hex(y));
        bytes
    }

    const ONE: &str = "0000000000000000000000000000000000000000000000000000000000000001";
    const R_MINUS_ONE: &str = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000";

    /// k·G for the generator G = (1, 2).
    const P2: [&str; 2] = [
        "030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3",
        "15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4",
    ];
    const P3: [&str; 2] = [
        "0769bf9ac56bea3ff40232bcb1b6bd159315d84715b8e679f2d355961915abf0",
        "2ab799bee0489429554fdb7c8d086475319e63b40b9c5b57cdf1ff3dd9fe2261",
    ];
    const P5: [&str; 2] = [
        "17c139df0efee0f766bc0204762b774362e4ded88953a39ce849a8a7fa163fa9",
        "01e0559bacb160664764a357af8a9fe70baa9258e0b959273ffc5718c6d4cc7c",
    ];

    /// The challenges of the sequence in `reproduces_the_pinned_challenges`, as the C++ test
    /// computes them by hand from A.4 (`PINNED_HEX` in pil2-stark/test/pilfflonk/
    /// pilfflonk_transcript_test.cpp): keep both in sync.
    const PINNED: [&str; 3] = [
        "26ecd6b31f13f24a81bb71f60ef433df1185225122b3b101c736948874b1a41f",
        "11dba4dd6851bff01335d5e5b90efe759da23fce9a86a1c050daefdcd6948c14",
        "1a94293663ade82c5745a10779392a00a28435f34adb91d46084d9e50bee7a69",
    ];

    /// Runs the pinned sequence, calling `between` on the transcript before each of its steps.
    fn pinned_sequence(mut between: impl FnMut(&mut PilFflonkTranscript)) -> [[u8; PILFFLONK_FR_BYTES]; 3] {
        let mut transcript = PilFflonkTranscript::new().unwrap();
        between(&mut transcript);
        transcript.absorb_fr(&[from_hex(ONE), from_hex(R_MINUS_ONE)]).unwrap();
        between(&mut transcript);
        transcript.absorb_g1(&[point(P2[0], P2[1]), point(P3[0], P3[1])]).unwrap();
        between(&mut transcript);
        let first = transcript.squeeze().unwrap();
        between(&mut transcript);
        transcript.absorb_g1(&[point(P5[0], P5[1])]).unwrap();
        between(&mut transcript);
        let second = transcript.squeeze().unwrap();
        between(&mut transcript);
        let third = transcript.squeeze().unwrap();
        [first, second, third]
    }

    #[test]
    fn transcript_reproduces_the_pinned_challenges() {
        assert_eq!(pinned_sequence(|_| {}), PINNED.map(from_hex));
    }

    #[test]
    fn transcript_empty_absorbs_change_nothing() {
        let challenges = pinned_sequence(|transcript| {
            transcript.absorb_fr(&[]).unwrap();
            transcript.absorb_g1(&[]).unwrap();
        });
        assert_eq!(challenges, PINNED.map(from_hex));
    }

    #[test]
    fn transcript_refuses_bad_input_and_stays_as_it_was() {
        let r = from_hex("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");
        let off_curve = point(ONE, "0000000000000000000000000000000000000000000000000000000000000003");
        let infinity = [0u8; PILFFLONK_G1_BYTES];
        let generator = point(ONE, "0000000000000000000000000000000000000000000000000000000000000002");

        let challenges = pinned_sequence(|transcript| {
            let refusals = [
                (transcript.absorb_fr(&[from_hex(ONE), r]), PilFflonkErrorKind::NonCanonical, "element 1"),
                (transcript.absorb_g1(&[off_curve]), PilFflonkErrorKind::InvalidPoint, "not on the curve"),
                (transcript.absorb_g1(&[infinity]), PilFflonkErrorKind::InvalidPoint, "point at infinity"),
                (transcript.absorb_g1(&[generator]), PilFflonkErrorKind::InvalidPoint, "below 2^192"),
            ];
            for (result, kind, text) in refusals {
                let err = result.unwrap_err();
                assert_eq!(err.kind, kind, "{err}");
                assert!(err.message.contains("pilfflonk_transcript_absorb") && err.message.contains(text), "{err}");
            }
        });
        assert_eq!(challenges, PINNED.map(from_hex));
    }

    #[test]
    fn transcript_refuses_to_squeeze_before_absorbing() {
        let mut transcript = PilFflonkTranscript::new().unwrap();
        let err = transcript.squeeze().unwrap_err();
        assert_eq!(err.kind, PilFflonkErrorKind::InvalidArgument, "{err}");
        assert!(err.message.contains("pilfflonk_transcript_squeeze"), "{err}");
    }
}
