//! Rust bindings to pil2-fflonk's C++ proving library.
//!
//! The split is the one pil2-proofman already uses for pil2-stark: Rust owns
//! setup and orchestration, C++ owns the field and curve arithmetic, and this
//! crate is only the seam. It holds no logic of its own -- anything that could
//! be decided in Rust belongs in `pil2-fflonk`, not here.
//!
//! # Representation
//!
//! Buffers cross the boundary in **ffiasm's own representation**, not a
//! canonical one:
//!
//! * a scalar is 32 bytes, four little-endian 64-bit limbs, in Montgomery form;
//! * an affine G1 point is 64 bytes, two base-field elements, also Montgomery.
//!
//! That is exactly how the proving key stores them, so a caller hands the key's
//! sections over untouched. The alternative -- canonicalising at the boundary --
//! would mean reimplementing Montgomery reduction in Rust for two different
//! primes to no benefit, and every conversion would be a chance to get it
//! wrong.
//!
//! Because these bytes are opaque, the functions here validate what they can
//! (buffer lengths) and trust the rest. Passing a buffer that is not in this
//! representation produces a wrong answer rather than an error.

use std::ffi::CStr;

/// Bytes in a scalar, as the key stores it.
pub const FR_BYTES: usize = 32;

/// Bytes in an affine G1 point, as the key stores it.
pub const G1_AFFINE_BYTES: usize = 64;

unsafe extern "C" {
    fn pilfflonk_msm(ptau: *const u8, coeffs: *const u8, n: u64, out: *mut u8) -> i32;
    fn pilfflonk_intt(src: *const u8, size: u64, ncols: u64, out: *mut u8) -> i32;
    fn pilfflonk_g1_to_bytes_be(point: *const u8, out: *mut u8) -> i32;
    fn pilfflonk_eval(coeffs: *const u8, n: u64, x: *const u8, out: *mut u8) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn pilfflonk_combine(
        stage: *const u8,
        stage_len: u64,
        stage_cols: u64,
        col_ids: *const u64,
        col_lens: *const u64,
        n: u64,
        out: *mut u8,
        out_cap: u64,
        out_len: *mut u64,
    ) -> i32;
    fn pilfflonk_transcript_new() -> *mut std::ffi::c_void;
    fn pilfflonk_transcript_free(handle: *mut std::ffi::c_void);
    fn pilfflonk_transcript_reset(handle: *mut std::ffi::c_void) -> i32;
    fn pilfflonk_transcript_add_scalar(handle: *mut std::ffi::c_void, value: *const u8) -> i32;
    fn pilfflonk_transcript_add_commitment(handle: *mut std::ffi::c_void, xy: *const u8) -> i32;
    fn pilfflonk_transcript_challenge(handle: *mut std::ffi::c_void, out: *mut u8) -> i32;
    fn pilfflonk_last_error() -> *const std::os::raw::c_char;
}

/// An affine G1 point in the key's representation. All zeroes is infinity.
pub type G1Affine = [u8; G1_AFFINE_BYTES];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A buffer's length is not a whole number of elements, or the two inputs
    /// describe different lengths.
    Shape(String),
    /// The C++ side failed, with the message it reported.
    Native(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Shape(m) => write!(f, "{m}"),
            Error::Native(m) => write!(f, "pil2-fflonk C++: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// The message from the last failed call on this thread.
fn last_error() -> String {
    // Safety: the pointer is owned by the library and valid until the next
    // call into it, which cannot happen while this borrow is alive.
    unsafe {
        let p = pilfflonk_last_error();
        if p.is_null() {
            "no error reported".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Multi-scalar multiplication: `sum_j coeffs[j] * ptau[j]`.
///
/// This is the operation every commitment is built from -- committing to a
/// polynomial is its coefficients against the key's powers of tau. Both slices
/// must describe the same number of terms.
pub fn msm(ptau: &[u8], coeffs: &[u8]) -> Result<G1Affine, Error> {
    if !ptau.len().is_multiple_of(G1_AFFINE_BYTES) {
        return Err(Error::Shape(format!(
            "ptau is {} bytes, not a whole number of {G1_AFFINE_BYTES}-byte points",
            ptau.len()
        )));
    }
    if !coeffs.len().is_multiple_of(FR_BYTES) {
        return Err(Error::Shape(format!(
            "coeffs is {} bytes, not a whole number of {FR_BYTES}-byte scalars",
            coeffs.len()
        )));
    }

    let points = ptau.len() / G1_AFFINE_BYTES;
    let scalars = coeffs.len() / FR_BYTES;
    if points != scalars {
        return Err(Error::Shape(format!("{scalars} scalars against {points} points")));
    }

    let mut out = [0u8; G1_AFFINE_BYTES];

    // Safety: both pointers are valid for the lengths just checked, `out` is a
    // local of exactly the size the C side writes, and the callee catches its
    // own exceptions rather than unwinding into Rust.
    let code = unsafe { pilfflonk_msm(ptau.as_ptr(), coeffs.as_ptr(), scalars as u64, out.as_mut_ptr()) };

    if code != 0 {
        return Err(Error::Native(last_error()));
    }
    Ok(out)
}

/// Interpolate columns of evaluations into coefficients.
///
/// `src` is `size` rows of `ncols` values, row-major -- the layout a trace is
/// stored in. Every column is interpolated over the same domain of `size`
/// points, and the result keeps that layout.
///
/// This is how a committed stage gets from its trace to something committable:
/// a commitment is over coefficients, and the trace is evaluations.
pub fn intt(src: &[u8], size: usize, ncols: usize) -> Result<Vec<u8>, Error> {
    if ncols == 0 {
        return Err(Error::Shape("no columns to interpolate".into()));
    }
    if size == 0 || !size.is_power_of_two() {
        return Err(Error::Shape(format!("domain size {size} is not a power of two")));
    }

    let want = size * ncols * FR_BYTES;
    if src.len() != want {
        return Err(Error::Shape(format!("{size} rows of {ncols} columns needs {want} bytes, got {}", src.len())));
    }

    let mut out = vec![0u8; want];

    // Safety: both buffers are exactly `want` bytes, which is what the callee
    // reads and writes, and it catches its own exceptions.
    let code = unsafe { pilfflonk_intt(src.as_ptr(), size as u64, ncols as u64, out.as_mut_ptr()) };

    if code != 0 {
        return Err(Error::Native(last_error()));
    }
    Ok(out)
}

/// Interleave columns of a stage's buffer into one combined polynomial.
///
/// `stage` is coefficient-major over `stage_cols` columns; `columns` names the
/// ones to pack, in slot order, with how many coefficients to take from each.
/// Slot `j`'s coefficient `i` lands at `i * n + j`, and trailing zeroes are
/// trimmed.
///
/// Wraps rapidsnark's `CPolynomial` -- the same class the fflonk prover builds
/// its `C0`/`C1`/`C2` with.
pub fn combine(stage: &[u8], stage_cols: usize, columns: &[(usize, usize)]) -> Result<Vec<u8>, Error> {
    if columns.is_empty() {
        return Err(Error::Shape("no columns to pack".into()));
    }
    if stage_cols == 0 {
        return Err(Error::Shape("the stage has no columns".into()));
    }
    if !stage.len().is_multiple_of(FR_BYTES) {
        return Err(Error::Shape(format!("the stage is {} bytes, not whole coefficients", stage.len())));
    }

    let ids: Vec<u64> = columns.iter().map(|&(id, _)| id as u64).collect();
    let lens: Vec<u64> = columns.iter().map(|&(_, len)| len as u64).collect();

    // The combined polynomial cannot exceed the highest index any slot reaches.
    let n = columns.len();
    let cap = columns.iter().enumerate().map(|(j, &(_, len))| len.saturating_sub(1) * n + j + 1).max().unwrap_or(0);

    let mut out = vec![0u8; cap * FR_BYTES];
    let mut written: u64 = 0;

    // Safety: every pointer is valid for the length passed alongside it, the
    // callee bounds-checks each column against `stage_len`, and it catches its
    // own exceptions.
    let code = unsafe {
        pilfflonk_combine(
            stage.as_ptr(),
            (stage.len() / FR_BYTES) as u64,
            stage_cols as u64,
            ids.as_ptr(),
            lens.as_ptr(),
            n as u64,
            out.as_mut_ptr(),
            cap as u64,
            &mut written,
        )
    };

    if code != 0 {
        return Err(Error::Native(last_error()));
    }
    out.truncate(written as usize * FR_BYTES);
    Ok(out)
}

/// Evaluate a polynomial at a point.
///
/// `coeffs` is in the key's representation, ascending degree; `x` and the
/// result are canonical big-endian, the form a proof records. This is what
/// produces the claimed openings a proof carries.
pub fn eval(coeffs: &[u8], x: &[u8; FR_BYTES]) -> Result<[u8; FR_BYTES], Error> {
    if !coeffs.len().is_multiple_of(FR_BYTES) {
        return Err(Error::Shape(format!(
            "coeffs is {} bytes, not a whole number of {FR_BYTES}-byte coefficients",
            coeffs.len()
        )));
    }

    let n = coeffs.len() / FR_BYTES;
    let mut out = [0u8; FR_BYTES];

    // Safety: `coeffs` holds the `n` elements just counted, `x` and `out` are
    // fixed-size arrays of exactly the width the callee reads and writes, and
    // it catches its own exceptions.
    let code = unsafe { pilfflonk_eval(coeffs.as_ptr(), n as u64, x.as_ptr(), out.as_mut_ptr()) };

    if code != 0 {
        return Err(Error::Native(last_error()));
    }
    Ok(out)
}

/// A point's canonical coordinates: `x` then `y`, big-endian, 32 bytes each.
///
/// This is the form a proof records and the form the Fiat-Shamir transcript
/// hashes, so it is how a commitment leaves the library. Infinity is all
/// zeroes.
pub fn to_bytes_be(point: &G1Affine) -> Result<[u8; 2 * FR_BYTES], Error> {
    let mut out = [0u8; 2 * FR_BYTES];

    // Safety: `point` is exactly the size the callee reads and `out` exactly
    // what it writes; it catches its own exceptions.
    let code = unsafe { pilfflonk_g1_to_bytes_be(point.as_ptr(), out.as_mut_ptr()) };

    if code != 0 {
        return Err(Error::Native(last_error()));
    }
    Ok(out)
}

/// The Fiat-Shamir transcript.
///
/// Wraps rapidsnark's `Keccak256Transcript` rather than reimplementing it. The
/// encoding has traps a second implementation would have to match exactly and
/// silently would not -- it is legacy Keccak-256 with `0x01` padding, not
/// SHA3's `0x06`, over fixed-width big-endian values.
///
/// Scalars and commitments go in canonical big-endian, the form a proof
/// records.
pub struct Transcript {
    handle: *mut std::ffi::c_void,
}

impl Transcript {
    pub fn new() -> Result<Self, Error> {
        // Safety: the constructor either returns an owned handle or null,
        // reporting the reason through last_error.
        let handle = unsafe { pilfflonk_transcript_new() };
        if handle.is_null() {
            return Err(Error::Native(last_error()));
        }
        Ok(Transcript { handle })
    }

    /// Clear it. The prover resets before each challenge rather than
    /// accumulating, so a caller that keeps appending diverges after the first.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.check(unsafe { pilfflonk_transcript_reset(self.handle) })
    }

    pub fn add_scalar(&mut self, value: &[u8; FR_BYTES]) -> Result<(), Error> {
        self.check(unsafe { pilfflonk_transcript_add_scalar(self.handle, value.as_ptr()) })
    }

    /// Add a commitment as canonical `x‖y`.
    pub fn add_commitment(&mut self, xy: &[u8; 2 * FR_BYTES]) -> Result<(), Error> {
        self.check(unsafe { pilfflonk_transcript_add_commitment(self.handle, xy.as_ptr()) })
    }

    pub fn challenge(&mut self) -> Result<[u8; FR_BYTES], Error> {
        let mut out = [0u8; FR_BYTES];
        self.check(unsafe { pilfflonk_transcript_challenge(self.handle, out.as_mut_ptr()) })?;
        Ok(out)
    }

    fn check(&self, code: i32) -> Result<(), Error> {
        if code != 0 {
            Err(Error::Native(last_error()))
        } else {
            Ok(())
        }
    }
}

impl Drop for Transcript {
    fn drop(&mut self) {
        // Safety: the handle came from pilfflonk_transcript_new and is freed
        // exactly once, since Transcript is not Copy or Clone.
        unsafe { pilfflonk_transcript_free(self.handle) }
    }
}

/// Whether a point is the representation's infinity.
pub fn is_infinity(point: &G1Affine) -> bool {
    point.iter().all(|&b| b == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intt_rejects_a_domain_that_is_not_a_power_of_two() {
        assert!(matches!(intt(&[0u8; FR_BYTES * 3], 3, 1), Err(Error::Shape(_))));
        assert!(matches!(intt(&[], 0, 1), Err(Error::Shape(_))));
    }

    #[test]
    fn intt_rejects_a_buffer_of_the_wrong_size() {
        assert!(matches!(intt(&[0u8; FR_BYTES * 3], 4, 1), Err(Error::Shape(_))));
        assert!(matches!(intt(&[0u8; FR_BYTES * 4], 4, 0), Err(Error::Shape(_))));
    }

    /// Interpolating a column of zeroes gives the zero polynomial, which is the
    /// cheapest check that the call reaches the C++ side and returns the right
    /// shape.
    #[test]
    fn intt_of_zeroes_is_zero() {
        let out = intt(&[0u8; FR_BYTES * 8], 8, 1).unwrap();
        assert_eq!(out.len(), FR_BYTES * 8);
        assert!(out.iter().all(|&b| b == 0));
    }

    /// The empty polynomial is zero everywhere, and a constant ignores the
    /// point -- the two ends of Horner's loop.
    #[test]
    fn eval_handles_the_degenerate_polynomials() {
        let x = [7u8; FR_BYTES];
        assert_eq!(eval(&[], &x).unwrap(), [0u8; FR_BYTES]);
        assert!(eval(&[0u8; FR_BYTES], &x).unwrap().iter().all(|&b| b == 0));
    }

    #[test]
    fn eval_rejects_a_ragged_coefficient_buffer() {
        assert!(matches!(eval(&[0u8; FR_BYTES + 1], &[0u8; FR_BYTES]), Err(Error::Shape(_))));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let one_point = [0u8; G1_AFFINE_BYTES];
        let two_scalars = [0u8; FR_BYTES * 2];
        assert!(matches!(msm(&one_point, &two_scalars), Err(Error::Shape(_))));
    }

    #[test]
    fn rejects_buffers_that_are_not_whole_elements() {
        assert!(matches!(msm(&[0u8; 63], &[0u8; FR_BYTES]), Err(Error::Shape(_))));
        assert!(matches!(msm(&[0u8; G1_AFFINE_BYTES], &[0u8; 31]), Err(Error::Shape(_))));
    }

    /// The empty sum is infinity, and crossing the boundary with nothing to do
    /// must not be an error.
    #[test]
    fn the_empty_sum_is_infinity() {
        let out = msm(&[], &[]).unwrap();
        assert!(is_infinity(&out));
    }

    /// Zero scalars against real points give infinity too, which is the check
    /// that the call reached the C++ side at all rather than short-circuiting.
    #[test]
    fn zero_scalars_give_infinity() {
        let ptau = [0u8; G1_AFFINE_BYTES * 4];
        let coeffs = [0u8; FR_BYTES * 4];
        assert!(is_infinity(&msm(&ptau, &coeffs).unwrap()));
    }
}

#[cfg(test)]
mod transcript_tests {
    use super::*;

    /// Values logged from a real run of pil-fflonk's prover: alpha and the W
    /// commitment produce challenge Y. This pins the whole encoding -- element
    /// order, width, the Keccak variant and the reduction -- against the
    /// prover, and is the check that wrapping the C++ transcript preserves what
    /// the previous Rust implementation was validated to do.
    const ALPHA: &str = "14560641611632097331351172125449579633774312004368406670544341241766131118492";
    const W_X: &str = "12419649577883164498958593584067611959324832377388732561925666137961148910692";
    const W_Y: &str = "21476078643738536497282674355688501033753144678841213640203400378618544673473";
    const EXPECTED_Y: &str = "10213856114127628084316690059097516037569626100727307492391249673580661585742";

    /// Decimal to fixed-width big-endian, by long multiplication. Written out
    /// rather than pulling in a bignum crate: this is the only place in this
    /// crate that needs it, and it is only for test vectors.
    fn be(decimal: &str) -> [u8; FR_BYTES] {
        let mut bytes = [0u8; FR_BYTES];
        for c in decimal.bytes() {
            let mut carry = (c - b'0') as u32;
            for b in bytes.iter_mut().rev() {
                let v = (*b as u32) * 10 + carry;
                *b = (v & 0xff) as u8;
                carry = v >> 8;
            }
            assert_eq!(carry, 0, "the value does not fit in {FR_BYTES} bytes");
        }
        bytes
    }

    #[test]
    fn reproduces_the_provers_challenge() {
        let mut t = Transcript::new().unwrap();
        t.add_scalar(&be(ALPHA)).unwrap();

        let mut xy = [0u8; 2 * FR_BYTES];
        xy[..FR_BYTES].copy_from_slice(&be(W_X));
        xy[FR_BYTES..].copy_from_slice(&be(W_Y));
        t.add_commitment(&xy).unwrap();

        assert_eq!(t.challenge().unwrap(), be(EXPECTED_Y), "the C++ transcript does not match the prover");
    }

    /// Reset clears it, so a reused transcript behaves like a fresh one.
    #[test]
    fn reset_clears_previous_items() {
        let mut t = Transcript::new().unwrap();
        t.add_scalar(&[7u8; FR_BYTES]).unwrap();
        t.reset().unwrap();

        t.add_scalar(&be(ALPHA)).unwrap();
        let mut xy = [0u8; 2 * FR_BYTES];
        xy[..FR_BYTES].copy_from_slice(&be(W_X));
        xy[FR_BYTES..].copy_from_slice(&be(W_Y));
        t.add_commitment(&xy).unwrap();

        assert_eq!(t.challenge().unwrap(), be(EXPECTED_Y));
    }
}
