//! Safe wrappers over the pilfflonk C API: every fallible call returns a `Result`.

use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::{c_int, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;

include!("../bindings_pilfflonk.rs");

/// Size of a BN254 scalar at the C API: a canonical (< r) little-endian integer.
pub const PILFFLONK_FR_BYTES: usize = 32;

/// Size of a G1 point at the C API: affine `x‖y`, each coordinate a canonical (< q) little-endian
/// integer of 32 bytes.
pub const PILFFLONK_G1_BYTES: usize = 64;

/// Size of a G2 point at the C API: affine `x‖y`, each coordinate an `Fq2` element `c0 + c1·u`
/// written `c0‖c1`, and each of the four `Fq` values a canonical (< q) little-endian integer of 32
/// bytes: `x.c0‖x.c1‖y.c0‖y.c1`.
pub const PILFFLONK_G2_BYTES: usize = 128;

/// The status code of a failed pilfflonk call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PilFflonkErrorKind {
    InvalidArgument,
    NonCanonical,
    Internal,
    InvalidPoint,
    /// A file cannot be opened, read or written.
    Io,
    /// A file is not in the format expected.
    Format,
    /// The witness does not satisfy the AIR's constraints: its constraint polynomial `Q` is not of
    /// its degree (spec A.1).
    Unsatisfied,
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

/// The kind of a failure status; `None` for `PILFFLONK_OK`.
fn error_kind(status: c_int) -> Option<PilFflonkErrorKind> {
    Some(match status {
        PILFFLONK_OK => return None,
        PILFFLONK_ERR_INVALID_ARGUMENT => PilFflonkErrorKind::InvalidArgument,
        PILFFLONK_ERR_NON_CANONICAL => PilFflonkErrorKind::NonCanonical,
        PILFFLONK_ERR_INTERNAL => PilFflonkErrorKind::Internal,
        PILFFLONK_ERR_INVALID_POINT => PilFflonkErrorKind::InvalidPoint,
        PILFFLONK_ERR_IO => PilFflonkErrorKind::Io,
        PILFFLONK_ERR_FORMAT => PilFflonkErrorKind::Format,
        PILFFLONK_ERR_UNSATISFIED => PilFflonkErrorKind::Unsatisfied,
        other => PilFflonkErrorKind::Unknown(other),
    })
}

/// Maps the status of the pilfflonk call just made on this thread to a `Result`.
fn check_status(status: c_int) -> Result<(), PilFflonkError> {
    error_kind(status).map_or(Ok(()), |kind| Err(last_error(kind)))
}

/// The failure of the pilfflonk call just made on this thread, which returned NULL: its status
/// comes from `pilfflonk_last_status`.
fn last_failure() -> PilFflonkError {
    // SAFETY: no arguments; it reads this thread's status of the latest call.
    let status = unsafe { pilfflonk_last_status() };
    // A NULL with PILFFLONK_OK would mean the C side and these bindings disagree.
    last_error(error_kind(status).unwrap_or(PilFflonkErrorKind::Unknown(status)))
}

/// A refusal decided on this side, before any call: it reads like the C side's.
fn invalid_argument(function: &str, message: String) -> PilFflonkError {
    PilFflonkError { kind: PilFflonkErrorKind::InvalidArgument, message: format!("{function}: {message}") }
}

/// `path` as the C side takes it: its bytes, NUL-terminated.
fn c_path(function: &str, path: &Path) -> Result<CString, PilFflonkError> {
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| invalid_argument(function, format!("{} contains a NUL byte", path.display())))
}

/// Checks that `scalar`, read as a little-endian integer, is below the BN254 scalar modulus r.
pub fn pilfflonk_fr_check_canonical_c(scalar: &[u8; PILFFLONK_FR_BYTES]) -> Result<(), PilFflonkError> {
    // SAFETY: `scalar` points to the 32 bytes the function reads.
    check_status(unsafe { pilfflonk_fr_check_canonical(scalar.as_ptr()) })
}

/// The Keccak-256 hash of `data` (Keccak's original padding, as Ethereum and snarkjs use it, not
/// SHA3-256): rapidsnark's `keccak_wrapper`, the hash of the transcript (spec A.4) and of the
/// vkey's digest (A.6).
pub fn pilfflonk_keccak256_c(data: &[u8]) -> Result<[u8; 32], PilFflonkError> {
    let mut hash = [0u8; 32];
    // SAFETY: `data` holds the `data.len()` bytes the call reads, and `hash` has the 32 bytes it
    // writes.
    check_status(unsafe { pilfflonk_keccak256(data.as_ptr(), data.len() as u64, hash.as_mut_ptr()) })?;
    Ok(hash)
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

/// Reads the first `n_g1` powers `[τ^i]₁`, and `[1]₂` and `[τ]₂`, of the snarkjs powers-of-tau file
/// at `ptau_path` (only those points, from sections 1 to 3), and writes them to `srs_path` as
/// `pilfflonk.srs.bin` (spec §4.2.5 and A.6), replacing any file there.
///
/// Fails with [`InvalidArgument`](PilFflonkErrorKind::InvalidArgument) if `n_g1` is 0 or above
/// 2^32 - 1, or the ptau holds fewer powers; [`Io`](PilFflonkErrorKind::Io) if a file cannot be
/// opened, read or written; [`Format`](PilFflonkErrorKind::Format) if the ptau is not a BN254 one,
/// is cut short or holds a point that is not valid.
pub fn pilfflonk_srs_from_ptau_c(ptau_path: &Path, n_g1: u64, srs_path: &Path) -> Result<(), PilFflonkError> {
    const FUNCTION: &str = "pilfflonk_srs_from_ptau";
    let ptau = c_path(FUNCTION, ptau_path)?;
    let srs = c_path(FUNCTION, srs_path)?;
    // SAFETY: both paths are NUL-terminated strings that outlive the call.
    check_status(unsafe { pilfflonk_srs_from_ptau(ptau.as_ptr(), n_g1, srs.as_ptr()) })
}

/// The structured reference string of a proof, owned by the C++ side: the powers `[τ^i]₁` and
/// `[1]₂`, `[τ]₂` loaded from `pilfflonk.srs.bin`, whose points are checked as they are read.
#[derive(Debug)]
pub struct PilFflonkSrs {
    handle: NonNull<c_void>,
}

impl PilFflonkSrs {
    /// Loads `pilfflonk.srs.bin`. Fails with [`Io`](PilFflonkErrorKind::Io) if the file cannot be
    /// opened or read, [`Format`](PilFflonkErrorKind::Format) if it is not such a file.
    pub fn load(srs_path: &Path) -> Result<Self, PilFflonkError> {
        let path = c_path("pilfflonk_srs_load", srs_path)?;
        // SAFETY: `path` is a NUL-terminated string that outlives the call; the result is either
        // NULL or a handle this value then owns.
        let handle = unsafe { pilfflonk_srs_load(path.as_ptr()) };
        NonNull::new(handle).map(|handle| Self { handle }).ok_or_else(last_failure)
    }

    /// `[τ^i]₂` for `i` = 0 (`[1]₂`) or 1 (`[τ]₂`), as [`PILFFLONK_G2_BYTES`] describes: the points of
    /// the verifier's pairing (spec A.5), and `[τ]₂` the vkey's `X_2` (A.6). Fails with
    /// [`InvalidArgument`](PilFflonkErrorKind::InvalidArgument) for any other `i`.
    pub fn g2(&self, i: u64) -> Result<[u8; PILFFLONK_G2_BYTES], PilFflonkError> {
        let mut point = [0u8; PILFFLONK_G2_BYTES];
        // SAFETY: `point` has the 128 bytes the call writes, and the handle is live.
        check_status(unsafe { pilfflonk_srs_g2(self.handle.as_ptr(), i, point.as_mut_ptr()) })?;
        Ok(point)
    }

    /// The KZG commitment `[f(τ)]₁` of a fixed `f(X) = Σ_{j<k} p_j(X^k)·X^j` (spec §4.2.5), where
    /// `p_j` interpolates column `j` on the domain of `N = 2^n_bits` points: `evals` holds the `k`
    /// columns one after another, `N` canonical scalars each, in the domain's natural order. The
    /// commitment is affine `x‖y`, canonical little-endian coordinates; the point at infinity is
    /// all zeros.
    ///
    /// Fails with [`InvalidArgument`](PilFflonkErrorKind::InvalidArgument) if `evals` does not
    /// hold `k·2^n_bits` scalars (checked here, before the call), if `k` is 0, `n_bits` exceeds
    /// 28 or `k·N` exceeds the SRS's powers; [`NonCanonical`](PilFflonkErrorKind::NonCanonical) if
    /// a scalar is not below r.
    pub fn commit_fixed(
        &self,
        n_bits: u64,
        k: u64,
        evals: &[[u8; PILFFLONK_FR_BYTES]],
    ) -> Result<[u8; PILFFLONK_G1_BYTES], PilFflonkError> {
        let expected =
            u32::try_from(n_bits).ok().and_then(|bits| 1u64.checked_shl(bits)).and_then(|n| k.checked_mul(n));
        if expected != Some(evals.len() as u64) {
            return Err(invalid_argument(
                "pilfflonk_commit_fixed",
                format!("evals holds {} scalars, not the k·2^n_bits = {k}·2^{n_bits} the call reads", evals.len()),
            ));
        }
        let mut commitment = [0u8; PILFFLONK_G1_BYTES];
        // SAFETY: `evals` holds the k·2^n_bits scalars the call reads, `commitment` has the 64 bytes
        // it writes, and the handle is live.
        check_status(unsafe {
            pilfflonk_commit_fixed(
                self.handle.as_ptr(),
                n_bits,
                k,
                evals.as_flattened().as_ptr(),
                commitment.as_mut_ptr(),
            )
        })?;
        Ok(commitment)
    }
}

impl Drop for PilFflonkSrs {
    fn drop(&mut self) {
        // SAFETY: the handle came from `pilfflonk_srs_load` and is released only here.
        unsafe { pilfflonk_srs_free(self.handle.as_ptr()) }
    }
}

/// `n` points of [`PILFFLONK_G1_BYTES`] from a buffer the C side wrote.
fn points(bytes: &[u8]) -> Vec<[u8; PILFFLONK_G1_BYTES]> {
    bytes
        .chunks_exact(PILFFLONK_G1_BYTES)
        .map(|chunk| {
            let mut point = [0u8; PILFFLONK_G1_BYTES];
            point.copy_from_slice(chunk);
            point
        })
        .collect()
}

/// A slice's pointer for the C side, NULL for an empty one: the C API reads nothing then, and a
/// dangling pointer of an empty slice is not one to hand it.
fn ptr_or_null<T>(slice: &[T]) -> *const u8 {
    if slice.is_empty() {
        std::ptr::null()
    } else {
        slice.as_ptr().cast()
    }
}

/// The proving key of the prover (spec §4.4, step 1), owned by the C++ side: the `provingKey/`
/// that `setup-pilfflonk` writes, loaded, with the fixed columns interpolated. Immutable.
#[derive(Debug)]
pub struct PilFflonkProverCtx {
    handle: NonNull<c_void>,
}

impl PilFflonkProverCtx {
    /// Loads the `provingKey/` at `dir`: `pilout.globalInfo.json`, the SRS and every AIR's
    /// pilfflonkinfo, `.bin` and `.const` (not the vkey, whose digest the caller absorbs). Fails with
    /// [`Io`](PilFflonkErrorKind::Io) if a file cannot be read, [`Format`](PilFflonkErrorKind::Format)
    /// if one is not what it should be or they do not agree.
    pub fn load(dir: &Path) -> Result<Self, PilFflonkError> {
        let path = c_path("pilfflonk_ctx_new", dir)?;
        // SAFETY: `path` is a NUL-terminated string that outlives the call; the result is either
        // NULL or a handle this value then owns.
        let handle = unsafe { pilfflonk_ctx_new(path.as_ptr()) };
        NonNull::new(handle).map(|handle| Self { handle }).ok_or_else(last_failure)
    }

    /// The `nBitsExt` of an AIR (spec A.1), as the C++ side derives it.
    pub fn n_bits_ext(&self, airgroup_id: u64, air_id: u64) -> Result<u64, PilFflonkError> {
        let mut out = 0u64;
        // SAFETY: the handle is live and `out` is a u64 the call writes.
        check_status(unsafe { pilfflonk_ctx_n_bits_ext(self.handle.as_ptr(), airgroup_id, air_id, &mut out) })?;
        Ok(out)
    }
}

impl Drop for PilFflonkProverCtx {
    fn drop(&mut self) {
        // SAFETY: the handle came from `pilfflonk_ctx_new` and is released only here; the instances
        // that borrow it are gone.
        unsafe { pilfflonk_ctx_free(self.handle.as_ptr()) }
    }
}

/// What an instance is made of (`pilfflonk_instance_new`): scalars are canonical little-endian.
#[derive(Clone, Copy, Debug)]
pub struct PilFflonkInstanceInputs<'a> {
    pub airgroup_id: u64,
    pub air_id: u64,
    /// The stage-1 witness, as the witness directory's `.bin` holds it (spec A.6): row after row,
    /// the stage-1 columns of each row.
    pub stage1: &'a [u8],
    /// The stage-1 air values, publics and stage-1 proof values.
    pub air_values: &'a [[u8; PILFFLONK_FR_BYTES]],
    pub publics: &'a [[u8; PILFFLONK_FR_BYTES]],
    pub proof_values: &'a [[u8; PILFFLONK_FR_BYTES]],
    /// `None` for a real proof, blinded with the OS's randomness. `Some(seed)` fixes the blinding
    /// (decision D6): for tests and CI only, as whoever knows the seed can remove the blinding.
    pub insecure_blinding_seed: Option<&'a [u8; 32]>,
}

/// An instance of an AIR being proved (spec §4.4, steps 2 and 3), owned by the C++ side. It
/// borrows the context it was made from.
#[derive(Debug)]
pub struct PilFflonkInstance<'ctx> {
    handle: NonNull<c_void>,
    _ctx: PhantomData<&'ctx PilFflonkProverCtx>,
}

impl<'ctx> PilFflonkInstance<'ctx> {
    /// Fails with [`InvalidArgument`](PilFflonkErrorKind::InvalidArgument) if there is no such AIR
    /// or a count or the size of the witness is not the AIR's, and
    /// [`NonCanonical`](PilFflonkErrorKind::NonCanonical) if a scalar is not below r.
    pub fn new(ctx: &'ctx PilFflonkProverCtx, inputs: &PilFflonkInstanceInputs<'_>) -> Result<Self, PilFflonkError> {
        let seed = inputs.insecure_blinding_seed.map_or(std::ptr::null(), |seed| seed.as_ptr());
        // SAFETY: every pointer is NULL with a count of 0 or holds the count of 32-byte scalars (or
        // the bytes of stage1) the call reads, the seed is NULL or 32 bytes, and the context is live;
        // the result is either NULL or a handle this value then owns.
        let handle = unsafe {
            pilfflonk_instance_new(
                ctx.handle.as_ptr(),
                inputs.airgroup_id,
                inputs.air_id,
                ptr_or_null(inputs.stage1),
                inputs.stage1.len() as u64,
                ptr_or_null(inputs.air_values),
                inputs.air_values.len() as u64,
                ptr_or_null(inputs.publics),
                inputs.publics.len() as u64,
                ptr_or_null(inputs.proof_values),
                inputs.proof_values.len() as u64,
                seed,
            )
        };
        NonNull::new(handle).map(|handle| Self { handle, _ctx: PhantomData }).ok_or_else(last_failure)
    }

    /// Commits stage `stage` with its challenges (none for stage 1): the commitments of its `n_out`
    /// f, in the order of the layout.
    pub fn commit_stage(
        &mut self,
        stage: u32,
        challenges: &[[u8; PILFFLONK_FR_BYTES]],
        n_out: usize,
    ) -> Result<Vec<[u8; PILFFLONK_G1_BYTES]>, PilFflonkError> {
        let mut out = vec![0u8; n_out * PILFFLONK_G1_BYTES];
        // SAFETY: `challenges` holds the scalars and `out` the room for the points the call reads
        // and writes, and the handle is live.
        check_status(unsafe {
            pilfflonk_commit_stage(
                self.handle.as_ptr(),
                stage,
                ptr_or_null(challenges),
                challenges.len() as u64,
                out.as_mut_ptr(),
                n_out as u64,
            )
        })?;
        Ok(points(&out))
    }

    /// Commits `Q` with the challenges of its stage (`std_vc`): the commitments of its `n_out` f.
    /// Fails with [`Unsatisfied`](PilFflonkErrorKind::Unsatisfied) if the witness does not satisfy
    /// the AIR's constraints.
    pub fn commit_q(
        &mut self,
        challenges: &[[u8; PILFFLONK_FR_BYTES]],
        n_out: usize,
    ) -> Result<Vec<[u8; PILFFLONK_G1_BYTES]>, PilFflonkError> {
        let mut out = vec![0u8; n_out * PILFFLONK_G1_BYTES];
        // SAFETY: as in commit_stage.
        check_status(unsafe {
            pilfflonk_commit_q(
                self.handle.as_ptr(),
                ptr_or_null(challenges),
                challenges.len() as u64,
                out.as_mut_ptr(),
                n_out as u64,
            )
        })?;
        Ok(points(&out))
    }
}

impl Drop for PilFflonkInstance<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle came from `pilfflonk_instance_new` and is released only here; the
        // openings that borrow it are gone.
        unsafe { pilfflonk_instance_free(self.handle.as_ptr()) }
    }
}

/// What the SHPLONK opening adds to the proof (spec A.4 step 5, A.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PilFflonkOpeningProof {
    pub w: [u8; PILFFLONK_G1_BYTES],
    pub wp: [u8; PILFFLONK_G1_BYTES],
    /// The inverse of the product of the denominators the verifier inverts in its SHPLONK check
    /// (`verifierInverse`, `pil2-stark/src/pilfflonk/pilfflonk_shplonk_prover.hpp`).
    pub inv: [u8; PILFFLONK_FR_BYTES],
    /// `1/Z_H(ξ)`.
    pub inv_zh: [u8; PILFFLONK_FR_BYTES],
}

/// The opening of a proof (spec §4.4, steps 4 and 5), owned by the C++ side: every f of its
/// instances evaluated at `ξ = xiSeed^powerW`. It borrows the instances.
#[derive(Debug)]
pub struct PilFflonkOpening<'a> {
    handle: NonNull<c_void>,
    _instances: PhantomData<&'a ()>,
}

impl<'a> PilFflonkOpening<'a> {
    /// The opening of `instances`, in canonical order and with `Q` committed, at `xi_seed`.
    pub fn new(
        instances: &[&'a PilFflonkInstance<'_>],
        xi_seed: &[u8; PILFFLONK_FR_BYTES],
    ) -> Result<Self, PilFflonkError> {
        let handles: Vec<*const c_void> = instances.iter().map(|i| i.handle.as_ptr().cast_const()).collect();
        // SAFETY: `handles` holds live instance handles, which the borrow keeps alive and unchanged
        // while the opening is; the result is either NULL or a handle this value then owns.
        let handle = unsafe { pilfflonk_opening_new(handles.as_ptr(), handles.len() as u64, xi_seed.as_ptr()) };
        NonNull::new(handle).map(|handle| Self { handle, _instances: PhantomData }).ok_or_else(last_failure)
    }

    /// The evaluations of the proof, in the order of spec A.4 step 4 and of the proof.
    pub fn evaluations(&self) -> Result<Vec<[u8; PILFFLONK_FR_BYTES]>, PilFflonkError> {
        // SAFETY: the handle is live.
        let n = unsafe { pilfflonk_opening_n_evaluations(self.handle.as_ptr()) } as usize;
        let mut out = vec![[0u8; PILFFLONK_FR_BYTES]; n];
        // SAFETY: `out` has room for the `n` scalars the call writes, and the handle is live.
        check_status(unsafe {
            pilfflonk_opening_evaluations(self.handle.as_ptr(), n as u64, out.as_flattened_mut().as_mut_ptr())
        })?;
        Ok(out)
    }

    /// `Q(ξ)` of instance `instance`: for tests and diagnostics, not part of the proof.
    pub fn q(&self, instance: u64) -> Result<[u8; PILFFLONK_FR_BYTES], PilFflonkError> {
        let mut out = [0u8; PILFFLONK_FR_BYTES];
        // SAFETY: `out` has the 32 bytes the call writes, and the handle is live.
        check_status(unsafe { pilfflonk_opening_q(self.handle.as_ptr(), instance, out.as_mut_ptr()) })?;
        Ok(out)
    }

    /// SHPLONK on `transcript`, which must hold everything absorbed before it (the evaluations
    /// last): squeezes `α_S`, absorbs `[W]₁`, squeezes `y`. On a failure after the transcript has
    /// moved on, the proof must be abandoned.
    pub fn open(&self, transcript: &mut PilFflonkTranscript) -> Result<PilFflonkOpeningProof, PilFflonkError> {
        let mut proof = PilFflonkOpeningProof {
            w: [0; PILFFLONK_G1_BYTES],
            wp: [0; PILFFLONK_G1_BYTES],
            inv: [0; PILFFLONK_FR_BYTES],
            inv_zh: [0; PILFFLONK_FR_BYTES],
        };
        // SAFETY: both handles are live, and each output has the bytes the call writes.
        check_status(unsafe {
            pilfflonk_opening_open(
                self.handle.as_ptr(),
                transcript.handle.as_ptr(),
                proof.w.as_mut_ptr(),
                proof.wp.as_mut_ptr(),
                proof.inv.as_mut_ptr(),
                proof.inv_zh.as_mut_ptr(),
            )
        })?;
        Ok(proof)
    }
}

impl Drop for PilFflonkOpening<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle came from `pilfflonk_opening_new` and is released only here.
        unsafe { pilfflonk_opening_free(self.handle.as_ptr()) }
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

    /// 32 bytes from 64 hex digits in the order they are written: a hash as Keccak outputs it.
    fn hash(hex: &str) -> [u8; 32] {
        let mut bytes = from_hex(hex);
        bytes.reverse();
        bytes
    }

    /// Keccak-256, not SHA3-256: "" and "abc" are the published vectors; the others, bytes `i mod
    /// 256` of the lengths around the rate (136 bytes), are those of @noble/hashes' `keccak_256`, an
    /// implementation independent of this one. The C++ test pins the same ones.
    #[test]
    fn keccak256_gives_the_pinned_hashes() {
        let counting: Vec<u8> = (0..272u32).map(|i| i as u8).collect();
        let vectors: [(&[u8], &str); 6] = [
            (b"", "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            (b"abc", "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
            (&counting[..135], "cbdfd9dee5faad3818d6b06f95a219fd290b0e1706f6a82e5a595b9ce9faca62"),
            (&counting[..136], "7ce759f1ab7f9ce437719970c26b0a66ff11fe3e38e17df89cf5d29c7d7f807e"),
            (&counting[..137], "ac73d4fae68b8453f764007c1a20ce95994187861f0c3227a3a8e99a73a3b1db"),
            (&counting, "fdf2ec49e749960d3c8521a0219af8d03e30e2b3bf19bd16150ee0eaf133d66e"),
        ];
        for (data, expected) in vectors {
            assert_eq!(pilfflonk_keccak256_c(data).unwrap(), hash(expected), "{} bytes", data.len());
        }
    }

    #[test]
    fn transcript_refuses_to_squeeze_before_absorbing() {
        let mut transcript = PilFflonkTranscript::new().unwrap();
        let err = transcript.squeeze().unwrap_err();
        assert_eq!(err.kind, PilFflonkErrorKind::InvalidArgument, "{err}");
        assert!(err.message.contains("pilfflonk_transcript_squeeze"), "{err}");
    }
}
