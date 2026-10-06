//! The pilfflonk final SNARK of the wrap (`setup-snark --final-snark pilfflonk`): the key
//! `provingKeySnark/final/provingKey/` in place of rapidsnark's `final.zkey`.
//!
//! - [`PilfflonkWrapProver`] proves a recursivef proof: the wrap's witness, the stage-1 columns of
//!   the AIR plonk2pil makes of the final circuit, computed in this process from the recursivef proof
//!   by `pilfflonk_wrap_witness` (the circuit's `final.so`, `final.dat` and `final.exec`), and then
//!   pilfflonk's proof of it;
//! - a proof's bytes are pilfflonk's ([`Proof::to_bytes`], pilfflonk/docs/formats.md#proof): only the
//!   vkey's names say where their parts end, and their length is the key's. Its one public is the
//!   final circuit's publics hash, 32 bytes big-endian, as rapidsnark writes the publics of PLONK and
//!   FFLONK;
//! - [`json_views`] names those bytes with the vkey, `proof.json` and `publics.json`, and
//!   [`verify_json`] runs pilfflonk's JS verifier on them, as `verify_snark_proof` runs snarkjs for
//!   PLONK and FFLONK.
//!
//! On the GPU the key's proofs keep their data in device memory of the wrap's ([`wrap_arena`],
//! pilfflonk/docs/performance.md#the-wraps-device-buffer), as rapidsnark's PLONK prover is carved out
//! of it by `pre_allocate_final_snark_prover_c`: proofman's unified buffer if the wrap has one, and
//! otherwise the recursivef's prover buffer, which the recursivef is done with when pilfflonk proves.

use std::ffi::c_void;
use std::path::Path;

use pilfflonk_wrap_witness::{WrapArtifacts, WrapWitness};
use proofman_common::{ProofmanError, ProofmanResult};
use proofman_pilfflonk::{
    gpu_device_bytes, gpu_free_bytes, js_verifier, prove, prove_exec, Device, DeviceBytes, FrBytes, JsonFile, Proof,
    ProofJson, ProofNames, ProveOptions, ProvingKey, ProvingKeyFiles, Publics, Vkey, Witness, WitnessShape,
};
use proofman_starks_lib_c::{
    acquire_first_gpu_buffer_c, get_aux_trace_end_c, get_first_gpu_buffer_c, get_first_gpu_id_c,
    get_stream_commit_floor_c, get_unified_buffer_gpu_size_c, release_first_gpu_buffer_c,
    reserve_recursivef_aux_trace_c,
};
use proofman_util::{timer_start_info, timer_stop_and_log_info};

/// Bytes of a public of a pilfflonk proof in a `SnarkProof`: an element of `Fr`, big-endian.
const PUBLIC_BYTES: usize = 32;

/// The device memory of the pilfflonk wrap's proofs on the GPU (their arena): `bytes` bytes at
/// `buffer` on CUDA device 0, of the wrap's ([`wrap_arena`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct WrapArena {
    pub(crate) buffer: *mut c_void,
    pub(crate) bytes: u64,
}

/// proofman's unified buffer of `d_buffers` on device 0, from where its aux traces start to where
/// they end, below its streaming-commit slots: nothing proofman keeps across proofs is there, so the
/// wrap leaves nothing to reload.
pub(crate) fn unified_buffer(d_buffers: *mut c_void) -> ProofmanResult<WrapArena> {
    let gpu_id = get_first_gpu_id_c(d_buffers);
    if gpu_id != 0 {
        return Err(ProofmanError::InvalidConfiguration(format!(
            "pilfflonk proves on CUDA device 0, and proofman's unified buffer is on device {gpu_id}"
        )));
    }
    // The first GPU's, as get_aux_trace_end measures: not the calling thread's current device's.
    Ok(WrapArena { buffer: get_first_gpu_buffer_c(d_buffers), bytes: get_aux_trace_end_c(d_buffers) })
}

/// proofman's first GPU borrowed (`acquire_first_gpu_buffer`) while a wrap writes its unified
/// buffer: no proofman work is placed there meanwhile, and the release drops the streams' cached
/// const pols and trees the wrap overwrote. None without proofman's buffers.
pub(crate) struct FirstGpuBorrow(*mut c_void);

impl FirstGpuBorrow {
    pub(crate) fn new(d_buffers: Option<*mut c_void>) -> Option<Self> {
        d_buffers.map(|d_buffers| {
            acquire_first_gpu_buffer_c(d_buffers);
            Self(d_buffers)
        })
    }
}

impl Drop for FirstGpuBorrow {
    fn drop(&mut self) {
        release_first_gpu_buffer_c(self.0);
    }
}

/// pilfflonk's arena in the wrap on the GPU, for the key at `proving_key`, as
/// `pre_allocate_final_snark_prover_c` carves rapidsnark's PLONK prover: proofman's unified buffer
/// `d_buffers` if there is one, its aux traces ([`unified_buffer`]; the key refuses it, as a GPU without the memory, if it
/// holds less than a proof's arena: pilfflonk/docs/performance.md#rules-of-the-device-path);
/// otherwise the recursivef's prover buffer `d_buffers_recursivef`, grown to the arena if it is
/// smaller. `None` on the CPU. Refused if the unified buffer is not on device 0, where pilfflonk
/// proves, if the recursivef's buffer cannot grow, or if, the arena in place, the device has less
/// free memory than the key needs beside it ([`DeviceBytes::beside`], its loading's scratch
/// included): before the key loads anything.
pub(crate) fn wrap_arena(
    gpu: bool,
    d_buffers: Option<*mut c_void>,
    d_buffers_recursivef: *mut c_void,
    proving_key: &Path,
) -> ProofmanResult<Option<WrapArena>> {
    if !gpu {
        return Ok(None);
    }
    let needed = gpu_device_bytes(proving_key).map_err(|e| invalid_key(proving_key, e))?;
    if let Some(d_buffers) = d_buffers {
        let mut arena = unified_buffer(d_buffers)?;
        // Past the aux traces only without streaming-commit slots, which commit during a borrow: the
        // whole buffer then, whose const pols the wrapper reloads and whose stagings the borrow drops.
        if needed.arena > arena.bytes && get_stream_commit_floor_c(d_buffers) == u64::MAX {
            arena.bytes = get_unified_buffer_gpu_size_c(d_buffers);
        }
        require_beside(&needed, gpu_free_bytes().map_err(|e| invalid_key(proving_key, e))?)?;
        tracing::info!(
            "pilfflonk's GPU arena: {} bytes of the unified buffer's {} (margin {}), and {} bytes beside it",
            needed.arena,
            arena.bytes,
            i128::from(arena.bytes) - i128::from(needed.arena),
            needed.beside
        );
        return Ok(Some(arena));
    }
    match reserve_recursivef_aux_trace_c(d_buffers_recursivef, needed.arena) {
        Ok((buffer, bytes)) => {
            require_beside(&needed, gpu_free_bytes().map_err(|e| invalid_key(proving_key, e))?)?;
            tracing::info!(
                "pilfflonk's GPU arena: {} bytes of the recursivef's prover buffer of {bytes}, and {} bytes beside it",
                needed.arena,
                needed.beside
            );
            Ok(Some(WrapArena { buffer, bytes }))
        }
        Err(most) => Err(ProofmanError::InvalidConfiguration(format!(
            "not enough GPU memory for the pilfflonk wrap's arena: it needs {} bytes, and the recursivef's prover \
             buffer can hold {most} (pilfflonk/docs/performance.md#selection-memory-and-errors)",
            needed.arena
        ))),
    }
}

/// Whether the arena of the key at `proving_key` passes the aux traces of the unified buffer
/// `d_buffers` ([`wrap_arena`]), or cannot be sized.
pub(crate) fn arena_passes_aux_traces(d_buffers: *mut c_void, proving_key: &Path) -> bool {
    gpu_device_bytes(proving_key).map_or(true, |needed| needed.arena > get_aux_trace_end_c(d_buffers))
}

/// Refuses a key that needs more device memory beside its arena (`needed.beside`) than the device has
/// `free`, once the arena is in place, as the key would refuse itself while it loads
/// (pilfflonk/docs/performance.md#rules-of-the-device-path).
fn require_beside(needed: &DeviceBytes, free: u64) -> ProofmanResult<()> {
    if free < needed.beside {
        return Err(ProofmanError::InvalidConfiguration(format!(
            "not enough GPU memory for the pilfflonk wrap's key: beside its arena of {} bytes it needs {} bytes, \
             and the device has {free} free (pilfflonk/docs/performance.md#selection-memory-and-errors)",
            needed.arena, needed.beside
        )));
    }
    Ok(())
}

/// The prover of the pilfflonk wrap: its key, loaded on its device, and the final circuit's witness
/// calculator and exec, which compute the wrap's witness from a recursivef proof.
pub(crate) struct PilfflonkWrapProver {
    key: ProvingKey,
    shape: WitnessShape,
    witness: WrapWitness,
    /// The key is on the GPU, which builds the blake3 wrap's witness from its parts.
    on_gpu: bool,
    /// The key's device memory is in a buffer others use between the proofs, written back before each.
    restorable: bool,
}

impl PilfflonkWrapProver {
    /// Loads the key at `proving_key` (`final/provingKey/`), on the GPU with an `arena`
    /// ([`wrap_arena`]'s, `None` on the CPU) and on the CPU otherwise, and the witness calculator and
    /// exec of the stem `setup_snark_path` (`final/final`: `WrapArtifacts::with_stem`). On the GPU
    /// the key's proofs keep their data in the arena ([`ProvingKey::load_on_device_buffer`]).
    ///
    /// # Safety
    ///
    /// An `arena` must be device memory of its bytes on device 0 that outlives the prover, and that
    /// nothing else uses while it proves.
    pub(crate) unsafe fn load(
        setup_snark_path: &Path,
        proving_key: &Path,
        arena: Option<WrapArena>,
    ) -> ProofmanResult<Self> {
        let key = match arena {
            // SAFETY: the caller's, as this function's contract says.
            Some(arena) => unsafe { ProvingKey::load_on_device_buffer(proving_key, arena.buffer, arena.bytes) },
            None => ProvingKey::load_on(proving_key, Device::Cpu),
        }
        .map_err(|e| invalid_key(proving_key, e))?;
        let shape = key.witness_shape().map_err(|e| invalid_key(proving_key, e))?;
        let witness = load_wrap_witness(setup_snark_path)?;
        Ok(Self { key, shape, witness, on_gpu: arena.is_some(), restorable: false })
    }

    /// [`load`](Self::load), the key on `device`: the wrap of a blake3 key, whose recursivef is
    /// proved before it ([`SnarkWrapper`](crate::SnarkWrapper)), with all
    /// its device memory in `buffer` if given (a wrap's buffer it is done with,
    /// [`ProvingKey::load_in_device_buffer`]), of its own otherwise.
    ///
    /// # Safety
    ///
    /// A `buffer` must be device memory of its bytes on device 0 that outlives the prover, and that
    /// nothing else uses while the prover lives.
    pub(crate) unsafe fn load_on(
        setup_snark_path: &Path,
        proving_key: &Path,
        device: Device,
        buffer: Option<WrapArena>,
        restorable: bool,
    ) -> ProofmanResult<Self> {
        let on_gpu = matches!(device, Device::Gpu);
        // The witness's files load while the key does.
        let (key, witness) = std::thread::scope(|scope| {
            let witness = scope.spawn(|| load_wrap_witness(setup_snark_path));
            let key = match buffer {
                // SAFETY: the caller's, as this function's contract says.
                Some(b) => unsafe { ProvingKey::load_in_device_buffer(proving_key, b.buffer, b.bytes, restorable) },
                None => ProvingKey::load_on(proving_key, device),
            }
            .map_err(|e| invalid_key(proving_key, e));
            (key, witness.join().expect("the wrap witness's loading thread panicked"))
        });
        let key = key?;
        let shape = key.witness_shape().map_err(|e| invalid_key(proving_key, e))?;
        let witness = witness?;
        if on_gpu {
            // The parts every proof's witness shares, on the device once; then all of the key is there.
            if let (Some(exec), [air]) = (witness.exec_static(), shape.airs()) {
                let air = proofman_pilfflonk::AirInstanceRef { airgroup_id: air.airgroup_id, air_id: air.air_id };
                key.set_exec(air, &exec).map_err(|e| invalid_key(proving_key, e))?;
            }
            if restorable {
                key.snapshot_device().map_err(|e| invalid_key(proving_key, e))?;
            }
        }
        Ok(Self { key, shape, witness, on_gpu, restorable })
    }

    /// The proof of the zkin `zkin`, a JSON object: its bytes and the bytes of its publics. On the
    /// GPU the device builds the witness from its parts ([`WrapWitness::device_witness`]).
    pub(crate) fn prove_zkin(&self, zkin: &serde_json::Value) -> ProofmanResult<(Vec<u8>, Vec<u8>)> {
        if self.on_gpu {
            timer_start_info!(CALCULATE_FINAL_WITNESS);
            // The key's device memory written back while the host computes the witness.
            // SAFETY: the witness never touches the key: the restore's thread is its only user meanwhile.
            struct KeyRef<'a>(&'a ProvingKey);
            unsafe impl Send for KeyRef<'_> {}
            let key = KeyRef(&self.key);
            let (device, restored) = std::thread::scope(|scope| {
                let restored = self.restorable.then(move || {
                    scope.spawn(move || {
                        let key = key;
                        key.0.restore_device()
                    })
                });
                let device = self.witness.device_witness(&self.shape, zkin);
                (device, restored.map(|r| r.join().expect("the key's restore panicked")))
            });
            if let Some(restored) = restored {
                restored.map_err(|e| invalid_key(self.key.dir(), e))?;
            }
            let device = device.map_err(|e| {
                ProofmanError::InvalidProof(format!("The pilfflonk wrap's witness cannot be computed: {e}"))
            })?;
            timer_stop_and_log_info!(CALCULATE_FINAL_WITNESS);
            timer_start_info!(CALCULATE_FINAL_PROOF);
            let output =
                prove_exec(&self.key, device.air, device.exec(), device.publics.clone(), &ProveOptions::default())
                    .map_err(|e| ProofmanError::InvalidProof(format!("The pilfflonk prover failed: {e}")))?;
            timer_stop_and_log_info!(CALCULATE_FINAL_PROOF);
            return Ok((output.proof.to_bytes(), publics_to_bytes(&output.publics)));
        }
        timer_start_info!(CALCULATE_FINAL_WITNESS);
        let witness = self.witness.witness_from_zkin(&self.shape, zkin).map_err(|e| {
            ProofmanError::InvalidProof(format!("The pilfflonk wrap's witness cannot be computed: {e}"))
        })?;
        timer_stop_and_log_info!(CALCULATE_FINAL_WITNESS);
        self.prove_witness(&witness)
    }

    fn prove_witness(&self, witness: &Witness) -> ProofmanResult<(Vec<u8>, Vec<u8>)> {
        timer_start_info!(CALCULATE_FINAL_PROOF);
        let output = prove(&self.key, witness, &ProveOptions::default())
            .map_err(|e| ProofmanError::InvalidProof(format!("The pilfflonk prover failed: {e}")))?;
        timer_stop_and_log_info!(CALCULATE_FINAL_PROOF);
        Ok((output.proof.to_bytes(), publics_to_bytes(&output.publics)))
    }

    /// The proof of `recursivef_proof`: its bytes and the bytes of its publics.
    ///
    /// # Safety
    ///
    /// `recursivef_proof` must be the recursivef proof, the `nlohmann::json` that
    /// `gen_recursive_proof_final_c` returns, alive and used by nothing else during the call.
    pub(crate) unsafe fn prove(&self, recursivef_proof: *mut c_void) -> ProofmanResult<(Vec<u8>, Vec<u8>)> {
        let witness = wrap_witness(&self.witness, &self.shape, recursivef_proof)?;
        self.prove_witness(&witness)
    }
}

/// The wrap's witness of `recursivef_proof`, as the prover computes it, without proving: what
/// `prove-snark --only-recursivef` checks of a pilfflonk final SNARK, as it computes the final
/// circuit's witness of a PLONK or FFLONK one. It reads the key's files the witness's shape comes
/// from (`ProvingKeyFiles`), and nothing of the C++ core.
///
/// # Safety
///
/// As [`PilfflonkWrapProver::prove`].
pub(crate) unsafe fn check_wrap_witness(
    setup_snark_path: &Path,
    proving_key: &Path,
    recursivef_proof: *mut c_void,
) -> ProofmanResult<()> {
    let shape = ProvingKeyFiles::read(proving_key)
        .and_then(|files| files.witness_shape())
        .map_err(|e| invalid_key(proving_key, e))?;
    wrap_witness(&load_wrap_witness(setup_snark_path)?, &shape, recursivef_proof)?;
    Ok(())
}

/// The witness calculator and exec of the stem `setup_snark_path`.
fn load_wrap_witness(setup_snark_path: &Path) -> ProofmanResult<WrapWitness> {
    WrapWitness::load(&WrapArtifacts::with_stem(setup_snark_path))
        .map_err(|e| ProofmanError::InvalidSetup(format!("The pilfflonk wrap's witness cannot be loaded: {e}")))
}

/// The wrap's witness of `recursivef_proof`, of the key's `shape`.
///
/// # Safety
///
/// As [`PilfflonkWrapProver::prove`].
unsafe fn wrap_witness(
    wrap: &WrapWitness,
    shape: &WitnessShape,
    recursivef_proof: *mut c_void,
) -> ProofmanResult<Witness> {
    timer_start_info!(CALCULATE_FINAL_WITNESS);
    // SAFETY: the caller's: `recursivef_proof` is the recursivef proof, a live nlohmann::json only
    // the witness calculator uses during the call.
    let witness = unsafe { wrap.witness_from_json(shape, recursivef_proof) }
        .map_err(|e| ProofmanError::InvalidProof(format!("The pilfflonk wrap's witness cannot be computed: {e}")))?;
    timer_stop_and_log_info!(CALCULATE_FINAL_WITNESS);
    Ok(witness)
}

/// `proof.json` and `publics.json` of a pilfflonk proof's bytes and of its publics' bytes
/// (pilfflonk/docs/formats.md#proof), named by `vkey`, as pilfflonk writes them. Refuses bytes that
/// are not those of a proof of `vkey`, or not as many publics as it has.
pub(crate) fn json_views(
    proof_bytes: &[u8],
    publics_bytes: &[u8],
    vkey: &Vkey,
) -> ProofmanResult<(ProofJson, Publics)> {
    let not_of_vkey = |e| ProofmanError::InvalidProof(format!("The bytes are not those of a proof of the vkey: {e}"));
    let names = ProofNames::of_vkey(vkey).map_err(not_of_vkey)?;
    let proof = Proof::from_bytes(proof_bytes, &names.shape()).map_err(not_of_vkey)?;
    let proof_json = proof.to_json(&names).map_err(not_of_vkey)?;
    let publics = publics_from_bytes(publics_bytes)?;
    if publics.0.len() as u64 != vkey.n_public {
        return Err(ProofmanError::InvalidProof(format!(
            "The proof has {} publics, and its vkey {}",
            publics.0.len(),
            vkey.n_public
        )));
    }
    Ok((proof_json, publics))
}

/// The pilfflonk vkey at `path`, `pilfflonk.vkey.json`.
pub(crate) fn read_vkey(path: &Path) -> ProofmanResult<Vkey> {
    Vkey::read(path).map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to read the pilfflonk vkey: {e}")))
}

/// Whether the proof of `proof_json` and `publics` verifies against the vkey at `vkey_path`, by
/// pilfflonk's JS verifier (`proofman_pilfflonk::js_verifier`), which reads them from `proof_path`
/// and `publics_path`, where this writes them. An error if the verifier cannot give a verdict.
pub(crate) fn verify_json(
    vkey_path: &Path,
    proof_json: &ProofJson,
    publics: &Publics,
    proof_path: &Path,
    publics_path: &Path,
) -> ProofmanResult<bool> {
    let write_failed =
        |e| ProofmanError::InvalidConfiguration(format!("Failed to write a JSON view of the proof: {e}"));
    proof_json.write(proof_path).map_err(write_failed)?;
    publics.write(publics_path).map_err(write_failed)?;
    js_verifier::verify(vkey_path, publics_path, proof_path)
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to run the pilfflonk verifier: {e}")))
}

/// The publics' bytes: each public 32 bytes big-endian, in order.
fn publics_to_bytes(publics: &Publics) -> Vec<u8> {
    publics.0.iter().flat_map(FrBytes::to_be_bytes).collect()
}

/// The publics of their bytes ([`publics_to_bytes`]). Refuses a length that is not a multiple of 32
/// and a public that is not below `r`.
fn publics_from_bytes(bytes: &[u8]) -> ProofmanResult<Publics> {
    let chunks = bytes.chunks_exact(PUBLIC_BYTES);
    if !chunks.remainder().is_empty() {
        return Err(ProofmanError::InvalidProof(format!(
            "The publics of a pilfflonk proof are {PUBLIC_BYTES} bytes each, and it has {} bytes of them",
            bytes.len()
        )));
    }
    let publics = chunks
        .map(|chunk| {
            let mut public = [0u8; PUBLIC_BYTES];
            public.copy_from_slice(chunk);
            FrBytes::from_be_bytes(public)
        })
        .collect::<Result<_, _>>()
        .map_err(|e| ProofmanError::InvalidProof(format!("A public of the pilfflonk proof is not in Fr: {e}")))?;
    Ok(Publics(publics))
}

/// A pilfflonk key that cannot be read or loaded.
fn invalid_key(proving_key: &Path, e: proofman_pilfflonk::PilfflonkError) -> ProofmanError {
    ProofmanError::InvalidSetup(format!("The pilfflonk key {} cannot be loaded: {e}", proving_key.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `r`, BN128's scalar field modulus, big-endian.
    fn r_be() -> [u8; 32] {
        let hex = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";
        let mut bytes = [0u8; 32];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        bytes
    }

    #[test]
    fn a_key_is_refused_when_the_device_has_less_free_than_it_needs_beside_its_arena() {
        let needed = DeviceBytes { arena: 1 << 20, beside: 3 << 20 };
        assert!(require_beside(&needed, needed.beside).is_ok());
        assert!(require_beside(&needed, u64::MAX).is_ok());
        let refused = require_beside(&needed, needed.beside - 1).unwrap_err().to_string();
        assert!(refused.contains("beside its arena of 1048576 bytes it needs 3145728 bytes"), "{refused}");
        assert!(refused.contains("and the device has 3145727 free"), "{refused}");
    }

    #[test]
    fn publics_are_32_bytes_big_endian_each() {
        let publics = Publics(vec![FrBytes::from_u64(1), FrBytes::from_u64(0x0102)]);
        let bytes = publics_to_bytes(&publics);
        assert_eq!(bytes.len(), 64);
        assert_eq!((bytes[31], bytes[62], bytes[63]), (1, 1, 2));
        assert!(bytes[..31].iter().chain(&bytes[32..62]).all(|&b| b == 0));
        assert_eq!(publics_from_bytes(&bytes).unwrap(), publics);
        assert_eq!(publics_from_bytes(&[]).unwrap(), Publics(vec![]));
    }

    #[test]
    fn publics_of_a_length_other_than_32_bytes_each_or_not_below_r_are_refused() {
        for length in [1, 31, 33, 63] {
            assert!(publics_from_bytes(&vec![0; length]).is_err(), "{length} bytes");
        }
        assert!(publics_from_bytes(&r_be()).is_err());
        let mut below_r = r_be();
        below_r[31] -= 1;
        assert!(publics_from_bytes(&below_r).is_ok());
    }
}
