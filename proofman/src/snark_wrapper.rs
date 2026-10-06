use proofman_common::{
    GlobalInfoAir, ProofmanError, ProofmanResult, ProofType, PublicsInfo, Setup, calculate_fixed_tree_snark,
    load_const_pols_recursivef, load_const_pols_tree, MemoryHandlerRecursive, VerboseMode, initialize_logger,
};
use proofman_util::{timer_start_info, timer_stop_and_log_info, timer_start_debug, timer_stop_and_log_debug};
use proofman_verifier::VadcopFinalProof;
use proofman_fields::PrimeField64;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::process::Command;
use colored::Colorize;
use std::io::Read;
use std::ffi::c_void;
use crate::check_const_tree;
use proofman_starks_lib_c::{
    init_final_snark_prover_c, free_final_snark_prover_c, get_snark_protocol_id_c, snark_proof_bytes_to_json_c,
    get_unified_buffer_gpu_for_recursivef_c, pre_allocate_final_snark_prover_c, free_device_buffers_recursivef_c,
    gen_device_buffers_recursivef_c, set_gpu_mode_c, get_num_gpus_c, init_gpu_setup_c, get_aux_trace_end_c,
};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::{
    verify_proof_bn128, generate_witness_final_snark, generate_recursivef_proof, generate_snark_proof, RecursivefProof,
};
use crate::pilfflonk_wrap::{self, PilfflonkWrapProver};
use proofman_pilfflonk::global_info::PROVING_KEY_DIR;
use proofman_pilfflonk::PilfflonkGlobalInfo;
use serde::{Deserialize, Serialize};

/// Sets GPU mode and verifies that a usable GPU is available when `gpu` is requested.
/// Returns an error if the library was built without CUDA support, or if GPU mode was
/// requested but no GPUs were found.
pub fn ensure_gpu_available(gpu: bool) -> ProofmanResult<()> {
    if !set_gpu_mode_c(gpu) {
        return Err(ProofmanError::InvalidConfiguration(
            "GPU mode requested but library was built without CUDA support".into(),
        ));
    }
    if gpu && get_num_gpus_c() == 0 {
        return Err(ProofmanError::InvalidConfiguration("No GPUs found".into()));
    }
    Ok(())
}

/// The protocol of the final SNARK: rapidsnark's PLONK or FFLONK, whose zkey names it, or pilfflonk
/// (pilfflonk/docs/README.md), whose key is a `provingKey/` ([`FinalSnarkKey`]).
pub enum SnarkProtocol {
    Fflonk,
    Plonk,
    Pilfflonk,
}

/// The protocol id of a pilfflonk proof in a [`SnarkProof`]. A zkey names the ids of PLONK (2) and
/// FFLONK (10), snarkjs's, and none for pilfflonk, so this one is proofman's own, far from those:
/// `0x7066`, the "pf" of pilfflonk's tags (the high half of its bytecode's version,
/// pilfflonk/docs/formats.md#bytecode).
pub const PILFFLONK_PROTOCOL_ID: u64 = 0x7066;

impl SnarkProtocol {
    pub fn protocol_id(&self) -> u64 {
        match self {
            SnarkProtocol::Fflonk => 10,
            SnarkProtocol::Plonk => 2,
            SnarkProtocol::Pilfflonk => PILFFLONK_PROTOCOL_ID,
        }
    }

    pub fn protocol_name(&self) -> &'static str {
        match self {
            SnarkProtocol::Plonk => "plonk",
            SnarkProtocol::Fflonk => "fflonk",
            SnarkProtocol::Pilfflonk => "pilfflonk",
        }
    }

    pub fn from_protocol_id(protocol_id: u64) -> ProofmanResult<Self> {
        match protocol_id {
            2 => Ok(SnarkProtocol::Plonk),
            10 => Ok(SnarkProtocol::Fflonk),
            PILFFLONK_PROTOCOL_ID => Ok(SnarkProtocol::Pilfflonk),
            _ => Err(ProofmanError::InvalidConfiguration(format!("Unsupported snark protocol id: {}", protocol_id))),
        }
    }

    /// The protocol of a loaded final SNARK prover: the one its zkey names, PLONK or FFLONK.
    fn from_snark_prover(snark_prover: *mut c_void) -> ProofmanResult<Self> {
        match Self::from_protocol_id(get_snark_protocol_id_c(snark_prover))? {
            SnarkProtocol::Pilfflonk => Err(ProofmanError::InvalidConfiguration(
                "A zkey names PLONK or FFLONK, and this one the id of pilfflonk, whose key is a provingKey/".into(),
            )),
            protocol => Ok(protocol),
        }
    }
}

/// The key of the final SNARK in a setup's `provingKeySnark/final/`, which setup-snark writes for
/// one protocol (`--final-snark`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinalSnarkKey {
    /// `final.zkey`: rapidsnark's PLONK or FFLONK key, which names its protocol.
    Zkey(PathBuf),
    /// `provingKey/`: a pilfflonk key, whose globalInfo says `"backend": "pilfflonk"`.
    Pilfflonk(PathBuf),
}

impl FinalSnarkKey {
    /// The key of the setup whose final SNARK files have the stem `setup_snark_path`
    /// (`provingKeySnark/final/final`): `final.zkey` or `provingKey/` beside them, exactly one. The
    /// zkey is read when its prover loads; the `provingKey/` must be a pilfflonk one now, by its
    /// globalInfo.
    pub fn find(setup_snark_path: &Path) -> ProofmanResult<Self> {
        let mut zkey = setup_snark_path.as_os_str().to_os_string();
        zkey.push(".zkey");
        let zkey = PathBuf::from(zkey);
        // setup-snark sets the final circuit up with `setup-pilfflonk -b provingKeySnark/final`.
        let proving_key = setup_snark_path.with_file_name(PROVING_KEY_DIR);
        match (zkey.exists(), proving_key.exists()) {
            (true, false) => Ok(Self::Zkey(zkey)),
            (false, true) => match PilfflonkGlobalInfo::from_proving_key(&proving_key) {
                Ok(_) => Ok(Self::Pilfflonk(proving_key)),
                Err(e) => Err(ProofmanError::InvalidSetup(format!(
                    "{} is not a pilfflonk key, the only final SNARK key that is a provingKey/: {e}",
                    proving_key.display()
                ))),
            },
            (true, true) => Err(ProofmanError::InvalidSetup(format!(
                "There are two final SNARK keys, {} (PLONK or FFLONK) and {} (pilfflonk): setup-snark writes the key \
                 of one protocol and removes the other's, so running it again with the --final-snark to prove with \
                 leaves only that one",
                zkey.display(),
                proving_key.display()
            ))),
            (false, false) => Err(ProofmanError::InvalidSetup(format!(
                "There is no final SNARK key: neither {} (PLONK or FFLONK) nor {} (pilfflonk)",
                zkey.display(),
                proving_key.display()
            ))),
        }
    }
}

/// [`SnarkWrapper`] of a poseidon key: the recursivef proves the vadcop_final proof, and then the final
/// SNARK of `provingKeySnark/final/` ([`FinalSnarkKey`]), rapidsnark's PLONK or FFLONK or pilfflonk,
/// proves the recursivef's verifier circuit.
///
/// **One proof at a time.** The wrapper is `Send` and `Sync`, and proofs of it from several threads
/// run one after another: each proof uses its buffers and provers alone.
///
/// **GPU memory with pilfflonk.** As for PLONK, the recursivef's device buffers are the wrapper's
/// (in `d_buffers`, proofman's unified buffer, if there is one), and pilfflonk's proofs keep their data
/// where `pre_allocate_final_snark_prover_c` carves the PLONK prover's (`pilfflonk_wrap::wrap_arena`):
/// in the unified buffer, which must hold a proof's arena (or the key is refused), or else in the
/// recursivef's prover buffer, grown to it; the recursivef is done with either when pilfflonk proves.
/// Its key's own device memory (the SRS's powers, the fixed columns' coefficients) is beside them,
/// from the start with `preload`.
pub(crate) struct PoseidonWrap<F: PrimeField64> {
    pub setup_snark_path: PathBuf,
    pub setup_recursivef: Setup<F>,
    pub vadcop_final_verkey: Vec<u64>,
    pub aux_trace: Arc<Vec<F>>,
    pub recursivef_const_pols: Arc<Vec<F>>,
    pub recursivef_const_tree: Arc<Vec<F>>,
    pub d_buffers: Option<*mut c_void>,
    pub reload_fixed_pols_gpu: Option<Arc<AtomicBool>>,
    /// rapidsnark's prover of `final.zkey`, with `preload`; never with pilfflonk.
    pub snark_prover: Option<*mut c_void>,
    /// The recursivef's device buffers (null on the CPU), out of which PLONK's GPU prover carves its
    /// own, and pilfflonk's its arena without `d_buffers`.
    pub d_buffers_recursivef: *mut c_void,
    pub proving_key_path: PathBuf,
    pub memory_handler_recursive_witness: Arc<MemoryHandlerRecursive<F>>,
    pub gpu: bool,
    pub final_snark_key: FinalSnarkKey,
    /// Whether a proof writes the unified buffer past its aux traces (`writes_past_aux_traces`).
    writes_past_aux_traces: bool,
    /// Held by each proof throughout ([`generate_final_snark_proof`](Self::generate_final_snark_proof)),
    /// so that the wrapper proves one at a time, and pilfflonk's prover, with `preload`. Two proofs at
    /// once would both write the recursivef's prover buffer `aux_trace` and its device buffers, and
    /// rapidsnark's prover. pilfflonk's prover, a C++ handle, is neither `Send` nor `Sync`: it is only
    /// used under this lock, which the wrapper's `Send` and `Sync` rely on.
    proving: Mutex<Option<PilfflonkWrapProver>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnarkProof {
    pub proof_bytes: Vec<u8>,
    pub public_bytes: Vec<u8>,
    pub public_snark_bytes: Vec<u8>,
    pub protocol_id: u64,
}

impl SnarkProof {
    pub fn new(proof_bytes: Vec<u8>, public_bytes: Vec<u8>, public_snark_bytes: Vec<u8>, protocol_id: u64) -> Self {
        Self { proof_bytes, public_bytes, public_snark_bytes, protocol_id }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = File::create(path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to create file for saving SNARK proof: {}: {}", path.display(), e),
            )
        })?;

        bincode::serde::encode_into_std_write(self, &mut file, bincode::config::standard())?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut file = File::open(path.as_ref()).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open file for loading SNARK proof: {}: {}", path.as_ref().display(), e),
            )
        })?;
        let proof: SnarkProof = bincode::serde::decode_from_std_read(&mut file, bincode::config::standard())?;
        Ok(proof)
    }

    /// The JSON views of a PLONK or FFLONK proof and of its publics, as snarkjs reads them. A
    /// pilfflonk proof's bytes do not say where their parts end, its vkey does, and it is refused:
    /// [`convert_to_json_with_vkey`](Self::convert_to_json_with_vkey).
    pub fn convert_to_json(
        &self,
    ) -> Result<(serde_json::Value, serde_json::Value), Box<dyn std::error::Error + Send + Sync>> {
        let protocol = SnarkProtocol::from_protocol_id(self.protocol_id)?;
        if let SnarkProtocol::Pilfflonk = protocol {
            return Err("A pilfflonk proof's bytes do not say where their parts end, its vkey does: \
                 convert it with convert_to_json_with_vkey"
                .into());
        }
        let (proof_json, publics_json) =
            snark_proof_bytes_to_json_c(&self.proof_bytes, &self.public_snark_bytes, protocol.protocol_id() as i32);

        let proof_json_value: serde_json::Value = serde_json::from_str(&proof_json)?;
        let publics_json_value: serde_json::Value = serde_json::from_str(&publics_json)?;

        Ok((proof_json_value, publics_json_value))
    }

    /// The JSON views of the proof and of its publics, as its verifier reads them, for every
    /// protocol: for PLONK and FFLONK, [`convert_to_json`](Self::convert_to_json)'s, which do not
    /// read `vkey`; for pilfflonk, `proof.json` and `publics.json` (pilfflonk/docs/formats.md#proof),
    /// the proof's bytes named by the vkey at `vkey`, `pilfflonk.vkey.json`.
    pub fn convert_to_json_with_vkey(
        &self,
        vkey: &Path,
    ) -> Result<(serde_json::Value, serde_json::Value), Box<dyn std::error::Error + Send + Sync>> {
        match SnarkProtocol::from_protocol_id(self.protocol_id)? {
            SnarkProtocol::Pilfflonk => {
                let vkey = pilfflonk_wrap::read_vkey(vkey)?;
                let (proof, publics) = pilfflonk_wrap::json_views(&self.proof_bytes, &self.public_snark_bytes, &vkey)?;
                Ok((serde_json::to_value(proof)?, serde_json::to_value(publics)?))
            }
            SnarkProtocol::Plonk | SnarkProtocol::Fflonk => self.convert_to_json(),
        }
    }

    pub fn get_public_bytes(&self) -> &[u8] {
        &self.public_bytes
    }
}

impl<F: PrimeField64> Drop for PoseidonWrap<F> {
    fn drop(&mut self) {
        if let Some(snark_prover) = self.snark_prover {
            free_final_snark_prover_c(snark_prover);
        }
        // pilfflonk's key first, whose arena may be the recursivef's prover buffer.
        drop(self.proving.get_mut().unwrap_or_else(|poisoned| poisoned.into_inner()).take());
        // Null on the CPU, whose backend has no device buffers to free.
        if !self.d_buffers_recursivef.is_null() {
            free_device_buffers_recursivef_c(self.d_buffers_recursivef);
        }
    }
}

/// The recursivef's device buffers (`gen_device_buffers_recursivef_c`), freed when dropped unless
/// handed over ([`into_raw`](Self::into_raw)). The CPU backend has none: the pointer is null, and
/// nothing is freed.
struct RecursivefDeviceBuffers(*mut c_void);

impl RecursivefDeviceBuffers {
    /// The device buffers of `setup`'s proofs, in proofman's unified buffer `d_buffers` if there is
    /// one, for the recursivef verkey `verkey`.
    fn new<F: PrimeField64>(setup: &Setup<F>, d_buffers: Option<*mut c_void>, verkey: &str) -> Self {
        let p_setup: *mut c_void = (&setup.p_setup).into();
        let d_buffers_vadcop = d_buffers.unwrap_or(std::ptr::null_mut());
        Self(gen_device_buffers_recursivef_c(
            p_setup as *mut u8,
            setup.prover_buffer_size,
            d_buffers_vadcop as *mut u8,
            verkey,
        ) as *mut c_void)
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0
    }

    /// The buffers, which the caller frees from now on.
    fn into_raw(self) -> *mut c_void {
        let buffers = self.0;
        std::mem::forget(self);
        buffers
    }
}

impl Drop for RecursivefDeviceBuffers {
    fn drop(&mut self) {
        if !self.0.is_null() {
            free_device_buffers_recursivef_c(self.0);
        }
    }
}

/// rapidsnark's prover of the zkey `zkey`, whose device buffers are carved out of the recursivef's
/// `d_buffers_recursivef`.
fn init_rapidsnark_prover(zkey: &Path, d_buffers_recursivef: *mut c_void) -> ProofmanResult<*mut c_void> {
    let zkey_filename = zkey.display().to_string();
    let snark_prover = init_final_snark_prover_c(&zkey_filename, d_buffers_recursivef);
    if snark_prover.is_null() {
        return Err(std::io::Error::other(format!(
            "Failed to initialize final snark prover from zkey file '{}'",
            zkey_filename
        ))
        .into());
    }
    Ok(snark_prover)
}

impl<F: PrimeField64> PoseidonWrap<F> {
    pub fn new_with_preallocated_buffers(
        proving_key_path: &Path,
        verbose_mode: VerboseMode,
        _aux_trace: Option<Arc<Vec<F>>>,
        d_buffers: Option<*mut c_void>,
        reload_fixed_pols_gpu: Option<Arc<AtomicBool>>,
        preload: bool,
        gpu: bool,
    ) -> ProofmanResult<Self> {
        initialize_logger(verbose_mode, None);

        ensure_gpu_available(gpu)?;

        let setup_recursivef_path =
            PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "recursivef", "recursivef"));
        let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));
        let final_snark_key = FinalSnarkKey::find(&setup_snark_path)?;

        let vadcop_final_verkey_path =
            PathBuf::from(format!("{}/vadcop_final.verkey.json", proving_key_path.display()));

        let mut file = File::open(&vadcop_final_verkey_path).expect("Unable to open file");
        let mut json_str = String::new();
        file.read_to_string(&mut json_str).expect("Unable to read file");
        let vadcop_final_verkey: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse JSON");

        timer_start_info!(LOADING_RECURSIVE_F_SETUP);

        let setup_recursivef = Setup::new(
            &setup_recursivef_path,
            0,
            0,
            &GlobalInfoAir::new("RecursiveF".to_string()),
            &ProofType::RecursiveF,
            false,
            gpu,
            None,
            &std::collections::HashMap::new(),
            false,
        )?;

        check_const_tree(&setup_recursivef, &d_buffers)?;

        let mut recursivef_const_pols_buf: Vec<F> = vec![F::ZERO; setup_recursivef.const_pols_size];
        load_const_pols_recursivef(&setup_recursivef, &mut recursivef_const_pols_buf);
        let recursivef_const_pols: Arc<Vec<F>> = Arc::new(recursivef_const_pols_buf);
        let mut recursivef_const_tree_buf: Vec<F> = vec![F::ZERO; setup_recursivef.const_tree_size];
        load_const_pols_tree(&setup_recursivef, &mut recursivef_const_tree_buf);
        let recursivef_const_tree: Arc<Vec<F>> = Arc::new(recursivef_const_tree_buf);

        timer_stop_and_log_info!(LOADING_RECURSIVE_F_SETUP);

        let aux_trace = if let Some(buffer) = _aux_trace {
            buffer
        } else if gpu {
            Arc::new(Vec::new())
        } else {
            Arc::new(vec![F::ZERO; setup_recursivef.prover_buffer_size as usize])
        };

        let verkey_path = setup_recursivef.verkey_file.clone();
        let mut contents = String::new();
        let mut file = File::open(verkey_path).unwrap();
        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));

        let verkey_str: String = serde_json::from_str(&contents)
            .map_err(|err| ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err)))?;

        // The recursivef's, freed if the final SNARK's prover cannot be loaded.
        let recursivef_buffers = RecursivefDeviceBuffers::new(&setup_recursivef, d_buffers, &verkey_str);
        let mut pilfflonk_prover = None;
        let mut snark_prover = None;
        match &final_snark_key {
            FinalSnarkKey::Zkey(zkey) => {
                timer_start_info!(INITIALIZING_FINAL_SNARK_PROVER);
                if preload {
                    snark_prover = Some(init_rapidsnark_prover(zkey, recursivef_buffers.as_ptr())?);
                }
                timer_stop_and_log_info!(INITIALIZING_FINAL_SNARK_PROVER);
            }
            // pilfflonk's arena is in the unified buffer or in the recursivef's buffers.
            FinalSnarkKey::Pilfflonk(pilfflonk_key) => {
                if preload {
                    timer_start_info!(INITIALIZING_FINAL_SNARK_PROVER);
                    let arena = pilfflonk_wrap::wrap_arena(gpu, d_buffers, recursivef_buffers.as_ptr(), pilfflonk_key)?;
                    // SAFETY: the arena is the unified buffer or the recursivef's prover buffer, which
                    // outlive the prover (the wrapper's `Drop` drops it first), and which a proof uses
                    // only for the recursivef until pilfflonk proves.
                    pilfflonk_prover =
                        Some(unsafe { PilfflonkWrapProver::load(&setup_snark_path, pilfflonk_key, arena) }?);
                    timer_stop_and_log_info!(INITIALIZING_FINAL_SNARK_PROVER);
                }
            }
        }
        let d_buffers_recursivef = recursivef_buffers.into_raw();

        let trace_size = setup_recursivef.stark_info.map_sections_n["cm1"]
            * (1 << setup_recursivef.stark_info.stark_struct.n_bits)
            + setup_recursivef.stark_info.n_publics;

        let memory_handler_recursive_witness = Arc::new(MemoryHandlerRecursive::new(1, trace_size as usize));
        let writes_past_aux_traces =
            d_buffers.is_some_and(|d| Self::writes_past_aux_traces(&setup_recursivef, &final_snark_key, gpu, d));

        Ok(Self {
            aux_trace,
            recursivef_const_pols,
            recursivef_const_tree,
            setup_recursivef,
            setup_snark_path,
            snark_prover,
            proving_key_path: proving_key_path.to_path_buf(),
            vadcop_final_verkey,
            d_buffers,
            d_buffers_recursivef,
            memory_handler_recursive_witness,
            reload_fixed_pols_gpu,
            gpu,
            final_snark_key,
            writes_past_aux_traces,
            proving: Mutex::new(pilfflonk_prover),
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_final_snark_proof(
        &self,
        vadcop_proof: &VadcopFinalProof,
        verkey_override: Option<&[u64]>,
    ) -> ProofmanResult<SnarkProof> {
        timer_start_info!(GENERATING_WRAPPER_SNARK_PROOF);

        if !proofman_common::hash_family::supports_snark(&vadcop_proof.hash) {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "{} proofs have no SNARK stage: the BN128 wrap is only built for the poseidon \
                 families",
                vadcop_proof.hash
            )));
        }
        if vadcop_proof.compressed {
            return Err(ProofmanError::InvalidConfiguration(
                "Compressed vadcop proofs are not supported for snark proof generation".to_string(),
            ));
        }
        let proof = vadcop_proof.proof_with_publics();

        // The RecursiveF verifier checks the proof against this verkey (stamped
        // into the leading 4 slots of the publics). Default is the vadcop_final
        // verkey; a recurser/aggregated proof committed under its own verkey, so
        // the caller passes it here — using vadcop_final's would diverge the
        // wrapper transcript (VerifyPoW aborts).
        let verkey = verkey_override.unwrap_or(&self.vadcop_final_verkey);

        // One proof at a time (`proving`), until this one is out. A proof that panicked poisons the
        // lock, and the next one takes it all the same: each proof writes its buffers afresh.
        let pilfflonk_prover = self.proving.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let _borrow = pilfflonk_wrap::FirstGpuBorrow::new(self.d_buffers.filter(|_| self.gpu));
        // The recursivef's carve and rapidsnark's prover may write the unified buffer over the STARK's
        // const pols, and one that fails may have written it too: the flag is set as this proof ends,
        // whatever its outcome, before the lock is let go. pilfflonk's arena stays in the aux traces.
        let reload = self.d_buffers.filter(|_| self.writes_past_aux_traces);
        let _reload = ReloadFixedPolsOnDrop(reload.and(self.reload_fixed_pols_gpu.as_deref()));
        let recursivef_proof = generate_recursivef_proof(
            &self.setup_recursivef,
            &self.memory_handler_recursive_witness,
            &proof,
            &self.aux_trace,
            &self.recursivef_const_pols,
            &self.recursivef_const_tree,
            verkey,
            self.setup_recursivef.prover_buffer_size as usize * std::mem::size_of::<F>(),
            self.d_buffers_recursivef,
        )?;
        let snark_proof = match &self.final_snark_key {
            FinalSnarkKey::Zkey(zkey) => self.generate_rapidsnark_proof(zkey, &proof, &recursivef_proof)?,
            FinalSnarkKey::Pilfflonk(pilfflonk_key) => {
                self.generate_pilfflonk_proof(pilfflonk_key, pilfflonk_prover.as_ref(), &proof, &recursivef_proof)?
            }
        };

        timer_stop_and_log_info!(GENERATING_WRAPPER_SNARK_PROOF);

        Ok(snark_proof)
    }

    /// Whether a proof writes proofman's unified buffer `d_buffers` past its aux traces: rapidsnark's
    /// prover, whose carve this does not know, a recursivef whose const tree and buffer pass them, or
    /// a pilfflonk arena that does.
    fn writes_past_aux_traces(
        setup_recursivef: &Setup<F>,
        final_snark_key: &FinalSnarkKey,
        gpu: bool,
        d_buffers: *mut c_void,
    ) -> bool {
        let recursivef = (setup_recursivef.const_tree_size as u64 + setup_recursivef.prover_buffer_size)
            * std::mem::size_of::<F>() as u64;
        recursivef > get_aux_trace_end_c(d_buffers)
            || match final_snark_key {
                FinalSnarkKey::Zkey(_) => true,
                FinalSnarkKey::Pilfflonk(key) => gpu && pilfflonk_wrap::arena_passes_aux_traces(d_buffers, key),
            }
    }

    /// The PLONK or FFLONK proof of the vadcop proof `proof` (with its publics), of its recursivef
    /// proof `recursivef_proof`: rapidsnark's prover of the zkey `zkey`. Under the lock of `proving`.
    fn generate_rapidsnark_proof(
        &self,
        zkey: &Path,
        proof: &[u64],
        recursivef_proof: &RecursivefProof,
    ) -> ProofmanResult<SnarkProof> {
        timer_start_debug!(GENERATING_SNARK_PROOF);

        let snark_prover = match self.snark_prover {
            Some(prover) => prover,
            None => init_rapidsnark_prover(zkey, self.d_buffers_recursivef)?,
        };

        let protocol = match SnarkProtocol::from_snark_prover(snark_prover) {
            Ok(protocol) => protocol,
            Err(e) => {
                if self.snark_prover.is_none() {
                    free_final_snark_prover_c(snark_prover);
                }
                return Err(e);
            }
        };

        //  Spawn GPU pre-allocation on a separate thread so it overlaps with CPU witness computation
        let prealloc_handle = {
            let snark_prover = snark_prover as usize;
            let unified_buffer_gpu = if let Some(d_buffers) = self.d_buffers {
                get_unified_buffer_gpu_for_recursivef_c(d_buffers, self.d_buffers_recursivef)
            } else {
                std::ptr::null_mut()
            };
            let buffer = unified_buffer_gpu as usize;
            let d_buffers_recursivef = self.d_buffers_recursivef as usize;
            std::thread::spawn(move || {
                pre_allocate_final_snark_prover_c(
                    snark_prover as *mut std::ffi::c_void,
                    buffer as *mut std::ffi::c_void,
                    d_buffers_recursivef as *mut std::ffi::c_void,
                );
            })
        };

        let (snark_proof_bytes, snark_publics_bytes) = generate_snark_proof(
            snark_prover,
            &self.setup_snark_path,
            recursivef_proof.as_ptr(),
            prealloc_handle,
            self.d_buffers_recursivef,
        )?;

        let public_bytes = self.public_bytes_solidity(proof)?;
        let snark_proof = SnarkProof::new(snark_proof_bytes, public_bytes, snark_publics_bytes, protocol.protocol_id());

        timer_stop_and_log_debug!(GENERATING_SNARK_PROOF);

        if self.snark_prover.is_none() {
            free_final_snark_prover_c(snark_prover);
        }

        Ok(snark_proof)
    }

    /// The pilfflonk proof of the vadcop proof `proof` (with its publics), of its recursivef proof
    /// `recursivef_proof`, with the key at `pilfflonk_key`, whose prover is `preloaded` with
    /// `preload` (the one `proving` holds). Under the lock of `proving`. The recursivef proved in the
    /// wrapper's device buffers; pilfflonk, its key loaded now without `preload`, proves the wrap's
    /// witness, which it computes from the recursivef proof, in its arena, which the recursivef is
    /// done with.
    fn generate_pilfflonk_proof(
        &self,
        pilfflonk_key: &Path,
        preloaded: Option<&PilfflonkWrapProver>,
        proof: &[u64],
        recursivef_proof: &RecursivefProof,
    ) -> ProofmanResult<SnarkProof> {
        timer_start_debug!(GENERATING_SNARK_PROOF);

        let loaded;
        let prover = match preloaded {
            Some(prover) => prover,
            None => {
                let arena =
                    pilfflonk_wrap::wrap_arena(self.gpu, self.d_buffers, self.d_buffers_recursivef, pilfflonk_key)?;
                // SAFETY: as for the preloaded prover (`new_with_preallocated_buffers`): the arena
                // outlives this prover, which this proof drops, and the recursivef is done with it.
                loaded = unsafe { PilfflonkWrapProver::load(&self.setup_snark_path, pilfflonk_key, arena) }?;
                &loaded
            }
        };
        // SAFETY: `recursivef_proof` is the recursivef proof, which nothing else uses.
        let (snark_proof_bytes, snark_publics_bytes) = unsafe { prover.prove(recursivef_proof.as_ptr()) }?;

        let public_bytes = self.public_bytes_solidity(proof)?;
        let snark_proof = SnarkProof::new(
            snark_proof_bytes,
            public_bytes,
            snark_publics_bytes,
            SnarkProtocol::Pilfflonk.protocol_id(),
        );

        timer_stop_and_log_debug!(GENERATING_SNARK_PROOF);

        Ok(snark_proof)
    }

    /// The publics of the vadcop proof `proof` as the Solidity verifier takes them
    /// ([`get_public_bytes_solidity`]).
    fn public_bytes_solidity(&self, proof: &[u64]) -> ProofmanResult<Vec<u8>> {
        let publics_info = PublicsInfo::from_folder(&self.proving_key_path)?;
        get_public_bytes_solidity(&publics_info, &proof[1..1 + proof[0] as usize])
    }
}

/// The recursivef's starkinfo setup-snark puts beside the final circuit of a blake3 key, from which
/// the zkin of a recursivef proof is laid out.
pub const BLAKE3_RECURSIVEF_STARKINFO: &str = "recursivef.starkinfo.json";

/// [`SnarkWrapper`] of a blake3 key: its recursivef, a Goldilocks STARK of the proving key, is proved
/// before it (`ProofMan::generate_final_snark_proof`), and this proves the final SNARK of the
/// recursivef's proofs, pilfflonk's.
///
/// **GPU memory.** With proofman's buffers (`d_buffers`), all of the key's device memory is in its
/// unified buffer, from where the aux traces start to where they end
/// (`pilfflonk_wrap::unified_buffer`): nothing proofman keeps across proofs is there, so nothing is
/// reloaded. Preloaded, the key lives across proofs while proofman uses that buffer in between: it
/// keeps a pinned host copy of what it holds there (~4 GB, its SRS and fixed coefficients), which each
/// proof writes back while the host computes the circuit's witness. Not preloaded, each proof loads
/// it. Without buffers, the key's memory is its own.
pub(crate) struct Blake3Wrap {
    setup_snark_path: PathBuf,
    pilfflonk_key: PathBuf,
    starkinfo: serde_json::Value,
    publics_info: PublicsInfo,
    d_buffers: Option<*mut c_void>,
    gpu: bool,
    /// The prover, preloaded; else loaded by each proof.
    prover: Option<PilfflonkWrapProver>,
    proving: Mutex<()>,
}

impl Blake3Wrap {
    fn new(proving_key_snark: &Path, d_buffers: Option<*mut c_void>, preload: bool, gpu: bool) -> ProofmanResult<Self> {
        ensure_gpu_available(gpu)?;
        let final_dir = proving_key_snark.join("final");
        let setup_snark_path = final_dir.join("final");
        let FinalSnarkKey::Pilfflonk(pilfflonk_key) = FinalSnarkKey::find(&setup_snark_path)? else {
            return Err(ProofmanError::InvalidSetup(
                "the final SNARK of a blake3 key is pilfflonk's, and this setup has a zkey".to_string(),
            ));
        };
        let starkinfo_path = final_dir.join(BLAKE3_RECURSIVEF_STARKINFO);
        let starkinfo: serde_json::Value = std::fs::read_to_string(&starkinfo_path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .ok_or_else(|| {
                ProofmanError::InvalidSetup(format!(
                    "Failed to read the recursivef's starkinfo {}",
                    starkinfo_path.display()
                ))
            })?;
        let mut wrap = Self {
            setup_snark_path,
            pilfflonk_key,
            starkinfo,
            publics_info: PublicsInfo::from_folder(proving_key_snark)?,
            d_buffers: d_buffers.filter(|_| gpu),
            gpu,
            prover: None,
            proving: Mutex::new(()),
        };
        if preload {
            let _borrow = pilfflonk_wrap::FirstGpuBorrow::new(wrap.d_buffers);
            wrap.prover = Some(wrap.load(true)?);
        }
        Ok(wrap)
    }

    /// The key's prover; `restorable` if it lives across proofs in proofman's buffer.
    fn load(&self, restorable: bool) -> ProofmanResult<PilfflonkWrapProver> {
        let device = if self.gpu { proofman_pilfflonk::Device::Gpu } else { proofman_pilfflonk::Device::Cpu };
        let buffer = self.d_buffers.map(pilfflonk_wrap::unified_buffer).transpose()?;
        timer_start_info!(INITIALIZING_FINAL_SNARK_PROVER);
        // SAFETY: SnarkWrapper::new_with_preallocated_buffers's contract: proofman's buffer is idle
        // while the key loads and while each proof runs.
        let prover = unsafe {
            PilfflonkWrapProver::load_on(&self.setup_snark_path, &self.pilfflonk_key, device, buffer, restorable)
        }?;
        timer_stop_and_log_info!(INITIALIZING_FINAL_SNARK_PROVER);
        Ok(prover)
    }

    /// The SNARK proof of `recursivef_proof`, whose publics are `[rootC(4) | the vadcop_final's]`.
    fn generate_snark_proof(&self, recursivef_proof: &VadcopFinalProof) -> ProofmanResult<SnarkProof> {
        timer_start_info!(GENERATING_WRAPPER_SNARK_PROOF);
        let _proving = self.proving.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let n_publics = self.starkinfo["nPublics"].as_u64().unwrap_or(0) as usize;
        if recursivef_proof.public_values.len() != n_publics {
            return Err(ProofmanError::InvalidProof(format!(
                "a blake3 key's SNARK wraps its recursivef proof, of {n_publics} publics, and this proof has {}: \
                 a vadcop_final proof goes through ProofMan::generate_final_snark_proof, which proves its recursivef",
                recursivef_proof.public_values.len()
            )));
        }
        let zkin = pilfflonk_wrap_witness::gl_proof_zkin(
            &self.starkinfo,
            &recursivef_proof.proof,
            &recursivef_proof.public_values,
        )
        .map_err(|e| ProofmanError::InvalidProof(e.to_string()))?;
        let _borrow = pilfflonk_wrap::FirstGpuBorrow::new(self.d_buffers);
        let loaded;
        let prover = match &self.prover {
            Some(prover) => prover,
            None => {
                loaded = self.load(false)?;
                &loaded
            }
        };
        let (snark_proof_bytes, snark_publics_bytes) = prover.prove_zkin(&zkin)?;
        let vadcop_publics = recursivef_proof.public_values.get(4..).unwrap_or_default();
        let public_bytes = get_public_bytes_solidity(&self.publics_info, vadcop_publics)?;
        timer_stop_and_log_info!(GENERATING_WRAPPER_SNARK_PROOF);
        Ok(SnarkProof::new(
            snark_proof_bytes,
            public_bytes,
            snark_publics_bytes,
            SnarkProtocol::Pilfflonk.protocol_id(),
        ))
    }
}

// SAFETY: as PoseidonWrap's: the key's C++ handles and proofman's device buffers are not tied to the
// thread that made them, and `proving` keeps the proofs one at a time.
unsafe impl Send for Blake3Wrap {}
unsafe impl Sync for Blake3Wrap {}

/// The final SNARK's wrapper of the key in `proving_key_path`, whose proofs are
/// `ProofMan::generate_final_snark_proof`'s, of either hash: a poseidon key's ([`PoseidonWrap`])
/// proves the recursivef and then the final SNARK of the vadcop_final proof, a blake3 key's
/// ([`Blake3Wrap`]) the final SNARK of the recursivef proof proofman proves before it. One proof at
/// a time.
pub struct SnarkWrapper<F: PrimeField64> {
    inner: Wrap<F>,
}

// One per process; boxing buys nothing.
#[allow(clippy::large_enum_variant)]
enum Wrap<F: PrimeField64> {
    Poseidon(PoseidonWrap<F>),
    Blake3(Blake3Wrap),
}

impl<F: PrimeField64> SnarkWrapper<F> {
    pub fn new(proving_key_path: &Path, verbose_mode: VerboseMode, preload: bool, gpu: bool) -> ProofmanResult<Self> {
        Self::new_with_preallocated_buffers(proving_key_path, verbose_mode, None, None, None, preload, gpu)
    }

    /// With proofman's buffers (`d_buffers`, `aux_trace`, `reload_fixed_pols_gpu`, of
    /// `ProofMan::get_preallocated_buffers`), which must outlive the wrapper, idle while it loads and
    /// while each of its proofs runs; and with `preload`, the final SNARK's prover loaded now.
    pub fn new_with_preallocated_buffers(
        proving_key_path: &Path,
        verbose_mode: VerboseMode,
        aux_trace: Option<Arc<Vec<F>>>,
        d_buffers: Option<*mut c_void>,
        reload_fixed_pols_gpu: Option<Arc<AtomicBool>>,
        preload: bool,
        gpu: bool,
    ) -> ProofmanResult<Self> {
        let inner = if Self::is_blake3_key(proving_key_path) {
            initialize_logger(verbose_mode, None);
            Wrap::Blake3(Blake3Wrap::new(proving_key_path, d_buffers, preload, gpu)?)
        } else {
            Wrap::Poseidon(PoseidonWrap::new_with_preallocated_buffers(
                proving_key_path,
                verbose_mode,
                aux_trace,
                d_buffers,
                reload_fixed_pols_gpu,
                preload,
                gpu,
            )?)
        };
        Ok(Self { inner })
    }

    /// Whether `proving_key_snark` is a blake3 key's: setup-snark puts its recursivef's starkinfo in it.
    pub fn is_blake3_key(proving_key_snark: &Path) -> bool {
        proving_key_snark.join("final").join(BLAKE3_RECURSIVEF_STARKINFO).is_file()
    }

    /// The final SNARK proof of `proof`: of a poseidon key the vadcop_final proof (checked against
    /// `verkey_override` if given, the vadcop_final's verkey otherwise), of a blake3 key the recursivef
    /// proof (`ProofMan::generate_recursivef_proof`), which takes no verkey.
    pub(crate) fn generate_final_snark_proof(
        &self,
        proof: &VadcopFinalProof,
        verkey_override: Option<&[u64]>,
    ) -> ProofmanResult<SnarkProof> {
        match &self.inner {
            Wrap::Poseidon(wrap) => wrap.generate_final_snark_proof(proof, verkey_override),
            Wrap::Blake3(_) if verkey_override.is_some() => Err(ProofmanError::InvalidConfiguration(
                "a blake3 key's SNARK wraps its recursivef proof, whose verkey the recursivef fixed".to_string(),
            )),
            Wrap::Blake3(wrap) => wrap.generate_snark_proof(proof),
        }
    }
}

pub fn get_public_bytes_solidity(publics_info: &PublicsInfo, vadcop_public_inputs: &[u64]) -> ProofmanResult<Vec<u8>> {
    let vadcop_public_inputs = if vadcop_public_inputs.len() == publics_info.n_publics + 1 {
        &vadcop_public_inputs[1..]
    } else {
        vadcop_public_inputs
    };
    if vadcop_public_inputs.len() != publics_info.n_publics {
        return Err(ProofmanError::InvalidConfiguration(format!(
            "Number of vadcop public inputs ({}) does not match expected number of publics ({})",
            vadcop_public_inputs.len(),
            publics_info.n_publics
        )));
    }

    let mut public_bytes = vec![];
    let mut index = 0;
    for public_def in &publics_info.definitions {
        let n_words = public_def.n_values;
        if !public_def.verification_key {
            let n_chunks_per_word = public_def.chunks[0];
            let n_bits_per_chunk = public_def.chunks[1];
            let n_bytes_per_chunk = n_bits_per_chunk / 8;
            for _ in 0..n_words {
                for i in 0..n_chunks_per_word {
                    let value = vadcop_public_inputs[index + n_chunks_per_word - i - 1];
                    let be_bytes = value.to_be_bytes();
                    public_bytes.extend_from_slice(&be_bytes[8 - n_bytes_per_chunk..]);
                }
                index += n_chunks_per_word;
            }
        } else {
            index += n_words;
        }
    }
    Ok(public_bytes)
}

pub fn check_setup_snark<F: PrimeField64>(
    proving_key_snark_path: &Path,
    verbose_mode: VerboseMode,
    gpu: bool,
) -> ProofmanResult<()> {
    initialize_logger(verbose_mode, None);

    ensure_gpu_available(gpu)?;

    let setup_recursivef_path =
        PathBuf::from(format!("{}/{}/{}", proving_key_snark_path.display(), "recursivef", "recursivef"));

    let setup_recursivef: Setup<F> = Setup::<F>::new(
        &setup_recursivef_path,
        0,
        0,
        &GlobalInfoAir::new("RecursiveF".to_string()),
        &ProofType::RecursiveF,
        false,
        gpu,
        None,
        &std::collections::HashMap::new(),
        false,
    )?;

    calculate_fixed_tree_snark(&setup_recursivef);

    Ok(())
}

pub fn generate_and_verify_recursivef<F: PrimeField64>(
    proving_key_path: &Path,
    vadcop_proof: &VadcopFinalProof,
    verbose_mode: VerboseMode,
    gpu: bool,
) -> ProofmanResult<bool> {
    initialize_logger(verbose_mode, None);

    ensure_gpu_available(gpu)?;

    if !proofman_common::hash_family::supports_snark(&vadcop_proof.hash) {
        return Err(ProofmanError::InvalidConfiguration(format!(
            "{} proofs have no SNARK stage: the BN128 wrap is only built for the poseidon families",
            vadcop_proof.hash
        )));
    }
    if vadcop_proof.compressed {
        return Err(ProofmanError::InvalidConfiguration(
            "Compressed vadcop proofs are not supported for snark proof generation".to_string(),
        ));
    }
    let proof = vadcop_proof.proof_with_publics();

    timer_start_info!(LOADING_RECURSIVE_F_SETUP);

    let setup_recursivef_path =
        PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "recursivef", "recursivef"));

    let setup_recursivef = Setup::<F>::new(
        &setup_recursivef_path,
        0,
        0,
        &GlobalInfoAir::new("RecursiveF".to_string()),
        &ProofType::RecursiveF,
        false,
        gpu,
        None,
        &std::collections::HashMap::new(),
        false,
    )?;

    ensure_gpu_available(gpu)?;
    if gpu {
        init_gpu_setup_c(setup_recursivef.stark_info.stark_struct.merkle_tree_arity);
    }

    check_const_tree(&setup_recursivef, &None)?;

    let mut recursivef_const_pols_buf: Vec<F> = vec![F::ZERO; setup_recursivef.const_pols_size];
    load_const_pols_recursivef(&setup_recursivef, &mut recursivef_const_pols_buf);
    let recursivef_const_pols: Arc<Vec<F>> = Arc::new(recursivef_const_pols_buf);
    let mut recursivef_const_tree_buf: Vec<F> = vec![F::ZERO; setup_recursivef.const_tree_size];
    load_const_pols_tree(&setup_recursivef, &mut recursivef_const_tree_buf);
    let recursivef_const_tree: Arc<Vec<F>> = Arc::new(recursivef_const_tree_buf);

    let aux_trace =
        if gpu { Arc::new(Vec::new()) } else { Arc::new(vec![F::ZERO; setup_recursivef.prover_buffer_size as usize]) };

    timer_stop_and_log_info!(LOADING_RECURSIVE_F_SETUP);

    let vadcop_final_verkey_path = PathBuf::from(format!("{}/vadcop_final.verkey.json", proving_key_path.display()));

    let mut file = File::open(&vadcop_final_verkey_path).expect("Unable to open file");
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).expect("Unable to read file");
    let vadcop_final_verkey: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse JSON");

    let verkey_path = setup_recursivef.verkey_file.clone();
    let mut contents = String::new();
    let mut file = File::open(verkey_path).unwrap();
    let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));

    let verkey_str: String = serde_json::from_str(&contents)
        .map_err(|err| ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err)))?;

    // Freed when this returns, whatever its outcome.
    let recursivef_buffers = RecursivefDeviceBuffers::new(&setup_recursivef, None, &verkey_str);

    let trace_size = setup_recursivef.stark_info.map_sections_n["cm1"]
        * (1 << setup_recursivef.stark_info.stark_struct.n_bits)
        + setup_recursivef.stark_info.n_publics;

    let memory_handler_recursive_witness = Arc::new(MemoryHandlerRecursive::new(1, trace_size as usize));

    timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
    let recursivef_proof = generate_recursivef_proof(
        &setup_recursivef,
        &memory_handler_recursive_witness,
        &proof,
        &aux_trace,
        &recursivef_const_pols,
        &recursivef_const_tree,
        &vadcop_final_verkey,
        setup_recursivef.prover_buffer_size as usize * std::mem::size_of::<F>(),
        recursivef_buffers.as_ptr(),
    )?;
    timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

    timer_start_info!(VERIFY_RECURSIVE_F_PROOF);
    let mut publics: Vec<F> = vadcop_final_verkey[0..4].iter().map(|&x| F::from_u64(x)).collect();
    publics.extend(proof[1..1 + proof[0] as usize].iter().map(|&x| F::from_u64(x)));

    let is_valid = verify_proof_bn128(recursivef_proof.as_ptr(), &setup_recursivef, Some(publics));
    timer_stop_and_log_info!(VERIFY_RECURSIVE_F_PROOF);

    let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));
    if setup_snark_path.parent().is_some_and(|p| p.exists()) {
        match FinalSnarkKey::find(&setup_snark_path)? {
            FinalSnarkKey::Zkey(_) => {
                generate_witness_final_snark(recursivef_proof.as_ptr(), &setup_snark_path)?;
            }
            // SAFETY: `recursivef_proof` is the recursivef proof, which nothing else uses.
            FinalSnarkKey::Pilfflonk(pilfflonk_key) => unsafe {
                pilfflonk_wrap::check_wrap_witness(&setup_snark_path, &pilfflonk_key, recursivef_proof.as_ptr())
            }?,
        }
    }

    Ok(is_valid)
}

/// Verifies `snark_proof` against the verification key at `vkey_path`: `snarkjs <protocol> verify`
/// with snarkjs's `final.verkey.json` for PLONK and FFLONK, and pilfflonk's JS verifier with
/// `pilfflonk.vkey.json` (`provingKeySnark/final/provingKey/<name>/pilfflonk/`) for pilfflonk.
pub fn verify_snark_proof(snark_proof: &SnarkProof, vkey_path: &Path) -> ProofmanResult<()> {
    let protocol = SnarkProtocol::from_protocol_id(snark_proof.protocol_id)?;
    // What `snarkjs <protocol> verify` prints when the proof verifies.
    let verified_message = match protocol {
        SnarkProtocol::Pilfflonk => return verify_pilfflonk_snark_proof(snark_proof, vkey_path),
        SnarkProtocol::Plonk => "OK",
        SnarkProtocol::Fflonk => "PROOF VERIFIED SUCCESSFULLY",
    };

    let (proof_json_value, publics_json_value) = snark_proof
        .convert_to_json()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to convert SNARK proof to JSON: {}", e)))?;

    let (proof_path, publics_path) = temp_json_paths();

    let proof_json_str = serde_json::to_string_pretty(&proof_json_value)
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to serialize proof JSON: {}", e)))?;
    let publics_json_str = serde_json::to_string_pretty(&publics_json_value)
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to serialize publics JSON: {}", e)))?;
    std::fs::write(&proof_path, proof_json_str)?;
    std::fs::write(&publics_path, publics_json_str)?;

    // Call snarkjs verify
    let output = Command::new("snarkjs")
        .arg(protocol.protocol_name())
        .arg("verify")
        .arg(vkey_path)
        .arg(&publics_path)
        .arg(&proof_path)
        .output()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to execute snarkjs: {}", e)))?;

    remove_temp_json(&proof_path, &publics_path);

    let verdict = if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).contains(verified_message))
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    };
    report_verdict(verdict)
}

/// [`verify_snark_proof`] of a pilfflonk proof: its JSON views, named by the vkey at `vkey_path`
/// (`pilfflonk_wrap::json_views`), go to pilfflonk's JS verifier. Bytes that are not those of a
/// proof of the vkey, or not as many publics as it has, are refused as the verifier refuses a proof.
fn verify_pilfflonk_snark_proof(snark_proof: &SnarkProof, vkey_path: &Path) -> ProofmanResult<()> {
    let views = pilfflonk_wrap::read_vkey(vkey_path)
        .and_then(|vkey| pilfflonk_wrap::json_views(&snark_proof.proof_bytes, &snark_proof.public_snark_bytes, &vkey));
    let verdict = views.and_then(|(proof_json, publics)| {
        let (proof_path, publics_path) = temp_json_paths();
        let verdict = pilfflonk_wrap::verify_json(vkey_path, &proof_json, &publics, &proof_path, &publics_path);
        remove_temp_json(&proof_path, &publics_path);
        verdict
    });
    report_verdict(verdict.map_err(|e| e.to_string()))
}

/// Paths for the JSON views of a proof and of its publics in the temporary directory, unique to
/// this process and this call, so that verifications do not race.
fn temp_json_paths() -> (PathBuf, PathBuf) {
    let temp_dir = std::env::temp_dir();
    let unique_id = format!(
        "{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_nanos()
    );
    let proof_path = temp_dir.join(format!("snark_proof_{}.json", unique_id));
    let publics_path = temp_dir.join(format!("snark_publics_{}.json", unique_id));
    (proof_path, publics_path)
}

/// Removes the files of [`temp_json_paths`], with a warning for each it cannot remove.
fn remove_temp_json(proof_path: &Path, publics_path: &Path) {
    if let Err(e) = std::fs::remove_file(proof_path) {
        tracing::warn!("Failed to remove temporary SNARK proof file {}: {}", proof_path.display(), e);
    }
    if let Err(e) = std::fs::remove_file(publics_path) {
        tracing::warn!("Failed to remove temporary SNARK publics file {}: {}", publics_path.display(), e);
    }
}

/// The result of a verifier's verdict, logged: whether the proof verifies, or why the verifier
/// could not say.
fn report_verdict(verdict: Result<bool, String>) -> ProofmanResult<()> {
    match verdict {
        Ok(true) => {
            tracing::info!("    {}", "\u{2713} SNARK proof was verified".bright_green().bold());
            Ok(())
        }
        Ok(false) => {
            tracing::info!("··· {}", "\u{2717} SNARK proof was not verified".bright_red().bold());
            Err(ProofmanError::InvalidProof("SNARK proof was not verified".to_string()))
        }
        Err(reason) => {
            tracing::info!("··· {}", "\u{2717} SNARK verification failed".bright_red().bold());
            Err(ProofmanError::InvalidProof(format!("SNARK proof verification failed: {}", reason)))
        }
    }
}

/// Sets proofman's `reload_fixed_pols_gpu`, if any, when it goes, on any return of a wrap's proof, an
/// error or a panic included: the snark's carve and pilfflonk's arena overwrote the const-pols regions
/// inside the unified buffer, or may have, and the flag, consumed after the next `wcm.execute()`,
/// re-uploads them before any proof.
struct ReloadFixedPolsOnDrop<'a>(Option<&'a AtomicBool>);

impl Drop for ReloadFixedPolsOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(flag) = self.0 {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

// SAFETY: every field is `Send` but the raw pointers and pilfflonk's prover in `proving`, and none
// of those is tied to the thread that made it:
// - `snark_prover` (rapidsnark's prover of `final.zkey`) and `d_buffers_recursivef` (the
//   recursivef's device buffers) are the wrapper's, which its `Drop` frees, and `d_buffers` (the
//   caller's unified buffer) it never frees. They are C++ objects of the process and CUDA memory of
//   its contexts, which each proof already hands to threads it spawns: the recursivef's loader of
//   its fixed columns, and the GPU preallocation of rapidsnark's prover;
// - pilfflonk's prover owns its key, whose `PilFflonkProverCtx` is a C++ object any thread may use
//   (pilfflonk_api.hpp), with the status of each call kept per thread and read on the calling
//   thread right after it, and the final circuit's witness library, a `WrapWitness`, which is
//   `Send`.
unsafe impl<F: PrimeField64> Send for PoseidonWrap<F> {}
// SAFETY: a shared wrapper proves one proof at a time: `generate_final_snark_proof` holds the lock
// of `proving` throughout, and only a proof uses pilfflonk's prover and the final circuit's witness
// calculator, or hands the C++ side what it writes: rapidsnark's prover, the recursivef's and the
// caller's device buffers (the raw pointers), and the recursivef's prover buffer `aux_trace`,
// through a pointer of the shared `Vec`. The threads a proof spawns are joined before it returns,
// on its errors too. Every other field is `Sync`. The raw pointers are public, and handing one to
// the C++ side outside a proof is the caller's to order with the wrapper's proofs, as for any call
// of `proofman_starks_lib_c` with them.
unsafe impl<F: PrimeField64> Sync for PoseidonWrap<F> {}

#[cfg(test)]
mod tests {
    use super::{
        verify_snark_proof, FinalSnarkKey, ReloadFixedPolsOnDrop, SnarkProof, SnarkProtocol, PILFFLONK_PROTOCOL_ID,
    };
    use proofman_common::ProofmanError;
    use proofman_starks_lib_c::{
        free_final_snark_prover_c, generate_fflonk_zkey_c, generate_plonk_zkey_c, init_final_snark_prover_c,
    };
    use serde_json::json;
    use std::path::{Path, PathBuf};

    /// A fresh directory under the temporary directory, removed when dropped.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("proofman_snark_wrapper_{name}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            TestDir(dir)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// 32 little-endian bytes from 64 hex digits of a big-endian number.
    fn le_from_hex(hex: &str) -> Vec<u8> {
        let mut bytes: Vec<u8> = (0..32).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        bytes.reverse();
        bytes
    }

    /// BN128's base field modulus q and scalar field modulus r.
    const Q: &str = "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";
    const R: &str = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";

    /// A binfile as snarkjs and rapidsnark write them: type, version, sections as (id, bytes).
    fn binfile(file_type: &[u8; 4], sections: &[(u32, Vec<u8>)]) -> Vec<u8> {
        let mut bytes = file_type.to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend((sections.len() as u32).to_le_bytes());
        for (id, contents) in sections {
            bytes.extend(id.to_le_bytes());
            bytes.extend((contents.len() as u64).to_le_bytes());
            bytes.extend(contents);
        }
        bytes
    }

    /// A ptau of `n_g1` G1 points and two G2 points, all zero: rapidsnark's setup copies and
    /// commits with them without checking them, and the protocol of the zkey does not depend on
    /// them. Section 12 is empty: the setup only checks that it exists.
    fn zero_ptau(n_g1: usize) -> Vec<u8> {
        let mut header = 32u32.to_le_bytes().to_vec();
        header.extend(le_from_hex(Q));
        header.extend(8u32.to_le_bytes());
        header.extend(8u32.to_le_bytes());
        binfile(b"ptau", &[(1, header), (2, vec![0; 64 * n_g1]), (3, vec![0; 2 * 128]), (12, vec![])])
    }

    /// The r1cs of `x * x = y`: wire 0 is the constant 1, wire 1 the public output y, wire 2 the
    /// private input x.
    fn square_r1cs() -> Vec<u8> {
        let mut header = 32u32.to_le_bytes().to_vec();
        header.extend(le_from_hex(R));
        for count in [3u32, 1, 0, 1] {
            header.extend(count.to_le_bytes()); // wires, outputs, public inputs, private inputs
        }
        header.extend(3u64.to_le_bytes()); // labels
        header.extend(1u32.to_le_bytes()); // constraints
        let one = le_from_hex(&format!("{:064x}", 1));
        let mut constraints = Vec::new();
        for wire in [2u32, 2, 1] {
            constraints.extend(1u32.to_le_bytes());
            constraints.extend(wire.to_le_bytes());
            constraints.extend(&one);
        }
        let wire_to_label = (0..3u64).flat_map(u64::to_le_bytes).collect();
        binfile(b"r1cs", &[(1, header), (2, constraints), (3, wire_to_label)])
    }

    /// The reload flag is set as a wrap's proof ends, on its errors and panics too, and without one
    /// (no unified buffer) nothing is.
    #[test]
    fn the_reload_flag_is_set_whatever_the_proof_ends_with() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let proof = |flag: Option<&AtomicBool>, fails: bool| -> Result<(), ()> {
            let _reload = ReloadFixedPolsOnDrop(flag);
            assert!(flag.is_none_or(|f| !f.load(Ordering::SeqCst)), "set before the proof ends");
            if fails {
                return Err(());
            }
            Ok(())
        };
        for fails in [false, true] {
            let flag = AtomicBool::new(false);
            assert_eq!(proof(Some(&flag), fails).is_err(), fails);
            assert!(flag.load(Ordering::SeqCst), "fails: {fails}");
            assert_eq!(proof(None, fails).is_err(), fails);
        }
        let flag = AtomicBool::new(false);
        let panicked = std::panic::catch_unwind(|| {
            let _reload = ReloadFixedPolsOnDrop(Some(&flag));
            panic!("a proof that panics");
        });
        assert!(panicked.is_err() && flag.load(Ordering::SeqCst));
    }

    #[test]
    fn protocol_ids_map_to_protocols() {
        assert!(matches!(SnarkProtocol::from_protocol_id(2), Ok(SnarkProtocol::Plonk)));
        assert!(matches!(SnarkProtocol::from_protocol_id(10), Ok(SnarkProtocol::Fflonk)));
        assert!(matches!(SnarkProtocol::from_protocol_id(0x7066), Ok(SnarkProtocol::Pilfflonk)));
        for protocol in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk, SnarkProtocol::Pilfflonk] {
            let id = protocol.protocol_id();
            assert_eq!(
                SnarkProtocol::from_protocol_id(id).map(|p| p.protocol_name()).ok(),
                Some(protocol.protocol_name())
            );
        }
        assert_eq!(SnarkProtocol::Plonk.protocol_name(), "plonk");
        assert_eq!(SnarkProtocol::Fflonk.protocol_name(), "fflonk");
        assert_eq!(SnarkProtocol::Pilfflonk.protocol_name(), "pilfflonk");
        for id in [0, 1, 3, 9, 11, 0x7065, 0x7067, u64::MAX] {
            assert!(SnarkProtocol::from_protocol_id(id).is_err(), "protocol id {id}");
        }
    }

    /// The globalInfo of a pilfflonk key of one AIR, as setup-pilfflonk writes the wrap's.
    const PILFFLONK_GLOBAL_INFO: &str = r#"{
 "name": "final",
 "airs": [[{"name": "Wrap", "num_rows": 16}]],
 "air_groups": ["Wrap"],
 "aggTypes": [[]],
 "backend": "pilfflonk",
 "formatVersion": 1,
 "field": "bn128",
 "modulus": "21888242871839275222246405745257275088548364400416034343698204186575808495617",
 "transcript": "keccak256",
 "setupParams": {"maxConstraintDegree": 9, "extraMuls": 2, "maxQDegree": 0, "packing": true},
 "nPublics": 1,
 "numChallenges": [0, 2],
 "numProofValues": [],
 "proofValuesMap": [],
 "publicsMap": [{"name": "publics", "stage": 1, "lengths": [0]}]
}"#;

    /// A `provingKeySnark/final/` in `dir` with a `final.zkey` if `zkey`, and a `provingKey/` whose
    /// globalInfo is `global_info` if there is one; and the stem of its files, `final/final`.
    fn final_dir(dir: &TestDir, zkey: bool, global_info: Option<&str>) -> PathBuf {
        let final_dir = dir.0.join("final");
        std::fs::create_dir_all(&final_dir).unwrap();
        if zkey {
            std::fs::write(final_dir.join("final.zkey"), b"zkey").unwrap();
        }
        if let Some(global_info) = global_info {
            std::fs::create_dir_all(final_dir.join("provingKey")).unwrap();
            std::fs::write(final_dir.join("provingKey/pilout.globalInfo.json"), global_info).unwrap();
        }
        final_dir.join("final")
    }

    fn setup_error(result: Result<FinalSnarkKey, ProofmanError>) -> String {
        match result {
            Err(ProofmanError::InvalidSetup(message)) => message,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_final_snark_key_is_final_zkey_or_a_pilfflonk_proving_key() {
        let dir = TestDir::new("zkey");
        let stem = final_dir(&dir, true, None);
        assert_eq!(FinalSnarkKey::find(&stem).unwrap(), FinalSnarkKey::Zkey(dir.0.join("final/final.zkey")));

        let dir = TestDir::new("pilfflonk");
        let stem = final_dir(&dir, false, Some(PILFFLONK_GLOBAL_INFO));
        assert_eq!(FinalSnarkKey::find(&stem).unwrap(), FinalSnarkKey::Pilfflonk(dir.0.join("final/provingKey")));
    }

    #[test]
    fn a_final_dir_with_both_keys_or_none_is_refused() {
        let dir = TestDir::new("both");
        let message = setup_error(FinalSnarkKey::find(&final_dir(&dir, true, Some(PILFFLONK_GLOBAL_INFO))));
        assert!(message.contains("two final SNARK keys") && message.contains("final.zkey"), "{message}");
        assert!(message.contains("provingKey"), "{message}");
        // What fixes the directory: setup-snark of the protocol to prove with removes the other key.
        assert!(message.contains("setup-snark writes the key of one protocol and removes the other's"), "{message}");
        assert!(message.contains("running it again with the --final-snark to prove with"), "{message}");

        let dir = TestDir::new("none");
        let message = setup_error(FinalSnarkKey::find(&final_dir(&dir, false, None)));
        assert!(message.contains("no final SNARK key") && message.contains("final.zkey"), "{message}");
    }

    /// A STARK's globalInfo, another backend's, one that is not JSON and none at all.
    #[test]
    fn a_proving_key_that_is_not_pilfflonks_is_refused() {
        let stark = r#"{"name": "t", "airs": [[{"name": "A", "num_rows": 16}]], "air_groups": ["G"],
            "aggTypes": [[]], "curve": "None", "latticeSize": 368, "transcriptArity": 4, "aggregationArity": 3,
            "hasCompressedFinal": true, "nPublics": 0, "numChallenges": [0], "numProofValues": [0],
            "proofValuesMap": [], "publicsMap": [], "hash": "Poseidon2"}"#;
        let other_backend = PILFFLONK_GLOBAL_INFO.replace("\"backend\": \"pilfflonk\"", "\"backend\": \"stark\"");
        for (name, global_info) in [("stark", stark), ("backend", &other_backend), ("text", "not JSON")] {
            let dir = TestDir::new(name);
            let message = setup_error(FinalSnarkKey::find(&final_dir(&dir, false, Some(global_info))));
            assert!(message.contains("is not a pilfflonk key"), "{name}: {message}");
        }
        let dir = TestDir::new("empty");
        std::fs::create_dir_all(dir.0.join("final/provingKey")).unwrap();
        let message = setup_error(FinalSnarkKey::find(&dir.0.join("final/final")));
        assert!(message.contains("is not a pilfflonk key"), "{message}");
    }

    /// rapidsnark's own setup writes a tiny zkey of each protocol, and the prover loaded from each
    /// reports the protocol that zkey names.
    #[test]
    fn the_protocol_is_the_one_the_zkey_names() {
        let dir = std::env::temp_dir().join(format!("proofman_snark_protocol_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let r1cs = dir.join("square.r1cs");
        let ptau = dir.join("zero.ptau");
        std::fs::write(&r1cs, square_r1cs()).unwrap();
        std::fs::write(&ptau, zero_ptau(256)).unwrap();

        type ZkeySetup = fn(&str, &str, &str) -> i32;
        let setups: [(ZkeySetup, SnarkProtocol); 2] =
            [(generate_plonk_zkey_c, SnarkProtocol::Plonk), (generate_fflonk_zkey_c, SnarkProtocol::Fflonk)];
        for (setup, expected) in setups {
            // As setup-snark writes it: provingKeySnark/final/final.zkey, which is the final SNARK key.
            let stem = dir.join(expected.protocol_name()).join("final").join("final");
            std::fs::create_dir_all(stem.parent().unwrap()).unwrap();
            let zkey = stem.with_extension("zkey");
            assert_eq!(FinalSnarkKey::find(&stem).ok(), None);
            let zkey = zkey.to_str().unwrap();
            assert_eq!(setup(r1cs.to_str().unwrap(), ptau.to_str().unwrap(), zkey), 0, "{zkey}");
            assert_eq!(FinalSnarkKey::find(&stem).unwrap(), FinalSnarkKey::Zkey(PathBuf::from(zkey)));

            let prover = init_final_snark_prover_c(zkey, std::ptr::null_mut());
            assert!(!prover.is_null(), "{zkey}");
            let protocol = SnarkProtocol::from_snark_prover(prover).map(|p| p.protocol_id());
            free_final_snark_prover_c(prover);
            assert_eq!(protocol.ok(), Some(expected.protocol_id()), "{zkey}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_prover_that_is_not_loaded_has_no_protocol() {
        assert!(SnarkProtocol::from_snark_prover(std::ptr::null_mut()).is_err());
    }

    /// 24 words 1, 2, ..., 24: 9 commitments and 6 evaluations for PLONK, 4 and 16 for FFLONK.
    fn proof_of_words(protocol_id: u64) -> SnarkProof {
        let word = |value: u8| {
            let mut word = [0u8; 32];
            word[31] = value;
            word
        };
        SnarkProof::new((1..=24).flat_map(word).collect(), vec![], word(25).to_vec(), protocol_id)
    }

    #[test]
    fn a_plonk_proof_converts_to_snarkjs_plonk_json() {
        let (proof, publics) = proof_of_words(2).convert_to_json().unwrap();
        assert_eq!(proof["protocol"], "plonk");
        assert_eq!(proof["curve"], "bn128");
        assert_eq!(proof["A"], json!(["1", "2", "1"]));
        assert_eq!(proof["Wxiw"], json!(["17", "18", "1"]));
        assert_eq!(proof["eval_a"], "19");
        assert_eq!(proof["eval_zw"], "24");
        assert_eq!(proof.as_object().unwrap().len(), 9 + 6 + 2);
        assert_eq!(publics, json!(["25"]));
    }

    #[test]
    fn an_fflonk_proof_converts_to_snarkjs_fflonk_json() {
        let (proof, publics) = proof_of_words(10).convert_to_json().unwrap();
        assert_eq!(proof["protocol"], "fflonk");
        assert_eq!(proof["curve"], "bn128");
        assert_eq!(proof["polynomials"]["C1"], json!(["1", "2", "1"]));
        assert_eq!(proof["polynomials"]["W2"], json!(["7", "8", "1"]));
        assert_eq!(proof["evaluations"]["ql"], "9");
        assert_eq!(proof["evaluations"]["inv"], "24");
        assert_eq!(proof["polynomials"].as_object().unwrap().len(), 4);
        assert_eq!(proof["evaluations"].as_object().unwrap().len(), 16);
        assert_eq!(proof.as_object().unwrap().len(), 4);
        assert_eq!(publics, json!(["25"]));
    }

    #[test]
    fn a_proof_of_an_unknown_protocol_does_not_convert() {
        assert!(proof_of_words(7).convert_to_json().is_err());
        assert!(proof_of_words(7).convert_to_json_with_vkey(Path::new("vkey.json")).is_err());
    }

    /// PLONK's and FFLONK's bytes name themselves: the vkey is not read, and need not be there.
    #[test]
    fn a_rapidsnark_proof_converts_as_snarkjs_reads_it_whatever_the_vkey() {
        for protocol in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk] {
            let proof = proof_of_words(protocol.protocol_id());
            let with_vkey = proof.convert_to_json_with_vkey(Path::new("no/such/vkey.json")).unwrap();
            assert_eq!(with_vkey, proof.convert_to_json().unwrap(), "{}", protocol.protocol_name());
        }
    }

    /// A pilfflonk proof's bytes need the vkey's names, and a vkey that cannot be read refuses them.
    #[test]
    fn a_pilfflonk_proof_converts_only_with_its_vkey() {
        let proof = proof_of_words(PILFFLONK_PROTOCOL_ID);
        let error = proof.convert_to_json().unwrap_err().to_string();
        assert!(error.contains("convert_to_json_with_vkey"), "{error}");
        let dir = TestDir::new("not_a_vkey");
        let not_a_vkey = dir.0.join("pilfflonk.vkey.json");
        std::fs::write(&not_a_vkey, "{}").unwrap();
        for vkey in [dir.0.join("none.json"), not_a_vkey] {
            assert!(proof.convert_to_json_with_vkey(&vkey).is_err(), "{}", vkey.display());
        }
    }

    /// A pilfflonk proof against a file that is not its vkey is refused before the JS verifier runs,
    /// as a proof that does not verify.
    #[test]
    fn a_pilfflonk_proof_without_its_vkey_does_not_verify() {
        let dir = TestDir::new("verify_not_a_vkey");
        let plonk_vkey = dir.0.join("final.verkey.json");
        std::fs::write(&plonk_vkey, r#"{"protocol": "plonk", "curve": "bn128", "nPublic": 1}"#).unwrap();
        for vkey in [dir.0.join("none.json"), plonk_vkey] {
            match verify_snark_proof(&proof_of_words(PILFFLONK_PROTOCOL_ID), &vkey) {
                Err(ProofmanError::InvalidProof(message)) => {
                    assert!(message.starts_with("SNARK proof verification failed"), "{message}")
                }
                other => panic!("{}: {other:?}", vkey.display()),
            }
        }
    }

    /// Each value's varint in bincode's standard configuration: one byte below 251, and 251 and
    /// the two bytes of a `u16` up to its maximum.
    fn varint(value: u64) -> Vec<u8> {
        match u16::try_from(value) {
            Ok(value) if value < 251 => vec![value as u8],
            Ok(value) => [vec![251], value.to_le_bytes().to_vec()].concat(),
            Err(_) => unreachable!("the test's values are below 2^16"),
        }
    }

    /// `snark_proof.bin` holds the fields of a [`SnarkProof`] in order, each byte vector its length
    /// and its bytes, and the protocol id last: the layout PLONK and FFLONK proofs have always had
    /// (768 bytes of proof, 24 words), which a pilfflonk proof shares with its own length and id.
    /// Every protocol's proof is read back as it was saved.
    #[test]
    fn a_snark_proof_is_saved_as_its_fields_and_read_back() {
        let dir = TestDir::new("save");
        for (protocol, proof_len) in
            [(SnarkProtocol::Plonk, 768), (SnarkProtocol::Fflonk, 768), (SnarkProtocol::Pilfflonk, 2048)]
        {
            let proof_bytes: Vec<u8> = (0..proof_len).map(|i| (i % 256) as u8).collect();
            let proof = SnarkProof::new(proof_bytes.clone(), vec![7; 32], vec![9; 32], protocol.protocol_id());
            let path = dir.0.join(format!("{}_snark_proof.bin", protocol.protocol_name()));
            proof.save(&path).unwrap();

            let expected = [
                varint(proof_len as u64),
                proof_bytes,
                varint(32),
                vec![7; 32],
                varint(32),
                vec![9; 32],
                varint(protocol.protocol_id()),
            ]
            .concat();
            assert_eq!(std::fs::read(&path).unwrap(), expected, "{}", protocol.protocol_name());

            let back = SnarkProof::load(&path).unwrap();
            assert_eq!(
                (back.proof_bytes, back.public_bytes, back.public_snark_bytes, back.protocol_id),
                (proof.proof_bytes, proof.public_bytes, proof.public_snark_bytes, proof.protocol_id)
            );
        }
    }
}
