use proofman_common::{
    format_bytes, GlobalInfoAir, ProofmanError, ProofmanResult, ProofType, PublicsInfo, Setup,
    calculate_fixed_tree_snark, load_const_pols_recursivef, load_const_pols_tree, MemoryHandlerRecursive, VerboseMode,
    initialize_logger,
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
    gen_device_buffers_recursivef_c, set_gpu_mode_c, get_num_gpus_c, init_gpu_setup_c, get_unified_buffer_gpu_size_c,
};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::{verify_proof_bn128, generate_witness_final_snark, generate_recursivef_proof, generate_snark_proof, PROVING};
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

pub enum SnarkProtocol {
    Fflonk,
    Plonk,
}

impl SnarkProtocol {
    pub fn protocol_id(&self) -> u64 {
        match self {
            SnarkProtocol::Fflonk => 10,
            SnarkProtocol::Plonk => 2,
        }
    }

    /// Serialized proof length: 24 big-endian 32-byte words for both protocols (PLONK: 9 G1
    /// points and 6 evaluations; FFLONK: 4 points and 16 evaluations).
    pub fn proof_bytes(&self) -> usize {
        24 * 32
    }

    pub fn protocol_name(&self) -> &'static str {
        match self {
            SnarkProtocol::Plonk => "plonk",
            SnarkProtocol::Fflonk => "fflonk",
        }
    }

    pub fn from_protocol_id(protocol_id: u64) -> ProofmanResult<Self> {
        match protocol_id {
            2 => Ok(SnarkProtocol::Plonk),
            10 => Ok(SnarkProtocol::Fflonk),
            _ => Err(ProofmanError::InvalidConfiguration(format!("Unsupported snark protocol id: {}", protocol_id))),
        }
    }

    /// The protocol of a loaded final SNARK prover: the one its zkey names.
    fn from_snark_prover(snark_prover: *mut c_void) -> ProofmanResult<Self> {
        Self::from_protocol_id(get_snark_protocol_id_c(snark_prover))
    }

    /// What `snarkjs <protocol> verify` prints when the proof verifies.
    fn verified_message(&self) -> &'static str {
        match self {
            SnarkProtocol::Plonk => "OK!",
            SnarkProtocol::Fflonk => "PROOF VERIFIED SUCCESSFULLY",
        }
    }
}

pub struct SnarkWrapper<F: PrimeField64> {
    pub setup_snark_path: PathBuf,
    pub setup_recursivef: Setup<F>,
    pub vadcop_final_verkey: Vec<u64>,
    pub aux_trace: Arc<Vec<F>>,
    pub recursivef_const_pols: Arc<Vec<F>>,
    pub recursivef_const_tree: Arc<Vec<F>>,
    pub d_buffers: Option<*mut c_void>,
    pub reload_fixed_pols_gpu: Option<Arc<AtomicBool>>,
    pub snark_prover: Option<*mut c_void>,
    pub d_buffers_recursivef: *mut c_void,
    pub proving_key_path: PathBuf,
    pub memory_handler_recursive_witness: Arc<MemoryHandlerRecursive<F>>,
    pub gpu: bool,
    /// Held by each proof throughout ([`generate_final_snark_proof`](Self::generate_final_snark_proof)),
    /// so that the wrapper proves one at a time: two proofs at once would both write the
    /// recursivef's prover buffer `aux_trace` and its device buffers, and rapidsnark's prover.
    proving: Mutex<()>,
}

/// The BN254 scalar order `r`, big-endian.
const BN254_R_BE: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d, 0x28, 0x33, 0xe8,
    0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
];

/// The first 32-byte big-endian word at or above `r`, if any. Words of equal length compare as
/// numbers.
fn first_word_not_below_r(bytes: &[u8]) -> Option<usize> {
    bytes.as_chunks::<32>().0.iter().position(|word| word >= &BN254_R_BE)
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

    pub fn convert_to_json(
        &self,
    ) -> Result<(serde_json::Value, serde_json::Value), Box<dyn std::error::Error + Send + Sync>> {
        // The C parser throws on an unknown protocol or a short proof, which aborts the process
        // across the FFI, and reads the digest in 32-byte words past a ragged end. Refuse here.
        let protocol = SnarkProtocol::from_protocol_id(self.protocol_id)?;
        if self.proof_bytes.len() != protocol.proof_bytes() {
            return Err(format!(
                "{} proof is {} bytes, expected {}",
                protocol.protocol_name(),
                self.proof_bytes.len(),
                protocol.proof_bytes()
            )
            .into());
        }
        if self.public_snark_bytes.is_empty() || !self.public_snark_bytes.len().is_multiple_of(32) {
            return Err(format!(
                "SNARK public input is {} bytes, not whole 32-byte field elements",
                self.public_snark_bytes.len()
            )
            .into());
        }
        // The Solidity verifiers snarkjs generates for both protocols require every proof word
        // below `r` (their `checkField`), while the parser reads coordinates mod q and
        // evaluations mod r: a word past either bound is another encoding of the same proof,
        // which would verify here and be rejected on-chain. A digest is reduced mod `r` too.
        if let Some(i) = first_word_not_below_r(&self.proof_bytes) {
            return Err(
                format!("{} proof word {i} is not below the BN254 scalar order", protocol.protocol_name()).into()
            );
        }
        if let Some(i) = first_word_not_below_r(&self.public_snark_bytes) {
            return Err(format!("SNARK public input {i} is not below the BN254 scalar order").into());
        }
        let (proof_json, publics_json) =
            snark_proof_bytes_to_json_c(&self.proof_bytes, &self.public_snark_bytes, self.protocol_id as i32);

        let proof_json_value: serde_json::Value = serde_json::from_str(&proof_json)?;
        let publics_json_value: serde_json::Value = serde_json::from_str(&publics_json)?;

        Ok((proof_json_value, publics_json_value))
    }

    pub fn get_public_bytes(&self) -> &[u8] {
        &self.public_bytes
    }
}

impl<F: PrimeField64> Drop for SnarkWrapper<F> {
    fn drop(&mut self) {
        if let Some(snark_prover) = self.snark_prover {
            free_final_snark_prover_c(snark_prover);
        }
        free_device_buffers_recursivef_c(self.d_buffers_recursivef);
    }
}

impl<F: PrimeField64> SnarkWrapper<F> {
    pub fn new(proving_key_path: &Path, verbose_mode: VerboseMode, preload: bool, gpu: bool) -> ProofmanResult<Self> {
        Self::new_with_preallocated_buffers(proving_key_path, verbose_mode, None, None, None, preload, gpu)
    }

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

        if d_buffers.is_some() && reload_fixed_pols_gpu.is_none() {
            return Err(ProofmanError::InvalidConfiguration(
                "A shared device buffer needs its owner's reload flag: the wrap overwrites it".to_string(),
            ));
        }

        let setup_recursivef_path =
            PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "recursivef", "recursivef"));
        let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));

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

        {
            // On a shared buffer the tree is built over its aux area: not while a ProofMan job runs.
            let _proving = d_buffers.map(|_| PROVING.lock().unwrap_or_else(|e| e.into_inner()));
            check_const_tree(&setup_recursivef, &d_buffers)?;
        }

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

        let d_buffers_vadcop = if let Some(d_buffers) = d_buffers { d_buffers } else { std::ptr::null_mut() };

        let p_setup: *mut c_void = (&setup_recursivef.p_setup).into();

        let verkey_path = setup_recursivef.verkey_file.clone();
        let mut contents = String::new();
        let mut file = File::open(verkey_path).unwrap();
        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));

        let verkey_str: String = serde_json::from_str(&contents)
            .map_err(|err| ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err)))?;

        if gpu && !d_buffers_vadcop.is_null() {
            let need = 8 * (setup_recursivef.const_tree_size as u64 + setup_recursivef.prover_buffer_size);
            let total = get_unified_buffer_gpu_size_c(d_buffers_vadcop);
            if need > total {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "recursiveF needs {} of device memory but the unified buffer holds {}",
                    format_bytes(need as f64),
                    format_bytes(total as f64),
                )));
            }
        }

        // Freed if the final SNARK's prover cannot be loaded.
        let d_buffers_recursivef = RecursivefDeviceBuffers(gen_device_buffers_recursivef_c(
            p_setup as *mut u8,
            setup_recursivef.prover_buffer_size,
            d_buffers_vadcop as *mut u8,
            &verkey_str,
        ) as *mut c_void);

        timer_start_info!(INITIALIZING_FINAL_SNARK_PROVER);
        let zkey_filename = setup_snark_path.display().to_string() + ".zkey";
        let snark_prover = if preload {
            let snark_prover = init_final_snark_prover_c(zkey_filename.as_str(), d_buffers_recursivef.as_ptr());
            if snark_prover.is_null() {
                return Err(std::io::Error::other(format!(
                    "Failed to initialize final snark prover from zkey file '{}'",
                    zkey_filename
                ))
                .into());
            }
            Some(snark_prover)
        } else {
            None
        };

        timer_stop_and_log_info!(INITIALIZING_FINAL_SNARK_PROVER);

        let trace_size = setup_recursivef.stark_info.map_sections_n["cm1"]
            * (1 << setup_recursivef.stark_info.stark_struct.n_bits)
            + setup_recursivef.stark_info.n_publics;

        let memory_handler_recursive_witness = Arc::new(MemoryHandlerRecursive::new(1, trace_size as usize));

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
            d_buffers_recursivef: d_buffers_recursivef.into_raw(),
            memory_handler_recursive_witness,
            reload_fixed_pols_gpu,
            gpu,
            proving: Mutex::new(()),
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
        let _proving = self.proving.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // A shared buffer: no ProofMan job or reset runs while recursiveF and the snark write it.
        let _shared = self.d_buffers.map(|_| PROVING.lock().unwrap_or_else(|e| e.into_inner()));

        // The recursivef's const tree and prover buffer, and the snark's carve, live in the unified
        // buffer and may overwrite its const-pols regions, also in a proof that fails. The flag,
        // consumed by the next job's restore_after_snark, re-uploads them before any proof; it is set as
        // this proof ends, whatever its outcome, before the locks are let go.
        let _reload = ReloadFixedPolsOnDrop(self.d_buffers.and(self.reload_fixed_pols_gpu.as_deref()));

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

        timer_start_debug!(GENERATING_SNARK_PROOF);

        let snark_prover = match self.snark_prover {
            Some(prover) => prover,
            None => {
                let prover = init_final_snark_prover_c(
                    &(self.setup_snark_path.display().to_string() + ".zkey"),
                    self.d_buffers_recursivef,
                );
                if prover.is_null() {
                    return Err(std::io::Error::other(format!(
                        "Failed to initialize final snark prover from zkey file '{}'",
                        self.setup_snark_path.display().to_string() + ".zkey"
                    ))
                    .into());
                }
                prover
            }
        };

        // The zkey names the protocol, and the loaded prover read it.
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

        let generated = generate_snark_proof(
            snark_prover,
            &self.setup_snark_path,
            recursivef_proof.as_ptr(),
            prealloc_handle,
            self.d_buffers_recursivef,
        );

        if self.snark_prover.is_none() {
            free_final_snark_prover_c(snark_prover);
        }
        let (snark_proof_bytes, snark_publics_bytes) = generated?;

        let publics_info = PublicsInfo::from_folder(&self.proving_key_path)?;
        let publics = &proof[1..1 + proof[0] as usize];
        // The digest verifiers recompute must be the one the circuit committed to.
        let rootc: &[u64; 4] = verkey.try_into().map_err(|_| {
            ProofmanError::InvalidConfiguration(format!("verkey has {} limbs, expected 4", verkey.len()))
        })?;
        if publics_info.snark_public_digest(publics, rootc)?.as_slice() != snark_publics_bytes.as_slice() {
            return Err(ProofmanError::InvalidProof(
                "the SNARK's public digest differs from PublicsInfo::snark_public_digest".to_string(),
            ));
        }
        let public_bytes = get_public_bytes_solidity(&publics_info, publics)?;
        let snark_proof = SnarkProof::new(snark_proof_bytes, public_bytes, snark_publics_bytes, protocol.protocol_id());

        timer_stop_and_log_debug!(GENERATING_SNARK_PROOF);

        timer_stop_and_log_info!(GENERATING_WRAPPER_SNARK_PROOF);

        Ok(snark_proof)
    }
}

/// The `publicValues` bytes a Solidity verifier takes. See
/// [`PublicsInfo::solidity_public_bytes`].
pub fn get_public_bytes_solidity(publics_info: &PublicsInfo, vadcop_public_inputs: &[u64]) -> ProofmanResult<Vec<u8>> {
    publics_info.solidity_public_bytes(vadcop_public_inputs)
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

    let p_setup: *mut c_void = (&setup_recursivef.p_setup).into();

    let verkey_path = setup_recursivef.verkey_file.clone();
    let mut contents = String::new();
    let mut file = File::open(verkey_path).unwrap();
    let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));

    let verkey_str: String = serde_json::from_str(&contents)
        .map_err(|err| ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err)))?;

    // Freed when this returns, whatever its outcome.
    let d_buffers_recursivef = RecursivefDeviceBuffers(gen_device_buffers_recursivef_c(
        p_setup as *mut u8,
        setup_recursivef.prover_buffer_size,
        std::ptr::null_mut(),
        &verkey_str,
    ) as *mut c_void);

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
        d_buffers_recursivef.as_ptr(),
    )?;
    timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

    timer_start_info!(VERIFY_RECURSIVE_F_PROOF);
    let mut publics: Vec<F> = vadcop_final_verkey[0..4].iter().map(|&x| F::from_u64(x)).collect();
    publics.extend(proof[1..1 + proof[0] as usize].iter().map(|&x| F::from_u64(x)));

    let is_valid = verify_proof_bn128(recursivef_proof.as_ptr(), &setup_recursivef, Some(publics));
    timer_stop_and_log_info!(VERIFY_RECURSIVE_F_PROOF);

    let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));
    if setup_snark_path.parent().is_some_and(|p| p.exists()) {
        generate_witness_final_snark(recursivef_proof.as_ptr(), &setup_snark_path)?;
    }

    Ok(is_valid)
}

/// Verify a SNARK proof against a trusted `vkey`, checking snark validity only — the
/// public input snarkjs is given is the digest the proof carries. That digest is the
/// committed statement only if the caller already trusts it (e.g. recomputed it from the
/// publics and a pinned rootC). On its own this does NOT bind the proof to any statement,
/// so prefer [`verify_snark_proof_with_expected`] with a pinned digest.
pub fn verify_snark_proof(snark_proof: &SnarkProof, vkey_path: &Path) -> ProofmanResult<()> {
    verify_snark_proof_with_expected(snark_proof, vkey_path, None)
}

/// As [`verify_snark_proof`], but when `expected_public_snark_bytes` is `Some`, require the
/// proof's committed public digest to equal it before verifying. That pins the statement:
/// the caller supplies the digest it recomputed from the publics and the trusted rootC, so
/// a proof carrying an attacker-chosen public input is rejected. With `None`, a warning is
/// emitted that the result attests snark self-consistency only.
pub fn verify_snark_proof_with_expected(
    snark_proof: &SnarkProof,
    vkey_path: &Path,
    expected_public_snark_bytes: Option<&[u8]>,
) -> ProofmanResult<()> {
    run_snarkjs_verify(snark_proof, vkey_path, expected_public_snark_bytes, &TempJsonFiles::new()?)
}

/// As [`verify_snark_proof_with_expected`], with the verification key given as its JSON
/// rather than a file.
pub fn verify_snark_proof_with_vkey(
    snark_proof: &SnarkProof,
    vkey_json: &str,
    expected_public_snark_bytes: Option<&[u8]>,
) -> ProofmanResult<()> {
    let temp = TempJsonFiles::new()?;
    let vkey_path = temp.write("snark_vkey", vkey_json)?;
    run_snarkjs_verify(snark_proof, &vkey_path, expected_public_snark_bytes, &temp)
}

/// JSON files snarkjs reads, in a private directory (random name, owner-only) that is
/// removed with them when dropped, on every return path.
struct TempJsonFiles(tempfile::TempDir);

impl TempJsonFiles {
    fn new() -> ProofmanResult<Self> {
        Ok(Self(tempfile::Builder::new().prefix("snarkjs_").tempdir()?))
    }

    fn write(&self, name: &str, contents: &str) -> ProofmanResult<PathBuf> {
        let path = self.0.path().join(format!("{name}.json"));
        std::fs::write(&path, contents)?;
        Ok(path)
    }
}

fn run_snarkjs_verify(
    snark_proof: &SnarkProof,
    vkey_path: &Path,
    expected_public_snark_bytes: Option<&[u8]>,
    temp: &TempJsonFiles,
) -> ProofmanResult<()> {
    match expected_public_snark_bytes {
        Some(expected) if expected != snark_proof.public_snark_bytes.as_slice() => {
            return Err(ProofmanError::InvalidProof(
                "SNARK public digest does not match the expected (pinned) statement".to_string(),
            ));
        }
        Some(_) => {}
        None => {
            tracing::warn!(
                "verify_snark_proof: no expected public digest pinned — this attests snark \
                 validity against the proof's own public input, not a statement. A relying \
                 party must pin the digest recomputed from the publics and the trusted rootC."
            );
        }
    }

    let (proof_json, publics_json) = snark_proof
        .convert_to_json()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to convert SNARK proof to JSON: {}", e)))?;
    let to_json = |value| {
        serde_json::to_string(value)
            .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to serialize SNARK JSON: {}", e)))
    };
    let proof_path = temp.write("snark_proof", &to_json(&proof_json)?)?;
    let publics_path = temp.write("snark_publics", &to_json(&publics_json)?)?;

    let protocol = SnarkProtocol::from_protocol_id(snark_proof.protocol_id)?;
    let output = Command::new("snarkjs")
        .arg(protocol.protocol_name())
        .arg("verify")
        .arg(vkey_path)
        .arg(&publics_path)
        .arg(&proof_path)
        .output()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to execute snarkjs: {}", e)))?;

    // snarkjs reports everything on stdout and exits 1 both when it rejects a proof and when it
    // cannot run (unreadable vkey, missing file, ...), so only its messages tell them apart.
    let stdout = String::from_utf8_lossy(&output.stdout);
    match snarkjs_verdict(&protocol, output.status.success(), &stdout) {
        SnarkjsVerdict::Verified => {
            tracing::info!("    {}", "\u{2713} SNARK proof was verified".bright_green().bold());
            Ok(())
        }
        SnarkjsVerdict::Rejected => {
            tracing::info!("··· {}", "\u{2717} SNARK proof was not verified".bright_red().bold());
            Err(ProofmanError::InvalidProof(format!("SNARK proof was not verified: {}", stdout.trim())))
        }
        SnarkjsVerdict::Failed => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(ProofmanError::InvalidConfiguration(format!(
                "snarkjs could not verify the proof ({}): {} {}",
                output.status,
                stdout.trim(),
                stderr.trim()
            )))
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SnarkjsVerdict {
    Verified,
    /// The checks ran and the proof failed them.
    Rejected,
    /// Nothing was decided: snarkjs could not read its inputs, or the key does not fit.
    Failed,
}

/// What snarkjs 0.7's `plonk verify` / `fflonk verify` concluded from its exit status and stdout.
fn snarkjs_verdict(protocol: &SnarkProtocol, success: bool, stdout: &str) -> SnarkjsVerdict {
    // A key with another public count is misconfiguration, not a false proof.
    const REJECTED: [&str; 5] = [
        "Invalid Proof",
        "Proof commitments are not valid",
        "Proof evaluations are not valid",
        "Public inputs are not valid",
        "Invalid number of public inputs",
    ];
    if success && stdout.contains(protocol.verified_message()) {
        SnarkjsVerdict::Verified
    } else if !success && REJECTED.iter().any(|m| stdout.contains(m)) {
        SnarkjsVerdict::Rejected
    } else {
        SnarkjsVerdict::Failed
    }
}

/// The recursivef's device buffers (`gen_device_buffers_recursivef_c`), freed when dropped unless
/// handed over with [`into_raw`](Self::into_raw), so that an error does not leak them. The CPU
/// backend has none: the pointer is null and nothing is freed.
struct RecursivefDeviceBuffers(*mut c_void);

impl RecursivefDeviceBuffers {
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

/// Sets proofman's `reload_fixed_pols_gpu`, if any, when it goes: on any return of a wrap's proof,
/// an error or a panic included.
struct ReloadFixedPolsOnDrop<'a>(Option<&'a AtomicBool>);

impl Drop for ReloadFixedPolsOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(flag) = self.0 {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

// SAFETY: every field is `Send` but the raw pointers, and none of those is tied to the thread
// that made it: `snark_prover` (rapidsnark's prover of `final.zkey`) and `d_buffers_recursivef`
// (the recursivef's device buffers) are the wrapper's, which its `Drop` frees, and `d_buffers`
// (the caller's unified buffer) it never frees. They are C++ objects of the process and CUDA
// memory of its contexts, which each proof already hands to threads it spawns: the recursivef's
// loader of its fixed columns, and the GPU pre-allocation of rapidsnark's prover.
unsafe impl<F: PrimeField64> Send for SnarkWrapper<F> {}
// SAFETY: a shared wrapper proves one proof at a time: `generate_final_snark_proof` holds the lock
// of `proving` throughout, and only a proof hands the C++ side what it writes: rapidsnark's
// prover, the recursivef's and the caller's device buffers (the raw pointers), and the
// recursivef's prover buffer `aux_trace`, through a pointer of the shared `Vec`. The threads a
// proof spawns are joined before it returns, on its errors too: its own (the recursivef's loader
// of its fixed columns, the GPU pre-allocation of rapidsnark's prover) and those rapidsnark's GPU
// prover starts, which its pre-allocation and its proof join before they return. Every other
// field is `Sync`. The raw pointers are public, and handing one to the C++ side outside a proof is
// the caller's to order with the wrapper's proofs, as for any call of `proofman_starks_lib_c`
// with them.
unsafe impl<F: PrimeField64> Sync for SnarkWrapper<F> {}

#[cfg(test)]
mod snarkjs_verdict_tests {
    use super::*;

    #[test]
    fn only_snarkjs_rejections_count_as_a_rejected_proof() {
        let plonk = &SnarkProtocol::Plonk;
        assert_eq!(snarkjs_verdict(plonk, true, "[INFO]  snarkJS: OK!\n"), SnarkjsVerdict::Verified);
        assert_eq!(snarkjs_verdict(plonk, false, "[WARN]  snarkJS: Invalid Proof\n"), SnarkjsVerdict::Rejected);
        assert_eq!(
            snarkjs_verdict(plonk, false, "[ERROR] snarkJS: Proof commitments are not valid.\n"),
            SnarkjsVerdict::Rejected
        );
        // Exit 0 without OK!, a key that does not fit, an unreadable file: nothing decided.
        assert_eq!(snarkjs_verdict(plonk, true, ""), SnarkjsVerdict::Failed);
        assert_eq!(
            snarkjs_verdict(plonk, false, "[ERROR] snarkJS: Number of public signals does not match with vk\n"),
            SnarkjsVerdict::Failed
        );
        assert_eq!(
            snarkjs_verdict(plonk, false, "[ERROR] snarkJS: Error: ENOENT: no such file\n"),
            SnarkjsVerdict::Failed
        );
        assert_eq!(snarkjs_verdict(plonk, false, ""), SnarkjsVerdict::Failed);
    }

    /// `snarkjs fflonk verify` prints its own success message, not PLONK's `OK!`, and each is
    /// only the verdict of its protocol.
    #[test]
    fn each_protocol_is_verified_by_its_own_message() {
        let fflonk = &SnarkProtocol::Fflonk;
        let fflonk_ok = "[INFO]  snarkJS: PROOF VERIFIED SUCCESSFULLY\n[INFO]  snarkJS: FFLONK VERIFIER FINISHED\n";
        assert_eq!(snarkjs_verdict(fflonk, true, fflonk_ok), SnarkjsVerdict::Verified);
        assert_eq!(snarkjs_verdict(fflonk, false, "[WARN]  snarkJS: Invalid Proof\n"), SnarkjsVerdict::Rejected);
        assert_eq!(
            snarkjs_verdict(fflonk, false, "[ERROR] snarkJS: Proof evaluations are not valid.\n"),
            SnarkjsVerdict::Rejected
        );
        assert_eq!(snarkjs_verdict(fflonk, true, "[INFO]  snarkJS: OK!\n"), SnarkjsVerdict::Failed);
        assert_eq!(snarkjs_verdict(&SnarkProtocol::Plonk, true, fflonk_ok), SnarkjsVerdict::Failed);
    }

    /// The reload flag is set as a wrap's proof ends, on its errors and panics too, and without one
    /// (no unified buffer) nothing is.
    #[test]
    fn the_reload_flag_is_set_whatever_the_proof_ends_with() {
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
        for protocol in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk] {
            let id = protocol.protocol_id();
            assert_eq!(
                SnarkProtocol::from_protocol_id(id).map(|p| p.protocol_name()).ok(),
                Some(protocol.protocol_name())
            );
        }
        assert_eq!(SnarkProtocol::Plonk.protocol_name(), "plonk");
        assert_eq!(SnarkProtocol::Fflonk.protocol_name(), "fflonk");
        for id in [0, 1, 3, 9, 11, u64::MAX] {
            assert!(SnarkProtocol::from_protocol_id(id).is_err(), "protocol id {id}");
        }
    }

    /// The PLONK and FFLONK zkeys of `x * x = y` in `tests/data/` report, once loaded, the protocol
    /// each names.
    ///
    /// They were made once with rapidsnark's setup, as setup-snark runs it:
    /// `generate_plonk_zkey_c(r1cs, ptau, "square_plonk.zkey")` and
    /// `generate_fflonk_zkey_c(r1cs, ptau, "square_fflonk.zkey")`, where the r1cs is
    /// `circom square.circom --r1cs` (circom 2.2.3) of
    /// `template Square() { signal input x; signal output y; y <== x * x; } component main = Square();`
    /// and the ptau is the snarkjs README's `powersOfTau28_hez_final_08.ptau`, whose blake2b is
    /// d6a8fb3a04feb600096c3b791f936a578c4e664d262e4aa24beed1b7a9a96aa5
    /// eb72864d628db247e9293384b74b36ffb52ca8d148d6e1b8b51e279fdf57b583.
    #[test]
    fn the_protocol_is_the_one_the_zkey_names() {
        for expected in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk] {
            let zkey = format!("{}/tests/data/square_{}.zkey", env!("CARGO_MANIFEST_DIR"), expected.protocol_name());
            // A zkey that cannot be read aborts the C++ loader: refuse a missing one first.
            assert!(Path::new(&zkey).is_file(), "no fixture at {zkey}");

            let prover = init_final_snark_prover_c(&zkey, std::ptr::null_mut());
            assert!(!prover.is_null(), "{zkey}");
            let protocol = SnarkProtocol::from_snark_prover(prover).map(|p| p.protocol_id());
            free_final_snark_prover_c(prover);
            assert_eq!(protocol.ok(), Some(expected.protocol_id()), "{zkey}");
        }
    }

    #[test]
    fn a_prover_that_is_not_loaded_has_no_protocol() {
        assert!(SnarkProtocol::from_snark_prover(std::ptr::null_mut()).is_err());
    }

    #[test]
    fn convert_to_json_refuses_what_the_native_parser_cannot_read() {
        let proof = |proof_len: usize, digest_len: usize, protocol_id: u64| SnarkProof {
            proof_bytes: vec![0; proof_len],
            public_bytes: vec![],
            public_snark_bytes: vec![0; digest_len],
            protocol_id,
        };
        let plonk = SnarkProtocol::Plonk.protocol_id();
        assert!(proof(767, 32, plonk).convert_to_json().is_err());
        assert!(proof(768, 31, plonk).convert_to_json().is_err());
        assert!(proof(768, 0, plonk).convert_to_json().is_err());
        assert!(proof(768, 32, 99).convert_to_json().is_err());
    }

    /// Adding `r` to a word, or `q` to a coordinate, re-encodes the same proof for the native
    /// parser; the on-chain verifier rejects both, so these are refused before parsing.
    #[test]
    fn convert_to_json_refuses_a_word_at_or_above_r() {
        const R_MINUS_1: [u8; 32] = {
            let mut w = BN254_R_BE;
            w[31] -= 1;
            w
        };
        let digest = R_MINUS_1.to_vec();
        for protocol in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk] {
            for i in [0, 17, 18, 23] {
                for word in [BN254_R_BE, [0xff; 32]] {
                    let mut proof_bytes = vec![0; protocol.proof_bytes()];
                    proof_bytes[32 * i..32 * (i + 1)].copy_from_slice(&word);
                    let proof = SnarkProof::new(proof_bytes, vec![], digest.clone(), protocol.protocol_id());
                    let err = proof.convert_to_json().unwrap_err().to_string();
                    assert!(err.contains(&format!("proof word {i} ")), "{err}");
                }
            }
        }

        let proof = SnarkProof::new(vec![0; 768], vec![], BN254_R_BE.to_vec(), SnarkProtocol::Plonk.protocol_id());
        assert!(proof.convert_to_json().unwrap_err().to_string().contains("public input 0 "));
    }

    #[test]
    fn first_word_not_below_r_compares_whole_words() {
        let mut below = BN254_R_BE;
        below[31] -= 1;
        let mut above_despite_a_smaller_last_byte = BN254_R_BE;
        above_despite_a_smaller_last_byte[1] += 1;
        above_despite_a_smaller_last_byte[31] = 0;

        assert_eq!(first_word_not_below_r(&below), None);
        assert_eq!(first_word_not_below_r(&[0; 64]), None);
        assert_eq!(first_word_not_below_r(&BN254_R_BE), Some(0));
        assert_eq!(first_word_not_below_r(&above_despite_a_smaller_last_byte), Some(0));
        assert_eq!(first_word_not_below_r(&[below, BN254_R_BE].concat()), Some(1));
    }

    /// 24 words 1, 2, ..., 24: 9 commitments and 6 evaluations for PLONK, 4 and 16 for FFLONK.
    fn proof_of_words(protocol: SnarkProtocol) -> SnarkProof {
        let word = |value: u8| {
            let mut word = [0u8; 32];
            word[31] = value;
            word
        };
        SnarkProof::new((1..=24).flat_map(word).collect(), vec![], word(25).to_vec(), protocol.protocol_id())
    }

    #[test]
    fn a_plonk_proof_converts_to_snarkjs_plonk_json() {
        let (proof, publics) = proof_of_words(SnarkProtocol::Plonk).convert_to_json().unwrap();
        assert_eq!(proof["protocol"], "plonk");
        assert_eq!(proof["curve"], "bn128");
        assert_eq!(proof["A"], serde_json::json!(["1", "2", "1"]));
        assert_eq!(proof["Wxiw"], serde_json::json!(["17", "18", "1"]));
        assert_eq!(proof["eval_a"], "19");
        assert_eq!(proof["eval_zw"], "24");
        assert_eq!(proof.as_object().unwrap().len(), 9 + 6 + 2);
        assert_eq!(publics, serde_json::json!(["25"]));
    }

    /// snarkjs reads an FFLONK proof's commitments under `polynomials` and its evaluations under
    /// `evaluations` (`fromObjectProof`).
    #[test]
    fn an_fflonk_proof_converts_to_snarkjs_fflonk_json() {
        let (proof, publics) = proof_of_words(SnarkProtocol::Fflonk).convert_to_json().unwrap();
        assert_eq!(proof["protocol"], "fflonk");
        assert_eq!(proof["curve"], "bn128");
        assert_eq!(proof["polynomials"]["C1"], serde_json::json!(["1", "2", "1"]));
        assert_eq!(proof["polynomials"]["W2"], serde_json::json!(["7", "8", "1"]));
        assert_eq!(proof["evaluations"]["ql"], "9");
        assert_eq!(proof["evaluations"]["inv"], "24");
        assert_eq!(proof["polynomials"].as_object().unwrap().len(), 4);
        assert_eq!(proof["evaluations"].as_object().unwrap().len(), 16);
        assert_eq!(proof.as_object().unwrap().len(), 4);
        assert_eq!(publics, serde_json::json!(["25"]));
    }
}
