use proofman_common::{
    GlobalInfoAir, ProofmanError, ProofmanResult, ProofType, PublicsInfo, Setup, calculate_fixed_tree_snark,
    load_const_pols_recursivef, load_const_pols_tree, MemoryHandlerRecursive, VerboseMode, initialize_logger,
};
use proofman_util::{timer_start_info, timer_stop_and_log_info, timer_start_debug, timer_stop_and_log_debug};
use proofman_verifier::VadcopFinalProof;
use fields::PrimeField64;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs::File;
use std::process::Command;
use colored::Colorize;
use std::io::Read;
use std::ffi::c_void;
use crate::check_const_tree;
use proofman_starks_lib_c::{
    init_final_snark_prover_c, free_final_snark_prover_c, snark_proof_bytes_to_json_c,
    get_unified_buffer_gpu_for_recursivef_c, free_fixed_pols_buffer_gpu_c, pre_allocate_final_snark_prover_c,
    alloc_fixed_pols_buffer_gpu_c, free_device_buffers_recursivef_c, gen_device_buffers_recursivef_c, set_gpu_mode_c,
    get_num_gpus_c, init_gpu_setup_c, GOLDILOCKS_MERKLE_TREE_ARITY,
};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::{verify_proof_bn128, generate_witness_final_snark, generate_recursivef_proof, generate_snark_proof};
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
    pub protocol: SnarkProtocol,
    pub memory_handler_recursive_witness: Arc<MemoryHandlerRecursive<F>>,
    pub gpu: bool,
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
            false,
            gpu,
            None,
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

        let d_buffers_vadcop = if let Some(d_buffers) = d_buffers { d_buffers } else { std::ptr::null_mut() };

        let p_setup: *mut c_void = (&setup_recursivef.p_setup).into();

        let verkey_path = setup_recursivef.verkey_file.clone();
        let mut contents = String::new();
        let mut file = File::open(verkey_path).unwrap();
        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));

        let verkey_str: String = serde_json::from_str(&contents)
            .map_err(|err| ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err)))?;

        let d_buffers_recursivef = gen_device_buffers_recursivef_c(
            p_setup as *mut u8,
            setup_recursivef.prover_buffer_size,
            d_buffers_vadcop as *mut u8,
            &verkey_str,
        ) as *mut c_void;

        timer_start_info!(INITIALIZING_FINAL_SNARK_PROVER);
        let zkey_filename = setup_snark_path.display().to_string() + ".zkey";
        let snark_prover = if preload {
            let snark_prover = init_final_snark_prover_c(zkey_filename.as_str(), d_buffers_recursivef);
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

        let witness_size = setup_recursivef.get_circom_witness_size();
        let trace_size = setup_recursivef.stark_info.map_sections_n["cm1"]
            * (1 << setup_recursivef.stark_info.stark_struct.n_bits)
            + setup_recursivef.stark_info.n_publics;

        let memory_handler_recursive_witness =
            Arc::new(MemoryHandlerRecursive::new(1, 0, witness_size, 0, trace_size as usize, 0));

        Ok(Self {
            aux_trace,
            recursivef_const_pols,
            recursivef_const_tree,
            setup_recursivef,
            setup_snark_path,
            snark_prover,
            proving_key_path: proving_key_path.to_path_buf(),
            protocol: SnarkProtocol::Plonk, // Default to Plonk, can be changed later if needed
            vadcop_final_verkey,
            d_buffers,
            d_buffers_recursivef,
            memory_handler_recursive_witness,
            reload_fixed_pols_gpu,
            gpu,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_final_snark_proof(
        &self,
        vadcop_proof: &VadcopFinalProof,
        verkey_override: Option<&[u64]>,
    ) -> ProofmanResult<SnarkProof> {
        timer_start_info!(GENERATING_WRAPPER_SNARK_PROOF);

        if let Some(d_buffer) = self.d_buffers {
            free_fixed_pols_buffer_gpu_c(d_buffer);
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
            recursivef_proof,
            prealloc_handle,
            self.d_buffers_recursivef,
        )?;

        let publics_info = PublicsInfo::from_folder(&self.proving_key_path)?;
        let public_bytes = get_public_bytes_solidity(&publics_info, &proof[1..1 + proof[0] as usize])?;
        let snark_proof =
            SnarkProof::new(snark_proof_bytes, public_bytes, snark_publics_bytes, self.protocol.protocol_id());

        timer_stop_and_log_debug!(GENERATING_SNARK_PROOF);

        timer_stop_and_log_info!(GENERATING_WRAPPER_SNARK_PROOF);

        if let Some(d_buffer) = self.d_buffers {
            alloc_fixed_pols_buffer_gpu_c(d_buffer);
            if let Some(reload_flag) = &self.reload_fixed_pols_gpu {
                reload_flag.store(true, Ordering::SeqCst);
            }
        }

        if self.snark_prover.is_none() {
            free_final_snark_prover_c(snark_prover);
        }

        Ok(snark_proof)
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
        false,
        gpu,
        None,
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
        false,
        gpu,
        None,
    )?;

    ensure_gpu_available(gpu)?;
    if gpu {
        init_gpu_setup_c(setup_recursivef.stark_info.stark_struct.n_bits_ext, GOLDILOCKS_MERKLE_TREE_ARITY);
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

    let d_buffers_recursivef = gen_device_buffers_recursivef_c(
        p_setup as *mut u8,
        setup_recursivef.prover_buffer_size,
        std::ptr::null_mut(),
        &verkey_str,
    ) as *mut c_void;

    let witness_size = setup_recursivef.get_circom_witness_size();
    let trace_size = setup_recursivef.stark_info.map_sections_n["cm1"]
        * (1 << setup_recursivef.stark_info.stark_struct.n_bits)
        + setup_recursivef.stark_info.n_publics;

    let memory_handler_recursive_witness =
        Arc::new(MemoryHandlerRecursive::new(1, 0, witness_size, 0, trace_size as usize, 0));

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
        d_buffers_recursivef,
    )?;
    timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

    timer_start_info!(VERIFY_RECURSIVE_F_PROOF);
    let mut publics: Vec<F> = vadcop_final_verkey[0..4].iter().map(|&x| F::from_u64(x)).collect();
    publics.extend(proof[1..1 + proof[0] as usize].iter().map(|&x| F::from_u64(x)));

    let is_valid = verify_proof_bn128(recursivef_proof, &setup_recursivef, Some(publics));
    timer_stop_and_log_info!(VERIFY_RECURSIVE_F_PROOF);

    let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));
    if setup_snark_path.parent().is_some_and(|p| p.exists()) {
        generate_witness_final_snark(recursivef_proof, &setup_snark_path)?;
    }

    free_device_buffers_recursivef_c(d_buffers_recursivef);

    Ok(is_valid)
}

pub fn verify_snark_proof(snark_proof: &SnarkProof, vkey_path: &Path) -> ProofmanResult<()> {
    let (proof_json_value, publics_json_value) = snark_proof
        .convert_to_json()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to convert SNARK proof to JSON: {}", e)))?;

    // Write JSON to temporary files with unique names to avoid race conditions
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

    let proof_json_str = serde_json::to_string_pretty(&proof_json_value)
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to serialize proof JSON: {}", e)))?;
    let publics_json_str = serde_json::to_string_pretty(&publics_json_value)
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to serialize publics JSON: {}", e)))?;
    std::fs::write(&proof_path, proof_json_str)?;
    std::fs::write(&publics_path, publics_json_str)?;

    // Determine protocol
    let protocol = SnarkProtocol::from_protocol_id(snark_proof.protocol_id)?;

    // Call snarkjs verify
    let output = Command::new("snarkjs")
        .arg(protocol.protocol_name())
        .arg("verify")
        .arg(vkey_path)
        .arg(&publics_path)
        .arg(&proof_path)
        .output()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to execute snarkjs: {}", e)))?;

    if let Err(e) = std::fs::remove_file(&proof_path) {
        tracing::warn!("Failed to remove temporary SNARK proof file {}: {}", proof_path.display(), e);
    }
    if let Err(e) = std::fs::remove_file(&publics_path) {
        tracing::warn!("Failed to remove temporary SNARK publics file {}: {}", publics_path.display(), e);
    }

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("OK") {
            tracing::info!("    {}", "\u{2713} SNARK proof was verified".bright_green().bold());
            Ok(())
        } else {
            tracing::info!("··· {}", "\u{2717} SNARK proof was not verified".bright_red().bold());
            Err(ProofmanError::InvalidProof("SNARK proof was not verified".to_string()))
        }
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::info!("··· {}", "\u{2717} SNARK verification failed".bright_red().bold());
        Err(ProofmanError::InvalidProof(format!("SNARK proof verification failed: {}", stderr)))
    }
}

unsafe impl<F: PrimeField64> Send for SnarkWrapper<F> {}
unsafe impl<F: PrimeField64> Sync for SnarkWrapper<F> {}
