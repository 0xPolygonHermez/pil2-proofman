use proofman_common::{
    GlobalInfoAir, ProofmanError, ProofmanResult, ProofType, PublicsInfo, Setup, calculate_fixed_tree, VerboseMode,
    initialize_logger,
};
use proofman_util::{timer_start_info, timer_stop_and_log_info, create_buffer_fast, VadcopFinalProof};
use fields::PrimeField64;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs::{File, create_dir_all};
use std::process::Command;
use colored::Colorize;
use std::ffi::c_void;
use proofman_starks_lib_c::{
    init_final_snark_prover_c, free_final_snark_prover_c, get_snark_protocol_id_c, snark_proof_bytes_to_json_c,
};
use crate::{verify_proof_bn128, generate_witness_final_snark, generate_recursivef_proof, generate_snark_proof};
use serde::{Deserialize, Serialize};

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
    pub aux_trace: Arc<Vec<F>>,
    pub snark_prover: *mut c_void,
    pub publics_info: PublicsInfo,
    pub protocol: SnarkProtocol,
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

    pub fn save(&self, dir: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let file_path = dir.as_ref().join("final_snark_proof.bin");

        let file = File::create(&file_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to create file for saving SNARK proof: {}: {}", file_path.display(), e),
            )
        })?;
        bincode::serialize_into(file, self)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path.as_ref()).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open file for loading SNARK proof: {}: {}", path.as_ref().display(), e),
            )
        })?;
        let proof: SnarkProof = bincode::deserialize_from(file)?;
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
}

impl<F: PrimeField64> Drop for SnarkWrapper<F> {
    fn drop(&mut self) {
        free_final_snark_prover_c(self.snark_prover);
    }
}

impl<F: PrimeField64> SnarkWrapper<F> {
    pub fn new(proving_key_path: &Path, verbose_mode: VerboseMode) -> ProofmanResult<Self> {
        initialize_logger(verbose_mode, None);

        let setup_recursivef_path =
            PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "recursivef", "recursivef"));
        let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));

        timer_start_info!(LOADING_RECURSIVE_F_SETUP);

        let setup_recursivef = Setup::new(
            &setup_recursivef_path,
            0,
            0,
            &GlobalInfoAir::new("RecursiveF".to_string()),
            &ProofType::RecursiveF,
            false,
            false,
            None,
        );

        setup_recursivef.set_circom_circuit()?;
        setup_recursivef.set_exec_file_data()?;

        calculate_fixed_tree(&setup_recursivef);

        setup_recursivef.load_const_pols();
        setup_recursivef.load_const_pols_tree();

        timer_stop_and_log_info!(LOADING_RECURSIVE_F_SETUP);

        let aux_trace = if cfg!(feature = "gpu") {
            Arc::new(Vec::new())
        } else {
            Arc::new(create_buffer_fast(setup_recursivef.prover_buffer_size as usize))
        };

        timer_start_info!(INITIALIZING_FINAL_SNARK_PROVER);
        let zkey_filename = setup_snark_path.display().to_string() + ".zkey";
        let snark_prover = init_final_snark_prover_c(zkey_filename.as_str());
        let protocol_id = get_snark_protocol_id_c(snark_prover);
        let protocol = SnarkProtocol::from_protocol_id(protocol_id)?;
        timer_stop_and_log_info!(INITIALIZING_FINAL_SNARK_PROVER);

        let publics_info = PublicsInfo::from_folder(proving_key_path)?;

        Ok(Self { aux_trace, setup_recursivef, setup_snark_path, snark_prover, publics_info, protocol })
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_final_snark_proof(
        &self,
        vadcop_proof: &VadcopFinalProof,
        output_dir_path: &Path,
    ) -> ProofmanResult<SnarkProof> {
        if vadcop_proof.compressed {
            return Err(ProofmanError::InvalidConfiguration(
                "Compressed vadcop proofs are not supported for snark proof generation".to_string(),
            ));
        }
        let proof = vadcop_proof.proof_with_publics_u64();
        timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
        let recursivef_proof =
            generate_recursivef_proof(&self.setup_recursivef, &proof, &self.aux_trace, output_dir_path)?;
        timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

        timer_start_info!(GENERATING_SNARK_PROOF);
        let (snark_proof_bytes, snark_publics_bytes) =
            generate_snark_proof(self.snark_prover, &self.setup_snark_path, recursivef_proof)?;

        let public_bytes = self.get_public_bytes(&proof[1..1 + proof[0] as usize])?;
        let snark_proof =
            SnarkProof::new(snark_proof_bytes, public_bytes, snark_publics_bytes, self.protocol.protocol_id());

        timer_stop_and_log_info!(GENERATING_SNARK_PROOF);

        let proofs_dir = output_dir_path.join("snark_proof");
        create_dir_all(&proofs_dir)?;

        snark_proof
            .save(&proofs_dir)
            .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to save SNARK proof: {}", e)))?;

        Ok(snark_proof)
    }

    fn get_public_bytes(&self, vadcop_public_inputs: &[u64]) -> ProofmanResult<Vec<u8>> {
        if vadcop_public_inputs.len() != self.publics_info.n_publics {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "Number of vadcop public inputs ({}) does not match expected number of publics ({})",
                vadcop_public_inputs.len(),
                self.publics_info.n_publics
            )));
        }

        let mut public_bytes = vec![];
        let mut index = 0;
        for public_def in &self.publics_info.definitions {
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
}

pub fn generate_and_verify_recursivef<F: PrimeField64>(
    proving_key_path: &Path,
    vadcop_proof: &VadcopFinalProof,
    output_dir_path: &Path,
    verbose_mode: VerboseMode,
) -> ProofmanResult<bool> {
    initialize_logger(verbose_mode, None);

    if vadcop_proof.compressed {
        return Err(ProofmanError::InvalidConfiguration(
            "Compressed vadcop proofs are not supported for snark proof generation".to_string(),
        ));
    }
    let proof = vadcop_proof.proof_with_publics_u64();

    timer_start_info!(LOADING_RECURSIVE_F_SETUP);

    let setup_recursivef_path =
        PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "recursivef", "recursivef"));

    let setup_recursivef = Setup::new(
        &setup_recursivef_path,
        0,
        0,
        &GlobalInfoAir::new("RecursiveF".to_string()),
        &ProofType::RecursiveF,
        false,
        false,
        None,
    );

    setup_recursivef.set_circom_circuit()?;
    setup_recursivef.set_exec_file_data()?;

    calculate_fixed_tree(&setup_recursivef);

    setup_recursivef.load_const_pols();
    setup_recursivef.load_const_pols_tree();

    let aux_trace = if cfg!(feature = "gpu") {
        Arc::new(Vec::new())
    } else {
        Arc::new(create_buffer_fast(setup_recursivef.prover_buffer_size as usize))
    };

    timer_stop_and_log_info!(LOADING_RECURSIVE_F_SETUP);

    timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
    let recursivef_proof = generate_recursivef_proof(&setup_recursivef, &proof, &aux_trace, output_dir_path)?;
    timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

    timer_start_info!(VERIFY_RECURSIVE_F_PROOF);
    let publics: Vec<F> = proof[1..1 + proof[0] as usize].iter().map(|&x| F::from_u64(x)).collect();
    let is_valid = verify_proof_bn128(recursivef_proof, &setup_recursivef, Some(publics));
    timer_stop_and_log_info!(VERIFY_RECURSIVE_F_PROOF);

    let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));
    if setup_snark_path.parent().is_some_and(|p| p.exists()) {
        generate_witness_final_snark(recursivef_proof, &setup_snark_path)?;
    }
    Ok(is_valid)
}

pub fn verify_snark_proof(snark_proof: &SnarkProof, vkey_path: &Path) -> ProofmanResult<()> {
    let (proof_json_value, publics_json_value) = snark_proof
        .convert_to_json()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to convert SNARK proof to JSON: {}", e)))?;

    // Write JSON to temporary files
    let temp_dir = std::env::temp_dir();
    let proof_path = temp_dir.join("snark_proof.json");
    let publics_path = temp_dir.join("snark_publics.json");

    std::fs::write(&proof_path, serde_json::to_string_pretty(&proof_json_value).unwrap())
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to write proof file: {}", e)))?;
    std::fs::write(&publics_path, serde_json::to_string_pretty(&publics_json_value).unwrap())
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to write publics file: {}", e)))?;

    // Determine protocol
    let protocol = SnarkProtocol::from_protocol_id(snark_proof.protocol_id)?;

    // Check if snarkjs is installed
    if Command::new("which").arg("snarkjs").output().map(|o| !o.status.success()).unwrap_or(true) {
        tracing::error!("··· {}", "snarkjs is not installed or not in PATH".bright_red().bold());
        tracing::error!("··· Please install snarkjs: npm install -g snarkjs");
        return Err(ProofmanError::InvalidConfiguration(
            "snarkjs is not installed. Please run: npm install -g snarkjs".to_string(),
        ));
    }

    // Call snarkjs verify
    let output = Command::new("snarkjs")
        .arg(protocol.protocol_name())
        .arg("verify")
        .arg(vkey_path)
        .arg(&publics_path)
        .arg(&proof_path)
        .output()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to execute snarkjs: {}", e)))?;

    let _ = std::fs::remove_file(&proof_path);
    let _ = std::fs::remove_file(&publics_path);

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
