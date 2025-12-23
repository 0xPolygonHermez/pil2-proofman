use crate::{generate_recursivef_proof, generate_snark_proof};
use proofman_common::{
    GlobalInfoAir, ProofmanError, ProofmanResult, ProofType, PublicsInfo, Setup, calculate_fixed_tree, VerboseMode,
    initialize_logger,
};
use proofman_util::{timer_start_info, timer_stop_and_log_info, create_buffer_fast};
use fields::PrimeField64;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs::File;
use std::io::Write;
use std::ffi::c_void;
use proofman_starks_lib_c::{init_final_snark_prover_c, free_final_snark_prover_c};

pub struct SnarkWrapper<F: PrimeField64> {
    pub setup_snark_path: PathBuf,
    pub setup_recursivef: Setup<F>,
    pub aux_trace: Arc<Vec<F>>,
    pub snark_prover: *mut c_void,
    pub publics_info: PublicsInfo,
}

#[derive(Debug)]
pub struct SnarkProof {
    pub proof_bytes: Vec<u8>,
    pub public_bytes: Vec<u8>,
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
        timer_stop_and_log_info!(INITIALIZING_FINAL_SNARK_PROVER);

        let publics_info = PublicsInfo::from_folder(proving_key_path)?;

        Ok(Self { aux_trace, setup_recursivef, setup_snark_path, snark_prover, publics_info })
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_final_snark_proof(
        &self,
        vadcop_proof: &[u64],
        output_dir_path: &Path,
        save_json: bool,
    ) -> ProofmanResult<SnarkProof> {
        timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
        let recursivef_proof =
            generate_recursivef_proof(&self.setup_recursivef, vadcop_proof, &self.aux_trace, output_dir_path)?;
        timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

        timer_start_info!(GENERATING_SNARK_PROOF);
        let snark_proof_bytes = generate_snark_proof(
            self.snark_prover,
            &self.setup_snark_path,
            recursivef_proof,
            output_dir_path,
            save_json,
        )?;

        timer_stop_and_log_info!(GENERATING_SNARK_PROOF);

        let public_bytes = self.get_public_bytes(&vadcop_proof[1..1 + vadcop_proof[0] as usize])?;
        let snark_proof = SnarkProof { proof_bytes: snark_proof_bytes, public_bytes };

        let proofs_file_path = output_dir_path.join("proofs/final_snark_proof.bin");
        let mut proof_file = File::create(&proofs_file_path)?;
        proof_file.write_all(&snark_proof.proof_bytes)?;
        proof_file.flush()?;

        let publics_file_path = output_dir_path.join("proofs/final_snark_publics.bin");
        let mut publics_file = File::create(&publics_file_path)?;
        publics_file.write_all(&snark_proof.public_bytes)?;
        publics_file.flush()?;

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
