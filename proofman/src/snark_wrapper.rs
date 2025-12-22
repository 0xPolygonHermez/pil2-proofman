use crate::{generate_recursivef_proof, generate_snark_proof};
use proofman_common::{
    GlobalInfoAir, ProofmanResult, ProofType, Setup, calculate_fixed_tree, VerboseMode, initialize_logger,
};
use proofman_util::{timer_start_info, timer_stop_and_log_info, create_buffer_fast};
use fields::PrimeField64;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::ffi::c_void;
use proofman_starks_lib_c::{init_final_snark_prover_c, free_final_snark_prover_c};

pub struct SnarkWrapper<F: PrimeField64> {
    pub setup_snark_path: PathBuf,
    pub setup_recursivef: Setup<F>,
    pub aux_trace: Arc<Vec<F>>,
    pub snark_prover: *mut c_void,
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

        Ok(Self { aux_trace, setup_recursivef, setup_snark_path, snark_prover })
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_final_snark_proof(&self, vadcop_proof: &[u64], output_dir_path: &Path) -> ProofmanResult<()> {
        timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
        let recursivef_proof =
            generate_recursivef_proof(&self.setup_recursivef, vadcop_proof, &self.aux_trace, output_dir_path)?;
        timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

        timer_start_info!(GENERATING_FFLONK_SNARK_PROOF);
        let _proof_file =
            generate_snark_proof(self.snark_prover, &self.setup_snark_path, recursivef_proof, output_dir_path);
        timer_stop_and_log_info!(GENERATING_FFLONK_SNARK_PROOF);

        Ok(())
    }
}
