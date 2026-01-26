use crate::WitnessManager;
use fields::PrimeField64;
use proofman_common::{PackedInfo, ProofCtx, ProofmanResult, ProofmanError, VerboseMode};
use proofman_util::{timer_start_info, timer_stop_and_log_info};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// This is the type of the function that is used to load a witness library.
pub type WitnessLibInitFn<F> = fn(VerboseMode, Option<i32>) -> ProofmanResult<Box<dyn WitnessLibrary<F>>>;

pub trait WitnessLibrary<F: PrimeField64> {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()>;

    /// Returns the weight indicating the complexity of the witness computation.
    ///
    /// Used as a heuristic for estimating computational cost.
    fn get_witness_weight(&self, _pctx: &ProofCtx<F>, _global_id: usize) -> ProofmanResult<usize> {
        Ok(1)
    }

    fn get_packed_info(&self) -> HashMap<(usize, usize), PackedInfo> {
        HashMap::new()
    }
}

#[macro_export]
macro_rules! witness_library {
    ($lib_name:ident, $field_type:ty) => {
        // Define the struct
        pub struct $lib_name;

        // Define the init_library function
        #[no_mangle]
        pub extern "Rust" fn init_library(
            verbose_mode: proofman_common::VerboseMode,
            rank: Option<i32>,
        ) -> proofman_common::ProofmanResult<Box<dyn witness::WitnessLibrary<$field_type>>> {
            proofman_common::initialize_logger(verbose_mode, rank);

            Ok(Box::new($lib_name))
        }
    };
}

pub fn load_witness_library<F: PrimeField64>(
    witness_lib_path: &Path,
    public_inputs_path: Option<PathBuf>,
    wcm: &WitnessManager<F>,
    verbose_mode: VerboseMode,
    rank: Option<i32>,
) -> ProofmanResult<Box<dyn WitnessLibrary<F>>> {
    if let Some(ref publics_path) = public_inputs_path {
        if !publics_path.exists() {
            return Err(ProofmanError::InvalidParameters(format!(
                "Public inputs file not found at path: {publics_path:?}"
            )));
        }
        wcm.set_public_inputs_path(public_inputs_path);
    }

    timer_start_info!(CREATE_WITNESS_LIB);
    let library = unsafe { libloading::Library::new(witness_lib_path)? };
    let witness_library: libloading::Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
    let mut witness_lib = witness_library(verbose_mode, rank)?;
    timer_stop_and_log_info!(CREATE_WITNESS_LIB);

    witness_lib.register_witness(wcm)?;

    Ok(witness_lib)
}
