use std::{any::Any};

use crate::WitnessManager;
use fields::PrimeField64;
use proofman_common::{ProofCtx, VerboseMode, ProofmanResult};

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

    fn get_execution_result(&self) -> Option<Box<dyn Any>> {
        None
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
