use std::error::Error;

use crate::WitnessManager;
use fields::PrimeField64;
use proofman_common::{PackedInfo, ProofCtx, VerboseMode};
use std::collections::HashMap;

/// This is the type of the function that is used to load a witness library.
pub type WitnessLibInitFn<F> =
    fn(VerboseMode, Option<i32>) -> Result<Box<dyn WitnessLibrary<F>>, Box<dyn Error + Send + Sync>>;

pub trait WitnessLibrary<F: PrimeField64> {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Returns the weight indicating the complexity of the witness computation.
    ///
    /// Used as a heuristic for estimating computational cost.
    fn get_witness_weight(
        &self,
        _pctx: &ProofCtx<F>,
        _global_id: usize,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
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
        ) -> Result<Box<dyn witness::WitnessLibrary<$field_type>>, Box<dyn std::error::Error + Send + Sync>> {
            proofman_common::initialize_logger(verbose_mode, rank);

            Ok(Box::new($lib_name))
        }
    };
}
