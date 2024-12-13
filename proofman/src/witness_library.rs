use std::{error::Error, path::PathBuf, sync::Arc};

use proofman_common::{ProofCtx, SetupCtx, VerboseMode};

/// This is the type of the function that is used to load a witness library.
pub type WitnessLibInitFn<F> =
    fn(Option<PathBuf>, Option<PathBuf>, VerboseMode) -> Result<Box<dyn WitnessLibrary<F>>, Box<dyn Error>>;

pub trait WitnessLibrary<F> {
    fn start_proof(&mut self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>);

    fn end_proof(&mut self);

    fn execute(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>);

    fn calculate_witness(&mut self, stage: u32, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>);

    fn debug(&mut self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx>) {}
}
