use std::sync::Arc;

use proofman_common::{ProofCtx, SetupCtx};

pub trait WitnessComponent<F: Clone>: Send + Sync {
    fn execute(&self, _pctx: Arc<ProofCtx<F>>) {}

    fn debug(&self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>) {}

    fn calculate_witness(&self, _stage: u32, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>) {}

    fn gen_custom_commits_fixed(
        &self,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _check: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}
