use std::sync::Arc;

use proofman_common::{ProofCtx, SetupCtx};

pub trait WitnessComponent<F: Clone>: Send + Sync {
    fn start_proof(&self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>) {}

    fn execute(&self, _pctx: Arc<ProofCtx<F>>) {}

    fn debug(&self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>) {}

    fn calculate_witness(&self, _stage: u32, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>) {}

    fn calculate_witness_airgroup(&self, _stage: u32, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>) {}

    fn calculate_witness_proof(&self, _stage: u32, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>) {}

    fn gen_custom_commits_fixed(
        &self,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _check: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}
