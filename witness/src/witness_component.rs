use std::sync::Arc;

use proofman_common::{ProofCtx, SetupCtx};

pub trait WitnessComponent<F: Clone>: Send + Sync {
    fn execute(&self, _pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        Vec::new()
    }

    fn debug(&self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, _instance_ids: &[usize]) {}

    fn calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
    ) {
    }

    fn end(&self, _pctx: Arc<ProofCtx<F>>) {}
}
