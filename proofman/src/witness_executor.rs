use proofman_common::ProofCtx;

pub trait WitnessExecutor<F> {
    fn execute(&self, pctx: &mut ProofCtx<F>);
}
