use std::sync::Arc;

use proofman_common::SetupCtx;

pub trait Decider<F> {
    fn decide(&self, sctx: Arc<SetupCtx>);
}
