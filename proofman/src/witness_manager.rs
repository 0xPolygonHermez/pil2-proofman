use std::sync::{Arc, RwLock};

use proofman_common::{ProofCtx, SetupCtx};
use proofman_util::{timer_start_debug, timer_stop_and_log_debug};
use crate::WitnessComponent;

pub struct WitnessManager<F> {
    components: RwLock<Vec<Arc<dyn WitnessComponent<F>>>>,
    pctx: Arc<ProofCtx<F>>,
    sctx: Arc<SetupCtx>,
}

impl<F> WitnessManager<F> {
    const MY_NAME: &'static str = "WCMnager";

    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) -> Self {
        WitnessManager { components: RwLock::new(Vec::new()), pctx, sctx }
    }

    pub fn register_component(&self, component: Arc<dyn WitnessComponent<F>>) {
        self.components.write().unwrap().push(component);
    }

    pub fn start_proof(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) {
        for component in self.components.read().unwrap().iter() {
            component.start_proof(pctx.clone(), sctx.clone());
        }
    }

    pub fn end_proof(&self) {
        for component in self.components.read().unwrap().iter() {
            component.end_proof();
        }
    }

    pub fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) {
        log::info!(
            "{}: Calculating witness for stage {} / {}",
            Self::MY_NAME,
            stage,
            pctx.global_info.n_challenges.len()
        );

        timer_start_debug!(CALCULATING_WITNESS);

        // Call one time all unused components
        for component in self.components.read().unwrap().iter() {
            component.calculate_witness(stage, pctx.clone(), sctx.clone());
        }

        timer_stop_and_log_debug!(CALCULATING_WITNESS);
    }

    pub fn get_pctx(&self) -> Arc<ProofCtx<F>> {
        self.pctx.clone()
    }

    pub fn get_sctx(&self) -> Arc<SetupCtx> {
        self.sctx.clone()
    }
}
