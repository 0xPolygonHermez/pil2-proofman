use std::{error::Error, path::PathBuf, sync::Arc};

use pil_std_lib::Std;
use proofman::{WitnessLibrary, WitnessManager};
use proofman_common::{initialize_logger, ExecutionCtx, ProofCtx, SetupCtx, VerboseMode, WitnessPilout};

use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{DirectUpdateProdLocal, DirectUpdateProdGlobal, DirectUpdateSumLocal, DirectUpdateSumGlobal, Pilout};

pub struct DirectUpdateWitness<F: PrimeField> {
    pub wcm: Option<Arc<WitnessManager<F>>>,
    pub direct_update_prod_local: Option<Arc<DirectUpdateProdLocal<F>>>,
    pub direct_update_prod_global: Option<Arc<DirectUpdateProdGlobal<F>>>,
    pub direct_update_sum_local: Option<Arc<DirectUpdateSumLocal<F>>>,
    pub direct_update_sum_global: Option<Arc<DirectUpdateSumGlobal<F>>>,
    pub std_lib: Option<Arc<Std<F>>>,
}

impl<F: PrimeField> Default for DirectUpdateWitness<F>
where
    Standard: Distribution<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField> DirectUpdateWitness<F>
where
    Standard: Distribution<F>,
{
    pub fn new() -> Self {
        DirectUpdateWitness {
            wcm: None,
            direct_update_prod_local: None,
            direct_update_prod_global: None,
            direct_update_sum_local: None,
            direct_update_sum_global: None,
            std_lib: None,
        }
    }

    pub fn initialize(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let wcm = Arc::new(WitnessManager::new(pctx, ectx, sctx));

        let std_lib = Std::new(wcm.clone());
        let direct_update_prod_local = DirectUpdateProdLocal::new(wcm.clone());
        let direct_update_prod_global = DirectUpdateProdGlobal::new(wcm.clone());
        let direct_update_sum_local = DirectUpdateSumLocal::new(wcm.clone());
        let direct_update_sum_global = DirectUpdateSumGlobal::new(wcm.clone());

        self.wcm = Some(wcm);
        self.direct_update_prod_local = Some(direct_update_prod_local);
        self.direct_update_prod_global = Some(direct_update_prod_global);
        self.direct_update_sum_local = Some(direct_update_sum_local);
        self.direct_update_sum_global = Some(direct_update_sum_global);
        self.std_lib = Some(std_lib);
    }
}

impl<F: PrimeField> WitnessLibrary<F> for DirectUpdateWitness<F>
where
    Standard: Distribution<F>,
{
    fn start_proof(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.initialize(pctx.clone(), ectx.clone(), sctx.clone());

        self.wcm.as_ref().unwrap().start_proof(pctx, ectx, sctx);
    }

    fn end_proof(&mut self) {
        self.wcm.as_ref().unwrap().end_proof();
    }

    fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        // Execute those components that need to be executed
        self.direct_update_prod_local.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.direct_update_prod_global.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.direct_update_sum_local.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.direct_update_sum_global.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
    }

    fn calculate_witness(&mut self, stage: u32, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.wcm.as_ref().unwrap().calculate_witness(stage, pctx, ectx, sctx);
    }

    fn pilout(&self) -> WitnessPilout {
        Pilout::pilout()
    }
}

#[no_mangle]
pub extern "Rust" fn init_library(
    _: Option<PathBuf>,
    _: Option<PathBuf>,
    verbose_mode: VerboseMode,
) -> Result<Box<dyn WitnessLibrary<Goldilocks>>, Box<dyn Error>> {
    initialize_logger(verbose_mode);
    let direct_update_witness = DirectUpdateWitness::new();
    Ok(Box::new(direct_update_witness))
}
