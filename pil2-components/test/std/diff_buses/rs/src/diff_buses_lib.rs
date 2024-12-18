use std::{error::Error, path::PathBuf, sync::Arc};

use pil_std_lib::Std;
use proofman::{WitnessLibrary, WitnessManager};
use proofman_common::{initialize_logger, ExecutionCtx, ProofCtx, SetupCtx, VerboseMode, WitnessPilout};

use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{Pilout, ProdBus, BothBuses, SumBus};

pub struct DiffBusesWitness<F: PrimeField> {
    pub wcm: Option<Arc<WitnessManager<F>>>,
    pub prod_bus: Option<Arc<ProdBus<F>>>,
    pub sum_bus: Option<Arc<SumBus<F>>>,
    pub both_buses: Option<Arc<BothBuses<F>>>,
    pub std_lib: Option<Arc<Std<F>>>,
}

impl<F: PrimeField> Default for DiffBusesWitness<F>
where
    Standard: Distribution<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField> DiffBusesWitness<F>
where
    Standard: Distribution<F>,
{
    pub fn new() -> Self {
        DiffBusesWitness { wcm: None, prod_bus: None, sum_bus: None, both_buses: None, std_lib: None }
    }

    pub fn initialize(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let wcm = Arc::new(WitnessManager::new(pctx, ectx, sctx));

        let std_lib = Std::new(wcm.clone());
        let prod_bus = ProdBus::new(wcm.clone());
        let sum_bus = SumBus::new(wcm.clone());
        let both_buses = BothBuses::new(wcm.clone());

        self.wcm = Some(wcm);
        self.prod_bus = Some(prod_bus);
        self.sum_bus = Some(sum_bus);
        self.both_buses = Some(both_buses);
        self.std_lib = Some(std_lib);
    }
}

impl<F: PrimeField> WitnessLibrary<F> for DiffBusesWitness<F>
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
        self.prod_bus.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.sum_bus.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.both_buses.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
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
    let diff_buses_witness = DiffBusesWitness::new();
    Ok(Box::new(diff_buses_witness))
}
