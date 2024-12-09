use std::sync::Arc;

use proofman_common::{initialize_logger, load_from_json, ExecutionCtx, ProofCtx, SetupCtx, VerboseMode};
use proofman::{WitnessLibrary, WitnessManager};
use pil_std_lib::Std;
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;

use std::error::Error;
use std::path::PathBuf;

use crate::{BuildPublics, BuildPublicValues, FibonacciSquare, Module};

pub struct FibonacciWitness<F: PrimeField64> {
    public_inputs_path: Option<PathBuf>,
    wcm: Option<Arc<WitnessManager<F>>>,
    fibonacci: Option<Arc<FibonacciSquare<F>>>,
    module: Option<Arc<Module<F>>>,
    std_lib: Option<Arc<Std<F>>>,
}

impl<F: PrimeField64> FibonacciWitness<F> {
    pub fn new(public_inputs_path: Option<PathBuf>) -> Self {
        Self { public_inputs_path, wcm: None, fibonacci: None, module: None, std_lib: None }
    }
}

impl<F: PrimeField64> WitnessLibrary<F> for FibonacciWitness<F> {
    fn start_proof(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let wcm = Arc::new(WitnessManager::new(pctx.clone(), ectx.clone(), sctx.clone()));

        let std_lib = Std::new(wcm.clone());
        let module = Module::new(wcm.clone(), std_lib.clone());
        let fibonacci = FibonacciSquare::new(wcm.clone(), module.clone());

        self.wcm = Some(wcm.clone());
        self.fibonacci = Some(fibonacci);
        self.module = Some(module);
        self.std_lib = Some(std_lib);

        let public_inputs: BuildPublics = load_from_json(&self.public_inputs_path);

        let mut publics = BuildPublicValues::from_vec_guard(pctx.get_publics());

        publics.module = F::from_canonical_u64(public_inputs.module);
        publics.in1 = F::from_canonical_u64(public_inputs.in1);
        publics.in2 = F::from_canonical_u64(public_inputs.in2);

        wcm.start_proof(pctx, ectx, sctx);
    }

    fn end_proof(&mut self) {
        self.wcm.as_ref().unwrap().end_proof();
    }

    fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.fibonacci.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.module.as_ref().unwrap().execute(pctx, ectx, sctx);
    }

    fn calculate_witness(&mut self, stage: u32, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.wcm.as_ref().unwrap().calculate_witness(stage, pctx, ectx, sctx);
    }
}

#[no_mangle]
pub extern "Rust" fn init_library(
    _: Option<PathBuf>,
    public_inputs_path: Option<PathBuf>,
    verbose_mode: VerboseMode,
) -> Result<Box<dyn WitnessLibrary<Goldilocks>>, Box<dyn Error>> {
    initialize_logger(verbose_mode);

    let fibonacci_witness = FibonacciWitness::new(public_inputs_path);
    Ok(Box::new(fibonacci_witness))
}
