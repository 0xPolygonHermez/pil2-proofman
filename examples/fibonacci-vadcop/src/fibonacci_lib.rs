use std::sync::Arc;

use proofman_common::{ExecutionCtx, ProofCtx, WitnessPilout};
use p3_field::AbstractField;
use p3_goldilocks::Goldilocks;
use witness_helpers::{WitnessComponent, WitnessLibrary};
use proofman::WitnessManager;

use std::error::Error;
use std::path::PathBuf;

use crate::MODULE_SUBPROOF_ID;

use crate::{FibonacciSquare, Pilout, Module /* , RangeCheck*/};

pub struct FibonacciVadcop<F> {
    pub wcm: WitnessManager<F>,
    pub fibonacci: Arc<FibonacciSquare>,
    pub module: Arc<Module>,
    //pub range_check: Arc<RangeCheck>,
}

impl<F: AbstractField + Copy> FibonacciVadcop<F> {
    pub fn new() -> Self {
        let mut wcm = WitnessManager::new();

        /*let range_check = RangeCheck::new_no_register(&mut wcm);*/
        let module = Module::new_no_register(&mut wcm /* , &range_check*/);
        let fibonacci = FibonacciSquare::new(&mut wcm, &module);
        // Register the module component after the fibonacci component
        wcm.register_component(Arc::clone(&module) as Arc<dyn WitnessComponent<F>>, Some(MODULE_SUBPROOF_ID));
        //wcm.register_component(Arc::clone(&range_check) as Arc<dyn WitnessComponent<F>>, Some(U_8_AIR_SUBPROOF_ID));

        FibonacciVadcop { wcm, fibonacci, module /* , range_check*/ }
    }
}

impl<F: AbstractField + Copy> WitnessLibrary<F> for FibonacciVadcop<F> {
    fn start_proof(&mut self, pctx: &mut ProofCtx<F>, ectx: &mut ExecutionCtx) {
        pctx.public_inputs =
            vec![25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0]; // TODO: NOT SHOULD BE HARDCODED!
        self.wcm.start_proof(pctx, ectx);
    }

    fn end_proof(&mut self) {
        self.wcm.end_proof();
    }

    fn execute(&self, pctx: &mut ProofCtx<F>, ectx: &mut ExecutionCtx) {
        self.fibonacci.execute(pctx, ectx);
    }

    fn calculate_plan(&mut self, ectx: &mut ExecutionCtx) {
        self.wcm.calculate_plan(ectx);
    }

    fn calculate_witness(&mut self, stage: u32, pctx: &mut ProofCtx<F>, ectx: &ExecutionCtx) {
        self.wcm.calculate_witness(stage, pctx, ectx);
    }

    fn pilout(&self) -> WitnessPilout {
        Pilout::pilout()
    }
}

#[no_mangle]
pub extern "Rust" fn init_library(
    _rom_path: Option<PathBuf>,
    _public_inputs_path: PathBuf,
    _proving_key_path: PathBuf,
) -> Result<Box<dyn WitnessLibrary<Goldilocks>>, Box<dyn Error>> {
    env_logger::builder()
        .format_timestamp(None)
        .format_level(true)
        .format_target(false)
        .filter_level(log::LevelFilter::Trace)
        .init();
    let fibonacci_witness = FibonacciVadcop::new();
    Ok(Box::new(fibonacci_witness))
}
