use std::sync::Arc;

use common::{ExecutionCtx, ProofCtx, WCPilout};
use p3_field::AbstractField;
use p3_goldilocks::Goldilocks;
use wchelpers::{WCComponent, WCLibrary};
use proofman::WCManager;
use common::Prover;
use crate::MODULE_SUBPROOF_ID;
//use crate::U_8_AIR_SUBPROOF_ID;

use crate::{FibonacciSquare, FibonacciVadcopPilout, Module/* , RangeCheck*/};

pub struct FibonacciVadcop<F> {
    pub wcm: WCManager<F>,
    pub fibonacci: Arc<FibonacciSquare>,
    pub module: Arc<Module>,
    //pub range_check: Arc<RangeCheck>,
}

impl<F: AbstractField> FibonacciVadcop<F> {
    pub fn new() -> Self {
        let mut wcm = WCManager::new();
        /*let range_check = RangeCheck::new_no_register(&mut wcm);*/
        let module = Module::new_no_register(&mut wcm/* , &range_check*/);
        let fibonacci = FibonacciSquare::new(&mut wcm, &module);
        // Register the module component after the fibonacci component
        wcm.register_component(Arc::clone(&module) as Arc<dyn WCComponent<F>>, Some(MODULE_SUBPROOF_ID));
        //wcm.register_component(Arc::clone(&range_check) as Arc<dyn WCComponent<F>>, Some(U_8_AIR_SUBPROOF_ID));
        FibonacciVadcop { wcm, fibonacci, module/* , range_check*/ }
    }
}

impl<F> WCLibrary<F> for FibonacciVadcop<F> {
    fn start_proof(&mut self, pctx: &mut ProofCtx<F>, ectx: &mut ExecutionCtx) {
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

    fn calculate_witness(
        &mut self,
        stage: u32,
        pctx: &mut ProofCtx<F>,
        ectx: &ExecutionCtx,
        provers: &Vec<Box<dyn Prover<F>>>,
    ) {
        self.wcm.calculate_witness(stage, pctx, ectx, provers);
    }

    fn pilout(&self) -> WCPilout {
        FibonacciVadcopPilout::pilout()
    }
}

#[no_mangle]
pub extern "Rust" fn init_library<'a>() -> Box<dyn WCLibrary<Goldilocks>> {
    env_logger::builder()
        .format_timestamp(None)
        .format_level(true)
        .format_target(false)
        .filter_level(log::LevelFilter::Trace)
        .init();

    Box::new(FibonacciVadcop::new())
}
