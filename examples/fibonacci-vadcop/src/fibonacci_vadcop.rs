use std::rc::Rc;

use common::{ExecutionCtx, ProofCtx, WitnessPilOut};
use wchelpers::WCLibrary;
use proofman::WCManager;

use crate::{FibonacciSquare, Module};
use crate::pilout::get_fibonacci_vadcop_pilout;

pub struct FibonacciVadcop {
    pub wcm: WCManager,
    pub fibonacci: Rc<FibonacciSquare>,
    pub module: Rc<Module>,
}

impl FibonacciVadcop {
    pub fn new() -> Self {
        let mut wcm = WCManager::new();

        let fibonacci = FibonacciSquare::new(&mut wcm);
        let module = Module::new(&mut wcm);

        FibonacciVadcop { wcm, fibonacci, module }
    }
}

impl WCLibrary for FibonacciVadcop {
    fn start_proof(&mut self, pctx: &mut ProofCtx, ectx: &mut ExecutionCtx) {
        self.wcm.start_proof(pctx, ectx);
    }

    fn end_proof(&mut self) {
        self.wcm.end_proof();
    }

    fn calculate_plan(&mut self, ectx: &mut ExecutionCtx) {
        self.wcm.calculate_plan(ectx);
    }

    fn initialize_air_instances(&mut self, pctx: &mut ProofCtx, ectx: &ExecutionCtx) {
        for id in ectx.owned_instances.iter() {
            pctx.air_instances.push((&ectx.instances[*id]).into());
        }
    }
    fn calculate_witness(&mut self, stage: u32, pctx: &mut ProofCtx, ectx: &ExecutionCtx) {
        self.wcm.calculate_witness(stage, pctx, ectx);
    }

    fn get_pilout(&self) -> WitnessPilOut {
        get_fibonacci_vadcop_pilout()
    }
}

#[no_mangle]
pub extern "Rust" fn init_library<'a>() -> Box<dyn WCLibrary> {
    env_logger::builder()
        .format_timestamp(None)
        .format_level(true)
        .format_target(false)
        .filter_level(log::LevelFilter::Trace)
        .init();

    Box::new(FibonacciVadcop::new())
}