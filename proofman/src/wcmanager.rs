use std::rc::Rc;

use log::info;

use common::{ExecutionCtx, ProofCtx};
use wchelpers::{WCComponent, WCExecutor};

pub struct WCManager {
    components: Vec<Rc<dyn WCComponent>>,
    executors: Vec<Rc<dyn WCExecutor>>,
}

impl WCManager {
    const MY_NAME: &'static str = "WCManager";

    pub fn new() -> Self {
        WCManager { components: Vec::new(), executors: Vec::new() }
    }

    pub fn register_component(&mut self, mem_sm: Rc<dyn WCComponent>) {
        self.components.push(mem_sm);
    }

    pub fn register_executor(&mut self, executor: Rc<dyn WCExecutor>) {
        self.executors.push(executor);
    }

    pub fn start_proof(&mut self, pctx: &mut ProofCtx, ectx: &mut ExecutionCtx) {
        println!("{}: Starting proof", Self::MY_NAME);
        for component in self.components.iter() {
            component.start_proof(pctx, ectx);
        }

        Self::execute(self, pctx, ectx);
    }

    pub fn end_proof(&mut self) {
        println!("{}: Ending proof", Self::MY_NAME);
        for component in self.components.iter() {
            component.end_proof();
        }
    }

    pub fn calculate_plan(&mut self, ectx: &mut ExecutionCtx) {
        println!("{}: Calculating plan", Self::MY_NAME);
        ectx.owned_instances = (0..ectx.instances.len()).collect();
    }

    pub fn calculate_witness(&mut self, stage: u32, pctx: &mut ProofCtx, ectx: &ExecutionCtx) {
        info!("{}: Calculating witness (stage {})", Self::MY_NAME, stage);
        for component in self.components.iter() {
            component.calculate_witness(stage, pctx, ectx);
        }
    }

    fn execute(&self, pctx: &mut ProofCtx, ectx: &mut ExecutionCtx) {
        println!("{}: Executing", Self::MY_NAME);
        for executor in self.executors.iter() {
            executor.execute(pctx, ectx);
        }
    }
}