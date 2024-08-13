use std::{collections::HashMap, sync::Arc};

use log::info;

use proofman_common::{ExecutionCtx, ProofCtx};
use witness_helpers::WitnessComponent;

use crate::{DefaultPlanner, Planner};

pub struct WitnessManager<F> {
    components: Vec<Arc<dyn WitnessComponent<F>>>,
    airs: HashMap<usize, Arc<dyn WitnessComponent<F>>>,
    planner: Box<dyn Planner<F>>,
}

impl<F> Default for WitnessManager<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> WitnessManager<F> {
    const MY_NAME: &'static str = "WCMnager";

    pub fn new() -> Self {
        WitnessManager { components: Vec::new(), airs: HashMap::new(), planner: Box::new(DefaultPlanner) }
    }

    pub fn register_component(&mut self, component: Arc<dyn WitnessComponent<F>>, air_ids: Option<&[usize]>) {
        if let Some(air_ids) = air_ids {
            self.register_airs(air_ids, component.clone());
        }

        self.components.push(component);
    }

    pub fn register_airs(&mut self, air_ids: &[usize], air: Arc<dyn WitnessComponent<F>>) {
        for air_id in air_ids.iter() {
            self.register_air(*air_id, air.clone());
        }
    }

    pub fn register_air(&mut self, air_id: usize, air: Arc<dyn WitnessComponent<F>>) {
        if self.airs.contains_key(&air_id) {
            panic!("{}: Air ID {} already registered", Self::MY_NAME, air_id);
        }

        self.airs.insert(air_id, air);
    }

    pub fn set_planner(&mut self, planner: Box<dyn Planner<F>>) {
        self.planner = planner;
    }

    pub fn start_proof(&mut self, pctx: &mut ProofCtx<F>, ectx: &mut ExecutionCtx) {
        for component in self.components.iter() {
            component.start_proof(pctx, ectx);
        }
    }

    pub fn end_proof(&mut self) {
        for component in self.components.iter() {
            component.end_proof();
        }
    }

    pub fn start_execute(&self, pctx: &mut ProofCtx<F>, ectx: &ExecutionCtx) {
        for component in self.components.iter() {
            component.start_execute(pctx, ectx);
        }
    }

    pub fn end_execute(&self, pctx: &mut ProofCtx<F>, ectx: &ExecutionCtx) {
        for component in self.components.iter() {
            component.end_execute(pctx, ectx);
        }
    }

    pub fn calculate_plan(&self, ectx: &mut ExecutionCtx) {
        self.planner.calculate_plan(&self.components, ectx);
    }

    pub fn calculate_witness(&self, stage: u32, pctx: &mut ProofCtx<F>, ectx: &ExecutionCtx) {
        info!("{}: Calculating witness (stage {})", Self::MY_NAME, stage);
        for air_instance_ctx in ectx.instances.iter().rev() {
            let component = self.airs.get(&air_instance_ctx.air_group_id).unwrap();
            component.calculate_witness(stage, air_instance_ctx, pctx, ectx);
        }
    }
}
