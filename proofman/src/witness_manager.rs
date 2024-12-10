use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use proofman_common::{ExecutionCtx, ProofCtx, SetupCtx};
use proofman_util::{timer_start_debug, timer_stop_and_log_debug};
use crate::WitnessComponent;

type AirGroupId = usize;
type AirId = usize;

pub struct WitnessManager<F> {
    components: RwLock<Vec<Arc<dyn WitnessComponent<F>>>>,
    airs: RwLock<HashMap<(AirGroupId, AirId), usize>>, // First usize is the air_id, second usize is the index of the component in the components vector

    pctx: Arc<ProofCtx<F>>,
    ectx: Arc<ExecutionCtx>,
    sctx: Arc<SetupCtx>,
}

impl<F> WitnessManager<F> {
    const MY_NAME: &'static str = "WCMnager";

    pub fn new(pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) -> Self {
        WitnessManager { components: RwLock::new(Vec::new()), airs: RwLock::new(HashMap::new()), pctx, ectx, sctx }
    }

    pub fn register_component(&self, component: Arc<dyn WitnessComponent<F>>, airgroup_id: AirGroupId, air_id: AirId) {
        self.components.write().unwrap().push(component);

        let idx = self.components.write().unwrap().len() - 1;

        self.register_air(airgroup_id, air_id, idx);
    }

    pub fn register_proxy_component(&self, component: Arc<dyn WitnessComponent<F>>) {
        self.components.write().unwrap().push(component);
    }

    pub fn register_airs(&self, airgroup_id: AirGroupId, air_ids: &[AirId], component_idx: usize) {
        for air_id in air_ids.iter() {
            self.register_air(airgroup_id, *air_id, component_idx);
        }
    }

    pub fn register_air(&self, airgroup_id: AirGroupId, air_id: AirId, component_idx: usize) {
        if self.airs.read().unwrap().contains_key(&(airgroup_id, air_id)) {
            panic!("{}: AirGroup ID + Air ID ({},{}) already registered", Self::MY_NAME, airgroup_id, air_id);
        }

        self.airs.write().unwrap().insert((airgroup_id, air_id), component_idx);
    }

    pub fn start_proof(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        for component in self.components.read().unwrap().iter() {
            component.start_proof(pctx.clone(), ectx.clone(), sctx.clone());
        }
    }

    pub fn end_proof(&self) {
        for component in self.components.read().unwrap().iter() {
            component.end_proof();
        }
    }

    pub fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        log::info!(
            "{}: Calculating witness for stage {} / {}",
            Self::MY_NAME,
            stage,
            pctx.global_info.n_challenges.len()
        );

        timer_start_debug!(CALCULATING_WITNESS);

        let air_instances = pctx.air_instance_repo.air_instances.read().unwrap();

        let mut components = HashMap::new();
        let airs = self.airs.read().unwrap();
        for (air_instance_id, air_instance) in air_instances.iter().enumerate() {
            if let Some(component) = airs.get(&(air_instance.airgroup_id, air_instance.air_id)) {
                components
                    .entry((air_instance.airgroup_id, air_instance.air_id))
                    .or_insert_with(Vec::new)
                    .push((component, air_instance_id));
            }
        }
        drop(air_instances);

        // Call all used components
        let mut used_components = Vec::new();
        let self_components = self.components.read().unwrap();
        for component_group in components.values() {
            for (component_idx, id) in component_group.iter() {
                let component = &self_components[**component_idx];
                component.calculate_witness(stage, Some(*id), pctx.clone(), ectx.clone(), sctx.clone());
                used_components.push(**component_idx);
            }
        }

        // Call one time all unused components
        for component_idx in 0..self.components.read().unwrap().len() {
            if !used_components.contains(&component_idx) {
                self_components[component_idx].calculate_witness(stage, None, pctx.clone(), ectx.clone(), sctx.clone());
            }
        }

        timer_stop_and_log_debug!(CALCULATING_WITNESS);
    }

    pub fn get_pctx(&self) -> Arc<ProofCtx<F>> {
        self.pctx.clone()
    }

    pub fn get_ectx(&self) -> Arc<ExecutionCtx> {
        self.ectx.clone()
    }

    pub fn get_sctx(&self) -> Arc<SetupCtx> {
        self.sctx.clone()
    }
}
