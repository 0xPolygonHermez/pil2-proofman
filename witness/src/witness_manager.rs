use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::path::PathBuf;

use p3_field::Field;
use proofman_common::{ModeName, ProofCtx, SetupCtx, DebugInfo};
use crate::WitnessComponent;

use std::sync::atomic::{AtomicBool, Ordering};

pub struct WitnessManager<F: Field> {
    components: RwLock<Vec<Arc<dyn WitnessComponent<F>>>>,
    components_instance_ids: RwLock<Vec<Vec<usize>>>,
    components_std: RwLock<Vec<Arc<dyn WitnessComponent<F>>>>,
    pctx: Arc<ProofCtx<F>>,
    sctx: Arc<SetupCtx<F>>,
    public_inputs_path: RwLock<Option<PathBuf>>,
    input_data_path: RwLock<Option<PathBuf>>,
    init: AtomicBool,
}

impl<F: Field> WitnessManager<F> {
    const _MY_NAME: &'static str = "WCMnager";

    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) -> Self {
        WitnessManager {
            components: RwLock::new(Vec::new()),
            components_instance_ids: RwLock::new(Vec::new()),
            components_std: RwLock::new(Vec::new()),
            pctx,
            sctx,
            public_inputs_path: RwLock::new(None),
            input_data_path: RwLock::new(None),
            init: AtomicBool::new(false),
        }
    }

    pub fn set_init_witness(&self, init: bool) {
        self.init.store(init, Ordering::SeqCst);
    }

    pub fn is_init_witness(&self) -> bool {
        self.init.load(Ordering::SeqCst)
    }

    pub fn set_public_inputs_path(&self, path: Option<PathBuf>) {
        *self.public_inputs_path.write().unwrap() = path;
    }

    pub fn set_input_data_path(&self, path: Option<PathBuf>) {
        *self.input_data_path.write().unwrap() = path;
    }

    pub fn register_component(&self, component: Arc<dyn WitnessComponent<F>>) {
        self.components.write().unwrap().push(component);
        self.components_instance_ids.write().unwrap().push(Vec::new());
    }

    pub fn register_component_std(&self, component: Arc<dyn WitnessComponent<F>>) {
        self.components_std.write().unwrap().push(component);
    }

    pub fn gen_custom_commits_fixed(&self, check: bool) -> Result<(), Box<dyn std::error::Error>> {
        for component in self.components.read().unwrap().iter() {
            component.gen_custom_commits_fixed(self.pctx.clone(), self.sctx.clone(), check)?;
        }

        Ok(())
    }

    pub fn execute(&self) {
        for (idx, component) in self.components.read().unwrap().iter().enumerate() {
            let global_ids = component.execute(self.pctx.clone(), self.input_data_path.read().unwrap().clone());
            self.components_instance_ids.write().unwrap()[idx] = global_ids;
        }
        for component in self.components_std.read().unwrap().iter() {
            component.execute(self.pctx.clone(), self.input_data_path.read().unwrap().clone());
        }
    }

    pub fn debug(&self, instance_ids: &[usize], debug_info: &DebugInfo) {
        if debug_info.std_mode.name == ModeName::Debug || !debug_info.debug_instances.is_empty() {
            for (idx, component) in self.components.read().unwrap().iter().enumerate() {
                let ids_hash_set: HashSet<_> = instance_ids.iter().collect();

                let instance_ids_filtered: Vec<_> = self.components_instance_ids.read().unwrap()[idx]
                    .iter()
                    .filter(|id| ids_hash_set.contains(id))
                    .cloned()
                    .collect();

                if !instance_ids_filtered.is_empty() {
                    component.debug(self.pctx.clone(), self.sctx.clone(), &instance_ids_filtered);
                }
            }
        }
        if debug_info.std_mode.name == ModeName::Debug {
            for component in self.components_std.read().unwrap().iter() {
                component.debug(self.pctx.clone(), self.sctx.clone(), instance_ids);
            }
        }
    }

    pub fn calculate_witness(&self, stage: u32, instance_ids: &[usize], core_id: usize, n_cores: usize) {
        for (idx, component) in self.components.read().unwrap().iter().enumerate() {
            let ids_hash_set: HashSet<_> = instance_ids.iter().collect();

            let instance_ids_filtered: Vec<_> = self.components_instance_ids.read().unwrap()[idx]
                .iter()
                .filter(|id| {
                    ids_hash_set.contains(id)
                        && (self.pctx.dctx_is_my_instance(**id) || self.pctx.dctx_is_instance_all(**id))
                })
                .cloned()
                .collect();

            if !instance_ids_filtered.is_empty() {
                component.calculate_witness(
                    stage,
                    self.pctx.clone(),
                    self.sctx.clone(),
                    &instance_ids_filtered,
                    core_id,
                    n_cores,
                );
            }
        }

        for component in self.components_std.read().unwrap().iter() {
            component.calculate_witness(stage, self.pctx.clone(), self.sctx.clone(), instance_ids, core_id, n_cores);
        }
    }

    pub fn end(&self, debug_info: &DebugInfo) {
        for component in self.components.read().unwrap().iter() {
            component.end(self.pctx.clone(), debug_info);
        }
        for component in self.components_std.read().unwrap().iter() {
            component.end(self.pctx.clone(), debug_info);
        }
    }

    pub fn get_pctx(&self) -> Arc<ProofCtx<F>> {
        self.pctx.clone()
    }

    pub fn get_sctx(&self) -> Arc<SetupCtx<F>> {
        self.sctx.clone()
    }

    pub fn get_public_inputs_path(&self) -> Option<PathBuf> {
        self.public_inputs_path.read().unwrap().clone()
    }

    pub fn get_input_data_path(&self) -> Option<PathBuf> {
        self.input_data_path.read().unwrap().clone()
    }
}
