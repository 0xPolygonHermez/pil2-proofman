use std::collections::HashSet;
use std::sync::{Arc, RwLock, Mutex};
use std::path::PathBuf;

use fields::PrimeField64;
use proofman_common::{BufferPool, DebugInfo, ModeName, ProofCtx, SetupCtx};
use crate::WitnessComponent;
use libloading::Library;
use std::sync::atomic::{AtomicBool, Ordering};

pub const MAX_COMPONENTS: usize = 1000;

pub struct WitnessManager<F: PrimeField64> {
    components: RwLock<Vec<Arc<dyn WitnessComponent<F>>>>,
    components_instance_ids: Vec<RwLock<Vec<usize>>>,
    components_std: RwLock<Vec<Arc<dyn WitnessComponent<F>>>>,
    pctx: Arc<ProofCtx<F>>,
    sctx: Arc<SetupCtx<F>>,
    public_inputs_path: RwLock<Option<PathBuf>>,
    input_data_path: RwLock<Option<PathBuf>>,
    init: AtomicBool,
    library: Mutex<Option<Library>>,
    execution_done: AtomicBool,
}

impl<F: PrimeField64> WitnessManager<F> {
    pub fn new(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) -> Self {
        WitnessManager {
            components: RwLock::new(Vec::new()),
            components_instance_ids: (0..MAX_COMPONENTS).map(|_| RwLock::new(Vec::new())).collect(),
            components_std: RwLock::new(Vec::new()),
            pctx,
            sctx,
            public_inputs_path: RwLock::new(None),
            input_data_path: RwLock::new(None),
            init: AtomicBool::new(false),
            library: Mutex::new(None),
            execution_done: AtomicBool::new(false),
        }
    }

    pub fn set_init_witness(&self, init: bool, library: Library) {
        self.init.store(init, Ordering::SeqCst);
        self.library.lock().unwrap().replace(library);
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

    pub fn execute(&self, minimal_memory: bool) {
        self.execution_done.store(false, Ordering::SeqCst);
        let n_components = self.components_std.read().unwrap().len();
        for (idx, component) in self.components_std.read().unwrap().iter().enumerate() {
            component.execute(
                self.pctx.clone(),
                &self.components_instance_ids[n_components + idx],
                self.input_data_path.read().unwrap().clone(),
            );
        }

        for (idx, component) in self.components.read().unwrap().iter().enumerate() {
            component.execute(
                self.pctx.clone(),
                &self.components_instance_ids[idx],
                self.input_data_path.read().unwrap().clone(),
            );
        }

        self.pctx.dctx_assign_instances(minimal_memory);

        self.execution_done.store(true, Ordering::SeqCst);
    }

    pub fn reset(&self) {
        self.components_instance_ids.iter().for_each(|ids| ids.write().unwrap().clear());
    }

    pub fn debug(&self, instance_ids: &[usize], debug_info: &DebugInfo) {
        if debug_info.std_mode.name == ModeName::Debug || !debug_info.debug_instances.is_empty() {
            for (idx, component) in self.components.read().unwrap().iter().enumerate() {
                let ids_hash_set: HashSet<usize> = instance_ids.iter().cloned().collect();

                let instance_ids_filtered: Vec<usize> = ids_hash_set
                    .iter()
                    .filter(|id| self.components_instance_ids[idx].read().unwrap().contains(id))
                    .cloned() // turn &&usize → usize
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

    pub fn pre_calculate_tables(&self) {
        for component in self.components.read().unwrap().iter() {
            component.pre_calculate_tables(self.pctx.clone());
        }
    }

    pub fn pre_calculate_witness(
        &self,
        stage: u32,
        instance_ids: &[usize],
        n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) {
        for (idx, component) in self.components.read().unwrap().iter().enumerate() {
            let ids_hash_set: HashSet<usize> = instance_ids.iter().cloned().collect();

            let instance_ids_filtered: Vec<_> = ids_hash_set
                .iter()
                .filter(|id| {
                    self.components_instance_ids[idx].read().unwrap().contains(id)
                        && (self.pctx.dctx_is_my_process_instance(**id) || self.pctx.dctx_is_table(**id))
                        && !self.pctx.dctx_is_instance_calculated(**id)
                })
                .cloned()
                .collect();

            if !instance_ids_filtered.is_empty() {
                component.pre_calculate_witness(
                    stage,
                    self.pctx.clone(),
                    self.sctx.clone(),
                    &instance_ids_filtered,
                    n_cores,
                    buffer_pool,
                );
            }
        }

        if self.execution_done.load(Ordering::SeqCst) {
            for component in self.components_std.read().unwrap().iter() {
                component.pre_calculate_witness(
                    stage,
                    self.pctx.clone(),
                    self.sctx.clone(),
                    instance_ids,
                    n_cores,
                    buffer_pool,
                );
            }
        }
    }

    pub fn calculate_witness(
        &self,
        stage: u32,
        instance_ids: &[usize],
        n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) {
        for (idx, component) in self.components.read().unwrap().iter().enumerate() {
            let ids_hash_set: HashSet<usize> = instance_ids.iter().cloned().collect();

            let instance_ids_filtered: Vec<_> = ids_hash_set
                .iter()
                .filter(|id| {
                    self.components_instance_ids[idx].read().unwrap().contains(id)
                        && (self.pctx.dctx_is_my_process_instance(**id) || self.pctx.dctx_is_table(**id))
                        && !self.pctx.dctx_is_instance_calculated(**id)
                })
                .cloned()
                .collect();

            if !instance_ids_filtered.is_empty() {
                for id in &instance_ids_filtered {
                    self.pctx.dctx_set_instance_calculated(*id);
                }
                component.calculate_witness(
                    stage,
                    self.pctx.clone(),
                    self.sctx.clone(),
                    &instance_ids_filtered,
                    n_cores,
                    buffer_pool,
                );
            }
        }

        if self.execution_done.load(Ordering::SeqCst) {
            for component in self.components_std.read().unwrap().iter() {
                component.calculate_witness(
                    stage,
                    self.pctx.clone(),
                    self.sctx.clone(),
                    instance_ids,
                    n_cores,
                    buffer_pool,
                );
            }
        }
    }

    pub fn end(&self, debug_info: &DebugInfo) {
        for component in self.components.read().unwrap().iter() {
            component.end(self.pctx.clone(), self.sctx.clone(), debug_info);
        }
        for component in self.components_std.read().unwrap().iter() {
            component.end(self.pctx.clone(), self.sctx.clone(), debug_info);
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
