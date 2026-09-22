use std::collections::HashSet;
use std::sync::{Arc, RwLock, Mutex};
use std::path::PathBuf;

use proofman_fields::PrimeField64;
use proofman_common::{BufferPool, DebugInfo, RankInfo, ModeName, ProofCtx, ProofmanResult, SetupCtx, WitnessState};
use crate::WitnessComponent;
use libloading::Library;
use std::sync::atomic::{AtomicBool, Ordering};

/// The lifecycle answers "does this instance have its witness". Only stage 1 produces one: later
/// stages run over a trace that already exists, so they must bypass the gate -- it would refuse
/// every instance as `Done` and silently skip the component.
const WITNESS_STAGE: u32 = 1;

/// Dedup while keeping the caller's order. The filters below used to iterate a `HashSet`, discarding
/// the caller's dispatch order and varying it run to run.
fn dedup_preserving_order(ids: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::with_capacity(ids.len());
    ids.iter().copied().filter(|id| seen.insert(*id)).collect()
}

pub const MAX_COMPONENTS: usize = 1000;

pub struct WitnessManager<F: PrimeField64> {
    components: RwLock<Vec<Arc<dyn WitnessComponent<F>>>>,
    components_instance_ids: Vec<RwLock<Vec<usize>>>,
    components_std: RwLock<Vec<Arc<dyn WitnessComponent<F>>>>,
    pctx: Arc<ProofCtx<F>>,
    sctx: Arc<SetupCtx<F>>,
    public_inputs_path: RwLock<Option<PathBuf>>,
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
            init: AtomicBool::new(false),
            library: Mutex::new(None),
            execution_done: AtomicBool::new(false),
        }
    }

    pub fn get_rank_info(&self) -> RankInfo {
        RankInfo {
            world_rank: self.pctx.mpi_ctx.rank,
            local_rank: self.pctx.mpi_ctx.node_rank,
            n_processes: self.pctx.mpi_ctx.n_processes,
        }
    }

    pub fn set_witness_initialized(&self) {
        self.init.store(true, Ordering::SeqCst);
    }

    pub fn set_init_witness(&self, init: bool, library: Library) {
        self.init.store(init, Ordering::SeqCst);
        let _ = self.library.lock().unwrap().take();
        std::mem::forget(library);
    }

    pub fn is_init_witness(&self) -> bool {
        self.init.load(Ordering::SeqCst)
    }

    pub fn set_public_inputs_path(&self, path: Option<PathBuf>) {
        *self.public_inputs_path.write().unwrap() = path;
    }

    pub fn register_component(&self, component: Arc<dyn WitnessComponent<F>>) {
        self.components.write().unwrap().push(component);
    }

    pub fn register_component_std(&self, component: Arc<dyn WitnessComponent<F>>) {
        self.components_std.write().unwrap().push(component);
    }

    /// Whether a witness library has registered any component through `register_component`.
    /// Deliberately ignores `components_std` (the std library registers those on its own): the
    /// callers gate work only the external witness library can do, such as producing the
    /// custom-commit fixed files.
    pub fn has_witness_lib_components(&self) -> bool {
        !self.components.read().unwrap().is_empty()
    }

    pub fn gen_custom_commits_fixed(&self) -> ProofmanResult<()> {
        for component in self.components.read().unwrap().iter() {
            component.gen_custom_commits_fixed(self.pctx.clone(), self.sctx.clone())?;
        }

        Ok(())
    }

    pub fn execute(&self) -> ProofmanResult<()> {
        self.execution_done.store(false, Ordering::SeqCst);

        let n_regular_components = self.components.read().unwrap().len();

        for (idx, component) in self.components_std.read().unwrap().iter().enumerate() {
            component.execute(
                self.pctx.clone(),
                self.sctx.clone(),
                &self.components_instance_ids[n_regular_components + idx],
            )?;
        }

        for (idx, component) in self.components.read().unwrap().iter().enumerate() {
            component.execute(self.pctx.clone(), self.sctx.clone(), &self.components_instance_ids[idx])?;
        }

        self.pctx.dctx_assign_instances()?;

        self.execution_done.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn reset(&self) {
        self.components_instance_ids.iter().for_each(|ids| ids.write().unwrap().clear());
    }

    pub fn debug(&self, instance_ids: &[usize], debug_info: &DebugInfo) -> ProofmanResult<()> {
        if debug_info.std_mode.name == ModeName::Debug {
            let unique = dedup_preserving_order(instance_ids);
            for (idx, component) in self.components.read().unwrap().iter().enumerate() {
                let instance_ids_filtered: Vec<usize> = unique
                    .iter()
                    .filter(|id| self.components_instance_ids[idx].read().unwrap().contains(id))
                    .cloned()
                    .collect();

                if !instance_ids_filtered.is_empty() {
                    component.debug(self.pctx.clone(), self.sctx.clone(), &instance_ids_filtered)?;
                }
            }
        }
        if debug_info.std_mode.name == ModeName::Debug {
            for component in self.components_std.read().unwrap().iter() {
                component.debug(self.pctx.clone(), self.sctx.clone(), instance_ids)?;
            }
        }
        Ok(())
    }

    /// Takes no ownership: the default impl queues through `announce_witness_ready`, which needs the
    /// instance still queueable. The old `!calculated` filter read the channel backlog as untouched.
    /// Like `calculate_witness`, only stage 1 is filtered by the lifecycle.
    pub fn pre_calculate_witness(
        &self,
        stage: u32,
        instance_ids: &[usize],
        n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        let unique = dedup_preserving_order(instance_ids);
        for (idx, component) in self.components.read().unwrap().iter().enumerate() {
            let mut instance_ids_filtered = Vec::new();

            for id in &unique {
                // Only what has not had a witness yet in this phase. `Evicted` is excluded with
                // `Done`: it already had one, and preparing it again is what re-announced it.
                if self.owns(idx, *id)?
                    && (stage != WITNESS_STAGE
                        || matches!(self.pctx.dctx_witness_state(*id), WitnessState::Absent | WitnessState::Queued))
                {
                    instance_ids_filtered.push(*id);
                }
            }

            if !instance_ids_filtered.is_empty() {
                component.pre_calculate_witness(
                    stage,
                    self.pctx.clone(),
                    self.sctx.clone(),
                    &instance_ids_filtered,
                    n_cores,
                    buffer_pool,
                )?;
            }
        }

        if self.execution_done.load(Ordering::SeqCst) {
            for component in self.components_std.read().unwrap().iter() {
                component.pre_calculate_witness(
                    stage,
                    self.pctx.clone(),
                    self.sctx.clone(),
                    &unique,
                    n_cores,
                    buffer_pool,
                )?;
            }
        }
        Ok(())
    }

    /// This component handles this instance, and it belongs to this process.
    fn owns(&self, idx: usize, id: usize) -> ProofmanResult<bool> {
        Ok(self.components_instance_ids[idx].read().unwrap().contains(&id)
            && (self.pctx.dctx_is_my_process_instance(id)? || self.pctx.dctx_is_table(id)))
    }

    /// Hand back every instance a hook owned, `Done` only for those whose trace is now resident.
    fn release_all(&self, owned: &[usize]) {
        for id in owned {
            let produced = self.pctx.is_air_instance_stored(*id);
            self.pctx.dctx_release_witness(*id, produced);
        }
    }

    /// Compute the stage-1 witnesses of `instance_ids` that nobody else owns or has already
    /// produced. Later stages run over an existing trace and bypass the gate entirely.
    pub fn calculate_witness(
        &self,
        stage: u32,
        instance_ids: &[usize],
        n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        let unique = dedup_preserving_order(instance_ids);
        let gated = stage == WITNESS_STAGE;
        for (idx, component) in self.components.read().unwrap().iter().enumerate() {
            // Two passes on purpose: the fallible one first. Mixed into the acquire loop, an error
            // on a later id would return with the earlier ones stuck in `Running` -- unreleasable,
            // and never computable again.
            let mut candidates = Vec::new();
            for id in &unique {
                if self.owns(idx, *id)? {
                    candidates.push(*id);
                }
            }
            let owned: Vec<usize> =
                candidates.into_iter().filter(|id| !gated || self.pctx.dctx_try_acquire_witness(*id)).collect();

            if !owned.is_empty() {
                let ids = owned.as_slice();
                if gated {
                    // Reported, not skipped: a component may legitimately take its buffer early.
                    for id in ids.iter().filter(|id| self.pctx.is_air_instance_stored(**id)) {
                        self.pctx.note_witness_recomputed_over_trace(*id);
                    }
                }
                let result =
                    component.calculate_witness(stage, self.pctx.clone(), self.sctx.clone(), ids, n_cores, buffer_pool);
                if gated {
                    self.release_all(ids);
                }
                result?;
            }
        }

        if self.execution_done.load(Ordering::SeqCst) {
            for component in self.components_std.read().unwrap().iter() {
                component.calculate_witness(
                    stage,
                    self.pctx.clone(),
                    self.sctx.clone(),
                    &unique,
                    n_cores,
                    buffer_pool,
                )?;
            }
        }
        Ok(())
    }

    pub fn end(&self, debug_info: &DebugInfo) -> ProofmanResult<()> {
        for component in self.components.read().unwrap().iter() {
            component.end(self.pctx.clone(), self.sctx.clone(), debug_info)?;
        }
        for component in self.components_std.read().unwrap().iter() {
            component.end(self.pctx.clone(), self.sctx.clone(), debug_info)?;
        }
        Ok(())
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
}
