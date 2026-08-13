use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
};
use rayon::prelude::*;

use proofman_fields::PrimeField64;

use proofman_witness::WitnessComponent;
use proofman_common::{
    register_host_buffer, unregister_host_buffer, AirInstance, BufferPool, ProofCtx, ProofmanError, ProofmanResult,
    SetupCtx, TraceInfo,
};
use proofman_hints::{get_hint_ids_by_name, HintFieldOptions};

use crate::{get_global_hint_field_constant_a_as, get_hint_field_constant_a_as, get_hint_field_constant_as};

pub struct StdVirtualTable<F: PrimeField64> {
    _phantom: std::marker::PhantomData<F>,
    pub global_id_by_uid: HashMap<usize, usize>,   // uid -> global_id
    pub indices_by_global_id: Vec<(usize, usize)>, // global_id -> (air_idx, uid_idx)
    pub virtual_table_airs: Option<Vec<Arc<VirtualTableAir<F>>>>,
}
pub struct VirtualTableAir<F: PrimeField64> {
    airgroup_id: usize,
    air_id: usize,
    shift: u64,
    mask: u64,
    num_rows: usize,
    num_cols: usize,
    table_ids: Vec<(usize, u64)>, // (table_id, acc_height)
    // Flat col-major: idx = col * num_rows + row. Single allocation.
    multiplicities: Vec<AtomicU64>,
    table_instance_id: AtomicU64,
    calculated: AtomicBool,
    shared_tables: bool,
    // Persistent trace buffer slot. Pre-allocated in `StdVirtualTable::new`; taken in
    // `calculate_witness` and refilled by `ProofCtx::free_instance_traces`.
    trace_buffer: Arc<Mutex<Option<Vec<F>>>>,
    // Pinned-registration page base for `trace_buffer`; unpinned it takes the staging H2D path.
    trace_buffer_pinned: Option<usize>,
}

impl<F: PrimeField64> Drop for VirtualTableAir<F> {
    fn drop(&mut self) {
        // Runs before the field drops, so the pages are still live here.
        if let Some(base) = self.trace_buffer_pinned {
            unregister_host_buffer(base);
        }
    }
}

impl<F: PrimeField64> StdVirtualTable<F> {
    pub fn new(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, shared_tables: bool) -> ProofmanResult<Arc<Self>> {
        // Get relevant data from the global hint
        let virtual_table_global_hint = get_hint_ids_by_name(sctx.get_global_bin(), "virtual_table_data_global");
        if virtual_table_global_hint.is_empty() {
            return Ok(Arc::new(Self {
                _phantom: std::marker::PhantomData,
                global_id_by_uid: HashMap::new(),
                indices_by_global_id: Vec::new(),
                virtual_table_airs: None,
            }));
        }

        let airgroup_ids =
            get_global_hint_field_constant_a_as::<usize, F>(sctx, virtual_table_global_hint[0], "airgroup_ids")?;
        let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, virtual_table_global_hint[0], "air_ids")?;

        let num_virtual_tables = airgroup_ids.len();
        let mut virtual_tables = Vec::with_capacity(num_virtual_tables);
        let mut global_id_by_uid = HashMap::new();
        let mut indices_by_global_id = Vec::new();
        let mut current_global_id = 0;
        for i in 0..num_virtual_tables {
            let airgroup_id = airgroup_ids[i];
            let air_id = air_ids[i];

            // Get the Virtual Table structure
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "virtual_table_data")[0] as usize;

            let hint_opt = HintFieldOptions::default();
            let table_ids = get_hint_field_constant_a_as::<usize, F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint_id,
                "table_ids",
                hint_opt.clone(),
            )?;
            let acc_heights = get_hint_field_constant_a_as::<u64, F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint_id,
                "acc_heights",
                hint_opt.clone(),
            )?;
            let num_muls = get_hint_field_constant_as::<usize, F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint_id,
                "num_muls",
                hint_opt.clone(),
            )?;

            // Map each table_id to an ordered set of indexes
            let num_table_ids = table_ids.len();
            let mut idxs = vec![(0, 0); num_table_ids];
            for j in 0..num_table_ids {
                idxs[j] = (table_ids[j], acc_heights[j]);

                // Update global ID mapping: global_idx -> (air_idx, uid, uid_idx)
                // global_id_map.insert(current_global_id, (i, table_ids[j], j));
                global_id_by_uid.insert(table_ids[j], current_global_id);
                indices_by_global_id.push((i, j));
                current_global_id += 1;
            }

            let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;
            let multiplicities: Vec<AtomicU64> =
                (0..(num_muls as usize * num_rows)).into_par_iter().map(|_| AtomicU64::new(0)).collect();

            let buffer = vec![F::ZERO; num_muls as usize * num_rows];
            // The reclaim slot returns this same allocation every iteration, so one pin covers every H2D.
            let trace_buffer_pinned = if pctx.gpu { register_host_buffer(&buffer) } else { None };
            let trace_buffer = Arc::new(Mutex::new(Some(buffer)));
            let virtual_table_air = VirtualTableAir::<F> {
                airgroup_id,
                air_id,
                shift: num_rows.trailing_zeros() as u64,
                mask: (num_rows - 1) as u64,
                num_rows,
                num_cols: num_muls as usize,
                table_ids: idxs,
                multiplicities,
                table_instance_id: AtomicU64::new(0),
                calculated: AtomicBool::new(false),
                shared_tables,
                trace_buffer,
                trace_buffer_pinned,
            };
            virtual_tables.push(Arc::new(virtual_table_air));
        }

        Ok(Arc::new(Self {
            _phantom: std::marker::PhantomData,
            global_id_by_uid,
            indices_by_global_id,
            virtual_table_airs: Some(virtual_tables),
        }))
    }

    pub fn get_global_id(&self, id: usize) -> ProofmanResult<usize> {
        self.global_id_by_uid
            .get(&id)
            .copied()
            .ok_or_else(|| ProofmanError::StdError(format!("ID {id} not found in the global ID map")))
    }

    pub fn inc_virtual_row(&self, global_id: usize, row: u64, multiplicity: u64) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_row(uid_idx, row, multiplicity);
    }

    pub fn inc_virtual_rows(&self, global_id: usize, rows: &[u64], multiplicities: &[u64]) {
        debug_assert!(!rows.is_empty() && rows.len() == multiplicities.len());
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_rows(uid_idx, rows, multiplicities);
    }

    pub fn inc_virtual_rows_same_mul(&self, global_id: usize, rows: &[u64], multiplicity: u64) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_rows_same_mul(uid_idx, rows, multiplicity);
    }

    pub fn inc_virtual_rows_ranged(&self, global_id: usize, start: Option<u64>, multiplicities: &[u64]) {
        let start = start.unwrap_or(0);
        // Compute (row, multiplicity) pairs on the fly — no Vec allocation.
        let pairs = multiplicities.iter().copied().enumerate().map(move |(i, m)| (start + i as u64, m));
        self.inc_virtual_pairs(global_id, pairs);
    }

    /// Increment multiplicities directly from an iterator of (row, multiplicity) pairs.
    /// Lets callers avoid materializing an intermediate Vec<u64> of rows.
    pub fn inc_virtual_pairs(&self, global_id: usize, pairs: impl Iterator<Item = (u64, u64)>) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_pairs(uid_idx, pairs);
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> WitnessComponent<F> for StdVirtualTable<F> {
    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        Ok(())
    }
}

impl<F: PrimeField64> VirtualTableAir<F> {
    pub fn get_id(&self, id: usize) -> ProofmanResult<usize> {
        if let Some(pos) = self.table_ids.iter().position(|&(table_id, _)| table_id == id) {
            Ok(pos)
        } else {
            Err(ProofmanError::StdError("ID not found in the virtual table".to_string()))
        }
    }

    /// Core update function: Updates multiplicities for row/multiplicity pairs
    fn update(&self, table_offset: u64, iter: impl Iterator<Item = (u64, u64)>) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        for (row, multiplicity) in iter {
            if multiplicity == 0 {
                continue;
            }

            // Get the offset
            let offset = table_offset + row;

            // Map it to the appropriate multiplicity
            let sub_table_idx = offset >> self.shift;

            // Get the row index
            let row_idx = offset & self.mask;

            // Update the multiplicity (col-major flat layout)
            self.multiplicities[sub_table_idx as usize * self.num_rows + row_idx as usize]
                .fetch_add(multiplicity, Ordering::Relaxed);
        }
    }

    pub fn inc_virtual_row(&self, id: usize, row: u64, multiplicity: u64) {
        let table_offset = self.table_ids[id].1;
        self.update(table_offset, std::iter::once((row, multiplicity)));
    }

    pub fn inc_virtual_rows(&self, id: usize, rows: &[u64], multiplicities: &[u64]) {
        let table_offset = self.table_ids[id].1;
        self.update(table_offset, rows.iter().copied().zip(multiplicities.iter().copied()));
    }

    pub fn inc_virtual_rows_same_mul(&self, id: usize, rows: &[u64], multiplicity: u64) {
        let table_offset = self.table_ids[id].1;
        self.update(table_offset, rows.iter().copied().map(|r| (r, multiplicity)));
    }

    /// Increment multiplicities directly from an iterator of (row, multiplicity) pairs.
    pub fn inc_virtual_pairs(&self, id: usize, pairs: impl Iterator<Item = (u64, u64)>) {
        let table_offset = self.table_ids[id].1;
        self.update(table_offset, pairs);
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> WitnessComponent<F> for VirtualTableAir<F> {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let (instance_found, mut table_instance_id) = pctx.dctx_find_process_table(self.airgroup_id, self.air_id)?;

        if !instance_found {
            if !self.shared_tables {
                table_instance_id = pctx.add_table_all(self.airgroup_id, self.air_id)?;
            } else {
                table_instance_id = pctx.add_table(self.airgroup_id, self.air_id)?;
            }
        }

        self.calculated.store(false, Ordering::Relaxed);
        self.multiplicities.par_iter().for_each(|v| {
            v.store(0, Ordering::Relaxed);
        });

        self.table_instance_id.store(table_instance_id as u64, Ordering::SeqCst);
        Ok(())
    }

    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        Ok(())
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage == 1 {
            let table_instance_id = self.table_instance_id.load(Ordering::Relaxed) as usize;

            let instance_id = pctx.dctx_get_table_instance_idx(table_instance_id)?;

            if !_instance_ids.contains(&instance_id) {
                return Ok(());
            }

            self.calculated.store(true, Ordering::Relaxed);

            if self.shared_tables {
                let owner_idx = pctx.dctx_get_process_owner_instance(instance_id)?;
                pctx.mpi_ctx.distribute_multiplicities(&self.multiplicities, self.num_cols, self.num_rows, owner_idx);
            }

            if !self.shared_tables || pctx.dctx_is_my_process_instance(instance_id)? {
                let buffer_size = self.num_cols * self.num_rows;
                // The slot is pre-populated by `new` and refilled by the reclaim hook
                // on every prior iteration's clear_traces / Drop. If it's empty here,
                // the reclaim path is broken.
                let mut buffer = self
                    .trace_buffer
                    .lock()
                    .unwrap()
                    .take()
                    .expect("VirtualTableAir trace_buffer must be populated by reclaim before calculate_witness");
                debug_assert_eq!(buffer.len(), buffer_size);
                let any_nonzero = std::sync::atomic::AtomicBool::new(false);
                let num_rows = self.num_rows;
                buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                    for (col, slot) in chunk.iter_mut().enumerate() {
                        let v = self.multiplicities[col * num_rows + row].load(Ordering::Relaxed);
                        if v != 0 {
                            any_nonzero.store(true, Ordering::Relaxed);
                        }
                        *slot = F::from_u64(v);
                    }
                });
                if !any_nonzero.load(Ordering::Relaxed) {
                    tracing::info!(
                        "Skipping uninitialized virtual table (airgroup_id: {}, air_id: {})",
                        self.airgroup_id,
                        self.air_id
                    );
                    pctx.dctx_skip_process_instance(instance_id);
                    *self.trace_buffer.lock().unwrap() = Some(buffer);
                    return Ok(());
                }
                let setup = sctx.get_setup(self.airgroup_id, self.air_id)?;
                let n_cols = setup.stark_info.map_sections_n["cm1"] as usize;
                let air_instance = AirInstance::new(
                    TraceInfo::new(self.airgroup_id, self.air_id, n_cols, self.num_rows, buffer, false, false)
                        .with_reclaim_slot(self.trace_buffer.clone()),
                );
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
        Ok(())
    }
}
