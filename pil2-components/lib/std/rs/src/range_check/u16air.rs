use std::sync::{
    atomic::{AtomicBool, AtomicU64},
    Arc, Mutex, RwLock,
};

use fields::PrimeField64;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
    prelude::*,
};
use witness::WitnessComponent;
use proofman_common::phase1_trace;
use proofman_common::{AirInstance, BufferPool, ProofCtx, ProofmanResult, SetupCtx, TraceInfo};
use std::sync::atomic::Ordering;
use crate::AirComponent;

const P2_16: usize = 65536;

pub struct U16Air<F: PrimeField64> {
    airgroup_id: usize,
    air_id: usize,
    shift: usize,
    mask: usize,
    num_rows: usize,
    num_cols: usize,
    // Flat col-major: idx = col * num_rows + row. Single allocation.
    multiplicities: Vec<AtomicU64>,
    table_instance_id: AtomicU64,
    calculated: AtomicBool,
    shared_tables: bool,
    // ZISK_TRACE_PHASE1, cached at construction: keeps the per-row `update` off
    // any atomic/OnceLock load when tracing is off
    traced: bool,
    // Persistent trace buffer slot. Pre-allocated in `new`; taken in `calculate_witness`
    // and refilled by `ProofCtx::free_instance_traces` via the reclaim registry.
    trace_buffer: Arc<Mutex<Option<Vec<F>>>>,
}

impl<F: PrimeField64> AirComponent<F> for U16Air<F> {
    fn new(
        pctx: &ProofCtx<F>,
        _sctx: &SetupCtx<F>,
        airgroup_id: usize,
        air_id: usize,
        shared_tables: bool,
    ) -> ProofmanResult<Arc<Self>> {
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        // Get and store the ranges
        let num_cols: usize = P2_16.div_ceil(num_rows);

        let multiplicities: Vec<AtomicU64> =
            (0..(num_cols * num_rows)).into_par_iter().map(|_| AtomicU64::new(0)).collect();
        let trace_buffer = Arc::new(Mutex::new(Some(vec![F::ZERO; num_cols * num_rows])));

        Ok(Arc::new(Self {
            airgroup_id,
            air_id,
            shift: num_rows.trailing_zeros() as usize,
            mask: num_rows - 1,
            num_rows,
            num_cols,
            multiplicities,
            table_instance_id: AtomicU64::new(0),
            calculated: AtomicBool::new(false),
            shared_tables,
            traced: phase1_trace::COMPILED && phase1_trace::enabled(),
            trace_buffer,
        }))
    }
}

impl<F: PrimeField64> U16Air<F> {
    pub const fn get_global_row(value: u16) -> u64 {
        value as u64
    }

    pub fn get_global_rows(values: &[u16]) -> Vec<u64> {
        values.iter().map(|&v| Self::get_global_row(v)).collect()
    }

    pub fn get_global_rows_into(values: &[u16], out: &mut Vec<u64>) {
        out.clear();
        out.extend(values.iter().map(|&v| Self::get_global_row(v)));
    }

    /// The contended atomic increment itself, shared by the plain and the traced loop below
    #[inline(always)]
    fn bump(&self, value: u16, multiplicity: u64) {
        // Convert the value to usize for bitwise operations
        let value = value as usize;

        // Identify to which sub-range the value belongs
        let range_idx = value >> self.shift;

        // Get the row index
        let row_idx = value & self.mask;

        // Update the multiplicity (col-major flat layout)
        self.multiplicities[range_idx * self.num_rows + row_idx].fetch_add(multiplicity, Ordering::Relaxed);
    }

    /// Core update function: Updates multiplicities for value/multiplicity pairs
    #[inline]
    fn update(&self, iter: impl Iterator<Item = (u16, u64)>) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        if phase1_trace::COMPILED && self.traced {
            return self.update_traced(iter);
        }

        for (value, multiplicity) in iter {
            if multiplicity == 0 {
                continue;
            }
            self.bump(value, multiplicity);
        }
    }

    /// See `VirtualTableAir::update_traced`
    #[cold]
    fn update_traced(&self, iter: impl Iterator<Item = (u16, u64)>) {
        // Reached once per table row by the single-row callers, so how this call is timed is
        // decided from its size: see phase1_trace::mul_begin
        let timing = phase1_trace::mul_begin(iter.size_hint().0);
        let mut rows = 0u64;
        let mut updates = 0u64;

        for (value, multiplicity) in iter {
            rows += 1;
            if multiplicity == 0 {
                continue;
            }
            updates += 1;
            self.bump(value, multiplicity);
        }

        phase1_trace::mul_end(timing, rows, updates);
    }

    /// Update a single value with a multiplicity
    pub fn update_value(&self, value: u16, multiplicity: u64) {
        self.update(std::iter::once((value, multiplicity)));
    }

    /// Update multiple values with corresponding multiplicities
    pub fn update_values(&self, values: &[u16], multiplicities: &[u64]) {
        debug_assert_eq!(values.len(), multiplicities.len());
        self.update(values.iter().copied().zip(multiplicities.iter().copied()));
    }

    /// Update multiple values with the same multiplicity
    pub fn update_values_same_mul(&self, values: &[u16], multiplicity: u64) {
        self.update(values.iter().copied().map(|v| (v, multiplicity)));
    }

    /// Update directly from an iterator of (value, multiplicity) pairs. Lets callers
    /// avoid materializing intermediate buffers when values come from a synthetic range
    /// or another iterator chain.
    pub fn update_pairs(&self, pairs: impl Iterator<Item = (u16, u64)>) {
        self.update(pairs);
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> WitnessComponent<F> for U16Air<F> {
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
        let zeroing_started = phase1_trace::enabled().then(std::time::Instant::now);
        self.multiplicities.par_iter().for_each(|v| {
            v.store(0, Ordering::Relaxed);
        });
        if let Some(started) = zeroing_started {
            phase1_trace::mul_stats().record_zeroing(self.multiplicities.len() as u64, started.elapsed());
        }
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
                    .expect("U16Air trace_buffer must be populated by reclaim before calculate_witness");
                debug_assert_eq!(buffer.len(), buffer_size);
                let any_nonzero = AtomicBool::new(false);
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
                        "Skipping uninitialized U16 range check table (airgroup_id: {}, air_id: {})",
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
