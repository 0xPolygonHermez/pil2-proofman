use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex, RwLock,
};
use rayon::prelude::*;

use fields::PrimeField64;

use witness::WitnessComponent;
use proofman_common::{AirInstance, BufferPool, ProofCtx, ProofmanError, ProofmanResult, SetupCtx, TraceInfo};
use proofman_hints::{get_hint_field_constant_a, get_hint_ids_by_name, HintFieldOptions, HintFieldValue};

use crate::{get_hint_field_constant_as, validate_binary_field, AirComponent, extract_field_element_as_usize};

#[derive(Debug, Clone)]
pub struct SpecifiedRange {
    acc_height: usize,
    min: i64,
}

pub struct SpecifiedRanges<F: PrimeField64> {
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
    ranges: Vec<SpecifiedRange>,
    shared_tables: bool,
    // Persistent trace buffer slot. Pre-allocated in `new`; taken in `calculate_witness`
    // and refilled by `ProofCtx::free_instance_traces` via the reclaim registry.
    trace_buffer: Arc<Mutex<Option<Vec<F>>>>,
}

impl<F: PrimeField64> AirComponent<F> for SpecifiedRanges<F> {
    fn new(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        airgroup_id: usize,
        air_id: usize,
        shared_tables: bool,
    ) -> ProofmanResult<Arc<Self>> {
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        let setup = sctx.get_setup(airgroup_id, air_id)?;
        let hint_opt = HintFieldOptions::default();
        let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "specified_ranges_data")[0] as usize;

        // Get the relevant data
        let num_muls = get_hint_field_constant_as::<u64, F>(
            pctx,
            setup,
            airgroup_id,
            air_id,
            hint_id,
            "num_muls",
            hint_opt.clone(),
        )?;

        let mins =
            get_hint_field_constant_a::<F>(pctx, setup, airgroup_id, air_id, hint_id, "mins", hint_opt.clone())?.values;
        let mins_neg =
            get_hint_field_constant_a::<F>(pctx, setup, airgroup_id, air_id, hint_id, "mins_neg", hint_opt.clone())?
                .values;

        let opids_count = get_hint_field_constant_as::<u64, F>(
            pctx,
            setup,
            airgroup_id,
            air_id,
            hint_id,
            "opids_count",
            hint_opt.clone(),
        )?;
        let acc_heights =
            get_hint_field_constant_a::<F>(pctx, setup, airgroup_id, air_id, hint_id, "acc_heights", hint_opt)?.values;

        // Get and store the ranges
        let mut ranges = Vec::with_capacity(opids_count as usize);
        for ((min_hint, min_neg_hint), acc_heights_hint) in mins.iter().zip(mins_neg.iter()).zip(acc_heights.iter()) {
            let min = match min_hint {
                HintFieldValue::Field(f) => f.as_canonical_u64(),
                _ => return Err(ProofmanError::StdError("min hint must be a field element".to_string())),
            };

            let min_neg = match min_neg_hint {
                HintFieldValue::Field(f) => validate_binary_field(*f, "Min neg")?,
                _ => return Err(ProofmanError::StdError("min neg hint must be a field element".to_string())),
            };

            let min = if min_neg { min as i128 - F::ORDER_U64 as i128 } else { min as i128 };

            let acc_heights = extract_field_element_as_usize(acc_heights_hint, "Acc Heights")?;

            // In this conversion we assume that min is at most of 63 bits
            // We can safely assume it because we have already check this minimum before
            ranges.push(SpecifiedRange { acc_height: acc_heights, min: min as i64 });
        }

        let num_cols = num_muls as usize;
        let multiplicities: Vec<AtomicU64> =
            (0..(num_cols * num_rows)).into_par_iter().map(|_| AtomicU64::new(0)).collect();
        let trace_buffer = Arc::new(Mutex::new(Some(vec![F::ZERO; num_cols * num_rows])));

        Ok(Arc::new(Self {
            airgroup_id,
            air_id,
            shift: num_rows.trailing_zeros() as usize,
            mask: num_rows - 1,
            num_cols,
            num_rows,
            multiplicities,
            table_instance_id: AtomicU64::new(0),
            calculated: AtomicBool::new(false),
            ranges,
            shared_tables,
            trace_buffer,
        }))
    }
}

impl<F: PrimeField64> SpecifiedRanges<F> {
    pub fn get_global_row(range_min: i64, value: i64) -> u64 {
        (value - range_min) as u64
    }

    pub fn get_global_rows(range_min: i64, values: &[i64]) -> Vec<u64> {
        values.iter().map(|&v| Self::get_global_row(range_min, v)).collect()
    }

    /// [leak fix] Variant that writes into a caller-supplied buffer to avoid any alloc.
    pub fn get_global_rows_into(range_min: i64, values: &[i64], out: &mut Vec<u64>) {
        out.clear();
        out.extend(values.iter().map(|&v| Self::get_global_row(range_min, v)));
    }

    /// Core update function: Updates multiplicities for value/multiplicity pairs
    #[inline]
    fn update(&self, table_offset: usize, range_min: i64, iter: impl Iterator<Item = (i64, u64)>) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        for (value, multiplicity) in iter {
            if multiplicity == 0 {
                continue;
            }

            // Get the value offset
            let val_offset = (value - range_min) as usize;

            // Get the overall offset
            let offset = table_offset + val_offset;

            // Get the multiplicity index
            let mul_idx = offset >> self.shift;

            // Get the row index
            let row_idx = offset & self.mask;

            // Update the multiplicity (col-major flat layout)
            self.multiplicities[mul_idx * self.num_rows + row_idx].fetch_add(multiplicity, Ordering::Relaxed);
        }
    }

    /// Update a single value with a multiplicity
    pub fn update_value(&self, id: usize, value: i64, multiplicity: u64) {
        let range = &self.ranges[id];
        self.update(range.acc_height, range.min, std::iter::once((value, multiplicity)));
    }

    /// Update multiple values with corresponding multiplicities
    pub fn update_values(&self, id: usize, values: &[i64], multiplicities: &[u64]) {
        debug_assert!(!values.is_empty() && values.len() == multiplicities.len());
        let range = &self.ranges[id];
        self.update(range.acc_height, range.min, values.iter().copied().zip(multiplicities.iter().copied()));
    }

    /// Update multiple values with the same multiplicity
    pub fn update_values_same_mul(&self, id: usize, values: &[i64], multiplicity: u64) {
        debug_assert!(!values.is_empty());
        let range = &self.ranges[id];
        self.update(range.acc_height, range.min, values.iter().copied().map(|v| (v, multiplicity)));
    }

    /// Update directly from an iterator of (value, multiplicity) pairs. Lets callers
    /// avoid materializing intermediate buffers when values come from a synthetic range
    /// or another iterator chain.
    pub fn update_pairs(&self, id: usize, pairs: impl Iterator<Item = (i64, u64)>) {
        let range = &self.ranges[id];
        self.update(range.acc_height, range.min, pairs);
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> WitnessComponent<F> for SpecifiedRanges<F> {
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
                    .expect("SpecifiedRanges trace_buffer must be populated by reclaim before calculate_witness");
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
                        "Skipping uninitialized specified ranges table (airgroup_id: {}, air_id: {})",
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
