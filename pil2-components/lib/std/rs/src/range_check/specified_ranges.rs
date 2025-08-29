use core::panic;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use proofman_util::create_buffer_fast;
use rayon::prelude::*;

use fields::PrimeField64;
use std::path::PathBuf;

use witness::WitnessComponent;
use proofman_common::{AirInstance, BufferPool, ProofCtx, SetupCtx, TraceInfo};
use proofman_hints::{get_hint_field_constant_a, get_hint_ids_by_name, HintFieldOptions, HintFieldValue};

use crate::{get_hint_field_constant_as, validate_binary_field, AirComponent};

#[derive(Debug, Clone)]
pub struct SpecifiedRange {
    mul_idx: usize,
    min: i64,
}

pub struct SpecifiedRanges {
    airgroup_id: usize,
    air_id: usize,
    shift: usize,
    mask: usize,
    num_rows: usize,
    num_cols: usize,
    multiplicities: Vec<Vec<AtomicU64>>,
    instance_id: AtomicU64,
    calculated: AtomicBool,
    ranges: Vec<SpecifiedRange>,
}

impl<F: PrimeField64> AirComponent<F> for SpecifiedRanges {
    fn new(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, airgroup_id: usize, air_id: usize) -> Arc<Self> {
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        let setup = sctx.get_setup(airgroup_id, air_id);
        let hint_opt = HintFieldOptions::default();
        let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "specified_ranges_data")[0] as usize;

        // Get the relevant data
        let col_num =
            get_hint_field_constant_as::<u64, F>(sctx, airgroup_id, air_id, hint_id, "col_num", hint_opt.clone());

        let mins = get_hint_field_constant_a::<F>(sctx, airgroup_id, air_id, hint_id, "mins", hint_opt.clone()).values;
        let mins_neg =
            get_hint_field_constant_a::<F>(sctx, airgroup_id, air_id, hint_id, "mins_neg", hint_opt.clone()).values;

        let opids_count =
            get_hint_field_constant_as::<u64, F>(sctx, airgroup_id, air_id, hint_id, "opids_count", hint_opt.clone());
        let opids_len =
            get_hint_field_constant_a::<F>(sctx, airgroup_id, air_id, hint_id, "opids_len", hint_opt).values;

        // Get and store the ranges
        let mut ranges = Vec::with_capacity(opids_count as usize);
        let mut offset = 0;
        for ((min_hint, min_neg_hint), opid_len_hint) in mins.iter().zip(mins_neg.iter()).zip(opids_len.iter()) {
            let min = match min_hint {
                HintFieldValue::Field(f) => f.as_canonical_u64(),
                _ => panic!("Opid hint must be a field element"),
            };

            let min_neg = match min_neg_hint {
                HintFieldValue::Field(f) => validate_binary_field(*f, "Min neg"),
                _ => panic!("Opid hint must be a field element"),
            };

            let min = if min_neg { min as i128 - F::ORDER_U64 as i128 } else { min as i128 };

            let opid_len = match opid_len_hint {
                HintFieldValue::Field(f) => f.as_canonical_u64() as usize,
                _ => panic!("Opid hint must be a field element"),
            };

            // In this conversion we assume that min is at most of 63 bits
            // We can safely assume it because we have already check this minimum before
            ranges.push(SpecifiedRange { mul_idx: offset, min: min as i64 });

            offset += opid_len;
        }

        let num_cols = col_num as usize;
        let multiplicities = (0..num_cols)
            .into_par_iter()
            .map(|_| (0..num_rows).into_par_iter().map(|_| AtomicU64::new(0)).collect())
            .collect();

        Arc::new(Self {
            airgroup_id,
            air_id,
            shift: num_rows.trailing_zeros() as usize,
            mask: num_rows - 1,
            num_cols,
            num_rows,
            multiplicities,
            instance_id: AtomicU64::new(0),
            calculated: AtomicBool::new(false),
            ranges,
        })
    }
}

impl SpecifiedRanges {
    pub fn get_global_row(range_min: i64, value: i64) -> u64 {
        (value - range_min) as u64
    }

    pub fn get_global_rows(range_min: i64, values: &[i64]) -> Vec<u64> {
        values.iter().map(|&v| Self::get_global_row(range_min, v)).collect()
    }

    #[inline(always)]
    pub fn update_input(&self, id: usize, value: i64, multiplicity: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the ranges for the given id
        let ranges = &self.ranges[id];
        let min_global = ranges.min;
        let base_offset = ranges.mul_idx;

        // Identify to which sub-range the value belongs
        let offset = (value - min_global) as usize;
        let range_idx = offset >> self.shift;

        // Get the row index
        let row_idx = offset & self.mask;

        // Update the multiplicity
        self.multiplicities[base_offset + range_idx][row_idx].fetch_add(multiplicity, Ordering::Relaxed);
    }

    pub fn update_inputs(&self, id: usize, values: Vec<u32>) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Get the ranges for the given id
        let ranges = &self.ranges[id];
        let min_global = ranges.min;
        let base_offset = ranges.mul_idx;

        // Identify to which sub-range the value belongs
        for (value, multiplicity) in values.iter().enumerate() {
            let offset = (value as i64 - min_global) as usize;
            let range_idx = offset >> self.shift;

            // Get the row index
            let row_idx = offset & self.mask;

            // Update the multiplicity
            self.multiplicities[base_offset + range_idx][row_idx].fetch_add(*multiplicity as u64, Ordering::Relaxed);
        }
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField64> WitnessComponent<F> for SpecifiedRanges {
    fn execute(&self, pctx: Arc<ProofCtx<F>>, _input_data_path: Option<PathBuf>) -> Vec<usize> {
        let (instance_found, mut instance_id) = pctx.dctx_find_instance_mine(self.airgroup_id, self.air_id);

        if !instance_found {
            instance_id = pctx.add_table(self.airgroup_id, self.air_id);
        }

        self.calculated.store(false, Ordering::Relaxed);
        self.instance_id.store(instance_id as u64, Ordering::SeqCst);
        Vec::new()
    }

    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
        if stage == 1 {
            let instance_id = self.instance_id.load(Ordering::Relaxed) as usize;

            if !_instance_ids.contains(&instance_id) {
                return;
            }

            self.calculated.store(true, Ordering::Relaxed);

            // pctx.dctx_distribute_multiplicities(&self.multiplicities, instance_id);

            // if pctx.dctx_is_my_instance(instance_id) {
            let buffer_size = self.num_cols * self.num_rows;
            let mut buffer = create_buffer_fast(buffer_size);
            buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                for (col, vec) in self.multiplicities.iter().enumerate() {
                    chunk[col] = F::from_u64(vec[row].swap(0, Ordering::Relaxed));
                }
            });

            let air_instance =
                AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, self.num_rows, buffer, false));
            pctx.add_air_instance(air_instance, instance_id);
            // } else {
            //     self.multiplicities.par_iter().for_each(|vec| {
            //         for vec_row in vec.iter() {
            //             vec_row.swap(0, Ordering::Relaxed);
            //         }
            //     });
            // }
        }
    }
}
