use std::sync::{
    atomic::{AtomicBool, AtomicU64},
    Arc,
};

use fields::PrimeField64;
use std::path::PathBuf;
use proofman_util::create_buffer_fast;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use witness::WitnessComponent;
use proofman_common::{AirInstance, BufferPool, ProofCtx, SetupCtx, TraceInfo};
use std::sync::atomic::Ordering;
use rayon::prelude::*;
use crate::AirComponent;

const P2_16: usize = 65536;

pub struct U16Air {
    airgroup_id: usize,
    air_id: usize,
    shift: usize,
    mask: usize,
    num_rows: usize,
    num_cols: usize,
    multiplicities: Vec<Vec<AtomicU64>>,
    instance_id: AtomicU64,
    calculated: AtomicBool,
}

impl<F: PrimeField64> AirComponent<F> for U16Air {
    fn new(pctx: &ProofCtx<F>, _sctx: &SetupCtx<F>, airgroup_id: usize, air_id: usize) -> Arc<Self> {
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        // Get and store the ranges
        let num_cols: usize = P2_16.div_ceil(num_rows);

        let multiplicities = (0..num_cols)
            .into_par_iter()
            .map(|_| (0..num_rows).into_par_iter().map(|_| AtomicU64::new(0)).collect())
            .collect();

        Arc::new(Self {
            airgroup_id,
            air_id,
            shift: num_rows.trailing_zeros() as usize,
            mask: num_rows - 1,
            num_rows,
            num_cols,
            multiplicities,
            instance_id: AtomicU64::new(0),
            calculated: AtomicBool::new(false),
        })
    }
}

impl U16Air {
    pub const fn get_global_row(value: u16) -> u64 {
        value as u64
    }

    pub fn get_global_rows(values: &[u16]) -> Vec<u64> {
        values.iter().map(|&v| Self::get_global_row(v)).collect()
    }

    #[inline(always)]
    pub fn update_input(&self, value: u16, multiplicity: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        // Identify to which sub-range the value belongs
        let range_idx = (value as usize) >> self.shift;

        // Get the row index
        let row_idx = (value as usize) & self.mask;

        // Update the multiplicity
        self.multiplicities[range_idx][row_idx].fetch_add(multiplicity, Ordering::Relaxed);
    }

    pub fn update_inputs(&self, values: Vec<u64>) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }

        for (value, multiplicity) in values.iter().enumerate() {
            if *multiplicity == 0 {
                continue;
            }

            // Identify to which sub-range the value belongs
            let range_idx = value >> self.shift;

            // Get the row index
            let row_idx = value & self.mask;

            // Update the multiplicity
            self.multiplicities[range_idx][row_idx].fetch_add(*multiplicity as u64, Ordering::Relaxed);
        }
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField64> WitnessComponent<F> for U16Air {
    fn execute(&self, pctx: Arc<ProofCtx<F>>, _input_data_path: Option<PathBuf>) -> Vec<usize> {
        let (instance_found, mut instance_id) = pctx.dctx_find_instance(self.airgroup_id, self.air_id);

        if !instance_found {
            instance_id = pctx.add_instance_all(self.airgroup_id, self.air_id);
        }

        self.calculated.store(false, Ordering::Relaxed);
        self.instance_id.store(instance_id as u64, Ordering::SeqCst);
        Vec::new()
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

            pctx.dctx_distribute_multiplicities(&self.multiplicities, instance_id);

            self.calculated.store(true, Ordering::Relaxed);

            if pctx.dctx_is_my_instance(instance_id) {
                let buffer_size = self.num_cols * self.num_rows;
                let mut buffer = create_buffer_fast::<F>(buffer_size);
                buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                    for (col, vec) in self.multiplicities.iter().enumerate() {
                        chunk[col] = F::from_u64(vec[row].swap(0, Ordering::Relaxed));
                    }
                });

                let air_instance = AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, buffer, false));
                pctx.add_air_instance(air_instance, instance_id);
            } else {
                self.multiplicities.par_iter().for_each(|vec| {
                    for vec_row in vec.iter() {
                        vec_row.swap(0, Ordering::Relaxed);
                    }
                });
            }
        }
    }
}
