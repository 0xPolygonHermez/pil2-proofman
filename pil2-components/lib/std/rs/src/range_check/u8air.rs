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
    prelude::*,
};
use witness::WitnessComponent;
use proofman_common::{AirInstance, BufferPool, ProofCtx, SetupCtx, TraceInfo};
use std::sync::atomic::Ordering;
use crate::AirComponent;

const P2_8: usize = 256;

pub struct U8Air {
    airgroup_id: usize,
    air_id: usize,
    shift: usize,
    mask: usize,
    num_rows: usize,
    num_cols: usize,
    multiplicities: Vec<Vec<AtomicU64>>,
    table_instance_id: AtomicU64,
    calculated: AtomicBool,
    shared_tables: bool,
}

impl<F: PrimeField64> AirComponent<F> for U8Air {
    fn new(
        pctx: &ProofCtx<F>,
        _sctx: &SetupCtx<F>,
        airgroup_id: usize,
        air_id: usize,
        shared_tables: bool,
    ) -> Arc<Self> {
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        // Get and store the ranges
        let num_cols = P2_8.div_ceil(num_rows);
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
            table_instance_id: AtomicU64::new(0),
            calculated: AtomicBool::new(false),
            shared_tables,
        })
    }
}

impl U8Air {
    pub const fn get_global_row(value: u8) -> u64 {
        value as u64
    }

    pub fn get_global_rows(values: &[u8]) -> Vec<u64> {
        values.iter().map(|&v| Self::get_global_row(v)).collect()
    }

    #[inline(always)]
    pub fn update_input(&self, value: u8, multiplicity: u64) {
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

    pub fn update_inputs(&self, values: Vec<u32>) {
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

impl<F: PrimeField64> WitnessComponent<F> for U8Air {
    fn execute(&self, pctx: Arc<ProofCtx<F>>, _input_data_path: Option<PathBuf>) -> Vec<usize> {
        let (instance_found, mut table_instance_id) = pctx.dctx_find_process_table(self.airgroup_id, self.air_id);

        if !instance_found {
            if !self.shared_tables {
                table_instance_id = pctx.add_table_all(self.airgroup_id, self.air_id);
            } else {
                table_instance_id = pctx.add_table(self.airgroup_id, self.air_id);
            }
        }

        self.calculated.store(false, Ordering::Relaxed);
        self.multiplicities.par_iter().for_each(|vec| {
            for v in vec.iter() {
                v.store(0, Ordering::Relaxed);
            }
        });
        self.table_instance_id.store(table_instance_id as u64, Ordering::SeqCst);

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
            let table_instance_id = self.table_instance_id.load(Ordering::Relaxed) as usize;

            let instance_id = pctx.dctx_get_table_instance_idx(table_instance_id);

            if !_instance_ids.contains(&instance_id) {
                return;
            }

            self.calculated.store(true, Ordering::Relaxed);

            if self.shared_tables {
                let owner_idx = pctx.dctx_get_process_owner_instance(instance_id);
                pctx.mpi_ctx.distribute_multiplicities(&self.multiplicities, owner_idx);
            }

            if !self.shared_tables || pctx.dctx_is_my_process_instance(instance_id) {
                let buffer_size = self.num_cols * self.num_rows;
                let mut buffer = create_buffer_fast(buffer_size);
                buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                    for (col, vec) in self.multiplicities.iter().enumerate() {
                        chunk[col] = F::from_u64(vec[row].load(Ordering::Relaxed));
                    }
                });
                let air_instance =
                    AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, self.num_rows, buffer, false));
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
