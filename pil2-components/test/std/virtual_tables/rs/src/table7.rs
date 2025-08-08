use std::{mem::MaybeUninit, sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc}};

use witness::{execute, WitnessComponent};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    Rng, SeedableRng,
};

use crate::Table7Trace;

pub struct Table7 {
    instance_ids: std::sync::RwLock<Vec<usize>>,
    multiplicity: Vec<AtomicU64>,
    calculated: AtomicBool,
}

impl Table7 {
    const N: u64 = 512; // 2**9

    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            instance_ids: std::sync::RwLock::new(Vec::new()),
            multiplicity: create_atomic_vec(Table7Trace::<usize>::NUM_ROWS),
            calculated: AtomicBool::new(false),
        })
    }

    pub fn calculate_table_row(val: u64) -> u64 {
        (Self::N - 1) - val
    }

    pub fn update_multiplicity(&self, row: u64, value: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }
        self.multiplicity[row as usize].fetch_add(value, Ordering::Relaxed);
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Table7
where
    StandardUniform: Distribution<F>,
{
    execute!(Table7Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
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

pub fn create_atomic_vec<DT>(size: usize) -> Vec<DT> {
    let mut vec: Vec<MaybeUninit<DT>> = Vec::with_capacity(size);

    unsafe {
        let ptr = vec.as_mut_ptr() as *mut u8;
        std::ptr::write_bytes(ptr, 0, size * std::mem::size_of::<DT>()); // Fast zeroing

        vec.set_len(size);
        std::mem::transmute(vec) // Convert MaybeUninit<Vec> -> Vec<AtomicU64>
    }
}