use std::sync::{atomic::{AtomicU64, Ordering}, Arc};

use witness::{execute, WitnessComponent};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    rngs::StdRng,
    Rng, SeedableRng,
};

use crate::{Component7Trace, Table7};

pub struct Component7 {
    instance_ids: std::sync::RwLock<Vec<usize>>,
    seed: AtomicU64,
    table7: Arc<Table7>,
}

impl Component7 {
    pub fn new(table7: Arc<Table7>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self { instance_ids: std::sync::RwLock::new(Vec::new()), seed: AtomicU64::new(0), table7 })
    }

    pub fn set_seed(&self, seed: u64) {
        self.seed.store(seed, Ordering::Relaxed);
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Component7

{
    execute!(Component7Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>{
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = Component7Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // Assumes
            let t = trace[0].a.len();
            for i in 0..num_rows {
                let val = rng.random_range(0..num_rows) as u64;
                for j in 0..t {
                    trace[i].a[j] = F::from_u64(val);
                }

                // Get the row
                let row = Table7::calculate_table_row(val);

                // Update the virtual table rows
                self.table7.update_multiplicity(row, 1);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
