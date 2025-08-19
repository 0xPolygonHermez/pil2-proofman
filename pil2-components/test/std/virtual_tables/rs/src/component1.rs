use std::sync::Arc;

use witness::{define_wc_with_std, execute, WitnessComponent};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{Component1Trace, Table1};

define_wc_with_std!(Component1, "Component1");

impl<F: PrimeField64> WitnessComponent<F> for Component1<F> {
    execute!(Component1Trace, 1);

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
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = Component1Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // Get the virtual table ID
            let id = self.std_lib.get_virtual_table_id(1);

            // Assumes
            let t = trace[0].a.len();
            for i in 0..num_rows {
                let val = rng.random_range(0..num_rows) as u64;
                for j in 0..t {
                    trace[i].a[j] = F::from_u64(val);
                }

                // Get the row
                let row = Table1::calculate_table_row(val);

                // Update the virtual table rows
                self.std_lib.inc_virtual_row(id, row, 1);

                // TODO?
                // self.std_lib.inc_virtual_row_one(id, row);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
