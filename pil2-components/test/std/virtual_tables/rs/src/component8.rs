use std::sync::Arc;

use witness::{define_wc_with_std, execute, WitnessComponent};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx, ProofmanResult};

use fields::PrimeField64;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{Component8Trace, Table8};

const P2_2: u8 = 1 << 2;

define_wc_with_std!(Component8, "Component8");

impl<F: PrimeField64> WitnessComponent<F> for Component8<F> {
    execute!(Component8Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = Component8Trace::new_from_vec(buffer_pool.take_buffer())?;
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // Get the virtual table ID
            let id = self.std_lib.get_virtual_table_id(8)?;

            // Assumes
            for i in 0..num_rows {
                let val1 = rng.random_range(0..P2_2);
                let val2 = rng.random_range(0..P2_2);
                let val3 = rng.random_range(0..3u8);
                trace[i].a[0] = F::from_u8(val1);
                trace[i].a[1] = F::from_u8(val2);
                trace[i].a[2] = F::from_u8(val3);

                // Get the row
                let row = Table8::calculate_table_row(val1, val2, val3);

                // Update the virtual table rows
                self.std_lib.inc_virtual_row(id, row, 1);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
