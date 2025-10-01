use std::sync::Arc;

use witness::{define_wc_with_std, execute, WitnessComponent};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{Component2Trace, Table2_1, Table2_2, Table2_3};

define_wc_with_std!(Component2, "Component2");

impl<F: PrimeField64> WitnessComponent<F> for Component2<F> {
    execute!(Component2Trace, 1);

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if stage == 1 {
            let mut rng = StdRng::seed_from_u64(self.seed.load(Ordering::Relaxed));

            let mut trace = Component2Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // Get the virtual table ID
            let id_1 = self.std_lib.get_virtual_table_id(60)?;
            let id_2 = self.std_lib.get_virtual_table_id(61)?;
            let id_3 = self.std_lib.get_virtual_table_id(62)?;

            // Assumes
            let t = trace[0].a.len();
            for i in 0..num_rows {
                let (val, row, id) = if i % 3 == 0 {
                    let val = rng.random_range(0..Table2_1::N);
                    // Get the row
                    let row = Table2_1::calculate_table_row(val);
                    (val, row, id_1)
                } else if i % 3 == 1 {
                    let val = rng.random_range(Table2_2::N..(2 * Table2_2::N));
                    // Get the row
                    let row = Table2_2::calculate_table_row(val);
                    (val, row, id_2)
                } else {
                    let val = rng.random_range((Table2_3::OFFSET + Table2_3::N)..(Table2_3::OFFSET + 2 * Table2_3::N));
                    // Get the row
                    let row = Table2_3::calculate_table_row(val);
                    (val, row, id_3)
                };

                for j in 0..t {
                    trace[i].a[j] = F::from_u64(val);
                }

                // Update the virtual table rows
                self.std_lib.inc_virtual_row(id, row, 1);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
