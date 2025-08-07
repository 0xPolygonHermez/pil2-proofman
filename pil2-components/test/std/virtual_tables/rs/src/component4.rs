use std::sync::Arc;

use witness::{define_wc_with_std, execute, WitnessComponent};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    Rng, SeedableRng,
};

use crate::{Component4Trace, Table4};

define_wc_with_std!(Component4, "Component4");

impl<F: PrimeField64> WitnessComponent<F> for Component4<F>
where
    StandardUniform: Distribution<F>,
{
    execute!(Component4Trace, 1);

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
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);

            let mut trace = Component4Trace::new_from_vec(buffer_pool.take_buffer());
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            // Get the range check IDs
            let range1 = self.std_lib.get_range_id(5, (1 << 8) - 1, Some(true));
            let range2 = self.std_lib.get_range_id(0, (1 << 16) - 1, Some(true));
            let range3 = self.std_lib.get_range_id(0, (1 << 6) - 1, Some(false));

            // Get the virtual table ID
            let id = self.std_lib.get_virtual_table_id(4);

            // Assumes
            let t = trace[0].a.len();
            for i in 0..num_rows {
                let val = rng.random_range(0..num_rows) as u64;
                for j in 0..t {
                    trace[i].a[j] = F::from_u64(val);
                }

                // Update the virtual table rows
                let row = Table4::calculate_table_row(val);
                self.std_lib.inc_virtual_row(id, row, 1);

                let val1 = rng.random_range(5..=(1 << 8) - 1);
                let val2 = rng.random_range(0..=(1 << 16) - 1);
                let val3 = rng.random_range(0..=(1 << 6) - 1);
                trace[i].b[0] = F::from_u64(val1);
                trace[i].b[1] = F::from_u64(val2);
                trace[i].b[2] = F::from_u64(val3);

                // Perform the range checks
                self.std_lib.range_check(range1, val1 as i64, 1);
                self.std_lib.range_check(range2, val2 as i64, 1);
                self.std_lib.range_check(range3, val3 as i64, 1);
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
