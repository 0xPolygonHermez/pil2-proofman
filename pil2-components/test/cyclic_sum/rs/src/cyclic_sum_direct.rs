use std::sync::Arc;

use proofman_witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx, ProofmanResult};

use proofman_fields::PrimeField64;

use crate::{CyclicSumDirectAirValues, CyclicSumDirectTrace, TAMPER};

define_wc!(CyclicSumDirect, "CycSumD ");

impl<F: PrimeField64> WitnessComponent<F> for CyclicSumDirect {
    execute!(CyclicSumDirectTrace, 1);

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
            let mut trace = CyclicSumDirectTrace::new_from_vec(buffer_pool.take_buffer())?;
            for i in 0..trace.num_rows() {
                trace[i].x = F::ZERO;
            }

            let mut air_values = CyclicSumDirectAirValues::<F>::new();
            air_values.proved = F::from_u64(5);
            air_values.assumed = F::from_u64(if std::env::var_os(TAMPER).is_some() { 6 } else { 5 });

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace).with_air_values(&mut air_values));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
