use std::sync::Arc;

use proofman_witness::{WitnessComponent, execute, define_wc};
use proofman_common::{BufferPool, FromTrace, AirInstance, ProofCtx, SetupCtx, ProofmanResult};

use proofman_fields::PrimeField64;

use crate::{CyclicSumQ0AirValues, CyclicSumQ0Trace, CyclicSumPublicValues};

define_wc!(CyclicSumQ0, "CycSumQ0");

/// Set: every AIR unbalances its bus (here row 3 proves once more than `count` assumes).
pub const TAMPER: &str = "CYCLIC_SUM_TAMPER";

impl<F: PrimeField64> WitnessComponent<F> for CyclicSumQ0 {
    execute!(CyclicSumQ0Trace, 1);

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
            let mut trace = CyclicSumQ0Trace::new_from_vec(buffer_pool.take_buffer())?;
            let num_rows = trace.num_rows();

            // Deterministic, so CPU and GPU proofs can be compared byte for byte.
            let mut count = 0u64;
            for i in 0..num_rows {
                let mul = (i % 3) as u64;
                trace[i].mul = F::from_u64(mul);
                count += mul;
            }
            if std::env::var_os(TAMPER).is_some() {
                trace[3].mul += F::ONE;
            }

            let mut public_values = CyclicSumPublicValues::from_vec_guard(pctx.get_publics());
            public_values.k = F::from_u64(12345);

            let mut air_values = CyclicSumQ0AirValues::<F>::new();
            air_values.count = F::from_u64(count);

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace).with_air_values(&mut air_values));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
        Ok(())
    }
}
