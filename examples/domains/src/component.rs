/// Module `$m`: a component that adds one `$Trace`, fills it with `$fill`, then applies `DOMAINS_TAMPER` to `$col`s.
#[macro_export]
macro_rules! domain_component {
    ($m:ident, $Name:ident, $Trace:ident, $air:literal, [$($col:ident),+], $fill:expr) => {
        pub mod $m {
            use std::sync::{Arc, RwLock};

            use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
            use proofman_fields::PrimeField64;
            use proofman_witness::{execute, WitnessComponent};

            use $crate::$Trace;

            pub struct $Name {
                instance_ids: RwLock<Vec<usize>>,
            }

            impl $Name {
                pub fn new() -> Arc<Self> {
                    Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
                }
            }

            impl<F: PrimeField64> WitnessComponent<F> for $Name {
                execute!($Trace, 1);

                fn calculate_witness(
                    &self,
                    stage: u32,
                    pctx: Arc<ProofCtx<F>>,
                    _sctx: Arc<SetupCtx<F>>,
                    instance_ids: &[usize],
                    _n_cores: usize,
                    buffer_pool: &dyn BufferPool<F>,
                ) -> ProofmanResult<()> {
                    if stage != 1 {
                        return Ok(());
                    }
                    for &instance_id in self.instance_ids.read().unwrap().iter().filter(|id| instance_ids.contains(id)) {
                        let mut trace = $Trace::new_from_vec_zeroes(buffer_pool.take_buffer())?;
                        let fill: fn(&mut $Trace<F>) = $fill;
                        fill(&mut trace);
                        $crate::tamper!(trace, $air, $($col),+);
                        pctx.add_air_instance(AirInstance::new_from_trace(FromTrace::new(&mut trace)), instance_id);
                    }
                    Ok(())
                }
            }
        }
    };
}
