#[allow(unused_imports)] // generated for every air shape
mod pil_helpers;

pub use pil_helpers::*;

use std::sync::{Arc, RwLock};

use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use proofman_fields::{Goldilocks, PrimeField64};
use proofman_witness::{execute, witness_library, WitnessComponent, WitnessLibrary, WitnessManager};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        wcm.register_component(DomNative::new());
        Ok(())
    }
}

/// `DOMAIN_NATIVE_TAMPER="<col>:<row>:<value>"`: the cell to overwrite after the witness is filled.
fn tamper() -> Option<(String, usize, u64)> {
    let spec = std::env::var("DOMAIN_NATIVE_TAMPER").ok()?;
    let parts: Vec<&str> = spec.split(':').collect();
    let [col, row, value] = parts[..] else {
        panic!("DOMAIN_NATIVE_TAMPER={spec:?}: expected <col>:<row>:<value>");
    };
    Some((col.to_string(), row.parse().expect("row"), value.parse().expect("value")))
}

pub struct DomNative {
    instance_ids: RwLock<Vec<usize>>,
}

impl DomNative {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for DomNative {
    execute!(DomNativeTrace, 1);

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
            let mut trace = DomNativeTrace::new_from_vec_zeroes(buffer_pool.take_buffer())?;
            // a = PHASE on !Q1 and b = 0 on ODD; both break their constraint on the other rows.
            for i in 0..trace.num_rows() {
                trace[i].a = F::from_u64(if i % 4 == 0 { 100 + i as u64 } else { i as u64 % 4 });
                trace[i].b = F::from_u64(if i % 2 == 1 { 0 } else { 5 });
            }
            if let Some((col, row, value)) = tamper() {
                match col.as_str() {
                    "a" => trace[row].a = F::from_u64(value),
                    "b" => trace[row].b = F::from_u64(value),
                    other => panic!("DOMAIN_NATIVE_TAMPER: DomNative has no column {other:?}"),
                }
            }
            pctx.add_air_instance(AirInstance::new_from_trace(FromTrace::new(&mut trace)), instance_id);
        }
        Ok(())
    }
}
