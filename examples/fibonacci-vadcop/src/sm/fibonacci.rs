use std::sync::Arc;
use log::debug;

use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, Prover};
use proofman::WitnessManager;
use witness_helpers::{WitnessComponent, WCOpCalculator};

use p3_field::AbstractField;

use crate::{
    FibonacciSquare0Trace, FibonacciVadcopPublicInputs, Module, FIBONACCI_SQUARE_SUBPROOF_ID, FIBONACCI_SQUARE_AIR_IDS,
};

pub struct FibonacciSquare {
    module: Arc<Module>,
}

impl FibonacciSquare {
    pub fn new<F: AbstractField + Copy>(wcm: &mut WitnessManager<F>, module: &Arc<Module>) -> Arc<Self> {
        let fibonacci = Arc::new(Self { module: Arc::clone(module) });
        wcm.register_component(
            Arc::clone(&fibonacci) as Arc<dyn WitnessComponent<F>>,
            Some(FIBONACCI_SQUARE_SUBPROOF_ID),
        );
        fibonacci
    }
    pub fn execute<F: AbstractField + Copy>(&self, pctx: &mut ProofCtx<F>, ectx: &mut ExecutionCtx) {
        let _provers: Vec<Box<dyn Prover<F>>> = Vec::new();
        Self::calculate_fibonacci(
            self,
            FIBONACCI_SQUARE_SUBPROOF_ID[0],
            FIBONACCI_SQUARE_AIR_IDS[0],
            pctx,
            ectx,
            &_provers,
        )
        .unwrap();
    }

    fn calculate_fibonacci<F: AbstractField + Copy>(
        &self,
        air_group_id: usize,
        air_id: usize,
        pctx: &mut ProofCtx<F>,
        ectx: &ExecutionCtx,
        provers: &[Box<dyn Prover<F>>],
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let pi: FibonacciVadcopPublicInputs = pctx.public_inputs.as_slice().into();
        let (module, mut a, mut b, _out) = pi.inner();
        let num_rows = pctx.pilout.get_air(air_group_id, air_id).num_rows();

        let air_instances = pctx.air_instances.lock().unwrap();
        let mut trace = if ectx.discovering {
            None
        } else {
            let air_idx = pctx.find_air_instances(air_group_id, air_id)[0];

            let offset = (provers[air_idx].get_map_offsets("cm1", false) * 8) as usize;
            let mut trace = FibonacciSquare0Trace::from_buffer(&air_instances[air_idx].buffer, num_rows, offset)?;

            trace[0].a = F::from_canonical_u64(a);
            trace[0].b = F::from_canonical_u64(b);
            Some(trace)
        };

        for i in 1..num_rows {
            let tmp = b;
            let result = self.module.calculate_verify(ectx.discovering, vec![a.pow(2) + b.pow(2), module])?;
            (a, b) = (tmp, result[0]);

            if let Some(trace) = &mut trace {
                trace[i].b = F::from_canonical_u64(b);
                trace[i].a = F::from_canonical_u64(a);
            }
        }
        pctx.public_inputs[24..32].copy_from_slice(&b.to_le_bytes());
        Ok(b)
    }
}

impl<F: AbstractField + Copy> WitnessComponent<F> for FibonacciSquare {
    fn calculate_witness(
        &self,
        stage: u32,
        air_instance: &AirInstance,
        pctx: &mut ProofCtx<F>,
        ectx: &ExecutionCtx,
        provers: &[Box<dyn Prover<F>>],
    ) {
        if stage != 1 {
            return;
        }

        debug!("Fiboncci: Calculating witness");
        self.calculate_fibonacci(air_instance.air_group_id, air_instance.air_id, pctx, ectx, provers).unwrap();
    }

    fn suggest_plan(&self, ectx: &mut ExecutionCtx) {
        ectx.instances.push(AirInstance::new(FIBONACCI_SQUARE_SUBPROOF_ID[0], FIBONACCI_SQUARE_AIR_IDS[0], None));
    }
}
