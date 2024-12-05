use std::sync::Arc;

use proofman_common::{get_custom_commit_id, AirInstance, ExecutionCtx, ProofCtx, SetupCtx};
use proofman::{WitnessManager, WitnessComponent};

use p3_field::PrimeField;

use crate::{FibonacciSquareTrace, FibonacciSquareRomTrace, Module, FIBONACCI_SQUARE_AIRGROUP_ID, FIBONACCI_SQUARE_AIR_IDS};

pub struct FibonacciSquare<F: PrimeField> {
    module: Arc<Module<F>>,
}

impl<F: PrimeField + Copy> FibonacciSquare<F> {
    const MY_NAME: &'static str = "FiboSqre";

    pub fn new(wcm: Arc<WitnessManager<F>>, module: Arc<Module<F>>) -> Arc<Self> {
        let fibonacci = Arc::new(Self { module });

        wcm.register_component(fibonacci.clone(), Some(FIBONACCI_SQUARE_AIRGROUP_ID), Some(FIBONACCI_SQUARE_AIR_IDS));

        fibonacci
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        if let Err(e) = Self::calculate_trace(self, pctx, ectx, sctx) {
            panic!("Failed to calculate fibonacci: {:?}", e);
        }
    }

    fn calculate_trace(
        &self,
        pctx: Arc<ProofCtx<F>>,
        ectx: Arc<ExecutionCtx>,
        sctx: Arc<SetupCtx>,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let module = pctx.get_public_value("module");
        let mut a = pctx.get_public_value("in1");
        let mut b = pctx.get_public_value("in2");

        let mut trace = FibonacciSquareTrace::new_zeroes();

        trace[0].a = F::from_canonical_u64(a);
        trace[0].b = F::from_canonical_u64(b);

        for i in 1..trace.num_rows() {
            let tmp = b;
            let result = self.module.calculate_module(a.pow(2) + b.pow(2), module);
            (a, b) = (tmp, result);

            trace[i].a = F::from_canonical_u64(a);
            trace[i].b = F::from_canonical_u64(b);
        }

        let mut trace_rom = FibonacciSquareRomTrace::new_zeroes();
        let commit_id = get_custom_commit_id(&sctx, FIBONACCI_SQUARE_AIRGROUP_ID, FIBONACCI_SQUARE_AIR_IDS[0], "rom");

        for i in 0..trace.num_rows() {
            trace_rom[i].line = F::from_canonical_u64(3 + i as u64);
            trace_rom[i].flags = F::from_canonical_u64(2 + i as u64);
        }

        pctx.set_public_value_by_name(b, "out", None);

        pctx.set_proof_value("value1", F::from_canonical_u64(5));
        pctx.set_proof_value("value2", F::from_canonical_u64(125));

        let mut air_instance = AirInstance::from_trace(pctx.clone(), ectx.clone(), sctx.clone(), Some(0), &mut trace);

        air_instance.set_airvalue("FibonacciSquare.fibo1", Some(vec![0]), F::from_canonical_u64(1));
        air_instance.set_airvalue("FibonacciSquare.fibo1", Some(vec![1]), F::from_canonical_u64(2));
        air_instance.set_airvalue_ext("FibonacciSquare.fibo3", None, vec![F::from_canonical_u64(5); 3]);

        air_instance.set_custom_commit_id_buffer(&sctx, trace_rom.detach_buffer(), commit_id);

        Ok(b)
    }
}

impl<F: PrimeField + Copy> WitnessComponent<F> for FibonacciSquare<F> {
    fn calculate_witness(
        &self,
        _stage: u32,
        _air_instance_id: Option<usize>,
        _pctx: Arc<ProofCtx<F>>,
        _ectx: Arc<ExecutionCtx>,
        _sctx: Arc<SetupCtx>,
    ) {
    }
}
