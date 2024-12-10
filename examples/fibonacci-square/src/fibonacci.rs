use std::sync::Arc;

use proofman_common::{AirInstance, FromTrace, ExecutionCtx, ProofCtx, SetupCtx};
use proofman::{WitnessManager, WitnessComponent};

use p3_field::PrimeField64;

use crate::{
    FibonacciSquareRomTrace, BuildPublicValues, BuildProofValues, FibonacciSquareAirValues, FibonacciSquareTrace,
    Module,
};

pub struct FibonacciSquare<F: PrimeField64> {
    module: Arc<Module<F>>,
}

impl<F: PrimeField64 + Copy> FibonacciSquare<F> {
    const MY_NAME: &'static str = "FiboSqre";

    pub fn new(wcm: Arc<WitnessManager<F>>, module: Arc<Module<F>>) -> Arc<Self> {
        let fibonacci = Arc::new(Self { module });

        wcm.register_component(
            fibonacci.clone(),
            FibonacciSquareTrace::<F>::AIRGROUP_ID,
            FibonacciSquareTrace::<F>::AIR_ID,
        );

        fibonacci
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, _ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        if let Err(e) = Self::calculate_trace(self, pctx, sctx) {
            panic!("Failed to calculate fibonacci: {:?}", e);
        }
    }

    fn calculate_trace(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) -> Result<u64, Box<dyn std::error::Error>> {
        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let mut publics = BuildPublicValues::from_vec_guard(pctx.get_publics());

        let module = F::as_canonical_u64(&publics.module);
        let mut a = F::as_canonical_u64(&publics.in1);
        let mut b = F::as_canonical_u64(&publics.in2);

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

        publics.out = trace[trace.num_rows() - 1].b;

        let mut trace_rom = FibonacciSquareRomTrace::new_zeroes();

        for i in 0..trace_rom.num_rows() {
            trace_rom[i].line = F::from_canonical_u64(3 + i as u64);
            trace_rom[i].flags = F::from_canonical_u64(2 + i as u64);
        }

        let mut proof_values = BuildProofValues::from_vec_guard(pctx.get_proof_values());
        proof_values.value1[0] = F::from_canonical_u64(5);
        proof_values.value2[0] = F::from_canonical_u64(125);

        let mut air_values = FibonacciSquareAirValues::<F>::new();
        air_values.fibo1[0][0] = F::from_canonical_u64(1);
        air_values.fibo1[1][0] = F::from_canonical_u64(2);
        air_values.fibo3 = [F::from_canonical_u64(5), F::from_canonical_u64(5), F::from_canonical_u64(5)];

        let air_instance = AirInstance::new_from_trace(
            sctx.clone(),
            FromTrace::new(&mut trace).with_custom_traces(vec![&mut trace_rom]).with_air_values(&mut air_values),
        );
        pctx.air_instance_repo.add_air_instance(air_instance, Some(0));

        Ok(b)
    }
}

impl<F: PrimeField64 + Copy> WitnessComponent<F> for FibonacciSquare<F> {
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
