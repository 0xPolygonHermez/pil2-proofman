use std::sync::{Arc, Mutex};

use proofman_common::{FromTrace, AirInstance, ExecutionCtx, ProofCtx, SetupCtx};
use proofman::{WitnessManager, WitnessComponent};
use pil_std_lib::Std;
use p3_field::{AbstractField, PrimeField64};
use num_bigint::BigInt;

use crate::{ModuleTrace};

pub struct Module<F: PrimeField64> {
    inputs: Mutex<Vec<(u64, u64)>>,
    std_lib: Arc<Std<F>>,
}

impl<F: PrimeField64 + AbstractField + Clone + Copy + Default + 'static> Module<F> {
    const MY_NAME: &'static str = "ModuleSM";

    pub fn new(wcm: Arc<WitnessManager<F>>, std_lib: Arc<Std<F>>) -> Arc<Self> {
        let module = Arc::new(Module { inputs: Mutex::new(Vec::new()), std_lib });

        wcm.register_component(module.clone(), ModuleTrace::<F>::AIRGROUP_ID, ModuleTrace::<F>::AIR_ID);

        // Register dependency relations
        module.std_lib.register_predecessor();

        module
    }

    pub fn calculate_module(&self, x: u64, module: u64) -> u64 {
        let x_mod = x % module;

        let mut inputs = self.inputs.lock().unwrap();

        inputs.push((x, x_mod));

        x_mod
    }

    pub fn execute(&self, pctx: Arc<ProofCtx<F>>, _ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.calculate_trace(pctx, sctx);
    }

    fn calculate_trace(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) {
        log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

        let module = F::as_canonical_u64(&pctx.get_public_value("module"));

        let mut trace = ModuleTrace::new_zeroes();

        //range_check(colu: mod - x_mod, min: 1, max: 2**8-1);
        let range = self.std_lib.get_range(BigInt::from(1), BigInt::from((1 << 8) - 1), None);

        let inputs = self.inputs.lock().unwrap();

        for (i, input) in inputs.iter().enumerate() {
            let x = input.0;
            let q = x / module;
            let x_mod = input.1;

            trace[i].x = F::from_canonical_u64(x);
            trace[i].q = F::from_canonical_u64(q);
            trace[i].x_mod = F::from_canonical_u64(x_mod);

            // Check if x_mod is in the range
            self.std_lib.range_check(F::from_canonical_u64(module - x_mod), F::one(), range);
        }

        // Trivial range check for the remaining rows
        for _ in inputs.len()..trace.num_rows() {
            self.std_lib.range_check(F::from_canonical_u64(module), F::one(), range);
        }

        let air_instance = AirInstance::new_from_trace(sctx.clone(), FromTrace::new(&mut trace));
        pctx.air_instance_repo.add_air_instance(air_instance, Some(0));

        self.std_lib.unregister_predecessor(pctx, None);
    }
}

impl<F: PrimeField64 + AbstractField + Copy> WitnessComponent<F> for Module<F> {
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
