use log::debug;
use std::{cell::RefCell, sync::Arc};

use std::mem;
use std::slice;

use proofman_common::{AirInstance, ExecutionCtx, ProofCtx, Prover};
use proofman::WitnessManager;
use witness_helpers::{WitnessComponent, WCOpCalculator};

use p3_goldilocks::Goldilocks;
use p3_field::AbstractField;
use crate::{FibonacciVadcopPublicInputs, ModuleTrace, MODULE_SUBPROOF_ID, MODULE_AIR_IDS};

//use super::RangeCheck;

pub struct Module {
    inputs: RefCell<Vec<(u64, u64)>>,
    //range_check: Arc<RangeCheck>,
}

impl Module {
    // TODO: REVIEW
    fn convert_u8_to_slice<F>(ptr: *mut u8, len: usize) -> &'static mut [F] { assert_eq!(len % mem::size_of::<F>(), 0, "Length must be a multiple of element size"); let len_f = len / mem::size_of::<F>(); unsafe { slice::from_raw_parts_mut(ptr as *mut F, len_f) } }

    pub fn new<F>(wcm: &mut WitnessManager<F>) -> Arc<Self> {
        let module = Arc::new(Module { inputs: RefCell::new(Vec::new()) });
        wcm.register_component(Arc::clone(&module) as Arc<dyn WitnessComponent<F>>, Some(MODULE_SUBPROOF_ID));

        module
    }
    pub fn new_no_register<F>(wcm: &mut WitnessManager<F>) -> Arc<Self> {
        let module = Arc::new(Module { inputs: RefCell::new(Vec::new()) });
        module
    }
    /*pub fn new<F>(wcm: &mut WitnessManager<F>, range_check: &Arc<RangeCheck>) -> Arc<Self> {
        let module = Arc::new(Module { inputs: RefCell::new(Vec::new()), range_check: Arc::clone(range_check) });
        wcm.register_component(Arc::clone(&module) as Arc<dyn WitnessComponent<F>>, Some(MODULE_SUBPROOF_ID));
        module
    }
    pub fn new_no_register<F>(wcm: &mut WitnessManager<F>, range_check: &Arc<RangeCheck>) -> Arc<Self> {
        let module = Arc::new(Module { inputs: RefCell::new(Vec::new()), range_check: Arc::clone(range_check) });
        module
    }*/
}

impl WCOpCalculator for Module {
    // 0:x, 1:module
    fn calculate_verify(&self, verify: bool, values: Vec<u64>) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let (x, module) = (values[0], values[1]);

        let x_mod = x % module;

        if verify {
            self.inputs.borrow_mut().push((x.into(), x_mod.into()));
        }

        Ok(vec![x_mod])
    }
}

impl<F> WitnessComponent<F> for Module {
    fn calculate_witness(
        &self,
        stage: u32,
        air_instance: &AirInstance,
        pctx: &mut ProofCtx<F>,
        _ectx: &ExecutionCtx,
        provers: &Vec<Box<dyn Prover<F>>>,
    ) {
        if stage != 1 {
            return;
        }

        debug!("Module  : Calculating witness");

        let pi: FibonacciVadcopPublicInputs = pctx.public_inputs.as_slice().into();
        let module = pi.module as u64;

        let air_idx = pctx.find_air_instances(MODULE_SUBPROOF_ID[0], MODULE_AIR_IDS[0])[0];
        let mut air_instances = pctx.air_instances.lock().unwrap();

        let interval = _ectx.instances[air_idx].inputs_interval.unwrap();
        let inputs = &self.inputs.borrow()[interval.0..interval.1];
        let offset = (provers[air_idx].get_map_offsets("cm1", false) * 8) as usize;
        let buffer = air_instances[air_idx].trace.as_mut().unwrap().get_buffer_ptr();
        let num_rows = pctx.pilout.get_air(MODULE_SUBPROOF_ID[0], MODULE_AIR_IDS[0]).num_rows();
        let mut trace =
                unsafe { Box::new(ModuleTrace::from_buffer(Self::convert_u8_to_slice(buffer, num_rows*8*2), num_rows, offset).unwrap()) };

        for (i, input) in inputs.iter().enumerate() {
            let x = input.0;
            let q = x / module;
            let x_mod = input.1;

            trace[i].x = Goldilocks::from_canonical_u64(x as u64);
            trace[i].q = Goldilocks::from_canonical_u64(q as u64);
            trace[i].x_mod = Goldilocks::from_canonical_u64(x_mod as u64);
            //self.range_check.proves(module - x_mod, 1, 255); //TODO: understnd -1
        }

        for i in inputs.len()..num_rows {
            trace[i].x = Goldilocks::zero();
            trace[i].q = Goldilocks::zero();
            trace[i].x_mod = Goldilocks::zero();
            //self.range_check.proves(module, 1, 255); //TODO: understnd -1
        }
    }

    fn suggest_plan(&self, ectx: &mut ExecutionCtx) {
        ectx.instances.push(AirInstance::new(
            MODULE_SUBPROOF_ID[0],
            MODULE_AIR_IDS[0],
            Some((0, self.inputs.borrow().len())),
        ));
    }
}
