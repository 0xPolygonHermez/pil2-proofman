use log::debug;
use std::{cell::RefCell, mem, sync::Arc};

use proofman_common::{AirInstance, ExecutionCtx, ProofCtx};
use proofman::WitnessManager;
use witness_helpers::{WitnessComponent, WCOpCalculator};

use p3_field::AbstractField;
use crate::{FibonacciVadcopPublicInputs, Module0Trace, MODULE_SUBPROOF_ID, MODULE_AIR_IDS};

//use super::RangeCheck;

pub struct Module {
    inputs: RefCell<Vec<(u64, u64)>>,
    //range_check: Arc<RangeCheck>,
}

impl Module {
    pub fn new<F: AbstractField + Copy>(wcm: &mut WitnessManager<F>) -> Arc<Self> {
        let module = Arc::new(Module { inputs: RefCell::new(Vec::new()) });
        wcm.register_component(Arc::clone(&module) as Arc<dyn WitnessComponent<F>>, Some(MODULE_SUBPROOF_ID));

        module
    }
    pub fn new_no_register<F>(_wcm: &mut WitnessManager<F>) -> Arc<Self> {
        Arc::new(Module { inputs: RefCell::new(Vec::new()) })
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
            self.inputs.borrow_mut().push((x, x_mod));
        }

        Ok(vec![x_mod])
    }
}

impl<F: AbstractField + Copy> WitnessComponent<F> for Module {
    fn calculate_witness(&self, stage: u32, _air_instance: &AirInstance, pctx: &mut ProofCtx<F>, ectx: &ExecutionCtx) {
        if stage != 1 {
            return;
        }

        debug!("Module  : Calculating witness");

        let pi: FibonacciVadcopPublicInputs = pctx.public_inputs.as_slice().into();
        let module = pi.module;

        let air_idx = pctx.find_air_instances(MODULE_SUBPROOF_ID[0], MODULE_AIR_IDS[0])[0];
        let interval = ectx.instances[air_idx].inputs_interval.unwrap();

        let inputs = &self.inputs.borrow()[interval.0..interval.1];

        let buffer_allocator = ectx.buffer_allocator.as_ref();
        let buffer_info = buffer_allocator.get_buffer_info("FibonacciSquare".to_owned(), MODULE_AIR_IDS[0]).unwrap();
        let buffer_size = buffer_info.0;
        let offset = buffer_info.1[0];

        let mut buffer = Some(vec![F::default(); buffer_size as usize / mem::size_of::<F>()]);

        let num_rows = pctx.pilout.get_air(MODULE_SUBPROOF_ID[0], MODULE_AIR_IDS[0]).num_rows();
        let mut trace = Module0Trace::map_buffer(buffer.as_mut().unwrap(), num_rows, offset as usize).unwrap();

        for (i, input) in inputs.iter().enumerate() {
            let x = input.0;
            let q = x / module;
            let x_mod = input.1;

            trace[i].x = F::from_canonical_u64(x);
            trace[i].q = F::from_canonical_u64(q);
            trace[i].x_mod = F::from_canonical_u64(x_mod);
            //self.range_check.proves(module - x_mod, 1, 255); //TODO: understnd -1
        }

        for i in inputs.len()..num_rows {
            trace[i].x = F::zero();
            trace[i].q = F::zero();
            trace[i].x_mod = F::zero();
            //self.range_check.proves(module, 1, 255); //TODO: understnd -1
        }

        let mut air_instances = pctx.air_instances.write().unwrap();
        air_instances[air_idx].buffer = buffer;
    }

    fn suggest_plan(&self, ectx: &mut ExecutionCtx) {
        ectx.instances.push(AirInstance::new(
            MODULE_SUBPROOF_ID[0],
            MODULE_AIR_IDS[0],
            Some((0, self.inputs.borrow().len())),
        ));
    }
}
