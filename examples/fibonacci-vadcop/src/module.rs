use log::debug;
use std::rc::Rc;

use common::{ExecutionCtx, ProofCtx};
use proofman::{trace, WCManager};
use wchelpers::WCComponent;

use p3_goldilocks::Goldilocks;
use p3_field::AbstractField;
use crate::{FibonacciVadcopInputs, ModuleTrace0};

trace!(ModuleTrace { x: Goldilocks, q: Goldilocks, x_mod: Goldilocks });

pub struct Module;

impl Module {
    pub fn new(wcm: &mut WCManager) -> Rc<Self> {
        let module = Rc::new(Module);
        wcm.register_component(Rc::clone(&module) as Rc<dyn WCComponent>);

        module
    }
}

impl WCComponent for Module {
    fn calculate_witness(&self, stage: u32, pctx: &mut ProofCtx, _ectx: &ExecutionCtx) {
        if stage != 1 {
            return;
        }

        debug!("Module   : Calculating witness");
        let air = pctx.pilout.get_air("Module", "Module").unwrap_or_else(|| panic!("Air group not found"));
        let air_instance_ctx = &mut pctx.air_instances[air.air_id()];

        let num_rows: usize = 1 << air.num_rows();

        let mut trace = Box::new(ModuleTrace0::from_buffer(&air_instance_ctx.buffer, num_rows));

        let pi: FibonacciVadcopInputs = pctx.public_inputs.as_slice().into();
        let mut a = pi.a as u64;
        let mut b = pi.b as u64;
        let module = pi.module as u64;

        for i in 0..num_rows {
            let x = a * a + b * b;
            let q = x / module;
            let x_mod = x % module;

            trace.x[i] = Goldilocks::from_canonical_u64(x);
            trace.q[i] = Goldilocks::from_canonical_u64(q);
            trace.x_mod[i] = Goldilocks::from_canonical_u64(x_mod);

            a = b;
            b = x_mod;
        }

        // pctx.air_groups[air.air_group_id()].airs[air.air_id()].add_trace(trace);
    }
}