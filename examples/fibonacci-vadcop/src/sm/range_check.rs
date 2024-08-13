use log::debug;
use std::{cell::RefCell, sync::Arc};
use common::{AirInstance, ExecutionCtx, ProofCtx, Prover};
use proofman::WitnessManager;
use wchelpers::WitnessComponent;
use p3_goldilocks::Goldilocks;
use p3_field::AbstractField;
use crate::{U8AirTrace, U_8_AIR_AIR_IDS, U_8_AIR_SUBPROOF_ID};

pub struct RangeCheck {
    inputs: RefCell<Vec<u64>>,
}

impl RangeCheck {
    pub fn new<F>(wcm: &mut WitnessManager<F>) -> Arc<Self> {
        let range_check = Arc::new(RangeCheck { inputs: RefCell::new(vec![0; 256]) });
        wcm.register_component(Arc::clone(&range_check) as Arc<dyn WitnessComponent<F>>, Some(U_8_AIR_SUBPROOF_ID));
        range_check
    }
    pub fn new_no_register<F>(wcm: &mut WitnessManager<F>) -> Arc<Self> {
        let range_check = Arc::new(RangeCheck { inputs: RefCell::new(vec![0; 256]) });
        range_check
    }
    pub fn proves(&self, val: u64, minval: u64, maxval: u64) {
        let mut inputs = self.inputs.borrow_mut();
        assert!(val < 256 && val >= 0);
        inputs[(val - minval) as usize] += 1;
        inputs[(maxval - val) as usize] += 1;
    }
}

impl<F> WitnessComponent<F> for RangeCheck {
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

        debug!("RangeCheck  : Calculating witness");
        let (air_idx, air_instance_ctx) = &mut pctx.find_air_instances(U_8_AIR_SUBPROOF_ID[0], U_8_AIR_AIR_IDS[0])[0];

        let interval = air_instance.inputs_interval.unwrap();
        let inputs = &self.inputs.borrow()[interval.0..interval.1];
        let offset = (provers[*air_idx].get_map_offsets("cm1", false) * 8) as usize;
        let num_rows = pctx.pilout.get_air(U_8_AIR_SUBPROOF_ID[0], U_8_AIR_AIR_IDS[0]).num_rows();
        let mut trace = unsafe { Box::new(U8AirTrace::from_buffer(&air_instance_ctx.buffer, num_rows, offset)) };

        for (i, input) in inputs.iter().enumerate() {
            trace.mul[i] = Goldilocks::from_canonical_u64(*input);
        }
    }

    fn suggest_plan(&self, ectx: &mut ExecutionCtx) {
        ectx.instances.push(AirInstance::new(U_8_AIR_SUBPROOF_ID[0], U_8_AIR_AIR_IDS[0], Some((0, 256))));
    }
}
