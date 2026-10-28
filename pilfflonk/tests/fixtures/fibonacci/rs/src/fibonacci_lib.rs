use std::path::Path;

use proofman_common::trace::Values;
use proofman_fields::Bn128;
use proofman_pilfflonk::{
    pilfflonk_witness_library, read_public_inputs, AirInstanceRef, InstanceWitness, PilfflonkResult,
    PilfflonkWitnessLibrary, Stage1Witness, Witness, WitnessShape,
};

use crate::{FibonacciPublicValues, FibonacciPublics, FibonacciTrace};

pilfflonk_witness_library!(WitnessLib);

/// The fixture's trace, over `Fr`.
type Trace = FibonacciTrace<Bn128>;

/// The witness of `sm_fibonacci.js`'s `execute` (as `pilfflonk/tests/data/fibonacci.rs`, in
/// `Bn128`): `l2[0] = in1`, `l1[0] = in2` and, for `i ≥ 1`, `l2[i] = l1[i-1]` and
/// `l1[i] = l2[i-1]² + l1[i-1]²`; the publics are `in1`, `in2` and `out = l1[N-1]`. The public
/// inputs are `in1` and `in2`, as decimal strings; either one missing is 0.
impl PilfflonkWitnessLibrary for WitnessLib {
    fn witness(&mut self, shape: &WitnessShape, public_inputs: Option<&Path>) -> PilfflonkResult<Witness> {
        let air = AirInstanceRef { airgroup_id: Trace::AIRGROUP_ID as u64, air_id: Trace::AIR_ID as u64 };
        shape.check_trace("Fibonacci", air, Trace::NUM_ROWS, Trace::ROW_SIZE)?;

        let inputs: FibonacciPublics = read_public_inputs(public_inputs)?;
        let mut trace = Trace::new_zeroes();
        trace[0].l2 = inputs.in1;
        trace[0].l1 = inputs.in2;
        for i in 1..Trace::NUM_ROWS {
            let previous = trace[i - 1];
            trace[i].l2 = previous.l1;
            trace[i].l1 = previous.l2 * previous.l2 + previous.l1 * previous.l1;
        }

        let mut publics = FibonacciPublicValues::<Bn128>::new();
        publics.in1 = inputs.in1;
        publics.in2 = inputs.in2;
        publics.out = trace[Trace::NUM_ROWS - 1].l1;

        let stage1 = Stage1Witness::from_rows(Trace::NUM_ROWS, Trace::ROW_SIZE, &trace.get_buffer(), vec![])?;
        Ok(Witness {
            instances: vec![InstanceWitness { air, stage1 }],
            publics: publics.get_buffer().into_iter().map(Into::into).collect(),
            proof_values: vec![],
        })
    }
}
