use std::path::Path;

use proofman_fields::{Bn254, QuotientMap};
use proofman_pilfflonk::{
    pilfflonk_witness_library, AirInstanceRef, InstanceWitness, PilfflonkResult, PilfflonkWitnessLibrary,
    Stage1Witness, Witness, WitnessShape,
};

use crate::ConnectionTrace;

pilfflonk_witness_library!(WitnessLib);

/// The fixture's trace, over `Fr`.
type Trace = ConnectionTrace<Bn254>;

/// The row of the column before it that row `i` of `b` (and of `c`) takes, of `n` rows: `b` is `a`'s
/// even rows and then its odd ones, and `c` is `b` the same way.
fn rearranged(i: usize, n: usize) -> usize {
    if i < n / 2 {
        2 * i
    } else {
        2 * (i - n / 2) + 1
    }
}

/// The witness of `sm_connection.js`'s `execute` (as `pilfflonk/tests/data/connection.rs`, in
/// `Bn254`): `a[i] = i`, and `b` and `c` are `a` and `b` rearranged, `b[i] = a[2·i]` and
/// `b[N/2 + i] = a[2·i + 1]` for `i < N/2`, so that the cells the permutations `S1`, `S2` and `S3`
/// connect are equal. The program has no publics, and takes no public inputs: it does not read the
/// file it may be given, as the STARK's libraries of programs without publics
/// (`pil2-components/test/connection/rs`) do not read theirs.
impl PilfflonkWitnessLibrary for WitnessLib {
    fn witness(&mut self, shape: &WitnessShape, _public_inputs: Option<&Path>) -> PilfflonkResult<Witness> {
        let air = AirInstanceRef { airgroup_id: Trace::AIRGROUP_ID as u64, air_id: Trace::AIR_ID as u64 };
        shape.check_trace("Connection", air, Trace::NUM_ROWS, Trace::ROW_SIZE)?;

        let n = Trace::NUM_ROWS;
        let mut trace = Trace::new_zeroes();
        for i in 0..n {
            trace[i].a = Bn254::from_int(i as u64);
        }
        for i in 0..n {
            trace[i].b = trace[rearranged(i, n)].a;
        }
        for i in 0..n {
            trace[i].c = trace[rearranged(i, n)].b;
        }

        let stage1 = Stage1Witness::from_rows(n, Trace::ROW_SIZE, &trace.get_buffer(), vec![])?;
        Ok(Witness { instances: vec![InstanceWitness { air, stage1 }], publics: vec![], proof_values: vec![] })
    }
}
