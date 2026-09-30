use std::path::Path;

use proofman_common::trace::Values;
use proofman_fields::{Bn254, Field, QuotientMap};
use proofman_pilfflonk::{
    pilfflonk_witness_library, read_public_inputs, AirInstanceRef, InstanceWitness, PilfflonkResult,
    PilfflonkWitnessLibrary, Stage1Witness, Witness, WitnessShape,
};

use crate::{AllPublicValues, AllPublics, AllTrace};

pilfflonk_witness_library!(WitnessLib);

/// The fixture's trace, over `Fr`.
type Trace = AllTrace<Bn254>;

/// The rows of the Plookup's table of `(i, j)`, `16·16`, and the rows that look it up.
const TABLE_ROWS: usize = 256;
const LOOKUPS: usize = 10;

// The table fits in the trace, and the lookups read the row after theirs.
const _: () = assert!(Trace::NUM_ROWS >= TABLE_ROWS && Trace::NUM_ROWS > LOOKUPS);

/// The witness of pil-fflonk's `all` (`pil/sm_all/all_main.pil`), whose state machines its
/// generators write one after another (as `pilfflonk/tests/data/all.rs`, in `Bn254`), each over its
/// own columns: the Fibonacci from the public inputs `in1` and `in2`, as decimal strings (either one
/// missing is 0), and the Connection, the Permutation and the Plookup, which take no input. The
/// publics are the Fibonacci's, `in1`, `in2` and `out`.
impl PilfflonkWitnessLibrary for WitnessLib {
    fn witness(&mut self, shape: &WitnessShape, public_inputs: Option<&Path>) -> PilfflonkResult<Witness> {
        let air = AirInstanceRef { airgroup_id: Trace::AIRGROUP_ID as u64, air_id: Trace::AIR_ID as u64 };
        shape.check_trace("All", air, Trace::NUM_ROWS, Trace::ROW_SIZE)?;

        let inputs: AllPublics = read_public_inputs(public_inputs)?;
        let mut trace = Trace::new_zeroes();
        fibonacci(&mut trace, inputs.in1, inputs.in2);
        connection(&mut trace);
        permutation(&mut trace);
        plookup(&mut trace);

        let mut publics = AllPublicValues::<Bn254>::new();
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

/// The Fibonacci of `sm_fibonacci.js`'s `execute`: `l2[0] = in1`, `l1[0] = in2` and, for `i ≥ 1`,
/// `l2[i] = l1[i-1]` and `l1[i] = l2[i-1]² + l1[i-1]²`. Its `out` is `l1[N-1]`.
fn fibonacci(trace: &mut Trace, in1: Bn254, in2: Bn254) {
    trace[0].l2 = in1;
    trace[0].l1 = in2;
    for i in 1..Trace::NUM_ROWS {
        let previous = trace[i - 1];
        trace[i].l2 = previous.l1;
        trace[i].l1 = previous.l2 * previous.l2 + previous.l1 * previous.l1;
    }
}

/// The row of the column before it that row `i` of `connection_b` (and of `connection_c`) takes,
/// of `n` rows: its even rows and then its odd ones.
fn rearranged(i: usize, n: usize) -> usize {
    if i < n / 2 {
        2 * i
    } else {
        2 * (i - n / 2) + 1
    }
}

/// The Connection of `sm_connection.js`'s `execute`: `a[i] = i`, and `b` and `c` are `a` and `b`
/// rearranged, `b[i] = a[2·i]` and `b[N/2 + i] = a[2·i + 1]` for `i < N/2`, so that the cells the
/// permutations `S1`, `S2` and `S3` connect are equal.
fn connection(trace: &mut Trace) {
    let n = Trace::NUM_ROWS;
    for i in 0..n {
        trace[i].connection_a = Bn254::from_int(i);
    }
    for i in 0..n {
        trace[i].connection_b = trace[rearranged(i, n)].connection_a;
    }
    for i in 0..n {
        trace[i].connection_c = trace[rearranged(i, n)].connection_b;
    }
}

/// The Permutation of `sm_permutation.js`'s `execute`: `a[i] = i² + i + 1`, and `b` is `a`
/// backwards; an even row `i` has `selC = 1` and `c = a[i]`, and row `i/2` has `selD = 1` and
/// `d = a[i]`; an odd row `i` has `selC = 0` and `c = 44`, and row `N/2 + (i − 1)/2` has `selD = 0`
/// and `d = 55`.
fn permutation(trace: &mut Trace) {
    let n = Trace::NUM_ROWS;
    let a = |i: usize| Bn254::from_int(i * i + i + 1);
    for i in 0..n {
        trace[i].permutation_a = a(i);
        trace[i].permutation_b = a(n - 1 - i);
        if i % 2 == 0 {
            trace[i].permutation_selC = Bn254::ONE;
            trace[i].permutation_c = a(i);
            trace[i / 2].permutation_selD = Bn254::ONE;
            trace[i / 2].permutation_d = a(i);
        } else {
            let j = n / 2 + (i - 1) / 2;
            trace[i].permutation_selC = Bn254::ZERO;
            trace[i].permutation_c = Bn254::from_int(44u64);
            trace[j].permutation_selD = Bn254::ZERO;
            trace[j].permutation_d = Bn254::from_int(55u64);
        }
    }
}

/// The Plookup of `sm_plookup.js`'s `execute`, and its multiplicities: `cc` completes the table
/// `(A, B) = (i, j)` at row `16·i + j` with `i·j` (and `cc[p] = p` on any row past the table's); the
/// rows `p < 10` look up `(a, b', a·b')`, with `sel = 1`, `a = p` and `b = p + 3` (55 at row 0),
/// every other row has `sel = 0` and `a = b = 55` (`b = 10` at row 10), and `mul` counts the lookups
/// of each row of the table.
fn plookup(trace: &mut Trace) {
    let n = Trace::NUM_ROWS;
    let (mut a, mut b) = (vec![55usize; n], vec![55usize; n]);
    for p in 0..LOOKUPS {
        a[p] = p;
        b[p] = if p == 0 { 55 } else { p + 3 };
    }
    b[LOOKUPS] = 10;
    let mut mul = vec![0usize; n];
    for p in 0..LOOKUPS {
        mul[16 * a[p] + b[p + 1]] += 1;
    }
    for p in 0..n {
        let row = &mut trace[p];
        row.plookup_sel = Bn254::from_int(usize::from(p < LOOKUPS));
        row.plookup_a = Bn254::from_int(a[p]);
        row.plookup_b = Bn254::from_int(b[p]);
        row.plookup_cc = Bn254::from_int(if p < TABLE_ROWS { (p / 16) * (p % 16) } else { p });
        row.plookup_mul = Bn254::from_int(mul[p]);
    }
}
