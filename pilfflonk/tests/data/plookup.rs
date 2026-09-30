//! The witness generator of the Plookup fixture (plan M34), `tests/fixtures/plookup/plookup.pil`: a
//! port of `execute` in pil-fflonk's `pil/sm_plookup/sm_plookup.js`, over BN254's `Fr` (every value
//! is a small integer). The same witness for the sum and the product bus.
//!
//! Its fixed columns are the pilout's: the table `(A, B) = (i, j)` at row `16·i + j`, `SEL` 1 on the
//! table's 256 rows (and the std's `__L1__`). Its witness columns, in the pilout's order (stage 1,
//! `colIdx` 0 to 4), are `sel`, `a`, `b`, `cc` and `mul`; it has no public. The stage-2 columns are
//! the prover's, from the std's hints.
//!
//! - `cc` completes the table, `cc[16·i + j] = i·j` (and `cc[p] = p` on any row past the table's, as
//!   the JS; none at `N = 2^8`);
//! - the rows `p < 10` look up `(a, b', a·b')`, with `sel = 1`, `a[p] = p` and `b[p] = p + 3` (55 at
//!   row 0, which no lookup reads): `(p, p + 4, p·(p + 4))` for `p < 9`, and `(9, 10, 90)` at row 9,
//!   whose `b'` is row 10's `b = 10`;
//! - every other row has `sel = 0`, `a = 55` and `b = 55` (10 at row 10);
//! - `mul[16·a + b']` counts the lookups of that row of the table (PIL1 has no such column, spec
//!   Annex B): the ten are different rows, so it is 0 or 1, as the product bus needs.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/plookup.rs"] mod plookup;`.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

/// The fixture's rows: `N = 2^8`, the table's.
pub const N_BITS: u32 = 8;

/// The rows of the table of `(i, j)`, `16·16`.
const TABLE_ROWS: usize = 256;

/// The witness of the fixture: one instance of air 0 of airgroup 0, with the columns
/// `[sel, a, b, cc, mul]` and no public.
pub fn witness() -> Witness {
    generate(columns(N_BITS))
}

/// [`witness`], but with the multiplicity of the table's row 4, `(0, 4, 0)`, which row 0 looks up,
/// moved to row 5, `(0, 5, 0)`, which nothing does: the multiplicities still add up to the number of
/// lookups, and are still 0 or 1 on the table's rows, but the bus does not balance.
pub fn witness_with_a_wrong_multiplicity() -> Witness {
    let mut columns = columns(N_BITS);
    move_a_multiplicity(&mut columns[4]);
    generate(columns)
}

/// Moves the multiplicity of the table's row 4 to its row 5 in `mul` (see
/// [`witness_with_a_wrong_multiplicity`]). Also the broken Plookup of `tests/data/all.rs`.
pub fn move_a_multiplicity(mul: &mut [u64]) {
    assert_eq!((mul[4], mul[5]), (1, 0), "row 0 looks up the table's row 4, and nothing its row 5");
    mul[4] = 0;
    mul[5] = 1;
}

/// The columns `[sel, a, b, cc, mul]` of `2^n_bits` rows (at least the table's): those of
/// `execute`, and `mul`. Also the Plookup of `tests/data/all.rs`.
pub fn columns(n_bits: u32) -> Vec<Vec<u64>> {
    let n = 1usize << n_bits;
    assert!(n >= TABLE_ROWS, "the table has {TABLE_ROWS} rows");
    let cc: Vec<u64> = (0..n as u64).map(|p| if p < TABLE_ROWS as u64 { (p / 16) * (p % 16) } else { p }).collect();

    let (mut sel, mut a, mut b) = (vec![0u64; n], vec![55u64; n], vec![55u64; n]);
    for p in 0..10 {
        sel[p] = 1;
        a[p] = p as u64;
        b[p] = if p == 0 { 55 } else { p as u64 + 3 };
    }
    b[10] = 10;

    let mut mul = vec![0u64; n];
    for p in (0..n).filter(|&p| sel[p] == 1) {
        let next = b[(p + 1) % n];
        assert!(a[p] < 16 && next < 16, "row {p} looks up a row of the table");
        mul[(16 * a[p] + next) as usize] += 1;
    }
    vec![sel, a, b, cc, mul]
}

fn generate(columns: Vec<Vec<u64>>) -> Witness {
    let n = 1usize << N_BITS;
    let columns: Vec<Vec<FrBytes>> =
        columns.iter().map(|col| col.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("five columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![],
        proof_values: vec![],
    }
}
