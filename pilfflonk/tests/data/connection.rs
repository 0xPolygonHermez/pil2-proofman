//! The witness generator of the Connection fixture (pilfflonk/docs/README.md#fixtures),
//! `tests/fixtures/connection/connection.pil`: a port of `execute` in pil-fflonk's
//! `pil/sm_connection/sm_connection.js`, over BN254's `Fr` (every value is a small integer). The same
//! witness for the sum and the product bus.
//!
//! Its fixed columns are the pilout's: the permutations `S1`, `S2` and `S3`, the std's `ID` and
//! `__L1__`. Its witness columns, in the pilout's order (stage 1, `colIdx` 0 to 2), are `a`, `b` and
//! `c`; it has no public. The stage-2 columns are the prover's, from the std's hints.
//!
//! - `a[i] = i`;
//! - `b` is `a`'s even rows and then its odd ones, `b[i] = a[2·i]` and `b[N/2 + i] = a[2·i + 1]`
//!   for `i < N/2`, and `c` is `b` the same way: the cells the swaps of `S1`, `S2` and `S3` connect
//!   (`sm_connection.js`, `buildConstants`), `a[i]` with `b[j]` and `b[i]` with `c[j]` for
//!   `j = i/2` (even `i`) or `N/2 + (i − 1)/2` (odd), are equal.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/connection.rs"] mod connection;`.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

/// The fixture's rows: `N = 2^10`, as pil-fflonk's `connection_main.pil`.
pub const N_BITS: u32 = 10;

/// The witness of the fixture: one instance of air 0 of airgroup 0, with the columns `[a, b, c]` and
/// no public.
pub fn witness() -> Witness {
    generate(columns(N_BITS))
}

/// [`witness`], but [`disconnect`]ed.
pub fn witness_not_connected() -> Witness {
    let mut columns = columns(N_BITS);
    disconnect(&mut columns);
    generate(columns)
}

/// `c[1] + 1` in the columns `[a, b, c]`: `c[1]` is connected to `b[2]` (and `a[4]`), and no longer
/// equal to them. Also the broken Connection of `tests/data/mixed_bus.rs`.
pub fn disconnect(columns: &mut [Vec<u64>]) {
    columns[2][1] += 1;
}

/// The columns `[a, b, c]` of `2^n_bits` rows, those of `execute`. Also the Connection of
/// `tests/data/all.rs`.
pub fn columns(n_bits: u32) -> Vec<Vec<u64>> {
    let n = 1usize << n_bits;
    let a: Vec<u64> = (0..n as u64).collect();
    let rearranged =
        |x: &[u64]| -> Vec<u64> { (0..n).map(|i| if i < n / 2 { x[2 * i] } else { x[2 * (i - n / 2) + 1] }).collect() };
    let b = rearranged(&a);
    let c = rearranged(&b);
    vec![a, b, c]
}

fn generate(columns: Vec<Vec<u64>>) -> Witness {
    let n = 1usize << N_BITS;
    let columns: Vec<Vec<FrBytes>> =
        columns.iter().map(|col| col.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("three columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![],
        proof_values: vec![],
    }
}
