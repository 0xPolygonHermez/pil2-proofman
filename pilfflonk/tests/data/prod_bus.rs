//! The witness generator of the fixture of the product bus (pilfflonk/docs/README.md#fixtures),
//! `tests/fixtures/prod_bus/prod_bus.pil`, over BN254's `Fr` (every value is a small integer).
//!
//! Its witness columns, in the pilout's order (stage 1, `colIdx` 0 to 3), are `a`, `c`, `b` and `d`,
//! and its public `first`; its only fixed column is the std's `__L1__`. The stage-2 column `gprod`
//! is the prover's, from the std's hint `gprod_col`: the witness does not have it.
//!
//! - `a[i] = i + 1` and `c[i] = 3·i + 7`;
//! - `(b, d)` is `(a, c)` rotated by 5 rows: `b[i] = a[i + 5]`, `d[i] = c[i + 5]`, cyclically, a
//!   permutation of its rows;
//! - `first = a[0]`.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/prod_bus.rs"] mod prod_bus;`.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

/// The fixture's rows: `N = 2^5`.
pub const N_BITS: u32 = 5;

/// The rows `(b, d)` are those of `(a, c)` moved up by.
pub const ROTATION: usize = 5;

/// The witness of the fixture: one instance of air 0 of airgroup 0, with the columns `[a, c, b, d]`
/// and the public `[first]`.
pub fn witness() -> Witness {
    generate(0)
}

/// [`witness`], but with `b[4] + 1`: the pair `(b[4], d[4])` is no row of `(a, c)`, and `(b, d)` no
/// permutation of it.
pub fn witness_not_a_permutation() -> Witness {
    generate(1)
}

/// The witness with `b[4] + b4_delta`.
fn generate(b4_delta: u64) -> Witness {
    let n = 1usize << N_BITS;
    let a: Vec<u64> = (0..n as u64).map(|i| i + 1).collect();
    let c: Vec<u64> = (0..n as u64).map(|i| 3 * i + 7).collect();
    let mut b: Vec<u64> = (0..n).map(|i| a[(i + ROTATION) % n]).collect();
    let d: Vec<u64> = (0..n).map(|i| c[(i + ROTATION) % n]).collect();
    b[4] += b4_delta;
    let columns: Vec<Vec<FrBytes>> =
        [&a, &c, &b, &d].iter().map(|col| col.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("four columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![FrBytes::from_u64(a[0])],
        proof_values: vec![],
    }
}
