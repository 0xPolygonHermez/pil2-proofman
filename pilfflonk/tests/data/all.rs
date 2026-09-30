//! The witness generator of the fixture `all` (plan M34), `tests/fixtures/all/all.pil`: pil-fflonk's
//! `all` (`pil/sm_all/all_main.pil`), whose witness its generators write one state machine after
//! another, over BN254's `Fr`. The same witness for the sum and the product bus. Each state machine's
//! columns are those of its own generator for `N = 2^8`:
//!
//! - `l1`, `l2`: the Fibonacci's, `tests/data/fibonacci.rs` for the inputs `[1, 2]`;
//! - `connection_a`, `connection_b`, `connection_c`: `tests/data/connection.rs`;
//! - `permutation_a`, `permutation_b`, `permutation_c`, `permutation_d`, `permutation_selC`,
//!   `permutation_selD`: `tests/data/permutation.rs`;
//! - `plookup_sel`, `plookup_a`, `plookup_b`, `plookup_cc`, `plookup_mul`: `tests/data/plookup.rs`.
//!
//! In the pilout's order (stage 1, `colIdx` 0 to 15), and its publics `in1`, `in2` and `out`, the
//! Fibonacci's: `[1, 2, out]`, pil-fflonk's `runtime/public.json`. The first fifteen columns, all but
//! `plookup_mul`, are PIL1's, in its order.
//!
//! Include it next to the four, as `#[path = ".../pilfflonk/tests/data/all.rs"] mod all;`: it reads
//! them as `super::fibonacci` and so on.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

use super::{connection, fibonacci, permutation, plookup};

/// The fixture's rows: `N = 2^8`, as pil-fflonk's `all_main.pil`.
pub const N_BITS: u32 = 8;

/// The inputs of the Fibonacci, as in pil-fflonk.
pub const INPUTS: [u64; 2] = [1, 2];

/// The witness of the fixture: one instance of air 0 of airgroup 0, with the columns of the four
/// state machines and the publics `[in1, in2, out]`.
pub fn witness() -> Witness {
    generate(plookup::columns(N_BITS))
}

/// [`witness`], but with the Plookup's wrong multiplicity of `tests/data/plookup.rs`
/// (`witness_with_a_wrong_multiplicity`): the bus does not balance.
pub fn witness_with_a_wrong_multiplicity() -> Witness {
    let mut plookup = plookup::columns(N_BITS);
    plookup::move_a_multiplicity(&mut plookup[4]);
    generate(plookup)
}

/// The witness with the Plookup's columns `plookup`.
fn generate(plookup: Vec<Vec<u64>>) -> Witness {
    let n = 1usize << N_BITS;
    let fibonacci = fibonacci::witness(N_BITS, INPUTS);
    let fr = |col: Vec<Vec<u64>>| -> Vec<Vec<FrBytes>> {
        col.iter().map(|c| c.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect()
    };
    let stage1 = &fibonacci.instances[0].stage1;
    let mut columns: Vec<Vec<FrBytes>> = (0..2).map(|col| stage1.column(col).expect("l1 and l2")).collect();
    columns.extend(fr(connection::columns(N_BITS)));
    columns.extend(fr(permutation::columns(N_BITS)));
    columns.extend(fr(plookup));
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("sixteen columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: fibonacci.publics,
        proof_values: vec![],
    }
}
