//! The witness generator of the fixture `all` (pilfflonk/docs/README.md#fixtures),
//! `tests/fixtures/all/all.pil`: pil-fflonk's `all` (`pil/sm_all/all_main.pil`), whose witness its
//! generators write one state machine after another, over BN254's `Fr`. The same witness for the
//! sum and the product bus. Each state machine's columns are those of its own generator for
//! `N = 2^8`:
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
    witness_of_inputs(INPUTS)
}

/// [`witness`] for the Fibonacci's inputs `inputs` instead of [`INPUTS`], as the witness library of
/// the fixture computes it from its public inputs (pilfflonk/docs/README.md#witness).
pub fn witness_of_inputs(inputs: [u64; 2]) -> Witness {
    witness_of_size(N_BITS, inputs)
}

/// [`witness_of_inputs`] for `2^n_bits` rows instead of `2^N_BITS`: each state machine's columns
/// of its generator for that many rows, as the benchmark (`pilfflonk/bench/`,
/// pilfflonk/docs/performance.md#method) proves `all` at every size.
pub fn witness_of_size(n_bits: u32, inputs: [u64; 2]) -> Witness {
    generate(n_bits, inputs, plookup::columns(n_bits))
}

/// [`witness`], but with the Plookup's wrong multiplicity of `tests/data/plookup.rs`
/// (`witness_with_a_wrong_multiplicity`): the bus does not balance.
pub fn witness_with_a_wrong_multiplicity() -> Witness {
    let mut plookup = plookup::columns(N_BITS);
    plookup::move_a_multiplicity(&mut plookup[4]);
    generate(N_BITS, INPUTS, plookup)
}

/// The witness of `2^n_bits` rows for the Fibonacci's inputs `inputs`, with the Plookup's columns
/// `plookup`.
fn generate(n_bits: u32, inputs: [u64; 2], plookup: Vec<Vec<u64>>) -> Witness {
    let n = 1usize << n_bits;
    let fibonacci = fibonacci::witness(n_bits, inputs);
    let fr = |col: Vec<Vec<u64>>| -> Vec<Vec<FrBytes>> {
        col.iter().map(|c| c.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect()
    };
    let stage1 = &fibonacci.instances[0].stage1;
    let mut columns: Vec<Vec<FrBytes>> = (0..2).map(|col| stage1.column(col).expect("l1 and l2")).collect();
    columns.extend(fr(connection::columns(n_bits)));
    columns.extend(fr(permutation::columns(n_bits)));
    columns.extend(fr(plookup));
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("sixteen columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: fibonacci.publics,
        proof_values: vec![],
    }
}
