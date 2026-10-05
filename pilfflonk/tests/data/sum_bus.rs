//! The witness generator of the fixture of the sum bus (pilfflonk/docs/README.md#fixtures),
//! `tests/fixtures/sum_bus/sum_bus.pil`, over BN128's `Fr` (every value is a small integer).
//!
//! Its fixed columns are the pilout's: the table `T = [0, 1, …, N − 1]` and `TT = [0, 1, 4, …]`, its
//! squares (and the std's `__L1__`). Its witness columns, in the pilout's order (stage 1, `colIdx` 0
//! to 2), are `a`, `b` and `mul`, and its public `first`. The stage-2 column `gsum` is the prover's,
//! from the std's hint `gsum_col`: the witness does not have it.
//!
//! - `a[i] = (5·i + 3) mod 13`, which repeats values of the table and leaves others out;
//! - `b[i] = a[i]²`, so that each pair `(a, b)` is a row of the table;
//! - `mul[j]` is the number of rows `i` with `a[i] = j`: how many times row `j` is looked up;
//! - `first = a[0]`.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/sum_bus.rs"] mod sum_bus;`.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

/// The fixture's rows: `N = 2^5`.
pub const N_BITS: u32 = 5;

/// The witness of the fixture: one instance of air 0 of airgroup 0, with the columns `[a, b, mul]`
/// and the public `[first]`.
pub fn witness() -> Witness {
    let n = 1u64 << N_BITS;
    generate((0..n).map(|i| (5 * i + 3) % 13).collect())
}

/// [`witness`], but with `a[7] = N` and `b[7] = N²`, a pair the table does not have: a value looked
/// up that no row of the table provides, which unbalances the bus (`mul` counts the others).
pub fn witness_looking_up_what_is_not_provided() -> Witness {
    let n = 1u64 << N_BITS;
    let mut a: Vec<u64> = (0..n).map(|i| (5 * i + 3) % 13).collect();
    a[7] = n;
    generate(a)
}

/// The witness of the column `a`: `b = a²`, and `mul` counts the rows of `a` of each value of the
/// table.
fn generate(a: Vec<u64>) -> Witness {
    let n = 1usize << N_BITS;
    let b: Vec<u64> = a.iter().map(|v| v * v).collect();
    let mut mul = vec![0u64; n];
    for &v in &a {
        if (v as usize) < n {
            mul[v as usize] += 1;
        }
    }
    let columns: Vec<Vec<FrBytes>> =
        [&a, &b, &mul].iter().map(|col| col.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("three columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![FrBytes::from_u64(a[0])],
        proof_values: vec![],
    }
}
