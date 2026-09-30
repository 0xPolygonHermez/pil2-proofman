//! The witness generator of the range check fixture (plan M34), `tests/fixtures/range_check/range_check.pil`,
//! over BN254's `Fr` (every value is a small integer). pil-fflonk has no range check example.
//!
//! Its fixed columns are the pilout's: the table `T = [0, 1, …, R − 1]` padded with `R − 1` (and, on
//! the product bus, `LLAST` and the std's `__L1__`). Its witness columns, in the pilout's order
//! (stage 1), are `v` and `mul` on the sum bus, and `v`, `h1` and `h2` on the product bus; it has no
//! public. The stage-2 columns are the prover's, from the std's hints.
//!
//! - `v[i] = (5·i + 3) mod R`, which takes every value of the range, each `N/R` times;
//! - sum bus: `mul[j]` is the number of rows of `v` equal to `j` for `j < R`, and 0 on the rows that
//!   repeat `R − 1`;
//! - product bus: `h1` followed by `h2` is the `2N` values of `v` and `T`, sorted.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/range_check.rs"] mod range_check;`.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

/// The fixture's rows: `N = 2^6`.
pub const N_BITS: u32 = 6;

/// The range: `[0, R)`, `R = 2^4`.
pub const R: u64 = 16;

/// The row of `v` a broken witness takes out of the range.
const OUT_OF_RANGE_ROW: usize = 7;

/// The witness of the fixture on the sum bus: one instance of air 0 of airgroup 0, with the columns
/// `[v, mul]` and no public.
pub fn sum_witness() -> Witness {
    let v = values();
    let mul = multiplicities(&v);
    generate(vec![v, mul])
}

/// [`sum_witness`], but with `v[7] = R`, which no row of `T` has: `mul` counts the others.
pub fn sum_witness_out_of_range() -> Witness {
    let mut v = values();
    v[OUT_OF_RANGE_ROW] = R;
    let mul = multiplicities(&v);
    generate(vec![v, mul])
}

/// The witness of the fixture on the product bus: one instance of air 0 of airgroup 0, with the
/// columns `[v, h1, h2]` and no public.
pub fn prod_witness() -> Witness {
    let v = values();
    let (h1, h2) = sorted(&v);
    generate(vec![v, h1, h2])
}

/// [`prod_witness`], but with `v[7] = R`: `h1` and `h2` are still those of the witness, which go
/// from 0 to `R − 1` in steps of 0 or 1, and no longer the values of `v` and `T`.
pub fn prod_witness_out_of_range() -> Witness {
    let v = values();
    let (h1, h2) = sorted(&v);
    let mut out = v;
    out[OUT_OF_RANGE_ROW] = R;
    generate(vec![out, h1, h2])
}

/// `v`: `(5·i + 3) mod R`.
fn values() -> Vec<u64> {
    let n = 1u64 << N_BITS;
    (0..n).map(|i| (5 * i + 3) % R).collect()
}

/// `T`: `0 … R − 1`, and `R − 1` on the rows after.
fn table() -> Vec<u64> {
    (0..1u64 << N_BITS).map(|j| j.min(R - 1)).collect()
}

/// `mul`: how many rows of `v` equal the row of `T`, counted at its first row for `R − 1`.
fn multiplicities(v: &[u64]) -> Vec<u64> {
    let mut mul = vec![0u64; 1 << N_BITS];
    for &x in v.iter().filter(|&&x| x < R) {
        mul[x as usize] += 1;
    }
    mul
}

/// `(h1, h2)`: the values of `v` and `T` sorted, the first `N` and the last.
fn sorted(v: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut h: Vec<u64> = v.iter().copied().chain(table()).collect();
    h.sort_unstable();
    let h2 = h.split_off(v.len());
    (h, h2)
}

fn generate(columns: Vec<Vec<u64>>) -> Witness {
    let n = 1usize << N_BITS;
    let columns: Vec<Vec<FrBytes>> =
        columns.iter().map(|col| col.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![],
        proof_values: vec![],
    }
}
