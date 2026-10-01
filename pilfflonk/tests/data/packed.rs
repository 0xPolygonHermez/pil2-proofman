//! The witness generator of the synthetic fixture of the packing
//! (pilfflonk/docs/README.md#fixtures), `tests/fixtures/packed/packed.pil`, over BN254's `Fr` with
//! `num-bigint`.
//!
//! Its fixed columns are the pilout's: `L1`, `LLAST`, `K[i] = [i+1, i+2, …]` and `S = [3, 4, …]`.
//! Its witness columns, in the pilout's order (stage 1, `colIdx` 0 to 9), are `a[0..7]`, `c` and
//! `b`, and its publics `in1` and `out`:
//!
//! - `a[0]` starts at `in1` and `a[i]` at `10 + i`, and each adds `K[i % 4]` from row to row;
//! - `c` starts at 7, and `c' = c·a[2] + a[3]`; `out` is `c` at the last row;
//! - `b = a[0]·a[1] + S'`, with `S'` the next row's, cyclically.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/packed.rs"] mod packed;`.

use num_bigint::BigUint;
use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness, BN254_R};

/// The fixture's rows: `N = 2^4`.
pub const N_BITS: u32 = 4;

/// The witness of the fixture for the input `in1`: one instance of air 0 of airgroup 0, with the
/// columns `[a[0], …, a[7], c, b]` and the publics `[in1, out]`.
pub fn witness(in1: u64) -> Witness {
    let r = BigUint::parse_bytes(BN254_R.as_bytes(), 10).expect("r in decimal");
    let n = 1usize << N_BITS;
    let k = |i: usize, row: usize| BigUint::from(i + 1 + row);
    let s = |row: usize| BigUint::from(3 + row);

    let mut a: Vec<Vec<BigUint>> =
        (0..8).map(|i| vec![BigUint::from(if i == 0 { in1 } else { 10 + i as u64 })]).collect();
    for (i, column) in a.iter_mut().enumerate() {
        for row in 1..n {
            let next = (&column[row - 1] + k(i % 4, row - 1)) % &r;
            column.push(next);
        }
    }
    let mut c = vec![BigUint::from(7u32)];
    for row in 1..n {
        c.push((&c[row - 1] * &a[2][row - 1] + &a[3][row - 1]) % &r);
    }
    let b: Vec<BigUint> = (0..n).map(|row| (&a[0][row] * &a[1][row] + s((row + 1) % n)) % &r).collect();
    let out = c[n - 1].clone();

    let fr = |v: &BigUint| FrBytes::from_decimal(&v.to_str_radix(10)).expect("a value below r");
    let mut columns: Vec<Vec<FrBytes>> = a.iter().map(|column| column.iter().map(fr).collect()).collect();
    columns.push(c.iter().map(fr).collect());
    columns.push(b.iter().map(fr).collect());
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("ten columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![FrBytes::from_u64(in1), fr(&out)],
        proof_values: vec![],
    }
}
