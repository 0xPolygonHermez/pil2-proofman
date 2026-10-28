//! The witness generator of the synthetic fixture of the im pols and the signed offsets
//! (pilfflonk/docs/README.md#fixtures), `tests/fixtures/signed/signed.pil`, over BN128's `Fr` with
//! `num-bigint`.
//!
//! Its fixed columns are the pilout's: `L1`, `LLAST`, `WIN` (1 on rows 1 to N − 3),
//! `K = [1, 2, …, N]` and `P = [5, 6, …, N + 4]`. Its witness columns, in the pilout's order
//! (stage 1, `colIdx` 0 to 5), are `a`, `b`, `c`, `d`, `e` and `g`, and its publics `in1`, `in2`
//! and `out`. Rows are cyclic: row `i + s` is row `(i + s) mod N`.
//!
//! - `a` starts at `in1`, `in2`, `in1 + in2`, and `a[i + 2] = a[i + 1] + a[i]·a[i − 1] + K[i − 1]`
//!   for `i` from 1 to N − 3;
//! - `e[i] = a[i]² + P[i + 2]`;
//! - `b[i] = a[i − 1]·a[i]·a[i + 1]·a[i + 2]·e[i]` on every row (the AIR asks it on WIN's only);
//! - `d[i] = a[i − 1]·b[i + 2]·e[i + 1]`;
//! - `c[0] = a[N − 1]`, and `c[i + 1] = c[i]·b[i]·e[i] + d[i]`; `out` is `c` at the last row;
//! - `g[i] = a[i − 1]`.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/signed.rs"] mod signed;`.

use num_bigint::BigUint;
use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness, BN128_R};

/// The fixture's rows: `N = 2^5`.
pub const N_BITS: u32 = 5;

/// The witness of the fixture for the inputs `[in1, in2]`: one instance of air 0 of airgroup 0,
/// with the columns `[a, b, c, d, e, g]` and the publics `[in1, in2, out]`.
pub fn witness(inputs: [u64; 2]) -> Witness {
    generate(inputs, 0)
}

/// [`witness`], but with `c` started at `a[N − 1] + 1`, continued by its recurrence, and `out` its
/// last row: it breaks one constraint at one row, `L1·(c − 'a)` at row 0, which reads `a` across
/// the wrap.
pub fn witness_broken_across_the_wrap(inputs: [u64; 2]) -> Witness {
    generate(inputs, 1)
}

/// The witness with `c[0] = a[N − 1] + c0_delta`.
fn generate(inputs: [u64; 2], c0_delta: u64) -> Witness {
    let r = BigUint::parse_bytes(BN128_R.as_bytes(), 10).expect("r in decimal");
    let n = 1usize << N_BITS;
    let k = |row: usize| BigUint::from(row + 1);
    let p = |row: usize| BigUint::from(row + 5);
    // Row i + s, cyclically, for s in −1..=2.
    let at = |i: usize, s: isize| (i as isize + s).rem_euclid(n as isize) as usize;

    let mut a = vec![BigUint::from(inputs[0]), BigUint::from(inputs[1]), BigUint::from(inputs[0] + inputs[1])];
    for i in 1..=n - 3 {
        a.push((&a[i + 1] + &a[i] * &a[i - 1] + k(i - 1)) % &r);
    }
    let e: Vec<BigUint> = (0..n).map(|i| (&a[i] * &a[i] + p(at(i, 2))) % &r).collect();
    let b: Vec<BigUint> = (0..n).map(|i| (&a[at(i, -1)] * &a[i] * &a[at(i, 1)] * &a[at(i, 2)] * &e[i]) % &r).collect();
    let d: Vec<BigUint> = (0..n).map(|i| (&a[at(i, -1)] * &b[at(i, 2)] * &e[at(i, 1)]) % &r).collect();
    let mut c = vec![(&a[n - 1] + c0_delta) % &r];
    for i in 0..n - 1 {
        c.push((&c[i] * &b[i] * &e[i] + &d[i]) % &r);
    }
    let out = c[n - 1].clone();
    let g: Vec<BigUint> = (0..n).map(|i| a[at(i, -1)].clone()).collect();

    let fr = |v: &BigUint| FrBytes::from_decimal(&v.to_str_radix(10)).expect("a value below r");
    let columns: Vec<Vec<FrBytes>> = [&a, &b, &c, &d, &e, &g].iter().map(|col| col.iter().map(fr).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("six columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![FrBytes::from_u64(inputs[0]), FrBytes::from_u64(inputs[1]), fr(&out)],
        proof_values: vec![],
    }
}
