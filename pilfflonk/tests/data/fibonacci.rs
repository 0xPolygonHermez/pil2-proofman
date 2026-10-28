//! The witness generator of the Fibonacci fixture, `tests/fixtures/fibonacci/fibonacci.pil`: a
//! port of `execute` in pil-fflonk's `pil/sm_fibonacci/sm_fibonacci.js`, over BN128's `Fr` with
//! `num-bigint`.
//!
//! The JS starts from `l2[0] = input[0]`, `l1[0] = input[1]` and, for `i ≥ 1`, sets
//! `l2[i] = l1[i-1]` and `l1[i] = l2[i-1]² + l1[i-1]²`, returning `l1[N-1]`. PIL1 bound the
//! publics to cells, `in1 = l2(0)`, `in2 = l1(0)` and `out = l1(N-1)`, so its publics are
//! `[input[0], input[1], l1[N-1]]` (`pil-fflonk/runtime/public.json` for `[1, 2]`). The PIL2
//! fixture ties them the same way with `L1·(l2 - in1)`, `L1·(l1 - in2)` and `LLAST·(l1 - out)`.
//!
//! Columns and publics in the pilout's order: witness column 0 is `l1` and 1 is `l2`; public 0 is
//! `in1`, 1 is `in2` and 2 is `out`. The fixed columns `L1` and `LLAST` are the pilout's.
//!
//! Include it with `mod data { pub mod fibonacci; }`.

use num_bigint::BigUint;
use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness, BN128_R};

/// The witness of the fixture for an AIR of `2^n_bits` rows and the inputs `[in1, in2]`: one
/// instance of air 0 of airgroup 0, with the columns `[l1, l2]` and the publics `[in1, in2, out]`.
pub fn witness(n_bits: u32, inputs: [u64; 2]) -> Witness {
    let r = BigUint::parse_bytes(BN128_R.as_bytes(), 10).expect("r in decimal");
    let n = 1usize << n_bits;
    let mut l1 = vec![BigUint::default(); n];
    let mut l2 = vec![BigUint::default(); n];
    l2[0] = BigUint::from(inputs[0]);
    l1[0] = BigUint::from(inputs[1]);
    for i in 1..n {
        l2[i] = l1[i - 1].clone();
        l1[i] = (&l2[i - 1] * &l2[i - 1] + &l1[i - 1] * &l1[i - 1]) % &r;
    }
    let out = l1[n - 1].clone();

    let fr = |v: &BigUint| FrBytes::from_decimal(&v.to_str_radix(10)).expect("a value below r");
    let columns = [l1.iter().map(fr).collect(), l2.iter().map(fr).collect()];
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("two columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![FrBytes::from_u64(inputs[0]), FrBytes::from_u64(inputs[1]), fr(&out)],
        proof_values: vec![],
    }
}
