//! The witness generator of the Permutation fixture (pilfflonk/docs/README.md#fixtures),
//! `tests/fixtures/permutation/permutation.pil`: a port of `execute` in pil-fflonk's
//! `pil/sm_permutation/sm_permutation.js`, over BN128's `Fr` (every value is a small integer). The
//! same witness for the sum and the product bus.
//!
//! Its only fixed column is the std's `__L1__`. Its witness columns, in the pilout's order (stage 1,
//! `colIdx` 0 to 5), are `a`, `b`, `c`, `d`, `selC` and `selD`; it has no public. The stage-2
//! columns are the prover's, from the std's hints.
//!
//! - `a[i] = i² + i + 1`, and `b` is `a` backwards, `b[N − 1 − i] = a[i]` (no constraint reads
//!   either);
//! - an even row `i` has `selC = 1` and `c = a[i]`, and row `i/2` has `selD = 1` and `d = a[i]`: the
//!   pairs `(c, c)` of selC's rows are those `(d, d)` of selD's, in another order;
//! - an odd row `i` has `selC = 0` and `c = 44`, and row `N/2 + (i − 1)/2` has `selD = 0` and
//!   `d = 55`.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/permutation.rs"] mod permutation;`.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

/// The fixture's rows: `N = 2^8`.
pub const N_BITS: u32 = 8;

/// The witness of the fixture: one instance of air 0 of airgroup 0, with the columns
/// `[a, b, c, d, selC, selD]` and no public.
pub fn witness() -> Witness {
    generate(columns(N_BITS))
}

/// [`witness`], but with `d[0] + 1`: the pair `(d[0], d[0])` of selD's row 0 is no pair of selC's
/// rows, and those of selD are no permutation of them.
pub fn witness_not_a_permutation() -> Witness {
    let mut columns = columns(N_BITS);
    assert_eq!(columns[5][0], 1, "row 0 is one of selD's");
    columns[3][0] += 1;
    generate(columns)
}

/// The columns `[a, b, c, d, selC, selD]` of `2^n_bits` rows, those of `execute`. Also the
/// Permutation of `tests/data/all.rs`.
pub fn columns(n_bits: u32) -> Vec<Vec<u64>> {
    let n = 1usize << n_bits;
    let a: Vec<u64> = (0..n as u64).map(|i| i * i + i + 1).collect();
    let b: Vec<u64> = (0..n).map(|i| a[n - 1 - i]).collect();
    let (mut c, mut d, mut sel_c, mut sel_d) = (vec![0u64; n], vec![0u64; n], vec![0u64; n], vec![0u64; n]);
    for i in 0..n {
        if i % 2 == 0 {
            (sel_c[i], c[i]) = (1, a[i]);
            (sel_d[i / 2], d[i / 2]) = (1, a[i]);
        } else {
            (sel_c[i], c[i]) = (0, 44);
            (sel_d[n / 2 + (i - 1) / 2], d[n / 2 + (i - 1) / 2]) = (0, 55);
        }
    }
    vec![a, b, c, d, sel_c, sel_d]
}

fn generate(columns: Vec<Vec<u64>>) -> Witness {
    let n = 1usize << N_BITS;
    let columns: Vec<Vec<FrBytes>> =
        columns.iter().map(|col| col.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("six columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![],
        proof_values: vec![],
    }
}
