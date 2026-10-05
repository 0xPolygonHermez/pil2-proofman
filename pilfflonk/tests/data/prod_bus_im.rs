//! The witness generator of the fixture of the product bus with im_col
//! (pilfflonk/docs/README.md#fixtures), `tests/fixtures/prod_bus_im/prod_bus_im.pil`, over BN128's
//! `Fr` (every value is a small integer).
//!
//! Its witness columns, in the pilout's order (stage 1, `colIdx` 0 to 5), are `a`, `c`, `b`, `d`,
//! `sa` and `sb`, and its public `first`; its only fixed column is the std's `__L1__`. The stage-2
//! columns, `gprod` and the std's two `im_low`, are the prover's, from the std's hints `gprod_col`
//! and `im_col`: the witness does not have them.
//!
//! - `a[i] = i + 1` and `c[i] = 3·i + 7`, as `prod_bus.rs`;
//! - `(b, d, sb)` is `(a, c, sa)` rotated by 5 rows: `b[i] = a[i + 5]`, `d[i] = c[i + 5]` and
//!   `sb[i] = sa[i + 5]`, cyclically, a permutation of its rows that keeps each pair's selector;
//! - `sa[i]` is 1 for the rows `i ≡ 0 or 1 (mod 3)`, and 0 for the others;
//! - `first = a[0]`.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/prod_bus_im.rs"] mod prod_bus_im;`.

use proofman_pilfflonk::{AirInstanceRef, FrBytes, InstanceWitness, Stage1Witness, Witness};

/// The fixture's rows: `N = 2^5`.
pub const N_BITS: u32 = 5;

/// The rows `(b, d, sb)` are those of `(a, c, sa)` moved up by.
pub const ROTATION: usize = 5;

/// The witness of the fixture: one instance of air 0 of airgroup 0, with the columns
/// `[a, c, b, d, sa, sb]` and the public `[first]`.
pub fn witness() -> Witness {
    generate(false)
}

/// [`witness`], but with `sb[4]` flipped: the pair `(b[4], d[4])` moves to the other permutation,
/// whose busid is another, and neither balances. The prover computes the std's intermediate columns
/// of this witness as of any other: every row's constraint holds but the bus's last one.
pub fn witness_with_a_pair_in_the_other_permutation() -> Witness {
    generate(true)
}

fn generate(flip: bool) -> Witness {
    let n = 1usize << N_BITS;
    let a: Vec<u64> = (0..n as u64).map(|i| i + 1).collect();
    let c: Vec<u64> = (0..n as u64).map(|i| 3 * i + 7).collect();
    let sa: Vec<u64> = (0..n as u64).map(|i| u64::from(i % 3 != 2)).collect();
    let b: Vec<u64> = (0..n).map(|i| a[(i + ROTATION) % n]).collect();
    let d: Vec<u64> = (0..n).map(|i| c[(i + ROTATION) % n]).collect();
    let mut sb: Vec<u64> = (0..n).map(|i| sa[(i + ROTATION) % n]).collect();
    if flip {
        sb[4] = 1 - sb[4];
    }
    let columns: Vec<Vec<FrBytes>> =
        [&a, &c, &b, &d, &sa, &sb].iter().map(|col| col.iter().map(|&v| FrBytes::from_u64(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("six columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: vec![FrBytes::from_u64(a[0])],
        proof_values: vec![],
    }
}
