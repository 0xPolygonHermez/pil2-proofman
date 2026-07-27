//! Infrastructure shared by the PCS schemes: parameters, hash family,
//! Merkle helpers, and the RS codeword folding primitives.

use crate::encoding::domain_point;
use crate::error::MlError;
use crate::hypercube::Ext;
use crate::merkle::MerkleTree;
use fields::{Field, Goldilocks, Poseidon2_16};

pub const MERKLE_ARITY: u64 = 4;

/// Hash family for the transcript, Merkle trees and grinding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MlHashFamily {
    Poseidon1,
    Poseidon2,
}

impl MlHashFamily {
    /// The `fields` hash-family id string (for `new_transcript` / `hash_state`).
    pub fn id(&self) -> &'static str {
        match self {
            MlHashFamily::Poseidon1 => "Poseidon1",
            MlHashFamily::Poseidon2 => "Poseidon2",
        }
    }

    /// Parse from `pctx.global_info.hash`.
    pub fn from_id(id: &str) -> Result<Self, MlError> {
        match id {
            "Poseidon1" => Ok(MlHashFamily::Poseidon1),
            "Poseidon2" => Ok(MlHashFamily::Poseidon2),
            other => Err(MlError::Unsupported(format!("hash family '{other}'"))),
        }
    }
}

/// Build a Merkle tree over `leaves` with the given hash family, offloading to
/// the C++ AVX/threaded backend.
#[inline]
pub(crate) fn build_merkle(leaves: &[Vec<Goldilocks>], arity: u64, hash: MlHashFamily) -> MerkleTree {
    MerkleTree::from_ffi(leaves, arity, hash)
}

/// Verify a Merkle path against `root` for the selected hash family.
#[inline]
pub(crate) fn verify_mt_leaf(
    hash: MlHashFamily,
    root: &[Goldilocks; 4],
    path: &[Vec<Goldilocks>],
    idx: u64,
    leaf: &[Goldilocks],
    arity: u64,
) -> bool {
    match hash {
        MlHashFamily::Poseidon1 => fields::verify_mt::<Goldilocks, fields::Poseidon1_16, fields::Poseidon1_16>(
            root,
            &[],
            path,
            idx,
            leaf,
            arity,
            0,
        ),
        MlHashFamily::Poseidon2 => {
            fields::verify_mt::<Goldilocks, Poseidon2_16, Poseidon2_16>(root, &[], path, idx, leaf, arity, 0)
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MlParams {
    /// Log2 of the RS blowup factor (rate = 2^-log_blowup).
    pub log_blowup: usize,
    /// Number of query repetitions (uniform fallback; see `whir_query_schedule`).
    pub n_queries: usize,
    /// Per-block WHIR query counts (length = number of fold rounds). WHIR's
    /// rate improves every block, so later blocks need far fewer queries;
    /// this carries the soundness-driven schedule pinned by the setup.
    /// Empty ⇒ uniform `n_queries` per block.
    pub whir_query_schedule: Vec<usize>,
    /// Per-block WHIR folding factors, in bits — the multilinear analogue of
    /// the univariate stark_struct steps; blocks need not fold uniformly.
    /// Empty ⇒ uniform `WhirParams::default().folding_factor` per block.
    pub whir_fold_schedule: Vec<usize>,
    /// Log2 of the length of the final in-clear polynomial (number of
    /// variables left unfolded); clamped so that at least one fold happens.
    pub log_final_poly_len: usize,
    /// Proof-of-work bits.
    pub grinding_bits: usize,
    /// Univariate-skip length `ℓ`: the zerocheck collapses its first `ℓ`
    /// rounds into one univariate round over a size-`2^ℓ` subgroup.
    /// `0` disables the skip. Must be `≤ n_bits`.
    pub univariate_skip_bits: usize,
    /// Hash family for the transcript, Merkle trees and grinding.
    pub hash: MlHashFamily,
}

impl Default for MlParams {
    // Default parameters for testing-only
    fn default() -> Self {
        Self {
            log_blowup: 2,
            n_queries: 50,
            whir_query_schedule: vec![],
            whir_fold_schedule: vec![],
            log_final_poly_len: 4,
            grinding_bits: 0,
            univariate_skip_bits: 0,
            hash: MlHashFamily::Poseidon2,
        }
    }
}

impl MlParams {
    /// Number of folding steps `L` for an `n_vars`-variate opening (`1 ≤ L ≤ n_vars`).
    pub fn num_folds(&self, n_vars: usize) -> usize {
        assert!(n_vars >= 1);
        n_vars - self.log_final_poly_len.min(n_vars - 1)
    }

    /// Log2 of the level-0 evaluation domain for `n_vars`-variate columns.
    pub fn n0_bits(&self, n_vars: usize) -> usize {
        n_vars + self.log_blowup
    }
}

/// Fold an RS codeword one level: `out(x²) = (1-r)·(f(x)+f(-x))/2 + r·(f(x)-f(-x))/(2x)`,
/// the Basefold/FRI folding identity (fold = partial evaluation of the MLE).
pub fn fold_codeword(vals: &[Ext], n0_bits: usize, level: usize, r: Ext) -> Vec<Ext> {
    let n = vals.len();
    debug_assert_eq!(n, 1usize << (n0_bits - level));
    let half = n / 2;
    let bits = n0_bits - level;

    let two_inv = Goldilocks::ONE_HALF;
    let w_inv = Goldilocks::new(Goldilocks::W_INV[bits]);
    let shift_inv = Goldilocks::new(Goldilocks::SHIFT_INV).exp_power_of_2(level);
    let one_minus_r = Ext::ONE - r;

    let mut out = Vec::with_capacity(half);
    let mut x_inv = shift_inv;
    for j in 0..half {
        let a = vals[j];
        let b = vals[j + half];
        let sum = (a + b) * two_inv;
        let diff = (a - b) * (two_inv * x_inv);
        out.push(one_minus_r * sum + r * diff);
        x_inv *= w_inv;
    }
    out
}

/// Compute the batched MLE table `Σ_j coeff_j · f_j` over the raw columns.
pub fn combine_columns(columns: &[&[Goldilocks]], coeffs: &[Ext]) -> Vec<Ext> {
    let n = columns[0].len();
    let mut out = vec![Ext::ZERO; n];
    for (col, &c) in columns.iter().zip(coeffs.iter()) {
        for (o, &v) in out.iter_mut().zip(col.iter()) {
            *o += c * v;
        }
    }
    out
}

/// Fold a single `(f(x), f(-x))` pair at domain position `j` of `level`.
#[inline]
pub(crate) fn fold_pair(a: Ext, b: Ext, n0_bits: usize, level: usize, j: u64, r: Ext) -> Ext {
    let two_inv = Goldilocks::TWO.inverse();
    let x_inv = domain_point(n0_bits, level, j).inverse();
    (Ext::ONE - r) * ((a + b) * two_inv) + r * ((a - b) * (two_inv * x_inv))
}
