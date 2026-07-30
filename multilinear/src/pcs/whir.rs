//! WHIR multilinear PCS — Arnon, Chiesa, Fenzi, Yogev (2024),
//! "WHIR: Reed–Solomon Proximity Testing with Super-Fast Verification".
//!
//! Proves the single batched evaluation `Φ̃(u) = σ` (`Σ_b Φ(b)·eq(b,u) = σ`)
//! behind [`MlPcs`], where `Φ = Σ_j δ^j f_j` is the random combination of the
//! committed base-field columns. The opening is a chain of `R` **blocks**; each
//! block runs `k` degree-2 sumcheck rounds (folding `k` variables), then applies
//! the STIR domain-shift and accumulates constraints:
//!
//! - **STIR re-encode.** The folded polynomial `g_i` (the bound MLE) is
//!   re-encoded and committed over a domain **halved only once** (not `2^k`),
//!   so the RS rate drops by `2^{1-k}` each block — the mechanism that lets a
//!   sound protocol use fewer queries at large n. The blowup therefore *grows*
//!   `blowup_i = log_blowup_0 + i·(k-1)`.
//! - **Out-of-domain (OOD) sampling.** One random multilinear point `z_0` per
//!   block; the prover reveals `y_0 = ĝ_i(z_0)`. Kernel: `eq(z_0)`.
//! - **In-domain queries.** `t` field points `z_j` of the previous oracle's
//!   folded domain; `y_j = Fold(f_{i-1}, α⃗)(z_j) = g_i^uni(z_j)`, the
//!   value-to-coefficient univariate eval. Kernel: the **power** kernel
//!   `pow(z_j)[b] = z_j^{index(b)}` (NOT `eq`).
//! - **Step-6 accumulation.** OOD + queries are batched with `γ` powers into the
//!   running weight `W` and claim, so the *next* block's sumcheck enforces them.
//!
//! The weight `W(X⃗) = Σ_terms coeff·kernel(X⃗)` is carried in the degree-2
//! [`ProductOracle`]'s `b` table; the interleaved sumcheck runs continuously
//! over all `n` variables (blocks of `k` + a `n − R·k` tail), ending in the
//! single check `claim == Φ̃(rs)·W(rs)`.
//!
//! Parameters: fixed folding factor `k`; per-block query counts from
//! `MlParams.whir_query_schedule` (the soundness-driven schedule pinned by the
//! setup — the improving rate lets later blocks spend far fewer queries), with
//! uniform `MlParams.n_queries` as the fallback when no schedule is set; and
//! `MlParams.grinding_bits` of proof-of-work per block, ground into the
//! transcript right before the query positions are drawn. Correctness is
//! gated by the roundtrip/tamper tests below.
//!
//! [`MlPcs`]: crate::pcs::MlPcs

use fields::Goldilocks;

use crate::encoding::{domain_point, encode_column_ext, encode_columns_rows};
use crate::eq::{eq_eval, eq_evals, pow_eval};
use crate::error::MlError;
use crate::hypercube::{mle_eval, Ext};
use crate::merkle::MerkleTree;
use crate::pcs::MlPcs;
use crate::sumcheck::{verifier_sumcheck_round, ProductOracle, SumcheckOracle};
use crate::transcript::MlTranscript;

use super::common::{fold_pair, verify_mt_leaf, MlParams, MERKLE_ARITY};

/// WHIR-specific parameters.
#[derive(Debug, Clone, Copy)]
pub struct WhirParams {
    /// Folding factor per block: each block folds `k` variables at once.
    pub folding_factor: usize,
}

impl Default for WhirParams {
    fn default() -> Self {
        Self { folding_factor: 4 }
    }
}

impl WhirParams {
    /// TODO: source the folding factor from `MlParams` / `mlinfo` under calibration.
    pub fn for_params(_params: &MlParams) -> Self {
        Self::default()
    }
}

/// Number of fold blocks `R`: fold `k` variables per block until at most
/// `log_final_poly_len` remain, with `1 ≤ R` and `R·k ≤ n`.
/// Public so setup and soundness tooling mirror the exact prover schedule.
pub fn num_fold_rounds(n: usize, k: usize, log_final_poly_len: usize) -> usize {
    assert!(k >= 1 && k <= n, "folding factor {k} out of range for {n} vars");
    let target = n.saturating_sub(log_final_poly_len);
    let mut r = target.div_ceil(k).max(1);
    while r * k > n {
        r -= 1;
    }
    r.max(1)
}

/// Per-block fold schedule (in bits) for an `n`-variate opening — the
/// multilinear analogue of the univariate stark_struct steps. The pinned
/// `whir_fold_schedule` when non-empty (validated against `n`), else uniform
/// `WhirParams::default().folding_factor` blocks via [`num_fold_rounds`].
/// Public so setup and soundness tooling mirror the exact prover schedule.
pub fn fold_schedule(params: &MlParams, n: usize) -> Result<Vec<usize>, MlError> {
    if params.whir_fold_schedule.is_empty() {
        let k = WhirParams::for_params(params).folding_factor.min(n.max(1));
        return Ok(vec![k; num_fold_rounds(n, k, params.log_final_poly_len)]);
    }
    let schedule = params.whir_fold_schedule.clone();
    let total: usize = schedule.iter().sum();
    if schedule.contains(&0) || total > n {
        return Err(MlError::Malformed(format!("invalid whir_fold_schedule {schedule:?} for {n} vars")));
    }
    Ok(schedule)
}

/// A WHIR commitment to a stage matrix: a Merkle tree over `2^k`-packed
/// fold-fiber leaves, kept as one **flat row-major canonical-`u64` buffer**
/// (leaf `j` = `leaves[j·leaf_width..(j+1)·leaf_width]` =
/// `[cw_c[j + t·half] per column c, for t in 0..2^k]`, `half` = leaf count).
/// The flat layout comes straight from the C++ NTT output with a single
/// parallel gather — no per-column codeword copy exists on the hot path.
pub struct WhirCommitment {
    pub leaves: Vec<u64>,
    pub leaf_width: usize,
    pub n_cols: usize,
    pub tree: MerkleTree,
    pub n0_bits: usize,
    pub folding_factor: usize,
}

impl WhirCommitment {
    pub fn root(&self) -> [Goldilocks; 4] {
        self.tree.root()
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Leaf `j` as field elements (a query opening: `2^k` positions × columns).
    pub fn leaf(&self, j: u64) -> Vec<Goldilocks> {
        let s = j as usize * self.leaf_width;
        self.leaves[s..s + self.leaf_width].iter().map(|&v| Goldilocks::new(v)).collect()
    }

    /// Per-column codewords, reassembled from the packed leaves — setup-time
    /// only (`.mlconst.bin` serialization keeps the column-major format).
    fn codewords(&self) -> Vec<Vec<Goldilocks>> {
        let half = self.leaves.len() / self.leaf_width;
        let n0 = 1usize << self.n0_bits;
        debug_assert_eq!(half << self.folding_factor, n0);
        (0..self.n_cols)
            .map(|c| {
                (0..n0)
                    .map(|i| {
                        let (t, j) = (i / half, i % half);
                        Goldilocks::new(self.leaves[j * self.leaf_width + t * self.n_cols + c])
                    })
                    .collect()
            })
            .collect()
    }

    /// Serialize as a `.mlconst.bin` proving-key artifact. Stores the
    /// column-major codewords + tree + shape (the historical format); the
    /// `2^k`-packed leaves are rebuilt on load.
    pub fn save(&self, path: &std::path::Path) -> Result<(), MlError> {
        let codewords = self.codewords();
        let payload = (self.n0_bits, self.folding_factor, &codewords, &self.tree);
        let bytes = bincode::serde::encode_to_vec(payload, bincode::config::standard())
            .map_err(|e| MlError::Io(format!("serializing WHIR commitment: {e}")))?;
        std::fs::write(path, bytes).map_err(|e| MlError::Io(format!("writing {}: {e}", path.display())))
    }

    /// Load a commitment written by [`save`](Self::save), rebuilding the leaves.
    pub fn load(path: &std::path::Path) -> Result<Self, MlError> {
        let bytes = std::fs::read(path).map_err(|e| MlError::Io(format!("reading {}: {e}", path.display())))?;
        let ((n0_bits, folding_factor, codewords, tree), _): ((usize, usize, Vec<Vec<Goldilocks>>, MerkleTree), _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map_err(|e| MlError::Io(format!("decoding WHIR commitment: {e}")))?;
        let n_cols = codewords.len();
        let (leaves, leaf_width) = pack_base_kary_flat(&codewords, folding_factor);
        Ok(WhirCommitment { leaves, leaf_width, n_cols, tree, n0_bits, folding_factor })
    }
}

/// Pack column-major base-field codewords into flat `2^k`-ary Merkle leaves
/// (see [`WhirCommitment`] for the layout). Load-path / test helper — the
/// commit hot path gathers the same layout from the row-major NTT output.
fn pack_base_kary_flat(codewords: &[Vec<Goldilocks>], k: usize) -> (Vec<u64>, usize) {
    use fields::PrimeField64;
    let ncols = codewords.len();
    let group = 1usize << k;
    let half = codewords[0].len() / group;
    let width = group * ncols;
    let mut out = vec![0u64; codewords[0].len() * ncols];
    crate::par::for_each_chunk_aligned_mut(&mut out, width, |start, chunk| {
        let first = start / width;
        for (local, leaf) in chunk.chunks_exact_mut(width).enumerate() {
            let j = first + local;
            for t in 0..group {
                for (c, cw) in codewords.iter().enumerate() {
                    leaf[t * ncols + c] = cw[j + t * half].as_canonical_u64();
                }
            }
        }
    });
    (out, width)
}

/// Pack an extension codeword into flat `2^k`-ary Merkle leaves (3 canonical
/// `u64` per value): leaf `j` = `[vals[j + t·half].limbs for t in 0..2^k]`.
fn pack_ext_kary_flat(vals: &[Ext], k: usize) -> (Vec<u64>, usize) {
    use fields::PrimeField64;
    let group = 1usize << k;
    let half = vals.len() / group;
    let width = group * 3;
    let mut out = vec![0u64; vals.len() * 3];
    crate::par::for_each_chunk_aligned_mut(&mut out, width, |start, chunk| {
        let first = start / width;
        for (local, leaf) in chunk.chunks_exact_mut(width).enumerate() {
            let j = first + local;
            for t in 0..group {
                let v = &vals[j + t * half];
                for (d, limb) in v.value.iter().enumerate() {
                    leaf[3 * t + d] = limb.as_canonical_u64();
                }
            }
        }
    });
    (out, width)
}

/// Pack an extension codeword into per-leaf vectors (test reference for
/// [`pack_ext_kary_flat`]).
#[cfg(test)]
fn pack_ext_kary(vals: &[Ext], k: usize) -> Vec<Vec<Goldilocks>> {
    let group = 1usize << k;
    let half = vals.len() / group;
    crate::par::map_range(half, |j| {
        let mut leaf = Vec::with_capacity(group * 3);
        for t in 0..group {
            leaf.extend_from_slice(&vals[j + t * half].value);
        }
        leaf
    })
}

#[cfg(test)]
fn unpack_ext_kary(leaf: &[Goldilocks], k: usize) -> Vec<Ext> {
    (0..(1usize << k)).map(|t| Ext::from_array(&leaf[3 * t..3 * t + 3])).collect()
}

/// Extension fold fiber from a flat leaf slice (canonical `u64` limbs).
fn unpack_ext_flat(leaf: &[u64], k: usize) -> Vec<Ext> {
    (0..(1usize << k))
        .map(|t| {
            Ext::from_array(&[
                Goldilocks::new(leaf[3 * t]),
                Goldilocks::new(leaf[3 * t + 1]),
                Goldilocks::new(leaf[3 * t + 2]),
            ])
        })
        .collect()
}

/// Fold an extension codeword by `2^k` in one block (used by tests).
#[cfg(test)]
pub(crate) fn fold_codeword_kary(vals: &[Ext], n0_bits: usize, start_level: usize, r: &[Ext]) -> Vec<Ext> {
    let mut cur = vals.to_vec();
    for (i, &ri) in r.iter().enumerate() {
        cur = super::common::fold_codeword(&cur, n0_bits, start_level + i, ri);
    }
    cur
}

/// Fold a single query's `2^k` fold-fiber values down to one value, mirroring
/// `k` global fold steps at levels `0..k` of a domain of `dom_bits` bits. `leaf`
/// is ordered `[value @ (base_pos + t·stride) for t in 0..2^k]`, `stride =
/// 2^(dom_bits − k)`. Returns `g^uni(domain_point(dom_bits, k, base_pos))`.
fn fold_leaf_query(leaf: &[Ext], dom_bits: usize, base_pos: u64, rs: &[Ext]) -> Ext {
    let k = rs.len();
    debug_assert_eq!(leaf.len(), 1usize << k);
    let stride = 1u64 << (dom_bits - k);
    let mut cur = leaf.to_vec();
    for (lvl, &r) in rs.iter().enumerate() {
        let h = cur.len() / 2;
        let mut next = Vec::with_capacity(h);
        for t in 0..h {
            let j = base_pos + (t as u64) * stride;
            next.push(fold_pair(cur[t], cur[t + h], dom_bits, lvl, j, r));
        }
        cur = next;
    }
    cur[0]
}

/// One query opening of the oracle `f_{i-1}` that block `i` reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WhirOracleOpening {
    /// Block 1 reads `Φ` via its per-stage-matrix `2^k`-packed leaves + paths;
    /// the verifier column-combines them into `Φ`'s fold fiber.
    Stage { leaves: Vec<Vec<Goldilocks>>, paths: Vec<Vec<Vec<Goldilocks>>> },
    /// Block `i ≥ 2` reads the re-encoded oracle `f_{i-1}`: one `2^k` ext fiber + path.
    Fold { leaf: Vec<Ext>, path: Vec<Vec<Goldilocks>> },
}

/// One fold block of the WHIR opening.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WhirBlock {
    /// `k` degree-2 sumcheck round polynomials as `[g(1), g(2)]` (Tweak 1).
    pub round_polys: Vec<[Ext; 2]>,
    /// OOD answer `y_0 = ĝ_i(z_0)`; `None` on the last block (no commit).
    pub ood_answer: Option<Ext>,
    /// Merkle root of the re-encoded oracle `f_i`; `None` on the last block.
    pub fold_root: Option<[Goldilocks; 4]>,
    /// Proof-of-work nonce grinding this block's query indices
    /// (0 when `grinding_bits == 0`).
    pub pow_nonce: u64,
    /// `tᵢ` query openings against `f_{i-1}`.
    pub query_openings: Vec<WhirOracleOpening>,
}

/// Per-block query counts: the schedule pinned in `params`, or uniform
/// `n_queries` when no schedule is set.
pub(crate) fn block_query_counts(params: &MlParams, n_rounds: usize) -> Result<Vec<usize>, MlError> {
    if params.whir_query_schedule.is_empty() {
        Ok(vec![params.n_queries; n_rounds])
    } else if params.whir_query_schedule.len() != n_rounds {
        Err(MlError::Malformed(format!(
            "whir_query_schedule has {} entries, expected {n_rounds} fold rounds",
            params.whir_query_schedule.len()
        )))
    } else if params.whir_query_schedule.contains(&0) {
        // A zero-query block would silently skip its proximity test.
        Err(MlError::Malformed("whir_query_schedule contains a zero-query block".into()))
    } else {
        Ok(params.whir_query_schedule.clone())
    }
}

/// A WHIR opening proof.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WhirOpeningProof {
    pub blocks: Vec<WhirBlock>,
    /// `n − R·k` tail sumcheck rounds on the in-clear final polynomial.
    pub tail_round_polys: Vec<[Ext; 2]>,
    /// The remaining polynomial after all folds, in clear (`2^{n − R·k}` values).
    pub final_poly: Vec<Ext>,
}

/// The WHIR scheme.
pub struct Whir;

impl MlPcs for Whir {
    type Commitment = WhirCommitment;
    type OpeningProof = WhirOpeningProof;

    fn commit(columns: &[&[Goldilocks]], params: &MlParams) -> WhirCommitment {
        assert!(!columns.is_empty());
        let n = columns[0].len();
        assert!(columns.iter().all(|c| c.len() == n));
        let ncols = columns.len();

        // The commitment's leaf packing serves block 1's queries: k = k₁.
        let k = fold_schedule(params, n.trailing_zeros() as usize).expect("invalid fold schedule")[0];
        // Row-major extended matrix straight from the batched C++ NTT.
        let rows = encode_columns_rows(columns, params.log_blowup);
        let n0 = rows.len() / ncols;
        assert!(
            n0 >= (1 << k) && n0.is_multiple_of(1 << k),
            "round-0 domain 2^{} too small for folding factor {k}",
            n0.trailing_zeros()
        );

        // Gather the 2^k fold fibers into flat leaves: leaf `j`, segment `t`
        // is row `j + t·half` — a contiguous `ncols` strip of the NTT output.
        let group = 1usize << k;
        let half = n0 / group;
        let leaf_width = group * ncols;
        let mut leaves = vec![0u64; n0 * ncols];
        crate::par::for_each_chunk_aligned_mut(&mut leaves, leaf_width, |start, chunk| {
            let first = start / leaf_width;
            for (local, leaf) in chunk.chunks_exact_mut(leaf_width).enumerate() {
                let j = first + local;
                for t in 0..group {
                    let row = (j + t * half) * ncols;
                    leaf[t * ncols..(t + 1) * ncols].copy_from_slice(&rows[row..row + ncols]);
                }
            }
        });
        let tree = MerkleTree::from_flat_u64(&leaves, leaf_width, MERKLE_ARITY, params.hash);
        WhirCommitment {
            leaves,
            leaf_width,
            n_cols: ncols,
            tree,
            n0_bits: n0.trailing_zeros() as usize,
            folding_factor: k,
        }
    }

    fn commitment_root(commitment: &WhirCommitment) -> [Goldilocks; 4] {
        commitment.root()
    }

    fn save_commitment(commitment: &WhirCommitment, path: &std::path::Path) -> Result<(), MlError> {
        commitment.save(path)
    }

    fn load_commitment(path: &std::path::Path) -> Result<WhirCommitment, MlError> {
        WhirCommitment::load(path)
    }

    fn open(
        params: &MlParams,
        transcript: &mut MlTranscript,
        phi_table: Vec<Ext>,
        point: &[Ext],
        matrices: &[&WhirCommitment],
    ) -> WhirOpeningProof {
        let n = phi_table.len().trailing_zeros() as usize;
        assert_eq!(point.len(), n);
        let n0_bits = params.n0_bits(n);
        assert!(n0_bits <= 32, "domain 2^{n0_bits} exceeds the 2-adic root table");
        let ks = fold_schedule(params, n).expect("invalid fold schedule");
        let n_rounds = ks.len();
        let fold_vars: usize = ks.iter().sum();
        let schedule = block_query_counts(params, n_rounds).expect("invalid query schedule");

        let mut oracle = ProductOracle::new(phi_table, eq_evals(point));
        let mut blocks: Vec<WhirBlock> = Vec::with_capacity(n_rounds);
        let mut final_poly: Vec<Ext> = Vec::new();

        // The oracle `f_{i-1}` that block `i` queries: `None` = the stage matrices
        // (`Φ`, block 1); `Some((flat leaves, leaf width, tree, dom_bits))` = a
        // re-encoded ext oracle.
        let mut prev_fold: Option<(Vec<u64>, usize, MerkleTree, usize)> = None;

        // Variables folded so far: Σ_{j<i} k_j entering block i.
        let mut folded = 0usize;
        for i in 1..=n_rounds {
            let k = ks[i - 1];
            let prev_dom_bits = if i == 1 { n0_bits } else { prev_fold.as_ref().unwrap().3 };

            // (1) k degree-2 sumcheck rounds.
            let mut round_polys = Vec::with_capacity(k);
            let mut alphas = Vec::with_capacity(k);
            for _ in 0..k {
                let evals = oracle.round_evals();
                transcript.absorb_ext(&evals[1]);
                transcript.absorb_ext(&evals[2]);
                round_polys.push([evals[1], evals[2]]);
                let a = transcript.challenge();
                oracle.bind(a);
                alphas.push(a);
            }
            folded += k;
            let m_i = n - folded;

            // (2) STIR re-encode + commit f_i (blocks < R), or send final_poly (last).
            let mut ood_answer = None;
            let mut fold_root = None;
            let mut committed: Option<(Vec<u64>, usize, MerkleTree, usize)> = None;
            let mut ood_eq: Option<Vec<Ext>> = None;
            if i < n_rounds {
                let blowup_i = params.log_blowup + (folded - i);
                let cw_i = encode_column_ext(&oracle.a, blowup_i);
                let dom_bits_i = n0_bits - i;
                debug_assert_eq!(cw_i.len(), 1usize << dom_bits_i);
                // The new tree's leaf packing serves block i+1's queries: k_{i+1}.
                let (leaves_i, width_i) = pack_ext_kary_flat(&cw_i, ks[i]);
                let tree_i = MerkleTree::from_flat_u64(&leaves_i, width_i, MERKLE_ARITY, params.hash);
                transcript.absorb_root(&tree_i.root());
                fold_root = Some(tree_i.root());
                // OOD on g_i: evaluate as `Σ_b a(b)·eq(b, z)` so the eq table is
                // shared with the batching step below.
                let z = transcript.challenges(m_i);
                let eqz = eq_evals(&z);
                let y0 = crate::par::map_chunks(oracle.a.len(), |s, e| {
                    let mut acc = Ext::ZERO;
                    for (av, ev) in oracle.a[s..e].iter().zip(eqz[s..e].iter()) {
                        acc += *av * *ev;
                    }
                    acc
                })
                .into_iter()
                .sum::<Ext>();
                transcript.absorb_ext(&y0);
                ood_answer = Some(y0);
                ood_eq = Some(eqz);
                committed = Some((leaves_i, width_i, tree_i, dom_bits_i));
            } else {
                final_poly = oracle.a.clone();
                transcript.absorb_exts(&final_poly);
            }

            // (3) Grinding, then batching randomness. The nonce is absorbed
            // before γ, so both γ and the query positions depend on it.
            let pow_nonce = transcript.grind(params.grinding_bits);
            let gamma = transcript.challenge();

            // (4) Queries on f_{i-1}; accumulate the γ-power kernels into `oracle.b`.
            let t = schedule[i - 1];
            let positions = transcript.query_indices(t as u64, (prev_dom_bits - k) as u64);
            let mut query_openings = Vec::with_capacity(positions.len());
            for &p in &positions {
                match &prev_fold {
                    None => {
                        // Block 1: open each stage matrix.
                        let leaves: Vec<Vec<Goldilocks>> = matrices.iter().map(|m| m.leaf(p)).collect();
                        let paths: Vec<Vec<Vec<Goldilocks>>> = matrices.iter().map(|m| m.tree.path(p)).collect();
                        query_openings.push(WhirOracleOpening::Stage { leaves, paths });
                    }
                    Some((leaves, width, tree, _)) => {
                        let s = p as usize * width;
                        query_openings.push(WhirOracleOpening::Fold {
                            leaf: unpack_ext_flat(&leaves[s..s + width], k),
                            path: tree.path(p),
                        });
                    }
                }
            }

            // OOD term (γ^1), then query terms (γ^2, γ^3, …) — same power schedule
            // whether or not this block has an OOD (last block skips the γ^1 slot).
            let mut gpow = gamma;
            if let Some(eqz) = &ood_eq {
                crate::par::zip_for_each_mut(&mut oracle.b, eqz, |bv, ev| *bv += gpow * *ev);
            }
            // Query kernels are power tables `pow(z_j)[i] = z_j^i`, so instead of
            // materializing one 2^{m_i} table per query, accumulate all queries
            // into each chunk of `b` directly, advancing `z_j^i` incrementally.
            let query_weights: Vec<(Ext, Ext)> = positions
                .iter()
                .map(|&p| {
                    gpow *= gamma;
                    (gpow, Ext::from_base(domain_point(prev_dom_bits, k, p)))
                })
                .collect();
            if !query_weights.is_empty() {
                crate::par::for_each_chunk_mut(&mut oracle.b, |start, chunk| {
                    for &(w, zj) in &query_weights {
                        let mut zpow = zj.pow(start as u64) * w;
                        for bv in chunk.iter_mut() {
                            *bv += zpow;
                            zpow *= zj;
                        }
                    }
                });
            }

            blocks.push(WhirBlock { round_polys, ood_answer, fold_root, pow_nonce, query_openings });
            prev_fold = committed;
        }

        // Tail sumcheck rounds on the (already accumulated) oracle.
        let mut tail_round_polys = Vec::with_capacity(n - fold_vars);
        for _ in fold_vars..n {
            let evals = oracle.round_evals();
            transcript.absorb_ext(&evals[1]);
            transcript.absorb_ext(&evals[2]);
            tail_round_polys.push([evals[1], evals[2]]);
            let a = transcript.challenge();
            oracle.bind(a);
        }

        WhirOpeningProof { blocks, tail_round_polys, final_poly }
    }

    #[allow(clippy::too_many_arguments)]
    fn verify(
        params: &MlParams,
        transcript: &mut MlTranscript,
        n_vars: usize,
        sigma: Ext,
        proof: &WhirOpeningProof,
        stage_roots: &[[Goldilocks; 4]],
        stage_n_cols: &[usize],
        column_coeffs: &[Ext],
        point: &[Ext],
    ) -> Result<Vec<Ext>, MlError> {
        let n = n_vars;
        let n0_bits = params.n0_bits(n);
        let total_cols: usize = stage_n_cols.iter().sum();
        let ks = fold_schedule(params, n)?;
        let n_rounds = ks.len();
        let fold_vars: usize = ks.iter().sum();
        let schedule = block_query_counts(params, n_rounds)?;

        // Structural checks.
        if proof.blocks.len() != n_rounds {
            return Err(MlError::Malformed(format!("expected {n_rounds} blocks")));
        }
        if proof.tail_round_polys.len() != n - fold_vars {
            return Err(MlError::Malformed("wrong tail length".into()));
        }
        if proof.final_poly.len() != 1usize << (n - fold_vars) {
            return Err(MlError::Malformed("final polynomial has wrong length".into()));
        }
        if column_coeffs.len() != total_cols {
            return Err(MlError::Malformed("column coefficient count mismatch".into()));
        }

        // Accumulated weight `W = Σ coeff·kernel(rs[offset..])`, seeded with eq(u).
        enum Kind {
            Eq(Vec<Ext>),
            Pow(Ext),
        }
        struct Term {
            offset: usize,
            coeff: Ext,
            kind: Kind,
        }
        let mut weight: Vec<Term> = vec![Term { offset: 0, coeff: Ext::ONE, kind: Kind::Eq(point.to_vec()) }];

        let mut claim = sigma;
        let mut rs: Vec<Ext> = Vec::with_capacity(n);
        // f_{i-1} root for i ≥ 2 (the previous block's fold_root); dom_bits[i-1].
        let mut prev_fold_root: Option<[Goldilocks; 4]> = None;
        let mut prev_dom_bits = n0_bits;

        // Variables folded so far: Σ_{j<i} k_j entering block i.
        let mut folded = 0usize;
        for i in 1..=n_rounds {
            let k = ks[i - 1];
            let group = 1usize << k;
            let block = &proof.blocks[i - 1];
            if block.round_polys.len() != k {
                return Err(MlError::Malformed(format!("block {i}: expected {k} round polys")));
            }

            // (1) k sumcheck rounds.
            let mut alphas = Vec::with_capacity(k);
            for ev in &block.round_polys {
                transcript.absorb_ext(&ev[0]);
                transcript.absorb_ext(&ev[1]);
                let a = transcript.challenge();
                claim = verifier_sumcheck_round(claim, &ev[..], a, rs.len())?;
                rs.push(a);
                alphas.push(a);
            }
            folded += k;
            let m_i = n - folded;

            // (2) commit / final.
            let mut z0: Option<Vec<Ext>> = None;
            if i < n_rounds {
                let root =
                    block.fold_root.ok_or_else(|| MlError::Malformed(format!("block {i}: missing fold root")))?;
                transcript.absorb_root(&root);
                let z = transcript.challenges(m_i);
                let y0 =
                    block.ood_answer.ok_or_else(|| MlError::Malformed(format!("block {i}: missing OOD answer")))?;
                transcript.absorb_ext(&y0);
                z0 = Some(z);
            } else {
                if block.fold_root.is_some() || block.ood_answer.is_some() {
                    return Err(MlError::Malformed("last block must not commit / OOD".into()));
                }
                transcript.absorb_exts(&proof.final_poly);
            }

            // (3) Grinding, then γ — mirrors the prover's transcript order.
            if !transcript.verify_grind(block.pow_nonce, params.grinding_bits) {
                return Err(MlError::FinalCheck(format!("block {i}: proof-of-work check failed")));
            }
            let gamma = transcript.challenge();

            // (4) queries on f_{i-1}: verify openings, fold to y_j, accumulate.
            let t = schedule[i - 1];
            let positions = transcript.query_indices(t as u64, (prev_dom_bits - k) as u64);
            if block.query_openings.len() != positions.len() {
                return Err(MlError::Malformed(format!("block {i}: query count mismatch")));
            }

            // OOD term (γ^1).
            let mut gpow = gamma;
            if let (Some(z), Some(y0)) = (&z0, block.ood_answer) {
                claim += gpow * y0;
                weight.push(Term { offset: folded, coeff: gpow, kind: Kind::Eq(z.clone()) });
            }

            for (&p, opening) in positions.iter().zip(block.query_openings.iter()) {
                let fiber = match opening {
                    WhirOracleOpening::Stage { leaves, paths } => {
                        if i != 1 {
                            return Err(MlError::Malformed(format!("block {i}: unexpected stage opening")));
                        }
                        if leaves.len() != stage_roots.len() || paths.len() != stage_roots.len() {
                            return Err(MlError::Malformed(format!("block {i}: stage opening count mismatch")));
                        }
                        let mut fib = vec![Ext::ZERO; group];
                        let mut col_idx = 0;
                        for (m, ((leaf, path), &n_cols)) in
                            leaves.iter().zip(paths.iter()).zip(stage_n_cols.iter()).enumerate()
                        {
                            if leaf.len() != group * n_cols {
                                return Err(MlError::Malformed(format!("block {i}: stage {m} leaf length")));
                            }
                            if !verify_mt_leaf(params.hash, &stage_roots[m], path, p, leaf, MERKLE_ARITY) {
                                return Err(MlError::MerklePath(format!("block {i}, stage {m}, pos {p}")));
                            }
                            for (tt, fb) in fib.iter_mut().enumerate() {
                                for c in 0..n_cols {
                                    *fb += column_coeffs[col_idx + c] * leaf[tt * n_cols + c];
                                }
                            }
                            col_idx += n_cols;
                        }
                        fib
                    }
                    WhirOracleOpening::Fold { leaf, path } => {
                        if i == 1 {
                            return Err(MlError::Malformed("block 1: unexpected fold opening".into()));
                        }
                        if leaf.len() != group {
                            return Err(MlError::Malformed(format!("block {i}: fold leaf length")));
                        }
                        let root = prev_fold_root.expect("prev fold root set for i>=2");
                        let mut flat = Vec::with_capacity(group * 3);
                        for v in leaf {
                            flat.extend_from_slice(&v.value);
                        }
                        if !verify_mt_leaf(params.hash, &root, path, p, &flat, MERKLE_ARITY) {
                            return Err(MlError::MerklePath(format!("block {i}, fold oracle, pos {p}")));
                        }
                        leaf.clone()
                    }
                };
                let y_j = fold_leaf_query(&fiber, prev_dom_bits, p, &alphas);
                gpow *= gamma;
                claim += gpow * y_j;
                let zj = Ext::from_base(domain_point(prev_dom_bits, k, p));
                weight.push(Term { offset: folded, coeff: gpow, kind: Kind::Pow(zj) });
            }

            prev_fold_root = block.fold_root;
            if i < n_rounds {
                prev_dom_bits = n0_bits - i;
            }
        }

        // Tail sumcheck rounds.
        for ev in &proof.tail_round_polys {
            transcript.absorb_ext(&ev[0]);
            transcript.absorb_ext(&ev[1]);
            let a = transcript.challenge();
            claim = verifier_sumcheck_round(claim, &ev[..], a, rs.len())?;
            rs.push(a);
        }

        // Final identity: claim == Φ̃(rs) · W(rs).
        let phi_final = mle_eval(&proof.final_poly, &rs[fold_vars..]);
        let mut w_final = Ext::ZERO;
        for term in &weight {
            let suffix = &rs[term.offset..];
            let kv = match &term.kind {
                Kind::Eq(z) => eq_eval(z, suffix),
                Kind::Pow(z) => pow_eval(*z, suffix),
            };
            w_final += term.coeff * kv;
        }
        if claim != phi_final * w_final {
            return Err(MlError::FinalCheck("sumcheck claim != Φ̃(rs)·W(rs)".into()));
        }

        Ok(rs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::dot_base_ext;
    use crate::pcs::MlHashFamily;
    use fields::{Field, PrimeField64};
    use rand::{rng, RngExt};

    fn random_col(len: usize) -> Vec<Goldilocks> {
        let mut r = rng();
        (0..len).map(|_| Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64)).collect()
    }

    fn ext(v: u64) -> Ext {
        Ext::from_base(Goldilocks::from_u64(v))
    }

    fn params(log_final: usize, n_queries: usize) -> MlParams {
        MlParams {
            log_blowup: 2,
            n_queries,
            whir_query_schedule: vec![],
            whir_fold_schedule: vec![],
            log_final_poly_len: log_final,
            grinding_bits: 0,
            univariate_skip_bits: 0,
            target_security_bits: 128,
            n_ood_samples: 1,
            hash: MlHashFamily::Poseidon2,
        }
    }

    /// Everything needed to open and (re-)verify a `Φ̃(u)=σ` statement.
    struct Setup {
        mats: Vec<WhirCommitment>,
        roots: Vec<[Goldilocks; 4]>,
        n_cols: Vec<usize>,
        coeffs: Vec<Ext>,
        claims: Vec<Ext>,
        lambda: Vec<Ext>,
        sigma: Ext,
        phi_table: Vec<Ext>,
    }

    fn setup(n: usize, stage_cols: &[Vec<Vec<Goldilocks>>], p: &MlParams) -> Setup {
        let mats: Vec<WhirCommitment> = stage_cols
            .iter()
            .map(|cols| Whir::commit(&cols.iter().map(|c| c.as_slice()).collect::<Vec<_>>(), p))
            .collect();
        let all_cols: Vec<&[Goldilocks]> = stage_cols.iter().flatten().map(|c| c.as_slice()).collect();
        let n_cols: Vec<usize> = stage_cols.iter().map(|c| c.len()).collect();
        let roots: Vec<[Goldilocks; 4]> = mats.iter().map(|m| m.root()).collect();

        // Derive λ, claims, δ, σ deterministically from the transcript (as the caller would).
        let mut tp = MlTranscript::new(p.hash);
        for r in &roots {
            tp.absorb_root(r);
        }
        let lambda = tp.challenges(n);
        let kernel = eq_evals(&lambda);
        let claims: Vec<Ext> = all_cols.iter().map(|c| dot_base_ext(c, &kernel)).collect();
        for v in &claims {
            tp.absorb_ext(v);
        }
        let delta = tp.challenge();
        let mut coeffs = Vec::with_capacity(all_cols.len());
        let mut d = Ext::ONE;
        for _ in 0..all_cols.len() {
            coeffs.push(d);
            d *= delta;
        }
        let sigma: Ext = claims.iter().zip(coeffs.iter()).map(|(&v, &c)| v * c).sum();
        let mut phi_table = vec![Ext::ZERO; 1 << n];
        for (col, &c) in all_cols.iter().zip(coeffs.iter()) {
            for (o, &v) in phi_table.iter_mut().zip(col.iter()) {
                *o += c * v;
            }
        }
        Setup { mats, roots, n_cols, coeffs, claims, lambda, sigma, phi_table }
    }

    /// Run `open` with a transcript aligned to `setup`.
    fn open(n: usize, s: &Setup, p: &MlParams) -> WhirOpeningProof {
        let refs: Vec<&WhirCommitment> = s.mats.iter().collect();
        let mut tp = MlTranscript::new(p.hash);
        for r in &s.roots {
            tp.absorb_root(r);
        }
        let _ = tp.challenges(n);
        for v in &s.claims {
            tp.absorb_ext(v);
        }
        let _ = tp.challenge();
        Whir::open(p, &mut tp, s.phi_table.clone(), &s.lambda, &refs)
    }

    /// Verify with a transcript aligned to `setup`, using `sigma` (overridable for tamper tests).
    fn verify(n: usize, s: &Setup, p: &MlParams, proof: &WhirOpeningProof, sigma: Ext) -> Result<Vec<Ext>, MlError> {
        let mut tv = MlTranscript::new(p.hash);
        for r in &s.roots {
            tv.absorb_root(r);
        }
        let _ = tv.challenges(n);
        for v in &s.claims {
            tv.absorb_ext(v);
        }
        let _ = tv.challenge();
        Whir::verify(p, &mut tv, n, sigma, proof, &s.roots, &s.n_cols, &s.coeffs, &s.lambda)
    }

    /// Full roundtrip, two stage matrices, multi-block (n=12, k=4 ⇒ R=2, no tail).
    #[test]
    fn opening_roundtrip_multiround() {
        let n = 12;
        let p = params(4, 8);
        let stage_cols = vec![
            (0..3).map(|_| random_col(1 << n)).collect::<Vec<_>>(),
            (0..2).map(|_| random_col(1 << n)).collect::<Vec<_>>(),
        ];
        let s = setup(n, &stage_cols, &p);
        let proof = open(n, &s, &p);
        let rs = verify(n, &s, &p, &proof, s.sigma).expect("must verify");
        assert_eq!(rs.len(), n);
    }

    /// Roundtrip with a non-empty tail (n=13, k=4 ⇒ R=2, tail = 13−8 = 5 rounds).
    #[test]
    fn opening_roundtrip_with_tail() {
        let n = 13;
        let p = params(1, 8);
        let stage_cols = vec![(0..2).map(|_| random_col(1 << n)).collect::<Vec<_>>()];
        let s = setup(n, &stage_cols, &p);
        let proof = open(n, &s, &p);
        // R = ceil((13-1)/4)=3 clamped to 3*4=12<=13 ⇒ R=3, tail=1.
        let rs = verify(n, &s, &p, &proof, s.sigma).expect("must verify");
        assert_eq!(rs.len(), n);
    }

    /// Roundtrip with a pinned non-uniform fold schedule (and a tail round).
    #[test]
    fn opening_roundtrip_non_uniform_folds() {
        let n = 12;
        let p = MlParams { whir_fold_schedule: vec![4, 3, 2, 2], ..params(4, 8) };
        let stage_cols = vec![
            (0..3).map(|_| random_col(1 << n)).collect::<Vec<_>>(),
            (0..2).map(|_| random_col(1 << n)).collect::<Vec<_>>(),
        ];
        let s = setup(n, &stage_cols, &p);
        let proof = open(n, &s, &p);
        // Σ folds = 11 ⇒ final poly of 2 values, tail = 1 round.
        assert_eq!(proof.blocks.len(), 4);
        assert_eq!(proof.blocks[0].round_polys.len(), 4);
        assert_eq!(proof.blocks[1].round_polys.len(), 3);
        assert_eq!(proof.final_poly.len(), 2);
        let rs = verify(n, &s, &p, &proof, s.sigma).expect("must verify");
        assert_eq!(rs.len(), n);
    }

    /// Roundtrip with a pinned decreasing per-block query schedule.
    #[test]
    fn opening_roundtrip_query_schedule() {
        let n = 12;
        let p =
            MlParams { whir_query_schedule: vec![8, 4, 2, 2], whir_fold_schedule: vec![4, 3, 2, 2], ..params(4, 8) };
        let stage_cols = vec![
            (0..3).map(|_| random_col(1 << n)).collect::<Vec<_>>(),
            (0..2).map(|_| random_col(1 << n)).collect::<Vec<_>>(),
        ];
        let s = setup(n, &stage_cols, &p);
        let proof = open(n, &s, &p);
        assert_eq!(proof.blocks.iter().map(|b| b.query_openings.len()).collect::<Vec<_>>(), vec![8, 4, 2, 2]);
        let rs = verify(n, &s, &p, &proof, s.sigma).expect("must verify");
        assert_eq!(rs.len(), n);
    }

    /// Bad schedules are rejected: wrong length, or a zero-query block.
    #[test]
    fn invalid_query_schedules_rejected() {
        assert!(block_query_counts(&MlParams { whir_query_schedule: vec![8, 4], ..params(4, 8) }, 3).is_err());
        assert!(block_query_counts(&MlParams { whir_query_schedule: vec![8, 4, 0], ..params(4, 8) }, 3).is_err());
        assert_eq!(block_query_counts(&params(4, 8), 3).unwrap(), vec![8, 8, 8]);
    }

    #[test]
    fn wrong_claim_rejected() {
        let n = 12;
        let p = params(4, 8);
        let stage_cols = vec![vec![random_col(1 << n)]];
        let s = setup(n, &stage_cols, &p);
        let proof = open(n, &s, &p);
        assert!(verify(n, &s, &p, &proof, s.sigma + Ext::ONE).is_err(), "inflated claim must fail");
    }

    #[test]
    fn tampered_ood_answer_rejected() {
        let n = 12;
        let p = params(4, 8);
        let stage_cols = vec![vec![random_col(1 << n)]];
        let s = setup(n, &stage_cols, &p);
        let mut proof = open(n, &s, &p);
        assert!(proof.blocks[0].ood_answer.is_some(), "block 0 must have an OOD answer");
        proof.blocks[0].ood_answer = Some(proof.blocks[0].ood_answer.unwrap() + Ext::ONE);
        assert!(verify(n, &s, &p, &proof, s.sigma).is_err(), "tampered OOD must fail");
    }

    #[test]
    fn tampered_round_poly_rejected() {
        let n = 13;
        let p = params(1, 8);
        let stage_cols = vec![vec![random_col(1 << n)]];
        let s = setup(n, &stage_cols, &p);
        // Tamper a block round poly.
        let mut proof = open(n, &s, &p);
        proof.blocks[0].round_polys[0][0] += Ext::ONE;
        assert!(verify(n, &s, &p, &proof, s.sigma).is_err(), "tampered block round poly must fail");
        // Tamper a tail round poly.
        let mut proof2 = open(n, &s, &p);
        assert!(!proof2.tail_round_polys.is_empty(), "config must have a tail");
        proof2.tail_round_polys[0][1] += Ext::ONE;
        assert!(verify(n, &s, &p, &proof2, s.sigma).is_err(), "tampered tail round poly must fail");
    }

    #[test]
    fn tampered_query_opening_rejected() {
        let n = 12;
        let p = params(4, 16);
        let stage_cols = vec![vec![random_col(1 << n)]];
        let s = setup(n, &stage_cols, &p);
        let mut proof = open(n, &s, &p);
        // Mutate a fold-oracle fiber value in block 2's openings (i≥2 ⇒ Fold).
        let b = proof.blocks.len() - 1; // last block queries f_{R-1} (Fold)
        if let WhirOracleOpening::Fold { leaf, .. } = &mut proof.blocks[b].query_openings[0] {
            leaf[0] += Ext::ONE;
        } else {
            panic!("expected a Fold opening in the last block");
        }
        assert!(verify(n, &s, &p, &proof, s.sigma).is_err(), "tampered query opening must fail");
    }

    /// Corrupt a committed stage codeword after honest claims; queries must catch it.
    #[test]
    fn corrupted_codeword_rejected() {
        let n = 10;
        let p = params(4, 32);
        let col = random_col(1 << n);
        let stage_cols = vec![vec![col.clone()]];
        let mut s = setup(n, &stage_cols, &p);
        // Corrupt the committed codeword and rebuild the tree/leaves + root so paths
        // still verify but the folded values disagree with the honest claim.
        let mut codewords = s.mats[0].codewords();
        let n0 = codewords[0].len();
        for j in 0..n0 / 2 {
            codewords[0][j] += Goldilocks::ONE;
        }
        let k = s.mats[0].folding_factor;
        let (leaves, leaf_width) = pack_base_kary_flat(&codewords, k);
        s.mats[0].tree = MerkleTree::from_flat_u64(&leaves, leaf_width, MERKLE_ARITY, p.hash);
        s.mats[0].leaves = leaves;
        s.mats[0].leaf_width = leaf_width;
        s.roots[0] = s.mats[0].root();
        // Note: `s.phi_table`/claims are the *honest* ones (from the clean columns),
        // so the corrupted commitment must be rejected by the query fold check.
        let proof = open(n, &s, &p);
        assert!(verify(n, &s, &p, &proof, s.sigma).is_err(), "corrupted codeword must fail");
    }

    #[test]
    fn fold_leaf_query_matches_global_kary_fold() {
        for k in 1..=3usize {
            let n = 5;
            let log_blowup = 2;
            let n0_bits = n + log_blowup;
            let col = random_col(1 << n);
            let codeword: Vec<Ext> =
                crate::encoding::encode_column(&col, log_blowup).into_iter().map(Ext::from_base).collect();
            let rs: Vec<Ext> = (0..k).map(|i| ext(3 * i as u64 + 1)).collect();
            let folded = fold_codeword_kary(&codeword, n0_bits, 0, &rs);
            let leaves = pack_ext_kary(&codeword, k);
            assert_eq!(leaves.len(), folded.len());
            for (pos, leaf) in leaves.iter().enumerate() {
                let fiber = unpack_ext_kary(leaf, k);
                let v = fold_leaf_query(&fiber, n0_bits, pos as u64, &rs);
                assert_eq!(v, folded[pos], "k={k} pos={pos}");
            }
        }
    }
}
