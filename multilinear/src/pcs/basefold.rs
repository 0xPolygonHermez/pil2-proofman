//! Basefold: batched multilinear opening via sumcheck interleaved with FRI folding.
//!
//! The statement proven is a single-point evaluation `f(u) = σ`, i.e. the
//! hypercube sum `Σ_b f(b)·eq(b, u) = σ`, where `f = Σ_j δ^j·f_j` is a random
//! linear combination of the committed columns (this is what gets FRI-folded)
//! and `u` is the point every claim was previously collapsed to by the
//! opening-reduction sumcheck. The caller (the STARK prover/verifier) forms
//! `f`, `u`, `σ`; this module runs the interleaved protocol:
//!
//! - `n` sumcheck rounds of the degree-2 product `f·eq(·,u)` (the eq factor is
//!   handled analytically by [`EqProductOracle`], never materialized); the
//!   round-`t` challenge also folds the codeword of `f` one step (fold =
//!   partial evaluation, the Basefold identity). Folded oracles `f_1 … f_{L−1}`
//!   are Merkle-committed; after `L` folds the remaining polynomial is sent in
//!   clear (`final_poly`, which is simultaneously the partially-bound MLE table
//!   and the coefficient vector of the remaining univariate).
//! - A query phase checking fold consistency at random domain positions,
//!   anchored in the stage commitments at level 0.

use crate::encoding::{domain_point, encode_columns, eval_ext_poly_at_base};
use crate::error::MlError;
use crate::pcs::MlPcs;
use crate::hypercube::{fold_mle, mle_eval, Ext};
use crate::merkle::MerkleTree;
use crate::sumcheck::{eq_product_verifier_sumcheck_round, EqProductOracle, SumcheckOracle};
use crate::transcript::MlTranscript;
use fields::{Field, Goldilocks, Poseidon2_16};

/// Hash used for Merkle trees.
pub type MlHash = Poseidon2_16;
pub const MERKLE_ARITY: u64 = 4;

/// Build a Merkle tree over `leaves`.
#[inline]
pub(crate) fn build_merkle(leaves: &[Vec<Goldilocks>], arity: u64) -> MerkleTree {
    MerkleTree::from_ffi(leaves, arity)
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MlParams {
    /// Log2 of the RS blowup factor (rate = 2^-log_blowup).
    pub log_blowup: usize,
    /// Number of FRI query repetitions.
    pub n_queries: usize,
    /// Log2 of the length of the final in-clear polynomial (number of
    /// variables left unfolded); clamped so that at least one fold happens.
    pub log_final_poly_len: usize,
    /// Proof-of-work bits.
    pub grinding_bits: usize,
    /// Univariate-skip length `ℓ`: the zerocheck collapses its first `ℓ`
    /// rounds into one univariate round over a size-`2^ℓ` subgroup.
    /// `0` disables the skip. Must be `≤ n_bits`.
    pub univariate_skip_bits: usize,
}

impl Default for MlParams {
    // Default parameters for testing-only
    fn default() -> Self {
        Self { log_blowup: 2, n_queries: 50, log_final_poly_len: 4, grinding_bits: 0, univariate_skip_bits: 0 }
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

/// A committed matrix of base-field columns (one commitment per stage).
pub struct CommittedMatrix {
    /// RS codewords, one per column, natural evaluation order (length `2^n0_bits`).
    pub codewords: Vec<Vec<Goldilocks>>,
    /// Pair-packed leaves: `leaves[j] = [cols at j..., cols at j + N/2...]`.
    pub leaves: Vec<Vec<Goldilocks>>,
    pub tree: MerkleTree,
    pub n0_bits: usize,
}

impl CommittedMatrix {
    pub fn root(&self) -> [Goldilocks; 4] {
        self.tree.root()
    }

    pub fn n_cols(&self) -> usize {
        self.codewords.len()
    }

    /// Serialize the committed matrix as a proving-key artifact. Only the
    /// codewords and the Merkle tree are stored — the `leaves` (a trivial
    /// pair-packing of the codewords) are rebuilt on [`load`](Self::load), so
    /// no NTT encode or Merkle hashing is repeated at prove time.
    pub fn save(&self, path: &std::path::Path) -> Result<(), MlError> {
        let payload = (self.n0_bits, &self.codewords, &self.tree);
        let bytes = bincode::serde::encode_to_vec(payload, bincode::config::standard())
            .map_err(|e| MlError::Io(format!("serializing committed matrix: {e}")))?;
        std::fs::write(path, bytes).map_err(|e| MlError::Io(format!("writing {}: {e}", path.display())))
    }

    /// Load a committed matrix written by [`save`](Self::save), rebuilding the
    /// Merkle leaves from the stored codewords.
    pub fn load(path: &std::path::Path) -> Result<Self, MlError> {
        let bytes = std::fs::read(path).map_err(|e| MlError::Io(format!("reading {}: {e}", path.display())))?;
        let ((n0_bits, codewords, tree), _): ((usize, Vec<Vec<Goldilocks>>, MerkleTree), _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map_err(|e| MlError::Io(format!("decoding committed matrix: {e}")))?;
        let leaves = pack_base_pairs(&codewords);
        Ok(CommittedMatrix { codewords, leaves, tree, n0_bits })
    }
}

/// Pair-pack RS codewords into Merkle leaves: `leaves[j] = [cols @ j…, cols @ j+N/2…]`.
/// Trivial (no NTT/hash), so it is cheap to rebuild on load.
fn pack_base_pairs(codewords: &[Vec<Goldilocks>]) -> Vec<Vec<Goldilocks>> {
    let half = codewords[0].len() / 2;
    (0..half)
        .map(|j| {
            let mut leaf = Vec::with_capacity(2 * codewords.len());
            for cw in codewords {
                leaf.push(cw[j]);
            }
            for cw in codewords {
                leaf.push(cw[j + half]);
            }
            leaf
        })
        .collect()
}

/// RS-encode and Merkle-commit a set of columns.
pub fn commit_matrix(columns: &[&[Goldilocks]], params: &MlParams) -> CommittedMatrix {
    assert!(!columns.is_empty());
    let n = columns[0].len();
    assert!(columns.iter().all(|c| c.len() == n));

    let codewords: Vec<Vec<Goldilocks>> = encode_columns(columns, params.log_blowup);
    let n0 = codewords[0].len();

    let leaves = pack_base_pairs(&codewords);
    let tree = build_merkle(&leaves, MERKLE_ARITY);
    CommittedMatrix { codewords, leaves, tree, n0_bits: n0.trailing_zeros() as usize }
}

/// One FRI fold step, in the **value-to-coefficient** convention:
/// `f_{level+1}(x²) = (1−r)·(f(x)+f(−x))/2 + r·(f(x)−f(−x))/(2x)`.
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

#[inline]
pub(crate) fn fold_pair(a: Ext, b: Ext, n0_bits: usize, level: usize, j: u64, r: Ext) -> Ext {
    let two_inv = Goldilocks::TWO.inverse();
    let x_inv = domain_point(n0_bits, level, j).inverse();
    (Ext::ONE - r) * ((a + b) * two_inv) + r * ((a - b) * (two_inv * x_inv))
}

/// Pair-pack an extension codeword into Merkle leaves (6 Goldilocks each).
fn pack_ext_pairs(vals: &[Ext]) -> Vec<Vec<Goldilocks>> {
    let half = vals.len() / 2;
    (0..half)
        .map(|j| {
            let mut leaf = Vec::with_capacity(6);
            leaf.extend_from_slice(&vals[j].value);
            leaf.extend_from_slice(&vals[j + half].value);
            leaf
        })
        .collect()
}

fn unpack_ext_pair(leaf: &[Goldilocks]) -> [Ext; 2] {
    [Ext::from_array(&leaf[0..3]), Ext::from_array(&leaf[3..6])]
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FoldOpening {
    pub pair: [Ext; 2],
    pub path: Vec<Vec<Goldilocks>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpeningQuery {
    /// Per stage matrix: the raw pair-packed leaf at position `p1` and its path.
    pub stage_leaves: Vec<Vec<Goldilocks>>,
    pub stage_paths: Vec<Vec<Vec<Goldilocks>>>,
    /// Openings of the committed fold oracles `f_1 … f_{L−1}`.
    pub fold_openings: Vec<FoldOpening>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpeningProof {
    /// Sumcheck round polynomials (evaluations at 0), one per variable.
    pub round_polys: Vec<Ext>,
    /// Roots of the committed fold oracles `f_1 … f_{L−1}`.
    pub fold_roots: Vec<[Goldilocks; 4]>,
    /// The remaining polynomial after `L` folds, in clear (`2^(n−L)` coefficients).
    pub final_poly: Vec<Ext>,
    pub queries: Vec<OpeningQuery>,
}

/// Prove the batched single-point opening `f(u) = σ`.
///
/// `phi_table`: MLE table of `f` on the hypercube.
/// `point`: the evaluation point `u`.
/// `phi_codeword`: codeword of `f`
/// `matrices`: the stage commitments, in the same order the verifier will use.
pub fn prove_opening(
    params: &MlParams,
    transcript: &mut MlTranscript,
    phi_table: Vec<Ext>,
    point: &[Ext],
    phi_codeword: Vec<Ext>,
    matrices: &[&CommittedMatrix],
) -> OpeningProof {
    assert_eq!(params.grinding_bits, 0, "grinding not implemented in v1");
    let n = phi_table.len().trailing_zeros() as usize;
    assert_eq!(point.len(), n);
    let n0_bits = params.n0_bits(n);
    assert_eq!(phi_codeword.len(), 1usize << n0_bits);
    let num_folds = params.num_folds(n);

    // In-clear shadow of f.
    let mut phi_vals = phi_table.clone();
    let mut oracle = EqProductOracle::new(phi_table, point.to_vec());
    let mut codeword = phi_codeword;
    let mut fold_trees: Vec<(MerkleTree, Vec<Vec<Goldilocks>>)> = Vec::with_capacity(num_folds - 1);
    let mut round_polys = Vec::with_capacity(n);
    let mut final_poly: Vec<Ext> = Vec::new();

    // Commit phase: sumcheck rounds interleaved with FRI folds.
    for t in 0..n {
        // Compute the sumcheck round polynomial
        let eval = oracle.round_evals()[0];
        transcript.absorb_ext(&eval);
        round_polys.push(eval);

        // Sample a challenge and update the running claim to s = gᵢ(rᵢ)
        let r = transcript.challenge();
        oracle.bind(r);

        if t < num_folds {
            // Compute the folded codeword
            codeword = fold_codeword(&codeword, n0_bits, t, r);
            fold_mle(&mut phi_vals, r);
            if t + 1 < num_folds {
                // Before the last fold, commit the folded codeword as a Merkle tree and absorb its root.
                let leaves = pack_ext_pairs(&codeword);
                let tree = build_merkle(&leaves, MERKLE_ARITY);
                transcript.absorb_root(&tree.root());
                fold_trees.push((tree, leaves));
            } else {
                // After the last fold, send the remaining polynomial in clear.
                // These values are simultaneously the remaining MLE table and
                // the coefficients of the remaining univariate codeword.
                final_poly = phi_vals.clone();
                transcript.absorb_exts(&final_poly);
            }
        }
    }

    // Query phase: indices are pair positions in [0, N_0/2).
    let indices = transcript.query_indices(params.n_queries as u64, (n0_bits - 1) as u64);

    let queries = indices
        .iter()
        .map(|&p1| {
            let stage_leaves: Vec<Vec<Goldilocks>> = matrices.iter().map(|m| m.leaves[p1 as usize].clone()).collect();
            let stage_paths: Vec<Vec<Vec<Goldilocks>>> = matrices.iter().map(|m| m.tree.path(p1)).collect();

            let mut fold_openings = Vec::with_capacity(num_folds.saturating_sub(1));
            let mut p = p1;
            for (i, (tree, leaves)) in fold_trees.iter().enumerate() {
                let level = i + 1;
                let half_bits = n0_bits - level - 1;
                let q = p & ((1u64 << half_bits) - 1);
                fold_openings.push(FoldOpening { pair: unpack_ext_pair(&leaves[q as usize]), path: tree.path(q) });
                p = q;
            }
            OpeningQuery { stage_leaves, stage_paths, fold_openings }
        })
        .collect();

    OpeningProof { round_polys, fold_roots: fold_trees.iter().map(|(t, _)| t.root()).collect(), final_poly, queries }
}

/// Verify a batched single-point opening `f(u) = σ`.
///
/// `sigma`: the claimed evaluation.
/// `proof`: the opening proof.
/// `stage_roots`/`stage_n_cols`: the stage commitments.
/// `column_coeffs`: the `δ`-coefficients of every column in commitment order.
/// `point`: the evaluation point `u`.
///
/// Returns the sumcheck challenge point on success.
#[allow(clippy::too_many_arguments)]
pub fn verify_opening(
    params: &MlParams,
    transcript: &mut MlTranscript,
    n_vars: usize,
    sigma: Ext,
    proof: &OpeningProof,
    stage_roots: &[[Goldilocks; 4]],
    stage_n_cols: &[usize],
    column_coeffs: &[Ext],
    point: &[Ext],
) -> Result<Vec<Ext>, MlError> {
    assert_eq!(params.grinding_bits, 0, "grinding not implemented in v1");
    let n0_bits = params.n0_bits(n_vars);
    let num_folds = params.num_folds(n_vars);
    let total_cols: usize = stage_n_cols.iter().sum();

    if proof.round_polys.len() != n_vars {
        return Err(MlError::Malformed(format!("expected {n_vars} round polynomials")));
    }
    if proof.fold_roots.len() != num_folds - 1 {
        return Err(MlError::Malformed(format!("expected {} fold roots", num_folds - 1)));
    }
    if proof.final_poly.len() != 1usize << (n_vars - num_folds) {
        return Err(MlError::Malformed("final polynomial has wrong length".into()));
    }
    if column_coeffs.len() != total_cols {
        return Err(MlError::Malformed("column coefficient count mismatch".into()));
    }

    // Sumcheck rounds (replaying the transcript). The prover sends only the
    // linear factor `h` of each round polynomial; we
    // reconstruct `g = prefix·eq₁(u_t,·)·h` from the known eq point `u` and the
    // running `prefix`, then run the standard sumcheck round check.
    let mut claim = sigma;
    let mut rs = Vec::with_capacity(n_vars);
    for (t, eval) in proof.round_polys.iter().enumerate() {
        transcript.absorb_ext(eval);
        let r = transcript.challenge();
        let next_claim = eq_product_verifier_sumcheck_round(claim, *eval, point[t], r)?;
        claim = next_claim;
        rs.push(r);

        if t + 1 < num_folds {
            transcript.absorb_root(&proof.fold_roots[t]);
        } else if t + 1 == num_folds {
            transcript.absorb_exts(&proof.final_poly);
        }
    }

    // Final algebraic check: claim == f(rs).
    let phi_final = mle_eval(&proof.final_poly, &rs[num_folds..]);
    if claim != phi_final {
        return Err(MlError::FinalCheck("sumcheck claim != f(λ)".into()));
    }

    // Query phase.
    let indices = transcript.query_indices(params.n_queries as u64, (n0_bits - 1) as u64);
    if proof.queries.len() != indices.len() {
        return Err(MlError::Malformed("query count mismatch".into()));
    }

    for (k, (&p1, query)) in indices.iter().zip(proof.queries.iter()).enumerate() {
        if query.stage_leaves.len() != stage_roots.len() || query.stage_paths.len() != stage_roots.len() {
            return Err(MlError::Malformed(format!("query {k}: stage opening count mismatch")));
        }

        // Verify stage openings and combine columns into the f_0 pair.
        let half0 = 1u64 << (n0_bits - 1);
        let mut a = Ext::ZERO;
        let mut b = Ext::ZERO;
        let mut col_idx = 0;
        for (m, ((leaf, path), &n_cols)) in
            query.stage_leaves.iter().zip(query.stage_paths.iter()).zip(stage_n_cols.iter()).enumerate()
        {
            if leaf.len() != 2 * n_cols {
                return Err(MlError::Malformed(format!("query {k}: stage {m} leaf has wrong length")));
            }
            if !fields::verify_mt::<Goldilocks, MlHash, MlHash>(&stage_roots[m], &[], path, p1, leaf, MERKLE_ARITY, 0) {
                return Err(MlError::MerklePath(format!("query {k}, stage {m}, position {p1}")));
            }
            for c in 0..n_cols {
                a += column_coeffs[col_idx + c] * leaf[c];
                b += column_coeffs[col_idx + c] * leaf[n_cols + c];
            }
            col_idx += n_cols;
        }
        let _ = half0;

        // Fold cascade: level 0 → 1 with r_0, then committed oracles.
        let mut v = fold_pair(a, b, n0_bits, 0, p1, rs[0]);
        let mut p = p1;
        for (i, opening) in query.fold_openings.iter().enumerate() {
            let level = i + 1;
            let half_bits = n0_bits - level - 1;
            let half = 1u64 << half_bits;
            let q = p & (half - 1);

            let mut leaf = Vec::with_capacity(6);
            leaf.extend_from_slice(&opening.pair[0].value);
            leaf.extend_from_slice(&opening.pair[1].value);
            if !fields::verify_mt::<Goldilocks, MlHash, MlHash>(
                &proof.fold_roots[i],
                &[],
                &opening.path,
                q,
                &leaf,
                MERKLE_ARITY,
                0,
            ) {
                return Err(MlError::MerklePath(format!("query {k}, fold oracle {level}, position {q}")));
            }

            let expected = if p < half { opening.pair[0] } else { opening.pair[1] };
            if v != expected {
                return Err(MlError::FoldConsistency { query: k, level });
            }
            v = fold_pair(opening.pair[0], opening.pair[1], n0_bits, level, q, rs[level]);
            p = q;
        }

        // Last fold lands on the in-clear polynomial.
        let x = domain_point(n0_bits, num_folds, p);
        if v != eval_ext_poly_at_base(&proof.final_poly, x) {
            return Err(MlError::FoldConsistency { query: k, level: num_folds });
        }
    }

    Ok(rs)
}

/// Compute the batched codeword `Σ_j coeff_j · codeword_j` over one or more matrices.
pub fn combine_codewords(matrices: &[&CommittedMatrix], coeffs: &[Ext]) -> Vec<Ext> {
    let n0 = matrices[0].codewords[0].len();
    let mut out = vec![Ext::ZERO; n0];
    let mut idx = 0;
    for m in matrices {
        for cw in &m.codewords {
            let c = coeffs[idx];
            for (o, &v) in out.iter_mut().zip(cw.iter()) {
                *o += c * v;
            }
            idx += 1;
        }
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

/// The Basefold scheme (sumcheck interleaved with 2:1 FRI folding + spot-check
/// queries): the [`MlPcs`] implementation whose methods delegate to the free
/// functions in this module.
pub struct Basefold;

impl MlPcs for Basefold {
    type Commitment = CommittedMatrix;
    type OpeningProof = OpeningProof;

    fn commit(columns: &[&[Goldilocks]], params: &MlParams) -> CommittedMatrix {
        commit_matrix(columns, params)
    }

    fn combine_codewords(matrices: &[&CommittedMatrix], coeffs: &[Ext]) -> Vec<Ext> {
        combine_codewords(matrices, coeffs)
    }

    fn open(
        params: &MlParams,
        transcript: &mut MlTranscript,
        phi_table: Vec<Ext>,
        point: &[Ext],
        phi_codeword: Vec<Ext>,
        matrices: &[&CommittedMatrix],
    ) -> OpeningProof {
        prove_opening(params, transcript, phi_table, point, phi_codeword, matrices)
    }

    #[allow(clippy::too_many_arguments)]
    fn verify(
        params: &MlParams,
        transcript: &mut MlTranscript,
        n_vars: usize,
        sigma: Ext,
        proof: &OpeningProof,
        stage_roots: &[[Goldilocks; 4]],
        stage_n_cols: &[usize],
        column_coeffs: &[Ext],
        point: &[Ext],
    ) -> Result<Vec<Ext>, MlError> {
        verify_opening(params, transcript, n_vars, sigma, proof, stage_roots, stage_n_cols, column_coeffs, point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq::eq_evals;
    use crate::hypercube::dot_base_ext;
    use fields::PrimeField64;
    use rand::{rng, RngExt};

    fn random_col(len: usize) -> Vec<Goldilocks> {
        let mut r = rng();
        (0..len).map(|_| Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64)).collect()
    }

    fn test_params() -> MlParams {
        MlParams { log_blowup: 2, n_queries: 8, log_final_poly_len: 2, grinding_bits: 0, univariate_skip_bits: 0 }
    }

    /// Full commit → open → verify roundtrip: two matrices (stages), several
    /// columns, a single eq(·,λ) kernel.
    #[test]
    fn opening_roundtrip() {
        let n = 6;
        let len = 1usize << n;
        let params = test_params();

        let cols_a: Vec<Vec<Goldilocks>> = (0..3).map(|_| random_col(len)).collect();
        let cols_b: Vec<Vec<Goldilocks>> = (0..2).map(|_| random_col(len)).collect();
        let mat_a = commit_matrix(&cols_a.iter().map(|c| c.as_slice()).collect::<Vec<_>>(), &params);
        let mat_b = commit_matrix(&cols_b.iter().map(|c| c.as_slice()).collect::<Vec<_>>(), &params);
        let matrices = [&mat_a, &mat_b];
        let all_cols: Vec<&[Goldilocks]> = cols_a.iter().chain(cols_b.iter()).map(|c| c.as_slice()).collect();

        // --- prover transcript ---
        let mut tp = MlTranscript::new();
        tp.absorb_root(&mat_a.root());
        tp.absorb_root(&mat_b.root());

        // Opening point and claims
        let lambda = tp.challenges(n);
        let kernel = eq_evals(&lambda);
        let claims: Vec<Ext> = all_cols.iter().map(|c| dot_base_ext(c, &kernel)).collect();
        for v in &claims {
            tp.absorb_ext(v);
        }

        // Batching: δ for columns (single kernel, so no γ needed beyond δ)
        let delta = tp.challenge();
        let mut coeffs = Vec::with_capacity(all_cols.len());
        let mut d = Ext::ONE;
        for _ in 0..all_cols.len() {
            coeffs.push(d);
            d *= delta;
        }
        let sigma: Ext = claims.iter().zip(coeffs.iter()).map(|(&v, &c)| v * c).sum();

        let phi_table = combine_columns(&all_cols, &coeffs);
        let phi_codeword = combine_codewords(&matrices, &coeffs);
        let proof = prove_opening(&params, &mut tp, phi_table, &lambda, phi_codeword, &matrices);

        // --- verifier transcript ---
        let mut tv = MlTranscript::new();
        tv.absorb_root(&mat_a.root());
        tv.absorb_root(&mat_b.root());
        let lambda_v = tv.challenges(n);
        assert_eq!(lambda_v, lambda);
        for v in &claims {
            tv.absorb_ext(v);
        }
        let delta_v = tv.challenge();
        assert_eq!(delta_v, delta);

        let rs = verify_opening(
            &params,
            &mut tv,
            n,
            sigma,
            &proof,
            &[mat_a.root(), mat_b.root()],
            &[3, 2],
            &coeffs,
            &lambda_v,
        )
        .expect("opening must verify");

        // The claim delivered by the fold cascade equals f at the challenge point.
        assert_eq!(rs.len(), n);
    }

    /// Corrupting a claimed evaluation must break verification.
    #[test]
    fn wrong_claim_rejected() {
        let n = 5;
        let len = 1usize << n;
        let params = test_params();
        let col = random_col(len);
        let mat = commit_matrix(&[&col], &params);

        let mut tp = MlTranscript::new();
        tp.absorb_root(&mat.root());
        let lambda = tp.challenges(n);
        let kernel = eq_evals(&lambda);
        // WRONG claim
        let sigma = dot_base_ext(&col, &kernel) + Ext::ONE;
        tp.absorb_ext(&sigma);
        let coeffs = vec![Ext::ONE];
        let phi_table = combine_columns(&[&col], &coeffs);
        let phi_codeword = combine_codewords(&[&mat], &coeffs);
        let proof = prove_opening(&params, &mut tp, phi_table, &lambda, phi_codeword, &[&mat]);

        let mut tv = MlTranscript::new();
        tv.absorb_root(&mat.root());
        let lambda_v = tv.challenges(n);
        tv.absorb_ext(&sigma);
        let res = verify_opening(&params, &mut tv, n, sigma, &proof, &[mat.root()], &[1], &coeffs, &lambda_v);
        assert!(res.is_err(), "inflated claim must be rejected");
    }

    /// Corrupting a committed codeword position must be caught by queries
    /// (with the small query count used here, at least with high probability —
    /// we corrupt many positions to make failure certain).
    #[test]
    fn corrupted_codeword_rejected() {
        let n = 5;
        let len = 1usize << n;
        let params = MlParams { n_queries: 16, ..test_params() };
        let col = random_col(len);
        let mut mat = commit_matrix(&[&col], &params);

        // Corrupt half of the codeword AFTER computing honest claims, then rebuild
        // the tree so paths verify but fold consistency breaks.
        let n0 = mat.codewords[0].len();
        for j in 0..n0 / 2 {
            mat.codewords[0][j] += Goldilocks::ONE;
        }
        let half = n0 / 2;
        let leaves: Vec<Vec<Goldilocks>> =
            (0..half).map(|j| vec![mat.codewords[0][j], mat.codewords[0][j + half]]).collect();
        mat.tree = MerkleTree::new::<MlHash>(&leaves, MERKLE_ARITY);
        mat.leaves = leaves;

        let mut tp = MlTranscript::new();
        tp.absorb_root(&mat.root());
        let lambda = tp.challenges(n);
        let kernel = eq_evals(&lambda);
        let sigma = dot_base_ext(&col, &kernel);
        tp.absorb_ext(&sigma);
        let coeffs = vec![Ext::ONE];
        let phi_table = combine_columns(&[&col], &coeffs);
        let phi_codeword: Vec<Ext> = mat.codewords[0].iter().map(|&v| Ext::from_base(v)).collect();
        let proof = prove_opening(&params, &mut tp, phi_table, &lambda, phi_codeword, &[&mat]);

        let mut tv = MlTranscript::new();
        tv.absorb_root(&mat.root());
        let lambda_v = tv.challenges(n);
        tv.absorb_ext(&sigma);
        let res = verify_opening(&params, &mut tv, n, sigma, &proof, &[mat.root()], &[1], &coeffs, &lambda_v);
        assert!(res.is_err(), "corrupted codeword must be rejected");
    }
}
