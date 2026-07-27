//! The multilinear polynomial commitment scheme (PCS) seam.

use fields::Goldilocks;

use crate::error::MlError;
use crate::hypercube::Ext;
use crate::transcript::MlTranscript;

use super::common::MlParams;

/// A multilinear polynomial commitment scheme: commit base-field column
/// matrices and prove/verify the batched single-point opening
/// `Φ̃(u) = σ`, i.e. `Σ_b Φ(b)·eq(b, u) = σ`.
///
/// `Φ` (the random linear combination of committed columns), the point `u`,
/// and the meaning of `σ` are defined by the caller; the scheme only has to
/// prove the eq-weighted sum and the underlying column openings.
pub trait MlPcs {
    /// A committed matrix of columns (carries the encoded codewords + Merkle tree).
    type Commitment;
    /// The opening proof produced by [`open`](MlPcs::open) and checked by [`verify`](MlPcs::verify).
    type OpeningProof: serde::Serialize + serde::de::DeserializeOwned + Clone + std::fmt::Debug;

    /// RS-encode and Merkle-commit a set of base-field columns.
    fn commit(columns: &[&[Goldilocks]], params: &MlParams) -> Self::Commitment;

    /// Merkle root of a commitment (absorbed into the transcript / stored in the proof).
    fn commitment_root(commitment: &Self::Commitment) -> [Goldilocks; 4];

    /// Serialize a commitment as a proving-key artifact (`.mlconst.bin`).
    fn save_commitment(commitment: &Self::Commitment, path: &std::path::Path) -> Result<(), MlError>;

    /// Load a commitment written by [`save_commitment`](MlPcs::save_commitment).
    fn load_commitment(path: &std::path::Path) -> Result<Self::Commitment, MlError>;

    /// Codeword of `Φ = Σ_j coeffs[j]·col_j` over the committed matrices, in the
    /// scheme's evaluation domain (folded together with the opening).
    fn combine_codewords(matrices: &[&Self::Commitment], coeffs: &[Ext]) -> Vec<Ext>;

    /// Prove `Σ_b Φ(b)·eq(b, u) = σ`. `phi_table` is the hypercube MLE table of
    /// `Φ` and `point` the evaluation point `u`; `phi_codeword` is `Φ`'s
    /// codeword (from [`combine_codewords`](MlPcs::combine_codewords));
    /// `matrices` are the commitments anchoring the query phase.
    fn open(
        params: &MlParams,
        transcript: &mut MlTranscript,
        phi_table: Vec<Ext>,
        point: &[Ext],
        phi_codeword: Vec<Ext>,
        matrices: &[&Self::Commitment],
    ) -> Self::OpeningProof;

    /// Verify the opening of `Φ̃(point) = σ` and return the scheme's sumcheck
    /// challenge point on success.
    #[allow(clippy::too_many_arguments)]
    fn verify(
        params: &MlParams,
        transcript: &mut MlTranscript,
        n_vars: usize,
        sigma: Ext,
        proof: &Self::OpeningProof,
        stage_roots: &[[Goldilocks; 4]],
        stage_n_cols: &[usize],
        column_coeffs: &[Ext],
        point: &[Ext],
    ) -> Result<Vec<Ext>, MlError>;
}

/// The multilinear PCS the prover and verifier use. Swap this alias (once the
/// scheme implements [`MlPcs`]) to change the whole prover's commitment scheme.
pub type Pcs = crate::pcs::Whir;

/// The active scheme's commitment type (`<Pcs as MlPcs>::Commitment`).
pub type PcsCommitment = <Pcs as MlPcs>::Commitment;

/// The active scheme's opening-proof type (`<Pcs as MlPcs>::OpeningProof`).
pub type PcsOpening = <Pcs as MlPcs>::OpeningProof;
