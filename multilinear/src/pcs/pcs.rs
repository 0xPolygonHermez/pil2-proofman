//! The multilinear polynomial commitment scheme (PCS) seam.
//!
//! The prover ([`crate::prove_air`]) and verifier ([`crate::verify_air`]) reduce
//! a STARK statement to a single batched weighted-sum opening
//! `Σ_b Φ(b)·W(b) = σ` and reach the commitment scheme only through the
//! [`MlPcs`] trait and the [`Pcs`] alias. Everything around the PCS — the
//! batching, transcript, zerocheck, challenge derivation, and Merkle /
//! `.mlconst.bin` plumbing — is scheme-agnostic, so a new scheme (e.g. WHIR,
//! see [`crate::whir`]) is added by implementing this trait and repointing
//! `Pcs` at it.

use fields::Goldilocks;

use crate::error::MlError;
use crate::hypercube::Ext;
use crate::transcript::MlTranscript;

use super::basefold::MlParams;

/// A multilinear polynomial commitment scheme: commit base-field column
/// matrices and prove/verify the batched weighted-sum opening
/// `Σ_b Φ(b)·W(b) = σ`.
///
/// The `Φ` (committed columns) / `W` (verifier-evaluable kernels) split, and
/// the meaning of `σ`, are defined by the caller; the scheme only has to prove
/// the weighted sum and the underlying column openings.
pub trait MlPcs {
    /// A committed matrix of columns (carries the encoded codewords + Merkle tree).
    type Commitment;
    /// The opening proof produced by [`open`](MlPcs::open) and checked by [`verify`](MlPcs::verify).
    type OpeningProof: serde::Serialize + serde::de::DeserializeOwned + Clone + std::fmt::Debug;

    /// RS-encode and Merkle-commit a set of base-field columns.
    fn commit(columns: &[&[Goldilocks]], params: &MlParams) -> Self::Commitment;

    /// Codeword of `Φ = Σ_j coeffs[j]·col_j` over the committed matrices, in the
    /// scheme's evaluation domain (folded together with the opening).
    fn combine_codewords(matrices: &[&Self::Commitment], coeffs: &[Ext]) -> Vec<Ext>;

    /// Prove `Σ_b Φ(b)·W(b) = σ`. `phi_table`/`w_table` are the hypercube MLE
    /// tables of `Φ` and `W`; `phi_codeword` is `Φ`'s codeword (from
    /// [`combine_codewords`](MlPcs::combine_codewords)); `matrices` are the
    /// commitments anchoring the query phase.
    fn open(
        params: &MlParams,
        transcript: &mut MlTranscript,
        phi_table: Vec<Ext>,
        w_table: Vec<Ext>,
        phi_codeword: Vec<Ext>,
        matrices: &[&Self::Commitment],
    ) -> Self::OpeningProof;

    /// Verify the opening and return the per-column evaluations at the final
    /// sumcheck point (the caller cross-checks these against the claimed values).
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
        w_final: impl FnOnce(&[Ext]) -> Ext,
    ) -> Result<Vec<Ext>, MlError>;
}

/// The multilinear PCS the prover and verifier use. Swap this alias (once the
/// scheme implements [`MlPcs`]) to change the whole prover's commitment scheme.
pub type Pcs = crate::pcs::Basefold;
