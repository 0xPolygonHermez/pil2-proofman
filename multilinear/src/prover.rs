//! The multilinear STARK prover: commit → zerocheck → batched Basefold opening.

use crate::basefold::{combine_codewords, combine_columns, commit_matrix, prove_opening, CommittedMatrix, OpeningProof};
use crate::eq::{eq_evals, rotate_table};
use crate::error::MlError;
use crate::hypercube::{dot_base_ext, ext_from_base, Ext};
use crate::ir::AirIr;
use crate::sumcheck::SumcheckOracle;
use crate::transcript::MlTranscript;
use crate::zerocheck::{build_kernels, KernelSpec, ZerocheckOracle};
use fields::{Goldilocks, PrimeField64};
use serde::{Deserialize, Serialize};

/// A multilinear STARK proof for one AIR instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlProof {
    pub airgroup_id: u32,
    pub air_id: u32,
    pub n_bits: u32,
    /// Merkle roots of the witness stage commitments, in stage order.
    pub stage_roots: Vec<[Goldilocks; 4]>,
    /// Merkle root of the fixed-column commitment.
    pub const_root: [Goldilocks; 4],
    /// Zerocheck round polynomials (evaluations at `0..=max_degree+1`).
    pub zerocheck_round_polys: Vec<Vec<Ext>>,
    /// Claimed weighted-sum openings: `claims[global_col][kernel]`.
    pub claims: Vec<Vec<Ext>>,
    pub opening: OpeningProof,
    pub publics: Vec<Goldilocks>,
}

impl MlProof {
    pub fn save(&self, path: &std::path::Path) -> Result<(), MlError> {
        let bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| MlError::Io(format!("serializing proof: {e}")))?;
        std::fs::write(path, bytes).map_err(|e| MlError::Io(format!("writing {}: {e}", path.display())))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, MlError> {
        let bytes = std::fs::read(path).map_err(|e| MlError::Io(format!("reading {}: {e}", path.display())))?;
        let (proof, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .map_err(|e| MlError::Io(format!("decoding proof: {e}")))?;
        Ok(proof)
    }
}

/// Seed the transcript with the statement: AIR identity and public inputs.
pub(crate) fn seed_transcript(transcript: &mut MlTranscript, ir: &AirIr, publics: &[Goldilocks]) {
    transcript.absorb(&[
        Goldilocks::from_u64(ir.airgroup_id as u64),
        Goldilocks::from_u64(ir.air_id as u64),
        Goldilocks::from_u64(ir.n_bits as u64),
    ]);
    transcript.absorb(publics);
}

/// The kernel's MLE table over the hypercube (prover side).
pub(crate) fn kernel_table(spec: &KernelSpec, lambda: &[Ext], eq_lambda: &[Ext]) -> Vec<Ext> {
    let _ = lambda;
    match spec {
        KernelSpec::Rot(s) => {
            if *s == 0 {
                eq_lambda.to_vec()
            } else {
                rotate_table(eq_lambda, *s as i64)
            }
        }
        KernelSpec::Point(row) => {
            let mut t = vec![Ext::zero(); eq_lambda.len()];
            t[*row as usize] = Ext::one();
            t
        }
    }
}

/// Generate a proof that `witness` (per-stage, column-major) satisfies `ir`'s
/// constraints, with fixed columns `consts` and public inputs `publics`.
///
/// Multi-stage AIRs with challenges are not supported yet (milestone 2).
pub fn prove_air(
    ir: &AirIr,
    witness: &[Vec<Vec<Goldilocks>>],
    consts: &[Vec<Goldilocks>],
    publics: &[Goldilocks],
) -> Result<MlProof, MlError> {
    if !ir.challenge_stages.is_empty() {
        return Err(MlError::Unsupported("multi-stage AIRs with challenges (milestone 2)".into()));
    }
    if witness.len() != ir.n_stages() {
        return Err(MlError::Malformed(format!("expected {} witness stages", ir.n_stages())));
    }
    let n = ir.n_bits as usize;
    let n_rows = 1usize << n;
    let params = &ir.params;

    let mut transcript = MlTranscript::new();
    seed_transcript(&mut transcript, ir, publics);

    // --- Commitments: const columns first (their root doubles as a verifying
    // key), then the witness stages.
    let const_refs: Vec<&[Goldilocks]> = consts.iter().map(|c| c.as_slice()).collect();
    let const_matrix = commit_matrix(&const_refs, params);
    transcript.absorb_root(&const_matrix.root());

    let mut stage_matrices: Vec<CommittedMatrix> = Vec::with_capacity(witness.len());
    for stage_cols in witness {
        let refs: Vec<&[Goldilocks]> = stage_cols.iter().map(|c| c.as_slice()).collect();
        let matrix = commit_matrix(&refs, params);
        transcript.absorb_root(&matrix.root());
        stage_matrices.push(matrix);
    }

    // --- Zerocheck: one sumcheck for all EveryRow constraints.
    let r = transcript.challenges(n);
    let alpha = transcript.challenge();
    let challenges: Vec<Ext> = Vec::new(); // populated in milestone 2

    let mut oracle = ZerocheckOracle::new(ir, witness, consts, publics, &challenges, &r, alpha);
    let mut zerocheck_round_polys = Vec::with_capacity(n);
    let mut lambda = Vec::with_capacity(n);
    for _ in 0..n {
        let evals = oracle.round_evals();
        transcript.absorb_exts(&evals);
        let ch = transcript.challenge();
        oracle.bind(ch);
        zerocheck_round_polys.push(evals);
        lambda.push(ch);
    }

    // --- Claimed openings: full (column × kernel) matrix.
    let kernels = build_kernels(ir);
    let eq_lambda = eq_evals(&lambda);
    let kernel_tables: Vec<Vec<Ext>> = kernels.iter().map(|k| kernel_table(k, &lambda, &eq_lambda)).collect();

    let all_cols: Vec<&[Goldilocks]> = witness
        .iter()
        .flat_map(|stage| stage.iter().map(|c| c.as_slice()))
        .chain(consts.iter().map(|c| c.as_slice()))
        .collect();

    let claims: Vec<Vec<Ext>> = all_cols
        .iter()
        .map(|col| {
            kernels
                .iter()
                .zip(kernel_tables.iter())
                .map(|(spec, table)| match spec {
                    // Corner claims are plain trace reads; no need for the dot product.
                    KernelSpec::Point(row) => ext_from_base(col[*row as usize]),
                    KernelSpec::Rot(_) => dot_base_ext(col, table),
                })
                .collect()
        })
        .collect();
    for row in &claims {
        transcript.absorb_exts(row);
    }

    // --- Two-level batching and the Basefold opening.
    let delta = transcript.challenge();
    let gamma = transcript.challenge();
    let col_coeffs = powers(delta, all_cols.len());
    let kernel_weights = powers(gamma, kernels.len());

    let mut sigma = Ext::zero();
    for (j, row) in claims.iter().enumerate() {
        for (i, v) in row.iter().enumerate() {
            sigma += col_coeffs[j] * kernel_weights[i] * *v;
        }
    }

    let phi_table = combine_columns(&all_cols, &col_coeffs);
    let mut w_table = vec![Ext::zero(); n_rows];
    for (wt, table) in kernel_weights.iter().zip(kernel_tables.iter()) {
        for (o, v) in w_table.iter_mut().zip(table.iter()) {
            *o += *wt * *v;
        }
    }

    let mut matrices: Vec<&CommittedMatrix> = stage_matrices.iter().collect();
    matrices.push(&const_matrix);
    let phi_codeword = combine_codewords(&matrices, &col_coeffs);

    let _ = sigma; // prover-side sanity value; the verifier recomputes it from the claims
    let opening = prove_opening(params, &mut transcript, phi_table, w_table, phi_codeword, &matrices);

    Ok(MlProof {
        airgroup_id: ir.airgroup_id,
        air_id: ir.air_id,
        n_bits: ir.n_bits,
        stage_roots: stage_matrices.iter().map(|m| m.root()).collect(),
        const_root: const_matrix.root(),
        zerocheck_round_polys,
        claims,
        opening,
        publics: publics.to_vec(),
    })
}

pub(crate) fn powers(base: Ext, n: usize) -> Vec<Ext> {
    let mut out = Vec::with_capacity(n);
    let mut cur = Ext::one();
    for _ in 0..n {
        out.push(cur);
        cur *= base;
    }
    out
}
