//! The multilinear STARK prover: commit → zerocheck → batched PCS opening.

use crate::basefold::{combine_codewords, combine_columns, commit_matrix, prove_opening, CommittedMatrix, OpeningProof};
use crate::eq::{rotate_table, skip_kernel_table};
use crate::error::MlError;
use crate::hypercube::{dot_base_ext, Ext};
use crate::ir::AirIr;
use crate::sumcheck::SumcheckOracle;
use crate::transcript::MlTranscript;
use crate::zerocheck::{build_kernels, KernelSpec, ZerocheckOracle};
use fields::{Goldilocks, PrimeField64};
use serde::{Deserialize, Serialize};
use std::time::Instant;

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
    /// Merkle roots of the custom (fixed) commitments, in `AirIr::custom_commits` order.
    pub custom_roots: Vec<[Goldilocks; 4]>,
    /// Zerocheck round polynomials (evaluations at `0..=max_degree+1`).
    pub zerocheck_round_polys: Vec<Vec<Ext>>,
    /// Claimed weighted-sum openings: `claims[global_col][kernel]`.
    pub claims: Vec<Vec<Ext>>,
    pub opening: OpeningProof,
    pub publics: Vec<Goldilocks>,
    /// Global transcript challenges used by the constraints (full global
    /// challenge vector; entries for stages the multilinear protocol does not
    /// derive are zero). Re-derived and checked at proof-set level.
    pub challenges: Vec<Ext>,
    /// Air values (per-instance prover messages), in global order.
    pub air_values: Vec<Ext>,
    /// Airgroup values (enter cross-instance global constraints), in global order.
    pub airgroup_values: Vec<Ext>,
    /// Global instance id assigned by the proving orchestrator; defines the
    /// root order for the global challenge derivation (a wrong or permuted id
    /// changes the derived challenges, so the set verifier rejects).
    pub global_instance_id: u32,
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

/// Generate a proof that `witness` satisfies `ir`'s constraints.
#[allow(clippy::too_many_arguments)]
pub fn prove_air(
    ir: &AirIr,
    witness: &[Vec<Vec<Goldilocks>>],
    consts: &[Vec<Goldilocks>],
    const_matrix: Option<&CommittedMatrix>,
    customs: &[Vec<Vec<Goldilocks>>],
    publics: &[Goldilocks],
    challenges: &[Ext],
    air_values: &[Ext],
    airgroup_values: &[Ext],
) -> Result<MlProof, MlError> {
    if witness.len() != ir.n_stages() {
        return Err(MlError::Malformed(format!("expected {} witness stages", ir.n_stages())));
    }
    if challenges.len() != ir.challenge_stages.len() {
        return Err(MlError::Malformed(format!("expected {} challenges", ir.challenge_stages.len())));
    }
    if air_values.len() != ir.airvalue_stages.len() || airgroup_values.len() != ir.airgroupvalue_stages.len() {
        return Err(MlError::Malformed("air/airgroup value count mismatch".into()));
    }
    if customs.len() != ir.custom_commits.len()
        || customs.iter().zip(ir.custom_commits.iter()).any(|(cols, cc)| cols.len() != cc.n_cols as usize)
    {
        return Err(MlError::Malformed("custom commit shape mismatch".into()));
    }

    let n = ir.n_bits as usize;
    let n_rows = 1usize << n;
    let params = &ir.params;

    // Start the transcript with the statement: AIR identity and public inputs.
    let mut transcript = MlTranscript::new();
    seed_transcript(&mut transcript, ir, publics);

    // --- Commitments ---
    let t_commit = Instant::now();
    // Fixed columns are known at setup time; reuse the prebuilt commitment
    // (loaded from the proving key) when supplied, otherwise build it here.
    let owned_const_matrix;
    let const_matrix: &CommittedMatrix = match const_matrix {
        Some(m) => m,
        None => {
            let const_refs: Vec<&[Goldilocks]> = consts.iter().map(|c| c.as_slice()).collect();
            owned_const_matrix = commit_matrix(&const_refs, params);
            &owned_const_matrix
        }
    };
    transcript.absorb_root(&const_matrix.root());

    // Custom columns are computed once before the first proof and reused for all proofs of the same AIR instance.
    let custom_matrices: Vec<CommittedMatrix> = customs
        .iter()
        .map(|cols| {
            let refs: Vec<&[Goldilocks]> = cols.iter().map(|c| c.as_slice()).collect();
            let matrix = commit_matrix(&refs, params);
            transcript.absorb_root(&matrix.root());
            matrix
        })
        .collect();

    // Stage commitments: one Merkle root per witness stage, in stage order.
    let mut stage_matrices: Vec<CommittedMatrix> = Vec::with_capacity(witness.len());
    for (stage_idx, stage_cols) in witness.iter().enumerate() {
        let refs: Vec<&[Goldilocks]> = stage_cols.iter().map(|c| c.as_slice()).collect();
        let matrix = commit_matrix(&refs, params);
        transcript.absorb_root(&matrix.root());
        stage_matrices.push(matrix);
        if stage_idx == 0 {
            // Bind the (globally derived) stage challenges right where the
            // protocol produces them: after the stage-1 commitment.
            for id in derived_challenge_ids(ir) {
                transcript.absorb_ext(&challenges[id]);
            }
        }
    }
    // Value messages are stage outputs: bind them before any zerocheck randomness.
    transcript.absorb_exts(air_values);
    transcript.absorb_exts(airgroup_values);
    let t_commit = t_commit.elapsed();

    // --- Zerocheck: one sumcheck for all EveryRow constraints.
    let t_zerocheck = Instant::now();
    let ell = ir.params.univariate_skip_bits.min(n);
    let r = transcript.challenges(n);
    let alpha = transcript.challenge();

    let mut oracle = ZerocheckOracle::new(
        ir,
        witness,
        consts,
        customs,
        publics,
        challenges,
        air_values,
        airgroup_values,
        &r,
        alpha,
        ell,
    );
    let mut zerocheck_round_polys = Vec::with_capacity(n - ell + 1);
    let mut skip_gamma = Ext::ZERO;
    let mut lambda_x = Vec::with_capacity(n - ell);
    if ell > 0 {
        let v = oracle.skip_round_evals();
        transcript.absorb_exts(&v);
        skip_gamma = transcript.challenge();
        oracle.skip_bind(skip_gamma);
        zerocheck_round_polys.push(v);
    }
    for _ in 0..(n - ell) {
        let evals = oracle.round_evals();
        transcript.absorb_exts(&evals);
        let ch = transcript.challenge();
        oracle.bind(ch);
        zerocheck_round_polys.push(evals);
        lambda_x.push(ch);
    }
    let t_zerocheck = t_zerocheck.elapsed();

    // --- Claimed openings: full (column × kernel) matrix, at the point `(γ, λ_X)`.
    let t_claims = Instant::now();
    let kernels = build_kernels(ir);
    let kernel_tables: Vec<Vec<Ext>> = kernels.iter().map(|k| kernel_table(k, ell, skip_gamma, &lambda_x)).collect();

    let all_cols: Vec<&[Goldilocks]> = witness
        .iter()
        .flat_map(|stage| stage.iter().map(|c| c.as_slice()))
        .chain(consts.iter().map(|c| c.as_slice()))
        .chain(customs.iter().flat_map(|cols| cols.iter().map(|c| c.as_slice())))
        .collect();

    let claims: Vec<Vec<Ext>> = all_cols
        .iter()
        .map(|col| {
            kernels
                .iter()
                .zip(kernel_tables.iter())
                .map(|(spec, table)| match spec {
                    // Corner claims are plain trace reads; no need for the dot product.
                    KernelSpec::Point(row) => Ext::from_base(col[*row as usize]),
                    KernelSpec::Rot(_) => dot_base_ext(col, table),
                })
                .collect()
        })
        .collect();
    for row in &claims {
        transcript.absorb_exts(row);
    }
    let t_claims = t_claims.elapsed();

    // --- Two-level batching and the Basefold opening.
    let t_opening = Instant::now();
    let delta = transcript.challenge();
    let gamma = transcript.challenge();
    let col_coeffs = powers(delta, all_cols.len());
    let kernel_weights = powers(gamma, kernels.len());

    let mut sigma = Ext::ZERO;
    for (j, row) in claims.iter().enumerate() {
        for (i, v) in row.iter().enumerate() {
            sigma += col_coeffs[j] * kernel_weights[i] * *v;
        }
    }

    let phi_table = combine_columns(&all_cols, &col_coeffs);
    let mut w_table = vec![Ext::ZERO; n_rows];
    for (wt, table) in kernel_weights.iter().zip(kernel_tables.iter()) {
        for (o, v) in w_table.iter_mut().zip(table.iter()) {
            *o += *wt * *v;
        }
    }

    let mut matrices: Vec<&CommittedMatrix> = stage_matrices.iter().collect();
    matrices.push(const_matrix);
    matrices.extend(custom_matrices.iter());
    let phi_codeword = combine_codewords(&matrices, &col_coeffs);

    let _ = sigma; // prover-side sanity value; the verifier recomputes it from the claims
    let opening = prove_opening(params, &mut transcript, phi_table, w_table, phi_codeword, &matrices);
    let t_opening = t_opening.elapsed();

    log::debug!(
        "ml prove_air[{}] n={n}: commit={t_commit:.1?} zerocheck={t_zerocheck:.1?} claims={t_claims:.1?} opening={t_opening:.1?}",
        ir.name,
    );

    Ok(MlProof {
        airgroup_id: ir.airgroup_id,
        air_id: ir.air_id,
        n_bits: ir.n_bits,
        stage_roots: stage_matrices.iter().map(|m| m.root()).collect(),
        const_root: const_matrix.root(),
        custom_roots: custom_matrices.iter().map(|m| m.root()).collect(),
        zerocheck_round_polys,
        claims,
        opening,
        publics: publics.to_vec(),
        challenges: challenges.to_vec(),
        air_values: air_values.to_vec(),
        airgroup_values: airgroup_values.to_vec(),
        global_instance_id: 0,
    })
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

/// The kernel's table over the hypercube `{0,1}^m` (prover side).
pub(crate) fn kernel_table(spec: &KernelSpec, ell: usize, gamma: Ext, lambda_x: &[Ext]) -> Vec<Ext> {
    match spec {
        KernelSpec::Rot(s) => {
            let base = skip_kernel_table(ell, gamma, lambda_x);
            if *s == 0 {
                base
            } else {
                rotate_table(&base, *s as i64)
            }
        }
        KernelSpec::Point(row) => {
            let mut t = vec![Ext::ZERO; 1 << (ell + lambda_x.len())];
            t[*row as usize] = Ext::ONE;
            t
        }
    }
}

/// Indices of the challenges the multilinear protocol actually derives:
/// those of stages `2..=n_stages`. Later stages (quotient/evals/FRI batching)
/// belong to the univariate protocol and stay zero.
pub fn derived_challenge_ids(ir: &AirIr) -> Vec<usize> {
    let n_stages = ir.n_stages() as u8;
    ir.challenge_stages.iter().enumerate().filter(|(_, &st)| st >= 2 && st <= n_stages).map(|(i, _)| i).collect()
}

/// Derive the global stage challenges from every instance's stage-1
/// commitment, in instance order.
///
/// Returns the full global challenge vector.
pub fn derive_global_challenges(ir: &AirIr, stage1_roots: &[[Goldilocks; 4]]) -> Vec<Ext> {
    derive_global_challenges_for(&ir.challenge_stages, ir.n_stages(), stage1_roots)
}

/// [`derive_global_challenges`] for a heterogeneous instance set: pass the
/// global challenge-stage list and the maximum number of witness stages among
/// the participating AIRs.
pub fn derive_global_challenges_for(
    challenge_stages: &[u8],
    n_stages: usize,
    stage1_roots: &[[Goldilocks; 4]],
) -> Vec<Ext> {
    let mut transcript = MlTranscript::new();
    for root in stage1_roots {
        transcript.absorb_root(root);
    }
    let mut challenges = vec![Ext::ZERO; challenge_stages.len()];
    for (id, &st) in challenge_stages.iter().enumerate() {
        if st >= 2 && st as usize <= n_stages {
            challenges[id] = transcript.challenge();
        }
    }
    challenges
}

pub(crate) fn powers(base: Ext, n: usize) -> Vec<Ext> {
    let mut out = Vec::with_capacity(n);
    let mut cur = Ext::ONE;
    for _ in 0..n {
        out.push(cur);
        cur *= base;
    }
    out
}
