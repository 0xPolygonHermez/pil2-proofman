//! The multilinear STARK prover.

use crate::pcs::{combine_columns, CommittedMatrix, OpeningProof};
use crate::pcs::{MlPcs, Pcs};
use crate::eq::{rotate_table, skip_kernel_table};
use crate::error::MlError;
use crate::hypercube::{dot_base_ext, Ext};
use crate::ir::AirIr;
use crate::sumcheck::{ProductOracle, SumcheckOracle};
use crate::transcript::MlTranscript;
use crate::zerocheck::{build_kernels, KernelSpec, ZerocheckOracle};
use fields::{Goldilocks, PrimeField64};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A multilinear STARK proof for one AIR instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlProof {
    /// Airgroup identifier.
    pub airgroup_id: u32,
    /// AIR identifier.
    pub air_id: u32,
    /// Number of bits in the hypercube (number of rows = 2^n_bits).
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
    /// Opening-reduction round polynomials.
    pub reduction_round_polys: Vec<Vec<Ext>>,
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

    let airgroup_id = ir.airgroup_id;
    let air_id = ir.air_id;
    let n_bits = ir.n_bits;

    let m = n_bits as usize;
    let n_rows = 1usize << m;
    let params = &ir.params;

    // Start the transcript with the statement: AIR identity and public inputs.
    let mut transcript = MlTranscript::new();
    seed_transcript(&mut transcript, airgroup_id, air_id, n_bits, publics);

    // --- Commitments ---
    let t_commit = Instant::now();

    // Fixed columns are known at setup time; reuse the prebuilt commitment
    // (loaded from the proving key) when supplied, otherwise build it here.
    let owned_const_matrix;
    let const_matrix: &CommittedMatrix = match const_matrix {
        Some(m) => m,
        None => {
            let const_refs: Vec<&[Goldilocks]> = consts.iter().map(|c| c.as_slice()).collect();
            owned_const_matrix = Pcs::commit(&const_refs, params);
            &owned_const_matrix
        }
    };
    transcript.absorb_root(&const_matrix.root());

    // Custom columns are computed once before the first proof and reused for all proofs of the same AIR instance.
    let custom_matrices: Vec<CommittedMatrix> = customs
        .iter()
        .map(|cols| {
            let refs: Vec<&[Goldilocks]> = cols.iter().map(|c| c.as_slice()).collect();
            let matrix = Pcs::commit(&refs, params);
            transcript.absorb_root(&matrix.root());
            matrix
        })
        .collect();

    // Stage commitments.
    let mut stage_matrices: Vec<CommittedMatrix> = Vec::with_capacity(witness.len());
    for (stage_idx, stage_cols) in witness.iter().enumerate() {
        let refs: Vec<&[Goldilocks]> = stage_cols.iter().map(|c| c.as_slice()).collect();
        let matrix = Pcs::commit(&refs, params);
        transcript.absorb_root(&matrix.root());
        stage_matrices.push(matrix);

        let stage = (stage_idx + 1) as u8;
        absorb_stage_values(&mut transcript, ir, air_values, airgroup_values, stage);
        for id in challenge_ids_for_stage(ir, stage + 1) {
            transcript.absorb_ext(&challenges[id]);
        }
    }

    let t_commit = t_commit.elapsed();

    // --- Zerocheck ---
    let t_zerocheck = Instant::now();

    let l = ir.params.univariate_skip_bits.min(m);
    let r = transcript.challenges(m);
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
        l,
    );

    // Compute the univariate skip polynomial
    let mut zerocheck_round_polys = Vec::with_capacity(m - l + 1);
    let mut skip_gamma = Ext::ZERO;
    if l > 0 {
        let v = oracle.skip_round_evals();
        transcript.absorb_exts(&v);
        skip_gamma = transcript.challenge();
        oracle.skip_bind(skip_gamma);
        zerocheck_round_polys.push(v);
    }

    // Compute the remaining rounds of the zerocheck protocol.
    let mut lambda_x = Vec::with_capacity(m - l);
    for _ in 0..(m - l) {
        let evals = oracle.round_evals();
        let sent = evals[1..].to_vec();
        transcript.absorb_exts(&sent);
        let ch = transcript.challenge();
        oracle.bind(ch);
        zerocheck_round_polys.push(sent);
        lambda_x.push(ch);
    }
    let t_zerocheck = t_zerocheck.elapsed();

    // --- Opening Reductions ---
    let t_claims = Instant::now();
    let kernels = build_kernels(ir);
    let kernel_tables: Vec<Vec<Ext>> = kernels.iter().map(|k| kernel_table(k, l, skip_gamma, &lambda_x)).collect();

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

    // --- Two-level batching and the opening reduction.
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
    let _ = sigma; // prover-side sanity value; the verifier recomputes it from the claims

    let phi_table = combine_columns(&all_cols, &col_coeffs);
    let mut w_table = vec![Ext::ZERO; n_rows];
    for (wt, table) in kernel_weights.iter().zip(kernel_tables.iter()) {
        for (o, v) in w_table.iter_mut().zip(table.iter()) {
            *o += *wt * *v;
        }
    }

    // Opening reduction: one plain sumcheck of `Σ_b Φ(b)·W(b) = σ` collapses
    // every (column, kernel) claim to evaluations at its challenge point `u`;
    // the sharing of challenges is what makes all claims land on one point.
    let mut reduction = ProductOracle::new(phi_table.clone(), w_table);
    let mut reduction_round_polys = Vec::with_capacity(m);
    let mut u = Vec::with_capacity(m);
    for _ in 0..m {
        // Tweak 1: omit g(0); the verifier recovers it from g(0)+g(1) = claim.
        let evals = reduction.round_evals();
        let sent = evals[1..].to_vec();
        transcript.absorb_exts(&sent);
        reduction_round_polys.push(sent);
        let ch = transcript.challenge();
        reduction.bind(ch);
        u.push(ch);
    }

    // --- The Basefold opening of `Φ̃(u)`.
    let mut matrices: Vec<&CommittedMatrix> = stage_matrices.iter().collect();
    matrices.push(const_matrix);
    matrices.extend(custom_matrices.iter());
    let phi_codeword = Pcs::combine_codewords(&matrices, &col_coeffs);

    let opening = Pcs::open(params, &mut transcript, phi_table, &u, phi_codeword, &matrices);
    let t_opening = t_opening.elapsed();

    log::debug!(
        "ml prove_air[{}] n={m}: commit={t_commit:.1?} zerocheck={t_zerocheck:.1?} claims={t_claims:.1?} opening={t_opening:.1?}",
        ir.name,
    );

    Ok(MlProof {
        airgroup_id,
        air_id,
        n_bits,
        stage_roots: stage_matrices.iter().map(|m| m.root()).collect(),
        const_root: const_matrix.root(),
        custom_roots: custom_matrices.iter().map(|m| m.root()).collect(),
        zerocheck_round_polys,
        claims,
        reduction_round_polys,
        opening,
        publics: publics.to_vec(),
        challenges: challenges.to_vec(),
        air_values: air_values.to_vec(),
        airgroup_values: airgroup_values.to_vec(),
        global_instance_id: 0,
    })
}

/// Seed the transcript with the statement: AIR identity and public inputs.
pub(crate) fn seed_transcript(
    transcript: &mut MlTranscript,
    airgroup_id: u32,
    air_id: u32,
    n_bits: u32,
    publics: &[Goldilocks],
) {
    transcript.absorb(&[Goldilocks::from_u32(airgroup_id), Goldilocks::from_u32(air_id), Goldilocks::from_u32(n_bits)]);
    transcript.absorb(publics);
}

/// The kernel's table over the hypercube `{0,1}^m` (prover side).
pub(crate) fn kernel_table(spec: &KernelSpec, l: usize, gamma: Ext, lambda_x: &[Ext]) -> Vec<Ext> {
    match spec {
        KernelSpec::Rot(s) => {
            let base = skip_kernel_table(l, gamma, lambda_x);
            if *s == 0 {
                base
            } else {
                rotate_table(&base, *s as i64)
            }
        }
        KernelSpec::Point(row) => {
            let mut t = vec![Ext::ZERO; 1 << (l + lambda_x.len())];
            t[*row as usize] = Ext::ONE;
            t
        }
    }
}

/// Indices of the challenges the multilinear protocol actually derives:
/// those of stages `2..=n_stages`. Later stages (quotient/evals/FRI batching)
/// belong to the univariate protocol and stay zero.
pub(crate) fn challenge_ids_for_stage(ir: &AirIr, stage: u8) -> impl Iterator<Item = usize> + '_ {
    ir.challenge_stages.iter().enumerate().filter(move |&(_, &st)| st == stage).map(|(i, _)| i)
}

/// Absorb, in global order, the air-value then airgroup-value messages that are
/// outputs of `stage` (`airvalue_stages[j] == stage`, resp. airgroup). Keeping
/// them with their stage's root is what the multi-stage protocol requires: the
/// stage-(i+1) challenge must bind stage-i's value messages.
pub(crate) fn absorb_stage_values(
    transcript: &mut MlTranscript,
    ir: &AirIr,
    air_values: &[Ext],
    airgroup_values: &[Ext],
    stage: u8,
) {
    for (j, &st) in ir.airvalue_stages.iter().enumerate() {
        if st == stage {
            transcript.absorb_ext(&air_values[j]);
        }
    }
    for (j, &st) in ir.airgroupvalue_stages.iter().enumerate() {
        if st == stage {
            transcript.absorb_ext(&airgroup_values[j]);
        }
    }
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
