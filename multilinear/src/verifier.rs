//! The multilinear STARK verifier.

use crate::pcs::{MlPcs, Pcs};
use crate::eq::{eq_eval, skip_kernel_eval};
use crate::error::MlError;
use crate::evaluator::{constraint_value, eval_constraint_cone, eval_instrs};
use crate::hypercube::{to_ext_vec, boolean_point, Ext};
use crate::ir::{AirIr, Boundary};
use crate::prover::{powers, seed_transcript, MlProof};
use crate::sumcheck::interpolate_at;
use crate::transcript::MlTranscript;
use crate::zerocheck::{
    build_kernels, constraint_weights, kernel_index_of_boundary, ClaimsAtCorner, ClaimsAtPoint, KernelSpec,
};
use fields::Goldilocks;

/// The kernel's MLE evaluated at the opening point `z` (verifier side).
fn kernel_mle_eval(spec: &KernelSpec, ell: usize, gamma: Ext, lambda_x: &[Ext], z: &[Ext]) -> Ext {
    match spec {
        KernelSpec::Rot(s) => skip_kernel_eval(ell, *s as i64, gamma, lambda_x, z),
        KernelSpec::Point(row) => eq_eval(&boolean_point(*row, z.len()), z),
    }
}

/// Verify a multilinear STARK proof against `ir` and the public inputs.
///
/// `expected_const_root` is the trusted commitment to the fixed columns
/// (computed at setup time); pass `None` to accept the root carried in the
/// proof (useful in tests).
///
/// `expected_challenges`: with a shared bus, the stage challenges are derived
/// globally from every instance's stage-1 commitment (by the orchestrator, the
/// same way the univariate prover derives its global challenge); the caller
/// passes them here. `None` accepts the challenges carried in the proof.
pub fn verify_air(
    ir: &AirIr,
    proof: &MlProof,
    publics: &[Goldilocks],
    expected_const_root: Option<&[Goldilocks; 4]>,
    expected_challenges: Option<&[Ext]>,
) -> Result<(), MlError> {
    if proof.airgroup_id != ir.airgroup_id || proof.air_id != ir.air_id || proof.n_bits != ir.n_bits {
        return Err(MlError::Malformed("proof does not match the AIR".into()));
    }
    if publics.len() != ir.n_publics as usize {
        return Err(MlError::Malformed(format!("expected {} public inputs", ir.n_publics)));
    }
    if proof.stage_roots.len() != ir.n_stages() {
        return Err(MlError::Malformed(format!("expected {} stage roots", ir.n_stages())));
    }
    if let Some(expected) = expected_const_root {
        if proof.const_root != *expected {
            return Err(MlError::Malformed("const-column commitment does not match the verifying key".into()));
        }
    }
    if proof.custom_roots.len() != ir.custom_commits.len() {
        return Err(MlError::Malformed("custom commit root count mismatch".into()));
    }
    if proof.challenges.len() != ir.challenge_stages.len()
        || proof.air_values.len() != ir.airvalue_stages.len()
        || proof.airgroup_values.len() != ir.airgroupvalue_stages.len()
    {
        return Err(MlError::Malformed("challenge/value vector shape mismatch".into()));
    }
    if let Some(expected) = expected_challenges {
        if proof.challenges != expected {
            return Err(MlError::Malformed("proof challenges do not match the globally derived challenges".into()));
        }
    }

    let n = ir.n_bits as usize;
    let ell = ir.params.univariate_skip_bits.min(n);
    let params = &ir.params;
    let kernels = build_kernels(ir);
    let total_cols = ir.total_cols();

    if proof.claims.len() != total_cols || proof.claims.iter().any(|row| row.len() != kernels.len()) {
        return Err(MlError::Malformed("claims matrix has wrong shape".into()));
    }
    // One skip round (if any) + `n − ell` Gruen rounds.
    let expected_polys = if ell > 0 { n - ell + 1 } else { n };
    if proof.zerocheck_round_polys.len() != expected_polys {
        return Err(MlError::Malformed(format!("expected {expected_polys} zerocheck round polynomials")));
    }

    // --- Transcript replay: statement, commitments.
    let mut transcript = MlTranscript::new();
    seed_transcript(&mut transcript, ir, publics);
    transcript.absorb_root(&proof.const_root);
    for root in &proof.custom_roots {
        transcript.absorb_root(root);
    }
    for (stage_idx, root) in proof.stage_roots.iter().enumerate() {
        transcript.absorb_root(root);
        if stage_idx == 0 {
            for id in crate::prover::derived_challenge_ids(ir) {
                transcript.absorb_ext(&proof.challenges[id]);
            }
        }
    }
    transcript.absorb_exts(&proof.air_values);
    transcript.absorb_exts(&proof.airgroup_values);

    // --- Zerocheck ---
    let r = transcript.challenges(n);
    let alpha = transcript.challenge();
    let d = ir.max_constraint_degree as usize;

    let mut claim = Ext::ZERO;
    let mut skip_gamma = Ext::ZERO;
    let mut lambda_x = Vec::with_capacity(n - ell);
    let mut round_polys = proof.zerocheck_round_polys.iter();

    if ell > 0 {
        // Skip round 0: v(Z) of degree d·(2^ell − 1) at Z = 0..deg. The check is
        // the Gruen check generalised to ell variables: Σ_p eq(p, r_P)·v(φ(p)) = prev.
        let v = round_polys.next().expect("skip round polynomial");
        let np = 1usize << ell;
        let want = d * (np - 1) + 1;
        if v.len() != want {
            return Err(MlError::Malformed(format!("skip round: expected {want} evaluations")));
        }
        transcript.absorb_exts(v);
        skip_gamma = transcript.challenge();
        let mut lhs = Ext::ZERO;
        for (p, &dp) in crate::eq::d_subgroup(ell).iter().enumerate() {
            lhs += eq_eval(&boolean_point(p as u64, ell), &r[..ell]) * interpolate_at(v, Ext::from_base(dp));
        }
        if lhs != claim {
            return Err(MlError::SumcheckRound { round: 0 });
        }
        claim = interpolate_at(v, skip_gamma);
    }

    // Gruen suffix rounds over r_X = r[ell..].
    let n_evals = d + 1;
    for (round, evals) in round_polys.enumerate() {
        if evals.len() != n_evals {
            return Err(MlError::Malformed(format!("zerocheck round {round}: expected {n_evals} evaluations")));
        }
        transcript.absorb_exts(evals);
        let ch = transcript.challenge();
        let rk = r[ell + round];
        if (Ext::ONE - rk) * evals[0] + rk * evals[1] != claim {
            return Err(MlError::SumcheckRound { round: round + usize::from(ell > 0) });
        }
        claim = interpolate_at(evals, ch);
        lambda_x.push(ch);
    }

    // --- Claims and batching challenges.
    for row in &proof.claims {
        transcript.absorb_exts(row);
    }
    let delta = transcript.challenge();
    let gamma = transcript.challenge();

    // --- Zerocheck final check: claim == eq(r,λ)·Σ_t α^t C_t(claimed openings).
    let publics_ext = to_ext_vec(publics);
    let src = ClaimsAtPoint {
        ir,
        claims: &proof.claims,
        publics: &publics_ext,
        challenges: &proof.challenges,
        air_values: &proof.air_values,
        airgroup_values: &proof.airgroup_values,
    };
    let mut temps = Vec::new();
    eval_instrs(ir, &src, &mut temps);
    let weights = constraint_weights(ir, alpha);
    let mut batched = Ext::ZERO;
    for (t, w) in weights.iter().enumerate() {
        if !w.is_zero() {
            batched += *w * constraint_value(ir, &src, &temps, t).to_ext();
        }
    }

    if claim != batched {
        return Err(MlError::FinalCheck("zerocheck claim inconsistent with claimed openings".into()));
    }

    // --- Boundary constraints: evaluate their dependency cones on the corner claims.
    for (t, c) in ir.constraints.iter().enumerate() {
        if c.boundary == Boundary::EveryRow {
            continue;
        }
        let kernel = kernel_index_of_boundary(ir, &kernels, c.boundary);
        let src = ClaimsAtCorner {
            ir,
            claims: &proof.claims,
            publics: &publics_ext,
            challenges: &proof.challenges,
            air_values: &proof.air_values,
            airgroup_values: &proof.airgroup_values,
            kernel,
        };
        let v = eval_constraint_cone(ir, &src, &mut temps, t).to_ext();
        if !v.is_zero() {
            return Err(MlError::Constraint(format!("boundary value {v}"), t));
        }
    }

    // --- Batched Basefold opening of all claims.
    let col_coeffs = powers(delta, total_cols);
    let kernel_weights = powers(gamma, kernels.len());
    let mut sigma = Ext::ZERO;
    for (j, row) in proof.claims.iter().enumerate() {
        for (i, v) in row.iter().enumerate() {
            sigma += col_coeffs[j] * kernel_weights[i] * *v;
        }
    }

    let mut roots: Vec<[Goldilocks; 4]> = proof.stage_roots.clone();
    roots.push(proof.const_root);
    roots.extend(proof.custom_roots.iter().copied());
    let mut stage_n_cols: Vec<usize> = ir.cols_per_stage.iter().map(|&c| c as usize).collect();
    stage_n_cols.push(ir.n_const_cols as usize);
    stage_n_cols.extend(ir.custom_commits.iter().map(|c| c.n_cols as usize));

    Pcs::verify(params, &mut transcript, n, sigma, &proof.opening, &roots, &stage_n_cols, &col_coeffs, |z| {
        kernels
            .iter()
            .zip(kernel_weights.iter())
            .map(|(spec, w)| *w * kernel_mle_eval(spec, ell, skip_gamma, &lambda_x, z))
            .sum()
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::MlParams;
    use crate::evaluator::test_air::{fib_ir, fib_trace};
    use crate::prover::prove_air;
    use fields::Field;

    fn test_params() -> MlParams {
        MlParams { log_blowup: 2, n_queries: 12, log_final_poly_len: 2, grinding_bits: 0, univariate_skip_bits: 0 }
    }

    #[test]
    fn prove_verify_roundtrip() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove");
        verify_air(&ir, &proof, &publics, None, None).expect("verify");
    }

    /// Full prove→verify with the univariate skip enabled (`ell = 1..3`), which
    /// exercises the skip round, the skip-kernel opening, and shifted (offset-1)
    /// reads through the skip block.
    #[test]
    fn prove_verify_roundtrip_with_skip() {
        let n_bits = 6;
        for ell in 1..=3usize {
            let params = MlParams { univariate_skip_bits: ell, ..test_params() };
            let ir = fib_ir(n_bits, params);
            let (witness, consts, publics) = fib_trace(n_bits);
            let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove");
            verify_air(&ir, &proof, &publics, None, None).unwrap_or_else(|e| panic!("verify ell={ell}: {e}"));
        }
    }

    /// A corrupted trace must be rejected with the skip enabled too.
    #[test]
    fn corrupted_trace_rejected_with_skip() {
        let n_bits = 6;
        for ell in 1..=3usize {
            let params = MlParams { univariate_skip_bits: ell, ..test_params() };
            let ir = fib_ir(n_bits, params);
            let (mut witness, consts, publics) = fib_trace(n_bits);
            witness[0][0][7] += Goldilocks::ONE;
            let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove runs");
            assert!(
                verify_air(&ir, &proof, &publics, None, None).is_err(),
                "invalid trace must not verify (ell={ell})"
            );
        }
    }

    #[test]
    fn corrupted_trace_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (mut witness, consts, publics) = fib_trace(n_bits);
        witness[0][0][7] += Goldilocks::ONE;
        let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove runs");
        assert!(verify_air(&ir, &proof, &publics, None, None).is_err(), "invalid trace must not verify");
    }

    /// A prebuilt fixed-column commitment, saved and reloaded as a proving-key
    /// artifact, must yield a proof identical to the one built inline — and it
    /// must verify. Locks in the `commit_matrix`/`CommittedMatrix::{save,load}`
    /// reuse path that the setup `.mlconst.bin` artifact feeds into `prove_air`.
    #[test]
    fn reused_const_matrix_matches_inline() {
        use crate::pcs::{commit_matrix, CommittedMatrix};

        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);

        // Build, persist, and reload the fixed-column commitment.
        let const_refs: Vec<&[Goldilocks]> = consts.iter().map(|c| c.as_slice()).collect();
        let built = commit_matrix(&const_refs, &ir.params);
        let path = std::env::temp_dir().join(format!("ml_reused_const_{}.bin", std::process::id()));
        built.save(&path).expect("save const matrix");
        let loaded = CommittedMatrix::load(&path).expect("load const matrix");
        let _ = std::fs::remove_file(&path);

        // Reloaded root and leaves must match the freshly built ones.
        assert_eq!(loaded.root(), built.root(), "reloaded root differs");
        assert_eq!(loaded.leaves, built.leaves, "rebuilt leaves differ");

        // A proof reusing the artifact must be byte-identical to one that builds
        // the commitment inline (same transcript, same challenges).
        let inline = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove inline");
        let reused =
            prove_air(&ir, &witness, &consts, Some(&loaded), &[], &publics, &[], &[], &[]).expect("prove reused");
        assert_eq!(reused.const_root, inline.const_root, "const roots differ");
        let enc = |p: &crate::MlProof| bincode::serde::encode_to_vec(p, bincode::config::standard()).expect("encode");
        assert_eq!(enc(&reused), enc(&inline), "reused proof differs from inline proof");
        verify_air(&ir, &reused, &publics, None, None).expect("reused proof must verify");
    }

    #[test]
    fn wrong_publics_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove");

        // Different publics break both the transcript binding and the
        // first-row constraints.
        let bad = vec![Goldilocks::TWO, Goldilocks::TWO];
        assert!(verify_air(&ir, &proof, &bad, None, None).is_err());
    }

    #[test]
    fn tampered_proof_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);

        // Tamper with a claimed opening.
        let mut proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove");
        proof.claims[0][0] += Ext::ONE;
        assert!(verify_air(&ir, &proof, &publics, None, None).is_err());

        // Tamper with a zerocheck round polynomial.
        let mut proof2 = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove");
        proof2.zerocheck_round_polys[0][0] += Ext::ONE;
        assert!(verify_air(&ir, &proof2, &publics, None, None).is_err());

        // Tamper with the final polynomial.
        let mut proof3 = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove");
        proof3.opening.final_poly[0] += Ext::ONE;
        assert!(verify_air(&ir, &proof3, &publics, None, None).is_err());
    }

    #[test]
    fn proof_serialization_roundtrip() {
        let n_bits = 4;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove");

        let dir = std::env::temp_dir().join("ml_proof_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fib.mlproof.bin");
        proof.save(&path).expect("save");
        let loaded = MlProof::load(&path).expect("load");
        verify_air(&ir, &loaded, &publics, None, None).expect("verify loaded proof");
    }
}
