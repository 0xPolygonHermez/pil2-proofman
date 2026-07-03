//! The multilinear STARK verifier.

use crate::basefold::verify_opening;
use crate::eq::{boolean_point, eq_eval, rot_kernel_eval};
use crate::error::MlError;
use crate::evaluator::{constraint_value, eval_constraint_cone, eval_instrs};
use crate::hypercube::{to_ext_vec, Ext};
use crate::ir::{AirIr, Boundary};
use crate::prover::{powers, seed_transcript, MlProof};
use crate::sumcheck::verify_sumcheck_round;
use crate::transcript::MlTranscript;
use crate::zerocheck::{
    build_kernels, constraint_weights, kernel_index_of_boundary, ClaimsAtCorner, ClaimsAtPoint, KernelSpec,
};
use fields::Goldilocks;

/// The kernel's MLE evaluated at an arbitrary point `z` (verifier side).
fn kernel_mle_eval(spec: &KernelSpec, n_bits: usize, lambda: &[Ext], z: &[Ext]) -> Ext {
    match spec {
        KernelSpec::Rot(s) => rot_kernel_eval(*s as i64, lambda, z),
        KernelSpec::Point(row) => eq_eval(&boolean_point(*row, n_bits), z),
    }
}

/// Verify a multilinear STARK proof against `ir` and the public inputs.
///
/// `expected_const_root` is the trusted commitment to the fixed columns
/// (computed at setup time); pass `None` to accept the root carried in the
/// proof (useful in tests).
pub fn verify_air(
    ir: &AirIr,
    proof: &MlProof,
    publics: &[Goldilocks],
    expected_const_root: Option<&[Goldilocks; 4]>,
) -> Result<(), MlError> {
    if !ir.challenge_stages.is_empty() {
        return Err(MlError::Unsupported("multi-stage AIRs with challenges (milestone 2)".into()));
    }
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

    let n = ir.n_bits as usize;
    let params = &ir.params;
    let kernels = build_kernels(ir);
    let total_cols = ir.total_cols();

    if proof.claims.len() != total_cols || proof.claims.iter().any(|row| row.len() != kernels.len()) {
        return Err(MlError::Malformed("claims matrix has wrong shape".into()));
    }
    if proof.zerocheck_round_polys.len() != n {
        return Err(MlError::Malformed(format!("expected {n} zerocheck round polynomials")));
    }

    // --- Transcript replay: statement, commitments.
    let mut transcript = MlTranscript::new();
    seed_transcript(&mut transcript, ir, publics);
    transcript.absorb_root(&proof.const_root);
    for root in &proof.stage_roots {
        transcript.absorb_root(root);
    }

    // --- Zerocheck.
    let r = transcript.challenges(n);
    let alpha = transcript.challenge();
    let n_evals = ir.max_constraint_degree as usize + 2;

    let mut claim = Ext::zero();
    let mut lambda = Vec::with_capacity(n);
    for (round, evals) in proof.zerocheck_round_polys.iter().enumerate() {
        if evals.len() != n_evals {
            return Err(MlError::Malformed(format!("zerocheck round {round}: expected {n_evals} evaluations")));
        }
        transcript.absorb_exts(evals);
        let ch = transcript.challenge();
        claim = verify_sumcheck_round(claim, evals, ch, round)?;
        lambda.push(ch);
    }

    // --- Claims and batching challenges.
    for row in &proof.claims {
        transcript.absorb_exts(row);
    }
    let delta = transcript.challenge();
    let gamma = transcript.challenge();

    // --- Zerocheck final check: claim == eq(r,λ)·Σ_t α^t C_t(claimed openings).
    let publics_ext = to_ext_vec(publics);
    let challenges: Vec<Ext> = Vec::new(); // milestone 2
    let src = ClaimsAtPoint { ir, claims: &proof.claims, publics: &publics_ext, challenges: &challenges };
    let mut temps = Vec::new();
    eval_instrs(ir, &src, &mut temps);
    let weights = constraint_weights(ir, alpha);
    let mut batched = Ext::zero();
    for (t, w) in weights.iter().enumerate() {
        if !w.is_zero() {
            batched += *w * constraint_value(ir, &src, &temps, t);
        }
    }
    if claim != eq_eval(&r, &lambda) * batched {
        return Err(MlError::FinalCheck("zerocheck claim inconsistent with claimed openings".into()));
    }

    // --- Boundary constraints: evaluate their dependency cones on the corner claims.
    for (t, c) in ir.constraints.iter().enumerate() {
        if c.boundary == Boundary::EveryRow {
            continue;
        }
        let kernel = kernel_index_of_boundary(ir, &kernels, c.boundary);
        let src = ClaimsAtCorner { ir, claims: &proof.claims, publics: &publics_ext, challenges: &challenges, kernel };
        let v = eval_constraint_cone(ir, &src, &mut temps, t);
        if !v.is_zero() {
            return Err(MlError::Constraint(format!("boundary value {v}"), t));
        }
    }

    // --- Batched Basefold opening of all claims.
    let col_coeffs = powers(delta, total_cols);
    let kernel_weights = powers(gamma, kernels.len());
    let mut sigma = Ext::zero();
    for (j, row) in proof.claims.iter().enumerate() {
        for (i, v) in row.iter().enumerate() {
            sigma += col_coeffs[j] * kernel_weights[i] * *v;
        }
    }

    let mut roots: Vec<[Goldilocks; 4]> = proof.stage_roots.clone();
    roots.push(proof.const_root);
    let mut stage_n_cols: Vec<usize> = ir.cols_per_stage.iter().map(|&c| c as usize).collect();
    stage_n_cols.push(ir.n_const_cols as usize);

    verify_opening(params, &mut transcript, n, sigma, &proof.opening, &roots, &stage_n_cols, &col_coeffs, |z| {
        kernels.iter().zip(kernel_weights.iter()).map(|(spec, w)| *w * kernel_mle_eval(spec, n, &lambda, z)).sum()
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basefold::MlParams;
    use crate::evaluator::test_air::{fib_ir, fib_trace};
    use crate::prover::prove_air;
    use fields::Field;

    fn test_params() -> MlParams {
        MlParams { log_blowup: 2, n_queries: 12, log_final_poly_len: 2, grinding_bits: 0 }
    }

    #[test]
    fn prove_verify_roundtrip() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, &publics).expect("prove");
        verify_air(&ir, &proof, &publics, None).expect("verify");
    }

    #[test]
    fn corrupted_trace_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (mut witness, consts, publics) = fib_trace(n_bits);
        witness[0][0][7] += Goldilocks::ONE;
        let proof = prove_air(&ir, &witness, &consts, &publics).expect("prove runs");
        assert!(verify_air(&ir, &proof, &publics, None).is_err(), "invalid trace must not verify");
    }

    #[test]
    fn wrong_publics_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, &publics).expect("prove");

        // Different publics break both the transcript binding and the
        // first-row constraints.
        let bad = vec![Goldilocks::TWO, Goldilocks::TWO];
        assert!(verify_air(&ir, &proof, &bad, None).is_err());
    }

    #[test]
    fn tampered_proof_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);

        // Tamper with a claimed opening.
        let mut proof = prove_air(&ir, &witness, &consts, &publics).expect("prove");
        proof.claims[0][0] += Ext::one();
        assert!(verify_air(&ir, &proof, &publics, None).is_err());

        // Tamper with a zerocheck round polynomial.
        let mut proof2 = prove_air(&ir, &witness, &consts, &publics).expect("prove");
        proof2.zerocheck_round_polys[0][0] += Ext::one();
        assert!(verify_air(&ir, &proof2, &publics, None).is_err());

        // Tamper with the final polynomial.
        let mut proof3 = prove_air(&ir, &witness, &consts, &publics).expect("prove");
        proof3.opening.final_poly[0] += Ext::one();
        assert!(verify_air(&ir, &proof3, &publics, None).is_err());
    }

    #[test]
    fn proof_serialization_roundtrip() {
        let n_bits = 4;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, &publics).expect("prove");

        let dir = std::env::temp_dir().join("ml_proof_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fib.mlproof.bin");
        proof.save(&path).expect("save");
        let loaded = MlProof::load(&path).expect("load");
        verify_air(&ir, &loaded, &publics, None).expect("verify loaded proof");
    }
}
