//! The multilinear STARK verifier.

use crate::pcs::{MlPcs, Pcs};
use crate::eq::{eq_eval, rot_kernel_eval, skip_kernel_eval};
use crate::error::MlError;
use crate::evaluator::{constraint_value, eval_constraint_cone, eval_instrs, operand_eval};
use crate::hypercube::{boolean_point, Ext};
use crate::ir::{AirIr, Boundary};
use crate::logup_gkr::{eval_scalar_fraction, verify_bus_phase};
use crate::prover::{powers, seed_transcript, MlProof};
use crate::sumcheck::{interpolate_at, verifier_sumcheck_round};
use crate::transcript::MlTranscript;
use crate::zerocheck::{
    build_kernels, constraint_weights, kernel_index_of_boundary, ClaimsAtBusPoint, ClaimsAtCorner, ClaimsAtPoint,
    KernelSpec,
};
use fields::Goldilocks;

/// The kernel's MLE evaluated at the opening point `z` (verifier side).
/// `bus_point` is the LogUp-GKR input-reduction point `v` (empty without a bus).
fn kernel_mle_eval(spec: &KernelSpec, l: usize, gamma: Ext, lambda_x: &[Ext], bus_point: &[Ext], z: &[Ext]) -> Ext {
    match spec {
        KernelSpec::Rot(s) => skip_kernel_eval(l, *s as i64, gamma, lambda_x, z),
        KernelSpec::Point(row) => eq_eval(&boolean_point(*row, z.len()), z),
        KernelSpec::BusRot(s) => rot_kernel_eval(*s as i64, bus_point, z),
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
    expected_proof_values: Option<&[Ext]>,
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
        || proof.proof_values.len() != ir.proofvalue_stages.len()
    {
        return Err(MlError::Malformed("challenge/value vector shape mismatch".into()));
    }
    if let Some(expected) = expected_challenges {
        if proof.challenges != expected {
            return Err(MlError::Malformed("proof challenges do not match the globally derived challenges".into()));
        }
    }
    if let Some(expected) = expected_proof_values {
        if proof.proof_values != expected {
            return Err(MlError::Malformed("proof values do not match the globally shared proof values".into()));
        }
    }
    if proof.bus.is_some() != ir.bus.is_some() {
        return Err(MlError::Malformed("bus phase presence does not match the AIR".into()));
    }

    let n = ir.n_bits as usize;
    let l = ir.params.univariate_skip_bits.min(n);
    let params = &ir.params;
    let kernels = build_kernels(ir);
    let total_cols = ir.total_cols();

    if proof.claims.len() != total_cols || proof.claims.iter().any(|row| row.len() != kernels.len()) {
        return Err(MlError::Malformed("claims matrix has wrong shape".into()));
    }
    // One skip round (if any) + `n − l` Gruen rounds.
    let expected_polys = if l > 0 { n - l + 1 } else { n };
    if proof.zerocheck_round_polys.len() != expected_polys {
        return Err(MlError::Malformed(format!("expected {expected_polys} zerocheck round polynomials")));
    }

    // --- Transcript replay: statement, commitments.
    let mut transcript = MlTranscript::new(params.hash);
    seed_transcript(&mut transcript, ir.airgroup_id, ir.air_id, ir.n_bits, publics);
    transcript.absorb_root(&proof.const_root);
    for root in &proof.custom_roots {
        transcript.absorb_root(root);
    }

    for (stage_idx, root) in proof.stage_roots.iter().enumerate() {
        transcript.absorb_root(root);
        let stage = (stage_idx + 1) as u8;
        crate::prover::absorb_stage_values(
            &mut transcript,
            ir,
            &proof.air_values,
            &proof.airgroup_values,
            &proof.proof_values,
            stage,
        );
        for id in crate::prover::challenge_ids_for_stage(ir, stage + 1) {
            transcript.absorb_ext(&proof.challenges[id]);
        }
    }

    // --- Zerocheck ---
    let r = transcript.challenges(n);
    let alpha = transcript.challenge();
    let d = ir.max_constraint_degree as usize;

    let mut claim = Ext::ZERO;
    let mut skip_gamma = Ext::ZERO;
    let mut lambda_x = Vec::with_capacity(n - l);
    let mut round_polys = proof.zerocheck_round_polys.iter();

    if l > 0 {
        // Skip round 0: v(Z) of degree d·(2^l − 1) at Z = 0..deg. The check is
        // the Gruen check generalised to l variables: Σ_p eq(p, r_P)·v(φ(p)) = prev.
        let v = round_polys.next().expect("skip round polynomial");
        let np = 1usize << l;
        let want = d * (np - 1) + 1;
        if v.len() != want {
            return Err(MlError::Malformed(format!("skip round: expected {want} evaluations")));
        }
        transcript.absorb_exts(v);
        skip_gamma = transcript.challenge();
        let mut lhs = Ext::ZERO;
        for (p, &dp) in crate::eq::d_subgroup(l).iter().enumerate() {
            lhs += eq_eval(&boolean_point(p as u64, l), &r[..l]) * interpolate_at(v, Ext::from_base(dp));
        }
        if lhs != claim {
            return Err(MlError::SumcheckRound { round: 0 });
        }
        claim = interpolate_at(v, skip_gamma);
    }

    // Gruen suffix rounds over r_X = r[l..].
    let n_evals = d;
    for (round, sent) in round_polys.enumerate() {
        if sent.len() != n_evals {
            return Err(MlError::Malformed(format!("zerocheck round {round}: expected {n_evals} evaluations")));
        }
        transcript.absorb_exts(sent);
        let ch = transcript.challenge();
        let rk = r[l + round];
        // sent[0] = g'(1); recover g'(0) = (claim − rₖ·g'(1)) / (1 − rₖ).
        let g0 = (claim - rk * sent[0]) * (Ext::ONE - rk).inverse();
        let mut evals = Vec::with_capacity(d + 1);
        evals.push(g0);
        evals.extend_from_slice(sent);
        claim = interpolate_at(&evals, ch);
        lambda_x.push(ch);
    }

    // --- Bus phase replay: GKR walk + input-layer reduction rounds.
    let bus_ver = match (&ir.bus, &proof.bus) {
        (Some(bus), Some(bp)) => Some(verify_bus_phase(bus, bp, n, &mut transcript)?),
        _ => None,
    };
    let bus_point: Vec<Ext> = bus_ver.as_ref().map(|b| b.point.clone()).unwrap_or_default();

    // --- Claims and batching challenges.
    for row in &proof.claims {
        transcript.absorb_exts(row);
    }
    let delta = transcript.challenge();
    let gamma = transcript.challenge();

    // --- Zerocheck final check: claim == eq(r,λ)·Σ_t α^t C_t(claimed openings).
    let publics_ext = Ext::from_base_batch(publics);
    let src = ClaimsAtPoint {
        ir,
        claims: &proof.claims,
        publics: &publics_ext,
        challenges: &proof.challenges,
        air_values: &proof.air_values,
        airgroup_values: &proof.airgroup_values,
        proof_values: &proof.proof_values,
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
            proof_values: &proof.proof_values,
            kernel,
        };
        let v = eval_constraint_cone(ir, &src, &mut temps, t).to_ext();
        if !v.is_zero() {
            return Err(MlError::Constraint(format!("boundary value {v}"), t));
        }
    }

    // --- Bus final checks: the reduction claim against the claimed openings
    // at `v`, and the result airgroup value against the output fraction.
    if let (Some(bus), Some(bv)) = (&ir.bus, &bus_ver) {
        let src = ClaimsAtBusPoint {
            ir,
            kernels: &kernels,
            claims: &proof.claims,
            publics: &publics_ext,
            challenges: &proof.challenges,
            air_values: &proof.air_values,
            airgroup_values: &proof.airgroup_values,
            proof_values: &proof.proof_values,
        };
        eval_instrs(ir, &src, &mut temps);
        let mut batched = Ext::ZERO;
        for (t, term) in bus.terms.iter().enumerate() {
            let num_v = operand_eval(ir, &src, &temps, &term.num).to_ext();
            let den_v = operand_eval(ir, &src, &temps, &term.den).to_ext();
            batched += bv.term_weights[t] * (num_v + bv.mu * den_v);
        }
        if batched != bv.final_claim {
            return Err(MlError::FinalCheck("bus reduction claim inconsistent with claimed openings".into()));
        }

        // result·q_out·s_den == p_out·s_den + s_num·q_out  (no inversions).
        let (s_num, s_den) = eval_scalar_fraction(
            ir,
            bus,
            publics,
            &proof.challenges,
            &proof.air_values,
            &proof.airgroup_values,
            &proof.proof_values,
        )?;
        let expected = match bus.result_airgroupvalue {
            Some(idx) => proof.airgroup_values[idx as usize],
            None => Ext::ZERO,
        };
        if expected * bv.q_out * s_den != bv.p_out * s_den + s_num * bv.q_out {
            return Err(MlError::FinalCheck("bus result airgroup value inconsistent with the output fraction".into()));
        }
    }

    // --- Opening reduction ---
    let col_coeffs = powers(delta, total_cols);
    let kernel_weights = powers(gamma, kernels.len());
    let mut sigma = Ext::ZERO;
    for (j, row) in proof.claims.iter().enumerate() {
        for (i, v) in row.iter().enumerate() {
            sigma += col_coeffs[j] * kernel_weights[i] * *v;
        }
    }

    if proof.reduction_round_polys.len() != n {
        return Err(MlError::Malformed(format!("expected {n} opening-reduction round polynomials")));
    }
    let mut red_claim = sigma;
    let mut u = Vec::with_capacity(n);
    for (t, evals) in proof.reduction_round_polys.iter().enumerate() {
        // Tweak 1: the prover omits g(0), sending the 2 evals g(1), g(2) of the
        // degree-2 round poly; `verifier_sumcheck_round` recovers g(0).
        if evals.len() != 2 {
            return Err(MlError::Malformed(format!("reduction round {t}: expected 2 evaluations")));
        }
        transcript.absorb_exts(evals);
        let ch = transcript.challenge();
        red_claim = verifier_sumcheck_round(red_claim, evals, ch, t)?;
        u.push(ch);
    }

    // The remaining claim is Φ̃(u)·W̃(u); the batched kernel MLE at `u` is
    // verifier-evaluable in O(kernels·n), leaving `Φ̃(u)` for the PCS.
    let w_at_u: Ext = kernels
        .iter()
        .zip(kernel_weights.iter())
        .map(|(spec, w)| *w * kernel_mle_eval(spec, l, skip_gamma, &lambda_x, &bus_point, &u))
        .sum();
    if w_at_u.is_zero() {
        return Err(MlError::FinalCheck("batched kernel vanishes at the reduction point".into()));
    }
    let phi_at_u = red_claim * w_at_u.inverse();

    // --- Batched Basefold opening of `Φ̃(u)`.
    let mut roots: Vec<[Goldilocks; 4]> = proof.stage_roots.clone();
    roots.push(proof.const_root);
    roots.extend(proof.custom_roots.iter().copied());
    let mut stage_n_cols: Vec<usize> = ir.cols_per_stage.iter().map(|&c| c as usize).collect();
    stage_n_cols.push(ir.n_const_cols as usize);
    stage_n_cols.extend(ir.custom_commits.iter().map(|c| c.n_cols as usize));

    Pcs::verify(params, &mut transcript, n, phi_at_u, &proof.opening, &roots, &stage_n_cols, &col_coeffs, &u)?;

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
        MlParams {
            log_blowup: 2,
            n_queries: 12,
            log_final_poly_len: 2,
            grinding_bits: 0,
            univariate_skip_bits: 0,
            hash: crate::MlHashFamily::Poseidon2,
        }
    }

    #[test]
    fn prove_verify_roundtrip() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove");
        verify_air(&ir, &proof, &publics, None, None, None).expect("verify");
    }

    /// Full prove→verify with the univariate skip enabled (`l = 1..3`), which
    /// exercises the skip round, the skip-kernel opening, and shifted (offset-1)
    /// reads through the skip block.
    #[test]
    fn prove_verify_roundtrip_with_skip() {
        let n_bits = 6;
        for l in 1..=3usize {
            let params = MlParams { univariate_skip_bits: l, ..test_params() };
            let ir = fib_ir(n_bits, params);
            let (witness, consts, publics) = fib_trace(n_bits);
            let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove");
            verify_air(&ir, &proof, &publics, None, None, None).unwrap_or_else(|e| panic!("verify l={l}: {e}"));
        }
    }

    /// A corrupted trace must be rejected with the skip enabled too.
    #[test]
    fn corrupted_trace_rejected_with_skip() {
        let n_bits = 6;
        for l in 1..=3usize {
            let params = MlParams { univariate_skip_bits: l, ..test_params() };
            let ir = fib_ir(n_bits, params);
            let (mut witness, consts, publics) = fib_trace(n_bits);
            witness[0][0][7] += Goldilocks::ONE;
            let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove runs");
            assert!(
                verify_air(&ir, &proof, &publics, None, None, None).is_err(),
                "invalid trace must not verify (l={l})"
            );
        }
    }

    #[test]
    fn corrupted_trace_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (mut witness, consts, publics) = fib_trace(n_bits);
        witness[0][0][7] += Goldilocks::ONE;
        let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove runs");
        assert!(verify_air(&ir, &proof, &publics, None, None, None).is_err(), "invalid trace must not verify");
    }

    /// A prebuilt fixed-column commitment, saved and reloaded as a proving-key
    /// artifact, must yield a proof identical to the one built inline — and it
    /// must verify. Locks in the `commit_matrix`/`CommittedMatrix::{save,load}`
    /// reuse path that the setup `.mlconst.bin` artifact feeds into `prove_air`.
    #[test]
    fn reused_const_matrix_matches_inline() {
        use crate::pcs::{MlPcs, Pcs};

        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);

        // Build, persist, and reload the fixed-column commitment via the active
        // PCS (the `.mlconst.bin` reuse path setup feeds into `prove_air`).
        let const_refs: Vec<&[Goldilocks]> = consts.iter().map(|c| c.as_slice()).collect();
        let built = Pcs::commit(&const_refs, &ir.params);
        let path = std::env::temp_dir().join(format!("ml_reused_const_{}.bin", std::process::id()));
        Pcs::save_commitment(&built, &path).expect("save const matrix");
        let loaded = Pcs::load_commitment(&path).expect("load const matrix");
        let _ = std::fs::remove_file(&path);

        // Reloaded root must match the freshly built one.
        assert_eq!(Pcs::commitment_root(&loaded), Pcs::commitment_root(&built), "reloaded root differs");

        // A proof reusing the artifact must be byte-identical to one that builds
        // the commitment inline (same transcript, same challenges).
        let inline = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove inline");
        let reused =
            prove_air(&ir, &witness, &consts, Some(&loaded), &[], &publics, &[], &[], &[], &[]).expect("prove reused");
        assert_eq!(reused.const_root, inline.const_root, "const roots differ");
        let enc = |p: &crate::MlProof| bincode::serde::encode_to_vec(p, bincode::config::standard()).expect("encode");
        assert_eq!(enc(&reused), enc(&inline), "reused proof differs from inline proof");
        verify_air(&ir, &reused, &publics, None, None, None).expect("reused proof must verify");
    }

    #[test]
    fn wrong_publics_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove");

        // Different publics break both the transcript binding and the
        // first-row constraints.
        let bad = vec![Goldilocks::TWO, Goldilocks::TWO];
        assert!(verify_air(&ir, &proof, &bad, None, None, None).is_err());
    }

    #[test]
    fn tampered_proof_rejected() {
        let n_bits = 5;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);

        // Tamper with a claimed opening.
        let mut proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove");
        proof.claims[0][0] += Ext::ONE;
        assert!(verify_air(&ir, &proof, &publics, None, None, None).is_err());

        // Tamper with a zerocheck round polynomial.
        let mut proof2 = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove");
        proof2.zerocheck_round_polys[0][0] += Ext::ONE;
        assert!(verify_air(&ir, &proof2, &publics, None, None, None).is_err());

        // Tamper with the final polynomial.
        let mut proof3 = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove");
        proof3.opening.final_poly[0] += Ext::ONE;
        assert!(verify_air(&ir, &proof3, &publics, None, None, None).is_err());

        // Tamper with an opening-reduction round polynomial.
        let mut proof4 = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove");
        proof4.reduction_round_polys[1][0] += Ext::ONE;
        assert!(verify_air(&ir, &proof4, &publics, None, None, None).is_err());
    }

    // --- LogUp-GKR bus tests on the hand-built lookup AIR ---

    use crate::evaluator::test_air::{lookup_ir, lookup_trace};

    fn random_gamma() -> Ext {
        use fields::PrimeField64;
        use rand::{rng, RngExt};
        let mut r = rng();
        Ext::from_array(&[
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
        ])
    }

    /// A balanced lookup proves and verifies with a zero bus result.
    #[test]
    fn bus_prove_verify_roundtrip() {
        let n_bits = 5;
        let ir = lookup_ir(n_bits, test_params(), false);
        let (witness, consts) = lookup_trace(n_bits);
        let gamma = random_gamma();

        let proof = prove_air(&ir, &witness, &consts, None, &[], &[], &[gamma], &[], &[Ext::ZERO], &[]).expect("prove");
        verify_air(&ir, &proof, &[], None, None, None).expect("verify");
        assert_eq!(proof.airgroup_values[0], Ext::ZERO, "balanced lookup must have zero bus result");
    }

    /// Bus + univariate skip: the bus reduction point coexists with the
    /// skip-form zerocheck kernels.
    #[test]
    fn bus_prove_verify_roundtrip_with_skip() {
        let n_bits = 5;
        for l in 1..=2usize {
            let params = MlParams { univariate_skip_bits: l, ..test_params() };
            let ir = lookup_ir(n_bits, params, false);
            let (witness, consts) = lookup_trace(n_bits);
            let gamma = random_gamma();
            let proof =
                prove_air(&ir, &witness, &consts, None, &[], &[], &[gamma], &[], &[Ext::ZERO], &[]).expect("prove");
            verify_air(&ir, &proof, &[], None, None, None).unwrap_or_else(|e| panic!("verify l={l}: {e}"));
        }
    }

    /// Scalar ("direct") bus terms enter the result airgroup value and its
    /// consistency check.
    #[test]
    fn bus_scalar_term_enters_result() {
        use fields::PrimeField64;
        let n_bits = 4;
        let ir = lookup_ir(n_bits, test_params(), true);
        let (witness, consts) = lookup_trace(n_bits);
        let gamma = random_gamma();
        let publics = vec![Goldilocks::from_u64(5), Goldilocks::from_u64(9)];
        let pv = Ext::from_base(Goldilocks::from_u64(7));

        let proof =
            prove_air(&ir, &witness, &consts, None, &[], &publics, &[gamma], &[], &[Ext::ZERO], &[pv]).expect("prove");
        verify_air(&ir, &proof, &publics, None, None, Some(&[pv])).expect("verify");

        // result = 0 (balanced rows) + pub0/(γ + pub1 + pv).
        let expected = Ext::from_base(publics[0]) * (gamma + publics[1] + pv).inverse();
        assert_eq!(proof.airgroup_values[0], expected);

        // A different proof value must be rejected by the expected check.
        let other = Ext::from_base(Goldilocks::from_u64(8));
        assert!(verify_air(&ir, &proof, &publics, None, None, Some(&[other])).is_err());
    }

    /// A corrupted multiplicity still yields a VALID per-instance proof — the
    /// imbalance surfaces only in the (nonzero) bus result, which the global
    /// balance check rejects at the set level.
    #[test]
    fn corrupted_multiplicity_shifts_bus_result() {
        let n_bits = 4;
        let ir = lookup_ir(n_bits, test_params(), false);
        let (mut witness, consts) = lookup_trace(n_bits);
        witness[0][2][3] += Goldilocks::ONE; // mul[3] += 1
        let gamma = random_gamma();

        let proof = prove_air(&ir, &witness, &consts, None, &[], &[], &[gamma], &[], &[Ext::ZERO], &[]).expect("prove");
        verify_air(&ir, &proof, &[], None, None, None).expect("per-instance proof stays valid");
        assert_ne!(proof.airgroup_values[0], Ext::ZERO, "imbalance must show in the bus result");
    }

    /// Tampering with the bus result (claiming balance for an unbalanced
    /// trace) must be rejected by the result consistency check.
    #[test]
    fn tampered_bus_result_rejected() {
        let n_bits = 4;
        let ir = lookup_ir(n_bits, test_params(), false);
        let (mut witness, consts) = lookup_trace(n_bits);
        witness[0][2][3] += Goldilocks::ONE;
        let gamma = random_gamma();

        let mut proof =
            prove_air(&ir, &witness, &consts, None, &[], &[], &[gamma], &[], &[Ext::ZERO], &[]).expect("prove");
        proof.airgroup_values[0] = Ext::ZERO;
        assert!(verify_air(&ir, &proof, &[], None, None, None).is_err(), "faked balance must be rejected");
    }

    /// Tampering with bus messages must be rejected.
    #[test]
    fn tampered_bus_messages_rejected() {
        let n_bits = 4;
        let ir = lookup_ir(n_bits, test_params(), false);
        let (witness, consts) = lookup_trace(n_bits);
        let gamma = random_gamma();
        let prove =
            || prove_air(&ir, &witness, &consts, None, &[], &[], &[gamma], &[], &[Ext::ZERO], &[]).expect("prove");

        let mut p1 = prove();
        p1.bus.as_mut().unwrap().fractional.p_out += Ext::ONE;
        assert!(verify_air(&ir, &p1, &[], None, None, None).is_err(), "tampered p_out");

        let mut p2 = prove();
        p2.bus.as_mut().unwrap().fractional.layer_claims[2][0] += Ext::ONE;
        assert!(verify_air(&ir, &p2, &[], None, None, None).is_err(), "tampered split value");

        let mut p3 = prove();
        p3.bus.as_mut().unwrap().reduction_round_polys[0][0] += Ext::ONE;
        assert!(verify_air(&ir, &p3, &[], None, None, None).is_err(), "tampered reduction round");

        let mut p4 = prove();
        let last = p4.bus.as_mut().unwrap().fractional.layer_round_polys.last_mut().unwrap();
        last[0][0] += Ext::ONE;
        assert!(verify_air(&ir, &p4, &[], None, None, None).is_err(), "tampered walk round");
    }

    #[test]
    fn proof_serialization_roundtrip() {
        let n_bits = 4;
        let ir = fib_ir(n_bits, test_params());
        let (witness, consts, publics) = fib_trace(n_bits);
        let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[], &[]).expect("prove");

        let dir = std::env::temp_dir().join("ml_proof_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fib.mlproof.bin");
        proof.save(&path).expect("save");
        let loaded = MlProof::load(&path).expect("load");
        verify_air(&ir, &loaded, &publics, None, None, None).expect("verify loaded proof");
    }
}
