//! Milestone-2 end-to-end test: a hand-built **two-stage logup AIR**, the
//! minimal shape of a std lookup argument.
//!
//! Stage 1 commits `a` (looked-up values), `k` (table) and `mul`
//! (multiplicities). The stage-2 challenge `α` is derived globally from the
//! stage-1 commitment, then the extension-valued running sum
//!
//!   gsum_i = gsum_{i−1} + 1/(α + a_i) − mul_i/(α + k_i)
//!
//! is committed as three base columns. Constraints (division-free):
//!
//!   EveryRow: (gsum − (1−L1)·gsum^{→−1})·(α+a)(α+k) − [(α+k) − mul·(α+a)] = 0
//!   LastRow:  gsum − gsum_result = 0            (gsum_result: airgroup value)
//!
//! A valid multiset assignment makes `gsum_result = 0`; across instances the
//! global bus constraint (milestone M2d) sums these airgroup values.

use fields::{Field, Goldilocks, PrimeField64};
use proofman_multilinear::{
    check_constraints_on_trace, commit_matrix, derive_global_challenges, prove_air, verify_air, AirIr, Boundary,
    ConstraintIr, Ext, IrBuilder, MlParams,
};

fn ext_from(v: u64) -> Ext {
    Ext::from_array(&[Goldilocks::from_u64(v), Goldilocks::ZERO, Goldilocks::ZERO])
}

fn logup_ir(n_bits: u32, params: MlParams) -> AirIr {
    let mut b = IrBuilder::default();
    let a = b.witness(1, 0, 0);
    let k = b.witness(1, 1, 0);
    let mul_col = b.witness(1, 2, 0);
    let gsum = b.witness_ext(2, 0, 0);
    let gsum_prev = b.witness_ext(2, 0, -1);
    let l1 = b.constant(0, 0);
    let alpha = b.challenge(0);
    let gsum_result = b.airgroup_value(0, 3);
    let one = b.number(1);

    // (gsum − (1−L1)·gsum⁻¹)·(α+a)(α+k) − [(α+k) − mul·(α+a)]
    let apa = b.add(alpha, a);
    let apk = b.add(alpha, k);
    let denom = b.mul(apa, apk);
    let mul_apa = b.mul(mul_col, apa);
    let numer = b.sub(apk, mul_apa);
    let one_minus_l1 = b.sub(one, l1);
    let prev_gated = b.mul(one_minus_l1, gsum_prev);
    let diff = b.sub(gsum, prev_gated);
    let lhs = b.mul(diff, denom);
    let c_every = b.sub(lhs, numer);

    // LastRow: gsum − gsum_result
    let c_last = b.sub(gsum, gsum_result);

    AirIr {
        name: "LogupTest".into(),
        airgroup_id: 0,
        air_id: 0,
        n_bits,
        cols_per_stage: vec![3, 3], // stage 1: a,k,mul; stage 2: gsum (dim 3)
        n_const_cols: 1,
        n_publics: 0,
        challenge_stages: vec![2],
        airvalue_stages: vec![],
        airgroupvalue_stages: vec![2],
        numbers: b.numbers.clone(),
        n_temps: b.n_temps(),
        instrs: b.instrs,
        constraints: vec![
            ConstraintIr { boundary: Boundary::EveryRow, root: c_every, degree: 4 },
            ConstraintIr { boundary: Boundary::LastRow, root: c_last, degree: 1 },
        ],
        max_constraint_degree: 4,
        opening_offsets: vec![-1, 0],
        params,
    }
}

/// Stage-1 trace: table `k = 0..N`, lookups `a_i = i mod N/2` (each low value
/// twice), multiplicities to match.
fn stage1_trace(n_rows: usize) -> Vec<Vec<Goldilocks>> {
    let a: Vec<Goldilocks> = (0..n_rows).map(|i| Goldilocks::from_u64((i % (n_rows / 2)) as u64)).collect();
    let k: Vec<Goldilocks> = (0..n_rows).map(|i| Goldilocks::from_u64(i as u64)).collect();
    let mul: Vec<Goldilocks> =
        (0..n_rows).map(|i| if i < n_rows / 2 { Goldilocks::TWO } else { Goldilocks::ZERO }).collect();
    vec![a, k, mul]
}

/// Stage-2 witness: the running logup sum in the extension field, decomposed
/// into its three coordinate base columns. Returns (columns, final value).
fn stage2_trace(stage1: &[Vec<Goldilocks>], alpha: Ext) -> (Vec<Vec<Goldilocks>>, Ext) {
    let n_rows = stage1[0].len();
    let mut cols = vec![vec![Goldilocks::ZERO; n_rows]; 3];
    let mut acc = Ext::zero();
    for i in 0..n_rows {
        let term = (alpha + stage1[0][i]).inverse() - (alpha + stage1[1][i]).inverse() * stage1[2][i];
        acc += term;
        for c in 0..3 {
            cols[c][i] = acc.value[c];
        }
    }
    (cols, acc)
}

struct Setup {
    ir: AirIr,
    witness: Vec<Vec<Vec<Goldilocks>>>,
    consts: Vec<Vec<Goldilocks>>,
    challenges: Vec<Ext>,
    airgroup_values: Vec<Ext>,
}

fn build(n_bits: u32) -> Setup {
    let params = MlParams { log_blowup: 2, n_queries: 12, log_final_poly_len: 2, grinding_bits: 0 };
    let ir = logup_ir(n_bits, params);
    let n_rows = 1usize << n_bits;

    let stage1 = stage1_trace(n_rows);
    let mut l1 = vec![Goldilocks::ZERO; n_rows];
    l1[0] = Goldilocks::ONE;
    let consts = vec![l1];

    // Global challenge derivation from the stage-1 commitment (the proving
    // orchestrator's job; with several instances, all stage-1 roots go in).
    let refs: Vec<&[Goldilocks]> = stage1.iter().map(|c| c.as_slice()).collect();
    let stage1_root = commit_matrix(&refs, &ir.params).root();
    let challenges = derive_global_challenges(&ir, &[stage1_root]);
    let alpha = challenges[0];

    let (stage2, gsum_result) = stage2_trace(&stage1, alpha);

    Setup { ir, witness: vec![stage1, stage2], consts, challenges, airgroup_values: vec![gsum_result] }
}

#[test]
fn two_stage_logup_roundtrip() {
    let s = build(4);

    // A matching multiset must balance the bus: gsum_result = 0.
    assert!(s.airgroup_values[0].is_zero(), "valid lookup must have zero bus balance");

    check_constraints_on_trace(&s.ir, &s.witness, &s.consts, &[], &s.challenges, &[], &s.airgroup_values)
        .expect("constraints hold row-by-row");

    let proof = prove_air(&s.ir, &s.witness, &s.consts, &[], &s.challenges, &[], &s.airgroup_values).expect("prove");
    verify_air(&s.ir, &proof, &[], None, Some(&s.challenges)).expect("verify with enforced challenges");
}

#[test]
fn unbalanced_bus_shows_in_airgroup_value() {
    // Wrong multiplicities: gsum still satisfies its defining constraints
    // row-by-row (the proof is about internal consistency), but the bus
    // balance carried in the airgroup value is nonzero — which the global
    // (cross-instance) constraint check must catch.
    let mut s = build(4);
    let n_rows = 1usize << s.ir.n_bits;
    s.witness[0][2] = vec![Goldilocks::ONE; n_rows]; // mul := 1 everywhere
    let (stage2, gsum_result) = stage2_trace(&s.witness[0], s.challenges[0]);
    s.witness[1] = stage2;
    s.airgroup_values = vec![gsum_result];

    assert!(!s.airgroup_values[0].is_zero(), "unbalanced bus must have nonzero balance");
    let proof = prove_air(&s.ir, &s.witness, &s.consts, &[], &s.challenges, &[], &s.airgroup_values).expect("prove");
    verify_air(&s.ir, &proof, &[], None, Some(&s.challenges)).expect("per-instance proof still verifies");
}

#[test]
fn corrupted_stage2_rejected() {
    let mut s = build(4);
    s.witness[1][1][5] += Goldilocks::ONE; // corrupt one gsum coordinate
    let proof =
        prove_air(&s.ir, &s.witness, &s.consts, &[], &s.challenges, &[], &s.airgroup_values).expect("prove runs");
    assert!(verify_air(&s.ir, &proof, &[], None, Some(&s.challenges)).is_err());
}

#[test]
fn wrong_challenges_rejected() {
    let s = build(4);
    let proof = prove_air(&s.ir, &s.witness, &s.consts, &[], &s.challenges, &[], &s.airgroup_values).expect("prove");

    // Set-level verification with different globally-derived challenges must fail.
    let mut bad = s.challenges.clone();
    bad[0] += Ext::one();
    assert!(verify_air(&s.ir, &proof, &[], None, Some(&bad)).is_err());
}

#[test]
fn tampered_airgroup_value_rejected() {
    let s = build(4);
    let mut proof =
        prove_air(&s.ir, &s.witness, &s.consts, &[], &s.challenges, &[], &s.airgroup_values).expect("prove");
    // Claiming a different bus balance must break the LastRow corner check
    // (and the transcript binding).
    proof.airgroup_values[0] += ext_from(1);
    assert!(verify_air(&s.ir, &proof, &[], None, Some(&s.challenges)).is_err());
}
