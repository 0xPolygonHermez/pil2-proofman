//! End-to-end test against a real proving key produced by `proofman-setup`:
//! loads `FibonacciML.mlinfo.bin` (the compiled AirIr) and `FibonacciML.const`,
//! builds the Fibonacci witness, and runs prove + verify.
//!
//! Skips silently when the proving key has not been generated (run
//! `proofman-setup compile-pil` + `setup` for examples/fibonacci-multilinear
//! first — see that example's README).

use std::path::PathBuf;

use fields::{Field, Goldilocks, PrimeField64};
use proofman_multilinear::{check_constraints_on_trace, prove_air, verify_air, AirIr, Ext};

fn air_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../examples/fibonacci-multilinear/build/provingKey/fibonacci/FibonacciML/airs/FibonacciML/air")
}

/// Headerless little-endian u64 `.const` file, row-major.
fn load_const_cols(path: &std::path::Path, n_cols: usize, n_rows: usize) -> Vec<Vec<Goldilocks>> {
    let bytes = std::fs::read(path).expect("read .const");
    assert_eq!(bytes.len(), n_cols * n_rows * 8, ".const size mismatch");
    let mut cols = vec![vec![Goldilocks::ZERO; n_rows]; n_cols];
    for row in 0..n_rows {
        for (col, col_vals) in cols.iter_mut().enumerate() {
            let off = (row * n_cols + col) * 8;
            let v = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            col_vals[row] = Goldilocks::from_u64(v);
        }
    }
    cols
}

#[test]
fn prove_verify_from_setup_artifact() {
    let dir = air_dir();
    let mlinfo = dir.join("FibonacciML.mlinfo.bin");
    if !mlinfo.exists() {
        eprintln!("skipping: {} not found (generate the proving key first)", mlinfo.display());
        return;
    }

    let ir = AirIr::load(&mlinfo).expect("load AirIr");
    assert_eq!(ir.name, "FibonacciML");
    assert_eq!(ir.cols_per_stage, vec![2], "expected two stage-1 witness columns (a, b)");
    assert_eq!(ir.n_const_cols, 1, "expected one fixed column (L1)");
    assert_eq!(ir.n_publics, 3);
    assert!(ir.opening_offsets.contains(&0) && ir.opening_offsets.contains(&1));

    let n_rows = 1usize << ir.n_bits;
    let consts = load_const_cols(&dir.join("FibonacciML.const"), ir.n_const_cols as usize, n_rows);
    // Sanity: L1 = [1, 0, 0, …]
    assert_eq!(consts[0][0], Goldilocks::ONE);
    assert_eq!(consts[0][1], Goldilocks::ZERO);

    // Build the witness: a' = b, b' = a + b.
    let in1 = Goldilocks::ONE;
    let in2 = Goldilocks::TWO;
    let mut a = vec![Goldilocks::ZERO; n_rows];
    let mut b = vec![Goldilocks::ZERO; n_rows];
    a[0] = in1;
    b[0] = in2;
    for i in 1..n_rows {
        a[i] = b[i - 1];
        b[i] = a[i - 1] + b[i - 1];
    }
    let out = b[n_rows - 1];
    let witness = vec![vec![a, b]];
    let publics = vec![in1, in2, out];

    // Single-stage AIR: no challenges are derived; the vectors still need the
    // full global shape (protocol-level symbols exist in every pilout).
    let challenges = vec![Ext::zero(); ir.challenge_stages.len()];
    let air_values = vec![Ext::zero(); ir.airvalue_stages.len()];
    let airgroup_values = vec![Ext::zero(); ir.airgroupvalue_stages.len()];

    // The compiled IR must agree row-by-row with the trace.
    check_constraints_on_trace(&ir, &witness, &consts, &publics, &challenges, &air_values, &airgroup_values)
        .expect("constraints hold on the trace");

    let proof = prove_air(&ir, &witness, &consts, &publics, &challenges, &air_values, &airgroup_values).expect("prove");
    verify_air(&ir, &proof, &publics, None, None).expect("verify");

    // Wrong publics must fail.
    let bad = vec![in1, in2, out + Goldilocks::ONE];
    assert!(verify_air(&ir, &proof, &bad, None, None).is_err());
}
