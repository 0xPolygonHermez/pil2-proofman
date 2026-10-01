//! The final SNARK wrap's family, PoseidonBN254 (`plonk2pil/setups/poseidon_bn254`), against
//! pilfflonk, on circuits circom compiles for BN254 with the custom gate `PoseidonT(5)`:
//!
//! 1. **The gate.** `fixtures/bn254/poseidon_chain.circom`, three permutations in a chain, on random
//!    inputs (a fixed seed). `pilfflonk check` holds on the trace, whose bands are circom's own `in`,
//!    `im` and `out`; and with any one intermediate state changed, the round constraints fail, on
//!    the two rows that read it and nowhere else. The circuit cannot catch that: the custom gate has
//!    no r1cs constraint, so `snarkjs wtns check` accepts a wrong intermediate (M45), and only the
//!    AIR constrains every round.
//! 2. **End to end.** `fixtures/bn254/wrap.circom`, the gate among multiplications, additions and
//!    copies: plonk2pil, pil2com over BN254, the pilfflonk setup with plonk2pil's fixed columns, the
//!    trace from circom's witness and the `.exec`; `pilfflonk check` holds and fails with a cell
//!    changed, and the JS verifier accepts a proof of it and refuses it with another public.
//!
//! The trace is the witness through the `.exec` by [`naive_trace`], the naive reference, until the
//! wrap's witness library (M51) replaces it.
//!
//! They need what the witness needs (`common`), `PIL2C_EXEC` (a pil2com that honours `prime`) and
//! the JS verifier (`PILFFLONK_JS`, or `pilfflonk/js`); without the first two they say why and
//! pass. They hold a lock around the C++ core, as pilfflonk's tests do: run them with
//! `--test-threads 2`.

mod common;

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, MutexGuard, PoisonError};

use num_bigint::BigUint;
use pil2_stark_recurser::plonk2pil::field::modulus;
use pil2_stark_recurser::plonk2pil::r1cs::types::{read_r1cs_header, PlonkOptions};
use pil2_stark_recurser::plonk2pil::setups::poseidon_bn254::constants::ROUNDS;
use pil2_stark_recurser::plonk2pil::setups::poseidon_bn254::wrap::BAND_ROWS;
use pil2_stark_recurser::plonk2pil::{plonk2pil, PlonkResult};
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk_with_external_fixed, ExternalFixedColumn, SetupPilfflonkOptions};
use proofman_common::exec_format::ExecFile;
use proofman_common::hash_family::BN254_WRAP_FAMILY;
use proofman_fields::{Bn254, Field};
use proofman_pilfflonk::{
    check, js_verifier, prove, AirInstanceRef, CheckOptions, CheckReport, FrBytes, InstanceWitness, JsonFile,
    PilfflonkGlobalInfo, ProveOptions, ProvingKey, Publics, Stage1Witness, Witness,
};

use common::{compile_and_witness, manifest, missing_prerequisite, read_wtns, repo_root, Scratch};

/// The blinding seed of the proofs (pilfflonk/docs/protocol.md#blinding).
const SEED: [u8; 32] = [0x49; 32];

/// The seed of the random inputs.
const INPUT_SEED: u64 = 0x4d49_5044_4f53;

/// Held for the whole of each test, which calls the C++ core's OpenMP code (pilfflonk's
/// `cpp_core`, pilfflonk/docs/README.md#tests).
fn cpp_core() -> MutexGuard<'static, ()> {
    static CPP_CORE: Mutex<()> = Mutex::new(());
    CPP_CORE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What the tests need and is missing, if anything.
fn missing() -> Option<String> {
    missing_prerequisite().or_else(|| match std::env::var("PIL2C_EXEC") {
        Ok(compiler) if !compiler.is_empty() => None,
        _ => Some("PIL2C_EXEC does not name a pil2com".into()),
    })
}

/// Field elements from a fixed seed (splitmix64, four words reduced mod r), in decimal.
struct Inputs(u64);

impl Inputs {
    fn next_word(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn element(&mut self) -> String {
        let bytes: Vec<u8> = (0..4).flat_map(|_| self.next_word().to_le_bytes()).collect();
        (BigUint::from_bytes_le(&bytes) % modulus::<Bn254>()).to_str_radix(10)
    }

    fn elements(&mut self, n: usize) -> Vec<String> {
        (0..n).map(|_| self.element()).collect()
    }
}

/// The JSON of a list of decimal elements.
fn json_list(values: &[String]) -> String {
    format!("[{}]", values.iter().map(|v| format!("\"{v}\"")).collect::<Vec<_>>().join(", "))
}

/// A circuit set up for pilfflonk: its witness, plonk2pil's result and the proving key.
struct Wrap {
    scratch: Scratch,
    /// circom's witness: wire 0 is the constant one, the publics follow.
    witness: Vec<Bn254>,
    n_publics: usize,
    res: PlonkResult<Bn254>,
    proving_key: PathBuf,
}

impl Wrap {
    /// Compiles `circuit` and computes its witness of `input` (JSON), runs plonk2pil's wrap on it,
    /// compiles the PIL over BN254 and sets it up with the fixed columns plonk2pil computed.
    fn set_up(name: &str, circuit: &str, input: &str) -> Self {
        let scratch = Scratch::new(name);
        let input_path = scratch.file("input.json");
        fs::write(&input_path, input).unwrap();
        let (r1cs, wtns) =
            compile_and_witness(&scratch.0, &manifest().join("tests/fixtures/bn254").join(circuit), &input_path);
        let header = read_r1cs_header(&r1cs).unwrap();
        let n_publics = (header.n_outputs + header.n_pub_inputs) as usize;

        let options = PlonkOptions { hash_id: BN254_WRAP_FAMILY.into(), ..Default::default() };
        let res: PlonkResult<Bn254> = plonk2pil(&r1cs, "wrap", &options).unwrap();

        let pilout = compile_pil(&scratch, &res.pil_str);
        let ptau = scratch.file("fixed_tau.ptau");
        // More powers than the layout's largest degree, 12·N + 11 for L1 (M45).
        write_fixed_tau_ptau(&ptau, 16 << res.n_bits, &test_tau()).unwrap();
        let setup = SetupPilfflonkOptions {
            airout_path: pilout,
            build_dir: scratch.file("build"),
            powers_of_tau: ptau,
            max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
            extra_muls: DEFAULT_EXTRA_MULS,
            max_q_degree: DEFAULT_MAX_Q_DEGREE,
            no_packing: false,
            solidity: false,
        };
        let external = res
            .fixed_pols
            .iter()
            .map(|p| ExternalFixedColumn {
                name: p.name.clone(),
                index: p.index,
                values: p.values.iter().map(|&v| FrBytes::from(v)).collect(),
            })
            .collect();
        run_setup_pilfflonk_with_external_fixed(&setup, external).unwrap_or_else(|e| panic!("{e:#}"));
        let proving_key = setup.build_dir.join(PROVING_KEY_DIR);
        Self { witness: read_wtns(&wtns), n_publics, res, proving_key, scratch }
    }

    fn n(&self) -> usize {
        1 << self.res.n_bits
    }

    /// The stage-1 trace, by column, and the publics.
    fn trace(&self) -> (Vec<Vec<Bn254>>, Vec<Bn254>) {
        naive_trace(&self.res.exec, &self.witness, self.n(), self.n_publics)
    }

    fn check(&self, pk: &ProvingKey, trace: &[Vec<Bn254>], publics: &[Bn254]) -> CheckReport {
        let witness = pilfflonk_witness(self.n(), trace, publics);
        check(pk, &witness, &CheckOptions::default()).unwrap_or_else(|e| panic!("{e}"))
    }
}

/// Compiles `pil` with `PIL2C_EXEC` over BN254 (`-P`), with plonk2pil's PIL and the std on the
/// include path, as the pilfflonk fixtures are compiled.
fn compile_pil(scratch: &Scratch, pil: &str) -> PathBuf {
    let (source, config, pilout) = (scratch.file("wrap.pil"), scratch.file("bn254.json"), scratch.file("wrap.pilout"));
    fs::write(&source, pil).unwrap();
    fs::write(&config, format!("{{\"prime\": \"{}\"}}", modulus::<Bn254>())).unwrap();
    let includes = [manifest().join("plonk2pil/pil"), repo_root().join("pil2-components/lib/std/pil")];
    let includes: Vec<String> = includes.iter().map(|p| p.display().to_string()).collect();
    let out = Command::new(std::env::var("PIL2C_EXEC").unwrap())
        .arg(&source)
        .arg("-I")
        .arg(includes.join(","))
        .arg("-P")
        .arg(&config)
        .arg("-o")
        .arg(&pilout)
        .output()
        .expect("PIL2C_EXEC runs");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "pil2com: {log}");
    pilout
}

/// THE NAIVE REFERENCE of the wrap's witness, until the wrap's witness library (M51) replaces it:
/// circom's witness through the `.exec`, as the STARK prover's `getCommitedPols` reads its own. The
/// additions extend the witness in order, each wire `n_vars + i`; a cell of the map's live extent
/// is the witness at its signal, but for the sentinel 0, which is 0; every other cell is 0. The
/// publics are wires `1..=n_publics`.
fn naive_trace(exec: &[u64], witness: &[Bn254], n: usize, n_publics: usize) -> (Vec<Vec<Bn254>>, Vec<Bn254>) {
    let file = ExecFile::<Bn254>::from_words(exec).unwrap_or_else(|e| panic!("{e}"));
    let mut w = witness.to_vec();
    for a in &file.additions {
        let v = a.coeffs[0] * w[a.wires[0] as usize] + a.coeffs[1] * w[a.wires[1] as usize];
        w.push(v);
    }
    let (rows, cols) = (file.layout.map_rows(), file.layout.map_cols());
    let mut trace = vec![vec![Bn254::ZERO; n]; 9];
    for (col, column) in trace.iter_mut().enumerate().take(cols) {
        for (row, cell) in column.iter_mut().enumerate().take(rows) {
            let signal = file.map_entry(row, col) as usize;
            if signal != 0 {
                *cell = w[signal];
            }
        }
    }
    (trace, w[1..=n_publics].to_vec())
}

/// The witness of the AIR's one instance: its stage-1 columns `a[0..8]` and the publics.
fn pilfflonk_witness(n: usize, trace: &[Vec<Bn254>], publics: &[Bn254]) -> Witness {
    let columns: Vec<Vec<FrBytes>> = trace.iter().map(|c| c.iter().map(|&v| FrBytes::from(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("nine columns of n rows");
    Witness {
        instances: vec![InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 }],
        publics: publics.iter().map(|&v| FrBytes::from(v)).collect(),
        proof_values: vec![],
    }
}

/// The failed rows of each failing constraint, with its PIL line.
fn failures(report: &CheckReport) -> Vec<(String, Vec<u64>)> {
    report.failures().map(|c| (c.line.clone(), c.failed_rows.iter().map(|r| r.row).collect())).collect()
}

#[test]
fn the_rounds_hold_on_circoms_intermediates_and_fail_on_any_other() {
    if let Some(why) = missing() {
        eprintln!("skipping the PoseidonT gate test: {why}");
        return;
    }
    let _cpp = cpp_core();
    const N_HASHES: usize = 3;
    let mut inputs = Inputs(INPUT_SEED);
    let initial_state = inputs.element();
    let blocks: Vec<String> = (0..N_HASHES).map(|_| json_list(&inputs.elements(4))).collect();
    let input = format!("{{\"initialState\": \"{initial_state}\", \"in\": [{}]}}", blocks.join(", "));
    let wrap = Wrap::set_up("poseidon_bn254_gate", "poseidon_chain.circom", &input);
    assert!(wrap.res.n_used >= N_HASHES * BAND_ROWS);

    let pk = ProvingKey::load(&wrap.proving_key).unwrap();
    let (trace, publics) = wrap.trace();
    let report = wrap.check(&pk, &trace, &publics);
    assert!(report.holds(), "the trace of circom's witness: {:?}", failures(&report));

    // An intermediate state of each band, after a full round, a partial one and the last but one:
    // band row k holds im[k - 1], the state after round k - 1.
    for (band, k, lane) in [(0, 1, 0), (1, 30, 3), (2, ROUNDS - 1, 4), (0, 4, 2)] {
        let row = BAND_ROWS * band + k;
        let mut wrong = trace.clone();
        wrong[lane][row] += Bn254::ONE;
        let report = wrap.check(&pk, &wrong, &publics);
        assert!(!report.holds(), "im[{}][{lane}] of band {band} changed, and the check holds", k - 1);
        let failures = failures(&report);
        for (line, _) in &failures {
            assert!(line.contains("poseidon_bn254.pil"), "{line} fails, not a round: {failures:?}");
        }
        let rows: BTreeSet<u64> = failures.iter().flat_map(|(_, rows)| rows.iter().copied()).collect();
        // Round k - 1 computes it, round k reads it.
        assert_eq!(rows, BTreeSet::from([row as u64 - 1, row as u64]), "{failures:?}");
    }
}

#[test]
fn a_small_wrap_checks_proves_and_verifies() {
    if let Some(why) = missing() {
        eprintln!("skipping the PoseidonBN254 wrap end to end: {why}");
        return;
    }
    let _cpp = cpp_core();
    let mut inputs = Inputs(INPUT_SEED + 1);
    let [a, b] = [inputs.element(), inputs.element()];
    let input = format!("{{\"a\": \"{a}\", \"b\": \"{b}\", \"x\": {}}}", json_list(&inputs.elements(4)));
    let wrap = Wrap::set_up("poseidon_bn254_wrap", "wrap.circom", &input);
    assert_eq!(wrap.n_publics, 1);
    assert!(wrap.res.n_used > BAND_ROWS + 1, "PLONK rows past the band, to check gates 0 and 1 too");

    let pk = ProvingKey::load(&wrap.proving_key).unwrap();
    let (trace, publics) = wrap.trace();
    let report = wrap.check(&pk, &trace, &publics);
    assert!(report.holds(), "the trace of circom's witness: {:?}", failures(&report));

    // The left wire of gate 0 on the first PLONK row: its gate no longer holds.
    let mut wrong = trace.clone();
    wrong[0][BAND_ROWS] += Bn254::ONE;
    let report = wrap.check(&pk, &wrong, &publics);
    let rows: BTreeSet<u64> = failures(&report).iter().flat_map(|(_, rows)| rows.clone()).collect();
    assert!(rows.contains(&(BAND_ROWS as u64)), "a changed PLONK cell: {:?}", failures(&report));

    let options = ProveOptions { insecure_blinding_seed: Some(SEED), q_part_bits: None };
    let out = prove(&pk, &pilfflonk_witness(wrap.n(), &trace, &publics), &options).unwrap_or_else(|e| panic!("{e}"));
    let (proof, publics_path) = (wrap.scratch.file("proof.json"), wrap.scratch.file("publics.json"));
    out.proof_json().unwrap().write(&proof).unwrap();
    out.publics.write(&publics_path).unwrap();
    assert_eq!(out.publics.0, vec![FrBytes::from(publics[0])], "the public is circom's output");
    let vkey = PilfflonkGlobalInfo::from_proving_key(&wrap.proving_key).unwrap().vkey_path(&wrap.proving_key);
    assert!(js_verifier::verify(&vkey, &publics_path, &proof).unwrap(), "the JS verifier accepts the proof");

    let other = wrap.scratch.file("other_publics.json");
    Publics(vec![FrBytes::from(publics[0] + Bn254::ONE)]).write(&other).unwrap();
    assert!(!js_verifier::verify(&vkey, &other, &proof).unwrap(), "the JS verifier refuses another public");
}
