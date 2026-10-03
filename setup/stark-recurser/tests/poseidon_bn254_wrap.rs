//! The final SNARK wrap's family, PoseidonBN254 (`plonk2pil/setups/poseidon_bn254`), against
//! pilfflonk, on circuits circom compiles for BN254 with the custom gates `PoseidonT(5)` and
//! `Num2Bytes(nBits)`, set up with the family's knobs (`wrap::MAX_CONSTRAINT_DEGREE` in the PIL,
//! `wrap::EXTRA_MULS`):
//!
//! 1. **The gate.** `fixtures/bn254/poseidon_chain.circom`, three permutations in a chain, on random
//!    inputs (a fixed seed). `pilfflonk check` holds on the trace, whose bands are circom's own `in`,
//!    `im` and `out`; and with any one intermediate state changed, the round constraints fail, on
//!    the two rows that read it and nowhere else. The circuit cannot catch that: the custom gate has
//!    no r1cs constraint, so `snarkjs wtns check` accepts a wrong intermediate, and only the AIR
//!    constrains every round.
//! 2. **End to end.** `fixtures/bn254/wrap.circom`, the gate among multiplications, additions and
//!    copies: plonk2pil, pil2com over BN254, the pilfflonk setup with plonk2pil's fixed columns, the
//!    trace from circom's witness and the `.exec`; `pilfflonk check` holds and fails with a cell
//!    changed, and the JS verifier accepts a proof of it and refuses it with another public.
//! 3. **The range checks.** `fixtures/bn254/num2bytes.circom`, uses of `Num2Bytes` of whole and
//!    partial chunks: `pilfflonk check` holds on the trace, and fails on each attack on a range
//!    check, a chunk outside the table, the original's free cell (`in + 2^64` with `a[5] = 1`), a
//!    wrong multiplicity and a wrong chunk; the JS verifier accepts a proof. The AIR has the
//!    connection on the std's product bus and the lookup on its sum bus.
//! 4. **More publics than a row.** `fixtures/bn254/publics.circom`, 14 publics on two public rows,
//!    each with a selector of its own, opened at its row alone: the AIR sets up at the family's
//!    knobs, `pilfflonk check` holds and fails with any public changed, and the JS verifier accepts
//!    a proof and refuses it with another public.
//!
//! The trace is the witness through the `.exec`, and the range checks' multiplicity counted, by
//! [`naive_trace`], the naive reference of the wrap's witness library (`wrap-witness`).
//!
//! They need what the witness needs (`common`), `PIL2C_EXEC` (a pil2com that has `--field`) and
//! the JS verifier (`PILFFLONK_JS`, or `pilfflonk/js`); without the first two they say why and
//! pass. They hold a lock around the C++ core, as pilfflonk's tests do: run them with
//! `--test-threads 2`.

mod common;

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use num_bigint::BigUint;
use pil2_stark_recurser::plonk2pil::field::modulus;
use pil2_stark_recurser::plonk2pil::r1cs::types::{read_r1cs_header, PlonkOptions};
use pil2_stark_recurser::plonk2pil::setups::poseidon_bn254::constants::ROUNDS;
use pil2_stark_recurser::plonk2pil::setups::poseidon_bn254::wrap::{BAND_ROWS, RANGE_MUL_COLUMN};
use pil2_stark_recurser::plonk2pil::{plonk2pil, PlonkResult};
use proofman_common::exec_format::{ExecFile, RANGE_CHECK_BAND_KIND, RANGE_CHECK_CHUNK_BITS, RANGE_CHECK_CHUNK_COLS};
use proofman_common::hash_family::BN254_WRAP_FAMILY;
use proofman_fields::{Bn254, Field, PrimeField, QuotientMap};
use proofman_pilfflonk::{
    check, js_verifier, prove, AirInstanceRef, CheckOptions, CheckReport, FrBytes, InstanceWitness, JsonFile,
    PilfflonkGlobalInfo, PolType, ProveOptions, ProvingKey, Publics, Stage1Witness, Witness,
};

use common::wrap_key::{set_up_key, Ptau};
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
    /// and sets up its key with `ptau` ([`set_up_key`]).
    fn set_up(name: &str, circuit: &str, input: &str, ptau: Ptau) -> Self {
        let scratch = Scratch::new(name);
        let input_path = scratch.file("input.json");
        fs::write(&input_path, input).unwrap();
        let (r1cs, wtns) =
            compile_and_witness(&scratch.0, &manifest().join("tests/fixtures/bn254").join(circuit), &input_path);
        let header = read_r1cs_header(&r1cs).unwrap();
        let n_publics = (header.n_outputs + header.n_pub_inputs) as usize;

        let options = PlonkOptions { hash_id: BN254_WRAP_FAMILY.into(), ..Default::default() };
        let res: PlonkResult<Bn254> = plonk2pil(&r1cs, "wrap", &options).unwrap();
        let proving_key = set_up_key(&repo_root(), &scratch.0, &res, ptau);
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

/// THE NAIVE REFERENCE of the wrap's witness: circom's witness through the `.exec`, as the STARK
/// prover's `getCommitedPols` reads its own, and the range checks' multiplicity. The additions
/// extend the witness in order, each wire `n_vars + i`; a cell of the map's live extent is the
/// witness at its signal, but for the sentinel 0, which is 0; every other cell is 0. With
/// range-check bands, the column the band section's aux word names is [`range_multiplicity`] of
/// their rows. The publics are wires `1..=n_publics`.
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
    if !file.bands.is_empty() {
        assert!(file.bands.iter().all(|b| b.kind == RANGE_CHECK_BAND_KIND), "the wrap's bands are range checks");
        assert_eq!(file.band_aux, trace.len() as u64, "RANGE_MUL is the column after the wires");
        let multiplicity = range_multiplicity(&trace, &range_check_rows(&file));
        trace.push(multiplicity);
    }
    (trace, w[1..=n_publics].to_vec())
}

/// The range-check rows of an exec: the rows of its bands.
fn range_check_rows(file: &ExecFile<Bn254>) -> Vec<usize> {
    file.bands.iter().map(|b| b.row as usize).collect()
}

/// `RANGE_MUL` of `trace`: at row `v < 2^16`, how many of the chunk cells of `rows` hold `v`; every
/// cell of `a[1..=5]` counts, as the AIR looks them all up, and a cell outside the table nowhere.
fn range_multiplicity(trace: &[Vec<Bn254>], rows: &[usize]) -> Vec<Bn254> {
    let mut counts = vec![0u64; trace[0].len()];
    for &row in rows {
        for col in RANGE_CHECK_CHUNK_COLS {
            if let Ok(v) = u16::try_from(&trace[col][row].as_canonical_biguint()) {
                counts[v as usize] += 1;
            }
        }
    }
    counts.into_iter().map(Bn254::from_int).collect()
}

/// The witness of the AIR's one instance: its stage-1 columns, `a[0..8]` and `RANGE_MUL` if it has
/// range checks, and the publics.
fn pilfflonk_witness(n: usize, trace: &[Vec<Bn254>], publics: &[Bn254]) -> Witness {
    let columns: Vec<Vec<FrBytes>> = trace.iter().map(|c| c.iter().map(|&v| FrBytes::from(v)).collect()).collect();
    let stage1 = Stage1Witness::from_columns(n, &columns, vec![]).expect("columns of n rows");
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
    let wrap = Wrap::set_up("poseidon_bn254_gate", "poseidon_chain.circom", &input, Ptau::TauOne);
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
    let wrap = Wrap::set_up("poseidon_bn254_wrap", "wrap.circom", &input, Ptau::FixedTau);
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

    assert_eq!(
        js_verifies(&wrap, &pk, &trace, &publics, 0),
        [true, false],
        "the JS verifier: the proof, another public"
    );
}

/// Proves `trace` and the publics, writes the proof, and asks the JS verifier of the key, with the
/// publics and with `publics[changed] + 1`: whether it accepts each.
fn js_verifies(wrap: &Wrap, pk: &ProvingKey, trace: &[Vec<Bn254>], publics: &[Bn254], changed: usize) -> [bool; 2] {
    let options = ProveOptions { insecure_blinding_seed: Some(SEED), q_part_bits: None };
    let out = prove(pk, &pilfflonk_witness(wrap.n(), trace, publics), &options).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out.publics.0, publics.iter().map(|&p| FrBytes::from(p)).collect::<Vec<_>>());
    let (proof, publics_path) = (wrap.scratch.file("proof.json"), wrap.scratch.file("publics.json"));
    out.proof_json().unwrap().write(&proof).unwrap();
    out.publics.write(&publics_path).unwrap();
    let vkey = PilfflonkGlobalInfo::from_proving_key(&wrap.proving_key).unwrap().vkey_path(&wrap.proving_key);

    let mut other = publics.to_vec();
    other[changed] += Bn254::ONE;
    let other_path = wrap.scratch.file("other_publics.json");
    Publics(other.into_iter().map(FrBytes::from).collect()).write(&other_path).unwrap();
    [&publics_path, &other_path].map(|path| js_verifier::verify(&vkey, path, &proof).unwrap())
}

/// `n_bits` random bits, `mask` or'ed in.
fn bits(inputs: &mut Inputs, n_bits: u32, mask: u128) -> u128 {
    let word = u128::from(inputs.next_word()) | (u128::from(inputs.next_word()) << 64);
    (word & ((1 << n_bits) - 1)) | mask
}

#[test]
fn range_checks_hold_and_every_attack_on_them_fails() {
    if let Some(why) = missing() {
        eprintln!("skipping the Num2Bytes test: {why}");
        return;
    }
    let _cpp = cpp_core();
    let mut inputs = Inputs(INPUT_SEED + 2);
    // x's chunks are all nonzero, so that one can lend 2^16 to the one below it.
    let x = bits(&mut inputs, 64, 0x0001_0001_0001_0001);
    let [y, a, b, c] = [70, 64, 80, 3].map(|n| bits(&mut inputs, n, 0));
    let input = format!(
        "{{\"x\": \"{x}\", \"y\": \"{y}\", \"a\": \"{a}\", \"b\": \"{b}\", \"c\": \"{c}\", \"s\": {}}}",
        json_list(&inputs.elements(4))
    );
    let wrap = Wrap::set_up("poseidon_bn254_num2bytes", "num2bytes.circom", &input, Ptau::FixedTau);
    assert_eq!((wrap.n_publics, wrap.res.n_bits), (1, RANGE_CHECK_CHUNK_BITS as usize), "the table's 2^16 rows");

    let file = ExecFile::<Bn254>::from_words(&wrap.res.exec).unwrap();
    let rows = range_check_rows(&file);
    let mut payloads: Vec<u64> = file.bands.iter().map(|b| b.payload).collect();
    payloads.sort_unstable();
    assert_eq!(payloads, [1, 1, 2, 4, 4, 4, 5, 5], "the chunks of the 8 uses, of 3 to 80 bits");

    let pk = ProvingKey::load(&wrap.proving_key).unwrap();
    let info = pk.air(AirInstanceRef { airgroup_id: 0, air_id: 0 }).unwrap();
    let column_9 = info.cm_pols_map.iter().find(|p| p.stage == 1 && p.stage_id == RANGE_MUL_COLUMN as u64).unwrap();
    assert!(column_9.name.ends_with("RANGE_MUL"), "stage-1 column {RANGE_MUL_COLUMN} is {}", column_9.name);
    let (trace, publics) = wrap.trace();
    let report = wrap.check(&pk, &trace, &publics);
    assert!(report.holds(), "the trace of circom's witness: {:?}", failures(&report));

    // The isolated uses: x's row, of 4 chunks, and y's, of 5.
    let row_of = |value: u128| rows.iter().copied().find(|&row| trace[0][row] == Bn254::from_int(value)).unwrap();
    let (x_row, y_row) = (row_of(x), row_of(y));
    let chunk_size = Bn254::from_int(1u64 << RANGE_CHECK_CHUNK_BITS);
    // The trace changed by `change`, with RANGE_MUL counted again: what a prover would commit.
    let attack = |change: &dyn Fn(&mut Vec<Vec<Bn254>>)| {
        let mut wrong = trace.clone();
        change(&mut wrong);
        wrong[RANGE_MUL_COLUMN] = range_multiplicity(&wrong, &rows);
        failures(&wrap.check(&pk, &wrong, &publics))
    };
    let in_the_sum_bus = |failures: &[(String, Vec<u64>)]| {
        !failures.is_empty() && failures.iter().all(|(line, _)| line.contains("std_sum.pil"))
    };
    let recomposition_at = |failures: &[(String, Vec<u64>)], row: usize| matches!(failures, [(line, rows)] if line.contains("num2bytes.pil") && *rows == [row as u64]);

    // A chunk outside the table, with the sum unchanged: 2^16 more in chunk 0, 1 less in chunk 1.
    let outside = attack(&|t| {
        t[1][x_row] += chunk_size;
        t[2][x_row] -= Bn254::ONE;
    });
    assert!(in_the_sum_bus(&outside), "a chunk of 2^16 or more fails the lookup alone: {outside:?}");

    // The original's free cell: a 64-bit check of in + 2^64, with the fifth chunk cell 1.
    let free_cell = attack(&|t| {
        t[0][x_row] += Bn254::from_int(1u128 << 64);
        t[5][x_row] = Bn254::ONE;
    });
    assert!(recomposition_at(&free_cell, x_row), "a cell past the chunks weighs 0: {free_cell:?}");

    // A wrong chunk, in range.
    let wrong_chunk = attack(&|t| t[3][y_row] += Bn254::ONE);
    assert!(recomposition_at(&wrong_chunk, y_row), "a wrong chunk: {wrong_chunk:?}");

    // A wrong multiplicity, the chunks honest.
    let mut wrong = trace.clone();
    wrong[RANGE_MUL_COLUMN][7] += Bn254::ONE;
    let wrong_multiplicity = failures(&wrap.check(&pk, &wrong, &publics));
    assert!(in_the_sum_bus(&wrong_multiplicity), "a wrong multiplicity: {wrong_multiplicity:?}");

    assert_eq!(
        js_verifies(&wrap, &pk, &trace, &publics, 0),
        [true, false],
        "the JS verifier: the proof, another public"
    );
}

#[test]
fn more_publics_than_a_row_check_prove_and_verify() {
    if let Some(why) = missing() {
        eprintln!("skipping the publics test: {why}");
        return;
    }
    let _cpp = cpp_core();
    let mut inputs = Inputs(INPUT_SEED + 3);
    let [s, k] = [inputs.element(), inputs.element()];
    let input = format!("{{\"s\": \"{s}\", \"k\": \"{k}\", \"x\": {}}}", json_list(&inputs.elements(4)));
    let wrap = Wrap::set_up("poseidon_bn254_publics", "publics.circom", &input, Ptau::FixedTau);
    assert_eq!(wrap.n_publics, 14);

    let pk = ProvingKey::load(&wrap.proving_key).unwrap();
    let info = pk.air(AirInstanceRef { airgroup_id: 0, air_id: 0 }).unwrap();
    // A selector for each public row, each opened at its row alone.
    let publics_rows: Vec<(u64, i64)> = info
        .ev_map
        .iter()
        .filter(|e| {
            e.pol_type == PolType::Const && info.pol(PolType::Const, e.id).unwrap().name.ends_with("PUBLICS_ROW")
        })
        .map(|e| (e.id, e.prime))
        .collect();
    assert_eq!(publics_rows.len(), 2, "{publics_rows:?}");
    assert!(publics_rows.iter().all(|&(_, prime)| prime == 0), "PUBLICS_ROW at another row: {publics_rows:?}");
    let (trace, publics) = wrap.trace();
    let report = wrap.check(&pk, &trace, &publics);
    assert!(report.holds(), "the trace of circom's witness: {:?}", failures(&report));

    // Public i is on public row i / 9, the last two rows.
    let first_public_row = wrap.res.n_used - 2;
    for i in [0, 8, 9, 13] {
        let mut other = publics.clone();
        other[i] += Bn254::ONE;
        let failures = failures(&wrap.check(&pk, &trace, &other));
        let row = (first_public_row + i / 9) as u64;
        assert!(
            !failures.is_empty() && failures.iter().all(|(line, rows)| line.contains("wrap.pil") && *rows == [row]),
            "public {i} changed: {failures:?}"
        );
    }

    assert_eq!(
        js_verifies(&wrap, &pk, &trace, &publics, 10),
        [true, false],
        "the JS verifier: the proof, another public"
    );
}
