//! The wrap's witness on a small circuit over BN254, before plonk2pil has a BN254 gate family: the
//! circuit of plonk2pil's BN254 test, `setup/stark-recurser/tests/fixtures/bn254/arith.circom`, and
//! its `input.json` as the zkin.
//!
//! The circuit's files are laid out as setup-snark lays out `provingKeySnark/final/`,
//! `final/final.{so,dat,exec}` (`tests/common`):
//! - it is compiled with the committed circom (`setup/circom`), for BN254;
//! - its witness calculator is built as setup-snark builds `final.so`: `WitnessTracker`, with the
//!   Makefile of `setup/final_snark_circom/`;
//! - its exec is written by plonk2pil's writer: the additions of `r1cs2plonk`, and a map of a row
//!   per public, whose column 0 is the public's wire, then a row per PLONK gate, its wires
//!   `(l, r, o)` in columns 0 to 2. The AIR has a fourth column, which the map leaves empty.
//!
//! The reference is snarkjs's witness of the same input, from the circom wasm, with the additions
//! applied naively. The tests need what setup-snark needs to build `final.so` (make, g++, nasm and
//! nlohmann/json), and Node.js with the snarkjs of `setup/pil2-stark/node_modules` (`npm install`
//! there): without node or snarkjs, they say why and pass, as plonk2pil's BN254 test does.
//!
//! `the_witness_proves_on_an_air_of_the_gates` proves the witness, loaded with `-w`'s
//! `load_witness_library`, on a PIL2 AIR of the gates (`tests/fixtures/plonk.pil`, no copy
//! constraints), and needs `PIL2C_EXEC`, a compiler that honours `prime`, and the JS verifier's
//! packages in `pilfflonk/js` (pilfflonk/docs/README.md#tests):
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pilfflonk-wrap-witness -p proofman-starks-lib-c \
//!     --features proofman-starks-lib-c/cpu-only -- --include-ignored
//! ```

#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use pil2_stark_recurser::plonk2pil::field::PlonkField;
use pil2_stark_recurser::plonk2pil::r1cs::to_plonk::{r1cs2plonk, PlonkAddition, PlonkConstraint};
use pil2_stark_recurser::plonk2pil::r1cs::types::{read_r1cs_from_bytes, GateBand, GateBandKind};
use pil2_stark_recurser::plonk2pil::write_exec_file;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk_with_external_fixed, ExternalFixedColumn, SetupPilfflonkOptions};
use pilfflonk_wrap_witness::{witness_from_circom, WrapArtifacts, WrapWitness, WrapWitnessError};
use proofman_fields::{Bn254, Field, Goldilocks};
use proofman_pilfflonk::{
    check, compute_witness, js_verifier, load_witness_library, prove, AirShape, CheckOptions, FrBytes, JsonFile,
    PilfflonkError, PilfflonkGlobalInfo, ProveOptions, ProvingKey, Publics, Witness, WitnessShape,
};
use common::{build_final, fixture, missing_prerequisite, repo_root, run, snarkjs_witness};
use witness_libraries::built_library;

/// The circuit's publics: its 4 outputs and its 2 public inputs, wires 1 to 6.
const N_PUBLICS: usize = 6;

/// The AIR's stage-1 columns: `l`, `r`, `o`, and one the map leaves empty.
const N_COLS: usize = 4;

/// The blinding seed of the proof (pilfflonk/docs/protocol.md#blinding).
const SEED: [u8; 32] = [0x51; 32];

/// The zkin: the circuit's `input.json`.
fn zkin() -> PathBuf {
    fixture("input.json")
}

/// The circuit, its files and its reference, built once for the tests of this binary.
struct Circuit {
    /// `final/` holds its files, as `provingKeySnark/final/`.
    dir: PathBuf,
    gates: Vec<PlonkConstraint<Bn254>>,
    additions: Vec<PlonkAddition<Bn254>>,
    /// The map, as plonk2pil's packers make one: a column at a time, `N` rows each.
    s_map: Vec<Vec<u32>>,
    n_bits: u64,
    /// snarkjs's witness of the zkin.
    reference: Vec<Bn254>,
}

impl Circuit {
    /// `final/final`, the stem of its files.
    fn stem(&self) -> PathBuf {
        self.dir.join("final/final")
    }

    fn artifacts(&self) -> WrapArtifacts {
        WrapArtifacts::with_stem(&self.stem())
    }

    fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn n_rows(&self) -> usize {
        1 << self.n_bits
    }

    /// The shape of the key of an AIR of `n_bits` and `n_cols`, and `n_publics`.
    fn shape_of(&self, n_bits: u64, n_cols: usize, n_publics: usize) -> WitnessShape {
        let air = AirShape { airgroup_id: 0, air_id: 0, n_bits, n_cols, n_air_values: 0 };
        WitnessShape::new(vec![air], n_publics, 0).unwrap()
    }

    /// The shape of the wrap's AIR of this circuit.
    fn shape(&self) -> WitnessShape {
        self.shape_of(self.n_bits, N_COLS, N_PUBLICS)
    }

    /// Writes the exec of these additions and bands, over `F`, at `path`.
    fn write_exec<F: PlonkField>(&self, path: &Path, additions: &[PlonkAddition<F>], bands: &[GateBand]) {
        let words = write_exec_file(additions, &self.s_map, bands, 0);
        fs::write(path, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
}

/// The circuit, or `None` (and why, on stderr) without what its reference needs.
fn circuit() -> Option<&'static Circuit> {
    static CIRCUIT: OnceLock<Option<Circuit>> = OnceLock::new();
    CIRCUIT
        .get_or_init(|| match missing_prerequisite() {
            Some(why) => {
                eprintln!("skipping the wrap's witness tests: {why}");
                None
            }
            None => Some(build_circuit()),
        })
        .as_ref()
}

fn build_circuit() -> Circuit {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("wrap_witness_arith");
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    let r1cs_bytes = build_final(&dir, &fixture("arith.circom"), &[]);

    let r1cs = read_r1cs_from_bytes::<Bn254>(&r1cs_bytes).unwrap();
    assert_eq!(r1cs.header.n_outputs as usize + r1cs.header.n_pub_inputs as usize, N_PUBLICS);
    let (gates, additions) = r1cs2plonk(&r1cs);
    assert!(!additions.is_empty(), "the circuit's wide sum must introduce additions");

    let n_rows = (N_PUBLICS + gates.len()).next_power_of_two();
    let mut s_map = vec![vec![0u32; n_rows]; 3];
    for (row, wire) in s_map[0].iter_mut().zip(1..=N_PUBLICS as u32) {
        *row = wire;
    }
    for (k, gate) in gates.iter().enumerate() {
        for (column, &wire) in s_map.iter_mut().zip(&gate.wires) {
            column[N_PUBLICS + k] = wire;
        }
    }

    let reference = snarkjs_witness(&dir, "arith", &zkin());
    assert_eq!(reference.len(), r1cs.header.n_vars as usize);

    let circuit = Circuit { dir, gates, additions, s_map, n_bits: n_rows.trailing_zeros() as u64, reference };
    circuit.write_exec(&circuit.artifacts().exec, &circuit.additions, &[]);
    circuit
}

/// The naive reference: snarkjs's witness, the additions appended one after another, and the cell
/// of each map entry, 0 for none; the columns, and the publics after wire 0.
fn reference(circuit: &Circuit) -> (Vec<Vec<Bn254>>, Vec<Bn254>) {
    let mut wires = circuit.reference.clone();
    for addition in &circuit.additions {
        let [l, r] = addition.wires.map(|wire| wires[wire as usize]);
        wires.push(addition.coeffs[0] * l + addition.coeffs[1] * r);
    }
    let mut columns: Vec<Vec<Bn254>> = circuit
        .s_map
        .iter()
        .map(|column| column.iter().map(|&wire| if wire == 0 { Bn254::ZERO } else { wires[wire as usize] }).collect())
        .collect();
    columns.resize(N_COLS, vec![Bn254::ZERO; circuit.n_rows()]);
    (columns, circuit.reference[1..=N_PUBLICS].to_vec())
}

fn value(witness: &Witness, row: usize, col: usize) -> Bn254 {
    witness.instances[0].stage1.get(row, col).unwrap().into()
}

/// The gates that fail on the rows of `witness`: `qM·l·r + qL·l + qR·r + qO·o + qC` of gate `k` on
/// row `N_PUBLICS + k`.
fn failing_gates(circuit: &Circuit, witness: &Witness) -> Vec<usize> {
    let fails = |(k, gate): &(usize, &PlonkConstraint<Bn254>)| {
        let [l, r, o] = [0, 1, 2].map(|col| value(witness, N_PUBLICS + k, col));
        let [q_m, q_l, q_r, q_o, q_c] = gate.coeffs;
        !(q_m * l * r + q_l * l + q_r * r + q_o * o + q_c).is_zero()
    };
    circuit.gates.iter().enumerate().filter(fails).map(|(k, _)| k).collect()
}

#[test]
fn the_columns_are_the_naive_reference() {
    let Some(circuit) = circuit() else { return };
    let wrap = WrapWitness::load(&circuit.artifacts()).unwrap();
    // final.so computes snarkjs's witness, value for value.
    assert_eq!(wrap.circom_witness(&zkin()).unwrap(), circuit.reference);

    let witness = wrap.witness(&circuit.shape(), &zkin()).unwrap();
    circuit.shape().check(&witness).unwrap();
    let (columns, publics) = reference(circuit);
    let stage1 = &witness.instances[0].stage1;
    for (col, column) in columns.iter().enumerate() {
        let expected: Vec<FrBytes> = column.iter().copied().map(FrBytes::from).collect();
        assert_eq!(stage1.column(col).unwrap(), expected, "column {col}");
    }
    assert_eq!(witness.publics, publics.into_iter().map(FrBytes::from).collect::<Vec<_>>());
    assert!(witness.proof_values.is_empty() && stage1.air_values().is_empty());
    // The rows past the gates, and the column the map leaves empty, are zero.
    let used = N_PUBLICS + circuit.gates.len();
    assert!(used < circuit.n_rows(), "the trace has rows past the map, which must be zero");
    assert!((used..circuit.n_rows()).all(|row| (0..N_COLS).all(|col| value(&witness, row, col).is_zero())));
    assert!(stage1.column(3).unwrap().iter().all(|v| *v == FrBytes::ZERO));

    // The same from the circuit's witness at hand.
    let again = witness_from_circom(wrap.exec(), circuit.reference.clone(), &circuit.shape()).unwrap();
    assert_eq!(again, witness);
}

#[test]
fn every_plonk_gate_holds_on_the_rows() {
    let Some(circuit) = circuit() else { return };
    let wrap = WrapWitness::load(&circuit.artifacts()).unwrap();
    let witness = wrap.witness(&circuit.shape(), &zkin()).unwrap();
    assert_eq!(failing_gates(circuit, &witness), Vec::<usize>::new());
    for i in 0..N_PUBLICS {
        assert_eq!(FrBytes::from(value(&witness, i, 0)), witness.publics[i], "the row of public {i}");
    }
    // Both kinds of gate, and a change of one cell breaks its gate: the check checks something.
    assert!(
        circuit.gates.iter().any(|g| g.coeffs[0].is_zero()) && circuit.gates.iter().any(|g| !g.coeffs[0].is_zero())
    );
    let (k, _) = circuit.gates.iter().enumerate().find(|(_, g)| !g.coeffs[3].is_zero()).unwrap();
    let mut wrong = witness.clone();
    let o = value(&wrong, N_PUBLICS + k, 2) + Bn254::ONE;
    wrong.instances[0].stage1.set(N_PUBLICS + k, 2, o.into()).unwrap();
    assert_eq!(failing_gates(circuit, &wrong), vec![k]);
}

/// The `-w` path: the cdylib, loaded as `pilfflonk prove -w` loads a witness library, computes the
/// same witness from the JSON of `-i`, whose relative paths are its directory's.
#[test]
fn the_dynamic_library_computes_the_same_witness() {
    let Some(circuit) = circuit() else { return };
    let inputs = circuit.file("inputs.json");
    let json = serde_json::json!({
        "zkin": zkin(),
        "witnessCalculator": format!("final/final.{}", if cfg!(target_os = "macos") { "dylib" } else { "so" }),
        "dat": "final/final.dat",
        "exec": "final/final.exec",
    });
    fs::write(&inputs, json.to_string()).unwrap();

    let mut library = load_witness_library(&built_library("pilfflonk_wrap_witness"), 0).unwrap();
    let witness = compute_witness(&mut *library, &circuit.shape(), Some(&inputs)).unwrap();
    let in_process = WrapWitness::load(&circuit.artifacts()).unwrap().witness(&circuit.shape(), &zkin()).unwrap();
    assert_eq!(witness, in_process);

    // Without -i it cannot know its files; with a JSON missing a file, or naming one that is not
    // there, it says which, as pilfflonk's errors.
    let err = compute_witness(&mut *library, &circuit.shape(), None).unwrap_err();
    assert!(err.to_string().contains("needs -i (--public-inputs)"), "{err}");
    fs::write(&inputs, r#"{"zkin": "z.json", "witnessCalculator": "final/final.so", "dat": "final/final.dat"}"#)
        .unwrap();
    let err = compute_witness(&mut *library, &circuit.shape(), Some(&inputs)).unwrap_err();
    assert!(matches!(&err, PilfflonkError::InFile { path, .. } if *path == inputs), "{err}");
    assert!(err.to_string().contains("missing field `exec`"), "{err}");
    fs::write(&inputs, json.to_string().replace("final.exec", "none.exec")).unwrap();
    let err = compute_witness(&mut *library, &circuit.shape(), Some(&inputs)).unwrap_err();
    assert!(err.to_string().contains("the exec ") && err.to_string().contains("none.exec is not a file"), "{err}");
    fs::write(&inputs, json.to_string().replace("input.json", "none.json")).unwrap();
    let err = compute_witness(&mut *library, &circuit.shape(), Some(&inputs)).unwrap_err();
    assert!(matches!(&err, PilfflonkError::Io { path, .. } if path.ends_with("none.json")), "{err}");
}

/// A file missing, an exec of another field or with a gate band of the STARK's, and a key that is
/// not the wrap's.
#[test]
fn files_and_keys_that_do_not_fit_are_refused() {
    let Some(circuit) = circuit() else { return };
    let artifacts = circuit.artifacts();
    for (what, artifacts) in [
        ("witness calculator", WrapArtifacts { witness_calculator: circuit.file("none.so"), ..artifacts.clone() }),
        ("witness calculator's dat", WrapArtifacts { dat: circuit.file("none.dat"), ..artifacts.clone() }),
        ("exec", WrapArtifacts { exec: circuit.file("none.exec"), ..artifacts.clone() }),
    ] {
        match WrapWitness::load(&artifacts) {
            Err(WrapWitnessError::Missing { what: missing, .. }) => assert_eq!(missing, what),
            other => panic!("{what}: {other:?}"),
        }
    }

    // A Goldilocks exec, and a BN254 one with a gate band of the STARK's.
    let goldilocks = circuit.file("goldilocks.exec");
    circuit.write_exec::<Goldilocks>(&goldilocks, &[], &[]);
    let err = WrapWitness::load(&WrapArtifacts { exec: goldilocks, ..artifacts.clone() }).unwrap_err();
    assert!(matches!(err, WrapWitnessError::Exec(_)), "{err}");
    assert!(err.to_string().contains("not the 4-word ones of BN254"), "{err}");
    let banded = circuit.file("banded.exec");
    let band = GateBand { row: 0, kind: GateBandKind::Poseidon1CompressorSponge, payload: 0 };
    circuit.write_exec(&banded, &circuit.additions, &[band]);
    let err = WrapWitness::load(&WrapArtifacts { exec: banded, ..artifacts.clone() }).unwrap_err();
    assert!(err.to_string().contains("has a gate band of kind 1 at row 0, which only the STARK's"), "{err}");

    // Keys whose AIR the map does not fit, with publics past the witness, or not the wrap's.
    let wrap = WrapWitness::load(&artifacts).unwrap();
    let n_wires = circuit.reference.len();
    let air =
        |n_air_values| AirShape { airgroup_id: 0, air_id: 0, n_bits: circuit.n_bits, n_cols: N_COLS, n_air_values };
    let other = AirShape { air_id: 1, ..air(0) };
    for (why, shape) in [
        ("larger than the trace's", circuit.shape_of(circuit.n_bits, 2, N_PUBLICS)),
        ("larger than the trace's", circuit.shape_of(circuit.n_bits - 1, N_COLS, N_PUBLICS)),
        ("too few for wire 0 and", circuit.shape_of(circuit.n_bits, N_COLS, n_wires)),
        ("the key has 2 AIRs", WitnessShape::new(vec![air(0), other], N_PUBLICS, 0).unwrap()),
        ("has 1 stage-1 air values", WitnessShape::new(vec![air(1)], N_PUBLICS, 0).unwrap()),
        ("the key has 1 stage-1 proof values", WitnessShape::new(vec![air(0)], N_PUBLICS, 1).unwrap()),
    ] {
        let err = wrap.witness(&shape, &zkin()).unwrap_err().to_string();
        assert!(err.contains(why), "expected \"{why}\", got: {err}");
    }
}

/// A zkin that is not a JSON object is refused before the calculator sees it, and one the
/// calculator fails on is refused with the calculator's failure.
#[test]
fn zkins_the_circuit_cannot_take_are_refused() {
    let Some(circuit) = circuit() else { return };
    let wrap = WrapWitness::load(&circuit.artifacts()).unwrap();
    let input: serde_json::Value = serde_json::from_slice(&fs::read(zkin()).unwrap()).unwrap();
    let mut no_c = input.clone();
    no_c.as_object_mut().unwrap().remove("c");
    let mut not_a_number = input.clone();
    not_a_number["a"] = "0xZZ".into();
    let mut too_many = input.clone();
    too_many["c"].as_array_mut().unwrap().push("1".into());

    for (name, text, calculator) in [
        ("not_json", "{\"a\": ".to_string(), false),
        ("array", "[1, 2]".to_string(), false),
        ("no_c", no_c.to_string(), true),
        ("not_a_number", not_a_number.to_string(), true),
        ("too_many", too_many.to_string(), true),
    ] {
        let path = circuit.file(&format!("zkin_{name}.json"));
        fs::write(&path, text).unwrap();
        match (wrap.witness(&circuit.shape(), &path), calculator) {
            (Err(WrapWitnessError::Json { path: at, .. }), false) => assert_eq!(at, path),
            (Err(WrapWitnessError::Calculator { zkin: Some(at), .. }), true) => assert_eq!(at, path),
            (other, _) => panic!("{name}: {other:?}"),
        }
    }
    let none = circuit.file("none.json");
    assert!(matches!(wrap.witness(&circuit.shape(), &none), Err(WrapWitnessError::Io { path, .. }) if path == none));
}

/// The witness, from the cdylib as `-w` loads it, proves on a PIL2 AIR of the circuit's gates over
/// BN254 (`tests/fixtures/plonk.pil`): `pilfflonk check` passes on it, the JS verifier accepts the
/// proof, and rejects it with another public. The selectors are the AIR's external fixed columns,
/// as plonk2pil's are the wrap's.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_witness_proves_on_an_air_of_the_gates() {
    let circuit = circuit().expect("node and the snarkjs of setup/pil2-stark/node_modules");
    let compiler = std::env::var("PIL2C_EXEC")
        .expect("PIL2C_EXEC must name a pil2com that honours `prime` (e.g. <pil2-compiler>/src/pil.js)");
    let dir = circuit.file("prove");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let pilout = dir.join("plonk.pilout");
    run(
        Command::new(compiler)
            .arg(manifest.join("tests/fixtures/plonk.pil"))
            .arg("-P")
            .arg(repo_root().join("pilfflonk/tests/fixtures/fibonacci/bn254.json"))
            .arg("-o")
            .arg(&pilout),
        "pil2com",
    );

    // The selectors of each row's gate, zero on the rows of the publics and past the gates, and
    // P[i], 1 on the row of public i.
    let n_rows = circuit.n_rows();
    let mut external = Vec::new();
    for (index, name) in ["QM", "QL", "QR", "QO", "QC"].into_iter().enumerate() {
        let mut values = vec![FrBytes::ZERO; n_rows];
        for (k, gate) in circuit.gates.iter().enumerate() {
            values[N_PUBLICS + k] = gate.coeffs[index].into();
        }
        external.push(ExternalFixedColumn { name: format!("Plonk.{name}"), index: 0, values });
    }
    for i in 0..N_PUBLICS {
        let mut values = vec![FrBytes::ZERO; n_rows];
        values[i] = FrBytes::from_u64(1);
        external.push(ExternalFixedColumn { name: "Plonk.P".to_string(), index: i, values });
    }
    let ptau = dir.join("fixed_tau.ptau");
    write_fixed_tau_ptau(&ptau, 1024, &test_tau()).unwrap();
    let opts = SetupPilfflonkOptions {
        airout_path: pilout,
        build_dir: dir.join("build"),
        powers_of_tau: ptau,
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: false,
        solidity: false,
    };
    run_setup_pilfflonk_with_external_fixed(&opts, external).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let pk = ProvingKey::load(&proving_key).unwrap();
    assert_eq!(pk.witness_shape().unwrap(), circuit.shape());

    let inputs = dir.join("inputs.json");
    let artifacts = circuit.artifacts();
    let json = serde_json::json!({
        "zkin": zkin(),
        "witnessCalculator": artifacts.witness_calculator,
        "dat": artifacts.dat,
        "exec": artifacts.exec,
    });
    fs::write(&inputs, json.to_string()).unwrap();
    let mut library = load_witness_library(&built_library("pilfflonk_wrap_witness"), 0).unwrap();
    let witness = compute_witness(&mut *library, &pk.witness_shape().unwrap(), Some(&inputs)).unwrap();

    let report = check(&pk, &witness, &CheckOptions::default()).unwrap();
    assert!(report.holds(), "pilfflonk check fails on the witness: {:?}", report.failures().collect::<Vec<_>>());
    let mut wrong = witness.clone();
    let k = N_PUBLICS + circuit.gates.iter().position(|g| !g.coeffs[3].is_zero()).unwrap();
    let o = value(&wrong, k, 2) + Bn254::ONE;
    wrong.instances[0].stage1.set(k, 2, o.into()).unwrap();
    assert!(!check(&pk, &wrong, &CheckOptions::default()).unwrap().holds(), "a wrong cell passes pilfflonk check");

    let options = ProveOptions { insecure_blinding_seed: Some(SEED), ..ProveOptions::default() };
    let output = prove(&pk, &witness, &options).unwrap();
    assert_eq!(output.publics.0, witness.publics);
    let proof_dir = dir.join("proof");
    output.write(&proof_dir).unwrap();
    let vkey = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap().vkey_path(&proving_key);
    let (publics, proof) = (proof_dir.join("publics.json"), proof_dir.join("proof.json"));
    assert!(js_verifier::verify(&vkey, &publics, &proof).unwrap(), "the verifier rejects the wrap's proof");

    let mut other = output.publics.0.clone();
    other[0] = FrBytes::from_u64(1);
    let other_publics = dir.join("other_publics.json");
    Publics(other).write(&other_publics).unwrap();
    assert!(!js_verifier::verify(&vkey, &other_publics, &proof).unwrap(), "the verifier accepts another public");
}
