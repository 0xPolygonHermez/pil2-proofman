//! The wrap's witness of an AIR with range checks: plonk2pil's wrap of
//! `setup/stark-recurser/tests/fixtures/bn128/num2bytes.circom`, uses of circom's `Num2Bytes` of
//! whole and partial chunks among PLONK gates and a `PoseidonT(5)` use, with its files laid out as
//! setup-snark lays out `provingKeySnark/final/` (`tests/common`) and plonk2pil's exec.
//!
//! - `the_witness_is_the_naive_reference`: the witness of its input is snarkjs's witness through
//!   the exec, and its `RANGE_MUL` the naive count, at each row `v < 2^16` the chunk cells of the
//!   range-check rows that hold `v`.
//! - `pilfflonk_check_holds_on_the_witness` needs `PIL2C_EXEC`: the check holds on the witness,
//!   with the key of the wrap's AIR, and fails with a multiplicity changed. The key is set up with
//!   the ptau of `τ = 1`, which the check does not read.
//! - `the_witness_of_a_final_circuit_checks` is the check on a real final circuit, ignored by
//!   default: `PILFFLONK_WRAP_FINAL` names a directory with its `final.r1cs`, `final.so` and
//!   `final.dat` (setup-snark's, or circom's `--c` with `setup/final_snark_circom`'s Makefile), and
//!   `PILFFLONK_WRAP_ZKIN` a zkin of the recursivef it verifies:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js PILFFLONK_WRAP_FINAL=<dir> PILFFLONK_WRAP_ZKIN=<zkin.json> \
//!     cargo test -p pilfflonk-wrap-witness --features proofman-common/cpu-only --test range_checks \
//!     -- --ignored the_witness_of_a_final_circuit_checks
//! ```

mod common;

#[path = "../../setup/stark-recurser/tests/common/wrap_key.rs"]
mod wrap_key;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use pil2_stark_recurser::plonk2pil::r1cs::types::{read_r1cs_header, PlonkOptions};
use pil2_stark_recurser::plonk2pil::setups::poseidon_bn128::wrap::RANGE_MUL_COLUMN;
use pil2_stark_recurser::plonk2pil::{plonk2pil, PlonkResult};
use pilfflonk_wrap_witness::{witness_from_circom, WrapArtifacts, WrapWitness};
use proofman_common::exec_format::{ExecFile, RANGE_CHECK_CHUNK_COLS};
use proofman_common::hash_family::BN128_WRAP_FAMILY;
use proofman_fields::{Bn128, Field, PrimeField, QuotientMap};
use proofman_pilfflonk::{check, AirShape, CheckOptions, FrBytes, ProvingKey, Witness, WitnessShape};

use common::{build_final, circuits_bn128, fixture, missing_prerequisite, repo_root, snarkjs_witness};
use wrap_key::{set_up_key, Ptau};

/// The circuit's input: values of the bits of their Num2Bytes, the top ones included (`a` and `b`
/// are all ones), and the PoseidonT(5) use's.
const INPUT: &str = r#"{"x": "1234605616436508552", "y": "1180591620717411303423", "a": "18446744073709551615",
    "b": "1208925819614629174706175", "c": "5", "s": ["1", "2", "3", "4"]}"#;

/// The circuit, its wrap and its reference, built once for the tests of this binary.
struct Circuit {
    dir: PathBuf,
    res: PlonkResult<Bn128>,
    n_publics: usize,
    /// snarkjs's witness of the input.
    reference: Vec<Bn128>,
}

impl Circuit {
    fn artifacts(&self) -> WrapArtifacts {
        WrapArtifacts::with_stem(&self.dir.join("final/final"))
    }

    fn zkin(&self) -> PathBuf {
        self.dir.join("input.json")
    }

    fn exec(&self) -> ExecFile<Bn128> {
        ExecFile::from_words(&self.res.exec).unwrap()
    }

    /// The shape of the wrap's AIR: the wires and `RANGE_MUL`.
    fn shape(&self) -> WitnessShape {
        let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: self.res.n_bits as u64, n_cols: 10, n_air_values: 0 };
        WitnessShape::new(vec![air], self.n_publics, 0).unwrap()
    }
}

/// The circuit, or `None` (and why, on stderr) without what its reference needs.
fn circuit() -> Option<&'static Circuit> {
    static CIRCUIT: OnceLock<Option<Circuit>> = OnceLock::new();
    CIRCUIT
        .get_or_init(|| match missing_prerequisite() {
            Some(why) => {
                eprintln!("skipping the wrap's range-check witness tests: {why}");
                None
            }
            None => Some(build_circuit()),
        })
        .as_ref()
}

fn build_circuit() -> Circuit {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("wrap_witness_range_checks");
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    let r1cs = build_final(&dir, &fixture("num2bytes.circom"), &[circuits_bn128()]);
    let header = read_r1cs_header(&r1cs).unwrap();
    let n_publics = (header.n_outputs + header.n_pub_inputs) as usize;
    let mut circuit = Circuit { dir, res: wrap(&r1cs), n_publics, reference: Vec::new() };
    write_exec(&circuit.res, &circuit.artifacts().exec);
    fs::write(circuit.zkin(), INPUT).unwrap();
    circuit.reference = snarkjs_witness(&circuit.dir, "num2bytes", &circuit.zkin());
    circuit
}

/// plonk2pil's wrap of `r1cs`.
fn wrap(r1cs: &[u8]) -> PlonkResult<Bn128> {
    let options = PlonkOptions { hash_id: BN128_WRAP_FAMILY.into(), ..Default::default() };
    plonk2pil(r1cs, "wrap", &options).unwrap()
}

fn write_exec(res: &PlonkResult<Bn128>, path: &Path) {
    fs::write(path, res.exec.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

/// THE NAIVE REFERENCE: snarkjs's witness, the additions appended one after another, the cell of
/// each map entry, 0 for none; and the multiplicity, at row `v` how many chunk cells of the
/// range-check rows hold `v`, each looked up in a map from values to counts.
fn naive_reference(circuit: &Circuit) -> Vec<Vec<Bn128>> {
    let exec = circuit.exec();
    let mut wires = circuit.reference.clone();
    for addition in &exec.additions {
        let [l, r] = addition.wires.map(|wire| wires[wire as usize]);
        wires.push(addition.coeffs[0] * l + addition.coeffs[1] * r);
    }
    let n_rows = 1 << circuit.res.n_bits;
    let cell = |row: usize, col: usize| match exec.map_entry(row, col) {
        0 => Bn128::ZERO,
        wire => wires[wire as usize],
    };
    let mut columns: Vec<Vec<Bn128>> =
        (0..RANGE_MUL_COLUMN).map(|col| (0..n_rows).map(|row| cell(row, col)).collect()).collect();

    let mut counts: HashMap<u64, u64> = HashMap::new();
    for band in &exec.bands {
        for col in RANGE_CHECK_CHUNK_COLS {
            let value = columns[col][band.row as usize].as_canonical_biguint();
            *counts.entry(u64::try_from(&value).unwrap()).or_insert(0) += 1;
        }
    }
    columns.push((0..n_rows as u64).map(|v| Bn128::from_int(counts.get(&v).copied().unwrap_or(0))).collect());
    columns
}

fn columns(witness: &Witness) -> Vec<Vec<FrBytes>> {
    let stage1 = &witness.instances[0].stage1;
    (0..=RANGE_MUL_COLUMN).map(|col| stage1.column(col).unwrap()).collect()
}

#[test]
fn the_witness_is_the_naive_reference() {
    let Some(circuit) = circuit() else { return };
    let exec = circuit.exec();
    assert_eq!(exec.bands.len(), 8, "the fixture's 8 uses of Num2Bytes");
    assert_eq!(exec.band_aux, RANGE_MUL_COLUMN as u64);

    let wrap = WrapWitness::load(&circuit.artifacts()).unwrap();
    let witness = wrap.witness(&circuit.shape(), &circuit.zkin()).unwrap();
    circuit.shape().check(&witness).unwrap();
    let expected: Vec<Vec<FrBytes>> =
        naive_reference(circuit).into_iter().map(|c| c.into_iter().map(FrBytes::from).collect()).collect();
    assert_eq!(columns(&witness), expected);

    // Every chunk cell is counted once: 5 a row, the zeros past a row's chunks too, and the top
    // value of a chunk, 2^16 − 1, at its row.
    let multiplicity = &expected[RANGE_MUL_COLUMN];
    let total: u64 = multiplicity.iter().map(|m| u64::try_from(&Bn128::from(*m).as_canonical_biguint()).unwrap()).sum();
    assert_eq!(total, 5 * exec.bands.len() as u64);
    assert!(multiplicity[0] != FrBytes::ZERO && multiplicity[0xffff] != FrBytes::ZERO);

    // The same from the circuit's witness at hand.
    let again = witness_from_circom(wrap.exec(), circuit.reference.clone(), &circuit.shape()).unwrap();
    assert_eq!(again, witness);
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn pilfflonk_check_holds_on_the_witness() {
    let circuit = circuit().expect("node and the snarkjs of setup/pil2-stark/node_modules");
    let key_dir = circuit.dir.join("key");
    fs::create_dir_all(&key_dir).unwrap();
    let pk = ProvingKey::load(&set_up_key(&repo_root(), &key_dir, &circuit.res, Ptau::TauOne)).unwrap();
    assert_eq!(pk.witness_shape().unwrap(), circuit.shape());

    let witness = WrapWitness::load(&circuit.artifacts()).unwrap().witness(&circuit.shape(), &circuit.zkin()).unwrap();
    let report = check(&pk, &witness, &CheckOptions::default()).unwrap();
    assert!(report.holds(), "pilfflonk check fails on the witness: {:?}", report.failures().collect::<Vec<_>>());
    let mut wrong = witness.clone();
    let m = Bn128::from(wrong.instances[0].stage1.get(3, RANGE_MUL_COLUMN).unwrap()) + Bn128::ONE;
    wrong.instances[0].stage1.set(3, RANGE_MUL_COLUMN, m.into()).unwrap();
    assert!(!check(&pk, &wrong, &CheckOptions::default()).unwrap().holds(), "a wrong multiplicity passes");
}

#[test]
#[ignore = "needs PIL2C_EXEC, PILFFLONK_WRAP_FINAL and PILFFLONK_WRAP_ZKIN"]
fn the_witness_of_a_final_circuit_checks() {
    let var = |name: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} is not set")));
    let (final_dir, zkin) = (var("PILFFLONK_WRAP_FINAL"), var("PILFFLONK_WRAP_ZKIN"));
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("wrap_witness_final_circuit");
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(&dir).unwrap();

    let start = Instant::now();
    let res = wrap(&fs::read(final_dir.join("final.r1cs")).unwrap());
    let exec = ExecFile::<Bn128>::from_words(&res.exec).unwrap();
    eprintln!(
        "plonk2pil: n_bits {}, n_used {}, {} range-check rows ({:.1} s)",
        res.n_bits,
        res.n_used,
        exec.bands.len(),
        start.elapsed().as_secs_f64()
    );
    let artifacts = WrapArtifacts {
        witness_calculator: WrapArtifacts::with_stem(&final_dir.join("final")).witness_calculator,
        dat: final_dir.join("final.dat"),
        exec: dir.join("final.exec"),
    };
    write_exec(&res, &artifacts.exec);

    let start = Instant::now();
    let pk = ProvingKey::load(&set_up_key(&repo_root(), &dir, &res, Ptau::TauOne)).unwrap();
    eprintln!("setup: {:.1} s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let witness = WrapWitness::load(&artifacts).unwrap().witness(&pk.witness_shape().unwrap(), &zkin).unwrap();
    eprintln!("witness: {:.1} s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let report = check(&pk, &witness, &CheckOptions::default()).unwrap();
    eprintln!("check: {:.1} s", start.elapsed().as_secs_f64());
    assert!(report.holds(), "pilfflonk check fails on the witness: {:?}", report.failures().collect::<Vec<_>>());
}
