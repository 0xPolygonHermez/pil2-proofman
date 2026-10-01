//! The Fibonacci's witness library (pilfflonk/docs/README.md#witness), loaded as a dynamic library:
//! its witness is the generator's (`pilfflonk/tests/data/fibonacci.rs`) byte for byte, and proves
//! and verifies; a STARK witness library is not taken for a pilfflonk one, nor the other way round;
//! and `src/pil_helpers` is what pil-helpers writes for the fixture's BN254 pilout.
//!
//! The library is this crate's, `libpilfflonk_fibonacci.so`, which Cargo builds for its tests, and the
//! STARK one `examples/fibonacci-square`'s, a dev-dependency built for the same reason.
//!
//! Pilouts are not versioned: the `#[ignore]` tests compile the fixture with the compiler `PIL2C_EXEC`
//! names, which must honour `prime`. The proof's needs Node.js, for the JS verifier:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pilfflonk-fibonacci -p proofman-starks-lib-c \
//!     --features proofman-starks-lib-c/cpu-only -- --include-ignored
//! ```

#[path = "../../../../data/fibonacci.rs"]
mod fibonacci;
#[path = "../../../../data/witness_libraries.rs"]
mod witness_libraries;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman_cli::commands::pil_helpers::PilHelpersCmd;
use proofman_pilfflonk::witness_library::INIT_SYMBOL;
use proofman_pilfflonk::{
    compute_witness, js_verifier, load_witness_library, prove, AirShape, FrBytes, JsonFile, PilfflonkError,
    PilfflonkGlobalInfo, ProveOptions, ProvingKey, Publics, WitnessShape, BN254_R,
};
use witness_libraries::built_library;

/// The fixture's size, and the inputs of pil-fflonk's `all` example
/// (pilfflonk/docs/README.md#fixtures).
const N_BITS: u32 = 8;
const INPUTS: [u64; 2] = [1, 2];

/// `pil-fflonk/runtime/public.json`, the publics pil-fflonk's prover gave for these inputs.
const PIL_FFLONK_PUBLICS: [&str; 3] =
    ["1", "2", "590308608561184158373097535019708483037277117989374906445627411437315467687"];

const SEED: [u8; 32] = [0x5e; 32];

/// A fresh directory for the test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir =
            Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_fibonacci_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../..").canonicalize().unwrap()
}

/// The library of this crate.
fn fibonacci_library() -> PathBuf {
    built_library("pilfflonk_fibonacci")
}

/// The fixture's shape, written out: one AIR of 2^8 rows and two stage-1 columns, three publics.
fn fibonacci_shape() -> WitnessShape {
    let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: N_BITS as u64, n_cols: 2, n_air_values: 0 };
    WitnessShape::new(vec![air], 3, 0).unwrap()
}

/// The public inputs file of `in1` and `in2`, as `--public-inputs` names it
/// (pilfflonk/docs/README.md#witness).
fn public_inputs(dir: &TestDir, in1: &str, in2: &str) -> PathBuf {
    let path = dir.file("inputs.json");
    fs::write(&path, format!(r#"{{"in1": "{in1}", "in2": "{in2}"}}"#)).unwrap();
    path
}

#[test]
fn the_librarys_witness_is_the_generators() {
    let dir = TestDir::new("witness");
    let mut library = load_witness_library(&fibonacci_library(), 0).unwrap();
    for inputs in [INPUTS, [3, 5], [0, 0], [u64::MAX, 7]] {
        let path = public_inputs(&dir, &inputs[0].to_string(), &inputs[1].to_string());
        let witness = compute_witness(&mut *library, &fibonacci_shape(), Some(&path)).unwrap();
        let generated = fibonacci::witness(N_BITS, inputs);
        assert_eq!(witness, generated, "{inputs:?}");
        assert_eq!(witness.instances[0].stage1.trace_bytes(), generated.instances[0].stage1.trace_bytes());
    }
    let witness = compute_witness(&mut *library, &fibonacci_shape(), Some(&public_inputs(&dir, "1", "2"))).unwrap();
    let publics: Vec<String> = witness.publics.iter().map(FrBytes::to_decimal).collect();
    assert_eq!(publics, PIL_FFLONK_PUBLICS);

    // With no public inputs, both are 0, as a STARK library's `load_from_json` gives.
    let witness = compute_witness(&mut *library, &fibonacci_shape(), None).unwrap();
    assert_eq!(witness, fibonacci::witness(N_BITS, [0, 0]));
}

/// The library computes in `Fr`: `r − 1` is `−1`, and `(−1)² + (−1)² = 2`.
#[test]
fn the_library_computes_modulo_r() {
    let dir = TestDir::new("modulo_r");
    let r_minus_1 = num_bigint::BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap() - 1u32;
    let path = public_inputs(&dir, &r_minus_1.to_string(), &r_minus_1.to_string());
    let mut library = load_witness_library(&fibonacci_library(), 0).unwrap();
    let witness = compute_witness(&mut *library, &fibonacci_shape(), Some(&path)).unwrap();
    let stage1 = &witness.instances[0].stage1;
    assert_eq!((stage1.get(1, 0), stage1.get(1, 1)), (Some(FrBytes::from_u64(2)), stage1.get(0, 0)));
    // l1[2] = l2[1]² + l1[1]² = (r − 1)² + 2² = 5.
    assert_eq!(stage1.get(2, 0), Some(FrBytes::from_u64(5)));
}

#[test]
fn the_library_refuses_what_it_cannot_compute() {
    let dir = TestDir::new("refusals");
    let mut library = load_witness_library(&fibonacci_library(), 0).unwrap();
    // Public inputs that are not Fr values: a JSON number, a value not below r.
    for text in [r#"{"in1": 1}"#, &format!(r#"{{"in1": "{BN254_R}"}}"#)] {
        let path = dir.file("bad.json");
        fs::write(&path, text).unwrap();
        let err = compute_witness(&mut *library, &fibonacci_shape(), Some(&path)).unwrap_err();
        assert!(matches!(err, PilfflonkError::InFile { .. }), "{text}: {err}");
    }
    // A key whose AIR is not the Fibonacci's.
    let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: N_BITS as u64 + 1, n_cols: 2, n_air_values: 0 };
    let err = compute_witness(&mut *library, &WitnessShape::new(vec![air], 3, 0).unwrap(), None).unwrap_err();
    assert!(err.to_string().contains("the Fibonacci has 256 rows and 2 columns"), "{err}");
    // A key of more publics than the program's: the host's check.
    let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: N_BITS as u64, n_cols: 2, n_air_values: 0 };
    let err = compute_witness(&mut *library, &WitnessShape::new(vec![air], 4, 0).unwrap(), None).unwrap_err();
    assert!(err.to_string().contains("3 publics, and nPublics is 4"), "{err}");
}

/// The two backends' libraries export different entry points: neither loader takes the other's.
#[test]
fn the_loaders_do_not_take_each_others_libraries() {
    let stark = built_library("fibonacci_square");
    match load_witness_library(&stark, 0) {
        Err(PilfflonkError::WitnessLibrary { path, reason }) => {
            assert_eq!(path, stark);
            assert!(reason.contains("STARK witness library"), "{reason}");
        }
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("a STARK witness library was loaded as a pilfflonk one"),
    }

    // The STARK's loader looks up `init_library` (`ProofMan::execute`), which this one lacks.
    let library = unsafe { libloading::Library::new(fibonacci_library()) }.unwrap();
    assert!(unsafe { library.get::<*const ()>(b"init_library") }.is_err());
    assert!(unsafe { library.get::<*const ()>(INIT_SYMBOL.as_bytes()) }.is_ok());
}

// ---------------------------------------------------------------------------------------------
// On the compiled fixture (needs PIL2C_EXEC)
// ---------------------------------------------------------------------------------------------

/// Compiles the fixture over BN254 with `PIL2C_EXEC` to `dir/fibonacci.pilout`, as `src/lib.rs`
/// says: the pilout's name, `Fibonacci`, is the stem of its file.
fn compile(dir: &TestDir) -> PathBuf {
    let compiler = std::env::var("PIL2C_EXEC")
        .expect("PIL2C_EXEC must name a pil2com that honours `prime` (e.g. <pil2-compiler>/src/pil.js)");
    let pilout = dir.file("fibonacci.pilout");
    let out = Command::new(compiler)
        .current_dir(repo_root())
        .arg("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil")
        .args(["-I", "pil2-components/lib/std/pil", "-P", "pilfflonk/tests/fixtures/fibonacci/bn254.json", "-o"])
        .arg(&pilout)
        .output()
        .expect("PIL2C_EXEC runs");
    assert!(out.status.success(), "pil2com: {}", String::from_utf8_lossy(&out.stderr));
    pilout
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_pil_helpers_are_those_pil_helpers_writes() {
    let dir = TestDir::new("pil_helpers");
    let pilout = compile(&dir);
    let out = dir.file("src");
    PilHelpersCmd { pilout, path: out.clone(), overide: false, verbose: 0 }.run().unwrap();
    let versioned = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pil_helpers");
    for file in ["mod.rs", "traces.rs"] {
        let generated = fs::read_to_string(out.join("pil_helpers").join(file)).unwrap();
        let expected = fs::read_to_string(versioned.join(file)).unwrap();
        assert!(generated == expected, "src/pil_helpers/{file} is not what pil-helpers writes:\n{generated}");
    }
}

/// The library's witness proves, the proof is the one of the generator's witness with the same
/// blinding, and the JS verifier accepts it and rejects it with another `out`.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_librarys_witness_proves_and_verifies() {
    let dir = TestDir::new("prove");
    let ptau = dir.file("fixed_tau.ptau");
    // More powers than the Fibonacci's largest `degree` (779, cli/tests/pilfflonk_prove.rs).
    write_fixed_tau_ptau(&ptau, 1024, &test_tau()).unwrap();
    let opts = SetupPilfflonkOptions {
        airout_path: compile(&dir),
        build_dir: dir.file("build"),
        powers_of_tau: ptau,
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: false,
        solidity: false,
    };
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let pk = ProvingKey::load(&proving_key).unwrap();
    assert_eq!(pk.witness_shape().unwrap(), fibonacci_shape());

    let mut library = load_witness_library(&fibonacci_library(), 0).unwrap();
    let inputs = public_inputs(&dir, "1", "2");
    let witness = compute_witness(&mut *library, &pk.witness_shape().unwrap(), Some(&inputs)).unwrap();
    let options = ProveOptions { insecure_blinding_seed: Some(SEED), ..ProveOptions::default() };
    let output = prove(&pk, &witness, &options).unwrap();
    let generated = prove(&pk, &fibonacci::witness(N_BITS, INPUTS), &options).unwrap();
    assert_eq!(output.proof, generated.proof);
    let publics: Vec<String> = output.publics.0.iter().map(FrBytes::to_decimal).collect();
    assert_eq!(publics, PIL_FFLONK_PUBLICS);

    let proof_dir = dir.file("proof");
    output.write(&proof_dir).unwrap();
    let vkey = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap().vkey_path(&proving_key);
    let (publics, proof) = (proof_dir.join("publics.json"), proof_dir.join("proof.json"));
    assert!(js_verifier::verify(&vkey, &publics, &proof).unwrap(), "the verifier rejects the library's proof");

    let mut other = output.publics.0.clone();
    other[2] = FrBytes::from_u64(1);
    let other_publics = dir.file("other_publics.json");
    Publics(other).write(&other_publics).unwrap();
    assert!(!js_verifier::verify(&vkey, &other_publics, &proof).unwrap(), "the verifier accepts another out");
}
