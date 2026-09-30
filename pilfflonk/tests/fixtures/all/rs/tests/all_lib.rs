//! The witness library of `all` (plan M38c, D4), loaded as a dynamic library: its witness is the
//! generator's (`pilfflonk/tests/data/all.rs`, plan M34) byte for byte, with pil-fflonk's publics for
//! pil-fflonk's inputs; its rows are the stage-1 columns of the keys of the fixture on either bus, in
//! their order; and `src/pil_helpers` is what pil-helpers writes for the fixture's BN254 pilouts.
//!
//! The library is this crate's, `libpilfflonk_all.so`, which Cargo builds for its tests. A test
//! cannot link it (a Rust `dylib` would bring a second `std`), so it includes the rows of
//! `src/pil_helpers`, which the library is built from, to look at their layout.
//!
//! Pilouts are not versioned: the `#[ignore]` tests compile the fixture with the compiler `PIL2C_EXEC`
//! names, which must honour `prime`:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pilfflonk-all -p proofman-starks-lib-c \
//!     --features proofman-starks-lib-c/cpu-only -- --include-ignored
//! ```

#[path = "../../../../data/all.rs"]
mod all;
// The generators of the state machines of `all`: its generator takes only their columns, and the CLI
// tests their witnesses, hence the `dead_code`.
#[allow(dead_code)]
#[path = "../../../../data/connection.rs"]
mod connection;
#[path = "../../../../data/fibonacci.rs"]
mod fibonacci;
#[allow(dead_code)]
#[path = "../../../../data/permutation.rs"]
mod permutation;
#[allow(unused_imports)]
#[path = "../src/pil_helpers/mod.rs"]
mod pil_helpers;
#[allow(dead_code)]
#[path = "../../../../data/plookup.rs"]
mod plookup;
#[path = "../../../../data/witness_libraries.rs"]
mod witness_libraries;

use std::fs;
use std::mem::{offset_of, size_of};
use std::path::{Path, PathBuf};
use std::process::Command;

use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman_cli::commands::pil_helpers::PilHelpersCmd;
use proofman_fields::Bn254;
use proofman_pilfflonk::{
    check, compute_witness, load_witness_library, AirInstanceRef, AirShape, CheckOptions, FrBytes, PilfflonkError,
    PilfflonkInfo, ProvingKey, WitnessShape, BN254_R,
};
use witness_libraries::built_library;

use pil_helpers::{AllTrace, AllTraceRow};

/// The fixture's size: `N = 2^8`, as pil-fflonk's `all_main.pil`.
const N_BITS: u64 = 8;

/// `pil-fflonk/runtime/public.json`, the publics pil-fflonk's prover gave for its inputs, `[1, 2]`.
const PIL_FFLONK_PUBLICS: [&str; 3] =
    ["1", "2", "590308608561184158373097535019708483037277117989374906445627411437315467687"];

/// A fresh directory for the test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_all_{name}_{}", std::process::id()));
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
fn all_library() -> PathBuf {
    built_library("pilfflonk_all")
}

/// The fixture's shape, written out: one AIR of 2^8 rows and sixteen stage-1 columns, three
/// publics.
fn all_shape() -> WitnessShape {
    let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: N_BITS, n_cols: 16, n_air_values: 0 };
    WitnessShape::new(vec![air], 3, 0).unwrap()
}

/// The public inputs file of `in1` and `in2`, as `--public-inputs` names it.
fn public_inputs(dir: &TestDir, in1: &str, in2: &str) -> PathBuf {
    let path = dir.file("inputs.json");
    fs::write(&path, format!(r#"{{"in1": "{in1}", "in2": "{in2}"}}"#)).unwrap();
    path
}

#[test]
fn the_librarys_witness_is_the_generators() {
    let dir = TestDir::new("witness");
    let mut library = load_witness_library(&all_library(), 0).unwrap();
    for inputs in [all::INPUTS, [3, 5], [0, 0], [u64::MAX, 7]] {
        let path = public_inputs(&dir, &inputs[0].to_string(), &inputs[1].to_string());
        let witness = compute_witness(&mut *library, &all_shape(), Some(&path)).unwrap();
        let generated = all::witness_of_inputs(inputs);
        assert_eq!(witness, generated, "{inputs:?}");
        assert_eq!(witness.instances[0].stage1.trace_bytes(), generated.instances[0].stage1.trace_bytes());
    }
    let path = public_inputs(&dir, "1", "2");
    let witness = compute_witness(&mut *library, &all_shape(), Some(&path)).unwrap();
    assert_eq!(witness, all::witness());
    let publics: Vec<String> = witness.publics.iter().map(FrBytes::to_decimal).collect();
    assert_eq!(publics, PIL_FFLONK_PUBLICS);

    // With no public inputs, both are 0, as a STARK library's `load_from_json` gives.
    let witness = compute_witness(&mut *library, &all_shape(), None).unwrap();
    assert_eq!(witness, all::witness_of_inputs([0, 0]));
}

#[test]
fn the_library_refuses_what_it_cannot_compute() {
    let dir = TestDir::new("refusals");
    let mut library = load_witness_library(&all_library(), 0).unwrap();
    // Public inputs that are not Fr values: a JSON number, a value not below r.
    for text in [r#"{"in1": 1}"#, &format!(r#"{{"in2": "{BN254_R}"}}"#)] {
        let path = dir.file("bad.json");
        fs::write(&path, text).unwrap();
        let err = compute_witness(&mut *library, &all_shape(), Some(&path)).unwrap_err();
        assert!(matches!(err, PilfflonkError::InFile { .. }), "{text}: {err}");
    }
    // A key whose AIR is not the fixture's.
    let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: N_BITS, n_cols: 15, n_air_values: 0 };
    let err = compute_witness(&mut *library, &WitnessShape::new(vec![air], 3, 0).unwrap(), None).unwrap_err();
    assert!(err.to_string().contains("the All has 256 rows and 16 columns"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// On the compiled fixture (needs PIL2C_EXEC)
// ---------------------------------------------------------------------------------------------

/// Compiles the fixture on `bus` (`sum` or `prod`) over BN254 with `PIL2C_EXEC` to `dir/all.pilout`,
/// as `src/lib.rs` says: the pilout's name, `all`, is the stem of its file, whatever the bus.
fn compile(dir: &TestDir, bus: &str) -> PathBuf {
    let compiler = std::env::var("PIL2C_EXEC")
        .expect("PIL2C_EXEC must name a pil2com that honours `prime` (e.g. <pil2-compiler>/src/pil.js)");
    let pilout = dir.file("all.pilout");
    let out = Command::new(compiler)
        .current_dir(repo_root())
        .arg(format!("pilfflonk/tests/fixtures/all/all_{bus}.pil"))
        .args(["-I", "pil2-components/lib/std/pil", "-P", "pilfflonk/tests/fixtures/fibonacci/bn254.json", "-o"])
        .arg(&pilout)
        .output()
        .expect("PIL2C_EXEC runs");
    assert!(out.status.success(), "pil2com: {}", String::from_utf8_lossy(&out.stderr));
    pilout
}

/// `src/pil_helpers` is what pil-helpers writes for the sum bus's pilout, and for the product bus's
/// but for `PILOUT_HASH`: the two have the same rows and publics.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_pil_helpers_are_those_pil_helpers_writes() {
    let versioned = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pil_helpers");
    for bus in ["sum", "prod"] {
        let dir = TestDir::new(&format!("pil_helpers_{bus}"));
        let pilout = compile(&dir, bus);
        let out = dir.file("src");
        PilHelpersCmd { pilout, path: out.clone(), overide: false, verbose: 0 }.run().unwrap();
        for file in ["mod.rs", "traces.rs"] {
            let generated = fs::read_to_string(out.join("pil_helpers").join(file)).unwrap();
            let expected = fs::read_to_string(versioned.join(file)).unwrap();
            let differ: Vec<(&str, &str)> = generated.lines().zip(expected.lines()).filter(|(g, e)| g != e).collect();
            let hash_only = differ.iter().all(|(g, _)| g.starts_with("pub const PILOUT_HASH: &str = "));
            assert_eq!(generated.lines().count(), expected.lines().count(), "{bus}: src/pil_helpers/{file}");
            assert!(hash_only && (bus == "prod" || differ.is_empty()), "{bus}: src/pil_helpers/{file}: {differ:?}");
        }
    }
}

/// The stage-1 columns of `info` that are not im pols, as `(name, stageId)`: the witness's columns
/// (`proofman_pilfflonk::witness`). None is an element of an array.
fn stage1_columns(info: &PilfflonkInfo) -> Vec<(String, usize)> {
    let columns = info.cm_pols_map.iter().filter(|p| p.stage == 1 && !p.im_pol);
    columns
        .map(|p| {
            assert!(p.lengths.is_empty(), "{}{:?} is an element of an array", p.name, p.lengths);
            (p.name.clone(), p.stage_id as usize)
        })
        .collect()
}

/// The fields of the row, each with its column: its offset in the row, in `Bn254` values.
macro_rules! row_columns {
    ($row:ty: $($field:ident),*) => {
        vec![$((stringify!($field).to_string(), offset_of!($row, $field) / size_of::<Bn254>())),*]
    };
}

/// On the keys of either bus: the library's rows are the key's stage-1 columns, each field the
/// column of its name at its `stageId`, with nothing else in a row; the library's witness is the
/// generator's on the key's shape, and its constraints hold, which those of the generator's with a
/// wrong multiplicity do not.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_rows_are_the_stage_1_columns_of_the_keys() {
    type Row = AllTraceRow<Bn254>;
    for bus in ["sum", "prod"] {
        let dir = TestDir::new(&format!("keys_{bus}"));
        let opts = SetupPilfflonkOptions {
            airout_path: compile(&dir, bus),
            build_dir: dir.file("build"),
            powers_of_tau: dir.file("tau_one.ptau"),
            max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
            extra_muls: DEFAULT_EXTRA_MULS,
            max_q_degree: DEFAULT_MAX_Q_DEGREE,
            no_packing: false,
            solidity: false,
        };
        // More powers than the largest degree of the layout, 2312 (`cli/tests/pilfflonk_prove.rs`); the
        // check commits nothing, so the ptau of τ = 1 does.
        write_tau_one_ptau(&opts.powers_of_tau, 4096).unwrap();
        run_setup_pilfflonk(&opts).unwrap();
        let pk = ProvingKey::load(&opts.build_dir.join(PROVING_KEY_DIR)).unwrap();
        let info = pk.air(AirInstanceRef { airgroup_id: 0, air_id: 0 }).unwrap();

        let fields = row_columns!(Row:
            l1, l2,
            connection_a, connection_b, connection_c,
            permutation_a, permutation_b, permutation_c, permutation_d, permutation_selC, permutation_selD,
            plookup_sel, plookup_a, plookup_b, plookup_cc, plookup_mul
        );
        assert_eq!(fields, stage1_columns(info), "{bus}");
        assert_eq!(AllTrace::<Bn254>::ROW_SIZE, stage1_columns(info).len(), "{bus}");
        assert_eq!(size_of::<Row>(), AllTrace::<Bn254>::ROW_SIZE * size_of::<Bn254>(), "{bus}: no padding");
        let shape = pk.witness_shape().unwrap();
        assert_eq!(shape, all_shape(), "{bus}");

        let inputs = public_inputs(&dir, "1", "2");
        let mut library = load_witness_library(&all_library(), 0).unwrap();
        let witness = compute_witness(&mut *library, &shape, Some(&inputs)).unwrap();
        assert_eq!(witness, all::witness(), "{bus}");
        assert!(check(&pk, &witness, &CheckOptions::default()).unwrap().holds(), "{bus}");
        let broken = all::witness_with_a_wrong_multiplicity();
        assert!(!check(&pk, &broken, &CheckOptions::default()).unwrap().holds(), "{bus}");
    }
}
