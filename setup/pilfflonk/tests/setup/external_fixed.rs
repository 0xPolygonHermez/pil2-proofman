//! The fixed columns that a pilout declares `#pragma fixed_external`
//! (pilfflonk/docs/formats.md#fixed-columns), on fixtures compiled by pil2com, each the copy of
//! another fixture with some of its fixed columns external:
//!
//! - `pilfflonk/tests/fixtures/packed/packed_external.pil`, `packed.pil` with the array `K[4]` and
//!   the column `S` external, which the default grouping packs with the inline `L1` and `LLAST`;
//! - `pilfflonk/tests/fixtures/connection/connection_external.pil`, `connection_prod.pil` with the
//!   permutations `S1`, `S2` and `S3` external, as the wrap's `S`; the std's `connection()` keeps
//!   its `ID` inline. Stage 2, the product bus.
//!
//! For each, the test:
//!
//! 1. compiles both programs, and checks that the external pilout is the other one without the
//!    values of the external columns: the same fixed columns, in the same order, the same symbols,
//!    constraints and hints, but for the source lines;
//! 2. sets up the inline pilout, and the external one with the inline key's values of the external
//!    columns (its `<air>.const`, at the positions of its `constPolsMap`), through
//!    `run_setup_pilfflonk_with_external_fixed`: the same `provingKey/`, byte for byte, its
//!    `<air>.const`, verkey (the fixed commitments) and vkey (its digest included) among the rest;
//!    but for the source lines of the constraints, which `<air>.bin` and the expressionsinfo hold;
//! 3. sets up the external pilout without them, which is refused as a pilout without the values of
//!    a fixed column was before;
//! 4. proves the fixture's witness with the external key and a fixed blinding seed: the JS verifier
//!    accepts the proof, which is the one the inline key gives.
//!
//! It needs `PIL2C_EXEC`, a compiler that has `--field`, and Node.js, and is `#[ignore]` without
//! them (pilfflonk/docs/README.md#tests):
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pilfflonk-setup \
//!     --features proofman-starks-lib-c/cpu-only --test setup external_fixed -- --ignored
//! ```

#[allow(dead_code)]
#[path = "../../../../pilfflonk/tests/data/connection.rs"]
mod connection;
#[allow(dead_code)]
#[path = "../../../../pilfflonk/tests/data/packed.rs"]
mod packed;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use pil2_pilout::pilout::{self as pb, constraint};
use pilfflonk_setup::bytecode::Bytecode;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{
    run_setup_pilfflonk, run_setup_pilfflonk_with_external_fixed, ExternalFixedColumn, SetupError,
    SetupPilfflonkOptions,
};
use prost::Message;
use proofman_pilfflonk::{
    js_verifier, prove, AirFile, JsonFile, PilfflonkGlobalInfo, PilfflonkInfo, ProveOptions, ProvingKey, Witness,
};

use crate::common::{cpp_core, TestDir};

/// The blinding seed of the proofs (pilfflonk/docs/protocol.md#blinding).
const SEED: [u8; 32] = [0x3c; 32];

/// A fixture and its copy with some fixed columns external.
struct Fixture {
    name: &'static str,
    inline: &'static str,
    external: &'static str,
    /// The names of the external columns' symbols.
    columns: &'static [&'static str],
    witness: fn() -> Witness,
    /// More powers than the largest degree of its layout: the Connection's (`N = 2^10`) is 3080.
    ptau_powers: usize,
}

const PACKED: Fixture = Fixture {
    name: "packed",
    inline: "pilfflonk/tests/fixtures/packed/packed.pil",
    external: "pilfflonk/tests/fixtures/packed/packed_external.pil",
    columns: &["Packed.K", "Packed.S"],
    // The input of cli/tests/pilfflonk_prove.rs.
    witness: || packed::witness(5),
    ptau_powers: 1024,
};

const CONNECTION: Fixture = Fixture {
    name: "connection",
    inline: "pilfflonk/tests/fixtures/connection/connection_prod.pil",
    external: "pilfflonk/tests/fixtures/connection/connection_external.pil",
    columns: &["Connection.S1", "Connection.S2", "Connection.S3"],
    witness: connection::witness,
    ptau_powers: 4096,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Compiles `pil` over BN254 to `pilout` with `PIL2C_EXEC`, and reads it back.
fn compile(pil: &str, pilout: &Path) -> pb::PilOut {
    let compiler = std::env::var("PIL2C_EXEC").expect("PIL2C_EXEC must name a pil2com that has `--field`");
    let out = Command::new(compiler)
        .current_dir(repo_root())
        .arg(pil)
        .args(["-I", "pil2-components/lib/std/pil", "--field", "bn254", "-o"])
        .arg(pilout)
        .output()
        .expect("PIL2C_EXEC runs");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "pil2com: {log}");
    pb::PilOut::decode(fs::read(pilout).unwrap().as_slice()).unwrap()
}

/// `pilout` without the source lines of its symbols and constraints, which tell the two programs
/// apart.
fn without_source_lines(mut pilout: pb::PilOut) -> pb::PilOut {
    for symbol in &mut pilout.symbols {
        symbol.debug_line = None;
    }
    for constraint in &mut pilout.constraints {
        constraint.debug_line = None;
    }
    let airs = pilout.air_groups.iter_mut().flat_map(|group| group.airs.iter_mut());
    for c in airs.flat_map(|air| air.constraints.iter_mut()).filter_map(|c| c.constraint.as_mut()) {
        match c {
            constraint::Constraint::FirstRow(c) => c.debug_line = None,
            constraint::Constraint::LastRow(c) => c.debug_line = None,
            constraint::Constraint::EveryRow(c) => c.debug_line = None,
            constraint::Constraint::EveryFrame(c) => c.debug_line = None,
        }
    }
    pilout
}

/// The options of a setup of the pilout `<dir>/program.pilout` into `<dir>/build`, grouped as by
/// default.
fn options(dir: &Path, ptau: &Path) -> SetupPilfflonkOptions {
    SetupPilfflonkOptions {
        airout_path: dir.join("program.pilout"),
        build_dir: dir.join("build"),
        powers_of_tau: ptau.to_path_buf(),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: false,
        solidity: false,
    }
}

/// A file of the `provingKey/` of `opts`.
fn key_file(opts: &SetupPilfflonkOptions, file: AirFile) -> PathBuf {
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap().air_file(&proving_key, 0, 0, file).unwrap()
}

/// The `columns` of the key of `opts`, from its `<air>.const`, as external columns: each fixed
/// column of `constPolsMap` that `columns` names, at the index its `lengths` give (the arrays of
/// the fixtures have one dimension).
fn external_columns(opts: &SetupPilfflonkOptions, columns: &[&str]) -> Vec<ExternalFixedColumn> {
    let info = PilfflonkInfo::read(&key_file(opts, AirFile::PilfflonkInfo)).unwrap();
    let fixed =
        FixedColumns::read_const(&key_file(opts, AirFile::Const), info.n_bits, info.const_pols_map.len()).unwrap();
    for name in columns {
        assert!(info.const_pols_map.iter().any(|p| p.name == *name), "{name} is a fixed column of the key");
    }
    info.const_pols_map
        .iter()
        .enumerate()
        .filter(|(_, p)| columns.contains(&p.name.as_str()))
        .map(|(i, p)| ExternalFixedColumn {
            name: p.name.clone(),
            index: p.lengths.first().map_or(0, |&l| l as usize),
            values: fixed.column(i).unwrap().to_vec(),
        })
        .collect()
}

/// Every file of the `provingKey/` of `opts`, relative to it, with its content: its bytes, but for
/// the two that hold the source lines of the constraints, which tell the two programs apart. Of
/// those, `<air>.bin` as `Bytecode::read` reads it, and the expressionsinfo as JSON, both without
/// their lines.
fn key_contents(opts: &SetupPilfflonkOptions) -> Vec<(String, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    fn without_lines(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                map.remove("line");
                map.values_mut().for_each(without_lines);
            }
            serde_json::Value::Array(values) => values.iter_mut().for_each(without_lines),
            _ => {}
        }
    }
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let mut paths = Vec::new();
    walk(&proving_key, &mut paths);
    paths.sort();
    let (bin, expressions_info) = (key_file(opts, AirFile::Bin), key_file(opts, AirFile::ExpressionsInfo));
    paths
        .into_iter()
        .map(|path| {
            let content = if path == bin {
                let mut bytecode = Bytecode::read(&path).unwrap();
                bytecode.expressions.iter_mut().for_each(|e| e.line.clear());
                bytecode.constraints.iter_mut().for_each(|c| c.line.clear());
                format!("{bytecode:?}")
            } else if path == expressions_info {
                let mut json: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                without_lines(&mut json);
                json.to_string()
            } else {
                format!("{:?}", fs::read(&path).unwrap())
            };
            (path.strip_prefix(&proving_key).unwrap().display().to_string(), content)
        })
        .collect()
}

/// Steps 1 to 4 of the module on `fixture`.
fn external_columns_give_the_inline_key(fixture: &Fixture) {
    let _cpp = cpp_core();
    let dir = TestDir::new(&format!("external_fixed_{}", fixture.name));
    let ptau = dir.file("fixed_tau.ptau");
    write_fixed_tau_ptau(&ptau, fixture.ptau_powers, &test_tau()).unwrap();
    let (inline_dir, external_dir) = (dir.file("inline"), dir.file("external"));
    fs::create_dir_all(&inline_dir).unwrap();
    fs::create_dir_all(&external_dir).unwrap();
    let (inline, external) = (options(&inline_dir, &ptau), options(&external_dir, &ptau));

    // 1. The fixture: the inline pilout without the values of the external columns; the others,
    // the std's ID among them, keep theirs.
    let mut expected = compile(fixture.inline, &inline.airout_path);
    let external_pilout = compile(fixture.external, &external.airout_path);
    run_setup_pilfflonk(&inline).unwrap();
    let info = PilfflonkInfo::read(&key_file(&inline, AirFile::PilfflonkInfo)).unwrap();
    let fixed_cols = &mut expected.air_groups[0].airs[0].fixed_cols;
    assert_eq!(fixed_cols.len(), info.const_pols_map.len());
    for (col, entry) in fixed_cols.iter_mut().zip(&info.const_pols_map) {
        assert!(!col.values.is_empty(), "{} has its values in {}", entry.name, fixture.inline);
        if fixture.columns.contains(&entry.name.as_str()) {
            col.values.clear();
        }
    }
    assert_eq!(without_source_lines(external_pilout), without_source_lines(expected));

    // 2. The same key, with the inline key's values.
    let columns = external_columns(&inline, fixture.columns);
    run_setup_pilfflonk_with_external_fixed(&external, columns).unwrap();
    assert_eq!(key_contents(&external), key_contents(&inline));

    // 3. Without them, refused as before.
    let refused = SetupPilfflonkOptions { build_dir: external_dir.join("refused"), ..external.clone() };
    let err = run_setup_pilfflonk(&refused).unwrap_err();
    let first = info.const_pols_map.iter().position(|p| fixture.columns.contains(&p.name.as_str())).unwrap();
    assert!(
        matches!(err.downcast_ref::<SetupError>(), Some(SetupError::FixedValues { column, n_values: 0, .. })
            if *column == first),
        "{err:#}"
    );
    assert!(!refused.build_dir.exists());

    // 4. The external key proves, and its proof is the inline key's.
    let options = ProveOptions { insecure_blinding_seed: Some(SEED), q_part_bits: None };
    let witness = (fixture.witness)();
    let external_key = external.build_dir.join(PROVING_KEY_DIR);
    let out = prove(&ProvingKey::load(&external_key).unwrap(), &witness, &options).unwrap();
    let (proof, publics) = (external_dir.join("proof.json"), external_dir.join("publics.json"));
    out.proof_json().unwrap().write(&proof).unwrap();
    out.publics.write(&publics).unwrap();
    let vkey = PilfflonkGlobalInfo::from_proving_key(&external_key).unwrap().vkey_path(&external_key);
    assert!(js_verifier::verify(&vkey, &publics, &proof).unwrap(), "the JS verifier accepts the proof");
    let inline_key = inline.build_dir.join(PROVING_KEY_DIR);
    let inline_out = prove(&ProvingKey::load(&inline_key).unwrap(), &witness, &options).unwrap();
    assert_eq!(out.proof, inline_out.proof);
}

#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn external_fixed_columns_of_the_packed_fixture_give_its_key() {
    external_columns_give_the_inline_key(&PACKED);
}

#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn external_fixed_columns_of_the_connection_give_its_key() {
    external_columns_give_the_inline_key(&CONNECTION);
}
