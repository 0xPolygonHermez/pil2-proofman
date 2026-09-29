//! `proofman-setup setup-pilfflonk` through the binary: its arguments (spec §4.2) and the files it
//! writes in the `provingKey/` (§4.2.6), on a pilout built in code and on the compiled Fibonacci
//! fixture. Its steps are tested in `setup/pilfflonk/tests/`.
//!
//! Pilouts are not versioned: the `#[ignore]` test compiles the fixture with `compile-pil` and
//! the compiler `PIL2C_EXEC` names, which must honour `prime` (the pinned one silently compiles
//! over Goldilocks):
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pil2-stark-setup --features proofman-starks-lib-c/cpu-only \
//!     --test setup_pilfflonk -- --include-ignored
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pil2_pilout::pilout::{self as pb, constraint, expression, operand};
use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use prost::Message;
use proofman_pilfflonk::{
    AirFile, AirVerkey, FqBytes, FrBytes, G1Affine, GlobalInfoAir, JsonFile, NameStageEntry, PilfflonkGlobalInfo,
    SetupParams,
};

const BN254_R_BE: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d, 0x28, 0x33, 0xe8,
    0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
];

/// The generator of G1, `(1, 2)`: with `τ = 1`, the commitment of a fixed column whose first
/// row is 1. One whose first row is 0 commits to the point at infinity, `(0, 0)`.
fn g() -> G1Affine {
    G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(2) }
}

/// A fresh directory for one test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("setup_pilfflonk_{name}_{}", std::process::id()));
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

fn proofman_setup(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proofman-setup")).args(args).output().expect("proofman-setup runs")
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// Every file under `dir`, relative to it, sorted.
fn files(dir: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.push(path.strip_prefix(root).unwrap().to_string_lossy().into_owned());
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

/// A BN254 pilout of one AIR of 4 rows: fixed columns `[1, 5, 6, 7]` and `[0, 1, 0, 1]`, a
/// witness column, one public and the constraint `a − F0 = 0`.
fn pilout() -> pb::PilOut {
    let op = |operand| Some(pb::Operand { operand: Some(operand) });
    let fixed_col = |values: [u8; 4]| pb::FixedCol { values: values.iter().map(|&v| vec![v]).collect() };
    let air = pb::Air {
        name: Some("Tiny".into()),
        num_rows: Some(4),
        fixed_cols: vec![fixed_col([1, 5, 6, 7]), fixed_col([0, 1, 0, 1])],
        stage_widths: vec![1],
        expressions: vec![pb::Expression {
            operation: Some(expression::Operation::Sub(expression::Sub {
                lhs: op(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx: 0, row_offset: 0 })),
                rhs: op(operand::Operand::FixedCol(operand::FixedCol { idx: 0, row_offset: 0 })),
            })),
        }],
        constraints: vec![pb::Constraint {
            constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
                expression_idx: Some(operand::Expression { idx: 0 }),
                debug_line: None,
            })),
        }],
        ..Default::default()
    };
    pb::PilOut {
        name: Some("tiny".into()),
        base_field: BN254_R_BE.to_vec(),
        air_groups: vec![pb::AirGroup { name: Some("TinyGroup".into()), air_group_values: vec![], airs: vec![air] }],
        num_challenges: vec![0],
        num_public_values: 1,
        symbols: vec![pb::Symbol {
            name: "in".into(),
            r#type: pb::SymbolType::PublicValue as i32,
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn the_subcommand_takes_the_arguments_of_spec_4_2() {
    let out = proofman_setup(&["setup-pilfflonk", "--help"]);
    assert!(out.status.success());
    let help = String::from_utf8_lossy(&out.stdout);
    for arg in [
        "-a, --airout",
        "-b, --build-dir",
        "--powers-of-tau",
        "--max-constraint-degree <MAX_CONSTRAINT_DEGREE>",
        "[default: 9]",
        "--extra-muls <EXTRA_MULS>",
        "[default: 2]",
        "--max-q-degree <MAX_Q_DEGREE>",
        "[default: 0]",
        "--no-packing",
    ] {
        assert!(help.contains(arg), "{arg} missing from:\n{help}");
    }
    // No -u: at BN254 nothing produces the STARK's 8-byte .fixed files (spec §4.2, C4).
    assert!(!help.contains("-u,") && !help.contains("--solidity"), "{help}");

    // -a, -b and --powers-of-tau are required.
    let out = proofman_setup(&["setup-pilfflonk", "-a", "x.pilout", "-b", "build"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--powers-of-tau"));
}

#[test]
fn it_writes_the_proving_key_of_a_pilout() {
    let dir = TestDir::new("tiny");
    let pilout_path = dir.file("tiny.pilout");
    let ptau = dir.file("tau_one.ptau");
    let build = dir.file("build");
    fs::write(&pilout_path, pilout().encode_to_vec()).unwrap();
    write_tau_one_ptau(&ptau, 16).unwrap();

    let args = ["setup-pilfflonk", "-a", path(&pilout_path), "-b", path(&build), "--powers-of-tau", path(&ptau)];
    let out =
        proofman_setup(&[&args[..], &["--no-packing", "--max-constraint-degree", "4", "--extra-muls", "0"]].concat());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let proving_key = build.join("provingKey");
    assert_eq!(
        files(&proving_key),
        [
            "pilout.globalInfo.json",
            "tiny/TinyGroup/airs/Tiny/air/Tiny.const",
            "tiny/TinyGroup/airs/Tiny/air/Tiny.verkey.json",
            "tiny/pilfflonk/pilfflonk.srs.bin",
        ]
    );
    let gi = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
    assert_eq!(
        gi.setup_params,
        SetupParams { max_constraint_degree: 4, extra_muls: 0, max_q_degree: 0, packing: false }
    );
    let fixed = FixedColumns::read_const(&gi.air_file(&proving_key, 0, 0, AirFile::Const).unwrap(), 2, 2).unwrap();
    let column: Vec<FrBytes> = [1, 5, 6, 7].map(FrBytes::from_u64).to_vec();
    assert_eq!(fixed.column(0).unwrap(), column.as_slice());
    let verkey = AirVerkey::read(&gi.air_file(&proving_key, 0, 0, AirFile::Verkey).unwrap()).unwrap();
    assert_eq!(verkey, AirVerkey(vec![g(), G1Affine::INFINITY]));

    // What the setup cannot do yet fails with the reason, and a non-zero status.
    for (extra, expected) in [
        (&[][..], "pass --no-packing"),
        (&["--no-packing", "--max-q-degree", "2"][..], "--max-q-degree 2"),
        (&["--no-packing", "--max-constraint-degree", "1"][..], "--max-constraint-degree 1"),
    ] {
        let out = proofman_setup(&[&args[..], extra].concat());
        assert!(!out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(expected), "{stderr}");
    }

    // A Goldilocks pilout, what the pinned compiler writes for `-P bn254.json` (plan M13).
    let mut goldilocks = pilout();
    goldilocks.base_field = 0xFFFF_FFFF_0000_0001u64.to_be_bytes().to_vec();
    fs::write(&pilout_path, goldilocks.encode_to_vec()).unwrap();
    let out = proofman_setup(&[&args[..], &["--no-packing"]].concat());
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("over Goldilocks") && stderr.contains("PIL2C_EXEC"), "{stderr}");
}

/// The Fibonacci fixture (plan M13), compiled over BN254: the files of M15 in the paths of spec
/// §4.2.6, and their contents.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn it_writes_the_proving_key_of_the_fibonacci_fixture() {
    std::env::var("PIL2C_EXEC").expect("PIL2C_EXEC must name a pil2com that honours `prime`");
    let dir = TestDir::new("fibonacci");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    let pilout_path = dir.file("fibonacci.pilout");
    let out = Command::new(env!("CARGO_BIN_EXE_proofman-setup"))
        .current_dir(&root)
        .args(["compile-pil", "-p", "pilfflonk/tests/fixtures/fibonacci/fibonacci.pil"])
        .args(["-I", "pil2-components/lib/std/pil", "-P", "pilfflonk/tests/fixtures/fibonacci/bn254.json"])
        .args(["-o", path(&pilout_path)])
        .output()
        .unwrap();
    assert!(out.status.success(), "compile-pil: {}", String::from_utf8_lossy(&out.stderr));

    let ptau = dir.file("tau_one.ptau");
    write_tau_one_ptau(&ptau, 1024).unwrap();
    let build = dir.file("build");
    let out = proofman_setup(&[
        "setup-pilfflonk",
        "-a",
        path(&pilout_path),
        "-b",
        path(&build),
        "--powers-of-tau",
        path(&ptau),
        "--no-packing",
    ]);
    assert!(out.status.success(), "setup-pilfflonk: {}", String::from_utf8_lossy(&out.stderr));

    let proving_key = build.join("provingKey");
    assert_eq!(
        files(&proving_key),
        [
            "fibonacci/Fibonacci/airs/Fibonacci/air/Fibonacci.const",
            "fibonacci/Fibonacci/airs/Fibonacci/air/Fibonacci.verkey.json",
            "fibonacci/pilfflonk/pilfflonk.srs.bin",
            "pilout.globalInfo.json",
        ]
    );

    let gi = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
    assert_eq!(gi.airs, vec![vec![GlobalInfoAir { name: "Fibonacci".into(), num_rows: 256 }]]);
    assert_eq!(gi.air_groups, vec!["Fibonacci".to_string()]);
    let publics: Vec<NameStageEntry> =
        ["in1", "in2", "out"].map(|name| NameStageEntry { name: name.into(), stage: 1, lengths: vec![] }).to_vec();
    assert_eq!((gi.n_publics, gi.publics_map.clone()), (3, publics));
    assert_eq!(
        gi.setup_params,
        SetupParams { max_constraint_degree: 9, extra_muls: 2, max_q_degree: 0, packing: false }
    );

    // L1 = [1, 0…] and LLAST = [0…, 1], row by row.
    let fixed = FixedColumns::read_const(&gi.air_file(&proving_key, 0, 0, AirFile::Const).unwrap(), 8, 2).unwrap();
    let one_at = |row: usize| -> Vec<FrBytes> {
        (0..256).map(|i| if i == row { FrBytes::from_u64(1) } else { FrBytes::ZERO }).collect()
    };
    assert_eq!(fixed.column(0).unwrap(), one_at(0).as_slice());
    assert_eq!(fixed.column(1).unwrap(), one_at(255).as_slice());

    // With τ = 1: L1(1)·G = G and LLAST(1)·G, the point at infinity.
    let verkey = AirVerkey::read(&gi.air_file(&proving_key, 0, 0, AirFile::Verkey).unwrap()).unwrap();
    assert_eq!(verkey, AirVerkey(vec![g(), G1Affine::INFINITY]));

    // The SRS: the 256 powers of the unpacked fixed f (M16 sizes it by the layout).
    let srs = fs::read(gi.srs_path(&proving_key)).unwrap();
    assert_eq!((&srs[..4], srs.len()), (&b"pfsr"[..], 12 + 3 * 12 + 88 + 256 * 64 + 2 * 128));
}
