//! `proofman-setup setup-pilfflonk` through the binary: its arguments
//! (pilfflonk/docs/README.md#setup-pilfflonk) and the files it writes in the `provingKey/`
//! (pilfflonk/docs/formats.md#provingkey), on a pilout built in code and on the compiled Fibonacci
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
use pilfflonk_setup::bytecode::Bytecode;
use pilfflonk_setup::digest::keccak256;
use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use prost::Message;
use proofman_pilfflonk::global_info::GLOBAL_CONSTRAINTS_FILE;
use proofman_pilfflonk::{
    AirFile, AirVerkey, FqBytes, FrBytes, G1Affine, GlobalInfoAir, JsonFile, NameStageEntry, PilfflonkGlobalInfo,
    PilfflonkInfo, PolType, ProofNames, SetupParams, Vkey, WitnessShape,
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

/// A BN254 pilout of one AIR of 4 rows: fixed columns `F0 = [1, 5, 6, 7]` and `F1 = [0, 1, 0,
/// 1]`, a witness column `a`, one public and the constraint `a − F0 = 0`. `F1` is in no
/// constraint: it is never opened.
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
        symbols: vec![
            pb::Symbol { name: "in".into(), r#type: pb::SymbolType::PublicValue as i32, ..Default::default() },
            column_symbol("Tiny.F0", pb::SymbolType::FixedCol, 0, 0),
            column_symbol("Tiny.F1", pb::SymbolType::FixedCol, 1, 0),
            column_symbol("Tiny.a", pb::SymbolType::WitnessCol, 0, 1),
        ],
        ..Default::default()
    }
}

/// The symbol of column `id` of `stage` of the AIR.
fn column_symbol(name: &str, kind: pb::SymbolType, id: u32, stage: u32) -> pb::Symbol {
    pb::Symbol {
        name: name.into(),
        air_group_id: Some(0),
        air_id: Some(0),
        r#type: kind as i32,
        id,
        stage: Some(stage),
        ..Default::default()
    }
}

/// The files of the `provingKey/` (pilfflonk/docs/formats.md#provingkey) for the pilout `name`, its
/// airgroup and its AIR.
fn proving_key_files(name: &str, airgroup: &str, air: &str) -> Vec<String> {
    let mut files: Vec<String> =
        ["bin", "const", "expressionsinfo.json", "pilfflonkinfo.json", "verifierinfo.json", "verkey.json"]
            .iter()
            .map(|ext| format!("{name}/{airgroup}/airs/{air}/air/{air}.{ext}"))
            .collect();
    files.push(format!("{name}/pilfflonk/pilfflonk.srs.bin"));
    files.push(format!("{name}/pilfflonk/pilfflonk.vkey.json"));
    files.push("pilout.globalConstraints.json".into());
    files.push("pilout.globalInfo.json".into());
    files
}

/// An `f` of a layout as `(stage, the names of its polynomials, k, offsets, degree)`.
type FShape<'a> = (u64, Vec<&'a str>, u64, Vec<i64>, u64);

/// Each `f` of the layout of `info` as an [`FShape`].
fn f_shapes(info: &PilfflonkInfo) -> Vec<FShape<'_>> {
    info.layout
        .0
        .iter()
        .map(|f| (f.stage, f.pols.iter().map(|p| p.name.as_str()).collect(), f.k, f.offsets.clone(), f.degree))
        .collect()
}

/// The layout of `info` as `(stage, id, name, offsets, degree)` for each `f`, all of `k = 1`.
fn layout(info: &PilfflonkInfo) -> Vec<(u64, u64, String, Vec<i64>, u64)> {
    assert!(info.layout.0.iter().all(|f| f.k == 1 && f.pols.len() == 1));
    info.layout.0.iter().map(|f| (f.stage, f.pols[0].id, f.pols[0].name.clone(), f.offsets.clone(), f.degree)).collect()
}

/// What the prover and the verifier read of a `provingKey/` accepts this one: every file by its
/// type, the vkey's digest, the names of the proof and the shape of the witness. Returns the
/// globalInfo, the pilfflonkinfo and the vkey.
fn check_proving_key(proving_key: &Path) -> (PilfflonkGlobalInfo, PilfflonkInfo, Vkey) {
    let gi = PilfflonkGlobalInfo::from_proving_key(proving_key).unwrap();
    let info = PilfflonkInfo::read(&gi.air_file(proving_key, 0, 0, AirFile::PilfflonkInfo).unwrap()).unwrap();
    let verkey = AirVerkey::read(&gi.air_file(proving_key, 0, 0, AirFile::Verkey).unwrap()).unwrap();
    let vkey = Vkey::read(&gi.vkey_path(proving_key)).unwrap();
    assert!(vkey.digest_matches(|data| keccak256(data).unwrap()).unwrap());
    assert_eq!((&vkey.layout, &vkey.ev_map, &vkey.fixed_commitments.0), (&info.layout, &info.ev_map, &verkey.0));
    Bytecode::read(&gi.air_file(proving_key, 0, 0, AirFile::Bin).unwrap()).unwrap();
    for file in [AirFile::ExpressionsInfo, AirFile::VerifierInfo] {
        let text = fs::read(gi.air_file(proving_key, 0, 0, file).unwrap()).unwrap();
        serde_json::from_slice::<serde_json::Value>(&text).unwrap();
    }
    let text = fs::read(proving_key.join(GLOBAL_CONSTRAINTS_FILE)).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&text).unwrap(),
        serde_json::json!({"constraints": [], "hints": []})
    );
    ProofNames::new(&gi, &[&info]).unwrap();
    WitnessShape::from_proving_key(&gi, &[&info]).unwrap();
    (gi, info, vkey)
}

/// The bytes of every file under `dir`, in the order of [`files`].
fn contents(dir: &Path) -> Vec<(String, Vec<u8>)> {
    files(dir).into_iter().map(|f| (f.clone(), fs::read(dir.join(&f)).unwrap())).collect()
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
        // The Solidity verifier (pilfflonk/docs/verifier.md#solidity-verifier).
        "--solidity",
    ] {
        assert!(help.contains(arg), "{arg} missing from:\n{help}");
    }
    // No -u: at BN254 nothing produces the STARK's 8-byte .fixed files
    // (pilfflonk/docs/README.md#compile-pil).
    assert!(!help.contains("-u,"), "{help}");

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
    let mut expected = proving_key_files("tiny", "TinyGroup", "Tiny");
    expected.sort();
    assert_eq!(files(&proving_key), expected);
    let (gi, info, _) = check_proving_key(&proving_key);
    // a − F0 has degree 1: qDeg = 0, and Q has |O|_max + 1 = 2 coefficients
    // (pilfflonk/docs/protocol.md#degrees). F1 is never opened, and not committed.
    let s = |name: &str| name.to_string();
    assert_eq!(
        layout(&info),
        [(0, 0, s("Tiny.F0"), vec![0], 4), (1, 0, s("Tiny.a"), vec![0], 6), (2, 1, s("Q0"), vec![0], 2)]
    );
    // The SRS holds the largest degree, a's 6.
    let srs = fs::read(gi.srs_path(&proving_key)).unwrap();
    assert_eq!(srs.len(), 12 + 3 * 12 + 88 + 6 * 64 + 2 * 128);
    assert_eq!(
        gi.setup_params,
        SetupParams { max_constraint_degree: 4, extra_muls: 0, max_q_degree: 0, packing: false }
    );
    let fixed = FixedColumns::read_const(&gi.air_file(&proving_key, 0, 0, AirFile::Const).unwrap(), 2, 2).unwrap();
    let column: Vec<FrBytes> = [1, 5, 6, 7].map(FrBytes::from_u64).to_vec();
    assert_eq!(fixed.column(0).unwrap(), column.as_slice());
    let verkey = AirVerkey::read(&gi.air_file(&proving_key, 0, 0, AirFile::Verkey).unwrap()).unwrap();
    assert_eq!(verkey, AirVerkey(vec![g()]));

    // A second run writes the same bytes.
    let before = contents(&proving_key);
    let out =
        proofman_setup(&[&args[..], &["--no-packing", "--max-constraint-degree", "4", "--extra-muls", "0"]].concat());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(contents(&proving_key), before);

    // Grouped, the default: with no extra mul, the same f, one per class, and the same SRS; the
    // default --extra-muls 2 is more than its 3 polynomials in 3 groups allow.
    let grouped = dir.file("grouped");
    let args_grouped =
        ["setup-pilfflonk", "-a", path(&pilout_path), "-b", path(&grouped), "--powers-of-tau", path(&ptau)];
    let out = proofman_setup(&[&args_grouped[..], &["--max-constraint-degree", "4", "--extra-muls", "0"]].concat());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let (gi, grouped_info, _) = check_proving_key(&grouped.join("provingKey"));
    assert_eq!(grouped_info.layout, info.layout);
    assert!(gi.setup_params.packing);

    // Every --max-q-degree is one (pilfflonk/docs/protocol.md#q-pieces): qDeg = 0 is not split by
    // 2, and the key is that of Q whole, maxQDegree = 0; the globalInfo records the option.
    let whole = dir.file("max_q_degree");
    let args_whole = ["setup-pilfflonk", "-a", path(&pilout_path), "-b", path(&whole), "--powers-of-tau", path(&ptau)];
    let options = ["--no-packing", "--max-constraint-degree", "4", "--extra-muls", "0", "--max-q-degree", "2"];
    let out = proofman_setup(&[&args_whole[..], &options[..]].concat());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let (whole_gi, whole_info, _) = check_proving_key(&whole.join("provingKey"));
    assert_eq!((whole_info.max_q_degree, whole_gi.setup_params.max_q_degree), (0, 2));
    assert_eq!(whole_info, info);

    // What the setup cannot do fails with the reason, and a non-zero status.
    for (extra, expected) in [
        (&[][..], "so at most 0 extra muls"),
        (&["--no-packing", "--max-constraint-degree", "1"][..], "--max-constraint-degree 1"),
    ] {
        let out = proofman_setup(&[&args[..], extra].concat());
        assert!(!out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(expected), "{stderr}");
    }

    // A Goldilocks pilout, what the pinned compiler writes for `-P bn254.json`
    // (pilfflonk/docs/README.md#compile-pil).
    let mut goldilocks = pilout();
    goldilocks.base_field = 0xFFFF_FFFF_0000_0001u64.to_be_bytes().to_vec();
    fs::write(&pilout_path, goldilocks.encode_to_vec()).unwrap();
    let out = proofman_setup(&[&args[..], &["--no-packing"]].concat());
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("over Goldilocks") && stderr.contains("PIL2C_EXEC"), "{stderr}");
}

/// `setup-pilfflonk --solidity` writes `pilfflonk.verifier.sol` next to the vkey and changes no
/// other file, and `pilfflonk-solidity` writes the same from the vkey alone
/// (pilfflonk/docs/verifier.md#solidity-verifier).
#[test]
fn it_writes_the_solidity_verifier_of_the_vkey() {
    let out = proofman_setup(&["pilfflonk-solidity", "--help"]);
    assert!(out.status.success());
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(help.contains("-k, --vkey <VKEY>") && help.contains("-o, --output <OUTPUT>"), "{help}");

    let dir = TestDir::new("solidity");
    let pilout_path = dir.file("tiny.pilout");
    let ptau = dir.file("tau_one.ptau");
    fs::write(&pilout_path, pilout().encode_to_vec()).unwrap();
    write_tau_one_ptau(&ptau, 16).unwrap();
    let run = |build: &Path, solidity: bool| {
        let mut args = vec!["setup-pilfflonk", "-a", path(&pilout_path), "-b", path(build), "--powers-of-tau"];
        args.extend([path(&ptau), "--no-packing", "--max-constraint-degree", "4", "--extra-muls", "0"]);
        if solidity {
            args.push("--solidity");
        }
        let out = proofman_setup(&args);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        build.join("provingKey")
    };
    let without = run(&dir.file("without"), false);
    let with = run(&dir.file("with"), true);
    let mut expected = proving_key_files("tiny", "TinyGroup", "Tiny");
    expected.push("tiny/pilfflonk/pilfflonk.verifier.sol".into());
    expected.sort();
    assert_eq!(files(&with), expected);
    let sol = with.join("tiny/pilfflonk/pilfflonk.verifier.sol");
    let others: Vec<_> = contents(&with).into_iter().filter(|(name, _)| !name.ends_with(".sol")).collect();
    assert_eq!(others, contents(&without));

    let exported = dir.file("exported.sol");
    let vkey = with.join("tiny/pilfflonk/pilfflonk.vkey.json");
    let out = proofman_setup(&["pilfflonk-solidity", "-k", path(&vkey), "-o", path(&exported)]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(fs::read(&exported).unwrap(), fs::read(&sol).unwrap());
    let text = String::from_utf8(fs::read(&sol).unwrap()).unwrap();
    assert!(text.contains("contract PilfflonkVerifier {"), "{text}");

    // A vkey whose digest is not its own is refused, with a non-zero status.
    let mut tampered = Vkey::read(&vkey).unwrap();
    tampered.digest.0[31] ^= 1;
    fs::write(&vkey, tampered.to_json_string().unwrap()).unwrap();
    let out = proofman_setup(&["pilfflonk-solidity", "-k", path(&vkey), "-o", path(&dir.file("refused.sol"))]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("digest"), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!dir.file("refused.sol").exists());
}

/// The Fibonacci fixture, compiled over BN254: the `provingKey/`
/// (pilfflonk/docs/formats.md#provingkey), with one im pol and `qDeg = 1`
/// (pilfflonk/docs/protocol.md#degree-search), the bounds of its polynomials
/// (pilfflonk/docs/protocol.md#degrees), and the same bytes on a second run.
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
    let args = [
        "setup-pilfflonk",
        "-a",
        path(&pilout_path),
        "-b",
        path(&build),
        "--powers-of-tau",
        path(&ptau),
        "--no-packing",
    ];
    let out = proofman_setup(&args);
    assert!(out.status.success(), "setup-pilfflonk: {}", String::from_utf8_lossy(&out.stderr));

    let proving_key = build.join("provingKey");
    let mut expected = proving_key_files("fibonacci", "Fibonacci", "Fibonacci");
    expected.sort();
    assert_eq!(files(&proving_key), expected);
    let (gi, info, vkey) = check_proving_key(&proving_key);

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

    // One im pol, l1' − (l1·l1 + l2·l2), after l1 and l2 in stage 1, and qDeg = 1. Its expId is
    // the index of that expression in the pilout (the oracle's assumption).
    assert_eq!((info.n_bits, info.n_stages, info.q_deg, info.q_dim, info.max_q_degree), (8, 1, 1, 1, 0));
    let names: Vec<(&str, u64, bool)> =
        info.cm_pols_map.iter().map(|p| (p.name.as_str(), p.stage_id, p.im_pol)).collect();
    assert_eq!(names, [("l1", 0, false), ("l2", 1, false), ("Fibonacci.ImPol", 2, true), ("Q0", 0, false)]);
    let pilout = pb::PilOut::decode(fs::read(&pilout_path).unwrap().as_slice()).unwrap();
    let exp_id = info.cm_pols_map[2].exp_id.unwrap() as usize;
    let witness = |col_idx, row_offset| {
        Some(pb::Operand {
            operand: Some(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx, row_offset })),
        })
    };
    match &pilout.air_groups[0].airs[0].expressions[exp_id].operation {
        Some(expression::Operation::Sub(sub)) => assert_eq!(sub.lhs, witness(0, 1)),
        other => panic!("expression {exp_id} is {other:?}"),
    }

    // The degrees (pilfflonk/docs/protocol.md#degrees): N = 256; l1 and l2 opened at {0, 1} have
    // 256 + 2 + 1 coefficients, the im pol at {0} 256 + 1 + 1, the fixed columns 256; |O|_max = 2,
    // so Q has 1·256 + 2·2 + 1 = 261, and the extended domain 2^9.
    let s = |name: &str| name.to_string();
    assert_eq!(
        layout(&info),
        [
            (0, 0, s("Fibonacci.L1"), vec![0], 256),
            (0, 1, s("Fibonacci.LLAST"), vec![0], 256),
            (1, 0, s("l1"), vec![0, 1], 259),
            (1, 1, s("l2"), vec![0, 1], 259),
            (1, 2, s("Fibonacci.ImPol[0]"), vec![0], 258),
            (2, 3, s("Q0"), vec![0], 261),
        ]
    );
    assert_eq!(info.opening_points, [0, 1]);
    assert_eq!(info.ev_map.len(), 7);

    // With τ = 1: L1(1)·G = G and LLAST(1)·G, the point at infinity.
    assert_eq!(vkey.fixed_commitments.0, [g(), G1Affine::INFINITY]);
    assert_eq!((vkey.n_public, vkey.power, vkey.power_w, vkey.q_deg), (3, 8, 1, 1));

    // The SRS: the 261 powers of Q's f, the largest of the layout.
    let srs = fs::read(gi.srs_path(&proving_key)).unwrap();
    assert_eq!((&srs[..4], srs.len()), (&b"pfsr"[..], 12 + 3 * 12 + 88 + 261 * 64 + 2 * 128));

    // A second run writes the same bytes.
    let before = contents(&proving_key);
    let out = proofman_setup(&args);
    assert!(out.status.success(), "setup-pilfflonk: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(contents(&proving_key), before);

    // Grouped, the default, with `--extra-muls 2` and 0: the same pilfflonkinfo but for its layout
    // and evMap, and the same files.
    let unpacked = info;
    for (extra_muls, expected, power_w) in [
        // L1 and LLAST in one f of k = 2, 256·2 + 1; the im pol, opened at {0}, alone in its class
        // of stage 1, moves to {0, 1} and joins l1 and l2
        // (pilfflonk/docs/protocol.md#grouping-rules, rule 1): 256 + 2 + 1 each, and Q's bound does
        // not change (|O|_max was 2). The two extra muls split the group of three.
        (
            "2",
            vec![
                (0, vec!["Fibonacci.LLAST", "Fibonacci.L1"], 2, vec![0], 513),
                (1, vec!["Fibonacci.ImPol[0]"], 1, vec![0, 1], 259),
                (1, vec!["l2"], 1, vec![0, 1], 259),
                (1, vec!["l1"], 1, vec![0, 1], 259),
                (2, vec!["Q0"], 1, vec![0], 261),
            ],
            2,
        ),
        // With none, the three in one f of k = 3: 259·3 + 2.
        (
            "0",
            vec![
                (0, vec!["Fibonacci.LLAST", "Fibonacci.L1"], 2, vec![0], 513),
                (1, vec!["Fibonacci.ImPol[0]", "l2", "l1"], 3, vec![0, 1], 779),
                (2, vec!["Q0"], 1, vec![0], 261),
            ],
            6,
        ),
    ] {
        let grouped = dir.file(&format!("grouped_{extra_muls}"));
        let args = [
            "setup-pilfflonk",
            "-a",
            path(&pilout_path),
            "-b",
            path(&grouped),
            "--powers-of-tau",
            path(&ptau),
            "--extra-muls",
            extra_muls,
        ];
        let out = proofman_setup(&args);
        assert!(out.status.success(), "setup-pilfflonk: {}", String::from_utf8_lossy(&out.stderr));
        let proving_key = grouped.join("provingKey");
        let mut expected_files = proving_key_files("fibonacci", "Fibonacci", "Fibonacci");
        expected_files.sort();
        assert_eq!(files(&proving_key), expected_files);
        let (gi, info, vkey) = check_proving_key(&proving_key);
        let layout = f_shapes(&info);
        assert_eq!(layout, expected, "--extra-muls {extra_muls}");
        assert_eq!((vkey.power_w, info.layout.power_w().unwrap()), (power_w, power_w));
        // The evMap: the unpacked one, and then the im pol at 1, the pair its fusion adds.
        assert_eq!(info.ev_map[..7], unpacked.ev_map[..]);
        assert_eq!(
            info.ev_map[7..].iter().map(|e| (e.pol_type, e.id, e.prime, e.opening_pos)).collect::<Vec<_>>(),
            [(PolType::Cm, 2, 1, 1)]
        );
        let names = ProofNames::new(&gi, &[&info]).unwrap();
        assert_eq!(names.evaluations().last().map(String::as_str), Some("Fibonacci.ImPol[0]w"));
        // Everything else is the unpacked pilfflonkinfo's, and so are its degrees
        // (pilfflonk/docs/protocol.md#degrees).
        let without_layout =
            |i: &PilfflonkInfo| PilfflonkInfo { layout: Default::default(), ev_map: vec![], ..i.clone() };
        assert_eq!(without_layout(&info), without_layout(&unpacked));
        assert_eq!(info.degrees().unwrap(), unpacked.degrees().unwrap());
        assert_eq!(
            gi.setup_params,
            SetupParams {
                max_constraint_degree: 9,
                extra_muls: extra_muls.parse().unwrap(),
                max_q_degree: 0,
                packing: true
            }
        );
        // τ = 1: the fixed f of LLAST and L1 commits to (LLAST(1) + L1(1))·G = G.
        assert_eq!(vkey.fixed_commitments.0, [g()]);
        // The SRS holds the powers of the largest f.
        let largest = expected.iter().map(|f| f.4).max().unwrap();
        let srs = fs::read(gi.srs_path(&proving_key)).unwrap();
        assert_eq!(srs.len() as u64, 12 + 3 * 12 + 88 + largest * 64 + 2 * 128);
        // Deterministic.
        let before = contents(&proving_key);
        let out = proofman_setup(&args);
        assert!(out.status.success(), "setup-pilfflonk: {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(contents(&proving_key), before);
    }
}
