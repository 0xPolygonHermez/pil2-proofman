//! `proofman-cli pilfflonk prove` (spec §4.4, plan M18) end to end: the setup (`setup-pilfflonk`),
//! the prover's CLI and the JS verifier's (`pilfflonk verify`, M19), and the prover against the Rust
//! oracle (M14), on three fixtures and their layouts (plans M22, M23):
//!
//! - the Fibonacci, grouped with the default `--extra-muls 2` (its fixed columns in one `f` of
//!   `k = 2`, `powerW = 2`), with `--extra-muls 0` (its committed columns in one `f` of `k = 3`, the
//!   im pol fused into it, `powerW = 6`), and with `--no-packing` (`k = 1`, `powerW = 1`);
//! - `pilfflonk/tests/fixtures/packed`, a synthetic AIR whose default grouping packs six fixed
//!   columns in one `f`, splits its eleven committed columns in `f` of `k = 3, 4, 4` (`powerW =
//!   12`), and fuses a fixed column and two committed ones, which adds their evaluations at the
//!   offsets they gain to the end of the evMap;
//! - `pilfflonk/tests/fixtures/signed`, a synthetic AIR that reads its columns at the offsets
//!   `{−1, 0, 1, 2}` and has constraints of degree up to 6, with the im pols the setup chooses by
//!   default (one, `qDeg = 3`) and with `--max-constraint-degree 3` (three, `qDeg = 2`) and `2`
//!   (eight, `qDeg = 1`), grouped and with `--no-packing`;
//! - the synthetic pilouts of `pilfflonk/tests/data/domains.rs`, built in code (plan M24), whose
//!   constraints hold on `firstRow`, `lastRow` and `everyFrame` of several `{offsetMin,
//!   offsetMax}`: the prover's zerofiers on the coset, the JS verifier's at `ξ` and the oracle's
//!   agree, and a witness that breaks a constraint at the edge row of its domain is refused;
//! - `Q` split (plan M33): the fixture of the signed offsets, `qDeg = 3`, with `--max-q-degree 1`
//!   and `2` (three pieces and two), grouped and with `--no-packing`; and a `--max-q-degree` that
//!   does not split `Q`, which sets up the key of `Q` whole;
//! - stage 2 (plans M30, M31): `pilfflonk/tests/fixtures/{sum_bus,prod_bus,prod_bus_im}`, a lookup on
//!   the std's sum bus, with the std's default `MAX_CONSTRAINT_DEGREE` (an `im_col`) and with 4
//!   (none), and a permutation on its product bus, without `im_col` and split by selectors into two
//!   (two chained `im_col`), in `STD_MODE_ONE_INSTANCE`, grouped and with `--no-packing`: the prover
//!   computes their stage-2 columns from the hints `im_col`, then `gprod_col` and `gsum_col`, with
//!   the challenges of stage 2, which are the transcript's (A.4) and the JS verifier's; the columns
//!   are the oracle's; the verifier accepts the proofs and rejects every change to one; and a
//!   witness that breaks the bus is refused.
//!
//! The ptau is `PILFFLONK_TEST_PTAU` if it is set, and otherwise one this test writes with the
//! full-width `τ` of the C++ test helper (`pilfflonk_setup::test_ptau::fixed_tau_ptau`, plan N13):
//! not the ptau of `τ = 1`, under which the blinding vanishes from every commitment and any proof
//! verifies. With it, the verifier accepts the prover's proofs only if they are sound, and rejects
//! every change to one.
//!
//! Pilouts are not versioned: the test compiles the fixtures with the compiler `PIL2C_EXEC` names,
//! which must honour `prime`, and is `#[ignore]` without it. Those of the domains build their
//! pilouts in code, and need only Node.js (the E2E) or nothing (`qDeg`). It needs Node.js:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only \
//!     --test pilfflonk_prove -- --ignored --test-threads 2
//! ```
//!
//! Its tests call the C++ core in this process, each from its own thread, which OpenMP makes a root
//! with a team of one thread per CPU, kept while that thread lives. libomp 14 (Ubuntu 22.04's) can
//! crash with SIGSEGV once the teams outgrow its first table of threads (4 per CPU): it replaces the
//! table while the workers it has just started may still be reading the old one (plan M26; the lock of
//! `setup/pilfflonk/tests/setup/common.rs`, `cpp_core`, has the details). It has not happened in
//! these tests, but nothing rules it out: run them with `--test-threads 2`, as above (and CI,
//! plan M28), which keeps the teams alive, counting those of tests that are just ending, within
//! that table.

#[path = "../../pilfflonk/tests/data/domains.rs"]
mod domains;
#[path = "../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;
#[path = "../../pilfflonk/tests/data/packed.rs"]
mod packed;
#[path = "../../pilfflonk/tests/data/prod_bus.rs"]
mod prod_bus;
#[path = "../../pilfflonk/tests/data/prod_bus_im.rs"]
mod prod_bus_im;
#[path = "../../pilfflonk/tests/data/signed.rs"]
mod signed;
#[path = "../../pilfflonk/tests/data/sum_bus.rs"]
mod sum_bus;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pil2_pilout::pilout_proxy::PilOutProxy;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::digest::seal_vkey;
use pilfflonk_setup::keys::write_srs;
use pilfflonk_setup::layout::max_degree;
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman_pilfflonk::oracle::{AirOracle, ColumnRef, Domain, Fr, HintKind};
use proofman_pilfflonk::oracle::Values;
use proofman_pilfflonk::{
    prove, stage_columns, AirFile, Boundary, FileWitnessSource, FrBytes, JsonFile, PilfflonkError, PilfflonkGlobalInfo,
    PilfflonkInfo, PolMapEntry, PolType, ProofChallenges, ProveOptions, ProvingKey, Vkey, Witness, WitnessSource,
    BN254_R,
};
use prost::Message;
use serde_json::{json, Value};

const SEED_A: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
const SEED_B: &str = "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100";

/// `in1` of the synthetic fixture's witness.
const PACKED_IN1: u64 = 5;

/// `[in1, in2]` of the witness of the fixture of the signed offsets.
const SIGNED_INPUTS: [u64; 2] = [3, 5];

/// A fresh directory for the test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_prove_{name}_{}", std::process::id()));
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap()
}

/// What a command wrote: the CLI logs to stdout, the verifier to stderr.
fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// A fixture of the tests: its PIL and its witness generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Program {
    Fibonacci,
    Packed,
    Signed,
    /// A lookup on the std's sum bus (`tests/fixtures/sum_bus`, plan M30), with the std's default
    /// `MAX_CONSTRAINT_DEGREE`: an `im_col` (plan M31).
    SumBus,
    /// The same with the std's `MAX_CONSTRAINT_DEGREE` raised to 4 (`sum_bus_degree4.pil`): no
    /// `im_col`.
    SumBusDegree4,
    /// A permutation on the std's product bus (`tests/fixtures/prod_bus`, plan M30).
    ProdBus,
    /// A permutation on the product bus split by selectors, with two chained `im_col`
    /// (`tests/fixtures/prod_bus_im`, plan M31).
    ProdBusIm,
    /// A pilout of `tests/data/domains.rs`, built in code.
    Domains(domains::Air),
}

impl Program {
    /// M13's generator for the Fibonacci (inputs [1, 2]), `tests/data/{packed,signed}.rs` for the
    /// others.
    fn witness(self) -> Witness {
        match self {
            Program::Fibonacci => fibonacci::witness(8, [1, 2]),
            Program::Packed => packed::witness(PACKED_IN1),
            Program::Signed => signed::witness(SIGNED_INPUTS),
            Program::SumBus | Program::SumBusDegree4 => sum_bus::witness(),
            Program::ProdBus => prod_bus::witness(),
            Program::ProdBusIm => prod_bus_im::witness(),
            Program::Domains(air) => domains::witness(air),
        }
    }
}

/// Compiles `program` over BN254 to `pilout` with `PIL2C_EXEC`, or writes the pilout it builds.
fn compile(program: Program, pilout: &Path) {
    let pil = match program {
        Program::Fibonacci => "pilfflonk/tests/fixtures/fibonacci/fibonacci.pil",
        Program::Packed => "pilfflonk/tests/fixtures/packed/packed.pil",
        Program::Signed => "pilfflonk/tests/fixtures/signed/signed.pil",
        Program::SumBus => "pilfflonk/tests/fixtures/sum_bus/sum_bus.pil",
        Program::SumBusDegree4 => "pilfflonk/tests/fixtures/sum_bus/sum_bus_degree4.pil",
        Program::ProdBus => "pilfflonk/tests/fixtures/prod_bus/prod_bus.pil",
        Program::ProdBusIm => "pilfflonk/tests/fixtures/prod_bus_im/prod_bus_im.pil",
        Program::Domains(air) => {
            fs::write(pilout, domains::pilout(air).encode_to_vec()).unwrap();
            return;
        }
    };
    let compiler = std::env::var("PIL2C_EXEC").expect("PIL2C_EXEC must name a pil2com that honours `prime`");
    let out = Command::new(compiler)
        .current_dir(repo_root())
        .arg(pil)
        .args(["-I", "pil2-components/lib/std/pil", "-P", "pilfflonk/tests/fixtures/fibonacci/bn254.json", "-o"])
        .arg(pilout)
        .output()
        .expect("PIL2C_EXEC runs");
    assert!(out.status.success(), "pil2com: {}", output(&out));
}

fn cli(args: &[&str], paths: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proofman-cli")).args(args).args(paths).output().expect("proofman-cli runs")
}

/// `proofman-cli pilfflonk prove -k <key> --witness <witness> -o <out> [--insecure-blinding-seed <seed>]`.
fn prove_cli(key: &Path, witness: &Path, out: &Path, seed: Option<&str>) -> Output {
    let mut args = vec!["pilfflonk", "prove", "-k", key.to_str().unwrap(), "--witness", witness.to_str().unwrap()];
    args.extend(["-o", out.to_str().unwrap()]);
    if let Some(seed) = seed {
        args.extend(["--insecure-blinding-seed", seed]);
    }
    cli(&args, &[])
}

/// `proofman-cli pilfflonk verify <vkey> <publics> <proof>`.
fn verify(vkey: &Path, publics: &Path, proof: &Path) -> Output {
    cli(&["pilfflonk", "verify"], &[vkey, publics, proof])
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_string_pretty(value).unwrap()).unwrap();
}

/// `value + 1 mod r`, in decimal.
fn plus_one(value: &Value) -> Value {
    let r = num_bigint::BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap();
    let v = num_bigint::BigUint::parse_bytes(value.as_str().unwrap().as_bytes(), 10).unwrap();
    json!(((v + 1u32) % r).to_string())
}

/// The ptau of the test (see the module), with more powers than the largest degree of every
/// layout here: 779, the Fibonacci's `f` of `k = 3` (`setup/pil2-stark/tests/setup_pilfflonk.rs`).
fn ptau(dir: &TestDir) -> PathBuf {
    match std::env::var_os("PILFFLONK_TEST_PTAU") {
        Some(path) => PathBuf::from(path),
        None => {
            let path = dir.file("fixed_tau.ptau");
            write_fixed_tau_ptau(&path, 1024, &test_tau()).unwrap();
            path
        }
    }
}

/// How the setup lays out the `f`: grouped with `--extra-muls`, or `--no-packing`.
#[derive(Clone, Copy, Debug)]
enum Packing {
    ExtraMuls(u64),
    NoPacking,
}

/// The default of the command: grouped with `--extra-muls 2`.
const DEFAULT: Packing = Packing::ExtraMuls(DEFAULT_EXTRA_MULS);

struct Fixture {
    dir: TestDir,
    pilout: PathBuf,
    proving_key: PathBuf,
    vkey: PathBuf,
    witness: PathBuf,
    /// The publics of the witness, as `publics.json` has them.
    publics: Value,
}

impl Fixture {
    fn info(&self) -> PilfflonkInfo {
        let global_info = PilfflonkGlobalInfo::from_proving_key(&self.proving_key).unwrap();
        PilfflonkInfo::read(&global_info.air_file(&self.proving_key, 0, 0, AirFile::PilfflonkInfo).unwrap()).unwrap()
    }

    /// Each `f` of the layout as `(stage, k, offsets, degree)`.
    fn layout(&self) -> Vec<(u64, u64, Vec<i64>, u64)> {
        self.info().layout.0.iter().map(|f| (f.stage, f.k, f.offsets.clone(), f.degree)).collect()
    }
}

/// The options of a setup in `dir`, of the pilout `program.pilout` there and with the test's ptau,
/// laid out as `packing` says.
fn setup_options(dir: &TestDir, packing: Packing) -> SetupPilfflonkOptions {
    let (extra_muls, no_packing) = match packing {
        Packing::ExtraMuls(extra_muls) => (extra_muls, false),
        Packing::NoPacking => (DEFAULT_EXTRA_MULS, true),
    };
    SetupPilfflonkOptions {
        airout_path: dir.file("program.pilout"),
        build_dir: dir.file("build"),
        powers_of_tau: ptau(dir),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing,
    }
}

/// `program` compiled, set up as `packing` says, and its witness written.
fn fixture(name: &str, program: Program, packing: Packing) -> Fixture {
    fixture_of_degree(name, program, packing, DEFAULT_MAX_CONSTRAINT_DEGREE)
}

/// [`fixture`], set up with `--max-constraint-degree max_constraint_degree`.
fn fixture_of_degree(name: &str, program: Program, packing: Packing, max_constraint_degree: u64) -> Fixture {
    fixture_split(name, program, packing, max_constraint_degree, DEFAULT_MAX_Q_DEGREE)
}

/// [`fixture_of_degree`], set up with `--max-q-degree max_q_degree` too.
fn fixture_split(
    name: &str,
    program: Program,
    packing: Packing,
    max_constraint_degree: u64,
    max_q_degree: u64,
) -> Fixture {
    let dir = TestDir::new(name);
    let opts = SetupPilfflonkOptions { max_constraint_degree, max_q_degree, ..setup_options(&dir, packing) };
    compile(program, &opts.airout_path);
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let vkey = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap().vkey_path(&proving_key);
    let witness = dir.file("witness");
    let shape = ProvingKey::load(&proving_key).unwrap().witness_shape().unwrap();
    let generated = program.witness();
    generated.write(&witness, &shape).unwrap();
    let publics = json!(generated.publics.iter().map(FrBytes::to_decimal).collect::<Vec<_>>());
    Fixture { pilout: opts.airout_path, dir, proving_key, vkey, witness, publics }
}

/// The keys of `map`, sorted: serde_json keeps the order of the file instead when its
/// `preserve_order` is on, which Cargo's feature unification can turn on for the whole build.
fn sorted_keys(map: &serde_json::Map<String, Value>) -> Vec<&str> {
    let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys
}

/// Proves the witness of `f` four times (twice with one seed, once with another, once with the
/// OS's randomness), and checks that:
/// - the same seed gives the same proof, and another seed other commitments;
/// - the proof holds the commitments `commitments` (the non-fixed `f`, `W` and `W'`) and the
///   evaluations `evaluations`, by name, and the witness's publics;
/// - the verifier accepts every proof, and rejects any change to a commitment, `W`, `W'`, an
///   evaluation, `inv`, `invZh` or a public, and the evaluations of another proof.
fn proves_and_rejects_every_change(f: &Fixture, commitments: &[&str], evaluations: &[&str]) {
    let (a1, a2, b, random) = (f.dir.file("a1"), f.dir.file("a2"), f.dir.file("b"), f.dir.file("random"));
    for (out, seed) in [(&a1, Some(SEED_A)), (&a2, Some(SEED_A)), (&b, Some(SEED_B)), (&random, None)] {
        let run = prove_cli(&f.proving_key, &f.witness, out, seed);
        assert!(run.status.success(), "prove: {}", output(&run));
        if seed.is_some() {
            assert!(output(&run).contains("--insecure-blinding-seed: the blinding is fixed"), "{}", output(&run));
        }
    }

    // The same seed, the same proof, byte for byte; another seed, other commitments.
    let proof = |dir: &Path| fs::read(dir.join("proof.json")).unwrap();
    assert_eq!(proof(&a1), proof(&a2));
    assert_eq!(fs::read(a1.join("publics.json")).unwrap(), fs::read(b.join("publics.json")).unwrap());
    let (pa, pb, pr) =
        (read_json(&a1.join("proof.json")), read_json(&b.join("proof.json")), read_json(&random.join("proof.json")));
    let polynomials = pa["polynomials"].as_object().unwrap();
    assert_eq!(sorted_keys(polynomials), commitments);
    for name in polynomials.keys() {
        assert_ne!(pa["polynomials"][name], pb["polynomials"][name], "{name} with another seed");
        assert_ne!(pa["polynomials"][name], pr["polynomials"][name], "{name} with the OS's randomness");
    }
    let names = pa["evaluations"].as_object().unwrap();
    assert_eq!(sorted_keys(names), evaluations);
    assert_eq!(read_json(&a1.join("publics.json")), f.publics);

    // Every one verifies.
    for dir in [&a1, &b, &random] {
        let out = verify(&f.vkey, &dir.join("publics.json"), &dir.join("proof.json"));
        assert!(out.status.success(), "{}", output(&out));
        assert!(output(&out).contains("OK: the proof verifies"), "{}", output(&out));
    }

    // Any change to a commitment, W, W', an evaluation, inv, invZh or a public is rejected.
    let publics = a1.join("publics.json");
    let other = f.dir.file("tampered.json");
    let rejected = |publics: &Path, proof: &Path, what: &str| {
        let out = verify(&f.vkey, publics, proof);
        assert!(!out.status.success(), "{what}: {}", output(&out));
        assert!(output(&out).contains("INVALID: the proof does not verify"), "{what}: {}", output(&out));
    };
    for name in polynomials.keys() {
        let mut tampered = pa.clone();
        // Another point of the curve: the same one as another commitment of the proof.
        tampered["polynomials"][name] = pb["polynomials"][name].clone();
        write_json(&other, &tampered);
        rejected(&publics, &other, name);
    }
    for name in names.keys() {
        let mut tampered = pa.clone();
        tampered["evaluations"][name] = plus_one(&pa["evaluations"][name]);
        write_json(&other, &tampered);
        rejected(&publics, &other, name);
    }
    let public_values = read_json(&publics);
    for i in 0..public_values.as_array().unwrap().len() {
        let mut tampered = public_values.clone();
        tampered[i] = plus_one(&public_values[i]);
        write_json(&other, &tampered);
        rejected(&other, &a1.join("proof.json"), &format!("publics[{i}]"));
    }
    // The proof of one seed with the evaluations of another.
    let mut mixed = pa.clone();
    mixed["evaluations"] = pb["evaluations"].clone();
    write_json(&other, &mixed);
    rejected(&publics, &other, "the evaluations of another proof");
}

/// The Fibonacci grouped by default (plan M22): `L1` and `LLAST` in `f0`, of `k = 2`, the vkey's;
/// the im pol, fused to `{0, 1}`, `l2` and `l1` in `f1` to `f3`, and `Q` in `f4`. The im pol's
/// evaluation at `ξ·ω`, the pair its fusion adds, is in the proof.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_the_grouped_fibonacci_and_the_verifier_rejects_every_change() {
    let f = fixture("e2e", Program::Fibonacci, DEFAULT);
    assert_eq!(
        f.layout(),
        [
            (0, 2, vec![0], 513),
            (1, 1, vec![0, 1], 259),
            (1, 1, vec![0, 1], 259),
            (1, 1, vec![0, 1], 259),
            (2, 1, vec![0], 261)
        ]
    );
    assert_eq!(f.info().layout.power_w().unwrap(), 2);
    proves_and_rejects_every_change(
        &f,
        &["W", "Wp", "f1", "f2", "f3", "f4"],
        &[
            "Fibonacci.ImPol[0]",
            "Fibonacci.ImPol[0]w",
            "Fibonacci.L1",
            "Fibonacci.LLAST",
            "inv",
            "invZh",
            "l1",
            "l1w",
            "l2",
            "l2w",
        ],
    );
}

/// The Fibonacci with `--extra-muls 0`: its three committed columns in one `f` of `k = 3`, opened at
/// the six roots of `ξ·ω^0` and `ξ·ω^1`, and `powerW = 6`.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_the_fibonacci_with_its_columns_in_one_f() {
    let f = fixture("e2e_k3", Program::Fibonacci, Packing::ExtraMuls(0));
    assert_eq!(f.layout(), [(0, 2, vec![0], 513), (1, 3, vec![0, 1], 779), (2, 1, vec![0], 261)]);
    assert_eq!(f.info().layout.power_w().unwrap(), 6);
    proves_and_rejects_every_change(
        &f,
        &["W", "Wp", "f1", "f2"],
        &[
            "Fibonacci.ImPol[0]",
            "Fibonacci.ImPol[0]w",
            "Fibonacci.L1",
            "Fibonacci.LLAST",
            "inv",
            "invZh",
            "l1",
            "l1w",
            "l2",
            "l2w",
        ],
    );
}

/// The Fibonacci with `--no-packing` (plan R1, the first slice's layout): an `f` of `k = 1` per
/// column, `L1` and `LLAST` in `f0` and `f1`.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_the_unpacked_fibonacci_and_the_verifier_rejects_every_change() {
    let f = fixture("e2e_unpacked", Program::Fibonacci, Packing::NoPacking);
    assert!(f.layout().iter().all(|(_, k, _, _)| *k == 1));
    assert_eq!(f.info().layout.power_w().unwrap(), 1);
    proves_and_rejects_every_change(
        &f,
        &["W", "Wp", "f2", "f3", "f4", "f5"],
        &["Fibonacci.ImPol[0]", "Fibonacci.L1", "Fibonacci.LLAST", "inv", "invZh", "l1", "l1w", "l2", "l2w"],
    );
}

/// The synthetic fixture (`tests/fixtures/packed`) grouped by default: six fixed columns in one `f`
/// of `k = 6`; `S`, opened at `{1}`, fused to `{0, 1}`; the eleven committed columns (`b` and the im
/// pol fused to `{0, 1}`) split in three `f` of `k = 3, 4, 4`; `powerW = 12`. The evMap ends with
/// the pairs the fusions add, `S` at 0, `b` and the im pol at 1, and the proof has them. With no
/// extra mul the eleven have no valid split, and the setup says to raise `--extra-muls`.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_a_layout_that_packs_and_splits_groups() {
    let f = fixture("e2e_packed", Program::Packed, DEFAULT);
    let info = f.info();
    assert_eq!(
        f.layout(),
        [
            (0, 6, vec![0], 6 * 16 + 5),
            (0, 1, vec![0, 1], 16),
            (1, 3, vec![0, 1], 3 * 19 + 2),
            (1, 4, vec![0, 1], 4 * 19 + 3),
            (1, 4, vec![0, 1], 4 * 19 + 3),
            (2, 1, vec![0], 16 + 2 * 2 + 1),
        ]
    );
    assert_eq!(info.layout.power_w().unwrap(), 12);
    let appended: Vec<(PolType, &str, i64)> = info.ev_map[info.ev_map.len() - 3..]
        .iter()
        .map(|e| (e.pol_type, info.pol(e.pol_type, e.id).unwrap().name.as_str(), e.prime))
        .collect();
    assert_eq!(appended, [(PolType::Const, "Packed.S", 0), (PolType::Cm, "b", 1), (PolType::Cm, "Packed.ImPol", 1)]);

    let mut evaluations: Vec<String> =
        ["Packed.L1", "Packed.LLAST", "Packed.S", "Packed.Sw", "inv", "invZh"].iter().map(|s| s.to_string()).collect();
    evaluations.extend((0..4).map(|i| format!("Packed.K[{i}]")));
    evaluations.extend((0..8).flat_map(|i| [format!("a[{i}]"), format!("a[{i}]w")]));
    evaluations.extend(["b", "bw", "c", "cw", "Packed.ImPol[0]", "Packed.ImPol[0]w"].map(String::from));
    evaluations.sort();
    let evaluations: Vec<&str> = evaluations.iter().map(String::as_str).collect();
    proves_and_rejects_every_change(&f, &["W", "Wp", "f2", "f3", "f4", "f5"], &evaluations);

    // No valid split of the eleven without an extra mul (A.2): the setup refuses, and says why.
    let dir = TestDir::new("packed_no_extra_muls");
    let opts = SetupPilfflonkOptions { airout_path: f.pilout.clone(), ..setup_options(&dir, Packing::ExtraMuls(0)) };
    let err = format!("{:#}", run_setup_pilfflonk(&opts).unwrap_err());
    // 19 polynomials in 4 groups: 15 extra muls at most.
    assert!(err.contains("a larger --extra-muls, up to 15, allows smaller chunks"), "{err}");
    assert!(!opts.build_dir.exists());
}

/// The number of im pols the setup chose.
fn n_im_pols(info: &PilfflonkInfo) -> usize {
    info.cm_pols_map.iter().filter(|p| p.im_pol).count()
}

/// The names of the evaluations of a proof of `info`, sorted: the evMap's `(column, offset)`, named
/// as spec A.6 says (`<column>` and `[i]` per entry of its lengths, then `""` for `ξ`, `w` for
/// `ξ·ω` and `w<s>` for `ξ·ω^s`), the pieces `Q0 … Q<m−1>` if `Q` is split, and `inv` and `invZh`.
fn evaluation_names(info: &PilfflonkInfo) -> Vec<String> {
    let m = info.q_split().unwrap().n_pieces();
    let pieces = (0..m).filter(|_| m > 1).map(|i| format!("Q{i}"));
    let mut names: Vec<String> = info
        .ev_map
        .iter()
        .map(|e| {
            let pol = info.pol(e.pol_type, e.id).unwrap();
            let indices: String = pol.lengths.iter().map(|i| format!("[{i}]")).collect();
            let suffix = match e.prime {
                0 => String::new(),
                1 => "w".to_string(),
                s => format!("w{s}"),
            };
            format!("{}{indices}{suffix}", pol.name)
        })
        .chain(pieces)
        .chain(["inv", "invZh"].map(String::from))
        .collect();
    names.sort();
    names
}

/// The names of the commitments of a proof of `info`, sorted: `W`, `Wp` and `f<g>` for each `f` not
/// of the fixed columns (A.5, A.6).
fn commitment_names(info: &PilfflonkInfo) -> Vec<String> {
    let fs = info.layout.0.iter().enumerate().filter(|(_, f)| f.stage > 0).map(|(g, _)| format!("f{g}"));
    let mut names: Vec<String> = ["W", "Wp"].map(String::from).into_iter().chain(fs).collect();
    names.sort();
    names
}

/// [`proves_and_rejects_every_change`] with the commitments and evaluations of the layout.
fn proves_its_layout_and_rejects_every_change(f: &Fixture) {
    let info = f.info();
    let (commitments, evaluations) = (commitment_names(&info), evaluation_names(&info));
    let commitments: Vec<&str> = commitments.iter().map(String::as_str).collect();
    let evaluations: Vec<&str> = evaluations.iter().map(String::as_str).collect();
    proves_and_rejects_every_change(f, &commitments, &evaluations);
}

/// The offsets the signed fixture reads its columns at.
const SIGNED_OFFSETS: [i64; 4] = [-1, 0, 1, 2];

/// The fixture of the signed offsets (`tests/fixtures/signed`, plan M23), grouped by default and
/// with the im pol the setup chooses by default: one, `'a·a·a'·a'2`, which brings the constraint of
/// degree 6 down to the 4 of the next one, and `qDeg = 3`. `K`, read at −1 only, and `P`, at 2
/// only, are fused to `{−1, 0, 2}` in an `f` of `k = 2`; `L1`, `LLAST` and `WIN` are in one of
/// `k = 3`. Every committed column of stage 1, the im pol too, is fused to `{−1, 0, 1, 2}` (the im
/// pol, which the prover computes on `H` from rows that wrap around, is opened at `ξ·ω^−1`, `ξ·ω`
/// and `ξ·ω^2` as well), and they go in `f` of `k = 1, 3, 3`: `powerW = 6`. The evMap interleaves
/// fixed and committed columns, and ends with the pairs the fusions add, of both. `Q` has
/// `qDeg·N + (qDeg + 1)·|O|max + 1` coefficients (A.1).
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_signed_offsets_with_the_im_pol_the_setup_chooses() {
    let f = fixture("e2e_signed", Program::Signed, DEFAULT);
    let info = f.info();
    assert_eq!((n_im_pols(&info), info.q_deg), (1, 3));
    assert_eq!(info.opening_points, SIGNED_OFFSETS);
    let (n, blinded) = (32, 32 + 4 + 1);
    let all = SIGNED_OFFSETS.to_vec();
    assert_eq!(
        f.layout(),
        [
            (0, 2, vec![-1, 0, 2], 2 * n + 1),
            (0, 3, vec![0], 3 * n + 2),
            (1, 1, all.clone(), blinded),
            (1, 3, all.clone(), 3 * blinded + 2),
            (1, 3, all, 3 * blinded + 2),
            (2, 1, vec![0], 3 * n + 4 * 4 + 1),
        ]
    );
    assert_eq!(info.layout.power_w().unwrap(), 6);

    // Every committed column at the four points, K and P at three.
    let mut evaluations: Vec<String> =
        ["Signed.L1", "Signed.LLAST", "Signed.WIN", "inv", "invZh"].map(String::from).to_vec();
    for column in ["Signed.K", "Signed.P"] {
        evaluations.extend(["w-1", "", "w2"].map(|suffix| format!("{column}{suffix}")));
    }
    for column in ["a", "b", "c", "d", "e", "g", "Signed.ImPol[0]"] {
        evaluations.extend(["w-1", "", "w", "w2"].map(|suffix| format!("{column}{suffix}")));
    }
    evaluations.sort();
    assert_eq!(evaluation_names(&info), evaluations);
    assert_eq!(commitment_names(&info), ["W", "Wp", "f2", "f3", "f4", "f5"]);
    proves_its_layout_and_rejects_every_change(&f);
}

/// The fixture of the signed offsets with a lower `--max-constraint-degree`, which forces more im
/// pols and a lower `qDeg`: 3 im pols and `qDeg = 2` with 3, 8 and `qDeg = 1` with 2. Between
/// them the im pols read `a` at every offset of `{−1, 0, 1, 2}` (and with 2, `K` at −1), and they
/// are opened at `{0}`, in `f` of their own and `d`'s.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_signed_offsets_with_more_im_pols_under_a_lower_max_constraint_degree() {
    for (name, degree, im_pols, q_deg, power_w) in [("e2e_signed_d3", 3, 3, 2, 6), ("e2e_signed_d2", 2, 8, 1, 6)] {
        let f = fixture_of_degree(name, Program::Signed, DEFAULT, degree);
        let info = f.info();
        assert_eq!((n_im_pols(&info), info.q_deg), (im_pols, q_deg), "{name}");
        assert_eq!(info.layout.power_w().unwrap(), power_w, "{name}");
        let im_pol_offsets: Vec<&[i64]> = info
            .layout
            .0
            .iter()
            .filter(|f| f.pols.iter().any(|p| info.cm_pols_map[p.id as usize].im_pol))
            .map(|f| f.offsets.as_slice())
            .collect();
        assert!(!im_pol_offsets.is_empty() && im_pol_offsets.iter().all(|o| *o == [0]), "{name}: {im_pol_offsets:?}");
        proves_its_layout_and_rejects_every_change(&f);
    }
}

/// The fixture of the signed offsets with `--no-packing`: an `f` of `k = 1` per column, each at the
/// offsets its column is read at (no fusion), by default and with `--max-constraint-degree 2`. `K`,
/// `P` and `g` are opened at a single point, not `ξ`: `ξ·ω^−1`, `ξ·ω^2` and `ξ·ω^2`.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_signed_offsets_unpacked() {
    for (name, degree, im_pols) in [("e2e_signed_unpacked", 9, 1), ("e2e_signed_unpacked_d2", 2, 8)] {
        let f = fixture_of_degree(name, Program::Signed, Packing::NoPacking, degree);
        let info = f.info();
        assert_eq!(n_im_pols(&info), im_pols, "{name}");
        assert!(f.layout().iter().all(|(_, k, _, _)| *k == 1), "{name}");
        let offsets = |column: &str| {
            let f = info.layout.0.iter().find(|f| f.pols[0].name == column).unwrap();
            f.offsets.clone()
        };
        assert_eq!(
            ["a", "b", "c", "d", "e", "g", "Signed.K", "Signed.P"].map(offsets),
            [SIGNED_OFFSETS.to_vec(), vec![0, 2], vec![0, 1], vec![0], vec![0, 1], vec![2], vec![-1], vec![2]],
            "{name}"
        );
        proves_its_layout_and_rejects_every_change(&f);
    }
}

/// Each `f` of `Q`'s stage as `(its pieces, k, degree)`.
fn q_layout(info: &PilfflonkInfo) -> Vec<(Vec<&str>, u64, u64)> {
    let fs = info.layout.0.iter().filter(|f| f.stage == info.q_stage());
    fs.map(|f| (f.pols.iter().map(|p| p.name.as_str()).collect(), f.k, f.degree)).collect()
}

/// `Q` split (spec A.1, A.3; plan M33): the fixture of the signed offsets, `qDeg = 3` on `N = 32`
/// with `|O|_max = 4` (`Q` of `3·32 + 4·4 + 1 = 113` coefficients), with `--max-q-degree 1`, three
/// pieces of `32 + 2`, `32 + 2` and `113 − 64 = 49` coefficients, and `2`, two of `64 + 2` and `49`.
/// Grouped by default, the pieces are one `f` (their group is not split: the extra muls go to the
/// committed columns, as without the split), `Q2, Q1, Q0` of `k = 3` and `Q1, Q0` of `k = 2`, of
/// A.2's cost; unpacked, an `f` each. The proof holds a commitment per `f` and each `Q_i(ξ)`, after
/// the other evaluations; the verifier accepts it, checks that the pieces add up to `Q(ξ)` and
/// rejects any change, to a piece's commitment or evaluation too; and the pieces add up to the
/// oracle's `Q(ξ)`.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_a_split_q_and_the_verifier_rejects_every_change() {
    let m1_pieces = vec![34, 34, 49];
    let m2_pieces = vec![66, 49];
    for (name, max_q_degree, packing, pieces, q_fs) in [
        ("e2e_split_m1", 1, DEFAULT, m1_pieces.clone(), vec![(vec!["Q2", "Q1", "Q0"], 3, 49 * 3)]),
        (
            "e2e_split_m1_unpacked",
            1,
            Packing::NoPacking,
            m1_pieces,
            vec![(vec!["Q0"], 1, 34), (vec!["Q1"], 1, 34), (vec!["Q2"], 1, 49)],
        ),
        ("e2e_split_m2", 2, DEFAULT, m2_pieces.clone(), vec![(vec!["Q1", "Q0"], 2, 66 * 2 + 1)]),
        ("e2e_split_m2_unpacked", 2, Packing::NoPacking, m2_pieces, vec![(vec!["Q0"], 1, 66), (vec!["Q1"], 1, 49)]),
    ] {
        let f = fixture_split(name, Program::Signed, packing, DEFAULT_MAX_CONSTRAINT_DEGREE, max_q_degree);
        let info = f.info();
        assert_eq!((info.q_deg, info.max_q_degree), (3, max_q_degree), "{name}");
        let split = info.q_split().unwrap();
        let n_pieces = split.n_pieces();
        assert_eq!((split.stride, split.coefficients), (max_q_degree * 32, pieces), "{name}");
        assert_eq!(q_layout(&info), q_fs, "{name}");
        let vkey = Vkey::read(&f.vkey).unwrap();
        assert_eq!((vkey.q_deg, vkey.max_q_degree, &vkey.layout), (3, max_q_degree, &info.layout), "{name}");
        proves_its_layout_and_rejects_every_change(&f);
        agrees_with_the_oracle(&f);

        // The verifier checks that the pieces add up to Q(ξ) (A.1) before the opening: any Q_i(ξ) off
        // by one fails there.
        let proof = f.dir.file("pieces");
        let run = prove_cli(&f.proving_key, &f.witness, &proof, Some(SEED_A));
        assert!(run.status.success(), "{name}: prove: {}", output(&run));
        let tampered = f.dir.file("pieces.json");
        for i in 0..n_pieces {
            let mut json = read_json(&proof.join("proof.json"));
            let piece = format!("Q{i}");
            json["evaluations"][&piece] = plus_one(&json["evaluations"][&piece]);
            write_json(&tampered, &json);
            let out = verify(&f.vkey, &proof.join("publics.json"), &tampered);
            assert!(!out.status.success(), "{name}: {piece}");
            assert!(output(&out).contains("The pieces of Q do not add up to Q(ξ) (A.1)"), "{name}: {}", output(&out));
        }
    }
}

/// A `--max-q-degree` that does not split `Q` (`qDeg ≤ maxQDegree`, A.1): the fixture of the signed
/// offsets with `--max-constraint-degree 3` (`qDeg = 2`) and `--max-q-degree 2`, and the Fibonacci
/// (`qDeg = 1`) with `--max-q-degree 1`, grouped and unpacked, write the key of `Q` whole, byte for
/// byte, with `maxQDegree = 0`, but for the globalInfo, which records the option; and it proves and
/// verifies.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn a_max_q_degree_that_does_not_split_q_sets_up_q_whole() {
    for (name, program, degree, max_q_degree, packing) in [
        ("whole_signed_d3", Program::Signed, 3, 2, DEFAULT),
        ("whole_fibonacci", Program::Fibonacci, DEFAULT_MAX_CONSTRAINT_DEGREE, 1, DEFAULT),
        ("whole_fibonacci_unpacked", Program::Fibonacci, DEFAULT_MAX_CONSTRAINT_DEGREE, 1, Packing::NoPacking),
    ] {
        let whole = fixture_split(&format!("{name}_0"), program, packing, degree, 0);
        let f = fixture_split(name, program, packing, degree, max_q_degree);
        let info = f.info();
        assert_eq!((info.max_q_degree, info.q_split().unwrap().n_pieces()), (0, 1), "{name}");
        assert!(info.q_deg <= max_q_degree, "{name}");
        let files = |root: &Path| {
            let mut out = BTreeMap::new();
            let mut dirs = vec![root.to_path_buf()];
            while let Some(dir) = dirs.pop() {
                for entry in fs::read_dir(&dir).unwrap() {
                    let path = entry.unwrap().path();
                    if path.is_dir() {
                        dirs.push(path);
                    } else {
                        out.insert(path.strip_prefix(root).unwrap().to_path_buf(), fs::read(&path).unwrap());
                    }
                }
            }
            out
        };
        let (mut a, mut b) = (files(&whole.proving_key), files(&f.proving_key));
        let global_info = Path::new("pilout.globalInfo.json");
        let (ga, gb) = (a.remove(global_info).unwrap(), b.remove(global_info).unwrap());
        assert_eq!(a, b, "{name}: the key of Q whole");
        let setup_params = |bytes: &[u8]| serde_json::from_slice::<PilfflonkGlobalInfo>(bytes).unwrap().setup_params;
        assert_eq!((setup_params(&ga).max_q_degree, setup_params(&gb).max_q_degree), (0, max_q_degree), "{name}");

        let out = f.dir.file("proof");
        let run = prove_cli(&f.proving_key, &f.witness, &out, Some(SEED_A));
        assert!(run.status.success(), "{name}: prove: {}", output(&run));
        let verified = verify(&f.vkey, &out.join("publics.json"), &out.join("proof.json"));
        assert!(verified.status.success(), "{name}: {}", output(&verified));
    }
}

/// The witness of the fixture of the signed offsets that breaks `L1·(c − 'a)` (constraint 9) at row
/// 0 only, which reads `a` at row N − 1 across the wrap: the oracle finds that row alone, and the
/// prover refuses it, whatever the layout.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn a_witness_that_breaks_a_constraint_across_the_wrap_is_refused() {
    let witness = signed::witness_broken_across_the_wrap(SIGNED_INPUTS);
    for (name, degree, packing) in [("wrap", 9, DEFAULT), ("wrap_d2_unpacked", 2, Packing::NoPacking)] {
        let f = fixture_of_degree(name, Program::Signed, packing, degree);
        let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
        let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
        let failures = oracle.check(&oracle.values(&witness, 0).unwrap()).unwrap();
        let rows: Vec<(usize, usize)> = failures.iter().map(|x| (x.constraint, x.row)).collect();
        assert_eq!(rows, [(9, 0)], "{name}");

        let pk = ProvingKey::load(&f.proving_key).unwrap();
        let options = ProveOptions { insecure_blinding_seed: Some([3; 32]) };
        match prove(&pk, &witness, &options) {
            Err(PilfflonkError::Unsatisfied(message)) => {
                assert!(message.contains("the witness does not satisfy the constraints of Signed"), "{message}")
            }
            other => panic!("{name}: expected Unsatisfied, got {:?}", other.map(|_| ())),
        }
    }
}

/// The setups of each domain AIR the tests go through: `(name, --max-constraint-degree, packing,
/// im pols, qDeg)`. By default the domain constraints, of degree 2, have degree 3 with their `Zi`
/// (δ = 1, A.1), and the search keeps them, `qDeg = 2`; with 2, each gets an im pol, `qDeg = 1`.
fn domain_setups(air: domains::Air) -> Vec<(String, u64, Packing, usize, u64)> {
    let n_rules = air.rules().len();
    [
        ("", 9, DEFAULT, 0, 2),
        ("_unpacked", 9, Packing::NoPacking, 0, 2),
        ("_d2", 2, DEFAULT, n_rules, 1),
        ("_d2_unpacked", 2, Packing::NoPacking, n_rules, 1),
    ]
    .into_iter()
    .map(|(suffix, degree, packing, im_pols, q_deg)| {
        (format!("{}{suffix}", air.name()), degree, packing, im_pols, q_deg)
    })
    .collect()
}

/// Prints, for each point of its arguments, `Zi` of each boundary at it as the JS verifier's
/// `computeZi` computes it, in decimal.
const COMPUTE_ZI: &str = r#"
import { pathToFileURL } from "node:url";

const [js, nBits, boundaries, ...points] = process.argv.slice(1);
const { newCurve } = await import(pathToFileURL(`${js}/test/support.js`).href);
const { computeZi } = await import(pathToFileURL(`${js}/src/qverifier.js`).href);
const curve = await newCurve();
const zi = (point) => computeZi(curve, JSON.parse(boundaries), Number(nBits), curve.Fr.e(BigInt(point)));
console.log(JSON.stringify(points.map((p) => zi(p).map((z) => curve.Fr.toString(z, 10)))));
"#;

/// `Zi` of each boundary of `info` at each of `points` (spec A.1, A.6): `1/Z_H` for `everyRow` and
/// `Z_H/Z_D` for the others, as the oracle computes `Z_D` (its closed forms, which its own tests
/// check against the products over the rows) and as the JS verifier does (`computeZi`). The C++
/// prover's are on the coset, and the proofs check them: `Q` is a polynomial only if they are
/// `Z_H/Z_D` there, and `Q(ξ)` is the oracle's.
fn zerofiers_agree(info: &PilfflonkInfo, points: &[Fr]) {
    let domain = |b: &Boundary| match *b {
        Boundary::EveryRow => Domain::EveryRow,
        Boundary::FirstRow => Domain::FirstRow,
        Boundary::LastRow => Domain::LastRow,
        Boundary::EveryFrame { offset_min, offset_max } => {
            Domain::EveryFrame { offset_min: offset_min as usize, offset_max: offset_max as usize }
        }
    };
    let n_bits = info.n_bits as u32;
    let oracle: Vec<Vec<String>> = points
        .iter()
        .map(|z| {
            let z_h = Domain::EveryRow.zerofier_at(z, n_bits).unwrap();
            info.boundaries
                .iter()
                .map(|b| {
                    let zi = match b {
                        Boundary::EveryRow => z_h.inv().unwrap(),
                        _ => &z_h * &domain(b).zerofier_at(z, n_bits).unwrap().inv().unwrap(),
                    };
                    zi.as_biguint().to_string()
                })
                .collect()
        })
        .collect();
    let boundaries = serde_json::to_string(&info.boundaries).unwrap();
    let out = Command::new("node")
        .args(["--input-type=module", "-e", COMPUTE_ZI])
        .arg(repo_root().join("pilfflonk/js"))
        .args([info.n_bits.to_string(), boundaries])
        .args(points.iter().map(|z| z.as_biguint().to_string()))
        .output()
        .expect("node runs");
    assert!(out.status.success(), "computeZi: {}", output(&out));
    let js: Vec<Vec<String>> = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(js, oracle, "{}: the JS verifier's Zi and the oracle's", info.name);
}

/// The E2E of a domain AIR (plan M24), for each of its [`domain_setups`]: its boundaries are
/// `boundaries`; the prover proves its witness, the verifier accepts the proofs and rejects every
/// change to one; the prover agrees with the oracle at `ξ`; the JS verifier's `Zi` are the
/// oracle's at `ξ` and at other points; and the witness that breaks one rule at the first row of
/// its domain or at the last, and nowhere else, is refused.
fn proves_a_domain(air: domains::Air, boundaries: &[Boundary]) {
    for (name, degree, packing, im_pols, q_deg) in domain_setups(air) {
        let f = fixture_of_degree(&name, Program::Domains(air), packing, degree);
        let info = f.info();
        assert_eq!(info.boundaries, boundaries, "{name}");
        assert_eq!((n_im_pols(&info), info.q_deg), (im_pols, q_deg), "{name}");
        proves_its_layout_and_rejects_every_change(&f);
        let xi = agrees_with_the_oracle(&f);
        let points = [xi, Fr::from_u64(7), -&Fr::from_u64(3), Fr::from_u64(1 << 40).pow_u64(3)];
        zerofiers_agree(&info, &points);
        breaks_at_the_edges_are_refused(&f, air);
    }
}

/// Each witness of `air` that breaks one rule at the first or the last row of its domain, and
/// nowhere else (as the oracle says), is refused by the prover as `Unsatisfied`.
fn breaks_at_the_edges_are_refused(f: &Fixture, air: domains::Air) {
    let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
    let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let pk = ProvingKey::load(&f.proving_key).unwrap();
    let options = ProveOptions { insecure_blinding_seed: Some([9; 32]) };
    for rule in 0..air.rules().len() {
        for edge in [domains::Edge::First, domains::Edge::Last] {
            let (witness, row) = domains::broken(air, rule, edge);
            let failures = oracle.check(&oracle.values(&witness, 0).unwrap()).unwrap();
            let rows: Vec<(usize, usize)> = failures.iter().map(|x| (x.constraint, x.row)).collect();
            assert_eq!(rows, [(rule, row)], "{}: rule {rule}, {edge:?}", air.name());
            match prove(&pk, &witness, &options) {
                Err(PilfflonkError::Unsatisfied(message)) => {
                    let expected = format!("the witness does not satisfy the constraints of {}", air.name());
                    assert!(message.contains(&expected), "{message}")
                }
                other => {
                    panic!("{}: rule {rule}, {edge:?}: expected Unsatisfied, got {:?}", air.name(), other.map(|_| ()))
                }
            }
        }
    }
}

/// The `δ = 1` of A.1: `qDeg = max_i(deg c_i + δ_i) − 1`. The domain AIRs set up with their
/// constraints on their domains and, as a control, with every constraint on `everyRow`: by default
/// `qDeg` is 2 on the domains and 1 on `everyRow`, with no im pols; with `--max-constraint-degree
/// 2`, each rule's constraint needs an im pol on its domain and none on `everyRow`, and `qDeg = 1`.
#[test]
fn a_domain_adds_one_to_the_degree_of_its_constraints() {
    use pil2_pilout::pilout::constraint::{self, Constraint as C};
    for air in [domains::Air::FirstRow, domains::Air::LastRow, domains::Air::Frames, domains::Air::All] {
        let n_rules = air.rules().len();
        let mut every_row = domains::pilout(air);
        for c in every_row.air_groups[0].airs[0].constraints.iter_mut() {
            let (expression_idx, debug_line) = match c.constraint.take().unwrap() {
                C::FirstRow(c) => (c.expression_idx, c.debug_line),
                C::LastRow(c) => (c.expression_idx, c.debug_line),
                C::EveryFrame(c) => (c.expression_idx, c.debug_line),
                C::EveryRow(c) => (c.expression_idx, c.debug_line),
            };
            c.constraint = Some(C::EveryRow(constraint::EveryRow { expression_idx, debug_line }));
        }
        for (what, pilout, [default, low]) in
            [("domains", domains::pilout(air), [(2, 0), (1, n_rules)]), ("everyRow", every_row, [(1, 0), (1, 0)])]
        {
            for (degree, (q_deg, im_pols)) in [(9, default), (2, low)] {
                let dir = TestDir::new(&format!("delta_{}_{what}_{degree}", air.name()));
                let opts = SetupPilfflonkOptions { max_constraint_degree: degree, ..setup_options(&dir, DEFAULT) };
                fs::write(&opts.airout_path, pilout.encode_to_vec()).unwrap();
                run_setup_pilfflonk(&opts).unwrap();
                let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
                let global_info = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
                let path = global_info.air_file(&proving_key, 0, 0, AirFile::PilfflonkInfo).unwrap();
                let info = PilfflonkInfo::read(&path).unwrap();
                assert_eq!(info.boundaries == [Boundary::EveryRow], what == "everyRow", "{} {what}", air.name());
                assert_eq!((info.q_deg, n_im_pols(&info)), (q_deg, im_pols), "{} {what}, degree {degree}", air.name());
            }
        }
    }
}

/// Two `firstRow` constraints, which share their boundary.
#[test]
#[ignore = "needs Node.js"]
fn the_prover_proves_first_row_constraints() {
    proves_a_domain(domains::Air::FirstRow, &[Boundary::EveryRow, Boundary::FirstRow]);
}

/// Two `lastRow` constraints: `Z_D = X − ω^(N−1)` (A.1), and not the `ω^N = 1` of spec F.8.
#[test]
#[ignore = "needs Node.js"]
fn the_prover_proves_last_row_constraints() {
    proves_a_domain(domains::Air::LastRow, &[Boundary::EveryRow, Boundary::LastRow]);
}

/// Six `everyFrame` constraints of as many `{offsetMin, offsetMax}`, which read the next row or the
/// previous one, across the wrap too.
#[test]
#[ignore = "needs Node.js"]
fn the_prover_proves_every_frame_constraints() {
    let frame = |offset_min, offset_max| Boundary::EveryFrame { offset_min, offset_max };
    proves_a_domain(
        domains::Air::Frames,
        &[Boundary::EveryRow, frame(1, 2), frame(0, 3), frame(2, 0), frame(3, 1), frame(1, 0), frame(2, 2)],
    );
}

/// One constraint of each domain in one AIR, and two that share an `everyFrame`.
#[test]
#[ignore = "needs Node.js"]
fn the_prover_proves_constraints_of_every_domain() {
    let frame = Boundary::EveryFrame { offset_min: 1, offset_max: 2 };
    proves_a_domain(domains::Air::All, &[Boundary::EveryRow, Boundary::FirstRow, Boundary::LastRow, frame]);
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn a_witness_that_breaks_a_constraint_is_refused() {
    let f = fixture("unsatisfied", Program::Fibonacci, DEFAULT);
    let pk = ProvingKey::load(&f.proving_key).unwrap();
    let source = FileWitnessSource::open(&f.witness, &pk.witness_shape().unwrap()).unwrap();
    let mut witness = Witness::from_source(&source).unwrap();
    // l1 at row 100: the transition constraints fail at rows 99 and 100 (as the oracle says, M14).
    let cell = witness.instances[0].stage1.get(100, 0).unwrap();
    let changed = FrBytes::from_decimal(&(big(&cell) + 1u32).to_string()).unwrap();
    witness.instances[0].stage1.set(100, 0, changed).unwrap();
    let mutated = f.dir.file("mutated");
    witness.write(&mutated, &pk.witness_shape().unwrap()).unwrap();

    let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
    let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let failures = oracle.check(&oracle.values(&witness, 0).unwrap()).unwrap();
    let rows: Vec<(usize, usize)> = failures.iter().map(|x| (x.constraint, x.row)).collect();
    assert_eq!(rows, [(0, 100), (1, 99), (1, 100)]);

    let options = ProveOptions { insecure_blinding_seed: Some([3; 32]) };
    match prove(&pk, &witness, &options) {
        Err(PilfflonkError::Unsatisfied(message)) => {
            assert!(message.contains("the witness does not satisfy the constraints of Fibonacci"), "{message}")
        }
        other => panic!("expected Unsatisfied, got {:?}", other.map(|_| ())),
    }
    let out = prove_cli(&f.proving_key, &mutated, &f.dir.file("proof"), Some(SEED_A));
    assert!(!out.status.success(), "{}", output(&out));
    assert!(output(&out).contains("the witness does not satisfy the constraints of Fibonacci"), "{}", output(&out));
    assert!(!f.dir.file("proof").join("proof.json").exists());
}

fn big(v: &FrBytes) -> num_bigint::BigUint {
    num_bigint::BigUint::from_bytes_le(&v.to_le_bytes())
}

/// The prover of the grouped `f` agrees with the oracle at `ξ = xiSeed^powerW` (A.2, rule 5): every
/// evaluation of a fixed column, at each offset its `f` opens it at (those a fusion adds too), is
/// the oracle's exactly; a committed one is blinded, and is not; and `Q(ξ)`, folded by the oracle
/// over the proof's evaluations, is the prover's, and, if `Q` is split, `Σ_i ξ^(i·M·N)·Q_i(ξ)` of the
/// proof's pieces (A.1), which follow the columns' evaluations in the order of the layout. Returns
/// that `ξ`.
fn agrees_with_the_oracle(f: &Fixture) -> Fr {
    let pk = ProvingKey::load(&f.proving_key).unwrap();
    let source = FileWitnessSource::open(&f.witness, &pk.witness_shape().unwrap()).unwrap();
    let options = ProveOptions { insecure_blinding_seed: Some([5; 32]) };
    let out = prove(&pk, &source, &options).unwrap();
    let info = pk.air(source.instances()[0]).unwrap();
    let power_w = info.layout.power_w().unwrap();
    let xi = Fr::from(out.challenges.xi_seed).pow_u64(power_w);
    let std_vc = Fr::from(out.challenges.std_vc);

    let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
    let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let values = oracle_values(&oracle, &source, info, &out.challenges);
    assert!(oracle.check(&values).unwrap().is_empty(), "the generator's witness satisfies the AIR");

    // Each evaluation of the proof, by its column (the oracle's) and offset: the evMap's const
    // entries, then its cm ones (A.4 step 4).
    let column = |t: PolType, id: u64| -> ColumnRef {
        match t {
            PolType::Const => ColumnRef::Fixed(id as usize),
            PolType::Cm => {
                let p = &info.cm_pols_map[id as usize];
                if p.im_pol {
                    ColumnRef::Im(p.exp_id.unwrap() as usize)
                } else {
                    ColumnRef::Witness { stage: p.stage as usize, idx: p.stage_id as usize }
                }
            }
        }
    };
    let entries = info
        .ev_map
        .iter()
        .filter(|e| e.pol_type == PolType::Const)
        .chain(info.ev_map.iter().filter(|e| e.pol_type == PolType::Cm));
    let n_columns = info.ev_map.len();
    let mut at_xi = BTreeMap::new();
    for (e, value) in entries.zip(&out.proof.evaluations[..n_columns]) {
        at_xi.insert((column(e.pol_type, e.id), e.prime as i32), Fr::from(*value));
    }
    assert_eq!(at_xi.len(), n_columns);

    // The fixed columns have no blinding: their evaluations are the oracle's, exactly.
    let fixed: Vec<(ColumnRef, i32)> =
        at_xi.keys().filter(|(c, _)| matches!(c, ColumnRef::Fixed(_))).copied().collect();
    assert!(!fixed.is_empty());
    for (c, offset) in fixed {
        let expected = oracle.column_at(&values, c, offset, &xi).unwrap();
        assert_eq!(at_xi[&(c, offset)], expected, "{c:?} at ξ·ω^{offset}");
    }
    // The committed ones are blinded (A.3): not the oracle's interpolants at ξ …
    let first = ColumnRef::Witness { stage: 1, idx: 0 };
    assert_ne!(at_xi[&(first, 0)], oracle.column_at(&values, first, 0, &xi).unwrap());
    // … but Q(ξ), as the oracle folds the constraints over the proof's evaluations, is the prover's Q
    // at ξ: the value the verifier computes, and SHPLONK opens Q's f at.
    let im_pols: Vec<usize> =
        info.cm_pols_map.iter().filter(|p| p.im_pol).map(|p| p.exp_id.unwrap() as usize).collect();
    let q = oracle.q_from_evaluations(&values, &at_xi, &im_pols, &std_vc, &xi).unwrap();
    assert_eq!(q, Fr::from(out.challenges.q_at_xi));
    // Split, the pieces' Q_i(ξ) follow, in the order of the layout, and add up to it: Q_0(ξ) + ξ^S·(
    // Q_1(ξ) + ξ^S·(…)), S = M·N.
    let split = info.q_split().unwrap();
    let pieces = &out.proof.evaluations[n_columns..];
    if split.n_pieces() == 1 {
        assert!(pieces.is_empty());
    } else {
        let order = info.layout.0.iter().filter(|f| f.stage == info.q_stage()).flat_map(|f| &f.pols);
        let mut by_index = vec![Fr::zero(); split.n_pieces()];
        for (pol, value) in order.zip(pieces) {
            by_index[info.cm_pols_map[pol.id as usize].stage_pos as usize] = Fr::from(*value);
        }
        assert_eq!(pieces.len(), split.n_pieces());
        let shift = xi.pow_u64(split.stride);
        let joined = by_index.iter().rev().fold(Fr::zero(), |acc, piece| &(&acc * &shift) + piece);
        assert_eq!(joined, q, "the pieces of Q add up to the oracle's Q(ξ)");
    }
    // Another evaluation, another Q(ξ).
    let mut changed = at_xi.clone();
    let value = changed.get_mut(&(first, 0)).unwrap();
    *value = &*value + &Fr::one();
    assert_ne!(oracle.q_from_evaluations(&values, &changed, &im_pols, &std_vc, &xi).unwrap(), q);
    // invZh = 1/Z_H(ξ).
    assert_eq!(&Fr::from(out.proof.inv_zh) * &(&xi.pow_u64(1 << info.n_bits) - &Fr::one()), Fr::one());
    xi
}

/// The oracle's values of the instance of `source`: its stage-1 columns and publics, and for each
/// stage `s` from 2 on, the challenges of the proof (`challenges`) and the columns the std's hints
/// give with them (plan M30).
fn oracle_values(
    oracle: &AirOracle,
    source: &impl WitnessSource,
    info: &PilfflonkInfo,
    challenges: &ProofChallenges,
) -> Values {
    let mut values = oracle.values(source, 0).unwrap();
    for stage in 2..=info.n_stages as usize {
        values.challenges[stage - 1] = challenges.stages[stage - 2].iter().map(Fr::from).collect();
        oracle.fill_hint_columns(&mut values, stage).unwrap();
    }
    values
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_prover_agrees_with_the_oracle() {
    for (name, program, packing, power_w) in [
        ("oracle", Program::Fibonacci, DEFAULT, 2),
        ("oracle_k3", Program::Fibonacci, Packing::ExtraMuls(0), 6),
        ("oracle_unpacked", Program::Fibonacci, Packing::NoPacking, 1),
        ("oracle_packed", Program::Packed, DEFAULT, 12),
    ] {
        let f = fixture(name, program, packing);
        assert_eq!(f.info().layout.power_w().unwrap(), power_w, "{name}");
        agrees_with_the_oracle(&f);
    }
}

/// The prover agrees with the oracle on the fixture of the signed offsets, for each choice of im
/// pols and layout: the evaluations of the fixed columns at `ξ·ω^−1`, `ξ` and `ξ·ω^2`, and `Q(ξ)`
/// folded over the pilout's constraints and the im pols' in the order of `cmPolsMap`.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_prover_agrees_with_the_oracle_on_signed_offsets() {
    for (name, degree, packing, im_pols) in [
        ("oracle_signed", 9, DEFAULT, 1),
        ("oracle_signed_d3", 3, DEFAULT, 3),
        ("oracle_signed_d2", 2, DEFAULT, 8),
        ("oracle_signed_unpacked", 9, Packing::NoPacking, 1),
        ("oracle_signed_unpacked_d2", 2, Packing::NoPacking, 8),
    ] {
        let f = fixture_of_degree(name, Program::Signed, packing, degree);
        assert_eq!(n_im_pols(&f.info()), im_pols, "{name}");
        agrees_with_the_oracle(&f);
    }
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_prover_refuses_a_proving_key_whose_files_disagree() {
    let f = fixture("disagree", Program::Fibonacci, DEFAULT);
    let original = fs::read_to_string(&f.vkey).unwrap();
    let load_error = || ProvingKey::load(&f.proving_key).unwrap_err().to_string();

    // A vkey whose digest is not that of its contents.
    let mut vkey = read_json(&f.vkey);
    vkey["qDeg"] = json!(2);
    write_json(&f.vkey, &vkey);
    assert!(load_error().contains("the digest of the vkey is not the digest of its contents"), "{}", load_error());

    // Sealed again, it is a vkey, but not the pilfflonkinfo's.
    let resealed = |change: fn(&mut Value)| {
        let mut vkey: Value = serde_json::from_str(&original).unwrap();
        change(&mut vkey);
        let sealed = seal_vkey(serde_json::from_value::<Vkey>(vkey).unwrap()).unwrap();
        fs::write(&f.vkey, serde_json::to_string_pretty(&sealed).unwrap()).unwrap();
    };
    resealed(|v| v["layout"][2]["degree"] = json!(260));
    assert!(load_error().contains("the vkey's layout is not the one of the pilfflonkinfo"), "{}", load_error());
    resealed(|v| v["qDeg"] = json!(2));
    assert!(load_error().contains("the vkey's qDeg or maxQDegree"), "{}", load_error());
    resealed(|v| {
        let frame = json!({"name": "everyFrame", "offsetMin": 1, "offsetMax": 1});
        v["boundaries"].as_array_mut().unwrap().push(frame);
    });
    assert!(load_error().contains("the vkey's boundaries"), "{}", load_error());
    // The evMap without the pair of the fusion: the layout opens the im pol at ξ·ω, and the vkey
    // no longer says so.
    resealed(|v| {
        v["evMap"].as_array_mut().unwrap().pop();
    });
    let err = load_error();
    assert!(err.contains("which the evMap does not have"), "{err}");

    // The vkey restored, the C++ loader's refusals come through: a .const cut short.
    fs::write(&f.vkey, &original).unwrap();
    ProvingKey::load(&f.proving_key).unwrap();
    let global_info = PilfflonkGlobalInfo::from_proving_key(&f.proving_key).unwrap();
    let constants = global_info.air_file(&f.proving_key, 0, 0, AirFile::Const).unwrap();
    let good_constants = fs::read(&constants).unwrap();
    let mut bytes = good_constants.clone();
    bytes.truncate(bytes.len() - 32);
    fs::write(&constants, bytes).unwrap();
    let err = load_error();
    assert!(
        err.contains("loading the provingKey/ into the C++ prover") && err.contains(".const has 16352 bytes"),
        "{err}"
    );

    // A .const of the right size and canonical values, but not the one the vkey was set up with
    // (plan M26): the prover would make proofs that do not verify, and refuses the key instead.
    // Row 5 of its first fixed column, 0 in both L1 and LLAST, becomes 1.
    let mut bytes = good_constants.clone();
    let n_fixed = f.info().const_pols_map.len();
    assert_eq!(bytes[5 * n_fixed * 32], 0);
    bytes[5 * n_fixed * 32] = 1;
    fs::write(&constants, &bytes).unwrap();
    let err = load_error();
    assert!(
        err.contains(&constants.display().to_string())
            && err.contains("its fixed columns commit to another f0 than the vkey's: this .const is not the one"),
        "{err}"
    );
    let out = prove_cli(&f.proving_key, &f.witness, &f.dir.file("tampered_const"), Some(SEED_A));
    assert!(!out.status.success(), "prove: {}", output(&out));
    assert!(output(&out).contains("commit to another f0 than the vkey's"), "{}", output(&out));
    fs::write(&constants, &good_constants).unwrap();
    ProvingKey::load(&f.proving_key).unwrap();

    // The SRS of another ptau, of another τ: [τ]₂ is not the vkey's X_2.
    let srs = global_info.srs_path(&f.proving_key);
    let good_srs = fs::read(&srs).unwrap();
    let other_ptau = f.dir.file("other_tau.ptau");
    write_fixed_tau_ptau(&other_ptau, 1024, &(test_tau() + 1u32)).unwrap();
    write_srs(&other_ptau, max_degree(&f.info().layout), &srs).unwrap();
    let err = load_error();
    assert!(
        err.contains(&srs.display().to_string())
            && err.contains("its [τ]₂ is not the vkey's X_2: this SRS is not of the ptau the vkey was set up with"),
        "{err}"
    );
    fs::write(&srs, &good_srs).unwrap();
    ProvingKey::load(&f.proving_key).unwrap();
}

/// Prints, for a vkey, publics and proof, the challenges the JS verifier's `computeChallenges`
/// derives from them (A.4): those of each stage from 2 on, `std_vc` and `xiSeed`, in decimal.
const COMPUTE_CHALLENGES: &str = r#"
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [js, vkeyPath, publicsPath, proofPath] = process.argv.slice(1);
const load = async (file) => import(pathToFileURL(`${js}/${file}`).href);
const { newCurve } = await load("test/support.js");
const { computeChallenges } = await load("src/challenges.js");
const { fromObjectProof, fromObjectPublics } = await load("src/proof.js");
const { fromObjectVk } = await load("src/vkey.js");
const read = (path) => JSON.parse(readFileSync(path, "utf8"));
const curve = await newCurve();
const vk = fromObjectVk(curve, read(vkeyPath));
const c = computeChallenges(curve, vk, fromObjectPublics(curve, read(publicsPath), vk), fromObjectProof(curve, read(proofPath), vk));
const s = (e) => curve.Fr.toString(e, 10);
const stages = [];
for (let stage = 2; stage <= vk.nStages; stage++) stages.push((c.stages[stage] ?? []).map(s));
console.log(JSON.stringify({ stages, stdVc: s(c.stdVc), xiSeed: s(c.xiSeed) }));
"#;

/// The challenges of the prover's transcript are the JS verifier's on its proof (A.4): of stage 2
/// (`numChallenges[1]` of them), `std_vc` and `xiSeed`. Returns the prover's.
fn the_transcripts_agree(f: &Fixture) -> ProofChallenges {
    let pk = ProvingKey::load(&f.proving_key).unwrap();
    let source = FileWitnessSource::open(&f.witness, &pk.witness_shape().unwrap()).unwrap();
    let out = prove(&pk, &source, &ProveOptions { insecure_blinding_seed: Some([7; 32]) }).unwrap();
    let proof = f.dir.file("transcript");
    out.write(&proof).unwrap();
    let run = Command::new("node")
        .args(["--input-type=module", "-e", COMPUTE_CHALLENGES])
        .arg(repo_root().join("pilfflonk/js"))
        .args([&f.vkey, &proof.join("publics.json"), &proof.join("proof.json")])
        .output()
        .expect("node runs");
    assert!(run.status.success(), "computeChallenges: {}", output(&run));
    let js: Value = serde_json::from_slice(&run.stdout).unwrap();
    let decimal = |v: &FrBytes| json!(v.to_decimal());
    let rust = json!({
        "stages": out.challenges.stages.iter().map(|s| s.iter().map(decimal).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "stdVc": decimal(&out.challenges.std_vc),
        "xiSeed": decimal(&out.challenges.xi_seed),
    });
    assert_eq!(js, rust, "the JS verifier's challenges and the prover's");
    out.challenges
}

/// The prover's columns of every stage (`stage_columns`, with the seed of a proof: the same
/// challenges as its) are the oracle's (plan M30): the witness's of stage 1, and from stage 2 on,
/// with the challenges of that proof, the columns the std's hints give, which the oracle computes
/// from the pilout's hints alone, row after row, and the im pols, the values of their expressions.
/// Returns the prover's stage-2 columns.
fn stage_columns_are_the_oracles(f: &Fixture) -> Vec<Vec<FrBytes>> {
    let pk = ProvingKey::load(&f.proving_key).unwrap();
    let source = FileWitnessSource::open(&f.witness, &pk.witness_shape().unwrap()).unwrap();
    let options = ProveOptions { insecure_blinding_seed: Some([11; 32]) };
    let stages = stage_columns(&pk, &source, &options).unwrap();
    let proof = prove(&pk, &source, &options).unwrap();
    assert_eq!(stages.challenges, proof.challenges.stages, "the challenges of the proof of the same seed");
    let info = pk.air(source.instances()[0]).unwrap();
    assert!(info.n_stages >= 2 && stages.challenges.iter().all(|c| !c.is_empty()));

    let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
    let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let values = oracle_values(&oracle, &source, info, &proof.challenges);
    assert!(oracle.check(&values).unwrap().is_empty(), "{}: the oracle's columns satisfy the AIR", info.name);
    let rows = |column: &[FrBytes]| column.iter().map(Fr::from).collect::<Vec<_>>();
    let mut n_hint_columns = 0;
    for p in info.cm_pols_map.iter().filter(|p| p.stage <= info.n_stages) {
        let prover = rows(&stages.columns[p.stage as usize - 1][p.stage_pos as usize]);
        let expected = if p.im_pol {
            oracle.expression_rows(&values, p.exp_id.unwrap() as usize).unwrap()
        } else {
            values.witness[p.stage as usize - 1][p.stage_id as usize].clone()
        };
        assert_eq!(prover, expected, "{}: column {} of stage {}", info.name, p.name, p.stage);
        n_hint_columns += usize::from(p.stage >= 2 && !p.im_pol);
    }
    assert_eq!(n_hint_columns, oracle.bus_hints().len(), "{}: a column per hint", info.name);
    stages.columns[1].clone()
}

/// The stage-2 fixtures (plans M30, M31): a lookup of pairs on the std's sum bus and a permutation of
/// pairs on its product bus, each with its hint (`gsum_col`, `gprod_col`), in `STD_MODE_ONE_INSTANCE`,
/// and the `im_col` hints of the std's default `MAX_CONSTRAINT_DEGREE` (3): the sum bus has one, which
/// its `gsum_col` reads, and with the degree raised to 4 none; the product bus of `prod_bus` none, and
/// that of `prod_bus_im` two, the second reading the first and its `gprod_col` the second. Two
/// stages, with the challenges `std_alpha` and `std_gamma` of stage 2: `numChallenges = [0, 2]`.
/// Grouped by default and with `--no-packing`: the prover proves, the verifier accepts the proof and
/// rejects any change to it or to its public; the prover's transcript is the JS verifier's; its
/// stage-2 columns, the `im_col` ones too, and `Q(ξ)`, the oracle's. The running sum of the sum bus
/// ends at 0, and the running product of the product bus at 1, but no column is constant.
#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_the_std_buses_of_stage_2() {
    use HintKind::{GprodCol, GsumCol, ImCol};
    for (name, program, bus, last, hints) in [
        ("e2e_sum_bus", Program::SumBus, "gsum", FrBytes::ZERO, &[ImCol, GsumCol][..]),
        ("e2e_sum_bus_degree4", Program::SumBusDegree4, "gsum", FrBytes::ZERO, &[GsumCol][..]),
        ("e2e_prod_bus", Program::ProdBus, "gprod", FrBytes::from_u64(1), &[GprodCol][..]),
        ("e2e_prod_bus_im", Program::ProdBusIm, "gprod", FrBytes::from_u64(1), &[ImCol, ImCol, GprodCol][..]),
    ] {
        for (suffix, packing) in [("", DEFAULT), ("_unpacked", Packing::NoPacking)] {
            let name = format!("{name}{suffix}");
            let f = fixture(&name, program, packing);
            let info = f.info();
            assert_eq!(info.n_stages, 2, "{name}");
            let challenges: Vec<(&str, u64)> = info.challenges_map.iter().map(|c| (c.name.as_str(), c.stage)).collect();
            assert_eq!(challenges, [("std_alpha", 2), ("std_gamma", 2), ("std_vc", 3), ("std_xi", 4)], "{name}");
            let global_info = PilfflonkGlobalInfo::from_proving_key(&f.proving_key).unwrap();
            assert_eq!(global_info.num_challenges, [0, 2], "{name}");
            let column = info.cm_pols_map.iter().find(|p| p.name == bus).unwrap();
            assert!(column.stage == 2 && !column.im_pol, "{name}");
            let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
            let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
            let kinds: Vec<HintKind> = oracle.bus_hints().iter().map(|h| h.kind).collect();
            assert_eq!(kinds, hints, "{name}: the std's hints");
            // Its f opens it at ξ·ω^−1 too: the bus reads its previous row.
            let f_of = info.layout.0.iter().find(|entry| entry.pols.iter().any(|p| p.name == bus)).unwrap();
            assert!(f_of.stage == 2 && f_of.offsets.contains(&-1), "{name}: {:?}", f_of.offsets);

            proves_its_layout_and_rejects_every_change(&f);
            let challenges = the_transcripts_agree(&f);
            assert_eq!(challenges.stages.len(), 1, "{name}");
            assert_eq!(challenges.stages[0].len(), 2, "{name}");
            agrees_with_the_oracle(&f);
            let stage_2 = stage_columns_are_the_oracles(&f);
            let bus_column = &stage_2[column.stage_pos as usize];
            assert_eq!(bus_column.last(), Some(&last), "{name}: the bus balances");
            assert!(bus_column.iter().any(|v| *v != last), "{name}: {bus} is not constant");
            for hint in oracle.bus_hints().iter().filter(|h| h.kind == ImCol) {
                let of_hint = |p: &&PolMapEntry| p.stage == 2 && !p.im_pol && p.stage_id == hint.idx as u64;
                let p = info.cm_pols_map.iter().find(of_hint).unwrap();
                let im = &stage_2[p.stage_pos as usize];
                assert!(im.iter().any(|v| *v != im[0]), "{name}: the im_col column {} is not constant", p.name);
            }
        }
    }
}

/// A witness that breaks the bus: a lookup of a pair the table does not provide, with and without the
/// sum bus's `im_col`; a pair of `(b, d)` that is no row of `(a, c)`; and, with the product bus's two
/// `im_col`, a pair moved to the other permutation. Each row's running sum or product, and each
/// `im_col`, holds by construction, and the constraint that breaks is the last row's (`L1'·…`), as the
/// oracle says with the prover's columns; the prover refuses the witness, whatever the layout.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn a_witness_that_breaks_a_bus_is_refused() {
    for (name, program, witness, line) in [
        ("bus_sum", Program::SumBus, sum_bus::witness_looking_up_what_is_not_provided(), "__L1__'*(0-gsum)"),
        (
            "bus_sum_degree4",
            Program::SumBusDegree4,
            sum_bus::witness_looking_up_what_is_not_provided(),
            "__L1__'*(0-gsum)",
        ),
        ("bus_prod", Program::ProdBus, prod_bus::witness_not_a_permutation(), "__L1__'*(1-gprod)"),
        (
            "bus_prod_im",
            Program::ProdBusIm,
            prod_bus_im::witness_with_a_pair_in_the_other_permutation(),
            "__L1__'*(1-gprod)",
        ),
    ] {
        for (suffix, packing) in [("", DEFAULT), ("_unpacked", Packing::NoPacking)] {
            let name = format!("{name}{suffix}");
            let f = fixture(&name, program, packing);
            let pk = ProvingKey::load(&f.proving_key).unwrap();
            let options = ProveOptions { insecure_blinding_seed: Some([3; 32]) };
            let stages = stage_columns(&pk, &witness, &options).unwrap();

            let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
            let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
            let info = f.info();
            let challenges = ProofChallenges {
                stages: stages.challenges.clone(),
                std_vc: FrBytes::ZERO,
                xi_seed: FrBytes::ZERO,
                q_at_xi: FrBytes::ZERO,
            };
            let values = oracle_values(&oracle, &witness, &info, &challenges);
            let failures = oracle.check(&values).unwrap();
            let n = 1usize << info.n_bits;
            let rows: Vec<(usize, usize)> = failures.iter().map(|x| (x.constraint, x.row)).collect();
            let last = oracle.constraints().iter().position(|c| c.debug_line.contains(line)).unwrap();
            assert_eq!(rows, [(last, n - 1)], "{name}");

            match prove(&pk, &witness, &options) {
                Err(PilfflonkError::Unsatisfied(message)) => {
                    let expected = format!("the witness does not satisfy the constraints of {}", info.name);
                    assert!(message.contains(&expected), "{name}: {message}")
                }
                other => panic!("{name}: expected Unsatisfied, got {:?}", other.map(|_| ())),
            }
        }
    }
}
