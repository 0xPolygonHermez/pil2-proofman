//! `proofman-cli pilfflonk check` (pilfflonk/docs/README.md#pilfflonk-check) on the Fibonacci
//! fixture: the witness of its generator passes, and a mutated one fails exactly the constraints
//! and rows the Rust oracle says, with its values, through the library
//! (`proofman_pilfflonk::check`) and the CLI, whose output is `verify-constraints`'s. And on the
//! fixture of the signed offsets, whose constraints read the rows −1 to 2 around each row, across
//! the wrap too, with the im pols the setup chooses for each `--max-constraint-degree`. And on the
//! pilouts of `tests/data/domains.rs`, built in code: each constraint is checked on the rows of its
//! domain only, `firstRow ≤ i < lastRow`, and a witness that breaks it at the edge row of its
//! domain is found there. And on the stage-2 fixtures of the std's buses, with and without
//! `im_col`, whose challenges of stage 2 the check takes from a transcript of fixed elements, as
//! the STARK's `verify-constraints` does, committing nothing: its columns, the `im_col` ones too,
//! are the oracle's with them, and a witness that breaks the bus fails the last row of its running
//! sum or product with the oracle's value, and the CLI names it. And the same on the pil-fflonk
//! examples ported to PIL2 (pilfflonk/docs/README.md#fixtures), on the sum and on the product bus,
//! with a wrong multiplicity, a broken permutation or connection, or a value out of a range. And on
//! the witness a witness library computes (pilfflonk/docs/README.md#witness), `--witness-lib` in
//! place of `--witness`: that of the libraries of the Fibonacci, the Connection and `all` passes,
//! as their generators' does, and the command refuses a STARK witness library, a library that is
//! not there, and the flags that do not go together.
//!
//! Pilouts are not versioned: the test compiles the fixture with the compiler `PIL2C_EXEC` names,
//! which must honour `prime`, and is `#[ignore]` without it. Those of the domains build their
//! pilouts in code, and always run:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only \
//!     --test pilfflonk_check -- --ignored --test-threads 2
//! ```
//!
//! Its tests call the C++ core in this process, each from its own thread, which OpenMP makes a root
//! with a team of one thread per CPU, kept while that thread lives. libomp 14 (Ubuntu 22.04's) can
//! crash with SIGSEGV once the teams outgrow its first table of threads (4 per CPU): it replaces
//! the table while the workers it has just started may still be reading the old one
//! (pilfflonk/docs/README.md#tests; the lock of `setup/pilfflonk/tests/setup/common.rs`,
//! `cpp_core`, has the details). It has not happened in these tests, but nothing rules it out: run
//! them with `--test-threads 2`, as above, which keeps the teams alive, counting those of
//! tests that are just ending, within that table.

#[path = "../../pilfflonk/tests/data/all.rs"]
mod all;
#[path = "../../pilfflonk/tests/data/connection.rs"]
mod connection;
#[path = "../../pilfflonk/tests/data/domains.rs"]
mod domains;
#[path = "../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;
#[path = "../../pilfflonk/tests/data/permutation.rs"]
mod permutation;
#[path = "../../pilfflonk/tests/data/plookup.rs"]
mod plookup;
#[path = "../../pilfflonk/tests/data/prod_bus.rs"]
mod prod_bus;
#[path = "../../pilfflonk/tests/data/prod_bus_im.rs"]
mod prod_bus_im;
#[path = "../../pilfflonk/tests/data/range_check.rs"]
mod range_check;
#[path = "../../pilfflonk/tests/data/signed.rs"]
mod signed;
#[path = "../../pilfflonk/tests/data/sum_bus.rs"]
mod sum_bus;
#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pil2_pilout::pilout_proxy::PilOutProxy;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman_pilfflonk::oracle::{AirOracle, Fr};
use proofman_pilfflonk::{
    check, check_columns, CheckOptions, CheckReport, FileWitnessSource, FrBytes, PilfflonkError, ProvingKey, Witness,
    BN254_R,
};
use proofman_starks_lib_c::PilFflonkTranscript;
use prost::Message;
use witness_libraries::built_library;

const N: usize = 256;
const L1: usize = 0;
const L2: usize = 1;

/// A fresh directory for the test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_check_{name}_{}", std::process::id()));
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

/// What a command wrote: the CLI logs to stdout.
fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// A fixture's pilout: compiled from a PIL, or built in code.
#[derive(Clone, Copy, Debug)]
enum Program {
    Pil(&'static str),
    Domains(domains::Air),
}

const FIBONACCI: Program = Program::Pil("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil");
const SIGNED: Program = Program::Pil("pilfflonk/tests/fixtures/signed/signed.pil");
const SUM_BUS: Program = Program::Pil("pilfflonk/tests/fixtures/sum_bus/sum_bus.pil");
const SUM_BUS_DEGREE4: Program = Program::Pil("pilfflonk/tests/fixtures/sum_bus/sum_bus_degree4.pil");
const PROD_BUS: Program = Program::Pil("pilfflonk/tests/fixtures/prod_bus/prod_bus.pil");
const PROD_BUS_IM: Program = Program::Pil("pilfflonk/tests/fixtures/prod_bus_im/prod_bus_im.pil");
const PLOOKUP_SUM: Program = Program::Pil("pilfflonk/tests/fixtures/plookup/plookup_sum.pil");
const PLOOKUP_PROD: Program = Program::Pil("pilfflonk/tests/fixtures/plookup/plookup_prod.pil");
const PERMUTATION_SUM: Program = Program::Pil("pilfflonk/tests/fixtures/permutation/permutation_sum.pil");
const PERMUTATION_PROD: Program = Program::Pil("pilfflonk/tests/fixtures/permutation/permutation_prod.pil");
const CONNECTION_SUM: Program = Program::Pil("pilfflonk/tests/fixtures/connection/connection_sum.pil");
const CONNECTION_PROD: Program = Program::Pil("pilfflonk/tests/fixtures/connection/connection_prod.pil");
const RANGE_CHECK_SUM: Program = Program::Pil("pilfflonk/tests/fixtures/range_check/range_check_sum.pil");
const RANGE_CHECK_PROD: Program = Program::Pil("pilfflonk/tests/fixtures/range_check/range_check_prod.pil");
const ALL_SUM: Program = Program::Pil("pilfflonk/tests/fixtures/all/all_sum.pil");
const ALL_PROD: Program = Program::Pil("pilfflonk/tests/fixtures/all/all_prod.pil");

impl Program {
    /// The name of its pilout file: that of the PIL, or `domains`.
    fn pilout_file(self) -> String {
        match self {
            Program::Pil(pil) => format!("{}.pilout", Path::new(pil).file_stem().unwrap().to_str().unwrap()),
            Program::Domains(_) => "domains.pilout".to_string(),
        }
    }
}

/// Compiles `program` over BN254 to `pilout` with `PIL2C_EXEC`, or writes the pilout it builds.
fn compile(program: Program, pilout: &Path) {
    let pil = match program {
        Program::Pil(pil) => pil,
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

struct Fixture {
    dir: TestDir,
    pilout: PathBuf,
    proving_key: PathBuf,
    /// The witness of the Fibonacci's generator for [1, 2], and its directory.
    witness: Witness,
    witness_dir: PathBuf,
}

/// The Fibonacci compiled and set up (`--no-packing`; the check commits nothing, so the ptau of
/// `τ = 1` does), and its witness written.
fn fixture(name: &str) -> Fixture {
    fixture_of(name, FIBONACCI, fibonacci::witness(8, [1, 2]), DEFAULT_MAX_CONSTRAINT_DEGREE, true)
}

/// `program` compiled and set up with `--max-constraint-degree max_constraint_degree`, grouped or
/// with `--no-packing`, and `witness` written.
fn fixture_of(name: &str, program: Program, witness: Witness, max_constraint_degree: u64, no_packing: bool) -> Fixture {
    let dir = TestDir::new(name);
    let opts = SetupPilfflonkOptions {
        airout_path: dir.file(&program.pilout_file()),
        build_dir: dir.file("build"),
        powers_of_tau: dir.file("tau_one.ptau"),
        max_constraint_degree,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing,
        solidity: false,
    };
    compile(program, &opts.airout_path);
    // More powers than the largest degree of every layout here: 3080, the Connection's of N = 2^10.
    write_tau_one_ptau(&opts.powers_of_tau, 4096).unwrap();
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let witness_dir = dir.file("witness");
    witness.write(&witness_dir, &ProvingKey::load(&proving_key).unwrap().witness_shape().unwrap()).unwrap();
    Fixture { pilout: opts.airout_path, dir, proving_key, witness, witness_dir }
}

fn plus(value: &FrBytes, delta: u64) -> FrBytes {
    (&Fr::from(value) + &Fr::from_u64(delta)).to_bytes()
}

/// The witness with cell (row, column) plus `delta`.
fn mutated(witness: &Witness, cells: &[(usize, usize, u64)]) -> Witness {
    let mut out = witness.clone();
    for &(column, row, delta) in cells {
        let stage1 = &mut out.instances[0].stage1;
        let value = plus(&stage1.get(row, column).unwrap(), delta);
        stage1.set(row, column, value).unwrap();
    }
    out
}

/// Every failed (constraint, row, value) of a report, which kept them all.
fn found(report: &CheckReport) -> BTreeSet<(usize, usize, String)> {
    let mut out = BTreeSet::new();
    for c in &report.constraints {
        assert_eq!(c.failed_rows.len() as u64, c.n_failed_rows, "constraint {}: every row kept", c.index);
        for r in &c.failed_rows {
            out.insert((c.index, r.row as usize, r.value.to_decimal()));
        }
    }
    out
}

/// A deterministic stream of numbers for the random mutations (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) % bound
    }
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_check_finds_what_the_oracle_does() {
    let f = fixture("oracle");
    let pk = ProvingKey::load(&f.proving_key).unwrap();
    let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
    let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let all_rows = CheckOptions { max_rows: N };

    // The generator's witness, read back from its directory: every constraint holds, and they are
    // the pilout's with their lines, then the im pol's.
    let source = FileWitnessSource::open(&f.witness_dir, &pk.witness_shape().unwrap()).unwrap();
    let report = check(&pk, &source, &CheckOptions::default()).unwrap();
    assert!(report.holds() && report.failures().next().is_none());
    assert_eq!((report.air_name.as_str(), report.constraints.len()), ("Fibonacci", 6));
    for (c, constraint) in report.constraints.iter().enumerate() {
        assert_eq!((c, constraint.stage, constraint.first_row, constraint.last_row), (constraint.index, 1, 0, 256));
        assert_eq!(constraint.im_pol, c == 5);
        if c < 5 {
            assert_eq!(constraint.line, format!("{} == 0", oracle.constraints()[c].debug_line));
        }
    }
    assert_eq!(report.constraints[5].line, "(Fibonacci.ImPol0 - (l1' - ((l1 * l1) + (l2 * l2)))) == 0");

    // Mutations: the oracle's own cases (`pilfflonk/tests/fibonacci.rs`), the publics, and random
    // ones of one cell and of several.
    let mut cases: Vec<(String, Witness)> = [(L1, 100), (L2, 100), (L1, 0), (L2, 0), (L1, N - 1), (L2, N - 1), (L1, 1)]
        .iter()
        .map(|&(column, row)| (format!("column {column}, row {row}"), mutated(&f.witness, &[(column, row, 1)])))
        .collect();
    for public in 0..3 {
        let mut w = f.witness.clone();
        w.publics[public] = plus(&w.publics[public], 1);
        cases.push((format!("public {public}"), w));
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for i in 0..30 {
        let n_cells = if i < 20 { 1 } else { 2 + rng.next(4) as usize };
        let cells: Vec<(usize, usize, u64)> =
            (0..n_cells).map(|_| (rng.next(2) as usize, rng.next(N as u64) as usize, 1 + rng.next(1 << 40))).collect();
        cases.push((format!("cells {cells:?}"), mutated(&f.witness, &cells)));
    }

    let mut total = 0;
    for (what, witness) in &cases {
        let report = check(&pk, witness, &all_rows).unwrap();
        let expected: BTreeSet<(usize, usize, String)> = oracle
            .check(&oracle.values(witness, 0).unwrap())
            .unwrap()
            .into_iter()
            .map(|failure| (failure.constraint, failure.row, failure.value.to_bytes().to_decimal()))
            .collect();
        assert!(!expected.is_empty(), "{what}: the oracle finds nothing");
        assert_eq!(found(&report), expected, "{what}");
        assert!(!report.holds() && report.constraints[5].holds(), "{what}: the im pol's constraint holds");
        total += expected.len();
    }
    println!("{} mutations, {total} failed rows: the check's, value for value, are the oracle's", cases.len());

    // The oracle's cases, as `pilfflonk/tests/fibonacci.rs` lists them.
    let pairs = |report: &CheckReport| -> Vec<(usize, u64)> {
        report.failures().flat_map(|c| c.failed_rows.iter().map(|r| (c.index, r.row))).collect()
    };
    assert_eq!(pairs(&check(&pk, &cases[0].1, &all_rows).unwrap()), [(0, 100), (1, 99), (1, 100)]);
    assert_eq!(pairs(&check(&pk, &cases[2].1, &all_rows).unwrap()), [(0, 0), (1, 0), (3, 0)]);
    assert_eq!(pairs(&check(&pk, &cases[9].1, &all_rows).unwrap()), [(4, 255)]);
}

fn check_cli(key: &Path, witness: &Path, extra: &[&str]) -> Output {
    check_from(key, &["--witness", witness.to_str().unwrap()], extra)
}

/// `proofman-cli pilfflonk check -k <key> <source> <extra>`, with the witness where `source` says:
/// `--witness <dir>`, or `--witness-lib <library> [--public-inputs <json>]`.
fn check_from(key: &Path, source: &[&str], extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proofman-cli"))
        .args(["pilfflonk", "check", "-k", key.to_str().unwrap()])
        .args(source)
        .args(extra)
        .output()
        .expect("proofman-cli runs")
}

/// `r − 1`, the value of a numerator that is −1.
const MINUS_ONE: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495616";

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_cli_says_which_constraint_and_row_fail() {
    let f = fixture("cli");
    let shape = ProvingKey::load(&f.proving_key).unwrap().witness_shape().unwrap();

    // The generator's witness: exit 0; with -v, every constraint is valid.
    let out = check_cli(&f.proving_key, &f.witness_dir, &["-v"]);
    let text = output(&out);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("► Instance #0 of Fibonacci [0:0]"), "{text}");
    assert!(
        text.contains("Constraint #0 (stage 1) is valid -> fibonacci.pil:24 (l2'-l1)*(1-Fibonacci.LLAST) == 0"),
        "{text}"
    );
    assert!(text.contains("✓ All constraints for Instance #0 of Fibonacci were verified"), "{text}");
    assert!(!text.contains("invalid rows"), "{text}");

    // l1[100] + 1: exit 1, the two transition constraints at their rows, with the values.
    let dir = f.dir.file("l1_100");
    mutated(&f.witness, &[(L1, 100, 1)]).write(&dir, &shape).unwrap();
    let out = check_cli(&f.proving_key, &dir, &[]);
    let text = output(&out);
    assert_eq!(out.status.code(), Some(1), "{text}");
    for expected in [
        "Constraint #0 (stage 1) has 1 invalid rows -> fibonacci.pil:24 (l2'-l1)*(1-Fibonacci.LLAST) == 0",
        &format!("✗ Failed at row 100 with value: {MINUS_ONE}"),
        "Constraint #1 (stage 1) has 2 invalid rows -> fibonacci.pil:27 (l1'-((l1*l1)+(l2*l2)))*(1-Fibonacci.LLAST) == 0",
        "✗ Failed at row 99 with value: 1\n",
        "✗ Not all constraints for Instance #0 of Fibonacci were verified",
        "Not all constraints for Instance #0 of Fibonacci were verified",
    ] {
        assert!(text.contains(expected), "{expected:?} not in:\n{text}");
    }
    assert_eq!(text.matches("Failed at row").count(), 3, "{text}");
    for valid in ["Constraint #2", "Constraint #3", "Constraint #4", "Constraint #5", "is valid"] {
        assert!(!text.contains(valid), "{valid} without -v:\n{text}");
    }
    println!("{text}");

    // A public: out + 1 breaks LLAST·(l1 − out) at the last row.
    let dir = f.dir.file("out");
    let mut w = f.witness.clone();
    w.publics[2] = plus(&w.publics[2], 1);
    w.write(&dir, &shape).unwrap();
    let out = check_cli(&f.proving_key, &dir, &[]);
    let text = output(&out);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(
        text.contains("Constraint #4 (stage 1) has 1 invalid rows -> fibonacci.pil:31 Fibonacci.LLAST*(l1-out) == 0"),
        "{text}"
    );
    assert!(text.contains(&format!("Failed at row 255 with value: {MINUS_ONE}")), "{text}");

    // Every l1 + 1, two rows printed of each constraint: all are counted. The transitions fail on
    // the 255 rows 1 − LLAST keeps (l2' − l1 = −1, and l1' − next = −2·l1, l1 never 0), and the
    // boundaries at theirs.
    let dir = f.dir.file("every_l1");
    let cells: Vec<(usize, usize, u64)> = (0..N).map(|row| (L1, row, 1)).collect();
    mutated(&f.witness, &cells).write(&dir, &shape).unwrap();
    let out = check_cli(&f.proving_key, &dir, &["--max-rows", "2"]);
    let text = output(&out);
    assert_eq!(out.status.code(), Some(1), "{text}");
    for expected in [
        "Constraint #0 (stage 1) has 255 invalid rows",
        "Constraint #1 (stage 1) has 255 invalid rows",
        "Constraint #3 (stage 1) has 1 invalid rows",
        "Constraint #4 (stage 1) has 1 invalid rows",
        &format!("✗ Failed at row 0 with value: {MINUS_ONE}"),
        &format!("✗ Failed at row 1 with value: {MINUS_ONE}"),
    ] {
        assert!(text.contains(expected), "{expected:?} not in:\n{text}");
    }
    assert_eq!(text.matches("… and 253 more invalid rows (--max-rows)").count(), 2, "{text}");
    assert!(!text.contains("Constraint #2 ") && !text.contains("Constraint #5 "), "{text}");
    assert_eq!(text.matches("Failed at row").count(), 2 + 2 + 1 + 1, "{text}");

    // A witness directory that is not there: an error, not a check.
    let out = check_cli(&f.proving_key, &f.dir.file("nothing"), &[]);
    assert_eq!(out.status.code(), Some(1), "{}", output(&out));
    assert!(!output(&out).contains("Instance #0"), "{}", output(&out));
}

/// The offset in a `.bin` of the stage of constraint `c`, the first word of its entry in section 2
/// (the format: `setup/pilfflonk/src/bytecode.rs`).
fn constraint_stage_offset(bin: &[u8], c: usize) -> usize {
    let u64_at = |at: usize| u64::from_le_bytes(bin[at..at + 8].try_into().unwrap()) as usize;
    // "chps", version, nSections; section 1's id, size and payload; section 2's id and size; its
    // nOps, nArgs, nNumbers and nConstraints.
    let mut at = 12 + 12 + u64_at(16) + 12 + 16;
    for _ in 0..c {
        at += 10 * 4; // stage … argsOffset, imPol
        at += bin[at..].iter().position(|&b| b == 0).unwrap() + 1; // the line
    }
    at
}

/// A constraint of a stage the AIR does not have: the Fibonacci's, of one stage, with a constraint
/// of stage 2.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn a_constraint_of_a_stage_the_air_does_not_have_is_refused() {
    let f = fixture("stage2");
    let bin = f.proving_key.join("fibonacci/Fibonacci/airs/Fibonacci/air/Fibonacci.bin");
    let mut bytes = fs::read(&bin).unwrap();
    for c in 0..6 {
        let at = constraint_stage_offset(&bytes, c);
        assert_eq!(bytes[at..at + 4], 1u32.to_le_bytes(), "constraint {c} is of stage 1");
    }
    let at = constraint_stage_offset(&bytes, 1);
    bytes[at..at + 4].copy_from_slice(&2u32.to_le_bytes());
    fs::write(&bin, bytes).unwrap();

    let pk = ProvingKey::load(&f.proving_key).unwrap();
    let err = check(&pk, &f.witness, &CheckOptions::default()).unwrap_err().to_string();
    let expected = "constraint 1 (fibonacci.pil:27 (l1'-((l1*l1)+(l2*l2)))*(1-Fibonacci.LLAST) == 0) is of stage 2";
    assert!(err.contains("checking the witness row by row") && err.contains(expected), "{err}");
    assert!(err.contains("is of stage 2, and Fibonacci has 1 stages"), "{err}");
    let out = check_cli(&f.proving_key, &f.witness_dir, &[]);
    assert_eq!(out.status.code(), Some(1), "{}", output(&out));
    assert!(output(&out).contains(expected), "{}", output(&out));
}

/// The fixture of the signed offsets: the inputs of its witness, its witness columns and its rows.
const SIGNED_INPUTS: [u64; 2] = [3, 5];
const A: usize = 0;
const B: usize = 1;
const E: usize = 4;
const G: usize = 5;
const SIGNED_N: usize = 32;

/// `(constraint, row)` of every failure of a report.
fn failed_rows(report: &CheckReport) -> BTreeSet<(usize, usize)> {
    found(report).into_iter().map(|(c, row, _)| (c, row)).collect()
}

/// On the fixture of the signed offsets, with the im pols of each `--max-constraint-degree` and
/// layout, the check finds the rows each mutation breaks: those whose window of rows −1 to 2 reads
/// the cell, where the constraint holds (`WIN` is 1 on rows 1 to N − 3), across the wrap too. And
/// they are the oracle's, value for value, for random mutations.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_check_finds_the_rows_the_signed_offsets_read() {
    let witness = signed::witness(SIGNED_INPUTS);
    // The constraints, in the pilout's order: 0 and 1 the windows of a (on WIN's rows), 2 c's
    // recurrence, 3 d = 'a·b'2·e', 4 e = a² + P'2, 5 g'2 = a', 6 to 9 at L1 (9 c = 'a), 10 at LLAST.
    // a[10]: the windows at rows 8 to 11, d at 11, e at 10, g at 9.
    let a_10: BTreeSet<(usize, usize)> =
        (8..=11).flat_map(|row| [(0, row), (1, row)]).chain([(3, 11), (4, 10), (5, 9)]).collect();
    // a[N − 1]: the windows at row N − 3 only (WIN is 0 at N − 2, N − 1 and 0), 'a at row 0 across
    // the wrap (d, and c = 'a at L1), e at N − 1, g at N − 2.
    let a_last: BTreeSet<(usize, usize)> = [(0, 29), (1, 29), (3, 0), (4, 31), (5, 30), (9, 0)].into_iter().collect();
    // b[1]: its constraint and c's recurrence at row 1, b'2 of d at row N − 1, across the wrap.
    let b_1: BTreeSet<(usize, usize)> = [(1, 1), (2, 1), (3, 31)].into_iter().collect();
    // e[0]: c's recurrence and e's constraint at row 0, e' of d at row N − 1; WIN is 0 at row 0.
    let e_0: BTreeSet<(usize, usize)> = [(2, 0), (3, 31), (4, 0)].into_iter().collect();

    for (name, degree, no_packing, n_im_pols) in [
        ("signed", 9, false, 1),
        ("signed_d3", 3, false, 3),
        ("signed_d2", 2, false, 8),
        ("signed_d2_unpacked", 2, true, 8),
    ] {
        let f = fixture_of(name, SIGNED, witness.clone(), degree, no_packing);
        let pk = ProvingKey::load(&f.proving_key).unwrap();
        let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
        let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
        let all_rows = CheckOptions { max_rows: SIGNED_N };

        // The generator's witness: every constraint holds, the pilout's eleven and then the im pols'.
        let source = FileWitnessSource::open(&f.witness_dir, &pk.witness_shape().unwrap()).unwrap();
        let report = check(&pk, &source, &CheckOptions::default()).unwrap();
        assert!(report.holds(), "{name}");
        assert_eq!(report.constraints.len(), 11 + n_im_pols, "{name}");
        for (c, constraint) in report.constraints.iter().enumerate() {
            assert_eq!(constraint.im_pol, c >= 11, "{name}: constraint {c}");
            if c < 11 {
                assert_eq!(constraint.line, format!("{} == 0", oracle.constraints()[c].debug_line), "{name}");
            }
        }

        let mut public = f.witness.clone();
        public.publics[1] = plus(&public.publics[1], 1);
        let broken = signed::witness_broken_across_the_wrap(SIGNED_INPUTS);
        for (what, mutated, expected) in [
            ("a[10]", mutated(&f.witness, &[(A, 10, 1)]), a_10.clone()),
            ("a[N - 1]", mutated(&f.witness, &[(A, SIGNED_N - 1, 1)]), a_last.clone()),
            ("b[1]", mutated(&f.witness, &[(B, 1, 1)]), b_1.clone()),
            ("e[0]", mutated(&f.witness, &[(E, 0, 1)]), e_0.clone()),
            // g'2 at row N − 2, across the wrap.
            ("g[0]", mutated(&f.witness, &[(G, 0, 1)]), [(5, 30)].into_iter().collect()),
            ("in2", public, [(7, 0)].into_iter().collect()),
            ("c[0] across the wrap", broken, [(9, 0)].into_iter().collect()),
        ] {
            let report = check(&pk, &mutated, &all_rows).unwrap();
            assert_eq!(failed_rows(&report), expected, "{name}: {what}");
            assert!(
                report.constraints[11..].iter().all(|c| c.holds()),
                "{name}: {what}: the im pols' constraints hold"
            );
            let oracle_found: BTreeSet<(usize, usize, String)> = oracle
                .check(&oracle.values(&mutated, 0).unwrap())
                .unwrap()
                .into_iter()
                .map(|failure| (failure.constraint, failure.row, failure.value.to_bytes().to_decimal()))
                .collect();
            assert_eq!(found(&report), oracle_found, "{name}: {what}");
        }

        // Random mutations of one cell and of several, value for value the oracle's.
        let mut rng = Rng(0x2545_f491_4f6c_dd1d ^ degree);
        for i in 0..20 {
            let n_cells = if i < 12 { 1 } else { 2 + rng.next(4) as usize };
            let cells: Vec<(usize, usize, u64)> = (0..n_cells)
                .map(|_| (rng.next(6) as usize, rng.next(SIGNED_N as u64) as usize, 1 + rng.next(1 << 40)))
                .collect();
            let mutated = mutated(&f.witness, &cells);
            let expected: BTreeSet<(usize, usize, String)> = oracle
                .check(&oracle.values(&mutated, 0).unwrap())
                .unwrap()
                .into_iter()
                .map(|failure| (failure.constraint, failure.row, failure.value.to_bytes().to_decimal()))
                .collect();
            assert!(!expected.is_empty(), "{name}: cells {cells:?}: the oracle finds nothing");
            assert_eq!(found(&check(&pk, &mutated, &all_rows).unwrap()), expected, "{name}: cells {cells:?}");
        }
    }
}

/// The CLI on the fixture of the signed offsets: a[10] + 1 fails constraints 0, 1, 3, 4 and 5 at
/// the rows their offsets read it from, each named by its PIL line.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_cli_says_which_rows_the_signed_offsets_break() {
    let f = fixture_of("signed_cli", SIGNED, signed::witness(SIGNED_INPUTS), DEFAULT_MAX_CONSTRAINT_DEGREE, false);
    let shape = ProvingKey::load(&f.proving_key).unwrap().witness_shape().unwrap();
    let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
    let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let line = |c: usize| oracle.constraints()[c].debug_line.clone();

    let out = check_cli(&f.proving_key, &f.witness_dir, &[]);
    assert!(out.status.success(), "{}", output(&out));
    assert!(output(&out).contains("✓ All constraints for Instance #0 of Signed were verified"), "{}", output(&out));

    let dir = f.dir.file("a_10");
    mutated(&f.witness, &[(A, 10, 1)]).write(&dir, &shape).unwrap();
    let out = check_cli(&f.proving_key, &dir, &[]);
    let text = output(&out);
    assert_eq!(out.status.code(), Some(1), "{text}");
    for (c, n) in [(0, 4), (1, 4), (3, 1), (4, 1), (5, 1)] {
        let expected = format!("Constraint #{c} (stage 1) has {n} invalid rows -> {} == 0", line(c));
        assert!(text.contains(&expected), "{expected:?} not in:\n{text}");
    }
    for row in 8..=11 {
        assert!(text.contains(&format!("✗ Failed at row {row} with value: ")), "row {row} not in:\n{text}");
    }
    assert_eq!(text.matches("Failed at row").count(), 4 + 4 + 1 + 1 + 1, "{text}");
    for holds in ["Constraint #2 ", "Constraint #6 ", "Constraint #11 "] {
        assert!(!text.contains(holds), "{holds} without -v:\n{text}");
    }
    println!("{text}");
}

/// The domain AIRs of `tests/data/domains.rs`, with no im pols (the default) and with one per
/// rule (`--max-constraint-degree 2`, where the whole constraint of each rule is an im pol),
/// grouped and with `--no-packing`: every constraint is checked on the rows of its domain, and the
/// witness, which breaks each rule on every row out of its domain, passes; a witness that breaks a
/// rule at the first or the last row of its domain, and nowhere else, fails there, with the
/// oracle's value.
#[test]
fn the_check_finds_the_edge_rows_of_every_domain() {
    use domains::{Air, Edge};
    for air in [Air::FirstRow, Air::LastRow, Air::Frames, Air::All] {
        let rules = air.rules();
        let n = 1usize << domains::N_BITS;
        for (degree, no_packing, n_im_pols) in [(9, false, 0), (2, false, rules.len()), (2, true, rules.len())] {
            let name = format!("{}_{degree}_{no_packing}", air.name());
            let f = fixture_of(&name, Program::Domains(air), domains::witness(air), degree, no_packing);
            let pk = ProvingKey::load(&f.proving_key).unwrap();
            let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
            let oracle = AirOracle::new(&pilout, 0, 0).unwrap();

            // The rules' constraints, on their rows; y − x0·K and the im pols' on every row.
            let report = check(&pk, &f.witness, &CheckOptions::default()).unwrap();
            assert!(report.holds(), "{name}");
            assert_eq!(report.constraints.len(), rules.len() + 1 + n_im_pols, "{name}");
            for (c, constraint) in report.constraints.iter().enumerate() {
                let rows = rules.get(c).map_or(0..n, |rule| rule.rows());
                assert_eq!(
                    (constraint.first_row, constraint.last_row),
                    (rows.start as u64, rows.end as u64),
                    "{name}: {c}"
                );
                assert_eq!(constraint.im_pol, c > rules.len(), "{name}: {c}");
            }
            // Each rule's expression is not 0 on any row out of its domain: the check reads its
            // domain's rows and no other.
            let numerators = oracle.numerators(&oracle.values(&f.witness, 0).unwrap()).unwrap();
            for (j, rule) in rules.iter().enumerate() {
                let out: Vec<usize> = (0..n).filter(|i| !rule.rows().contains(i)).collect();
                assert!(out.iter().all(|&i| !numerators[j][i].is_zero()), "{name}: rule {j} out of {:?}", rule.rows());
            }

            for (j, _) in rules.iter().enumerate() {
                for edge in [Edge::First, Edge::Last] {
                    let (witness, row) = domains::broken(air, j, edge);
                    let report = check(&pk, &witness, &CheckOptions { max_rows: n }).unwrap();
                    let expected: BTreeSet<(usize, usize, String)> = oracle
                        .check(&oracle.values(&witness, 0).unwrap())
                        .unwrap()
                        .into_iter()
                        .map(|failure| (failure.constraint, failure.row, failure.value.to_bytes().to_decimal()))
                        .collect();
                    assert_eq!(expected.iter().map(|(c, r, _)| (*c, *r)).collect::<Vec<_>>(), [(j, row)], "{name}");
                    assert_eq!(found(&report), expected, "{name}: rule {j}, {edge:?}");
                }
            }
        }
    }
}

/// The CLI names the constraint of a domain and its edge row: `Domains`, with `x0·x0 − p0` on
/// `firstRow` broken at row 0 and `x2' − x2·x2 − K` on `everyFrame {1, 2}` at its last row, N − 3.
#[test]
fn the_cli_names_the_edge_row_of_a_domain() {
    use domains::{Air, Edge};
    let f = fixture_of("domains_cli", Program::Domains(Air::All), domains::witness(Air::All), 9, false);
    let shape = ProvingKey::load(&f.proving_key).unwrap().witness_shape().unwrap();
    let out = check_cli(&f.proving_key, &f.witness_dir, &[]);
    assert!(out.status.success(), "{}", output(&out));

    for (rule, edge, expected) in [
        (
            0,
            Edge::First,
            ["Constraint #0 (stage 1) has 1 invalid rows -> Domains: x0*x0 - p0 == 0", "Failed at row 0 "],
        ),
        (
            2,
            Edge::Last,
            ["Constraint #2 (stage 1) has 1 invalid rows -> Domains: x2' - x2*x2 - K == 0", "Failed at row 13 "],
        ),
    ] {
        let (witness, _) = domains::broken(Air::All, rule, edge);
        let dir = f.dir.file(&format!("broken_{rule}"));
        witness.write(&dir, &shape).unwrap();
        let out = check_cli(&f.proving_key, &dir, &[]);
        let text = output(&out);
        assert_eq!(out.status.code(), Some(1), "{text}");
        for expected in expected {
            assert!(text.contains(expected), "{expected:?} not in:\n{text}");
        }
        assert_eq!(text.matches("Failed at row").count(), 1, "{text}");
    }
}

/// The challenges of stage 2 of `check`, by hand (pilfflonk/docs/README.md#pilfflonk-check,
/// `_verify_proof_constraints`): a transcript that absorbs the STARK's `dummy_element`
/// `[0, 1, 2, r − 1]` and squeezes two.
fn fixed_element_challenges() -> Vec<FrBytes> {
    let r = num_bigint::BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap();
    let minus_one = FrBytes::from_decimal(&(r - 1u32).to_string()).unwrap();
    let mut t = PilFflonkTranscript::new().unwrap();
    let dummy = [FrBytes::ZERO, FrBytes::from_u64(1), FrBytes::from_u64(2), minus_one];
    t.absorb_fr(&dummy.map(|v| v.to_le_bytes())).unwrap();
    (0..2).map(|_| FrBytes::from_le_bytes(t.squeeze().unwrap()).unwrap()).collect()
}

/// The last constraint of the sum bus and of the product bus: its running sum is 0 at the last
/// row, and its running product 1.
const SUM_BUS_LAST: &str = "__L1__'*(0-gsum) == 0";
const PROD_BUS_LAST: &str = "__L1__'*(1-gprod) == 0";

/// A term of a bus that is 0 at row 5: its busid, and the hint whose denominator it is in (see
/// [`checks_a_bus_with_fixed_challenges`]).
struct ZeroTerm {
    busid: u64,
    hint: &'static str,
}

/// A stage-2 fixture of the std's buses, grouped and with `--no-packing`: the check takes the
/// challenges of stage 2 from fixed elements, as the STARK's `verify-constraints` does, and commits
/// nothing: they are those of a transcript of `[0, 1, 2, r − 1]`, and the columns it checks, the
/// `im_col` ones too, the oracle's with them. `witness` passes, and every constraint is checked, of
/// the stages `stages`, the bus's of stage 2 too, whose last is `line`; `broken`, which breaks the
/// bus, fails the last row of
/// its running sum or product, `L1'·…` at row N − 1, and nothing else, with the oracle's value; and,
/// with `zero`, the first term of the bus made 0 at row 5, a denominator 0 on a row is
/// `Unsatisfied`, naming the hint whose it is. Twice the same witness, twice the same report.
fn checks_a_bus_with_fixed_challenges(
    name: &str,
    program: Program,
    (witness, broken): (Witness, Witness),
    stages: &[u64],
    line: &str,
    zero: Option<ZeroTerm>,
) {
    let challenges = fixed_element_challenges();
    for no_packing in [false, true] {
        let name = format!("{name}_{no_packing}");
        let f = fixture_of(&name, program, witness.clone(), DEFAULT_MAX_CONSTRAINT_DEGREE, no_packing);
        let pk = ProvingKey::load(&f.proving_key).unwrap();
        let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
        let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
        let info = pk.air(witness.instances[0].air).unwrap();
        let n = 1usize << info.n_bits;
        let oracle_values = |w: &Witness| {
            let mut values = oracle.values(w, 0).unwrap();
            values.challenges[1] = challenges.iter().map(Fr::from).collect();
            oracle.fill_hint_columns(&mut values, 2).unwrap();
            values
        };

        // The columns the check checks, and its challenges.
        let columns = check_columns(&pk, &f.witness).unwrap();
        assert_eq!(columns.challenges, std::slice::from_ref(&challenges), "{name}: the fixed elements' challenges");
        let values = oracle_values(&witness);
        for p in info.cm_pols_map.iter().filter(|p| p.stage <= info.n_stages) {
            let checked: Vec<Fr> =
                columns.columns[p.stage as usize - 1][p.stage_pos as usize].iter().map(Fr::from).collect();
            let expected = if p.im_pol {
                oracle.expression_rows(&values, p.exp_id.unwrap() as usize).unwrap()
            } else {
                values.witness[p.stage as usize - 1][p.stage_id as usize].clone()
            };
            assert_eq!(checked, expected, "{name}: column {} of stage {}", p.name, p.stage);
        }

        let report = check(&pk, &f.witness, &CheckOptions::default()).unwrap();
        assert!(report.holds(), "{name}");
        let checked: BTreeSet<u64> = report.constraints.iter().map(|c| c.stage).collect();
        assert_eq!(checked, stages.iter().copied().collect(), "{name}");
        let last = report.constraints.iter().position(|c| c.line.ends_with(line)).unwrap();
        assert_eq!(report.constraints[last].stage, 2, "{name}");

        let options = CheckOptions { max_rows: n };
        let report = check(&pk, &broken, &options).unwrap();
        assert_eq!(check(&pk, &broken, &options).unwrap(), report, "{name}: the same witness, the same report");
        let expected: BTreeSet<(usize, usize, String)> = oracle
            .check(&oracle_values(&broken))
            .unwrap()
            .into_iter()
            .map(|failure| (failure.constraint, failure.row, failure.value.to_bytes().to_decimal()))
            .collect();
        assert_eq!(expected.iter().map(|(c, r, _)| (*c, *r)).collect::<Vec<_>>(), [(last, n - 1)], "{name}");
        assert_eq!(found(&report), expected, "{name}");

        // A denominator 0 on a row, with the check's challenges: the first term's (a, …)
        // compressed, busid + a·α + e·α², plus γ, is 0 at row 5, e the second expression.
        let Some(ZeroTerm { busid, hint }) = &zero else { continue };
        let (alpha, gamma) = (Fr::from(&challenges[0]), Fr::from(&challenges[1]));
        let mut zeroed = witness.clone();
        let e = Fr::from(&zeroed.instances[0].stage1.get(5, 1).unwrap());
        let busid = Fr::from_u64(*busid);
        let a = -&(&(&(&busid + &(&e * &(&alpha * &alpha))) + &gamma) * &alpha.inv().unwrap());
        zeroed.instances[0].stage1.set(5, 0, a.to_bytes()).unwrap();
        match check(&pk, &zeroed, &options) {
            Err(PilfflonkError::Unsatisfied(message)) => {
                assert!(message.contains(&format!("{hint} is 0 at row 5")), "{name}: {message}")
            }
            other => panic!("{name}: expected Unsatisfied, got {:?}", other.map(|_| ())),
        }
    }
}

/// The stage-2 fixtures of the std's buses, with [`checks_a_bus_with_fixed_challenges`]: a lookup
/// on the sum bus, with an `im_col` (the std's default `MAX_CONSTRAINT_DEGREE`) and without (4),
/// and a permutation on the product bus, without `im_col` and split by selectors, with two chained
/// ones. The term made 0 at row 5 is the first one: its busid (prod_bus_im's row 5 is of opid 2,
/// sa[5] = 0) and the hint whose denominator it is in (the sum bus's lookup is a direct term of
/// gsum_col's, not an im_col's).
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_check_takes_the_challenges_of_stage_2_from_fixed_elements() {
    let sum = || (sum_bus::witness(), sum_bus::witness_looking_up_what_is_not_provided());
    let zero = |busid, hint| Some(ZeroTerm { busid, hint });
    checks_a_bus_with_fixed_challenges(
        "sum_bus",
        SUM_BUS,
        sum(),
        &[1, 2],
        SUM_BUS_LAST,
        zero(1, "(gsum_col, column gsum)"),
    );
    checks_a_bus_with_fixed_challenges(
        "sum_bus_degree4",
        SUM_BUS_DEGREE4,
        sum(),
        &[1, 2],
        SUM_BUS_LAST,
        zero(1, "(gsum_col, column gsum)"),
    );
    checks_a_bus_with_fixed_challenges(
        "prod_bus",
        PROD_BUS,
        (prod_bus::witness(), prod_bus::witness_not_a_permutation()),
        &[1, 2],
        PROD_BUS_LAST,
        zero(1, "(gprod_col, column gprod)"),
    );
    checks_a_bus_with_fixed_challenges(
        "prod_bus_im",
        PROD_BUS_IM,
        (prod_bus_im::witness(), prod_bus_im::witness_with_a_pair_in_the_other_permutation()),
        &[1, 2],
        PROD_BUS_LAST,
        zero(2, "(im_col, column im_low)"),
    );
}

/// The pil-fflonk examples ported to PIL2 (pilfflonk/docs/README.md#fixtures), on the std's sum bus
/// and on its product bus, with [`checks_a_bus_with_fixed_challenges`]: the check's stage-2 columns
/// are the oracle's, and a wrong multiplicity (Plookup, `all`), a broken permutation or connection,
/// or a value out of the range fails the last row of the bus, which the check names, and nothing
/// else. The Connection and the range check on the sum bus have no constraint of stage 1: the std
/// adds none for them.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_check_finds_a_broken_bus_in_the_pil_fflonk_examples() {
    // Each example, its programs on the sum and the product bus, and the witnesses of each: one that
    // holds and one that breaks the bus.
    type Witnesses = fn() -> (Witness, Witness);
    let plookup: Witnesses = || (plookup::witness(), plookup::witness_with_a_wrong_multiplicity());
    let permutation: Witnesses = || (permutation::witness(), permutation::witness_not_a_permutation());
    let connection: Witnesses = || (connection::witness(), connection::witness_not_connected());
    let range_check_sum: Witnesses = || (range_check::sum_witness(), range_check::sum_witness_out_of_range());
    let range_check_prod: Witnesses = || (range_check::prod_witness(), range_check::prod_witness_out_of_range());
    let all: Witnesses = || (all::witness(), all::witness_with_a_wrong_multiplicity());
    let (both, second): (&[u64], &[u64]) = (&[1, 2], &[2]);
    for (name, (sum, on_sum, sum_stages), (prod, on_prod, prod_stages)) in [
        ("plookup", (PLOOKUP_SUM, plookup, both), (PLOOKUP_PROD, plookup, both)),
        ("permutation", (PERMUTATION_SUM, permutation, both), (PERMUTATION_PROD, permutation, both)),
        ("connection", (CONNECTION_SUM, connection, second), (CONNECTION_PROD, connection, second)),
        ("range_check", (RANGE_CHECK_SUM, range_check_sum, second), (RANGE_CHECK_PROD, range_check_prod, both)),
        ("all", (ALL_SUM, all, both), (ALL_PROD, all, both)),
    ] {
        checks_a_bus_with_fixed_challenges(&format!("{name}_sum"), sum, on_sum(), sum_stages, SUM_BUS_LAST, None);
        checks_a_bus_with_fixed_challenges(&format!("{name}_prod"), prod, on_prod(), prod_stages, PROD_BUS_LAST, None);
    }
}

/// The CLI on a broken bus: the `sum_bus` fixture, and pil-fflonk's `all` on the sum and the
/// product bus. The generator's witness passes; one that breaks the bus (a lookup of a pair the
/// table does not provide, a wrong multiplicity) fails the bus's last row, which it names, of
/// stage 2, at row N − 1; twice the same output, the challenges being fixed.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_cli_names_the_constraint_of_a_broken_bus() {
    let sum_bus = (sum_bus::witness(), sum_bus::witness_looking_up_what_is_not_provided());
    let all = || (all::witness(), all::witness_with_a_wrong_multiplicity());
    for (name, program, (witness, broken), air, (file, line), n) in [
        ("sum_bus_cli", SUM_BUS, sum_bus, "SumBus", ("std_sum.pil:", SUM_BUS_LAST), 32),
        ("all_sum_cli", ALL_SUM, all(), "All", ("std_sum.pil:", SUM_BUS_LAST), 256),
        ("all_prod_cli", ALL_PROD, all(), "All", ("std_prod.pil:", PROD_BUS_LAST), 256),
    ] {
        let f = fixture_of(name, program, witness, DEFAULT_MAX_CONSTRAINT_DEGREE, false);
        let shape = ProvingKey::load(&f.proving_key).unwrap().witness_shape().unwrap();
        let out = check_cli(&f.proving_key, &f.witness_dir, &[]);
        assert!(out.status.success(), "{name}: {}", output(&out));
        let verified = format!("✓ All constraints for Instance #0 of {air} were verified");
        assert!(output(&out).contains(&verified), "{name}: {}", output(&out));

        let dir = f.dir.file("broken");
        broken.write(&dir, &shape).unwrap();
        let out = check_cli(&f.proving_key, &dir, &[]);
        let text = output(&out);
        assert_eq!(out.status.code(), Some(1), "{name}: {text}");
        for expected in [
            &format!("(stage 2) has 1 invalid rows -> {file}"),
            line,
            &format!("✗ Failed at row {} with value: ", n - 1),
            &format!("✗ Not all constraints for Instance #0 of {air} were verified"),
        ] {
            assert!(text.contains(expected), "{name}: {expected:?} not in:\n{text}");
        }
        assert_eq!(text.matches("Failed at row").count(), 1, "{name}: {text}");
        println!("{text}");
        // The line of the failed row, without the log's timestamp.
        let failed = |text: &str| text.lines().find_map(|l| l.find("Failed at row").map(|at| l[at..].to_string()));
        let again = check_cli(&f.proving_key, &dir, &[]);
        assert_eq!(failed(&output(&again)), failed(&text), "{name}: the same witness, the same value");
    }
}

/// The lines of the constraints in the output of `check -v`, without the log's timestamps.
fn constraint_lines(text: &str) -> Vec<String> {
    text.lines().filter_map(|l| l.find("Constraint #").map(|at| l[at..].to_string())).collect()
}

/// The witness a library computes (pilfflonk/docs/README.md#witness): `check --witness-lib` passes
/// on that of the libraries of the Fibonacci (with pil-fflonk's inputs), the Connection and `all`
/// (on the sum bus), constraint by constraint as `check --witness` on their generators' witness
/// directories.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_cli_checks_the_witness_a_library_computes() {
    // Each fixture, set up with its generator's witness, and its library, AIR and public inputs.
    type Setup = fn() -> Fixture;
    let fibonacci: Setup = || fixture("lib_fibonacci");
    let connection: Setup =
        || fixture_of("lib_connection", CONNECTION_SUM, connection::witness(), DEFAULT_MAX_CONSTRAINT_DEGREE, false);
    let all: Setup = || fixture_of("lib_all", ALL_SUM, all::witness(), DEFAULT_MAX_CONSTRAINT_DEGREE, false);
    let inputs = Some(r#"{"in1": "1", "in2": "2"}"#);
    for (setup, library, air, inputs) in [
        (fibonacci, "pilfflonk_fibonacci", "Fibonacci", inputs),
        (connection, "pilfflonk_connection", "Connection", None),
        (all, "pilfflonk_all", "All", inputs),
    ] {
        let f = setup();
        let library = built_library(library);
        let mut source = vec!["--witness-lib", library.to_str().unwrap()];
        let inputs_file = f.dir.file("inputs.json");
        if let Some(inputs) = inputs {
            fs::write(&inputs_file, inputs).unwrap();
            source.extend(["--public-inputs", inputs_file.to_str().unwrap()]);
        }
        let out = check_from(&f.proving_key, &source, &["-v"]);
        let text = output(&out);
        assert!(out.status.success(), "{air}: {text}");
        let verified = format!("✓ All constraints for Instance #0 of {air} were verified");
        assert!(text.contains(&verified), "{air}: {text}");

        let from_dir = check_cli(&f.proving_key, &f.witness_dir, &["-v"]);
        assert!(from_dir.status.success(), "{air}: {}", output(&from_dir));
        assert!(!constraint_lines(&text).is_empty(), "{air}: {text}");
        assert_eq!(constraint_lines(&text), constraint_lines(&output(&from_dir)), "{air}");
    }
}

/// `check --witness-lib` refuses a STARK witness library and a library that is not there, as
/// `prove --witness-lib` does (`pilfflonk_prove.rs`), with an error and no report.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_cli_refuses_what_is_not_a_pilfflonk_witness_library() {
    let f = fixture("lib_refusals");
    let stark = built_library("fibonacci_square");
    let missing = f.dir.file("missing.so");
    for (library, expected) in [
        (&stark, format!("witness library {}: it is a STARK witness library", stark.display())),
        (&missing, format!("witness library {}: there is no such file", missing.display())),
    ] {
        let out = check_from(&f.proving_key, &["--witness-lib", library.to_str().unwrap()], &[]);
        let text = output(&out);
        assert_eq!(out.status.code(), Some(1), "{text}");
        assert!(text.contains(&expected), "{expected:?} not in:\n{text}");
        assert!(!text.contains("Instance #0"), "{text}");
    }
}

/// The flags of the witness, as `prove`'s (`pilfflonk_prove.rs`): exactly one of `--witness` and
/// `--witness-lib`, and `--public-inputs` only with `--witness-lib`. clap refuses the others, with
/// exit code 2, before anything is read.
#[test]
fn the_cli_takes_one_witness() {
    let key = Path::new("provingKey");
    for (source, expected) in [
        (
            &["--witness", "witness", "-w", "library.so"][..],
            "the argument '--witness <WITNESS>' cannot be used with '--witness-lib <WITNESS_LIB>'",
        ),
        (
            &[],
            "the following required arguments were not provided:\n  <--witness <WITNESS>|--witness-lib <WITNESS_LIB>>",
        ),
        (
            &["--witness", "witness", "-i", "inputs.json"],
            "the argument '--witness <WITNESS>' cannot be used with '--public-inputs <PUBLIC_INPUTS>'",
        ),
        (&["--public-inputs", "inputs.json"], "<--witness <WITNESS>|--witness-lib <WITNESS_LIB>>"),
    ] {
        let out = check_from(key, source, &[]);
        assert_eq!(out.status.code(), Some(2), "{source:?}: {}", output(&out));
        assert!(output(&out).contains(expected), "{expected:?} not in:\n{}", output(&out));
    }
}
