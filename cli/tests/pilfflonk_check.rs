//! `proofman-cli pilfflonk check` (spec §4.4, "Depuració"; plan M25) on the Fibonacci fixture: the
//! witness of M13's generator passes, and a mutated one fails exactly the constraints and rows the
//! Rust oracle (M14) says, with its values, through the library (`proofman_pilfflonk::check`) and
//! the CLI, whose output is `verify-constraints`'s (plan validation 3).
//!
//! Pilouts are not versioned: the test compiles the fixture with the compiler `PIL2C_EXEC` names,
//! which must honour `prime`, and is `#[ignore]` without it:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only \
//!     --test pilfflonk_check -- --ignored
//! ```

#[path = "../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pil2_pilout::pilout_proxy::PilOutProxy;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman_pilfflonk::oracle::{AirOracle, Fr};
use proofman_pilfflonk::{check, CheckOptions, CheckReport, FileWitnessSource, FrBytes, ProvingKey, Witness};

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

/// Compiles the Fibonacci fixture over BN254 to `pilout` with `PIL2C_EXEC`.
fn compile_fibonacci(pilout: &Path) {
    let compiler = std::env::var("PIL2C_EXEC").expect("PIL2C_EXEC must name a pil2com that honours `prime`");
    let out = Command::new(compiler)
        .current_dir(repo_root())
        .arg("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil")
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
    /// M13's generator's witness for [1, 2], and its directory.
    witness: Witness,
    witness_dir: PathBuf,
}

/// The Fibonacci compiled and set up (`--no-packing`; the check commits nothing, so the ptau of
/// `τ = 1` does), and its witness written.
fn fixture(name: &str) -> Fixture {
    let dir = TestDir::new(name);
    let opts = SetupPilfflonkOptions {
        airout_path: dir.file("fibonacci.pilout"),
        build_dir: dir.file("build"),
        powers_of_tau: dir.file("tau_one.ptau"),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: true,
    };
    compile_fibonacci(&opts.airout_path);
    write_tau_one_ptau(&opts.powers_of_tau, 512).unwrap();
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let witness = fibonacci::witness(8, [1, 2]);
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

    // Mutations: the oracle's own cases (M14), the publics, and random ones of one cell and of several.
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

    // The oracle's cases, as M14 lists them.
    let pairs = |report: &CheckReport| -> Vec<(usize, u64)> {
        report.failures().flat_map(|c| c.failed_rows.iter().map(|r| (c.index, r.row))).collect()
    };
    assert_eq!(pairs(&check(&pk, &cases[0].1, &all_rows).unwrap()), [(0, 100), (1, 99), (1, 100)]);
    assert_eq!(pairs(&check(&pk, &cases[2].1, &all_rows).unwrap()), [(0, 0), (1, 0), (3, 0)]);
    assert_eq!(pairs(&check(&pk, &cases[9].1, &all_rows).unwrap()), [(4, 255)]);
}

fn check_cli(key: &Path, witness: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_proofman-cli"))
        .args(["pilfflonk", "check", "-k", key.to_str().unwrap(), "--witness", witness.to_str().unwrap()])
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

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn a_constraint_of_stage_2_is_refused() {
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
    assert!(err.contains("(plan M30)"), "{err}");
    let out = check_cli(&f.proving_key, &f.witness_dir, &[]);
    assert_eq!(out.status.code(), Some(1), "{}", output(&out));
    assert!(output(&out).contains(expected), "{}", output(&out));
}
