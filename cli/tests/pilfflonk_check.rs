//! `proofman-cli pilfflonk check` (spec §4.4, "Depuració"; plan M25) on the Fibonacci fixture: the
//! witness of M13's generator passes, and a mutated one fails exactly the constraints and rows the
//! Rust oracle (M14) says, with its values, through the library (`proofman_pilfflonk::check`) and
//! the CLI, whose output is `verify-constraints`'s (plan validation 3). And on the fixture of the
//! signed offsets (plan M23), whose constraints read the rows −1 to 2 around each row, across the
//! wrap too, with the im pols the setup chooses for each `--max-constraint-degree`. And on the
//! pilouts of `tests/data/domains.rs`, built in code (plan M24): each constraint is checked on the
//! rows of its domain only, `firstRow ≤ i < lastRow`, and a witness that breaks it at the edge row
//! of its domain is found there.
//!
//! Pilouts are not versioned: the test compiles the fixture with the compiler `PIL2C_EXEC` names,
//! which must honour `prime`, and is `#[ignore]` without it. Those of the domains build their
//! pilouts in code, and always run:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only \
//!     --test pilfflonk_check -- --ignored
//! ```

#[path = "../../pilfflonk/tests/data/domains.rs"]
mod domains;
#[path = "../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;
#[path = "../../pilfflonk/tests/data/signed.rs"]
mod signed;

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
use prost::Message;

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
    /// M13's generator's witness for [1, 2], and its directory.
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
    };
    compile(program, &opts.airout_path);
    write_tau_one_ptau(&opts.powers_of_tau, 512).unwrap();
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
