//! The Solidity verifier on Foundry, for the tests (pilfflonk/docs/verifier.md#tests): the pinned
//! tools, solc's compilation of a verifier, and runs of the Foundry project `pilfflonk/solidity` on
//! cases, each the calldata of a call to `verifyProof` and what the call must do. The Solidity tests
//! of `pilfflonk-setup` (`setup/pilfflonk/tests/solidity.rs`) and the CLI's end to end
//! (`cli/tests/pilfflonk_prove.rs`) share it. The differential fuzzer (`fuzz.rs`,
//! pilfflonk/docs/verifier.md#differential-fuzzer) runs the project's other test,
//! `fuzz/PilfflonkFuzz.t.sol`, with [`FuzzProject`]: the verifier and its probe on every case,
//! whose outcomes it reads and does not check.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/foundry.rs"] mod foundry;`.
//!
//! The tools are pinned (pilfflonk/docs/verifier.md#tools): Foundry v1.8.3 and solc 0.8.37, at the
//! paths `PILFFLONK_FORGE` and `PILFFLONK_SOLC` name. Foundry downloads nothing: the project's
//! `foundry.toml` is `offline`, and `FOUNDRY_SOLC` gives it the pinned solc, so nothing goes to
//! `~/.svm`. The project is copied to a directory of the test's for each run: nothing is built in the
//! repository.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{json, Value};

/// EIP-170: the largest runtime code of a contract.
pub const MAX_CODE_SIZE: usize = 24576;

/// Foundry's `forge` and solc, pinned, from `PILFFLONK_FORGE` and `PILFFLONK_SOLC`.
pub struct Tools {
    pub forge: PathBuf,
    pub solc: PathBuf,
}

impl Tools {
    pub fn from_env() -> Self {
        let path = |var: &str| {
            let path =
                PathBuf::from(std::env::var_os(var).unwrap_or_else(|| panic!("{var} must name the pinned tool")));
            assert!(path.is_file(), "{var} = {} is not a file", path.display());
            path
        };
        Tools { forge: path("PILFFLONK_FORGE"), solc: path("PILFFLONK_SOLC") }
    }
}

/// What `verifyProof` must do with a case: return `true`, return `false`, or revert (the ABI
/// decoder, on calldata shorter than its arguments). `BadReturn`, a call that returns something
/// other than a bool, is never what a case must do: the fuzzer's Foundry test reports it, and the
/// verifier's (`test/PilfflonkVerifier.t.sol`) fails on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Accept,
    Reject,
    Revert,
    BadReturn,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Accept => "accept",
            Outcome::Reject => "reject",
            Outcome::Revert => "revert",
            Outcome::BadReturn => "badreturn",
        }
    }

    fn parse(word: &str) -> Self {
        match word {
            "accept" => Outcome::Accept,
            "reject" => Outcome::Reject,
            "revert" => Outcome::Revert,
            "badreturn" => Outcome::BadReturn,
            other => panic!("an outcome {other}"),
        }
    }

    pub fn of_js(verdict: bool) -> Self {
        if verdict {
            Outcome::Accept
        } else {
            Outcome::Reject
        }
    }
}

/// A case: the calldata of a call to `verifyProof`, what the JS verifier says of its proof and
/// publics, and what the call must do.
pub struct Case {
    pub label: String,
    /// `None` for a case of the calldata alone, which the JS verifier does not see.
    pub js: Option<bool>,
    pub expected: Outcome,
    /// The selector of `verifyProof` and its arguments, ABI-encoded: what `proofman-cli pilfflonk
    /// calldata --format hex` writes (`Calldata::to_abi_bytes`). The Foundry test checks that the
    /// selector is solc's, and makes the call with these bytes.
    pub calldata: Vec<u8>,
}

fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// The Foundry project of the tests, `pilfflonk/solidity` of the workspace this test is built in.
fn project_template() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let found =
        manifest.ancestors().map(|dir| dir.join("pilfflonk/solidity")).find(|dir| dir.join("foundry.toml").is_file());
    found.unwrap_or_else(|| panic!("no pilfflonk/solidity/foundry.toml above {}", manifest.display()))
}

/// The verifier at `sol` compiled as the Foundry project compiles it, in `dir`: no warning, and its
/// runtime code, whose size it returns, within EIP-170.
pub fn compile_with_solc(tools: &Tools, dir: &Path, sol: &Path) -> usize {
    let out_dir = dir.join("solc");
    let out = Command::new(&tools.solc)
        .args(["--optimize", "--optimize-runs", "200", "--bin-runtime", "--overwrite", "-o"])
        .arg(&out_dir)
        .arg(sol)
        .output()
        .expect("PILFFLONK_SOLC runs");
    assert!(out.status.success(), "solc: {}", output(&out));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("Warning") && !stderr.contains("Error"), "solc: {stderr}");
    let hex = fs::read_to_string(out_dir.join("PilfflonkVerifier.bin-runtime")).unwrap();
    let size = hex.trim().len() / 2;
    assert!(size <= MAX_CODE_SIZE, "the verifier's runtime code has {size} bytes, above EIP-170's {MAX_CODE_SIZE}");
    size
}

/// A copy of the Foundry project in `dir/<name>`: its `foundry.toml`, the test `test` of the project
/// (a path in it) in `test/`, and each source `(name, contents)` in `src/`; and an empty `cases/`.
fn write_project(dir: &Path, name: &str, test: &str, sources: &[(&str, &[u8])]) -> PathBuf {
    let project = dir.join(name);
    let template = project_template();
    for sub in ["src", "test", "cases"] {
        fs::create_dir_all(project.join(sub)).unwrap();
    }
    fs::copy(template.join("foundry.toml"), project.join("foundry.toml")).unwrap();
    let test_file = Path::new(test).file_name().unwrap();
    fs::copy(template.join(test), project.join("test").join(test_file)).unwrap();
    for (name, contents) in sources {
        fs::write(project.join("src").join(name), contents).unwrap();
    }
    project
}

/// `forge test` in `project`, offline and with the pinned solc (see the module), with `env` set too;
/// it must pass.
fn forge_test(tools: &Tools, project: &Path, env: &[(&str, String)]) {
    let out = Command::new(&tools.forge)
        .args(["test", "--offline", "--root"])
        .arg(project)
        .env("FOUNDRY_SOLC", &tools.solc)
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .output()
        .expect("PILFFLONK_FORGE runs");
    assert!(out.status.success(), "forge test: {}", output(&out));
}

/// Runs the Foundry project, in `dir`, on `cases` with the verifier at `sol`, and returns what each
/// call did and its gas.
fn run_foundry(tools: &Tools, dir: &Path, sol: &Path, cases: &[Case]) -> Vec<(Outcome, u64)> {
    let verifier = fs::read(sol).unwrap();
    let project =
        write_project(dir, "foundry", "test/PilfflonkVerifier.t.sol", &[("PilfflonkVerifier.sol", &verifier)]);
    let hex = |bytes: &[u8]| format!("0x{}", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>());
    let cases_json = json!({
        "n": cases.len(),
        "cases": cases.iter().map(|c| json!({
            "label": c.label,
            "calldata": hex(&c.calldata),
            "expected": c.expected.as_str(),
        })).collect::<Vec<Value>>(),
    });
    fs::write(project.join("cases/cases.json"), cases_json.to_string()).unwrap();
    let _ = fs::remove_file(project.join("cases/results.txt"));

    forge_test(tools, &project, &[]);
    let results = fs::read_to_string(project.join("cases/results.txt")).unwrap();
    let outcomes: Vec<(Outcome, u64)> = results
        .lines()
        .map(|line| {
            let fields: Vec<&str> = line.split(' ').collect();
            (Outcome::parse(fields[1]), fields[2].parse().unwrap())
        })
        .collect();
    assert_eq!(outcomes.len(), cases.len(), "{results}");
    outcomes
}

/// The gas of the calldata of a case (EIP-2028): 16 a non-zero byte, 4 a zero.
pub fn calldata_gas(case: &Case) -> u64 {
    case.calldata.iter().map(|&b| if b == 0 { 4 } else { 16 }).sum()
}

/// Runs `cases` on Foundry, prints `title` and a line per case, all at once (tests may run keys on
/// several threads), checks each outcome and returns their gas.
pub fn check_on_foundry(tools: &Tools, dir: &Path, sol: &Path, title: &str, cases: &[Case]) -> Vec<u64> {
    let outcomes = run_foundry(tools, dir, sol, cases);
    let mut report = format!("{title}\n");
    for (case, (outcome, used)) in cases.iter().zip(&outcomes) {
        let js = case.js.map_or("-", |v| if v { "accept" } else { "reject" });
        report.push_str(&format!(
            "  {:<36} JS {js:<6} Solidity {:<6} verifyProof gas {used:>7}, calldata gas {:>6}\n",
            case.label,
            outcome.as_str(),
            calldata_gas(case)
        ));
    }
    print!("{report}");
    for (case, (outcome, _)) in cases.iter().zip(&outcomes) {
        assert_eq!(*outcome, case.expected, "{} on {title}", case.label);
    }
    outcomes.into_iter().map(|(_, used)| used).collect()
}

/// What the fuzzer's Foundry test (`fuzz/PilfflonkFuzz.t.sol`) saw of a case: what the verifier's
/// call did and its gas, measured as [`check_on_foundry`] measures it, and the gas of the probe's
/// call and the words it returned (`None` if it reverted).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuzzRun {
    pub outcome: Outcome,
    pub gas: u64,
    pub probe_gas: u64,
    pub probe: Option<Vec<u64>>,
}

/// The Foundry project of the differential fuzzer for one key, in `dir/fuzz`: the key's verifier
/// and its probe (an instrumented copy that the fuzzer makes, `fuzz.rs`), and the test
/// `fuzz/PilfflonkFuzz.t.sol`, which calls both on each case. A project is run on as many batches
/// of cases as the fuzzer likes; solc compiles it once.
pub struct FuzzProject {
    project: PathBuf,
}

impl FuzzProject {
    /// The project of the verifier at `sol` and of the probe `probe`, a contract `PilfflonkProbe`.
    pub fn new(dir: &Path, sol: &Path, probe: &str) -> Self {
        let verifier = fs::read(sol).unwrap();
        let sources: [(&str, &[u8]); 2] =
            [("PilfflonkVerifier.sol", &verifier), ("PilfflonkProbe.sol", probe.as_bytes())];
        FuzzProject { project: write_project(dir, "fuzz", "fuzz/PilfflonkFuzz.t.sol", &sources) }
    }

    /// Runs the test on `cases`, the calldata of each call, as the cases `first`, `first + 1`, …: it
    /// writes each as `cases/<i>.bin`. Foundry runs it without isolation (`FOUNDRY_ISOLATE=false`):
    /// every call is a call of the test's transaction, to a contract it has called before, as the
    /// first call of the verifier's test (`test/PilfflonkVerifier.t.sol`) is with isolation
    /// (Foundry's default); with isolation, Foundry makes each call a transaction of its own, and
    /// every call after the first costs 2500 more (EIP-2929, a cold account), which a gas of the
    /// cases could not be compared with.
    pub fn run(&self, tools: &Tools, first: usize, cases: &[Vec<u8>]) -> Vec<FuzzRun> {
        let dir = self.project.join("cases");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for (i, calldata) in cases.iter().enumerate() {
            fs::write(dir.join(format!("{}.bin", first + i)), calldata).unwrap();
        }
        let env = [
            ("PILFFLONK_FUZZ_FIRST", first.to_string()),
            ("PILFFLONK_FUZZ_N", cases.len().to_string()),
            ("FOUNDRY_ISOLATE", "false".to_string()),
        ];
        forge_test(tools, &self.project, &env);
        let results = fs::read_to_string(dir.join("results.txt")).unwrap();
        let runs: Vec<FuzzRun> = results
            .lines()
            .enumerate()
            .map(|(i, line)| {
                let fields: Vec<&str> = line.split(' ').collect();
                assert_eq!(fields[0], (first + i).to_string(), "{results}");
                let probe = match fields[3] {
                    "revert" => None,
                    "return" => Some(fields[5..].iter().map(|w| w.parse().unwrap()).collect()),
                    other => panic!("a probe's outcome {other}"),
                };
                let (gas, probe_gas) = (fields[2].parse().unwrap(), fields[4].parse().unwrap());
                FuzzRun { outcome: Outcome::parse(fields[1]), gas, probe_gas, probe }
            })
            .collect();
        assert_eq!(runs.len(), cases.len(), "{results}");
        runs
    }
}
