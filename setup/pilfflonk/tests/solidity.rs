//! The Solidity verifier on Foundry (spec §4.5, plan M40): `setup-pilfflonk --solidity` writes
//! `pilfflonk.verifier.sol` for real keys, solc 0.8.37 compiles it, and Foundry runs it on their
//! proofs and on the same proofs mutated, where it must say what the JS verifier (D8, the
//! reference) says.
//!
//! For each key, the test:
//! 1. sets it up with `--solidity` and checks that `proofman-setup pilfflonk-solidity` writes the
//!    same verifier from the vkey alone;
//! 2. compiles the verifier with solc (no warning, and within EIP-170's 24576 bytes);
//! 3. proves a witness with a fixed blinding seed, and replays the transcript (A.4) to encode the
//!    calldata: the proof's bytes and the auxiliary inverses of `firstRow` and `lastRow`
//!    (`CalldataLayout`, spec §4.5 "Calldata");
//! 4. makes the cases: the proof; the proof with an evaluation, a commitment, a public or `W'`
//!    changed, the first three also "fixed up" (`fixup`: `invZh`, `inv` and the auxiliary inverses
//!    recomputed for the changed transcript, as the M40 review's harness does), so that they get
//!    to the pairing, or to `checkQPieces` if `Q` is split; split, the pieces of `Q` changed with
//!    their sum kept (to the pairing) and one changed (to `checkQPieces`); points off the curve or
//!    that the transcript refuses; and values only the calldata can hold (a coordinate `≥ q`, a
//!    scalar `≥ r`, a wrong auxiliary inverse, calldata a word short);
//! 5. asks the JS verifier about each case (`js_verifier::verify`), and runs Foundry on all of them
//!    (`pilfflonk/solidity`, copied to a directory of its own): `verifyProof` must return what the
//!    JS verifier says, `false` for every calldata-only case but the short one, which reverts;
//! 6. prints the gas of every `verifyProof` call, and the gas of its calldata; the fixed-up case
//!    that reaches the pairing must cost about what the proof does.
//!
//! A key no pilout gives, a split `Q` and no evaluation, is made by hand with its proof
//! ([`foundry_verifies_a_split_q_without_evaluations`]).
//!
//! The tools are pinned (spec §4.5): Foundry v1.8.3 and solc 0.8.37, at the paths `PILFFLONK_FORGE`
//! and `PILFFLONK_SOLC` name. The tests are `#[ignore]`d without them; those of the compiled
//! fixtures also need `PIL2C_EXEC`, a compiler that honours `prime`. All need Node.js:
//!
//! ```text
//! PILFFLONK_FORGE=<forge> PILFFLONK_SOLC=<solc> PIL2C_EXEC=<pil2-compiler>/src/pil.js \
//!     cargo test -p pilfflonk-setup --features proofman-starks-lib-c/cpu-only --test solidity -- \
//!     --ignored --nocapture
//! ```

#[allow(dead_code)]
#[path = "../../../pilfflonk/tests/data/all.rs"]
mod all;
#[allow(dead_code)]
#[path = "../../../pilfflonk/tests/data/connection.rs"]
mod connection;
#[allow(dead_code)]
#[path = "../../../pilfflonk/tests/data/domains.rs"]
mod domains;
#[allow(dead_code)]
#[path = "../../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;
#[allow(dead_code)]
#[path = "../../../pilfflonk/tests/data/packed.rs"]
mod packed;
#[allow(dead_code)]
#[path = "../../../pilfflonk/tests/data/permutation.rs"]
mod permutation;
#[allow(dead_code)]
#[path = "../../../pilfflonk/tests/data/plookup.rs"]
mod plookup;
#[allow(dead_code)]
#[path = "../../../pilfflonk/tests/data/signed.rs"]
mod signed;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard, PoisonError};

use num_bigint::BigUint;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::digest::seal_vkey;
use pilfflonk_setup::solidity::{export_verifier_sol, CalldataLayout, VERIFIER_SOL_FILE};
use pilfflonk_setup::test_ptau::{g1_times, g2_times, test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use prost::Message;
use proofman_pilfflonk::{
    js_verifier, prove, FqBytes, FrBytes, G1Affine, JsonFile, PilfflonkGlobalInfo, Proof, ProofNames, ProveOptions,
    ProvingKey, Publics, Vkey, Witness, BN254_Q, BN254_R,
};
use proofman_starks_lib_c::PilFflonkTranscript;
use serde_json::{json, Value};

/// The blinding seed of the proofs (D6: fixed in tests).
const SEED: [u8; 32] = [0x5a; 32];

/// EIP-170: the largest runtime code of a contract.
const MAX_CODE_SIZE: usize = 24576;

/// Held by each test for its whole run: they call the C++ core's OpenMP code (the setup, the
/// prover), which must not run from several test threads at once (plan M26).
fn cpp_core() -> MutexGuard<'static, ()> {
    static CPP_CORE: Mutex<()> = Mutex::new(());
    CPP_CORE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Foundry's `forge` and solc, pinned (spec §4.5), from `PILFFLONK_FORGE` and `PILFFLONK_SOLC`.
struct Tools {
    forge: PathBuf,
    solc: PathBuf,
}

impl Tools {
    fn from_env() -> Self {
        let path = |var: &str| {
            let path =
                PathBuf::from(std::env::var_os(var).unwrap_or_else(|| panic!("{var} must name the pinned tool")));
            assert!(path.is_file(), "{var} = {} is not a file", path.display());
            path
        };
        Tools { forge: path("PILFFLONK_FORGE"), solc: path("PILFFLONK_SOLC") }
    }
}

/// A fresh directory for the test under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir =
            Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_solidity_{name}_{}", std::process::id()));
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn output(out: &std::process::Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// A program of the tests: a compiled fixture, or a pilout of `domains.rs`, built in code.
#[derive(Clone, Copy, Debug)]
enum Program {
    Fibonacci,
    /// Six fixed columns in an `f`, eleven committed ones in `f` of `k = 3, 4, 4` (`powerW = 12`),
    /// and fusions (plan M22).
    Packed,
    /// Offsets `{−1, 0, 1, 2}` and constraints of degree 6 (plan M23).
    Signed,
    /// `all` on the std's sum bus (plan M34): stage 2, a bus, 9 `f`.
    AllSum,
    Domains(domains::Air),
}

impl Program {
    fn compile(self, pilout: &Path) {
        let pil = match self {
            Program::Fibonacci => "pilfflonk/tests/fixtures/fibonacci/fibonacci.pil",
            Program::Packed => "pilfflonk/tests/fixtures/packed/packed.pil",
            Program::Signed => "pilfflonk/tests/fixtures/signed/signed.pil",
            Program::AllSum => "pilfflonk/tests/fixtures/all/all_sum.pil",
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

    fn witness(self) -> Witness {
        match self {
            Program::Fibonacci => fibonacci::witness(8, [1, 2]),
            // The inputs of cli/tests/pilfflonk_prove.rs.
            Program::Packed => packed::witness(5),
            Program::Signed => signed::witness([3, 5]),
            Program::AllSum => all::witness(),
            Program::Domains(air) => domains::witness(air),
        }
    }

    /// More powers than the largest degree of its layouts: `all`'s is 2312 (plan M34).
    fn ptau_powers(self) -> usize {
        match self {
            Program::AllSum => 4096,
            Program::Fibonacci | Program::Packed | Program::Signed | Program::Domains(_) => 1024,
        }
    }
}

/// A key of the tests: a program, set up as these options say.
struct Setup {
    name: &'static str,
    program: Program,
    no_packing: bool,
    max_q_degree: u64,
}

/// A key set up with `--solidity`, and its verifier.
struct Key {
    dir: TestDir,
    proving_key: PathBuf,
    vkey_path: PathBuf,
    vkey: Vkey,
    sol: PathBuf,
}

impl Key {
    fn new(setup: &Setup) -> Self {
        let dir = TestDir::new(setup.name);
        let ptau = dir.file("fixed_tau.ptau");
        write_fixed_tau_ptau(&ptau, setup.program.ptau_powers(), &test_tau()).unwrap();
        let opts = SetupPilfflonkOptions {
            airout_path: dir.file("program.pilout"),
            build_dir: dir.file("build"),
            powers_of_tau: ptau,
            max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
            extra_muls: DEFAULT_EXTRA_MULS,
            max_q_degree: setup.max_q_degree,
            no_packing: setup.no_packing,
            solidity: true,
        };
        setup.program.compile(&opts.airout_path);
        run_setup_pilfflonk(&opts).unwrap();
        let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
        let global_info = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
        let vkey_path = global_info.vkey_path(&proving_key);
        let sol = global_info.backend_dir(&proving_key).join(VERIFIER_SOL_FILE);
        assert!(sol.is_file(), "--solidity writes {}", sol.display());

        // The same verifier from the vkey alone.
        let exported = dir.file("exported.sol");
        export_verifier_sol(&vkey_path, &exported).unwrap();
        assert_eq!(fs::read(&exported).unwrap(), fs::read(&sol).unwrap(), "pilfflonk-solidity and --solidity");

        let vkey = Vkey::read(&vkey_path).unwrap();
        Key { dir, proving_key, vkey_path, vkey, sol }
    }
}

fn r() -> BigUint {
    BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap()
}

fn q() -> BigUint {
    BigUint::parse_bytes(BN254_Q.as_bytes(), 10).unwrap()
}

fn big(value: &FrBytes) -> BigUint {
    BigUint::from_bytes_le(&value.to_le_bytes())
}

fn plus_one(value: &FrBytes) -> FrBytes {
    FrBytes::from_decimal(&((big(value) + 1u32) % r()).to_string()).unwrap()
}

/// The 32 big-endian bytes of `value < 2^256`.
fn be_word(value: &BigUint) -> [u8; 32] {
    let bytes = value.to_bytes_be();
    let mut word = [0u8; 32];
    word[32 - bytes.len()..].copy_from_slice(&bytes);
    word
}

/// `s mod r` as a scalar.
fn fr(s: &BigUint) -> FrBytes {
    FrBytes::from_decimal(&(s % r()).to_str_radix(10)).unwrap()
}

fn fr_inv(a: &BigUint) -> BigUint {
    a.modpow(&(r() - 2u32), &r())
}

/// `a − b mod r`.
fn fr_sub(a: &BigUint, b: &BigUint) -> BigUint {
    (a + r() - (b % r())) % r()
}

/// `5^((r−1)/n)`, a primitive `n`-th root of unity (`shplonk.js`, `rootOfUnity`).
fn root_of_unity(n: u64) -> BigUint {
    BigUint::from(5u32).modpow(&((r() - 1u32) / n), &r())
}

/// The challenges the calldata and the fixups need, the transcript of A.4 replayed on rapidsnark's
/// (the verifier's sequence, `challenges.js`).
struct Challenges {
    xi_seed: BigUint,
    y: BigUint,
}

/// The challenges of a proof. `None` if the transcript refuses a point of the proof (A.4): off the
/// curve, or with a coordinate below 2^192.
fn challenges(vkey: &Vkey, proof: &Proof, publics: &[FrBytes]) -> Option<Challenges> {
    let le = |values: &[FrBytes]| values.iter().map(FrBytes::to_le_bytes).collect::<Vec<_>>();
    let mut t = PilFflonkTranscript::new().ok()?;
    t.absorb_fr(&[vkey.digest.to_fr().to_le_bytes(), FrBytes::from_u64(1).to_le_bytes()]).ok()?;
    if !publics.is_empty() {
        t.absorb_fr(&le(publics)).ok()?;
    }
    let layout = &vkey.layout.0;
    let n_fixed = vkey.layout.n_fixed();
    let q_stage = layout.last()?.stage;
    let absorb = |t: &mut PilFflonkTranscript, stage: u64| -> Option<()> {
        let points: Vec<[u8; 64]> = proof
            .commitments
            .iter()
            .zip(&layout[n_fixed..])
            .filter(|(_, f)| f.stage == stage)
            .map(|(p, _)| p.to_le_bytes())
            .collect();
        if !points.is_empty() {
            t.absorb_g1(&points).ok()?;
        }
        Some(())
    };
    for s in 1..q_stage {
        absorb(&mut t, s)?;
        if s + 1 < q_stage {
            for _ in 0..vkey.num_challenges[s as usize] {
                t.squeeze().ok()?;
            }
        }
    }
    t.squeeze().ok()?;
    absorb(&mut t, q_stage)?;
    let xi_seed = BigUint::from_bytes_le(&t.squeeze().ok()?);
    // The evaluations, the pieces of Q with them (Proof::evaluations), in the proof's order.
    if !proof.evaluations.is_empty() {
        t.absorb_fr(&le(&proof.evaluations)).ok()?;
    }
    t.squeeze().ok()?;
    t.absorb_g1(&[proof.w.to_le_bytes()]).ok()?;
    let y = BigUint::from_bytes_le(&t.squeeze().ok()?);
    Some(Challenges { xi_seed, y })
}

/// `ξ = xiSeed^powerW`.
fn xi_of(vkey: &Vkey, xi_seed: &BigUint) -> BigUint {
    xi_seed.modpow(&BigUint::from(vkey.power_w), &r())
}

/// The roots `T` of an `f` of `k` polynomials and these offsets, offset-major (`shplonk.js`,
/// `computeRoots`): `x_j = xiSeed^(powerW/k)·ω_{kN}^s·w_k^j`.
fn roots(vkey: &Vkey, k: u64, offsets: &[i64], xi_seed: &BigUint) -> Vec<BigUint> {
    let r = r();
    let seed = xi_seed.modpow(&BigUint::from(vkey.power_w / k), &r);
    let (omega_kn, w_k) = (root_of_unity(k << vkey.power), root_of_unity(k));
    let mut t = Vec::new();
    for &s in offsets {
        let power = omega_kn.modpow(&BigUint::from(s.unsigned_abs()), &r);
        let mut x = &seed * if s < 0 { fr_inv(&power) } else { power } % &r;
        for _ in 0..k {
            t.push(x.clone());
            x = x * &w_k % &r;
        }
    }
    t
}

/// `proof` with `invZh` and `inv` recomputed for its own transcript (A.5; `shplonk.js`,
/// `computeZerofiers` and `computeInverseDenominators`, as `review40/harness.mjs` of the M40 review
/// does), so that a mutated proof, with the auxiliary inverses of its `ξ` (`calldata`), gets past
/// those checks to the deeper ones: `checkQPieces` and the pairing.
fn fixup(vkey: &Vkey, mut proof: Proof, publics: &[FrBytes]) -> Proof {
    let ch = challenges(vkey, &proof, publics).expect("a proof whose points the transcript absorbs");
    let r = r();
    let xi = xi_of(vkey, &ch.xi_seed);
    proof.inv_zh = fr(&fr_inv(&fr_sub(&xi.modpow(&BigUint::from(1u64 << vkey.power), &r), &BigUint::from(1u32))));
    let mut product = BigUint::from(1u32);
    for (i, f) in vkey.layout.0.iter().enumerate() {
        let t = roots(vkey, f.k, &f.offsets, &ch.xi_seed);
        if i > 0 {
            product = t.iter().fold(product, |z, x| z * fr_sub(&ch.y, x) % &r);
        }
        for (m, x) in t.iter().enumerate() {
            let den = t
                .iter()
                .enumerate()
                .filter(|&(l, _)| l != m)
                .fold(fr_sub(&ch.y, x), |d, (_, xl)| d * fr_sub(x, xl) % &r);
            product = product * den % &r;
        }
    }
    proof.inv = fr(&fr_inv(&product));
    proof
}

/// The auxiliary inverses of the calldata (`CalldataLayout::aux_rows`): `1/(ξ − ω^j)`, with `ω` the
/// `N`-th root of unity of the domain.
fn aux_inverses(vkey: &Vkey, xi_seed: &BigUint) -> Vec<BigUint> {
    let r = r();
    let xi = xi_of(vkey, xi_seed);
    let omega = root_of_unity(1u64 << vkey.power);
    CalldataLayout::of(vkey)
        .aux_rows
        .iter()
        .map(|&j| fr_inv(&fr_sub(&xi, &omega.modpow(&BigUint::from(j), &r))))
        .collect()
}

/// What `verifyProof` must do with a case: return `true`, return `false`, or revert (the ABI
/// decoder, on calldata shorter than its arguments).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Accept,
    Reject,
    Revert,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Outcome::Accept => "accept",
            Outcome::Reject => "reject",
            Outcome::Revert => "revert",
        }
    }

    fn of_js(verdict: bool) -> Self {
        if verdict {
            Outcome::Accept
        } else {
            Outcome::Reject
        }
    }
}

/// A case: a proof and its publics, what the JS verifier says of them, and their calldata.
struct Case {
    label: String,
    /// `None` for a case of the calldata alone, which the JS verifier does not see.
    js: Option<bool>,
    expected: Outcome,
    proof: Vec<u8>,
    publics: Vec<u8>,
}

/// The calldata of `proof` and `publics` (spec §4.5, "Calldata"): the proof's bytes and the
/// auxiliary inverses of its own `ξ`, and the publics as words. A proof whose points the transcript
/// refuses (A.4) has no `ξ`: its auxiliary inverses are 0, and the verifier refuses the point first.
fn calldata(vkey: &Vkey, proof: &Proof, publics: &[FrBytes]) -> (Vec<u8>, Vec<u8>) {
    let mut words = proof.to_bytes();
    let layout = CalldataLayout::of(vkey);
    if !layout.aux_rows.is_empty() {
        let aux = match challenges(vkey, proof, publics) {
            Some(ch) => aux_inverses(vkey, &ch.xi_seed),
            None => vec![BigUint::ZERO; layout.aux_rows.len()],
        };
        for aux in aux {
            words.extend_from_slice(&be_word(&aux));
        }
    }
    assert_eq!(words.len() as u64, 32 * layout.words());
    (words, publics.iter().flat_map(|p| p.to_be_bytes()).collect())
}

/// What the JS verifier says of the proof and publics files in `dir`.
fn js_verdict_of(vkey_path: &Path, dir: &Path) -> bool {
    js_verifier::verify(vkey_path, &dir.join("publics.json"), &dir.join("proof.json")).unwrap()
}

/// What the JS verifier says of `proof` and `publics`, from their files in `dir`.
fn js_verdict(key: &Key, names: &ProofNames, dir: &Path, proof: &Proof, publics: &[FrBytes]) -> bool {
    fs::create_dir_all(dir).unwrap();
    proof.to_json(names).unwrap().write(&dir.join("proof.json")).unwrap();
    Publics(publics.to_vec()).write(&dir.join("publics.json")).unwrap();
    js_verdict_of(&key.vkey_path, dir)
}

/// The verifier at `sol` compiled as the Foundry project compiles it, in `dir`: no warning, and its
/// runtime code, whose size it returns, within EIP-170.
fn compile_with_solc(tools: &Tools, dir: &Path, sol: &Path) -> usize {
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

/// Runs the Foundry project, in `dir`, on `cases` with the verifier at `sol`, and returns what each
/// call did and its gas.
fn run_foundry(tools: &Tools, dir: &Path, sol: &Path, cases: &[Case]) -> Vec<(Outcome, u64)> {
    let project = dir.join("foundry");
    let template = repo_root().join("pilfflonk/solidity");
    for sub in ["src", "test", "cases"] {
        fs::create_dir_all(project.join(sub)).unwrap();
    }
    fs::copy(template.join("foundry.toml"), project.join("foundry.toml")).unwrap();
    fs::copy(template.join("test/PilfflonkVerifier.t.sol"), project.join("test/PilfflonkVerifier.t.sol")).unwrap();
    fs::copy(sol, project.join("src/PilfflonkVerifier.sol")).unwrap();
    let hex = |bytes: &[u8]| format!("0x{}", bytes.iter().map(|b| format!("{b:02x}")).collect::<String>());
    let cases_json = json!({
        "n": cases.len(),
        "cases": cases.iter().map(|c| json!({
            "label": c.label,
            "proof": hex(&c.proof),
            "publics": hex(&c.publics),
            "expected": c.expected.as_str(),
        })).collect::<Vec<Value>>(),
    });
    fs::write(project.join("cases/cases.json"), cases_json.to_string()).unwrap();
    let _ = fs::remove_file(project.join("cases/results.txt"));

    let out = Command::new(&tools.forge)
        .args(["test", "--offline", "--root"])
        .arg(&project)
        .env("FOUNDRY_SOLC", &tools.solc)
        .output()
        .expect("PILFFLONK_FORGE runs");
    assert!(out.status.success(), "forge test: {}", output(&out));
    let results = fs::read_to_string(project.join("cases/results.txt")).unwrap();
    let outcomes: Vec<(Outcome, u64)> = results
        .lines()
        .map(|line| {
            let fields: Vec<&str> = line.split(' ').collect();
            let outcome = match fields[1] {
                "accept" => Outcome::Accept,
                "reject" => Outcome::Reject,
                "revert" => Outcome::Revert,
                other => panic!("an outcome {other}"),
            };
            (outcome, fields[2].parse().unwrap())
        })
        .collect();
    assert_eq!(outcomes.len(), cases.len(), "{results}");
    outcomes
}

/// The gas of the calldata of a call with these arguments (EIP-2028): 16 a non-zero byte, 4 a zero.
fn calldata_gas(case: &Case) -> u64 {
    let selector = [0xffu8; 4];
    selector.iter().chain(&case.proof).chain(&case.publics).map(|&b| if b == 0 { 4 } else { 16 }).sum()
}

/// Runs `cases` on Foundry, prints a line per case, checks each outcome and returns their gas.
fn check_on_foundry(tools: &Tools, dir: &Path, sol: &Path, name: &str, cases: &[Case]) -> Vec<u64> {
    let outcomes = run_foundry(tools, dir, sol, cases);
    let mut gas = Vec::with_capacity(cases.len());
    for (case, (outcome, used)) in cases.iter().zip(outcomes) {
        let js = case.js.map_or("-", |v| if v { "accept" } else { "reject" });
        println!(
            "  {:<36} JS {js:<6} Solidity {:<6} verifyProof gas {used:>7}, calldata gas {:>6}",
            case.label,
            outcome.as_str(),
            calldata_gas(case)
        );
        assert_eq!(outcome, case.expected, "{} on {name}", case.label);
        gas.push(used);
    }
    gas
}

/// The position of the piece `Q<i>` among the proof's evaluations, if `Q` is split: after the
/// evMap's, in the order of the layout (A.6).
fn piece_position(vkey: &Vkey, piece: u64) -> Option<usize> {
    let q_stage = vkey.layout.0.last()?.stage;
    let names: Vec<&str> = vkey
        .layout
        .0
        .iter()
        .filter(|f| f.stage == q_stage)
        .flat_map(|f| f.pols.iter().map(|p| p.name.as_str()))
        .collect();
    if names.len() < 2 {
        return None;
    }
    let name = format!("Q{piece}");
    names.iter().position(|n| *n == name).map(|at| vkey.ev_map.len() + at)
}

/// Adds to `proof`'s split `Q` a multiple of `ξ^(M·N)·Q_1 − Q_0` that leaves `Σ_i ξ^(i·M·N)·Q_i(ξ)`
/// as it is: `Q_0 += ξ^(M·N)·d`, `Q_1 −= d` (A.1). `checkQPieces` passes, and the pairing refuses it.
fn rebalance_pieces(vkey: &Vkey, proof: &mut Proof, xi_seed: &BigUint, d: u64) {
    let (Some(q0), Some(q1)) = (piece_position(vkey, 0), piece_position(vkey, 1)) else { return };
    let shift = xi_of(vkey, xi_seed).modpow(&BigUint::from(vkey.max_q_degree << vkey.power), &r());
    proof.evaluations[q0] = fr(&(big(&proof.evaluations[q0]) + shift * d));
    proof.evaluations[q1] = fr(&fr_sub(&big(&proof.evaluations[q1]), &BigUint::from(d)));
}

/// Sets up, proves, and runs the cases of `setup` on the JS verifier and on Foundry (see the
/// module); prints a line per case.
fn verify_on_foundry(tools: &Tools, setup: &Setup) {
    let key = Key::new(setup);
    let size = compile_with_solc(tools, &key.dir.0, &key.sol);
    let pk = ProvingKey::load(&key.proving_key).unwrap();
    let options = ProveOptions { insecure_blinding_seed: Some(SEED), q_part_bits: None };
    let out = prove(&pk, &setup.program.witness(), &options).unwrap();
    let (proof, publics, names) = (&out.proof, &out.publics.0, &out.names);
    let vkey = &key.vkey;
    // The transcript the encoder replays is the prover's, and so are the inv and invZh of fixup.
    let ch = challenges(vkey, proof, publics).unwrap();
    assert_eq!(fr(&ch.xi_seed), out.challenges.xi_seed);
    assert_eq!(&fixup(vkey, proof.clone(), publics), proof, "fixup of the prover's proof is the proof");

    let mut cases = Vec::new();
    let mut add = |label: &str, proof: &Proof, publics: &[FrBytes], expected: bool| {
        let js = js_verdict(&key, names, &key.dir.file(label), proof, publics);
        assert_eq!(js, expected, "the JS verifier on {label} of {}", setup.name);
        let (proof, publics) = calldata(vkey, proof, publics);
        cases.push(Case { label: label.to_string(), js: Some(js), expected: Outcome::of_js(js), proof, publics });
    };
    add("proof", proof, publics, true);
    let mut evaluation = proof.clone();
    evaluation.evaluations[0] = plus_one(&evaluation.evaluations[0]);
    add("mutated evaluation", &evaluation, publics, false);
    // With invZh, inv and the auxiliary inverses of its own transcript: to the pairing if Q is
    // whole, to checkQPieces if it is split (Q(ξ) is not the pieces').
    add("mutated evaluation, fixed up", &fixup(vkey, evaluation, publics), publics, false);
    // Another point of the curve, whose coordinates the transcript absorbs: W.
    let mut commitment = proof.clone();
    commitment.commitments[0] = commitment.w;
    add("mutated commitment", &commitment, publics, false);
    add("mutated commitment, fixed up", &fixup(vkey, commitment, publics), publics, false);
    if !publics.is_empty() {
        let mut public = publics.clone();
        public[0] = plus_one(&public[0]);
        add("mutated public", proof, &public, false);
        add("mutated public, fixed up", &fixup(vkey, proof.clone(), &public), &public, false);
    }
    // Split, the pieces of Q changed but not their sum: past checkQPieces, to the pairing; and one
    // piece changed, refused by checkQPieces.
    if piece_position(vkey, 1).is_some() {
        let mut pieces = proof.clone();
        rebalance_pieces(vkey, &mut pieces, &ch.xi_seed, 12345);
        add("Q pieces rebalanced, fixed up", &fixup(vkey, pieces, publics), publics, false);
        let mut piece = proof.clone();
        let q1 = piece_position(vkey, 1).unwrap();
        piece.evaluations[q1] = plus_one(&piece.evaluations[q1]);
        add("Q1 + 1, fixed up", &fixup(vkey, piece, publics), publics, false);
    }
    // W' is not absorbed: only the pairing sees it.
    let mut wp = proof.clone();
    wp.wp = wp.w;
    add("mutated W'", &wp, publics, false);
    // Not a point of the curve (elements.js, g1FromObject).
    let mut off_curve = proof.clone();
    let y = BigUint::from_bytes_le(&off_curve.commitments[0].y.to_le_bytes());
    off_curve.commitments[0].y = FqBytes::from_decimal(&((y + 1u32) % q()).to_string()).unwrap();
    add("commitment off the curve", &off_curve, publics, false);
    // A point the transcript cannot absorb (transcript.js, addPolCommitment): G = (1, 2).
    let mut short = proof.clone();
    short.commitments[0] = G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(2) };
    add("commitment (1, 2)", &short, publics, false);

    // What only the calldata can hold, which is no proof (the proof's bytes are canonical, A.6): a
    // coordinate x + q, a scalar e + r and, when there are any, a wrong auxiliary inverse, which
    // verifyProof refuses with false; and calldata one word short, which the ABI decoder reverts.
    let layout = CalldataLayout::of(vkey);
    let good = (cases[0].proof.clone(), cases[0].publics.clone());
    let mut calldata_only = |label: &str, word: u64, add: &BigUint| {
        let (mut proof, publics) = good.clone();
        let at = 32 * word as usize;
        let value = BigUint::from_bytes_be(&proof[at..at + 32]) + add;
        proof[at..at + 32].copy_from_slice(&be_word(&value));
        cases.push(Case { label: label.into(), js: None, expected: Outcome::Reject, proof, publics });
    };
    calldata_only("coordinate x + q", 0, &q());
    calldata_only("evaluation + r", 2 * (layout.n_commitments + 2), &r());
    if !layout.aux_rows.is_empty() {
        // 1/(ξ − ω^j) + 1 mod r: below r, and not the inverse.
        let at = 32 * layout.proof_words() as usize;
        let aux = BigUint::from_bytes_be(&good.0[at..at + 32]);
        let one = if aux == r() - 1u32 { r() - aux } else { BigUint::from(1u32) };
        calldata_only("wrong auxiliary inverse", layout.proof_words(), &one);
    }
    let (mut short_proof, mut short_publics) = good.clone();
    if short_publics.is_empty() {
        short_proof.truncate(short_proof.len() - 32);
    } else {
        short_publics.truncate(short_publics.len() - 32);
    }
    cases.push(Case {
        label: "calldata a word short".into(),
        js: None,
        expected: Outcome::Revert,
        proof: short_proof,
        publics: short_publics,
    });

    println!(
        "{}: {} f, powerW {}, runtime code {size} bytes, {} words of `proof`",
        setup.name,
        vkey.layout.0.len(),
        vkey.power_w,
        layout.words()
    );
    let gas = check_on_foundry(tools, &key.dir.0, &key.sol, setup.name, &cases);
    // The fixed-up cases get past invZh, inv and the auxiliary inverses: the one that reaches the
    // pairing costs about what the proof does.
    let deep = if piece_position(vkey, 1).is_some() {
        "Q pieces rebalanced, fixed up"
    } else {
        "mutated evaluation, fixed up"
    };
    let at = cases.iter().position(|c| c.label == deep).unwrap();
    assert!(
        10 * gas[at] >= 9 * gas[0],
        "{deep} of {} stops before the pairing: {} gas, the proof {}",
        setup.name,
        gas[at],
        gas[0]
    );
}

/// The Fibonacci grouped (its fixed columns in an `f` of `k = 2`) and with `--no-packing`, the
/// fixtures of the packing and of the signed offsets (`Q` split in three, too), and the example
/// `all` on the sum bus (stage 2, `powerW = 72`), with `Q` whole and split in two.
#[test]
#[ignore = "needs PILFFLONK_FORGE, PILFFLONK_SOLC, PIL2C_EXEC and Node.js"]
fn foundry_verifies_the_proofs_of_the_fixtures_as_the_js_verifier_does() {
    let _cpp = cpp_core();
    let tools = Tools::from_env();
    let setups = [
        Setup { name: "fibonacci", program: Program::Fibonacci, no_packing: false, max_q_degree: DEFAULT_MAX_Q_DEGREE },
        Setup {
            name: "fibonacci_no_packing",
            program: Program::Fibonacci,
            no_packing: true,
            max_q_degree: DEFAULT_MAX_Q_DEGREE,
        },
        Setup { name: "packed", program: Program::Packed, no_packing: false, max_q_degree: DEFAULT_MAX_Q_DEGREE },
        Setup { name: "signed", program: Program::Signed, no_packing: false, max_q_degree: DEFAULT_MAX_Q_DEGREE },
        Setup { name: "signed_q_split", program: Program::Signed, no_packing: false, max_q_degree: 1 },
        Setup { name: "all_sum", program: Program::AllSum, no_packing: false, max_q_degree: DEFAULT_MAX_Q_DEGREE },
        Setup { name: "all_sum_q_split", program: Program::AllSum, no_packing: false, max_q_degree: 1 },
    ];
    for setup in &setups {
        verify_on_foundry(&tools, setup);
    }
}

/// The zerofiers of every domain (A.1, plan M24): one rule of each, `firstRow` and `lastRow`
/// among them, whose calldata has auxiliary inverses; and six `everyFrame`, at offsets −1 and 1,
/// with `--no-packing` too.
#[test]
#[ignore = "needs PILFFLONK_FORGE, PILFFLONK_SOLC and Node.js"]
fn foundry_verifies_the_proofs_of_every_domain() {
    let _cpp = cpp_core();
    let tools = Tools::from_env();
    let setups = [
        Setup {
            name: "domains",
            program: Program::Domains(domains::Air::All),
            no_packing: false,
            max_q_degree: DEFAULT_MAX_Q_DEGREE,
        },
        Setup {
            name: "frames",
            program: Program::Domains(domains::Air::Frames),
            no_packing: false,
            max_q_degree: DEFAULT_MAX_Q_DEGREE,
        },
        Setup {
            name: "frames_no_packing",
            program: Program::Domains(domains::Air::Frames),
            no_packing: true,
            max_q_degree: DEFAULT_MAX_Q_DEGREE,
        },
    ];
    for setup in &setups {
        verify_on_foundry(&tools, setup);
    }
}

/// A split `Q` and no evaluation (the M40 review's `mk_noeval.mjs`): one `f`, `Q`'s, of `k = 2` with
/// its pieces in the order `Q1`, `Q0`, so that the calldata's scalars start with `Q1`, and the
/// transcript absorbs the pieces from there, not from `Q0` (spec §4.5). No pilout gives such a key:
/// the vkey and its proof are made here with the test ptau's `τ`, for the statement `public = 5`
/// (`Q = (p − 5)/Z_H` is then 0, and its pieces only PLONK's blinding, A.1): `f(X) = Q1(X²) +
/// X·Q0(X²)` with `Q0 = b0·X^N + b1·X^(N+1)` and `Q1 = −b0 − b1·X`, committed as `f(τ)·G`, and `W`
/// and `W'` as A.5 defines them for one `f`.
#[test]
#[ignore = "needs PILFFLONK_FORGE, PILFFLONK_SOLC and Node.js"]
fn foundry_verifies_a_split_q_without_evaluations() {
    let _cpp = cpp_core();
    let tools = Tools::from_env();
    let dir = TestDir::new("noeval");
    let tau = test_tau();
    let r = r();
    let n: u64 = 1 << 8;
    let tmp = |id: u64| json!({"type": "tmp", "id": id, "dim": 1});
    let vkey_json = json!({
        "protocol": "pilfflonk", "curve": "bn128", "formatVersion": 1, "nPublic": 1, "power": 8, "powerW": 2,
        "X_2": g2_times(&tau), "numChallenges": [0], "evMap": [],
        "layout": [{
            "stage": 2, "pols": [{"id": 0, "name": "Q1"}, {"id": 1, "name": "Q0"}], "k": 2, "offsets": [0],
            "degree": 2 * n + 4,
        }],
        "boundaries": [{"name": "everyRow"}], "qDeg": 2, "maxQDegree": 1,
        "qVerifier": {"tmpUsed": 2, "code": [
            {"op": "sub", "dest": tmp(0), "src": [{"type": "public", "id": 0, "dim": 1}, {"type": "number", "value": "5", "dim": 1}]},
            {"op": "mul", "dest": tmp(1), "src": [tmp(0), {"type": "Zi", "boundaryId": 0, "dim": 1}]},
        ]},
        "digest": format!("0x{}", "0".repeat(64)),
    });
    let vkey = seal_vkey(Vkey::from_json_str(&vkey_json.to_string()).unwrap()).unwrap();
    let vkey_path = dir.file("pilfflonk.vkey.json");
    vkey.write(&vkey_path).unwrap();
    let sol = dir.file("pilfflonk.verifier.sol");
    export_verifier_sol(&vkey_path, &sol).unwrap();
    let size = compile_with_solc(&tools, &dir.0, &sol);

    // The proof of public = 5.
    let publics = vec![FrBytes::from_u64(5)];
    let (b0, b1) = (BigUint::from(7u32), BigUint::from(11u32));
    let pow = |b: &BigUint, e: u64| b.modpow(&BigUint::from(e), &r);
    let q0 = |x: &BigUint| (&b0 * pow(x, n) + &b1 * pow(x, n + 1)) % &r;
    let q1 = |x: &BigUint| fr_sub(&BigUint::ZERO, &((&b0 + &b1 * x) % &r));
    let f_tau = (q1(&pow(&tau, 2)) + &tau * q0(&pow(&tau, 2))) % &r;
    let mut proof = Proof {
        commitments: vec![g1_times(&f_tau)],
        // Any W for now: the transcript absorbs it after xiSeed.
        w: g1_times(&f_tau),
        wp: g1_times(&BigUint::from(1u32)),
        evaluations: vec![],
        air_values: vec![],
        airgroup_values: vec![],
        proof_values: vec![],
        inv: FrBytes::ZERO,
        inv_zh: FrBytes::ZERO,
    };
    let xi = xi_of(&vkey, &challenges(&vkey, &proof, &publics).unwrap().xi_seed);
    let (q1_xi, q0_xi) = (q1(&xi), q0(&xi));
    // The pieces in the order of the layout: Q1, Q0.
    proof.evaluations = vec![fr(&q1_xi), fr(&q0_xi)];
    // r(X) = Q1(ξ) + X·Q0(ξ) interpolates f on T = {±xiSeed}, the roots of Z_T = X² − ξ; h = (f − r)/Z_T.
    let r_at = |x: &BigUint| (&q1_xi + x * &q0_xi) % &r;
    let h_tau = fr_sub(&f_tau, &r_at(&tau)) * fr_inv(&fr_sub(&pow(&tau, 2), &xi)) % &r;
    proof.w = g1_times(&h_tau);
    let y = challenges(&vkey, &proof, &publics).unwrap().y;
    // W' = L/(X − y) at τ, with L = f − r(y) − Z_T(y)·h: q_0 = Z_T(y) for one f (A.5).
    let l_tau = fr_sub(&fr_sub(&f_tau, &r_at(&y)), &(fr_sub(&pow(&y, 2), &xi) * &h_tau % &r));
    proof.wp = g1_times(&(l_tau * fr_inv(&fr_sub(&tau, &y)) % &r));
    let proof = fixup(&vkey, proof, &publics);

    // The JSON view, by hand: no pilfflonkinfo names this proof (A.6: f0, W, Wp; Q1, Q0, inv, invZh).
    let point = |p: &G1Affine| json!([p.x.to_decimal(), p.y.to_decimal(), "1"]);
    let js = |label: &str, proof: &Proof, publics: &[FrBytes]| {
        let at = dir.file(label);
        fs::create_dir_all(&at).unwrap();
        let view = json!({
            "protocol": "pilfflonk", "curve": "bn128",
            "polynomials": {"f0": point(&proof.commitments[0]), "W": point(&proof.w), "Wp": point(&proof.wp)},
            "evaluations": {
                "Q1": proof.evaluations[0].to_decimal(), "Q0": proof.evaluations[1].to_decimal(),
                "inv": proof.inv.to_decimal(), "invZh": proof.inv_zh.to_decimal(),
            },
        });
        fs::write(at.join("proof.json"), view.to_string()).unwrap();
        Publics(publics.to_vec()).write(&at.join("publics.json")).unwrap();
        js_verdict_of(&vkey_path, &at)
    };
    let mut cases = Vec::new();
    let mut add = |label: &str, proof: &Proof, publics: &[FrBytes], expected: bool| {
        let verdict = js(label, proof, publics);
        assert_eq!(verdict, expected, "the JS verifier on {label}");
        let (proof, publics) = calldata(&vkey, proof, publics);
        cases.push(Case { label: label.into(), js: Some(verdict), expected: Outcome::of_js(verdict), proof, publics });
    };
    add("proof", &proof, &publics, true);
    let mut piece = proof.clone();
    piece.evaluations[0] = plus_one(&piece.evaluations[0]);
    add("Q1 + 1, fixed up", &fixup(&vkey, piece, &publics), &publics, false);
    let mut pieces = proof.clone();
    rebalance_pieces(&vkey, &mut pieces, &challenges(&vkey, &proof, &publics).unwrap().xi_seed, 12345);
    add("Q pieces rebalanced, fixed up", &fixup(&vkey, pieces, &publics), &publics, false);
    let other = vec![FrBytes::from_u64(6)];
    add("public 6, fixed up", &fixup(&vkey, proof.clone(), &other), &other, false);

    println!(
        "noeval: 1 f, powerW 2, runtime code {size} bytes, {} words of `proof`",
        CalldataLayout::of(&vkey).words()
    );
    let gas = check_on_foundry(&tools, &dir.0, &sol, "noeval", &cases);
    // The rebalanced pieces pass checkQPieces: the pairing refuses them.
    assert!(
        10 * gas[2] >= 9 * gas[0],
        "the rebalanced pieces stop before the pairing: {} gas, the proof {}",
        gas[2],
        gas[0]
    );
}
