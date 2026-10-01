//! The Solidity verifier on Foundry (spec §4.5, plan M40): `setup-pilfflonk --solidity` writes
//! `pilfflonk.verifier.sol` for real keys, solc 0.8.37 compiles it, and Foundry runs it on their
//! proofs and on the same proofs mutated, where it must say what the JS verifier (D8, the
//! reference) says.
//!
//! For each key, the test:
//! 1. sets it up with `--solidity` and checks that `proofman-setup pilfflonk-solidity` writes the
//!    same verifier from the vkey alone;
//! 2. compiles the verifier with solc (no warning, and within EIP-170's 24576 bytes);
//! 3. proves a witness with a fixed blinding seed, and encodes its calldata with the encoder of
//!    `proofman-cli pilfflonk calldata` (`proofman_pilfflonk::Calldata`, plan M41): the proof's bytes
//!    and the auxiliary inverses of `firstRow` and `lastRow` (spec §4.5 "Calldata"), of the `ξ` of
//!    the transcript (A.4) it replays, which must be the prover's;
//! 4. makes the cases: the proof; the proof with an evaluation, a commitment, a public or `W'`
//!    changed, the first three also "fixed up" (`fixup`: `invZh`, `inv` and the auxiliary inverses
//!    recomputed for the changed transcript, as the M40 review's harness does), so that they get
//!    to the pairing, or to `checkQPieces` if `Q` is split; split, the pieces of `Q` changed with
//!    their sum kept (to the pairing) and one changed (to `checkQPieces`); points off the curve or
//!    that the transcript refuses; and values only the calldata can hold (a coordinate `≥ q`, a
//!    scalar `≥ r`, a wrong auxiliary inverse, calldata a word short);
//! 5. asks the JS verifier about each case (`js_verifier::verify`), and runs Foundry on all of them
//!    (`pilfflonk/solidity`, copied to a directory of its own; `pilfflonk/tests/data/foundry.rs`):
//!    `verifyProof` must return what the JS verifier says, `false` for every calldata-only case but
//!    the short one, which reverts;
//! 6. prints the gas of every `verifyProof` call, and the gas of its calldata; the fixed-up case
//!    that reaches the pairing must cost about what the proof does.
//!
//! A key no pilout gives, a split `Q` and no evaluation, is made by hand with its proof
//! ([`foundry_verifies_a_split_q_without_evaluations`]). The proofs of every fixture of phases 1 to
//! 3, with the calldata of the CLI, are `cli/tests/pilfflonk_prove.rs`'s (plan M41).
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
#[path = "../../../pilfflonk/tests/data/foundry.rs"]
mod foundry;
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
use pilfflonk_setup::solidity::{export_verifier_sol, VERIFIER_SOL_FILE};
use pilfflonk_setup::test_ptau::{g1_times, g2_times, test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use prost::Message;
use proofman_pilfflonk::calldata::SELECTOR_BYTES;
use proofman_pilfflonk::{
    js_verifier, prove, verifier_challenges, Calldata, CalldataLayout, FqBytes, FrBytes, G1Affine, JsonFile,
    PilfflonkGlobalInfo, Proof, ProofNames, ProveOptions, ProvingKey, Publics, Vkey, Witness, BN254_Q, BN254_R,
};
use serde_json::json;

use foundry::{check_on_foundry, compile_with_solc, Case, Outcome, Tools};

/// The blinding seed of the proofs (D6: fixed in tests).
const SEED: [u8; 32] = [0x5a; 32];

/// Held by each test for its whole run: they call the C++ core's OpenMP code (the setup, the
/// prover), which must not run from several test threads at once (plan M26).
fn cpp_core() -> MutexGuard<'static, ()> {
    static CPP_CORE: Mutex<()> = Mutex::new(());
    CPP_CORE.lock().unwrap_or_else(PoisonError::into_inner)
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
    let ch = verifier_challenges(vkey, &proof, publics).expect("a proof whose points the transcript absorbs");
    let (xi_seed, y) = (big(&ch.xi_seed), big(&ch.y));
    let r = r();
    let xi = xi_of(vkey, &xi_seed);
    proof.inv_zh = fr(&fr_inv(&fr_sub(&xi.modpow(&BigUint::from(1u64 << vkey.power), &r), &BigUint::from(1u32))));
    let mut product = BigUint::from(1u32);
    for (i, f) in vkey.layout.0.iter().enumerate() {
        let t = roots(vkey, f.k, &f.offsets, &xi_seed);
        if i > 0 {
            product = t.iter().fold(product, |z, x| z * fr_sub(&y, x) % &r);
        }
        for (m, x) in t.iter().enumerate() {
            let den =
                t.iter().enumerate().filter(|&(l, _)| l != m).fold(fr_sub(&y, x), |d, (_, xl)| d * fr_sub(x, xl) % &r);
            product = product * den % &r;
        }
    }
    proof.inv = fr(&fr_inv(&product));
    proof
}

/// The calldata of `proof` and `publics` (spec §4.5, "Calldata"), ABI-encoded as `proofman-cli
/// pilfflonk calldata --format hex` writes it: `Calldata::encode`, the proof's bytes and the
/// auxiliary inverses of its own `ξ`, and the publics.
fn calldata(vkey: &Vkey, proof: &Proof, publics: &[FrBytes]) -> Vec<u8> {
    Calldata::encode(vkey, proof, publics).unwrap().to_abi_bytes().unwrap()
}

/// The calldata of a proof the encoder refuses, because the transcript refuses one of its points
/// (A.4), which it names: the proof has no `ξ`, and its auxiliary inverses are 0 (the verifier
/// refuses the point first).
fn calldata_without_xi(vkey: &Vkey, proof: &Proof, publics: &[FrBytes], point: &str) -> Vec<u8> {
    let err = Calldata::encode(vkey, proof, publics).unwrap_err().to_string();
    assert!(err.contains(&format!("{point} of the proof is not a point the transcript absorbs")), "{err}");
    let zeros = vec![FrBytes::ZERO; CalldataLayout::of(vkey).aux_rows.len()];
    Calldata::with_auxiliary_inverses(vkey, proof, publics, &zeros).unwrap().to_abi_bytes().unwrap()
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
    // The transcript the encoder replays is the prover's, and so are the inv and invZh of fixup; the
    // names of the proof are the vkey's.
    let ch = verifier_challenges(vkey, proof, publics).unwrap();
    assert_eq!(
        (&ch.stages, ch.std_vc, ch.xi_seed),
        (&out.challenges.stages, out.challenges.std_vc, out.challenges.xi_seed)
    );
    assert_eq!(&fixup(vkey, proof.clone(), publics), proof, "fixup of the prover's proof is the proof");
    assert_eq!(&ProofNames::of_vkey(vkey).unwrap(), names, "the names of the vkey and of the pilfflonkinfo");
    let xi_seed = big(&ch.xi_seed);

    let mut cases = Vec::new();
    // `refused`: the point of the proof the transcript refuses, if any (calldata_without_xi).
    let mut add = |label: &str, proof: &Proof, publics: &[FrBytes], expected: bool, refused: Option<&str>| {
        let js = js_verdict(&key, names, &key.dir.file(label), proof, publics);
        assert_eq!(js, expected, "the JS verifier on {label} of {}", setup.name);
        let calldata = match refused {
            None => calldata(vkey, proof, publics),
            Some(point) => calldata_without_xi(vkey, proof, publics, point),
        };
        cases.push(Case { label: label.to_string(), js: Some(js), expected: Outcome::of_js(js), calldata });
    };
    add("proof", proof, publics, true, None);
    let mut evaluation = proof.clone();
    evaluation.evaluations[0] = plus_one(&evaluation.evaluations[0]);
    add("mutated evaluation", &evaluation, publics, false, None);
    // With invZh, inv and the auxiliary inverses of its own transcript: to the pairing if Q is
    // whole, to checkQPieces if it is split (Q(ξ) is not the pieces').
    add("mutated evaluation, fixed up", &fixup(vkey, evaluation, publics), publics, false, None);
    // Another point of the curve, whose coordinates the transcript absorbs: W.
    let mut commitment = proof.clone();
    commitment.commitments[0] = commitment.w;
    add("mutated commitment", &commitment, publics, false, None);
    add("mutated commitment, fixed up", &fixup(vkey, commitment, publics), publics, false, None);
    if !publics.is_empty() {
        let mut public = publics.clone();
        public[0] = plus_one(&public[0]);
        add("mutated public", proof, &public, false, None);
        add("mutated public, fixed up", &fixup(vkey, proof.clone(), &public), &public, false, None);
    }
    // Split, the pieces of Q changed but not their sum: past checkQPieces, to the pairing; and one
    // piece changed, refused by checkQPieces.
    if piece_position(vkey, 1).is_some() {
        let mut pieces = proof.clone();
        rebalance_pieces(vkey, &mut pieces, &xi_seed, 12345);
        add("Q pieces rebalanced, fixed up", &fixup(vkey, pieces, publics), publics, false, None);
        let mut piece = proof.clone();
        let q1 = piece_position(vkey, 1).unwrap();
        piece.evaluations[q1] = plus_one(&piece.evaluations[q1]);
        add("Q1 + 1, fixed up", &fixup(vkey, piece, publics), publics, false, None);
    }
    // W' is not absorbed: only the pairing sees it.
    let mut wp = proof.clone();
    wp.wp = wp.w;
    add("mutated W'", &wp, publics, false, None);
    // Not a point of the curve (elements.js, g1FromObject), which the transcript refuses: the encoder
    // does too.
    let first = format!("f{}", vkey.layout.n_fixed());
    let mut off_curve = proof.clone();
    let y = BigUint::from_bytes_le(&off_curve.commitments[0].y.to_le_bytes());
    off_curve.commitments[0].y = FqBytes::from_decimal(&((y + 1u32) % q()).to_string()).unwrap();
    add("commitment off the curve", &off_curve, publics, false, Some(&first));
    // A point the transcript cannot absorb (transcript.js, addPolCommitment): G = (1, 2).
    let mut short = proof.clone();
    short.commitments[0] = G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(2) };
    add("commitment (1, 2)", &short, publics, false, Some(&first));

    // What only the calldata can hold, which is no proof (the proof's bytes are canonical, A.6): a
    // coordinate x + q, a scalar e + r and, when there are any, a wrong auxiliary inverse, which
    // verifyProof refuses with false; and calldata one word short, which the ABI decoder reverts.
    let layout = CalldataLayout::of(vkey);
    let good = cases[0].calldata.clone();
    let word_at = |word: u64| SELECTOR_BYTES + 32 * word as usize;
    let mut calldata_only = |label: &str, word: u64, add: &BigUint| {
        let mut calldata = good.clone();
        let at = word_at(word);
        let value = BigUint::from_bytes_be(&calldata[at..at + 32]) + add;
        calldata[at..at + 32].copy_from_slice(&be_word(&value));
        cases.push(Case { label: label.into(), js: None, expected: Outcome::Reject, calldata });
    };
    calldata_only("coordinate x + q", 0, &q());
    calldata_only("evaluation + r", layout.first_scalar(), &r());
    if !layout.aux_rows.is_empty() {
        // 1/(ξ − ω^j) + 1 mod r: below r, and not the inverse.
        let at = word_at(layout.proof_words());
        let aux = BigUint::from_bytes_be(&good[at..at + 32]);
        let one = if aux == r() - 1u32 { r() - aux } else { BigUint::from(1u32) };
        calldata_only("wrong auxiliary inverse", layout.proof_words(), &one);
    }
    let mut short_calldata = good.clone();
    short_calldata.truncate(good.len() - 32);
    cases.push(Case {
        label: "calldata a word short".into(),
        js: None,
        expected: Outcome::Revert,
        calldata: short_calldata,
    });

    let title = format!(
        "{}: {} f, powerW {}, runtime code {size} bytes, {} words of `proof`",
        setup.name,
        vkey.layout.0.len(),
        vkey.power_w,
        layout.words()
    );
    let gas = check_on_foundry(tools, &key.dir.0, &key.sol, &title, &cases);
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
        // Any W and evaluations for now: the transcript absorbs them after xiSeed.
        w: g1_times(&f_tau),
        wp: g1_times(&BigUint::from(1u32)),
        evaluations: vec![FrBytes::ZERO; 2],
        air_values: vec![],
        airgroup_values: vec![],
        proof_values: vec![],
        inv: FrBytes::ZERO,
        inv_zh: FrBytes::ZERO,
    };
    let xi = xi_of(&vkey, &big(&verifier_challenges(&vkey, &proof, &publics).unwrap().xi_seed));
    let (q1_xi, q0_xi) = (q1(&xi), q0(&xi));
    // The pieces in the order of the layout: Q1, Q0.
    proof.evaluations = vec![fr(&q1_xi), fr(&q0_xi)];
    // r(X) = Q1(ξ) + X·Q0(ξ) interpolates f on T = {±xiSeed}, the roots of Z_T = X² − ξ; h = (f − r)/Z_T.
    let r_at = |x: &BigUint| (&q1_xi + x * &q0_xi) % &r;
    let h_tau = fr_sub(&f_tau, &r_at(&tau)) * fr_inv(&fr_sub(&pow(&tau, 2), &xi)) % &r;
    proof.w = g1_times(&h_tau);
    let y = big(&verifier_challenges(&vkey, &proof, &publics).unwrap().y);
    // W' = L/(X − y) at τ, with L = f − r(y) − Z_T(y)·h: q_0 = Z_T(y) for one f (A.5).
    let l_tau = fr_sub(&fr_sub(&f_tau, &r_at(&y)), &(fr_sub(&pow(&y, 2), &xi) * &h_tau % &r));
    proof.wp = g1_times(&(l_tau * fr_inv(&fr_sub(&tau, &y)) % &r));
    let proof = fixup(&vkey, proof, &publics);

    // The JSON view, by hand: no pilfflonkinfo names this proof (A.6: f0, W, Wp; Q1, Q0, inv, invZh),
    // and those are the names of the vkey's.
    let names = ProofNames::of_vkey(&vkey).unwrap();
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
        assert_eq!(serde_json::to_value(proof.to_json(&names).unwrap()).unwrap(), view, "{label}");
        fs::write(at.join("proof.json"), view.to_string()).unwrap();
        Publics(publics.to_vec()).write(&at.join("publics.json")).unwrap();
        js_verdict_of(&vkey_path, &at)
    };
    let mut cases = Vec::new();
    let mut add = |label: &str, proof: &Proof, publics: &[FrBytes], expected: bool| {
        let verdict = js(label, proof, publics);
        assert_eq!(verdict, expected, "the JS verifier on {label}");
        let calldata = calldata(&vkey, proof, publics);
        cases.push(Case { label: label.into(), js: Some(verdict), expected: Outcome::of_js(verdict), calldata });
    };
    add("proof", &proof, &publics, true);
    let mut piece = proof.clone();
    piece.evaluations[0] = plus_one(&piece.evaluations[0]);
    add("Q1 + 1, fixed up", &fixup(&vkey, piece, &publics), &publics, false);
    let mut pieces = proof.clone();
    let xi_seed = big(&verifier_challenges(&vkey, &proof, &publics).unwrap().xi_seed);
    rebalance_pieces(&vkey, &mut pieces, &xi_seed, 12345);
    add("Q pieces rebalanced, fixed up", &fixup(&vkey, pieces, &publics), &publics, false);
    let other = vec![FrBytes::from_u64(6)];
    add("public 6, fixed up", &fixup(&vkey, proof.clone(), &other), &other, false);

    let title = format!(
        "noeval: 1 f, powerW 2, runtime code {size} bytes, {} words of `proof`",
        CalldataLayout::of(&vkey).words()
    );
    let gas = check_on_foundry(&tools, &dir.0, &sol, &title, &cases);
    // The rebalanced pieces pass checkQPieces: the pairing refuses them.
    assert!(
        10 * gas[2] >= 9 * gas[0],
        "the rebalanced pieces stop before the pairing: {} gas, the proof {}",
        gas[2],
        gas[0]
    );
}
