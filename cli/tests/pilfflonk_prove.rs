//! `proofman-cli pilfflonk prove` (spec §4.4, plan M18) end to end on the Fibonacci fixture: the
//! setup (`setup-pilfflonk --no-packing`), the prover's CLI and the JS verifier's (`pilfflonk
//! verify`, M19), and the prover against the Rust oracle (M14).
//!
//! The ptau is `PILFFLONK_TEST_PTAU` if it is set, and otherwise one this test writes with the
//! full-width `τ` of the C++ test helper (`pilfflonk_setup::test_ptau::fixed_tau_ptau`, plan N13):
//! not the ptau of `τ = 1`, under which the blinding vanishes from every commitment and any proof
//! verifies. With it, the verifier accepts the prover's proofs only if they are sound, and rejects
//! every change to one.
//!
//! Pilouts are not versioned: the test compiles the fixture with the compiler `PIL2C_EXEC` names,
//! which must honour `prime`, and is `#[ignore]` without it. It needs Node.js:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only \
//!     --test pilfflonk_prove -- --ignored
//! ```

#[path = "../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pil2_pilout::pilout_proxy::PilOutProxy;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::digest::seal_vkey;
use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman_pilfflonk::oracle::{AirOracle, ColumnRef, Fr};
use proofman_pilfflonk::{
    prove, AirFile, Vkey, FileWitnessSource, FrBytes, PilfflonkError, PilfflonkGlobalInfo, PolType, ProveOptions,
    ProvingKey, Witness, WitnessSource, BN254_R,
};
use serde_json::{json, Value};

const SEED_A: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
const SEED_B: &str = "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100";

/// The expression of the Fibonacci's im pol, `l1' − (l1² + l2²)` (`pilfflonk/tests/fibonacci.rs`).
const IM_POL: usize = 6;

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

/// The ptau of the test (see the module), with more powers than the Fibonacci's largest degree,
/// 261 (plan M16).
fn ptau(dir: &TestDir) -> PathBuf {
    match std::env::var_os("PILFFLONK_TEST_PTAU") {
        Some(path) => PathBuf::from(path),
        None => {
            let path = dir.file("fixed_tau.ptau");
            write_fixed_tau_ptau(&path, 512, &test_tau()).unwrap();
            path
        }
    }
}

struct Fixture {
    dir: TestDir,
    pilout: PathBuf,
    proving_key: PathBuf,
    vkey: PathBuf,
    witness: PathBuf,
}

/// The Fibonacci compiled, set up and its witness written (M13's generator, inputs [1, 2]).
fn fixture(name: &str) -> Fixture {
    let dir = TestDir::new(name);
    let opts = SetupPilfflonkOptions {
        airout_path: dir.file("fibonacci.pilout"),
        build_dir: dir.file("build"),
        powers_of_tau: ptau(&dir),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: true,
    };
    compile_fibonacci(&opts.airout_path);
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let vkey = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap().vkey_path(&proving_key);
    let witness = dir.file("witness");
    let shape = ProvingKey::load(&proving_key).unwrap().witness_shape().unwrap();
    fibonacci::witness(8, [1, 2]).write(&witness, &shape).unwrap();
    Fixture { pilout: opts.airout_path, dir, proving_key, vkey, witness }
}

#[test]
#[ignore = "needs PIL2C_EXEC and Node.js"]
fn the_prover_proves_the_fibonacci_and_the_verifier_rejects_every_change() {
    let f = fixture("e2e");
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
    // The non-fixed f (f2 … f5: l1, l2, the im pol and Q), W and W'; the fixed ones are the vkey's.
    assert_eq!(polynomials.keys().collect::<Vec<_>>(), ["W", "Wp", "f2", "f3", "f4", "f5"]);
    for name in polynomials.keys() {
        assert_ne!(pa["polynomials"][name], pb["polynomials"][name], "{name} with another seed");
        assert_ne!(pa["polynomials"][name], pr["polynomials"][name], "{name} with the OS's randomness");
    }
    let evaluations = pa["evaluations"].as_object().unwrap();
    let names: Vec<&String> = evaluations.keys().collect();
    assert_eq!(
        names,
        ["Fibonacci.ImPol[0]", "Fibonacci.L1", "Fibonacci.LLAST", "inv", "invZh", "l1", "l1w", "l2", "l2w"]
    );
    assert_eq!(read_json(&a1.join("publics.json")), json!(["1", "2", fibonacci_out()]));

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
    for name in evaluations.keys() {
        let mut tampered = pa.clone();
        tampered["evaluations"][name] = plus_one(&pa["evaluations"][name]);
        write_json(&other, &tampered);
        rejected(&publics, &other, name);
    }
    let public_values = read_json(&publics);
    for i in 0..3 {
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

/// `out` of the Fibonacci for [1, 2] (`pil-fflonk/runtime/public.json`).
fn fibonacci_out() -> &'static str {
    "590308608561184158373097535019708483037277117989374906445627411437315467687"
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn a_witness_that_breaks_a_constraint_is_refused() {
    let f = fixture("unsatisfied");
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

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_prover_agrees_with_the_oracle() {
    let f = fixture("oracle");
    let pk = ProvingKey::load(&f.proving_key).unwrap();
    let source = FileWitnessSource::open(&f.witness, &pk.witness_shape().unwrap()).unwrap();
    let options = ProveOptions { insecure_blinding_seed: Some([5; 32]) };
    let out = prove(&pk, &source, &options).unwrap();
    let info = pk.air(source.instances()[0]).unwrap();
    assert_eq!(info.layout.power_w().unwrap(), 1, "ξ is xiSeed");
    let xi = Fr::from(out.challenges.xi_seed);
    let std_vc = Fr::from(out.challenges.std_vc);

    let pilout = PilOutProxy::new(f.pilout.to_str().unwrap()).unwrap().pilout;
    let oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let values = oracle.values(&source, 0).unwrap();
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
    let mut at_xi = BTreeMap::new();
    for (e, value) in entries.zip(&out.proof.evaluations) {
        at_xi.insert((column(e.pol_type, e.id), e.prime as i32), Fr::from(*value));
    }
    assert_eq!(at_xi.len(), out.proof.evaluations.len());

    // The fixed columns have no blinding: their evaluations are the oracle's, exactly.
    for id in 0..2 {
        let expected = oracle.column_at(&values, ColumnRef::Fixed(id), 0, &xi).unwrap();
        assert_eq!(at_xi[&(ColumnRef::Fixed(id), 0)], expected, "fixed column {id} at ξ");
    }
    // The committed ones are blinded (A.3): not the oracle's interpolants at ξ …
    assert_ne!(
        at_xi[&(ColumnRef::Witness { stage: 1, idx: 0 }, 0)],
        oracle.column_at(&values, ColumnRef::Witness { stage: 1, idx: 0 }, 0, &xi).unwrap()
    );
    // … but Q(ξ), as the oracle folds the constraints over the proof's evaluations, is the prover's Q
    // at ξ: the value the verifier computes, and SHPLONK opens Q's f at.
    let q = oracle.q_from_evaluations(&values, &at_xi, &[IM_POL], &std_vc, &xi).unwrap();
    assert_eq!(q, Fr::from(out.challenges.q_at_xi));
    // Another evaluation, another Q(ξ).
    let mut changed = at_xi.clone();
    let l1w = changed.get_mut(&(ColumnRef::Witness { stage: 1, idx: 0 }, 1)).unwrap();
    *l1w = &*l1w + &Fr::one();
    assert_ne!(oracle.q_from_evaluations(&values, &changed, &[IM_POL], &std_vc, &xi).unwrap(), q);
    // invZh = 1/Z_H(ξ).
    assert_eq!(&Fr::from(out.proof.inv_zh) * &(&xi.pow_u64(256) - &Fr::one()), Fr::one());
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_prover_refuses_a_proving_key_whose_files_disagree() {
    let f = fixture("disagree");
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

    // The vkey restored, the C++ loader's refusals come through: a .const cut short.
    fs::write(&f.vkey, &original).unwrap();
    ProvingKey::load(&f.proving_key).unwrap();
    let global_info = PilfflonkGlobalInfo::from_proving_key(&f.proving_key).unwrap();
    let constants = global_info.air_file(&f.proving_key, 0, 0, AirFile::Const).unwrap();
    let mut bytes = fs::read(&constants).unwrap();
    bytes.truncate(bytes.len() - 32);
    fs::write(&constants, bytes).unwrap();
    let err = load_error();
    assert!(
        err.contains("loading the provingKey/ into the C++ prover") && err.contains(".const has 16352 bytes"),
        "{err}"
    );
}
