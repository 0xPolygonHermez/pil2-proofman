//! plonk2pil over BN254: a circuit circom compiles for BN254 is read and converted to PLONK gates and
//! additions by the same generic code the Goldilocks recursion runs, and the result is checked on a
//! real witness of the circuit.
//!
//! The circuit is `fixtures/bn254/arith.circom`, compiled here with the committed circom
//! (`setup/circom`), so the r1cs cannot go stale. Its witness is computed from
//! `fixtures/bn254/input.json` by the circom-generated wasm and the snarkjs of
//! `setup/pil2-stark/node_modules` (`npm install` there). Without Node.js or that snarkjs the test
//! says why and passes, as the circom tests of `stark2circom` do.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use pil2_stark_recurser::plonk2pil::field::{PlonkField, R1csPrime};
use pil2_stark_recurser::plonk2pil::merge_copies::r1cs2plonk_merged;
use pil2_stark_recurser::plonk2pil::r1cs::to_plonk::{r1cs2plonk, PlonkAddition, PlonkConstraint};
use pil2_stark_recurser::plonk2pil::r1cs::types::{
    r1cs_prime, read_r1cs_from_bytes, read_r1cs_header, LinearCombination, PlonkOptions, R1csFile,
};
use pil2_stark_recurser::plonk2pil::plonk2pil;
use proofman_fields::{Bn254, Field, PrimeField};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    manifest().join("../..")
}

/// The committed circom, as the plonk2pil golden picks it.
fn circom() -> PathBuf {
    repo_root().join("setup/circom").join(if cfg!(target_os = "macos") { "circom_mac" } else { "circom" })
}

fn snarkjs() -> PathBuf {
    repo_root().join("setup/pil2-stark/node_modules/snarkjs/build/cli.cjs")
}

/// What the witness needs and is missing, if anything.
fn missing_prerequisite() -> Option<String> {
    let node = Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !node {
        return Some("node not on PATH".into());
    }
    if !snarkjs().is_file() {
        return Some(format!("{} not present (npm install in setup/pil2-stark)", snarkjs().display()));
    }
    None
}

/// A directory of this test's own under the temporary directory, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or_default();
        let dir = std::env::temp_dir().join(format!("plonk2pil_bn254_{}_{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover directory in the temporary directory harms nothing.
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(cmd: &mut Command, what: &str) {
    let out = cmd.output().unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        out.status.success(),
        "{what} failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Compiles the fixture for BN254 into `dir` and computes its witness: the r1cs and the `.wtns`.
fn compile_and_witness(dir: &Path) -> (Vec<u8>, Vec<u8>) {
    let src = manifest().join("tests/fixtures/bn254/arith.circom");
    // --O1 keeps the linear constraints, which are what reach plonk2pil's sum gates.
    run(
        Command::new(circom()).args(["--O1", "--r1cs", "--wasm", "--prime", "bn128"]).arg(&src).arg("-o").arg(dir),
        "circom",
    );
    let wtns = dir.join("arith.wtns");
    run(
        Command::new("node")
            .arg(snarkjs())
            .args(["wtns", "calculate"])
            .arg(dir.join("arith_js/arith.wasm"))
            .arg(manifest().join("tests/fixtures/bn254/input.json"))
            .arg(&wtns),
        "snarkjs wtns calculate",
    );
    let read = |p: PathBuf| fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    (read(dir.join("arith.r1cs")), read(wtns))
}

/// The witness in a `.wtns` file, as snarkjs's `wtns_utils.js` writes one: the magic `wtns`, a
/// version and a section count, then sections of `(type: u32, size: u64, data)`. Section 1 is
/// `n8: u32`, the prime in `n8` bytes and the witness length `u32`; section 2 the values, `n8` bytes
/// each, canonical and little-endian.
fn read_wtns<F: PlonkField>(data: &[u8]) -> Vec<F> {
    let u32_at = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
    let u64_at = |at: usize| u64::from_le_bytes(data[at..at + 8].try_into().unwrap()) as usize;
    assert_eq!(&data[..4], b"wtns", "not a .wtns file");
    let mut sections = std::collections::HashMap::new();
    let mut at = 12;
    for _ in 0..u32_at(8) {
        let (kind, size) = (u32_at(at), u64_at(at + 4));
        sections.insert(kind, &data[at + 12..at + 12 + size]);
        at += 12 + size;
    }

    let header = sections[&1];
    let n8 = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    assert_eq!(header[4..4 + n8], F::PRIME.modulus_le(), "the witness is not over {}", F::PRIME);
    let n = u32::from_le_bytes(header[4 + n8..8 + n8].try_into().unwrap()) as usize;
    let values = sections[&2];
    assert_eq!(values.len(), n * n8);
    values.chunks_exact(n8).map(|v| F::from_canonical_le(v).expect("a canonical witness value")).collect()
}

fn eval<F: Field>(lc: &LinearCombination<F>, w: &[F]) -> F {
    lc.iter().fold(F::ZERO, |acc, (&wire, &q)| acc + q * w[wire as usize])
}

fn r1cs_holds<F: Field>(r1cs: &R1csFile<F>, w: &[F]) -> bool {
    r1cs.constraints.iter().all(|c| eval(&c.a, w) * eval(&c.b, w) == eval(&c.c, w))
}

/// The witness with the wires the additions introduce appended: the `i`-th is wire `n_vars + i`,
/// and reads only wires defined before it.
fn with_additions<F: Field>(witness: &[F], adds: &[PlonkAddition<F>]) -> Vec<F> {
    let mut w = witness.to_vec();
    for (i, a) in adds.iter().enumerate() {
        let wire = w.len();
        assert!(
            a.wires.iter().all(|&x| (x as usize) < wire),
            "addition {i}, wire {wire}, reads {:?}: a wire defined after it",
            a.wires
        );
        w.push(a.coeffs[0] * w[a.wires[0] as usize] + a.coeffs[1] * w[a.wires[1] as usize]);
    }
    w
}

/// The gates `qM·l·r + qL·l + qR·r + qO·o + qC` that are not zero on `w`.
fn failing_gates<F: Field>(cs: &[PlonkConstraint<F>], w: &[F]) -> Vec<usize> {
    let fails = |c: &PlonkConstraint<F>| {
        let [l, r, o] = c.wires.map(|x| w[x as usize]);
        let [q_m, q_l, q_r, q_o, q_c] = c.coeffs;
        !(q_m * l * r + q_l * l + q_r * r + q_o * o + q_c).is_zero()
    };
    cs.iter().enumerate().filter(|(_, c)| fails(c)).map(|(i, _)| i).collect()
}

fn plonk_holds<F: Field>(cs: &[PlonkConstraint<F>], adds: &[PlonkAddition<F>], witness: &[F]) -> bool {
    failing_gates(cs, &with_additions(witness, adds)).is_empty()
}

#[test]
fn a_bn254_circuit_converts_to_plonk_gates_that_hold_on_its_witness() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("skipping the BN254 plonk2pil test: {why}");
        return;
    }
    let scratch = Scratch::new();
    let (r1cs_bytes, wtns_bytes) = compile_and_witness(&scratch.0);

    assert_eq!(r1cs_prime(&read_r1cs_header(&r1cs_bytes).unwrap()).unwrap(), R1csPrime::Bn254);
    let r1cs = read_r1cs_from_bytes::<Bn254>(&r1cs_bytes).expect("a BN254 r1cs reads into Bn254");
    let witness = read_wtns::<Bn254>(&wtns_bytes);
    assert_eq!(r1cs.header.n8, 32);
    assert_eq!(witness.len(), r1cs.header.n_vars as usize);
    assert_eq!(witness[0], Bn254::ONE, "wire 0 is the constant one");
    assert!(r1cs_holds(&r1cs, &witness), "the r1cs as read must hold on snarkjs's witness");

    // What the fixture is for: coefficients no u64 holds, sums wide enough to need additions, and
    // both kinds of gate.
    let wide = r1cs.constraints.iter().flat_map(|c| [&c.a, &c.b, &c.c]).flat_map(|lc| lc.values());
    assert!(wide.into_iter().any(|q| q.as_canonical_biguint().bits() > 64), "no coefficient is wider than 64 bits");
    let (cs, adds) = r1cs2plonk(&r1cs);
    assert!(!adds.is_empty(), "the wide sum must introduce additions");
    assert!(cs.iter().any(|c| c.coeffs[0].is_zero()) && cs.iter().any(|c| !c.coeffs[0].is_zero()));

    // Every gate holds once the additions are applied, in order.
    let extended = with_additions(&witness, &adds);
    assert_eq!(extended.len(), r1cs.header.n_vars as usize + adds.len());
    assert_eq!(failing_gates(&cs, &extended), Vec::<usize>::new(), "gates that fail on the witness");

    // And the gates say what the r1cs says: change any one signal and the two agree on whether the
    // assignment still holds, so no constraint was lost on the way.
    let mut checked = 0;
    for wire in 1..witness.len() {
        let mut wrong = witness.clone();
        wrong[wire] += Bn254::from_decimal("340282366920938463463374607431768211457").unwrap();
        let (by_r1cs, by_plonk) = (r1cs_holds(&r1cs, &wrong), plonk_holds(&cs, &adds, &wrong));
        assert_eq!(by_r1cs, by_plonk, "wire {wire}: the r1cs says {by_r1cs}, the PLONK gates {by_plonk}");
        checked += usize::from(!by_r1cs);
    }
    assert!(checked > 0, "no single-wire change broke the r1cs, so this checked nothing");

    // The copy-merged conversion the setups run holds on the same witness.
    let (merged_cs, merged_adds, _) = r1cs2plonk_merged(&r1cs, true);
    assert!(plonk_holds(&merged_cs, &merged_adds, &witness), "the copy-merged gates fail on the witness");

    // The entry point still refuses it: the families are Goldilocks-only.
    let err = plonk2pil(&r1cs_bytes, "aggregation", &PlonkOptions::default()).unwrap_err().to_string();
    assert!(err.contains("BN254 is not supported yet"), "{err}");
}
