//! Emitted domain-Zi circom (GL, BN128) vs the native `domain_zi_vectors.json`; Zh(ξ)=0 has no witness.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::Deserialize;
use serde_json::Value;

use super::blake3_equiv_tests::{build, manifest, missing_prerequisite, repo_root, DRIVER_CPP};
use super::circom_verifier::domain_zi::{render_fragment, FRAGMENT_BN128, FRAGMENT_GL};
use super::circom_verifier::gl_field::{gl_exp, GL_W};

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The vector file's point; `zi` is empty when it must be rejected.
#[derive(Deserialize)]
struct Point {
    xi: [u64; 3],
    #[serde(default)]
    zi: Vec<[u64; 3]>,
    #[serde(default)]
    reject: bool,
}

#[derive(Deserialize)]
struct Vectors {
    #[serde(rename = "nBits")]
    n_bits: usize,
    boundaries: Vec<Value>,
    points: Vec<Point>,
}

fn vectors() -> Vectors {
    let path = repo_root().join("verifier/tests/data/domain_zi_vectors.json");
    let mut v: Vectors = serde_json::from_str(&fs::read_to_string(&path).expect("vector file")).unwrap();
    let root = zh_only_root(&v);
    v.points.push(Point { xi: [root, 0, 0], zi: Vec::new(), reject: true });
    v
}

/// A row g^i where every domain's V is nonzero, so only Zh's CInv can reject it.
fn zh_only_root(v: &Vectors) -> u64 {
    let bs: Vec<proofman_common::Boundary> = serde_json::from_value(Value::Array(v.boundaries.clone())).unwrap();
    let n = 1u64 << v.n_bits;
    (1..n)
        .map(|i| gl_exp(GL_W[v.n_bits], i))
        .find(|&x| {
            bs[1..].iter().all(|b| {
                let xm = gl_exp(x, n >> b.cycle_bits);
                b.offsets.iter().all(|&o| xm != gl_exp(GL_W[b.cycle_bits as usize], o))
            })
        })
        .expect("no row outside every domain")
}

/// `zi[0] = Zh`, `zi[k+1] = Zd{k}`: the order of the vector file.
fn output_wiring(n_boundaries: usize) -> String {
    let mut s = String::from("    zi[0] <== Zh;\n");
    for k in 0..n_boundaries - 1 {
        s += &format!("    zi[{}] <== Zd{k};\n", k + 1);
    }
    s
}

/// Runs the circuit on each point (`run(i, ξ)`) and checks its Zi outputs, read mod p, or its rejection.
fn check_points(field: &str, v: &Vectors, run: impl Fn(usize, [u64; 3]) -> Output, rejected: impl Fn(&Output)) {
    for (i, p) in v.points.iter().enumerate() {
        let out = run(i, p.xi);
        let xi = p.xi;
        if p.reject {
            assert!(!out.status.success(), "{field} circuit accepted xi={xi:?} where Zh(xi) = 0");
            rejected(&out);
            continue;
        }
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{field} witness failed at xi={xi:?}:\n{err}");
        // BN128 Zd may be an unreduced GLCSub output
        let got: Vec<u64> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|l| (l.trim().parse::<u128>().unwrap() % P as u128) as u64)
            .collect();
        let want: Vec<u64> = p.zi.iter().flatten().copied().collect();
        assert_eq!(got, want, "{field}: Zi mismatch at xi={xi:?}");
    }
}

#[test]
fn circom_domain_zi_matches_native() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP circom_domain_zi_matches_native: {why}");
        return;
    }
    let v = vectors();
    let nb = v.boundaries.len();
    let frag = render_fragment(FRAGMENT_GL, &v.boundaries, v.n_bits).unwrap();
    let src = format!(
        "pragma circom 2.1.0;\npragma custom_templates;\ninclude \"cmul.circom\";\ninclude \"cinv.circom\";\n\n\
         template T() {{\n    signal input challengeOut[3];\n    signal output zi[{nb}][3];\n{frag}\n{}}}\n\
         component main = T();\n",
        output_wiring(nb)
    );
    let (dir, exe) = build("domain_zi_gl", &src, DRIVER_CPP);
    let run = |i: usize, xi: [u64; 3]| {
        let input = dir.join(format!("in{i}.txt"));
        fs::write(&input, xi.iter().map(u64::to_string).collect::<Vec<_>>().join("\n")).unwrap();
        Command::new(&exe)
            .arg(dir.join("c_cpp/c.dat"))
            .arg(&input)
            .arg((3 * nb).to_string())
            .output()
            .expect("failed to run witness driver")
    };
    check_points("GL", &v, run, |_| {});
    let _ = fs::remove_dir_all(&dir);
}

/// Prints the first `n` output cells of the wasm witness on one input file.
const WASM_DRIVER_JS: &str = r#"
const fs = require("fs");
const builder = require(process.argv[2]);
const n = Number(process.argv[5]);
builder(fs.readFileSync(process.argv[3]))
    .then(async (wc) => {
        const w = await wc.calculateWitness(JSON.parse(fs.readFileSync(process.argv[4])), 0);
        console.log(w.slice(1, 1 + n).map(String).join("\n"));
    })
    .catch((e) => { console.error(e.message || e); process.exit(1); });
"#;

fn compile_wasm(dir: &Path, src: &str, circomlib: &Path) {
    let circuits = manifest().join("stark2circom/circom_verifier/circuits.bn128");
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("c.circom"), src).unwrap();
    let out = Command::new("circom")
        .arg(dir.join("c.circom"))
        .args(["--wasm", "--O2", "-o"])
        .arg(dir)
        .arg("-l")
        .arg(&circuits)
        .arg("-l")
        .arg(circomlib)
        .output()
        .expect("failed to run circom");
    let text = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "circom failed:\n{text}");
    fs::write(dir.join("driver.js"), WASM_DRIVER_JS).unwrap();
}

#[test]
fn circom_domain_zi_matches_native_bn128() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP circom_domain_zi_matches_native_bn128: {why}");
        return;
    }
    if !Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("SKIP circom_domain_zi_matches_native_bn128: node not on PATH");
        return;
    }
    let circomlib = repo_root().join("setup/pil2-stark/node_modules/circomlib/circuits");
    if !circomlib.is_dir() {
        eprintln!("SKIP circom_domain_zi_matches_native_bn128: {} not present", circomlib.display());
        return;
    }
    let v = vectors();
    let nb = v.boundaries.len();
    let frag = render_fragment(FRAGMENT_BN128, &v.boundaries, v.n_bits).unwrap();
    let src = format!(
        "pragma circom 2.1.0;\ninclude \"gl.circom\";\n\n\
         template T() {{\n    signal input in[3];\n    signal output zi[{nb}][3];\n\
         \x20   signal {{maxNum}} challengeOut[3];\n    challengeOut.maxNum = 0xFFFFFFFFFFFFFFFF;\n    challengeOut <== in;\n\
         {frag}\n{}}}\ncomponent main = T();\n",
        output_wiring(nb)
    );
    let dir: PathBuf = std::env::temp_dir().join(format!("domain_zi_bn128_{}", std::process::id()));
    compile_wasm(&dir, &src, &circomlib);
    let run = |i: usize, xi: [u64; 3]| {
        let input = dir.join(format!("in{i}.json"));
        fs::write(&input, format!("{{\"in\": [\"{}\", \"{}\", \"{}\"]}}", xi[0], xi[1], xi[2])).unwrap();
        Command::new("node")
            .arg(dir.join("driver.js"))
            .arg(dir.join("c_js/witness_calculator.js"))
            .arg(dir.join("c_js/c.wasm"))
            .arg(&input)
            .arg((3 * nb).to_string())
            .output()
            .expect("failed to run node")
    };
    check_points("BN128", &v, run, |out| {
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("GLCInv"), "BN128 rejected outside GLCInv:\n{err}");
    });
    let _ = fs::remove_dir_all(&dir);
}
