//! Emitted domain-Zi circom (GL, BN128) vs the native `domain_zi_vectors.json`; Zh(ξ)=0 has no witness.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

use super::blake3_equiv_tests::{build, manifest, missing_prerequisite, repo_root, DRIVER_CPP};
use super::circom_verifier::domain_zi::{render_fragment, FRAGMENT_BN128, FRAGMENT_GL};
use super::circom_verifier::gl_field::{gl_exp, GL_W};

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// (ξ, expected Zi per boundary), or None when the point must be rejected.
type Point = ([u64; 3], Option<Vec<[u64; 3]>>);

struct Vectors {
    boundaries: Vec<Value>,
    n_bits: usize,
    points: Vec<Point>,
}

fn ext(v: &Value) -> [u64; 3] {
    let a = v.as_array().expect("ext element");
    [0, 1, 2].map(|i| a[i].as_u64().expect("u64 limb"))
}

fn vectors() -> Vectors {
    let path = repo_root().join("verifier/tests/data/domain_zi_vectors.json");
    let v: Value = serde_json::from_str(&fs::read_to_string(&path).expect("vector file")).unwrap();
    let points = v["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let zi = if p["reject"].as_bool() == Some(true) {
                None
            } else {
                Some(p["zi"].as_array().unwrap().iter().map(ext).collect())
            };
            (ext(&p["xi"]), zi)
        })
        .collect();
    let mut vs = Vectors {
        boundaries: v["boundaries"].as_array().unwrap().clone(),
        n_bits: v["nBits"].as_u64().unwrap() as usize,
        points,
    };
    vs.points.push(([zh_only_root(&vs), 0, 0], None));
    vs
}

/// A row g^i where every domain's V is nonzero, so only Zh's CInv can reject it.
fn zh_only_root(v: &Vectors) -> u64 {
    let n = 1u64 << v.n_bits;
    (1..n)
        .map(|i| gl_exp(GL_W[v.n_bits], i))
        .find(|&x| {
            v.boundaries[1..].iter().all(|b| {
                let c = b["cycleBits"].as_u64().unwrap() as usize;
                let xm = gl_exp(x, n >> c);
                b["offsets"].as_array().unwrap().iter().all(|o| xm != gl_exp(GL_W[c], o.as_u64().unwrap()))
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

fn assert_matches(field: &str, xi: [u64; 3], got: &[u64], want: &[[u64; 3]]) {
    let want: Vec<u64> = want.iter().flatten().copied().collect();
    assert_eq!(got, want.as_slice(), "{field}: Zi mismatch at xi={xi:?}");
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

    for (i, (xi, want)) in v.points.iter().enumerate() {
        let input = dir.join(format!("in{i}.txt"));
        fs::write(&input, xi.iter().map(u64::to_string).collect::<Vec<_>>().join("\n")).unwrap();
        let out = Command::new(&exe)
            .arg(dir.join("c_cpp/c.dat"))
            .arg(&input)
            .arg((3 * nb).to_string())
            .output()
            .expect("failed to run witness driver");
        match want {
            Some(want) => {
                assert!(
                    out.status.success(),
                    "GL witness failed at xi={xi:?}:\n{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let got: Vec<u64> =
                    String::from_utf8_lossy(&out.stdout).lines().map(|l| l.trim().parse().unwrap()).collect();
                assert_matches("GL", *xi, &got, want);
            }
            None => assert!(!out.status.success(), "GL circuit accepted xi={xi:?} where Zh(xi) = 0"),
        }
    }
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

    let circuits = manifest().join("stark2circom/circom_verifier/circuits.bn128");
    let dir: PathBuf = std::env::temp_dir().join("domain_zi_bn128");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("c.circom"), &src).unwrap();
    let out = Command::new("circom")
        .arg(dir.join("c.circom"))
        .args(["--wasm", "--O2", "-o"])
        .arg(&dir)
        .arg("-l")
        .arg(&circuits)
        .arg("-l")
        .arg(&circomlib)
        .output()
        .expect("failed to run circom");
    assert!(
        out.status.success(),
        "circom failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    fs::write(dir.join("driver.js"), WASM_DRIVER_JS).unwrap();

    for (i, (xi, want)) in v.points.iter().enumerate() {
        let input = dir.join(format!("in{i}.json"));
        fs::write(&input, format!("{{\"in\": [\"{}\", \"{}\", \"{}\"]}}", xi[0], xi[1], xi[2])).unwrap();
        let out = Command::new("node")
            .arg(dir.join("driver.js"))
            .arg(dir.join("c_js/witness_calculator.js"))
            .arg(dir.join("c_js/c.wasm"))
            .arg(&input)
            .arg((3 * nb).to_string())
            .output()
            .expect("failed to run node");
        match want {
            Some(want) => {
                assert!(
                    out.status.success(),
                    "BN128 witness failed at xi={xi:?}:\n{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                // BN128 Zd may be an unreduced GLCSub output; compare mod p.
                let got: Vec<u64> = String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .map(|l| (l.trim().parse::<u128>().unwrap() % P as u128) as u64)
                    .collect();
                assert_matches("BN128", *xi, &got, want);
            }
            None => {
                let err = String::from_utf8_lossy(&out.stderr);
                assert!(!out.status.success(), "BN128 circuit accepted xi={xi:?} where Zh(xi) = 0");
                assert!(err.contains("GLCInv"), "BN128 rejected xi={xi:?} outside GLCInv:\n{err}");
            }
        }
    }
    let _ = fs::remove_dir_all(&dir);
}
