//! Equivalence tests for the emitted BLAKE3 circom: it must hash exactly like
//! the reference `blake3` crate.
//!
//! Both the leaf hash (`LinearHash` in circuits.gl/hash/blake3/linearhash.circom)
//! and the Fiat-Shamir transcript (emitted round by round by [`super::transcript`])
//! are hand-written BLAKE3, and since the linearhash template writes its chunk
//! tree out directly while the transcript emulates it with a cv stack, the two
//! share no code. A bug in either would not show up in the other, nor in any
//! test that only checks the *shape* of the emitted circom. So both are checked
//! here against the same oracle: [`Blake3Transcript`], which wraps the upstream
//! `blake3` crate and is what the Rust prover uses.
//!
//! The circuits are actually compiled and run: circom emits C++, that is linked
//! against the real custom-gate witness implementation (setup/circom), and the
//! resulting witness is compared word for word.
//!
//! End-to-end hash agreement is necessary but not sufficient: the gate also
//! writes 3080 `im` cells per compression that the AIR constrains but the digest
//! does not depend on, so a bug there is invisible to both tests above.
//! `gate_witness_cells_match_scalar_path` closes that gap.
//!
//! Requires the goldilocks `circom` fork and a C++ toolchain. When either is
//! missing the tests print why and pass, so `cargo test --workspace` still works
//! on a machine (or CI runner) without them.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use proofman_fields::{Blake3Transcript, Goldilocks, PrimeField64};

use super::transcript::Transcript;

/// Runs the compiled circuit and prints the first `n_out` witness cells after
/// the leading constant 1, i.e. the main component's outputs.
const DRIVER_CPP: &str = r#"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" uint64_t getSizeWitness();
extern "C" void *initCircuit(char *datFile);
extern "C" int64_t getWitness(uint64_t *proof, void *circuit, void *pWitness, uint64_t nMutexes);

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: driver <dat> <inputs> <n_out>\n"); return 1; }
    FILE *f = fopen(argv[2], "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    std::vector<uint64_t> in;
    unsigned long long v;
    while (fscanf(f, "%llu", &v) == 1) in.push_back((uint64_t)v);
    fclose(f);

    void *circuit = initCircuit(argv[1]);
    if (!circuit) { fprintf(stderr, "initCircuit failed\n"); return 1; }
    std::vector<uint64_t> w(getSizeWitness());
    if (getWitness(in.data(), circuit, w.data(), 8) != 0) { fprintf(stderr, "getWitness failed\n"); return 1; }

    const int n_out = atoi(argv[3]);
    for (int i = 0; i < n_out; i++) printf("%llu\n", (unsigned long long)w[1 + i]);
    return 0;
}
"#;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_root() -> PathBuf {
    manifest().join("..").join("..")
}

fn have(cmd: &str, args: &[&str]) -> bool {
    Command::new(cmd).args(args).output().map(|o| o.status.success()).unwrap_or(false)
}

/// The pipeline needs the goldilocks circom fork, a C++ compiler, and the
/// witness-side sources. Report what is missing rather than failing.
fn missing_prerequisite() -> Option<String> {
    if !have("circom", &["--version"]) {
        return Some("circom not on PATH".into());
    }
    if !have("g++", &["--version"]) {
        return Some("g++ not on PATH".into());
    }
    let gate = repo_root().join("setup/circom/blake3_goldilocks.cpp");
    if !gate.exists() {
        return Some(format!("{} not present", gate.display()));
    }
    None
}

/// Compile `src` and return the main component's first `n_out` output cells,
/// evaluated on `inputs`.
/// Compile `src` with circom, link `driver_src` against the generated witness
/// code and the real custom-gate implementation, and return (build dir, exe).
fn build(tag: &str, src: &str, driver_src: &str) -> (PathBuf, PathBuf) {
    let root = repo_root();
    let circuits = manifest().join("stark2circom/circom_verifier/circuits.gl");
    let blake3 = circuits.join("hash/blake3");
    let support = root.join("setup/circom");
    let goldilocks = root.join("pil2-stark/src/goldilocks/src");

    let dir = std::env::temp_dir().join(format!("b3_equiv_{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let circuit = dir.join("c.circom");
    fs::write(&circuit, src).unwrap();

    // circom needs circuits.gl on its include path, and --prime goldilocks is
    // what makes the generated C++ use u64 rather than the BN128 FrElement ABI
    // the custom gate is written against.
    let out = Command::new("circom")
        .current_dir(&circuits)
        .args([circuit.to_str().unwrap(), "--c", "--O2", "--prime", "goldilocks", "-o"])
        .arg(&dir)
        .args(["-l", circuits.to_str().unwrap(), "-l", blake3.to_str().unwrap()])
        .output()
        .expect("failed to run circom");
    assert!(
        out.status.success(),
        "circom failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // The generated file needs the goldilocks fr.hpp, which circom does not
    // include itself, and must sit outside its own directory so the quoted
    // include resolves to setup/circom's copy rather than the BN128 one.
    let generated = fs::read_to_string(dir.join("c_cpp/c.cpp")).expect("circom produced no C++");
    let mut lines: Vec<&str> = generated.lines().collect();
    let after_includes = lines.iter().position(|l| !l.starts_with("#include")).unwrap_or(0);
    lines.insert(after_includes, "#include \"fr.hpp\"");
    fs::write(dir.join("verifier.cpp"), lines.join("\n")).unwrap();
    fs::write(dir.join("driver.cpp"), driver_src).unwrap();

    let exe = dir.join("driver");
    let out = Command::new("g++")
        .args(["-std=c++17", "-O1", "-fPIC", "-mavx2", "-D__AVX2__", "-D__USE_ASSEMBLY__"])
        .args(["-flarge-source-files", "-w"])
        .arg("-I")
        .arg(&support)
        .arg("-I")
        .arg(&goldilocks)
        .arg(dir.join("driver.cpp"))
        .arg(dir.join("verifier.cpp"))
        .arg(support.join("calcwit.cpp"))
        .arg(support.join("main.cpp"))
        .arg(support.join("blake3_goldilocks.cpp"))
        .arg(goldilocks.join("goldilocks_base_field.cpp"))
        .arg("-o")
        .arg(&exe)
        .args(["-lgmp", "-lgmpxx", "-fopenmp", "-pthread"])
        .output()
        .expect("failed to run g++");
    assert!(out.status.success(), "g++ failed:\n{}", String::from_utf8_lossy(&out.stderr));
    (dir, exe)
}

/// Compile `src`, evaluate it on `inputs`, and return the main component's first
/// `n_out` output cells.
fn run_circuit(tag: &str, src: &str, inputs: &[u64], n_out: usize) -> Vec<u64> {
    let (dir, exe) = build(tag, src, DRIVER_CPP);

    let input_file = dir.join("in.txt");
    fs::write(&input_file, inputs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n")).unwrap();

    let out = Command::new(&exe)
        .arg(dir.join("c_cpp/c.dat"))
        .arg(&input_file)
        .arg(n_out.to_string())
        .output()
        .expect("failed to run witness driver");
    assert!(out.status.success(), "witness driver failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let got: Vec<u64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().parse().expect("driver printed a non-number"))
        .collect();
    assert_eq!(got.len(), n_out, "driver printed {} cells, wanted {n_out}", got.len());
    let _ = fs::remove_dir_all(&dir);
    got
}

/// Compile `src`, run `driver_src` against it, and return the driver's stdout.
/// The driver decides what to compare and exits non-zero on disagreement.
fn compile_and_run_cpp(tag: &str, src: &str, driver_src: &str) -> String {
    let (dir, exe) = build(tag, src, driver_src);
    let out = Command::new(&exe).arg(dir.join("c_cpp/c.dat")).output().expect("failed to run driver");
    let report = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "driver reported a mismatch:\n{report}{}", String::from_utf8_lossy(&out.stderr));
    let _ = fs::remove_dir_all(&dir);
    report
}

/// Deterministic full-range field elements, including values whose high half is
/// 2^32-1 -- the boundary where the gate's Goldilocks-to-u32 split is not
/// injective and its canonicity witness has to fire.
fn sample_inputs(n: usize) -> Vec<u64> {
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    (0..n)
        .map(|i| {
            let v = match i % 5 {
                0 => (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                1 => 0xFFFF_FFFF_0000_0000 + i as u64,
                2 => P - 1,
                3 => i as u64,
                _ => !((i as u64).wrapping_mul(7).wrapping_add(1)),
            };
            if v >= P {
                v - P
            } else {
                v
            }
        })
        .collect()
}

/// `LinearHash` is `Blake3Goldilocks::linearHash`, i.e. the 32-byte BLAKE3
/// digest of the leaf's words in canonical little-endian -- which is exactly
/// what `Blake3Transcript::get_state` returns.
fn reference_linear_hash(words: &[u64]) -> Vec<u64> {
    let mut t = Blake3Transcript::<Goldilocks>::new();
    t.put(&words.iter().map(|&w| Goldilocks::from_u64(w)).collect::<Vec<_>>());
    t.get_state().iter().map(|f| f.as_canonical_u64()).collect()
}

/// Sizes worth covering: below/at/above one 128-word chunk, and the +-1 chunk
/// boundaries where the tree shape changes and the root moves between
/// Blake3FinalizeChunk and Blake3FinalizeParent.
const LEAF_SIZES: &[usize] = &[1, 8, 9, 128, 129, 257];

#[test]
fn linear_hash_matches_reference_blake3() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP linear_hash_matches_reference_blake3: {why}");
        return;
    }

    let n_max = *LEAF_SIZES.iter().max().unwrap();
    let mut src = String::from(
        "pragma circom 2.1.0;\npragma custom_templates;\ninclude \"hash/blake3/linearhash.circom\";\n\ntemplate T() {\n",
    );
    src += &format!("    signal input in[{n_max}];\n");
    src += &format!("    signal output out[{}][4];\n", LEAF_SIZES.len());
    for (k, n) in LEAF_SIZES.iter().enumerate() {
        src += &format!("    signal w{k}[{n}][1];\n");
        src += &format!("    for (var i = 0; i < {n}; i++) {{ w{k}[i][0] <== in[i]; }}\n");
        src += &format!("    out[{k}] <== LinearHash({n}, 2, 1)(w{k});\n");
    }
    src += "}\ncomponent main = T();\n";

    let inputs = sample_inputs(n_max);
    let got = run_circuit("leaf", &src, &inputs, 4 * LEAF_SIZES.len());

    for (k, &n) in LEAF_SIZES.iter().enumerate() {
        let want = reference_linear_hash(&inputs[..n]);
        assert_eq!(&got[4 * k..4 * k + 4], &want[..], "LinearHash({n}) disagrees with reference BLAKE3");
    }
}

/// Absorb/squeeze script: word counts to absorb, then challenges to draw. The
/// total exceeds one 128-word chunk on purpose, so the emitter has to build a
/// cv stack and squeeze through Blake3FinalizeParent rather than only
/// Blake3FinalizeChunk -- i.e. so the raw=1 gate shape is exercised at all.
const SCRIPT: &[(usize, usize)] = &[(3, 1), (5, 1), (120, 1), (8, 3), (1, 1)];

#[test]
fn transcript_matches_reference_blake3() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP transcript_matches_reference_blake3: {why}");
        return;
    }

    let n_words: usize = SCRIPT.iter().map(|(n, _)| n).sum();
    let n_challenges: usize = SCRIPT.iter().map(|(_, g)| g).sum();
    assert!(n_words > 128, "script must cross a chunk boundary to exercise raw=1");

    // Drive the emitter and the reference through the identical script.
    let inputs = sample_inputs(n_words);
    let mut emitter = Transcript::new(None, "blake3");
    let mut oracle = Blake3Transcript::<Goldilocks>::new();
    let mut want: Vec<u64> = Vec::with_capacity(3 * n_challenges);

    let mut w = 0usize;
    let mut c = 0usize;
    for (absorb, squeeze) in SCRIPT {
        for _ in 0..*absorb {
            emitter.put_single(&format!("in[{w}]"));
            oracle.put(&[Goldilocks::from_u64(inputs[w])]);
            w += 1;
        }
        for _ in 0..*squeeze {
            emitter.get_field(&format!("out[{c}]"));
            let mut got = [Goldilocks::from_u64(0); 3];
            oracle.get_field(&mut got);
            want.extend(got.iter().map(|f| f.as_canonical_u64()));
            c += 1;
        }
    }

    let body = emitter.get_code();
    assert!(body.contains("Blake3FinalizeParent()("), "script did not reach a parent-rooted squeeze:\n{body}");

    let src = format!(
        "pragma circom 2.1.0;\npragma custom_templates;\ninclude \"hash/blake3/blake3.circom\";\n\n\
         template T() {{\n    signal input in[{n_words}];\n    signal output out[{n_challenges}][3];\n{body}\n}}\n\
         component main = T();\n"
    );

    let got = run_circuit("transcript", &src, &inputs, 3 * n_challenges);
    assert_eq!(got, want, "emitted transcript disagrees with reference BLAKE3\n{src}");
}

/// Every `im` cell the gate writes, compared between the AVX2 store path and the
/// `#else` scalar fallback.
///
/// These must agree bit for bit: they are two spellings of one decomposition.
/// The end-to-end tests above cannot see a disagreement, because `im` holds the
/// byte and limb witnesses the AIR range-checks while the digest is computed
/// independently of them -- so corrupting a `put_bytes` shift silently changes
/// every affected AIR cell and leaves the hash correct.
///
/// Needs only a C++ compiler, not circom.
const CELL_CMP_CPP: &str = r##"
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <sys/types.h>   // uint, as the gate's extern signature spells it
static const uint64_t GP = 18446744069414584321ull;
extern void Blake3Compress(uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,
                           uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*);
extern void Blake3Compress_scalar(uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,
                                  uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*);
int main() {
    uint s16 = 16, s3080 = 3080, sO = 16, s1 = 1;
    long cases = 0, bad = 0;
    for (int trial = 0; trial < 400; trial++) {
        uint64_t in[16];
        for (int i = 0; i < 16; i++) {
            uint64_t v = (uint64_t)(trial * 16 + i + 1) * 0x9E3779B97F4A7C15ull;
            // Force the packing boundary (hi = 2^32-1) into some block words.
            if (i >= 8 && (trial % 7) == (i % 7)) v = 0xFFFFFFFF00000000ull + (trial % 3);
            in[i] = (i < 8) ? (v >> 32) : (v >= GP ? v - GP : v);
        }
        for (uint64_t raw = 0; raw < 2; raw++)
        for (uint64_t key = 0; key < 2; key++)
        for (uint64_t bl : {8ull, 40ull, 64ull})
        for (uint64_t fl : {1ull, 2ull, 3ull, 11ull, 12ull}) {
            uint64_t a[3080] = {0}, b[3080] = {0}, oa[16] = {0}, ob[16] = {0};
            uint64_t ctr = trial % 97, r = raw, k = key, l = bl, f = fl;
            Blake3Compress       (a,&s3080,oa,&sO,in,&s16,&l,&s1,&ctr,&s1,&f,&s1,&k,&s1,&r,&s1);
            Blake3Compress_scalar(b,&s3080,ob,&sO,in,&s16,&l,&s1,&ctr,&s1,&f,&s1,&k,&s1,&r,&s1);
            cases++;
            for (int i = 0; i < 3080; i++) if (a[i] != b[i]) {
                if (bad < 3) printf("im[%d] avx=%llu scalar=%llu raw=%llu key=%llu blockLen=%llu flags=%llu\n",
                                    i, (unsigned long long)a[i], (unsigned long long)b[i],
                                    (unsigned long long)raw, (unsigned long long)key,
                                    (unsigned long long)bl, (unsigned long long)fl);
                bad++; break;
            }
            for (int i = 0; i < 16; i++) if (oa[i] != ob[i]) {
                if (bad < 3) printf("out[%d] differs raw=%llu\n", i, (unsigned long long)raw);
                bad++; break;
            }
        }
    }
    printf("compared %ld invocations, %ld mismatching\n", cases, bad);
    return bad != 0;
}
"##;

#[test]
fn gate_witness_cells_match_scalar_path() {
    if !have("g++", &["--version"]) {
        eprintln!("SKIP gate_witness_cells_match_scalar_path: g++ not on PATH");
        return;
    }
    let root = repo_root();
    let support = root.join("setup/circom");
    let goldilocks = root.join("pil2-stark/src/goldilocks/src");
    let gate = support.join("blake3_goldilocks.cpp");
    if !gate.exists() {
        eprintln!("SKIP gate_witness_cells_match_scalar_path: {} not present", gate.display());
        return;
    }

    let dir = std::env::temp_dir().join("b3_equiv_cells");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("cmp.cpp"), CELL_CMP_CPP).unwrap();

    // The fallback is selected by __AVX2__, which gcc defines from -mavx2 alone,
    // so the scalar object must be built with neither. Only the symbol is
    // renamed, so both objects really are the same source.
    let scalar_obj = dir.join("scalar.o");
    let out = Command::new("g++")
        .args(["-std=c++17", "-O2", "-D__USE_ASSEMBLY__", "-DBlake3Compress=Blake3Compress_scalar", "-w", "-c"])
        .arg(&gate)
        .arg("-I")
        .arg(&support)
        .arg("-I")
        .arg(&goldilocks)
        .arg("-o")
        .arg(&scalar_obj)
        .output()
        .expect("failed to run g++");
    assert!(out.status.success(), "scalar build failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let exe = dir.join("cmp");
    let out = Command::new("g++")
        .args(["-std=c++17", "-O2", "-mavx2", "-D__AVX2__", "-D__USE_ASSEMBLY__", "-w"])
        .arg("-I")
        .arg(&support)
        .arg("-I")
        .arg(&goldilocks)
        .arg(dir.join("cmp.cpp"))
        .arg(&scalar_obj)
        .arg(&gate)
        .arg(goldilocks.join("goldilocks_base_field.cpp"))
        .arg("-o")
        .arg(&exe)
        .args(["-lgmp", "-lgmpxx", "-fopenmp", "-pthread"])
        .output()
        .expect("failed to run g++");
    assert!(out.status.success(), "avx2 build failed:\n{}", String::from_utf8_lossy(&out.stderr));

    let out = Command::new(&exe).output().expect("failed to run comparator");
    let report = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "gate witness cells differ between store paths:\n{report}");
    assert!(report.contains(", 0 mismatching"), "unexpected comparator output:\n{report}");
    let _ = fs::remove_dir_all(&dir);
}

/// Driver for `spec_and_witness_gate_agree`: runs the probe circuit (which
/// executes the circom `b3_compress_gate`) and the C++ `Blake3Compress` on the
/// same arguments, comparing all 3096 values.
const SPEC_CMP_CPP: &str = r##"
#include <cstdint>
#include <cstdio>
#include <vector>
#include <sys/types.h>
extern "C" uint64_t getSizeWitness();
extern "C" void *initCircuit(char *);
extern "C" int64_t getWitness(uint64_t *, void *, void *, uint64_t);
extern void Blake3Compress(uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,
                           uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*);
static const uint64_t GP = 18446744069414584321ull;
int main(int argc, char **argv) {
    void *circuit = initCircuit(argv[1]);
    if (!circuit) { fprintf(stderr, "initCircuit failed\n"); return 1; }
    uint s16 = 16, s3080 = 3080, sO = 16, s1 = 1;
    long bad = 0, cases = 0;
    for (int trial = 0; trial < 12; trial++)
    for (uint64_t raw = 0; raw < 2; raw++)
    for (uint64_t key = 0; key < 2; key++)
    for (uint64_t bl : {8ull, 64ull}) {
        uint64_t args[21];
        for (int i = 0; i < 16; i++) {
            uint64_t v = (uint64_t)(trial * 16 + i + 1) * 0x9E3779B97F4A7C15ull;
            // Force the packing boundary into some block words, so the split
            // section's isMax branch is exercised alongside the dInv one.
            if (i >= 8 && (trial % 5) == (i % 5)) v = 0xFFFFFFFF00000000ull + (trial % 3);
            // cv words and raw blocks are u32; a Goldilocks block is full-range.
            args[i] = (i < 8 || raw) ? (v >> 32) : (v >= GP ? v - GP : v);
        }
        args[16] = bl; args[17] = trial; args[18] = (trial % 2) ? 3 : 11;
        args[19] = key; args[20] = raw;

        std::vector<uint64_t> w(getSizeWitness());
        if (getWitness(args, circuit, w.data(), 8) != 0) { fprintf(stderr, "getWitness failed\n"); return 1; }

        uint64_t im[3080] = {0}, out[16] = {0}, in16[16];
        for (int i = 0; i < 16; i++) in16[i] = args[i];
        uint64_t l = args[16], c = args[17], f = args[18], k = key, r = raw;
        Blake3Compress(im,&s3080,out,&sO,in16,&s16,&l,&s1,&c,&s1,&f,&s1,&k,&s1,&r,&s1);

        cases++;
        for (int i = 0; i < 3096; i++) {
            const uint64_t want = (i < 3080) ? im[i] : out[i - 3080];
            if (w[1 + i] != want) {
                if (bad < 6) printf("cell %d circom=%llu cpp=%llu raw=%llu key=%llu blockLen=%llu\n",
                    i, (unsigned long long)w[1 + i], (unsigned long long)want,
                    (unsigned long long)raw, (unsigned long long)key, (unsigned long long)bl);
                bad++;
            }
        }
    }
    printf("compared %ld gate evaluations, %ld differing cells\n", cases, bad);
    return bad != 0;
}
"##;

/// blake3_core.circom's `b3_compress_gate` against blake3_goldilocks.cpp's
/// `Blake3Compress`, all 3096 values.
///
/// These are two independent implementations of one function, and normally only
/// the C++ one runs: the circom gate body is dead under `extern_c`, replaced by
/// a call into the witness library. But the circom side is the readable spec,
/// it is where the `im` layout is documented, and the PIL will be written from
/// it -- so a divergence would mean constraining a layout the prover does not
/// produce, with nothing to catch it. Instantiating the function in a plain
/// (non-custom) template makes it execute, so the two can be differenced.
#[test]
fn spec_and_witness_gate_agree() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP spec_and_witness_gate_agree: {why}");
        return;
    }
    let src = "pragma circom 2.1.0;\n\
        include \"hash/blake3/blake3_core.circom\";\n\n\
        template Probe() {\n\
        \x20   signal input args[21];   // in[0..16], then blockLen, counterLo, flags, key, raw\n\
        \x20   signal output r[3096];\n\
        \x20   var iv[16];\n\
        \x20   for (var i = 0; i < 16; i++) { iv[i] = args[i]; }\n\
        \x20   var res[3096] = b3_compress_gate(iv, args[16], args[17], 0, args[18], args[19], args[20]);\n\
        \x20   for (var i = 0; i < 3096; i++) { r[i] <-- res[i]; }\n\
        }\n\
        component main = Probe();\n";

    let report = compile_and_run_cpp("spec", src, SPEC_CMP_CPP);
    assert!(report.contains(", 0 differing cells"), "spec and witness gate diverge:\n{report}");
}
