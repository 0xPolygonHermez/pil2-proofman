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
//! Three narrower tests cover what the two above cannot reach.
//! `spec_and_witness_gate_agree` differences the gate against the circom function
//! that is its spec, over both input shapes, both key orderings and the packing
//! boundary. `node_gate_matches_general_gate` holds Blake3Node to the general gate
//! driven into the same shape. `lattice_chain_step_matches_prover` holds the
//! emitted contributions chain to the prover's, which nothing else checks: the
//! compressor publishes it as an unconstrained output, so a divergence would only
//! surface a recursion level later.
//!
//! Requires the goldilocks `circom` fork and a C++ toolchain. When either is
//! missing the tests print why and pass, so `cargo test --workspace` still works
//! on a machine (or CI runner) without them.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use proofman_fields::{hash_state, Blake3Transcript, Goldilocks, PrimeField64};

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
    let gate = repo_root().join("setup/circom/blake3_gate.cpp");
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
        .arg(support.join("blake3_gate.cpp"))
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
const LEAF_SIZES: &[usize] = &[1, 8, 9, 128, 129, 256, 257, 512];

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

/// One round of the contributions lattice chain, circom against the prover.
///
/// `gen_calculate_hashes` emits the blake3 chain as `Blake3Permute8` rounds eight words
/// wide; the prover steps it with `hash_state` at the width
/// `TranscriptDyn::get_chain_state` returns. Those are two separate implementations of one
/// function and they must agree, or prover and verifier derive different contributions --
/// and nothing downstream would catch it, because the compressor emits `sv_stage1Hash` as
/// an unconstrained output. Its values are only consumed a recursion level later.
///
/// The seed is checked the same way: the chain starts from the eight words the transcript
/// squeezes, which is what the emitted circom reads one field at a time.
#[test]
fn lattice_chain_step_matches_prover() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP lattice_chain_step_matches_prover: {why}");
        return;
    }
    let src = "pragma circom 2.1.0;\n\
        pragma custom_templates;\n\
        include \"hash/blake3/blake3.circom\";\n\n\
        template Probe() {\n\
        \x20   signal input a[8];\n\
        \x20   signal output o[8];\n\
        \x20   o <== Blake3Permute8()(a);\n\
        }\n\
        component main = Probe();\n";

    // Words that straddle the packing boundary, where a split can disagree.
    let mut words = sample_inputs(8);
    words[2] = Goldilocks::ORDER_U64 - 1;
    words[5] = 0xFFFFFFFF_00000000;

    let got = run_circuit("lattice", src, &words, 8);

    let mut state: Vec<Goldilocks> = words.iter().map(|&w| Goldilocks::from_u64(w)).collect();
    hash_state("blake3", &mut state);
    let want: Vec<u64> = state.iter().map(|f| f.as_canonical_u64()).collect();

    assert_eq!(got, want, "circom lattice round disagrees with the prover's hash_state");
}

/// `Blake3Node` against the general gate driven into the same shape.
///
/// Blake3Node exists only to spend fewer wires on a node hash than
/// Blake3Compress does (see blake3.circom), so it has to compute exactly what
/// the general gate computes at the IV with the root flags -- including the key
/// ordering and the packing of the digest halves. Both gates run for real here:
/// circom emits the circuit, it links against setup/circom, and the two halves
/// of the output must agree.
///
/// Since Blake3Compress carries no key, the two sides no longer share the ordering path: Node
/// exercises `split_ordered`'s key branch while the general gate is driven with the halves
/// already ordered by the caller. A flipped ordering inside `split_ordered` therefore shows up
/// here, which it could not when both sides went through it.
#[test]
fn node_gate_matches_general_gate() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP node_gate_matches_general_gate: {why}");
        return;
    }
    let src = "pragma circom 2.1.0;\n\
        pragma custom_templates;\n\
        include \"hash/blake3/blake3.circom\";\n\n\
        template Probe() {\n\
        \x20   signal input args[17];  // Node in[0..8], key at 8, pre-ordered block at 9..17\n\
        \x20   signal output o[8];     // Blake3Node, then the general gate\n\
        \x20   component n = Blake3Node();\n\
        \x20   for (var i = 0; i < 8; i++) { n.in[i] <== args[i]; }\n\
        \x20   n.key <== args[8];\n\
        \x20   for (var i = 0; i < 4; i++) { o[i] <== n.out[i]; }\n\
        \x20   component c = Blake3Compress(B3_CHUNK_START() + B3_CHUNK_END() + B3_ROOT(), 0);\n\
        \x20   for (var i = 0; i < 8; i++) { c.in[i] <== B3_IV(i); c.in[8 + i] <== args[9 + i]; }\n\
        \x20   c.blockLen <== 64;\n\
        \x20   c.counterLo <== 0;\n\
        \x20   for (var i = 0; i < 4; i++) { o[4 + i] <== c.out[2 * i] + 4294967296 * c.out[2 * i + 1]; }\n\
        \x20   for (var i = 8; i < 16; i++) { _ <== c.out[i]; }\n\
        }\n\
        component main = Probe();\n";

    // Both key orderings, and words that straddle the packing boundary
    // (hi = 2^32-1), which is where a split can disagree.
    let mut words = sample_inputs(8);
    words[3] = Goldilocks::ORDER_U64 - 1;
    words[6] = 0xFFFFFFFF_00000000;
    for key in [0u64, 1] {
        let mut inputs = words.clone();
        inputs.push(key);
        // Blake3Compress has no key, so the ordering it would have applied is applied here.
        if key == 1 {
            inputs.extend_from_slice(&words[4..8]);
            inputs.extend_from_slice(&words[0..4]);
        } else {
            inputs.extend_from_slice(&words);
        }
        let got = run_circuit("node", src, &inputs, 8);
        assert_eq!(
            &got[0..4],
            &got[4..8],
            "Blake3Node disagrees with the general gate at key={key}\n  node={:?}\n  general={:?}",
            &got[0..4],
            &got[4..8]
        );
    }
}

/// Driver for `spec_and_witness_gate_agree`: runs the probe circuit (which
/// executes the circom `b3_compress_gate`) and the C++ `Blake3Compress` on the
/// same arguments, comparing the sixteen xof words the gate publishes.
const SPEC_CMP_CPP: &str = r##"
#include <cstdint>
#include <cstdio>
#include <vector>
#include <sys/types.h>
extern "C" uint64_t getSizeWitness();
extern "C" void *initCircuit(char *);
extern "C" int64_t getWitness(uint64_t *, void *, void *, uint64_t);
// flags and isParent are TEMPLATE PARAMETERS of the gate, so the fork passes them by value as the
// two leading arguments of the single generated function.
extern void Blake3Compress(uint64_t,uint64_t,uint64_t*,uint*,uint64_t*,uint*,uint64_t*,uint*,
                           uint64_t*,uint*);
static const uint64_t GP = 18446744069414584321ull;
int main(int argc, char **argv) {
    void *circuit = initCircuit(argv[1]);
    if (!circuit) { fprintf(stderr, "initCircuit failed\n"); return 1; }
    uint s16 = 16, sO = 16, s1 = 1;
    long bad = 0, cases = 0;
    for (int trial = 0; trial < 12; trial++)
    for (uint64_t isParent = 0; isParent < 2; isParent++)
    // 8 and 64 are what callers produce; 300 and 511 are out of the u8 domain, where the spec
    // used not to mask and the C++ did -- they are here so that divergence cannot come back.
    for (uint64_t bl : {8ull, 64ull, 300ull, 511ull}) {
        uint64_t args[20];
        for (int i = 0; i < 16; i++) {
            uint64_t v = (uint64_t)(trial * 16 + i + 1) * 0x9E3779B97F4A7C15ull;
            // Force the packing boundary (hi = 2^32-1) into some block words:
            // p-1 = (2^32-1)*2^32, so that is where the split can disagree.
            if (i >= 8 && (trial % 5) == (i % 5)) v = 0xFFFFFFFF00000000ull + (trial % 3);
            // cv words and parent blocks are u32; a chunk's Goldilocks block is full-range.
            args[i] = (i < 8 || isParent) ? (v >> 32) : (v >= GP ? v - GP : v);
        }
        args[16] = bl; args[17] = trial; args[18] = (trial % 3 == 0) ? 300 : ((trial % 2) ? 3 : 11);
        args[19] = isParent;

        std::vector<uint64_t> w(getSizeWitness());
        if (getWitness(args, circuit, w.data(), 8) != 0) { fprintf(stderr, "getWitness failed\n"); return 1; }

        uint64_t out[16] = {0}, in16[16];
        for (int i = 0; i < 16; i++) in16[i] = args[i];
        uint64_t l = args[16], c = args[17], f = args[18], ip = isParent;
        Blake3Compress(f,ip,out,&sO,in16,&s16,&l,&s1,&c,&s1);

        cases++;
        for (int i = 0; i < 16; i++) {
            const uint64_t want = out[i];
            if (w[1 + i] != want) {
                if (bad < 6) printf("cell %d circom=%llu cpp=%llu isParent=%llu blockLen=%llu\n",
                    i, (unsigned long long)w[1 + i], (unsigned long long)want,
                    (unsigned long long)isParent, (unsigned long long)bl);
                bad++;
            }
        }
    }
    printf("compared %ld gate evaluations, %ld differing cells\n", cases, bad);
    return bad != 0;
}
"##;

/// blake3_core.circom's `b3_compress_gate` against blake3_gate.cpp's
/// `Blake3Compress`, over the sixteen xof words.
///
/// These are two independent implementations of one function, and normally only
/// the C++ one runs: the circom gate body is dead under `extern_c`, replaced by
/// a call into the witness library. But the circom side is the readable spec,
/// and the PIL will be written from it -- so a divergence would mean
/// constraining something the prover does not produce, with nothing to catch
/// it. Instantiating the function in a plain (non-custom) template makes it
/// execute, so the two can be differenced. Covers both input shapes, both key
/// orderings and the packing boundary, which is where the split can disagree.
#[test]
fn spec_and_witness_gate_agree() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP spec_and_witness_gate_agree: {why}");
        return;
    }
    let src = "pragma circom 2.1.0;\n\
        include \"hash/blake3/blake3_core.circom\";\n\n\
        template Probe() {\n\
        \x20   signal input args[20];   // in[0..16], then blockLen, counterLo, flags, isParent\n\
        \x20   signal output r[16];\n\
        \x20   var iv[16];\n\
        \x20   for (var i = 0; i < 16; i++) { iv[i] = args[i]; }\n\
        \x20   var res[16] = b3_compress_gate(iv, args[16], args[17], 0, args[18], 0, args[19]);\n\
        \x20   for (var i = 0; i < 16; i++) { r[i] <-- res[i]; }\n\
        }\n\
        component main = Probe();\n";

    let report = compile_and_run_cpp("spec", src, SPEC_CMP_CPP);
    assert!(report.contains(", 0 differing cells"), "spec and witness gate diverge:\n{report}");
}

/// The AIR splits a Goldilocks word into two u32s by constraining `v === lo + 2^32*hi`, but that
/// alone does not pin `(lo, hi)`: `lo + 2^32*hi` covers `[0, 2^64)` bijectively, so `v` and `v + p`
/// are both representable whenever `v + p < 2^64`, i.e. for every `v <= 2^32 - 2`. A prover taking
/// the alias would feed BLAKE3 a different byte string than the verifier hashed.
///
/// `p - 1 = 2^32 * (2^32 - 1)` exactly, so every alias has `hi = 2^32 - 1` with `lo != 0`, and the
/// only canonical word with `hi = 2^32 - 1` is `p - 1`, which has `lo = 0`. The AIR therefore
/// enforces `hi != 2^32 - 1 || lo == 0`. This pins that predicate against `b3_split_word`.
#[test]
fn canonical_split_predicate_matches_b3_split_word() {
    const P: u128 = (1u128 << 64) - (1u128 << 32) + 1;
    const HI_MAX: u64 = 0xFFFF_FFFF;

    // The predicate blake3SplitCanonical enforces.
    fn air_accepts(lo: u64, hi: u64) -> bool {
        lo <= HI_MAX && hi <= HI_MAX && (hi != HI_MAX || lo == 0)
    }
    // What b3_split_word computes, on the canonical representative.
    fn canonical_split(v: u64) -> (u64, u64) {
        (v & 0xFFFF_FFFF, v >> 32)
    }

    // Boundaries: 0, the last aliasable word, the first non-aliasable one, and p - 1 (the only
    // canonical word with hi at its maximum).
    for v in [0u64, 1, 0xFFFF_FFFE, HI_MAX, 1 << 32, (P - 1) as u64] {
        let (lo, hi) = canonical_split(v);
        assert!(air_accepts(lo, hi), "canonical split of {v:#x} rejected: lo={lo:#x} hi={hi:#x}");
        assert_eq!(lo as u128 + ((hi as u128) << 32), v as u128, "split of {v:#x} does not repack");
    }

    // Every alias must be rejected, and an alias only exists for v <= 2^32 - 2.
    for v in [0u64, 1, 2, 0xFFFF_FFFD, 0xFFFF_FFFE] {
        let aliased = v as u128 + P;
        assert!(aliased < 1u128 << 64, "alias of {v:#x} must still fit a u64");
        let (lo, hi) = ((aliased & 0xFFFF_FFFF) as u64, (aliased >> 32) as u64);
        assert!(!air_accepts(lo, hi), "alias of {v:#x} accepted: lo={lo:#x} hi={hi:#x}");
        assert_eq!(hi, HI_MAX, "alias of {v:#x} should sit at hi = 2^32 - 1");
    }

    // The AIR does not spell the predicate out -- it enforces the single constraint
    //     sel * (lo * (d*dinv - 1)) === 0,   d = hi - (2^32 - 1)
    // and leaves `dinv` to the prover. So the claim that actually needs pinning is that a prover can
    // satisfy that constraint for exactly the pairs the predicate accepts: freely when lo = 0, and
    // otherwise only by exhibiting an inverse of d, which does not exist at hi = 2^32 - 1.
    fn constraint_is_satisfiable(lo: u64, hi: u64) -> bool {
        if lo == 0 {
            return true; // the factor lo vanishes; any dinv works
        }
        let d = (hi as u128 + P - HI_MAX as u128) % P;
        d != 0 // dinv = d^-1 exists iff d is nonzero
    }
    for (lo, hi) in [
        (0u64, HI_MAX),        // p - 1, the one canonical word at hi max
        (1, HI_MAX),           // the alias of 0
        (0xFFFF_FFFE, HI_MAX), // the alias of 2^32 - 3
        (0, 0),
        (1, 0),
        (HI_MAX, HI_MAX - 1), // legal: lo nonzero, hi one below max
        (HI_MAX, 0),
    ] {
        assert_eq!(
            constraint_is_satisfiable(lo, hi),
            air_accepts(lo, hi),
            "one-constraint gadget disagrees with the predicate at lo={lo:#x} hi={hi:#x}"
        );
    }

    // And a sweep, so the two clauses above are not the only evidence.
    let mut v: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..20_000 {
        v = v.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) % (P as u64);
        let (lo, hi) = canonical_split(v);
        assert!(air_accepts(lo, hi), "canonical {v:#x} rejected");
        let aliased = v as u128 + P;
        if aliased < 1u128 << 64 {
            let (alo, ahi) = ((aliased & 0xFFFF_FFFF) as u64, (aliased >> 32) as u64);
            assert!(!air_accepts(alo, ahi), "alias of {v:#x} accepted");
        }
    }
}

/// The key bit acts as exactly a swap of the block's two 4-word halves.
///
/// `blake3OrderByKey` in circuits/blake3.pil encodes that as
///     ordered[i] = (1-key)*input[i] + key*input[4+i]
///     ordered[4+i] = (1-key)*input[4+i] + key*input[i]
/// Blake3Node is the gate that carries a key -- Blake3Compress has none -- so this drives Node,
/// which is also the only AIR kind that applies the ordering. The AIR is right only if key = 1
/// gives the same digest as
/// pre-swapping the halves and driving it with key = 0. That is what this checks, against the
/// gate as built -- if `split_ordered` applied any other permutation (a rotation, say) the two
/// sides would diverge.
///
/// Unlike `node_gate_matches_general_gate` this is not comparing two gates that share the
/// ordering code: one side exercises the ordering, the other bypasses it.
#[test]
fn key_bit_is_exactly_a_half_swap() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("SKIP key_bit_is_exactly_a_half_swap: {why}");
        return;
    }
    let src = "pragma circom 2.1.0;\n\
        pragma custom_templates;\n\
        include \"hash/blake3/blake3.circom\";\n\n\
        template Probe() {\n\
        \x20   signal input args[9];   // block[0..8], then key\n\
        \x20   signal output o[4];\n\
        \x20   component c = Blake3Node();\n\
        \x20   for (var i = 0; i < 8; i++) { c.in[i] <== args[i]; }\n\
        \x20   c.key <== args[8];\n\
        \x20   o <== c.out;\n\
        }\n\
        component main = Probe();\n";

    // Include words at the packing boundary, where a split can disagree.
    let mut words = sample_inputs(8);
    words[2] = Goldilocks::ORDER_U64 - 1;
    words[5] = 0xFFFFFFFF_00000000;

    // key = 1 on the block as given
    let mut keyed = words.clone();
    keyed.push(1);
    let with_key = run_circuit("keyswap_k1", src, &keyed, 4);

    // key = 0 on the block with its halves already swapped
    let mut swapped: Vec<u64> = words[4..8].to_vec();
    swapped.extend_from_slice(&words[0..4]);
    swapped.push(0);
    let pre_swapped = run_circuit("keyswap_k0", src, &swapped, 4);

    assert_eq!(
        with_key, pre_swapped,
        "key=1 is not a plain half-swap, so blake3OrderByKey's formula is wrong\n  \
         key=1:      {with_key:?}\n  pre-swapped: {pre_swapped:?}"
    );
}
