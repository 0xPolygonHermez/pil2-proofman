//! The domains example (`examples/domains`) end to end, one stage per task that lands domain support:
//! eight AIRs, each exercising one domain shape, so a failure points at one shape.
//!
//! - Stage 0: setup (Task 3). The boundaries and the `qDeg` of each starkinfo.
//! - Stage 1: the CPU prover, no aggregation (Task 4): `verify-constraints`, `prove --verify-proofs`
//!   and tampered witnesses (`DOMAINS_TAMPER`, see `examples/domains/src/tamper.rs`).
//! - Stage 2: the GPU prover, no aggregation (Task 5): the same, plus GPU proofs byte-identical to CPU ones
//!   (each run's basic proofs dumped with `PIL2_DUMP_ZKIN=basic`).
//! - Stage 3: aggregation (Tasks 6-7): `setup -r`, `prove --aggregation --verify-proofs`, then `verify-stark`
//!   (the C++ verifier, through FFI) and the emitted native Rust verifier on the vadcop_final proof; on the
//!   GPU, tampered witnesses must be rejected (or not) by the recursive1 circuit.
//!
//! The native Rust verifier on a domain proof is `tests/domain_native_e2e.rs` (`examples/domain-native`): these
//! AIRs carry a std bus, which that verifier does not replay.
//!
//! The pilout is checked in (`examples/domains/pil/build.pilout`): it needs the domain-aware
//! compiler. Keys go to `examples/domains/build` (stages 0-2) and `build-recursive` (stage 3).
//!
//! GPU commands (`--gpu`, `gen-exps`) run one at a time across the test threads, so the default
//! parallel runner is safe.
//!
//! ```text
//! cargo test --release -p proofman-cli --test domains_e2e [-- --ignored]
//! ```

#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use pil2_stark_setup::commands::setup::{run_setup, SetupOptions};
use proofman_common::hash_family::{compressed_final_by_default, default_aggregation_arity, DEFAULT_HASH_ID};
use serde_json::{json, Value};

const AIRS: [&str; 8] =
    ["DomFirst", "DomLast", "DomNotFirst", "DomNotLast", "DomOdd", "DomLimits16", "DomNotQuarter", "DomMixed"];

/// `(cycleBits, offsets, complement)` of a normalized domain boundary (N = 2^10).
type Shape = (u64, &'static [u64], bool);

/// Each AIR's domain boundaries after `everyRow`, as setup normalizes and deduplicates them.
fn expected_domains(air: &str) -> Vec<Shape> {
    match air {
        "DomFirst" => vec![(10, &[0], false)],
        "DomLast" => vec![(10, &[1023], false)],
        "DomNotFirst" => vec![(10, &[0], true)],
        "DomNotLast" => vec![(10, &[1023], true)],
        "DomOdd" => vec![(1, &[1], false)],
        "DomLimits16" => vec![(4, &[0, 15], false)],
        "DomNotQuarter" => vec![(2, &[0], true)],
        // EVEN = [1,0] and EVEN4 = [1,0,1,0] are one boundary.
        "DomMixed" => vec![(10, &[0], false), (10, &[0], true), (1, &[0], false)],
        _ => unreachable!("{air}"),
    }
}

/// `(qDeg, im pols)` of each AIR. Blowup 1 bounds the degree at 3. A constraint of degree d on a domain of
/// R rows has degree d + 1, or d when R + d > N. Every AIR also carries the self-permutation's gsum, which
/// adds one im pol (`<Air>.ImPol`, the gsum's) in the qDeg 1 AIRs; in DomLimits16 and DomMixed qDeg is 2, the
/// gsum needs no im pol, and the one im pol is the promoted domain constraint (see expressionsinfo `imPol`).
fn expected_q_deg_and_im_pols(air: &str) -> (u64, usize) {
    match air {
        // degree <= 2 after the zerofier; the one im pol is the gsum's.
        "DomFirst" | "DomLast" | "DomNotLast" | "DomOdd" | "DomNotQuarter" | "DomNotFirst" => (1, 1),
        // s * s * (1 - s): degree 3 on 128 rows is 4, past qDeg 2, so the constraint itself is the im pol.
        "DomLimits16" => (2, 1),
        // x^3 on FIRST is degree 4: promoted to an im pol; x^3 on !FIRST stays 3 and the gsum needs none.
        "DomMixed" => (2, 1),
        _ => unreachable!("{air}"),
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap()
}

fn example() -> PathBuf {
    repo_root().join("examples/domains")
}

/// Runs `proofman-setup setup -a pil/build.pilout -b <build> [-r]` with the CLI's defaults.
fn setup(build: &Path, recursive: bool) -> PathBuf {
    let _ = fs::remove_dir_all(build);
    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let opts = SetupOptions {
        airout_path: example().join("pil/build.pilout").display().to_string(),
        build_dir: build.display().to_string(),
        fixed_dir: None,
        stark_structs_path: None,
        recursive,
        recursive_jobs: jobs.min(2),
        setup_jobs: jobs.min(4),
        stats_output_path: None,
        hash: DEFAULT_HASH_ID.to_string(),
        agg_arity: default_aggregation_arity(DEFAULT_HASH_ID),
        compressed_final: compressed_final_by_default(DEFAULT_HASH_ID),
        setup_version: None,
        recursive_n_bits: None,
        gen_exps: false,
        exps_arch: "auto".to_string(),
        exps_cap: 60000,
        exps_chunk: None,
        exps_stark_src: None,
    };
    // The recursive setup recurses deeper than a test thread's stack; the CLI runs it on 64 MB too.
    let result = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || run_setup(&opts).map_err(|e| format!("{e:#}")))
        .unwrap()
        .join()
        .unwrap();
    result.unwrap_or_else(|e| panic!("setup of {} failed: {e}", build.display()));
    build.join("provingKey")
}

/// The stage 0-2 key, set up once per test process.
fn key() -> &'static Path {
    static KEY: OnceLock<PathBuf> = OnceLock::new();
    KEY.get_or_init(|| setup(&example().join("build"), false))
}

/// The CUDA arch for `gen-exps` (its auto-detection fails on some hosts): `DOMAINS_CUDA_ARCH`, else the
/// first GPU's compute capability from `nvidia-smi` (12.0 -> sm_120).
fn cuda_arch() -> String {
    if let Ok(arch) = std::env::var("DOMAINS_CUDA_ARCH") {
        return arch;
    }
    let cap = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8_lossy(&o.stdout).lines().next().map(|l| l.trim().replace('.', "")))
        .filter(|cap| !cap.is_empty() && cap.chars().all(|c| c.is_ascii_digit()));
    match cap {
        Some(cap) => format!("sm_{cap}"),
        None => panic!("no CUDA arch for gen-exps: set DOMAINS_CUDA_ARCH (e.g. sm_120) or put nvidia-smi on PATH"),
    }
}

/// `gen-exps` of `key`: its `.exps.so` (the GPU's Q kernels).
fn gen_exps(key: &Path) {
    let arch = cuda_arch();
    assert_ok(&cli(&["gen-exps", "--proving-key", key.to_str().unwrap(), "--arch", &arch], None), "gen-exps");
}

/// The stage 0-2 key with its `.exps.so`, generated once per test process.
fn gpu_key() -> &'static Path {
    static GPU_KEY: OnceLock<PathBuf> = OnceLock::new();
    GPU_KEY.get_or_init(|| {
        let key = key();
        gen_exps(key);
        key.to_path_buf()
    })
}

/// The stage 3 key (`setup -r`), set up once per test process.
fn recursive_key() -> &'static Path {
    static KEY: OnceLock<PathBuf> = OnceLock::new();
    KEY.get_or_init(|| setup(&example().join("build-recursive"), true))
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|f| f == name) {
            return Some(path);
        }
    }
    None
}

fn starkinfo(key: &Path, air: &str) -> Value {
    let path = find_file(key, &format!("{air}.starkinfo.json")).unwrap_or_else(|| panic!("no starkinfo of {air}"));
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn expressionsinfo(key: &Path, air: &str) -> Value {
    let name = format!("{air}.expressionsinfo.json");
    let path = find_file(key, &name).unwrap_or_else(|| panic!("no expressionsinfo of {air}"));
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn witness_lib() -> PathBuf {
    witness_libraries::built_library("domains")
}

/// What a command wrote, stdout then stderr.
fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// Held around each GPU command: each process sizes its buffers from the device's free memory, so two
/// at once race for it.
fn gpu() -> MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `proofman-cli <args>`, with `DOMAINS_TAMPER` set for this process only; GPU commands one at a time.
fn cli(args: &[&str], tamper: Option<&str>) -> Output {
    let _gpu = (args.contains(&"--gpu") || args.first() == Some(&"gen-exps")).then(gpu);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_proofman-cli"));
    cmd.args(args).env("NO_COLOR", "1").env_remove("DOMAINS_TAMPER");
    if let Some(tamper) = tamper {
        cmd.env("DOMAINS_TAMPER", tamper);
    }
    cmd.output().expect("proofman-cli runs")
}

fn assert_ok(out: &Output, what: &str) {
    assert!(out.status.success(), "{what} failed:\n{}", output(out));
}

/// What verify-constraints prints when a constraint fails.
const CONSTRAINTS_REJECTED: &str = "Constraints were not verified";
/// What `prove --verify-proofs` prints when a basic proof does not verify.
const PROOFS_REJECTED: &str = "Basic proofs were not verified";

/// The command failed because the verifier rejected the witness or proof (`rejection`), not on a panic.
fn assert_rejected(out: &Output, what: &str, rejection: &str) {
    let text = output(out);
    assert!(!out.status.success(), "{what} succeeded, and should not:\n{text}");
    assert!(!text.contains("panicked"), "{what} panicked instead of being rejected:\n{text}");
    assert!(text.contains(rejection), "{what} failed without `{rejection}`:\n{text}");
}

/// A scratch directory under the target's temporary directory, removed when dropped.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("domains_e2e_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Public inputs that make the witness library emit only `airs`.
fn inputs(dir: &TestDir, airs: &[&str]) -> PathBuf {
    let path = dir.path(&format!("inputs_{}.json", airs.join("_")));
    fs::write(&path, json!({ "airs": airs }).to_string()).unwrap();
    path
}

fn all_inputs() -> PathBuf {
    example().join("src/inputs.json")
}

fn verify_constraints(key: &Path, inputs: &Path, gpu: bool, tamper: Option<&str>) -> Output {
    let mut args = vec!["verify-constraints".to_owned(), "--witness-lib".into(), witness_lib().display().to_string()];
    args.extend(["--proving-key".into(), key.display().to_string()]);
    args.extend(["--public-inputs".into(), inputs.display().to_string()]);
    if gpu {
        args.push("--gpu".into());
    }
    cli(&args.iter().map(String::as_str).collect::<Vec<_>>(), tamper)
}

fn prove(key: &Path, inputs: &Path, out_dir: &Path, gpu: bool, aggregation: bool, tamper: Option<&str>) -> Output {
    let mut args = vec!["prove".to_owned(), "--witness-lib".into(), witness_lib().display().to_string()];
    args.extend(["--proving-key".into(), key.display().to_string()]);
    args.extend(["--public-inputs".into(), inputs.display().to_string()]);
    args.extend(["--output-dir".into(), out_dir.display().to_string(), "--verify-proofs".into()]);
    if gpu {
        args.push("--gpu".into());
    }
    if aggregation {
        args.push("--aggregation".into());
    }
    cli(&args.iter().map(String::as_str).collect::<Vec<_>>(), tamper)
}

/// `prove --verify-proofs` of `inputs` with its basic proofs dumped into `dump` (`PIL2_DUMP_ZKIN=basic`):
/// the bytes a GPU proof must match.
fn prove_dumping_proofs(key: &Path, inputs: &Path, dump: &Path, gpu: bool) -> Output {
    fs::create_dir_all(dump).unwrap();
    let mut args = vec!["prove".to_owned(), "--witness-lib".into(), witness_lib().display().to_string()];
    args.extend(["--proving-key".into(), key.display().to_string()]);
    args.extend(["--public-inputs".into(), inputs.display().to_string()]);
    args.extend(["--output-dir".into(), dump.display().to_string(), "--verify-proofs".into()]);
    if gpu {
        args.push("--gpu".into());
    }
    let _gpu = gpu.then(self::gpu);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_proofman-cli"));
    cmd.args(&args).env("NO_COLOR", "1").env_remove("DOMAINS_TAMPER");
    cmd.env("PIL2_DUMP_ZKIN", "basic").env("TMPDIR", dump);
    cmd.output().expect("proofman-cli runs")
}

/// The basic proofs `prove_dumping_proofs` wrote, by file name.
fn dumped_proofs(dump: &Path) -> Vec<(String, Vec<u8>)> {
    let mut proofs: Vec<(String, Vec<u8>)> = fs::read_dir(dump)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().ends_with("_tbasic.bin"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), fs::read(&p).unwrap()))
        .collect();
    proofs.sort();
    proofs
}

/// Cells whose change breaks a domain constraint: `(tamper, the constraint as verify-constraints prints
/// it, its failing rows)`.
const TAMPER_INSIDE: [(&str, &str, &[u64]); 9] = [
    ("DomFirst:a:0:5", "a-1 == 0", &[0]),
    ("DomLast:b:1023:5", "b+0 == 0", &[1023]),
    // a[500] is read by `(a - 'a - 1)^2` on rows 500 and 501.
    ("DomNotFirst:a:500:0", "((a-'a)-1)*((a-'a)-1) == 0", &[500, 501]),
    ("DomNotQuarter:u:1:5", "u-7 == 0", &[1]),
    ("DomOdd:t:1:5", "t+0 == 0", &[1]),
    ("DomLimits16:s:15:5", "(s*s)*(1-s) == 0", &[15]),
    // a[500] is read by `a' === a + 1` on rows 499 and 500.
    ("DomNotLast:a:500:0", "a'-(a+1) == 0", &[499, 500]),
    // EVEN4 (block form) normalizes to EVEN: a constraint routed to another boundary misses row 2.
    ("DomMixed:f:2:5", "f+0 == 0", &[2]),
    // The FIRST constraint, promoted whole to an im pol; `v === x + w` fails on row 0 too.
    ("DomMixed:x:0:3", "((x*x)*x)-8 == 0", &[0]),
];

/// Cells outside every domain of their AIR: changing them must not matter.
const TAMPER_OUTSIDE: [&str; 3] = ["DomOdd:t:2:5", "DomFirst:a:1:5", "DomNotQuarter:u:4:5"];

/// verify-constraints names `constraint` as broken on `rows`, and only those.
fn assert_names_rows(tamper: &str, text: &str, constraint: &str, rows: &[u64]) {
    let header = format!("has {} invalid rows -> ", rows.len());
    let mut lines = text.lines().skip_while(|l| !(l.contains(&header) && l.trim_end().ends_with(constraint)));
    assert!(lines.next().is_some(), "{tamper}: `{constraint}` should fail on {} rows:\n{text}", rows.len());
    // its failing rows follow it
    let failed: Vec<&str> = lines.take(rows.len()).collect();
    for row in rows {
        let named = failed.iter().any(|l| l.contains(&format!("Failed at row {row} ")));
        assert!(named, "{tamper}: row {row} of `{constraint}` not named:\n{text}");
    }
}

// ---------------------------------------------------------------- Stage 0: setup (Task 3)

#[test]
fn stage0_setup_emits_each_domain_boundary_and_its_q_degree() {
    let key = key();
    for air in AIRS {
        let info = starkinfo(key, air);
        let boundaries = info["boundaries"].as_array().unwrap_or_else(|| panic!("{air}: no boundaries"));
        assert_eq!(boundaries[0], json!({ "name": "everyRow" }), "{air}: boundary 0");
        let domains: Vec<Value> = expected_domains(air)
            .into_iter()
            .map(|(cycle_bits, offsets, complement)| {
                json!({ "name": "domain", "cycleBits": cycle_bits, "offsets": offsets, "complement": complement })
            })
            .collect();
        let mut got = boundaries[1..].to_vec();
        let mut want = domains;
        // Boundary order is setup's; the set and its size are the contract.
        got.sort_by_key(|b| b.to_string());
        want.sort_by_key(|b| b.to_string());
        assert_eq!(got, want, "{air}: domain boundaries");

        // DomMixed's EVEN4 constraint, in block form, is a domain constraint too
        let constraints = expressionsinfo(key, air)["constraints"].as_array().unwrap().clone();
        let n_domain = constraints.iter().filter(|c| c["boundary"] == "domain").count();
        assert_eq!(n_domain, if air == "DomMixed" { 4 } else { 1 }, "{air}: domain constraints");

        // the self-permutation gives every AIR stage 2 and the `__L1__` fixed column (a verkey)
        assert_eq!(info["nStages"].as_u64(), Some(2), "{air}: nStages");
        assert!(info["nConstants"].as_u64().unwrap() >= 1, "{air}: nConstants");

        let im_pols =
            info["cmPolsMap"].as_array().unwrap().iter().filter(|p| p["imPol"].as_bool() == Some(true)).count();
        let q_deg = info["qDeg"].as_u64().unwrap();
        assert_eq!((q_deg, im_pols), expected_q_deg_and_im_pols(air), "{air}: (qDeg, im pols)");
    }
}

// ---------------------------------------------------------------- Stage 1: CPU prover (Task 4)

#[test]
fn stage1_cpu_verify_constraints_passes() {
    assert_ok(&verify_constraints(key(), &all_inputs(), false, None), "verify-constraints");
}

#[test]
fn stage1_cpu_proves_each_air() {
    let dir = TestDir::new("stage1_prove");
    for air in AIRS {
        let out = prove(key(), &inputs(&dir, &[air]), &dir.path(air), false, false, None);
        assert_ok(&out, &format!("prove of {air}"));
    }
}

/// Every AIR in one non-aggregated run.
#[test]
fn stage1_cpu_proves_all_airs_in_one_run() {
    let dir = TestDir::new("stage1_prove_all");
    assert_ok(&prove(key(), &all_inputs(), &dir.path("all"), false, false, None), "prove of every AIR");
}

/// DomNotLast's honest witness (a[i] = i + 5) breaks `a' === a + 1` only on the wrap row, outside !LAST.
#[test]
fn stage1_cpu_wrap_row_excluded() {
    let dir = TestDir::new("stage1_wrap");
    let inputs = inputs(&dir, &["DomNotLast"]);
    assert_ok(&verify_constraints(key(), &inputs, false, None), "verify-constraints of DomNotLast");
    assert_ok(&prove(key(), &inputs, &dir.path("DomNotLast"), false, false, None), "prove of DomNotLast");
}

#[test]
fn stage1_cpu_tamper_inside_a_domain_is_caught() {
    let dir = TestDir::new("stage1_inside");
    for (tamper, constraint, rows) in TAMPER_INSIDE {
        let air = tamper.split(':').next().unwrap();
        let inputs = inputs(&dir, &[air]);
        let out = verify_constraints(key(), &inputs, false, Some(tamper));
        assert_rejected(&out, &format!("verify-constraints with {tamper}"), CONSTRAINTS_REJECTED);
        assert_names_rows(tamper, &output(&out), constraint, rows);
        let out = prove(key(), &inputs, &dir.path(air), false, false, Some(tamper));
        assert_rejected(&out, &format!("prove with {tamper}"), PROOFS_REJECTED);
    }
}

#[test]
fn stage1_cpu_tamper_outside_every_domain_still_verifies() {
    let dir = TestDir::new("stage1_outside");
    for tamper in TAMPER_OUTSIDE {
        let air = tamper.split(':').next().unwrap();
        let inputs = inputs(&dir, &[air]);
        assert_ok(
            &verify_constraints(key(), &inputs, false, Some(tamper)),
            &format!("verify-constraints with {tamper}"),
        );
        assert_ok(&prove(key(), &inputs, &dir.path(air), false, false, Some(tamper)), &format!("prove with {tamper}"));
    }
}

// ---------------------------------------------------------------- Stage 2: GPU prover (Task 5)

#[test]
fn stage2_gpu_verify_constraints_passes() {
    assert_ok(&verify_constraints(gpu_key(), &all_inputs(), true, None), "verify-constraints --gpu");
}

#[test]
fn stage2_gpu_proves_each_air() {
    let dir = TestDir::new("stage2_prove");
    for air in AIRS {
        let out = prove(gpu_key(), &inputs(&dir, &[air]), &dir.path(air), true, false, None);
        assert_ok(&out, &format!("prove --gpu of {air}"));
    }
}

/// Every AIR in one non-aggregated run.
#[test]
fn stage2_gpu_proves_all_airs_in_one_run() {
    let dir = TestDir::new("stage2_prove_all");
    assert_ok(&prove(gpu_key(), &all_inputs(), &dir.path("all"), true, false, None), "prove --gpu of every AIR");
}

#[test]
fn stage2_gpu_wrap_row_excluded() {
    let dir = TestDir::new("stage2_wrap");
    let inputs = inputs(&dir, &["DomNotLast"]);
    assert_ok(&verify_constraints(gpu_key(), &inputs, true, None), "verify-constraints --gpu of DomNotLast");
    assert_ok(&prove(gpu_key(), &inputs, &dir.path("DomNotLast"), true, false, None), "prove --gpu of DomNotLast");
}

/// `src` copied to `dst` without its `.exps.so` libraries, so the GPU evaluates Q on the interpreter.
fn copy_without_exps(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let path = entry.unwrap().path();
        let to = dst.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_without_exps(&path, &to);
        } else if !path.to_string_lossy().ends_with(".exps.so") {
            fs::copy(&path, &to).unwrap();
        }
    }
}

/// The interpreter's domain zerofiers (the fallback when a Q has no generated kernel): every AIR proves,
/// and a broken domain constraint is still caught.
#[test]
fn stage2_gpu_interpreter_proves_and_catches_a_tamper() {
    let dir = TestDir::new("stage2_interpreter");
    let key = dir.path("provingKey");
    copy_without_exps(gpu_key(), &key);
    assert!(find_file(&key, "DomFirst.starkinfo.json").is_some() && find_file(&key, "DomFirst.exps.so").is_none());
    assert_ok(&prove(&key, &all_inputs(), &dir.path("all"), true, false, None), "prove --gpu on the interpreter");
    let (tamper, _, _) = TAMPER_INSIDE[5];
    let air = tamper.split(':').next().unwrap();
    let out = prove(&key, &inputs(&dir, &[air]), &dir.path(air), true, false, Some(tamper));
    assert_rejected(&out, &format!("prove --gpu on the interpreter with {tamper}"), PROOFS_REJECTED);
}

/// CPU and GPU basic proofs of the same witness are the same bytes (as on fibonacci-square).
#[test]
fn stage2_gpu_proofs_are_byte_identical_to_cpu_proofs() {
    let dir = TestDir::new("stage2_bytes");
    for air in AIRS {
        let inputs = inputs(&dir, &[air]);
        let (cpu, gpu) = (dir.path(&format!("{air}_cpu")), dir.path(&format!("{air}_gpu")));
        assert_ok(&prove_dumping_proofs(gpu_key(), &inputs, &cpu, false), &format!("prove of {air}"));
        assert_ok(&prove_dumping_proofs(gpu_key(), &inputs, &gpu, true), &format!("prove --gpu of {air}"));
        let (cpu, gpu) = (dumped_proofs(&cpu), dumped_proofs(&gpu));
        assert!(!cpu.is_empty(), "{air}: no basic proof dumped");
        assert_eq!(cpu.iter().map(|p| &p.0).collect::<Vec<_>>(), gpu.iter().map(|p| &p.0).collect::<Vec<_>>(), "{air}");
        for ((name, c), (_, g)) in cpu.iter().zip(&gpu) {
            assert!(c == g, "{air}: GPU proof {name} differs from the CPU proof");
        }
    }
}

#[test]
fn stage2_gpu_tamper_inside_a_domain_is_caught() {
    let dir = TestDir::new("stage2_inside");
    for (tamper, constraint, rows) in TAMPER_INSIDE {
        let air = tamper.split(':').next().unwrap();
        let inputs = inputs(&dir, &[air]);
        let out = verify_constraints(gpu_key(), &inputs, true, Some(tamper));
        assert_rejected(&out, &format!("verify-constraints --gpu with {tamper}"), CONSTRAINTS_REJECTED);
        assert_names_rows(tamper, &output(&out), constraint, rows);
        let out = prove(gpu_key(), &inputs, &dir.path(air), true, false, Some(tamper));
        assert_rejected(&out, &format!("prove --gpu with {tamper}"), PROOFS_REJECTED);
    }
}

#[test]
fn stage2_gpu_tamper_outside_every_domain_still_verifies() {
    let dir = TestDir::new("stage2_outside");
    for tamper in TAMPER_OUTSIDE {
        let air = tamper.split(':').next().unwrap();
        let inputs = inputs(&dir, &[air]);
        let out = verify_constraints(gpu_key(), &inputs, true, Some(tamper));
        assert_ok(&out, &format!("verify-constraints --gpu with {tamper}"));
        assert_ok(
            &prove(gpu_key(), &inputs, &dir.path(air), true, false, Some(tamper)),
            &format!("prove --gpu with {tamper}"),
        );
    }
}

// ---------------------------------------------------------------- Stage 3: aggregation (Tasks 6-7)

/// A binary crate around the `vadcop_final.verifier.rs` that `setup -r` emitted:
/// `harness <verkey.bin> <proof.bin>` exits 0 iff the native Rust verifier accepts the proof.
fn native_final_verifier() -> &'static Path {
    static HARNESS: OnceLock<PathBuf> = OnceLock::new();
    HARNESS.get_or_init(|| {
        let root = repo_root();
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("domains_e2e_native_final");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src")).unwrap();
        let emitted = recursive_key().join("build/vadcop_final/vadcop_final.verifier.rs");
        fs::copy(&emitted, dir.join("src/verifier.rs")).unwrap();
        fs::write(
            dir.join("Cargo.toml"),
            format!(
                "[package]\nname = \"domains_final_harness\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n\n\
                 [dependencies]\nproofman-verifier = {{ path = \"{}\" }}\nproofman-fields = {{ path = \"{}\" }}\n",
                root.join("verifier").display(),
                root.join("fields").display()
            ),
        )
        .unwrap();
        // The workspace's lock, so the offline build resolves to what is already downloaded.
        fs::copy(root.join("Cargo.lock"), dir.join("Cargo.lock")).unwrap();
        fs::write(
            dir.join("src/main.rs"),
            r#"extern crate alloc;
pub use proofman_verifier::*;
#[allow(dead_code)]
mod verifier;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vk: Vec<u64> =
        std::fs::read(&args[1]).unwrap().chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    let proof = VadcopFinalProof::load(&args[2]).unwrap();
    assert!(!proof.compressed, "the harness holds the uncompressed vadcop_final verifier");
    let ok = verifier::verify(&proof, &vk);
    println!("native verify: {ok}");
    std::process::exit(if ok { 0 } else { 1 });
}
"#,
        )
        .unwrap();
        let target = Path::new(env!("CARGO_TARGET_TMPDIR")).join("domains_e2e_native_final_target");
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        let out = Command::new(cargo)
            .args(["build", "--release", "--offline", "--manifest-path"])
            .arg(dir.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", &target)
            .output()
            .unwrap();
        assert!(out.status.success(), "the emitted vadcop_final verifier does not build:\n{}", output(&out));
        target.join("release/domains_final_harness")
    })
}

fn native_verify_final(proof: &Path) -> Output {
    let verkey = recursive_key().join("build/vadcop_final/vadcop_final.verkey.bin");
    Command::new(native_final_verifier()).arg(verkey).arg(proof).output().unwrap()
}

/// `prove --aggregation --verify-proofs` of every AIR, then `verify-stark` (the C++ verifier, through FFI)
/// and the emitted native Rust verifier on its vadcop_final proof; a flipped proof word fails both.
fn aggregates_and_verifies(gpu: bool) {
    let key = recursive_key();
    if gpu {
        recursive_gpu_key();
    }
    let dir = TestDir::new(if gpu { "stage3_gpu" } else { "stage3_cpu" });
    assert_ok(&prove(key, &all_inputs(), &dir.0, gpu, true, None), "prove --aggregation");
    let proof = dir.path("vadcop_final_proof.bin");
    let verkey = key.join("build/vadcop_final/vadcop_final.verkey.bin");
    let verify_stark = |proof: &Path| {
        cli(&["verify-stark", "--proof", proof.to_str().unwrap(), "--verkey", verkey.to_str().unwrap()], None)
    };
    assert_ok(&verify_stark(&proof), "verify-stark");
    assert_ok(&native_verify_final(&proof), "the native vadcop_final verifier");

    let mut bytes = fs::read(&proof).unwrap();
    let mid = bytes.len() / 16 * 8;
    bytes[mid] ^= 1;
    let bad = dir.path("vadcop_final_proof_flipped.bin");
    fs::write(&bad, bytes).unwrap();
    assert!(!verify_stark(&bad).status.success(), "verify-stark accepts a flipped proof word");
    assert_eq!(native_verify_final(&bad).status.code(), Some(1), "the native verifier accepts a flipped proof word");
}

/// The stage 3 key with its `.exps.so`, generated once per test process.
fn recursive_gpu_key() -> &'static Path {
    static GPU_KEY: OnceLock<PathBuf> = OnceLock::new();
    GPU_KEY.get_or_init(|| {
        let key = recursive_key();
        gen_exps(key);
        key.to_path_buf()
    })
}

#[test]
fn stage3_cpu_aggregated_proof_verifies() {
    aggregates_and_verifies(false);
}

#[test]
fn stage3_gpu_aggregated_proof_verifies() {
    aggregates_and_verifies(true);
}

/// Aggregation skips the basic verifier, so a basic proof of a tampered witness reaches the recursive1
/// circuit: its domain zerofiers must reject one inside a domain and accept one outside every domain.
#[test]
fn stage3_gpu_recursive1_enforces_domains() {
    let key = recursive_gpu_key();
    for (tamper, ..) in TAMPER_INSIDE {
        let dir = TestDir::new(&format!("stage3_in_{}", tamper.replace(':', "_")));
        let text = output(&prove(key, &all_inputs(), &dir.0, true, true, Some(tamper)));
        assert!(text.contains("Failed assert in template/function VerifyEvaluations"), "{tamper}:\n{text}");
        assert!(text.contains("of type Recursive1"), "{tamper}: not rejected by recursive1:\n{text}");
    }
    for tamper in TAMPER_OUTSIDE {
        let dir = TestDir::new(&format!("stage3_out_{}", tamper.replace(':', "_")));
        assert_ok(&prove(key, &all_inputs(), &dir.0, true, true, Some(tamper)), tamper);
    }
}
