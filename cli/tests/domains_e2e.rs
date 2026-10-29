//! The domains example end to end: `cargo test --release -p proofman-cli --test domains_e2e [-- --include-ignored]`.

#[path = "../../pilfflonk/tests/data/witness_libraries.rs"]
mod witness_libraries;
#[path = "common/domains.rs"]
mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

use common::{find_file, gpu_lock, gpu_ready, output, repo_root, scratch, TestDir};
use serde_json::{json, Value};

const AIRS: [&str; 9] = [
    "DomFirst",
    "DomFirstTwin",
    "DomLast",
    "DomNotFirst",
    "DomNotLast",
    "DomOdd",
    "DomLimits16",
    "DomNotQuarter",
    "DomMixed",
];

/// The std range-check table AIR that `DomMixed`'s `range_check` brings in: the one AIR with fixed columns.
const RANGE_TABLE: &str = "SpecifiedRanges";

/// `(cycleBits, offsets, complement)` of a normalized domain boundary (N = 2^10).
type Shape = (u64, &'static [u64], bool);

/// Each AIR's domain boundaries after `everyRow`, as setup normalizes and deduplicates them (std's gsum adds none).
fn expected_domains(air: &str) -> Vec<Shape> {
    match air {
        "DomFirst" | "DomFirstTwin" => vec![(10, &[0], false)],
        "DomLast" => vec![(10, &[1023], false)],
        "DomNotFirst" => vec![(10, &[0], true)],
        "DomNotLast" => vec![(10, &[1023], true)],
        "DomOdd" => vec![(1, &[1], false)],
        "DomLimits16" => vec![(4, &[0, 15], false)],
        "DomNotQuarter" => vec![(2, &[0], true)],
        // EVEN = [1,0] and EVEN4 = [1,0,1,0] are one boundary.
        "DomMixed" => vec![(10, &[0], false), (10, &[0], true), (1, &[0], false)],
        RANGE_TABLE => vec![],
        _ => unreachable!("{air}"),
    }
}

/// `(qDeg, im pols)` of each AIR at blowup 1; every AIR's gsum adds an im pol unless qDeg is 2.
fn expected_q_deg_and_im_pols(air: &str) -> (u64, usize) {
    match air {
        "DomFirst" | "DomFirstTwin" | "DomLast" | "DomNotLast" | "DomOdd" | "DomNotQuarter" | "DomNotFirst" => (1, 1),
        // its gsum, (gsum - 'gsum) * (table + ɣ) - mul, is degree 2
        RANGE_TABLE => (1, 0),
        // degree 3 on 128 rows is 4, past qDeg 2: the constraint itself is the im pol
        "DomLimits16" => (2, 1),
        // x^3 on FIRST is degree 4 (an im pol); on !FIRST it stays 3
        "DomMixed" => (2, 1),
        _ => unreachable!("{air}"),
    }
}

fn example() -> PathBuf {
    repo_root().join("examples/domains")
}

fn setup(name: &str, recursive: bool) -> PathBuf {
    common::setup(&example().join("pil/build.pilout"), &scratch(name).join("build"), recursive)
}

/// The stage 0-2 key, set up once per test process.
fn key() -> &'static Path {
    static KEY: OnceLock<PathBuf> = OnceLock::new();
    KEY.get_or_init(|| setup("domains_e2e_key", false))
}

/// The stage 3 key (`setup -r`), set up once per test process.
fn recursive_key() -> &'static Path {
    static KEY: OnceLock<PathBuf> = OnceLock::new();
    KEY.get_or_init(|| setup("domains_e2e_recursive_key", true))
}

/// `key` with its `.exps.so` (the GPU's Q kernels).
fn with_exps(key: &'static Path) -> &'static Path {
    let arch = common::cuda_arch().expect("gpu_ready checked it");
    assert_ok(&cli(&["gen-exps", "--proving-key", key.to_str().unwrap(), "--arch", &arch], &[]), "gen-exps");
    key
}

fn gpu_key() -> &'static Path {
    static KEY: OnceLock<&Path> = OnceLock::new();
    KEY.get_or_init(|| with_exps(key()))
}

fn recursive_gpu_key() -> &'static Path {
    static KEY: OnceLock<&Path> = OnceLock::new();
    KEY.get_or_init(|| with_exps(recursive_key()))
}

fn json_of(key: &Path, file: &str) -> Value {
    let path = find_file(key, file).unwrap_or_else(|| panic!("no {file}"));
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn witness_lib() -> PathBuf {
    witness_libraries::built_library("domains")
}

/// `proofman-cli <args>` with `envs`; GPU commands one at a time.
fn cli(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let _gpu = (args.contains(&"--gpu") || args.first() == Some(&"gen-exps")).then(gpu_lock);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_proofman-cli"));
    cmd.args(args).env("NO_COLOR", "1").env_remove("DOMAINS_TAMPER").envs(envs.iter().copied());
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

/// Public inputs that make the witness library emit only `airs`.
fn inputs(dir: &TestDir, airs: &[&str]) -> PathBuf {
    let path = dir.path(&format!("inputs_{}.json", airs.join("_")));
    fs::write(&path, json!({ "airs": airs }).to_string()).unwrap();
    path
}

fn all_inputs() -> PathBuf {
    example().join("src/inputs.json")
}

fn tamper_env(tamper: Option<&str>) -> Vec<(&str, &str)> {
    tamper.map(|t| ("DOMAINS_TAMPER", t)).into_iter().collect()
}

fn verify_constraints(key: &Path, inputs: &Path, gpu: bool, tamper: Option<&str>) -> Output {
    let (lib, key, inputs) = (witness_lib(), key.display().to_string(), inputs.display().to_string());
    let mut args = vec!["verify-constraints", "--witness-lib", lib.to_str().unwrap(), "--proving-key", &key];
    args.extend(["--public-inputs", &inputs]);
    if gpu {
        args.push("--gpu");
    }
    cli(&args, &tamper_env(tamper))
}

/// `prove --verify-proofs [--gpu] [--aggregation]`, writing to `out_dir`.
fn prove(key: &Path, inputs: &Path, out_dir: &Path, flags: &[&str], envs: &[(&str, &str)]) -> Output {
    let (lib, key, inputs, out) =
        (witness_lib(), key.display().to_string(), inputs.display().to_string(), out_dir.display().to_string());
    let mut args = vec!["prove", "--witness-lib", lib.to_str().unwrap(), "--proving-key", &key];
    args.extend(["--public-inputs", &inputs, "--output-dir", &out, "--verify-proofs"]);
    args.extend(flags);
    cli(&args, envs)
}

fn gpu_flag(gpu: bool) -> &'static [&'static str] {
    if gpu {
        &["--gpu"]
    } else {
        &[]
    }
}

/// The basic proofs a `PIL2_DUMP_ZKIN=basic` run wrote into `dump`, by file name.
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

/// `(tamper, the broken constraint as verify-constraints prints it, its failing rows)`.
const TAMPER_INSIDE: [(&str, &str, &[u64]); 12] = [
    ("DomFirst:a:0:5", "a-1 == 0", &[0]),
    ("DomLast:b:1023:5", "b+0 == 0", &[1023]),
    // a[500] is read by `(a - 'a - 1)^2` on rows 500 and 501.
    ("DomNotFirst:a:500:0", "((a-'a)-1)*((a-'a)-1) == 0", &[500, 501]),
    ("DomNotQuarter:u:1:5", "u-7 == 0", &[1]),
    ("DomOdd:t:1:5", "t+0 == 0", &[1]),
    ("DomLimits16:s:15:5", "(s*s)*(1-s) == 0", &[15]),
    // later cycles of the periodic domains
    ("DomOdd:t:1023:5", "t+0 == 0", &[1023]),
    ("DomLimits16:s:1008:5", "(s*s)*(1-s) == 0", &[1008]),
    ("DomNotQuarter:u:1021:5", "u-7 == 0", &[1021]),
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
    let failed: Vec<&str> = lines.take(rows.len()).collect();
    for row in rows {
        let named = failed.iter().any(|l| l.contains(&format!("Failed at row {row} ")));
        assert!(named, "{tamper}: row {row} of `{constraint}` not named:\n{text}");
    }
}

// ---------------------------------------------------------------- Stage 0: setup

#[test]
fn stage0_setup_emits_each_domain_boundary_and_its_q_degree() {
    let key = key();
    for air in AIRS.into_iter().chain([RANGE_TABLE]) {
        let info = json_of(key, &format!("{air}.starkinfo.json"));
        let boundaries = info["boundaries"].as_array().unwrap_or_else(|| panic!("{air}: no boundaries"));
        assert_eq!(boundaries[0], json!({ "name": "everyRow" }), "{air}: boundary 0");
        let mut want: Vec<Value> = expected_domains(air)
            .into_iter()
            .map(|(cycle_bits, offsets, complement)| {
                json!({ "name": "domain", "cycleBits": cycle_bits, "offsets": offsets, "complement": complement })
            })
            .collect();
        let mut got = boundaries[1..].to_vec();
        // boundary order is setup's; the set is the contract
        got.sort_by_key(|b| b.to_string());
        want.sort_by_key(|b| b.to_string());
        assert_eq!(got, want, "{air}: domain boundaries");

        // DomMixed's EVEN4 constraint, in block form, is a domain constraint too
        let constraints = json_of(key, &format!("{air}.expressionsinfo.json"))["constraints"].clone();
        let n_domain = constraints.as_array().unwrap().iter().filter(|c| c["boundary"] == "domain").count();
        let own = match air {
            "DomMixed" => 4,
            RANGE_TABLE => 0,
            _ => 1,
        };
        assert_eq!(n_domain, own, "{air}: domain constraints");

        // the self-permutation gives every AIR stage 2
        assert_eq!(info["nStages"].as_u64(), Some(2), "{air}: nStages");
        let zero_fixed = air != RANGE_TABLE;
        assert_eq!(info["nConstants"].as_u64() == Some(0), zero_fixed, "{air}: nConstants");
        let verkey = find_file(key, &format!("{air}.verkey.json")).is_some();
        assert_eq!(verkey, !zero_fixed, "{air}: .verkey.json");

        let im_pols =
            info["cmPolsMap"].as_array().unwrap().iter().filter(|p| p["imPol"].as_bool() == Some(true)).count();
        let q_deg = info["qDeg"].as_u64().unwrap();
        assert_eq!((q_deg, im_pols), expected_q_deg_and_im_pols(air), "{air}: (qDeg, im pols)");
    }
}

#[test]
fn stage0_zero_fixed_airs_have_no_constants_group() {
    let key = key();
    for air in AIRS {
        let info = json_of(key, &format!("{air}.starkinfo.json"));
        assert_eq!(info["nConstants"].as_u64(), Some(0), "{air}: nConstants");
        assert_eq!(info["mapSectionsN"]["const"].as_u64(), Some(0), "{air}: mapSectionsN.const");
        assert_eq!(info["nStages"].as_u64(), Some(2), "{air}: nStages");
        assert!(info["circuitType"].is_u64(), "{air}: circuitType");
        let ev_map = info["evMap"].as_array().unwrap();
        assert!(!ev_map.iter().any(|e| e["type"] == "const"), "{air}: evMap has a const entry");
        for ext in ["const", "verkey.json", "verkey.bin", "consttree"] {
            assert!(find_file(key, &format!("{air}.{ext}")).is_none(), "{air}: has a .{ext}");
        }
    }
}

#[test]
fn stage0_control_air_keeps_constants() {
    let key = key();
    let info = json_of(key, &format!("{RANGE_TABLE}.starkinfo.json"));
    assert!(info["nConstants"].as_u64().unwrap() >= 1, "{RANGE_TABLE}: nConstants");
    for ext in ["verkey.json", "verkey.bin"] {
        assert!(find_file(key, &format!("{RANGE_TABLE}.{ext}")).is_some(), "{RANGE_TABLE}: no .{ext}");
    }
}

#[test]
fn stage0_nonempty_fixed_file_for_zero_fixed_air_is_rejected() {
    let dir = TestDir::new("domains_e2e_fixed_dir");
    let fixed = dir.path("fixed");
    fs::create_dir_all(&fixed).unwrap();
    let pilout = example().join("pil/build.pilout");
    let run = || common::try_setup_with_fixed_dir(&pilout, &dir.path("build"), false, Some(&fixed));

    fs::write(fixed.join("DomFirst.fixed"), [0u8; 16]).unwrap();
    let err = run().expect_err("setup accepted a non-empty .fixed for a zero-fixed air");
    assert!(err.contains("DomFirst") && err.contains("no fixed columns"), "{err}");

    // the empty file compile-pil --fixed-to-file writes is ignored
    fs::write(fixed.join("DomFirst.fixed"), []).unwrap();
    let key = run().unwrap();
    assert!(find_file(&key, "DomFirst.starkinfo.json").is_some());
    assert!(find_file(&key, "DomFirst.const").is_none(), "DomFirst: an empty .fixed became a .const");
}

fn starkinfos(dir: &Path, out: &mut Vec<PathBuf>) {
    for path in fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()) {
        if path.is_dir() {
            starkinfos(&path, out);
        } else if path.to_string_lossy().ends_with(".starkinfo.json") {
            out.push(path);
        }
    }
}

/// The C++ flat proof size is the Rust layout's plus the air(group) values, with and without constants.
#[test]
fn stage0_proof_size_parity_cpp_vs_rust() {
    use pil2_stark_setup::types::stark_info::StarkInfo;
    use proofman_starks_lib_c::{get_proof_size_c, stark_info_free_c, stark_info_new_c};

    let dir = TestDir::new("domains_e2e_parity");
    let native = common::setup(&repo_root().join("examples/domain-native/pil/build.pilout"), &dir.path("dn"), false);
    let mut files = Vec::new();
    starkinfos(key(), &mut files);
    starkinfos(&native, &mut files);
    let names: Vec<String> = files.iter().map(|f| f.file_name().unwrap().to_string_lossy().into_owned()).collect();
    for air in AIRS.into_iter().chain([RANGE_TABLE, "DomNative"]) {
        assert!(names.contains(&format!("{air}.starkinfo.json")), "{air}: no starkinfo");
    }
    for file in &files {
        let json: Value = serde_json::from_str(&fs::read_to_string(file).unwrap()).unwrap();
        let layout = StarkInfo::from_json(&json).unwrap().layout.unwrap().unwrap();
        let values =
            json["airgroupValuesMap"].as_array().unwrap().len() + json["airValuesMap"].as_array().unwrap().len();
        let si = stark_info_new_c(file.to_str().unwrap(), false, false, false, true, false);
        let cpp = get_proof_size_c(si);
        stark_info_free_c(si);
        assert_eq!(cpp as usize, 3 * values + layout.flat_proof_words(), "{}", file.display());
    }
}

// ---------------------------------------------------------------- Stages 1 (CPU) and 2 (GPU): no aggregation

fn prover_key(gpu: bool) -> &'static Path {
    if gpu {
        gpu_key()
    } else {
        key()
    }
}

fn verify_constraints_passes(gpu: bool) {
    let out = verify_constraints(prover_key(gpu), &all_inputs(), gpu, None);
    assert_ok(&out, "verify-constraints");
    let text = output(&out);
    assert!(text.contains("All global constraints were successfully verified"), "{text}");
    for air in AIRS.into_iter().chain([RANGE_TABLE]) {
        assert!(text.contains(&format!(" of {air} were verified")), "{air}: constraints not verified:\n{text}");
    }
}

fn proves_all_airs_in_one_run(gpu: bool) {
    let dir = TestDir::new(if gpu { "domains_e2e_gpu_prove_all" } else { "domains_e2e_prove_all" });
    let out = prove(prover_key(gpu), &all_inputs(), &dir.path("all"), gpu_flag(gpu), &[]);
    assert_ok(&out, "prove of every AIR");
    assert!(output(&out).contains("All proofs were successfully verified"), "{}", output(&out));
}

fn tamper_inside_a_domain_is_caught(gpu: bool) {
    let dir = TestDir::new(if gpu { "domains_e2e_gpu_inside" } else { "domains_e2e_inside" });
    for (tamper, constraint, rows) in TAMPER_INSIDE {
        let air = tamper.split(':').next().unwrap();
        let inputs = inputs(&dir, &[air]);
        let out = verify_constraints(prover_key(gpu), &inputs, gpu, Some(tamper));
        assert_rejected(&out, &format!("verify-constraints with {tamper}"), CONSTRAINTS_REJECTED);
        assert_names_rows(tamper, &output(&out), constraint, rows);
        let out = prove(prover_key(gpu), &inputs, &dir.path(air), gpu_flag(gpu), &tamper_env(Some(tamper)));
        assert_rejected(&out, &format!("prove with {tamper}"), PROOFS_REJECTED);
    }
}

fn tamper_outside_every_domain_still_verifies(gpu: bool) {
    let dir = TestDir::new(if gpu { "domains_e2e_gpu_outside" } else { "domains_e2e_outside" });
    for tamper in TAMPER_OUTSIDE {
        let air = tamper.split(':').next().unwrap();
        let inputs = inputs(&dir, &[air]);
        let what = format!("verify-constraints with {tamper}");
        assert_ok(&verify_constraints(prover_key(gpu), &inputs, gpu, Some(tamper)), &what);
        let out = prove(prover_key(gpu), &inputs, &dir.path(air), gpu_flag(gpu), &tamper_env(Some(tamper)));
        assert_ok(&out, &format!("prove with {tamper}"));
    }
}

#[test]
fn stage1_cpu_verify_constraints_passes() {
    verify_constraints_passes(false);
}

/// `src` copied to `dst`.
fn copy_dir(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for path in fs::read_dir(src).unwrap().map(|e| e.unwrap().path()) {
        let to = dst.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_dir(&path, &to);
        } else {
            fs::copy(&path, &to).unwrap();
        }
    }
}

/// A copy of the stage 0-2 key in `dir`, and the directory holding `air`'s files in it.
fn key_copy(dir: &TestDir, air: &str) -> (PathBuf, PathBuf) {
    let copy = dir.path("provingKey");
    copy_dir(key(), &copy);
    let files = find_file(&copy, &format!("{air}.starkinfo.json")).unwrap().parent().unwrap().to_path_buf();
    (copy, files)
}

#[test]
fn stage1_stale_verkey_for_zero_fixed_air_is_rejected() {
    let dir = TestDir::new("domains_e2e_stale_verkey");
    let (key, files) = key_copy(&dir, "DomFirst");
    fs::write(files.join("DomFirst.verkey.json"), "[1,2,3,4]").unwrap();
    let out = verify_constraints(&key, &all_inputs(), false, None);
    assert_rejected(&out, "verify-constraints with a stale verkey", "stale verkey for DomFirst");
}

#[test]
fn stage1_short_verkey_for_air_with_constants_is_rejected() {
    let dir = TestDir::new("domains_e2e_short_verkey");
    let (key, files) = key_copy(&dir, RANGE_TABLE);
    fs::write(files.join(format!("{RANGE_TABLE}.verkey.json")), "[1,2,3]").unwrap();
    let out = verify_constraints(&key, &all_inputs(), false, None);
    assert_rejected(&out, "verify-constraints with a 3-word verkey", "3 words, expected 4");
    assert!(output(&out).contains(&format!("for {RANGE_TABLE}")), "{}", output(&out));
}

#[test]
fn stage1_missing_verkey_for_air_with_constants_is_rejected() {
    let dir = TestDir::new("domains_e2e_missing_verkey");
    let (key, files) = key_copy(&dir, RANGE_TABLE);
    fs::remove_file(files.join(format!("{RANGE_TABLE}.verkey.json"))).unwrap();
    let out = verify_constraints(&key, &all_inputs(), false, None);
    assert_rejected(&out, "verify-constraints without a verkey", "missing verkey");
    assert!(output(&out).contains(&format!("for {RANGE_TABLE}")), "{}", output(&out));
}

/// `prove-air --air <air>` of the CPU, saved as `[nPublics, publics…, proof…]` words.
fn prove_air(lib: &Path, key: &Path, air: &str, proof: &Path) -> Vec<u64> {
    let (lib, key, out) = (lib.display().to_string(), key.display().to_string(), proof.display().to_string());
    let args = ["prove-air", "--witness-lib", &lib, "--air", air, "--proving-key", &key, "--save-proof", &out];
    assert_ok(&cli(&args, &[]), &format!("prove-air of {air}"));
    fs::read(proof).unwrap().as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect()
}

/// Whether the C++ verifier accepts saved `prove-air` words under `air`'s files in `key`.
fn verify_saved(words: &[u64], key: &Path, air: &str) -> bool {
    use proofman_fields::{Goldilocks, PrimeField64};
    proofman_starks_lib_c::set_hash_family_c(proofman_common::hash_family::DEFAULT_HASH_ID);
    let n = words[0] as usize;
    let publics: Vec<Goldilocks> = words[1..1 + n].iter().map(|&w| Goldilocks::from_u64(w)).collect();
    let starkinfo = find_file(key, &format!("{air}.starkinfo.json")).unwrap().display().to_string();
    let stem = starkinfo.strip_suffix(".starkinfo.json").unwrap();
    proofman::verify_proof::<Goldilocks>(
        &words[1 + n..],
        starkinfo.clone(),
        format!("{stem}.verifier.bin"),
        format!("{stem}.verkey.json"),
        Some(publics),
        None,
        None,
    )
}

/// DomFirst's self-contained proof, made once per test process.
fn dom_first_saved() -> &'static [u64] {
    static WORDS: OnceLock<Vec<u64>> = OnceLock::new();
    WORDS.get_or_init(|| {
        let dir = TestDir::new("domains_e2e_prove_air");
        prove_air(&witness_lib(), key(), "DomFirst", &dir.path("p.bin"))
    })
}

#[test]
fn stage1_cpu_prove_air_self_contained_round_trip() {
    assert!(verify_saved(dom_first_saved(), key(), "DomFirst"), "DomFirst's self-contained proof does not verify");
    // the control has constants
    let dir = TestDir::new("domains_e2e_prove_air_native");
    let native = common::setup(&repo_root().join("examples/domain-native/pil/build.pilout"), &dir.path("dn"), false);
    let lib = witness_libraries::built_library("domain_native");
    let words = prove_air(&lib, &native, "DomNative", &dir.path("p.bin"));
    assert!(verify_saved(&words, &native, "DomNative"), "DomNative's self-contained proof does not verify");
}

/// `DomFirstTwin` is `DomFirst`'s PIL: only the circuitType the seed absorbs tells them apart.
#[test]
fn stage1_self_contained_proof_does_not_verify_under_twin_air() {
    let words = dom_first_saved();
    assert!(verify_saved(words, key(), "DomFirst"));
    assert!(!verify_saved(words, key(), "DomFirstTwin"), "DomFirst's proof verifies as DomFirstTwin's");
}

/// A const segment for a 0-column tree, spliced in before cm1's or appended, does not verify:
/// the verifier parses with its own const-free layout and takes exactly proofSize words.
#[test]
fn stage1_spliced_const_segment_is_rejected() {
    use pil2_stark_setup::types::stark_info::StarkInfo;
    use proofman_verifier::LowDegreeLayout;

    let words = dom_first_saved();
    let info = json_of(key(), "DomFirst.starkinfo.json");
    let layout = StarkInfo::from_json(&info).unwrap().layout.unwrap().unwrap();
    let LowDegreeLayout::Fri { n_queries, .. } = layout.ldt else { panic!("DomFirst is not FRI") };
    let mut one = layout.clone();
    one.n_constants = 1;
    // a 1-column const segment less its one value per query: the siblings and the last level
    let spliced = one.flat_proof_words() - layout.flat_proof_words() - n_queries as usize;
    let len = |k: &str| info[k].as_array().unwrap().len();
    let n_publics = info["nPublics"].as_u64().unwrap() as usize;
    let at = 1
        + n_publics
        + 3 * (len("airgroupValuesMap") + len("airValuesMap"))
        + 4 * (layout.n_stages as usize + 1)
        + 3 * layout.n_evals as usize;
    let mut bad = words.to_vec();
    bad.splice(at..at, std::iter::repeat_n(0, spliced));
    assert!(verify_saved(words, key(), "DomFirst"));
    assert!(!verify_saved(&bad, key(), "DomFirst"), "a spliced const segment verifies");
    let appended: Vec<u64> = words.iter().copied().chain(std::iter::repeat_n(0, spliced)).collect();
    assert!(!verify_saved(&appended, key(), "DomFirst"), "an appended const segment verifies");
}

#[test]
fn stage1_cpu_proves_each_air() {
    let dir = TestDir::new("domains_e2e_prove_each");
    for air in AIRS {
        assert_ok(&prove(key(), &inputs(&dir, &[air]), &dir.path(air), &[], &[]), &format!("prove of {air}"));
    }
}

#[test]
fn stage1_cpu_proves_all_airs_in_one_run() {
    proves_all_airs_in_one_run(false);
}

#[test]
fn stage1_cpu_tamper_inside_a_domain_is_caught() {
    tamper_inside_a_domain_is_caught(false);
}

#[test]
fn stage1_cpu_tamper_outside_every_domain_still_verifies() {
    tamper_outside_every_domain_still_verifies(false);
}

#[test]
fn stage2_gpu_verify_constraints_passes() {
    if gpu_ready("stage2_gpu_verify_constraints_passes") {
        verify_constraints_passes(true);
    }
}

/// What the GPU stream commit logs when it refuses an air for want of its const pols.
const NO_CONST_POLS: &str = "no accumulator or const pols";

/// `{"airs": airs, "instances": n}`.
fn inputs_n(dir: &TestDir, airs: &[&str], n: usize) -> PathBuf {
    let path = dir.path(&format!("inputs_{}_x{n}.json", airs.join("_")));
    fs::write(&path, json!({ "airs": airs, "instances": n }).to_string()).unwrap();
    path
}

/// Several instances of every AIR prove and verify on the GPU.
#[test]
fn stage2_gpu_prove_verify_proofs() {
    if !gpu_ready("stage2_gpu_prove_verify_proofs") {
        return;
    }
    let dir = TestDir::new("domains_e2e_gpu_prove_n");
    let out = prove(gpu_key(), &inputs_n(&dir, &AIRS, 3), &dir.path("out"), &["--gpu"], &[]);
    assert_ok(&out, "prove --gpu of 3 instances per AIR");
    let text = output(&out);
    assert!(text.contains("All proofs were successfully verified"), "{text}");
    assert!(!text.contains(NO_CONST_POLS), "{text}");
}

/// Zero-fixed and with-constants proofs interleave on one stream (the slot-reuse rule: test_fixed_slot_cpu.cpp).
#[test]
fn stage2_gpu_zero_fixed_after_fixed_on_one_stream() {
    if !gpu_ready("stage2_gpu_zero_fixed_after_fixed_on_one_stream") {
        return;
    }
    let dir = TestDir::new("domains_e2e_gpu_one_stream");
    let inputs = inputs_n(&dir, &["DomFirst", "DomMixed"], 4);
    let out = prove(gpu_key(), &inputs, &dir.path("out"), &["--gpu", "-t", "1"], &[]);
    assert_ok(&out, "prove --gpu -t 1");
    let text = output(&out);
    assert!(text.contains("All proofs were successfully verified"), "{text}");
    assert!(!text.contains(NO_CONST_POLS), "{text}");
}

#[test]
fn stage2_gpu_proves_all_airs_in_one_run() {
    if gpu_ready("stage2_gpu_proves_all_airs_in_one_run") {
        proves_all_airs_in_one_run(true);
    }
}

#[test]
fn stage2_gpu_tamper_inside_a_domain_is_caught() {
    if gpu_ready("stage2_gpu_tamper_inside_a_domain_is_caught") {
        tamper_inside_a_domain_is_caught(true);
    }
}

#[test]
fn stage2_gpu_tamper_outside_every_domain_still_verifies() {
    if gpu_ready("stage2_gpu_tamper_outside_every_domain_still_verifies") {
        tamper_outside_every_domain_still_verifies(true);
    }
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

/// The interpreter's domain zerofiers (Q without a generated kernel) prove, and still catch a tamper.
#[test]
fn stage2_gpu_interpreter_proves_and_catches_a_tamper() {
    if !gpu_ready("stage2_gpu_interpreter_proves_and_catches_a_tamper") {
        return;
    }
    let dir = TestDir::new("domains_e2e_interpreter");
    let key = dir.path("provingKey");
    copy_without_exps(gpu_key(), &key);
    assert!(find_file(&key, "DomFirst.starkinfo.json").is_some() && find_file(&key, "DomFirst.exps.so").is_none());
    assert_ok(&prove(&key, &all_inputs(), &dir.path("all"), &["--gpu"], &[]), "prove --gpu on the interpreter");
    let (tamper, _, _) = TAMPER_INSIDE[5];
    let air = tamper.split(':').next().unwrap();
    let out = prove(&key, &inputs(&dir, &[air]), &dir.path(air), &["--gpu"], &tamper_env(Some(tamper)));
    assert_rejected(&out, &format!("prove --gpu on the interpreter with {tamper}"), PROOFS_REJECTED);
}

/// CPU and GPU basic proofs of the same witness are the same bytes.
#[test]
fn stage2_gpu_proofs_are_byte_identical_to_cpu_proofs() {
    if !gpu_ready("stage2_gpu_proofs_are_byte_identical_to_cpu_proofs") {
        return;
    }
    let dir = TestDir::new("domains_e2e_bytes");
    for air in AIRS {
        let inputs = inputs(&dir, &[air]);
        let mut dumps = Vec::new();
        for gpu in [false, true] {
            let dump = dir.path(&format!("{air}_{gpu}"));
            fs::create_dir_all(&dump).unwrap();
            let envs = [("PIL2_DUMP_ZKIN", "basic"), ("TMPDIR", dump.to_str().unwrap())];
            assert_ok(&prove(gpu_key(), &inputs, &dump, gpu_flag(gpu), &envs), &format!("prove (gpu {gpu}) of {air}"));
            dumps.push(dumped_proofs(&dump));
        }
        let (cpu, gpu) = (&dumps[0], &dumps[1]);
        assert!(!cpu.is_empty(), "{air}: no basic proof dumped");
        assert_eq!(cpu.iter().map(|p| &p.0).collect::<Vec<_>>(), gpu.iter().map(|p| &p.0).collect::<Vec<_>>(), "{air}");
        for ((name, c), (_, g)) in cpu.iter().zip(gpu) {
            assert!(c == g, "{air}: GPU proof {name} differs from the CPU proof");
        }
    }
}

// ---------------------------------------------------------------- Stage 3: aggregation

/// The native Rust verifier `setup -r` emitted for vadcop_final: `harness <verkey.bin> <proof.bin>`.
fn native_final_verifier() -> &'static Path {
    static HARNESS: OnceLock<PathBuf> = OnceLock::new();
    HARNESS.get_or_init(|| {
        let dir = scratch("domains_e2e_native_final");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src")).unwrap();
        let emitted = recursive_key().join("build/vadcop_final/vadcop_final.verifier.rs");
        fs::copy(&emitted, dir.join("src/verifier.rs")).unwrap();
        common::build_harness(&dir, "domains_final_harness", FINAL_HARNESS_MAIN, "")
    })
}

const FINAL_HARNESS_MAIN: &str = r#"extern crate alloc;
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
"#;

fn native_verify_final(proof: &Path) -> Output {
    let verkey = recursive_key().join("build/vadcop_final/vadcop_final.verkey.bin");
    Command::new(native_final_verifier()).arg(verkey).arg(proof).output().unwrap()
}

/// `prove --aggregation`, then `verify-stark` and the native verifier accept the proof and reject a flipped word.
fn aggregates_and_verifies(gpu: bool) {
    let key = if gpu { recursive_gpu_key() } else { recursive_key() };
    let dir = TestDir::new(if gpu { "domains_e2e_stage3_gpu" } else { "domains_e2e_stage3_cpu" });
    let flags: &[&str] = if gpu { &["--gpu", "--aggregation"] } else { &["--aggregation"] };
    assert_ok(&prove(key, &all_inputs(), &dir.0, flags, &[]), "prove --aggregation");
    let proof = dir.path("vadcop_final_proof.bin");
    let verkey = key.join("build/vadcop_final/vadcop_final.verkey.bin");
    let verify_stark = |proof: &Path| {
        cli(&["verify-stark", "--proof", proof.to_str().unwrap(), "--verkey", verkey.to_str().unwrap()], &[])
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

#[test]
#[ignore = "setup -r takes minutes: cargo test --release -p proofman-cli --test domains_e2e -- --ignored"]
fn stage3_cpu_aggregated_proof_verifies() {
    aggregates_and_verifies(false);
}

#[test]
#[ignore = "setup -r takes minutes, and a GPU: cargo test --release -p proofman-cli --test domains_e2e -- --ignored"]
fn stage3_gpu_aggregated_proof_verifies() {
    if gpu_ready("stage3_gpu_aggregated_proof_verifies") {
        aggregates_and_verifies(true);
    }
}

/// Aggregation skips the basic verifier: recursive1 must reject a tamper inside a domain, accept one outside.
#[test]
#[ignore = "setup -r takes minutes, and a GPU: cargo test --release -p proofman-cli --test domains_e2e -- --ignored"]
fn stage3_gpu_recursive1_enforces_domains() {
    if !gpu_ready("stage3_gpu_recursive1_enforces_domains") {
        return;
    }
    let key = recursive_gpu_key();
    let flags = ["--gpu", "--aggregation"];
    for (tamper, ..) in TAMPER_INSIDE {
        let dir = TestDir::new(&format!("domains_e2e_stage3_in_{}", tamper.replace(':', "_")));
        let text = output(&prove(key, &all_inputs(), &dir.0, &flags, &tamper_env(Some(tamper))));
        assert!(text.contains("Failed assert in template/function VerifyEvaluations"), "{tamper}:\n{text}");
        assert!(text.contains("of type Recursive1"), "{tamper}: not rejected by recursive1:\n{text}");
    }
    for tamper in TAMPER_OUTSIDE {
        let dir = TestDir::new(&format!("domains_e2e_stage3_out_{}", tamper.replace(':', "_")));
        assert_ok(&prove(key, &all_inputs(), &dir.0, &flags, &tamper_env(Some(tamper))), tamper);
    }
}
