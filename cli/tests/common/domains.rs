//! What `domains_e2e.rs` and `domain_native_e2e.rs` share: setup, scratch dirs, the GPU gate, native harnesses.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pil2_stark_setup::commands::setup::{run_setup, SetupOptions};
use proofman_common::hash_family::{compressed_final_by_default, default_aggregation_arity, DEFAULT_HASH_ID};

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap()
}

/// What a command wrote, stdout then stderr.
pub fn output(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

pub fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
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

/// A scratch directory under the target's temporary directory, per test process, removed when dropped.
pub struct TestDir(pub PathBuf);

impl TestDir {
    pub fn new(name: &str) -> Self {
        let dir = scratch(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `<target tmp>/<name>_<pid>`; a key there outlives the process, so the ones of dead processes are removed.
pub fn scratch(name: &str) -> PathBuf {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let prefix = format!("{name}_");
    for entry in fs::read_dir(tmp).into_iter().flatten().flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        let pid = file.strip_prefix(&prefix).and_then(|p| p.parse::<u32>().ok());
        if pid.is_some_and(|pid| !Path::new(&format!("/proc/{pid}")).exists()) {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
    tmp.join(format!("{name}_{}", std::process::id()))
}

/// `proofman-setup setup -a <pilout> -b <build> [-r]` with the CLI's defaults; returns the proving key.
pub fn setup(pilout: &Path, build: &Path, recursive: bool) -> PathBuf {
    setup_with_fixed_dir(pilout, build, recursive, None)
}

/// `setup`, with `-u <fixed_dir>` when given.
pub fn setup_with_fixed_dir(pilout: &Path, build: &Path, recursive: bool, fixed_dir: Option<&Path>) -> PathBuf {
    let _ = fs::remove_dir_all(build);
    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let opts = SetupOptions {
        airout_path: pilout.display().to_string(),
        build_dir: build.display().to_string(),
        fixed_dir: fixed_dir.map(|d| d.display().to_string()),
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
    // the recursive setup recurses deeper than a test thread's stack; the CLI runs it on 64 MB too
    let result = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || run_setup(&opts).map_err(|e| format!("{e:#}")))
        .unwrap()
        .join()
        .unwrap();
    result.unwrap_or_else(|e| panic!("setup of {} failed: {e}", build.display()));
    build.join("provingKey")
}

/// `gen-exps`'s arch: `DOMAINS_CUDA_ARCH`, else the first GPU's compute capability (12.0 -> sm_120).
pub fn cuda_arch() -> Option<String> {
    if let Ok(arch) = std::env::var("DOMAINS_CUDA_ARCH") {
        return Some(arch);
    }
    let out = Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output().ok()?;
    let cap = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().replace('.', "");
    (out.status.success() && !cap.is_empty() && cap.chars().all(|c| c.is_ascii_digit())).then(|| format!("sm_{cap}"))
}

/// Whether the GPU tests run here (a CUDA build, a GPU, nvcc); else says why and skips, or fails with `DOMAINS_GPU=1`.
pub fn gpu_ready(test: &str) -> bool {
    let nvcc = || Command::new("nvcc").arg("--version").output().is_ok_and(|o| o.status.success());
    let why = if !proofman_pilfflonk::gpu_available() {
        "no GPU"
    } else if cuda_arch().is_none() {
        "no CUDA arch (set DOMAINS_CUDA_ARCH)"
    } else if !nvcc() {
        "no nvcc on PATH"
    } else {
        return true;
    };
    assert_ne!(std::env::var("DOMAINS_GPU").as_deref(), Ok("1"), "DOMAINS_GPU=1, and {why}");
    eprintln!("skipping {test}: {why}");
    false
}

/// Held around each GPU command, across test processes too: each one sizes its buffers from the free memory.
pub fn gpu_lock() -> fs::File {
    let file = fs::File::create(Path::new(env!("CARGO_TARGET_TMPDIR")).join("domains_gpu.lock")).unwrap();
    file.lock().unwrap();
    file
}

/// Builds `dir`, whose `src/verifier.rs` is in place, as the binary `name` around `main_rs`, offline.
pub fn build_harness(dir: &Path, name: &str, main_rs: &str, extra_deps: &str) -> PathBuf {
    let root = repo_root();
    fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n\n\
             [dependencies]\nproofman-verifier = {{ path = \"{}\" }}\nproofman-fields = {{ path = \"{}\" }}\n{extra_deps}",
            root.join("verifier").display(),
            root.join("fields").display()
        ),
    )
    .unwrap();
    // the workspace's lock, so the offline build resolves to what is already downloaded
    fs::copy(root.join("Cargo.lock"), dir.join("Cargo.lock")).unwrap();
    fs::write(dir.join("src/main.rs"), main_rs).unwrap();
    let target = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}_target"));
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = Command::new(cargo)
        .args(["build", "--release", "--offline", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", &target)
        .output()
        .unwrap();
    assert!(out.status.success(), "the emitted verifier does not build:\n{}", output(&out));
    target.join("release").join(name)
}
