//! Rebuild circom witness libraries (`.so` / `.dylib`) directly from a
//! `provingKey/` tree, without re-running the full setup pipeline.
//!
//! For each circuit that has a witness library in the proving key, this:
//!   1. Loads the persisted `starkinfo.json` / `verifierinfo.json` / `verkey.json`.
//!   2. Calls `pil2circom` and `gen_circom` to regenerate the verifier and wrapper
//!      circom sources (cheap, in-process).
//!   3. Invokes the `circom` compiler to produce the C++ witness calculator.
//!   4. Reuses [`WitnessTracker::run_witness_library_generation`] to compile the
//!      C++ into a shared library.
//!
//! The expensive setup steps (`pil_info`, `bctree`, `plonk2pil`, const-tree)
//! are skipped — the proving key already contains the artifacts they produced.
//!
//! Notes on inputs by template:
//!   - **compressor**       : air's si/vi/vk
//!   - **recursive1** (no compressor): air's si/vi/vk
//!   - **recursive1** (with compressor): compressor's si/vi/vk
//!   - **recursive2**       : recursive2's own si/vi (functionally equal to
//!     recursive1's, which is not persisted)
//!   - **vadcop_final**     : per-airgroup recursive2 si/vi + vks
//!   - **vadcop_final_compressed**: vadcop_final si/vi + verkey

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde_json::Value;

use crate::io::recurser::{gen_circom, pil2circom, GenCircomInput, GenCircomOptions, Pil2CircomOptions};
use crate::output::witness_gen::WitnessTracker;

/// External tool paths needed by every rebuild.
pub struct RebuildPaths<'a> {
    pub circom_exec: &'a str,
    pub circuits_gl_path: &'a str,
    pub recurser_circuits_path: &'a str,
    pub recurser_circuits_compressed_final_path: &'a str,
    pub circom_helpers_dir: &'a str,
}

/// One unit of rebuild work, materialised after the discovery walk so all
/// rebuilds can run in parallel without re-walking the proving key.
#[derive(Debug)]
enum RebuildTask {
    Compressor { air_root: PathBuf, ag_name: String, air_name: String },
    Recursive1 { air_root: PathBuf, ag_idx: usize, ag_name: String, air_name: String, has_compressor: bool },
    Recursive2 { r2_dir: PathBuf, ag_idx: usize, ag_name: String, airs: Vec<Value> },
    VadcopFinal { pk_root: PathBuf, air_groups: Vec<String> },
    VadcopFinalCompressed { pk_root: PathBuf },
}

impl RebuildTask {
    /// Human-readable label used in progress logs.
    fn label(&self) -> String {
        match self {
            RebuildTask::Compressor { ag_name, air_name, .. } => format!("compressor / {ag_name}.{air_name}"),
            RebuildTask::Recursive1 { ag_name, air_name, has_compressor, .. } => {
                let suffix = if *has_compressor { " (with compressor)" } else { "" };
                format!("recursive1 / {ag_name}.{air_name}{suffix}")
            }
            RebuildTask::Recursive2 { ag_name, .. } => format!("recursive2 / {ag_name}"),
            RebuildTask::VadcopFinal { .. } => "vadcop_final".to_string(),
            RebuildTask::VadcopFinalCompressed { .. } => "vadcop_final_compressed".to_string(),
        }
    }
}

/// Walk a `provingKey/` tree and rebuild every witness library found.
///
/// `proving_key` points directly at the `provingKey/` directory.  `build_dir`
/// holds the regenerated `.circom` and `.cpp` files used during the build; it
/// must outlive every spawned witness build (i.e. the caller must wait on
/// `witness_tracker.await_all()` before dropping it).
///
/// `jobs` controls how many circom compiles run concurrently.  Each circom
/// invocation is single-threaded but RAM-hungry (~10–20 GB peak for large
/// recursive2 / vadcop_final circuits), so size by available memory rather
/// than CPU count.  Default 1 = serial.
pub fn rebuild_all_witness_libs(
    proving_key: &str,
    build_dir: &str,
    paths: &RebuildPaths<'_>,
    witness_tracker: &WitnessTracker,
    jobs: usize,
) -> Result<usize> {
    let proving_key = PathBuf::from(proving_key);
    if !proving_key.is_dir() {
        bail!("provingKey directory not found at {}", proving_key.display());
    }

    let global_info_path = find_global_info(&proving_key)?;
    let global_info: Value = serde_json::from_str(&fs::read_to_string(&global_info_path)?)
        .with_context(|| format!("Failed to parse {}", global_info_path.display()))?;
    let global_name =
        global_info.get("name").and_then(|v| v.as_str()).context("globalInfo.json missing 'name' field")?.to_string();

    let global_constraints_path = global_info_path.with_file_name(
        global_info_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.replace(".globalInfo.json", ".globalConstraints.json"))
            .unwrap_or_else(|| "pilout.globalConstraints.json".to_string()),
    );
    let global_constraints: Value = if global_constraints_path.exists() {
        serde_json::from_str(&fs::read_to_string(&global_constraints_path)?)
            .with_context(|| format!("Failed to parse {}", global_constraints_path.display()))?
    } else {
        Value::Null
    };

    let pk_root = proving_key.join(&global_name);
    if !pk_root.is_dir() {
        bail!("Expected proving key root at {} (from globalInfo.name)", pk_root.display());
    }

    let air_groups: Vec<String> = global_info
        .get("air_groups")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let airs_per_group: Vec<Vec<Value>> = global_info
        .get("airs")
        .and_then(|v| v.as_array())
        .map(|outer| outer.iter().map(|inner| inner.as_array().cloned().unwrap_or_default()).collect())
        .unwrap_or_default();

    // ── Discovery: walk the proving key and collect all rebuild work upfront.
    let mut tasks: Vec<RebuildTask> = Vec::new();

    for (ag_idx, ag_name) in air_groups.iter().enumerate() {
        let airs = airs_per_group.get(ag_idx).cloned().unwrap_or_default();
        let ag_dir = pk_root.join(ag_name);
        if !ag_dir.is_dir() {
            tracing::warn!("Airgroup '{}' missing in proving key, skipping", ag_name);
            continue;
        }

        for air in airs.iter() {
            let air_name = match air.get("name").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let air_root = ag_dir.join("airs").join(&air_name);
            let compressor_dir = air_root.join("compressor");
            let recursive1_dir = air_root.join("recursive1");
            let has_compressor =
                compressor_dir.join("compressor.so").exists() || compressor_dir.join("compressor.dylib").exists();

            if has_compressor {
                tasks.push(RebuildTask::Compressor {
                    air_root: air_root.clone(),
                    ag_name: ag_name.clone(),
                    air_name: air_name.clone(),
                });
            }
            if recursive1_dir.join("recursive1.so").exists() || recursive1_dir.join("recursive1.dylib").exists() {
                tasks.push(RebuildTask::Recursive1 {
                    air_root: air_root.clone(),
                    ag_idx,
                    ag_name: ag_name.clone(),
                    air_name: air_name.clone(),
                    has_compressor,
                });
            }
        }

        let r2_dir = ag_dir.join("recursive2");
        if r2_dir.join("recursive2.so").exists() || r2_dir.join("recursive2.dylib").exists() {
            tasks.push(RebuildTask::Recursive2 { r2_dir, ag_idx, ag_name: ag_name.clone(), airs });
        }
    }

    let final_dir = pk_root.join("vadcop_final");
    if final_dir.join("vadcop_final.so").exists() || final_dir.join("vadcop_final.dylib").exists() {
        tasks.push(RebuildTask::VadcopFinal { pk_root: pk_root.clone(), air_groups: air_groups.clone() });
    }

    let compressed_dir = pk_root.join("vadcop_final_compressed");
    if compressed_dir.join("vadcop_final_compressed.so").exists()
        || compressed_dir.join("vadcop_final_compressed.dylib").exists()
    {
        tasks.push(RebuildTask::VadcopFinalCompressed { pk_root: pk_root.clone() });
    }

    let total = tasks.len();
    if total == 0 {
        tracing::warn!("No witness libraries found under {}", pk_root.display());
        return Ok(0);
    }
    tracing::info!("Discovered {} witness library(ies) to rebuild:", total);
    for (i, t) in tasks.iter().enumerate() {
        tracing::info!("  [{}/{}] {}", i + 1, total, t.label());
    }

    // ── Execute: run circom compiles in parallel, bounded by `jobs`.
    let n_jobs = jobs.max(1);
    tracing::info!("Running circom compiles with {} job(s) in parallel", n_jobs);
    // Expression trees in `gen_stark_verifier` can be thousands of levels deep,
    // overflowing the default thread stack.  Match the 64 MB stack the rest of
    // the setup pipeline uses (see proofman-setup main / setup_jobs pool).
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_jobs)
        .stack_size(64 * 1024 * 1024)
        .build()
        .context("Failed to build rebuild thread pool")?;

    pool.install(|| {
        tasks.par_iter().enumerate().try_for_each(|(idx, task)| -> Result<()> {
            let progress = (idx + 1, total);
            execute_task(task, build_dir, &global_info, &global_constraints, paths, witness_tracker, progress)
        })
    })?;

    tracing::info!("All circom compiles complete; {} witness library build(s) running in background", total);
    Ok(total)
}

/// Dispatch a discovered task to the matching rebuilder, with progress logging
/// around each step so the user can see what is happening.
#[allow(clippy::too_many_arguments)]
fn execute_task(
    task: &RebuildTask,
    build_dir: &str,
    global_info: &Value,
    global_constraints: &Value,
    paths: &RebuildPaths<'_>,
    witness_tracker: &WitnessTracker,
    progress: (usize, usize),
) -> Result<()> {
    let label = task.label();
    let (i, n) = progress;
    tracing::info!("[{i}/{n}] Starting rebuild: {label}");
    let started = Instant::now();
    match task {
        RebuildTask::Compressor { air_root, ag_name, air_name } => {
            rebuild_compressor(build_dir, air_root, ag_name, air_name, global_info, paths, witness_tracker, progress)?;
        }
        RebuildTask::Recursive1 { air_root, ag_idx, ag_name, air_name, has_compressor } => {
            rebuild_recursive1(
                build_dir,
                air_root,
                *ag_idx,
                ag_name,
                air_name,
                *has_compressor,
                global_info,
                paths,
                witness_tracker,
                progress,
            )?;
        }
        RebuildTask::Recursive2 { r2_dir, ag_idx, ag_name, airs } => {
            rebuild_recursive2(
                build_dir,
                r2_dir,
                *ag_idx,
                ag_name,
                airs,
                global_info,
                paths,
                witness_tracker,
                progress,
            )?;
        }
        RebuildTask::VadcopFinal { pk_root, air_groups } => {
            rebuild_vadcop_final(
                build_dir,
                pk_root,
                air_groups,
                global_info,
                global_constraints,
                paths,
                witness_tracker,
                progress,
            )?;
        }
        RebuildTask::VadcopFinalCompressed { pk_root } => {
            rebuild_vadcop_final_compressed(build_dir, pk_root, paths, witness_tracker, progress)?;
        }
    }
    tracing::info!(
        "[{i}/{n}] circom compile finished for {label} in {:.1}s; witness library build queued",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Locate `<provingKey>/*.globalInfo.json`.
fn find_global_info(proving_key: &Path) -> Result<PathBuf> {
    for entry in fs::read_dir(proving_key)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            if name.ends_with(".globalInfo.json") {
                return Ok(entry.path());
            }
        }
    }
    bail!("Could not find *.globalInfo.json under {}", proving_key.display())
}

/// Load and parse a JSON file.
fn load_json(path: &Path) -> Result<Value> {
    serde_json::from_str(&fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

/// Read a verkey.json file as 4 stringified u64 limbs (matches the format
/// used elsewhere in the setup pipeline for `const_root`).
fn load_verkey_as_strings(path: &Path) -> Result<[String; 4]> {
    let v = load_json(path)?;
    let arr = v.as_array().with_context(|| format!("verkey is not an array: {}", path.display()))?;
    if arr.len() != 4 {
        bail!("verkey {} has {} entries, expected 4", path.display(), arr.len());
    }
    let mut out = [String::new(), String::new(), String::new(), String::new()];
    for (i, item) in arr.iter().enumerate() {
        out[i] = item
            .as_u64()
            .map(|n| n.to_string())
            .or_else(|| item.as_str().map(|s| s.to_string()))
            .with_context(|| format!("verkey limb {} not u64/str in {}", i, path.display()))?;
    }
    Ok(out)
}

/// Run a closure, logging start and elapsed time.  Useful for instrumenting
/// the in-process `pil2circom` / `gen_circom` calls so the user can tell
/// which step is slow without grepping for the next log line.
fn timed<T, F: FnOnce() -> Result<T>>(label: &str, progress: (usize, usize), step: &str, f: F) -> Result<T> {
    let (i, n) = progress;
    tracing::info!("[{i}/{n}] {label}: {step}...");
    let started = Instant::now();
    let r = f()?;
    tracing::info!("[{i}/{n}] {label}: {step} done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(r)
}

/// Shell-quote a string so the logged command can be copy-pasted into a shell.
fn shell_quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./_-".contains(c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// Spawn circom with inherited stdio so its `--verbose` output streams live
/// (otherwise long compiles look indistinguishable from a hang).  Logs the full
/// command upfront so it can be copy-pasted into a shell for debugging.
fn spawn_circom(args: &[&str], label: &str, progress: (usize, usize)) -> Result<()> {
    let (i, n) = progress;
    let cmd_str: Vec<String> = args.iter().map(|s| shell_quote(s)).collect();
    tracing::info!("[{i}/{n}] {label}: $ {}", cmd_str.join(" "));
    let started = Instant::now();
    let status = Command::new(args[0])
        .args(&args[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("Failed to spawn circom compiler")?;
    if !status.success() {
        bail!(
            "circom compilation failed for {} (exit {}); see streamed output above",
            label,
            status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string())
        );
    }
    tracing::info!("[{i}/{n}] {label}: circom done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

/// Run `circom <input> --c -O1 --prime goldilocks ...`.
fn run_circom_compile(
    circom_input: &Path,
    build_path: &Path,
    paths: &RebuildPaths<'_>,
    label: &str,
    progress: (usize, usize),
) -> Result<()> {
    fs::create_dir_all(build_path)?;
    let circom_input_s = circom_input.to_str().unwrap_or("");
    let build_path_s = build_path.to_str().unwrap_or("");
    let args = [
        paths.circom_exec,
        "--O1",
        "--r1cs",
        "--prime",
        "goldilocks",
        "--c",
        "--verbose",
        "-l",
        paths.recurser_circuits_path,
        "-l",
        paths.circuits_gl_path,
        circom_input_s,
        "-o",
        build_path_s,
    ];
    spawn_circom(&args, label, progress)
}

/// Run circom with the *compressed-final* circuits library on `-l`.  The
/// vadcop_final_compressed setup uses `recursion/helpers/circuits` instead of
/// `vadcop/helpers/circuits`.
fn run_circom_compile_compressed(
    circom_input: &Path,
    build_path: &Path,
    paths: &RebuildPaths<'_>,
    label: &str,
    progress: (usize, usize),
) -> Result<()> {
    fs::create_dir_all(build_path)?;
    let circom_input_s = circom_input.to_str().unwrap_or("");
    let build_path_s = build_path.to_str().unwrap_or("");
    let args = [
        paths.circom_exec,
        "--O1",
        "--r1cs",
        "--prime",
        "goldilocks",
        "--c",
        "--verbose",
        "-l",
        paths.recurser_circuits_compressed_final_path,
        "-l",
        paths.circuits_gl_path,
        circom_input_s,
        "-o",
        build_path_s,
    ];
    spawn_circom(&args, label, progress)
}

/// Ensure the scratch subdirectories needed by the setup pipeline exist.
fn ensure_scratch_dirs(build_dir: &str) -> Result<(PathBuf, PathBuf)> {
    let circom_dir = PathBuf::from(build_dir).join("circom");
    let build_path = PathBuf::from(build_dir).join("build");
    fs::create_dir_all(&circom_dir)?;
    fs::create_dir_all(&build_path)?;
    Ok((circom_dir, build_path))
}

// ── compressor ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn rebuild_compressor(
    build_dir: &str,
    air_root: &Path,
    ag_name: &str,
    air_name: &str,
    global_info: &Value,
    paths: &RebuildPaths<'_>,
    witness_tracker: &WitnessTracker,
    progress: (usize, usize),
) -> Result<()> {
    let _ = ag_name;
    let (i, n) = progress;
    let label = format!("compressor / {ag_name}.{air_name}");
    tracing::info!("[{i}/{n}] {label}: loading starkinfo / verifierinfo / verkey");
    let air_dir = air_root.join("air");
    let stark_info = load_json(&air_dir.join(format!("{}.starkinfo.json", air_name)))?;
    let verifier_info = load_json(&air_dir.join(format!("{}.verifierinfo.json", air_name)))?;
    let const_root = load_verkey_as_strings(&air_dir.join(format!("{}.verkey.json", air_name)))?;

    let (circom_dir, build_path) = ensure_scratch_dirs(build_dir)?;

    let verifier_name = format!("{}.verifier.circom", air_name);
    let name_filename = format!("{}_compressor", air_name);

    let verifier_circom = timed(&label, progress, "running pil2circom (verifier)", || {
        pil2circom(
            &const_root,
            &stark_info,
            &verifier_info,
            &Pil2CircomOptions { skip_main: true, verkey_input: false, enable_input: false, input_challenges: true },
        )
    })?;
    fs::write(circom_dir.join(&verifier_name), &verifier_circom)?;

    let verifier_filenames = vec![verifier_name];
    let circom_str = timed(&label, progress, "running gen_circom (wrapper)", || {
        gen_circom(&GenCircomInput {
            template_name: "src/vadcop/templates/compressor.circom.ejs",
            stark_infos: std::slice::from_ref(&stark_info),
            // Compressor's gen_circom uses vadcop_info heavily (n_publics,
            // numProofValues, latticeSize, aggTypes). Passing Null here makes
            // gen_calculate_hashes loop forever (lattice_size=0 → n_rounds=0 →
            // `0..(0-1)` underflows usize in release).
            vadcop_info: global_info,
            verifier_filenames: &verifier_filenames,
            basic_verification_keys: &[],
            agg_verification_keys: &[],
            publics: &[],
            options: &GenCircomOptions { ..Default::default() },
        })
    })?;
    let circom_out = circom_dir.join(format!("{}.circom", name_filename));
    fs::write(&circom_out, &circom_str)?;

    run_circom_compile(&circom_out, &build_path, paths, &label, progress)?;

    let files_dir = air_root.join("compressor");
    tracing::info!("[{i}/{n}] {label}: spawning witness library build (.so/.dylib)");
    witness_tracker.run_witness_library_generation(
        build_dir,
        files_dir.to_str().unwrap_or(""),
        &name_filename,
        "compressor",
        paths.circom_helpers_dir,
    );
    Ok(())
}

// ── recursive1 ──────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn rebuild_recursive1(
    build_dir: &str,
    air_root: &Path,
    ag_idx: usize,
    ag_name: &str,
    air_name: &str,
    has_compressor: bool,
    global_info: &Value,
    paths: &RebuildPaths<'_>,
    witness_tracker: &WitnessTracker,
    progress: (usize, usize),
) -> Result<()> {
    let _ = ag_name;
    let (i, n) = progress;
    let label_suffix = if has_compressor { " (with compressor)" } else { "" };
    let label = format!("recursive1 / {ag_name}.{air_name}{label_suffix}");
    tracing::info!("[{i}/{n}] {label}: loading starkinfo / verifierinfo / verkey");
    // pil2circom inputs:
    //   - if compressor: read from <air>/compressor/
    //   - else:           read from <air>/air/<air>.*
    let (stark_info, verifier_info, const_root, verifier_name) = if has_compressor {
        let cdir = air_root.join("compressor");
        let si = load_json(&cdir.join("compressor.starkinfo.json"))?;
        let vi = load_json(&cdir.join("compressor.verifierinfo.json"))?;
        let vk = load_verkey_as_strings(&cdir.join("compressor.verkey.json"))?;
        (si, vi, vk, format!("{}_compressor.verifier.circom", air_name))
    } else {
        let adir = air_root.join("air");
        let si = load_json(&adir.join(format!("{}.starkinfo.json", air_name)))?;
        let vi = load_json(&adir.join(format!("{}.verifierinfo.json", air_name)))?;
        let vk = load_verkey_as_strings(&adir.join(format!("{}.verkey.json", air_name)))?;
        (si, vi, vk, format!("{}.verifier.circom", air_name))
    };

    let (circom_dir, build_path) = ensure_scratch_dirs(build_dir)?;

    let input_challenges = !has_compressor;
    let verifier_circom = timed(&label, progress, "running pil2circom (verifier)", || {
        pil2circom(
            &const_root,
            &stark_info,
            &verifier_info,
            &Pil2CircomOptions { skip_main: true, verkey_input: false, enable_input: false, input_challenges },
        )
    })?;
    fs::write(circom_dir.join(&verifier_name), &verifier_circom)?;

    let name_filename = format!("{}_recursive1", air_name);
    let verifier_filenames = vec![verifier_name];
    let circom_str = timed(&label, progress, "running gen_circom (wrapper)", || {
        gen_circom(&GenCircomInput {
            template_name: "src/vadcop/templates/recursive1.circom.ejs",
            stark_infos: std::slice::from_ref(&stark_info),
            vadcop_info: global_info,
            verifier_filenames: &verifier_filenames,
            basic_verification_keys: &[],
            agg_verification_keys: &[],
            publics: &[],
            options: &GenCircomOptions { airgroup_id: Some(ag_idx as u64), has_compressor, ..Default::default() },
        })
    })?;
    let circom_out = circom_dir.join(format!("{}.circom", name_filename));
    fs::write(&circom_out, &circom_str)?;

    run_circom_compile(&circom_out, &build_path, paths, &label, progress)?;

    let files_dir = air_root.join("recursive1");
    tracing::info!("[{i}/{n}] {label}: spawning witness library build (.so/.dylib)");
    witness_tracker.run_witness_library_generation(
        build_dir,
        files_dir.to_str().unwrap_or(""),
        &name_filename,
        "recursive1",
        paths.circom_helpers_dir,
    );
    Ok(())
}

// ── recursive2 ──────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn rebuild_recursive2(
    build_dir: &str,
    r2_dir: &Path,
    ag_idx: usize,
    ag_name: &str,
    airs: &[Value],
    global_info: &Value,
    paths: &RebuildPaths<'_>,
    witness_tracker: &WitnessTracker,
    progress: (usize, usize),
) -> Result<()> {
    let (i, n) = progress;
    let label = format!("recursive2 / {ag_name}");
    tracing::info!("[{i}/{n}] {label}: loading recursive2 starkinfo / verifierinfo");
    // Inputs to pil2circom for the recursive2 verifier circom:
    //   - stark_info + verifier_info come from recursive1's setup, which is not
    //     persisted.  However recursive1 and recursive2 share the same plonk2pil
    //     "aggregation" template, so recursive2's persisted starkinfo/verifierinfo
    //     are functionally equivalent for generating the inner verifier.
    let stark_info = load_json(&r2_dir.join("recursive2.starkinfo.json"))?;
    let verifier_info = load_json(&r2_dir.join("recursive2.verifierinfo.json"))?;
    // verkey_input=true, so const_root is unused — pass zeros.
    let zero_root: [String; 4] = ["0".into(), "0".into(), "0".into(), "0".into()];

    // `enable_input` is true when there are multiple airgroups OR multiple airs in
    // the first airgroup (matches resolve_names_and_paths in recursive.rs).
    let n_airgroups = global_info.get("air_groups").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(1);
    let n_airs_first = global_info
        .get("airs")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(1);
    let enable_input = n_airgroups > 1 || n_airs_first > 1;

    let (circom_dir, build_path) = ensure_scratch_dirs(build_dir)?;

    let verifier_name = format!("{}_recursive2.verifier.circom", ag_name);
    let verifier_circom = timed(&label, progress, "running pil2circom (verifier)", || {
        pil2circom(
            &zero_root,
            &stark_info,
            &verifier_info,
            &Pil2CircomOptions { skip_main: true, verkey_input: true, enable_input, input_challenges: false },
        )
    })?;
    fs::write(circom_dir.join(&verifier_name), &verifier_circom)?;

    // basic_verification_keys = per-air recursive1 verkeys for this airgroup,
    // wrapped as [[vk_air0, vk_air1, ...]] (single-airgroup outer slice).
    let mut per_air_vks: Vec<Vec<String>> = Vec::new();
    for air in airs {
        let air_name = match air.get("name").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let r1_vk = r2_dir
            .parent()
            .map(|p| p.join("airs").join(air_name).join("recursive1").join("recursive1.verkey.json"))
            .unwrap();
        if r1_vk.exists() {
            let vk = load_verkey_as_strings(&r1_vk)?;
            per_air_vks.push(vk.to_vec());
        }
    }
    let basic_vk: Vec<Vec<Vec<String>>> = vec![per_air_vks];

    let name_filename = format!("{}_recursive2", ag_name);
    let verifier_filenames = vec![verifier_name];
    let circom_str = timed(&label, progress, "running gen_circom (wrapper)", || {
        gen_circom(&GenCircomInput {
            template_name: "src/vadcop/templates/recursive2.circom.ejs",
            stark_infos: std::slice::from_ref(&stark_info),
            vadcop_info: global_info,
            verifier_filenames: &verifier_filenames,
            basic_verification_keys: &basic_vk,
            agg_verification_keys: &[],
            publics: &[],
            options: &GenCircomOptions { airgroup_id: Some(ag_idx as u64), ..Default::default() },
        })
    })?;
    let circom_out = circom_dir.join(format!("{}.circom", name_filename));
    fs::write(&circom_out, &circom_str)?;

    run_circom_compile(&circom_out, &build_path, paths, &label, progress)?;

    tracing::info!("[{i}/{n}] {label}: spawning witness library build (.so/.dylib)");
    witness_tracker.run_witness_library_generation(
        build_dir,
        r2_dir.to_str().unwrap_or(""),
        &name_filename,
        "recursive2",
        paths.circom_helpers_dir,
    );
    Ok(())
}

// ── vadcop_final ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn rebuild_vadcop_final(
    build_dir: &str,
    pk_root: &Path,
    air_groups: &[String],
    global_info: &Value,
    global_constraints: &Value,
    paths: &RebuildPaths<'_>,
    witness_tracker: &WitnessTracker,
    progress: (usize, usize),
) -> Result<()> {
    let (i, n) = progress;
    let label = "vadcop_final".to_string();
    tracing::info!("[{i}/{n}] {label}: loading per-airgroup recursive2 starkinfo / verifierinfo / vks");
    let mut stark_infos: Vec<Value> = Vec::new();
    let mut verifier_infos: Vec<Value> = Vec::new();
    let mut agg_keys_recursive2: Vec<Vec<String>> = Vec::new();
    let mut basic_keys_recursive1: Vec<Vec<Vec<String>>> = Vec::new();
    let mut verifier_names: Vec<String> = Vec::new();

    for ag in air_groups {
        let r2_dir = pk_root.join(ag).join("recursive2");
        let si = load_json(&r2_dir.join("recursive2.starkinfo.json"))?;
        let vi = load_json(&r2_dir.join("recursive2.verifierinfo.json"))?;
        let vks = load_json(&r2_dir.join("recursive2.vks.json"))?;

        stark_infos.push(si);
        verifier_infos.push(vi);

        let agg = vks
            .get("rootCRecursive2")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(json_val_to_str).collect::<Vec<_>>())
            .unwrap_or_default();
        agg_keys_recursive2.push(agg);

        let basic = vks
            .get("rootCRecursives1")
            .and_then(|v| v.as_array())
            .map(|airs| {
                airs.iter()
                    .map(|air_vk| {
                        air_vk
                            .as_array()
                            .map(|vals| vals.iter().map(json_val_to_str).collect::<Vec<_>>())
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        basic_keys_recursive1.push(basic);

        verifier_names.push(format!("{}_recursive2.verifier.circom", ag));
    }

    let (circom_dir, build_path) = ensure_scratch_dirs(build_dir)?;

    // Per-airgroup verifier circom (each takes the airgroup's recursive2 verkey
    // as const_root via the rootCRecursive2 limbs from vks.json).
    let n_ags = stark_infos.len();
    let step = format!("running pil2circom for {n_ags} airgroup verifier(s)");
    timed(&label, progress, &step, || {
        for (idx, (si, vi)) in stark_infos.iter().zip(verifier_infos.iter()).enumerate() {
            let const_root: [String; 4] = if agg_keys_recursive2[idx].len() == 4 {
                [
                    agg_keys_recursive2[idx][0].clone(),
                    agg_keys_recursive2[idx][1].clone(),
                    agg_keys_recursive2[idx][2].clone(),
                    agg_keys_recursive2[idx][3].clone(),
                ]
            } else {
                ["0".into(), "0".into(), "0".into(), "0".into()]
            };
            let verifier_circom = pil2circom(
                &const_root,
                si,
                vi,
                &Pil2CircomOptions { skip_main: true, verkey_input: true, enable_input: true, input_challenges: false },
            )?;
            fs::write(circom_dir.join(&verifier_names[idx]), &verifier_circom)?;
        }
        Ok(())
    })?;

    // Wrapper circom — vadcop_info needs globalConstraints merged in.
    let mut final_global_info = global_info.clone();
    if let Some(constraints) = global_constraints.get("constraints") {
        if let Some(obj) = final_global_info.as_object_mut() {
            obj.insert("globalConstraints".to_string(), constraints.clone());
        }
    }

    let circom_str = timed(&label, progress, "running gen_circom (wrapper)", || {
        gen_circom(&GenCircomInput {
            template_name: "src/vadcop/templates/final.circom.ejs",
            stark_infos: &stark_infos,
            vadcop_info: &final_global_info,
            verifier_filenames: &verifier_names,
            basic_verification_keys: &basic_keys_recursive1,
            agg_verification_keys: &agg_keys_recursive2,
            publics: &[],
            options: &GenCircomOptions { is_final: true, ..Default::default() },
        })
    })?;
    let circom_out = circom_dir.join("vadcop_final.circom");
    fs::write(&circom_out, &circom_str)?;

    run_circom_compile(&circom_out, &build_path, paths, &label, progress)?;

    let files_dir = pk_root.join("vadcop_final");
    tracing::info!("[{i}/{n}] {label}: spawning witness library build (.so/.dylib)");
    witness_tracker.run_witness_library_generation(
        build_dir,
        files_dir.to_str().unwrap_or(""),
        "vadcop_final",
        "vadcop_final",
        paths.circom_helpers_dir,
    );
    Ok(())
}

// ── vadcop_final_compressed ──────────────────────────────────────────────────

fn rebuild_vadcop_final_compressed(
    build_dir: &str,
    pk_root: &Path,
    paths: &RebuildPaths<'_>,
    witness_tracker: &WitnessTracker,
    progress: (usize, usize),
) -> Result<()> {
    let (i, n) = progress;
    let label = "vadcop_final_compressed".to_string();
    tracing::info!("[{i}/{n}] {label}: loading vadcop_final starkinfo / verifierinfo / verkey");
    let final_dir = pk_root.join("vadcop_final");
    let stark_info = load_json(&final_dir.join("vadcop_final.starkinfo.json"))?;
    let verifier_info = load_json(&final_dir.join("vadcop_final.verifierinfo.json"))?;
    let const_root = load_verkey_as_strings(&final_dir.join("vadcop_final.verkey.json"))?;

    let (circom_dir, build_path) = ensure_scratch_dirs(build_dir)?;

    let verifier_name = "vadcop_final_stark.verifier.circom";
    let verifier_circom = timed(&label, progress, "running pil2circom (verifier)", || {
        pil2circom(
            &const_root,
            &stark_info,
            &verifier_info,
            &Pil2CircomOptions { skip_main: true, verkey_input: false, enable_input: false, input_challenges: false },
        )
    })?;
    fs::write(circom_dir.join(verifier_name), &verifier_circom)?;

    let template = "vadcop_final_compressed";
    let verifier_filenames = vec![verifier_name.to_string()];
    let circom_str = timed(&label, progress, "running gen_circom (wrapper)", || {
        gen_circom(&GenCircomInput {
            template_name: "src/vadcop/templates/final_compressed.circom.ejs",
            stark_infos: std::slice::from_ref(&stark_info),
            vadcop_info: &Value::Null,
            verifier_filenames: &verifier_filenames,
            basic_verification_keys: &[],
            agg_verification_keys: &[],
            publics: &[],
            options: &GenCircomOptions::default(),
        })
    })?;
    let circom_out = circom_dir.join(format!("{}.circom", template));
    fs::write(&circom_out, &circom_str)?;

    run_circom_compile_compressed(&circom_out, &build_path, paths, &label, progress)?;

    let files_dir = pk_root.join(template);
    tracing::info!("[{i}/{n}] {label}: spawning witness library build (.so/.dylib)");
    witness_tracker.run_witness_library_generation(
        build_dir,
        files_dir.to_str().unwrap_or(""),
        template,
        template,
        paths.circom_helpers_dir,
    );
    Ok(())
}

fn json_val_to_str(v: &Value) -> String {
    v.as_str().map(|s| s.to_string()).or_else(|| v.as_u64().map(|n| n.to_string())).unwrap_or_else(|| "0".to_string())
}
