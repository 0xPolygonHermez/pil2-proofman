//! Rebuild circom witness libraries (`.so` / `.dylib`) directly from a
//! `provingKey/` tree, without re-running circom.
//!
//! Setup persists the circom-generated C++ (`{template}.cpp`) next to each
//! witness library. This command discovers those stored sources and recompiles
//! them with `make` — skipping the expensive `pil2circom` / `gen_circom` /
//! `circom --c` regeneration entirely. A proving key produced before `.cpp`
//! persistence has no stored sources; rebuilding it errors and asks the user to
//! re-run `proofman-setup setup`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde_json::Value;

use crate::output::witness_gen::WitnessTracker;

/// A witness library to rebuild: its directory in the proving key and the
/// template name shared by its `{template}.cpp` / `{template}.so` files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLib {
    pub files_dir: PathBuf,
    pub template: String,
}

/// Locate `<provingKey>/*.globalInfo.json`.
fn find_global_info(proving_key: &Path) -> Result<PathBuf> {
    for entry in
        fs::read_dir(proving_key).with_context(|| format!("Cannot read provingKey dir {}", proving_key.display()))?
    {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            if name.ends_with(".globalInfo.json") {
                return Ok(entry.path());
            }
        }
    }
    bail!("Could not find *.globalInfo.json under {}", proving_key.display())
}

/// Classify a candidate witness-lib directory:
/// - `{template}.cpp` present  → return the lib (rebuildable).
/// - `.so`/`.dylib` present but no `.cpp` → error (proving key predates `.cpp`
///   persistence).
/// - neither present → `None` (this lib does not exist in the key).
fn classify_lib(dir: &Path, template: &str) -> Result<Option<WitnessLib>> {
    if dir.join(format!("{template}.cpp")).exists() {
        return Ok(Some(WitnessLib { files_dir: dir.to_path_buf(), template: template.to_string() }));
    }
    let so = dir.join(format!("{template}.so"));
    let dylib = dir.join(format!("{template}.dylib"));
    if so.exists() || dylib.exists() {
        bail!(
            "Found a {template} witness library in {} but no {template}.cpp — this proving key \
             predates .cpp persistence. Re-run `proofman-setup setup`.",
            dir.display()
        );
    }
    Ok(None)
}

/// Walk a `provingKey/` tree and collect every witness library with a stored
/// `.cpp`. Structure-aware (driven by `globalInfo`) so it never picks up the
/// final-snark `.cpp`, which requires a different helpers dir.
pub fn discover_witness_libs(proving_key: &Path) -> Result<Vec<WitnessLib>> {
    if !proving_key.is_dir() {
        bail!("provingKey directory not found at {}", proving_key.display());
    }

    let global_info_path = find_global_info(proving_key)?;
    let global_info: Value = serde_json::from_str(&fs::read_to_string(&global_info_path)?)
        .with_context(|| format!("Failed to parse {}", global_info_path.display()))?;
    let global_name =
        global_info.get("name").and_then(|v| v.as_str()).context("globalInfo.json missing 'name' field")?;
    let pk_root = proving_key.join(global_name);
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

    let mut libs = Vec::new();
    for (ag_idx, ag_name) in air_groups.iter().enumerate() {
        let ag_dir = pk_root.join(ag_name);
        if !ag_dir.is_dir() {
            tracing::warn!("Airgroup '{}' missing in proving key, skipping", ag_name);
            continue;
        }
        let airs = airs_per_group.get(ag_idx).cloned().unwrap_or_default();
        for air in &airs {
            let air_name = match air.get("name").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => continue,
            };
            let air_root = ag_dir.join("airs").join(air_name);
            if let Some(lib) = classify_lib(&air_root.join("compressor"), "compressor")? {
                libs.push(lib);
            }
            if let Some(lib) = classify_lib(&air_root.join("recursive1"), "recursive1")? {
                libs.push(lib);
            }
        }
        if let Some(lib) = classify_lib(&ag_dir.join("recursive2"), "recursive2")? {
            libs.push(lib);
        }
    }
    if let Some(lib) = classify_lib(&pk_root.join("vadcop_final"), "vadcop_final")? {
        libs.push(lib);
    }
    if let Some(lib) = classify_lib(&pk_root.join("vadcop_final_compressed"), "vadcop_final_compressed")? {
        libs.push(lib);
    }

    Ok(libs)
}

/// Rebuild every witness library found under `proving_key` by recompiling its
/// stored `.cpp` with `make`. `jobs` bounds how many `make` builds run at once
/// (each is g++-bound and uses ~1–2 GB RAM). Returns the number rebuilt.
pub fn rebuild_all_witness_libs(
    proving_key: &str,
    circom_helpers_dir: &str,
    witness_tracker: &WitnessTracker,
    jobs: usize,
) -> Result<usize> {
    let libs = discover_witness_libs(Path::new(proving_key))?;
    if libs.is_empty() {
        tracing::warn!("No witness libraries with a stored .cpp found under {}", proving_key);
        return Ok(0);
    }

    tracing::info!("Discovered {} witness library(ies) to rebuild:", libs.len());
    for (i, lib) in libs.iter().enumerate() {
        tracing::info!("  [{}/{}] {} ({})", i + 1, libs.len(), lib.template, lib.files_dir.display());
    }

    let n_jobs = jobs.max(1);
    tracing::info!("Compiling witness libraries with up to {} concurrent make build(s)", n_jobs);
    let pool =
        rayon::ThreadPoolBuilder::new().num_threads(n_jobs).build().context("Failed to build rebuild thread pool")?;

    pool.install(|| {
        libs.par_iter().try_for_each(|lib| -> Result<()> {
            tracing::info!("Rebuilding {} in {}", lib.template, lib.files_dir.display());
            witness_tracker.build_witness_library_from_stored(
                lib.files_dir.to_str().unwrap_or(""),
                &lib.template,
                circom_helpers_dir,
            )
        })
    })?;

    Ok(libs.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal proving key: `<root>/provingKey/` with a globalInfo
    /// naming one airgroup (`AgA`) holding one air (`Air0`), and `pk_root` at
    /// `provingKey/myvk`. Returns the `provingKey/` path.
    fn make_proving_key(root: &Path) -> PathBuf {
        let pk = root.join("provingKey");
        fs::create_dir_all(&pk).unwrap();
        fs::write(
            pk.join("myvk.globalInfo.json"),
            r#"{"name":"myvk","air_groups":["AgA"],"airs":[[{"name":"Air0"}]]}"#,
        )
        .unwrap();
        fs::create_dir_all(pk.join("myvk").join("AgA").join("airs").join("Air0")).unwrap();
        pk
    }

    fn rec1_dir(pk: &Path) -> PathBuf {
        pk.join("myvk").join("AgA").join("airs").join("Air0").join("recursive1")
    }

    #[test]
    fn discovers_lib_with_stored_cpp() {
        let tmp = tempfile::tempdir().unwrap();
        let pk = make_proving_key(tmp.path());
        let dir = rec1_dir(&pk);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("recursive1.cpp"), b"// witness\n").unwrap();

        let libs = discover_witness_libs(&pk).unwrap();
        assert_eq!(libs, vec![WitnessLib { files_dir: dir, template: "recursive1".to_string() }]);
    }

    #[test]
    fn errors_when_so_present_but_cpp_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let pk = make_proving_key(tmp.path());
        let dir = rec1_dir(&pk);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("recursive1.so"), b"\x7fELF").unwrap();

        let err = discover_witness_libs(&pk).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("recursive1.cpp"), "unexpected error: {msg}");
        assert!(msg.contains("Re-run"), "error should ask to re-run setup: {msg}");
    }

    #[test]
    fn empty_when_no_witness_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let pk = make_proving_key(tmp.path());
        let libs = discover_witness_libs(&pk).unwrap();
        assert!(libs.is_empty());
    }

    #[test]
    fn discovers_all_templates_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let pk = make_proving_key(tmp.path());
        let air0 = pk.join("myvk").join("AgA").join("airs").join("Air0");

        // compressor + recursive1 under the air; recursive2 under the airgroup;
        // vadcop_final + vadcop_final_compressed at the proving-key root.
        for (dir, file) in [
            (air0.join("compressor"), "compressor.cpp"),
            (air0.join("recursive1"), "recursive1.cpp"),
            (pk.join("myvk").join("AgA").join("recursive2"), "recursive2.cpp"),
            (pk.join("myvk").join("vadcop_final"), "vadcop_final.cpp"),
            (pk.join("myvk").join("vadcop_final_compressed"), "vadcop_final_compressed.cpp"),
        ] {
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(file), b"// witness\n").unwrap();
        }

        let libs = discover_witness_libs(&pk).unwrap();
        let templates: Vec<&str> = libs.iter().map(|l| l.template.as_str()).collect();
        assert_eq!(
            templates,
            vec!["compressor", "recursive1", "recursive2", "vadcop_final", "vadcop_final_compressed"]
        );
    }

    #[test]
    fn orphan_dylib_without_cpp_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let pk = make_proving_key(tmp.path());
        let dir = rec1_dir(&pk);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("recursive1.dylib"), b"\xcf\xfa\xed\xfe").unwrap();

        let err = discover_witness_libs(&pk).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("recursive1.cpp"), "unexpected error: {msg}");
    }
}
