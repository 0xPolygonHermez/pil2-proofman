//! Port of `generateWitness.js`: compile circom C++ witness code into a
//! shared library (.so on Linux, .dylib on macOS) via `make`.
//!
//! The JS version copies helper C++ files to a temp directory, overlays the
//! generated verifier.cpp, and invokes `make -j witness`. We do the same here,
//! but spawn the build asynchronously so that multiple witness libraries can
//! be built in parallel.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};

/// Tracks pending witness library builds so we can wait for all of them.
#[derive(Clone, Default)]
pub struct WitnessTracker {
    pending: Arc<Mutex<Vec<std::thread::JoinHandle<Result<()>>>>>,
    /// Directory containing goldilocks source files (e.g. pil2-stark/src/goldilocks/src).
    /// These are copied into the temp build dir before the circom helpers so they are
    /// available for compilation (goldilocks_base_field.hpp, poseidon2_goldilocks_constants.hpp, etc.)
    goldilocks_src_dir: Option<String>,
}

impl WitnessTracker {
    pub fn new() -> Self {
        Self { pending: Arc::new(Mutex::new(Vec::new())), goldilocks_src_dir: None }
    }

    pub fn with_goldilocks_src(goldilocks_src_dir: impl Into<String>) -> Self {
        Self { pending: Arc::new(Mutex::new(Vec::new())), goldilocks_src_dir: Some(goldilocks_src_dir.into()) }
    }

    /// Kick off a witness library generation in a background thread.
    ///
    /// Ports `runWitnessLibraryGeneration()` from `generateWitness.js`.
    ///
    /// # Arguments
    /// * `build_dir` - The build output directory (contains `build/<name>_cpp/`)
    /// * `files_dir` - Where the .so/.dylib output goes
    /// * `name_filename` - Base name of the circom circuit (e.g. "Fibonacci_recursive1")
    /// * `template` - Template name used for the output file
    /// * `circom_helpers_dir` - Directory containing the Makefile and helper C++ files
    pub fn run_witness_library_generation(
        &self,
        build_dir: &str,
        files_dir: &str,
        name_filename: &str,
        template: &str,
        circom_helpers_dir: &str,
    ) {
        let build_dir = build_dir.to_string();
        let files_dir = files_dir.to_string();
        let name_filename = name_filename.to_string();
        let template = template.to_string();
        let circom_helpers_dir = circom_helpers_dir.to_string();
        let goldilocks_src_dir = self.goldilocks_src_dir.clone();

        let handle = std::thread::spawn(move || {
            generate_witness_library(
                &build_dir,
                &files_dir,
                &name_filename,
                &template,
                &circom_helpers_dir,
                goldilocks_src_dir.as_deref(),
            )
        });

        let mut pending = self.pending.lock().unwrap();
        pending.push(handle);
    }

    /// Wait for all pending witness library builds to complete.
    ///
    /// Ports `witnessLibraryGenerationAwait()` from `generateWitness.js`.
    pub fn await_all(&self) -> Result<()> {
        tracing::info!("Waiting for all witness library generation to complete");
        let handles: Vec<_> = {
            let mut pending = self.pending.lock().unwrap();
            std::mem::take(&mut *pending)
        };

        let count = handles.len();
        if count > 0 {
            tracing::info!("Waiting for {} witness libraries...", count);
        }

        let mut errors = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => errors.push(format!("{:#}", e)),
                Err(_) => errors.push("Witness generation thread panicked".to_string()),
            }
        }

        if !errors.is_empty() {
            bail!("Witness library generation errors:\n{}", errors.join("\n"));
        }

        Ok(())
    }
}

/// Generate a witness shared library by copying helper files to a temp
/// directory, overlaying the generated verifier.cpp, and running make.
fn generate_witness_library(
    build_dir: &str,
    files_dir: &str,
    name_filename: &str,
    template: &str,
    circom_helpers_dir: &str,
    goldilocks_src_dir: Option<&str>,
) -> Result<()> {
    let tmp_dir = tempfile::tempdir().context("Failed to create temp dir for witness gen")?;
    let tmp_path = tmp_dir.path();

    // Copy goldilocks sources first so circom helpers can rely on them.
    // These provide goldilocks_base_field.hpp, poseidon2_goldilocks_constants.hpp, etc.
    if let Some(gdir) = goldilocks_src_dir {
        let gpath = Path::new(gdir);
        if gpath.exists() {
            copy_dir_contents(gpath, tmp_path)?;
        } else {
            tracing::warn!("GOLDILOCKS_SRC_DIR '{}' not found, skipping goldilocks copy", gdir);
        }
    }

    // Copy helper files from circom_helpers_dir into tmp
    if Path::new(circom_helpers_dir).exists() {
        copy_dir_contents(Path::new(circom_helpers_dir), tmp_path)?;
    }

    // Copy generated C++ file
    let cpp_src = PathBuf::from(build_dir)
        .join("build")
        .join(format!("{}_cpp", name_filename))
        .join(format!("{}.cpp", name_filename));
    let cpp_dst = tmp_path.join("verifier.cpp");
    if cpp_src.exists() {
        fs::copy(&cpp_src, &cpp_dst)
            .with_context(|| format!("Failed to copy {} to {}", cpp_src.display(), cpp_dst.display()))?;
    } else {
        tracing::warn!("C++ source file not found: {}, witness lib may fail", cpp_src.display());
    }

    // Ensure output directory exists
    fs::create_dir_all(files_dir)?;

    let file_extension = if cfg!(target_os = "macos") { "dylib" } else { "so" };

    tracing::info!("Generating witness library for {}...", name_filename);

    let output = Command::new("make")
        .args([
            "-C",
            tmp_path.to_str().unwrap_or(""),
            "-j",
            "witness",
            &format!(
                "WITNESS_DIR={}",
                fs::canonicalize(files_dir).unwrap_or_else(|_| PathBuf::from(files_dir)).display()
            ),
            &format!("WITNESS_FILE={}.{}", template, file_extension),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("Failed to execute make for witness library")?;

    if !output.status.success() {
        // Write build logs so the full output is inspectable even when truncated in the terminal.
        let log_path = PathBuf::from(files_dir).join("build.log");
        let err_path = PathBuf::from(files_dir).join("build.err");
        let _ = fs::write(&log_path, &output.stdout);
        let _ = fs::write(&err_path, &output.stderr);

        // Collect the last ~50 lines of combined output for the error message
        // (make -j interleaves stdout/stderr; errors commonly appear in stdout).
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined: String = stdout
            .lines()
            .chain(stderr.lines())
            .filter(|l| l.contains("error:") || l.contains("Error") || l.starts_with("make"))
            .take(20)
            .collect::<Vec<_>>()
            .join("\n");
        bail!(
            "make failed for witness library '{}' (logs: {}, {})\n{}",
            name_filename,
            log_path.display(),
            err_path.display(),
            combined
        );
    }

    tracing::info!("Witness library for {} generated", name_filename);
    Ok(())
}

/// Recursively copy directory contents, skipping any `build` subdirectory
/// (which may contain stale pre-compiled .o files from a prior in-place build).
fn copy_dir_contents(src: &Path, dst: &Path) -> Result<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let name = entry.file_name();
        // Skip the `build` directory — it may contain stale object files that
        // would fool `make` into skipping recompilation with the wrong sources.
        if ty.is_dir() && name == "build" {
            continue;
        }
        let dest = dst.join(&name);
        if ty.is_file() {
            fs::copy(entry.path(), &dest)?;
        } else if ty.is_dir() {
            fs::create_dir_all(&dest)?;
            copy_dir_contents(&entry.path(), &dest)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_tracker_empty() {
        let tracker = WitnessTracker::new();
        // Awaiting with no pending tasks should succeed immediately
        assert!(tracker.await_all().is_ok());
    }
}
