use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// Detects whether GPU (CUDA) support is available.
/// Returns false if the cpu-only feature is set or if no CUDA toolkit is found.
fn detect_gpu() -> bool {
    if cfg!(feature = "cpu-only") {
        return false;
    }
    // Check for nvcc in standard CUDA location or PATH
    let nvcc_in_cuda = Path::new("/usr/local/cuda/bin/nvcc").exists();
    let nvcc_in_path = Command::new("nvcc").arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    nvcc_in_cuda || nvcc_in_path
}

fn main() {
    // CUDA arch resolution lives entirely in pil2-stark/Makefile
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_GENCODE_FLAGS");

    // Force this script to run on every cargo build: the bump file is
    // rewritten on each run, so its mtime always post-dates the previous
    // fingerprint.
    let bump = Path::new(&env::var("OUT_DIR").unwrap()).join("always-rerun");
    fs::write(&bump, format!("{:?}", SystemTime::now())).ok();
    println!("cargo:rerun-if-changed={}", bump.display());

    // Determine if GPU support should be used:
    // - If cpu-only feature is set, always use CPU
    // - Otherwise, auto-detect CUDA availability
    let use_gpu = if cfg!(feature = "cpu-only") {
        println!("cargo:warning=[BUILD INFO] STARKS compiled with CPU-only support (feature enabled)");
        false
    } else if detect_gpu() {
        println!("cargo:warning=[BUILD INFO] STARKS compiled with GPU support");
        true
    } else {
        println!("cargo:warning=[BUILD INFO] STARKS compiled with CPU-only support (CUDA not detected)");
        false
    };

    // Set build mode as environment variable for runtime access
    if use_gpu {
        println!("cargo:rustc-env=STARKS_BUILD_MODE=GPU");
    } else {
        println!("cargo:rustc-env=STARKS_BUILD_MODE=CPU");
    }

    // Canonicalize to avoid ".." in rerun-if-changed paths
    let pil2_stark_path_raw = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pil2-stark");
    let pil2_stark_path = pil2_stark_path_raw.canonicalize().unwrap_or_else(|_| pil2_stark_path_raw.clone());
    let library_folder = if use_gpu { pil2_stark_path.join("lib-gpu") } else { pil2_stark_path.join("lib") };
    let library_name = if use_gpu { "starksgpu" } else { "starks" };
    let lib_file = library_folder.join(format!("lib{library_name}.a"));

    // The CPU and GPU variants of this script can run concurrently against the
    // same checkout (one cargo build graph may contain both, and a second cargo
    // process or rust-analyzer can overlap with either). All staleness probes,
    // cleans and make invocations below mutate shared in-tree state, so they
    // must be serialized; the lock is held until this process exits.
    let lock_path = pil2_stark_path.join(".build_lock");
    let build_lock = fs::File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .unwrap_or_else(|e| panic!("Failed to open build lock {}: {e}", lock_path.display()));
    build_lock.lock().unwrap_or_else(|e| panic!("Failed to acquire build lock {}: {e}", lock_path.display()));

    // For GPU builds, ensure submodules (blst, sppark) are initialized and blst is compiled
    if use_gpu {
        ensure_gpu_submodules_initialized(&pil2_stark_path);
        ensure_blst_compiled(&pil2_stark_path);
    }

    let tracked_files = find_tracked_files(&pil2_stark_path);
    for file in &tracked_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    println!("cargo:rerun-if-changed={}", lib_file.display());

    // Detect if Makefile changed since last build. Compiler flag edits (e.g.
    // toggling -D__AVX512__) aren't tracked by make's .d files, so a flag flip
    // would otherwise leave stale objects linked against the new library.
    let makefile_path = pil2_stark_path.join("Makefile");
    let makefile_stamp_path = library_folder.join(".makefile_stamp");
    let current_makefile = fs::read(&makefile_path).ok();
    let stored_makefile = fs::read(&makefile_stamp_path).ok();
    let makefile_changed = current_makefile.is_some() && current_makefile != stored_makefile;

    // Staleness gate: probe everything that could invalidate the library and
    // invoke make only when one of the probes fires.
    let simd_changed = cfg!(target_os = "linux")
        && fs::read_to_string(pil2_stark_path.join(".simd_stamp"))
            .map(|s| s.trim() != host_simd_level())
            .unwrap_or(true);

    // Raw CUDA env vars consumed by the Makefile's arch resolution; any change
    // must reach make,
    let cuda_env = format!(
        "CUDA_ARCHS={};CUDA_ARCH={};CUDA_GENCODE_FLAGS={}",
        env::var("CUDA_ARCHS").unwrap_or_default(),
        env::var("CUDA_ARCH").unwrap_or_default(),
        env::var("CUDA_GENCODE_FLAGS").unwrap_or_default()
    );
    let cuda_env_stamp_path = library_folder.join(".cuda_env_stamp");
    let cuda_env_changed =
        use_gpu && fs::read_to_string(&cuda_env_stamp_path).map(|s| s.trim() != cuda_env).unwrap_or(true);

    // Under auto-detect (no CUDA env set), check that the GPU visible on this
    // machine matches an arch the last build was compiled for.
    let auto_detect = cuda_env == "CUDA_ARCHS=;CUDA_ARCH=;CUDA_GENCODE_FLAGS=";
    let host_arch = if use_gpu && auto_detect { host_gpu_arch() } else { None };

    let gpu_changed = use_gpu && auto_detect && {
        let stamp = fs::read_to_string(pil2_stark_path.join(".cuda_arch_stamp")).unwrap_or_default();
        match &host_arch {
            // Concrete arch detected: rebuild iff the last build doesn't cover it.
            Some(arch) => !stamp.contains(&format!("code=sm_{arch}")),
            // No arch detected: the Makefile falls back to the multi-arch build.
            // Rebuild only if the previous build was single-arch (or absent), so
            // GPU-less hosts settle on the fallback instead of rebuilding forever.
            None => stamp.matches("code=sm_").count() <= 1,
        }
    };

    let lib_mtime = fs::metadata(&lib_file).and_then(|m| m.modified()).ok();
    let sources_newer = lib_mtime.is_none_or(|lib| newest_mtime(&tracked_files) > lib);

    let make_reason = if !lib_file.exists() {
        Some("library missing")
    } else if makefile_changed {
        Some("Makefile changed")
    } else if simd_changed {
        Some("host SIMD level changed")
    } else if cuda_env_changed {
        Some("CUDA_ARCHS/CUDA_ARCH/CUDA_GENCODE_FLAGS changed")
    } else if gpu_changed {
        Some("host GPU arch changed or unknown")
    } else if sources_newer {
        Some("sources newer than library")
    } else {
        None
    };

    let target = if use_gpu { "starks_lib_gpu" } else { "starks_lib" };
    if let Some(reason) = make_reason {
        // Clean build when the Makefile itself changes (CUDA arch / SIMD
        // changes are reconciled by the Makefile's own stamps)
        if makefile_changed {
            eprintln!("Makefile changed — running clean rebuild...");
            // Variant-scoped: a full `make clean` would delete the other
            // variant's build dirs, which may belong to a build that just
            // finished or (without the lock) one still in flight.
            run_command("make", &[if use_gpu { "clean_gpu" } else { "clean_cpu" }], &pil2_stark_path);
        }
        eprintln!("Running make -j {target} ({reason})...");
        run_command("make", &["-j", target], &pil2_stark_path);

        // Write stamps after make succeeds (make creates the output directory).
        if let Some(content) = &current_makefile {
            if let Err(e) = fs::write(&makefile_stamp_path, content) {
                eprintln!(
                    "Warning: failed to write Makefile stamp {:?}: {e} — next build will recompile",
                    makefile_stamp_path
                );
            }
        }
        if use_gpu {
            if let Err(e) = fs::write(&cuda_env_stamp_path, &cuda_env) {
                eprintln!(
                    "Warning: failed to write CUDA env stamp {:?}: {e} — next build will recompile",
                    cuda_env_stamp_path
                );
            }
        }
    } else {
        eprintln!("starks library up to date — skipping make");
    }
    // Absolute path to the library
    let abs_lib_path = library_folder.canonicalize().unwrap_or_else(|_| library_folder.clone());

    if !lib_file.exists() {
        if use_gpu {
            panic!("`libstarksgpu.a` was not found at {}", lib_file.display());
        } else {
            panic!("`libstarks.a` was not found at {}", lib_file.display());
        }
    }

    // Add platform-specific library search paths
    if cfg!(target_os = "macos") {
        // Get Homebrew prefix for macOS
        let homebrew_prefix = Command::new("brew")
            .arg("--prefix")
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|_| "/opt/homebrew".to_string()); // Default for Apple Silicon

        println!("cargo:rustc-link-search=native={homebrew_prefix}/lib");
        println!("cargo:rustc-link-search=native={homebrew_prefix}/opt/libomp/lib");
        println!("cargo:rustc-link-search=native={homebrew_prefix}/opt/libsodium/lib");
        println!("cargo:rustc-link-search=native={homebrew_prefix}/opt/gmp/lib");
        println!("cargo:rustc-link-search=native={homebrew_prefix}/opt/openssl/lib");

        // Also add system paths
        println!("cargo:rustc-link-search=native=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib");
    } else if cfg!(target_os = "linux") {
        // Standard Linux library paths
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }

    // Link the static library
    println!("cargo:rustc-link-search=native={}", abs_lib_path.display());
    println!("cargo:rustc-link-lib=static={library_name}");
    if use_gpu {
        // Add the CUDA library path
        let cuda_path = "/usr/local/cuda/lib64"; // Adjust this path if necessary
        println!("cargo:rustc-link-search=native={cuda_path}");
        println!("cargo:rustc-link-lib=static=cudart_static"); // Link the CUDA runtime library statically
                                                               // cudart_static requires additional system libraries
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=rt");

        // Add the blst library for GPU MSM
        let blst_path = pil2_stark_path.join("external/blst");
        let blst_lib_path = blst_path.canonicalize().unwrap_or_else(|_| blst_path.clone());
        println!("cargo:rustc-link-search=native={}", blst_lib_path.display());
        println!("cargo:rustc-link-lib=static=blst");
    }

    // Link required libraries with platform-specific handling
    if cfg!(target_os = "macos") {
        // macOS library linking (matches Makefile LDFLAGS)
        for lib in &["sodium", "pthread", "gmp", "gmpxx", "c++", "omp"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    } else {
        // Linux library linking
        for lib in &["sodium", "pthread", "gmp", "stdc++", "gmpxx", "crypto", "iomp5"] {
            println!("cargo:rustc-link-lib={lib}");
        }
        // libstarks.a is always compiled with -D__USE_MPI_RMA__ on Linux, so link MPI
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openmpi/lib");
        println!("cargo:rustc-link-lib=mpi");
    }
}

/// SIMD level of the host CPU, mirroring the Makefile's AVX detection
fn host_simd_level() -> &'static str {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    if cpuinfo.contains("avx512") {
        "avx512"
    } else if cpuinfo.contains("avx2") {
        "avx2"
    } else {
        "none"
    }
}

/// Compute capability of the first visible GPU as an sm number (e.g. "120"),
/// or None when nvidia-smi can't see a GPU. Mirrors configure.sh's probe.
fn host_gpu_arch() -> Option<String> {
    // nvidia-smi is the same probe configure.sh uses — gate and authority agree.
    let out = Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let arch: String = stdout.lines().next()?.chars().filter(char::is_ascii_digit).collect();
    (!arch.is_empty()).then_some(arch)
}

/// Newest modification time among the given files (UNIX_EPOCH if none readable).
fn newest_mtime(files: &[PathBuf]) -> SystemTime {
    files
        .iter()
        .filter_map(|f| fs::metadata(f).ok().and_then(|m| m.modified().ok()))
        .max()
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Runs an external command and checks for errors
fn run_command(cmd: &str, args: &[&str], dir: &Path) {
    let status = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .status()
        .unwrap_or_else(|e| panic!("Failed to execute `{cmd}`: {e}"));

    if !status.success() {
        panic!("Command `{}` failed with exit code {:?}", cmd, status.code());
    }
}

/// Recursively finds all files in `pil2-stark`, skipping build output directories.
fn find_tracked_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
        if matches!(name, "build" | "build-gpu" | "build_gpu" | "lib" | "lib-gpu" | ".vscode" | ".git") {
            return files;
        }
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_tracked_files(&path));
            } else {
                // Skip build-generated files: .mk (make includes), .d (dependency
                // files), and the Makefile's staleness stamps
                let ext = path.extension().and_then(|e| e.to_str());
                if matches!(ext, Some("mk" | "d")) {
                    continue;
                }
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(name, ".simd_stamp" | ".cuda_arch_stamp" | ".build_lock") {
                    continue;
                }
                files.push(path);
            }
        }
    }
    files
}

/// Ensures GPU-required submodules (blst and sppark) are initialized
fn ensure_gpu_submodules_initialized(pil2_stark_path: &Path) {
    let blst_path = pil2_stark_path.join("external/blst");
    let sppark_path = pil2_stark_path.join("external/sppark");

    if !is_submodule_initialized(&blst_path) || !is_submodule_initialized(&sppark_path) {
        eprintln!("GPU submodules not fully initialized, running git submodule update...");
        let workspace_root = pil2_stark_path.parent().unwrap_or(pil2_stark_path);
        run_command("git", &["submodule", "update", "--init", "--recursive"], workspace_root);
    }
}

/// Ensures the blst library is compiled for GPU builds
fn ensure_blst_compiled(pil2_stark_path: &Path) {
    let blst_path = pil2_stark_path.join("external/blst");
    let blst_lib = blst_path.join("libblst.a");

    println!("cargo:rerun-if-changed={}", blst_lib.display());

    if blst_lib.exists() {
        eprintln!("blst library already exists at {}", blst_lib.display());
        return;
    }

    eprintln!("blst library not found at {}, compiling...", blst_lib.display());

    let build_script = blst_path.join("build.sh");

    // Track blst build script and source files for changes
    println!("cargo:rerun-if-changed={}", build_script.display());
    println!("cargo:rerun-if-changed={}", blst_path.join("src").display());
    println!("cargo:rerun-if-changed={}", blst_path.join("build").display());
    if !build_script.exists() {
        panic!("blst build.sh not found at {}. Submodule init may have failed.", build_script.display());
    }

    // Run the blst build script
    let status = Command::new("sh")
        .arg("build.sh")
        .current_dir(&blst_path)
        .status()
        .unwrap_or_else(|e| panic!("Failed to execute blst build.sh: {e}"));

    if !status.success() {
        panic!("blst build.sh failed with exit code {:?}", status.code());
    }

    // Verify the library was created
    if !blst_lib.exists() {
        panic!("blst compilation completed but libblst.a was not created at {}", blst_lib.display());
    }

    eprintln!("blst library successfully compiled at {}", blst_lib.display());
}

/// Checks if a git submodule is initialized (has .git file or directory with content)
fn is_submodule_initialized(path: &Path) -> bool {
    // Initialized submodules have a .git file (not directory) pointing to parent's .git/modules/
    let git_path = path.join(".git");
    if git_path.exists() {
        return true;
    }
    // Fallback: check if directory exists and is not empty
    if let Ok(mut entries) = fs::read_dir(path) {
        return entries.next().is_some();
    }
    false
}
