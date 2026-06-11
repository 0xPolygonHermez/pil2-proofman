use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");

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

    // For GPU builds, ensure submodules (blst, sppark) are initialized and blst is compiled
    if use_gpu {
        ensure_gpu_submodules_initialized(&pil2_stark_path);
        ensure_blst_compiled(&pil2_stark_path);
    }

    // gencode_flags: None = auto-detect (delegate to Makefile configure.sh), Some = explicit archs
    println!("cargo:rerun-if-changed={}", pil2_stark_path.join("cuda-major-archs.txt").display());
    let gencode_flags: Option<String> = if use_gpu {
        match parse_cuda_archs(&pil2_stark_path) {
            None => {
                eprintln!("CUDA_ARCHS not set — auto-detecting GPU arch from host");
                None
            }
            Some(archs) => {
                let flags = cuda_gencode_flags(&archs);
                eprintln!("CUDA gencode flags: {}", flags);
                Some(flags)
            }
        }
    } else {
        None
    };

    // Detect if CUDA_ARCHS changed since last build.
    // Stamp stores "auto" for host-detected builds, or the gencode flags string for explicit builds.
    let archs_stamp_path = library_folder.join(".cuda_archs_stamp");
    let stamp_content = gencode_flags.as_deref().unwrap_or("auto");
    let archs_changed = if use_gpu {
        fs::read_to_string(&archs_stamp_path).map(|s| s.trim() != stamp_content).unwrap_or(true)
    } else {
        false
    };

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

    // Clean build when CUDA architecture flags change or the Makefile itself changes
    if archs_changed {
        eprintln!("CUDA_ARCHS changed — running clean rebuild...");
        run_command("make", &["clean"], &pil2_stark_path);
    } else if makefile_changed {
        eprintln!("Makefile changed — running clean rebuild...");
        run_command("make", &["clean"], &pil2_stark_path);
    }

    // Call make to build the library, passing gencode flags if set
    let target = if use_gpu { "starks_lib_gpu" } else { "starks_lib" };
    eprintln!("Running make -j {target}...");
    if use_gpu {
        match &gencode_flags {
            Some(flags) => {
                let gencode_arg = format!("CUDA_GENCODE_FLAGS={}", flags);
                run_command("make", &["-j", &gencode_arg, target], &pil2_stark_path);
            }
            None => run_command("make", &["-j", target], &pil2_stark_path),
        }
    } else {
        run_command("make", &["-j", target], &pil2_stark_path);
    }

    // Write stamps after make succeeds (make creates the output directory).
    if use_gpu {
        if let Err(e) = fs::write(&archs_stamp_path, stamp_content) {
            eprintln!(
                "Warning: failed to write CUDA arch stamp {:?}: {e} — next build will recompile",
                archs_stamp_path
            );
        }
    }
    if let Some(content) = &current_makefile {
        if let Err(e) = fs::write(&makefile_stamp_path, content) {
            eprintln!(
                "Warning: failed to write Makefile stamp {:?}: {e} — next build will recompile",
                makefile_stamp_path
            );
        }
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

fn parse_cuda_archs(pil2_stark_path: &Path) -> Option<Vec<u32>> {
    match env::var("CUDA_ARCHS") {
        Err(_) => None, // Not set → auto-detect via Makefile configure.sh
        Ok(val) if val.trim().eq_ignore_ascii_case("major") => {
            // All major architectures since Ampere: Ampere/Ada/Hopper/Blackwell-DC/Blackwell-consumer
            // Note: sm_100 = B100/B200/GB200 (datacenter Blackwell); sm_120 = RTX 5090/5080/5070/5060 (consumer Blackwell)
            // Note: sm_100 and sm_120 are NOT cross-compatible (sm_100 has TMEM hardware sm_120 lacks)
            // Canonical list lives in pil2-stark/cuda-major-archs.txt — single source
            // of truth shared with the Makefile's auto-detect fallback.
            let path = pil2_stark_path.join("cuda-major-archs.txt");
            let content = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("CUDA_ARCHS=major: cannot read {}: {e}", path.display()));
            let mut archs: Vec<u32> = content
                .lines()
                .map(|line| line.split('#').next().unwrap_or(""))
                .flat_map(str::split_whitespace)
                .map(|s| {
                    s.parse::<u32>().unwrap_or_else(|_| {
                        panic!("invalid arch {:?} in {} — expected integers", s, path.display())
                    })
                })
                .collect();
            if archs.is_empty() {
                panic!("{} contains no architectures", path.display());
            }
            archs.sort_unstable();
            archs.dedup();
            Some(archs)
        }
        Ok(val) => {
            let mut archs = Vec::new();
            for token in val.split(',') {
                let s = token.trim();
                match s.parse::<u32>() {
                    Ok(n) => archs.push(n),
                    Err(_) => panic!(
                        "CUDA_ARCHS contains invalid entry {:?} — expected integers (e.g. '89', '89,90', or 'major')",
                        s
                    ),
                }
            }
            if archs.is_empty() {
                panic!("CUDA_ARCHS is set but empty — expected integers (e.g. '89', '89,90', or 'major')");
            }
            // Normalize: sort + dedup so stamp comparison is order/duplicate-independent
            archs.sort_unstable();
            archs.dedup();
            Some(archs)
        }
    }
}

fn cuda_gencode_flags(archs: &[u32]) -> String {
    let mut flags = Vec::new();
    for &arch in archs {
        flags.push(format!("-gencode arch=compute_{arch},code=sm_{arch}"));
    }
    // sm_100-119 are separate, incompatible lineages — embed PTX for the highest arch in each lineage present.
    let max_dc_blackwell = archs.iter().filter(|&&a| (100..120).contains(&a)).max().copied();
    let max_other = archs.iter().filter(|&&a| !(100..120).contains(&a)).max().copied();
    match (max_dc_blackwell, max_other) {
        (Some(dc), Some(other)) => {
            flags.push(format!("-gencode arch=compute_{dc},code=compute_{dc}"));
            flags.push(format!("-gencode arch=compute_{other},code=compute_{other}"));
        }
        _ => {
            let max_arch = *archs.iter().max().expect("archs list is empty");
            flags.push(format!("-gencode arch=compute_{max_arch},code=compute_{max_arch}"));
        }
    }
    flags.join(" ")
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
                // Skip build-generated files: .mk (make includes), .d (dependency files)
                let ext = path.extension().and_then(|e| e.to_str());
                if matches!(ext, Some("mk" | "d")) {
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
