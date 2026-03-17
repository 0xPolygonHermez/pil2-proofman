use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");

    // **Check if the `no_lib_link` feature is enabled**
    if env::var("CARGO_FEATURE_NO_LIB_LINK").is_ok() {
        println!("Skipping linking because `no_lib_link` feature is enabled.");
        return;
    }

    // Paths
    let pil2_stark_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pil2-stark");
    let library_folder =
        if cfg!(feature = "gpu") { pil2_stark_path.join("lib-gpu") } else { pil2_stark_path.join("lib") };
    let library_name = if cfg!(feature = "gpu") { "starksgpu" } else { "starks" };
    let lib_file = library_folder.join(format!("lib{library_name}.a"));

    if !pil2_stark_path.exists() {
        panic!("Missing `pil2-stark` submodule! Run `git submodule update --init --recursive`");
    }

    // Ensure `git submodule update --init --recursive` runs only if needed
    if !pil2_stark_path.join(".git").exists() {
        run_command("git", &["submodule", "init"], &pil2_stark_path);
        run_command("git", &["submodule", "update", "--recursive"], &pil2_stark_path);
    }

    // For GPU builds, ensure submodules are initialized and blst is compiled
    if cfg!(feature = "gpu") {
        ensure_gpu_submodules_initialized(&pil2_stark_path);
        ensure_blst_compiled(&pil2_stark_path);
    }

    // gencode_flags: None = auto-detect (delegate to Makefile configure.sh), Some = explicit archs
    let gencode_flags: Option<String> = if cfg!(feature = "gpu") {
        match parse_cuda_archs() {
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
    let archs_changed = if cfg!(feature = "gpu") {
        fs::read_to_string(&archs_stamp_path).map(|s| s.trim() != stamp_content).unwrap_or(true)
    } else {
        false
    };

    // Check if the `no_cpp_compilation` feature is enabled
    if cfg!(feature = "no_cpp_compilation") {
        println!("Skipping C++ compilation because `no_cpp_compilation` feature is enabled.");
        if !lib_file.exists() {
            eprintln!("Warning: Library `{}` not found. Make sure to compile it manually.", lib_file.display());
            eprintln!(
                "Run: cd pil2-stark && make {}",
                if cfg!(feature = "gpu") { "starks_lib_gpu" } else { "starks_lib" }
            );
        }
    } else {
        // Rebuild if library is missing or CUDA_ARCHS changed since last build
        if !lib_file.exists() || archs_changed {
            if cfg!(feature = "gpu") {
                eprintln!("`libstarksgpu.a` missing or CUDA_ARCHS changed — recompiling...");
                run_command("make", &["clean"], &pil2_stark_path);
                match &gencode_flags {
                    Some(flags) => {
                        let gencode_arg = format!("CUDA_GENCODE_FLAGS={}", flags);
                        run_command("make", &["-j", &gencode_arg, "starks_lib_gpu"], &pil2_stark_path);
                    }
                    None => run_command("make", &["-j", "starks_lib_gpu"], &pil2_stark_path),
                }
                if let Err(e) = fs::write(&archs_stamp_path, stamp_content) {
                    eprintln!(
                        "Warning: failed to write CUDA arch stamp {:?}: {e} — next build will recompile",
                        archs_stamp_path
                    );
                }
            } else {
                eprintln!("`libstarks.a` not found! Compiling...");
                run_command("make", &["clean"], &pil2_stark_path);
                run_command("make", &["-j", "starks_lib"], &pil2_stark_path);
            }
        } else {
            println!("C++ library already compiled, skipping rebuild.");
        }
    }

    // Absolute path to the library
    let abs_lib_path = library_folder.canonicalize().unwrap_or_else(|_| library_folder.clone());

    if !lib_file.exists() {
        if cfg!(feature = "gpu") {
            panic!("`libstarksgpu.a` was not found at {}", lib_file.display());
        } else {
            panic!("`libstarks.a` was not found at {}", lib_file.display());
        }
    }

    // Ensure Rust triggers a rebuild if the C++ source code changes
    // Skip this if no_cpp_compilation is enabled
    if !cfg!(feature = "no_cpp_compilation") {
        track_file_changes(&pil2_stark_path, gencode_flags.as_deref(), &archs_stamp_path);
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
    if cfg!(feature = "gpu") {
        // Add the CUDA library path
        let cuda_path = "/usr/local/cuda/lib64"; // Adjust this path if necessary
        println!("cargo:rustc-link-search=native={cuda_path}");
        println!("cargo:rustc-link-lib=dylib=cudart"); // Link the CUDA runtime library

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
    }
}

fn parse_cuda_archs() -> Option<Vec<u32>> {
    match env::var("CUDA_ARCHS") {
        Err(_) => None, // Not set → auto-detect via Makefile configure.sh
        Ok(val) if val.trim().eq_ignore_ascii_case("major") => {
            // All major architectures since Ampere: Ampere/Ada/Hopper/Blackwell-DC/Blackwell-consumer
            // sm_100 = B100/B200/GB200 (datacenter Blackwell); sm_120 = RTX 5090/5080/5070/5060 (consumer Blackwell)
            // Note: sm_100 and sm_120 are NOT cross-compatible (sm_100 has TMEM hardware sm_120 lacks)
            Some(vec![80, 86, 89, 90, 100, 120])
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

/// Tracks changes in the `pil2-stark` directory to trigger recompilation only when needed
fn track_file_changes(pil2_stark_path: &Path, gencode_flags: Option<&str>, stamp_path: &Path) {
    let source_files = find_source_files(pil2_stark_path);
    let lib_file: PathBuf = if cfg!(feature = "gpu") {
        pil2_stark_path.join("lib-gpu/libstarksgpu.a")
    } else {
        pil2_stark_path.join("lib/libstarks.a")
    };

    for file in &source_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    // If any C++ source file changed, force a rebuild
    if source_files_have_changed(&source_files, &lib_file) {
        eprintln!("Changes detected! Running `make clean` and recompiling...");
        run_command("make", &["clean"], pil2_stark_path);
        if cfg!(feature = "gpu") {
            match gencode_flags {
                Some(flags) => {
                    let gencode_arg = format!("CUDA_GENCODE_FLAGS={}", flags);
                    run_command("make", &["-j", &gencode_arg, "starks_lib_gpu"], pil2_stark_path);
                }
                None => run_command("make", &["-j", "starks_lib_gpu"], pil2_stark_path),
            }
            if let Err(e) = fs::write(stamp_path, gencode_flags.unwrap_or("auto")) {
                eprintln!("Warning: failed to write CUDA arch stamp {:?}: {e} — next build will recompile", stamp_path);
            }
        } else {
            run_command("make", &["-j", "starks_lib"], pil2_stark_path);
        }
    } else {
        println!("No C++ source changes detected, skipping rebuild.");
    }
}

/// Checks if any `.cpp`, `.h`, or `.hpp` file has changed since the last build
fn source_files_have_changed(source_files: &[PathBuf], lib_file: &Path) -> bool {
    let mut modified_files: Vec<PathBuf> = Vec::new();

    // Get the modification time of `libstarks.a`
    let lib_modified_time = match fs::metadata(lib_file) {
        Ok(metadata) => {
            let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
            eprintln!("`{}` last modified: {:?}", lib_file.display(), modified);
            modified
        }
        Err(_) => {
            eprintln!("Library `{}` does not exist, triggering rebuild.", lib_file.display());
            return true; // If `libstarks.a` is missing, we must rebuild.
        }
    };

    // Check if any `.cpp`, `.h`, or `.hpp` file has been modified after `libstarks.a`
    for file in source_files {
        if let Ok(metadata) = fs::metadata(file) {
            if let Ok(modified_time) = metadata.modified() {
                if modified_time > lib_modified_time {
                    modified_files.push(file.clone());
                }
            }
        }
    }

    // Print the list of modified files (if any)
    if !modified_files.is_empty() {
        eprintln!("Modified files detected:");
        for file in &modified_files {
            eprintln!(" - {}", file.display());
        }
        return true;
    }

    false // No changes detected
}

/// Finds all `.cpp`, `.h`, `.hpp`, `.c`, `.cuh` and `.asm` files in `pil2-stark` (recursive search)
fn find_source_files(dir: &Path) -> Vec<PathBuf> {
    let mut source_files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                source_files.extend(find_source_files(&path));
            } else if let Some(ext) = path.extension() {
                if cfg!(feature = "gpu") {
                    if (ext == "c"
                        || ext == "cpp"
                        || ext == "h"
                        || ext == "hpp"
                        || ext == "cu"
                        || ext == "cuh"
                        || ext == "asm")
                        && path.file_name() != Some(std::ffi::OsStr::new("starks_lib_gpu.h"))
                    {
                        source_files.push(path);
                    }
                } else if (ext == "c" || ext == "cpp" || ext == "h" || ext == "hpp" || ext == "asm")
                    && path.file_name() != Some(std::ffi::OsStr::new("starks_lib.h"))
                {
                    source_files.push(path);
                }
            }
        }
    }
    source_files
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
