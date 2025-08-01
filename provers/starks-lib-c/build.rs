use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

fn main() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-cfg=feature=\"no_lib_link\"");
        return;
    }

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

    // Check if the C++ library exists before recompiling
    if !lib_file.exists() {
        if cfg!(feature = "gpu") {
            eprintln!("`libstarksgpu.a` not found! Compiling...");
            run_command("make", &["clean"], &pil2_stark_path);
            run_command("make", &["-j", "starks_lib_gpu"], &pil2_stark_path);
        } else {
            eprintln!("`libstarks.a` not found! Compiling...");
            run_command("make", &["clean"], &pil2_stark_path);
            run_command("make", &["-j", "starks_lib"], &pil2_stark_path);
        }
    } else {
        println!("C++ library already compiled, skipping rebuild.");
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
    track_file_changes(&pil2_stark_path);

    // Link the static library
    println!("cargo:rustc-link-search=native={}", abs_lib_path.display());
    println!("cargo:rustc-link-lib=static={library_name}");
    if cfg!(feature = "gpu") {
        // Add the CUDA library path
        let cuda_path = "/usr/local/cuda/lib64"; // Adjust this path if necessary
        println!("cargo:rustc-link-search=native={cuda_path}");
        println!("cargo:rustc-link-lib=dylib=cudart"); // Link the CUDA runtime library

        // Specify the CUDA architecture
        println!("cargo:rustc-env=CUDA_ARCH=sm_75"); // Adjust the architecture as needed
    }

    // Link required libraries
    for lib in &["sodium", "pthread", "gmp", "stdc++", "gmpxx", "crypto", "iomp5"] {
        println!("cargo:rustc-link-lib={lib}");
    }
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
fn track_file_changes(pil2_stark_path: &Path) {
    let source_files = find_source_files(pil2_stark_path);
    let lib_file: PathBuf = if cfg!(feature = "gpu") {
        pil2_stark_path.join("lib-gpu/lib-gpu/libstarksgpu.a")
    } else {
        pil2_stark_path.join("lib/libstarks.a")
    };

    // Print tracked files for debugging
    eprintln!("Tracking {} source files:", source_files.len());
    for file in &source_files {
        eprintln!(" - {}", file.display());
        println!("cargo:rerun-if-changed={}", file.display());
    }

    // If any C++ source file changed, force a rebuild
    if source_files_have_changed(&source_files, &lib_file) {
        eprintln!("Changes detected! Running `make clean` and recompiling...");
        run_command("make", &["clean"], pil2_stark_path);
        if cfg!(feature = "gpu") {
            run_command("make", &["-j", "starks_lib_gpu"], pil2_stark_path);
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
                    if (ext == "cpp" || ext == "h" || ext == "hpp" || ext == "cu" || ext == "cuh" || ext == "asm")
                        && path.file_name() != Some(std::ffi::OsStr::new("starks_lib.h"))
                    {
                        source_files.push(path);
                    }
                } else if (ext == "cpp" || ext == "h" || ext == "hpp" || ext == "asm")
                    && path.file_name() != Some(std::ffi::OsStr::new("starks_lib.h"))
                {
                    source_files.push(path);
                }
            }
        }
    }
    source_files
}
