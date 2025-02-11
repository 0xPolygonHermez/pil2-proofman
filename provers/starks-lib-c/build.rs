use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

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
    let library_folder = pil2_stark_path.join("lib");
    let library_name = "starks";
    let lib_file = library_folder.join(format!("lib{}.a", library_name));

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
        eprintln!("`libstarks.a` not found! Compiling...");
        run_command("make", &["clean"], &pil2_stark_path);
        run_command("make", &["-j", "starks_lib"], &pil2_stark_path);
        run_command("make", &["-j", "bctree"], &pil2_stark_path);
    } else {
        println!("C++ library already compiled, skipping rebuild.");
    }

    // Absolute path to the library
    let abs_lib_path = library_folder.canonicalize().unwrap_or_else(|_| library_folder.clone());

    if !lib_file.exists() {
        panic!("`libstarks.a` was not found at {}", lib_file.display());
    }

    // Ensure Rust triggers a rebuild if the C++ source code changes
    track_cpp_changes(&pil2_stark_path);

    // Link the static library
    println!("cargo:rustc-link-search=native={}", abs_lib_path.display());
    println!("cargo:rustc-link-lib=static={}", library_name);

    // Link required libraries
    for lib in &["sodium", "pthread", "gmp", "stdc++", "gmpxx", "crypto", "iomp5"] {
        println!("cargo:rustc-link-lib={}", lib);
    }
}

/// Runs an external command and checks for errors
fn run_command(cmd: &str, args: &[&str], dir: &Path) {
    let status = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .status()
        .unwrap_or_else(|e| panic!("Failed to execute `{}`: {}", cmd, e));

    if !status.success() {
        panic!("Command `{}` failed with exit code {:?}", cmd, status.code());
    }
}

/// Tracks changes in the `pil2-stark` directory to trigger recompilation only when needed
fn track_cpp_changes(pil2_stark_path: &Path) {
    let cpp_files = find_cpp_files(pil2_stark_path);

    for file in cpp_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
}

/// Finds all `.cpp` and `.h` files in `pil2-stark` (recursive search)
fn find_cpp_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut cpp_files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                cpp_files.extend(find_cpp_files(&path));
            } else if let Some(ext) = path.extension() {
                if ext == "cpp" || ext == "h" {
                    cpp_files.push(path);
                }
            }
        }
    }
    cpp_files
}
