use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-cfg=feature=\"no_lib_link\"");
        return;
    }

    // Check if the "NO_LIB_LINK" feature is enabled
    if env::var("CARGO_FEATURE_NO_LIB_LINK").is_err() {
        let pil2_stark_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pil2-stark");
        let library_folder = pil2_stark_path.join("lib");
        let library_name = "starks";

        if !pil2_stark_path.exists() {
            panic!("Missing `pil2-stark` submodule! Run `git submodule update --init --recursive`");
        }

        let _ = Command::new("git")
            .args(&["submodule", "init"])
            .current_dir(&pil2_stark_path)
            .status()
            .expect("Failed to initialize submodules");

        let _ = Command::new("git")
            .args(&["submodule", "update", "--recursive"])
            .current_dir(&pil2_stark_path)
            .status()
            .expect("Failed to update submodules");

        let _ = Command::new("make")
            .arg("clean")
            .current_dir(&pil2_stark_path)
            .status()
            .expect("Failed to clean previous build");

        let _ = Command::new("make")
            .args(&["-j", "starks_lib"])
            .current_dir(&pil2_stark_path)
            .status()
            .expect("Failed to compile `starks_lib`");

        let abs_lib_path = library_folder.canonicalize().unwrap_or(library_folder.clone());
        let lib_file = abs_lib_path.join(format!("lib{}.a", library_name));

        if !lib_file.exists() {
            panic!("`libstarks.a` was not found at {}", lib_file.display());
        }

        println!("cargo:rustc-link-search=native={}", abs_lib_path.display());
        println!("cargo:rustc-link-lib=static={}", library_name);

        for lib in &["sodium", "pthread", "gmp", "stdc++", "gmpxx", "crypto", "iomp5"] {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }
}
