//! Builds pil2-fflonk's C++ library and links it.
//!
//! Mirrors `provers/starks-lib-c/build.rs`, minus two things it needs and this
//! does not: there is no CUDA variant, since pil-fflonk had no GPU path to
//! match, and the sources sit in the workspace rather than being carried by a
//! source crate, so there is no read-only checkout to mirror into OUT_DIR.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let cpp = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("../../pil2-fflonk/cpp")
        .canonicalize()
        .expect("pil2-fflonk/cpp not found");

    // Rebuild when a source moves. Build outputs are skipped or the script
    // would re-run on its own products forever.
    for file in sources(&cpp) {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    // ffiasm and the snark-side support files are compiled into the library
    // from pil2-stark, so changes there matter too.
    let stark_src = cpp.join("../../pil2-stark/src");
    for sub in ["bn128/src/ffiasm", "rapidsnark", "XKCP"] {
        println!("cargo:rerun-if-changed={}", stark_src.join(sub).display());
    }

    // Respect cargo's parallelism budget: this make runs inside a build cargo
    // is already scheduling.
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "1".to_string());
    run("make", &["-j", &jobs, "lib"], &cpp);

    let lib_dir = cpp.join("lib");
    if !lib_dir.join("libpilfflonk.a").exists() {
        panic!("`libpilfflonk.a` was not found in {} after a successful make", lib_dir.display());
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=pilfflonk");

    // The library is C++ built with OpenMP, over GMP.
    if cfg!(target_os = "macos") {
        let prefix = Command::new("brew")
            .arg("--prefix")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "/opt/homebrew".to_string());
        for sub in ["lib", "opt/libomp/lib", "opt/gmp/lib"] {
            println!("cargo:rustc-link-search=native={prefix}/{sub}");
        }
        for lib in ["gmp", "c++", "omp"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    } else {
        for path in ["/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"] {
            println!("cargo:rustc-link-search=native={path}");
        }
        for lib in ["gmp", "stdc++", "gomp", "pthread"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

fn run(cmd: &str, args: &[&str], dir: &Path) {
    let status = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .status()
        .unwrap_or_else(|e| panic!("failed to execute `{cmd}` in {}: {e}", dir.display()));

    if !status.success() {
        panic!("`{cmd} {}` failed with {:?}", args.join(" "), status.code());
    }
}

/// Every source under `cpp/`, skipping what the build itself writes.
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if matches!(dir.file_name().and_then(|n| n.to_str()), Some("build" | "lib")) {
        return out;
    }

    let Ok(entries) = fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(sources(&path));
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str());
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if ext == Some("d") || name == "testscpu" {
            continue;
        }
        out.push(path);
    }
    out
}
