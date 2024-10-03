use proofman_cli::commands::pil_helpers::PilHelpersCmd;
use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let root_path =
        std::fs::canonicalize(std::env::current_dir().expect("Failed to get current directory").join("../../../../"))
            .expect("Failed to canonicalize root path");

    let pil_file = root_path.join("test/std/permutation/permutation.pil");
    let build_dir = root_path.join("test/std/permutation/build");
    let pilout_file = build_dir.join("permutation.pilout");
    let pil_helpers_dir = root_path.join("test/std/permutation/src/pil_helpers");

    // Always rerun if the pil changes
    println!("cargo:rerun-if-changed={}", pil_file.display());

    // Check if the "build" directory exists
    if !build_dir.exists() {
        println!("Directory build does not exist, generating...");
        std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");
    }

    println!("cargo:rerun-if-changed={}", build_dir.display());
    println!("cargo:rerun-if-changed={}", pil_helpers_dir.display());

    // Compile the pil file
    let pil_compiler_path = if let Ok(path) = env::var("PIL_COMPILER") {
        // If PIL_COMPILER is set, use its value
        fs::canonicalize(Path::new(&path).parent().unwrap_or_else(|| Path::new("."))).expect("Failed")
    } else {
        // Fallback if PIL_COMPILER is not set
        root_path.join("../../pil2-compiler")
    };

    let pil_compilation = std::process::Command::new("node")
        .arg(pil_compiler_path.join("src/pil.js"))
        .arg("-I")
        .arg(root_path.join("lib/std/pil"))
        .arg(pil_file.clone())
        .arg("-o")
        .arg(pilout_file.clone())
        .status()
        .expect("Failed to execute pil compilation command");

    if !pil_compilation.success() {
        eprintln!("Error: Pil file compilation failed.");
        std::process::exit(1);
    }

    // Generate pil_helpers
    let pil_helpers = PilHelpersCmd {
        pilout: pilout_file.clone(),
        path: root_path.join("test/std/permutation/rs/src"),
        overide: true,
        verbose: 0,
    };

    if let Err(e) = pil_helpers.run() {
        eprintln!("Error: Failed to generate pil_helpers: {:?}", e);
        std::process::exit(1);
    }

    // Generate proving key
    let pil2_proofman_js_path = if let Ok(path) = env::var("PIL_PROOFMAN_JS") {
        // If PIL_PROOFMAN_JS is set, use its value
        fs::canonicalize(Path::new(&path).parent().unwrap_or_else(|| Path::new("."))).expect("Failed")
    } else {
        // Fallback if PIL_PROOFMAN_JS is not set
        root_path.join("../../pil2-proofman-js")
    };
    let proving_key_generation = std::process::Command::new("node")
        .arg(pil2_proofman_js_path.join("src/main_setup.js"))
        .arg("-a")
        .arg(pilout_file.clone())
        .arg("-b")
        .arg(build_dir.clone())
        .status()
        .expect("Failed to execute proving key generation command");

    if !proving_key_generation.success() {
        eprintln!("Error: Proving key generation failed.");
        std::process::exit(1);
    }

    println!("Build completed successfully.");
}
