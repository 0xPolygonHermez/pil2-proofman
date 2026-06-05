use std::path::PathBuf;

use pil2_stark_setup::commands::compile_pil::{run_compile_pil, CompilePilOptions};
use proofman_cli::commands::pil_helpers::PilHelpersCmd;

const MANIFEST: &str = env!("CARGO_MANIFEST_DIR");
const PIL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../build.pil");
const STD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../lib/std/pil");

fn main() {
    println!("cargo:rerun-if-changed={PIL}");
    println!("cargo:rerun-if-changed={STD}");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let pilout = format!("{out_dir}/range_check.pilout");

    // 1. Compile the PIL.
    run_compile_pil(&CompilePilOptions {
        pil_path: PIL.to_string(),
        output_path: pilout.clone(),
        include_paths: vec![STD.to_string()],
        fixed_dir: None,
        fixed_to_file: false,
        no_proto_fixed_data: false,
    })
    .expect("build.rs: compile-pil failed");

    // 2. Regenerate the pil_helpers bindings.
    PilHelpersCmd {
        pilout: PathBuf::from(&pilout),
        path: PathBuf::from(format!("{MANIFEST}/src")),
        overide: true,
        verbose: 0,
    }
    .run()
    .expect("build.rs: pil-helpers failed");
}
