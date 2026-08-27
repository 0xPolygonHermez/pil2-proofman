use std::path::Path;

use anyhow::Result;

use crate::commands::recursive_setup::{resolve_circom_exec, resolve_path_env};
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::recursive_test::gen_recursive_test_setup;

/// Options for the `setup-recursive-test` subcommand.
pub struct SetupRecursiveTestOptions {
    pub build_dir: String,
    pub circom_path: String,
    pub circom_name: String,
    pub setup_type: String,
    pub hash: String,
    pub blake3_lanes: Option<usize>,
}

/// Resolve the per-hash-family circom fixture.
///
/// The test-recursive fixtures (test.circom + test.verifier.circom + proof.bin)
/// live in per-hash subfolders (`poseidon1/`, `poseidon2/`). If the directory of
/// the given `circom_path` has a subfolder named after the lowercased hash that
/// contains a same-named circom file, prefer that one. This lets a single
/// `-c ./examples/test-recursive/test.circom` auto-select the right fixture from
/// `--hash`, while still honouring an explicit subfolder path if one is passed.
fn resolve_hash_circom_path(circom_path: &str, hash: &str) -> String {
    let path = Path::new(circom_path);
    let (Some(parent), Some(file_name)) = (path.parent(), path.file_name()) else {
        return circom_path.to_string();
    };
    let candidate = parent.join(hash.to_lowercase()).join(file_name);
    if candidate.is_file() {
        return candidate.to_string_lossy().into_owned();
    }
    circom_path.to_string()
}

/// Run the recursive test setup.
///
/// Ports `main_setup_recursive.js` behaviour for the test-recursive CI job.
pub fn run_setup_recursive_test(opts: &SetupRecursiveTestOptions) -> Result<()> {
    proofman_starks_lib_c::set_hash_family_c(&opts.hash);

    let circuits_gl_path =
        resolve_path_env("CIRCUITS_GL_PATH", "setup/stark-recurser/stark2circom/circom_verifier/circuits.gl");
    let recurser_circuits_path =
        resolve_path_env("RECURSER_CIRCUITS_PATH", "setup/stark-recurser/stark2circom/circom_verifier/helper_circuits");
    let recurser_pil_path = resolve_path_env("RECURSER_PIL_PATH", "setup/stark-recurser/plonk2pil/pil");
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");

    let circom_exec = resolve_circom_exec(&circom_helpers_dir);
    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    // Auto-select the per-hash-family fixture (poseidon1/ or poseidon2/) when one
    // exists beside the given circom path; otherwise use the path as-is.
    let circom_path = resolve_hash_circom_path(&opts.circom_path, &opts.hash);
    tracing::info!("Using circom fixture: {} (hash={})", circom_path, opts.hash);

    gen_recursive_test_setup(
        &opts.build_dir,
        &circom_path,
        &opts.circom_name,
        &opts.setup_type,
        &opts.hash,
        &circom_exec,
        &circuits_gl_path,
        &recurser_circuits_path,
        &recurser_pil_path,
        &circom_helpers_dir,
        &witness_tracker,
        opts.blake3_lanes,
    )
}
