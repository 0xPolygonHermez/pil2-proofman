use anyhow::Result;

use crate::commands::recursive_setup::{resolve_circom_exec, resolve_path_env};
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::recursive_test::gen_recursive_test_setup;

/// Options for the `setup-recursive-test` subcommand.
pub struct SetupRecursiveTestOptions {
    pub build_dir: String,
    pub circom_path: String,
    pub circom_name: String,
    pub std_pil_path: String,
    pub setup_type: String,
}

/// Run the recursive test setup.
///
/// Ports `main_setup_recursive.js` behaviour for the test-recursive CI job.
pub fn run_setup_recursive_test(opts: &SetupRecursiveTestOptions) -> Result<()> {
    let circuits_gl_path =
        resolve_path_env("CIRCUITS_GL_PATH", "node_modules/stark-recurser/src/pil2circom/circuits.gl");
    let recurser_circuits_path =
        resolve_path_env("RECURSER_CIRCUITS_PATH", "node_modules/stark-recurser/src/vadcop/helpers/circuits");
    let recurser_pil_path = resolve_path_env("RECURSER_PIL_PATH", "setup/stark-recurser/plonk2pil/pil");
    let circom_helpers_dir = resolve_path_env("CIRCOM_HELPERS_DIR", "setup/circom");
    let goldilocks_src_dir = resolve_path_env("GOLDILOCKS_SRC_DIR", "pil2-stark/src/goldilocks/src");

    let circom_exec = resolve_circom_exec(&circom_helpers_dir);
    let witness_tracker = WitnessTracker::with_goldilocks_src(&goldilocks_src_dir);

    gen_recursive_test_setup(
        &opts.build_dir,
        &opts.circom_path,
        &opts.circom_name,
        &opts.std_pil_path,
        &opts.setup_type,
        &circom_exec,
        &circuits_gl_path,
        &recurser_circuits_path,
        &recurser_pil_path,
        &circom_helpers_dir,
        &witness_tracker,
    )
}
