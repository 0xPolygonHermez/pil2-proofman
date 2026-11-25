//! Setup-compressed-final command: run only `gen_compressed_final_setup` on top
//! of an existing `provingKey/<name>/vadcop_final/`. Useful for iterating on the
//! compressed_final stage without re-running the full recursive pipeline.

use anyhow::{Context, Result};

use crate::commands::common::{witness_tracker_from_env, RecursiveKey};
use crate::proving_key::compressed_final::{gen_compressed_final_setup, CompressedFinalConfig};
use crate::proving_key::pipeline::Tools;

pub struct SetupCompressedFinalOptions {
    /// Build directory containing `provingKey/<name>/vadcop_final/`.
    pub build_dir: String,
    /// Low-degree test of the compressed proof; `None` is FRI.
    pub low_degree_test: Option<crate::types::stark_struct::LowDegreeTestKind>,
}

pub fn run_setup_compressed_final(opts: &SetupCompressedFinalOptions) -> Result<()> {
    let key = RecursiveKey::load(&opts.build_dir)?;
    let vf = key.vadcop_final()?;
    let tools = Tools::compressed_final_from_env();
    let witness_tracker = witness_tracker_from_env();
    let config = CompressedFinalConfig {
        build_dir: &opts.build_dir,
        hash: key.hash,
        name: &key.name,
        const_root: &vf.const_root.map(|v| v.to_string()),
        stark_info: &vf.stark_info,
        verifier_info: &vf.verifier_info,
        tools: &tools,
        low_degree_test: opts.low_degree_test,
    };

    tracing::info!("Running compressed final setup (standalone) for '{}'", key.name);
    gen_compressed_final_setup(&config, &witness_tracker).context("Compressed final setup failed")?;
    witness_tracker.await_all()?;
    tracing::info!("Compressed final setup complete");
    Ok(())
}
