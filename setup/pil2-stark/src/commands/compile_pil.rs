//! Compile-pil command: invoke the JS `pil2com` compiler on a `.pil` source.
//!
//! Thin wrapper around the existing executable resolver in
//! `proving_key::recursive` so users can compile a pilout directly from the
//! `proofman-setup` CLI without going through the recursive setup pipeline.

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::proving_key::recursive::ensure_pil2com_exec;

/// Options for the `compile-pil` subcommand.
pub struct CompilePilOptions {
    /// Path to the entry `.pil` file (e.g. `pil/zisk.pil`).
    pub pil_path: String,
    /// Output `.pilout` path.
    pub output_path: String,
    /// `-I` include search paths (one or more directories).
    pub include_paths: Vec<String>,
    /// Optional `-u` directory for fixed columns.
    pub fixed_dir: Option<String>,
    /// Pass `-O fixed-to-file` to write fixed columns to disk.
    pub fixed_to_file: bool,
    /// Pass `-O no-proto-fixed-data` to omit fixed-column values from the
    /// protobuf-encoded pilout. Combined with `fixed_to_file` this keeps
    /// fixed data off the V8 heap during proto encoding — a major memory
    /// win on PILs the size of zisk.pil.
    pub no_proto_fixed_data: bool,
}

pub fn run_compile_pil(opts: &CompilePilOptions) -> Result<()> {
    tracing::info!("Compiling PIL: {}", opts.pil_path);

    let pil2com_exec = ensure_pil2com_exec().context(
        "Cannot find pil2com and automatic `npm install` did not produce it. Install Node.js/npm \
         and run `npm install` in the pil2-proofman repo root",
    )?;
    tracing::info!("Using pil2com: {}", pil2com_exec);

    if opts.include_paths.is_empty() {
        bail!("at least one include path (-I) is required");
    }
    let include_arg = opts.include_paths.join(",");

    let mut cmd = std::process::Command::new(&pil2com_exec);
    cmd.arg(&opts.pil_path).arg("-I").arg(&include_arg).arg("-o").arg(&opts.output_path);

    if let Some(fixed_dir) = &opts.fixed_dir {
        cmd.arg("-u").arg(fixed_dir);
    }
    if opts.fixed_to_file {
        cmd.arg("-O").arg("fixed-to-file");
    }
    if opts.no_proto_fixed_data {
        cmd.arg("-O").arg("no-proto-fixed-data");
    }

    // pil2com is a wrapper script, so Node CLI flags can't be passed directly.
    // The default Node heap (~4 GB) overflows on large PILs; set the equivalent
    // `NODE_OPTIONS` env var so the spawned `node` process picks it up. Skip
    // when the caller has already set NODE_OPTIONS (their value takes priority,
    // inherited automatically).
    if std::env::var_os("NODE_OPTIONS").is_none() {
        cmd.env("NODE_OPTIONS", "--max-old-space-size=65536");
    }

    let status = cmd
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .with_context(|| format!("Failed to execute pil2com: {}", pil2com_exec))?;

    if !status.success() {
        bail!("pil2com exited with status {} for {}", status.code().unwrap_or(-1), opts.pil_path);
    }
    if !Path::new(&opts.output_path).exists() {
        bail!("pil2com did not produce output file: {}", opts.output_path);
    }

    Ok(())
}
