use clap::Parser;
use colored::Colorize;
use proofman_exps_codegen::{generate_all, GenConfig};
use std::path::PathBuf;

/// Generate per-AIR Q-expression CUDA kernels and compile each into a
/// self-contained `<base>.exps.so` next to that AIR's `.bin`. The prover loads
/// it by convention; AIRs without a `.so` stay on the bytecode interpreter.
#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct GenExpsCmd {
    /// provingKey build dir (globbed for each AIR's *.starkinfo.json + *.expressionsinfo.json)
    #[clap(long)]
    pub proving_key: PathBuf,

    /// CUDA arch(s): auto (default; detect host GPU), major, or a list like 89,120 / sm_120
    #[clap(long, default_value = "auto")]
    pub arch: String,

    /// Skip an AIR whose Q has more than N ops (it stays on the interpreter)
    #[clap(long, default_value_t = 60000)]
    pub cap: usize,

    /// Fixed ops/chunk for every AIR; omit to auto-tune the largest no-spill size
    #[clap(long)]
    pub chunk: Option<usize>,

    /// Retain the generated .cu/.o here (default: a temp dir removed on exit).
    /// Must be OUTSIDE pil2-stark/src/starkpil or the pil2-stark Makefile will
    /// try to compile the generated kernels into the main library.
    #[clap(long)]
    pub keep_dir: Option<PathBuf>,

    /// pil2-stark source root (default: resolved relative to this binary's crate)
    #[clap(long)]
    pub stark_src: Option<PathBuf>,

    /// Emit the .cu sources only (no compile/link); the provingKey is untouched.
    /// Requires --keep-dir. For inspection / parity checks.
    #[clap(long)]
    pub emit_only: bool,
}

impl GenExpsCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        tracing::info!("{}", format!("{} GenExps", format!("{: >12}", "Command").bright_green().bold()));

        let cfg = GenConfig {
            cap: self.cap,
            chunk: self.chunk,
            archspec: self.arch.clone(),
            stark_src: self.stark_src.clone(),
            keep_dir: self.keep_dir.clone(),
            dry_run: self.emit_only,
        };
        let summary = generate_all(&self.proving_key, &cfg)?;
        tracing::info!(
            "{} {} kernels -> {} .exps.so ({} skipped)",
            format!("{: >12}", "GenExps").bright_green().bold(),
            summary.generated.len(),
            summary.placed,
            summary.skipped.len()
        );
        Ok(())
    }
}
