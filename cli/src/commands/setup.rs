use clap::Parser;
use colored::Colorize;
use proofman_common::initialize_logger;
use proofman_util::{timer_start_info, timer_stop_and_log_info};
use setup::{generate_setup, SetupConfig};
use std::path::{Path, PathBuf};

use anyhow::Result;

/// Resolve a path to absolute using the Rust process's cwd as base.
/// Unlike `canonicalize`, this works even if the path doesn't exist yet (e.g. builddir).
fn abs(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

#[derive(Parser)]
#[command(version, about = "Generate a setup given a pilout file", long_about = None)]
#[command(propagate_version = true)]
pub struct SetupCmd {
    /// Path to the pilout .ptb file
    #[clap(short = 'a', long = "pilout")]
    pilout: PathBuf,

    /// Build output directory
    #[clap(short = 'b', long = "builddir", default_value = "tmp")]
    builddir: PathBuf,

    /// Binary files (repeatable: -i file1 -i file2)
    #[clap(short = 'i', long = "binfiles", num_args = 0..)]
    binfiles: Vec<PathBuf>,

    /// Stark structs JSON file
    #[clap(short = 's', long = "starkstructs")]
    starkstructs: Option<PathBuf>,

    /// Standard path
    #[clap(short = 't', long = "std-path")]
    std_path: Option<PathBuf>,

    /// Generate aggregation/recursive setup
    #[clap(short = 'r', long = "recursive")]
    recursive: bool,

    /// Optimize intermediate polynomials
    #[clap(short = 'm', long = "impols")]
    impols: bool,

    /// Fixed polynomials path
    #[clap(short = 'u', long = "fixed")]
    fixed: Option<PathBuf>,

    /// Path to the pil2-proofman-js root directory
    #[clap(long = "js-root")]
    js_root: PathBuf,

    /// Node.js max old space size in MB (e.g. 65536)
    #[clap(long = "max-old-space-size")]
    max_old_space_size: Option<u64>,
}

impl SetupCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Generate Setup", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(proofman_common::VerboseMode::Info, None);

        timer_start_info!(GENERATE_SETUP);

        // Resolve all paths to absolute so they work regardless of Node's cwd.
        let starkstructs = self.starkstructs.as_deref().map(abs).transpose()?;
        let pilout = abs(&self.pilout)?;
        let builddir = abs(&self.builddir)?;
        let binfiles = self.binfiles.iter().map(|p| abs(p)).collect::<Result<Vec<PathBuf>, _>>()?;
        let std_path = self.std_path.as_deref().map(abs).transpose()?;
        let fixed = self.fixed.as_deref().map(abs).transpose()?;

        let config = SetupConfig {
            pilout,
            builddir,
            binfiles,
            starkstructs,
            std_path,
            fixed,
            recursive: self.recursive,
            impols: self.impols,
            js_root: self.js_root.clone(),
            max_old_space_size: self.max_old_space_size,
        };

        generate_setup(&config)?;

        tracing::info!("{}", "\u{2713} Setup completed successfully".green().bold());

        timer_stop_and_log_info!(GENERATE_SETUP);

        Ok(())
    }
}
