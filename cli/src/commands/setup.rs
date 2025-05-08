// extern crate env_logger;
use clap::Parser;
use proofman_common::{initialize_logger, DebugInfo};
use std::path::PathBuf;
use colored::Colorize;
use crate::commands::field::Field;

use p3_goldilocks::Goldilocks;

use proofman::ProofMan;
use proofman_common::{VerboseMode, ProofOptions};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct CheckSetupCmd {
    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    #[clap(short = 'a', long, default_value_t = false)]
    pub aggregation: bool,

    #[clap(short = 'f', long, default_value_t = false)]
    pub final_snark: bool,
}

impl CheckSetupCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} CheckSetup", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let verbose_mode = VerboseMode::Debug;

        initialize_logger(verbose_mode);

        match self.field {
            Field::Goldilocks => ProofMan::<Goldilocks>::check_setup(
                self.proving_key.clone(),
                ProofOptions::new(
                    false,
                    verbose_mode,
                    self.aggregation,
                    self.final_snark,
                    false,
                    false,
                    DebugInfo::default(),
                ),
            )?,
        };

        Ok(())
    }
}
