// extern crate env_logger;
use clap::Parser;
use proofman_common::{initialize_logger, DebugInfo};
use std::path::PathBuf;
use colored::Colorize;
use crate::commands::field::Field;

use p3_goldilocks::Goldilocks;

use proofman::ProofMan;
use proofman_common::ProofOptions;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct SetupCmd {
    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    #[clap(short = 'a', long, default_value_t = false)]
    pub aggregation: bool,

    #[clap(short = 'f', long, default_value_t = false)]
    pub final_snark: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl SetupCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} CheckSetup", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into());

        match self.field {
            Field::Goldilocks => ProofMan::<Goldilocks>::check_setup(
                self.proving_key.clone(),
                ProofOptions::new(
                    false,
                    self.verbose.into(),
                    self.aggregation,
                    self.final_snark,
                    false,
                    DebugInfo::default(),
                ),
            )?,
        };

        Ok(())
    }
}
