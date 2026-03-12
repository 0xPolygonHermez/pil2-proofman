// extern crate env_logger;
use clap::Parser;
use std::path::PathBuf;
use colored::Colorize;
use crate::commands::field::Field;

use fields::Goldilocks;

use proofman::check_setup_snark;
use proofman_common::initialize_logger;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct CheckSetupSnarkCmd {
    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key_snark: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl CheckSetupSnarkCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} CheckSetupSnark", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        match self.field {
            Field::Goldilocks => check_setup_snark::<Goldilocks>(&self.proving_key_snark.clone(), self.verbose.into())?,
        };

        Ok(())
    }
}
