// extern crate env_logger;
use clap::Parser;
use std::path::PathBuf;
use colored::Colorize;
use crate::commands::field::Field;

use fields::Goldilocks;

use proofman::ProofMan;
use proofman_common::initialize_logger;

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

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl CheckSetupCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} CheckSetup", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        match self.field {
            Field::Goldilocks => {
                ProofMan::<Goldilocks>::check_setup(self.proving_key.clone(), self.aggregation, self.verbose.into())?
            }
        };

        Ok(())
    }
}
