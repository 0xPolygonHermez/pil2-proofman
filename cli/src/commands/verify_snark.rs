// extern crate env_logger;
use clap::Parser;
use proofman_common::initialize_logger;
use proofman::{SnarkProof, verify_snark_proof};
use colored::Colorize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct VerifySnark {
    #[clap(short = 'p', long)]
    pub proof: String,

    #[clap(short = 'k', long)]
    pub verkey: PathBuf,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl VerifySnark {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} VerifySnark", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        let proof = SnarkProof::load(&self.proof)?;

        Ok(verify_snark_proof(&proof, &self.verkey)?)
    }
}
