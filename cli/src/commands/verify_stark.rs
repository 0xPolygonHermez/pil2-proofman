// extern crate env_logger;
use clap::Parser;
use proofman_verifier::verify;
use proofman_common::{initialize_logger, VerboseMode};
use std::fs::File;
use std::io::Read;
use colored::Colorize;
use bytemuck::cast_slice;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct VerifyStark {
    #[clap(short = 'p', long)]
    pub proof: String,

    #[clap(short = 'k', long)]
    pub verkey: String,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl VerifyStark {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} VerifyStark", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(VerboseMode::Info, None);

        let mut proof_file = File::open(self.proof.clone())?;
        let mut proof_buffer = Vec::new();
        proof_file.read_to_end(&mut proof_buffer)?;

        let proof_slice: &[u64] = cast_slice(&proof_buffer);

        let mut verkey_file = File::open(self.verkey.clone())?;
        let mut verkey_buffer = Vec::new();
        verkey_file.read_to_end(&mut verkey_buffer)?;

        let verkey_slice: &[u64] = cast_slice(&verkey_buffer);

        let valid = verify(proof_slice, verkey_slice);

        if !valid {
            tracing::info!("··· {}", "\u{2717} Stark proof was not verified".bright_red().bold());
            Err("Stark proof was not verified".into())
        } else {
            tracing::info!("    {}", "\u{2713} Stark proof was verified".bright_green().bold());
            Ok(())
        }
    }
}
