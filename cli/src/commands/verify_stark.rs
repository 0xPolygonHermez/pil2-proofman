// extern crate env_logger;
use clap::Parser;
use proofman::verify_final_proof;
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

    #[clap(short = 'j', long, default_value_t = false)]
    pub json: bool,

    #[clap(short = 's', long)]
    pub stark_info: String,

    #[clap(short = 'e', long)]
    pub verifier_bin: String,

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

        let mut file = File::open(self.proof.clone())?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let proof_slice: &[u64] = cast_slice(&buffer);

        let valid =
            verify_final_proof(proof_slice, self.stark_info.clone(), self.verifier_bin.clone(), self.verkey.clone());

        if !valid {
            tracing::info!("··· {}", "\u{2717} Stark proof was not verified".bright_red().bold());
            Err("Stark proof was not verified".into())
        } else {
            tracing::info!("    {}", "\u{2713} Stark proof was verified".bright_green().bold());
            Ok(())
        }
    }
}
