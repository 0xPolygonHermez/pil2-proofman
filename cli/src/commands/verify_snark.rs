// extern crate env_logger;
use clap::Parser;
use proofman_common::initialize_logger;
use proofman::{SnarkProof, verify_snark_proof_with_expected};
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

    /// Hex-encoded 32-byte expected public digest (the committed statement),
    /// recomputed by the caller from the publics and the trusted rootC. When
    /// omitted, verification only attests snark self-consistency, not a statement.
    #[clap(long)]
    pub public_digest: Option<String>,

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

        let expected: Option<Vec<u8>> = match &self.public_digest {
            Some(hex) => {
                let hex = hex.strip_prefix("0x").unwrap_or(hex);
                let bytes = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16))
                    .collect::<Result<Vec<u8>, _>>()
                    .map_err(|e| format!("invalid --public-digest hex: {e}"))?;
                Some(bytes)
            }
            None => None,
        };

        Ok(verify_snark_proof_with_expected(&proof, &self.verkey, expected.as_deref())?)
    }
}
