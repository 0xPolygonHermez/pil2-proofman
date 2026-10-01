// extern crate env_logger;
use clap::Parser;
use proofman_common::initialize_logger;
use proofman::{SnarkProof, verify_snark_proof};
use colored::Colorize;
use std::path::PathBuf;

/// Verify a final SNARK proof, snark_proof.bin: PLONK and FFLONK with snarkjs, pilfflonk with its JS
/// verifier (Node.js). Exits with 0 only if it verifies
#[derive(Parser)]
#[command(version, long_about = None)]
#[command(propagate_version = true)]
pub struct VerifySnark {
    /// The proof prove-snark wrote, snark_proof.bin
    #[clap(short = 'p', long)]
    pub proof: String,

    /// The final SNARK's verification key: final/final.verkey.json, snarkjs's, for PLONK and FFLONK;
    /// final/provingKey/<name>/pilfflonk/pilfflonk.vkey.json for pilfflonk
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
