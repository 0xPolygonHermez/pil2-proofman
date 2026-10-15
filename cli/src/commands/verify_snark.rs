// extern crate env_logger;
use clap::Parser;
use proofman_common::{initialize_logger, PublicsInfo};
use proofman::{SnarkProof, verify_snark_proof_with_expected};
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

    /// `publics_info.json` of the proving key. With `--publics` and `--rootc`, the expected
    /// public digest is recomputed from them, binding the proof to that statement. Without
    /// them, verification only attests snark self-consistency.
    #[clap(long, requires_all = ["publics", "rootc"])]
    pub publics_info: Option<PathBuf>,

    /// JSON array of the vadcop_final publics (u64).
    #[clap(long, requires = "publics_info")]
    pub publics: Option<PathBuf>,

    /// The trusted rootC: a `verkey.json`, a JSON array of 4 u64 limbs.
    #[clap(long, requires = "publics_info")]
    pub rootc: Option<PathBuf>,

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

        let expected = match (&self.publics_info, &self.publics, &self.rootc) {
            (Some(info), Some(publics), Some(rootc)) => {
                let info = PublicsInfo::from_file(info)?;
                let publics: Vec<u64> = serde_json::from_str(&std::fs::read_to_string(publics)?)?;
                let rootc: [u64; 4] = serde_json::from_str(&std::fs::read_to_string(rootc)?)?;
                Some(info.snark_public_digest(&publics, &rootc)?)
            }
            _ => None,
        };

        Ok(verify_snark_proof_with_expected(&proof, &self.verkey, expected.as_ref().map(|d| d.as_slice()))?)
    }
}
