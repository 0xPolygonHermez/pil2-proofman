// extern crate env_logger;
use clap::Parser;
use std::path::PathBuf;
use colored::Colorize;
use fields::Goldilocks;

use proofman::SnarkWrapper;
use proofman::generate_and_verify_recursivef;
use proofman_util::VadcopFinalProof;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveSnarkCmd {
    #[clap(short = 'p', long)]
    pub proof: String,

    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key_snark: PathBuf,

    /// Output dir path
    #[clap(short = 'o', long, default_value = "tmp")]
    pub output_dir: PathBuf,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    #[clap(short = 'r', long, default_value_t = false)]
    pub only_recursivef: bool,

    #[clap(short = 'j', long, default_value_t = false)]
    pub save_json: bool,
}

impl ProveSnarkCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} ProveSnark", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let proof = VadcopFinalProof::load(&self.proof)
            .map_err(|e| format!("Failed to load VadcopFinalProof from file {}: {}", self.proof, e))?;

        if self.only_recursivef {
            let valid = generate_and_verify_recursivef::<Goldilocks>(
                &self.proving_key_snark,
                &proof,
                &self.output_dir,
                self.verbose.into(),
            )?;
            if !valid {
                tracing::info!("··· {}", "\u{2717} Stark RecursiveF proof was not verified".bright_red().bold());
                Err("Stark proof was not verified".into())
            } else {
                tracing::info!("    {}", "\u{2713} Stark RecursiveF proof was verified".bright_green().bold());
                Ok(())
            }
        } else {
            let snark_wrapper: SnarkWrapper<Goldilocks> =
                SnarkWrapper::new(&self.proving_key_snark, self.verbose.into())?;
            let snark_proof = snark_wrapper.generate_final_snark_proof(&proof, Some(self.output_dir.clone()))?;
            snark_proof.save(self.output_dir.join("snark_proof.bin"))?;
            Ok(())
        }
    }
}
