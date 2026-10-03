// extern crate env_logger;
use clap::Parser;
use std::path::PathBuf;
use colored::Colorize;
use proofman_fields::Goldilocks;

use proofman::SnarkWrapper;
use proofman::generate_and_verify_recursivef;
use proofman_verifier::VadcopFinalProof;

/// Wrap a vadcop_final proof in the recursivef and the final SNARK of setup-snark: PLONK or FFLONK
/// (final/final.zkey) or pilfflonk (final/provingKey/). Writes snark_proof.bin
#[derive(Parser)]
#[command(version, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveSnarkCmd {
    /// The vadcop_final proof, vadcop_final_proof.bin
    #[clap(short = 'p', long)]
    pub proof: String,

    /// Setup folder path: the provingKeySnark/ of setup-snark, with recursivef/ and final/
    #[clap(short = 'k', long)]
    pub proving_key_snark: PathBuf,

    /// Output dir path
    #[clap(short = 'o', long, default_value = "tmp")]
    pub output_dir: PathBuf,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    /// Prove and verify the recursivef only, and compute the final SNARK's witness, without its proof
    #[clap(short = 'r', long, default_value_t = false)]
    pub only_recursivef: bool,

    /// Prove on the GPU: the recursivef and the final SNARK (with pilfflonk, its MSMs and NTTs). Needs a
    /// build with CUDA (nvcc found, no feature cpu-only) and a GPU
    #[clap(short = 'g', long, default_value_t = false)]
    pub gpu: bool,
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
                self.verbose.into(),
                self.gpu,
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
                SnarkWrapper::new(&self.proving_key_snark, self.verbose.into(), true, self.gpu)?;
            let snark_proof = snark_wrapper.generate_final_snark_proof(&proof, None)?;
            snark_proof.save(self.output_dir.join("snark_proof.bin"))?;
            Ok(())
        }
    }
}
