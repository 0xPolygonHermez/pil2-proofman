// extern crate env_logger;
use clap::Parser;
use proofman_verifier::VadcopFinalProof;
use proofman_common::initialize_logger;
use proofman_fields::{Goldilocks, PrimeField64};
use proofman::verify_proof;
use colored::Colorize;
use proofman_util::{timer_start_info, timer_stop_and_log_info};

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
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} VerifyStark", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        let proof = VadcopFinalProof::load(&self.proof)?;

        // The verifier is read from the proving key rather than compiled in: the
        // aggregator binds the application's publics into q_verify, so a verifier
        // generated for one application rejects proofs another's correct prover
        // produced. The setup artifacts sit next to the verkey, as
        // <base>.starkinfo.json / <base>.verifier.bin / <base>.verkey.json.
        let base = self
            .verkey
            .strip_suffix(".verkey.bin")
            .or_else(|| self.verkey.strip_suffix(".verkey.json"))
            .ok_or_else(|| {
                format!(
                    "--verkey must be a <name>.verkey.bin or <name>.verkey.json from a provingKey, got {}",
                    self.verkey
                )
            })?
            .to_string();

        timer_start_info!(VERIFY_STARK);
        let publics: Vec<Goldilocks> = proof.public_values.iter().map(|&x| Goldilocks::from_u64(x)).collect();
        let valid = verify_proof::<Goldilocks>(
            proof.proof.as_ptr() as *mut u64,
            base.clone() + ".starkinfo.json",
            base.clone() + ".verifier.bin",
            base + ".verkey.json",
            Some(publics),
            None,
            None,
        );
        timer_stop_and_log_info!(VERIFY_STARK);

        if !valid {
            tracing::info!("··· {}", "\u{2717} Stark proof was not verified".bright_red().bold());
            Err("Stark proof was not verified".into())
        } else {
            tracing::info!("    {}", "\u{2713} Stark proof was verified".bright_green().bold());
            Ok(())
        }
    }
}
