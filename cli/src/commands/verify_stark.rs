// extern crate env_logger;
use clap::Parser;
use proofman_verifier::{verifier, VadcopFinalProof};
use proofman_common::initialize_logger;
use std::fs::File;
use std::io::Read;
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

        let mut verkey_file = File::open(&self.verkey)?;
        let mut vk_bytes = Vec::new();
        verkey_file.read_to_end(&mut vk_bytes)?;
        if vk_bytes.len() % 8 != 0 {
            return Err(format!("Verkey file size ({} bytes) is not a multiple of 8", vk_bytes.len()).into());
        }
        let vk: Vec<u64> = vk_bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();

        timer_start_info!(VERIFY_STARK);
        // The hash family travels inside the proof, so the verifier dispatches
        // without an out-of-band flag or compile-time feature.
        let v = verifier(&proof.hash);
        let valid = if proof.compressed {
            v.verify_vadcop_final_compressed(&proof, &vk)
        } else {
            v.verify_vadcop_final(&proof, &vk)
        };
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
