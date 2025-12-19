// extern crate env_logger;
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use colored::Colorize;
use fields::Goldilocks;
use std::io::Read;
use bytemuck::cast_slice;

use proofman::ProofMan;
use proofman_common::{ProofOptions, ParamsGPU};
use std::fs::{self, File};
use std::path::Path;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveSnarkCmd {
    #[clap(short = 'p', long)]
    pub proof: String,

    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    /// Output dir path
    #[clap(short = 'o', long, default_value = "tmp")]
    pub output_dir: PathBuf,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    #[clap(short = 'b', long, default_value_t = false)]
    pub save_proofs: bool,
}

impl ProveSnarkCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} ProveSnark", format!("{: >12}", "Command").bright_green().bold());
        println!();

        if Path::new(&self.output_dir.join("proofs")).exists() {
            // In distributed mode two different processes may enter here at the same time and try to remove the same directory
            if let Err(e) = fs::remove_dir_all(self.output_dir.join("proofs")) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    return Err(format!("Failed to remove the proofs directory: {e:?}").into());
                }
            }
        }

        if let Err(e) = fs::create_dir_all(self.output_dir.join("proofs")) {
            if e.kind() != std::io::ErrorKind::AlreadyExists {
                // prevent collision in distributed mode
                return Err(format!("Failed to create the proofs directory: {e:?}").into());
            }
        }

        let mut proof_file = File::open(&self.proof)?;
        let mut proof_u64 = Vec::new();
        proof_file.read_to_end(&mut proof_u64)?;
        let proof = cast_slice::<u8, u64>(&proof_u64);

        let proofman = ProofMan::<Goldilocks>::new(
            self.proving_key.clone(),
            HashMap::new(),
            false,
            false,
            true,
            ParamsGPU::new(false),
            self.verbose.into(),
            HashMap::new(),
        )?;

        let proof_options =
            ProofOptions::new(false, false, false, true, false, false, self.save_proofs, self.output_dir.clone());

        proofman.generate_final_snark_proof(proof, proof_options.clone())?;

        Ok(())
    }
}
