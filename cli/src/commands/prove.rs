// extern crate env_logger;
use clap::{Parser, ValueEnum};
use std::{fmt::Display, path::PathBuf, sync::PoisonError};
use colored::Colorize;

use p3_goldilocks::Goldilocks;

use proofman::ProofMan;

use std::str::FromStr;
use proofman_common::{ProofCtx, ProofType};

#[derive(Parser, Debug, Clone, ValueEnum)]
pub enum Field {
    Goldilocks,
    // Add other variants here as needed
}

impl FromStr for Field {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Goldilocks" => Ok(Field::Goldilocks),
            // Add parsing for other variants here
            _ => Err(format!("'{}' is not a valid value for Field", s)),
        }
    }
}

impl Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Field::Goldilocks => write!(f, "goldilocks"),
        }
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveCmd {
    /// Witness computation dynamic library path
    #[clap(short, long)]
    pub witness_lib: PathBuf,

    /// ROM file path
    /// This is the path to the ROM file that the witness computation dynamic library will use
    /// to generate the witness.
    #[clap(short, long)]
    pub rom: Option<PathBuf>,

    /// Public inputs path
    #[clap(short = 'i', long)]
    pub public_inputs: Option<PathBuf>,

    /// Setup folder path
    #[clap(long)]
    pub proving_key: PathBuf,

    /// Output dir path
    #[clap(short = 'o', long, default_value = "tmp")]
    pub output_dir: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,
}

impl ProveCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Prove", format!("{: >12}", "Command").bright_green().bold());
        println!();

        type GL = Goldilocks;

        let proof_out = match self.field {
            Field::Goldilocks => ProofMan::<GL>::generate_proof(
                self.witness_lib.clone(),
                self.rom.clone(),
                self.public_inputs.clone(),
                self.proving_key.clone(),
                self.output_dir.clone(),
                0,
            )?,
        };
        println!("Proof generated successfully");
        match self.field {
            Field::Goldilocks => ProofMan::<GL>::generate_recursion_proof(
                &proof_out.0,
                &proof_out.1,
                &proof_out.2,
                &ProofType::Compressor,
            ),
        };
        println!("Compressor proofs generated successfully");

        println!("Recursive1 proofs generated successfully");

        Ok(())
    }
}
