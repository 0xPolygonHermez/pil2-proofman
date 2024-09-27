// extern crate env_logger;
use clap::{Parser, ValueEnum};
use proofman_common::VerboseMode;
use std::{fmt::Display, path::PathBuf};
use colored::Colorize;

use p3_goldilocks::Goldilocks;

use proofman::ProofMan;

use std::str::FromStr;

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
pub struct VerifyConstraintsCmd {
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

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    // Debug mode (-d, -dd)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase debug level")]
    pub debug: u8, // Using u8 to hold the number of `-d`
}

impl VerifyConstraintsCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} VerifyConstraints", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let verbose_mode: VerboseMode = self.verbose.into();
        env_logger::builder()
            .format_timestamp(None)
            .format_level(true)
            .format_target(false)
            .filter_level(verbose_mode.clone().into())
            .init();

        type GL = Goldilocks;

        let debug_mode = match self.debug {
            0 => 1, // Default to Error
            1 => 2, // -v
            2 => 3, // -vv _ => log::LevelFilter::Trace,
            _ => 1,
        };

        let _valid_constraints = match self.field {
            Field::Goldilocks => ProofMan::<GL>::generate_proof(
                self.witness_lib.clone(),
                self.rom.clone(),
                self.public_inputs.clone(),
                self.proving_key.clone(),
                PathBuf::new(),
                verbose_mode,
                debug_mode,
            )?,
        };

        Ok(())
    }
}
