// extern crate env_logger;
use clap::Parser;
use std::{collections::HashMap, path::PathBuf};
use colored::Colorize;
use crate::commands::field::Field;

use fields::Goldilocks;

use proofman::ProofMan;
use proofman_common::ParamsGPU;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct ExecuteCmd {
    /// Witness computation dynamic library path
    #[clap(short, long)]
    pub witness_lib: PathBuf,

    /// ROM file path
    /// This is the path to the ROM file that the witness computation dynamic library will use
    /// to generate the witness.
    #[clap(short = 'e', long)]
    pub elf: Option<PathBuf>,

    /// Inputs path
    #[clap(short = 'i', long)]
    pub input_data: Option<PathBuf>,

    /// Public inputs path
    #[clap(short = 'p', long)]
    pub public_inputs: Option<PathBuf>,

    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    #[clap(short = 'o', long)]
    pub output_path: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    #[clap(short = 'c', long, value_name="KEY=VALUE", num_args(1..))]
    pub custom_commits: Vec<String>,
}

impl ExecuteCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Stats", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let mut custom_commits_map: HashMap<String, PathBuf> = HashMap::new();
        for commit in &self.custom_commits {
            if let Some((key, value)) = commit.split_once('=') {
                custom_commits_map.insert(key.to_string(), PathBuf::from(value));
            } else {
                eprintln!("Invalid commit format: {commit:?}");
            }
        }

        let proofman;
        #[cfg(distributed)]
        {
            proofman = ProofMan::<Goldilocks>::new(
                self.proving_key.clone(),
                custom_commits_map,
                true,
                false,
                false,
                ParamsGPU::default(),
                self.verbose.into(),
                None,
            )?;
        }
        #[cfg(not(distributed))]
        {
            proofman = ProofMan::<Goldilocks>::new(
                self.proving_key.clone(),
                custom_commits_map,
                true,
                false,
                false,
                ParamsGPU::default(),
                self.verbose.into(),
            )?;
        }

        match self.field {
            Field::Goldilocks => proofman.execute(
                self.witness_lib.clone(),
                self.public_inputs.clone(),
                self.input_data.clone(),
                self.output_path.clone(),
                self.verbose.into(),
            )?,
        };

        Ok(())
    }
}
