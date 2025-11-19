// extern crate env_logger;
use clap::Parser;
use proofman_common::{json_to_debug_instances_map, DebugInfo, ProofOptions};
use std::{collections::HashMap, path::PathBuf};
use colored::Colorize;
use crate::commands::field::Field;

use fields::Goldilocks;

use proofman::ProofMan;
use proofman_common::ParamsGPU;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct StatsCmd {
    /// Witness computation dynamic library path
    #[clap(short, long)]
    pub witness_lib: PathBuf,

    /// ROM file path
    /// This is the path to the ROM file that the witness computation dynamic library will use
    /// to generate the witness.
    #[clap(short = 'e', long)]
    pub elf: Option<PathBuf>,

    /// Public inputs path
    #[clap(short = 'p', long)]
    pub public_inputs: Option<PathBuf>,

    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    #[clap(short = 'd', long)]
    pub debug: Option<Option<String>>,

    #[clap(short = 'c', long, value_name="KEY=VALUE", num_args(1..))]
    pub custom_commits: Vec<String>,

    #[clap(short = 'n', long)]
    pub number_threads_witness: Option<usize>,

    #[clap(short = 'x', long)]
    pub max_witness_stored: Option<usize>,

    #[clap(short = 'm', long, default_value_t = false)]
    pub minimal_memory: bool,
}

impl StatsCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Stats", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let debug_info = match &self.debug {
            None => DebugInfo::default(),
            Some(None) => DebugInfo::new_debug(),
            Some(Some(debug_value)) => json_to_debug_instances_map(self.proving_key.clone(), debug_value.clone())?,
        };

        let mut custom_commits_map: HashMap<String, PathBuf> = HashMap::new();
        for commit in &self.custom_commits {
            if let Some((key, value)) = commit.split_once('=') {
                custom_commits_map.insert(key.to_string(), PathBuf::from(value));
            } else {
                eprintln!("Invalid commit format: {commit:?}");
            }
        }

        let mut gpu_params = ParamsGPU::default();
        if self.number_threads_witness.is_some() {
            gpu_params.with_number_threads_pools_witness(self.number_threads_witness.unwrap());
        }
        if self.max_witness_stored.is_some() {
            gpu_params.with_max_witness_stored(self.max_witness_stored.unwrap());
        }

        let proofman = ProofMan::<Goldilocks>::new(
            self.proving_key.clone(),
            custom_commits_map,
            true,
            false,
            false,
            gpu_params,
            self.verbose.into(),
            HashMap::new(),
        )?;

        match self.field {
            Field::Goldilocks => proofman.compute_witness(
                self.witness_lib.clone(),
                self.public_inputs.clone(),
                &debug_info,
                self.verbose.into(),
                ProofOptions::new(false, false, false, false, false, self.minimal_memory, false, PathBuf::new()),
            )?,
        };

        Ok(())
    }
}
