// extern crate env_logger;
use clap::Parser;
use proofman_common::{initialize_logger, json_to_debug_instances_map, DebugInfo};
use std::collections::HashMap;
use std::path::PathBuf;
use colored::Colorize;
use crate::commands::field::Field;

use p3_goldilocks::Goldilocks;

use proofman::ProofMan;
use proofman_common::{ModeName, ProofOptions};
use std::fs;
use std::path::Path;

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

    /// Output dir path
    #[clap(short = 'o', long, default_value = "tmp")]
    pub output_dir: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    #[clap(short = 'a', long, default_value_t = false)]
    pub aggregation: bool,

    #[clap(short = 'f', long, default_value_t = false)]
    pub final_snark: bool,

    #[clap(short = 'y', long, default_value_t = false)]
    pub verify_proofs: bool,

    #[clap(short = 'c', long, default_value_t = false)]
    pub preallocate: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    #[clap(short = 'd', long)]
    pub debug: Option<Option<String>>,

    #[clap(short = 's', long, value_name="KEY=VALUE", num_args(1..))]
    pub custom_commits: Vec<String>,
}

impl ProveCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Prove", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into());

        if Path::new(&self.output_dir.join("proofs")).exists() {
            // In distributed mode two different processes may enter here at the same time and try to remove the same directory
            if let Err(e) = fs::remove_dir_all(self.output_dir.join("proofs")) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    panic!("Failed to remove the proofs directory: {:?}", e);
                }
            }
        }

        if let Err(e) = fs::create_dir_all(self.output_dir.join("proofs")) {
            if e.kind() != std::io::ErrorKind::AlreadyExists {
                // prevent collision in distributed mode
                panic!("Failed to create the proofs directory: {:?}", e);
            }
        }

        let debug_info = match &self.debug {
            None => DebugInfo::default(),
            Some(None) => DebugInfo::new_debug(),
            Some(Some(debug_value)) => json_to_debug_instances_map(self.proving_key.clone(), debug_value.clone()),
        };

        let mut custom_commits_map: HashMap<String, PathBuf> = HashMap::new();
        for commit in &self.custom_commits {
            if let Some((key, value)) = commit.split_once('=') {
                custom_commits_map.insert(key.to_string(), PathBuf::from(value));
            } else {
                eprintln!("Invalid commit format: {:?}", commit);
            }
        }

        if debug_info.std_mode.name == ModeName::Debug {
            match self.field {
                Field::Goldilocks => ProofMan::<Goldilocks>::verify_proof_constraints(
                    self.witness_lib.clone(),
                    self.public_inputs.clone(),
                    self.input_data.clone(),
                    self.proving_key.clone(),
                    self.output_dir.clone(),
                    custom_commits_map,
                    ProofOptions::new(
                        false,
                        self.verbose.into(),
                        self.aggregation,
                        self.final_snark,
                        self.verify_proofs,
                        false,
                        debug_info,
                    ),
                )?,
            };
        } else {
            match self.field {
                Field::Goldilocks => ProofMan::<Goldilocks>::generate_proof(
                    self.witness_lib.clone(),
                    self.public_inputs.clone(),
                    self.input_data.clone(),
                    self.proving_key.clone(),
                    self.output_dir.clone(),
                    custom_commits_map,
                    ProofOptions::new(
                        false,
                        self.verbose.into(),
                        self.aggregation,
                        self.final_snark,
                        self.verify_proofs,
                        self.preallocate,
                        debug_info,
                    ),
                )?,
            };
        }

        Ok(())
    }
}
