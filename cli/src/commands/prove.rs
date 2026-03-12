// extern crate env_logger;
use clap::Parser;
use proofman_common::{json_to_debug_instances_map, DebugInfo};
use std::collections::HashMap;
use std::path::PathBuf;
use colored::Colorize;
use crate::commands::field::Field;
use fields::Goldilocks;

use proofman::SnarkWrapper;
use proofman::ProofMan;
use proofman::ProvePhaseResult;
use proofman_common::{ModeName, ProofOptions, ParamsGPU};
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveCmd {
    /// Witness computation dynamic library path
    #[clap(short = 'w', long)]
    pub witness_lib: PathBuf,

    /// ROM file path
    /// This is the path to the ROM file that the witness computation dynamic library will use
    /// to generate the witness.
    #[clap(short = 'e', long)]
    pub elf: Option<PathBuf>,

    /// Public inputs path
    #[clap(short = 'i', long)]
    pub public_inputs: Option<PathBuf>,

    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    /// Setup folder path
    #[clap(short = 's', long)]
    pub proving_key_snark: Option<PathBuf>,

    /// Output dir path
    #[clap(short = 'o', long, default_value = "tmp")]
    pub output_dir: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    #[clap(short = 'a', long, default_value_t = false)]
    pub aggregation: bool,

    #[clap(short = 'f', long, default_value_t = false)]
    pub compressed: bool,

    #[clap(short = 'y', long, default_value_t = false)]
    pub verify_proofs: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    #[clap(short = 'd', long)]
    pub debug: Option<Option<String>>,

    #[clap(short = 'c', long, value_name="KEY=VALUE", num_args(1..))]
    pub custom_commits: Vec<String>,

    #[clap(short = 'z', long, default_value_t = false)]
    pub preallocate: bool,

    #[clap(short = 'r', long, default_value_t = false)]
    pub rma: bool,

    #[clap(short = 'm', long, default_value_t = false)]
    pub minimal_memory: bool,

    #[clap(short = 't', long)]
    pub max_streams: Option<usize>,

    #[clap(short = 'n', long)]
    pub number_threads_witness: Option<usize>,

    #[clap(short = 'x', long)]
    pub max_witness_stored: Option<usize>,

    #[clap(short = 'b', long, default_value_t = false)]
    pub save_proofs: bool,
}

impl ProveCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Prove", format!("{: >12}", "Command").bright_green().bold());
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

        let debug_info = match &self.debug {
            None => DebugInfo::default(),
            Some(None) => DebugInfo::new_debug(),
            Some(Some(debug_value)) => json_to_debug_instances_map(self.proving_key.clone(), debug_value.clone())?,
        };

        let verify_constraints = debug_info.std_mode.name == ModeName::Debug;

        let mut gpu_params = ParamsGPU::new(self.preallocate);

        if let Some(max_streams) = self.max_streams {
            gpu_params.with_max_number_streams(max_streams);
        }
        if let Some(number_threads_witness) = self.number_threads_witness {
            gpu_params.with_number_threads_pools_witness(number_threads_witness);
        }
        if let Some(max_witness_stored) = self.max_witness_stored {
            gpu_params.with_max_witness_stored(max_witness_stored);
        }

        let proofman = ProofMan::<Goldilocks>::new(
            self.proving_key.clone(),
            verify_constraints,
            self.aggregation,
            gpu_params,
            self.verbose.into(),
            HashMap::new(),
        )?;

        let mut custom_commits_map: HashMap<String, PathBuf> = HashMap::new();
        for commit in &self.custom_commits {
            if let Some((key, value)) = commit.split_once('=') {
                custom_commits_map.insert(key.to_string(), PathBuf::from(value));
            } else {
                eprintln!("Invalid commit format: {commit:?}");
            }
        }
        proofman.register_custom_commits(custom_commits_map)?;

        let proof_options = ProofOptions::new(
            false,
            self.aggregation,
            self.rma,
            self.compressed,
            self.verify_proofs,
            self.minimal_memory,
            self.save_proofs,
            Some(self.output_dir.clone()),
        );
        if debug_info.std_mode.name == ModeName::Debug {
            match self.field {
                Field::Goldilocks => proofman.verify_proof_constraints(
                    self.witness_lib.clone(),
                    self.public_inputs.clone(),
                    None,
                    &debug_info.clone(),
                    self.verbose.into(),
                    false,
                )?,
            };
        } else {
            proofman.set_barrier();
            let result = match self.field {
                Field::Goldilocks => proofman.generate_proof(
                    self.witness_lib.clone(),
                    self.public_inputs.clone(),
                    None,
                    self.verbose.into(),
                    proof_options.clone(),
                )?,
            };

            if let ProvePhaseResult::Full(_, Some(vadcop_final_proof)) = result {
                // Save the vadcop final proof using the struct's save method
                vadcop_final_proof.save(self.output_dir.join("vadcop_final_proof.bin"))?;

                if let Some(proving_key_snark) = &self.proving_key_snark {
                    let snark_wrapper: SnarkWrapper<Goldilocks> =
                        SnarkWrapper::new(proving_key_snark, self.verbose.into())?;
                    snark_wrapper.generate_final_snark_proof(&vadcop_final_proof, Some(self.output_dir.clone()))?;
                }
            }
        }

        Ok(())
    }
}
