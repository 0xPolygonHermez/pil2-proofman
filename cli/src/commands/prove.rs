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
use proofman_common::{ModeName, ProofOptions, ProofmanOptions};

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

    #[clap(short = 'r', long, default_value_t = false)]
    pub no_rma: bool,

    #[clap(short = 'm', long, default_value_t = false)]
    pub minimal_memory: bool,

    #[clap(short = 't', long)]
    pub max_streams: Option<usize>,

    /// Cap on per-GPU recursive (aggregation) streams. Also memory-bounded.
    #[clap(long)]
    pub max_recursive_streams: Option<usize>,

    #[clap(short = 'n', long)]
    pub number_threads_witness: Option<usize>,

    #[clap(short = 'x', long)]
    pub max_witness_stored: Option<usize>,

    #[clap(short = 'g', long, default_value_t = false)]
    pub gpu: bool,
}

impl ProveCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Prove", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let debug_info = match &self.debug {
            None => DebugInfo::default(),
            Some(None) => DebugInfo::new_debug(),
            Some(Some(debug_value)) => json_to_debug_instances_map(self.proving_key.clone(), debug_value.clone())?,
        };

        let verify_constraints = debug_info.std_mode.name == ModeName::Debug;

        let mut options = ProofmanOptions::new();

        if let Some(max_streams) = self.max_streams {
            options.with_max_number_streams(max_streams);
        }
        if let Some(max_recursive_streams) = self.max_recursive_streams {
            options.with_max_number_recursive_streams(max_recursive_streams);
        }
        if let Some(number_threads_witness) = self.number_threads_witness {
            options.with_number_threads_pools_witness(number_threads_witness);
        }
        if let Some(max_witness_stored) = self.max_witness_stored {
            options.with_max_witness_stored(max_witness_stored);
        }
        if verify_constraints {
            options.verify_constraints();
        } else if !self.aggregation {
            options.no_aggregation();
        }
        if self.gpu {
            options.gpu();
        }
        options.verbose_mode(self.verbose.into());

        let proofman = ProofMan::<Goldilocks>::new(self.proving_key.clone(), options)?;

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
            !self.no_rma,
            self.compressed,
            self.verify_proofs,
            self.minimal_memory,
        );
        if debug_info.std_mode.name == ModeName::Debug {
            match self.field {
                Field::Goldilocks => proofman.verify_proof_constraints(
                    self.witness_lib.clone(),
                    self.public_inputs.clone(),
                    None,
                    &debug_info.clone(),
                    self.verbose.into(),
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
                        SnarkWrapper::new(proving_key_snark, self.verbose.into(), true, self.gpu)?;
                    snark_wrapper.generate_final_snark_proof(&vadcop_final_proof, None)?;
                }
            }
        }

        Ok(())
    }
}
