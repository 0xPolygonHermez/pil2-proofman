use clap::Parser;
use colored::Colorize;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::commands::field::Field;
use fields::Goldilocks;
use proofman::ProofMan;
use proofman_common::{ProofOptions, ProofmanOptions};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveMultilinearCmd {
    /// Witness computation dynamic library path
    #[clap(short = 'w', long)]
    pub witness_lib: PathBuf,

    /// Public inputs path
    #[clap(short = 'i', long)]
    pub public_inputs: Option<PathBuf>,

    /// Proving key folder path
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    #[clap(short = 'c', long, value_name = "KEY=VALUE", num_args(1..))]
    pub custom_commits: Vec<String>,

    /// Verify each proof right after generating it.
    #[arg(short = 'y', long = "verify_proofs", default_value_t = false)]
    pub verify_proofs: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8,
}

impl ProveMultilinearCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} ProveMultilinear", format!("{: >12}", "Command").bright_green().bold());
        println!();

        // The multilinear prover reuses the lighter (no aggregation setups)
        // context that the verify-constraints flow uses, and is CPU-only.
        let mut options = ProofmanOptions::new();
        options.verify_constraints();
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

        let mut proof_options = ProofOptions::default();
        proof_options.multilinear();
        proof_options.verify_proofs = self.verify_proofs;

        match self.field {
            Field::Goldilocks => proofman.generate_proof(
                self.witness_lib.clone(),
                self.public_inputs.clone(),
                None,
                self.verbose.into(),
                proof_options,
            )?,
        };

        Ok(())
    }
}
