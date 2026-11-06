use clap::Parser;
use colored::Colorize;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::commands::field::Field;
use proofman_fields::Goldilocks;
use proofman::ProofMan;
use proofman_common::ProofmanOptions;

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

    /// Check every constraint over the full trace before proving (debug).
    #[arg(short = 'd', long = "verify_constraints", default_value_t = false)]
    pub verify_constraints: bool,

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

        if self.verify_constraints {
            eprintln!("--verify_constraints is not supported by this port; ignoring it");
        }
        let stats = match self.field {
            Field::Goldilocks => proofman.generate_multilinear_proof(
                self.witness_lib.clone(),
                self.public_inputs.clone(),
                self.verify_proofs,
                self.verbose.into(),
            )?,
        };

        // Per-air totals: instances, summed prove time, proof bytes.
        let mut per_air: Vec<(String, u32, usize, f64, f64, usize)> = Vec::new();
        for (name, n_bits, prove_ms, verify_ms, bytes) in stats {
            match per_air.iter_mut().find(|e| e.0 == name && e.1 == n_bits) {
                Some(e) => {
                    e.2 += 1;
                    e.3 += prove_ms;
                    e.4 += verify_ms;
                    e.5 += bytes;
                }
                None => per_air.push((name, n_bits, 1, prove_ms, verify_ms, bytes)),
            }
        }
        println!("{:<24} {:>6} {:>5} {:>12} {:>12} {:>12}", "air", "n_bits", "inst", "prove ms", "verify ms", "proof B");
        for (name, n_bits, inst, p, v, b) in per_air {
            println!("{name:<24} {n_bits:>6} {inst:>5} {p:>12.1} {v:>12.1} {b:>12}");
        }

        Ok(())
    }
}
