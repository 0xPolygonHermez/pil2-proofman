use clap::Parser;
use colored::Colorize;
use std::path::PathBuf;

use fields::{Goldilocks, PrimeField64};
use proofman_common::{initialize_logger, GlobalInfo, ProofType};
use proofman_multilinear::{verify_air, AirIr, MlProof};
use proofman_util::{timer_start_info, timer_stop_and_log_info};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct VerifyMultilinearCmd {
    /// Multilinear proof file(s) (`.mlproof.bin`, as written by `prove --multilinear`)
    #[clap(short = 'p', long, num_args(1..), required = true)]
    pub proof: Vec<PathBuf>,

    /// Proving key folder (used to resolve each AIR's `.mlinfo.bin`)
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    /// Public inputs, as a JSON array of decimal field elements
    #[clap(short = 'i', long)]
    pub public_inputs: Option<PathBuf>,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8,
}

impl VerifyMultilinearCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} VerifyMultilinear", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        let global_info = GlobalInfo::new(&self.proving_key)?;

        let publics_override: Option<Vec<Goldilocks>> = match &self.public_inputs {
            Some(path) => {
                let data = std::fs::read_to_string(path)?;
                let values: Vec<serde_json::Value> = serde_json::from_str(&data)?;
                Some(
                    values
                        .iter()
                        .map(|v| {
                            let u = match v {
                                serde_json::Value::String(s) => s.parse::<u64>(),
                                serde_json::Value::Number(n) => Ok(n.as_u64().unwrap_or_default()),
                                _ => Ok(0),
                            }
                            .map_err(|e| format!("invalid public input {v}: {e}"))?;
                            Ok::<_, String>(Goldilocks::from_u64(u))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            None => None,
        };

        timer_start_info!(VERIFY_MULTILINEAR);
        for proof_path in &self.proof {
            let proof = MlProof::load(proof_path)?;

            let setup_path =
                global_info.get_air_setup_path(proof.airgroup_id as usize, proof.air_id as usize, &ProofType::Basic);
            let mlinfo_path = setup_path.with_extension("mlinfo.bin");
            let ir = AirIr::load(&mlinfo_path)?;

            // The proof carries its publics; an explicit --public-inputs file
            // overrides them (and must match what the proof committed to).
            let publics = match &publics_override {
                Some(p) => p.clone(),
                None => proof.publics.iter().map(|v| Goldilocks::from_u64(v.as_canonical_u64())).collect(),
            };

            match verify_air(&ir, &proof, &publics, None) {
                Ok(()) => {
                    println!(
                        "{}: {} {}",
                        proof_path.display(),
                        format!("{: >2}", "\u{2713}").bright_green().bold(),
                        "Multilinear proof was verified".bright_green().bold()
                    );
                }
                Err(e) => {
                    println!(
                        "{}: {} {}",
                        proof_path.display(),
                        format!("{: >2}", "\u{2717}").bright_red().bold(),
                        format!("Multilinear proof failed: {e}").bright_red().bold()
                    );
                    return Err(format!("proof {} did not verify: {e}", proof_path.display()).into());
                }
            }
        }
        timer_stop_and_log_info!(VERIFY_MULTILINEAR);

        Ok(())
    }
}
