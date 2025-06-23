// extern crate env_logger;
use clap::Parser;
use std::{fs::File, path::PathBuf};
use std::io::Read;
use colored::Colorize;
use bytemuck::cast_slice;

use fields::{Goldilocks, PrimeField64};

use proofman::{verify_proof_from_file, verify_proof};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct VerifyStark {
    #[clap(short = 'p', long)]
    pub proof: String,

    #[clap(short = 'j', long, default_value_t = false)]
    pub json: bool,

    #[clap(short = 's', long)]
    pub stark_info: String,

    #[clap(short = 'e', long)]
    pub verifier_bin: String,

    #[clap(short = 'k', long)]
    pub verkey: String,

    #[clap(short = 'u', long)]
    pub public_inputs: Option<PathBuf>,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl VerifyStark {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} VerifyStark", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let publics = if let Some(publics) = &self.public_inputs {
            let mut contents = String::new();
            let mut file = File::open(publics).unwrap();

            let _ =
                file.read_to_string(&mut contents).map_err(|err| format!("Failed to read public inputs file: {}", err));
            let verkey_json_string: Vec<String> = serde_json::from_str(&contents).unwrap();
            let verkey_json: Vec<u64> =
                verkey_json_string.iter().map(|s| s.parse::<u64>().expect("Failed to parse string as u64")).collect();
            Some(verkey_json.into_iter().map(Goldilocks::from_u64).collect::<Vec<Goldilocks>>())
        } else {
            None
        };

        let valid = if self.json {
            verify_proof_from_file::<Goldilocks>(
                self.proof.clone(),
                self.stark_info.clone(),
                self.verifier_bin.clone(),
                self.verkey.clone(),
                publics,
                None,
                None,
            )
        } else {
            let mut file = File::open("path/to/proof.bin")?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let proof_slice: &[u64] = cast_slice(&buffer);
            let proof_ptr = proof_slice.as_ptr() as *mut u64;

            verify_proof::<Goldilocks>(
                proof_ptr,
                self.stark_info.clone(),
                self.verifier_bin.clone(),
                self.verkey.clone(),
                publics,
                None,
                None,
            )
        };

        if !valid {
            tracing::info!("··· {}", "\u{2717} Stark proof was not verified".bright_red().bold());
            Err("Stark proof was not verified".into())
        } else {
            tracing::info!("    {}", "\u{2713} Stark proof was verified".bright_green().bold());
            Ok(())
        }
    }
}
