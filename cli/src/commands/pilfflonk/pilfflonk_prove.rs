use clap::Args;
use colored::Colorize;
use proofman_common::initialize_logger;
use proofman_pilfflonk::{prove, Device, ProveOptions, ProvingKey};
use std::path::PathBuf;

use super::PilfflonkWitnessArgs;

// The prover of spec §4.4: the provingKey/ of setup-pilfflonk and a witness directory (spec A.6) or a
// witness library (D4, plan M38c) in, proof.json and publics.json out, with the argument names of
// `prove`, `-g/--gpu` included (spec Fase 5, plan M43).
/// Prove a pilfflonk witness: writes proof.json and publics.json
#[derive(Args)]
pub struct PilfflonkProveCmd {
    /// The provingKey/ that setup-pilfflonk wrote
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    #[clap(flatten)]
    pub witness: PilfflonkWitnessArgs,

    /// Where proof.json and publics.json go (created if it does not exist)
    #[clap(short = 'o', long, visible_alias = "output")]
    pub output_dir: PathBuf,

    /// INSECURE, for tests only: fixes the blinding with this seed, 64 hexadecimal digits, so that
    /// the same seed gives the same proof. Whoever knows the seed can remove the blinding: the
    /// proof is not zero-knowledge
    #[clap(long, value_name = "HEX", value_parser = parse_seed)]
    pub insecure_blinding_seed: Option<[u8; 32]>,

    /// Runs the MSMs and the NTTs on the GPU: the same proof, bit for bit. Needs a build with CUDA
    /// (nvcc found, no feature cpu-only) and a GPU
    #[clap(short = 'g', long, default_value_t = false)]
    pub gpu: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

/// 32 bytes from 64 hexadecimal digits, in the order they are written.
fn parse_seed(text: &str) -> Result<[u8; 32], String> {
    let digits = text.as_bytes();
    if digits.len() != 64 || !digits.iter().all(u8::is_ascii_hexdigit) {
        return Err("a seed is 64 hexadecimal digits".to_string());
    }
    let mut seed = [0u8; 32];
    for (byte, pair) in seed.iter_mut().zip(digits.chunks(2)) {
        // Two ASCII hex digits: always valid UTF-8 and a u8.
        *byte = u8::from_str_radix(std::str::from_utf8(pair).unwrap_or("00"), 16).unwrap_or(0);
    }
    Ok(seed)
}

impl PilfflonkProveCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Pilfflonk prove subcommand", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        if self.insecure_blinding_seed.is_some() {
            tracing::warn!(
                "{}",
                "--insecure-blinding-seed: the blinding is fixed, and the proof is not zero-knowledge (tests only)"
                    .bright_yellow()
                    .bold()
            );
        }
        let device = if self.gpu {
            tracing::info!("··· The MSMs and the NTTs run on the GPU");
            Device::Gpu
        } else {
            Device::Cpu
        };
        let pk = ProvingKey::load_on(&self.proving_key, device)?;
        let witness = self.witness.open(&pk, self.verbose)?;
        let options = ProveOptions { insecure_blinding_seed: self.insecure_blinding_seed, ..ProveOptions::default() };
        let output = prove(&pk, &witness, &options)?;
        output.write(&self.output_dir)?;
        tracing::info!(
            "    {} {}",
            "\u{2713} pilfflonk proof written to".bright_green().bold(),
            self.output_dir.display()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_is_64_hex_digits_in_their_order() {
        let seed = parse_seed(&format!("0a{}ff", "00".repeat(30))).unwrap();
        assert_eq!((seed[0], seed[1], seed[31]), (0x0a, 0, 0xff));
        for bad in ["", "0a", &"g0".repeat(32), &"00".repeat(33)] {
            assert!(parse_seed(bad).is_err(), "{bad}");
        }
    }
}
