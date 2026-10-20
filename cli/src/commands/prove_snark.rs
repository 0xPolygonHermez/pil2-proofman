// extern crate env_logger;
use clap::Parser;
use std::path::{Path, PathBuf};
use colored::Colorize;
use proofman_fields::Goldilocks;

use proofman::{ProofMan, SnarkWrapper};
use proofman_common::ProofmanOptions;
use proofman::generate_and_verify_recursivef;
use proofman_verifier::VadcopFinalProof;

/// Wrap a vadcop_final proof in the recursivef and the final SNARK of setup-snark: PLONK or FFLONK
/// (final/final.zkey) or pilfflonk (final/provingKey/), through proofman's final SNARK, as `prove -s`
/// does. Writes snark_proof.bin
#[derive(Parser)]
#[command(version, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveSnarkCmd {
    /// The vadcop_final proof, vadcop_final_proof.bin
    #[clap(short = 'p', long)]
    pub proof: String,

    /// Setup folder path: the provingKeySnark/ of setup-snark, with recursivef/ and final/
    #[clap(short = 'k', long)]
    pub proving_key_snark: PathBuf,

    /// The provingKey/ of the vadcop_final proof, which proofman loads: by default the one beside the
    /// provingKeySnark/, as setup-snark builds them
    #[clap(long)]
    pub proving_key: Option<PathBuf>,

    /// Output dir path
    #[clap(short = 'o', long, default_value = "tmp")]
    pub output_dir: PathBuf,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    /// Prove and verify the recursivef only, and compute the final SNARK's witness, without its proof
    #[clap(short = 'r', long, default_value_t = false)]
    pub only_recursivef: bool,

    /// Prove on the GPU: the recursivef and the final SNARK (with pilfflonk, its MSMs and NTTs). Needs a
    /// build with CUDA (nvcc found, no feature cpu-only) and a GPU
    #[clap(short = 'g', long, default_value_t = false)]
    pub gpu: bool,
}

impl ProveSnarkCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} ProveSnark", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let proof = VadcopFinalProof::load(&self.proof)
            .map_err(|e| format!("Failed to load VadcopFinalProof from file {}: {}", self.proof, e))?;

        if self.only_recursivef {
            if proof.hash == "blake3" {
                return Err("a blake3 key's recursivef is a STARK of its proving key, not of setup-snark".into());
            }
            let valid = generate_and_verify_recursivef::<Goldilocks>(
                &self.proving_key_snark,
                &proof,
                self.verbose.into(),
                self.gpu,
            )?;
            if !valid {
                tracing::info!("··· {}", "\u{2717} Stark RecursiveF proof was not verified".bright_red().bold());
                return Err("Stark proof was not verified".into());
            }
            tracing::info!("    {}", "\u{2713} Stark RecursiveF proof was verified".bright_green().bold());
            return Ok(());
        }

        // As `prove -s`: the wrapper on proofman's buffers, and proofman's final SNARK, of either hash.
        let proving_key = match &self.proving_key {
            Some(key) => key.clone(),
            None => self.proving_key_snark.parent().unwrap_or(Path::new(".")).join("provingKey"),
        };
        let mut options = ProofmanOptions::new();
        if self.gpu {
            options.gpu();
        }
        options.final_snark();
        options.verbose_mode(self.verbose.into());
        let proofman = ProofMan::<Goldilocks>::new(proving_key, options)?;
        let (aux_trace, d_buffers, reload_fixed_pols_gpu) = proofman.get_preallocated_buffers();
        let wrapper = SnarkWrapper::<Goldilocks>::new_with_preallocated_buffers(
            &self.proving_key_snark,
            self.verbose.into(),
            Some(aux_trace),
            self.gpu.then_some(d_buffers),
            Some(reload_fixed_pols_gpu),
            false,
            self.gpu,
        )?;
        let snark_proof = proofman.generate_final_snark_proof(&wrapper, &proof, None)?;
        snark_proof.save(self.output_dir.join("snark_proof.bin"))?;
        Ok(())
    }
}
