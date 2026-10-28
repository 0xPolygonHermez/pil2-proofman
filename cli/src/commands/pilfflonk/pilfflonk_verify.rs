use clap::Args;
use colored::Colorize;
use proofman_common::initialize_logger;
use proofman_pilfflonk::js_verifier;
use std::path::PathBuf;

// The JS verifier (pilfflonk/docs/verifier.md#js-verifier), run as `verify-snark` runs snarkjs,
// with the arguments of `snarkjs fflonk verify`.
/// Verify a pilfflonk proof with the JS verifier (Node.js): exits with 0 only if it verifies
#[derive(Args)]
pub struct PilfflonkVerifyCmd {
    /// pilfflonk.vkey.json, in the pilfflonk/ directory of the provingKey/
    pub vkey: PathBuf,

    /// publics.json
    pub publics: PathBuf,

    /// proof.json
    pub proof: PathBuf,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl PilfflonkVerifyCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Pilfflonk verify subcommand", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        if js_verifier::verify(&self.vkey, &self.publics, &self.proof)? {
            tracing::info!("    {}", "\u{2713} pilfflonk proof was verified".bright_green().bold());
            Ok(())
        } else {
            tracing::info!("··· {}", "\u{2717} pilfflonk proof was not verified".bright_red().bold());
            Err("pilfflonk proof was not verified".into())
        }
    }
}
