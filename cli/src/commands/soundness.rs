// extern crate env_logger;
use clap::Parser;
use std::path::PathBuf;
use colored::Colorize;
use std::fs::File;
use std::io::Write;

use fields::Goldilocks;

use proofman_common::{initialize_logger, print_soundness_table, soundness_info};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct SoundnessCmd {
    /// Setup folder path
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    #[clap(short = 'a', long, default_value_t = false)]
    pub aggregation: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    #[clap(short = 'o', long)]
    pub output_path: Option<PathBuf>,
}

impl SoundnessCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Soundness", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        let soundness_info =
            soundness_info::<Goldilocks>(self.proving_key.clone(), self.aggregation, self.verbose.into())?;

        print_soundness_table(&soundness_info);
        let soundness_yaml = serde_yaml::to_string(&soundness_info).unwrap();

        if let Some(output_path) = &self.output_path {
            let mut file = File::create(output_path)?;
            file.write_all(soundness_yaml.as_bytes())?;
            println!("Soundness info written to {}", output_path.display());
        }

        Ok(())
    }
}
