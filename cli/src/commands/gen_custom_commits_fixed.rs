// extern crate env_logger;
use clap::Parser;
use libloading::{Library, Symbol};
use std::sync::Arc;
use proofman_common::{initialize_logger, DebugInfo, ProofCtx, ProofType, SetupCtx};
use std::{collections::HashMap, path::PathBuf};
use colored::Colorize;
use crate::commands::field::Field;
use witness::{WitnessLibInitFn, WitnessManager};

use p3_goldilocks::Goldilocks;

use proofman_common::ProofOptions;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct GenCustomCommitsFixedCmd {
    /// Witness computation dynamic library path
    #[clap(short, long)]
    pub witness_lib: PathBuf,

    /// ROM file path
    /// This is the path to the ROM file that the witness computation dynamic library will use
    /// to generate the witness.
    #[clap(short, long)]
    pub rom: Option<PathBuf>,

    /// Setup folder path
    #[clap(long)]
    pub proving_key: PathBuf,

    #[clap(long, default_value_t = Field::Goldilocks)]
    pub field: Field,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`

    #[clap(short = 'c', long, value_name="KEY=VALUE", num_args(1..))]
    pub custom_commits: Vec<String>,

    #[clap(long, short = 'k')]
    pub check: bool,
}

impl GenCustomCommitsFixedCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} VerifyConstraints", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into());

        let mut custom_commits_map: HashMap<String, PathBuf> = HashMap::new();
        for commit in &self.custom_commits {
            if let Some((key, value)) = commit.split_once('=') {
                custom_commits_map.insert(key.to_string(), PathBuf::from(value));
            } else {
                eprintln!("Invalid commit format: {:?}", commit);
            }
        }

        let options = ProofOptions::new(false, self.verbose.into(), false, false, DebugInfo::default());
        let pctx = Arc::new(ProofCtx::create_ctx(self.proving_key.clone(), custom_commits_map, options));

        let sctx = Arc::new(SetupCtx::<Goldilocks>::new(&pctx.global_info, &ProofType::Basic, false));

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone(), self.rom.clone(), None, None));

        // Load the witness computation dynamic library
        let library = unsafe { Library::new(&self.witness_lib)? };

        let witness_lib: Symbol<WitnessLibInitFn<Goldilocks>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(wcm.get_pctx().options.verbose_mode)?;
        witness_lib.register_witness(wcm.clone());

        wcm.gen_custom_commits_fixed(self.check)
    }
}
