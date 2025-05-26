use std::process::exit;

use clap::{Parser, Subcommand};
mod commands;
use commands::gen_custom_commits_fixed::GenCustomCommitsFixedCmd;
use commands::get_constraints::GetConstraintsCmd;
use commands::pil_helpers::PilHelpersCmd;
use commands::prove::ProveCmd;
use commands::verify_constraints::VerifyConstraintsCmd;
use commands::verify_stark::VerifyStark;
use commands::pilout::{PiloutSubcommands, PiloutCmd};
use commands::setup::CheckSetupCmd;
use proofman_util::cli::print_banner;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Pilout(PiloutCmd),
    CheckSetup(CheckSetupCmd),
    Prove(ProveCmd),
    PilHelpers(PilHelpersCmd),
    VerifyConstraints(VerifyConstraintsCmd),
    VerifyStark(VerifyStark),
    GetConstraints(GetConstraintsCmd),
    GenCustomCommitsFixed(GenCustomCommitsFixedCmd),
}

fn main() {
    print_banner(false);

    let cli = Cli::parse();
    let result = match &cli.command {
        Commands::Pilout(args) => match &args.pilout_commands {
            PiloutSubcommands::Inspect(args) => args.run(),
        },
        Commands::CheckSetup(args) => args.run(),
        Commands::Prove(args) => args.run(),
        Commands::PilHelpers(args) => args.run(),
        Commands::VerifyConstraints(args) => args.run(),
        Commands::GenCustomCommitsFixed(args) => args.run(),
        Commands::GetConstraints(args) => args.run(),
        Commands::VerifyStark(args) => args.run(),
    };

    if let Err(e) = result {
        log::error!("{}", e);
        exit(1);
    }

    log::info!("Done");
}
