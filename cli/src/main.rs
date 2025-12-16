use std::process::exit;

use clap::{Parser, Subcommand};
mod commands;
use commands::gen_custom_commits_fixed::GenCustomCommitsFixedCmd;
use commands::get_constraints::GetConstraintsCmd;
use commands::pil_helpers::PilHelpersCmd;
use commands::prove::ProveCmd;
use commands::verify_constraints::VerifyConstraintsCmd;
use commands::stats::StatsCmd;
use commands::verify_stark::VerifyStark;
use commands::gen_witness::GenWitnessCmd;
use commands::execute::ExecuteCmd;
use commands::pilout::{PiloutSubcommands, PiloutCmd};
use commands::setup::CheckSetupCmd;
use commands::soundness::SoundnessCmd;
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
    Soundness(SoundnessCmd),
    Prove(ProveCmd),
    PilHelpers(PilHelpersCmd),
    VerifyConstraints(VerifyConstraintsCmd),
    Stats(StatsCmd),
    Execute(ExecuteCmd),
    VerifyStark(VerifyStark),
    GetConstraints(GetConstraintsCmd),
    GenCustomCommitsFixed(GenCustomCommitsFixedCmd),
    GenWitness(GenWitnessCmd),
}

fn main() {
    print_banner(false);

    let cli = Cli::parse();
    let result = match &cli.command {
        Commands::Pilout(args) => match &args.pilout_commands {
            PiloutSubcommands::Inspect(args) => args.run(),
        },
        Commands::CheckSetup(args) => args.run(),
        Commands::Soundness(args) => args.run(),
        Commands::Prove(args) => args.run(),
        Commands::PilHelpers(args) => args.run(),
        Commands::VerifyConstraints(args) => args.run(),
        Commands::GenCustomCommitsFixed(args) => args.run(),
        Commands::GetConstraints(args) => args.run(),
        Commands::VerifyStark(args) => args.run(),
        Commands::Stats(args) => args.run(),
        Commands::Execute(args) => args.run(),
        Commands::GenWitness(args) => args.run(),
    };

    if let Err(e) = result {
        tracing::error!("{}", e);
        exit(1);
    }

    tracing::info!("Done");
}
