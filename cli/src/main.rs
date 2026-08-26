use std::process::exit;

use clap::{Parser, Subcommand};
mod commands;
use commands::gen_custom_commits_fixed::GenCustomCommitsFixedCmd;
use commands::gen_exps::GenExpsCmd;
use commands::get_constraints::GetConstraintsCmd;
use commands::pil_helpers::PilHelpersCmd;
use commands::prove::ProveCmd;
use commands::prove_snark::ProveSnarkCmd;
use commands::verify_constraints::VerifyConstraintsCmd;
use commands::debug_info::DebugInfoCmd;
use commands::stats::StatsCmd;
use commands::verify_stark::VerifyStark;
use commands::verify_snark::VerifySnark;
use commands::prove_air::ProveAirCmd;
use commands::execute::ExecuteCmd;
use commands::pilout::{PiloutSubcommands, PiloutCmd};
use commands::setup::CheckSetupCmd;
use commands::setup_snark::CheckSetupSnarkCmd;
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
    CheckSetupSnark(CheckSetupSnarkCmd),
    Soundness(SoundnessCmd),
    Prove(ProveCmd),
    ProveSnark(ProveSnarkCmd),
    PilHelpers(PilHelpersCmd),
    VerifyConstraints(VerifyConstraintsCmd),
    DebugInfo(DebugInfoCmd),
    Stats(StatsCmd),
    Execute(ExecuteCmd),
    VerifyStark(VerifyStark),
    VerifySnark(VerifySnark),
    GetConstraints(GetConstraintsCmd),
    GenCustomCommitsFixed(GenCustomCommitsFixedCmd),
    ProveAir(ProveAirCmd),
    GenExps(GenExpsCmd),
}

fn main() {
    print_banner(false);

    let cli = Cli::parse();

    // PIL expression trees in large AIRs (e.g. ZisK) can be thousands of levels
    // deep, which overflows the default 8 MB main-thread stack. Run all commands
    // on a thread with a larger stack to avoid the overflow.
    let result = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || match &cli.command {
            Commands::Pilout(args) => match &args.pilout_commands {
                PiloutSubcommands::Inspect(args) => args.run(),
            },
            Commands::CheckSetup(args) => args.run(),
            Commands::CheckSetupSnark(args) => args.run(),
            Commands::Soundness(args) => args.run(),
            Commands::Prove(args) => args.run(),
            Commands::ProveSnark(args) => args.run(),
            Commands::PilHelpers(args) => args.run(),
            Commands::VerifyConstraints(args) => args.run(),
            Commands::DebugInfo(args) => args.run(),
            Commands::GenCustomCommitsFixed(args) => args.run(),
            Commands::GetConstraints(args) => args.run(),
            Commands::VerifyStark(args) => args.run(),
            Commands::VerifySnark(args) => args.run(),
            Commands::Stats(args) => args.run(),
            Commands::Execute(args) => args.run(),
            Commands::ProveAir(args) => args.run(),
            Commands::GenExps(args) => args.run(),
        })
        .expect("failed to spawn main thread")
        .join()
        .expect("main thread panicked");

    if let Err(e) = result {
        tracing::error!("{}", e);
        exit(1);
    }

    tracing::info!("Done");
}
