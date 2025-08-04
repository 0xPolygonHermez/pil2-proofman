// extern crate env_logger;
use clap::Parser;
use fields::Goldilocks;
use proofman_common::initialize_logger;
use std::path::PathBuf;
use colored::Colorize;

use proofman_common::{get_global_constraints_lines_str, get_constraints_lines_str, GlobalInfo, SetupCtx, ProofType};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct GetConstraintsCmd {
    /// Setup folder path
    #[clap(long)]
    pub proving_key: PathBuf,
}

impl GetConstraintsCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        initialize_logger(proofman_common::VerboseMode::Info, None);

        tracing::info!("{}", format!("{} GetConstraints", format!("{: >12}", "Command").bright_green().bold()));
        tracing::info!("");

        let global_info = GlobalInfo::new(&self.proving_key);
        let sctx: SetupCtx<Goldilocks> = SetupCtx::new(&global_info, &ProofType::Basic, false);

        for airgroup_id in 0..global_info.air_groups.len() {
            for air_id in 0..global_info.airs[airgroup_id].len() {
                tracing::info!(
                    "{}",
                    format!(
                        "    ► Constraints of {} - {}",
                        global_info.air_groups[airgroup_id], global_info.airs[airgroup_id][air_id].name,
                    )
                    .bright_white()
                    .bold()
                );
                let constraints_lines = get_constraints_lines_str(&sctx, airgroup_id, air_id);
                for (idx, line) in constraints_lines.iter().enumerate() {
                    tracing::info!("        · Constraint #{} : {}", idx, line);
                }
            }
        }

        let global_constraints_lines = get_global_constraints_lines_str(&sctx);

        tracing::info!("{}", "    ► Global Constraints".bright_white().bold());
        for (idx, line) in global_constraints_lines.iter().enumerate() {
            tracing::info!("        · Global Constraint #{} -> {}", idx, line);
        }

        Ok(())
    }
}
