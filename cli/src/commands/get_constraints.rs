// extern crate env_logger;
use clap::Parser;
use p3_goldilocks::Goldilocks;
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
    const MY_NAME: &str = "Cnstrnts";

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} GetConstraints", format!("{: >12}", "Command").bright_green().bold());
        println!();

        let global_info = GlobalInfo::new(&self.proving_key);
        let sctx: SetupCtx<Goldilocks> = SetupCtx::new(&global_info, &ProofType::Basic, false);

        initialize_logger(proofman_common::VerboseMode::Info);

        for airgroup_id in 0..global_info.air_groups.len() {
            for air_id in 0..global_info.airs[airgroup_id].len() {
                log::info!(
                    "{}",
                    format!(
                        "{}:     ► Constraints of {} - {}",
                        Self::MY_NAME,
                        global_info.air_groups[airgroup_id],
                        global_info.airs[airgroup_id][air_id].name,
                    )
                    .bright_white()
                    .bold()
                );
                let constraints_lines = get_constraints_lines_str(&sctx, airgroup_id, air_id);
                for (idx, line) in constraints_lines.iter().enumerate() {
                    log::info!("{}:         · Constraint #{} : {}", Self::MY_NAME, idx, line);
                }
            }
        }

        let global_constraints_lines = get_global_constraints_lines_str(&sctx);

        log::info!("{}", format!("{}:     ► Global Constraints", Self::MY_NAME,).bright_white().bold());
        for (idx, line) in global_constraints_lines.iter().enumerate() {
            log::info!("{}:         · Global Constraint #{} -> {}", Self::MY_NAME, idx, line);
        }

        Ok(())
    }
}
