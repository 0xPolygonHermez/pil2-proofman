use clap::Args;
use colored::Colorize;
use proofman_common::initialize_logger;
use proofman_pilfflonk::{check, CheckOptions, CheckReport, ProvingKey, DEFAULT_MAX_ROWS};
use std::path::PathBuf;

use super::PilfflonkWitnessArgs;

// The check (pilfflonk/docs/README.md#pilfflonk-check): the witness of prove, from a directory or a
// witness library, row by row against the constraints of its AIR, without proving; its output is
// verify-constraints'.
/// Check a pilfflonk witness row by row against its constraints, without proving: exits with 0 only if every constraint holds
#[derive(Args)]
pub struct PilfflonkCheckCmd {
    /// The provingKey/ that setup-pilfflonk wrote
    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    #[clap(flatten)]
    pub witness: PilfflonkWitnessArgs,

    /// The failing rows printed of each constraint, the first ones (all are counted)
    #[clap(long, value_name = "N", default_value_t = DEFAULT_MAX_ROWS)]
    pub max_rows: usize,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl PilfflonkCheckCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Pilfflonk check subcommand", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        let pk = ProvingKey::load(&self.proving_key)?;
        let witness = self.witness.open(&pk, self.verbose)?;
        let report = check(&pk, &witness, &CheckOptions { max_rows: self.max_rows })?;
        log_report(&report);
        if report.holds() {
            Ok(())
        } else {
            Err(format!("Not all constraints for Instance #0 of {} were verified", report.air_name).into())
        }
    }
}

/// The report as `verify-constraints` logs its own (`proofman/src/verify_constraints.rs`): the
/// constraints that fail and their first rows at info, the others at debug, the im pols' at trace.
fn log_report(report: &CheckReport) {
    // A proof has one instance (pilfflonk/docs/README.md#scope): instance 0 of its AIR.
    let instance = format!("Instance #0 of {}", report.air_name);
    tracing::info!("    ► {} [{}:{}]", instance, report.air.airgroup_id, report.air.air_id);
    for c in &report.constraints {
        if c.holds() {
            if c.im_pol {
                tracing::trace!(
                    "···    Intermediate polynomial (stage {}) {} -> {}",
                    c.stage,
                    "is valid".bright_green(),
                    c.line
                );
            } else {
                tracing::debug!(
                    "    · Constraint #{} (stage {}) {} -> {}",
                    c.index,
                    c.stage,
                    "is valid".bright_green(),
                    c.line
                );
            }
            continue;
        }
        let invalid = format!("has {} invalid rows", c.n_failed_rows).bright_red();
        tracing::info!("    · Constraint #{} (stage {}) {} -> {}", c.index, c.stage, invalid, c.line);
        for row in &c.failed_rows {
            tracing::info!("···        \u{2717} Failed at row {} with value: {}", row.row, row.value.to_decimal());
        }
        let hidden = c.n_failed_rows - c.failed_rows.len() as u64;
        if hidden > 0 {
            tracing::info!("···        \u{2026} and {} more invalid rows (--max-rows)", hidden);
        }
    }
    if report.holds() {
        tracing::info!(
            "    {}",
            format!("\u{2713} All constraints for {instance} were verified").bright_green().bold()
        );
    } else {
        tracing::info!(
            "··· {}",
            format!("\u{2717} Not all constraints for {instance} were verified").bright_red().bold()
        );
    }
}
