use p3_field::Field;
use std::cmp;

use std::sync::Arc;

use crate::{verify_global_constraints_proof, WitnessLibrary};

use proofman_common::{ExecutionCtx, ProofCtx, Prover, SetupCtx};

use colored::*;

pub fn verify_constraints_proof<F: Field>(
    pctx: Arc<ProofCtx<F>>,
    ectx: Arc<ExecutionCtx>,
    sctx: Arc<SetupCtx>,
    provers: Vec<Box<dyn Prover<F>>>,
    mut witness_lib: Box<dyn WitnessLibrary<F>>,
) -> Result<(), Box<dyn std::error::Error>> {
    const MY_NAME: &str = "CstrVrfy";

    log::info!("{}: --> Checking constraints", MY_NAME);

    witness_lib.debug(pctx.clone(), ectx.clone(), sctx.clone());

    let mut constraints = Vec::new();
    for prover in provers.iter() {
        let constraints_prover_info = prover.verify_constraints(sctx.clone(), pctx.clone());
        constraints.push(constraints_prover_info);
    }

    let mut valid_constraints = true;
    for (air_instance_index, air_instance) in pctx.air_instance_repo.air_instances.read().unwrap().iter().enumerate() {
        let air_name = &pctx.global_info.airs[air_instance.airgroup_id][air_instance.air_id].name;
        let mut valid_constraints_prover = true;
        log::info!(
            "{}:     ► Instance #{}: Air [{}:{}] {}",
            MY_NAME,
            air_instance_index,
            air_instance.airgroup_id,
            air_instance.air_id,
            air_name,
        );
        for constraint in &constraints[air_instance_index] {
            let str_slice = unsafe { std::slice::from_raw_parts(constraint.line, constraint.line_size as usize) };
            let line_str = std::str::from_utf8(str_slice).unwrap();
            let valid = if constraint.n_rows > 0 {
                format!("has {} invalid rows", constraint.n_rows).bright_red()
            } else {
                "is valid".bright_green()
            };
            if constraint.im_pol {
                log::trace!(
                    "{}: ···    Intermediate polynomial (stage {}) {} -> {}",
                    MY_NAME,
                    constraint.stage,
                    valid,
                    line_str
                );
            } else if constraint.n_rows == 0 {
                log::debug!(
                    "{}:     · Constraint #{} (stage {}) {} -> {}",
                    MY_NAME,
                    constraint.id,
                    constraint.stage,
                    valid,
                    line_str
                );
            } else {
                log::info!(
                    "{}:     · Constraint #{} (stage {}) {} -> {}",
                    MY_NAME,
                    constraint.id,
                    constraint.stage,
                    valid,
                    line_str
                );
            }
            if constraint.n_rows > 0 {
                valid_constraints_prover = false;
            }
            let n_rows = cmp::min(constraint.n_rows, 10);
            for i in 0..n_rows {
                let row = constraint.rows[i as usize];
                if row.dim == 1 {
                    log::info!(
                        "{}: ···        \u{2717} Failed at row {} with value: {}",
                        MY_NAME,
                        row.row,
                        row.value[0]
                    );
                } else {
                    log::info!(
                        "{}: ···        \u{2717} Failed at row {} with value: [{}, {}, {}]",
                        MY_NAME,
                        row.row,
                        row.value[0],
                        row.value[1],
                        row.value[2]
                    );
                }
            }
        }

        if !valid_constraints_prover {
            log::info!(
                "{}: ··· {}",
                MY_NAME,
                format!("\u{2717} Not all constraints for Instance #{} were verified", air_instance_index,)
                    .bright_red()
                    .bold()
            );
        } else {
            log::info!(
                "{}:     {}",
                MY_NAME,
                format!("\u{2713} All constraints for Instance #{} were verified", air_instance_index,)
                    .bright_green()
                    .bold()
            );
        }

        if !valid_constraints_prover {
            valid_constraints = false;
        }
    }

    let global_constraints = verify_global_constraints_proof(pctx.clone(), sctx.clone());
    let mut valid_global_constraints = true;
    for constraint in &global_constraints {
        let str_slice = unsafe { std::slice::from_raw_parts(constraint.line, constraint.line_size as usize) };
        let line_str = std::str::from_utf8(str_slice).unwrap();

        let valid = if !constraint.valid { "is invalid".bright_red() } else { "is valid".bright_green() };
        if constraint.valid {
            log::debug!("{}:     · Global Constraint #{} {} -> {}", MY_NAME, constraint.id, valid, line_str);
        } else {
            log::info!("{}:     · Global Constraint #{} {} -> {}", MY_NAME, constraint.id, valid, line_str);
        }
        if !constraint.valid {
            valid_global_constraints = false;
            if constraint.dim == 1 {
                log::info!("{}: ···        \u{2717} Failed with value: {}", MY_NAME, constraint.value[0]);
            } else {
                log::info!(
                    "{}: ···        \u{2717} Failed with value: [{}, {}, {}]",
                    MY_NAME,
                    constraint.value[0],
                    constraint.value[1],
                    constraint.value[2]
                );
            }
        }
    }

    if valid_global_constraints {
        log::info!(
            "{}: ··· {}",
            MY_NAME,
            "\u{2713} All global constraints were successfully verified".bright_green().bold()
        );
    } else {
        log::info!("{}: ··· {}", MY_NAME, "\u{2717} Not all global constraints were verified".bright_red().bold());
    }

    if valid_constraints && valid_global_constraints {
        log::info!("{}: ··· {}", MY_NAME, "\u{2713} All constraints were verified".bright_green().bold());
        Ok(())
    } else {
        log::info!("{}: ··· {}", MY_NAME, "\u{2717} Not all constraints were verified.".bright_red().bold());
        Err(Box::new(std::io::Error::new(
            // <-- Return a boxed error
            std::io::ErrorKind::Other,
            format!("{}: Not all constraints were verified.", MY_NAME),
        )))
    }
}
