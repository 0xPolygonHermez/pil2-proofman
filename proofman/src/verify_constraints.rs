use fields::PrimeField64;
use proofman_starks_lib_c::{
    get_n_constraints_c, get_n_global_constraints_c, verify_global_constraints_c, verify_constraints_c,
};
use std::cmp;
use proofman_common::{
    get_constraints_lines_str, get_global_constraints_lines_str, skip_prover_instance, ConstraintInfo, ConstraintInfoC,
    DebugInfo, GlobalConstraintInfo, ProofCtx, SetupCtx,
};

use std::os::raw::c_void;
use colored::*;

pub fn verify_constraints<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    global_id: usize,
    n_print_constraints: u64,
) -> Vec<ConstraintInfo> {
    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(global_id);
    let setup = sctx.get_setup(airgroup_id, air_id);

    let steps_params = pctx.get_air_instance_params(sctx, global_id, false);

    let p_setup = (&setup.p_setup).into();

    let n_constraints = get_n_constraints_c(p_setup);

    let mut constraints_info = vec![ConstraintInfo::new(n_print_constraints); n_constraints as usize];

    let (skip, constraints_skip) = skip_prover_instance(pctx, global_id);

    if !skip {
        if !constraints_skip.is_empty() {
            constraints_info.iter_mut().for_each(|constraint| constraint.skip = true);
            for constraint_id in &constraints_skip {
                constraints_info[*constraint_id].skip = false;
            }
        }

        let mut constraints_info_c: Vec<ConstraintInfoC> = constraints_info
            .iter_mut()
            .map(|info| ConstraintInfoC {
                id: info.id,
                stage: info.stage,
                im_pol: info.im_pol,
                n_rows: info.n_rows,
                skip: info.skip,
                n_print_constraints: info.n_print_constraints,
                // point at the inside of the rows Vec
                rows: info.rows.as_mut_ptr(),
            })
            .collect();

        verify_constraints_c(p_setup, (&steps_params).into(), constraints_info_c.as_mut_ptr() as *mut c_void);

        for (info_c, info_rust) in constraints_info_c.iter().zip(constraints_info.iter_mut()) {
            info_rust.id = info_c.id;
            info_rust.stage = info_c.stage;
            info_rust.im_pol = info_c.im_pol;
            info_rust.n_rows = info_c.n_rows;
            info_rust.skip = info_c.skip;
            info_rust.n_print_constraints = info_c.n_print_constraints;
        }
    }

    constraints_info
}

pub fn verify_global_constraints_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    debug_info: &DebugInfo,
    airgroupvalues: Vec<Vec<F>>,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("--> Checking global constraints");

    let mut airgroup_values_ptrs: Vec<*mut F> = airgroupvalues
        .iter() // Iterate mutably over the inner Vecs
        .map(|inner_vec| inner_vec.as_ptr() as *mut F) // Get a raw pointer to each inner Vec
        .collect();

    let n_global_constraints = get_n_global_constraints_c(sctx.get_global_bin());
    let mut global_constraints = vec![GlobalConstraintInfo::default(); n_global_constraints as usize];

    if !debug_info.debug_global_instances.is_empty() {
        global_constraints.iter_mut().for_each(|constraint| constraint.skip = true);
        for constraint_id in &debug_info.debug_global_instances {
            global_constraints[*constraint_id].skip = false;
        }
    }

    verify_global_constraints_c(
        sctx.get_global_info_file().as_str(),
        sctx.get_global_bin(),
        pctx.get_publics_ptr(),
        pctx.get_challenges_ptr(),
        pctx.get_proof_values_ptr(),
        airgroup_values_ptrs.as_mut_ptr() as *mut *mut u8,
        global_constraints.as_mut_ptr() as *mut c_void,
    );

    let mut valid_global_constraints = true;

    let global_constraints_lines = get_global_constraints_lines_str(sctx);

    for idx in 0..global_constraints.len() {
        let constraint = global_constraints[idx];
        let line_str = if global_constraints_lines[idx].len() > 100 { "" } else { &global_constraints_lines[idx] };

        if constraint.skip {
            tracing::debug!("    · Global Constraint #{} {} -> {}", idx, "is skipped".bright_yellow(), line_str,);
            continue;
        }

        let valid = if !constraint.valid { "is invalid".bright_red() } else { "is valid".bright_green() };
        if constraint.valid {
            tracing::debug!("    · Global Constraint #{} {} -> {}", constraint.id, valid, line_str);
        } else {
            tracing::info!("    · Global Constraint #{} {} -> {}", constraint.id, valid, line_str);
        }
        if !constraint.valid {
            valid_global_constraints = false;
            if constraint.dim == 1 {
                tracing::info!("···        \u{2717} Failed with value: {}", constraint.value[0]);
            } else {
                tracing::info!(
                    "···        \u{2717} Failed with value: [{}, {}, {}]",
                    constraint.value[0],
                    constraint.value[1],
                    constraint.value[2]
                );
            }
        }
    }

    if valid_global_constraints {
        tracing::info!("··· {}", "\u{2713} All global constraints were successfully verified".bright_green().bold());
        Ok(())
    } else {
        tracing::info!("··· {}", "\u{2717} Not all global constraints were verified".bright_red().bold());
        Err(Box::new(std::io::Error::other("Not all global constraints were verified.".to_string())))
    }
}

pub fn verify_constraints_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    instance_id: usize,
    n_print_constraints: u64,
) -> bool {
    let constraints = verify_constraints(pctx, sctx, instance_id, n_print_constraints);

    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;
    let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
    let (skip, _) = skip_prover_instance(pctx, instance_id);
    if skip {
        tracing::info!(
            "{}",
            format!("··· \u{2713} Skipping Instance #{air_instance_id} of {air_name} [{airgroup_id}:{air_id}]")
                .bright_yellow()
                .bold()
        );
        return true;
    };

    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;

    let constraints_lines = get_constraints_lines_str(sctx, airgroup_id, air_id);

    let mut valid_constraints_instance = true;
    let skipping = "is skipped".bright_yellow();

    tracing::info!("    ► Instance #{} of {} [{}:{}]", air_instance_id, air_name, airgroup_id, air_id,);
    for constraint in &constraints {
        if constraint.skip {
            tracing::debug!(
                "    · Constraint #{} (stage {}) {} -> {}",
                constraint.id,
                constraint.stage,
                skipping,
                constraints_lines[constraint.id as usize]
            );
            continue;
        }
        let valid = if constraint.n_rows > 0 {
            format!("has {} invalid rows", constraint.n_rows).bright_red()
        } else {
            "is valid".bright_green()
        };
        if constraint.im_pol {
            if constraint.n_rows == 0 {
                tracing::trace!(
                    "···    Intermediate polynomial (stage {}) {} -> {}",
                    constraint.stage,
                    valid,
                    constraints_lines[constraint.id as usize]
                );
            } else {
                tracing::info!(
                    "    · Constraint #{} (stage {}) {} -> {}",
                    constraint.id,
                    constraint.stage,
                    valid,
                    constraints_lines[constraint.id as usize]
                );
            }
        } else if constraint.n_rows == 0 {
            tracing::debug!(
                "    · Constraint #{} (stage {}) {} -> {}",
                constraint.id,
                constraint.stage,
                valid,
                constraints_lines[constraint.id as usize]
            );
        } else {
            tracing::info!(
                "    · Constraint #{} (stage {}) {} -> {}",
                constraint.id,
                constraint.stage,
                valid,
                constraints_lines[constraint.id as usize]
            );
        }
        if constraint.n_rows > 0 {
            valid_constraints_instance = false;
        }
        let n_rows = cmp::min(constraint.n_rows, constraint.n_print_constraints);
        for i in 0..n_rows {
            let row = constraint.rows[i as usize];
            if row.dim == 1 {
                tracing::info!("···        \u{2717} Failed at row {} with value: {}", row.row, row.value[0]);
            } else {
                tracing::info!(
                    "···        \u{2717} Failed at row {} with value: [{}, {}, {}]",
                    row.row,
                    row.value[0],
                    row.value[1],
                    row.value[2]
                );
            }
        }
    }

    if !valid_constraints_instance {
        tracing::info!(
            "··· {}",
            format!("\u{2717} Not all constraints for Instance #{air_instance_id} of {air_name} were verified")
                .bright_red()
                .bold()
        );
    } else {
        tracing::info!(
            "    {}",
            format!("\u{2713} All constraints for Instance #{air_instance_id} of {air_name} were verified")
                .bright_green()
                .bold()
        );
    }

    valid_constraints_instance
}
