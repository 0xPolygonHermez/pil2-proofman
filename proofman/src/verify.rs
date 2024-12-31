use p3_field::Field;

use proofman_starks_lib_c::{stark_info_new_c, expressions_bin_new_c, stark_verify_c};
use std::fs::File;
use std::io::Read;

use colored::*;

use std::sync::Arc;

use proofman_common::{ProofCtx, ProofType, Prover, SetupCtx, get_global_constraints_lines_str};

use proofman_hints::aggregate_airgroupvals;
use proofman_util::{timer_start_info, timer_stop_and_log_info};

use std::os::raw::c_void;

use crate::verify_global_constraints_proof;

// This method is not ready to use!
pub fn verify_proof<F: Field>(
    provers: &mut [Box<dyn Prover<F>>],
    proof_ctx: Arc<ProofCtx<F>>,
    sctx: Arc<SetupCtx>,
) -> bool {
    const MY_NAME: &str = "Verify  ";
    timer_start_info!(VERIFYING_PROOF);
    let mut is_valid = true;

    for prover in provers.iter() {
        let p_proof = prover.get_proof();
        let prover_info = prover.get_prover_info();

        let setup_path =
            proof_ctx.global_info.get_air_setup_path(prover_info.airgroup_id, prover_info.air_id, &ProofType::Basic);

        let stark_info_path = setup_path.display().to_string() + ".starkinfo.json";
        let expressions_bin_path = setup_path.display().to_string() + ".verifier.bin";

        let p_stark_info = stark_info_new_c(stark_info_path.as_str(), true);
        let p_expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, true);

        let air_name = &proof_ctx.global_info.airs[prover_info.airgroup_id][prover_info.air_id].name;

        let verkey_file = proof_ctx
            .global_info
            .get_air_setup_path(prover_info.airgroup_id, prover_info.air_id, &ProofType::Basic)
            .with_extension("verkey.json");
        let mut contents = String::new();
        let mut file = File::open(verkey_file).unwrap();

        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read public inputs file: {}", err));
        let verkey_json: Vec<u64> = serde_json::from_str(&contents).unwrap();
        let verkey: Vec<F> = verkey_json.into_iter().map(|element| F::from_canonical_u64(element)).collect();

        let steps_fri: Vec<usize> = proof_ctx.global_info.steps_fri.iter().map(|step| step.n_bits).collect();
        let proof_challenges = prover.get_proof_challenges(steps_fri, proof_ctx.get_challenges().to_vec());

        let is_valid_proof = stark_verify_c(
            p_proof,
            p_stark_info,
            p_expressions_bin,
            verkey.as_ptr() as *mut c_void,
            proof_ctx.get_publics_ptr(),
            proof_ctx.get_proof_values_ptr(),
            proof_challenges.as_ptr() as *mut u8,
        );
        if !is_valid_proof {
            is_valid = false;
            log::info!(
                "{}: ··· {}",
                MY_NAME,
                format!("\u{2717} Proof of {}: Instance #{} was verified", air_name, prover_info.instance_id,)
                    .bright_red()
                    .bold()
            );
        } else {
            log::info!(
                "{}:     {}",
                MY_NAME,
                format!("\u{2713} Proof of {}: Instance #{} was verified", air_name, prover_info.instance_id,)
                    .bright_green()
                    .bold()
            );
        }
    }

    let airgroupvalues = aggregate_airgroupvals(proof_ctx.clone());

    let global_constraints = verify_global_constraints_proof(proof_ctx.clone(), sctx.clone(), airgroupvalues);
    let mut valid_global_constraints = true;

    let global_constraints_lines = get_global_constraints_lines_str(sctx.clone());

    for idx in 0..global_constraints.len() {
        let constraint = global_constraints[idx];
        let line_str = &global_constraints_lines[idx];

        if constraint.skip {
            log::debug!("{}:     · Skipping Global Constraint #{} -> {}", MY_NAME, idx, line_str,);
            continue;
        }

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

    if is_valid && valid_global_constraints {
        log::info!("{}: ··· {}", MY_NAME, "\u{2713} All proofs were verified".bright_green().bold());
    } else {
        log::info!("{}: ··· {}", MY_NAME, "\u{2717} Not all proofs were verified.".bright_red().bold());
    }

    timer_stop_and_log_info!(VERIFYING_PROOF);
    is_valid
}
