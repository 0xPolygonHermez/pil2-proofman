use fields::PrimeField64;

use proofman_starks_lib_c::{
    expressions_bin_free_c, expressions_bin_new_c, get_max_n_tmp1_c, get_max_n_tmp3_c, set_memory_expressions_c,
    stark_info_free_c, stark_info_new_c, stark_verify_bn128_c, stark_verify_c, stark_verify_from_file_c,
};

use colored::*;

use proofman_common::{ProofCtx, ProofType};
use proofman_util::{timer_start_info, timer_stop_and_log_info};

use std::os::raw::c_void;

pub fn verify_final_proof(
    proof: &[u64],
    stark_info_path: String,
    expressions_bin_path: String,
    verkey_path: String,
) -> bool {
    timer_start_info!(VERIFY_FINAL_PROOF);
    let p_stark_info = stark_info_new_c(stark_info_path.as_str(), false, false, false, true, false, false);
    let p_expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, true);

    let n_max_tmp1 = get_max_n_tmp1_c(p_expressions_bin);
    let n_max_tmp3 = get_max_n_tmp3_c(p_expressions_bin);
    set_memory_expressions_c(p_stark_info, n_max_tmp1, n_max_tmp3);

    let n_publics = proof[0];
    let proof = &proof[1..];
    let (publics, vadcop_proof) = proof.split_at(n_publics as usize);

    let valid = stark_verify_c(
        &verkey_path,
        vadcop_proof.as_ptr() as *mut u64,
        p_stark_info,
        p_expressions_bin,
        publics.as_ptr() as *mut u8,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );

    // Clean up allocated resources
    expressions_bin_free_c(p_expressions_bin);
    stark_info_free_c(p_stark_info);

    timer_stop_and_log_info!(VERIFY_FINAL_PROOF);
    valid
}

pub fn verify_proof_from_file<F: PrimeField64>(
    proof_file: String,
    stark_info_path: String,
    expressions_bin_path: String,
    verkey_path: String,
    publics: Option<Vec<F>>,
    proof_values: Option<Vec<F>>,
    challenges: Option<Vec<F>>,
) -> bool {
    let p_stark_info = stark_info_new_c(stark_info_path.as_str(), false, false, false, true, false, false);
    let p_expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, true);

    let n_max_tmp1 = get_max_n_tmp1_c(p_expressions_bin);
    let n_max_tmp3 = get_max_n_tmp3_c(p_expressions_bin);
    set_memory_expressions_c(p_stark_info, n_max_tmp1, n_max_tmp3);

    let proof_challenges_ptr = match challenges {
        Some(ref challenges) => challenges.as_ptr() as *mut u8,
        None => std::ptr::null_mut(),
    };

    let publics_ptr = match publics {
        Some(ref publics) => publics.as_ptr() as *mut u8,
        None => std::ptr::null_mut(),
    };

    let proof_values_ptr = match proof_values {
        Some(ref proof_values) => proof_values.as_ptr() as *mut u8,
        None => std::ptr::null_mut(),
    };

    let result = stark_verify_from_file_c(
        &verkey_path,
        &proof_file,
        p_stark_info,
        p_expressions_bin,
        publics_ptr,
        proof_values_ptr,
        proof_challenges_ptr,
    );

    // Clean up allocated resources
    expressions_bin_free_c(p_expressions_bin);
    stark_info_free_c(p_stark_info);

    result
}

pub fn verify_proof<F: PrimeField64>(
    p_proof: *mut u64,
    stark_info_path: String,
    expressions_bin_path: String,
    verkey_path: String,
    publics: Option<Vec<F>>,
    proof_values: Option<Vec<F>>,
    global_challenge: Option<Vec<F>>,
) -> bool {
    let p_stark_info = stark_info_new_c(stark_info_path.as_str(), false, false, false, true, false, false);
    let p_expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, true);

    let n_max_tmp1 = get_max_n_tmp1_c(p_expressions_bin);
    let n_max_tmp3 = get_max_n_tmp3_c(p_expressions_bin);
    set_memory_expressions_c(p_stark_info, n_max_tmp1, n_max_tmp3);

    let global_challenge_ptr = match global_challenge {
        Some(ref global_challenge) => global_challenge.as_ptr() as *mut u8,
        None => std::ptr::null_mut(),
    };

    let publics_ptr = match publics {
        Some(ref publics) => publics.as_ptr() as *mut u8,
        None => std::ptr::null_mut(),
    };

    let proof_values_ptr = match proof_values {
        Some(ref proof_values) => proof_values.as_ptr() as *mut u8,
        None => std::ptr::null_mut(),
    };

    let result = stark_verify_c(
        &verkey_path,
        p_proof,
        p_stark_info,
        p_expressions_bin,
        publics_ptr,
        proof_values_ptr,
        global_challenge_ptr,
    );

    // Clean up allocated resources
    expressions_bin_free_c(p_expressions_bin);
    stark_info_free_c(p_stark_info);

    result
}

pub fn verify_proof_bn128<F: PrimeField64>(
    p_proof: *mut c_void,
    stark_info_path: String,
    expressions_bin_path: String,
    verkey_path: String,
    publics: Option<Vec<F>>,
) -> bool {
    let p_stark_info = stark_info_new_c(stark_info_path.as_str(), false, false, false, true, false, false);
    let p_expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, true);

    let n_max_tmp1 = get_max_n_tmp1_c(p_expressions_bin);
    let n_max_tmp3 = get_max_n_tmp3_c(p_expressions_bin);
    set_memory_expressions_c(p_stark_info, n_max_tmp1, n_max_tmp3);

    let publics_ptr = match publics {
        Some(ref publics) => publics.as_ptr() as *mut u8,
        None => std::ptr::null_mut(),
    };

    let result = stark_verify_bn128_c(&verkey_path, p_proof, p_stark_info, p_expressions_bin, publics_ptr);

    // Clean up allocated resources
    expressions_bin_free_c(p_expressions_bin);
    stark_info_free_c(p_stark_info);

    result
}

pub fn verify_basic_proof<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    instance_id: usize,
    proof: &[u64],
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let mut is_valid = true;

    let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
    let air_instance_id = pctx.dctx_find_air_instance_id(instance_id)?;

    let setup_path = pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Basic);

    let stark_info_path = setup_path.display().to_string() + ".starkinfo.json";
    let expressions_bin_path = setup_path.display().to_string() + ".verifier.bin";
    let verkey_path = setup_path.display().to_string() + ".verkey.json";
    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;

    tracing::info!(
        "    Verifying proof of {}: Instance #{} with global index #{}",
        air_name,
        air_instance_id,
        instance_id
    );

    let is_valid_proof = verify_proof(
        proof.as_ptr() as *mut u64,
        stark_info_path,
        expressions_bin_path,
        verkey_path,
        Some(pctx.get_publics().clone()),
        Some(pctx.get_proof_values().clone()),
        Some(pctx.get_global_challenge().clone()),
    );

    if !is_valid_proof {
        is_valid = false;
        tracing::info!(
            "··· {}",
            format!("\u{2717} Proof of {air_name}: Instance #{air_instance_id} was not verified",).bright_red().bold()
        );
    } else {
        tracing::info!(
            "    {}",
            format!("\u{2713} Proof of {air_name}: Instance #{air_instance_id} was verified",).bright_green().bold()
        );
    }

    Ok(is_valid)
}
