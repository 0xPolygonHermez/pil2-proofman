use proofman_fields::PrimeField64;

use proofman_starks_lib_c::{
    expressions_bin_free_c, expressions_bin_new_c, get_max_n_tmp1_c, get_max_n_tmp3_c, set_memory_expressions_c,
    get_n_publics_c, get_proof_size_c, stark_info_free_c, stark_info_new_c, stark_verify_bn128_c, stark_verify_c,
    stark_verify_from_file_c,
};

use colored::*;

use proofman_common::{ProofCtx, ProofType, ProofmanResult, Setup};

use std::os::raw::c_void;

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

/// Verify one proof against the setup files at the given paths.
///
/// `proof` is the proof body only (no publics prefix); its length and `publics`' are checked
/// against the setup before anything is handed to C++. `stark_verify` walks both through raw
/// pointers -- `pointer2json` reads `starkInfo.proofSize` words and `starkVerify` reads
/// `starkInfo.nPublics` field elements, neither with a bound of its own -- so a proof loaded from
/// an untrusted file (`proofman-cli verify-stark`) or received over the network must be measured
/// here or it reads off the end of the allocation.
pub fn verify_proof<F: PrimeField64>(
    proof: &[u64],
    stark_info_path: String,
    expressions_bin_path: String,
    verkey_path: String,
    publics: Option<Vec<F>>,
    proof_values: Option<Vec<F>>,
    global_challenge: Option<Vec<F>>,
) -> bool {
    let p_stark_info = stark_info_new_c(stark_info_path.as_str(), false, false, false, true, false, false);
    let p_expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, true);

    let proof_size = get_proof_size_c(p_stark_info) as usize;
    let n_publics = get_n_publics_c(p_stark_info) as usize;
    let n_publics_given = publics.as_ref().map_or(0, |p| p.len());
    // Short is a read past the end; long only means the caller handed us a pooled or padded
    // buffer, which the verifier ignores -- so only short is rejected.
    let too_short = if proof.len() < proof_size {
        Some(format!("proof has {} words but the setup's proofSize is {proof_size}", proof.len()))
    } else if n_publics_given < n_publics {
        Some(format!("{n_publics_given} publics were given but the setup declares {n_publics}"))
    } else {
        None
    };
    if let Some(reason) = too_short {
        tracing::error!(
            "··· {}",
            format!("\u{2717} Proof does not match {stark_info_path}: {reason}").bright_red().bold()
        );
        expressions_bin_free_c(p_expressions_bin);
        stark_info_free_c(p_stark_info);
        return false;
    }

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
        proof.as_ptr() as *mut u64,
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

pub fn verify_proof_bn128<F: PrimeField64>(p_proof: *mut c_void, setup: &Setup<F>, publics: Option<Vec<F>>) -> bool {
    let stark_info_path = setup.setup_path.display().to_string() + ".starkinfo.json";
    let expressions_bin_path = setup.setup_path.display().to_string() + ".verifier.bin";
    let verkey_path = setup.setup_path.display().to_string() + ".verkey.json";

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
) -> ProofmanResult<bool> {
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
        proof,
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
