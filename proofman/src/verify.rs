use proofman_fields::PrimeField64;

use proofman_starks_lib_c::{
    constraint_dag_free_c, constraint_dag_new_c, get_n_publics_c, get_proof_size_c, get_proof_values_size_c,
    stark_info_free_c, stark_info_new_c, stark_verify_bn128_c, stark_verify_c,
};

use colored::*;

use proofman_common::stark_info::check_key_version_file;
use proofman_common::{ProofCtx, ProofType, ProofmanResult, Setup};

use std::os::raw::c_void;

/// `<base>.{starkinfo,verifierinfo,verkey}.json`.
fn key_paths(base: &str) -> (String, String, String) {
    (format!("{base}.starkinfo.json"), format!("{base}.verifierinfo.json"), format!("{base}.verkey.json"))
}

fn refuse(stark_info_path: &str, reason: &str) -> bool {
    tracing::error!("··· {}", format!("\u{2717} Proof does not match {stark_info_path}: {reason}").bright_red().bold());
    false
}

/// Verify one proof against the setup `<base>.{starkinfo,verifierinfo,verkey}.json`.
///
/// `proof` is the proof body only (no publics prefix). C++ reads `proofSize` words, `nPublics`
/// publics and `proofValuesSize` proof values through raw pointers with no bound of its own, so
/// an untrusted proof is measured here first. Long buffers are accepted (pooled/padded).
pub fn verify_proof<F: PrimeField64>(
    proof: &[u64],
    base: &str,
    publics: Option<Vec<F>>,
    proof_values: Option<Vec<F>>,
    global_challenge: Option<Vec<F>>,
) -> bool {
    let (stark_info_path, verifier_info_path, verkey_path) = key_paths(base);
    // C++ `StarkInfo::load` exits the process on a stale key.
    if let Err(e) = check_key_version_file(std::path::Path::new(&stark_info_path)) {
        return refuse(&stark_info_path, &e.to_string());
    }
    let p_stark_info = stark_info_new_c(stark_info_path.as_str(), false, false, false, true, false);
    let p_constraint_dag = constraint_dag_new_c(verifier_info_path.as_str(), p_stark_info);
    if p_constraint_dag.is_null() {
        stark_info_free_c(p_stark_info);
        return refuse(&verifier_info_path, "missing, pre-DAG or not this starkinfo's");
    }

    let proof_size = get_proof_size_c(p_stark_info) as usize;
    let n_publics = get_n_publics_c(p_stark_info) as usize;
    let proof_values_size = get_proof_values_size_c(p_stark_info) as usize;
    let n_publics_given = publics.as_ref().map_or(0, |p| p.len());
    let n_proof_values_given = proof_values.as_ref().map_or(0, |p| p.len());
    let too_short = if proof.len() < proof_size {
        Some(format!("proof has {} words but the setup's proofSize is {proof_size}", proof.len()))
    } else if n_publics_given < n_publics {
        Some(format!("{n_publics_given} publics were given but the setup declares {n_publics}"))
    } else if proof_values.is_some() && n_proof_values_given < proof_values_size {
        Some(format!("{n_proof_values_given} proof values were given but the setup declares {proof_values_size}"))
    } else {
        None
    };
    if let Some(reason) = too_short {
        constraint_dag_free_c(p_constraint_dag);
        stark_info_free_c(p_stark_info);
        return refuse(&stark_info_path, &reason);
    }

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
        p_constraint_dag,
        publics_ptr,
        proof_values_ptr,
        global_challenge_ptr,
    );

    // Clean up allocated resources
    constraint_dag_free_c(p_constraint_dag);
    stark_info_free_c(p_stark_info);

    result
}

pub fn verify_proof_bn128<F: PrimeField64>(p_proof: *mut c_void, setup: &Setup<F>, publics: Option<Vec<F>>) -> bool {
    let (stark_info_path, verifier_info_path, verkey_path) = key_paths(&setup.setup_path.display().to_string());

    let p_stark_info = stark_info_new_c(stark_info_path.as_str(), false, false, false, true, false);
    let p_constraint_dag = constraint_dag_new_c(verifier_info_path.as_str(), p_stark_info);
    if p_constraint_dag.is_null() {
        stark_info_free_c(p_stark_info);
        return false;
    }

    let publics_ptr = match publics {
        Some(ref publics) => publics.as_ptr() as *mut u8,
        None => std::ptr::null_mut(),
    };

    let result = stark_verify_bn128_c(&verkey_path, p_proof, p_stark_info, p_constraint_dag, publics_ptr);

    // Clean up allocated resources
    constraint_dag_free_c(p_constraint_dag);
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

    let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;

    tracing::info!(
        "    Verifying proof of {}: Instance #{} with global index #{}",
        air_name,
        air_instance_id,
        instance_id
    );

    let is_valid_proof = verify_proof(
        proof,
        &setup_path.display().to_string(),
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

#[cfg(test)]
mod tests {
    use super::verify_proof;
    use proofman_fields::Goldilocks;

    // C++ StarkInfo::load would exit the process on these; the Rust gate returns false first.
    #[test]
    fn stale_or_missing_key_is_rejected_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("air").display().to_string();
        assert!(!verify_proof::<Goldilocks>(&[], &base, None, None, None));
        std::fs::write(format!("{base}.starkinfo.json"), r#"{"keyVersion": 1}"#).unwrap();
        assert!(!verify_proof::<Goldilocks>(&[], &base, None, None, None));
        std::fs::write(format!("{base}.starkinfo.json"), "not json").unwrap();
        assert!(!verify_proof::<Goldilocks>(&[], &base, None, None, None));
    }
}
