#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use ::std::os::raw::c_void;

#[cfg(feature = "no_lib_link")]
use log::trace;

#[cfg(not(feature = "no_lib_link"))]
include!("../bindings_starks.rs");

#[cfg(not(feature = "no_lib_link"))]
use std::ffi::CString;

#[cfg(not(feature = "no_lib_link"))]
pub fn save_proof_c<T>(
    p_stark_info: *mut ::std::os::raw::c_void,
    p_fri_proof: *mut ::std::os::raw::c_void,
    public_inputs: &[T],
    public_outputs_file: &str,
    file_prefix: &str,
) {
    unsafe {
        let public_outputs_file = CString::new(public_outputs_file).unwrap();
        let file_prefix = CString::new(file_prefix).unwrap();

        save_proof(
            p_stark_info,
            p_fri_proof,
            public_inputs.len() as std::os::raw::c_ulong,
            public_inputs.as_ptr() as *mut std::os::raw::c_void,
            public_outputs_file.as_ptr() as *mut std::os::raw::c_char,
            file_prefix.as_ptr() as *mut std::os::raw::c_char,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_new_c(p_stark: *mut c_void) -> *mut c_void {
    unsafe { fri_proof_new(p_stark) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_get_root_c(pFriProof: *mut c_void, root_index: u64, root_subindex: u64) -> *mut c_void {
    unsafe { fri_proof_get_root(pFriProof, root_index, root_subindex) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_get_tree_root_c(pFriProof: *mut c_void, tree_index: u64, root_index: u64) -> *mut c_void {
    unsafe { fri_proof_get_tree_root(pFriProof, tree_index, root_index) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_free_c(p_zkevm_steps: *mut c_void) {
    unsafe {
        fri_proof_free(p_zkevm_steps);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn config_new_c(config_filename: &str) -> *mut std::os::raw::c_void {
    unsafe {
        let config_filename = CString::new(config_filename).unwrap();

        config_new(config_filename.as_ptr() as *mut std::os::raw::c_char)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn config_free_c(pConfig: *mut c_void) {
    unsafe {
        config_free(pConfig);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_info_new_c(filename: &str) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();

        starkinfo_new(filename.as_ptr() as *mut std::os::raw::c_char)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_map_totaln_c(p_stark_info: *mut ::std::os::raw::c_void) -> u64 {
    unsafe { get_mapTotalN(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_map_offsets_c(pStarkInfo: *mut c_void, stage: &str, flag: bool) -> u64 {
    let stage = CString::new(stage).unwrap();
    unsafe { get_map_offsets(pStarkInfo, stage.as_ptr() as *mut std::os::raw::c_char, flag) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_map_sections_n_c(pStarkInfo: *mut ::std::os::raw::c_void, stage: &str) -> u64 {
    let stage = CString::new(stage).unwrap();
    unsafe { get_map_sections_n(pStarkInfo, stage.as_ptr() as *mut std::os::raw::c_char) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_info_free_c(p_stark_info: *mut c_void) {
    unsafe {
        starkinfo_free(p_stark_info);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn starks_new_c(
    p_config: *mut c_void,
    stark_info: *mut c_void,
    chelpers: *mut c_void,
    const_pols: *mut c_void,
    p_address: *mut c_void,
) -> *mut c_void {
    unsafe { starks_new(p_config, stark_info, chelpers, const_pols, p_address) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn starks_new_default_c(
    stark_info: *mut c_void,
    chelpers: *mut c_void,
    const_pols: *mut c_void,
    p_address: *mut c_void,
) -> *mut c_void {
    unsafe { starks_new_default(stark_info, chelpers, const_pols, p_address) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_stark_info_c(p_stark: *mut c_void) -> *mut c_void {
    unsafe { get_stark_info(p_stark) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn starks_free_c(p_stark: *mut c_void) {
    unsafe {
        starks_free(p_stark);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn chelpers_new_c(chelpers_filename: &str) -> *mut ::std::os::raw::c_void {
    let chelpers_filename = CString::new(chelpers_filename).unwrap();

    unsafe { chelpers_new(chelpers_filename.as_ptr() as *mut std::os::raw::c_char) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn const_pols_new_c(p_stark_info: *mut c_void, _const_pols_filename: &str) -> *mut ::std::os::raw::c_void {
    let const_pols_f = CString::new(_const_pols_filename).unwrap();

    unsafe { const_pols_new(p_stark_info, const_pols_f.as_ptr() as *mut std::os::raw::c_char) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn chelpers_free_c(p_chelpers: *mut ::std::os::raw::c_void) {
    unsafe {
        chelpers_free(p_chelpers);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn init_hints_c() {
    unsafe {
        init_hints();
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn steps_params_new_c(
    p_pols: *mut c_void,
    p_const_pols: *mut c_void,
    p_challenges: *mut c_void,
    p_subproof_values: *mut c_void,
    p_evals: *mut c_void,
    p_public_inputs: *mut c_void,
) -> *mut c_void {
    unsafe { steps_params_new(p_pols, p_const_pols, p_challenges, p_subproof_values, p_evals, p_public_inputs) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn steps_params_free_c(p_steps_params: *mut c_void) {
    unsafe {
        steps_params_free(p_steps_params);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn extend_and_merkelize_c(p_stark: *mut c_void, step: u64, p_chelpers_steps: *mut c_void, proof: *mut c_void) {
    unsafe {
        extend_and_merkelize(p_stark, step, p_chelpers_steps, proof);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn treesGL_get_root_c(pStark: *mut c_void, index: u64, root: *mut c_void) {
    unsafe {
        treesGL_get_root(pStark, index, root);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_stage_expressions_c(
    p_starks: *mut ::std::os::raw::c_void,
    element_type: u32,
    step: u64,
    p_chelpers_steps: *mut ::std::os::raw::c_void,
    p_proof: *mut ::std::os::raw::c_void,
) {
    unsafe {
        compute_stage_expressions(p_starks, element_type, step, p_chelpers_steps, p_proof);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_impols_expressions_c(
    p_starks: *mut ::std::os::raw::c_void,
    id: u64,
    p_chelpers_steps: *mut ::std::os::raw::c_void,
) {
    unsafe {
        calculate_impols_expressions(p_starks, id, p_chelpers_steps);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_quotient_polynomial_c(
    p_starks: *mut ::std::os::raw::c_void,
    p_chelpers_steps: *mut ::std::os::raw::c_void,
) {
    unsafe {
        calculate_quotient_polynomial(p_starks, p_chelpers_steps);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn commit_stage_c(
    p_starks: *mut ::std::os::raw::c_void,
    element_type: u32,
    step: u64,
    p_chelpers_steps: *mut ::std::os::raw::c_void,
    p_proof: *mut ::std::os::raw::c_void,
) {
    unsafe {
        commit_stage(p_starks, element_type, step, p_chelpers_steps, p_proof);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_evals_c(p_stark: *mut c_void, p_chelpers_steps: *mut c_void, pProof: *mut c_void) {
    unsafe {
        compute_evals(p_stark, p_chelpers_steps, pProof);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_fri_pol_c(
    p_stark: *mut c_void,
    step: u64,
    p_chelpers_steps: *mut c_void,
) -> *mut c_void {
    unsafe { compute_fri_pol(p_stark, step, p_chelpers_steps) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_fri_pol_c(p_stark: *mut c_void, p_chelpers_steps: *mut c_void) -> *mut ::std::os::raw::c_void {
    unsafe { get_fri_pol(p_stark, p_chelpers_steps) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_fri_folding_c(
    p_stark: *mut c_void,
    step: u64,
    p_chelpers_steps: *mut c_void,
    challenge: *mut c_void,
    pProof: *mut c_void,
) {
    unsafe {
        compute_fri_folding(p_stark, step, p_chelpers_steps, challenge, pProof);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_fri_queries_c(p_stark: *mut c_void, p_proof: *mut c_void, p_fri_queries: *mut u64) {
    unsafe {
        compute_fri_queries(p_stark, p_proof, p_fri_queries);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_proof_root_c(
    p_proof: *mut ::std::os::raw::c_void,
    stage_id: u64,
    index: u64,
) -> *mut ::std::os::raw::c_void {
    unsafe { get_proof_root(p_proof, stage_id, index) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn resize_vector_c(p_vector: *mut c_void, new_size: u64, value: bool) {
    unsafe {
        resize_vector(p_vector, new_size, value);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_bool_vector_value_c(p_vector: *mut c_void, index: u64, value: bool) {
    unsafe {
        set_bool_vector_value(p_vector, index, value);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_hash_c(pStarks: *mut c_void, pHhash: *mut c_void, pBuffer: *mut c_void, nElements: u64) {
    unsafe {
        calculate_hash(pStarks, pHhash, pBuffer, nElements);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn commit_pols_starks_new_c(
    p_address: *mut c_void,
    degree: u64,
    n_committed_pols: u64,
) -> *mut std::os::raw::c_void {
    unsafe { commit_pols_starks_new(p_address, degree, n_committed_pols) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn commit_pols_starks_free_c(p_commit_pols_starks: *mut c_void) {
    unsafe {
        commit_pols_starks_free(p_commit_pols_starks);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn circom_get_commited_pols_c(
    p_commit_pols_starks: *mut c_void,
    zkevm_verifier: &str,
    exec_file: &str,
    zkin: *mut c_void,
    n: u64,
    n_cols: u64,
) {
    unsafe {
        let zkevm_verifier = CString::new(zkevm_verifier).unwrap();
        let exec_file = CString::new(exec_file).unwrap();

        circom_get_commited_pols(
            p_commit_pols_starks,
            zkevm_verifier.as_ptr() as *mut std::os::raw::c_char,
            exec_file.as_ptr() as *mut std::os::raw::c_char,
            zkin,
            n,
            n_cols,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn circom_recursive1_get_commited_pols_c(
    p_commit_pols_starks: *mut c_void,
    zkevm_verifier: &str,
    exec_file: &str,
    zkin: *mut c_void,
    n: u64,
    n_cols: u64,
) {
    unsafe {
        let zkevm_verifier = CString::new(zkevm_verifier).unwrap();
        let exec_file = CString::new(exec_file).unwrap();

        circom_recursive1_get_commited_pols(
            p_commit_pols_starks,
            zkevm_verifier.as_ptr() as *mut std::os::raw::c_char,
            exec_file.as_ptr() as *mut std::os::raw::c_char,
            zkin,
            n,
            n_cols,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn zkin_new_c<T>(
    p_stark_info: *mut c_void,
    p_fri_proof: *mut c_void,
    public_inputs: &[T],
    root_c: &[T],
) -> *mut c_void {
    unsafe {
        zkin_new(
            p_stark_info,
            p_fri_proof,
            public_inputs.len() as std::os::raw::c_ulong,
            public_inputs.as_ptr() as *mut std::os::raw::c_void,
            root_c.len() as std::os::raw::c_ulong,
            root_c.as_ptr() as *mut std::os::raw::c_void,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn transcript_new_c(element_type: u32, arity: u64, custom: bool) -> *mut ::std::os::raw::c_void {
    unsafe { transcript_new(element_type, arity, custom) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn transcript_add_c(p_transcript: *mut c_void, p_input: *mut c_void, size: u64) {
    unsafe {
        transcript_add(p_transcript, p_input, size);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn transcript_add_polinomial_c(p_transcript: *mut c_void, p_polinomial: *mut c_void) {
    unsafe {
        transcript_add_polinomial(p_transcript, p_polinomial);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn transcript_free_c(p_transcript: *mut ::std::os::raw::c_void, type_: u32) {
    unsafe {
        transcript_free(p_transcript, type_);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_challenge_c(p_starks: *mut c_void, p_transcript: *mut c_void, p_element: *mut c_void) {
    unsafe {
        get_challenge(p_starks, p_transcript, p_element);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_permutations_c(p_transcript: *mut c_void, res: *mut u64, n: u64, n_bits: u64) {
    unsafe {
        get_permutations(p_transcript, res, n, n_bits);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn polinomial_new_c(degree: u64, dim: u64, name: &str) -> *mut c_void {
    unsafe {
        let name = CString::new(name).unwrap();

        polinomial_new(degree, dim, name.as_ptr() as *mut std::os::raw::c_char)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn polinomial_get_p_element_c(p_polinomial: *mut c_void, index: u64) -> *mut c_void {
    unsafe { polinomial_get_p_element(p_polinomial, index) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn polinomial_free_c(p_polinomial: *mut c_void) {
    unsafe {
        polinomial_free(p_polinomial);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn goldilocks_linear_hash_c(pInput: *mut c_void, pOutput: *mut c_void) {
    unsafe {
        goldilocks_linear_hash(pInput, pOutput);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn chelpers_steps_new_c(p_stark_info: *mut c_void, p_chelpers: *mut c_void, p_params: *mut c_void) -> *mut c_void {
    unsafe { chelpers_steps_new(p_stark_info, p_chelpers, p_params) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_commit_calculated_c(p_chelpers_steps: *mut c_void, id: u64) {
    unsafe { set_commit_calculated(p_chelpers_steps, id) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn can_stage_be_calculated_c(p_chelpers_steps: *mut c_void, step: u64) {
    unsafe { can_stage_be_calculated(p_chelpers_steps, step) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn can_impols_be_calculated_c(p_chelpers_steps: *mut c_void, step: u64) {
    unsafe { can_impols_be_calculated(p_chelpers_steps, step) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn chelpers_steps_free_c(p_chelpers_steps: *mut c_void) {
    unsafe {
        chelpers_steps_free(p_chelpers_steps);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_hint_field_c(p_chelpers_steps: *mut c_void, values: *mut c_void, hint_id: u64, hint_field_name: &str) {
    unsafe {
        let field_name = CString::new(hint_field_name).unwrap();
        set_hint_field(p_chelpers_steps, values, hint_id, field_name.as_ptr() as *mut std::os::raw::c_char);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_c(p_chelpers_steps: *mut c_void, hint_id: u64, hint_field_name: &str, dest: bool) -> *mut c_void {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe { get_hint_field(p_chelpers_steps, hint_id, field_name.as_ptr() as *mut std::os::raw::c_char, dest) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn verify_constraints_c(
    p_chelpers_steps: *mut c_void,
    stage_id: u64,
) -> bool {
    unsafe { verify_constraints(p_chelpers_steps, stage_id) }
}

// ------------------------
// MOCK METHODS FOR TESTING
// ------------------------
#[cfg(feature = "no_lib_link")]
pub fn save_proof_c<T>(
    _p_stark_info: *mut ::std::os::raw::c_void,
    _p_fri_proof: *mut ::std::os::raw::c_void,
    _public_inputs: &[T],
    _public_outputs_file: &str,
    _file_prefix: &str,
) {
    trace!("{}: ··· {}", "ffi     ", "save_proof: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_new_c(_p_starks: *mut c_void) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "fri_proof_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_get_root_c(_pFriProof: *mut c_void, _root_index: u64, _root_subindex: u64) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "fri_proof_get_root: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_get_tree_root_c(_pFriProof: *mut c_void, _tree_index: u64, _root_index: u64) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "fri_proof_get_tree_root: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_free_c(_p_zkevm_steps: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "fri_proof_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn config_new_c(config_filename: &str) -> *mut std::os::raw::c_void {
    trace!(
        "{}: ··· {} {}",
        "ffi     ",
        "config_new: This is a mock call because there is no linked library.",
        config_filename
    );
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn config_free_c(_pConfig: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "config_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn stark_info_new_c(_filename: &str) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "starkinfo_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_map_totaln_c(_p_stark_info: *mut ::std::os::raw::c_void) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_map_totaln_c: This is a mock call because there is no linked library");
    10000
}

#[cfg(feature = "no_lib_link")]
pub fn get_map_offsets_c(_p_stark_info: *mut c_void, _stage: &str, _flag: bool) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_map_offsets: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn get_map_sections_n_c(_p_stark_info: *mut ::std::os::raw::c_void, _stage: &str) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_map_sections_n: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn stark_info_free_c(_p_stark_info: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "starkinfo_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn starks_new_c(
    _p_config: *mut c_void,
    _const_pols: &str,
    _map_const_pols_file: bool,
    _constants_tree: &str,
    _stark_info: *mut c_void,
    _chelpers: *mut c_void,
    _p_address: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "starks_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn starks_new_default_c(
    _stark_info: *mut c_void,
    _chelpers: *mut c_void,
    _const_pols: *mut c_void,
    _p_address: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "starks_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_stark_info_c(_p_stark: *mut c_void) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "get_stark_info: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn starks_free_c(_p_stark: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "starks_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn chelpers_new_c(_chelpers_filename: &str) -> *mut ::std::os::raw::c_void {
    trace!("{}: ··· {}", "ffi     ", "chelpers_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn const_pols_new_c(_p_stark_info: *mut c_void, _const_pols_filename: &str) -> *mut ::std::os::raw::c_void {
    trace!("{}: ··· {}", "mckzkevm", "const_pols_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn chelpers_free_c(_p_chelpers: *mut ::std::os::raw::c_void) {
    trace!("{}: ··· {}", "ffi     ", "chelpers_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn init_hints_c() {
    trace!("{}: ··· {}", "ffi     ", "init_hints: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn steps_params_new_c(
    _p_stark: *mut c_void,
    _p_const_pols: *mut c_void,
    _p_challenges: *mut c_void,
    _p_subproof_values: *mut c_void,
    _p_evals: *mut c_void,
    _p_public_inputs: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "steps_params_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_steps_params_field_c(_p_steps_params: *mut c_void, _name: &str) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "get_steps_params_field: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn steps_params_free_c(_p_stepsParams: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "steps_params_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn extend_and_merkelize_c(_p_stark: *mut c_void, _step: u64, _p_chelpers_steps: *mut c_void, _proof: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "extend_and_merkelize: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn treesGL_get_root_c(_pStark: *mut c_void, _index: u64, _root: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "treesGL_get_root: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_stage_c(
    _p_starks: *mut ::std::os::raw::c_void,
    _element_type: u32,
    _step: u64,
    _p_chelpers_steps: *mut ::std::os::raw::c_void,
    _p_proof: *mut ::std::os::raw::c_void,
    _p_transcript: *mut ::std::os::raw::c_void,
) {
    trace!("{}: ··· {}", "ffi     ", "compute_stage: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_stage_expressions_c(
    _p_starks: *mut ::std::os::raw::c_void,
    _element_type: u32,
    _step: u64,
    _p_chelpers_steps: *mut ::std::os::raw::c_void,
    _p_proof: *mut ::std::os::raw::c_void,
) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "compute_stage_expressions: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_impols_expressions_c(
    _p_starks: *mut ::std::os::raw::c_void,
    _id: u64,
    _p_chelpers_steps: *mut ::std::os::raw::c_void,
) {
    trace!(
        "{}: ··· {}",
        "mckzkevm",
        "calculate_impols_expression: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_quotient_polynomial_c(
    _p_starks: *mut ::std::os::raw::c_void,
    _p_chelpers_steps: *mut ::std::os::raw::c_void,
) {
    trace!("mckzkevm: ··· {}", "calculate_quotient_polynomial: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn commit_stage_c(
    _p_starks: *mut ::std::os::raw::c_void,
    _element_type: u32,
    _step: u64,
    _p_chelpers_steps: *mut ::std::os::raw::c_void,
    _p_proof: *mut ::std::os::raw::c_void,
) {
    trace!("{}: ··· {}", "ffi     ", "commit_stage: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_evals_c(_p_stark: *mut c_void, _p_chelpers_steps: *mut c_void, _pProof: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "compute_evals: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_fri_pol_c(
    _p_stark: *mut c_void,
    _step: u64,
    _p_chelpers_steps: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "compute_fri_pol: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_fri_pol_c(_p_stark: *mut c_void, _p_chelpers_steps: *mut c_void) -> *mut ::std::os::raw::c_void {
    trace!("ffi     : ··· {}", "get_fri_pol: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn compute_fri_folding_c(
    _p_stark: *mut c_void,
    _step: u64,
    _p_chelpers_steps: *mut c_void,
    _challenge: *mut c_void,
    _pProof: *mut c_void,
) {
    trace!("{}: ··· {}", "ffi     ", "compute_fri_folding: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_fri_queries_c(_p_stark: *mut c_void, _p_proof: *mut c_void, _p_fri_queries: *mut u64) {
    trace!("{}: ··· {}", "ffi     ", "compute_fri_queries: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_proof_root_c(
    _p_proof: *mut ::std::os::raw::c_void,
    _stage_id: u64,
    _index: u64,
) -> *mut ::std::os::raw::c_void {
    trace!("{}: ··· {}", "ffi     ", "get_proof_root: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn resize_vector_c(_p_vector: *mut c_void, _new_size: u64, _value: bool) {
    trace!("{}: ··· {}", "ffi     ", "resize_vector: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn set_bool_vector_value_c(_p_vector: *mut c_void, _index: u64, _value: bool) {
    trace!("{}: ··· {}", "ffi     ", "set_bool_vector_value: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_hash_c(_pStarks: *mut c_void, _pHhash: *mut c_void, _pBuffer: *mut c_void, _nElements: u64) {
    trace!("{}: ··· {}", "ffi     ", "calculate_hash: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn commit_pols_starks_new_c(
    _p_address: *mut c_void,
    _degree: u64,
    _n_committed_pols: u64,
) -> *mut std::os::raw::c_void {
    trace!("{}: ··· {}", "ffi     ", "commit_pols_starks_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn commit_pols_starks_free_c(_p_commit_pols_starks: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "commit_pols_starks_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn circom_get_commited_pols_c(
    _p_commit_pols_starks: *mut c_void,
    _zkevm_verifier: &str,
    _exec_file: &str,
    _zkin: *mut c_void,
    _n: u64,
    _n_cols: u64,
) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "circom_get_commited_pols: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn circom_recursive1_get_commited_pols_c(
    _p_commit_pols_starks: *mut c_void,
    _zkevm_verifier: &str,
    _exec_file: &str,
    _zkin: *mut c_void,
    _n: u64,
    _n_cols: u64,
) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "circom_recursive1_get_commited_pols: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn zkin_new_c<T>(
    _p_stark_info: *mut c_void,
    _p_fri_proof: *mut c_void,
    _public_inputs: &[T],
    _root_c: &[T],
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "zkin_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn transcript_new_c(_element_type: u32, _arity: u64, _custom: bool) -> *mut ::std::os::raw::c_void {
    trace!("{}: ··· {}", "ffi     ", "transcript_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn transcript_add_c(_p_transcript: *mut c_void, _p_input: *mut c_void, _size: u64) {
    trace!("{}: ··· {}", "ffi     ", "transcript_add: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn transcript_add_polinomial_c(_p_transcript: *mut c_void, _p_polinomial: *mut c_void) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "transcript_add_polinomial: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn transcript_free_c(_p_transcript: *mut ::std::os::raw::c_void, _element_type: u32) {
    trace!("{}: ··· {}", "ffi     ", "transcript_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_challenge_c(_p_starks: *mut c_void, _p_transcript: *mut c_void, _p_element: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "get_challenges: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_permutations_c(_p_transcript: *mut c_void, _res: *mut u64, _n: u64, _n_bits: u64) {
    trace!("{}: ··· {}", "ffi     ", "get_permutations: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn polinomial_new_c(_degree: u64, _dim: u64, _name: &str) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "polinomial_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn polinomial_get_p_element_c(_p_polinomial: *mut c_void, _index: u64) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "get_p_element: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn polinomial_free_c(_p_polinomial: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "polinomial_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn goldilocks_linear_hash_c(_p_input: *mut c_void, _p_output: *mut c_void) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "goldilocks_linear_hash_c: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn chelpers_steps_new_c(
    _p_stark_info: *mut c_void,
    _p_chelpers: *mut c_void,
    _p_params: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "chelpers_steps_new_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn set_commit_calculated_c(_p_chelpers_steps: *mut c_void, _id: u64) {
    trace!("{}: ··· {}", "ffi     ", "set_commit_calculated_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn can_stage_be_calculated_c(_p_chelpers_steps: *mut c_void, _step: u64) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "can_stage_be_calculated_c: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn can_impols_be_calculated_c(_p_chelpers_steps: *mut c_void, _step: u64) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "can_impols_be_calculated_c: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn chelpers_steps_free_c(_p_chelpers_steps: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "chelpers_steps_free_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn set_hint_field_c(_p_chelpers_steps: *mut c_void, _values: *mut c_void, _hint_id: u64, _hint_field_name: &str) {
    trace!("{}: ··· {}", "ffi     ", "set_hint_field_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_c(
    _p_chelpers_steps: *mut c_void,
    _hint_id: u64,
    _hint_field_name: &str,
    _dest: bool,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "get_hint_field_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn verify_constraints_c(
    _p_chelpers_steps: *mut c_void,
    _stage_id: u64,
) -> bool {
    trace!("{}: ··· {}", "ffi     ", "verify_constraints_c: This is a mock call because there is no linked library");
    true
}
