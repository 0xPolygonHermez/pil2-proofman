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
pub fn save_challenges_c(p_challenges: *mut std::os::raw::c_void, global_info_file: &str, output_dir: &str) {
    unsafe {
        let file_dir = CString::new(output_dir).unwrap();
        let file_ptr = file_dir.as_ptr() as *mut std::os::raw::c_char;

        let global_info_file_name = CString::new(global_info_file).unwrap();
        let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

        save_challenges(p_challenges as *mut std::os::raw::c_void, global_info_file_ptr, file_ptr);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn save_publics_c(n_publics: u64, public_inputs: *mut std::os::raw::c_void, output_dir: &str) {
    let file_dir: CString = CString::new(output_dir).unwrap();
    unsafe {
        save_publics(n_publics, public_inputs, file_dir.as_ptr() as *mut std::os::raw::c_char);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_new_c(p_setup_ctx: *mut c_void) -> *mut c_void {
    unsafe { fri_proof_new(p_setup_ctx) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_get_tree_root_c(p_fri_proof: *mut c_void, root: *mut c_void, tree_index: u64) {
    unsafe {
        fri_proof_get_tree_root(p_fri_proof, root, tree_index);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_set_airgroup_values_c(p_fri_proof: *mut c_void, p_airgroup_values: *mut c_void) {
    unsafe { fri_proof_set_airgroupvalues(p_fri_proof, p_airgroup_values) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_set_air_values_c(p_fri_proof: *mut c_void, p_air_values: *mut c_void) {
    unsafe { fri_proof_set_airvalues(p_fri_proof, p_air_values) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_get_zkinproof_c(
    proof_id: u64,
    p_fri_proof: *mut c_void,
    p_publics: *mut c_void,
    p_challenges: *mut c_void,
    p_stark_info: *mut c_void,
    global_info_file: &str,
    output_dir_file: &str,
) -> *mut c_void {
    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    let file_dir = CString::new(output_dir_file).unwrap();
    let file_ptr = file_dir.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        fri_proof_get_zkinproof(
            proof_id,
            p_fri_proof,
            p_publics,
            p_challenges,
            p_stark_info,
            global_info_file_ptr,
            file_ptr,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_free_zkinproof_c(p_fri_proof: *mut c_void) {
    unsafe {
        fri_proof_free_zkinproof(p_fri_proof);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn fri_proof_free_c(p_fri_proof: *mut c_void) {
    unsafe {
        fri_proof_free(p_fri_proof);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn setup_ctx_new_c(
    p_stark_info: *mut c_void,
    p_expressions_bin: *mut c_void,
    p_const_pols: *mut c_void,
) -> *mut c_void {
    unsafe { setup_ctx_new(p_stark_info, p_expressions_bin, p_const_pols) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn setup_ctx_free_c(p_setup_ctx: *mut c_void) {
    unsafe {
        setup_ctx_free(p_setup_ctx);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_info_new_c(filename: &str) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();

        stark_info_new(filename.as_ptr() as *mut std::os::raw::c_char)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_stark_info_n_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_stark_info_n(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_stark_info_n_publics_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_stark_info_n_publics(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_map_totaln_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_map_total_n(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_n_airvals_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_n_airvals(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_n_airgroupvals_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_n_airgroupvals(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_n_evals_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_n_evals(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_airgroupval_id_by_name_c(p_stark_info: *mut c_void, name: &str) -> i64 {
    let airgroupval_name = CString::new(name).unwrap();
    unsafe { get_airgroupvalue_id_by_name(p_stark_info, airgroupval_name.as_ptr() as *mut std::os::raw::c_char) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_airval_id_by_name_c(p_stark_info: *mut c_void, name: &str) -> i64 {
    let airval_name = CString::new(name).unwrap();
    unsafe { get_airvalue_id_by_name(p_stark_info, airval_name.as_ptr() as *mut std::os::raw::c_char) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_map_offsets_c(pStarkInfo: *mut c_void, stage: &str, flag: bool) -> u64 {
    let stage = CString::new(stage).unwrap();
    unsafe { get_map_offsets(pStarkInfo, stage.as_ptr() as *mut std::os::raw::c_char, flag) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_info_free_c(p_stark_info: *mut c_void) {
    unsafe {
        stark_info_free(p_stark_info);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn const_pols_new_c(filename: &str, p_stark_info: *mut c_void, calculate_tree: bool) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();

        const_pols_new(filename.as_ptr() as *mut std::os::raw::c_char, p_stark_info, calculate_tree)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn const_pols_with_tree_new_c(filename: &str, tree_filename: &str, p_stark_info: *mut c_void) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();
        let tree_filename = CString::new(tree_filename).unwrap();

        const_pols_with_tree_new(
            filename.as_ptr() as *mut std::os::raw::c_char,
            tree_filename.as_ptr() as *mut std::os::raw::c_char,
            p_stark_info,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn load_const_tree_c(pConstPols: *mut c_void, pStarkInfo: *mut c_void, tree_filename: &str) {
    unsafe {
        let tree_filename = CString::new(tree_filename).unwrap();

        load_const_tree(pConstPols, pStarkInfo, tree_filename.as_ptr() as *mut std::os::raw::c_char);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_const_tree_c(pConstPols: *mut c_void, pStarkInfo: *mut c_void) {
    unsafe {
        calculate_const_tree(pConstPols, pStarkInfo);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn const_pols_free_c(p_const_pols: *mut c_void) {
    unsafe {
        const_pols_free(p_const_pols);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn expressions_bin_new_c(filename: &str, global: bool) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();

        expressions_bin_new(filename.as_ptr() as *mut std::os::raw::c_char, global)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn expressions_bin_free_c(p_expressions_bin: *mut c_void) {
    unsafe {
        expressions_bin_free(p_expressions_bin);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_ids_by_name_c(p_expressions_bin: *mut c_void, hint_name: &str) -> *mut c_void {
    let name = CString::new(hint_name).unwrap();
    unsafe { get_hint_ids_by_name(p_expressions_bin, name.as_ptr() as *mut std::os::raw::c_char) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut c_void,
    hint_id: u64,
    hint_field_name: &str,
    hint_options: *mut c_void,
) -> *mut c_void {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe {
        get_hint_field(
            p_setup_ctx,
            p_steps_params,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
            hint_options,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn mul_hint_fields_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut c_void,
    hint_id: u64,
    hint_field_dest: &str,
    hint_field_name1: &str,
    hint_field_name2: &str,
    hint_options1: *mut c_void,
    hint_options2: *mut c_void,
) -> u64 {
    let field_dest = CString::new(hint_field_dest).unwrap();
    let field_name1 = CString::new(hint_field_name1).unwrap();
    let field_name2 = CString::new(hint_field_name2).unwrap();

    unsafe {
        mul_hint_fields(
            p_setup_ctx,
            p_steps_params,
            hint_id,
            field_dest.as_ptr() as *mut std::os::raw::c_char,
            field_name1.as_ptr() as *mut std::os::raw::c_char,
            field_name2.as_ptr() as *mut std::os::raw::c_char,
            hint_options1,
            hint_options2,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn acc_hint_field_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut c_void,
    hint_id: u64,
    hint_field_dest: &str,
    hint_field_airgroupvalue: &str,
    hint_field_name: &str,
) -> *mut c_void {
    let field_dest = CString::new(hint_field_dest).unwrap();
    let field_airgroupvalue = CString::new(hint_field_airgroupvalue).unwrap();
    let field_name = CString::new(hint_field_name).unwrap();

    unsafe {
        acc_hint_field(
            p_setup_ctx,
            p_steps_params,
            hint_id,
            field_dest.as_ptr() as *mut std::os::raw::c_char,
            field_airgroupvalue.as_ptr() as *mut std::os::raw::c_char,
            field_name.as_ptr() as *mut std::os::raw::c_char,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn acc_mul_hint_fields_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut c_void,
    hint_id: u64,
    hint_field_dest: &str,
    hint_field_airgroupvalue: &str,
    hint_field_name1: &str,
    hint_field_name2: &str,
    hint_options1: *mut c_void,
    hint_options2: *mut c_void,
) -> *mut c_void {
    let field_dest = CString::new(hint_field_dest).unwrap();
    let field_airgroupvalue = CString::new(hint_field_airgroupvalue).unwrap();
    let field_name1 = CString::new(hint_field_name1).unwrap();
    let field_name2: CString = CString::new(hint_field_name2).unwrap();

    unsafe {
        acc_mul_hint_fields(
            p_setup_ctx,
            p_steps_params,
            hint_id,
            field_dest.as_ptr() as *mut std::os::raw::c_char,
            field_airgroupvalue.as_ptr() as *mut std::os::raw::c_char,
            field_name1.as_ptr() as *mut std::os::raw::c_char,
            field_name2.as_ptr() as *mut std::os::raw::c_char,
            hint_options1,
            hint_options2,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_hint_field_c(
    p_setup_ctx: *mut c_void,
    p_params: *mut c_void,
    values: *mut c_void,
    hint_id: u64,
    hint_field_name: &str,
) -> u64 {
    unsafe {
        let field_name = CString::new(hint_field_name).unwrap();
        set_hint_field(p_setup_ctx, p_params, values, hint_id, field_name.as_ptr() as *mut std::os::raw::c_char)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn starks_new_c(p_setup_ctx: *mut c_void) -> *mut c_void {
    unsafe { starks_new(p_setup_ctx) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn starks_free_c(p_stark: *mut c_void) {
    unsafe {
        starks_free(p_stark);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn merkle_tree_new_c(height: u64, width: u64, arity: u64, custom: bool) -> *mut c_void {
    unsafe { merkle_tree_new(height, width, arity, custom) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn merkle_tree_free_c(merkle_tree: *mut c_void) {
    unsafe {
        merkle_tree_free(merkle_tree);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn treesGL_get_root_c(pStark: *mut c_void, index: u64, root: *mut c_void) {
    unsafe {
        treesGL_get_root(pStark, index, root);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_xdivxsub_c(p_stark: *mut c_void, xi_challenge: *mut c_void, xdivxsub: *mut c_void) {
    unsafe {
        calculate_xdivxsub(p_stark, xi_challenge, xdivxsub);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_fri_pol_c(p_setup_ctx: *mut c_void, buffer: *mut c_void) -> *mut c_void {
    unsafe { get_fri_pol(p_setup_ctx, buffer) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_fri_polynomial_c(p_starks: *mut c_void, p_steps_params: *mut c_void) {
    unsafe {
        calculate_fri_polynomial(p_starks, p_steps_params);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_quotient_polynomial_c(p_starks: *mut c_void, p_steps_params: *mut c_void) {
    unsafe {
        calculate_quotient_polynomial(p_starks, p_steps_params);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_impols_expressions_c(p_starks: *mut c_void, step: u64, p_steps_params: *mut c_void) {
    unsafe {
        calculate_impols_expressions(p_starks, step, p_steps_params);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn commit_stage_c(
    p_starks: *mut c_void,
    element_type: u32,
    step: u64,
    buffer: *mut c_void,
    p_proof: *mut c_void,
    p_buff_helper: *mut c_void,
) {
    unsafe {
        commit_stage(p_starks, element_type, step, buffer, p_proof, p_buff_helper);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_lev_c(p_stark: *mut c_void, xi_challenge: *mut c_void, lev: *mut c_void) {
    unsafe {
        compute_lev(p_stark, xi_challenge, lev);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_evals_c(
    p_stark: *mut c_void,
    buffer: *mut c_void,
    lev: *mut c_void,
    evals: *mut c_void,
    pProof: *mut c_void,
) {
    unsafe {
        compute_evals(p_stark, buffer, lev, evals, pProof);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_fri_folding_c(
    step: u64,
    buffer: *mut c_void,
    challenge: *mut c_void,
    n_bits_ext: u64,
    prev_bits: u64,
    current_bits: u64,
) {
    unsafe {
        compute_fri_folding(step, buffer, challenge, n_bits_ext, prev_bits, current_bits);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_fri_merkelize_c(
    p_starks: *mut c_void,
    p_proof: *mut c_void,
    step: u64,
    buffer: *mut c_void,
    current_bits: u64,
    next_bits: u64,
) {
    unsafe {
        compute_fri_merkelize(p_starks, p_proof, step, buffer, current_bits, next_bits);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_queries_c(
    p_stark: *mut c_void,
    p_proof: *mut c_void,
    p_fri_queries: *mut u64,
    n_queries: u64,
    n_trees: u64,
) {
    unsafe {
        compute_queries(p_stark, p_proof, p_fri_queries, n_queries, n_trees);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn compute_fri_queries_c(
    p_starks: *mut c_void,
    p_proof: *mut c_void,
    p_fri_queries: *mut u64,
    n_queries: u64,
    step: u64,
    current_bits: u64,
) {
    unsafe {
        compute_fri_queries(p_starks, p_proof, p_fri_queries, n_queries, step, current_bits);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_fri_final_pol_c(p_proof: *mut c_void, buffer: *mut c_void, n_bits: u64) {
    unsafe {
        set_fri_final_pol(p_proof, buffer, n_bits);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_hash_c(pStarks: *mut c_void, pHhash: *mut c_void, pBuffer: *mut c_void, nElements: u64) {
    unsafe {
        calculate_hash(pStarks, pHhash, pBuffer, nElements);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_const_tree_c(pStarks: *mut c_void, pConstPols: *mut c_void) {
    unsafe {
        set_const_tree(pStarks, pConstPols);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn transcript_new_c(element_type: u32, arity: u64, custom: bool) -> *mut c_void {
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
pub fn transcript_free_c(p_transcript: *mut c_void, type_: u32) {
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
pub fn verify_constraints_c(p_setup: *mut c_void, p_steps_params: *mut c_void) -> *mut c_void {
    unsafe { verify_constraints(p_setup, p_steps_params) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn verify_global_constraints_c(
    p_global_constraints_bin: *mut c_void,
    publics: *mut c_void,
    airgroupvalues: *mut *mut c_void,
) -> bool {
    unsafe { verify_global_constraints(p_global_constraints_bin, publics, airgroupvalues) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_global_constraints_c(
    p_global_constraints_bin: *mut c_void,
    publics: *mut c_void,
    airgroupvalues: *mut *mut c_void,
    hint_id: u64,
    hint_field_name: &str,
    print_expression: bool,
) -> *mut c_void {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe {
        get_hint_field_global_constraints(
            p_global_constraints_bin,
            publics,
            airgroupvalues,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
            print_expression,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn print_by_name_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut c_void,
    name: &str,
    lengths: *mut u64,
    first_print_value: u64,
    last_print_value: u64,
    return_values: bool,
) -> *mut c_void {
    let name_string = CString::new(name).unwrap();
    let name_ptr = name_string.as_ptr() as *mut std::os::raw::c_char;
    unsafe {
        print_by_name(
            p_setup_ctx,
            p_steps_params,
            name_ptr,
            lengths,
            first_print_value,
            last_print_value,
            return_values,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn print_expression_c(
    p_setup_ctx: *mut c_void,
    pol: *mut c_void,
    dim: u64,
    first_print_value: u64,
    last_print_value: u64,
) {
    unsafe {
        print_expression(p_setup_ctx, pol, dim, first_print_value, last_print_value);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn print_row_c(p_setup_ctx: *mut c_void, buffer: *mut c_void, stage: u64, row: u64) {
    unsafe {
        print_row(p_setup_ctx, buffer, stage, row);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn gen_recursive_proof_c(
    p_setup_ctx: *mut c_void,
    p_address: *mut c_void,
    p_public_inputs: *mut c_void,
    proof_file: &str,
) -> *mut c_void {
    let proof_file_name = CString::new(proof_file).unwrap();
    let proof_file_ptr = proof_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe { gen_recursive_proof(p_setup_ctx, p_address, p_public_inputs, proof_file_ptr) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_zkin_ptr_c(zkin_file: &str) -> *mut c_void {
    let zkin_file_name = CString::new(zkin_file).unwrap();
    let zkin_file_ptr = zkin_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe { get_zkin_ptr(zkin_file_ptr) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn publics2zkin_c(
    p_zkin: *mut c_void,
    p_publics: *mut c_void,
    global_info_file: &str,
    airgroup_id: u64,
    is_aggregated: bool,
) -> *mut c_void {
    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe { public2zkin(p_zkin, p_publics, global_info_file_ptr, airgroup_id, is_aggregated) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn add_recursive2_verkey_c(p_zkin: *mut c_void, recursive2_verkey: &str) -> *mut c_void {
    let recursive2_verkey_name = CString::new(recursive2_verkey).unwrap();
    let recursive2_verkey_ptr = recursive2_verkey_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe { add_recursive2_verkey(p_zkin, recursive2_verkey_ptr) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn join_zkin_final_c(
    p_publics: *mut c_void,
    p_challenges: *mut c_void,
    global_info_file: &str,
    zkin_recursive2: *mut *mut c_void,
    stark_info_recursive2: *mut *mut c_void,
) -> *mut c_void {
    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe { join_zkin_final(p_publics, p_challenges, global_info_file_ptr, zkin_recursive2, stark_info_recursive2) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn join_zkin_recursive2_c(
    airgroup_id: u64,
    p_publics: *mut c_void,
    p_challenges: *mut c_void,
    global_info_file: &str,
    zkin1: *mut c_void,
    zkin2: *mut c_void,
    stark_info_recursive2: *mut c_void,
) -> *mut c_void {
    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        join_zkin_recursive2(
            global_info_file_ptr,
            airgroup_id,
            p_publics,
            p_challenges,
            zkin1,
            zkin2,
            stark_info_recursive2,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_log_level_c(level: u64) {
    unsafe {
        setLogLevel(level);
    }
}

// ------------------------
// MOCK METHODS FOR TESTING
// ------------------------
#[cfg(feature = "no_lib_link")]
pub fn save_challenges_c(_p_challenges: *mut std::os::raw::c_void, _global_info_file: &str, _output_dir: &str) {
    trace!("{}: ··· {}", "ffi     ", "save_challenges_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn save_publics_c(_n_publics: u64, _public_inputs: *mut c_void, _output_dir: &str) {
    trace!("{}: ··· {}", "ffi     ", "save_publics_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_new_c(_p_setup_ctx: *mut c_void) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "fri_proof_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_get_tree_root_c(_p_fri_proof: *mut c_void, _root: *mut c_void, _tree_index: u64) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "fri_proof_get_tree_root: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_set_airgroup_values_c(_p_fri_proof: *mut c_void, _p_params: *mut c_void) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "fri_proof_set_airgroup_values_c: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_set_air_values_c(_p_fri_proof: *mut c_void, _p_params: *mut c_void) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "fri_proof_set_air_values_c: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_get_zkinproof_c(
    _proof_id: u64,
    _p_fri_proof: *mut c_void,
    _p_publics: *mut c_void,
    _p_challenges: *mut c_void,
    _p_stark_info: *mut c_void,
    _global_info_file: &str,
    _output_dir_file: &str,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "fri_proof_get_zkinproof: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_free_zkinproof_c(_p_fri_proof: *mut c_void) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "fri_proof_free_zkinproof: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn fri_proof_free_c(_p_fri_proof: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "fri_proof_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn setup_ctx_new_c(
    _p_stark_info: *mut c_void,
    _p_expressions_bin: *mut c_void,
    _p_const_pols: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "setup_ctx_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn setup_ctx_new1_c(
    _stark_info_file: &str,
    _expressions_bin_file: &str,
    _const_pols_file: &str,
    _const_tree_file: &str,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "setup_ctx_new1: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn setup_ctx_free_c(_p_setup_ctx: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "starkinfo_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn stark_info_new_c(_filename: &str) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "starkinfo_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_stark_info_n_publics_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_stark_info_n_c: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_stark_info_n_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_stark_info_n_c: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_map_totaln_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_map_totaln_c: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_n_airvals_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_n_airvals_c: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_n_airgroupvals_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_n_airgroupvals_c: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_n_evals_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_n_evals_c: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_airgroupval_id_by_name_c(_p_stark_info: *mut c_void, _name: &str) -> i64 {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "get_airgroupval_id_by_name_c: This is a mock call because there is no linked library"
    );
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_airval_id_by_name_c(_p_stark_info: *mut c_void, _name: &str) -> i64 {
    trace!("{}: ··· {}", "ffi     ", "get_airval_id_by_name_c: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_map_offsets_c(_p_stark_info: *mut c_void, _stage: &str, _flag: bool) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_map_offsets: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn stark_info_free_c(_p_stark_info: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "starkinfo_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn const_pols_new_c(_filename: &str, _p_stark_info: *mut c_void, _calculate_tree: bool) -> *mut c_void {
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn const_pols_with_tree_new_c(_filename: &str, _tree_filename: &str, _p_stark_info: *mut c_void) -> *mut c_void {
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn load_const_tree_c(_pConstPols: *mut c_void, _pStarkInfo: *mut c_void, _tree_filename: &str) {
    trace!("{}: ··· {}", "ffi     ", "load_const_tree: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_const_tree_c(_pConstPols: *mut c_void, _pStarkInfo: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "calculate_const_tree: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn const_pols_free_c(_p_const_pols: *mut c_void) {}

#[cfg(feature = "no_lib_link")]
pub fn expressions_bin_new_c(_filename: &str, _global: bool) -> *mut c_void {
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn expressions_bin_free_c(_p_expressions_bin: *mut c_void) {}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_ids_by_name_c(_p_expressions_bin: *mut c_void, _hint_name: &str) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "get_hint_ids_by_name: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut c_void,
    _hint_id: u64,
    _hint_field_name: &str,
    _hint_options: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "get_hint_field: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn mul_hint_fields_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut c_void,
    _hint_id: u64,
    _hint_field_dest: &str,
    _hint_field_name1: &str,
    _hint_field_name2: &str,
    _hint_options1: *mut c_void,
    _hint_options2: *mut c_void,
) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "mul_hint_fields: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn acc_hint_field_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut c_void,
    _hint_id: u64,
    _hint_field_dest: &str,
    _hint_field_airgroupvalue: &str,
    _hint_field_name: &str,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "acc_hint_fields: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn acc_mul_hint_fields_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut c_void,
    _hint_id: u64,
    _hint_field_dest: &str,
    _hint_field_airgroupvalue: &str,
    _hint_field_name1: &str,
    _hint_field_name2: &str,
    _hint_options1: *mut c_void,
    _hint_options2: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "acc_mul_hint_fields: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn set_hint_field_c(
    _p_setup_ctx: *mut c_void,
    _p_params: *mut c_void,
    _values: *mut c_void,
    _hint_id: u64,
    _hint_field_name: &str,
) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "set_hint_field_c: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn starks_new_c(_p_config: *mut c_void) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "starks_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn starks_free_c(_p_stark: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "starks_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn merkle_tree_new_c(_height: u64, _width: u64, _arity: u64, _custom: bool) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "merkle_tree_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn merkle_tree_free_c(_merkle_tree: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "merkle_tree_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn treesGL_get_root_c(_pStark: *mut c_void, _index: u64, _root: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "treesGL_get_root: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_fri_polynomial_c(_p_starks: *mut c_void, _p_steps_params: *mut c_void) {
    trace!("mckzkevm: ··· {}", "calculate_fri_polynomial: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_quotient_polynomial_c(_p_starks: *mut c_void, _p_steps_params: *mut c_void) {
    trace!("mckzkevm: ··· {}", "calculate_quotient_polynomial: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_impols_expressions_c(_p_starks: *mut c_void, _step: u64, _p_steps_params: *mut c_void) {
    trace!(
        "{}: ··· {}",
        "mckzkevm",
        "calculate_impols_expression: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn commit_stage_c(
    _p_starks: *mut c_void,
    _element_type: u32,
    _step: u64,
    _buffer: *mut c_void,
    _p_proof: *mut c_void,
    _p_buff_helper: *mut c_void,
) {
    trace!("{}: ··· {}", "ffi     ", "commit_stage: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_lev_c(_p_stark: *mut c_void, _xi_challenge: *mut c_void, _lev: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "compute_lev_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_evals_c(
    _p_stark: *mut c_void,
    _buffer: *mut c_void,
    _lev: *mut c_void,
    _evals: *mut c_void,
    _pProof: *mut c_void,
) {
    trace!("{}: ··· {}", "ffi     ", "compute_evals: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_xdivxsub_c(_p_stark: *mut c_void, _xi_challenge: *mut c_void, _buffer: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "calculate_xdivxsub_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_fri_pol_c(_p_setup_ctx: *mut c_void, _buffer: *mut c_void) -> *mut c_void {
    trace!("ffi     : ··· {}", "get_fri_pol: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn compute_fri_folding_c(
    _step: u64,
    _buffer: *mut c_void,
    _challenge: *mut c_void,
    _n_bits_ext: u64,
    _prev_bits: u64,
    _current_bits: u64,
) {
    trace!("{}: ··· {}", "ffi     ", "compute_fri_folding: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_fri_merkelize_c(
    _p_starks: *mut c_void,
    _p_proof: *mut c_void,
    _step: u64,
    _buffer: *mut c_void,
    _current_bits: u64,
    _next_bits: u64,
) {
    trace!("{}: ··· {}", "ffi     ", "compute_fri_merkelize: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_queries_c(
    _p_stark: *mut c_void,
    _p_proof: *mut c_void,
    _p_fri_queries: *mut u64,
    _n_queries: u64,
    _n_trees: u64,
) {
    trace!("{}: ··· {}", "ffi     ", "compute_queries: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn compute_fri_queries_c(
    _p_starks: *mut c_void,
    _p_proof: *mut c_void,
    _p_fri_queries: *mut u64,
    _n_queries: u64,
    _step: u64,
    _current_bits: u64,
) {
    trace!("{}: ··· {}", "ffi     ", "compute_fri_queries: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn set_fri_final_pol_c(_p_proof: *mut c_void, _buffer: *mut c_void, _n_bits: u64) {
    trace!("{}: ··· {}", "ffi     ", "set_fri_final_pol: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_hash_c(_pStarks: *mut c_void, _pHhash: *mut c_void, _pBuffer: *mut c_void, _nElements: u64) {
    trace!("{}: ··· {}", "ffi     ", "calculate_hash: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn set_const_tree_c(_pStarks: *mut c_void, _pConstPols: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "set_const_tree: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn transcript_new_c(_element_type: u32, _arity: u64, _custom: bool) -> *mut c_void {
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
pub fn transcript_free_c(_p_transcript: *mut c_void, _element_type: u32) {
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
pub fn verify_constraints_c(_p_setup: *mut c_void, _p_steps_params: *mut c_void) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "verify_constraints_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn verify_global_constraints_c(
    _p_global_constraints_bin: *mut c_void,
    _publics: *mut c_void,
    _airgroupvalues: *mut *mut c_void,
) -> bool {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "verify_global_constraints_c: This is a mock call because there is no linked library"
    );
    true
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_global_constraints_c(
    _p_global_constraints_bin: *mut c_void,
    _publics: *mut c_void,
    _airgroupvalues: *mut *mut c_void,
    _hint_id: u64,
    _hint_field_name: &str,
    _print_expression: bool,
) -> *mut c_void {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "get_hint_field_global_constraints_c: This is a mock call because there is no linked library"
    );
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn print_by_name_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut c_void,
    _name: &str,
    _lengths: *mut u64,
    _first_print_value: u64,
    _last_print_value: u64,
    _return_values: bool,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "print_by_name_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn print_expression_c(
    _p_setup_ctx: *mut c_void,
    _pol: *mut c_void,
    _dim: u64,
    _first_print_value: u64,
    _last_print_value: u64,
) {
    trace!("{}: ··· {}", "ffi     ", "print_expression_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn print_row_c(_p_setup_ctx: *mut c_void, _buffer: *mut c_void, _stage: u64, _row: u64) {
    trace!("{}: ··· {}", "ffi     ", "print_row_c: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn gen_recursive_proof_c(
    _p_setup_ctx: *mut c_void,
    _p_address: *mut c_void,
    _p_public_inputs: *mut c_void,
    _proof_file: &str,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "gen_recursive_proof_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_zkin_ptr_c(_zkin_file: &str) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "get_zkin_ptr_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn publics2zkin_c(
    _p_zkin: *mut c_void,
    _p_publics: *mut c_void,
    _global_info_file: &str,
    _airgroup_id: u64,
    _is_aggregated: bool,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "publics2zkin_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn add_recursive2_verkey_c(_p_zkin: *mut c_void, _recursive2_verkey: &str) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "add_recursive2_verkey_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn join_zkin_recursive2_c(
    _airgroup_id: u64,
    _p_publics: *mut c_void,
    _p_challenges: *mut c_void,
    _global_info_file: &str,
    _zkin1: *mut c_void,
    _zkin2: *mut c_void,
    _stark_info_recursive2: *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "join_zkin_recursive2_c: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn join_zkin_final_c(
    _p_publics: *mut c_void,
    _p_challenges: *mut c_void,
    _global_info_file: &str,
    _zkin_recursive2: *mut *mut c_void,
    _stark_info_recursive2: *mut *mut c_void,
) -> *mut c_void {
    trace!("{}: ··· {}", "ffi     ", "join_zkin_final: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn set_log_level_c(_level: u64) {
    trace!("{}: ··· {}", "ffi     ", "set_log_level_c: This is a mock call because there is no linked library");
}
