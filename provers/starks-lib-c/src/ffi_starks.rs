#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use ::std::os::raw::c_void;

#[cfg(not(feature = "no_lib_link"))]
use ::std::os::raw::c_char;

#[cfg(feature = "no_lib_link")]
use tracing::trace;

#[cfg(not(feature = "no_lib_link"))]
include!("../bindings_starks.rs");

#[cfg(not(feature = "no_lib_link"))]
use std::ffi::CString;

#[cfg(not(feature = "no_lib_link"))]
use std::ffi::CStr;

#[cfg(not(feature = "no_lib_link"))]
static mut PROOFS_DONE: Option<crossbeam_channel::Sender<(u64, String)>> = None;

#[cfg(not(feature = "no_lib_link"))]
extern "C" fn on_proof_done(instance_id: u64, proof_type: *const c_char) {
    let proof_type_str = unsafe { CStr::from_ptr(proof_type).to_string_lossy().into_owned() };

    unsafe {
        if let Some(ref tx) = PROOFS_DONE {
            let _ = tx.send((instance_id, proof_type_str));
        }
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn register_proof_done_callback_c(tx: crossbeam_channel::Sender<(u64, String)>) {
    unsafe {
        PROOFS_DONE = Some(tx);
        register_proof_done_callback(Some(on_proof_done));
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn initialize_agg_readiness_tracker_c() {
    unsafe {
        initialize_agg_readiness_tracker();
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn free_agg_readiness_tracker_c() {
    unsafe {
        free_agg_readiness_tracker();
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn agg_is_ready_c() -> i32 {
    unsafe { agg_is_ready() }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn reset_agg_readiness_tracker_c() {
    unsafe {
        reset_agg_readiness_tracker();
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn launch_callback_c(instance_id: u64, proof_type: &str) {
    let proof_type_str = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_str.as_ptr() as *mut std::os::raw::c_char;
    unsafe {
        launch_callback(instance_id, proof_type_ptr);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn clear_proof_done_callback_c() {
    unsafe {
        PROOFS_DONE = None;
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn save_challenges_c(p_challenges: *mut u8, global_info_file: &str, output_dir: &str) {
    unsafe {
        let file_dir = CString::new(output_dir).unwrap();
        let file_ptr = file_dir.as_ptr() as *mut std::os::raw::c_char;

        let global_info_file_name = CString::new(global_info_file).unwrap();
        let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

        save_challenges(p_challenges as *mut std::os::raw::c_void, global_info_file_ptr, file_ptr);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn save_publics_c(n_publics: u64, public_inputs: *mut u8, output_dir: &str) {
    let file_dir: CString = CString::new(output_dir).unwrap();
    unsafe {
        save_publics(
            n_publics,
            public_inputs as *mut std::os::raw::c_void,
            file_dir.as_ptr() as *mut std::os::raw::c_char,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn save_proof_values_c(proof_values: *mut u8, global_info_file: &str, output_dir: &str) {
    let file_dir: CString = CString::new(output_dir).unwrap();

    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        save_proof_values(
            proof_values as *mut std::os::raw::c_void,
            global_info_file_ptr,
            file_dir.as_ptr() as *mut std::os::raw::c_char,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_info_new_c(
    filename: &str,
    recursive: bool,
    verify_constraints: bool,
    verify: bool,
    gpu: bool,
    preallocate: bool,
) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();

        stark_info_new(
            filename.as_ptr() as *mut std::os::raw::c_char,
            recursive,
            verify_constraints,
            verify,
            gpu,
            preallocate,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_map_totaln_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_map_total_n(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_tree_size_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_tree_size(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_const_offset_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_const_pols_offset(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_map_totaln_custom_commits_fixed_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_map_total_n_custom_commits_fixed(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_proof_size_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_proof_size(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_proof_pinned_size_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_proof_pinned_size(p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_memory_expressions_c(p_stark_info: *mut c_void, n_tmp1: u64, n_tmp3: u64) {
    unsafe {
        set_memory_expressions(p_stark_info, n_tmp1, n_tmp3);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_info_free_c(p_stark_info: *mut c_void) {
    unsafe {
        stark_info_free(p_stark_info);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn load_const_pols_c(pConstPolsAddress: *mut u8, const_filename: &str, const_size: u64) {
    unsafe {
        let const_filename: CString = CString::new(const_filename).unwrap();

        load_const_pols(
            pConstPolsAddress as *mut std::os::raw::c_void,
            const_filename.as_ptr() as *mut std::os::raw::c_char,
            const_size,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_const_size_c(pStarkInfo: *mut c_void) -> u64 {
    unsafe { get_const_size(pStarkInfo) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_const_tree_size_c(pStarkInfo: *mut c_void) -> u64 {
    unsafe { get_const_tree_size(pStarkInfo) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn load_const_tree_c(
    pStarkInfo: *mut c_void,
    pConstPolsTreeAddress: *mut u8,
    tree_filename: &str,
    const_tree_size: u64,
    verkey_filename: &str,
) -> bool {
    unsafe {
        let tree_filename: CString = CString::new(tree_filename).unwrap();
        let verkey_filename: CString = CString::new(verkey_filename).unwrap();

        load_const_tree(
            pStarkInfo,
            pConstPolsTreeAddress as *mut std::os::raw::c_void,
            tree_filename.as_ptr() as *mut std::os::raw::c_char,
            const_tree_size,
            verkey_filename.as_ptr() as *mut std::os::raw::c_char,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn init_gpu_setup_c(maxBitsExt: u64) {
    unsafe {
        init_gpu_setup(maxBitsExt);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn prepare_blocks_c(pol: *mut u64, N: u64, nCols: u64) {
    unsafe {
        prepare_blocks(pol, N, nCols);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_const_tree_c(pStarkInfo: *mut c_void, pConstPols: *mut u8, pConstPolsTreeAddress: *mut u8) {
    unsafe {
        calculate_const_tree(
            pStarkInfo,
            pConstPols as *mut std::os::raw::c_void,
            pConstPolsTreeAddress as *mut std::os::raw::c_void,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_const_tree_bn128_c(pStarkInfo: *mut c_void, pConstPols: *mut u8, pConstPolsTreeAddress: *mut u8) {
    unsafe {
        calculate_const_tree_bn128(
            pStarkInfo,
            pConstPols as *mut std::os::raw::c_void,
            pConstPolsTreeAddress as *mut std::os::raw::c_void,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn write_const_tree_c(pStarkInfo: *mut c_void, pConstPolsTreeAddress: *mut u8, tree_filename: &str) {
    unsafe {
        let tree_filename: CString = CString::new(tree_filename).unwrap();

        write_const_tree(
            pStarkInfo,
            pConstPolsTreeAddress as *mut std::os::raw::c_void,
            tree_filename.as_ptr() as *mut std::os::raw::c_char,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn write_const_tree_bn128_c(pStarkInfo: *mut c_void, pConstPolsTreeAddress: *mut u8, tree_filename: &str) {
    unsafe {
        let tree_filename: CString = CString::new(tree_filename).unwrap();

        write_const_tree_bn128(
            pStarkInfo,
            pConstPolsTreeAddress as *mut std::os::raw::c_void,
            tree_filename.as_ptr() as *mut std::os::raw::c_char,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn expressions_bin_new_c(filename: &str, global: bool, verify: bool) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();

        expressions_bin_new(filename.as_ptr() as *mut std::os::raw::c_char, global, verify)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_max_n_tmp1_c(p_expressions_bin: *mut c_void) -> u64 {
    unsafe { get_max_n_tmp1(p_expressions_bin) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_max_n_tmp3_c(p_expressions_bin: *mut c_void) -> u64 {
    unsafe { get_max_n_tmp3(p_expressions_bin) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_max_n_args_c(p_expressions_bin: *mut c_void) -> u64 {
    unsafe { get_max_args(p_expressions_bin) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_max_n_ops_c(p_expressions_bin: *mut c_void) -> u64 {
    unsafe { get_max_ops(p_expressions_bin) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_operations_quotient_c(p_expressions_bin: *mut c_void, p_stark_info: *mut c_void) -> u64 {
    unsafe { get_operations_quotient(p_expressions_bin, p_stark_info) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn expressions_bin_free_c(p_expressions_bin: *mut c_void) {
    unsafe {
        expressions_bin_free(p_expressions_bin);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn n_hint_ids_by_name_c(p_expressions_bin: *mut c_void, hint_name: &str) -> u64 {
    let name = CString::new(hint_name).unwrap();
    unsafe { n_hints_by_name(p_expressions_bin, name.as_ptr() as *mut std::os::raw::c_char) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_ids_by_name_c(p_expressions_bin: *mut c_void, hint_ids: *mut u64, hint_name: &str) {
    let name = CString::new(hint_name).unwrap();
    unsafe {
        get_hint_ids_by_name(p_expressions_bin, hint_ids, name.as_ptr() as *mut std::os::raw::c_char);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut u8,
    hint_field_values: *mut c_void,
    hint_id: u64,
    hint_field_name: &str,
    hint_options: *mut u8,
) {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe {
        get_hint_field(
            p_setup_ctx,
            p_steps_params as *mut std::os::raw::c_void,
            hint_field_values,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
            hint_options as *mut std::os::raw::c_void,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_values_c(p_setup_ctx: *mut c_void, hint_id: u64, hint_field_name: &str) -> u64 {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe { get_hint_field_values(p_setup_ctx, hint_id, field_name.as_ptr() as *mut std::os::raw::c_char) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_sizes_c(
    p_setup_ctx: *mut c_void,
    hint_field_values: *mut c_void,
    hint_id: u64,
    hint_field_name: &str,
    hint_options: *mut u8,
) {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe {
        get_hint_field_sizes(
            p_setup_ctx,
            hint_field_values,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
            hint_options as *mut std::os::raw::c_void,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn mul_hint_fields_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut u8,
    n_hints: u64,
    hint_id: *mut u64,
    hint_field_dest: Vec<&str>,
    hint_field_name1: Vec<&str>,
    hint_field_name2: Vec<&str>,
    hint_options1: *mut *mut u8,
    hint_options2: *mut *mut u8,
) {
    use std::os::raw::c_char;

    let c_hint_field_dest: Vec<CString> = hint_field_dest.iter().map(|&s| CString::new(s).unwrap()).collect();

    let c_hint_field_name1: Vec<CString> = hint_field_name1.iter().map(|&s| CString::new(s).unwrap()).collect();

    let c_hint_field_name2: Vec<CString> = hint_field_name2.iter().map(|&s| CString::new(s).unwrap()).collect();

    // Convert Vec<CString> to Vec<*mut c_char>
    let mut hint_field_dest_ptrs: Vec<*mut c_char> =
        c_hint_field_dest.iter().map(|s| s.as_ptr() as *mut c_char).collect();

    let mut hint_field_name1_ptrs: Vec<*mut c_char> =
        c_hint_field_name1.iter().map(|s| s.as_ptr() as *mut c_char).collect();

    let mut hint_field_name2_ptrs: Vec<*mut c_char> =
        c_hint_field_name2.iter().map(|s| s.as_ptr() as *mut c_char).collect();

    unsafe {
        mul_hint_fields(
            p_setup_ctx,
            p_steps_params as *mut std::os::raw::c_void,
            n_hints,
            hint_id,
            hint_field_dest_ptrs.as_mut_ptr(),
            hint_field_name1_ptrs.as_mut_ptr(),
            hint_field_name2_ptrs.as_mut_ptr(),
            hint_options1 as *mut *mut std::os::raw::c_void,
            hint_options2 as *mut *mut std::os::raw::c_void,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn acc_hint_field_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut u8,
    hint_id: u64,
    hint_field_dest: &str,
    hint_field_airgroupvalue: &str,
    hint_field_name: &str,
    add: bool,
) {
    let field_dest = CString::new(hint_field_dest).unwrap();
    let field_airgroupvalue = CString::new(hint_field_airgroupvalue).unwrap();
    let field_name = CString::new(hint_field_name).unwrap();

    unsafe {
        acc_hint_field(
            p_setup_ctx,
            p_steps_params as *mut std::os::raw::c_void,
            hint_id,
            field_dest.as_ptr() as *mut std::os::raw::c_char,
            field_airgroupvalue.as_ptr() as *mut std::os::raw::c_char,
            field_name.as_ptr() as *mut std::os::raw::c_char,
            add,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn acc_mul_hint_fields_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut u8,
    hint_id: u64,
    hint_field_dest: &str,
    hint_field_airgroupvalue: &str,
    hint_field_name1: &str,
    hint_field_name2: &str,
    hint_options1: *mut u8,
    hint_options2: *mut u8,
    add: bool,
) {
    let field_dest = CString::new(hint_field_dest).unwrap();
    let field_airgroupvalue = CString::new(hint_field_airgroupvalue).unwrap();
    let field_name1 = CString::new(hint_field_name1).unwrap();
    let field_name2: CString = CString::new(hint_field_name2).unwrap();

    unsafe {
        acc_mul_hint_fields(
            p_setup_ctx,
            p_steps_params as *mut std::os::raw::c_void,
            hint_id,
            field_dest.as_ptr() as *mut std::os::raw::c_char,
            field_airgroupvalue.as_ptr() as *mut std::os::raw::c_char,
            field_name1.as_ptr() as *mut std::os::raw::c_char,
            field_name2.as_ptr() as *mut std::os::raw::c_char,
            hint_options1 as *mut std::os::raw::c_void,
            hint_options2 as *mut std::os::raw::c_void,
            add,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn update_airgroupvalue_c(
    p_setup_ctx: *mut c_void,
    p_steps_params: *mut u8,
    hint_id: u64,
    hint_field_airgroupvalue: &str,
    hint_field_name1: &str,
    hint_field_name2: &str,
    hint_options1: *mut u8,
    hint_options2: *mut u8,
    add: bool,
) -> u64 {
    let field_airgroupvalue = CString::new(hint_field_airgroupvalue).unwrap();
    let field_name1 = CString::new(hint_field_name1).unwrap();
    let field_name2: CString = CString::new(hint_field_name2).unwrap();

    unsafe {
        update_airgroupvalue(
            p_setup_ctx,
            p_steps_params as *mut std::os::raw::c_void,
            hint_id,
            field_airgroupvalue.as_ptr() as *mut std::os::raw::c_char,
            field_name1.as_ptr() as *mut std::os::raw::c_char,
            field_name2.as_ptr() as *mut std::os::raw::c_char,
            hint_options1 as *mut std::os::raw::c_void,
            hint_options2 as *mut std::os::raw::c_void,
            add,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_hint_field_c(
    p_setup_ctx: *mut c_void,
    p_params: *mut u8,
    values: *mut u8,
    hint_id: u64,
    hint_field_name: &str,
) -> u64 {
    unsafe {
        let field_name = CString::new(hint_field_name).unwrap();
        set_hint_field(
            p_setup_ctx,
            p_params as *mut std::os::raw::c_void,
            values as *mut std::os::raw::c_void,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_id_c(p_setup_ctx: *mut c_void, hint_id: u64, hint_field_name: &str) -> u64 {
    unsafe {
        let field_name = CString::new(hint_field_name).unwrap();
        get_hint_id(p_setup_ctx, hint_id, field_name.as_ptr() as *mut std::os::raw::c_char)
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_impols_expressions_c(p_setup: *mut c_void, step: u64, p_steps_params: *mut u8) {
    unsafe {
        calculate_impols_expressions(p_setup, step, p_steps_params as *mut std::os::raw::c_void);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn custom_commit_size_c(p_setup: *mut c_void, commit_id: u64) -> u64 {
    unsafe { custom_commit_size(p_setup, commit_id) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn load_custom_commit_c(setup: *mut c_void, commit_id: u64, buffer: *mut u8, buffer_file: &str) {
    let buffer_file_name = CString::new(buffer_file).unwrap();
    unsafe {
        load_custom_commit(
            setup,
            commit_id,
            buffer as *mut std::os::raw::c_void,
            buffer_file_name.as_ptr() as *mut std::os::raw::c_char,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn write_custom_commit_c(
    root: *mut u8,
    n_bits: u64,
    n_bits_ext: u64,
    n_cols: u64,
    buffer: *mut u8,
    buffer_file: &str,
    check: bool,
) {
    let buffer_file_name = CString::new(buffer_file).unwrap();
    unsafe {
        write_custom_commit(
            root as *mut std::os::raw::c_void,
            n_bits,
            n_bits_ext,
            n_cols,
            buffer as *mut std::os::raw::c_void,
            buffer_file_name.as_ptr() as *mut std::os::raw::c_char,
            check,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn commit_witness_c(
    arity: u64,
    n_bits: u64,
    n_bits_ext: u64,
    n_cols: u64,
    instance_id: u64,
    airgroup_id: u64,
    air_id: u64,
    root: *mut u8,
    witness: *mut u8,
    aux_trace: *mut u8,
    d_buffers: *mut c_void,
    setup: *mut c_void,
) -> u64 {
    unsafe {
        commit_witness(
            arity,
            n_bits,
            n_bits_ext,
            n_cols,
            instance_id,
            airgroup_id,
            air_id,
            root as *mut std::os::raw::c_void,
            witness as *mut std::os::raw::c_void,
            aux_trace as *mut std::os::raw::c_void,
            d_buffers,
            setup,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn calculate_hash_c(pValue: *mut u8, pBuffer: *mut u8, nElements: u64, nOutputs: u64) {
    unsafe {
        calculate_hash(pValue as *mut std::os::raw::c_void, pBuffer as *mut std::os::raw::c_void, nElements, nOutputs);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn transcript_new_c(arity: u64, custom: bool) -> *mut c_void {
    unsafe { transcript_new(arity, custom) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn transcript_add_c(p_transcript: *mut c_void, p_input: *mut u8, size: u64) {
    unsafe {
        transcript_add(p_transcript, p_input as *mut std::os::raw::c_void, size);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn transcript_free_c(p_transcript: *mut c_void) {
    unsafe {
        transcript_free(p_transcript);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_challenge_c(p_transcript: *mut c_void, p_element: *mut c_void) {
    unsafe {
        get_challenge(p_transcript, p_element);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_n_constraints_c(p_setup: *mut c_void) -> u64 {
    unsafe { get_n_constraints(p_setup) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_constraints_lines_sizes_c(p_setup: *mut c_void, constraints_sizes: *mut u64) {
    unsafe {
        get_constraints_lines_sizes(p_setup, constraints_sizes);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_constraints_lines_c(p_setup: *mut c_void, constraints_lines: *mut *mut u8) {
    unsafe {
        get_constraints_lines(p_setup, constraints_lines);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn verify_constraints_c(p_setup: *mut c_void, p_steps_params: *mut u8, constraints_info: *mut c_void) {
    unsafe {
        verify_constraints(p_setup, p_steps_params as *mut std::os::raw::c_void, constraints_info);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_n_global_constraints_c(p_global_constraints_bin: *mut c_void) -> u64 {
    unsafe { get_n_global_constraints(p_global_constraints_bin) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_global_constraints_lines_sizes_c(p_global_constraints_bin: *mut c_void, global_constraints_sizes: *mut u64) {
    unsafe {
        get_global_constraints_lines_sizes(p_global_constraints_bin, global_constraints_sizes);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_global_constraints_lines_c(p_global_constraints_bin: *mut c_void, global_constraints_lines: *mut *mut u8) {
    unsafe {
        get_global_constraints_lines(p_global_constraints_bin, global_constraints_lines);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn verify_global_constraints_c(
    global_info_file: &str,
    p_global_constraints_bin: *mut c_void,
    publics: *mut u8,
    challenges: *mut u8,
    proof_values: *mut u8,
    airgroupvalues: *mut *mut u8,
    global_constraints_info: *mut c_void,
) {
    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        verify_global_constraints(
            global_info_file_ptr,
            p_global_constraints_bin,
            publics as *mut std::os::raw::c_void,
            challenges as *mut std::os::raw::c_void,
            proof_values as *mut std::os::raw::c_void,
            airgroupvalues as *mut *mut std::os::raw::c_void,
            global_constraints_info,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn get_hint_field_global_constraints_c(
    global_info_file: &str,
    p_global_constraints_bin: *mut c_void,
    hint_field_values: *mut c_void,
    publics: *mut u8,
    challenges: *mut u8,
    proof_values: *mut u8,
    airgroupvalues: *mut *mut u8,
    hint_id: u64,
    hint_field_name: &str,
    print_expression: bool,
) {
    let field_name = CString::new(hint_field_name).unwrap();

    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        get_hint_field_global_constraints(
            global_info_file_ptr,
            p_global_constraints_bin,
            hint_field_values,
            publics as *mut std::os::raw::c_void,
            challenges as *mut std::os::raw::c_void,
            proof_values as *mut std::os::raw::c_void,
            airgroupvalues as *mut *mut std::os::raw::c_void,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
            print_expression,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_global_constraints_values_c(
    p_global_constraints_bin: *mut c_void,
    hint_id: u64,
    hint_field_name: &str,
) -> u64 {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe {
        get_hint_field_global_constraints_values(
            p_global_constraints_bin,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_hint_field_global_constraints_sizes_c(
    global_info_file: &str,
    p_global_constraints_bin: *mut c_void,
    hint_field_values: *mut c_void,
    hint_id: u64,
    hint_field_name: &str,
    print_expression: bool,
) {
    let field_name = CString::new(hint_field_name).unwrap();

    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        get_hint_field_global_constraints_sizes(
            global_info_file_ptr,
            p_global_constraints_bin,
            hint_field_values,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
            print_expression,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_hint_field_global_constraints_c(
    global_info_file: &str,
    p_global_constraints_bin: *mut c_void,
    proof_values: *mut u8,
    values: *mut u8,
    hint_id: u64,
    hint_field_name: &str,
) -> u64 {
    let field_name = CString::new(hint_field_name).unwrap();

    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        set_hint_field_global_constraints(
            global_info_file_ptr,
            p_global_constraints_bin,
            proof_values as *mut std::os::raw::c_void,
            values as *mut std::os::raw::c_void,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn gen_proof_c(
    p_setup: *mut c_void,
    p_params: *mut u8,
    p_global_challenge: *mut u8,
    proof_buffer: *mut u64,
    proof_file: &str,
    airgroup_id: u64,
    air_id: u64,
    instance_id: u64,
    d_buffers: *mut c_void,
    skip_recalculation: bool,
    stream_id: u64,
    const_pols_path: &str,
    const_tree_path: &str,
) -> u64 {
    let proof_file_name = CString::new(proof_file).unwrap();
    let proof_file_ptr = proof_file_name.as_ptr() as *mut std::os::raw::c_char;

    let const_filename_name = CString::new(const_pols_path).unwrap();
    let const_filename_ptr = const_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let const_tree_filename_name = CString::new(const_tree_path).unwrap();
    let const_tree_filename_ptr = const_tree_filename_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        gen_proof(
            p_setup,
            airgroup_id,
            air_id,
            instance_id,
            p_params as *mut std::os::raw::c_void,
            p_global_challenge as *mut std::os::raw::c_void,
            proof_buffer,
            proof_file_ptr,
            d_buffers,
            skip_recalculation,
            stream_id,
            const_filename_ptr,
            const_tree_filename_ptr,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn get_stream_proofs_c(d_buffers: *mut c_void) {
    unsafe {
        get_stream_proofs(d_buffers);
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn get_stream_proofs_non_blocking_c(d_buffers: *mut c_void) {
    unsafe {
        get_stream_proofs_non_blocking(d_buffers);
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn get_stream_id_proof_c(d_buffers: *mut c_void, stream_id: u64) {
    unsafe {
        get_stream_id_proof(d_buffers, stream_id);
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn gen_recursive_proof_c(
    p_setup_ctx: *mut c_void,
    p_witness: *mut u8,
    p_aux_trace: *mut u8,
    p_const_pols: *mut u8,
    p_const_tree: *mut u8,
    p_public_inputs: *mut u8,
    proof_buffer: *mut u64,
    proof_file: &str,
    global_info_file: &str,
    airgroup_id: u64,
    air_id: u64,
    instance_id: u64,
    vadcop: bool,
    d_buffers: *mut c_void,
    const_pols_path: &str,
    const_tree_path: &str,
    proof_type: &str,
) -> u64 {
    let proof_file_name = CString::new(proof_file).unwrap();
    let proof_file_ptr = proof_file_name.as_ptr() as *mut std::os::raw::c_char;

    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    let const_filename_name = CString::new(const_pols_path).unwrap();
    let const_filename_ptr = const_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let const_tree_filename_name = CString::new(const_tree_path).unwrap();
    let const_tree_filename_ptr = const_tree_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let proof_type_name = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        gen_recursive_proof(
            p_setup_ctx,
            global_info_file_ptr,
            airgroup_id,
            air_id,
            instance_id,
            p_witness as *mut std::os::raw::c_void,
            p_aux_trace as *mut std::os::raw::c_void,
            p_const_pols as *mut std::os::raw::c_void,
            p_const_tree as *mut std::os::raw::c_void,
            p_public_inputs as *mut std::os::raw::c_void,
            proof_buffer,
            proof_file_ptr,
            vadcop,
            d_buffers,
            const_filename_ptr,
            const_tree_filename_ptr,
            proof_type_ptr,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn gen_recursive_proof_final_c(
    p_setup_ctx: *mut c_void,
    p_witness: *mut u8,
    p_aux_trace: *mut u8,
    p_const_pols: *mut u8,
    p_const_tree: *mut u8,
    p_public_inputs: *mut u8,
    proof_file: &str,
    global_info_file: &str,
    airgroup_id: u64,
    air_id: u64,
    instance_id: u64,
) -> *mut c_void {
    let proof_file_name = CString::new(proof_file).unwrap();
    let proof_file_ptr = proof_file_name.as_ptr() as *mut std::os::raw::c_char;

    let global_info_file_name = CString::new(global_info_file).unwrap();
    let global_info_file_ptr = global_info_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        gen_recursive_proof_final(
            p_setup_ctx,
            global_info_file_ptr,
            airgroup_id,
            air_id,
            instance_id,
            p_witness as *mut std::os::raw::c_void,
            p_aux_trace as *mut std::os::raw::c_void,
            p_const_pols as *mut std::os::raw::c_void,
            p_const_tree as *mut std::os::raw::c_void,
            p_public_inputs as *mut std::os::raw::c_void,
            proof_file_ptr,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn read_exec_file_c(exec_data: *mut u64, exec_file: *const i8, nCols: u64) {
    unsafe {
        read_exec_file(exec_data, exec_file as *mut std::os::raw::c_char, nCols);
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn get_committed_pols_c(
    circomWitness: *mut u8,
    exec_data: *mut u64,
    witness: *mut u8,
    pPublics: *mut u8,
    sizeWitness: u64,
    N: u64,
    nPublics: u64,
    nCols: u64,
) {
    unsafe {
        get_committed_pols(
            circomWitness as *mut std::os::raw::c_void,
            exec_data,
            witness as *mut std::os::raw::c_void,
            pPublics as *mut std::os::raw::c_void,
            sizeWitness,
            N,
            nPublics,
            nCols,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn add_publics_aggregation_c(proof: *mut u8, offset: u64, publics: *mut u8, nPublics: u64) {
    unsafe {
        add_publics_aggregation(
            proof as *mut std::os::raw::c_void,
            offset,
            publics as *mut std::os::raw::c_void,
            nPublics,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn gen_final_snark_proof_c(circomWitnessFinal: *mut u8, zkeyFile: &str, outputDir: &str) {
    let zkey_file_name = CString::new(zkeyFile).unwrap();
    let zkey_file_ptr = zkey_file_name.as_ptr() as *mut std::os::raw::c_char;

    let output_dir_name = CString::new(outputDir).unwrap();
    let output_dir_ptr = output_dir_name.as_ptr() as *mut std::os::raw::c_char;
    unsafe {
        gen_final_snark_proof(circomWitnessFinal as *mut std::os::raw::c_void, zkey_file_ptr, output_dir_ptr);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_log_level_c(level: u64) {
    unsafe {
        setLogLevel(level);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_verify_c(
    verkey: &str,
    p_proof: *mut u64,
    p_stark_info: *mut c_void,
    p_expressions_bin: *mut c_void,
    p_publics: *mut u8,
    p_proof_values: *mut u8,
    p_challenges: *mut u8,
) -> bool {
    let verkey_file = CString::new(verkey).unwrap();
    let verkey_file_ptr = verkey_file.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        stark_verify(
            p_proof,
            p_stark_info,
            p_expressions_bin,
            verkey_file_ptr,
            p_publics as *mut c_void,
            p_proof_values as *mut c_void,
            p_challenges as *mut c_void,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_verify_bn128_c(
    verkey: &str,
    p_proof: *mut c_void,
    p_stark_info: *mut c_void,
    p_expressions_bin: *mut c_void,
    p_publics: *mut u8,
) -> bool {
    let verkey_file = CString::new(verkey).unwrap();
    let verkey_file_ptr = verkey_file.as_ptr() as *mut std::os::raw::c_char;

    unsafe { stark_verify_bn128(p_proof, p_stark_info, p_expressions_bin, verkey_file_ptr, p_publics as *mut c_void) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn stark_verify_from_file_c(
    verkey: &str,
    proof: &str,
    p_stark_info: *mut c_void,
    p_expressions_bin: *mut c_void,
    p_publics: *mut u8,
    p_proof_values: *mut u8,
    p_challenges: *mut u8,
) -> bool {
    let verkey_file = CString::new(verkey).unwrap();
    let verkey_file_ptr = verkey_file.as_ptr() as *mut std::os::raw::c_char;

    let proof_file = CString::new(proof).unwrap();
    let proof_file_ptr = proof_file.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        stark_verify_from_file(
            proof_file_ptr,
            p_stark_info,
            p_expressions_bin,
            verkey_file_ptr,
            p_publics as *mut c_void,
            p_proof_values as *mut c_void,
            p_challenges as *mut c_void,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn write_fixed_cols_bin_c(
    binfile: &str,
    airgroup: &str,
    air: &str,
    n: u64,
    n_fixed_pols: u64,
    fixed_pols_info: *mut c_void,
) {
    let binfile_name = CString::new(binfile).unwrap();
    let binfile_name_ptr = binfile_name.as_ptr() as *mut std::os::raw::c_char;

    let airgroup_name = CString::new(airgroup).unwrap();
    let airgroup_name_ptr = airgroup_name.as_ptr() as *mut std::os::raw::c_char;

    let air_name = CString::new(air).unwrap();
    let air_name_ptr = air_name.as_ptr() as *mut std::os::raw::c_char;
    unsafe {
        write_fixed_cols_bin(binfile_name_ptr, airgroup_name_ptr, air_name_ptr, n, n_fixed_pols, fixed_pols_info);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_omp_max_threads_c() -> u64 {
    unsafe { get_omp_max_threads() }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn set_omp_num_threads_c(num_threads: u64) {
    unsafe {
        set_omp_num_threads(num_threads);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn gen_device_buffers_c(
    max_sizes: *mut ::std::os::raw::c_void,
    node_rank: u32,
    node_n_processes: u32,
) -> *mut ::std::os::raw::c_void {
    unsafe { gen_device_buffers(max_sizes, node_rank, node_n_processes) }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn gen_device_streams_c(
    d_buffers: *mut ::std::os::raw::c_void,
    max_size_buffer: u64,
    max_size_buffer_aggregation: u64,
    max_pinned_proof_size: u64,
    max_n_bits_ext: u64,
) -> u64 {
    unsafe {
        gen_device_streams(
            d_buffers,
            max_size_buffer,
            max_size_buffer_aggregation,
            max_pinned_proof_size,
            max_n_bits_ext,
        )
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn get_instances_ready_c(d_buffers: *mut ::std::os::raw::c_void, instances_ready: *mut i64) {
    unsafe {
        get_instances_ready(d_buffers, instances_ready);
    }
}

#[cfg(not(feature = "no_lib_link"))]
#[allow(clippy::too_many_arguments)]
pub fn reset_device_streams_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe {
        reset_device_streams(d_buffers);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn check_device_memory_c(node_rank: u32, node_size: u32) -> u64 {
    unsafe { check_device_memory(node_rank, node_size) }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn get_num_gpus_c() -> u64 {
    unsafe { get_num_gpus() }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn free_device_buffers_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe {
        free_device_buffers(d_buffers);
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn load_device_setup_c(
    airgroup_id: u64,
    air_id: u64,
    proof_type: &str,
    p_setup: *mut ::std::os::raw::c_void,
    d_buffers: *mut ::std::os::raw::c_void,
    verkey_root: *mut u8,
    packed_info: *mut ::std::os::raw::c_void,
    n_streams: u64,
) {
    let proof_type_name = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        load_device_setup(
            airgroup_id,
            air_id,
            proof_type_ptr,
            p_setup,
            d_buffers,
            verkey_root as *mut std::os::raw::c_void,
            packed_info,
            n_streams,
        );
    }
}

#[cfg(not(feature = "no_lib_link"))]
pub fn load_device_const_pols_c(
    airgroup_id: u64,
    air_id: u64,
    initial_offset: u64,
    d_buffers: *mut ::std::os::raw::c_void,
    const_filename: &str,
    const_size: u64,
    const_tree_filename: &str,
    const_tree_size: u64,
    proof_type: &str,
) {
    let const_filename_name = CString::new(const_filename).unwrap();
    let const_filename_ptr = const_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let const_tree_filename_name = CString::new(const_tree_filename).unwrap();
    let const_tree_filename_ptr = const_tree_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let proof_type_name = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        load_device_const_pols(
            airgroup_id,
            air_id,
            initial_offset,
            d_buffers,
            const_filename_ptr,
            const_size,
            const_tree_filename_ptr,
            const_tree_size,
            proof_type_ptr,
        );
    }
}

// ------------------------
// MOCK METHODS FOR TESTING
// ------------------------
#[cfg(feature = "no_lib_link")]
pub fn launch_callback_c(_instance_id: u64, _proof_type: &str) {
    trace!("{}: ··· {}", "ffi     ", "launch_callback: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn register_proof_done_callback_c(_tx: crossbeam_channel::Sender<(u64, String)>) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "register_proof_done_callback: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn initialize_agg_readiness_tracker_c() {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "initialize_agg_readiness_tracker: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn free_agg_readiness_tracker_c() {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "free_agg_readiness_tracker: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn agg_is_ready_c() -> i32 {
    0
}

#[cfg(feature = "no_lib_link")]
pub fn reset_agg_readiness_tracker_c() {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "reset_agg_readiness_tracker: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn clear_proof_done_callback_c() {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "_clear_proof_done_callback: This is a mock call because there is no linked library"
    );
}

#[cfg(feature = "no_lib_link")]
pub fn save_challenges_c(_p_challenges: *mut u8, _global_info_file: &str, _output_dir: &str) {
    trace!("··· {}", "save_challenges: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn save_publics_c(_n_publics: u64, _public_inputs: *mut u8, _output_dir: &str) {
    trace!("··· {}", "save_publics: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn save_proof_values_c(_proof_values: *mut u8, _global_info_file: &str, _output_dir: &str) {
    trace!("··· {}", "save_proof_values: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn stark_info_new_c(
    _filename: &str,
    _recursive: bool,
    _verify_constraints: bool,
    _verify: bool,
    _gpu: bool,
    _preallocate: bool,
) -> *mut c_void {
    trace!("··· {}", "starkinfo_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_map_totaln_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("··· {}", "get_map_totaln: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_tree_size_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_tree_size: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_const_offset_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("··· {}", "get_const_offset: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_proof_size_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("··· {}", "get_proof_size: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_proof_pinned_size_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("··· {}", "get_pinned_proof_size: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn set_memory_expressions_c(_p_stark_info: *mut c_void, _n_tmp1: u64, _n_tmp3: u64) {
    trace!("··· {}", "set_memory_expressions: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_map_totaln_custom_commits_fixed_c(_p_stark_info: *mut c_void) -> u64 {
    trace!("··· {}", "get_map_totaln: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_custom_commit_id_c(_p_stark_info: *mut c_void, _name: &str) -> u64 {
    trace!("··· {}", "get_custom_commit_id: This is a mock call because there is no linked library");
    100000000
}

#[cfg(feature = "no_lib_link")]
pub fn stark_info_free_c(_p_stark_info: *mut c_void) {
    trace!("··· {}", "starkinfo_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn load_const_pols_c(_pConstPolsAddress: *mut u8, _const_filename: &str, _const_size: u64) {
    trace!("··· {}", "load_const_pols: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_const_tree_size_c(_pStarkInfo: *mut c_void) -> u64 {
    trace!("··· {}", "get_const_tree_size: This is a mock call because there is no linked library");
    1000000
}

#[cfg(feature = "no_lib_link")]
pub fn get_const_size_c(_pStarkInfo: *mut c_void) -> u64 {
    trace!("··· {}", "get_const_size: This is a mock call because there is no linked library");
    1000000
}

#[cfg(feature = "no_lib_link")]
pub fn load_const_tree_c(
    _pStarkInfo: *mut c_void,
    _pConstPolsTreeAddress: *mut u8,
    _tree_filename: &str,
    _const_tree_size: u64,
    _verkey_path: &str,
) -> bool {
    trace!("··· {}", "load_const_tree: This is a mock call because there is no linked library");
    true
}

#[cfg(feature = "no_lib_link")]
pub fn init_gpu_setup_c(_maxBitsExt: u64) {
    trace!("··· {}", "init_gpu_setup: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn prepare_blocks_c(_pol: *mut u64, _N: u64, _nCols: u64) {
    trace!("··· {}", "prepare_blocks: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_const_tree_c(_pStarkInfo: *mut c_void, _pConstPols: *mut u8, _pConstPolsTreeAddress: *mut u8) {
    trace!("··· {}", "calculate_const_tree: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_const_tree_bn128_c(_pStarkInfo: *mut c_void, _pConstPols: *mut u8, _pConstPolsTreeAddress: *mut u8) {
    trace!("··· {}", "calculate_const_tree: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn write_const_tree_c(_pStarkInfo: *mut c_void, _pConstPolsTreeAddress: *mut u8, _tree_filename: &str) {
    trace!("··· {}", "write_const_tree: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn write_const_tree_bn128_c(_pStarkInfo: *mut c_void, _pConstPolsTreeAddress: *mut u8, _tree_filename: &str) {
    trace!("··· {}", "write_const_tree_bn128: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn expressions_bin_new_c(_filename: &str, _global: bool, _verify: bool) -> *mut c_void {
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn get_max_n_tmp1_c(_p_expressions_bin: *mut c_void) -> u64 {
    10000
}

#[cfg(feature = "no_lib_link")]
pub fn get_max_n_tmp3_c(_p_expressions_bin: *mut c_void) -> u64 {
    10000
}

#[cfg(feature = "no_lib_link")]
pub fn get_max_n_args_c(_p_expressions_bin: *mut c_void) -> u64 {
    10000
}

#[cfg(feature = "no_lib_link")]
pub fn get_max_n_ops_c(_p_expressions_bin: *mut c_void) -> u64 {
    10000
}

#[cfg(feature = "no_lib_link")]
pub fn get_operations_quotient_c(_p_expressions_bin: *mut c_void, _p_stark_info: *mut c_void) -> u64 {
    0
}

#[cfg(feature = "no_lib_link")]
pub fn expressions_bin_free_c(_p_expressions_bin: *mut c_void) {}

#[cfg(feature = "no_lib_link")]
pub fn n_hint_ids_by_name_c(_p_expressions_bin: *mut c_void, _hint_name: &str) -> u64 {
    trace!("··· {}", "n_hint_ids_by_name: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_ids_by_name_c(_p_expressions_bin: *mut c_void, _hint_ids: *mut u64, _hint_name: &str) {
    trace!("··· {}", "get_hint_ids_by_name: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut u8,
    _hint_field_values: *mut c_void,
    _hint_id: u64,
    _hint_field_name: &str,
    _hint_options: *mut u8,
) {
    trace!("··· {}", "get_hint_field: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_sizes_c(
    _p_setup_ctx: *mut c_void,
    _hint_field_values: *mut c_void,
    _hint_id: u64,
    _hint_field_name: &str,
    _hint_options: *mut u8,
) {
    trace!("··· {}", "get_hint_field_sizes: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_values_c(_p_setup_ctx: *mut c_void, _hint_id: u64, _hint_field_name: &str) -> u64 {
    trace!("··· {}", "get_hint_field: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn mul_hint_fields_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut u8,
    _n_hints: u64,
    _hint_id: *mut u64,
    _hint_field_dest: Vec<&str>,
    _hint_field_name1: Vec<&str>,
    _hint_field_name2: Vec<&str>,
    _hint_options1: *mut *mut u8,
    _hint_options2: *mut *mut u8,
) {
    trace!("··· {}", "mul_hint_fields: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn acc_hint_field_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut u8,
    _hint_id: u64,
    _hint_field_dest: &str,
    _hint_field_airgroupvalue: &str,
    _hint_field_name: &str,
    _add: bool,
) {
    trace!("··· {}", "acc_hint_fields: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn acc_mul_hint_fields_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut u8,
    _hint_id: u64,
    _hint_field_dest: &str,
    _hint_field_airgroupvalue: &str,
    _hint_field_name1: &str,
    _hint_field_name2: &str,
    _hint_options1: *mut u8,
    _hint_options2: *mut u8,
    _add: bool,
) {
    trace!("··· {}", "acc_mul_hint_fields: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn update_airgroupvalue_c(
    _p_setup_ctx: *mut c_void,
    _p_steps_params: *mut u8,
    _hint_id: u64,
    _hint_field_airgroupvalue: &str,
    _hint_field_name1: &str,
    _hint_field_name2: &str,
    _hint_options1: *mut u8,
    _hint_options2: *mut u8,
    _add: bool,
) -> u64 {
    trace!("··· {}", "update_airgroupvalue: This is a mock call because there is no linked library");
    10000
}

#[cfg(feature = "no_lib_link")]
pub fn set_hint_field_c(
    _p_setup_ctx: *mut c_void,
    _p_params: *mut u8,
    _values: *mut u8,
    _hint_id: u64,
    _hint_field_name: &str,
) -> u64 {
    trace!("··· {}", "set_hint_field: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_id_c(_p_setup_ctx: *mut c_void, _hint_id: u64, _hint_field_name: &str) -> u64 {
    trace!("··· {}", "get_hint_field_id: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_impols_expressions_c(_p_setup: *mut c_void, _step: u64, _p_steps_params: *mut u8) {
    trace!("··· {}", "calculate_impols_expression: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn custom_commit_size_c(_p_setup: *mut c_void, _commit_id: u64) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "custom_commit_size: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn load_custom_commit_c(_p_setup: *mut c_void, _commit_id: u64, _buffer: *mut u8, _tree_file: &str) {
    trace!("··· {}", "load_custom_commit: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn write_custom_commit_c(
    _root: *mut u8,
    _n_bits: u64,
    _n_bits_ext: u64,
    _n_cols: u64,
    _buffer: *mut u8,
    _buffer_file: &str,
    _check: bool,
) {
    trace!("··· {}", "write_custom_commit: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn commit_witness_c(
    _arity: u64,
    _n_bits: u64,
    _n_bits_ext: u64,
    _n_cols: u64,
    _instance_id: u64,
    _airgroup_id: u64,
    _air_id: u64,
    _root: *mut u8,
    _witness: *mut u8,
    _aux_trace: *mut u8,
    _d_buffers: *mut c_void,
    _setup: *mut c_void,
) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "commit_witness: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn calculate_hash_c(_pValue: *mut u8, _pBuffer: *mut u8, _nElements: u64, _nOutputs: u64) {
    trace!("··· {}", "calculate_hash: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn transcript_new_c(_arity: u64, _custom: bool) -> *mut c_void {
    trace!("··· {}", "transcript_new: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn transcript_add_c(_p_transcript: *mut c_void, _p_input: *mut u8, _size: u64) {
    trace!("··· {}", "transcript_add: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn transcript_free_c(_p_transcript: *mut c_void) {
    trace!("··· {}", "transcript_free: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_challenge_c(_p_transcript: *mut c_void, _p_element: *mut c_void) {
    trace!("··· {}", "get_challenges: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_n_constraints_c(_p_setup: *mut c_void) -> u64 {
    trace!("··· {}", "get_n_constraints: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn get_constraints_lines_sizes_c(_p_setup: *mut c_void, _constraints_sizes: *mut u64) {
    trace!("··· {}", "get_constraints_lines_sizes: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_constraints_lines_c(_p_setup: *mut c_void, _constraints_lines: *mut *mut u8) {
    trace!("··· {}", "get_constraints_lines: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn verify_constraints_c(_p_setup: *mut c_void, _p_steps_params: *mut u8, _constraints_info: *mut c_void) {
    trace!("··· {}", "verify_constraints: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_n_global_constraints_c(_p_global_constraints_bin: *mut c_void) -> u64 {
    trace!("··· {}", "get_n_global_constraints: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn get_global_constraints_lines_sizes_c(
    _p_global_constraints_bin: *mut c_void,
    _global_constraints_sizes: *mut u64,
) {
    trace!("··· {}", "get_global_constraints_lines_sizes: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_global_constraints_lines_c(_p_global_constraints_bin: *mut c_void, _global_constraints_lines: *mut *mut u8) {
    trace!("··· {}", "get_global_constraints_lines: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn verify_global_constraints_c(
    _global_info_file: &str,
    _p_global_constraints_bin: *mut c_void,
    _publics: *mut u8,
    _challenges: *mut u8,
    _proof_values: *mut u8,
    _airgroupvalues: *mut *mut u8,
    _global_constraints_info: *mut c_void,
) {
    trace!("··· {}", "verify_global_constraints: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn get_hint_field_global_constraints_c(
    _global_info_file: &str,
    _p_global_constraints_bin: *mut c_void,
    _hint_field_values: *mut c_void,
    _publics: *mut u8,
    _challenges: *mut u8,
    _proof_values: *mut u8,
    _airgroupvalues: *mut *mut u8,
    _hint_id: u64,
    _hint_field_name: &str,
    _print_expression: bool,
) {
    trace!("··· {}", "get_hint_field_global_constraints: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_global_constraints_values_c(
    _p_global_constraints_bin: *mut c_void,
    _hint_id: u64,
    _hint_field_name: &str,
) -> u64 {
    trace!(
        "··· {}",
        "get_hint_field_global_constraints_values: This is a mock call because there is no linked library"
    );
    0
}

#[cfg(feature = "no_lib_link")]
pub fn get_hint_field_global_constraints_sizes_c(
    _global_info_file: &str,
    _p_global_constraints_bin: *mut c_void,
    _hint_field_values: *mut c_void,
    _hint_id: u64,
    _hint_field_name: &str,
    _print_expression: bool,
) {
    trace!("··· {}", "get_hint_field_global_constraints_sizes: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn set_hint_field_global_constraints_c(
    _global_info_file: &str,
    _p_global_constraints_bin: *mut c_void,
    _proof_values: *mut u8,
    _values: *mut u8,
    _hint_id: u64,
    _hint_field_name: &str,
) -> u64 {
    trace!("··· {}", "set_hint_field_global_constraints: This is a mock call because there is no linked library");
    100000
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn gen_proof_c(
    _p_setup_ctx: *mut c_void,
    _p_params: *mut u8,
    _p_global_challenge: *mut u8,
    _proof_buffer: *mut u64,
    _proof_file: &str,
    _airgroup_id: u64,
    _air_id: u64,
    _instance_id: u64,
    _d_buffers: *mut c_void,
    _skip_recalculation: bool,
    _stream_id: u64,
    _const_pols_path: &str,
    _const_tree_path: &str,
) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "gen_proof: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn get_proof_c(
    _p_setup: *mut c_void,
    _proof_buffer: *mut u64,
    _proof_file: &str,
    _thread_id: u64,
    _airgroup_id: u64,
    _air_id: u64,
    _instance_id: u64,
    _d_buffers: *mut c_void,
    _mpi_node_rank: u32,
) {
    trace!("{}: ··· {}", "ffi     ", "get_proof: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn get_stream_proofs_c(_d_buffers: *mut c_void) {
    trace!("{}: ··· {}", "ffi     ", "get_stream_proofs: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn get_stream_proofs_non_blocking_c(_d_buffers: *mut c_void) {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "get_stream_proofs_non_blocking: This is a mock call because there is no linked library"
    );
}
#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn get_stream_id_proof_c(_d_buffers: *mut c_void, _stream_id: u64) {
    trace!("{}: ··· {}", "ffi     ", "get_stream_id_proof: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn gen_recursive_proof_c(
    _p_setup_ctx: *mut c_void,
    _p_address: *mut u8,
    _p_aux_trace: *mut u8,
    _p_const_pols: *mut u8,
    _p_const_tree: *mut u8,
    _p_public_inputs: *mut u8,
    _proof_buffer: *mut u64,
    _proof_file: &str,
    _global_info_file: &str,
    _airgroup_id: u64,
    _air_id: u64,
    _instance_id: u64,
    _vadcop: bool,
    _d_buffers: *mut c_void,
    _const_pols_path: &str,
    _const_tree_path: &str,
    _proof_type: &str,
) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "gen_recursive_proof: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn gen_recursive_proof_final_c(
    _p_setup_ctx: *mut c_void,
    _p_witness: *mut u8,
    _p_aux_trace: *mut u8,
    _p_const_pols: *mut u8,
    _p_const_tree: *mut u8,
    _p_public_inputs: *mut u8,
    _proof_file: &str,
    _global_info_file: &str,
    _airgroup_id: u64,
    _air_id: u64,
    _instance_id: u64,
) -> *mut c_void {
    trace!("··· {}", "gen_recursive_proof_final: This is a mock call because there is no linked library");
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
pub fn read_exec_file_c(_exec_data: *mut u64, _exec_file: *const i8, _nCols: u64) {
    trace!("{}: ··· {}", "ffi     ", "read_exec_file: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn get_committed_pols_c(
    _circomWitness: *mut u8,
    _exec_data: *mut u64,
    _witness: *mut u8,
    _pPublics: *mut u8,
    _sizeWitness: u64,
    _N: u64,
    _nPublics: u64,
    _nCols: u64,
) {
    trace!("··· {}", "get_committed_pols: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn add_publics_aggregation_c(_proof: *mut u8, _offset: u64, _publics: *mut u8, _nPublics: u64) {
    trace!("{}: ··· {}", "ffi     ", "add_publics_aggregation: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn gen_final_snark_proof_c(_circomWitnessFinal: *mut u8, _zkeyFile: &str, _outputDir: &str) {
    trace!("··· {}", "gen_final_snark_proof: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn set_log_level_c(_level: u64) {
    trace!("··· {}", "set_log_level: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn stark_verify_c(
    _verkey: &str,
    _p_proof: *mut u64,
    _p_stark_info: *mut c_void,
    _p_expressions_bin: *mut c_void,
    _p_publics: *mut u8,
    _p_proof_values: *mut u8,
    _p_challenges: *mut u8,
) -> bool {
    trace!("··· {}", "stark_verify_c: This is a mock call because there is no linked library");
    true
}

#[cfg(feature = "no_lib_link")]
pub fn stark_verify_bn128_c(
    _verkey: &str,
    _p_proof: *mut c_void,
    _p_stark_info: *mut c_void,
    _p_expressions_bin: *mut c_void,
    _p_publics: *mut u8,
) -> bool {
    trace!("··· {}", "stark_verify_bn128_c: This is a mock call because there is no linked library");
    false
}

#[cfg(feature = "no_lib_link")]
pub fn stark_verify_from_file_c(
    _verkey: &str,
    _proof: &str,
    _p_stark_info: *mut c_void,
    _p_expressions_bin: *mut c_void,
    _p_publics: *mut u8,
    _p_proof_values: *mut u8,
    _p_challenges: *mut u8,
) -> bool {
    trace!("··· {}", "stark_verify_from_file_c: This is a mock call because there is no linked library");
    true
}

#[cfg(feature = "no_lib_link")]
pub fn write_fixed_cols_bin_c(
    _binfile: &str,
    _airgroup: &str,
    _air: &str,
    _n: u64,
    _n_fixed_pols: u64,
    _fixed_pols_info: *mut c_void,
) {
    trace!("··· {}", "write_fixed_cols_bi: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn get_omp_max_threads() -> u64 {
    trace!("··· {}", "get_omp_max_threads: This is a mock call because there is no linked library");
    1
}

#[cfg(feature = "no_lib_link")]
pub fn set_omp_num_threads(_num_threads: u64) {
    trace!("··· {}", "set_omp_num_threads: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn gen_device_buffers_c(
    _max_sizes: *mut ::std::os::raw::c_void,
    _node_rank: u32,
    _node_n_processes: u32,
) -> *mut ::std::os::raw::c_void {
    trace!(
        "{}: ··· {}",
        "ffi     ",
        "gen_device_commit_buffers: This is a mock call because there is no linked library"
    );
    std::ptr::null_mut()
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn gen_device_streams_c(
    _d_buffers: *mut ::std::os::raw::c_void,
    _max_size_buffer: u64,
    _max_size_buffer_aggregation: u64,
    _max_pinned_proof_size: u64,
    _max_n_bits_ext: u64,
) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "set_max_size_thread: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn get_instances_ready_c(_d_buffers: *mut ::std::os::raw::c_void, _instances_ready: *mut i64) {
    trace!("{}: ··· {}", "ffi     ", "get_instances_ready: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn reset_device_streams_c(_d_buffers: *mut ::std::os::raw::c_void) {
    trace!("{}: ··· {}", "ffi     ", "reset_device_streams: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
pub fn check_device_memory_c(_node_rank: u32, _node_size: u32) -> u64 {
    trace!("{}: ··· {}", "ffi     ", "check_device_memory: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn get_num_gpus_c() -> u64 {
    trace!("{}: ··· {}", "ffi     ", "get_num_gpus: This is a mock call because there is no linked library");
    0
}

#[cfg(feature = "no_lib_link")]
pub fn free_device_buffers_c(_d_buffers: *mut ::std::os::raw::c_void) {
    trace!("{}: ··· {}", "ffi     ", "free_device_buffers: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn load_device_setup_c(
    _airgroup_id: u64,
    _air_id: u64,
    _proof_type: &str,
    _p_setup: *mut ::std::os::raw::c_void,
    _d_buffers: *mut ::std::os::raw::c_void,
    _verkey_root: *mut u8,
    _packed_info: *mut ::std::os::raw::c_void,
    _n_streams: u64,
) {
    trace!("{}: ··· {}", "ffi     ", "load_device_setup: This is a mock call because there is no linked library");
}

#[cfg(feature = "no_lib_link")]
#[allow(clippy::too_many_arguments)]
pub fn load_device_const_pols_c(
    _airgroup_id: u64,
    _air_id: u64,
    _initial_offset: u64,
    _d_buffers: *mut ::std::os::raw::c_void,
    _const_filename: &str,
    _const_size: u64,
    _const_tree_filename: &str,
    _const_tree_size: u64,
    _proof_type: &str,
) {
    trace!("{}: ··· {}", "ffi     ", "load_device_const_pols: This is a mock call because there is no linked library");
}
