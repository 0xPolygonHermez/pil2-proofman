#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use ::std::os::raw::c_void;

use ::std::os::raw::c_char;

include!("../bindings_starks.rs");

use std::ffi::CString;

use std::ffi::CStr;

/// A proof-done completion crossing the FFI boundary: the `(id, proof_type)` the C++ harvest
/// reports for a finished proof.
#[derive(Debug, Clone)]
pub struct CompletionMsg {
    pub id: u64,
    pub proof_type: String,
}

// Read by `on_proof_done` (C++/CUDA harvest threads), written by register/clear. The C++ callback
// pointer is never nulled, so a late harvest can race a `clear`; the RwLock keeps read-vs-drop
// exclusive so a send can't drop/free the channel mid-send -> SEGV (status=11).
static PROOFS_DONE: std::sync::RwLock<Option<crossbeam_channel::Sender<CompletionMsg>>> = std::sync::RwLock::new(None);

extern "C" fn on_proof_done(instance_id: u64, proof_type: *const c_char) {
    let proof_type_str = unsafe { CStr::from_ptr(proof_type).to_string_lossy().into_owned() };

    // Hold the read lock across the send so a concurrent `clear` can't drop the Sender mid-send.
    // Recover from poison rather than unwind: an unwind out of an `extern "C"` fn aborts.
    let guard = PROOFS_DONE.read().unwrap_or_else(|e| e.into_inner());
    if let Some(ref tx) = *guard {
        let _ = tx.send(CompletionMsg { id: instance_id, proof_type: proof_type_str });
    }
}

pub fn register_proof_done_callback_c(tx: crossbeam_channel::Sender<CompletionMsg>) {
    // Swap under the write lock: any in-flight `on_proof_done` read has released its guard
    // before this replaces (and drops) the previous Sender, so no send races the drop.
    *PROOFS_DONE.write().unwrap_or_else(|e| e.into_inner()) = Some(tx);
    unsafe {
        register_proof_done_callback(Some(on_proof_done));
    }
}

pub fn compute_const_tree_c(
    const_path: &str,
    stark_info_path: &str,
    const_tree_path: &str,
    verkey_path: &str,
) -> [u64; 4] {
    let c_const = CString::new(const_path).unwrap();
    let c_stark_info = CString::new(stark_info_path).unwrap();
    let c_tree = CString::new(const_tree_path).unwrap();
    let c_verkey = CString::new(verkey_path).unwrap();

    let mut root = [0u64; 4];
    let status = unsafe {
        build_const_tree_c(
            c_const.as_ptr(),
            c_stark_info.as_ptr(),
            c_tree.as_ptr(),
            c_verkey.as_ptr(),
            root.as_mut_ptr(),
        )
    };
    assert_eq!(
        status, 0,
        "build_const_tree_c failed (status {status}) for const {const_path}, starkinfo {stark_info_path}, verkey {verkey_path}"
    );
    root
}

pub fn generate_fflonk_zkey_c(r1cs_file: &str, ptau_file: &str, zkey_file: &str) -> i32 {
    let c_r1cs = CString::new(r1cs_file).unwrap();
    let c_ptau = CString::new(ptau_file).unwrap();
    let c_zkey = CString::new(zkey_file).unwrap();
    unsafe { fflonk_setup_c(c_r1cs.as_ptr(), c_ptau.as_ptr(), c_zkey.as_ptr()) }
}

pub fn generate_plonk_zkey_c(r1cs_file: &str, ptau_file: &str, zkey_file: &str) -> i32 {
    let c_r1cs = CString::new(r1cs_file).unwrap();
    let c_ptau = CString::new(ptau_file).unwrap();
    let c_zkey = CString::new(zkey_file).unwrap();
    unsafe { plonk_setup_c(c_r1cs.as_ptr(), c_ptau.as_ptr(), c_zkey.as_ptr()) }
}

/// Plonkish gate counts (constraints, additions) for an r1cs file — no ptau
/// needed. Returns `None` if the file could not be processed.
pub fn get_plonk_circuit_stats_c(r1cs_file: &str) -> Option<(u64, u64)> {
    let c_r1cs = CString::new(r1cs_file).unwrap();
    let mut n_constraints: u64 = 0;
    let mut n_additions: u64 = 0;
    let ret = unsafe { plonk_circuit_stats_c(c_r1cs.as_ptr(), &mut n_constraints, &mut n_additions) };
    (ret == 0).then_some((n_constraints, n_additions))
}

pub fn initialize_agg_readiness_tracker_c() {
    unsafe {
        initialize_agg_readiness_tracker();
    }
}

pub fn free_agg_readiness_tracker_c() {
    unsafe {
        free_agg_readiness_tracker();
    }
}

pub fn agg_is_ready_c() -> i32 {
    unsafe { agg_is_ready() }
}

pub fn reset_agg_readiness_tracker_c() {
    unsafe {
        reset_agg_readiness_tracker();
    }
}

pub fn launch_callback_c(instance_id: u64, proof_type: &str) {
    let proof_type_str = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_str.as_ptr() as *mut std::os::raw::c_char;
    unsafe {
        launch_callback(instance_id, proof_type_ptr);
    }
}

pub fn clear_proof_done_callback_c() {
    // Take the write lock so the Sender is dropped only when no `on_proof_done` holds the read lock
    // mid-send. A late harvest after clearing just observes `None` (the C++ pointer is never nulled).
    *PROOFS_DONE.write().unwrap_or_else(|e| e.into_inner()) = None;
}

#[allow(clippy::too_many_arguments)]
pub fn stark_info_new_c(
    filename: &str,
    recursive_final: bool,
    recursive: bool,
    verify_constraints: bool,
    verify: bool,
    gpu: bool,
    preallocate: bool,
    // Table airs: proved at most once, so the const pols need not survive the proof.
    single_use: bool,
) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();

        stark_info_new(
            filename.as_ptr() as *mut std::os::raw::c_char,
            recursive_final,
            recursive,
            verify_constraints,
            verify,
            gpu,
            preallocate,
            single_use,
        )
    }
}

pub fn get_map_totaln_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_map_total_n(p_stark_info) }
}

pub fn get_map_totaln_contributions_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_map_total_n_contributions(p_stark_info) }
}

pub fn get_tree_size_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_tree_size(p_stark_info) }
}

pub fn get_map_totaln_custom_commits_fixed_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_map_total_n_custom_commits_fixed(p_stark_info) }
}

pub fn get_proof_size_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_proof_size(p_stark_info) }
}

pub fn set_hash_family_c(family: &str) {
    let fam: u8 = match family {
        "Poseidon1" => 1,
        "Poseidon2" => 2,
        "blake3" => 3,
        other => panic!(
            "set_hash_family_c: unknown hash family {other:?} (expected \"Poseidon1\", \"Poseidon2\" or \"blake3\")"
        ),
    };
    unsafe { set_hash_family(fam) }
}

pub fn get_proof_pinned_size_c(p_stark_info: *mut c_void) -> u64 {
    unsafe { get_proof_pinned_size(p_stark_info) }
}

pub fn register_host_memory_c(ptr: *mut c_void, size: u64) -> bool {
    unsafe { register_host_memory(ptr, size) != 0 }
}

pub fn unregister_host_memory_c(ptr: *mut c_void) {
    unsafe {
        unregister_host_memory(ptr);
    }
}

/// Block until the commit work on `stream_id` has completed on the GPU, so a
/// shared trace buffer whose async H2D copy rode that stream can be safely
/// reused. No-op on the CPU backend (the symbol is GPU-only; callers gate on
/// `pctx.gpu`).
pub fn wait_trace_h2d_done_c(d_buffers: *mut c_void, stream_id: u64) {
    unsafe {
        wait_trace_h2d_done(d_buffers, stream_id);
    }
}

pub fn set_memory_expressions_c(p_stark_info: *mut c_void, n_tmp1: u64, n_tmp3: u64) {
    unsafe {
        set_memory_expressions(p_stark_info, n_tmp1, n_tmp3);
    }
}

pub fn stark_info_free_c(p_stark_info: *mut c_void) {
    unsafe {
        stark_info_free(p_stark_info);
    }
}

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

pub fn get_const_size_c(pStarkInfo: *mut c_void) -> u64 {
    unsafe { get_const_size(pStarkInfo) }
}

pub fn get_const_tree_size_c(pStarkInfo: *mut c_void) -> u64 {
    unsafe { get_const_tree_size(pStarkInfo) }
}

pub fn calculate_words_per_row_c(pStarkInfo: *mut c_void, const_pols_path: &str) -> u64 {
    unsafe {
        let const_pols_path_cstr = CString::new(const_pols_path).unwrap();
        calculate_words_per_row(pStarkInfo, const_pols_path_cstr.as_ptr() as *mut std::os::raw::c_char)
    }
}

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

pub fn init_gpu_setup_c(arity: u64) {
    unsafe {
        init_gpu_setup(arity);
    }
}

pub fn pack_const_pols_c(pStarkinfo: *mut c_void, pConstPols: *mut u8, constFile: &str) {
    let const_file_cstr: CString = CString::new(constFile).unwrap();
    unsafe {
        pack_const_pols(
            pStarkinfo,
            pConstPols as *mut std::os::raw::c_void,
            const_file_cstr.as_ptr() as *mut std::os::raw::c_char,
        );
    }
}

pub fn tile_const_pols_c(
    pStarkInfo: *mut c_void,
    pConstPols: *mut u8,
    constFile: &str,
    pConstTree: *mut u8,
    constTreeFile: &str,
    unified_buffer_gpu: *mut c_void,
) {
    let const_file_cstr: CString = CString::new(constFile).unwrap();
    let const_tree_file_cstr: CString = CString::new(constTreeFile).unwrap();
    unsafe {
        tile_const_pols(
            pStarkInfo,
            pConstPols as *mut std::os::raw::c_void,
            const_file_cstr.as_ptr() as *mut std::os::raw::c_char,
            pConstTree as *mut std::os::raw::c_void,
            const_tree_file_cstr.as_ptr() as *mut std::os::raw::c_char,
            unified_buffer_gpu,
        );
    }
}

pub fn prepare_blocks_c(pol: *mut u64, N: u64, nCols: u64, unified_buffer_gpu: *mut c_void) {
    unsafe {
        prepare_blocks(pol, N, nCols, unified_buffer_gpu);
    }
}

pub fn calculate_const_tree_c(
    pStarkInfo: *mut c_void,
    pConstPols: *mut u8,
    pConstPolsTreeAddress: *mut u8,
    unified_buffer_gpu: *mut c_void,
) {
    unsafe {
        calculate_const_tree(
            pStarkInfo,
            pConstPols as *mut std::os::raw::c_void,
            pConstPolsTreeAddress as *mut std::os::raw::c_void,
            unified_buffer_gpu,
        );
    }
}

pub fn calculate_const_tree_bn128_c(pStarkInfo: *mut c_void, pConstPols: *mut u8, pConstPolsTreeAddress: *mut u8) {
    unsafe {
        calculate_const_tree_bn128(
            pStarkInfo,
            pConstPols as *mut std::os::raw::c_void,
            pConstPolsTreeAddress as *mut std::os::raw::c_void,
        );
    }
}

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

pub fn verify_root_bn128_from_tree_c(tree_filename: &str, expected_root: &str) -> bool {
    unsafe {
        let tree_filename_cstr = CString::new(tree_filename).unwrap();
        let expected_root_cstr = CString::new(expected_root).unwrap();

        verify_root_bn128_from_tree(
            tree_filename_cstr.as_ptr() as *mut std::os::raw::c_char,
            expected_root_cstr.as_ptr() as *mut std::os::raw::c_char,
        )
    }
}

pub fn expressions_bin_new_c(filename: &str, global: bool, verify: bool) -> *mut c_void {
    unsafe {
        let filename = CString::new(filename).unwrap();

        expressions_bin_new(filename.as_ptr() as *mut std::os::raw::c_char, global, verify)
    }
}

pub fn get_max_n_tmp1_c(p_expressions_bin: *mut c_void) -> u64 {
    unsafe { get_max_n_tmp1(p_expressions_bin) }
}

pub fn get_max_n_tmp3_c(p_expressions_bin: *mut c_void) -> u64 {
    unsafe { get_max_n_tmp3(p_expressions_bin) }
}

pub fn get_max_n_args_c(p_expressions_bin: *mut c_void) -> u64 {
    unsafe { get_max_args(p_expressions_bin) }
}

pub fn get_max_n_ops_c(p_expressions_bin: *mut c_void) -> u64 {
    unsafe { get_max_ops(p_expressions_bin) }
}

pub fn get_operations_quotient_c(p_expressions_bin: *mut c_void, p_stark_info: *mut c_void) -> u64 {
    unsafe { get_operations_quotient(p_expressions_bin, p_stark_info) }
}

pub fn expressions_bin_free_c(p_expressions_bin: *mut c_void) {
    unsafe {
        expressions_bin_free(p_expressions_bin);
    }
}

pub fn n_hint_ids_by_name_c(p_expressions_bin: *mut c_void, hint_name: &str) -> u64 {
    let name = CString::new(hint_name).unwrap();
    unsafe { n_hints_by_name(p_expressions_bin, name.as_ptr() as *mut std::os::raw::c_char) }
}

pub fn get_hint_ids_by_name_c(p_expressions_bin: *mut c_void, hint_ids: *mut u64, hint_name: &str) {
    let name = CString::new(hint_name).unwrap();
    unsafe {
        get_hint_ids_by_name(p_expressions_bin, hint_ids, name.as_ptr() as *mut std::os::raw::c_char);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_hint_field_c(
    p_setup_ctx: *mut c_void,
    airgroup_id: u64,
    air_id: u64,
    p_steps_params: *mut u8,
    hint_field_values: *mut c_void,
    hint_id: u64,
    hint_field_name: &str,
    hint_options: *mut u8,
    d_buffers: *mut c_void,
    stream_id: u64,
    constant: bool,
) {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe {
        get_hint_field(
            p_setup_ctx,
            airgroup_id,
            air_id,
            p_steps_params as *mut std::os::raw::c_void,
            hint_field_values,
            hint_id,
            field_name.as_ptr() as *mut std::os::raw::c_char,
            hint_options as *mut std::os::raw::c_void,
            d_buffers,
            stream_id,
            constant,
        )
    }
}

pub fn get_hint_field_values_c(p_setup_ctx: *mut c_void, hint_id: u64, hint_field_name: &str) -> u64 {
    let field_name = CString::new(hint_field_name).unwrap();
    unsafe { get_hint_field_values(p_setup_ctx, hint_id, field_name.as_ptr() as *mut std::os::raw::c_char) }
}

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

pub fn get_hint_field_id_c(p_setup_ctx: *mut c_void, hint_id: u64, hint_field_name: &str) -> u64 {
    unsafe {
        let field_name = CString::new(hint_field_name).unwrap();
        get_hint_id(p_setup_ctx, hint_id, field_name.as_ptr() as *mut std::os::raw::c_char)
    }
}

pub fn calculate_impols_expressions_c(p_setup: *mut c_void, step: u64, p_steps_params: *mut u8) {
    unsafe {
        calculate_impols_expressions(p_setup, step, p_steps_params as *mut std::os::raw::c_void);
    }
}

pub fn calculate_witness_expressions_c(p_setup: *mut c_void, p_steps_params: *mut u8) {
    unsafe {
        calculate_witness_expr(p_setup, p_steps_params as *mut std::os::raw::c_void);
    }
}

pub fn custom_commit_size_c(p_setup: *mut c_void, commit_id: u64) -> u64 {
    unsafe { custom_commit_size(p_setup, commit_id) }
}

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

#[allow(clippy::too_many_arguments)]
pub fn write_custom_commit_c(
    root: *mut u8,
    arity: u64,
    n_bits: u64,
    n_bits_ext: u64,
    n_cols: u64,
    d_buffers: *mut c_void,
    buffer: *mut u8,
    buffer_file: &str,
) {
    let buffer_file_name = CString::new(buffer_file).unwrap();
    unsafe {
        write_custom_commit(
            root as *mut std::os::raw::c_void,
            arity,
            n_bits,
            n_bits_ext,
            n_cols,
            d_buffers,
            buffer as *mut std::os::raw::c_void,
            buffer_file_name.as_ptr() as *mut std::os::raw::c_char,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn commit_witness_c(
    p_setup: *mut c_void,
    p_params: *mut u8,
    instance_id: u64,
    airgroup_id: u64,
    air_id: u64,
    root: *mut u8,
    d_buffers: *mut c_void,
    custom_commits_fixed_path: &str,
) -> u64 {
    let custom_commits_path_name = CString::new(custom_commits_fixed_path).unwrap();
    let custom_commits_path_ptr = custom_commits_path_name.as_ptr() as *mut std::os::raw::c_char;
    unsafe {
        commit_witness(
            p_setup,
            p_params as *mut std::os::raw::c_void,
            instance_id,
            airgroup_id,
            air_id,
            root as *mut std::os::raw::c_void,
            d_buffers,
            custom_commits_path_ptr,
        )
    }
}

pub fn get_n_constraints_c(p_setup: *mut c_void) -> u64 {
    unsafe { get_n_constraints(p_setup) }
}

pub fn get_constraints_lines_sizes_c(p_setup: *mut c_void, constraints_sizes: *mut u64) {
    unsafe {
        get_constraints_lines_sizes(p_setup, constraints_sizes);
    }
}

pub fn get_constraints_lines_c(p_setup: *mut c_void, constraints_lines: *mut *mut u8) {
    unsafe {
        get_constraints_lines(p_setup, constraints_lines);
    }
}

pub fn initialize_instance_c(
    p_setup: *mut c_void,
    airgroup_id: u64,
    air_id: u64,
    instance_id: u64,
    p_steps_params: *mut u8,
    d_buffers: *mut c_void,
    custom_commits_fixed_path: &str,
) -> u64 {
    let custom_commits_path_name = CString::new(custom_commits_fixed_path).unwrap();
    let custom_commits_path_ptr = custom_commits_path_name.as_ptr() as *mut std::os::raw::c_char;
    unsafe {
        initialize_instance(
            p_setup,
            airgroup_id,
            air_id,
            instance_id,
            p_steps_params as *mut std::os::raw::c_void,
            d_buffers,
            custom_commits_path_ptr,
        )
    }
}

pub fn calculate_trace_instance_c(
    p_setup: *mut c_void,
    airgroup_id: u64,
    air_id: u64,
    p_steps_params: *mut u8,
    d_buffers: *mut c_void,
    stream_id: u64,
) {
    unsafe {
        calculate_trace_instance(
            p_setup,
            airgroup_id,
            air_id,
            p_steps_params as *mut std::os::raw::c_void,
            d_buffers,
            stream_id,
        )
    }
}

pub fn verify_constraints_c(
    p_setup: *mut c_void,
    airgroup_id: u64,
    air_id: u64,
    p_steps_params: *mut u8,
    constraints_info: *mut c_void,
    d_buffers: *mut c_void,
    stream_id: u64,
) {
    unsafe {
        verify_constraints(
            p_setup,
            airgroup_id,
            air_id,
            p_steps_params as *mut std::os::raw::c_void,
            constraints_info,
            d_buffers,
            stream_id,
        );
    }
}

pub fn get_n_global_constraints_c(p_global_constraints_bin: *mut c_void) -> u64 {
    unsafe { get_n_global_constraints(p_global_constraints_bin) }
}

pub fn get_global_constraints_lines_sizes_c(p_global_constraints_bin: *mut c_void, global_constraints_sizes: *mut u64) {
    unsafe {
        get_global_constraints_lines_sizes(p_global_constraints_bin, global_constraints_sizes);
    }
}

pub fn get_global_constraints_lines_c(p_global_constraints_bin: *mut c_void, global_constraints_lines: *mut *mut u8) {
    unsafe {
        get_global_constraints_lines(p_global_constraints_bin, global_constraints_lines);
    }
}

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
    custom_commits_fixed_path: &str,
    // true: seed the transcript from this AIR's own verkey + publics instead of the
    // global challenge (self-contained proof, no contributions phase).
    self_contained: bool,
) -> u64 {
    let proof_file_name = CString::new(proof_file).unwrap();
    let proof_file_ptr = proof_file_name.as_ptr() as *mut std::os::raw::c_char;

    let const_filename_name = CString::new(const_pols_path).unwrap();
    let const_filename_ptr = const_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let const_tree_filename_name = CString::new(const_tree_path).unwrap();
    let const_tree_filename_ptr = const_tree_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let custom_commits_path_name = CString::new(custom_commits_fixed_path).unwrap();
    let custom_commits_path_ptr = custom_commits_path_name.as_ptr() as *mut std::os::raw::c_char;

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
            custom_commits_path_ptr,
            self_contained,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_stream_proofs_c(d_buffers: *mut c_void) {
    unsafe {
        get_stream_proofs(d_buffers);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_stream_proofs_non_blocking_c(d_buffers: *mut c_void) {
    unsafe {
        get_stream_proofs_non_blocking(d_buffers);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_stream_id_proof_c(d_buffers: *mut c_void, stream_id: u64) {
    unsafe {
        get_stream_id_proof(d_buffers, stream_id);
    }
}

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
    airgroup_id: u64,
    air_id: u64,
    instance_id: u64,
    vadcop: bool,
    d_buffers: *mut c_void,
    const_pols_path: &str,
    const_tree_path: &str,
    proof_type: &str,
    force_recursive_stream: bool,
    recurser_id: &str,
    stream_id: u64,
) -> u64 {
    let proof_file_name = CString::new(proof_file).unwrap();
    let proof_file_ptr = proof_file_name.as_ptr() as *mut std::os::raw::c_char;

    let const_filename_name = CString::new(const_pols_path).unwrap();
    let const_filename_ptr = const_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let const_tree_filename_name = CString::new(const_tree_path).unwrap();
    let const_tree_filename_ptr = const_tree_filename_name.as_ptr() as *mut std::os::raw::c_char;

    let proof_type_name = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_name.as_ptr() as *mut std::os::raw::c_char;

    let recurser_id_name = CString::new(recurser_id).unwrap();
    let recurser_id_ptr = recurser_id_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        gen_recursive_proof(
            p_setup_ctx,
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
            force_recursive_stream,
            recurser_id_ptr,
            stream_id,
        )
    }
}

/// Scheduler: reserve the best free stream for this key without blocking.
/// Returns `u32::MAX` when nothing is free right now (caller retries).
pub fn reserve_best_stream_nonblock_c(
    d_buffers: *mut c_void,
    airgroup_id: u64,
    air_id: u64,
    proof_type: &str,
    recursive: bool,
    force_recursive: bool,
) -> u32 {
    let proof_type_name = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_name.as_ptr() as *mut std::os::raw::c_char;
    unsafe { reserve_best_stream_nonblock(d_buffers, airgroup_id, air_id, proof_type_ptr, recursive, force_recursive) }
}

/// Scheduler warm fast path: reserve `stream_id` iff it is free right now *and* its buffer size
/// class holds this air. Returns true on success.
pub fn reserve_stream_if_free_c(
    d_buffers: *mut c_void,
    stream_id: u32,
    airgroup_id: u64,
    air_id: u64,
    proof_type: &str,
    force_recursive: bool,
) -> bool {
    let proof_type_name = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe { reserve_stream_if_free(d_buffers, stream_id, airgroup_id, air_id, proof_type_ptr, force_recursive) != 0 }
}

/// Scheduler: give back a reservation the caller never launched on. A reserved stream reads as busy
/// to every selection pass, so failing between reserve and launch without this would strand the slot
/// for the process lifetime. Only releases a stream still in the reserved state, so it is safe to
/// call unconditionally on an error path.
pub fn release_stream_reservation_c(d_buffers: *mut c_void, stream_id: u32) {
    unsafe { release_stream_reservation(d_buffers, stream_id) }
}

pub fn calculate_const_tree_fixed_c(
    p_setup_ctx: *mut c_void,
    airgroup_id: u64,
    air_id: u64,
    proof_type: &str,
    d_buffers: *mut c_void,
) {
    let proof_type_name = CString::new(proof_type).unwrap();
    let proof_type_ptr = proof_type_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        calculate_const_tree_fixed(p_setup_ctx, airgroup_id, air_id, proof_type_ptr, d_buffers);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gen_recursive_proof_final_c(
    p_setup_ctx: *mut c_void,
    p_witness: *mut u8,
    p_aux_trace: *mut u8,
    p_const_pols: *mut u8,
    p_const_tree: *mut u8,
    p_public_inputs: *mut u8,
    proof_file: &str,
    airgroup_id: u64,
    air_id: u64,
    instance_id: u64,
    prover_buffer_size: u64,
    d_buffers: *mut u8,
) -> *mut c_void {
    let proof_file_name = CString::new(proof_file).unwrap();
    let proof_file_ptr = proof_file_name.as_ptr() as *mut std::os::raw::c_char;

    unsafe {
        gen_recursive_proof_final(
            p_setup_ctx,
            airgroup_id,
            air_id,
            instance_id,
            p_witness as *mut std::os::raw::c_void,
            p_aux_trace as *mut std::os::raw::c_void,
            p_const_pols as *mut std::os::raw::c_void,
            p_const_tree as *mut std::os::raw::c_void,
            p_public_inputs as *mut std::os::raw::c_void,
            proof_file_ptr,
            prover_buffer_size,
            d_buffers as *mut std::os::raw::c_void,
        )
    }
}

pub fn read_exec_file_c(exec_data: *mut u64, exec_file: &str, nCols: u64) {
    let exec_file_name = CString::new(exec_file).unwrap();
    let exec_file_ptr = exec_file_name.as_ptr() as *mut std::os::raw::c_char;
    unsafe {
        read_exec_file(exec_data, exec_file_ptr, nCols);
    }
}

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

pub fn init_final_snark_prover_c(zkeyFile: &str, d_buffers_recursivef: *mut c_void) -> *mut c_void {
    let zkey_file_name = CString::new(zkeyFile).unwrap();
    let zkey_file_ptr = zkey_file_name.as_ptr() as *mut std::os::raw::c_char;
    unsafe { init_final_snark_prover(zkey_file_ptr, d_buffers_recursivef) }
}

pub fn get_snark_protocol_id_c(snark_prover: *mut c_void) -> u64 {
    if snark_prover.is_null() {
        return 0;
    }
    unsafe { get_snark_protocol_id(snark_prover) }
}

pub fn free_final_snark_prover_c(snark_prover: *mut c_void) {
    unsafe { free_final_snark_prover(snark_prover) }
}

pub fn pre_allocate_final_snark_prover_c(
    snark_prover: *mut c_void,
    unified_buffer_gpu: *mut c_void,
    d_buffers_recursivef: *mut c_void,
) {
    unsafe { pre_allocate_final_snark_prover(snark_prover, unified_buffer_gpu, d_buffers_recursivef) }
}

pub fn gen_final_snark_proof_c(
    prover: *mut c_void,
    circomWitnessFinal: *mut u8,
    proof: *mut u8,
    publics_snark: *mut u8,
    d_buffers_recursivef: *mut c_void,
) {
    unsafe {
        gen_final_snark_proof(
            prover,
            circomWitnessFinal as *mut std::os::raw::c_void,
            proof,
            publics_snark,
            d_buffers_recursivef,
        );
    }
}

pub fn snark_proof_bytes_to_json_c(proof_bytes: &[u8], public_bytes: &[u8], protocol_id: i32) -> (String, String) {
    unsafe {
        let mut proof_json_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut publics_json_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();

        snark_proof_bytes_to_json(
            proof_bytes.as_ptr(),
            proof_bytes.len() as u64,
            public_bytes.as_ptr(),
            public_bytes.len() as u64,
            protocol_id,
            &mut proof_json_ptr as *mut *mut std::os::raw::c_char,
            &mut publics_json_ptr as *mut *mut std::os::raw::c_char,
        );

        // Convert C strings to Rust strings
        let proof_json = if !proof_json_ptr.is_null() {
            CStr::from_ptr(proof_json_ptr).to_string_lossy().into_owned()
        } else {
            String::new()
        };

        let publics_json = if !publics_json_ptr.is_null() {
            CStr::from_ptr(publics_json_ptr).to_string_lossy().into_owned()
        } else {
            String::new()
        };

        // Free the C strings
        if !proof_json_ptr.is_null() {
            free_json_string(proof_json_ptr);
        }
        if !publics_json_ptr.is_null() {
            free_json_string(publics_json_ptr);
        }

        (proof_json, publics_json)
    }
}

pub fn set_log_level_c(level: u64) {
    unsafe {
        setLogLevel(level);
    }
}

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

pub fn get_omp_max_threads_c() -> u64 {
    unsafe { get_omp_max_threads() }
}

pub fn set_omp_num_threads_c(num_threads: u64) {
    unsafe {
        set_omp_num_threads(num_threads);
    }
}

pub fn gen_device_buffers_c(
    node_rank: u32,
    node_n_processes: u32,
    numa_nodes: &[i32],
    arity: u32,
    max_n_bits_ext: u32,
) -> *mut ::std::os::raw::c_void {
    debug_assert!(
        numa_nodes.len() >= node_n_processes as usize,
        "numa_nodes slice length ({}) must be at least node_n_processes ({})",
        numa_nodes.len(),
        node_n_processes
    );
    unsafe { gen_device_buffers(node_rank, node_n_processes, numa_nodes.as_ptr(), arity, max_n_bits_ext) }
}

pub fn use_packed_trace_c(d_commit_buffers: *mut ::std::os::raw::c_void, packed_trace: bool) {
    unsafe {
        use_packed_trace(d_commit_buffers, packed_trace);
    }
}

pub fn register_instruction_table_c(
    d_commit_buffers: *mut ::std::os::raw::c_void,
    airgroup_id: u64,
    air_id: u64,
    table: &[u64],
    num_entries: u64,
    words_per_entry: u64,
) {
    unsafe {
        register_instruction_table(d_commit_buffers, airgroup_id, air_id, table.as_ptr(), num_entries, words_per_entry);
    }
}

pub fn gen_device_buffers_recursivef_c(
    p_setup_ctx: *mut u8,
    prover_buffer_size: u64,
    d_commit_buffers: *mut u8,
    verkey: &str,
) -> *mut u8 {
    let verkey_cstr = CString::new(verkey).unwrap();

    unsafe {
        gen_device_buffers_recursivef(
            p_setup_ctx as *mut c_void,
            prover_buffer_size,
            d_commit_buffers as *mut c_void,
            verkey_cstr.as_ptr() as *mut std::os::raw::c_char,
        ) as *mut u8
    }
}

pub fn free_device_buffers_recursivef_c(d_buffers: *mut c_void) {
    unsafe { free_device_buffers_recursivef(d_buffers) }
}

pub fn gen_device_streams_c(
    d_buffers: *mut ::std::os::raw::c_void,
    aux_trace_sizes: &[u64],
    n_recursive_streams: u64,
    max_size_buffer_aggregation: u64,
    max_pinned_proof_size: u64,
    merkle_tree_arity: u64,
) -> u64 {
    unsafe {
        gen_device_streams(
            d_buffers,
            aux_trace_sizes.len() as u64,
            n_recursive_streams,
            aux_trace_sizes.as_ptr(),
            max_size_buffer_aggregation,
            max_pinned_proof_size,
            merkle_tree_arity,
        )
    }
}

pub fn alloc_device_large_buffers_c(
    d_buffers: *mut ::std::os::raw::c_void,
    aux_trace_recursive_area: u64,
    const_pols_area: u64,
    const_pols_aggregation_area: u64,
) {
    unsafe {
        alloc_device_large_buffers(d_buffers, aux_trace_recursive_area, const_pols_area, const_pols_aggregation_area);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn get_instances_ready_c(d_buffers: *mut ::std::os::raw::c_void, instances_ready: *mut i64) {
    unsafe {
        get_instances_ready(d_buffers, instances_ready);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reset_device_streams_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe {
        reset_device_streams(d_buffers);
    }
}

pub fn check_device_memory_c(node_rank: u32, node_size: u32) -> u64 {
    unsafe { check_device_memory(node_rank, node_size) }
}

pub fn get_num_gpus_c() -> u64 {
    unsafe { get_num_gpus() }
}

pub fn set_gpu_mode_c(use_gpu: bool) -> bool {
    unsafe { set_gpu_mode(use_gpu) }
}

pub fn get_unified_buffer_gpu_c(d_buffers: *mut ::std::os::raw::c_void) -> *mut ::std::os::raw::c_void {
    unsafe { get_unified_buffer_gpu(d_buffers) }
}

pub fn get_unified_buffer_gpu_size_c(d_buffers: *mut ::std::os::raw::c_void) -> u64 {
    unsafe { get_unified_buffer_gpu_size(d_buffers) }
}

pub fn acquire_first_gpu_buffer_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe { acquire_first_gpu_buffer(d_buffers) }
}

pub fn release_first_gpu_buffer_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe { release_first_gpu_buffer(d_buffers) }
}

pub fn is_first_gpu_buffer_borrowed_c(d_buffers: *mut ::std::os::raw::c_void) -> bool {
    unsafe { is_first_gpu_buffer_borrowed(d_buffers) != 0 }
}

/// Device of the first GPU (my_gpu_ids[0]); not always 0 (NUMA can reorder).
pub fn get_first_gpu_id_c(d_buffers: *mut ::std::os::raw::c_void) -> u32 {
    unsafe { get_first_gpu_id(d_buffers) }
}

/// Unified buffer of the first GPU (my_gpu_ids[0]). Does not touch the caller's current device.
pub fn get_first_gpu_buffer_c(d_buffers: *mut ::std::os::raw::c_void) -> *mut ::std::os::raw::c_void {
    unsafe { get_first_gpu_buffer(d_buffers) }
}

/// Byte offset of the aggregation const-pols region within the first GPU's unified buffer.
pub fn get_const_pols_aggregation_offset_c(d_buffers: *mut ::std::os::raw::c_void) -> u64 {
    unsafe { get_const_pols_aggregation_offset(d_buffers) }
}

pub fn stream_commit_pause_c() {
    unsafe { stream_commit_pause() }
}

pub fn get_stream_commit_slots_c(d_buffers: *mut ::std::os::raw::c_void) -> u64 {
    unsafe { get_stream_commit_slots(d_buffers) }
}

pub fn get_stream_commit_floor_c(d_buffers: *mut ::std::os::raw::c_void) -> u64 {
    unsafe { get_stream_commit_floor(d_buffers) }
}

pub fn stream_commit_slot_bytes_c(n_bits: u64, n_bits_ext: u64, n_cols: u64, words_per_row: u64) -> u64 {
    unsafe { stream_commit_slot_bytes(n_bits, n_bits_ext, n_cols, words_per_row) }
}

pub fn configure_stream_commit_slots_c(d_buffers: *mut ::std::os::raw::c_void, n_slots: u64, slot_bytes: u64) {
    unsafe { configure_stream_commit_slots(d_buffers, n_slots, slot_bytes) }
}

pub fn set_pipeline_mode_c(d_buffers: *mut ::std::os::raw::c_void, enable: bool) {
    unsafe { set_pipeline_mode(d_buffers, enable) }
}

pub fn harvest_pipeline_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe { harvest_pipeline(d_buffers) }
}

pub fn configure_prefetch_zone_c(
    d_buffers: *mut ::std::os::raw::c_void,
    witness_bytes: u64,
    fixed_bytes: u64,
    packed_bytes: u64,
) {
    unsafe { configure_prefetch_zone(d_buffers, witness_bytes, fixed_bytes, packed_bytes) }
}

pub fn prefetch_fixed_c(
    d_buffers: *mut ::std::os::raw::c_void,
    airgroup_id: u64,
    air_id: u64,
    const_tree_path: &str,
    bytes: u64,
) -> i64 {
    let path = std::ffi::CString::new(const_tree_path).unwrap();
    unsafe { prefetch_fixed(d_buffers, airgroup_id, air_id, path.as_ptr(), bytes) }
}

pub fn prefetch_witness_c(
    p_setup_ctx: *mut ::std::os::raw::c_void,
    d_buffers: *mut ::std::os::raw::c_void,
    instance_id: u64,
    airgroup_id: u64,
    air_id: u64,
    trace: *mut ::std::os::raw::c_void,
) -> i64 {
    unsafe { prefetch_witness(p_setup_ctx, d_buffers, instance_id, airgroup_id, air_id, trace) }
}

#[allow(clippy::too_many_arguments)]
pub fn commit_witness_streaming_c(
    d_buffers: *mut ::std::os::raw::c_void,
    slot_idx: u64,
    airgroup_id: u64,
    air_id: u64,
    packed: *mut ::std::os::raw::c_void,
    n_bits: u64,
    n_bits_ext: u64,
    n_cols: u64,
    words_per_row: u64,
    col_widths: *mut ::std::os::raw::c_void,
    root: *mut ::std::os::raw::c_void,
) -> i64 {
    unsafe {
        commit_witness_streaming(
            d_buffers,
            slot_idx,
            airgroup_id,
            air_id,
            packed,
            n_bits,
            n_bits_ext,
            n_cols,
            words_per_row,
            col_widths,
            root,
        )
    }
}

pub fn get_unified_buffer_gpu_for_recursivef_c(
    d_buffers: *mut ::std::os::raw::c_void,
    d_buffers_recursivef: *mut ::std::os::raw::c_void,
) -> *mut ::std::os::raw::c_void {
    unsafe { get_unified_buffer_gpu_for_recursivef(d_buffers, d_buffers_recursivef) }
}

pub fn alloc_fixed_pols_buffer_gpu_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe {
        alloc_fixed_pols_buffer_gpu(d_buffers);
    }
}

pub fn free_fixed_pols_buffer_gpu_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe {
        free_fixed_pols_buffer_gpu(d_buffers);
    }
}

pub fn load_fixed_pols_recursivef_c(
    pSetupCtx: *mut ::std::os::raw::c_void,
    pConstTree: *mut ::std::os::raw::c_void,
    d_buffers: *mut ::std::os::raw::c_void,
) {
    unsafe {
        load_fixed_pols_recursivef(pSetupCtx, pConstTree, d_buffers);
    }
}

pub fn free_device_buffers_c(d_buffers: *mut ::std::os::raw::c_void) {
    unsafe {
        free_device_buffers(d_buffers);
    }
}

pub fn load_device_setup_c(
    airgroup_id: u64,
    air_id: u64,
    proof_type: &str,
    p_setup: *mut ::std::os::raw::c_void,
    d_buffers: *mut ::std::os::raw::c_void,
    verkey_root: *mut u8,
    packed_info: *mut ::std::os::raw::c_void,
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
        );
    }
}

#[allow(clippy::too_many_arguments)]
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
    only_first_gpu: bool,
    // This air shares its slot with one already uploaded: record the offsets, transfer nothing.
    already_loaded: bool,
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
            only_first_gpu,
            already_loaded,
        );
    }
}
