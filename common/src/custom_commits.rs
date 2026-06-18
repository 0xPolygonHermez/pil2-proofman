use std::path::Path;

use fields::PrimeField64;
use proofman_starks_lib_c::write_custom_commit_c;

use crate::trace::Trace;
use crate::ProofmanResult;
use crate::ProofCtx;

pub fn write_custom_commit_trace<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    custom_trace: &mut dyn Trace<F>,
    blowup_factor: u64,
    merkle_tree_arity: u64,
    file_name: &Path,
) -> ProofmanResult<Vec<F>> {
    let buffer = custom_trace.get_buffer();
    let arity = merkle_tree_arity;
    let n = custom_trace.num_rows() as u64;
    let n_extended = blowup_factor * custom_trace.num_rows() as u64;
    let n_bits = n.trailing_zeros() as u64;
    let n_bits_ext = n_extended.trailing_zeros() as u64;
    let n_cols = custom_trace.num_cols() as u64;
    let mut root = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];

    write_custom_commit_c(
        root.as_mut_ptr() as *mut u8,
        arity,
        n_bits,
        n_bits_ext,
        n_cols,
        pctx.get_device_buffers_ptr(),
        buffer.as_ptr() as *mut u8,
        file_name.to_str().expect("Invalid file name"),
    );

    Ok(root)
}

fn num_nodes_mt(height: u64, arity: u64) -> u64 {
    const HASH_SIZE: u64 = 4;
    let mut num_nodes = height;
    let mut nodes_level = height;
    while nodes_level > 1 {
        let extra_zeros = (arity - (nodes_level % arity)) % arity;
        num_nodes += extra_zeros;
        let next_n = nodes_level.div_ceil(arity);
        num_nodes += next_n;
        nodes_level = next_n;
    }
    num_nodes * HASH_SIZE
}

pub fn custom_commit_num_elements(n: u64, n_extended: u64, n_cols: u64, arity: u64) -> u64 {
    (n + n_extended) * n_cols + num_nodes_mt(n_extended, arity)
}

pub fn custom_commit_file_size_bytes(n: u64, n_extended: u64, n_cols: u64, arity: u64) -> u64 {
    (custom_commit_num_elements(n, n_extended, n_cols, arity) + 4) * 8
}
