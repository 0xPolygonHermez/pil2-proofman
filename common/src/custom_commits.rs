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
