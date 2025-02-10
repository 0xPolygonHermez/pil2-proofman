use p3_field::Field;
use proofman_starks_lib_c::{starks_new_c, write_custom_commit_c};

use crate::{trace::Trace, ProofCtx, SetupCtx};

pub fn write_custom_commit_trace<F: Field>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    custom_trace: &mut dyn Trace<F>,
    name: &str,
    mut hash_file: [u8; 32],
) {
    let setup = sctx.get_setup(custom_trace.airgroup_id(), custom_trace.air_id());

    let file_name = pctx.get_custom_commits_fixed_buffer(name);

    let file_name_str = match file_name {
        Some(path) => path.to_str().expect("Invalid UTF-8 in path"),
        None => panic!("Custom commit fixed buffer not found"),
    };

    let p_starks = starks_new_c((&setup.p_setup).into(), std::ptr::null_mut());
    let buffer: Vec<F> = custom_trace.get_buffer();

    let commit_id = custom_trace.commit_id().unwrap() as u64;

    write_custom_commit_c(p_starks, commit_id, buffer.as_ptr() as *mut u8, file_name_str, hash_file.as_mut_ptr());
}
