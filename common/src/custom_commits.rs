use std::fs::File;
use std::io::Read;

use p3_field::Field;
use proofman_starks_lib_c::{starks_new_c, write_custom_commit_c};

use crate::{trace::Trace, ProofCtx, SetupCtx};

pub fn write_custom_commit_trace<F: Field>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    custom_trace: &mut dyn Trace<F>,
    name: &str,
    mut hash_file: [u8; 32],
    check: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = pctx.get_custom_commits_fixed_buffer(name);

    let file_name_str = match file_name {
        Some(path) => path.to_str().expect("Invalid UTF-8 in path"),
        None => {
            // Return error
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Custom Commit Fixed {:?} not found", file_name),
            )));
        }
    };

    if check {
        let mut file = File::open(file_name_str).unwrap();
        let mut hash_file_stored = [0u8; 32];
        file.read_exact(&mut hash_file_stored).unwrap();

        if hash_file_stored != hash_file {
            return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Hash does not match")));
        }
        Ok(())
    } else {
        let setup = sctx.get_setup(custom_trace.airgroup_id(), custom_trace.air_id());

        let p_starks = starks_new_c((&setup.p_setup).into(), std::ptr::null_mut());
        let buffer: Vec<F> = custom_trace.get_buffer();

        let commit_id = custom_trace.commit_id().unwrap() as u64;

        write_custom_commit_c(p_starks, commit_id, buffer.as_ptr() as *mut u8, file_name_str, hash_file.as_mut_ptr());
        Ok(())
    }
}
