use std::fs::File;
use std::io::Read;
use std::path::Path;

use fields::PrimeField64;
use proofman_starks_lib_c::{write_custom_commit_c, init_gpu_setup_c, get_num_gpus_c};

use crate::trace::Trace;
use crate::{ProofmanResult, ProofmanError};

pub fn write_custom_commit_trace<F: PrimeField64>(
    custom_trace: &mut dyn Trace<F>,
    blowup_factor: u64,
    merkle_tree_arity: u64,
    file_name: &Path,
    check: bool,
) -> ProofmanResult<Vec<F>> {
    let buffer = custom_trace.get_buffer();
    let arity = merkle_tree_arity;
    let n = custom_trace.num_rows() as u64;
    let n_extended = blowup_factor * custom_trace.num_rows() as u64;
    let n_bits = n.trailing_zeros() as u64;
    let n_bits_ext = n_extended.trailing_zeros() as u64;
    let n_cols = custom_trace.n_cols() as u64;
    let mut root = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];

    let mut root_file = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    if check {
        let mut file = File::open(file_name).unwrap();
        let mut root_bytes = [0u8; 32];
        file.read_exact(&mut root_bytes).unwrap();

        for (idx, val) in root_file.iter_mut().enumerate().take(4) {
            let byte_range = idx * 8..(idx + 1) * 8;
            let value = u64::from_le_bytes(root_bytes[byte_range].try_into()?);
            *val = F::from_u64(value);
        }

        println!("Root from file: {root_file:?}");
    }

    if cfg!(feature = "gpu") {
        let n_gpus = get_num_gpus_c();
        if n_gpus == 0 {
            return Err(ProofmanError::InvalidConfiguration("No GPUs found".into()));
        }

        init_gpu_setup_c(n_bits_ext);
    }

    write_custom_commit_c(
        root.as_mut_ptr() as *mut u8,
        arity,
        n_bits,
        n_bits_ext,
        n_cols,
        buffer.as_ptr() as *mut u8,
        file_name.to_str().expect("Invalid file name"),
        check,
    );

    if check {
        for idx in 0..4 {
            if root_file[idx] != root[idx] {
                return Err(ProofmanError::ProofmanError("Root does not match".into()));
            }
        }
    }
    Ok(root)
}
