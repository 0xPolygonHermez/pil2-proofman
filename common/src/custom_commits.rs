use std::fs::File;
use std::io::Read;
use std::path::Path;

use fields::Field;
use proofman_starks_lib_c::write_custom_commit_c;

use crate::trace::Trace;

pub fn write_custom_commit_trace<F: Field>(
    custom_trace: &mut dyn Trace<F>,
    blowup_factor: u64,
    file_name: &Path,
    check: bool,
) -> Result<Vec<F>, Box<dyn std::error::Error>> {
    let buffer = custom_trace.get_buffer();
    let n = custom_trace.num_rows() as u64;
    let n_extended = blowup_factor * custom_trace.num_rows() as u64;
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

        println!("Root from file: {:?}", root_file);
    }

    write_custom_commit_c(
        root.as_mut_ptr() as *mut u8,
        n,
        n_extended,
        n_cols,
        buffer.as_ptr() as *mut u8,
        file_name.to_str().expect("Invalid file name"),
        check,
    );

    if check {
        for idx in 0..4 {
            if root_file[idx] != root[idx] {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Root does not match")));
            }
        }
    }
    Ok(root)
}
