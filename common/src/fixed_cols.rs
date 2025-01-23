use std::os::raw::c_void;

use p3_field::Field;
use proofman_starks_lib_c::write_fixed_cols_bin_c;




#[repr(C)]
#[derive(Debug)]
pub struct FixedColsInfoC<F: Field> {
    name_size: u64,
    name: *mut u8,
    n_lengths: u64,
    lengths: *mut u64,
    values: *mut F,
}

impl<F: Field> FixedColsInfoC<F> {
    pub fn from_fixed_cols_info_vec(fixed_cols: &mut [FixedColsInfo<F>]) -> Vec<FixedColsInfoC<F>> {
        fixed_cols
            .iter_mut()
            .map(|info| FixedColsInfoC {
                name_size: info.name.len() as u64,
                name: info.name.as_mut_ptr(),
                n_lengths: info.lengths.len() as u64,
                lengths: info.lengths.as_mut_ptr(),
                values: info.values.as_mut_ptr(),
            })
            .collect()
    }
}
#[derive(Clone, Debug)]
#[repr(C)]
pub struct FixedColsInfo<F: Field> {
    name: String,
    lengths: Vec<u64>,
    values: Vec<F>,
}

impl<F: Field> FixedColsInfo<F> {
    pub fn new(name: String, lengths: Option<Vec<u64>>, values: Vec<F>) -> Self {         
        FixedColsInfo {
            name,
            lengths: lengths.unwrap_or_else(Vec::new),
            values,
        }
    }
}

pub fn write_fixed_cols_bin<F:Field>(bin_file: String, n: u64, fixed_cols: &mut Vec<FixedColsInfo<F>>) {
    let mut fixed_cols_info_c = FixedColsInfoC::<F>::from_fixed_cols_info_vec(fixed_cols);
    let fixed_cols_info_c_ptr = fixed_cols_info_c.as_mut_ptr() as *mut c_void;
    write_fixed_cols_bin_c(&bin_file, n, fixed_cols.len() as u64, fixed_cols_info_c_ptr);
}