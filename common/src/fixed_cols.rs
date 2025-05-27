use std::{
    os::raw::c_void,
    path::{PathBuf, Path},
};

use p3_field::Field;
use proofman_starks_lib_c::{
    calculate_const_tree_c, load_const_pols_c, load_const_tree_c, write_const_tree_c, write_fixed_cols_bin_c,
};
use proofman_util::{create_buffer_fast, timer_start_debug, timer_stop_and_log_debug};

use crate::Setup;

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
    name: String, // AirName.ColumnName
    lengths: Vec<u64>,
    values: Vec<F>,
}

impl<F: Field> FixedColsInfo<F> {
    pub fn new(name_: &str, lengths: Option<Vec<u64>>, values: Vec<F>) -> Self {
        FixedColsInfo { name: name_.to_string(), lengths: lengths.unwrap_or_default(), values }
    }
}

pub fn write_fixed_cols_bin<F: Field>(
    bin_file: &str,
    airgroup_name: &str,
    air_name: &str,
    n: u64,
    fixed_cols: &mut [FixedColsInfo<F>],
) {
    let mut fixed_cols_info_c = FixedColsInfoC::from_fixed_cols_info_vec(fixed_cols);
    let fixed_cols_info_c_ptr = fixed_cols_info_c.as_mut_ptr() as *mut c_void;
    write_fixed_cols_bin_c(bin_file, airgroup_name, air_name, n, fixed_cols.len() as u64, fixed_cols_info_c_ptr);
}

pub fn calculate_fixed_tree<F: Field>(setup: &Setup<F>) {
    let const_pols_size = (setup.stark_info.n_constants * (1 << setup.stark_info.stark_struct.n_bits)) as usize;
    let const_pols_tree_size = setup.const_tree_size;

    let const_pols: Vec<F> = create_buffer_fast(const_pols_size);
    let const_tree: Vec<F> = create_buffer_fast(const_pols_tree_size);

    let const_pols_path = setup.setup_path.display().to_string() + ".const";
    let const_pols_tree_path = setup.setup_path.display().to_string() + ".consttree";

    tracing::debug!("··· Loading const pols for AIR {} of type {:?}", setup.air_name, setup.setup_type);

    load_const_pols_c(const_pols.as_ptr() as *mut u8, const_pols_path.as_str(), const_pols.len() as u64 * 8);

    tracing::debug!("··· Loading const tree for AIR {} of type {:?}", setup.air_name, setup.setup_type);

    let verkey_path = setup.setup_path.display().to_string() + ".verkey.json";

    let p_stark_info = setup.p_setup.p_stark_info;

    let valid_root = if PathBuf::from(&const_pols_tree_path).exists() {
        load_const_tree_c(
            setup.p_setup.p_stark_info,
            const_tree.as_ptr() as *mut u8,
            const_pols_tree_path.as_str(),
            (const_tree.len() * 8) as u64,
            verkey_path.as_str(),
        )
    } else {
        false
    };

    if !valid_root {
        timer_start_debug!(WRITING_CONST_TREE);
        calculate_const_tree_c(p_stark_info, const_pols.as_ptr() as *mut u8, const_tree.as_ptr() as *mut u8);
        write_const_tree_c(p_stark_info, const_tree.as_ptr() as *mut u8, const_pols_tree_path.as_str());
        timer_stop_and_log_debug!(WRITING_CONST_TREE);
    }
}

pub fn load_const_pols<F: Field>(setup_path: &Path, const_pols_size: usize, const_pols: &[F]) {
    let const_pols_path = setup_path.to_string_lossy().to_string() + ".const";
    load_const_pols_c(const_pols.as_ptr() as *mut u8, const_pols_path.as_str(), const_pols_size as u64 * 8);
}

pub fn load_const_pols_tree<F: Field>(setup: &Setup<F>, const_tree: &[F]) {
    let const_pols_tree_path = setup.setup_path.display().to_string() + ".consttree";
    let const_pols_tree_size = setup.const_tree_size;

    tracing::debug!("FixedCol   : ··· Loading const tree for AIR {} of type {:?}", setup.air_name, setup.setup_type);

    load_const_tree_c(
        setup.p_setup.p_stark_info,
        const_tree.as_ptr() as *mut u8,
        const_pols_tree_path.as_str(),
        (const_pols_tree_size * 8) as u64,
        &(setup.setup_path.display().to_string() + ".verkey.json"),
    );
}
