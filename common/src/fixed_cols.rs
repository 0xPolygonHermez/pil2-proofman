use std::{os::raw::c_void, path::PathBuf};

use fields::PrimeField64;
use proofman_starks_lib_c::{
    calculate_const_tree_c, calculate_const_tree_bn128_c, load_const_pols_c, load_const_tree_c, write_const_tree_c,
    write_const_tree_bn128_c, write_fixed_cols_bin_c, prepare_blocks_c, pack_const_pols_c,
};
use proofman_util::{create_buffer_fast, timer_start_info, timer_stop_and_log_info};

use crate::Setup;

#[repr(C)]
#[derive(Debug)]
pub struct FixedColsInfoC<F: PrimeField64> {
    name_size: u64,
    name: *mut u8,
    n_lengths: u64,
    lengths: *mut u64,
    values: *mut F,
}

impl<F: PrimeField64> FixedColsInfoC<F> {
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
pub struct FixedColsInfo<F: PrimeField64> {
    name: String, // AirName.ColumnName
    lengths: Vec<u64>,
    values: Vec<F>,
}

impl<F: PrimeField64> FixedColsInfo<F> {
    pub fn new(name_: &str, lengths: Option<Vec<u64>>, values: Vec<F>) -> Self {
        FixedColsInfo { name: name_.to_string(), lengths: lengths.unwrap_or_default(), values }
    }
}

pub fn write_fixed_cols_bin<F: PrimeField64>(
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

pub fn calculate_fixed_tree<F: PrimeField64>(setup: &Setup<F>) {
    let const_pols_size = (setup.stark_info.n_constants * (1 << setup.stark_info.stark_struct.n_bits)) as usize;
    let const_pols_tree_size = setup.const_tree_size;

    let const_pols: Vec<F> = create_buffer_fast(const_pols_size);
    let const_tree: Vec<F> = create_buffer_fast(const_pols_tree_size);

    let const_pols_path = setup.setup_path.display().to_string() + ".const";
    let const_pols_tree_path = &setup.const_pols_tree_path.clone();

    tracing::info!("··· Loading const pols for AIR {} of type {:?}", setup.air_name, setup.setup_type);

    load_const_pols_c(const_pols.as_ptr() as *mut u8, const_pols_path.as_str(), const_pols.len() as u64 * 8);

    tracing::debug!("··· Loading const tree for AIR {} of type {:?}", setup.air_name, setup.setup_type);

    let verkey_path = setup.setup_path.display().to_string() + ".verkey.json";

    let p_stark_info = setup.p_setup.p_stark_info;

    let valid_root = if PathBuf::from(&const_pols_tree_path).exists() {
        let const_pols_tree_size = setup.const_tree_size;
        let valid_file = match std::fs::metadata(const_pols_tree_path) {
            Ok(metadata) => {
                let actual_size = metadata.len() as usize;
                actual_size == const_pols_tree_size * 8
            }
            Err(_) => false,
        };

        if valid_file {
            load_const_tree_c(
                setup.p_setup.p_stark_info,
                const_tree.as_ptr() as *mut u8,
                const_pols_tree_path.as_str(),
                (const_tree.len() * 8) as u64,
                verkey_path.as_str(),
            )
        } else {
            false
        }
    } else {
        false
    };

    if cfg!(feature = "gpu") {
        pack_const_pols_c(p_stark_info, const_pols.as_ptr() as *mut u8, setup.const_pols_path.as_str());
    }

    if !valid_root {
        timer_start_info!(WRITING_CONST_TREE);
        if setup.stark_info.stark_struct.verification_hash_type == "GL" {
            if cfg!(feature = "gpu") {
                let mut const_pols_transposed = const_pols.clone();
                prepare_blocks_c(
                    const_pols_transposed.as_mut_ptr() as *mut u64,
                    1 << setup.stark_info.stark_struct.n_bits,
                    setup.stark_info.n_constants,
                );
                calculate_const_tree_c(
                    p_stark_info,
                    const_pols_transposed.as_ptr() as *mut u8,
                    const_tree.as_ptr() as *mut u8,
                );
                write_const_tree_c(p_stark_info, const_tree.as_ptr() as *mut u8, const_pols_tree_path.as_str());
            } else {
                calculate_const_tree_c(p_stark_info, const_pols.as_ptr() as *mut u8, const_tree.as_ptr() as *mut u8);
                write_const_tree_c(p_stark_info, const_tree.as_ptr() as *mut u8, const_pols_tree_path.as_str());
            }
        } else {
            calculate_const_tree_bn128_c(p_stark_info, const_pols.as_ptr() as *mut u8, const_tree.as_ptr() as *mut u8);
            write_const_tree_bn128_c(p_stark_info, const_tree.as_ptr() as *mut u8, const_pols_tree_path.as_str());
        }
        timer_stop_and_log_info!(WRITING_CONST_TREE);
    }
}

pub fn load_const_pols<F: PrimeField64>(setup: &Setup<F>, const_pols: &[F]) {
    let const_pols_path = setup.setup_path.display().to_string() + ".const";
    let const_pols_size = setup.const_pols_size;
    load_const_pols_c(const_pols.as_ptr() as *mut u8, const_pols_path.as_str(), const_pols_size as u64 * 8);
}

pub fn load_const_pols_tree<F: PrimeField64>(setup: &Setup<F>, const_tree: &[F]) {
    let const_pols_tree_path = &setup.const_pols_tree_path;
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
