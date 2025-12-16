use std::os::raw::{c_void, c_char};
use fields::PrimeField64;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::fs::File;
use std::fs;
use std::io::Read;
use libloading::{Library, Symbol};
use std::ffi::CString;

use proofman_starks_lib_c::set_memory_expressions_c;
use proofman_starks_lib_c::{
    expressions_bin_new_c, stark_info_new_c, stark_info_free_c, expressions_bin_free_c, get_map_totaln_c,
    get_map_totaln_custom_commits_fixed_c, get_proof_size_c, get_max_n_tmp1_c, get_max_n_tmp3_c, get_const_tree_size_c,
    load_const_pols_c, load_const_tree_c, read_exec_file_c, get_proof_pinned_size_c, get_operations_quotient_c,
};
use proofman_util::create_buffer_fast;

use crate::{GlobalInfo, ProofmanError};
use crate::ProofType;
use crate::StarkInfo;
use crate::ProofmanResult;

pub type GetSizeWitnessFunc = unsafe extern "C" fn() -> u64;

pub type GetCircomCircuitFunc = unsafe extern "C" fn(dat_file: *const c_char) -> *mut c_void;

pub type FreeCircomCircuitFunc = unsafe extern "C" fn(circuit: *mut c_void);

#[derive(Debug)]
#[repr(C)]
pub struct SetupC {
    pub p_stark_info: *mut c_void,
    pub p_expressions_bin: *mut c_void,
}

unsafe impl Send for SetupC {}
unsafe impl Sync for SetupC {}

impl From<&SetupC> for *mut c_void {
    fn from(setup: &SetupC) -> *mut c_void {
        setup as *const SetupC as *mut c_void
    }
}

impl Drop for SetupC {
    fn drop(&mut self) {
        stark_info_free_c(self.p_stark_info);
        expressions_bin_free_c(self.p_expressions_bin);
    }
}

/// Air instance context for managing air instances (traces)
#[derive(Debug)]
#[allow(dead_code)]
pub struct Setup<F: PrimeField64> {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub p_setup: SetupC,
    pub stark_info: StarkInfo,
    pub const_pols_size: usize,
    pub const_pols_size_packed: usize,
    pub const_tree_size: usize,
    pub const_pols_path: String,
    pub const_pols_tree_path: String,
    pub const_pols: Vec<F>,
    pub const_pols_tree: Vec<F>,
    pub prover_buffer_size: u64,
    pub custom_commits_fixed_buffer_size: u64,
    pub proof_size: u64,
    pub pinned_proof_size: u64,
    pub setup_path: PathBuf,
    pub setup_type: ProofType,
    pub size_witness: RwLock<Option<u64>>,
    pub circom_library: RwLock<Option<Library>>,
    pub circom_circuit: RwLock<Option<*mut c_void>>,
    pub air_name: String,
    pub verkey: Vec<F>,
    pub exec_data: RwLock<Option<Vec<u64>>>,
    pub n_cols: u64,
    pub n_operations_quotient: u64,
    pub single_instance: bool,
}

impl<F: PrimeField64> Drop for Setup<F> {
    fn drop(&mut self) {
        let mut circom_circuit_guard = self.circom_circuit.write().unwrap();

        if circom_circuit_guard.is_some() {
            let circom_circuit = circom_circuit_guard.take().unwrap();

            let circom_library_guard = self.circom_library.read().unwrap();

            if let Some(circom_library) = circom_library_guard.as_ref() {
                unsafe {
                    let free_circom_circuit: Symbol<FreeCircomCircuitFunc> =
                        circom_library.get(b"freeCircuit\0").expect("Failed to get freeCircuit symbol");

                    free_circom_circuit(circom_circuit);
                }
            }
        }
    }
}

impl<F: PrimeField64> Setup<F> {
    pub fn new(
        global_info: &GlobalInfo,
        airgroup_id: usize,
        air_id: usize,
        setup_type: &ProofType,
        verify_constraints: bool,
        preallocate: bool,
        single_instance: bool,
    ) -> Self {
        let setup_path = match setup_type {
            ProofType::VadcopFinal => global_info.get_setup_path("vadcop_final"),
            ProofType::RecursiveF => global_info.get_setup_path("recursivef"),
            _ => global_info.get_air_setup_path(airgroup_id, air_id, setup_type),
        };

        let gpu = cfg!(feature = "gpu");

        let stark_info_path = match setup_type {
            ProofType::Recursive1 => {
                let setup_path_recursive2 = global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Recursive2);
                setup_path_recursive2.display().to_string() + ".starkinfo.json"
            }
            _ => setup_path.display().to_string() + ".starkinfo.json",
        };

        let expressions_bin_path = match setup_type {
            ProofType::Recursive1 => {
                let setup_path_recursive2 = global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Recursive2);
                setup_path_recursive2.display().to_string() + ".bin"
            }
            _ => setup_path.display().to_string() + ".bin",
        };
        let const_pols_path = match !gpu {
            true => setup_path.display().to_string() + ".const",
            false => setup_path.display().to_string() + ".const_gpu",
        };
        let const_pols_tree_path = match !gpu {
            true => setup_path.display().to_string() + ".consttree",
            false => setup_path.display().to_string() + ".consttree_gpu",
        };

        let (
            stark_info,
            p_stark_info,
            p_expressions_bin,
            const_pols,
            const_pols_tree,
            verkey,
            const_pols_size,
            const_pols_size_packed,
            const_tree_size,
            prover_buffer_size,
            custom_commits_fixed_buffer_size,
            proof_size,
            pinned_proof_size,
            n_cols,
            n_operations_quotient,
        ) = if setup_type == &ProofType::Compressor && !global_info.get_air_has_compressor(airgroup_id, air_id) {
            // If the condition is met, use None for each pointer
            (
                StarkInfo::default(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        } else {
            // Otherwise, initialize the pointers with their respective values
            let stark_info_json = std::fs::read_to_string(&stark_info_path)
                .unwrap_or_else(|_| panic!("Failed to read file {}", &stark_info_path));
            let stark_info = StarkInfo::from_json(&stark_info_json);
            let recursive = setup_type != &ProofType::Basic;
            let recursive_final = setup_type == &ProofType::RecursiveF;
            let preallocate_const = preallocate && gpu;
            let p_stark_info = stark_info_new_c(
                stark_info_path.as_str(),
                recursive_final,
                recursive,
                verify_constraints,
                false,
                gpu,
                preallocate_const,
            );
            let expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, false);
            let n_max_tmp1 = get_max_n_tmp1_c(expressions_bin);
            let n_max_tmp3 = get_max_n_tmp3_c(expressions_bin);
            set_memory_expressions_c(p_stark_info, n_max_tmp1, n_max_tmp3);
            let prover_buffer_size = get_map_totaln_c(p_stark_info);
            let custom_commits_fixed_buffer_size = get_map_totaln_custom_commits_fixed_c(p_stark_info);
            let proof_size = get_proof_size_c(p_stark_info);
            let pinned_proof_size = get_proof_pinned_size_c(p_stark_info);
            let const_pols_size = (stark_info.n_constants * (1 << stark_info.stark_struct.n_bits)) as usize;

            let const_tree_size = get_const_tree_size_c(p_stark_info) as usize;

            let n_operations_quotient = get_operations_quotient_c(expressions_bin, p_stark_info) as u64;

            let verkey_file = setup_path.with_extension("verkey.json");

            let verkey = if setup_type == &ProofType::RecursiveF {
                vec![]
            } else {
                let mut file = File::open(&verkey_file).expect("Unable to open file");
                let mut json_str = String::new();
                file.read_to_string(&mut json_str).expect("Unable to read file");
                let vk: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse JSON");
                vk.iter().map(|&x| F::from_u64(x)).collect::<Vec<F>>()
            };

            let n_cols = stark_info.map_sections_n["cm1"];

            if verify_constraints {
                let const_pols: Vec<F> = create_buffer_fast(const_pols_size);
                (
                    stark_info,
                    p_stark_info,
                    expressions_bin,
                    const_pols,
                    Vec::new(),
                    verkey,
                    const_pols_size,
                    0,
                    const_tree_size,
                    prover_buffer_size,
                    custom_commits_fixed_buffer_size,
                    proof_size,
                    pinned_proof_size,
                    n_cols,
                    n_operations_quotient,
                )
            } else {
                let const_pols: Vec<F> = create_buffer_fast(const_pols_size);
                let const_pols_tree: Vec<F> = create_buffer_fast(const_tree_size);
                let mut const_pols_size_packed = 0;
                if gpu {
                    let words_per_row: u64 = if Path::new(&const_pols_path).exists() {
                        let bytes = fs::read(&const_pols_path).expect("Failed to read const_pols file");
                        if bytes.len() >= 8 {
                            u64::from_le_bytes(bytes[..8].try_into().unwrap())
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    const_pols_size_packed =
                        (words_per_row * (1 << stark_info.stark_struct.n_bits) + 1 + stark_info.n_constants) as usize;
                }
                (
                    stark_info,
                    p_stark_info,
                    expressions_bin,
                    const_pols,
                    const_pols_tree,
                    verkey,
                    const_pols_size,
                    const_pols_size_packed,
                    const_tree_size,
                    prover_buffer_size,
                    custom_commits_fixed_buffer_size,
                    proof_size,
                    pinned_proof_size,
                    n_cols,
                    n_operations_quotient,
                )
            }
        };

        Self {
            air_id,
            airgroup_id,
            stark_info,
            p_setup: SetupC { p_stark_info, p_expressions_bin },
            const_pols_size,
            const_pols_size_packed,
            const_tree_size,
            const_pols,
            const_pols_tree,
            verkey,
            prover_buffer_size,
            custom_commits_fixed_buffer_size,
            proof_size,
            pinned_proof_size,
            size_witness: RwLock::new(None),
            circom_circuit: RwLock::new(None),
            circom_library: RwLock::new(None),
            exec_data: RwLock::new(None),
            setup_path: setup_path.clone(),
            setup_type: setup_type.clone(),
            air_name: global_info.airs[airgroup_id][air_id].name.clone(),
            const_pols_path,
            const_pols_tree_path,
            n_cols,
            n_operations_quotient,
            single_instance,
        }
    }

    pub fn load_const_pols(&self) {
        load_const_pols_c(
            self.const_pols.as_ptr() as *mut u8,
            self.const_pols_path.as_str(),
            self.const_pols_size as u64 * 8,
        );
    }

    pub fn load_const_pols_tree(&self) {
        let const_pols_tree_size = self.const_tree_size;

        load_const_tree_c(
            self.p_setup.p_stark_info,
            self.const_pols_tree.as_ptr() as *mut u8,
            self.const_pols_tree_path.as_str(),
            (const_pols_tree_size * 8) as u64,
            &(self.setup_path.display().to_string() + ".verkey.json"),
        );
    }

    pub fn get_const_ptr(&self) -> *mut u8 {
        self.const_pols.as_ptr() as *mut u8
    }

    pub fn get_const_tree_ptr(&self) -> *mut u8 {
        self.const_pols_tree.as_ptr() as *mut u8
    }

    pub fn set_circom_circuit(&self) -> ProofmanResult<()> {
        let lib_extension = if cfg!(target_os = "macos") { ".dylib" } else { ".so" };
        let rust_lib_filename = self.setup_path.display().to_string() + lib_extension;
        let rust_lib_path = Path::new(rust_lib_filename.as_str());

        let dat_filename = self.setup_path.display().to_string() + ".dat";
        let dat_filename_str = CString::new(dat_filename.as_str()).unwrap();
        let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

        if !rust_lib_path.exists() {
            return Err(ProofmanError::InvalidSetup(format!(
                "Rust lib dynamic library not found at path: {rust_lib_path:?}"
            )));
        }

        let library: Library = unsafe { Library::new(rust_lib_path)? };

        let circom_circuit = unsafe {
            let init_circom_circuit: Symbol<GetCircomCircuitFunc> = library.get(b"initCircuit\0")?;
            Some(init_circom_circuit(dat_filename_ptr))
        };

        let size_witness = unsafe {
            let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
            Some(get_size_witness())
        };

        *self.circom_library.write().unwrap() = Some(library);
        *self.size_witness.write().unwrap() = size_witness;
        *self.circom_circuit.write().unwrap() = circom_circuit;
        Ok(())
    }

    pub fn set_exec_file_data(&self) -> ProofmanResult<()> {
        let exec_filename = self.setup_path.display().to_string() + ".exec";
        let exec_filename_str = CString::new(exec_filename.as_str()).unwrap();
        let exec_filename_ptr = exec_filename_str.as_ptr() as *mut std::os::raw::c_char;

        let mut file = File::open(exec_filename)?;

        let mut bytes = [0u8; 8];

        file.read_exact(&mut bytes)?;
        let n_adds = u64::from_le_bytes(bytes);

        file.read_exact(&mut bytes)?;
        let n_smap = u64::from_le_bytes(bytes);

        let exec_data_size = 2 + n_adds * 4 + n_smap * self.n_cols;
        let mut exec_file_data: Vec<u64> = vec![0; exec_data_size as usize];
        read_exec_file_c(exec_file_data.as_mut_ptr(), exec_filename_ptr, self.n_cols);
        *self.exec_data.write().unwrap() = Some(exec_file_data);
        Ok(())
    }
}
