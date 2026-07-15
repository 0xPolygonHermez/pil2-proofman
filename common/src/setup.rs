use std::os::raw::{c_void, c_char};
use fields::PrimeField64;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::fs;
use std::io::Read;
use libloading::{Library, Symbol};
use std::ffi::CString;
use std::sync::{Arc, RwLock};

pub type GetWitnessFunc =
    unsafe extern "C" fn(zkin: *mut u64, circom_circuit: *mut c_void, witness: *mut c_void, n_mutexes: u64) -> i64;

#[derive(Debug)]
pub struct CircomState {
    library: Option<Library>,
    pub circuit: Option<*mut c_void>,
    pub get_witness_fn: Option<GetWitnessFunc>,
}

unsafe impl Send for CircomState {}
unsafe impl Sync for CircomState {}

use proofman_starks_lib_c::set_memory_expressions_c;
use proofman_starks_lib_c::{
    expressions_bin_new_c, stark_info_new_c, stark_info_free_c, expressions_bin_free_c, get_map_totaln_c,
    get_map_totaln_custom_commits_fixed_c, get_map_totaln_contributions_c, get_proof_size_c, get_max_n_tmp1_c,
    get_max_n_tmp3_c, get_const_tree_size_c, get_proof_pinned_size_c, get_operations_quotient_c,
    calculate_words_per_row_c,
};

use crate::{GlobalInfoAir, ProofmanError};
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
    pub prover_buffer_size: u64,
    pub contributions_size: u64,
    pub custom_commits_fixed_buffer_size: u64,
    pub proof_size: u64,
    pub pinned_proof_size: u64,
    pub setup_path: PathBuf,
    pub setup_type: ProofType,
    pub size_witness: Option<u64>,
    pub circom_state: RwLock<CircomState>,
    pub exec_data_path: Option<String>,
    pub exec_data: Option<Arc<Vec<u64>>>,
    pub n_adds: Option<u64>,
    pub air_name: String,
    pub verkey: Vec<F>,
    pub verkey_file: String,
    pub n_cols: u64,
    pub n_operations_quotient: u64,
    pub preallocate: bool,
    pub gpu: bool,
}

impl<F: PrimeField64> Drop for Setup<F> {
    fn drop(&mut self) {
        let mut state = self.circom_state.write().unwrap();
        if let Some(circom_circuit) = state.circuit.take() {
            if let Some(circom_library) = &state.library {
                unsafe {
                    let free_circom_circuit: Symbol<FreeCircomCircuitFunc> =
                        circom_library.get(b"freeCircuit\0").expect("Failed to get freeCircuit symbol");
                    free_circom_circuit(circom_circuit);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl<F: PrimeField64> Setup<F> {
    pub fn new(
        setup_path: &Path,
        airgroup_id: usize,
        air_id: usize,
        air_info: &GlobalInfoAir,
        setup_type: &ProofType,
        verify_constraints: bool,
        preallocate: bool,
        gpu: bool,
        starkinfo_source_path: Option<&PathBuf>,
    ) -> ProofmanResult<Self> {
        let starkinfo_borrow_path = match setup_type {
            ProofType::Recursive1 => Some(
                starkinfo_source_path
                    .expect("starkinfo_source_path (Recursive2 stem) must be provided for Recursive1")
                    .clone(),
            ),
            ProofType::RecurserAggregator => Some(
                starkinfo_source_path
                    .expect("starkinfo_source_path (vadcop_final stem) must be provided for RecurserAggregator")
                    .clone(),
            ),
            _ => None,
        };
        let stark_info_path = match &starkinfo_borrow_path {
            Some(p) => p.display().to_string() + ".starkinfo.json",
            None => setup_path.display().to_string() + ".starkinfo.json",
        };

        let expressions_bin_path = match &starkinfo_borrow_path {
            Some(p) => p.display().to_string() + ".bin",
            None => setup_path.display().to_string() + ".bin",
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
            verkey,
            verkey_file,
            const_pols_size,
            const_pols_size_packed,
            const_tree_size,
            prover_buffer_size,
            contributions_size,
            custom_commits_fixed_buffer_size,
            proof_size,
            pinned_proof_size,
            n_cols,
            n_operations_quotient,
        ) = if setup_type == &ProofType::Compressor && !air_info.has_compressor.unwrap_or(false) {
            // If the condition is met, use None for each pointer
            (
                StarkInfo::default(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                Vec::new(),
                String::new(),
                0,
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
                .unwrap_or_else(|_| panic!("Failed to read file {}", stark_info_path));
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
            let contributions_size = get_map_totaln_contributions_c(p_stark_info);
            let custom_commits_fixed_buffer_size = get_map_totaln_custom_commits_fixed_c(p_stark_info);
            let proof_size = get_proof_size_c(p_stark_info);
            let pinned_proof_size = get_proof_pinned_size_c(p_stark_info);
            let const_pols_size = (stark_info.n_constants * (1 << stark_info.stark_struct.n_bits)) as usize;

            let const_tree_size = get_const_tree_size_c(p_stark_info) as usize;

            let n_operations_quotient = get_operations_quotient_c(expressions_bin, p_stark_info) as u64;

            let verkey_file = setup_path.display().to_string() + ".verkey.json";

            let verkey = if setup_type == &ProofType::RecursiveF {
                vec![]
            } else {
                let mut file = File::open(&verkey_file).unwrap_or_else(|e| {
                    panic!("Unable to open verkey file {verkey_file} (setup_type {setup_type:?}): {e}")
                });
                let mut json_str = String::new();
                file.read_to_string(&mut json_str).expect("Unable to read file");
                let vk: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse JSON");
                vk.iter().map(|&x| F::from_u64(x)).collect::<Vec<F>>()
            };

            let n_cols = stark_info.map_sections_n["cm1"];

            if verify_constraints && !gpu {
                (
                    stark_info,
                    p_stark_info,
                    expressions_bin,
                    verkey,
                    verkey_file,
                    const_pols_size,
                    0,
                    const_tree_size,
                    prover_buffer_size,
                    contributions_size,
                    custom_commits_fixed_buffer_size,
                    proof_size,
                    pinned_proof_size,
                    n_cols,
                    n_operations_quotient,
                )
            } else {
                let mut const_pols_size_packed = 0;
                if gpu && setup_type != &ProofType::RecursiveF {
                    let words_per_row: u64 = if Path::new(&const_pols_path).exists() {
                        let bytes = fs::read(&const_pols_path).expect("Failed to read const_pols file");
                        if bytes.len() >= 8 {
                            u64::from_le_bytes(bytes[..8].try_into().unwrap())
                        } else {
                            0
                        }
                    } else {
                        calculate_words_per_row_c(p_stark_info, &(setup_path.display().to_string() + ".const"))
                    };
                    const_pols_size_packed =
                        (words_per_row * (1 << stark_info.stark_struct.n_bits) + 1 + stark_info.n_constants) as usize;
                }
                (
                    stark_info,
                    p_stark_info,
                    expressions_bin,
                    verkey,
                    verkey_file,
                    const_pols_size,
                    const_pols_size_packed,
                    const_tree_size,
                    prover_buffer_size,
                    contributions_size,
                    custom_commits_fixed_buffer_size,
                    proof_size,
                    pinned_proof_size,
                    n_cols,
                    n_operations_quotient,
                )
            }
        };

        // Initialize circom circuit and exec data for proof types that need it
        // Skip compressors that don't exist
        let needs_circom = match setup_type {
            ProofType::Compressor => air_info.has_compressor.unwrap_or(false),
            _ => setup_type != &ProofType::Basic,
        };

        let (circom_library, circom_circuit, get_witness_fn, size_witness, exec_data_path, exec_data, n_adds) =
            if needs_circom {
                let lib_extension = if cfg!(target_os = "macos") { ".dylib" } else { ".so" };
                let rust_lib_filename = setup_path.display().to_string() + lib_extension;
                let rust_lib_path = Path::new(rust_lib_filename.as_str());

                if !rust_lib_path.exists() {
                    return Err(ProofmanError::InvalidSetup(format!(
                        "Rust lib dynamic library not found at path: {rust_lib_path:?}"
                    )));
                }

                let library: Library = unsafe { Library::new(rust_lib_path)? };

                let dat_filename = setup_path.display().to_string() + ".dat";
                let dat_filename_str = CString::new(dat_filename.as_str()).unwrap();
                let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

                let circom_circuit_ptr = unsafe {
                    let init_circom_circuit: Symbol<GetCircomCircuitFunc> = library.get(b"initCircuit\0")?;
                    init_circom_circuit(dat_filename_ptr)
                };

                let witness_size = unsafe {
                    let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
                    get_size_witness()
                };

                // Load the getWitness function pointer for later use
                let get_witness_fn = unsafe {
                    let get_witness_symbol: Symbol<GetWitnessFunc> = library.get(b"getWitness\0")?;
                    Some(*get_witness_symbol)
                };

                // Pre-load the entire .exec file into memory. Header is the first two u64s
                // (n_adds, n_smap); the rest is read sequentially. Pre-loading gives RAM-speed
                // access during every `get_committed_pols_c` call — no on-demand page faults.
                let exec_filename = setup_path.display().to_string() + ".exec";
                let mut file = File::open(&exec_filename)?;
                let mut bytes = [0u8; 8];
                file.read_exact(&mut bytes)?;
                let n_adds = u64::from_le_bytes(bytes);
                file.read_exact(&mut bytes)?;
                let n_smap = u64::from_le_bytes(bytes);
                let exec_data_size_elements: usize = (|| -> Option<usize> {
                    let adds_terms = n_adds.checked_mul(4)?;
                    let smap_terms = n_smap.checked_mul(n_cols)?;
                    let elements = 2u64.checked_add(adds_terms)?.checked_add(smap_terms)?;
                    usize::try_from(elements).ok()
                })()
                .ok_or_else(|| {
                    ProofmanError::InvalidSetup(format!(
                        "exec header for {exec_filename}: size overflow (n_adds={n_adds}, n_smap={n_smap}, n_cols={n_cols})"
                    ))
                })?;
                // Header already consumed; read the remaining (size - 2) u64s.
                let mut exec_data: Vec<u64> = vec![0; exec_data_size_elements];
                exec_data[0] = n_adds;
                exec_data[1] = n_smap;
                let body_bytes = (exec_data_size_elements - 2) * 8;
                let body_slice =
                    unsafe { std::slice::from_raw_parts_mut(exec_data[2..].as_mut_ptr() as *mut u8, body_bytes) };
                file.read_exact(body_slice)?;

                (
                    Some(library),
                    Some(circom_circuit_ptr),
                    get_witness_fn,
                    Some(witness_size),
                    Some(exec_filename),
                    Some(Arc::new(exec_data)),
                    Some(n_adds),
                )
            } else {
                (None, None, None, None, None, None, None)
            };

        Ok(Self {
            air_id,
            airgroup_id,
            stark_info,
            p_setup: SetupC { p_stark_info, p_expressions_bin },
            const_pols_size,
            const_pols_size_packed,
            const_tree_size,
            verkey,
            verkey_file,
            prover_buffer_size,
            custom_commits_fixed_buffer_size,
            contributions_size,
            proof_size,
            pinned_proof_size,
            size_witness,
            circom_state: RwLock::new(CircomState { library: circom_library, circuit: circom_circuit, get_witness_fn }),
            exec_data_path,
            exec_data,
            n_adds,
            setup_path: setup_path.to_path_buf().clone(),
            setup_type: setup_type.clone(),
            air_name: air_info.name.clone(),
            const_pols_path,
            const_pols_tree_path,
            n_cols,
            n_operations_quotient,
            preallocate,
            gpu,
        })
    }

    pub fn get_vk(&self) -> Vec<u64> {
        self.verkey.iter().map(|x| x.as_canonical_u64()).collect()
    }

    pub fn get_circom_witness_size(&self) -> usize {
        let base_size = self.size_witness.unwrap_or(0) as usize;
        let exec_offset = self.n_adds.unwrap_or(0) as usize;
        base_size + exec_offset
    }
}
