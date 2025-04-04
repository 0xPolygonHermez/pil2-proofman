use std::os::raw::c_void;
use p3_field::Field;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use libloading::{Library, Symbol};

use proofman_starks_lib_c::set_memory_expressions_c;
use proofman_starks_lib_c::{
    expressions_bin_new_c, stark_info_new_c, stark_info_free_c, expressions_bin_free_c, get_map_totaln_c,
    get_map_totaln_custom_commits_fixed_c, get_proof_size_c, get_max_n_tmp1_c, get_max_n_tmp3_c, get_const_tree_size_c,
    load_const_pols_c, load_const_tree_c,
};
use proofman_util::create_buffer_fast;

use crate::GlobalInfo;
use crate::ProofType;
use crate::StarkInfo;
use crate::load_const_pols;

type GetSizeWitnessFunc = unsafe extern "C" fn() -> u64;

#[derive(Debug, Clone)]
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

/// Air instance context for managing air instances (traces)
#[derive(Debug)]
#[allow(dead_code)]
pub struct Setup<F: Field> {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub p_setup: SetupC,
    pub stark_info: StarkInfo,
    pub const_pols_size: usize,
    pub const_tree_size: usize,
    pub const_pols: Vec<F>,
    pub const_pols_tree: Vec<F>,
    pub prover_buffer_size: u64,
    pub custom_commits_fixed_buffer_size: u64,
    pub proof_size: u64,
    pub setup_path: PathBuf,
    pub setup_type: ProofType,
    pub size_witness: RwLock<Option<u64>>,
    pub air_name: String,
}

impl<F: Field> Setup<F> {
    pub fn new(
        global_info: &GlobalInfo,
        airgroup_id: usize,
        air_id: usize,
        setup_type: &ProofType,
        verify_constraints: bool,
    ) -> Self {
        let setup_path = match setup_type {
            ProofType::VadcopFinal => global_info.get_setup_path("vadcop_final"),
            ProofType::RecursiveF => global_info.get_setup_path("recursivef"),
            _ => global_info.get_air_setup_path(airgroup_id, air_id, setup_type),
        };

        let stark_info_path = setup_path.display().to_string() + ".starkinfo.json";
        let expressions_bin_path = setup_path.display().to_string() + ".bin";

        let gpu = cfg!(feature = "gpu");

        let (
            stark_info,
            p_stark_info,
            p_expressions_bin,
            const_pols,
            const_pols_tree,
            const_pols_size,
            const_tree_size,
            prover_buffer_size,
            custom_commits_fixed_buffer_size,
            proof_size,
        ) = if setup_type == &ProofType::Compressor && !global_info.get_air_has_compressor(airgroup_id, air_id) {
            // If the condition is met, use None for each pointer
            (StarkInfo::default(), std::ptr::null_mut(), std::ptr::null_mut(), Vec::new(), Vec::new(), 0, 0, 0, 0, 0)
        } else {
            // Otherwise, initialize the pointers with their respective values
            let stark_info_json = std::fs::read_to_string(&stark_info_path)
                .unwrap_or_else(|_| panic!("Failed to read file {}", &stark_info_path));
            let stark_info = StarkInfo::from_json(&stark_info_json);
            let recursive = setup_type != &ProofType::Basic;
            let p_stark_info = stark_info_new_c(stark_info_path.as_str(), recursive, verify_constraints, false, gpu);
            let expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, false);
            let n_max_tmp1 = get_max_n_tmp1_c(expressions_bin);
            let n_max_tmp3 = get_max_n_tmp3_c(expressions_bin);
            set_memory_expressions_c(p_stark_info, n_max_tmp1, n_max_tmp3);
            let prover_buffer_size = get_map_totaln_c(p_stark_info);
            let custom_commits_fixed_buffer_size = get_map_totaln_custom_commits_fixed_c(p_stark_info);
            let proof_size = get_proof_size_c(p_stark_info);

            let const_pols_size = (stark_info.n_constants * (1 << stark_info.stark_struct.n_bits)) as usize;

            let const_tree_size = get_const_tree_size_c(p_stark_info) as usize;

            if verify_constraints {
                let const_pols: Vec<F> = create_buffer_fast(const_pols_size);
                load_const_pols(&setup_path, const_pols_size, &const_pols);
                (
                    stark_info,
                    p_stark_info,
                    expressions_bin,
                    const_pols,
                    Vec::new(),
                    const_pols_size,
                    const_tree_size,
                    prover_buffer_size,
                    custom_commits_fixed_buffer_size,
                    proof_size,
                )
            } else {
                let const_pols: Vec<F> = create_buffer_fast(const_pols_size);
                let const_pols_tree: Vec<F> = create_buffer_fast(const_tree_size);
                (
                    stark_info,
                    p_stark_info,
                    expressions_bin,
                    const_pols,
                    const_pols_tree,
                    const_pols_size,
                    const_tree_size,
                    prover_buffer_size,
                    custom_commits_fixed_buffer_size,
                    proof_size,
                )
            }
        };

        Self {
            air_id,
            airgroup_id,
            stark_info,
            p_setup: SetupC { p_stark_info, p_expressions_bin },
            const_pols_size,
            const_tree_size,
            const_pols,
            const_pols_tree,
            prover_buffer_size,
            custom_commits_fixed_buffer_size,
            proof_size,
            size_witness: RwLock::new(None),
            setup_path: setup_path.clone(),
            setup_type: setup_type.clone(),
            air_name: global_info.airs[airgroup_id][air_id].name.clone(),
        }
    }

    pub fn free(&self) {
        stark_info_free_c(self.p_setup.p_stark_info);
        expressions_bin_free_c(self.p_setup.p_expressions_bin);
    }

    pub fn load_const_pols(&self) {
        let const_pols_path = self.setup_path.to_string_lossy().to_string() + ".const";
        load_const_pols_c(
            self.const_pols.as_ptr() as *mut u8,
            const_pols_path.as_str(),
            self.const_pols_size as u64 * 8,
        );
    }

    pub fn load_const_pols_tree(&self) {
        let const_pols_tree_path = self.setup_path.display().to_string() + ".consttree";
        let const_pols_tree_size = self.const_tree_size;

        load_const_tree_c(
            self.p_setup.p_stark_info,
            self.const_pols_tree.as_ptr() as *mut u8,
            const_pols_tree_path.as_str(),
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

    pub fn set_size_witness(&self) -> Result<(), Box<dyn std::error::Error>> {
        let rust_lib_filename = self.setup_path.display().to_string() + ".so";
        let rust_lib_path = Path::new(rust_lib_filename.as_str());

        if !rust_lib_path.exists() {
            return Err(format!("Rust lib dynamic library not found at path: {:?}", rust_lib_path).into());
        }

        let library: Library = unsafe { Library::new(rust_lib_path)? };

        let size_witness = unsafe {
            let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
            Some(get_size_witness())
        };

        *self.size_witness.write().unwrap() = size_witness;
        Ok(())
    }
}
