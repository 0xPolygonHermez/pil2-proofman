use std::os::raw::c_void;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use proofman_starks_lib_c::{
    get_const_tree_size_c, prover_helpers_new_c, expressions_bin_new_c, stark_info_new_c, load_const_tree_c,
    load_const_pols_c, calculate_const_tree_c, stark_info_free_c, expressions_bin_free_c, prover_helpers_free_c,
    get_map_totaln_c, write_const_tree_c,
};
use proofman_util::create_buffer_fast;

use crate::GlobalInfo;
use crate::ProofType;
use crate::StarkInfo;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct SetupC {
    pub p_stark_info: *mut c_void,
    pub p_expressions_bin: *mut c_void,
    pub p_prover_helpers: *mut c_void,
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
pub struct Setup<F: Clone> {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub p_setup: SetupC,
    pub stark_info: StarkInfo,
    pub const_pols: Vec<F>,
    pub const_tree: Vec<F>,
    pub prover_buffer_size: u64,
    pub write_const_tree: AtomicBool,
    pub setup_path: PathBuf,
    pub setup_type: ProofType,
    pub air_name: String,
}

impl<F: Clone> Setup<F> {
    const MY_NAME: &'static str = "Setup";

    pub fn new(global_info: &GlobalInfo, airgroup_id: usize, air_id: usize, setup_type: &ProofType) -> Self {
        let setup_path = match setup_type {
            ProofType::VadcopFinal => global_info.get_setup_path("vadcop_final"),
            ProofType::RecursiveF => global_info.get_setup_path("recursivef"),
            _ => global_info.get_air_setup_path(airgroup_id, air_id, setup_type),
        };

        let stark_info_path = setup_path.display().to_string() + ".starkinfo.json";
        let expressions_bin_path = setup_path.display().to_string() + ".bin";

        let (
            stark_info,
            p_stark_info,
            p_expressions_bin,
            p_prover_helpers,
            prover_buffer_size,
            const_pols_size,
            const_tree_size,
        ) = if setup_type == &ProofType::Compressor && !global_info.get_air_has_compressor(airgroup_id, air_id) {
            // If the condition is met, use None for each pointer
            (StarkInfo::default(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), 0, 0, 0)
        } else {
            // Otherwise, initialize the pointers with their respective values
            let stark_info_json = std::fs::read_to_string(&stark_info_path)
                .unwrap_or_else(|_| panic!("Failed to read file {}", &stark_info_path));
            let stark_info = StarkInfo::from_json(&stark_info_json);
            let p_stark_info = stark_info_new_c(stark_info_path.as_str(), false);
            let recursive = &ProofType::Basic != setup_type;
            let prover_buffer_size = get_map_totaln_c(p_stark_info, recursive);
            let expressions_bin = expressions_bin_new_c(expressions_bin_path.as_str(), false, false);
            let prover_helpers = prover_helpers_new_c(p_stark_info, recursive);
            let const_pols_size = (stark_info.n_constants * (1 << stark_info.stark_struct.n_bits)) as usize;
            let const_pols_tree_size = get_const_tree_size_c(p_stark_info) as usize;

            (
                stark_info,
                p_stark_info,
                expressions_bin,
                prover_helpers,
                prover_buffer_size,
                const_pols_size,
                const_pols_tree_size,
            )
        };

        Self {
            air_id,
            airgroup_id,
            stark_info,
            p_setup: SetupC { p_stark_info, p_expressions_bin, p_prover_helpers },
            const_pols: create_buffer_fast(const_pols_size),
            const_tree: create_buffer_fast(const_tree_size),
            prover_buffer_size,
            write_const_tree: AtomicBool::new(false),
            setup_path: setup_path.clone(),
            setup_type: setup_type.clone(),
            air_name: global_info.airs[airgroup_id][air_id].name.clone(),
        }
    }

    pub fn free(&self) {
        stark_info_free_c(self.p_setup.p_stark_info);
        expressions_bin_free_c(self.p_setup.p_expressions_bin);
        prover_helpers_free_c(self.p_setup.p_prover_helpers);
    }

    pub fn load_const_pols(&self) {
        log::debug!(
            "{}   : ··· Loading const pols for AIR {} of type {:?}",
            Self::MY_NAME,
            self.air_name,
            self.setup_type
        );

        let const_pols_path = self.setup_path.display().to_string() + ".const";

        load_const_pols_c(
            self.const_pols.as_ptr() as *mut u8,
            const_pols_path.as_str(),
            self.const_pols.len() as u64 * 8,
        );
    }

    pub fn load_const_pols_tree(&self) {
        log::debug!(
            "{}   : ··· Loading const tree for AIR {} of type {:?}",
            Self::MY_NAME,
            self.air_name,
            self.setup_type
        );

        let const_pols_tree_path = self.setup_path.display().to_string() + ".consttree";

        let verkey_path = self.setup_path.display().to_string() + ".verkey.json";

        let p_stark_info = self.p_setup.p_stark_info;

        let valid_root = if PathBuf::from(&const_pols_tree_path).exists() {
            load_const_tree_c(
                p_stark_info,
                self.const_tree.as_ptr() as *mut u8,
                const_pols_tree_path.as_str(),
                (self.const_tree.len() * 8) as u64,
                verkey_path.as_str(),
            )
        } else {
            false
        };

        if !valid_root {
            calculate_const_tree_c(
                p_stark_info,
                self.const_pols.as_ptr() as *mut u8,
                self.const_tree.as_ptr() as *mut u8,
            );
            self.write_const_tree.store(true, Ordering::SeqCst)
        };
    }

    pub fn to_write_tree(&self) -> bool {
        self.write_const_tree.load(Ordering::SeqCst)
    }

    pub fn set_write_const_tree(&self, write: bool) {
        self.write_const_tree.store(write, Ordering::SeqCst)
    }

    pub fn write_const_tree(&self) {
        let const_pols_tree_path = self.setup_path.display().to_string() + ".consttree";

        let p_stark_info = self.p_setup.p_stark_info;

        write_const_tree_c(p_stark_info, self.const_tree.as_ptr() as *mut u8, const_pols_tree_path.as_str());
    }

    pub fn get_const_ptr(&self) -> *mut u8 {
        self.const_pols.as_ptr() as *mut u8
    }

    pub fn get_const_tree_ptr(&self) -> *mut u8 {
        self.const_tree.as_ptr() as *mut u8
    }
}
