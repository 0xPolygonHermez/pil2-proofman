use std::collections::HashMap;
use std::ffi::c_void;

use p3_field::Field;
use proofman_starks_lib_c::{expressions_bin_new_c, get_const_tree_size_c};

use crate::GlobalInfo;
use crate::Setup;
use crate::ProofType;

pub struct SetupsVadcop<F: Field> {
    pub sctx_compressor: Option<SetupCtx<F>>,
    pub sctx_recursive1: Option<SetupCtx<F>>,
    pub sctx_recursive2: Option<SetupCtx<F>>,
    pub setup_vadcop_final: Option<Setup<F>>,
    pub setup_recursivef: Option<Setup<F>>,
    pub max_const_tree_size_aggregation: usize,
}

impl<F: Field> SetupsVadcop<F> {
    pub fn new(global_info: &GlobalInfo, verify_constraints: bool, aggregation: bool, final_snark: bool) -> Self {
        let mut max_const_tree_size_aggregation = 0;
        if aggregation {
            let sctx_compressor = SetupCtx::new(global_info, &ProofType::Compressor, false);
            if sctx_compressor.max_const_tree_size > max_const_tree_size_aggregation {
                max_const_tree_size_aggregation = sctx_compressor.max_const_tree_size;
            }
            let sctx_recursive1 = SetupCtx::new(global_info, &ProofType::Recursive1, false);
            if sctx_recursive1.max_const_tree_size > max_const_tree_size_aggregation {
                max_const_tree_size_aggregation = sctx_recursive1.max_const_tree_size;
            }
            let sctx_recursive2 = SetupCtx::new(global_info, &ProofType::Recursive2, false);
            if sctx_recursive2.max_const_tree_size > max_const_tree_size_aggregation {
                max_const_tree_size_aggregation = sctx_recursive2.max_const_tree_size;
            }
            let setup_vadcop_final = Setup::new(global_info, 0, 0, &ProofType::VadcopFinal, verify_constraints);
            let const_pols_tree_size = get_const_tree_size_c(setup_vadcop_final.p_setup.p_stark_info) as usize;
            if const_pols_tree_size > max_const_tree_size_aggregation {
                max_const_tree_size_aggregation = const_pols_tree_size;
            }

            let mut setup_recursivef = None;
            if final_snark {
                setup_recursivef = Some(Setup::new(global_info, 0, 0, &ProofType::RecursiveF, verify_constraints));
                let const_pols_tree_size =
                    get_const_tree_size_c(setup_recursivef.as_ref().unwrap().p_setup.p_stark_info) as usize;
                if const_pols_tree_size > max_const_tree_size_aggregation {
                    max_const_tree_size_aggregation = const_pols_tree_size;
                }
            }

            SetupsVadcop {
                sctx_compressor: Some(sctx_compressor),
                sctx_recursive1: Some(sctx_recursive1),
                sctx_recursive2: Some(sctx_recursive2),
                setup_vadcop_final: Some(setup_vadcop_final),
                setup_recursivef,
                max_const_tree_size_aggregation,
            }
        } else {
            SetupsVadcop {
                sctx_compressor: None,
                sctx_recursive1: None,
                sctx_recursive2: None,
                setup_vadcop_final: None,
                setup_recursivef: None,
                max_const_tree_size_aggregation,
            }
        }
    }

    pub fn free(&self) {
        if self.sctx_compressor.is_some() {
            self.sctx_compressor.as_ref().unwrap().free();
        }
        if self.sctx_recursive1.is_some() {
            self.sctx_recursive1.as_ref().unwrap().free();
        }
        if self.sctx_recursive2.is_some() {
            self.sctx_recursive2.as_ref().unwrap().free();
        }
        if self.setup_vadcop_final.is_some() {
            self.setup_vadcop_final.as_ref().unwrap().free();
        }
        if self.setup_recursivef.is_some() {
            self.setup_recursivef.as_ref().unwrap().free();
        }
    }
}

#[derive(Debug)]
pub struct SetupRepository<F: Field> {
    setups: HashMap<(usize, usize), Setup<F>>,
    max_const_tree_size: usize,
    global_bin: Option<*mut c_void>,
    global_info_file: String,
}

unsafe impl<F: Field> Send for SetupRepository<F> {}
unsafe impl<F: Field> Sync for SetupRepository<F> {}

impl<F: Field> SetupRepository<F> {
    pub fn new(global_info: &GlobalInfo, setup_type: &ProofType, verify_constraints: bool) -> Self {
        let mut setups = HashMap::new();

        let global_bin = match setup_type == &ProofType::Basic {
            true => {
                let global_bin_path =
                    &global_info.get_proving_key_path().join("pilout.globalConstraints.bin").display().to_string();
                Some(expressions_bin_new_c(global_bin_path.as_str(), true, false))
            }
            false => None,
        };

        let global_info_path = &global_info.get_proving_key_path().join("pilout.globalInfo.json");
        let global_info_file = global_info_path.to_str().unwrap().to_string();

        let mut max_const_tree_size = 0;

        // Initialize Hashmap for each airgroup_id, air_id
        if setup_type != &ProofType::VadcopFinal {
            for (airgroup_id, air_group) in global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    let setup = Setup::new(global_info, airgroup_id, air_id, setup_type, verify_constraints);
                    if setup_type != &ProofType::Compressor || global_info.get_air_has_compressor(airgroup_id, air_id) {
                        let const_pols_tree_size = get_const_tree_size_c(setup.p_setup.p_stark_info) as usize;
                        if max_const_tree_size < const_pols_tree_size {
                            max_const_tree_size = const_pols_tree_size;
                        }
                    }
                    setups.insert((airgroup_id, air_id), setup);
                }
            }
        } else {
            setups.insert((0, 0), Setup::new(global_info, 0, 0, setup_type, verify_constraints));
        }

        Self { setups, global_bin, global_info_file, max_const_tree_size }
    }

    pub fn free(&self) {
        for setup in self.setups.values() {
            setup.free();
        }
    }
}
/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
pub struct SetupCtx<F: Field> {
    setup_repository: SetupRepository<F>,
    pub max_const_tree_size: usize,
    setup_type: ProofType,
}

impl<F: Field> SetupCtx<F> {
    pub fn new(global_info: &GlobalInfo, setup_type: &ProofType, verify_constraints: bool) -> Self {
        let setup_repository = SetupRepository::new(global_info, setup_type, verify_constraints);
        let max_const_tree_size = setup_repository.max_const_tree_size;
        SetupCtx { setup_repository, max_const_tree_size, setup_type: setup_type.clone() }
    }

    pub fn get_setup(&self, airgroup_id: usize, air_id: usize) -> &Setup<F> {
        match self.setup_repository.setups.get(&(airgroup_id, air_id)) {
            Some(setup) => setup,
            None => {
                // Handle the error case as needed
                log::error!("Setup not found for airgroup_id: {}, air_id: {}", airgroup_id, air_id);
                // You might want to return a default value or panic
                panic!("Setup not found"); // or return a default value if applicable
            }
        }
    }

    pub fn get_fixed_slice(&self, airgroup_id: usize, air_id: usize) -> &[F] {
        match self.setup_repository.setups.get(&(airgroup_id, air_id)) {
            Some(setup) => setup.const_pols.as_slice(),
            None => {
                // Handle the error case as needed
                log::error!("Setup not found for airgroup_id: {}, air_id: {}", airgroup_id, air_id);
                // You might want to return a default value or panic
                panic!("Setup not found"); // or return a default value if applicable
            }
        }
    }

    pub fn get_setups_list(&self) -> Vec<(usize, usize)> {
        self.setup_repository.setups.keys().cloned().collect()
    }

    pub fn get_global_bin(&self) -> *mut c_void {
        self.setup_repository.global_bin.unwrap()
    }

    pub fn get_global_info_file(&self) -> String {
        self.setup_repository.global_info_file.clone()
    }

    pub fn free(&self) {
        self.setup_repository.free();
    }
}
