use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use log::info;
use proofman_starks_lib_c::expressions_bin_new_c;
use proofman_util::{timer_start_debug, timer_stop_and_log_debug, timer_start_info, timer_stop_and_log_info};

use crate::GlobalInfo;
use crate::Setup;
use crate::ProofType;

pub struct SetupsVadcop<F: Clone> {
    pub sctx: Arc<SetupCtx<F>>,
    pub sctx_compressor: Option<SetupCtx<F>>,
    pub sctx_recursive1: Option<SetupCtx<F>>,
    pub sctx_recursive2: Option<SetupCtx<F>>,
    pub setup_vadcop_final: Option<Setup<F>>,
    pub setup_recursivef: Option<Setup<F>>,
}

impl<F: Clone> SetupsVadcop<F> {
    pub fn new(global_info: &GlobalInfo, verify_constraints: bool, aggregation: bool, final_snark: bool) -> Self {
        info!("Initializing setups");
        timer_start_info!(INITIALIZING_BASIC_SETUP);
        let sctx = SetupCtx::<F>::new(global_info, &ProofType::Basic, verify_constraints);
        timer_stop_and_log_info!(INITIALIZING_BASIC_SETUP);
        if aggregation {
            timer_start_info!(INITIALIZING_AGGREGATION_SETUP);
            info!("Initializing setups aggregation");

            timer_start_debug!(INITIALIZING_SETUP_COMPRESSOR);
            info!(" ··· Initializing setups compressor");
            let sctx_compressor = SetupCtx::<F>::new(global_info, &ProofType::Compressor, false);
            timer_stop_and_log_debug!(INITIALIZING_SETUP_COMPRESSOR);

            timer_start_debug!(INITIALIZING_SETUP_RECURSIVE1);
            info!(" ··· Initializing setups recursive1");
            let sctx_recursive1 = SetupCtx::<F>::new(global_info, &ProofType::Recursive1, false);
            timer_stop_and_log_debug!(INITIALIZING_SETUP_RECURSIVE1);

            timer_start_debug!(INITIALIZING_SETUP_RECURSIVE2);
            info!(" ··· Initializing setups recursive2");
            let sctx_recursive2 = SetupCtx::<F>::new(global_info, &ProofType::Recursive2, false);
            timer_stop_and_log_debug!(INITIALIZING_SETUP_RECURSIVE2);

            timer_start_debug!(INITIALIZING_SETUP_VADCOP_FINAL);
            info!(" ··· Initializing setups vadcop final");
            let setup_vadcop_final = Setup::<F>::new(global_info, 0, 0, &ProofType::VadcopFinal, verify_constraints);
            timer_stop_and_log_debug!(INITIALIZING_SETUP_VADCOP_FINAL);
            timer_stop_and_log_info!(INITIALIZING_AGGREGATION_SETUP);

            let mut setup_recursivef = None;
            if final_snark {
                timer_start_debug!(INITIALIZING_SETUP_RECURSION);
                timer_start_debug!(INITIALIZING_SETUP_RECURSIVEF);
                info!(" ··· Initializing setups recursivef");
                setup_recursivef = Some(Setup::<F>::new(global_info, 0, 0, &ProofType::RecursiveF, verify_constraints));
                timer_stop_and_log_debug!(INITIALIZING_SETUP_RECURSIVEF);
                timer_stop_and_log_debug!(INITIALIZING_SETUP_RECURSION);
            }

            SetupsVadcop {
                sctx: Arc::new(sctx),
                sctx_compressor: Some(sctx_compressor),
                sctx_recursive1: Some(sctx_recursive1),
                sctx_recursive2: Some(sctx_recursive2),
                setup_vadcop_final: Some(setup_vadcop_final),
                setup_recursivef,
            }
        } else {
            SetupsVadcop {
                sctx: Arc::new(sctx),
                sctx_compressor: None,
                sctx_recursive1: None,
                sctx_recursive2: None,
                setup_vadcop_final: None,
                setup_recursivef: None,
            }
        }
    }
}

#[derive(Debug)]
pub struct SetupRepository<F: Clone> {
    setups: HashMap<(usize, usize), Setup<F>>,
    global_bin: Option<*mut c_void>,
    global_info_file: String,
}

unsafe impl<F: Clone> Send for SetupRepository<F> {}
unsafe impl<F: Clone> Sync for SetupRepository<F> {}

impl<F: Clone> SetupRepository<F> {
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

        // Initialize Hashmap for each airgroup_id, air_id
        if setup_type != &ProofType::VadcopFinal {
            for (airgroup_id, air_group) in global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    setups.insert(
                        (airgroup_id, air_id),
                        Setup::new(global_info, airgroup_id, air_id, setup_type, verify_constraints),
                    );
                }
            }
        } else {
            setups.insert((0, 0), Setup::new(global_info, 0, 0, setup_type, verify_constraints));
        }

        Self { setups, global_bin, global_info_file }
    }

    pub fn free(&self) {
        // TODO
    }
}
/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
pub struct SetupCtx<F: Clone> {
    setup_repository: SetupRepository<F>,
    setup_type: ProofType,
}

impl<F: Clone> SetupCtx<F> {
    pub fn new(global_info: &GlobalInfo, setup_type: &ProofType, verify_constraints: bool) -> Self {
        SetupCtx {
            setup_repository: SetupRepository::<F>::new(global_info, setup_type, verify_constraints),
            setup_type: setup_type.clone(),
        }
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
}
