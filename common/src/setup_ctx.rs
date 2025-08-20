use std::collections::HashMap;
use std::ffi::c_void;

use fields::PrimeField64;
use proofman_starks_lib_c::{expressions_bin_new_c, expressions_bin_free_c};

use crate::load_const_pols;
use crate::GlobalInfo;
use crate::Setup;
use crate::ProofType;
use crate::ParamsGPU;

pub struct SetupsVadcop<F: PrimeField64> {
    pub sctx_compressor: Option<SetupCtx<F>>,
    pub sctx_recursive1: Option<SetupCtx<F>>,
    pub sctx_recursive2: Option<SetupCtx<F>>,
    pub setup_vadcop_final: Option<Setup<F>>,
    pub setup_recursivef: Option<Setup<F>>,
    pub max_const_size: usize,
    pub max_const_tree_size: usize,
    pub max_prover_trace_size: usize,
    pub max_prover_buffer_size: usize,
    pub max_prover_recursive_buffer_size: usize,
    pub max_pinned_proof_size: usize,
    pub max_n_bits_ext: usize,
    pub total_const_size: usize,
}

unsafe impl<F: PrimeField64> Send for SetupsVadcop<F> {}
unsafe impl<F: PrimeField64> Sync for SetupsVadcop<F> {}

impl<F: PrimeField64> SetupsVadcop<F> {
    pub fn new(
        global_info: &GlobalInfo,
        verify_constraints: bool,
        aggregation: bool,
        final_snark: bool,
        gpu_params: &ParamsGPU,
    ) -> Self {
        if aggregation {
            let sctx_compressor = SetupCtx::new(global_info, &ProofType::Compressor, verify_constraints, gpu_params);
            let sctx_recursive1 = SetupCtx::new(global_info, &ProofType::Recursive1, verify_constraints, gpu_params);
            let sctx_recursive2 = SetupCtx::new(global_info, &ProofType::Recursive2, verify_constraints, gpu_params);
            let setup_vadcop_final = Setup::new(
                global_info,
                0,
                0,
                &ProofType::VadcopFinal,
                verify_constraints,
                gpu_params.preallocate,
                false,
            );
            let mut setup_recursivef = None;
            if final_snark {
                setup_recursivef =
                    Some(Setup::new(global_info, 0, 0, &ProofType::RecursiveF, verify_constraints, false, false));
            }

            let total_const_size = sctx_compressor.total_const_size
                + sctx_recursive1.total_const_size
                + sctx_recursive2.total_const_size
                + setup_vadcop_final.const_pols_size
                + setup_vadcop_final.const_tree_size;

            let vadcop_final_trace_size = setup_vadcop_final.stark_info.map_sections_n["cm1"]
                * (1 << setup_vadcop_final.stark_info.stark_struct.n_bits)
                + setup_vadcop_final.stark_info.n_publics;

            let max_const_size = sctx_compressor
                .max_const_size
                .max(sctx_recursive1.max_const_size)
                .max(sctx_recursive2.max_const_size)
                .max(setup_vadcop_final.const_pols_size);
            let max_const_tree_size = sctx_compressor
                .max_const_tree_size
                .max(sctx_recursive1.max_const_tree_size)
                .max(sctx_recursive2.max_const_tree_size)
                .max(setup_vadcop_final.const_tree_size);
            let max_prover_trace_size = sctx_compressor
                .max_prover_trace_size
                .max(sctx_recursive1.max_prover_trace_size)
                .max(sctx_recursive2.max_prover_trace_size)
                .max(vadcop_final_trace_size as usize);
            let max_prover_buffer_size = sctx_compressor
                .max_prover_buffer_size
                .max(sctx_recursive1.max_prover_buffer_size)
                .max(sctx_recursive2.max_prover_buffer_size)
                .max(setup_vadcop_final.prover_buffer_size as usize);

            let max_prover_recursive_buffer_size = (sctx_recursive1.max_prover_buffer_size
                + sctx_recursive1.max_prover_trace_size)
                .max(sctx_recursive2.max_prover_buffer_size + sctx_recursive2.max_prover_trace_size);

            let max_pinned_proof_size = sctx_compressor
                .max_pinned_proof_size
                .max(sctx_recursive1.max_pinned_proof_size)
                .max(sctx_recursive2.max_pinned_proof_size)
                .max(setup_vadcop_final.proof_size as usize);

            let max_n_bits_ext = sctx_compressor
                .max_n_bits_ext
                .max(sctx_recursive1.max_n_bits_ext)
                .max(sctx_recursive2.max_n_bits_ext)
                .max(setup_vadcop_final.stark_info.stark_struct.n_bits_ext as usize);

            SetupsVadcop {
                sctx_compressor: Some(sctx_compressor),
                sctx_recursive1: Some(sctx_recursive1),
                sctx_recursive2: Some(sctx_recursive2),
                setup_vadcop_final: Some(setup_vadcop_final),
                setup_recursivef,
                max_const_tree_size,
                max_const_size,
                max_prover_trace_size,
                max_prover_buffer_size,
                max_prover_recursive_buffer_size,
                max_pinned_proof_size,
                max_n_bits_ext,
                total_const_size,
            }
        } else {
            SetupsVadcop {
                sctx_compressor: None,
                sctx_recursive1: None,
                sctx_recursive2: None,
                setup_vadcop_final: None,
                setup_recursivef: None,
                total_const_size: 0,
                max_const_tree_size: 0,
                max_const_size: 0,
                max_prover_trace_size: 0,
                max_prover_buffer_size: 0,
                max_prover_recursive_buffer_size: 0,
                max_pinned_proof_size: 0,
                max_n_bits_ext: 0,
            }
        }
    }

    pub fn get_setup(&self, airgroup_id: usize, air_id: usize, setup_type: &ProofType) -> &Setup<F> {
        match setup_type {
            ProofType::Compressor => self.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::Recursive1 => self.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::Recursive2 => self.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, air_id),
            ProofType::VadcopFinal => self.setup_vadcop_final.as_ref().unwrap(),
            ProofType::RecursiveF => self.setup_recursivef.as_ref().unwrap(),
            _ => panic!("Invalid setup type"),
        }
    }
}

#[derive(Debug)]
pub struct SetupRepository<F: PrimeField64> {
    setups: HashMap<(usize, usize), Setup<F>>,
    max_const_tree_size: usize,
    max_const_size: usize,
    max_prover_buffer_size: usize,
    max_prover_trace_size: usize,
    max_witness_trace_size: usize,
    max_pinned_proof_size: usize,
    max_single_buffer_size: usize,
    total_const_size: usize,
    global_bin: Option<*mut c_void>,
    global_info_file: String,
    max_n_bits_ext: usize,
}

unsafe impl<F: PrimeField64> Send for SetupRepository<F> {}
unsafe impl<F: PrimeField64> Sync for SetupRepository<F> {}

impl<F: PrimeField64> Drop for SetupRepository<F> {
    fn drop(&mut self) {
        if let Some(global_bin_ptr) = self.global_bin {
            expressions_bin_free_c(global_bin_ptr);
        }
    }
}

impl<F: PrimeField64> SetupRepository<F> {
    pub fn new(
        global_info: &GlobalInfo,
        setup_type: &ProofType,
        verify_constraints: bool,
        gpu_params: &ParamsGPU,
    ) -> Self {
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
        let mut max_const_size = 0;
        let mut max_n_bits_ext = 0;
        let mut max_prover_buffer_size = 0;
        let mut max_prover_trace_size = 0;
        let mut max_witness_trace_size = 0;
        let mut max_pinned_proof_size = 0;
        let mut total_const_size = 0;
        let mut max_single_buffer_size = 0;

        // Initialize Hashmap for each airgroup_id, air_id
        if setup_type != &ProofType::VadcopFinal {
            for (airgroup_id, air_group) in global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    let single_instance = cfg!(feature = "gpu")
                        && gpu_params.max_number_streams > 1
                        && gpu_params.single_instances.contains(&(airgroup_id, air_id))
                        && setup_type == &ProofType::Basic;
                    let setup = Setup::new(
                        global_info,
                        airgroup_id,
                        air_id,
                        setup_type,
                        verify_constraints,
                        gpu_params.preallocate,
                        single_instance,
                    );
                    if setup_type != &ProofType::Compressor || global_info.get_air_has_compressor(airgroup_id, air_id) {
                        let n = 1 << setup.stark_info.stark_struct.n_bits;
                        let n_bits_ext = setup.stark_info.stark_struct.n_bits_ext;
                        let trace_size = setup.stark_info.map_sections_n["cm1"] * n;
                        let mut total_prover_trace_size = trace_size as usize;
                        total_prover_trace_size += setup.stark_info.n_publics as usize;
                        total_prover_trace_size += setup.stark_info.airvalues_map.as_ref().map_or(0, |v| 3 * v.len());
                        total_prover_trace_size +=
                            setup.stark_info.airgroupvalues_map.as_ref().map_or(0, |v| 3 * v.len());
                        total_prover_trace_size += global_info.proof_values_map.as_ref().map_or(0, |v| 3 * v.len());
                        total_prover_trace_size += 3;
                        if !single_instance {
                            if max_const_tree_size < setup.const_tree_size {
                                max_const_tree_size = setup.const_tree_size;
                            }
                            if max_const_size < setup.const_pols_size {
                                max_const_size = setup.const_pols_size;
                            }
                            if max_prover_buffer_size < setup.prover_buffer_size {
                                max_prover_buffer_size = setup.prover_buffer_size;
                            }
                            max_prover_trace_size = max_prover_trace_size.max(total_prover_trace_size);
                        } else if max_single_buffer_size < setup.prover_buffer_size {
                            max_single_buffer_size = setup.prover_buffer_size;
                        }
                        max_witness_trace_size = max_witness_trace_size.max(trace_size as usize);

                        total_const_size += setup.const_pols_size + setup.const_tree_size;
                        max_pinned_proof_size = max_pinned_proof_size.max(setup.pinned_proof_size);
                        max_n_bits_ext = max_n_bits_ext.max(n_bits_ext);
                    }
                    setups.insert((airgroup_id, air_id), setup);
                    if setup_type == &ProofType::Recursive2 {
                        break;
                    }
                }
            }
        } else {
            setups.insert(
                (0, 0),
                Setup::new(global_info, 0, 0, setup_type, verify_constraints, gpu_params.preallocate, false),
            );
        }

        Self {
            setups,
            global_bin,
            global_info_file,
            max_const_tree_size,
            max_const_size,
            max_prover_buffer_size: max_prover_buffer_size as usize,
            max_prover_trace_size,
            max_witness_trace_size,
            max_single_buffer_size: max_single_buffer_size as usize,
            max_pinned_proof_size: max_pinned_proof_size as usize,
            total_const_size,
            max_n_bits_ext: max_n_bits_ext as usize,
        }
    }
}

/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
pub struct SetupCtx<F: PrimeField64> {
    setup_repository: SetupRepository<F>,
    pub max_const_tree_size: usize,
    pub max_const_size: usize,
    pub max_prover_buffer_size: usize,
    pub max_prover_trace_size: usize,
    pub max_witness_trace_size: usize,
    pub max_pinned_proof_size: usize,
    pub max_n_bits_ext: usize,
    pub max_single_buffer_size: usize,
    pub total_const_size: usize,
    setup_type: ProofType,
}

impl<F: PrimeField64> SetupCtx<F> {
    pub fn new(
        global_info: &GlobalInfo,
        setup_type: &ProofType,
        verify_constraints: bool,
        gpu_params: &ParamsGPU,
    ) -> Self {
        let setup_repository = SetupRepository::new(global_info, setup_type, verify_constraints, gpu_params);
        let max_const_tree_size = setup_repository.max_const_tree_size;
        let max_const_size = setup_repository.max_const_size;
        let max_prover_buffer_size = setup_repository.max_prover_buffer_size;
        let max_prover_trace_size = setup_repository.max_prover_trace_size;
        let max_witness_trace_size = setup_repository.max_witness_trace_size;
        let max_pinned_proof_size = setup_repository.max_pinned_proof_size;
        let max_single_buffer_size = setup_repository.max_single_buffer_size;
        let total_const_size = setup_repository.total_const_size;
        let max_n_bits_ext = setup_repository.max_n_bits_ext;
        SetupCtx {
            setup_repository,
            max_const_tree_size,
            max_const_size,
            max_prover_buffer_size,
            max_prover_trace_size,
            max_witness_trace_size,
            max_pinned_proof_size,
            max_n_bits_ext,
            max_single_buffer_size,
            total_const_size,
            setup_type: setup_type.clone(),
        }
    }

    pub fn get_setup(&self, airgroup_id: usize, air_id: usize) -> &Setup<F> {
        match self.setup_repository.setups.get(&(airgroup_id, air_id)) {
            Some(setup) => setup,
            None => {
                // Handle the error case as needed
                tracing::error!("Setup not found for airgroup_id: {}, air_id: {}", airgroup_id, air_id);
                // You might want to return a default value or panic
                panic!("Setup not found"); // or return a default value if applicable
            }
        }
    }

    pub fn get_fixed(&self, airgroup_id: usize, air_id: usize) -> Vec<F> {
        match self.setup_repository.setups.get(&(airgroup_id, air_id)) {
            Some(setup) => {
                let const_pols: Vec<F> = vec![F::ZERO; setup.const_pols_size];
                load_const_pols(&setup.setup_path, setup.const_pols_size, &const_pols);
                const_pols
            }
            None => {
                // Handle the error case as needed
                tracing::error!("Setup not found for airgroup_id: {}, air_id: {}", airgroup_id, air_id);
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
