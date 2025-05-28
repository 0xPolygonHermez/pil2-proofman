use curves::{EcGFp5, EcMasFp5, curve::EllipticCurve, goldilocks_quintic_extension::GoldilocksQuinticExtension};
use libloading::{Library, Symbol};
use p3_field::extension::BinomialExtensionField;
use p3_field::BasedVectorSpace;
use std::ops::Add;
use proofman_common::CurveType;
use proofman_common::{
    calculate_fixed_tree, skip_prover_instance, Proof, ProofCtx, ProofType, ProofOptions, SetupCtx, SetupsVadcop,
    ParamsGPU, DebugInfo, VerboseMode,
};
use dashmap::DashMap;
use colored::Colorize;
use proofman_hints::aggregate_airgroupvals;
use proofman_starks_lib_c::{gen_device_buffers_c, free_device_buffers_c};
use proofman_starks_lib_c::{
    save_challenges_c, save_proof_values_c, save_publics_c, check_device_memory_c, gen_device_streams_c,
    get_stream_proofs_c, get_stream_proofs_non_blocking_c, register_proof_done_callback_c,
};
use rayon::prelude::*;
use crossbeam_channel::{bounded, unbounded};
use std::fs;
use std::collections::HashMap;
use std::fs::File;
use std::fmt::Write;
use std::io::Read;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, RwLock};
use p3_goldilocks::Goldilocks;

use rand::{SeedableRng, seq::SliceRandom};
use rand::rngs::StdRng;

use p3_field::PrimeField64;
use proofman_starks_lib_c::{
    gen_proof_c, commit_witness_c, calculate_hash_c, load_custom_commit_c, calculate_impols_expressions_c,
    clear_proof_done_callback_c, launch_callback_c,
};

use std::{path::PathBuf, sync::Arc};

use transcript::FFITranscript;

use witness::{WitnessLibInitFn, WitnessLibrary, WitnessManager};
use crate::{check_tree_paths_vadcop, initialize_fixed_pols_tree, contributions_done_listener};
use crate::{verify_basic_proof, verify_proof, verify_global_constraints_proof};
use crate::MaxSizes;
use crate::{verify_constraints_proof, print_summary_info, get_recursive_buffer_sizes};
use crate::{
    gen_witness_recursive, gen_witness_aggregation, generate_recursive_proof, generate_vadcop_final_proof,
    generate_fflonk_snark_proof, generate_recursivef_proof, initialize_witness_circom,
};
use crate::total_recursive_proofs;
use crate::check_tree_paths;
use crate::Counter;
use crate::aggregate_recursive2_proofs;

use std::ffi::c_void;

use proofman_util::{create_buffer_fast, timer_start_info, timer_stop_and_log_info, DeviceBuffer};

pub struct ProofMan<F: PrimeField64> {
    pctx: Arc<ProofCtx<F>>,
    sctx: Arc<SetupCtx<F>>,
    setups: Arc<SetupsVadcop<F>>,
    d_buffers: Arc<DeviceBuffer>,
    trace_size: usize,
    prover_buffer_size: usize,
    wcm: Arc<WitnessManager<F>>,
    gpu_params: ParamsGPU,
    verify_constraints: bool,
    aggregation: bool,
    final_snark: bool,
    n_streams_per_gpu: u64,
    n_gpus: u64,
}

impl<F: PrimeField64> Drop for ProofMan<F> {
    fn drop(&mut self) {
        free_device_buffers_c(self.d_buffers.get_ptr());
    }
}
impl<F: PrimeField64> ProofMan<F>
where
    BinomialExtensionField<Goldilocks, 5>: BasedVectorSpace<F>,
{
    pub fn check_setup(
        proving_key_path: PathBuf,
        aggregation: bool,
        final_snark: bool,
        gpu_params: ParamsGPU,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {:?}", proving_key_path).into());
        }

        let pctx = ProofCtx::<F>::create_ctx(proving_key_path.clone(), HashMap::new(), aggregation, final_snark);

        let setups_aggregation =
            Arc::new(SetupsVadcop::<F>::new(&pctx.global_info, false, aggregation, false, final_snark));

        let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false, gpu_params.preallocate);

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                calculate_fixed_tree(sctx.get_setup(airgroup_id, air_id));
            }
        }

        if aggregation {
            let sctx_compressor = setups_aggregation.sctx_compressor.as_ref().unwrap();
            for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                        calculate_fixed_tree(sctx_compressor.get_setup(airgroup_id, air_id));
                    }
                }
            }

            let sctx_recursive1 = setups_aggregation.sctx_recursive1.as_ref().unwrap();
            for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    calculate_fixed_tree(sctx_recursive1.get_setup(airgroup_id, air_id));
                }
            }

            let sctx_recursive2 = setups_aggregation.sctx_recursive2.as_ref().unwrap();
            let n_airgroups = pctx.global_info.air_groups.len();
            for airgroup in 0..n_airgroups {
                calculate_fixed_tree(sctx_recursive2.get_setup(airgroup, 0));
            }

            let setup_vadcop_final = setups_aggregation.setup_vadcop_final.as_ref().unwrap();
            calculate_fixed_tree(setup_vadcop_final);

            if final_snark {
                let setup_recursivef = setups_aggregation.setup_recursivef.as_ref().unwrap();
                calculate_fixed_tree(setup_recursivef);
            }
        }

        Ok(())
    }

    /// Computes only the witness without generating a proof neither verifying constraints.
    /// This is useful for debugging or benchmarking purposes.
    pub fn compute_witness(
        &self,
        input_data_path: Option<PathBuf>,
        debug_info: &DebugInfo,
    ) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(EXECUTE);
        self.pctx.set_debug_info(debug_info.clone());
        self.wcm.set_input_data_path(input_data_path);

        if !self.wcm.is_init_witness() {
            println!("Witness computation dynamic library not initialized");
            return Err("Witness computation dynamic library not initialized".into());
        }

        self.pctx.dctx_reset();

        self.wcm.execute();

        self.pctx.dctx_assign_instances();
        self.pctx.dctx_close();

        print_summary_info(&self.pctx, &self.sctx);

        timer_stop_and_log_info!(EXECUTE);

        self.pctx.set_global_challenge(2, &[F::ZERO; 3]);

        let instances = self.pctx.dctx_get_instances();

        let max_num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            let is_my_instance = self.pctx.dctx_is_my_instance(instance_id);
            let (skip, _) = skip_prover_instance(&self.pctx, instance_id);

            if skip || (!all && !is_my_instance) {
                continue;
            };

            self.wcm.calculate_witness(1, &[instance_id], 0, max_num_threads);
        }

        Ok(())
    }

    pub fn verify_proof_constraints(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        output_dir_path: PathBuf,
        debug_info: &DebugInfo,
        verbose_mode: VerboseMode,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check witness_lib path exists
        if !witness_lib_path.exists() {
            return Err(format!("Witness computation dynamic library not found at path: {:?}", witness_lib_path).into());
        }

        // Check input data path
        if let Some(ref input_data_path) = input_data_path {
            if !input_data_path.exists() {
                return Err(format!("Input data file not found at path: {:?}", input_data_path).into());
            }
        }

        // Check public_inputs_path is a folder
        if let Some(ref publics_path) = public_inputs_path {
            if !publics_path.exists() {
                return Err(format!("Public inputs file not found at path: {:?}", publics_path).into());
            }
        }

        if !output_dir_path.exists() {
            fs::create_dir_all(&output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {:?}", err))?;
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode)?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);

        self.register_witness(&mut *witness_lib, library);

        self._verify_proof_constraints(debug_info)
    }

    pub fn verify_proof_constraints_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
        debug_info: &DebugInfo,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.wcm.set_input_data_path(input_data_path);

        self._verify_proof_constraints(debug_info)
    }

    fn _verify_proof_constraints(&self, debug_info: &DebugInfo) -> Result<(), Box<dyn std::error::Error>> {
        self.pctx.dctx_reset();

        self.pctx.set_debug_info(debug_info.clone());

        if !self.wcm.is_init_witness() {
            return Err("Witness computation dynamic library not initialized".into());
        }

        timer_start_info!(EXECUTE);
        self.wcm.execute();
        timer_stop_and_log_info!(EXECUTE);

        self.pctx.dctx_assign_instances();
        self.pctx.dctx_close();

        print_summary_info(&self.pctx, &self.sctx);

        let transcript = FFITranscript::new(2, true);
        let dummy_element = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let global_challenge = [F::ZERO; 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);
        self.pctx.set_global_challenge(2, &global_challenge);
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let instances = self.pctx.dctx_get_instances();
        let my_instances = self.pctx.dctx_get_my_instances();
        let airgroup_values_air_instances = Arc::new(Mutex::new(vec![Vec::new(); my_instances.len()]));
        let valid_constraints = Arc::new(AtomicBool::new(true));
        let mut thread_handle: Option<std::thread::JoinHandle<()>> = None;

        for (instance_id, (airgroup_id, air_id, all)) in instances.iter().enumerate() {
            let is_my_instance = self.pctx.dctx_is_my_instance(instance_id);
            let (skip, _) = skip_prover_instance(&self.pctx, instance_id);

            if skip || (!all && !is_my_instance) {
                continue;
            };

            let max_num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

            self.wcm.calculate_witness(1, &[instance_id], 0, max_num_threads);

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            thread_handle = is_my_instance.then(|| {
                Self::verify_proof_constraints_stage(
                    self.pctx.clone(),
                    self.sctx.clone(),
                    self.wcm.clone(),
                    valid_constraints.clone(),
                    airgroup_values_air_instances.clone(),
                    instance_id,
                    *airgroup_id,
                    *air_id,
                    self.pctx.dctx_find_air_instance_id(instance_id),
                    debug_info,
                )
            });
        }

        if let Some(handle) = thread_handle {
            handle.join().unwrap()
        }

        self.wcm.end(debug_info);

        let check_global_constraints =
            debug_info.debug_instances.is_empty() || !debug_info.debug_global_instances.is_empty();

        if check_global_constraints {
            let airgroup_values_air_instances =
                Arc::try_unwrap(airgroup_values_air_instances).unwrap().into_inner().unwrap();
            let airgroupvalues_u64 = aggregate_airgroupvals(&self.pctx, &airgroup_values_air_instances);
            let airgroupvalues = self.pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

            if self.pctx.dctx_get_rank() == 0 {
                let valid_global_constraints =
                    verify_global_constraints_proof(&self.pctx, &self.sctx, debug_info, airgroupvalues);

                if valid_constraints.load(Ordering::Relaxed) && valid_global_constraints.is_ok() {
                    return Ok(());
                } else {
                    return Err("Constraints were not verified".into());
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_proof_constraints_stage(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        wcm: Arc<WitnessManager<F>>,
        valid_constraints: Arc<AtomicBool>,
        airgroup_values_air_instances: Arc<Mutex<Vec<Vec<F>>>>,
        instance_id: usize,
        airgroup_id: usize,
        air_id: usize,
        air_instance_id: usize,
        debug_info: &DebugInfo,
    ) -> std::thread::JoinHandle<()> {
        let debug_info = debug_info.clone();
        std::thread::spawn(move || {
            Self::initialize_air_instance(&pctx, &sctx, instance_id, true);

            #[cfg(feature = "diagnostic")]
            {
                let invalid_initialization = Self::diagnostic_instance(&pctx, &sctx, instance_id);
                if invalid_initialization {
                    panic!("Invalid initialization");
                    // return Some(Err("Invalid initialization".into()));
                }
            }

            let max_num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            wcm.calculate_witness(2, &[instance_id], 0, max_num_threads);
            Self::calculate_im_pols(2, &sctx, &pctx, instance_id);

            wcm.debug(&[instance_id], &debug_info);

            let valid = verify_constraints_proof(&pctx, &sctx, instance_id);
            if !valid {
                valid_constraints.fetch_and(valid, Ordering::Relaxed);
            }

            let airgroup_values = pctx.get_air_instance_airgroup_values(airgroup_id, air_id, air_instance_id);
            airgroup_values_air_instances.lock().unwrap()[pctx.dctx_get_instance_idx(instance_id)] = airgroup_values;
            pctx.free_instance(instance_id);
        })
    }

    pub fn generate_proof(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        verbose_mode: VerboseMode,
        options: ProofOptions,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        // Check witness_lib path exists
        if !witness_lib_path.exists() {
            return Err(format!("Witness computation dynamic library not found at path: {:?}", witness_lib_path).into());
        }

        // Check input data path
        if let Some(ref input_data_path) = input_data_path {
            if !input_data_path.exists() {
                return Err(format!("Input data file not found at path: {:?}", input_data_path).into());
            }
        }

        // Check public_inputs_path is a folder
        if let Some(ref publics_path) = public_inputs_path {
            if !publics_path.exists() {
                return Err(format!("Public inputs file not found at path: {:?}", publics_path).into());
            }
        }

        if !options.output_dir_path.exists() {
            fs::create_dir_all(&options.output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {:?}", err))?;
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode)?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);

        self.register_witness(&mut *witness_lib, library);

        self._generate_proof(options)
    }

    pub fn generate_proof_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
        options: ProofOptions,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if !options.output_dir_path.exists() {
            fs::create_dir_all(&options.output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {:?}", err))?;
        }

        self.wcm.set_input_data_path(input_data_path);
        self._generate_proof(options)
    }

    pub fn new(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        verify_constraints: bool,
        aggregation: bool,
        final_snark: bool,
        gpu_params: ParamsGPU,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        timer_start_info!(INIT_PROOFMAN);

        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {:?}", proving_key_path).into());
        }

        // Check proving_key_path is a folder
        if !proving_key_path.is_dir() {
            return Err(format!("Proving key parameter must be a folder: {:?}", proving_key_path).into());
        }

        let (pctx, sctx, setups_vadcop) = Self::initialize_proofman(
            proving_key_path,
            custom_commits_fixed,
            verify_constraints,
            aggregation,
            final_snark,
            &gpu_params,
        )?;

        let (d_buffers, n_streams_per_gpu, n_gpus) =
            Self::prepare_gpu(pctx.clone(), sctx.clone(), setups_vadcop.clone(), aggregation, &gpu_params);

        let (trace_size, prover_buffer_size) =
            if aggregation { get_recursive_buffer_sizes(&pctx, &setups_vadcop)? } else { (0, 0) };

        if !verify_constraints {
            initialize_fixed_pols_tree(
                &pctx,
                &sctx,
                &setups_vadcop,
                d_buffers.clone(),
                aggregation,
                final_snark,
                &gpu_params,
            );
        }

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone()));

        timer_stop_and_log_info!(INIT_PROOFMAN);

        Ok(Self {
            pctx,
            sctx,
            wcm,
            setups: setups_vadcop,
            d_buffers,
            trace_size,
            prover_buffer_size,
            gpu_params,
            aggregation,
            final_snark,
            verify_constraints,
            n_streams_per_gpu,
            n_gpus,
        })
    }

    pub fn register_witness(&self, witness_lib: &mut dyn WitnessLibrary<F>, library: Library) {
        timer_start_info!(REGISTERING_WITNESS);
        witness_lib.register_witness(self.wcm.clone());
        self.wcm.set_init_witness(true, library);
        timer_stop_and_log_info!(REGISTERING_WITNESS);
    }

    #[allow(clippy::too_many_arguments)]
    fn _generate_proof(&self, options: ProofOptions) -> Result<Option<String>, Box<dyn std::error::Error>> {
        timer_start_info!(GENERATING_VADCOP_PROOF);

        timer_start_info!(GENERATING_PROOFS);

        timer_start_info!(EXECUTE);

        if !self.wcm.is_init_witness() {
            println!("Witness computation dynamic library not initialized");
            return Err("Witness computation dynamic library not initialized".into());
        }

        if self.verify_constraints {
            return Err("Proofman has been initialized in verify_constraints mode".into());
        }

        if options.aggregation && !self.aggregation {
            return Err("Proofman has not been initialized in aggregation mode".into());
        }

        if options.final_snark && !self.final_snark {
            return Err("Proofman has not been initialized in final snark mode".into());
        }

        self.pctx.dctx_reset();

        self.wcm.execute();

        self.pctx.dctx_assign_instances();
        self.pctx.dctx_close();

        print_summary_info(&self.pctx, &self.sctx);

        timer_stop_and_log_info!(EXECUTE);

        timer_start_info!(CALCULATING_CONTRIBUTIONS);
        let mut rng = StdRng::seed_from_u64(self.pctx.dctx_get_rank() as u64);

        let instances = self.pctx.dctx_get_instances();
        let my_instances = self.pctx.dctx_get_my_instances();
        let mut my_instances_sorted = my_instances.clone();
        my_instances_sorted.shuffle(&mut rng);
        let instances_mine = my_instances.len();

        let values_contributions: Arc<DashMap<usize, Vec<F>>> = Arc::new(DashMap::new());
        let roots_contributions: Arc<DashMap<usize, [F; 4]>> = Arc::new(DashMap::new());
        for instance_id in my_instances.iter() {
            roots_contributions.insert(*instance_id, [F::ZERO; 4]);
        }

        let aux_trace_size = match cfg!(feature = "gpu") {
            true => 0,
            false => self.sctx.max_prover_buffer_size,
        };
        let const_pols_size = match cfg!(feature = "gpu") {
            true => 0,
            false => self.sctx.max_const_size,
        };
        let const_tree_size = match cfg!(feature = "gpu") {
            true => 0,
            false => self.sctx.max_const_tree_size,
        };
        let aux_trace: Arc<Vec<F>> = Arc::new(create_buffer_fast(aux_trace_size));
        let const_pols: Arc<Vec<F>> = Arc::new(create_buffer_fast(const_pols_size));
        let const_tree: Arc<Vec<F>> = Arc::new(create_buffer_fast(const_tree_size));

        let max_witness_stored = instances_mine.min(self.gpu_params.max_witness_stored);

        let max_num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let (threads_per_pool, max_concurrent_pools) = match cfg!(feature = "gpu") {
            true => {
                let max_concurrent_pools = self
                    .gpu_params
                    .max_number_witness_pools
                    .min(max_num_threads / self.gpu_params.number_threads_pools_witness);
                (self.gpu_params.number_threads_pools_witness, max_concurrent_pools)
            }
            false => (max_num_threads, 1),
        };

        let contributions_pending = Arc::new(Counter::new());

        let contributions_listener = contributions_done_listener(contributions_pending.clone());

        let n_proof_threads = match cfg!(feature = "gpu") {
            true => self.n_gpus,
            false => 1,
        };

        let n_streams = self.n_streams_per_gpu * n_proof_threads;
        let streams = Arc::new(Mutex::new(vec![None; n_streams as usize]));

        let (tx, rx) = bounded::<usize>(max_concurrent_pools);

        for pool_id in 0..max_concurrent_pools {
            tx.send(pool_id).unwrap();
        }

        let mut handles = Vec::new();
        for (count, &instance_id) in my_instances_sorted.iter().enumerate() {
            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let values_contributions_clone = values_contributions.clone();
            let roots_contributions_clone = roots_contributions.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let aux_trace_clone = aux_trace.clone();
            let streams_clone = streams.clone();
            let tx_clone = tx.clone();
            let instances = instances.clone();
            let wcm = self.wcm.clone();
            let contributions_pending = contributions_pending.clone();

            let pool_id = rx.recv().unwrap();

            let handle = std::thread::spawn(move || {
                let (_, _, all) = instances[instance_id];
                if !all {
                    contributions_pending.increment();
                    wcm.calculate_witness(1, &[instance_id], pool_id * threads_per_pool, threads_per_pool);
                    Self::get_contribution_air(
                        &pctx_clone,
                        &sctx_clone,
                        roots_contributions_clone.clone(),
                        values_contributions_clone.clone(),
                        instance_id,
                        aux_trace_clone.clone().as_ptr() as *mut u8,
                        d_buffers_clone.clone(),
                        streams_clone.clone(),
                    );

                    if (instances_mine - count) > max_witness_stored {
                        pctx_clone.free_instance_traces(instance_id);
                    }
                }
                tx_clone.send(pool_id).unwrap();
            });
            handles.push(handle);
        }

        // Join all threads
        for handle in handles {
            handle.join().unwrap();
        }

        let mut handles = Vec::new();
        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            let is_all = *all;
            if !is_all {
                continue;
            };
            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let values_contributions_clone = values_contributions.clone();
            let roots_contributions_clone = roots_contributions.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let aux_trace_clone = aux_trace.clone();
            let streams_clone = streams.clone();
            let tx_clone = tx.clone();
            let wcm = self.wcm.clone();
            let contributions_pending = contributions_pending.clone();

            let pool_id = rx.recv().unwrap();
            let handle = std::thread::spawn(move || {
                wcm.calculate_witness(1, &[instance_id], pool_id * threads_per_pool, threads_per_pool);
                contributions_pending.increment();
                Self::get_contribution_air(
                    &pctx_clone,
                    &sctx_clone,
                    roots_contributions_clone.clone(),
                    values_contributions_clone.clone(),
                    instance_id,
                    aux_trace_clone.clone().as_ptr() as *mut u8,
                    d_buffers_clone.clone(),
                    streams_clone.clone(),
                );
                tx_clone.send(pool_id).unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        contributions_pending
            .wait_until_zero_and_check_streams(|| get_stream_proofs_non_blocking_c(self.d_buffers.get_ptr()));

        get_stream_proofs_c(self.d_buffers.get_ptr());

        clear_proof_done_callback_c();
        contributions_listener.join().unwrap();

        Self::calculate_global_challenge(&self.pctx, roots_contributions, values_contributions);

        timer_stop_and_log_info!(CALCULATING_CONTRIBUTIONS);

        timer_start_info!(GENERATING_BASIC_PROOFS);

        let n_airgroups = self.pctx.global_info.air_groups.len();

        let proofs: Arc<DashMap<usize, Proof<F>>> = Arc::new(DashMap::new());
        let compressor_proofs: Arc<DashMap<usize, Proof<F>>> = Arc::new(DashMap::new());
        let recursive1_proofs: Arc<DashMap<usize, Proof<F>>> = Arc::new(DashMap::new());
        let recursive2_proofs: Arc<DashMap<usize, Vec<Proof<F>>>> = Arc::new(DashMap::new());
        let recursive2_proofs_ongoing: Arc<RwLock<Vec<Option<Proof<F>>>>> = Arc::new(RwLock::new(Vec::new()));

        let vec_streams: Vec<Option<u64>> = {
            let mut guard = streams.lock().unwrap();
            std::mem::take(&mut *guard)
        };

        let mut n_airgroup_proofs = vec![0; n_airgroups];
        for instance_id in my_instances.iter() {
            let (airgroup_id, air_id, _) = instances[*instance_id];
            n_airgroup_proofs[airgroup_id] += 1;
            let setup = self.sctx.get_setup(airgroup_id, air_id);
            let proof = create_buffer_fast(setup.proof_size as usize);

            proofs.insert(*instance_id, Proof::new(ProofType::Basic, airgroup_id, air_id, Some(*instance_id), proof));
        }

        if options.aggregation {
            for (airgroup, &n_proofs) in n_airgroup_proofs.iter().enumerate().take(n_airgroups) {
                let n_recursive2_proofs = total_recursive_proofs(n_proofs);
                if n_recursive2_proofs.has_remaining {
                    let setup = self.setups.get_setup(airgroup, 0, &ProofType::Recursive2);
                    let publics_aggregation = 1 + 4 * self.pctx.global_info.agg_types[airgroup].len() + 10;
                    let null_proof_buffer = vec![0; setup.proof_size as usize + publics_aggregation];
                    let null_proof = Proof::new(ProofType::Recursive2, airgroup, 0, None, null_proof_buffer);
                    let mut recursive2_proofs_airgroup = recursive2_proofs.entry(airgroup).or_default();
                    recursive2_proofs_airgroup.push(null_proof);
                }
            }
        }

        let proofs_pending = Arc::new(Counter::new());

        let (recursive_tx, recursive_rx) = unbounded::<(u64, String)>();
        register_proof_done_callback_c(recursive_tx.clone());

        let (rec_proof_tx, rec_proof_rx) = bounded::<usize>(n_proof_threads as usize);

        for pool_id in 0..n_proof_threads {
            rec_proof_tx.send(pool_id as usize).unwrap();
        }

        let pctx_clone = self.pctx.clone();
        let setups_clone = self.setups.clone();
        let proofs_clone = proofs.clone();
        let recursive2_proofs_clone = recursive2_proofs.clone();
        let d_buffers_clone = self.d_buffers.clone();
        let trace_size = self.trace_size;
        let prover_buffer_size = self.prover_buffer_size;
        let output_dir_path_clone = options.output_dir_path.clone();
        let proofs_pending_clone = proofs_pending.clone();
        let instances_clone = instances.clone();
        let const_pols_clone = const_pols.clone();
        let const_tree_clone = const_tree.clone();
        let rec_proof_tx_clone = rec_proof_tx.clone();
        let _ = std::thread::spawn(move || {
            while let Ok((id, proof_type)) = recursive_rx.recv() {
                if !options.aggregation {
                    proofs_pending_clone.decrement();
                    continue;
                }
                let rec_proof_tx_clone = rec_proof_tx_clone.clone();
                let rec_proof_rx_clone = rec_proof_rx.clone();
                let pctx_clone = pctx_clone.clone();
                let setups_clone = setups_clone.clone();
                let proofs_clone = proofs_clone.clone();
                let compressor_proofs_clone = compressor_proofs.clone();

                let recursive1_proofs_clone = recursive1_proofs.clone();
                let recursive2_proofs_clone = recursive2_proofs_clone.clone();
                let recursive2_proofs_ongoing_clone = recursive2_proofs_ongoing.clone();

                let instances_clone = instances_clone.clone();

                let pool_id = rec_proof_rx_clone.recv().unwrap();

                let output_dir_path_clone = output_dir_path_clone.clone();
                let d_buffers_clone = d_buffers_clone.clone();
                let proofs_pending_clone = proofs_pending_clone.clone();

                let const_pols_clone = const_pols_clone.clone();
                let const_tree_clone = const_tree_clone.clone();

                let recursive_handle = std::thread::spawn(move || {
                    let p: ProofType = proof_type.parse().unwrap();
                    let new_proof_type = if p == ProofType::Basic {
                        let (airgroup_id, air_id, _) = instances_clone[id as usize];
                        if pctx_clone.global_info.get_air_has_compressor(airgroup_id, air_id) {
                            ProofType::Compressor as usize
                        } else {
                            ProofType::Recursive1 as usize
                        }
                    } else if p == ProofType::Compressor {
                        ProofType::Recursive1 as usize
                    } else {
                        ProofType::Recursive2 as usize
                    };

                    let witness = if new_proof_type == ProofType::Recursive2 as usize {
                        let proof = if p == ProofType::Recursive1 {
                            recursive1_proofs_clone.get(&(id as usize)).unwrap().clone()
                        } else {
                            recursive2_proofs_ongoing_clone.read().unwrap()[id as usize].as_ref().unwrap().clone()
                        };

                        let mut recursive2_proofs_airgroup =
                            recursive2_proofs_clone.entry(proof.airgroup_id).or_default();
                        recursive2_proofs_airgroup.push(proof);

                        if recursive2_proofs_airgroup.len() >= 3 {
                            let p1 = recursive2_proofs_airgroup.pop().unwrap();
                            let p2 = recursive2_proofs_airgroup.pop().unwrap();
                            let p3 = recursive2_proofs_airgroup.pop().unwrap();

                            Some(gen_witness_aggregation(&pctx_clone, &setups_clone, &p1, &p2, &p3, 1).unwrap())
                        } else {
                            None
                        }
                    } else if new_proof_type == ProofType::Recursive1 as usize {
                        let proof = if p == ProofType::Compressor {
                            compressor_proofs_clone.get(&(id as usize)).unwrap()
                        } else {
                            proofs_clone.get(&(id as usize)).unwrap()
                        };
                        Some(gen_witness_recursive(&pctx_clone, &setups_clone, &proof, 1).unwrap())
                    } else {
                        let proof = proofs_clone.get(&(id as usize)).expect("missing proof");
                        Some(gen_witness_recursive(&pctx_clone, &setups_clone, &proof, 1).unwrap())
                    };

                    if let Some(mut witness) = witness {
                        let trace: Vec<F> = create_buffer_fast(trace_size);
                        let prover_buffer: Vec<F> = create_buffer_fast(prover_buffer_size);
                        if new_proof_type == ProofType::Recursive2 as usize {
                            let id = {
                                let mut proofs = recursive2_proofs_ongoing_clone.write().unwrap();
                                let id = proofs.len();
                                proofs.push(None);
                                id
                            };

                            witness.global_idx = Some(id);
                        }

                        proofs_pending_clone.increment();
                        let (_, proof) = generate_recursive_proof(
                            &pctx_clone,
                            &setups_clone,
                            &witness,
                            &trace,
                            &prover_buffer,
                            &output_dir_path_clone,
                            d_buffers_clone.get_ptr(),
                            const_tree_clone.clone(),
                            const_pols_clone.clone(),
                            options.save_proofs,
                        );

                        let new_proof_type_str: &str = proof.proof_type.clone().into();

                        let id = proof.global_idx.unwrap();
                        if new_proof_type == ProofType::Recursive2 as usize {
                            recursive2_proofs_ongoing_clone.write().unwrap()[id] = Some(proof);
                        } else if new_proof_type == ProofType::Compressor as usize {
                            compressor_proofs_clone.insert(id, proof);
                        } else if new_proof_type == ProofType::Recursive1 as usize {
                            recursive1_proofs_clone.insert(id, proof);
                        }

                        if cfg!(not(feature = "gpu")) {
                            launch_callback_c(id as u64, new_proof_type_str);
                        }
                    }
                    proofs_pending_clone.decrement();
                    rec_proof_tx_clone.send(pool_id).unwrap();
                });

                if cfg!(not(feature = "gpu")) {
                    recursive_handle.join().unwrap();
                }
            }
        });

        let processed_ids = Mutex::new(Vec::new());

        if cfg!(feature = "gpu") {
            vec_streams
                .par_iter()
                .enumerate()
                .filter_map(|(stream_id, instance)| instance.map(|id| (stream_id, id)))
                .for_each(|(stream_id, instance_id)| {
                    proofs_pending.increment();

                    Self::gen_proof(
                        proofs.clone(),
                        self.pctx.clone(),
                        self.sctx.clone(),
                        instance_id as usize,
                        options.output_dir_path.clone(),
                        aux_trace.clone(),
                        const_pols.clone(),
                        const_tree.clone(),
                        self.d_buffers.clone(),
                        Some(stream_id),
                        options.save_proofs,
                    );
                    processed_ids.lock().unwrap().push(instance_id);
                });
        }

        let mut my_instances_calculated = vec![false; instances.len()];
        for idx in processed_ids.into_inner().unwrap() {
            my_instances_calculated[idx as usize] = true;
        }

        let mut handles = Vec::new();
        for &instance_id in my_instances_sorted.iter() {
            if my_instances_calculated[instance_id] {
                continue;
            }
            let pool_id = rx.recv().unwrap();
            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let aux_trace_clone = aux_trace.clone();
            let tx_clone = tx.clone();
            let wcm = self.wcm.clone();
            let proofs_pending_clone = proofs_pending.clone();
            let stream_clone = vec_streams.clone();
            let proofs_clone = proofs.clone();
            let output_dir_path_clone = options.output_dir_path.clone();
            let is_stored =
                self.pctx.is_air_instance_stored(instance_id) || vec_streams.contains(&Some(instance_id as u64));

            let const_pols_clone = const_pols.clone();
            let const_tree_clone = const_tree.clone();
            my_instances_calculated[instance_id] = true;

            let handle = std::thread::spawn(move || {
                proofs_pending_clone.increment();
                if !is_stored {
                    wcm.calculate_witness(1, &[instance_id], pool_id * threads_per_pool, threads_per_pool);
                }
                let stream_id = stream_clone.iter().position(|&stream| stream == Some(instance_id as u64));
                Self::gen_proof(
                    proofs_clone.clone(),
                    pctx_clone.clone(),
                    sctx_clone.clone(),
                    instance_id,
                    output_dir_path_clone.clone(),
                    aux_trace_clone.clone(),
                    const_pols_clone.clone(),
                    const_tree_clone.clone(),
                    d_buffers_clone.clone(),
                    stream_id,
                    options.save_proofs,
                );
                pctx_clone.free_instance(instance_id);
                tx_clone.send(pool_id).unwrap();
            });
            if cfg!(feature = "gpu") {
                handles.push(handle);
            } else {
                handle.join().unwrap();
            }
        }

        for &instance_id in my_instances_sorted.iter() {
            if cfg!(not(feature = "gpu")) {
                launch_callback_c(instance_id as u64, "basic");
            }
        }

        // Join all threads
        for handle in handles {
            handle.join().unwrap();
        }

        proofs_pending.wait_until_zero_and_check_streams(|| get_stream_proofs_non_blocking_c(self.d_buffers.get_ptr()));

        get_stream_proofs_c(self.d_buffers.get_ptr());

        clear_proof_done_callback_c();

        timer_stop_and_log_info!(GENERATING_BASIC_PROOFS);

        timer_stop_and_log_info!(GENERATING_PROOFS);

        let global_info_path = self.pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
        let global_info_file = global_info_path.to_str().unwrap();

        save_challenges_c(
            self.pctx.get_challenges_ptr(),
            global_info_file,
            options.output_dir_path.to_string_lossy().as_ref(),
        );
        save_proof_values_c(
            self.pctx.get_proof_values_ptr(),
            global_info_file,
            options.output_dir_path.to_string_lossy().as_ref(),
        );
        save_publics_c(
            self.pctx.global_info.n_publics as u64,
            self.pctx.get_publics_ptr(),
            options.output_dir_path.to_string_lossy().as_ref(),
        );

        if !options.aggregation {
            let mut valid_proofs = true;

            if options.verify_proofs {
                timer_start_info!(VERIFYING_PROOFS);
                for instance_id in my_instances.iter() {
                    let valid_proof =
                        verify_basic_proof(&self.pctx, *instance_id, &proofs.get(instance_id).unwrap().proof);
                    if !valid_proof {
                        valid_proofs = false;
                    }
                }
                timer_stop_and_log_info!(VERIFYING_PROOFS);

                let mut airgroup_values_air_instances = vec![Vec::new(); my_instances.len()];

                for instance_id in my_instances.iter() {
                    let (airgroup_id, air_id, _) = instances[*instance_id];
                    let setup = self.sctx.get_setup(airgroup_id, air_id);
                    let n_airgroup_values = setup
                        .stark_info
                        .airgroupvalues_map
                        .as_ref()
                        .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
                        .unwrap_or(0);

                    let proof = proofs.get(instance_id).expect("Missing proof");

                    let airgroup_values: Vec<F> =
                        proof.proof[0..n_airgroup_values].to_vec().iter().map(|&x| F::from_u64(x)).collect();

                    airgroup_values_air_instances[self.pctx.dctx_get_instance_idx(*instance_id)] = airgroup_values;
                }

                let airgroupvalues_u64 = aggregate_airgroupvals(&self.pctx, &airgroup_values_air_instances);
                let airgroupvalues = self.pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

                if self.pctx.dctx_get_rank() == 0 {
                    let valid_global_constraints =
                        verify_global_constraints_proof(&self.pctx, &self.sctx, &DebugInfo::default(), airgroupvalues);
                    if valid_global_constraints.is_err() {
                        valid_proofs = false;
                    }
                }

                if valid_proofs {
                    tracing::info!("··· {}", "\u{2713} All proofs were successfully verified".bright_green().bold());
                    return Ok(None);
                } else {
                    return Err("Basic proofs were not verified".into());
                }
            } else {
                tracing::info!(
                    "··· {}",
                    "\u{2713} All proofs were successfully generated. Verification Skipped".bright_yellow().bold()
                );
                return Ok(None);
            }
        }

        timer_start_info!(GENERATING_OUTER_COMPRESSED_PROOFS);
        let trace = create_buffer_fast::<F>(self.trace_size);
        let prover_buffer = create_buffer_fast::<F>(self.prover_buffer_size);
        let recursive2_proofs_data: Vec<Vec<Proof<F>>> =
            recursive2_proofs.iter().map(|entry| entry.value().clone()).collect();

        let agg_recursive2_proof = aggregate_recursive2_proofs(
            &self.pctx,
            &self.setups,
            &recursive2_proofs_data,
            &trace,
            &prover_buffer,
            const_pols.clone(),
            const_tree.clone(),
            options.output_dir_path.clone(),
            self.d_buffers.get_ptr(),
            false,
        )?;
        self.pctx.dctx.read().unwrap().barrier();
        timer_stop_and_log_info!(GENERATING_OUTER_COMPRESSED_PROOFS);

        let mut proof_id = None;
        let mut vadcop_final_proof = Vec::new();
        if self.pctx.dctx_get_rank() == 0 {
            let vadcop_proof_final = generate_vadcop_final_proof(
                &self.pctx,
                &self.setups,
                &agg_recursive2_proof,
                &trace,
                &prover_buffer,
                options.output_dir_path.clone(),
                const_pols.clone(),
                const_tree.clone(),
                self.d_buffers.get_ptr(),
                false,
            )?;

            vadcop_final_proof = vadcop_proof_final.proof.clone();

            proof_id = Some(
                blake3::hash(unsafe {
                    std::slice::from_raw_parts(vadcop_final_proof.as_ptr() as *const u8, vadcop_final_proof.len() * 8)
                })
                .to_hex()
                .to_string(),
            );

            if options.final_snark {
                timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
                let recursivef_proof = generate_recursivef_proof(
                    &self.pctx,
                    &self.setups,
                    &vadcop_final_proof,
                    &trace,
                    &prover_buffer,
                    options.output_dir_path.clone(),
                    false,
                )?;
                timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

                timer_start_info!(GENERATING_FFLONK_SNARK_PROOF);
                let _ = generate_fflonk_snark_proof(&self.pctx, recursivef_proof, options.output_dir_path.clone());
                timer_stop_and_log_info!(GENERATING_FFLONK_SNARK_PROOF);
            }
        }
        timer_stop_and_log_info!(GENERATING_VADCOP_PROOF);

        if self.pctx.dctx_get_rank() == 0 && options.verify_proofs {
            let setup_path = self.pctx.global_info.get_setup_path("vadcop_final");
            let stark_info_path = setup_path.display().to_string() + ".starkinfo.json";
            let expressions_bin_path = setup_path.display().to_string() + ".verifier.bin";
            let verkey_path = setup_path.display().to_string() + ".verkey.json";

            timer_start_info!(VERIFYING_VADCOP_FINAL_PROOF);
            let valid_proofs = verify_proof(
                vadcop_final_proof.as_mut_ptr(),
                stark_info_path,
                expressions_bin_path,
                verkey_path,
                Some(self.pctx.get_publics().clone()),
                None,
                None,
            );
            timer_stop_and_log_info!(VERIFYING_VADCOP_FINAL_PROOF);
            if !valid_proofs {
                tracing::info!("··· {}", "\u{2717} Vadcop Final proof was not verified".bright_red().bold());
                return Err("Vadcop Final proof was not verified".into());
            } else {
                tracing::info!("··· {}", "\u{2713} Vadcop Final proof was verified".bright_green().bold());
            }
        }

        Ok(proof_id)
    }

    fn prepare_gpu(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        setups_vadcop: Arc<SetupsVadcop<F>>,
        aggregation: bool,
        gpu_params: &ParamsGPU,
    ) -> (Arc<DeviceBuffer>, u64, u64) {
        let free_memory_gpu = match cfg!(feature = "gpu") {
            true => check_device_memory_c() as f64 * 0.98,
            false => 0.0,
        };

        let total_const_area = match gpu_params.preallocate {
            true => sctx.total_const_size as u64,
            false => 0,
        };

        let total_const_area_aggregation = match aggregation && gpu_params.preallocate {
            true => setups_vadcop.total_const_size as u64,
            false => 0,
        };

        let mut max_size_buffer = (free_memory_gpu / 8.0).floor() as u64; //measured in GL elements
        if gpu_params.preallocate {
            max_size_buffer -= sctx.total_const_size as u64;
            if aggregation {
                max_size_buffer -= setups_vadcop.total_const_size as u64;
            }
        }

        let n_streams_per_gpu = match cfg!(feature = "gpu") {
            true => {
                let max_number_proofs_per_gpu = gpu_params
                    .max_number_streams
                    .min((max_size_buffer as usize / sctx.max_prover_buffer_size) as usize);
                if max_number_proofs_per_gpu < 1 {
                    panic!("Not enough GPU memory to run the proof");
                }
                max_number_proofs_per_gpu
            }
            false => 1,
        };

        let max_aux_trace_area = (n_streams_per_gpu * sctx.max_prover_buffer_size) as u64;

        let max_sizes = MaxSizes { total_const_area, max_aux_trace_area, total_const_area_aggregation };

        let max_sizes_ptr = &max_sizes as *const MaxSizes as *mut c_void;
        let d_buffers = Arc::new(DeviceBuffer(gen_device_buffers_c(
            max_sizes_ptr,
            pctx.dctx_get_node_rank() as u32,
            pctx.dctx_get_node_n_processes() as u32,
        )));

        let max_size_const = match !gpu_params.preallocate {
            true => sctx.max_const_size as u64,
            false => 0,
        };

        let max_size_const_tree = match !gpu_params.preallocate {
            true => sctx.max_const_tree_size as u64,
            false => 0,
        };

        let max_size_const_aggregation = match aggregation && !gpu_params.preallocate {
            true => setups_vadcop.max_const_size as u64,
            false => 0,
        };

        let max_size_const_tree_aggregation = match aggregation && !gpu_params.preallocate {
            true => setups_vadcop.max_const_tree_size as u64,
            false => 0,
        };

        let max_proof_size = match aggregation {
            true => sctx.max_proof_size.max(setups_vadcop.max_proof_size) as u64,
            false => sctx.max_proof_size as u64,
        };

        let n_gpus: u64 = gen_device_streams_c(
            d_buffers.get_ptr(),
            sctx.max_prover_trace_size as u64,
            sctx.max_prover_contribution_area as u64,
            sctx.max_prover_buffer_size as u64,
            max_size_const,
            max_size_const_tree,
            setups_vadcop.max_prover_trace_size as u64,
            setups_vadcop.max_prover_buffer_size as u64,
            max_size_const_aggregation,
            max_size_const_tree_aggregation,
            max_proof_size,
            n_streams_per_gpu as u64,
        );

        (d_buffers, n_gpus, n_streams_per_gpu as u64)
    }

    #[allow(clippy::too_many_arguments)]
    fn gen_proof(
        proofs: Arc<DashMap<usize, Proof<F>>>,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        instance_id: usize,
        output_dir_path: PathBuf,
        aux_trace: Arc<Vec<F>>,
        const_pols: Arc<Vec<F>>,
        const_tree: Arc<Vec<F>>,
        d_buffers: Arc<DeviceBuffer>,
        stream_id_: Option<usize>,
        save_proof: bool,
    ) {
        timer_start_info!(GEN_PROOF);
        Self::initialize_air_instance(&pctx, &sctx, instance_id, false);

        let instances = pctx.dctx_get_instances();
        let (airgroup_id, air_id, _) = instances[instance_id];

        let setup = sctx.get_setup(airgroup_id, air_id);
        let p_setup: *mut c_void = (&setup.p_setup).into();
        let air_instance_name = &pctx.global_info.airs[airgroup_id][air_id].name;

        let mut steps_params = pctx.get_air_instance_params(&sctx, instance_id, true);

        if cfg!(not(feature = "gpu")) {
            steps_params.aux_trace = aux_trace.as_ptr() as *mut u8;
            steps_params.p_const_pols = const_pols.as_ptr() as *mut u8;
            steps_params.p_const_tree = const_tree.as_ptr() as *mut u8;
        }

        let p_steps_params: *mut u8 = (&steps_params).into();

        let output_file_path = output_dir_path.join(format!("proofs/{}_{}.json", air_instance_name, instance_id));

        let proof_file = match save_proof {
            true => output_file_path.to_string_lossy().into_owned(),
            false => String::from(""),
        };

        let const_pols_path = setup.setup_path.to_string_lossy().to_string() + ".const";
        let const_pols_tree_path = setup.setup_path.display().to_string() + ".consttree";

        let (skip_recalculation, stream_id) = match stream_id_ {
            Some(stream_id) => (true, stream_id),
            None => (false, 0),
        };

        gen_proof_c(
            p_setup,
            p_steps_params,
            pctx.get_global_challenge_ptr(),
            proofs.get(&instance_id).unwrap().proof.as_ptr() as *mut u64,
            &proof_file,
            airgroup_id as u64,
            air_id as u64,
            instance_id as u64,
            d_buffers.get_ptr(),
            skip_recalculation,
            stream_id as u64,
            &const_pols_path,
            &const_pols_tree_path,
        );

        timer_stop_and_log_info!(GEN_PROOF);
    }

    #[allow(clippy::type_complexity)]
    fn initialize_proofman(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        verify_constraints: bool,
        aggregation: bool,
        final_snark: bool,
        gpu_params: &ParamsGPU,
    ) -> Result<(Arc<ProofCtx<F>>, Arc<SetupCtx<F>>, Arc<SetupsVadcop<F>>), Box<dyn std::error::Error>> {
        timer_start_info!(INITIALIZING_PROOFMAN);

        let mut pctx = ProofCtx::create_ctx(proving_key_path.clone(), custom_commits_fixed, aggregation, final_snark);
        let sctx: Arc<SetupCtx<F>> =
            Arc::new(SetupCtx::new(&pctx.global_info, &ProofType::Basic, verify_constraints, gpu_params.preallocate));
        pctx.set_weights(&sctx);

        let pctx = Arc::new(pctx);
        check_tree_paths(&pctx, &sctx)?;

        Self::initialize_publics(&sctx, &pctx)?;

        let setups_vadcop = Arc::new(SetupsVadcop::new(
            &pctx.global_info,
            verify_constraints,
            aggregation,
            final_snark,
            gpu_params.preallocate,
        ));

        if aggregation {
            check_tree_paths_vadcop(&pctx, &setups_vadcop, final_snark)?;
            initialize_witness_circom(&pctx, &setups_vadcop, final_snark)?;
        }

        timer_stop_and_log_info!(INITIALIZING_PROOFMAN);

        Ok((pctx, sctx, setups_vadcop))
    }

    fn calculate_global_challenge(
        pctx: &ProofCtx<F>,
        roots_contributions: Arc<DashMap<usize, [F; 4]>>,
        values_contributions: Arc<DashMap<usize, Vec<F>>>,
    ) {
        timer_start_info!(CALCULATE_GLOBAL_CHALLENGE);
        let my_instances = pctx.dctx_get_my_instances();

        let mut values = vec![0u64; my_instances.len() * 10];

        for instance_id in my_instances.iter() {
            let mut contribution = vec![F::ZERO; 10];

            let root_contribution = *roots_contributions.get(instance_id).expect("Missing root_contribution");

            let mut values_to_hash =
                values_contributions.get(instance_id).expect("Missing values_contribution").clone();
            values_to_hash[4..(4 + 4)].copy_from_slice(&root_contribution[..4]);

            calculate_hash_c(
                contribution.as_mut_ptr() as *mut u8,
                values_to_hash.as_mut_ptr() as *mut u8,
                values_to_hash.len() as u64,
                10,
            );

            let base_idx = pctx.dctx_get_instance_idx(*instance_id) * 10;
            for (id, v) in contribution.iter().enumerate().take(10) {
                values[base_idx + id] = v.as_canonical_u64();
            }
        }

        let transcript = FFITranscript::new(2, true);

        transcript.add_elements(pctx.get_publics_ptr(), pctx.global_info.n_publics);

        let proof_values_stage = pctx.get_proof_values_by_stage(1);
        if !proof_values_stage.is_empty() {
            transcript.add_elements(proof_values_stage.as_ptr() as *mut u8, proof_values_stage.len());
        }

        let all_roots = pctx.dctx_distribute_roots(values);

        // Add challenges to transcript in order
        for group_idxs in pctx.dctx_get_my_groups() {
            let mut values = Vec::new();
            for idx in group_idxs.iter() {
                let value = vec![
                    F::from_u64(all_roots[*idx]),
                    F::from_u64(all_roots[*idx + 1]),
                    F::from_u64(all_roots[*idx + 2]),
                    F::from_u64(all_roots[*idx + 3]),
                    F::from_u64(all_roots[*idx + 4]),
                    F::from_u64(all_roots[*idx + 5]),
                    F::from_u64(all_roots[*idx + 6]),
                    F::from_u64(all_roots[*idx + 7]),
                    F::from_u64(all_roots[*idx + 8]),
                    F::from_u64(all_roots[*idx + 9]),
                ];
                values.push(value);
            }
            if !values.is_empty() {
                let value = Self::add_contributions(&pctx.global_info.curve, &values);
                transcript.add_elements(value.as_ptr() as *mut u8, value.len());
            }
        }

        let global_challenge = [F::ZERO; 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);

        tracing::info!(
            "··· Global challenge: [{}, {}, {}]",
            global_challenge[0],
            global_challenge[1],
            global_challenge[2]
        );
        pctx.set_global_challenge(2, &global_challenge);

        timer_stop_and_log_info!(CALCULATE_GLOBAL_CHALLENGE);
    }

    #[allow(dead_code)]
    fn diagnostic_instance(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, instance_id: usize) -> bool {
        let instances = pctx.dctx_get_instances();

        let (airgroup_id, air_id, _) = instances[instance_id];
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let air_name = pctx.global_info.airs[airgroup_id][air_id].clone().name;
        let setup = sctx.get_setup(airgroup_id, air_id);
        let cm_pols_map = setup.stark_info.cm_pols_map.as_ref().unwrap();
        let n_cols = *setup.stark_info.map_sections_n.get("cm1").unwrap() as usize;
        let n_rows = 1 << setup.stark_info.stark_struct.n_bits;

        let vals = unsafe {
            std::slice::from_raw_parts(pctx.get_air_instance_trace_ptr(instance_id) as *mut u64, n_cols * n_rows)
        };

        let mut invalid_initialization = false;

        for (pos, val) in vals.iter().enumerate() {
            if *val == u64::MAX - 1 {
                let row = pos / n_cols;
                let col_id = pos % n_cols;
                let col = cm_pols_map.get(col_id).unwrap();
                let col_name = if !col.lengths.is_empty() {
                    let lengths = col.lengths.iter().fold(String::new(), |mut acc, l| {
                        write!(acc, "[{}]", l).unwrap();
                        acc
                    });
                    format!("{}{}", col.name, lengths)
                } else {
                    col.name.clone()
                };
                tracing::warn!(
                    "Missing initialization {} at row {} of {} in instance {}",
                    col_name,
                    row,
                    air_name,
                    air_instance_id,
                );
                invalid_initialization = true;
            }
        }

        invalid_initialization
    }

    fn initialize_air_instance(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, instance_id: usize, init_aux_trace: bool) {
        let instances = pctx.dctx_get_instances();

        let (airgroup_id, air_id, _) = instances[instance_id];
        let setup = sctx.get_setup(airgroup_id, air_id);

        let mut air_instance = pctx.air_instances.get_mut(&instance_id).unwrap();
        if init_aux_trace {
            air_instance.init_aux_trace(setup.prover_buffer_size as usize);
        }
        air_instance.init_evals(setup.stark_info.ev_map.len() * 3);
        air_instance.init_challenges(
            (setup.stark_info.challenges_map.as_ref().unwrap().len() + setup.stark_info.stark_struct.steps.len() + 1)
                * 3,
        );

        air_instance.init_custom_commit_fixed_trace(setup.custom_commits_fixed_buffer_size as usize);

        let n_custom_commits = setup.stark_info.custom_commits.len();

        for commit_id in 0..n_custom_commits {
            if setup.stark_info.custom_commits[commit_id].stage_widths[0] > 0 {
                let custom_commit_file_path =
                    pctx.get_custom_commits_fixed_buffer(&setup.stark_info.custom_commits[commit_id].name).unwrap();

                load_custom_commit_c(
                    (&setup.p_setup).into(),
                    commit_id as u64,
                    air_instance.get_custom_commits_fixed_ptr(),
                    custom_commit_file_path.to_str().expect("Invalid path"),
                );
            }
        }

        let n_airgroup_values = setup
            .stark_info
            .airgroupvalues_map
            .as_ref()
            .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
            .unwrap_or(0);

        let n_air_values = setup
            .stark_info
            .airvalues_map
            .as_ref()
            .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
            .unwrap_or(0);

        if n_air_values > 0 && air_instance.airvalues.is_empty() {
            air_instance.init_airvalues(n_air_values);
        }

        if n_airgroup_values > 0 && air_instance.airgroup_values.is_empty() {
            air_instance.init_airgroup_values(n_airgroup_values);
        }
    }

    fn initialize_publics(sctx: &SetupCtx<F>, pctx: &ProofCtx<F>) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Initializing publics custom_commits");
        for (airgroup_id, airs) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in airs.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id);
                for custom_commit in &setup.stark_info.custom_commits {
                    if custom_commit.stage_widths[0] > 0 {
                        // Handle the possibility that this returns None
                        let custom_file_path = pctx.get_custom_commits_fixed_buffer(&custom_commit.name)?;

                        let mut file = File::open(custom_file_path)?;
                        let mut root_bytes = [0u8; 32];
                        file.read_exact(&mut root_bytes)?;

                        for (idx, p) in custom_commit.public_values.iter().enumerate() {
                            let public_id = p.idx as usize;
                            let byte_range = idx * 8..(idx + 1) * 8;
                            let value = u64::from_le_bytes(root_bytes[byte_range].try_into()?);
                            pctx.set_public_value(value, public_id);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn calculate_im_pols(stage: u32, sctx: &SetupCtx<F>, pctx: &ProofCtx<F>, instance_id: usize) {
        let instances = pctx.dctx_get_instances();
        let (airgroup_id, air_id, _) = instances[instance_id];
        let setup = sctx.get_setup(airgroup_id, air_id);

        let steps_params = pctx.get_air_instance_params(sctx, instance_id, false);

        calculate_impols_expressions_c((&setup.p_setup).into(), stage as u64, (&steps_params).into());
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_contribution_air(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        roots_contributions: Arc<DashMap<usize, [F; 4]>>,
        values_contributions: Arc<DashMap<usize, Vec<F>>>,
        instance_id: usize,
        aux_trace_contribution_ptr: *mut u8,
        d_buffers: Arc<DeviceBuffer>,
        streams: Arc<Mutex<Vec<Option<u64>>>>,
    ) {
        let n_field_elements = 4;

        timer_start_info!(GET_CONTRIBUTION_AIR);
        let instances = pctx.dctx_get_instances();

        let (airgroup_id, air_id, _) = instances[instance_id];
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let air_values = pctx.get_air_instance_air_values(airgroup_id, air_id, air_instance_id).clone();

        let root_ptr = roots_contributions.get(&instance_id).unwrap().value().as_ptr() as *mut u8;
        let stream_id = commit_witness_c(
            3,
            setup.stark_info.stark_struct.n_bits,
            setup.stark_info.stark_struct.n_bits_ext,
            *setup.stark_info.map_sections_n.get("cm1").unwrap(),
            instance_id as u64,
            root_ptr,
            pctx.get_air_instance_trace_ptr(instance_id),
            aux_trace_contribution_ptr,
            d_buffers.get_ptr(),
            (&setup.p_setup).into(),
        );
        streams.lock().unwrap()[stream_id as usize] = Some(instance_id as u64);

        let n_airvalues = setup
            .stark_info
            .airvalues_map
            .as_ref()
            .map(|map| map.iter().filter(|entry| entry.stage == 1).count())
            .unwrap_or(0);

        let size = 2 * n_field_elements + n_airvalues;

        let mut values_hash = vec![F::ZERO; size];

        values_hash[..n_field_elements].copy_from_slice(&setup.verkey[..n_field_elements]);

        let airvalues_map = setup.stark_info.airvalues_map.as_ref().unwrap();
        let mut p = 0;
        let mut count = 0;
        for air_value in airvalues_map {
            if air_value.stage == 1 {
                values_hash[2 * n_field_elements + count] = air_values[p];
                count += 1;
                p += 1;
            }
        }

        values_contributions.insert(instance_id, values_hash);

        timer_stop_and_log_info!(GET_CONTRIBUTION_AIR);
    }

    fn add_contributions(curve_type: &CurveType, values: &[Vec<F>]) -> Vec<F> {
        if *curve_type == CurveType::EcGFp5 {
            let mut result = EcGFp5::hash_to_curve(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcGFp5::hash_to_curve(
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[0..5]),
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[5..10]),
                );

                result = result.add(&curve_point);
            }

            let mut curve_point_values = vec![F::ZERO; 10];
            curve_point_values[0..5].copy_from_slice(result.x().as_basis_coefficients_slice());
            curve_point_values[5..10].copy_from_slice(result.y().as_basis_coefficients_slice());
            curve_point_values
        } else {
            let mut result = EcMasFp5::hash_to_curve(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcMasFp5::hash_to_curve(
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[0..5]),
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[5..10]),
                );
                result = result.add(&curve_point);
            }

            let mut curve_point_values = vec![F::ZERO; 10];
            curve_point_values[0..5].copy_from_slice(result.x().as_basis_coefficients_slice());
            curve_point_values[5..10].copy_from_slice(result.y().as_basis_coefficients_slice());
            curve_point_values
        }
    }
}
