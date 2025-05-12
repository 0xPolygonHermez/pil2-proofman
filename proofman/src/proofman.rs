use curves::{EcGFp5, EcMasFp5, curve::EllipticCurve, goldilocks_quintic_extension::GoldilocksQuinticExtension};
use libloading::{Library, Symbol};
use log::info;
use p3_field::extension::BinomialExtensionField;
use p3_field::BasedVectorSpace;
use std::ops::Add;
use proofman_common::{load_const_pols, load_const_pols_tree, CurveType};
use proofman_common::{
    calculate_fixed_tree, skip_prover_instance, Proof, ProofCtx, ProofType, ProofOptions, SetupCtx, SetupsVadcop,
};
use colored::Colorize;
use proofman_hints::aggregate_airgroupvals;
use proofman_starks_lib_c::{gen_device_commit_buffers_c, gen_device_commit_buffers_free_c};
use proofman_starks_lib_c::{
    save_challenges_c, save_proof_values_c, save_publics_c, get_const_offset_c, check_gpu_memory_c,
    set_max_size_thread_c,
};
use std::fs;
use std::collections::HashMap;
use std::fs::File;
use std::fmt::Write;
use std::io::Read;
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool};
use std::sync::atomic::Ordering;
use std::sync::{Mutex, RwLock};
use p3_goldilocks::Goldilocks;

use p3_field::PrimeField64;
use proofman_starks_lib_c::{
    gen_proof_c, get_proof_c, commit_witness_c, get_commit_root_c, calculate_hash_c, load_custom_commit_c,
    calculate_impols_expressions_c,
};

use std::{path::PathBuf, sync::Arc};

use transcript::FFITranscript;

use witness::{WitnessLibInitFn, WitnessLibrary, WitnessManager};
use crate::{check_tree_paths_vadcop, initialize_fixed_pols_tree};
use crate::{verify_basic_proof, verify_proof, verify_global_constraints_proof};
use crate::MaxSizes;
use crate::{verify_constraints_proof, print_summary_info, get_recursive_buffer_sizes};
use crate::{
    gen_witness_recursive, gen_witness_aggregation, generate_recursive_proof, generate_vadcop_final_proof,
    generate_fflonk_snark_proof, generate_recursivef_proof, initialize_size_witness,
};
use crate::check_tree_paths;
use crate::WitnessBuffer;
use crate::ThreadInstanceInfo;
use crate::WitnessPendingCounter;
use crate::aggregate_recursive2_proofs;

use std::ffi::c_void;

use proofman_util::{create_buffer_fast, timer_start_info, timer_stop_and_log_info, DeviceBuffer};

pub struct ProofMan<F: PrimeField64> {
    pctx: Arc<ProofCtx<F>>,
    sctx: Arc<SetupCtx<F>>,
    setups: Arc<SetupsVadcop<F>>,
    d_buffers: Arc<DeviceBuffer>,
    max_number_proofs: usize,
    traces: Vec<Arc<Vec<F>>>,
    prover_buffers: Vec<Arc<Vec<F>>>,
    wcm: Arc<WitnessManager<F>>,
}

impl<F: PrimeField64> Drop for ProofMan<F> {
    fn drop(&mut self) {
        let mpi_node_rank = self.pctx.dctx_get_node_rank() as u32;
        gen_device_commit_buffers_free_c(self.d_buffers.get_ptr(), mpi_node_rank);
    }
}
impl<F: PrimeField64> ProofMan<F>
where
    BinomialExtensionField<Goldilocks, 5>: BasedVectorSpace<F>,
{
    const MY_NAME: &'static str = "ProofMan";

    pub fn check_setup(proving_key_path: PathBuf, options: ProofOptions) -> Result<(), Box<dyn std::error::Error>> {
        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {:?}", proving_key_path).into());
        }

        let pctx = ProofCtx::<F>::create_ctx(proving_key_path.clone(), HashMap::new(), options);

        let setups_aggregation = Arc::new(SetupsVadcop::<F>::new(
            &pctx.global_info,
            false,
            pctx.options.aggregation,
            false,
            pctx.options.final_snark,
        ));

        let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false, pctx.options.preallocate);

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                calculate_fixed_tree(sctx.get_setup(airgroup_id, air_id));
            }
        }

        if pctx.options.aggregation {
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

            if pctx.options.final_snark {
                let setup_recursivef = setups_aggregation.setup_recursivef.as_ref().unwrap();
                calculate_fixed_tree(setup_recursivef);
            }
        }

        Ok(())
    }

    pub fn verify_proof_constraints(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        output_dir_path: PathBuf,
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
        let mut witness_lib = witness_lib(self.pctx.options.verbose_mode)?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);

        self.register_witness(&mut *witness_lib);

        self._verify_proof_constraints()
    }

    pub fn verify_proof_constraints_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.wcm.set_input_data_path(input_data_path);

        self._verify_proof_constraints()
    }

    fn _verify_proof_constraints(
        &self,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.pctx.dctx_reset();

        if !self.wcm.is_init_witness() {
            return Err("Witness computation dynamic library not initialized".into());
        }

        timer_start_info!(EXECUTE);
        self.wcm.execute();
        timer_stop_and_log_info!(EXECUTE);

        self.pctx.dctx_assign_instances();
        self.pctx.dctx_close();

        print_summary_info(Self::MY_NAME, &self.pctx, &self.sctx);

        let transcript = FFITranscript::new(2, true);
        let dummy_element = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let global_challenge = [F::ZERO; 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);
        self.pctx.set_global_challenge(2, &global_challenge);
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let instances = self.pctx.dctx_get_instances();
        let airgroup_values_air_instances = Arc::new(Mutex::new(Vec::new()));
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
                )
            });
        }

        if let Some(handle) = thread_handle {
            handle.join().unwrap()
        }

        self.wcm.end();

        let check_global_constraints = self.pctx.options.debug_info.debug_instances.is_empty()
            || !self.pctx.options.debug_info.debug_global_instances.is_empty();

        if check_global_constraints {
            let airgroup_values_air_instances =
                Arc::try_unwrap(airgroup_values_air_instances).unwrap().into_inner().unwrap();
            let airgroupvalues_u64 = aggregate_airgroupvals(&self.pctx, &airgroup_values_air_instances);
            let airgroupvalues = self.pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

            if self.pctx.dctx_get_rank() == 0 {
                let valid_global_constraints = verify_global_constraints_proof(&self.pctx, &self.sctx, airgroupvalues);

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
    ) -> std::thread::JoinHandle<()> {
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

            wcm.debug(&[instance_id]);

            if pctx.options.verify_constraints {
                let valid = verify_constraints_proof(&pctx, &sctx, instance_id);
                if !valid {
                    valid_constraints.fetch_and(valid, Ordering::Relaxed);
                }
            }

            airgroup_values_air_instances.lock().unwrap().push(pctx.get_air_instance_airgroup_values(
                airgroup_id,
                air_id,
                air_instance_id,
            ));
            pctx.free_instance(instance_id);
        })
    }

    pub fn generate_proof(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        output_dir_path: PathBuf,
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

        if !output_dir_path.exists() {
            fs::create_dir_all(&output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {:?}", err))?;
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(self.pctx.options.verbose_mode)?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);

        self.register_witness(&mut *witness_lib);

        self._generate_proof(output_dir_path)
    }

    pub fn generate_proof_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
        output_dir_path: PathBuf,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if !output_dir_path.exists() {
            fs::create_dir_all(&output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {:?}", err))?;
        }

        self.wcm.set_input_data_path(input_data_path);
        self._generate_proof(output_dir_path)
    }

    pub fn new(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
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

        let (pctx, sctx, setups_vadcop) =
            Self::initialize_proofman(proving_key_path, custom_commits_fixed, options.clone())?;

        let (max_number_proofs, d_buffers) = Self::prepare_gpu(pctx.clone(), sctx.clone(), setups_vadcop.clone());

        let (traces, prover_buffers): (Vec<_>, Vec<_>) = if pctx.options.aggregation {
            let (trace_size, prover_buffer_size) = get_recursive_buffer_sizes(&pctx, &setups_vadcop)?;

            (0..max_number_proofs)
                .map(|_| {
                    let trace = Arc::new(create_buffer_fast::<F>(trace_size));
                    let prover_buffer = Arc::new(create_buffer_fast::<F>(prover_buffer_size));
                    (trace, prover_buffer)
                })
                .unzip()
        } else {
            let empty_buffer = Arc::new(Vec::new());
            (vec![empty_buffer.clone(); max_number_proofs], vec![empty_buffer.clone(); max_number_proofs])
        };

        initialize_fixed_pols_tree(&pctx, &sctx, &setups_vadcop, d_buffers.clone());

        let wcm = Arc::new(WitnessManager::new(
            pctx.clone(),
            sctx.clone(),
        ));

        timer_stop_and_log_info!(INIT_PROOFMAN);

        Ok(Self { pctx, sctx, wcm, setups: setups_vadcop, d_buffers, max_number_proofs, traces, prover_buffers })
    }
    
    pub fn register_witness(&self, witness_lib: &mut dyn WitnessLibrary<F>,
    ) {
        timer_start_info!(REGISTERING_WITNESS);
        witness_lib.register_witness(self.wcm.clone());
        self.wcm.set_init_witness(true);
        timer_stop_and_log_info!(REGISTERING_WITNESS);
    }

    #[allow(clippy::too_many_arguments)]
    fn _generate_proof(
        &self,
        output_dir_path: PathBuf,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        timer_start_info!(GENERATING_VADCOP_PROOF);

        timer_start_info!(GENERATING_PROOFS);

        timer_start_info!(EXECUTE);

        if !self.wcm.is_init_witness() {
            println!("Witness computation dynamic library not initialized");
            return Err("Witness computation dynamic library not initialized".into());
        }

        self.pctx.dctx_reset();

        self.wcm.execute();

        self.pctx.dctx_assign_instances();
        self.pctx.dctx_close();

        print_summary_info(Self::MY_NAME, &self.pctx, &self.sctx);

        timer_stop_and_log_info!(EXECUTE);

        timer_start_info!(CALCULATING_CONTRIBUTIONS);

        let instances = self.pctx.dctx_get_instances();
        let my_instances = self.pctx.dctx_get_my_instances();
        let mpi_node_rank = self.pctx.dctx_get_node_rank() as u32;

        let values = Arc::new((0..my_instances.len() * 10).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());

        let aux_trace_size = match cfg!(feature = "gpu") {
            true => {
                if self.pctx.options.preallocate {
                    0
                } else {
                    self.max_number_proofs * (self.sctx.max_const_tree_size + self.sctx.max_const_size)
                }
            }
            false => self.sctx.max_prover_buffer_size,
        };
        let aux_trace: Arc<Vec<F>> = Arc::new(create_buffer_fast(aux_trace_size));

        // TODO: ADD AS OPTIONS
        let n_threads_pool = 4;
        let max_witness_stored = 32;

        let max_num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let (threads_per_pool_contributions, max_concurrent_pools_contributions) = match cfg!(feature = "gpu") {
            true => {
                let max_concurrent_pools = max_num_threads / n_threads_pool;
                (n_threads_pool, max_concurrent_pools)
            }
            false => (max_num_threads, 1),
        };

        let witness_pending = Arc::new(WitnessPendingCounter::new());

        let (witness_tx, witness_rx) = crossbeam_channel::unbounded::<usize>();
        let precomputed_witnesses = Arc::new(WitnessBuffer::new(max_witness_stored));

        let contribution_pools: Vec<_> = (0..self.max_number_proofs)
            .map(|contribution_thread_id| {
                let pctx_clone = self.pctx.clone();
                let sctx_clone = self.sctx.clone();
                let aux_trace_clone = aux_trace.clone();
                let values_clone = values.clone();
                let d_buffers_clone = self.d_buffers.clone();
                let precomputed_witnesses = precomputed_witnesses.clone();

                std::thread::spawn(move || {
                    while let Some(instance_id) = precomputed_witnesses.pop() {
                        let value = Self::get_contribution_air(
                            pctx_clone.clone(),
                            &sctx_clone,
                            instance_id,
                            aux_trace_clone.clone().as_ptr() as *mut u8,
                            contribution_thread_id,
                            d_buffers_clone.clone(),
                        );

                        let base_idx = pctx_clone.dctx_get_instance_idx(instance_id) * 10;
                        for (id, v) in value.iter().enumerate().take(10) {
                            values_clone[base_idx + id].store(*v, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        let witness_pools: Vec<_> = (0..max_concurrent_pools_contributions)
            .map(|thread_id| {
                let witness_rx = witness_rx.clone();
                let pctx_clone = self.pctx.clone();
                let instances = instances.clone();
                let wcm_clone = self.wcm.clone();
                let precomputed_witnesses = precomputed_witnesses.clone();
                let witness_pending_clone = witness_pending.clone();

                std::thread::spawn(move || loop {
                    if !precomputed_witnesses.wait_until_below_capacity() {
                        break;
                    }

                    let instance_id = match witness_rx.recv() {
                        Ok(id) => id,
                        Err(_) => break,
                    };

                    wcm_clone.calculate_witness(
                        1,
                        &[instance_id],
                        threads_per_pool_contributions * thread_id,
                        threads_per_pool_contributions,
                    );
                    let (_, _, all) = instances[instance_id];
                    if !all {
                        witness_pending_clone.decrement();
                    }

                    if pctx_clone.dctx_is_my_instance(instance_id) {
                        precomputed_witnesses.push(instance_id);
                    }
                })
            })
            .collect();

        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            if !*all && self.pctx.dctx_is_my_instance(instance_id) {
                witness_pending.increment();
                witness_tx.send(instance_id).unwrap();
            }
        }

        witness_pending.wait_for_tables_ready();

        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            if *all && self.pctx.dctx_is_my_instance(instance_id) {
                witness_tx.send(instance_id).unwrap();
            }
        }

        drop(witness_tx);
        for pool in witness_pools {
            pool.join().unwrap();
        }

        precomputed_witnesses.close();
        for worker in contribution_pools {
            worker.join().unwrap();
        }

        timer_start_info!(CALCULATE_GLOBAL_CHALLENGE);
        let values_challenge = Arc::try_unwrap(values)
            .expect("Arc still has other references")
            .into_iter()
            .map(|atomic| atomic.load(std::sync::atomic::Ordering::Relaxed))
            .collect::<Vec<u64>>();
        Self::calculate_global_challenge(&self.pctx, values_challenge);
        timer_stop_and_log_info!(CALCULATE_GLOBAL_CHALLENGE);

        timer_stop_and_log_info!(CALCULATING_CONTRIBUTIONS);

        timer_start_info!(GENERATING_BASIC_PROOFS);

        if self.pctx.options.aggregation {
            initialize_size_witness(&self.pctx, &self.setups)?;
        }

        let proofs = Arc::new(RwLock::new(vec![Proof::default(); my_instances.len()]));
        let recursive_witness = Arc::new(RwLock::new(vec![None; my_instances.len()]));

        let airgroup_values_air_instances = Arc::new(Mutex::new(vec![Vec::new(); my_instances.len()]));

        let (threads_per_pool_basic_proofs, max_concurrent_pools_basic_proofs) = match cfg!(feature = "gpu") {
            true => {
                let threads_per_pool = 4;
                let max_concurrent_pools = max_num_threads / threads_per_pool;
                (threads_per_pool, max_concurrent_pools)
            }
            false => (max_num_threads, 1),
        };

        let (witness_tx, witness_rx) = crossbeam_channel::unbounded::<usize>();
        let precomputed_witnesses = Arc::new(WitnessBuffer::new(max_witness_stored));

        let proofs_clone = proofs.clone();
        let pctx_clone = self.pctx.clone();
        let setups_clone = self.setups.clone();
        let recursive_witness_clone = recursive_witness.clone();

        let (witness_recursive_tx, witness_recursive_rx) = crossbeam_channel::unbounded::<usize>();
        let witness_recursive_worker = std::thread::spawn(move || {
            while let Ok(instance_id) = witness_recursive_rx.recv() {
                let inst_id = pctx_clone.dctx_get_instance_idx(instance_id);
                let proof = proofs_clone.read().unwrap()[inst_id].clone();
                let witness = gen_witness_recursive(&pctx_clone, &setups_clone, &proof).unwrap();
                recursive_witness_clone.write().unwrap()[inst_id] = Some(witness);
            }
        });

        let witness_pools: Vec<_> = (0..max_concurrent_pools_basic_proofs)
            .map(|thread_id| {
                let witness_rx = witness_rx.clone();
                let wcm_clone = self.wcm.clone();
                let precomputed_witnesses = precomputed_witnesses.clone();
                let instances_clone = instances.clone();
                std::thread::spawn(move || loop {
                    if !precomputed_witnesses.wait_until_below_capacity() {
                        break;
                    }

                    let instance_id = match witness_rx.recv() {
                        Ok(id) => id,
                        Err(_) => break,
                    };

                    let (_, _, all) = instances_clone[instance_id];
                    if !all {
                        wcm_clone.calculate_witness(
                            1,
                            &[instance_id],
                            threads_per_pool_basic_proofs * thread_id,
                            threads_per_pool_basic_proofs,
                        );
                    }
                    precomputed_witnesses.push(instance_id);
                })
            })
            .collect();

        let thread_info = Arc::new(
            (0..self.max_number_proofs)
                .map(|_| ThreadInstanceInfo {
                    airgroup_id: AtomicUsize::new(0),
                    air_id: AtomicUsize::new(0),
                    proof_type: AtomicUsize::new(0),
                })
                .collect::<Vec<_>>(),
        );

        let basic_proof_pools: Vec<_> = (0..self.max_number_proofs)
            .map(|proof_thread_id| {
                let instances_clone = instances.clone();
                let witness_recursive_tx = witness_recursive_tx.clone();
                let aux_trace_clone = aux_trace.clone();
                let pctx_clone = self.pctx.clone();
                let sctx_clone = self.sctx.clone();
                let proofs_clone = proofs.clone();
                let output_dir_path_clone = output_dir_path.clone();
                let airgroup_values_air_instances_clone = airgroup_values_air_instances.clone();
                let d_buffers_clone = self.d_buffers.clone();
                let precomputed_witnesses = precomputed_witnesses.clone();
                let thread_info_clone = thread_info.clone();

                std::thread::spawn(move || {
                    while let Some(instance_id) = precomputed_witnesses.pop() {
                        let (airgroup_id, air_id, _) = instances_clone[instance_id];
                        let last_airgroup_id = thread_info_clone[proof_thread_id].airgroup_id.load(Ordering::Relaxed);
                        let last_air_id = thread_info_clone[proof_thread_id].air_id.load(Ordering::Relaxed);
                        let gen_const_tree = !pctx_clone.options.preallocate
                            && (airgroup_id != last_airgroup_id || air_id != last_air_id);
                        if gen_const_tree {
                            thread_info_clone[proof_thread_id].airgroup_id.store(airgroup_id, Ordering::Relaxed);
                            thread_info_clone[proof_thread_id].air_id.store(air_id, Ordering::Relaxed);
                            thread_info_clone[proof_thread_id]
                                .proof_type
                                .store(ProofType::Basic as usize, Ordering::Relaxed);
                        }
                        Self::generate_proof_thread(
                            proofs_clone.clone(),
                            pctx_clone.clone(),
                            sctx_clone.clone(),
                            proof_thread_id,
                            instance_id,
                            airgroup_id,
                            air_id,
                            output_dir_path_clone.clone(),
                            aux_trace_clone.clone(),
                            airgroup_values_air_instances_clone.clone(),
                            gen_const_tree,
                            d_buffers_clone.clone(),
                        );

                        pctx_clone.free_instance(instance_id);
                        if pctx_clone.options.aggregation {
                            witness_recursive_tx.send(instance_id).unwrap();
                        }
                    }
                })
            })
            .collect();

        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            if *all && self.pctx.dctx_is_my_instance(instance_id) {
                witness_tx.send(instance_id).unwrap();
            }
        }

        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            if !*all && self.pctx.dctx_is_my_instance(instance_id) {
                witness_tx.send(instance_id).unwrap();
            }
        }

        drop(witness_tx);
        for pool in witness_pools {
            pool.join().unwrap();
        }

        precomputed_witnesses.close();
        for worker in basic_proof_pools {
            worker.join().unwrap();
        }

        drop(witness_recursive_tx);
        witness_recursive_worker.join().unwrap();

        timer_stop_and_log_info!(GENERATING_BASIC_PROOFS);

        timer_stop_and_log_info!(GENERATING_PROOFS);

        let global_info_path = self.pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
        let global_info_file = global_info_path.to_str().unwrap();

        save_challenges_c(self.pctx.get_challenges_ptr(), global_info_file, output_dir_path.to_string_lossy().as_ref());
        save_proof_values_c(
            self.pctx.get_proof_values_ptr(),
            global_info_file,
            output_dir_path.to_string_lossy().as_ref(),
        );
        save_publics_c(
            self.pctx.global_info.n_publics as u64,
            self.pctx.get_publics_ptr(),
            output_dir_path.to_string_lossy().as_ref(),
        );

        let airgroup_values_air_instances =
            Arc::try_unwrap(airgroup_values_air_instances).unwrap().into_inner().unwrap();

        if !self.pctx.options.aggregation {
            let mut valid_proofs = true;

            if self.pctx.options.verify_proofs {
                timer_start_info!(VERIFYING_PROOFS);
                let proofs_ = Arc::try_unwrap(proofs).unwrap().into_inner().unwrap();
                for instance_id in my_instances.iter() {
                    let valid_proof = verify_basic_proof(
                        &self.pctx,
                        *instance_id,
                        &proofs_[self.pctx.dctx_get_instance_idx(*instance_id)].proof,
                    );
                    if !valid_proof {
                        valid_proofs = false;
                    }
                }
                timer_stop_and_log_info!(VERIFYING_PROOFS);
            }

            let check_global_constraints = self.pctx.options.debug_info.debug_instances.is_empty()
                || !self.pctx.options.debug_info.debug_global_instances.is_empty();

            if check_global_constraints {
                let airgroupvalues_u64 = aggregate_airgroupvals(&self.pctx, &airgroup_values_air_instances);
                let airgroupvalues = self.pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

                if self.pctx.dctx_get_rank() == 0 {
                    let valid_global_constraints =
                        verify_global_constraints_proof(&self.pctx, &self.sctx, airgroupvalues);
                    if valid_global_constraints.is_err() {
                        valid_proofs = false;
                    }
                }
            }

            if valid_proofs {
                log::info!(
                    "{}: ··· {}",
                    Self::MY_NAME,
                    "\u{2713} All proofs were successfully verified".bright_green().bold()
                );
                return Ok(None);
            } else {
                return Err("Basic proofs were not verified".into());
            }
        }

        timer_start_info!(GENERATING_COMPRESSED_PROOFS);
        timer_start_info!(GENERATING_INNER_COMPRESSED_PROOFS);

        let max_number_recursive_proofs = self.max_number_proofs;

        let n_airgroups = self.pctx.global_info.air_groups.len();
        let recursive2_witnesses = Arc::new(RwLock::new(vec![Vec::new(); n_airgroups]));
        let recursive2_proofs = Arc::new(RwLock::new(vec![vec![]; n_airgroups]));

        let (witness_recursive2_tx, witness_recursive2_rx) = crossbeam_channel::unbounded::<usize>();
        let (recursive_proofs_tx, recursive_proofs_rx) = crossbeam_channel::unbounded::<(usize, usize)>();

        let num_witness_workers = 4;

        let recursive_proofs_counter = Arc::new(AtomicUsize::new(0));

        let pctx_clone = self.pctx.clone();
        let setups_clone = self.setups.clone();
        let recursive2_witness_clone = recursive2_witnesses.clone();
        let recursive2_proofs_clone = recursive2_proofs.clone();
        let witness_recursive2_workers: Vec<_> = (0..num_witness_workers)
            .map(|_| {
                let pctx_clone = Arc::clone(&pctx_clone);
                let setups_clone = Arc::clone(&setups_clone);
                let proofs = Arc::clone(&recursive2_proofs_clone);
                let witnesses = Arc::clone(&recursive2_witness_clone);
                let witness_recursive2_rx = witness_recursive2_rx.clone();
                let recursive_proofs_tx = recursive_proofs_tx.clone();
                let recursive_proofs_counter = recursive_proofs_counter.clone();

                std::thread::spawn(move || {
                    while let Ok(airgroup_id) = witness_recursive2_rx.recv() {
                        if airgroup_id == usize::MAX {
                            break;
                        }
                        let mut proofs_guard = proofs.write().unwrap();
                        let proofs_airgroup = &mut proofs_guard[airgroup_id];
                        if proofs_airgroup.len() >= 3 {
                            let p1 = proofs_airgroup.pop().unwrap();
                            let p2 = proofs_airgroup.pop().unwrap();
                            let p3 = proofs_airgroup.pop().unwrap();
                            let witness_recursive2 =
                                gen_witness_aggregation(&pctx_clone, &setups_clone, &p1, &p2, &p3).unwrap();

                            witnesses.write().unwrap()[airgroup_id].push(Some(witness_recursive2));

                            recursive_proofs_counter.fetch_add(1, Ordering::SeqCst);
                            recursive_proofs_tx.send((airgroup_id, ProofType::Recursive2 as usize)).unwrap();
                        }
                        recursive_proofs_counter.fetch_sub(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        let recursive_proof_pools: Vec<_> = (0..max_number_recursive_proofs)
            .map(|proof_thread_id| {
                let instances_clone = instances.clone();
                let pctx_clone = self.pctx.clone();
                let setups_clone = self.setups.clone();
                let recursive_witness_clone = recursive_witness.clone();
                let recursive2_witness_clone = recursive2_witnesses.clone();
                let output_dir_path_clone = output_dir_path.clone();
                let d_buffers_clone = self.d_buffers.clone();
                let thread_info_clone = thread_info.clone();
                let traces_clone = self.traces.clone();
                let prover_buffers_clone = self.prover_buffers.clone();
                let recursive_proofs_rx = recursive_proofs_rx.clone();
                let recursive2_proofs = recursive2_proofs.clone();
                let witness_recursive2_tx = witness_recursive2_tx.clone();
                let recursive_proofs_tx = recursive_proofs_tx.clone();
                let recursive_proofs_counter = recursive_proofs_counter.clone();

                std::thread::spawn(move || {
                    while let Ok((id, proof_type)) = recursive_proofs_rx.recv() {
                        if id == usize::MAX && proof_type == usize::MAX {
                            break;
                        }
                        let (airgroup_id, air_id, _) = match proof_type == ProofType::Recursive2 as usize {
                            true => (id, 0, false),
                            false => instances_clone[id],
                        };

                        let witness = match proof_type == ProofType::Recursive2 as usize {
                            true => recursive2_witness_clone.write().unwrap()[airgroup_id].pop().unwrap().unwrap(),
                            false => recursive_witness_clone.write().unwrap()[id].take().unwrap(),
                        };

                        let last_airgroup_id = thread_info_clone[proof_thread_id].airgroup_id.load(Ordering::Relaxed);
                        let last_air_id = thread_info_clone[proof_thread_id].air_id.load(Ordering::Relaxed);
                        let last_proof_type = thread_info_clone[proof_thread_id].proof_type.load(Ordering::Relaxed);
                        let gen_const_tree = !pctx_clone.options.preallocate
                            && (airgroup_id != last_airgroup_id
                                || air_id != last_air_id
                                || proof_type != last_proof_type);
                        if gen_const_tree {
                            thread_info_clone[proof_thread_id].airgroup_id.store(airgroup_id, Ordering::Relaxed);
                            thread_info_clone[proof_thread_id].air_id.store(air_id, Ordering::Relaxed);
                            thread_info_clone[proof_thread_id]
                                .proof_type
                                .store(ProofType::Basic as usize, Ordering::Relaxed);
                        }

                        let proof = generate_recursive_proof(
                            &pctx_clone,
                            &setups_clone,
                            &witness,
                            &traces_clone[proof_thread_id],
                            &prover_buffers_clone[proof_thread_id],
                            &output_dir_path_clone,
                            d_buffers_clone.get_ptr(),
                            proof_thread_id,
                            gen_const_tree,
                            mpi_node_rank,
                        );

                        if proof_type == ProofType::Compressor as usize {
                            let witness = gen_witness_recursive(&pctx_clone, &setups_clone, &proof).unwrap();
                            recursive_witness_clone.write().unwrap()[id] = Some(witness);
                            recursive_proofs_counter.fetch_add(1, Ordering::SeqCst);
                            recursive_proofs_tx.send((id, ProofType::Recursive1 as usize)).unwrap();
                        } else {
                            recursive2_proofs.write().unwrap()[airgroup_id].push(proof);
                            recursive_proofs_counter.fetch_add(1, Ordering::SeqCst);
                            witness_recursive2_tx.send(airgroup_id).unwrap();
                        }
                        recursive_proofs_counter.fetch_sub(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for (instance_id, (airgroup_id, air_id, _)) in instances.iter().enumerate() {
            if self.pctx.dctx_is_my_instance(instance_id) {
                let proof_type = if self.pctx.global_info.get_air_has_compressor(*airgroup_id, *air_id) {
                    ProofType::Compressor as usize
                } else {
                    ProofType::Recursive1 as usize
                };
                recursive_proofs_counter.fetch_add(1, Ordering::SeqCst);
                recursive_proofs_tx.send((instance_id, proof_type)).unwrap();
            }
        }

        std::thread::spawn(move || loop {
            if recursive_proofs_counter.load(Ordering::SeqCst) == 0 {
                for _ in 0..num_witness_workers {
                    witness_recursive2_tx.send(usize::MAX).unwrap();
                }
                for _ in 0..max_number_recursive_proofs {
                    recursive_proofs_tx.send((usize::MAX, usize::MAX)).unwrap();
                }
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        });

        for pool in witness_recursive2_workers {
            pool.join().unwrap();
        }
        for pool in recursive_proof_pools {
            pool.join().unwrap();
        }

        let mut recursive2_proofs_data = recursive2_proofs.read().unwrap().clone();
        for (airgroup, data) in recursive2_proofs_data.iter_mut().enumerate().take(n_airgroups) {
            if data.len() == 2 {
                let setup = self.setups.get_setup(airgroup, 0, &ProofType::Recursive2);
                let publics_aggregation = 1 + 4 * self.pctx.global_info.agg_types[airgroup].len() + 10;
                let null_proof = Proof::new(
                    ProofType::Recursive2,
                    airgroup,
                    0,
                    None,
                    vec![0; setup.proof_size as usize + publics_aggregation],
                );
                let circom_witness =
                    gen_witness_aggregation(&self.pctx, &self.setups, &data[0], &data[1], &null_proof)?;
                let gen_const_tree = !self.pctx.options.preallocate;
                let recursive2_proof = generate_recursive_proof::<F>(
                    &self.pctx,
                    &self.setups,
                    &circom_witness,
                    &self.traces[0],
                    &self.prover_buffers[0],
                    &output_dir_path,
                    self.d_buffers.get_ptr(),
                    0,
                    gen_const_tree,
                    mpi_node_rank,
                );
                *data = vec![recursive2_proof];
            } else {
                *data = data.to_vec();
            }
        }

        timer_stop_and_log_info!(GENERATING_INNER_COMPRESSED_PROOFS);
        timer_start_info!(GENERATING_OUTER_COMPRESSED_PROOFS);
        let agg_recursive2_proof = aggregate_recursive2_proofs(
            &self.pctx,
            &self.setups,
            &recursive2_proofs_data,
            &self.traces[0],
            &self.prover_buffers[0],
            output_dir_path.clone(),
            self.d_buffers.get_ptr(),
            mpi_node_rank,
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
                &self.traces[0],
                &self.prover_buffers[0],
                output_dir_path.clone(),
                self.d_buffers.get_ptr(),
            )?;

            vadcop_final_proof = vadcop_proof_final.proof.clone();

            proof_id = Some(
                blake3::hash(unsafe {
                    std::slice::from_raw_parts(vadcop_final_proof.as_ptr() as *const u8, vadcop_final_proof.len() * 8)
                })
                .to_hex()
                .to_string(),
            );

            if self.pctx.options.final_snark {
                timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
                let recursivef_proof = generate_recursivef_proof(
                    &self.pctx,
                    &self.setups,
                    &vadcop_final_proof,
                    &self.traces[0],
                    &self.prover_buffers[0],
                    output_dir_path.clone(),
                )?;
                timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

                timer_start_info!(GENERATING_FFLONK_SNARK_PROOF);
                let _ = generate_fflonk_snark_proof(&self.pctx, recursivef_proof, output_dir_path.clone());
                timer_stop_and_log_info!(GENERATING_FFLONK_SNARK_PROOF);
            }
        }
        timer_stop_and_log_info!(GENERATING_COMPRESSED_PROOFS);
        timer_stop_and_log_info!(GENERATING_VADCOP_PROOF);

        if self.pctx.dctx_get_rank() == 0 && self.pctx.options.verify_proofs {
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
                log::info!(
                    "{}: ··· {}",
                    Self::MY_NAME,
                    "\u{2717} Vadcop Final proof was not verified".bright_red().bold()
                );
                return Err("Vadcop Final proof was not verified".into());
            } else {
                log::info!(
                    "{}:     {}",
                    Self::MY_NAME,
                    "\u{2713} Vadcop Final proof was verified".bright_green().bold()
                );
            }
        }

        Ok(proof_id)
    }

    fn prepare_gpu(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        setups_vadcop: Arc<SetupsVadcop<F>>,
    ) -> (usize, Arc<DeviceBuffer>) {
        let mpi_node_rank = pctx.dctx_get_node_rank() as u32;

        let free_memory_gpu = match cfg!(feature = "gpu") {
            true => check_gpu_memory_c(mpi_node_rank) as f64 * 0.98,
            false => 0.0,
        };

        let total_const_area = match pctx.options.preallocate {
            true => sctx.total_const_size as u64,
            false => 0,
        };

        let total_const_area_aggregation = match pctx.options.aggregation && pctx.options.preallocate {
            true => setups_vadcop.total_const_size as u64,
            false => 0,
        };

        let mut max_size_buffer = (free_memory_gpu / 8.0).floor() as u64;
        if pctx.options.preallocate {
            max_size_buffer -= sctx.total_const_size as u64;
            if pctx.options.aggregation {
                max_size_buffer -= setups_vadcop.total_const_size as u64;
            }
        }

        let max_sizes =
            MaxSizes { total_const_area, max_aux_trace_area: max_size_buffer, total_const_area_aggregation };

        let max_sizes_ptr = &max_sizes as *const MaxSizes as *mut c_void;
        let d_buffers = Arc::new(DeviceBuffer(gen_device_commit_buffers_c(max_sizes_ptr, mpi_node_rank)));

        let max_number_proofs = match cfg!(feature = "gpu") {
            true => {
                let max_number_proofs = (max_size_buffer as usize / sctx.max_prover_buffer_size) as usize;
                if max_number_proofs < 1 {
                    panic!("Not enough GPU memory to run the proof");
                }
                max_number_proofs
            }
            false => 1,
        };

        let max_size_const = match !pctx.options.preallocate {
            true => sctx.max_const_size as u64,
            false => 0,
        };

        let max_size_const_tree = match !pctx.options.preallocate {
            true => sctx.max_const_tree_size as u64,
            false => 0,
        };

        let max_size_const_aggregation = match pctx.options.aggregation && !pctx.options.preallocate {
            true => setups_vadcop.max_const_size as u64,
            false => 0,
        };

        let max_size_const_tree_aggregation = match pctx.options.aggregation && !pctx.options.preallocate {
            true => setups_vadcop.max_const_tree_size as u64,
            false => 0,
        };

        let max_proof_size = match pctx.options.aggregation {
            true => sctx.max_proof_size.max(setups_vadcop.max_proof_size) as u64,
            false => sctx.max_proof_size as u64,
        };

        set_max_size_thread_c(
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
            max_number_proofs as u64,
        );

        (max_number_proofs, d_buffers)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_proof_thread(
        proofs: Arc<RwLock<Vec<Proof<F>>>>,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        thread_id: usize,
        instance_id: usize,
        airgroup_id: usize,
        air_id: usize,
        output_dir_path: PathBuf,
        aux_trace: Arc<Vec<F>>,
        airgroup_values_air_instances: Arc<Mutex<Vec<Vec<F>>>>,
        gen_const_tree: bool,
        d_buffers: Arc<DeviceBuffer>,
    ) {
        timer_start_info!(GEN_PROOF);
        Self::initialize_air_instance(&pctx, &sctx, instance_id, false);

        let setup = sctx.get_setup(airgroup_id, air_id);
        let p_setup: *mut c_void = (&setup.p_setup).into();
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let air_instance_name = &pctx.global_info.airs[airgroup_id][air_id].name;

        let offset_const = get_const_offset_c(setup.p_setup.p_stark_info) as usize;

        let mut steps_params = pctx.get_air_instance_params(&sctx, instance_id, true);

        if cfg!(not(feature = "gpu")) {
            let offset = thread_id * (sctx.max_const_tree_size + sctx.max_const_size);
            let const_pols = &aux_trace[offset + offset_const..offset + offset_const + setup.const_pols_size];
            let const_tree = &aux_trace[offset..offset + setup.const_tree_size];
            if gen_const_tree {
                timer_start_info!(LOAD_CONSTANTS);
                load_const_pols(&setup.setup_path, setup.const_pols_size, const_pols);
                load_const_pols_tree(setup, const_tree);
                timer_stop_and_log_info!(LOAD_CONSTANTS);
            }
            steps_params.p_const_tree = const_tree.as_ptr() as *mut u8;
            steps_params.p_const_pols = const_pols.as_ptr() as *mut u8;
        }

        let p_steps_params: *mut u8 = (&steps_params).into();

        let output_file_path = output_dir_path.join(format!("proofs/{}_{}.json", air_instance_name, instance_id));

        let proof_file = match pctx.options.debug_info.save_proofs_to_file {
            true => output_file_path.to_string_lossy().into_owned(),
            false => String::from(""),
        };

        let mut proof: Vec<u64> = create_buffer_fast(setup.proof_size as usize);

        let const_pols_path = setup.setup_path.to_string_lossy().to_string() + ".const";
        let const_pols_tree_path = setup.setup_path.display().to_string() + ".consttree";

        gen_proof_c(
            p_setup,
            p_steps_params,
            pctx.get_global_challenge_ptr(),
            proof.as_mut_ptr(),
            &proof_file,
            thread_id as u64,
            airgroup_id as u64,
            air_id as u64,
            air_instance_id as u64,
            d_buffers.get_ptr(),
            gen_const_tree,
            &const_pols_path,
            &const_pols_tree_path,
            pctx.dctx_get_node_rank() as u32,
        );

        get_proof_c(
            p_setup,
            proof.as_mut_ptr(),
            &proof_file,
            thread_id as u64,
            airgroup_id as u64,
            air_id as u64,
            air_instance_id as u64,
            d_buffers.get_ptr(),
            pctx.dctx_get_node_rank() as u32,
        );

        let n_airgroup_values = setup
            .stark_info
            .airgroupvalues_map
            .as_ref()
            .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
            .unwrap_or(0);

        let airgroup_values: Vec<F> = proof[0..n_airgroup_values].to_vec().iter().map(|&x| F::from_u64(x)).collect();

        airgroup_values_air_instances.lock().unwrap()[pctx.dctx_get_instance_idx(instance_id)] = airgroup_values;

        proofs.write().unwrap()[pctx.dctx_get_instance_idx(instance_id)] =
            Proof::new(ProofType::Basic, airgroup_id, air_id, Some(instance_id), proof);
        timer_stop_and_log_info!(GEN_PROOF);
    }

    #[allow(clippy::type_complexity)]
    fn initialize_proofman(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(Arc<ProofCtx<F>>, Arc<SetupCtx<F>>, Arc<SetupsVadcop<F>>), Box<dyn std::error::Error>> {
        timer_start_info!(INITIALIZING_PROOFMAN);

        let mut pctx = ProofCtx::create_ctx(proving_key_path.clone(), custom_commits_fixed, options.clone());
        let sctx: Arc<SetupCtx<F>> = Arc::new(SetupCtx::new(
            &pctx.global_info,
            &ProofType::Basic,
            options.verify_constraints,
            options.preallocate,
        ));
        pctx.set_weights(&sctx);

        let pctx = Arc::new(pctx);
        check_tree_paths(&pctx, &sctx)?;

        Self::initialize_publics(&sctx, &pctx)?;

        let setups_vadcop = Arc::new(SetupsVadcop::new(
            &pctx.global_info,
            options.verify_constraints,
            options.aggregation,
            false,
            options.final_snark,
        ));

        check_tree_paths_vadcop(&pctx, &setups_vadcop)?;

        timer_stop_and_log_info!(INITIALIZING_PROOFMAN);

        Ok((pctx, sctx, setups_vadcop))
    }

    fn calculate_global_challenge(pctx: &ProofCtx<F>, values: Vec<u64>) {
        timer_start_info!(CALCULATE_GLOBAL_CHALLENGE);
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

        println!(
            "{}: Global challenge: [{}, {}, {}]",
            Self::MY_NAME,
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
                log::warn!(
                    "{}: Missing initialization {} at row {} of {} in instance {}",
                    Self::MY_NAME,
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
        info!("{}: Initializing publics custom_commits", Self::MY_NAME);
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
        pctx: Arc<ProofCtx<F>>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        aux_trace_contribution_ptr: *mut u8,
        thread_id: usize,
        d_buffers: Arc<DeviceBuffer>,
    ) -> Vec<u64> {
        let n_field_elements = 4;

        timer_start_info!(GET_CONTRIBUTION_AIR);
        let instances = pctx.dctx_get_instances();

        let (airgroup_id, air_id, all) = instances[instance_id];
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let air_values = pctx.get_air_instance_air_values(airgroup_id, air_id, air_instance_id).clone();

        let root = vec![F::ZERO; n_field_elements];
        commit_witness_c(
            3,
            setup.stark_info.stark_struct.n_bits,
            setup.stark_info.stark_struct.n_bits_ext,
            *setup.stark_info.map_sections_n.get("cm1").unwrap(),
            root.as_ptr() as *mut u8,
            pctx.get_air_instance_trace_ptr(instance_id),
            aux_trace_contribution_ptr,
            thread_id as u64,
            d_buffers.get_ptr(),
            pctx.dctx_get_node_rank() as u32,
        );

        if !all {
            let pctx_clone = pctx.clone();
            std::thread::spawn(move || {
                pctx_clone.free_instance(instance_id);
            });
        }

        get_commit_root_c(
            3,
            setup.stark_info.stark_struct.n_bits_ext,
            *setup.stark_info.map_sections_n.get("cm1").unwrap(),
            root.as_ptr() as *mut u8,
            thread_id as u64,
            d_buffers.get_ptr(),
            pctx.dctx_get_node_rank() as u32,
        );

        let mut value = vec![F::ZERO; 10];

        let n_airvalues = setup
            .stark_info
            .airvalues_map
            .as_ref()
            .map(|map| map.iter().filter(|entry| entry.stage == 1).count())
            .unwrap_or(0);

        let size = 2 * n_field_elements + n_airvalues;

        let mut values_hash = vec![F::ZERO; size];

        let verkey = pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Basic).display().to_string()
            + ".verkey.json";

        let mut file = File::open(&verkey).expect("Unable to open file");
        let mut json_str = String::new();
        file.read_to_string(&mut json_str).expect("Unable to read file");
        let vk: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse JSON");
        for j in 0..n_field_elements {
            values_hash[j] = F::from_u64(vk[j]);
            values_hash[j + n_field_elements] = root[j];
        }

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

        calculate_hash_c(value.as_mut_ptr() as *mut u8, values_hash.as_mut_ptr() as *mut u8, size as u64, 10);

        timer_stop_and_log_info!(GET_CONTRIBUTION_AIR);

        value.iter().map(|x| x.as_canonical_u64()).collect::<Vec<u64>>()
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
