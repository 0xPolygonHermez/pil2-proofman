use libloading::{Library, Symbol};
use curves::{EcGFp5, EcMasFp5, curve::EllipticCurve};
use fields::{ExtensionField, PrimeField64, GoldilocksQuinticExtension};
#[cfg(distributed)]
use mpi::environment::Universe;
use std::ops::Add;
use std::sync::atomic::AtomicUsize;
use proofman_common::{
    calculate_fixed_tree, configured_num_threads, skip_prover_instance, CurveType, DebugInfo, MemoryHandler, ParamsGPU,
    Proof, ProofCtx, ProofOptions, ProofType, SetupCtx, SetupsVadcop, VerboseMode,
};
use colored::Colorize;
use proofman_hints::aggregate_airgroupvals;
use proofman_starks_lib_c::{free_device_buffers_c, gen_device_buffers_c, get_num_gpus_c};
use proofman_starks_lib_c::{
    save_challenges_c, save_proof_values_c, save_publics_c, check_device_memory_c, gen_device_streams_c,
    get_stream_proofs_c, get_stream_proofs_non_blocking_c, register_proof_done_callback_c,
};
use rayon::prelude::*;
use crossbeam_channel::{bounded, unbounded, Sender, Receiver};
use std::fs;
use std::collections::HashMap;
use std::fs::File;
use std::fmt::Write as FmtWrite;
use std::io::Read;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, RwLock};

use csv::Writer;

use rand::{SeedableRng, seq::SliceRandom};
use rand::rngs::StdRng;

use proofman_starks_lib_c::{
    gen_proof_c, commit_witness_c, calculate_hash_c, load_custom_commit_c, calculate_impols_expressions_c,
    clear_proof_done_callback_c, launch_callback_c,
};

use std::{path::PathBuf, sync::Arc};

use transcript::FFITranscript;

use witness::{WitnessLibInitFn, WitnessLibrary, WitnessManager};
use crate::{check_tree_paths_vadcop, gen_recursive_proof_size, initialize_fixed_pols_tree};
use crate::{verify_basic_proof, verify_final_proof, verify_global_constraints_proof};
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

use serde::Serialize;

#[derive(Serialize)]
struct CsvInfo {
    version: String,
    airgroup_id: usize,
    air_id: usize,
    name: String,
    instance_count: usize,
    total_area: u64,
}

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
    memory_handler: Arc<MemoryHandler<F>>,
    n_gpus: u64,
}

impl<F: PrimeField64> Drop for ProofMan<F> {
    fn drop(&mut self) {
        free_device_buffers_c(self.d_buffers.get_ptr());
    }
}
impl<F: PrimeField64> ProofMan<F>
where
    GoldilocksQuinticExtension: ExtensionField<F>,
{
    pub fn get_rank(&self) -> Option<i32> {
        if self.pctx.dctx_get_n_processes() > 1 {
            Some(self.pctx.dctx_get_rank() as i32)
        } else {
            None
        }
    }

    pub fn set_barrier(&self) {
        self.pctx.dctx_barrier();
    }

    #[cfg(distributed)]
    pub fn check_setup(
        proving_key_path: PathBuf,
        aggregation: bool,
        final_snark: bool,
        verbose_mode: VerboseMode,
        mpi_universe: Option<Universe>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {proving_key_path:?}").into());
        }

        let pctx = ProofCtx::<F>::create_ctx(
            proving_key_path.clone(),
            HashMap::new(),
            aggregation,
            final_snark,
            verbose_mode,
            mpi_universe,
        );

        Self::check_setup_(&pctx, aggregation, final_snark)
    }

    #[cfg(not(distributed))]
    pub fn check_setup(
        proving_key_path: PathBuf,
        aggregation: bool,
        final_snark: bool,
        verbose_mode: VerboseMode,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {:?}", proving_key_path).into());
        }

        let pctx =
            ProofCtx::<F>::create_ctx(proving_key_path.clone(), HashMap::new(), aggregation, final_snark, verbose_mode);

        Self::check_setup_(&pctx, aggregation, final_snark)
    }

    pub fn check_setup_(
        pctx: &ProofCtx<F>,
        aggregation: bool,
        final_snark: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let setups_aggregation =
            Arc::new(SetupsVadcop::<F>::new(&pctx.global_info, false, aggregation, false, final_snark));

        let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false, false);

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

    pub fn execute(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        output_path: PathBuf,
        verbose_mode: VerboseMode,
    ) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, self.get_rank())?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);

        self.register_witness(&mut *witness_lib, library);

        self.execute_(output_path)
    }

    pub fn execute_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
        output_path: PathBuf,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.wcm.set_input_data_path(input_data_path);
        self.execute_(output_path)
    }

    pub fn execute_(&self, output_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(EXECUTE);

        if !self.wcm.is_init_witness() {
            println!("Witness computation dynamic library not initialized");
            return Err("Witness computation dynamic library not initialized".into());
        }

        self.pctx.dctx_reset();

        self.wcm.execute();

        self.pctx.dctx_assign_instances();
        self.pctx.dctx_close();
        timer_stop_and_log_info!(EXECUTE);

        print_summary_info(&self.pctx, &self.sctx);

        let mut air_info: HashMap<String, CsvInfo> = HashMap::new();

        let instances = self.pctx.dctx_get_instances();

        for (airgroup_id, air_group) in self.pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let air_name = self.pctx.global_info.airs[airgroup_id][air_id].clone().name;

                air_info.insert(
                    air_name.clone(),
                    CsvInfo {
                        version: env!("CARGO_PKG_VERSION").to_string(),
                        name: air_name,
                        airgroup_id,
                        air_id,
                        total_area: 0,
                        instance_count: 0,
                    },
                );
            }
        }
        for instance_info in instances.iter() {
            let airgroup_id = instance_info.airgroup_id;
            let air_id = instance_info.air_id;

            let air_name = self.pctx.global_info.airs[airgroup_id][air_id].clone().name;

            let setup = self.sctx.get_setup(airgroup_id, air_id);
            let n_bits = setup.stark_info.stark_struct.n_bits;
            let total_cols: u64 = setup
                .stark_info
                .map_sections_n
                .iter()
                .filter(|(key, _)| *key != "const")
                .map(|(_, value)| *value)
                .sum();
            let area = (1 << n_bits) * total_cols;

            air_info.entry(air_name.clone()).and_modify(|info| {
                info.total_area += area;
                info.instance_count += 1;
            });
        }

        let mut wtr = Writer::from_path(output_path)?;

        let mut total_area = 0;
        let mut total_instances = 0;

        for info in air_info.values() {
            total_area += info.total_area;
            total_instances += info.instance_count;
        }

        for (airgroup_id, air_group) in self.pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let air_name = self.pctx.global_info.airs[airgroup_id][air_id].clone().name;
                let info = air_info.get_mut(&air_name).unwrap();
                wtr.serialize(&info)?;
            }
        }

        #[derive(Serialize)]
        struct Summary {
            version: String,
            airgroup_id: Option<usize>,
            air_id: Option<usize>,
            name: String,
            total_instances: usize,
            total_area: u64,
        }

        wtr.serialize(Summary {
            version: env!("CARGO_PKG_VERSION").to_string(),
            name: "TOTAL".into(),
            airgroup_id: None,
            air_id: None,
            total_area,
            total_instances,
        })?;

        wtr.flush()?;

        Ok(())
    }

    pub fn compute_witness(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        debug_info: &DebugInfo,
        verbose_mode: VerboseMode,
    ) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, self.get_rank())?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);
        self.pctx.set_debug_info(debug_info.clone());

        self.register_witness(&mut *witness_lib, library);

        self.compute_witness_()
    }

    /// Computes only the witness without generating a proof neither verifying constraints.
    /// This is useful for debugging or benchmarking purposes.
    pub fn compute_witness_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
        debug_info: &DebugInfo,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.pctx.set_debug_info(debug_info.clone());
        self.wcm.set_input_data_path(input_data_path);
        self.compute_witness_()
    }

    pub fn compute_witness_(&self) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(EXECUTE);

        if !self.wcm.is_init_witness() {
            println!("Witness computation dynamic library not initialized");
            return Err("Witness computation dynamic library not initialized".into());
        }

        self.pctx.dctx_reset();

        self.wcm.execute();

        // create a vector of instances wc weights
        self.pctx.dctx_assign_instances();
        self.pctx.dctx_close();

        print_summary_info(&self.pctx, &self.sctx);

        timer_stop_and_log_info!(EXECUTE);

        let instances = self.pctx.dctx_get_instances();
        let instances_all: Vec<(usize, _)> =
            instances.iter().enumerate().filter(|(_, instance_info)| instance_info.all).collect();
        let my_instances = self.pctx.dctx_get_my_instances();

        let instances_mine_no_precalculate = my_instances
            .iter()
            .filter(|idx| !self.pctx.dctx_is_instance_all(**idx) && !self.pctx.dctx_instance_precalculate(**idx))
            .copied()
            .collect::<Vec<_>>();
        let mut instances_mine_precalculate_fast = my_instances
            .iter()
            .filter(|idx| !self.pctx.dctx_is_instance_all(**idx) && self.pctx.dctx_instance_precalculate_fast(**idx))
            .copied()
            .collect::<Vec<_>>();
        let instances_mine_precalculate_slow = my_instances
            .iter()
            .filter(|idx| !self.pctx.dctx_is_instance_all(**idx) && self.pctx.dctx_instance_precalculate_slow(**idx))
            .copied()
            .collect::<Vec<_>>();

        let mut rng = StdRng::seed_from_u64(self.pctx.dctx_get_rank() as u64);
        instances_mine_precalculate_fast.shuffle(&mut rng);

        let instances_mine = my_instances.len();
        let instances_mine_all = instances_all.iter().filter(|(id, _)| self.pctx.dctx_is_my_instance(*id)).count();

        let instances_mine_no_all = instances_mine - instances_mine_all;

        let max_witness_stored = self.gpu_params.max_witness_stored.min(instances_mine_no_all);

        let max_num_threads = configured_num_threads(self.pctx.dctx_get_node_n_processes());

        let (tx_threads, rx_threads) = bounded::<()>(max_num_threads);
        let (tx_witness, rx_witness) = bounded::<()>(instances_mine);

        for _ in 0..max_num_threads {
            tx_threads.send(()).unwrap();
        }

        let n_threads_witness = self.gpu_params.number_threads_pools_witness.max(max_num_threads / max_witness_stored);

        let mut handles = Vec::new();

        timer_start_info!(COMPUTE_WITNESS);
        timer_start_info!(CALCULATE_MAIN_WITNESS);
        for &instance_id in instances_mine_no_precalculate.iter() {
            let instances = instances.clone();
            let instance_info = &instances[instance_id];
            let (airgroup_id, air_id, all) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);
            if all {
                continue;
            }

            let tx_threads_clone: Sender<()> = tx_threads.clone();
            let tx_witness_clone = tx_witness.clone();
            let wcm = self.wcm.clone();
            let pctx = self.pctx.clone();

            for _ in 0..n_threads_witness {
                rx_threads.recv().unwrap();
            }

            let memory_handler_clone = self.memory_handler.clone();

            let handle = std::thread::spawn(move || {
                timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref());
                for _ in 0..n_threads_witness {
                    tx_threads_clone.send(()).unwrap();
                }
                let (is_shared_buffer, witness_buffer) = pctx.free_instance(instance_id);
                if is_shared_buffer {
                    memory_handler_clone.release_buffer(witness_buffer);
                }

                timer_stop_and_log_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                tx_witness_clone.send(()).unwrap();
            });
            handles.push(handle);
        }
        timer_stop_and_log_info!(CALCULATE_MAIN_WITNESS);

        timer_start_info!(PRE_CALCULATE_WITNESS_FAST);
        self.wcm.pre_calculate_witness(
            1,
            &instances_mine_precalculate_fast,
            max_num_threads / 2,
            self.memory_handler.as_ref(),
        );
        timer_stop_and_log_info!(PRE_CALCULATE_WITNESS_FAST);
        timer_start_info!(CALCULATE_FAST_WITNESS);
        for &instance_id in instances_mine_precalculate_fast.iter() {
            let instances = instances.clone();
            let instance_info = &instances[instance_id];
            let (airgroup_id, air_id, all) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);
            if all {
                continue;
            }

            let tx_threads_clone: Sender<()> = tx_threads.clone();
            let tx_witness_clone = tx_witness.clone();
            let wcm = self.wcm.clone();

            let pctx_clone = self.pctx.clone();

            for _ in 0..n_threads_witness {
                rx_threads.recv().unwrap();
            }

            let memory_handler_clone = self.memory_handler.clone();

            let handle = std::thread::spawn(move || {
                timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref());
                for _ in 0..n_threads_witness {
                    tx_threads_clone.send(()).unwrap();
                }
                let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance(instance_id);
                if is_shared_buffer {
                    memory_handler_clone.release_buffer(witness_buffer);
                }
                timer_stop_and_log_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                tx_witness_clone.send(()).unwrap();
            });
            handles.push(handle);
        }
        timer_stop_and_log_info!(CALCULATE_FAST_WITNESS);

        timer_start_info!(PRE_CALCULATE_WITNESS_SLOW);
        let memory_handler_clone = self.memory_handler.clone();
        self.wcm.pre_calculate_witness(
            1,
            &instances_mine_precalculate_slow,
            max_num_threads / 2,
            memory_handler_clone.as_ref(),
        );
        timer_stop_and_log_info!(PRE_CALCULATE_WITNESS_SLOW);

        timer_start_info!(CALCULATE_SLOW_WITNESS);
        for &instance_id in instances_mine_precalculate_slow.iter() {
            let instances = instances.clone();
            let instance_info = &instances[instance_id];
            let (airgroup_id, air_id, all) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);
            if all {
                continue;
            }

            let tx_threads_clone: Sender<()> = tx_threads.clone();
            let tx_witness_clone = tx_witness.clone();
            let wcm = self.wcm.clone();
            let pctx_clone = self.pctx.clone();

            for _ in 0..n_threads_witness {
                rx_threads.recv().unwrap();
            }

            let memory_handler_clone = self.memory_handler.clone();

            let handle = std::thread::spawn(move || {
                timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref());
                for _ in 0..n_threads_witness {
                    tx_threads_clone.send(()).unwrap();
                }
                let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance(instance_id);
                if is_shared_buffer {
                    memory_handler_clone.release_buffer(witness_buffer);
                }
                timer_stop_and_log_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                tx_witness_clone.send(()).unwrap();
            });
            handles.push(handle);
        }
        timer_stop_and_log_info!(CALCULATE_SLOW_WITNESS);

        // syncronize to the non-all witnesses being evaluated
        for _ in 0..instances_mine_no_all {
            rx_witness.recv().unwrap();
        }
        timer_stop_and_log_info!(COMPUTE_WITNESS);

        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn verify_proof_constraints(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        output_dir_path: PathBuf,
        debug_info: &DebugInfo,
        verbose_mode: VerboseMode,
        test_mode: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check witness_lib path exists
        if !witness_lib_path.exists() {
            return Err(format!("Witness computation dynamic library not found at path: {witness_lib_path:?}").into());
        }

        // Check input data path
        if let Some(ref input_data_path) = input_data_path {
            if !input_data_path.exists() {
                return Err(format!("Input data file not found at path: {input_data_path:?}").into());
            }
        }

        // Check public_inputs_path is a folder
        if let Some(ref publics_path) = public_inputs_path {
            if !publics_path.exists() {
                return Err(format!("Public inputs file not found at path: {publics_path:?}").into());
            }
        }

        if !output_dir_path.exists() {
            fs::create_dir_all(&output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {err:?}"))?;
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, self.get_rank())?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);

        self.register_witness(&mut *witness_lib, library);

        self._verify_proof_constraints(debug_info, test_mode)
    }

    pub fn verify_proof_constraints_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
        debug_info: &DebugInfo,
        test_mode: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.wcm.set_input_data_path(input_data_path);

        self._verify_proof_constraints(debug_info, test_mode)
    }

    fn _verify_proof_constraints(
        &self,
        debug_info: &DebugInfo,
        test_mode: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.pctx.dctx_reset();

        self.pctx.set_debug_info(debug_info.clone());

        if !self.wcm.is_init_witness() {
            return Err("Witness computation dynamic library not initialized".into());
        }

        timer_start_info!(EXECUTE);
        self.wcm.execute();
        timer_stop_and_log_info!(EXECUTE);

        // create a vector of instances wc weights
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

        let max_num_threads = configured_num_threads(self.pctx.dctx_get_node_n_processes());

        for &instance_id in my_instances.iter() {
            let instance_info = instances[instance_id];
            let (airgroup_id, air_id, all) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);
            let (skip, _) = skip_prover_instance(&self.pctx, instance_id);
            if all || skip {
                continue;
            }

            self.wcm.pre_calculate_witness(1, &[instance_id], max_num_threads, self.memory_handler.as_ref());
            self.wcm.calculate_witness(1, &[instance_id], max_num_threads, self.memory_handler.as_ref());

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            Self::verify_proof_constraints_stage(
                self.pctx.clone(),
                self.sctx.clone(),
                self.wcm.clone(),
                valid_constraints.clone(),
                airgroup_values_air_instances.clone(),
                instance_id,
                airgroup_id,
                air_id,
                self.pctx.dctx_find_air_instance_id(instance_id),
                debug_info,
                max_num_threads,
                self.memory_handler.clone(),
            );
        }

        self.pctx.dctx_barrier();

        let instances_all: Vec<(usize, _)> =
            instances.iter().enumerate().filter(|(_, instance_info)| instance_info.all).collect();

        timer_start_info!(CALCULATING_TABLES);
        for (instance_id, _) in instances_all.iter() {
            self.wcm.calculate_witness(1, &[*instance_id], max_num_threads, self.memory_handler.as_ref());
        }
        timer_stop_and_log_info!(CALCULATING_TABLES);

        for &(instance_id, instance_info) in instances_all.iter() {
            let (skip, _) = skip_prover_instance(&self.pctx, instance_id);

            if skip || !self.pctx.dctx_is_my_instance(instance_id) {
                continue;
            };

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            let (airgroup_id, air_id) = (instance_info.airgroup_id, instance_info.air_id);
            Self::verify_proof_constraints_stage(
                self.pctx.clone(),
                self.sctx.clone(),
                self.wcm.clone(),
                valid_constraints.clone(),
                airgroup_values_air_instances.clone(),
                instance_id,
                airgroup_id,
                air_id,
                self.pctx.dctx_find_air_instance_id(instance_id),
                debug_info,
                max_num_threads,
                self.memory_handler.clone(),
            );
        }

        self.wcm.end(debug_info);

        let check_global_constraints =
            debug_info.debug_instances.is_empty() || !debug_info.debug_global_instances.is_empty();

        if check_global_constraints && !test_mode {
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
        max_num_threads: usize,
        memory_handler: Arc<MemoryHandler<F>>,
    ) {
        Self::initialize_air_instance(&pctx, &sctx, instance_id, true);

        #[cfg(feature = "diagnostic")]
        {
            let invalid_initialization = Self::diagnostic_instance(&pctx, &sctx, instance_id);
            if invalid_initialization {
                panic!("Invalid initialization");
                // return Some(Err("Invalid initialization".into()));
            }
        }

        wcm.calculate_witness(2, &[instance_id], max_num_threads, memory_handler.as_ref());
        Self::calculate_im_pols(2, &sctx, &pctx, instance_id);

        wcm.debug(&[instance_id], debug_info);

        let valid = verify_constraints_proof(&pctx, &sctx, instance_id, debug_info.n_print_constraints as u64);
        if !valid {
            valid_constraints.fetch_and(valid, Ordering::Relaxed);
        }

        let airgroup_values = pctx.get_air_instance_airgroup_values(airgroup_id, air_id, air_instance_id);
        airgroup_values_air_instances.lock().unwrap()[pctx.dctx_get_instance_idx(instance_id)] = airgroup_values;
        let (is_shared_buffer, witness_buffer) = pctx.free_instance(instance_id);
        if is_shared_buffer {
            memory_handler.release_buffer(witness_buffer);
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_proof(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        verbose_mode: VerboseMode,
        options: ProofOptions,
    ) -> Result<(Option<String>, Option<Vec<u64>>), Box<dyn std::error::Error>> {
        // Check witness_lib path exists
        if !witness_lib_path.exists() {
            return Err(format!("Witness computation dynamic library not found at path: {witness_lib_path:?}").into());
        }

        // Check input data path
        if let Some(ref input_data_path) = input_data_path {
            if !input_data_path.exists() {
                return Err(format!("Input data file not found at path: {input_data_path:?}").into());
            }
        }

        // Check public_inputs_path is a folder
        if let Some(ref publics_path) = public_inputs_path {
            if !publics_path.exists() {
                return Err(format!("Public inputs file not found at path: {publics_path:?}").into());
            }
        }

        if !options.output_dir_path.exists() {
            fs::create_dir_all(&options.output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {err:?}"))?;
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, self.get_rank())?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);

        self.register_witness(&mut *witness_lib, library);

        self._generate_proof(options)
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_proof_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
        options: ProofOptions,
    ) -> Result<(Option<String>, Option<Vec<u64>>), Box<dyn std::error::Error>> {
        if !options.output_dir_path.exists() {
            fs::create_dir_all(&options.output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {err:?}"))?;
        }

        self.wcm.set_input_data_path(input_data_path);
        self._generate_proof(options)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(distributed)]
    pub fn new(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        verify_constraints: bool,
        aggregation: bool,
        final_snark: bool,
        gpu_params: ParamsGPU,
        verbose_mode: VerboseMode,
        mpi_universe: Option<Universe>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {proving_key_path:?}").into());
        }

        // Check proving_key_path is a folder
        if !proving_key_path.is_dir() {
            return Err(format!("Proving key parameter must be a folder: {proving_key_path:?}").into());
        }

        let (pctx, sctx, setups_vadcop) = Self::initialize_proofman(
            proving_key_path,
            custom_commits_fixed,
            verify_constraints,
            aggregation,
            final_snark,
            &gpu_params,
            verbose_mode,
            mpi_universe,
        )?;

        timer_start_info!(INIT_PROOFMAN);

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

        let max_witness_stored = match cfg!(feature = "gpu") {
            true => gpu_params.max_witness_stored,
            false => 1,
        };

        let memory_handler = Arc::new(MemoryHandler::new(max_witness_stored, sctx.max_witness_trace_size));

        pctx.dctx_barrier();

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
            memory_handler,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(not(distributed))]
    pub fn new(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        verify_constraints: bool,
        aggregation: bool,
        final_snark: bool,
        gpu_params: ParamsGPU,
        verbose_mode: VerboseMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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
            verbose_mode,
        )?;

        timer_start_info!(INIT_PROOFMAN);

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

        let max_witness_stored = match cfg!(feature = "gpu") {
            true => gpu_params.max_witness_stored,
            false => 1,
        };

        let memory_handler = Arc::new(MemoryHandler::new(max_witness_stored, sctx.max_witness_trace_size));

        pctx.dctx_barrier();

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
            memory_handler,
        })
    }

    pub fn register_witness(&self, witness_lib: &mut dyn WitnessLibrary<F>, library: Library) {
        timer_start_info!(REGISTERING_WITNESS);
        witness_lib.register_witness(self.wcm.clone());
        self.wcm.set_init_witness(true, library);
        timer_stop_and_log_info!(REGISTERING_WITNESS);
        self.pctx.dctx_barrier();
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn _generate_proof(
        &self,
        options: ProofOptions,
    ) -> Result<(Option<String>, Option<Vec<u64>>), Box<dyn std::error::Error>> {
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

        if !options.test_mode {
            Self::initialize_publics_custom_commits(&self.sctx, &self.pctx)?;
        }

        self.wcm.execute();

        self.pctx.dctx_assign_instances();
        self.pctx.dctx_close();

        print_summary_info(&self.pctx, &self.sctx);

        timer_stop_and_log_info!(EXECUTE);

        timer_start_info!(CALCULATING_CONTRIBUTIONS);
        timer_start_info!(CALCULATING_INNER_CONTRIBUTIONS);
        timer_start_info!(PREPARING_CONTRIBUTIONS);
        let mut rng = StdRng::seed_from_u64(self.pctx.dctx_get_rank() as u64);

        let instances = self.pctx.dctx_get_instances();
        let instances_all: Vec<(usize, _)> =
            instances.iter().enumerate().filter(|(_, &instance_info)| instance_info.all).collect();
        let my_instances = self.pctx.dctx_get_my_instances();

        let instances_mine_no_precalculate = my_instances
            .iter()
            .filter(|idx| !self.pctx.dctx_is_instance_all(**idx) && !self.pctx.dctx_instance_precalculate(**idx))
            .copied()
            .collect::<Vec<_>>();
        let mut instances_mine_precalculate_fast = my_instances
            .iter()
            .filter(|idx| !self.pctx.dctx_is_instance_all(**idx) && self.pctx.dctx_instance_precalculate_fast(**idx))
            .copied()
            .collect::<Vec<_>>();
        let instances_mine_precalculate_slow = my_instances
            .iter()
            .filter(|idx| !self.pctx.dctx_is_instance_all(**idx) && self.pctx.dctx_instance_precalculate_slow(**idx))
            .copied()
            .collect::<Vec<_>>();

        instances_mine_precalculate_fast.shuffle(&mut rng);
        let mut my_instances_sorted = my_instances.clone();
        my_instances_sorted.shuffle(&mut rng);

        let instances_mine = my_instances.len();
        let instances_mine_all = instances_all.iter().filter(|(id, _)| self.pctx.dctx_is_my_instance(*id)).count();
        let instances_mine_no_all = instances_mine - instances_mine_all;

        let values_contributions: Arc<Vec<Mutex<Vec<F>>>> =
            Arc::new((0..instances.len()).map(|_| Mutex::new(Vec::<F>::new())).collect());

        let roots_contributions: Arc<Vec<Mutex<[F; 4]>>> =
            Arc::new((0..instances.len()).map(|_| Mutex::new([F::default(); 4])).collect());

        let aux_trace_size = match cfg!(feature = "gpu") {
            true => 0,
            false => self.sctx.max_prover_buffer_size.max(self.setups.max_prover_buffer_size),
        };
        let const_pols_size = match cfg!(feature = "gpu") {
            true => 0,
            false => self.sctx.max_const_size.max(self.setups.max_const_size),
        };
        let const_tree_size = match cfg!(feature = "gpu") {
            true => 0,
            false => self.sctx.max_const_tree_size.max(self.setups.max_const_tree_size),
        };
        let aux_trace: Arc<Vec<F>> = Arc::new(create_buffer_fast(aux_trace_size));
        let const_pols: Arc<Vec<F>> = Arc::new(create_buffer_fast(const_pols_size));
        let const_tree: Arc<Vec<F>> = Arc::new(create_buffer_fast(const_tree_size));

        let max_witness_stored = match cfg!(feature = "gpu") {
            true => instances_mine_no_all.min(self.gpu_params.max_witness_stored),
            false => 1,
        };

        let max_num_threads = configured_num_threads(self.pctx.dctx_get_node_n_processes());
        let n_proof_threads = match cfg!(feature = "gpu") {
            true => self.n_gpus,
            false => 1,
        };

        let n_streams = self.n_streams_per_gpu * n_proof_threads;
        let streams = Arc::new(Mutex::new(vec![None; n_streams as usize]));

        // define managment channels and counters
        let (tx_threads, rx_threads) = bounded::<()>(max_num_threads);
        let (tx_witness, rx_witness) = bounded::<()>(instances_mine);

        for _ in 0..max_num_threads {
            tx_threads.send(()).unwrap();
        }

        let witnesses_done = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        timer_stop_and_log_info!(PREPARING_CONTRIBUTIONS);

        let n_threads_witness = match cfg!(feature = "gpu") {
            true => self.gpu_params.number_threads_pools_witness.max(max_num_threads / max_witness_stored),
            false => max_num_threads,
        };

        let stop_watch = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_watch.clone();
        let d_buffers_clone = self.d_buffers.clone();
        let watch_contributions = std::thread::spawn(move || {
            while !stop_flag_clone.load(Ordering::Relaxed) {
                get_stream_proofs_non_blocking_c(d_buffers_clone.get_ptr());
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        });

        timer_start_info!(CALCULATING_WITNESS);
        timer_start_info!(CALCULATE_MAIN_WITNESS);
        for &instance_id in instances_mine_no_precalculate.iter() {
            let instances = instances.clone();
            let instance_info = instances[instance_id];
            let (airgroup_id, air_id, all) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);
            if all {
                continue;
            }

            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let values_contributions_clone = values_contributions.clone();
            let roots_contributions_clone = roots_contributions.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let aux_trace_clone = aux_trace.clone();
            let streams_clone = streams.clone();
            let tx_threads_clone: Sender<()> = tx_threads.clone();
            let tx_witness_clone = tx_witness.clone();
            let wcm = self.wcm.clone();
            let witnesses_done_clone = witnesses_done.clone();

            for _ in 0..n_threads_witness {
                rx_threads.recv().unwrap();
            }

            let memory_handler_clone = self.memory_handler.clone();

            let handle = std::thread::spawn(move || {
                timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref());
                for _ in 0..n_threads_witness {
                    tx_threads_clone.send(()).unwrap();
                }

                timer_stop_and_log_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                tx_witness_clone.send(()).unwrap();

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

                witnesses_done_clone.fetch_add(1, Ordering::AcqRel);
                if (instances_mine_no_all - witnesses_done_clone.load(Ordering::Acquire)) >= max_witness_stored {
                    let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance_traces(instance_id);
                    if is_shared_buffer {
                        memory_handler_clone.release_buffer(witness_buffer);
                    }
                }
            });
            if cfg!(not(feature = "gpu")) {
                handle.join().unwrap();
            } else {
                handles.push(handle);
            }
        }
        timer_stop_and_log_info!(CALCULATE_MAIN_WITNESS);
        timer_start_info!(PRE_CALCULATE_WITNESS_FAST);
        self.wcm.pre_calculate_witness(
            1,
            &instances_mine_precalculate_fast,
            max_num_threads / 2,
            self.memory_handler.as_ref(),
        );
        timer_stop_and_log_info!(PRE_CALCULATE_WITNESS_FAST);
        timer_start_info!(CALCULATE_FAST_WITNESS);
        for &instance_id in instances_mine_precalculate_fast.iter() {
            let instances = instances.clone();
            let instance_info = instances[instance_id];
            let (airgroup_id, air_id, all) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);
            if all {
                continue;
            }

            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let values_contributions_clone = values_contributions.clone();
            let roots_contributions_clone = roots_contributions.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let aux_trace_clone = aux_trace.clone();
            let streams_clone = streams.clone();
            let tx_threads_clone: Sender<()> = tx_threads.clone();
            let tx_witness_clone = tx_witness.clone();
            let wcm = self.wcm.clone();
            let witnesses_done_clone = witnesses_done.clone();

            for _ in 0..n_threads_witness {
                rx_threads.recv().unwrap();
            }

            let memory_handler_clone = self.memory_handler.clone();

            let handle = std::thread::spawn(move || {
                timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref());
                for _ in 0..n_threads_witness {
                    tx_threads_clone.send(()).unwrap();
                }
                timer_stop_and_log_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                tx_witness_clone.send(()).unwrap();

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

                witnesses_done_clone.fetch_add(1, Ordering::AcqRel);
                if (instances_mine_no_all - witnesses_done_clone.load(Ordering::Acquire)) >= max_witness_stored {
                    let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance_traces(instance_id);
                    if is_shared_buffer {
                        memory_handler_clone.release_buffer(witness_buffer);
                    }
                }
            });
            if cfg!(not(feature = "gpu")) {
                handle.join().unwrap();
            } else {
                handles.push(handle);
            }
        }
        timer_stop_and_log_info!(CALCULATE_FAST_WITNESS);

        timer_start_info!(PRE_CALCULATE_WITNESS_SLOW);
        self.wcm.pre_calculate_witness(
            1,
            &instances_mine_precalculate_slow,
            max_num_threads / 2,
            self.memory_handler.as_ref(),
        );
        timer_stop_and_log_info!(PRE_CALCULATE_WITNESS_SLOW);

        timer_start_info!(CALCULATE_SLOW_WITNESS);
        for &instance_id in instances_mine_precalculate_slow.iter() {
            let instances = instances.clone();
            let instance_info = instances[instance_id];
            let (airgroup_id, air_id, all) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);
            if all {
                continue;
            }

            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let values_contributions_clone = values_contributions.clone();
            let roots_contributions_clone = roots_contributions.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let aux_trace_clone = aux_trace.clone();
            let streams_clone = streams.clone();
            let tx_threads_clone: Sender<()> = tx_threads.clone();
            let tx_witness_clone = tx_witness.clone();
            let wcm = self.wcm.clone();
            let witnesses_done_clone = witnesses_done.clone();

            for _ in 0..n_threads_witness {
                rx_threads.recv().unwrap();
            }

            let memory_handler_clone = self.memory_handler.clone();

            let handle = std::thread::spawn(move || {
                timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref());
                for _ in 0..n_threads_witness {
                    tx_threads_clone.send(()).unwrap();
                }
                timer_stop_and_log_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                tx_witness_clone.send(()).unwrap();

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

                witnesses_done_clone.fetch_add(1, Ordering::AcqRel);
                if (instances_mine_no_all - witnesses_done_clone.load(Ordering::Acquire)) >= max_witness_stored {
                    let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance_traces(instance_id);
                    if is_shared_buffer {
                        memory_handler_clone.release_buffer(witness_buffer);
                    }
                }
            });
            if cfg!(not(feature = "gpu")) {
                handle.join().unwrap();
            } else {
                handles.push(handle);
            }
        }
        timer_stop_and_log_info!(CALCULATE_SLOW_WITNESS);

        stop_watch.store(true, Ordering::Relaxed);
        watch_contributions.join().unwrap();

        // syncronize to the non-all witnesses being evaluated
        for _ in 0..instances_mine_no_all {
            rx_witness.recv().unwrap();
        }

        timer_stop_and_log_info!(CALCULATING_WITNESS);

        timer_start_info!(TIME_WAIT);
        self.pctx.dctx_barrier();
        timer_stop_and_log_info!(TIME_WAIT);

        timer_start_info!(CALCULATING_TABLES);

        //evalutate witness for instances of type "all"
        for (instance_id, _) in instances_all.iter() {
            self.wcm.pre_calculate_witness(1, &[*instance_id], max_num_threads, self.memory_handler.as_ref());
            self.wcm.calculate_witness(1, &[*instance_id], max_num_threads, self.memory_handler.as_ref());
        }

        timer_stop_and_log_info!(CALCULATING_TABLES);

        for (instance_id, _) in instances_all.iter() {
            if self.pctx.dctx_is_my_instance(*instance_id) {
                Self::get_contribution_air(
                    &self.pctx,
                    &self.sctx,
                    roots_contributions.clone(),
                    values_contributions.clone(),
                    *instance_id,
                    aux_trace.clone().as_ptr() as *mut u8,
                    self.d_buffers.clone(),
                    streams.clone(),
                );
            }
        }

        // ensure all threads have finishes, this ensures all contributions have been launched
        if cfg!(feature = "gpu") {
            for handle in handles {
                handle.join().unwrap();
            }
        }

        // get roots still in the streams
        get_stream_proofs_c(self.d_buffers.get_ptr());

        timer_stop_and_log_info!(CALCULATING_INNER_CONTRIBUTIONS);

        //calculate-challenge
        Self::calculate_global_challenge(&self.pctx, roots_contributions, values_contributions);

        timer_stop_and_log_info!(CALCULATING_CONTRIBUTIONS);

        timer_start_info!(GENERATING_INNER_PROOFS);

        let n_airgroups = self.pctx.global_info.air_groups.len();

        let mut proofs: Vec<RwLock<Proof<F>>> = Vec::new();
        let compressor_proofs: Arc<Vec<RwLock<Option<Proof<F>>>>> =
            Arc::new((0..instances.len()).map(|_| RwLock::new(None)).collect());
        let recursive1_proofs: Arc<Vec<RwLock<Option<Proof<F>>>>> =
            Arc::new((0..instances.len()).map(|_| RwLock::new(None)).collect());
        let recursive2_proofs: Arc<Vec<RwLock<Vec<Proof<F>>>>> =
            Arc::new((0..n_airgroups).map(|_| RwLock::new(Vec::new())).collect());
        let recursive2_proofs_ongoing: Arc<RwLock<Vec<Option<Proof<F>>>>> = Arc::new(RwLock::new(Vec::new()));

        let vec_streams: Vec<Option<u64>> = {
            let mut guard = streams.lock().unwrap();
            std::mem::take(&mut *guard)
        };

        let mut n_airgroup_proofs = vec![0; n_airgroups];
        for (instance_id, instance_info) in instances.iter().enumerate() {
            let (airgroup_id, air_id) = (instance_info.airgroup_id, instance_info.air_id);
            if self.pctx.dctx_is_my_instance(instance_id) {
                n_airgroup_proofs[airgroup_id] += 1;
            }
            let setup = self.sctx.get_setup(airgroup_id, air_id);
            let proof = create_buffer_fast(setup.proof_size as usize);
            proofs.push(RwLock::new(Proof::new(ProofType::Basic, airgroup_id, air_id, Some(instance_id), proof)));
        }

        let proofs = Arc::new(proofs);

        if options.aggregation {
            for (airgroup, &n_proofs) in n_airgroup_proofs.iter().enumerate().take(n_airgroups) {
                let n_recursive2_proofs = total_recursive_proofs(n_proofs);
                if n_recursive2_proofs.has_remaining {
                    let setup = self.setups.get_setup(airgroup, 0, &ProofType::Recursive2);
                    let publics_aggregation = 1 + 4 * self.pctx.global_info.agg_types[airgroup].len() + 10;
                    let null_proof_buffer = vec![0; setup.proof_size as usize + publics_aggregation];
                    let null_proof = Proof::new(ProofType::Recursive2, airgroup, 0, None, null_proof_buffer);
                    recursive2_proofs[airgroup].write().unwrap().push(null_proof);
                }
            }
        }
        let recursive2_proofs = Arc::new(recursive2_proofs);

        let proofs_pending = Arc::new(Counter::new());

        let basic_proofs_threshold = instances_mine.saturating_sub(n_streams as usize).max(0);
        let basic_proofs_done = Arc::new(Counter::new_with_threshold(basic_proofs_threshold));

        let (recursive_tx, recursive_rx) = unbounded::<(u64, String)>();
        register_proof_done_callback_c(recursive_tx.clone());

        let (compressor_witness_tx, compressor_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();
        let (rec1_witness_tx, rec1_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();
        let (rec2_witness_tx, rec2_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();

        let mut handle_recursives = Vec::new();
        for _ in 0..n_streams {
            let pctx_clone = self.pctx.clone();
            let setups_clone = self.setups.clone();
            let proofs_clone = proofs.clone();
            let compressor_proofs_clone = compressor_proofs.clone();
            let recursive1_proofs_clone = recursive1_proofs.clone();
            let recursive2_proofs_clone = recursive2_proofs.clone();
            let recursive2_proofs_ongoing_clone = recursive2_proofs_ongoing.clone();
            let proofs_pending_clone = proofs_pending.clone();
            let basic_proofs_done_clone = basic_proofs_done.clone();
            let rec1_witness_tx_clone = rec1_witness_tx.clone();
            let rec2_witness_tx_clone = rec2_witness_tx.clone();
            let compressor_witness_tx_clone = compressor_witness_tx.clone();
            let recursive_rx_clone = recursive_rx.clone();
            let handle_recursive = std::thread::spawn(move || {
                while let Ok((id, proof_type)) = recursive_rx_clone.recv() {
                    if id == u64::MAX - 1 {
                        return;
                    }
                    let p: ProofType = proof_type.parse().unwrap();
                    if p == ProofType::Basic {
                        basic_proofs_done_clone.increment();
                    }
                    if !options.aggregation {
                        proofs_pending_clone.decrement();
                        continue;
                    }

                    let new_proof_type = if p == ProofType::Basic {
                        let (airgroup_id, air_id) = pctx_clone.dctx_get_instance_info(id as usize);
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
                            recursive1_proofs_clone[id as usize].read().unwrap().as_ref().unwrap().clone()
                        } else {
                            recursive2_proofs_ongoing_clone.read().unwrap()[id as usize].as_ref().unwrap().clone()
                        };

                        let recursive2_proof = {
                            let mut recursive2_airgroup_proofs =
                                recursive2_proofs_clone[proof.airgroup_id].write().unwrap();
                            recursive2_airgroup_proofs.push(proof);

                            if recursive2_airgroup_proofs.len() >= 3 {
                                let p1 = recursive2_airgroup_proofs.pop().unwrap();
                                let p2 = recursive2_airgroup_proofs.pop().unwrap();
                                let p3 = recursive2_airgroup_proofs.pop().unwrap();
                                Some((p1, p2, p3))
                            } else {
                                None
                            }
                        };

                        recursive2_proof.map(|(p1, p2, p3)| {
                            gen_witness_aggregation(&pctx_clone, &setups_clone, &p1, &p2, &p3).unwrap()
                        })
                    } else if new_proof_type == ProofType::Recursive1 as usize && p == ProofType::Compressor {
                        let guard = compressor_proofs_clone[id as usize].read().unwrap();
                        let proof = guard.as_ref().unwrap();
                        Some(gen_witness_recursive(&pctx_clone, &setups_clone, proof).unwrap())
                    } else {
                        let proof = proofs_clone[id as usize].read().unwrap();
                        Some(gen_witness_recursive(&pctx_clone, &setups_clone, &proof).unwrap())
                    };

                    if let Some(witness) = witness {
                        proofs_pending_clone.increment();
                        if new_proof_type == ProofType::Compressor as usize {
                            compressor_witness_tx_clone.send(witness).unwrap();
                        } else if new_proof_type == ProofType::Recursive1 as usize {
                            rec1_witness_tx_clone.send(witness).unwrap();
                        } else {
                            rec2_witness_tx_clone.send(witness).unwrap();
                        }
                    }
                    proofs_pending_clone.decrement();
                }
            });
            handle_recursives.push(handle_recursive);
        }

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
                        self.gpu_params.preallocate,
                    );
                    let (is_shared_buffer, witness_buffer) = self.pctx.free_instance(instance_id as usize);
                    if is_shared_buffer {
                        self.memory_handler.release_buffer(witness_buffer);
                    }
                    processed_ids.lock().unwrap().push(instance_id);
                });
        }

        let mut my_instances_calculated = vec![false; instances.len()];
        for idx in processed_ids.into_inner().unwrap() {
            my_instances_calculated[idx as usize] = true;
        }

        let mut precalculate_instances = Vec::new();
        for &instance_id in my_instances_sorted.iter() {
            if my_instances_calculated[instance_id] {
                continue;
            }

            let is_stored =
                self.pctx.is_air_instance_stored(instance_id) || vec_streams.contains(&Some(instance_id as u64));

            if !is_stored && self.pctx.dctx_instance_precalculate(instance_id) {
                precalculate_instances.push(instance_id);
            }
        }

        timer_start_info!(PRECALCULATE_WITNESS);
        self.wcm.pre_calculate_witness(1, &precalculate_instances, max_num_threads / 2, self.memory_handler.as_ref());
        timer_stop_and_log_info!(PRECALCULATE_WITNESS);

        my_instances_sorted.sort_by_key(|&id| {
            let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(id);
            (
                if self.pctx.is_air_instance_stored(id) { 0 } else { 1 },
                if self.pctx.global_info.get_air_has_compressor(airgroup_id, air_id) { 0 } else { 1 },
            )
        });

        for &instance_id in my_instances_sorted.iter() {
            if my_instances_calculated[instance_id] {
                continue;
            }

            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let aux_trace_clone = aux_trace.clone();
            let tx_threads_clone: Sender<()> = tx_threads.clone();
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

            let instance_info = &instances[instance_id];
            let (airgroup_id, air_id, _) = (instance_info.airgroup_id, instance_info.air_id, instance_info.all);

            let preallocate = self.gpu_params.preallocate;

            if !is_stored {
                for _ in 0..n_threads_witness {
                    rx_threads.recv().unwrap();
                }
            }

            let memory_handler_clone = self.memory_handler.clone();

            let handle = std::thread::spawn(move || {
                proofs_pending_clone.increment();
                if !is_stored {
                    timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref());
                    for _ in 0..n_threads_witness {
                        tx_threads_clone.send(()).unwrap();
                    }
                    timer_stop_and_log_info!(
                        GENERATING_WC,
                        "GENERATING_WC_{} [{}:{}]",
                        instance_id,
                        airgroup_id,
                        air_id
                    );
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
                    preallocate,
                );
                let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance(instance_id);
                if is_shared_buffer {
                    memory_handler_clone.release_buffer(witness_buffer);
                }
            });
            if cfg!(not(feature = "gpu")) {
                handle.join().unwrap();
            }
        }

        for &instance_id in my_instances_sorted.iter() {
            if cfg!(not(feature = "gpu")) {
                launch_callback_c(instance_id as u64, "basic");
            }
        }

        basic_proofs_done
            .wait_until_threshold_and_check_streams(|| get_stream_proofs_non_blocking_c(self.d_buffers.get_ptr()));

        let proofs_finished = Arc::new(AtomicBool::new(false));
        for _ in 0..n_streams {
            let pctx_clone = self.pctx.clone();
            let setups_clone = self.setups.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let trace_size = self.trace_size;
            let prover_buffer_size = self.prover_buffer_size;
            let output_dir_path_clone = options.output_dir_path.clone();
            let const_pols_clone = const_pols.clone();
            let const_tree_clone = const_tree.clone();
            let compressor_proofs_clone = compressor_proofs.clone();
            let recursive1_proofs_clone = recursive1_proofs.clone();
            let recursive2_proofs_ongoing_clone = recursive2_proofs_ongoing.clone();

            let compressor_rx = compressor_witness_rx.clone();
            let rec2_rx = rec2_witness_rx.clone();
            let rec1_rx = rec1_witness_rx.clone();

            let proofs_finished_clone = proofs_finished.clone();

            let handle_recursive = std::thread::spawn(move || loop {
                let witness = rec2_rx.try_recv().or_else(|_| compressor_rx.try_recv()).or_else(|_| rec1_rx.try_recv());

                let mut witness = match witness {
                    Ok(w) => w,
                    Err(_) => {
                        if proofs_finished_clone.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        continue;
                    }
                };

                let trace: Vec<F> = create_buffer_fast(trace_size);
                let prover_buffer: Vec<F> = create_buffer_fast(prover_buffer_size);
                if witness.proof_type == ProofType::Recursive2 {
                    let id = {
                        let mut proofs = recursive2_proofs_ongoing_clone.write().unwrap();
                        let id = proofs.len();
                        proofs.push(None);
                        id
                    };

                    witness.global_idx = Some(id);
                }

                let new_proof = gen_recursive_proof_size(&pctx_clone, &setups_clone, &witness);

                let new_proof_type = new_proof.proof_type.clone();
                let new_proof_type_str: &str = new_proof_type.clone().into();

                let id = new_proof.global_idx.unwrap();
                if new_proof_type == ProofType::Recursive2 {
                    recursive2_proofs_ongoing_clone.write().unwrap()[id] = Some(new_proof);
                } else if new_proof_type == ProofType::Compressor {
                    *compressor_proofs_clone[id].write().unwrap() = Some(new_proof);
                } else if new_proof_type == ProofType::Recursive1 {
                    *recursive1_proofs_clone[id].write().unwrap() = Some(new_proof);
                }

                if new_proof_type == ProofType::Recursive2 {
                    let recursive2_lock = recursive2_proofs_ongoing_clone.read().unwrap();
                    let new_proof_ref = recursive2_lock[id].as_ref().unwrap();
                    let _ = generate_recursive_proof(
                        &pctx_clone,
                        &setups_clone,
                        &witness,
                        new_proof_ref,
                        &trace,
                        &prover_buffer,
                        &output_dir_path_clone,
                        d_buffers_clone.get_ptr(),
                        const_tree_clone.clone(),
                        const_pols_clone.clone(),
                        options.save_proofs,
                    );
                } else if new_proof_type == ProofType::Compressor {
                    let compressor_lock = compressor_proofs_clone[id].read().unwrap();
                    let new_proof_ref = compressor_lock.as_ref().unwrap();
                    let _ = generate_recursive_proof(
                        &pctx_clone,
                        &setups_clone,
                        &witness,
                        new_proof_ref,
                        &trace,
                        &prover_buffer,
                        &output_dir_path_clone,
                        d_buffers_clone.get_ptr(),
                        const_tree_clone.clone(),
                        const_pols_clone.clone(),
                        options.save_proofs,
                    );
                } else {
                    let recursive1_lock = recursive1_proofs_clone[id].read().unwrap();
                    let new_proof_ref = recursive1_lock.as_ref().unwrap();
                    let _ = generate_recursive_proof(
                        &pctx_clone,
                        &setups_clone,
                        &witness,
                        new_proof_ref,
                        &trace,
                        &prover_buffer,
                        &output_dir_path_clone,
                        d_buffers_clone.get_ptr(),
                        const_tree_clone.clone(),
                        const_pols_clone.clone(),
                        options.save_proofs,
                    );
                }

                if cfg!(not(feature = "gpu")) {
                    launch_callback_c(id as u64, new_proof_type_str);
                }
            });
            handle_recursives.push(handle_recursive);
        }

        proofs_pending.wait_until_zero_and_check_streams(|| get_stream_proofs_non_blocking_c(self.d_buffers.get_ptr()));

        get_stream_proofs_c(self.d_buffers.get_ptr());

        proofs_finished.store(true, Ordering::Relaxed);
        clear_proof_done_callback_c();
        recursive_tx.send((u64::MAX - 1, "Basic".to_string())).unwrap();
        drop(recursive_tx);
        drop(rec2_witness_tx);
        drop(compressor_witness_tx);
        drop(rec1_witness_tx);
        for handle in handle_recursives {
            handle.join().unwrap();
        }

        timer_stop_and_log_info!(GENERATING_INNER_PROOFS);

        timer_stop_and_log_info!(GENERATING_PROOFS);

        if options.save_proofs {
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
        }

        if !options.aggregation {
            let mut valid_proofs = true;

            if options.verify_proofs {
                timer_start_info!(VERIFYING_PROOFS);
                for instance_id in my_instances.iter() {
                    let valid_proof = verify_basic_proof(
                        &self.pctx,
                        *instance_id,
                        &proofs[*instance_id].read().unwrap().proof.clone(),
                    );
                    if !valid_proof {
                        valid_proofs = false;
                    }
                }
                timer_stop_and_log_info!(VERIFYING_PROOFS);

                let mut airgroup_values_air_instances = vec![Vec::new(); my_instances.len()];

                for instance_id in my_instances.iter() {
                    let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(*instance_id);
                    let setup = self.sctx.get_setup(airgroup_id, air_id);
                    let n_airgroup_values = setup
                        .stark_info
                        .airgroupvalues_map
                        .as_ref()
                        .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
                        .unwrap_or(0);

                    let proof = proofs[*instance_id].read().expect("Missing proof");

                    let airgroup_values: Vec<F> =
                        proof.proof[0..n_airgroup_values].to_vec().iter().map(|&x| F::from_u64(x)).collect();

                    airgroup_values_air_instances[self.pctx.dctx_get_instance_idx(*instance_id)] = airgroup_values;
                }

                let airgroupvalues_u64 = aggregate_airgroupvals(&self.pctx, &airgroup_values_air_instances);
                let airgroupvalues = self.pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

                if !options.test_mode && self.pctx.dctx_get_rank() == 0 {
                    let valid_global_constraints =
                        verify_global_constraints_proof(&self.pctx, &self.sctx, &DebugInfo::default(), airgroupvalues);
                    if valid_global_constraints.is_err() {
                        valid_proofs = false;
                    }
                }

                if valid_proofs {
                    tracing::info!("··· {}", "\u{2713} All proofs were successfully verified".bright_green().bold());
                    return Ok((None, None));
                } else {
                    return Err("Basic proofs were not verified".into());
                }
            } else {
                tracing::info!(
                    "··· {}",
                    "\u{2713} All proofs were successfully generated. Verification Skipped".bright_yellow().bold()
                );
                return Ok((None, None));
            }
        }

        timer_start_info!(GENERATING_OUTER_COMPRESSED_PROOFS);
        let trace = create_buffer_fast::<F>(self.trace_size);
        let prover_buffer = create_buffer_fast::<F>(self.prover_buffer_size);
        let recursive2_proofs_data: Vec<Vec<Proof<F>>> =
            recursive2_proofs.iter().map(|lock| lock.read().unwrap().clone()).collect();

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
        let mut vadcop_final_proof = None;
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

            proof_id = Some(
                blake3::hash(unsafe {
                    std::slice::from_raw_parts(
                        vadcop_proof_final.proof.as_ptr() as *const u8,
                        vadcop_proof_final.proof.len() * 8,
                    )
                })
                .to_hex()
                .to_string(),
            );

            vadcop_final_proof = Some(vadcop_proof_final.proof.clone());

            if options.final_snark {
                timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
                let recursivef_proof = generate_recursivef_proof(
                    &self.pctx,
                    &self.setups,
                    &vadcop_proof_final.proof,
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
            let valid_proofs = verify_final_proof(
                &vadcop_final_proof.clone().unwrap(),
                stark_info_path,
                expressions_bin_path,
                verkey_path,
            );
            timer_stop_and_log_info!(VERIFYING_VADCOP_FINAL_PROOF);
            if !valid_proofs {
                tracing::info!("··· {}", "\u{2717} Vadcop Final proof was not verified".bright_red().bold());
                return Err("Vadcop Final proof was not verified".into());
            } else {
                tracing::info!("··· {}", "\u{2713} Vadcop Final proof was verified".bright_green().bold());
            }
        }

        Ok((proof_id, vadcop_final_proof))
    }

    fn prepare_gpu(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        setups_vadcop: Arc<SetupsVadcop<F>>,
        aggregation: bool,
        gpu_params: &ParamsGPU,
    ) -> (Arc<DeviceBuffer>, u64, u64) {
        let mut free_memory_gpu = match cfg!(feature = "gpu") {
            true => check_device_memory_c() as f64 * 0.98,
            false => 0.0,
        };

        let n_gpus = get_num_gpus_c();
        let n_processes_node = pctx.dctx_get_node_n_processes() as u64;

        let n_partitions = match cfg!(feature = "gpu") {
            true => {
                if n_gpus > n_processes_node {
                    1
                } else {
                    n_processes_node.div_ceil(n_gpus)
                }
            }
            false => 1,
        };

        free_memory_gpu /= n_partitions as f64;

        pctx.dctx_barrier(); // important: all processes synchronize before allocation GPU memory

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

        (d_buffers, n_streams_per_gpu as u64, n_gpus)
    }

    #[allow(clippy::too_many_arguments)]
    fn gen_proof(
        proofs: Arc<Vec<RwLock<Proof<F>>>>,
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
        gpu_preallocate: bool,
    ) {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
        timer_start_info!(GEN_PROOF, "GEN_PROOF_{} [{}:{}]", instance_id, airgroup_id, air_id);
        Self::initialize_air_instance(&pctx, &sctx, instance_id, false);

        let setup = sctx.get_setup(airgroup_id, air_id);
        let p_setup: *mut c_void = (&setup.p_setup).into();
        let air_instance_name = &pctx.global_info.airs[airgroup_id][air_id].name;

        let mut steps_params = pctx.get_air_instance_params(&sctx, instance_id, true);

        if cfg!(not(feature = "gpu")) {
            steps_params.aux_trace = aux_trace.as_ptr() as *mut u8;
            steps_params.p_const_pols = const_pols.as_ptr() as *mut u8;
            steps_params.p_const_tree = const_tree.as_ptr() as *mut u8;
        } else if !gpu_preallocate {
            steps_params.p_const_pols = setup.get_const_ptr();
            steps_params.p_const_tree = setup.get_const_tree_ptr();
        }

        let p_steps_params: *mut u8 = (&steps_params).into();

        let output_file_path = output_dir_path.join(format!("proofs/{air_instance_name}_{instance_id}.json"));

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
            proofs[instance_id].read().unwrap().proof.as_ptr() as *mut u64,
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

        timer_stop_and_log_info!(GEN_PROOF, "GEN_PROOF_{} [{}:{}]", instance_id, airgroup_id, air_id);
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    #[cfg(distributed)]
    fn initialize_proofman(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        verify_constraints: bool,
        aggregation: bool,
        final_snark: bool,
        gpu_params: &ParamsGPU,
        verbose_mode: VerboseMode,
        mpi_universe: Option<Universe>,
    ) -> Result<(Arc<ProofCtx<F>>, Arc<SetupCtx<F>>, Arc<SetupsVadcop<F>>), Box<dyn std::error::Error>> {
        let mut pctx = ProofCtx::create_ctx(
            proving_key_path.clone(),
            custom_commits_fixed,
            aggregation,
            final_snark,
            verbose_mode,
            mpi_universe,
        );
        timer_start_info!(INITIALIZING_PROOFMAN);

        let sctx: Arc<SetupCtx<F>> =
            Arc::new(SetupCtx::new(&pctx.global_info, &ProofType::Basic, verify_constraints, gpu_params.preallocate));
        pctx.set_weights(&sctx);

        let pctx = Arc::new(pctx);
        if !verify_constraints {
            check_tree_paths(&pctx, &sctx)?;
        }

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

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    #[cfg(not(distributed))]
    fn initialize_proofman(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        verify_constraints: bool,
        aggregation: bool,
        final_snark: bool,
        gpu_params: &ParamsGPU,
        verbose_mode: VerboseMode,
    ) -> Result<(Arc<ProofCtx<F>>, Arc<SetupCtx<F>>, Arc<SetupsVadcop<F>>), Box<dyn std::error::Error>> {
        let mut pctx = ProofCtx::create_ctx(
            proving_key_path.clone(),
            custom_commits_fixed,
            aggregation,
            final_snark,
            verbose_mode,
        );
        timer_start_info!(INITIALIZING_PROOFMAN);

        let sctx: Arc<SetupCtx<F>> =
            Arc::new(SetupCtx::new(&pctx.global_info, &ProofType::Basic, verify_constraints, gpu_params.preallocate));
        pctx.set_weights(&sctx);

        let pctx = Arc::new(pctx);
        if !verify_constraints {
            check_tree_paths(&pctx, &sctx)?;
        }
        Self::initialize_publics_custom_commits(&sctx, &pctx)?;

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
        roots_contributions: Arc<Vec<Mutex<[F; 4]>>>,
        values_contributions: Arc<Vec<Mutex<Vec<F>>>>,
    ) {
        timer_start_info!(CALCULATE_GLOBAL_CHALLENGE);
        let my_instances = pctx.dctx_get_my_instances();

        let mut values = vec![0u64; my_instances.len() * 10];

        for instance_id in my_instances.iter() {
            let mut contribution = vec![F::ZERO; 10];

            let root_contribution = *roots_contributions[*instance_id].lock().expect("Missing root_contribution");

            let mut values_to_hash =
                values_contributions[*instance_id].lock().expect("Missing values_contribution").clone();
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
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
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
                        write!(acc, "[{l}]").unwrap();
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
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let mut air_instance = pctx.air_instances[instance_id].write().unwrap();
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
                let custom_commit_file_path = pctx
                    .get_custom_commits_fixed_buffer(&setup.stark_info.custom_commits[commit_id].name, true)
                    .unwrap();

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

    fn initialize_publics_custom_commits(
        sctx: &SetupCtx<F>,
        pctx: &ProofCtx<F>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Initializing publics custom_commits");
        for (airgroup_id, airs) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in airs.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id);
                for custom_commit in &setup.stark_info.custom_commits {
                    if custom_commit.stage_widths[0] > 0 {
                        // Handle the possibility that this returns None
                        let custom_file_path = pctx.get_custom_commits_fixed_buffer(&custom_commit.name, true)?;

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
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let steps_params = pctx.get_air_instance_params(sctx, instance_id, false);

        calculate_impols_expressions_c((&setup.p_setup).into(), stage as u64, (&steps_params).into());
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_contribution_air(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        roots_contributions: Arc<Vec<Mutex<[F; 4]>>>,
        values_contributions: Arc<Vec<Mutex<Vec<F>>>>,
        instance_id: usize,
        aux_trace_contribution_ptr: *mut u8,
        d_buffers: Arc<DeviceBuffer>,
        streams: Arc<Mutex<Vec<Option<u64>>>>,
    ) {
        let n_field_elements = 4;
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);

        timer_start_info!(GET_CONTRIBUTION_AIR, "GET_CONTRIBUTION_AIR_{} [{}:{}]", instance_id, airgroup_id, air_id);

        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let air_values = pctx.get_air_instance_air_values(airgroup_id, air_id, air_instance_id).clone();

        let root_ptr = roots_contributions[instance_id].lock().unwrap().as_ptr() as *mut u8;
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

        *values_contributions[instance_id].lock().unwrap() = values_hash;

        timer_stop_and_log_info!(
            GET_CONTRIBUTION_AIR,
            "GET_CONTRIBUTION_AIR_{} [{}:{}]",
            instance_id,
            airgroup_id,
            air_id
        );
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
