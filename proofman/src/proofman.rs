use libloading::{Library, Symbol};
use curves::{EcGFp5, EcMasFp5, curve::EllipticCurve};
use fields::{ExtensionField, PrimeField64, GoldilocksQuinticExtension};
#[cfg(distributed)]
use mpi::environment::Universe;
use std::ops::Add;
use std::sync::atomic::AtomicUsize;
use proofman_common::{
    calculate_fixed_tree, configured_num_threads, load_const_pols, skip_prover_instance, CurveType, DebugInfo,
    MemoryHandler, ParamsGPU, Proof, ProofCtx, ProofOptions, ProofType, SetupCtx, SetupsVadcop, VerboseMode,
    MAX_INSTANCES,
};
use rand::Rng;
use colored::Colorize;
use proofman_hints::aggregate_airgroupvals;
use proofman_starks_lib_c::{free_device_buffers_c, gen_device_buffers_c, get_num_gpus_c};
use proofman_starks_lib_c::{
    save_challenges_c, save_proof_values_c, save_publics_c, check_device_memory_c, gen_device_streams_c,
    get_stream_proofs_c, get_stream_proofs_non_blocking_c, register_proof_done_callback_c, reset_device_streams_c,
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

use std::{
    path::{PathBuf, Path},
    sync::Arc,
};

use transcript::FFITranscript;

use witness::{WitnessLibInitFn, WitnessLibrary, WitnessManager};
use crate::{check_tree_paths_vadcop, gen_recursive_proof_size, initialize_fixed_pols_tree};
use crate::{verify_constraints_proof, verify_basic_proof, verify_final_proof, verify_global_constraints_proof};
use crate::MaxSizes;
use crate::{print_summary_info, get_recursive_buffer_sizes, n_publics_aggregation};
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
    wcm: Arc<WitnessManager<F>>,
    gpu_params: ParamsGPU,
    verify_constraints: bool,
    aggregation: bool,
    final_snark: bool,
    n_streams: usize,
    memory_handler: Arc<MemoryHandler<F>>,
    proofs: Arc<Vec<RwLock<Option<Proof<F>>>>>,
    compressor_proofs: Arc<Vec<RwLock<Option<Proof<F>>>>>,
    recursive1_proofs: Arc<Vec<RwLock<Option<Proof<F>>>>>,
    recursive2_proofs: Arc<Vec<RwLock<Vec<Proof<F>>>>>,
    recursive2_proofs_ongoing: Arc<RwLock<Vec<Option<Proof<F>>>>>,
    roots_contributions: Arc<Vec<[F; 4]>>,
    values_contributions: Arc<Vec<Mutex<Vec<F>>>>,
    aux_trace: Arc<Vec<F>>,
    const_pols: Arc<Vec<F>>,
    const_tree: Arc<Vec<F>>,
    prover_buffer_recursive: Arc<Vec<F>>,
    max_num_threads: usize,
    tx_threads: Sender<()>,
    rx_threads: Receiver<()>,
    witness_tx: Sender<usize>,
    witness_rx: Receiver<usize>,
    witness_tx_priority: Sender<usize>,
    witness_rx_priority: Receiver<usize>,
    contributions_tx: Sender<usize>,
    contributions_rx: Receiver<usize>,
    proofs_tx: Sender<usize>,
    proofs_rx: Receiver<usize>,
    compressor_witness_tx: Sender<Proof<F>>,
    compressor_witness_rx: Receiver<Proof<F>>,
    rec1_witness_tx: Sender<Proof<F>>,
    rec1_witness_rx: Receiver<Proof<F>>,
    rec2_witness_tx: Sender<Proof<F>>,
    rec2_witness_rx: Receiver<Proof<F>>,
    recursive_tx: Sender<(u64, String)>,
    recursive_rx: Receiver<(u64, String)>,
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
            proving_key_path,
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
            return Err(format!("Proving key folder not found at path: {proving_key_path:?}").into());
        }

        let pctx = ProofCtx::<F>::create_ctx(proving_key_path, HashMap::new(), aggregation, final_snark, verbose_mode);

        Self::check_setup_(&pctx, aggregation, final_snark)
    }

    pub fn check_setup_(
        pctx: &ProofCtx<F>,
        aggregation: bool,
        final_snark: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let setups_aggregation =
            Arc::new(SetupsVadcop::<F>::new(&pctx.global_info, false, aggregation, false, &ParamsGPU::new(false)));

        let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false, &ParamsGPU::new(false));

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

        self.reset();

        self.wcm.execute();

        self.pctx.dctx_assign_instances(false);
        self.pctx.dctx_close();
        timer_stop_and_log_info!(EXECUTE);

        print_summary_info(&self.pctx, &self.sctx);

        let mut air_info: HashMap<&String, CsvInfo> = HashMap::new();

        let instances = self.pctx.dctx_get_instances();

        for (airgroup_id, air_group) in self.pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let air_name = &self.pctx.global_info.airs[airgroup_id][air_id].name;

                air_info.insert(
                    air_name,
                    CsvInfo {
                        version: env!("CARGO_PKG_VERSION").to_string(),
                        name: air_name.to_string(),
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

            let air_name = &self.pctx.global_info.airs[airgroup_id][air_id].name;

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

            air_info.entry(air_name).and_modify(|info| {
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
                let air_name = &self.pctx.global_info.airs[airgroup_id][air_id].name;
                let info = air_info.get_mut(air_name).unwrap();
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
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, self.get_rank())?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.wcm.set_input_data_path(input_data_path);
        self.pctx.set_debug_info(debug_info);

        self.register_witness(&mut *witness_lib, library);

        self.compute_witness_(options)
    }

    /// Computes only the witness without generating a proof neither verifying constraints.
    /// This is useful for debugging or benchmarking purposes.
    pub fn compute_witness_from_lib(
        &self,
        input_data_path: Option<PathBuf>,
        debug_info: &DebugInfo,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.pctx.set_debug_info(debug_info);
        self.wcm.set_input_data_path(input_data_path);
        self.compute_witness_(options)
    }

    pub fn compute_witness_(&self, options: ProofOptions) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(EXECUTE);

        if !self.wcm.is_init_witness() {
            println!("Witness computation dynamic library not initialized");
            return Err("Witness computation dynamic library not initialized".into());
        }

        self.reset();

        self.wcm.execute();

        // create a vector of instances wc weights
        self.pctx.dctx_assign_instances(options.minimal_memory);
        self.pctx.dctx_close();

        print_summary_info(&self.pctx, &self.sctx);

        timer_stop_and_log_info!(EXECUTE);

        let mut my_instances_sorted = self.pctx.dctx_get_my_instances();
        let mut rng = StdRng::seed_from_u64(self.pctx.dctx_get_rank() as u64);
        my_instances_sorted.shuffle(&mut rng);

        let my_instances_sorted_no_tables =
            my_instances_sorted.iter().filter(|idx| !self.pctx.dctx_is_table(**idx)).copied().collect::<Vec<_>>();

        let memory_handler =
            Arc::new(MemoryHandler::new(self.gpu_params.max_witness_stored, self.sctx.max_witness_trace_size));

        if !options.minimal_memory {
            self.pctx.set_witness_tx(Some(self.witness_tx.clone()));
            self.pctx.set_witness_tx_priority(Some(self.witness_tx_priority.clone()));
        }

        let witness_done = Arc::new(Counter::new());

        let (witness_handler, witness_handles) =
            self.calc_witness_handler(witness_done.clone(), memory_handler.clone(), options.minimal_memory, true);
        self.calculate_witness(
            &my_instances_sorted_no_tables,
            memory_handler.clone(),
            witness_done.clone(),
            options.minimal_memory,
            true,
        );

        if !options.minimal_memory {
            self.pctx.set_witness_tx(None);
            self.pctx.set_witness_tx_priority(None);
        }

        self.witness_tx.send(usize::MAX).ok();

        if let Some(h) = witness_handler {
            h.join().unwrap();
        }

        let handles_to_join = witness_handles.lock().unwrap().drain(..).collect::<Vec<_>>();
        for handle in handles_to_join {
            handle.join().unwrap();
        }

        drop(witness_handles);

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
        self.reset();

        self.pctx.set_debug_info(debug_info);

        if !self.wcm.is_init_witness() {
            return Err("Witness computation dynamic library not initialized".into());
        }

        timer_start_info!(EXECUTE);
        self.wcm.execute();
        timer_stop_and_log_info!(EXECUTE);

        // create a vector of instances wc weights
        self.pctx.dctx_assign_instances(false);
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
        let airgroup_values_air_instances = Mutex::new(vec![Vec::new(); my_instances.len()]);
        let valid_constraints = AtomicBool::new(true);
        let mut thread_handle: Option<std::thread::JoinHandle<()>> = None;

        let max_num_threads = configured_num_threads(self.pctx.dctx_get_node_n_processes());

        for &instance_id in my_instances.iter() {
            let instance_info = instances[instance_id];
            let (airgroup_id, air_id, is_table) =
                (instance_info.airgroup_id, instance_info.air_id, instance_info.table);
            let (skip, _) = skip_prover_instance(&self.pctx, instance_id);
            if is_table || skip {
                continue;
            }

            self.wcm.pre_calculate_witness(1, &[instance_id], max_num_threads, self.memory_handler.as_ref());
            self.wcm.calculate_witness(1, &[instance_id], max_num_threads, self.memory_handler.as_ref());

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            self.verify_proof_constraints_stage(
                &valid_constraints,
                &airgroup_values_air_instances,
                instance_id,
                airgroup_id,
                air_id,
                debug_info,
                max_num_threads,
            );
        }

        let my_instances_tables = self.pctx.dctx_get_my_tables();

        timer_start_info!(CALCULATING_TABLES);
        for instance_id in my_instances_tables.iter() {
            self.wcm.calculate_witness(1, &[*instance_id], max_num_threads, self.memory_handler.as_ref());
        }
        timer_stop_and_log_info!(CALCULATING_TABLES);

        for instance_id in my_instances_tables.iter() {
            let (skip, _) = skip_prover_instance(&self.pctx, *instance_id);

            if skip || !self.pctx.dctx_is_my_instance(*instance_id) {
                continue;
            };

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            let instance_info = &instances[*instance_id];
            let (airgroup_id, air_id) = (instance_info.airgroup_id, instance_info.air_id);
            self.verify_proof_constraints_stage(
                &valid_constraints,
                &airgroup_values_air_instances,
                *instance_id,
                airgroup_id,
                air_id,
                debug_info,
                max_num_threads,
            );
        }

        self.wcm.end(debug_info);

        let check_global_constraints =
            debug_info.debug_instances.is_empty() || !debug_info.debug_global_instances.is_empty();

        if check_global_constraints && !test_mode {
            let airgroup_values_air_instances = airgroup_values_air_instances.lock().unwrap();
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
        &self,
        valid_constraints: &AtomicBool,
        airgroup_values_air_instances: &Mutex<Vec<Vec<F>>>,
        instance_id: usize,
        airgroup_id: usize,
        air_id: usize,
        debug_info: &DebugInfo,
        max_num_threads: usize,
    ) {
        Self::initialize_air_instance(&self.pctx, &self.sctx, instance_id, true, true);

        #[cfg(feature = "diagnostic")]
        {
            let invalid_initialization = Self::diagnostic_instance(&self.pctx, &self.sctx, instance_id);
            if invalid_initialization {
                panic!("Invalid initialization");
                // return Some(Err("Invalid initialization".into()));
            }
        }

        self.wcm.calculate_witness(2, &[instance_id], max_num_threads, self.memory_handler.as_ref());
        Self::calculate_im_pols(2, &self.sctx, &self.pctx, instance_id);

        self.wcm.debug(&[instance_id], debug_info);

        let valid =
            verify_constraints_proof(&self.pctx, &self.sctx, instance_id, debug_info.n_print_constraints as u64);
        if !valid {
            valid_constraints.fetch_and(valid, Ordering::Relaxed);
        }

        let air_instance_id = self.pctx.dctx_find_air_instance_id(instance_id);
        let airgroup_values = self.pctx.get_air_instance_airgroup_values(airgroup_id, air_id, air_instance_id);
        airgroup_values_air_instances.lock().unwrap()[self.pctx.dctx_get_instance_idx(instance_id)] = airgroup_values;
        let (is_shared_buffer, witness_buffer) = self.pctx.free_instance(instance_id);
        if is_shared_buffer {
            self.memory_handler.release_buffer(witness_buffer);
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

        let (d_buffers, n_streams_per_gpu, n_recursive_streams_per_gpu, n_gpus) =
            Self::prepare_gpu(&pctx, &sctx, &setups_vadcop, aggregation, &gpu_params);

        if !verify_constraints {
            initialize_fixed_pols_tree(&pctx, &sctx, &setups_vadcop, &d_buffers, aggregation, &gpu_params);
        }

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone()));

        timer_stop_and_log_info!(INIT_PROOFMAN);

        let max_witness_stored = match cfg!(feature = "gpu") {
            true => gpu_params.max_witness_stored,
            false => 1,
        };

        let memory_handler = Arc::new(MemoryHandler::new(max_witness_stored, sctx.max_witness_trace_size));

        let n_airgroups = pctx.global_info.air_groups.len();
        let proofs: Arc<Vec<RwLock<Option<Proof<F>>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| RwLock::new(None)).collect());
        let compressor_proofs: Arc<Vec<RwLock<Option<Proof<F>>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| RwLock::new(None)).collect());
        let recursive1_proofs: Arc<Vec<RwLock<Option<Proof<F>>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| RwLock::new(None)).collect());
        let recursive2_proofs: Arc<Vec<RwLock<Vec<Proof<F>>>>> =
            Arc::new((0..n_airgroups).map(|_| RwLock::new(Vec::new())).collect());
        let recursive2_proofs_ongoing: Arc<RwLock<Vec<Option<Proof<F>>>>> = Arc::new(RwLock::new(Vec::new()));

        let (aux_trace, const_pols, const_tree) = if cfg!(feature = "gpu") {
            (Arc::new(Vec::new()), Arc::new(Vec::new()), Arc::new(Vec::new()))
        } else {
            (
                Arc::new(create_buffer_fast(sctx.max_prover_buffer_size.max(setups_vadcop.max_prover_buffer_size))),
                Arc::new(create_buffer_fast(sctx.max_const_size.max(setups_vadcop.max_const_size))),
                Arc::new(create_buffer_fast(sctx.max_const_tree_size.max(setups_vadcop.max_const_tree_size))),
            )
        };

        let n_proof_threads = match cfg!(feature = "gpu") {
            true => n_gpus,
            false => 1,
        };

        let n_streams = ((n_streams_per_gpu + n_recursive_streams_per_gpu) * n_proof_threads) as usize;

        let max_num_threads = configured_num_threads(pctx.dctx_get_node_n_processes());

        let prover_buffer_recursive = if cfg!(not(feature = "gpu")) && aggregation {
            let prover_buffer_size = get_recursive_buffer_sizes(&pctx, &setups_vadcop)?;
            Arc::new(create_buffer_fast(prover_buffer_size))
        } else {
            Arc::new(Vec::new())
        };

        let values_contributions: Arc<Vec<Mutex<Vec<F>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| Mutex::new(Vec::<F>::new())).collect());

        let roots_contributions: Arc<Vec<[F; 4]>> = Arc::new((0..MAX_INSTANCES).map(|_| [F::default(); 4]).collect());

        // define managment channels and counters
        let (tx_threads, rx_threads) = bounded::<()>(max_num_threads);

        for _ in 0..max_num_threads {
            tx_threads.send(()).unwrap();
        }

        let (witness_tx, witness_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (witness_tx_priority, witness_rx_priority): (Sender<usize>, Receiver<usize>) = unbounded();
        let (contributions_tx, contributions_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (recursive_tx, recursive_rx) = unbounded::<(u64, String)>();
        let (proofs_tx, proofs_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (compressor_witness_tx, compressor_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();
        let (rec1_witness_tx, rec1_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();
        let (rec2_witness_tx, rec2_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();

        Ok(Self {
            pctx,
            sctx,
            wcm,
            setups: setups_vadcop,
            d_buffers,
            prover_buffer_recursive,
            gpu_params,
            aggregation,
            final_snark,
            verify_constraints,
            n_streams,
            max_num_threads,
            memory_handler,
            proofs,
            compressor_proofs,
            recursive1_proofs,
            recursive2_proofs,
            recursive2_proofs_ongoing,
            aux_trace,
            const_pols,
            const_tree,
            roots_contributions,
            values_contributions,
            tx_threads,
            rx_threads,
            witness_tx,
            witness_rx,
            witness_tx_priority,
            witness_rx_priority,
            contributions_tx,
            contributions_rx,
            recursive_tx,
            recursive_rx,
            proofs_tx,
            proofs_rx,
            compressor_witness_tx,
            compressor_witness_rx,
            rec1_witness_tx,
            rec1_witness_rx,
            rec2_witness_tx,
            rec2_witness_rx,
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
        )?;

        timer_start_info!(INIT_PROOFMAN);

        let (d_buffers, n_streams_per_gpu, n_recursive_streams_per_gpu, n_gpus) =
            Self::prepare_gpu(&pctx, &sctx, &setups_vadcop, aggregation, &gpu_params);

        if !verify_constraints {
            initialize_fixed_pols_tree(&pctx, &sctx, &setups_vadcop, &d_buffers, aggregation, &gpu_params);
        }

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone()));

        timer_stop_and_log_info!(INIT_PROOFMAN);

        let max_witness_stored = match cfg!(feature = "gpu") {
            true => gpu_params.max_witness_stored,
            false => 1,
        };

        let memory_handler = Arc::new(MemoryHandler::new(max_witness_stored, sctx.max_witness_trace_size));

        let n_airgroups = pctx.global_info.air_groups.len();
        let proofs: Arc<Vec<RwLock<Option<Proof<F>>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| RwLock::new(None)).collect());
        let compressor_proofs: Arc<Vec<RwLock<Option<Proof<F>>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| RwLock::new(None)).collect());
        let recursive1_proofs: Arc<Vec<RwLock<Option<Proof<F>>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| RwLock::new(None)).collect());
        let recursive2_proofs: Arc<Vec<RwLock<Vec<Proof<F>>>>> =
            Arc::new((0..n_airgroups).map(|_| RwLock::new(Vec::new())).collect());
        let recursive2_proofs_ongoing: Arc<RwLock<Vec<Option<Proof<F>>>>> = Arc::new(RwLock::new(Vec::new()));

        let (aux_trace, const_pols, const_tree) = if cfg!(feature = "gpu") {
            (Arc::new(Vec::new()), Arc::new(Vec::new()), Arc::new(Vec::new()))
        } else {
            (
                Arc::new(create_buffer_fast(sctx.max_prover_buffer_size.max(setups_vadcop.max_prover_buffer_size))),
                Arc::new(create_buffer_fast(sctx.max_const_size.max(setups_vadcop.max_const_size))),
                Arc::new(create_buffer_fast(sctx.max_const_tree_size.max(setups_vadcop.max_const_tree_size))),
            )
        };

        let n_proof_threads = match cfg!(feature = "gpu") {
            true => n_gpus,
            false => 1,
        };

        let n_streams = ((n_streams_per_gpu + n_recursive_streams_per_gpu) * n_proof_threads) as usize;

        let max_num_threads = configured_num_threads(pctx.dctx_get_node_n_processes());

        let prover_buffer_recursive = if aggregation {
            let prover_buffer_size = get_recursive_buffer_sizes(&pctx, &setups_vadcop)?;
            Arc::new(create_buffer_fast(prover_buffer_size))
        } else {
            Arc::new(Vec::new())
        };

        let values_contributions: Arc<Vec<Mutex<Vec<F>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| Mutex::new(Vec::<F>::new())).collect());

        let roots_contributions: Arc<Vec<[F; 4]>> = Arc::new((0..MAX_INSTANCES).map(|_| [F::default(); 4]).collect());

        // define managment channels and counters
        let (tx_threads, rx_threads) = bounded::<()>(max_num_threads);

        for _ in 0..max_num_threads {
            tx_threads.send(()).unwrap();
        }

        let (witness_tx, witness_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (witness_tx_priority, witness_rx_priority): (Sender<usize>, Receiver<usize>) = unbounded();
        let (contributions_tx, contributions_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (recursive_tx, recursive_rx) = unbounded::<(u64, String)>();
        let (proofs_tx, proofs_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (compressor_witness_tx, compressor_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();
        let (rec1_witness_tx, rec1_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();
        let (rec2_witness_tx, rec2_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();

        Ok(Self {
            pctx,
            sctx,
            wcm,
            setups: setups_vadcop,
            d_buffers,
            prover_buffer_recursive,
            gpu_params,
            aggregation,
            final_snark,
            verify_constraints,
            n_streams,
            max_num_threads,
            memory_handler,
            proofs,
            compressor_proofs,
            recursive1_proofs,
            recursive2_proofs,
            recursive2_proofs_ongoing,
            aux_trace,
            const_pols,
            const_tree,
            roots_contributions,
            values_contributions,
            tx_threads,
            rx_threads,
            witness_tx,
            witness_rx,
            witness_tx_priority,
            witness_rx_priority,
            contributions_tx,
            contributions_rx,
            recursive_tx,
            recursive_rx,
            proofs_tx,
            proofs_rx,
            compressor_witness_tx,
            compressor_witness_rx,
            rec1_witness_tx,
            rec1_witness_rx,
            rec2_witness_tx,
            rec2_witness_rx,
        })
    }

    pub fn reset(&self) {
        self.pctx.dctx_reset();

        for proof_lock in self.proofs.iter() {
            let mut proof = proof_lock.write().unwrap();
            *proof = None;
        }

        for proof_lock in self.compressor_proofs.iter() {
            let mut proof = proof_lock.write().unwrap();
            *proof = None;
        }

        for proof_lock in self.recursive1_proofs.iter() {
            let mut proof = proof_lock.write().unwrap();
            *proof = None;
        }

        for proof_lock in self.recursive2_proofs.iter() {
            let mut proofs = proof_lock.write().unwrap();
            proofs.clear();
        }

        let mut ongoing_proofs = self.recursive2_proofs_ongoing.write().unwrap();
        ongoing_proofs.clear();

        // Drain all relevant channels to ensure they are empty
        while self.witness_rx.try_recv().is_ok() {}
        while self.witness_rx_priority.try_recv().is_ok() {}
        while self.contributions_rx.try_recv().is_ok() {}
        while self.recursive_rx.try_recv().is_ok() {}
        while self.proofs_rx.try_recv().is_ok() {}
        while self.compressor_witness_rx.try_recv().is_ok() {}
        while self.rec1_witness_rx.try_recv().is_ok() {}
        while self.rec2_witness_rx.try_recv().is_ok() {}

        reset_device_streams_c(self.d_buffers.get_ptr());
    }

    pub fn register_witness(&self, witness_lib: &mut dyn WitnessLibrary<F>, library: Library) {
        timer_start_info!(REGISTERING_WITNESS);
        witness_lib.register_witness(&self.wcm);
        self.wcm.set_init_witness(true, library);
        timer_stop_and_log_info!(REGISTERING_WITNESS);
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

        self.reset();

        if !options.test_mode {
            Self::initialize_publics_custom_commits(&self.sctx, &self.pctx)?;
        }

        if !options.minimal_memory && cfg!(feature = "gpu") {
            self.pctx.set_witness_tx(Some(self.witness_tx.clone()));
            self.pctx.set_witness_tx_priority(Some(self.witness_tx_priority.clone()));
        }
        let witness_done = Arc::new(Counter::new());

        // let (witness_handler, witness_handles) =
        //     self.calc_witness_handler(witness_done.clone(), self.memory_handler.clone(), options.minimal_memory, false);

        self.wcm.execute();

        self.pctx.dctx_assign_instances(options.minimal_memory);
        self.pctx.dctx_close();

        print_summary_info(&self.pctx, &self.sctx);

        timer_stop_and_log_info!(EXECUTE);

        timer_start_info!(CALCULATING_CONTRIBUTIONS);
        timer_start_info!(CALCULATING_INNER_CONTRIBUTIONS);
        // timer_start_info!(PREPARING_CONTRIBUTIONS);

        let instances = self.pctx.dctx_get_instances();
        let my_instances = self.pctx.dctx_get_my_instances();
        let my_instances_tables = self.pctx.dctx_get_my_tables();

        let mut my_instances_sorted = self.pctx.dctx_get_my_instances();
        let mut rng = StdRng::seed_from_u64(self.pctx.dctx_get_rank() as u64);
        my_instances_sorted.shuffle(&mut rng);

        // let my_instances_sorted_no_tables =
        //     my_instances_sorted.iter().filter(|idx| !self.pctx.dctx_is_table(**idx)).copied().collect::<Vec<_>>();

        // let instances_mine_no_tables = my_instances_sorted_no_tables.len();

        // let max_witness_stored = match cfg!(feature = "gpu") {
        //     true => instances_mine_no_tables.min(self.gpu_params.max_witness_stored),
        //     false => 1,
        // };

        // let streams = Arc::new(Mutex::new(vec![None; self.n_streams]));

        // let witnesses_done = Arc::new(AtomicUsize::new(0));

        // self.pctx.set_proof_tx(Some(self.contributions_tx.clone()));

        // timer_stop_and_log_info!(PREPARING_CONTRIBUTIONS);

        // let mut handle_contributions = Vec::new();
        // for _ in 0..self.n_streams {
        //     let pctx_clone = self.pctx.clone();
        //     let sctx_clone = self.sctx.clone();
        //     let roots_contributions_clone = self.roots_contributions.clone();
        //     let values_contributions_clone = self.values_contributions.clone();
        //     let aux_trace_clone = self.aux_trace.clone();
        //     let d_buffers_clone = self.d_buffers.clone();
        //     let streams_clone = streams.clone();
        //     let witnesses_done_clone = witnesses_done.clone();
        //     let memory_handler_clone = self.memory_handler.clone();
        //     let contributions_rx_clone = self.contributions_rx.clone();
        //     let contribution_handle = std::thread::spawn(move || loop {
        //         match contributions_rx_clone.try_recv() {
        //             Ok(instance_id) => {
        //                 if instance_id == usize::MAX {
        //                     break;
        //                 }
        //                 Self::get_contribution_air(
        //                     &pctx_clone,
        //                     &sctx_clone,
        //                     &roots_contributions_clone,
        //                     &values_contributions_clone,
        //                     instance_id,
        //                     aux_trace_clone.as_ptr() as *mut u8,
        //                     &d_buffers_clone,
        //                     &streams_clone,
        //                 );

        //                 if !pctx_clone.dctx_is_table(instance_id) {
        //                     witnesses_done_clone.fetch_add(1, Ordering::AcqRel);
        //                     if (instances_mine_no_tables - witnesses_done_clone.load(Ordering::Acquire))
        //                         >= max_witness_stored
        //                     {
        //                         let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance_traces(instance_id);
        //                         if is_shared_buffer {
        //                             memory_handler_clone.release_buffer(witness_buffer);
        //                         }
        //                     }
        //                 }
        //             }
        //             Err(crossbeam_channel::TryRecvError::Empty) => {
        //                 std::thread::sleep(std::time::Duration::from_micros(100));
        //                 continue;
        //             }
        //             Err(crossbeam_channel::TryRecvError::Disconnected) => {
        //                 break;
        //             }
        //         }
        //     });
        //     handle_contributions.push(contribution_handle);
        // }

        // self.calculate_witness(
        //     &my_instances_sorted_no_tables,
        //     self.memory_handler.clone(),
        //     witness_done.clone(),
        //     options.minimal_memory,
        //     false,
        // );

        // if !options.minimal_memory && cfg!(feature = "gpu") {
        //     self.pctx.set_witness_tx(None);
        //     self.pctx.set_witness_tx_priority(None);
        // }
        // self.witness_tx.send(usize::MAX).ok();

        // if let Some(h) = witness_handler {
        //     h.join().unwrap();
        // }
        // if cfg!(feature = "gpu") {
        //     let handles_to_join = witness_handles.lock().unwrap().drain(..).collect::<Vec<_>>();
        //     for handle in handles_to_join {
        //         handle.join().unwrap();
        //     }
        // }

        // drop(witness_handles);

        // timer_start_info!(CALCULATING_TABLES);

        // //evalutate witness for instances of type "tables"
        // for instance_id in my_instances_tables.iter() {
        //     self.wcm.pre_calculate_witness(1, &[*instance_id], self.max_num_threads, self.memory_handler.as_ref());
        //     self.wcm.calculate_witness(1, &[*instance_id], self.max_num_threads, self.memory_handler.as_ref());
        // }

        // timer_stop_and_log_info!(CALCULATING_TABLES);

        // self.pctx.set_proof_tx(None);

        // for _ in 0..self.n_streams {
        //     self.contributions_tx.send(usize::MAX).ok();
        // }

        // for handle in handle_contributions {
        //     handle.join().unwrap();
        // }

        // // get roots still in the streams
        // get_stream_proofs_c(self.d_buffers.get_ptr());

        // timer_stop_and_log_info!(CALCULATING_INNER_CONTRIBUTIONS);

        // //calculate-challenge
        // let internal_contribution = self.calculate_internal_contributions();

        // let all_partial_contributions = self.pctx.dctx_distribute_roots(internal_contribution);
        // let all_partial_contributions_u64 = all_partial_contributions
        //     .chunks(10)
        //     .map(|chunk| chunk.try_into().expect("Each chunk should be exactly 10 elements"))
        //     .collect::<Vec<[u64; 10]>>();

        // self.calculate_global_challenge(&all_partial_contributions_u64);

        // timer_stop_and_log_info!(CALCULATING_CONTRIBUTIONS);

        let global_challenge = vec![F::from_u64(rng.random_range(0..=(1 << 63) - 1)), F::from_u64(rng.random_range(0..=(1 << 63) - 1)), F::from_u64(rng.random_range(0..=(1 << 63) - 1))];
        self.pctx.set_global_challenge(2, &global_challenge);

        timer_start_info!(GENERATING_INNER_PROOFS);

        let n_airgroups = self.pctx.global_info.air_groups.len();

        // let vec_streams: Vec<(u64, u64)> = {
        //     let mut guard = streams.lock().unwrap();
        //     let taken = std::mem::take(&mut *guard);

        //     let mut result = Vec::new();
        //     for (idx, maybe_id) in taken.into_iter().enumerate() {
        //         if let Some(id) = maybe_id {
        //             result.push((idx as u64, id));
        //         }
        //     }

        //     result
        // };

        let mut n_airgroup_proofs = vec![0; n_airgroups];
        for (instance_id, instance_info) in instances.iter().enumerate() {
            if self.pctx.dctx_is_my_instance(instance_id) {
                n_airgroup_proofs[instance_info.airgroup_id] += 1;
            }
        }

        if options.aggregation {
            for (airgroup, &n_proofs) in n_airgroup_proofs.iter().enumerate().take(n_airgroups) {
                let n_recursive2_proofs = total_recursive_proofs(n_proofs);
                if n_recursive2_proofs.has_remaining {
                    let setup = self.setups.get_setup(airgroup, 0, &ProofType::Recursive2);
                    let publics_aggregation = n_publics_aggregation(&self.pctx, airgroup);
                    let null_proof_buffer = vec![0; setup.proof_size as usize + publics_aggregation];
                    let null_proof = Proof::new(ProofType::Recursive2, airgroup, 0, None, null_proof_buffer);
                    self.recursive2_proofs[airgroup].write().unwrap().push(null_proof);
                }
            }
        }

        let proofs_pending = Arc::new(Counter::new());

        register_proof_done_callback_c(self.recursive_tx.clone());

        self.pctx.set_proof_tx(Some(self.proofs_tx.clone()));

        let mut handle_recursives = Vec::new();
        for _ in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let setups_clone = self.setups.clone();
            let proofs_clone = self.proofs.clone();
            let compressor_proofs_clone = self.compressor_proofs.clone();
            let recursive1_proofs_clone = self.recursive1_proofs.clone();
            let recursive2_proofs_clone = self.recursive2_proofs.clone();
            let recursive2_proofs_ongoing_clone = self.recursive2_proofs_ongoing.clone();
            let proofs_pending_clone = proofs_pending.clone();
            let rec1_witness_tx_clone = self.rec1_witness_tx.clone();
            let rec2_witness_tx_clone = self.rec2_witness_tx.clone();
            let compressor_witness_tx_clone = self.compressor_witness_tx.clone();
            let recursive_rx_clone = self.recursive_rx.clone();
            let handle_recursive = std::thread::spawn(move || {
                while let Ok((id, proof_type)) = recursive_rx_clone.recv() {
                    if id == u64::MAX - 1 {
                        return;
                    }
                    let p: ProofType = proof_type.parse().unwrap();
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
                            recursive1_proofs_clone[id as usize].write().unwrap().take().unwrap()
                        } else {
                            recursive2_proofs_ongoing_clone.write().unwrap()[id as usize].take().unwrap()
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
                        let compressor_proof = compressor_proofs_clone[id as usize].write().unwrap().take().unwrap();
                        Some(gen_witness_recursive(&pctx_clone, &setups_clone, &compressor_proof).unwrap())
                    } else {
                        let proof = proofs_clone[id as usize].write().unwrap().take().unwrap();
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

        // let processed_ids = Mutex::new(Vec::new());

        // if cfg!(feature = "gpu") && !vec_streams.is_empty() {
        //     let processed: Vec<u64> = vec_streams
        //         .par_iter()
        //         .map(|&(stream_id, instance_id)| {
        //             proofs_pending.increment();

        //             Self::gen_proof(
        //                 &self.proofs,
        //                 &self.pctx,
        //                 &self.sctx,
        //                 instance_id as usize,
        //                 &options.output_dir_path,
        //                 &self.aux_trace,
        //                 &self.const_pols,
        //                 &self.const_tree,
        //                 &self.d_buffers,
        //                 Some(stream_id as usize),
        //                 options.save_proofs,
        //                 self.gpu_params.preallocate,
        //             );

        //             let (is_shared_buffer, witness_buffer) = self.pctx.free_instance(instance_id as usize);
        //             if is_shared_buffer {
        //                 self.memory_handler.release_buffer(witness_buffer);
        //             }

        //             instance_id
        //         })
        //         .collect();

        //     processed_ids.lock().unwrap().extend(processed);
        // }

        // let mut my_instances_calculated = vec![false; instances.len()];
        // for idx in processed_ids.into_inner().unwrap() {
        //     my_instances_calculated[idx as usize] = true;
        // }

        my_instances_sorted.sort_by_key(|&id| {
            let setup = self.sctx.get_setup(instances[id].airgroup_id, instances[id].air_id);
            (
                if setup.single_instance { 1 } else { 0 },
                if self.pctx.is_air_instance_stored(id) { 0 } else { 1 },
                if self.pctx.global_info.get_air_has_compressor(instances[id].airgroup_id, instances[id].air_id) {
                    0
                } else {
                    1
                },
            )
        });

        let proofs_finished = Arc::new(AtomicBool::new(false));
        for _ in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let setups_clone = self.setups.clone();
            let d_buffers_clone = self.d_buffers.clone();
            let output_dir_path_clone = options.output_dir_path.clone();
            let aux_trace_clone = self.aux_trace.clone();
            let const_pols_clone = self.const_pols.clone();
            let const_tree_clone = self.const_tree.clone();
            let prover_buffer_recursive = self.prover_buffer_recursive.clone();
            let proofs_clone = self.proofs.clone();
            let compressor_proofs_clone = self.compressor_proofs.clone();
            let recursive1_proofs_clone = self.recursive1_proofs.clone();
            let recursive2_proofs_ongoing_clone = self.recursive2_proofs_ongoing.clone();
            // let stream_clone = vec_streams.clone();
            let proofs_rx = self.proofs_rx.clone();
            let compressor_rx = self.compressor_witness_rx.clone();
            let rec2_rx = self.rec2_witness_rx.clone();
            let rec1_rx = self.rec1_witness_rx.clone();
            let preallocate = self.gpu_params.preallocate;

            let memory_handler_clone = self.memory_handler.clone();

            let proofs_finished_clone = proofs_finished.clone();

            let handle_recursive = std::thread::spawn(move || loop {
                // Handle proof witnesses (Proof<F> type)
                let witness = rec2_rx.try_recv().or_else(|_| compressor_rx.try_recv()).or_else(|_| rec1_rx.try_recv());

                // If not witness, check if there's a proof
                if witness.is_err() {
                    // Check if proof received
                    if let Ok(instance_id) = proofs_rx.try_recv() {
                        // let stream_id: Option<usize> =
                        //     stream_clone.iter().position(|&(_, id)| id == instance_id as u64);
                        Self::gen_proof(
                            &proofs_clone,
                            &pctx_clone,
                            &sctx_clone,
                            instance_id,
                            &output_dir_path_clone,
                            &aux_trace_clone,
                            &const_pols_clone,
                            &const_tree_clone,
                            &d_buffers_clone,
                            None,
                            options.save_proofs,
                            preallocate,
                        );
                        let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance(instance_id);
                        if is_shared_buffer {
                            memory_handler_clone.release_buffer(witness_buffer);
                        }
                    }

                    if proofs_finished_clone.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(100));
                    continue;
                }

                let mut witness = witness.unwrap();
                if witness.proof_type == ProofType::Recursive2 {
                    let id = {
                        let mut rec2_proofs = recursive2_proofs_ongoing_clone.write().unwrap();
                        let id = rec2_proofs.len();
                        rec2_proofs.push(None);
                        id
                    };

                    witness.global_idx = Some(id);
                }

                let new_proof = gen_recursive_proof_size(&pctx_clone, &setups_clone, &witness);
                let new_proof_type_str: &str = new_proof.proof_type.clone().into();

                let new_proof_type = &new_proof.proof_type.clone();

                let id = new_proof.global_idx.unwrap();
                if *new_proof_type == ProofType::Recursive2 {
                    recursive2_proofs_ongoing_clone.write().unwrap()[id] = Some(new_proof);
                } else if *new_proof_type == ProofType::Compressor {
                    *compressor_proofs_clone[id].write().unwrap() = Some(new_proof);
                } else if *new_proof_type == ProofType::Recursive1 {
                    *recursive1_proofs_clone[id].write().unwrap() = Some(new_proof);
                }

                if *new_proof_type == ProofType::Recursive2 {
                    let recursive2_lock = recursive2_proofs_ongoing_clone.read().unwrap();
                    let new_proof_ref = recursive2_lock[id].as_ref().unwrap();

                    let _ = generate_recursive_proof(
                        &pctx_clone,
                        &setups_clone,
                        &witness,
                        new_proof_ref,
                        &prover_buffer_recursive,
                        &output_dir_path_clone,
                        d_buffers_clone.get_ptr(),
                        &const_tree_clone,
                        &const_pols_clone,
                        options.save_proofs,
                    );
                } else if *new_proof_type == ProofType::Compressor {
                    let compressor_lock = compressor_proofs_clone[id].read().unwrap();
                    let new_proof_ref = compressor_lock.as_ref().unwrap();
                    let _ = generate_recursive_proof(
                        &pctx_clone,
                        &setups_clone,
                        &witness,
                        new_proof_ref,
                        &prover_buffer_recursive,
                        &output_dir_path_clone,
                        d_buffers_clone.get_ptr(),
                        &const_tree_clone,
                        &const_pols_clone,
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
                        &prover_buffer_recursive,
                        &output_dir_path_clone,
                        d_buffers_clone.get_ptr(),
                        &const_tree_clone,
                        &const_pols_clone,
                        options.save_proofs,
                    );
                }

                if cfg!(not(feature = "gpu")) {
                    launch_callback_c(id as u64, new_proof_type_str);
                }
            });
            handle_recursives.push(handle_recursive);
        }

        for &instance_id in my_instances_sorted.iter() {
            // if my_instances_calculated[instance_id] {
            //     continue;
            // }

            let tx_threads_clone: Sender<()> = self.tx_threads.clone();
            let proofs_tx_clone = self.proofs_tx.clone();
            let wcm = self.wcm.clone();
            let proofs_pending_clone = proofs_pending.clone();
            // let is_stored = self.pctx.is_air_instance_stored(instance_id)
            //     || vec_streams.iter().any(|&(_, id)| id == instance_id as u64);

            // my_instances_calculated[instance_id] = true;

            let instance_info = &instances[instance_id];
            let (airgroup_id, air_id) = (instance_info.airgroup_id, instance_info.air_id);

            let n_threads_witness = instance_info.threads_witness.min(self.max_num_threads);

            let threads_to_use_collect =
                (instance_info.n_chunks / 16).min(self.max_num_threads / 4).max(n_threads_witness);

            // if !is_stored {
                for _ in 0..threads_to_use_collect {
                    self.rx_threads.recv().unwrap();
                }
            // }

            let threads_to_use_witness = threads_to_use_collect.min(n_threads_witness);
            let threads_to_return = threads_to_use_collect - threads_to_use_witness;

            let memory_handler_clone = self.memory_handler.clone();

            let handle = std::thread::spawn(move || {
                proofs_pending_clone.increment();
                // if !is_stored {
                    timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    wcm.pre_calculate_witness(1, &[instance_id], threads_to_use_collect, memory_handler_clone.as_ref());
                    for _ in 0..threads_to_return {
                        tx_threads_clone.send(()).unwrap();
                    }
                    wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref());
                    for _ in 0..threads_to_use_witness {
                        tx_threads_clone.send(()).unwrap();
                    }
                    timer_stop_and_log_info!(
                        GENERATING_WC,
                        "GENERATING_WC_{} [{}:{}]",
                        instance_id,
                        airgroup_id,
                        air_id
                    );
                // } else {
                //     proofs_tx_clone.send(instance_id).unwrap();
                // }
            });
            if cfg!(not(feature = "gpu")) {
                handle.join().unwrap();
            } else {
                handle_recursives.push(handle);
            }
        }

        proofs_pending.wait_until_zero_and_check_streams(|| get_stream_proofs_non_blocking_c(self.d_buffers.get_ptr()));
        get_stream_proofs_c(self.d_buffers.get_ptr());
        proofs_finished.store(true, Ordering::Relaxed);
        clear_proof_done_callback_c();
        for _ in 0..self.n_streams {
            self.recursive_tx.send((u64::MAX - 1, "Basic".to_string())).unwrap();
        }

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
                let mut airgroup_values_air_instances = vec![Vec::new(); my_instances.len()];
                for instance_id in my_instances.iter() {
                    let proof = {
                        let mut lock = self.proofs[*instance_id].write().unwrap();
                        std::mem::take(&mut *lock)
                    };
                    let valid_proof = verify_basic_proof(&self.pctx, *instance_id, &proof.as_ref().unwrap().proof);
                    if !valid_proof {
                        valid_proofs = false;
                    }

                    let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(*instance_id);
                    let setup = self.sctx.get_setup(airgroup_id, air_id);
                    let n_airgroup_values = setup
                        .stark_info
                        .airgroupvalues_map
                        .as_ref()
                        .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
                        .unwrap_or(0);

                    let airgroup_values: Vec<F> = proof.as_ref().unwrap().proof[0..n_airgroup_values]
                        .to_vec()
                        .iter()
                        .map(|&x| F::from_u64(x))
                        .collect();

                    airgroup_values_air_instances[self.pctx.dctx_get_instance_idx(*instance_id)] = airgroup_values;
                }
                timer_stop_and_log_info!(VERIFYING_PROOFS);

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

        let recursive2_proofs_data: Vec<Vec<Proof<F>>> =
            self.recursive2_proofs.iter().map(|lock| std::mem::take(&mut *lock.write().unwrap())).collect();

        let agg_recursive2_proof = aggregate_recursive2_proofs(
            &self.pctx,
            &self.setups,
            recursive2_proofs_data,
            &self.prover_buffer_recursive,
            &self.const_pols,
            &self.const_tree,
            &options.output_dir_path,
            self.d_buffers.get_ptr(),
            false,
        )?;
        timer_stop_and_log_info!(GENERATING_OUTER_COMPRESSED_PROOFS);

        let mut proof_id = None;
        let mut vadcop_final_proof = None;
        if self.pctx.dctx_get_rank() == 0 {
            let vadcop_proof_final = generate_vadcop_final_proof(
                &self.pctx,
                &self.setups,
                &agg_recursive2_proof,
                &self.prover_buffer_recursive,
                &options.output_dir_path,
                &self.const_pols,
                &self.const_tree,
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
                    &self.prover_buffer_recursive,
                    &options.output_dir_path,
                    false,
                )?;
                timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

                timer_start_info!(GENERATING_FFLONK_SNARK_PROOF);
                let _ = generate_fflonk_snark_proof(&self.pctx, recursivef_proof, &options.output_dir_path);
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

    #[allow(clippy::type_complexity)]
    fn calc_witness_handler(
        &self,
        witness_done: Arc<Counter>,
        memory_handler: Arc<MemoryHandler<F>>,
        minimal_memory: bool,
        stats: bool,
    ) -> (Option<std::thread::JoinHandle<()>>, Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>) {
        let witness_done_clone = witness_done.clone();
        let tx_threads_clone = self.tx_threads.clone();
        let rx_threads_clone = self.rx_threads.clone();
        let pctx_clone = self.pctx.clone();
        let wcm_clone = self.wcm.clone();
        let memory_handler_clone = memory_handler.clone();
        let witness_handles = Arc::new(Mutex::new(Vec::new()));
        let witness_handles_clone = witness_handles.clone();
        let witness_rx = self.witness_rx.clone();
        let witness_rx_priority = self.witness_rx_priority.clone();
        let witness_handler = if !minimal_memory && (cfg!(feature = "gpu") || stats) {
            Some(std::thread::spawn(move || loop {
                let instance_id = match witness_rx_priority.try_recv() {
                    Ok(id) => id,
                    Err(crossbeam_channel::TryRecvError::Empty) => match witness_rx.try_recv() {
                        Ok(id) => {
                            if id == usize::MAX {
                                break;
                            }
                            id
                        }
                        Err(crossbeam_channel::TryRecvError::Empty) => {
                            std::thread::sleep(std::time::Duration::from_micros(100));
                            continue;
                        }
                        Err(crossbeam_channel::TryRecvError::Disconnected) => match witness_rx_priority.try_recv() {
                            Ok(id) => id,
                            Err(_) => break,
                        },
                    },
                    Err(crossbeam_channel::TryRecvError::Disconnected) => match witness_rx.recv() {
                        Ok(id) => {
                            if id == usize::MAX {
                                break;
                            }
                            id
                        }
                        Err(_) => break,
                    },
                };

                let (airgroup_id, air_id) = pctx_clone.dctx_get_instance_info(instance_id);

                let n_threads_witness = pctx_clone.dctx_instance_threads_witness(instance_id);

                let tx_threads_clone: Sender<()> = tx_threads_clone.clone();
                let wcm = wcm_clone.clone();
                let memory_handler_clone = memory_handler_clone.clone();

                let witness_done_clone = witness_done_clone.clone();
                for _ in 0..n_threads_witness {
                    rx_threads_clone.recv().unwrap();
                }

                let pctx_clone = pctx_clone.clone();
                let handle = std::thread::spawn(move || {
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
                    witness_done_clone.increment();
                    if stats {
                        let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance_traces(instance_id);
                        if is_shared_buffer {
                            memory_handler_clone.release_buffer(witness_buffer);
                        }
                    }
                });
                if !stats && cfg!(not(feature = "gpu")) {
                    handle.join().unwrap();
                } else {
                    witness_handles_clone.lock().unwrap().push(handle);
                }
            }))
        } else {
            None
        };
        (witness_handler, witness_handles)
    }

    fn calculate_witness(
        &self,
        instances: &[usize],
        memory_handler: Arc<MemoryHandler<F>>,
        witness_done: Arc<Counter>,
        minimal_memory: bool,
        stats: bool,
    ) {
        timer_start_info!(CALCULATING_WITNESS);

        let mut witness_minimal_memory_handles = Vec::new();
        if !minimal_memory && (cfg!(feature = "gpu") || stats) {
            timer_start_info!(PRE_CALCULATE_WC);
            self.wcm.pre_calculate_witness(1, instances, self.max_num_threads, memory_handler.as_ref());
            timer_stop_and_log_info!(PRE_CALCULATE_WC);
        } else {
            for &instance_id in instances.iter() {
                let n_threads_witness = self.pctx.dctx_instance_threads_witness(instance_id);

                let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(instance_id);
                let threads_to_use_collect = match cfg!(feature = "gpu") || stats {
                    true => (self.pctx.dctx_get_instance_chunks(instance_id) / 16)
                        .max(self.max_num_threads / 4)
                        .min(n_threads_witness)
                        .min(self.max_num_threads),
                    false => self.max_num_threads,
                };

                for _ in 0..threads_to_use_collect {
                    self.rx_threads.recv().unwrap();
                }

                let threads_to_use_witness = match cfg!(feature = "gpu") || stats {
                    true => threads_to_use_collect.min(n_threads_witness),
                    false => self.max_num_threads,
                };

                let threads_to_return = threads_to_use_collect - threads_to_use_witness;

                let pctx_clone = self.pctx.clone();
                let wcm_clone = self.wcm.clone();
                let tx_threads_clone = self.tx_threads.clone();
                let memory_handler_clone = memory_handler.clone();
                let witness_done_clone = witness_done.clone();
                let handle = std::thread::spawn(move || {
                    timer_start_info!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    timer_start_info!(PREPARING_WC, "PREPARING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    wcm_clone.pre_calculate_witness(
                        1,
                        &[instance_id],
                        threads_to_use_collect,
                        memory_handler_clone.as_ref(),
                    );
                    timer_stop_and_log_info!(PREPARING_WC, "PREPARING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    for _ in 0..threads_to_return {
                        tx_threads_clone.send(()).unwrap();
                    }
                    timer_start_info!(COMPUTING_WC, "COMPUTING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    wcm_clone.calculate_witness(
                        1,
                        &[instance_id],
                        threads_to_use_witness,
                        memory_handler_clone.as_ref(),
                    );
                    timer_stop_and_log_info!(COMPUTING_WC, "COMPUTING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    for _ in 0..threads_to_use_witness {
                        tx_threads_clone.send(()).unwrap();
                    }
                    timer_stop_and_log_info!(
                        GENERATING_WC,
                        "GENERATING_WC_{} [{}:{}]",
                        instance_id,
                        airgroup_id,
                        air_id
                    );
                    witness_done_clone.increment();
                    if stats {
                        let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance_traces(instance_id);
                        if is_shared_buffer {
                            memory_handler_clone.release_buffer(witness_buffer);
                        }
                    }
                });
                if !stats && cfg!(not(feature = "gpu")) {
                    handle.join().unwrap();
                } else {
                    witness_minimal_memory_handles.push(handle);
                }
            }
        }

        witness_done.wait_until_value_and_check_streams(instances.len(), || {
            get_stream_proofs_non_blocking_c(self.d_buffers.get_ptr())
        });

        for handle in witness_minimal_memory_handles {
            handle.join().unwrap();
        }

        timer_stop_and_log_info!(CALCULATING_WITNESS);
    }

    fn prepare_gpu(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        setups_vadcop: &SetupsVadcop<F>,
        aggregation: bool,
        gpu_params: &ParamsGPU,
    ) -> (Arc<DeviceBuffer>, u64, u64, u64) {
        let mut free_memory_gpu = match cfg!(feature = "gpu") {
            true => {
                check_device_memory_c(pctx.dctx_get_node_rank() as u32, pctx.dctx_get_node_n_processes() as u32) as f64
                    * 0.98
            }
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

        let mut gpu_available_memory = match cfg!(feature = "gpu") {
            true => max_size_buffer as i64 - (n_streams_per_gpu * sctx.max_prover_buffer_size) as i64,
            false => 0,
        };
        let mut n_recursive_streams_per_gpu = 0;
        if aggregation {
            while gpu_available_memory > 0 {
                gpu_available_memory -= setups_vadcop.max_prover_recursive_buffer_size as i64;
                if gpu_available_memory < 0 {
                    break;
                }
                n_recursive_streams_per_gpu += 1;
            }
        }

        let max_aux_trace_area = (n_streams_per_gpu * sctx.max_prover_buffer_size
            + n_recursive_streams_per_gpu * setups_vadcop.max_prover_recursive_buffer_size)
            as u64;

        let max_sizes = MaxSizes { total_const_area, max_aux_trace_area, total_const_area_aggregation };

        let max_sizes_ptr = &max_sizes as *const MaxSizes as *mut c_void;
        let d_buffers = Arc::new(DeviceBuffer(gen_device_buffers_c(
            max_sizes_ptr,
            pctx.dctx_get_node_rank() as u32,
            pctx.dctx_get_node_n_processes() as u32,
        )));

        let max_pinned_proof_size = match aggregation {
            true => sctx.max_pinned_proof_size.max(setups_vadcop.max_pinned_proof_size) as u64,
            false => sctx.max_pinned_proof_size as u64,
        };

        let n_gpus: u64 = gen_device_streams_c(
            d_buffers.get_ptr(),
            sctx.max_prover_buffer_size as u64,
            setups_vadcop.max_prover_recursive_buffer_size as u64,
            max_pinned_proof_size,
            n_streams_per_gpu as u64,
            n_recursive_streams_per_gpu as u64,
            sctx.max_n_bits_ext as u64,
        );

        (d_buffers, n_streams_per_gpu as u64, n_recursive_streams_per_gpu as u64, n_gpus)
    }

    #[allow(clippy::too_many_arguments)]
    fn gen_proof(
        proofs: &[RwLock<Option<Proof<F>>>],
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        output_dir_path: &Path,
        aux_trace: &[F],
        const_pols: &[F],
        const_tree: &[F],
        d_buffers: &DeviceBuffer,
        stream_id_: Option<usize>,
        save_proof: bool,
        gpu_preallocate: bool,
    ) {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
        timer_start_info!(GEN_PROOF, "GEN_PROOF_{} [{}:{}]", instance_id, airgroup_id, air_id);
        Self::initialize_air_instance(pctx, sctx, instance_id, false, false);

        let setup = sctx.get_setup(airgroup_id, air_id);
        let p_setup: *mut c_void = (&setup.p_setup).into();
        let air_instance_name = &pctx.global_info.airs[airgroup_id][air_id].name;

        let mut steps_params = pctx.get_air_instance_params(instance_id, true);

        if cfg!(not(feature = "gpu")) {
            steps_params.aux_trace = aux_trace.as_ptr() as *mut u8;
            steps_params.p_const_pols = const_pols.as_ptr() as *mut u8;
            steps_params.p_const_tree = const_tree.as_ptr() as *mut u8;
        } else if !gpu_preallocate {
            steps_params.p_const_pols = std::ptr::null_mut();
            steps_params.p_const_tree = std::ptr::null_mut();
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

        let proof = create_buffer_fast(setup.proof_size as usize);
        *proofs[instance_id].write().unwrap() =
            Some(Proof::new(ProofType::Basic, airgroup_id, air_id, Some(instance_id), proof));

        gen_proof_c(
            p_setup,
            p_steps_params,
            pctx.get_global_challenge_ptr(),
            proofs[instance_id].read().unwrap().as_ref().unwrap().proof.as_ptr() as *mut u64,
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

        if cfg!(not(feature = "gpu")) {
            launch_callback_c(instance_id as u64, "basic");
        }

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
            proving_key_path,
            custom_commits_fixed,
            aggregation,
            final_snark,
            verbose_mode,
            mpi_universe,
        );
        timer_start_info!(INITIALIZING_PROOFMAN);

        let sctx: Arc<SetupCtx<F>> =
            Arc::new(SetupCtx::new(&pctx.global_info, &ProofType::Basic, verify_constraints, gpu_params));
        pctx.set_weights(&sctx);

        let pctx = Arc::new(pctx);
        if !verify_constraints {
            check_tree_paths(&pctx, &sctx)?;
        }

        let setups_vadcop =
            Arc::new(SetupsVadcop::new(&pctx.global_info, verify_constraints, aggregation, final_snark, gpu_params));

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
        let mut pctx =
            ProofCtx::create_ctx(proving_key_path, custom_commits_fixed, aggregation, final_snark, verbose_mode);
        timer_start_info!(INITIALIZING_PROOFMAN);

        let sctx: Arc<SetupCtx<F>> =
            Arc::new(SetupCtx::new(&pctx.global_info, &ProofType::Basic, verify_constraints, gpu_params));
        pctx.set_weights(&sctx);

        let pctx = Arc::new(pctx);
        if !verify_constraints {
            check_tree_paths(&pctx, &sctx)?;
        }
        Self::initialize_publics_custom_commits(&sctx, &pctx)?;

        let setups_vadcop =
            Arc::new(SetupsVadcop::new(&pctx.global_info, verify_constraints, aggregation, final_snark, gpu_params));

        if aggregation {
            check_tree_paths_vadcop(&pctx, &setups_vadcop, final_snark)?;
            initialize_witness_circom(&pctx, &setups_vadcop, final_snark)?;
        }

        timer_stop_and_log_info!(INITIALIZING_PROOFMAN);

        Ok((pctx, sctx, setups_vadcop))
    }

    fn calculate_internal_contributions(&self) -> [u64; 10] {
        timer_start_info!(CALCULATE_INTERNAL_CONTRIBUTION);
        let my_instances = self.pctx.dctx_get_my_instances();

        let mut values = vec![vec![F::ZERO; 10]; my_instances.len()];

        for (idx, instance_id) in my_instances.iter().enumerate() {
            let mut contribution = vec![F::ZERO; 10];

            let root_contribution = self.roots_contributions[*instance_id];

            let values_to_hash =
                &mut self.values_contributions[*instance_id].lock().expect("Missing values_contribution");
            values_to_hash[4..8].copy_from_slice(&root_contribution[..4]);

            calculate_hash_c(
                contribution.as_mut_ptr() as *mut u8,
                values_to_hash.as_mut_ptr() as *mut u8,
                values_to_hash.len() as u64,
                10,
            );

            for (i, v) in contribution.iter().enumerate().take(10) {
                values[idx][i] = *v;
            }
        }

        let partial_contribution = self.add_contributions(&values);

        let partial_contribution_u64: [u64; 10] = partial_contribution
            .iter()
            .map(|&x| x.as_canonical_u64())
            .collect::<Vec<u64>>()
            .try_into()
            .expect("Expected exactly 10 elements");

        timer_stop_and_log_info!(CALCULATE_INTERNAL_CONTRIBUTION);

        partial_contribution_u64
    }

    fn calculate_global_challenge(&self, all_partial_contributions_u64: &[[u64; 10]]) {
        timer_start_info!(CALCULATE_GLOBAL_CHALLENGE);

        let transcript = FFITranscript::new(2, true);

        transcript.add_elements(self.pctx.get_publics_ptr(), self.pctx.global_info.n_publics);

        let proof_values_stage = self.pctx.get_proof_values_by_stage(1);
        if !proof_values_stage.is_empty() {
            transcript.add_elements(proof_values_stage.as_ptr() as *mut u8, proof_values_stage.len());
        }

        let all_partial_contributions: Vec<Vec<F>> =
            all_partial_contributions_u64.iter().map(|&arr| arr.iter().map(|&x| F::from_u64(x)).collect()).collect();

        let value = self.aggregate_contributions(&all_partial_contributions);
        transcript.add_elements(value.as_ptr() as *mut u8, value.len());

        let global_challenge = [F::ZERO; 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);

        tracing::info!(
            "··· Global challenge: [{}, {}, {}]",
            global_challenge[0],
            global_challenge[1],
            global_challenge[2]
        );
        self.pctx.set_global_challenge(2, &global_challenge);

        timer_stop_and_log_info!(CALCULATE_GLOBAL_CHALLENGE);
    }

    #[allow(dead_code)]
    fn diagnostic_instance(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, instance_id: usize) -> bool {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;
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
                    col.name.to_string()
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

    fn initialize_air_instance(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        init_aux_trace: bool,
        verify_constraints: bool,
    ) {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let mut air_instance = pctx.air_instances[instance_id].write().unwrap();

        assert!(
            air_instance.num_rows == (1 << setup.stark_info.stark_struct.n_bits),
            "Row count mismatch for airgroup_id={}, air_id={}: expected {} rows (from proving key), but got {} rows (from pil-helpers).",
            airgroup_id,
            air_id,
            1 << setup.stark_info.stark_struct.n_bits,
            air_instance.num_rows
        );

        if init_aux_trace {
            air_instance.init_aux_trace(setup.prover_buffer_size as usize);
        }
        air_instance.init_evals(setup.stark_info.ev_map.len() * 3);
        air_instance.init_challenges(
            (setup.stark_info.challenges_map.as_ref().unwrap().len() + setup.stark_info.stark_struct.steps.len() + 1)
                * 3,
        );

        if verify_constraints {
            let const_pols: Vec<F> = create_buffer_fast(setup.const_pols_size);
            load_const_pols(&setup.setup_path, setup.const_pols_size, &const_pols);
            air_instance.init_fixed(const_pols);
        }
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

        let steps_params = pctx.get_air_instance_params(instance_id, false);

        calculate_impols_expressions_c((&setup.p_setup).into(), stage as u64, (&steps_params).into());
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_contribution_air(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        roots_contributions: &[[F; 4]],
        values_contributions: &[Mutex<Vec<F>>],
        instance_id: usize,
        aux_trace_contribution_ptr: *mut u8,
        d_buffers: &DeviceBuffer,
        streams: &Mutex<Vec<Option<u64>>>,
    ) {
        let n_field_elements = 4;
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id);

        timer_start_info!(GET_CONTRIBUTION_AIR, "GET_CONTRIBUTION_AIR_{} [{}:{}]", instance_id, airgroup_id, air_id);

        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let air_values = &pctx.get_air_instance_air_values(airgroup_id, air_id, air_instance_id);

        let stream_id = commit_witness_c(
            3,
            setup.stark_info.stark_struct.n_bits,
            setup.stark_info.stark_struct.n_bits_ext,
            *setup.stark_info.map_sections_n.get("cm1").unwrap(),
            instance_id as u64,
            airgroup_id as u64,
            air_id as u64,
            roots_contributions[instance_id].as_ptr() as *mut u8,
            pctx.get_air_instance_trace_ptr(instance_id),
            aux_trace_contribution_ptr,
            d_buffers.get_ptr(),
            (&setup.p_setup).into(),
        );
        if !setup.single_instance {
            streams.lock().unwrap()[stream_id as usize] = Some(instance_id as u64);
        } else {
            streams.lock().unwrap()[stream_id as usize] = None;
        }

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

    fn add_contributions(&self, values: &[Vec<F>]) -> Vec<F> {
        if self.pctx.global_info.curve == CurveType::EcGFp5 {
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

    fn aggregate_contributions(&self, values: &[Vec<F>]) -> Vec<F> {
        if self.pctx.global_info.curve == CurveType::EcGFp5 {
            let mut result = EcGFp5::new(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcGFp5::new(
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
            let mut result = EcMasFp5::new(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcMasFp5::new(
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
