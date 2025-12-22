use std::{collections::HashMap, sync::RwLock};
use std::path::PathBuf;
use std::sync::Arc;
use crate::{MpiCtx, ProofmanError};
use borsh::{BorshDeserialize, BorshSerialize};
use std::fs::File;
use std::io::Read;
use std::fs;
use fields::{PrimeField64, Transcript, Poseidon16};
use proofman_starks_lib_c::custom_commit_size_c;
use crate::{
    initialize_logger, AirInstance, DistributionCtx, GlobalInfo, InstanceInfo, SetupCtx, StdMode, StepsParams,
    VerboseMode, ProofmanResult,
};

#[derive(Debug)]
pub struct Values<F> {
    pub values: RwLock<Vec<F>>,
}

impl<F: PrimeField64> Values<F> {
    pub fn new(n_values: usize) -> Self {
        Self { values: RwLock::new(vec![F::ZERO; n_values]) }
    }
}

impl<F> Default for Values<F> {
    fn default() -> Self {
        Self { values: RwLock::new(Vec::new()) }
    }
}

pub type AirGroupMap = HashMap<usize, AirIdMap>;
pub type AirIdMap = HashMap<usize, InstanceMap>;
pub type InstanceMap = HashMap<usize, Vec<usize>>;

pub const DEFAULT_N_PRINT_CONSTRAINTS: usize = 10;

#[derive(Clone)]
pub struct ProofOptions {
    pub verify_constraints: bool,
    pub aggregation: bool,
    pub rma: bool,
    pub final_snark: bool,
    pub verify_proofs: bool,
    pub save_proofs: bool,
    pub test_mode: bool,
    pub output_dir_path: PathBuf,
    pub minimal_memory: bool,
}

impl BorshSerialize for ProofOptions {
    fn serialize<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        BorshSerialize::serialize(&self.verify_constraints, writer)?;
        BorshSerialize::serialize(&self.aggregation, writer)?;
        BorshSerialize::serialize(&self.rma, writer)?;
        BorshSerialize::serialize(&self.final_snark, writer)?;
        BorshSerialize::serialize(&self.verify_proofs, writer)?;
        BorshSerialize::serialize(&self.save_proofs, writer)?;
        BorshSerialize::serialize(&self.test_mode, writer)?;
        BorshSerialize::serialize(&self.output_dir_path.to_string_lossy().to_string(), writer)?;
        BorshSerialize::serialize(&self.minimal_memory, writer)?;
        Ok(())
    }
}

impl BorshDeserialize for ProofOptions {
    fn deserialize_reader<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let verify_constraints = bool::deserialize_reader(reader)?;
        let aggregation = bool::deserialize_reader(reader)?;
        let rma = bool::deserialize_reader(reader)?;
        let final_snark = bool::deserialize_reader(reader)?;
        let verify_proofs = bool::deserialize_reader(reader)?;
        let save_proofs = bool::deserialize_reader(reader)?;
        let test_mode = bool::deserialize_reader(reader)?;
        let output_dir_path_str = String::deserialize_reader(reader)?;
        let minimal_memory = bool::deserialize_reader(reader)?;

        Ok(Self {
            verify_constraints,
            aggregation,
            rma,
            final_snark,
            verify_proofs,
            save_proofs,
            test_mode,
            output_dir_path: PathBuf::from(output_dir_path_str),
            minimal_memory,
        })
    }
}

#[derive(Clone)]
pub struct DebugInfo {
    pub debug_instances: AirGroupMap,
    pub debug_global_instances: Vec<usize>,
    pub std_mode: StdMode,
    pub n_print_constraints: usize,
}

impl Default for DebugInfo {
    fn default() -> Self {
        Self {
            debug_instances: Default::default(),
            debug_global_instances: Default::default(),
            std_mode: Default::default(),
            n_print_constraints: DEFAULT_N_PRINT_CONSTRAINTS,
        }
    }
}

impl DebugInfo {
    pub fn new_debug() -> Self {
        Self {
            debug_instances: HashMap::new(),
            debug_global_instances: Vec::new(),
            std_mode: StdMode::new_debug(),
            n_print_constraints: DEFAULT_N_PRINT_CONSTRAINTS,
        }
    }
}
impl ProofOptions {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        verify_constraints: bool,
        aggregation: bool,
        rma: bool,
        final_snark: bool,
        verify_proofs: bool,
        minimal_memory: bool,
        save_proofs: bool,
        output_dir_path: PathBuf,
    ) -> Self {
        Self {
            verify_constraints,
            aggregation,
            rma,
            final_snark,
            verify_proofs,
            minimal_memory,
            save_proofs,
            output_dir_path,
            test_mode: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_test(
        verify_constraints: bool,
        aggregation: bool,
        rma: bool,
        final_snark: bool,
        verify_proofs: bool,
        minimal_memory: bool,
        save_proofs: bool,
        output_dir_path: PathBuf,
    ) -> Self {
        Self {
            verify_constraints,
            aggregation,
            rma,
            final_snark,
            verify_proofs,
            save_proofs,
            minimal_memory,
            output_dir_path,
            test_mode: true,
        }
    }
}

#[derive(Clone)]
pub struct ParamsGPU {
    pub preallocate: bool,
    pub max_number_streams: usize,
    pub number_threads_pools_witness: usize,
    pub are_threads_per_witness_set: bool,
    pub max_witness_stored: usize,
    pub pack_trace: bool,
}

impl Default for ParamsGPU {
    fn default() -> Self {
        Self {
            preallocate: false,
            max_number_streams: 20,
            number_threads_pools_witness: 4,
            #[cfg(feature = "packed")]
            max_witness_stored: 10,
            #[cfg(not(feature = "packed"))]
            max_witness_stored: 4,
            pack_trace: true,
            are_threads_per_witness_set: false,
        }
    }
}

impl ParamsGPU {
    pub fn new(preallocate: bool) -> Self {
        Self { preallocate, ..Self::default() }
    }

    pub fn with_max_number_streams(&mut self, max_number_streams: usize) {
        self.max_number_streams = max_number_streams;
    }

    pub fn with_number_threads_pools_witness(&mut self, number_threads_pools_witness: usize) {
        self.number_threads_pools_witness = number_threads_pools_witness;
        self.are_threads_per_witness_set = true;
    }
    pub fn with_max_witness_stored(&mut self, max_witness_stored: usize) {
        self.max_witness_stored = max_witness_stored;
    }

    pub fn with_pack_trace(&mut self, pack_trace: bool) {
        self.pack_trace = pack_trace;
    }
}

#[allow(dead_code)]
pub struct ProofCtx<F: PrimeField64> {
    pub mpi_ctx: Arc<MpiCtx>,
    pub public_inputs: Values<F>,
    pub proof_values: Values<F>,
    pub global_challenge: Values<F>,
    pub challenges: Values<F>,
    pub global_info: GlobalInfo,
    pub air_instances: Vec<RwLock<AirInstance<F>>>,
    pub weights: HashMap<(usize, usize), u64>,
    pub custom_commits_fixed: HashMap<String, PathBuf>,
    pub custom_commits_values: HashMap<String, Vec<u8>>,
    pub dctx: RwLock<DistributionCtx>,
    pub debug_info: RwLock<DebugInfo>,
    pub aggregation: bool,
    pub proof_tx: RwLock<Option<crossbeam_channel::Sender<usize>>>,
    pub witness_tx: RwLock<Option<crossbeam_channel::Sender<usize>>>,
    pub witness_tx_priority: RwLock<Option<crossbeam_channel::Sender<usize>>>,
}

pub const MAX_INSTANCES: u64 = 10000;

impl<F: PrimeField64> ProofCtx<F> {
    pub fn create_ctx(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        aggregation: bool,
        verbose_mode: VerboseMode,
        mpi_ctx: Arc<MpiCtx>,
    ) -> ProofmanResult<Self> {
        tracing::info!("Creating proof context");

        let mut dctx = DistributionCtx::new();

        dctx.setup_processes(mpi_ctx.n_processes as usize, mpi_ctx.rank as usize)?;

        initialize_logger(verbose_mode, None);
        let global_info: GlobalInfo = GlobalInfo::new(&proving_key_path)?;
        let n_publics = global_info.n_publics;
        let n_proof_values = global_info
            .proof_values_map
            .as_ref()
            .map(|map| map.iter().filter(|entry| entry.stage == 1).count())
            .unwrap_or(0);
        let n_challenges = global_info.n_challenges.iter().sum::<usize>();

        let weights = HashMap::new();

        let air_instances: Vec<RwLock<AirInstance<F>>> =
            (0..MAX_INSTANCES).map(|_| RwLock::new(AirInstance::<F>::default())).collect();

        Ok(Self {
            mpi_ctx,
            global_info,
            public_inputs: Values::new(n_publics),
            proof_values: Values::new(n_proof_values),
            challenges: Values::new(n_challenges * 3),
            global_challenge: Values::new(3),
            air_instances,
            dctx: RwLock::new(dctx),
            debug_info: RwLock::new(DebugInfo::default()),
            custom_commits_fixed,
            custom_commits_values: HashMap::new(),
            weights,
            aggregation,
            witness_tx: RwLock::new(None),
            witness_tx_priority: RwLock::new(None),
            proof_tx: RwLock::new(None),
        })
    }

    pub fn set_debug_info(&self, debug_info: &DebugInfo) {
        let mut debug_info_guard = self.debug_info.write().unwrap();
        *debug_info_guard = debug_info.clone();
    }

    pub fn dctx_reset(&self) {
        let mut dctx = self.dctx.write().unwrap();
        dctx.reset_instances();
        self.mpi_ctx.reset();
    }

    pub fn set_proof_tx(&self, proof_tx: Option<crossbeam_channel::Sender<usize>>) {
        *self.proof_tx.write().unwrap() = proof_tx;
    }

    pub fn set_witness_tx_priority(&self, witness_tx_priority: Option<crossbeam_channel::Sender<usize>>) {
        *self.witness_tx_priority.write().unwrap() = witness_tx_priority;
    }

    pub fn set_witness_tx(&self, witness_tx: Option<crossbeam_channel::Sender<usize>>) {
        *self.witness_tx.write().unwrap() = witness_tx;
    }

    pub fn set_witness_ready(&self, global_id: usize, priority: bool) {
        if priority {
            if let Some(witness_tx_priority) = &*self.witness_tx_priority.read().unwrap() {
                witness_tx_priority.send(global_id).unwrap();
                return;
            }
        }
        if let Some(witness_tx) = &*self.witness_tx.read().unwrap() {
            witness_tx.send(global_id).unwrap();
        }
    }

    pub fn initialize_custom_commits(&mut self, sctx: &SetupCtx<F>) -> ProofmanResult<()> {
        tracing::info!("Initializing publics custom_commits");
        for (airgroup_id, airs) in self.global_info.airs.iter().enumerate() {
            for (air_id, _) in airs.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id)?;
                for (commit_id, custom_commit) in setup.stark_info.custom_commits.iter().enumerate() {
                    if custom_commit.stage_widths[0] > 0 {
                        let custom_file_path = self.get_custom_commits_fixed_buffer(&custom_commit.name, true)?;

                        if !PathBuf::from(&custom_file_path).exists() {
                            let error_message = format!(
                                "Error: Unable to find {} custom commit at '{}'.\n\
                                Please run the following command:\n\
                                \x1b[1mcargo run --bin proofman-cli gen-custom-commits-fixed --witness-lib <WITNESS_LIB> --proving-key <PROVING_KEY> --custom-commits <CUSTOM_COMMITS_DIR> \x1b[0m",
                                custom_commit.name,
                                custom_file_path.display(),
                            );
                            tracing::warn!("{}", error_message);
                            return Err(ProofmanError::ProofmanError(error_message));
                        }

                        let error_message = format!(
                            "Error: The custom commit file for {} at '{}' exists but is invalid or corrupted.\n\
                            Please regenerate it by running:\n\
                            \x1b[1mcargo run --bin proofman-cli gen-custom-commits-fixed --witness-lib <WITNESS_LIB> --proving-key <PROVING_KEY> --custom-commits <CUSTOM_COMMITS_DIR> \x1b[0m",
                            custom_commit.name,
                            custom_file_path.display(),
                        );

                        let size = custom_commit_size_c((&setup.p_setup).into(), commit_id as u64) as usize;

                        match fs::metadata(&custom_file_path) {
                            Ok(metadata) => {
                                let actual_size = metadata.len() as usize;
                                if actual_size != (size + 4) * 8 {
                                    tracing::warn!("{}", error_message);
                                    return Err(ProofmanError::ProofmanError(error_message));
                                }
                            }
                            Err(err) => {
                                let error_message = format!(
                                    "Failed to open {} for custom_commit {}: {}",
                                    setup.air_name, custom_commit.name, err
                                );
                                tracing::warn!("{}", error_message);
                                return Err(ProofmanError::ProofmanError(error_message));
                            }
                        }

                        let mut file = File::open(&custom_file_path)?;
                        let mut root_bytes = [0u8; 32];
                        file.read_exact(&mut root_bytes)?;

                        self.custom_commits_values.insert(custom_commit.name.clone(), root_bytes.to_vec());
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_custom_commit_root(&self, name: &str) -> ProofmanResult<&[u8]> {
        let root_bytes = self.custom_commits_values.get(name);
        match root_bytes {
            Some(bytes) => Ok(bytes.as_slice()),
            None => Err(ProofmanError::ProofmanError(format!("Custom Commit {name} not found"))),
        }
    }

    pub fn set_weights(&mut self, sctx: &SetupCtx<F>) -> ProofmanResult<()> {
        for (airgroup_id, air_group) in self.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id)?;
                let mut total_cols = setup
                    .stark_info
                    .map_sections_n
                    .iter()
                    .filter(|(key, _)| *key != "const")
                    .map(|(_, value)| *value)
                    .sum::<u64>();
                total_cols += 3; // FRI polinomial
                let n_openings = setup.stark_info.opening_points.len() as u64;
                // let n_ops_quotient = setup.n_operations_quotient;
                let weight = (total_cols + n_openings * 3) * (1 << (setup.stark_info.stark_struct.n_bits_ext));
                // weight += (n_ops_quotient / 10) * (1 << (setup.stark_info.stark_struct.n_bits_ext));
                self.weights.insert((airgroup_id, air_id), weight);
            }
        }
        Ok(())
    }

    pub fn get_weight(&self, airgroup_id: usize, air_id: usize) -> u64 {
        *self.weights.get(&(airgroup_id, air_id)).unwrap()
    }

    pub fn get_custom_commits_fixed_buffer(&self, name: &str, return_error: bool) -> ProofmanResult<PathBuf> {
        let file_name = self.custom_commits_fixed.get(name);
        match file_name {
            Some(path) => Ok(path.to_path_buf()),
            None => {
                if return_error {
                    Err(ProofmanError::ProofmanError(format!("Custom Commit Fixed {file_name:?} not found")))
                } else {
                    tracing::warn!("Custom Commit Fixed {file_name:?} not found");
                    Ok(PathBuf::new())
                }
            }
        }
    }

    pub fn add_air_instance(&self, air_instance: AirInstance<F>, global_idx: usize) {
        *self.air_instances[global_idx].write().unwrap() = air_instance;
        if let Some(proof_tx) = &*self.proof_tx.read().unwrap() {
            proof_tx.send(global_idx).unwrap();
        }
    }

    pub fn is_air_instance_stored(&self, global_idx: usize) -> bool {
        !self.air_instances[global_idx].read().unwrap().trace.is_empty()
    }

    pub fn dctx_get_instances(&self) -> Vec<InstanceInfo> {
        let dctx = self.dctx.read().unwrap();
        dctx.instances.clone()
    }

    pub fn dctx_get_worker_instances(&self) -> Vec<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.worker_instances.clone()
    }

    pub fn dctx_is_first_partition(&self) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.partition_mask[0]
    }

    pub fn dctx_reset_instances_calculated(&self) {
        let dctx = self.dctx.read().unwrap();
        for instance in dctx.instances_calculated.iter() {
            instance.store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn dctx_set_instance_calculated(&self, global_idx: usize) {
        let dctx = self.dctx.read().unwrap();
        dctx.instances_calculated[global_idx].store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn dctx_is_instance_calculated(&self, global_idx: usize) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.instances_calculated[global_idx].load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn dctx_get_my_tables(&self) -> Vec<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.instances
            .iter()
            .enumerate()
            .filter(|(id, inst)| inst.table && (dctx.process_instances.contains(id) || inst.shared))
            .map(|(id, _)| id)
            .collect()
    }

    pub fn dctx_get_process_instances(&self) -> Vec<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.process_instances.clone()
    }

    pub fn dctx_get_process_owner_instance(&self, instance_id: usize) -> ProofmanResult<i32> {
        let dctx = self.dctx.read().unwrap();
        dctx.get_process_owner_instance(instance_id)
    }

    pub fn dctx_get_instance_info(&self, global_idx: usize) -> ProofmanResult<(usize, usize)> {
        let dctx = self.dctx.read().unwrap();
        dctx.get_instance_info(global_idx)
    }

    pub fn dctx_get_instance_chunks(&self, global_idx: usize) -> ProofmanResult<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.get_instance_chunks(global_idx)
    }

    pub fn dctx_get_instance_local_idx(&self, global_idx: usize) -> ProofmanResult<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.get_instance_local_idx(global_idx)
    }

    pub fn dctx_is_my_process_instance(&self, global_idx: usize) -> ProofmanResult<bool> {
        let dctx = self.dctx.read().unwrap();
        dctx.is_my_process_instance(global_idx)
    }

    pub fn dctx_is_table(&self, global_idx: usize) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.instances[global_idx].table
    }

    pub fn is_shared_buffer(&self, global_idx: usize) -> bool {
        self.air_instances[global_idx].read().unwrap().is_shared_buffer()
    }

    pub fn dctx_find_air_instance_id(&self, global_idx: usize) -> ProofmanResult<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.find_air_instance_id(global_idx)
    }

    pub fn dctx_find_process_instance(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<(bool, usize)> {
        let dctx = self.dctx.read().unwrap();
        dctx.find_process_instance(airgroup_id, air_id)
    }

    pub fn dctx_find_process_table(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<(bool, usize)> {
        let dctx = self.dctx.read().unwrap();
        dctx.find_process_table(airgroup_id, air_id)
    }

    pub fn dctx_get_table_instance_idx(&self, table_idx: usize) -> ProofmanResult<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.get_table_instance_idx(table_idx)
    }

    pub fn dctx_set_chunks(&self, global_idx: usize, chunks: Vec<usize>, slow: bool) {
        let mut dctx = self.dctx.write().unwrap();
        dctx.set_chunks(global_idx, chunks, slow);
    }

    pub fn add_instance_assign(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<usize> {
        let mut dctx = self.dctx.write().unwrap();
        let weight = self.get_weight(airgroup_id, air_id);
        dctx.add_instance(airgroup_id, air_id, weight)
    }

    pub fn add_instance_assign_first_partition(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<usize> {
        let mut dctx = self.dctx.write().unwrap();
        let weight = self.get_weight(airgroup_id, air_id);
        dctx.add_instance_first_partition(airgroup_id, air_id, weight)
    }

    pub fn add_instance(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<usize> {
        let mut dctx = self.dctx.write().unwrap();
        let weight = self.get_weight(airgroup_id, air_id);
        dctx.add_instance_no_assign(airgroup_id, air_id, weight)
    }

    pub fn add_table(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<usize> {
        let mut dctx = self.dctx.write().unwrap();
        let weight = self.get_weight(airgroup_id, air_id);
        dctx.add_table(airgroup_id, air_id, weight)
    }

    pub fn add_table_all(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<usize> {
        let mut dctx = self.dctx.write().unwrap();
        let weight = self.get_weight(airgroup_id, air_id);
        dctx.add_table_all(airgroup_id, air_id, weight)
    }

    pub fn dctx_add_instance_no_assign(&self, airgroup_id: usize, air_id: usize, weight: u64) -> ProofmanResult<usize> {
        let mut dctx = self.dctx.write().unwrap();
        dctx.add_instance_no_assign(airgroup_id, air_id, weight)
    }

    pub fn dctx_assign_instances(&self) -> ProofmanResult<()> {
        let mut dctx = self.dctx.write().unwrap();
        dctx.assign_instances()
    }

    pub fn dctx_load_balance_info_process(&self) -> (f64, u64, u64, f64) {
        let dctx = self.dctx.read().unwrap();
        dctx.load_balance_info_process()
    }
    pub fn dctx_load_balance_info_partition(&self) -> (f64, u64, u64, f64) {
        let dctx = self.dctx.read().unwrap();
        dctx.load_balance_info_partition()
    }

    pub fn dctx_setup(&self, n_partitions: usize, partition_ids: Vec<u32>, worker_index: usize) -> ProofmanResult<()> {
        let mut dctx = self.dctx.write().unwrap();
        dctx.setup_partitions(n_partitions, partition_ids)?;
        dctx.setup_worker_index(worker_index);
        Ok(())
    }

    pub fn get_n_partitions(&self) -> usize {
        let dctx = self.dctx.read().unwrap();
        dctx.n_partitions
    }

    pub fn get_worker_index(&self) -> ProofmanResult<usize> {
        let dctx = self.dctx.read().unwrap();
        if dctx.worker_index < 0 {
            return Err(ProofmanError::InvalidAssignation("Worker index not set".into()));
        }
        Ok(dctx.worker_index as usize)
    }

    pub fn get_proof_values_ptr(&self) -> *mut u8 {
        let guard = &self.proof_values.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn set_public_value(&self, value: u64, public_id: usize) {
        self.public_inputs.values.write().unwrap()[public_id] = F::from_u64(value);
    }

    pub fn set_global_challenge(&self, stage: usize, global_challenge: &mut [F]) {
        let mut global_challenge_guard = self.global_challenge.values.write().unwrap();
        global_challenge_guard[0] = global_challenge[0];
        global_challenge_guard[1] = global_challenge[1];
        global_challenge_guard[2] = global_challenge[2];

        let mut transcript: Transcript<F, Poseidon16, 16> = Transcript::new();

        transcript.put(global_challenge);
        let mut challenges_guard = self.challenges.values.write().unwrap();

        let initial_pos = self.global_info.n_challenges.iter().take(stage - 1).sum::<usize>();
        let num_challenges = self.global_info.n_challenges[stage - 1];
        for i in 0..num_challenges {
            transcript.get_field(&mut challenges_guard[(initial_pos + i) * 3..(initial_pos + i) * 3 + 3]);
        }
    }

    pub fn set_challenge(&self, index: usize, challenge: &[F]) {
        let mut challenges_guard = self.challenges.values.write().unwrap();
        challenges_guard[index] = challenge[0];
        challenges_guard[index + 1] = challenge[1];
        challenges_guard[index + 2] = challenge[2];
    }

    pub fn get_publics(&self) -> std::sync::RwLockWriteGuard<'_, Vec<F>> {
        self.public_inputs.values.write().unwrap()
    }

    pub fn get_proof_values(&self) -> std::sync::RwLockWriteGuard<'_, Vec<F>> {
        self.proof_values.values.write().unwrap()
    }

    pub fn get_proof_values_by_stage(&self, stage: u32) -> Vec<F> {
        let proof_vals = self.proof_values.values.read().unwrap();

        let mut values = Vec::new();
        let mut p = 0;
        for proof_value in self.global_info.proof_values_map.as_ref().unwrap() {
            if proof_value.stage > stage as u64 {
                break;
            }
            if proof_value.stage == 1 {
                if stage == 1 {
                    values.push(proof_vals[p]);
                }
                p += 1;
            } else {
                if proof_value.stage == stage as u64 {
                    values.push(proof_vals[p]);
                    values.push(proof_vals[p + 1]);
                    values.push(proof_vals[p + 2]);
                }
                p += 3;
            }
        }

        values
    }

    pub fn get_publics_ptr(&self) -> *mut u8 {
        let guard = &self.public_inputs.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn get_challenges(&self) -> std::sync::RwLockWriteGuard<'_, Vec<F>> {
        self.challenges.values.write().unwrap()
    }

    pub fn get_challenges_ptr(&self) -> *mut u8 {
        let guard = &self.challenges.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn get_global_challenge(&self) -> std::sync::RwLockWriteGuard<'_, Vec<F>> {
        self.global_challenge.values.write().unwrap()
    }

    pub fn get_global_challenge_ptr(&self) -> *mut u8 {
        let guard = &self.global_challenge.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn get_air_instance_params(&self, instance_id: usize, gen_proof: bool) -> StepsParams {
        let air_instance = self.air_instances[instance_id].read().unwrap();

        let challenges = if gen_proof { air_instance.get_challenges_ptr() } else { self.get_challenges_ptr() };
        let aux_trace: *mut u8 = if gen_proof { std::ptr::null_mut() } else { air_instance.get_aux_trace_ptr() };
        let const_pols: *mut u8 = if gen_proof { std::ptr::null_mut() } else { air_instance.get_fixed_ptr() };

        StepsParams {
            trace: air_instance.get_trace_ptr(),
            aux_trace,
            public_inputs: self.get_publics_ptr(),
            proof_values: self.get_proof_values_ptr(),
            challenges,
            airgroup_values: air_instance.get_airgroup_values_ptr(),
            airvalues: air_instance.get_airvalues_ptr(),
            evals: air_instance.get_evals_ptr(),
            xdivxsub: std::ptr::null_mut(),
            p_const_pols: const_pols,
            p_const_tree: std::ptr::null_mut(),
            custom_commits_fixed: air_instance.get_custom_commits_fixed_ptr(),
        }
    }

    pub fn get_air_instance_trace_ptr(&self, instance_id: usize) -> *mut u8 {
        self.air_instances[instance_id].read().unwrap().get_trace_ptr()
    }

    pub fn get_air_instance_trace(
        &self,
        airgroup_id: usize,
        air_id: usize,
        air_instance_id: usize,
    ) -> ProofmanResult<Vec<F>> {
        let dctx = self.dctx.read().unwrap();
        let index = dctx.find_instance_id(airgroup_id, air_id, air_instance_id);
        if let Some(index) = index {
            Ok(self.air_instances[index].read().unwrap().get_trace())
        } else {
            Err(ProofmanError::OutOfBounds(format!(
                "Air Instance with id {air_instance_id} for airgroup {airgroup_id} and air {air_id} not found"
            )))
        }
    }

    pub fn get_air_instance_air_values(
        &self,
        airgroup_id: usize,
        air_id: usize,
        air_instance_id: usize,
    ) -> ProofmanResult<Vec<F>> {
        let dctx = self.dctx.read().unwrap();
        let index = dctx.find_instance_id(airgroup_id, air_id, air_instance_id);
        if let Some(index) = index {
            Ok(self.air_instances[index].read().unwrap().get_air_values())
        } else {
            Err(ProofmanError::OutOfBounds(format!(
                "Air Instance with id {air_instance_id} for airgroup {airgroup_id} and air {air_id} not found"
            )))
        }
    }

    pub fn get_air_instance_airgroup_values(
        &self,
        airgroup_id: usize,
        air_id: usize,
        air_instance_id: usize,
    ) -> ProofmanResult<Vec<F>> {
        let dctx = self.dctx.read().unwrap();
        let index = dctx.find_instance_id(airgroup_id, air_id, air_instance_id);
        if let Some(index) = index {
            Ok(self.air_instances[index].read().unwrap().get_airgroup_values())
        } else {
            Err(ProofmanError::OutOfBounds(format!(
                "Air Instance with id {air_instance_id} for airgroup {airgroup_id} and air {air_id} not found"
            )))
        }
    }

    pub fn free_instance(&self, instance_id: usize) -> (bool, Vec<F>) {
        self.air_instances[instance_id].write().unwrap().reset()
    }

    pub fn free_instance_traces(&self, instance_id: usize) -> (bool, Vec<F>) {
        self.air_instances[instance_id].write().unwrap().clear_traces()
    }
}
