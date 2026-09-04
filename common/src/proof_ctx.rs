use std::{
    collections::{HashMap, HashSet},
    sync::RwLock,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::Mutex;
use crate::{plan_stream_layout, StreamClass, StreamLayout};
use crate::{MpiCtx, ProofmanError};
use borsh::{BorshDeserialize, BorshSerialize};
use std::fs::File;
use std::io::Read;
use proofman_fields::{new_transcript, PrimeField64};
use crate::{
    initialize_logger, format_bytes, AirInstance, DistributionCtx, GlobalInfo, InstanceInfo, PolMap, SetupCtx, StdMode,
    CustomCommits, PackedInfo, RowInfo, Setup, StepsParams, SetupsVadcop, VerboseMode, ProofmanResult,
    custom_commit_words_per_row, CustomCommitEntry, CustomCommitValidation,
};

use std::ffi::c_void;
use proofman_starks_lib_c::{
    upload_custom_commit_packed_c, check_device_memory_c, configure_phase_b_c, get_num_gpus_c, gen_device_buffers_c,
    gen_device_streams_c, alloc_device_large_buffers_c, acquire_first_gpu_buffer_c, release_first_gpu_buffer_c,
    get_stream_commit_floor_c, get_unified_buffer_gpu_size_c, get_first_gpu_id_c, get_first_gpu_buffer_c,
    get_const_pols_aggregation_offset_c,
};
use proofman_util::DeviceBuffer;

#[derive(Debug)]
pub struct Values<F> {
    pub values: RwLock<Vec<F>>,
}

impl<F: PrimeField64> Values<F> {
    pub fn new(n_values: usize) -> Self {
        Self { values: RwLock::new(vec![F::ZERO; n_values]) }
    }

    pub fn reset(&self) {
        self.values.write().unwrap().fill(F::ZERO);
    }
}

impl<F> Default for Values<F> {
    fn default() -> Self {
        Self { values: RwLock::new(Vec::new()) }
    }
}

#[derive(Debug, Clone)]
pub struct InstancesInfo {
    pub constraints: Vec<usize>,
    pub hint_ids: Vec<usize>,
    pub rows: Vec<usize>,
    pub store_row_info: bool,
}

pub type AirGroupMap = HashMap<usize, AirIdMap>;
pub type AirIdMap = HashMap<usize, (bool, InstanceMap)>;
pub type InstanceMap = HashMap<usize, InstancesInfo>;

pub const DEFAULT_N_PRINT_CONSTRAINTS: usize = 10;

/// GPU memory (in MB) left unallocated for consumers outside our arena.
const GPU_MEMORY_RESERVE_MB: u64 = 1536;

/// Unified-buffer floor (bytes) for final-snark runs: the snark prover borrows the buffer
/// whole and carves ~27.97 GiB (2^24 plonk key), regardless of what the streams need.
/// Padded from the layout's unused slack only, so it never eats into the reserve.
const GPU_UNIFIED_BUFFER_MIN_SNARK_BYTES: u64 = 30_386_893_620;

#[derive(Clone)]
pub struct ProofOptions {
    pub verify_constraints: bool,
    pub aggregation: bool,
    pub rma: bool,
    pub compressed: bool,
    pub verify_proofs: bool,
    pub minimal_memory: bool,
}

impl BorshSerialize for ProofOptions {
    fn serialize<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        BorshSerialize::serialize(&self.verify_constraints, writer)?;
        BorshSerialize::serialize(&self.aggregation, writer)?;
        BorshSerialize::serialize(&self.rma, writer)?;
        BorshSerialize::serialize(&self.compressed, writer)?;
        BorshSerialize::serialize(&self.verify_proofs, writer)?;
        BorshSerialize::serialize(&self.minimal_memory, writer)?;
        Ok(())
    }
}

impl BorshDeserialize for ProofOptions {
    fn deserialize_reader<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let verify_constraints = bool::deserialize_reader(reader)?;
        let aggregation = bool::deserialize_reader(reader)?;
        let rma = bool::deserialize_reader(reader)?;
        let compressed = bool::deserialize_reader(reader)?;
        let verify_proofs = bool::deserialize_reader(reader)?;
        let minimal_memory = bool::deserialize_reader(reader)?;

        Ok(Self { verify_constraints, aggregation, rma, compressed, verify_proofs, minimal_memory })
    }
}

#[derive(Debug, Clone)]
pub struct DebugInfo {
    pub debug_instances: AirGroupMap,
    pub debug_global_instances: Vec<usize>,
    pub std_mode: StdMode,
    pub n_print_constraints: usize,
    pub skip_prover_instances: bool,
    pub store_row_info: bool,
}

impl Default for DebugInfo {
    fn default() -> Self {
        Self {
            debug_instances: Default::default(),
            debug_global_instances: Default::default(),
            std_mode: Default::default(),
            n_print_constraints: DEFAULT_N_PRINT_CONSTRAINTS,
            skip_prover_instances: false,
            store_row_info: false,
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
            skip_prover_instances: false,
            store_row_info: false,
        }
    }
}
impl Default for ProofOptions {
    fn default() -> Self {
        Self {
            verify_constraints: false,
            aggregation: true,
            rma: true,
            compressed: false,
            verify_proofs: false,
            minimal_memory: false,
        }
    }
}

impl ProofOptions {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        verify_constraints: bool,
        aggregation: bool,
        rma: bool,
        compressed: bool,
        verify_proofs: bool,
        minimal_memory: bool,
    ) -> Self {
        Self { verify_constraints, aggregation, rma, compressed, verify_proofs, minimal_memory }
    }

    pub fn minimal_memory(&mut self) {
        self.minimal_memory = true;
    }

    pub fn compressed(&mut self) {
        self.compressed = true;
    }
}

#[derive(Clone)]
pub struct ProofmanOptions {
    pub max_number_streams: usize,
    /// Upper bound on per-GPU recursive (aggregation) streams. The actual count is
    /// also memory-bounded; this caps it. Defaults to 10 (the prior hardcoded cap).
    pub max_number_recursive_streams: usize,
    pub number_threads_pools_witness: usize,
    pub are_threads_per_witness_set: bool,
    pub max_witness_stored: usize,
    pub verify_constraints: bool,
    pub aggregation: bool,
    pub verbose_mode: VerboseMode,
    pub gpu: bool,
    pub packed: bool,
    pub packed_info: HashMap<(usize, usize), PackedInfo>,
    /// Airs whose const *tree* is kept GPU-resident, for each of their Basic, Compressor
    /// (when they have one) and Recursive1 circuits: `const_tree_size` VRAM each, against a
    /// per-proof load from disk. Airgroup 0's Recursive2 is always preloaded and must not be
    /// listed.
    pub preloaded_const_tree_gpu: Vec<(usize, usize)>,
    /// Tables: airs proved at most once, so their const pols need not survive the proof and
    /// the layout can alias them onto the const tree's node area, saving `N * nConstants` per
    /// stream (see StarkInfo::constPolsAliasTree). Airs that do not qualify keep the normal
    /// layout. Listing a non-table air costs a re-merkelize of its fixed on every proof.
    pub table_airs_gpu: Vec<(usize, usize)>,
    /// This run produces a final SNARK
    pub final_snark: bool,
}

impl Default for ProofmanOptions {
    fn default() -> Self {
        Self {
            max_number_streams: 20,
            max_number_recursive_streams: 10,
            number_threads_pools_witness: 4,
            max_witness_stored: 10,
            are_threads_per_witness_set: false,
            packed: false,
            gpu: false,
            verify_constraints: false,
            aggregation: true,
            verbose_mode: VerboseMode::Info,
            packed_info: HashMap::new(),
            preloaded_const_tree_gpu: Vec::new(),
            table_airs_gpu: Vec::new(),
            final_snark: false,
        }
    }
}

impl ProofmanOptions {
    pub fn new() -> Self {
        Self { ..Self::default() }
    }

    pub fn with_max_number_streams(&mut self, max_number_streams: usize) {
        self.max_number_streams = max_number_streams;
    }

    pub fn with_max_number_recursive_streams(&mut self, max_number_recursive_streams: usize) {
        self.max_number_recursive_streams = max_number_recursive_streams;
    }

    pub fn with_number_threads_pools_witness(&mut self, number_threads_pools_witness: usize) {
        self.number_threads_pools_witness = number_threads_pools_witness;
        self.are_threads_per_witness_set = true;
    }
    pub fn with_max_witness_stored(&mut self, max_witness_stored: usize) {
        self.max_witness_stored = max_witness_stored;
    }

    pub fn packed(&mut self) {
        self.packed = true;
    }

    /// Declare that this run will produce a final SNARK (plonk/fflonk wrapper). Gates the
    /// unified-buffer snark floor (GPU_UNIFIED_BUFFER_MIN_SNARK_BYTES): runs without a wrapper never
    /// borrow the buffer at that size, so they skip the padding and keep the layout slack free.
    pub fn final_snark(&mut self) {
        self.final_snark = true;
    }

    /// Undoes `gpu()`'s implied packing. Without this `--no-packed` relies on `packed_info` being
    /// empty, which stops being true as soon as a caller populates it unconditionally.
    pub fn no_packed(&mut self) {
        self.packed = false;
    }

    pub fn gpu(&mut self) {
        self.gpu = true;
        self.packed = true;
    }

    pub fn verify_constraints(&mut self) {
        self.verify_constraints = true;
        self.aggregation = false;
    }

    pub fn no_aggregation(&mut self) {
        self.aggregation = false;
    }

    pub fn verbose_mode(&mut self, verbose_mode: VerboseMode) {
        self.verbose_mode = verbose_mode;
    }

    pub fn packed_info(&mut self, packed_info: HashMap<(usize, usize), PackedInfo>) {
        self.packed_info = packed_info;
    }

    pub fn preloaded_const_tree_gpu(&mut self, preloaded_const_tree_gpu: Vec<(usize, usize)>) {
        self.preloaded_const_tree_gpu = preloaded_const_tree_gpu;
    }

    pub fn table_airs_gpu(&mut self, table_airs_gpu: Vec<(usize, usize)>) {
        self.table_airs_gpu = table_airs_gpu;
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
    pub compressor_weights: HashMap<(usize, usize), u64>,
    pub custom_commits_values: Mutex<HashMap<String, CustomCommitEntry>>,
    pub dctx: RwLock<DistributionCtx>,
    pub debug_info: RwLock<DebugInfo>,
    pub aggregation: bool,
    pub proof_tx: RwLock<Option<crossbeam_channel::Sender<usize>>>,
    pub witness_tx: RwLock<Option<crossbeam_channel::Sender<usize>>>,
    pub witness_tx_priority: RwLock<Option<crossbeam_channel::Sender<usize>>>,
    pub d_buffers: Arc<DeviceBuffer>,
    pub gpu: bool,
    /// Airs whose rows components must write packed. Holds the same `packedTrace && is_packed`
    /// pair the device gates on, so a global flag cannot disagree with the per-air setup.
    pub packed_airs: HashSet<(usize, usize)>,
    pub reload_fixed_pols_gpu: Arc<AtomicBool>,
    /// Aux-trace size of each basic GPU stream, largest class first (empty until `set_device_buffers`,
    /// and on CPU). An air can only run on a stream at least as large as its `prover_buffer_size`, so
    /// this is what makes stream eligibility visible to the Rust-side schedulers.
    pub basic_stream_sizes: Vec<usize>,
    /// Phase B registered: the two recursive streams alias the single basic stream's buffer and
    /// open only after every basic and compressor completed (set_phase_b_c from the proofs phase).
    pub phase_b: bool,
}

pub const MAX_INSTANCES: u64 = 1 << 17;

impl<F: PrimeField64> ProofCtx<F> {
    pub fn create_ctx(
        proving_key_path: PathBuf,
        aggregation: bool,
        verbose_mode: VerboseMode,
        mpi_ctx: Arc<MpiCtx>,
        gpu: bool,
    ) -> ProofmanResult<Self> {
        tracing::info!("Creating proof context");

        let mut dctx = DistributionCtx::new();

        dctx.setup_processes(mpi_ctx.n_processes as usize, mpi_ctx.rank as usize)?;

        initialize_logger(verbose_mode, None);
        let global_info: GlobalInfo = GlobalInfo::new(&proving_key_path)?;
        tracing::info!("Using hash function: {}", global_info.hash);
        let n_publics = global_info.n_publics;
        let n_proof_values = global_info
            .proof_values_map
            .as_ref()
            .map(|map| map.iter().filter(|entry| entry.stage == 1).count())
            .unwrap_or(0);
        let n_challenges = global_info.n_challenges.iter().sum::<usize>();

        let weights = HashMap::new();
        let compressor_weights = HashMap::new();

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
            custom_commits_values: Mutex::new(HashMap::new()),
            weights,
            compressor_weights,
            aggregation,
            witness_tx: RwLock::new(None),
            witness_tx_priority: RwLock::new(None),
            proof_tx: RwLock::new(None),
            d_buffers: Arc::new(DeviceBuffer::default()),
            gpu,
            packed_airs: HashSet::new(),
            reload_fixed_pols_gpu: Arc::new(AtomicBool::new(false)),
            basic_stream_sizes: Vec::new(),
            phase_b: false,
        })
    }

    pub fn get_rank_info(&self) -> crate::RankInfo {
        crate::RankInfo {
            world_rank: self.mpi_ctx.rank,
            local_rank: self.mpi_ctx.node_rank,
            n_processes: self.mpi_ctx.n_processes,
        }
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

    pub fn is_setup_partition_init(&self) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.is_setup_partition_init()
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

    /// `words_per_row` and root of a registered custom-commit file, or why it cannot be used.
    fn read_custom_commit_header(
        setup: &Setup<F>,
        custom_commit: &CustomCommits,
        path: &Path,
    ) -> ProofmanResult<(u64, [u8; 32])> {
        if !path.exists() {
            return Err(ProofmanError::ProofmanError(format!("{} does not exist", path.display())));
        }
        let n = 1u64 << setup.stark_info.stark_struct.n_bits;
        let n_extended = 1u64 << setup.stark_info.stark_struct.n_bits_ext;
        let n_cols = custom_commit.stage_widths[0] as u64;
        let arity = setup.stark_info.stark_struct.merkle_tree_arity;
        let words_per_row = custom_commit_words_per_row(path, n, n_extended, n_cols, arity)?;

        let mut root_bytes = [0u8; 32];
        File::open(path)?.read_exact(&mut root_bytes)?;
        Ok((words_per_row, root_bytes))
    }

    /// Packed `words_per_row` of a registered custom commit; 0 if its file is not generated yet.
    pub fn get_custom_commit_words_per_row(&self, name: &str) -> u64 {
        let lock = self.custom_commits_values.lock().unwrap();
        lock.get(name).map(|(_, _, wpr)| *wpr).unwrap_or(0)
    }

    /// Names of the custom commits whose files still have to be generated.
    pub fn custom_commits_pending(&self) -> Vec<String> {
        let lock = self.custom_commits_values.lock().unwrap();
        lock.iter().filter(|(_, (_, _, wpr))| *wpr == 0).map(|(name, _)| name.clone()).collect()
    }

    /// The registered name -> file map, to re-run validation after a regeneration.
    pub fn custom_commits_paths(&self) -> HashMap<String, PathBuf> {
        let lock = self.custom_commits_values.lock().unwrap();
        lock.iter().map(|(name, (path, _, _))| (name.clone(), path.clone())).collect()
    }

    pub fn initialize_custom_commits(
        &self,
        custom_commits_fixed: HashMap<String, PathBuf>,
        sctx: &SetupCtx<F>,
        validation: CustomCommitValidation,
    ) -> ProofmanResult<()> {
        tracing::info!("Initializing publics custom_commits");
        for (airgroup_id, airs) in self.global_info.airs.iter().enumerate() {
            for (air_id, _) in airs.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id)?;
                for custom_commit in setup.stark_info.custom_commits.iter() {
                    if custom_commit.stage_widths[0] > 0 {
                        let custom_file_path = custom_commits_fixed.get(&custom_commit.name).ok_or_else(|| {
                            ProofmanError::ProofmanError(format!(
                                "Custom commit file path for {} not found",
                                custom_commit.name
                            ))
                        })?;

                        // words_per_row == 0 marks the entry as still needing generation.
                        let mut root_bytes = [0u8; 32];
                        let mut words_per_row = 0u64;
                        if validation != CustomCommitValidation::Skip {
                            match Self::read_custom_commit_header(setup, custom_commit, custom_file_path) {
                                Ok((wpr, root)) => {
                                    words_per_row = wpr;
                                    root_bytes = root;
                                }
                                Err(err) => {
                                    if validation == CustomCommitValidation::Strict {
                                        let error_message = format!(
                                            "Error: The custom commit file for {} at '{}' cannot be used ({}) and \
                                            regenerating it did not help.\n\
                                            Please regenerate it by running:\n\
                                            \x1b[1mcargo run --bin proofman-cli gen-custom-commits-fixed --witness-lib <WITNESS_LIB> --proving-key <PROVING_KEY> --custom-commits <CUSTOM_COMMITS_DIR> \x1b[0m",
                                            custom_commit.name,
                                            custom_file_path.display(),
                                            err,
                                        );
                                        tracing::warn!("{}", error_message);
                                        return Err(ProofmanError::ProofmanError(error_message));
                                    }
                                    tracing::info!(
                                        "Custom commit {} at '{}' will be regenerated ({})",
                                        custom_commit.name,
                                        custom_file_path.display(),
                                        err
                                    );
                                }
                            }
                        }

                        // Resident for the process lifetime: no proof DMAs a custom commit.
                        if setup.gpu && words_per_row > 0 {
                            upload_custom_commit_packed_c(
                                airgroup_id as u64,
                                air_id as u64,
                                setup.setup_type.into(),
                                &custom_file_path.to_string_lossy(),
                                words_per_row,
                                (&setup.p_setup).into(),
                                self.get_device_buffers_ptr(),
                            );
                        }

                        self.custom_commits_values.lock().unwrap().insert(
                            custom_commit.name.clone(),
                            (custom_file_path.clone(), root_bytes.to_vec(), words_per_row),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_custom_commit_root(&self, name: &str) -> ProofmanResult<Vec<u8>> {
        let custom_commit_lock = self.custom_commits_values.lock().unwrap();
        let root_bytes = custom_commit_lock.get(name);
        match root_bytes {
            Some((_, bytes, _)) => Ok(bytes.clone()),
            None => Err(ProofmanError::ProofmanError(format!("Custom Commit {name} not found"))),
        }
    }

    /// Estimated prover cost of a single proof of this setup
    fn setup_weight(setup: &Setup<F>) -> u64 {
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
        // weight += (n_ops_quotient / 10) * (1 << (setup.stark_info.stark_struct.n_bits_ext));
        (total_cols + n_openings * 3) * (1 << (setup.stark_info.stark_struct.n_bits_ext))
    }

    /// Cost of the basic proof of every air, plus the compressor proof it triggers if it has one.
    /// Both are needed to balance instances: a compressor proof runs on the owner of its basic
    /// proof, and its cost varies ~2x between airs in the current zisk proving key.
    pub fn set_weights(&mut self, sctx: &SetupCtx<F>, setups_vadcop: &SetupsVadcop<F>) -> ProofmanResult<()> {
        for (airgroup_id, air_group) in self.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id)?;
                self.weights.insert((airgroup_id, air_id), Self::setup_weight(setup));

                if self.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    if let Some(sctx_compressor) = setups_vadcop.sctx_compressor.as_ref() {
                        let compressor_setup = sctx_compressor.get_setup(airgroup_id, air_id)?;
                        self.compressor_weights.insert((airgroup_id, air_id), Self::setup_weight(compressor_setup));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_weight(&self, airgroup_id: usize, air_id: usize) -> u64 {
        *self.weights.get(&(airgroup_id, air_id)).unwrap()
    }

    /// 0 if the air has no compressor, or if the compressor setups are not loaded (no aggregation)
    pub fn get_compressor_weight(&self, airgroup_id: usize, air_id: usize) -> u64 {
        self.compressor_weights.get(&(airgroup_id, air_id)).copied().unwrap_or(0)
    }

    pub fn get_custom_commits_fixed_buffer(&self, name: &str, return_error: bool) -> ProofmanResult<PathBuf> {
        let custom_commits_lock = self.custom_commits_values.lock().unwrap();
        let file_name = custom_commits_lock.get(name);
        match file_name {
            Some((path, _, _)) => Ok(path.to_path_buf()),
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

    pub fn dctx_is_first_process(&self) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.is_first_process()
    }

    pub fn dctx_reset_instances_calculated(&self) {
        let dctx = self.dctx.read().unwrap();
        for instance in dctx.instances_calculated.iter() {
            instance.store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn dctx_try_mark_instance_calculated(&self, global_idx: usize) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.instances_calculated[global_idx]
            .compare_exchange(false, true, std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst)
            .is_ok()
    }

    pub fn dctx_reset_instance_calculated(&self, global_idx: usize) {
        let dctx = self.dctx.read().unwrap();
        dctx.instances_calculated[global_idx].store(false, std::sync::atomic::Ordering::SeqCst);
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
            .filter(|(id, inst)| {
                inst.table && (dctx.process_instances.contains(id) || inst.shared) && !dctx.is_skipped_instance(*id)
            })
            .map(|(id, _)| id)
            .collect()
    }

    pub fn dctx_get_process_instances(&self) -> Vec<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.process_instances.iter().copied().filter(|id| !dctx.is_skipped_instance(*id)).collect()
    }

    pub fn dctx_skip_process_instance(&self, instance_id: usize) {
        let mut dctx = self.dctx.write().unwrap();
        dctx.skip_instance(instance_id);
    }

    pub fn dctx_is_skipped_process_instance(&self, instance_id: usize) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.is_skipped_instance(instance_id)
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

    /// Whether this air's witness rows must be written packed.
    pub fn is_packed(&self, airgroup_id: usize, air_id: usize) -> bool {
        self.packed_airs.contains(&(airgroup_id, air_id))
    }

    pub fn is_shared_buffer(&self, global_idx: usize) -> bool {
        self.air_instances[global_idx].read().unwrap().is_shared_buffer()
    }

    pub fn dctx_find_air_instance_id(&self, global_idx: usize) -> ProofmanResult<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.find_air_instance_id(global_idx)
    }

    pub fn dctx_find_instance_id(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<(bool, usize)> {
        let dctx = self.dctx.read().unwrap();
        dctx.find_instance_id(airgroup_id, air_id)
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
        let compressor_weight = self.get_compressor_weight(airgroup_id, air_id);
        dctx.add_instance(airgroup_id, air_id, weight, compressor_weight)
    }

    pub fn add_instance(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<usize> {
        let mut dctx = self.dctx.write().unwrap();
        let weight = self.get_weight(airgroup_id, air_id);
        let compressor_weight = self.get_compressor_weight(airgroup_id, air_id);
        dctx.add_instance_no_assign(airgroup_id, air_id, weight, compressor_weight)
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
        let compressor_weight = self.get_compressor_weight(airgroup_id, air_id);
        dctx.add_instance_no_assign(airgroup_id, air_id, weight, compressor_weight)
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

        let mut transcript = new_transcript::<F>(&self.global_info.hash);

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

    pub fn get_air_instance_stream_id(&self, instance_id: usize) -> u64 {
        self.air_instances[instance_id].read().unwrap().get_stream_id()
    }

    pub fn get_air_instance_trace(
        &self,
        instance_id: usize,
        first_row: usize,
        n_rows: usize,
        offset: Option<usize>,
    ) -> Vec<RowInfo> {
        self.air_instances[instance_id].read().unwrap().get_trace(first_row, n_rows, offset)
    }

    pub fn get_instance_air_values(&self, instance_id: usize, airvalues_map: &[PolMap]) -> ProofmanResult<Vec<u64>> {
        let air_values = self.air_instances[instance_id].read().unwrap().get_air_values();

        let mut result = Vec::new();
        for (p, air_value) in airvalues_map.iter().enumerate() {
            if air_value.stage == 1 {
                result.push(air_values[p].as_canonical_u64());
            }
        }

        Ok(result)
    }

    pub fn get_air_instance_air_values(
        &self,
        airgroup_id: usize,
        air_id: usize,
        air_instance_id: usize,
    ) -> ProofmanResult<Vec<F>> {
        let dctx = self.dctx.read().unwrap();
        let index = dctx.find_by_air_instance_id(airgroup_id, air_id, air_instance_id);
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
        let index = dctx.find_by_air_instance_id(airgroup_id, air_id, air_instance_id);
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

    pub fn set_instance_stream_id(&self, instance_id: usize, stream_id: u64) {
        self.air_instances[instance_id].write().unwrap().set_stream_id(stream_id);
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_device_buffers(
        &mut self,
        sctx: &SetupCtx<F>,
        setups_vadcop: &SetupsVadcop<F>,
        aggregation: bool,
        gpu: bool,
        max_number_streams_gpu: usize,
        max_number_recursive_streams_gpu: usize,
        final_snark: bool,
        // Witness prefetch-region area (elements), carved inside the unified buffer (0 = none).
        prefetch_region_area: u64,
        // -> (basic streams/GPU, recursive streams/GPU, aggregation workers/GPU, GPUs)
    ) -> ProofmanResult<(u64, u64, u64, u64)> {
        let d_buffers = Arc::new(DeviceBuffer(gen_device_buffers_c(
            self.mpi_ctx.node_rank as u32,
            self.mpi_ctx.node_n_processes as u32,
            &self.mpi_ctx.numa_nodes,
            self.global_info.transcript_arity as u32,
            sctx.max_n_bits_ext as u32,
        )));

        let mut free_memory_gpu = match gpu {
            true => check_device_memory_c(self.mpi_ctx.node_rank as u32, self.mpi_ctx.node_n_processes as u32) as f64,
            false => 0.0,
        };

        if gpu {
            let reserve = (GPU_MEMORY_RESERVE_MB * 1024 * 1024) as f64;
            free_memory_gpu = (free_memory_gpu - reserve).max(0.0);
            tracing::info!("Reserving {} of GPU memory for other device consumers", format_bytes(reserve));
        }

        self.mpi_ctx.barrier();

        let n_gpus = get_num_gpus_c();
        let n_processes_node = self.mpi_ctx.node_n_processes as usize as u64;

        let n_partitions = match gpu {
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

        let mut total_const_area = 0;
        let mut total_const_area_aggregation = 0;

        if gpu {
            total_const_area += sctx.total_const_pols_size as u64;
            total_const_area += sctx.total_const_tree_size as u64;
            if aggregation {
                total_const_area_aggregation += setups_vadcop.total_const_pols_size as u64;
                total_const_area_aggregation += setups_vadcop.total_const_tree_size as u64;
            }
        }

        // Wrapping here would carve streams out of a budget the card does not have, and only fail
        // later inside cudaMalloc.
        let max_size_buffer = ((free_memory_gpu / 8.0).floor() as u64)
            .checked_sub(total_const_area + total_const_area_aggregation)
            .ok_or_else(|| {
                ProofmanError::InvalidConfiguration(format!(
                    "Fixed polynomials need {} but only {} is free on the GPU",
                    format_bytes((total_const_area + total_const_area_aggregation) as f64 * 8.0),
                    format_bytes(free_memory_gpu),
                ))
            })?;

        // Non-recursive streams also take compressor/vadcop_final launches, which floors their classes.
        let recursive_capable_size = setups_vadcop.max_prover_recursive_buffer_size;
        let max_prover_recursive2_buffer_size =
            if aggregation { setups_vadcop.max_prover_recursive2_buffer_size } else { 0 };

        tracing::info!(
            "Max prover buffer size: {}",
            format_bytes(sctx.max_prover_buffer_size.max(recursive_capable_size) as f64 * 8.0)
        );
        tracing::info!("Max prover recursive buffer size: {}", format_bytes(recursive_capable_size as f64 * 8.0));
        tracing::info!(
            "Max prover recursive1/recursive2 buffer size: {}",
            format_bytes(setups_vadcop.max_prover_recursive2_buffer_size as f64 * 8.0)
        );

        let basic_sizes: Vec<usize> = sctx.prover_buffer_sizes.iter().map(|(_, size)| *size).collect();

        // The prefetch region is carved INSIDE the unified buffer after planning;
        // hide it from the planner's budget or its streams overrun VRAM.
        let max_size_buffer = max_size_buffer.saturating_sub(prefetch_region_area);

        let layout = match gpu {
            true => plan_stream_layout(
                max_size_buffer as usize,
                &basic_sizes,
                recursive_capable_size,
                max_prover_recursive2_buffer_size,
                max_number_streams_gpu,
                if aggregation { max_number_recursive_streams_gpu } else { 0 },
            )
            .ok_or_else(|| ProofmanError::InvalidConfiguration("Not enough GPU memory to run the proof".into()))?,
            // No device streams to carve: one nominal stream, sized for the largest proof.
            false => StreamLayout {
                basic: vec![StreamClass { size: sctx.max_prover_buffer_size.max(recursive_capable_size), count: 1 }],
                recursive: StreamClass { size: max_prover_recursive2_buffer_size, count: 0 },
                aggregation_workers: 0,
                unused: 0,
            },
        };

        // Phase B aliases two recursive1/recursive2 streams over the first GPU's pre-const area,
        // which with one basic stream is that stream plus the prefetch zone and the mops-floor pad.
        // Make the footprint a planned property: with aggregation and a single basic stream, the
        // stream itself holds two recursive classes. Memory-neutral on the device -- the pad below
        // the const pols shrinks by the same amount (the pre-const area stays at the mops floor).
        let mut layout = layout;
        if gpu && aggregation && layout.n_basic_streams() == 1 && layout.recursive.count == 0 {
            let phase_b_floor = 2 * max_prover_recursive2_buffer_size;
            let large = &mut layout.basic[0];
            if large.size < phase_b_floor {
                let grow = phase_b_floor - large.size;
                if grow <= layout.unused {
                    large.size = phase_b_floor;
                    layout.unused -= grow;
                } else {
                    tracing::warn!(
                        "single basic stream ({}) cannot grow to hold two recursive streams ({}): only {} unused -- phase-B two-stream recursion will not be available on this layout",
                        format_bytes(large.size as f64 * 8.0),
                        format_bytes(phase_b_floor as f64 * 8.0),
                        format_bytes(layout.unused as f64 * 8.0)
                    );
                }
            }
        }

        let aux_trace_sizes: Vec<u64> = layout.basic_stream_sizes().iter().map(|&s| s as u64).collect();
        // Retained so the witness admission can tell which airs are confined to a subset of streams.
        self.basic_stream_sizes = layout.basic_stream_sizes();
        let n_streams_per_gpu = layout.n_basic_streams();
        let mut n_recursive_streams_per_gpu = layout.recursive.count;

        // Phase B: with aggregation and one basic stream, two recursive streams alias that
        // stream's buffer (sized above to hold both). They cost no memory and open once every
        // basic and compressor has completed. PROOFMAN_NO_PHASE_B=1 disables.
        self.phase_b = gpu
            && aggregation
            && n_streams_per_gpu == 1
            && n_recursive_streams_per_gpu == 0
            && layout.basic[0].size >= 2 * max_prover_recursive2_buffer_size
            && !std::env::var("PROOFMAN_NO_PHASE_B").map(|v| v == "1").unwrap_or(false);
        if self.phase_b {
            configure_phase_b_c(d_buffers.get_ptr());
            n_recursive_streams_per_gpu = 2;
        }

        if gpu {
            let classes = layout
                .basic
                .iter()
                .map(|c| format!("{} x {}", c.count, format_bytes(c.size as f64 * 8.0)))
                .collect::<Vec<_>>()
                .join(" + ");
            let aggregation_desc = match n_recursive_streams_per_gpu {
                0 => format!("{} aggregation workers sharing the basic streams", layout.aggregation_workers),
                2 if self.phase_b => "2 phase-B recursive streams aliased over the basic stream".to_string(),
                n => format!("{n} dedicated streams per GPU for recursive proofs"),
            };
            tracing::info!(
                "Using {} streams per GPU for basic proofs ({}) and {}. \
                 Using {} for fixed pols, {} unused",
                n_streams_per_gpu,
                classes,
                aggregation_desc,
                format_bytes((total_const_area + total_const_area_aggregation) as f64 * 8.0),
                format_bytes(layout.unused as f64 * 8.0),
            );
        }

        let max_pinned_proof_size = match aggregation {
            true => sctx.max_pinned_proof_size.max(setups_vadcop.max_pinned_proof_size) as u64,
            false => sctx.max_pinned_proof_size as u64,
        };

        let n_gpus: u64 = gen_device_streams_c(
            d_buffers.get_ptr(),
            &aux_trace_sizes,
            n_recursive_streams_per_gpu as u64,
            max_prover_recursive2_buffer_size as u64,
            max_pinned_proof_size,
            self.global_info.transcript_arity as u64,
        );

        // Pad the unified buffer up to the snark floor (see GPU_UNIFIED_BUFFER_MIN_SNARK_BYTES),
        // taking only the layout's unused slack. Runs without a wrapper skip it.
        let unified_buffer_pad_area: u64 = if gpu && final_snark {
            let predicted_unified_buffer: u64 = aux_trace_sizes.iter().sum::<u64>()
                + if self.phase_b { 0 } else { n_recursive_streams_per_gpu as u64 * max_prover_recursive2_buffer_size as u64 }
                + total_const_area_aggregation
                + total_const_area;
            let floor_elems = GPU_UNIFIED_BUFFER_MIN_SNARK_BYTES.div_ceil(8);
            let pad = floor_elems.saturating_sub(predicted_unified_buffer).min(layout.unused as u64);
            if pad > 0 {
                tracing::info!(
                    "Padding the unified buffer by {} to reach the {} snark floor",
                    format_bytes(pad as f64 * 8.0),
                    format_bytes(GPU_UNIFIED_BUFFER_MIN_SNARK_BYTES as f64),
                );
                if floor_elems.saturating_sub(predicted_unified_buffer) > layout.unused as u64 {
                    tracing::warn!(
                        "Snark floor not reachable: layout slack is {} short; the final snark \
                         prover will not fit the unified buffer",
                        format_bytes((floor_elems - predicted_unified_buffer - layout.unused as u64) as f64 * 8.0),
                    );
                }
            }
            pad
        } else {
            0
        };

        alloc_device_large_buffers_c(
            d_buffers.get_ptr(),
            max_prover_recursive2_buffer_size as u64,
            total_const_area,
            total_const_area_aggregation,
            unified_buffer_pad_area,
            prefetch_region_area,
        );

        self.d_buffers = d_buffers;

        Ok((n_streams_per_gpu as u64, n_recursive_streams_per_gpu as u64, layout.aggregation_workers as u64, n_gpus))
    }

    pub fn get_device_buffers_ptr(&self) -> *mut c_void {
        self.d_buffers.get_ptr()
    }

    pub fn acquire_first_gpu_buffer(&self) {
        if self.gpu {
            acquire_first_gpu_buffer_c(self.d_buffers.get_ptr());
        }
    }

    pub fn release_first_gpu_buffer(&self) {
        if self.gpu {
            release_first_gpu_buffer_c(self.d_buffers.get_ptr());
        }
    }

    /// Unified buffer of the FIRST GPU (my_gpu_ids[0]) — the one borrowed via
    /// `acquire_first_gpu_buffer`, does not touch the current device.
    ///
    /// When streaming-commit slots are enabled they occupy the top of the
    /// buffer and host commits WHILE it is borrowed, so the borrower's usable
    /// region is capped at the slot floor (allocation-time enforcement: the
    /// borrower's planner sizes itself within what it is handed). No-op when
    /// slots are disabled (floor = u64::MAX); the const-pols tail then stays
    /// reachable and the post-hoc reload check keeps covering it.
    pub fn get_first_gpu_buffer(&self) -> (usize, u64) {
        let device_buffers_ptr = self.d_buffers.get_ptr();
        let gpu_buf_ptr = get_first_gpu_buffer_c(device_buffers_ptr) as usize;
        let gpu_buf_size = get_unified_buffer_gpu_size_c(device_buffers_ptr);
        let usable = gpu_buf_size.min(get_stream_commit_floor_c(device_buffers_ptr));
        (gpu_buf_ptr, usable)
    }

    /// Report how many bytes a borrower of the first GPU's unified buffer actually used.
    ///
    /// The buffer's tail holds the once-uploaded fixed pols of every aggregation
    /// setup (compressor/recursive1/recursive2/vadcop_final). If the borrower's
    /// usage reached that region they are now garbage, so this raises
    /// `reload_fixed_pols_gpu`; the proving flow consumes it right after
    /// `wcm.execute()` and re-uploads them before any proof of the block.
    /// Call after the borrower finished writing, before or at buffer release.
    /// No-op on CPU.
    pub fn report_first_gpu_buffer_usage(&self, used_bytes: u64) {
        if !self.gpu {
            return;
        }
        let consts_offset = get_const_pols_aggregation_offset_c(self.d_buffers.get_ptr());
        let buffer_size = get_unified_buffer_gpu_size_c(self.d_buffers.get_ptr());
        const GB: f64 = (1u64 << 30) as f64;
        tracing::info!(
            "GPU mops used {:.2} GB of {:.2} GB unified buffer (const pols at {:.2} GB)",
            used_bytes as f64 / GB,
            buffer_size as f64 / GB,
            consts_offset as f64 / GB,
        );

        // Streaming-commit slots sit below the const-pols region (u64::MAX when
        // disabled): usage reaching the slot floor means commits running during
        // the borrow window may have read clobbered data. Corrupted contribution
        // roots cannot be repaired post-hoc, so this is a hard error -- the real
        // protection is the allocation-time capacity handed to the borrower.
        // `used_bytes` is a count, so the borrower occupies [0, used_bytes) and
        // the slots start AT `slots_floor`: using exactly the capacity it was
        // handed is legal, overlapping requires strictly more.
        let slots_floor = get_stream_commit_floor_c(self.d_buffers.get_ptr());
        if used_bytes > slots_floor {
            panic!(
                "first-GPU unified buffer borrower used {used_bytes} bytes, reaching the \
                 streaming-commit slots at offset {slots_floor}; slot commits issued during \
                 the borrow window are untrustworthy"
            );
        }

        if used_bytes >= consts_offset {
            tracing::warn!(
                "first-GPU unified buffer borrower used {used_bytes} bytes, reaching the \
                 aggregation const-pols region at offset {consts_offset}; scheduling \
                 fixed-pols re-upload"
            );
            self.reload_fixed_pols_gpu.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Device of the first GPU (my_gpu_ids[0]) — the GPU `get_first_gpu_buffer`
    /// points to. Not always 0 (NUMA can reorder). 0 when GPU mode is off.
    pub fn first_gpu_id(&self) -> u32 {
        if self.gpu {
            get_first_gpu_id_c(self.d_buffers.get_ptr())
        } else {
            0
        }
    }
}
