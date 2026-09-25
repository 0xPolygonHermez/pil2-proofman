use borsh::{BorshDeserialize, BorshSerialize};
use libloading::{Library, Symbol};
use proofman_fields::{new_transcript, ExtensionField, GoldilocksQuinticExtension, PrimeField64};
use proofman_common::{
    calculate_fixed_tree, configured_num_threads, initialize_logger, load_const_pols, skip_prover_instance,
    CustomCommitValidation, CurveType, GlobalInfoAir, PolMap, RowInfo, DebugInfo, MemoryHandler,
    MemoryHandlerRecursive, MpiCtx, ProofmanOptions, Proof, ProofCtx, ProofOptions, ProofType, RankInfo, SetupCtx,
    SetupsVadcop, VerboseMode, MAX_INSTANCES, PackedInfo, WITNESS_NOT_STAGED, WITNESS_STAGED, WITNESS_STAGED_RELEASED,
    WITNESS_STAGED_KIND, witness_staged_on, witness_staged_gpu,
};
use colored::Colorize;
use proofman_hints::aggregate_airgroupvals;
use proofman_starks_lib_c::{
    mul_alloc_c, mul_fold_c, mul_migrated_tables_c, mul_register_range_tables_c, mul_register_table_map_c, mul_reset_c,
    register_mul_vt_c,
};
use pil2_std_lib::{collect_prover_owned_ranges, collect_virtual_table_layouts, fit_virtual_table_maps};
use proofman_starks_lib_c::{
    configure_prefetch_zone_c, get_prefetch_witness_slots_c, stage_witness_c, release_staged_witness_c,
    gpu_witness_count_c, harvest_pipeline_c, dump_pipeline_state_c, set_gpu_mode_c, set_pipeline_mode_c,
    load_device_const_pols_c,
};
use proofman_starks_lib_c::{
    get_stream_proofs_c, get_stream_proofs_non_blocking_c, reset_device_streams_c, set_phase_b_c,
    free_device_buffers_c, use_packed_trace_c, register_instruction_table_c,
};
use crate::add_publics_circom;
use crossbeam_channel::{bounded, unbounded, Sender, Receiver};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as FmtWrite;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::atomic::Ordering;
use std::sync::{LazyLock, Mutex, RwLock};

/// Releases an admission slot on drop. A witness thread that panics would otherwise leak one, and a
/// leaked slot permanently blocks its air once the cap is reached — the admission loop then spins.
/// Cancellation re-check interval. Was 1ms: a dispatcher spinning a core while the pool drained.
const TOKEN_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

/// Cancellation re-check for the admission loop. Not a latency knob: every event that can make work
/// admissible wakes it immediately.
const ADMISSION_WAIT: std::time::Duration = std::time::Duration::from_millis(20);

/// Ceiling on the shared recursive trace pool (compressor included). Each buffer is the largest
/// recursive trace -- 216 MB on the blake3 zisk key -- and pinned, so depth is expensive; it only
/// needs to keep the streams fed while the next witnesses are solved.
const MAX_RECURSIVE_TRACE_BUFFERS: usize = 10;

/// Counting semaphore for the CPU thread budget. Acquisition is all-or-nothing: taking tokens one
/// at a time deadlocks two concurrent acquirers, each holding a partial set and waiting for more.
struct ThreadBudget {
    available: Mutex<usize>,
    ready: std::sync::Condvar,
    capacity: usize,
}

impl ThreadBudget {
    fn new(capacity: usize) -> Self {
        Self { available: Mutex::new(capacity), ready: std::sync::Condvar::new(), capacity }
    }

    /// Reserve `n`, or fewer if cancelled. Clamped: a request over capacity would wait forever.
    fn acquire(self: &Arc<Self>, n: usize, cancellation: &RwLock<CancellationInfo>) -> ThreadTokens {
        let want = n.min(self.capacity);
        let mut available = self.available.lock().unwrap_or_else(|e| e.into_inner());
        while *available < want {
            if cancellation.read_recover().token.is_cancelled() {
                return ThreadTokens { budget: self.clone(), held: 0 };
            }
            let (guard, _) = self.ready.wait_timeout(available, TOKEN_POLL_INTERVAL).unwrap_or_else(|e| e.into_inner());
            available = guard;
        }
        *available -= want;
        ThreadTokens { budget: self.clone(), held: want }
    }

    fn give_back(&self, n: usize) {
        if n == 0 {
            return;
        }
        let mut available = self.available.lock().unwrap_or_else(|e| e.into_inner());
        *available = (*available + n).min(self.capacity);
        self.ready.notify_all();
    }

    /// Restore the full budget after a cancelled run leaked reservations.
    fn reset(&self) {
        *self.available.lock().unwrap_or_else(|e| e.into_inner()) = self.capacity;
        self.ready.notify_all();
    }
}

/// Tokens returned on drop, so a panic or an early return cannot shrink the budget.
struct ThreadTokens {
    budget: Arc<ThreadBudget>,
    held: usize,
}

impl ThreadTokens {
    /// Hand back up to `n` now; the remainder goes back on drop.
    fn release(&mut self, n: usize) {
        let n = n.min(self.held);
        self.held -= n;
        self.budget.give_back(n);
    }
}

impl Drop for ThreadTokens {
    fn drop(&mut self) {
        self.budget.give_back(self.held);
        self.held = 0;
    }
}

struct SlotGuard {
    in_flight: Arc<Mutex<HashMap<(usize, usize), usize>>>,
    key: (usize, usize),
    /// A freed slot is the one thing that can make queued work admissible with nothing arriving, so
    /// it announces itself rather than being polled for.
    wake: Sender<()>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        // Recover from poison rather than panic inside a drop during unwind.
        if let Some(n) = self.in_flight.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&self.key) {
            *n = n.saturating_sub(1);
        }
        // Best-effort: one pending wakeup is all the loop needs, and a drop must never block.
        let _ = self.wake.try_send(());
    }
}

/// Master switch for per-proof debug logging of airgroup values, stage roots, and the
/// challenge/contribution dump (`print_challenges`). Off by default; enable with
/// `PROOFMAN_DEBUG_CHALLENGES=1` (matching PROOFMAN_SUMCHECK: any other value, or unset,
/// means off). Read once so the hot recursive handler doesn't hit getenv per proof.
static DEBUG_CHALLENGES: LazyLock<bool> =
    LazyLock::new(|| std::env::var("PROOFMAN_DEBUG_CHALLENGES").map(|v| v == "1").unwrap_or(false));
use csv::Writer;

use tokio_util::sync::CancellationToken;

use proofman_common::{ProofmanResult, ProofmanError, Setup};
use proofman_verifier::VadcopFinalProof;
use crate::{
    check_const_paths, check_const_paths_vadcop, needs_regeneration_fixed, needs_regeneration_vadcop_fixed,
    schedule_key,
};

use proofman_starks_lib_c::{
    gen_proof_c, commit_witness_c, load_custom_commit_c, calculate_impols_expressions_c, mul_scatter_c,
    mul_air_device_owned_c, mul_air_has_owned_c, mul_set_device_export_c, mul_sync_commits_c,
    calculate_witness_expressions_c, launch_callback_c, initialize_instance_c, calculate_trace_instance_c,
    wait_trace_h2d_done_c, get_stream_commit_slots_c, get_stream_commit_gpus_c, commit_witness_streaming_c,
    stream_commit_slot_bytes_c, configure_stream_commit_slots_c, get_stream_id_proof_c,
};

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use proofman_witness::{WitnessLibInitFn, WitnessLibrary, WitnessManager};
use crate::challenge_accumulation::{aggregate_contributions, calculate_global_challenge, calculate_internal_contributions};
use crate::{
    calculate_max_witness_trace_size, check_tree_paths_vadcop, gen_recursive_proof_size, load_device_setups,
    load_device_const_pols,
};
use crate::{verify_constraints_proof, verify_basic_proof, verify_global_constraints_proof, verify_proof};
use crate::{print_summary_info, get_recursive_buffer_sizes, n_publics_aggregation};
use crate::{
    get_accumulated_challenge, gen_witness_recursive, gen_witness_aggregation, generate_recursive_proof,
    generate_vadcop_final_proof, generate_vadcop_final_compressed_proof,
};
use crate::total_recursive_proofs;
use crate::check_const_pols_gpu;
use crate::check_const_tree;
use crate::check_tree_paths;
use crate::Counter;
use crate::{CompletionOwner, DeviceBuffersPtr, DeviceCompletions};
use crate::{AggProofs, AggProofsRegister};
use crate::aggregate_worker_proofs;

use std::ffi::c_void;

use proofman_util::{
    timer_start_info, timer_stop_and_log_info, timer_start_debug, timer_stop_and_log_debug,
    timer_stop_and_log_debug_net, create_buffer_fast,
};

use serde::Serialize;

#[derive(Default, Debug, Clone)]
pub struct WitnessInfo {
    pub witness_time: f32,
    pub publics: Vec<u64>,
    pub proof_values: Vec<u64>,
    pub summary_info: String,
    pub total_instances: usize,
}

#[derive(Serialize)]
struct CsvInfo {
    version: String,
    airgroup_id: usize,
    air_id: usize,
    name: String,
    instance_count: usize,
    percentage_instances: f64,
    total_area: u64,
    percentage_area: f64,
    instance_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AirExecuteInfo {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub num_instances: usize,
    pub instance_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColumnName {
    pub name: String,
    pub lengths: Vec<u64>,
}

impl ColumnName {
    pub fn new(pol: &PolMap) -> Self {
        Self { name: pol.name.clone(), lengths: pol.lengths.clone() }
    }

    pub fn expand_column_name(&self) -> String {
        if self.lengths.is_empty() {
            return self.name.clone();
        }

        let suffix = self.lengths.iter().map(|i| format!("[{}]", i)).collect::<String>();
        format!("{}{}", self.name, suffix)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AirInfo {
    pub name: String,
    pub airgroup_id: u64,
    pub air_id: u64,
    pub num_instances: usize,
    pub instance_ids: Vec<usize>,
    pub num_columns_trace: u64,
    pub name_columns_trace: Vec<ColumnName>,
    pub num_columns_fixed: u64,
    pub name_columns_fixed: Vec<ColumnName>,
    pub name_airvalues: Vec<ColumnName>,
    pub num_airvalues: u64,
    pub num_rows: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanningInfo {
    pub planning_info: Vec<AirInfo>,
    pub num_instances: usize,
}

struct CancellationThread {
    stop_flag: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CancellationThread {
    fn new(cancellation_info: Arc<RwLock<CancellationInfo>>, mpi_ctx: Arc<MpiCtx>) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();

        let handle = std::thread::spawn(move || loop {
            std::thread::park_timeout(std::time::Duration::from_millis(100));
            if stop_flag_clone.load(Ordering::Relaxed) {
                break;
            }
            if cancellation_info.read_recover().token.is_cancelled() {
                break;
            }
            if let Some(error) = mpi_ctx.check_cancellation() {
                cancellation_info.write_recover().cancel(Some(error));
                break;
            }
        });

        Self { stop_flag, handle: Some(handle) }
    }
}

impl Drop for CancellationThread {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

struct WorkerPoolGuard<T: Send + Clone + 'static> {
    sentinel: T,
    tx: Sender<T>,
    handles: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,
}

impl<T: Send + Clone + 'static> Drop for WorkerPoolGuard<T> {
    fn drop(&mut self) {
        let handles: Vec<std::thread::JoinHandle<()>> = {
            let mut guard = match self.handles.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            if guard.is_empty() {
                return;
            }
            std::mem::take(&mut *guard)
        };
        // One per worker: a pool can hold more workers than streams (the slot commit pool does).
        for _ in 0..handles.len() {
            let _ = self.tx.send(self.sentinel.clone());
        }
        for h in handles {
            let _ = h.join();
        }
    }
}

struct JoinAllGuard {
    handles: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,
    /// Runs once, immediately after the join. Recovery that is only safe when no worker can still
    /// be running (see `set_after_join`); `None` when there is nothing to recover.
    after_join: Option<Box<dyn FnOnce() + Send>>,
}

impl JoinAllGuard {
    fn new(handles: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>) -> Self {
        Self { handles, after_join: None }
    }

    /// Install work to run after the join. Used for state that is shared with the workers and so can
    /// only be reclaimed once they are all gone — e.g. draining the scheduler's queues, where a live
    /// worker could otherwise push a witness in right after the drain. Set after construction
    /// because the state it closes over does not exist yet when the guard is declared, and the guard
    /// must be declared early so it drops last.
    fn set_after_join(&mut self, f: impl FnOnce() + Send + 'static) {
        self.after_join = Some(Box::new(f));
    }
}

impl Drop for JoinAllGuard {
    fn drop(&mut self) {
        let handles: Vec<std::thread::JoinHandle<()>> = {
            let mut guard = match self.handles.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            std::mem::take(&mut *guard)
        };
        for h in handles {
            let _ = h.join();
        }
        if let Some(f) = self.after_join.take() {
            f();
        }
    }
}

/// Sets `proofs_finished` on drop. Declared after `JoinAllGuard` so it *drops before* it, letting
/// generators leave their `select!` loops on any early return; otherwise the guard's unbounded join
/// would wedge the phase.
struct FinishOnDrop(Arc<AtomicBool>);

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// `Arc<Vec<F>>` scratch written by C++ through `&mut [F]`: empty in GPU mode, one writer at a time
/// in CPU mode. Localizes the `unsafe` reborrow and debug-asserts that single-writer invariant.
///
/// The check only means anything between *clones of one instance* — they share the flag. So build
/// one per underlying buffer (`ProofMan::aux_scratch` / `const_scratch`) and clone that; a fresh
/// `new()` per call site would hand every writer its own flag and quietly check nothing.
struct SharedScratch<F> {
    buf: Arc<Vec<F>>,
    #[cfg(debug_assertions)]
    writing: Arc<AtomicBool>,
}

impl<F> Clone for SharedScratch<F> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            #[cfg(debug_assertions)]
            writing: self.writing.clone(),
        }
    }
}

impl<F> SharedScratch<F> {
    fn new(buf: Arc<Vec<F>>) -> Self {
        Self {
            buf,
            #[cfg(debug_assertions)]
            writing: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The underlying buffer, for callers that need to share the `Arc` itself rather than write
    /// through it (e.g. `init_fixed`).
    fn arc(&self) -> &Arc<Vec<F>> {
        &self.buf
    }

    /// Borrow as `&mut [F]` for the guard's lifetime. Sound under the type's single-writer
    /// invariant; debug builds detect a violation.
    fn borrow_mut(&self) -> ScratchGuard<'_, F> {
        // An empty scratch is the GPU mode, where every borrow yields a zero-length slice and there
        // is nothing to alias. Don't track those: GPU runs many workers concurrently, and tripping
        // the assert on provably harmless borrows would just teach people to remove it.
        #[cfg(debug_assertions)]
        let tracked = !self.buf.is_empty();
        #[cfg(debug_assertions)]
        assert!(
            !tracked || !self.writing.swap(true, Ordering::AcqRel),
            "SharedScratch aliased: two writers to a shared scratch buffer at once \
             (the single-writer invariant is violated)"
        );
        // SAFETY: single-writer invariant (see type docs) — this guard is the only live `&mut`.
        let slice = unsafe { std::slice::from_raw_parts_mut(self.buf.as_ptr() as *mut F, self.buf.len()) };
        ScratchGuard {
            slice,
            #[cfg(debug_assertions)]
            writing: self.writing.clone(),
            #[cfg(debug_assertions)]
            tracked,
        }
    }
}

/// RAII guard yielding `&mut [F]` from a [`SharedScratch`]; releases the debug single-writer flag
/// on drop.
struct ScratchGuard<'a, F> {
    slice: &'a mut [F],
    #[cfg(debug_assertions)]
    writing: Arc<AtomicBool>,
    /// Whether this borrow claimed the flag (see `borrow_mut`); an untracked borrow must not
    /// clear a flag it never set.
    #[cfg(debug_assertions)]
    tracked: bool,
}

impl<F> std::ops::Deref for ScratchGuard<'_, F> {
    type Target = [F];
    fn deref(&self) -> &[F] {
        self.slice
    }
}

impl<F> std::ops::DerefMut for ScratchGuard<'_, F> {
    fn deref_mut(&mut self) -> &mut [F] {
        self.slice
    }
}

#[cfg(debug_assertions)]
impl<F> Drop for ScratchGuard<'_, F> {
    fn drop(&mut self) {
        if self.tracked {
            self.writing.store(false, Ordering::Release);
        }
    }
}

struct WitnessGuard {
    witness_tx: Sender<usize>,
    handler: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
    handles: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,
}

impl Drop for WitnessGuard {
    fn drop(&mut self) {
        let handler = {
            let mut guard = match self.handler.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.take()
        };
        let handles: Vec<std::thread::JoinHandle<()>> = {
            let mut guard = match self.handles.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            std::mem::take(&mut *guard)
        };
        if handler.is_none() && handles.is_empty() {
            return;
        }
        // Sentinel wakes the dispatcher; the per-instance threads break on cancellation
        // or natural end-of-work.
        let _ = self.witness_tx.send(usize::MAX);
        if let Some(h) = handler {
            let _ = h.join();
        }
        for h in handles {
            let _ = h.join();
        }
    }
}

#[derive(Debug, Default)]
pub struct CancellationInfo {
    pub token: CancellationToken,
    pub error: Option<ProofmanError>,
}

impl CancellationInfo {
    pub fn cancel(&mut self, error: Option<ProofmanError>) {
        self.token.cancel();
        if self.error.is_some() {
            return;
        }
        if let Some(err) = error {
            self.error = Some(err);
        }
    }

    pub fn reset(&mut self) {
        self.token = CancellationToken::new();
        self.error = None;
    }
}

/// Poison-tolerant access to the shared cancellation hub: recovering the guard avoids a poison
/// storm aborting teardown on every thread. Safe because `token` is atomic and `error` a plain
/// field, so the worst a reader sees after a mid-write panic is a stale error, never UB.
pub trait CancellationInfoExt {
    fn read_recover(&self) -> std::sync::RwLockReadGuard<'_, CancellationInfo>;
    fn write_recover(&self) -> std::sync::RwLockWriteGuard<'_, CancellationInfo>;
}

impl CancellationInfoExt for RwLock<CancellationInfo> {
    fn read_recover(&self) -> std::sync::RwLockReadGuard<'_, CancellationInfo> {
        self.read().unwrap_or_else(|e| e.into_inner())
    }
    fn write_recover(&self) -> std::sync::RwLockWriteGuard<'_, CancellationInfo> {
        self.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// Cancellation re-check interval of a parked contribution worker. A send wakes it immediately, so
/// this is not dispatch latency. Was 1ms: `n_streams` workers waking 1000x/s doing nothing.
const CONTRIB_CANCEL_POLL: std::time::Duration = std::time::Duration::from_millis(25);

/// Streaming-slot commit context for the contributions phase: one pool of free-slot tokens per
/// GPU (a token is the global slot index, gpu * slots_per_gpu + j) plus the packed-trace metadata
/// needed to drive commit_witness_streaming. On the GPU every contribution commits on a slot; a
/// refusal is fatal. Slots are the regions reserved below each GPU's prefetch region; the first
/// GPU's stay usable while gpu-mops borrows its unified buffer.
struct SlotCommitCtx {
    pools: Vec<(Sender<u64>, Receiver<u64>)>,
    /// What the host rows actually are: empty unless the run writes packed traces.
    packed_info: HashMap<(usize, usize), PackedInfo>,
    /// Airs a slot can take. See `slot_commit_airs`.
    airs: std::collections::HashSet<(usize, usize)>,
    committed: AtomicU64,
}

fn stream_commit_eligible<F: PrimeField64>(hash: &str, setup: &Setup<F>) -> bool {
    // witness_calc hints do not disqualify an air: the slot evaluates them (witness_hints_slot.hpp)
    // and refuses, with a reason, only hints it cannot express.
    proofman_common::hash_family::supports_stream_commit(hash)
        && setup.stark_info.stark_struct.merkle_tree_arity == proofman_common::hash_family::merkle_tree_arity(hash)
}

/// Bytes a streaming slot needs to commit this air, or None when it cannot take one.
fn slot_commit_bytes<F: PrimeField64>(hash: &str, setup: &Setup<F>, pi: &PackedInfo) -> Option<u64> {
    if !stream_commit_eligible(hash, setup) {
        return None;
    }
    let &n_cols = setup.stark_info.map_sections_n.get("cm1")?;
    let ss = &setup.stark_info.stark_struct;
    // An unpacked air is the degenerate packing: one word per column.
    let words = if pi.is_packed { pi.num_packed_words } else { n_cols };
    let bytes = stream_commit_slot_bytes_c(ss.n_bits, ss.n_bits_ext, n_cols, words);
    if bytes == 0 {
        return None;
    }
    // Tail room for a column-major copy of the rows (read by unpack and the multiplicity scatter).
    // Reserved for every non-indexed packed air so the copy is never refused mid-commit, which would
    // corrupt the global challenge. The indexed layout and unpacked rows take no copy.
    let tail = if pi.is_packed && pi.col_source.is_empty() { (1u64 << ss.n_bits) * words * 8 } else { 0 };
    Some(bytes + tail)
}

/// The slot size (the largest air's commit) and every air a slot can take, packed or not: an air
/// without packing info (virtual tables, Rom), or any air of a run that writes unpacked traces,
/// commits with the identity packing. `packed_info` must describe the host rows (see
/// `slot_packed_info`). The third value names the airs no slot can take (family, arity, width).
fn slot_commit_airs<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    packed_info: &HashMap<(usize, usize), PackedInfo>,
) -> (u64, std::collections::HashSet<(usize, usize)>, Vec<String>) {
    let identity = PackedInfo::default();
    let mut slot_bytes = 0u64;
    let mut airs = std::collections::HashSet::new();
    let mut unfit = Vec::new();
    for (ag, group) in pctx.global_info.airs.iter().enumerate() {
        for (ai, air) in group.iter().enumerate() {
            let Ok(setup) = sctx.get_setup(ag, ai) else { continue };
            let pi = packed_info.get(&(ag, ai)).unwrap_or(&identity);
            match slot_commit_bytes(&pctx.global_info.hash, setup, pi) {
                Some(b) => {
                    slot_bytes = slot_bytes.max(b);
                    airs.insert((ag, ai));
                }
                None => unfit.push(format!("{} [{ag}:{ai}]", air.name)),
            }
        }
    }
    (slot_bytes, airs, unfit)
}

/// The packing info the slots must read the host rows with: none unless the run writes packed traces.
fn slot_packed_info(options: &ProofmanOptions) -> HashMap<(usize, usize), PackedInfo> {
    if options.packed {
        options.packed_info.clone()
    } else {
        HashMap::new()
    }
}

pub struct ProofMan<F: PrimeField64> {
    pctx: Arc<ProofCtx<F>>,
    sctx: Arc<SetupCtx<F>>,
    mpi_ctx: Arc<MpiCtx>,
    setups: Arc<SetupsVadcop<F>>,
    recurser_setups: RwLock<HashMap<String, Arc<Setup<F>>>>,
    recurser_const_offset: u64,
    recurser_device_registered: Mutex<Option<String>>,
    /// Folds write the instance-shared scratch buffers (`aux_trace`, const
    /// slices) through FFI, so they must be single-flight per instance.
    recurser_fold_lock: Mutex<()>,
    wcm: Arc<WitnessManager<F>>,
    n_streams: usize,
    n_streams_non_recursive: usize,
    memory_handler: Arc<MemoryHandler<F>>,
    memory_handler_recursive_witness: Arc<MemoryHandlerRecursive<F>>,
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
    /// The two scratch buffers above, wrapped for the writers that reborrow them as `&mut [F]`.
    /// One instance each, cloned to every writer, so the debug single-writer check actually spans
    /// them (see [`SharedScratch`]). Never rebuilt: `aux_trace`/`const_pols` are set once in `new`.
    aux_scratch: SharedScratch<F>,
    const_scratch: SharedScratch<F>,
    max_num_threads: usize,
    num_threads_per_witness: usize,
    thread_budget: Arc<ThreadBudget>,
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
    /// Owns the single proof-done callback registration. Each phase takes a `CompletionOwner` and
    /// releases it on drop, so exactly one is live at a time (see `completion.rs`).
    completions: DeviceCompletions,
    outer_aggregation_state: Mutex<OuterAggregationState>,
    outer_agg_proofs_finished: Arc<AtomicBool>,
    total_outer_agg_proofs: Arc<Counter>,
    received_agg_proofs: Arc<RwLock<Vec<Vec<usize>>>>,
    handle_recursives: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,
    handle_contributions: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,
    worker_contributions: Arc<RwLock<Vec<ContributionsInfo>>>,
    cancellation_info: Arc<RwLock<CancellationInfo>>,
    witness_info: RwLock<WitnessInfo>,
    options: ProofmanOptions,

    /// Serializes proof-generation entry points. Use `acquire_computing()`.
    computing: Mutex<()>,
}

#[derive(Debug, PartialEq, Clone, BorshSerialize, BorshDeserialize)]
pub enum ProvePhase {
    Contributions,
    Internal,
    Full,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct ContributionsInfo {
    pub challenge: Vec<u64>,
    pub airgroup_id: usize,
    pub worker_index: u32,
    pub aggregated: bool,
}

#[derive(Debug, BorshSerialize, BorshDeserialize)]
pub enum ProvePhaseInputs {
    Contributions(),
    Internal(Vec<ContributionsInfo>),
    Full(),
}

#[derive(Debug)]
pub enum ProvePhaseResult {
    Contributions(Vec<ContributionsInfo>),
    Internal(Vec<AggProofs>),
    Full(Option<String>, Option<VadcopFinalProof>),
}

enum OuterAggregationState {
    Idle,
    /// Holds the completion capability for the service's lifetime; dropping the owner (back to
    /// `Idle`) releases the callback registration and disconnects the consumer threads.
    Running(CompletionOwner),
}

impl<F: PrimeField64> Drop for ProofMan<F> {
    fn drop(&mut self) {
        self.memory_handler.cancel();
        self.memory_handler_recursive_witness.cancel();
        if let Err(e) = self.reset() {
            eprintln!("Error during ProofMan cleanup: {:?}", e);
        }
        free_device_buffers_c(self.pctx.get_device_buffers_ptr());
    }
}

impl<F: PrimeField64> ProofMan<F> {
    fn ensure_outer_aggregations_started(&self)
    where
        GoldilocksQuinticExtension: ExtensionField<F>,
    {
        let mut outer_aggregation_state = self.outer_aggregation_state.lock().unwrap();
        if matches!(*outer_aggregation_state, OuterAggregationState::Running(_))
            || self.cancellation_info.read_recover().token.is_cancelled()
        {
            return;
        }

        // `outer_aggregations()` acquires the owner, which spins until any prior owner drops (under
        // this same lock in `stop_outer_aggregations`). The `Running(_)` check above guarantees we
        // are `Idle`, so the acquire can't block here — keep that check immediately before this.
        *outer_aggregation_state = OuterAggregationState::Running(self.outer_aggregations());
    }

    fn stop_outer_aggregations(&self) {
        // Take the owner out (leaving the service Idle) so the state lock is not held across the
        // owner's bounded drain in `Drop`.
        let owner = {
            let mut outer_aggregation_state = self.outer_aggregation_state.lock().unwrap();
            match std::mem::replace(&mut *outer_aggregation_state, OuterAggregationState::Idle) {
                OuterAggregationState::Idle => return,
                OuterAggregationState::Running(owner) => owner,
            }
        };

        // Stop the generator threads, then drop the owner: its drain-then-release `Drop` clears the
        // callback registration and disconnects the consumer threads (no sentinel sends needed).
        self.outer_agg_proofs_finished.store(true, Ordering::SeqCst);
        drop(owner);

        let handles = self.handle_recursives.lock().unwrap().drain(..).collect::<Vec<_>>();
        for handle in handles {
            let _ = handle.join();
        }
    }

    pub fn get_options(&self) -> ProofmanOptions {
        self.options.clone()
    }

    pub fn reset(&self) -> ProofmanResult<()> {
        self.wcm.reset();
        // A cancelled job can leave an instance handed back or deferred; the next job reuses ids.
        for flags in [&self.pctx.dispatch_deferred, &self.pctx.dispatch_pending] {
            for f in flags.iter() {
                f.store(false, Ordering::Relaxed);
            }
        }
        for s in self.pctx.witness_staged.iter() {
            s.store(WITNESS_NOT_STAGED, Ordering::Relaxed);
        }

        for proof_lock in self.proofs.iter() {
            let mut proof = proof_lock.write().unwrap_or_else(|e| e.into_inner());
            *proof = None;
        }

        for proof_lock in self.compressor_proofs.iter() {
            let mut proof = proof_lock.write().unwrap_or_else(|e| e.into_inner());
            *proof = None;
        }

        for proof_lock in self.recursive1_proofs.iter() {
            let mut proof = proof_lock.write().unwrap_or_else(|e| e.into_inner());
            *proof = None;
        }

        for proof_lock in self.recursive2_proofs.iter() {
            let mut proofs = proof_lock.write().unwrap_or_else(|e| e.into_inner());
            proofs.clear();
        }

        let mut ongoing_proofs = self.recursive2_proofs_ongoing.write().unwrap_or_else(|e| e.into_inner());
        ongoing_proofs.clear();

        self.pctx.set_witness_tx(None);
        self.pctx.set_witness_tx_priority(None);
        self.pctx.set_proof_tx(None);

        // Releases the completion capability and joins the recursive workers. They disconnect only
        // when their owner is dropped, so they must be joined through it, not via a sentinel send.
        self.stop_outer_aggregations();

        for _ in 0..self.handle_contributions.lock().unwrap_or_else(|e| e.into_inner()).len() {
            self.contributions_tx.send(usize::MAX).ok();
        }

        let handles = self.handle_contributions.lock().unwrap_or_else(|e| e.into_inner()).drain(..).collect::<Vec<_>>();
        for handle in handles {
            let _ = handle.join();
        }
        // The zone outlives the job; a leftover staging would match the next job's commit by instance id.
        // Safe here: all commits are joined.
        if self.pctx.gpu {
            for instance_id in self.pctx.dctx_get_process_instances() {
                release_staged_witness_c(self.pctx.get_device_buffers_ptr(), instance_id as u64);
            }
        }

        // Drain all relevant channels to ensure they are empty
        while self.witness_rx.try_recv().is_ok() {}
        while self.witness_rx_priority.try_recv().is_ok() {}
        while self.contributions_rx.try_recv().is_ok() {}
        while self.proofs_rx.try_recv().is_ok() {}

        // The three witness channels carry `Proof`s whose `trace` came out of the recursive
        // witness pools, so they must be drained by RETURNING those buffers, not by dropping them.
        // A cancel leaves undelivered witnesses here, and dropping them shrinks the pools for the
        // rest of the process — the compressor pool first, since it is the smallest — which then
        // trips (or silently under-fills) the pool-integrity check in `reset()` below.
        for rx in [&self.compressor_witness_rx, &self.rec1_witness_rx, &self.rec2_witness_rx] {
            while let Ok(mut w) = rx.try_recv() {
                drop(self.memory_handler_recursive_witness.adopt_trace(std::mem::take(&mut w.trace)));
            }
        }

        self.worker_contributions.write().unwrap_or_else(|e| e.into_inner()).clear();
        reset_device_streams_c(self.pctx.get_device_buffers_ptr());
        if self.pctx.gpu {
            unsafe { mul_alloc_c(self.pctx.get_device_buffers_ptr()) };
            // Record which airs the device can produce whole; see `device_owned_table_airs`.
            if let Ok(layouts) = collect_virtual_table_layouts(&self.pctx, &self.sctx) {
                let mut owned = self.pctx.device_owned_table_airs.write().unwrap();
                owned.clear();
                for l in layouts {
                    if mul_air_device_owned_c(l.air_id) {
                        owned.push(l.air_id as usize);
                    }
                }
                if !owned.is_empty() {
                    tracing::info!("Virtual tables: air(s) {:?} are produced and committed on the device", owned);
                }
            }
        }
        // Not in VirtualTableAir::execute: it is fork-exposed under --asm, where this FFI call kills
        // the process silently.
        mul_reset_c();

        for inner_vec in self.received_agg_proofs.write().unwrap_or_else(|e| e.into_inner()).iter_mut() {
            inner_vec.clear();
        }

        self.thread_budget.reset();

        self.total_outer_agg_proofs.reset();

        // Shared buffers must be returned to the pool, else its capacity shrinks and
        // memory_handler.reset() fails its `free.len() == n_buffers` invariant. No `?` inside the
        // sweep: bailing on the first bad release would leave every later instance's buffer
        // unreturned, converting one failure into a pool-wide shortfall.
        let mut first_err = None;
        for instance_id in 0..MAX_INSTANCES as usize {
            let (is_shared, buf) = self.pctx.free_instance(instance_id);
            if is_shared {
                if let Err(e) = self.memory_handler.release_buffer(buf) {
                    tracing::error!("reset: failed to return instance {instance_id}'s shared buffer: {e}");
                    first_err = first_err.or(Some(e));
                }
            }
        }

        // Both handlers get reset even if the first fails: each clears its own sticky `cancelled`
        // flag, and skipping that leaves the next run allocating fresh unpinned buffers without bound.
        let basic = self.memory_handler.reset();
        let recursive = self.memory_handler_recursive_witness.reset();
        for result in [basic, recursive] {
            if let Err(e) = result {
                first_err = first_err.or(Some(e));
            }
        }

        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl<F: PrimeField64> ProofMan<F>
where
    GoldilocksQuinticExtension: ExtensionField<F>,
{
    pub fn get_wcm(&self) -> Arc<WitnessManager<F>> {
        self.wcm.clone()
    }

    pub fn get_proving_key_path(&self) -> PathBuf {
        self.pctx.global_info.get_proving_key_path()
    }

    pub fn get_device_buffers_ptr(&self) -> *mut c_void {
        self.pctx.get_device_buffers_ptr()
    }

    pub fn get_preallocated_buffers(&self) -> (Arc<Vec<F>>, *mut c_void, Arc<AtomicBool>) {
        (self.aux_trace.clone(), self.pctx.get_device_buffers_ptr(), self.pctx.reload_fixed_pols_gpu.clone())
    }

    pub fn set_barrier(&self) {
        self.mpi_ctx.barrier();
    }

    pub fn rank(&self) -> Option<i32> {
        (self.pctx.mpi_ctx.n_processes > 1).then(|| self.mpi_ctx.rank)
    }

    pub fn mpi_broadcast(&self, buf: &mut Vec<u8>) {
        self.pctx.mpi_ctx.broadcast(buf);
    }

    pub fn get_rank_info(&self) -> RankInfo {
        self.pctx.get_rank_info()
    }

    pub fn get_n_processes(&self) -> i32 {
        self.pctx.mpi_ctx.n_processes
    }

    pub fn get_setup(&self, airgroup_id: usize, air_id: usize) -> ProofmanResult<&Setup<F>> {
        self.sctx.get_setup(airgroup_id, air_id)
    }

    pub fn is_first_process(&self) -> bool {
        self.pctx.dctx_is_first_process()
    }

    pub fn get_witness_info(&self) -> WitnessInfo {
        self.witness_info.read().unwrap().clone()
    }

    pub fn get_publics(&self) -> Vec<u8> {
        self.pctx.get_publics().iter().flat_map(|x| x.as_canonical_u64().to_le_bytes()).collect()
    }

    pub fn split_active_processes(&self, is_active: bool) {
        self.pctx.mpi_ctx.split_active_processes(is_active);
    }

    fn check_cancel(&self, notify_mpi: bool) -> ProofmanResult<()> {
        let local_cancelled = self.cancellation_info.read_recover().token.is_cancelled();

        let cluster_cancelled =
            if notify_mpi { !self.mpi_ctx.all_finished_ok(!local_cancelled) } else { local_cancelled };

        if !cluster_cancelled {
            return Ok(());
        }

        // Cancellation confirmed: unblock parked workers before we join/reset.
        self.cancel_memory_handlers();

        let error = {
            let mut info = self.cancellation_info.write_recover();
            if !info.token.is_cancelled() {
                info.cancel(Some(ProofmanError::MpiCancellation("peer rank reported cancellation".into())));
            }
            info.error.take()
        };

        let error = if let Some(e) = error {
            if !matches!(e, ProofmanError::MpiCancellation(_)) && notify_mpi {
                tracing::warn!("Notifying error to other MPI processes: {:?}", e);
                self.mpi_ctx.notify_cancellation();
            }
            Err(e)
        } else {
            Err(ProofmanError::Cancelled)
        };
        self.reset()?;
        error
    }

    pub fn cancel(&self) {
        let mut cancellation_info = self.cancellation_info.write_recover();
        cancellation_info.cancel(None);
        self.cancel_memory_handlers();
    }

    /// Unblock any worker parked in a buffer-pool take() so teardown doesn't hang on
    /// a buffer that will never be released. Must run before joining such workers.
    fn cancel_memory_handlers(&self) {
        self.memory_handler.cancel();
        self.memory_handler_recursive_witness.cancel();
    }

    /// Acquire `computing`. Warns if the wait exceeded 50ms.
    fn acquire_computing(&self, caller: &'static str) -> std::sync::MutexGuard<'_, ()> {
        let t0 = std::time::Instant::now();
        let g = self.computing.lock().unwrap_or_else(|e| e.into_inner());
        let waited = t0.elapsed();
        if waited.as_millis() > 50 {
            tracing::warn!("[ProofMan::{caller}] blocked {}ms acquiring `computing`", waited.as_millis());
        }
        g
    }

    /// Block until any in-flight proof-generation call has returned. Call
    /// from the worker's recovery path before advertising `Ready`.
    pub fn wait_until_proofman_ready(&self) {
        let _computing = self.acquire_computing("wait_until_proofman_ready");
    }

    pub fn notify_cancellation(&self) {
        self.mpi_ctx.notify_cancellation();
    }

    pub fn check_setup(
        proving_key_path: PathBuf,
        aggregation: bool,
        verbose_mode: VerboseMode,
        gpu: bool,
    ) -> ProofmanResult<()> {
        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(ProofmanError::InvalidParameters(format!(
                "Proving key folder not found at path: {proving_key_path:?}"
            )));
        }

        let mpi_ctx = Arc::new(MpiCtx::new());

        let pctx = ProofCtx::<F>::create_ctx(proving_key_path, aggregation, verbose_mode, mpi_ctx, gpu)?;

        let setups_aggregation = Arc::new(SetupsVadcop::<F>::new(&pctx.global_info, false, aggregation, gpu)?);

        let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false, gpu)?;

        proofman_common::init_gpu_setup(&pctx.global_info.hash, gpu)?;

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                calculate_fixed_tree(sctx.get_setup(airgroup_id, air_id)?);
            }
        }

        if aggregation {
            let sctx_compressor = setups_aggregation.sctx_compressor.as_ref().unwrap();
            for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                        calculate_fixed_tree(sctx_compressor.get_setup(airgroup_id, air_id)?);
                    }
                }
            }

            let sctx_recursive1 = setups_aggregation.sctx_recursive1.as_ref().unwrap();
            for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    calculate_fixed_tree(sctx_recursive1.get_setup(airgroup_id, air_id)?);
                }
            }

            let sctx_recursive2 = setups_aggregation.sctx_recursive2.as_ref().unwrap();
            let n_airgroups = pctx.global_info.air_groups.len();
            for airgroup in 0..n_airgroups {
                calculate_fixed_tree(sctx_recursive2.get_setup(airgroup, 0)?);
            }

            let setup_vadcop_final = setups_aggregation.setup_vadcop_final.as_ref().unwrap();
            calculate_fixed_tree(setup_vadcop_final);
            if let Some(setup_vadcop_final_compressed) = setups_aggregation.setup_vadcop_final_compressed.as_ref() {
                calculate_fixed_tree(setup_vadcop_final_compressed);
            }
        }

        Ok(())
    }

    pub fn set_partition(
        &self,
        n_partitions: usize,
        partition_ids: Vec<u32>,
        worker_index: usize,
    ) -> ProofmanResult<()> {
        self.pctx.dctx_setup(n_partitions, partition_ids, worker_index)
    }

    pub fn execute(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        output_path: Option<PathBuf>,
        verbose_mode: VerboseMode,
    ) -> ProofmanResult<PlanningInfo> {
        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, Some(self.get_rank_info()))?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);

        self.register_witness(&mut *witness_lib, library)?;

        self.execute_(output_path)
    }

    pub fn execute_from_lib(&self, output_path: Option<PathBuf>) -> ProofmanResult<PlanningInfo> {
        // A statically linked witness library never goes through `register_witness`.
        self.register_prover_multiplicities()?;
        self.execute_(output_path)
    }

    pub fn execute_(&self, output_path: Option<PathBuf>) -> ProofmanResult<PlanningInfo> {
        let _computing = self.acquire_computing("execute_");

        self.set_partition(1, vec![0], 0)?;

        self.cancellation_info.write_recover().reset();
        self.reset()?;
        self.pctx.dctx_reset();

        let _ = self.exec()?;

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
                        percentage_area: 0f64,
                        instance_count: 0,
                        percentage_instances: 0f64,
                        instance_ids: Vec::new(),
                    },
                );
            }
        }

        let mut total_area = 0;
        let mut total_instances = 0;

        for (instance_id, instance_info) in instances.iter().enumerate() {
            let airgroup_id = instance_info.airgroup_id;
            let air_id = instance_info.air_id;

            let air_name = &self.pctx.global_info.airs[airgroup_id][air_id].name;

            let setup = self.sctx.get_setup(airgroup_id, air_id)?;
            let n_bits = setup.stark_info.stark_struct.n_bits;
            let total_cols: u64 = setup
                .stark_info
                .map_sections_n
                .iter()
                .filter(|(key, _)| *key != "const")
                .map(|(_, value)| *value)
                .sum();
            let area = (1 << n_bits) * total_cols;
            total_area += area;
            total_instances += 1;
            air_info.entry(air_name).and_modify(|info| {
                info.total_area += area;
                info.instance_count += 1;
                info.instance_ids.push(instance_id);
            });
        }

        if let Some(output_path) = output_path {
            let mut wtr = Writer::from_path(output_path)?;

            for info in air_info.values_mut() {
                info.percentage_area = info.total_area as f64 / total_area as f64 * 100f64;
                info.percentage_instances = info.instance_count as f64 / total_instances as f64 * 100f64;
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
                percentage_instances: f64,
                total_area: u64,
                percentage_area: f64,
            }

            wtr.serialize(Summary {
                version: env!("CARGO_PKG_VERSION").to_string(),
                name: "TOTAL".into(),
                airgroup_id: None,
                air_id: None,
                percentage_area: 100f64,
                total_area,
                percentage_instances: 100f64,
                total_instances,
            })?;

            wtr.flush()?;
        }

        let mut planning_info = Vec::new();
        for (airgroup_id, _) in self.pctx.global_info.air_groups.iter().enumerate() {
            for (air_id, air) in self.pctx.global_info.airs[airgroup_id].iter().enumerate() {
                let setup = self.sctx.get_setup(airgroup_id, air_id)?;
                let air_name = &air.name;
                if let Some(info) = air_info.get(air_name) {
                    let num_columns_trace = setup.stark_info.map_sections_n["cm1"];
                    let name_columns_trace: Vec<ColumnName> = setup
                        .stark_info
                        .cm_pols_map
                        .as_ref()
                        .map(|pols| pols.iter().filter(|pol| pol.stage == 1).map(ColumnName::new).collect())
                        .unwrap();

                    let name_columns_fixed: Vec<ColumnName> = setup
                        .stark_info
                        .const_pols_map
                        .as_ref()
                        .map(|pols| pols.iter().map(ColumnName::new).collect())
                        .unwrap();

                    let name_airvalues: Vec<ColumnName> = setup
                        .stark_info
                        .airvalues_map
                        .as_ref()
                        .map(|pols| pols.iter().filter(|pol| pol.stage == 1).map(ColumnName::new).collect())
                        .unwrap();

                    planning_info.push(AirInfo {
                        name: air.name.clone(),
                        airgroup_id: airgroup_id as u64,
                        air_id: air_id as u64,
                        num_instances: info.instance_count,
                        instance_ids: info.instance_ids.clone(),
                        num_columns_trace,
                        name_columns_trace,
                        num_columns_fixed: setup.stark_info.n_constants,
                        name_columns_fixed,
                        num_airvalues: setup
                            .stark_info
                            .airvalues_map
                            .as_ref()
                            .map_or(0, |pols| pols.iter().filter(|pol| pol.stage == 1).count() as u64),
                        name_airvalues,
                        num_rows: air.num_rows,
                    });
                } else {
                    println!("  No execution result found for Air ID: {}", air_id);
                }
            }
        }

        let result = PlanningInfo { planning_info, num_instances: total_instances };

        Ok(result)
    }

    pub fn get_instance_fixed(
        &self,
        instance_id: usize,
        first_row: usize,
        num_rows: usize,
        offset: Option<usize>,
    ) -> ProofmanResult<Vec<RowInfo>> {
        let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(instance_id)?;
        let setup = self.sctx.get_setup(airgroup_id, air_id)?;

        let mut const_pols: Vec<F> = create_buffer_fast(setup.const_pols_size);
        load_const_pols(setup, &mut const_pols);

        let offset = offset.unwrap_or(1);
        let n_constants = setup.stark_info.n_constants as usize;
        let num_rows_available = const_pols.len() / n_constants;

        Ok((0..num_rows)
            .map(|i| first_row + i * offset)
            .take_while(|&row| row < num_rows_available)
            .map(|row| {
                let start = row * n_constants;
                let end = start + n_constants;
                let values = const_pols[start..end].iter().map(|v| F::as_canonical_u64(v)).collect();
                RowInfo { row, values }
            })
            .collect())
    }

    pub fn get_instance_trace(
        &self,
        instance_id: usize,
        first_row: usize,
        num_rows: usize,
        offset: Option<usize>,
    ) -> ProofmanResult<Vec<RowInfo>> {
        let _computing = self.acquire_computing("get_instance_trace");
        // Residency, not "dispatched": a reclaimed instance used to answer with an empty trace.
        if self.pctx.is_air_instance_stored(instance_id) {
            return Ok(self.pctx.get_air_instance_trace(instance_id, first_row, num_rows, offset));
        }
        // A query wants it built from scratch, unlike a trace the pipeline already consumed.
        self.pctx.dctx_rearm_witness(instance_id);

        self.wcm.pre_calculate_witness(1, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
        self.wcm.calculate_witness(1, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;

        let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(instance_id)?;
        Self::initialize_air_instance(
            &self.pctx,
            &self.sctx,
            instance_id,
            true,
            true,
            Some(&self.const_scratch),
            Some(&self.aux_trace),
        )?;
        let setup = self.sctx.get_setup(airgroup_id, air_id)?;
        let steps_params = self.pctx.get_air_instance_params(instance_id, false);

        calculate_witness_expressions_c((&setup.p_setup).into(), (&steps_params).into());

        let is_shared_buffer = self.pctx.is_shared_buffer(instance_id);
        if is_shared_buffer {
            self.memory_handler.to_be_released_buffer(instance_id);
        }

        Ok(self.pctx.get_air_instance_trace(instance_id, first_row, num_rows, offset))
    }

    pub fn get_instance_air_values(&self, instance_id: usize) -> ProofmanResult<Vec<u64>> {
        let _computing = self.acquire_computing("get_instance_air_values");
        let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(instance_id)?;
        let setup = self.sctx.get_setup(airgroup_id, air_id)?;
        let airvalues_map = setup.stark_info.airvalues_map.as_ref().unwrap();

        if self.pctx.is_air_instance_stored(instance_id) {
            return self.pctx.get_instance_air_values(instance_id, airvalues_map);
        }
        self.pctx.dctx_rearm_witness(instance_id);

        self.wcm.pre_calculate_witness(1, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
        self.wcm.calculate_witness(1, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;

        Self::initialize_air_instance(
            &self.pctx,
            &self.sctx,
            instance_id,
            true,
            true,
            Some(&self.const_scratch),
            Some(&self.aux_trace),
        )?;
        let steps_params = self.pctx.get_air_instance_params(instance_id, false);

        calculate_witness_expressions_c((&setup.p_setup).into(), (&steps_params).into());

        let is_shared_buffer = self.pctx.is_shared_buffer(instance_id);
        if is_shared_buffer {
            self.memory_handler.to_be_released_buffer(instance_id);
        }

        self.pctx.get_instance_air_values(instance_id, airvalues_map)
    }

    pub fn compute_witness(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        debug_info: &DebugInfo,
        verbose_mode: VerboseMode,
        options: ProofOptions,
    ) -> ProofmanResult<()> {
        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, Some(self.get_rank_info()))?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.pctx.set_debug_info(debug_info);

        self.register_witness(&mut *witness_lib, library)?;

        self.compute_witness_(options)
    }

    /// Computes only the witness without generating a proof neither verifying constraints.
    /// This is useful for debugging or benchmarking purposes.
    pub fn compute_witness_from_lib(&self, debug_info: &DebugInfo, options: ProofOptions) -> ProofmanResult<()> {
        // A statically linked witness library never goes through `register_witness`.
        self.register_prover_multiplicities()?;
        self.pctx.set_debug_info(debug_info);
        self.compute_witness_(options)
    }

    pub fn compute_witness_(&self, options: ProofOptions) -> ProofmanResult<()> {
        let _computing = self.acquire_computing("compute_witness_");

        self.set_partition(1, vec![0], 0)?;

        self.cancellation_info.write_recover().reset();
        self.reset()?;
        self.pctx.dctx_reset();

        if !options.minimal_memory {
            self.pctx.set_witness_tx(Some(self.witness_tx.clone()));
            self.pctx.set_witness_tx_priority(Some(self.witness_tx_priority.clone()));
        }

        let witness_done = Arc::new(Counter::new());

        let (witness_handler, witness_handles) = self.calc_witness_handler(
            witness_done.clone(),
            self.memory_handler.clone(),
            options.minimal_memory,
            None,
            true,
        );

        let _witness_guard = WitnessGuard {
            witness_tx: self.witness_tx.clone(),
            handler: witness_handler.clone(),
            handles: witness_handles.clone(),
        };

        let _ = self.exec()?;

        let my_instances_no_tables = self.pctx.dctx_get_process_instances_no_tables();

        timer_start_info!(CALCULATING_WITNESS);
        self.calculate_witness(
            &my_instances_no_tables,
            self.memory_handler.clone(),
            witness_done.clone(),
            options.minimal_memory,
            true,
        )?;
        timer_stop_and_log_info!(CALCULATING_WITNESS);

        if !options.minimal_memory {
            self.pctx.set_witness_tx(None);
            self.pctx.set_witness_tx_priority(None);
        }

        self.witness_tx.send(usize::MAX).ok();

        if let Some(h) = witness_handler.lock().unwrap().take() {
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
    pub fn get_debug_info(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        debug_info: &DebugInfo,
        verbose_mode: VerboseMode,
    ) -> ProofmanResult<()> {
        // Check witness_lib path exists
        if !witness_lib_path.exists() {
            return Err(ProofmanError::InvalidParameters(format!(
                "Witness computation dynamic library not found at path: {witness_lib_path:?}"
            )));
        }

        // Check input data path
        if let Some(ref input_data_path) = input_data_path {
            if !input_data_path.exists() {
                return Err(ProofmanError::InvalidParameters(format!(
                    "Input data file not found at path: {input_data_path:?}"
                )));
            }
        }

        // Check public_inputs_path is a folder
        if let Some(ref publics_path) = public_inputs_path {
            if !publics_path.exists() {
                return Err(ProofmanError::InvalidParameters(format!(
                    "Public inputs file not found at path: {publics_path:?}"
                )));
            }
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, Some(self.get_rank_info()))?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);

        self.register_witness(&mut *witness_lib, library)?;

        self._get_debug_info(debug_info)
    }

    pub fn get_debug_info_from_lib(&self, debug_info: &DebugInfo) -> ProofmanResult<()> {
        // A statically linked witness library never goes through `register_witness`.
        self.register_prover_multiplicities()?;
        self._get_debug_info(debug_info)
    }

    fn _get_debug_info(&self, debug_info: &DebugInfo) -> ProofmanResult<()> {
        let _computing = self.acquire_computing("_get_debug_info");

        self.set_partition(1, vec![0], 0)?;

        self.pctx.set_debug_info(debug_info);
        self.cancellation_info.write_recover().reset();
        self.reset()?;
        self.pctx.dctx_reset();

        self.exec()?;

        let mut transcript = new_transcript::<F>(&self.pctx.global_info.hash);
        let dummy_element = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        transcript.put(&dummy_element);

        let mut global_challenge = [F::ZERO; 3];
        transcript.get_field(&mut global_challenge);
        self.pctx.set_global_challenge(2, &mut global_challenge);
        transcript.put(&dummy_element);

        let instances = self.pctx.dctx_get_instances();
        let my_instances = self.pctx.dctx_get_process_instances();
        let mut thread_handle: Option<std::thread::JoinHandle<()>> = None;
        // Each scatters once in `calculate_instance_witness`; a debug-skipped one never does.
        let mut scattered = 0u64;

        for &instance_id in my_instances.iter() {
            let instance_info = instances[instance_id];
            let (skip, _) = skip_prover_instance(&self.pctx, instance_id)?;
            if instance_info.table || skip {
                continue;
            }
            scattered += 1;

            self.wcm.pre_calculate_witness(1, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
            self.wcm.calculate_witness(1, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            Self::initialize_air_instance(
                &self.pctx,
                &self.sctx,
                instance_id,
                true,
                true,
                Some(&self.const_scratch),
                Some(&self.aux_trace),
            )?;
            self.calculate_instance_witness(instance_id)?;
            self.wcm.debug(&[instance_id], debug_info)?;
        }

        let my_instances_tables = self.pctx.dctx_get_my_tables();

        self.export_prover_multiplicities(scattered)?;
        timer_start_info!(CALCULATING_TABLES);
        for instance_id in my_instances_tables.iter() {
            self.wcm.calculate_witness(1, &[*instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
        }
        timer_stop_and_log_info!(CALCULATING_TABLES);

        for instance_id in my_instances_tables.iter() {
            let (skip, _) = skip_prover_instance(&self.pctx, *instance_id)?;

            if skip || !self.pctx.dctx_is_my_process_instance(*instance_id)? {
                continue;
            };

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            Self::initialize_air_instance(
                &self.pctx,
                &self.sctx,
                *instance_id,
                true,
                true,
                Some(&self.const_scratch),
                Some(&self.aux_trace),
            )?;
            self.calculate_instance_witness(*instance_id)?;
            self.wcm.debug(&[*instance_id], debug_info)?;
        }

        self.wcm.end(debug_info)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_proof_constraints(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        debug_info: &DebugInfo,
        verbose_mode: VerboseMode,
    ) -> ProofmanResult<()> {
        // Check witness_lib path exists
        if !witness_lib_path.exists() {
            return Err(ProofmanError::InvalidParameters(format!(
                "Witness computation dynamic library not found at path: {witness_lib_path:?}"
            )));
        }

        // Check input data path
        if let Some(ref input_data_path) = input_data_path {
            if !input_data_path.exists() {
                return Err(ProofmanError::InvalidParameters(format!(
                    "Input data file not found at path: {input_data_path:?}"
                )));
            }
        }

        // Check public_inputs_path is a folder
        if let Some(ref publics_path) = public_inputs_path {
            if !publics_path.exists() {
                return Err(ProofmanError::InvalidParameters(format!(
                    "Public inputs file not found at path: {publics_path:?}"
                )));
            }
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, Some(self.get_rank_info()))?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);

        self.register_witness(&mut *witness_lib, library)?;

        self._verify_proof_constraints(debug_info)
    }

    pub fn verify_proof_constraints_from_lib(&self, debug_info: &DebugInfo) -> ProofmanResult<()> {
        // A statically linked witness library never goes through `register_witness`.
        self.register_prover_multiplicities()?;
        self._verify_proof_constraints(debug_info)
    }

    fn _verify_proof_constraints(&self, debug_info: &DebugInfo) -> ProofmanResult<()> {
        timer_start_info!(VERIFYING_PROOF_CONSTRAINTS);

        let _computing = self.acquire_computing("_verify_proof_constraints");

        self.set_partition(1, vec![0], 0)?;

        self.pctx.set_debug_info(debug_info);
        self.cancellation_info.write_recover().reset();
        self.reset()?;
        self.pctx.dctx_reset();

        let _ = self.exec()?;

        let mut transcript = new_transcript::<F>(&self.pctx.global_info.hash);
        let dummy_element = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        transcript.put(&dummy_element);

        let mut global_challenge = [F::ZERO; 3];
        transcript.get_field(&mut global_challenge);
        self.pctx.set_global_challenge(2, &mut global_challenge);
        transcript.put(&dummy_element);

        let witness_done = Arc::new(Counter::new());

        self.pctx.set_proof_tx(Some(self.contributions_tx.clone()));

        let minimal_memory = true;

        let my_instances = self.pctx.dctx_get_process_instances();
        let airgroup_values_air_instances = Arc::new(Mutex::new(vec![Vec::new(); my_instances.len()]));
        let valid_constraints = Arc::new(AtomicBool::new(true));

        let _contributions_guard = WorkerPoolGuard {
            sentinel: usize::MAX,
            tx: self.contributions_tx.clone(),
            handles: self.handle_contributions.clone(),
        };

        for _ in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let memory_handler_clone = self.memory_handler.clone();
            let contributions_rx_clone = self.contributions_rx.clone();
            let cancellation_info_clone = self.cancellation_info.clone();
            let valid_constraints = valid_constraints.clone();
            let airgroup_values_air_instances = airgroup_values_air_instances.clone();
            let wcm_clone = self.wcm.clone();
            let debug_info_clone = debug_info.clone();
            // Reuse the process-wide aux_trace / const_pols buffers: CPU verify runs one instance
            // at a time; in GPU mode they are empty Vecs and host-side init is skipped. Clone the
            // process-wide `const_scratch` so its single-writer check spans every writer.
            let aux_trace_arc = self.aux_trace.clone();
            let const_scratch = self.const_scratch.clone();
            let contribution_handle = std::thread::spawn(move || loop {
                match contributions_rx_clone.recv_timeout(CONTRIB_CANCEL_POLL) {
                    Ok(instance_id) => {
                        if instance_id == usize::MAX {
                            break;
                        }
                        if cancellation_info_clone.read_recover().token.is_cancelled() {
                            break;
                        }
                        if let Err(e) = Self::process_verify_constraints_instance(
                            &pctx_clone,
                            &sctx_clone,
                            memory_handler_clone.clone(),
                            &wcm_clone,
                            instance_id,
                            &debug_info_clone,
                            valid_constraints.clone(),
                            airgroup_values_air_instances.clone(),
                            &const_scratch,
                            &aux_trace_arc,
                        ) {
                            cancellation_info_clone.write_recover().cancel(Some(e));
                            break;
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        if cancellation_info_clone.read_recover().token.is_cancelled() {
                            break;
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                }
            });

            self.handle_contributions.lock().unwrap().push(contribution_handle);
        }

        let (witness_handler, witness_handles) =
            self.calc_witness_handler(witness_done.clone(), self.memory_handler.clone(), minimal_memory, None, false);

        let _witness_guard = WitnessGuard {
            witness_tx: self.witness_tx.clone(),
            handler: witness_handler.clone(),
            handles: witness_handles.clone(),
        };

        let my_instances_no_tables = my_instances
            .iter()
            .filter(|idx| {
                !self.pctx.dctx_is_table(**idx)
                    && skip_prover_instance(&self.pctx, **idx).map(|(skip, _)| !skip).unwrap_or(false)
            })
            .copied()
            .collect::<Vec<_>>();

        timer_start_debug!(CALCULATING_WITNESS);
        self.calculate_witness(
            &my_instances_no_tables,
            self.memory_handler.clone(),
            witness_done.clone(),
            minimal_memory,
            false,
        )?;
        timer_stop_and_log_debug!(CALCULATING_WITNESS);

        if let Some(h) = witness_handler.lock().unwrap().take() {
            h.join().unwrap();
        }
        if self.pctx.gpu {
            let handles_to_join = witness_handles.lock().unwrap().drain(..).collect::<Vec<_>>();
            for handle in handles_to_join {
                handle.join().unwrap();
            }
        }

        drop(witness_handles);

        let my_instances_tables = self
            .pctx
            .dctx_get_my_tables()
            .into_iter()
            .filter(|idx| skip_prover_instance(&self.pctx, *idx).map(|(skip, _)| !skip).unwrap_or(false))
            .collect::<Vec<_>>();

        self.export_prover_multiplicities(my_instances_no_tables.len() as u64)?;
        timer_start_debug!(CALCULATING_TABLES);

        for instance_id in my_instances_tables.iter() {
            self.wcm.pre_calculate_witness(1, &[*instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
            self.wcm.calculate_witness(1, &[*instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
        }

        timer_stop_and_log_debug!(CALCULATING_TABLES);

        self.pctx.set_proof_tx(None);

        for _ in 0..self.handle_contributions.lock().unwrap_or_else(|e| e.into_inner()).len() {
            self.contributions_tx.send(usize::MAX).ok();
        }

        let handles = self.handle_contributions.lock().unwrap().drain(..).collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }

        self.check_cancel(true)?;

        self.wcm.end(debug_info)?;

        let check_global_constraints = !debug_info.skip_prover_instances
            && debug_info.std_mode.debug_values.is_empty()
            && (debug_info.debug_instances.is_empty() || !debug_info.debug_global_instances.is_empty());

        if check_global_constraints {
            let airgroup_values_air_instances = airgroup_values_air_instances.lock().unwrap();
            let airgroupvalues_u64 = aggregate_airgroupvals(&self.pctx, &airgroup_values_air_instances)?;
            let airgroupvalues = self.mpi_ctx.distribute_airgroupvalues(airgroupvalues_u64, &self.pctx.global_info);

            if self.mpi_ctx.rank == 0 {
                let valid_global_constraints =
                    verify_global_constraints_proof(&self.pctx, &self.sctx, debug_info, airgroupvalues);

                timer_stop_and_log_info!(VERIFYING_PROOF_CONSTRAINTS);
                if valid_constraints.load(Ordering::Relaxed) && valid_global_constraints.is_ok() {
                    return Ok(());
                } else {
                    return Err(ProofmanError::InvalidProof("Constraints were not verified".into()));
                }
            }
        }

        timer_stop_and_log_info!(VERIFYING_PROOF_CONSTRAINTS);

        if !valid_constraints.load(Ordering::Relaxed) {
            return Err(ProofmanError::InvalidProof("Constraints were not verified".into()));
        }

        Ok(())
    }

    fn calculate_instance_witness(&self, instance_id: usize) -> ProofmanResult<()> {
        let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(instance_id)?;
        let setup = self.sctx.get_setup(airgroup_id, air_id)?;
        let steps_params = self.pctx.get_air_instance_params(instance_id, false);

        calculate_witness_expressions_c((&setup.p_setup).into(), (&steps_params).into());

        // The prove path counts from its commit hook, which never runs here.
        // CPU only: on the GPU path the trace is on the device and this pointer must not be read.
        if !self.pctx.gpu {
            unsafe {
                mul_scatter_c((&setup.p_setup).into(), (&steps_params).into(), airgroup_id as u64, air_id as u64)
            };
        }

        #[cfg(feature = "diagnostic")]
        {
            let invalid_initialization = Self::diagnostic_instance(&self.pctx, &self.sctx, instance_id)?;
            if invalid_initialization {
                return Err(ProofmanError::InvalidProof("Invalid initialization".into()));
            }
        }

        self.wcm.calculate_witness(2, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;

        calculate_impols_expressions_c((&setup.p_setup).into(), 2, (&steps_params).into());

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn process_verify_constraints_instance(
        pctx: &Arc<ProofCtx<F>>,
        sctx: &Arc<SetupCtx<F>>,
        memory_handler: Arc<MemoryHandler<F>>,
        wcm: &Arc<WitnessManager<F>>,
        instance_id: usize,
        debug_info: &DebugInfo,
        valid_constraints: Arc<AtomicBool>,
        airgroup_values_air_instances: Arc<Mutex<Vec<Vec<F>>>>,
        const_scratch: &SharedScratch<F>,
        aux_trace: &Arc<Vec<F>>,
    ) -> ProofmanResult<()> {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
        Self::initialize_air_instance(pctx, sctx, instance_id, true, true, Some(const_scratch), Some(aux_trace))?;

        let setup = sctx.get_setup(airgroup_id, air_id)?;
        let steps_params = pctx.get_air_instance_params(instance_id, false);

        let custom_commits_fixed_path = match setup.stark_info.custom_commits.iter().find(|c| c.stage_widths[0] > 0) {
            Some(c) => pctx.get_custom_commits_fixed_buffer(&c.name, true)?.to_string_lossy().into_owned(),
            None => String::new(),
        };

        let stream_id = initialize_instance_c(
            (&setup.p_setup).into(),
            airgroup_id as u64,
            air_id as u64,
            instance_id as u64,
            (&steps_params).into(),
            pctx.get_device_buffers_ptr(),
            &custom_commits_fixed_path,
        );

        pctx.set_instance_stream_id(instance_id, stream_id);

        if !pctx.gpu {
            calculate_witness_expressions_c((&setup.p_setup).into(), (&steps_params).into());
            #[cfg(feature = "diagnostic")]
            {
                let invalid_initialization = Self::diagnostic_instance(pctx, sctx, instance_id)?;
                if invalid_initialization {
                    return Err(ProofmanError::InvalidProof("Invalid initialization".into()));
                }
            }
        }

        if !pctx.gpu {
            wcm.calculate_witness(2, &[instance_id], 1, memory_handler.as_ref())?;
            calculate_impols_expressions_c((&setup.p_setup).into(), 2, (&steps_params).into());
        } else {
            calculate_trace_instance_c(
                (&setup.p_setup).into(),
                airgroup_id as u64,
                air_id as u64,
                (&steps_params).into(),
                pctx.get_device_buffers_ptr(),
                stream_id,
            );
        }

        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id)?;
        let airgroup_values = pctx.get_air_instance_airgroup_values(airgroup_id, air_id, air_instance_id)?;
        airgroup_values_air_instances.lock().unwrap()[pctx.dctx_get_instance_local_idx(instance_id)?] =
            airgroup_values.clone();

        wcm.debug(&[instance_id], debug_info)?;

        // Verify-constraints never runs the commit hook that counts on the prove path. Must follow the
        // constraint evaluation, which populates the const pols read here. CPU only.
        if !pctx.gpu {
            unsafe {
                mul_scatter_c((&setup.p_setup).into(), (&steps_params).into(), airgroup_id as u64, air_id as u64)
            };
        }

        let valid =
            verify_constraints_proof(pctx, sctx, instance_id, debug_info.n_print_constraints as u64, stream_id)?;

        if !valid {
            valid_constraints.fetch_and(valid, Ordering::Relaxed);
        }

        let (is_shared_buffer, witness_buffer) = pctx.free_instance(instance_id);
        if is_shared_buffer {
            memory_handler.release_buffer(witness_buffer)?;
        }
        Ok(())
    }

    /// Proves a single AIR standing alone: the transcript is seeded from its own verkey + publics,
    /// so the proof verifies by itself without a global challenge.
    pub fn generate_air_proof(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        air_name: &str,
        verbose_mode: VerboseMode,
        verify: bool,
    ) -> ProofmanResult<()> {
        if !witness_lib_path.exists() {
            return Err(ProofmanError::InvalidParameters(format!(
                "Witness computation dynamic library not found at path: {witness_lib_path:?}"
            )));
        }

        if let Some(ref publics_path) = public_inputs_path {
            if !publics_path.exists() {
                return Err(ProofmanError::InvalidParameters(format!(
                    "Public inputs file not found at path: {publics_path:?}"
                )));
            }
        }

        if self.options.verify_constraints {
            return Err(ProofmanError::InvalidParameters(
                "Proofman has been initialized in verify_constraints mode".into(),
            ));
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, Some(self.get_rank_info()))?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);
        self.register_witness(&mut *witness_lib, library)?;

        self.generate_air_proof_from_lib(air_name, verify)
    }

    pub fn generate_air_proof_from_lib(&self, air_name: &str, verify: bool) -> ProofmanResult<()> {
        // A statically linked witness library never goes through `register_witness`.
        self.register_prover_multiplicities()?;
        let _computing = self.acquire_computing("generate_air_proof");

        self.set_partition(1, vec![0], 0)?;
        self.cancellation_info.write_recover().reset();
        self.reset()?;
        self.pctx.dctx_reset();

        let _ = self.exec()?;

        // Each custom commit's tree root is a public input; without this they stay zero.
        Self::set_publics_custom_commits(&self.sctx, &self.pctx)?;

        // Resolve the requested AIR by name over the planned instances of this process.
        let instances = self.pctx.dctx_get_process_instances();
        let mut targets: Vec<(usize, usize, usize)> = Vec::new();
        for &instance_id in instances.iter() {
            let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(instance_id)?;
            if self.pctx.global_info.get_air_name(airgroup_id, air_id) == air_name {
                targets.push((instance_id, airgroup_id, air_id));
            }
        }

        if targets.is_empty() {
            // Only now is the name list worth building.
            let available: BTreeSet<&str> = instances
                .iter()
                .filter_map(|&id| self.pctx.dctx_get_instance_info(id).ok())
                .map(|(ag, air)| self.pctx.global_info.get_air_name(ag, air))
                .collect();
            return Err(ProofmanError::InvalidParameters(format!(
                "No planned instance of air '{air_name}'. Planned airs: [{}]",
                available.into_iter().collect::<Vec<_>>().join(", ")
            )));
        }

        for &(instance_id, airgroup_id, air_id) in targets.iter() {
            tracing::info!("··· Proving {air_name} (instance #{instance_id}) standalone, without global challenge");

            timer_start_info!(WITNESS_GENERATION);
            self.wcm.pre_calculate_witness(1, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
            self.wcm.calculate_witness(1, &[instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
            timer_stop_and_log_info!(WITNESS_GENERATION);
            self.check_cancel(false)?;

            timer_start_info!(GENERATING_AIR_PROOF);
            let stream_id = Self::gen_proof(
                &self.proofs,
                &self.pctx,
                &self.sctx,
                instance_id,
                &self.aux_trace,
                &self.const_pols,
                &self.const_tree,
                None,
                true,
            )?;
            if self.pctx.gpu {
                // gen_proof is async on GPU: the proof buffer is only filled once the stream drains.
                get_stream_id_proof_c(self.pctx.get_device_buffers_ptr(), stream_id as u64);
            }
            timer_stop_and_log_info!(GENERATING_AIR_PROOF);

            // Safe now that the stream is drained: the H2D copy of this trace is done.
            if self.pctx.is_shared_buffer(instance_id) {
                self.memory_handler.to_be_released_buffer(instance_id);
            }

            let proof =
                self.proofs[instance_id].write().unwrap().take().ok_or_else(|| {
                    ProofmanError::ProofmanError(format!("No proof produced for instance {instance_id}"))
                })?;

            if verify {
                timer_start_info!(VERIFYING_AIR_PROOF);
                let setup_path = self.pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Basic);
                // global_challenge = None: the verifier reseeds from verkey + publics, matching
                // the self-contained prover transcript.
                let valid = verify_proof::<F>(
                    &proof.proof,
                    setup_path.display().to_string() + ".starkinfo.json",
                    setup_path.display().to_string() + ".verifier.bin",
                    setup_path.display().to_string() + ".verkey.json",
                    Some(self.pctx.get_publics().clone()),
                    Some(self.pctx.get_proof_values().clone()),
                    None,
                );
                timer_stop_and_log_info!(VERIFYING_AIR_PROOF);

                if !valid {
                    tracing::info!(
                        "··· {}",
                        format!("\u{2717} Proof of {air_name} (instance #{instance_id}) was NOT verified")
                            .bright_red()
                            .bold()
                    );
                    return Err(ProofmanError::InvalidProof(format!(
                        "Proof of {air_name} (instance #{instance_id}) failed verification"
                    )));
                }
                tracing::info!(
                    "    {}",
                    format!("\u{2713} Proof of {air_name} (instance #{instance_id}) was verified")
                        .bright_green()
                        .bold()
                );
            }

            drop(proof);
        }

        Ok(())
    }

    pub fn generate_proof(
        &self,
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        verbose_mode: VerboseMode,
        proof_options: ProofOptions,
    ) -> ProofmanResult<ProvePhaseResult> {
        // Check witness_lib path exists
        if !witness_lib_path.exists() {
            return Err(ProofmanError::InvalidParameters(format!(
                "Witness computation dynamic library not found at path: {witness_lib_path:?}"
            )));
        }

        // Check input data path
        if let Some(ref input_data_path) = input_data_path {
            if !input_data_path.exists() {
                return Err(ProofmanError::InvalidParameters(format!(
                    "Input data file not found at path: {input_data_path:?}"
                )));
            }
        }

        // Check public_inputs_path is a folder
        if let Some(ref publics_path) = public_inputs_path {
            if !publics_path.exists() {
                return Err(ProofmanError::InvalidParameters(format!(
                    "Public inputs file not found at path: {publics_path:?}"
                )));
            }
        }

        timer_start_info!(CREATE_WITNESS_LIB);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(verbose_mode, Some(self.get_rank_info()))?;
        timer_stop_and_log_info!(CREATE_WITNESS_LIB);

        self.wcm.set_public_inputs_path(public_inputs_path);

        self.register_witness(&mut *witness_lib, library)?;

        if self.options.verify_constraints {
            return Err(ProofmanError::InvalidParameters(
                "Proofman has been initialized in verify_constraints mode".into(),
            ));
        }

        if proof_options.aggregation && !self.options.aggregation {
            return Err(ProofmanError::InvalidParameters(
                "Proofman has not been initialized in aggregation mode".into(),
            ));
        }

        self.set_partition(1, vec![0], 0)?;
        self._generate_proof(ProvePhaseInputs::Full(), proof_options, ProvePhase::Full)
    }

    pub fn register_recurser_setup(&self, recurser_id: &str, recurser_path_stem: &Path) -> ProofmanResult<()> {
        let _computing = self.acquire_computing("register_recurser_setup");
        {
            let cache = self.recurser_setups.read().unwrap();
            if cache.contains_key(recurser_id) {
                return Ok(());
            }
        }

        let _fold_guard = self.recurser_fold_lock.lock().unwrap();
        let mut device_slot = self.recurser_device_registered.lock().unwrap();
        {
            let cache = self.recurser_setups.read().unwrap();
            if cache.contains_key(recurser_id) {
                return Ok(());
            }
        }

        let vadcop_final_stem = self.pctx.global_info.get_setup_path("vadcop_final");
        let air_info = GlobalInfoAir::new(format!("recurser_aggregator_{recurser_id}"));
        let setup = Setup::<F>::new(
            recurser_path_stem,
            0,
            0,
            &air_info,
            &ProofType::RecurserAggregator,
            false,
            self.options.gpu,
            Some(&vadcop_final_stem),
            &self.options.custom_commits_fixed,
        )?;

        tracing::info!(
            "Preparing const-tree for recurser-aggregator setup '{recurser_id}' ({} mode)",
            if self.options.gpu { "GPU" } else { "CPU" }
        );

        let d_buffers = if self.options.gpu { Some(self.pctx.get_device_buffers_ptr()) } else { None };
        check_const_pols_gpu(&setup)?;
        check_const_tree(&setup, &d_buffers)?;

        let setup = Arc::new(setup);

        if self.options.gpu {
            match device_slot.as_deref() {
                Some(existing) if existing != recurser_id => {
                    tracing::info!(
                        "recurser '{existing}' currently occupies the GPU const slot; \
                         '{recurser_id}' will be swapped in on demand at prove time"
                    );
                }
                Some(_) => {}
                None => {
                    self.load_recurser_setup_on_device(&setup)?;
                    *device_slot = Some(recurser_id.to_string());
                }
            }
        }

        tracing::info!(
            "Registered recurser-aggregator setup '{recurser_id}' (files at {:?}, starkinfo borrowed from {:?})",
            recurser_path_stem,
            vadcop_final_stem,
        );

        let mut cache = self.recurser_setups.write().unwrap();
        cache.entry(recurser_id.to_string()).or_insert_with(|| setup);
        drop(cache);
        drop(device_slot);
        Ok(())
    }

    fn load_recurser_setup_on_device(&self, setup: &Setup<F>) -> ProofmanResult<()> {
        let packed_len_bytes = std::fs::metadata(&setup.const_pols_path)
            .map_err(|e| {
                ProofmanError::InvalidSetup(format!(
                    "recurser packed const pols missing at {}: {e}",
                    setup.const_pols_path
                ))
            })?
            .len();
        let packed_len = packed_len_bytes / 8;
        let slot = self.setups.recurser_const_slot_size as u64;
        if packed_len > slot {
            return Err(ProofmanError::InvalidSetup(format!(
                "recurser packed const pols ({packed_len} elements) exceed the reserved GPU slot ({slot})"
            )));
        }
        let proof_type: &str = setup.setup_type.into();
        let d_buffers_ptr = self.pctx.get_device_buffers_ptr();
        setup.load_device(0, 0, d_buffers_ptr, std::ptr::null_mut());
        load_device_const_pols_c(
            0,
            0,
            self.recurser_const_offset,
            d_buffers_ptr,
            &setup.const_pols_path,
            packed_len,
            proof_type,
            false,
            // Recursers swap through a single reserved slot; each one really is uploaded.
            false,
        );
        Ok(())
    }

    pub fn prove_recurser_aggregator(
        &self,
        recurser_id: &str,
        proof_a: &VadcopFinalProof,
        proof_b: &VadcopFinalProof,
        free_inputs_a: &[u64],
        free_inputs_b: &[u64],
        root_c_recurser_agg: &[u64; 4],
    ) -> ProofmanResult<VadcopFinalProof> {
        let _computing = self.acquire_computing("prove_recurser_aggregator");
        if proof_a.compressed || proof_b.compressed {
            return Err(ProofmanError::InvalidConfiguration(
                "prove_recurser_aggregator: compressed inputs are not supported".to_string(),
            ));
        }

        if proof_a.hash != self.pctx.global_info.hash || proof_b.hash != self.pctx.global_info.hash {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "prove_recurser_aggregator: hash family mismatch: proofs are ({}, {}) but this \
                 prover's proving key uses {}",
                proof_a.hash, proof_b.hash, self.pctx.global_info.hash
            )));
        }

        let setup = {
            let cache = self.recurser_setups.read().unwrap();
            cache.get(recurser_id).cloned().ok_or_else(|| {
                ProofmanError::InvalidParameters(format!(
                    "Recurser id '{recurser_id}' not registered. Call register_recurser_setup first."
                ))
            })?
        };

        let a = proof_a.proof_with_publics();
        let b = proof_b.proof_with_publics();
        let a_body = a.get(1..).unwrap_or(&[]);
        let b_body = b.get(1..).unwrap_or(&[]);

        let expected_body = (setup.stark_info.n_publics + setup.proof_size) as usize;
        for (side, body) in [('a', a_body), ('b', b_body)] {
            if body.len() != expected_body {
                return Err(ProofmanError::InvalidParameters(format!(
                    "prove_recurser_aggregator: proof_{side} body has {} words, expected \
                     {expected_body} (vadcop_final n_publics + proof_size); the proof blob is \
                     malformed or truncated",
                    body.len()
                )));
            }
        }

        let _fold_guard = self.recurser_fold_lock.lock().unwrap();

        // The GPU const slot holds one recurser at a time; swap this one in if another is resident
        // (same-recurser folds pay nothing). Safe under the fold lock: nothing reads the slot while
        // we overwrite it.
        if self.options.gpu {
            let mut device_slot = self.recurser_device_registered.lock().unwrap();
            if device_slot.as_deref() != Some(recurser_id) {
                tracing::info!(
                    "Swapping recurser '{recurser_id}' into the GPU const slot (was {:?})",
                    device_slot.as_deref()
                );
                // Advance the marker only after a confirmed load. A failed load may leave the const
                // buffer partially overwritten, so mark the slot None — else the next fold proves
                // against those partial bytes (a silent wrong proof).
                match self.load_recurser_setup_on_device(&setup) {
                    Ok(()) => *device_slot = Some(recurser_id.to_string()),
                    Err(e) => {
                        *device_slot = None;
                        return Err(e);
                    }
                }
            }
        }

        let raw_proof = crate::generate_recurser_aggregator_proof::<F>(
            &setup,
            &self.memory_handler_recursive_witness,
            a_body,
            b_body,
            free_inputs_a,
            free_inputs_b,
            root_c_recurser_agg,
            &self.aux_trace,
            &self.const_pols,
            &self.const_tree,
            self.pctx.get_device_buffers_ptr(),
            recurser_id,
        )?;

        VadcopFinalProof::new_from_proof(&raw_proof, false, self.pctx.global_info.hash.clone()).map_err(|e| {
            ProofmanError::InvalidConfiguration(format!("Failed to wrap recurser output as VadcopFinalProof: {e}"))
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_proof_from_lib(
        &self,
        phase_inputs: ProvePhaseInputs,
        proof_options: ProofOptions,
        phase: ProvePhase,
    ) -> ProofmanResult<ProvePhaseResult> {
        // A statically linked witness library never goes through `register_witness`.
        self.register_prover_multiplicities()?;
        if self.options.verify_constraints {
            return Err(ProofmanError::InvalidParameters(
                "Proofman has been initialized in verify_constraints mode".into(),
            ));
        }

        if proof_options.aggregation && !self.options.aggregation {
            return Err(ProofmanError::InvalidParameters(
                "Proofman has not been initialized in aggregation mode".into(),
            ));
        }

        self._generate_proof(phase_inputs, proof_options, phase)
    }

    /// Error unless the loaded proving key was built with the `vadcop_final_compressed` stage.
    pub fn ensure_compressed_final_supported(&self) -> ProofmanResult<()> {
        if !self.pctx.global_info.has_compressed_final {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "Proving key {:?} (hash {}) was built without the vadcop_final_compressed stage, \
                 so a compressed final proof cannot be generated. Rebuild the key with \
                 --compressed-final, add the stage with `setup-compressed-final`, or ask for an \
                 uncompressed vadcop final proof.",
                self.get_proving_key_path(),
                self.pctx.global_info.hash,
            )));
        }
        Ok(())
    }

    pub fn generate_vadcop_final_proof_compressed(
        &self,
        vadcop_final_proof: &VadcopFinalProof,
    ) -> ProofmanResult<VadcopFinalProof> {
        let _computing = self.acquire_computing("generate_vadcop_final_proof_compressed");
        self.ensure_compressed_final_supported()?;
        if vadcop_final_proof.compressed {
            return Err(ProofmanError::InvalidConfiguration(
                "Cannot generate a compressed vadcop proof from an already compressed vadcop proof".to_string(),
            ));
        }

        let vadcop_final_proof_compressed = generate_vadcop_final_compressed_proof(
            &self.pctx,
            &self.memory_handler_recursive_witness,
            &self.setups,
            &vadcop_final_proof.proof_with_publics(),
            &self.aux_trace,
            &self.const_pols,
            &self.const_tree,
        )?;

        VadcopFinalProof::new_from_proof(&vadcop_final_proof_compressed.proof, true, self.pctx.global_info.hash.clone())
            .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to create VadcopFinalProof: {}", e)))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(proving_key_path: PathBuf, options: ProofmanOptions) -> ProofmanResult<Self> {
        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(ProofmanError::InvalidParameters(format!(
                "Proving key folder not found at path: {proving_key_path:?}"
            )));
        }

        // Check proving_key_path is a folder
        if !proving_key_path.is_dir() {
            return Err(ProofmanError::InvalidParameters(format!(
                "Proving key parameter must be a folder: {proving_key_path:?}"
            )));
        }

        let mpi_ctx = Arc::new(MpiCtx::new());

        let rank_info =
            RankInfo { world_rank: mpi_ctx.rank, local_rank: mpi_ctx.node_rank, n_processes: mpi_ctx.n_processes };
        initialize_logger(options.verbose_mode, Some(&rank_info));

        let (
            pctx,
            sctx,
            setups_vadcop,
            n_streams_per_gpu,
            n_aggregation_workers_per_gpu,
            n_gpus,
            recurser_const_offset,
        ) = Self::initialize_proofman(mpi_ctx.clone(), proving_key_path, &options)?;

        timer_start_info!(INIT_PROOFMAN);

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone()));

        timer_stop_and_log_info!(INIT_PROOFMAN);

        // Basic pool: trace buffers for witness computation. One slot per concurrent
        // witness — only > 1 when --packed enables parallel witness generation.
        let max_witness_stored = if options.packed { n_gpus as usize * options.max_witness_stored } else { 1 };

        // Recursive pool: trace buffers for in-flight recursive proofs. With solve+scatter
        // fused, one buffer covers a proof's whole witness→prove span (no separate witness
        // pool), so queued and proving work share this depth — see `generate_witness`.
        // Recursive trace buffers, compressor included -- they share one pool, so one count. The
        // depth only has to keep the streams fed while the next witnesses are solved, and the old
        // formula (max_witness_stored/2 per GPU, plus 2 per GPU for compressors) put 18 pinned
        // 216 MB buffers behind 2 streams on one GPU. Capped instead.
        let max_witness_stored_recursive = if options.gpu {
            ((n_gpus as usize * options.max_witness_stored) / 2 + n_gpus as usize * 2)
                .clamp(1, MAX_RECURSIVE_TRACE_BUFFERS)
        } else {
            2
        };

        let (max_witness_trace_size, max_witness_trace_size_packed) =
            calculate_max_witness_trace_size(&pctx, &sctx, &options.packed_info, &options.gpu_witness_airs)?;

        let max_buffer_size = if options.packed { max_witness_trace_size_packed } else { max_witness_trace_size };

        // Report what was declared and its effect: both sizings are a `max` over airs, so a
        // declared air that was not the widest frees nothing.
        if !options.gpu_witness_airs.is_empty() {
            let names: Vec<String> =
                options.gpu_witness_airs.iter().map(|a| format!("{}:{}", a.airgroup_id, a.air_id)).collect();
            tracing::info!(
                "GPU witness airs: {} ({}); host trace pool sized to {} without them",
                options.gpu_witness_airs.len(),
                names.join(", "),
                proofman_common::format_bytes((max_buffer_size * std::mem::size_of::<F>()) as f64),
            );
        }

        let n_proof_threads = match options.gpu {
            true => n_gpus,
            false => 1,
        };

        // Workers are not device streams: without a pool they share the basic ones.
        let n_streams = ((n_streams_per_gpu + n_aggregation_workers_per_gpu) * n_proof_threads) as usize;
        let n_streams_non_recursive = (n_streams_per_gpu * n_proof_threads) as usize;

        let memory_handler = Arc::new(MemoryHandler::new(pctx.clone(), max_witness_stored, max_buffer_size));
        // signalValues pool: one size for every buffer, one per concurrent recursive witness
        // (at most `n_streams`, one per worker). The single biggest circuit does not fit and
        // self-allocates -- see `signal_pool_cap`. Reuse needs no zeroing (write-before-read).
        let signal_pool = {
            let cap = setups_vadcop.signal_pool_cap();
            if cap == 0 {
                None
            } else {
                Some((cap, n_streams.max(1)))
            }
        };
        // Threads per recursive witness: 8, regardless of how many streams run concurrently.
        //
        // Capped at 8. Lifting it DOES speed the witness up -- it also drives the scatter, which is
        // row-parallel and memory-latency bound (~20 ns/cell, two dependent loads) -- but it makes
        // the pipeline slower, because the extra threads deschedule the ones driving the GPU.
        // Measured on 1x RTX 5090, 24 cores, 2 streams (PIL2_CIRCOM_TIMERS, 482 warm calls):
        //   threads   witness total   wall clock/proof
        //         8        48.13 ms         52.5 s
        //        12        41.00 ms         54.3 s
        //        24        35.45 ms         55.4 s
        // The GPU is the constraint here (99% utilisation), so witness time is not on the critical
        // path and buying it with cores is a net loss. Re-measure wall clock, not just the phase
        // timers, before raising this.
        let max_num_threads = configured_num_threads(mpi_ctx.node_n_processes as usize);
        let recursive_witness_threads = max_num_threads.clamp(1, 8);
        let memory_handler_recursive_witness = Arc::new(MemoryHandlerRecursive::new_with_signal_pool(
            max_witness_stored_recursive,
            setups_vadcop.max_compact_trace_size,
            signal_pool,
            recursive_witness_threads,
        ));
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

        let (aux_trace, const_pols, const_tree) = if options.gpu {
            (Arc::new(Vec::new()), Arc::new(Vec::new()), Arc::new(Vec::new()))
        } else {
            let mut aux_trace_size = sctx.max_prover_buffer_size.max(setups_vadcop.max_prover_buffer_size);
            if options.aggregation {
                aux_trace_size = aux_trace_size.max(get_recursive_buffer_sizes(&pctx, &setups_vadcop)?);
            }
            (
                Arc::new(vec![F::ZERO; aux_trace_size]),
                Arc::new(vec![F::ZERO; sctx.max_const_size.max(setups_vadcop.max_const_size)]),
                Arc::new(vec![F::ZERO; sctx.max_const_tree_size.max(setups_vadcop.max_const_tree_size)]),
            )
        };

        // Measured on a 24-core host: aggregate witness CPU is flat between 2 and 6 and only
        // degrades past 12, while per-witness latency keeps improving, so the wide end is the
        // safer default. Prove time is insensitive to this either way.
        const DEFAULT_THREADS_PER_WITNESS: usize = 4;
        let num_threads_per_witness = match options.are_threads_per_witness_set {
            true => options.number_threads_pools_witness,
            false => DEFAULT_THREADS_PER_WITNESS.clamp(1, max_num_threads.max(1)),
        };
        tracing::info!("Using {num_threads_per_witness} threads per witness computation");

        let values_contributions: Arc<Vec<Mutex<Vec<F>>>> =
            Arc::new((0..MAX_INSTANCES).map(|_| Mutex::new(Vec::<F>::new())).collect());

        let roots_contributions: Arc<Vec<[F; 4]>> = Arc::new((0..MAX_INSTANCES).map(|_| [F::default(); 4]).collect());

        let thread_budget = Arc::new(ThreadBudget::new(max_num_threads));

        let (witness_tx, witness_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (witness_tx_priority, witness_rx_priority): (Sender<usize>, Receiver<usize>) = unbounded();
        let (contributions_tx, contributions_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (proofs_tx, proofs_rx): (Sender<usize>, Receiver<usize>) = unbounded();
        let (compressor_witness_tx, compressor_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();
        let (rec1_witness_tx, rec1_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();
        let (rec2_witness_tx, rec2_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();

        let received_agg_proofs = Arc::new(RwLock::new((0..n_airgroups).map(|_| Vec::new()).collect::<Vec<Vec<_>>>()));

        let proofman = Self {
            pctx,
            sctx,
            mpi_ctx,
            wcm,
            setups: setups_vadcop,
            recurser_setups: RwLock::new(HashMap::new()),
            recurser_const_offset,
            recurser_device_registered: Mutex::new(None),
            recurser_fold_lock: Mutex::new(()),
            n_streams,
            n_streams_non_recursive,
            max_num_threads,
            num_threads_per_witness,
            memory_handler,
            memory_handler_recursive_witness,
            proofs,
            compressor_proofs,
            recursive1_proofs,
            recursive2_proofs,
            recursive2_proofs_ongoing,
            aux_scratch: SharedScratch::new(aux_trace.clone()),
            const_scratch: SharedScratch::new(const_pols.clone()),
            aux_trace,
            const_pols,
            const_tree,
            roots_contributions,
            values_contributions,
            thread_budget,
            witness_tx,
            witness_rx,
            witness_tx_priority,
            witness_rx_priority,
            contributions_tx,
            contributions_rx,
            completions: DeviceCompletions::new(),
            proofs_tx,
            proofs_rx,
            compressor_witness_tx,
            compressor_witness_rx,
            rec1_witness_tx,
            rec1_witness_rx,
            rec2_witness_tx,
            rec2_witness_rx,
            outer_aggregation_state: Mutex::new(OuterAggregationState::Idle),
            total_outer_agg_proofs: Arc::new(Counter::new()),
            received_agg_proofs,
            handle_recursives: Arc::new(Mutex::new(Vec::new())),
            handle_contributions: Arc::new(Mutex::new(Vec::new())),
            outer_agg_proofs_finished: Arc::new(AtomicBool::new(true)),
            worker_contributions: Arc::new(RwLock::new(Vec::new())),
            cancellation_info: Arc::new(RwLock::new(CancellationInfo::default())),
            options,
            witness_info: RwLock::new(WitnessInfo::default()),
            computing: Mutex::new(()),
        };

        // Fitting reads only the proving key, so do it here rather than inside the first proof.
        // Memoized (see `register_prover_multiplicities`).
        timer_start_info!(FITTING_VIRTUAL_TABLES);
        proofman.register_prover_multiplicities()?;
        // Single rank: a fully prover-owned table can be produced and committed on the device. With
        // several ranks the host must gather every rank's share.
        mul_set_device_export_c(proofman.mpi_ctx.n_processes == 1);
        timer_stop_and_log_info!(FITTING_VIRTUAL_TABLES);

        Ok(proofman)
    }

    pub fn register_custom_commits(&self, custom_commits_fixed: HashMap<String, PathBuf>) -> ProofmanResult<()> {
        let _computing = self.acquire_computing("register_custom_commits");
        self.pctx.initialize_custom_commits(custom_commits_fixed, &self.sctx, CustomCommitValidation::Lenient)?;
        self.ensure_custom_commits_fixed()
    }

    /// Generate any custom-commit file that is missing or not in the packed layout. Only the
    /// witness library can produce them, so this is a no-op until one is registered.
    pub fn ensure_custom_commits_fixed(&self) -> ProofmanResult<()> {
        let pending = self.pctx.custom_commits_pending();
        if pending.is_empty() || !self.wcm.has_witness_lib_components() {
            return Ok(());
        }
        tracing::info!("Generating custom commits fixed: {}", pending.join(", "));
        timer_start_info!(GENERATING_CUSTOM_COMMITS_FIXED);
        // One writer per node: the files are shared, and every rank validates them below.
        if self.mpi_ctx.rank == 0 {
            self.wcm.gen_custom_commits_fixed()?;
        }
        self.mpi_ctx.barrier();
        timer_stop_and_log_info!(GENERATING_CUSTOM_COMMITS_FIXED);
        self.pctx.initialize_custom_commits(
            self.pctx.custom_commits_paths(),
            &self.sctx,
            CustomCommitValidation::Strict,
        )
    }

    /// Upload (per program) the instruction table for an indexed air. `table` is
    /// `num_entries * words_per_entry` packed u64 words.
    ///
    /// Must be called AFTER the setups are loaded (the air's `AirInstanceInfo` has to
    /// exist to receive it) and BEFORE the first commit of that program -- the contribution
    /// phase already unpacks. An air with an indexed descriptor but no table aborts at
    /// unpack rather than silently decoding compact rows as full ones.
    pub fn register_instruction_table(
        &self,
        airgroup_id: usize,
        air_id: usize,
        table: &[u64],
        num_entries: u64,
        words_per_entry: u64,
    ) -> ProofmanResult<()> {
        // An empty descriptor would register nothing, leave d_instr_table null, and only
        // surface as an abort at the first unpack -- far from the cause. Reject it here.
        if num_entries == 0 || words_per_entry == 0 {
            return Err(ProofmanError::InvalidParameters(format!(
                "register_instruction_table [{airgroup_id}:{air_id}]: num_entries ({num_entries}) and \
                 words_per_entry ({words_per_entry}) must both be > 0"
            )));
        }
        // The C side copies num_entries * words_per_entry words out of `table`; a short
        // slice would be an out-of-bounds read there, so reject it here where we can.
        let expected = (num_entries as usize).saturating_mul(words_per_entry as usize);
        if table.len() != expected {
            return Err(ProofmanError::InvalidParameters(format!(
                "register_instruction_table [{airgroup_id}:{air_id}]: table has {} words, expected \
                 num_entries ({num_entries}) * words_per_entry ({words_per_entry}) = {expected}",
                table.len()
            )));
        }
        register_instruction_table_c(
            self.pctx.get_device_buffers_ptr(),
            airgroup_id as u64,
            air_id as u64,
            table,
            num_entries,
            words_per_entry,
        );
        Ok(())
    }

    pub fn register_witness(&self, witness_lib: &mut dyn WitnessLibrary<F>, library: Library) -> ProofmanResult<()> {
        timer_start_info!(REGISTERING_WITNESS);
        // Normally already fitted in `new`; memoized safety net for other construction paths.
        self.register_prover_multiplicities()?;
        witness_lib.register_witness(&self.wcm)?;
        self.wcm.set_init_witness(true, library);
        timer_stop_and_log_info!(REGISTERING_WITNESS);
        // Custom commits registered before the library was loaded could not be generated then.
        self.ensure_custom_commits_fixed()
    }

    /// Hand the prover the range tables it can count itself plus the geometry those counters live
    /// in, and record which it accepted. Must run in the host binary: the witness library links its
    /// own copy of libstarks, so a registration made there is invisible to the scatter and fold.
    fn register_prover_multiplicities(&self) -> ProofmanResult<()> {
        // Reached from `register_witness` and every `*_from_lib` entry; held throughout so a
        // concurrent caller waits instead of registering twice.
        let mut registered = self.pctx.prover_multiplicities_registered.lock().unwrap();
        if *registered {
            return Ok(());
        }
        // What the caller kept for itself. Everything else is the prover's; a table it cannot fit is
        // an error, since the fallback silently costs a full-table pass every proof.
        let std_owned: std::collections::HashSet<u64> = self.options.std_owned_tables.iter().copied().collect();

        let owned: Vec<(u64, i64)> = collect_prover_owned_ranges(&self.pctx, &self.sctx)?
            .into_iter()
            .filter(|(id, _)| !std_owned.contains(id))
            .collect();
        // Range tables (`std_rc_users`) and generic virtual tables (`virtual_table_data_global`) are
        // independent hints, so neither may gate the other's registration.
        let (range_ids, range_biases): (Vec<u64>, Vec<i64>) = owned.into_iter().unzip();
        if !range_ids.is_empty() {
            mul_register_range_tables_c(&range_ids, &range_biases);
        }

        // An exact map over each table's entries, verified against every entry. A table that does not
        // fit must be in `std_owned_tables`.
        let layouts = collect_virtual_table_layouts(&self.pctx, &self.sctx)?;
        let (fitted, vt_summary) = fit_virtual_table_maps(&self.pctx, &self.sctx, &layouts)?;

        // Unkept and unclaimed by both the range path and the fitter (`unclaimed_ids` excludes the
        // fitted ones): fail with the ids rather than run a slow path.
        let range_owned: std::collections::HashSet<u64> = range_ids.iter().copied().collect();
        let orphans: Vec<u64> = vt_summary
            .unclaimed_ids
            .iter()
            .copied()
            .filter(|t| !std_owned.contains(t) && !range_owned.contains(t))
            .collect();
        if !orphans.is_empty() {
            return Err(ProofmanError::InvalidSetup(format!(
                "virtual tables {orphans:?} are neither declared in ProofmanOptions::std_owned_tables \
                 nor derivable by the prover: no row map fits them. Either declare them, so the \
                 witness counts them as before, or make their layout fittable."
            )));
        }

        for m in fitted.iter().filter(|m| !std_owned.contains(&m.table_id)) {
            let (nkey, kv, slots) = &m.map;
            mul_register_table_map_c(m.table_id, kv, *nkey, *slots);
        }

        // Unfitted tables also in `range_ids` are prover-owned via the range path; only the rest are
        // left to the std. Exact-map bytes are GPU-resident, per device.
        if vt_summary.considered > 0 {
            let n_range_in_scope = vt_summary.unclaimed_ids.iter().filter(|t| range_owned.contains(t)).count();
            let n_std_left = vt_summary.unclaimed_ids.len() - n_range_in_scope;
            tracing::info!(
                "Virtual tables: {}/{} prover-owned in {} ms -- {n_range_in_scope} range, {} exact map \
                 ({} MB); {n_std_left} left to the std",
                fitted.len() + n_range_in_scope,
                vt_summary.considered,
                vt_summary.elapsed_ms,
                fitted.len(),
                vt_summary.exact_bytes / 1_000_000,
            );
        }

        for l in &layouts {
            register_mul_vt_c(l.airgroup_id, l.air_id, l.num_rows, l.num_cols, &l.table_ids, &l.acc_bases);
        }

        let migrated = mul_migrated_tables_c();
        // A range table no virtual-table air hosts gets no decoder, so the std keeps counting it.
        let unhosted: Vec<u64> = range_ids.iter().copied().filter(|t| !migrated.contains(t)).collect();
        if !unhosted.is_empty() {
            tracing::warn!("Range tables {unhosted:?} are hosted by no virtual-table air; the std counts them");
        }
        // Read lazily by the std: the virtual table airs are built before this, so a value captured at
        // construction would be empty and every claimed table double-counted.
        pil2_std_lib::set_prover_owned_tables(migrated.clone());
        *self.pctx.prover_owned_tables.write().unwrap() = migrated;
        *registered = true;
        Ok(())
    }

    /// Pull what the prover counted into pctx, for the virtual-table airs to merge in. In the host
    /// binary for the same linkage reason as `register_prover_multiplicities`. `expected_commits` is
    /// how many instances this run scatters: the caller knows which ones its debug filter skipped.
    fn export_prover_multiplicities(&self, expected_commits: u64) -> ProofmanResult<()> {
        if self.pctx.prover_owned_tables.read().unwrap().is_empty() {
            return Ok(());
        }
        // Ordering point: every instance has launched its scatter. Device-owned airs transpose the
        // accumulator into the trace in their own commit.
        mul_sync_commits_c(expected_commits);

        let mut counts = self.pctx.prover_counts.write().unwrap();
        for l in collect_virtual_table_layouts(&self.pctx, &self.sctx)? {
            // Nothing to hand over: the device commits the whole air, or the prover counts none of its
            // tables (building the accumulator would only clear the table for nothing).
            if mul_air_device_owned_c(l.air_id) || !mul_air_has_owned_c(l.air_id) {
                continue;
            }
            let buf = counts.entry(l.air_id as usize).or_insert_with(|| vec![0u64; (l.num_rows * l.num_cols) as usize]);
            // The fold adds into the destination, so clear first: exporting twice must not double.
            buf.iter_mut().for_each(|c| *c = 0);
            unsafe { mul_fold_c(l.air_id, buf.as_mut_ptr()) };
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn _generate_proof(
        &self,
        phase_inputs: ProvePhaseInputs,
        options: ProofOptions,
        phase: ProvePhase,
    ) -> ProofmanResult<ProvePhaseResult> {
        let _computing = self.acquire_computing("_generate_proof");

        // Fail before the inner proofs, not after them: the compressed stage is the last thing a
        // full run does, so an unsupported request would otherwise burn the whole proof first.
        if options.compressed {
            self.ensure_compressed_final_supported()?;
        }

        if phase == ProvePhase::Contributions || phase == ProvePhase::Full {
            if !self.pctx.is_setup_partition_init() {
                return Err(ProofmanError::InvalidParameters(
                    "Setup partition must be initialized before generating contributions".into(),
                ));
            }

            self.cancellation_info.write_recover().reset();
            self.reset()?;
            self.pctx.dctx_reset();
        }

        let _cancellation_thread = CancellationThread::new(self.cancellation_info.clone(), self.mpi_ctx.clone());

        let _contributions_guard = WorkerPoolGuard {
            sentinel: usize::MAX,
            tx: self.contributions_tx.clone(),
            handles: self.handle_contributions.clone(),
        };
        // Join-only backstop for the recursive workers. They exit when the owner drops (consumers)
        // and when `proofs_finished` is set via `FinishOnDrop` (generators). Declared before both so
        // it drops LAST — joining before either has signalled would hang.
        let mut _recursives_guard = JoinAllGuard::new(self.handle_recursives.clone());

        let all_partial_contributions_u64 = if phase == ProvePhase::Contributions || phase == ProvePhase::Full {
            if !options.minimal_memory && self.pctx.gpu {
                self.pctx.set_witness_tx(Some(self.witness_tx.clone()));
                self.pctx.set_witness_tx_priority(Some(self.witness_tx_priority.clone()));
            }
            let witness_done = Arc::new(Counter::new());

            let witness_start_time: Arc<RwLock<Option<std::time::Instant>>> = Arc::new(RwLock::new(None));

            self.pctx.set_proof_tx(Some(self.contributions_tx.clone()));

            let first_contribution_logged = Arc::new(AtomicBool::new(false));

            // Reuse the process-wide aux_trace / const_pols buffers: CPU runs one proof at a time so
            // workers never touch them concurrently; in GPU mode they are empty Vecs. `SharedScratch`
            // localizes the reborrow and debug-asserts the single-writer invariant — clone the
            // process-wide instances so that check spans every writer, not just these workers.
            let aux_scratch = self.aux_scratch.clone();
            let const_scratch = self.const_scratch.clone();

            // Streaming-slot pools: one token per reserved commit slot, per GPU. Contribution workers take
            // a token, commit synchronously on the slot, and return it. The GPU has no other
            // contributions path; None on the CPU backend.
            let slot_commit_ctx: Option<Arc<SlotCommitCtx>> = if self.pctx.gpu {
                let n_slots = get_stream_commit_slots_c(self.pctx.get_device_buffers_ptr());
                (n_slots > 0).then(|| {
                    // This process's GPUs: with several ranks per node the node count is larger.
                    let pools = (0..get_stream_commit_gpus_c(self.pctx.get_device_buffers_ptr()))
                        .map(|g| {
                            let (tx, rx) = unbounded();
                            for j in 0..n_slots {
                                tx.send(g * n_slots + j).unwrap();
                            }
                            (tx, rx)
                        })
                        .collect();
                    let packed_info = slot_packed_info(&self.options);
                    let airs = slot_commit_airs(&self.pctx, &self.sctx, &packed_info).1;
                    Arc::new(SlotCommitCtx { pools, packed_info, airs, committed: AtomicU64::new(0) })
                })
            } else {
                None
            };

            let zone_airs = Arc::new(if self.pctx.gpu { self.zone_staging_airs() } else { HashMap::new() });
            // Twice the slots: a worker stages and prepares on the CPU first, so one extra keeps each slot fed.
            let n_workers = match slot_commit_ctx.as_ref() {
                Some(ctx) => {
                    let n_slots = get_stream_commit_slots_c(self.pctx.get_device_buffers_ptr()) as usize;
                    (2 * n_slots * ctx.pools.len()).max(self.n_streams)
                }
                None => self.n_streams,
            };
            for _ in 0..n_workers {
                let pctx_clone = self.pctx.clone();
                let first_contribution_logged = first_contribution_logged.clone();
                let sctx_clone = self.sctx.clone();
                let values_contributions_clone = self.values_contributions.clone();
                let roots_contributions_clone = self.roots_contributions.clone();
                let memory_handler_clone = self.memory_handler.clone();
                let contributions_rx_clone = self.contributions_rx.clone();
                let cancellation_info_clone = self.cancellation_info.clone();
                let aux_scratch = aux_scratch.clone();
                let const_scratch = const_scratch.clone();
                let slot_commit_ctx_clone = slot_commit_ctx.clone();
                let zone_airs = zone_airs.clone();
                let no_slot_airs: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
                let contribution_handle = std::thread::spawn(move || loop {
                    match contributions_rx_clone.recv_timeout(CONTRIB_CANCEL_POLL) {
                        Ok(instance_id) => {
                            if instance_id == usize::MAX {
                                break;
                            }
                            if cancellation_info_clone.read_recover().token.is_cancelled() {
                                break;
                            }
                            // Stage now if the witness thread found the zone full, so the commit still takes one D2D.
                            let mut stage_state =
                                pctx_clone.witness_staged[instance_id].swap(WITNESS_NOT_STAGED, Ordering::AcqRel);
                            if stage_state == WITNESS_NOT_STAGED {
                                stage_state = match Self::stage_for_commit(
                                    &pctx_clone,
                                    &zone_airs,
                                    &memory_handler_clone,
                                    slot_commit_ctx_clone.as_ref().map_or(&no_slot_airs, |c| &c.airs),
                                    instance_id,
                                ) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        cancellation_info_clone.write_recover().cancel(Some(e));
                                        break;
                                    }
                                };
                            }
                            // The host buffer is gone: the commit reads the zone and releases nothing.
                            let staged = stage_state & WITNESS_STAGED_KIND == WITNESS_STAGED_RELEASED;
                            // Single-writer borrow of the shared scratch (see `SharedScratch`); the
                            // guards hold the invariant for the duration of get_contribution_air.
                            let mut aux_trace_local = aux_scratch.borrow_mut();
                            let mut const_pols_local = const_scratch.borrow_mut();
                            if let Err(e) = Self::get_contribution_air(
                                &pctx_clone,
                                &sctx_clone,
                                &roots_contributions_clone,
                                &values_contributions_clone,
                                instance_id,
                                &mut aux_trace_local,
                                &mut const_pols_local,
                                slot_commit_ctx_clone.as_deref(),
                                staged,
                                witness_staged_gpu(stage_state),
                            ) {
                                cancellation_info_clone.write_recover().cancel(Some(e));
                                break;
                            }

                            if pctx_clone.gpu && !first_contribution_logged.swap(true, Ordering::Relaxed) {
                                tracing::info!("First GPU contribution queued");
                            }

                            // Release a staging no commit consumed (the commit ran before the witness thread staged it);
                            // otherwise it holds its slot for the rest of the run. No-op when consumed.
                            if pctx_clone.gpu {
                                release_staged_witness_c(pctx_clone.get_device_buffers_ptr(), instance_id as u64);
                            }
                            // A slot commit is synchronous, and the CPU commit reads the host buffer in
                            // place: either way it is free to recycle now.
                            let is_shared_buffer = pctx_clone.is_shared_buffer(instance_id) && !staged;
                            if is_shared_buffer {
                                memory_handler_clone.to_be_released_buffer(instance_id);
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            if cancellation_info_clone.read_recover().token.is_cancelled() {
                                break;
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                });
                self.handle_contributions.lock().unwrap().push(contribution_handle);
            }

            let (witness_handler, witness_handles) = self.calc_witness_handler(
                witness_done.clone(),
                self.memory_handler.clone(),
                options.minimal_memory,
                Some(witness_start_time.clone()),
                false,
            );

            let _witness_guard = WitnessGuard {
                witness_tx: self.witness_tx.clone(),
                handler: witness_handler.clone(),
                handles: witness_handles.clone(),
            };

            let summary_info = self.exec()?;

            Self::set_publics_custom_commits(&self.sctx, &self.pctx)?;

            // Phase B is per job. A cluster worker never runs VadcopFinal, the only other point
            // that hands the basic stream back, so the previous job can leave the aliases open:
            // its basics would then find no eligible stream and starve here.
            if self.pctx.gpu && self.pctx.phase_b {
                let _ = set_phase_b_c(self.pctx.get_device_buffers_ptr(), 0);
            }

            timer_start_info!(CALCULATING_CONTRIBUTIONS);
            timer_start_debug!(CALCULATING_INNER_CONTRIBUTIONS);
            timer_start_debug!(PREPARING_CONTRIBUTIONS);

            if witness_start_time.read().unwrap().is_none() {
                *witness_start_time.write().unwrap() = Some(std::time::Instant::now());
            }

            timer_stop_and_log_debug!(PREPARING_CONTRIBUTIONS);

            let my_instances_no_tables = self.pctx.dctx_get_process_instances_no_tables();

            timer_start_debug!(CALCULATING_WITNESS);
            self.calculate_witness(
                &my_instances_no_tables,
                self.memory_handler.clone(),
                witness_done.clone(),
                options.minimal_memory,
                false,
            )?;
            timer_stop_and_log_debug!(CALCULATING_WITNESS);

            if !options.minimal_memory && self.pctx.gpu {
                self.pctx.set_witness_tx(None);
                self.pctx.set_witness_tx_priority(None);
            }
            self.witness_tx.send(usize::MAX).ok();

            if let Some(h) = witness_handler.lock().unwrap().take() {
                h.join().unwrap();
            }
            if self.pctx.gpu {
                let handles_to_join = witness_handles.lock().unwrap().drain(..).collect::<Vec<_>>();
                for handle in handles_to_join {
                    handle.join().unwrap();
                }
            }

            drop(witness_handles);

            self.export_prover_multiplicities(my_instances_no_tables.len() as u64)?;
            timer_start_debug!(CALCULATING_TABLES);

            let my_instances_tables = self.pctx.dctx_get_my_tables();

            //evaluate witness for instances of type "tables"
            for instance_id in my_instances_tables.iter() {
                self.wcm.pre_calculate_witness(
                    1,
                    &[*instance_id],
                    self.max_num_threads,
                    self.memory_handler.as_ref(),
                )?;
                self.wcm.calculate_witness(1, &[*instance_id], self.max_num_threads, self.memory_handler.as_ref())?;
            }

            timer_stop_and_log_debug!(CALCULATING_TABLES);

            self.pctx.set_proof_tx(None);

            for _ in 0..self.handle_contributions.lock().unwrap_or_else(|e| e.into_inner()).len() {
                self.contributions_tx.send(usize::MAX).ok();
            }

            let handles = self.handle_contributions.lock().unwrap().drain(..).collect::<Vec<_>>();
            for handle in handles {
                handle.join().unwrap();
            }

            self.check_cancel(true)?;

            // get roots still in the gpu
            get_stream_proofs_c(self.pctx.get_device_buffers_ptr());

            timer_stop_and_log_debug!(CALCULATING_INNER_CONTRIBUTIONS);

            //calculate-challenge
            let internal_contribution = calculate_internal_contributions(
                &self.pctx,
                &self.roots_contributions,
                &self.values_contributions,
                *DEBUG_CHALLENGES,
            );

            if let Some(ctx) = &slot_commit_ctx {
                tracing::info!(
                    "Streaming slots: {} of {} contributions committed on a slot (tables included)",
                    ctx.committed.load(Ordering::Relaxed),
                    my_instances_no_tables.len() + my_instances_tables.len()
                );
            }
            timer_stop_and_log_info!(CALCULATING_CONTRIBUTIONS);

            let contributions_size = match self.pctx.global_info.curve {
                CurveType::None => self.pctx.global_info.lattice_size.unwrap(),
                _ => 10,
            };

            let all_internal_partial_contributions = self.mpi_ctx.distribute_roots(internal_contribution);
            let all_internal_partial_contributions_split: Vec<Vec<F>> = all_internal_partial_contributions
                .chunks(contributions_size)
                .map(|chunk| chunk.iter().map(|&x| F::from_u64(x)).collect())
                .collect();

            let internal_contribution = aggregate_contributions(&self.pctx, &all_internal_partial_contributions_split);

            let internal_contribution_u64: Vec<u64> =
                internal_contribution.iter().map(|&x| x.as_canonical_u64()).collect::<Vec<u64>>();

            if phase == ProvePhase::Contributions {
                let witness_time =
                    witness_start_time.read().unwrap().map(|start| start.elapsed().as_millis() as f32).unwrap_or(0.0);

                *self.witness_info.write().unwrap() = WitnessInfo {
                    publics: self.pctx.get_publics().clone().into_iter().map(|p| p.as_canonical_u64()).collect(),
                    proof_values: self
                        .pctx
                        .get_proof_values()
                        .clone()
                        .into_iter()
                        .map(|p| p.as_canonical_u64())
                        .collect(),
                    summary_info,
                    witness_time,
                    total_instances: self.pctx.dctx_get_instances().len(),
                };
                return Ok(ProvePhaseResult::Contributions(vec![ContributionsInfo {
                    challenge: internal_contribution_u64,
                    worker_index: self.pctx.get_worker_index()? as u32,
                    airgroup_id: 0,
                    aggregated: false,
                }]));
            }
            &vec![ContributionsInfo {
                challenge: internal_contribution_u64,
                worker_index: 0,
                airgroup_id: 0,
                aggregated: false,
            }]
        } else {
            match phase_inputs {
                ProvePhaseInputs::Internal(ref contributions) => contributions,
                _ => return Err(ProofmanError::ProofmanError("Internal phase requires Internal phase inputs".into())),
            }
        };

        let n_workers =
            all_partial_contributions_u64.iter().map(|contribution| contribution.worker_index).max().unwrap_or(0) + 1;

        {
            let mut worker_contributions = self.worker_contributions.write().unwrap();
            for contribution in all_partial_contributions_u64 {
                tracing::debug!(
                    "Worker contribution received: worker_index={}, airgroup_id={}, challenge(first 10)={:?}",
                    contribution.worker_index,
                    contribution.airgroup_id,
                    &contribution.challenge[..contribution.challenge.len().min(10)]
                );
                if contribution.worker_index < n_workers {
                    worker_contributions.push(contribution.clone());
                } else {
                    return Err(ProofmanError::ProofmanError("Invalid worker index in contributions".into()));
                }
            }
        }

        let mut global_challenge = calculate_global_challenge(&self.pctx, all_partial_contributions_u64);
        tracing::info!(
            "··· Global challenge: [{}, {}, {}]",
            global_challenge[0],
            global_challenge[1],
            global_challenge[2]
        );
        self.pctx.set_global_challenge(2, &mut global_challenge);

        timer_start_info!(GENERATING_PROOFS);

        timer_start_info!(GENERATING_INNER_PROOFS);

        // Zone stagings match by instance id; drop contribution leftovers (all those commits are done).
        if self.pctx.gpu {
            for instance_id in self.pctx.dctx_get_process_instances() {
                release_staged_witness_c(self.pctx.get_device_buffers_ptr(), instance_id as u64);
            }
        }

        debug_assert_eq!(
            self.pctx.dctx_count_witness_state(proofman_common::WitnessState::Running),
            0,
            "witness hooks must all be joined before the proofs phase"
        );
        // Targeted, unlike the blanket reset it replaces: an instance whose trace is still resident
        // stays `Done` and is reused rather than becoming indistinguishable from never computed.
        let rearmed = self.pctx.dctx_rearm_witnesses();
        let (duplicates, evictions, over_trace) = self.pctx.witness_stats.snapshot();
        tracing::info!(
            "··· Witness reuse: {evictions} traces reclaimed, {rearmed} re-armed for the proofs phase, \
             {duplicates} duplicate dispatches refused, {over_trace} recomputed over a live trace"
        );
        self.memory_handler.empty_queue_to_be_released();

        let n_airgroups = self.pctx.global_info.air_groups.len();

        let instances = self.pctx.dctx_get_instances();
        let mut my_instances = self.pctx.dctx_get_process_instances();

        let mut n_airgroup_proofs = vec![0; n_airgroups];
        for &instance_id in my_instances.iter() {
            let instance_info = instances[instance_id];
            n_airgroup_proofs[instance_info.airgroup_id] += 1;
        }

        if options.aggregation {
            for (airgroup, &n_proofs) in n_airgroup_proofs.iter().enumerate().take(n_airgroups) {
                let n_recursive2_proofs = total_recursive_proofs(n_proofs, self.pctx.global_info.aggregation_arity);
                if n_recursive2_proofs.has_remaining || n_proofs == 0 {
                    let setup = self.setups.get_setup(airgroup, 0, &ProofType::Recursive2)?;
                    let publics_aggregation = n_publics_aggregation(&self.pctx, airgroup);
                    let null_proof_buffer = vec![0; setup.proof_size as usize + publics_aggregation];
                    let null_proof = Proof::new(ProofType::Recursive2, airgroup, 0, None, null_proof_buffer);
                    self.recursive2_proofs[airgroup].write().unwrap().push(null_proof);
                }
            }
        }

        // Tear down any running outer-aggregation service before taking the capability: the Internal
        // phase never reset()s, so it could still hold it and `acquire` would block.
        self.stop_outer_aggregations();
        let completions = self.completions.acquire(DeviceBuffersPtr(self.pctx.get_device_buffers_ptr()));
        let proofs_pending = completions.ledger();

        // Dequeue-ahead prefetch (witness-zone mode): the worker dequeues the next
        // ready basics from the scheduler ahead of dispatch. Only READY
        // (stored/computed) witnesses ever enter the scheduler queue, so a held
        // instance is always launchable. Single basic stream only; default ON with
        // the zone, PROOFMAN_PREFETCH=0 disables.
        let prefetch_dequeue_ahead = self.pctx.gpu
            && self.n_streams_non_recursive == 1
            && std::env::var("PROOFMAN_PREFETCH").map(|v| v != "0").unwrap_or(true);
        self.pctx.set_proof_tx(Some(self.proofs_tx.clone()));

        // Key-affinity recursive scheduler (GPU only; CPU has no streams and uses the witness
        // channels). Condvar parks idle stream workers.
        let scheduler: Option<Arc<crate::SharedScheduler<F>>> = if self.pctx.gpu {
            let mut sched = crate::RecursiveScheduler::<F>::new(self.pctx.get_device_buffers_ptr());
            if self.pctx.phase_b {
                // Dispatch ranks: phase-A-only airs first so the halves open as early as possible (with
                // a compressor first among them: the longest chain), then the other airs with a
                // compressor (basic, CPU witness, compressor, recursive1, recursive2: the chain that
                // ends the block if it starts late), then the rest by backlog.
                let half = self.pctx.phase_b_half as u64;
                let ranks: std::collections::HashMap<(usize, usize), u8> = self
                    .pctx
                    .global_info
                    .airs
                    .iter()
                    .enumerate()
                    .flat_map(|(ag, group)| (0..group.len()).map(move |air| (ag, air)))
                    .filter_map(|(ag, air)| {
                        let phase_a_only =
                            self.sctx.get_setup(ag, air).map(|s| s.prover_buffer_size > half).unwrap_or(false);
                        let has_compressor = self.pctx.global_info.get_air_has_compressor(ag, air);
                        match (phase_a_only, has_compressor) {
                            (true, true) => Some(((ag, air), 0)),
                            (true, false) => Some(((ag, air), 1)),
                            (false, true) => Some(((ag, air), 2)),
                            (false, false) => None,
                        }
                    })
                    .collect();
                sched.set_big_keys(ranks);
            }
            Some(Arc::new(crate::SharedScheduler::new(sched)))
        } else {
            None
        };

        // Recover whatever is still queued when the phase ends. Runs after `_recursives_guard` has
        // joined every producer and consumer — draining earlier would race a worker that pushes right
        // after — and the queued witnesses hold pooled `trace` buffers, so dropping the
        // scheduler with work in it shrinks the recursive witness pools (the compressor pool first:
        // it is the smallest) and the next `reset()` finds them short. Settling their ledger units
        // keeps the epoch's accounting truthful for the release diagnostic.
        if let Some(sched) = scheduler.clone() {
            let ledger = proofs_pending.clone();
            let pctx = self.pctx.clone();
            let memory_handler = self.memory_handler.clone();
            let memory_handler_recursive_witness = self.memory_handler_recursive_witness.clone();
            _recursives_guard.set_after_join(move || {
                let (witnesses, basics) = {
                    let mut guard = sched.lock.lock().unwrap_or_else(|p| p.into_inner());
                    guard.drain_all()
                };
                if !witnesses.is_empty() || !basics.is_empty() {
                    tracing::debug!(
                        "Scheduler drained at teardown: {} witness(es) and {} stored basic(s) never dispatched",
                        witnesses.len(),
                        basics.len()
                    );
                }
                crate::recover_drained_witnesses(witnesses, &memory_handler_recursive_witness, &ledger);
                for instance_id in basics {
                    let (is_shared_buffer, witness_buffer) = pctx.free_instance(instance_id);
                    if is_shared_buffer {
                        if let Err(e) = memory_handler.release_buffer(witness_buffer) {
                            tracing::warn!("Failed to return a drained basic's shared buffer to the pool: {e}");
                        }
                    }
                    ledger.settle(instance_id as u64, ProofType::Basic as usize);
                }
            });
        }

        // Phase B: counts the completions that can only happen in phase A -- the Basic of every
        // instance whose buffer exceeds a half, and every Compressor that does (its proof needs the
        // compact witness tail too, as the class sizing does). When the last lands, the basic stream
        // closes and everything left -- recursion and the small basics -- moves to the two halves
        // (set_phase_b 1). Nothing to wait for means the request goes out at once. Phase A starts
        // fresh on every job.
        let phase_a_remaining: Option<Arc<std::sync::atomic::AtomicI64>> = if self.pctx.phase_b {
            let _ = set_phase_b_c(self.pctx.get_device_buffers_ptr(), 0);
            let half = self.pctx.phase_b_half as u64;
            let mut expected: i64 = 0;
            for &instance_id in my_instances.iter() {
                let (ag, air) = self.pctx.dctx_get_instance_info(instance_id)?;
                if self.sctx.get_setup(ag, air)?.prover_buffer_size > half {
                    expected += 1;
                }
                if self.pctx.global_info.get_air_has_compressor(ag, air) {
                    let compressor_big =
                        self.setups.sctx_compressor.as_ref().and_then(|c| c.get_setup(ag, air).ok()).is_none_or(|s| {
                            let n = 1u64 << s.stark_info.stark_struct.n_bits;
                            s.prover_buffer_size + proofman_common::recursion_staging_cols(s, true) * n > half
                        });
                    expected += compressor_big as i64;
                }
            }
            tracing::debug!("Phase B armed: {expected} phase-A completions expected");
            if expected == 0 {
                let rc = set_phase_b_c(self.pctx.get_device_buffers_ptr(), 1);
                tracing::info!("Phase B requested: no phase-A-only instance in this job (rc={rc})");
            }
            Some(Arc::new(std::sync::atomic::AtomicI64::new(expected)))
        } else {
            None
        };
        for _ in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let memory_handler_recursive_witness = self.memory_handler_recursive_witness.clone();
            let setups_clone = self.setups.clone();
            let sctx_clone = self.sctx.clone();
            let proofs_clone = self.proofs.clone();
            let compressor_proofs_clone = self.compressor_proofs.clone();
            let recursive1_proofs_clone = self.recursive1_proofs.clone();
            let recursive2_proofs_clone = self.recursive2_proofs.clone();
            let recursive2_proofs_ongoing_clone = self.recursive2_proofs_ongoing.clone();
            let proofs_pending_clone = proofs_pending.clone();
            let rec1_witness_tx_clone = self.rec1_witness_tx.clone();
            let rec2_witness_tx_clone = self.rec2_witness_tx.clone();
            let compressor_witness_tx_clone = self.compressor_witness_tx.clone();
            let recursive_rx_clone = completions.receiver();
            let cancellation_info_clone = self.cancellation_info.clone();
            let scheduler_clone = scheduler.clone();
            let phase_a_remaining_clone = phase_a_remaining.clone();
            let handle_recursive = std::thread::spawn(move || {
                // Exits when the owner is dropped and the channel disconnects (no sentinel).
                while let Ok(msg) = recursive_rx_clone.recv() {
                    let id = msg.id;
                    let p: ProofType = msg.proof_type.parse().unwrap();
                    // Settles the outstanding unit this message represents, on any exit of this
                    // iteration — after any downstream child has been armed below.
                    let _settled = proofs_pending_clone.adopt(id, p.as_usize());
                    if cancellation_info_clone.read_recover().token.is_cancelled() {
                        break;
                    }
                    if let Some(remaining) = phase_a_remaining_clone.as_ref() {
                        // Only the phase-A-only completions count (see the countdown's arming).
                        let (ag, air) =
                            pctx_clone.dctx_get_instance_info(id as usize).unwrap_or((usize::MAX, usize::MAX));
                        let counts = ag != usize::MAX
                            && match p {
                                ProofType::Basic => sctx_clone
                                    .get_setup(ag, air)
                                    .map(|s| s.prover_buffer_size > pctx_clone.phase_b_half as u64)
                                    .unwrap_or(false),
                                ProofType::Compressor => setups_clone
                                    .sctx_compressor
                                    .as_ref()
                                    .and_then(|c| c.get_setup(ag, air).ok())
                                    .is_none_or(|s| {
                                        let n = 1u64 << s.stark_info.stark_struct.n_bits;
                                        s.prover_buffer_size + proofman_common::recursion_staging_cols(s, true) * n
                                            > pctx_clone.phase_b_half as u64
                                    }),
                                _ => false,
                            };
                        if counts && remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
                            let rc = set_phase_b_c(pctx_clone.get_device_buffers_ptr(), 1);
                            tracing::info!("Phase B requested: every phase-A-only instance completed (rc={rc})");
                        }
                    }
                    if *DEBUG_CHALLENGES {
                        Self::debug_print_airgroup_values(&pctx_clone, &sctx_clone, &proofs_clone, id, &p);
                    }

                    if !options.aggregation {
                        continue;
                    }

                    let new_proof_type = if p == ProofType::Basic {
                        match pctx_clone.dctx_get_instance_info(id as usize) {
                            Ok((airgroup_id, air_id)) => {
                                if pctx_clone.global_info.get_air_has_compressor(airgroup_id, air_id) {
                                    ProofType::Compressor as usize
                                } else {
                                    ProofType::Recursive1 as usize
                                }
                            }
                            Err(e) => {
                                cancellation_info_clone.write_recover().cancel(Some(e));
                                return;
                            }
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

                            let arity = pctx_clone.global_info.aggregation_arity;
                            if recursive2_airgroup_proofs.len() >= arity {
                                let chunk: Vec<_> =
                                    (0..arity).map(|_| recursive2_airgroup_proofs.pop().unwrap()).collect();
                                Some(chunk)
                            } else {
                                None
                            }
                        };

                        match recursive2_proof {
                            Some(chunk) => {
                                match gen_witness_aggregation(
                                    &pctx_clone,
                                    &memory_handler_recursive_witness,
                                    &setups_clone,
                                    &chunk.iter().collect::<Vec<_>>(),
                                ) {
                                    Ok(witness) => Some(witness),
                                    Err(e) => {
                                        tracing::info!(
                                            "Error generating recursive2 witness from recursive proofs: {}",
                                            e
                                        );
                                        cancellation_info_clone.write_recover().cancel(Some(e));
                                        break;
                                    }
                                }
                            }
                            None => None,
                        }
                    } else if new_proof_type == ProofType::Recursive1 as usize && p == ProofType::Compressor {
                        let compressor_proof = compressor_proofs_clone[id as usize].write().unwrap().take().unwrap();
                        let w = gen_witness_recursive(
                            &pctx_clone,
                            &memory_handler_recursive_witness,
                            &setups_clone,
                            &compressor_proof,
                        );
                        match w {
                            Ok(witness) => Some(witness),
                            Err(e) => {
                                tracing::info!("Error generating recursive1 witness from compressor proof: {}", e);
                                cancellation_info_clone.write_recover().cancel(Some(e));
                                break;
                            }
                        }
                    } else {
                        let proof = proofs_clone[id as usize].write().unwrap().take().unwrap();
                        let w = gen_witness_recursive(
                            &pctx_clone,
                            &memory_handler_recursive_witness,
                            &setups_clone,
                            &proof,
                        );
                        match w {
                            Ok(witness) => Some(witness),
                            Err(e) => {
                                tracing::info!("Error generating recursive1 witness from basic proof: {}", e);
                                cancellation_info_clone.write_recover().cancel(Some(e));
                                break;
                            }
                        }
                    };

                    if let Some(mut witness) = witness {
                        // Arm by the `(id, kind)` the completion will report. A recursive2 proof gets
                        // its ongoing slot assigned here, before hand-off, so it is armed before it is
                        // in flight; compressor/recursive1 reuse the parent's slot from `global_idx`.
                        let child_id = if new_proof_type == ProofType::Recursive2 as usize {
                            let mut rec2 = recursive2_proofs_ongoing_clone.write().unwrap();
                            let id = rec2.len();
                            rec2.push(None);
                            witness.global_idx = Some(id);
                            id as u64
                        } else {
                            witness.global_idx.unwrap() as u64
                        };
                        // New downstream unit; commit to the callback only if the handoff
                        // succeeds, else the guard drops and settles it.
                        let child = proofs_pending_clone.arm(child_id, new_proof_type);
                        if let Some(sched) = scheduler_clone.as_ref() {
                            // Key-affinity scheduler (GPU): push into the shared queue + wake a
                            // stream worker. The unbounded push can't fail, so always commit.
                            sched.push(witness);
                            child.commit();
                        } else {
                            // CPU: hand off through the witness channels.
                            let compressor = new_proof_type == ProofType::Compressor as usize;
                            let sent = if compressor {
                                compressor_witness_tx_clone.send(witness)
                            } else if new_proof_type == ProofType::Recursive1 as usize {
                                rec1_witness_tx_clone.send(witness)
                            } else {
                                rec2_witness_tx_clone.send(witness)
                            };
                            match sent {
                                Ok(()) => child.commit(),
                                Err(crossbeam_channel::SendError(mut returned)) => {
                                    // Witness channels live on `self`, so a failed send means the
                                    // pipeline is torn down mid-run. Return the witness buffer to its
                                    // pool (else it leaks in the SendError), and surface it.
                                    drop(
                                        memory_handler_recursive_witness
                                            .adopt_trace(std::mem::take(&mut returned.trace)),
                                    );
                                    cancellation_info_clone
                                        .write_recover()
                                        .cancel(Some(ProofmanError::ProofmanError("witness channel closed".into())));
                                    break;
                                }
                            }
                        }
                    }
                }
            });
            self.handle_recursives.lock().unwrap().push(handle_recursive);
        }

        // Pipeline: the next proof is enqueued on the stream while the current one still
        // runs (no host sync at reserve; completions come off the harvest ring). Proofs phase
        // only. PROOFMAN_NO_PIPELINE=1 disables it.
        let pipeline_enabled =
            prefetch_dequeue_ahead && !std::env::var("PROOFMAN_NO_PIPELINE").map(|v| v == "1").unwrap_or(false);
        if pipeline_enabled {
            set_pipeline_mode_c(self.pctx.get_device_buffers_ptr(), true);
        }

        // Per-AIR per-proof cost proxy for LPT ordering; computed once (not in the
        // comparator). PROOFMAN_CLUSTER_SCHEDULE=0 falls back to tier-only order.
        let cluster_schedule =
            std::env::var("PROOFMAN_CLUSTER_SCHEDULE").map(|v| v != "0" && v != "false").unwrap_or(true);
        let mut group_cost: HashMap<(usize, usize), u64> = HashMap::new();
        for &id in my_instances.iter() {
            let (ag, air) = (instances[id].airgroup_id, instances[id].air_id);
            group_cost.entry((ag, air)).or_insert_with(|| match self.sctx.get_setup(ag, air) {
                // Saturating so a pathologically large AIR can't wrap to a tiny cost and defeat LPT.
                Ok(setup) => 1u64
                    .checked_shl(setup.stark_info.stark_struct.n_bits as u32)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(setup.n_cols),
                Err(_) => 0,
            });
        }

        my_instances.sort_by_key(|&id| {
            let (airgroup_id, air_id) = (instances[id].airgroup_id, instances[id].air_id);
            let is_stored = self.pctx.is_air_instance_stored(id);
            let has_compressor = self.pctx.global_info.get_air_has_compressor(airgroup_id, air_id);
            if cluster_schedule {
                schedule_key(airgroup_id, air_id, is_stored, has_compressor, group_cost[&(airgroup_id, air_id)])
            } else {
                // Legacy: tier only. Equal keys within a tier keep prior order
                // (stable sort), reproducing today's behavior for A/B comparison.
                let priority_tier: u8 = if is_stored && has_compressor {
                    0
                } else if is_stored {
                    1
                } else if has_compressor {
                    2
                } else {
                    3
                };
                (priority_tier, std::cmp::Reverse(0u64), 0usize, 0usize)
            }
        });

        let proofs_finished = Arc::new(AtomicBool::new(false));
        // Backstop: guarantees the generator threads below are told to stop on *any* early return,
        // so the `_recursives_guard` join at teardown cannot wedge (see `FinishOnDrop`).
        let _finish_generators = FinishOnDrop(proofs_finished.clone());
        for stream_id in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let sctx_clone = self.sctx.clone();
            let setups_clone = self.setups.clone();
            let aux_trace_clone = self.aux_trace.clone();
            let const_pols_clone = self.const_pols.clone();
            let const_tree_clone = self.const_tree.clone();
            let proofs_clone = self.proofs.clone();
            let compressor_proofs_clone = self.compressor_proofs.clone();
            let recursive1_proofs_clone = self.recursive1_proofs.clone();
            let recursive2_proofs_ongoing_clone = self.recursive2_proofs_ongoing.clone();
            let proofs_rx = self.proofs_rx.clone();
            let compressor_rx = self.compressor_witness_rx.clone();
            let rec2_rx = self.rec2_witness_rx.clone();
            let rec1_rx = self.rec1_witness_rx.clone();
            let n_streams_non_recursive = self.n_streams_non_recursive;
            let memory_handler_clone = self.memory_handler.clone();
            let memory_handler_recursive_witness = self.memory_handler_recursive_witness.clone();
            let proofs_finished_clone = proofs_finished.clone();
            let cancellation_info_clone = self.cancellation_info.clone();
            let proofs_pending_clone = proofs_pending.clone();
            let scheduler_clone = scheduler.clone();
            let handle_recursive = std::thread::spawn(move || {
                // Dequeued-ahead basics (id, airgroup, air), dispatch order, depth <= 2.
                // Front: launches next.
                let mut held: std::collections::VecDeque<(usize, usize, usize)> = std::collections::VecDeque::new();
                loop {
                    let force_recursive_stream = stream_id >= n_streams_non_recursive;
                    // Pipeline: collect any finished proofs (writeProof + completion
                    // callback) off the ring; non-blocking, cheap when nothing fired.
                    if pipeline_enabled {
                        harvest_pipeline_c(pctx_clone.get_device_buffers_ptr());
                    }

                    // One locked pick per iteration (GPU): a Basic is dispatched right below; a
                    // Recursive witness (in `gpu_witness`) falls through to the recursive dispatch.
                    // CPU: take a stored basic off the channel and let gen_proof select the stream.
                    let mut reserved_stream: u64 = u64::MAX;
                    // Holds the reservation from pick until the launch commits it. Every early exit
                    // below (`continue`, `break`, `return`, panic) drops it and gives the stream back,
                    // so no error path can strand a stream at status=1.
                    let mut reservation: Option<crate::StreamReservation> = None;
                    let mut gpu_witness: Option<Proof<F>> = None;
                    let basic: Option<(usize, Option<usize>)> = if let Some(sched) = scheduler_clone.as_ref() {
                        let (lock, cvar) = (&sched.lock, &sched.ready);
                        let mut guard = lock.lock().unwrap();
                        loop {
                            if !force_recursive_stream {
                                // Drain ready stored basics into the key-affinity queue.
                                while let Ok(id) = proofs_rx.try_recv() {
                                    match pctx_clone.dctx_get_instance_info(id) {
                                        Ok((ag, air)) => {
                                            guard.push_basic(id, ag, air);
                                        }
                                        Err(e) => {
                                            cancellation_info_clone.write_recover().cancel(Some(e));
                                            for (hid, _, _) in held.drain(..) {
                                                let (is_shared, buf) = pctx_clone.free_instance(hid);
                                                if is_shared {
                                                    let _ = memory_handler_clone.release_buffer(buf);
                                                }
                                                proofs_pending_clone.settle(hid as u64, ProofType::Basic as usize);
                                            }
                                            return;
                                        }
                                    }
                                }
                            }
                            let pick = if force_recursive_stream {
                                guard.next_recursive().map(|(w, s)| crate::WorkerPick::Recursive(w, s))
                            } else if let Some(p) = guard.next_compressor() {
                                // Ahead of the held basics: only this worker dispatches compressors, and
                                // a big air's compressor holds phase B closed until it has run.
                                Some(p)
                            } else if let Some(&(hid, hag, hair)) = held.front() {
                                // A held instance always launches next; only the stream reservation can
                                // make it wait.
                                let r = guard.reserve_for_basic(hag, hair).map(|s| crate::WorkerPick::Basic(hid, s));
                                if r.is_some() {
                                    held.pop_front();
                                }
                                r
                            } else {
                                guard.next_nonrecursive()
                            };
                            match pick {
                                Some(crate::WorkerPick::Recursive(w, s)) => {
                                    reserved_stream = s.stream_id() as u64;
                                    reservation = Some(s);
                                    gpu_witness = Some(w);
                                    break None;
                                }
                                Some(crate::WorkerPick::Basic(id, s)) => {
                                    let sid = s.stream_id() as usize;
                                    reservation = Some(s);
                                    break Some((id, Some(sid)));
                                }
                                None => {
                                    // Exit only once everything is drained. Non-force workers
                                    // also own the basic queue, so they must wait for it too.
                                    if proofs_finished_clone.load(Ordering::Relaxed)
                                        && guard.is_empty()
                                        && (force_recursive_stream || guard.basic_is_empty())
                                        && held.is_empty()
                                    {
                                        return;
                                    }
                                    // Pipeline: a full ring is what usually parks us here --
                                    // harvest finished proofs NOW (not at the outer loop top,
                                    // which a parked worker never revisits) so the reserve
                                    // above succeeds the moment one proof completes.
                                    if pipeline_enabled {
                                        harvest_pipeline_c(pctx_clone.get_device_buffers_ptr());
                                    }
                                    let (g, _) = cvar.wait_timeout(guard, std::time::Duration::from_millis(1)).unwrap();
                                    guard = g;
                                }
                            }
                        }
                    } else if !force_recursive_stream {
                        proofs_rx.try_recv().ok().map(|id| (id, None))
                    } else {
                        None
                    };

                    if let Some((instance_id, reserved)) = basic {
                        // Armed when queued; adopt (no re-count) so it settles on the cancel/free
                        // path or a gen_proof error, and commits only on a successful launch.
                        let pending = proofs_pending_clone.adopt(instance_id as u64, ProofType::Basic as usize);
                        if cancellation_info_clone.read_recover().token.is_cancelled() {
                            let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance(instance_id);
                            if is_shared_buffer {
                                if let Err(e) = memory_handler_clone.release_buffer(witness_buffer) {
                                    cancellation_info_clone.write_recover().cancel(Some(e));
                                    return;
                                }
                            }
                            continue;
                        } else {
                            let proof_stream_id = match Self::gen_proof(
                                &proofs_clone,
                                &pctx_clone,
                                &sctx_clone,
                                instance_id,
                                &aux_trace_clone,
                                &const_pols_clone,
                                &const_tree_clone,
                                reserved,
                                false,
                            ) {
                                Ok(sid) => {
                                    pending.commit();
                                    // Launched: the stream now carries real work, so hand the
                                    // reservation over instead of releasing it on drop.
                                    if let Some(r) = reservation.take() {
                                        r.commit();
                                    }
                                    sid
                                }
                                Err(e) => {
                                    cancellation_info_clone.write_recover().cancel(Some(e));
                                    break;
                                }
                            };
                            // Dequeue-ahead: refill the held queue (ready witnesses only -- the
                            // scheduler queue never holds an uncomputed one). Witness buffers stay
                            // alive because held instances are not freed until their own launch.
                            if prefetch_dequeue_ahead {
                                if let Some(sched) = scheduler_clone.as_ref() {
                                    let mut guard = sched.lock.lock().unwrap();
                                    // Make newly-ready instances visible to the lookahead.
                                    while let Ok(id) = proofs_rx.try_recv() {
                                        match pctx_clone.dctx_get_instance_info(id) {
                                            Ok((ag, air)) => {
                                                guard.push_basic(id, ag, air);
                                            }
                                            Err(e) => {
                                                cancellation_info_clone.write_recover().cancel(Some(e));
                                                break;
                                            }
                                        }
                                    }
                                    while held.len() < 2 {
                                        match guard.pop_basic_prefetch() {
                                            Some(h) => held.push_back(h),
                                            None => break,
                                        }
                                    }
                                }
                            }
                            let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance(instance_id);
                            if is_shared_buffer {
                                if pctx_clone.gpu {
                                    wait_trace_h2d_done_c(pctx_clone.get_device_buffers_ptr(), proof_stream_id as u64);
                                }
                                if let Err(e) = memory_handler_clone.release_buffer(witness_buffer) {
                                    cancellation_info_clone.write_recover().cancel(Some(e));
                                    return;
                                }
                            }
                            continue;
                        }
                    }

                    if cancellation_info_clone.read_recover().token.is_cancelled() {
                        // Held (dequeued-ahead) basics are no longer in the scheduler's queues, so
                        // the teardown drain can't recover them either: same recovery inline.
                        for (hid, _, _) in held.drain(..) {
                            let (is_shared, buf) = pctx_clone.free_instance(hid);
                            if is_shared {
                                let _ = memory_handler_clone.release_buffer(buf);
                            }
                            proofs_pending_clone.settle(hid as u64, ProofType::Basic as usize);
                        }
                        // The pick above may already have handed us a witness. Dropping it here would
                        // lose its pooled `trace` for the rest of the process — the teardown
                        // drain can't recover it, since it is no longer in the scheduler's queues.
                        // Same recovery as that drain: pool the buffer, settle the unit armed at hand-off.
                        if let Some(w) = gpu_witness.take() {
                            crate::recover_drained_witnesses(
                                vec![w],
                                &memory_handler_recursive_witness,
                                &proofs_pending_clone,
                            );
                        }
                        break;
                    }

                    // The GPU pick above already reserved the stream (`reserved_stream`) and
                    // produced the recursive witness; on CPU gen_recursive_proof selects internally.
                    let mut witness = if let Some(w) = gpu_witness {
                        w
                    } else {
                        // CPU: `select!` among the ready witness channels.
                        let witness_opt = if force_recursive_stream {
                            crossbeam_channel::select! {
                                recv(rec2_rx) -> msg => match msg { Ok(w) => Some(w), Err(_) => return },
                                recv(rec1_rx) -> msg => match msg { Ok(w) => Some(w), Err(_) => return },
                                default(std::time::Duration::from_millis(1)) => None,
                            }
                        } else {
                            crossbeam_channel::select! {
                                recv(rec2_rx) -> msg => match msg { Ok(w) => Some(w), Err(_) => return },
                                recv(compressor_rx) -> msg => match msg { Ok(w) => Some(w), Err(_) => return },
                                recv(rec1_rx) -> msg => match msg { Ok(w) => Some(w), Err(_) => return },
                                default(std::time::Duration::from_millis(1)) => None,
                            }
                        };
                        match witness_opt {
                            Some(w) => w,
                            None => {
                                if proofs_finished_clone.load(Ordering::Relaxed) {
                                    return;
                                }
                                continue;
                            }
                        }
                    };

                    // Adopt the downstream unit armed at hand-off (no re-count): settles on an error
                    // `break` below, commits once launched. Its id was assigned before hand-off, so
                    // `global_idx` is set for every proof type here.
                    let pending =
                        proofs_pending_clone.adopt(witness.global_idx.unwrap() as u64, witness.proof_type.as_usize());

                    let force_recursive_stream = stream_id >= n_streams_non_recursive;

                    let new_proof = match gen_recursive_proof_size(&pctx_clone, &setups_clone, &witness) {
                        Ok(p) => p,
                        Err(e) => {
                            // generate_recursive_proof (which normally returns the witness buffer to its
                            // pool) is not reached on this error path, so return it here — adopt-then-drop.
                            drop(memory_handler_recursive_witness.adopt_trace(std::mem::take(&mut witness.trace)));
                            cancellation_info_clone.write_recover().cancel(Some(e));
                            break;
                        }
                    };
                    let new_proof_type_str: &str = new_proof.proof_type.into();

                    let new_proof_type = new_proof.proof_type;

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

                        if let Err(e) = generate_recursive_proof(
                            &pctx_clone,
                            &memory_handler_recursive_witness,
                            &setups_clone,
                            &mut witness,
                            new_proof_ref,
                            &aux_trace_clone,
                            &const_tree_clone,
                            &const_pols_clone,
                            force_recursive_stream,
                            reserved_stream,
                            None,
                        ) {
                            cancellation_info_clone.write_recover().cancel(Some(e));
                            break;
                        }
                    } else if new_proof_type == ProofType::Compressor {
                        let compressor_lock = compressor_proofs_clone[id].read().unwrap();
                        let new_proof_ref = compressor_lock.as_ref().unwrap();
                        if let Err(e) = generate_recursive_proof(
                            &pctx_clone,
                            &memory_handler_recursive_witness,
                            &setups_clone,
                            &mut witness,
                            new_proof_ref,
                            &aux_trace_clone,
                            &const_tree_clone,
                            &const_pols_clone,
                            force_recursive_stream,
                            reserved_stream,
                            None,
                        ) {
                            cancellation_info_clone.write_recover().cancel(Some(e));
                            break;
                        }
                    } else {
                        let recursive1_lock = recursive1_proofs_clone[id].read().unwrap();
                        let new_proof_ref = recursive1_lock.as_ref().unwrap();
                        if let Err(e) = generate_recursive_proof(
                            &pctx_clone,
                            &memory_handler_recursive_witness,
                            &setups_clone,
                            &mut witness,
                            new_proof_ref,
                            &aux_trace_clone,
                            &const_tree_clone,
                            &const_pols_clone,
                            force_recursive_stream,
                            reserved_stream,
                            None,
                        ) {
                            cancellation_info_clone.write_recover().cancel(Some(e));
                            break;
                        }
                    }

                    pending.commit();
                    // Launched: the stream now carries real work, so hand the reservation over
                    // instead of releasing it on drop.
                    if let Some(r) = reservation.take() {
                        r.commit();
                    }

                    if !pctx_clone.gpu {
                        launch_callback_c(id as u64, new_proof_type_str);
                    }
                }
            });
            self.handle_recursives.lock().unwrap().push(handle_recursive);
        }

        let mut instances_to_be_calculated = Vec::with_capacity(my_instances.len());
        for &instance_id in my_instances.iter() {
            // Committed to the async callback; if the send panics the guard settles it. The basic
            // proof's completion reports its instance id.
            let pending = proofs_pending.arm(instance_id as u64, ProofType::Basic as usize);
            if self.pctx.is_air_instance_stored(instance_id) {
                self.proofs_tx.send(instance_id).unwrap();
            } else {
                instances_to_be_calculated.push(instance_id);
            }
            pending.commit();
        }

        let witness_done = Arc::new(Counter::new());

        let (witness_handler, witness_handles) =
            self.calc_witness_handler(witness_done.clone(), self.memory_handler.clone(), true, None, false);

        let _witness_guard = WitnessGuard {
            witness_tx: self.witness_tx.clone(),
            handler: witness_handler.clone(),
            handles: witness_handles.clone(),
        };

        timer_start_debug!(CALCULATING_WITNESS);
        self.calculate_witness(
            &instances_to_be_calculated,
            self.memory_handler.clone(),
            witness_done.clone(),
            true,
            false,
        )?;
        timer_stop_and_log_debug!(CALCULATING_WITNESS);
        self.witness_tx.send(usize::MAX).ok();
        if let Some(h) = witness_handler.lock().unwrap().take() {
            h.join().unwrap();
        }
        if self.pctx.gpu {
            let handles_to_join = witness_handles.lock().unwrap().drain(..).collect::<Vec<_>>();
            for handle in handles_to_join {
                handle.join().unwrap();
            }
        }

        drop(witness_handles);

        // Wait for every launched proof to settle. The 600s backstop returns false without
        // cancelling, so cancel then — an incomplete result must not be mistaken for success.
        // PROOFMAN_SETTLE_TIMEOUT_S shortens the backstop for wedge diagnosis (default 600).
        let settle_timeout_s: u64 =
            std::env::var("PROOFMAN_SETTLE_TIMEOUT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
        let settled = completions.wait_settled(
            || get_stream_proofs_non_blocking_c(self.pctx.get_device_buffers_ptr()),
            &self.cancellation_info,
            Some(std::time::Duration::from_secs(settle_timeout_s)),
        );
        if !settled && !self.cancellation_info.read_recover().token.is_cancelled() {
            // The ledger already named the unsettled units; add what the device streams hold.
            dump_pipeline_state_c(self.pctx.get_device_buffers_ptr());
            self.cancellation_info
                .write_recover()
                .cancel(Some(ProofmanError::ProofmanError("timed out waiting for proofs to settle".into())));
        }

        // Signal the generators to stop, then release the capability: its `Drop` runs the final
        // harvest and clears the registration, disconnecting the consumers. A late completion is an
        // idempotent no-op.
        proofs_finished.store(true, Ordering::Relaxed);
        drop(completions);
        if pipeline_enabled {
            set_pipeline_mode_c(self.pctx.get_device_buffers_ptr(), false);
        }

        if self.cancellation_info.read_recover().token.is_cancelled() {
            self.cancel_memory_handlers();
        }

        let handles = self.handle_recursives.lock().unwrap().drain(..).collect::<Vec<_>>();
        let mut panicked = 0usize;
        for handle in handles {
            if handle.join().is_err() {
                panicked += 1;
            }
        }
        // A panic settles the worker's ProofToken on unwind, so `wait_settled` above already
        // returned true for a proof that was never produced. Cancel so `check_cancel` reports it.
        if panicked > 0 {
            self.cancellation_info.write_recover().cancel(Some(ProofmanError::ProofmanError(format!(
                "{panicked} recursive proof worker thread(s) panicked"
            ))));
        }

        self.check_cancel(true)?;

        timer_stop_and_log_info!(GENERATING_INNER_PROOFS);
        // Anchors the per-span waits: they must sum to these totals.
        self.memory_handler.log_wait_summary();
        self.memory_handler_recursive_witness.log_wait_summary();

        let mut proof_id = None;
        let mut vadcop_final_proof = None;
        if options.aggregation {
            let mut agg_proofs = Vec::new();

            if !options.rma {
                timer_start_debug!(WAITING_FOR_COMPRESSED_PROOFS);
                self.mpi_ctx.barrier();
                timer_stop_and_log_debug!(WAITING_FOR_COMPRESSED_PROOFS);
                timer_start_debug!(GENERATING_WORKER_COMPRESSED_PROOFS);
                let recursive2_proofs_data: Vec<Vec<Proof<F>>> = self
                    .recursive2_proofs
                    .iter()
                    .map(|lock| {
                        let mut write_lock = lock.write().unwrap();
                        let mut proofs = vec![];
                        while let Some(proof) = write_lock.pop() {
                            proofs.push(proof);
                        }
                        proofs
                    })
                    .collect();

                aggregate_worker_proofs(
                    &self.pctx,
                    &self.memory_handler_recursive_witness,
                    &self.mpi_ctx,
                    &self.setups,
                    recursive2_proofs_data,
                    &self.aux_trace,
                    &self.const_pols,
                    &self.const_tree,
                    &mut agg_proofs,
                )?;

                self.check_cancel(true)?;

                timer_stop_and_log_debug!(GENERATING_WORKER_COMPRESSED_PROOFS);
            } else {
                timer_start_debug!(GET_OUTER_RANK);
                self.mpi_ctx.process_ready_for_outer_agg();
                timer_stop_and_log_debug!(GET_OUTER_RANK);
                let outer_rank = self.mpi_ctx.get_outer_agg_rank()? as usize;
                if self.pctx.mpi_ctx.rank as usize == outer_rank {
                    self.worker_aggregations_rma(outer_rank != 0)?;
                } else {
                    for airgroup in 0..self.pctx.global_info.air_groups.len() {
                        let mut write_lock = self.recursive2_proofs[airgroup].write().unwrap();

                        while let Some(proof) = write_lock.pop() {
                            self.pctx.mpi_ctx.send_proof_agg_rank(&proof);
                        }
                    }
                };
            }

            if self.mpi_ctx.rank == 0 {
                let worker_index = self.pctx.get_worker_index()?;
                if options.rma {
                    let outer_rank = self.mpi_ctx.get_outer_agg_rank()?;
                    if outer_rank != 0 {
                        let mut airgroups_with_instances = vec![false; n_airgroups];
                        for global_id in self.pctx.dctx_get_worker_instances().iter() {
                            airgroups_with_instances[instances[*global_id].airgroup_id] = true;
                        }
                        for (airgroup, has_instance) in airgroups_with_instances.iter().enumerate() {
                            if *has_instance {
                                let proof = self.pctx.mpi_ctx.recv_proof_from_rank(airgroup, outer_rank);
                                agg_proofs.push(AggProofs::new(airgroup as u64, proof.clone(), vec![worker_index]));
                            }
                        }
                    } else {
                        for airgroup in 0..n_airgroups {
                            let mut write_lock = self.recursive2_proofs[airgroup].write().unwrap();
                            if let Some(proof) = write_lock.pop() {
                                agg_proofs.push(AggProofs::new(airgroup as u64, proof.proof, vec![worker_index]));
                            }
                        }
                    }
                }

                for proof in &agg_proofs {
                    let agg_proof =
                        Proof::new(ProofType::Recursive2, proof.airgroup_id as usize, 0, None, proof.proof.clone());

                    let proof_acc_challenge = get_accumulated_challenge(&self.pctx, &proof.proof);
                    let mut worker_contributions = self.worker_contributions.write().unwrap();
                    if let Some(contrib) = worker_contributions.iter_mut().find(|contrib| {
                        contrib.worker_index == worker_index as u32 && contrib.airgroup_id == proof.airgroup_id as usize
                    }) {
                        if contrib.aggregated {
                            self.cancellation_info.write_recover().cancel(Some(ProofmanError::InvalidProof(
                                "Proof contribution was already aggregated".into(),
                            )));
                        }
                        contrib.aggregated = true;
                        for (c, value) in contrib.challenge.iter().enumerate() {
                            if *value != proof_acc_challenge[c] {
                                self.cancellation_info.write_recover().cancel(Some(ProofmanError::InvalidProof(
                                    "Proof contribution challenge does not match expected accumulated challenge previously committed".into(),
                                )));
                                break;
                            }
                        }
                    } else {
                        self.cancellation_info.write_recover().cancel(Some(ProofmanError::ProofmanError(format!(
                            "Missing contribution from worker {} and airgroup id {}",
                            worker_index, proof.airgroup_id
                        ))));
                        break;
                    }

                    self.recursive2_proofs[proof.airgroup_id as usize].write().unwrap().push(agg_proof);
                    self.received_agg_proofs.write().unwrap()[proof.airgroup_id as usize].push(worker_index);
                }
                if phase == ProvePhase::Internal {
                    timer_stop_and_log_info!(GENERATING_PROOFS);
                    return Ok(ProvePhaseResult::Internal(agg_proofs));
                }
            }

            if self.mpi_ctx.rank == 0 {
                let vadcop_final = self.receive_aggregated_proofs_inner(vec![], true, true, &options)?;

                let proof = vadcop_final.unwrap().into_iter().next().unwrap().proof;

                vadcop_final_proof = Some(
                    VadcopFinalProof::new_from_proof(&proof, options.compressed, self.pctx.global_info.hash.clone())
                        .map_err(|e| {
                            ProofmanError::InvalidConfiguration(format!("Failed to create VadcopFinalProof: {}", e))
                        })?,
                );

                proof_id = Some(
                    blake3::hash(unsafe { std::slice::from_raw_parts(proof.as_ptr() as *const u8, proof.len() * 8) })
                        .to_hex()
                        .to_string(),
                );
            }
        }

        if options.verify_proofs {
            if options.aggregation {
                if self.mpi_ctx.rank == 0 {
                    timer_start_info!(VERIFYING_VADCOP_FINAL_PROOF);

                    // Verified from the proving key, not from a committed Rust
                    // verifier: the aggregator binds the application's publics
                    // into q_verify, so a verifier generated for one application
                    // rejects proofs another's correct prover produced. Same
                    // mechanism verify_agg_proof already uses for recursive1/2.
                    let setup = match options.compressed {
                        true => self.setups.setup_vadcop_final_compressed.as_ref().unwrap(),
                        false => self.setups.setup_vadcop_final.as_ref().unwrap(),
                    };
                    let base = setup.setup_path.display().to_string();

                    let proof = vadcop_final_proof.as_ref().unwrap();
                    let publics: Vec<F> = proof.public_values.iter().map(|&x| F::from_u64(x)).collect();
                    let valid_proofs = verify_proof::<F>(
                        &proof.proof,
                        base.clone() + ".starkinfo.json",
                        base.clone() + ".verifier.bin",
                        base + ".verkey.json",
                        Some(publics),
                        None,
                        None,
                    );
                    timer_stop_and_log_info!(VERIFYING_VADCOP_FINAL_PROOF);
                    if !valid_proofs {
                        tracing::info!("··· {}", "\u{2717} Vadcop Final proof was not verified".bright_red().bold());
                        return Err(ProofmanError::InvalidProof("Vadcop Final proof was not verified".into()));
                    } else {
                        tracing::info!("··· {}", "\u{2713} Vadcop Final proof was verified".bright_green().bold());
                    }
                }
            } else {
                return self.verify_proofs();
            }
        }

        if phase == ProvePhase::Full {
            Ok(ProvePhaseResult::Full(proof_id, vadcop_final_proof))
        } else {
            Ok(ProvePhaseResult::Internal(Vec::new()))
        }
    }

    pub fn register_aggregated_proofs(&self, agg_proofs: Vec<AggProofsRegister>) -> ProofmanResult<()> {
        let mut received = self.received_agg_proofs.write().unwrap();

        for proof in agg_proofs {
            let airgroup_vec = &mut received[proof.airgroup_id as usize];

            for &worker_idx in &proof.worker_indexes {
                if airgroup_vec.contains(&worker_idx) {
                    let error_message = ProofmanError::InvalidProof(format!(
                        "Received duplicated proof from worker {} for airgroup {}",
                        worker_idx, proof.airgroup_id
                    ));
                    self.cancellation_info.write_recover().cancel(Some(error_message));
                    break;
                }
                airgroup_vec.push(worker_idx);
            }
        }

        self.check_cancel(false)?;

        Ok(())
    }

    pub fn receive_aggregated_proofs(
        &self,
        agg_proofs: Vec<AggProofs>,
        last_proof: bool,
        final_proof: bool,
        options: &ProofOptions,
    ) -> ProofmanResult<Option<Vec<AggProofs>>> {
        let _computing = self.acquire_computing("receive_aggregated_proofs");
        if options.compressed {
            self.ensure_compressed_final_supported()?;
        }
        self.receive_aggregated_proofs_inner(agg_proofs, last_proof, final_proof, options)
    }

    fn receive_aggregated_proofs_inner(
        &self,
        agg_proofs: Vec<AggProofs>,
        last_proof: bool,
        final_proof: bool,
        options: &ProofOptions,
    ) -> ProofmanResult<Option<Vec<AggProofs>>> {
        if !agg_proofs.is_empty() {
            tracing::info!("Received {:?} aggregated proofs", agg_proofs);
        }

        if !agg_proofs.is_empty() {
            self.ensure_outer_aggregations_started();
        }

        for proof in agg_proofs {
            {
                let received = self.received_agg_proofs.read().unwrap();
                let airgroup_vec = &received[proof.airgroup_id as usize];

                for &worker_idx in &proof.worker_indexes {
                    if !airgroup_vec.contains(&worker_idx) {
                        self.cancellation_info.write_recover().cancel(Some(ProofmanError::InvalidProof(format!(
                            "Received proof from worker {} for airgroup {} was not registered",
                            worker_idx, proof.airgroup_id
                        ))));
                        break;
                    }
                }
            }

            if self.cancellation_info.read_recover().token.is_cancelled() {
                break;
            }

            {
                let setup = self.setups.sctx_recursive2.as_ref().unwrap().get_setup(proof.airgroup_id as usize, 0)?;
                let publics_aggregation = n_publics_aggregation(&self.pctx, proof.airgroup_id as usize);
                let expected = setup.proof_size as usize + publics_aggregation;
                if proof.proof.len() != expected {
                    self.cancellation_info.write_recover().cancel(Some(ProofmanError::InvalidProof(format!(
                        "Aggregated proof from workers {:?} airgroup {} has wrong length {} (expected {}) — malformed or truncated in transit",
                        proof.worker_indexes, proof.airgroup_id, proof.proof.len(), expected
                    ))));
                    break;
                }
            }
            let proof_acc_challenge = get_accumulated_challenge(&self.pctx, &proof.proof);
            let mut stored_contributions = Vec::new();
            for w in &proof.worker_indexes {
                let mut worker_contributions = self.worker_contributions.write().unwrap();
                if let Some(contrib) = worker_contributions.iter_mut().find(|contrib| {
                    contrib.worker_index == *w as u32 && contrib.airgroup_id == proof.airgroup_id as usize
                }) {
                    if contrib.aggregated {
                        self.cancellation_info.write_recover().cancel(Some(ProofmanError::InvalidProof(
                            "Proof contribution was already aggregated".into(),
                        )));
                        break;
                    }
                    contrib.aggregated = true;
                    stored_contributions.push(contrib.challenge.iter().map(|&x| F::from_u64(x)).collect());
                } else {
                    self.cancellation_info.write_recover().cancel(Some(ProofmanError::ProofmanError(format!(
                        "Missing contribution from worker {} and airgroup id {}",
                        w, proof.airgroup_id
                    ))));
                    break;
                }
            }

            timer_start_debug!(VERIFYING_OUTER_AGGREGATED_PROOF);
            let valid_recursive_proof = self.verify_agg_proof(proof.airgroup_id as usize, &proof.proof)?;

            if !valid_recursive_proof {
                self.cancellation_info
                    .write_recover()
                    .cancel(Some(ProofmanError::InvalidProof("Received aggregated proof is invalid!".into())));
                break;
            }
            timer_stop_and_log_debug!(VERIFYING_OUTER_AGGREGATED_PROOF);

            let workers_acc_challenge = aggregate_contributions(&self.pctx, &stored_contributions);
            for (c, value) in workers_acc_challenge.iter().enumerate() {
                if value.as_canonical_u64() != proof_acc_challenge[c] {
                    self.cancellation_info.write_recover().cancel(Some(ProofmanError::InvalidProof(
                        "Aggregated proof challenge does not match the expected challenge".into(),
                    )));
                    break;
                }
            }
            let id = {
                let mut rec2_proofs = self.recursive2_proofs_ongoing.write().unwrap();
                let id = rec2_proofs.len();
                let agg_proof = Proof::new(ProofType::Recursive2, proof.airgroup_id as usize, 0, Some(id), proof.proof);
                rec2_proofs.push(Some(agg_proof));
                id
            };

            launch_callback_c(id as u64, ProofType::Recursive2.into());
        }

        if last_proof || self.cancellation_info.read_recover().token.is_cancelled() {
            let mut total_proofs_to_be_done = 0;
            let mut total_proofs_received = vec![0; self.received_agg_proofs.read().unwrap().len()];
            if !self.cancellation_info.read_recover().token.is_cancelled() {
                for (airgroup_id, worker_indexes) in self.received_agg_proofs.read().unwrap().iter().enumerate() {
                    let n_agg_proofs = worker_indexes.len();
                    if n_agg_proofs == 1 && worker_indexes[0] == self.pctx.get_worker_index()? {
                        continue;
                    }
                    total_proofs_received[airgroup_id] = n_agg_proofs;
                    let n_agg_proofs_to_be_done =
                        total_recursive_proofs(n_agg_proofs, self.pctx.global_info.aggregation_arity);
                    if n_agg_proofs_to_be_done.has_remaining {
                        let setup = self.setups.get_setup(airgroup_id, 0, &ProofType::Recursive2)?;
                        let publics_aggregation = n_publics_aggregation(&self.pctx, airgroup_id);
                        let null_proof_buffer = vec![0; setup.proof_size as usize + publics_aggregation];

                        let id = {
                            let mut rec2_proofs = self.recursive2_proofs_ongoing.write().unwrap();
                            let id = rec2_proofs.len();
                            let null_proof =
                                Proof::new(ProofType::Recursive2, airgroup_id, 0, Some(id), null_proof_buffer);
                            rec2_proofs.push(Some(null_proof));
                            id
                        };

                        launch_callback_c(id as u64, ProofType::Recursive2.into());
                    }
                    total_proofs_to_be_done += n_agg_proofs_to_be_done.n_proofs;
                }
            }

            if total_proofs_to_be_done > 0 {
                tracing::info!("Last proof received. {:?} proofs were received and waiting for {} aggregated proofs to be generated...", total_proofs_received, total_proofs_to_be_done);
            }

            self.total_outer_agg_proofs.wait_until_value_and_check_streams(
                total_proofs_to_be_done,
                || get_stream_proofs_non_blocking_c(self.pctx.get_device_buffers_ptr()),
                &self.cancellation_info,
            );
            if self.cancellation_info.read_recover().token.is_cancelled() {
                self.cancel_memory_handlers();
            }
            get_stream_proofs_c(self.pctx.get_device_buffers_ptr());
            self.stop_outer_aggregations();

            self.check_cancel(false)?;

            let agg_proofs_data: Vec<AggProofs> = (0..self.pctx.global_info.air_groups.len())
                .map(|airgroup_id| {
                    let mut lock = self.recursive2_proofs[airgroup_id].write().unwrap();
                    let proof = std::mem::take(
                        &mut lock
                            .first_mut()
                            .ok_or_else(|| {
                                ProofmanError::InvalidProof(format!(
                                    "Expected at least one proof for airgroup {}",
                                    airgroup_id
                                ))
                            })?
                            .proof,
                    );
                    Ok(AggProofs::new(airgroup_id as u64, proof, vec![]))
                })
                .collect::<ProofmanResult<Vec<_>>>()?;

            if !final_proof {
                return Ok(Some(agg_proofs_data));
            } else {
                let worker_contributions = self.worker_contributions.read().unwrap();
                let received_agg_proofs = self.received_agg_proofs.read().unwrap();
                let current_worker_index = self.pctx.get_worker_index().unwrap_or(0) as u32;

                let requires_aggregation = received_agg_proofs.iter().any(|worker_indexes| {
                    worker_indexes.len() > 1
                        || (worker_indexes.len() == 1 && worker_indexes[0] != current_worker_index as usize)
                });

                if requires_aggregation {
                    let mut not_received_contributions = Vec::new();
                    for contrib in worker_contributions.iter() {
                        if !contrib.aggregated {
                            not_received_contributions.push((contrib.worker_index, contrib.airgroup_id));
                        }
                    }

                    if !not_received_contributions.is_empty() {
                        let error = format!(
                            "Not received contributions from workers: {:?}",
                            not_received_contributions
                                .iter()
                                .map(|(worker_index, airgroup_id)| format!(
                                    "(worker {}, airgroup {})",
                                    worker_index, airgroup_id
                                ))
                                .collect::<Vec<_>>()
                        );

                        self.cancellation_info.write_recover().cancel(Some(ProofmanError::InvalidProof(error.clone())));
                        return Err(ProofmanError::InvalidProof(error));
                    }
                }

                let global_challenge = self.pctx.get_global_challenge().clone();
                let accumulated_challenge = get_accumulated_challenge(&self.pctx, &agg_proofs_data[0].proof);
                let global_challenge_calculated = calculate_global_challenge(
                    &self.pctx,
                    &[ContributionsInfo {
                        challenge: accumulated_challenge.clone(),
                        airgroup_id: 0,
                        worker_index: 0,
                        aggregated: true,
                    }],
                );

                if global_challenge_calculated != *global_challenge {
                    let error =
                        "Global challenge calculated from contributions does not match the global challenge from pctx"
                            .to_string();

                    self.cancellation_info.write_recover().cancel(Some(ProofmanError::InvalidProof(error.clone())));
                    return Err(ProofmanError::InvalidProof(error));
                }

                let vadcop_proof_final = generate_vadcop_final_proof(
                    &self.pctx,
                    &self.memory_handler_recursive_witness,
                    &self.setups,
                    &agg_proofs_data,
                    &self.aux_trace,
                    &self.const_pols,
                    &self.const_tree,
                )?;

                if options.compressed {
                    let vadcop_final_proof_compressed = generate_vadcop_final_compressed_proof(
                        &self.pctx,
                        &self.memory_handler_recursive_witness,
                        &self.setups,
                        &vadcop_proof_final.proof,
                        &self.aux_trace,
                        &self.const_pols,
                        &self.const_tree,
                    )?;

                    return Ok(Some(vec![AggProofs::new(0, vadcop_final_proof_compressed.proof, vec![])]));
                } else {
                    return Ok(Some(vec![AggProofs::new(0, vadcop_proof_final.proof, vec![])]));
                }
            }
        }

        Ok(None)
    }

    fn outer_aggregations(&self) -> CompletionOwner {
        self.outer_agg_proofs_finished.store(false, Ordering::SeqCst);
        let completions = self.completions.acquire(DeviceBuffersPtr(self.pctx.get_device_buffers_ptr()));

        if self.pctx.gpu {
            let pctx_clone = self.pctx.clone();
            let outer_agg_proofs_finished = self.outer_agg_proofs_finished.clone();
            let cancellation_info_clone = self.cancellation_info.clone();
            let handle_pump = std::thread::spawn(move || loop {
                if outer_agg_proofs_finished.load(Ordering::Relaxed)
                    || cancellation_info_clone.read_recover().token.is_cancelled()
                {
                    break;
                }
                get_stream_proofs_non_blocking_c(pctx_clone.get_device_buffers_ptr());
                std::thread::sleep(std::time::Duration::from_millis(1));
            });
            self.handle_recursives.lock().unwrap().push(handle_pump);
        }

        for _ in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let memory_handler_recursive_witness = self.memory_handler_recursive_witness.clone();
            let setups_clone = self.setups.clone();
            let recursive2_proofs_clone = self.recursive2_proofs.clone();
            let recursive2_proofs_ongoing_clone = self.recursive2_proofs_ongoing.clone();
            let rec2_witness_tx_clone = self.rec2_witness_tx.clone();
            let recursive_rx_clone = completions.receiver();
            let cancellation_info_clone = self.cancellation_info.clone();
            let handle_recursive = std::thread::spawn(move || {
                // Exits when the owner is dropped and the channel disconnects (no sentinel).
                while let Ok(msg) = recursive_rx_clone.recv() {
                    let id = msg.id;

                    if cancellation_info_clone.read_recover().token.is_cancelled() {
                        break;
                    }

                    let proof = recursive2_proofs_ongoing_clone.write().unwrap()[id as usize].take().unwrap();

                    let mut recursive2_airgroup_proofs = recursive2_proofs_clone[proof.airgroup_id].write().unwrap();
                    recursive2_airgroup_proofs.push(proof);

                    let arity = pctx_clone.global_info.aggregation_arity;
                    if recursive2_airgroup_proofs.len() >= arity {
                        let chunk: Vec<_> = (0..arity).map(|_| recursive2_airgroup_proofs.pop().unwrap()).collect();

                        let w = gen_witness_aggregation(
                            &pctx_clone,
                            &memory_handler_recursive_witness,
                            &setups_clone,
                            &chunk.iter().collect::<Vec<_>>(),
                        );

                        let witness = match w {
                            Ok(witness) => witness,
                            Err(e) => {
                                tracing::info!("Error generating recursive2 witness from recursive proofs: {}", e);
                                cancellation_info_clone.write_recover().cancel(Some(e));
                                break;
                            }
                        };
                        rec2_witness_tx_clone.send(witness).unwrap();
                    }
                }
            });
            self.handle_recursives.lock().unwrap().push(handle_recursive);
        }

        for _ in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let setups_clone = self.setups.clone();
            let const_pols_clone = self.const_pols.clone();
            let const_tree_clone = self.const_tree.clone();
            let aux_trace_clone = self.aux_trace.clone();
            let recursive2_proofs_ongoing_clone = self.recursive2_proofs_ongoing.clone();
            let outer_agg_proofs_finished = self.outer_agg_proofs_finished.clone();
            let rec2_witness_rx = self.rec2_witness_rx.clone();
            let cancellation_info_clone = self.cancellation_info.clone();
            let total_outer_agg_proofs = self.total_outer_agg_proofs.clone();
            let memory_handler_recursive_witness = self.memory_handler_recursive_witness.clone();
            let handle = std::thread::spawn(move || loop {
                if cancellation_info_clone.read_recover().token.is_cancelled() {
                    break;
                }
                let witness = rec2_witness_rx.recv_timeout(std::time::Duration::from_millis(1));
                let mut witness = match witness {
                    Ok(w) => w,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        if outer_agg_proofs_finished.load(Ordering::Relaxed) {
                            return;
                        }
                        continue;
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                };

                let id = {
                    let mut rec2_proofs = recursive2_proofs_ongoing_clone.write().unwrap();
                    let id = rec2_proofs.len();
                    rec2_proofs.push(None);
                    id
                };

                witness.global_idx = Some(id);

                let new_proof = match gen_recursive_proof_size(&pctx_clone, &setups_clone, &witness) {
                    Ok(p) => p,
                    Err(e) => {
                        // generate_recursive_proof (which returns the buffer to its pool) isn't reached
                        // here; return it (adopt-then-drop) or the pool comes back short and wedges the next job.
                        drop(memory_handler_recursive_witness.adopt_trace(std::mem::take(&mut witness.trace)));
                        cancellation_info_clone.write_recover().cancel(Some(e));
                        break;
                    }
                };

                recursive2_proofs_ongoing_clone.write().unwrap()[id] = Some(new_proof);

                let recursive2_lock = recursive2_proofs_ongoing_clone.read().unwrap();
                let new_proof_ref = recursive2_lock[id].as_ref().unwrap();

                if let Err(e) = generate_recursive_proof(
                    &pctx_clone,
                    &memory_handler_recursive_witness,
                    &setups_clone,
                    &mut witness,
                    new_proof_ref,
                    &aux_trace_clone,
                    &const_tree_clone,
                    &const_pols_clone,
                    false,
                    u64::MAX, // one-off launch: reserve stream internally
                    None,
                ) {
                    cancellation_info_clone.write_recover().cancel(Some(e));
                    break;
                }

                if !pctx_clone.gpu {
                    launch_callback_c(id as u64, ProofType::Recursive2.into());
                }
                total_outer_agg_proofs.increment();
            });
            self.handle_recursives.lock().unwrap().push(handle);
        }

        // The caller stores this owner for the service's lifetime; dropping it (in
        // `stop_outer_aggregations`) clears the registration and disconnects the consumers above.
        completions
    }

    fn verify_proofs(&self) -> ProofmanResult<ProvePhaseResult> {
        timer_start_info!(VERIFYING_PROOFS);
        let mut valid_proofs = true;

        let my_instances = self.pctx.dctx_get_process_instances();

        let mut airgroup_values_air_instances = vec![Vec::new(); my_instances.len()];
        for instance_id in my_instances.iter() {
            let proof = {
                let mut lock = self.proofs[*instance_id].write().unwrap();
                std::mem::take(&mut *lock)
            };
            let valid_proof = verify_basic_proof(&self.pctx, *instance_id, &proof.as_ref().unwrap().proof)?;
            if !valid_proof {
                valid_proofs = false;
            }

            let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(*instance_id)?;
            let setup = self.sctx.get_setup(airgroup_id, air_id)?;
            let n_airgroup_values = setup
                .stark_info
                .airgroupvalues_map
                .as_ref()
                .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
                .unwrap_or(0);

            let airgroup_values: Vec<F> = proof
                .as_ref()
                .map(|p| p.proof[0..n_airgroup_values].iter().map(|&x| F::from_u64(x)).collect())
                .unwrap();

            airgroup_values_air_instances[self.pctx.dctx_get_instance_local_idx(*instance_id)?] = airgroup_values;
        }
        timer_stop_and_log_info!(VERIFYING_PROOFS);

        let airgroupvalues_u64 = aggregate_airgroupvals(&self.pctx, &airgroup_values_air_instances)?;
        let airgroupvalues = self.mpi_ctx.distribute_airgroupvalues(airgroupvalues_u64, &self.pctx.global_info);

        if self.mpi_ctx.rank == 0 {
            let valid_global_constraints =
                verify_global_constraints_proof(&self.pctx, &self.sctx, &DebugInfo::default(), airgroupvalues);
            if valid_global_constraints.is_err() {
                valid_proofs = false;
            }
        }

        if valid_proofs {
            tracing::info!("··· {}", "\u{2713} All proofs were successfully verified".bright_green().bold());
            Ok(ProvePhaseResult::Internal(Vec::new()))
        } else {
            Err(ProofmanError::InvalidProof("Basic proofs were not verified".into()))
        }
    }

    fn exec(&self) -> ProofmanResult<String> {
        timer_start_info!(EXECUTE);

        if !self.wcm.is_init_witness() {
            return Err(ProofmanError::ProofmanError("Witness computation dynamic library not initialized".into()));
        }

        if let Err(e) = self.wcm.execute() {
            self.cancellation_info.write_recover().cancel(Some(e));
        }

        if self.pctx.gpu && self.pctx.reload_fixed_pols_gpu.load(Ordering::SeqCst) {
            timer_start_info!(RELOAD_FIXED_POLS);
            let _ = load_device_const_pols(&self.pctx, &self.sctx, &self.setups, self.options.aggregation, true)?;
            self.pctx.reload_fixed_pols_gpu.store(false, Ordering::SeqCst);
            timer_stop_and_log_info!(RELOAD_FIXED_POLS);
        }

        self.check_cancel(true)?;

        let global_summary = print_summary_info(
            &self.pctx,
            &self.sctx,
            &self.mpi_ctx,
            &self.options.packed_info,
            self.options.verbose_mode,
        )?;

        timer_stop_and_log_info!(EXECUTE);
        Ok(global_summary)
    }

    #[allow(clippy::type_complexity)]
    fn worker_aggregations_rma(&self, send_proofs: bool) -> ProofmanResult<()> {
        timer_start_debug!(GENERATING_WORKER_RMA_COMPRESSED_PROOFS);

        let my_rank = self.mpi_ctx.rank as usize;
        let n_processes = self.mpi_ctx.n_processes as usize;

        let (rec2_witness_tx, rec2_witness_rx): (Sender<Proof<F>>, Receiver<Proof<F>>) = unbounded();

        let completions = self.completions.acquire(DeviceBuffersPtr(self.pctx.get_device_buffers_ptr()));

        let instances = self.pctx.dctx_get_instances();
        let n_airgroups = self.pctx.global_info.air_groups.len();
        let mut airgroup_instances_alive = vec![vec![0; n_processes]; n_airgroups];
        for global_id in self.pctx.dctx_get_worker_instances().iter() {
            if let Ok(owner) = self.pctx.dctx_get_process_owner_instance(*global_id) {
                airgroup_instances_alive[instances[*global_id].airgroup_id][owner as usize] = 1;
            }
        }
        let mut alives = vec![0; n_airgroups];
        let mut n_proofs_to_be_received = 0;
        for (airgroup, instances) in airgroup_instances_alive.iter().enumerate().take(n_airgroups) {
            for (p, &alive) in instances.iter().enumerate().take(n_processes) {
                alives[airgroup] += alive;
                if p != my_rank {
                    n_proofs_to_be_received += alive;
                }
            }
        }

        let mut total_proofs: usize = 0;
        for (airgroup, &n_proofs) in alives.iter().enumerate() {
            let n_recursive2_proofs = total_recursive_proofs(n_proofs, self.pctx.global_info.aggregation_arity);
            if n_recursive2_proofs.has_remaining {
                let setup = self.setups.get_setup(airgroup, 0, &ProofType::Recursive2)?;
                let publics_aggregation = n_publics_aggregation(&self.pctx, airgroup);
                let null_proof_buffer = vec![0; setup.proof_size as usize + publics_aggregation];
                let null_proof = Proof::new(ProofType::Recursive2, airgroup, 0, None, null_proof_buffer);
                self.recursive2_proofs[airgroup].write().unwrap().push(null_proof);
            }
            total_proofs += n_recursive2_proofs.n_proofs;
        }
        total_proofs += n_proofs_to_be_received;

        let recursive2_done = Arc::new(Counter::new());

        let mut recursive2_handles = Vec::new();
        for _ in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let setups_clone = self.setups.clone();
            let const_pols_clone = self.const_pols.clone();
            let const_tree_clone = self.const_tree.clone();
            let aux_trace_clone = self.aux_trace.clone();
            let recursive2_proofs_ongoing_clone = self.recursive2_proofs_ongoing.clone();
            let cancellation_info_clone = self.cancellation_info.clone();
            let memory_handler_recursive_witness = self.memory_handler_recursive_witness.clone();
            let rec2_witness_rx_clone = rec2_witness_rx.clone();
            let handle = std::thread::spawn(move || {
                while let Ok(mut witness) = rec2_witness_rx_clone.recv() {
                    if cancellation_info_clone.read_recover().token.is_cancelled() {
                        // Return the received witness buffer to its pool before bailing.
                        drop(memory_handler_recursive_witness.adopt_trace(std::mem::take(&mut witness.trace)));
                        break;
                    }
                    let id = {
                        let mut rec2_proofs = recursive2_proofs_ongoing_clone.write().unwrap();
                        let id = rec2_proofs.len();
                        rec2_proofs.push(None);
                        id
                    };

                    witness.global_idx = Some(id);

                    let new_proof = match gen_recursive_proof_size(&pctx_clone, &setups_clone, &witness) {
                        Ok(p) => p,
                        Err(e) => {
                            // generate_recursive_proof (which returns the buffer to its pool) is not
                            // reached here; return it so the recursive-witness pool doesn't shrink.
                            drop(memory_handler_recursive_witness.adopt_trace(std::mem::take(&mut witness.trace)));
                            cancellation_info_clone.write_recover().cancel(Some(e));
                            break;
                        }
                    };

                    let id = new_proof.global_idx.unwrap();
                    recursive2_proofs_ongoing_clone.write().unwrap()[id] = Some(new_proof);

                    let recursive2_lock = recursive2_proofs_ongoing_clone.read().unwrap();
                    let new_proof_ref = recursive2_lock[id].as_ref().unwrap();

                    if let Err(e) = generate_recursive_proof(
                        &pctx_clone,
                        &memory_handler_recursive_witness,
                        &setups_clone,
                        &mut witness,
                        new_proof_ref,
                        &aux_trace_clone,
                        &const_tree_clone,
                        &const_pols_clone,
                        false,
                        u64::MAX, // one-off launch: reserve stream internally
                        None,
                    ) {
                        cancellation_info_clone.write_recover().cancel(Some(e));
                        break;
                    };

                    if !pctx_clone.gpu {
                        launch_callback_c(id as u64, ProofType::Recursive2.into());
                    }
                }
            });
            recursive2_handles.push(handle);
        }

        let mut handle_recursives = Vec::new();
        for _ in 0..self.n_streams {
            let pctx_clone = self.pctx.clone();
            let memory_handler_recursive_witness = self.memory_handler_recursive_witness.clone();
            let setups_clone = self.setups.clone();
            let recursive2_proofs_clone = self.recursive2_proofs.clone();
            let recursive2_proofs_ongoing_clone = self.recursive2_proofs_ongoing.clone();
            let rec2_witness_tx_clone = rec2_witness_tx.clone();
            let recursive_rx_clone = completions.receiver();
            let recursive2_done_clone = recursive2_done.clone();
            let cancellation_info_clone = self.cancellation_info.clone();
            let handle_recursive = std::thread::spawn(move || {
                // Exits when the owner is dropped and the channel disconnects (no sentinel).
                while let Ok(msg) = recursive_rx_clone.recv() {
                    recursive2_done_clone.increment();
                    let id = msg.id;

                    if cancellation_info_clone.read_recover().token.is_cancelled() {
                        break;
                    }

                    let proof = recursive2_proofs_ongoing_clone.write().unwrap()[id as usize].take().unwrap();

                    let mut recursive2_airgroup_proofs = recursive2_proofs_clone[proof.airgroup_id].write().unwrap();
                    recursive2_airgroup_proofs.push(proof);

                    let arity = pctx_clone.global_info.aggregation_arity;
                    if recursive2_airgroup_proofs.len() >= arity {
                        let chunk: Vec<_> = (0..arity).map(|_| recursive2_airgroup_proofs.pop().unwrap()).collect();
                        let w = gen_witness_aggregation(
                            &pctx_clone,
                            &memory_handler_recursive_witness,
                            &setups_clone,
                            &chunk.iter().collect::<Vec<_>>(),
                        );
                        let witness = match w {
                            Ok(witness) => witness,
                            Err(e) => {
                                tracing::info!("Error generating recursive2 witness from recursive proofs: {}", e);
                                cancellation_info_clone.write_recover().cancel(Some(e));
                                break;
                            }
                        };
                        if let Err(crossbeam_channel::SendError(mut returned)) = rec2_witness_tx_clone.send(witness) {
                            // Every rec2 worker has exited, so the pipeline is torn down mid-run.
                            // Return the witness buffer to its pool (else it leaks inside the
                            // SendError and the pool comes back short) before surfacing the failure.
                            drop(memory_handler_recursive_witness.adopt_trace(std::mem::take(&mut returned.trace)));
                            cancellation_info_clone.write_recover().cancel(None);
                            break;
                        }
                    }
                }
            });
            handle_recursives.push(handle_recursive);
        }

        while n_proofs_to_be_received > 0 {
            // A dead or cancelled peer must not spin this loop forever: bail on cancellation and let
            // check_cancel surface it. We can't locally time out waiting on a specific peer.
            if self.cancellation_info.read_recover().token.is_cancelled() {
                break;
            }
            let mut progressed = false;
            for airgroup_id in 0..n_airgroups {
                let new_proof = self.mpi_ctx.check_incoming_proofs(airgroup_id);
                if let Some(proof) = new_proof {
                    let mut rec2_proofs = self.recursive2_proofs_ongoing.write().unwrap();
                    let id = rec2_proofs.len();
                    let recursive2_proof = Proof::new(ProofType::Recursive2, airgroup_id, 0, Some(id), proof);
                    rec2_proofs.push(Some(recursive2_proof));

                    launch_callback_c(id as u64, ProofType::Recursive2.into());

                    n_proofs_to_be_received -= 1;
                    progressed = true;
                }
            }
            // Nothing arrived this pass: yield instead of hot-spinning a core while waiting on peers.
            if !progressed {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }

        recursive2_done.wait_until_value_and_check_streams(
            total_proofs,
            || get_stream_proofs_non_blocking_c(self.pctx.get_device_buffers_ptr()),
            &self.cancellation_info,
        );
        // Release the completion capability: drain-then-release `Drop` clears the registration and
        // disconnects the consumer threads below.
        drop(completions);

        for handle in handle_recursives {
            handle.join().unwrap();
        }
        drop(rec2_witness_tx);
        drop(rec2_witness_rx);

        for handle in recursive2_handles {
            handle.join().unwrap();
        }

        self.check_cancel(false)?;

        if send_proofs {
            self.recursive2_proofs.iter().enumerate().for_each(|(airgroup_id, lock)| {
                let mut write_lock = lock.write().unwrap();
                while let Some(proof) = write_lock.pop() {
                    let proof = proof.proof;
                    self.pctx.mpi_ctx.send_proof_to_rank(&proof, airgroup_id, 0);
                }
            });
        }

        timer_stop_and_log_debug!(GENERATING_WORKER_RMA_COMPRESSED_PROOFS);

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn calc_witness_handler(
        &self,
        witness_done: Arc<Counter>,
        memory_handler: Arc<MemoryHandler<F>>,
        minimal_memory: bool,
        witness_start_time: Option<Arc<RwLock<Option<std::time::Instant>>>>,
        stats: bool,
    ) -> (Arc<Mutex<Option<std::thread::JoinHandle<()>>>>, Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>) {
        let witness_done_clone = witness_done.clone();
        let thread_budget_clone = self.thread_budget.clone();
        let pctx_clone = self.pctx.clone();
        let wcm_clone = self.wcm.clone();
        let memory_handler_clone = memory_handler.clone();
        let witness_handles = Arc::new(Mutex::new(Vec::new()));
        let witness_handles_clone = witness_handles.clone();
        let witness_rx = self.witness_rx.clone();
        let witness_rx_priority = self.witness_rx_priority.clone();
        let cancellation_info_clone = self.cancellation_info.clone();
        let n_threads_witness = self.num_threads_per_witness;
        let witness_start_time_clone = witness_start_time.clone();
        let sctx_admission = self.sctx.clone();
        let zone_airs = Arc::new(if self.pctx.gpu { self.zone_staging_airs() } else { HashMap::new() });
        // The same set the contribution workers commit on a slot, so a released buffer is never needed.
        let slot_airs = Arc::new(if self.pctx.gpu {
            slot_commit_airs(&self.pctx, &self.sctx, &slot_packed_info(&self.options)).1
        } else {
            std::collections::HashSet::new()
        });
        let class_sizes = self.pctx.basic_stream_sizes.clone();
        let n_classes = class_sizes.len();
        let witness_handler = if !minimal_memory && (self.pctx.gpu || stats) {
            // Ready instances waiting to be admitted, priority-first, and how many slots each air
            // currently holds. Taking straight off the channels would be pure FIFO with no choice;
            // pooling is what lets `witness_slot_cap` hold back an air that can only drain on one
            // stream so the slots it would have taken go to work the other streams can run.
            // Two pools, not one queue: the priority pool is always scanned first, so a priority
            // instance outranks every normal one however late it arrives, while arrival order is kept
            // within each pool. That is all the two channels mean now — a ranking hint, no longer a
            // queue-jump able to take every slot.
            let mut pending_priority: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            let mut pending: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            let in_flight: Arc<Mutex<HashMap<(usize, usize), usize>>> = Arc::new(Mutex::new(HashMap::new()));
            // Depth 1: a wakeup is a hint, not a count.
            let (slot_freed_tx, slot_freed_rx): (Sender<()>, Receiver<()>) = bounded(1);
            let mut arrivals_done = false;
            Some(std::thread::spawn(move || loop {
                // Take everything available without committing to any of it yet.
                while let Ok(id) = witness_rx_priority.try_recv() {
                    pending_priority.push_back(id);
                }
                while let Ok(id) = witness_rx.try_recv() {
                    if id == usize::MAX {
                        arrivals_done = true;
                    } else {
                        pending.push_back(id);
                    }
                }

                // 0 = unknown (CPU, or an air the carve does not know): never held back.
                let eligible_of = |id: usize| -> usize {
                    let Ok(key) = pctx_clone.dctx_get_instance_info(id) else { return 0 };
                    let need =
                        sctx_admission.get_setup(key.0, key.1).map(|s| s.prover_buffer_size as usize).unwrap_or(0);
                    crate::eligible_stream_count(need, &class_sizes)
                };
                let admissible = |id: usize, held: &HashMap<(usize, usize), usize>| -> bool {
                    let Ok(key) = pctx_clone.dctx_get_instance_info(id) else {
                        return true; // unknown air: never hold back work we cannot reason about
                    };
                    let cap = crate::witness_slot_cap(eligible_of(id), n_classes);
                    held.get(&key).copied().unwrap_or(0) < cap
                };
                // First admissible, priority pool first. Not reordered: `weight` is the proof cost
                // of the air, so it is equal for every instance of one, and the instances of an air
                // spread 4.6x in witness time -- it cannot rank the thing we would want to rank.
                let first = |pool: &std::collections::VecDeque<usize>,
                             held: &HashMap<(usize, usize), usize>|
                 -> Option<usize> { pool.iter().position(|&id| admissible(id, held)) };
                let chosen: Option<(bool, usize)> = {
                    let held = in_flight.lock().unwrap();
                    first(&pending_priority, &held)
                        .map(|pos| (true, pos))
                        .or_else(|| first(&pending, &held).map(|pos| (false, pos)))
                };

                let instance_id =
                    match chosen.and_then(
                        |(is_priority, pos)| {
                            if is_priority {
                                pending_priority.remove(pos)
                            } else {
                                pending.remove(pos)
                            }
                        },
                    ) {
                        Some(id) => id,
                        None => {
                            // Nothing admissible. Exit only once no more can arrive and nothing is queued.
                            if cancellation_info_clone.read_recover().token.is_cancelled() {
                                break;
                            }
                            if arrivals_done && pending.is_empty() && pending_priority.is_empty() {
                                break;
                            }
                            // Wait on every event that can make work admissible -- an arrival on
                            // either channel, or a freed slot -- rather than polling. Was a flat 1 ms
                            // sleep: measured at ~8 ms per witness, 540 ms over a phase.
                            let mut select = crossbeam_channel::Select::new();
                            let priority_op = select.recv(&witness_rx_priority);
                            let normal_op = select.recv(&witness_rx);
                            let slot_op = select.recv(&slot_freed_rx);
                            if let Ok(op) = select.select_timeout(ADMISSION_WAIT) {
                                let index = op.index();
                                if index == priority_op {
                                    if let Ok(id) = op.recv(&witness_rx_priority) {
                                        pending_priority.push_back(id);
                                    }
                                } else if index == normal_op {
                                    match op.recv(&witness_rx) {
                                        Ok(id) if id == usize::MAX => arrivals_done = true,
                                        Ok(id) => pending.push_back(id),
                                        Err(_) => {}
                                    }
                                } else if index == slot_op {
                                    let _ = op.recv(&slot_freed_rx);
                                }
                            }
                            continue;
                        }
                    };

                if let Some(witness_start_time_clone) = &witness_start_time_clone {
                    if witness_start_time_clone.read().unwrap().is_none() {
                        *witness_start_time_clone.write().unwrap() = Some(std::time::Instant::now());
                    }
                }

                let (airgroup_id, air_id) = match pctx_clone.dctx_get_instance_info(instance_id) {
                    Ok(v) => v,
                    Err(e) => {
                        cancellation_info_clone.write_recover().cancel(Some(e));
                        break;
                    }
                };

                // Held until the witness thread finishes: the span this air occupies a slot.
                *in_flight.lock().unwrap().entry((airgroup_id, air_id)).or_insert(0) += 1;
                let slot =
                    SlotGuard { in_flight: in_flight.clone(), key: (airgroup_id, air_id), wake: slot_freed_tx.clone() };

                let wcm = wcm_clone.clone();
                let memory_handler_clone = memory_handler_clone.clone();

                let witness_done_clone = witness_done_clone.clone();
                let tokens = thread_budget_clone.acquire(n_threads_witness, &cancellation_info_clone);

                if cancellation_info_clone.read_recover().token.is_cancelled() {
                    break;
                }

                let pctx_clone = pctx_clone.clone();
                let gpu = pctx_clone.gpu;
                let cancellation_info_clone = cancellation_info_clone.clone();
                let zone_airs = zone_airs.clone();
                let slot_airs = slot_airs.clone();
                // Contributions: stage the finished witness and give its buffer back BEFORE the
                // commit can see the instance, so the two never race for the buffer.
                let stage_first = gpu
                    && !stats
                    && witness_start_time_clone.is_some()
                    && zone_airs.contains_key(&(airgroup_id, air_id));
                let handle = std::thread::spawn(move || {
                    timer_start_debug!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    if stage_first {
                        pctx_clone.dispatch_deferred[instance_id].store(true, Ordering::SeqCst);
                    }
                    if let Err(e) =
                        wcm.calculate_witness(1, &[instance_id], n_threads_witness, memory_handler_clone.as_ref())
                    {
                        cancellation_info_clone.write_recover().cancel(Some(e));
                    }
                    // The staging below is an H2D wait, not CPU work: give the tokens and the
                    // admission slot back first so the next witness can start.
                    drop(tokens);
                    drop(slot);
                    if stage_first {
                        match Self::stage_for_commit(
                            &pctx_clone,
                            &zone_airs,
                            memory_handler_clone.as_ref(),
                            &slot_airs,
                            instance_id,
                        ) {
                            Ok(s) => pctx_clone.witness_staged[instance_id].store(s, Ordering::Release),
                            Err(e) => cancellation_info_clone.write_recover().cancel(Some(e)),
                        }
                        pctx_clone.end_deferred_dispatch(instance_id);
                    }
                    // The buffer carries its own wait, whichever worker blocked for it.
                    let waited = proofman_common::take_buffer_wait(pctx_clone.get_air_instance_trace_ptr(instance_id));
                    timer_stop_and_log_debug_net!(
                        GENERATING_WC,
                        waited,
                        "GENERATING_WC_{} [{}:{}]",
                        instance_id,
                        airgroup_id,
                        air_id
                    );
                    witness_done_clone.increment();
                    if stats {
                        let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance_traces(instance_id);
                        if is_shared_buffer {
                            if let Err(e) = memory_handler_clone.release_buffer(witness_buffer) {
                                cancellation_info_clone.write_recover().cancel(Some(e));
                            }
                        }
                    }
                });
                witness_handles_clone.lock().unwrap().push(handle);
            }))
        } else {
            None
        };
        (Arc::new(Mutex::new(witness_handler)), witness_handles)
    }

    fn calculate_witness(
        &self,
        instances: &[usize],
        memory_handler: Arc<MemoryHandler<F>>,
        witness_done: Arc<Counter>,
        minimal_memory: bool,
        stats: bool,
    ) -> ProofmanResult<()> {
        let witness_minimal_memory_handles: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let _join_guard = JoinAllGuard::new(witness_minimal_memory_handles.clone());
        // Skipped instances never increment, so exclude them from the wait target, else witness_done
        // can't reach it and the wait stalls with no cancellation.
        let mut expected = instances.len();
        if !minimal_memory && (self.pctx.gpu || stats) {
            timer_start_debug!(PRE_CALCULATE_WC);
            self.wcm.pre_calculate_witness(1, instances, self.max_num_threads, memory_handler.as_ref())?;
            timer_stop_and_log_debug!(PRE_CALCULATE_WC);
        } else {
            for &instance_id in instances.iter() {
                let (skip, _) = skip_prover_instance(&self.pctx, instance_id)?;
                if skip {
                    expected -= 1;
                    continue;
                }
                let n_threads_witness = self.num_threads_per_witness;

                let (airgroup_id, air_id) = self.pctx.dctx_get_instance_info(instance_id)?;
                let threads_to_use_collect = match self.pctx.gpu || stats {
                    true => (self.pctx.dctx_get_instance_chunks(instance_id)? / 16)
                        .max(self.max_num_threads / 4)
                        .min(n_threads_witness)
                        .min(self.max_num_threads),
                    false => self.max_num_threads,
                };

                let mut tokens = self.thread_budget.acquire(threads_to_use_collect, &self.cancellation_info);

                if self.cancellation_info.read_recover().token.is_cancelled() {
                    break;
                }

                let threads_to_use_witness = match self.pctx.gpu || stats {
                    true => threads_to_use_collect.min(n_threads_witness),
                    false => self.max_num_threads,
                };

                let threads_to_return = threads_to_use_collect - threads_to_use_witness;

                let pctx_clone = self.pctx.clone();
                let wcm_clone = self.wcm.clone();
                let memory_handler_clone = memory_handler.clone();
                let witness_done_clone = witness_done.clone();
                let cancellation_info_clone = self.cancellation_info.clone();
                let handle = std::thread::spawn(move || {
                    timer_start_debug!(GENERATING_WC, "GENERATING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    timer_start_debug!(PREPARING_WC, "PREPARING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    if let Err(e) = wcm_clone.pre_calculate_witness(
                        1,
                        &[instance_id],
                        threads_to_use_collect,
                        memory_handler_clone.as_ref(),
                    ) {
                        cancellation_info_clone.write_recover().cancel(Some(e));
                        return;
                    }
                    // The lookup clears it, so the second span sees zero rather than double-counting.
                    let preparing_waited =
                        proofman_common::take_buffer_wait(pctx_clone.get_air_instance_trace_ptr(instance_id));
                    timer_stop_and_log_debug_net!(
                        PREPARING_WC,
                        preparing_waited,
                        "PREPARING_WC_{} [{}:{}]",
                        instance_id,
                        airgroup_id,
                        air_id
                    );
                    tokens.release(threads_to_return);

                    timer_start_debug!(COMPUTING_WC, "COMPUTING_WC_{} [{}:{}]", instance_id, airgroup_id, air_id);
                    if let Err(e) = wcm_clone.calculate_witness(
                        1,
                        &[instance_id],
                        threads_to_use_witness,
                        memory_handler_clone.as_ref(),
                    ) {
                        cancellation_info_clone.write_recover().cancel(Some(e));
                        return;
                    }
                    let computing_waited =
                        proofman_common::take_buffer_wait(pctx_clone.get_air_instance_trace_ptr(instance_id));
                    timer_stop_and_log_debug_net!(
                        COMPUTING_WC,
                        computing_waited,
                        "COMPUTING_WC_{} [{}:{}]",
                        instance_id,
                        airgroup_id,
                        air_id
                    );
                    tokens.release(threads_to_use_witness);
                    timer_stop_and_log_debug_net!(
                        GENERATING_WC,
                        preparing_waited + computing_waited,
                        "GENERATING_WC_{} [{}:{}]",
                        instance_id,
                        airgroup_id,
                        air_id
                    );
                    witness_done_clone.increment();
                    if stats {
                        let (is_shared_buffer, witness_buffer) = pctx_clone.free_instance_traces(instance_id);
                        if is_shared_buffer {
                            if let Err(e) = memory_handler_clone.release_buffer(witness_buffer) {
                                cancellation_info_clone.write_recover().cancel(Some(e));
                            }
                        }
                    }
                });
                if !stats && !self.pctx.gpu {
                    handle.join().unwrap();
                } else {
                    witness_minimal_memory_handles.lock().unwrap().push(handle);
                }
            }
        }

        witness_done.wait_until_value_and_check_streams(
            expected,
            || get_stream_proofs_non_blocking_c(self.pctx.get_device_buffers_ptr()),
            &self.cancellation_info,
        );

        let handles_to_join: Vec<_> = witness_minimal_memory_handles.lock().unwrap().drain(..).collect();
        for handle in handles_to_join {
            handle.join().unwrap();
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn debug_print_airgroup_values(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        proofs: &[RwLock<Option<Proof<F>>>],
        id: u64,
        proof_type: &ProofType,
    ) {
        const HASH_SIZE: usize = 4;
        if *proof_type != ProofType::Basic {
            return;
        }
        let (airgroup_id, air_id) = match pctx.dctx_get_instance_info(id as usize) {
            Ok(info) => info,
            Err(_) => return,
        };
        let Ok(setup) = sctx.get_setup(airgroup_id, air_id) else { return };
        const FE: usize = 3;
        let n_airgroup = setup.stark_info.airgroupvalues_map.as_deref().map(|m| m.len()).unwrap_or(0);
        let n_air = setup.stark_info.airvalues_map.as_deref().map(|m| m.len()).unwrap_or(0);
        let airgroup_words = n_airgroup * FE;
        let n_stage_roots = setup.stark_info.n_stages as usize + 1; // +1 = Q stage

        let guard = proofs[id as usize].read().unwrap();
        let Some(proof) = guard.as_ref() else { return };
        let buf = &proof.proof;
        if airgroup_words > buf.len() {
            return;
        }

        let vals = buf[0..airgroup_words].iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
        tracing::info!("··· Instance {} [{}:{}]: airgroup values: [{}]", id, airgroup_id, air_id, vals);

        let roots_base = airgroup_words + n_air * FE;
        for s in 0..n_stage_roots {
            let off = roots_base + s * HASH_SIZE;
            if off + HASH_SIZE > buf.len() {
                break;
            }
            let label = if s + 1 == n_stage_roots { "Q".to_string() } else { (s + 1).to_string() };
            let root = buf[off..off + HASH_SIZE].iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
            tracing::info!("··· Instance {} [{}:{}]: root stage {}: [{}]", id, airgroup_id, air_id, label, root);
        }

        if let Some(nonce) = buf.last() {
            tracing::info!("··· Instance {} [{}:{}]: nonce: {}", id, airgroup_id, air_id, nonce);
        }
    }

    /// Verify an aggregated proof received from a worker with the C++ STARK verifier on the key's
    /// own setup files. circuit_type (publics[0]): 0 = null proof (no-op), 1 = recursive2, k >= 2 =
    /// the un-aggregated recursive1 of air k-2 that a single-instance worker sends -- verified with
    /// that air's recursive1 setup (same circuit shape, its own root_c).
    fn verify_agg_proof(&self, airgroup_id: usize, proof_data: &[u64]) -> ProofmanResult<bool> {
        let publics_aggregation = n_publics_aggregation(&self.pctx, airgroup_id);
        let (publics, rec_proof) = proof_data.split_at(publics_aggregation);
        // These words come off the wire and are read back as the proof's outputs, so pin them to
        // one encoding: verification reduces, making `x` and `x + p` pass alike. Only the challenge
        // slice is otherwise covered, by the caller's `as_canonical_u64` comparison.
        if let Some(i) = publics.iter().position(|&word| word >= F::ORDER_U64) {
            tracing::error!(
                "Aggregated public {i} from airgroup {airgroup_id} is not canonical: {} >= {}",
                publics[i],
                F::ORDER_U64
            );
            return Ok(false);
        }
        let circuit_type = publics[0];
        if circuit_type == 0 {
            return Ok(true);
        }
        let (setup, setup_path) = if circuit_type == 1 {
            (
                self.setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0)?,
                self.pctx.global_info.get_air_setup_path(airgroup_id, 0, &ProofType::Recursive2),
            )
        } else {
            let air_id = circuit_type as usize - 2;
            (
                self.setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id)?,
                self.pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Recursive1),
            )
        };
        let mut publics_extended = vec![0u64; setup.stark_info.n_publics as usize];
        publics_extended[0..publics.len()].copy_from_slice(publics);
        add_publics_circom(&mut publics_extended, publics_aggregation, &self.pctx, Some(&setup.verkey));
        let publics_f: Vec<F> = publics_extended.iter().map(|&x| F::from_u64(x)).collect();
        let base = setup_path.display().to_string();
        Ok(verify_proof::<F>(
            rec_proof,
            base.clone() + ".starkinfo.json",
            base.clone() + ".verifier.bin",
            base + ".verkey.json",
            Some(publics_f),
            None,
            None,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn gen_proof(
        proofs: &[RwLock<Option<Proof<F>>>],
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        aux_trace: &[F],
        const_pols: &[F],
        const_tree: &[F],
        reserved_stream: Option<usize>,
        // true: no global challenge -- the transcript is seeded from this AIR's own
        // verkey + publics (see genProof's `recursive` branch).
        self_contained: bool,
    ) -> ProofmanResult<usize> {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
        timer_start_debug!(GEN_PROOF, "GEN_PROOF_{} [{}:{}]", instance_id, airgroup_id, air_id);
        Self::initialize_air_instance(pctx, sctx, instance_id, false, false, None, None)?;

        let setup = sctx.get_setup(airgroup_id, air_id)?;
        let p_setup: *mut c_void = (&setup.p_setup).into();

        let mut steps_params = pctx.get_air_instance_params(instance_id, true);

        if !pctx.gpu {
            steps_params.aux_trace = aux_trace.as_ptr() as *mut u8;
            steps_params.p_const_pols = const_pols.as_ptr() as *mut u8;
            steps_params.p_const_tree = const_tree.as_ptr() as *mut u8;
        } else {
            steps_params.p_const_pols = std::ptr::null_mut();
            steps_params.p_const_tree = std::ptr::null_mut();
        }

        let p_steps_params: *mut u8 = (&steps_params).into();

        let const_pols_path = &setup.const_pols_path;
        let const_pols_tree_path = &setup.const_pols_tree_path;

        // Resolve the custom-commits-fixed file path for this instance, if any.
        // Empty string means no custom commits — C++ side reads only when non-empty.
        let custom_commits_fixed_path = match setup.stark_info.custom_commits.iter().find(|c| c.stage_widths[0] > 0) {
            Some(c) => pctx.get_custom_commits_fixed_buffer(&c.name, true)?.to_string_lossy().into_owned(),
            None => String::new(),
        };

        // reserved_stream Some -> scheduler-reserved stream; None -> u64::MAX (CPU path —
        // gen_proof_cpu ignores it; on GPU the scheduler always reserves).
        let stream_id: u64 = reserved_stream.map_or(u64::MAX, |s| s as u64);

        let proof = create_buffer_fast(setup.proof_size as usize);
        *proofs[instance_id].write().unwrap() =
            Some(Proof::new(ProofType::Basic, airgroup_id, air_id, Some(instance_id), proof));

        // Returns the stream the (async) trace H2D + commit ran on; the caller
        // must wait on it before recycling this instance's shared trace buffer.
        let proof_stream_id = gen_proof_c(
            p_setup,
            p_steps_params,
            pctx.get_global_challenge_ptr(),
            proofs[instance_id].read().unwrap().as_ref().unwrap().proof.as_ptr() as *mut u64,
            "",
            airgroup_id as u64,
            air_id as u64,
            instance_id as u64,
            pctx.get_device_buffers_ptr(),
            stream_id,
            const_pols_path,
            const_pols_tree_path,
            &custom_commits_fixed_path,
            self_contained,
        );

        if !pctx.gpu {
            launch_callback_c(instance_id as u64, "basic");
        }

        timer_stop_and_log_debug!(GEN_PROOF, "GEN_PROOF_{} [{}:{}]", instance_id, airgroup_id, air_id);
        Ok(proof_stream_id as usize)
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn initialize_proofman(
        mpi_ctx: Arc<MpiCtx>,
        proving_key_path: PathBuf,
        options: &ProofmanOptions,
    ) -> ProofmanResult<(Arc<ProofCtx<F>>, Arc<SetupCtx<F>>, Arc<SetupsVadcop<F>>, u64, u64, u64, u64)> {
        if !set_gpu_mode_c(options.gpu) {
            return Err(ProofmanError::InvalidConfiguration(
                "GPU mode requested but library was built without CUDA support".into(),
            ));
        }

        let mut pctx = ProofCtx::create_ctx(
            proving_key_path,
            options.aggregation,
            options.verbose_mode,
            mpi_ctx.clone(),
            options.gpu,
        )?;

        // Components must pack exactly the airs the device will unpack: it gates every packed
        // read path on `packedTrace && is_packed`, so a global flag alone would corrupt the trace.
        if options.packed {
            pctx.packed_airs =
                options.packed_info.iter().filter(|(_, info)| info.is_packed).map(|(key, _)| *key).collect();
        }
        timer_start_info!(INITIALIZING_PROOFMAN);

        let sctx: Arc<SetupCtx<F>> = Arc::new(SetupCtx::new_with_commit_files(
            &pctx.global_info,
            &ProofType::Basic,
            options.verify_constraints,
            options.gpu,
            &options.custom_commits_fixed,
        )?);

        let setups_vadcop = Arc::new(SetupsVadcop::new(
            &pctx.global_info,
            options.verify_constraints,
            options.aggregation,
            options.gpu,
        )?);

        pctx.set_weights(&sctx, &setups_vadcop)?;

        // Prefetch-zone sizing, needed BEFORE the unified-buffer allocation: the zone is
        // carved from a region INSIDE the unified buffer (below the consts), so it shares
        // one planned budget. Witness bytes = the largest basic trace (packed width when
        // the air is packed). Default ON; PROOFMAN_PREFETCH=0 disables.
        let prefetch_witness_bytes: u64 =
            if options.gpu && std::env::var("PROOFMAN_PREFETCH").map(|v| v != "0").unwrap_or(true) {
                let mut witness_bytes: u64 = 0;
                for (airgroup_id, group) in pctx.global_info.airs.iter().enumerate() {
                    for (air_id, _) in group.iter().enumerate() {
                        // A GPU-witness air never uploads a trace, so it must not
                        // widen the slots the uploads share.
                        if options.gpu_witness_airs.contains(airgroup_id, air_id) {
                            continue;
                        }
                        let Ok(setup) = sctx.get_setup(airgroup_id, air_id) else { continue };
                        let n = 1u64 << setup.stark_info.stark_struct.n_bits;
                        let cm1 = setup.stark_info.map_sections_n.get("cm1").copied().unwrap_or(0);
                        let packed_words = options
                            .packed_info
                            .get(&(airgroup_id, air_id))
                            .filter(|pi| pi.is_packed && options.packed)
                            .map(|pi| pi.num_packed_words);
                        witness_bytes = witness_bytes.max(packed_words.unwrap_or(cm1) * n * 8);
                    }
                }
                witness_bytes
            } else {
                0
            };
        // Slot count comes from the C++ side (DeviceCommitBuffers::PREFETCH_WITNESS_SLOTS),
        // the single source of truth configure_prefetch_zone sizes against.
        let prefetch_region_area: u64 = (prefetch_witness_bytes * get_prefetch_witness_slots_c() as u64).div_ceil(8);

        let (n_streams_per_gpu, n_aggregation_workers_per_gpu, n_gpus) = pctx.set_device_buffers(
            &sctx,
            &setups_vadcop,
            options.aggregation,
            options.gpu,
            options.max_number_streams,
            options.max_number_recursive_streams,
            options.final_snark,
            prefetch_region_area,
        )?;

        use_packed_trace_c(pctx.get_device_buffers_ptr(), options.packed);

        // Arm the prefetch zone with the size computed above: the zone IS the unified
        // buffer's region (arming refuses on a mismatch).
        if prefetch_witness_bytes > 0 {
            configure_prefetch_zone_c(pctx.get_device_buffers_ptr(), prefetch_witness_bytes);
        }

        // GPU witness airs, once the zone's fate is known. They have no host trace, so every
        // condition below must hold or the run stops; there is no host fallback.
        // The C++ kernel registry is process-wide and outlives this ProofMan; clear it when this job
        // declares none, or a later commit runs a stale kernel and aborts in cudaMemcpy.
        if options.gpu_witness_airs.is_empty() {
            proofman_starks_lib_c::gpu_witness_clear_c();
        } else {
            if !options.gpu {
                return Err(ProofmanError::InvalidConfiguration(
                    "GPU witness airs were declared but this is not a GPU run".into(),
                ));
            }
            options.gpu_witness_airs.register();
            let registered = gpu_witness_count_c();
            if registered != options.gpu_witness_airs.len() as u64 {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "declared {} GPU witness airs but the prover registered {registered}",
                    options.gpu_witness_airs.len(),
                )));
            }
            tracing::info!("GPU witness kernels registered: {registered}");
        }

        // Streaming-commit slots, the GPU's only contributions path. The slot COUNT is a memory-budget
        // knob (each slot lowers the ceiling on what gpu-mops may borrow); the slot SIZE is the largest
        // air's commit, from `slot_commit_airs`, the same set try_slot_commit admits.
        const STREAM_COMMIT_SLOTS: u64 = 2;
        if options.gpu {
            let (slot_bytes, _, unfit) = slot_commit_airs(&pctx, &sctx, &slot_packed_info(options));
            // Fatal only if one is instantiated (try_slot_commit): an unused air must not stop the run.
            if !unfit.is_empty() {
                tracing::warn!("No streaming slot can commit these airs: {}", unfit.join(", "));
            }
            if slot_bytes > 0 {
                configure_stream_commit_slots_c(pctx.get_device_buffers_ptr(), STREAM_COMMIT_SLOTS, slot_bytes);
            }
            if get_stream_commit_slots_c(pctx.get_device_buffers_ptr()) == 0 {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "no streaming-commit slot of {} fits the unified buffer",
                    proofman_common::format_bytes(slot_bytes as f64)
                )));
            }
        }

        load_device_setups(&pctx, &sctx, &setups_vadcop, options.aggregation, &options.packed_info)?;

        if mpi_ctx.rank == 0 {
            let (needs_const_regen, needs_tree_regen) = needs_regeneration_fixed(&pctx, &sctx)?;
            if needs_const_regen {
                tracing::info!("Regenerating GPU constant polynomials (one-time setup)...");
                timer_start_info!(REGENERATING_GPU_CONST_POLS);
                check_const_paths(&pctx, &sctx)?;
                timer_stop_and_log_info!(REGENERATING_GPU_CONST_POLS);
            }

            if !options.verify_constraints && needs_tree_regen {
                tracing::info!("Regenerating constant trees (one-time setup)...");
                timer_start_info!(REGENERATING_CONST_TREE);
                check_tree_paths(&pctx, &sctx)?;
                timer_stop_and_log_info!(REGENERATING_CONST_TREE);
            }

            if options.aggregation {
                let (needs_vadcop_const_regen, needs_vadcop_tree_regen) =
                    needs_regeneration_vadcop_fixed(&pctx, &setups_vadcop)?;
                if needs_vadcop_const_regen {
                    tracing::info!("Regenerating Vadcop constant polynomials (one-time setup)...");
                    timer_start_info!(REGENERATING_VADCOP_CONST_POLS);
                    check_const_paths_vadcop(&pctx, &setups_vadcop)?;
                    timer_stop_and_log_info!(REGENERATING_VADCOP_CONST_POLS);
                }

                if needs_vadcop_tree_regen {
                    tracing::info!("Regenerating Vadcop constant trees (one-time setup)...");
                    timer_start_info!(REGENERATING_VADCOP_CONST_TREE);
                    check_tree_paths_vadcop(&pctx, &setups_vadcop)?;
                    timer_stop_and_log_info!(REGENERATING_VADCOP_CONST_TREE);
                }
            }
        }
        mpi_ctx.barrier();

        timer_start_info!(LOADING_FIXED_POLS);
        // End of the init-time aggregation const-pols uploads = start of the
        // reserved recurser slot (see register_recurser_setup).
        let aggregation_const_end = load_device_const_pols(&pctx, &sctx, &setups_vadcop, options.aggregation, false)?;
        timer_stop_and_log_info!(LOADING_FIXED_POLS);

        let pctx = Arc::new(pctx);

        timer_stop_and_log_info!(INITIALIZING_PROOFMAN);

        Ok((pctx, sctx, setups_vadcop, n_streams_per_gpu, n_aggregation_workers_per_gpu, n_gpus, aggregation_const_end))
    }

    #[allow(dead_code)]
    fn diagnostic_instance(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, instance_id: usize) -> ProofmanResult<bool> {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id)?;
        let air_name = &pctx.global_info.airs[airgroup_id][air_id].name;
        let setup = sctx.get_setup(airgroup_id, air_id)?;
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
                break;
            }
        }

        Ok(invalid_initialization)
    }

    fn initialize_air_instance(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        init_aux_trace: bool,
        verify_constraints: bool,
        shared_const_pols: Option<&SharedScratch<F>>,
        shared_aux_trace: Option<&Arc<Vec<F>>>,
    ) -> ProofmanResult<()> {
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
        let setup = sctx.get_setup(airgroup_id, air_id)?;

        let mut air_instance = pctx.air_instances[instance_id].write().unwrap();

        if air_instance.num_rows != (1 << setup.stark_info.stark_struct.n_bits) {
            return Err(ProofmanError::InvalidSetup(format!(
                "Row count mismatch for airgroup_id={}, air_id={}: expected {} rows (from proving key), but got {} rows (from pil-helpers).",
                airgroup_id,
                air_id,
                1 << setup.stark_info.stark_struct.n_bits,
                air_instance.num_rows
            )));
        }

        // Host aux_trace / const_pols are only used by the CPU verify path. On GPU the C side reads
        // pre-loaded device buffers, so skip the host allocation and file load (see
        // verify_constraints_gpu in starks_api.cu).
        if init_aux_trace && !pctx.gpu {
            air_instance.init_aux_trace(shared_aux_trace.expect("CPU verify requires shared aux_trace").clone());
        }
        air_instance.init_evals(setup.stark_info.ev_map.len() * 3);
        air_instance.init_challenges(
            (setup.stark_info.challenges_map.as_ref().unwrap().len() + setup.stark_info.stark_struct.steps.len() + 1)
                * 3,
        );

        if verify_constraints && !pctx.gpu {
            // Reuse the process-wide const_pols buffer, loading per-air const data each call. CPU
            // verify runs sequentially (single writer); `SharedScratch` localizes the reborrow.
            let scratch = shared_const_pols.expect("CPU verify requires shared const_pols");
            {
                let mut const_pols_view = scratch.borrow_mut();
                load_const_pols(setup, &mut const_pols_view);
            }
            air_instance.init_fixed(scratch.arc().clone());
        }

        // CPU only: allocate the host buffer and load each custom-commit file. In GPU mode custom
        // commits stream disk → device via customCommitsFixedPath, so the host buffer is dead weight.
        if !pctx.gpu {
            air_instance.init_custom_commit_fixed_trace(setup.custom_commits_fixed_buffer_size as usize);

            let n_custom_commits = setup.stark_info.custom_commits.len();
            for commit_id in 0..n_custom_commits {
                if setup.stark_info.custom_commits[commit_id].stage_widths[0] > 0 {
                    let name = &setup.stark_info.custom_commits[commit_id].name;
                    let custom_commit_file_path = pctx.get_custom_commits_fixed_buffer(name, true).unwrap();

                    load_custom_commit_c(
                        (&setup.p_setup).into(),
                        commit_id as u64,
                        air_instance.get_custom_commits_fixed_ptr(),
                        custom_commit_file_path.to_str().expect("Invalid path"),
                        pctx.get_custom_commit_words_per_row(name),
                    );
                }
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
        Ok(())
    }

    fn set_publics_custom_commits(sctx: &SetupCtx<F>, pctx: &ProofCtx<F>) -> ProofmanResult<()> {
        tracing::debug!("Initializing publics custom_commits");
        for (airgroup_id, airs) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in airs.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id)?;
                for custom_commit in &setup.stark_info.custom_commits {
                    if custom_commit.stage_widths[0] > 0 {
                        let root_bytes = pctx.get_custom_commit_root(&custom_commit.name)?;

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

    /// Commit `instance_id` on a streaming slot, the GPU's only contributions path: an air no slot can
    /// take is an error. A witness staged in a GPU's zone commits on that GPU's slots; any other takes
    /// the first free slot on any GPU.
    ///
    /// Indexed AIRs are eligible: the C side looks up the air's indexed descriptor
    /// (col_source / col_lane / index_bits / lanes / words_per_entry) and its uploaded
    /// instruction table from the slot GPU's AirInstanceInfo, and drives the indexed
    /// slot unpack -- hence airgroup_id/air_id are passed through.
    #[allow(clippy::too_many_arguments)]
    fn try_slot_commit(
        pctx: &ProofCtx<F>,
        setup: &Setup<F>,
        ctx: &SlotCommitCtx,
        instance_id: usize,
        airgroup_id: usize,
        air_id: usize,
        trace: *mut u8,
        params: *mut u8,
        roots_contributions: &[[F; 4]],
        staged: bool,
        staged_gpu: Option<usize>,
    ) -> ProofmanResult<()> {
        let refuse = |why: &str| {
            Err(ProofmanError::ProofmanError(format!(
                "instance {instance_id} [{airgroup_id}:{air_id}] cannot commit on a streaming slot: {why}"
            )))
        };
        if !ctx.airs.contains(&(airgroup_id, air_id)) {
            return refuse("no slot takes this air (hash family, merkle arity or width)");
        }
        // No packing info (virtual tables, Rom): the identity packing, one word per column.
        let identity = PackedInfo::default();
        let pi = ctx.packed_info.get(&(airgroup_id, air_id)).unwrap_or(&identity);
        // A null trace is fine when staged (the slot reads the zone) or when the device produces the
        // rows (a prover-owned virtual table, exported from its accumulator).
        let device_rows = pctx.device_owned_table_airs.read().unwrap().contains(&air_id);
        if trace.is_null() && !staged && !device_rows {
            return refuse("no host trace and no staging");
        }
        let ss = &setup.stark_info.stark_struct;
        // `ctx.airs` only holds airs with a cm1 section.
        let n_cols = setup.stark_info.map_sections_n["cm1"];
        // Slots are allowed outside the gpu-mops borrow window too. An unpacked air is the degenerate
        // packing (one 64-bit word per column), which the unpack cursor reads as identity.
        // Any width fits: the C side refuses more than SC_MAX_COLS columns before reading these.
        static IDENTITY_WIDTHS: [u64; 512] = [64; 512];
        let (words_per_row, widths): (u64, &[u64]) = if pi.is_packed {
            (pi.num_packed_words, pi.unpack_info.as_slice())
        } else {
            (n_cols, &IDENTITY_WIDTHS[..(n_cols as usize).min(IDENTITY_WIDTHS.len())])
        };
        // Wait for a token (C waits out quiesce and region), so one call commits or errors.
        let timeout = std::time::Duration::from_secs(60);
        let (pool, slot) = match staged_gpu {
            Some(g) => (g, ctx.pools[g].1.recv_timeout(timeout).ok()),
            None => {
                let mut sel = crossbeam_channel::Select::new();
                for (_, rx) in &ctx.pools {
                    sel.recv(rx);
                }
                match sel.select_timeout(timeout) {
                    Ok(op) => {
                        let g = op.index();
                        (g, op.recv(&ctx.pools[g].1).ok())
                    }
                    Err(_) => (0, None),
                }
            }
        };
        let Some(slot) = slot else {
            dump_pipeline_state_c(pctx.get_device_buffers_ptr());
            return Err(ProofmanError::ProofmanError(format!(
                "no streaming slot came free in 60 s for instance {instance_id} [{airgroup_id}:{air_id}]"
            )));
        };
        let rc = commit_witness_streaming_c(
            pctx.get_device_buffers_ptr(),
            slot,
            instance_id as u64,
            airgroup_id as u64,
            air_id as u64,
            trace as *mut c_void,
            ss.n_bits,
            ss.n_bits_ext,
            n_cols,
            words_per_row,
            widths.as_ptr() as *mut c_void,
            roots_contributions[instance_id].as_ptr() as *mut c_void,
            params as *mut c_void,
        );
        ctx.pools[pool].0.send(slot).ok();
        if rc != 0 {
            if rc == -20 || rc == -21 {
                dump_pipeline_state_c(pctx.get_device_buffers_ptr());
            }
            return Err(ProofmanError::ProofmanError(format!(
                "Streaming slot commit refused (rc={rc}) for instance {instance_id} [{airgroup_id}:{air_id}]; \
                 -20/-21 mean the quiesce or the region did not lift in 60 s, anything else is a configuration error"
            )));
        }
        ctx.committed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Per air: wire bytes of its witness (what the zone stages), and whether its host buffer may
    /// return to the pool once staged. GPU-witness airs are absent.
    fn zone_staging_airs(&self) -> HashMap<(usize, usize), (u64, bool)> {
        let mut m = HashMap::new();
        for (airgroup_id, group) in self.pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in group.iter().enumerate() {
                if self.options.gpu_witness_airs.contains(airgroup_id, air_id) {
                    continue;
                }
                let Ok(setup) = self.sctx.get_setup(airgroup_id, air_id) else { continue };
                let n = 1u64 << setup.stark_info.stark_struct.n_bits;
                let cm1 = setup.stark_info.map_sections_n.get("cm1").copied().unwrap_or(0);
                let words = self
                    .options
                    .packed_info
                    .get(&(airgroup_id, air_id))
                    .filter(|pi| pi.is_packed && self.options.packed)
                    .map(|pi| pi.num_packed_words)
                    .unwrap_or(cm1);
                // clear_traces() also drops custom_commits_fixed, which such a commit still needs.
                let releasable = !setup.stark_info.custom_commits.iter().any(|c| c.stage_widths[0] > 0);
                m.insert((airgroup_id, air_id), (words * n * 8, releasable));
            }
        }
        m
    }

    /// Stage `instance_id`'s witness into a GPU's prefetch zone. Returns a `ProofCtx::witness_staged`
    /// state: not staged, staged with the host buffer kept, or staged with it given back.
    fn stage_for_commit(
        pctx: &ProofCtx<F>,
        zone_airs: &HashMap<(usize, usize), (u64, bool)>,
        memory_handler: &MemoryHandler<F>,
        slot_airs: &std::collections::HashSet<(usize, usize)>,
        instance_id: usize,
    ) -> ProofmanResult<u8> {
        if !pctx.gpu {
            return Ok(WITNESS_NOT_STAGED);
        }
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;
        let Some(&(bytes, releasable)) = zone_airs.get(&(airgroup_id, air_id)) else {
            return Ok(WITNESS_NOT_STAGED);
        };
        let trace = pctx.get_air_instance_trace_ptr(instance_id);
        if trace.is_null() || bytes == 0 {
            return Ok(WITNESS_NOT_STAGED);
        }
        // Released only when the commit is certain to read the zone: a slot commit on the zone's GPU
        // does. Only then must the H2D finish before return; a kept buffer lets the commit wait on it.
        let release = releasable && slot_airs.contains(&(airgroup_id, air_id)) && pctx.is_shared_buffer(instance_id);
        let gpu =
            stage_witness_c(pctx.get_device_buffers_ptr(), instance_id as u64, trace as *mut c_void, bytes, release);
        if gpu < 0 {
            return Ok(WITNESS_NOT_STAGED);
        }
        if release {
            memory_handler.to_be_released_buffer(instance_id);
            return Ok(witness_staged_on(WITNESS_STAGED_RELEASED, gpu as u8));
        }
        Ok(witness_staged_on(WITNESS_STAGED, gpu as u8))
    }

    #[allow(clippy::too_many_arguments)]
    fn get_contribution_air(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        roots_contributions: &[[F; 4]],
        values_contributions: &[Mutex<Vec<F>>],
        instance_id: usize,
        aux_trace: &mut [F],
        const_pols: &mut [F],
        slot_commit: Option<&SlotCommitCtx>,
        staged: bool,
        staged_gpu: Option<usize>,
    ) -> ProofmanResult<()> {
        let n_field_elements = 4;
        let (airgroup_id, air_id) = pctx.dctx_get_instance_info(instance_id)?;

        timer_start_debug!(GET_CONTRIBUTION_AIR, "GET_CONTRIBUTION_AIR_{} [{}:{}]", instance_id, airgroup_id, air_id);

        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id)?;
        let setup = sctx.get_setup(airgroup_id, air_id)?;
        let p_setup: *mut c_void = (&setup.p_setup).into();

        let air_values = &pctx.get_air_instance_air_values(airgroup_id, air_id, air_instance_id)?;

        Self::initialize_air_instance(pctx, sctx, instance_id, false, false, None, None)?;

        let mut steps_params = pctx.get_air_instance_params(instance_id, true);

        if !pctx.gpu {
            steps_params.aux_trace = aux_trace.as_mut_ptr() as *mut u8;
            load_const_pols(setup, const_pols);
            steps_params.p_const_pols = const_pols.as_mut_ptr() as *mut u8;
        }

        // Staged with the host buffer handed back, so the inherited pointer may be recycled. Null it;
        // C reads the packed rows from the zone.
        if staged {
            steps_params.trace = std::ptr::null_mut();
        }

        let p_steps_params: *mut u8 = (&steps_params).into();

        // The GPU commits every contribution on a streaming slot, synchronously, on a dedicated
        // stream that stays usable while the first GPU's unified buffer is borrowed by gpu-mops.
        if pctx.gpu {
            let Some(ctx) = slot_commit else {
                return Err(ProofmanError::ProofmanError("GPU contributions need streaming slots".into()));
            };
            Self::try_slot_commit(
                pctx,
                setup,
                ctx,
                instance_id,
                airgroup_id,
                air_id,
                steps_params.trace,
                p_steps_params,
                roots_contributions,
                staged,
                staged_gpu,
            )?;
        } else {
            let custom_commits_fixed_path = match setup.stark_info.custom_commits.iter().find(|c| c.stage_widths[0] > 0)
            {
                Some(c) => pctx.get_custom_commits_fixed_buffer(&c.name, true)?.to_string_lossy().into_owned(),
                None => String::new(),
            };
            commit_witness_c(
                p_setup,
                p_steps_params,
                instance_id as u64,
                airgroup_id as u64,
                air_id as u64,
                roots_contributions[instance_id].as_ptr() as *mut u8,
                pctx.get_device_buffers_ptr(),
                &custom_commits_fixed_path,
            );
        }

        let n_airvalues = setup
            .stark_info
            .airvalues_map
            .as_ref()
            .map(|map| map.iter().filter(|entry| entry.stage == 1).count())
            .unwrap_or(0);

        let size = 2 * n_field_elements + n_airvalues;

        let mut values_hash = vec![F::ZERO; size];

        let vk = setup.get_vk();
        for (i, value) in values_hash.iter_mut().enumerate().take(n_field_elements) {
            *value = F::from_u64(vk[i]);
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

        *values_contributions[instance_id].lock().unwrap() = values_hash;

        timer_stop_and_log_debug!(
            GET_CONTRIBUTION_AIR,
            "GET_CONTRIBUTION_AIR_{} [{}:{}]",
            instance_id,
            airgroup_id,
            air_id
        );
        Ok(())
    }
}

#[cfg(test)]
mod thread_budget_tests {
    use super::*;

    /// Six acquirers wanting 8 from a 24-token budget. Taking tokens one at a time deadlocks here:
    /// each ends up holding a partial set and waiting for the rest, and none reaches its release.
    #[test]
    fn concurrent_multi_token_acquirers_cannot_deadlock() {
        let budget = Arc::new(ThreadBudget::new(24));
        let cancellation = Arc::new(RwLock::new(CancellationInfo::default()));
        let done = Arc::new(AtomicU64::new(0));

        let workers: Vec<_> = (0..6)
            .map(|_| {
                let (budget, cancellation, done) = (budget.clone(), cancellation.clone(), done.clone());
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        let tokens = budget.acquire(8, &cancellation);
                        assert_eq!(tokens.held, 8, "acquire must be all-or-nothing");
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        drop(tokens);
                        done.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        // Generous, but finite: a deadlock here hangs the suite rather than failing it, so bound it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while done.load(Ordering::Relaxed) < 120 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(done.load(Ordering::Relaxed), 120, "acquirers made no progress: deadlocked");
        for w in workers {
            w.join().unwrap();
        }
        assert_eq!(*budget.available.lock().unwrap(), 24, "every token must come back");
    }

    /// A request larger than the whole budget is clamped, not left waiting for tokens that can
    /// never exist.
    #[test]
    fn an_oversized_request_is_clamped_to_capacity() {
        let budget = Arc::new(ThreadBudget::new(4));
        let cancellation = Arc::new(RwLock::new(CancellationInfo::default()));
        let tokens = budget.acquire(8, &cancellation);
        assert_eq!(tokens.held, 4);
        drop(tokens);
        assert_eq!(*budget.available.lock().unwrap(), 4);
    }

    /// A partial early release returns exactly that many; the rest goes back on drop.
    #[test]
    fn a_partial_release_returns_the_rest_on_drop() {
        let budget = Arc::new(ThreadBudget::new(10));
        let cancellation = Arc::new(RwLock::new(CancellationInfo::default()));
        let mut tokens = budget.acquire(6, &cancellation);
        tokens.release(4);
        assert_eq!(*budget.available.lock().unwrap(), 8);
        drop(tokens);
        assert_eq!(*budget.available.lock().unwrap(), 10);
    }
}
