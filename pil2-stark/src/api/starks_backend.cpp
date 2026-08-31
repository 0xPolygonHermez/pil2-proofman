#include "starks_api.hpp"
#include "starks_backend.hpp"
#include <cstdio>
#include <atomic>

// ============================================================================
// Forward declarations: CPU backend implementations (always available)
// Only functions with real CPU logic are listed here.
// No-op stubs are replaced by nullptr in the cpu_backend table.
// ============================================================================

void calculate_const_tree_cpu(void *pStarkInfo, void *pConstPolsAddress, void *pConstTree, void *unified_buffer_gpu);
void write_custom_commit_cpu(void *root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile);
uint64_t commit_witness_cpu(void *pSetupCtx, void *params, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers, char *customCommitsFixedPath);
void verify_constraints_cpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *constraintsInfo, void *d_buffers, uint64_t streamId);
uint64_t gen_proof_cpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers, bool skipRecalculation, uint64_t streamId, char *constPolsPath, char *constTreePath, char *customCommitsFixedPath, bool selfContained);
void *gen_device_buffers_cpu(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext);
void use_packed_trace_cpu(void *d_buffers_, bool packed);
void register_instruction_table_cpu(void *d_buffers_, uint64_t airgroupId, uint64_t airId, uint64_t *table, uint64_t num_entries, uint64_t words_per_entry);
void free_device_buffers_cpu(void *d_buffers);
void load_device_setup_cpu(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packedInfo, uint64_t *execData, uint64_t execWords);
uint64_t gen_recursive_proof_cpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream, char *recurser_id, uint64_t streamId_);
void *gen_recursive_proof_final_cpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers);
void *init_final_snark_prover_cpu(char* zkeyFile, void* d_buffers_recursivef);
void free_final_snark_prover_cpu(void *snark_prover);
void gen_final_snark_proof_cpu(void *snark_prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark, void* d_buffers_recursivef);

// ============================================================================
// Forward declarations: GPU backend implementations (only in unified build)
// ============================================================================

#ifdef __USE_CUDA__
uint32_t register_host_memory_gpu(void *ptr, uint64_t size);
void unregister_host_memory_gpu(void *ptr);
void wait_trace_h2d_done_gpu(void *d_buffers, uint64_t stream_id);
void init_gpu_setup_gpu(uint64_t arity);
void tile_const_pols_gpu(void *pStarkInfo, void *pConstPols, char *constFile, void *pConstTree, char *constTreeFile, void *unified_buffer_gpu);
void prepare_blocks_gpu(uint64_t* pol, uint64_t N, uint64_t nCols, void *unified_buffer_gpu);
void calculate_const_tree_gpu(void *pStarkInfo, void *pConstPolsAddress, void *pConstTree, void *unified_buffer_gpu);
void write_custom_commit_gpu(void *root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile);
uint64_t commit_witness_gpu(void *pSetupCtx, void *params, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers, char *customCommitsFixedPath);
uint64_t initialize_instance_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* params_, void *d_buffers_, char *customCommitsFixedPath);
void calculate_trace_instance_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *d_buffers, uint64_t streamId);
void verify_constraints_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *constraintsInfo, void *d_buffers, uint64_t streamId);
uint64_t gen_proof_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers, bool skipRecalculation, uint64_t streamId, char *constPolsPath, char *constTreePath, char *customCommitsFixedPath, bool selfContained);
void get_stream_proofs_gpu(void *d_buffers_);
void get_stream_proofs_non_blocking_gpu(void *d_buffers_);
void get_stream_id_proof_gpu(void *d_buffers_, uint64_t streamId);
uint32_t reserve_best_stream_nonblock_gpu(void *d_buffers_, uint64_t airgroupId, uint64_t airId, char *proofType, bool recursive, bool force_recursive);
uint32_t reserve_stream_if_free_gpu(void *d_buffers_, uint32_t streamId, uint64_t airgroupId, uint64_t airId, char *proofType, bool force_recursive);
void release_stream_reservation_gpu(void *d_buffers_, uint32_t streamId);
uint64_t gen_recursive_proof_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream, char *recurser_id, uint64_t streamId_);
void *gen_recursive_proof_final_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers);
void calculate_const_tree_fixed_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, char *proofType, void *d_buffers_);
void *gen_device_buffers_gpu(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext);
void use_packed_trace_gpu(void *d_buffers_, bool packed);
void register_instruction_table_gpu(void *d_buffers_, uint64_t airgroupId, uint64_t airId, uint64_t *table, uint64_t num_entries, uint64_t words_per_entry);
void free_device_buffers_gpu(void *d_buffers);
void *gen_device_buffers_recursivef_gpu(void *pSetupCtx_, uint64_t proverBufferSize, void *d_commit_buffers, char* verkey);
void free_device_buffers_recursivef_gpu(void *d_buffers);
void load_device_const_pols_gpu(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char* proofType, bool onlyFirstGPU, bool alreadyLoaded);
void load_device_setup_gpu(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packedInfo, uint64_t *execData, uint64_t execWords);
uint64_t gen_device_streams_gpu(void *d_buffers_, uint64_t n_streams, uint64_t n_recursive_streams, const uint64_t *auxTraceSizes, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t merkleTreeArity);
void alloc_device_large_buffers_gpu(void *d_buffers_, uint64_t auxTraceRecursiveArea, uint64_t totalConstPols, uint64_t totalConstPolsAggregation, uint64_t unifiedBufferPadArea, uint64_t prefetchRegionArea);
void get_instances_ready_gpu(void *d_buffers, int64_t* instances_ready);
void reset_device_streams_gpu(void *d_buffers_);
uint64_t check_device_memory_gpu(uint32_t node_rank, uint32_t node_size);
uint64_t get_num_gpus_gpu();
void *get_unified_buffer_gpu_gpu(void *d_buffers_);
uint64_t get_unified_buffer_gpu_size_gpu(void *d_buffers_);
void acquire_first_gpu_buffer_gpu(void *d_buffers_);
void release_first_gpu_buffer_gpu(void *d_buffers_);
uint32_t is_first_gpu_buffer_borrowed_gpu(void *d_buffers_);
uint32_t get_first_gpu_id_gpu(void *d_buffers_);
void *get_first_gpu_buffer_gpu(void *d_buffers_);
uint64_t get_const_pols_aggregation_offset_gpu(void *d_buffers_);
uint64_t get_stream_commit_slots_gpu(void *d_buffers_);
uint64_t get_stream_commit_floor_gpu(void *d_buffers_);
uint64_t stream_commit_slot_bytes_gpu(uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, uint64_t wordsPerRow);
void configure_stream_commit_slots_gpu(void *d_buffers_, uint64_t nSlots, uint64_t slotBytes);
void configure_prefetch_zone_gpu(void *d_buffers_, uint64_t witnessBytes, uint64_t fixedTreeBytes, uint64_t packedConstBytes, uint64_t recWitnessBytes);
uint32_t get_prefetch_witness_slots_gpu();
int64_t prefetch_witness_gpu(void *pSetupCtx_, void *d_buffers_, uint64_t instanceId,
                             uint64_t airgroupId, uint64_t airId, void *trace);
int64_t commit_witness_streaming_gpu(void *d_buffers_, uint64_t slotIdx, uint64_t airgroupId, uint64_t airId, void *packed, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, uint64_t wordsPerRow, void *colWidths, void *root);
void stream_commit_pause_gpu();
void *get_unified_buffer_gpu_for_recursivef_gpu(void *d_buffers_, void *d_buffers_recursivef_);
void load_fixed_pols_recursivef_gpu(void *pSetupCtx_, void *pConstTree, void *d_buffers_);
void *init_final_snark_prover_gpu(char* zkeyFile, void* d_buffers_recursivef);
void free_final_snark_prover_gpu(void *snark_prover);
void gen_final_snark_proof_gpu(void *snark_prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark, void* d_buffers_recursivef);
void pre_allocate_final_snark_prover_gpu(void *snark_prover, void* unified_buffer_gpu, void* d_buffers_recursivef);
#endif

// ============================================================================
// Backend tables
// nullptr entries = no-op on CPU (dispatch functions handle the defaults)
// ============================================================================

StarksBackend cpu_backend = []() {
    StarksBackend backend{};
    backend.init_gpu_setup = nullptr;
    backend.tile_const_pols = nullptr;
    backend.prepare_blocks = nullptr;
    backend.calculate_const_tree = calculate_const_tree_cpu;
    backend.write_custom_commit = write_custom_commit_cpu;
    backend.commit_witness = commit_witness_cpu;
    backend.initialize_instance = nullptr;               // default: 0
    backend.calculate_trace_instance = nullptr;
    backend.verify_constraints = verify_constraints_cpu;
    backend.gen_proof = gen_proof_cpu;
    backend.get_stream_proofs = nullptr;
    backend.get_stream_proofs_non_blocking = nullptr;
    backend.get_stream_id_proof = nullptr;
    backend.reserve_best_stream_nonblock = nullptr;       // default: UINT32_MAX (no streams)
    backend.reserve_stream_if_free = nullptr;             // default: 0 (not reserved)
    backend.release_stream_reservation = nullptr;         // default: no-op (nothing to release)
    backend.gen_recursive_proof = gen_recursive_proof_cpu;
    backend.gen_recursive_proof_final = gen_recursive_proof_final_cpu;
    backend.calculate_const_tree_fixed = nullptr;
    backend.register_host_memory = nullptr;               // CPU: no GPU to pin for
    backend.unregister_host_memory = nullptr;
    backend.wait_trace_h2d_done = nullptr;            // CPU: no async stream to wait on
    backend.gen_device_buffers = gen_device_buffers_cpu;
    backend.use_packed_trace = use_packed_trace_cpu;
    backend.register_instruction_table = register_instruction_table_cpu;
    backend.free_device_buffers = free_device_buffers_cpu;
    backend.gen_device_buffers_recursivef = nullptr;      // default: nullptr
    backend.free_device_buffers_recursivef = nullptr;
    backend.load_device_const_pols = nullptr;
    backend.load_device_setup = load_device_setup_cpu;
    backend.gen_device_streams = nullptr;                 // default: 1
    backend.alloc_device_large_buffers = nullptr;
    backend.get_instances_ready = nullptr;
    backend.reset_device_streams = nullptr;
    backend.check_device_memory = nullptr;                // default: 0
    backend.get_num_gpus = nullptr;                       // default: 1
    backend.get_unified_buffer_gpu = nullptr;             // default: nullptr
    backend.get_unified_buffer_gpu_size = nullptr;        // default: 0
    backend.acquire_first_gpu_buffer = nullptr;           // default: no-op
    backend.release_first_gpu_buffer = nullptr;           // default: no-op
    backend.is_first_gpu_buffer_borrowed = nullptr;       // default: 0 (free)
    backend.get_first_gpu_id = nullptr;                   // default: 0
    backend.get_first_gpu_buffer = nullptr;               // default: nullptr
    backend.get_const_pols_aggregation_offset = nullptr;
    backend.get_stream_commit_slots = nullptr;            // default: 0 (disabled)
    backend.get_stream_commit_floor = nullptr;            // default: UINT64_MAX
    backend.stream_commit_slot_bytes = nullptr;           // default: 0 (not committable)
    backend.configure_stream_commit_slots = nullptr;      // default: no-op
    backend.configure_prefetch_zone = nullptr;            // default: no-op
    backend.get_prefetch_witness_slots = nullptr;         // default: 0 (no zone)
    backend.prefetch_witness = nullptr;                   // default: declined
    backend.commit_witness_streaming = nullptr;           // default: error (-1)
    backend.stream_commit_pause = nullptr;                // default: no-op
    backend.get_unified_buffer_gpu_for_recursivef = nullptr;
    backend.load_fixed_pols_recursivef = nullptr;
    backend.init_final_snark_prover = init_final_snark_prover_cpu;
    backend.free_final_snark_prover = free_final_snark_prover_cpu;
    backend.gen_final_snark_proof = gen_final_snark_proof_cpu;
    backend.pre_allocate_final_snark_prover = nullptr;
    return backend;
}();

#ifdef __USE_CUDA__
StarksBackend gpu_backend = []() {
    StarksBackend backend{};
    backend.init_gpu_setup = init_gpu_setup_gpu;
    backend.tile_const_pols = tile_const_pols_gpu;
    backend.prepare_blocks = prepare_blocks_gpu;
    backend.calculate_const_tree = calculate_const_tree_gpu;
    backend.write_custom_commit = write_custom_commit_gpu;
    backend.commit_witness = commit_witness_gpu;
    backend.initialize_instance = initialize_instance_gpu;
    backend.calculate_trace_instance = calculate_trace_instance_gpu;
    backend.verify_constraints = verify_constraints_gpu;
    backend.gen_proof = gen_proof_gpu;
    backend.get_stream_proofs = get_stream_proofs_gpu;
    backend.get_stream_proofs_non_blocking = get_stream_proofs_non_blocking_gpu;
    backend.get_stream_id_proof = get_stream_id_proof_gpu;
    backend.reserve_best_stream_nonblock = reserve_best_stream_nonblock_gpu;
    backend.reserve_stream_if_free = reserve_stream_if_free_gpu;
    backend.release_stream_reservation = release_stream_reservation_gpu;
    backend.gen_recursive_proof = gen_recursive_proof_gpu;
    backend.gen_recursive_proof_final = gen_recursive_proof_final_gpu;
    backend.calculate_const_tree_fixed = calculate_const_tree_fixed_gpu;
    backend.register_host_memory = register_host_memory_gpu;
    backend.unregister_host_memory = unregister_host_memory_gpu;
    backend.wait_trace_h2d_done = wait_trace_h2d_done_gpu;
    backend.gen_device_buffers = gen_device_buffers_gpu;
    backend.use_packed_trace = use_packed_trace_gpu;
    backend.register_instruction_table = register_instruction_table_gpu;
    backend.free_device_buffers = free_device_buffers_gpu;
    backend.gen_device_buffers_recursivef = gen_device_buffers_recursivef_gpu;
    backend.free_device_buffers_recursivef = free_device_buffers_recursivef_gpu;
    backend.load_device_const_pols = load_device_const_pols_gpu;
    backend.load_device_setup = load_device_setup_gpu;
    backend.gen_device_streams = gen_device_streams_gpu;
    backend.alloc_device_large_buffers = alloc_device_large_buffers_gpu;
    backend.get_instances_ready = get_instances_ready_gpu;
    backend.reset_device_streams = reset_device_streams_gpu;
    backend.check_device_memory = check_device_memory_gpu;
    backend.get_num_gpus = get_num_gpus_gpu;
    backend.get_unified_buffer_gpu = get_unified_buffer_gpu_gpu;
    backend.get_unified_buffer_gpu_size = get_unified_buffer_gpu_size_gpu;
    backend.acquire_first_gpu_buffer = acquire_first_gpu_buffer_gpu;
    backend.release_first_gpu_buffer = release_first_gpu_buffer_gpu;
    backend.is_first_gpu_buffer_borrowed = is_first_gpu_buffer_borrowed_gpu;
    backend.get_first_gpu_id = get_first_gpu_id_gpu;
    backend.get_first_gpu_buffer = get_first_gpu_buffer_gpu;
    backend.get_const_pols_aggregation_offset = get_const_pols_aggregation_offset_gpu;
    backend.get_stream_commit_slots = get_stream_commit_slots_gpu;
    backend.get_stream_commit_floor = get_stream_commit_floor_gpu;
    backend.stream_commit_slot_bytes = stream_commit_slot_bytes_gpu;
    backend.configure_stream_commit_slots = configure_stream_commit_slots_gpu;
    backend.configure_prefetch_zone = configure_prefetch_zone_gpu;
    backend.get_prefetch_witness_slots = get_prefetch_witness_slots_gpu;
    backend.prefetch_witness = prefetch_witness_gpu;
    backend.commit_witness_streaming = commit_witness_streaming_gpu;
    backend.stream_commit_pause = stream_commit_pause_gpu;
    backend.get_unified_buffer_gpu_for_recursivef = get_unified_buffer_gpu_for_recursivef_gpu;
    backend.load_fixed_pols_recursivef = load_fixed_pols_recursivef_gpu;
    backend.init_final_snark_prover = init_final_snark_prover_gpu;
    backend.free_final_snark_prover = free_final_snark_prover_gpu;
    backend.gen_final_snark_proof = gen_final_snark_proof_gpu;
    backend.pre_allocate_final_snark_prover = pre_allocate_final_snark_prover_gpu;
    return backend;
}();
#endif

// Active backend — defaults to CPU
std::atomic<StarksBackend*> active_backend(&cpu_backend);

// ============================================================================
// Runtime backend switch
// ============================================================================

bool set_gpu_mode(bool use_gpu) {
#ifdef __USE_CUDA__
    active_backend.store(use_gpu ? &gpu_backend : &cpu_backend, std::memory_order_release);
    return true;
#else
    if (use_gpu) {
        return false;
    }
    return true;
#endif
}

// ============================================================================
// Public API dispatch functions
// For nullptr entries, the dispatcher provides the CPU default behavior.
// ============================================================================

// Const Pols
void init_gpu_setup(uint64_t arity) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->init_gpu_setup) backend->init_gpu_setup(arity);
}

void tile_const_pols(void *pStarkInfo, void *pConstPols, char *constFile, void *pConstTree, char *constTreeFile, void *unified_buffer_gpu) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->tile_const_pols) backend->tile_const_pols(pStarkInfo, pConstPols, constFile, pConstTree, constTreeFile, unified_buffer_gpu);
}

void prepare_blocks(uint64_t* pol, uint64_t N, uint64_t nCols, void *unified_buffer_gpu) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->prepare_blocks) backend->prepare_blocks(pol, N, nCols, unified_buffer_gpu);
}

void calculate_const_tree(void *pStarkInfo, void *pConstPolsAddress, void *pConstTree, void *unified_buffer_gpu) {
    auto backend = active_backend.load(std::memory_order_acquire);
    backend->calculate_const_tree(pStarkInfo, pConstPolsAddress, pConstTree, unified_buffer_gpu);
}

// Witness
void write_custom_commit(void *root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile) {
    auto backend = active_backend.load(std::memory_order_acquire);
    backend->write_custom_commit(root, arity, nBits, nBitsExt, nCols, d_buffers_, buffer, bufferFile);
}

uint64_t commit_witness(void *pSetupCtx, void *params, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers, char *customCommitsFixedPath) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->commit_witness(pSetupCtx, params, instanceId, airgroupId, airId, root, d_buffers, customCommitsFixedPath);
}

// Constraints
uint64_t initialize_instance(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* params_, void *d_buffers_, char *customCommitsFixedPath) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->initialize_instance ? backend->initialize_instance(pSetupCtx_, airgroupId, airId, instanceId, params_, d_buffers_, customCommitsFixedPath) : 0;
}

void calculate_trace_instance(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *d_buffers, uint64_t streamId) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->calculate_trace_instance) backend->calculate_trace_instance(pSetupCtx, airgroupId, airId, stepsParams, d_buffers, streamId);
}

void verify_constraints(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *constraintsInfo, void *d_buffers, uint64_t streamId) {
    auto backend = active_backend.load(std::memory_order_acquire);
    backend->verify_constraints(pSetupCtx, airgroupId, airId, stepsParams, constraintsInfo, d_buffers, streamId);
}

// Proof generation
uint64_t gen_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers, bool skipRecalculation, uint64_t streamId, char *constPolsPath, char *constTreePath, char *customCommitsFixedPath, bool selfContained) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->gen_proof(pSetupCtx, airgroupId, airId, instanceId, params, globalChallenge, proofBuffer, proofFile, d_buffers, skipRecalculation, streamId, constPolsPath, constTreePath, customCommitsFixedPath, selfContained);
}

void get_stream_proofs(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->get_stream_proofs) backend->get_stream_proofs(d_buffers_);
}

void get_stream_proofs_non_blocking(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->get_stream_proofs_non_blocking) backend->get_stream_proofs_non_blocking(d_buffers_);
}

void get_stream_id_proof(void *d_buffers_, uint64_t streamId) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->get_stream_id_proof) backend->get_stream_id_proof(d_buffers_, streamId);
}

// No streams on CPU: report "nothing reserved" so a caller that ever reaches here falls
// back instead of believing it owns stream 0. Today the Rust scheduler is GPU-only.
uint32_t reserve_best_stream_nonblock(void *d_buffers_, uint64_t airgroupId, uint64_t airId, char *proofType, bool recursive, bool force_recursive) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (!backend->reserve_best_stream_nonblock) return UINT32_MAX;
    return backend->reserve_best_stream_nonblock(d_buffers_, airgroupId, airId, proofType, recursive, force_recursive);
}

uint32_t reserve_stream_if_free(void *d_buffers_, uint32_t streamId, uint64_t airgroupId, uint64_t airId, char *proofType, bool force_recursive) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (!backend->reserve_stream_if_free) return 0;
    return backend->reserve_stream_if_free(d_buffers_, streamId, airgroupId, airId, proofType, force_recursive);
}

// Symmetric to the two reserve entries above: hand a reservation back when the caller failed
// before launching. A no-op on CPU, where nothing was ever reserved.
void release_stream_reservation(void *d_buffers_, uint32_t streamId) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (!backend->release_stream_reservation) return;
    backend->release_stream_reservation(d_buffers_, streamId);
}

uint64_t gen_recursive_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream, char *recurser_id, uint64_t streamId_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->gen_recursive_proof(pSetupCtx, airgroupId, airId, instanceId, witness, aux_trace, pConstPols, pConstTree, pPublicInputs, proofBuffer, proof_file, vadcop, d_buffers, constPolsPath, constTreePath, proofType, force_recursive_stream, recurser_id, streamId_);
}

void *gen_recursive_proof_final(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->gen_recursive_proof_final(pSetupCtx, airgroupId, airId, instanceId, witness, aux_trace, pConstPols, pConstTree, pPublicInputs, proof_file, proverBufferSize, d_buffers);
}

void calculate_const_tree_fixed(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, char *proofType, void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->calculate_const_tree_fixed) backend->calculate_const_tree_fixed(pSetupCtx_, airgroupId, airId, proofType, d_buffers_);
}

// Device management
void *gen_device_buffers(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->gen_device_buffers(node_rank, node_size, numa_nodes, arity, max_n_bits_ext);
}

void use_packed_trace(void *d_buffers, bool packed) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->use_packed_trace) backend->use_packed_trace(d_buffers, packed);
}

void register_instruction_table(void *d_buffers, uint64_t airgroupId, uint64_t airId, uint64_t *table, uint64_t num_entries, uint64_t words_per_entry) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->register_instruction_table) backend->register_instruction_table(d_buffers, airgroupId, airId, table, num_entries, words_per_entry);
}

uint32_t register_host_memory(void *ptr, uint64_t size) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->register_host_memory ? backend->register_host_memory(ptr, size) : 0;
}

void unregister_host_memory(void *ptr) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->unregister_host_memory) backend->unregister_host_memory(ptr);
}

void wait_trace_h2d_done(void *d_buffers, uint64_t stream_id) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->wait_trace_h2d_done) backend->wait_trace_h2d_done(d_buffers, stream_id);
}

void free_device_buffers(void *d_buffers) {
    auto backend = active_backend.load(std::memory_order_acquire);
    backend->free_device_buffers(d_buffers);
}

void *gen_device_buffers_recursivef(void *pSetupCtx_, uint64_t proverBufferSize, void *d_commit_buffers, char* verkey) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->gen_device_buffers_recursivef ? backend->gen_device_buffers_recursivef(pSetupCtx_, proverBufferSize, d_commit_buffers, verkey) : nullptr;
}

void free_device_buffers_recursivef(void *d_buffers) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->free_device_buffers_recursivef) backend->free_device_buffers_recursivef(d_buffers);
}

void load_device_const_pols(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char* proofType, bool onlyFirstGPU, bool alreadyLoaded) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->load_device_const_pols) backend->load_device_const_pols(airgroupId, airId, initial_offset, d_buffers, constFilename, constSize, constTreeFilename, constTreeSize, proofType, onlyFirstGPU, alreadyLoaded);
}

void load_device_setup(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packedInfo, uint64_t *execData, uint64_t execWords) {
    auto backend = active_backend.load(std::memory_order_acquire);
    backend->load_device_setup(airgroupId, airId, proofType, pSetupCtx_, d_buffers_, verkeyRoot_, packedInfo, execData, execWords);
}

uint64_t gen_device_streams(void *d_buffers_, uint64_t n_streams, uint64_t n_recursive_streams, const uint64_t *auxTraceSizes, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t merkleTreeArity) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->gen_device_streams ? backend->gen_device_streams(d_buffers_, n_streams, n_recursive_streams, auxTraceSizes, maxSizeProverBufferAggregation, maxProofSize, merkleTreeArity) : 1;
}

void alloc_device_large_buffers(void *d_buffers_, uint64_t auxTraceRecursiveArea, uint64_t totalConstPols, uint64_t totalConstPolsAggregation, uint64_t unifiedBufferPadArea, uint64_t prefetchRegionArea) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->alloc_device_large_buffers) backend->alloc_device_large_buffers(d_buffers_, auxTraceRecursiveArea, totalConstPols, totalConstPolsAggregation, unifiedBufferPadArea, prefetchRegionArea);
}

void get_instances_ready(void *d_buffers, int64_t* instances_ready) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->get_instances_ready) backend->get_instances_ready(d_buffers, instances_ready);
}

void reset_device_streams(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->reset_device_streams) backend->reset_device_streams(d_buffers_);
}

uint64_t check_device_memory(uint32_t node_rank, uint32_t node_size) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->check_device_memory ? backend->check_device_memory(node_rank, node_size) : 0;
}

uint64_t get_num_gpus() {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_num_gpus ? backend->get_num_gpus() : 1;
}

void *get_unified_buffer_gpu(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_unified_buffer_gpu ? backend->get_unified_buffer_gpu(d_buffers_) : nullptr;
}

uint64_t get_unified_buffer_gpu_size(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_unified_buffer_gpu_size ? backend->get_unified_buffer_gpu_size(d_buffers_) : 0;
}

void acquire_first_gpu_buffer(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->acquire_first_gpu_buffer) backend->acquire_first_gpu_buffer(d_buffers_);
}

void release_first_gpu_buffer(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->release_first_gpu_buffer) backend->release_first_gpu_buffer(d_buffers_);
}

uint32_t is_first_gpu_buffer_borrowed(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->is_first_gpu_buffer_borrowed ? backend->is_first_gpu_buffer_borrowed(d_buffers_) : 0;
}

uint32_t get_first_gpu_id(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_first_gpu_id ? backend->get_first_gpu_id(d_buffers_) : 0;
}

// CPU backend (no GPU implementation): UINT64_MAX -- there is no GPU const region to clobber,
// so `used >= offset` must never fire. The GPU implementation instead returns 0 on null
// buffers: GPU mode with missing buffers is an anomaly, and failing toward 0 turns it into a
// redundant reload rather than a silently corrupted proof.
uint64_t get_const_pols_aggregation_offset(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_const_pols_aggregation_offset
               ? backend->get_const_pols_aggregation_offset(d_buffers_)
               : UINT64_MAX;
}

// Streaming-commit slots: 0 / UINT64_MAX on the CPU backend (no slots, no
// GPU region for gpu-mops to stay under), same rationale as the const offset.
uint64_t get_stream_commit_slots(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_stream_commit_slots ? backend->get_stream_commit_slots(d_buffers_) : 0;
}

uint64_t stream_commit_slot_bytes(uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, uint64_t wordsPerRow) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->stream_commit_slot_bytes
               ? backend->stream_commit_slot_bytes(nBits, nBitsExt, nCols, wordsPerRow)
               : 0;
}

void configure_stream_commit_slots(void *d_buffers_, uint64_t nSlots, uint64_t slotBytes) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->configure_stream_commit_slots)
        backend->configure_stream_commit_slots(d_buffers_, nSlots, slotBytes);
}

void configure_prefetch_zone(void *d_buffers_, uint64_t witnessBytes, uint64_t fixedTreeBytes, uint64_t packedConstBytes, uint64_t recWitnessBytes) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->configure_prefetch_zone)
        backend->configure_prefetch_zone(d_buffers_, witnessBytes, fixedTreeBytes, packedConstBytes, recWitnessBytes);
}

uint32_t get_prefetch_witness_slots() {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_prefetch_witness_slots ? backend->get_prefetch_witness_slots() : 0;
}

int64_t prefetch_witness(void *pSetupCtx_, void *d_buffers_, uint64_t instanceId,
                         uint64_t airgroupId, uint64_t airId, void *trace) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->prefetch_witness
               ? backend->prefetch_witness(pSetupCtx_, d_buffers_, instanceId, airgroupId, airId, trace)
               : -1;
}

uint64_t get_stream_commit_floor(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_stream_commit_floor ? backend->get_stream_commit_floor(d_buffers_) : UINT64_MAX;
}

// Quiesce the streaming-commit slots before the gpu-mops final planning
// phase (no-op on CPU / when slots are disabled).
void stream_commit_pause() {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->stream_commit_pause) backend->stream_commit_pause();
}

int64_t commit_witness_streaming(void *d_buffers_, uint64_t slotIdx,
                                 uint64_t airgroupId, uint64_t airId,
                                 void *packed, uint64_t nBits, uint64_t nBitsExt,
                                 uint64_t nCols, uint64_t wordsPerRow,
                                 void *colWidths, void *root) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->commit_witness_streaming
               ? backend->commit_witness_streaming(d_buffers_, slotIdx, airgroupId, airId, packed,
                                                   nBits, nBitsExt, nCols, wordsPerRow, colWidths, root)
               : -1;
}

void *get_first_gpu_buffer(void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_first_gpu_buffer ? backend->get_first_gpu_buffer(d_buffers_) : nullptr;
}

void *get_unified_buffer_gpu_for_recursivef(void *d_buffers_, void *d_buffers_recursivef_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->get_unified_buffer_gpu_for_recursivef ? backend->get_unified_buffer_gpu_for_recursivef(d_buffers_, d_buffers_recursivef_) : nullptr;
}



void load_fixed_pols_recursivef(void *pSetupCtx_, void *pConstTree, void *d_buffers_) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->load_fixed_pols_recursivef) backend->load_fixed_pols_recursivef(pSetupCtx_, pConstTree, d_buffers_);
}

// Final SNARK
void *init_final_snark_prover(char* zkeyFile, void* d_buffers_recursivef) {
    auto backend = active_backend.load(std::memory_order_acquire);
    return backend->init_final_snark_prover(zkeyFile, d_buffers_recursivef);
}

void free_final_snark_prover(void *snark_prover) {
    auto backend = active_backend.load(std::memory_order_acquire);
    backend->free_final_snark_prover(snark_prover);
}

void gen_final_snark_proof(void *snark_prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark, void* d_buffers_recursivef) {
    auto backend = active_backend.load(std::memory_order_acquire);
    backend->gen_final_snark_proof(snark_prover, circomWitnessFinal, proof, publicsSnark, d_buffers_recursivef);
}

void pre_allocate_final_snark_prover(void *snark_prover, void* unified_buffer_gpu, void* d_buffers_recursivef) {
    auto backend = active_backend.load(std::memory_order_acquire);
    if (backend->pre_allocate_final_snark_prover) backend->pre_allocate_final_snark_prover(snark_prover, unified_buffer_gpu, d_buffers_recursivef);
}
