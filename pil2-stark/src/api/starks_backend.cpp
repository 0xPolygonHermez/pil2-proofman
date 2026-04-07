#include "starks_api.hpp"
#include "starks_backend.hpp"
#include <cstdio>

// ============================================================================
// Forward declarations: CPU backend implementations (always available)
// Only functions with real CPU logic are listed here.
// No-op stubs are replaced by nullptr in the cpu_backend table.
// ============================================================================

void calculate_const_tree_cpu(void *pStarkInfo, void *pConstPolsAddress, void *pConstTree, void *unified_buffer_gpu);
void write_custom_commit_cpu(void *root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile);
uint64_t commit_witness_cpu(void *pSetupCtx, void *params, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers);
void verify_constraints_cpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *constraintsInfo, void *d_buffers, uint64_t streamId);
uint64_t gen_proof_cpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers, bool skipRecalculation, uint64_t streamId, char *constPolsPath, char *constTreePath);
void *gen_device_buffers_cpu(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext);
void free_device_buffers_cpu(void *d_buffers);
void load_device_setup_cpu(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packedInfo);
uint64_t gen_recursive_proof_cpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream);
void *gen_recursive_proof_final_cpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers);
void *init_final_snark_prover_cpu(char* zkeyFile);
void free_final_snark_prover_cpu(void *snark_prover);
void gen_final_snark_proof_cpu(void *snark_prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark);

// ============================================================================
// Forward declarations: GPU backend implementations (only in unified build)
// ============================================================================

#ifdef __USE_CUDA__
void init_gpu_setup_gpu(uint64_t maxBitsExt);
void tile_const_pols_gpu(void *pStarkInfo, void *pConstPols, char *constFile, void *pConstTree, char *constTreeFile, void *unified_buffer_gpu);
void prepare_blocks_gpu(uint64_t* pol, uint64_t N, uint64_t nCols, void *unified_buffer_gpu);
void calculate_const_tree_gpu(void *pStarkInfo, void *pConstPolsAddress, void *pConstTree, void *unified_buffer_gpu);
void write_custom_commit_gpu(void *root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile);
uint64_t commit_witness_gpu(void *pSetupCtx, void *params, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers);
uint64_t initialize_instance_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* params_, void *d_buffers_);
void calculate_trace_instance_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *d_buffers, uint64_t streamId);
void verify_constraints_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *constraintsInfo, void *d_buffers, uint64_t streamId);
uint64_t gen_proof_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers, bool skipRecalculation, uint64_t streamId, char *constPolsPath, char *constTreePath);
void get_stream_proofs_gpu(void *d_buffers_);
void get_stream_proofs_non_blocking_gpu(void *d_buffers_);
void get_stream_id_proof_gpu(void *d_buffers_, uint64_t streamId);
uint64_t gen_recursive_proof_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream);
void *gen_recursive_proof_final_gpu(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers);
void calculate_const_tree_fixed_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, char *proofType, void *d_buffers_);
void *gen_device_buffers_gpu(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext);
void free_device_buffers_gpu(void *d_buffers);
void *gen_device_buffers_recursivef_gpu(void *pSetupCtx_, uint64_t proverBufferSize, void *d_commit_buffers, char* verkey);
void free_device_buffers_recursivef_gpu(void *d_buffers);
void load_device_const_pols_gpu(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char* proofType, bool onlyFirstGPU);
void load_device_setup_gpu(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packedInfo);
uint64_t gen_device_streams_gpu(void *d_buffers_, uint64_t n_streams, uint64_t n_recursive_streams, uint64_t maxSizeProverBuffer, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t merkleTreeArity);
void alloc_device_large_buffers_gpu(void *d_buffers_, uint64_t auxTraceArea, uint64_t auxTraceRecursiveArea, uint64_t totalConstPols, uint64_t totalConstPolsAggregation);
void get_instances_ready_gpu(void *d_buffers, int64_t* instances_ready);
void reset_device_streams_gpu(void *d_buffers_);
uint64_t check_device_memory_gpu(uint32_t node_rank, uint32_t node_size);
uint64_t get_num_gpus_gpu();
void *get_unified_buffer_gpu_gpu(void *d_buffers_);
void alloc_fixed_pols_buffer_gpu_gpu(void *d_buffers_);
void free_fixed_pols_buffer_gpu_gpu(void *d_buffers_);
void load_fixed_pols_recursivef_gpu(void *pSetupCtx_, void *pConstTree, void *d_buffers_);
void *init_final_snark_prover_gpu(char* zkeyFile);
void free_final_snark_prover_gpu(void *snark_prover);
void gen_final_snark_proof_gpu(void *snark_prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark);
void pre_allocate_final_snark_prover_gpu(void *snark_prover, void* unified_buffer_gpu);
#endif

// ============================================================================
// Backend tables
// nullptr entries = no-op on CPU (dispatch functions handle the defaults)
// ============================================================================

StarksBackend cpu_backend = {
    .init_gpu_setup = nullptr,
    .tile_const_pols = nullptr,
    .prepare_blocks = nullptr,
    .calculate_const_tree = calculate_const_tree_cpu,
    .write_custom_commit = write_custom_commit_cpu,
    .commit_witness = commit_witness_cpu,
    .initialize_instance = nullptr,               // default: 0
    .calculate_trace_instance = nullptr,
    .verify_constraints = verify_constraints_cpu,
    .gen_proof = gen_proof_cpu,
    .get_stream_proofs = nullptr,
    .get_stream_proofs_non_blocking = nullptr,
    .get_stream_id_proof = nullptr,
    .gen_recursive_proof = gen_recursive_proof_cpu,
    .gen_recursive_proof_final = gen_recursive_proof_final_cpu,
    .calculate_const_tree_fixed = nullptr,
    .gen_device_buffers = gen_device_buffers_cpu,
    .free_device_buffers = free_device_buffers_cpu,
    .gen_device_buffers_recursivef = nullptr,      // default: nullptr
    .free_device_buffers_recursivef = nullptr,
    .load_device_const_pols = nullptr,
    .load_device_setup = load_device_setup_cpu,
    .gen_device_streams = nullptr,                 // default: 1
    .alloc_device_large_buffers = nullptr,
    .get_instances_ready = nullptr,
    .reset_device_streams = nullptr,
    .check_device_memory = nullptr,                // default: 0
    .get_num_gpus = nullptr,                       // default: 1
    .get_unified_buffer_gpu = nullptr,             // default: nullptr
    .alloc_fixed_pols_buffer_gpu = nullptr,
    .free_fixed_pols_buffer_gpu = nullptr,
    .load_fixed_pols_recursivef = nullptr,
    .init_final_snark_prover = init_final_snark_prover_cpu,
    .free_final_snark_prover = free_final_snark_prover_cpu,
    .gen_final_snark_proof = gen_final_snark_proof_cpu,
    .pre_allocate_final_snark_prover = nullptr,
};

#ifdef __USE_CUDA__
StarksBackend gpu_backend = {
    .init_gpu_setup = init_gpu_setup_gpu,
    .tile_const_pols = tile_const_pols_gpu,
    .prepare_blocks = prepare_blocks_gpu,
    .calculate_const_tree = calculate_const_tree_gpu,
    .write_custom_commit = write_custom_commit_gpu,
    .commit_witness = commit_witness_gpu,
    .initialize_instance = initialize_instance_gpu,
    .calculate_trace_instance = calculate_trace_instance_gpu,
    .verify_constraints = verify_constraints_gpu,
    .gen_proof = gen_proof_gpu,
    .get_stream_proofs = get_stream_proofs_gpu,
    .get_stream_proofs_non_blocking = get_stream_proofs_non_blocking_gpu,
    .get_stream_id_proof = get_stream_id_proof_gpu,
    .gen_recursive_proof = gen_recursive_proof_gpu,
    .gen_recursive_proof_final = gen_recursive_proof_final_gpu,
    .calculate_const_tree_fixed = calculate_const_tree_fixed_gpu,
    .gen_device_buffers = gen_device_buffers_gpu,
    .free_device_buffers = free_device_buffers_gpu,
    .gen_device_buffers_recursivef = gen_device_buffers_recursivef_gpu,
    .free_device_buffers_recursivef = free_device_buffers_recursivef_gpu,
    .load_device_const_pols = load_device_const_pols_gpu,
    .load_device_setup = load_device_setup_gpu,
    .gen_device_streams = gen_device_streams_gpu,
    .alloc_device_large_buffers = alloc_device_large_buffers_gpu,
    .get_instances_ready = get_instances_ready_gpu,
    .reset_device_streams = reset_device_streams_gpu,
    .check_device_memory = check_device_memory_gpu,
    .get_num_gpus = get_num_gpus_gpu,
    .get_unified_buffer_gpu = get_unified_buffer_gpu_gpu,
    .alloc_fixed_pols_buffer_gpu = alloc_fixed_pols_buffer_gpu_gpu,
    .free_fixed_pols_buffer_gpu = free_fixed_pols_buffer_gpu_gpu,
    .load_fixed_pols_recursivef = load_fixed_pols_recursivef_gpu,
    .init_final_snark_prover = init_final_snark_prover_gpu,
    .free_final_snark_prover = free_final_snark_prover_gpu,
    .gen_final_snark_proof = gen_final_snark_proof_gpu,
    .pre_allocate_final_snark_prover = pre_allocate_final_snark_prover_gpu,
};
#endif

// Active backend — defaults to CPU
StarksBackend* active_backend = &cpu_backend;

// ============================================================================
// Runtime backend switch
// ============================================================================

void set_gpu_mode(bool use_gpu) {
#ifdef __USE_CUDA__
    active_backend = use_gpu ? &gpu_backend : &cpu_backend;
#else
    if (use_gpu) {
        fprintf(stderr, "Warning: GPU mode requested but library was built without CUDA support. Using CPU backend.\n");
    }
#endif
}

// ============================================================================
// Public API dispatch functions
// For nullptr entries, the dispatcher provides the CPU default behavior.
// ============================================================================

// Const Pols
void init_gpu_setup(uint64_t maxBitsExt) {
    if (active_backend->init_gpu_setup) active_backend->init_gpu_setup(maxBitsExt);
}

void tile_const_pols(void *pStarkInfo, void *pConstPols, char *constFile, void *pConstTree, char *constTreeFile, void *unified_buffer_gpu) {
    if (active_backend->tile_const_pols) active_backend->tile_const_pols(pStarkInfo, pConstPols, constFile, pConstTree, constTreeFile, unified_buffer_gpu);
}

void prepare_blocks(uint64_t* pol, uint64_t N, uint64_t nCols, void *unified_buffer_gpu) {
    if (active_backend->prepare_blocks) active_backend->prepare_blocks(pol, N, nCols, unified_buffer_gpu);
}

void calculate_const_tree(void *pStarkInfo, void *pConstPolsAddress, void *pConstTree, void *unified_buffer_gpu) {
    active_backend->calculate_const_tree(pStarkInfo, pConstPolsAddress, pConstTree, unified_buffer_gpu);
}

// Witness
void write_custom_commit(void *root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile) {
    active_backend->write_custom_commit(root, arity, nBits, nBitsExt, nCols, d_buffers_, buffer, bufferFile);
}

uint64_t commit_witness(void *pSetupCtx, void *params, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers) {
    return active_backend->commit_witness(pSetupCtx, params, instanceId, airgroupId, airId, root, d_buffers);
}

// Constraints
uint64_t initialize_instance(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* params_, void *d_buffers_) {
    return active_backend->initialize_instance ? active_backend->initialize_instance(pSetupCtx_, airgroupId, airId, instanceId, params_, d_buffers_) : 0;
}

void calculate_trace_instance(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *d_buffers, uint64_t streamId) {
    if (active_backend->calculate_trace_instance) active_backend->calculate_trace_instance(pSetupCtx, airgroupId, airId, stepsParams, d_buffers, streamId);
}

void verify_constraints(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *constraintsInfo, void *d_buffers, uint64_t streamId) {
    active_backend->verify_constraints(pSetupCtx, airgroupId, airId, stepsParams, constraintsInfo, d_buffers, streamId);
}

// Proof generation
uint64_t gen_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers, bool skipRecalculation, uint64_t streamId, char *constPolsPath, char *constTreePath) {
    return active_backend->gen_proof(pSetupCtx, airgroupId, airId, instanceId, params, globalChallenge, proofBuffer, proofFile, d_buffers, skipRecalculation, streamId, constPolsPath, constTreePath);
}

void get_stream_proofs(void *d_buffers_) {
    if (active_backend->get_stream_proofs) active_backend->get_stream_proofs(d_buffers_);
}

void get_stream_proofs_non_blocking(void *d_buffers_) {
    if (active_backend->get_stream_proofs_non_blocking) active_backend->get_stream_proofs_non_blocking(d_buffers_);
}

void get_stream_id_proof(void *d_buffers_, uint64_t streamId) {
    if (active_backend->get_stream_id_proof) active_backend->get_stream_id_proof(d_buffers_, streamId);
}

uint64_t gen_recursive_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream) {
    return active_backend->gen_recursive_proof(pSetupCtx, airgroupId, airId, instanceId, witness, aux_trace, pConstPols, pConstTree, pPublicInputs, proofBuffer, proof_file, vadcop, d_buffers, constPolsPath, constTreePath, proofType, force_recursive_stream);
}

void *gen_recursive_proof_final(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers) {
    return active_backend->gen_recursive_proof_final(pSetupCtx, airgroupId, airId, instanceId, witness, aux_trace, pConstPols, pConstTree, pPublicInputs, proof_file, proverBufferSize, d_buffers);
}

void calculate_const_tree_fixed(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, char *proofType, void *d_buffers_) {
    if (active_backend->calculate_const_tree_fixed) active_backend->calculate_const_tree_fixed(pSetupCtx_, airgroupId, airId, proofType, d_buffers_);
}

// Device management
void *gen_device_buffers(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext) {
    return active_backend->gen_device_buffers(node_rank, node_size, numa_nodes, arity, max_n_bits_ext);
}

void free_device_buffers(void *d_buffers) {
    active_backend->free_device_buffers(d_buffers);
}

void *gen_device_buffers_recursivef(void *pSetupCtx_, uint64_t proverBufferSize, void *d_commit_buffers, char* verkey) {
    return active_backend->gen_device_buffers_recursivef ? active_backend->gen_device_buffers_recursivef(pSetupCtx_, proverBufferSize, d_commit_buffers, verkey) : nullptr;
}

void free_device_buffers_recursivef(void *d_buffers) {
    if (active_backend->free_device_buffers_recursivef) active_backend->free_device_buffers_recursivef(d_buffers);
}

void load_device_const_pols(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char* proofType, bool onlyFirstGPU) {
    if (active_backend->load_device_const_pols) active_backend->load_device_const_pols(airgroupId, airId, initial_offset, d_buffers, constFilename, constSize, constTreeFilename, constTreeSize, proofType, onlyFirstGPU);
}

void load_device_setup(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packedInfo) {
    active_backend->load_device_setup(airgroupId, airId, proofType, pSetupCtx_, d_buffers_, verkeyRoot_, packedInfo);
}

uint64_t gen_device_streams(void *d_buffers_, uint64_t n_streams, uint64_t n_recursive_streams, uint64_t maxSizeProverBuffer, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t merkleTreeArity) {
    return active_backend->gen_device_streams ? active_backend->gen_device_streams(d_buffers_, n_streams, n_recursive_streams, maxSizeProverBuffer, maxSizeProverBufferAggregation, maxProofSize, merkleTreeArity) : 1;
}

void alloc_device_large_buffers(void *d_buffers_, uint64_t auxTraceArea, uint64_t auxTraceRecursiveArea, uint64_t totalConstPols, uint64_t totalConstPolsAggregation) {
    if (active_backend->alloc_device_large_buffers) active_backend->alloc_device_large_buffers(d_buffers_, auxTraceArea, auxTraceRecursiveArea, totalConstPols, totalConstPolsAggregation);
}

void get_instances_ready(void *d_buffers, int64_t* instances_ready) {
    if (active_backend->get_instances_ready) active_backend->get_instances_ready(d_buffers, instances_ready);
}

void reset_device_streams(void *d_buffers_) {
    if (active_backend->reset_device_streams) active_backend->reset_device_streams(d_buffers_);
}

uint64_t check_device_memory(uint32_t node_rank, uint32_t node_size) {
    return active_backend->check_device_memory ? active_backend->check_device_memory(node_rank, node_size) : 0;
}

uint64_t get_num_gpus() {
    return active_backend->get_num_gpus ? active_backend->get_num_gpus() : 1;
}

void *get_unified_buffer_gpu(void *d_buffers_) {
    return active_backend->get_unified_buffer_gpu ? active_backend->get_unified_buffer_gpu(d_buffers_) : nullptr;
}

void alloc_fixed_pols_buffer_gpu(void *d_buffers_) {
    if (active_backend->alloc_fixed_pols_buffer_gpu) active_backend->alloc_fixed_pols_buffer_gpu(d_buffers_);
}

void free_fixed_pols_buffer_gpu(void *d_buffers_) {
    if (active_backend->free_fixed_pols_buffer_gpu) active_backend->free_fixed_pols_buffer_gpu(d_buffers_);
}

void load_fixed_pols_recursivef(void *pSetupCtx_, void *pConstTree, void *d_buffers_) {
    if (active_backend->load_fixed_pols_recursivef) active_backend->load_fixed_pols_recursivef(pSetupCtx_, pConstTree, d_buffers_);
}

// Final SNARK
void *init_final_snark_prover(char* zkeyFile) {
    return active_backend->init_final_snark_prover(zkeyFile);
}

void free_final_snark_prover(void *snark_prover) {
    active_backend->free_final_snark_prover(snark_prover);
}

void gen_final_snark_proof(void *snark_prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark) {
    active_backend->gen_final_snark_proof(snark_prover, circomWitnessFinal, proof, publicsSnark);
}

void pre_allocate_final_snark_prover(void *snark_prover, void* unified_buffer_gpu) {
    if (active_backend->pre_allocate_final_snark_prover) active_backend->pre_allocate_final_snark_prover(snark_prover, unified_buffer_gpu);
}
