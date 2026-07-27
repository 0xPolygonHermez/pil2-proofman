#ifndef STARKS_BACKEND_H
#define STARKS_BACKEND_H

#include <stdint.h>
#include <stdbool.h>
#include <atomic>

// Backend interface: function pointer table for CPU/GPU dispatched functions.
// Only functions with distinct CPU and GPU implementations are listed here.
// Shared functions (hints, goldilocks arithmetic, etc.) are called directly.
struct StarksBackend {
    // Const Pols
    void (*init_gpu_setup)(uint64_t maxBitsExt, uint64_t arity);
    void (*tile_const_pols)(void *pStarkInfo, void *pConstPols, char *constFile, void *pConstTree, char *constTreeFile, void *unified_buffer_gpu);
    void (*prepare_blocks)(uint64_t* pol, uint64_t N, uint64_t nCols, void *unified_buffer_gpu);
    void (*calculate_const_tree)(void *pStarkInfo, void *pConstPolsAddress, void *pConstTree, void *unified_buffer_gpu);

    // Witness
    void (*write_custom_commit)(void *root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile);
    uint64_t (*commit_witness)(void *pSetupCtx, void *params, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers, char *customCommitsFixedPath);

    // Constraints
    uint64_t (*initialize_instance)(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* params_, void *d_buffers_, char *customCommitsFixedPath);
    void (*calculate_trace_instance)(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *d_buffers, uint64_t streamId);
    void (*verify_constraints)(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, void *stepsParams, void *constraintsInfo, void *d_buffers, uint64_t streamId);

    // Proof generation
    // customCommitsFixedPath: path to the on-disk custom-commits-fixed file for this
    // (airgroupId, airId). Empty string means no custom commits for this instance.
    uint64_t (*gen_proof)(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers, bool skipRecalculation, uint64_t streamId, char *constPolsPath, char *constTreePath, char *customCommitsFixedPath);
    void (*get_stream_proofs)(void *d_buffers_);
    void (*get_stream_proofs_non_blocking)(void *d_buffers_);
    void (*get_stream_id_proof)(void *d_buffers_, uint64_t streamId);
    uint64_t (*gen_recursive_proof)(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream, char *recurser_id);
    void *(*gen_recursive_proof_final)(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers);
    void (*calculate_const_tree_fixed)(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, char *proofType, void *d_buffers_);

    // Host-memory pinning for direct H2D (GPU backend pins; CPU backend no-ops)
    uint32_t (*register_host_memory)(void *ptr, uint64_t size);
    void (*unregister_host_memory)(void *ptr);
    // Wait for a stream's async commit (incl. the now-unsynced trace H2D) to
    // finish, so the shared trace buffer can be reused. GPU backend only.
    void (*wait_stream_commit_done)(void *d_buffers, uint64_t stream_id);

    // Device management
    void *(*gen_device_buffers)(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext);
    void (*use_packed_trace)(void *d_buffers, bool packed);
    void (*free_device_buffers)(void *d_buffers);
    void *(*gen_device_buffers_recursivef)(void *pSetupCtx_, uint64_t proverBufferSize, void *d_commit_buffers, char* verkey);
    void (*free_device_buffers_recursivef)(void *d_buffers);
    void (*load_device_const_pols)(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char* proofType, bool onlyFirstGPU);
    void (*load_device_setup)(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packedInfo);
    uint64_t (*gen_device_streams)(void *d_buffers_, uint64_t n_streams, uint64_t n_recursive_streams, uint64_t maxSizeProverBuffer, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t merkleTreeArity);
    void (*alloc_device_large_buffers)(void *d_buffers_, uint64_t auxTraceArea, uint64_t auxTraceRecursiveArea, uint64_t totalConstPols, uint64_t totalConstPolsAggregation);
    void (*get_instances_ready)(void *d_buffers, int64_t* instances_ready);
    void (*reset_device_streams)(void *d_buffers_);
    uint64_t (*check_device_memory)(uint32_t node_rank, uint32_t node_size);
    uint64_t (*get_num_gpus)();
    void *(*get_unified_buffer_gpu)(void *d_buffers_);
    uint64_t (*get_unified_buffer_gpu_size)(void *d_buffers_);
    void (*acquire_first_gpu_buffer)(void *d_buffers_);
    void (*release_first_gpu_buffer)(void *d_buffers_);
    uint32_t (*is_first_gpu_buffer_borrowed)(void *d_buffers_);
    uint32_t (*get_first_gpu_id)(void *d_buffers_);
    void *(*get_first_gpu_buffer)(void *d_buffers_);
    void *(*get_unified_buffer_gpu_for_recursivef)(void *d_buffers_, void *d_buffers_recursivef_);
    void (*alloc_fixed_pols_buffer_gpu)(void *d_buffers_);
    void (*free_fixed_pols_buffer_gpu)(void *d_buffers_);
    void (*load_fixed_pols_recursivef)(void *pSetupCtx_, void *pConstTree, void *d_buffers_);

    // Final SNARK
    void *(*init_final_snark_prover)(char* zkeyFile, void* d_buffers_recursivef);
    void (*free_final_snark_prover)(void *snark_prover);
    void (*gen_final_snark_proof)(void *snark_prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark, void* d_buffers_recursivef);
    void (*pre_allocate_final_snark_prover)(void *snark_prover, void* unified_buffer_gpu, void* d_buffers_recursivef);
};

// Active backend pointer — set via set_gpu_mode()
extern std::atomic<StarksBackend*> active_backend;

// CPU backend (always available)
extern StarksBackend cpu_backend;

#ifdef __USE_CUDA__
// GPU backend (only in unified build)
extern StarksBackend gpu_backend;
#endif

#endif
