#include "bn128.cuh"
#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#include "starks_api.hpp"
#include "starks_api_internal.cuh"
#include "starks_api_internal.hpp"
#include "pack_columns.hpp"
#include <cstring>
#include <thread>
#include <chrono>
#include <util/gpu_t.cuh>


struct FinalSnarkGPU;
extern void *initFinalSnarkProverGPU(char* zkeyFile, int gpuId);
extern void freeFinalSnarkProverGPU(void *snark_prover);
extern void genFinalSnarkProofGPU(void *proverSnark, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark);
extern void preAllocateFinalSnarkProverGPU(void *snark_prover, void* unified_buffer_gpu);
extern uint64_t getFinalSnarkProverRequiredGpuSizeGPU(void *snark_prover);
extern uint64_t getFinalSnarkProtocolIdGPU(void *snark_prover);
#ifdef __USE_CUDA__
#include "verify_constraints.cuh"
#include "gen_proof.cuh"
#include "poseidon_goldilocks.cuh"
#include "poseidon2_goldilocks.cuh"
#include "blake3_goldilocks.cuh"
#include "hints.cuh"
#include "gen_recursivef_proof.cuh"
#include "poseidon_bn128.cuh"
#include "proofman_sumcheck.cuh"
#include <cuda_runtime.h>
#include <mutex>
#include <algorithm>
#include <map>
#include "stream_commit.cuh"
#include "recursion_trace/gate_bands/gate_bands.hpp"

// gate_bands_gpu.cu
extern "C" void uploadGateBandConstantsGPU(uint64_t family);
extern "C" uint64_t gateBandScratchWordsGPU(uint64_t family);
extern "C" void expandGateBandsGPU(uint64_t *d_trace, uint64_t nCols, uint64_t nRows,
                                   const uint64_t *d_bands, uint64_t nBands, uint64_t aux,
                                   uint64_t family, uint64_t *d_scratch, void *stream);
extern "C" void widenCompactWitnessGPU(uint64_t *d_trace, uint64_t nCols, uint64_t nRows,
                                       const uint64_t *d_compact, uint64_t mapCols, void *stream);

// Process-global handle for stream_commit_pause: the gpu-mops borrower calls
// it from zisk's MO runner thread, which has no DeviceCommitBuffers pointer.
static std::atomic<DeviceCommitBuffers *> gStreamCommitBuffers{nullptr};


uint32_t selectStream(DeviceCommitBuffers* d_buffers, uint64_t airgroupId, uint64_t airId, std::string proofType, bool recursive = false, bool force_recursive = false);
void reserveStream(DeviceCommitBuffers* d_buffers, uint32_t streamId);
void reserveStreamLocked(DeviceCommitBuffers* d_buffers, uint32_t streamId);
static void harvestPipelineStream(DeviceCommitBuffers *d_buffers, uint64_t streamId, bool blocking);
void closeStreamTimer(TimerGPU &timer, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, bool isProve);
void get_proof(DeviceCommitBuffers *d_buffers, uint64_t streamId);
void get_commit_root(DeviceCommitBuffers *d_buffers, uint64_t streamId);


void buildMerkleTreeGPU(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                         uint64_t nCols, uint64_t nRows, Layout layout, cudaStream_t stream)
{
    if (get_hash_family() == HashFamily::Blake3) {
        Blake3GoldilocksGPU::merkletree(arity, d_tree, d_input, nCols, nRows, layout, stream);
    } else if (get_hash_family() == HashFamily::Poseidon1) {
        switch (arity) {
        case 2: PoseidonGoldilocksGPU<8>::merkletree(arity, d_tree, d_input, nCols, nRows, layout, stream);  break;
        case 3: PoseidonGoldilocksGPU<12>::merkletree(arity, d_tree, d_input, nCols, nRows, layout, stream); break;
        case 4: PoseidonGoldilocksGPU<16>::merkletree(arity, d_tree, d_input, nCols, nRows, layout, stream); break;
        default:
            zklog.error("buildMerkleTreeGPU: Poseidon1 supports arity 2, 3 or 4");
            exitProcess();
            exit(-1);
        }
    } else {
        switch (arity) {
        case 2: Poseidon2GoldilocksGPU<8>::merkletree(arity, d_tree, d_input, nCols, nRows, layout, stream);  break;
        case 3: Poseidon2GoldilocksGPU<12>::merkletree(arity, d_tree, d_input, nCols, nRows, layout, stream); break;
        case 4: Poseidon2GoldilocksGPU<16>::merkletree(arity, d_tree, d_input, nCols, nRows, layout, stream); break;
        default:
            zklog.error("buildMerkleTreeGPU: Poseidon2 supports arity 2, 3 or 4");
            exitProcess();
            exit(-1);
        }
    }
}

void runGrindingGPU(uint64_t *d_nonce, uint64_t *d_nonceBlock, const uint64_t *d_in,
                    uint32_t n_bits, cudaStream_t stream)
{
    if (get_hash_family() == HashFamily::Blake3) {
        Blake3GoldilocksGPU::grinding(d_nonce, d_nonceBlock, d_in, n_bits, stream);
    } else if (get_hash_family() == HashFamily::Poseidon1) {
        PoseidonGoldilocksGPU<8>::grinding(d_nonce, d_nonceBlock, d_in, n_bits, stream);
    } else {
        Poseidon2GoldilocksGPUGrinding::grinding(d_nonce, d_nonceBlock, d_in, n_bits, stream);
    }
}

uint32_t register_host_memory_gpu(void *ptr, uint64_t size) {
    if (ptr == nullptr || size == 0) return 0;
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return 1;
}

void unregister_host_memory_gpu(void *ptr) {
    if (ptr == nullptr) return;
    cudaError_t err = cudaHostUnregister(ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
    }
}

// Block until `streamId` has finished reading the caller's host trace buffer, i.e. until the
// trace H2D completes. The copy is no longer synced at copy time (see
// copy_direct_registered_h2d_if_enabled), so the buffer must not be recycled before that or the
// in-flight DMA reads reused bytes. Called from the buffer pool before reusing a shared trace
// buffer. No-op if the stream has no outstanding commit (status != 2).
//
// This waits on trace_copy_event, not end_event: the commit's LDE/Merkle work reads the device
// copy, not the host buffer, so gating recycling on the whole commit held pool buffers for the
// entire GPU pipeline and left witness threads queueing in take_buffer for them.
void wait_trace_h2d_done_gpu(void *d_buffers_, uint64_t streamId) {
    if (d_buffers_ == nullptr) return;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    // Guard the C-ABI surface: an out-of-range streamId would index streamsData OOB and segfault.
    if (streamId >= d_buffers->n_total_streams) return;
    cudaSetDevice(d_buffers->streamsData[streamId].gpuId);
    if (d_buffers->streamsData[streamId].status == 2) {
        CHECKCUDAERR(cudaEventSynchronize(d_buffers->streamsData[streamId].trace_copy_event));
    }
}

void get_instances_ready_gpu(void *d_buffers_, int64_t* instances_ready) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        // Resident witness = status 3 AND witnessResident. Reads only scalars: reading the
        // std::string proofType here would race concurrent writes on other streams.
        StreamData &sd = d_buffers->streamsData[i];
        instances_ready[i] = (sd.status == 3 && sd.witnessResident) ? sd.instanceId : -1;
    }
}

void *gen_device_buffers_gpu(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext)
{
    int32_t numa_node = (numa_nodes != nullptr && node_rank < node_size) ? numa_nodes[node_rank] : -1;

    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error getting device count: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    if (deviceCount < (int)node_size) {
        zklog.error("GPU sharing not supported: " + std::to_string(node_size) + 
                   " processes but only " + std::to_string(deviceCount) + " GPUs available");
        exit(1);
    }

    if (deviceCount % node_size != 0) {
        zklog.warning("Uneven GPU distribution: " + std::to_string(deviceCount) + 
                     " GPUs across " + std::to_string(node_size) + " processes");
    }

    // Helper lambda to get GPU NUMA node
    auto get_gpu_numa_node = [](int gpu_id) -> int {
        int numa_node = -1;
#if CUDART_VERSION >= 12000
        // CUDA 12+: cudaDevAttrHostNumaId
        cudaError_t err = cudaDeviceGetAttribute(&numa_node, cudaDevAttrHostNumaId, gpu_id);
#elif CUDART_VERSION >= 10020
        // CUDA 10.2-11.x: cudaDevAttrNumaNodeId
        cudaError_t err = cudaDeviceGetAttribute(&numa_node, cudaDevAttrNumaNodeId, gpu_id);
#else
        // Older CUDA: no NUMA support
        cudaError_t err = cudaErrorNotSupported;
#endif
        if (err != cudaSuccess || numa_node < 0) {
            return -1;
        }
        return numa_node;
    };

    // Build GPU NUMA affinity map
    // If no process NUMA info available, put all GPUs in bucket -1 for simple distribution
    std::vector<int> gpu_numa_nodes(deviceCount);
    std::map<int, std::vector<int>> gpus_by_numa;
    
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        int gpu_numa = (numa_nodes != nullptr) ? get_gpu_numa_node(gpu) : -1;
        gpu_numa_nodes[gpu] = gpu_numa;
        gpus_by_numa[gpu_numa].push_back(gpu);
    }

    // Calculate how many GPUs each process should get
    uint32_t base_gpus_per_process = deviceCount / node_size;
    uint32_t remainder = deviceCount % node_size;
    uint32_t my_gpu_count = base_gpus_per_process + (node_rank < remainder ? 1 : 0);
    
    // Map: rank -> assigned GPUs
    std::map<uint32_t, std::vector<int>> rank_to_gpus;
    
    // First pass: each rank picks from its own NUMA node (or -1 if unknown)
    for (uint32_t r = 0; r < node_size; r++) {
        uint32_t r_gpu_count = base_gpus_per_process + (r < remainder ? 1 : 0);
        int r_numa = (numa_nodes != nullptr) ? numa_nodes[r] : -1;
        
        while (rank_to_gpus[r].size() < r_gpu_count && !gpus_by_numa[r_numa].empty()) {
            int gpu = gpus_by_numa[r_numa].back();
            gpus_by_numa[r_numa].pop_back();
            rank_to_gpus[r].push_back(gpu);
        }
    }
    
    // Collect remaining GPUs into a pool (deterministic order - std::map iterates by key)
    std::vector<int> remaining_gpus;
    for (auto& kv : gpus_by_numa) {
        for (int gpu : kv.second) {
            remaining_gpus.push_back(gpu);
        }
    }
    
    // Second pass: fill ranks that didn't get enough GPUs
    size_t remaining_idx = 0;
    for (uint32_t r = 0; r < node_size; r++) {
        uint32_t r_gpu_count = base_gpus_per_process + (r < remainder ? 1 : 0);
        while (rank_to_gpus[r].size() < r_gpu_count && remaining_idx < remaining_gpus.size()) {
            rank_to_gpus[r].push_back(remaining_gpus[remaining_idx++]);
        }
    }
    
    // Extract my assignment
    std::vector<uint32_t> assigned_gpus;
    for (int gpu : rank_to_gpus[node_rank]) {
        assigned_gpus.push_back(static_cast<uint32_t>(gpu));
    }
    
    // Verify we got the right number of GPUs (balance guarantee)
    if(assigned_gpus.size() != my_gpu_count){
        zklog.error("GPU assignment error: rank " + std::to_string(node_rank) + 
                   " expected " + std::to_string(my_gpu_count) + " GPUs but got " + 
                   std::to_string(assigned_gpus.size()));
        exit(1);
    }
    
    // Print GPU assignment for this rank
    {
        std::string gpu_info;
        for (size_t i = 0; i < assigned_gpus.size(); i++) {
            if (i > 0) gpu_info += " ";
            gpu_info += std::to_string(assigned_gpus[i]) + "(numa" + std::to_string(gpu_numa_nodes[assigned_gpus[i]]) + ")";
        }
        zklog.info("GPU assignment: node_rank=" + std::to_string(node_rank) + 
                  " numa=" + std::to_string(numa_node) + 
                  " GPUs=[" + gpu_info + "]");
    }
    
    // Warn only if NUMA affinity couldn't be fully satisfied    
    uint32_t numa_local_count = 0;
    for (auto g : assigned_gpus) {
        if (gpu_numa_nodes[g] == numa_node && numa_node >= 0) numa_local_count++;
    }
    if (numa_local_count < my_gpu_count) {
        std::string gpu_list;
        for (size_t i = 0; i < assigned_gpus.size(); i++) {
            if (i > 0) gpu_list += " ";
            auto g = assigned_gpus[i];
            gpu_list += std::to_string(g);
            if (gpu_numa_nodes[g] == numa_node && numa_node >= 0) {
                gpu_list += "(local)";
            } else {
                gpu_list += "(numa" + std::to_string(gpu_numa_nodes[g]) + ")";
            }
        }
        zklog.warning("GPU NUMA affinity: node_rank=" + std::to_string(node_rank) + 
                        " on NUMA " + std::to_string(numa_node) + " got " + 
                        std::to_string(numa_local_count) + "/" + std::to_string(my_gpu_count) + 
                        " NUMA-local GPUs: [" + gpu_list + "]");
    }
    
    
    uint32_t n_gpus = assigned_gpus.size();
    assert(n_gpus > 0 && n_gpus < 32);
    
    uint32_t my_gpu_ids[32];
    for (uint32_t i = 0; i < n_gpus; i++) {
        my_gpu_ids[i] = assigned_gpus[i];
    }

    // Scope sppark's GPU registry to this rank's devices before it probes all GPUs.
    {
        int ords[32];
        uint32_t n = n_gpus < 32 ? n_gpus : 32;
        for (uint32_t i = 0; i < n; i++) ords[i] = (int)my_gpu_ids[i];
        sppark_set_visible_devices(ords, (int)n);
    }

    // Create primary contexts only on this rank's assigned GPUs; never implicitly touch GPU 0.
    // Non-owning ranks would each create a ~300 MB context there and starve GPU 0's owner, so
    // we end on an assigned GPU rather than syncing back to GPU 0.
    for (uint32_t i = 0; i < n_gpus; i++) {
        cudaSetDevice(my_gpu_ids[i]);
        cudaFree(0);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(my_gpu_ids[0]);

    // Initialize small GPU constants for BOTH Poseidon families unconditionally.
    switch(arity){
        case 2:
            PoseidonGoldilocksGPU<8>::initConstants(my_gpu_ids, n_gpus);
            Poseidon2GoldilocksGPU<8>::initConstants(my_gpu_ids, n_gpus);
            break;
        case 3:
            PoseidonGoldilocksGPU<12>::initConstants(my_gpu_ids, n_gpus);
            Poseidon2GoldilocksGPU<12>::initConstants(my_gpu_ids, n_gpus);
            break;
        case 4:
            PoseidonGoldilocksGPU<16>::initConstants(my_gpu_ids, n_gpus);
            Poseidon2GoldilocksGPU<16>::initConstants(my_gpu_ids, n_gpus);
            break;
        default:
            zklog.error("Unsupported merkle tree arity. Supported arities are 2, 3 and 4.");
            exit(1);
    }
    PoseidonGoldilocksGPUGrinding::initConstants(my_gpu_ids, n_gpus);
    Poseidon2GoldilocksGPUGrinding::initConstants(my_gpu_ids, n_gpus);
    TranscriptGL_GPU::init_const(my_gpu_ids, n_gpus, arity);


    cudaDeviceSynchronize();

    // Create and initialize DeviceCommitBuffers structure
    DeviceCommitBuffers *d_buffers = new DeviceCommitBuffers();
    d_buffers->n_gpus = n_gpus;
    d_buffers->gpus_g2l = (uint32_t *)malloc(deviceCount * sizeof(uint32_t));
    d_buffers->my_gpu_ids = (uint32_t *)malloc(d_buffers->n_gpus * sizeof(uint32_t));
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        d_buffers->my_gpu_ids[i] = my_gpu_ids[i];
        d_buffers->gpus_g2l[d_buffers->my_gpu_ids[i]] = i;
    }
    d_buffers->d_aux_trace = (gl64_t ***)malloc(d_buffers->n_gpus * sizeof(gl64_t**));
    d_buffers->d_aux_traceAggregation = (gl64_t ***)malloc(d_buffers->n_gpus * sizeof(gl64_t**));
    d_buffers->d_constPols = (gl64_t **)malloc(d_buffers->n_gpus * sizeof(gl64_t*));
    d_buffers->d_constPolsAggregation = (gl64_t **)malloc(d_buffers->n_gpus * sizeof(gl64_t*));
    d_buffers->pinned_buffer = (Goldilocks::Element **)malloc(d_buffers->n_gpus * sizeof(Goldilocks::Element *));
    d_buffers->pinned_buffer_extra = (Goldilocks::Element **)malloc(d_buffers->n_gpus * sizeof(Goldilocks::Element *));
    d_buffers->pinned_copy_done = (cudaEvent_t (*)[2])malloc(d_buffers->n_gpus * sizeof(cudaEvent_t[2]));
    d_buffers->gpuMemoryBuffer = (gl64_t **)malloc(d_buffers->n_gpus * sizeof(gl64_t*));
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        d_buffers->gpuMemoryBuffer[i] = nullptr;
    }
    
    // Allocate mutex array using placement new
    d_buffers->mutex_pinned = (std::mutex*)malloc(d_buffers->n_gpus * sizeof(std::mutex));
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        new (&d_buffers->mutex_pinned[i]) std::mutex();
    }
    
    return (void *)d_buffers;
}

void use_packed_trace_gpu(void *d_buffers_, bool packed) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    d_buffers->packedTrace = packed;
}

// Upload (per program) the instruction table for an indexed air onto every local GPU's
// AirInstanceInfo. Non-indexed instances (d_col_source == nullptr) are skipped.
void register_instruction_table_gpu(void *d_buffers_, uint64_t airgroupId, uint64_t airId,
                                    uint64_t *table, uint64_t num_entries, uint64_t words_per_entry) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};
    uint64_t uploaded = 0;
    auto it = d_buffers->air_instances.find(key);
    if (it != d_buffers->air_instances.end()) {
        for (auto &per_proof_type : it->second) {
            std::vector<AirInstanceInfo *> &instances = per_proof_type.second;
            for (int i = 0; i < d_buffers->n_gpus && i < (int)instances.size(); ++i) {
                AirInstanceInfo *aii = instances[i];
                if (aii == nullptr || aii->d_col_source == nullptr) continue; // indexed airs only
                cudaSetDevice(d_buffers->my_gpu_ids[i]);
                aii->set_instruction_table(table, num_entries, words_per_entry);
                uploaded++;
            }
        }
    }
    // Nothing uploaded means the air was not set up yet (register before load_device_setups) or it
    // carries no indexed descriptor. Either way the later unpack aborts, so say so at the actual
    // cause rather than letting it fail silently here.
    if (uploaded == 0) {
        zklog.warning("register_instruction_table: air (" + std::to_string(airgroupId) + "," +
                      std::to_string(airId) + ") matched no indexed AirInstanceInfo; the table was "
                      "NOT uploaded (register after load_device_setups, and only for indexed airs)");
    }
}

// Non-recursive areas are sized per stream from d_buffers->aux_trace_sizes, not one uniform size.
void alloc_device_large_buffers_gpu(void *d_buffers_, uint64_t auxTraceRecursiveArea, uint64_t totalConstPols, uint64_t totalConstPolsAggregation, uint64_t unifiedBufferPadArea, uint64_t prefetchRegionArea) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint64_t constPolsSize = totalConstPols * sizeof(Goldilocks::Element);
    uint64_t constPolsAggregationSize = totalConstPolsAggregation * sizeof(Goldilocks::Element);
    uint64_t auxTraceRecursiveSize = auxTraceRecursiveArea * sizeof(Goldilocks::Element);
    uint64_t unifiedBufferPadSize = unifiedBufferPadArea * sizeof(Goldilocks::Element);
    uint64_t prefetchRegionSize = prefetchRegionArea * sizeof(Goldilocks::Element);

    uint64_t totalAuxTraceArea = 0;
    for (uint32_t j = 0; j < d_buffers->n_streams; ++j) {
        totalAuxTraceArea += d_buffers->aux_trace_sizes[j];
    }
    uint64_t totalAuxTraceSize = totalAuxTraceArea * sizeof(Goldilocks::Element);
    // Phase-B aliases live inside the basic stream's buffer: no area of their own.
    uint64_t totalAuxTraceRecursiveSize =
        d_buffers->phaseBAliased ? 0 : d_buffers->n_recursive_streams * auxTraceRecursiveSize;
    if (d_buffers->phaseBAliased &&
        (d_buffers->n_streams != 1 || d_buffers->n_recursive_streams != 2 ||
         d_buffers->aux_trace_sizes[0] < 2 * auxTraceRecursiveArea)) {
        zklog.error("Phase B needs one basic stream holding two recursive classes (" +
                    std::to_string(d_buffers->n_streams) + " basic, " +
                    std::to_string(d_buffers->n_recursive_streams) + " recursive, basic " +
                    std::to_string(d_buffers->aux_trace_sizes[0]) + " vs 2 x " +
                    std::to_string(auxTraceRecursiveArea) + " elements)");
        exitProcess();
    }

    // Mops-floor pad (see DeviceCommitBuffers::MOPS_FLOOR_BYTES): raise the region BELOW the
    // const pols to the floor so the mem-ops planner's borrow fits (its fixed regions plus the
    // streaming-commit slots it must stay under; short of it the planner falls back to CPU mops,
    // loudly). Clamped to the memory actually free on the first GPU minus the post-allocation margin.
    uint64_t mopsFloorPadSize = 0;
    {
        uint64_t belowConsts = totalAuxTraceSize + totalAuxTraceRecursiveSize + prefetchRegionSize;
        if (DeviceCommitBuffers::MOPS_FLOOR_BYTES > belowConsts) {
            mopsFloorPadSize = DeviceCommitBuffers::MOPS_FLOOR_BYTES - belowConsts;
            size_t freeMem = 0, totalMem = 0;
            cudaSetDevice(d_buffers->my_gpu_ids[0]);
            CHECKCUDAERR(cudaMemGetInfo(&freeMem, &totalMem));
            uint64_t base = constPolsAggregationSize + constPolsSize + unifiedBufferPadSize + belowConsts;
            uint64_t room = (freeMem > base + DeviceCommitBuffers::POST_ALLOC_HEADROOM_BYTES)
                                ? freeMem - base - DeviceCommitBuffers::POST_ALLOC_HEADROOM_BYTES : 0;
            if (mopsFloorPadSize > room) mopsFloorPadSize = room;
            mopsFloorPadSize &= ~((1ull << 20) - 1);  // MiB-align, keeps offsets tidy
        }
    }
    d_buffers->mopsFloorPadBytes = mopsFloorPadSize;
    uint64_t mopsFloorPadArea = mopsFloorPadSize / sizeof(Goldilocks::Element);

    uint64_t totalGpuMemoryPerGpu = constPolsAggregationSize + constPolsSize + unifiedBufferPadSize +
                                     totalAuxTraceSize + totalAuxTraceRecursiveSize + prefetchRegionSize + mopsFloorPadSize;

    uint64_t totalPinnedMemoryPerGpu = 2 * d_buffers->pinned_size * sizeof(Goldilocks::Element);

    zklog.info("Memory allocation per GPU:");
    zklog.info("  - Constant polynomials: " + std::to_string(constPolsSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Constant polynomials aggregation: " + std::to_string(constPolsAggregationSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Snark floor padding: " + std::to_string(unifiedBufferPadSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Prefetch region: " + std::to_string(prefetchRegionSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Mops floor padding: " + std::to_string(mopsFloorPadSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    // Collapse the per-stream sizes back into "count x size" classes; they arrive grouped.
    std::string auxTraceClasses;
    for (uint32_t j = 0; j < d_buffers->n_streams; ) {
        uint32_t k = j;
        while (k < d_buffers->n_streams && d_buffers->aux_trace_sizes[k] == d_buffers->aux_trace_sizes[j]) ++k;
        if (j != 0) auxTraceClasses += " + ";
        auxTraceClasses += std::to_string(k - j) + " x " +
            std::to_string(d_buffers->aux_trace_sizes[j] * sizeof(Goldilocks::Element) / (1024.0 * 1024.0 * 1024.0)) +
            " GB";
        j = k;
    }
    zklog.info("  - Auxiliary trace (" + std::to_string(d_buffers->n_streams) + " streams): " + std::to_string(totalAuxTraceSize / (1024.0 * 1024.0 * 1024.0)) + " GB [" + auxTraceClasses + "]");
    zklog.info("  - Auxiliary trace recursive (" + std::to_string(d_buffers->n_recursive_streams) + " streams): " + std::to_string(totalAuxTraceRecursiveSize / (1024.0 * 1024.0 * 1024.0)) + " GB" +
               (d_buffers->phaseBAliased ? " (phase B: aliased over the basic stream)" : ""));
    zklog.info("  - Unified buffer per GPU: " + std::to_string(totalGpuMemoryPerGpu / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Pinned host memory per GPU: " + std::to_string(totalPinnedMemoryPerGpu / (1024.0 * 1024.0 * 1024.0)) + " GB");

    d_buffers->constPolsSize = constPolsSize;
    d_buffers->unifiedBufferSize = totalGpuMemoryPerGpu;
    d_buffers->firstGpuBufferBorrowed.store(0, std::memory_order_relaxed);
   
    gStreamCommitBuffers.store(d_buffers, std::memory_order_release);

    d_buffers->auxTraceTotalBytes = totalAuxTraceSize;
    d_buffers->auxTraceRecursiveBytes = auxTraceRecursiveSize;

    // Allocate large GPU buffers with a single malloc per GPU
    for (int i = 0; i < d_buffers->n_gpus; i++) {
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        
        // Check available GPU memory
        size_t freeMem, totalMem;
        CHECKCUDAERR(cudaMemGetInfo(&freeMem, &totalMem));
        zklog.info("GPU " + std::to_string(d_buffers->my_gpu_ids[i]) + ": Available memory: " + 
                   std::to_string(freeMem / (1024.0 * 1024.0 * 1024.0)) + " GB / " + 
                   std::to_string(totalMem / (1024.0 * 1024.0 * 1024.0)) + " GB");
        
        if (freeMem < totalGpuMemoryPerGpu) {
            zklog.error("GPU " + std::to_string(d_buffers->my_gpu_ids[i]) +
                       ": Insufficient memory. Need " + std::to_string(totalGpuMemoryPerGpu / (1024.0 * 1024.0 * 1024.0)) +
                       " GB but only " + std::to_string(freeMem / (1024.0 * 1024.0 * 1024.0)) + " GB available");
            exit(1);
        }

        // Allocate one large contiguous block of GPU memory (unified buffer)
        gl64_t *gpuMemoryBlock;
        CHECKCUDAERR(cudaMalloc(&gpuMemoryBlock, totalGpuMemoryPerGpu));
        d_buffers->gpuMemoryBuffer[i] = gpuMemoryBlock;  // Store the base pointer

        zklog.info("GPU " + std::to_string(d_buffers->my_gpu_ids[i]) +
                   ": Allocated " + std::to_string(totalGpuMemoryPerGpu / (1024.0 * 1024.0 * 1024.0)) +
                   " GB unified (const pols and floor padding included)");

        // Set up pointers to different sections of the memory block
        uint64_t offset = 0;

        // Auxiliary trace buffers (non-recursive), one size class each
        for (int j = 0; j < d_buffers->n_streams; ++j) {
            d_buffers->d_aux_trace[i][j] = gpuMemoryBlock + offset;
            offset += d_buffers->aux_trace_sizes[j];
        }

        // Auxiliary trace buffers (recursive). Phase B: [0..A) and [A..2A) over the basic
        // stream's buffer (validated above), usable only while that stream is idle.
        for (int j = 0; j < d_buffers->n_recursive_streams; ++j) {
            if (d_buffers->phaseBAliased) {
                d_buffers->d_aux_traceAggregation[i][j] = gpuMemoryBlock + (uint64_t)j * auxTraceRecursiveArea;
            } else {
                d_buffers->d_aux_traceAggregation[i][j] = gpuMemoryBlock + offset;
                offset += auxTraceRecursiveArea;
            }
        }

        // Prefetch region lives INSIDE the unified buffer
        if (i == 0) {
            d_buffers->prefetchRegionBase = prefetchRegionArea > 0 ? gpuMemoryBlock + offset : nullptr;
            d_buffers->prefetchRegionBytes = prefetchRegionSize;
        }
        offset += prefetchRegionArea;

        // Mops-floor pad: nothing lives here, it only pushes the const pols up.
        offset += mopsFloorPadArea;

        // Constant polynomials aggregation
        d_buffers->d_constPolsAggregation[i] = gpuMemoryBlock + offset;
        offset += totalConstPolsAggregation;

        // Basic const pols, above the aggregation region. A borrower that reaches them has
        // already crossed the aggregation offset, so the existing used>=aggOffset trigger
        // (report_first_gpu_buffer_usage) schedules the reload that covers both regions.
        d_buffers->d_constPols[i] = gpuMemoryBlock + offset;
        offset += totalConstPols;

        // Allocate pinned host buffers separately (one block per buffer type)
        CHECKCUDAERR(cudaMallocHost(&d_buffers->pinned_buffer[i], d_buffers->pinned_size * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMallocHost(&d_buffers->pinned_buffer_extra[i], d_buffers->pinned_size * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->pinned_copy_done[i][0], cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->pinned_copy_done[i][1], cudaEventDisableTiming));

        // Verify we used exactly the amount we calculated 
        if (offset + unifiedBufferPadArea != totalGpuMemoryPerGpu / sizeof(Goldilocks::Element)) {
            zklog.error("GPU " + std::to_string(d_buffers->my_gpu_ids[i]) +
                       ": Memory offset mismatch! Expected " + std::to_string(totalGpuMemoryPerGpu / sizeof(Goldilocks::Element)) +
                       " but got " + std::to_string(offset + unifiedBufferPadArea) + " elements");
            exit(1);
        }
    }

    zklog.info("All GPU memory allocations successful");
}

// `auxTraceSizes`: field elements per non-recursive stream, largest class first (plan_stream_layout).
uint64_t gen_device_streams_gpu(void *d_buffers_, uint64_t n_streams, uint64_t n_recursive_streams, const uint64_t *auxTraceSizes, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t merkleTreeArity) {

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    d_buffers->n_streams = n_streams;
    d_buffers->n_recursive_streams = n_recursive_streams;
    d_buffers->n_total_streams = d_buffers->n_gpus * (d_buffers->n_streams + d_buffers->n_recursive_streams);

    // Retained for the whole run: the memory carve below and every stream selection read it.
    free(d_buffers->aux_trace_sizes);
    d_buffers->aux_trace_sizes = (uint64_t *)malloc(n_streams * sizeof(uint64_t));
    for (uint64_t j = 0; j < n_streams; ++j) {
        d_buffers->aux_trace_sizes[j] = auxTraceSizes[j];
    }

    // Allocate d_aux_trace arrays now that we know stream counts
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        d_buffers->d_aux_trace[i] = (gl64_t **)malloc(n_streams * sizeof(gl64_t*));
        d_buffers->d_aux_traceAggregation[i] = (gl64_t **)malloc(n_recursive_streams * sizeof(gl64_t*));
    }
    d_buffers->max_size_proof = maxProofSize;

    if (d_buffers->streamsData != nullptr) {
        for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
            d_buffers->streamsData[i].free();
        }
        delete[] d_buffers->streamsData;
    }
    d_buffers->streamsData = new StreamData[d_buffers->n_total_streams];

    for(uint64_t i=0; i< d_buffers->n_gpus; ++i){
        uint64_t gpu_stream_start = i * (d_buffers->n_streams + d_buffers->n_recursive_streams);

        for (uint64_t j = 0; j < d_buffers->n_streams; j++) {
            StreamData &sd = d_buffers->streamsData[gpu_stream_start + j];
            sd.initialize(maxProofSize, d_buffers->my_gpu_ids[i], j, false, merkleTreeArity);
            sd.auxTraceCapacity = auxTraceSizes[j];
        }

        for (uint64_t j = 0; j < d_buffers->n_recursive_streams; j++) {
            StreamData &sd = d_buffers->streamsData[gpu_stream_start + d_buffers->n_streams + j];
            sd.initialize(maxProofSize, d_buffers->my_gpu_ids[i], j, true, merkleTreeArity);
            sd.auxTraceCapacity = maxSizeProverBufferAggregation;
        }
    }

    return d_buffers->n_gpus;
}

// Fence all device work before a terminal free: wait (bounded) for the entries counted by
// InFlightScope to leave, then fence the work they queued. Bounded so a wedged stream degrades to a
// diagnostic, not an unbounded hang. `cancelled` is raised for entries that opt into checking it
// (see InFlightScope::cancelled); today none do, so this relies on the refcount wait alone.
static void wait_device_idle_before_teardown(DeviceCommitBuffers *d_buffers) {
    d_buffers->cancelled.store(true, std::memory_order_seq_cst);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (d_buffers->device_active.load(std::memory_order_seq_cst) > 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    const int64_t still_active = d_buffers->device_active.load(std::memory_order_acquire);
    if (still_active > 0) {
        printf("[teardown] WARNING: %ld device operation(s) still in flight after 30s; freeing anyway\n",
               (long)still_active);
        fflush(stdout);
    }

    // Fence work queued by entries that already returned, before freeing the memory it references.
    // Bounded per-stream so a wedged stream degrades to a diagnostic, not an unbounded hang. All
    // proof work runs on the named streams, so polling those covers it.
    if (d_buffers->streamsData != nullptr) {
        bool any_timed_out = false;
        for (uint32_t s = 0; s < d_buffers->n_total_streams; ++s) {
            cudaSetDevice(d_buffers->streamsData[s].gpuId);
            cudaStream_t stream = d_buffers->streamsData[s].stream;
            // Per-stream deadline so one wedged stream can't starve the fencing of the rest: a
            // shared deadline let a stuck stream burn the whole budget, freeing later streams still in use.
            const auto stream_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (cudaStreamQuery(stream) == cudaErrorNotReady &&
                   std::chrono::steady_clock::now() < stream_deadline) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            if (d_buffers->pipelineMode) harvestPipelineStream(d_buffers, s, true);
            if (std::chrono::steady_clock::now() >= stream_deadline) {
                any_timed_out = true;
            }
            cudaGetLastError();  // clear the sticky status left by the last query
        }
        if (any_timed_out) {
            printf("[teardown] WARNING: a device stream did not drain within 10s; freeing anyway\n");
            fflush(stdout);
        }
    }
}

void reset_device_streams_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    for(uint64_t i=0; i< d_buffers->n_total_streams; ++i){
        // Fence the stream BEFORE taking the lock: this sync can block indefinitely on a wedged
        // cancelled stream, and holding the per-stream lock across it would wedge every concurrent
        // harvest/selectStream on that stream too.
        cudaSetDevice(d_buffers->streamsData[i].gpuId);
        CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[i].stream));
        // Mutate under the per-stream lock or SEGV: a concurrent get_stream_proofs harvest reads
        // `proofType` (a std::string) under this same lock; unlocked, invalidateContext()'s
        // `proofType = ""` frees it mid-read.
        std::lock_guard<std::mutex> lg(d_buffers->streamsData[i].mutex_stream_selection);
        d_buffers->streamsData[i].invalidateContext();
        d_buffers->streamsData[i].instanceId = -1;   // full teardown: no resident witness
        d_buffers->streamsData[i].reset(true);
    }

    // Clear a stranded first-GPU-buffer borrow: zisk's Phase-1 error paths skip release, so a cancel
    // can leave it set and then selectStream skips every first-GPU stream forever. Safe here because
    // reset() runs between jobs, so no legitimate borrow is live.
    d_buffers->firstGpuBufferBorrowed.store(0, std::memory_order_release);
}

void free_device_buffers_gpu(void *d_buffers_)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    wait_device_idle_before_teardown(d_buffers);

    if (d_buffers->streamsData != nullptr) {
        for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
            d_buffers->streamsData[i].free();
        }
        delete[] d_buffers->streamsData;
        d_buffers->streamsData = nullptr;
    }

    for (int i = 0; i < d_buffers->n_gpus; ++i) {
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        
        // All other GPU pointers point into this single large block, so free it once via the base pointer.
        if (d_buffers->gpuMemoryBuffer != nullptr && d_buffers->gpuMemoryBuffer[i] != nullptr) {
            CHECKCUDAERR(cudaFree(d_buffers->gpuMemoryBuffer[i]));
        }
        

        // Free CPU pointer arrays
        if (d_buffers->d_aux_trace[i] != nullptr) {
            free(d_buffers->d_aux_trace[i]);
        }
        if (d_buffers->d_aux_traceAggregation[i] != nullptr) {
            free(d_buffers->d_aux_traceAggregation[i]);
        }
        
        // Free pinned host buffers
        CHECKCUDAERR(cudaFreeHost(d_buffers->pinned_buffer[i]));
        CHECKCUDAERR(cudaFreeHost(d_buffers->pinned_buffer_extra[i]));
        CHECKCUDAERR(cudaEventDestroy(d_buffers->pinned_copy_done[i][0]));
        CHECKCUDAERR(cudaEventDestroy(d_buffers->pinned_copy_done[i][1]));
    }
    if (d_buffers->prefetchArmed) {
        int prevDevice = 0;
        CHECKCUDAERR(cudaGetDevice(&prevDevice));
        CHECKCUDAERR(cudaSetDevice(d_buffers->my_gpu_ids[0]));
        for (uint32_t s = 0; s < DeviceCommitBuffers::PREFETCH_WITNESS_SLOTS; s++) {
            if (d_buffers->prefetchReady[s] != nullptr) CHECKCUDAERR(cudaEventDestroy(d_buffers->prefetchReady[s]));
            if (d_buffers->prefetchDrained[s] != nullptr) CHECKCUDAERR(cudaEventDestroy(d_buffers->prefetchDrained[s]));
        }
        if (d_buffers->prefetchStream != nullptr) CHECKCUDAERR(cudaStreamDestroy(d_buffers->prefetchStream));
        // The zone itself is part of the unified buffer, already freed above.
        d_buffers->prefetchArmed = false;
        CHECKCUDAERR(cudaSetDevice(prevDevice));
    }
    if (d_buffers->streamCommitStreams != nullptr) {
        // The slot streams belong to the FIRST GPU's context (created there in
        // configure_stream_commit_slots). The per-GPU loop above left the last
        // GPU current, so rebind before destroying and restore afterwards --
        // destroying a stream from another device's context is invalid.
        int prevDevice = 0;
        CHECKCUDAERR(cudaGetDevice(&prevDevice));
        CHECKCUDAERR(cudaSetDevice(d_buffers->my_gpu_ids[0]));
        for (uint64_t j = 0; j < d_buffers->streamCommitSlots; j++)
            CHECKCUDAERR(cudaStreamDestroy(d_buffers->streamCommitStreams[j]));
        CHECKCUDAERR(cudaSetDevice(prevDevice));
        free(d_buffers->streamCommitStreams);
        d_buffers->streamCommitStreams = nullptr;
        d_buffers->streamCommitSlots = 0;
    }
    // Drop the process-global pause handle if it points at this instance.
    DeviceCommitBuffers *expected = d_buffers;
    gStreamCommitBuffers.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
    free(d_buffers->d_aux_trace);
    free(d_buffers->d_aux_traceAggregation);
    free(d_buffers->aux_trace_sizes);
    d_buffers->aux_trace_sizes = nullptr;
    free(d_buffers->d_constPols);
    free(d_buffers->d_constPolsAggregation);
    free(d_buffers->pinned_buffer);
    free(d_buffers->pinned_buffer_extra);
    free(d_buffers->pinned_copy_done);
    free(d_buffers->gpuMemoryBuffer);

    for (auto &outer_pair : d_buffers->air_instances) {
        for (auto &inner_pair : outer_pair.second) {
            for (AirInstanceInfo *ptr : inner_pair.second) {
                if (ptr != nullptr) {
                    delete ptr;
                }
            }
            inner_pair.second.clear();
        }
        outer_pair.second.clear();
    }
    d_buffers->air_instances.clear();
    // Manually destroy mutexes before freeing memory
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        d_buffers->mutex_pinned[i].~mutex();
    }
    free(d_buffers->mutex_pinned);

    if (d_buffers->gpus_g2l != nullptr) {
        free(d_buffers->gpus_g2l);
    }
    if (d_buffers->my_gpu_ids != nullptr) {
        free(d_buffers->my_gpu_ids);
    }
    
    delete d_buffers;
}


void load_device_setup_gpu(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packed_info, uint64_t *execData, uint64_t execWords) {
    
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    Goldilocks::Element *verkeyRoot = (Goldilocks::Element *)verkeyRoot_;

    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

    PackedInfo *packedInfo = (PackedInfo *)packed_info;

    if (d_buffers->air_instances[key][proofType].empty()) {
        d_buffers->air_instances[key][proofType].resize(d_buffers->n_gpus, nullptr);
    }

    // Circom setups carry the hash gates' row bands past their exec map. Uploaded here, with
    // the setup, so a proof that finds an AirInstanceInfo finds its bands too.
    gate_bands::BandsView bandView;
    if (execData != nullptr) {
        bandView = gate_bands::band_section(execData, execWords);
        const std::string air = "air (" + std::to_string(airgroupId) + "," + std::to_string(airId) + ")";
        if (bandView.status == gate_bands::BandSection::Malformed) {
            zklog.error("load_device_setup: " + air + " has a gate-band section that does not describe "
                        "its exec buffer; the proving key is corrupt");
            exitProcess();
        }
        if (bandView.status == gate_bands::BandSection::UnsupportedExecFormat) {
            zklog.error("load_device_setup: " + air + " has exec file format version " +
                        std::to_string(bandView.version) + ", but this build reads version " +
                        std::to_string(exec_layout::EXEC_FORMAT_VERSION) +
                        "; regenerate the proving key with a matching setup");
            exitProcess();
        }
        if (bandView.status == gate_bands::BandSection::UnsupportedVersion) {
            zklog.error("load_device_setup: " + air + " has gate-band section format version " +
                        std::to_string(bandView.version) + ", but this build understands version " +
                        std::to_string(gate_bands::GATE_BAND_FORMAT_VERSION) +
                        "; the proving key and this build disagree on the exec format -- "
                        "regenerate the proving key with a matching setup");
            exitProcess();
        }
    }
    const uint64_t *hostBands = bandView.bands;
    const uint64_t nBands = bandView.n;
    // The exec map's width. Narrower than cm1 exactly when an expander owns the rest of the
    // columns, which is what lets the host hand over a compact trace.
    uint64_t execMapCols = 0;
    if (execData != nullptr) {
        const exec_layout::Header h = exec_layout::header(execData, execWords);
        if (h.valid) execMapCols = h.mapCols;
    }
    if (nBands > 0) {
        // Checked once here rather than per thread in the kernel.
        uint64_t nRows = 1ULL << setupCtx->starkInfo.starkStruct.nBits;
        uint64_t bad = gate_bands::first_bad_band(hostBands, nBands, nRows);
        if (bad != nBands) {
            zklog.error("load_device_setup: air (" + std::to_string(airgroupId) + "," + std::to_string(airId) +
                        ") band " + std::to_string(bad) + " is row " + std::to_string(hostBands[bad * 3]) +
                        " kind " + std::to_string(hostBands[bad * 3 + 1]) +
                        ", which this build cannot expand into a trace of " + std::to_string(nRows) + " rows");
            exitProcess();
        }
    }

    // A band list is one hash family: each back-end skips kinds it does not own, so a mixed list
    // would leave the other family's bands unwritten rather than failing. Decided once, at setup.
    const gate_bands::Family family = gate_bands::family_of_bands(hostBands, nBands);
    if (family == gate_bands::Family::Mixed) {
        zklog.error("load_device_setup: air (" + std::to_string(airgroupId) + "," + std::to_string(airId) +
                    ") has gate bands of two different hash families; no expander owns them all");
        exitProcess();
    }
    // LANES and the band width both have to have travelled for a BLAKE3 air: the kernel can recover
    // neither, and a wrong value silently writes the whole band into the wrong columns. Checked once,
    // at setup.
    if (family == gate_bands::Family::Blake3 && ((bandView.aux & 0xFFFFFFFFull) == 0 || (bandView.aux >> 32) == 0)) {
        zklog.error("load_device_setup: air (" + std::to_string(airgroupId) + "," + std::to_string(airId) +
                    ") has BLAKE3 gate bands but its exec file carries no LANES or no band width");
        exitProcess();
    }

    for(int i=0; i<d_buffers->n_gpus; ++i){
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        if (d_buffers->air_instances[key][proofType][i] != nullptr) {
            delete d_buffers->air_instances[key][proofType][i];
        }
        d_buffers->air_instances[key][proofType][i] = new AirInstanceInfo(airgroupId, airId, setupCtx, verkeyRoot, packedInfo);
        // The exec map's width, so the proof path knows the host trace's row stride. See
        // AirInstanceInfo::witness_map_cols.
        d_buffers->air_instances[key][proofType][i]->set_witness_map(execMapCols);
        if (nBands > 0) {
            uploadGateBandConstantsGPU((uint64_t)family);
            d_buffers->air_instances[key][proofType][i]->set_gate_bands(hostBands, nBands, bandView.aux, (uint64_t)family);
        }
    }
}

void load_device_const_pols_gpu(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers_, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char *proofType, bool onlyFirstGPU, bool alreadyLoaded) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint64_t sizeConstPols = constSize * sizeof(Goldilocks::Element);

    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

    uint64_t const_pols_offset = initial_offset;

    // Sharing a slot with an air already uploaded: point this air's info at it, transfer
    // nothing. Layout is the same either way, so the tree offset still derives from it.
    if (alreadyLoaded) {
        for(int i=0; i<d_buffers->n_gpus; ++i){
            if (onlyFirstGPU && i > 0) break;
            AirInstanceInfo* air_instance_info = d_buffers->air_instances[key][proofType][i];
            air_instance_info->const_pols_offset = const_pols_offset;
            if (strcmp(constTreeFilename, "") != 0) {
                air_instance_info->const_tree_offset = const_pols_offset + constSize;
                air_instance_info->stored_tree = true;
            }
        }
        return;
    }

    Goldilocks::Element *constPols = new Goldilocks::Element[constSize];

    loadFileParallel(constPols, constFilename, sizeConstPols);
    
    for(int i=0; i<d_buffers->n_gpus; ++i){
        if (onlyFirstGPU && i > 0) break;
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        gl64_t *d_constPols = (strcmp(proofType, "basic") == 0) ? d_buffers->d_constPols[i] : d_buffers->d_constPolsAggregation[i];
        CHECKCUDAERR(cudaMemcpy(d_constPols + const_pols_offset, constPols, sizeConstPols, cudaMemcpyHostToDevice));
        AirInstanceInfo* air_instance_info = d_buffers->air_instances[key][proofType][i];
        air_instance_info->const_pols_offset = const_pols_offset;
    }

    delete[] constPols;

    if (strcmp(constTreeFilename, "") != 0) {
        uint64_t sizeConstTree = constTreeSize * sizeof(Goldilocks::Element);
        
        std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

        uint64_t const_tree_offset = initial_offset + constSize;

        Goldilocks::Element *constTree = new Goldilocks::Element[constTreeSize];

        loadFileParallel(constTree, constTreeFilename, sizeConstTree);
        
        for(int i=0; i<d_buffers->n_gpus; ++i){
            if (onlyFirstGPU && i > 0) break;
            cudaSetDevice(d_buffers->my_gpu_ids[i]);
            gl64_t *d_constTree = (strcmp(proofType, "basic") == 0) ? d_buffers->d_constPols[i] : d_buffers->d_constPolsAggregation[i];
            CHECKCUDAERR(cudaMemcpy(d_constTree + const_tree_offset, constTree, sizeConstTree, cudaMemcpyHostToDevice));
            AirInstanceInfo* air_instance_info = d_buffers->air_instances[key][proofType][i];
            air_instance_info->const_tree_offset = const_tree_offset;
            air_instance_info->stored_tree = true;
        }

        delete[] constTree;
    }
}

// Rebuild the region from the resident packed blob on the stream's LOWEST-priority lane: unpack,
// then LDE + merkelize with the same calls that produced the root. No H2D -- the source never left
// the device. preserve_src keeps the small domain, which the expressions read. customFixedFork
// fences against the previous proof's reads of custom_fixed (conservatively: all prior work on
// `stream`); the caller waits customFixedDone before the first use, AFTER enqueuing the witness
// copies, or the overlap is lost.
static void rebuildCustomCommitsFixed(DeviceCommitBuffers *d_buffers, SetupCtx *setupCtx, AirInstanceInfo *air,
                                      uint32_t gpuLocalId, Goldilocks::Element *dst, StreamData &sd,
                                      TimerGPU &timer) {
    if (air == nullptr || air->customPolsPackedWords == 0) {
        zklog.error("rebuildCustomCommitsFixed: no resident packed custom commit for this air");
        exitProcess();
    }
    StarkInfo &si = setupCtx->starkInfo;
    uint64_t N = 1ull << si.starkStruct.nBits;
    uint64_t w = si.mapSectionsN[si.customCommits[0].name + "0"];

    // Custom commits are a basic-air section, so they only ever live in the basic const buffer.
    Goldilocks::Element *packed =
        (Goldilocks::Element *)(d_buffers->d_constPols[gpuLocalId] + air->custom_pols_offset);

    CHECKCUDAERR(cudaEventRecord(sd.customFixedFork, sd.stream));
    CHECKCUDAERR(cudaStreamWaitEvent(sd.customStream, sd.customFixedFork, 0));

    unpack_fixed((uint64_t *)packed, (uint64_t *)(packed + 1), (uint64_t *)(packed + 1 + w),
                 (uint64_t *)dst, w, N, sd.customStream, timer);
    extendAndMerkelizeSection(w, si.starkStruct.nBits, si.starkStruct.nBitsExt,
                              si.starkStruct.merkleTreeArity,
                              si.getNumNodesMT(1ull << si.starkStruct.nBitsExt),
                              dst, dst + N * w, true, timer, sd.customStream);

    CHECKCUDAERR(cudaEventRecord(sd.customFixedDone, sd.customStream));
}

// Fill the slot reserved by reserve_custom_commit_slot. Once per air per GPU, from
// register_custom_commits: the blob then stays resident, so no proof ever DMAs a custom commit.
void upload_custom_commit_packed_gpu(uint64_t airgroupId, uint64_t airId, char *proofType, char *customFile, uint64_t wordsPerRow, void *pSetupCtx_, void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StarkInfo &si = setupCtx->starkInfo;
    uint64_t N = 1ull << si.starkStruct.nBits;
    uint64_t w = si.mapSectionsN[si.customCommits[0].name + "0"];
    uint64_t words = 1 + w + N * wordsPerRow;

    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};
    auto itK = d_buffers->air_instances.find(key);
    if (itK == d_buffers->air_instances.end()) return;
    auto itT = itK->second.find(std::string(proofType));
    if (itT == itK->second.end()) return;

    // new[], not vector: the load overwrites every word, so zero-init would be 256 MB of waste.
    // Header word + widths + rows -- the device blob keeps the file layout, so its own header
    // describes it, the same trick the packed const pols use.
    std::unique_ptr<uint64_t[]> host(new uint64_t[words]);
    loadFileParallel(host.get(), std::string(customFile), words * sizeof(uint64_t), true, CUSTOM_COMMIT_ROOT_BYTES);

    for (int i = 0; i < d_buffers->n_gpus && i < (int)itT->second.size(); ++i) {
        AirInstanceInfo *air = itT->second[i];
        // Reserved-only: a path that reserved just GPU 0 leaves the others at offset 0, which is
        // the const pols' own slot.
        if (air == nullptr || air->customPolsReservedWords == 0) continue;
        if (words > air->customPolsReservedWords) {
            zklog.error("upload_custom_commit_packed: " + std::to_string(words) +
                        " words exceeds the reserved " + std::to_string(air->customPolsReservedWords));
            exitProcess();
        }
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        CHECKCUDAERR(cudaMemcpy(d_buffers->d_constPols[i] + air->custom_pols_offset, host.get(),
                                words * sizeof(uint64_t), cudaMemcpyHostToDevice));
        air->customPolsPackedWords = words;
    }
}

// Record the custom-commit slot this air was given in the const-pols buffer. Nothing is uploaded
// here: the commit's file path only arrives with register_custom_commits, later.
void reserve_custom_commit_slot_gpu(uint64_t airgroupId, uint64_t airId, char *proofType, uint64_t offset, uint64_t reservedWords, void *d_buffers_, bool onlyFirstGPU) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};
    auto itK = d_buffers->air_instances.find(key);
    if (itK == d_buffers->air_instances.end()) return;
    auto itT = itK->second.find(std::string(proofType));
    if (itT == itK->second.end()) return;
    for (int i = 0; i < d_buffers->n_gpus && i < (int)itT->second.size(); ++i) {
        if (onlyFirstGPU && i > 0) break;
        AirInstanceInfo *air = itT->second[i];
        if (air == nullptr) continue;
        air->custom_pols_offset = offset;
        air->customPolsReservedWords = reservedWords;
    }
}

// Stage `trace` into the cursor slot on the copy stream and tag it for `instanceId`.
// Caller MUST hold prefetchMutex. Drops a stale unconsumed entry (host-syncing its
// in-flight upload first), orders the copy behind the slot's drain, records
// prefetchReady and advances the cursor. Returns the slot used.
static uint32_t stageWitnessSlotLocked(DeviceCommitBuffers *d_buffers, uint64_t instanceId,
                                       const void *trace, uint64_t total_size) {
    uint32_t slot = d_buffers->prefetchStageSlot;
    if (d_buffers->prefetchInstanceId[slot] != -1) {
        // Stale unconsumed entry (single staging producer: safe to drop after waiting
        // out its in-flight upload -- its host buffer is untracked past this point).
        zklog.warning("prefetch: dropping stale witness staging of instance " +
                      std::to_string(d_buffers->prefetchInstanceId[slot]) + " (want " +
                      std::to_string(instanceId) + ")");
        CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchReady[slot]));
        d_buffers->prefetchInstanceId[slot] = -1;
        d_buffers->prefetchTraceBytes[slot] = 0;
    }
    uint8_t *slotBase = (uint8_t *)(d_buffers->prefetchRegionBase + (uint64_t)slot * d_buffers->prefetchSlotStride);
    // Never overwrite a slot the proof stream has not drained yet. Waiting on a
    // never-recorded event is a no-op, so the first staging passes through.
    CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->prefetchDrained[slot], 0));
    // Chunked so no single transfer monopolizes PCIe.
    const uint64_t blockBytes = 32ull << 20;
    for (uint64_t off = 0; off < total_size; off += blockBytes) {
        uint64_t len = std::min(blockBytes, total_size - off);
        CHECKCUDAERR(cudaMemcpyAsync(slotBase + off, (const uint8_t *)trace + off, len,
                                     cudaMemcpyHostToDevice, d_buffers->prefetchStream));
    }
    CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchReady[slot], d_buffers->prefetchStream));
    d_buffers->prefetchInstanceId[slot] = (int64_t)instanceId;
    d_buffers->prefetchTraceBytes[slot] = total_size;
    d_buffers->prefetchStageSlot = (slot + 1) % DeviceCommitBuffers::PREFETCH_WITNESS_SLOTS;
    return slot;
}

uint64_t gen_proof_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params_, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, bool skipRecalculation, uint64_t streamId_, char *constPolsPath,  char *constTreePath, char *customCommitsFixedPath, bool selfContained) {

    auto key = std::make_pair(airgroupId, airId);
    std::string proofType = "basic";

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    // Count this thread as inside device work so a concurrent teardown waits for it before freeing.
    InFlightScope in_flight(d_buffers);
    uint32_t streamId;
    if (skipRecalculation) {
        // Validate the witness is still resident under the mutex; the stream may have
        // been reused since the snapshot. No fallback — the host trace may be recycled.
        streamId = streamId_;
        StreamData &sd = d_buffers->streamsData[streamId];
        std::lock_guard<std::mutex> lock(sd.mutex_stream_selection);
        bool resident = sd.status == 3 && sd.witnessResident && sd.instanceId == (int64_t)instanceId &&
                        sd.airgroupId == airgroupId && sd.airId == airId;
        if (!resident) {
            zklog.error("gen_proof: instance " + std::to_string(instanceId) +
                        " witness no longer resident on stream " + std::to_string(streamId) +
                        " (status " + std::to_string(sd.status) + ", instanceId " +
                        std::to_string(sd.instanceId) + ", proofType " + sd.proofType + ")");
            return UINT64_MAX;
        }
        reserveStreamLocked(d_buffers, streamId); // mutex held by lock_guard above
    } else if (streamId_ == UINT64_MAX) {
        // No reservation supplied (one-off / non-scheduler caller): select internally.
        streamId = selectStream(d_buffers, airgroupId, airId, proofType, false, false);
    } else {
        // Recompute path: the scheduler already reserved this stream (status=1).
        streamId = (uint32_t)streamId_;
    }
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    cudaSetDevice(gpuId);

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].curTimer();

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][proofType][gpuLocalId];

    const bool pipeline = d_buffers->pipelineMode;

    // Read the prior context before overwriting it below. Fixed columns are keyed by slot,
    // so an air sharing them with the stream's previous air reuses them; custom_fixed is
    // per-air.
    StreamData &sd = d_buffers->streamsData[streamId];
    bool reuse_custom_fixed = sd.airgroupId == airgroupId && sd.airId == airId && sd.proofType == string("basic");
    // constPolsAliasTree airs cannot reuse: their pols sit in the tree's node area, which the
    // previous proof's merkelize overwrote.
    bool reuse_constants = sd.adoptFixedSlot(air_instance_info->const_pols_offset,
                                             setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)], false, "")
                           && !setupCtx->starkInfo.constPolsAliasTree;
    bool reuse_const_tree = reuse_constants && sd.constTreeResident;

    sd.pSetupCtx = pSetupCtx_;
    // Pipeline: completion metadata lives in the ring (per proof); leaving
    // proofBuffer set would make a stray collectStreamResult read stale fields.
    sd.proofBuffer = pipeline ? nullptr : proofBuffer;
    sd.proofFile = string(proofFile);
    sd.airgroupId = airgroupId;
    sd.airId = airId;
    sd.instanceId = instanceId;
    sd.proofType = "basic";
    sd.witnessResident = false;

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetStage1Extended = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];

    bool customFixedRebuilt = false;
    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0) {
        if (!reuse_custom_fixed) {
            Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            rebuildCustomCommitsFixed(d_buffers, setupCtx, air_instance_info, gpuLocalId, pCustomCommitsFixed, sd, timer);
            customFixedRebuilt = true;
        }
    }

    if (!skipRecalculation) {
        uint64_t total_size = (d_buffers->packedTrace && air_instance_info->is_packed) ? air_instance_info->num_packed_words * N * sizeof(Goldilocks::Element) : N * nCols * sizeof(Goldilocks::Element);
        uint64_t *dst = (uint64_t *)(d_aux_trace + offsetStage1Extended);
        // Zone is FIRST-GPU only for now (extending to all GPUs is planned once the
        // first version is in production); other GPUs use the legacy upload.
        if (d_buffers->prefetchArmed && sd.gpuId == d_buffers->my_gpu_ids[0]) {
            std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
            // Find the slot holding this instance's staged witness.
            int slot = -1;
            for (uint32_t s = 0; s < DeviceCommitBuffers::PREFETCH_WITNESS_SLOTS; s++) {
                if (d_buffers->prefetchInstanceId[s] == (int64_t)instanceId &&
                    d_buffers->prefetchTraceBytes[s] == total_size) { slot = (int)s; break; }
            }
            if (slot >= 0) {
                // Hit: the trace already uploaded to the zone on the copy stream while the
                // previous proof computed. No host sync -- the proof stream waits on the
                // copy's event.
                CHECKCUDAERR(cudaStreamWaitEvent(stream, d_buffers->prefetchReady[slot], 0));
            } else {
                // Miss: nothing staged for this instance. Stage host -> slot here,
                // host-synced so the caller may recycle the buffer at once.
                slot = (int)stageWitnessSlotLocked(d_buffers, instanceId, params->trace, total_size);
                CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchReady[slot]));
            }
            d_buffers->prefetchInstanceId[slot] = -1;
            d_buffers->prefetchTraceBytes[slot] = 0;
            // Land the staged trace into cm1ext with one D2D on the proof stream, then mark
            // the slot recyclable for the next staging.
            gl64_t *slotBase = d_buffers->prefetchRegionBase + (uint64_t)slot * d_buffers->prefetchSlotStride;
            CHECKCUDAERR(cudaMemcpyAsync(dst, slotBase, total_size,
                                         cudaMemcpyDeviceToDevice, stream));
            CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchDrained[slot], stream));
            // Host-buffer release gate: the copy stream's tail is at/after this trace's H2D,
            // so the event fires when the HOST buffer is free -- not when the proof runs.
            CHECKCUDAERR(cudaEventRecord(d_buffers->streamsData[streamId].trace_copy_event, d_buffers->prefetchStream));
        } else {
            copy_to_device_in_chunks(d_buffers, params->trace, dst, total_size, streamId, timer);
        }
    }
    
    size_t totalCopySize = 0;
    totalCopySize += setupCtx->starkInfo.nPublics;
    totalCopySize += setupCtx->starkInfo.proofValuesSize;
    totalCopySize += setupCtx->starkInfo.airgroupValuesSize;
    totalCopySize += setupCtx->starkInfo.airValuesSize;
    totalCopySize += FIELD_EXTENSION;

    // Stage into the per-stream pinned region for an async copy (no stream sync);
    // reuse gated by end_event on stream reselect. Runtime check survives NDEBUG.
    if (totalCopySize > PINNED_AUX_VALUES_MAX) {
        zklog.error("gen_proof_gpu: aux_values size " + std::to_string(totalCopySize) +
                    " exceeds PINNED_AUX_VALUES_MAX " + std::to_string(PINNED_AUX_VALUES_MAX));
        exitProcess();
    }
    // Parity slot: proof N+1's CPU staging must not overwrite the region proof N's
    // still-pending async H2D reads (launchSeq increments at ring push, below).
    const uint32_t pinnedSlot = (pipeline && !skipRecalculation)
        ? (uint32_t)(d_buffers->streamsData[streamId].launchSeq & 1) : 0;
    Goldilocks::Element *aux_values = d_buffers->streamsData[streamId].pinned_aux_values
        + (uint64_t)pinnedSlot * PINNED_AUX_VALUES_MAX;
    uint64_t offset = 0;
    memcpy(aux_values + offset, params->publicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.nPublics;
    if (setupCtx->starkInfo.proofValuesSize > 0) {
        memcpy(aux_values + offset, params->proofValues, setupCtx->starkInfo.proofValuesSize * sizeof(Goldilocks::Element));
        offset += setupCtx->starkInfo.proofValuesSize;
    }
    if (setupCtx->starkInfo.airgroupValuesSize > 0) {
        memcpy(aux_values + offset, params->airgroupValues, setupCtx->starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element));
        offset += setupCtx->starkInfo.airgroupValuesSize;
    }
    if (setupCtx->starkInfo.airValuesSize > 0) {
        memcpy(aux_values + offset, params->airValues, setupCtx->starkInfo.airValuesSize * sizeof(Goldilocks::Element));
        offset += setupCtx->starkInfo.airValuesSize;
    }
    memcpy(aux_values + offset, (Goldilocks::Element *)globalChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));

    CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)(d_aux_trace + offsetPublicInputs), aux_values, totalCopySize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));

    gl64_t *d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
    gl64_t *d_const_tree;
    if (air_instance_info->stored_tree) {
        // Preallocated in the const buffer, so it is in place unconditionally.
        d_const_tree = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_tree_offset;
        reuse_const_tree = reuse_constants;
    } else {
        // find(), not operator[]: a preallocate-layout starkinfo has NO aux tree slot, and
        // operator[] would silently insert offset 0 -- the tree would then be written over the
        // aux BASE.
        auto itConstTree = setupCtx->starkInfo.mapOffsets.find(std::make_pair("const", true));
        if (itConstTree == setupCtx->starkInfo.mapOffsets.end()) {
            zklog.error("air " + std::to_string(airgroupId) + ":" + std::to_string(airId) +
                        " has no aux const-tree slot (preallocate layout) but stored_tree is false");
            exitProcess();
        }
        uint64_t offsetConstTree = itConstTree->second;
        d_const_tree = d_aux_trace + offsetConstTree;

        // calculateFixedExtended airs merkelize inside genProof_gpu instead, on the same
        // flag -- either way ("const", true) holds this slot's tree on exit.
        if (!reuse_const_tree && !setupCtx->starkInfo.calculateFixedExtended) {
            load_and_copy_to_device_in_chunks(d_buffers, constTreePath, (uint8_t*)d_const_tree, sizeConstTree, streamId);
        }
    }
    sd.constTreeResident = true;


    proofman_sumcheck_set_context(instanceId, airgroupId, airId);
    // The tree-aware flag, not the slot one: genProof_gpu's reuse also gates the
    // calculateFixedExtended merkelize, and a slot claimed by commit_witness has pols but no
    // tree. Costs a redundant unpack in exactly that case.
    if (customFixedRebuilt) CHECKCUDAERR(cudaStreamWaitEvent(stream, sd.customFixedDone, 0));
    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, constTreePath, streamId, instanceId, d_buffers, air_instance_info, skipRecalculation, timer, stream, selfContained, reuse_const_tree);
    if (pipeline) {
        // Snapshot the completion into the ring (the per-stream sd fields will be
        // overwritten by the next launch); the harvester writeProofs + fires the callback.
        std::lock_guard<std::mutex> plk(sd.pipeMutex);
        StreamData::PipelineSlot &ps = sd.pipeSlots[(sd.pipeHead + sd.pipeCount) % 2];
        ps.instanceId = (int64_t)instanceId;
        ps.airgroupId = airgroupId;
        ps.airId = airId;
        ps.pSetupCtx = pSetupCtx_;
        ps.proofBuffer = proofBuffer;
        ps.proofFile = string(proofFile);
        ps.proofType = "basic";
        ps.pinnedProof = sd.pinned_buffer_proof + (uint64_t)pinnedSlot * sd.maxProofSize;
        ps.timer = &sd.curTimer();
        CHECKCUDAERR(cudaEventRecord(ps.done, stream));
        sd.pipeCount++;
        sd.launchSeq++;
    }
    cudaEventRecord(sd.end_event, stream);
    sd.status = 2;
    return streamId;
}

uint64_t initialize_instance_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* params_, void *d_buffers_, char *customCommitsFixedPath) {
    auto key = std::make_pair(airgroupId, airId);
    std::string proofType = "basic";

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint32_t streamId = selectStream(d_buffers, airgroupId, airId, proofType, false);
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    cudaSetDevice(gpuId);

    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][string(proofType)][gpuLocalId];

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].curTimer();

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
   
    // Same split as gen_proof_gpu. This path never loads the tree, so it must not claim
    // constTreeResident; adoptFixedSlot clears it on a slot change and leaves it otherwise.
    StreamData &sd = d_buffers->streamsData[streamId];
    bool reuse_custom_fixed = sd.airgroupId == airgroupId && sd.airId == airId && sd.proofType == string("basic");
    bool reuse_constants = sd.adoptFixedSlot(air_instance_info->const_pols_offset,
                                             setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)], false, "")
                           && !setupCtx->starkInfo.constPolsAliasTree;

    sd.pSetupCtx = pSetupCtx_;
    sd.airgroupId = airgroupId;
    sd.airId = airId;
    sd.proofType = "basic";
    sd.instanceId = instanceId;
    sd.witnessResident = false;

    proofman_sumcheck_set_context(instanceId, airgroupId, airId);

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];

    // Rebuild now, wait at the end of the function: whatever consumes this instance next runs on
    // the same stream, so the wait there orders it.
    bool customFixedRebuilt = false;
    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0) {
        if (!reuse_custom_fixed) {
            Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            rebuildCustomCommitsFixed(d_buffers, setupCtx, air_instance_info, gpuLocalId, pCustomCommitsFixed, sd, timer);
            customFixedRebuilt = true;
        }
    }

    uint64_t total_size = (d_buffers->packedTrace && air_instance_info->is_packed) ? air_instance_info->num_packed_words * N * sizeof(Goldilocks::Element) : N * nCols * sizeof(Goldilocks::Element);
    uint64_t *dst = (uint64_t *)(d_aux_trace + offsetStage1 + N * nCols);
    copy_to_device_in_chunks(d_buffers, params->trace, dst, total_size, streamId, timer);
    PROOFMAN_SUMCHECK("proof_before_unpack", dst, total_size / sizeof(uint64_t), stream);

    size_t totalCopySize = 0;
    totalCopySize += setupCtx->starkInfo.nPublics;
    totalCopySize += setupCtx->starkInfo.proofValuesSize;
    totalCopySize += setupCtx->starkInfo.airgroupValuesSize;
    totalCopySize += setupCtx->starkInfo.airValuesSize;
    totalCopySize += 2 * FIELD_EXTENSION;

    // Stage into the per-stream pinned region for an async copy (no stream sync);
    // reuse gated by end_event on stream reselect. Runtime check survives NDEBUG.
    if (totalCopySize > PINNED_AUX_VALUES_MAX) {
        zklog.error("initialize_instance_gpu: aux_values size " + std::to_string(totalCopySize) +
                    " exceeds PINNED_AUX_VALUES_MAX " + std::to_string(PINNED_AUX_VALUES_MAX));
        exitProcess();
    }
    Goldilocks::Element *aux_values = d_buffers->streamsData[streamId].pinned_aux_values;
    uint64_t offset = 0;
    memcpy(aux_values + offset, params->publicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.nPublics;
    if (setupCtx->starkInfo.proofValuesSize > 0) {
        memcpy(aux_values + offset, params->proofValues, setupCtx->starkInfo.proofValuesSize * sizeof(Goldilocks::Element));
        offset += setupCtx->starkInfo.proofValuesSize;
    }
    if (setupCtx->starkInfo.airgroupValuesSize > 0) {
        memcpy(aux_values + offset, params->airgroupValues, setupCtx->starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element));
        offset += setupCtx->starkInfo.airgroupValuesSize;
    }
    if (setupCtx->starkInfo.airValuesSize > 0) {
        memcpy(aux_values + offset, params->airValues, setupCtx->starkInfo.airValuesSize * sizeof(Goldilocks::Element));
        offset += setupCtx->starkInfo.airValuesSize;
    }
    memcpy(aux_values + offset, (Goldilocks::Element *)params->challenges, 2 * FIELD_EXTENSION * sizeof(Goldilocks::Element));

    CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)(d_aux_trace + offsetPublicInputs), aux_values, totalCopySize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
    
    gl64_t *d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
    
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;
    if (!reuse_constants) {
        unpack_fixed((uint64_t*)d_const_pols, (uint64_t*)(d_const_pols + 1), (uint64_t*)(d_const_pols + 1 + setupCtx->starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants, N, stream, timer);
        CHECKCUDAERR(cudaGetLastError());
    }

    uint64_t offsetCm1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    if (d_buffers->packedTrace && air_instance_info->is_packed) {
        unpack_trace(air_instance_info, (uint64_t*)(d_aux_trace + offsetCm1 + N * nCols), (uint64_t*)(d_aux_trace + offsetCm1), nCols, N, stream, timer);
    } else {
        fromRowMajorToColMajor(N, nCols, (gl64_t *)(d_aux_trace + offsetCm1 + N * nCols), (gl64_t*)(d_aux_trace + offsetCm1), resolveLayout(setupCtx->starkInfo.starkStruct.nBits, nCols), stream);
    }
    PROOFMAN_SUMCHECK("proof_after_unpack", d_aux_trace + offsetCm1, N * nCols, stream);

    if (customFixedRebuilt) CHECKCUDAERR(cudaStreamWaitEvent(stream, sd.customFixedDone, 0));

    return streamId;
}

void calculate_trace_instance_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, void *params_, void *d_buffers_, uint64_t streamId) {
    auto key = std::make_pair(airgroupId, airId);
    std::string proofType = "basic";

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    cudaSetDevice(gpuId);

    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][string(proofType)][gpuLocalId];

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].curTimer();

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];

    calculateTraceInstance(*setupCtx, d_aux_trace, streamId, d_buffers, air_instance_info, params->airgroupValues, timer, stream);
}

void verify_constraints_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, void* params_, void* constraintsInfo, void *d_buffers_, uint64_t streamId) {

    auto key = std::make_pair(airgroupId, airId);
    std::string proofType = "basic";

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    cudaSetDevice(gpuId);

    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][string(proofType)][gpuLocalId];

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].curTimer();

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];

    verifyConstraintsGPU(*setupCtx, d_aux_trace, streamId, d_buffers, air_instance_info, (ConstraintInfo *)constraintsInfo, timer, stream);
    cudaEventRecord(d_buffers->streamsData[streamId].end_event, stream);
    d_buffers->streamsData[streamId].status = 2;
}

void get_proof(DeviceCommitBuffers *d_buffers, uint64_t streamId) {
    SetupCtx *setupCtx = (SetupCtx*) d_buffers->streamsData[streamId].pSetupCtx;
    uint64_t airgroupId = d_buffers->streamsData[streamId].airgroupId;
    uint64_t airId = d_buffers->streamsData[streamId].airId;
    uint64_t instanceId = d_buffers->streamsData[streamId].instanceId;
    uint64_t * proofBuffer = d_buffers->streamsData[streamId].proofBuffer;
    string proofType = d_buffers->streamsData[streamId].proofType;
    string proofFile = d_buffers->streamsData[streamId].proofFile;
    TimerGPU &timer = d_buffers->streamsData[streamId].curTimer();

    closeStreamTimer(timer, instanceId, airgroupId, airId, true);

    writeProof(*setupCtx, d_buffers->streamsData[streamId].pinned_buffer_proof, proofBuffer, airgroupId, airId, instanceId, proofFile);

    if (proof_done_callback != nullptr) {
        proof_done_callback(instanceId, proofType.c_str());
    }
}

static void collectStreamResult(DeviceCommitBuffers *d_buffers, uint64_t streamId) {
    StreamData &sd = d_buffers->streamsData[streamId];
    bool commitRoot = sd.root != nullptr;
    if (commitRoot) {
        get_commit_root(d_buffers, streamId);
    } else if (sd.proofBuffer != nullptr) {
        get_proof(d_buffers, streamId);
    }
    // reset() leaves instanceId/proofType untouched, so a committed witness stays resident;
    // get_instances_ready's proofType gate keeps finished proof streams out of the scan.
    sd.reset(false);
}

// Requires the caller to hold streamsData[streamId].mutex_stream_selection.
// Deep pipeline: a busy basic stream is reservable while it holds fewer than 2
// in-flight proofs. pipeCount has its own lock.
static bool pipelineReservable(DeviceCommitBuffers *d_buffers, StreamData &sd) {
    if (!d_buffers->pipelineMode || sd.recursive) return false;
    if (sd.status.load(std::memory_order_relaxed) != 2) return false;
    std::lock_guard<std::mutex> plk(sd.pipeMutex);
    return sd.pipeCount < 2;
}

// Harvest fired pipeline ring entries on one stream: writeProof from the per-proof
// pinned slot + completion callback. Never touches sd.* proof fields (stale under
// pipelining) and never host-blocks unless `blocking`.
static void harvestPipelineStream(DeviceCommitBuffers *d_buffers, uint64_t streamId, bool blocking) {
    StreamData &sd = d_buffers->streamsData[streamId];
    // One harvester at a time: a non-blocking caller yields to whoever is already
    // draining (the entries WILL be collected); a blocking caller must wait for
    // the ring to be truly empty, so it takes the lock unconditionally.
    std::unique_lock<std::mutex> hlk(sd.harvestMutex, std::defer_lock);
    if (blocking) {
        hlk.lock();
    } else if (!hlk.try_lock()) {
        return;
    }
    for (;;) {
        StreamData::PipelineSlot *ps = nullptr;
        {
            std::lock_guard<std::mutex> plk(sd.pipeMutex);
            if (sd.pipeCount == 0) return;
            ps = &sd.pipeSlots[sd.pipeHead];
        }
        cudaSetDevice(sd.gpuId);
        if (blocking) {
            CHECKCUDAERR(cudaEventSynchronize(ps->done));
        } else if (cudaEventQuery(ps->done) != cudaSuccess) {
            return;
        }
        // Every event of this proof precedes `done` on the stream, so the syncs inside are immediate.
        closeStreamTimer(*ps->timer, (uint64_t)ps->instanceId, ps->airgroupId, ps->airId, true);
        SetupCtx *setupCtx = (SetupCtx *)ps->pSetupCtx;
        writeProof(*setupCtx, ps->pinnedProof, ps->proofBuffer, ps->airgroupId, ps->airId,
                   (uint64_t)ps->instanceId, ps->proofFile);
        if (proof_done_callback != nullptr) {
            proof_done_callback((uint64_t)ps->instanceId, ps->proofType.c_str());
        }
        {
            std::lock_guard<std::mutex> plk(sd.pipeMutex);
            sd.pipeHead = (sd.pipeHead + 1) % 2;
            sd.pipeCount--;
        }
    }
}

// Diagnostic for a wedged proofs phase (called from the settle-timeout path): every stream's
// status, ring occupancy and slot identities, and whether its last event has fired.
void dump_pipeline_state_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr) return;
    fprintf(stderr, "[pipeline] mode=%d streams=%u phaseB=%d state=%u closing=%d\n", (int)d_buffers->pipelineMode,
            d_buffers->n_total_streams, (int)d_buffers->phaseBAliased, d_buffers->phaseBState.load(),
            (int)d_buffers->phaseBClosing.load());
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        StreamData &sd = d_buffers->streamsData[i];
        cudaSetDevice(sd.gpuId);
        cudaError_t ev = cudaEventQuery(sd.end_event);
        uint32_t cnt, head;
        { std::lock_guard<std::mutex> plk(sd.pipeMutex); cnt = sd.pipeCount; head = sd.pipeHead; }
        fprintf(stderr, "[pipeline] stream %lu gpu %u recursive=%d status=%u end_event=%s inst=%ld type=%s witnessResident=%d ring count=%u head=%u\n",
                i, sd.gpuId, (int)sd.recursive, sd.status.load(), ev == cudaSuccess ? "done" : cudaGetErrorName(ev),
                (long)sd.instanceId, sd.proofType.c_str(), (int)sd.witnessResident, cnt, head);
        for (uint32_t k = 0; k < cnt; k++) {
            StreamData::PipelineSlot &ps = sd.pipeSlots[(head + k) % 2];
            cudaError_t d = ps.done ? cudaEventQuery(ps.done) : cudaErrorInvalidValue;
            fprintf(stderr, "[pipeline]    slot %u: inst=%ld air %lu:%lu type=%s done=%s\n", (head + k) % 2, (long)ps.instanceId,
                    ps.airgroupId, ps.airId, ps.proofType.c_str(), d == cudaSuccess ? "yes" : cudaGetErrorName(d));
        }
        cudaGetLastError();
    }
    fflush(stderr);
}

void harvest_pipeline_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || !d_buffers->pipelineMode) return;
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        harvestPipelineStream(d_buffers, i, false);
    }
}

// Toggle deep pipelining (proofs phase only: the contributions phase relies on
// harvest-on-reserve for commit roots, so it must stay off there).
void set_pipeline_mode_gpu(void *d_buffers_, bool enable) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr) return;
    d_buffers->pipelineMode = enable;
}

// Phase B registration, before gen_device_streams: the two recursive streams alias the basic
// stream's buffer (see DeviceCommitBuffers::phaseBAliased).
void configure_phase_b_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr) return;
    d_buffers->phaseBAliased = true;
}

// Phase-B transitions. 0: job start, phase A. 1: every basic and compressor completed -- close
// the basic stream to new reservations; the aliases open once it has drained (tryOpenPhaseB).
// 2: recursion complete -- fence the aliases, drop the basic stream's const identity (the
// aliases overwrote its buffer) and hand it back for VadcopFinal. Returns 0, or -1 when phase B
// is not configured.
int64_t set_phase_b_gpu(void *d_buffers_, uint32_t state) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || !d_buffers->phaseBAliased) return -1;
    if (state == 1) {
        d_buffers->phaseBClosing.store(true, std::memory_order_release);
        return 0;
    }
    if (state == 2) {
        for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
            StreamData &sd = d_buffers->streamsData[i];
            cudaSetDevice(sd.gpuId);
            if (sd.recursive) {
                CHECKCUDAERR(cudaStreamSynchronize(sd.stream));
            } else {
                std::lock_guard<std::mutex> lk(sd.mutex_stream_selection);
                sd.invalidateContext();
            }
        }
    }
    d_buffers->phaseBClosing.store(false, std::memory_order_release);
    d_buffers->phaseBState.store(state, std::memory_order_release);
    return 0;
}

void get_stream_proofs_gpu(void *d_buffers_){
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        d_buffers->streamsData[i].mutex_stream_selection.lock();
        uint32_t status = d_buffers->streamsData[i].status;
        if (status != 2) {
            if (status == 1) {
                zklog.warning("get_stream_proofs: skipping stream " + std::to_string(i) +
                              " still being enqueued (instanceId " +
                              std::to_string(d_buffers->streamsData[i].instanceId) + ")");
            }
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
            continue;
        }
        cudaSetDevice(d_buffers->streamsData[i].gpuId);
        CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[i].stream));
        if (d_buffers->pipelineMode) harvestPipelineStream(d_buffers, i, true);
        collectStreamResult(d_buffers, i);
        d_buffers->streamsData[i].mutex_stream_selection.unlock();
    }
}

void get_stream_proofs_non_blocking_gpu(void *d_buffers_){
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        if (d_buffers->pipelineMode) harvestPipelineStream(d_buffers, i, false);
        if (d_buffers->streamsData[i].mutex_stream_selection.try_lock()) {
            if(d_buffers->streamsData[i].status==2 &&  cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess) {
                cudaSetDevice(d_buffers->streamsData[i].gpuId);
                collectStreamResult(d_buffers, i);
            }
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
        }
    }
}

void get_stream_id_proof_gpu(void *d_buffers_, uint64_t streamId) {
    if (d_buffers_ == nullptr) return;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    // Guard the C-ABI surface: an out-of-range streamId would index streamsData OOB and segfault.
    if (streamId >= d_buffers->n_total_streams) {
        zklog.warning("get_stream_id_proof: stream " + std::to_string(streamId) +
                      " out of range (n_total_streams " + std::to_string(d_buffers->n_total_streams) +
                      "); ignoring");
        return;
    }
    cudaSetDevice(d_buffers->streamsData[streamId].gpuId);
    // Hold the per-stream lock across harvest + reset or SEGV: a concurrent reset_device_streams_gpu
    // (same lock) races the `proofType` (std::string) read and frees the string mid-read.
    std::lock_guard<std::mutex> lock(d_buffers->streamsData[streamId].mutex_stream_selection);
    if (d_buffers->streamsData[streamId].status != 2) {
        if (d_buffers->streamsData[streamId].status == 1) {
            zklog.warning("get_stream_id_proof: stream " + std::to_string(streamId) +
                          " already re-assigned and being enqueued (instanceId " +
                          std::to_string(d_buffers->streamsData[streamId].instanceId) +
                          "); caller's proof was already collected");
        }
        return;
    }
    CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[streamId].stream));
    if (d_buffers->pipelineMode) harvestPipelineStream(d_buffers, streamId, true);
    collectStreamResult(d_buffers, streamId);
}

uint64_t gen_recursive_proof_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *trace, void *aux_trace, void *pConstPols, void *pConstTree, void *pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers_, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream, char *recurser_id, uint64_t streamId_)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    bool aggregation = false;
    if(string(proofType) == "recursive1" || string(proofType) == "recursive2") {
        aggregation = true;
    }
    // streamId_ == UINT64_MAX: select internally (one-off launches). Otherwise the scheduler
    // already reserved this stream — use it directly.
    uint32_t streamId = (streamId_ == UINT64_MAX)
        ? selectStream(d_buffers, airgroupId, airId, proofType, aggregation, force_recursive_stream)
        : (uint32_t)streamId_;
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].curTimer();
    
    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];

    gl64_t * d_aux_trace = d_buffers->streamsData[streamId].recursive
        ? (gl64_t *)d_buffers->d_aux_traceAggregation[gpuLocalId][d_buffers->streamsData[streamId].localStreamId]
        : d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];
    uint64_t sizeTrace = N * nCols * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);

    auto key = std::make_pair(airgroupId, airId);
    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][string(proofType)][gpuLocalId];

    // Keyed on the slot, so airs with identical fixed reuse each other's pols. Recursers are
    // the exception: they share one slot, so recurser_id stays part of the key.
    StreamData &sd = d_buffers->streamsData[streamId];
    bool reuse_constants = sd.adoptFixedSlot(air_instance_info->const_pols_offset,
                                             setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)], true,
                                             string(recurser_id))
                           && !setupCtx->starkInfo.constPolsAliasTree;
    bool reuse_const_tree = reuse_constants && sd.constTreeResident;

    sd.pSetupCtx = pSetupCtx_;
    // Pipeline on the shared stream: completion metadata rides the ring below.
    sd.proofBuffer = (d_buffers->pipelineMode && !sd.recursive) ? nullptr : proofBuffer;
    sd.proofFile = string(proof_file);
    sd.recurserId = string(recurser_id);
    sd.airgroupId = airgroupId;
    sd.airId = airId;
    sd.instanceId = instanceId;
    sd.proofType = string(proofType);
    sd.witnessResident = false;

    uint64_t offsetStage1Extended = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    // When the exec map is narrower than cm1 the host hands over a COMPACT N x mapCols trace and the
    // columns it omits -- the expander's, which getCommitedPols could only zero -- never cross PCIe.
    // Every Rust caller that can reach this function fills compactly under the same condition, since
    // both sides read mapCols out of the same exec header. See widenCompactWitnessKernel.
    const uint64_t mapCols = air_instance_info->witness_map_cols;
    const bool compactWitness = mapCols > 0 && mapCols < nCols;
    // The compact trace lands in the stream's WITNESS TAIL past mapTotalN: every recursive-capable
    // class is planned as mapTotalN + the largest compact trace of its repository (staging cols x N,
    // SetupsVadcop max_compact_trace_size) and nothing on the device addresses past mapTotalN.
    gl64_t *d_witnessTail = d_aux_trace + setupCtx->starkInfo.mapTotalN;
    if (compactWitness && sd.auxTraceCapacity < setupCtx->starkInfo.mapTotalN + N * mapCols) {
        zklog.error("gen_recursive_proof: stream " + std::to_string(streamId) + " (capacity " +
                    std::to_string(sd.auxTraceCapacity) + " elements) has no witness tail for the compact witness of air (" +
                    std::to_string(airgroupId) + "," + std::to_string(airId) + ") " + proofType);
        exitProcess();
    }
    // Getting the witness onto the device happens BEFORE genProof_gpu opens STARK_GPU_PROOF, so it
    // is a timer of its own rather than a category: a category here would be divided by a window
    // that does not contain it, which is what made that table total 102% with OTHER pinned at zero.
    TimerStartGPU(timer, STARK_GPU_WITNESS);
    if (compactWitness) {
        CHECKCUDAERR(cudaMemsetAsync((uint8_t*)(d_aux_trace + offsetStage1Extended), 0, sizeTrace, stream));
        copy_to_device_in_chunks(d_buffers, trace, (uint8_t*)d_witnessTail,
                                 N * mapCols * sizeof(Goldilocks::Element), streamId, timer, false);
        widenCompactWitnessGPU((uint64_t*)(d_aux_trace + offsetStage1Extended), nCols, N,
                               (const uint64_t*)d_witnessTail, mapCols, stream);
    } else {
        copy_to_device_in_chunks(d_buffers, trace, (uint8_t*)(d_aux_trace + offsetStage1Extended), sizeTrace, streamId, timer, false);
    }

    // Stream-owned scratch; see StreamData::d_gate_band_scratch. Once per stream, and only for a
    // family that asks for one.
    const uint64_t gateBandScratchWords = gateBandScratchWordsGPU(air_instance_info->gate_band_family);
    if (gateBandScratchWords > 0 && sd.d_gate_band_scratch == nullptr) {
        cudaSetDevice(gpuId);
        CHECKCUDAERR(cudaMalloc(&sd.d_gate_band_scratch, gateBandScratchWords * sizeof(uint64_t)));
    }

    // The host copied up the boundary cells; the interiors get rebuilt here. Stream-ordered
    // behind the copy. Airs whose setup registered no bands skip it.
    //
    // Part of STARK_GPU_WITNESS above: like the copy, it runs before the proof window opens.
    expandGateBandsGPU((uint64_t*)(d_aux_trace + offsetStage1Extended), nCols,
                       1ULL << setupCtx->starkInfo.starkStruct.nBits,
                       air_instance_info->d_gate_bands, air_instance_info->n_gate_bands,
                       air_instance_info->gate_band_aux,
                       air_instance_info->gate_band_family,
                       sd.d_gate_band_scratch, stream);
    TimerStopGPU(timer, STARK_GPU_WITNESS);
    
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    // Stage publics into the per-stream pinned region for an async copy (no stream
    // sync); reuse gated by end_event on stream reselect. Runtime check survives NDEBUG.
    if (setupCtx->starkInfo.nPublics > PINNED_AUX_VALUES_MAX) {
        zklog.error("gen_recursive_proof_gpu: nPublics " + std::to_string(setupCtx->starkInfo.nPublics) +
                    " exceeds PINNED_AUX_VALUES_MAX " + std::to_string(PINNED_AUX_VALUES_MAX));
        exitProcess();
    }
    // Pinned staging parity, as in gen_proof_gpu. `sd.recursive` is the STREAM class: on the
    // shared basic stream this recursive proof rides the ring; a dedicated recursive-class
    // stream has no ring (launchSeq stays 0), so it always uses slot 0.
    const uint32_t pipeSlot = (d_buffers->pipelineMode && !d_buffers->streamsData[streamId].recursive)
        ? (uint32_t)(d_buffers->streamsData[streamId].launchSeq & 1) : 0;
    Goldilocks::Element *pinned_publics = d_buffers->streamsData[streamId].pinned_aux_values
        + (uint64_t)pipeSlot * PINNED_AUX_VALUES_MAX;
    memcpy(pinned_publics, pPublicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element));
    CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)(d_aux_trace + offsetPublicInputs), pinned_publics, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));

    gl64_t *d_const_pols = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_pols_offset;
    gl64_t *d_const_tree;
    if (air_instance_info->stored_tree) {
        // Preallocated in the const buffer, so it is in place unconditionally.
        d_const_tree = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_tree_offset;
        reuse_const_tree = reuse_constants;
    } else {
        // find(), not operator[]: a preallocate-layout starkinfo has NO aux tree slot, and
        // operator[] would silently insert offset 0 -- the tree would then be written over the
        // aux BASE.
        auto itConstTree = setupCtx->starkInfo.mapOffsets.find(std::make_pair("const", true));
        if (itConstTree == setupCtx->starkInfo.mapOffsets.end()) {
            zklog.error("air " + std::to_string(airgroupId) + ":" + std::to_string(airId) +
                        " has no aux const-tree slot (preallocate layout) but stored_tree is false");
            exitProcess();
        }
        uint64_t offsetConstTree = itConstTree->second;
        d_const_tree = d_aux_trace + offsetConstTree;

        // calculateFixedExtended airs rebuild the tree inside genProof_gpu (same flag, same reuse
        // gate); the rest still upload the consttree file. Either way ("const", true) holds this
        // slot's tree on exit.
        if (!reuse_const_tree && !setupCtx->starkInfo.calculateFixedExtended) {
            load_and_copy_to_device_in_chunks(d_buffers, constTreePath, (uint8_t*)d_const_tree, sizeConstTree, streamId);
        }
    }
    sd.constTreeResident = true;

    // See gen_proof_gpu: the tree-aware flag, not the slot one.
    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, constTreePath, streamId, instanceId, d_buffers, air_instance_info, false, timer, stream, true, reuse_const_tree);
    if (d_buffers->pipelineMode && !sd.recursive) {
        // Depth-2 in-flight on the shared stream: snapshot the completion into the ring
        // (the per-stream sd fields will be overwritten by the next launch) and let the
        // harvester writeProof + fire the callback with the REAL proof type.
        std::lock_guard<std::mutex> plk(sd.pipeMutex);
        StreamData::PipelineSlot &ps = sd.pipeSlots[(sd.pipeHead + sd.pipeCount) % 2];
        ps.instanceId = (int64_t)instanceId;
        ps.airgroupId = airgroupId;
        ps.airId = airId;
        ps.pSetupCtx = pSetupCtx_;
        ps.proofBuffer = proofBuffer;
        ps.proofFile = string(proof_file);
        ps.proofType = string(proofType);
        ps.pinnedProof = sd.pinned_buffer_proof + (uint64_t)(sd.launchSeq & 1) * sd.maxProofSize;
        ps.timer = &sd.curTimer();
        CHECKCUDAERR(cudaEventRecord(ps.done, stream));
        sd.pipeCount++;
        sd.launchSeq++;
    }
    cudaEventRecord(d_buffers->streamsData[streamId].end_event, stream);
    d_buffers->streamsData[streamId].status = 2;
    return streamId;
}

void calculate_const_tree_fixed_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, char *proofType, void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint32_t streamId = selectStream(d_buffers, airgroupId, airId, proofType, false, false);
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].curTimer();

    auto key = std::make_pair(airgroupId, airId);
    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][string(proofType)][gpuLocalId];

    if (air_instance_info->stored_tree) {
        // The stream was reserved by selectStream (status=1); returning without
        // releasing it would leak the slot for the process lifetime (selectStream
        // never considers status==1 eligible), eventually starving the pool.
        d_buffers->streamsData[streamId].mutex_stream_selection.lock();
        d_buffers->streamsData[streamId].reset(false);
        d_buffers->streamsData[streamId].mutex_stream_selection.unlock();
        return;
    }

    StreamData &sd = d_buffers->streamsData[streamId];
    sd.airgroupId = airgroupId;
    sd.airId = airId;
    sd.proofType = string(proofType);
    sd.witnessResident = false;
    sd.adoptFixedSlot(air_instance_info->const_pols_offset,
                      setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)], true, "");

    gl64_t *d_const_pols = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_pols_offset;

    gl64_t * d_aux_trace = d_buffers->streamsData[streamId].recursive
        ? (gl64_t *)d_buffers->d_aux_traceAggregation[gpuLocalId][d_buffers->streamsData[streamId].localStreamId]
        : d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];

    uint64_t N = 1 << setupCtx->starkInfo.starkStruct.nBits;
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    auto itConstTree = setupCtx->starkInfo.mapOffsets.find(std::make_pair("const", true));
    if (itConstTree == setupCtx->starkInfo.mapOffsets.end()) {
        zklog.error("const-tree rebuild: air " + std::to_string(airgroupId) + ":" + std::to_string(airId) +
                    " has no aux const-tree slot (preallocate layout)");
        exitProcess();
    }
    uint64_t offsetConstTree = itConstTree->second;
    Goldilocks::Element *packed_const_pols = (Goldilocks::Element *)d_const_pols;
    Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;
    uint64_t* d_num_packed_words = (uint64_t*) d_const_pols;
    unpack_fixed(d_num_packed_words, (uint64_t*)(packed_const_pols + 1), (uint64_t*)(packed_const_pols + 1 + setupCtx->starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants, N, stream, timer);
    
    gl64_t *d_const_tree = d_aux_trace + offsetConstTree;
    extendAndMerkelizeFixed(*setupCtx, d_const_pols_unpacked, (Goldilocks::Element *)d_const_tree, true, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    // Both regions now hold this slot's data, so any air in the group can skip both.
    sd.constTreeResident = true;
    sd.status = 3;
}

void tile_const_pols_gpu(void *pStarkinfo, void *pConstPols, char *constFile, void *pConstTree, char *constTreeFile, void *unified_buffer_gpu) {

    StarkInfo &starkInfo = *(StarkInfo *)pStarkinfo;
    uint64_t *h_constPols = (uint64_t *)pConstPols;
    uint64_t *h_constTree = (uint64_t *)pConstTree;

    uint64_t N = (1 << starkInfo.starkStruct.nBits);
    uint64_t NExtended = (1 << starkInfo.starkStruct.nBitsExt);
    uint64_t nConst = starkInfo.nConstants;
    uint64_t sizeConstPols = N * nConst * sizeof(Goldilocks::Element);
    uint64_t sizeConstPolsExtended = NExtended * nConst * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&starkInfo) * sizeof(Goldilocks::Element);
    uint64_t sizeConstOnlyTree = sizeConstTree - sizeConstPolsExtended;

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_helper;
    gl64_t *d_helperAux;
    if (unified_buffer_gpu == nullptr) {
        CHECKCUDAERR(cudaMalloc(&d_helper, sizeConstPolsExtended));
        CHECKCUDAERR(cudaMalloc(&d_helperAux, sizeConstPolsExtended));
    } else {
        gl64_t * d_unifiedBuffer = (gl64_t *)unified_buffer_gpu;
        d_helper = d_unifiedBuffer;
        d_helperAux = d_unifiedBuffer + sizeConstPolsExtended;
    }

    Goldilocks::Element *h_helperTiled = (Goldilocks::Element *)malloc(sizeConstTree);

    dim3 gridSize;
    dim3 blockSize(32,32,1);
    
    // ConstPols 
    CHECKCUDAERR(cudaMemcpy(d_helper, h_constPols, sizeConstPols, cudaMemcpyHostToDevice));
    gridSize = dim3((N + blockSize.x - 1) / blockSize.x, (nConst + blockSize.y - 1) / blockSize.y, 1);
    fromRowMajorToColMajor<<<gridSize, blockSize, 0, stream>>>(N, nConst, (uint64_t*)d_helper, (uint64_t*)d_helperAux, fixedLayout());
    CHECKCUDAERR(cudaMemcpy(h_helperTiled, d_helperAux, sizeConstPols, cudaMemcpyDeviceToHost));
    ofstream fw(constFile, std::ios::out | std::ios::binary);
    if (!fw.is_open()) {
        zklog.error("Failed to open file for writing: " + string(constFile));
        exitProcess();
    }
    fw.write((const char *)h_helperTiled, sizeConstPols);
    fw.close();

    // ConstTree
    CHECKCUDAERR(cudaMemcpy(d_helper, h_constTree, sizeConstPolsExtended, cudaMemcpyHostToDevice));
    gridSize = dim3((NExtended + blockSize.x - 1) / blockSize.x, (nConst + blockSize.y - 1) / blockSize.y, 1);
    fromRowMajorToColMajor<<<gridSize, blockSize, 0, stream>>>(NExtended, nConst, (uint64_t*)d_helper, (uint64_t*)d_helperAux, fixedLayout());
    CHECKCUDAERR(cudaMemcpy(h_helperTiled, d_helperAux, sizeConstPolsExtended, cudaMemcpyDeviceToHost));
    memcpy(h_helperTiled + (sizeConstPolsExtended / sizeof(Goldilocks::Element)), (uint8_t*)pConstTree + sizeConstPolsExtended, sizeConstOnlyTree);
    ofstream fwTree(constTreeFile, std::ios::out | std::ios::binary);
    if (!fwTree.is_open()) {
        zklog.error("Failed to open file for writing: " + string(constTreeFile));
        exitProcess();
    }
    fwTree.write((const char *)h_helperTiled, sizeConstTree);
    fwTree.close();

    free(h_helperTiled);
    if (unified_buffer_gpu == nullptr) {
        CHECKCUDAERR(cudaFree(d_helper));
        CHECKCUDAERR(cudaFree(d_helperAux));
    }
    CHECKCUDAERR(cudaStreamDestroy(stream));

}

void *gen_device_buffers_recursivef_gpu(void *pSetupCtx_, uint64_t proverBufferSize, void *d_commit_buffer_,  char* verkey) {
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    uint32_t gpuId = 0;
    DeviceCommitBuffers *d_commit_buffer = (DeviceCommitBuffers *)d_commit_buffer_;
    if (d_commit_buffer != nullptr) {
        gpuId = d_commit_buffer->my_gpu_ids[0];
    }

    // Scope sppark's GPU registry to this rank's devices before ngpus() below builds it, else a
    // standalone SNARK-wrap process (no prior gen_device_buffers_gpu) probes every GPU on the node.
    {
        int ords[32];
        uint32_t n = (d_commit_buffer != nullptr) ? d_commit_buffer->n_gpus : 1;
        if (n > 32) n = 32;
        for (uint32_t i = 0; i < n; i++)
            ords[i] = (int)((d_commit_buffer != nullptr) ? d_commit_buffer->my_gpu_ids[i] : gpuId);
        sppark_set_visible_devices(ords, (int)n);
    }

    // Force sppark's lazy GPU registry to init now, while we control the device. Its first entry
    // point builds a static gpus_t that ends with cudaSetDevice(0), clobbering the caller's device;
    // triggering it here and restoring the device makes later cudaSetDevice(N) stick.
    (void)ngpus();
    cudaSetDevice(gpuId);

    DeviceRecursiveFBuffers *d_buffers = new DeviceRecursiveFBuffers();
    d_buffers->gpuId = gpuId;

    // Initialize BN128 Poseidon GPU constants for merkletree and transcript
    PoseidonBN128GPU::initGPUConstants(&gpuId, 1);
    uint64_t transcriptArity = setupCtx->starkInfo.starkStruct.merkleTreeCustom ? setupCtx->starkInfo.starkStruct.merkleTreeArity : 16;
    TranscriptBN128_GPU::init_const(&gpuId, 1, transcriptArity);

    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
    uint64_t sizeAuxTrace = proverBufferSize * sizeof(Goldilocks::Element);

    if (d_commit_buffer_ == nullptr) {
        // Allocate new device buffers
        d_buffers->owns_aux_trace = true;
        d_buffers->owns_const_tree = true;
        CHECKCUDAERR(cudaMalloc(&d_buffers->d_aux_trace, sizeAuxTrace));
        CHECKCUDAERR(cudaMalloc(&d_buffers->d_const_tree, sizeConstTree));
        d_buffers->aux_trace_size = sizeAuxTrace;
    } else {
        DeviceCommitBuffers *d_commit_buffer = (DeviceCommitBuffers *)d_commit_buffer_;
        gl64_t *d_unifiedBuffer = d_commit_buffer->gpuMemoryBuffer[d_commit_buffer->gpus_g2l[gpuId]];
        // Always reuse first buffer for d_aux_trace
        d_buffers->owns_aux_trace = false;
        d_buffers->owns_const_tree = false;
        d_buffers->d_const_tree = d_unifiedBuffer;
        d_buffers->d_aux_trace = d_unifiedBuffer + (sizeConstTree / 8);
    }

    RawFr rawFr;
    RawFr::Element verkeyElement;
    rawFr.fromString(verkeyElement, verkey);
    
    // Allocate GPU memory and copy verkey to device
    CHECKCUDAERR(cudaMalloc(&d_buffers->d_verkey, sizeof(RawFr::Element)));
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_verkey, &verkeyElement, sizeof(RawFr::Element), cudaMemcpyHostToDevice));

    return (void*)d_buffers;
}   

void load_fixed_pols_recursivef_gpu(void *pSetupCtx_, void *pConstTree, void *d_buffers_) {
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    DeviceRecursiveFBuffers *d_buffers = (DeviceRecursiveFBuffers *)d_buffers_;
    
    uint32_t gpuId = d_buffers->gpuId;
    cudaSetDevice(gpuId);

    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);

    gl64_t * d_const_tree = (gl64_t *)d_buffers->d_const_tree;
    uint8_t * pinnedBuffer = d_buffers->pinnedBufferConstTree;
    uint64_t pinnedBufferSize = d_buffers->pinnedBufferSize;
    cudaStream_t stream = d_buffers->stream_const_tree;
    // Reset const tree loaded flag before starting a new copy
    d_buffers->const_tree_loaded.store(false, std::memory_order_relaxed);
    
    // Copy const tree to device (synchronizes internally)
    copy_to_device_in_chunks((const uint8_t*)pConstTree, (uint8_t*)d_const_tree, sizeConstTree, pinnedBuffer, pinnedBufferSize, stream);
    CHECKCUDAERR(cudaGetLastError());
    
    // Signal that const tree copy is complete
    d_buffers->const_tree_loaded.store(true, std::memory_order_release);
    
}

void free_device_buffers_recursivef_gpu(void *d_buffers_) {
    DeviceRecursiveFBuffers *d_buffers = (DeviceRecursiveFBuffers *)d_buffers_;
    cudaSetDevice(d_buffers->gpuId);
    // Fence work queued on this context before freeing the memory it references, bounded so a wedged
    // context can't hang teardown forever. All recursivef work runs on these two streams.
    cudaStream_t recursivef_streams[2] = { d_buffers->stream, d_buffers->stream_const_tree };
    for (cudaStream_t st : recursivef_streams) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (cudaStreamQuery(st) == cudaErrorNotReady &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            printf("[teardown] WARNING: a recursivef stream did not drain within 10s; freeing anyway\n");
            fflush(stdout);
        }
        cudaGetLastError();  // clear the sticky status left by the last query
    }
    if (d_buffers->owns_const_tree) {
        CHECKCUDAERR(cudaFree(d_buffers->d_const_tree));
    }
    if (d_buffers->owns_aux_trace) {
        CHECKCUDAERR(cudaFree(d_buffers->d_aux_trace));
    }
    delete d_buffers;
}

void *gen_recursive_proof_final_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers_) {
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    DeviceRecursiveFBuffers *d_buffers = (DeviceRecursiveFBuffers *)d_buffers_;
    
    uint32_t gpuId = d_buffers->gpuId;
    cudaSetDevice(gpuId);

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t sizeWitness = N * nCols * sizeof(Goldilocks::Element);
    uint64_t sizePublicInputs = setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element);

    gl64_t* d_aux_trace = d_buffers->d_aux_trace;
    uint8_t* pinnedBuffer = d_buffers->pinnedBuffer;
    uint64_t pinnedBufferSize = d_buffers->pinnedBufferSize;

    dim3 gridSize;
    dim3 blockSize(32,32,1);

    // Copy and tile witness
    uint64_t offsetCm1Extended = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t offsetCm1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    gl64_t * d_witness_temp = d_aux_trace + offsetCm1Extended;
    gl64_t * d_witness = d_aux_trace + offsetCm1;
    copy_to_device_in_chunks((const uint8_t*)witness, (uint8_t*)d_witness_temp, sizeWitness, pinnedBuffer, pinnedBufferSize, d_buffers->stream);
    gridSize = dim3((N + blockSize.x - 1) / blockSize.x, (nCols + blockSize.y - 1) / blockSize.y, 1);
    fromRowMajorToColMajor<<<gridSize, blockSize, 0, d_buffers->stream>>>(N, nCols, (uint64_t*)d_witness_temp, (uint64_t*)d_witness, resolveLayout(setupCtx->starkInfo.starkStruct.nBits, nCols));
    CHECKCUDAERR(cudaGetLastError());

    // Copy public inputs
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetPublicInputs, (const gl64_t*)pPublicInputs, sizePublicInputs, cudaMemcpyHostToDevice, d_buffers->stream));

    uint64_t nConst = setupCtx->starkInfo.nConstants;
    uint64_t sizeConstPols = N * nConst * sizeof(Goldilocks::Element);
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    copy_to_device_in_chunks((const uint8_t*)pConstPols, (uint8_t*)(d_aux_trace + offsetConstPols), sizeConstPols, pinnedBuffer, pinnedBufferSize, d_buffers->stream);
    CHECKCUDAERR(cudaGetLastError());

    void* result = genRecursiveProofBN128_gpu(*setupCtx, airgroupId, airId, instanceId, (Goldilocks::Element *)d_aux_trace, (Goldilocks::Element *)pPublicInputs, string(proof_file), d_buffers);

    cudaStreamSynchronize(d_buffers->stream);

    return result;
}

uint64_t commit_witness_gpu(void *pSetupCtx_, void *params_, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers_, char *customCommitsFixedPath) {
    // Set by the custom-commit rebuild below; the matching wait sits before its first reader.
    bool customFixedRebuilt = false;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    // Count this thread as inside device work so a concurrent teardown waits for it.
    InFlightScope in_flight(d_buffers);
    uint32_t streamId = selectStream(d_buffers, airgroupId, airId, "basic");
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];

    // Check reuse against the stream's prior context before it is overwritten below.
    StreamData &sd = d_buffers->streamsData[streamId];
    bool reuse_custom_fixed = sd.airgroupId == airgroupId && sd.airId == airId && sd.proofType == string("basic");

    sd.root = root;
    sd.instanceId = instanceId;
    sd.airgroupId = airgroupId;
    sd.airId = airId;
    sd.proofType = "witness";
    // A stream sized for the contributions footprint can be too small for this air's proof, and a
    // resident witness pins gen_proof here (skip_recalculation). Only claim residency if the proof fits;
    // otherwise the instance takes the normal recompute path and re-uploads its trace.
    sd.witnessResident = sd.auxTraceCapacity >= setupCtx->starkInfo.mapTotalN;

    proofman_sumcheck_set_context(instanceId, airgroupId, airId);

    auto key = std::make_pair(airgroupId, airId);
    cudaSetDevice(gpuId);
    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key]["basic"][gpuLocalId];

    uint64_t N = 1 << setupCtx->starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx->starkInfo.starkStruct.nBitsExt;
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t arity = setupCtx->starkInfo.starkStruct.merkleTreeArity;
    uint64_t nBits = setupCtx->starkInfo.starkStruct.nBits;
    uint64_t nBitsExt = setupCtx->starkInfo.starkStruct.nBitsExt;

    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].curTimer();
    TimerStartGPU(timer, STARK_GPU_COMMIT);

#ifdef USE_CUDA_GRAPH
    // Contributions-phase capture regions: this path runs outside genProof_gpu, so bind
    // the thread-local cache to this stream's for the call. All host staging into pinned
    // buffers stays OUTSIDE the regions (it must re-run before every replay); the regions
    // hold only stream work with per-(air,stream) stable arguments. A body that hits the
    // interpreter fallback poisons its capture (see stageExpsSlot).
    cudagraph::current() = d_buffers->streamsData[streamId].graph_cache.get();
    struct WitnessGraphCtxGuard {
        ~WitnessGraphCtxGuard() { cudagraph::current() = nullptr; }
    } witnessGraphCtxGuard;
#endif
    // Key tags 0x57455843 "WEXC" / 0x574c4445 "WLDE" — full tag table at graphCtxId in gen_proof.cuh.
    const uint64_t witnessCtxId = (uint64_t)(uintptr_t)setupCtx;

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];
    uint64_t sizeTrace = N * nCols * sizeof(Goldilocks::Element);
    uint64_t offsetStage1Extended = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t total_size = (d_buffers->packedTrace && air_instance_info->is_packed) ? air_instance_info->num_packed_words * N * sizeof(Goldilocks::Element) : sizeTrace;
    uint64_t *dst = (uint64_t*)(d_aux_trace + offsetStage1Extended);
    copy_to_device_in_chunks(d_buffers, params->trace, dst, total_size, streamId, timer);
    PROOFMAN_SUMCHECK("contrib_before_unpack", dst, total_size / sizeof(uint64_t), stream);

    uint64_t tree_size = MerkleTreeGL::getTreeNumElements(NExtended, arity);

    uint64_t offset_src = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offset_dst = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t offset_mt = setupCtx->starkInfo.mapOffsets[make_pair("mt1", true)];

    Goldilocks::Element *pNodes = (Goldilocks::Element*)d_aux_trace + offset_mt;
    NTTGoldilocksGPU ntt;

    if (d_buffers->packedTrace && air_instance_info->is_packed) {
        unpack_trace(air_instance_info, (uint64_t *)(d_aux_trace + offset_dst), (uint64_t *)(d_aux_trace + offset_src), nCols, N, stream, timer);
    } else {
        fromRowMajorToColMajor(N, nCols, (gl64_t *)(d_aux_trace + offset_dst), (gl64_t *)(d_aux_trace + offset_src), resolveLayout(nBits, nCols), stream);
    }
    PROOFMAN_SUMCHECK("contrib_after_unpack", d_aux_trace + offset_src, N * nCols, stream);

    uint64_t nWitnessHints = setupCtx->expressionsBin.getNumberHintIdsByName("witness_calc");
    // The trace write above lands inside a larger previous air's const region, so the claim
    // has to be refreshed either way: adopted below when this path unpacks, dropped when it
    // does not. Leaving a stale claim would let a later proof of that air skip its unpack --
    // and the new slot-keyed affinity actively steers it back to this stream.
    if (nWitnessHints == 0) sd.dropFixedSlot();
    if(nWitnessHints > 0) {
        uint64_t countId = 0;
        uint64_t offsetCm1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
        uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
        uint64_t offsetAirgroupValues = setupCtx->starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)];
        uint64_t offsetAirValues = setupCtx->starkInfo.mapOffsets[std::make_pair("airvalues", false)];
        uint64_t offsetProofValues = setupCtx->starkInfo.mapOffsets[std::make_pair("proofvalues", false)];

        uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
        gl64_t *d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
        gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];
        Goldilocks::Element *packed_const_pols = (Goldilocks::Element *)d_const_pols;
        Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;
        uint64_t* d_num_packed_words = (uint64_t*) d_const_pols;
        // Claims the slot but not constTreeResident: this never touches the const tree.
        if (!sd.adoptFixedSlot(air_instance_info->const_pols_offset, offsetConstPols, false, "")
            || setupCtx->starkInfo.constPolsAliasTree) {
            unpack_fixed(d_num_packed_words, (uint64_t*)(packed_const_pols + 1), (uint64_t*)(packed_const_pols + 1 + setupCtx->starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants, N, stream, timer);
            CHECKCUDAERR(cudaGetLastError());
        }

        // Rebuild now, wait just before the commit body below reads it.
        if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0 && !reuse_custom_fixed) {
            Goldilocks::Element *pCustomCommitsFixedDst = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            rebuildCustomCommitsFixed(d_buffers, setupCtx, air_instance_info, gpuLocalId, pCustomCommitsFixedDst, sd, timer);
            customFixedRebuilt = true;
        }

        size_t totalCopySize = 0;
        totalCopySize += setupCtx->starkInfo.nPublics;
        totalCopySize += setupCtx->starkInfo.proofValuesSize;
        totalCopySize += setupCtx->starkInfo.airgroupValuesSize;
        totalCopySize += setupCtx->starkInfo.airValuesSize;

        // Stage into the per-stream pinned region for an async copy. Hard runtime check, not assert:
        // must survive NDEBUG release builds, else it silently overflows the fixed pinned buffer.
        if (totalCopySize > PINNED_AUX_VALUES_MAX) {
            zklog.error("commit_witness_gpu: aux_values size " + std::to_string(totalCopySize) +
                        " exceeds PINNED_AUX_VALUES_MAX " + std::to_string(PINNED_AUX_VALUES_MAX));
            exitProcess();
        }
        Goldilocks::Element *aux_values = d_buffers->streamsData[streamId].pinned_aux_values;
        uint64_t offset = 0;
        memcpy(aux_values + offset, params->publicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element));
        offset += setupCtx->starkInfo.nPublics;
        if (setupCtx->starkInfo.proofValuesSize > 0) {
            memcpy(aux_values + offset, params->proofValues, setupCtx->starkInfo.proofValuesSize * sizeof(Goldilocks::Element));
            offset += setupCtx->starkInfo.proofValuesSize;
        }
        if (setupCtx->starkInfo.airgroupValuesSize > 0) {
            memcpy(aux_values + offset, params->airgroupValues, setupCtx->starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element));
            offset += setupCtx->starkInfo.airgroupValuesSize;
        }
        if (setupCtx->starkInfo.airValuesSize > 0) {
            memcpy(aux_values + offset, params->airValues, setupCtx->starkInfo.airValuesSize * sizeof(Goldilocks::Element));
            offset += setupCtx->starkInfo.airValuesSize;
        }

        StepsParams h_params = {
            trace : (Goldilocks::Element *)d_aux_trace + offsetCm1,
            aux_trace : (Goldilocks::Element *)d_aux_trace,
            publicInputs : (Goldilocks::Element *)d_aux_trace + offsetPublicInputs,
            proofValues : (Goldilocks::Element *)d_aux_trace + offsetProofValues,
            challenges : nullptr,
            airgroupValues : (Goldilocks::Element *)d_aux_trace + offsetAirgroupValues,
            airValues : (Goldilocks::Element *)d_aux_trace + offsetAirValues,
            evals : nullptr,
            xDivXSub : nullptr,
            pConstPolsAddress: d_const_pols_unpacked,
            pConstPolsExtendedTreeAddress: nullptr,
            pCustomCommitsFixed: setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0
                ? (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)]
                : nullptr,
        };

        // Host staging BEFORE the capture region: replays skip the region body, so pinned
        // content must already be correct when the graph's H2D nodes read it.
        StepsParams *params_pinned = d_buffers->streamsData[streamId].pinned_params;
        memcpy(params_pinned, &h_params, sizeof(StepsParams));
        StepsParams *d_params =  d_buffers->streamsData[streamId].params;

        ExpsArguments *d_expsArgs = d_buffers->streamsData[streamId].d_expsArgs;
        DestParamsGPU *d_destParams = d_buffers->streamsData[streamId].d_destParams;
        Goldilocks::Element *pinned_exps_params = d_buffers->streamsData[streamId].pinned_buffer_exps_params;
        Goldilocks::Element *pinned_exps_args = d_buffers->streamsData[streamId].pinned_buffer_exps_args;

        auto witnessExprBody = [&] {
            CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)(d_aux_trace + offsetPublicInputs), aux_values, totalCopySize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
            CHECKCUDAERR(cudaMemcpyAsync(d_params, params_pinned, sizeof(StepsParams), cudaMemcpyHostToDevice, stream));
            calculateWitnessExpr_gpu(*setupCtx, h_params, d_params, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
        };
        cudagraph::run(cudagraph::key(0x57455843ULL ^ witnessCtxId), countId, stream, witnessExprBody);
    }

    if (customFixedRebuilt) CHECKCUDAERR(cudaStreamWaitEvent(stream, sd.customFixedDone, 0));

    PROOFMAN_SUMCHECK("contrib_before_lde", d_aux_trace + offset_src, N * nCols, stream);
    auto commitLdeBody = [&] {
        ntt.LDE(d_aux_trace, offset_dst, d_aux_trace, offset_src, nBits, nBitsExt, nCols, timer, stream, true, (gl64_t*)pNodes, setupCtx->starkInfo.getNumNodesMT(NExtended));
        TimerStartCategoryGPU(timer, MERKLE_TREE);
        // cm1 contribution commit: read the extended trace in the layout the LDE wrote (resolveLayout on
        // the small domain). When tiled AIRs existed, hardcoding ColMajor here made the tiled contribution
        // root read uninitialised in-tile padding -> non-det; keep the shared predicate.
        buildMerkleTreeGPU(arity, (uint64_t*)pNodes, (uint64_t*)(d_aux_trace + offset_dst), nCols, 1ULL << nBitsExt, resolveLayout(nBits, nCols), stream);
        TimerStopCategoryGPU(timer, MERKLE_TREE);
        CHECKCUDAERR(cudaMemcpyAsync(d_buffers->streamsData[streamId].pinned_buffer_proof, &pNodes[tree_size - HASH_SIZE], HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    };
    uint64_t commitLdeCountId = 0;   // no expression launches in this body; dummy cursor
    cudagraph::run(cudagraph::key(0x574c4445ULL ^ witnessCtxId), commitLdeCountId, stream, commitLdeBody);
    PROOFMAN_SUMCHECK("contrib_after_lde", d_aux_trace + offset_dst, NExtended * nCols, stream);
    TimerStopGPU(timer, STARK_GPU_COMMIT);
    cudaEventRecord(d_buffers->streamsData[streamId].end_event, stream);
    d_buffers->streamsData[streamId].status = 2;
    return streamId;
}

void get_commit_root(DeviceCommitBuffers *d_buffers, uint64_t streamId) {

    Goldilocks::Element *root = (Goldilocks::Element *)d_buffers->streamsData[streamId].root;
    memcpy((Goldilocks::Element *)root, d_buffers->streamsData[streamId].pinned_buffer_proof, HASH_SIZE * sizeof(uint64_t));
    uint64_t instanceId = d_buffers->streamsData[streamId].instanceId;
    uint64_t airgroupId = d_buffers->streamsData[streamId].airgroupId;
    uint64_t airId = d_buffers->streamsData[streamId].airId;
    closeStreamTimer(d_buffers->streamsData[streamId].curTimer(), instanceId, airgroupId, airId, false);
    // Contributions commit_root does NOT fire proof_done_callback: that decrement is owned by the
    // Prove-path accounting, and firing it here could hit the NULL-callback window and wedge proofs_pending.
}

void init_gpu_setup_gpu(uint64_t arity) {
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    cudaSetDevice(deviceId);
    uint32_t my_gpu_ids[1] = {(uint32_t)deviceId};

    // Initialize Poseidon1 + Poseidon2 GPU constants unconditionally.
    switch (arity) {
        case 2:
            PoseidonGoldilocksGPU<8>::initConstants(my_gpu_ids, 1);
            Poseidon2GoldilocksGPU<8>::initConstants(my_gpu_ids, 1);
            break;
        case 3:
            PoseidonGoldilocksGPU<12>::initConstants(my_gpu_ids, 1);
            Poseidon2GoldilocksGPU<12>::initConstants(my_gpu_ids, 1);
            break;
        case 4:
            PoseidonGoldilocksGPU<16>::initConstants(my_gpu_ids, 1);
            Poseidon2GoldilocksGPU<16>::initConstants(my_gpu_ids, 1);
            break;
        default:
            zklog.error("init_gpu_setup_gpu: supports merkle tree arity 2, 3 or 4");
            exit(1);
    }
}

void prepare_blocks_gpu(uint64_t *pol, uint64_t N, uint64_t nCols, void *unified_buffer_gpu) {
    gl64_t *d_pol;
    gl64_t *d_aux;
    if (unified_buffer_gpu == nullptr) {
        CHECKCUDAERR(cudaMalloc(&d_pol, N * nCols * sizeof(gl64_t)));
        CHECKCUDAERR(cudaMalloc(&d_aux, N * nCols * sizeof(gl64_t)));
    } else {
        gl64_t *d_unifiedBuffer = (gl64_t *)unified_buffer_gpu;
        d_pol = d_unifiedBuffer;
        d_aux = d_unifiedBuffer + (N * nCols);
    }
    cudaMemcpy(d_pol, pol, N * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    TimerGPU timer;
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    cudaSetDevice(deviceId);
    // prepare_blocks transposes const pols into fixedLayout() (ColMajor) on the host -- this is the
    // input layout calculate_const_tree_gpu (via ldeColMajor) expects.
    fromRowMajorToColMajor(N, nCols, d_pol, d_aux, fixedLayout(), stream);

    cudaMemcpy(pol, d_aux, N * nCols * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    if (unified_buffer_gpu == nullptr) {
        CHECKCUDAERR(cudaFree(d_pol));
        CHECKCUDAERR(cudaFree(d_aux));
    }
    cudaStreamDestroy(stream);
}

void write_custom_commit_gpu(void* root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    cudaSetDevice(d_buffers->my_gpu_ids[0]);

    uint64_t N = 1 << nBits;
    uint64_t NExtended = 1 << nBitsExt;

    // Not allocating: only numNodes is wanted, the tree itself stays on the device.
    MerkleTreeGL mt(arity, 0, true, NExtended, nCols);

    uint64_t treeSize = (NExtended * nCols) + mt.numNodes;

    uint32_t streamId = 0;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;

    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];

    gl64_t* d_buffer = d_aux_trace;
    gl64_t* d_customCommitsPols = d_aux_trace + N * nCols;
    gl64_t* d_customCommitsTree = d_customCommitsPols + N * nCols;
    cudaMemset(d_customCommitsTree, 0, treeSize * sizeof(gl64_t));
    cudaMemcpy(d_buffer, buffer, N * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice);

    // Custom commits are a fixed/preprocessed section -> fixedLayout() (ColMajor). Transpose the
    // row-major input into the storage layout.
    fromRowMajorToColMajor(N, nCols, d_buffer, d_customCommitsPols, fixedLayout(), stream);

    NTTGoldilocksGPU ntt;
    Goldilocks::Element *pNodes = (Goldilocks::Element *)&d_customCommitsTree[nCols * NExtended];
    ntt.ldeColMajor((gl64_t *)d_customCommitsTree, (gl64_t *)d_customCommitsPols, nBits, nBitsExt, nCols, stream, false, (gl64_t *)pNodes, mt.numNodes);
    buildMerkleTreeGPU(arity, (uint64_t*)pNodes, (uint64_t*)d_customCommitsTree, nCols, 1ULL << nBitsExt, fixedLayout(), stream);

    // Only the root leaves the device: the file no longer stores the tree it came from.
    Goldilocks::Element *rootGL = (Goldilocks::Element *)root;
    CHECKCUDAERR(cudaMemcpyAsync(rootGL, pNodes + (mt.numNodes - HASH_SIZE), HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    if(std::string(bufferFile) != "") {
        // The packed rows are LOGICAL rows, so this file is layout-agnostic: the same file serves
        // the CPU and GPU provers.
        std::vector<uint64_t> pack_info(nCols, 0);
        uint64_t words_per_row = packWidthsRowMajor((const uint64_t *)buffer, N, nCols, pack_info.data());
        std::vector<uint64_t> packed(N * words_per_row, 0);
        packRowsBits((const uint64_t *)buffer, packed.data(), N, nCols, pack_info.data(), words_per_row);

        std::string buffFile = string(bufferFile);
        ofstream fw(buffFile.c_str(), std::fstream::out | std::fstream::binary);
        writeFileParallel(buffFile, root, 32, 0);
        writeFileParallel(buffFile, &words_per_row, sizeof(uint64_t), 32);
        writeFileParallel(buffFile, pack_info.data(), nCols * sizeof(uint64_t), 40);
        writeFileParallel(buffFile, packed.data(), N * words_per_row * sizeof(uint64_t), 40 + nCols * sizeof(uint64_t));
        fw.close();
    }
}

void calculate_const_tree_gpu(void *pStarkInfo, void *pConstPolsAddress, void *pConstTreeAddress_, void *unified_buffer_gpu) {
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    cudaSetDevice(deviceId);

    StarkInfo &starkInfo = *((StarkInfo *)pStarkInfo);
    assert(starkInfo.starkStruct.verificationHashType == "GL");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    TimerGPU timer;
    TimerStartGPU(timer, STARK_GPU_CONST_TREE);

    uint64_t N = 1 << starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
    MerkleTreeGL mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, true, NExtended, starkInfo.nConstants);
    uint64_t treeSize = (NExtended * starkInfo.nConstants) + mt.numNodes;

    Goldilocks::Element* d_fixedPols;
    Goldilocks::Element* d_fixedTree;
    if (unified_buffer_gpu == nullptr) {
        cudaMalloc((void**)&d_fixedPols, NExtended * starkInfo.nConstants * sizeof(Goldilocks::Element));
        cudaMalloc((void**)&d_fixedTree, treeSize * sizeof(Goldilocks::Element));
    } else {
        Goldilocks::Element *d_unifiedBuffer = (Goldilocks::Element *)unified_buffer_gpu;
        d_fixedPols = d_unifiedBuffer;
        d_fixedTree = d_unifiedBuffer + (NExtended * starkInfo.nConstants);
    }
    
    cudaMemcpy(d_fixedPols, pConstPolsAddress, N * starkInfo.nConstants * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    cudaMemset(d_fixedTree, 0, treeSize * sizeof(Goldilocks::Element));

    NTTGoldilocksGPU ntt;

    Goldilocks::Element *pNodes = d_fixedTree + starkInfo.nConstants * NExtended;
    // Const tree uses fixedLayout() (ColMajor). d_fixedPols is a throwaway copy of the host const
    // pols, so preserve_src=false is fine even where the serial flow runs its iNTT in place on it.
    ntt.ldeColMajor((gl64_t *)d_fixedTree, (gl64_t *)d_fixedPols, starkInfo.starkStruct.nBits, starkInfo.starkStruct.nBitsExt, starkInfo.nConstants, stream, false, (gl64_t *)pNodes, mt.numNodes);
    buildMerkleTreeGPU(starkInfo.starkStruct.merkleTreeArity, (uint64_t*)pNodes, (uint64_t*)d_fixedTree, starkInfo.nConstants, 1ULL << starkInfo.starkStruct.nBitsExt, fixedLayout(), stream);

    Goldilocks::Element *pConstTreeAddress = (Goldilocks::Element *)pConstTreeAddress_;
    cudaMemcpy(pConstTreeAddress, d_fixedTree, treeSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
    if (unified_buffer_gpu == nullptr) {
        cudaFree(d_fixedPols);
        cudaFree(d_fixedTree);
    }
    TimerStopGPU(timer, STARK_GPU_CONST_TREE);
    cudaStreamDestroy(stream);
}

uint64_t check_device_memory_gpu(uint32_t node_rank, uint32_t node_size)
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error getting device count: "
                  << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 0;
    }

    uint64_t min_free_mem = std::numeric_limits<uint64_t>::max();
    bool multi_gpu_per_process = deviceCount >= (int)node_size;
    uint32_t n_gpus;
    
    if (multi_gpu_per_process) {
        n_gpus = (uint32_t)deviceCount / node_size;
        uint32_t first_gpu = node_rank * n_gpus;
        
        for (uint32_t i = 0; i < n_gpus; i++) {
            uint32_t device_id = first_gpu + i;
            
            if (device_id >= (uint32_t)deviceCount) {
                std::cerr << "Invalid device_id " << device_id
                          << " (deviceCount=" << deviceCount << ")"
                          << std::endl;
                continue;
            }
            
            cudaSetDevice(device_id);
            
            uint64_t freeMem, totalMem;
            err = cudaMemGetInfo(&freeMem, &totalMem);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error on GPU " << device_id << ": "
                          << cudaGetErrorString(err) << std::endl;
                continue;
            }
            
            zklog.info("Process rank " + std::to_string(node_rank) +
                       " - GPU " + std::to_string(device_id) +
                       " [" + std::to_string(i) + "/" + std::to_string(n_gpus) + "]: " +
                       std::to_string(freeMem / (1024.0 * 1024.0 * 1024.0)) + " GB free / " +
                       std::to_string(totalMem / (1024.0 * 1024.0 * 1024.0)) + " GB total");
            
            min_free_mem = std::min(min_free_mem, freeMem);
        }
        
        if (min_free_mem != std::numeric_limits<uint64_t>::max()) {
            zklog.info("Process rank " + std::to_string(node_rank) +
                       ": Using minimum memory across " + std::to_string(n_gpus) +
                       " GPUs: " + std::to_string(min_free_mem / (1024.0 * 1024.0 * 1024.0)) + " GB");
        }
    } else {
        uint32_t device_id = node_rank % deviceCount;
        cudaSetDevice(device_id);
        
        uint64_t freeMem, totalMem;
        err = cudaMemGetInfo(&freeMem, &totalMem);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error on GPU " << device_id << ": "
                      << cudaGetErrorString(err) << std::endl;
            return 0;
        }
        
        zklog.info("Process rank " + std::to_string(node_rank) +
                   " uses shared GPU " + std::to_string(device_id) +
                   ": " + std::to_string(freeMem / (1024.0 * 1024.0 * 1024.0)) + " GB free / " +
                   std::to_string(totalMem / (1024.0 * 1024.0 * 1024.0)) + " GB total");
        
        min_free_mem = freeMem;
    }
    
    // Check if we got valid memory info
    if (min_free_mem == std::numeric_limits<uint64_t>::max()) {
        std::cerr << "Failed to get memory info from any GPU for process rank " 
                  << node_rank << std::endl;
        return 0;
    }

    zklog.info("Minimum free memory available for GPU usage: " + 
               std::to_string(min_free_mem / (1024.0 * 1024.0 * 1024.0)) + " GB");

    return min_free_mem;
}

uint64_t get_num_gpus_gpu() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error getting device count: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    return deviceCount;
}

// Buffer of the caller's CURRENT device: callers that run kernels on a specific GPU bind the
// device first and get that device's buffer.
void *get_unified_buffer_gpu_gpu(void *d_buffers_) {
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return (void *)d_buffers->gpuMemoryBuffer[d_buffers->gpus_g2l[deviceId]];
}

// Buffer of the FIRST GPU (my_gpu_ids[0], not necessarily device 0 — NUMA can reorder), for
// consumers of the acquire/release_first_gpu_buffer borrow.
void *get_first_gpu_buffer_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return nullptr;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return (void *)d_buffers->gpuMemoryBuffer[0];
}

// Byte offset of the aggregation const-pols region (d_constPolsAggregation) from the base of the
// first GPU's unified buffer. 0 on null buffers: the consumer reloads when `used >= offset`, so
// failing toward 0 yields a redundant reload rather than a silently corrupted proof.
uint64_t get_const_pols_aggregation_offset_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return 0;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return (uint64_t)((uint8_t *)d_buffers->d_constPolsAggregation[0] -
                      (uint8_t *)d_buffers->gpuMemoryBuffer[0]);
}

uint64_t get_unified_buffer_gpu_size_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return 0;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return d_buffers->unifiedBufferSize;
}

uint64_t get_stream_commit_slots_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return 0;
    return ((DeviceCommitBuffers *)d_buffers_)->streamCommitSlots;
}

uint64_t get_stream_commit_floor_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return UINT64_MAX;
    return ((DeviceCommitBuffers *)d_buffers_)->streamCommitFloorBytes;
}

// Byte size of one streaming-commit slot for the given packed-AIR shape (the
// layout in streamCommitSlotElems). 0 = shape not slot-committable;
uint64_t stream_commit_slot_bytes_gpu(uint64_t nBits, uint64_t nBitsExt,
                                      uint64_t nCols, uint64_t wordsPerRow) {
    if (nCols == 0 || nCols > SC_MAX_COLS || nBitsExt <= nBits || wordsPerRow == 0) return 0;
    StreamCommitDims dims{nBits, nBitsExt, nCols, wordsPerRow};
    const StreamCommitHash hash = (get_hash_family() == HashFamily::Blake3)
                                      ? StreamCommitHash::Blake3
                                      : StreamCommitHash::Poseidon1;
    return streamCommitSlotElems(dims, hash) * sizeof(Goldilocks::Element);
}

// Enable nSlots streaming-commit slots of slotBytes each (FIRST GPU only --
// the borrowable one). Called by proofman after buffer allocation with the
// DERIVED slot size (stream_commit_slot_bytes over the eligible AIRs): carves
// the slots top-down from the const-pols aggregation offset, flags the legacy
// streams whose aux regions they overlap, and creates one lowest-priority
// stream per slot. The byte offset where the lowest slot starts is the
// "floor": slots live above it, and gpu-mops is only ever handed the region
// below it (get_first_gpu_buffer clamps the borrowed size to it), which is
// what lets commits run while the buffer is borrowed.
//
// LOWEST priority streams: during the gpu-mops borrow window the commit
// kernels saturate the SMs back-to-back; at default priority they starve the
// (light) mops kernels and stretch the whole window. Low priority makes the
// device dispatch mops first at each block boundary, so commits fill the true
// gaps instead of creating a queue in front of mops.
void configure_stream_commit_slots_gpu(void *d_buffers_, uint64_t nSlots, uint64_t slotBytes) {
    if (d_buffers_ == nullptr || nSlots == 0 || slotBytes == 0) return;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers->streamCommitSlots != 0) return;  // already configured

    // Round up to 1 MiB: keeps slot bases 16-byte aligned for vectorized
    // kernel accesses and the carve log readable.
    slotBytes = (slotBytes + ((1ull << 20) - 1)) & ~((1ull << 20) - 1);

    uint64_t totalAuxTraceSize = d_buffers->auxTraceTotalBytes;
    uint64_t constAggOffsetBytes =
        totalAuxTraceSize + (d_buffers->phaseBAliased ? 0 : d_buffers->n_recursive_streams * d_buffers->auxTraceRecursiveBytes) +
        d_buffers->prefetchRegionBytes + d_buffers->mopsFloorPadBytes;
    if (nSlots * slotBytes > constAggOffsetBytes) {
        zklog.error("stream commit slots: " + std::to_string(nSlots) + " x " +
                    std::to_string(slotBytes >> 20) + " MB does not fit below the const pols offset (" +
                    std::to_string(constAggOffsetBytes >> 20) + " MB) -- disabling slots");
        return;
    }
    d_buffers->streamCommitSlotBytes = slotBytes;
    d_buffers->streamCommitFloorBytes = constAggOffsetBytes - nSlots * slotBytes;

    // The slot area may overlap legacy aux-trace regions (recursive ones, and
    // the tail basic ones when it reaches further down). Overlapped first-GPU
    // streams are flagged: slot commits hold their selection mutexes while in
    // flight, so legacy work never runs on an overlapped region concurrently
    // with a slot commit, and the streams return to full rotation the moment
    // the slots go idle.
    uint32_t overlappedBasic = 0, overlappedRecursive = 0;
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        StreamData &sd = d_buffers->streamsData[i];
        if (d_buffers->gpus_g2l[sd.gpuId] != 0) continue;
        // Sizes differ per stream, so the offset is a prefix sum of the carve, not localStreamId * size.
        uint64_t start, end;
        if (sd.recursive) {
            start = (d_buffers->phaseBAliased ? 0 : totalAuxTraceSize) + sd.localStreamId * d_buffers->auxTraceRecursiveBytes;
            end = start + d_buffers->auxTraceRecursiveBytes;
        } else {
            start = 0;
            for (uint32_t j = 0; j < sd.localStreamId; ++j) {
                start += d_buffers->aux_trace_sizes[j] * sizeof(Goldilocks::Element);
            }
            end = start + d_buffers->aux_trace_sizes[sd.localStreamId] * sizeof(Goldilocks::Element);
        }
        sd.overlapsStreamCommitRegion =
            start < constAggOffsetBytes && end > d_buffers->streamCommitFloorBytes;
        if (sd.overlapsStreamCommitRegion) {
            if (sd.recursive) overlappedRecursive++; else overlappedBasic++;
        }
    }

    // A stream belongs to the device current at creation, so bind the first GPU
    // here (and again at each commit, which runs on a different thread). Restore
    // the caller's device: this is a library entry, it should not leave thread
    // state changed under its caller.
    d_buffers->streamCommitStreams = (cudaStream_t *)malloc(nSlots * sizeof(cudaStream_t));
    int prevDevice = 0;
    CHECKCUDAERR(cudaGetDevice(&prevDevice));
    CHECKCUDAERR(cudaSetDevice(d_buffers->my_gpu_ids[0]));
    int leastPriority = 0, greatestPriority = 0;
    CHECKCUDAERR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    for (uint64_t j = 0; j < nSlots; j++)
        CHECKCUDAERR(cudaStreamCreateWithPriority(&d_buffers->streamCommitStreams[j],
                                                  cudaStreamNonBlocking, leastPriority));
    CHECKCUDAERR(cudaSetDevice(prevDevice));
    // Set last: the count is the enable flag readers check.
    d_buffers->streamCommitSlots = nSlots;

    zklog.info("Streaming-commit slots: " + std::to_string(nSlots) + " x " +
               std::to_string(slotBytes >> 20) + " MB (derived), floor at " +
               std::to_string(d_buffers->streamCommitFloorBytes / (1024.0 * 1024.0 * 1024.0)) +
               " GB (overlapping " + std::to_string(overlappedBasic) + " basic + " +
               std::to_string(overlappedRecursive) + " recursive streams on the first GPU)");
}

// Shared hold of the legacy streams whose aux regions overlap the slot area
// (first-GPU streams only -- slots exist only there): the FIRST in-flight slot
// commit claims every overlapped stream's selection mutex (only if the stream
// is idle/drained); the LAST releases them. While held, selectStream/
// reserveStream try_lock and skip them, so no legacy work ever touches an
// overlapped region concurrently with a slot commit -- and the streams return
// to full rotation as soon as slots go idle.
static bool streamCommitAcquireRegion(DeviceCommitBuffers *d_buffers) {
    std::lock_guard<std::mutex> lk(d_buffers->streamCommitRegionMutex);
    if (d_buffers->streamCommitInFlight == 0) {
        std::vector<uint32_t> locked;
        for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
            StreamData &sd = d_buffers->streamsData[i];
            if (!sd.overlapsStreamCommitRegion) continue;
            bool ok = sd.mutex_stream_selection.try_lock();
            if (ok) {
                // Selected streams stay locked by their owner, so a successful
                // try_lock means unselected -- but kernels may still be draining.
                bool drained = sd.status == 2 && cudaEventQuery(sd.end_event) == cudaSuccess;
                if (!(sd.status == 0 || sd.status == 3 || drained)) {
                    sd.mutex_stream_selection.unlock();
                    ok = false;
                }
            }
            if (!ok) {
                for (uint32_t j : locked) d_buffers->streamsData[j].mutex_stream_selection.unlock();
                return false;
            }
            locked.push_back(i);
        }
    }
    d_buffers->streamCommitInFlight++;
    return true;
}

static void streamCommitReleaseRegion(DeviceCommitBuffers *d_buffers) {
    std::lock_guard<std::mutex> lk(d_buffers->streamCommitRegionMutex);
    if (--d_buffers->streamCommitInFlight == 0) {
        for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
            StreamData &sd = d_buffers->streamsData[i];
            if (sd.overlapsStreamCommitRegion)
                sd.mutex_stream_selection.unlock();
        }
    }
}

// Commit a bit-packed witness on a streaming-commit slot (first GPU): upload,
// chunked unpack+LDE+sponge absorb, node reduction, root (4 u64) to `root`.
// Poseidon1 arity 4 only (family checked here, arity by the caller).
// Synchronous; safe to call concurrently on distinct slots, and while the
// first GPU's buffer is borrowed by gpu-mops (touches only the slot).
//
// Indexed airs are supported: the compact-row descriptor and the uploaded
// instruction table are read from the first GPU's AirInstanceInfo (separate
// cudaMalloc'd allocations, so the gpu-mops borrow of gpuMemoryBuffer[0] does
// not disturb them) -- hence airgroupId/airId.
//
// Returns 0 on success, negative on misuse; -14 (region busy: legacy work is
// running on an overlapped stream) is an expected transient -- callers fall
// back to the legacy path silently.
int64_t commit_witness_streaming_gpu(void *d_buffers_, uint64_t slotIdx,
                                     uint64_t airgroupId, uint64_t airId,
                                     void *packed, uint64_t nBits, uint64_t nBitsExt,
                                     uint64_t nCols, uint64_t wordsPerRow,
                                     void *colWidths, void *root) {
    if (d_buffers_ == nullptr) return -10;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers->streamCommitSlots == 0) return -11;
    if (slotIdx >= d_buffers->streamCommitSlots) return -12;
    // The slot pipeline has kernels for Poseidon1 W=16 (arity 4) and blake3
    // (arity 2) -- the caller checks the arity, this checks the family. 
    const HashFamily scFamily = get_hash_family();
    if (scFamily != HashFamily::Poseidon1 && scFamily != HashFamily::Blake3) return -15;
    // Quiesced: gpu-mops is in its final planning phase — stay off the GPU.
    if (d_buffers->streamCommitQuiesced.load(std::memory_order_acquire)) return -14;

    StreamCommitDims dims{nBits, nBitsExt, nCols, wordsPerRow};

    // Indexed descriptor from the first GPU's AirInstanceInfo (d_col_source is set
    // at setup from PackedInfo; d_instr_table arrives per program via
    // register_instruction_table). Both live outside gpuMemoryBuffer[0].
    const uint8_t *dColSource = nullptr;
    const uint64_t *dTable = nullptr;
    AirInstanceInfo *aii = nullptr;
    auto it = d_buffers->air_instances.find({airgroupId, airId});
    if (it != d_buffers->air_instances.end()) {
        auto pit = it->second.find("basic");
        if (pit != it->second.end() && !pit->second.empty()) aii = pit->second[0];
    }
    if (aii != nullptr && aii->d_col_source != nullptr) {
        if (aii->d_instr_table == nullptr) {
            // Indexed air with no table yet: the plain walk would decode the compact
            // rows as full ones, so refuse rather than emit a wrong root. Distinct
            // from -14 so the caller surfaces it.
            zklog.error("commit_witness_streaming: air (" + std::to_string(airgroupId) + "," +
                        std::to_string(airId) + ") is indexed but no instruction table is "
                        "registered; call register_instruction_table first");
            return -16;
        }
        dColSource = aii->d_col_source;
        dTable = aii->d_instr_table;
        dims.indexBits = aii->index_bits;
        dims.wordsPerEntry = aii->words_per_entry;
        dims.numEntries = aii->num_entries;
    }

    const StreamCommitHash scHash = (scFamily == HashFamily::Blake3)
                                        ? StreamCommitHash::Blake3
                                        : StreamCommitHash::Poseidon1;
    if (streamCommitSlotElems(dims, scHash) * sizeof(Goldilocks::Element) > d_buffers->streamCommitSlotBytes)
        return -13;

    if (!streamCommitAcquireRegion(d_buffers)) return -14;

    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    gl64_t *slotBase = d_buffers->gpuMemoryBuffer[0] +
                       (d_buffers->streamCommitFloorBytes + slotIdx * d_buffers->streamCommitSlotBytes) /
                           sizeof(Goldilocks::Element);
    int64_t rc = streamCommitPacked(slotBase, dims, (const uint64_t *)colWidths, packed,
                                    (uint64_t *)root, d_buffers->streamCommitStreams[slotIdx],
                                    dColSource, dTable, scHash);
    streamCommitReleaseRegion(d_buffers);
    return rc;
}

// Quiesce the streaming-commit slots: reject new slot commits and wait for the
// in-flight ones to drain. Called by the gpu-mops borrower RIGHT BEFORE its
// final planning phase
void stream_commit_pause_gpu() {
    DeviceCommitBuffers *d_buffers = gStreamCommitBuffers.load(std::memory_order_acquire);
    if (d_buffers == nullptr || d_buffers->streamCommitSlots == 0) return;
    d_buffers->streamCommitQuiesced.store(1, std::memory_order_release);
    for (int spins = 0; spins < 4000; spins++) {  // ~2 s cap; commits take ~250 ms max
        bool busy;
        {
            std::lock_guard<std::mutex> lk(d_buffers->streamCommitRegionMutex);
            busy = d_buffers->streamCommitInFlight != 0;
        }
        if (!busy) return;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    zklog.warning("stream_commit_pause: in-flight slot commits did not drain within 2 s");
}

// Acquires exclusive use of the FIRST GPU's unified buffer (my_gpu_ids[0]) for the
// caller.
void acquire_first_gpu_buffer_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    // New borrow cycle: slots are usable again until the next quiesce.
    d_buffers->streamCommitQuiesced.store(0, std::memory_order_release);
    const uint32_t firstGpuId = d_buffers->my_gpu_ids[0];

    // Flip the flag atomically w.r.t. stream selection on the first GPU.
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        if (d_buffers->streamsData[i].gpuId == firstGpuId)
            d_buffers->streamsData[i].mutex_stream_selection.lock();
    }
    d_buffers->firstGpuBufferBorrowed.store(1, std::memory_order_release);
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        if (d_buffers->streamsData[i].gpuId == firstGpuId)
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
    }

    // Drain: wait until no prover work is queued or running on the first GPU.
    bool firstGpuIdle = false;
    while (!firstGpuIdle) {
        firstGpuIdle = true;
        for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
            if (d_buffers->streamsData[i].gpuId != firstGpuId) continue;
            d_buffers->streamsData[i].mutex_stream_selection.lock();
            uint32_t st = d_buffers->streamsData[i].status;
            bool idle = (st == 0 || st == 3 ||
                         (st == 2 && cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess));
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
            if (!idle) { firstGpuIdle = false; break; }
        }
        if (!firstGpuIdle) std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
    CHECKCUDAERR(cudaSetDevice(firstGpuId));
    CHECKCUDAERR(cudaDeviceSynchronize());
}


// Slot count of the witness prefetch zone; the Rust side sizes the region with it.
uint32_t get_prefetch_witness_slots_gpu() {
    return DeviceCommitBuffers::PREFETCH_WITNESS_SLOTS;
}

// Arm the witness prefetch zone (idempotent; no-op when witnessBytes == 0, i.e. the
// feature is off). The other segments -- fixedTreeBytes (const-tree staging),
// packedConstBytes (packed const-pols staging), recWitnessBytes (recursive-witness
// slots) -- are accepted for ABI stability but not implemented yet: callers pass 0.
void configure_prefetch_zone_gpu(void *d_buffers_, uint64_t witnessBytes, uint64_t fixedTreeBytes, uint64_t packedConstBytes, uint64_t recWitnessBytes) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || witnessBytes == 0 || d_buffers->prefetchArmed) return;
    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    // PREFETCH_WITNESS_SLOTS (2) slots: upload/compute ping-pong (instance i+1 stages
    // on the copy stream while instance i computes).
    uint64_t witnessArea = witnessBytes * DeviceCommitBuffers::PREFETCH_WITNESS_SLOTS;
    d_buffers->prefetchSlotStride = witnessBytes / sizeof(gl64_t);
    const uint64_t needed = witnessArea + fixedTreeBytes + packedConstBytes + 2 * recWitnessBytes;
    // The zone IS the unified buffer's prefetch region, always: the Rust side sizes the
    // region from the same numbers it passes here, so a mismatch is a bug -- refuse to
    // arm (proofs fall back to the legacy upload) rather than allocate elsewhere.
    if (d_buffers->prefetchRegionBase == nullptr || d_buffers->prefetchRegionBytes < needed) {
        zklog.warning("Prefetch region absent or too small (" +
                      std::to_string(d_buffers->prefetchRegionBytes >> 20) + " MB < " +
                      std::to_string(needed >> 20) + " MB); prefetch zone NOT armed");
        return;
    }
    zklog.info("Prefetch zone armed (" + std::to_string(needed >> 20) + " MB of " +
               std::to_string(d_buffers->prefetchRegionBytes >> 20) + " MB region)");
    CHECKCUDAERR(cudaStreamCreateWithFlags(&d_buffers->prefetchStream, cudaStreamNonBlocking));
    for (uint32_t s = 0; s < DeviceCommitBuffers::PREFETCH_WITNESS_SLOTS; s++) {
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->prefetchReady[s], cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->prefetchDrained[s], cudaEventDisableTiming));
        d_buffers->prefetchInstanceId[s] = -1;
    }
    d_buffers->prefetchArmed = true;
}

// Upload `instanceId`'s trace into the prefetch zone on the copy stream, while the
// current proof runs. Chunked so no single transfer monopolizes PCIe. The caller must
// keep `trace` alive until the matching gen_proof consumes the zone. Returns 0 on
// success; negative = zone absent/unknown air/too small (caller falls back to the
// legacy upload silently).
int64_t prefetch_witness_gpu(void *pSetupCtx_, void *d_buffers_, uint64_t instanceId,
                             uint64_t airgroupId, uint64_t airId, void *trace) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || !d_buffers->prefetchArmed || trace == nullptr) return -1;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    auto key = std::make_pair(airgroupId, airId);
    auto it = d_buffers->air_instances.find(key);
    if (it == d_buffers->air_instances.end()) return -2;
    auto pit = it->second.find("basic");
    if (pit == it->second.end() || pit->second.empty()) return -2;
    AirInstanceInfo *aii = pit->second[0];

    uint64_t N = (1ull << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t total_size = (d_buffers->packedTrace && aii->is_packed)
                              ? aii->num_packed_words * N * sizeof(Goldilocks::Element)
                              : N * nCols * sizeof(Goldilocks::Element);

    if (total_size > d_buffers->prefetchSlotStride * sizeof(gl64_t)) return -4;

    std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    stageWitnessSlotLocked(d_buffers, instanceId, trace, total_size);
    return 0;
}

void release_first_gpu_buffer_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    CHECKCUDAERR(cudaSetDevice(d_buffers->my_gpu_ids[0]));
    CHECKCUDAERR(cudaDeviceSynchronize());
    // The borrower overwrote this GPU's aux traces (incl. the cached const pols/tree),
    // so invalidate every affected stream's reuse context, forcing a constants reload.
    const uint32_t firstGpuId = d_buffers->my_gpu_ids[0];
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        if (d_buffers->streamsData[i].gpuId != firstGpuId) continue;
        d_buffers->streamsData[i].invalidateContext();
        d_buffers->streamsData[i].instanceId = -1;        // clobbered witness, not ready
    }
    // The prefetch zone lives inside the borrowed buffer, so the borrower scribbled over
    // it too -- drop every staging key (device is synchronized above, no upload in flight).
    if (d_buffers->prefetchArmed) {
        std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
        for (uint32_t sIdx = 0; sIdx < DeviceCommitBuffers::PREFETCH_WITNESS_SLOTS; sIdx++) { d_buffers->prefetchInstanceId[sIdx] = -1; d_buffers->prefetchTraceBytes[sIdx] = 0; }
    }
    d_buffers->firstGpuBufferBorrowed.store(0, std::memory_order_release);
}

uint32_t is_first_gpu_buffer_borrowed_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return 0;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire);
}

// Device id of the FIRST GPU (my_gpu_ids[0]) — the borrowed buffer's GPU. NOT
// necessarily 0 (NUMA can reorder). Consumers bind this before using the buffer.
uint32_t get_first_gpu_id_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return 0;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return d_buffers->my_gpu_ids[0];
}

void *get_unified_buffer_gpu_for_recursivef_gpu(void *d_buffers_, void *d_buffers_recursivef_) {
    if (d_buffers_ == nullptr) return nullptr;
    if (d_buffers_recursivef_ == nullptr) return get_unified_buffer_gpu_gpu(d_buffers_);
    DeviceRecursiveFBuffers *d_bufs_rec = (DeviceRecursiveFBuffers *)d_buffers_recursivef_;
    CHECKCUDAERR(cudaSetDevice(d_bufs_rec->gpuId));
    return get_unified_buffer_gpu(d_buffers_);
}

// The const-pols slot a request will land on; UINT64_MAX when unknowable here: no
// AirInstanceInfo for the key, or a recurser (one shared slot, so the offset says nothing
// about which recurser is in it).
static const AirInstanceInfo *requestedAirInstance(DeviceCommitBuffers* d_buffers, uint64_t airgroupId, uint64_t airId, const std::string &proofType){
    if (proofType == "recurser_aggregator") return nullptr;
    // commit_witness tags its stream "witness", but its setup key is the basic one.
    const std::string keyType = (proofType == "witness") ? std::string("basic") : proofType;
    auto air = d_buffers->air_instances.find({airgroupId, airId});
    if (air == d_buffers->air_instances.end()) return nullptr;
    auto byType = air->second.find(keyType);
    if (byType == air->second.end() || byType->second.empty()) return nullptr;
    // Every GPU places a given air at the same offset, so gpu 0 is representative.
    return byType->second[0];
}

// A forced request may only be held to a pool that exists: the Rust carve drops it when an
// aggregation-only stream costs what a basic one does.
static bool hasRecursivePool(DeviceCommitBuffers* d_buffers){
    // Phase-B aliases count as a pool only while open; otherwise a forced recursive request
    // would refuse the basic-stream fallback for the whole phase A.
    if (d_buffers->phaseBAliased && d_buffers->phaseBState.load(std::memory_order_acquire) != 1) return false;
    return d_buffers->n_recursive_streams > 0;
}

// Phase B (DeviceCommitBuffers::phaseBAliased): may `sd` be reserved in the current phase?
static bool phaseBEligible(DeviceCommitBuffers* d_buffers, const StreamData &sd){
    if (!d_buffers->phaseBAliased) return true;
    const uint32_t st = d_buffers->phaseBState.load(std::memory_order_acquire);
    if (sd.recursive) return st == 1;
    return st != 1 && !d_buffers->phaseBClosing.load(std::memory_order_acquire);
}

// Completes a requested phase-B switch once every basic stream is idle: no reservation held
// (status 1, or a pick in progress holding its selection mutex), nothing queued (status 2 with
// the end event pending, or ring entries). Then a hard stream fence, and the aliases' const
// identity is dropped: the basic stream's proofs overwrote their buffers. Caller holds
// stream_selection_mutex, so no basic reservation can slip in between the check and the flip.
static void tryOpenPhaseB(DeviceCommitBuffers* d_buffers){
    if (!d_buffers->phaseBClosing.load(std::memory_order_acquire)) return;
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        StreamData &sd = d_buffers->streamsData[i];
        if (sd.recursive) continue;
        if (!sd.mutex_stream_selection.try_lock()) return;
        const uint32_t st = sd.status.load(std::memory_order_acquire);
        uint32_t ring;
        { std::lock_guard<std::mutex> plk(sd.pipeMutex); ring = sd.pipeCount; }
        cudaSetDevice(sd.gpuId);
        const bool idle = ring == 0 &&
            (st == 0 || st == 3 || (st == 2 && cudaEventQuery(sd.end_event) == cudaSuccess));
        sd.mutex_stream_selection.unlock();
        if (!idle) return;
    }
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        StreamData &sd = d_buffers->streamsData[i];
        cudaSetDevice(sd.gpuId);
        if (!sd.recursive) {
            CHECKCUDAERR(cudaStreamSynchronize(sd.stream));
        } else {
            std::lock_guard<std::mutex> lk(sd.mutex_stream_selection);
            sd.invalidateContext();
        }
    }
    d_buffers->phaseBState.store(1, std::memory_order_release);
    d_buffers->phaseBClosing.store(false, std::memory_order_release);
    zklog.info("Phase B open: recursion continues on the two aliased streams");
}

static uint64_t largestBasicCapacity(DeviceCommitBuffers* d_buffers){
    uint64_t largest = 0;
    for (uint32_t j = 0; j < d_buffers->n_streams; ++j) {
        largest = std::max(largest, d_buffers->aux_trace_sizes[j]);
    }
    return largest;
}

// Aux-trace elements this launch needs, 0 when the air is not registered (see requirementFor).
// Always full mapTotalN, even for a "witness" (contributions) launch. Sizing those by the regions the
// commit touches looked safe and is not: besides cm1/mt1 the commit also writes const, custom_fixed,
// publics, airgroupvalues, airvalues and proofvalues, so a stream sized below mapTotalN lets those
// writes run past its slice into the next stream's buffer -- observed as a contribution-challenge
// mismatch, not a crash. Any narrower bound must cover every one of those sections.
// Under-counts compressor/recursive launches by their trace, which only ever land on classes the
// Rust carve floored high enough (plan_stream_layout), so they always fit.
static uint64_t requiredAuxTrace(DeviceCommitBuffers* d_buffers, uint64_t airgroupId, uint64_t airId, const std::string &proofType){
    const AirInstanceInfo *air = requestedAirInstance(d_buffers, airgroupId, airId, proofType);
    if (air == nullptr || air->setupCtx == nullptr) return 0;
    return air->setupCtx->starkInfo.mapTotalN;
}

// What `stream` must hold for this launch. An unknown requirement falls back to the largest class on
// the sized non-recursive pool (the old uniform placement); the recursive pool is one size, so
// holding it to that guess would leave a forced recursive request with no eligible stream at all.
static uint64_t requirementFor(DeviceCommitBuffers* d_buffers, const StreamData &stream, uint64_t known){
    if (known != 0) return known;
    return stream.recursive ? 0 : largestBasicCapacity(d_buffers);
}

// One non-blocking scan+pick+reserve pass (the body of selectStream's old while-loop).
// Returns a reserved streamId (status=1), or UINT32_MAX if none is free right now. Lets the
// Rust scheduler drive stream assignment; selectStream just retries this in a loop.
uint32_t reserve_best_stream_scan(DeviceCommitBuffers* d_buffers, uint64_t airgroupId, uint64_t airId, std::string proofType, bool recursive, bool force_recursive){
    // Affinity follows the fixed columns, not the air: a stream that last served a different
    // air sharing this slot is just as warm, since that is what gen_*_proof keys reuse on.
    const AirInstanceInfo *wantAir = requestedAirInstance(d_buffers, airgroupId, airId, proofType);
    // Must match adoptFixedSlot's key exactly, or this steers to a stream that then cannot
    // reuse anyway. An aliased air never reuses, so it is never warm by slot.
    const bool slotUsable = wantAir != nullptr && wantAir->setupCtx != nullptr
                            && !wantAir->setupCtx->starkInfo.constPolsAliasTree;
    const uint64_t wantSlot = slotUsable ? wantAir->const_pols_offset : UINT64_MAX;
    const uint64_t wantAux = slotUsable
        ? wantAir->setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)] : 0;
    const bool wantAgg = !(proofType == "basic" || proofType == "witness");
    // Streams come in size classes, so a free one is only a candidate when its buffer holds this launch.
    const uint64_t needAux = requiredAuxTrace(d_buffers, airgroupId, airId, proofType);
    auto fitsCapacity = [&](const StreamData &sd) {
        return sd.auxTraceCapacity >= requirementFor(d_buffers, sd, needAux);
    };
    auto isWarm = [&](const StreamData &sd, bool drained) {
        if (!(sd.status==3 || drained)) return false;
        if (sd.airgroupId == airgroupId && sd.airId == airId && sd.proofType == proofType) return true;
        return wantSlot != UINT64_MAX && sd.constPolsOffset == wantSlot && sd.constAuxOffset == wantAux
               && sd.constAggBuffer == wantAgg && sd.constRecurserId.empty();
    };
    // The best free stream on one GPU, and how much it has free (which picks between GPUs).
    struct GpuPick { int stream = -1; uint32_t free = 0, unused = 0; bool warm = false, wasUnused = false; };
    std::vector<GpuPick> picks(d_buffers->n_gpus);

    // Tightest fit first, so larger classes stay free for the airs with nowhere else to go. Only free
    // streams are compared, so a busy tight class never idles the big one. Within a class: warm, then unused.
    auto betterCandidate = [&](uint32_t cand, const GpuPick &best, bool candWarm, bool candUnused) {
        if (best.stream < 0) return true;
        const uint64_t candCap = d_buffers->streamsData[cand].auxTraceCapacity;
        const uint64_t bestCap = d_buffers->streamsData[best.stream].auxTraceCapacity;
        if (candCap != bestCap) return candCap < bestCap;
        if (candWarm != best.warm) return candWarm;
        return candUnused && !best.wasUnused;
    };

    bool someFree = false;

    std::vector<bool> streams_locked(d_buffers->n_total_streams, false);

    const uint32_t firstGpuId = d_buffers->my_gpu_ids[0];

    {
        // Serialize the scan so this caller sees the full set of free streams
        // (kills the fragmented try_lock race). Scoped to the scan only: released
        // before the pick/reserve below, so the global lock is NEVER held across
        // proof execution (the deadlock trap). Per-stream locks are still taken via
        // try_lock, so this never blocks on harvest/reserve.
        std::lock_guard<std::mutex> gsel(d_buffers->stream_selection_mutex);
        const bool firstGpuBorrowed = d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire);
        tryOpenPhaseB(d_buffers);
        // One pool pass; the best candidate per GPU is left locked.
        auto scanPool = [&](auto pool) {
            for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
                StreamData &sd = d_buffers->streamsData[i];
                if (firstGpuBorrowed && sd.gpuId == firstGpuId) continue;
                if (!phaseBEligible(d_buffers, sd)) continue;
                if (!pool(sd) || !sd.mutex_stream_selection.try_lock()) continue;
                // Re-check the borrow flag under the lock.
                if (sd.gpuId == firstGpuId && d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire)) {
                    sd.mutex_stream_selection.unlock();
                    continue;
                }
                // Ran to completion but not yet harvested: free to take, and its const-tree is still
                // loaded. Queried once and reused by the warm test.
                const bool drained = (sd.status==2 && cudaEventQuery(sd.end_event) == cudaSuccess)
                    || pipelineReservable(d_buffers, sd);
                if (!(sd.status==0 || sd.status==3 || drained) || !fitsCapacity(sd)) {
                    sd.mutex_stream_selection.unlock();
                    continue;
                }
                GpuPick &pick = picks[d_buffers->gpus_g2l[sd.gpuId]];
                const bool unused = sd.status==0;
                pick.free++;
                pick.unused += unused;
                // status==0 is deliberately absent from isWarm: the only path to it
                // (reset_device_streams_gpu) calls invalidateContext() first, so the key comparison
                // there can never match an unused stream anyway.
                const bool warm = isWarm(sd, drained);
                if (betterCandidate(i, pick, warm, unused)) {
                    pick.stream = i;
                    pick.warm = warm;
                    pick.wasUnused = unused;
                }
                someFree = true;
                streams_locked[i] = true;
            }
        };

        if (recursive) {
            scanPool([](const StreamData &sd) { return sd.recursive; });
        }

        // A recursive request with no free recursive stream falls back here. The reverse is NOT
        // allowed: only gen_recursive_proof_gpu resolves its aux trace by `sd.recursive`, so a basic
        // launch on a recursive stream would index the basic pool with a recursive localStreamId.
        if (!someFree && (!recursive || !(force_recursive && hasRecursivePool(d_buffers)))) {
            scanPool([](const StreamData &sd) { return !sd.recursive; });
        }
    }

    if (!someFree) return UINT32_MAX;  // nothing free this pass; caller retries

    // Most free streams wins; ties break on unused count. someFree guarantees a candidate.
    int bestGpu = -1;
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        if (picks[i].stream < 0) continue;
        if (bestGpu == -1 || picks[i].free > picks[bestGpu].free ||
            (picks[i].free == picks[bestGpu].free && picks[i].unused > picks[bestGpu].unused)) {
            bestGpu = i;
        }
    }
    uint32_t selectedStreamId = picks[bestGpu].stream;
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        if (streams_locked[i] && i != selectedStreamId) {
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
        }
    }

    reserveStreamLocked(d_buffers, selectedStreamId);
    d_buffers->streamsData[selectedStreamId].mutex_stream_selection.unlock();

    return selectedStreamId;
}

// Blocking wrapper: retry the scan until a stream is reserved. Used by the paths that
// select internally (contributions/commit/setup, and one-off recursive launches).
uint32_t selectStream(DeviceCommitBuffers* d_buffers, uint64_t airgroupId, uint64_t airId, std::string proofType, bool recursive, bool force_recursive){
    for (uint64_t spins = 0;; ++spins) {
        uint32_t s = reserve_best_stream_scan(d_buffers, airgroupId, airId, proofType, recursive, force_recursive);
        if (s != UINT32_MAX) return s;
        // Two very different causes spin here: no stream is large enough (permanent -- this would
        // spin forever), or every eligible stream is merely busy (transient, and a sign the class is
        // under-provisioned). Reported once; both keep retrying.
        if (spins == 20000) {
            const uint64_t need = requiredAuxTrace(d_buffers, airgroupId, airId, proofType);
            uint32_t eligible = 0;
            uint64_t largest = 0;
            for (uint32_t i = 0; i < d_buffers->n_total_streams; ++i) {
                const StreamData &sd = d_buffers->streamsData[i];
                if (!phaseBEligible(d_buffers, sd)) continue;
                if (force_recursive && hasRecursivePool(d_buffers) && !sd.recursive) continue;
                largest = std::max(largest, sd.auxTraceCapacity);
                if (sd.auxTraceCapacity >= requirementFor(d_buffers, sd, need)) eligible++;
            }
            const double gb = sizeof(Goldilocks::Element) / (1024.0 * 1024.0 * 1024.0);
            const std::string what = proofType + " [" + std::to_string(airgroupId) + ":" +
                                     std::to_string(airId) + "] (needs " + std::to_string(need * gb) + " GB)";
            if (eligible == 0) {
                zklog.error("selectStream: " + what + " exceeds every stream -- largest holds " +
                            std::to_string(largest * gb) + " GB, so this can never be placed");
            } else {
                zklog.warning("selectStream: " + what + " starved 6s behind " + std::to_string(eligible) +
                              " eligible stream(s) -- that class may be under-provisioned");
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
}

// GPU backend entry for the Rust scheduler: reserve a stream, then pass it to
// gen_*_proof(..., streamId_). Returns UINT32_MAX when nothing is free (caller retries).
uint32_t reserve_best_stream_nonblock_gpu(void* d_buffers_, uint64_t airgroupId, uint64_t airId, char* proofType, bool recursive, bool force_recursive){
    DeviceCommitBuffers* d_buffers = (DeviceCommitBuffers*)d_buffers_;
    return reserve_best_stream_scan(d_buffers, airgroupId, airId, std::string(proofType), recursive, force_recursive);
}

// Is some other free non-recursive stream on this GPU a tighter fit for `need` than `cap`? Advisory:
// statuses are read unlocked, so a wrong answer costs a cold scan or one oversized placement, no more.
static bool tighterFreeStreamExists(DeviceCommitBuffers* d_buffers, uint64_t need, uint64_t cap, uint32_t self, uint32_t gpuId){
    for (uint32_t i = 0; i < d_buffers->n_total_streams; ++i) {
        if (i == self) continue;
        const StreamData &sd = d_buffers->streamsData[i];
        if (sd.recursive || sd.gpuId != gpuId) continue;
        if (sd.auxTraceCapacity < need || sd.auxTraceCapacity >= cap) continue;
        const uint32_t status = sd.status.load(std::memory_order_relaxed);
        if (status == 0 || status == 3) return true;
    }
    return false;
}

// Warm-affinity fast path: reserve `streamId` IFF free right now (and a recursive stream
// for a forced request). Returns 1 on success, 0 otherwise. Same lock order as
// reserve_best_stream_scan (gsel, then per-stream try_lock) so they can't deadlock.
uint32_t reserve_stream_if_free_gpu(void* d_buffers_, uint32_t streamId, uint64_t airgroupId, uint64_t airId, char* proofType, bool force_recursive){
    DeviceCommitBuffers* d_buffers = (DeviceCommitBuffers*)d_buffers_;
    if (streamId >= d_buffers->n_total_streams) return 0;
    StreamData& sd = d_buffers->streamsData[streamId];
    if (!phaseBEligible(d_buffers, sd)) return 0;
    // A forced recursive launch must stay on a recursive stream while a pool exists; refuse
    // otherwise so the caller falls back to the cold scan.
    if (force_recursive && hasRecursivePool(d_buffers) && !sd.recursive) return 0;
    // Warm is worthless if the launch does not fit; the cold scan will find a class that does.
    const uint64_t known = requiredAuxTrace(d_buffers, airgroupId, airId, std::string(proofType));
    const uint64_t need = requirementFor(d_buffers, sd, known);
    if (sd.auxTraceCapacity < need) return 0;
    const uint32_t firstGpuId = d_buffers->my_gpu_ids[0];
    if (d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire) && sd.gpuId == firstGpuId) return 0;

    {
        // Scoped like the cold scan's: reserving harvests a proof, and the global lock must never be
        // held across that. The per-stream lock, which guards the state, stays held.
        std::lock_guard<std::mutex> gsel(d_buffers->stream_selection_mutex);
        if (!sd.mutex_stream_selection.try_lock()) return 0;
        if (sd.gpuId == firstGpuId && d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire)) {
            sd.mutex_stream_selection.unlock();
            return 0;
        }
        bool free = sd.status==0 || sd.status==3 || (sd.status==2 && cudaEventQuery(sd.end_event) == cudaSuccess)
                    || pipelineReservable(d_buffers, sd);
        if (!free) { sd.mutex_stream_selection.unlock(); return 0; }
        // Warm but too roomy: taking it would park this launch on a stream some larger air may be the
        // only user of. Refuse, and let the scan apply its tightest-fit rule. Sized pool only -- the
        // recursive streams are one class, and comparing them against basic ones just loses affinity.
        if (!sd.recursive && sd.auxTraceCapacity > need
            && tighterFreeStreamExists(d_buffers, need, sd.auxTraceCapacity, streamId, sd.gpuId)) {
            sd.mutex_stream_selection.unlock();
            return 0;
        }
    }
    reserveStreamLocked(d_buffers, streamId);
    sd.mutex_stream_selection.unlock();
    return 1;
}

// Give a reservation back without launching on it. A reserved stream sits at status==1, which every
// selection pass treats as busy, so a caller that reserves and then fails before launching would
// strand the slot for the process lifetime. Only status==1 is released, so this can never steal a
// stream that has since been launched on (2) or torn down (0).
// Takes ONLY the per-stream lock -- never stream_selection_mutex. The reserve paths take gsel then
// the per-stream lock; acquiring them in the other order here would close a deadlock cycle.
void release_stream_reservation_gpu(void* d_buffers_, uint32_t streamId){
    DeviceCommitBuffers* d_buffers = (DeviceCommitBuffers*)d_buffers_;
    if (d_buffers == nullptr || d_buffers->streamsData == nullptr) return;
    if (streamId >= d_buffers->n_total_streams) return;
    StreamData& sd = d_buffers->streamsData[streamId];
    std::lock_guard<std::mutex> lg(sd.mutex_stream_selection);
    if (sd.status != 1) return;
    // Back to "reusable, not unused": reserveStreamLocked already ran reset(false) and left the
    // (airgroup,air,type) identity intact, so warm affinity still holds for the next pick.
    sd.status = 3;
}

// Requires the caller to hold streamsData[streamId].mutex_stream_selection
void reserveStreamLocked(DeviceCommitBuffers* d_buffers, uint32_t streamId){
    StreamData &sd = d_buffers->streamsData[streamId];
    cudaSetDevice(sd.gpuId);
    if(sd.status==2) {
        // A proof launched outside the ring (sd.proofBuffer set: legacy path, e.g. before the
        // pipeline was enabled) is collected only by the sync-and-collect below; skipping it
        // would drop its completion for good.
        if (d_buffers->pipelineMode && !sd.recursive && sd.proofBuffer == nullptr && sd.root == nullptr) {
            // Pipeline: harvest whatever already fired, and reserve WITHOUT the
            // host sync as long as a ring slot is free -- that is the whole point.
            harvestPipelineStream(d_buffers, streamId, false);
            bool slotFree;
            {
                std::lock_guard<std::mutex> plk(sd.pipeMutex);
                slotFree = sd.pipeCount < 2;
            }
            if (slotFree) {
                // No reset: warm/const identity fields stay valid (maintained at
                // enqueue), and the ring keeps the in-flight proof's metadata.
                sd.status = 1;
                return;
            }
            CHECKCUDAERR(cudaEventSynchronize(sd.end_event));
            harvestPipelineStream(d_buffers, streamId, true);
        } else {
            // No-op via selectStream (event already fired); any other caller must wait.
            CHECKCUDAERR(cudaEventSynchronize(sd.end_event));
            collectStreamResult(d_buffers, streamId);
        }
    }
    sd.reset(false);
    sd.status = 1;
}

void reserveStream(DeviceCommitBuffers* d_buffers, uint32_t streamId){
    d_buffers->streamsData[streamId].mutex_stream_selection.lock();
    reserveStreamLocked(d_buffers, streamId);
    d_buffers->streamsData[streamId].mutex_stream_selection.unlock();
}
void closeStreamTimer(TimerGPU &timer, uint64_t instance_id, uint64_t airgroup_id, uint64_t air_id, bool isProve) {
    TimerSyncAndLogAllGPU(timer, instance_id, airgroup_id, air_id);
    TimerSyncCategoriesGPU(timer);
    if(isProve)
        TimerLogCategoryContributionsGPU(timer, STARK_GPU_PROOF);
    else
        TimerLogCategoryContributionsGPU(timer, STARK_GPU_COMMIT);
    TimerResetGPU(timer);
}

void *init_final_snark_prover_gpu(char* zkeyFile, void* d_buffers_recursivef) {
    int gpuId = 0;
    if (d_buffers_recursivef != nullptr) {
        DeviceRecursiveFBuffers *d_bufs = (DeviceRecursiveFBuffers *)d_buffers_recursivef;
        gpuId = d_bufs->gpuId;
        cudaSetDevice(gpuId);
    }
    return initFinalSnarkProverGPU(zkeyFile, gpuId);
}

void free_final_snark_prover_gpu(void *snark_prover) {
    freeFinalSnarkProverGPU(snark_prover);
}

void gen_final_snark_proof_gpu(void *prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark, void* d_buffers_recursivef) {
    if (d_buffers_recursivef != nullptr) {
        DeviceRecursiveFBuffers *d_buffers = (DeviceRecursiveFBuffers *)d_buffers_recursivef;
        cudaSetDevice(d_buffers->gpuId);
    }
    genFinalSnarkProofGPU(prover, circomWitnessFinal, proof, publicsSnark);
}

void pre_allocate_final_snark_prover_gpu(void *snark_prover, void* unified_buffer_gpu, void* d_buffers_recursivef) {

    if (unified_buffer_gpu != nullptr) {
        uint64_t requiredSize = getFinalSnarkProverRequiredGpuSizeGPU(snark_prover);
        DeviceCommitBuffers *d_commit_buffers = gStreamCommitBuffers.load(std::memory_order_acquire);
        uint64_t availableSize = d_commit_buffers != nullptr ? d_commit_buffers->unifiedBufferSize : 0;
        zklog.info("Final snark prover GPU requirement: " + std::to_string(requiredSize) +
                   " bytes; unified buffer holds " + std::to_string(availableSize) + " bytes (margin " +
                   std::to_string((int64_t)availableSize - (int64_t)requiredSize) + ")");
        if (requiredSize > availableSize) {
            zklog.warning("Final snark prover does not fit the unified buffer; falling back to a dedicated allocation");
            unified_buffer_gpu = nullptr;
        }
    }
    if (d_buffers_recursivef != nullptr) {
        DeviceRecursiveFBuffers *d_buffers = (DeviceRecursiveFBuffers *)d_buffers_recursivef;
        cudaSetDevice(d_buffers->gpuId);
        if (unified_buffer_gpu == nullptr && d_buffers->owns_aux_trace) {
            uint64_t requiredSize = getFinalSnarkProverRequiredGpuSizeGPU(snark_prover);
            if (requiredSize > 0) {
                if (requiredSize > d_buffers->aux_trace_size) {
                    CHECKCUDAERR(cudaFree(d_buffers->d_aux_trace));
                    CHECKCUDAERR(cudaMalloc((void **)&d_buffers->d_aux_trace, requiredSize));
                    d_buffers->aux_trace_size = requiredSize;
                }
                unified_buffer_gpu = d_buffers->d_aux_trace;
            }
        }
    }
    preAllocateFinalSnarkProverGPU(snark_prover, unified_buffer_gpu);
}
#endif