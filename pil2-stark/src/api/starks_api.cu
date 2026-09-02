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
#include <vector>
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
static bool pipelineReservable(DeviceCommitBuffers *d_buffers, StreamData &sd);
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
    d_buffers->pinned_copy_done = (cudaEvent_t (*)[2])malloc(d_buffers->n_gpus * sizeof(cudaEvent_t[2]));
    d_buffers->pinned_buffer_extra = (Goldilocks::Element **)malloc(d_buffers->n_gpus * sizeof(Goldilocks::Element *));
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
    // Phase-B aliased recursive streams live INSIDE the pre-const area: no extra VRAM.
    uint64_t totalAuxTraceRecursiveSize =
        d_buffers->phaseBAliased ? 0 : d_buffers->n_recursive_streams * auxTraceRecursiveSize;

    // Mops-floor pad (see DeviceCommitBuffers::mopsFloorPadBytes): raise the region BELOW
    // the const pols to PROOFMAN_MOPS_FLOOR_GB (default 20) so the mem-ops planner's borrow
    // fits its fixed regions. Clamped to the memory actually free on the first GPU.
    uint64_t mopsFloorPadSize = 0;
    {
        const char *fg = std::getenv("PROOFMAN_MOPS_FLOOR_GB");
        uint64_t floorBytes = (fg != nullptr ? (uint64_t)atoll(fg) : 0ull) << 30;  // default off: unsatisfiable next to the prefetch zone on 32 GB
        // Scratch-arena minimum (compact-witness staging lives here): reserved from the
        // planner's budget in proof_ctx, so claiming it can never overrun VRAM.
        if (prefetchRegionSize > 0) {
            const char *ag = std::getenv("PROOFMAN_ARENA_GB");
            uint64_t arenaBytes = (ag != nullptr ? (uint64_t)atoll(ag) : 3ull) << 30;
            uint64_t belowConstsA = totalAuxTraceSize + totalAuxTraceRecursiveSize + prefetchRegionSize;
            if (floorBytes < belowConstsA + arenaBytes) floorBytes = belowConstsA + arenaBytes;
        }
        uint64_t belowConsts = totalAuxTraceSize + totalAuxTraceRecursiveSize + prefetchRegionSize;
        if (floorBytes > belowConsts) {
            mopsFloorPadSize = floorBytes - belowConsts;
            size_t freeMem = 0, totalMem = 0;
            cudaSetDevice(d_buffers->my_gpu_ids[0]);
            CHECKCUDAERR(cudaMemGetInfo(&freeMem, &totalMem));
            uint64_t base = constPolsAggregationSize + constPolsSize + unifiedBufferPadSize + belowConsts;
            // Post-allocation headroom. Must cover every device consumer that appears after
            // the unified malloc: per-air setup buffers (witness_compact, opening points,
            // exps args), module loads of the generated .exps.so kernels (code + local-mem
            // pool, claimed at setup by expsWarmupQ), rec-tree helpers, transcript state --
            // measured ~2.5 GB on the blake key. 512 MB was enough for the aliased layouts
            // but the base-split layout OOMed the first Q launch silently at 512 MB (kernel
            // never ran -> stale quotient, rejected downstream). Trimming here only shrinks
            // the mops pad ABOVE the planner's real usage (measured 15.3 GB of the 20 GB
            // floor). PROOFMAN_MOPS_PAD_MARGIN_MB overrides.
            const char *mg = std::getenv("PROOFMAN_MOPS_PAD_MARGIN_MB");
            uint64_t margin = (mg != nullptr ? (uint64_t)atoll(mg) : 1024ull) << 20;
            uint64_t room = (freeMem > base + margin) ? freeMem - base - margin : 0;
            if (mopsFloorPadSize > room) mopsFloorPadSize = room;
            // The scratch arena (compact-witness staging) lives INSIDE this pad and its
            // minimum was already folded into floorBytes above: clamping below it would
            // silently push per-air witness allocations onto cudaMalloc fallback and OOM
            // setup. Keep the arena whole; if even that doesn't fit, the unified malloc
            // fails loudly instead.
            if (prefetchRegionSize > 0) {
                const char *ag2 = std::getenv("PROOFMAN_ARENA_GB");
                uint64_t arenaMin = (ag2 != nullptr ? (uint64_t)atoll(ag2) : 3ull) << 30;
                if (mopsFloorPadSize < arenaMin) mopsFloorPadSize = arenaMin;
            }
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
    zklog.info("  - Mops floor padding: " + std::to_string(mopsFloorPadSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Prefetch region: " + std::to_string(prefetchRegionSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
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
    zklog.info("  - Auxiliary trace recursive (" + std::to_string(d_buffers->n_recursive_streams) + " streams): " + std::to_string(totalAuxTraceRecursiveSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
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

        // Auxiliary trace buffers (recursive). Phase-B aliased pair: [0..A) and [A..2A)
        // over the basic stream buffer + prefetch region -- dead space once every basic
        // and compressor has completed, which is the only time these streams are eligible.
        if (d_buffers->phaseBAliased) {
            for (int j = 0; j < d_buffers->n_recursive_streams; ++j) {
                d_buffers->d_aux_traceAggregation[i][j] = gpuMemoryBlock + (uint64_t)j * auxTraceRecursiveArea;
            }
        } else {
            for (int j = 0; j < d_buffers->n_recursive_streams; ++j) {
                d_buffers->d_aux_traceAggregation[i][j] = gpuMemoryBlock + offset;
                offset += auxTraceRecursiveArea;
            }
        }

        // Prefetch region (zone + recursive-witness slots) lives INSIDE the unified buffer,
        // below the consts: one planned budget, and it doubles as mops-borrow donor space
        // (its staging keys are invalidated on borrow release).
        if (i == 0) {
            d_buffers->prefetchRegionBase = prefetchRegionArea > 0 ? gpuMemoryBlock + offset : nullptr;
            d_buffers->prefetchRegionBytes = prefetchRegionSize;
        }
        offset += prefetchRegionArea;

        // Mops-floor pad doubles as the scratch arena (per-air compact-witness staging):
        // its contents are per-proof transient, so the mops borrow clobbering it is fine.
        if (i == 0) {
            d_buffers->scratchArenaBase = mopsFloorPadArea > 0 ? (uint8_t *)(gpuMemoryBlock + offset) : nullptr;
            d_buffers->scratchArenaBytes = mopsFloorPadSize;
            d_buffers->scratchArenaCursor = 0;
        }
        offset += mopsFloorPadArea;
        // Phase-B feasibility: the two aliases plus staging must fit BELOW the consts.
        if (i == 0 && d_buffers->phaseBAliased) {
            uint64_t preConstBytes = offset * sizeof(Goldilocks::Element);
            uint64_t aliasedBytes = 2ull * auxTraceRecursiveSize;
            if (d_buffers->n_recursive_streams == 2 && aliasedBytes + (512ull << 20) <= preConstBytes) {
                d_buffers->phaseBSpareBase = (uint8_t *)gpuMemoryBlock + aliasedBytes;
                d_buffers->phaseBSpareBytes = preConstBytes - aliasedBytes;
                zklog.info("Phase-B aliased recursion streams: 2 x " +
                           std::to_string(auxTraceRecursiveSize / (1024.0*1024.0*1024.0)) +
                           " GB inside the pre-const area, spare " +
                           std::to_string(d_buffers->phaseBSpareBytes / (1024.0*1024.0*1024.0)) + " GB");
            } else {
                // Keep the aliased flag (sizing and eligibility gates depend on it) but
                // leave spareBase null: set_phase_b(1) refuses, the streams stay
                // permanently ineligible, and the run proceeds single-stream.
                d_buffers->phaseBSpareBase = nullptr;
                zklog.warning("Phase-B unusable: 2 x " + std::to_string(auxTraceRecursiveSize) +
                              " B does not fit the pre-const area (" + std::to_string(offset * 8) + " B)");
            }
        }

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
        if (d_buffers->pipelineMode) harvestPipelineStream(d_buffers, i, true);
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
        CHECKCUDAERR(cudaEventDestroy(d_buffers->pinned_copy_done[i][0]));
        CHECKCUDAERR(cudaEventDestroy(d_buffers->pinned_copy_done[i][1]));
        CHECKCUDAERR(cudaFreeHost(d_buffers->pinned_buffer[i]));
        CHECKCUDAERR(cudaFreeHost(d_buffers->pinned_buffer_extra[i]));
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
    free(d_buffers->pinned_copy_done);
    free(d_buffers->pinned_buffer_extra);
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
        // Arena only on the first GPU (the glued region lives there); others cudaMalloc.
        d_buffers->air_instances[key][proofType][i]->set_witness_map(
            execMapCols, 1ULL << setupCtx->starkInfo.starkStruct.nBits,
            setupCtx->starkInfo.mapSectionsN["cm1"],
            (i == 0) ? (void *)d_buffers : nullptr);
        if (nBands > 0) {
            uploadGateBandConstantsGPU((uint64_t)family);
            d_buffers->air_instances[key][proofType][i]->set_gate_bands(hostBands, nBands, bandView.aux, (uint64_t)family);
        }
    }
}

// PROOFMAN_NO_CONST_BUF=1: no GPU-resident const-pols buffer for basic airs.
// Packed blobs stay on disk (page cache) and stage into the prefetch zone's
// packed segment on each fixed-slot switch; trees load through the zone's
// fixed segment like any non-preloaded air. Rust reads the same env to size
// the zone and zero the const area.
static bool noConstBufMode() {
    static const bool v = getenv("PROOFMAN_NO_CONST_BUF") != nullptr &&
                          std::string(getenv("PROOFMAN_NO_CONST_BUF")) == "1";
    return v;
}

// Host artifact cache for zone staging (single-proof mode): each const tree /
// packed-pols file is read ONCE into host memory and cudaHostRegister'd, so
// every later staging is a single full-speed async DMA -- no fread, no pinned
// bounce, no CPU copies on the switch path. Warm profiles showed fread
// (_IO_file_xsgetn) as a top gap contributor. PROOFMAN_HOST_CACHE=0 disables.
static std::mutex gHostCacheMutex;
static std::map<std::string, std::pair<uint8_t *, uint64_t>> gHostArtifactCache;
static uint64_t gHostCacheBytes = 0;

// Cap: registered host memory is pinned RAM; unbounded caching of per-air
// artifacts (43 recursive1 trees x ~0.84 GB on the blake key) would pin most
// of the machine (24 GB OOM-killed the 712-tx block on a 62 GB box). With
// recursive trees rebuilt ON DEVICE on a cache miss (see gen_recursive), the
// cache only needs the artifacts that cannot be recomputed (custom-commits
// fixed blobs), so the default is small. Measured: 4 GB beats 16 GB (the tree
// DMAs competed with witness staging for PCIe). PROOFMAN_HOST_CACHE_GB overrides.
static uint64_t hostCacheCapBytes() {
    static const uint64_t v = [] {
        const char *e = getenv("PROOFMAN_HOST_CACHE_GB");
        uint64_t gb = 4;
        if (e != nullptr) { char *q; uint64_t p = strtoull(e, &q, 10); if (q != e) gb = p; }
        return gb << 30;
    }();
    return v;
}

static bool hostCacheEnabled() {
    static const bool v = getenv("PROOFMAN_HOST_CACHE") == nullptr ||
                          std::string(getenv("PROOFMAN_HOST_CACHE")) != "0";
    return v;
}

// Returns a registered host pointer holding >= bytes of `path` starting at
// `skip` bytes into the file, or nullptr (caller falls back to the chunked
// fread path). Cache key includes the skip so distinct views never collide.
static const uint8_t *hostCacheGetSkip(const char *path, uint64_t bytes, uint64_t skip) {
    if (!hostCacheEnabled()) return nullptr;
    std::lock_guard<std::mutex> lk(gHostCacheMutex);
    std::string key = std::string(path) + ":" + std::to_string(skip);
    auto it = gHostArtifactCache.find(key);
    if (it != gHostArtifactCache.end()) {
        return it->second.second >= bytes ? it->second.first : nullptr;
    }
    if (gHostCacheBytes + bytes > hostCacheCapBytes()) return nullptr;
    FILE *f = fopen(path, "rb");
    if (f == nullptr) return nullptr;
    if (skip != 0 && fseek(f, (long)skip, SEEK_SET) != 0) { fclose(f); return nullptr; }
    uint8_t *buf = (uint8_t *)malloc(bytes);
    if (buf == nullptr) { fclose(f); return nullptr; }
    if (fread(buf, 1, bytes, f) != bytes) { fclose(f); free(buf); return nullptr; }
    fclose(f);
    if (cudaHostRegister(buf, bytes, cudaHostRegisterDefault) != cudaSuccess) {
        cudaGetLastError();
        free(buf);
        return nullptr;
    }
    gHostCacheBytes += bytes;
    gHostArtifactCache.emplace(std::move(key), std::make_pair(buf, bytes));
    return buf;
}

static const uint8_t *hostCacheGet(const char *path, uint64_t bytes) {
    return hostCacheGetSkip(path, bytes, 0);
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

void load_device_const_pols_gpu(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers_, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char *proofType, bool onlyFirstGPU, bool alreadyLoaded) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint64_t sizeConstPols = constSize * sizeof(Goldilocks::Element);

    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

    uint64_t const_pols_offset = initial_offset;

    const bool skipResident = noConstBufMode() && strcmp(proofType, "basic") == 0;
    if (getenv("PROOFMAN_COPY_TRACE")) {
        fprintf(stderr, "[const-load] air=%lu:%lu type=%s off=%lu pols=%.1fMB tree=%.1fMB treeFile=%s already=%d skip=%d\n",
                airgroupId, airId, proofType, (unsigned long)initial_offset,
                constSize * 8.0 / 1048576.0, constTreeSize * 8.0 / 1048576.0,
                (constTreeFilename != nullptr && constTreeFilename[0] != '\0') ? "yes" : "no",
                (int)alreadyLoaded, (int)skipResident);
    }

    // Sharing a slot with an air already uploaded: point this air's info at it, transfer
    // nothing. Layout is the same either way, so the tree offset still derives from it.
    if (alreadyLoaded) {
        for(int i=0; i<d_buffers->n_gpus; ++i){
            if (onlyFirstGPU && i > 0) break;
            AirInstanceInfo* air_instance_info = d_buffers->air_instances[key][proofType][i];
            air_instance_info->const_pols_offset = const_pols_offset;
            air_instance_info->constPolsFile = constFilename;
            air_instance_info->constPolsPackedBytes = sizeConstPols;
            if (strcmp(constTreeFilename, "") != 0 && !skipResident) {
                air_instance_info->const_tree_offset = const_pols_offset + constSize;
                air_instance_info->stored_tree = true;
            }
        }
        return;
    }

    if (skipResident) {
        // Offsets are kept as slot/reuse KEYS (adoptFixedSlot); nothing uploads.
        for(int i=0; i<d_buffers->n_gpus; ++i){
            if (onlyFirstGPU && i > 0) break;
            AirInstanceInfo* air_instance_info = d_buffers->air_instances[key][proofType][i];
            air_instance_info->const_pols_offset = const_pols_offset;
            air_instance_info->constPolsFile = constFilename;
            air_instance_info->constPolsPackedBytes = sizeConstPols;
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
        air_instance_info->constPolsFile = constFilename;
        air_instance_info->constPolsPackedBytes = sizeConstPols;
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

// No-const-buffer mode: make sure the zone's packed segment holds `air`'s
// fixed-slot blob and order `stream` after the upload. Keyed on the slot
// offset, so slot-sharing airs hit without re-staging. Inline synchronous
// staging on a miss (file -> pinned pair -> zone on the copy stream); the
// pinned pair is shared with the tree stager, hence pinnedPairMutex.
static gl64_t *ensurePackedConstPols(DeviceCommitBuffers *d_buffers, AirInstanceInfo *air, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(d_buffers->packedMutex);
    if (d_buffers->packedSlotKey != (int64_t)air->const_pols_offset) {
        FILE *f = fopen(air->constPolsFile.c_str(), "rb");
        if (f == nullptr) {
            zklog.error("ensurePackedConstPols: cannot open " + air->constPolsFile);
            exitProcess();
        }
        uint64_t bytes = air->constPolsPackedBytes;
        if (bytes > d_buffers->prefetchPackedBytes) {
            zklog.error("ensurePackedConstPols: blob " + std::to_string(bytes) +
                        " exceeds packed segment " + std::to_string(d_buffers->prefetchPackedBytes));
            exitProcess();
        }
        if (const uint8_t *cached = hostCacheGet(air->constPolsFile.c_str(), bytes)) {
            fclose(f);
            CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->packedDrained, 0));
            CHECKCUDAERR(cudaMemcpyAsync(d_buffers->prefetchPacked, cached, bytes,
                                         cudaMemcpyHostToDevice, d_buffers->prefetchStream));
        } else {
        std::lock_guard<std::mutex> plk(d_buffers->pinnedPairMutex);
        const uint64_t chunkBytes = 128ull << 20;
        uint64_t off = 0;
        int buf = 0;
        while (off < bytes) {
            uint64_t len = std::min(chunkBytes, bytes - off);
            CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchPinnedFree[buf]));
            if (fread(d_buffers->prefetchPinned[buf], 1, len, f) != len) {
                zklog.error("ensurePackedConstPols: short read on " + air->constPolsFile);
                exitProcess();
            }
            // Never overwrite content a previous launch's unpack still reads:
            // packedDrained is recorded on the consumer stream after each unpack.
            if (off == 0) CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->packedDrained, 0));
            CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_buffers->prefetchPacked + off,
                                         d_buffers->prefetchPinned[buf], len,
                                         cudaMemcpyHostToDevice, d_buffers->prefetchStream));
            CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchPinnedFree[buf], d_buffers->prefetchStream));
            off += len;
            buf ^= 1;
        }
        fclose(f);
        }
        CHECKCUDAERR(cudaEventRecord(d_buffers->packedReady, d_buffers->prefetchStream));
        d_buffers->packedSlotKey = (int64_t)air->const_pols_offset;
    }
    CHECKCUDAERR(cudaStreamWaitEvent(stream, d_buffers->packedReady, 0));
    // Caller records packedDrained on `stream` AFTER its unpack of the segment.
    return d_buffers->prefetchPacked;
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
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][proofType][gpuLocalId];

    const bool pipeline = d_buffers->pipelineMode;
    if (pipeline && air_instance_info->pinnedExpsParams == nullptr) {
        // First pipelined launch of this air: give it its own pinned exps staging
        // so the CUDA-graph capture below never writes the shared per-stream
        // buffers an in-flight replay may still read (see AirInstanceInfo). The
        // cudaMallocHost is host-side work, overlapped with the running proof.
        // x2: parity halves -- base/ext split graphs are parity-keyed and each capture bakes
        // its own half, so an odd-parity capture never rewrites content an even replay reads.
        CHECKCUDAERR(cudaMallocHost((void **)&air_instance_info->pinnedExpsParams,
                                    2 * (uint64_t)PINNED_EXPS_SLOTS * 2 * sizeof(DestParamsGPU)));
        CHECKCUDAERR(cudaMallocHost((void **)&air_instance_info->pinnedExpsArgs,
                                    2 * (uint64_t)PINNED_EXPS_SLOTS * sizeof(ExpsArguments)));
    }

    // Read the prior context before overwriting it below. Fixed columns are keyed by slot,
    // so an air sharing them with the stream's previous air reuses them; custom_fixed is
    // per-air.
    StreamData &sd = d_buffers->streamsData[streamId];
    bool reuse_custom_fixed = sd.airgroupId == airgroupId && sd.airId == airId && sd.proofType == string("basic");
    // Base/ext split: overlap this proof's early phase with the previous proof's tail --
    // only when the previous proof on this stream was the SAME air (identical layout, so the
    // base zone this proof writes is exactly the zone the previous proof's extends released).
    bool wantSplitPhases = setupCtx->starkInfo.baseSplit && pipeline && !skipRecalculation
                           && sd.airgroupId == airgroupId && sd.airId == airId && sd.proofType == string("basic");
    bool splitPhases = false;
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
        if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[ccf] air=%lu:%lu reuse=%d\n", airgroupId, airId, (int)reuse_custom_fixed);
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
        if (d_buffers->prefetchZone != nullptr && sd.gpuId == d_buffers->my_gpu_ids[0]) {
            std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
            // Find the slot holding this instance's staged witness (2 slots under the split).
            int slot = -1;
            for (uint32_t s = 0; s < d_buffers->prefetchNSlots; s++) {
                if (d_buffers->prefetchInstanceId[s] == (int64_t)instanceId &&
                    d_buffers->prefetchTraceBytes[s] == total_size) { slot = (int)s; break; }
            }
            if (slot >= 0) {
                gl64_t *slotBase = d_buffers->prefetchZone + (uint64_t)slot * d_buffers->prefetchSlotStride;
                if (wantSplitPhases) {
                    // Split phases: no zone->cm1ext landing D2D (the ext zone still belongs to
                    // the previous proof). Transpose/unpack the zone slot DIRECTLY into the base
                    // trace on the phase stream, gated on the previous proof's base-zone release
                    // and the slot upload. Enqueued under the prefetch mutex so the drained
                    // record is host-ordered before the next staging's drained wait.
                    splitPhases = true;
                    uint64_t nColsCm1 = setupCtx->starkInfo.mapSectionsN["cm1"];
                    Goldilocks::Element *traceDst = (Goldilocks::Element *)d_aux_trace + offsetStage1;
                    CHECKCUDAERR(cudaStreamWaitEvent(sd.phaseStream, sd.baseFree, 0));
                    CHECKCUDAERR(cudaStreamWaitEvent(sd.phaseStream, d_buffers->prefetchReady[slot], 0));
                    if (d_buffers->packedTrace && air_instance_info->is_packed) {
                        unpack_trace(air_instance_info, (uint64_t*)slotBase, (uint64_t*)traceDst, nColsCm1, N, sd.phaseStream, sd.timer);
                    } else {
                        fromRowMajorToColMajor(N, nColsCm1, (gl64_t *)slotBase, (gl64_t*)traceDst, resolveLayout(setupCtx->starkInfo.starkStruct.nBits, nColsCm1), sd.phaseStream);
                    }
                    CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchDrained[slot], sd.phaseStream));
                } else {
                // Hit (the common case): the worker dequeued this instance
                // ahead and its trace already uploaded to the zone on the copy
                // stream during the previous proof. No host sync -- the proof
                // stream waits on the copy's event, and trace_copy_event
                // (recorded below) gates recycling of the host buffer.
                CHECKCUDAERR(cudaStreamWaitEvent(stream, d_buffers->prefetchReady[slot], 0));
                }
            } else {
                // Miss: first proof of the phase, or the held prediction was
                // bypassed. Stage host -> the cursor slot here, host-synced so this
                // path is valid even for a caller that recycles the buffer at once.
                slot = (int)d_buffers->prefetchStageSlot;
                if (d_buffers->prefetchInstanceId[slot] != -1) {
                    zklog.warning("gen_proof: dropping stale witness prefetch of instance " +
                                  std::to_string(d_buffers->prefetchInstanceId[slot]) + " (want " +
                                  std::to_string(instanceId) + ")");
                    CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchReady[slot]));
                    d_buffers->prefetchInstanceId[slot] = -1;
                }
                gl64_t *slotBase = d_buffers->prefetchZone + (uint64_t)slot * d_buffers->prefetchSlotStride;
                CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->prefetchDrained[slot], 0));
                CHECKCUDAERR(cudaMemcpyAsync(slotBase, params->trace, total_size,
                                             cudaMemcpyHostToDevice, d_buffers->prefetchStream));
                CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchReady[slot], d_buffers->prefetchStream));
                CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchReady[slot]));
                d_buffers->prefetchStageSlot = (uint32_t)(slot + 1) % d_buffers->prefetchNSlots;
            }
            d_buffers->prefetchInstanceId[slot] = -1;
            d_buffers->prefetchTraceBytes[slot] = 0;
            if (!splitPhases) {
            gl64_t *slotBase = d_buffers->prefetchZone + (uint64_t)slot * d_buffers->prefetchSlotStride;
            CHECKCUDAERR(cudaMemcpyAsync(dst, slotBase, total_size,
                                         cudaMemcpyDeviceToDevice, stream));
            CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchDrained[slot], stream));
            }
            // Host-buffer release gate: the copy stream's tail is at/after this
            // trace's H2D, so the event fires when the HOST buffer is free -- not
            // when the proof runs (under pipelining the D2D executes much later).
            CHECKCUDAERR(cudaEventRecord(d_buffers->streamsData[streamId].trace_copy_event, d_buffers->prefetchStream));
        } else {
            if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[site-GENPROOF-NOZONE]\n");
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
    const uint32_t pinnedSlot = (pipeline && !skipRecalculation) ? (uint32_t)(d_buffers->streamsData[streamId].launchSeq & 1) : 0;
    Goldilocks::Element *aux_values = d_buffers->streamsData[streamId].pinned_aux_values + (uint64_t)pinnedSlot * PINNED_AUX_VALUES_MAX;
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

    // Base/ext split parity: land the smalls in the parity copy for odd-parity proofs
    // (must match genProof_gpu's smallsShift, same launchSeq read).
    const uint64_t smallsShiftUp = (setupCtx->starkInfo.baseSplit && pinnedSlot)
        ? setupCtx->starkInfo.mapOffsets[std::make_pair("smalls_parity", false)] - setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)]
        : 0;
    // Split phases: the upload must not queue behind the previous proof's tail on the main
    // stream, or phase-A (which needs publics/airValues for STD2) serializes after it and the
    // whole overlap collapses (measured: main idles ~220ms waiting phaseADone). The copy stream
    // is free; the parity smalls region's previous user (the same-parity proof two back) is
    // long gone.
    static const int earlyUp = [](){ const char* v = std::getenv("PROOFMAN_SPLIT_EARLY_UPLOAD"); return v ? atoi(v) : 0; }();
    cudaStream_t smallsStream = (splitPhases && (earlyUp & 1)) ? sd.phaseStream : stream;
    CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)(d_aux_trace + offsetPublicInputs + smallsShiftUp), aux_values, totalCopySize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, smallsStream));
    CHECKCUDAERR(cudaEventRecord(sd.smallsUp, smallsStream));

    gl64_t *d_const_pols;
    if (d_buffers->prefetchPacked != nullptr) {
        // No-const-buffer mode: only the unpack reads the packed blob, so stage
        // it only when this launch will unpack (genProof gates on reuse).
        d_const_pols = reuse_const_tree ? d_buffers->prefetchPacked
                                        : ensurePackedConstPols(d_buffers, air_instance_info, stream);
    } else {
        d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
    }
    gl64_t *d_const_tree;
    if (air_instance_info->stored_tree) {
        // Preallocated in the const buffer, so it is in place unconditionally.
        d_const_tree = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_tree_offset;
        reuse_const_tree = reuse_constants;
    } else {
        // find(), not operator[]: a preallocate-layout starkinfo has NO aux tree
        // slot, and operator[] would silently insert offset 0 -- the merkelize
        // would then write the tree over the aux BASE (14 corrupted Mains'
        // worth of debugging, 2026-08-18).
        auto itConstTree = setupCtx->starkInfo.mapOffsets.find(std::make_pair("const", true));
        if (itConstTree == setupCtx->starkInfo.mapOffsets.end()) {
            zklog.error("gen_proof: air " + std::to_string(airgroupId) + ":" + std::to_string(airId) +
                        " has no aux const-tree slot (preallocate layout) but stored_tree is false");
            exitProcess();
        }
        uint64_t offsetConstTree = itConstTree->second;
        d_const_tree = d_aux_trace + offsetConstTree;

        // calculateFixedExtended airs merkelize inside genProof_gpu instead, on the same
        // flag -- either way ("const", true) holds this slot's tree on exit.
        if (!reuse_const_tree && !setupCtx->starkInfo.calculateFixedExtended) {
            if (d_buffers->prefetchFixed != nullptr) {
                std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
                // An early stager thread may still be filling the fixed half --
                // for THIS air (then we consume it below) or, on a misprediction,
                // for another (then the inline route below would race its writes).
                // Either way: wait the staging out. The stager clears the claim
                // on failure, which also ends the wait.
                while (d_buffers->prefetchFixedAirgroup != -1 && d_buffers->prefetchFixedSize == 0) {
                    d_buffers->prefetchFixedMutex.unlock();
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                    d_buffers->prefetchFixedMutex.lock();
                }
                bool hit = d_buffers->prefetchFixedAirgroup == (int64_t)airgroupId &&
                           d_buffers->prefetchFixedAir == (int64_t)airId &&
                           d_buffers->prefetchFixedSize == sizeConstTree;
                if (!hit) {
                    if (d_buffers->prefetchFixedAirgroup != -1) {
                        zklog.warning("gen_proof: dropping stale fixed prefetch of air " +
                                      std::to_string(d_buffers->prefetchFixedAirgroup) + ":" +
                                      std::to_string(d_buffers->prefetchFixedAir) + " (want " +
                                      std::to_string(airgroupId) + ":" + std::to_string(airId) + ")");
                        CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchFixedReady));
                    }
                    // Inline staging: file -> pinned -> zone on the copy stream.
                    // The pinned-free sync only waits for the previous chunk's
                    // H2D, never for the proof stream's kernel backlog.
                    if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[site-TREE-ZONESTAGE] size=%lu\n", sizeConstTree);
                    FILE *f = fopen(constTreePath, "rb");
                    if (f == nullptr) {
                        zklog.error("gen_proof: cannot open const tree " + std::string(constTreePath));
                        exitProcess();
                    }
                    CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->prefetchFixedDrained, 0));
                    if (const uint8_t *cached = hostCacheGet(constTreePath, sizeConstTree)) {
                        fclose(f);
                        CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_buffers->prefetchFixed, cached, sizeConstTree,
                                                     cudaMemcpyHostToDevice, d_buffers->prefetchStream));
                        CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchFixedReady, d_buffers->prefetchStream));
                        goto treeStaged;
                    }
                    {
                    std::lock_guard<std::mutex> pplk(d_buffers->pinnedPairMutex);
                    const uint64_t chunkBytes = 128ull << 20;
                    uint64_t off = 0;
                    int buf = 0;
                    while (off < sizeConstTree) {
                        uint64_t len = std::min(chunkBytes, sizeConstTree - off);
                        CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchPinnedFree[buf]));
                        if (fread(d_buffers->prefetchPinned[buf], 1, len, f) != len) {
                            zklog.error("gen_proof: short read on const tree " + std::string(constTreePath));
                            exitProcess();
                        }
                        CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_buffers->prefetchFixed + off,
                                                     d_buffers->prefetchPinned[buf], len,
                                                     cudaMemcpyHostToDevice, d_buffers->prefetchStream));
                        CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchPinnedFree[buf], d_buffers->prefetchStream));
                        off += len;
                        buf ^= 1;
                    }
                    fclose(f);
                    CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchFixedReady, d_buffers->prefetchStream));
                    }
                    treeStaged:;
                }
                d_buffers->prefetchFixedAirgroup = -1;
                d_buffers->prefetchFixedAir = -1;
                d_buffers->prefetchFixedSize = 0;
                CHECKCUDAERR(cudaStreamWaitEvent(stream, d_buffers->prefetchFixedReady, 0));
                CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_const_tree, d_buffers->prefetchFixed,
                                             sizeConstTree, cudaMemcpyDeviceToDevice, stream));
                CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchFixedDrained, stream));
            } else {
                if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[site-TREE-FALLBACK] size=%lu\n", sizeConstTree);
                load_and_copy_to_device_in_chunks(d_buffers, constTreePath, (uint8_t*)d_const_tree, sizeConstTree, streamId);
            }
        }
    }
    sd.constTreeResident = true;


    proofman_sumcheck_set_context(instanceId, airgroupId, airId);
    // The tree-aware flag, not the slot one: genProof_gpu's reuse also gates the
    // calculateFixedExtended merkelize, and a slot claimed by commit_witness has pols but no
    // tree. Costs a redundant unpack in exactly that case.
    static const bool pbTraceB = [] {
        const char *e = getenv("PROOFMAN_PHASE_B_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    if (pbTraceB) {
        fprintf(stderr, "[PBTRACE] basic launch inst=%lu air=%lu:%lu stream=%u skipRecalc=%d launchSeq=%lu split=%d\n",
                instanceId, airgroupId, airId, streamId, (int)skipRecalculation,
                (unsigned long)sd.launchSeq, (int)splitPhases);
    }

    if (customFixedRebuilt) CHECKCUDAERR(cudaStreamWaitEvent(stream, sd.customFixedDone, 0));
    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, constTreePath, streamId, instanceId, d_buffers, air_instance_info, skipRecalculation, timer, stream, selfContained, reuse_const_tree, splitPhases);
    if (pipeline) {
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
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

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
        if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[ccf] air=%lu:%lu reuse=%d\n", airgroupId, airId, (int)reuse_custom_fixed);
        if (!reuse_custom_fixed) {
            Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            rebuildCustomCommitsFixed(d_buffers, setupCtx, air_instance_info, gpuLocalId, pCustomCommitsFixed, sd, timer);
            customFixedRebuilt = true;
        }
    }

    uint64_t total_size = (d_buffers->packedTrace && air_instance_info->is_packed) ? air_instance_info->num_packed_words * N * sizeof(Goldilocks::Element) : N * nCols * sizeof(Goldilocks::Element);
    uint64_t *dst = (uint64_t *)(d_aux_trace + offsetStage1 + N * nCols);
    if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[site-INITINST]\n");
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
    // Contributions/initialize path: never pipelined, slot 0.
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
    
    gl64_t *d_const_pols;
    if (d_buffers->prefetchPacked != nullptr) {
        d_const_pols = reuse_constants ? d_buffers->prefetchPacked
                                       : ensurePackedConstPols(d_buffers, air_instance_info, stream);
    } else {
        d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
    }
    
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;
    if (!reuse_constants) {
        unpack_fixed((uint64_t*)d_const_pols, (uint64_t*)(d_const_pols + 1), (uint64_t*)(d_const_pols + 1 + setupCtx->starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants, N, stream, timer);
        CHECKCUDAERR(cudaGetLastError());
        if (d_buffers->prefetchPacked != nullptr) {
            CHECKCUDAERR(cudaEventRecord(d_buffers->packedDrained, stream));
        }
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
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

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
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

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
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

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
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;
    
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
    const bool compactWitness = mapCols > 0 && mapCols < nCols && air_instance_info->d_witness_compact != nullptr;
    // Getting the witness onto the device happens BEFORE genProof_gpu opens STARK_GPU_PROOF, so it
    // is a timer of its own rather than a category: a category here would be divided by a window
    // that does not contain it, which is what made that table total 102% with OTHER pinned at zero.
    TimerStartGPU(timer, STARK_GPU_WITNESS);
    // Prefetched witness: staged on the copy stream at dispatch time (see
    // prefetch_recursive_witness_gpu), so it overlapped the previous proof's
    // kernels. Consume from the slot; fall through to the inline upload on miss.
    int recSlot = -1;
    const uint64_t upBytes = compactWitness ? N * mapCols * sizeof(Goldilocks::Element) : sizeTrace;
    if (d_buffers->prefetchZone != nullptr) {
        // The whole consume -- lookup AND the wait/read/drained-record enqueues -- runs
        // under the mutex, so the drained record is HOST-ORDERED before any later
        // staging's drained wait (the stager enqueues under this same mutex). Enqueuing
        // outside the lock let another worker's staging bind its overwrite-wait to the
        // PREVIOUS drained instance and race the widen (invalid rec1 proofs under the
        // two-stream phase-B drain, ~30%/run).
        std::lock_guard<std::mutex> lk(d_buffers->recWitMutex);
        for (int sIdx = 0; sIdx < 2; sIdx++) {
            if (d_buffers->recWitKey[sIdx] == trace && d_buffers->recWitBytes[sIdx] == upBytes) { recSlot = sIdx; break; }
        }
        if (recSlot >= 0) {
            d_buffers->recWitKey[recSlot] = nullptr;
            // PROOFMAN_RECWIT_CHECK=1: deterministic staging validation -- compare the
            // slot's device content against the host trace before the widen consumes it.
            static const bool recwitCheck = [] {
                const char *e = getenv("PROOFMAN_RECWIT_CHECK");
                return e != nullptr && e[0] == '1';
            }();
            // PROOFMAN_RECWIT_SYNC=1: host-side completion barrier only (no data check).
            static const bool recwitSync = [] {
                const char *e = getenv("PROOFMAN_RECWIT_SYNC");
                return e != nullptr && e[0] == '1';
            }();
            if (recwitSync && !recwitCheck) {
                CHECKCUDAERR(cudaEventSynchronize(d_buffers->recWitReady[recSlot]));
            }
            if (recwitCheck) {
                CHECKCUDAERR(cudaEventSynchronize(d_buffers->recWitReady[recSlot]));
                static std::mutex chkMx;
                std::lock_guard<std::mutex> cg(chkMx);
                static std::vector<uint8_t> chk;
                chk.resize(upBytes);
                CHECKCUDAERR(cudaMemcpy(chk.data(), d_buffers->recWitSlot[recSlot], upBytes, cudaMemcpyDeviceToHost));
                if (memcmp(chk.data(), trace, upBytes) != 0) {
                    uint64_t firstBad = 0;
                    for (uint64_t o = 0; o < upBytes; o++) { if (chk[o] != ((const uint8_t*)trace)[o]) { firstBad = o; break; } }
                    fprintf(stderr, "[RECWIT-CHECK] MISMATCH air %lu:%lu type %s slot %d bytes %lu firstBad @%lu\n",
                            airgroupId, airId, proofType, recSlot, upBytes, firstBad);
                } else {
                    fprintf(stderr, "[RECWIT-CHECK] ok air %lu:%lu slot %d\n", airgroupId, airId, recSlot);
                }
            }
            CHECKCUDAERR(cudaStreamWaitEvent(stream, d_buffers->recWitReady[recSlot], 0));
            if (compactWitness) {
                CHECKCUDAERR(cudaMemsetAsync((uint8_t*)(d_aux_trace + offsetStage1Extended), 0, sizeTrace, stream));
                widenCompactWitnessGPU((uint64_t*)(d_aux_trace + offsetStage1Extended), nCols, N,
                                       (const uint64_t*)d_buffers->recWitSlot[recSlot], mapCols, stream);
            } else {
                CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)(d_aux_trace + offsetStage1Extended),
                                             d_buffers->recWitSlot[recSlot], sizeTrace,
                                             cudaMemcpyDeviceToDevice, stream));
            }
            CHECKCUDAERR(cudaEventRecord(d_buffers->recWitDrained[recSlot], stream));
            // Host-buffer release gate: the staging H2D ran on the COPY stream (pinned
            // source, truly asynchronous); gate the pool lease on the copy stream tail.
            CHECKCUDAERR(cudaEventRecord(d_buffers->streamsData[streamId].trace_copy_event, d_buffers->prefetchStream));
        }
    }
    if (recSlot >= 0) {
        // consumed above, under the mutex
    } else if (compactWitness) {
        CHECKCUDAERR(cudaMemsetAsync((uint8_t*)(d_aux_trace + offsetStage1Extended), 0, sizeTrace, stream));
        if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[site-REC-MISS-COMPACT]\n");
        // Phase-B: the per-air compact buffer lives in the scratch arena, which the second
        // aliased rec stream overlays. Bounce through the per-stream spare scratch instead.
        const uint64_t *compactDst = (const uint64_t *)air_instance_info->d_witness_compact;
        if (d_buffers->phaseBAliased && d_buffers->phaseBState.load(std::memory_order_acquire) == 1 &&
            d_buffers->streamsData[streamId].recursive &&
            d_buffers->phaseBMissScratch[d_buffers->streamsData[streamId].localStreamId] != nullptr) {
            compactDst = (const uint64_t *)d_buffers->phaseBMissScratch[d_buffers->streamsData[streamId].localStreamId];
        }
        copy_to_device_in_chunks(d_buffers, trace, (uint8_t*)compactDst,
                                 N * mapCols * sizeof(Goldilocks::Element), streamId, timer, false);
        widenCompactWitnessGPU((uint64_t*)(d_aux_trace + offsetStage1Extended), nCols, N,
                               compactDst, mapCols, stream);
    } else {
        if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[site-REC-MISS-FULL]\n");
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
    // Parity slot under the depth-2 ring: the previous in-flight proof's async H2D
    // may still read its half of the pinned region.
    const uint32_t recPinSlot = (d_buffers->pipelineMode && !sd.recursive)
        ? (uint32_t)(sd.launchSeq & 1) : 0;
    Goldilocks::Element *pinned_publics = d_buffers->streamsData[streamId].pinned_aux_values
        + (uint64_t)recPinSlot * PINNED_AUX_VALUES_MAX;
    memcpy(pinned_publics, pPublicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element));
    CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)(d_aux_trace + offsetPublicInputs), pinned_publics, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));

    gl64_t *d_const_pols = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_pols_offset;
    gl64_t *d_const_tree;
    if (air_instance_info->stored_tree) {
        // Preallocated in the const buffer, so it is in place unconditionally.
        d_const_tree = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_tree_offset;
        reuse_const_tree = reuse_constants;
    } else {
        uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
        d_const_tree = d_aux_trace + offsetConstTree;

        if (!reuse_const_tree) {
            // Recursive circuits without a preallocated tree slot reload their tree on
            // every air switch. Preference order: (1) the zone's fixed segment, staged
            // on the copy stream at dispatch time (prefetch_recursive_tree) -- a D2D
            // here, fully overlapped; (2) the pinned host cache -- one direct DMA;
            // (3) the chunked file loader.
            bool zoneHit = false;
            const bool pbActive = d_buffers->phaseBAliased &&
                                  d_buffers->phaseBState.load(std::memory_order_acquire) == 1;
            if (!pbActive && d_buffers->prefetchFixed != nullptr && sizeConstTree <= d_buffers->prefetchFixedBytes) {
                std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
                // Wait out an in-flight staging only when it targets OUR air (a foreign
                // staging never races this path: the fallbacks below write d_const_tree,
                // not the zone). Size stays 0 while staging is in flight.
                while (d_buffers->prefetchFixedAirgroup == (int64_t)airgroupId &&
                       d_buffers->prefetchFixedAir == (int64_t)airId &&
                       d_buffers->prefetchFixedSize == 0) {
                    d_buffers->prefetchFixedMutex.unlock();
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                    d_buffers->prefetchFixedMutex.lock();
                }
                if (d_buffers->prefetchFixedAirgroup == (int64_t)airgroupId &&
                    d_buffers->prefetchFixedAir == (int64_t)airId &&
                    d_buffers->prefetchFixedSize == sizeConstTree) {
                    d_buffers->prefetchFixedAirgroup = -1;
                    d_buffers->prefetchFixedAir = -1;
                    d_buffers->prefetchFixedSize = 0;
                    CHECKCUDAERR(cudaStreamWaitEvent(stream, d_buffers->prefetchFixedReady, 0));
                    CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_const_tree, d_buffers->prefetchFixed,
                                                 sizeConstTree, cudaMemcpyDeviceToDevice, stream));
                    CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchFixedDrained, stream));
                    zoneHit = true;
                }
            }
            if (!zoneHit) {
                if (const uint8_t *cached = hostCacheGet(constTreePath, sizeConstTree)) {
                    CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)d_const_tree, cached, sizeConstTree,
                                                 cudaMemcpyHostToDevice, stream));
                } else if (getenv("PROOFMAN_REC_TREE_COMPUTE") == nullptr ||
                           getenv("PROOFMAN_REC_TREE_COMPUTE")[0] != '0') {
                    // Cache miss: REBUILD the tree on device from the resident packed
                    // aggregation const pols (unpack -> LDE -> merkelize, ~50-70 ms of
                    // busy GPU) instead of the chunked disk load (~160 ms of copies plus
                    // ~17 ms of GPU idle per 128 MB fread). Same code path VadcopFinal
                    // uses (calculate_const_tree_fixed_gpu); scratch stays inside the
                    // tree area, so the staged witness in cm1ext is untouched.
                    if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[site-TREE-COMPUTE] size=%lu\n", sizeConstTree);
                    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
                    Goldilocks::Element *packed_const_pols = (Goldilocks::Element *)d_const_pols;
                    Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;
                    unpack_fixed((uint64_t*)d_const_pols, (uint64_t*)(packed_const_pols + 1),
                                 (uint64_t*)(packed_const_pols + 1 + setupCtx->starkInfo.nConstants),
                                 (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants,
                                 1ull << setupCtx->starkInfo.starkStruct.nBits, stream, timer);
                    extendAndMerkelizeFixed(*setupCtx, d_const_pols_unpacked,
                                            (Goldilocks::Element *)d_const_tree, true, timer, stream);
                } else {
                    load_and_copy_to_device_in_chunks(d_buffers, constTreePath, (uint8_t*)d_const_tree, sizeConstTree, streamId);
                }
            }
        }
    }
    sd.constTreeResident = true;

    static const bool pbTrace = [] {
        const char *e = getenv("PROOFMAN_PHASE_B_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    if (pbTrace) {
        fprintf(stderr, "[PBTRACE] rec launch inst=%lu air=%lu:%lu type=%s stream=%u rec=%d slotHit=%d reuseTree=%d storedTree=%d\n",
                instanceId, airgroupId, airId, proofType, streamId,
                (int)d_buffers->streamsData[streamId].recursive, (int)(recSlot >= 0),
                (int)reuse_const_tree, (int)air_instance_info->stored_tree);
    }
    // See gen_proof_gpu: the tree-aware flag, not the slot one.
    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, constTreePath, streamId, instanceId, d_buffers, air_instance_info, false, timer, stream, true, reuse_const_tree);
    if (d_buffers->pipelineMode && !sd.recursive) {
        // Phase-split protocol: a recursive proof owns the WHOLE stream buffer (no
        // base/ext split in its layout), so the next basic's early phase must order
        // after it. Recording baseFree here makes the recursive a phase barrier;
        // without it the next basic transposes its witness over live recursive state.
        CHECKCUDAERR(cudaEventRecord(sd.baseFree, stream));
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
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

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
    uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
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
    // Contributions upload/compute ping-pong: stage this instance's trace into a
    // zone witness slot on the COPY stream BEFORE claiming the compute stream.
    // With one basic stream and two completion workers, worker B's upload here
    // overlaps worker A's in-flight commit kernels; the consume below is a D2D.
    // Skip (legacy inline upload) when the zone is absent, the trace exceeds the
    // slot, or the cursor slot still holds an unconsumed fresh entry.
    {
        SetupCtx *sc = (SetupCtx *)pSetupCtx_;
        StepsParams *pp = (StepsParams *)params_;
        auto k0 = std::make_pair(airgroupId, airId);
        auto itA = d_buffers->air_instances.find(k0);
        AirInstanceInfo *ai0 = (itA != d_buffers->air_instances.end() && itA->second.count("basic"))
                                   ? itA->second["basic"][d_buffers->gpus_g2l[d_buffers->my_gpu_ids[0]]]
                                   : nullptr;
        // NEVER while gpu-mops borrows the first GPU's unified buffer: the glued
        // zone lives inside it, and a slot upload would clobber mops device state.
        if (d_buffers->prefetchZone != nullptr && ai0 != nullptr && pp->trace != nullptr &&
            d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire) == 0) {
            uint64_t N0 = 1ull << sc->starkInfo.starkStruct.nBits;
            uint64_t nCols0 = sc->starkInfo.mapSectionsN["cm1"];
            uint64_t upBytes = (d_buffers->packedTrace && ai0->is_packed)
                                   ? ai0->num_packed_words * N0 * sizeof(Goldilocks::Element)
                                   : N0 * nCols0 * sizeof(Goldilocks::Element);
            std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
            uint32_t slot = d_buffers->prefetchStageSlot;
            if (upBytes <= d_buffers->prefetchSlotStride * sizeof(gl64_t)
                && d_buffers->prefetchInstanceId[slot] == -1) {
                cudaSetDevice(d_buffers->my_gpu_ids[0]);
                gl64_t *slotBase = d_buffers->prefetchZone + (uint64_t)slot * d_buffers->prefetchSlotStride;
                CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->prefetchDrained[slot], 0));
                CHECKCUDAERR(cudaMemcpyAsync(slotBase, pp->trace, upBytes,
                                             cudaMemcpyHostToDevice, d_buffers->prefetchStream));
                CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchReady[slot], d_buffers->prefetchStream));
                d_buffers->prefetchInstanceId[slot] = (int64_t)instanceId;
                d_buffers->prefetchTraceBytes[slot] = upBytes;
                d_buffers->prefetchStageSlot = (slot + 1) % d_buffers->prefetchNSlots;
            }
        }
    }
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
    // Zone mode (single-compute-stream prefetch): the resident-witness fast
    // path is disabled for uniformity -- every proof takes the recompute route
    // through the zone.
    sd.witnessResident =
        d_buffers->prefetchZone == nullptr && sd.auxTraceCapacity >= setupCtx->starkInfo.mapTotalN;

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
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;
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
    // Consume the zone slot staged at the top of this call (or by a concurrent
    // worker): the H2D already ran on the copy stream, overlapping the previous
    // commit's kernels; landing it is a device-side copy. Miss -> legacy inline.
    {
        int slot = -1;
        if (d_buffers->prefetchZone != nullptr) {
            std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
            for (uint32_t sIdx = 0; sIdx < d_buffers->prefetchNSlots; sIdx++) {
                if (d_buffers->prefetchInstanceId[sIdx] == (int64_t)instanceId &&
                    d_buffers->prefetchTraceBytes[sIdx] == total_size) { slot = (int)sIdx; break; }
            }
            if (slot >= 0) {
                d_buffers->prefetchInstanceId[slot] = -1;
                d_buffers->prefetchTraceBytes[slot] = 0;
                gl64_t *slotBase = d_buffers->prefetchZone + (uint64_t)slot * d_buffers->prefetchSlotStride;
                CHECKCUDAERR(cudaStreamWaitEvent(stream, d_buffers->prefetchReady[slot], 0));
                CHECKCUDAERR(cudaMemcpyAsync(dst, slotBase, total_size, cudaMemcpyDeviceToDevice, stream));
                CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchDrained[slot], stream));
                // Host-buffer release gate (see gen_proof_gpu): the copy stream's
                // tail is at/after this trace's H2D.
                CHECKCUDAERR(cudaEventRecord(d_buffers->streamsData[streamId].trace_copy_event, d_buffers->prefetchStream));
            }
        }
        if (slot < 0) {
            if (getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[site-CONTRIB]\n");
            copy_to_device_in_chunks(d_buffers, params->trace, dst, total_size, streamId, timer);
        }
    }
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
        // Claims the slot but not constTreeResident: this never touches the const tree.
        bool needUnpack = !sd.adoptFixedSlot(air_instance_info->const_pols_offset, offsetConstPols, false, "")
            || setupCtx->starkInfo.constPolsAliasTree;
        gl64_t *d_const_pols;
        if (d_buffers->prefetchPacked != nullptr) {
            d_const_pols = needUnpack ? ensurePackedConstPols(d_buffers, air_instance_info, stream)
                                      : d_buffers->prefetchPacked;
        } else {
            d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
        }
        gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];
        Goldilocks::Element *packed_const_pols = (Goldilocks::Element *)d_const_pols;
        Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;
        uint64_t* d_num_packed_words = (uint64_t*) d_const_pols;
        if (needUnpack) {
            unpack_fixed(d_num_packed_words, (uint64_t*)(packed_const_pols + 1), (uint64_t*)(packed_const_pols + 1 + setupCtx->starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants, N, stream, timer);
            CHECKCUDAERR(cudaGetLastError());
            if (d_buffers->prefetchPacked != nullptr) {
                CHECKCUDAERR(cudaEventRecord(d_buffers->packedDrained, stream));
            }
        }

        // Rebuild now, wait just before the commit body below reads it.
        if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0 && getenv("PROOFMAN_COPY_TRACE")) fprintf(stderr, "[ccf-contrib] air=%lu:%lu reuse=%d\n", airgroupId, airId, (int)reuse_custom_fixed);
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
    closeStreamTimer(d_buffers->streamsData[streamId].timer, instanceId, airgroupId, airId, false);
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
// Allocate the witness prefetch zone on the first GPU: `bytes` must cover the
// largest basic trace (packed or full). Idempotent per process run.
void configure_prefetch_zone_gpu(void *d_buffers_, uint64_t witnessBytes, uint64_t fixedTreeBytes, uint64_t packedConstBytes, uint64_t recWitnessBytes) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || witnessBytes == 0 || d_buffers->prefetchZone != nullptr) return;
    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    // Two witness slots under the phase split: staging k+1 must not wait for k's transpose.
    // Two witness slots always: depth-2 for the dispatch-ahead proof staging AND
    // the contributions-phase upload/compute ping-pong (commit_witness pre-stages
    // instance i+1 on the copy stream while instance i computes).
    d_buffers->prefetchNSlots = 2;
    uint64_t witnessArea = witnessBytes * d_buffers->prefetchNSlots;
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
    {
        d_buffers->prefetchZone = d_buffers->prefetchRegionBase;
        if (recWitnessBytes > 0) {
            uint8_t *recBase = (uint8_t *)d_buffers->prefetchRegionBase + witnessArea + fixedTreeBytes + packedConstBytes;
            for (int r = 0; r < 2; r++) {
                d_buffers->recWitSlot[r] = (gl64_t *)(recBase + (uint64_t)r * recWitnessBytes);
                d_buffers->recWitSlotBytes[r] = recWitnessBytes;
            }
            d_buffers->recWitCarved = true;
        }
        zklog.info("Prefetch zone armed (" + std::to_string(needed >> 20) + " MB of " +
                   std::to_string(d_buffers->prefetchRegionBytes >> 20) + " MB region)");
    }
    CHECKCUDAERR(cudaStreamCreateWithFlags(&d_buffers->prefetchStream, cudaStreamNonBlocking));
    for (int s = 0; s < 2; s++) {
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->prefetchReady[s], cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->prefetchDrained[s], cudaEventDisableTiming));
        d_buffers->prefetchInstanceId[s] = -1;
    }
    d_buffers->prefetchZoneBytes = witnessBytes;
    if (fixedTreeBytes > 0) {
        d_buffers->prefetchFixed = d_buffers->prefetchZone + witnessArea / sizeof(gl64_t);
        d_buffers->prefetchFixedBytes = fixedTreeBytes;
        CHECKCUDAERR(cudaMallocHost((void **)&d_buffers->prefetchPinned[0], 128ull << 20));
        CHECKCUDAERR(cudaMallocHost((void **)&d_buffers->prefetchPinned[1], 128ull << 20));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->prefetchFixedReady, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->prefetchFixedDrained, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->prefetchPinnedFree[0], cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->prefetchPinnedFree[1], cudaEventDisableTiming));
        // Lowest-priority side stream: rec-tree PRE-builds fill the in-flight
        // proof's SM gaps instead of serializing in front of the next proof.
        int prLo = 0, prHi = 0;
        CHECKCUDAERR(cudaDeviceGetStreamPriorityRange(&prLo, &prHi));
        CHECKCUDAERR(cudaStreamCreateWithPriority(&d_buffers->treeBuildStream, cudaStreamNonBlocking, prLo));
        d_buffers->treeBuildTimer.init(d_buffers->treeBuildStream);
        d_buffers->treeBuildTimer.enabled = false;
    }
    if (packedConstBytes > 0) {
        d_buffers->prefetchPacked = d_buffers->prefetchZone + (witnessBytes * d_buffers->prefetchNSlots + fixedTreeBytes) / sizeof(gl64_t);
        d_buffers->prefetchPackedBytes = packedConstBytes;
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->packedReady, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->packedDrained, cudaEventDisableTiming));
    }
    for (int s = 0; s < 2; s++) {
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->recWitReady[s], cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&d_buffers->recWitDrained[s], cudaEventDisableTiming));
    }
    zklog.info("Prefetch zone: " + std::to_string(witnessBytes >> 20) + " MB witness + " +
               std::to_string(fixedTreeBytes >> 20) + " MB fixed + " +
               std::to_string(packedConstBytes >> 20) + " MB packed const pols on GPU " +
               std::to_string(d_buffers->my_gpu_ids[0]));
}

// Stage the const TREE file for (airgroupId, airId) into the zone's fixed
// half: disk -> pinned chunk -> zone, on the copy stream. Runs on the Rust
// prefetch worker thread, so per-chunk synchronization here costs no compute
// overlap. Returns 0 staged, negative = busy/oversized/io-error.
// Phase-B configuration (before gen_device_streams/alloc): mark the two recursive
// streams as aliased over the pre-const area. Sizing and eligibility key off this.
void configure_phase_b_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    d_buffers->phaseBAliased = true;
}

// Phase-B state transitions. Returns 0 on success, negative when unusable.
//   1: every basic+compressor completed -> quiesce the basic stream, relocate the
//      rec-witness slots + miss scratch into the spare above the aliases, open the pair.
//   2: recursion complete -> drain the pair, hand the basic stream back (VadcopFinal).
//   0: job start -> back to phase A.
int64_t set_phase_b_gpu(void *d_buffers_, uint32_t state) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (!d_buffers->phaseBAliased) return -1;
    if (state == 1 && d_buffers->phaseBSpareBase == nullptr) return -2;
    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    if (state == 1) {
        // The trigger runs after the LAST basic/compressor completion was collected,
        // so the basic stream should already be idle; sync anyway for a hard fence.
        for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
            if (!d_buffers->streamsData[i].recursive) {
                CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[i].stream));
            }
        }
        // Relocate the rec-witness slots (their zone home is inside the second alias).
        {
            std::lock_guard<std::mutex> lk(d_buffers->recWitMutex);
            uint64_t slotBytes = d_buffers->recWitSlotBytes[0];
            if (slotBytes == 0) slotBytes = 256ull << 20;
            uint64_t need = 4 * slotBytes;  // 2 slots + 2 miss-scratch
            if (need > d_buffers->phaseBSpareBytes) return -3;
            for (int r = 0; r < 2; r++) {
                d_buffers->recWitSlot[r] = (gl64_t *)(d_buffers->phaseBSpareBase + (uint64_t)r * slotBytes);
                d_buffers->recWitSlotBytes[r] = slotBytes;
                d_buffers->recWitKey[r] = nullptr;
                d_buffers->recWitBytes[r] = 0;
            }
            d_buffers->recWitCarved = true;
            for (int r = 0; r < 2; r++) {
                d_buffers->phaseBMissScratch[r] = d_buffers->phaseBSpareBase + (2 + (uint64_t)r) * slotBytes;
            }
        }
        // Basic-witness zone entries die with the basic stream.
        {
            std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
            for (uint32_t sl = 0; sl < d_buffers->prefetchNSlots; sl++) d_buffers->prefetchInstanceId[sl] = -1;
        }
        {
            std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
            if (d_buffers->prefetchFixedAirgroup != -1 && d_buffers->prefetchFixedSize == 0) {
                d_buffers->prefetchFixedAbandon = true;  // an in-flight prestage lands into B2: poison it
            } else {
                d_buffers->prefetchFixedAirgroup = -1;
                d_buffers->prefetchFixedAir = -1;
                d_buffers->prefetchFixedSize = 0;
            }
        }
        // Any prestage kernels already queued on the build stream write into B2: fence them
        // out before the pair opens.
        if (d_buffers->treeBuildStream != nullptr) CHECKCUDAERR(cudaStreamSynchronize(d_buffers->treeBuildStream));
    } else if (state == 2) {
        for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
            if (d_buffers->streamsData[i].recursive) {
                CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[i].stream));
            }
        }
    }
    d_buffers->phaseBState.store(state, std::memory_order_release);
    return 0;
}

int64_t prefetch_fixed_gpu(void *d_buffers_, uint64_t airgroupId, uint64_t airId,
                           const char *constTreePath, uint64_t bytes);

// Self-staged rec-tree PRE-computation: called at gen_recursive entry, BEFORE the
// stream wait, while the previous proof still owns the compute stream. Rebuilds
// this circuit's const tree from the resident packed aggregation pols into the
// zone's fixed segment on the low-priority treeBuildStream (unpack scratch sits
// in the segment's tail; extendAndMerkelize's node scratch is inside the tree
// area). The consume in gen_recursive lands it with one D2D. Claims follow the
// prefetch_fixed protocol (Size==0 while in flight); a busy segment or any
// mismatch simply falls back to the inline rebuild.
static void prestageRecTreeCompute(DeviceCommitBuffers *d_buffers, SetupCtx *setupCtx,
                                   uint64_t airgroupId, uint64_t airId, const char *proofType) {
    if (d_buffers->prefetchFixed == nullptr || d_buffers->treeBuildStream == nullptr) return;
    if (d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire) != 0) return;
    // Phase-B: the fixed segment lies inside the second aliased rec stream's aux.
    if (d_buffers->phaseBAliased && d_buffers->phaseBState.load(std::memory_order_acquire) == 1) return;
    const char *e = getenv("PROOFMAN_REC_TREE_PRESTAGE");
    if (e != nullptr && e[0] == '0') return;
    auto key = std::make_pair(airgroupId, airId);
    auto itK = d_buffers->air_instances.find(key);
    if (itK == d_buffers->air_instances.end()) return;
    auto itT = itK->second.find(std::string(proofType));
    if (itT == itK->second.end() || itT->second.empty()) return;
    AirInstanceInfo *air = itT->second[0];
    if (air == nullptr || air->stored_tree) return;
    // Skip when some stream already holds this circuit warm (reuse_const_tree will
    // short-circuit there and the staged entry would only go stale).
    for (uint32_t si = 0; si < d_buffers->n_total_streams; si++) {
        StreamData &wsd = d_buffers->streamsData[si];
        if (wsd.constTreeResident && wsd.airgroupId == airgroupId && wsd.airId == airId &&
            wsd.proofType == std::string(proofType)) return;
    }
    uint64_t N = 1ull << setupCtx->starkInfo.starkStruct.nBits;
    uint64_t treeBytes = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
    uint64_t scratchBytes = setupCtx->starkInfo.nConstants * N * sizeof(Goldilocks::Element);
    if (treeBytes + scratchBytes > d_buffers->prefetchFixedBytes) return;
    {
        std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
        if (d_buffers->prefetchFixedAirgroup != -1) {
            if (d_buffers->prefetchFixedSize == 0) return;  // another staging in flight
            if (d_buffers->prefetchFixedAirgroup == (int64_t)airgroupId &&
                d_buffers->prefetchFixedAir == (int64_t)airId &&
                d_buffers->prefetchFixedSize == treeBytes) return;  // already staged for us
            // Complete but never consumed (mispredict / same-air reuse): drop it.
            CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchFixedReady));
            d_buffers->prefetchFixedAirgroup = -1;
            d_buffers->prefetchFixedAir = -1;
            d_buffers->prefetchFixedSize = 0;
        }
        d_buffers->prefetchFixedAirgroup = (int64_t)airgroupId;
        d_buffers->prefetchFixedAir = (int64_t)airId;
        d_buffers->prefetchFixedSize = 0;  // in flight
    }
    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    uint32_t gpuLocalId = d_buffers->gpus_g2l[d_buffers->my_gpu_ids[0]];
    gl64_t *d_const_pols = d_buffers->d_constPolsAggregation[gpuLocalId] + air->const_pols_offset;
    cudaStream_t bs = d_buffers->treeBuildStream;
    // Never overwrite an undrained tree (consume records Drained on the proof stream).
    CHECKCUDAERR(cudaStreamWaitEvent(bs, d_buffers->prefetchFixedDrained, 0));
    Goldilocks::Element *segTree = (Goldilocks::Element *)d_buffers->prefetchFixed;
    Goldilocks::Element *segScratch = (Goldilocks::Element *)((uint8_t *)d_buffers->prefetchFixed + treeBytes);
    Goldilocks::Element *packed = (Goldilocks::Element *)d_const_pols;
    unpack_fixed((uint64_t *)d_const_pols, (uint64_t *)(packed + 1),
                 (uint64_t *)(packed + 1 + setupCtx->starkInfo.nConstants),
                 (uint64_t *)segScratch, setupCtx->starkInfo.nConstants, N, bs, d_buffers->treeBuildTimer);
    extendAndMerkelizeFixed(*setupCtx, segScratch, segTree, true, d_buffers->treeBuildTimer, bs);
    CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchFixedReady, bs));
    {
        std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
        if (d_buffers->prefetchFixedAbandon) {
            d_buffers->prefetchFixedAirgroup = -1;
            d_buffers->prefetchFixedAir = -1;
            d_buffers->prefetchFixedSize = 0;
            d_buffers->prefetchFixedAbandon = false;
        } else {
            d_buffers->prefetchFixedSize = treeBytes;
        }
    }
}

// Dispatch-time tree staging for RECURSIVE proofs: called next to
// prefetch_recursive_witness, while the previous proof still owns the GPU.
// Skips circuits whose tree is preallocated in the aggregation const buffer
// (stored_tree) -- those never reload. Delegates to prefetch_fixed_gpu.
int64_t prefetch_recursive_tree_gpu(void *d_buffers_, void *pSetupCtx_, uint64_t airgroupId,
                                    uint64_t airId, const char *proofType, const char *constTreePath) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || d_buffers->prefetchFixed == nullptr) return -1;
    auto key = std::make_pair(airgroupId, airId);
    auto itK = d_buffers->air_instances.find(key);
    if (itK == d_buffers->air_instances.end()) return -2;
    auto itT = itK->second.find(std::string(proofType));
    if (itT == itK->second.end() || itT->second.empty()) return -2;
    AirInstanceInfo *air = itT->second[0];
    if (air == nullptr || air->stored_tree) return 1;  // resident: nothing to stage
    (void)constTreePath;  // compute prestage rebuilds from resident pols; no file needed
    prestageRecTreeCompute(d_buffers, (SetupCtx *)pSetupCtx_, airgroupId, airId, proofType);
    return 0;
}

int64_t prefetch_fixed_gpu(void *d_buffers_, uint64_t airgroupId, uint64_t airId,
                           const char *constTreePath, uint64_t bytes) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || d_buffers->prefetchFixed == nullptr) return -1;
    {
        std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
        if (d_buffers->prefetchFixedAirgroup != -1) {
            if (d_buffers->prefetchFixedSize == 0) return -3;   // staging in flight elsewhere
            // Complete but never consumed: drop it (single-threaded producer).
            zklog.warning("prefetch_fixed: dropping stale entry for air " +
                          std::to_string(d_buffers->prefetchFixedAirgroup) + ":" +
                          std::to_string(d_buffers->prefetchFixedAir));
            CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchFixedReady));
            d_buffers->prefetchFixedAirgroup = -1;
            d_buffers->prefetchFixedAir = -1;
            d_buffers->prefetchFixedSize = 0;
        }
        if (bytes > d_buffers->prefetchFixedBytes) return -4;
        // Claim before the slow IO so a second caller backs off immediately.
        d_buffers->prefetchFixedAirgroup = (int64_t)airgroupId;
        d_buffers->prefetchFixedAir = (int64_t)airId;
        d_buffers->prefetchFixedSize = 0;   // 0 = staging in progress
    }
    FILE *f = fopen(constTreePath, "rb");
    if (f == nullptr) {
        std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
        d_buffers->prefetchFixedAirgroup = -1;
        d_buffers->prefetchFixedAir = -1;
        return -5;
    }
    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    // Never overwrite an undrained fixed half; no-op if never recorded.
    CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->prefetchFixedDrained, 0));
    if (const uint8_t *cached = hostCacheGet(constTreePath, bytes)) {
        fclose(f);
        CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_buffers->prefetchFixed, cached, bytes,
                                     cudaMemcpyHostToDevice, d_buffers->prefetchStream));
        CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchFixedReady, d_buffers->prefetchStream));
        {
            std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
            if (d_buffers->prefetchFixedAbandon) {
                d_buffers->prefetchFixedAirgroup = -1;
                d_buffers->prefetchFixedAir = -1;
                d_buffers->prefetchFixedSize = 0;
                d_buffers->prefetchFixedAbandon = false;
            } else {
                d_buffers->prefetchFixedSize = bytes;
            }
        }
        return 0;
    }
    // The pinned pair is shared with the packed-const staging on other threads.
    std::lock_guard<std::mutex> pplk(d_buffers->pinnedPairMutex);
    const uint64_t chunkBytes = 128ull << 20;
    uint64_t off = 0;
    int buf = 0;
    while (off < bytes) {
        uint64_t len = std::min(chunkBytes, bytes - off);
        // Two pinned chunks used alternately: this fread overlaps the previous
        // chunk's in-flight H2D; only every second turn waits, and only for the
        // transfer issued two chunks ago.
        CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchPinnedFree[buf]));
        if (fread(d_buffers->prefetchPinned[buf], 1, len, f) != len) {
            fclose(f);
            std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
            d_buffers->prefetchFixedAirgroup = -1;
            d_buffers->prefetchFixedAir = -1;
            return -6;
        }
        CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_buffers->prefetchFixed + off,
                                     d_buffers->prefetchPinned[buf], len,
                                     cudaMemcpyHostToDevice, d_buffers->prefetchStream));
        CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchPinnedFree[buf], d_buffers->prefetchStream));
        off += len;
        buf ^= 1;
    }
    fclose(f);
    CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchFixedReady, d_buffers->prefetchStream));
    {
        std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
        if (d_buffers->prefetchFixedAbandon) {
            // The air's first proof already passed (legacy load): drop, freeing
            // the zone for the next air instead of wedging it.
            d_buffers->prefetchFixedAirgroup = -1;
            d_buffers->prefetchFixedAir = -1;
            d_buffers->prefetchFixedSize = 0;
            d_buffers->prefetchFixedAbandon = false;
        } else {
            d_buffers->prefetchFixedSize = bytes;   // staged & complete
        }
    }
    return 0;
}

// Upload `instanceId`'s trace into the prefetch zone on the copy stream, while
// the current proof runs. Chunked so no single transfer monopolizes PCIe. The
// caller must keep `trace` alive until the matching gen_proof consumes the
// zone (which host-syncs prefetchReady before recycling). Returns 0 on
// success; negative = zone busy/absent/too small (caller falls back to the
// legacy upload silently).
int64_t prefetch_witness_gpu(void *pSetupCtx_, void *d_buffers_, uint64_t instanceId,
                             uint64_t airgroupId, uint64_t airId, void *trace) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    zklog.info("prefetch_witness called: instance " + std::to_string(instanceId) + " air " + std::to_string(airgroupId) + ":" + std::to_string(airId));
    if (d_buffers == nullptr || d_buffers->prefetchZone == nullptr || trace == nullptr) return -1;
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

    std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
    uint32_t slot = d_buffers->prefetchStageSlot;
    if (d_buffers->prefetchInstanceId[slot] != -1) {
        // Stale unconsumed entry (single-threaded producer: safe to drop after
        // waiting out its in-flight upload).
        zklog.warning("prefetch_witness: dropping stale entry for instance " +
                      std::to_string(d_buffers->prefetchInstanceId[slot]));
        CHECKCUDAERR(cudaEventSynchronize(d_buffers->prefetchReady[slot]));
        d_buffers->prefetchInstanceId[slot] = -1;
        d_buffers->prefetchTraceBytes[slot] = 0;
    }
    if (total_size > d_buffers->prefetchZoneBytes) return -4;

    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    // Never overwrite a slot the proof stream has not drained yet. Waiting on a
    // never-recorded event is a no-op, so the first prefetch passes through.
    CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->prefetchDrained[slot], 0));
    uint8_t *slotBase = (uint8_t *)(d_buffers->prefetchZone + (uint64_t)slot * d_buffers->prefetchSlotStride);
    const uint64_t blockBytes = 32ull << 20;
    for (uint64_t off = 0; off < total_size; off += blockBytes) {
        uint64_t len = std::min(blockBytes, total_size - off);
        CHECKCUDAERR(cudaMemcpyAsync(slotBase + off,
                                     (const uint8_t *)trace + off, len,
                                     cudaMemcpyHostToDevice, d_buffers->prefetchStream));
    }
    CHECKCUDAERR(cudaEventRecord(d_buffers->prefetchReady[slot], d_buffers->prefetchStream));
    d_buffers->prefetchInstanceId[slot] = (int64_t)instanceId;
    d_buffers->prefetchTraceBytes[slot] = total_size;
    d_buffers->prefetchStageSlot = (slot + 1) % d_buffers->prefetchNSlots;
    return 0;
}

// Stage a RECURSIVE proof's witness on the copy stream while the current proof
// computes. Keyed by the host trace pointer (unique per in-flight buffer); the
// slot is lazily (re)sized to the request. gen_recursive_proof_gpu consumes it
// (waits recWitReady, reads the slot, records recWitDrained). Returns 0 staged,
// negative = zone absent (caller silently keeps the legacy inline upload).
int64_t prefetch_recursive_witness_gpu(void *d_buffers_, const void *trace, uint64_t bytes) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || d_buffers->prefetchZone == nullptr || trace == nullptr || bytes == 0) return -1;
    // Debug bisect: PROOFMAN_NO_RECWIT_STAGE=1 forces the consume-side upload path.
    {
        static const bool noStage = [] {
            const char *e = getenv("PROOFMAN_NO_RECWIT_STAGE");
            return e != nullptr && e[0] == '1';
        }();
        if (noStage) return -1;
    }
    cudaSetDevice(d_buffers->my_gpu_ids[0]);
    std::lock_guard<std::mutex> lk(d_buffers->recWitMutex);
    uint32_t slot = d_buffers->recWitCursor;
    if (d_buffers->recWitKey[slot] != nullptr) {
        // Stale unconsumed entry (mispredicted or dropped launch): wait out its
        // upload and reuse the slot.
        CHECKCUDAERR(cudaEventSynchronize(d_buffers->recWitReady[slot]));
        d_buffers->recWitKey[slot] = nullptr;
    }
    if (d_buffers->recWitSlotBytes[slot] < bytes) {
        if (d_buffers->recWitCarved) return -4;  // carved slots are fixed-size; caller falls back
        // Lazy (re)size: never while its consumer may still read it.
        CHECKCUDAERR(cudaEventSynchronize(d_buffers->recWitDrained[slot]));
        if (d_buffers->recWitSlot[slot] != nullptr) CHECKCUDAERR(cudaFree(d_buffers->recWitSlot[slot]));
        CHECKCUDAERR(cudaMalloc((void **)&d_buffers->recWitSlot[slot], bytes));
        d_buffers->recWitSlotBytes[slot] = bytes;
    }
    CHECKCUDAERR(cudaStreamWaitEvent(d_buffers->prefetchStream, d_buffers->recWitDrained[slot], 0));
    const uint64_t blockBytes = 32ull << 20;
    for (uint64_t off = 0; off < bytes; off += blockBytes) {
        uint64_t len = std::min(blockBytes, bytes - off);
        CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_buffers->recWitSlot[slot] + off,
                                     (const uint8_t *)trace + off, len,
                                     cudaMemcpyHostToDevice, d_buffers->prefetchStream));
    }
    CHECKCUDAERR(cudaEventRecord(d_buffers->recWitReady[slot], d_buffers->prefetchStream));
    d_buffers->recWitKey[slot] = trace;
    d_buffers->recWitBytes[slot] = bytes;
    d_buffers->recWitCursor = (slot + 1) % 2;
    return 0;
}

void configure_stream_commit_slots_gpu(void *d_buffers_, uint64_t nSlots, uint64_t slotBytes) {
    if (d_buffers_ == nullptr || nSlots == 0 || slotBytes == 0) return;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers->streamCommitSlots != 0) return;  // already configured

    // Round up to 1 MiB: keeps slot bases 16-byte aligned for vectorized
    // kernel accesses and the carve log readable.
    slotBytes = (slotBytes + ((1ull << 20) - 1)) & ~((1ull << 20) - 1);

    uint64_t totalAuxTraceSize = d_buffers->auxTraceTotalBytes;
    // Phase-B aliased recursive streams occupy no space of their own.
    const uint64_t recAreaBytes =
        d_buffers->phaseBAliased ? 0 : d_buffers->n_recursive_streams * d_buffers->auxTraceRecursiveBytes;
    uint64_t constAggOffsetBytes =
        totalAuxTraceSize + recAreaBytes + d_buffers->prefetchRegionBytes + d_buffers->mopsFloorPadBytes;
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
            // Aliased pair: [0..A) and [A..2A) over the pre-const area.
            start = d_buffers->phaseBAliased
                        ? sd.localStreamId * d_buffers->auxTraceRecursiveBytes
                        : totalAuxTraceSize + sd.localStreamId * d_buffers->auxTraceRecursiveBytes;
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
    // Glued prefetch zone: the borrower scribbled over it too -- drop every staging key
    // (device is synchronized above, so no upload is in flight).
    if (d_buffers->prefetchZone != nullptr && d_buffers->prefetchZone == d_buffers->prefetchRegionBase) {
        { std::lock_guard<std::mutex> lk(d_buffers->prefetchMutex);
          for (int sIdx = 0; sIdx < 2; sIdx++) { d_buffers->prefetchInstanceId[sIdx] = -1; d_buffers->prefetchTraceBytes[sIdx] = 0; } }
        { std::lock_guard<std::mutex> lk(d_buffers->prefetchFixedMutex);
          d_buffers->prefetchFixedAirgroup = -1; d_buffers->prefetchFixedAir = -1; d_buffers->prefetchFixedSize = 0; }
        { std::lock_guard<std::mutex> lk(d_buffers->packedMutex); d_buffers->packedSlotKey = -1; }
        { std::lock_guard<std::mutex> lk(d_buffers->recWitMutex);
          for (int sIdx = 0; sIdx < 2; sIdx++) { d_buffers->recWitKey[sIdx] = nullptr; d_buffers->recWitBytes[sIdx] = 0; } }
    }
    d_buffers->firstGpuBufferBorrowed.store(0, std::memory_order_release);
    // Second commit lane (PROOFMAN_SLOT_LANE!=0): after the borrow releases, the
    // streaming slots keep serving contribution commits concurrently with the
    // legacy stream instead of staying quiesced until the next borrow cycle.
    {
        const char *e = getenv("PROOFMAN_SLOT_LANE");
        if (e == nullptr || e[0] != '0') {
            d_buffers->streamCommitQuiesced.store(0, std::memory_order_release);
        }
    }
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
    // Phase-B aliased streams only count as a pool while the pair is OPEN: otherwise a
    // force-recursive worker would refuse the non-recursive fallback for the whole run
    // and dispatch single-worker (measured +2.4 s on 712 tx with the pair registered
    // but gated off).
    if (d_buffers->phaseBAliased && d_buffers->phaseBState.load(std::memory_order_acquire) != 1) return false;
    return d_buffers->n_recursive_streams > 0;
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
        // One pool pass; the best candidate per GPU is left locked.
        auto scanPool = [&](auto pool) {
            const uint32_t pbState = d_buffers->phaseBAliased
                ? d_buffers->phaseBState.load(std::memory_order_acquire) : 0;
            for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
                StreamData &sd = d_buffers->streamsData[i];
                if (firstGpuBorrowed && sd.gpuId == firstGpuId) continue;
                // Phase-B aliasing: the rec pair only exists in state 1; the basic
                // stream's buffer IS the rec pair's memory, so it is off in state 1.
                if (d_buffers->phaseBAliased) {
                    if (sd.recursive && pbState != 1) continue;
                    if (!sd.recursive && pbState == 1) continue;
                    // Debug bisect: PROOFMAN_PHASE_B_ONE_STREAM=1 keeps B2 closed, so the
                    // deferral machinery runs but every rec proof serializes on B1.
                    static const bool oneStream = [] {
                        const char *e = getenv("PROOFMAN_PHASE_B_ONE_STREAM");
                        return e != nullptr && e[0] == '1';
                    }();
                    if (oneStream && sd.recursive && sd.localStreamId == 1) continue;
                }
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
    if (d_buffers->phaseBAliased) {
        const uint32_t pbState = d_buffers->phaseBState.load(std::memory_order_acquire);
        if (sd.recursive && pbState != 1) return 0;
        if (!sd.recursive && pbState == 1) return 0;
    }
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
// Deep pipeline: a busy basic stream is reservable while it holds fewer than 2
// in-flight proofs. Caller must hold the stream's selection mutex (like every
// status check-then-act); pipeCount has its own lock.
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

void harvest_pipeline_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr || !d_buffers->pipelineMode) return;
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        harvestPipelineStream(d_buffers, i, false);
    }
}

// Toggle deep pipelining (proofs phase of the single-stream zone mode only: the
// contributions phase relies on harvest-on-reserve for commit roots, so it must
// stay off there). Also gates the per-stream GPU timers, whose per-proof timings
// interleave meaninglessly with 2 proofs in flight.
void set_pipeline_mode_gpu(void *d_buffers_, bool enable) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    if (d_buffers == nullptr) return;
    d_buffers->pipelineMode = enable;
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        d_buffers->streamsData[i].timer.enabled = !enable;
    }
}

void reserveStreamLocked(DeviceCommitBuffers* d_buffers, uint32_t streamId){
    StreamData &sd = d_buffers->streamsData[streamId];
    cudaSetDevice(sd.gpuId);
    if(sd.status==2) {
        if (d_buffers->pipelineMode && !sd.recursive) {
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