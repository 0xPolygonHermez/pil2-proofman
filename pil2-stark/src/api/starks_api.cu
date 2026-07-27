#include "bn128.cuh"
#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#include "starks_api.hpp"
#include "starks_api_internal.cuh"
#include "starks_api_internal.hpp"
#include <cstring>
#include <thread>
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
#include "hints.cuh"
#include "gen_recursivef_proof.cuh"
#include "poseidon_bn128.cuh"
#include "proofman_sumcheck.cuh"
#include <cuda_runtime.h>
#include <mutex>
#include <algorithm>
#include <map>


uint32_t selectStream(DeviceCommitBuffers* d_buffers, uint64_t airgroupId, uint64_t airId, std::string proofType, bool recursive = false, bool force_recursive = false);
void reserveStream(DeviceCommitBuffers* d_buffers, uint32_t streamId);
void reserveStreamLocked(DeviceCommitBuffers* d_buffers, uint32_t streamId);
void closeStreamTimer(TimerGPU &timer, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, bool isProve);
void get_proof(DeviceCommitBuffers *d_buffers, uint64_t streamId);
void get_commit_root(DeviceCommitBuffers *d_buffers, uint64_t streamId);


void buildMerkleTreeGPU(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                         uint64_t nCols, uint64_t nRows, Layout layout, cudaStream_t stream)
{
    if (get_hash_family() == HashFamily::Poseidon1) {
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
    if (get_hash_family() == HashFamily::Poseidon1) {
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

// Block until the async commit on `streamId` (trace H2D + tiling + LDE + Merkle)
// has finished on the GPU. The trace H2D is no longer synced at copy time (see
// copy_direct_registered_h2d_if_enabled), so the host trace buffer must not be
// reused until end_event fires, or the in-flight DMA reads recycled bytes. Called
// from the buffer pool before reusing a shared trace buffer. No-op if the stream
// has no outstanding commit (status != 2).
void wait_stream_commit_done_gpu(void *d_buffers_, uint64_t streamId) {
    if (d_buffers_ == nullptr) return;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    // Guard the C-ABI surface: an out-of-range streamId (e.g. from a future caller
    // or an error path) would index streamsData OOB and segfault before any CUDA
    // error handling could run.
    if (streamId >= d_buffers->n_total_streams) return;
    cudaSetDevice(d_buffers->streamsData[streamId].gpuId);
    if (d_buffers->streamsData[streamId].status == 2) {
        CHECKCUDAERR(cudaEventSynchronize(d_buffers->streamsData[streamId].end_event));
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

    // Force CUDA primary context creation only on this rank's assigned GPUs.
    // Why: never touch the default device (GPU 0) implicitly — non-owning ranks
    // would each create a ~300 MB primary context there and starve the rank that
    // actually owns GPU 0. cudaDeviceSynchronize/cudaSetDevice back to GPU 0
    // would do exactly that, so we end on an assigned GPU instead.
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

    //Generate static twiddles for the NTT
    NTTGoldilocksGPU::initConstants(max_n_bits_ext, n_gpus, my_gpu_ids);

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

void alloc_device_large_buffers_gpu(void *d_buffers_, uint64_t auxTraceArea, uint64_t auxTraceRecursiveArea, uint64_t totalConstPols, uint64_t totalConstPolsAggregation) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint64_t constPolsSize = totalConstPols * sizeof(Goldilocks::Element);
    uint64_t constPolsAggregationSize = totalConstPolsAggregation * sizeof(Goldilocks::Element);
    uint64_t auxTraceSize = auxTraceArea * sizeof(Goldilocks::Element);
    uint64_t auxTraceRecursiveSize = auxTraceRecursiveArea * sizeof(Goldilocks::Element);
    
    uint64_t totalAuxTraceSize = d_buffers->n_streams * auxTraceSize;
    uint64_t totalAuxTraceRecursiveSize = d_buffers->n_recursive_streams * auxTraceRecursiveSize;
    
    uint64_t totalGpuMemoryPerGpu = constPolsAggregationSize + 
                                     totalAuxTraceSize + totalAuxTraceRecursiveSize;
    
    uint64_t totalPinnedMemoryPerGpu = 2 * d_buffers->pinned_size * sizeof(Goldilocks::Element);

    zklog.info("Memory allocation per GPU:");
    zklog.info("  - Constant polynomials (separate): " + std::to_string(constPolsSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Constant polynomials aggregation: " + std::to_string(constPolsAggregationSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Auxiliary trace (" + std::to_string(d_buffers->n_streams) + " streams): " + std::to_string(totalAuxTraceSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Auxiliary trace recursive (" + std::to_string(d_buffers->n_recursive_streams) + " streams): " + std::to_string(totalAuxTraceRecursiveSize / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Unified buffer per GPU: " + std::to_string(totalGpuMemoryPerGpu / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Total GPU memory per GPU: " + std::to_string((totalGpuMemoryPerGpu + constPolsSize) / (1024.0 * 1024.0 * 1024.0)) + " GB");
    zklog.info("  - Pinned host memory per GPU: " + std::to_string(totalPinnedMemoryPerGpu / (1024.0 * 1024.0 * 1024.0)) + " GB");

    d_buffers->constPolsSize = constPolsSize;
    d_buffers->unifiedBufferSize = totalGpuMemoryPerGpu;
    d_buffers->firstGpuBufferBorrowed.store(0, std::memory_order_relaxed);

    // Allocate large GPU buffers with a single malloc per GPU
    for (int i = 0; i < d_buffers->n_gpus; i++) {
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        
        // Check available GPU memory
        size_t freeMem, totalMem;
        CHECKCUDAERR(cudaMemGetInfo(&freeMem, &totalMem));
        zklog.info("GPU " + std::to_string(d_buffers->my_gpu_ids[i]) + ": Available memory: " + 
                   std::to_string(freeMem / (1024.0 * 1024.0 * 1024.0)) + " GB / " + 
                   std::to_string(totalMem / (1024.0 * 1024.0 * 1024.0)) + " GB");
        
        if (freeMem < totalGpuMemoryPerGpu + constPolsSize) {
            zklog.error("GPU " + std::to_string(d_buffers->my_gpu_ids[i]) + 
                       ": Insufficient memory. Need " + std::to_string((totalGpuMemoryPerGpu + constPolsSize) / (1024.0 * 1024.0 * 1024.0)) + 
                       " GB but only " + std::to_string(freeMem / (1024.0 * 1024.0 * 1024.0)) + " GB available");
            exit(1);
        }
        
        // Allocate one large contiguous block of GPU memory (unified buffer)
        gl64_t *gpuMemoryBlock;
        CHECKCUDAERR(cudaMalloc(&gpuMemoryBlock, totalGpuMemoryPerGpu));
        d_buffers->gpuMemoryBuffer[i] = gpuMemoryBlock;  // Store the base pointer
        
        // Allocate separate buffer for constant polynomials
        CHECKCUDAERR(cudaMalloc(&d_buffers->d_constPols[i], constPolsSize));
        
        zklog.info("GPU " + std::to_string(d_buffers->my_gpu_ids[i]) + 
                   ": Allocated " + std::to_string((totalGpuMemoryPerGpu + constPolsSize) / (1024.0 * 1024.0 * 1024.0)) + 
                   " GB (" + std::to_string(totalGpuMemoryPerGpu / (1024.0 * 1024.0 * 1024.0)) + 
                   " GB unified + " + std::to_string(constPolsSize / (1024.0 * 1024.0 * 1024.0)) + " GB const pols)");
        
        // Set up pointers to different sections of the memory block
        uint64_t offset = 0;
                
        // Auxiliary trace buffers (non-recursive)
        for (int j = 0; j < d_buffers->n_streams; ++j) {
            d_buffers->d_aux_trace[i][j] = gpuMemoryBlock + offset;
            offset += auxTraceArea;
        }
        
        // Auxiliary trace buffers (recursive)
        for (int j = 0; j < d_buffers->n_recursive_streams; ++j) {
            d_buffers->d_aux_traceAggregation[i][j] = gpuMemoryBlock + offset;
            offset += auxTraceRecursiveArea;
        }

        // Constant polynomials aggregation
        d_buffers->d_constPolsAggregation[i] = gpuMemoryBlock + offset;
        offset += totalConstPolsAggregation;
        
        // Allocate pinned host buffers separately (one block per buffer type)
        CHECKCUDAERR(cudaMallocHost(&d_buffers->pinned_buffer[i], d_buffers->pinned_size * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMallocHost(&d_buffers->pinned_buffer_extra[i], d_buffers->pinned_size * sizeof(Goldilocks::Element)));
        
        // Verify we used exactly the amount we calculated
        if (offset != totalGpuMemoryPerGpu / sizeof(Goldilocks::Element)) {
            zklog.error("GPU " + std::to_string(d_buffers->my_gpu_ids[i]) + 
                       ": Memory offset mismatch! Expected " + std::to_string(totalGpuMemoryPerGpu / sizeof(Goldilocks::Element)) + 
                       " but got " + std::to_string(offset) + " elements");
            exit(1);
        }
    }
    
    zklog.info("All GPU memory allocations successful");
}

uint64_t gen_device_streams_gpu(void *d_buffers_, uint64_t n_streams, uint64_t n_recursive_streams, uint64_t maxSizeProverBuffer, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t merkleTreeArity) {
    
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    d_buffers->n_streams = n_streams;
    d_buffers->n_recursive_streams = n_recursive_streams;
    d_buffers->n_total_streams = d_buffers->n_gpus * (d_buffers->n_streams + d_buffers->n_recursive_streams);
    
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
            d_buffers->streamsData[gpu_stream_start + j].initialize(maxProofSize, d_buffers->my_gpu_ids[i], j, false, merkleTreeArity);
        }

        for (uint64_t j = 0; j < d_buffers->n_recursive_streams; j++) {
            d_buffers->streamsData[gpu_stream_start + d_buffers->n_streams + j].initialize(maxProofSize, d_buffers->my_gpu_ids[i], j, true, merkleTreeArity);
        }
    }

    return d_buffers->n_gpus;
}

void reset_device_streams_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    for(uint64_t i=0; i< d_buffers->n_total_streams; ++i){
        cudaSetDevice(d_buffers->streamsData[i].gpuId);
        CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[i].stream));
        d_buffers->streamsData[i].invalidateContext();
        d_buffers->streamsData[i].instanceId = -1;   // full teardown: no resident witness
        d_buffers->streamsData[i].reset(true);
    }
}

void free_device_buffers_gpu(void *d_buffers_)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    if (d_buffers->streamsData != nullptr) {
        for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
            d_buffers->streamsData[i].free();
        }
        delete[] d_buffers->streamsData;
        d_buffers->streamsData = nullptr;
    }

    for (int i = 0; i < d_buffers->n_gpus; ++i) {
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        
        // Free the single large GPU memory block
        // All other GPU pointers (d_constPols, d_constPolsAggregation, d_aux_trace, d_aux_traceAggregation) 
        // point into this same block, so we only free it once using the stored base pointer
        if (d_buffers->gpuMemoryBuffer != nullptr && d_buffers->gpuMemoryBuffer[i] != nullptr) {
            CHECKCUDAERR(cudaFree(d_buffers->gpuMemoryBuffer[i]));
        }
        
        if (d_buffers->d_constPols != nullptr && d_buffers->d_constPols[i] != nullptr) {
            CHECKCUDAERR(cudaFree(d_buffers->d_constPols[i]));
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
    }
    free(d_buffers->d_aux_trace);
    free(d_buffers->d_aux_traceAggregation);
    free(d_buffers->d_constPols);
    free(d_buffers->d_constPolsAggregation);
    free(d_buffers->pinned_buffer);
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


void load_device_setup_gpu(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packed_info) {
    
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    Goldilocks::Element *verkeyRoot = (Goldilocks::Element *)verkeyRoot_;

    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

    PackedInfo *packedInfo = (PackedInfo *)packed_info;

    if (d_buffers->air_instances[key][proofType].empty()) {
        d_buffers->air_instances[key][proofType].resize(d_buffers->n_gpus, nullptr);
    }

    for(int i=0; i<d_buffers->n_gpus; ++i){
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        if (d_buffers->air_instances[key][proofType][i] != nullptr) {
            delete d_buffers->air_instances[key][proofType][i];
        }
        d_buffers->air_instances[key][proofType][i] = new AirInstanceInfo(airgroupId, airId, setupCtx, verkeyRoot, packedInfo);
    }
}

void load_device_const_pols_gpu(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers_, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char *proofType, bool onlyFirstGPU) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint64_t sizeConstPols = constSize * sizeof(Goldilocks::Element);
    
    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

    uint64_t const_pols_offset = initial_offset;

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

uint64_t gen_proof_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params_, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, bool skipRecalculation, uint64_t streamId_, char *constPolsPath,  char *constTreePath, char *customCommitsFixedPath) {

    auto key = std::make_pair(airgroupId, airId);
    std::string proofType = "basic";

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
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
    } else {
        streamId = selectStream(d_buffers, airgroupId, airId, proofType, false);
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

    // Basic proofs never alias: (airgroupId,airId) is unique per AIR, so the tuple suffices.
    bool reuse_constants = d_buffers->streamsData[streamId].airgroupId == airgroupId && d_buffers->streamsData[streamId].airId == airId && d_buffers->streamsData[streamId].proofType == string("basic");

    d_buffers->streamsData[streamId].pSetupCtx = pSetupCtx_;
    d_buffers->streamsData[streamId].proofBuffer = proofBuffer;
    d_buffers->streamsData[streamId].proofFile = string(proofFile);
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].proofType = "basic";
    d_buffers->streamsData[streamId].witnessResident = false;

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetStage1Extended = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];

    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0 && !reuse_constants) {
        Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
        uint64_t customCommitsSize = setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element);
        // Skip the 32-byte Merkle-root header at the start of the file (assumes 1 custom commit per AIR).
        load_and_copy_to_device_in_chunks(d_buffers, customCommitsFixedPath, (uint8_t*)pCustomCommitsFixed, customCommitsSize, streamId, 32);
    }

    if (!skipRecalculation) {
        uint64_t total_size = (d_buffers->packedTrace && air_instance_info->is_packed) ? air_instance_info->num_packed_words * N * sizeof(Goldilocks::Element) : N * nCols * sizeof(Goldilocks::Element);
        uint64_t *dst = (uint64_t *)(d_aux_trace + offsetStage1Extended);
        copy_to_device_in_chunks(d_buffers, params->trace, dst, total_size, streamId, timer);
    }
    
    size_t totalCopySize = 0;
    totalCopySize += setupCtx->starkInfo.nPublics;
    totalCopySize += setupCtx->starkInfo.proofValuesSize;
    totalCopySize += setupCtx->starkInfo.airgroupValuesSize;
    totalCopySize += setupCtx->starkInfo.airValuesSize;
    totalCopySize += FIELD_EXTENSION;

    Goldilocks::Element aux_values[totalCopySize];
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

    copy_to_device_in_chunks(d_buffers, aux_values, (uint8_t*)(d_aux_trace + offsetPublicInputs), totalCopySize * sizeof(Goldilocks::Element), streamId, timer);

    gl64_t *d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
    gl64_t *d_const_tree;
    if (air_instance_info->stored_tree) {
        d_const_tree = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_tree_offset;
    } else {
        uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
        d_const_tree = d_aux_trace + offsetConstTree;

        if (!reuse_constants && !setupCtx->starkInfo.calculateFixedExtended) {
            load_and_copy_to_device_in_chunks(d_buffers, constTreePath, (uint8_t*)d_const_tree, sizeConstTree, streamId);
        }
    }


    proofman_sumcheck_set_context(instanceId, airgroupId, airId);
    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, constTreePath, streamId, instanceId, d_buffers, air_instance_info, skipRecalculation, timer, stream, false, reuse_constants);
    cudaEventRecord(d_buffers->streamsData[streamId].end_event, stream);
    d_buffers->streamsData[streamId].status = 2;
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
   
    // Basic proofs never alias: (airgroupId,airId) is unique per AIR (see gen_proof_gpu).
    bool reuse_constants = d_buffers->streamsData[streamId].airgroupId == airgroupId && d_buffers->streamsData[streamId].airId == airId && d_buffers->streamsData[streamId].proofType == string("basic");

    d_buffers->streamsData[streamId].pSetupCtx = pSetupCtx_;
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].proofType = "basic";
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].witnessResident = false;

    proofman_sumcheck_set_context(instanceId, airgroupId, airId);

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];

    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0 && !reuse_constants) {
        Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
        uint64_t customCommitsSize = setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element);
        load_and_copy_to_device_in_chunks(d_buffers, customCommitsFixedPath, (uint8_t*)pCustomCommitsFixed, customCommitsSize, streamId, 32);
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

    Goldilocks::Element aux_values[totalCopySize];
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

    copy_to_device_in_chunks(d_buffers, aux_values, (uint8_t*)(d_aux_trace + offsetPublicInputs), totalCopySize * sizeof(Goldilocks::Element), streamId, timer);
    
    gl64_t *d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
    
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;
    unpack_fixed((uint64_t*)d_const_pols, (uint64_t*)(d_const_pols + 1), (uint64_t*)(d_const_pols + 1 + setupCtx->starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants, N, stream, timer);
    CHECKCUDAERR(cudaGetLastError());

    uint64_t offsetCm1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    if (d_buffers->packedTrace && air_instance_info->is_packed) {
        unpack_trace(air_instance_info, (uint64_t*)(d_aux_trace + offsetCm1 + N * nCols), (uint64_t*)(d_aux_trace + offsetCm1), nCols, N, stream, timer);
    } else {
        fromRowMajorToColMajor(N, nCols, (gl64_t *)(d_aux_trace + offsetCm1 + N * nCols), (gl64_t*)(d_aux_trace + offsetCm1), resolveLayout(setupCtx->starkInfo.starkStruct.nBits, nCols), stream);
    }
    PROOFMAN_SUMCHECK("proof_after_unpack", d_aux_trace + offsetCm1, N * nCols, stream);

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
        collectStreamResult(d_buffers, i);
        d_buffers->streamsData[i].mutex_stream_selection.unlock();
    }
}

void get_stream_proofs_non_blocking_gpu(void *d_buffers_){
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
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
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    cudaSetDevice(d_buffers->streamsData[streamId].gpuId);
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
    collectStreamResult(d_buffers, streamId);
}

uint64_t gen_recursive_proof_gpu(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *trace, void *aux_trace, void *pConstPols, void *pConstTree, void *pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers_, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream, char *recurser_id)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    bool aggregation = false;
    if(string(proofType) == "recursive1" || string(proofType) == "recursive2") {
        aggregation = true;
    }
    uint32_t streamId = selectStream(d_buffers, airgroupId, airId, proofType, aggregation, force_recursive_stream);
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

    // Recurser setups all share (0,0,"recursive2"), so recurser_id disambiguates them
    // (empty for normal recursion, where the tuple is already unique).
    bool reuse_constants = d_buffers->streamsData[streamId].recurserId == string(recurser_id) && d_buffers->streamsData[streamId].airgroupId == airgroupId && d_buffers->streamsData[streamId].airId == airId && d_buffers->streamsData[streamId].proofType == string(proofType);

    d_buffers->streamsData[streamId].pSetupCtx = pSetupCtx_;
    d_buffers->streamsData[streamId].proofBuffer = proofBuffer;
    d_buffers->streamsData[streamId].proofFile = string(proof_file);
    d_buffers->streamsData[streamId].recurserId = string(recurser_id);
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].proofType = string(proofType);
    d_buffers->streamsData[streamId].witnessResident = false;

    uint64_t offsetStage1Extended = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    copy_to_device_in_chunks(d_buffers, trace, (uint8_t*)(d_aux_trace + offsetStage1Extended), sizeTrace, streamId, timer);
    
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    copy_to_device_in_chunks(d_buffers, pPublicInputs, (uint8_t*)(d_aux_trace + offsetPublicInputs), setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element), streamId, timer);

    gl64_t *d_const_pols = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_pols_offset;
    gl64_t *d_const_tree;
    if (air_instance_info->stored_tree) {
        d_const_tree = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_tree_offset;
    } else {        
        uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
        d_const_tree = d_aux_trace + offsetConstTree;

        if (!reuse_constants) {
            load_and_copy_to_device_in_chunks(d_buffers, constTreePath, (uint8_t*)d_const_tree, sizeConstTree, streamId);
        }
    }

    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, constTreePath, streamId, instanceId, d_buffers, air_instance_info, false, timer, stream, true, reuse_constants);
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

    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].proofType = string(proofType);
    d_buffers->streamsData[streamId].witnessResident = false;

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
    extendAndMerkelizeFixed(*setupCtx, d_const_pols_unpacked, (Goldilocks::Element *)d_const_tree, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    d_buffers->streamsData[streamId].status = 3;
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

    // Scope sppark's GPU registry to this rank's devices before ngpus() below
    // builds it. Without this, a standalone SNARK-wrap process (no prior
    // gen_device_buffers_gpu) would probe every GPU on the node.
    {
        int ords[32];
        uint32_t n = (d_commit_buffer != nullptr) ? d_commit_buffer->n_gpus : 1;
        if (n > 32) n = 32;
        for (uint32_t i = 0; i < n; i++)
            ords[i] = (int)((d_commit_buffer != nullptr) ? d_commit_buffer->my_gpu_ids[i] : gpuId);
        sppark_set_visible_devices(ords, (int)n);
    }

    // Force sppark's lazy GPU registry to initialize now, while we still control
    // the current CUDA device. The first call into any sppark entry point
    // (select_gpu, gpu_props, ngpus, all_gpus) constructs a function-local static
    // gpus_t that probes every device and ends with cudaSetDevice(0) — silently
    // clobbering whatever device the caller had selected. By triggering that
    // one-time init here and restoring the device around it, later cudaSetDevice(N)
    // calls stick and select_gpu(-1) resolves to the device we actually want.
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
        NTTGoldilocksGPU::initConstants(22, 1, &gpuId); //max nBitsExt=21
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

void alloc_fixed_pols_buffer_gpu_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    uint32_t gpuId = d_buffers->my_gpu_ids[0];
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    cudaSetDevice(gpuId);
    
    if (d_buffers->d_constPols != nullptr && d_buffers->d_constPols[gpuLocalId] != nullptr) {
        return;
    }
    
    CHECKCUDAERR(cudaMalloc(&d_buffers->d_constPols[gpuLocalId], d_buffers->constPolsSize));
}

void free_fixed_pols_buffer_gpu_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    uint32_t gpuId = d_buffers->my_gpu_ids[0];
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    cudaSetDevice(gpuId);
    CHECKCUDAERR(cudaFree(d_buffers->d_constPols[gpuLocalId]));
    d_buffers->d_constPols[gpuLocalId] = nullptr;
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
    // Copy and tile const pols: pConstPols is row-major (.const file) but the
    // expression kernels read the tiled layout at offsetConstPols (same layout
    // unpack_fixed produces in the main flows). Stage through the witness
    // scratch area, which the tiling above has already consumed (stream-ordered).
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    copy_to_device_in_chunks((const uint8_t*)pConstPols, (uint8_t*)d_witness_temp, sizeConstPols, pinnedBuffer, pinnedBufferSize, d_buffers->stream);
    gridSize = dim3((N + blockSize.x - 1) / blockSize.x, (nConst + blockSize.y - 1) / blockSize.y, 1);
    fromRowMajorToColMajor<<<gridSize, blockSize, 0, d_buffers->stream>>>(N, nConst, (uint64_t*)d_witness_temp, (uint64_t*)(d_aux_trace + offsetConstPols), fixedLayout());
    CHECKCUDAERR(cudaGetLastError());

    void* result = genRecursiveProofBN128_gpu(*setupCtx, airgroupId, airId, instanceId, (Goldilocks::Element *)d_aux_trace, (Goldilocks::Element *)pPublicInputs, string(proof_file), d_buffers);

    cudaStreamSynchronize(d_buffers->stream);

    return result;
}

uint64_t commit_witness_gpu(void *pSetupCtx_, void *params_, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers_, char *customCommitsFixedPath) {
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint32_t streamId = selectStream(d_buffers, airgroupId, airId, "basic");
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];

    // Check reuse against the stream's prior context before it is overwritten below.
    bool reuse_custom_fixed = d_buffers->streamsData[streamId].airgroupId == airgroupId && d_buffers->streamsData[streamId].airId == airId && d_buffers->streamsData[streamId].proofType == string("basic");

    d_buffers->streamsData[streamId].root = root;
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].proofType = "witness";
    d_buffers->streamsData[streamId].witnessResident = true;

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
        unpack_fixed(d_num_packed_words, (uint64_t*)(packed_const_pols + 1), (uint64_t*)(packed_const_pols + 1 + setupCtx->starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants, N, stream, timer);
        CHECKCUDAERR(cudaGetLastError());

        if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0 && !reuse_custom_fixed) {
            Goldilocks::Element *pCustomCommitsFixedDst = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            uint64_t customCommitsSize = setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element);
            load_and_copy_to_device_in_chunks(d_buffers, customCommitsFixedPath, (uint8_t*)pCustomCommitsFixedDst, customCommitsSize, streamId, 32);
        }

        size_t totalCopySize = 0;
        totalCopySize += setupCtx->starkInfo.nPublics;
        totalCopySize += setupCtx->starkInfo.proofValuesSize;
        totalCopySize += setupCtx->starkInfo.airgroupValuesSize;
        totalCopySize += setupCtx->starkInfo.airValuesSize;

        // Stage into the per-stream pinned region for an async copy (no stream
        // sync); reused only on event-gated stream reselect. Hard runtime check
        // (not assert: must survive NDEBUG release builds, else this would silently
        // overflow the fixed pinned buffer).
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

        CHECKCUDAERR(cudaMemcpyAsync((uint8_t*)(d_aux_trace + offsetPublicInputs), aux_values, totalCopySize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));

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

        StepsParams *params_pinned = d_buffers->streamsData[streamId].pinned_params;
        memcpy(params_pinned, &h_params, sizeof(StepsParams));
        StepsParams *d_params =  d_buffers->streamsData[streamId].params;
        CHECKCUDAERR(cudaMemcpyAsync(d_params, params_pinned, sizeof(StepsParams), cudaMemcpyHostToDevice, stream));

        ExpsArguments *d_expsArgs = d_buffers->streamsData[streamId].d_expsArgs;
        DestParamsGPU *d_destParams = d_buffers->streamsData[streamId].d_destParams;
        Goldilocks::Element *pinned_exps_params = d_buffers->streamsData[streamId].pinned_buffer_exps_params;
        Goldilocks::Element *pinned_exps_args = d_buffers->streamsData[streamId].pinned_buffer_exps_args;
        
        calculateWitnessExpr_gpu(*setupCtx, h_params, d_params, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    }

    PROOFMAN_SUMCHECK("contrib_before_lde", d_aux_trace + offset_src, N * nCols, stream);
    ntt.LDE(d_aux_trace, offset_dst, d_aux_trace, offset_src, nBits, nBitsExt, nCols, timer, stream, true, (gl64_t*)pNodes);
    PROOFMAN_SUMCHECK("contrib_after_lde", d_aux_trace + offset_dst, NExtended * nCols, stream);
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    // cm1 contribution commit: read the extended trace in the layout the LDE wrote (resolveLayout on the
    // small domain) -- ColMajorTiled for tiled AIRs (e.g. Keccakf cm1), else ColMajor. Hardcoding
    // ColMajor here made the tiled contribution root read uninitialised in-tile padding -> non-det.
    buildMerkleTreeGPU(arity, (uint64_t*)pNodes, (uint64_t*)(d_aux_trace + offset_dst), nCols, 1ULL << nBitsExt, resolveLayout(nBits, nCols), stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);
    CHECKCUDAERR(cudaMemcpyAsync(d_buffers->streamsData[streamId].pinned_buffer_proof, &pNodes[tree_size - HASH_SIZE], HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
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
    // NOTE: contributions commit_root does NOT fire proof_done_callback. That decrement
    // is owned by the proofs_pending accounting on the Prove path; firing it here (a
    // contributions harvest) could land in the NULL-callback window between prove runs
    // and lose/mis-drive a decrement → proofs_pending never reaches zero → Prove wedges.
}

void init_gpu_setup_gpu(uint64_t maxBitsExt, uint64_t arity) {
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
    NTTGoldilocksGPU::initConstants(maxBitsExt, 1, my_gpu_ids);
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
    // prepare_blocks transposes const pols into fixedLayout() (ColMajorTiled) on the host -- this is the
    // input layout calculate_const_tree_gpu (via ldeNativeTiled) expects. Restores pre-1.0.0-beta behavior.
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

    TimerGPU timer;

    uint64_t N = 1 << nBits;
    uint64_t NExtended = 1 << nBitsExt;

    MerkleTreeGL mt(arity, 0, true, NExtended, nCols);

    uint64_t treeSize = (NExtended * nCols) + mt.numNodes;
    Goldilocks::Element* customCommitsTree = new Goldilocks::Element[treeSize];
    mt.setSource(customCommitsTree);
    mt.setNodes(&customCommitsTree[NExtended * nCols]);

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

    // Custom commits are a fixed/preprocessed section -> fixedLayout() (ColMajorTiled), restoring the
    // pre-1.0.0-beta GPU format. Transpose row-major input straight to tiled.
    fromRowMajorToColMajor(N, nCols, d_buffer, d_customCommitsPols, fixedLayout(), stream);

    Goldilocks::Element *customCommitsPols = new Goldilocks::Element[N * nCols];
    cudaMemcpyAsync(customCommitsPols, d_customCommitsPols, N * nCols * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    NTTGoldilocksGPU ntt;
    Goldilocks::Element *pNodes = (Goldilocks::Element *)&d_customCommitsTree[nCols * NExtended];
    // ldeNativeTiled directly: plain ntt.LDE would dispatch to sppark (flat) for custom dims. Out-of-place.
    ntt.ldeNativeTiled((gl64_t *)d_customCommitsTree, (gl64_t *)d_customCommitsPols, nBits, nBitsExt, nCols, stream);
    buildMerkleTreeGPU(arity, (uint64_t*)pNodes, (uint64_t*)d_customCommitsTree, nCols, 1ULL << nBitsExt, fixedLayout(), stream);

    cudaMemcpy(customCommitsTree, d_customCommitsTree, treeSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);

    Goldilocks::Element *rootGL = (Goldilocks::Element *)root;
    mt.getRoot(&rootGL[0]);

    if(std::string(bufferFile) != "") {
        std::string buffFile = string(bufferFile);
        ofstream fw(buffFile.c_str(), std::fstream::out | std::fstream::binary);
        writeFileParallel(buffFile, root, 32, 0);
        writeFileParallel(buffFile, customCommitsPols, N * nCols * sizeof(Goldilocks::Element), 32);
        writeFileParallel(buffFile, mt.source, NExtended * nCols * sizeof(Goldilocks::Element), 32 + N * nCols * sizeof(Goldilocks::Element));
        writeFileParallel(buffFile, mt.nodes, mt.numNodes * sizeof(Goldilocks::Element), 32 + (NExtended + N) * nCols * sizeof(Goldilocks::Element));
        fw.close();
    }

    delete[] customCommitsTree;
    delete[] customCommitsPols;
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
    // Const tree uses fixedLayout() (ColMajorTiled), restoring the pre-sppark (pre-1.0.0-beta) GPU format
    // so the produced .consttree_gpu matches that baseline. Call ldeNativeTiled directly: plain ntt.LDE
    // would dispatch to sppark (flat) for const dims. It is out-of-place so src stays intact.
    ntt.ldeNativeTiled((gl64_t *)d_fixedTree, (gl64_t *)d_fixedPols, starkInfo.starkStruct.nBits, starkInfo.starkStruct.nBitsExt, starkInfo.nConstants, stream);
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

// Buffer of the caller's CURRENT device. Callers that run kernels on a
// specific GPU (const-tree regeneration, recursivef) bind the device first
// and get that device's buffer.
void *get_unified_buffer_gpu_gpu(void *d_buffers_) {
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return (void *)d_buffers->gpuMemoryBuffer[d_buffers->gpus_g2l[deviceId]];
}

// Buffer of the FIRST GPU (my_gpu_ids[0], not necessarily device 0 — NUMA can
// reorder), for consumers of the acquire/release_first_gpu_buffer borrow (mem
// ops). 
void *get_first_gpu_buffer_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return nullptr;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return (void *)d_buffers->gpuMemoryBuffer[0];
}

uint64_t get_unified_buffer_gpu_size_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return 0;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    return d_buffers->unifiedBufferSize;
}

// Acquires exclusive use of the FIRST GPU's unified buffer (my_gpu_ids[0]) for the
// caller.
void acquire_first_gpu_buffer_gpu(void *d_buffers_) {
    if (d_buffers_ == nullptr) return;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
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

uint32_t selectStream(DeviceCommitBuffers* d_buffers, uint64_t airgroupId, uint64_t airId, std::string proofType, bool recursive, bool force_recursive){
    uint32_t countFreeStreamsGPU[d_buffers->n_gpus];
    uint32_t countUnusedStreams[d_buffers->n_gpus];
    int streamIdxGPU[d_buffers->n_gpus];

    for( uint32_t i = 0; i < d_buffers->n_gpus; i++){
        countUnusedStreams[i] = 0;
        countFreeStreamsGPU[i] = 0;
        streamIdxGPU[i] = -1;
    }

    bool someFree = false;
    uint32_t selectedStreamId = 0;

    std::vector<bool> streams_locked(d_buffers->n_total_streams, false);

    const uint32_t firstGpuId = d_buffers->my_gpu_ids[0];

    while (!someFree){
        // Re-read every iteration so a release by the borrower is picked up.
        const bool firstGpuBorrowed = d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire);
        if (recursive) {
            for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
                if (firstGpuBorrowed && d_buffers->streamsData[i].gpuId == firstGpuId) continue;
                if (d_buffers->streamsData[i].recursive && d_buffers->streamsData[i].mutex_stream_selection.try_lock()) {
                    // Re-check the borrow flag under the lock.
                    if (d_buffers->streamsData[i].gpuId == firstGpuId && d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire)) {
                        d_buffers->streamsData[i].mutex_stream_selection.unlock();
                        continue;
                    }
                    if (d_buffers->streamsData[i].status==0 || d_buffers->streamsData[i].status==3 || (d_buffers->streamsData[i].status==2 &&  cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess)) {
                        uint32_t gpuLocalId = d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId];

                        countFreeStreamsGPU[gpuLocalId]++;
                        if(d_buffers->streamsData[i].status==0){
                            countUnusedStreams[gpuLocalId]++;
                            streamIdxGPU[gpuLocalId] = i;
                        }
                        if (d_buffers->streamsData[i].airgroupId == airgroupId && d_buffers->streamsData[i].airId == airId && d_buffers->streamsData[i].proofType == proofType && (d_buffers->streamsData[i].status==0 || d_buffers->streamsData[i].status==3)){
                            streamIdxGPU[gpuLocalId] = i;
                        }
                        if( streamIdxGPU[gpuLocalId] == -1 ){
                            streamIdxGPU[gpuLocalId] = i;
                        }
                        someFree = true;
                        streams_locked[i] = true;
                    } else {
                        d_buffers->streamsData[i].mutex_stream_selection.unlock();
                    }
                }
            }
            if(someFree) break;
        }

        if (!recursive || !force_recursive) {
            for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
                if (firstGpuBorrowed && d_buffers->streamsData[i].gpuId == firstGpuId) continue;
                if (!d_buffers->streamsData[i].recursive && d_buffers->streamsData[i].mutex_stream_selection.try_lock()) {
                    // Re-check the borrow flag under the lock (see the recursive loop above).
                    if (d_buffers->streamsData[i].gpuId == firstGpuId && d_buffers->firstGpuBufferBorrowed.load(std::memory_order_acquire)) {
                        d_buffers->streamsData[i].mutex_stream_selection.unlock();
                        continue;
                    }
                    if (d_buffers->streamsData[i].status==0 || d_buffers->streamsData[i].status==3 || (d_buffers->streamsData[i].status==2 &&  cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess)) {
                        uint32_t gpuLocalId = d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId];

                        countFreeStreamsGPU[gpuLocalId]++;
                        if(d_buffers->streamsData[i].status==0){
                            countUnusedStreams[gpuLocalId]++;
                            streamIdxGPU[gpuLocalId] = i;
                        }
                        if (d_buffers->streamsData[i].airgroupId == airgroupId && d_buffers->streamsData[i].airId == airId && d_buffers->streamsData[i].proofType == proofType && (d_buffers->streamsData[i].status==0 || d_buffers->streamsData[i].status==3)){
                            streamIdxGPU[gpuLocalId] = i;
                        }
                        if( streamIdxGPU[gpuLocalId] == -1 ){
                            streamIdxGPU[gpuLocalId] = i;
                        }
                        someFree = true;
                        streams_locked[i] = true;
                    } else {
                        d_buffers->streamsData[i].mutex_stream_selection.unlock();
                    }
                }
            }
        }

        if (!someFree)
            std::this_thread::sleep_for(std::chrono::microseconds(300)); 
    }
    // Most free streams wins; ties break on unused count. someFree guarantees a candidate.
    int bestGpu = -1;
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        if (streamIdxGPU[i] == -1) continue;
        if (bestGpu == -1 || countFreeStreamsGPU[i] > countFreeStreamsGPU[bestGpu] ||
            (countFreeStreamsGPU[i] == countFreeStreamsGPU[bestGpu] && countUnusedStreams[i] > countUnusedStreams[bestGpu])) {
            bestGpu = i;
        }
    }
    selectedStreamId = streamIdxGPU[bestGpu];
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        if (streams_locked[i] && i != selectedStreamId) {
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
        }
    }

    reserveStreamLocked(d_buffers, selectedStreamId);
    d_buffers->streamsData[selectedStreamId].mutex_stream_selection.unlock();

    return selectedStreamId;
}

// Requires the caller to hold streamsData[streamId].mutex_stream_selection
void reserveStreamLocked(DeviceCommitBuffers* d_buffers, uint32_t streamId){
    cudaSetDevice(d_buffers->streamsData[streamId].gpuId);
    if(d_buffers->streamsData[streamId].status==2) {
        // No-op via selectStream (event already fired); any other caller must wait.
        CHECKCUDAERR(cudaEventSynchronize(d_buffers->streamsData[streamId].end_event));
        collectStreamResult(d_buffers, streamId);
    }
    d_buffers->streamsData[streamId].reset(false);
    d_buffers->streamsData[streamId].status = 1;
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