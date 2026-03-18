#include "bn128.cuh"
#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#include "starks_api.hpp"
#include "starks_api_internal.hpp"
#include <cstring>
#include <thread>


struct FinalSnarkGPU;
extern void *initFinalSnarkProverGPU(char* zkeyFile);
extern void freeFinalSnarkProverGPU(void *snark_prover);
extern void genFinalSnarkProofGPU(void *proverSnark, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark);
extern void preAllocateFinalSnarkProverGPU(void *snark_prover, void* unified_buffer_gpu);
extern uint64_t getFinalSnarkProtocolIdGPU(void *snark_prover);
#ifdef __USE_CUDA__
#include "verify_constraints.cuh"
#include "gen_proof.cuh"
#include "poseidon2_goldilocks.cuh"
#include "hints.cuh"
#include "gen_recursivef_proof.cuh"
#include "poseidon_bn128.cuh"
#include <cuda_runtime.h>
#include <mutex>
#include <algorithm>
#include <map>


uint32_t selectStream(DeviceCommitBuffers* d_buffers, uint64_t airgroupId, uint64_t airId, std::string proofType, bool recursive = false, bool force_recursive = false);
void reserveStream(DeviceCommitBuffers* d_buffers, uint32_t streamId);
void closeStreamTimer(TimerGPU &timer, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, bool isProve);
void get_proof(DeviceCommitBuffers *d_buffers, uint64_t streamId);
void get_commit_root(DeviceCommitBuffers *d_buffers, uint64_t streamId);


void get_instances_ready(void *d_buffers_, int64_t* instances_ready) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        instances_ready[i] = d_buffers->streamsData[i].instanceId;
    }
}

void *gen_device_buffers(uint32_t node_rank, uint32_t node_size, const int32_t* numa_nodes, uint32_t arity, uint32_t max_n_bits_ext)
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

    // Force CUDA context initialization
    int device_id;
    cudaGetDevice(&device_id);
    for (uint32_t i = 0; i < n_gpus; i++) {
        cudaSetDevice(my_gpu_ids[i]);
        cudaFree(0);
    }
    cudaSetDevice(device_id);
    cudaDeviceSynchronize();

    // Initialize small GPU constants (Poseidon2 and Transcript)
    switch(arity){
        case 2:
            Poseidon2GoldilocksGPU<8>::initPoseidon2GPUConstants(my_gpu_ids, n_gpus);
            break;
        case 3:
            Poseidon2GoldilocksGPU<12>::initPoseidon2GPUConstants(my_gpu_ids, n_gpus);
            break;
        case 4:
            Poseidon2GoldilocksGPU<16>::initPoseidon2GPUConstants(my_gpu_ids, n_gpus);
            break;
        default:
            zklog.error("Unsupported merkle tree arity. Supported arities are 2, 3 and 4.");
            exit(1);
    }

    Poseidon2GoldilocksGPUGrinding::initPoseidon2GPUConstants(my_gpu_ids, n_gpus);
    TranscriptGL_GPU::init_const(my_gpu_ids, n_gpus, arity);

    //Generate static twiddles for the NTT
    NTT_Goldilocks_GPU::init_twiddle_factors_and_r(max_n_bits_ext, n_gpus, my_gpu_ids);

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

void alloc_device_large_buffers(void *d_buffers_, uint64_t auxTraceArea, uint64_t auxTraceRecursiveArea, uint64_t totalConstPols, uint64_t totalConstPolsAggregation)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    // Calculate total memory needed per GPU
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

uint64_t gen_device_streams(void *d_buffers_, uint64_t n_streams, uint64_t n_recursive_streams, uint64_t maxSizeProverBuffer, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t merkleTreeArity) {
    
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

void reset_device_streams(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
   
    for(uint64_t i=0; i< d_buffers->n_total_streams; ++i){
        d_buffers->streamsData[i].instanceId = -1;
        d_buffers->streamsData[i].reset(true);
    }
}

void free_device_buffers(void *d_buffers_)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

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

    if (d_buffers->streamsData != nullptr) {
        for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
            d_buffers->streamsData[i].free();
        }
        delete[] d_buffers->streamsData;
    }

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


void load_device_setup(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, void *packed_info) {
    
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
        d_buffers->air_instances[key][proofType][i] = new AirInstanceInfo(airgroupId, airId, setupCtx, verkeyRoot, packedInfo);
    }
}

void load_device_const_pols(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers_, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char *proofType, bool onlyFirstGPU) {
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

uint64_t gen_proof(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params_, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, bool skipRecalculation, uint64_t streamId_, char *constPolsPath,  char *constTreePath) {

    auto key = std::make_pair(airgroupId, airId);
    std::string proofType = "basic";

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint32_t streamId = skipRecalculation ? streamId_ : selectStream(d_buffers, airgroupId, airId, proofType, false);
    if (skipRecalculation) reserveStream(d_buffers, streamId);
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

    bool reuse_constants = !air_instance_info->stored_tree && d_buffers->streamsData[streamId].airgroupId == airgroupId && d_buffers->streamsData[streamId].airId == airId && d_buffers->streamsData[streamId].proofType == string("basic");

    d_buffers->streamsData[streamId].pSetupCtx = pSetupCtx_;
    d_buffers->streamsData[streamId].proofBuffer = proofBuffer;
    d_buffers->streamsData[streamId].proofFile = string(proofFile);
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].proofType = "basic";

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetStage1Extended = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];

    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0) {
        Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
        copy_to_device_in_chunks(d_buffers, params->pCustomCommitsFixed, pCustomCommitsFixed, setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element), streamId, timer);
    }

    if (!skipRecalculation) {
        uint64_t total_size = air_instance_info->is_packed ? air_instance_info->num_packed_words * N * sizeof(Goldilocks::Element) : N * nCols * sizeof(Goldilocks::Element);
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


    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, constTreePath, streamId, instanceId, d_buffers, air_instance_info, skipRecalculation, timer, stream, false, reuse_constants);
    cudaEventRecord(d_buffers->streamsData[streamId].end_event, stream);
    d_buffers->streamsData[streamId].status = 2;
    return streamId;
}

uint64_t initialize_instance(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* params_, void *d_buffers_) {
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
   
    bool reuse_constants = d_buffers->streamsData[streamId].airgroupId == airgroupId && d_buffers->streamsData[streamId].airId == airId && d_buffers->streamsData[streamId].proofType == string("basic");

    d_buffers->streamsData[streamId].pSetupCtx = pSetupCtx_;
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].proofType = "basic";
    d_buffers->streamsData[streamId].instanceId = instanceId;

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];

    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0) {
        Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
        copy_to_device_in_chunks(d_buffers, params->pCustomCommitsFixed, pCustomCommitsFixed, setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element), streamId, timer);
    }

    uint64_t total_size = air_instance_info->is_packed ? air_instance_info->num_packed_words * N * sizeof(Goldilocks::Element) : N * nCols * sizeof(Goldilocks::Element);
    uint64_t *dst = (uint64_t *)(d_aux_trace + offsetStage1 + N * nCols);
    copy_to_device_in_chunks(d_buffers, params->trace, dst, total_size, streamId, timer);    
    
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
    if(!reuse_constants) {
        unpack_fixed((uint64_t*)d_const_pols, (uint64_t*)(d_const_pols + 1), (uint64_t*)(d_const_pols + 1 + setupCtx->starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx->starkInfo.nConstants, N, stream, timer);
        CHECKCUDAERR(cudaGetLastError());
    }

    uint64_t offsetCm1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    if (air_instance_info->is_packed) {
        unpack_trace(air_instance_info, (uint64_t*)(d_aux_trace + offsetCm1 + N * nCols), (uint64_t*)(d_aux_trace + offsetCm1), nCols, N, stream, timer); 
    } else {
        NTT_Goldilocks_GPU ntt;
        ntt.prepare_blocks_trace((gl64_t*)(d_aux_trace + offsetCm1), (gl64_t *)(d_aux_trace + offsetCm1 + N * nCols), nCols, N, stream, timer);
    }

    return streamId;
}

void calculate_trace_instance(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, void *params_, void *d_buffers_, uint64_t streamId) {
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

void verify_constraints(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, void* params_, void* constraintsInfo, void *d_buffers_, uint64_t streamId) {

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

void get_stream_proofs(void *d_buffers_){
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        d_buffers->streamsData[i].mutex_stream_selection.lock();
        if (d_buffers->streamsData[i].status == 0 || d_buffers->streamsData[i].status == 3) {
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
            continue;
        }
        cudaSetDevice(d_buffers->streamsData[i].gpuId);
        CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[i].stream));
        if(d_buffers->streamsData[i].root != nullptr) {
            get_commit_root(d_buffers, i);
        }else if (d_buffers->streamsData[i].proofBuffer != nullptr) {
            get_proof(d_buffers, i);
        }
        d_buffers->streamsData[i].reset(false);
        d_buffers->streamsData[i].mutex_stream_selection.unlock();
    }
}

void get_stream_proofs_non_blocking(void *d_buffers_){
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    for (uint64_t i = 0; i < d_buffers->n_total_streams; i++) {
        if (d_buffers->streamsData[i].mutex_stream_selection.try_lock()) {
            if(d_buffers->streamsData[i].status==2 &&  cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess) {
                cudaSetDevice(d_buffers->streamsData[i].gpuId);
                if(d_buffers->streamsData[i].root != nullptr) {
                    get_commit_root(d_buffers, i);
                } else if (d_buffers->streamsData[i].proofBuffer != nullptr) {
                    get_proof(d_buffers, i);
                }
                d_buffers->streamsData[i].reset(false);
            }
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
        }
    }
}

void get_stream_id_proof(void *d_buffers_, uint64_t streamId) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    cudaSetDevice(d_buffers->streamsData[streamId].gpuId);
    CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[streamId].stream));
    if(d_buffers->streamsData[streamId].root != nullptr) {
            get_commit_root(d_buffers, streamId);
        } else if (d_buffers->streamsData[streamId].proofBuffer != nullptr) {
            get_proof(d_buffers, streamId);
        }

    d_buffers->streamsData[streamId].reset(false); 
}

uint64_t gen_recursive_proof(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *trace, void *aux_trace, void *pConstPols, void *pConstTree, void *pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers_, char *constPolsPath, char *constTreePath, char *proofType, bool force_recursive_stream)
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

    bool reuse_constants = !air_instance_info->stored_tree && d_buffers->streamsData[streamId].airgroupId == airgroupId && d_buffers->streamsData[streamId].airId == airId && d_buffers->streamsData[streamId].proofType == string(proofType);

    d_buffers->streamsData[streamId].pSetupCtx = pSetupCtx_;
    d_buffers->streamsData[streamId].proofBuffer = proofBuffer;
    d_buffers->streamsData[streamId].proofFile = string(proof_file);
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].proofType = string(proofType);

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

void tile_const_pols(void *pStarkinfo, void *pConstPols, char *constFile, void *pConstTree, char *constTreeFile, void *unified_buffer_gpu) {

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
    fromRowMajorToTiled<<<gridSize, blockSize, 0, stream>>>(N, nConst, (uint64_t*)d_helper, (uint64_t*)d_helperAux);
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
    fromRowMajorToTiled<<<gridSize, blockSize, 0, stream>>>(NExtended, nConst, (uint64_t*)d_helper, (uint64_t*)d_helperAux);
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

void *gen_device_buffers_recursivef(void *pSetupCtx_, uint64_t proverBufferSize, void *d_commit_buffer_,  char* verkey) {
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    uint32_t gpuId = 0;
    if (d_commit_buffer_ != nullptr) {
        DeviceCommitBuffers *d_commit_buffer = (DeviceCommitBuffers *)d_commit_buffer_;
        gpuId = d_commit_buffer->my_gpu_ids[0];
    }
    cudaSetDevice(gpuId);
    
    DeviceRecursiveFBuffers *d_buffers = new DeviceRecursiveFBuffers();
    d_buffers->gpuId = gpuId;
    
    // Initialize BN128 Poseidon GPU constants for merkletree and transcript
    PoseidonBN128GPU::initGPUConstants(&gpuId, 1);
    uint64_t transcriptArity = setupCtx->starkInfo.starkStruct.merkleTreeCustom ? setupCtx->starkInfo.starkStruct.merkleTreeArity : 16;
    TranscriptBN128_GPU::init_const(&gpuId, 1, transcriptArity);

    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
    uint64_t sizeAuxTrace = proverBufferSize;

    if (d_commit_buffer_ == nullptr) {
        NTT_Goldilocks_GPU::init_twiddle_factors_and_r(22, 1, &gpuId); //max nBitsExt=21
        // Allocate new device buffers
        d_buffers->owns_aux_trace = true;
        d_buffers->owns_const_tree = true;
        CHECKCUDAERR(cudaMalloc(&d_buffers->d_aux_trace, sizeAuxTrace));
        CHECKCUDAERR(cudaMalloc(&d_buffers->d_const_tree, sizeConstTree));
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

void alloc_fixed_pols_buffer_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    uint32_t gpuId = d_buffers->my_gpu_ids[0];
    cudaSetDevice(gpuId);
    CHECKCUDAERR(cudaMalloc(&d_buffers->d_constPols[d_buffers->gpus_g2l[gpuId]], d_buffers->constPolsSize));
}

void free_fixed_pols_buffer_gpu(void *d_buffers_) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    uint32_t gpuId = d_buffers->my_gpu_ids[0];
    cudaSetDevice(gpuId);
    CHECKCUDAERR(cudaFree(d_buffers->d_constPols[d_buffers->gpus_g2l[gpuId]]));
}

void load_fixed_pols_recursivef(void *pSetupCtx_, void *pConstTree, void *d_buffers_) {
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

void free_device_buffers_recursivef(void *d_buffers_) {
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

void *gen_recursive_proof_final(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, uint64_t proverBufferSize, void* d_buffers_) {
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
    fromRowMajorToTiled<<<gridSize, blockSize, 0, d_buffers->stream>>>(N, nCols, (uint64_t*)d_witness_temp, (uint64_t*)d_witness);
    CHECKCUDAERR(cudaGetLastError());

    // Copy public inputs
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetPublicInputs, (const gl64_t*)pPublicInputs, sizePublicInputs, cudaMemcpyHostToDevice, d_buffers->stream));

    uint64_t nConst = setupCtx->starkInfo.nConstants;
    uint64_t sizeConstPols = N * nConst * sizeof(Goldilocks::Element);
    // Copy const pols to device
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    copy_to_device_in_chunks((const uint8_t*)pConstPols, (uint8_t*)(d_aux_trace + offsetConstPols), sizeConstPols, pinnedBuffer, pinnedBufferSize, d_buffers->stream);
    CHECKCUDAERR(cudaGetLastError());

    void* result = genRecursiveProofBN128_gpu(*setupCtx, airgroupId, airId, instanceId, (Goldilocks::Element *)d_aux_trace, (Goldilocks::Element *)pPublicInputs, string(proof_file), d_buffers);

    cudaStreamSynchronize(d_buffers->stream);

    return result;
}

uint64_t commit_witness(void *pSetupCtx_, void *params_, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *d_buffers_) {

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint32_t streamId = selectStream(d_buffers, airgroupId, airId, "basic");
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];

    d_buffers->streamsData[streamId].root = root;
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].proofType = "witness";

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
    
    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace[gpuLocalId][d_buffers->streamsData[streamId].localStreamId];
    uint64_t sizeTrace = N * nCols * sizeof(Goldilocks::Element);
    uint64_t offsetStage1Extended = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t total_size = air_instance_info->is_packed ? air_instance_info->num_packed_words * N * sizeof(Goldilocks::Element) : sizeTrace;
    uint64_t *dst = (uint64_t*)(d_aux_trace + offsetStage1Extended);
    copy_to_device_in_chunks(d_buffers, params->trace, dst, total_size, streamId, timer);
    
    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(NExtended, arity);

    uint64_t offset_src = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offset_dst = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)];
    uint64_t offset_mt = setupCtx->starkInfo.mapOffsets[make_pair("mt1", true)];

    Goldilocks::Element *pNodes = (Goldilocks::Element*)d_aux_trace + offset_mt;
    NTT_Goldilocks_GPU ntt;

    if (air_instance_info->is_packed) {
        unpack_trace(air_instance_info, (uint64_t *)(d_aux_trace + offset_dst), (uint64_t *)(d_aux_trace + offset_src), nCols, N, stream, timer);
    } else {
        ntt.prepare_blocks_trace((gl64_t *)(d_aux_trace + offset_src), (gl64_t *)(d_aux_trace + offset_dst), nCols, N, stream, timer);
    }

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

        Goldilocks::Element *pCustomCommitsFixed = nullptr;
        if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0) {
            pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            copy_to_device_in_chunks(d_buffers, params->pCustomCommitsFixed, pCustomCommitsFixed, setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element), streamId, timer);
        }

        size_t totalCopySize = 0;
        totalCopySize += setupCtx->starkInfo.nPublics;
        totalCopySize += setupCtx->starkInfo.proofValuesSize;
        totalCopySize += setupCtx->starkInfo.airgroupValuesSize;
        totalCopySize += setupCtx->starkInfo.airValuesSize;

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

        copy_to_device_in_chunks(d_buffers, aux_values, (uint8_t*)(d_aux_trace + offsetPublicInputs), totalCopySize * sizeof(Goldilocks::Element), streamId, timer);

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
            pCustomCommitsFixed,
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

    ntt.LDE_MerkleTree_GPU(pNodes, d_aux_trace, offset_dst, d_aux_trace, offset_src, nBits, nBitsExt, nCols, arity, timer, stream);
    CHECKCUDAERR(cudaMemcpyAsync(d_buffers->streamsData[streamId].pinned_buffer_proof, &pNodes[tree_size - HASH_SIZE], HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
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
    
   

    if (proof_done_callback != nullptr) {
        proof_done_callback(instanceId, "");
    }

}

void init_gpu_setup(uint64_t maxBitsExt) {
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    cudaSetDevice(deviceId);
    uint32_t my_gpu_ids[1] = {(uint32_t)deviceId};

    // Uploads constants for all possible arities
    Poseidon2GoldilocksGPU<16>::initPoseidon2GPUConstants(my_gpu_ids, 1);
    NTT_Goldilocks_GPU::init_twiddle_factors_and_r(maxBitsExt, 1, my_gpu_ids);
}

void prepare_blocks(uint64_t *pol, uint64_t N, uint64_t nCols, void *unified_buffer_gpu) {
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
    NTT_Goldilocks_GPU ntt;
    ntt.prepare_blocks_trace(d_aux, d_pol, nCols, N, stream, timer);

    cudaMemcpy(pol, d_aux, N * nCols * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    if (unified_buffer_gpu == nullptr) {
        CHECKCUDAERR(cudaFree(d_pol));
        CHECKCUDAERR(cudaFree(d_aux));
    }
    cudaStreamDestroy(stream);
}

void write_custom_commit(void* root, uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *d_buffers_, void *buffer, char *bufferFile)
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

    NTT_Goldilocks_GPU ntt;
    ntt.prepare_blocks_trace(d_customCommitsPols, d_buffer, nCols, N, stream, timer);

    Goldilocks::Element *pNodes = (Goldilocks::Element *)&d_customCommitsTree[nCols * NExtended];
    ntt.LDE_MerkleTree_GPU(pNodes, (gl64_t *)d_customCommitsTree, 0, (gl64_t *)d_customCommitsPols, 0, nBits, nBitsExt, nCols, arity, timer, stream);

    cudaMemcpy(customCommitsTree, d_customCommitsTree, treeSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);

    Goldilocks::Element *rootGL = (Goldilocks::Element *)root;
    mt.getRoot(&rootGL[0]);

    Goldilocks::Element *customCommitsPols = new Goldilocks::Element[N * nCols];
    cudaMemcpy(customCommitsPols, d_customCommitsPols, N * nCols * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
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

void calculate_const_tree(void *pStarkInfo, void *pConstPolsAddress, void *pConstTreeAddress_, void *unified_buffer_gpu) {
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

    NTT_Goldilocks_GPU ntt;

    Goldilocks::Element *pNodes = d_fixedTree + starkInfo.nConstants * NExtended;
    ntt.LDE_MerkleTree_GPU(pNodes, (gl64_t *)d_fixedTree, 0, (gl64_t *)d_fixedPols, 0, starkInfo.starkStruct.nBits, starkInfo.starkStruct.nBitsExt, starkInfo.nConstants, starkInfo.starkStruct.merkleTreeArity, timer, stream);

    Goldilocks::Element *pConstTreeAddress = (Goldilocks::Element *)pConstTreeAddress_;
    cudaMemcpy(pConstTreeAddress, d_fixedTree, treeSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
    if (unified_buffer_gpu == nullptr) {
        cudaFree(d_fixedPols);
        cudaFree(d_fixedTree);
    }
    TimerStopGPU(timer, STARK_GPU_CONST_TREE);
    cudaStreamDestroy(stream);
}

uint64_t check_device_memory(uint32_t node_rank, uint32_t node_size)
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

uint64_t get_num_gpus() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error getting device count: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    return deviceCount;
}

void *get_unified_buffer_gpu(void *d_buffers_) {
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    cudaSetDevice(deviceId);

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    gl64_t *d_unifiedBuffer = d_buffers->gpuMemoryBuffer[d_buffers->gpus_g2l[deviceId]];
    return (void *)d_unifiedBuffer;
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
    
    while (!someFree){
        if (recursive) {
            for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
                if (d_buffers->streamsData[i].recursive && d_buffers->streamsData[i].mutex_stream_selection.try_lock()) {
                    if (d_buffers->streamsData[i].status==0 || d_buffers->streamsData[i].status==3 || (d_buffers->streamsData[i].status==2 &&  cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess)) {

                        countFreeStreamsGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]]++;
                        if(d_buffers->streamsData[i].status==0){
                            countUnusedStreams[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]]++;
                            streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] = i;
                        }
                        if (d_buffers->streamsData[i].airgroupId == airgroupId && d_buffers->streamsData[i].airId == airId && d_buffers->streamsData[i].proofType == proofType && d_buffers->streamsData[i].status==0){
                            streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] = i;
                        }
                        if( streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] == -1 ){
                            streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] = i;
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
                if (!d_buffers->streamsData[i].recursive && d_buffers->streamsData[i].mutex_stream_selection.try_lock()) {
                    if (d_buffers->streamsData[i].status==0 || d_buffers->streamsData[i].status==3 || (d_buffers->streamsData[i].status==2 &&  cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess)) {
                        countFreeStreamsGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]]++;
                        if(d_buffers->streamsData[i].status==0){
                            countUnusedStreams[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]]++;
                            streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] = i;
                        }
                        if (d_buffers->streamsData[i].airgroupId == airgroupId && d_buffers->streamsData[i].airId == airId && d_buffers->streamsData[i].proofType == proofType && d_buffers->streamsData[i].status==0){
                            streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] = i;
                        }
                        if( streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] == -1 ){
                            streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] = i;
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
    // Original selection logic for single stream
    uint32_t maxFree = 0;
    uint32_t streamId = 0;
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        if (countFreeStreamsGPU[i] > maxFree || (countFreeStreamsGPU[i] == maxFree && countUnusedStreams[i] > countUnusedStreams[streamId])) {
            maxFree = countFreeStreamsGPU[i];
            streamId = streamIdxGPU[i];
        }
    }
    selectedStreamId = streamId;
    for (uint32_t i = 0; i < d_buffers->n_total_streams; i++) {
        if (streams_locked[i] && i != selectedStreamId) {
            d_buffers->streamsData[i].mutex_stream_selection.unlock();
        }
    }

    reserveStream(d_buffers, selectedStreamId);
    d_buffers->streamsData[selectedStreamId].mutex_stream_selection.unlock();

    return selectedStreamId;
}

void reserveStream(DeviceCommitBuffers* d_buffers, uint32_t streamId){
    cudaSetDevice(d_buffers->streamsData[streamId].gpuId);
    if(d_buffers->streamsData[streamId].status==2 && cudaEventQuery(d_buffers->streamsData[streamId].end_event) == cudaSuccess) {

        if(d_buffers->streamsData[streamId].root != nullptr) {
            get_commit_root(d_buffers, streamId);
        } else if (d_buffers->streamsData[streamId].proofBuffer != nullptr) {
            get_proof(d_buffers, streamId);
        }
    }
    d_buffers->streamsData[streamId].reset(false);
    d_buffers->streamsData[streamId].status = 1;
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

void *init_final_snark_prover(char* zkeyFile) {
    return initFinalSnarkProverGPU(zkeyFile);
}

void free_final_snark_prover(void *snark_prover) {
    freeFinalSnarkProverGPU(snark_prover);
}

void gen_final_snark_proof(void *prover, void *circomWitnessFinal, uint8_t* proof, uint8_t* publicsSnark) {
    genFinalSnarkProofGPU(prover, circomWitnessFinal, proof, publicsSnark);
}

void pre_allocate_final_snark_prover(void *snark_prover, void* unified_buffer_gpu) {
    preAllocateFinalSnarkProverGPU(snark_prover, unified_buffer_gpu);
}
#endif