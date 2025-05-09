#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#include "starks_api.hpp"
#ifdef __USE_CUDA__
#include "gen_recursive_proof.cuh"
#include "gen_proof.cuh"
#include "gen_commit.cuh"
#include "poseidon2_goldilocks.cu"
#include <cuda_runtime.h>

struct MaxSizes
{
    uint64_t maxTraceArea;
    uint64_t maxConstArea;
    uint64_t maxAuxTraceArea;
    uint64_t maxConstTreeSize;
    bool recursive;
};


void *gen_device_commit_buffers(void *maxSizes_, uint32_t mpi_node_rank)
{
    set_device(mpi_node_rank);
    MaxSizes *maxSizes = (MaxSizes *)maxSizes_;
    DeviceCommitBuffers *buffers = new DeviceCommitBuffers();
    CHECKCUDAERR(cudaMalloc(&buffers->d_aux_trace, maxSizes->maxAuxTraceArea * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_trace, maxSizes->maxTraceArea * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_constPols, maxSizes->maxConstArea * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_constTree, maxSizes->maxConstTreeSize * sizeof(Goldilocks::Element)));
    init_gpu_const_2();
    return (void *)buffers;
}

void set_max_size_thread(void *d_buffers, uint64_t maxSizeTrace, uint64_t maxSizeContribution, uint64_t maxSizeProverBuffer, uint64_t maxSizeConst, uint64_t maxSizeConstTree, uint64_t maxProofSizeThread, uint64_t nThreads) {
    DeviceCommitBuffers *buffers = (DeviceCommitBuffers *)d_buffers;
    buffers->max_size_prover_buffer = maxSizeProverBuffer;
    buffers->max_size_trace = maxSizeTrace;
    buffers->max_size_contribution = maxSizeContribution;
    buffers->max_size_const = maxSizeConst;
    buffers->max_size_const_tree = maxSizeConstTree;
    buffers->max_size_proof = maxProofSizeThread;
    buffers->n_threads = nThreads;

    if (buffers->streams != nullptr) {
        delete[] buffers->timers;
        for (uint64_t i = 0; i < buffers->n_threads; i++) {
            cudaStreamDestroy(buffers->streams[i]);
        }
        delete[] buffers->streams;
        buffers->streams = nullptr;
    }
    
    buffers->streams = new cudaStream_t[nThreads];
    buffers->timers = new TimerGPU[nThreads];
    for (uint64_t i = 0; i < nThreads; i++) {
        CHECKCUDAERR(cudaStreamCreate(&buffers->streams[i]));
        buffers->timers[i].init(buffers->streams[i]);
    }

    buffers->pinned_buffers = new Goldilocks::Element*[nThreads];
    buffers->pinned_buffers_proof = new Goldilocks::Element*[nThreads];
    for (uint64_t i = 0; i < nThreads; i++) {
        CHECKCUDAERR(cudaMallocHost((void **)&(buffers->pinned_buffers[i]), maxSizeTrace * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMallocHost((void **)&(buffers->pinned_buffers_proof[i]), maxProofSizeThread * sizeof(Goldilocks::Element)));
    }

    if (maxSizeConst > 0) {
        buffers->pinned_buffers_const = new Goldilocks::Element*[nThreads];
        buffers->pinned_buffers_const_tree = new Goldilocks::Element*[nThreads];
        for (uint64_t i = 0; i < nThreads; i++) {
            CHECKCUDAERR(cudaMallocHost((void **)&(buffers->pinned_buffers_const[i]), maxSizeConst * sizeof(Goldilocks::Element)));
            CHECKCUDAERR(cudaMallocHost((void **)&(buffers->pinned_buffers_const_tree[i]), maxSizeConstTree * sizeof(Goldilocks::Element)));
        }
    }
}

void gen_device_commit_buffers_free(void *d_buffers, uint32_t mpi_node_rank)
{
    set_device(mpi_node_rank);
    DeviceCommitBuffers *buffers = (DeviceCommitBuffers *)d_buffers;
    CHECKCUDAERR(cudaFree(buffers->d_aux_trace));
    CHECKCUDAERR(cudaFree(buffers->d_trace));
    CHECKCUDAERR(cudaFree(buffers->d_constPols));
    CHECKCUDAERR(cudaFree(buffers->d_constTree));
    if (buffers->streams != nullptr) {
        delete[] buffers->timers;
        for (uint64_t i = 0; i < buffers->n_threads; i++) {
            cudaStreamDestroy(buffers->streams[i]);
        }
        delete[] buffers->streams;
    }
    delete buffers;
}

void load_const_pols_gpu(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char *proofType) {
    DeviceCommitBuffers *buffers = (DeviceCommitBuffers *)d_buffers;
    uint64_t sizeConstPols = constSize * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = constTreeSize * sizeof(Goldilocks::Element);
    
    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

    AirInstanceInfo& instance = buffers->air_instances[key][proofType];

    instance.const_pols_offset = initial_offset;
    instance.const_tree_offset = initial_offset + constSize;

    Goldilocks::Element *constPols = new Goldilocks::Element[constSize];
    Goldilocks::Element *constTree = new Goldilocks::Element[constTreeSize];

    loadFileParallel(constPols, constFilename, sizeConstPols);
    loadFileParallel(constTree, constTreeFilename, sizeConstTree);
    
    CHECKCUDAERR(cudaMemcpy(buffers->d_constPols + instance.const_pols_offset, constPols, sizeConstPols, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(buffers->d_constPols + instance.const_tree_offset, constTree, sizeConstTree, cudaMemcpyHostToDevice));

    delete[] constPols;
    delete[] constTree;
}

void gen_proof(void *pSetupCtx_, uint64_t threadId, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params_, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, bool loadConstants, char *constPolsPath,  char *constTreePath, uint32_t mpi_node_rank) {
    set_device(mpi_node_rank);
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;
    cudaStream_t stream = d_buffers->streams[threadId];
    TimerGPU &timer = d_buffers->timers[threadId];

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace + threadId*d_buffers->max_size_prover_buffer;

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
  
    auto key = std::make_pair(airgroupId, airId);
    auto it = d_buffers->air_instances.find(key);
    
    uint64_t offset = 0;
    // TODO!
    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0) {
        Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
        CHECKCUDAERR(cudaMemcpy(pCustomCommitsFixed, params->pCustomCommitsFixed, setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }

    if (it == d_buffers->air_instances.end() && loadConstants) {
        loadFileParallel(d_buffers->pinned_buffers_const[threadId], constPolsPath, sizeConstPols);
        loadFileParallel(d_buffers->pinned_buffers_const_tree[threadId], constTreePath, sizeConstTree);
    }
    Goldilocks::parcpy(d_buffers->pinned_buffers[threadId], (Goldilocks::Element *)params->trace, N * nCols, 4);
    offset = N * nCols;
    memcpy(&d_buffers->pinned_buffers[threadId][offset], params->publicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.nPublics;
    memcpy(&d_buffers->pinned_buffers[threadId][offset], params->proofValues, setupCtx->starkInfo.proofValuesSize * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.proofValuesSize;
    memcpy(&d_buffers->pinned_buffers[threadId][offset], params->airgroupValues, setupCtx->starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.airgroupValuesSize;
    memcpy(&d_buffers->pinned_buffers[threadId][offset], params->airValues, setupCtx->starkInfo.airValuesSize * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.airValuesSize;
    memcpy(&d_buffers->pinned_buffers[threadId][offset], globalChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    uint64_t offsetAirgroupValues = setupCtx->starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)];
    uint64_t offsetAirValues = setupCtx->starkInfo.mapOffsets[std::make_pair("airvalues", false)];
    uint64_t offsetProofValues = setupCtx->starkInfo.mapOffsets[std::make_pair("proofvalues", false)];
    uint64_t offsetChallenge = setupCtx->starkInfo.mapOffsets[std::make_pair("challenge", false)];

    offset = 0;
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetStage1, &d_buffers->pinned_buffers[threadId][offset], N * nCols * sizeof(Goldilocks::Element ), cudaMemcpyHostToDevice, stream));
    offset += N * nCols;
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetPublicInputs, &d_buffers->pinned_buffers[threadId][offset], setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
    offset += setupCtx->starkInfo.nPublics;
    if (setupCtx->starkInfo.proofValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetProofValues, &d_buffers->pinned_buffers[threadId][offset], setupCtx->starkInfo.proofValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
        offset += setupCtx->starkInfo.proofValuesSize;
    }
    if (setupCtx->starkInfo.airgroupValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetAirgroupValues, &d_buffers->pinned_buffers[threadId][offset], setupCtx->starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
        offset += setupCtx->starkInfo.airgroupValuesSize;
    }
    if (setupCtx->starkInfo.airValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetAirValues, &d_buffers->pinned_buffers[threadId][offset], setupCtx->starkInfo.airValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
        offset += setupCtx->starkInfo.airValuesSize;
    }
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetChallenge, &d_buffers->pinned_buffers[threadId][offset], FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
    offset += FIELD_EXTENSION;

    gl64_t *d_const_pols;
    gl64_t *d_const_tree;
    if (it != d_buffers->air_instances.end() && it->second.find("basic") != it->second.end()) {
        AirInstanceInfo& instance = it->second["basic"];
        d_const_pols = d_buffers->d_constPols + instance.const_pols_offset;
        d_const_tree = d_buffers->d_constPols + instance.const_tree_offset;
    } else {
        uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
        uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
        CHECKCUDAERR(cudaMemcpy(d_aux_trace + offsetConstPols, d_buffers->pinned_buffers_const[threadId], sizeConstPols, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(d_aux_trace + offsetConstTree, d_buffers->pinned_buffers_const_tree[threadId], sizeConstTree, cudaMemcpyHostToDevice));
        d_const_pols = d_aux_trace + offsetConstPols;
        d_const_tree = d_aux_trace + offsetConstTree;
    }

    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, d_buffers->pinned_buffers_proof[threadId], timer, stream);
}

void get_proof(void *pSetupCtx_, uint64_t threadId, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, uint32_t mpi_node_rank) {
    set_device(mpi_node_rank);
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    cudaStream_t stream = d_buffers->streams[threadId];
    TimerGPU timer = d_buffers->timers[threadId];

    CHECKCUDAERR(cudaStreamSynchronize(stream));
    
    TimerSyncAndLogAllGPU(timer); 

    TimerSyncCategoriesGPU(timer);
    
    TimerLogCategoryContributionsGPU(timer, STARK_GPU_PROOF);

    writeProof(*setupCtx, d_buffers->pinned_buffers_proof[threadId], proofBuffer, airgroupId, airId, instanceId, proofFile);

    TimerResetGPU(timer);
}


void gen_recursive_proof(void *pSetupCtx_, char *globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *trace, void *aux_trace, void *pConstPols, void *pConstTree, void *pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers_, bool loadConstants, uint32_t mpi_node_rank)
{
    set_device(mpi_node_rank);

    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    double time = omp_get_wtime();

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);

    CHECKCUDAERR(cudaMemcpy(d_buffers->d_trace, trace, sizeTrace, cudaMemcpyHostToDevice));
    if(loadConstants) {
        CHECKCUDAERR(cudaMemcpy(d_buffers->d_constPols, pConstPols, sizeConstPols, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(d_buffers->d_constTree, pConstTree, sizeConstTree, cudaMemcpyHostToDevice));
    }

    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetPublicInputs, (Goldilocks::Element *)pPublicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    
    time = omp_get_wtime() - time;

    time = omp_get_wtime();
    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);
    genRecursiveProof_gpu<Goldilocks::Element>(*setupCtx, globalInfo, airgroupId, airId, instanceId, (Goldilocks::Element *)trace, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, proofBuffer, string(proof_file), d_buffers, vadcop, timer, stream);
    time = omp_get_wtime() - time;
}

void commit_witness(uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *root, void *trace, void *auxTrace, uint64_t thread_id, void *d_buffers_, uint32_t mpi_node_rank) {

    double time = omp_get_wtime();
    set_device(mpi_node_rank);

    uint64_t N = 1 << nBits;

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    cudaStream_t stream = d_buffers->streams[thread_id];
    TimerGPU &timer = d_buffers->timers[thread_id];

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace + thread_id*d_buffers->max_size_contribution;
    uint64_t sizeTrace = N * nCols * sizeof(Goldilocks::Element);
    uint64_t offsetStage1 = 0;

    Goldilocks::parcpy(d_buffers->pinned_buffers[thread_id], (Goldilocks::Element *)trace, N * nCols, 4);
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetStage1, d_buffers->pinned_buffers[thread_id], sizeTrace, cudaMemcpyHostToDevice, stream));
    genCommit_gpu(arity, nBits, nBitsExt, nCols, d_aux_trace, d_buffers->pinned_buffers_proof[thread_id], timer, stream);
    time = omp_get_wtime() - time;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << time << "s";
    zklog.trace("        TIME COPY:   " + oss.str());
    oss.str("");
    oss.clear();    
}

void get_commit_root(uint64_t arity, uint64_t nBitsExt, uint64_t nCols, void *root, uint64_t thread_id, void *d_buffers_, uint32_t mpi_node_rank) {\
    double time = omp_get_wtime();

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    cudaStream_t stream = d_buffers->streams[thread_id];
    TimerGPU timer = d_buffers->timers[thread_id];

    CHECKCUDAERR(cudaStreamSynchronize(stream));
    memcpy((Goldilocks::Element *)root, d_buffers->pinned_buffers_proof[thread_id], HASH_SIZE * sizeof(uint64_t));

    TimerSyncAndLogAllGPU(timer);

    TimerSyncCategoriesGPU(timer);
    TimerLogCategoryContributionsGPU(timer, GEN_COMMIT_GPU);
    TimerResetGPU(timer);

    time = omp_get_wtime() - time;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << time << "s";
    zklog.trace("        TIME COMMIT:   " + oss.str());
    oss.str("");
    oss.clear();    
}

uint64_t check_gpu_memory(uint32_t mpi_node_rank) {
    set_device(0);
    uint64_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    std::cout << "Free memory: " << freeMem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total memory: " << totalMem / (1024.0 * 1024.0) << " MB" << std::endl;

    return freeMem;
}

// Function to set the CUDA device based on the MPI rank
// Needs to be evolved to ensuer global balance between mpi ranks and GPU devices
void set_device(uint32_t mpi_node_rank){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        exit(1);
    }
    int device = mpi_node_rank % deviceCount;
    cudaSetDevice(device);
}

#endif