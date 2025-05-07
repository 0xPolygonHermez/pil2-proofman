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
    buffers->recursive = maxSizes->recursive;
    CHECKCUDAERR(cudaMalloc(&buffers->d_aux_trace, maxSizes->maxAuxTraceArea * sizeof(Goldilocks::Element)));
    if(buffers->recursive) {
        CHECKCUDAERR(cudaMalloc(&buffers->d_trace, maxSizes->maxTraceArea * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc(&buffers->d_constPols, maxSizes->maxConstArea * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc(&buffers->d_constTree, maxSizes->maxConstTreeSize * sizeof(Goldilocks::Element)));
    }
    init_gpu_const_2();
    return (void *)buffers;
}

void set_max_size_thread(void *d_buffers, uint64_t maxSizeTrace, uint64_t maxSizeContribution, uint64_t maxSizeProverBuffer, uint64_t nThreads) {
    DeviceCommitBuffers *buffers = (DeviceCommitBuffers *)d_buffers;
    buffers->max_size_prover_buffer = maxSizeProverBuffer;
    buffers->max_size_trace = maxSizeTrace;
    buffers->max_size_contribution = maxSizeContribution;
    buffers->n_threads = nThreads;

    if (buffers->streams != nullptr) {
        for (uint64_t i = 0; i < buffers->n_threads; i++) {
            cudaStreamDestroy(buffers->streams[i]);
        }
        delete[] buffers->streams;
        buffers->streams = nullptr;
    }
    
    buffers->streams = new cudaStream_t[nThreads];
    for (uint64_t i = 0; i < nThreads; i++) {
        CHECKCUDAERR(cudaStreamCreate(&buffers->streams[i]));
    }

    buffers->pinned_buffers = new Goldilocks::Element*[nThreads];
    for (uint64_t i = 0; i < nThreads; i++) {
        CHECKCUDAERR(cudaMallocHost((void **)&(buffers->pinned_buffers[i]), maxSizeTrace * sizeof(Goldilocks::Element)));
    }
}

void gen_device_commit_buffers_free(void *d_buffers, uint32_t mpi_node_rank)
{
    set_device(mpi_node_rank);
    DeviceCommitBuffers *buffers = (DeviceCommitBuffers *)d_buffers;
    CHECKCUDAERR(cudaFree(buffers->d_aux_trace));
    if(buffers->recursive) {
        CHECKCUDAERR(cudaFree(buffers->d_trace));
        CHECKCUDAERR(cudaFree(buffers->d_constPols));
        CHECKCUDAERR(cudaFree(buffers->d_constTree));
    }
    if (buffers->streams != nullptr) {
        for (uint64_t i = 0; i < buffers->n_threads; i++) {
            cudaStreamDestroy(buffers->streams[i]);
        }
        delete[] buffers->streams;
        buffers->streams = nullptr;
    }
    delete buffers;
}

void gen_proof(void *pSetupCtx_, uint64_t threadId, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params_, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, bool loadConstants, uint32_t mpi_node_rank) {

    double time = omp_get_wtime();
    set_device(mpi_node_rank);
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace + threadId*d_buffers->max_size_prover_buffer;

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    uint64_t offsetAirgroupValues = setupCtx->starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)];
    uint64_t offsetAirValues = setupCtx->starkInfo.mapOffsets[std::make_pair("airvalues", false)];
    uint64_t offsetProofValues = setupCtx->starkInfo.mapOffsets[std::make_pair("proofvalues", false)];
    uint64_t offsetChallenge = setupCtx->starkInfo.mapOffsets[std::make_pair("challenge", false)];

    double timeCopy = omp_get_wtime();
    CHECKCUDAERR(cudaMemcpy(d_aux_trace + offsetStage1, params->trace, sizeTrace, cudaMemcpyHostToDevice));
    timeCopy = omp_get_wtime() - timeCopy;
    double timeCopyConstants = omp_get_wtime();
    if(loadConstants) {
        CHECKCUDAERR(cudaMemcpy(d_aux_trace + offsetConstPols, params->pConstPolsAddress, sizeConstPols, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(d_aux_trace + offsetConstTree, params->pConstPolsExtendedTreeAddress, sizeConstTree, cudaMemcpyHostToDevice));
    }
    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0) {
        Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_buffers->d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
        CHECKCUDAERR(cudaMemcpy(pCustomCommitsFixed, params->pCustomCommitsFixed, setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetPublicInputs, params->publicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    if (setupCtx->starkInfo.proofValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetProofValues, params->proofValues, setupCtx->starkInfo.proofValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }
    if (setupCtx->starkInfo.airgroupValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetAirgroupValues, params->airgroupValues, setupCtx->starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }
    if (setupCtx->starkInfo.airValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetAirValues, params->airValues, setupCtx->starkInfo.airValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }

    Goldilocks::Element *d_global_challenge = (Goldilocks::Element *)d_buffers->d_aux_trace + offsetChallenge;
    CHECKCUDAERR(cudaMemcpy(d_global_challenge, globalChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
           
    timeCopyConstants = omp_get_wtime() - timeCopyConstants;

    time = omp_get_wtime();
    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);
    genProof_gpu(*setupCtx, d_buffers->d_aux_trace, timer, stream);
    getProof_gpu(*setupCtx, airgroupId, airId, instanceId, proofBuffer, string(proofFile), d_buffers->d_aux_trace);
    time = omp_get_wtime() - time;

    std::ostringstream oss;
    
    oss << std::fixed << std::setprecision(2) << timeCopy << "s";
    zklog.trace("        COPY_TRACE:   " + oss.str());
    oss.str("");
    oss.clear();

    oss << std::fixed << std::setprecision(2) << timeCopyConstants << "s";
    zklog.trace("        COPY_CONST:   " + oss.str());
    oss.str("");
    oss.clear();
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
    cudaEvent_t commitWitness;
    cudaEventCreate(&commitWitness);
    set_device(mpi_node_rank);

    uint64_t N = 1 << nBits;

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    cudaStream_t stream = d_buffers->streams[thread_id];

    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace + thread_id*d_buffers->max_size_contribution;
    uint64_t sizeTrace = N * nCols * sizeof(Goldilocks::Element);
    uint64_t offsetStage1 = 0;

    Goldilocks::parcpy(d_buffers->pinned_buffers[thread_id], (Goldilocks::Element *)trace, N * nCols, 4);
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetStage1, d_buffers->pinned_buffers[thread_id], sizeTrace, cudaMemcpyHostToDevice, stream));
    cudaEventRecord(commitWitness, stream);
    genCommit_gpu(arity, nBits, nBitsExt, nCols, d_aux_trace, stream);
    cudaEventSynchronize(commitWitness);
    time = omp_get_wtime() - time;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << time << "s";
    zklog.trace("        TIME COPY:   " + oss.str());
    oss.str("");
    oss.clear();    
}

void get_commit_root(uint64_t arity, uint64_t nBitsExt, uint64_t nCols, void *root, uint64_t thread_id, void *d_buffers_, uint32_t mpi_node_rank) {\
    double time = omp_get_wtime();

    uint64_t NExtended = 1 << nBitsExt;

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    cudaStream_t stream = d_buffers->streams[thread_id];
    gl64_t *d_aux_trace = (gl64_t *)d_buffers->d_aux_trace + thread_id*d_buffers->max_size_contribution;

    Goldilocks::Element *rootGL = (Goldilocks::Element *)root;
    Goldilocks::Element *pNodes = (Goldilocks::Element*) d_aux_trace + nCols * NExtended;
    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(NExtended, arity);
    CHECKCUDAERR(cudaMemcpyAsync(rootGL, pNodes + tree_size - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));
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