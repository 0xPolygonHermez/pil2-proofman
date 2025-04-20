#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#include "starks_api.hpp"
#ifdef __USE_CUDA__
#include "gen_recursive_proof.cuh"
#include "gen_proof.cuh"
#include "gen_commit.cuh"

struct MaxSizes
{
    uint64_t maxTraceArea;
    uint64_t maxConstArea;
    uint64_t maxAuxTraceArea;
    uint64_t maxConstTreeSize;
    bool recursive;
};


void *gen_device_commit_buffers(void *maxSizes_, uint32_t mpi_rank)
{

    MaxSizes *maxSizes = (MaxSizes *)maxSizes_;
    DeviceCommitBuffers *buffers = new DeviceCommitBuffers();
    buffers->recursive = maxSizes->recursive;
    CHECKCUDAERR(cudaMalloc(&buffers->d_aux_trace, maxSizes->maxAuxTraceArea * sizeof(Goldilocks::Element)));
    if(buffers->recursive) {
        CHECKCUDAERR(cudaMalloc(&buffers->d_trace, maxSizes->maxTraceArea * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc(&buffers->d_constPols, maxSizes->maxConstArea * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc(&buffers->d_constTree, maxSizes->maxConstTreeSize * sizeof(Goldilocks::Element)));
    }
    return (void *)buffers;
}

void gen_device_commit_buffers_free(void *d_buffers, uint32_t mpi_rank)
{
    DeviceCommitBuffers *buffers = (DeviceCommitBuffers *)d_buffers;
    CHECKCUDAERR(cudaFree(buffers->d_aux_trace));
    if(buffers->recursive) {
        CHECKCUDAERR(cudaFree(buffers->d_trace));
        CHECKCUDAERR(cudaFree(buffers->d_constPols));
        CHECKCUDAERR(cudaFree(buffers->d_constTree));
    }
    delete buffers;
}

void gen_proof(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params_, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, bool loadConstants, uint32_t mpi_rank) {

    double time = omp_get_wtime();
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
    uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
    double timeCopy = omp_get_wtime();
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetStage1, params->trace, sizeTrace, cudaMemcpyHostToDevice));
    timeCopy = omp_get_wtime() - timeCopy;
    double timeCopyConstants = omp_get_wtime();
    if(loadConstants) {
        CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetConstPols, &params->aux_trace[offsetConstPols], sizeConstPols, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetConstTree, &params->aux_trace[offsetConstTree], sizeConstTree, cudaMemcpyHostToDevice));
    }
    timeCopyConstants = omp_get_wtime() - timeCopyConstants;

    //std::cout << "rick genDeviceBuffers time: " << time << std::endl;

    time = omp_get_wtime();
    genProof_gpu(*setupCtx, airgroupId, airId, instanceId, *params, (Goldilocks::Element *)globalChallenge, proofBuffer, string(proofFile), d_buffers);
    time = omp_get_wtime() - time;
    //std::cout << "rick genRecursiveProof_gpu time: " << time << std::endl;

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

void gen_recursive_proof(void *pSetupCtx_, char *globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *trace, void *aux_trace, void *pConstPols, void *pConstTree, void *pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers_, bool loadConstants, uint32_t mpi_rank)
{

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
    
    time = omp_get_wtime() - time;
    // std::cout << "rick genDeviceBuffers time: " << time << std::endl;

    time = omp_get_wtime();
    genRecursiveProof_gpu<Goldilocks::Element>(*setupCtx, globalInfo, airgroupId, airId, instanceId, (Goldilocks::Element *)trace, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, proofBuffer, string(proof_file), d_buffers, vadcop);
    time = omp_get_wtime() - time;
    // std::cout << "rick genRecursiveProof_gpu time: " << time << std::endl;
}

void commit_witness(uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *root, void *trace, void *auxTrace, void *d_buffers_, uint32_t mpi_rank) {

    double time = omp_get_wtime();

    Goldilocks::Element *rootGL = (Goldilocks::Element *)root;
    uint64_t N = 1 << nBits;
    uint64_t NExtended = 1 << nBitsExt;


    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint64_t sizeTrace = N * nCols * sizeof(Goldilocks::Element);
    uint64_t offsetStage1 = 0;

    double timeCopy = omp_get_wtime();
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_aux_trace + offsetStage1, trace, sizeTrace, cudaMemcpyHostToDevice));
    timeCopy = omp_get_wtime() - timeCopy;
    genCommit_gpu(arity, rootGL, N, NExtended, nCols, d_buffers);
    time = omp_get_wtime() - time;

    std::ostringstream oss;

    oss << std::fixed << std::setprecision(2) << timeCopy << "s (" << (timeCopy / time) * 100 << "%)";
    zklog.trace("        COPY_TRACE:   " + oss.str());
    oss.str("");
    oss.clear();
}

// Function to set the CUDA device based on the MPI rank
// Needs to be evolved to ensuer global balance between mpi ranks and GPU devices
void set_device(uint32_t mpi_rank){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        exit(1);
    }
    int device = mpi_rank % deviceCount;
    cudaSetDevice(device);
    printf("Using CUDA device %d for MPI rank %d\n", device, mpi_rank);
    cudaDeviceSynchronize();
}
#endif