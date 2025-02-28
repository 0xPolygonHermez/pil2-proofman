#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#include "starks_api.hpp"
#ifdef __USE_CUDA__
#include "gen_recursive_proof.cuh"
#include "gen_proof.cuh"

struct MaxSizes
{
    uint64_t maxN;
    uint64_t maxNExtended;
    uint64_t maxTraceArea;
    uint64_t maxNTTArea;
    uint64_t maxConstArea;
    uint64_t maxNPublics;
    uint64_t maxAuxTraceArea;
    uint64_t maxConstTreeSize;
};


void *gen_device_commit_buffers(void *maxSizes_)
{
    MaxSizes *maxSizes = (MaxSizes *)maxSizes_;
    CHECKCUDAERR(cudaSetDevice(0));
    DeviceCommitBuffers *buffers = new DeviceCommitBuffers();
    CHECKCUDAERR(cudaMalloc(&buffers->d_trace, maxSizes->maxTraceArea * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_constPols, maxSizes->maxConstArea * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_constTree, maxSizes->maxConstTreeSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_publicInputs, maxSizes->maxNPublics * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_aux_trace, maxSizes->maxAuxTraceArea * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_forwardTwiddleFactors, maxSizes->maxNExtended * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_inverseTwiddleFactors, maxSizes->maxNExtended * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_r, maxSizes->maxNExtended * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_ntt, maxSizes->maxNTTArea * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_tree, maxSizes->maxNExtended * sizeof(uint64_t)));
    return (void *)buffers;

}
void gen_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, void* pBuffHelper, uint64_t* proofBuffer, char *proofFile) {

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx;

    genProof_gpu(*(SetupCtx *)pSetupCtx, airgroupId, airId, instanceId, *(StepsParams *)params, (Goldilocks::Element *)globalChallenge, (Goldilocks::Element *)pBuffHelper, proofBuffer, string(proofFile));
}

void gen_recursive_proof(void *pSetupCtx_, char *globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *trace, void *aux_trace, void *pConstPols, void *pConstTree, void *pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers_)
{

    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    double time = omp_get_wtime();

    CHECKCUDAERR(cudaSetDevice(0));
    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t sizeWitness = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo);

    CHECKCUDAERR(cudaMemcpy(d_buffers->d_trace, trace, sizeWitness, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_constPols, pConstPols, sizeConstPols, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_constTree, pConstTree, sizeConstTree*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    time = omp_get_wtime() - time;
    std::cout << "rick genDeviceBuffers time: " << time << std::endl;

    time = omp_get_wtime();
    genRecursiveProof_gpu<Goldilocks::Element>(*setupCtx, globalInfo, airgroupId, airId, instanceId, (Goldilocks::Element *)trace, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, proofBuffer, string(proof_file), d_buffers, vadcop);
    time = omp_get_wtime() - time;
    std::cout << "rick genRecursiveProof_gpu time: " << time << std::endl;
}
#endif