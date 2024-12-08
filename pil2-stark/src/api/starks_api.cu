#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#ifdef __USE_CUDA__
#include "gen_recursive_proof.cuh"

void *gen_recursive_proof(void *pSetupCtx_, char* globalInfoFile, uint64_t airgroupId, void* witness, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, bool vadcop, void* d_buffers_) {

    json globalInfo;
    file2json(globalInfoFile, globalInfo);
    
    DeviceCommitBuffers* d_buffers = (DeviceCommitBuffers*) d_buffers_;
    SetupCtx * setupCtx = (SetupCtx *)pSetupCtx_;
    double time = omp_get_wtime();

    CHECKCUDAERR(cudaSetDevice(0));
    uint64_t size = (1 << setupCtx->starkInfo.starkStruct.nBits) * (setupCtx->starkInfo.mapSectionsN["cm1"])* sizeof(Goldilocks::Element);
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_witness, witness, size, cudaMemcpyHostToDevice));
    time = omp_get_wtime() - time;
    std::cout << "rick genDeviceTrace time: " << time << std::endl;

    time = omp_get_wtime();
    void * proof = genRecursiveProof_gpu<Goldilocks::Element>(*setupCtx, globalInfo, airgroupId, (Goldilocks::Element *)witness, d_buffers->d_witness, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, string(proof_file), d_buffers);
    time = omp_get_wtime() - time;
    std::cout << "rick genRecursiveProof_gpu time: " << time << std::endl;
    return proof;
}

void* genDeviceCommitBuffers(uint64_t maxNExtended, uint64_t maxWitCols, uint64_t maxTraceSize){
    
    CHECKCUDAERR(cudaSetDevice(0));
    DeviceCommitBuffers *buffers = new DeviceCommitBuffers();
    CHECKCUDAERR(cudaMalloc(&buffers->d_witness, maxNExtended * maxWitCols * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_trace, maxTraceSize * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_forwardTwiddleFactors, maxNExtended * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_inverseTwiddleFactors, maxNExtended * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_r, maxNExtended * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_ntt, maxNExtended * maxWitCols * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&buffers->d_tree, maxNExtended * sizeof(uint64_t)));
    return (void *) buffers;
}
#endif