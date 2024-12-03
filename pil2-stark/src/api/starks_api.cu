#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"

#ifdef __USE_CUDA__
#include "gen_recursive_proof.cuh"

void *gen_recursive_proof(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, void* pAddress, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file) {
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    double time = omp_get_wtime();
    gl64_t * d_pAddress = genDeviceTrace((Goldilocks::Element *) pAddress, *(SetupCtx *)pSetupCtx);
    time = omp_get_wtime() - time;
    std::cout << "rick genDeviceTrace time: " << time << std::endl;

    time = omp_get_wtime();
    void * proof = genRecursiveProof_gpu<Goldilocks::Element>(*(SetupCtx *)pSetupCtx, globalInfo, airgroupId, (Goldilocks::Element *)pAddress,
    d_pAddress, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, string(proof_file));
    time = omp_get_wtime() - time;
    std::cout << "rick genRecursiveProof_gpu time: " << time << std::endl;

    freeDeviceTrace(d_pAddress);

    return proof;
}
#endif