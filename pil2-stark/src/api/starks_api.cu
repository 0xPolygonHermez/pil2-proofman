#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#define __USE_CUDA__
#ifdef __USE_CUDA__
#include "gen_recursive_proof.cuh"

void *gen_recursive_proof(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, void* witness, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file, bool vadcop) {

    json globalInfo;
    file2json(globalInfoFile, globalInfo);
    double time = omp_get_wtime();
    gl64_t * d_witness = genDeviceWitness((Goldilocks::Element *) witness, *(SetupCtx *)pSetupCtx);
    time = omp_get_wtime() - time;
    std::cout << "rick genDeviceTrace time: " << time << std::endl;

    time = omp_get_wtime();
    void * proof = genRecursiveProof_gpu<Goldilocks::Element>(*(SetupCtx *)pSetupCtx, globalInfo, airgroupId, (Goldilocks::Element *)witness,
    d_witness, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, string(proof_file));
    time = omp_get_wtime() - time;
    std::cout << "rick genRecursiveProof_gpu time: " << time << std::endl;

    freeDeviceWitness(d_witness);

    return proof;
}
#endif