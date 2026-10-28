#ifndef STIR_GPU_CUH
#define STIR_GPU_CUH
#include <functional>
#include <cuda_runtime.h>
#include "stir.hpp"
#include "stir_layout.hpp"
class TranscriptGL_GPU; class gl64_t; class TimerGPU;

struct StirGpuIo {
    gl64_t *base;               // aux_trace; every StirLayout offset is relative to it
    StirLayout layout;
    gl64_t *f;                  // f_0 on L_0 (interleaved) on entry; clobbered
    uint64_t *rawQueriesL0;     // t_0 words: round-1 raw indices (the stage trees open here)
    uint64_t *permScratch;      // fri_queries_perm
    uint64_t *nonce, *nonceBlocks, *inputHashNonce;   // grinding scratch (as FRI's)
    gl64_t *queriesBuff;        // query block of T_0; T_i's block is at + i·queryBlockStride
    uint64_t queryBlockStride;  // nQueries · maxProofBuffSize
};

// Runs Construction 5.2 on the device, transcript order identical to STIR::prove. `openStageTrees`
// is called once, right after the round-1 raw indices are in io.rawQueriesL0.
void stirProve_gpu(const StirParams &p, const StirGpuIo &io, TranscriptGL_GPU *transcript, TranscriptGL_GPU *helper,
                   const std::function<void()> &openStageTrees, TimerGPU &timer, cudaStream_t stream);
#endif
