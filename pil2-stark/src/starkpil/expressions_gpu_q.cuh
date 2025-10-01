#ifndef EXPRESSIONS_GPU_Q_CUH
#define EXPRESSIONS_GPU_Q_CUH
#include "expressions_ctx.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gpu_timer.cuh"
#include <omp.h>

struct DeviceArgumentsQ
{
    uint64_t N;
    uint64_t NExtended;
    uint64_t nBlocks;
    uint32_t nStages;
    uint32_t nCustomCommits;
    uint32_t bufferCommitSize;
    
    uint64_t xn_offset;
    uint64_t x_offset;
    uint64_t zi_offset;

    uint64_t *mapOffsets;
    uint64_t *mapOffsetsExtended;
    uint64_t *nextStrides;
    uint64_t *nextStridesExtended;
    uint64_t *mapOffsetsCustomFixed;
    uint64_t *mapOffsetsCustomFixedExtended;
    uint64_t *mapSectionsN;
    uint64_t *mapSectionsNCustomFixed;

    // Expressions bin
    uint8_t *ops;
    uint16_t *args;
    Goldilocks::Element *numbers;

    uint8_t *opsConstraints;
    uint16_t *argsConstraints;
    Goldilocks::Element *numbersConstraints;
};

__global__  void computeExpressions_q__(StepsParams *d_params, DeviceArgumentsQ *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, const bool debug, uint64_t challengeId, Goldilocks::Element *d_challengePowers);

__global__  void computeExpressions_q_cyclic__(StepsParams *d_params, DeviceArgumentsQ *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, const bool debug, uint64_t challengeId, Goldilocks::Element *d_challengePowers);
class ExpressionsGPUQ : public ExpressionsCtx
{
public:
    uint32_t nRowsPack;
    uint32_t nBlocks;
    uint64_t bufferCommitSize;
   
    DeviceArgumentsQ *d_deviceArgs;
    DeviceArgumentsQ h_deviceArgs;

    ExpressionsGPUQ(SetupCtx &setupCtx, uint32_t nRowsPack = 128, uint32_t nBlocks = 4096);
    ~ExpressionsGPUQ();

    void calculateExpressions_gpu_q(StepsParams *d_params, Dest dest, uint64_t domainSize, uint64_t challengeId, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, Goldilocks::Element *d_challengePowers, uint64_t& countId, TimerGPU &timer, cudaStream_t stream, bool debug = false);
};
#endif

