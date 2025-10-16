#ifndef EXPRESSIONS_GPU_HPP
#define EXPRESSIONS_GPU_HPP
#include "expressions_ctx.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gpu_timer.cuh"
#include "data_layout.cuh"
#include <omp.h>

struct DeviceArguments
{
    uint64_t N;
    uint64_t NExtended;
    uint64_t nBlocks;
    uint32_t nStages;
    uint32_t nCustomCommits;
    uint32_t bufferCommitSize;
    
    uint64_t zi_offset;

    uint64_t *mapOffsets;  //rick: passar a uint32_t
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

__global__  void computeExpressions_(StepsParams *h_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams);
__global__  void computeExpressionQ_(StepsParams *h_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, uint64_t challengeId, Goldilocks::Element *d_challengePowers);

class ExpressionsGPU : public ExpressionsCtx
{
public:
    uint32_t nRowsPack;
    uint32_t nBlocks;
    uint64_t bufferCommitSize;
   
    DeviceArguments *d_deviceArgs;
    DeviceArguments h_deviceArgs;

    ExpressionsGPU(SetupCtx &setupCtx, uint32_t nRowsPack = 128, uint32_t nBlocks = 4096);
    ~ExpressionsGPU();

    void calculateExpressions_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, bool domainExtended, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream);
      void calculateExpressionsQ_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, uint64_t challengeId, bool domainExtended, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, Goldilocks::Element *d_challengesPowers, uint64_t& countId, TimerGPU &timer, cudaStream_t stream);
};
#endif

