#ifndef EXPRESSIONS_GPU_HPP
#define EXPRESSIONS_GPU_HPP
#include "expressions_ctx.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"
#include "gpu_timer.cuh"
#include "goldilocks_cubic_extension.cuh"
#include <omp.h>


struct DestParamsGPU
{
    uint64_t dim; 
    uint64_t stage; 
    uint64_t stagePos; 
    uint64_t polsMapId; 
    uint64_t rowOffsetIndex; 
    bool inverse = false; 
    opType op; 
    uint64_t value; 
    uint64_t nOps;
    uint64_t opsOffset; 
    uint64_t nArgs;
    uint64_t argsOffset; 
};


struct DeviceArguments
{
    uint64_t N;
    uint64_t NExtended;
    uint64_t domainSize;
    bool domainExtended;
    uint64_t nRowsPack;
    uint64_t nBlocks;
    uint64_t k_min;
    uint64_t k_max;
    uint32_t maxTemp1Size;
    uint32_t maxTemp3Size;
    uint32_t nStages;
    uint32_t nCustomCommits;
    uint32_t bufferCommitSize;
    uint64_t offsetTmp1;
    uint64_t offsetTmp3;
    uint64_t offsetDestVals;
    
    uint64_t xn_offset;
    uint64_t x_offset;
    uint64_t zi_offset;

    uint64_t *mapOffsets;  //rick: passar a uint32_t
    uint64_t *mapOffsetsExtended;
    uint64_t *mapOffsetsExps;
    uint64_t *nextStrides;
    uint64_t *nextStridesExtended;
    uint64_t *nextStridesExps;
    uint64_t *mapOffsetsCustomFixed;
    uint64_t *mapOffsetsCustomFixedExtended;
    uint64_t *mapOffsetsCustomExps;
    uint64_t *mapSectionsN;
    uint64_t *mapSectionsNCustomFixed;

    Goldilocks::Element *numbers;

    // Dest
    Goldilocks::Element *dest_gpu = nullptr;
    uint64_t dest_domainSize;
    uint64_t dest_offset = 0;
    uint64_t dest_dim = 1;
    uint32_t dest_nParams;

    // Expressions bin
    uint8_t *ops;
    uint16_t *args;
};

__device__ __noinline__ void storePolynomial__(DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row);
__device__ __noinline__ void multiplyPolynomials__(DeviceArguments *deviceArgs, gl64_t *destVals, uint64_t row);
__device__ __noinline__ bool caseNoOprations__(StepsParams *h_params, DeviceArguments *d_deviceArgs, Goldilocks::Element *destVals, uint32_t k, uint64_t row);
__device__ __noinline__ void getInversePolinomial__(gl64_t *polynomial, uint64_t dim);
__device__ __noinline__ Goldilocks::Element*  load__(DeviceArguments *d_deviceArgs, Goldilocks::Element *value, StepsParams* h_params, Goldilocks::Element** expressions_params, uint16_t* args, uint64_t i_args, uint64_t row, uint64_t dim, bool isCyclic);
__global__  void computeExpressions_(StepsParams *h_params, DeviceArguments *d_deviceArgs, DestParamsGPU *d_destParams);

class ExpressionsGPU : public ExpressionsCtx
{
public:
    uint32_t nRowsPack;
    uint32_t nBlocks;
   
    DeviceArguments *d_deviceArgs;
    DeviceArguments h_deviceArgs;

    DestParamsGPU *d_dest_params;
    DestParamsGPU *h_dest_params;

    ExpressionsGPU(SetupCtx &setupCtx, uint32_t nRowsPack = 128, uint32_t nBlocks = 4096);
    ~ExpressionsGPU();

    void loadDeviceArgs(uint64_t domainSize, Dest &dest, cudaStream_t stream);
    void calculateExpressions_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, bool domainExtended, TimerGPU &timer, cudaStream_t stream);
    
};
#endif

