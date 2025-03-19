#ifndef EXPRESSIONS_GPU_HPP
#define EXPRESSIONS_GPU_HPP
#include "expressions_ctx.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"
#include <omp.h>

// #define _ROW_DEBUG_ 0

//extern __shared__ uint16_t args[];
struct ParserParamsGPU
{
    uint32_t stage;
    uint32_t expId;
    uint32_t nTemp1;
    uint32_t nTemp3;
    uint32_t nOps;
    uint32_t opsOffset;
    uint32_t nArgs;
    uint32_t argsOffset;
    uint32_t constPolsOffset;
    uint32_t cmPolsOffset;
    uint32_t challengesOffset;
    uint32_t publicsOffset;
    uint32_t airgroupValuesOffset;
    uint32_t airValuesOffset;
    uint32_t firstRow;
    uint32_t lastRow;
    uint32_t destDim;
    uint32_t destId;
    bool imPol;
};
struct ParamsGPU
{
    ParserParamsGPU parserParams;
    uint64_t dim;
    uint64_t stage;
    uint64_t stagePos;
    uint64_t polsMapId;
    uint64_t rowOffsetIndex;
    bool inverse = false;
    bool batch = true;
    opType op;
    uint64_t value;
};
struct DestGPU
{
    Goldilocks::Element *dest_gpu = nullptr;
    uint64_t domainSize;
    uint64_t offset = 0;
    uint64_t dim = 1;
    uint32_t nParams;
    ParamsGPU *params;
};
struct DeviceArguments
{
    uint64_t N;
    uint64_t NExtended;
    uint64_t domainSize;
    uint64_t nrowsPack;
    uint64_t nCols;
    uint64_t nOpenings;
    uint64_t ns;
    bool domainExtended;
    uint64_t *nextStrides;
    uint64_t *nColsStages;
    uint64_t *nColsStagesAcc;
    uint64_t *offsetsStages;
    Goldilocks::Element *constPols;
    uint64_t constPolsSize;
    uint64_t *cmPolsInfo;
    uint64_t cmPolsInfoSize;
    Goldilocks::Element *trace;
    Goldilocks::Element *aux_trace;
    uint32_t expType;
    uint64_t boundSize;
    Goldilocks::Element *zi;
    Goldilocks::Element *x_n;
    Goldilocks::Element *x_2ns;
    Goldilocks::Element *xDivXSub;
    // non polnomial arguments
    uint32_t nChallenges;
    Goldilocks::Element *challenges;
    uint32_t nNumbers;
    Goldilocks::Element *numbers;
    uint32_t nPublics;
    Goldilocks::Element *publics;
    uint32_t nEvals;
    Goldilocks::Element *evals;
    uint32_t nAirgroupValues;
    Goldilocks::Element *airgroupValues;
    uint32_t nAirValues;
    Goldilocks::Element *airValues;
    uint32_t nProofValues;
    Goldilocks::Element *proofValues;

    // Dests
    DestGPU *dests;
    uint32_t nDests;
    // Expressions bin
    uint8_t *ops;
    uint32_t nOpsTotal;
    uint16_t *args;
    uint32_t nArgsTotal;
    // buffer
    uint64_t nBlocks;
    uint64_t bufferSize;
    Goldilocks::Element *bufferT_;
    // destVals
    uint64_t destValsSize;
    Goldilocks::Element *destVals;
    // tmps
    uint32_t tmp1Size;
    Goldilocks::Element *tmp1;
    uint32_t tmp3Size;
    Goldilocks::Element *tmp3;
    
    // customCommits
    Goldilocks::Element *customCommits;
    uint64_t customCommitsCols;
};

__device__ __noinline__ void storeOnePolynomial__(DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row, uint32_t idest);
__device__ __noinline__ void copyPolynomial__(DeviceArguments *d_deviceArgs, gl64_t *d_destVals, bool inverse, uint64_t dim, gl64_t *d_tmp);
__device__ __noinline__ void loadPolynomials__(DeviceArguments *d_deviceArgs, uint64_t row, uint32_t iBlock);
__device__ __noinline__ void multiplyPolynomials__(DeviceArguments *deviceArgs, DestGPU &dest, gl64_t *destVals);
__global__ __global__ __launch_bounds__(128) void computeExpressions_(DeviceArguments *d_deviceArgs);

class ExpressionsGPU : public ExpressionsCtx
{
public:
    uint64_t nrowsPack;
    uint32_t nBlocks;
    uint64_t nCols;
    uint32_t nParamsMax;
    uint32_t nTemp1Max;
    uint32_t nTemp3Max;
   
    DeviceArguments *d_deviceArgs;
    DeviceArguments h_deviceArgs;
    std::vector< ParamsGPU *> dest_params; //pointer for cudaFree

    ExpressionsGPU(SetupCtx &setupCtx, ProverHelpers& proverHelpers, uint32_t nParamsMax, uint32_t nTemp1Max, uint32_t nTemp3Max, uint64_t nrowsPack_ = 64, uint32_t nBlocks_ = 256);

    ~ExpressionsGPU();

    void setBufferTInfo(uint64_t domainSize, StepsParams &params,  StepsParams & params_gpu, ParserArgs &parserArgs, std::vector<Dest> &dests);
    
    void calculateExpressions_gpu(StepsParams &params, StepsParams &params_gpu, ParserArgs &parserArgs, std::vector<Dest> dests, uint64_t domainSize);
    
    void freeDeviceArguments();
};
#endif

