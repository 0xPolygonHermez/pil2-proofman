#include "expressions_gpu.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_tooling.cuh"
#include "goldilocks_cubic_extension.cuh"

extern __shared__ Goldilocks::Element scratchpad[];

__device__ __noinline__ void storePolynomial__(ExpsArguments *d_expsArgs, gl64_t *destVals, uint64_t row);
__device__ __noinline__ void multiplyPolynomials__(ExpsArguments *d_expsArgs,  DestParamsGPU *d_destParams, DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row);
__device__ __noinline__ bool caseNoOperations__(StepsParams *h_params, DeviceArguments *d_deviceArgs, Goldilocks::Element *destVals, uint32_t k, uint64_t row);
__device__ __noinline__ void getInversePolinomial__(gl64_t *polynomial, uint64_t dim);

__global__ void computeChallengePowers_(uint64_t dest_nParams, uint64_t challengeId, StepsParams *d_params, Goldilocks::Element *d_challengePowers)
{
    if(threadIdx.x == 0){
        Goldilocks3GPU::Element challenge;
        challenge[0] = (gl64_t)d_params->challenges[challengeId*FIELD_EXTENSION].fe;
        challenge[1] = (gl64_t)d_params->challenges[challengeId*FIELD_EXTENSION + 1].fe;
        challenge[2] = (gl64_t)d_params->challenges[challengeId*FIELD_EXTENSION + 2].fe;
        d_challengePowers[0].fe = uint64_t(1);
        d_challengePowers[1].fe = uint64_t(0);
        d_challengePowers[2].fe = uint64_t(0);
        for(uint64_t i = 1; i < dest_nParams; ++i) {
            Goldilocks3GPU::mul( (Goldilocks3GPU::Element&)d_challengePowers[FIELD_EXTENSION*i], (Goldilocks3GPU::Element&)d_challengePowers[FIELD_EXTENSION*(i-1)], challenge);
        }
    }
}

ExpressionsGPU::ExpressionsGPU(SetupCtx &setupCtx, uint32_t nRowsPack, uint32_t nBlocks) : ExpressionsCtx(setupCtx), nRowsPack(nRowsPack), nBlocks(nBlocks)
{
    
    uint32_t ns = 1 + setupCtx.starkInfo.nStages + 1;
    uint32_t nCustoms = setupCtx.starkInfo.customCommits.size();
    uint32_t nOpenings = setupCtx.starkInfo.openingPoints.size();
    uint32_t nStages_ = setupCtx.starkInfo.nStages;
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    bufferCommitSize = 1 + nStages_ + 3 + nCustoms;

    h_deviceArgs.N = N;
    h_deviceArgs.NExtended = NExtended;
    h_deviceArgs.nBlocks = nBlocks;
    h_deviceArgs.nStages = nStages_;
    h_deviceArgs.nCustomCommits = nCustoms;
    h_deviceArgs.bufferCommitSize = bufferCommitSize;
    
    h_deviceArgs.zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];

    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapOffsets, ns * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapOffsetsExtended, ns * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapOffsetsCustomFixed, nCustoms * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapOffsetsCustomFixedExtended, nCustoms * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.nextStrides, nOpenings * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.nextStridesExtended, nOpenings * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapSectionsN, ns * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapSectionsNCustomFixed, nCustoms * sizeof(uint64_t)));

    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapOffsets, mapOffsets, ns * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapOffsetsExtended, mapOffsetsExtended, ns * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapOffsetsCustomFixed, mapOffsetsCustomFixed, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapOffsetsCustomFixedExtended, mapOffsetsCustomFixedExtended, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.nextStrides, nextStrides, nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.nextStridesExtended, nextStridesExtended, nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapSectionsN, mapSectionsN, ns * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapSectionsNCustomFixed, mapSectionsNCustomFixed, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice));


    ParserArgs parserArgs = setupCtx.expressionsBin.expressionsBinArgsExpressions;
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.numbers, parserArgs.nNumbers * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.numbers, (Goldilocks::Element *)parserArgs.numbers, parserArgs.nNumbers * sizeof(Goldilocks::Element),cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.ops, setupCtx.expressionsBin.nOpsTotal * sizeof(uint8_t)));   
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.args, setupCtx.expressionsBin.nArgsTotal * sizeof(uint16_t))); 
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.ops, parserArgs.ops, setupCtx.expressionsBin.nOpsTotal * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.args, parserArgs.args, setupCtx.expressionsBin.nArgsTotal * sizeof(uint16_t), cudaMemcpyHostToDevice));

    ParserArgs parserArgsConstraints = setupCtx.expressionsBin.expressionsBinArgsConstraints;
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.numbersConstraints, parserArgsConstraints.nNumbers * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.numbersConstraints, (Goldilocks::Element *)parserArgsConstraints.numbers, parserArgsConstraints.nNumbers * sizeof(Goldilocks::Element),cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.opsConstraints, setupCtx.expressionsBin.nOpsDebug * sizeof(uint8_t)));   
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.argsConstraints, setupCtx.expressionsBin.nArgsDebug * sizeof(uint16_t))); 
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.opsConstraints, parserArgsConstraints.ops, setupCtx.expressionsBin.nOpsDebug * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.argsConstraints, parserArgsConstraints.args, setupCtx.expressionsBin.nArgsDebug * sizeof(uint16_t), cudaMemcpyHostToDevice));


    CHECKCUDAERR(cudaMalloc(&d_deviceArgs, sizeof(DeviceArguments)));
    CHECKCUDAERR(cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArguments), cudaMemcpyHostToDevice));
};

ExpressionsGPU::~ExpressionsGPU()
{
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsets));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.nextStrides));
    CHECKCUDAERR(cudaFree(h_deviceArgs.nextStridesExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsCustomFixed));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsCustomFixedExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapSectionsN));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapSectionsNCustomFixed));
    CHECKCUDAERR(cudaFree(h_deviceArgs.numbers));
    CHECKCUDAERR(cudaFree(h_deviceArgs.ops));
    CHECKCUDAERR(cudaFree(h_deviceArgs.args));
    CHECKCUDAERR(cudaFree(h_deviceArgs.numbersConstraints));
    CHECKCUDAERR(cudaFree(h_deviceArgs.opsConstraints));
    CHECKCUDAERR(cudaFree(h_deviceArgs.argsConstraints));

    CHECKCUDAERR(cudaFree(d_deviceArgs));
}

void ExpressionsGPU::calculateExpressions_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, bool domainExtended, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream)
{
    ExpsArguments h_expsArgs;

    uint32_t nrowsPack = std::min(static_cast<uint32_t>(nRowsPack), static_cast<uint32_t>(domainSize));
    h_expsArgs.nRowsPack = nrowsPack;
    
    h_expsArgs.mapOffsetsExps = domainExtended ? h_deviceArgs.mapOffsetsExtended : h_deviceArgs.mapOffsets;            
    h_expsArgs.mapOffsetsCustomExps = domainExtended ? h_deviceArgs.mapOffsetsCustomFixedExtended : h_deviceArgs.mapOffsetsCustomFixed;
    h_expsArgs.nextStridesExps = domainExtended ? h_deviceArgs.nextStridesExtended : h_deviceArgs.nextStrides;

    h_expsArgs.k_min = domainExtended
                             ? uint64_t((minRowExtended + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack
                             : uint64_t((minRow + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;
    h_expsArgs.k_max = domainExtended
                             ? uint64_t(maxRowExtended / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack
                             : uint64_t(maxRow / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;

    h_expsArgs.maxTemp1Size = 0;
    h_expsArgs.maxTemp3Size = 0;

    h_expsArgs.offsetTmp1 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    h_expsArgs.offsetTmp3 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)];
    h_expsArgs.offsetDestVals = setupCtx.starkInfo.mapOffsets[std::make_pair("destVals", false)];

    for (uint64_t k = 0; k < dest.params.size(); ++k)
    {
        ParserParams &parserParams = setupCtx.expressionsBin.expressionsInfo[dest.params[k].expId];
        if (parserParams.nTemp1*h_expsArgs.nRowsPack > h_expsArgs.maxTemp1Size) {
            h_expsArgs.maxTemp1Size = parserParams.nTemp1*h_expsArgs.nRowsPack;
        }
        if (parserParams.nTemp3*h_expsArgs.nRowsPack*FIELD_EXTENSION > h_expsArgs.maxTemp3Size) {
            h_expsArgs.maxTemp3Size = parserParams.nTemp3*h_expsArgs.nRowsPack*FIELD_EXTENSION;
        }
    }

    h_expsArgs.domainSize = domainSize;
    h_expsArgs.domainExtended = domainExtended;

    h_expsArgs.dest_gpu = dest.dest_gpu;
    h_expsArgs.dest_domainSize = dest.domainSize;
    h_expsArgs.dest_stageCols = dest.stageCols;
    h_expsArgs.dest_stagePos = dest.stagePos;
    h_expsArgs.dest_dim = dest.dim;
    h_expsArgs.dest_expr = dest.expr;
    h_expsArgs.dest_nParams = dest.params.size();

    assert(dest.params.size() == 1 || dest.params.size() == 2);

    DestParamsGPU* h_dest_params = new DestParamsGPU[h_expsArgs.dest_nParams];
    for (uint64_t j = 0; j < h_expsArgs.dest_nParams; ++j){

        ParserParams &parserParams = setupCtx.expressionsBin.expressionsInfo[dest.params[j].expId];
        h_dest_params[j].dim = dest.params[j].dim;
        h_dest_params[j].stage = dest.params[j].stage;
        h_dest_params[j].stagePos = dest.params[j].stagePos;
        h_dest_params[j].polsMapId = dest.params[j].polsMapId;
        h_dest_params[j].rowOffsetIndex = dest.params[j].rowOffsetIndex;
        h_dest_params[j].inverse = dest.params[j].inverse;
        h_dest_params[j].op = dest.params[j].op;
        h_dest_params[j].value = dest.params[j].value;
        h_dest_params[j].nOps = parserParams.nOps;
        h_dest_params[j].opsOffset = parserParams.opsOffset;
        h_dest_params[j].nArgs = parserParams.nArgs;
        h_dest_params[j].argsOffset =parserParams.argsOffset;
    }

    memcpy(pinned_exps_params + countId * 2 * sizeof(DestParamsGPU), h_dest_params, h_expsArgs.dest_nParams * sizeof(DestParamsGPU));
    CHECKCUDAERR(cudaMemcpyAsync(d_destParams, pinned_exps_params + countId * 2 * sizeof(DestParamsGPU), h_expsArgs.dest_nParams * sizeof(DestParamsGPU), cudaMemcpyHostToDevice, stream));
    delete[] h_dest_params;

    memcpy(pinned_exps_args + countId * sizeof(ExpsArguments), &h_expsArgs, sizeof(ExpsArguments));
    CHECKCUDAERR(cudaMemcpyAsync(d_expsArgs, pinned_exps_args + countId * sizeof(ExpsArguments), sizeof(ExpsArguments), cudaMemcpyHostToDevice, stream));

    uint32_t nblocks_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(nBlocks),(domainSize + nrowsPack - 1) / nrowsPack));
    uint32_t nthreads_ = nblocks_ == 1 ? domainSize : nrowsPack;
    dim3 nBlocks_ =  nblocks_;
    dim3 nThreads_ = nthreads_;
    
    assert(bufferCommitSize  + 9  < 32);
    size_t sharedMem = 32 * sizeof(Goldilocks::Element);

    TimerStartCategoryGPU(timer, EXPRESSIONS);
    computeExpressions_<<<nBlocks_, nThreads_, sharedMem, stream>>>(d_params, d_deviceArgs, d_expsArgs, d_destParams);
    TimerStopCategoryGPU(timer, EXPRESSIONS);
}

void ExpressionsGPU::calculateExpressionsQ_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, uint64_t challengeId, bool domainExtended, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, Goldilocks::Element *d_challengePowers, uint64_t& countId, TimerGPU &timer, cudaStream_t stream)
{
    ExpsArguments h_expsArgs;

    uint32_t nrowsPack = std::min(static_cast<uint32_t>(nRowsPack), static_cast<uint32_t>(domainSize));
    h_expsArgs.nRowsPack = nrowsPack;
    
    h_expsArgs.mapOffsetsExps = domainExtended ? h_deviceArgs.mapOffsetsExtended : h_deviceArgs.mapOffsets;            
    h_expsArgs.mapOffsetsCustomExps = domainExtended ? h_deviceArgs.mapOffsetsCustomFixedExtended : h_deviceArgs.mapOffsetsCustomFixed;
    h_expsArgs.nextStridesExps = domainExtended ? h_deviceArgs.nextStridesExtended : h_deviceArgs.nextStrides;

    h_expsArgs.k_min = domainExtended
                             ? uint64_t((minRowExtended + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack
                             : uint64_t((minRow + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;
    h_expsArgs.k_max = domainExtended
                             ? uint64_t(maxRowExtended / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack
                             : uint64_t(maxRow / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;

    h_expsArgs.maxTemp1Size = 0;
    h_expsArgs.maxTemp3Size = 0;

    h_expsArgs.offsetTmp1 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    h_expsArgs.offsetTmp3 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)];
    h_expsArgs.offsetDestVals = setupCtx.starkInfo.mapOffsets[std::make_pair("destVals", false)];

    for (uint64_t k = 0; k < dest.params.size(); ++k)
    {
        ParserParams &parserParams = setupCtx.expressionsBin.constraintsInfoDebug[dest.params[k].expId];
        if (parserParams.nTemp1*h_expsArgs.nRowsPack > h_expsArgs.maxTemp1Size) {
            h_expsArgs.maxTemp1Size = parserParams.nTemp1*h_expsArgs.nRowsPack;
        }
        if (parserParams.nTemp3*h_expsArgs.nRowsPack*FIELD_EXTENSION > h_expsArgs.maxTemp3Size) {
            h_expsArgs.maxTemp3Size = parserParams.nTemp3*h_expsArgs.nRowsPack*FIELD_EXTENSION;
        }
    }

    h_expsArgs.domainSize = domainSize;
    h_expsArgs.domainExtended = domainExtended;

    h_expsArgs.dest_gpu = dest.dest_gpu;
    h_expsArgs.dest_domainSize = dest.domainSize;
    h_expsArgs.dest_stageCols = dest.stageCols;
    h_expsArgs.dest_stagePos = dest.stagePos;
    h_expsArgs.dest_dim = dest.dim;
    h_expsArgs.dest_expr = dest.expr;
    h_expsArgs.dest_nParams = dest.params.size();

    DestParamsGPU* h_dest_params = new DestParamsGPU[h_expsArgs.dest_nParams];
    for (uint64_t j = 0; j < h_expsArgs.dest_nParams; ++j){

        ParserParams &parserParams = setupCtx.expressionsBin.constraintsInfoDebug[dest.params[j].expId];
        h_dest_params[j].dim = dest.params[j].dim;
        h_dest_params[j].stage = dest.params[j].stage;
        h_dest_params[j].stagePos = dest.params[j].stagePos;
        h_dest_params[j].polsMapId = dest.params[j].polsMapId;
        h_dest_params[j].rowOffsetIndex = dest.params[j].rowOffsetIndex;
        h_dest_params[j].inverse = dest.params[j].inverse;
        h_dest_params[j].op = dest.params[j].op;
        h_dest_params[j].value = dest.params[j].value;
        h_dest_params[j].nOps = parserParams.nOps;
        h_dest_params[j].opsOffset = parserParams.opsOffset;
        h_dest_params[j].nArgs = parserParams.nArgs;
        h_dest_params[j].argsOffset =parserParams.argsOffset;
    }

    memcpy(pinned_exps_params + countId * 2 * sizeof(DestParamsGPU), h_dest_params, h_expsArgs.dest_nParams * sizeof(DestParamsGPU));
    CHECKCUDAERR(cudaMemcpyAsync(d_destParams, pinned_exps_params + countId * 2 * sizeof(DestParamsGPU), h_expsArgs.dest_nParams * sizeof(DestParamsGPU), cudaMemcpyHostToDevice, stream));
    delete[] h_dest_params;

    memcpy(pinned_exps_args + countId * sizeof(ExpsArguments), &h_expsArgs, sizeof(ExpsArguments));
    CHECKCUDAERR(cudaMemcpyAsync(d_expsArgs, pinned_exps_args + countId * sizeof(ExpsArguments), sizeof(ExpsArguments), cudaMemcpyHostToDevice, stream));

    dim3 nBlocks_ = nBlocks;
    dim3 nThreads_(nrowsPack, 2, 1);
    
    assert(bufferCommitSize  + 9  < 32);
    size_t sharedMem = 32 * sizeof(Goldilocks::Element);

    TimerStartCategoryGPU(timer, EXPRESSIONS);
    computeChallengePowers_<<<1, 1, 0, stream>>>(h_expsArgs.dest_nParams, challengeId, d_params, d_challengePowers);
    computeExpressionQ_<<<nBlocks_, nThreads_, sharedMem, stream>>>(d_params, d_deviceArgs, d_expsArgs, d_destParams, challengeId, d_challengePowers);
    TimerStopCategoryGPU(timer, EXPRESSIONS);
}

__device__ __forceinline__ void load__(
    const DeviceArguments* __restrict__ dArgs,
    const ExpsArguments* __restrict__ dExpsArgs,
    const StepsParams* __restrict__ dParams,
    Goldilocks::Element** __restrict__ exprParams,
    const uint16_t type,
    const uint16_t argIdx,
    const uint16_t argOffset,
    const uint64_t row,
    const uint64_t dim,
    const bool isCyclic,
    gl64_t*& out0,
    gl64_t*& out1,
    gl64_t*& out2,
    const bool constraints
) {

    const uint32_t r = row + threadIdx.x;
    const uint64_t base = dArgs->bufferCommitSize;
    const uint64_t domainSize = dExpsArgs->domainSize;

    // Fast-path: temporary/intermediate buffers
    if (type == base || type == base + 1) {
        if (constraints) {   
            if(dim == 1 ){
                out0 = (gl64_t*)&exprParams[type][threadIdx.y * dExpsArgs->maxTemp1Size + argIdx * blockDim.x + threadIdx.x];
                out1 = nullptr;
                out2 = nullptr;
                return;
            } else {
                out0 =  (gl64_t*)&exprParams[type][threadIdx.y * dExpsArgs->maxTemp1Size + argIdx * blockDim.x + threadIdx.x];
                out1 =  (gl64_t*)&exprParams[type][threadIdx.y * dExpsArgs->maxTemp1Size + argIdx * blockDim.x + threadIdx.x + blockDim.x];
                out2 =  (gl64_t*)&exprParams[type][threadIdx.y * dExpsArgs->maxTemp1Size + argIdx * blockDim.x + threadIdx.x + 2*blockDim.x];
                return;
            }  
        } else {
            if(dim == 1 ){
                out0 = (gl64_t*)&exprParams[type][argIdx * blockDim.x + threadIdx.x];
                out1 = nullptr;
                out2 = nullptr;
                return;
            } else {
                out0 =  (gl64_t*)&exprParams[type][argIdx * blockDim.x + threadIdx.x];
                out1 =  (gl64_t*)&exprParams[type][argIdx * blockDim.x + threadIdx.x + blockDim.x];
                out2 =  (gl64_t*)&exprParams[type][argIdx * blockDim.x + threadIdx.x + 2*blockDim.x];
                return;
            }
        }
    }

    // Fast-path: constants
    if (type >= base + 2) {
        if(dim == 1 ){
            out0 = (gl64_t*)&exprParams[type][argIdx];
            out1 = nullptr;
            out2 = nullptr;
            return;
        } else {
            out0 = (gl64_t*)&exprParams[type][argIdx];
            out1 = (gl64_t*)&exprParams[type][argIdx + 1];
            out2 = (gl64_t*)&exprParams[type][argIdx + 2];
            return;
        }
    }

    const int64_t stride = dExpsArgs->nextStridesExps[argOffset];
    const uint64_t logicalRow = isCyclic ? (r + stride) % domainSize : (r + stride);

    // ConstPols
    if (type == 0) {
        const Goldilocks::Element* basePtr = dExpsArgs->domainExtended
            ? dParams->pConstPolsExtendedTreeAddress
            : dParams->pConstPolsAddress;

        //const uint64_t pos = logicalRow * dArgs->mapSectionsN[0] + argIdx;
        const uint64_t pos = getBufferOffset(logicalRow, argIdx, domainSize, dArgs->mapSectionsN[0]);
        out0 = (gl64_t*)&basePtr[pos];
        out1 = nullptr;
        out2 = nullptr;
        return;
    }

    // Trace and aux_trace
    if (type >= 1 && type <= 3) {
        const uint64_t offset = dExpsArgs->mapOffsetsExps[type];
        const uint64_t nCols = dArgs->mapSectionsN[type];
        const uint64_t pos = getBufferOffset(logicalRow, argIdx, domainSize, nCols);

        if (type == 1 && !dExpsArgs->domainExtended) {
            out0 = (gl64_t*)&dParams->trace[pos];
            out1 = nullptr;
            out2 = nullptr;
            return;
        } else {
            #pragma unroll
            for (uint64_t d = 0; d < dim; d++) {
                const uint64_t pos_ = getBufferOffset(logicalRow, argIdx+d, domainSize, nCols);
                if(d == 0) out0 = (gl64_t*)&dParams->aux_trace[offset + pos_];
                if(d == 1) out1 = (gl64_t*)&dParams->aux_trace[offset + pos_];
                if(d == 2) out2 = (gl64_t*)&dParams->aux_trace[offset + pos_];
            }
            return;
        }
    }

    // Special case: zi
    if (type == 4) {
        //return &dParams->aux_trace[dArgs->zi_offset + (argIdx - 1) * domainSize + row];
        out0 = (gl64_t*)&dParams->aux_trace[dArgs->zi_offset + (argIdx - 1) * domainSize + row + threadIdx.x];
        out1 = nullptr;
        out2 = nullptr;
        return;
    }
    // Custom commits
    const uint64_t idx = type - (dArgs->nStages + 4);
    const uint64_t offset = dExpsArgs->mapOffsetsCustomExps[idx];
    const uint64_t nCols = dArgs->mapSectionsNCustomFixed[idx];
    const uint64_t pos = getBufferOffset(logicalRow, argIdx, domainSize, nCols);

    out0 = (gl64_t*)&dParams->pCustomCommitsFixed[offset + pos];
    out1 = nullptr;
    out2 = nullptr;
    return;
}

__device__ __noinline__ void storePolynomial__(ExpsArguments *d_expsArgs, Goldilocks::Element *destVals, uint64_t row)
{
    #pragma unroll
    for (uint32_t i = 0; i < d_expsArgs->dest_dim; i++) {
        if (!d_expsArgs->dest_expr) {
            uint64_t col = d_expsArgs->dest_stagePos + i;
            uint64_t nRows = d_expsArgs->dest_domainSize;
            uint64_t nCols = d_expsArgs->dest_stageCols;
            uint64_t idx = getBufferOffset(row + threadIdx.x, col, nRows, nCols);
            d_expsArgs->dest_gpu[idx] = destVals[i * blockDim.x + threadIdx.x];
        } else {
            d_expsArgs->dest_gpu[(row + threadIdx.x) * d_expsArgs->dest_dim + i] = destVals[i * blockDim.x + threadIdx.x];
        }
    }
}

__device__ __noinline__ void multiplyPolynomials__(ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row)
{
    if (d_expsArgs->dest_dim == 1)
    {
        gl64_gpu::op_gpu(2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * blockDim.x], false);
    }
    else
    {
        if (d_destParams[0].dim == FIELD_EXTENSION && d_destParams[1].dim == FIELD_EXTENSION)
        {
            Goldilocks3GPU::mul_gpu_no_const(&destVals[0], &destVals[0], &destVals[FIELD_EXTENSION * blockDim.x]);
        }
        else if (d_destParams[0].dim == FIELD_EXTENSION && d_destParams[1].dim == 1)
        {
            Goldilocks3GPU::mul_31_gpu_no_const(&destVals[0], &destVals[0], &destVals[FIELD_EXTENSION * blockDim.x]);
        }
        else
        {
            Goldilocks3GPU::mul_31_gpu_no_const(&destVals[FIELD_EXTENSION * blockDim.x], &destVals[FIELD_EXTENSION * blockDim.x], &destVals[0]);
            destVals[threadIdx.x] = destVals[FIELD_EXTENSION * blockDim.x + threadIdx.x];
            destVals[blockDim.x + threadIdx.x] = destVals[(FIELD_EXTENSION + 1) * blockDim.x + threadIdx.x];
            destVals[2 * blockDim.x + threadIdx.x] = destVals[(FIELD_EXTENSION + 2) * blockDim.x + threadIdx.x];
        }
    }
    storePolynomial__(d_expsArgs, (Goldilocks::Element *)destVals, row);
}

__device__ __noinline__ void getInversePolinomial__(gl64_t *polynomial, uint64_t dim)
{
    int idx = threadIdx.x;
    if (dim == 1)
    {
        polynomial[idx] = polynomial[idx].reciprocal();
    }
    else if (dim == FIELD_EXTENSION)
    {
        Goldilocks3GPU::Element aux;
        aux[0] = polynomial[idx];
        aux[1] = polynomial[blockDim.x + idx];
        aux[2] = polynomial[2 * blockDim.x + idx];
        Goldilocks3GPU::inv(aux, aux);
        polynomial[idx] = aux[0];
        polynomial[blockDim.x + idx] = aux[1];
        polynomial[2 * blockDim.x + idx] = aux[2];
    }
}

__device__ __noinline__ bool caseNoOperations__(StepsParams *d_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *destVals, uint32_t k, uint64_t row)
{

    uint32_t r = row + threadIdx.x;

    if (d_destParams[k].op == opType::cm || d_destParams[k].op == opType::const_)
    { // roger: assumeixes k==0 en aqeusta part?
        uint64_t openingPointIndex = d_destParams[k].rowOffsetIndex;
        uint64_t stagePos = d_destParams[k].stagePos;
        int64_t o = d_expsArgs->nextStridesExps[openingPointIndex];
        uint64_t l = (r + o) % d_expsArgs->domainSize;
        uint64_t nCols = d_deviceArgs->mapSectionsN[0];
        if (d_destParams[k].op == opType::const_)
        {
            uint64_t pos = getBufferOffset(l, stagePos, d_expsArgs->domainSize, nCols);
            destVals[threadIdx.x] = d_params->pConstPolsAddress[pos];
        }
        else
        {
            uint64_t offset = d_expsArgs->mapOffsetsExps[d_destParams[k].stage];
            uint64_t nCols = d_deviceArgs->mapSectionsN[d_destParams[k].stage];
            if (d_destParams[k].stage == 1)
            {
                uint64_t pos = getBufferOffset(l, stagePos, d_expsArgs->domainSize, nCols); 
                destVals[threadIdx.x] = d_params->trace[pos];
            }
            else
            {
                for (uint64_t d = 0; d < d_destParams[k].dim; ++d)
                {
                    uint64_t pos = getBufferOffset(l, stagePos + d, d_expsArgs->domainSize, nCols);
                    destVals[threadIdx.x + d * blockDim.x] = d_params->aux_trace[offset + pos];
                }
            }
        }

        if (d_destParams[k].inverse)
        {
            getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_destParams[k].dim);
        }
        return true;
    }
    else if (d_destParams[k].op == opType::number)
    {
        destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x].fe = d_destParams[k].value;
        return true;
    }
    else if (d_destParams[k].op == opType::airvalue)
    {
        if(d_destParams[k].dim == 1) {
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x] = d_params->airValues[d_destParams[k].polsMapId];
        } else {
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x] = d_params->airValues[d_destParams[k].polsMapId];
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x + blockDim.x] = d_params->airValues[d_destParams[k].polsMapId + 1];
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x + 2 * blockDim.x] = d_params->airValues[d_destParams[k].polsMapId + 2];
        }
        return true;
    }
    return false;
}

__device__ __forceinline__ void op_gpu_p2(uint64_t op, gl64_t *C, const gl64_t *a, const gl64_t *b)
{
    switch (op)
    {
        case 0: C[threadIdx.x] = *a + *b; break;
        case 1: C[threadIdx.x] = *a - *b; break;
        case 2: C[threadIdx.x] = *a * (*b); break;
        case 3: C[threadIdx.x] = *b - *a; break;
    }
}

__device__ __forceinline__ void op_31_gpu_p2(uint64_t op, gl64_t *C, const gl64_t *a0, const gl64_t *a1, const gl64_t *a2, const gl64_t *b ){

    switch (op)
    {
    case 0: {
        C[threadIdx.x] = *a0 + *b;
        C[blockDim.x + threadIdx.x] = *a1;
        C[2 * blockDim.x + threadIdx.x] = *a2;
        break;
    }
    case 1: {
        C[threadIdx.x] = *a0 - *b;
        C[blockDim.x + threadIdx.x] = *a1;
        C[2 * blockDim.x + threadIdx.x] = *a2;
        break;
    }
    case 2: {
        C[threadIdx.x] = *a0 * (*b);
        C[blockDim.x + threadIdx.x] = *a1 * (*b);
        C[2 * blockDim.x + threadIdx.x] = *a2 * (*b);
        break;
    }
    case 3: {
        C[threadIdx.x] = *b - *a0;
        C[blockDim.x + threadIdx.x] = -(*a1);
        C[2 * blockDim.x + threadIdx.x] = -(*a2);
        break;
    }
    }
}

__device__ __forceinline__ void op_33_gpu_p2(uint64_t op, gl64_t *C, const gl64_t *a0, const gl64_t *a1, const gl64_t *a2, const gl64_t *b0, const gl64_t *b1, const gl64_t *b2){
    switch (op)
    {
    case 0: {
            C[threadIdx.x] = (*a0) + (*b0);
            C[blockDim.x + threadIdx.x] = (*a1) + (*b1);
            C[2 * blockDim.x + threadIdx.x] = (*a2) + (*b2);

        break;
    }
    case 1: {

            C[threadIdx.x] = (*a0) - (*b0);
            C[blockDim.x + threadIdx.x] = (*a1) - (*b1);
            C[2 * blockDim.x + threadIdx.x] = (*a2) - (*b2);

        break;
    }
    case 2: {
            gl64_t A_ = ((*a0) + (*a1)) * ((*b0) + (*b1));
            gl64_t B_ = ((*a0) + (*a2)) * ((*b0) + (*b2));
            gl64_t C_ = ((*a1) + (*a2)) * ((*b1) + (*b2));
            gl64_t D_ = (*a0) * (*b0 );
            gl64_t E_ = (*a1) * (*b1);
            gl64_t F_ = (*a2) * (*b2);
            gl64_t G_ = D_ - E_;
            C[threadIdx.x] = (C_ + G_) - F_;
            C[blockDim.x + threadIdx.x] = ((((A_ + C_) - E_) - E_) - D_);
            C[2 * blockDim.x + threadIdx.x] = B_ - G_;
        break;
    }
    case 3: {
            C[threadIdx.x] = (*b0) - (*a0);
            C[blockDim.x + threadIdx.x] =  (*b1) -  (*a1);
            C[2 * blockDim.x + threadIdx.x] = (*b2) -  (*a2);
        break;
    }
    }
}

__global__  void computeExpressions_(StepsParams *d_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams)
{

    int chunk_idx = blockIdx.x;
    uint64_t nchunks = d_expsArgs->domainSize / blockDim.x;

    uint32_t bufferCommitsSize = d_deviceArgs->bufferCommitSize;
    Goldilocks::Element **expressions_params = (Goldilocks::Element **)scratchpad;

    if (threadIdx.x == 0)
    {
        expressions_params[bufferCommitsSize + 0] = (&d_params->aux_trace[d_expsArgs->offsetTmp1 + blockIdx.x * d_expsArgs->maxTemp1Size]);
        expressions_params[bufferCommitsSize + 1] = (&d_params->aux_trace[d_expsArgs->offsetTmp3 + blockIdx.x * d_expsArgs->maxTemp3Size]);
        expressions_params[bufferCommitsSize + 2] = d_params->publicInputs;
        expressions_params[bufferCommitsSize + 3] = d_deviceArgs->numbers;
        expressions_params[bufferCommitsSize + 4] = d_params->airValues;
        expressions_params[bufferCommitsSize + 5] = d_params->proofValues;
        expressions_params[bufferCommitsSize + 6] = d_params->airgroupValues;
        expressions_params[bufferCommitsSize + 7] = d_params->challenges;
        expressions_params[bufferCommitsSize + 8] = d_params->evals;
    }
    __syncthreads();
    Goldilocks::Element *destVals = &(d_params->aux_trace[d_expsArgs->offsetDestVals + blockIdx.x * d_expsArgs->dest_nParams * blockDim.x * FIELD_EXTENSION]); 

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool isCyclic = i < d_expsArgs->k_min || i >= d_expsArgs->k_max;
#pragma unroll 1
        for (uint64_t k = 0; k < d_expsArgs->dest_nParams; ++k)
        {
            if(caseNoOperations__(d_params, d_deviceArgs, d_expsArgs, d_destParams, destVals, k, i)){
                continue;
            }
            uint8_t *ops = &d_deviceArgs->ops[d_destParams[k].opsOffset];
            uint16_t *args = &d_deviceArgs->args[d_destParams[k].argsOffset];
            gl64_t *a0, *a1, *a2, *b0, *b1, *b2;

            uint64_t i_args = 0;
            uint64_t nOps = d_destParams[k].nOps;
            for (uint64_t kk = 0; kk < nOps; ++kk)

            {

                switch (ops[kk])
                {
                case 0:
                {
                    // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, isCyclic, a0, a1, a2, false);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, b0, b1, b2, false);
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize][args[i_args + 1] * blockDim.x]);
                    op_gpu_p2(args[i_args], res, a0, b0);
                    i_args += 8;
                    break;
                }
                case 1:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, a0, a1, a2, false);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, b0, b1, b2, false);
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    op_31_gpu_p2(args[i_args], res, a0, a1, a2, b0);
                    i_args += 8;
                    break;
                }
                case 2:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, a0, a1, a2, false);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, isCyclic, b0, b1, b2, false);
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    op_33_gpu_p2(args[i_args], res, a0, a1, a2, b0, b1, b2);
                    i_args += 8;
                    break;
                }
                default:
                {
                    printf(" Wrong operation! %d \n", ops[kk]);
                }
                }
            }
            if (i_args !=  d_destParams[k].nArgs){
                printf(" %lu consumed args - %lu expected args \n", i_args, d_destParams[k].nArgs);
            }
            if (d_destParams[k].inverse)
            {
                getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_destParams[k].dim);
            }
            
        }

        if (d_expsArgs->dest_nParams == 2)
        {

            multiplyPolynomials__(d_expsArgs, d_destParams, d_deviceArgs, (gl64_t*) destVals, i);
        } else {
            storePolynomial__(d_expsArgs, destVals, i);
        }

        chunk_idx += gridDim.x;
    }

}


__device__ __noinline__ void ziAndstorePolynomial_q__(ExpsArguments *d_expsArgs, Goldilocks::Element *accumulator, Goldilocks::Element* d_zi, uint64_t row)
{
    if (threadIdx.y == 0){
        gl64_t* dest = (gl64_t*) d_expsArgs->dest_gpu;
        gl64_t* acc = (gl64_t*)accumulator;
        
        // Sum accumulators from all threadIdx.y values
        for(uint32_t i = 1; i < blockDim.y; i++){
            gl64_t* acc_other = (gl64_t*)(accumulator + i * blockDim.x * FIELD_EXTENSION);
            acc[threadIdx.x] = acc[threadIdx.x] + acc_other[threadIdx.x];
            acc[threadIdx.x + blockDim.x] = acc[threadIdx.x + blockDim.x] + acc_other[threadIdx.x + blockDim.x];
            acc[threadIdx.x + 2*blockDim.x] = acc[threadIdx.x + 2*blockDim.x] + acc_other[threadIdx.x + 2*blockDim.x];
        }
        
        gl64_t* zi = (gl64_t*)d_zi;
        dest[(row + threadIdx.x) * FIELD_EXTENSION] = acc[threadIdx.x] * zi[row + threadIdx.x];
        dest[(row + threadIdx.x) * FIELD_EXTENSION + 1] = acc[threadIdx.x + blockDim.x] * zi[row + threadIdx.x];
        dest[(row + threadIdx.x) * FIELD_EXTENSION + 2] = acc[threadIdx.x +  2 * blockDim.x] * zi[row + threadIdx.x];
    }
}

__device__ __noinline__ void accumulate_q__(ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, gl64_t *accumulator, gl64_t* tmp, gl64_t* d_challengePowers)
{
    
    // if (d_destParams->dim == 1)
    // {
    //     Goldilocks3GPU::op_31_gpu(2,tmp, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], true, tmp, false);
    //     Goldilocks3GPU::op_gpu(0, accumulator, accumulator, false, tmp, false);
    // }
    // else
    // {   
    //     Goldilocks3GPU::op_gpu(2,tmp, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], true, tmp, false);
    //     Goldilocks3GPU::op_gpu(0, accumulator, accumulator, false, tmp, false);
    // }
}

__global__  void computeExpressionQ_(StepsParams *d_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, uint64_t challengeId, Goldilocks::Element *d_challengePowers)
{

    int chunk_idx = blockIdx.x;
    uint64_t nchunks = d_expsArgs->domainSize / blockDim.x;

    uint32_t bufferCommitsSize = d_deviceArgs->bufferCommitSize;
    Goldilocks::Element **expressions_params = (Goldilocks::Element **)scratchpad;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        expressions_params[bufferCommitsSize + 0] = (&d_params->aux_trace[d_expsArgs->offsetTmp1 + (blockIdx.x * blockDim.y) * d_expsArgs->maxTemp1Size]);
        expressions_params[bufferCommitsSize + 1] = (&d_params->aux_trace[d_expsArgs->offsetTmp3 + (blockIdx.x * blockDim.y) * d_expsArgs->maxTemp3Size]);
        expressions_params[bufferCommitsSize + 2] = d_params->publicInputs;
        expressions_params[bufferCommitsSize + 3] = d_deviceArgs->numbersConstraints;
        expressions_params[bufferCommitsSize + 4] = d_params->airValues;
        expressions_params[bufferCommitsSize + 5] = d_params->proofValues;
        expressions_params[bufferCommitsSize + 6] = d_params->airgroupValues;
        expressions_params[bufferCommitsSize + 7] = d_params->challenges;
        expressions_params[bufferCommitsSize + 8] = d_params->evals;
    }
    __syncthreads();
    Goldilocks::Element *accumulator = &(d_params->aux_trace[d_expsArgs->offsetDestVals + (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x * FIELD_EXTENSION]);
    Goldilocks::Element *d_zi = &d_params->aux_trace[d_deviceArgs->zi_offset];
    Goldilocks::Element *tmp1 = (&d_params->aux_trace[d_expsArgs->offsetTmp1 + (blockIdx.x * blockDim.y + threadIdx.y) * d_expsArgs->maxTemp1Size]);
    Goldilocks::Element *tmp3 = (&d_params->aux_trace[d_expsArgs->offsetTmp3 + (blockIdx.x * blockDim.y + threadIdx.y) * d_expsArgs->maxTemp3Size]);

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool isCyclic = i < d_expsArgs->k_min || i >= d_expsArgs->k_max;
        accumulator[threadIdx.x].fe = uint64_t(0);
        accumulator[threadIdx.x + blockDim.x].fe = uint64_t(0);
        accumulator[threadIdx.x + 2 * blockDim.x].fe = uint64_t(0);

#pragma unroll 1
        for (uint64_t k = threadIdx.y; k < d_expsArgs->dest_nParams; k+=blockDim.y) {
            uint8_t *ops =  &d_deviceArgs->opsConstraints[d_destParams[k].opsOffset];
            uint16_t *args =  &d_deviceArgs->argsConstraints[d_destParams[k].argsOffset];

            gl64_t *a0, *a1, *a2, *b0, *b1, *b2;

            uint64_t i_args = 0;
            uint64_t nOps = d_destParams[k].nOps;
            for (uint64_t kk = 0; kk < nOps; ++kk)
            {

                switch (ops[kk])
                {
                case 0:
                {
                    // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, isCyclic, a0, a1, a2, true);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, b0, b1, b2, true);
                    gl64_t *res = (gl64_t *)&tmp1[args[i_args + 1] * blockDim.x];
                    op_gpu_p2(args[i_args], res, a0, b0);
                    i_args += 8;
                    break;
                }
                case 1:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, a0, a1, a2, true);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, b0, b1, b2, true);
                    gl64_t *res = (gl64_t *)&tmp3[args[i_args + 1] * blockDim.x];
                    op_31_gpu_p2(args[i_args], res, a0, a1, a2, b0);
                    i_args += 8;
                    break;
                }
                case 2:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, a0, a1, a2, true);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, isCyclic, b0, b1, b2, true);
                    gl64_t *res = (gl64_t *)&tmp3[args[i_args + 1] * blockDim.x];
                    op_33_gpu_p2(args[i_args], res, a0, a1, a2, b0, b1, b2);
                    i_args += 8;
                    break;
                }
                default:
                {
                    printf(" Wrong operation! %d \n", ops[kk]);
                }
                }
            }
            if (i_args !=  d_destParams[k].nArgs){
                printf(" %lu consumed args - %lu expected args \n", i_args, d_destParams[k].nArgs);
            }

            gl64_t *res = ops[nOps - 1] == 0 ? (gl64_t*) &expressions_params[bufferCommitsSize][args[i_args - 7] * blockDim.x] : (gl64_t*) &expressions_params[bufferCommitsSize + 1][args[i_args - 7] * blockDim.x];
            accumulate_q__(d_expsArgs, &d_destParams[k], (gl64_t *)accumulator, res, (gl64_t*)d_challengePowers);
        }

         __syncthreads();
        ziAndstorePolynomial_q__(d_expsArgs, accumulator, d_zi, i);
        __syncthreads();

        chunk_idx += gridDim.x;
    }

}