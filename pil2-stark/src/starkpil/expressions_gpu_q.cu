#include "expressions_gpu_q.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_tooling.cuh"
#include "goldilocks_cubic_extension.cuh"

#define COUNTERS 0

extern __shared__ Goldilocks::Element scratchpad[];

__device__ __forceinline__ void op_gpu_(uint64_t op);
__device__ __forceinline__ void op_31_gpu_(uint64_t op);
__device__ __forceinline__ void op_33_gpu_(uint64_t op);

__device__ __noinline__ void ziAndstorePolynomial_q__(ExpsArguments *d_expsArgs, Goldilocks::Element *accumulator, Goldilocks::Element* d_zi, uint64_t row);
__device__ __noinline__ void ziAndstorePolynomial_q_cyclic__(ExpsArguments *d_expsArgs, Goldilocks::Element *accumulator, Goldilocks::Element* d_zi, uint64_t row);
__device__ __noinline__ void accumulate_q__(ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, gl64_t *accumulator, gl64_t* tmp, gl64_t* d_challengePowers, gl64_t* helper, bool print);
__device__ __noinline__ void accumulate_q_cyclic__(ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, gl64_t *accumulator, gl64_t* tmp, gl64_t* d_challengePowers, gl64_t* helper, bool print);

__device__ __noinline__ Goldilocks::Element* load_q__(
    const DeviceArgumentsQ*  dArgs,
    const ExpsArguments*  dExpsArgs,
    Goldilocks::Element*  temp,
    const StepsParams*  dParams,
    Goldilocks::Element**  exprParams,
    const uint16_t type,
    const uint16_t argIdx,
    const uint16_t argOffset,
    const uint64_t row,
    const uint64_t dim,
#if COUNTERS
    const bool debug,
    uint64_t* counter,
#else
    const bool debug, 
#endif
    uint32_t& stride
);
__device__ __noinline__ Goldilocks::Element* load_q_cyclic__(
    const DeviceArgumentsQ*  dArgs,
    const ExpsArguments*  dExpsArgs,
    Goldilocks::Element*  temp1,
    Goldilocks::Element*  temp3,
    const StepsParams*  dParams,
    Goldilocks::Element**  exprParams,
    const uint16_t type,
    const uint16_t argIdx,
    const uint16_t argOffset,
    const uint64_t row,
    const uint64_t dim,
    const bool isCyclic,
#if COUNTERS
    const bool debug,
    uint64_t* counter,
#else
    const bool debug, 
#endif
    uint32_t& stride
);
__global__ void computeChallengePowers_q_(uint64_t dest_nParams, uint64_t challengeId, StepsParams *d_params, Goldilocks::Element *d_challengePowers);

#define DEBUG_ROW 161699
__device__ __noinline__ void printArguments_q(Goldilocks::Element *a, uint32_t dimA,  bool constA, Goldilocks::Element *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, uint64_t nOps, bool debug);
__device__ __noinline__ void printRes_q(Goldilocks::Element *res, uint32_t dimRes, int i, bool debug);

ExpressionsGPUQ::ExpressionsGPUQ(SetupCtx &setupCtx, uint32_t nRowsPack, uint32_t nBlocks) : ExpressionsCtx(setupCtx), nRowsPack(nRowsPack), nBlocks(nBlocks)
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
    
    h_deviceArgs.xn_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x_n", false)];
    h_deviceArgs.x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
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


    CHECKCUDAERR(cudaMalloc(&d_deviceArgs, sizeof(DeviceArgumentsQ)));
    CHECKCUDAERR(cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArgumentsQ), cudaMemcpyHostToDevice));
};

ExpressionsGPUQ::~ExpressionsGPUQ()
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

__global__ void print_q__(ExpsArguments *d_expsArgs)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for(uint32_t i = 0; i < d_expsArgs->domainSize; i++){
            printf("row: %d %llu %llu %llu \n", i, d_expsArgs->dest_gpu[i*3], d_expsArgs->dest_gpu[i*3+1], d_expsArgs->dest_gpu[i*3+2]);
        }
    }
}

void ExpressionsGPUQ::calculateExpressions_gpu_q(StepsParams *d_params, Dest dest, uint64_t domainSize, uint64_t challengeId, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, Goldilocks::Element *d_challengePowers, uint64_t& countId, TimerGPU &timer, cudaStream_t stream, bool debug)
{
    ExpsArguments h_expsArgs;
    bool domainExtended = true;

    uint32_t nrowsPack = 256;//256; //std::min(static_cast<uint32_t>(nRowsPack), static_cast<uint32_t>(domainSize));  
    h_expsArgs.nRowsPack = nrowsPack;
    
    h_expsArgs.mapOffsetsExps = h_deviceArgs.mapOffsetsExtended;            
    h_expsArgs.mapOffsetsCustomExps = h_deviceArgs.mapOffsetsCustomFixedExtended;
    h_expsArgs.nextStridesExps = h_deviceArgs.nextStridesExtended;

    h_expsArgs.k_min = uint64_t((minRowExtended + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;
    h_expsArgs.k_max = uint64_t(maxRowExtended / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;

    h_expsArgs.maxTemp1Size = 0;
    h_expsArgs.maxTemp3Size = 0;

    h_expsArgs.offsetTmp1 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    h_expsArgs.offsetTmp3 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)];
    h_expsArgs.offsetDestVals = setupCtx.starkInfo.mapOffsets[std::make_pair("destVals", false)];

    for (uint64_t k = 0; k < dest.params.size(); ++k)
    {
        ParserParams &parserParams =  setupCtx.expressionsBin.constraintsInfoDebug[dest.params[k].expId];
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
    h_expsArgs.dest_offset = dest.offset;
    h_expsArgs.dest_dim = dest.dim;
    h_expsArgs.dest_nParams = dest.params.size();


    DestParamsGPU* h_dest_params = new DestParamsGPU[h_expsArgs.dest_nParams];
    for (uint64_t j = 0; j < h_expsArgs.dest_nParams; ++j){

        ParserParams &parserParams = setupCtx.expressionsBin.constraintsInfoDebug[dest.params[j].expId];
        h_dest_params[j].dim = dest.params[j].dim;
        h_dest_params[j].expId = dest.params[j].expId;
        h_dest_params[j].nTemp1 = parserParams.nTemp1;
        h_dest_params[j].nTemp3 = parserParams.nTemp3;
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

    memcpy(pinned_exps_params, h_dest_params, h_expsArgs.dest_nParams * sizeof(DestParamsGPU));
    CHECKCUDAERR(cudaMemcpyAsync(d_destParams, pinned_exps_params, h_expsArgs.dest_nParams * sizeof(DestParamsGPU), cudaMemcpyHostToDevice, stream));
    delete[] h_dest_params;

    memcpy(pinned_exps_args + countId * sizeof(ExpsArguments), &h_expsArgs, sizeof(ExpsArguments));
    CHECKCUDAERR(cudaMemcpyAsync(d_expsArgs, pinned_exps_args + countId * sizeof(ExpsArguments), sizeof(ExpsArguments), cudaMemcpyHostToDevice, stream));

    uint32_t nblocks_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(nBlocks),(domainSize + nrowsPack - 1) / nrowsPack));
    //uint32_t nthreads_ = nblocks_ == 1 ? domainSize : nrowsPack;
    //dim3 nBlocks_ =  nblocks_;
    //dim3 nThreads_ = nthreads_;

    dim3 nThreads_(nrowsPack, 2, 1); //max: 512
    dim3 nBlocks_ = 2048;  //max: 2048

    assert(bufferCommitSize + 9 < 32); //rick: to be fixed latter

    size_t sharedMem_cyclic = (bufferCommitSize  + 9) * sizeof(Goldilocks::Element *) + 2 * (nThreads_.x * nThreads_.y) * (FIELD_EXTENSION + 1) * sizeof(Goldilocks::Element);
    size_t sharedMem = 32 * sizeof(Goldilocks::Element) + 2 * (nThreads_.x * nThreads_.y) * FIELD_EXTENSION * sizeof(Goldilocks::Element);

    


    computeChallengePowers_q_<<<1, 1, 0, stream>>>(h_expsArgs.dest_nParams, challengeId, d_params, d_challengePowers);
    computeExpressions_q_cyclic__<<<2, nThreads_, sharedMem_cyclic, stream>>>(d_params, d_deviceArgs, d_expsArgs, d_destParams, debug, challengeId, d_challengePowers);
    TimerStartCategoryGPU(timer, EXPRESSIONS);
    computeExpressions_q__<<<nBlocks_, nThreads_, sharedMem, stream>>>(d_params, d_deviceArgs, d_expsArgs, d_destParams, debug, challengeId, d_challengePowers);
    //print_q__<<<1, 1, 0, stream>>>(d_expsArgs);

    TimerStopCategoryGPU(timer, EXPRESSIONS);
}


__device__ __noinline__ Goldilocks::Element* load_q__(
    const DeviceArgumentsQ*  dArgs,
    const ExpsArguments*  dExpsArgs,
    Goldilocks::Element*  temp,
    const StepsParams*  dParams,
    Goldilocks::Element**  exprParams,
    const uint16_t type,
    const uint16_t argIdx,
    const uint16_t argOffset,
    const uint64_t row,
    const uint64_t dim,
#if COUNTERS
    const bool debug,
    uint64_t* counter,
#else
   const bool debug
#endif
) {
    
    const uint32_t r = row + threadIdx.x;
    const uint64_t base = dArgs->bufferCommitSize;

    if (type == base ) {
        temp[threadIdx.x] = exprParams[type][threadIdx.y * dExpsArgs->maxTemp1Size + argIdx * blockDim.x + threadIdx.x];
        temp[threadIdx.x + blockDim.x].fe = 0;
        temp[threadIdx.x + 2 * blockDim.x].fe = 0;
        return temp;
    }
    if (type == base + 1) {
        temp[threadIdx.x]= exprParams[type][threadIdx.y * dExpsArgs->maxTemp3Size + argIdx * blockDim.x + threadIdx.x];
        temp[threadIdx.x + blockDim.x] = exprParams[type][threadIdx.y * dExpsArgs->maxTemp3Size + argIdx * blockDim.x + threadIdx.x + blockDim.x];
        temp[threadIdx.x + 2 * blockDim.x] = exprParams[type][threadIdx.y * dExpsArgs->maxTemp3Size + argIdx * blockDim.x + threadIdx.x + 2 * blockDim.x];
        return temp;
    }
    if (type >= base + 2) {
        if(dim == 1){
            temp[threadIdx.x] = exprParams[type][argIdx];
            temp[threadIdx.x + blockDim.x].fe = 0;
            temp[threadIdx.x + 2 * blockDim.x].fe = 0;
            return temp;
        }
        if(dim == 3){
            temp[threadIdx.x]= exprParams[type][argIdx];
            temp[threadIdx.x + blockDim.x] = exprParams[type][argIdx+ 1];
            temp[threadIdx.x + 2 * blockDim.x] = exprParams[type][argIdx + 2];
            return temp;
        }
    }

    const int64_t stride_ = dExpsArgs->nextStridesExps[argOffset];
    const uint64_t logicalRow = r + stride_;

    // ConstPols
    if (type == 0) {
        const Goldilocks::Element* basePtr =  &dParams->pConstPolsExtendedTreeAddress[2];
        const uint64_t pos = logicalRow * dArgs->mapSectionsN[0] + (uint64_t)(argIdx);
        temp[threadIdx.x] = basePtr[pos];
        temp[threadIdx.x + blockDim.x].fe = 0;
        temp[threadIdx.x + 2 * blockDim.x].fe = 0;
        return temp;
    }

    // Trace and aux_trace
    if (type >= 1 && type <= 3) {
        const uint64_t offset = dExpsArgs->mapOffsetsExps[type];
        const uint64_t nCols = dArgs->mapSectionsN[type];
        const uint64_t pos = logicalRow * nCols + argIdx;

        if(dim == 1){
            temp[threadIdx.x] = dParams->aux_trace[offset + pos];
            temp[threadIdx.x + blockDim.x].fe = 0;
            temp[threadIdx.x + 2 * blockDim.x].fe = 0;
            return temp;
        } else{
            temp[threadIdx.x]= dParams->aux_trace[offset + pos];
            temp[threadIdx.x + blockDim.x] = dParams->aux_trace[offset + pos + 1];
            temp[threadIdx.x + 2 * blockDim.x] = dParams->aux_trace[offset + pos + 2]; 
            return temp;
        }
    }
    const uint64_t idx = type - (dArgs->nStages + 4);
    const uint64_t offset = dExpsArgs->mapOffsetsCustomExps[idx];
    const uint64_t nCols = dArgs->mapSectionsNCustomFixed[idx];
    const uint64_t pos = logicalRow * nCols + argIdx;
    temp[threadIdx.x] = dParams->pCustomCommitsFixed[offset + pos];
    temp[threadIdx.x + blockDim.x].fe = 0;
    temp[threadIdx.x + 2 * blockDim.x].fe = 0;
    return temp;
}


__device__ __noinline__ Goldilocks::Element* load_q_cyclic__(
    const DeviceArgumentsQ*  dArgs,
    const ExpsArguments*  dExpsArgs,
    Goldilocks::Element*  temp1,
    Goldilocks::Element*  temp3,
    const StepsParams*  dParams,
    Goldilocks::Element**  exprParams,
    const uint16_t type,
    const uint16_t argIdx,
    const uint16_t argOffset,
    const uint64_t row,
    const uint64_t dim,
    const bool isCyclic,
#if COUNTERS
    const bool debug,
    uint64_t* counter,
#else
   const bool debug,
#endif
    uint32_t& stride
) {
    
    const uint32_t r = row + threadIdx.x;
    const uint64_t base = dArgs->bufferCommitSize;
    const uint64_t domainSize = dExpsArgs->domainSize;

    // Fast-path: temporary/intermediate buffers
    if (type == base ) {
        stride = 1;
        return &exprParams[type][threadIdx.y * dExpsArgs->maxTemp1Size + argIdx * blockDim.x];
    }
     if (type == base + 1) {
        stride = FIELD_EXTENSION;
        return &exprParams[type][threadIdx.y * dExpsArgs->maxTemp3Size + argIdx * blockDim.x];
    }
    // Fast-path: constants
    if (type >= base + 2) {
        stride = 0;
        return &exprParams[type][argIdx];
    }

    const int64_t stride_ = dExpsArgs->nextStridesExps[argOffset];
    const uint64_t logicalRow = isCyclic ? (r + stride_) % domainSize : (r + stride_);

    // ConstPols
    if (type == 0) {
        stride = 1;
        const Goldilocks::Element* basePtr =  &dParams->pConstPolsExtendedTreeAddress[2];
        const uint64_t pos = logicalRow * dArgs->mapSectionsN[0] + (uint64_t)(argIdx);
        temp1[threadIdx.x] = basePtr[pos];
        return temp1;
    }

    // Trace and aux_trace
    if (type >= 1 && type <= 3) {
        const uint64_t offset = dExpsArgs->mapOffsetsExps[type];
        const uint64_t nCols = dArgs->mapSectionsN[type];
        const uint64_t pos = logicalRow * nCols + argIdx;

        if(dim == 1){
            stride = 1;
            temp1[threadIdx.x] = dParams->aux_trace[offset + pos];
            return temp1;
        } else{
            stride = FIELD_EXTENSION;
            temp3[threadIdx.x*FIELD_EXTENSION]= dParams->aux_trace[offset + pos];
            temp3[threadIdx.x*FIELD_EXTENSION + 1] = dParams->aux_trace[offset + pos + 1];
            temp3[threadIdx.x*FIELD_EXTENSION + 2] = dParams->aux_trace[offset + pos + 2]; 
            return temp3;
        }
    }
        
    const uint64_t idx = type - (dArgs->nStages + 4);
    const uint64_t offset = dExpsArgs->mapOffsetsCustomExps[idx];
    const uint64_t nCols = dArgs->mapSectionsNCustomFixed[idx];
    const uint64_t pos = logicalRow * nCols + argIdx;
    stride = 1;
    temp1[threadIdx.x] = dParams->pCustomCommitsFixed[offset + pos];
    return temp1;
}

__device__ __noinline__ void ziAndstorePolynomial_q__(ExpsArguments *d_expsArgs, Goldilocks::Element *accumulator, Goldilocks::Element* d_zi, uint64_t row)
{
    /*if (threadIdx.y == 0){
        gl64_t* dest = (gl64_t*) d_expsArgs->dest_gpu;
        gl64_t* acc = (gl64_t*)accumulator;
        
        // Sum accumulators from all threadIdx.y values
       for(uint32_t i = 1; i < blockDim.y; i++){
            gl64_t* acc_other = (gl64_t*)(accumulator + i * blockDim.x * FIELD_EXTENSION);
            acc[threadIdx.x*FIELD_EXTENSION] = acc[threadIdx.x*FIELD_EXTENSION] + acc_other[threadIdx.x*FIELD_EXTENSION];
            acc[threadIdx.x*FIELD_EXTENSION + 1] = acc[threadIdx.x*FIELD_EXTENSION + 1] + acc_other[threadIdx.x*FIELD_EXTENSION + 1];
            acc[threadIdx.x*FIELD_EXTENSION + 2] = acc[threadIdx.x*FIELD_EXTENSION + 2] + acc_other[threadIdx.x*FIELD_EXTENSION + 2];
        }
        
        gl64_t* zi = (gl64_t*)d_zi;
        dest[(row + threadIdx.x) * FIELD_EXTENSION] = acc[threadIdx.x*FIELD_EXTENSION] * zi[row + threadIdx.x];
        dest[(row + threadIdx.x) * FIELD_EXTENSION + 1] = acc[threadIdx.x*FIELD_EXTENSION + 1] * zi[row + threadIdx.x];
        dest[(row + threadIdx.x) * FIELD_EXTENSION + 2] = acc[threadIdx.x*FIELD_EXTENSION + 2] * zi[row + threadIdx.x];
    }*/ //rick
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

__device__ __noinline__ void ziAndstorePolynomial_q_cyclic__(ExpsArguments *d_expsArgs, Goldilocks::Element *accumulator, Goldilocks::Element* d_zi, uint64_t row)
{
    if (threadIdx.y == 0){
        gl64_t* dest = (gl64_t*) d_expsArgs->dest_gpu;
        gl64_t* acc = (gl64_t*)accumulator;
        
        // Sum accumulators from all threadIdx.y values
       for(uint32_t i = 1; i < blockDim.y; i++){
            gl64_t* acc_other = (gl64_t*)(accumulator + i * blockDim.x * FIELD_EXTENSION);
            acc[threadIdx.x*FIELD_EXTENSION] = acc[threadIdx.x*FIELD_EXTENSION] + acc_other[threadIdx.x*FIELD_EXTENSION];
            acc[threadIdx.x*FIELD_EXTENSION + 1] = acc[threadIdx.x*FIELD_EXTENSION + 1] + acc_other[threadIdx.x*FIELD_EXTENSION + 1];
            acc[threadIdx.x*FIELD_EXTENSION + 2] = acc[threadIdx.x*FIELD_EXTENSION + 2] + acc_other[threadIdx.x*FIELD_EXTENSION + 2];
        }
        
        gl64_t* zi = (gl64_t*)d_zi;
        dest[(row + threadIdx.x) * FIELD_EXTENSION] = acc[threadIdx.x*FIELD_EXTENSION] * zi[row + threadIdx.x];
        dest[(row + threadIdx.x) * FIELD_EXTENSION + 1] = acc[threadIdx.x*FIELD_EXTENSION + 1] * zi[row + threadIdx.x];
        dest[(row + threadIdx.x) * FIELD_EXTENSION + 2] = acc[threadIdx.x*FIELD_EXTENSION + 2] * zi[row + threadIdx.x];
    } //rick
    /*if (threadIdx.y == 0){
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
    }*/
}

__device__ __noinline__ void    accumulate_q__(ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, gl64_t *accumulator, gl64_t* tmp, gl64_t* d_challengePowers, gl64_t* helper, bool print)
{
    Goldilocks::Element* challenge_print = (Goldilocks::Element*) d_challengePowers;

    /*if(print){
        printf("factor %lu %lu %lu\n", challenge_print[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION].fe, challenge_print[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION + 1].fe, challenge_print[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION + 2].fe);
    }*/
    
    /*if (d_destParams->dim == 1)
    {
        Goldilocks3GPU::op_31_gpu_stride(2,helper, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], 0, tmp, 1);
        Goldilocks3GPU::op_gpu_stride(0, accumulator, accumulator, FIELD_EXTENSION, helper, FIELD_EXTENSION);
    }
    else
    {   
        Goldilocks3GPU::op_gpu_stride(2,helper, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], 0, tmp, 3);
        Goldilocks3GPU::op_gpu_stride(0, accumulator, accumulator, FIELD_EXTENSION, helper, FIELD_EXTENSION);
    }*/  //rick
    if (d_destParams->dim == 1)
    {
        Goldilocks3GPU::op_31_gpu(2,helper, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], true, tmp, false);
        Goldilocks3GPU::op_gpu(0, accumulator, accumulator, false, helper, false);
    }
    else
    {   
        Goldilocks3GPU::op_gpu(2,helper, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], true, tmp, false);
        Goldilocks3GPU::op_gpu(0, accumulator, accumulator, false, helper, false);
    }

    /*if(print){
        Goldilocks::Element* helper_print = (Goldilocks::Element*) helper;
        printf("value %lu %lu %lu\n", helper_print[0].fe, helper_print[1].fe, helper_print[2].fe);
    }*/
}

__device__ __noinline__ void    accumulate_q_cyclic__(ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, gl64_t *accumulator, gl64_t* tmp, gl64_t* d_challengePowers, gl64_t* helper, bool print)
{
    Goldilocks::Element* challenge_print = (Goldilocks::Element*) d_challengePowers;

    /*if(print){
        printf("factor %lu %lu %lu\n", challenge_print[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION].fe, challenge_print[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION + 1].fe, challenge_print[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION + 2].fe);
    }*/
    
    if (d_destParams->dim == 1)
    {
        Goldilocks3GPU::op_31_gpu_stride(2,helper, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], 0, tmp, 1);
        Goldilocks3GPU::op_gpu_stride(0, accumulator, accumulator, FIELD_EXTENSION, helper, FIELD_EXTENSION);
    }
    else
    {   
        Goldilocks3GPU::op_gpu_stride(2,helper, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], 0, tmp, 3);
        Goldilocks3GPU::op_gpu_stride(0, accumulator, accumulator, FIELD_EXTENSION, helper, FIELD_EXTENSION);
    }  //rick
   /*if (d_destParams->dim == 1)
    {
        Goldilocks3GPU::op_31_gpu(2,helper, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], true, tmp, false);
        Goldilocks3GPU::op_gpu(0, accumulator, accumulator, false, helper, false);
    }
    else
    {   
        Goldilocks3GPU::op_gpu(2,helper, &d_challengePowers[(d_expsArgs->dest_nParams - (d_destParams->expId + 1))*FIELD_EXTENSION], true, tmp, false);
        Goldilocks3GPU::op_gpu(0, accumulator, accumulator, false, helper, false);
    }*/

    /*if(print){
        Goldilocks::Element* helper_print = (Goldilocks::Element*) helper;
        printf("value %lu %lu %lu\n", helper_print[0].fe, helper_print[1].fe, helper_print[2].fe);
    }*/
}

__global__ void computeChallengePowers_q_(uint64_t dest_nParams, uint64_t challengeId, StepsParams *d_params, Goldilocks::Element *d_challengePowers)
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


__global__  void computeExpressions_q__(StepsParams *d_params, DeviceArgumentsQ *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, const bool debug, uint64_t challengeId, Goldilocks::Element *d_challengePowers)
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
    
    // Use direct shared memory allocation instead of pointer arithmetic to avoid races
    Goldilocks::Element *sharedWorkspace = &scratchpad[32];
    Goldilocks::Element *valueA = &sharedWorkspace[threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION)];
    Goldilocks::Element *valueB = valueA + blockDim.x * FIELD_EXTENSION;

    Goldilocks::Element *d_zi = &d_params->aux_trace[d_deviceArgs->zi_offset];
    Goldilocks::Element *tmp1 = (&d_params->aux_trace[d_expsArgs->offsetTmp1 + (blockIdx.x * blockDim.y + threadIdx.y) * d_expsArgs->maxTemp1Size]);
    Goldilocks::Element *tmp3 = (&d_params->aux_trace[d_expsArgs->offsetTmp3 + (blockIdx.x * blockDim.y + threadIdx.y) * d_expsArgs->maxTemp3Size]);


    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool isCyclic = i < d_expsArgs->k_min || i >= d_expsArgs->k_max;
        if(isCyclic){
            chunk_idx += gridDim.x;
            continue;
        }

        //set to zero the accumulator
        accumulator[threadIdx.x].fe = uint64_t(0);
        accumulator[threadIdx.x + blockDim.x].fe = uint64_t(0);
        accumulator[threadIdx.x + 2 * blockDim.x].fe = uint64_t(0); //rick
        
        
#pragma unroll 1
        for (uint64_t k = threadIdx.y; k < d_expsArgs->dest_nParams; k+=blockDim.y)
        {   
            for (uint64_t kk = 0; kk < d_destParams[k].nOps; ++kk)
            {
                uint8_t *ops =  &d_deviceArgs->opsConstraints[d_destParams[k].opsOffset];
                uint16_t *args =  &d_deviceArgs->argsConstraints[d_destParams[k].argsOffset];
                if(ops[kk] == 0) {
                    uint32_t i_args = kk * 8;
                    load_q__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, debug);
                    load_q__(d_deviceArgs, d_expsArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, debug);
                    op_gpu_(args[i_args]);
                    tmp1[args[i_args + 1] * blockDim.x + threadIdx.x].fe = valueA[threadIdx.x].fe;
                    i_args +=8;
                }else{
                    uint32_t i_args = kk * 8;
                    uint32_t dimb2 = ops[kk] == 2 ? 3 : 1;
                    load_q__(d_deviceArgs, d_expsArgs,valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3,  debug);
                    load_q__(d_deviceArgs, d_expsArgs,valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, dimb2,  debug);
                    op_33_gpu_(args[i_args]);
                    tmp3[args[i_args + 1] * blockDim.x +threadIdx.x] = valueA[threadIdx.x]; 
                    tmp3[args[i_args + 1] * blockDim.x +threadIdx.x + blockDim.x] = valueA[threadIdx.x + blockDim.x];
                    tmp3[args[i_args + 1] * blockDim.x +threadIdx.x + 2 * blockDim.x] = valueA[threadIdx.x + 2 * blockDim.x];
                }
            }
           accumulate_q__(d_expsArgs, &d_destParams[k], (gl64_t *)accumulator, (gl64_t *)(valueA), (gl64_t*)d_challengePowers, (gl64_t *)(valueB),  threadIdx.x == 0 && blockIdx.x == 0 && chunk_idx == 0);
        }
        __syncthreads();
        ziAndstorePolynomial_q__(d_expsArgs, accumulator, d_zi, i);
        __syncthreads();
        chunk_idx += gridDim.x;
    }
}


__global__  void computeExpressions_q_cyclic__(StepsParams *d_params, DeviceArgumentsQ *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, const bool debug, uint64_t challengeId, Goldilocks::Element *d_challengePowers)
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
    
    // Use direct shared memory allocation instead of pointer arithmetic to avoid races
    Goldilocks::Element *sharedWorkspace = (Goldilocks::Element *)(expressions_params + bufferCommitsSize + 9);
    Goldilocks::Element *valueA3 = &sharedWorkspace[threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION+1)];
    Goldilocks::Element *valueB3 = valueA3 + blockDim.x * FIELD_EXTENSION;
    Goldilocks::Element *valueA1 = valueB3 + blockDim.x * FIELD_EXTENSION;
    Goldilocks::Element *valueB1 = valueA1 + blockDim.x;

    Goldilocks::Element *d_zi = &d_params->aux_trace[d_deviceArgs->zi_offset];
    Goldilocks::Element *tmp1 = (&d_params->aux_trace[d_expsArgs->offsetTmp1 + (blockIdx.x * blockDim.y + threadIdx.y) * d_expsArgs->maxTemp1Size]);
    Goldilocks::Element *tmp3 = (&d_params->aux_trace[d_expsArgs->offsetTmp3 + (blockIdx.x * blockDim.y + threadIdx.y) * d_expsArgs->maxTemp3Size]);


    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool isCyclic = i < d_expsArgs->k_min || i >= d_expsArgs->k_max;
        if(!isCyclic){
            chunk_idx += gridDim.x;
            continue;
        }

        accumulator[threadIdx.x * FIELD_EXTENSION].fe = uint64_t(0);
        accumulator[threadIdx.x * FIELD_EXTENSION + 1].fe = uint64_t(0);
        accumulator[threadIdx.x * FIELD_EXTENSION + 2].fe = uint64_t(0); //rick
       
        
        
#if COUNTERS
        uint64_t global_counter[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
#endif
        
        
#pragma unroll 1
        for (uint64_t k = threadIdx.y; k < d_expsArgs->dest_nParams; k+=blockDim.y)
        {   
#if COUNTERS
            uint64_t counter[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
#endif
            


            uint8_t *ops =  &d_deviceArgs->opsConstraints[d_destParams[k].opsOffset];
            uint16_t *args =  &d_deviceArgs->argsConstraints[d_destParams[k].argsOffset];

            uint64_t i_args = 0;
            uint64_t nOps = d_destParams[k].nOps;
            for (uint64_t kk = 0; kk < nOps; ++kk)
            {
                switch (ops[kk])
                {
                case 0:
                {
                    // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                    uint32_t stride_a;
                    uint32_t stride_b;
    #if COUNTERS
                    gl64_t* a = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueA1, valueA3, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, isCyclic, debug, counter, stride_a);
                    gl64_t* b = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueB1, valueB3, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, debug, counter, stride_b);
    #else
                    gl64_t* a = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueA1, valueA3, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, isCyclic, debug, stride_a);
                    gl64_t* b = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueB1, valueB3, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, debug, stride_b);
    #endif
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? valueA1 : &(tmp1[args[i_args + 1] * blockDim.x]));
                    //res =  &(tmp1[args[i_args + 1] * blockDim.x]);
                    #if DEBUG
                    printArguments_q((Goldilocks::Element *)a, 1, isConstantA, (Goldilocks::Element *)b, 1, isConstantB, i, args[i_args], kk, nOps, debug);
                    #endif
                    gl64_gpu::op_gpu_stride( args[i_args], res, a, stride_a, b, stride_b);
                    //gl64_gpu::op_gpu( args[i_args], res, a, isConstantA, b, isConstantB);
                    #if DEBUG
                    printRes_q((Goldilocks::Element *) res, 1, i, debug);
                    #endif
                    i_args += 8;
                    break;
                }
                case 1:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                    uint32_t stride_a;
                    uint32_t stride_b;
#if COUNTERS
                    gl64_t* a = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueA1, valueA3, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, debug, counter, stride_a);
                    gl64_t* b = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueB1, valueB3, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, debug, counter, stride_b);
#else
                    gl64_t* a = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueA1, valueA3, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, debug, stride_a);
                    gl64_t* b = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueB1, valueB3, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, debug, stride_b);
#endif  
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? valueA3 : &(tmp3[args[i_args + 1] * blockDim.x]));
                    //res =  &(tmp3[args[i_args + 1] * blockDim.x]);
                    #if DEBUG
                    printArguments_q((Goldilocks::Element *)a, 3, isConstantA, (Goldilocks::Element *)b, 1, isConstantB, i, args[i_args], kk, nOps, debug);
                    #endif
                    Goldilocks3GPU::op_31_gpu_stride(args[i_args], res, a, stride_a, b, stride_b);
                    //Goldilocks3GPU::op_31_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    #if DEBUG
                    printRes_q((Goldilocks::Element *) res, 3, i, debug);
                    #endif
                    i_args += 8;
                    break;
                }
                case 2:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                    uint32_t stride_a;
                    uint32_t stride_b;
#if COUNTERS
                    gl64_t* a = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueA1, valueA3, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, debug, counter, stride_a);
                    gl64_t* b = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueB1, valueB3, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, isCyclic, debug, counter, stride_b);
#else
                    gl64_t* a = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueA1, valueA3, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, debug, stride_a);
                    gl64_t* b = (gl64_t*)load_q_cyclic__(d_deviceArgs, d_expsArgs, valueB1, valueB3, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, isCyclic, debug, stride_b);
#endif  
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? valueA3 : &(tmp3[args[i_args + 1] * blockDim.x]));
                    //res =  &(tmp3[args[i_args + 1] * blockDim.x]);
                    #if DEBUG
                    printArguments_q((Goldilocks::Element *)a, 3, isConstantA, (Goldilocks::Element *)b, 3, isConstantB, i, args[i_args], kk, nOps, debug);
                    #endif
                    //Goldilocks3GPU::op_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    Goldilocks3GPU::op_gpu_stride(args[i_args], res, a, stride_a, b, stride_b);
                    #if DEBUG
                    printRes_q((Goldilocks::Element *) res, 3, i, debug);
                    #endif
                    i_args += 8;
                    break;
                }
                default:
                {
                    printf(" Wrong operation! %d \n", ops[kk]);
                    assert(0);
                }
                }
            }
#if COUNTERS
            for(uint32_t c = 0; c < 14; c++){
                global_counter[c] += counter[c];
            }

            if(threadIdx.x == 0 && blockIdx.x == 0 && chunk_idx == 0 && threadIdx.y == 0){
                printf("res %lu %lu %lu\n", valueA3[threadIdx.x*FIELD_EXTENSION].fe, valueA3[threadIdx.x*FIELD_EXTENSION + 1].fe, valueA3[threadIdx.x*FIELD_EXTENSION + 2].fe);
            }

            if(threadIdx.x == 0 && blockIdx.x == 0 && chunk_idx == 0 && threadIdx.y == 0 && false){
                printf("Counters temp1 %lu tmp1> %lu temp3 %lu temp3> %lu publicInputs %lu numbers %lu airValues %lu proofValues %lu airgroupValues %lu challenges %lu evals %lu constPols %lu aux_trace %lu customCommits %lu\n", counter[0], counter[1], counter[2], counter[3], counter[4], counter[5], counter[6], counter[7], counter[8], counter[9], counter[10], counter[11], counter[12], counter[13]);
                uint64_t totalLoads = counter[0] + counter[1] + counter[2] + counter[3] + counter[4] + counter[5] + counter[6] + counter[7] + counter[8] + counter[9] + counter[10] + counter[11] + counter[12] + counter[13];
                
                // Calculate percentages
                float pct[14];
                pct[0] = (float)counter[0]/totalLoads*100.0f;  // temp1
                pct[1] = (float)counter[1]/totalLoads*100.0f;  // tmp1>
                pct[2] = (float)counter[2]/totalLoads*100.0f;  // temp3
                pct[3] = (float)counter[3]/totalLoads*100.0f;  // temp3>
                pct[4] = (float)counter[4]/totalLoads*100.0f;  // publicInputs
                pct[5] = (float)counter[5]/totalLoads*100.0f;  // numbers
                pct[6] = (float)counter[6]/totalLoads*100.0f;  // airValues
                pct[7] = (float)counter[7]/totalLoads*100.0f;  // proofValues
                pct[8] = (float)counter[8]/totalLoads*100.0f;  // airgroupValues
                pct[9] = (float)counter[9]/totalLoads*100.0f;  // challenges
                pct[10] = (float)counter[10]/totalLoads*100.0f; // evals
                pct[11] = (float)counter[11]/totalLoads*100.0f; // constPols
                pct[12] = (float)counter[12]/totalLoads*100.0f; // aux_trace
                pct[13] = (float)counter[13]/totalLoads*100.0f; // customCommits
                
                // Find top 5 categories (simple approach without full sort)
                const char* names[14] = {"temp1","tmp1>","temp3","temp3>","publicInputs","numbers","airValues","proofValues","airgroupValues","challenges","evals","constPols","aux_trace","customCommits"};
                
                printf("Top memory access patterns:\n");
                
                // Find and print top 5 without sorting entire array
                for(int rank = 0; rank < 5; rank++) {
                    int maxIdx = -1;
                    float maxPct = -1.0f;
                    
                    // Find highest remaining percentage
                    for(int i = 0; i < 14; i++) {
                        if(pct[i] > maxPct) {
                            maxPct = pct[i];
                            maxIdx = i;
                        }
                    }
                    
                    if(maxIdx >= 0 && maxPct > 0.0f) {
                        printf("  %d. %s: %.1f%% (%lu accesses)\n", rank+1, names[maxIdx], maxPct, counter[maxIdx]);
                        pct[maxIdx] = -1.0f; // Mark as used
                    }
                }
                
                printf("Total memory accesses: %lu\n", totalLoads);
            }
#endif
            
            if(ops[nOps-1] == 0){
                accumulate_q_cyclic__(d_expsArgs, &d_destParams[k], (gl64_t *)accumulator, (gl64_t *)(valueA1), (gl64_t*)d_challengePowers, (gl64_t *)(valueB3),  threadIdx.x == 0 && blockIdx.x == 0 && chunk_idx == 0); 
            }else{

                accumulate_q_cyclic__(d_expsArgs, &d_destParams[k], (gl64_t *)accumulator, (gl64_t *)(valueA3), (gl64_t*)d_challengePowers, (gl64_t *)(valueB3),  threadIdx.x == 0 && blockIdx.x == 0 && chunk_idx == 0);
            } 
        }
#if COUNTERS
        if(threadIdx.x == 0 && blockIdx.x == 0 && chunk_idx == 0 && threadIdx.y == 0 && false){
            printf("Global Counters temp1 %lu tmp1> %lu temp3 %lu temp3> %lu publicInputs %lu numbers %lu airValues %lu proofValues %lu airgroupValues %lu challenges %lu evals %lu constPols %lu aux_trace %lu customCommits %lu\n", global_counter[0], global_counter[1], global_counter[2], global_counter[3], global_counter[4], global_counter[5], global_counter[6], global_counter[7], global_counter[8], global_counter[9], global_counter[10], global_counter[11], global_counter[12], global_counter[13]);
            // percentages of top 10 counters
            uint64_t totalLoads = global_counter[0] + global_counter[1] + global_counter[2] + global_counter[3] + global_counter[4] + global_counter[5] + global_counter[6] + global_counter[7] + global_counter[8] + global_counter[9] + global_counter[10] + global_counter[11] + global_counter[12] + global_counter[13];
            float pct[14];
            pct[0]      = (float)global_counter[0]/totalLoads*100.0f;  // temp1
            pct[1]      = (float)global_counter[1]/totalLoads*100.0f;  // tmp1>
            pct[2]      = (float)global_counter[2]/totalLoads*100.0f;  // temp3
            pct[3]      = (float)global_counter[3]/totalLoads*100.0f;  // temp3>
            pct[4]      = (float)global_counter[4]/totalLoads*100.0f;  // publicInputs
            pct[5]      = (float)global_counter[5]/totalLoads*100.0f;  // numbers
            pct[6]      = (float)global_counter[6]/totalLoads*100.0f;  // airValues
            pct[7]      = (float)global_counter[7]/totalLoads*100.0f;  // proofValues
            pct[8]      = (float)global_counter[8]/totalLoads*100.0f;  // airgroupValues
            pct[9]      = (float)global_counter[9]/totalLoads*100.0f;  // challenges
            pct[10]     = (float)global_counter[10]/totalLoads*100.0f; // evals
            pct[11]     = (float)global_counter[11]/totalLoads*100.0f; // constPols
            pct[12]     = (float)global_counter[12]/totalLoads*100.0f; // aux_trace
            pct[13]     = (float)global_counter[13]/totalLoads*100.0f; // customCommits 
            const char* names[14] = {"temp1","tmp1>","temp3","temp3>","publicInputs","numbers","airValues","proofValues","airgroupValues","challenges","evals","constPols","aux_trace","customCommits"};
            printf("Top memory access patterns:\n");
            // Find and print top 14 without sorting entire array
            float accumulatedPct = 0.0f;
            for(int rank = 0; rank < 14; rank++) {
                int maxIdx = -1;
                float maxPct = -1.0f;
                // Find highest remaining percentage
                for(int i = 0; i < 14; i++) {
                    if(pct[i] > maxPct) {
                        maxPct = pct[i];
                        maxIdx = i;
                    }
                }
                accumulatedPct += maxPct;
                if(maxIdx >= 0 && maxPct > 0.0f) {
                    printf("  %d. %s: %.1f%% (%lu accesses), accumulated: %.1f%%\n", rank+1, names[maxIdx], maxPct, global_counter[maxIdx], accumulatedPct);
                    pct[maxIdx] = -1.0f; // Mark as used
                }
            }
            printf("Total memory accesses: %lu\n", totalLoads);
        }
#endif
        __syncthreads();
        ziAndstorePolynomial_q_cyclic__(d_expsArgs, accumulator, d_zi, i);
        __syncthreads();
        chunk_idx += gridDim.x;
    }

}

__device__ __forceinline__ void op_gpu_(uint64_t op)
{
    gl64_t& valA = ((gl64_t*)scratchpad)[32 + threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION) + threadIdx.x];
    gl64_t& valB = ((gl64_t*)scratchpad)[32 + threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION) + blockDim.x * FIELD_EXTENSION + threadIdx.x];

    switch (op)
    {
        case 0: valA = valA + valB; break;
        case 1: valA = valA - valB; break;
        case 2: valA = valA * valB; break;
        case 3: valA = valB - valA; break;
    }
}

__device__ __forceinline__ void op_31_gpu_(uint64_t op)
{
    gl64_t* valA = ((gl64_t*) &(scratchpad)[32 + threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION)]);
    gl64_t* valB = ((gl64_t*) &(scratchpad)[32 + threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION) + blockDim.x * FIELD_EXTENSION]);
    switch (op)
    {
    case 0:
        valA[threadIdx.x] = valA[threadIdx.x] + valB[threadIdx.x];
        break;
    case 1:
        valA[threadIdx.x] = valA[threadIdx.x] - valB[threadIdx.x];
        break;
    case 2:
        valA[threadIdx.x] = valA[threadIdx.x] * valB[threadIdx.x];
        valA[blockDim.x + threadIdx.x] = valA[blockDim.x + threadIdx.x] * valB[threadIdx.x];
        valA[2 * blockDim.x + threadIdx.x] = valA[2 * blockDim.x + threadIdx.x] * valB[threadIdx.x];
        break;
    case 3:
        valA[threadIdx.x] = valB[threadIdx.x] - valA[threadIdx.x];
        valA[blockDim.x + threadIdx.x] = -valA[blockDim.x + threadIdx.x];
        valA[2 * blockDim.x + threadIdx.x] = -valA[2 * blockDim.x + threadIdx.x];
        break;
    }
};

__device__ __forceinline__ void op_33_gpu_(uint64_t op)
{
    gl64_t* valA = ((gl64_t*) &(scratchpad)[32 + threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION)]);
    gl64_t* valB = ((gl64_t*) &(scratchpad)[32 + threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION) + blockDim.x * FIELD_EXTENSION]);
    Goldilocks::Element* valB_ = ((Goldilocks::Element*) &(scratchpad)[32 + threadIdx.y * 2 * blockDim.x * (FIELD_EXTENSION) + blockDim.x * FIELD_EXTENSION]);
    switch (op)
    {
    case 0:
        if( valB_[blockDim.x + threadIdx.x].fe==0 || valB_[2 * blockDim.x + threadIdx.x].fe==0){
            valA[threadIdx.x] = valA[threadIdx.x] + valB[threadIdx.x];
        } else{
            valA[threadIdx.x] = valA[threadIdx.x] + valB[threadIdx.x];
            valA[blockDim.x + threadIdx.x] = valA[blockDim.x + threadIdx.x] + valB[blockDim.x + threadIdx.x];
            valA[2 * blockDim.x + threadIdx.x] = valA[2 * blockDim.x + threadIdx.x] + valB[2 * blockDim.x + threadIdx.x];
        }
        break;
    case 1:
        if( valB_[blockDim.x + threadIdx.x].fe==0 || valB_[2 * blockDim.x + threadIdx.x].fe==0){
            valA[threadIdx.x] = valA[threadIdx.x] - valB[threadIdx.x];
        } else{
            valA[threadIdx.x] = valA[threadIdx.x] - valB[threadIdx.x];
            valA[blockDim.x + threadIdx.x] =  valA[blockDim.x + threadIdx.x] -  valB[blockDim.x + threadIdx.x];
            valA[2 * blockDim.x + threadIdx.x] = valA[2 * blockDim.x + threadIdx.x] -  valB[2 * blockDim.x + threadIdx.x];
        }
        break;
    case 2:
        if( valB_[blockDim.x + threadIdx.x].fe==0 || valB_[2 * blockDim.x + threadIdx.x].fe==0){
            valA[threadIdx.x] = valA[threadIdx.x] * valB[threadIdx.x];
            valA[blockDim.x + threadIdx.x] = valA[blockDim.x + threadIdx.x] * valB[threadIdx.x];
            valA[2 * blockDim.x + threadIdx.x] = valA[2 * blockDim.x + threadIdx.x] * valB[threadIdx.x];
            break;
        }else{
            gl64_t A, B, C, D, E, F, G;
            A = (valA[threadIdx.x] + valA[blockDim.x + threadIdx.x]) * (valB[threadIdx.x] + valB[blockDim.x + threadIdx.x]);
            B = (valA[threadIdx.x] + valA[2 * blockDim.x + threadIdx.x]) * (valB[threadIdx.x] + valB[2 * blockDim.x + threadIdx.x]);
            C = (valA[blockDim.x + threadIdx.x] + valA[2 * blockDim.x + threadIdx.x]) * (valB[blockDim.x + threadIdx.x] + valB[2 * blockDim.x + threadIdx.x]);
            D = valA[threadIdx.x] * valB[threadIdx.x];
            E = valA[blockDim.x + threadIdx.x] * valB[blockDim.x + threadIdx.x];
            F = valA[2 * blockDim.x + threadIdx.x] * valB[2 * blockDim.x + threadIdx.x];
            G = D - E;
            valA[threadIdx.x] = (C + G) - F;
            valA[blockDim.x + threadIdx.x] = ((((A + C) - E) - E) - D);
            valA[2 * blockDim.x + threadIdx.x] = B - G;
        }
        break;
    case 3:
        if( valB_[blockDim.x + threadIdx.x].fe==0 || valB_[2 * blockDim.x + threadIdx.x].fe==0){
            valA[threadIdx.x] = valB[threadIdx.x] - valA[threadIdx.x];
            valA[blockDim.x + threadIdx.x] = -valA[blockDim.x + threadIdx.x];
            valA[2 * blockDim.x + threadIdx.x] = -valA[2 * blockDim.x + threadIdx.x];
        } else{
            valA[threadIdx.x] = valB[threadIdx.x] - valA[threadIdx.x];
            valA[blockDim.x + threadIdx.x] =  valB[blockDim.x + threadIdx.x] -  valA[blockDim.x + threadIdx.x];
            valA[2 * blockDim.x + threadIdx.x] = valB[2 * blockDim.x + threadIdx.x] -  valA[2 * blockDim.x + threadIdx.x];
        }
        break;
    }
};