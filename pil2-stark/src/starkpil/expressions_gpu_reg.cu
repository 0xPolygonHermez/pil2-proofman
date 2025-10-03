#include "expressions_gpu_reg.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_tooling.cuh"
#include "goldilocks_cubic_extension.cuh"


//Next steps:
// 1. All reads through temporals and operations on temporals (lost)
// 2. Constants in first position of each worp (restore)
// 3. First temporals in shared memory
// 4. Join operations 33 - 31
// 5. Join operations 33 - 11
// 6. Separate cyclic and non-cyclic (equal)
// 7. Separate cyclic and non-cyclic (lost)

extern __shared__ Goldilocks::Element scratchpad[];

__device__ __noinline__ void storePolynomial_reg__(ExpsArguments *d_expsArgs, gl64_t *destVals, uint64_t row);
__device__ __noinline__ Goldilocks::Element*  load_reg__(DeviceArgumentsREG *d_deviceArgs, Goldilocks::Element *value, StepsParams* h_params, Goldilocks::Element** expressions_params, uint16_t* args, uint64_t i_args, uint64_t row, uint64_t dim, bool isCyclic, bool debug);

ExpressionsGPUREG::ExpressionsGPUREG(SetupCtx &setupCtx, uint32_t nRowsPack, uint32_t nBlocks) : ExpressionsCtx(setupCtx), nRowsPack(nRowsPack), nBlocks(nBlocks){
    
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


    CHECKCUDAERR(cudaMalloc(&d_deviceArgs, sizeof(DeviceArgumentsREG)));
    CHECKCUDAERR(cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArgumentsREG), cudaMemcpyHostToDevice));
};

ExpressionsGPUREG::~ExpressionsGPUREG()
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

void ExpressionsGPUREG::calculateExpressions_gpu_reg(StepsParams *d_params, Dest dest, uint64_t domainSize, bool domainExtended, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream, bool debug, bool constraints)
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
        ParserParams &parserParams = constraints 
            ? setupCtx.expressionsBin.constraintsInfoDebug[dest.params[k].expId]
            : setupCtx.expressionsBin.expressionsInfo[dest.params[k].expId];
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

    assert(dest.params.size() == 1 || dest.params.size() == 2);

    DestParamsGPU* h_dest_params = new DestParamsGPU[h_expsArgs.dest_nParams];
    for (uint64_t j = 0; j < h_expsArgs.dest_nParams; ++j){

        ParserParams &parserParams = constraints 
            ? setupCtx.expressionsBin.constraintsInfoDebug[dest.params[j].expId]
            : setupCtx.expressionsBin.expressionsInfo[dest.params[j].expId];
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

    size_t sharedMem = (32 + 6 * nthreads_ ) * sizeof(Goldilocks::Element);

    TimerStartCategoryGPU(timer, EXPRESSIONS);
    computeExpressions_reg__<<<nBlocks_, nThreads_, sharedMem, stream>>>(d_params, d_deviceArgs, d_expsArgs, d_destParams, debug, constraints);
    TimerStopCategoryGPU(timer, EXPRESSIONS);
}

__device__ __forceinline__ Goldilocks::Element* load_reg__(
    const DeviceArgumentsREG* __restrict__ dArgs,
    const ExpsArguments* __restrict__ dExpsArgs,
    Goldilocks::Element* __restrict__ temp,
    const StepsParams* __restrict__ dParams,
    Goldilocks::Element** __restrict__ exprParams,
    const uint16_t type,
    const uint16_t argIdx,
    const uint16_t argOffset,
    const uint64_t row,
    const uint64_t dim,
    const bool isCyclic,
    const bool debug
) {

    const uint32_t r = row + threadIdx.x;
    const uint64_t base = dArgs->bufferCommitSize;
    const uint64_t domainSize = dExpsArgs->domainSize;

    // Fast-path: temporary/intermediate buffers
    if (type == base || type == base + 1) {
        return &exprParams[type][argIdx * blockDim.x];
    }

    // Fast-path: constants
    if (type >= base + 2) {
        return &exprParams[type][argIdx];
    }

    const int64_t stride = dExpsArgs->nextStridesExps[argOffset];
    const uint64_t logicalRow = isCyclic ? (r + stride) % domainSize : (r + stride);

    // ConstPols
    if (type == 0) {
        const Goldilocks::Element* basePtr = dExpsArgs->domainExtended
            ? &dParams->pConstPolsExtendedTreeAddress[2]
            : dParams->pConstPolsAddress;

        const uint64_t pos = logicalRow * dArgs->mapSectionsN[0] + argIdx;
        temp[threadIdx.x] = basePtr[pos];
        return temp;
    }

    // Trace and aux_trace
    if (type >= 1 && type <= 3) {
        const uint64_t offset = dExpsArgs->mapOffsetsExps[type];
        const uint64_t nCols = dArgs->mapSectionsN[type];
        const uint64_t pos = logicalRow * nCols + argIdx;

        if (type == 1 && !dExpsArgs->domainExtended) {
            temp[threadIdx.x] = dParams->trace[pos];
        } else {
            #pragma unroll
            for (uint64_t d = 0; d < dim; d++) {
                temp[threadIdx.x + d * blockDim.x] =
                    dParams->aux_trace[offset + pos + d];
            }
        }
        return temp;
    }

    // Special case: x, x_n, zi
    if (type == 4) {
        return (argIdx == 0)
            ? (dExpsArgs->domainExtended
                ? &dParams->aux_trace[dArgs->x_offset + row]
                : &dParams->aux_trace[dArgs->xn_offset + row])
            : &dParams->aux_trace[dArgs->zi_offset + (argIdx - 1) * domainSize + row];
    }

   
    const uint64_t idx = type - (dArgs->nStages + 4);
    const uint64_t offset = dExpsArgs->mapOffsetsCustomExps[idx];
    const uint64_t nCols = dArgs->mapSectionsNCustomFixed[idx];
    const uint64_t pos = logicalRow * nCols + argIdx;

    temp[threadIdx.x] = dParams->pCustomCommitsFixed[offset + pos];
    return temp;
}

__device__ __noinline__ void storePolynomial_reg__(ExpsArguments *d_expsArgs, Goldilocks::Element *destVals, uint64_t row)
{
    if (d_expsArgs->dest_dim == 1)
    {
        uint64_t offset = d_expsArgs->dest_offset != 0 ? d_expsArgs->dest_offset : 1;
        gl64_gpu::copy_gpu((gl64_t*) &d_expsArgs->dest_gpu[row  * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
    }
    else
    {        
        uint64_t offset = d_expsArgs->dest_offset != 0 ? d_expsArgs->dest_offset : FIELD_EXTENSION;
        gl64_gpu::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
        gl64_gpu::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 1], uint64_t(offset), (gl64_t*)&destVals[blockDim.x], false);
        gl64_gpu::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 2], uint64_t(offset), (gl64_t*)&destVals[2*blockDim.x], false);

    }
}

__global__  void computeExpressions_reg__(StepsParams *d_params, DeviceArgumentsREG *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, const bool debug, const bool constraints)
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
        expressions_params[bufferCommitsSize + 3] = constraints ? d_deviceArgs->numbersConstraints : d_deviceArgs->numbers;
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

            uint8_t *ops = constraints ? &d_deviceArgs->opsConstraints[d_destParams[k].opsOffset] : &d_deviceArgs->ops[d_destParams[k].opsOffset];
            uint16_t *args = constraints ? &d_deviceArgs->argsConstraints[d_destParams[k].argsOffset] : &d_deviceArgs->args[d_destParams[k].argsOffset];
            Goldilocks::Element *valueA =  &scratchpad[32];
            Goldilocks::Element *valueB =  valueA + blockDim.x * FIELD_EXTENSION;

            uint64_t i_args = 0;
            uint64_t nOps = d_destParams[k].nOps;
            for (uint64_t kk = 0; kk < nOps; ++kk)

            {

                switch (ops[kk])
                {
                case 0:
                {
                    // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                    gl64_t* a = (gl64_t*)load_reg__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, isCyclic, debug);
                    gl64_t* b = (gl64_t*)load_reg__(d_deviceArgs, d_expsArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, debug);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize][args[i_args + 1] * blockDim.x]);
                    gl64_gpu::op_gpu( args[i_args], res, a, isConstantA, b, isConstantB);
                    i_args += 8;
                    break;
                }
                case 1:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                    gl64_t* a = (gl64_t*)load_reg__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, debug);
                    gl64_t* b = (gl64_t*)load_reg__(d_deviceArgs, d_expsArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, debug);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    Goldilocks3GPU::op_31_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    i_args += 8;
                    break;
                }
                case 2:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                    gl64_t* a = (gl64_t*)load_reg__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, debug);
                    gl64_t* b = (gl64_t*)load_reg__(d_deviceArgs, d_expsArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, isCyclic, debug);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    Goldilocks3GPU::op_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
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
            if (i_args !=  d_destParams[k].nArgs){
                printf(" %lu consumed args - %lu expected args \n", i_args, d_destParams[k].nArgs);
                assert(0);
            }            
        }
        storePolynomial_reg__(d_expsArgs, destVals, i);
        chunk_idx += gridDim.x;
    }
}
