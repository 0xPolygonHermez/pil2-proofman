#include "expressions_gpu.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"

extern __shared__ Goldilocks::Element scratchpad[];

__device__ __noinline__ void storePolynomial__(ExpsArguments *d_expsArgs, gl64_t *destVals, uint64_t row);
__device__ __noinline__ void multiplyPolynomials__(ExpsArguments *d_expsArgs,  DestParamsGPU *d_destParams, gl64_t *destVals, uint64_t row);
__device__ __noinline__ bool caseNoOprations__(StepsParams *h_params, DeviceArguments *d_deviceArgs, Goldilocks::Element *destVals, uint32_t k, uint64_t row);
__device__ __noinline__ void getInversePolinomial__(gl64_t *polynomial, uint64_t dim);
__device__ __noinline__ Goldilocks::Element*  load__(DeviceArguments *d_deviceArgs, Goldilocks::Element *value, StepsParams* h_params, Goldilocks::Element** expressions_params, uint16_t* args, uint64_t i_args, uint64_t row, uint64_t dim, bool isCyclic);

#define DEBUG 0
#define DEBUG_ROW 0
__device__ __forceinline__ void printArguments(Goldilocks::Element *a, uint32_t dimA,  bool constA, Goldilocks::Element *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, uint64_t nOps);
__device__ __forceinline__ void printRes(Goldilocks::Element *res, uint32_t dimRes, int i);

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

    CHECKCUDAERR(cudaFree(d_deviceArgs));
}

void ExpressionsGPU::calculateExpressions_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, bool domainExtended, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, TimerGPU &timer, cudaStream_t stream)
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
    h_expsArgs.dest_offset = dest.offset;
    h_expsArgs.dest_dim = dest.dim;
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

    CHECKCUDAERR(cudaMemcpyAsync(d_destParams, h_dest_params, h_expsArgs.dest_nParams * sizeof(DestParamsGPU), cudaMemcpyHostToDevice, stream));
    delete[] h_dest_params;

    CHECKCUDAERR(cudaMemcpyAsync(d_expsArgs, &h_expsArgs, sizeof(ExpsArguments), cudaMemcpyHostToDevice, stream));

    uint32_t nblocks_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(nBlocks),(domainSize + nrowsPack - 1) / nrowsPack));
    uint32_t nthreads_ = nblocks_ == 1 ? domainSize : nrowsPack;
    dim3 nBlocks_ =  nblocks_;
    dim3 nThreads_ = nthreads_;

    size_t sharedMem = (bufferCommitSize  + 9) * sizeof(Goldilocks::Element *) + 2 * nthreads_ * FIELD_EXTENSION * sizeof(Goldilocks::Element);

    TimerStartCategoryGPU(timer, EXPRESSIONS);
    computeExpressions_<<<nBlocks_, nThreads_, sharedMem, stream>>>(d_params, d_deviceArgs, d_expsArgs, d_destParams);
    TimerStopCategoryGPU(timer, EXPRESSIONS);
}

__device__ __forceinline__ Goldilocks::Element* load__(
    const DeviceArguments* __restrict__ dArgs,
    const ExpsArguments* __restrict__ dExpsArgs,
    Goldilocks::Element* __restrict__ temp,
    const StepsParams* __restrict__ dParams,
    Goldilocks::Element** __restrict__ exprParams,
    const uint16_t type,
    const uint16_t argIdx,
    const uint16_t argOffset,
    const uint64_t row,
    const uint64_t dim,
    const bool isCyclic
) {

#if DEBUG 
    bool print = threadIdx.x == 0 && row == DEBUG_ROW;
#endif

    const uint32_t r = row + threadIdx.x;
    const uint64_t base = dArgs->bufferCommitSize;
    const uint64_t domainSize = dExpsArgs->domainSize;

    // Fast-path: temporary/intermediate buffers
    if (type == base || type == base + 1) {
#if DEBUG
        if(print){ 
            if(type == dArgs->bufferCommitSize) printf("Expression debug tmp1\n");
            if(type == dArgs->bufferCommitSize + 1) printf("Expression debug tmp3\n");
        }
#endif
        return &exprParams[type][argIdx * blockDim.x];
    }

    // Fast-path: constants
    if (type >= base + 2) {
#if DEBUG
        if(print){
            if(type == dArgs->bufferCommitSize + 2 ) printf("Expression debug publicInputs\n");
            if(type == dArgs->bufferCommitSize + 3 ) printf("Expression debug numbers\n");
            if(type == dArgs->bufferCommitSize + 4 ) printf("Expression debug airValues\n");
            if(type == dArgs->bufferCommitSize + 5 ) printf("Expression debug proofValues\n");
            if(type == dArgs->bufferCommitSize + 6 ) printf("Expression debug airgroupValues\n");
            if(type == dArgs->bufferCommitSize + 7 ) printf("Expression debug challenges\n");
            if(type == dArgs->bufferCommitSize + 8 ) printf("Expression debug evals\n");
        }
#endif
        return &exprParams[type][argIdx];
    }

    const int64_t stride = dExpsArgs->nextStridesExps[argOffset];
    const uint64_t logicalRow = isCyclic ? (r + stride) % domainSize : (r + stride);

    // ConstPols
    if (type == 0) {
        const Goldilocks::Element* basePtr = dExpsArgs->domainExtended
            ? &dParams->pConstPolsExtendedTreeAddress[2]
            : dParams->pConstPolsAddress;

#if DEBUG 
            if(print) {
                if(isCyclic) {
                    printf("Expression debug constPols cyclic\n");
                } else {
                    printf("Expression debug constPols\n");   
                }
            }
#endif
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
#if DEBUG
        if(print) {
            if(isCyclic) {
                printf("Expression debug trace cyclic: %lu\n",logicalRow * nCols + argIdx );
            } else {
                printf("Expression debug trace\n");
            }
        }
#endif
            temp[threadIdx.x] = dParams->trace[pos];
        } else {
#if DEBUG
        if(print) {
            if(isCyclic) {
                printf("Expression debug aux_trace cyclic %lu\n", offset + logicalRow * nCols + argIdx);
            } else {
                printf("Expression debug aux_trace\n");
            }
        }
#endif
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
#if DEBUG
            if(print) {
                if(argIdx == 0) {
                    printf("Expression debug x or x_n\n");
                } else {
                    printf("Expression debug zi\n");
                }
            }
#endif
        return (argIdx == 0)
            ? (dExpsArgs->domainExtended
                ? &dParams->aux_trace[dArgs->x_offset + row]
                : &dParams->aux_trace[dArgs->xn_offset + row])
            : &dParams->aux_trace[dArgs->zi_offset + (argIdx - 1) * domainSize + row];
    }

    // xi^-1 = inv(x - x_i)
    if (type == 5) {
#if DEBUG
        if(print) printf("Expression debug xi\n");
#endif
        const gl64_t* xDivX = (gl64_t*)&dParams->xDivXSub[argIdx * FIELD_EXTENSION];
        const gl64_t* x = (gl64_t*)&dParams->aux_trace[dArgs->x_offset + row];
        #if DEBUG
            printArguments((Goldilocks::Element *)x, 1, false, &dParams->xDivXSub[argIdx * FIELD_EXTENSION], 3, true, row, 3, 0, 0);
        #endif
        Goldilocks3GPU::sub_13_gpu_b_const((gl64_t*)temp, x, xDivX);
        getInversePolinomial__((gl64_t*)temp, 3);
        #if DEBUG
            printArguments(temp, 3, false, &dParams->aux_trace[dArgs->x_offset + row], 1, false, row, 2, 0, 0);
        #endif
        Goldilocks3GPU::mul_31_gpu_no_const((gl64_t*)temp, (gl64_t*)temp, x);
        return temp;
    }

    // Custom commits
#if DEBUG
        if(print) {
            if(isCyclic) {
                printf("Expression debug customCommits cyclic\n");
            } else {
                printf("Expression debug customCommits\n");
            }
        }
#endif
    const uint64_t idx = type - (dArgs->nStages + 4);
    const uint64_t offset = dExpsArgs->mapOffsetsCustomExps[idx];
    const uint64_t nCols = dArgs->mapSectionsNCustomFixed[idx];
    const uint64_t pos = logicalRow * nCols + argIdx;

    temp[threadIdx.x] = dParams->pCustomCommitsFixed[offset + pos];
    return temp;
}

__device__ __noinline__ void storePolynomial__(ExpsArguments *d_expsArgs, Goldilocks::Element *destVals, uint64_t row)
{
    if (d_expsArgs->dest_dim == 1)
    {
        uint64_t offset = d_expsArgs->dest_offset != 0 ? d_expsArgs->dest_offset : 1;
        gl64_t::copy_gpu((gl64_t*) &d_expsArgs->dest_gpu[row  * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
    }
    else
    {        
        uint64_t offset = d_expsArgs->dest_offset != 0 ? d_expsArgs->dest_offset : FIELD_EXTENSION;
        gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
        gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 1], uint64_t(offset), (gl64_t*)&destVals[blockDim.x], false);
        gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 2], uint64_t(offset), (gl64_t*)&destVals[2*blockDim.x], false);

    }
}

__device__ __noinline__ void multiplyPolynomials__(ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, gl64_t *destVals, uint64_t row)
{
    if (d_expsArgs->dest_dim == 1)
    {
        gl64_t::op_gpu(2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * blockDim.x], false);
        uint64_t offset = d_expsArgs->dest_offset != 0 ? d_expsArgs->dest_offset : 1;
        gl64_t::copy_gpu((gl64_t*) &d_expsArgs->dest_gpu[row  * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
    }
    else
    {
        uint64_t offset = d_expsArgs->dest_offset != 0 ? d_expsArgs->dest_offset : FIELD_EXTENSION;
        if (d_destParams[0].dim == FIELD_EXTENSION && d_destParams[1].dim == FIELD_EXTENSION)
        {
            Goldilocks3GPU::mul_gpu_no_const(&destVals[0], &destVals[0], &destVals[FIELD_EXTENSION * blockDim.x]);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 1], uint64_t(offset), (gl64_t*)&destVals[blockDim.x], false);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 2], uint64_t(offset), (gl64_t*)&destVals[2*blockDim.x], false);
        }
        else if (d_destParams[0].dim == FIELD_EXTENSION && d_destParams[1].dim == 1)
        {
            Goldilocks3GPU::mul_31_gpu_no_const(&destVals[0], &destVals[0], &destVals[FIELD_EXTENSION * blockDim.x]);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 1], uint64_t(offset), (gl64_t*)&destVals[blockDim.x], false);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 2], uint64_t(offset), (gl64_t*)&destVals[2*blockDim.x], false);
        }
        else
        {
            Goldilocks3GPU::mul_31_gpu_no_const(&destVals[FIELD_EXTENSION * blockDim.x], &destVals[FIELD_EXTENSION * blockDim.x], &destVals[0]);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset], uint64_t(offset), (gl64_t*)&destVals[FIELD_EXTENSION * blockDim.x], false);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 1], uint64_t(offset), (gl64_t*)&destVals[(FIELD_EXTENSION + 1) * blockDim.x], false);
            gl64_t::copy_gpu((gl64_t*)&d_expsArgs->dest_gpu[row * offset + 2], uint64_t(offset), (gl64_t*)&destVals[(FIELD_EXTENSION + 2)*blockDim.x], false);
        }
        
    }
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

#if DEBUG 
    bool print = blockIdx.x == 0 && threadIdx.x == 0 && row == DEBUG_ROW;
#endif

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
#if DEBUG
            if(print) printf("Expression debug constPols\n");
#endif
            destVals[threadIdx.x] = d_params->pConstPolsAddress[l * nCols + stagePos];
        }
        else
        {
            uint64_t offset = d_expsArgs->mapOffsetsExps[d_destParams[k].stage];
            uint64_t nCols = d_deviceArgs->mapSectionsN[d_destParams[k].stage];
            if (d_destParams[k].stage == 1)
            {
#if DEBUG
                if(print) printf("Expression debug trace\n");
#endif
                destVals[threadIdx.x] = d_params->trace[l * nCols + stagePos];
            }
            else
            {
#if DEBUG
                if(print) printf("Expression debug aux_trace\n");
#endif
                for (uint64_t d = 0; d < d_destParams[k].dim; ++d)
                {
                    destVals[threadIdx.x + d * blockDim.x] = d_params->aux_trace[offset + l * nCols + stagePos + d];
                }
            }
        }

        if (d_destParams[k].inverse)
        {
#if DEBUG
            if(print) printf("Expression debug inverse\n");
#endif
            getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_destParams[k].dim);
        }
        return true;
    }
    else if (d_destParams[k].op == opType::number)
    {
#if DEBUG
        if(print) printf("Expression debug number\n");
#endif
        destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x].fe = d_destParams[k].value;
        return true;
    }
    else if (d_destParams[k].op == opType::airvalue)
    {
#if DEBUG
        if(print) printf("Expression debug airvalue\n");
#endif
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

__device__ __forceinline__ void printArguments(Goldilocks::Element *a, uint32_t dimA, bool constA, Goldilocks::Element *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, uint64_t nOps){
    bool print = (threadIdx.x == 0  && i == DEBUG_ROW);
    if(print){
        printf("Expression debug op: %lu of %lu with type %lu\n", op, nOps, op_type);
        if(a!= NULL){
            for(uint32_t i = 0; i < dimA; i++){
                Goldilocks::Element val = constA ? a[i] : a[i*blockDim.x];
                printf("Expression debug a[%d]: %lu (constant %u)\n", i, val.fe % GOLDILOCKS_PRIME, constA);
            }
        }
        if(b!= NULL){
            for(uint32_t i = 0; i < dimB; i++){
                Goldilocks::Element val = constB ? b[i] : b[i*blockDim.x];
                printf("Expression debug b[%d]: %lu (constant %u)\n", i, val.fe % GOLDILOCKS_PRIME, constB);
            }

        }
    }
}

__device__ __forceinline__ void printRes(Goldilocks::Element *res, uint32_t dimRes, int i){
    bool print = threadIdx.x == 0  && i == DEBUG_ROW;
    if(print){
        for(uint32_t i = 0; i < dimRes; i++){
            printf("Expression debug res[%d]: %lu\n", i, res[i*blockDim.x].fe % GOLDILOCKS_PRIME);
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
            Goldilocks::Element *valueA = (Goldilocks::Element *)( expressions_params + bufferCommitsSize + 9);
            Goldilocks::Element *valueB =  valueA + blockDim.x * FIELD_EXTENSION;

            uint64_t i_args = 0;
            uint64_t nOps = d_destParams[k].nOps;
            for (uint64_t kk = 0; kk < nOps; ++kk)

            {

                switch (ops[kk])
                {

                case 0:
                {
                    // COPY dim1 to dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 1], args[i_args + 2], args[i_args + 3], i, 1, isCyclic);
                    bool isConstant = args[i_args + 1] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize][args[i_args] * blockDim.x]);
                    #if DEBUG
                    printArguments((Goldilocks::Element *) a, 1, isConstant, NULL, true, 0, i, 4, kk, nOps);
                    #endif
                    gl64_t::copy_gpu(res, a, isConstant);
                    #if DEBUG
                    printRes((Goldilocks::Element *) res, 1, i);
                    #endif
                    i_args += 4;
                    break;
                }
                case 1:
                {
                    // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, d_expsArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize][args[i_args + 1] * blockDim.x]);
                    #if DEBUG
                    printArguments((Goldilocks::Element *)a, 1, isConstantA, (Goldilocks::Element *)b, 1, isConstantB, i, args[i_args], kk, nOps);
                    #endif
                    gl64_t::op_gpu( args[i_args], res, a, isConstantA, b, isConstantB);
                    #if DEBUG
                    printRes((Goldilocks::Element *) res, 1, i);
                    #endif
                    i_args += 8;
                    break;
                }
                case 2:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, d_expsArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    #if DEBUG
                    printArguments((Goldilocks::Element *)a, 3, isConstantA, (Goldilocks::Element *)b, 1, isConstantB, i, args[i_args], kk, nOps);
                    #endif
                    Goldilocks3GPU::op_31_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    #if DEBUG
                    printRes((Goldilocks::Element *) res, 3, i);
                    #endif
                    i_args += 8;
                    break;
                }
                case 3:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, d_expsArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    #if DEBUG
                    printArguments((Goldilocks::Element *)a, 3, isConstantA, (Goldilocks::Element *)b, 3, isConstantB, i, args[i_args], kk, nOps);
                    #endif
                    Goldilocks3GPU::op_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    #if DEBUG
                    printRes((Goldilocks::Element *) res, 3, i);
                    #endif
                    i_args += 8;
                    break;
                }
                case 4:
                {
                    // COPY dim3 to dim3
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, d_expsArgs, valueA, d_params, expressions_params, args[i_args + 1], args[i_args + 2], args[i_args + 3], i, 3, isCyclic);
                    bool isConstant = args[i_args + 1] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args] * blockDim.x]);
                    Goldilocks3GPU::copy_gpu(res, a, isConstant);
                    #if DEBUG
                    printRes((Goldilocks::Element *) res, 3, i);
                    #endif
                    i_args += 4;
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
            if (d_destParams[k].inverse)
            {
                getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_destParams[k].dim);
            }
            
        }

        if (d_expsArgs->dest_nParams == 2)
        {

            multiplyPolynomials__(d_expsArgs, d_destParams, (gl64_t*) destVals, i);
        } else {
            storePolynomial__(d_expsArgs, destVals, i);
        }

        chunk_idx += gridDim.x;
    }

}
