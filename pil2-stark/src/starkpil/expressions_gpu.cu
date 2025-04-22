#include "expressions_gpu.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"

extern __shared__ Goldilocks::Element scratchpad[];

#define DEBUG 0
#define DEBUG_ROW 0
__device__ __forceinline__ void printArguments(Goldilocks::Element *a, uint32_t dimA,  bool constA, Goldilocks::Element *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, uint64_t nOps);
__device__ __forceinline__ void printRes(Goldilocks::Element *res, uint32_t dimRes, int i);

ExpressionsGPU::ExpressionsGPU(SetupCtx &setupCtx, ProverHelpers &proverHelpers, uint32_t nRowsPack_) : ExpressionsCtx(setupCtx, proverHelpers), nRowsPack(nRowsPack_)
{
    
    uint32_t ns = 1 + setupCtx.starkInfo.nStages + 1;
    uint32_t nCustoms = setupCtx.starkInfo.customCommits.size();
    uint32_t nOpenings = setupCtx.starkInfo.openingPoints.size();
    uint32_t nStages_ = setupCtx.starkInfo.nStages;
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    h_deviceArgs.N = N;
    h_deviceArgs.NExtended = NExtended;
    h_deviceArgs.nStages = nStages_;
    h_deviceArgs.nCustomCommits = nCustoms;
    h_deviceArgs.bufferCommitSize = 1 + nStages_ + 3 + nCustoms;
    
    h_deviceArgs.cExpId = setupCtx.starkInfo.cExpId;
    h_deviceArgs.recursive = setupCtx.starkInfo.recursive;

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
    CHECKCUDAERR(cudaFree(h_deviceArgs.numbersConstraints));
    CHECKCUDAERR(cudaFree(h_deviceArgs.ops));
    CHECKCUDAERR(cudaFree(h_deviceArgs.opsConstraints));
    CHECKCUDAERR(cudaFree(h_deviceArgs.args));
    CHECKCUDAERR(cudaFree(h_deviceArgs.argsConstraints));
}

void ExpressionsGPU::loadDeviceArgs(uint64_t domainSize, Dest &dest)
{

    bool domainExtended = domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false;

    h_deviceArgs.nRowsPack = std::min(static_cast<uint64_t>(nRowsPack), domainSize);
    
    h_deviceArgs.mapOffsetsExps = domainExtended ? h_deviceArgs.mapOffsetsExtended : h_deviceArgs.mapOffsets;            
    h_deviceArgs.mapOffsetsCustomExps = domainExtended ? h_deviceArgs.mapOffsetsCustomFixedExtended : h_deviceArgs.mapOffsetsCustomFixed;
    h_deviceArgs.nextStridesExps = domainExtended ? h_deviceArgs.nextStridesExtended : h_deviceArgs.nextStrides;

    h_deviceArgs.k_min = domainExtended
                             ? uint64_t((minRowExtended + h_deviceArgs.nRowsPack - 1) / h_deviceArgs.nRowsPack) * h_deviceArgs.nRowsPack
                             : uint64_t((minRow + h_deviceArgs.nRowsPack - 1) / h_deviceArgs.nRowsPack) * h_deviceArgs.nRowsPack;
    h_deviceArgs.k_max = domainExtended
                             ? uint64_t(maxRowExtended / h_deviceArgs.nRowsPack) * h_deviceArgs.nRowsPack
                             : uint64_t(maxRow / h_deviceArgs.nRowsPack) * h_deviceArgs.nRowsPack;

    h_deviceArgs.maxTmp1 = 0;
    h_deviceArgs.maxTmp3 = 0;
    for (uint64_t k = 0; k < dest.params.size(); ++k)
    {
        ParserParams &parserParams = dest.expId == setupCtx.starkInfo.cExpId && !setupCtx.starkInfo.recursive
            ? setupCtx.expressionsBin.constraintsInfoDebug[dest.params[k].expId]
            : setupCtx.expressionsBin.expressionsInfo[dest.params[k].expId];
        if (parserParams.nTemp1 > h_deviceArgs.maxTmp1) {
            h_deviceArgs.maxTmp1 = parserParams.nTemp1;
        }
        if (parserParams.nTemp3*FIELD_EXTENSION > h_deviceArgs.maxTmp3) {
            h_deviceArgs.maxTmp3 = parserParams.nTemp3*FIELD_EXTENSION;
        }
    }

    h_deviceArgs.offsetTmp1 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    h_deviceArgs.offsetTmp3 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)];

    h_deviceArgs.domainSize = domainSize;
    h_deviceArgs.domainExtended = domainExtended;

    h_deviceArgs.dest_gpu = dest.dest_gpu;
    h_deviceArgs.dest_domainSize = dest.domainSize;
    h_deviceArgs.dest_offset = dest.offset;
    h_deviceArgs.dest_dim = dest.dim;
    h_deviceArgs.dest_id = dest.expId;
    h_deviceArgs.dest_nParams = dest.params.size();
    assert(dest.params.size() == 1 || dest.params.size() == 2 || dest.expId == setupCtx.starkInfo.cExpId);

    DestParamsGPU* aux_params = new DestParamsGPU[h_deviceArgs.dest_nParams];
    for (uint64_t j = 0; j < h_deviceArgs.dest_nParams; ++j){

        ParserParams &parserParams = dest.expId == setupCtx.starkInfo.cExpId && !setupCtx.starkInfo.recursive
            ? setupCtx.expressionsBin.constraintsInfoDebug[dest.params[j].expId]
            : setupCtx.expressionsBin.expressionsInfo[dest.params[j].expId];
        aux_params[j].dim = dest.params[j].dim;
        aux_params[j].stage = dest.params[j].stage;
        aux_params[j].stagePos = dest.params[j].stagePos;
        aux_params[j].polsMapId = dest.params[j].polsMapId;
        aux_params[j].rowOffsetIndex = dest.params[j].rowOffsetIndex;
        aux_params[j].inverse = dest.params[j].inverse;
        aux_params[j].op = dest.params[j].op;
        aux_params[j].value = dest.params[j].value;
        aux_params[j].nOps = parserParams.nOps;
        aux_params[j].opsOffset = parserParams.opsOffset;
        aux_params[j].nArgs = parserParams.nArgs;
        aux_params[j].argsOffset =parserParams.argsOffset;
    }
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.dest_params, h_deviceArgs.dest_nParams * sizeof(DestParamsGPU)));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.dest_params, aux_params, h_deviceArgs.dest_nParams * sizeof(DestParamsGPU), cudaMemcpyHostToDevice));
    delete[] aux_params;

    // Allocate memory for the struct on the device
    CHECKCUDAERR(cudaMalloc(&d_deviceArgs, sizeof(DeviceArguments)));
    CHECKCUDAERR(cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArguments), cudaMemcpyHostToDevice));
}

void ExpressionsGPU::calculateExpressions_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, bool domainExtended)
{
    loadDeviceArgs(domainSize, dest);

    uint32_t nblocks_ = (domainSize + h_deviceArgs.nRowsPack-1)/ h_deviceArgs.nRowsPack;
    uint32_t nThreads_ = nblocks_ == 1 ? domainSize : h_deviceArgs.nRowsPack;
    dim3 nBlocks =  nblocks_;
    dim3 nThreads = nThreads_;

    size_t sharedMem = (bufferCommitsSize + 9) * sizeof(Goldilocks::Element *);
    sharedMem += 2 * nThreads_ * FIELD_EXTENSION * sizeof(Goldilocks::Element);
    if(dest.params.size() == 2 || (dest.expId == setupCtx.starkInfo.cExpId && !setupCtx.starkInfo.recursive)) {
        sharedMem += nThreads_ * FIELD_EXTENSION * sizeof(Goldilocks::Element);
    }

    if(!setupCtx.starkInfo.recursive) {
        sharedMem += nThreads_ * (h_deviceArgs.maxTmp1 + h_deviceArgs.maxTmp3) * sizeof(Goldilocks::Element);
    }

    computeExpressions_<<<nBlocks, nThreads, sharedMem>>>(d_params, d_deviceArgs);
    
    if (dest.dest != NULL)
    {
        CHECKCUDAERR(cudaMemcpy(dest.dest, dest.dest_gpu, dest.domainSize * dest.dim * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    }
    
    
    CHECKCUDAERR(cudaFree(h_deviceArgs.dest_params));
    CHECKCUDAERR(cudaFree(d_deviceArgs));
}
__device__ __forceinline__ Goldilocks::Element* load__(
    const DeviceArguments* __restrict__ dArgs,
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
    const uint64_t domainSize = dArgs->domainSize;

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

    const int64_t stride = dArgs->nextStridesExps[argOffset];
    const uint64_t logicalRow = isCyclic ? (r + stride) % domainSize : (r + stride);

    // ConstPols
    if (type == 0) {
        const Goldilocks::Element* basePtr = dArgs->domainExtended
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
        const uint64_t offset = dArgs->mapOffsetsExps[type];
        const uint64_t nCols = dArgs->mapSectionsN[type];
        const uint64_t pos = logicalRow * nCols + argIdx;

        if (type == 1 && !dArgs->domainExtended) {
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
            ? (dArgs->domainExtended
                ? &dParams->aux_trace[dArgs->x_offset + row]
                : &dParams->aux_trace[dArgs->xn_offset + row])
            : &dParams->aux_trace[dArgs->zi_offset + (argIdx - 1) * domainSize + row];
    }

    // xi^-1 = inv(x - x_i)
    if (type == 5) {
        const gl64_t* xDivX = (gl64_t*)&dParams->xDivXSub[argIdx * FIELD_EXTENSION];
        const gl64_t* x = (gl64_t*)&dParams->aux_trace[dArgs->x_offset + row];
        Goldilocks3GPU::sub_13_gpu_a_const((gl64_t*)temp, xDivX, x);
        getInversePolinomial__((gl64_t*)temp, 3);
        Goldilocks3GPU::mul_31_gpu_no_const((gl64_t*)temp, (gl64_t*)temp, x);
        return temp;
    }

    // Custom commits
    const uint64_t idx = type - (dArgs->nStages + 4);
    const uint64_t offset = dArgs->mapOffsetsCustomExps[idx];
    const uint64_t nCols = dArgs->mapSectionsNCustomFixed[idx];
    const uint64_t pos = logicalRow * nCols + argIdx;

    temp[threadIdx.x] = dParams->pCustomCommitsFixed[offset + pos];
    return temp;
}

__device__ __noinline__ void storePolynomial__(DeviceArguments *d_deviceArgs, Goldilocks::Element *value, uint64_t row)
{
    if (d_deviceArgs->dest_dim == 1)
    {
        uint64_t offset = d_deviceArgs->dest_offset != 0 ? d_deviceArgs->dest_offset : 1;
        gl64_t::copy_gpu((gl64_t*) &d_deviceArgs->dest_gpu[row  * offset], uint64_t(offset), (gl64_t*)&value[0], false);
    }
    else
    {        
        uint64_t offset = d_deviceArgs->dest_offset != 0 ? d_deviceArgs->dest_offset : FIELD_EXTENSION;
        gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset], uint64_t(offset), (gl64_t*)&value[0], false);
        gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 1], uint64_t(offset), (gl64_t*)&value[blockDim.x], false);
        gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 2], uint64_t(offset), (gl64_t*)&value[2*blockDim.x], false);

    }
}

__device__ __noinline__ void multiplyPolynomials__(DeviceArguments *d_deviceArgs, gl64_t *valueA, gl64_t *valueB, uint64_t row)
{
    if (d_deviceArgs->dest_dim == 1)
    {
        valueA[threadIdx.x] = valueA[threadIdx.x] * valueB[threadIdx.x];
        uint64_t offset = d_deviceArgs->dest_offset != 0 ? d_deviceArgs->dest_offset : 1;
        gl64_t::copy_gpu((gl64_t*) &d_deviceArgs->dest_gpu[row  * offset], uint64_t(offset), valueA, false);
    }
    else
    {
        uint64_t offset = d_deviceArgs->dest_offset != 0 ? d_deviceArgs->dest_offset : FIELD_EXTENSION;
        if (d_deviceArgs->dest_params[0].dim == FIELD_EXTENSION && d_deviceArgs->dest_params[1].dim == FIELD_EXTENSION)
        {
            Goldilocks3GPU::mul_gpu_no_const(valueA, valueA, valueB);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset], uint64_t(offset), valueA, false);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 1], uint64_t(offset), &valueA[blockDim.x], false);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 2], uint64_t(offset), &valueA[2*blockDim.x], false);
        }
        else if (d_deviceArgs->dest_params[0].dim == FIELD_EXTENSION && d_deviceArgs->dest_params[1].dim == 1)
        {
            Goldilocks3GPU::mul_31_gpu_no_const(valueA, valueA, valueB);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset], uint64_t(offset), valueA, false);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 1], uint64_t(offset), &valueA[blockDim.x], false);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 2], uint64_t(offset), &valueA[2*blockDim.x], false);
        }
        else
        {
            Goldilocks3GPU::mul_31_gpu_no_const(valueB, valueB, valueA);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset], uint64_t(offset), (gl64_t*)&valueB[0], false);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 1], uint64_t(offset), (gl64_t*)&valueB[blockDim.x], false);
            gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 2], uint64_t(offset), (gl64_t*)&valueB[2*blockDim.x], false);
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

__device__ __noinline__ bool caseNoOperations__(StepsParams *d_params, DeviceArguments *d_deviceArgs, Goldilocks::Element *value, uint32_t k, uint64_t row)
{

#if DEBUG 
    bool print = blockIdx.x == 0 && threadIdx.x == 0 && row == DEBUG_ROW;
#endif

    uint32_t r = row + threadIdx.x;

    if (d_deviceArgs->dest_params[k].op == opType::cm || d_deviceArgs->dest_params[k].op == opType::const_)
    { // roger: assumeixes k==0 en aqeusta part?
        uint64_t openingPointIndex = d_deviceArgs->dest_params[k].rowOffsetIndex;
        uint64_t stagePos = d_deviceArgs->dest_params[k].stagePos;
        int64_t o = d_deviceArgs->nextStridesExps[openingPointIndex];
        uint64_t l = (r + o) % d_deviceArgs->domainSize;
        uint64_t nCols = d_deviceArgs->mapSectionsN[0];
        if (d_deviceArgs->dest_params[k].op == opType::const_)
        {
#if DEBUG
            if(print) printf("Expression debug constPols\n");
#endif
            value[threadIdx.x] = d_params->pConstPolsAddress[l * nCols + stagePos];
        }
        else
        {
            uint64_t offset = d_deviceArgs->mapOffsetsExps[d_deviceArgs->dest_params[k].stage];
            uint64_t nCols = d_deviceArgs->mapSectionsN[d_deviceArgs->dest_params[k].stage];
            if (d_deviceArgs->dest_params[k].stage == 1)
            {
#if DEBUG
                if(print) printf("Expression debug trace\n");
#endif
                value[threadIdx.x] = d_params->trace[l * nCols + stagePos];
            }
            else
            {
#if DEBUG
                if(print) printf("Expression debug aux_trace\n");
#endif
                for (uint64_t d = 0; d < d_deviceArgs->dest_params[k].dim; ++d)
                {
                    value[threadIdx.x + d * blockDim.x] = d_params->aux_trace[offset + l * nCols + stagePos + d];
                }
            }
        }

        if (d_deviceArgs->dest_params[k].inverse)
        {
#if DEBUG
            if(print) printf("Expression debug inverse\n");
#endif
            getInversePolinomial__((gl64_t*) value, d_deviceArgs->dest_params[k].dim);
        }
        return true;
    }
    else if (d_deviceArgs->dest_params[k].op == opType::number)
    {
#if DEBUG
        if(print) printf("Expression debug number\n");
#endif
        value[threadIdx.x].fe = d_deviceArgs->dest_params[k].value;
        return true;
    }
    else if (d_deviceArgs->dest_params[k].op == opType::airvalue)
    {
#if DEBUG
        if(print) printf("Expression debug airvalue\n");
#endif
        if(d_deviceArgs->dest_params[k].dim == 1) {
            value[threadIdx.x] = d_params->airValues[d_deviceArgs->dest_params[k].polsMapId];
        } else {
            value[threadIdx.x] = d_params->airValues[d_deviceArgs->dest_params[k].polsMapId];
            value[threadIdx.x + blockDim.x] = d_params->airValues[d_deviceArgs->dest_params[k].polsMapId + 1];
            value[threadIdx.x + 2 * blockDim.x] = d_params->airValues[d_deviceArgs->dest_params[k].polsMapId + 2];
        }
        return true;
    }
    return false;
}

__device__ __forceinline__ void printArguments(Goldilocks::Element *a, uint32_t dimA, bool constA, Goldilocks::Element *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, uint64_t nOps){
#if DEBUG
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
#endif
}

__device__ __forceinline__ void printRes(Goldilocks::Element *res, uint32_t dimRes, int i){
#if DEBUG
    bool print = threadIdx.x == 0  && i == DEBUG_ROW;
    if(print){
        for(uint32_t i = 0; i < dimRes; i++){
            printf("Expression debug res[%d]: %lu\n", i, res[i*blockDim.x].fe % GOLDILOCKS_PRIME);
        }
    }
#endif
}
__global__  void computeExpressions_(StepsParams *d_params, DeviceArguments *d_deviceArgs)
{

    int chunk_idx = blockIdx.x;
    uint64_t nchunks = d_deviceArgs->domainSize / blockDim.x;

    uint32_t bufferCommitsSize = d_deviceArgs->bufferCommitSize;
    Goldilocks::Element **expressions_params = (Goldilocks::Element **)scratchpad;

    Goldilocks::Element* tmp1_shared = (Goldilocks::Element *)(expressions_params + bufferCommitsSize + 9);
    Goldilocks::Element* tmp3_shared = tmp1_shared + blockDim.x * d_deviceArgs->maxTmp1;
    Goldilocks::Element* shared = d_deviceArgs->recursive ? (Goldilocks::Element *)(expressions_params + bufferCommitsSize + 9) : tmp3_shared + blockDim.x * d_deviceArgs->maxTmp3;
    if (threadIdx.x == 0)
    {
        expressions_params[bufferCommitsSize + 0] = d_deviceArgs->recursive ? (&d_params->aux_trace[d_deviceArgs->offsetTmp1 + blockIdx.x * d_deviceArgs->maxTmp1 * d_deviceArgs->nRowsPack]) : tmp1_shared;
        expressions_params[bufferCommitsSize + 1] = d_deviceArgs->recursive ? (&d_params->aux_trace[d_deviceArgs->offsetTmp3 + blockIdx.x * d_deviceArgs->maxTmp3 * d_deviceArgs->nRowsPack]) : tmp3_shared;
        expressions_params[bufferCommitsSize + 2] = d_params->publicInputs;
        expressions_params[bufferCommitsSize + 3] = d_deviceArgs->cExpId == d_deviceArgs->dest_id && !d_deviceArgs->recursive ? d_deviceArgs->numbersConstraints : d_deviceArgs->numbers;
        expressions_params[bufferCommitsSize + 4] = d_params->airValues;
        expressions_params[bufferCommitsSize + 5] = d_params->proofValues;
        expressions_params[bufferCommitsSize + 6] = d_params->airgroupValues;
        expressions_params[bufferCommitsSize + 7] = d_params->challenges;
        expressions_params[bufferCommitsSize + 8] = d_params->evals;
    }
    __syncthreads();

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool isCyclic = i < d_deviceArgs->k_min || i >= d_deviceArgs->k_max;
#pragma unroll
        for (uint64_t k = 0; k < d_deviceArgs->dest_nParams; ++k)
        {
            Goldilocks::Element *valueA = d_deviceArgs->cExpId == d_deviceArgs->dest_id && !d_deviceArgs->recursive ? shared : shared + k * blockDim.x * FIELD_EXTENSION;
            Goldilocks::Element *valueB =  valueA + blockDim.x * FIELD_EXTENSION;
            Goldilocks::Element *valueAcc;

            if(d_deviceArgs->cExpId == d_deviceArgs->dest_id && !d_deviceArgs->recursive) {
                valueAcc = valueB + blockDim.x * FIELD_EXTENSION;
            }
            if(caseNoOperations__(d_params, d_deviceArgs, valueA, k, i)){
                continue;
            }
            uint8_t *ops = d_deviceArgs->cExpId == d_deviceArgs->dest_id && !d_deviceArgs->recursive ? &d_deviceArgs->opsConstraints[d_deviceArgs->dest_params[k].opsOffset] : &d_deviceArgs->ops[d_deviceArgs->dest_params[k].opsOffset];
            uint16_t *args = d_deviceArgs->cExpId == d_deviceArgs->dest_id && !d_deviceArgs->recursive ? &d_deviceArgs->argsConstraints[d_deviceArgs->dest_params[k].argsOffset] : &d_deviceArgs->args[d_deviceArgs->dest_params[k].argsOffset];
            
            uint64_t i_args = 0;
            uint64_t nOps = d_deviceArgs->dest_params[k].nOps;
            for (uint64_t kk = 0; kk < nOps; ++kk)
            {

                switch (ops[kk])
                {

                case 0:
                {
                    // COPY dim1 to dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args[i_args + 1], args[i_args + 2], args[i_args + 3], i, 1, isCyclic);
                    bool isConstant = args[i_args + 1] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? valueA : &expressions_params[bufferCommitsSize][args[i_args] * blockDim.x]);
                    printArguments((Goldilocks::Element *) a, 1, isConstant, NULL, true, 0, i, 4, kk, nOps);
                    gl64_t::copy_gpu(res, a, isConstant);
                    printRes((Goldilocks::Element *) res, 1, i);
                    i_args += 4;
                    break;
                }
                case 1:
                {
                    // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? valueA : &expressions_params[bufferCommitsSize][args[i_args + 1] * blockDim.x]);
                    printArguments((Goldilocks::Element *)a, 1, isConstantA, (Goldilocks::Element *)b, 1, isConstantB, i, args[i_args], kk, nOps);
                    gl64_t::op_gpu( args[i_args], res, a, isConstantA, b, isConstantB);
                    printRes((Goldilocks::Element *) res, 1, i);
                    i_args += 8;
                    break;
                }
                case 2:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? valueA : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    printArguments((Goldilocks::Element *)a, 3, isConstantA, (Goldilocks::Element *)b, 1, isConstantB, i, args[i_args], kk, nOps);
                    Goldilocks3GPU::op_31_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    printRes((Goldilocks::Element *) res, 3, i);
                    i_args += 8;
                    break;
                }
                case 3:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, valueB, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? valueA : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    printArguments((Goldilocks::Element *)a, 3, isConstantA, (Goldilocks::Element *)b, 3, isConstantB, i, args[i_args], kk, nOps);
                    Goldilocks3GPU::op_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    printRes((Goldilocks::Element *) res, 3, i);
                    i_args += 8;
                    break;
                }
                case 4:
                {
                    // COPY dim3 to dim3
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args[i_args + 1], args[i_args + 2], args[i_args + 3], i, 3, isCyclic);
                    bool isConstant = args[i_args + 1] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? valueA : &expressions_params[bufferCommitsSize + 1][args[i_args] * blockDim.x]);
                    Goldilocks3GPU::copy_gpu(res, a, isConstant);
                    printRes((Goldilocks::Element *) res, 3, i);
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
            if (i_args !=  d_deviceArgs->dest_params[k].nArgs){
                printf(" %lu consumed args - %lu expected args \n", i_args, d_deviceArgs->dest_params[k].nArgs);
                assert(0);
            }
            if (d_deviceArgs->dest_params[k].inverse)
            {
                getInversePolinomial__((gl64_t*) valueA, d_deviceArgs->dest_params[k].dim);
            }
            
            if (d_deviceArgs->dest_id == d_deviceArgs->cExpId && !d_deviceArgs->recursive)
            {
                if(k == 0) {
                    Goldilocks3GPU::copy_gpu((gl64_t *)valueAcc, (gl64_t *)valueA, false);   
                } else {
                    // TODO: NOT HARDCODE!!
                    Goldilocks::Element *challenge = &d_params->challenges[2*FIELD_EXTENSION];
                    if(k == 1 && d_deviceArgs->dest_params[0].dim == 1) {
                        printArguments(challenge, 3, true, valueAcc, 1, false, i, 2, k, d_deviceArgs->dest_nParams);
                        Goldilocks3GPU::mul_31_gpu_a_const((gl64_t *)valueB, (gl64_t *)challenge, (gl64_t *)valueAcc);
                        Goldilocks3GPU::copy_gpu((gl64_t *)valueAcc, (gl64_t *)valueB, false);
                    } else {
                        printArguments(valueAcc, 3, false, challenge, 3, true, i, 2, k, d_deviceArgs->dest_nParams);
                        Goldilocks3GPU::mul_gpu_b_const((gl64_t *)valueAcc, (gl64_t *)valueAcc, (gl64_t *)challenge);
                    }
                    printRes(valueAcc, 3, i);
                    if (d_deviceArgs->dest_params[k].dim == 1) {
                        printArguments(valueAcc, 3, false, valueA, 1, false, i, 0, k, d_deviceArgs->dest_nParams);
                        Goldilocks3GPU::add_31_gpu_no_const((gl64_t *)valueAcc, (gl64_t *)valueAcc, (gl64_t *)valueA);
                        printRes(valueAcc, 3, i);
                    } else {
                        printArguments(valueAcc, 3, false, valueA, 3, false, i, 0, k, d_deviceArgs->dest_nParams);
                        Goldilocks3GPU::add_gpu_no_const((gl64_t *)valueAcc, (gl64_t *)valueAcc, (gl64_t *)valueA);
                        printRes(valueAcc, 3, i);
                    }
                }
            }
        }

        if (d_deviceArgs->dest_nParams == 2)
        {
            multiplyPolynomials__(d_deviceArgs, (gl64_t*) shared, (gl64_t*) shared + blockDim.x * FIELD_EXTENSION, i);
        } else if (d_deviceArgs->dest_id == d_deviceArgs->cExpId && !d_deviceArgs->recursive) {
            Goldilocks::Element *valueAcc = shared + 2 * blockDim.x * FIELD_EXTENSION;
            Goldilocks::Element *zi = &d_params->aux_trace[d_deviceArgs->zi_offset + i];
            printArguments(valueAcc, 3, false, zi, 1, false, i, 0, d_deviceArgs->dest_nParams - 1, d_deviceArgs->dest_nParams);
            Goldilocks3GPU::mul_31_gpu_no_const((gl64_t *)valueAcc, (gl64_t *)valueAcc, (gl64_t *)zi);
            printRes(valueAcc, 3, i);
            storePolynomial__(d_deviceArgs, valueAcc, i);
        } else {
            storePolynomial__(d_deviceArgs, shared, i);
        }

        chunk_idx += gridDim.x;
    }

}
