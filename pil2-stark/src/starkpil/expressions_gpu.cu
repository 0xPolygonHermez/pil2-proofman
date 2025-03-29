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

ExpressionsGPU::ExpressionsGPU(SetupCtx &setupCtx, ProverHelpers &proverHelpers, uint32_t nRowsPack, uint32_t nBlocks) : ExpressionsCtx(setupCtx, proverHelpers), nRowsPack(nRowsPack), nBlocks(nBlocks)
{
    
    uint32_t ns = 1 + setupCtx.starkInfo.nStages + 1;
    uint32_t nCustoms = setupCtx.starkInfo.customCommits.size();
    uint32_t nOpenings = setupCtx.starkInfo.openingPoints.size();
    uint32_t nStages_ = setupCtx.starkInfo.nStages;
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    h_deviceArgs.N = N;
    h_deviceArgs.NExtended = NExtended;
    h_deviceArgs.nBlocks = nBlocks;
    h_deviceArgs.nStages = nStages_;
    h_deviceArgs.nCustomCommits = nCustoms;
    h_deviceArgs.bufferCommitSize = 1 + nStages_ + 3 + nCustoms;

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

    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.zi, setupCtx.starkInfo.boundaries.size() * NExtended * sizeof(Goldilocks::Element))); 
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.x, NExtended * sizeof(Goldilocks::Element)));

    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.zi, proverHelpers.zi, setupCtx.starkInfo.boundaries.size() * h_deviceArgs.NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice)); 
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.x, proverHelpers.x, NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));                                

    if(proverHelpers.x_n != nullptr) {
        CHECKCUDAERR(cudaMalloc(&h_deviceArgs.x_n, N * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMemcpy(h_deviceArgs.x_n, proverHelpers.x_n, N * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));                
    }

};

ExpressionsGPU::~ExpressionsGPU()
{
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsets));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.nextStrides));
    CHECKCUDAERR(cudaFree(h_deviceArgs.nextStridesExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsCustomFixed));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsCustomFixedExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapSectionsN));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapSectionsNCustomFixed));
    CHECKCUDAERR(cudaFree(h_deviceArgs.zi));
    if(proverHelpers.x_n != nullptr) {
        CHECKCUDAERR(cudaFree(h_deviceArgs.x_n));
    }
    CHECKCUDAERR(cudaFree(h_deviceArgs.x));
    CHECKCUDAERR(cudaFree(h_deviceArgs.numbers));
    CHECKCUDAERR(cudaFree(h_deviceArgs.ops));
    CHECKCUDAERR(cudaFree(h_deviceArgs.args));
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

    h_deviceArgs.maxTemp1Size = 0;
    h_deviceArgs.maxTemp3Size = 0;

    h_deviceArgs.offsetTmp1 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    h_deviceArgs.offsetTmp3 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)];
    h_deviceArgs.offsetDestVals = setupCtx.starkInfo.mapOffsets[std::make_pair("destVals", false)];
    
    for (uint64_t k = 0; k < dest.params.size(); ++k)
    {
        ParserParams &parserParams = setupCtx.expressionsBin.expressionsInfo[dest.params[k].expId];
        if (parserParams.nTemp1*h_deviceArgs.nRowsPack > h_deviceArgs.maxTemp1Size) {
            h_deviceArgs.maxTemp1Size = parserParams.nTemp1*h_deviceArgs.nRowsPack;
        }
        if (parserParams.nTemp3*h_deviceArgs.nRowsPack*FIELD_EXTENSION > h_deviceArgs.maxTemp3Size) {
            h_deviceArgs.maxTemp3Size = parserParams.nTemp3*h_deviceArgs.nRowsPack*FIELD_EXTENSION;
        }
    }

    h_deviceArgs.domainSize = domainSize;
    h_deviceArgs.domainExtended = domainExtended;

    h_deviceArgs.dest_gpu = dest.dest_gpu;
    h_deviceArgs.dest_domainSize = dest.domainSize;
    h_deviceArgs.dest_offset = dest.offset;
    h_deviceArgs.dest_dim = dest.dim;
    h_deviceArgs.dest_nParams = dest.params.size();
    assert(dest.params.size() == 1 || dest.params.size() == 2);

    DestParamsGPU* aux_params = new DestParamsGPU[h_deviceArgs.dest_nParams];
    for (uint64_t j = 0; j < h_deviceArgs.dest_nParams; ++j){

        ParserParams &parserParams = setupCtx.expressionsBin.expressionsInfo[dest.params[j].expId];
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
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();
    loadDeviceArgs(domainSize, dest);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    //std::cout << "goal2_ setBufferTInfo time: " << time << std::endl;

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    uint32_t nblocks_ = std::min(h_deviceArgs.nBlocks, (domainSize + h_deviceArgs.nRowsPack-1)/ h_deviceArgs.nRowsPack);
    uint32_t nThreads_ = nblocks_ == 1 ? domainSize : h_deviceArgs.nRowsPack;
    dim3 nBlocks =  nblocks_;
    dim3 nThreads = nThreads_;

    size_t sharedMem = (bufferCommitsSize  + 9) * sizeof(Goldilocks::Element *) + 2 * nThreads_ * FIELD_EXTENSION * sizeof(Goldilocks::Element);

    computeExpressions_<<<nBlocks, nThreads, sharedMem>>>(d_params, d_deviceArgs);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    //std::cout << "goal2_ de computeExpressions: " << time << std::endl;

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    
    if (dest.dest != NULL)
    {
        CHECKCUDAERR(cudaMemcpy(dest.dest, dest.dest_gpu, dest.domainSize * dest.dim * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    }
    
    time = omp_get_wtime() - time;
    //std::cout << "goal2_ de cudaMemcpy dests time: " << time << std::endl;

    
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaFree(h_deviceArgs.dest_params));
    CHECKCUDAERR(cudaFree(d_deviceArgs));
    CHECKCUDAERR(cudaGetLastError());
}

__device__ __forceinline__ Goldilocks::Element*  load__(DeviceArguments *d_deviceArgs, Goldilocks::Element *value, StepsParams* d_params, Goldilocks::Element** expressions_params, uint16_t* args, uint64_t i_args, uint64_t row, uint64_t dim, bool isCyclic) {        

#if DEBUG 
    bool print = threadIdx.x == 0 && row == DEBUG_ROW;
#endif

    uint32_t r = row + threadIdx.x;
    uint32_t nStages = d_deviceArgs->nStages;
    uint64_t type = args[i_args];

    if (type ==  d_deviceArgs->bufferCommitSize || type == d_deviceArgs->bufferCommitSize + 1) {
#if DEBUG
        if(print){ 
            if(type == d_deviceArgs->bufferCommitSize) printf("Expression debug tmp1\n");
            if(type == d_deviceArgs->bufferCommitSize + 1) printf("Expression debug tmp3\n");
        }
#endif
        return &expressions_params[type][args[i_args + 1]*blockDim.x];
    } else if (type >= d_deviceArgs->bufferCommitSize + 2) {
#if DEBUG
        if(print){
            if(type == d_deviceArgs->bufferCommitSize + 2 ) printf("Expression debug publicInputs\n");
            if(type == d_deviceArgs->bufferCommitSize + 3 ) printf("Expression debug numbers\n");
            if(type == d_deviceArgs->bufferCommitSize + 4 ) printf("Expression debug airValues\n");
            if(type == d_deviceArgs->bufferCommitSize + 5 ) printf("Expression debug proofValues\n");
            if(type == d_deviceArgs->bufferCommitSize + 6 ) printf("Expression debug airgroupValues\n");
            if(type == d_deviceArgs->bufferCommitSize + 7 ) printf("Expression debug challenges\n");
            if(type == d_deviceArgs->bufferCommitSize + 8 ) printf("Expression debug evals\n");
        }
#endif
        return &expressions_params[type][args[i_args + 1]];
    }

    switch (type)
    {
    case 0:
    {
        if(dim == FIELD_EXTENSION) { assert(0); }
        Goldilocks::Element *constPols = d_deviceArgs->domainExtended ?  &(d_params->pConstPolsExtendedTreeAddress[2]) :  d_params->pConstPolsAddress;
        uint64_t stagePos = args[i_args + 1];
        int64_t o = d_deviceArgs->nextStridesExps[args[i_args + 2]];
        uint64_t nCols = d_deviceArgs->mapSectionsN[0];
        if(isCyclic) {
            uint64_t l = (r + o) % d_deviceArgs->domainSize;
#if DEBUG 
            if(print) printf("Expression debug constPols cyclic\n");
#endif
            value[threadIdx.x] = constPols[l * nCols + stagePos];
            return value;
        } else {
#if DEBUG
            if(print) printf("Expression debug constPols\n");
#endif
            value[threadIdx.x] = constPols[(r + o)*nCols + stagePos];
            return value;
        }
        break;
    }
    case 1:
    case 2: //rick: harcoded nStages=2
    case 3:
    {
        uint64_t stagePos = args[i_args + 1];
        uint64_t offset = d_deviceArgs->mapOffsetsExps[type];
        uint64_t nCols = d_deviceArgs->mapSectionsN[type];
        int64_t o = d_deviceArgs->nextStridesExps[args[i_args + 2]];
        if(isCyclic) {
            uint64_t l = (r + o) % d_deviceArgs->domainSize;
            if(type == 1 && !d_deviceArgs->domainExtended) {
#if DEBUG
                if(print) printf("Expression debug trace cyclic: %lu\n",l * nCols + stagePos );
#endif
                value[threadIdx.x] = d_params->trace[l * nCols + stagePos];
            } else {
#if DEBUG
                if(print) printf("Expression debug aux_trace cyclic %lu\n", offset + l * nCols + stagePos);
#endif
                for(uint64_t d = 0; d < dim; ++d) {
                    value[threadIdx.x + d*blockDim.x] = d_params->aux_trace[offset + l * nCols + stagePos + d];
                }
            }
        } else {
            if(type == 1 && !d_deviceArgs->domainExtended) {
#if DEBUG
                if(print) printf("Expression debug trace\n");
#endif
                value[threadIdx.x] = d_params->trace[(r + o)*nCols+ stagePos];
                
            } else {
#if DEBUG
                if(print) printf("Expression debug aux_trace\n");
#endif
                for(uint64_t d = 0; d < dim; ++d) {
                    value[threadIdx.x + d*blockDim.x] = d_params->aux_trace[offset + (r + o) *nCols + stagePos + d];
                }                        
            }
        }
        return value;
        break;
    }
    case 4:
    {
        uint64_t boundary = args[i_args + 1];        
        if(boundary == 0) {
#if DEBUG
            if(print) printf("Expression debug x or x_n\n");
#endif
            Goldilocks::Element *x = d_deviceArgs->domainExtended ? &d_deviceArgs->x[row] : &d_deviceArgs->x_n[row];
            return x;
        } else {
#if DEBUG
            if(print) printf("Expression debug zi\n");
#endif
            return &d_deviceArgs->zi[(boundary - 1)*d_deviceArgs->domainSize  + row];
        }
        break;
    }
    case 5:
    {
#if DEBUG
        if(print) printf("Expression debug xi\n");
#endif
        if(dim == 1) { assert(0); }
        uint64_t o = args[i_args + 1];
        Goldilocks3GPU::op_31_gpu(3, (gl64_t*)value, (gl64_t*) (&d_params->xDivXSub[o * FIELD_EXTENSION]), true,  (gl64_t*) &d_deviceArgs->x[row], false);
        getInversePolinomial__((gl64_t*) value, 3);  
        Goldilocks3GPU::op_31_gpu(2, (gl64_t*)value, (gl64_t*)value, false, (gl64_t*) &d_deviceArgs->x[row], false);
        return value;
        break;
    }
    default:
    {
        uint64_t index = type - (nStages + 4);
        uint64_t stagePos = args[i_args + 1];
        uint64_t offset = d_deviceArgs->mapOffsetsCustomExps[index];
        uint64_t nCols = d_deviceArgs->mapSectionsNCustomFixed[index];
        int64_t o = d_deviceArgs->nextStridesExps[args[i_args + 2]];
        if(isCyclic) {
            uint64_t l = (r + o) % d_deviceArgs->domainSize;
#if DEBUG
            if(print) printf("Expression debug customCommits cyclic\n");
#endif
            value[threadIdx.x] = d_params->pCustomCommitsFixed[offset + l * nCols + stagePos];
        } else {
#if DEBUG
            if(print) printf("Expression debug customCommits\n");
#endif
            value[threadIdx.x] = d_params->pCustomCommitsFixed[offset + (r + o) * nCols + stagePos];            
        }
        return value;
        break;
    }    
    }

}

__device__ __noinline__ void storePolynomial__(DeviceArguments *d_deviceArgs, Goldilocks::Element *destVals, uint64_t row)
{
    if (d_deviceArgs->dest_dim == 1)
    {
        uint64_t offset = d_deviceArgs->dest_offset != 0 ? d_deviceArgs->dest_offset : 1;
        gl64_t::copy_gpu((gl64_t*) &d_deviceArgs->dest_gpu[row  * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
    }
    else
    {        
        uint64_t offset = d_deviceArgs->dest_offset != 0 ? d_deviceArgs->dest_offset : FIELD_EXTENSION;
        gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset], uint64_t(offset), (gl64_t*)&destVals[0], false);
        gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 1], uint64_t(offset), (gl64_t*)&destVals[blockDim.x], false);
        gl64_t::copy_gpu((gl64_t*)&d_deviceArgs->dest_gpu[row * offset + 2], uint64_t(offset), (gl64_t*)&destVals[2*blockDim.x], false);

    }
}

__device__ __noinline__ void multiplyPolynomials__(DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row)
{
    if (d_deviceArgs->dest_dim == 1)
    {
        gl64_t::op_gpu(2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * blockDim.x], false);
    }
    else
    {
        Goldilocks::Element **expressions_params = (Goldilocks::Element **)scratchpad;
        gl64_t*  vals = (gl64_t*) ( expressions_params + d_deviceArgs->bufferCommitSize + 9); //rick
        if (d_deviceArgs->dest_params[0].dim == FIELD_EXTENSION && d_deviceArgs->dest_params[1].dim == FIELD_EXTENSION)
        {
            Goldilocks3GPU::op_gpu(2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * blockDim.x], false);
        }
        else if (d_deviceArgs->dest_params[0].dim == FIELD_EXTENSION && d_deviceArgs->dest_params[1].dim == 1)
        {
            Goldilocks3GPU::op_31_gpu(2, &vals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * blockDim.x], false);
        }
        else
        {
            Goldilocks3GPU::op_31_gpu(2, &vals[0], &destVals[FIELD_EXTENSION * blockDim.x], false, &destVals[0], false);
        }
        gl64_t::copy_gpu(&destVals[0], &vals[0], false);
        gl64_t::copy_gpu(&destVals[blockDim.x], &vals[blockDim.x], false);
        gl64_t::copy_gpu(&destVals[2 * blockDim.x], &vals[2 * blockDim.x], false);
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

__device__ __noinline__ bool caseNoOperations__(StepsParams *d_params, DeviceArguments *d_deviceArgs, Goldilocks::Element *destVals, uint32_t k, uint64_t row)
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
            destVals[threadIdx.x] = d_params->pConstPolsAddress[l * nCols + stagePos];
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
                destVals[threadIdx.x] = d_params->trace[l * nCols + stagePos];
            }
            else
            {
#if DEBUG
                if(print) printf("Expression debug aux_trace\n");
#endif
                for (uint64_t d = 0; d < d_deviceArgs->dest_params[k].dim; ++d)
                {
                    destVals[threadIdx.x + d * blockDim.x] = d_params->aux_trace[offset + l * nCols + stagePos + d];
                }
            }
        }

        if (d_deviceArgs->dest_params[k].inverse)
        {
#if DEBUG
            if(print) printf("Expression debug inverse\n");
#endif
            getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_deviceArgs->dest_params[k].dim);
        }
        return true;
    }
    else if (d_deviceArgs->dest_params[k].op == opType::number)
    {
#if DEBUG
        if(print) printf("Expression debug number\n");
#endif
        destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x].fe = d_deviceArgs->dest_params[k].value;
        return true;
    }
    else if (d_deviceArgs->dest_params[k].op == opType::airvalue)
    {
#if DEBUG
        if(print) printf("Expression debug airvalue\n");
#endif
        if(d_deviceArgs->dest_params[k].dim == 1) {
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x] = d_params->airValues[d_deviceArgs->dest_params[k].polsMapId];
        } else {
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x] = d_params->airValues[d_deviceArgs->dest_params[k].polsMapId];
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x + blockDim.x] = d_params->airValues[d_deviceArgs->dest_params[k].polsMapId + 1];
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x + 2 * blockDim.x] = d_params->airValues[d_deviceArgs->dest_params[k].polsMapId + 2];
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

    if (threadIdx.x == 0)
    {
        expressions_params[bufferCommitsSize + 0] = (&d_params->aux_trace[d_deviceArgs->offsetTmp1 + blockIdx.x * d_deviceArgs->maxTemp1Size]);
        expressions_params[bufferCommitsSize + 1] = (&d_params->aux_trace[d_deviceArgs->offsetTmp3 + blockIdx.x * d_deviceArgs->maxTemp3Size]);
        expressions_params[bufferCommitsSize + 2] = d_params->publicInputs;
        expressions_params[bufferCommitsSize + 3] = d_deviceArgs->numbers;
        expressions_params[bufferCommitsSize + 4] = d_params->airValues;
        expressions_params[bufferCommitsSize + 5] = d_params->proofValues;
        expressions_params[bufferCommitsSize + 6] = d_params->airgroupValues;
        expressions_params[bufferCommitsSize + 7] = d_params->challenges;
        expressions_params[bufferCommitsSize + 8] = d_params->evals;
    }
    __syncthreads();
    Goldilocks::Element *destVals = &(d_params->aux_trace[d_deviceArgs->offsetDestVals + blockIdx.x * d_deviceArgs->dest_nParams * blockDim.x * FIELD_EXTENSION]); 

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool isCyclic = i < d_deviceArgs->k_min || i >= d_deviceArgs->k_max;
#pragma unroll 1
        for (uint64_t k = 0; k < d_deviceArgs->dest_nParams; ++k)
        {
            if(caseNoOperations__(d_params, d_deviceArgs, destVals, k, i)){
                continue;
            }
            uint8_t *ops = &d_deviceArgs->ops[d_deviceArgs->dest_params[k].opsOffset];
            uint16_t *args = &d_deviceArgs->args[d_deviceArgs->dest_params[k].argsOffset];
            Goldilocks::Element *valueA = (Goldilocks::Element *)( expressions_params + bufferCommitsSize + 9);
            Goldilocks::Element *valueB =  valueA + blockDim.x * FIELD_EXTENSION;

            uint64_t i_args = 0;
            uint64_t nOps = d_deviceArgs->dest_params[k].nOps;
            for (uint64_t kk = 0; kk < nOps; ++kk)

            {

                switch (ops[kk])
                {

                case 0:
                {
                    // COPY dim1 to dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args, i_args + 1, i, 1, isCyclic);
                    bool isConstant = args[i_args + 1] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize][args[i_args] * blockDim.x]);
                    printArguments((Goldilocks::Element *) a, 1, isConstant, NULL, true, 0, i, 4, kk, nOps);
                    gl64_t::copy_gpu(res, a, isConstant);
                    printRes((Goldilocks::Element *) res, 1, i);
                    i_args += 4;
                    break;
                }
                case 1:
                {
                    // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args, i_args + 2, i, 1, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, valueB, d_params, expressions_params, args, i_args + 5, i, 1, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize][args[i_args + 1] * blockDim.x]);
                    printArguments((Goldilocks::Element *)a, 1, isConstantA, (Goldilocks::Element *)b, 1, isConstantB, i, args[i_args], kk, nOps);
                    gl64_t::op_gpu( args[i_args], res, a, isConstantA, b, isConstantB);
                    printRes((Goldilocks::Element *) res, 1, i);
                    i_args += 8;
                    break;
                }
                case 2:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args, i_args + 2, i, 3, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, valueB, d_params, expressions_params, args, i_args + 5, i, 1, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    printArguments((Goldilocks::Element *)a, 3, isConstantA, (Goldilocks::Element *)b, 1, isConstantB, i, args[i_args], kk, nOps);
                    Goldilocks3GPU::op_31_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    printRes((Goldilocks::Element *) res, 3, i);
                    i_args += 8;
                    break;
                }
                case 3:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args, i_args + 2, i, 3, isCyclic);
                    gl64_t* b = (gl64_t*)load__(d_deviceArgs, valueB, d_params, expressions_params, args, i_args + 5, i, 3, isCyclic);
                    bool isConstantA = args[i_args + 2] > bufferCommitsSize + 1 ? true : false;
                    bool isConstantB = args[i_args + 5] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    printArguments((Goldilocks::Element *)a, 3, isConstantA, (Goldilocks::Element *)b, 3, isConstantB, i, args[i_args], kk, nOps);
                    Goldilocks3GPU::op_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
                    printRes((Goldilocks::Element *) res, 3, i);
                    i_args += 8;
                    break;
                }
                case 4:
                {
                    // COPY dim3 to dim3
                    gl64_t* a = (gl64_t*)load__(d_deviceArgs, valueA, d_params, expressions_params, args, i_args + 1, i, 3, isCyclic);
                    bool isConstant = args[i_args + 1] > bufferCommitsSize + 1 ? true : false;
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args] * blockDim.x]);
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
                getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_deviceArgs->dest_params[k].dim);
            }
            
        }

        if (d_deviceArgs->dest_nParams == 2)
        {

            multiplyPolynomials__(d_deviceArgs, (gl64_t*) destVals, i);
        }
        storePolynomial__(d_deviceArgs, destVals, i);

        chunk_idx += gridDim.x;
    }

}
