#include "expressions_gpu.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"

ExpressionsGPU::ExpressionsGPU(SetupCtx &setupCtx, ProverHelpers &proverHelpers, uint32_t nRowsPack, uint32_t nBlocks) : ExpressionsCtx(setupCtx, proverHelpers), nRowsPack(nRowsPack), nBlocks(nBlocks)
{

    uint32_t ns = 1 + setupCtx.starkInfo.nStages + 1;
    uint32_t nCustoms = setupCtx.starkInfo.customCommits.size();
    uint32_t nOpenings = setupCtx.starkInfo.openingPoints.size();
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    h_deviceArgs.N = N;
    h_deviceArgs.NExtended = NExtended;
    h_deviceArgs.nRowsPack = nRowsPack;
    h_deviceArgs.nBlocks = nBlocks;
    h_deviceArgs.nStages = nStages;
    h_deviceArgs.nCustomCommits = setupCtx.starkInfo.customCommits.size();
    h_deviceArgs.bufferCommitSize = bufferCommitsSize;

    cudaMalloc(&h_deviceArgs.mapOffsets, ns * sizeof(uint64_t));
    cudaMalloc(&h_deviceArgs.mapOffsetsExtended, ns * sizeof(uint64_t));
    cudaMalloc(&h_deviceArgs.mapOffsetsCustomFixed, nCustoms * sizeof(uint64_t));
    cudaMalloc(&h_deviceArgs.mapOffsetsCustomFixedExtended, nCustoms * sizeof(uint64_t));
    cudaMalloc(&h_deviceArgs.nextStrides, nOpenings * sizeof(uint64_t));
    cudaMalloc(&h_deviceArgs.nextStridesExtended, nOpenings * sizeof(uint64_t));
    cudaMalloc(&h_deviceArgs.mapSectionsN, ns * sizeof(uint64_t));
    cudaMalloc(&h_deviceArgs.mapSectionsNCustomFixed, nCustoms * sizeof(uint64_t));

    cudaMemcpy(h_deviceArgs.mapOffsets, mapOffsets, ns * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.mapOffsetsExtended, mapOffsetsExtended, ns * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.mapOffsetsCustomFixed, mapOffsetsCustomFixed, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.mapOffsetsCustomFixedExtended, mapOffsetsCustomFixedExtended, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.nextStrides, nextStrides, nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.nextStridesExtended, nextStridesExtended, nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.mapSectionsN, mapSectionsN, ns * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.mapSectionsNCustomFixed, mapSectionsNCustomFixed, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice);


    ParserArgs parserArgs = setupCtx.expressionsBin.expressionsBinArgsExpressions;
    cudaMalloc(&h_deviceArgs.numbers, parserArgs.nNumbers * sizeof(Goldilocks::Element));
    cudaMemcpy(h_deviceArgs.numbers, (Goldilocks::Element *)parserArgs.numbers, parserArgs.nNumbers * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

    cudaMalloc(&h_deviceArgs.ops, setupCtx.expressionsBin.nOpsTotal * sizeof(uint8_t));   
    cudaMalloc(&h_deviceArgs.args, setupCtx.expressionsBin.nArgsTotal * sizeof(uint16_t)); 
    cudaMemcpy(h_deviceArgs.ops, parserArgs.ops, setupCtx.expressionsBin.nOpsTotal * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(h_deviceArgs.args, parserArgs.args, setupCtx.expressionsBin.nArgsTotal * sizeof(uint16_t), cudaMemcpyHostToDevice);

    cudaMalloc(&h_deviceArgs.zi, setupCtx.starkInfo.boundaries.size() * NExtended * sizeof(Goldilocks::Element)); 
    cudaMalloc(&h_deviceArgs.x_n, N * sizeof(Goldilocks::Element));
    cudaMalloc(&h_deviceArgs.x, NExtended * sizeof(Goldilocks::Element));

    cudaMemcpy(h_deviceArgs.zi, proverHelpers.zi, setupCtx.starkInfo.boundaries.size() * h_deviceArgs.NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice); 
    cudaMemcpy(h_deviceArgs.x_n, proverHelpers.x_n, N * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);                
    cudaMemcpy(h_deviceArgs.x, proverHelpers.x, NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);                                

    // buffers for execution
    uint64_t tmp1Size = setupCtx.expressionsBin.maxTmp1 * nRowsPack;
    uint64_t tmp3Size = setupCtx.expressionsBin.maxTmp3 * FIELD_EXTENSION * nRowsPack;
    cudaMalloc(&h_deviceArgs.tmp1, nBlocks * tmp1Size * sizeof(Goldilocks::Element));
    cudaMalloc(&h_deviceArgs.tmp3, nBlocks * tmp3Size * sizeof(Goldilocks::Element));
    cudaMalloc(&h_deviceArgs.destVals, nBlocks * nRowsPack * 2 * FIELD_EXTENSION * sizeof(Goldilocks::Element));

};

ExpressionsGPU::~ExpressionsGPU()
{
    cudaFree(h_deviceArgs.mapOffsets);
    cudaFree(h_deviceArgs.mapOffsetsExtended);
    cudaFree(h_deviceArgs.nextStrides);
    cudaFree(h_deviceArgs.nextStridesExtended);
    cudaFree(h_deviceArgs.mapOffsetsCustomFixed);
    cudaFree(h_deviceArgs.mapOffsetsCustomFixedExtended);
    cudaFree(h_deviceArgs.mapSectionsN);
    cudaFree(h_deviceArgs.mapSectionsNCustomFixed);
    cudaFree(h_deviceArgs.zi);
    cudaFree(h_deviceArgs.x_n);
    cudaFree(h_deviceArgs.x);
    cudaFree(h_deviceArgs.numbers);
    cudaFree(h_deviceArgs.ops);
    cudaFree(h_deviceArgs.args);
    //next three will be eliminated
    cudaFree(h_deviceArgs.destVals);
    cudaFree(h_deviceArgs.tmp1);
    cudaFree(h_deviceArgs.tmp3);

}

void ExpressionsGPU::loadDeviceArgs(uint64_t domainSize, StepsParams &d_params, Dest &dest)
{

    bool domainExtended = domainSize == uint64_t(1 << setupCtx.starkInfo.starkStruct.nBitsExt) ? true : false;

    h_deviceArgs.mapOffsetsExps = domainExtended ? h_deviceArgs.mapOffsetsExtended : h_deviceArgs.mapOffsets;            
    h_deviceArgs.mapOffsetsCustomExps = domainExtended ? h_deviceArgs.mapOffsetsCustomFixedExtended : h_deviceArgs.mapOffsetsCustomFixed;
    h_deviceArgs.nextStridesExps = domainExtended ? h_deviceArgs.nextStridesExtended : h_deviceArgs.nextStrides;

    h_deviceArgs.k_min = domainExtended
                             ? uint64_t((minRowExtended + nRowsPack - 1) / nRowsPack) * nRowsPack
                             : uint64_t((minRow + nRowsPack - 1) / nRowsPack) * nRowsPack;
    h_deviceArgs.k_max = domainExtended
                             ? uint64_t(maxRowExtended / nRowsPack) * nRowsPack
                             : uint64_t(maxRow / nRowsPack) * nRowsPack;

    ParserParams parserParams[dest.params.size()];

    h_deviceArgs.maxNTemp1 = 0;
    h_deviceArgs.maxNTemp3 = 0;

    for (uint64_t k = 0; k < dest.params.size(); ++k)
    {
        ParserParams &parserParams = setupCtx.expressionsBin.expressionsInfo[dest.params[k].expId];
        if (parserParams.nTemp1)
        {
            h_deviceArgs.maxNTemp1 = parserParams.nTemp1;
        }
        if (parserParams.nTemp3)
        {
            h_deviceArgs.maxNTemp3 = parserParams.nTemp3;
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
    cudaMalloc(&h_deviceArgs.dest_params, h_deviceArgs.dest_nParams * sizeof(DestParamsGPU));
    cudaMemcpy(h_deviceArgs.dest_params, aux_params, h_deviceArgs.dest_nParams * sizeof(DestParamsGPU), cudaMemcpyHostToDevice);
    delete[] aux_params;

    // Allocate memory for the struct on the device
    cudaMalloc(&d_deviceArgs, sizeof(DeviceArguments));
    cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArguments), cudaMemcpyHostToDevice);
}

void ExpressionsGPU::calculateExpressions_gpu(StepsParams &d_params, Dest dest, uint64_t domainSize, bool domainExtended)
{

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();
    loadDeviceArgs(domainSize, d_params, dest);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    std::cout << "goal2_ setBufferTInfo time: " << time << std::endl;

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    dim3 nBlocks =  h_deviceArgs.nBlocks;
    dim3 nThreads = h_deviceArgs.nRowsPack;
    computeExpressions_<<<nBlocks, nThreads>>>(d_params, d_deviceArgs);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    std::cout << "goal2_ de computeExpressions: " << time << std::endl;

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    
    if (dest.dest != NULL)
    {
        cudaMemcpy(dest.dest, dest.dest_gpu, dest.domainSize * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
    }
    
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    std::cout << "goal2_ de cudaMemcpy dests time: " << time << std::endl;

    
    cudaFree(h_deviceArgs.dest_params);
    cudaFree(d_deviceArgs);
}

__device__ __noinline__ Goldilocks::Element*  load__(DeviceArguments *d_deviceArgs, Goldilocks::Element *value, StepsParams& d_params, Goldilocks::Element** expressions_params, uint16_t* args, uint64_t i_args, uint64_t row, uint64_t dim, bool isCyclic) {        
    
    uint32_t r = row + threadIdx.x;
    uint32_t nStages = d_deviceArgs->nStages;
    uint64_t type = args[i_args];
    if (type == 0) {
        if(dim == FIELD_EXTENSION) { assert(0); }
        Goldilocks::Element *constPols = d_deviceArgs->domainExtended ?  &d_params.pConstPolsExtendedTreeAddress[2] :  d_params.pConstPolsAddress;
        uint64_t stagePos = args[i_args + 1];
        int64_t o = d_deviceArgs->nextStridesExps[args[i_args + 2]];
        uint64_t nCols = d_deviceArgs->mapSectionsN[0];
        if(isCyclic) {
            uint64_t l = (r + o) % d_deviceArgs->domainSize;
            value[threadIdx.x] = constPols[l * nCols + stagePos];
            return value;
        } else {
            value[threadIdx.x] = constPols[(r + o)*nCols];
            return value;
        }
    } else if (type <= nStages + 1) { //rick: nStages1
        uint64_t stagePos = args[i_args + 1];
        uint64_t offset = d_deviceArgs->mapOffsetsExps[type];
        uint64_t nCols = d_deviceArgs->mapSectionsN[type];
        int64_t o = d_deviceArgs->nextStridesExps[args[i_args + 2]];
        if(isCyclic) {
            uint64_t l = (r + o) % d_deviceArgs->domainSize;
            if(type == 1 && !d_deviceArgs->domainExtended) {
                value[threadIdx.x] = d_params.trace[l * nCols + stagePos];
            } else {
                for(uint64_t d = 0; d < dim; ++d) {
                    value[threadIdx.x + d*blockDim.x] = d_params.aux_trace[offset + l * nCols + stagePos + d];
                }
            }
        } else {
            if(type == 1 && !d_deviceArgs->domainExtended) {
                value[threadIdx.x] = d_params.trace[(r + o)*nCols+ stagePos];
                
            } else {
                for(uint64_t d = 0; d < dim; ++d) {
                    value[threadIdx.x + d*blockDim.x] = d_params.aux_trace[offset + (r + o) *nCols + stagePos + d];
                }                        
            }
        }
        return value;
    } else if (type == nStages + 2) {
        uint64_t boundary = args[i_args + 1];        
        if(boundary == 0) {
            Goldilocks::Element *x = d_deviceArgs->domainExtended ? &d_deviceArgs->x[row] : &d_deviceArgs->x_n[row];
            return x;
        } else {
            return &d_deviceArgs->zi[(boundary - 1)*d_deviceArgs->domainSize  + row];
        }
        
    } else if (type == nStages + 3) {
        if(dim == 1) { assert(0); }
        /*uint64_t o = args[i_args + 1];
        gl64_t xdivxsub[FIELD_EXTENSION];
        //Goldilocks3::sub((Goldilocks3::Element &)(xdivxsub[k * FIELD_EXTENSION]), proverHelpers.x[row + k], (Goldilocks3::Element &)(xis[o * FIELD_EXTENSION]));
        Goldilocks3GPU::op_31_gpu(2, (gl64_t*)xdivxsub, (gl64_t*)(xis[o * FIELD_EXTENSION]), true, (gl64_t*)(&d_deviceArgs->x_n[r]), true);
        

        /*Goldilocks3::batchInverse((Goldilocks3::Element *)xdivxsub, (Goldilocks3::Element *)xdivxsub, nrowsPack);

        for (uint64_t k = 0; k < nrowsPack; k++) {
            Goldilocks3::mul((Goldilocks3::Element &)(xdivxsub[k * FIELD_EXTENSION]), (Goldilocks3::Element &)(xdivxsub[k * FIELD_EXTENSION]), proverHelpers.x[row + k]);
        }

        for(uint64_t k = 0; k < nrowsPack; ++k) {
            for(uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
                value[k + e*nrowsPack] = xdivxsub[k * FIELD_EXTENSION + e];
            }
        }*/
        return value;
    } else if (type >= nStages + 4 && type < d_deviceArgs->nCustomCommits + nStages + 4) {
        uint64_t index = type - (nStages + 4);
        uint64_t stagePos = args[i_args + 1];
        uint64_t offset = d_deviceArgs->mapOffsetsCustomExps[index];
        uint64_t nCols = d_deviceArgs->mapSectionsNCustomFixed[index];
        int64_t o = d_deviceArgs->nextStridesExps[args[i_args + 2]];
        if(isCyclic) {
            uint64_t l = (r + o) % d_deviceArgs->domainSize;
            value[threadIdx.x] = d_params.pCustomCommitsFixed[offset + l * nCols + stagePos];
        } else {
            value[threadIdx.x] = d_params.pCustomCommitsFixed[offset + (r + o) * nCols + stagePos];            
        }
        return value;
    } else if (type ==  d_deviceArgs->bufferCommitSize || type == d_deviceArgs->bufferCommitSize + 1) {
        return &expressions_params[type][args[i_args + 1]*blockDim.x];
    } else {
        return &expressions_params[type][args[i_args + 1]];
    }
}

__device__ __noinline__ void storePolynomial__(DeviceArguments *d_deviceArgs, Goldilocks::Element *destVals, uint64_t row)
{
    int tid = threadIdx.x;
    if (row + blockDim.x > d_deviceArgs->dest_domainSize)
    {
        return;
    }
    if (d_deviceArgs->dest_dim == 1)
    {
        uint64_t offset = d_deviceArgs->dest_offset != 0 ? d_deviceArgs->dest_offset : 1;
        if(row + threadIdx.x < d_deviceArgs->dest_domainSize) {
            d_deviceArgs->dest_gpu[(row + threadIdx.x) * offset ] = destVals[threadIdx.x];
        }
    }
    else
    {
        uint64_t offset = d_deviceArgs->dest_offset != 0 ? d_deviceArgs->dest_offset : FIELD_EXTENSION;
        if(row + threadIdx.x < d_deviceArgs->dest_domainSize) {
            d_deviceArgs->dest_gpu[(row + threadIdx.x) * offset] = destVals[threadIdx.x];
            d_deviceArgs->dest_gpu[(row + threadIdx.x) * offset + 1] = destVals[blockDim.x + threadIdx.x];
            d_deviceArgs->dest_gpu[(row + threadIdx.x) * offset + 2] = destVals[2 * blockDim.x + threadIdx.x];
        }
    }
}

__device__ __noinline__ void multiplyPolynomials__(DeviceArguments *d_deviceArgs, gl64_t *destVals)
{
    if (d_deviceArgs->dest_dim == 1)
    {
        gl64_t::op_gpu(2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * blockDim.x], false);
    }
    else
    {
        assert(blockDim.x <= 512);
        __shared__ gl64_t vals[FIELD_EXTENSION * 512]; // rick: corregir
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
            Goldilocks3GPU::op_31_gpu(2, &vals[0], &destVals[FIELD_EXTENSION * d_deviceArgs->nRowsPack], false, &destVals[0], false);
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

__device__ __noinline__ bool caseNoOprations__(StepsParams &d_params, DeviceArguments *d_deviceArgs, Goldilocks::Element *destVals, uint32_t k, uint64_t row)
{

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
            destVals[r] = d_params.pConstPolsAddress[l * nCols + stagePos];
        }
        else
        {
            uint64_t offset = d_deviceArgs->mapOffsetsExps[d_deviceArgs->dest_params[k].stage];
            uint64_t nCols = d_deviceArgs->mapSectionsN[d_deviceArgs->dest_params[k].stage];
            if (d_deviceArgs->dest_params[k].stage == 1)
            {
                destVals[r] = d_params.trace[l * nCols + stagePos];
            }
            else
            {
                for (uint64_t d = 0; d < d_deviceArgs->dest_params[k].dim; ++d)
                {
                    destVals[r + d * blockDim.x] = d_params.aux_trace[offset + l * nCols + stagePos + d];
                }
            }
        }

        if (d_deviceArgs->dest_params[k].inverse)
        {
            getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_deviceArgs->dest_params[k].dim);
        }
        return true;
    }
    else if (d_deviceArgs->dest_params[k].op == opType::number)
    {
        destVals[k * FIELD_EXTENSION * blockDim.x + r].fe = d_deviceArgs->dest_params[k].value;
        return true;
    }
    else if (d_deviceArgs->dest_params[k].op == opType::airvalue)
    {
        destVals[k * FIELD_EXTENSION * blockDim.x + r] = d_params.airValues[d_deviceArgs->dest_params[k].polsMapId];
        destVals[k * FIELD_EXTENSION * blockDim.x + r + blockDim.x] = d_params.airValues[d_deviceArgs->dest_params[k].polsMapId + 1];
        destVals[k * FIELD_EXTENSION * blockDim.x + r + 2 * blockDim.x] = d_params.airValues[d_deviceArgs->dest_params[k].polsMapId + 2];
        return true;
    }
    return false;
}

__global__ __launch_bounds__(128) void computeExpressions_(StepsParams &d_params, DeviceArguments *d_deviceArgs)
{

    int chunk_idx = blockIdx.x;
    assert(d_deviceArgs->nRowsPack == blockDim.x);
    uint64_t nchunks = d_deviceArgs->domainSize / blockDim.x;
    extern __shared__ gl64_t values_[];

    // aixo s'ha de fer en el setbuffers
    uint32_t bufferCommitsSize = d_deviceArgs->bufferCommitSize;
    assert(bufferCommitsSize  + 9 < 30);
    __shared__ Goldilocks::Element *expressions_params[30];

    if (threadIdx.x == 0)
    {
        // rick: falta tmp1 i tmp3
        expressions_params[bufferCommitsSize + 0] = (&d_deviceArgs->tmp1[blockDim.x * d_deviceArgs->maxNTemp1 * blockDim.x]); //nTemp1
        expressions_params[bufferCommitsSize + 1] = (&d_deviceArgs->tmp3[blockDim.x * d_deviceArgs->maxNTemp3 * blockDim.x * FIELD_EXTENSION]); //nTemp3
        expressions_params[bufferCommitsSize + 2] = d_params.publicInputs;
        expressions_params[bufferCommitsSize + 3] = d_deviceArgs->numbers;
        expressions_params[bufferCommitsSize + 4] = d_params.airValues;
        expressions_params[bufferCommitsSize + 5] = d_params.proofValues;
        expressions_params[bufferCommitsSize + 6] = d_params.airgroupValues;
        expressions_params[bufferCommitsSize + 7] = d_params.challenges;
        expressions_params[bufferCommitsSize + 8] = d_params.evals;
    }
    __syncthreads();
    Goldilocks::Element *destVals = &d_deviceArgs->destVals[blockIdx.x * d_deviceArgs->dest_nParams * blockDim.x]; 

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool isCyclic = i < d_deviceArgs->k_min || i >= d_deviceArgs->k_max;
#pragma unroll 1
        for (uint64_t k = 0; k < d_deviceArgs->dest_nParams; ++k)
        {

            if(caseNoOprations__(d_params,d_deviceArgs, destVals, k, i)){
                continue;
            }
            uint8_t *ops = &d_deviceArgs->ops[d_deviceArgs->dest_params[k].opsOffset];
            uint16_t *args = &d_deviceArgs->args[d_deviceArgs->dest_params[k].argsOffset];
            Goldilocks::Element *valueA = (Goldilocks::Element *)(&values_[0]);
            Goldilocks::Element *valueB = (Goldilocks::Element *)(&values_[blockDim.x * FIELD_EXTENSION]);

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
                    gl64_t::copy_gpu(res, a, isConstant);
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
                    gl64_t::op_gpu( args[i_args], res, a, isConstantA, b, isConstantB);
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
                    Goldilocks3GPU::op_31_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
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
                    Goldilocks3GPU::op_gpu(args[i_args], res, a, isConstantA, b, isConstantB);
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
                printf(" %d consumed args - %d expected args \n", i_args, d_deviceArgs->dest_params[k].nArgs);
                assert(0);
            }
            if (d_deviceArgs->dest_params[k].inverse)
            {
                getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_deviceArgs->dest_params[k].dim);
            }
            
        }

        if (d_deviceArgs->dest_nParams == 2)
        {
            multiplyPolynomials__(d_deviceArgs, (gl64_t*) destVals);
        }
        storePolynomial__(d_deviceArgs, destVals, i);

        chunk_idx += gridDim.x;
    }
}
