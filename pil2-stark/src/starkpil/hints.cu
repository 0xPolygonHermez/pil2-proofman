#include "hints.cuh"
#include "expressions_gpu.cuh"
#include "expressions_pack.hpp"
#include "polinomial.hpp"

void opHintFieldsGPU(StepsParams *d_params, Dest &dest, uint64_t nRows, bool domainExtended, void* GPUExpressionsCtx, TimerGPU &timer, cudaStream_t stream){

    ExpressionsGPU* expressionsCtx = (ExpressionsGPU*)GPUExpressionsCtx;
    expressionsCtx->calculateExpressions_gpu( d_params, dest, nRows, domainExtended, timer, stream);
}

__global__ void setPolynomial_(Goldilocks::Element *pol, Goldilocks::Element *values, uint64_t deg, uint64_t dim, uint64_t nCols) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < deg) {
        for (uint64_t j = 0; j < dim; ++j) {
            pol[i * nCols + j] = values[i * dim + j];
        }
    }
}

void setPolynomialGPU(SetupCtx& setupCtx, Goldilocks::Element *aux_trace, Goldilocks::Element *values, uint64_t idPol, cudaStream_t stream) {
    PolMap polInfo = setupCtx.starkInfo.cmPolsMap[idPol];
    uint64_t deg = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t dim = polInfo.dim;
    std::string stage = "cm" + to_string(polInfo.stage);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
    uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, false)];
    offset += polInfo.stagePos;
    
    dim3 threads(512);
    dim3 blocks((deg + threads.x - 1) / threads.x);
    setPolynomial_<<<blocks, threads, 0, stream>>>(aux_trace + offset, values, deg, dim, nCols);    
}

void copyValueGPU( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size, cudaStream_t stream) {
    CHECKCUDAERR(cudaMemcpyAsync(target, src, size * sizeof(Goldilocks::Element), cudaMemcpyDeviceToDevice, stream));
}

__global__ void opAirgroupValue_(gl64_t * airgroupValue,  gl64_t* val, uint32_t dim, bool add){
    
    if(add){
        if(dim == 1){
            airgroupValue[0] += val[0];
        } else {
            airgroupValue[0] += val[0];
            airgroupValue[1] += val[1];
            airgroupValue[2] += val[2];
        }
    }else{
        if (dim ==1)
        {
            airgroupValue[0] *= val[0];
            
        }else{
            Goldilocks3GPU::mul( (Goldilocks3GPU::Element*)airgroupValue, (Goldilocks3GPU::Element*)airgroupValue, (Goldilocks3GPU::Element*)val);
        }
    }
        
}
void opAirgroupValueGPU(Goldilocks::Element * airgroupValue,  Goldilocks::Element* val, uint32_t dim, bool add, cudaStream_t stream){
    opAirgroupValue_<<<1, 1, 0, stream>>>((gl64_t*)airgroupValue, (gl64_t*)val, dim, add);
}

uint64_t setHintFieldGPU(SetupCtx& setupCtx, StepsParams& params, Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName, cudaStream_t stream) {
    Hint hint = setupCtx.expressionsBin.hints[hintId];

    auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldName](const HintField& hintField) {
        return hintField.name == hintFieldName;
    });

    if(hintField == hint.fields.end()) {
        zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
        exitProcess();
        exit(-1);
    }

    if(hintField->values.size() != 1) {
        zklog.error("Hint field " + hintFieldName + " in " + hint.name + "has more than one destination.");
        exitProcess();
        exit(-1);
    }

    auto hintFieldVal = hintField->values[0];
    if(hintFieldVal.operand == opType::cm) {
        setPolynomialGPU(setupCtx, params.aux_trace, values, hintFieldVal.id, stream);
    } else if(hintFieldVal.operand == opType::airgroupvalue) {
        if(setupCtx.starkInfo.airgroupValuesMap[hintFieldVal.id].stage > 1) {
            copyValueGPU(params.airgroupValues + FIELD_EXTENSION*hintFieldVal.id, values, FIELD_EXTENSION, stream);      
        } else {
            copyValueGPU(params.airgroupValues + FIELD_EXTENSION*hintFieldVal.id, values, 1, stream);
        }
    } else if(hintFieldVal.operand == opType::airvalue) {
        if(setupCtx.starkInfo.airValuesMap[hintFieldVal.id].stage > 1) {
            copyValueGPU(params.airValues + FIELD_EXTENSION*hintFieldVal.id, values, FIELD_EXTENSION, stream);
        } else {
            copyValueGPU(params.airValues + FIELD_EXTENSION*hintFieldVal.id, values, 1, stream);
        }
    } else {
        zklog.error("Only committed pols and airgroupvalues can be set");
        exitProcess();
        exit(-1);  
    }

    return hintFieldVal.id;
}

void multiplyHintFieldsGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams &d_params, uint64_t nHints, uint64_t* hintId, std::string *hintFieldNameDest, std::string* hintFieldName1, std::string* hintFieldName2,  HintFieldOptions *hintOptions1, HintFieldOptions *hintOptions2, void* GPUExpressionsCtx, TimerGPU &timer, cudaStream_t stream) {
    if(setupCtx.expressionsBin.hints.size() == 0) {
        zklog.error("No hints were found.");
        exitProcess();
        exit(-1);
    }

    std::vector<Dest> dests;
    Goldilocks::Element *buff = NULL;

    for(uint64_t i = 0; i < nHints; ++i) {
        Hint hint = setupCtx.expressionsBin.hints[hintId[i]];
        Goldilocks::Element *buff_gpu = NULL;

        std::string hintDest = hintFieldNameDest[i];
        auto hintFieldDest = std::find_if(hint.fields.begin(), hint.fields.end(), [hintDest](const HintField& hintField) {
            return hintField.name == hintDest;
        });
        HintFieldValue hintFieldDestVal = hintFieldDest->values[0];

        uint64_t offset = 0;
        uint64_t nRows;
        if(hintFieldDestVal.operand == opType::cm) {
            offset = setupCtx.starkInfo.mapSectionsN["cm" + to_string(setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].stage)];
            uint64_t offsetAuxTrace = setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].stage), false)] + setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].stagePos;           
            buff = NULL;
            buff_gpu = h_params.aux_trace + offsetAuxTrace;
            nRows = 1 << setupCtx.starkInfo.starkStruct.nBits;
        } else if (hintFieldDestVal.operand == opType::airvalue) {
            nRows = 1;
            uint64_t pos = 0;
            for(uint64_t i = 0; i < hintFieldDestVal.id; ++i) {
                pos += setupCtx.starkInfo.airValuesMap[i].stage == 1 ? 1 : FIELD_EXTENSION;
            }
            buff = NULL;
            buff_gpu = h_params.airValues + pos;
        } else {
            zklog.error("Only committed pols and airvalues can be set");
            exitProcess();
            exit(-1);
        }

        Dest destStruct(buff, nRows, offset);
        destStruct.dest_gpu = buff_gpu;

        addHintField(setupCtx, h_params, hintId[i], destStruct, hintFieldName1[i], hintOptions1[i]);
        addHintField(setupCtx, h_params, hintId[i], destStruct, hintFieldName2[i], hintOptions2[i]);
        
        opHintFieldsGPU(&d_params, destStruct, nRows, false, GPUExpressionsCtx, timer, stream);
    }
}

void accMulHintFieldsGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams &d_params, uint64_t hintId, std::string hintFieldNameDest, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2, bool add, void* GPUExpressionsCtx, TimerGPU &timer, cudaStream_t stream) {
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    Hint hint = setupCtx.expressionsBin.hints[hintId];

    auto hintFieldDest = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldNameDest](const HintField& hintField) {
        return hintField.name == hintFieldNameDest;
    });
    HintFieldValue hintFieldDestVal = hintFieldDest->values[0];

    uint64_t dim = setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].dim;
    
    uint64_t offsetAuxTrace = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    Goldilocks::Element* vals_gpu = h_params.aux_trace + offsetAuxTrace;
    
    Dest destStruct(nullptr, 1 << setupCtx.starkInfo.starkStruct.nBits, 0);
    destStruct.dest_gpu = vals_gpu;
    addHintField(setupCtx, h_params, hintId, destStruct, hintFieldName1, hintOptions1);
    addHintField(setupCtx, h_params, hintId, destStruct, hintFieldName2, hintOptions2);

    opHintFieldsGPU(&d_params, destStruct, N, false, GPUExpressionsCtx, timer, stream);
    
    // copy vals to the GPU
    Goldilocks::Element* helpers = h_params.aux_trace + offsetAuxTrace + dim*N;    
    accOperationGPU((gl64_t *)vals_gpu, N, add, dim, (gl64_t *)helpers, stream);

    setHintFieldGPU(setupCtx, h_params, vals_gpu, hintId, hintFieldNameDest,stream);
    setHintFieldGPU(setupCtx, h_params, &vals_gpu[(N - 1)*FIELD_EXTENSION], hintId, hintFieldNameAirgroupVal, stream);
}

uint64_t updateAirgroupValueGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams &d_params, uint64_t hintId, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2, bool add, void* GPUExpressionsCtx, TimerGPU &timer, cudaStream_t stream) {
    
    Hint hint = setupCtx.expressionsBin.hints[hintId];

    auto hintFieldAirgroup = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldNameAirgroupVal](const HintField& hintField) {
        return hintField.name == hintFieldNameAirgroupVal;
    });
    HintFieldValue hintFieldAirgroupVal = hintFieldAirgroup->values[0];

    Goldilocks::Element vals[3];
    
    Dest destStruct(vals, 1, 0);
    uint64_t offsetAuxTrace = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    destStruct.dest_gpu = h_params.aux_trace + offsetAuxTrace;
    destStruct.dest = nullptr;
    addHintField(setupCtx, h_params, hintId, destStruct, hintFieldName1, hintOptions1);
    addHintField(setupCtx, h_params, hintId, destStruct, hintFieldName2, hintOptions2);

    opHintFieldsGPU(&d_params, destStruct, 1, false, GPUExpressionsCtx, timer, stream); 
    opAirgroupValueGPU(h_params.airgroupValues + FIELD_EXTENSION*hintFieldAirgroupVal.id, destStruct.dest_gpu, destStruct.dim, add, stream);
    return hintFieldAirgroupVal.id;
}

 
void accOperation(Goldilocks::Element* vals, uint64_t N, bool add, uint32_t dim) {
    for(uint64_t i = 1; i < N; ++i) {
        if(add) {
            if(dim == 1) {
                Goldilocks::add(vals[i], vals[i], vals[(i - 1)]);
            } else {
                Goldilocks3::add((Goldilocks3::Element &)vals[i * FIELD_EXTENSION], (Goldilocks3::Element &)vals[i * FIELD_EXTENSION], (Goldilocks3::Element &)vals[(i - 1) * FIELD_EXTENSION]);
            }
        } else {
            if(dim == 1) {
                Goldilocks::mul(vals[i], vals[i], vals[(i - 1)]);
            } else {
                Goldilocks3::mul((Goldilocks3::Element &)vals[i * FIELD_EXTENSION], (Goldilocks3::Element &)vals[i * FIELD_EXTENSION], (Goldilocks3::Element &)vals[(i - 1) * FIELD_EXTENSION]);
            }
        }
    }

}


//
//Algorithm based in: https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
//
// todo: bank collisions pending

__device__ void scan_sum_1(gl64_t* temp, uint32_t N){

    uint32_t thid = threadIdx.x;
    int offset = 1;

    // build sum in place up the tree 
    for (int d = N>>1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // clear the last element
    if (thid == 0) { 
        temp[(N- 1)].set_val(0); 
    } 

    // traverse down tree & build scan
    for (int d = 1; d < N; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            gl64_t t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
}

__device__ void scan_prod_1(gl64_t* temp, uint32_t N){

    uint32_t thid = threadIdx.x;
    int offset = 1;

    // build sum in place up the tree 
    for (int d = N>>1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] *= temp[ai];
        }
        offset *= 2;
    }
    
    // clear the last element
    if (thid == 0) { 
        temp[(N- 1)].set_val(1); 
    } 

    // traverse down tree & build scan
    for (int d = 1; d < N; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            gl64_t t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] *= t;
        }
    }
}

__device__ void scan_sum_3(gl64_t* temp, uint32_t N){
    
    uint32_t thid = threadIdx.x;
    int offset = 1;

    // build sum in place up the tree 
    for (int d = N>>1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = 3 * (offset*(2*thid+1)-1);
            int bi = 3 * (offset*(2*thid+2)-1);
            assert(bi < 6*blockDim.x);
            assert(ai < 6*blockDim.x);
            Goldilocks3GPU::add(*((Goldilocks3GPU::Element *)&temp[bi]), *((Goldilocks3GPU::Element *)&temp[bi]), *((Goldilocks3GPU::Element *)&temp[ai]));
        }
        offset *= 2;
    }
    __syncthreads(); 
    // clear the last element
    if (thid == 0) { 
        temp[3*(N- 1)].set_val(0); 
        temp[3*(N- 1)+1].set_val(0);
        temp[3*(N- 1)+2].set_val(0);
    } 

    // traverse down tree & build scan
    for (int d = 1; d < N; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = 3 * (offset*(2*thid+1)-1);
            int bi = 3 * (offset*(2*thid+2)-1);
            Goldilocks3GPU::Element t;
            Goldilocks3GPU::copy((Goldilocks3GPU::Element *)&t, (Goldilocks3GPU::Element *)&temp[ai]);
            Goldilocks3GPU::copy((Goldilocks3GPU::Element *)&temp[ai], (Goldilocks3GPU::Element *)&temp[bi]);
            Goldilocks3GPU::add(*((Goldilocks3GPU::Element *)&temp[bi]), *((Goldilocks3GPU::Element *)&temp[bi]), t);
        }
    }

}

__device__ void scan_prod_3(gl64_t* temp, uint32_t N){
    
    uint32_t thid = threadIdx.x;
    int offset = 1;

    // build sum in place up the tree 
    for (int d = N>>1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = 3 * (offset*(2*thid+1)-1);
            int bi = 3 * (offset*(2*thid+2)-1);
            Goldilocks3GPU::mul(*((Goldilocks3GPU::Element *)&temp[bi]), *((Goldilocks3GPU::Element *)&temp[bi]), *((Goldilocks3GPU::Element *)&temp[ai]));
        }
        offset *= 2;
    }
    __syncthreads(); 
    // clear the last element
    if (thid == 0) { 
        temp[3*(N- 1)].set_val(1); 
        temp[3*(N- 1)+1].set_val(0);
        temp[3*(N- 1)+2].set_val(0);
    } 

    // traverse down tree & build scan
    for (int d = 1; d < N; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = 3 * (offset*(2*thid+1)-1);
            int bi = 3 * (offset*(2*thid+2)-1);
            Goldilocks3GPU::Element t;
            Goldilocks3GPU::copy((Goldilocks3GPU::Element *)&t, (Goldilocks3GPU::Element *)&temp[ai]);
            Goldilocks3GPU::copy((Goldilocks3GPU::Element *)&temp[ai], (Goldilocks3GPU::Element *)&temp[bi]);
            Goldilocks3GPU::mul(*((Goldilocks3GPU::Element *)&temp[bi]), *((Goldilocks3GPU::Element *)&temp[bi]), t);
        }
    }

}

__global__ void prescan(gl64_t *g_odata, gl64_t *g_idata, bool isSum, uint32_t chunk, uint32_t dim, uint32_t N)
{
    extern __shared__ gl64_t temp[]; 

    uint32_t thid = threadIdx.x;
    uint32_t indx1 = (blockIdx.x * blockDim.x + thid)*2;
    uint32_t indx2 = (blockIdx.x * blockDim.x + thid)*2 + 1;
    uint32_t pos1 = (indx1+1) * chunk -1;
    uint32_t pos2 = (indx2+1) * chunk -1;

    uint32_t dimx2xthid = (dim << 1) * thid;
    uint32_t dimxpos1 = dim * pos1;
    uint32_t dimxpos2 = dim * pos2;

    for(uint32_t i=0; i<dim; i++){
        temp[dimx2xthid+i] = g_idata[dimxpos1+i]; 
        temp[dimx2xthid+dim+i] = g_idata[dimxpos2+i]; 
    }

    // build sum in place up the tree
    if(isSum) {
        if(dim == 1) {
           scan_sum_1(temp, blockDim.x*2);
        } else {
           scan_sum_3(temp, blockDim.x*2);
        }
    } else {
        if(dim == 1) {
            scan_prod_1(temp, blockDim.x*2);
        } else {
            scan_prod_3(temp, blockDim.x*2);
        }
    }
    
    __syncthreads();
    // exclusive scan has been evaluated but wee need inclusive
    
    uint32_t dimxindx1 = dim * indx1;
    uint32_t dimxindx2 = dim * indx2;
    uint32_t indxtmp1 = dimx2xthid+dim;
    uint32_t indxtmp2 = dimx2xthid+(dim << 1);
    if(indxtmp2 < dim*2*blockDim.x){
        for(uint32_t i=0; i<dim; i++){
            g_odata[dimxindx1+i] = temp[indxtmp1+i];
            g_odata[dimxindx2+i] = temp[indxtmp2+i];
        }
    } else{
        assert(indxtmp2 == dim*2*blockDim.x);
        for(uint32_t i=0; i<dim; i++){
            g_odata[dimxindx1+i] = temp[indxtmp1+i];
        }
        indxtmp2 -= dim;
        if(dim == 1){
            if(isSum)
                g_odata[dimxindx2] = temp[indxtmp2] + g_idata[dimxpos2];
            else{
                g_odata[dimxindx2] = temp[indxtmp2] * g_idata[dimxpos2];
            }
        } else {
            if(isSum){
                Goldilocks3GPU::add(*((Goldilocks3GPU::Element *)&g_odata[dimxindx2]), *((Goldilocks3GPU::Element *)&temp[indxtmp2]), *((Goldilocks3GPU::Element *)&g_idata[dimxpos2]));
            } else {
                Goldilocks3GPU::mul(*((Goldilocks3GPU::Element *)&g_odata[dimxindx2]), *((Goldilocks3GPU::Element *)&temp[indxtmp2]), *((Goldilocks3GPU::Element *)&g_idata[dimxpos2]));
            }
        }
    }
} 

__global__ void prescan_correction(gl64_t *g_odata, gl64_t* correction, bool isSum, uint32_t dim, uint32_t N){

    if(blockIdx.x == 0) {
        return;
    }
    uint32_t pos_out = (blockIdx.x * blockDim.x + threadIdx.x) * dim;
    uint32_t pos_corr = (blockIdx.x-1) * dim;

    if(isSum) {
        if(dim == 1) {
            g_odata[pos_out] = g_odata[pos_out] + correction[pos_corr];
        } else {
            Goldilocks3GPU::add(*((Goldilocks3GPU::Element *)&g_odata[pos_out]), *((Goldilocks3GPU::Element *)&g_odata[pos_out]), *((Goldilocks3GPU::Element *)&correction[pos_corr]));
        }
    } else {
        if(dim == 1) {
            g_odata[pos_out] = g_odata[pos_out] * correction[pos_corr];
        } else {
            Goldilocks3GPU::mul(*((Goldilocks3GPU::Element *)&g_odata[pos_out]), *((Goldilocks3GPU::Element *)&g_odata[pos_out]), *((Goldilocks3GPU::Element *)&correction[pos_corr]));
        }
    }
}

void accOperationGPU(gl64_t* vals, uint64_t N, bool add, uint32_t dim, gl64_t* helper, cudaStream_t stream) {    
    gl64_t* helper1;
    gl64_t* helper2;
    uint32_t nthreads1 = min(256, (uint32_t)N>>1);
    dim3 threads1(nthreads1);
    dim3 blocks1((N + 2*threads1.x - 1) / (2*threads1.x));
    prescan<<<blocks1, threads1, 2*dim*threads1.x*sizeof(gl64_t), stream>>>(vals, vals, add, 1, dim, N);  
    if(N > 2*nthreads1){
        helper1 = helper;
        uint32_t N2 = blocks1.x;
        uint32_t nthreads2 = min(256, N2>>1);
        dim3 threads2(nthreads2);
        dim3 blocks2((N2 + 2*threads2.x - 1) / (2*threads2.x));
        prescan<<<blocks2, threads2, 2*dim*threads2.x*sizeof(gl64_t), stream>>>(helper1, vals, add, nthreads1 << 1, dim, N2);
        if(N2 > 2*nthreads2){
            helper2 = helper + dim*N2;
            uint32_t N3 = blocks2.x;
            assert(N3 <= 2048);
            uint32_t nthreads3 = N3 >> 1;
            dim3 threads3(nthreads3);
            dim3 blocks3(1);
            prescan<<<blocks3, threads3, 2*dim*threads3.x*sizeof(gl64_t), stream>>>(helper2, helper1, add, nthreads2 << 1, dim, N3);
            prescan_correction<<<blocks2, 2*threads2.x, 0, stream>>>(helper1, helper2, add, dim, N2);

        }
        prescan_correction<<<blocks1, 2*threads1.x, 0, stream>>>(vals, helper1, add, dim, N);
    }
    CHECKCUDAERR(cudaGetLastError());
}
