#include "hints.hpp"
#include "expressions_gpu.cuh"
#include "polinomial.hpp"

void opHintFieldsGPU(StepsParams *d_params, Dest &dest, uint64_t nRows, bool domainExtended, void* GPUExpressionsCtx){

    ExpressionsGPU* expressionsCtx = (ExpressionsGPU*)GPUExpressionsCtx;
    expressionsCtx->calculateExpressions_gpu( d_params, dest, nRows, domainExtended);
}

void allocateDestGPU(Goldilocks::Element**buff, uint64_t size){
    cudaMalloc((void**) buff, size * sizeof(Goldilocks::Element));
}
void freeDestGPU(Goldilocks::Element* buff){
    CHECKCUDAERR(cudaDeviceSynchronize());
    CHECKCUDAERR(cudaFree(buff));
}
__global__ void setPolynomial_(Goldilocks::Element *pol, Goldilocks::Element *values, uint64_t deg, uint64_t dim, uint64_t nCols) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < deg) {
        for (uint64_t j = 0; j < dim; ++j) {
            pol[i * nCols + j] = values[i * dim + j];
        }
    }
}

void setPolynomialGPU(SetupCtx& setupCtx, Goldilocks::Element *aux_trace, Goldilocks::Element *values, uint64_t idPol) {
    PolMap polInfo = setupCtx.starkInfo.cmPolsMap[idPol];
    uint64_t deg = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t dim = polInfo.dim;
    std::string stage = "cm" + to_string(polInfo.stage);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
    uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, false)];
    offset += polInfo.stagePos;
    uint64_t offset_values = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    
    // copy values into the GPU
    Goldilocks::Element* d_values = aux_trace + offset_values;
    CHECKCUDAERR(cudaMemcpy(d_values, values, deg * dim * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));

    dim3 threds(512);
    dim3 blocks((deg + threds.x - 1) / threds.x);
    setPolynomial_<<<blocks, threds>>>(aux_trace + offset, d_values, deg, dim, nCols);
    CHECKCUDAERR(cudaDeviceSynchronize());
    
}



void copyValueGPU( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size){
    CHECKCUDAERR(cudaMemcpy(target, src, size * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
}
void copyValueHost( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size){
    CHECKCUDAERR(cudaMemcpy(target, src, size * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
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
void opAirgroupValueGPU(Goldilocks::Element * airgroupValue,  Goldilocks::Element* val, uint32_t dim, bool add){
    opAirgroupValue_<<<1, 1>>>((gl64_t*)airgroupValue, (gl64_t*)val, dim, add);
}

void copyValueGPUGPU( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size){
    CHECKCUDAERR(cudaMemcpy(target, src, size * sizeof(Goldilocks::Element), cudaMemcpyDeviceToDevice));
}




