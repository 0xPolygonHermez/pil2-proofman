#include "hints.hpp"
#include "expressions_gpu.cuh"

void opHintFieldsGPU(StepsParams *d_params, Dest &dest, uint64_t nRows, bool domainExtended, void* GPUExpressionsCtx){

    ExpressionsGPU* expressionsCtx = (ExpressionsGPU*)GPUExpressionsCtx;
    expressionsCtx->calculateExpressions_gpu( d_params, dest, nRows, domainExtended);
}

void allocateDestGPU(Goldilocks::Element**buff, uint64_t size){
    cudaMalloc((void**) buff, size * sizeof(Goldilocks::Element));
}
void freeDestGPU(Goldilocks::Element* buff){
    CHECKCUDAERR(cudaFree(buff));
}

void setPolynomialGPU(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *values, uint64_t idPol) {
    /*PolMap polInfo = setupCtx.starkInfo.cmPolsMap[idPol];
    uint64_t deg = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t dim = polInfo.dim;
    std::string stage = "cm" + to_string(polInfo.stage);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
    uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, false)];
    offset += polInfo.stagePos;
#pragma omp parallel for
    for(uint64_t j = 0; j < deg; ++j) {
        CHECKCUDAERR(cudaMemcpy(buffer + offset + j * nCols, &values[j * dim], dim * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    }*/

    PolMap polInfo = setupCtx.starkInfo.cmPolsMap[idPol];
    uint64_t deg = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t dim = polInfo.dim;
    std::string stage = "cm" + to_string(polInfo.stage);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
    Goldilocks::Element* auxSection = new Goldilocks::Element[nCols * deg];
    uint64_t sectionOffest = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, false)];
    CHECKCUDAERR(cudaMemcpy(auxSection, buffer + sectionOffest, nCols * deg * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    uint64_t offset = polInfo.stagePos;
    Polinomial pol = Polinomial(&auxSection[offset], deg, dim, nCols, std::to_string(idPol));
#pragma omp parallel for
    for(uint64_t j = 0; j < deg; ++j) {
        std::memcpy(pol[j], &values[j*dim], dim * sizeof(Goldilocks::Element));
    }
    CHECKCUDAERR(cudaMemcpy(buffer + sectionOffest, auxSection, nCols * deg * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    delete[] auxSection;
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




