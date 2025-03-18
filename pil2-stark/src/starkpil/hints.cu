#include "hints.hpp"
#include "expressions_gpu.cuh"

void opHintFieldsGPU(SetupCtx& setupCtx, StepsParams& params, StepsParams& d_params, std::vector<Dest> &dests, void* GPUExpressionsCtx){

    ExpressionsGPU* expressionsCtx = (ExpressionsGPU*)GPUExpressionsCtx;
    uint64_t domainSize = 1 << setupCtx.starkInfo.starkStruct.nBits;
    expressionsCtx->calculateExpressions_gpu(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, domainSize);
}

void allocateDestGPU(Goldilocks::Element**buff, uint64_t size){
    cudaMalloc((void**) buff, size * sizeof(Goldilocks::Element));
}
void freeDestGPU(Goldilocks::Element* buff){
    cudaFree(buff);
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
