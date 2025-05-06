#include "hints.hpp"
#include "expressions_gpu.cuh"
#include "expressions_pack.hpp"
#include "polinomial.hpp"

void opHintFieldsGPU(StepsParams *d_params, Dest &dest, uint64_t nRows, bool domainExtended, void* GPUExpressionsCtx, cudaStream_t stream){

    ExpressionsGPU* expressionsCtx = (ExpressionsGPU*)GPUExpressionsCtx;
    expressionsCtx->calculateExpressions_gpu( d_params, dest, nRows, domainExtended, stream);
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
}

void copyValueGPU( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size){
    CHECKCUDAERR(cudaMemcpy(target, src, size * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
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

uint64_t setHintFieldGPU(SetupCtx& setupCtx, StepsParams& params, Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName) {
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
        setPolynomialGPU(setupCtx, params.aux_trace, values, hintFieldVal.id);
    } else if(hintFieldVal.operand == opType::airgroupvalue) {
        if(setupCtx.starkInfo.airgroupValuesMap[hintFieldVal.id].stage > 1) {
            copyValueGPU(params.airgroupValues + FIELD_EXTENSION*hintFieldVal.id, values, FIELD_EXTENSION);      
        } else {
            copyValueGPU(params.airgroupValues + FIELD_EXTENSION*hintFieldVal.id, values, 1);
        }
    } else if(hintFieldVal.operand == opType::airvalue) {
        if(setupCtx.starkInfo.airValuesMap[hintFieldVal.id].stage > 1) {
            copyValueGPU(params.airValues + FIELD_EXTENSION*hintFieldVal.id, values, FIELD_EXTENSION);
        } else {
            copyValueGPU(params.airValues + FIELD_EXTENSION*hintFieldVal.id, values, 1);
        }
    } else {
        zklog.error("Only committed pols and airgroupvalues can be set");
        exitProcess();
        exit(-1);  
    }

    return hintFieldVal.id;
}

void multiplyHintFieldsGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams &d_params, ExpressionsCtx& expressionsCtx, uint64_t nHints, uint64_t* hintId, std::string *hintFieldNameDest, std::string* hintFieldName1, std::string* hintFieldName2,  HintFieldOptions *hintOptions1, HintFieldOptions *hintOptions2, void* GPUExpressionsCtx, double* time_expressions, cudaStream_t stream) {
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
        double time_start = omp_get_wtime();
        opHintFieldsGPU(&d_params, destStruct, nRows, false, GPUExpressionsCtx, stream);
        *time_expressions += omp_get_wtime() - time_start;
    }
}

void accMulHintFieldsGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams &d_params, ExpressionsCtx &expressionsCtx, uint64_t hintId, std::string hintFieldNameDest, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2, bool add, void* GPUExpressionsCtx, double* time_expressions, cudaStream_t stream) {
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    Hint hint = setupCtx.expressionsBin.hints[hintId];

    auto hintFieldDest = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldNameDest](const HintField& hintField) {
        return hintField.name == hintFieldNameDest;
    });
    HintFieldValue hintFieldDestVal = hintFieldDest->values[0];

    uint64_t dim = setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].dim;
    
    uint64_t offsetAuxTrace = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    Goldilocks::Element* vals = new Goldilocks::Element[dim*N];
    
    Dest destStruct(vals, 1 << setupCtx.starkInfo.starkStruct.nBits, 0);
    destStruct.dest_gpu = h_params.aux_trace + offsetAuxTrace;
    addHintField(setupCtx, h_params, hintId, destStruct, hintFieldName1, hintOptions1);
    addHintField(setupCtx, h_params, hintId, destStruct, hintFieldName2, hintOptions2);

    double time_start = omp_get_wtime();
    opHintFieldsGPU(&d_params, destStruct, N, false, GPUExpressionsCtx, stream);
    *time_expressions += omp_get_wtime() - time_start; 
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
    setHintFieldGPU(setupCtx, h_params, vals, hintId, hintFieldNameDest);
    setHintFieldGPU(setupCtx, h_params, &vals[(N - 1)*FIELD_EXTENSION], hintId, hintFieldNameAirgroupVal);

    delete[] vals;
}

uint64_t updateAirgroupValueGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams &d_params, uint64_t hintId, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2, bool add, void* GPUExpressionsCtx, double* time_expressions, cudaStream_t stream) {
    
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

    double time_start = omp_get_wtime();
    opHintFieldsGPU(&d_params, destStruct, 1, false, GPUExpressionsCtx, stream); 
    opAirgroupValueGPU(h_params.airgroupValues + FIELD_EXTENSION*hintFieldAirgroupVal.id, destStruct.dest_gpu, destStruct.dim, add);
    *time_expressions += omp_get_wtime() - time_start;
    return hintFieldAirgroupVal.id;
}


