#ifndef HINTS_GPU_HPP
#define HINTS_GPU_HPP

#include "expressions_ctx.hpp"
#include "expressions_gpu.cuh"
#include <cuda_runtime.h>
#include "gpu_timer.cuh"
#include "hints.hpp"

class gl64_gpu;

void opHintFieldsGPU(StepsParams* d_params, Dest &dest, uint64_t nRows, bool domainExtended, void* GPUExpressionsCtx, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream);
void setPolynomialGPU(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *values, uint64_t idPol, cudaStream_t stream);
void copyValueGPU( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size, cudaStream_t stream);
void opAirgroupValueGPU(Goldilocks::Element * airgroupValue,  Goldilocks::Element* val, uint32_t dim, bool add, cudaStream_t stream);
uint64_t setHintFieldGPU(SetupCtx& setupCtx, StepsParams &params, Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName, cudaStream_t stream);
void multiplyHintFieldsGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams *d_params, uint64_t nHints, uint64_t* hintId, std::string *hintFieldNameDest, std::string* hintFieldName1, std::string* hintFieldName2,  HintFieldOptions *hintOptions1, HintFieldOptions *hintOptions2, void* GPUExpressionsCtx, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream);
void accMulHintFieldsGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams *d_params, uint64_t hintId, std::string hintFieldNameDest, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2, bool add, void* GPUExpressionsCtx, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream);
uint64_t updateAirgroupValueGPU(SetupCtx& setupCtx, StepsParams &h_params, StepsParams *d_params, uint64_t hintId, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2, bool add, void* GPUExpressionsCtx, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream);
void accOperationGPU(gl64_gpu* vals, uint64_t N, bool add, uint32_t dim, gl64_gpu* helper, cudaStream_t stream);

#endif