#include "verify_constraints.hpp"
#include "gpu_timer.cuh"
#include "gl64_tooling.cuh"
#include "starks_gpu.cuh"
#include "gen_proof.cuh"
#include <algorithm>
#include <vector>

// Helper to check if a value is zero in Goldilocks field (handles alias: 0 and GOLDILOCKS_PRIME both represent 0)
__device__ __forceinline__ bool isGoldilocksZero(uint64_t val) {
    return (val == 0) || (val == GOLDILOCKS_PRIME);
}


template<int DIM>
__global__ void verifyConstraintKernel(
    const Goldilocks::Element* __restrict__ dest,
    uint64_t N,
    uint64_t firstRow,
    uint64_t lastRow,

    uint32_t* __restrict__ d_totalInvalid,

    uint32_t* __restrict__ d_invalidRows,
    uint64_t* __restrict__ d_invalidValues,

    uint32_t halfSample
)
{
    constexpr uint32_t BLOCK_MAX = 256;

    __shared__ uint32_t s_rows[BLOCK_MAX];
    __shared__ uint64_t s_vals[BLOCK_MAX][3];
    __shared__ uint32_t s_count;

    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    bool outOfRange = (row >= N || row < firstRow || row > lastRow);

    uint64_t v0 = 0, v1 = 0, v2 = 0;
    bool invalid = false;

    if (!outOfRange) {
        if constexpr (DIM == 1) {
            v0 = dest[row].fe;
            invalid = !isGoldilocksZero(v0);
        } else {
            uint64_t base = 3 * row;
            v0 = dest[base].fe;
            v1 = dest[base + 1].fe;
            v2 = dest[base + 2].fe;
            invalid = !isGoldilocksZero(v0)
                   | !isGoldilocksZero(v1)
                   | !isGoldilocksZero(v2);
        }
    }

    if (invalid) {
        uint32_t idx = atomicAdd(&s_count, 1);
        
        if (idx < BLOCK_MAX) {
            s_rows[idx] = row;
            s_vals[idx][0] = v0;
            s_vals[idx][1] = v1;
            s_vals[idx][2] = v2;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0) {
        uint32_t base = atomicAdd(d_totalInvalid, s_count);

        for (uint32_t i = 0; i < s_count; i++) {
            uint32_t globalIdx = base + i;

            // First half (stable)
            if (globalIdx < halfSample) {
                d_invalidRows[globalIdx] = s_rows[i];
                d_invalidValues[3 * globalIdx + 0] = s_vals[i][0];
                d_invalidValues[3 * globalIdx + 1] = s_vals[i][1];
                d_invalidValues[3 * globalIdx + 2] = s_vals[i][2];
            }
            // Last half (ring buffer overwrite)
            else {
                uint32_t slot = halfSample + (globalIdx % halfSample);
                d_invalidRows[slot] = s_rows[i];
                d_invalidValues[3 * slot + 0] = s_vals[i][0];
                d_invalidValues[3 * slot + 1] = s_vals[i][1];
                d_invalidValues[3 * slot + 2] = s_vals[i][2];
            }
        }
    }
}

void verifyConstraintGPU(
    Goldilocks::Element* d_dest,
    uint64_t N,
    uint64_t destDim,
    uint64_t firstRow,
    uint64_t lastRow,
    ConstraintInfo& constraintInfo,
    cudaStream_t stream
)
{
    uint32_t maxSample = constraintInfo.n_print_constraints;
    uint32_t halfSample = maxSample / 2;

    uint32_t* d_totalInvalid;
    uint32_t* d_invalidRows;
    uint64_t* d_invalidValues;

    CHECKCUDAERR(cudaMalloc(&d_totalInvalid, sizeof(uint32_t)));
    CHECKCUDAERR(cudaMalloc(&d_invalidRows, maxSample * sizeof(uint32_t)));
    CHECKCUDAERR(cudaMalloc(&d_invalidValues, maxSample * 3 * sizeof(uint64_t)));

    CHECKCUDAERR(cudaMemsetAsync(d_totalInvalid, 0, sizeof(uint32_t), stream));

    uint32_t blockSize = 256;
    uint32_t numBlocks = (N + blockSize - 1) / blockSize;

    if (destDim == 1) {
        verifyConstraintKernel<1>
            <<<numBlocks, blockSize, 0, stream>>>(
                d_dest, N, firstRow, lastRow,
                d_totalInvalid,
                d_invalidRows, d_invalidValues,
                halfSample);
    } else {
        verifyConstraintKernel<3>
            <<<numBlocks, blockSize, 0, stream>>>(
                d_dest, N, firstRow, lastRow,
                d_totalInvalid,
                d_invalidRows, d_invalidValues,
                halfSample);
    }

    uint32_t h_invalidCount = 0;
    CHECKCUDAERR(cudaMemcpyAsync(
        &h_invalidCount, d_totalInvalid,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    constraintInfo.nrows = h_invalidCount;

    uint32_t copyCount = std::min(h_invalidCount, maxSample);
    if (copyCount > 0) {
        std::vector<uint32_t> rows(copyCount);
        std::vector<uint64_t> values(copyCount * 3);

        CHECKCUDAERR(cudaMemcpy(rows.data(), d_invalidRows,
                                copyCount * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost));
        CHECKCUDAERR(cudaMemcpy(values.data(), d_invalidValues,
                                copyCount * 3 * sizeof(uint64_t),
                                cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < copyCount; i++) {
            constraintInfo.rows[i].row = rows[i];
            constraintInfo.rows[i].dim = destDim;
            constraintInfo.rows[i].value[0] = values[3*i];
            constraintInfo.rows[i].value[1] = values[3*i+1];
            constraintInfo.rows[i].value[2] = values[3*i+2];
        }
    }

    cudaFree(d_totalInvalid);
    cudaFree(d_invalidRows);
    cudaFree(d_invalidValues);
}

void calculateTraceInstance(SetupCtx& setupCtx, gl64_t *d_aux_trace, uint32_t stream_id, DeviceCommitBuffers *d_buffers, AirInstanceInfo *air_instance_info, Goldilocks::Element *airgroupValuesCPU, TimerGPU &timer, cudaStream_t stream) {
    
    uint64_t countId = 0;

    StepsParams *params_pinned = d_buffers->streamsData[stream_id].pinned_params;
    Goldilocks::Element *pinned_exps_params = d_buffers->streamsData[stream_id].pinned_buffer_exps_params;
    Goldilocks::Element *pinned_exps_args = d_buffers->streamsData[stream_id].pinned_buffer_exps_args;
    StepsParams *d_params =  d_buffers->streamsData[stream_id].params;
    ExpsArguments *d_expsArgs = d_buffers->streamsData[stream_id].d_expsArgs;
    DestParamsGPU *d_destParams = d_buffers->streamsData[stream_id].d_destParams;

    Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
    
    uint64_t offsetCm1 = setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetConstraints = setupCtx.starkInfo.mapOffsets[std::make_pair("constraints", false)];
    uint64_t offsetPublicInputs = setupCtx.starkInfo.mapOffsets[std::make_pair("publics", false)];
    uint64_t offsetAirgroupValues = setupCtx.starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)];
    uint64_t offsetAirValues = setupCtx.starkInfo.mapOffsets[std::make_pair("airvalues", false)];
    uint64_t offsetProofValues = setupCtx.starkInfo.mapOffsets[std::make_pair("proofvalues", false)];
    uint64_t offsetChallenges = setupCtx.starkInfo.mapOffsets[std::make_pair("challenge", false)];
    uint64_t offsetConstPols = setupCtx.starkInfo.mapOffsets[std::make_pair("const", false)];

    Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;

    StepsParams h_params = {
        trace : (Goldilocks::Element *)d_aux_trace + offsetCm1,
        aux_trace : (Goldilocks::Element *)d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_aux_trace + offsetPublicInputs,
        proofValues : (Goldilocks::Element *)d_aux_trace + offsetProofValues,
        challenges : (Goldilocks::Element *)d_aux_trace + offsetChallenges,
        airgroupValues : (Goldilocks::Element *)d_aux_trace + offsetAirgroupValues,
        airValues : (Goldilocks::Element *)d_aux_trace + offsetAirValues,
        evals : nullptr,
        xDivXSub : nullptr,
        pConstPolsAddress: d_const_pols_unpacked,
        pConstPolsExtendedTreeAddress: nullptr,
        pCustomCommitsFixed,
    };

    memcpy(params_pinned, &h_params, sizeof(StepsParams));
    
    CHECKCUDAERR(cudaMemcpyAsync(d_params, params_pinned, sizeof(StepsParams), cudaMemcpyHostToDevice, stream));
        
    TimerStartGPU(timer, STARK_CALCULATE_WITNESS_STD);
    calculateWitnessExpr_gpu(setupCtx, h_params, d_params, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);

    calculateWitnessSTD_gpu(setupCtx, h_params, d_params, true, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    calculateWitnessSTD_gpu(setupCtx, h_params, d_params, false, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, STARK_CALCULATE_WITNESS_STD);

    TimerStartGPU(timer, CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, air_instance_info->expressions_gpu, h_params, d_params, 2, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, CALCULATE_IM_POLS);

    CHECKCUDAERR(cudaMemcpyAsync(airgroupValuesCPU, d_aux_trace + offsetAirgroupValues, setupCtx.starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));
}

void verifyConstraintsGPU(SetupCtx& setupCtx, gl64_t *d_aux_trace, uint32_t stream_id, DeviceCommitBuffers *d_buffers, AirInstanceInfo *air_instance_info, ConstraintInfo *constraintsInfo, TimerGPU &timer, cudaStream_t stream) {
    
    uint64_t countId = 0;

    Goldilocks::Element *pinned_exps_params = d_buffers->streamsData[stream_id].pinned_buffer_exps_params;
    Goldilocks::Element *pinned_exps_args = d_buffers->streamsData[stream_id].pinned_buffer_exps_args;
    StepsParams *d_params =  d_buffers->streamsData[stream_id].params;
    ExpsArguments *d_expsArgs = d_buffers->streamsData[stream_id].d_expsArgs;
    DestParamsGPU *d_destParams = d_buffers->streamsData[stream_id].d_destParams;

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    
    uint64_t offsetConstraints = setupCtx.starkInfo.mapOffsets[std::make_pair("constraints", false)];
    
    Goldilocks::Element *pBufferGPU = (Goldilocks::Element *)(d_aux_trace + offsetConstraints);
    
    // Process each constraint: calculate on GPU, verify on GPU
    for (uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); i++) {
        constraintsInfo[i].id = i;
        constraintsInfo[i].stage = setupCtx.expressionsBin.constraintsInfoDebug[i].stage;
        constraintsInfo[i].imPol = setupCtx.expressionsBin.constraintsInfoDebug[i].imPol;

        if(!constraintsInfo[i].skip) {
            uint64_t destDim = setupCtx.expressionsBin.constraintsInfoDebug[i].destDim;
            uint64_t firstRow = setupCtx.expressionsBin.constraintsInfoDebug[i].firstRow;
            uint64_t lastRow = setupCtx.expressionsBin.constraintsInfoDebug[i].lastRow;
            
            // Initialize buffer to zero before each constraint
            CHECKCUDAERR(cudaMemsetAsync(pBufferGPU, 0, N * FIELD_EXTENSION * sizeof(Goldilocks::Element), stream));
            
            // Calculate constraint expression on GPU
            Dest constraintDest(NULL, N, 0, 0, true, i);
            constraintDest.addParams(i, destDim);
            constraintDest.dest_gpu = pBufferGPU;
            countId++;
            air_instance_info->expressions_gpu->calculateExpressions_gpu(d_params, constraintDest, N, false, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, true);
            
            // Verify constraint directly on GPU
            verifyConstraintGPU(pBufferGPU, N, destDim, firstRow, lastRow, constraintsInfo[i], stream);
        }
    }
    
    CHECKCUDAERR(cudaStreamSynchronize(stream));
}