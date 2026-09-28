#include "verify_constraints.hpp"
#include "gpu_timer.cuh"
#include "goldilocks_tooling.cuh"
#include "starks_gpu.cuh"
#include "gen_proof.cuh"
#include "multiplicity.cuh"
#include "hints.cuh"
#include <algorithm>
#include <vector>

// Helper to check if a value is zero in Goldilocks field (handles alias: 0 and GOLDILOCKS_PRIME both represent 0)
__device__ __forceinline__ bool isGoldilocksZero(uint64_t val) {
    return (val == 0) || (val == GOLDILOCKS_PRIME);
}


// Collects up to maxSample invalid rows of one constraint: the first maxSample/2 in arrival order,
// the rest as a ring over the remaining slots.
template<int DIM>
__global__ void verifyConstraintKernel(
    const Goldilocks::Element* __restrict__ dest,
    uint64_t N,
    uint64_t firstRow,
    uint64_t lastRow,
    uint32_t* __restrict__ d_totalInvalid,
    uint32_t* __restrict__ d_invalidRows,
    uint64_t* __restrict__ d_invalidValues,
    uint32_t maxSample
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
        if (maxSample == 0) return;
        const uint32_t stable = maxSample / 2;
        const uint32_t ring = maxSample - stable;
        for (uint32_t i = 0; i < s_count; i++) {
            const uint32_t g = base + i;
            const uint32_t slot = g < stable ? g : stable + (g - stable) % ring;
            d_invalidRows[slot] = s_rows[i];
            d_invalidValues[3 * slot + 0] = s_vals[i][0];
            d_invalidValues[3 * slot + 1] = s_vals[i][1];
            d_invalidValues[3 * slot + 2] = s_vals[i][2];
        }
    }
}

void calculateTraceInstance(SetupCtx& setupCtx, gl64_t *d_aux_trace, uint32_t stream_id, DeviceCommitBuffers *d_buffers, AirInstanceInfo *air_instance_info, Goldilocks::Element *airgroupValuesCPU, uint64_t airgroupId, uint64_t airId, TimerGPU &timer, cudaStream_t stream) {
    
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

    calculateImHints_gpu(setupCtx, h_params, d_params, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    calculateWitnessSTD_gpu(setupCtx, h_params, d_params, true, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    calculateWitnessSTD_gpu(setupCtx, h_params, d_params, false, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, STARK_CALCULATE_WITNESS_STD);

    TimerStartGPU(timer, CALCULATE_IM_POLS);
    calculateImPolsExpressions(setupCtx, air_instance_info->expressions_gpu, h_params, d_params, 2, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
    TimerStopGPU(timer, CALCULATE_IM_POLS);

    // Count lookups into prover-owned tables: verify-constraints never commits, so the commit's
    // scatter does not run and those tables would stay at zero.
    {
        int gpuId = 0;
        CHECKCUDAERR(cudaGetDevice(&gpuId));
        MulAcc *mulAcc = mulAccOnGpu(gpuId);
        if (mulAcc != nullptr) {
            calculateMulCalcGPU(setupCtx, h_params, airgroupId, airId, mulAcc->d_acc, timer, stream);
            mul_note_commit();
        }
    }

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

    const uint64_t nConstraints = setupCtx.expressionsBin.constraintsInfoDebug.size();
    if (nConstraints == 0) return;
    // Every constraint gets its own result slots, so the whole air needs one host sync. Stream-ordered
    // allocations: a plain cudaFree fences every stream on the device.
    const uint64_t maxSample = constraintsInfo[0].n_print_constraints;
    const uint64_t nSlots = std::max<uint64_t>(1, nConstraints * maxSample);
    uint32_t *d_counts, *d_rows;
    uint64_t *d_vals;
    CHECKCUDAERR(cudaMallocAsync(&d_counts, nConstraints * sizeof(uint32_t), stream));
    CHECKCUDAERR(cudaMallocAsync(&d_rows, nSlots * sizeof(uint32_t), stream));
    CHECKCUDAERR(cudaMallocAsync(&d_vals, nSlots * 3 * sizeof(uint64_t), stream));
    CHECKCUDAERR(cudaMemsetAsync(d_counts, 0, nConstraints * sizeof(uint32_t), stream));

    const uint32_t blockSize = 256;
    const uint32_t numBlocks = (N + blockSize - 1) / blockSize;
    for (uint64_t i = 0; i < nConstraints; i++) {
        const auto &dbg = setupCtx.expressionsBin.constraintsInfoDebug[i];
        constraintsInfo[i].id = i;
        constraintsInfo[i].stage = dbg.stage;
        constraintsInfo[i].imPol = dbg.imPol;
        if (constraintsInfo[i].skip) continue;

        CHECKCUDAERR(cudaMemsetAsync(pBufferGPU, 0, N * FIELD_EXTENSION * sizeof(Goldilocks::Element), stream));
        Dest constraintDest(NULL, N, 0, 0, true, i);
        constraintDest.addParams(i, dbg.destDim);
        constraintDest.dest_gpu = pBufferGPU;
        countId++;
        air_instance_info->expressions_gpu->calculateExpressions_gpu(d_params, constraintDest, N, false, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, true);

        uint32_t *rows = d_rows + i * maxSample;
        uint64_t *vals = d_vals + 3 * i * maxSample;
        if (dbg.destDim == 1) {
            verifyConstraintKernel<1><<<numBlocks, blockSize, 0, stream>>>(pBufferGPU, N, dbg.firstRow, dbg.lastRow, d_counts + i, rows, vals, (uint32_t)maxSample);
        } else {
            verifyConstraintKernel<3><<<numBlocks, blockSize, 0, stream>>>(pBufferGPU, N, dbg.firstRow, dbg.lastRow, d_counts + i, rows, vals, (uint32_t)maxSample);
        }
        CHECKCUDAERR(cudaGetLastError());
    }

    std::vector<uint32_t> counts(nConstraints);
    std::vector<uint32_t> rows(nSlots);
    std::vector<uint64_t> vals(3 * nSlots);
    CHECKCUDAERR(cudaMemcpyAsync(counts.data(), d_counts, counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CHECKCUDAERR(cudaMemcpyAsync(rows.data(), d_rows, rows.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CHECKCUDAERR(cudaMemcpyAsync(vals.data(), d_vals, vals.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CHECKCUDAERR(cudaFreeAsync(d_counts, stream));
    CHECKCUDAERR(cudaFreeAsync(d_rows, stream));
    CHECKCUDAERR(cudaFreeAsync(d_vals, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (uint64_t i = 0; i < nConstraints; i++) {
        if (constraintsInfo[i].skip) continue;
        constraintsInfo[i].nrows = counts[i];
        const uint64_t destDim = setupCtx.expressionsBin.constraintsInfoDebug[i].destDim;
        const uint64_t copyCount = std::min<uint64_t>(counts[i], maxSample);
        for (uint64_t k = 0; k < copyCount; k++) {
            const uint64_t j = i * maxSample + k;
            constraintsInfo[i].rows[k].row = rows[j];
            constraintsInfo[i].rows[k].dim = destDim;
            constraintsInfo[i].rows[k].value[0] = vals[3 * j];
            constraintsInfo[i].rows[k].value[1] = vals[3 * j + 1];
            constraintsInfo[i].rows[k].value[2] = vals[3 * j + 2];
        }
    }
}
