#include "starks.hpp"
#include "starks_gpu.cuh"
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_cubic_extension.cuh"
#include "proof2zkinStark.hpp"

Goldilocks::Element omegas_inv_[33] = {
    0x1,
    0xffffffff00000000,
    0xfffeffff00000001,
    0xfffffeff00000101,
    0xffefffff00100001,
    0xfbffffff04000001,
    0xdfffffff20000001,
    0x3fffbfffc0,
    0x7f4949dce07bf05d,
    0x4bd6bb172e15d48c,
    0x38bc97652b54c741,
    0x553a9b711648c890,
    0x55da9bb68958caa,
    0xa0a62f8f0bb8e2b6,
    0x276fd7ae450aee4b,
    0x7b687b64f5de658f,
    0x7de5776cbda187e9,
    0xd2199b156a6f3b06,
    0xd01c8acd8ea0e8c0,
    0x4f38b2439950a4cf,
    0x5987c395dd5dfdcf,
    0x46cf3d56125452b1,
    0x909c4b1a44a69ccb,
    0xc188678a32a54199,
    0xf3650f9ddfcaffa8,
    0xe8ef0e3e40a92655,
    0x7c8abec072bb46a6,
    0xe0bfc17d5c5a7a04,
    0x4c6b8a5a0b79f23a,
    0x6b4d20533ce584fe,
    0xe5cceae468a70ec2,
    0x8958579f296dac7a,
    0x16d265893b5b7e85,
};

__global__ void unpack(
    const uint64_t* src,
    uint64_t* dst,
    uint64_t nRows,
    uint64_t nCols,
    uint64_t words_per_row,
    const uint64_t *d_unpack_info
) {
    extern __shared__ uint64_t shared_mem[];
    uint64_t* shared_unpack_info = shared_mem;

    // Load unpack info
    if (threadIdx.x < nCols)
        shared_unpack_info[threadIdx.x] = d_unpack_info[threadIdx.x];
    __syncthreads();

    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) return;
    const uint64_t* packed_row = src + row * words_per_row;

    uint64_t word = packed_row[0];
    uint64_t word_idx = 0;
    uint64_t bit_offset = 0;

    #pragma unroll
    for (uint64_t c = 0; c < nCols; c++) {
        uint64_t nbits = shared_unpack_info[c];
        uint64_t val;
        uint64_t bits_left = 64 - bit_offset;

        if (nbits <= bits_left) {
            uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
            val = (word >> bit_offset) & mask;
            bit_offset += nbits;
            if (bit_offset == 64 && word_idx + 1 < words_per_row) {
                word = packed_row[++word_idx];
                bit_offset = 0;
            }
        } else {
            uint64_t low = word >> bit_offset;
            word = packed_row[++word_idx];
            uint64_t high = word & ((1ULL << (nbits - bits_left)) - 1ULL);
            val = (high << bits_left) | low;
            bit_offset = nbits - bits_left;
        }

        uint32_t blockY = c >> 2;
        uint32_t ncols_block = nCols - 4 * blockY < 4 ? nCols - 4 * blockY : 4;
        uint32_t colId_block = (c - 4 * blockY);
        uint32_t dst_idx = blockY * 4 * nRows + blockIdx.x * 256 * ncols_block + colId_block * 256 + threadIdx.x;

        dst[dst_idx] = val;
    }
}

void unpack_trace(
    AirInstanceInfo *air_instance_info,
    uint64_t* src,
    uint64_t* dst,
    uint64_t nCols,
    uint64_t nRows,
    cudaStream_t stream,
    TimerGPU &timer
) {
    dim3 threads(256);
    dim3 blocks((nRows + threads.x - 1) / threads.x);

    size_t sharedMemSize = nCols * sizeof(uint64_t);
    TimerStartCategoryGPU(timer, UNPACK_TRACE);
    unpack<<<blocks, threads, sharedMemSize, stream>>>(
        src,
        dst,
        nRows,
        nCols,
        air_instance_info->num_packed_words,
        air_instance_info->unpack_info
    );
    TimerStopCategoryGPU(timer, UNPACK_TRACE);
    CHECKCUDAERR(cudaGetLastError());
}

void computeZerofier(Goldilocks::Element *d_zi, uint64_t nBits, uint64_t nBitsExt, cudaStream_t stream) {
    uint64_t NExtended = 1 << nBitsExt;
    uint64_t extendBits = nBitsExt - nBits;
    uint64_t extend = (1 << extendBits);

    Goldilocks::Element w = Goldilocks::w(extendBits);
    Goldilocks::Element sn = Goldilocks::shift();
    for (uint64_t i = 0; i < nBits; i++) Goldilocks::square(sn, sn);
    
    dim3 threads(256);
    dim3 blocks((NExtended + threads.x - 1) / threads.x);
    size_t shared_mem_size = extend * sizeof(gl64_t);
    buildZHInv_kernel<<<blocks, threads, shared_mem_size, stream>>>((gl64_t *)d_zi, extend, NExtended, w, sn);
    
    // TODO!
    // for(uint64_t i = 1; i < boundaries.size(); ++i) {
    //         Boundary boundary = boundaries[i];
    //     if(boundary.name == "everyRow") {
    //         buildZHInv(nBits, nBitsExt);
    //     } else if(boundary.name == "firstRow") {
    //         buildOneRowZerofierInv(nBits, nBitsExt, i, 0);
    //     } else if(boundary.name == "lastRow") {
    //         buildOneRowZerofierInv(nBits, nBitsExt, i, N);
    //     } else if(boundary.name == "everyFrame") {
    //         buildFrameZerofierInv(nBits, nBitsExt, i, boundary.offsetMin, boundary.offsetMax);
    //     }
    // }
}

__global__ void setProdIdentity3(gl64_t *pol) {
    pol[0] = gl64_t(uint64_t(1));
    pol[1] = gl64_t(uint64_t(0));
    pol[2] = gl64_t(uint64_t(0));
}

__global__ void buildZHInv_kernel(gl64_t *d_zi, uint64_t extend, uint64_t NExtended, Goldilocks::Element w, Goldilocks::Element sn) {
    extern __shared__ gl64_t zi_shared[];

    uint32_t k = threadIdx.x;

    if (k < extend) {
        gl64_t w_k = gl64_t(w.fe) ^ k;
        gl64_t val = (gl64_t(sn.fe) * w_k) - gl64_t(uint64_t(1));
        zi_shared[k] = val.reciprocal();
    }

    __syncthreads();

    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NExtended) {
        d_zi[idx] = zi_shared[idx % extend];
    }
}

__global__ void computeX_kernel(gl64_t *x, uint64_t NExtended, Goldilocks::Element shift, Goldilocks::Element w) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= NExtended) return;

    gl64_t w_k = gl64_t(w.fe) ^ k;
    x[k] = gl64_t(shift.fe) * w_k;
}

void commitStage_inplace(uint64_t step, SetupCtx &setupCtx, MerkleTreeGL **treesGL, gl64_t *d_trace, gl64_t *d_aux_trace, TranscriptGL_GPU *d_transcript,  bool skipRecalculation, TimerGPU &timer, cudaStream_t stream)
{
    if (step <= setupCtx.starkInfo.nStages)
    {
        extendAndMerkelize_inplace(step, setupCtx, treesGL, d_trace, d_aux_trace, d_transcript, skipRecalculation, timer, stream);
    }
    else
    {
        computeQ_inplace(step, setupCtx, treesGL, d_aux_trace, d_transcript, timer, stream);
    }
}

void extendAndMerkelize_inplace(uint64_t step, SetupCtx& setupCtx, MerkleTreeGL** treesGL, gl64_t *d_trace, gl64_t *d_aux_trace, TranscriptGL_GPU *d_transcript, bool skipRecalculation, TimerGPU &timer, cudaStream_t stream)
{
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    gl64_t *src = step == 1 ? d_trace : d_aux_trace;
    uint64_t offset_src = step == 1 ? 0 : setupCtx.starkInfo.mapOffsets[make_pair(section, false)];
    gl64_t *dst = d_aux_trace;
    uint64_t offset_dst = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    Goldilocks::Element * dstGL = (Goldilocks::Element*) (d_aux_trace);

    treesGL[step - 1]->setSource(dstGL + offset_dst);
    Goldilocks::Element *pNodes = dstGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];
    treesGL[step - 1]->setNodes(pNodes);

    if(!skipRecalculation) {
        NTT_Goldilocks_GPU ntt;

        if (nCols > 0)
        {
            ntt.LDE_MerkleTree_GPU_inplace(pNodes, dst, offset_dst, src, offset_src, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, timer, stream);
        }
    }

    if (nCols > 0)
    {
        if(d_transcript != nullptr) {
            uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
            d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
        }
    }
}

void extendAndMerkelizeFixed(SetupCtx& setupCtx, Goldilocks::Element *d_fixedPols, Goldilocks::Element *d_fixedPolsExtended, TimerGPU &timer, cudaStream_t stream) {
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t nCols = setupCtx.starkInfo.nConstants;
    NTT_Goldilocks_GPU ntt;

    Goldilocks::Element *src = d_fixedPols;
    Goldilocks::Element *dst = d_fixedPolsExtended;
    Goldilocks::Element *pNodes = dst + nCols * NExtended;
    ntt.LDE_MerkleTree_GPU_inplace(pNodes, (gl64_t *)dst, 0, (gl64_t *)src, 0, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, setupCtx.starkInfo.nConstants, timer, stream);
}

void computeQ_inplace(uint64_t step, SetupCtx &setupCtx, MerkleTreeGL **treesGL, gl64_t *d_aux_trace,TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    uint64_t offset_cmQ = setupCtx.starkInfo.mapOffsets[std::make_pair(section, true)];
    uint64_t offset_q = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    uint64_t qDeg = setupCtx.starkInfo.qDeg;
    uint64_t qDim = setupCtx.starkInfo.qDim;

    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);
     
    Goldilocks::Element* d_aux_traceGL = (Goldilocks::Element*) d_aux_trace;

    treesGL[step - 1]->setSource(d_aux_traceGL + offset_q);
    Goldilocks::Element *pNodes = d_aux_traceGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];
    treesGL[step - 1]->setNodes(pNodes);

    if (nCols > 0)
    {
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft", false)];
        NTT_Goldilocks_GPU nttExtended;
        nttExtended.computeQ_inplace(pNodes, offset_cmQ, offset_q, qDeg, qDim, shiftIn, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, d_aux_trace, offset_helper, timer, stream);
        uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
        if(d_transcript != nullptr) {
            d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
        }
    }
}

__global__ void insertTracePol(Goldilocks::Element *d_aux_trace, uint64_t offset, uint64_t stride, Goldilocks::Element *d_pol, uint64_t dim, uint64_t N)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        if (dim == 1)
            d_aux_trace[offset + idx * stride] = d_pol[idx];
        else
        {
            d_aux_trace[offset + idx * stride] = d_pol[idx * dim];
            d_aux_trace[offset + idx * stride + 1] = d_pol[idx * dim + 1];
            d_aux_trace[offset + idx * stride + 2] = d_pol[idx * dim + 2];
        }
    }
}

__global__ void fillLEv_2d(gl64_t *d_LEv,  uint64_t nOpeningPoints, uint64_t N, gl64_t *d_shiftedValues)
{
    uint64_t i  = blockIdx.x;                  // opening point index
    uint64_t k0 = blockIdx.y * blockDim.y;     // start exponent for this block
    uint64_t row  = k0 + threadIdx.y;          // this thread's exponent index
    if (i >= nOpeningPoints || row >= N) return;

    Goldilocks3GPU::Element xi;
    xi[0] = d_shiftedValues[i * FIELD_EXTENSION + 0];
    xi[1] = d_shiftedValues[i * FIELD_EXTENSION + 1];
    xi[2] = d_shiftedValues[i * FIELD_EXTENSION + 2];

    __shared__ Goldilocks3GPU::Element basePow;

    if (threadIdx.y == 0) {
        Goldilocks3GPU::pow(xi, k0, basePow);
    }
    __syncthreads();

    Goldilocks3GPU::Element xi_t;
    Goldilocks3GPU::pow(xi, threadIdx.y, xi_t);

    Goldilocks3GPU::Element res;
    Goldilocks3GPU::mul(res, basePow, xi_t);

    d_LEv[getBufferOffset(row, i*FIELD_EXTENSION, N, nOpeningPoints * FIELD_EXTENSION)] = res[0];
    d_LEv[getBufferOffset(row, i*FIELD_EXTENSION + 1, N, nOpeningPoints * FIELD_EXTENSION)] = res[1];
    d_LEv[getBufferOffset(row, i*FIELD_EXTENSION + 2, N, nOpeningPoints * FIELD_EXTENSION)] = res[2];
}

__global__ void evalXiShifted(gl64_t* d_shiftedValues, gl64_t *d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints, uint64_t invShift_)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nOpeningPoints )
    {
        uint32_t openingAbs = d_openingPoints[i] < 0 ? -d_openingPoints[i] : d_openingPoints[i];
        gl64_t w(W_);
        w^=openingAbs;
        if (d_openingPoints[i] < 0)
        {
            w = w.reciprocal();
        }
        
        Goldilocks3GPU::Element xi;
        gl64_t invShift(invShift_);
        Goldilocks3GPU::mul(xi, *((Goldilocks3GPU::Element *)d_xiChallenge), w);
        Goldilocks3GPU::mul(xi, xi, invShift);
        d_shiftedValues[i * FIELD_EXTENSION] = xi[0];
        d_shiftedValues[i * FIELD_EXTENSION + 1] = xi[1];
        d_shiftedValues[i * FIELD_EXTENSION + 2] = xi[2];
    }
}

void computeLEv_inplace(Goldilocks::Element *d_xiChallenge, uint64_t nBits, uint64_t nOpeningPoints, int64_t *d_openingPoints, gl64_t *d_aux_trace, uint64_t offset_helper, gl64_t* d_LEv, TimerGPU &timer, cudaStream_t stream)
{
    TimerStartCategoryGPU(timer, LEV);
    uint64_t N = 1 << nBits;

    gl64_t * d_shiftedValues = d_aux_trace + offset_helper;

    Goldilocks::Element invShift = Goldilocks::inv(Goldilocks::shift());

    // Evaluate the shifted value for each opening point
    dim3 nThreads_(32);
    dim3 nBlocks_((nOpeningPoints + nThreads_.x - 1) / nThreads_.x);
    evalXiShifted<<<nBlocks_, nThreads_, 0, stream>>>(d_shiftedValues, (gl64_t*)d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints, invShift.fe);

    dim3 nThreads(1, 512);
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x, (N + nThreads.y - 1) / nThreads.y);
    fillLEv_2d<<<nBlocks, nThreads, 0, stream>>>(d_LEv, nOpeningPoints, N,  d_shiftedValues);
    TimerStopCategoryGPU(timer, LEV);
    CHECKCUDAERR(cudaGetLastError());

    TimerStartCategoryGPU(timer, NTT);
    NTT_Goldilocks_GPU ntt;
    ntt.INTT_inplace(d_LEv, nBits, FIELD_EXTENSION * nOpeningPoints, stream);
    TimerStopCategoryGPU(timer, NTT);
   
}

__global__ void calcXis(Goldilocks::Element * d_xis, gl64_t *d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nOpeningPoints)
    {
        uint64_t openingAbs = d_openingPoints[i] < 0 ? -d_openingPoints[i] : d_openingPoints[i];
        gl64_t W(W_);
        gl64_t w = W ^ uint32_t(openingAbs);
        if (d_openingPoints[i] < 0)
        {
            w = w.reciprocal();
        }
        Goldilocks3GPU::mul(*((Goldilocks3GPU::Element *) &d_xis[i * FIELD_EXTENSION]), *((Goldilocks3GPU::Element *)d_xiChallenge), w);
    }
}


void calculateXis_inplace(SetupCtx &setupCtx, StepsParams &h_params, int64_t *d_openingPoints, Goldilocks::Element *d_xiChallenge, cudaStream_t stream)
{

    uint64_t nOpeningPoints = setupCtx.starkInfo.openingPoints.size();
    int64_t *openingPoints = setupCtx.starkInfo.openingPoints.data();
    uint64_t nBits = setupCtx.starkInfo.starkStruct.nBits;
 
    dim3 nThreads(16);
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x);
    calcXis<<<nBlocks, nThreads, 0, stream>>>(h_params.xDivXSub, (gl64_t*)d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints);
    CHECKCUDAERR(cudaGetLastError());
}

__global__ void computeEvals_v2(
    uint64_t NExtended,
    uint64_t extendBits,
    uint64_t size_eval,
    uint64_t N,
    uint64_t openingsSize,
    gl64_t *d_evals,
    EvalInfo *d_evalInfo,
    gl64_t *d_cmPols,
    gl64_t *d_fixedPols,
    gl64_t *d_customComits,
    gl64_t *d_LEv,
    gl64_t *d_helper)
{

    extern __shared__ Goldilocks3GPU::Element shared_sum[];
    uint64_t evalIdx = blockIdx.x;
    uint64_t chunkIdx = blockIdx.y;

    if (evalIdx < size_eval)
    {
        EvalInfo evalInfo = d_evalInfo[evalIdx];
        gl64_t *pol;
        if (evalInfo.type == 0)
        {
            pol = d_cmPols;
        }
        else if (evalInfo.type == 1)
        {
            pol = d_customComits;
        }
        else
        {
            pol = d_fixedPols;
        }

        for (int i = 0; i < FIELD_EXTENSION; i++)
        {
            shared_sum[threadIdx.x][i]= gl64_t(uint64_t(0)); 
        }
        uint64_t tid = chunkIdx * blockDim.x + threadIdx.x;
        while (tid < N)
        {
            uint64_t row = (tid << extendBits);
            Goldilocks3GPU::Element LEv;
            LEv[0] = d_LEv[getBufferOffset(tid, evalInfo.openingPos * FIELD_EXTENSION, N, openingsSize * FIELD_EXTENSION)];
            LEv[1] = d_LEv[getBufferOffset(tid, evalInfo.openingPos * FIELD_EXTENSION + 1, N, openingsSize * FIELD_EXTENSION)];
            LEv[2] = d_LEv[getBufferOffset(tid, evalInfo.openingPos * FIELD_EXTENSION + 2, N, openingsSize * FIELD_EXTENSION)];
            Goldilocks3GPU::Element res;
            if (evalInfo.dim == 1)
            {
                Goldilocks3GPU::mul(res, LEv, pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos, NExtended, evalInfo.stageCols)]);
            }
            else
            {
                Goldilocks3GPU::Element val;
                val[0] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos, NExtended, evalInfo.stageCols)];
                val[1] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos + 1, NExtended, evalInfo.stageCols)];
                val[2] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos + 2, NExtended, evalInfo.stageCols)];
                Goldilocks3GPU::mul(res, LEv, val);
            }
            Goldilocks3GPU::add(shared_sum[threadIdx.x], shared_sum[threadIdx.x], res);
            tid += blockDim.x * gridDim.y;
        }
        __syncthreads();
        int s = (blockDim.x + 1) / 2;
        while (s > 0)
        {
            if (threadIdx.x < s)
            {
                Goldilocks3GPU::add(shared_sum[threadIdx.x], shared_sum[threadIdx.x], shared_sum[threadIdx.x + s]);
            }
            __syncthreads();
            if (s == 1)
                break;
            s = (s + 1) / 2;
        }
        
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t partial_pos = evalIdx * gridDim.y + chunkIdx;
            d_helper[partial_pos * FIELD_EXTENSION] = shared_sum[0][0];
            d_helper[partial_pos * FIELD_EXTENSION + 1] = shared_sum[0][1];
            d_helper[partial_pos * FIELD_EXTENSION + 2] = shared_sum[0][2];
        }
    }
}

__global__ void computeEvalsReduction(gl64_t *d_evals, gl64_t *d_helper, EvalInfo *d_evalInfo, uint64_t size_eval, uint64_t n_eval_chunks) {
    uint64_t evalIdx = threadIdx.x;
    if (evalIdx < size_eval) {
        uint64_t base = evalIdx * n_eval_chunks * FIELD_EXTENSION;
        d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION] = d_helper[base + 0];
        d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION + 1] = d_helper[base + 1];
        d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION + 2] = d_helper[base + 2];
        for (int i = 1; i < n_eval_chunks; ++i) {
            d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION] += d_helper[base + i * FIELD_EXTENSION];
            d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION + 1] += d_helper[base + i * FIELD_EXTENSION + 1];
            d_evals[d_evalInfo[evalIdx].evalPos * FIELD_EXTENSION + 2] += d_helper[base + i * FIELD_EXTENSION + 2];
        }
    }
}

void evmap_inplace(SetupCtx &setupCtx, StepsParams &h_params, uint64_t chunk, uint64_t nOpeningPoints, int64_t *openingPoints, AirInstanceInfo *air_instance_info, Goldilocks::Element *d_LEv, uint64_t offset_helper, TimerGPU &timer, cudaStream_t stream)
{

    TimerStartCategoryGPU(timer, EVALS);
    gl64_t *d_constTree = (gl64_t *)h_params.pConstPolsExtendedTreeAddress;

    uint64_t extendBits = setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits;
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    
    EvalInfo *d_evalsInfo = air_instance_info->evalsInfo[chunk];
    uint64_t nEvals = air_instance_info->evalsInfoSizes[chunk];

    uint64_t n_eval_chunks = 16;

    gl64_t *d_helper = (gl64_t *)h_params.aux_trace + offset_helper;
    
    dim3 nThreads(256);
    dim3 nBlocks(nEvals, n_eval_chunks);
    computeEvals_v2<<<nBlocks, nThreads, nThreads.x * sizeof(Goldilocks3GPU::Element), stream>>>(NExtended, extendBits, nEvals, N, nOpeningPoints, (gl64_t *)h_params.evals, d_evalsInfo, (gl64_t *)h_params.aux_trace, d_constTree, (gl64_t *)h_params.pCustomCommitsFixed, (gl64_t *)d_LEv, d_helper);
    computeEvalsReduction<<<1, nEvals, 0, stream>>>((gl64_t *)h_params.evals, d_helper, d_evalsInfo, nEvals, n_eval_chunks);
    CHECKCUDAERR(cudaGetLastError());
    TimerStopCategoryGPU(timer, EVALS);
}

__device__ void intt_tinny(gl64_t *data, uint32_t N, uint32_t logN, gl64_t *d_twiddles, uint32_t ncols)
{

    uint32_t halfN = N >> 1;
    // Reverse permutation
    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t ibr = __brev(i) >> (32 - logN);
        if (ibr > i)
        {
            gl64_t tmp;
            for (uint32_t j = 0; j < ncols; j++)
            {
                tmp = data[i * ncols + j];
                data[i * ncols + j] = data[ibr * ncols + j];
                data[ibr * ncols + j] = tmp;
            }
        }
    }
    // Inverse NTT
    for (uint32_t i = 0; i < logN; i++)
    {
        for (uint32_t j = 0; j < halfN; j++)
        {
            for (uint32_t col = 0; col < ncols; col++)
            {
                uint32_t half_group_size = 1 << i;
                uint32_t group = j >> i;
                uint32_t offset = j & (half_group_size - 1);
                uint32_t index1 = (group << i + 1) + offset;
                uint32_t index2 = index1 + half_group_size;
                gl64_t factor = d_twiddles[offset * (N >> i + 1)];
                gl64_t odd_sub = gl64_t((uint64_t)data[index2 * ncols + col]) * factor;
                data[index2 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) - odd_sub;
                data[index1 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) + odd_sub;
            }
        }
    }
    // Scale by N^{-1}
    gl64_t factor = gl64_t(domain_size_inverse_[logN]);
    for (uint32_t i = 0; i < N * ncols; i++)
    {
        data[i] = gl64_t((uint64_t)data[i]) * factor;
    }
}

__global__ void fold(uint64_t step, gl64_t *friPol, gl64_t *d_challenge, gl64_t *d_ppar, Goldilocks::Element omega_inv, gl64_t *d_twiddles, uint64_t shift_, uint64_t W_, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits)
{

    uint64_t halfRatio = (1 << (prevBits - currentBits)) >> 1;
    d_twiddles[0] = gl64_t(uint64_t(1));
    for (uint32_t i = 1; i < halfRatio; i++)
    {
        d_twiddles[i] = d_twiddles[i - 1] * gl64_t(omega_inv.fe);
    }
    uint32_t polBits = prevBits;
    uint64_t sizePol = 1 << polBits;
    uint32_t foldedPolBits = currentBits;
    uint64_t sizeFoldedPol = 1 << foldedPolBits;
    uint32_t ratio = sizePol / sizeFoldedPol;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < sizeFoldedPol)
    {

        if (step == 0)
            return;
        gl64_t shift(shift_);
        gl64_t invShift = shift.reciprocal();
        for (uint32_t j = 0; j < nBitsExt - prevBits; j++)
        {
            invShift *= invShift;
        }

        gl64_t W(W_);
        gl64_t invW = W.reciprocal();
        // Evaluate the sinv value for the id current component
        gl64_t sinv = invShift;
        gl64_t base = invW;
        uint32_t exponent = id;

        while (exponent > 0)
        {
            if (exponent % 2 == 1)
            {
                sinv *= base;
            }
            base *= base;
            exponent /= 2;
        }

        gl64_t *ppar = (gl64_t *)d_ppar + id * ratio * FIELD_EXTENSION;
        for (int i = 0; i < ratio; i++)
        {
            int ind = i * FIELD_EXTENSION;
            for (int k = 0; k < FIELD_EXTENSION; k++)
            {
                ppar[ind + k] = gl64_t(friPol[(i * sizeFoldedPol + id) * FIELD_EXTENSION + k]);
            }
        }
        intt_tinny(ppar, ratio, prevBits - currentBits, d_twiddles, FIELD_EXTENSION);

        // Multiply coefs by 1, shiftInv, shiftInv^2, shiftInv^3, ......
        gl64_t r(1);
        for (uint64_t i = 0; i < ratio; i++)
        {
            Goldilocks3GPU::Element *component = (Goldilocks3GPU::Element *)&ppar[i * FIELD_EXTENSION];
            Goldilocks3GPU::mul(*component, *component, r);
            r *= sinv;
        }
        // evalPol
        if (ratio == 0)
        {
            for (uint32_t i = 0; i < FIELD_EXTENSION; i++)
            {
                friPol[id * FIELD_EXTENSION + i]= gl64_t(uint64_t(0)); 
            }
        }
        else
        {
            for (uint32_t i = 0; i < FIELD_EXTENSION; i++)
            {
                friPol[id * FIELD_EXTENSION + i] = ppar[(ratio - 1) * FIELD_EXTENSION + i];
            }
            for (int i = ratio - 2; i >= 0; i--)
            {
                Goldilocks3GPU::Element aux;
                Goldilocks3GPU::mul(aux, *((Goldilocks3GPU::Element *)&friPol[id * FIELD_EXTENSION]), *((Goldilocks3GPU::Element *)&d_challenge[0]));
                Goldilocks3GPU::add(*((Goldilocks3GPU::Element *)&friPol[id * FIELD_EXTENSION]), aux, *((Goldilocks3GPU::Element *)&ppar[i * FIELD_EXTENSION]));
            }
        }
    }
}

void fold_inplace(uint64_t step, uint64_t friPol_offset, uint64_t offset_helper, Goldilocks::Element *d_challenge, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits, gl64_t *d_aux_trace, TimerGPU &timer, cudaStream_t stream)
{

    uint32_t ratio = 1 << (prevBits - currentBits);
    uint64_t halfRatio = ratio >> 1;
    gl64_t *d_friPol = (gl64_t *)(d_aux_trace + friPol_offset);
    gl64_t *d_twiddles = (gl64_t *)d_aux_trace + offset_helper;
    gl64_t *d_ppar = (gl64_t *)d_aux_trace + offset_helper + halfRatio;

    uint64_t sizeFoldedPol = 1 << currentBits;

    Goldilocks::Element omega_inv = omegas_inv_[prevBits - currentBits];
    
    dim3 nThreads(256);
    dim3 nBlocks((sizeFoldedPol) + nThreads.x - 1 / nThreads.x);
    TimerStartCategoryGPU(timer, FRI);
    fold<<<nBlocks, nThreads, 0, stream>>>(step, d_friPol, (gl64_t *)d_challenge, d_ppar, omega_inv, d_twiddles, Goldilocks::shift().fe, Goldilocks::w(prevBits).fe, nBitsExt, prevBits, currentBits);
    TimerStopCategoryGPU(timer, FRI);
    CHECKCUDAERR(cudaGetLastError());
}

__global__ void transposeFRI(gl64_t *d_aux, gl64_t *pol, uint64_t degree, uint64_t width)
{
    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t height = degree / width;

    if (idx_x < width && idx_y < height)
    {
        uint64_t fi = idx_y * width + idx_x;
        uint64_t di = idx_x * height + idx_y;
        for (uint64_t k = 0; k < FIELD_EXTENSION; k++)
        {
            d_aux[di * FIELD_EXTENSION + k] = pol[fi * FIELD_EXTENSION + k];
        }
    }
}

void merkelizeFRI_inplace(SetupCtx& setupCtx, StepsParams &h_params, uint64_t step, gl64_t *pol, MerkleTreeGL *treeFRI, uint64_t currentBits, uint64_t nextBits, TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
{
    uint64_t pol2N = 1 << currentBits;

    uint64_t width = 1 << nextBits;
    uint64_t height = pol2N / width;
    dim3 nThreads(32, 32);
    dim3 nBlocks((width + nThreads.x - 1) / nThreads.x, (height + nThreads.y - 1) / nThreads.y);
    transposeFRI<<<nBlocks, nThreads, 0, stream>>>((gl64_t *)treeFRI->source, (gl64_t *)pol, pol2N, width);
    
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    Poseidon2GoldilocksGPU::merkletree_cuda_coalesced(3, (uint64_t*) treeFRI->nodes, (uint64_t *)treeFRI->source, treeFRI->width, treeFRI->height, stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);

    uint64_t tree_size = treeFRI->numNodes;
    if(d_transcript != nullptr) {
        d_transcript->put(&treeFRI->nodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
    }
}

__global__ void getTreeTracePols(gl64_t *d_treeTrace, uint64_t traceWidth, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth)
{

    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < traceWidth && idx_y < nQueries)
    {
        uint64_t row = d_friQueries[idx_y];
        uint64_t idx_trace = row * traceWidth + idx_x;
        uint64_t idx_buffer = idx_y * bufferWidth + idx_x;
        d_buffer[idx_buffer] = d_treeTrace[idx_trace];
    }
}

__global__ void getTreeTracePolsBlocks(gl64_t *d_treeTrace, uint64_t nCols, uint64_t nRows, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth)
{

    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < nCols && idx_y < nQueries)
    {
        uint64_t row = d_friQueries[idx_y];
        uint64_t idx_buffer = idx_y * bufferWidth + idx_x;
        uint64_t idx_trace = getBufferOffset(row, idx_x, nRows, nCols);
        d_buffer[idx_buffer] = d_treeTrace[idx_trace];
    }
}

__device__ void genMerkleProof_(gl64_t *nodes, gl64_t *proof, uint64_t idx, uint64_t offset, uint64_t n, uint64_t nFieldElements, uint32_t arity)
{
    if (n == 1)
        return;

    uint64_t currIdx = idx % arity;
    uint64_t nextIdx = idx / arity;
    uint64_t si = idx - currIdx;  //start index

    gl64_t *proofPtr = proof;
    for (uint64_t i = 0; i < arity; i++)
    {
        if (i == currIdx) continue;  // Skip the current index
        for( uint32_t j = 0; j < nFieldElements; j++){
            proofPtr[j]= gl64_t(nodes[(offset + (si + i)) * nFieldElements + j][0]); 
        }
        proofPtr += nFieldElements;
    }

    uint64_t nextN = (n + (arity - 1)) /arity;
    genMerkleProof_(nodes, &proof[(arity - 1) * nFieldElements], nextIdx, offset + nextN * arity, nextN, nFieldElements, 3);
}

__global__ void genMerkleProof(gl64_t *d_nodes, uint64_t nLeaves, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, uint64_t maxTreeWidth, uint64_t nFieldElements)
{

    uint64_t idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_query < nQueries)
    {
        uint64_t row = d_friQueries[idx_query];
        uint64_t idx_buffer = idx_query * bufferWidth + maxTreeWidth;
        genMerkleProof_(d_nodes, &d_buffer[idx_buffer], row, 0, nLeaves, nFieldElements, 3);
    }
}

void proveQueries_inplace(SetupCtx& setupCtx, gl64_t *d_queries_buff, uint64_t *d_friQueries, uint64_t nQueries, MerkleTreeGL **trees, uint64_t nTrees, gl64_t *d_aux_trace, gl64_t* d_constTree, uint32_t nStages, cudaStream_t stream)
{   
    uint64_t maxBuffSize = setupCtx.starkInfo.maxProofBuffSize;
    uint64_t maxTreeWidth = setupCtx.starkInfo.maxTreeWidth;

    for (uint k = 0; k < nTrees; k++)
    {
        dim3 nThreads(32, 32);
        dim3 nBlocks((trees[k]->getMerkleTreeWidth() + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
        if (k < nStages + 1)
        {
            std::string section = "cm" + to_string(k+1);
            uint64_t offset = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
            uint64_t nCols = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_aux_trace + offset, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize);
        }
        else if (k == nStages + 1)
        {
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_constTree, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize);
        } else{
            uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[0].name + "0"];
            uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_aux_trace + offset + N*nCols, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize);
        }
    }
    CHECKCUDAERR(cudaGetLastError());


    for (uint k = 0; k < nStages + 1; k++)
    {
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[k]->get_nodes_ptr(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE);
        CHECKCUDAERR(cudaGetLastError());
    }
    CHECKCUDAERR(cudaGetLastError());

    dim3 nthreads(64);
    dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
    genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[nStages + 1]->get_nodes_ptr(), trees[nStages + 1]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + (nStages + 1) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE);
    CHECKCUDAERR(cudaGetLastError());

    if(nTrees > nStages + 2){
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[nStages + 2]->get_nodes_ptr(), trees[nStages + 2]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + (nStages + 2) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE);
        CHECKCUDAERR(cudaGetLastError());
    }
}

__global__ void moduleQueries(uint64_t* d_friQueries, uint64_t nQueries, uint64_t currentBits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nQueries) {
        d_friQueries[idx] %= (1ULL << currentBits);
    }
}

void proveFRIQueries_inplace(SetupCtx& setupCtx, gl64_t *d_queries_buff, uint64_t step, uint64_t currentBits, uint64_t *d_friQueries, uint64_t nQueries, MerkleTreeGL *treeFRI, cudaStream_t stream) {
    uint64_t buffSize = treeFRI->getMerkleTreeWidth() + treeFRI->getMerkleProofSize();
    dim3 nthreads_(64);
    dim3 nblocks_((nQueries + nthreads_.x - 1) / nthreads_.x);
    moduleQueries<<<nblocks_, nthreads_, 0, stream>>>(d_friQueries, nQueries, currentBits);
    CHECKCUDAERR(cudaGetLastError());
    dim3 nThreads(32, 32);
    dim3 nBlocks((treeFRI->getMerkleTreeWidth() + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
    getTreeTracePols<<<nBlocks, nThreads, 0, stream>>>((gl64_t *)treeFRI->source, treeFRI->getMerkleTreeWidth(), d_friQueries, nQueries, d_queries_buff, buffSize);
    CHECKCUDAERR(cudaGetLastError());
    dim3 nthreads(64);
    dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);

    genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)treeFRI->nodes, treeFRI->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff, buffSize, treeFRI->getMerkleTreeWidth(), HASH_SIZE);

    CHECKCUDAERR(cudaGetLastError());
}

void calculateImPolsExpressions(SetupCtx& setupCtx, ExpressionsGPU* expressionsCtx, StepsParams &h_params, StepsParams *d_params, int64_t step, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream){

    uint64_t domainSize = (1 << setupCtx.starkInfo.starkStruct.nBits);
    std::vector<Dest> dests;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {
            Goldilocks::Element* pAddress = step == 1 ? h_params.trace : h_params.aux_trace;
            Dest destStruct(NULL, domainSize, setupCtx.starkInfo.cmPolsMap[i].stagePos, setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)], false);
            destStruct.addParams(setupCtx.starkInfo.cmPolsMap[i].expId, setupCtx.starkInfo.cmPolsMap[i].dim, false);
            uint64_t offset_aux_trace = setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), false)];
            destStruct.dest_gpu = (Goldilocks::Element *)(pAddress + offset_aux_trace);
            countId++;
            expressionsCtx->calculateExpressions_gpu(d_params, destStruct, domainSize, false, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
        }
    }
        
}

void calculateExpression(SetupCtx& setupCtx, ExpressionsGPU* expressionsCtx, StepsParams *d_params, Goldilocks::Element* dest_gpu, uint64_t expressionId, bool inverse, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU& timer, cudaStream_t stream, bool debug){
    
    uint64_t domainSize;
    bool domainExtended;
    if (expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId)
    {
        setupCtx.expressionsBin.expressionsInfo[expressionId].destDim = 3;
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
        domainExtended = true;
    }
    else
    {
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBits;
        domainExtended = false;
    }
    Dest destStruct(NULL, domainSize, 0, 3, false, expressionId);
    destStruct.addParams(expressionId, setupCtx.expressionsBin.expressionsInfo[expressionId].destDim, inverse);
    destStruct.dest_gpu = dest_gpu;
    countId++;
    expressionsCtx->calculateExpressions_gpu(d_params, destStruct, domainSize, domainExtended, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, debug);

}

void setProof(SetupCtx &setupCtx, Goldilocks::Element *h_aux_trace, Goldilocks::Element *proof_buffer_pinned, cudaStream_t stream) {
    uint64_t initialOffset = 0;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t numNodes = setupCtx.starkInfo.getNumNodesMT(NExtended);
    for(uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; ++i) {
        uint64_t stage = i + 1;
        Goldilocks::Element *nodes = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt" + to_string(stage), true)];
        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + numNodes - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        initialOffset += HASH_SIZE;
    }

    for (uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
        if(setupCtx.starkInfo.customCommits[i].stageWidths[0] != 0) {
            Goldilocks::Element *nodes = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair(setupCtx.starkInfo.customCommits[i].name + "0", true)];
            CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + numNodes - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            initialOffset += HASH_SIZE;
        }
    }

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; step++)
    {
        uint64_t height = 1 << setupCtx.starkInfo.starkStruct.steps[step + 1].nBits;
        uint64_t numNodes = setupCtx.starkInfo.getNumNodesMT(height);
        Goldilocks::Element *nodes = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)];
        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + numNodes - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        initialOffset += HASH_SIZE;
    }

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;
    uint64_t queriesProofSize = (nTrees + nTreesFRI) * setupCtx.starkInfo.maxProofBuffSize * setupCtx.starkInfo.starkStruct.nQueries;
    uint64_t offsetProofQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("proof_queries", false)];
    uint64_t finalPolDegree = 1 << setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits;

    Goldilocks::Element *d_queries_buff = h_aux_trace + offsetProofQueries;
    Goldilocks::Element *d_evals = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("evals", false)];
    Goldilocks::Element *d_airgroupValues = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)];
    Goldilocks::Element *d_airValues = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("airvalues", false)];
    Goldilocks::Element *d_fri_pol = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];


    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_queries_buff, queriesProofSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += queriesProofSize;
    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION;
    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_airgroupValues, setupCtx.starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += setupCtx.starkInfo.airgroupValuesSize;
    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_airValues, setupCtx.starkInfo.airValuesSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += setupCtx.starkInfo.airValuesSize;
    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_fri_pol, finalPolDegree * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += finalPolDegree * FIELD_EXTENSION;
}

void writeProof(SetupCtx &setupCtx, Goldilocks::Element *proof_buffer_pinned, uint64_t *proof_buffer, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, std::string proofFile) {
    uint64_t initialOffset = 0;

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);

    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t numNodes = setupCtx.starkInfo.getNumNodesMT(NExtended);
    for(uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; ++i) {
        memcpy(&proof.proof.roots[i][0], &proof_buffer_pinned[initialOffset], HASH_SIZE * sizeof(uint64_t));
        initialOffset += HASH_SIZE;
    }

    for (uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
        if(setupCtx.starkInfo.customCommits[i].stageWidths[0] != 0) {
            uint64_t pos = setupCtx.starkInfo.nStages + 2 + i;
            memcpy(&proof.proof.roots[pos - 1][0], &proof_buffer_pinned[initialOffset], HASH_SIZE * sizeof(uint64_t));
            initialOffset += HASH_SIZE;
        }
    }

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; step++)
    {
        uint64_t height = 1 << setupCtx.starkInfo.starkStruct.steps[step + 1].nBits;
        uint64_t numNodes = setupCtx.starkInfo.getNumNodesMT(height);
        memcpy(&proof.proof.fri.treesFRI[step].root[0], &proof_buffer_pinned[initialOffset], HASH_SIZE * sizeof(uint64_t));
        initialOffset += HASH_SIZE;
    }

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;
    uint64_t queriesProofSize = (nTrees + nTreesFRI) * setupCtx.starkInfo.maxProofBuffSize * setupCtx.starkInfo.starkStruct.nQueries;
    uint64_t offsetProofQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("proof_queries", false)];

    uint64_t finalPolDegree = 1 << setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits;

    Goldilocks::Element *queries = &proof_buffer_pinned[initialOffset];
    initialOffset += queriesProofSize;
    Goldilocks::Element *evals = &proof_buffer_pinned[initialOffset];
    initialOffset += setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION;
    Goldilocks::Element *airgroupValues = &proof_buffer_pinned[initialOffset];
    initialOffset += setupCtx.starkInfo.airgroupValuesSize;
    Goldilocks::Element *airValues = &proof_buffer_pinned[initialOffset];
    initialOffset += setupCtx.starkInfo.airValuesSize;
    Goldilocks::Element *finalPol = &proof_buffer_pinned[initialOffset];
    initialOffset += finalPolDegree * FIELD_EXTENSION;

    int count = 0;
    for (uint k = 0; k < setupCtx.starkInfo.nStages + 1; k++)
    {
        for (uint64_t i = 0; i < setupCtx.starkInfo.starkStruct.nQueries; i++)
        {
            uint64_t width = setupCtx.starkInfo.mapSectionsN["cm" + to_string(k + 1)];
            uint64_t proofLength = (uint64_t)ceil(log10(NExtended) / log10(setupCtx.starkInfo.starkStruct.merkleTreeArity));
            uint64_t numSiblings = (setupCtx.starkInfo.starkStruct.merkleTreeArity - 1) * HASH_SIZE;
            MerkleProof<Goldilocks::Element> mkProof(width, proofLength, numSiblings, (void *) &queries[count * setupCtx.starkInfo.maxProofBuffSize], setupCtx.starkInfo.maxTreeWidth);
            proof.proof.fri.trees.polQueries[i].push_back(mkProof);
            ++count;
        }
    }

    for (uint64_t i = 0; i < setupCtx.starkInfo.starkStruct.nQueries; i++)
    {
        uint64_t width = setupCtx.starkInfo.nConstants;
        uint64_t proofLength = (uint64_t)ceil(log10(NExtended) / log10(setupCtx.starkInfo.starkStruct.merkleTreeArity));
        uint64_t numSiblings = (setupCtx.starkInfo.starkStruct.merkleTreeArity - 1) * HASH_SIZE;
        MerkleProof<Goldilocks::Element> mkProof(width, proofLength, numSiblings, (void *) &queries[count * setupCtx.starkInfo.maxProofBuffSize], setupCtx.starkInfo.maxTreeWidth);
        proof.proof.fri.trees.polQueries[i].push_back(mkProof);
        ++count;
    }

    if(nTrees > setupCtx.starkInfo.nStages + 2){
        for (uint64_t i = 0; i < setupCtx.starkInfo.starkStruct.nQueries; i++)
        {
            uint64_t width = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[0].name + "0"];
            uint64_t proofLength = (uint64_t)ceil(log10(NExtended) / log10(setupCtx.starkInfo.starkStruct.merkleTreeArity));
            uint64_t numSiblings = (setupCtx.starkInfo.starkStruct.merkleTreeArity - 1) * HASH_SIZE;
            MerkleProof<Goldilocks::Element> mkProof(width, proofLength, numSiblings, (void *) &queries[count * setupCtx.starkInfo.maxProofBuffSize], setupCtx.starkInfo.maxTreeWidth);
            proof.proof.fri.trees.polQueries[i].push_back(mkProof);
            ++count;
        }
    }

#pragma omp parallel for collapse(2)
    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; step++)
    {   
        for (uint64_t i = 0; i < setupCtx.starkInfo.starkStruct.nQueries; i++)
        {
            Goldilocks::Element *queriesFRI = &queries[(nTrees + step) * setupCtx.starkInfo.starkStruct.nQueries * setupCtx.starkInfo.maxProofBuffSize];
            uint64_t width = FIELD_EXTENSION * (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) / (1 << setupCtx.starkInfo.starkStruct.steps[step + 1].nBits);
            uint64_t proofLength = (uint64_t)ceil(log10(1 << setupCtx.starkInfo.starkStruct.steps[step + 1].nBits) / log10(setupCtx.starkInfo.starkStruct.merkleTreeArity));
            uint64_t numSiblings = (setupCtx.starkInfo.starkStruct.merkleTreeArity - 1) * HASH_SIZE;
            uint64_t proofSize = proofLength * numSiblings;
            uint64_t buffSize = width + proofSize;
            MerkleProof<Goldilocks::Element> mkProof(width, proofLength, numSiblings, (void *)&queriesFRI[i * buffSize], width);
            proof.proof.fri.treesFRI[step].polQueries[i].push_back(mkProof);
        }
    }

    proof.proof.setEvals(evals);
    proof.proof.setAirgroupValues(airgroupValues); 
    proof.proof.setAirValues(airValues);
    proof.proof.fri.setPol(finalPol, finalPolDegree);

    proof.proof.proof2pointer(proof_buffer);

    if(!proofFile.empty()) {
        json2file(pointer2json(proof_buffer, setupCtx.starkInfo), proofFile);
    }
}

void calculateHash(TranscriptGL_GPU *d_transcript, Goldilocks::Element* hash, SetupCtx &setupCtx, Goldilocks::Element* buffer, uint64_t nElements, cudaStream_t stream) {
    d_transcript->reset(stream);
    d_transcript->put(buffer, nElements, stream);
    d_transcript->getState(hash, stream);
};


__device__ __forceinline__ void printArgs(gl64_t *a, uint32_t dimA,  bool constA, gl64_t *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, bool debug = false);
__device__ __forceinline__ void printFRI(gl64_t *res, uint32_t dimRes, int i, bool debug = false);


__device__ __forceinline__ void printArgs(gl64_t *a, uint32_t dimA, bool constA, gl64_t *b, uint32_t dimB, bool constB, int i, uint64_t op_type, uint64_t op, bool debug) {
    uint64_t debug_row = 0;
    bool print = debug && (i + threadIdx.x == debug_row);
    Goldilocks::Element *a_ = (Goldilocks::Element *)a; 
    Goldilocks::Element *b_ = (Goldilocks::Element *)b; 
    if(print){
        printf("Expression debug op: %lu with type %lu\n", op, op_type);
        if(a!= NULL){
            for(uint32_t j = 0; j < dimA; j++){
                Goldilocks::Element val = constA ? a_[j] : a_[j*blockDim.x + debug_row%blockDim.x];
                printf("Expression debug a[%d]: %lu (constant %u)\n", j, val.fe % GOLDILOCKS_PRIME, constA);
            }
        }
        if(b!= NULL){
            for(uint32_t j = 0; j < dimB; j++){
                Goldilocks::Element val = constB ? b_[j] : b_[j*blockDim.x + debug_row%blockDim.x];
                printf("Expression debug b[%d]: %lu (constant %u)\n", j, val.fe % GOLDILOCKS_PRIME, constB);
            }
        }
    }
}

__device__ __forceinline__ void printFRI(gl64_t *res, int i, bool debug){
    uint64_t debug_row = 0;
    bool print = debug && (i + threadIdx.x == debug_row);
    Goldilocks::Element *res_ = (Goldilocks::Element *)res; 
    if(print){
        for(uint32_t j = 0; j < FIELD_EXTENSION; j++){
            printf("Expression debug res[%d]: %lu\n", j, res_[j*blockDim.x + debug_row%blockDim.x].fe % GOLDILOCKS_PRIME);
        }
    }
}

__global__  void computeFRIExpression(uint64_t domainSize, uint64_t nOpeningPoints, gl64_t *d_fri, uint64_t* d_countsPerOpeningPos, EvalInfo **d_evalInfoPerOpening, gl64_t *d_evals, gl64_t *vf1, gl64_t *vf2, gl64_t *d_cmPols, gl64_t *d_xDivXSub, gl64_t *d_x, gl64_t *d_fixedPols, gl64_t *d_customComits, bool debug)
{
    int chunk_idx = blockIdx.x;
    uint64_t nchunks = domainSize / blockDim.x;

    extern __shared__ Goldilocks::Element shared[];

    while (chunk_idx < nchunks) {
        gl64_t *fri_pol = (gl64_t *)shared;
        gl64_t *accum = fri_pol + blockDim.x * FIELD_EXTENSION;
        gl64_t *res = accum + blockDim.x * FIELD_EXTENSION;

        uint64_t i = chunk_idx * blockDim.x;
        uint64_t r = i + threadIdx.x;
        for(uint64_t o = 0; o < nOpeningPoints; ++o) {
            for(uint64_t j = 0; j < d_countsPerOpeningPos[o]; ++j) {
                EvalInfo evalInfo = d_evalInfoPerOpening[o][j];
                gl64_t* eval = d_evals + evalInfo.evalPos * FIELD_EXTENSION;
                gl64_t *pol;
                if (evalInfo.type == 0)
                {
                    pol = d_cmPols;
                }
                else if (evalInfo.type == 1)
                {
                    pol = d_customComits;
                }
                else
                {
                    pol = d_fixedPols;
                }
    
                gl64_t *out = (j == 0) ? accum : res;
                if(evalInfo.dim == 1) {
                    out[threadIdx.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos, domainSize, evalInfo.stageCols)];
                    // printArgs(out, 1, false, eval, 3, true, i, 3, nOp++, debug);
                    Goldilocks3GPU::sub_13_gpu_b_const(out, out, eval);
                    // printFRI(out, i, debug);
                } else {
                    out[threadIdx.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos, domainSize, evalInfo.stageCols)];
                    out[threadIdx.x + blockDim.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos + 1, domainSize, evalInfo.stageCols)];
                    out[threadIdx.x + 2*blockDim.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos + 2, domainSize, evalInfo.stageCols)];
                    // printArgs(out, 3, false, eval, 3, true, i, 1, nOp++, debug);
                    Goldilocks3GPU::sub_gpu_b_const(out, out, eval);
                    // printFRI(out, i, debug);
                }
                if(j != 0) {
                    // printArgs(accum, 3, false, vf2, 3, true, i, 2, nOp++, debug);
                    Goldilocks3GPU::mul_gpu_b_const(accum, accum, vf2);
                    // printFRI(accum, i, debug);
                    // printArgs(accum, 3, false, out, 3, false, i, 0, nOp++, debug);
                    Goldilocks3GPU::add_gpu_no_const(accum, accum, (gl64_t *)out);
                    // printFRI(accum, i, debug);
                }
            }

            // printArgs(d_x + i, 1, false, &d_xDivXSub[o * FIELD_EXTENSION], 3, true, i, 3, nOp, debug);
            Goldilocks3GPU::sub_13_gpu_b_const(res, d_x + i, &d_xDivXSub[o * FIELD_EXTENSION]);
            Goldilocks3GPU::Element aux;
            aux[0] = res[threadIdx.x];
            aux[1] = res[blockDim.x + threadIdx.x];
            aux[2] = res[2 * blockDim.x + threadIdx.x];
            Goldilocks3GPU::inv(aux, aux);
            res[threadIdx.x] = aux[0];
            res[blockDim.x + threadIdx.x] = aux[1];
            res[2 * blockDim.x + threadIdx.x] = aux[2];
            // printArgs(res, 3, false, accum, 3, false, i, 2, nOp++, debug);

            gl64_t *out = o == 0 ? fri_pol : accum;
            Goldilocks3GPU::mul_gpu_no_const(out, accum, res);
            // printFRI(out, i, debug);
            if(o != 0) {
                // printArgs(fri_pol, 3, false, accum, 3, false, i, 2, nOp++, debug);
                Goldilocks3GPU::mul_gpu_b_const(fri_pol, fri_pol, vf1);
                // printFRI(fri_pol, i, debug);
                // printArgs(fri_pol, 3, false, accum, 3, false, i, 0, nOp++, debug);
                Goldilocks3GPU::add_gpu_no_const(fri_pol, fri_pol, accum);
                // printFRI(fri_pol, i, debug);
            }
        }

        gl64_gpu::copy_gpu(d_fri + i * FIELD_EXTENSION, uint64_t(FIELD_EXTENSION), &fri_pol[0], false);
        gl64_gpu::copy_gpu(d_fri + i * FIELD_EXTENSION + 1, uint64_t(FIELD_EXTENSION), &fri_pol[blockDim.x], false);
        gl64_gpu::copy_gpu(d_fri + i * FIELD_EXTENSION + 2, uint64_t(FIELD_EXTENSION), &fri_pol[2*blockDim.x], false);
        chunk_idx += gridDim.x;
    }
}

void calculateFRIExpression(SetupCtx& setupCtx, StepsParams &h_params, AirInstanceInfo *air_instance_info, cudaStream_t stream) {
    Goldilocks::Element *dest = (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]);

    uint64_t domainSize = (1 << setupCtx.starkInfo.starkStruct.nBitsExt);
    uint32_t nthreads_ = setupCtx.starkInfo.nrowsPack;
    uint32_t nblocks_ = std::min((uint32_t)setupCtx.starkInfo.maxNBlocks, (uint32_t)((domainSize + nthreads_-1)/ nthreads_));
    size_t sharedMem = nthreads_ * 3 * FIELD_EXTENSION * sizeof(Goldilocks::Element);
    dim3 nThreads(nthreads_);    
    dim3 nBlocks(nblocks_);
    computeFRIExpression<<<nBlocks, nThreads, sharedMem, stream>>>(
        domainSize, 
        setupCtx.starkInfo.openingPoints.size(), 
        (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)],
        air_instance_info->evalsInfoFRISizes,
        air_instance_info->evalsInfoFRI,
        (gl64_t*)h_params.evals,
        (gl64_t*)h_params.challenges + 4 * FIELD_EXTENSION,
        (gl64_t*)h_params.challenges + 5 * FIELD_EXTENSION,
        (gl64_t*)h_params.aux_trace,
        (gl64_t*)h_params.xDivXSub,
        (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)],
        (gl64_t *)h_params.pConstPolsExtendedTreeAddress,
        (gl64_t *)h_params.pCustomCommitsFixed,
        air_instance_info->airId == 0
    );
    CHECKCUDAERR(cudaGetLastError());
}