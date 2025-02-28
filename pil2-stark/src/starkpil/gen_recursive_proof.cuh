#ifndef GEN_RECURSIVE_PROOF_GPU_HPP
#define GEN_RECURSIVE_PROOF_GPU_HPP

#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"

struct GPUTree
{
    gl64_t *nodes;
    uint32_t nFieldElements;
};

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

__device__ __constant__ uint64_t domain_size_inverse_[33] = {
    0x0000000000000001, // 1^{-1}
    0x7fffffff80000001, // 2^{-1}
    0xbfffffff40000001, // (1 << 2)^{-1}
    0xdfffffff20000001, // (1 << 3)^{-1}
    0xefffffff10000001,
    0xf7ffffff08000001,
    0xfbffffff04000001,
    0xfdffffff02000001,
    0xfeffffff01000001,
    0xff7fffff00800001,
    0xffbfffff00400001,
    0xffdfffff00200001,
    0xffefffff00100001,
    0xfff7ffff00080001,
    0xfffbffff00040001,
    0xfffdffff00020001,
    0xfffeffff00010001,
    0xffff7fff00008001,
    0xffffbfff00004001,
    0xffffdfff00002001,
    0xffffefff00001001,
    0xfffff7ff00000801,
    0xfffffbff00000401,
    0xfffffdff00000201,
    0xfffffeff00000101,
    0xffffff7f00000081,
    0xffffffbf00000041,
    0xffffffdf00000021,
    0xffffffef00000011,
    0xfffffff700000009,
    0xfffffffb00000005,
    0xfffffffd00000003,
    0xfffffffe00000002, // (1 << 32)^{-1}
};

void offloadCommit(uint64_t step, MerkleTreeGL **treesGL, gl64_t *d_aux_trace, uint64_t *d_tree, FRIProof<Goldilocks::Element> &proof, SetupCtx &setupCtx)
{

    double time = omp_get_wtime();
    uint64_t ncols = setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)];
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
    std::string section = "cm" + to_string(step);
    uint64_t offset = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    // treesGL[step - 1]->setSource(trace + offset);
    treesGL[step - 1]->souceTraceOffset = offset;
    time = omp_get_wtime() - time;
    std::cout << "offloadPart1: " << time << std::endl;

    uint32_t nFielsElements = treesGL[step - 1]->getMerkleTreeNFieldElements();
    // sync the GPU
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    CHECKCUDAERR(cudaMemcpy(&proof.proof.roots[step - 1][0], &d_tree[tree_size - nFielsElements], nFielsElements * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    time = omp_get_wtime() - time;
    std::cout << "offloadPart3: " << time << std::endl;
}

__global__ void fillLEv(uint64_t LEv_offset, gl64_t *d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints, uint64_t shift_, gl64_t *d_aux_trace, uint64_t N)
{

    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nOpeningPoints)
    {
        gl64_t w(1);
        Goldilocks3GPU::Element xi;
        Goldilocks3GPU::Element xiShifted;
        uint64_t openingAbs = d_openingPoints[i] < 0 ? -d_openingPoints[i] : d_openingPoints[i];
        gl64_t *LEv = (gl64_t *)d_aux_trace + LEv_offset;
        gl64_t W(W_);
        gl64_t shift(shift_);
        gl64_t invShift = shift.reciprocal();
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w *= W;
        }

        if (d_openingPoints[i] < 0)
        {
            w = w.reciprocal();
        }
        Goldilocks3GPU::mul(xi, *((Goldilocks3GPU::Element *)d_xiChallenge), w);
        Goldilocks3GPU::mul(xiShifted, xi, invShift);
        Goldilocks3GPU::one((*(Goldilocks3GPU::Element *)&LEv[i * FIELD_EXTENSION]));

        for (uint64_t k = 1; k < N; k++)
        {
            Goldilocks3GPU::mul((*(Goldilocks3GPU::Element *)&LEv[(k * nOpeningPoints + i) * FIELD_EXTENSION]), (*(Goldilocks3GPU::Element *)&LEv[((k - 1) * nOpeningPoints + i) * FIELD_EXTENSION]), xiShifted);
        }
    }
}

__global__ void fillLEv_2d(uint64_t LEv_offset, gl64_t *d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints, uint64_t shift_, gl64_t *d_aux_trace, uint64_t N)
{

    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nOpeningPoints && k < N)
    {
        gl64_t w(1);
        Goldilocks3GPU::Element xi;
        Goldilocks3GPU::Element xiShifted;
        uint64_t openingAbs = d_openingPoints[i] < 0 ? -d_openingPoints[i] : d_openingPoints[i];
        gl64_t *LEv = (gl64_t *)d_aux_trace + LEv_offset;
        gl64_t W(W_);
        gl64_t shift(shift_);
        gl64_t invShift = shift.reciprocal();
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w *= W;
        }

        if (d_openingPoints[i] < 0)
        {
            w = w.reciprocal();
        }
        Goldilocks3GPU::mul(xi, *((Goldilocks3GPU::Element *)d_xiChallenge), w);
        Goldilocks3GPU::mul(xiShifted, xi, invShift);
        Goldilocks3GPU::Element xiShiftedPow;
        Goldilocks3GPU::pow(xiShifted, k, xiShiftedPow);
        LEv[(k * nOpeningPoints + i) * FIELD_EXTENSION] = xiShiftedPow[0];
        LEv[(k * nOpeningPoints + i) * FIELD_EXTENSION + 1] = xiShiftedPow[1];
        LEv[(k * nOpeningPoints + i) * FIELD_EXTENSION + 2] = xiShiftedPow[2];
    }
}

void computeLEv_inplace(Goldilocks::Element *xiChallenge, uint64_t LEv_offset, uint64_t nBits, uint64_t nOpeningPoints, int64_t *openingPoints, DeviceCommitBuffers *d_buffers)
{
    cudaDeviceSynchronize();
    double time = omp_get_wtime();
    uint64_t N = 1 << nBits;

    gl64_t *d_xiChallenge;
    int64_t *d_openingPoints;
    cudaMalloc(&d_xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_xiChallenge, xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    cudaMalloc(&d_openingPoints, nOpeningPoints * sizeof(int64_t));
    cudaMemcpy(d_openingPoints, openingPoints, nOpeningPoints * sizeof(int64_t), cudaMemcpyHostToDevice);

    /*dim3 nThreads(32);
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x);
    fillLEv<<<nBlocks, nThreads>>>(LEv_offset, d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints, Goldilocks::shift().fe, d_buffers->d_aux_trace, N);*/
    dim3 nThreads(1, 64);
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x, (N + nThreads.y - 1) / nThreads.y);
    fillLEv_2d<<<nBlocks, nThreads>>>(LEv_offset, d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints, Goldilocks::shift().fe, d_buffers->d_aux_trace, N);
    cudaDeviceSynchronize();
    time = omp_get_wtime() - time;
    std::cout << "LEv inplace: " << time << std::endl;

    cudaDeviceSynchronize();
    time = omp_get_wtime();
    NTT_Goldilocks ntt(N);
    ntt.INTT_inplace(LEv_offset, N, FIELD_EXTENSION * nOpeningPoints, d_buffers);
    cudaDeviceSynchronize();
    time = omp_get_wtime() - time;
    std::cout << "INTT: " << time << std::endl;
    cudaFree(d_xiChallenge);
    cudaFree(d_openingPoints);
}

__global__ void calcXDivXSub(uint64_t xDivXSub_offset, gl64_t *d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints, gl64_t *d_x, gl64_t *d_aux_trace, uint64_t NExtended)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nOpeningPoints)
    {
        Goldilocks3GPU::Element xi;
        gl64_t w(1);
        uint64_t openingAbs = d_openingPoints[i] < 0 ? -d_openingPoints[i] : d_openingPoints[i];
        gl64_t W(W_);
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w *= W;
        }
        if (d_openingPoints[i] < 0)
        {
            w = w.reciprocal();
        }
        Goldilocks3GPU::mul(xi, *((Goldilocks3GPU::Element *)d_xiChallenge), w);

        if (k < NExtended)
        {
            gl64_t *d_xDivXSub = (gl64_t *)(d_aux_trace + xDivXSub_offset);
            Goldilocks3GPU::Element *xDivXSubComp = (Goldilocks3GPU::Element *)&d_xDivXSub[(k + i * NExtended) * FIELD_EXTENSION];
            Goldilocks3GPU::sub(*xDivXSubComp, d_x[k], xi);
            Goldilocks3GPU::inv(xDivXSubComp, xDivXSubComp);
            Goldilocks3GPU::mul(*xDivXSubComp, *xDivXSubComp, d_x[k]);
        }
    }
}

void calculateXDivXSub_inplace(uint64_t xDivXSub_offset, Goldilocks::Element *xiChallenge, SetupCtx &setupCtx, DeviceCommitBuffers *d_buffers)
{

    double time = omp_get_wtime();
    uint64_t nOpeningPoints = setupCtx.starkInfo.openingPoints.size();
    int64_t *openingPoints = setupCtx.starkInfo.openingPoints.data();
    gl64_t *x = (gl64_t *)setupCtx.proverHelpers.x;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t nBits = setupCtx.starkInfo.starkStruct.nBits;

    gl64_t *d_xiChallenge;
    int64_t *d_openingPoints;
    cudaMalloc(&d_xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_xiChallenge, xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    cudaMalloc(&d_openingPoints, nOpeningPoints * sizeof(int64_t));
    cudaMemcpy(d_openingPoints, openingPoints, nOpeningPoints * sizeof(int64_t), cudaMemcpyHostToDevice);
    gl64_t *d_x;
    cudaMalloc(&d_x, NExtended * sizeof(Goldilocks::Element));
    cudaMemcpy(d_x, x, NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

    dim3 nThreads(1, 128);
    std::cout << "nOpeningPoints: " << nOpeningPoints << std::endl;
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x, (NExtended + nThreads.y - 1) / nThreads.y);
    calcXDivXSub<<<nBlocks, nThreads>>>(xDivXSub_offset, d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints, d_x, d_buffers->d_aux_trace, NExtended);
    
    cudaFree(d_xiChallenge);
    cudaFree(d_openingPoints);
    cudaFree(d_x);
}

struct EvalInfo
{
    uint64_t type; // 0: cm, 1: custom, 2: fixed
    uint64_t offset;
    uint64_t stride;
    uint64_t dim;
    uint64_t openingPos;
};

__global__ void computeEvals(
    uint64_t extendBits,
    uint64_t size_eval,
    uint64_t N,
    uint64_t openingsSize,
    uint64_t LEv_offset,
    gl64_t *d_evals,
    EvalInfo *d_evalInfo,
    gl64_t *d_cmPols,
    gl64_t *d_fixedPols)
{

    uint64_t evalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (evalIdx < size_eval)
    {
        gl64_t *LEv = (gl64_t *)d_cmPols + LEv_offset;
        EvalInfo evalInfo = d_evalInfo[evalIdx];
        gl64_t *pol;
        if (evalInfo.type == 0)
        {
            pol = d_cmPols;
        }
        else if (evalInfo.type == 1)
        {
            assert(false);
        }
        else
        {
            pol = &d_fixedPols[2];
        }

        for (uint64_t k = 0; k < N; k++)
        {
            uint64_t row = (k << extendBits);
            uint64_t pos = (evalInfo.openingPos + k * openingsSize) * FIELD_EXTENSION;
            Goldilocks3GPU::Element res;
            if (evalInfo.dim == 1)
            {
                Goldilocks3GPU::mul(res, *((Goldilocks3GPU::Element *)&LEv[pos]), pol[evalInfo.offset + row * evalInfo.stride]);
            }
            else
            {
                Goldilocks3GPU::mul(res, *((Goldilocks3GPU::Element *)&LEv[pos]), *((Goldilocks3GPU::Element *)(&pol[evalInfo.offset + row * evalInfo.stride])));
            }
            Goldilocks3GPU::add((Goldilocks3GPU::Element &)(d_evals[evalIdx * FIELD_EXTENSION]), (Goldilocks3GPU::Element &)(d_evals[evalIdx * FIELD_EXTENSION]), res);
        }
    }
}

__global__ void computeEvals_v2(
    uint64_t extendBits,
    uint64_t size_eval,
    uint64_t N,
    uint64_t openingsSize,
    uint64_t LEv_offset,
    gl64_t *d_evals,
    EvalInfo *d_evalInfo,
    gl64_t *d_cmPols,
    gl64_t *d_fixedPols)
{

    extern __shared__ Goldilocks3GPU::Element shared_sum[];
    uint64_t evalIdx = blockIdx.x;

    if (evalIdx < size_eval)
    {
        gl64_t *LEv = (gl64_t *)d_cmPols + LEv_offset;
        EvalInfo evalInfo = d_evalInfo[evalIdx];
        gl64_t *pol;
        if (evalInfo.type == 0)
        {
            pol = d_cmPols;
        }
        else if (evalInfo.type == 1)
        {
            assert(false);
        }
        else
        {
            pol = &d_fixedPols[2];
        }

        for (int i = 0; i < FIELD_EXTENSION; i++)
        {
            shared_sum[threadIdx.x][i].set_val(0);
        }
        uint64_t tid = threadIdx.x;
        while (tid < N)
        {
            uint64_t row = (tid << extendBits);
            uint64_t pos = (evalInfo.openingPos + tid * openingsSize) * FIELD_EXTENSION;
            Goldilocks3GPU::Element res;
            if (evalInfo.dim == 1)
            {
                Goldilocks3GPU::mul(res, *((Goldilocks3GPU::Element *)&LEv[pos]), pol[evalInfo.offset + row * evalInfo.stride]);
            }
            else
            {
                Goldilocks3GPU::mul(res, *((Goldilocks3GPU::Element *)&LEv[pos]), *((Goldilocks3GPU::Element *)(&pol[evalInfo.offset + row * evalInfo.stride])));
            }
            Goldilocks3GPU::add(shared_sum[threadIdx.x], shared_sum[threadIdx.x], res);
            tid += blockDim.x;
        }
        __syncthreads();
        int s = (blockDim.x + 1) / 2;
        while (s > 0)
        {
            if (threadIdx.x < s && threadIdx.x + s < N)
            {
                Goldilocks3GPU::add(shared_sum[threadIdx.x], shared_sum[threadIdx.x], shared_sum[threadIdx.x + s]);
            }
            __syncthreads();
            if (s == 1)
                break;
            s = (s + 1) / 2;
        }
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < FIELD_EXTENSION; i++)
            {
                d_evals[evalIdx * FIELD_EXTENSION + i] = shared_sum[0][i];
            }
        }
    }
}

void evmap_inplace(Goldilocks::Element * evals, StepsParams &d_params, uint64_t LEv_offset, FRIProof<Goldilocks::Element> &proof, Starks<Goldilocks::Element> *starks, DeviceCommitBuffers *d_buffers)
{

    uint64_t extendBits = starks->setupCtx.starkInfo.starkStruct.nBitsExt - starks->setupCtx.starkInfo.starkStruct.nBits;
    uint64_t size_eval = starks->setupCtx.starkInfo.evMap.size();
    uint64_t N = 1 << starks->setupCtx.starkInfo.starkStruct.nBits;
    uint64_t openingsSize = (uint64_t)starks->setupCtx.starkInfo.openingPoints.size();

    CHECKCUDAERR(cudaMemset(d_params.evals, 0, size_eval * FIELD_EXTENSION * sizeof(Goldilocks::Element)));

    EvalInfo *evalsInfo = new EvalInfo[size_eval];

    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = starks->setupCtx.starkInfo.evMap[i];
        string type = ev.type == EvMap::eType::cm ? "cm" : ev.type == EvMap::eType::custom ? "custom"
                                                                                           : "fixed";
        PolMap polInfo = type == "cm" ? starks->setupCtx.starkInfo.cmPolsMap[ev.id] : type == "custom" ? starks->setupCtx.starkInfo.customCommitsMap[ev.commitId][ev.id]
                                                                                                       : starks->setupCtx.starkInfo.constPolsMap[ev.id];
        evalsInfo[i].type = type == "cm" ? 0 : type == "custom" ? 1
                                                                : 2;
        evalsInfo[i].offset = starks->setupCtx.starkInfo.getTraceOffset(type, polInfo, true);
        evalsInfo[i].stride = starks->setupCtx.starkInfo.getTraceNColsSection(type, polInfo, true);
        evalsInfo[i].dim = polInfo.dim;
        evalsInfo[i].openingPos = ev.openingPos;
    }

    EvalInfo *d_evalsInfo;
    CHECKCUDAERR(cudaMalloc(&d_evalsInfo, size_eval * sizeof(EvalInfo)));
    CHECKCUDAERR(cudaMemcpy(d_evalsInfo, evalsInfo, size_eval * sizeof(EvalInfo), cudaMemcpyHostToDevice));
    delete[] evalsInfo;

    dim3 nThreads(256);
    dim3 nBlocks(size_eval);
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();
    computeEvals_v2<<<nBlocks, nThreads, nThreads.x * sizeof(Goldilocks3GPU::Element)>>>(extendBits, size_eval, N, openingsSize, LEv_offset, (gl64_t *)d_params.evals, d_evalsInfo, (gl64_t *)d_buffers->d_aux_trace, (gl64_t *)d_buffers->d_constTree);
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    std::cout << "rick computeEvals_v2: " << time << std::endl;

    cudaMemcpy(evals, d_params.evals, size_eval * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_evalsInfo);

    proof.proof.setEvals(evals);
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

__global__ void fold(uint64_t step, gl64_t *friPol, gl64_t *d_challenge, gl64_t *d_ppar, gl64_t *d_twiddles, uint64_t shift_, uint64_t W_, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits)
{

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
                ppar[ind + k].set_val(friPol[(i * sizeFoldedPol + id) * FIELD_EXTENSION + k]);
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
                friPol[id * FIELD_EXTENSION + i].set_val(0);
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

void fold_inplace(uint64_t step, uint64_t friPol_offset, Goldilocks::Element *challenge, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits, DeviceCommitBuffers *d_buffers)
{

    gl64_t *d_friPol = (gl64_t *)(d_buffers->d_aux_trace + friPol_offset);
    gl64_t *d_challenge;
    gl64_t *d_ppar;
    gl64_t *d_twiddles;
    uint32_t ratio = 1 << (prevBits - currentBits);
    uint64_t halfRatio = ratio >> 1;

    uint64_t sizeFoldedPol = 1 << currentBits;

    CHECKCUDAERR(cudaMalloc(&d_challenge, FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_challenge, challenge, sizeof(Goldilocks::Element) * FIELD_EXTENSION, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMalloc(&d_ppar, (1 << prevBits) * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&d_twiddles, halfRatio * sizeof(Goldilocks::Element)));

    // Generate inverse twiddle factors
    Goldilocks::Element *inv_twiddles = (Goldilocks::Element *)malloc(halfRatio * sizeof(Goldilocks::Element));
    Goldilocks::Element omega_inv = omegas_inv_[prevBits - currentBits];
    inv_twiddles[0] = Goldilocks::one();

    for (uint32_t i = 1; i < halfRatio; i++)
    {
        inv_twiddles[i] = inv_twiddles[i - 1] * omega_inv;
    }
    CHECKCUDAERR(cudaMemcpy(d_twiddles, inv_twiddles, halfRatio * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    free(inv_twiddles);

    dim3 nThreads(256);
    dim3 nBlocks((sizeFoldedPol) + nThreads.x - 1 / nThreads.x);
    fold<<<nBlocks, nThreads>>>(step, d_friPol, d_challenge, d_ppar, d_twiddles, Goldilocks::shift().fe, Goldilocks::w(prevBits).fe, nBitsExt, prevBits, currentBits);

    cudaFree(d_challenge);
    cudaFree(d_ppar);
    cudaFree(d_twiddles);
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

void merkelizeFRI_inplace(uint64_t step, FRIProof<Goldilocks::Element> &proof, gl64_t *pol, MerkleTreeGL *treeFRI, uint64_t currentBits, uint64_t nextBits)
{
    uint64_t pol2N = 1 << currentBits;
    gl64_t *d_aux;
    cudaMalloc(&d_aux, pol2N * FIELD_EXTENSION * sizeof(Goldilocks::Element));

    uint64_t width = 1 << nextBits;
    uint64_t height = pol2N / width;
    dim3 nThreads(32, 32);
    dim3 nBlocks((width + nThreads.x - 1) / nThreads.x, (height + nThreads.y - 1) / nThreads.y);
    transposeFRI<<<nBlocks, nThreads>>>(d_aux, (gl64_t *)pol, pol2N, width);

    cudaMemcpy(treeFRI->source, d_aux, pol2N * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);

    uint64_t **d_tree = new uint64_t *[1];
    // PoseidonGoldilocks::merkletree_cuda_coalesced(d_tree, (uint64_t *)d_aux, treeFRI->width, treeFRI->height);
    Poseidon2Goldilocks::merkletree_cuda_streams(d_tree, (uint64_t *)d_aux, treeFRI->width, treeFRI->height);
    uint64_t tree_size = treeFRI->getNumNodes(treeFRI->height) * sizeof(uint64_t);
    CHECKCUDAERR(cudaMemcpy(treeFRI->get_nodes_ptr(), *d_tree, tree_size, cudaMemcpyDeviceToHost));
    treeFRI->getRoot(&proof.proof.fri.treesFRI[step].root[0]);
    cudaFree(d_aux);
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

__device__ void genMerkleProof_(gl64_t *nodes, gl64_t *proof, uint64_t idx, uint64_t offset, uint64_t n, uint64_t nFieldElements)
{
    if (n <= nFieldElements)
        return;

    uint64_t nextIdx = idx >> 1;
    uint64_t si = (idx ^ 1) * nFieldElements;

    for (uint64_t i = 0; i < nFieldElements; i++)
    {
        proof[i].set_val(nodes[offset + si + i].get_val());
    }

    uint64_t nextN = ((n - 1) / 8 + 1) * nFieldElements;
    genMerkleProof_(nodes, &proof[nFieldElements], nextIdx, offset + nextN * 2, nextN, nFieldElements);
}

__global__ void genMerkleProof(gl64_t *d_nodes, uint64_t sizeLeaves, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, uint64_t maxTreeWidth, uint64_t nFieldElements)
{

    uint64_t idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_query < nQueries)
    {
        uint64_t row = d_friQueries[idx_query];
        uint64_t idx_buffer = idx_query * bufferWidth + maxTreeWidth;
        genMerkleProof_(d_nodes, &d_buffer[idx_buffer], row, 0, sizeLeaves, nFieldElements);
    }
}

void proveQueries_inplace(uint64_t *friQueries, uint64_t nQueries, FRIProof<Goldilocks::Element> &fproof, MerkleTreeGL **trees, GPUTree *d_trees, uint64_t nTrees, DeviceCommitBuffers *d_buffers)
{

    uint64_t maxTreeWidth = 0;
    uint64_t maxProofSize = 0;
    for (uint64_t i = 0; i < nTrees; ++i)
    {
        if (trees[i]->getMerkleTreeWidth() > maxTreeWidth)
        {
            maxTreeWidth = trees[i]->getMerkleTreeWidth();
        }
        if (trees[i]->getMerkleProofSize() > maxProofSize)
        {
            maxProofSize = trees[i]->getMerkleProofSize();
        }
    }
    uint64_t maxBuffSize = maxTreeWidth + maxProofSize;

    Goldilocks::Element *buff = new Goldilocks::Element[maxBuffSize * nQueries * nTrees];
    gl64_t *d_buff;
    CHECKCUDAERR(cudaMalloc(&d_buff, maxBuffSize * nQueries * nTrees * sizeof(Goldilocks::Element)));
    uint64_t *d_friQueries;
    CHECKCUDAERR(cudaMalloc(&d_friQueries, nQueries * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(d_friQueries, friQueries, nQueries * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int count = 0;
    for (uint k = 0; k < nTrees; k++)
    {
        dim3 nThreads(32, 32);
        dim3 nBlocks((trees[k]->getMerkleTreeWidth() + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
        if (k < nTrees - 1)
        {
            getTreeTracePols<<<nBlocks, nThreads>>>(d_buffers->d_aux_trace + trees[k]->souceTraceOffset, trees[k]->getMerkleTreeWidth(), d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize);
        }
        else
        {
            getTreeTracePols<<<nBlocks, nThreads>>>(&d_buffers->d_constTree[2], trees[k]->getMerkleTreeWidth(), d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize); // rick: this last should be done in the CPU
        }
    }

    for (uint k = 0; k < nTrees - 1; k++)
    {
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads>>>(d_trees[k].nodes, trees[k]->getMerkleTreeHeight() * d_trees[k].nFieldElements, d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, d_trees[k].nFieldElements);
    }

    CHECKCUDAERR(cudaMemcpy(buff, d_buff, maxBuffSize * nQueries * nTrees * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(d_buff));
    CHECKCUDAERR(cudaFree(d_friQueries));

    // the last path is done offline becose is ot the constantTree which is allready in the CPU
    uint64_t aux_offset = (nTrees - 1) * nQueries;
    for (uint64_t i = 0; i < nQueries; i++)
    {
        trees[nTrees - 1]->genMerkleProof(&buff[(aux_offset + i) * maxBuffSize] + maxTreeWidth, friQueries[i], 0, trees[nTrees - 1]->getMerkleTreeHeight() * trees[nTrees - 1]->getMerkleTreeNFieldElements());
    }
    count = 0;
    for (uint k = 0; k < nTrees; k++)
    {
        for (uint64_t i = 0; i < nQueries; i++)
        {
            MerkleProof<Goldilocks::Element> mkProof(trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleProofLength(), trees[k]->getNumSiblings(), &buff[count * maxBuffSize], maxTreeWidth);
            fproof.proof.fri.trees.polQueries[i].push_back(mkProof);
            ++count;
        }
    }

    delete[] buff;
    return;
}

/*void proveFRIQueries_inplace(uint64_t* friQueries, uint64_t nQueries, SetupCtx setupCtx, FRIProof<Goldilocks::Element> &proof, MerkleTreeGL** treesFRI, DeviceCommitBuffers* d_buffers){

    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {
        MerkleTreeGL* treeFRI = treesFRI[step - 1];
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        Goldilocks::Element *buff = new Goldilocks::Element[treeFRI->getMerkleTreeWidth() + treeFRI->getMerkleProofSize()];
        for (uint64_t i = 0; i < nQueries; i++) {
            proof.proof.fri.treesFRI[step - 1].polQueries[i].clear();
            treeFRI->getGroupProof(&buff[0], friQueries[i] % (1 << currentBits));
            MerkleProof<Goldilocks::Element> mkProof(treeFRI->getMerkleTreeWidth(), treeFRI->getMerkleProofLength(), treeFRI->getNumSiblings(), &buff[0]);
            proof.proof.fri.treesFRI[step - 1].polQueries[i].push_back(mkProof);
        }
        delete[] buff;
    }

    return;
}*/

template <typename ElementType>
void genRecursiveProof_gpu(SetupCtx &setupCtx, json &globalInfo, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, Goldilocks::Element *trace, Goldilocks::Element *pConstPols, Goldilocks::Element *pConstTree, Goldilocks::Element *publicInputs, uint64_t *proofBuffer, std::string proofFile, DeviceCommitBuffers *d_buffers, bool vadcop)
{

    TimerStart(STARK_PROOF);
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time0 = omp_get_wtime();
    double time_prev = time0;

    Goldilocks::Element *aux_trace = nullptr;
    CHECKCUDAERR(cudaMemset(d_buffers->d_aux_trace, 0, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element)));
    std::cout << " rick: total meme allocated: " << setupCtx.starkInfo.mapTotalN << std::endl;
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo, airgroupId, airId, instanceId);

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;

    Starks<ElementType> starks(setupCtx, pConstTree, nullptr, true); //initializeTrees

    // GPU tree-nodes
    GPUTree *d_trees = new GPUTree[setupCtx.starkInfo.nStages + 2];
    for (uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; i++)
    {
        std::string section = "cm" + to_string(i + 1);
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
        d_trees[i].nFieldElements = 4;
        // uint64_t numNodes = NExtended * d_trees[i].nFieldElements + (NExtended - 1) * d_trees[i].nFieldElements;
        // CHECKCUDAERR(cudaMalloc(&d_trees[i].nodes, numNodes * sizeof(gl64_t)));
    }

    ExpressionsGPU expressionsCtx(setupCtx, 2, 1176, 465, 128, 2048); //maxNparams, maxNTemp1, maxNTemp3

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    Goldilocks::Element *evals = new Goldilocks::Element[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element *challenges = new Goldilocks::Element[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element *airgroupValues = nullptr;

    Goldilocks::Element *d_evals;
    CHECKCUDAERR(cudaMalloc(&d_evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element)));

    StepsParams params = {
        trace : trace,
        aux_trace : aux_trace,
        publicInputs : publicInputs,
        challenges : challenges,
        airgroupValues : nullptr,
        evals : evals,
        xDivXSub : nullptr,
        pConstPolsAddress : pConstPols,
        pConstPolsExtendedTreeAddress : pConstTree,
    };

    StepsParams d_params = {
        trace : (Goldilocks::Element *)d_buffers->d_trace,
        aux_trace : (Goldilocks::Element *)d_buffers->d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_buffers->d_publicInputs,
        challenges : nullptr,
        airgroupValues : nullptr,
        evals : d_evals,
        xDivXSub : nullptr,
        pConstPolsAddress : (Goldilocks::Element *)d_buffers->d_constPols,
        pConstPolsExtendedTreeAddress : (Goldilocks::Element *)d_buffers->d_constTree,
    };

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();
    std::cout << "Rick fins PUNT1 (pre-process) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------
    TimerStart(STARK_STEP_0);
    ElementType verkey[nFieldElements];
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->getRoot(verkey);
    starks.addTranscript(transcript, &verkey[0], nFieldElements);
    if (setupCtx.starkInfo.nPublics > 0)
    {
        if (!setupCtx.starkInfo.starkStruct.hashCommits)
        {
            starks.addTranscriptGL(transcript, &publicInputs[0], setupCtx.starkInfo.nPublics);
        }
        else
        {
            ElementType hash[nFieldElements];
            starks.calculateHash(hash, &publicInputs[0], setupCtx.starkInfo.nPublics);
            starks.addTranscript(transcript, hash, nFieldElements);
        }
    }
    TimerStopAndLog(STARK_STEP_0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT2 (step0) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();

    TimerStart(STARK_STEP_1);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 1)
        {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStart(STARK_COMMIT_STAGE_1);
    starks.commitStage_inplace(1, d_buffers->d_trace, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[0].nodes), d_buffers);
    TimerStopAndLog(STARK_COMMIT_STAGE_1);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT3 (step1) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = omp_get_wtime();

    offloadCommit(1, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[0].nodes, proof, setupCtx);
    starks.addTranscript(transcript, &proof.proof.roots[0][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_1);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT4 (offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == 2)
        {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *res = new Goldilocks::Element[N * FIELD_EXTENSION];
    Goldilocks::Element *gprod = new Goldilocks::Element[N * FIELD_EXTENSION];
   
    uint64_t gprodFieldId = setupCtx.expressionsBin.hints[0].fields[0].values[0].id;
    uint64_t numFieldId = setupCtx.expressionsBin.hints[0].fields[1].values[0].id;
    uint64_t denFieldId = setupCtx.expressionsBin.hints[0].fields[2].values[0].id;

    Dest destStruct(res, N);
    cudaMalloc(&destStruct.dest_gpu, N * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[numFieldId]);
    destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[denFieldId], true);
    std::vector<Dest> dests = {destStruct};

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT5 (pre-expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    expressionsCtx.calculateExpressions_gpu(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));

    cudaFree(destStruct.dest_gpu);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT6 (expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    Goldilocks3::copy((Goldilocks3::Element *)&gprod[0], &Goldilocks3::one());
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks3::mul((Goldilocks3::Element *)&gprod[i * FIELD_EXTENSION], (Goldilocks3::Element *)&gprod[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)&res[(i - 1) * FIELD_EXTENSION]);
    }


    Goldilocks::Element *d_grod;
    cudaMalloc(&d_grod, N * FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_grod, gprod, N * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

    uint64_t offset = setupCtx.starkInfo.getTraceOffset("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);
    uint64_t nCols = setupCtx.starkInfo.getTraceNColsSection("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT7 (gprod) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    dim3 nThreads(256);
    dim3 nBlocks((N + nThreads.x - 1) / nThreads.x);
    insertTracePol<<<nBlocks, nThreads>>>((Goldilocks::Element *)d_buffers->d_aux_trace, offset, nCols, d_grod, FIELD_EXTENSION, N);

    delete res;
    delete gprod;
    cudaFree(d_grod);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT8 (upload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(CALCULATE_IM_POLS);

    std::vector<Dest> dests2;
    for (uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++)
    {
        if (setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == 2)
        {
            uint64_t offset_ = setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(2), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos;
            Dest destStruct(NULL, N, setupCtx.starkInfo.mapSectionsN["cm" + to_string(2)]);
            destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[setupCtx.starkInfo.cmPolsMap[i].expId], false);
            destStruct.dest_gpu = (Goldilocks::Element *)(d_buffers->d_aux_trace + offset_);
            dests2.push_back(destStruct);
        }
    }

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT9 (pre-expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    expressionsCtx.calculateExpressions_gpu2(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests2, N);
    TimerStopAndLog(CALCULATE_IM_POLS);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT10 (expressions im pols) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage_inplace(2, d_buffers->d_trace, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[1].nodes), d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT11 (commit) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    offloadCommit(2, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[1].nodes, proof, setupCtx);

    TimerStopAndLog(STARK_COMMIT_STAGE_2);
    starks.addTranscript(transcript, &proof.proof.roots[1][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_2);
    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT12 (offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(STARK_STEP_Q);

    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1)
        {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    uint64_t domainSize;
    uint64_t expressionId = setupCtx.starkInfo.cExpId;
    if (expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId)
    {
        setupCtx.expressionsBin.expressionsInfo[expressionId].destDim = 3;
        domainSize = NExtended;
    }
    else
    {
        domainSize = N;
    }
    Dest destStructq(NULL, domainSize);
    destStructq.addParams(setupCtx.expressionsBin.expressionsInfo[expressionId], false);
    destStructq.dest_gpu = (Goldilocks::Element *)(d_buffers->d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]);
    std::vector<Dest> dests3 = {destStructq};

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT13 (Q expressions preparation) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    expressionsCtx.calculateExpressions_gpu2(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests3, domainSize);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT14 (Q expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.commitStage_inplace(setupCtx.starkInfo.nStages + 1, nullptr, d_buffers->d_aux_trace, (uint64_t **)(&d_trees[setupCtx.starkInfo.nStages].nodes), d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT15 (Q commit) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    offloadCommit(setupCtx.starkInfo.nStages + 1, starks.treesGL, d_buffers->d_aux_trace, (uint64_t *)d_trees[setupCtx.starkInfo.nStages].nodes, proof, setupCtx);

    TimerStopAndLog(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2)
        {
            if (setupCtx.starkInfo.challengesMap[i].stageId == 0)
                xiChallengeIndex = i;
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *xiChallenge = &challenges[xiChallengeIndex * FIELD_EXTENSION];
    uint64_t LEv_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("LEv", true)];

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT16 (Q offload) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    computeLEv_inplace(xiChallenge, setupCtx.starkInfo.mapOffsets[make_pair("LEv", true)], setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.openingPoints.size(), setupCtx.starkInfo.openingPoints.data(), d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT17 (LEv) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    evmap_inplace(evals, d_params, LEv_offset, proof, &starks, d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT18 (Evmap) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    if (!setupCtx.starkInfo.starkStruct.hashCommits)
    {
        starks.addTranscriptGL(transcript, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    }
    else
    {
        ElementType hash[nFieldElements];
        starks.calculateHash(hash, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        starks.addTranscript(transcript, hash, nFieldElements);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if (setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3)
        {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    TimerStart(COMPUTE_FRI_POLYNOMIAL);
    uint64_t xDivXSub_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)];
    d_params.xDivXSub = (Goldilocks::Element *)(d_buffers->d_aux_trace + xDivXSub_offset);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT19 (transition) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    calculateXDivXSub_inplace(xDivXSub_offset, xiChallenge, setupCtx, d_buffers);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT20 (xDivxSub) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    // FRI expressions
    expressionId = setupCtx.starkInfo.friExpId;
    if (expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId)
    {
        setupCtx.expressionsBin.expressionsInfo[expressionId].destDim = 3;
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    }
    else
    {
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBits;
    }
    Dest destStructf(NULL, domainSize);
    destStructf.addParams(setupCtx.expressionsBin.expressionsInfo[expressionId], false);
    destStructf.dest_gpu = (Goldilocks::Element *)(d_buffers->d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]);
    std::vector<Dest> destsf = {destStructf};

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT21 (pre-expressions) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    expressionsCtx.calculateExpressions_gpu2(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, destsf, domainSize);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT22 (expressions FRI) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);
    Goldilocks::Element challenge[FIELD_EXTENSION];
    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    gl64_t *d_friPol = (gl64_t *)(d_buffers->d_aux_trace + friPol_offset);

    TimerStart(STARK_FRI_FOLDING);
    uint64_t nBitsExt = setupCtx.starkInfo.starkStruct.steps[0].nBits;
    Goldilocks::Element *foldedFRIPol = new Goldilocks::Element[(1 << setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits) * FIELD_EXTENSION];
    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        fold_inplace(step, friPol_offset, challenge, nBitsExt, prevBits, currentBits, d_buffers);

        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(step, proof, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits);
            starks.addTranscript(transcript, &proof.proof.fri.treesFRI[step].root[0], nFieldElements);
        }
        else
        {
            CHECKCUDAERR(cudaMemcpy(foldedFRIPol, d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
            if (!setupCtx.starkInfo.starkStruct.hashCommits)
            {
                starks.addTranscriptGL(transcript, foldedFRIPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            }
            else
            {
                ElementType hash[nFieldElements];
                starks.calculateHash(hash, foldedFRIPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                starks.addTranscript(transcript, hash, nFieldElements);
            }
        }
        starks.getChallenge(transcript, *challenge);
    }
    TimerStopAndLog(STARK_FRI_FOLDING);

    TimerStart(STARK_FRI_QUERIES);
    uint64_t friQueries[setupCtx.starkInfo.starkStruct.nQueries];
    TranscriptType transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    starks.addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);

    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    proveQueries_inplace(friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, d_trees, nTrees, d_buffers);

    /*proveFRIQueries_inplace(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx, proof, starks.treesFRI, d_buffers);*/ // Not run in the GPU at this point

    for (uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step)
    {
        FRI<Goldilocks::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
    }

    FRI<ElementType>::setFinalPol(proof, foldedFRIPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_FRI_QUERIES);

    TimerStopAndLog(STARK_STEP_FRI);

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime();
    std::cout << "Rick fins PUNT23 (FRI) " << time - time0 << " " << time - time_prev << std::endl;
    time_prev = time;

    delete challenges;
    delete evals;
    delete airgroupValues;
    delete foldedFRIPol;
    TimerStopAndLog(STARK_PROOF);

    proof.proof.proof2pointer(proofBuffer);
    if(!proofFile.empty()) {
        json2file(pointer2json(proofBuffer, setupCtx.starkInfo), proofFile);
    }

    cudaFree(d_evals);
}
#endif
