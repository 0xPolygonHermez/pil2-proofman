#include "starks.hpp"
#include "starks_gpu.cuh"
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_cubic_extension.cuh"

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

void commitStage_inplace(uint64_t step, SetupCtx &setupCtx, MerkleTreeGL **treesGL, gl64_t *d_trace, gl64_t *d_aux_trace, DeviceCommitBuffers *d_buffers, TranscriptGL_GPU *d_transcript, double *nttTime, double *merkleTime)
{
    if (step <= setupCtx.starkInfo.nStages)
    {
        extendAndMerkelize_inplace(step, setupCtx, treesGL, d_trace, d_aux_trace, d_buffers, d_transcript, nttTime, merkleTime);
    }
    else
    {
        computeQ_inplace(step, setupCtx, treesGL, d_aux_trace, d_buffers, d_transcript, nttTime, merkleTime);
    }
}

void extendAndMerkelize_inplace(uint64_t step, SetupCtx& setupCtx, MerkleTreeGL** treesGL, gl64_t *d_trace, gl64_t *d_aux_trace, DeviceCommitBuffers *d_buffers, TranscriptGL_GPU *d_transcript, double *nttTime, double *merkleTime)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
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
    
    NTT_Goldilocks ntt(N);

    if (nCols > 0)
    {
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft_" + to_string(step), false)];
        ntt.LDE_MerkleTree_GPU_inplace(pNodes, dst, offset_dst, src, offset_src, N, NExtended, nCols, d_buffers, offset_helper, nttTime, merkleTime);
        uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
        if(d_transcript != nullptr) {
            d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE);
        }
    }
}

void computeQ_inplace(uint64_t step, SetupCtx &setupCtx, MerkleTreeGL **treesGL, gl64_t *d_aux_trace, DeviceCommitBuffers *d_buffers,TranscriptGL_GPU *d_transcript, double *nttTime, double *merkleTime)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    uint64_t offset_cmQ = setupCtx.starkInfo.mapOffsets[std::make_pair(section, true)];
    uint64_t offset_q = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    uint64_t qDeg = setupCtx.starkInfo.qDeg;
    uint64_t qDim = setupCtx.starkInfo.qDim;

    Goldilocks::Element S[setupCtx.starkInfo.qDeg];
    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);
    S[0] = Goldilocks::one();
    for(uint64_t i = 1; i < setupCtx.starkInfo.qDeg; i++) {
        S[i] = Goldilocks::mul(S[i - 1], shiftIn);
    }
    Goldilocks::Element* d_aux_traceGL = (Goldilocks::Element*) d_aux_trace;

    treesGL[step - 1]->setSource( d_aux_traceGL + offset_q);
    Goldilocks::Element *pNodes = d_aux_traceGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];
    treesGL[step - 1]->setNodes(pNodes);

    if (nCols > 0)
    {
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft_" + to_string(step), false)];
        NTT_Goldilocks nttExtended(NExtended);
        nttExtended.computeQ_inplace(pNodes, offset_cmQ, offset_q, qDeg, qDim, S, N, NExtended, nCols, d_buffers, offset_helper, nttTime, merkleTime);
        uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
        if(d_transcript != nullptr) {
            d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE);
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

    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nOpeningPoints && k < N)
    {
        
        Goldilocks3GPU::Element xi;
        xi[0] = d_shiftedValues[i * FIELD_EXTENSION];
        xi[1] = d_shiftedValues[i * FIELD_EXTENSION + 1];
        xi[2] = d_shiftedValues[i * FIELD_EXTENSION + 2];
        Goldilocks3GPU::Element xiShiftedPow;
        Goldilocks3GPU::pow(xi, k, xiShiftedPow);
        d_LEv[(k * nOpeningPoints + i) * FIELD_EXTENSION] = xiShiftedPow[0];
        d_LEv[(k * nOpeningPoints + i) * FIELD_EXTENSION + 1] = xiShiftedPow[1];
        d_LEv[(k * nOpeningPoints + i) * FIELD_EXTENSION + 2] = xiShiftedPow[2];

    }

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

void computeLEv_inplace(Goldilocks::Element *xiChallenge, uint64_t nBits, uint64_t nOpeningPoints, int64_t *openingPoints, DeviceCommitBuffers *d_buffers, uint64_t offset_helper, gl64_t* d_LEv, double *nttTime)
{
    uint64_t N = 1 << nBits;

    gl64_t *d_xiChallenge;
    int64_t *d_openingPoints;
    gl64_t * d_shiftedValues;

    cudaMalloc(&d_xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_xiChallenge, xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    cudaMalloc(&d_openingPoints, nOpeningPoints * sizeof(int64_t));
    cudaMemcpy(d_openingPoints, openingPoints, nOpeningPoints * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMalloc(&d_shiftedValues, nOpeningPoints * FIELD_EXTENSION * sizeof(Goldilocks::Element));

    Goldilocks::Element invShift = Goldilocks::inv(Goldilocks::shift());


    // Evaluate the shifted value for each opening point
    dim3 nThreads_(32);
    dim3 nBlocks_((nOpeningPoints + nThreads_.x - 1) / nThreads_.x);
    evalXiShifted<<<nBlocks_, nThreads_>>>(d_shiftedValues, d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints, invShift.fe);

    dim3 nThreads(1, 512);
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x, (N + nThreads.y - 1) / nThreads.y);
    fillLEv_2d<<<nBlocks, nThreads>>>(d_LEv, nOpeningPoints, N,  d_shiftedValues);
    CHECKCUDAERR(cudaGetLastError());

    cudaEvent_t point1, point2;
    cudaEventCreate(&point1);
    cudaEventCreate(&point2);
    cudaEventRecord(point1);

    NTT_Goldilocks ntt(N);
    ntt.INTT_inplace(0, N, FIELD_EXTENSION * nOpeningPoints, d_buffers, offset_helper, d_LEv);

    cudaEventRecord(point2);
    if(nttTime!= nullptr){
        cudaEventSynchronize(point2);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, point1, point2);
        *nttTime = elapsedTime/1000;
    }
    cudaEventDestroy(point1);
    cudaEventDestroy(point2);    
    CHECKCUDAERR(cudaFree(d_xiChallenge));
    CHECKCUDAERR(cudaFree(d_openingPoints));
    CHECKCUDAERR(cudaFree(d_shiftedValues));
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


void calculateXis_inplace(SetupCtx &setupCtx, StepsParams &h_params, Goldilocks::Element *xiChallenge)
{

    uint64_t nOpeningPoints = setupCtx.starkInfo.openingPoints.size();
    int64_t *openingPoints = setupCtx.starkInfo.openingPoints.data();
    uint64_t nBits = setupCtx.starkInfo.starkStruct.nBits;

    gl64_t *d_xiChallenge;
    int64_t *d_openingPoints;
    cudaMalloc(&d_xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_xiChallenge, xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    cudaMalloc(&d_openingPoints, nOpeningPoints * sizeof(int64_t));
    cudaMemcpy(d_openingPoints, openingPoints, nOpeningPoints * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    dim3 nThreads(16);
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x);
    calcXis<<<nBlocks, nThreads>>>(h_params.xDivXSub, d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints);
    CHECKCUDAERR(cudaGetLastError());
    
    CHECKCUDAERR(cudaFree(d_xiChallenge));
    CHECKCUDAERR(cudaFree(d_openingPoints));
}

__global__ void computeEvals_v2(
    uint64_t extendBits,
    uint64_t size_eval,
    uint64_t N,
    uint64_t openingsSize,
    gl64_t *d_evals,
    EvalInfo *d_evalInfo,
    gl64_t *d_cmPols,
    gl64_t *d_fixedPols,
    gl64_t *d_customComits,
    gl64_t *d_LEv)
{

    extern __shared__ Goldilocks3GPU::Element shared_sum[];
    uint64_t evalIdx = blockIdx.x;

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
                Goldilocks3GPU::mul(res, *((Goldilocks3GPU::Element *)&d_LEv[pos]), pol[evalInfo.offset + row * evalInfo.stride]);
            }
            else
            {
                Goldilocks3GPU::mul(res, *((Goldilocks3GPU::Element *)&d_LEv[pos]), *((Goldilocks3GPU::Element *)(&pol[evalInfo.offset + row * evalInfo.stride])));
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
                d_evals[evalInfo.evalPos * FIELD_EXTENSION + i] = shared_sum[0][i];
            }
        }
    }
}

void evmap_inplace(Goldilocks::Element * evals, StepsParams &h_params, FRIProof<Goldilocks::Element> &proof, Starks<Goldilocks::Element> *starks, DeviceCommitBuffers *d_buffers, uint64_t nOpeningPoints, int64_t *openingPoints, Goldilocks::Element *d_LEv)
{

    uint64_t offsetConstTree = starks->setupCtx.starkInfo.mapOffsets[std::make_pair("const", true)];
    gl64_t *d_constTree = (gl64_t *)h_params.pConstPolsExtendedTreeAddress;

    uint64_t extendBits = starks->setupCtx.starkInfo.starkStruct.nBitsExt - starks->setupCtx.starkInfo.starkStruct.nBits;
    uint64_t size_eval = starks->setupCtx.starkInfo.evMap.size();
    uint64_t N = 1 << starks->setupCtx.starkInfo.starkStruct.nBits;

    EvalInfo *evalsInfo = new EvalInfo[size_eval];

    uint64_t nEvals = 0;

    for (uint64_t i = 0; i < size_eval; i++)
    {
        EvMap ev = starks->setupCtx.starkInfo.evMap[i];
        auto it = std::find(openingPoints, openingPoints + nOpeningPoints, ev.prime);
        bool containsPrime = it != openingPoints + nOpeningPoints;
        if(!containsPrime) continue;
        string type = ev.type == EvMap::eType::cm ? "cm" : ev.type == EvMap::eType::custom ? "custom"
                                                                                           : "fixed";
        PolMap polInfo = type == "cm" ? starks->setupCtx.starkInfo.cmPolsMap[ev.id] : type == "custom" ? starks->setupCtx.starkInfo.customCommitsMap[ev.commitId][ev.id]
                                                                                                       : starks->setupCtx.starkInfo.constPolsMap[ev.id];
        evalsInfo[nEvals].type = type == "cm" ? 0 : type == "custom" ? 1
                                                                : 2; //rick: harcoded
        evalsInfo[nEvals].offset = starks->setupCtx.starkInfo.getTraceOffset(type, polInfo, true);
        evalsInfo[nEvals].stride = starks->setupCtx.starkInfo.getTraceNColsSection(type, polInfo, true);
        evalsInfo[nEvals].dim = polInfo.dim;
        evalsInfo[nEvals].openingPos = std::distance(openingPoints, it);
        evalsInfo[nEvals].evalPos = i;
        nEvals++;
    }

    EvalInfo *d_evalsInfo;
    CHECKCUDAERR(cudaMalloc(&d_evalsInfo, nEvals * sizeof(EvalInfo)));
    CHECKCUDAERR(cudaMemcpy(d_evalsInfo, evalsInfo, nEvals * sizeof(EvalInfo), cudaMemcpyHostToDevice));

    dim3 nThreads(256);
    dim3 nBlocks(nEvals);
    computeEvals_v2<<<nBlocks, nThreads, nThreads.x * sizeof(Goldilocks3GPU::Element)>>>(extendBits, nEvals, N, nOpeningPoints, (gl64_t *)h_params.evals, d_evalsInfo, (gl64_t *)d_buffers->d_aux_trace, d_constTree, (gl64_t *)h_params.pCustomCommitsFixed, (gl64_t *)d_LEv);
    CHECKCUDAERR(cudaGetLastError());

    for(uint64_t i = 0; i < nEvals; i++) {
        CHECKCUDAERR(cudaMemcpy(&evals[evalsInfo[i].evalPos * FIELD_EXTENSION], h_params.evals + evalsInfo[i].evalPos * FIELD_EXTENSION, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    }
    delete[] evalsInfo;
    CHECKCUDAERR(cudaFree(d_evalsInfo));

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

void fold_inplace(uint64_t step, uint64_t friPol_offset, uint64_t offset_helper, Goldilocks::Element *challenge, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits, DeviceCommitBuffers *d_buffers)
{

    gl64_t *d_friPol = (gl64_t *)(d_buffers->d_aux_trace + friPol_offset);
    gl64_t *d_ppar = (gl64_t *)d_buffers->d_aux_trace + offset_helper;
    gl64_t *d_challenge;
    gl64_t *d_twiddles;
    uint32_t ratio = 1 << (prevBits - currentBits);
    uint64_t halfRatio = ratio >> 1;

    uint64_t sizeFoldedPol = 1 << currentBits;

    CHECKCUDAERR(cudaMalloc(&d_challenge, FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_challenge, challenge, sizeof(Goldilocks::Element) * FIELD_EXTENSION, cudaMemcpyHostToDevice));
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
    CHECKCUDAERR(cudaGetLastError());

    CHECKCUDAERR(cudaFree(d_challenge));
    CHECKCUDAERR(cudaFree(d_twiddles));
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

void merkelizeFRI_inplace(SetupCtx& setupCtx, StepsParams &h_params, uint64_t step, FRIProof<Goldilocks::Element> &proof, gl64_t *pol, MerkleTreeGL *treeFRI, uint64_t currentBits, uint64_t nextBits, TranscriptGL_GPU *d_transcript, double * merkleTime)
{
    uint64_t pol2N = 1 << currentBits;

    uint64_t width = 1 << nextBits;
    uint64_t height = pol2N / width;
    dim3 nThreads(32, 32);
    dim3 nBlocks((width + nThreads.x - 1) / nThreads.x, (height + nThreads.y - 1) / nThreads.y);
    transposeFRI<<<nBlocks, nThreads>>>((gl64_t *)treeFRI->source, (gl64_t *)pol, pol2N, width);
    
    cudaEvent_t point1, point2;
    cudaEventCreate(&point1);
    cudaEventCreate(&point2);
    cudaEventRecord(point1);    
    Poseidon2Goldilocks::merkletree_cuda_coalesced(3, (uint64_t*) treeFRI->nodes, (uint64_t *)treeFRI->source, treeFRI->width, treeFRI->height);
    cudaEventRecord(point2);
    if(merkleTime!= nullptr){
        cudaEventSynchronize(point2);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, point1, point2);
        *merkleTime = elapsedTime/1000;
    }
    cudaEventDestroy(point1);
    cudaEventDestroy(point2);

    uint64_t tree_size = treeFRI->numNodes;
    if(d_transcript != nullptr) {
        d_transcript->put(&treeFRI->nodes[tree_size - HASH_SIZE], HASH_SIZE);
    }
    CHECKCUDAERR(cudaMemcpy(&proof.proof.fri.treesFRI[step].root[0], treeFRI->nodes + tree_size - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
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
            proofPtr[j].set_val(nodes[(offset + (si + i)) * nFieldElements + j].get_val());
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

void proveQueries_inplace(SetupCtx& setupCtx, uint64_t *friQueries, uint64_t nQueries, FRIProof<Goldilocks::Element> &fproof, MerkleTreeGL **trees, uint64_t nTrees, DeviceCommitBuffers *d_buffers, gl64_t* d_constTree, uint32_t nStages, StepsParams &d_params)
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
        if (k < nStages + 1)
        {
            std::string section = "cm" + to_string(k+1);
            uint64_t offset = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
            getTreeTracePols<<<nBlocks, nThreads>>>(d_buffers->d_aux_trace + offset, trees[k]->getMerkleTreeWidth(), d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize);
        }
        else if (k == nStages + 1)
        {
            getTreeTracePols<<<nBlocks, nThreads>>>(&d_constTree[2], trees[k]->getMerkleTreeWidth(), d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize); // rick: this last should be done in the CPU
        } else{
            uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[0].name + "0"];
            getTreeTracePols<<<nBlocks, nThreads>>>((gl64_t *)(d_params.pCustomCommitsFixed+N*nCols), trees[k]->getMerkleTreeWidth(), d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize);
        }
    }
    CHECKCUDAERR(cudaGetLastError());


    for (uint k = 0; k < nStages + 1; k++)
    {
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads>>>((gl64_t *)trees[k]->get_nodes_ptr(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE);
        CHECKCUDAERR(cudaGetLastError());
    }
    CHECKCUDAERR(cudaGetLastError());

    dim3 nthreads(64);
    dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
    genMerkleProof<<<nblocks, nthreads>>>((gl64_t *)trees[nStages + 1]->get_nodes_ptr(), trees[nStages + 1]->getMerkleTreeHeight(), d_friQueries, nQueries, d_buff + (nStages + 1) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE);
    CHECKCUDAERR(cudaGetLastError());

    if(nTrees > nStages + 2){
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads>>>((gl64_t *)trees[nStages + 2]->get_nodes_ptr(), trees[nStages + 2]->getMerkleTreeHeight(), d_friQueries, nQueries, d_buff + (nStages + 2) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE);
        CHECKCUDAERR(cudaGetLastError());
    }

    CHECKCUDAERR(cudaMemcpy(buff, d_buff, maxBuffSize * nQueries * nTrees * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaGetLastError());

    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaFree(d_buff));
    CHECKCUDAERR(cudaFree(d_friQueries));

    count = 0;
    for (uint k = 0; k < nTrees; k++)
    {
        for (uint64_t i = 0; i < nQueries; i++)
        {
            MerkleProof<Goldilocks::Element> mkProof(trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleProofLength(), trees[k]->getNumSiblings(), (void *) &buff[count * maxBuffSize], maxTreeWidth);
            fproof.proof.fri.trees.polQueries[i].push_back(mkProof);
            ++count;
        }
    }

    delete[] buff;
    return;
}

void proveFRIQueries_inplace(SetupCtx& setupCtx, uint64_t step, uint64_t currentBits, uint64_t *friQueries, uint64_t nQueries, FRIProof<Goldilocks::Element> &fproof, MerkleTreeGL *treeFRI) {
    uint64_t buffSize = treeFRI->getMerkleTreeWidth() + treeFRI->getMerkleProofSize();
    Goldilocks::Element *buff = new Goldilocks::Element[buffSize * nQueries];
    gl64_t *d_buff;
    CHECKCUDAERR(cudaMalloc(&d_buff, buffSize * nQueries * sizeof(Goldilocks::Element)));
    uint64_t *d_friQueries;
    uint64_t stepQueries[nQueries];
    for(uint64_t i = 0; i < nQueries; i++){
        stepQueries[i] = friQueries[i] % (1 << currentBits);
    }
    CHECKCUDAERR(cudaMalloc(&d_friQueries, nQueries * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(d_friQueries, stepQueries, nQueries * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaGetLastError());
    dim3 nThreads(32, 32);
    dim3 nBlocks((treeFRI->getMerkleTreeWidth() + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
    getTreeTracePols<<<nBlocks, nThreads>>>((gl64_t *)treeFRI->source, treeFRI->getMerkleTreeWidth(), d_friQueries, nQueries, d_buff, buffSize);
    CHECKCUDAERR(cudaGetLastError());
    dim3 nthreads(64);
    dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);

    genMerkleProof<<<nblocks, nthreads>>>((gl64_t *)treeFRI->nodes, treeFRI->getMerkleTreeHeight(), d_friQueries, nQueries, d_buff, buffSize, treeFRI->getMerkleTreeWidth(), HASH_SIZE);

    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaMemcpy(buff, d_buff, buffSize * nQueries * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaGetLastError());
    for (uint64_t i = 0; i < nQueries; i++)
    {
        MerkleProof<Goldilocks::Element> mkProof(treeFRI->getMerkleTreeWidth(), treeFRI->getMerkleProofLength(), treeFRI->getNumSiblings(), (void *) &buff[i * buffSize], treeFRI->getMerkleTreeWidth());
        CHECKCUDAERR(cudaGetLastError());
        fproof.proof.fri.treesFRI[step - 1].polQueries[i].push_back(mkProof);
    }
    CHECKCUDAERR(cudaGetLastError());
    delete[] buff;
    return;
}

void calculateImPolsExpressions(SetupCtx& setupCtx, ExpressionsGPU& expressionsCtx, StepsParams& h_params, StepsParams *d_params, int64_t step){

    uint64_t domainSize = (1 << setupCtx.starkInfo.starkStruct.nBits);
    std::vector<Dest> dests;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {

            Goldilocks::Element* pAddress = step == 1 ? h_params.trace : h_params.aux_trace;
            uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos;
            Dest destStruct(NULL, domainSize, setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)]);
            destStruct.addParams(setupCtx.starkInfo.cmPolsMap[i].expId, setupCtx.starkInfo.cmPolsMap[i].dim, false);
            destStruct.dest_gpu = (Goldilocks::Element *)(pAddress + offset);            
            expressionsCtx.calculateExpressions_gpu(d_params, destStruct, domainSize, false);
        }
    }
        
}

void calculateExpression(SetupCtx& setupCtx, ExpressionsGPU& expressionsCtx, StepsParams* d_params,Goldilocks::Element* dest_gpu, uint64_t expressionId, bool inverse){
    
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
    Dest destStruct(NULL, domainSize, 0, expressionId);
    destStruct.addParams(expressionId, setupCtx.expressionsBin.expressionsInfo[expressionId].destDim, inverse);
    destStruct.dest_gpu = dest_gpu;
    
    expressionsCtx.calculateExpressions_gpu(d_params, destStruct, domainSize, domainExtended);

}

void offloadCommit(uint64_t step, MerkleTreeGL **treesGL, gl64_t *d_aux_trace, FRIProof<Goldilocks::Element> &proof, SetupCtx &setupCtx)
{
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
    Goldilocks::Element *pNodes = (Goldilocks::Element*)d_aux_trace + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];
    CHECKCUDAERR(cudaMemcpy(&proof.proof.roots[step - 1][0], pNodes + tree_size - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

void offloadCommitFRI(uint64_t step, MerkleTreeGL *treeFRI, FRIProof<Goldilocks::Element> &proof, SetupCtx &setupCtx)
{
    uint64_t tree_size = treeFRI->numNodes;
    CHECKCUDAERR(cudaMemcpy(&proof.proof.fri.treesFRI[step].root[0], treeFRI->nodes + tree_size - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

void offloadFinalPol(gl64_t *d_fri_pol, FRIProof<Goldilocks::Element> &proof, uint64_t nBits) {
    for (uint64_t i = 0; i < nBits; i++)
    {
        CHECKCUDAERR(cudaMemcpy(&proof.proof.fri.pol[i][0], &d_fri_pol[i * FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    }
}

void calculateHash(Goldilocks::Element* hash, SetupCtx &setupCtx, Goldilocks::Element* buffer, uint64_t nElements) {
    TranscriptGL_GPU d_transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    d_transcript.put(buffer, nElements);
    d_transcript.getState(hash);
};
