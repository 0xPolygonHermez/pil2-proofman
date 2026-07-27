#include "starks.hpp"
#include "starks_api_internal.cuh"
#include "starks_gpu.cuh"
#ifdef USE_CUDA_GRAPH
#include "cuda_graph_cache.cuh"
#endif
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_cubic_extension.cuh"
#include "proof2zkinStark.hpp"
#include "proofman_sumcheck.cuh"

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
    uint64_t* d_words_per_row,
    const uint64_t *d_unpack_info,
    Layout layout
) {
    extern __shared__ uint64_t shared_mem[];
    uint64_t* shared_unpack_info = shared_mem;
    uint64_t words_per_row = *d_words_per_row;

    // Load unpack info
    for (uint64_t i = threadIdx.x; i < nCols; i += blockDim.x) {
        shared_unpack_info[i] = d_unpack_info[i];
    }
    __syncthreads();

    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) return;
    const uint64_t* packed_row = src + row * words_per_row;

    uint64_t word = packed_row[0];
    uint64_t word_idx = 0;
    uint64_t bit_offset = 0;

    // Storage layout passed by the caller: unpack_trace (cm1) passes resolveLayout(nBits,nCols);
    // unpack_fixed (const) passes fixedLayout(). Loop-invariant.

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

        dst[getBufferOffset(row, c, nRows, nCols, layout)] = val;
    }
}

void unpack_fixed(
    uint64_t* d_num_packed_words,
    uint64_t* d_unpack_info,
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
    TimerStartCategoryGPU(timer, UNPACK_FIXED);
    // Const pols are stored fixedLayout() (ColMajorTiled) -- uniform-native prover.
    unpack<<<blocks, threads, sharedMemSize, stream>>>(
        src,
        dst,
        nRows,
        nCols,
        d_num_packed_words,
        d_unpack_info,
        fixedLayout()
    );
    TimerStopCategoryGPU(timer, UNPACK_FIXED);
    CHECKCUDAERR(cudaGetLastError());
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
    // cm1 unpack: same storage layout the commit/LDE uses (resolveLayout on the small domain).
    unpack<<<blocks, threads, sharedMemSize, stream>>>(
        src,
        dst,
        nRows,
        nCols,
        air_instance_info->d_num_packed_words,
        air_instance_info->unpack_info,
        resolveLayout(63 - __builtin_clzll(nRows), nCols)
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
        computeQ_MerkleTree_inplace(step, setupCtx, treesGL, d_aux_trace, d_transcript, timer, stream);
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

#ifdef USE_CUDA_GRAPH
    CudaGraphCache *graphCache = cudagraph::current();
    // Only the native (ColMajorTiled) LDE is graph-capturable: the flat (ColMajor) path delegates to
    // sppark, which runs the NTT on its own private stream (joined to the caller via events) --
    // cross-stream work mid single-stream capture is illegal. isGraphCapturableLayout is the shared
    // predicate the LDE dispatch also uses, so capture can never target the sppark path.
    bool capturable = isGraphCapturableLayout(setupCtx.starkInfo.starkStruct.nBits, nCols);
    if (graphCache && capturable && !skipRecalculation && nCols > 0) {
        uint64_t nBits = setupCtx.starkInfo.starkStruct.nBits;
        uint64_t nBitsExt = setupCtx.starkInfo.starkStruct.nBitsExt;
        uint64_t arity = setupCtx.starkInfo.starkStruct.merkleTreeArity;
        uint64_t ctxId = (uint64_t)(uintptr_t)&setupCtx;
        uint64_t key = CudaGraphCache::makeKey(0x4C4445ULL ^ ctxId, nBits, nBitsExt, nCols, arity, step);
        if (graphCache->contains(key)) {
            TimerStartCategoryGPU(timer, MERKLE_TREE);
            bool launched = graphCache->tryLaunch(key, stream);
            TimerStopCategoryGPU(timer, MERKLE_TREE);
            if (launched) {
                if (d_transcript != nullptr) {
                    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
                    d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
                }
                return;
            }
        }
        if (graphCache->shouldCapture(key)) {
            if (graphCache->beginCapture(key, stream)) {
                NTTGoldilocksGPU ntt;
                ntt.LDE(dst, offset_dst, src, offset_src, nBits, nBitsExt, nCols, timer, stream,true, (gl64_t*)pNodes);
                buildMerkleTreeGPU(arity, (uint64_t*)pNodes, (uint64_t*)(dst + offset_dst), nCols, NExtended, resolveLayout(nBits, nCols), stream);
                bool launched = graphCache->endCaptureAndLaunch(stream);
                if (launched) {
                    if (d_transcript != nullptr) {
                        uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
                        d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
                    }
                    return;
                }
            }
        }
    }
#endif

    if(!skipRecalculation) {
        NTTGoldilocksGPU ntt;

        if (nCols > 0)
        {
            // Stage label carries the commit step (cm1, cm2, ...) so each is distinguishable in the log.
            PROOFMAN_SUMCHECK("proof_before_lde_cm%u", src + offset_src, ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBits) * nCols, stream, (unsigned)step);
            ntt.LDE(dst, offset_dst, src, offset_src, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, timer, stream, true, (gl64_t*)pNodes);
            PROOFMAN_SUMCHECK("proof_after_lde_cm%u", dst + offset_dst, (uint64_t)NExtended * nCols, stream, (unsigned)step);
            TimerStartCategoryGPU(timer, MERKLE_TREE);
            buildMerkleTreeGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)pNodes, (uint64_t*)(dst + offset_dst), nCols, NExtended, resolveLayout(setupCtx.starkInfo.starkStruct.nBits, nCols), stream);
            TimerStopCategoryGPU(timer, MERKLE_TREE);
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
    NTTGoldilocksGPU ntt;

    Goldilocks::Element *src = d_fixedPols;
    Goldilocks::Element *dst = d_fixedPolsExtended;
    Goldilocks::Element *pNodes = dst + nCols * NExtended;
    // Const sections are stored fixedLayout() (ColMajorTiled) -- uniform-native prover. Call the native
    // tiled LDE directly: it is out-of-place (reads src, writes dst), so the small-domain const pols
    // (reread by every expression as const operands) stay intact -- no preserve_src/preserve_scratch dance
    // needed. Merkle build reads dst in the same layout the LDE wrote. (resolveLayout would pick sppark for
    // const's nBits/nCols, so we bypass it.)
    TimerStartCategoryGPU(timer, NTT);
    ntt.ldeNativeTiled((gl64_t *)dst, (gl64_t *)src, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, stream);
    TimerStopCategoryGPU(timer, NTT);
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    buildMerkleTreeGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)pNodes, (uint64_t*)dst, nCols, NExtended, fixedLayout(), stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);
}

void computeQ_MerkleTree_inplace(uint64_t step, SetupCtx &setupCtx, MerkleTreeGL **treesGL, gl64_t *d_aux_trace,TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
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

    treesGL[step - 1]->setSource(d_aux_traceGL + offset_cmQ);
    Goldilocks::Element *pNodes = d_aux_traceGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];
    treesGL[step - 1]->setNodes(pNodes);

    if (nCols > 0)
    {
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft", false)];
        NTTGoldilocksGPU nttExtended;

#ifdef USE_CUDA_GRAPH
        // Only the native (ColMajorTiled) computeQ is graph-capturable; the flat (ColMajor) path uses
        // sppark (host sync + own stream), illegal mid-capture. isGraphCapturableLayout is the shared
        // predicate the computeQ dispatch also uses, so capture can never target the sppark path.
        bool capturable = isGraphCapturableLayout(setupCtx.starkInfo.starkStruct.nBits, nCols);
        if (cudagraph::aggressive() && capturable) {
            CudaGraphCache *graphCache = cudagraph::current();
            if (graphCache) {
                uint64_t nBits = setupCtx.starkInfo.starkStruct.nBits;
                uint64_t nBitsExt = setupCtx.starkInfo.starkStruct.nBitsExt;
                uint64_t arity = setupCtx.starkInfo.starkStruct.merkleTreeArity;
                uint64_t ctxId = (uint64_t)(uintptr_t)&setupCtx;
                uint64_t key = CudaGraphCache::makeKey(0x514D5400ULL ^ ctxId, nBits, nBitsExt, nCols, (qDeg << 32) | qDim, arity);
                if (graphCache->tryLaunch(key, stream)) {
                    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
                    if (d_transcript != nullptr) {
                        d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
                    }
                    return;
                }
                if (graphCache->shouldCapture(key)) {
                    if (graphCache->beginCapture(key, stream)) {
                        nttExtended.computeQ(offset_cmQ, offset_q, qDeg, qDim, shiftIn, nBits, nBitsExt, nCols, d_aux_trace, offset_helper, timer, stream);
                        buildMerkleTreeGPU(arity, (uint64_t*)pNodes, (uint64_t*)(d_aux_trace + offset_cmQ), nCols, NExtended, resolveLayout(nBits, nCols), stream);
                        if (graphCache->endCaptureAndLaunch(stream)) {
                            uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
                            if (d_transcript != nullptr) {
                                d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
                            }
                            return;
                        }
                    }
                }
            }
        }
#endif

        nttExtended.computeQ(offset_cmQ, offset_q, qDeg, qDim, shiftIn, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, d_aux_trace, offset_helper, timer, stream);
        TimerStartCategoryGPU(timer, MERKLE_TREE);
        buildMerkleTreeGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)pNodes, (uint64_t*)(d_aux_trace + offset_cmQ), nCols, NExtended, resolveLayout(setupCtx.starkInfo.starkStruct.nBits, nCols), stream);
        TimerStopCategoryGPU(timer, MERKLE_TREE);
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
    uint64_t i  = blockIdx.y;                  // opening point index
    uint64_t k0 = blockIdx.x * blockDim.x;     // start exponent for this block
    uint64_t row  = k0 + threadIdx.x;          // this thread's exponent index
    if (i >= nOpeningPoints || row >= N) return;

    Goldilocks3GPU::Element xi;
    xi[0] = d_shiftedValues[i * FIELD_EXTENSION + 0];
    xi[1] = d_shiftedValues[i * FIELD_EXTENSION + 1];
    xi[2] = d_shiftedValues[i * FIELD_EXTENSION + 2];

    __shared__ Goldilocks3GPU::Element basePow;

    if (threadIdx.x == 0) {
        Goldilocks3GPU::pow(xi, k0, basePow);
    }
    __syncthreads();

    Goldilocks3GPU::Element xi_t;
    Goldilocks3GPU::pow(xi, threadIdx.x, xi_t);

    Goldilocks3GPU::Element res;
    Goldilocks3GPU::mul(res, basePow, xi_t);

    Layout layout = resolveLayout(63 - __clzll(N), nOpeningPoints * FIELD_EXTENSION);
    
    d_LEv[getBufferOffset(row, i*FIELD_EXTENSION, N, nOpeningPoints * FIELD_EXTENSION, layout)] = res[0];
    d_LEv[getBufferOffset(row, i*FIELD_EXTENSION + 1, N, nOpeningPoints * FIELD_EXTENSION, layout)] = res[1];
    d_LEv[getBufferOffset(row, i*FIELD_EXTENSION + 2, N, nOpeningPoints * FIELD_EXTENSION, layout)] = res[2];
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

    dim3 nThreads(512, 1);
    dim3 nBlocks((N + nThreads.x - 1) / nThreads.x, (nOpeningPoints + nThreads.y - 1) / nThreads.y);
    fillLEv_2d<<<nBlocks, nThreads, 0, stream>>>(d_LEv, nOpeningPoints, N,  d_shiftedValues);
    TimerStopCategoryGPU(timer, LEV);
    CHECKCUDAERR(cudaGetLastError());

    TimerStartCategoryGPU(timer, NTT);
    NTTGoldilocksGPU ntt;
    ntt.INTT(d_LEv, nBits, FIELD_EXTENSION * nOpeningPoints, stream);
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
        // cm sections (type 0) follow resolveLayout (keyed on the small domain log2(N)); custom commits
        // (1) and fixed/const (2) follow fixedLayout(). d_LEv follows resolveLayout keyed on ITS OWN
        // dimensions (openingsSize*FIELD_EXTENSION cols) -- must match how evalLEv wrote it, so it works
        // under both sppark (ColMajor) and non-sppark (ColMajorTiled).
        Layout levLayout = resolveLayout(63 - __clzll(N), openingsSize * FIELD_EXTENSION);
        Layout polLayout;
        if (evalInfo.type == 0)
        {
            pol = d_cmPols;
            polLayout = resolveLayout(63 - __clzll(N), evalInfo.stageCols);
        }
        else if (evalInfo.type == 1)
        {
            pol = d_customComits;
            polLayout = fixedLayout();
        }
        else
        {
            pol = d_fixedPols;
            polLayout = fixedLayout();
        }

        for (int i = 0; i < FIELD_EXTENSION; i++)
        {
            shared_sum[threadIdx.x][i]= gl64_t(uint64_t(0));
        }
        uint64_t tid = chunkIdx * blockDim.x + threadIdx.x;
        while (tid < N)
        {
            uint64_t row = (tid << extendBits);
            uint64_t chunkBase = tid - threadIdx.x;
            Goldilocks3GPU::Element LEv;
            LEv[0] = d_LEv[getBufferOffset_pack256(chunkBase, evalInfo.openingPos * FIELD_EXTENSION, N, openingsSize * FIELD_EXTENSION, levLayout)];
            LEv[1] = d_LEv[getBufferOffset_pack256(chunkBase, evalInfo.openingPos * FIELD_EXTENSION + 1, N, openingsSize * FIELD_EXTENSION, levLayout)];
            LEv[2] = d_LEv[getBufferOffset_pack256(chunkBase, evalInfo.openingPos * FIELD_EXTENSION + 2, N, openingsSize * FIELD_EXTENSION, levLayout)];
            Goldilocks3GPU::Element res;
            if (evalInfo.dim == 1)
            {
                Goldilocks3GPU::mul(res, LEv, pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos, NExtended, evalInfo.stageCols, polLayout)]);
            }
            else
            {
                Goldilocks3GPU::Element val;
                val[0] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos, NExtended, evalInfo.stageCols, polLayout)];
                val[1] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos + 1, NExtended, evalInfo.stageCols, polLayout)];
                val[2] = pol[evalInfo.offset + getBufferOffset(row, evalInfo.stagePos + 2, NExtended, evalInfo.stageCols, polLayout)];
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
    uint64_t evalIdx = blockIdx.x * blockDim.x + threadIdx.x;
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

    dim3 nBlocks_2((nEvals + nThreads.x - 1) / nThreads.x);
    computeEvalsReduction<<<nBlocks_2, nThreads, 0, stream>>>((gl64_t *)h_params.evals, d_helper, d_evalsInfo, nEvals, n_eval_chunks);
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

__global__ void fold(uint64_t step, gl64_t *friPol, gl64_t *d_challenge, gl64_t *d_ppar, Goldilocks::Element omega_inv, uint64_t invShiftPow_, uint64_t invW_, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits)
{

    extern __shared__ gl64_t s_twiddles[];
    if (threadIdx.x == 0) {
        uint64_t halfRatio = (1 << (prevBits - currentBits)) >> 1;
        s_twiddles[0] = gl64_t(uint64_t(1));
        for (uint32_t i = 1; i < halfRatio; i++) {
            s_twiddles[i] = s_twiddles[i - 1] * gl64_t(omega_inv.fe);
        }
    }
    __syncthreads();

    uint32_t polBits = prevBits;
    uint64_t sizePol = 1 << polBits;
    uint32_t foldedPolBits = currentBits;
    uint64_t sizeFoldedPol = 1 << foldedPolBits;
    uint32_t ratio = sizePol / sizeFoldedPol;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < sizeFoldedPol)
    {

        gl64_t invShift(invShiftPow_);
        gl64_t invW(invW_);
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
        intt_tinny(ppar, ratio, prevBits - currentBits, s_twiddles, FIELD_EXTENSION);

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
    gl64_t *d_ppar = (gl64_t *)d_aux_trace + offset_helper;

    uint64_t sizeFoldedPol = 1 << currentBits;

    Goldilocks::Element omega_inv = omegas_inv_[prevBits - currentBits];

    // Precompute invShift^(2^(nBitsExt-prevBits)) on CPU to avoid redundant per-thread computation
    Goldilocks::Element invShiftPow = Goldilocks::inv(Goldilocks::shift());
    for (uint32_t j = 0; j < nBitsExt - prevBits; j++) {
        Goldilocks::square(invShiftPow, invShiftPow);
    }
    // Precompute invW on CPU
    Goldilocks::Element invW = Goldilocks::inv(Goldilocks::w(prevBits));

    dim3 nThreads(256);
    dim3 nBlocks((sizeFoldedPol + nThreads.x - 1) / nThreads.x);
    size_t sharedMem = halfRatio * sizeof(gl64_t);
    TimerStartCategoryGPU(timer, FRI);
    fold<<<nBlocks, nThreads, sharedMem, stream>>>(step, d_friPol, (gl64_t *)d_challenge, d_ppar, omega_inv, invShiftPow.fe, invW.fe, nBitsExt, prevBits, currentBits);
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
    buildMerkleTreeGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)treeFRI->nodes, (uint64_t *)treeFRI->source, treeFRI->width, treeFRI->height, Layout::RowMajor, stream);
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

__global__ void getTreeTracePolsBlocks(gl64_t *d_treeTrace, uint64_t nCols, uint64_t nRows, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, Layout layout)
{

    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < nCols && idx_y < nQueries)
    {
        uint64_t row = d_friQueries[idx_y];
        uint64_t idx_buffer = idx_y * bufferWidth + idx_x;
        uint64_t idx_trace = getBufferOffset(row, idx_x, nRows, nCols, layout);
        d_buffer[idx_buffer] = d_treeTrace[idx_trace];
    }
}

__device__ void genMerkleProof_(gl64_t *nodes, gl64_t *proof, uint64_t idx, uint64_t offset, uint64_t n, uint64_t nFieldElements, uint32_t arity, uint64_t lastLevel)
{
    if ((lastLevel == 0 && n == 1) || (lastLevel > 0 && (n <= std::pow(arity, lastLevel)))) return;

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
    genMerkleProof_(nodes, &proof[(arity - 1) * nFieldElements], nextIdx, offset + nextN * arity, nextN, nFieldElements, arity, lastLevel);
}

__global__ void genMerkleProof(gl64_t *d_nodes, uint64_t nLeaves, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, uint64_t maxTreeWidth, uint64_t nFieldElements, uint64_t arity, uint64_t lastLevel)
{

    uint64_t idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_query < nQueries)
    {
        uint64_t row = d_friQueries[idx_query];
        uint64_t idx_buffer = idx_query * bufferWidth + maxTreeWidth;
        genMerkleProof_(d_nodes, &d_buffer[idx_buffer], row, 0, nLeaves, nFieldElements, arity, lastLevel);
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
            // cm section: same storage layout the commit used (resolveLayout on the small domain).
            Layout layout = resolveLayout(setupCtx.starkInfo.starkStruct.nBits, trees[k]->getMerkleTreeWidth());
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_aux_trace + offset, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, layout);
        }
        else if (k == nStages + 1)
        {
            // Const tree leaves were written fixedLayout() (ColMajorTiled) by extendAndMerkelizeFixed.
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_constTree, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, fixedLayout());
        } else{
            uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[0].name + "0"];
            uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            // Custom commit tree leaves were written fixedLayout() (ColMajorTiled) by write_custom_commit_gpu.
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_aux_trace + offset + N*nCols, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, fixedLayout());
        }
    }
    CHECKCUDAERR(cudaGetLastError());


    for (uint k = 0; k < nStages + 1; k++)
    {
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[k]->get_nodes_ptr(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification);
        CHECKCUDAERR(cudaGetLastError());
    }
    CHECKCUDAERR(cudaGetLastError());

    dim3 nthreads(64);
    dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
    genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[nStages + 1]->get_nodes_ptr(), trees[nStages + 1]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + (nStages + 1) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification);
    CHECKCUDAERR(cudaGetLastError());

    if(nTrees > nStages + 2){
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[nStages + 2]->get_nodes_ptr(), trees[nStages + 2]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + (nStages + 2) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification);
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

    genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)treeFRI->nodes, treeFRI->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff, buffSize, treeFRI->getMerkleTreeWidth(), HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification);

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

void calculateExpressionQ(SetupCtx& setupCtx, ExpressionsGPU* expressionsCtx, StepsParams *d_params, Goldilocks::Element* dest_gpu, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU& timer, cudaStream_t stream){
    
    uint64_t domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    bool domainExtended = true;
    setupCtx.expressionsBin.expressionsInfo[setupCtx.starkInfo.cExpId].destDim = 3;
    Dest destStruct(NULL, domainSize, 0, 3, false, setupCtx.starkInfo.cExpId);
    destStruct.addParams(setupCtx.starkInfo.cExpId, setupCtx.expressionsBin.expressionsInfo[setupCtx.starkInfo.cExpId].destDim, false);
    destStruct.dest_gpu = dest_gpu;
    countId++;
    expressionsCtx->calculateExpressionsQ_gpu(d_params, destStruct, domainSize, domainExtended, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);

}

void setProof(SetupCtx &setupCtx, Goldilocks::Element *h_aux_trace, Goldilocks::Element *h_const_tree, Goldilocks::Element *proof_buffer_pinned, cudaStream_t stream) {
    uint64_t initialOffset = 0;
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t numNodes = setupCtx.starkInfo.getNumNodesMT(NExtended);
    uint64_t arity = setupCtx.starkInfo.starkStruct.merkleTreeArity;
    uint32_t lastLevelVerification = setupCtx.starkInfo.starkStruct.lastLevelVerification;
    uint64_t numNodesLevel = std::pow(arity, lastLevelVerification);
    for(uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; ++i) {
        uint64_t stage = i + 1;
        Goldilocks::Element *nodes = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt" + to_string(stage), true)];
        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + numNodes - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        initialOffset += HASH_SIZE;

        if (lastLevelVerification > 0) {
            uint64_t n = NExtended;
            uint64_t offset = 0;
            while (n > std::pow(arity, lastLevelVerification)) {
                n = (n + (arity - 1))/arity;
                offset += n * arity * HASH_SIZE;
            }

            CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + offset, n * HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

            memset(&proof_buffer_pinned[initialOffset + n * HASH_SIZE], 0, (numNodesLevel - n) * HASH_SIZE * sizeof(uint64_t));

            initialOffset += numNodesLevel * HASH_SIZE;
        }
    }

    if (lastLevelVerification > 0) {
        Goldilocks::Element *nodes = h_const_tree + NExtended * setupCtx.starkInfo.nConstants;
        uint64_t n = NExtended;
        uint64_t offset = 0;
        while (n > std::pow(arity, lastLevelVerification)) {
            n = (n + (arity - 1))/arity;
            offset += n * arity * HASH_SIZE;
        }

        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + offset, n * HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

        memset(&proof_buffer_pinned[initialOffset + n * HASH_SIZE], 0, (numNodesLevel - n) * HASH_SIZE * sizeof(uint64_t));
        
        initialOffset += numNodesLevel * HASH_SIZE;
    }

    if (lastLevelVerification > 0) {
        for (uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
            if(setupCtx.starkInfo.customCommits[i].stageWidths[0] != 0) {
                Goldilocks::Element *nodes = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)] + (N + NExtended) * setupCtx.starkInfo.customCommits[i].stageWidths[0];
                
                uint64_t n = NExtended;
                uint64_t offset = 0;
                while (n > std::pow(arity, lastLevelVerification)) {
                    n = (n + (arity - 1))/arity;
                    offset += n * arity * HASH_SIZE;
                }

                CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + offset, n * HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

                memset(&proof_buffer_pinned[initialOffset + n * HASH_SIZE], 0, (numNodesLevel - n) * HASH_SIZE * sizeof(uint64_t));
                
                initialOffset += numNodesLevel * HASH_SIZE;
                
            }
        }
    }

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; step++)
    {
        uint64_t height = 1 << setupCtx.starkInfo.starkStruct.steps[step + 1].nBits;
        uint64_t numNodes = setupCtx.starkInfo.getNumNodesMT(height);
        Goldilocks::Element *nodes = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)];
        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + numNodes - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        initialOffset += HASH_SIZE;

        if (lastLevelVerification > 0) {
            uint64_t n = height;
            uint64_t offset = 0;
            while (n > std::pow(arity, lastLevelVerification)) {
                n = (n + (arity - 1))/arity;
                offset += n * arity * HASH_SIZE;
            }

            CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + offset, n * HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

            memset(&proof_buffer_pinned[initialOffset + n * HASH_SIZE], 0, (numNodesLevel - n) * HASH_SIZE * sizeof(uint64_t));
            
            initialOffset += numNodesLevel * HASH_SIZE;
        }
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

    Goldilocks::Element *d_nonce = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("nonce", false)];
    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_nonce, sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += 1;
}

// Serialize the pinned proof buffer directly into proof_buffer, in the same layout
// FRIProof::proof2pointer produces, but without building the intermediate FRIProof object.
void writeProof(SetupCtx &setupCtx, Goldilocks::Element *proof_buffer_pinned, uint64_t *proof_buffer, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, std::string proofFile) {
    StarkInfo &starkInfo = setupCtx.starkInfo;

    uint64_t nStages = starkInfo.nStages + 1;
    uint64_t nFriSteps = starkInfo.starkStruct.steps.size() - 1;
    uint64_t nQueries = starkInfo.starkStruct.nQueries;
    uint64_t arity = starkInfo.starkStruct.merkleTreeArity;
    uint64_t lastLevelVerification = starkInfo.starkStruct.lastLevelVerification;
    uint64_t numNodesLevel = lastLevelVerification == 0 ? 0 : (uint64_t)std::pow(arity, lastLevelVerification);
    uint64_t lastLevelWords = numNodesLevel * HASH_SIZE;

    uint64_t *out = proof_buffer;
    auto emit = [&](const Goldilocks::Element *src, uint64_t count) {
        for (uint64_t i = 0; i < count; ++i) *out++ = Goldilocks::toU64(src[i]);
    };
    auto emitZeros = [&](uint64_t count) {
        for (uint64_t i = 0; i < count; ++i) *out++ = 0;
    };

    uint64_t treeHeaderStride = HASH_SIZE + (lastLevelVerification > 0 ? lastLevelWords : 0);
    Goldilocks::Element *rootsBase = proof_buffer_pinned;
    Goldilocks::Element *constLastLevel = lastLevelVerification > 0 ? rootsBase + nStages * treeHeaderStride : nullptr;
    Goldilocks::Element *customLastBase = rootsBase + nStages * treeHeaderStride + (lastLevelVerification > 0 ? lastLevelWords : 0);

    uint64_t customLastLevelCount = 0;
    if (lastLevelVerification > 0) {
        for (uint64_t i = 0; i < starkInfo.customCommits.size(); ++i) {
            if (starkInfo.customCommits[i].stageWidths[0] != 0) customLastLevelCount++;
        }
    }
    Goldilocks::Element *friHeadersBase = customLastBase + customLastLevelCount * lastLevelWords;

    uint64_t nTrees = starkInfo.nStages + starkInfo.customCommits.size() + 2;
    uint64_t queriesProofSize = (nTrees + nFriSteps) * starkInfo.maxProofBuffSize * nQueries;
    Goldilocks::Element *queries = friHeadersBase + nFriSteps * treeHeaderStride;
    Goldilocks::Element *evals = queries + queriesProofSize;
    Goldilocks::Element *airgroupValues = evals + starkInfo.evMap.size() * FIELD_EXTENSION;
    Goldilocks::Element *airValues = airgroupValues + starkInfo.airgroupValuesSize;
    uint64_t finalPolDegree = 1 << starkInfo.starkStruct.steps.back().nBits;
    Goldilocks::Element *finalPol = airValues + starkInfo.airValuesSize;
    Goldilocks::Element *nonce = finalPol + finalPolDegree * FIELD_EXTENSION;

    double arityLog2 = std::log2(arity);
    uint64_t nSiblings = (uint64_t)std::ceil(starkInfo.starkStruct.steps[0].nBits / arityLog2) - lastLevelVerification;
    uint64_t siblingWords = nSiblings * (arity - 1) * HASH_SIZE;

    // Address of query `q` in tree `tree` within the query openings block.
    auto queryBase = [&](uint64_t tree, uint64_t q) {
        return queries + (tree * nQueries + q) * starkInfo.maxProofBuffSize;
    };

    // Air group / air values: stage-1 entries hold a single base-field word, the rest are extension elements.
    Goldilocks::Element *cursor = airgroupValues;
    for (uint64_t i = 0; i < starkInfo.airgroupValuesMap.size(); ++i) {
        uint64_t width = starkInfo.airgroupValuesMap[i].stage == 1 ? 1 : FIELD_EXTENSION;
        emit(cursor, width);
        emitZeros(FIELD_EXTENSION - width);
        cursor += width;
    }
    cursor = airValues;
    for (uint64_t i = 0; i < starkInfo.airValuesMap.size(); ++i) {
        uint64_t width = starkInfo.airValuesMap[i].stage == 1 ? 1 : FIELD_EXTENSION;
        emit(cursor, width);
        emitZeros(FIELD_EXTENSION - width);
        cursor += width;
    }

    // Stage roots and evals.
    for (uint64_t i = 0; i < nStages; ++i) emit(rootsBase + i * treeHeaderStride, HASH_SIZE);
    emit(evals, starkInfo.evMap.size() * FIELD_EXTENSION);

    // Constants tree openings + siblings (+ last level).
    uint64_t constantsTree = starkInfo.nStages + 1;
    for (uint64_t q = 0; q < nQueries; ++q) emit(queryBase(constantsTree, q), starkInfo.nConstants);
    for (uint64_t q = 0; q < nQueries; ++q) emit(queryBase(constantsTree, q) + starkInfo.maxTreeWidth, siblingWords);
    if (lastLevelVerification != 0) emit(constLastLevel, lastLevelWords);

    // Custom commit trees.
    uint64_t customLastIndex = 0;
    for (uint64_t c = 0; c < starkInfo.customCommits.size(); ++c) {
        uint64_t tree = starkInfo.nStages + 2 + c;
        uint64_t width = starkInfo.mapSectionsN[starkInfo.customCommits[c].name + "0"];
        for (uint64_t q = 0; q < nQueries; ++q) emit(queryBase(tree, q), width);
        for (uint64_t q = 0; q < nQueries; ++q) emit(queryBase(tree, q) + starkInfo.maxTreeWidth, siblingWords);
        if (lastLevelVerification != 0) {
            if (starkInfo.customCommits[c].stageWidths[0] != 0) {
                emit(customLastBase + customLastIndex * lastLevelWords, lastLevelWords);
                customLastIndex++;
            } else {
                emitZeros(lastLevelWords);
            }
        }
    }

    // Committed stage trees.
    for (uint64_t s = 0; s < nStages; ++s) {
        uint64_t width = starkInfo.mapSectionsN["cm" + to_string(s + 1)];
        for (uint64_t q = 0; q < nQueries; ++q) emit(queryBase(s, q), width);
        for (uint64_t q = 0; q < nQueries; ++q) emit(queryBase(s, q) + starkInfo.maxTreeWidth, siblingWords);
        if (lastLevelVerification != 0) emit(rootsBase + s * treeHeaderStride + HASH_SIZE, lastLevelWords);
    }

    // FRI step roots, then per-step openings + siblings (+ last level).
    for (uint64_t step = 0; step < nFriSteps; ++step) emit(friHeadersBase + step * treeHeaderStride, HASH_SIZE);

    for (uint64_t step = 1; step < starkInfo.starkStruct.steps.size(); ++step) {
        uint64_t stepIndex = step - 1;
        uint64_t width = (1ULL << (starkInfo.starkStruct.steps[step - 1].nBits - starkInfo.starkStruct.steps[step].nBits)) * FIELD_EXTENSION;
        uint64_t stepSiblings = (uint64_t)std::ceil(starkInfo.starkStruct.steps[step].nBits / arityLog2) - lastLevelVerification;
        uint64_t stepSiblingWords = stepSiblings * (arity - 1) * HASH_SIZE;
        uint64_t buffSize = width + stepSiblingWords;
        Goldilocks::Element *queriesFRI = queries + (nTrees + stepIndex) * nQueries * starkInfo.maxProofBuffSize;

        for (uint64_t q = 0; q < nQueries; ++q) emit(queriesFRI + q * buffSize, width);
        for (uint64_t q = 0; q < nQueries; ++q) emit(queriesFRI + q * buffSize + width, stepSiblingWords);
        if (lastLevelVerification != 0) emit(friHeadersBase + stepIndex * treeHeaderStride + HASH_SIZE, lastLevelWords);
    }

    emit(finalPol, finalPolDegree * FIELD_EXTENSION);
    emit(nonce, 1);

    if ((uint64_t)(out - proof_buffer) != starkInfo.proofSize) {
        throw std::runtime_error("writeProof: serialized " + to_string(out - proof_buffer) + " words, expected " + to_string(starkInfo.proofSize));
    }

    if(!proofFile.empty()) {
        json2file(pointer2json(proof_buffer, setupCtx.starkInfo), proofFile);
    }
}

void calculateHash(TranscriptGL_GPU *d_transcript, Goldilocks::Element* hash, SetupCtx &setupCtx, Goldilocks::Element* buffer, uint64_t nElements, cudaStream_t stream) {
    d_transcript->reset(stream);
    d_transcript->put(buffer, nElements, stream);
    d_transcript->getState(hash, stream);
};

__global__  void computeFRIExpression(uint64_t domainSize, uint64_t nBits, uint64_t nOpeningPoints, gl64_t *d_fri, uint64_t* d_countsPerOpeningPos, EvalInfo **d_evalInfoPerOpening, gl64_t *d_evals, gl64_t *vf1, gl64_t *vf2, gl64_t *d_cmPols, gl64_t *d_xDivXSub, gl64_t *d_x, gl64_t *d_fixedPols, gl64_t *d_customComits, bool debug)
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
                // cm sections (type 0) follow resolveLayout (keyed on the small domain nBits); custom
                // commits (1) and fixed/const (2) follow fixedLayout().
                Layout polLayout = Layout::ColMajor;
                if (evalInfo.type == 0)
                {
                    pol = d_cmPols;
                    polLayout = resolveLayout(nBits, evalInfo.stageCols);
                }
                else if (evalInfo.type == 1)
                {
                    pol = d_customComits;
                    polLayout = fixedLayout();
                }
                else
                {
                    pol = d_fixedPols;
                    polLayout = fixedLayout();
                }

                gl64_t *out = (j == 0) ? accum : res;
                if(evalInfo.dim == 1) {
                    out[threadIdx.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos, domainSize, evalInfo.stageCols, polLayout)];
                    Goldilocks3GPU::sub_13_gpu_b_const(out, out, eval);
                } else {
                    out[threadIdx.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos, domainSize, evalInfo.stageCols, polLayout)];
                    out[threadIdx.x + blockDim.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos + 1, domainSize, evalInfo.stageCols, polLayout)];
                    out[threadIdx.x + 2*blockDim.x] = pol[evalInfo.offset + getBufferOffset(r, evalInfo.stagePos + 2, domainSize, evalInfo.stageCols, polLayout)];
                    Goldilocks3GPU::sub_gpu_b_const(out, out, eval);
                }
                if(j != 0) {
                    Goldilocks3GPU::mul_gpu_b_const(accum, accum, vf2);
                    Goldilocks3GPU::add_gpu_no_const(accum, accum, (gl64_t *)out);
                }
            }

            Goldilocks3GPU::sub_13_gpu_b_const(res, d_x + i, &d_xDivXSub[o * FIELD_EXTENSION]);
            Goldilocks3GPU::Element aux;
            aux[0] = res[threadIdx.x];
            aux[1] = res[blockDim.x + threadIdx.x];
            aux[2] = res[2 * blockDim.x + threadIdx.x];
            Goldilocks3GPU::inv(aux, aux);
            res[threadIdx.x] = aux[0];
            res[blockDim.x + threadIdx.x] = aux[1];
            res[2 * blockDim.x + threadIdx.x] = aux[2];

            gl64_t *out = o == 0 ? fri_pol : accum;
            Goldilocks3GPU::mul_gpu_no_const(out, accum, res);
            if(o != 0) {
                Goldilocks3GPU::mul_gpu_b_const(fri_pol, fri_pol, vf1);
                Goldilocks3GPU::add_gpu_no_const(fri_pol, fri_pol, accum);
            }
        }
        d_fri[r * FIELD_EXTENSION] = fri_pol[threadIdx.x];
        d_fri[r * FIELD_EXTENSION + 1] = fri_pol[threadIdx.x + blockDim.x];
        d_fri[r * FIELD_EXTENSION + 2] = fri_pol[threadIdx.x + 2*blockDim.x];
        chunk_idx += gridDim.x;
    }
}

void calculateFRIExpression(SetupCtx& setupCtx, StepsParams &h_params, AirInstanceInfo *air_instance_info, cudaStream_t stream) {
    Goldilocks::Element *dest = (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]);

    uint64_t domainSize = (1 << setupCtx.starkInfo.starkStruct.nBitsExt);
    // computeFRIExpression strides by domainSize/blockDim.x with no remainder handling,
    // so blockDim.x must divide domainSize or leftover rows are silently dropped.
    // Raise nrowsPack up to a 256-thread minimum (memory-constrained instances halve
    // nrowsPack below 256 and would otherwise underpopulate blocks), then cap at
    // domainSize. This is only reached on the prover path (gen_proof / gen_recursivef),
    // where nrowsPack is a power of two <= 256, so blockDim.x stays a power-of-two
    // divisor of the power-of-two domainSize. It is NOT called on the verify path, whose
    // nrowsPack = nQueries can be a non-power-of-two > 256 and would break the invariant.
    uint32_t nthreads_ = std::min<uint32_t>(std::max<uint32_t>(setupCtx.starkInfo.nrowsPack, 256), (uint32_t)domainSize);
    uint32_t nblocks_ = std::min((uint32_t)setupCtx.starkInfo.maxNBlocks, (uint32_t)((domainSize + nthreads_-1)/ nthreads_));
    size_t sharedMem = nthreads_ * 3 * FIELD_EXTENSION * sizeof(Goldilocks::Element);
    dim3 nThreads(nthreads_);    
    dim3 nBlocks(nblocks_);
    computeFRIExpression<<<nBlocks, nThreads, sharedMem, stream>>>(
        domainSize,
        setupCtx.starkInfo.starkStruct.nBits,
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
        false
    );
    CHECKCUDAERR(cudaGetLastError());
}