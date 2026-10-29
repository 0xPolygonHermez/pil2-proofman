#include "starks.hpp"
#include "starks_api_internal.cuh"
#include "starks_gpu.cuh"
#include "unpack_indexed_device.cuh"
#ifdef USE_CUDA_GRAPH
#include "cuda_graph_cache.cuh"
#endif
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_cubic_extension.cuh"
#include "deep_expression.cuh"
#include "proof2zkinStark.hpp"
#include "proofman_sumcheck.cuh"
#include "stir/stir_layout.hpp"

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
    // Const pols are stored fixedLayout() (ColMajor).
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
    Layout layout = resolveLayout(63 - __builtin_clzll(nRows), nCols);
    TimerStartCategoryGPU(timer, UNPACK_TRACE);
    // d_col_source (set at setup from PackedInfo) is what makes an air indexed; the
    // table arrives later per program. Dispatch on the descriptor, NOT on the table:
    // an indexed air with no table must abort, because the plain walk would happily
    // decode compact rows as full ones and yield a silently wrong trace.
    if (air_instance_info->d_col_source != nullptr) {
        if (air_instance_info->d_instr_table == nullptr) {
            zklog.error("unpack_trace: air (" + std::to_string(air_instance_info->airgroupId) + "," +
                        std::to_string(air_instance_info->airId) + ") is indexed but no instruction "
                        "table is registered; call register_instruction_table first");
            exitProcess();
        }
        // Without the lane map every column would decode from lane 0's entry: a wrong
        // trace with no other symptom.
        if (air_instance_info->lanes > 1 && air_instance_info->d_col_lane == nullptr) {
            zklog.error("unpack_trace: air (" + std::to_string(air_instance_info->airgroupId) + "," +
                        std::to_string(air_instance_info->airId) + ") packs " +
                        std::to_string(air_instance_info->lanes) + " lanes per row but carries no "
                        "col_lane map");
            exitProcess();
        }
        // Indexed cm1 unpack: compact rows + shared instruction table reconstruct the full
        // nCols output. Same storage layout as the plain path.
        unpack_indexed<<<blocks, threads, sharedMemSize, stream>>>(
            src,
            air_instance_info->d_instr_table,
            dst,
            nRows,
            nCols,
            air_instance_info->num_packed_words,
            air_instance_info->words_per_entry,
            air_instance_info->unpack_info,
            air_instance_info->d_col_source,
            air_instance_info->d_col_lane,
            air_instance_info->index_bits,
            air_instance_info->lanes,
            air_instance_info->num_entries,
            layout
        );
    } else {
        // cm1 unpack: same storage layout the commit/LDE uses (resolveLayout on the small domain).
        unpack<<<blocks, threads, sharedMemSize, stream>>>(
            src,
            dst,
            nRows,
            nCols,
            air_instance_info->d_num_packed_words,
            air_instance_info->unpack_info,
            layout
        );
    }
    TimerStopCategoryGPU(timer, UNPACK_TRACE);
    CHECKCUDAERR(cudaGetLastError());
}

// Entries per thread of buildZerofier_kernel: one inversion each (Montgomery's trick).
#define ZEROFIER_BATCH 8

void computeZerofier(Goldilocks::Element *d_zi, StarkInfo &starkInfo, cudaStream_t stream) {
    uint64_t nBits = starkInfo.starkStruct.nBits;
    uint64_t nBitsExt = starkInfo.starkStruct.nBitsExt;
    for (uint64_t b = 0; b < starkInfo.boundaries.size(); b++) {
        const Boundary &boundary = starkInfo.boundaries[b];
        ZerofierTable t = {};
        t.everyRow = boundary.name == "everyRow";
        t.complement = boundary.complement;
        uint64_t pBits = periodBits(boundary, nBits, nBitsExt);
        t.P = uint64_t(1) << pBits;
        t.wP = Goldilocks::toU64(Goldilocks::w(pBits));
        t.cycleBits = t.everyRow ? 0 : boundary.cycleBits;
        t.nRoots = t.everyRow ? 0 : boundary.offsets.size();
        Goldilocks::Element roots[MAX_DOMAIN_OFFSETS];
        t.shiftM = Goldilocks::toU64(domainRoots(boundary, nBits, roots));
        for (uint32_t k = 0; k < t.nRoots; k++) t.roots[k] = Goldilocks::toU64(roots[k]);
        dim3 threads(256);
        dim3 blocks((t.P + threads.x * ZEROFIER_BATCH - 1) / (threads.x * ZEROFIER_BATCH));
        buildZerofier_kernel<<<blocks, threads, 0, stream>>>((gl64_t *)d_zi + starkInfo.ziOffsets[b], t);
    }
    CHECKCUDAERR(cudaGetLastError());
}

// ProverHelpers::buildZHInv / buildDomainZerofier at x^m = shift^m * w(log P)^i, stored canonical.
__global__ void buildZerofier_kernel(gl64_t *d_zi, ZerofierTable t) {
    uint64_t i0 = (threadIdx.x + blockIdx.x * (uint64_t)blockDim.x) * ZEROFIER_BATCH;
    if (i0 >= t.P) return;
    uint32_t n = (uint32_t)min((uint64_t)ZEROFIER_BATCH, t.P - i0);
    uint64_t *out = (uint64_t *)d_zi + i0;
    gl64_t one(uint64_t(1));
    gl64_t xm = gl64_t(t.shiftM) * (gl64_t(t.wP) ^ (uint32_t)i0);
    gl64_t num[ZEROFIER_BATCH], den[ZEROFIER_BATCH], prefix[ZEROFIER_BATCH];
    gl64_t acc = one;
    for (uint32_t j = 0; j < n; j++) {
        if (t.everyRow) {
            num[j] = one;
            den[j] = xm - one;
        } else {
            gl64_t v = one;
            for (uint32_t k = 0; k < t.nRoots; k++) v = v * (xm - gl64_t(t.roots[k]));
            gl64_t zh = xm;
            for (uint64_t c = 0; c < t.cycleBits; c++) zh = zh * zh;
            num[j] = zh - one;
            den[j] = v;
        }
        if (t.complement) out[j] = (uint64_t)den[j];
        prefix[j] = acc;
        acc = acc * den[j];
        xm = xm * gl64_t(t.wP);
    }
    if (t.complement) return;
    gl64_t inv = acc.reciprocal();
    for (int j = (int)n - 1; j >= 0; j--) {
        out[j] = (uint64_t)(num[j] * (inv * prefix[j]));
        inv = inv * den[j];
    }
}

__global__ void setProdIdentity3(gl64_t *pol) {
    pol[0] = gl64_t(uint64_t(1));
    pol[1] = gl64_t(uint64_t(0));
    pol[2] = gl64_t(uint64_t(0));
}

static void buildCommitTreeGPU(bool dropLeaves, uint32_t arity, uint64_t *nodes, uint64_t *src, uint64_t nCols,
                               uint64_t nRows, Layout layout, cudaStream_t stream)
{
    if (dropLeaves) buildMerkleTreeNoLeavesGPU(arity, nodes, src, nCols, nRows, layout, stream);
    else buildMerkleTreeGPU(arity, nodes, src, nCols, nRows, layout, stream);
}

void commitStage_inplace(uint64_t step, SetupCtx &setupCtx, MerkleTreeGL **treesGL, gl64_t *d_trace, gl64_t *d_aux_trace, TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream, uint32_t qStepLg)
{
    if (step <= setupCtx.starkInfo.nStages)
    {
    
        extendAndMerkelize_inplace(step, setupCtx, treesGL, d_trace, d_aux_trace, d_transcript, timer, stream);
    }
    else
    {
        computeQ_MerkleTree_inplace(step, setupCtx, treesGL, d_aux_trace, d_transcript, timer, stream, qStepLg);
    }
}

void extendAndMerkelize_inplace(uint64_t step, SetupCtx& setupCtx, MerkleTreeGL** treesGL, gl64_t *d_trace, gl64_t *d_aux_trace, TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
{
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    gl64_t *src = step == 1 ? d_trace : d_aux_trace;
    uint64_t offset_src = step == 1 ? 0 : setupCtx.starkInfo.mapOffsets[make_pair(section, false)];
    gl64_t *dst = d_aux_trace;
    uint64_t offset_dst = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    Goldilocks::Element * dstGL = (Goldilocks::Element*) (d_aux_trace);

    // source/nodes were set by genProof_gpu right after the Starks ctor (outside the capture
    // regions this runs in); a setter here would be skipped on replay.
    Goldilocks::Element *pNodes = dstGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];

    NTTGoldilocksGPU ntt;

    if (nCols > 0)
    {
        // Stage label carries the commit step (cm1, cm2, ...) so each is distinguishable in the log.
        PROOFMAN_SUMCHECK("proof_before_lde_cm%u", src + offset_src, ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBits) * nCols, stream, (unsigned)step);
        // pNodes is LDE scratch until the merkelize fills it. An in-place stage (equal bases) cannot
        // preserve its source.
        const bool aliased = (src + offset_src) == (dst + offset_dst);
        ntt.LDE(dst, offset_dst, src, offset_src, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, timer, stream, !aliased, (gl64_t*)pNodes, setupCtx.starkInfo.getNumNodesMTCommit(NExtended));
        PROOFMAN_SUMCHECK("proof_after_lde_cm%u", dst + offset_dst, (uint64_t)NExtended * nCols, stream, (unsigned)step);
        TimerStartCategoryGPU(timer, MERKLE_TREE);
        buildCommitTreeGPU(setupCtx.starkInfo.dropLeafLevel, setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)pNodes, (uint64_t*)(dst + offset_dst), nCols, NExtended, resolveLayout(setupCtx.starkInfo.starkStruct.nBits, nCols), stream);
        TimerStopCategoryGPU(timer, MERKLE_TREE);
    }

    if (nCols > 0)
    {
        uint64_t tree_size = setupCtx.starkInfo.getNumNodesMTCommit(NExtended);
        PROOFMAN_SUMCHECK("proof_root_cm%u", &pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream, (unsigned)step);
        if(d_transcript != nullptr) {
            d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
        }
    }
}

// preserve_src: must the unpacked const pols survive? Yes whenever a later proof of the same air
// can reuse them instead of re-unpacking -- so for everything except an aliased air.
// Extend one fixed/preprocessed section and build its Merkle tree in place. Sections are stored
// fixedLayout() (ColMajor); pNodes sits above the LDE's writes, so it doubles as LDE scratch.
void extendAndMerkelizeSection(uint64_t nCols, uint64_t nBits, uint64_t nBitsExt, uint64_t arity, uint64_t numNodes, Goldilocks::Element *d_pols, Goldilocks::Element *d_polsExtended, bool preserve_src, TimerGPU &timer, cudaStream_t stream, bool dropLeaves) {
    uint64_t NExtended = 1ull << nBitsExt;
    NTTGoldilocksGPU ntt;
    Goldilocks::Element *pNodes = d_polsExtended + nCols * NExtended;
    TimerStartCategoryGPU(timer, NTT);
    ntt.ldeColMajor((gl64_t *)d_polsExtended, (gl64_t *)d_pols, nBits, nBitsExt, nCols, stream, preserve_src, (gl64_t *)pNodes, numNodes);
    TimerStopCategoryGPU(timer, NTT);
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    buildCommitTreeGPU(dropLeaves, arity, (uint64_t*)pNodes, (uint64_t*)d_polsExtended, nCols, NExtended, fixedLayout(), stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);
}

void extendAndMerkelizeFixed(SetupCtx& setupCtx, Goldilocks::Element *d_fixedPols, Goldilocks::Element *d_fixedPolsExtended, bool preserve_src, TimerGPU &timer, cudaStream_t stream) {
    if (!setupCtx.starkInfo.hasConstants()) {
        zklog.error("extendAndMerkelizeFixed: the air has no fixed columns");
        exitProcess();
    }
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    extendAndMerkelizeSection(setupCtx.starkInfo.nConstants, setupCtx.starkInfo.starkStruct.nBits,
                              setupCtx.starkInfo.starkStruct.nBitsExt,
                              setupCtx.starkInfo.starkStruct.merkleTreeArity,
                              setupCtx.starkInfo.getNumNodesMTCommit(NExtended),
                              d_fixedPols, d_fixedPolsExtended, preserve_src, timer, stream,
                              setupCtx.starkInfo.dropLeafLevel);
}

void computeQ_MerkleTree_inplace(uint64_t step, SetupCtx &setupCtx, MerkleTreeGL **treesGL, gl64_t *d_aux_trace,TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream, uint32_t qStepLg)
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

    Goldilocks::Element *pNodes = d_aux_traceGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];

    if (nCols > 0)
    {
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft", false)];
        NTTGoldilocksGPU nttExtended;

        nttExtended.computeQ(offset_cmQ, offset_q, qDeg, qDim, shiftIn, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, d_aux_trace, offset_helper, timer, stream, qStepLg);
        TimerStartCategoryGPU(timer, MERKLE_TREE);
        buildCommitTreeGPU(setupCtx.starkInfo.dropLeafLevel, setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)pNodes, (uint64_t*)(d_aux_trace + offset_cmQ), nCols, NExtended, resolveLayout(setupCtx.starkInfo.starkStruct.nBits, nCols), stream);
        TimerStopCategoryGPU(timer, MERKLE_TREE);
        uint64_t tree_size = setupCtx.starkInfo.getNumNodesMTCommit(NExtended);
        PROOFMAN_SUMCHECK("proof_root_cm%u", &pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream, (unsigned)step);
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

// Field exponentiation with a 64-bit exponent 
static __device__ __forceinline__ gl64_t gl64Pow(gl64_t base, uint64_t e)
{
    gl64_t acc(uint64_t(1));
    while (e) {
        if (e & 1) acc *= base;
        base *= base;
        e >>= 1;
    }
    return acc;
}

// LEv is N x FIELD_EXTENSION, ColMajor.
static __device__ __forceinline__ void storeLEv(gl64_t *d_LEv, uint64_t row, uint64_t N, const Goldilocks3GPU::Element &v)
{
    for (uint32_t k = 0; k < FIELD_EXTENSION; ++k) d_LEv[k * N + row] = v[k];
}

// LEv over the first N extended rows x_j = s g^j, contiguous so the evals read whole sectors:
// L_j(y) = M(y) / ((y - g^j) D_j), M = prod_k (y - g^k), D_j = (-1)^(N-1-j) g^(j(j-1)/2 + j(N-1-j)) A_j A_(N-1-j),
// A_m = prod_{t=1..m} (g^t - 1). LEV_FILL_BATCH = 4: 8 spills more than its shared inversion saves.
constexpr uint32_t LEV_THREADS = 256, LEV_SCAN_BATCH = 8, LEV_FILL_BATCH = 4, LEV_CHAIN_THREADS = 1024;
constexpr uint64_t LEV_CHUNK = (uint64_t)LEV_THREADS * LEV_SCAN_BATCH;

// y = xi / s, the opening point in the nodes' frame.
static __device__ __forceinline__ void levY(Goldilocks3GPU::Element &y, const gl64_t *dXi, uint64_t invShift_)
{
    Goldilocks3GPU::Element xi = {dXi[0], dXi[1], dXi[2]};
    Goldilocks3GPU::mul(y, xi, gl64_t(invShift_));
}

// Product scans over a warp and a block; a shift fills lane 0 with one.
static __device__ __forceinline__ gl64_t levShflUp(gl64_t v, uint32_t off, uint32_t lane)
{
    gl64_t o;
    o[0] = (uint64_t)__shfl_up_sync(0xffffffffu, (unsigned long long)v[0], off);
    return lane >= off ? o : gl64_t(uint64_t(1));
}

static __device__ __forceinline__ gl64_t levWarpScan(gl64_t v, uint32_t lane)
{
    #pragma unroll
    for (uint32_t off = 1; off < 32; off <<= 1) v *= levShflUp(v, off, lane);
    return v;
}

// Exclusive block prefix product; every thread gets the total. sh: blockDim/32 + 1 elements.
static __device__ __forceinline__ gl64_t levBlockScan(gl64_t v, gl64_t &total, Goldilocks3GPU::Element *sh)
{
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nWarps = blockDim.x >> 5;
    const gl64_t inc = levWarpScan(v, lane);
    if (lane == 31) sh[warp][0] = inc;
    __syncthreads();
    if (warp == 0) {
        const gl64_t w = lane < nWarps ? sh[lane][0] : gl64_t(uint64_t(1));
        const gl64_t wInc = levWarpScan(w, lane);
        const gl64_t wEx = levShflUp(wInc, 1, lane);
        if (lane < nWarps) sh[lane][0] = wEx;
        if (lane == nWarps - 1) sh[nWarps][0] = wInc;
    }
    __syncthreads();
    const gl64_t ex = sh[warp][0] * levShflUp(inc, 1, lane);
    total = sh[nWarps][0];
    __syncthreads();
    return ex;
}

// Product of m over the block, valid in thread 0. sh holds blockDim/32 elements.
static __device__ __forceinline__ void levBlockProd3(Goldilocks3GPU::Element &m, Goldilocks3GPU::Element *sh)
{
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nWarps = blockDim.x >> 5;
    Goldilocks3GPU::Element o;
    #pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        for (uint32_t k = 0; k < FIELD_EXTENSION; k++)
            o[k][0] = (uint64_t)__shfl_down_sync(0xffffffffu, (unsigned long long)m[k][0], off);
        if (lane + off < 32) Goldilocks3GPU::mul(m, m, o);
    }
    if (lane == 0) Goldilocks3GPU::copy(sh[warp], m);
    __syncthreads();
    if (threadIdx.x == 0)
        for (uint32_t w = 1; w < nWarps; w++) Goldilocks3GPU::mul(m, m, sh[w]);
    __syncthreads();
}

// Per chunk: local prefix of a_m (a_0 = 1, a_m = g^m - 1) into dA, its total into dTot, its M(y) share into dM.
// A single chunk has nothing to chain: y and M(y) go straight to dYM.
__global__ void levChunkScan(gl64_t *dA, gl64_t *dTot, gl64_t *dM, uint64_t N, uint64_t g_, const gl64_t *dXi, uint64_t invShift_,
                             gl64_t *dYM)
{
    extern __shared__ Goldilocks3GPU::Element levShared[];
    const uint64_t base = (uint64_t)blockIdx.x * LEV_CHUNK + (uint64_t)threadIdx.x * LEV_SCAN_BATCH;
    Goldilocks3GPU::Element y, m, d;
    levY(y, dXi, invShift_);
    const gl64_t g(g_), one(uint64_t(1));
    gl64_t gm = gl64Pow(g, base), run = one, pre[LEV_SCAN_BATCH];
    Goldilocks3GPU::one(m);
    #pragma unroll
    for (uint32_t b = 0; b < LEV_SCAN_BATCH; b++) {
        if (base + b < N) {
            run *= (base + b == 0) ? one : (gm - one);
            Goldilocks3GPU::copy(d, y);
            d[0] = d[0] - gm;
            Goldilocks3GPU::mul(m, m, d);
        }
        pre[b] = run;
        gm *= g;
    }
    gl64_t total;
    const gl64_t ex = levBlockScan(run, total, levShared);
    levBlockProd3(m, levShared);
    if (threadIdx.x == 0 && gridDim.x == 1) {
        dTot[0] = one;
        for (uint32_t k = 0; k < FIELD_EXTENSION; k++) {
            dYM[k] = y[k];
            dYM[FIELD_EXTENSION + k] = m[k];
        }
    } else if (threadIdx.x == 0) {
        dTot[blockIdx.x] = total;
        for (uint32_t k = 0; k < FIELD_EXTENSION; k++) dM[blockIdx.x * FIELD_EXTENSION + k] = m[k];
    }
    #pragma unroll
    for (uint32_t b = 0; b < LEV_SCAN_BATCH; b++)
        if (base + b < N) dA[base + b] = ex * pre[b];
}

// One block chains the chunks (dTot becomes exclusive prefixes) and writes y, M(y) into dYM.
__global__ void __launch_bounds__(LEV_CHAIN_THREADS) levChainChunks(gl64_t *dTot, const gl64_t *dM, uint64_t nChunks, const gl64_t *dXi, uint64_t invShift_,
                               gl64_t *dYM)
{
    extern __shared__ Goldilocks3GPU::Element levShared[];
    const uint64_t per = (nChunks + blockDim.x - 1) / blockDim.x;
    const uint64_t c0 = (uint64_t)threadIdx.x * per, c1 = min(c0 + per, nChunks);
    gl64_t run(uint64_t(1));
    Goldilocks3GPU::Element m, p;
    Goldilocks3GPU::one(m);
    for (uint64_t c = c0; c < c1; c++) {
        run *= dTot[c];
        for (uint32_t k = 0; k < FIELD_EXTENSION; k++) p[k] = dM[c * FIELD_EXTENSION + k];
        Goldilocks3GPU::mul(m, m, p);
    }
    gl64_t total;
    gl64_t acc = levBlockScan(run, total, levShared);
    for (uint64_t c = c0; c < c1; c++) {
        const gl64_t v = dTot[c];
        dTot[c] = acc;
        acc *= v;
    }
    levBlockProd3(m, levShared);
    if (threadIdx.x == 0) {
        Goldilocks3GPU::Element y;
        levY(y, dXi, invShift_);
        for (uint32_t k = 0; k < FIELD_EXTENSION; k++) {
            dYM[k] = y[k];
            dYM[FIELD_EXTENSION + k] = m[k];
        }
    }
}

// D_j, LEV_SCAN_BATCH rows per thread: e_j steps by N-2-j, so one power per thread, then two muls per row.
__global__ void levNodeDen(gl64_t *dD, const gl64_t *dA, const gl64_t *dTot, uint64_t N, uint64_t lgNExt, uint64_t g_,
                           uint64_t gInv_)
{
    const uint64_t j0 = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * LEV_SCAN_BATCH;
    if (j0 >= N) return;
    const uint64_t mask = (1ull << lgNExt) - 1;
    const gl64_t g(g_), gInv(gInv_);
    // Exponents mod NExt (g's order); j(j-1) is 0 at j = 0 despite the wrap.
    gl64_t gE = gl64Pow(g, (j0 * (j0 - 1) / 2 + j0 * (N - 1 - j0)) & mask);
    gl64_t step = gl64Pow(g, (N - 2 - j0) & mask);
    #pragma unroll
    for (uint32_t b = 0; b < LEV_SCAN_BATCH; b++) {
        const uint64_t j = j0 + b;
        if (j >= N) break;
        const uint64_t r = N - 1 - j;
        gl64_t D = gE * (dA[j] * dTot[j / LEV_CHUNK]) * (dA[r] * dTot[r / LEV_CHUNK]);
        if (r & 1) D = gl64_t(uint64_t(0)) - D;
        dD[j] = D;
        gE *= step;
        step *= gInv;
    }
}

// LEv[j] = M(y) / ((y - g^j) D_j), D_j from plane 0 (own rows only); M(y) == 0: y is a node, LEv its indicator.
template<uint32_t BATCH>
__global__ void fillLEvContiguous(gl64_t *d_LEv, uint64_t N, const gl64_t *dYM, uint64_t g_)
{
    const uint64_t row0 = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * BATCH;
    if (row0 >= N) return;
    const uint32_t count = (uint32_t)min((uint64_t)BATCH, N - row0);
    Goldilocks3GPU::Element y = {dYM[0], dYM[1], dYM[2]}, M = {dYM[3], dYM[4], dYM[5]};
    const bool mZero = M[0].is_zero() && M[1].is_zero() && M[2].is_zero();
    const gl64_t g(g_);
    gl64_t gj = gl64Pow(g, row0), roots[BATCH], D[BATCH];
    Goldilocks3GPU::Element prefix[BATCH], diff, den, out;
    #pragma unroll
    for (uint32_t b = 0; b < BATCH; b++) {
        if (b >= count) break;
        roots[b] = gj;
        D[b] = d_LEv[row0 + b];
        gj *= g;
    }
    if (mZero) {
        for (uint32_t b = 0; b < count; b++) {
            Goldilocks3GPU::copy(diff, y);
            diff[0] = diff[0] - roots[b];
            const bool match = diff[0].is_zero() && diff[1].is_zero() && diff[2].is_zero();
            out[0] = match ? gl64_t(uint64_t(1)) : gl64_t(uint64_t(0));
            out[1] = gl64_t(uint64_t(0));
            out[2] = gl64_t(uint64_t(0));
            storeLEv(d_LEv, row0 + b, N, out);
        }
        return;
    }
    #pragma unroll
    for (uint32_t b = 0; b < BATCH; b++) {
        if (b >= count) break;
        Goldilocks3GPU::copy(diff, y);
        diff[0] = diff[0] - roots[b];
        Goldilocks3GPU::mul(den, diff, D[b]);
        if (b == 0) Goldilocks3GPU::copy(prefix[0], den);
        else Goldilocks3GPU::mul(prefix[b], prefix[b - 1], den);
    }
    Goldilocks3GPU::Element inv, weight;
    Goldilocks3GPU::inv(inv, prefix[count - 1]);
    for (uint32_t b = count - 1; b > 0; --b) {
        Goldilocks3GPU::mul(weight, inv, prefix[b - 1]);
        Goldilocks3GPU::mul(out, M, weight);
        storeLEv(d_LEv, row0 + b, N, out);
        Goldilocks3GPU::copy(diff, y);
        diff[0] = diff[0] - roots[b];
        Goldilocks3GPU::mul(den, diff, D[b]);
        Goldilocks3GPU::mul(inv, inv, den);
    }
    Goldilocks3GPU::mul(out, M, inv);
    storeLEv(d_LEv, row0, N, out);
}

// The four passes, given the domain sizes; dXi is the challenge, dYM receives y and M(y).
static void levLaunch(const gl64_t *dXi, uint64_t nBits, uint64_t nBitsExt, gl64_t *dYM, gl64_t *d_LEv, cudaStream_t stream)
{
    const uint64_t N = 1ull << nBits;
    const uint64_t nChunks = (N + LEV_CHUNK - 1) / LEV_CHUNK;
    const uint64_t invShift = Goldilocks::inv(Goldilocks::shift()).fe;
    const uint64_t g = Goldilocks::w(nBitsExt).fe;
    gl64_t *plane0 = d_LEv, *plane1 = d_LEv + N, *plane2 = d_LEv + 2 * N;
    const size_t scanShared = (LEV_THREADS / 32 + 1) * sizeof(Goldilocks3GPU::Element);
    const size_t chainShared = (LEV_CHAIN_THREADS / 32 + 1) * sizeof(Goldilocks3GPU::Element);
    // Several chunks mean N > LEV_CHUNK, so their M(y) shares always fit plane 0.
    levChunkScan<<<(unsigned)nChunks, LEV_THREADS, scanShared, stream>>>(plane1, plane2, plane0, N, g, dXi, invShift, dYM);
    if (nChunks > 1)
        levChainChunks<<<1, LEV_CHAIN_THREADS, chainShared, stream>>>(plane2, plane0, nChunks, dXi, invShift, dYM);
    const uint64_t denRows = (uint64_t)LEV_THREADS * LEV_SCAN_BATCH;
    levNodeDen<<<(unsigned)((N + denRows - 1) / denRows), LEV_THREADS, 0, stream>>>(plane0, plane1, plane2, N, nBitsExt, g,
                                                                                   Goldilocks::inv(Goldilocks::w(nBitsExt)).fe);
    const uint64_t fillRows = (uint64_t)LEV_THREADS * LEV_FILL_BATCH;
    fillLEvContiguous<LEV_FILL_BATCH><<<(unsigned)((N + fillRows - 1) / fillRows), LEV_THREADS, 0, stream>>>(d_LEv, N, dYM, g);
}

// Opening 0's LEv, shared by every opening; d_yM gets y and M(y), the LEv planes are the scratch.
void computeLEv_inplace(Goldilocks::Element *d_xiChallenge, uint64_t nBits, uint64_t nBitsExt, gl64_t *d_yM, gl64_t *d_LEv, TimerGPU &timer, cudaStream_t stream)
{
    TimerStartCategoryGPU(timer, LEV);
    levLaunch((const gl64_t *)d_xiChallenge, nBits, nBitsExt, d_yM, d_LEv, stream);
    TimerStopCategoryGPU(timer, LEV);
    CHECKCUDAERR(cudaGetLastError());
}

// p(xi w^o) = SUM_i L_i(y) ext[(i + o*b) mod NExt] (w = g^b): opening 0's weights serve every opening,
// and a group reads its column once for all its openings.
// grid (groups, stripes); each eval's stripe partial goes to partials[evalPos][stripe].
__global__ void computeEvalsShifted(uint64_t N, uint64_t extendBits, const EvalGroup *d_groups,
                                    const gl64_t *d_cmPols, const gl64_t *d_customCommits,
                                    const gl64_t *d_fixedPols, const gl64_t *d_LEv, gl64_t *d_partials)
{
    extern __shared__ Goldilocks3GPU::Element warpSums[];   // [nWarps][EVALS_GROUP_OPENINGS]
    const EvalGroup &g = d_groups[blockIdx.x];
    const uint32_t nOpen = g.nOpen, dim = g.dim;
    const uint64_t NExt = N << extendBits;
    const gl64_t *pol = (g.src == 0 ? d_cmPols : g.src == 1 ? d_customCommits : d_fixedPols) + g.col;
    uint64_t shift[EVALS_GROUP_OPENINGS];
    Goldilocks3GPU::Element acc[EVALS_GROUP_OPENINGS];
    #pragma unroll
    for (uint32_t o = 0; o < EVALS_GROUP_OPENINGS; o++) {
        shift[o] = o < nOpen ? g.shift[o] : 0;
        Goldilocks3GPU::zero(acc[o]);
    }

    for (uint64_t i = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x; i < N; i += (uint64_t)blockDim.x * gridDim.y) {
        Goldilocks3GPU::Element L = {d_LEv[i], d_LEv[N + i], d_LEv[2 * N + i]};
        #pragma unroll
        for (uint32_t o = 0; o < EVALS_GROUP_OPENINGS; o++) {
            if (o >= nOpen) break;
            const uint64_t row = (i + (shift[o] << extendBits)) & (NExt - 1);
            Goldilocks3GPU::Element res;
            if (dim == 1) {
                gl64_t v = pol[row];
                Goldilocks3GPU::mul(res, L, v);
            } else {
                Goldilocks3GPU::Element v = {pol[row], pol[NExt + row], pol[2 * NExt + row]};
                Goldilocks3GPU::mul(res, L, v);
            }
            Goldilocks3GPU::add(acc[o], acc[o], res);
        }
    }

    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nWarps = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (uint32_t o = 0; o < EVALS_GROUP_OPENINGS; o++) {
        if (o >= nOpen) break;
        for (uint32_t off = 16; off > 0; off >>= 1)
            for (uint32_t k = 0; k < FIELD_EXTENSION; k++) {
                gl64_t other;
                other[0] = (uint64_t)__shfl_down_sync(0xffffffffu, (unsigned long long)acc[o][k][0], off);
                if (lane < off) acc[o][k] += other;
            }
        if (lane == 0) Goldilocks3GPU::copy(warpSums[warp * EVALS_GROUP_OPENINGS + o], acc[o]);
    }
    __syncthreads();
    if (threadIdx.x < nOpen * FIELD_EXTENSION) {
        const uint32_t o = threadIdx.x / FIELD_EXTENSION, k = threadIdx.x % FIELD_EXTENSION;
        gl64_t s = warpSums[o][k];
        for (uint32_t w = 1; w < nWarps; w++) s += warpSums[w * EVALS_GROUP_OPENINGS + o][k];
        d_partials[((uint64_t)g.evalPos[o] * gridDim.y + blockIdx.y) * FIELD_EXTENSION + k] = s;
    }
}

__global__ void reduceEvalsShifted(uint64_t nEvals, uint64_t nStripes, const gl64_t *d_partials, gl64_t *d_evals)
{
    const uint64_t e = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEvals) return;
    for (uint32_t k = 0; k < FIELD_EXTENSION; k++) {
        gl64_t s = d_partials[e * nStripes * FIELD_EXTENSION + k];
        for (uint64_t c = 1; c < nStripes; c++) s += d_partials[(e * nStripes + c) * FIELD_EXTENSION + k];
        d_evals[e * FIELD_EXTENSION + k] = s;
    }
}

// Every eval of the map, written in full (no memset needed), from opening 0's Lagrange vector.
void evmap_inplace(SetupCtx &setupCtx, StepsParams &h_params, AirInstanceInfo *air_instance_info, Goldilocks::Element *d_LEv, TimerGPU &timer, cudaStream_t stream)
{
    TimerStartCategoryGPU(timer, EVALS);
    const uint64_t nEvals = setupCtx.starkInfo.evMap.size();
    if (air_instance_info->nEvalGroups != 0) {
        const uint64_t nBits = setupCtx.starkInfo.starkStruct.nBits;
        gl64_t *d_partials = (gl64_t *)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("evals_partials", false)];
        const dim3 threads(256);
        const size_t shmem = ((threads.x + 31) / 32) * EVALS_GROUP_OPENINGS * sizeof(Goldilocks3GPU::Element);
        computeEvalsShifted<<<dim3((unsigned)air_instance_info->nEvalGroups, EVALS_HELPER_CHUNKS), threads, shmem, stream>>>(
            1ULL << nBits, setupCtx.starkInfo.starkStruct.nBitsExt - nBits, air_instance_info->evalGroups,
            (gl64_t *)h_params.aux_trace, (gl64_t *)h_params.pCustomCommitsFixed,
            (gl64_t *)h_params.pConstPolsExtendedTreeAddress, (gl64_t *)d_LEv, d_partials);
        reduceEvalsShifted<<<(unsigned)((nEvals + 255) / 256), 256, 0, stream>>>(nEvals, EVALS_HELPER_CHUNKS, d_partials, (gl64_t *)h_params.evals);
    }
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

// fold with the per-thread ppar workspace in local memory instead of global
// scratch. The generic kernel keys each thread's RATIO*FIELD_EXTENSION slots
// contiguously in d_ppar, so every intt_tinny access is a fully scattered
// global round-trip. Local memory is hardware-interleaved per thread, so the same
// accesses coalesce; the arithmetic and its order are bit-identical.
// challengeSquarings: fold with challenge^(2^challengeSquarings) (a sub-step of a larger fold).
template<uint32_t RATIO>
__global__ void fold_reg(gl64_t *deepPol, gl64_t *d_challenge, Goldilocks::Element omega_inv,
                         uint64_t invShiftPow_, uint64_t invW_, uint64_t currentBits, uint32_t challengeSquarings)
{
    extern __shared__ gl64_t s_twiddles[];
    if (threadIdx.x == 0) {
        s_twiddles[0] = gl64_t(uint64_t(1));
        for (uint32_t i = 1; i < RATIO / 2; i++) {
            s_twiddles[i] = s_twiddles[i - 1] * gl64_t(omega_inv.fe);
        }
    }
    __syncthreads();

    constexpr uint32_t LOG_RATIO = (RATIO == 2) ? 1 : (RATIO == 4) ? 2 : (RATIO == 8) ? 3 : (RATIO == 16) ? 4 : 5;
    static_assert((1u << LOG_RATIO) == RATIO, "RATIO must be a power of two");
    uint64_t sizeFoldedPol = 1ull << currentBits;

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (int64_t)sizeFoldedPol)
        return;

    gl64_t invShift(invShiftPow_);
    gl64_t invW(invW_);
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

    gl64_t ppar[RATIO * FIELD_EXTENSION];
    for (uint32_t i = 0; i < RATIO; i++)
    {
        uint32_t ind = i * FIELD_EXTENSION;
        for (uint32_t k = 0; k < FIELD_EXTENSION; k++)
        {
            ppar[ind + k] = gl64_t(deepPol[(i * sizeFoldedPol + id) * FIELD_EXTENSION + k]);
        }
    }
    intt_tinny(ppar, RATIO, LOG_RATIO, s_twiddles, FIELD_EXTENSION);

    gl64_t r(1);
    for (uint32_t i = 0; i < RATIO; i++)
    {
        Goldilocks3GPU::Element *component = (Goldilocks3GPU::Element *)&ppar[i * FIELD_EXTENSION];
        Goldilocks3GPU::mul(*component, *component, r);
        r *= sinv;
    }
    Goldilocks3GPU::Element challenge;
    for (uint32_t k = 0; k < FIELD_EXTENSION; k++) challenge[k] = d_challenge[k];
    for (uint32_t k = 0; k < challengeSquarings; k++) Goldilocks3GPU::mul(challenge, challenge, challenge);
    for (uint32_t i = 0; i < FIELD_EXTENSION; i++)
    {
        deepPol[id * FIELD_EXTENSION + i] = ppar[(RATIO - 1) * FIELD_EXTENSION + i];
    }
    for (int i = RATIO - 2; i >= 0; i--)
    {
        Goldilocks3GPU::Element aux;
        Goldilocks3GPU::mul(aux, *((Goldilocks3GPU::Element *)&deepPol[id * FIELD_EXTENSION]), challenge);
        Goldilocks3GPU::add(*((Goldilocks3GPU::Element *)&deepPol[id * FIELD_EXTENSION]), aux, *((Goldilocks3GPU::Element *)&ppar[i * FIELD_EXTENSION]));
    }
}

void fold_inplace(uint64_t step, uint64_t deepPol_offset, Goldilocks::Element *d_challenge, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits, gl64_t *d_aux_trace, TimerGPU &timer, cudaStream_t stream)
{
    gl64_t *d_deepPol = (gl64_t *)(d_aux_trace + deepPol_offset);
    TimerStartCategoryGPU(timer, FRI);
    // A fold by R1*R2 is a fold by R1 with challenge a, then by R2 with a^R1, so any fold runs as
    // in-place steps of at most 4 bits on the register kernels, with no scratch.
    uint32_t squarings = 0;
    for (uint64_t from = prevBits; from > currentBits;) {
        const uint64_t bits = std::min<uint64_t>(4, from - currentBits);
        const uint64_t to = from - bits;
        const uint32_t ratio = 1u << bits;
        const uint64_t sizeFoldedPol = 1ull << to;
        const Goldilocks::Element omega_inv = omegas_inv_[bits];
        // invShift^(2^(nBitsExt-from)) and w(from)^-1, on the host once per step.
        Goldilocks::Element invShiftPow = Goldilocks::inv(Goldilocks::shift());
        for (uint32_t j = 0; j < nBitsExt - from; j++) Goldilocks::square(invShiftPow, invShiftPow);
        const Goldilocks::Element invW = Goldilocks::inv(Goldilocks::w(from));

        dim3 nThreads(256);
        dim3 nBlocks((sizeFoldedPol + nThreads.x - 1) / nThreads.x);
        const size_t sharedMem = (ratio >> 1) * sizeof(gl64_t);
        switch (ratio) {
        case 2:  fold_reg<2><<<nBlocks, nThreads, sharedMem, stream>>>(d_deepPol, (gl64_t *)d_challenge, omega_inv, invShiftPow.fe, invW.fe, to, squarings); break;
        case 4:  fold_reg<4><<<nBlocks, nThreads, sharedMem, stream>>>(d_deepPol, (gl64_t *)d_challenge, omega_inv, invShiftPow.fe, invW.fe, to, squarings); break;
        case 8:  fold_reg<8><<<nBlocks, nThreads, sharedMem, stream>>>(d_deepPol, (gl64_t *)d_challenge, omega_inv, invShiftPow.fe, invW.fe, to, squarings); break;
        default: fold_reg<16><<<nBlocks, nThreads, sharedMem, stream>>>(d_deepPol, (gl64_t *)d_challenge, omega_inv, invShiftPow.fe, invW.fe, to, squarings); break;
        }
        squarings += bits;
        from = to;
    }
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
    // The transpose lays the polynomial out for the tree, so it belongs to the same category.
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    transposeFRI<<<nBlocks, nThreads, 0, stream>>>((gl64_t *)treeFRI->source, (gl64_t *)pol, pol2N, width);

    buildMerkleTreeGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)treeFRI->nodes, (uint64_t *)treeFRI->source, treeFRI->width, treeFRI->height, Layout::RowMajor, stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);

    uint64_t tree_size = treeFRI->numNodes;
    if(d_transcript != nullptr) {
        TimerStartCategoryGPU(timer, TRANSCRIPT);
        d_transcript->put(&treeFRI->nodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
        TimerStopCategoryGPU(timer, TRANSCRIPT);
    }
}

__global__ void getTreeTracePols(gl64_t *d_treeTrace, uint64_t traceWidth, uint64_t *d_queriesL0, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth)
{

    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < traceWidth && idx_y < nQueries)
    {
        uint64_t row = d_queriesL0[idx_y];
        uint64_t idx_trace = row * traceWidth + idx_x;
        uint64_t idx_buffer = idx_y * bufferWidth + idx_x;
        d_buffer[idx_buffer] = d_treeTrace[idx_trace];
    }
}

__global__ void getTreeTracePolsBlocks(gl64_t *d_treeTrace, uint64_t nCols, uint64_t nRows, uint64_t *d_queriesL0, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, Layout layout)
{

    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < nCols && idx_y < nQueries)
    {
        uint64_t row = d_queriesL0[idx_y];
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

// dropLeaves: d_nodes starts at level 1; leafSiblingsNoLeavesGPU already wrote the level-0 siblings.
__global__ void genMerkleProof(gl64_t *d_nodes, uint64_t nLeaves, uint64_t *d_queriesL0, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, uint64_t maxTreeWidth, uint64_t nFieldElements, uint64_t arity, uint64_t lastLevel, bool dropLeaves)
{

    uint64_t idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_query < nQueries)
    {
        uint64_t row = d_queriesL0[idx_query];
        uint64_t idx_buffer = idx_query * bufferWidth + maxTreeWidth;
        if (dropLeaves) {
            genMerkleProof_(d_nodes, &d_buffer[idx_buffer + (arity - 1) * nFieldElements], row / arity, 0, (nLeaves + arity - 1) / arity, nFieldElements, arity, lastLevel);
        } else {
            genMerkleProof_(d_nodes, &d_buffer[idx_buffer], row, 0, nLeaves, nFieldElements, arity, lastLevel);
        }
    }
}

void proveQueries_inplace(SetupCtx& setupCtx, gl64_t *d_queries_buff, uint64_t *d_queriesL0, uint64_t nQueries, MerkleTreeGL **trees, uint64_t nTrees, gl64_t *d_aux_trace, gl64_t* d_constTree, uint32_t nStages, cudaStream_t stream)
{   
    uint64_t maxBuffSize = setupCtx.starkInfo.maxProofBuffSize;
    uint64_t maxTreeWidth = setupCtx.starkInfo.maxTreeWidth;

    for (uint k = 0; k < nTrees; k++)
    {
        if (trees[k] == nullptr) continue;  // the empty const slot of a zero-fixed air
        dim3 nThreads(32, 32);
        dim3 nBlocks((trees[k]->getMerkleTreeWidth() + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
        if (k < nStages + 1)
        {
            std::string section = "cm" + to_string(k+1);
            uint64_t offset = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
            // cm section: same storage layout the commit used (resolveLayout on the small domain).
            Layout layout = resolveLayout(setupCtx.starkInfo.starkStruct.nBits, trees[k]->getMerkleTreeWidth());
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_aux_trace + offset, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_queriesL0, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, layout);
        }
        else if (k == nStages + 1)
        {
            // Const tree leaves were written fixedLayout() by extendAndMerkelizeFixed.
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_constTree, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_queriesL0, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, fixedLayout());
        } else{
            uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[0].name + "0"];
            uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            // Custom commit tree leaves were written fixedLayout() by write_custom_commit_gpu.
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_aux_trace + offset + N*nCols, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_queriesL0, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, fixedLayout());
        }
    }
    CHECKCUDAERR(cudaGetLastError());


    const bool dropLeaves = setupCtx.starkInfo.dropLeafLevel;
    auto leafSiblings = [&](uint64_t k, const gl64_t *trace, Layout layout) {
        if (!dropLeaves) return;
        uint64_t *scratch = (uint64_t *)(d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("query_rows_scratch", false)]);
        leafSiblingsNoLeavesGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (const uint64_t *)trace, trees[k]->getMerkleTreeWidth(),
                                trees[k]->getMerkleTreeHeight(), layout, d_queriesL0, nQueries, scratch,
                                (uint64_t *)(d_queries_buff + k * nQueries * maxBuffSize), maxBuffSize, maxTreeWidth, stream);
    };

    // Node arrays come from the layout, never from tree-object state: the same source setProof
    // reads roots/ll from. A mismatch means a tree object was consumed with stale pointers.
    for (uint k = 0; k < nStages + 1; k++)
    {
        gl64_t *nodesK = d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt" + std::to_string(k + 1), true)];
        if ((gl64_t *)trees[k]->get_nodes_ptr() != nodesK) {
            zklog.error("proveQueries: tree " + std::to_string(k) + " nodes pointer disagrees with the layout (stale tree object)");
            exitProcess();
        }
        leafSiblings(k, d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + std::to_string(k + 1), true)],
                     resolveLayout(setupCtx.starkInfo.starkStruct.nBits, trees[k]->getMerkleTreeWidth()));
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads, 0, stream>>>(nodesK, trees[k]->getMerkleTreeHeight(), d_queriesL0, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification, dropLeaves);
        CHECKCUDAERR(cudaGetLastError());
    }
    CHECKCUDAERR(cudaGetLastError());

    if (trees[nStages + 1] != nullptr) {
        leafSiblings(nStages + 1, d_constTree, fixedLayout());
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[nStages + 1]->get_nodes_ptr(), trees[nStages + 1]->getMerkleTreeHeight(), d_queriesL0, nQueries, d_queries_buff + (nStages + 1) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification, dropLeaves);
        CHECKCUDAERR(cudaGetLastError());
    }

    if(nTrees > nStages + 2){
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[nStages + 2]->get_nodes_ptr(), trees[nStages + 2]->getMerkleTreeHeight(), d_queriesL0, nQueries, d_queries_buff + (nStages + 2) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification);
        CHECKCUDAERR(cudaGetLastError());
    }
}

__global__ void moduleQueries(uint64_t* d_queriesL0, uint64_t nQueries, uint64_t currentBits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nQueries) {
        d_queriesL0[idx] %= (1ULL << currentBits);
    }
}

void proveLeafQueries_gpu(gl64_t *d_buff, const gl64_t *treeSrc, const gl64_t *treeNodes, uint64_t width, uint64_t height,
                          const uint64_t *d_idx, uint64_t nQueries, uint64_t arity, uint64_t lastLevel, cudaStream_t stream) {
    // Same expression as writeProof's sibling count.
    uint64_t buffSize = width + merkleProofLevels(log2(height), arity, lastLevel, false) * (arity - 1) * HASH_SIZE;
    dim3 nThreads(32, 32);
    dim3 nBlocks((width + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
    getTreeTracePols<<<nBlocks, nThreads, 0, stream>>>((gl64_t *)treeSrc, width, (uint64_t *)d_idx, nQueries, d_buff, buffSize);
    CHECKCUDAERR(cudaGetLastError());
    dim3 nthreads(64);
    dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
    genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)treeNodes, height, (uint64_t *)d_idx, nQueries, d_buff, buffSize, width, HASH_SIZE, arity, lastLevel);
    CHECKCUDAERR(cudaGetLastError());
}

void proveFRIQueries_inplace(SetupCtx& setupCtx, gl64_t *d_queries_buff, uint64_t step, uint64_t currentBits, uint64_t *d_queriesL0, uint64_t nQueries, MerkleTreeGL *treeFRI, cudaStream_t stream) {
    dim3 nthreads_(64);
    dim3 nblocks_((nQueries + nthreads_.x - 1) / nthreads_.x);
    moduleQueries<<<nblocks_, nthreads_, 0, stream>>>(d_queriesL0, nQueries, currentBits);
    CHECKCUDAERR(cudaGetLastError());
    proveLeafQueries_gpu(d_queries_buff, (gl64_t *)treeFRI->source, (gl64_t *)treeFRI->nodes, treeFRI->getMerkleTreeWidth(), treeFRI->getMerkleTreeHeight(), d_queriesL0, nQueries,
                         setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification, stream);
}

void calculateImPolsExpressions(SetupCtx& setupCtx, ExpressionsGPU* expressionsCtx, StepsParams &h_params, StepsParams *d_params, int64_t step, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream){

    uint64_t domainSize = (1 << setupCtx.starkInfo.starkStruct.nBits);
    std::vector<Dest> dests;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {
            Goldilocks::Element* pAddress = step == 1 ? h_params.trace : h_params.aux_trace;
            Dest destStruct(NULL, domainSize, setupCtx.starkInfo.cmPolsMap[i].stagePos, setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)], false);
            destStruct.addParams(setupCtx.starkInfo.cmPolsMap[i].expId, setupCtx.starkInfo.cmPolsMap[i].dim, false);
            const std::string section = "cm" + to_string(step);
            uint64_t offset_aux_trace = setupCtx.starkInfo.mapOffsets[std::make_pair(section, false)];
            destStruct.dest_gpu = (Goldilocks::Element *)(pAddress + offset_aux_trace);
            if (step == 1) destStruct.setOutsideAux();  // the trace buffer, not aux_trace
            else destStruct.setAuxRange(offset_aux_trace, domainSize * setupCtx.starkInfo.mapSectionsN[section]);
            countId++;
            expressionsCtx->calculateExpressions_gpu(d_params, destStruct, domainSize, false, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream);
        }
    }
        
}

uint32_t qStepLgFor(const StarkInfo& starkInfo) {
    uint32_t lgQDeg = 0;
    while ((1ull << lgQDeg) < starkInfo.qDeg) lgQDeg++;
    const uint64_t lgBlowup = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
    return lgBlowup > lgQDeg ? (uint32_t)(lgBlowup - lgQDeg) : 0;
}

uint32_t calculateExpressionQ(SetupCtx& setupCtx, ExpressionsGPU* expressionsCtx, StepsParams *d_params, Goldilocks::Element* dest_gpu, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU& timer, cudaStream_t stream, uint32_t qStepLg){
    
    uint64_t domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    bool domainExtended = true;
    setupCtx.expressionsBin.expressionsInfo[setupCtx.starkInfo.cExpId].destDim = 3;
    Dest destStruct(NULL, domainSize, 0, 3, false, setupCtx.starkInfo.cExpId);
    destStruct.addParams(setupCtx.starkInfo.cExpId, setupCtx.expressionsBin.expressionsInfo[setupCtx.starkInfo.cExpId].destDim, false);
    destStruct.dest_gpu = dest_gpu;
    countId++;
    return expressionsCtx->calculateExpressionsQ_gpu(d_params, destStruct, domainSize, domainExtended, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, qStepLg);
}

void setProof(SetupCtx &setupCtx, Goldilocks::Element *h_aux_trace, Goldilocks::Element *h_const_tree, Goldilocks::Element *proof_buffer_pinned, cudaStream_t stream) {
    uint64_t initialOffset = 0;
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    // cm* and const trees may lack their leaf level; custom commit trees never do.
    uint64_t numNodes = setupCtx.starkInfo.getNumNodesMTCommit(NExtended);
    uint64_t arity = setupCtx.starkInfo.starkStruct.merkleTreeArity;
    uint32_t lastLevelVerification = setupCtx.starkInfo.starkStruct.lastLevelVerification;
    uint64_t numNodesLevel = std::pow(arity, lastLevelVerification);
    uint64_t commitLastLevelCount;
    const uint64_t commitLastLevelOffset = setupCtx.starkInfo.getLastLevelOffset(NExtended, commitLastLevelCount);
    for(uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; ++i) {
        uint64_t stage = i + 1;
        Goldilocks::Element *nodes = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt" + to_string(stage), true)];
        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + numNodes - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        initialOffset += HASH_SIZE;

        if (lastLevelVerification > 0) {
            uint64_t n = commitLastLevelCount;
            uint64_t offset = commitLastLevelOffset;

            CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], nodes + offset, n * HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

            memset(&proof_buffer_pinned[initialOffset + n * HASH_SIZE], 0, (numNodesLevel - n) * HASH_SIZE * sizeof(uint64_t));

            initialOffset += numNodesLevel * HASH_SIZE;
        }
    }

    if (lastLevelVerification > 0 && setupCtx.starkInfo.hasConstants()) {
        Goldilocks::Element *nodes = h_const_tree + NExtended * setupCtx.starkInfo.nConstants;
        uint64_t n = commitLastLevelCount;
        uint64_t offset = commitLastLevelOffset;

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

    const bool stirLdt = setupCtx.starkInfo.starkStruct.lowDegreeTest == LowDegreeTestKind::STIR;
    StirLayout sl;
    if (stirLdt) sl = setupCtx.starkInfo.stirLayout();
    const uint64_t nLdtTrees = setupCtx.starkInfo.numLowDegreeTestTrees();
    for (uint64_t step = 0; step < nLdtTrees; step++)
    {
        uint64_t height, numNodes;
        Goldilocks::Element *nodes;
        if (stirLdt) {
            const StirStruct &sx = setupCtx.starkInfo.starkStruct.stir;
            height = (1ull << sx.logDomainSizes[step]) >> sx.foldingFactors[step];
            numNodes = sl.treeNumNodes[step];
            nodes = h_aux_trace + sl.treeNodes[step];
        } else {
            height = 1 << setupCtx.starkInfo.starkStruct.logDomainSizes[step + 1];
            numNodes = setupCtx.starkInfo.getNumNodesMT(height);
            nodes = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)];
        }
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
    uint64_t queriesProofSize = (nTrees + nLdtTrees) * setupCtx.starkInfo.maxProofBuffSize * setupCtx.starkInfo.starkStruct.nQueries;
    uint64_t offsetProofQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("proof_queries", false)];

    Goldilocks::Element *d_queries_buff = h_aux_trace + offsetProofQueries;
    Goldilocks::Element *d_evals = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("evals", false)];
    Goldilocks::Element *d_airgroupValues = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)];
    Goldilocks::Element *d_airValues = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("airvalues", false)];

    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_queries_buff, queriesProofSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += queriesProofSize;
    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION;
    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_airgroupValues, setupCtx.starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += setupCtx.starkInfo.airgroupValuesSize;
    CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_airValues, setupCtx.starkInfo.airValuesSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
    initialOffset += setupCtx.starkInfo.airValuesSize;
    if (stirLdt) {
        // betas | finalPol | nonces | ans | shake | oodFlag, contiguous on the device.
        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], h_aux_trace + sl.proof, sl.proofSize * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
        initialOffset += sl.proofSize;
    } else {
        uint64_t finalPolDegree = 1 << setupCtx.starkInfo.starkStruct.logDomainSizes[setupCtx.starkInfo.starkStruct.logDomainSizes.size() - 1];
        Goldilocks::Element *d_fri_pol = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_fri_pol, finalPolDegree * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
        initialOffset += finalPolDegree * FIELD_EXTENSION;

        Goldilocks::Element *d_nonce = h_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("nonce", false)];
        CHECKCUDAERR(cudaMemcpyAsync(&proof_buffer_pinned[initialOffset], d_nonce, sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost, stream));
        initialOffset += 1;
    }
}

// Serialize the pinned proof buffer directly into proof_buffer, in the same layout
// FRIProof::proof2pointer produces, but without building the intermediate FRIProof object.
void writeProof(SetupCtx &setupCtx, Goldilocks::Element *proof_buffer_pinned, uint64_t *proof_buffer, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, std::string proofFile) {
    StarkInfo &starkInfo = setupCtx.starkInfo;

    uint64_t nStages = starkInfo.nStages + 1;
    const bool stirLdt = starkInfo.starkStruct.lowDegreeTest == LowDegreeTestKind::STIR;
    uint64_t nFriSteps = starkInfo.numLowDegreeTestTrees();
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
    const bool hasConstants = starkInfo.hasConstants();
    Goldilocks::Element *constLastLevel = lastLevelVerification > 0 ? rootsBase + nStages * treeHeaderStride : nullptr;
    Goldilocks::Element *customLastBase = rootsBase + nStages * treeHeaderStride + (lastLevelVerification > 0 && hasConstants ? lastLevelWords : 0);

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
    // FRI only: logDomainSizes is empty under STIR.
    uint64_t finalPolDegree = stirLdt ? 0 : 1 << starkInfo.starkStruct.logDomainSizes.back();
    Goldilocks::Element *finalPol = airValues + starkInfo.airValuesSize;
    Goldilocks::Element *nonce = finalPol + finalPolDegree * FIELD_EXTENSION;

    uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.nBitsExt, arity, lastLevelVerification, false);
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
    if (hasConstants) {
        uint64_t constantsTree = starkInfo.nStages + 1;
        for (uint64_t q = 0; q < nQueries; ++q) emit(queryBase(constantsTree, q), starkInfo.nConstants);
        for (uint64_t q = 0; q < nQueries; ++q) emit(queryBase(constantsTree, q) + starkInfo.maxTreeWidth, siblingWords);
        if (lastLevelVerification != 0) emit(constLastLevel, lastLevelWords);
    }

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

    if (stirLdt) {
        // proof2pointer's STIR section; the device proof block sits where FRI's finalPol would.
        const StirStruct &sx = starkInfo.starkStruct.stir;
        const uint64_t M = sx.numIterations();
        StirLayout sl = starkInfo.stirLayout();
        const Goldilocks::Element *sp = airValues + starkInfo.airValuesSize;
        if (sp[sl.oodFlag].fe != 0) {
            throw std::runtime_error("STIR: an out-of-domain sample fell in L_i (probability |L_i|/|F^3| < 2^-160); the verifier would reject this proof");
        }
        for (uint64_t i = 0; i < M; ++i) emit(friHeadersBase + i * treeHeaderStride, HASH_SIZE);
        for (uint64_t i = 0; i < M; ++i) {
            uint64_t width = (1ULL << sx.foldingFactors[i]) * FIELD_EXTENSION;
            uint64_t sib = merkleProofLevels(sx.logDomainSizes[i] - sx.foldingFactors[i], arity, lastLevelVerification, false) * (arity - 1) * HASH_SIZE;
            const Goldilocks::Element *qb = queries + (nTrees + i) * nQueries * starkInfo.maxProofBuffSize;
            for (uint64_t q = 0; q < sx.numQueries[i]; ++q) emit(qb + q * (width + sib), width);
            for (uint64_t q = 0; q < sx.numQueries[i]; ++q) emit(qb + q * (width + sib) + width, sib);
            if (lastLevelVerification != 0) emit(friHeadersBase + i * treeHeaderStride + HASH_SIZE, lastLevelWords);
        }
        emit(sp + sl.betas, (M - 1) * sx.numOodSamples * FIELD_EXTENSION);
        emit(sp + sl.finalPol, (1ULL << sx.logDegrees[M]) * FIELD_EXTENSION);
        for (uint64_t i = 0; i < M; ++i) *out++ = sp[sl.nonces + i].fe;   // raw, as proof2pointer
        for (uint64_t i = 1; i < M; ++i) emit(sp + sl.ans[i - 1], (sx.numOodSamples + sx.numQueries[i - 1]) * FIELD_EXTENSION);
        for (uint64_t i = 1; i < M; ++i) emit(sp + sl.shake[i - 1], (sx.numOodSamples + sx.numQueries[i - 1]) * FIELD_EXTENSION);
    } else {
        // FRI step roots, then per-step openings + siblings (+ last level).
        for (uint64_t step = 0; step < nFriSteps; ++step) emit(friHeadersBase + step * treeHeaderStride, HASH_SIZE);

        for (uint64_t step = 1; step < starkInfo.starkStruct.logDomainSizes.size(); ++step) {
            uint64_t stepIndex = step - 1;
            uint64_t width = (1ULL << (starkInfo.starkStruct.logDomainSizes[step - 1] - starkInfo.starkStruct.logDomainSizes[step])) * FIELD_EXTENSION;
            uint64_t stepSiblings = merkleProofLevels(starkInfo.starkStruct.logDomainSizes[step], arity, lastLevelVerification, false);
            uint64_t stepSiblingWords = stepSiblings * (arity - 1) * HASH_SIZE;
            uint64_t buffSize = width + stepSiblingWords;
            Goldilocks::Element *queriesFRI = queries + (nTrees + stepIndex) * nQueries * starkInfo.maxProofBuffSize;

            for (uint64_t q = 0; q < nQueries; ++q) emit(queriesFRI + q * buffSize, width);
            for (uint64_t q = 0; q < nQueries; ++q) emit(queriesFRI + q * buffSize + width, stepSiblingWords);
            if (lastLevelVerification != 0) emit(friHeadersBase + stepIndex * treeHeaderStride + HASH_SIZE, lastLevelWords);
        }

        emit(finalPol, finalPolDegree * FIELD_EXTENSION);
        emit(nonce, 1);
    }

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

void calculateFRIExpression(SetupCtx& setupCtx, StepsParams &h_params, AirInstanceInfo *air_instance_info, Goldilocks::Element *d_xiChallenge, cudaStream_t stream) {
    // The regions and the block shape are checked at setup (AirInstanceInfo): this runs inside a cudagraph capture region.
    StarkInfo &si = setupCtx.starkInfo;
    const uint64_t domainSize = 1ULL << si.starkStruct.nBitsExt, nOpenings = si.openingPoints.size();
    gl64_t *d_fri = (gl64_t*)h_params.aux_trace + si.mapOffsets[std::make_pair("f", true)];
    gl64_t *d_b = (gl64_t*)h_params.aux_trace + si.mapOffsets[std::make_pair("fri_constants", false)];
    gl64_t *d_a = d_b + si.evMap.size() * FIELD_EXTENSION;
    gl64_t *d_k = d_a + nOpenings * FIELD_EXTENSION;
    gl64_t *challenges = (gl64_t*)h_params.challenges;

    computeFRIConstants(nOpenings, air_instance_info->nFriPols, air_instance_info->opening_points,
                        Goldilocks::w(si.starkStruct.nBits).fe, air_instance_info->friTermStart, air_instance_info->friTerms,
                        (gl64_t*)h_params.evals, challenges + 4 * FIELD_EXTENSION, challenges + 5 * FIELD_EXTENSION, d_b,
                        d_a, d_k, stream);
    // 1/(x - xi) once over the domain: dead zi/helper space behind q, overwritten by the first fold (fri_1).
    gl64_t *d_den = (gl64_t*)h_params.aux_trace + si.mapOffsets[std::make_pair("fri_denominators", true)];
    const uint32_t denThreads = friDenominatorThreads(domainSize);
    const Goldilocks::Element wExt = Goldilocks::w(si.starkStruct.nBitsExt), wDen = Goldilocks::pow(wExt, denThreads);
    computeFRIDenominators<<<friDenominatorBlocks(domainSize, denThreads), denThreads, 0, stream>>>(
        domainSize, Goldilocks::shift().fe, wExt.fe, wDen.fe, Goldilocks::inv(wDen).fe, (gl64_t*)d_xiChallenge, d_den);
    CHECKCUDAERR(cudaGetLastError());
    const uint32_t nThreads = 256;
    // ceil: an extended domain under 256 rows would otherwise launch 0 blocks
    computeFRIExpression<<<(domainSize + nThreads - 1) / nThreads, nThreads, 0, stream>>>(
        domainSize, nOpenings, si.starkStruct.nBitsExt - si.starkStruct.nBits, air_instance_info->opening_points, d_den,
        air_instance_info->nFriGroups, air_instance_info->friColStart, air_instance_info->friCols,
        air_instance_info->friOpStart, air_instance_info->friOpGroups, d_b, d_a, d_k, (gl64_t*)h_params.aux_trace,
        (gl64_t *)h_params.pCustomCommitsFixed, (gl64_t *)h_params.pConstPolsExtendedTreeAddress, d_fri);
    CHECKCUDAERR(cudaGetLastError());
}
