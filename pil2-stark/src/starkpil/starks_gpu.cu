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
#include "fri_expression.cuh"
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

static void buildCommitTreeGPU(bool dropLeaves, uint32_t arity, uint64_t *nodes, uint64_t *src, uint64_t nCols,
                               uint64_t nRows, Layout layout, cudaStream_t stream)
{
    if (dropLeaves) buildMerkleTreeNoLeavesGPU(arity, nodes, src, nCols, nRows, layout, stream);
    else buildMerkleTreeGPU(arity, nodes, src, nCols, nRows, layout, stream);
}

void commitStage_inplace(uint64_t step, SetupCtx &setupCtx, MerkleTreeGL **treesGL, gl64_t *d_trace, gl64_t *d_aux_trace, TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
{
    if (step <= setupCtx.starkInfo.nStages)
    {
    
        extendAndMerkelize_inplace(step, setupCtx, treesGL, d_trace, d_aux_trace, d_transcript, timer, stream);
    }
    else
    {
        computeQ_MerkleTree_inplace(step, setupCtx, treesGL, d_aux_trace, d_transcript, timer, stream);
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
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    extendAndMerkelizeSection(setupCtx.starkInfo.nConstants, setupCtx.starkInfo.starkStruct.nBits,
                              setupCtx.starkInfo.starkStruct.nBitsExt,
                              setupCtx.starkInfo.starkStruct.merkleTreeArity,
                              setupCtx.starkInfo.getNumNodesMTCommit(NExtended),
                              d_fixedPols, d_fixedPolsExtended, preserve_src, timer, stream,
                              setupCtx.starkInfo.dropLeafLevel);
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

    Goldilocks::Element *pNodes = d_aux_traceGL + setupCtx.starkInfo.mapOffsets[make_pair("mt" + to_string(step), true)];

    if (nCols > 0)
    {
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft", false)];
        NTTGoldilocksGPU nttExtended;

        nttExtended.computeQ(offset_cmQ, offset_q, qDeg, qDim, shiftIn, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, d_aux_trace, offset_helper, timer, stream);
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

// Opening 0's shifted point z = xi / s and its factor (1 - z^N) / N, for fillLEvDirectBatched.
__global__ void evalXiShifted(gl64_t *d_shiftedValues, gl64_t *d_xiChallenge, uint64_t invShift_, uint64_t nBits, uint64_t domainInv_)
{
    Goldilocks3GPU::Element xi, xiN, factor, one;
    gl64_t invShift(invShift_);
    Goldilocks3GPU::mul(xi, *((Goldilocks3GPU::Element *)d_xiChallenge), invShift);
    Goldilocks3GPU::copy(xiN, xi);
    for (uint64_t bit = 0; bit < nBits; ++bit)
        Goldilocks3GPU::mul(xiN, xiN, xiN);
    Goldilocks3GPU::one(one);
    Goldilocks3GPU::sub(factor, one, xiN);
    gl64_t domainInv(domainInv_);
    Goldilocks3GPU::mul(factor, factor, domainInv);
    for (uint32_t k = 0; k < FIELD_EXTENSION; ++k) {
        d_shiftedValues[k] = xi[k];
        d_shiftedValues[FIELD_EXTENSION + k] = factor[k];
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

// Direct (barycentric) Lagrange-kernel fill: LEv[row] = factor / (1 - z*w^{-row}) with
// factor = (1 - z^N)/N precomputed (see evalXiShifted), which equals
// L_row(z) by the closed form (z^N - 1) w^row / (N (z - w^row)). Each thread owns BATCH
// consecutive rows and inverts their denominators with ONE cubic inversion (Montgomery
// prefix trick), so the inversion cost is amortized BATCH ways.
// factor == 0 means z^N = 1, i.e. z landed exactly on a domain node (negligible-probability
// challenge, but exact): LEv is then the indicator vector of the matching row.
template<uint32_t BATCH>
__global__ void fillLEvDirectBatched(gl64_t *d_LEv, uint64_t N, gl64_t *d_shiftedValues, uint64_t rootInv_)
{
    const uint64_t batch = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t row0 = batch * BATCH;
    if (row0 >= N) return;

    const uint32_t count = (uint32_t)min((uint64_t)BATCH, N - row0);
    Goldilocks3GPU::Element xi, factor;
    for (uint32_t k = 0; k < FIELD_EXTENSION; ++k) {
        xi[k] = d_shiftedValues[k];
        factor[k] = d_shiftedValues[FIELD_EXTENSION + k];
    }
    const bool factorZero = factor[0].is_zero() && factor[1].is_zero() && factor[2].is_zero();
    gl64_t rootInv(rootInv_);
    gl64_t root = gl64Pow(rootInv, row0);
    gl64_t roots[BATCH];
    Goldilocks3GPU::Element prefix[BATCH], scaled, den, one;
    Goldilocks3GPU::one(one);
    #pragma unroll
    for (uint32_t b = 0; b < BATCH; ++b) {
        if (b >= count) break;
        roots[b] = root;
        Goldilocks3GPU::mul(scaled, xi, root);
        Goldilocks3GPU::sub(den, one, scaled);
        if (b == 0) Goldilocks3GPU::copy(prefix[0], den);
        else Goldilocks3GPU::mul(prefix[b], prefix[b - 1], den);
        root *= rootInv;
    }
    if (factorZero) {
        Goldilocks3GPU::Element out;
        for (uint32_t b = 0; b < count; ++b) {
            Goldilocks3GPU::mul(scaled, xi, roots[b]);
            Goldilocks3GPU::sub(den, one, scaled);
            const bool match = den[0].is_zero() && den[1].is_zero() && den[2].is_zero();
            out[0] = match ? gl64_t(uint64_t(1)) : gl64_t(uint64_t(0));
            out[1] = gl64_t(uint64_t(0));
            out[2] = gl64_t(uint64_t(0));
            storeLEv(d_LEv, row0 + b, N, out);
        }
        return;
    }
    // Montgomery unwind: inv = (prod of remaining dens)^{-1}; each step peels one den
    // (recomputed rather than stored -- one cubic mul against BATCH extra registers).
    Goldilocks3GPU::Element inv, weight, out;
    Goldilocks3GPU::inv(inv, prefix[count - 1]);
    for (uint32_t b = count - 1; b > 0; --b) {
        Goldilocks3GPU::mul(weight, inv, prefix[b - 1]);
        Goldilocks3GPU::mul(out, factor, weight);
        storeLEv(d_LEv, row0 + b, N, out);
        Goldilocks3GPU::mul(scaled, xi, roots[b]);
        Goldilocks3GPU::sub(den, one, scaled);
        Goldilocks3GPU::mul(inv, inv, den);
    }
    Goldilocks3GPU::mul(out, factor, inv);
    storeLEv(d_LEv, row0, N, out);
}

// Opening 0's Lagrange vector, which serves every opening (evmap_inplace). d_shiftedValues takes
// 2 * FIELD_EXTENSION elements.
void computeLEv_inplace(Goldilocks::Element *d_xiChallenge, uint64_t nBits, gl64_t *d_shiftedValues, gl64_t *d_LEv, TimerGPU &timer, cudaStream_t stream)
{
    TimerStartCategoryGPU(timer, LEV);
    uint64_t N = 1 << nBits;
    Goldilocks::Element invShift = Goldilocks::inv(Goldilocks::shift());
    Goldilocks::Element domainInv = Goldilocks::inv(Goldilocks::fromU64(N));
    evalXiShifted<<<1, 1, 0, stream>>>(d_shiftedValues, (gl64_t*)d_xiChallenge, invShift.fe, nBits, domainInv.fe);

    // BATCH = 4: measured optimum (BATCH = 8 amortizes the per-thread inversion further
    // but the extra cubic registers cost more than it saves: 31.4 vs 22.0 ms per phase).
    constexpr uint32_t directBatch = 4;
    dim3 nThreads(256);
    dim3 nBlocks((N + nThreads.x * directBatch - 1) / (nThreads.x * directBatch));
    Goldilocks::Element rootInv = Goldilocks::inv(Goldilocks::w(nBits));
    fillLEvDirectBatched<directBatch><<<nBlocks, nThreads, 0, stream>>>(d_LEv, N, d_shiftedValues, rootInv.fe);
    TimerStopCategoryGPU(timer, LEV);
    CHECKCUDAERR(cudaGetLastError());
}

// Opening evaluations p(xi w^o). Every opening point is z = xi / s shifted by a power of the trace
// root w, so L_j(z w^o) = L_{j-o}(z) and p(xi w^o) = SUM_i L_i(z) * p[(i + o) mod N]: opening 0's
// Lagrange vector serves every opening, and a group reads its column once for all its openings.
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
            const uint64_t row = ((i + shift[o]) & (N - 1)) << extendBits;
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
__global__ void fold_reg(gl64_t *friPol, gl64_t *d_challenge, Goldilocks::Element omega_inv,
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
            ppar[ind + k] = gl64_t(friPol[(i * sizeFoldedPol + id) * FIELD_EXTENSION + k]);
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
        friPol[id * FIELD_EXTENSION + i] = ppar[(RATIO - 1) * FIELD_EXTENSION + i];
    }
    for (int i = RATIO - 2; i >= 0; i--)
    {
        Goldilocks3GPU::Element aux;
        Goldilocks3GPU::mul(aux, *((Goldilocks3GPU::Element *)&friPol[id * FIELD_EXTENSION]), challenge);
        Goldilocks3GPU::add(*((Goldilocks3GPU::Element *)&friPol[id * FIELD_EXTENSION]), aux, *((Goldilocks3GPU::Element *)&ppar[i * FIELD_EXTENSION]));
    }
}

void fold_inplace(uint64_t step, uint64_t friPol_offset, Goldilocks::Element *d_challenge, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits, gl64_t *d_aux_trace, TimerGPU &timer, cudaStream_t stream)
{
    gl64_t *d_friPol = (gl64_t *)(d_aux_trace + friPol_offset);
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
        case 2:  fold_reg<2><<<nBlocks, nThreads, sharedMem, stream>>>(d_friPol, (gl64_t *)d_challenge, omega_inv, invShiftPow.fe, invW.fe, to, squarings); break;
        case 4:  fold_reg<4><<<nBlocks, nThreads, sharedMem, stream>>>(d_friPol, (gl64_t *)d_challenge, omega_inv, invShiftPow.fe, invW.fe, to, squarings); break;
        case 8:  fold_reg<8><<<nBlocks, nThreads, sharedMem, stream>>>(d_friPol, (gl64_t *)d_challenge, omega_inv, invShiftPow.fe, invW.fe, to, squarings); break;
        default: fold_reg<16><<<nBlocks, nThreads, sharedMem, stream>>>(d_friPol, (gl64_t *)d_challenge, omega_inv, invShiftPow.fe, invW.fe, to, squarings); break;
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

// dropLeaves: d_nodes starts at level 1; leafSiblingsNoLeavesGPU already wrote the level-0 siblings.
__global__ void genMerkleProof(gl64_t *d_nodes, uint64_t nLeaves, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, uint64_t maxTreeWidth, uint64_t nFieldElements, uint64_t arity, uint64_t lastLevel, bool dropLeaves)
{

    uint64_t idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_query < nQueries)
    {
        uint64_t row = d_friQueries[idx_query];
        uint64_t idx_buffer = idx_query * bufferWidth + maxTreeWidth;
        if (dropLeaves) {
            genMerkleProof_(d_nodes, &d_buffer[idx_buffer + (arity - 1) * nFieldElements], row / arity, 0, (nLeaves + arity - 1) / arity, nFieldElements, arity, lastLevel);
        } else {
            genMerkleProof_(d_nodes, &d_buffer[idx_buffer], row, 0, nLeaves, nFieldElements, arity, lastLevel);
        }
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
            // Const tree leaves were written fixedLayout() by extendAndMerkelizeFixed.
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_constTree, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, fixedLayout());
        } else{
            uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[0].name + "0"];
            uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
            // Custom commit tree leaves were written fixedLayout() by write_custom_commit_gpu.
            getTreeTracePolsBlocks<<<nBlocks, nThreads, 0, stream>>>(d_aux_trace + offset + N*nCols, trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, fixedLayout());
        }
    }
    CHECKCUDAERR(cudaGetLastError());


    const bool dropLeaves = setupCtx.starkInfo.dropLeafLevel;
    auto leafSiblings = [&](uint64_t k, const gl64_t *trace, Layout layout) {
        if (!dropLeaves) return;
        uint64_t *scratch = (uint64_t *)(d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("query_rows_scratch", false)]);
        leafSiblingsNoLeavesGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (const uint64_t *)trace, trees[k]->getMerkleTreeWidth(),
                                trees[k]->getMerkleTreeHeight(), layout, d_friQueries, nQueries, scratch,
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
        genMerkleProof<<<nblocks, nthreads, 0, stream>>>(nodesK, trees[k]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + k * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification, dropLeaves);
        CHECKCUDAERR(cudaGetLastError());
    }
    CHECKCUDAERR(cudaGetLastError());

    leafSiblings(nStages + 1, d_constTree, fixedLayout());
    dim3 nthreads(64);
    dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
    genMerkleProof<<<nblocks, nthreads, 0, stream>>>((gl64_t *)trees[nStages + 1]->get_nodes_ptr(), trees[nStages + 1]->getMerkleTreeHeight(), d_friQueries, nQueries, d_queries_buff + (nStages + 1) * nQueries * maxBuffSize, maxBuffSize, maxTreeWidth, HASH_SIZE, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.lastLevelVerification, dropLeaves);
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

    if (lastLevelVerification > 0) {
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

    uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.steps[0].nBits, arity, lastLevelVerification, false);
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
        uint64_t stepSiblings = merkleProofLevels(starkInfo.starkStruct.steps[step].nBits, arity, lastLevelVerification, false);
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

void calculateFRIExpression(SetupCtx& setupCtx, StepsParams &h_params, AirInstanceInfo *air_instance_info, Goldilocks::Element *d_xiChallenge, cudaStream_t stream) {
    uint64_t domainSize = (1 << setupCtx.starkInfo.starkStruct.nBitsExt);
    dim3 nThreads(friThreads(setupCtx.starkInfo.nrowsPack, domainSize));
    dim3 nBlocks((uint32_t)(domainSize / nThreads.x));

    // The fri_folded region is checked and the window sized at setup (AirInstanceInfo); this runs inside a
    // cudagraph capture region, where throwing would strand the stream in capture mode.
    const std::vector<int64_t> &openings = setupCtx.starkInfo.openingPoints;
    const uint64_t nOpeningPoints = openings.size();
    const uint64_t extendBits = setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits;
    gl64_t *d_fri = (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    gl64_t *d_coef = (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_folded", false)];
    gl64_t *d_k = d_coef + setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION;
    gl64_t *d_x = (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];

    computeFRIFoldedConstants<<<(nOpeningPoints + 63) / 64, 64, 0, stream>>>(
        nOpeningPoints, air_instance_info->opening_points, Goldilocks::w(setupCtx.starkInfo.starkStruct.nBits).fe,
        air_instance_info->friTermStart, air_instance_info->friTerms, (gl64_t*)h_params.evals,
        (gl64_t*)h_params.challenges + 4 * FIELD_EXTENSION, (gl64_t*)h_params.challenges + 5 * FIELD_EXTENSION, d_coef, d_k);
    CHECKCUDAERR(cudaGetLastError());

    const uint64_t window = air_instance_info->friWindow;
    if (window != 0) {
        computeFRIExpressionShifted<<<nBlocks, nThreads, window * sizeof(Goldilocks3GPU::Element), stream>>>(
            domainSize, extendBits, nOpeningPoints, air_instance_info->opening_points,
            *std::max_element(openings.begin(), openings.end()), window, air_instance_info->friTermStart,
            air_instance_info->friTerms, d_coef, d_k, (gl64_t*)h_params.aux_trace, (gl64_t *)h_params.pCustomCommitsFixed,
            (gl64_t *)h_params.pConstPolsExtendedTreeAddress, (gl64_t*)d_xiChallenge, d_x, d_fri);
    } else {
        computeFRIExpressionFolded<<<nBlocks, nThreads, 0, stream>>>(
            domainSize, extendBits, nOpeningPoints, air_instance_info->opening_points, air_instance_info->friTermStart,
            air_instance_info->friTerms, d_coef, d_k, (gl64_t*)h_params.aux_trace, (gl64_t *)h_params.pCustomCommitsFixed,
            (gl64_t *)h_params.pConstPolsExtendedTreeAddress, (gl64_t*)d_xiChallenge, d_x, d_fri);
    }
    CHECKCUDAERR(cudaGetLastError());
}
