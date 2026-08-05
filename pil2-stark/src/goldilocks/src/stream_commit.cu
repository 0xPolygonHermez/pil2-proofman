#include "stream_commit.cuh"

#include <mutex>
#include <algorithm>
#include "ntt_goldilocks.cuh"
#include "poseidon_goldilocks.cuh"  
#include "cuda_utils.cuh"
#include "poseidon_goldilocks_constants.hpp"

using P16 = PoseidonGoldilocksGPU<16>;
static constexpr uint32_t SC_W      = 16;
static constexpr uint32_t SC_RATE   = P16::RATE;          // 12
static constexpr uint32_t SC_CAP    = P16::CAPACITY;      // 4
static constexpr uint32_t SC_HALF_F = P16::HALF_N_FULL_ROUNDS;
static constexpr uint32_t SC_NPART  = P16::N_PARTIAL_ROUNDS;
static constexpr uint32_t SC_ARITY  = 4;                  // fixed by the W=16 sponge
static constexpr uint32_t SC_TPB    = 128;

// The node kernel packs arity*CAPACITY children into one W-wide permutation:
// fewer children are zero-padded, but MORE would be silently truncated (the
// min() in scNodeKernel clamps the read) -- i.e. a wrong tree, not a crash.
static_assert(SC_ARITY * SC_CAP <= SC_W, "arity*CAPACITY must fit the sponge width");

// Own copies of the Poseidon1 W=16 round tables: the lib's __constant__
// symbols are TU-local, so this TU uploads its own from the shared host
// tables, once per device (guarded below).
__device__ __constant__ uint64_t SC_C16[150];
__device__ __constant__ uint64_t SC_M16[256];
__device__ __constant__ uint64_t SC_P16[256];
__device__ __constant__ uint64_t SC_S16[683];

// __constant__ memory is per-device state, so the tables are uploaded once per
// device. Devices outside the memoized range still get correct tables (they
// just re-upload ~9 KB on every commit); the bound only sizes the flag array.
static constexpr int SC_MAX_DEVICES = 64;

static void scEnsureConstants()
{
    static std::mutex mtx;
    static bool uploaded[SC_MAX_DEVICES] = {};
    int dev = 0;
    CHECKCUDAERR(cudaGetDevice(&dev));
    const bool memoized = (dev >= 0 && dev < SC_MAX_DEVICES);
    std::lock_guard<std::mutex> lk(mtx);
    if (memoized && uploaded[dev]) return;
    CHECKCUDAERR(cudaMemcpyToSymbol(SC_C16, PoseidonGoldilocksConstants::C16, 150 * 8));
    CHECKCUDAERR(cudaMemcpyToSymbol(SC_M16, PoseidonGoldilocksConstants::M16, 256 * 8));
    CHECKCUDAERR(cudaMemcpyToSymbol(SC_P16, PoseidonGoldilocksConstants::P16, 256 * 8));
    CHECKCUDAERR(cudaMemcpyToSymbol(SC_S16, PoseidonGoldilocksConstants::S16, 683 * 8));
    if (memoized) uploaded[dev] = true;
}

// Advance the cursor (word,idx,off) over `nbits` of a packed stream, returning the
// value when Extract. The prover's unpack bit walk (starks_gpu.cu unpack /
// idx_read_bits) -- the cursor updates are kept character-for-character identical so
// slot roots match the prover's cm1.
//
// A chunked commit re-walks columns [0, c0) purely to reposition the cursors, and the
// value there is dead. Extract=false is that case: one template keeps a single copy of
// the cursor logic (so skip and read can never drift) while the mask/shift/or folds
// away. The word loads stay -- they are what advances the cursor, and they are also
// why this is not a speedup: the kernel is DRAM-bound, so dropping the arithmetic
// measures as noise. It is kept for the single-source-of-truth cursor, not for time.
template <bool Extract>
__device__ __forceinline__ static uint64_t scStepBits(
    const uint64_t *__restrict__ base, uint64_t words,
    uint64_t &word, uint64_t &idx, uint64_t &off, uint64_t nbits)
{
    uint64_t val = 0;
    uint64_t bits_left = 64 - off;
    if (nbits <= bits_left) {
        if (Extract) {
            uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
            val = (word >> off) & mask;
        }
        off += nbits;
        if (off == 64 && idx + 1 < words) { word = base[++idx]; off = 0; }
    } else {
        uint64_t low = word >> off;
        word = base[++idx];
        if (Extract) {
            uint64_t high = word & ((1ULL << (nbits - bits_left)) - 1ULL);
            val = (high << bits_left) | low;
        }
        off = nbits - bits_left;
    }
    return val;
}

// The prover's unpack bit walk (starks_gpu.cu unpack), writing only columns
// [c0, c0+cc) into cc ColMajor columns of dst (columns before c0 are skipped
// by advancing the cursor). Widths come from global memory so concurrent
// slots with different shapes never race on a shared __constant__ symbol.
__global__ static void scUnpackRangeKernel(const uint64_t *__restrict__ src,
                                           const uint64_t *__restrict__ widths,
                                           uint64_t *__restrict__ dst,
                                           uint64_t nCols, uint64_t nRows,
                                           uint64_t wordsPerRow, uint32_t c0, uint32_t cc)
{
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) return;
    const uint64_t *packed_row = src + row * wordsPerRow;
    uint64_t word = packed_row[0];
    uint64_t word_idx = 0, bit_offset = 0;
    // Reposition over the columns this chunk does not write, then extract.
    for (uint64_t c = 0; c < (uint64_t)c0 && c < nCols; c++)
        scStepBits<false>(packed_row, wordsPerRow, word, word_idx, bit_offset, widths[c]);
    for (uint64_t c = c0; c < nCols && c < (uint64_t)c0 + cc; c++) {
        uint64_t val = scStepBits<true>(packed_row, wordsPerRow, word, word_idx, bit_offset, widths[c]);
        dst[(uint64_t)(c - c0) * nRows + row] = val;
    }
}

// Indexed counterpart of scUnpackRangeKernel: two cursors (compact row, shared
// instruction table), each output column sourced per colSource. Mirrors
// unpack_indexed (starks_gpu.cu), so a slot root equals the prover's cm1 root.
// Columns before c0 still have to be walked -- both cursors are sequential.
__global__ static void scUnpackRangeIndexedKernel(const uint64_t *__restrict__ src,
                                                  const uint64_t *__restrict__ table,
                                                  const uint64_t *__restrict__ widths,
                                                  const uint8_t *__restrict__ colSource,
                                                  uint64_t *__restrict__ dst,
                                                  uint64_t nCols, uint64_t nRows,
                                                  uint64_t wordsPerRow, uint64_t wordsPerEntry,
                                                  uint64_t numEntries, uint64_t indexBits,
                                                  uint32_t c0, uint32_t cc)
{
    // Per-column metadata is uniform across rows, so stage it once per block the way
    // the prover's unpack does with its widths. One shared word carries BOTH the width
    // and the source flag (nbits in the low 32 bits, source in bit 32) -- nbits <= 64,
    // so they fit, and the inner loops take one shared read instead of two global ones.
    // nCols <= SC_MAX_COLS, so at most 512 B per block. Note this is a tidiness/latency
    // measure, not a throughput lever: the kernel runs at ~83% of DRAM roofline, so its
    // cost is the row traffic below, not this metadata.
    extern __shared__ uint64_t scInfo[];
    for (uint64_t i = threadIdx.x; i < nCols; i += blockDim.x)
        scInfo[i] = widths[i] | ((uint64_t)(colSource[i] != 0) << 32);
    __syncthreads();

    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) return;

    const uint64_t *rbase = src + row * wordsPerRow;
    uint64_t rword = rbase[0], ridx = 0, roff = 0;
    uint64_t index = scStepBits<true>(rbase, wordsPerRow, rword, ridx, roff, indexBits);
    // A witness bug can put an out-of-range index here. The CPU unpack reports
    // it and aborts; a kernel cannot, so fall back to entry 0 to stay in bounds
    // -- the root then simply fails verification instead of reading past the table.
    if (index >= numEntries) index = 0;

    const uint64_t *tbase = table + index * wordsPerEntry;
    uint64_t tword = tbase[0], tidx = 0, toff = 0;

    // Both cursors are sequential, so the columns this chunk does not write still have
    // to be walked -- but only to reposition, so their values are never materialized.
    for (uint64_t c = 0; c < (uint64_t)c0 && c < nCols; c++) {
        uint64_t info = scInfo[c];
        if (info >> 32) scStepBits<false>(tbase, wordsPerEntry, tword, tidx, toff, info & 0xFFFFFFFFull);
        else            scStepBits<false>(rbase, wordsPerRow,   rword, ridx, roff, info & 0xFFFFFFFFull);
    }
    for (uint64_t c = c0; c < nCols && c < (uint64_t)c0 + cc; c++) {
        uint64_t info = scInfo[c];
        uint64_t nbits = info & 0xFFFFFFFFull;
        // Warp-uniform: colSource depends only on c, so this never diverges.
        uint64_t val = (info >> 32)
            ? scStepBits<true>(tbase, wordsPerEntry, tword, tidx, toff, nbits)
            : scStepBits<true>(rbase, wordsPerRow,   rword, ridx, roff, nbits);
        dst[(uint64_t)(c - c0) * nRows + row] = val;
    }
}

// Absorb one <=RATE-column chunk into the sponge, one thread per extended row.
// Matches linearHashKernel_pos1: rate slots [0..cc) = data, [cc..RATE) = 0,
// capacity slots = 0 on the first chunk else the previous digest. The digest
// stays in the capacity columns: after the last chunk they ARE the leaves.
__global__ static void scAbsorbChunkKernel(const gl64_t *__restrict__ rate,
                                           gl64_t *__restrict__ cap,
                                           uint32_t cc, bool first, uint64_t nRows)
{
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nRows) return;

    for (uint32_t i = 0; i < SC_RATE; ++i)
        scratchpad[i * blockDim.x + threadIdx.x] =
            (i < cc) ? rate[(uint64_t)i * nRows + tid] : gl64_t(uint64_t(0));
#pragma unroll
    for (uint32_t i = 0; i < SC_CAP; ++i)
        scratchpad[(SC_RATE + i) * blockDim.x + threadIdx.x] =
            first ? gl64_t(uint64_t(0)) : cap[(uint64_t)i * nRows + tid];

    poseidon1PermuteSmem<SC_W, SC_HALF_F, SC_NPART>(
        (const gl64_t *)SC_C16, (const gl64_t *)SC_S16,
        (const gl64_t *)SC_M16, (const gl64_t *)SC_P16);

#pragma unroll
    for (uint32_t i = 0; i < SC_CAP; ++i)
        cap[(uint64_t)i * nRows + tid] = scratchpad[i * blockDim.x + threadIdx.x];
}

// After the last absorb the capacity columns ARE the leaf digests (ColMajor).
// Lay them out row-major at the tree base -- carved from the rate region,
// which is dead once the final absorb has consumed it (cols 12..15 vs 0..).
__global__ static void scCapToLeavesKernel(const gl64_t *__restrict__ cap,
                                           uint64_t *__restrict__ tree, uint64_t nRows)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nRows) return;
#pragma unroll
    for (uint32_t i = 0; i < SC_CAP; ++i)
        tree[tid * SC_CAP + i] = ((const uint64_t *)cap)[(uint64_t)i * nRows + tid];
}

// Same node hash as merkleNodeKernel_pos1 (TU-local in the lib).
__global__ static void scNodeKernel(uint64_t nextN, uint64_t nextIndex, uint64_t pending,
                                    uint32_t arity, uint64_t *cursor)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN) return;
    const uint32_t stride = arity * SC_CAP;
    const uint64_t base = nextIndex + tid * (uint64_t)stride;
    const uint32_t n = (stride < SC_W) ? stride : SC_W;
    for (uint32_t i = 0; i < n; ++i)
        scratchpad[i * blockDim.x + threadIdx.x] = ((gl64_t *)cursor)[base + i];
#pragma unroll
    for (uint32_t i = 0; i < SC_W; ++i)
        if (i >= n) scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));
    poseidon1PermuteSmem<SC_W, SC_HALF_F, SC_NPART>(
        (const gl64_t *)SC_C16, (const gl64_t *)SC_S16,
        (const gl64_t *)SC_M16, (const gl64_t *)SC_P16);
    gl64_t *out = (gl64_t *)(&cursor[nextIndex + (pending + tid) * SC_CAP]);
#pragma unroll
    for (uint32_t i = 0; i < SC_CAP; ++i)
        out[i] = scratchpad[i * blockDim.x + threadIdx.x];
}

// Same level loop as PoseidonGoldilocksGPU::merkletree, starting from leaves.
static void scReduceTree(uint64_t *d_tree, uint64_t nLeaves, cudaStream_t s)
{
    uint64_t pending = nLeaves, nextIndex = 0;
    uint64_t nextN = (pending + SC_ARITY - 1) / SC_ARITY;
    while (pending > 1) {
        uint64_t extraZeros = (SC_ARITY - (pending % SC_ARITY)) % SC_ARITY;
        if (extraZeros)
            CHECKCUDAERR(cudaMemsetAsync(d_tree + nextIndex + pending * SC_CAP, 0,
                                         extraZeros * SC_CAP * 8, s));
        uint32_t tpb = (nextN < SC_TPB) ? (uint32_t)nextN : SC_TPB;
        uint32_t blks = (uint32_t)((nextN + SC_TPB - 1) / SC_TPB);
        scNodeKernel<<<blks, tpb, (size_t)tpb * SC_W * 8, s>>>(nextN, nextIndex,
                                                               pending + extraZeros, SC_ARITY, d_tree);
        CHECKCUDAERR(cudaGetLastError());
        nextIndex += (pending + extraZeros) * SC_CAP;
        pending = nextN;
        nextN = (pending + SC_ARITY - 1) / SC_ARITY;
    }
}

static uint64_t scTreeNumElements(uint64_t nLeaves)
{
    uint64_t total = 0, pending = nLeaves;
    while (pending > 1) {
        uint64_t extraZeros = (SC_ARITY - (pending % SC_ARITY)) % SC_ARITY;
        total += (pending + extraZeros) * SC_CAP;
        pending = (pending + SC_ARITY - 1) / SC_ARITY;
    }
    return total + SC_CAP; // root
}

uint64_t streamCommitSlotElems(const StreamCommitDims &dims)
{
    uint64_t N = 1ull << dims.nBits, NExt = 1ull << dims.nBitsExt;
    return SC_MAX_COLS + N * dims.wordsPerRow + (uint64_t)SC_W * NExt + N;
}

int64_t streamCommitPacked(gl64_t *slotBase, const StreamCommitDims &dims,
                           const uint64_t *colWidths, const void *hPacked,
                           uint64_t *hRoot, cudaStream_t stream,
                           const uint8_t *dColSource, const uint64_t *dTable)
{
    if (dims.nCols == 0 || dims.nCols > SC_MAX_COLS) return -1;
    if (dims.nBitsExt <= dims.nBits) return -2;
    // Indexed descriptor must be complete or entirely absent.
    const bool indexed = (dColSource != nullptr);
    if (indexed && (dTable == nullptr || dims.wordsPerEntry == 0 || dims.numEntries == 0 ||
                    dims.indexBits == 0 || dims.indexBits > 64))
        return -4;

    const uint64_t N = 1ull << dims.nBits, NExt = 1ull << dims.nBitsExt;
    const uint64_t treeElems = scTreeNumElements(NExt);
    // Tree carved from the rate region: holds for arity 4 (16/3*NExt < 12*NExt).
    if (treeElems > (uint64_t)SC_RATE * NExt) return -3;

    scEnsureConstants();

    // Slot layout (see streamCommitSlotElems).
    uint64_t *d_widths = (uint64_t *)slotBase;
    uint64_t *d_packed = d_widths + SC_MAX_COLS;
    gl64_t *d_state    = (gl64_t *)(d_packed + N * dims.wordsPerRow);
    gl64_t *d_scratch  = d_state + (uint64_t)SC_W * NExt;
    gl64_t *d_rate     = d_state;                              // cols 0..11
    gl64_t *d_cap      = d_state + (uint64_t)SC_RATE * NExt;   // cols 12..15
    uint64_t *d_tree   = (uint64_t *)d_state;                  // valid only after last absorb

    CHECKCUDAERR(cudaMemcpyAsync(d_widths, colWidths, dims.nCols * 8, cudaMemcpyHostToDevice, stream));
    // Chunked DIRECT copy: the witness pool is host-registered (MemoryHandler),
    // so each block is a plain pinned DMA with no staging memcpy; short blocks
    // keep any single transfer from monopolizing the copy engine.
    const uint64_t packedBytes = N * dims.wordsPerRow * 8;
    const uint64_t blockBytes = 32ull << 20;
    for (uint64_t off = 0; off < packedBytes; off += blockBytes) {
        uint64_t len = std::min(blockBytes, packedBytes - off);
        CHECKCUDAERR(cudaMemcpyAsync((uint8_t *)d_packed + off, (const uint8_t *)hPacked + off, len,
                                     cudaMemcpyHostToDevice, stream));
    }

    NTTGoldilocksGPU ntt;
    const uint32_t ublk = (uint32_t)((N + SC_TPB - 1) / SC_TPB);
    const uint32_t ablk = (uint32_t)((NExt + SC_TPB - 1) / SC_TPB);
    const uint32_t nChunks = (uint32_t)((dims.nCols + SC_RATE - 1) / SC_RATE);

    for (uint32_t k = 0; k < nChunks; k++) {
        uint32_t cc = (uint32_t)((uint64_t)(k + 1) * SC_RATE <= dims.nCols ? SC_RATE
                                                                           : dims.nCols - (uint64_t)k * SC_RATE);
        if (indexed) {
            scUnpackRangeIndexedKernel<<<ublk, SC_TPB, dims.nCols * sizeof(uint64_t), stream>>>(
                d_packed, dTable, d_widths, dColSource, (uint64_t *)d_rate,
                dims.nCols, N, dims.wordsPerRow, dims.wordsPerEntry, dims.numEntries,
                dims.indexBits, (uint32_t)(k * SC_RATE), cc);
        } else {
            scUnpackRangeKernel<<<ublk, SC_TPB, 0, stream>>>(d_packed, d_widths, (uint64_t *)d_rate,
                                                             dims.nCols, N, dims.wordsPerRow,
                                                             (uint32_t)(k * SC_RATE), cc);
        }
        CHECKCUDAERR(cudaGetLastError());
        // In-place spread: src == dst base (equal-base aliasing path);
        // preserve_src must be false under aliasing.
        ntt.ldeColMajor(d_rate, d_rate, dims.nBits, dims.nBitsExt, cc, stream, false, d_scratch);
        scAbsorbChunkKernel<<<ablk, SC_TPB, (size_t)SC_TPB * SC_W * 8, stream>>>(
            d_rate, d_cap, cc, k == 0, NExt);
        CHECKCUDAERR(cudaGetLastError());
    }

    scCapToLeavesKernel<<<ablk, SC_TPB, 0, stream>>>(d_cap, d_tree, NExt);
    CHECKCUDAERR(cudaGetLastError());
    scReduceTree(d_tree, NExt, stream);

    CHECKCUDAERR(cudaMemcpyAsync(hRoot, d_tree + treeElems - SC_CAP, SC_CAP * 8,
                                 cudaMemcpyDeviceToHost, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    return 0;
}
