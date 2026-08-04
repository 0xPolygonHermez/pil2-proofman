// Streaming sponge commit (stream_commit.cu) validation.
//
// The production entry streamCommitPacked commits a bit-packed wide-trace
// witness (zisk Main shape) inside one slot buffer: chunked unpack ->
// in-place aliased LDE -> Poseidon1 W=16 sponge absorb (capacity columns
// carry the digest) -> arity-4 node reduction carved from the dead rate
// region. These tests assert the root is BIT-IDENTICAL to the production
// path (full unpack + ldeColMajor + PoseidonGoldilocksGPU<16>::merkletree)
// and exercise both aliased-LDE flows of ldeColMajor (batched at small NExt,
// serial per-column at the Main shape).
#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include "stream_commit.cuh"
#include "ntt_goldilocks.cuh"
#include "poseidon_goldilocks.cuh"
#include "cuda_utils.cuh"

// Main cm1 bit widths (zisk pil_helpers PACKED_INFO for airgroup 0 / air 0):
// 38 columns bit-packed into 14 words per row.
static const uint64_t MAIN_WIDTHS[38] = {
    32,32,32,32,32,32,1,32,1,1,64,32,1,1,1,64,32,1,4,1,8,1,1,1,64,1,64,64,1,32,38,38,38,32,32,1,1,1};
static const uint64_t MAIN_WORDS = 14;
__device__ __constant__ uint64_t TEST_WIDTHS[64];

// Host-side packer: exact inverse of the prover's unpack kernel bit walk
// (starks_gpu.cu unpack): values written LSB-first at a running bit cursor,
// split across the word boundary as low(bits_left)|high.
static void packRow(const uint64_t *vals, uint64_t nCols, const uint64_t *widths,
                    uint64_t wordsPerRow, uint64_t *out)
{
    for (uint64_t w = 0; w < wordsPerRow; w++) out[w] = 0;
    uint64_t word_idx = 0, bit_offset = 0;
    for (uint64_t c = 0; c < nCols; c++) {
        uint64_t nbits = widths[c];
        uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
        uint64_t v = vals[c] & mask;
        uint64_t bits_left = 64 - bit_offset;
        if (nbits <= bits_left) {
            out[word_idx] |= v << bit_offset;
            bit_offset += nbits;
            if (bit_offset == 64 && word_idx + 1 < wordsPerRow) { word_idx++; bit_offset = 0; }
        } else {
            out[word_idx] |= (v & ((1ULL << bits_left) - 1ULL)) << bit_offset;
            word_idx++;
            out[word_idx] |= v >> bits_left;
            bit_offset = nbits - bits_left;
        }
    }
}

// Full unpack for the reference path (all columns, ColMajor), same bit walk
// the prover uses.
__global__ static void refUnpackKernel(const uint64_t *__restrict__ src, uint64_t *__restrict__ dst,
                                       uint64_t nCols, uint64_t nRows, uint64_t wordsPerRow)
{
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) return;
    const uint64_t *packed_row = src + row * wordsPerRow;
    uint64_t word = packed_row[0];
    uint64_t word_idx = 0, bit_offset = 0;
    for (uint64_t c = 0; c < nCols; c++) {
        uint64_t nbits = TEST_WIDTHS[c];
        uint64_t val;
        uint64_t bits_left = 64 - bit_offset;
        if (nbits <= bits_left) {
            uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
            val = (word >> bit_offset) & mask;
            bit_offset += nbits;
            if (bit_offset == 64 && word_idx + 1 < wordsPerRow) { word = packed_row[++word_idx]; bit_offset = 0; }
        } else {
            uint64_t low = word >> bit_offset;
            word = packed_row[++word_idx];
            uint64_t high = word & ((1ULL << (nbits - bits_left)) - 1ULL);
            val = (high << bits_left) | low;
            bit_offset = nbits - bits_left;
        }
        dst[c * nRows + row] = val;
    }
}

// Tree element count for arity-4 Poseidon1 W=16 trees (mirrors merkletree's
// level loop including the extra-zeros padding).
static uint64_t treeNumElements(uint64_t nLeaves, uint32_t arity)
{
    const uint64_t CAP = 4;
    uint64_t total = 0, pending = nLeaves;
    while (pending > 1) {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        total += (pending + extraZeros) * CAP;
        pending = (pending + arity - 1) / arity;
    }
    return total + CAP; // root
}

// Assert streamCommitPacked's root equals the production commit path's root.
static void runStreamCommitReduced(uint64_t nBits, uint64_t nCols, int reps)
{
    using P16 = PoseidonGoldilocksGPU<16>;
    const uint64_t nBitsExt = nBits + 1;
    const uint32_t arity    = 4;
    const uint32_t CAP      = 4;
    const uint32_t TPB      = 128;
    const uint64_t N = 1ull << nBits, NExt = 1ull << nBitsExt;

    uint32_t gpu = 0; cudaGetDevice((int*)&gpu);
    P16::initConstants(&gpu, 1);
    cudaStream_t s; CHECKCUDAERR(cudaStreamCreate(&s));
    NTTGoldilocksGPU ntt;

    ASSERT_LE(nCols, 38u);
    CHECKCUDAERR(cudaMemcpyToSymbol(TEST_WIDTHS, MAIN_WIDTHS, 38*8));

    // Packed witness on the host: the production caller (proofman) hands the
    // packed trace pointer straight to commit_witness_streaming.
    std::vector<uint64_t> hPacked(N * MAIN_WORDS);
    {
        std::vector<uint64_t> rowVals(nCols);
        uint64_t x = 0x243F6A8885A308D3ull;
        for (uint64_t r = 0; r < N; r++) {
            for (uint64_t c = 0; c < nCols; c++) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;      // xorshift
                rowVals[c] = x;                                // packRow masks per width
            }
            packRow(rowVals.data(), nCols, MAIN_WIDTHS, MAIN_WORDS, &hPacked[r * MAIN_WORDS]);
        }
    }
    const uint64_t treeElems = treeNumElements(NExt, arity);

    // Reference root: full unpack + LDE + production merkletree.
    std::vector<uint64_t> rootRef(CAP);
    {
        uint64_t *d_packed; CHECKCUDAERR(cudaMalloc(&d_packed, N * MAIN_WORDS * 8));
        CHECKCUDAERR(cudaMemcpy(d_packed, hPacked.data(), hPacked.size() * 8, cudaMemcpyHostToDevice));
        gl64_t *d_src;  CHECKCUDAERR(cudaMalloc(&d_src, N * nCols * 8));
        gl64_t *d_ext;  CHECKCUDAERR(cudaMalloc(&d_ext, NExt * nCols * 8));
        uint64_t *d_tref; CHECKCUDAERR(cudaMalloc(&d_tref, treeElems * 8));
        const uint32_t ublk = (uint32_t)((N + TPB - 1) / TPB);
        refUnpackKernel<<<ublk, TPB, 0, s>>>(d_packed, (uint64_t*)d_src, nCols, N, MAIN_WORDS);
        ntt.ldeColMajor(d_ext, d_src, nBits, nBitsExt, nCols, s, true, nullptr);
        P16::merkletree(arity, d_tref, (uint64_t*)d_ext, nCols, NExt, Layout::ColMajor, s);
        CHECKCUDAERR(cudaStreamSynchronize(s));
        CHECKCUDAERR(cudaMemcpy(rootRef.data(), d_tref + treeElems - CAP, CAP*8, cudaMemcpyDeviceToHost));
        CHECKCUDAERR(cudaFree(d_packed)); CHECKCUDAERR(cudaFree(d_src));
        CHECKCUDAERR(cudaFree(d_ext));    CHECKCUDAERR(cudaFree(d_tref));
    }

    // Production entry: one slot buffer, packed witness uploaded by the lib.
    StreamCommitDims dims{nBits, nBitsExt, nCols, MAIN_WORDS};
    const uint64_t slotElems = streamCommitSlotElems(dims);
    gl64_t *d_slot; CHECKCUDAERR(cudaMalloc(&d_slot, slotElems * 8));

    std::vector<uint64_t> rootCmp(CAP);
    double total_ms = 0;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        int64_t rc = streamCommitPacked(d_slot, dims, MAIN_WIDTHS, hPacked.data(),
                                        rootCmp.data(), s);
        auto t1 = std::chrono::steady_clock::now();
        ASSERT_EQ(rc, 0);
        if (r) total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    if (reps > 1) total_ms /= (reps - 1);

    printf("[stream-commit] nBits=%lu nCols=%lu lib entry (H2D + commit): %.2f ms, slot %.0f MiB\n",
           nBits, nCols, total_ms, (double)slotElems * 8 / (1 << 20));

    for (uint32_t i = 0; i < CAP; i++)
        ASSERT_EQ(rootRef[i], rootCmp[i]) << "root element " << i << " differs";

    CHECKCUDAERR(cudaFree(d_slot));
    CHECKCUDAERR(cudaStreamDestroy(s));
}

// Small shape: exercises the BATCHED aliased-LDE flow (chunk >= 2 at small
// NExt) plus the padded last chunk (38 = 12+12+12+2).
TEST(GOLDILOCKS_TEST, stream_commit_reduced_small)
{
    runStreamCommitReduced(16, 38, 2);
}

// Main cm1 shape (2^22 x 38, blowup 2): exercises the SERIAL aliased-LDE
// flow -- the production slot configuration.
TEST(GOLDILOCKS_TEST, stream_commit_reduced_main_shape)
{
    runStreamCommitReduced(22, 38, 4);
}
