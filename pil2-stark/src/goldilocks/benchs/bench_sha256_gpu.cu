// GPU SHA-256 leaf-hash and merklization benchmark, a mirror of
// bench_blake3_gpu.cu: same row counts, widths, arity and timing protocol in the
// same binary, so the ratio is two adjacent lines of one run. A self-check aborts
// before timing if the device hash disagrees with reference digests.
// BENCHMARK_MAIN() lives in bench_ntt_gpu.cu.

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/sha256_goldilocks.cuh"   // the prover class, not a bench-local copy
#include "../src/sha256_core.hpp"
#include "../src/goldilocks_tooling.hpp"   // getTreeNumElements (tree buffer sizing)

// Self-contained CUDA error check (keeps this bench independent of the prover
// utility headers; SHA-256 stays confined to the benchmark binary).
#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error '%s' at %s:%d -> %s\n", #call,         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

static constexpr uint64_t BENCH_NROWS      = 1ULL << 23;  // 8M rows (matches BLAKE3)
static constexpr uint64_t BENCH_NROWS_WIDE = 1ULL << 19;  // 512K rows (wide traces)
static constexpr uint32_t MT_ARITY         = 2;  // production geometry: 256-bit digests -> binary tree

// Self-check against hashlib.sha256 over in[i] = i & 0xFF, at the lengths
// bench_blake3_gpu.cu uses. Covers empty, an exact block, every benchmark width,
// all three padding cases, and lengths past one BLAKE3 chunk.
static void sha256_self_check()
{
    struct Case { uint32_t n_u64; uint8_t expect[32]; };
    static const Case cases[] = {
        {     0, { 0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
                   0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
                   0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
                   0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55 } },
        {     8, { 0xfd, 0xea, 0xb9, 0xac, 0xf3, 0x71, 0x03, 0x62,
                   0xbd, 0x26, 0x58, 0xcd, 0xc9, 0xa2, 0x9e, 0x8f,
                   0x9c, 0x75, 0x7f, 0xcf, 0x98, 0x11, 0x60, 0x3a,
                   0x8c, 0x44, 0x7c, 0xd1, 0xd9, 0x15, 0x11, 0x08 } },
        {     9, { 0x10, 0x7d, 0xe2, 0xbc, 0x78, 0x8e, 0x11, 0x02,
                   0x9f, 0x78, 0x51, 0xf8, 0xe1, 0xb0, 0xb5, 0xaf,
                   0xb4, 0xe3, 0x43, 0x79, 0xc7, 0x09, 0xfc, 0x84,
                   0x06, 0x89, 0xeb, 0xd3, 0xd1, 0xf5, 0x1b, 0x5b } },
        {    16, { 0x47, 0x1f, 0xb9, 0x43, 0xaa, 0x23, 0xc5, 0x11,
                   0xf6, 0xf7, 0x2f, 0x8d, 0x16, 0x52, 0xd9, 0xc8,
                   0x80, 0xcf, 0xa3, 0x92, 0xad, 0x80, 0x50, 0x31,
                   0x20, 0x54, 0x77, 0x03, 0xe5, 0x6a, 0x2b, 0xe5 } },
        {    18, { 0x66, 0xf9, 0x52, 0xa8, 0x33, 0x39, 0x27, 0x4e,
                   0xb2, 0x87, 0xb6, 0x4e, 0xf7, 0xb0, 0x28, 0xd8,
                   0x89, 0x15, 0xac, 0x6d, 0xf0, 0x6a, 0x18, 0x3f,
                   0x7c, 0x04, 0x36, 0xfa, 0x2b, 0x25, 0x10, 0x7b } },
        {    24, { 0x8b, 0x4a, 0x54, 0x48, 0x37, 0xa1, 0xa0, 0x28,
                   0x0f, 0xa8, 0xa7, 0xc8, 0x28, 0x65, 0xc2, 0x7a,
                   0x10, 0x64, 0xb3, 0xcc, 0x62, 0x81, 0xfd, 0xa0,
                   0x75, 0x35, 0x66, 0xb9, 0xbb, 0x10, 0x4a, 0x87 } },
        {    36, { 0x20, 0xd7, 0x45, 0xcd, 0x02, 0xe8, 0x9b, 0x54,
                   0x67, 0x5d, 0x3c, 0x8f, 0x10, 0x5e, 0x43, 0xb2,
                   0xcf, 0xae, 0xe9, 0xba, 0xb3, 0x20, 0xca, 0xd4,
                   0xb7, 0x85, 0x13, 0x17, 0xb1, 0x5e, 0x12, 0x3e } },
        {    56, { 0xaf, 0xcd, 0xb4, 0x64, 0x68, 0x01, 0xa7, 0xf0,
                   0xc7, 0x80, 0x48, 0x75, 0x4f, 0xf0, 0x1a, 0xde,
                   0xc0, 0xda, 0x00, 0xeb, 0x73, 0xb2, 0x0d, 0xc0,
                   0xdd, 0xe7, 0xf0, 0x89, 0xc2, 0xc2, 0x46, 0x40 } },
        {   128, { 0x78, 0x5b, 0x07, 0x51, 0xfc, 0x2c, 0x53, 0xdc,
                   0x14, 0xa4, 0xce, 0x3d, 0x80, 0x0e, 0x69, 0xef,
                   0x9c, 0xe1, 0x00, 0x9e, 0xb3, 0x27, 0xcc, 0xf4,
                   0x58, 0xaf, 0xe0, 0x9c, 0x24, 0x2c, 0x26, 0xc9 } },
        {   129, { 0x60, 0xd8, 0x70, 0x64, 0xdd, 0x49, 0x9f, 0xb1,
                   0x40, 0x9f, 0x27, 0x31, 0x82, 0x97, 0xc1, 0x9a,
                   0x46, 0x0b, 0x38, 0xdd, 0x90, 0x7d, 0xbe, 0xd7,
                   0xe6, 0xaf, 0xc1, 0xed, 0xa3, 0x20, 0x55, 0x8e } },
        {   200, { 0x35, 0x4a, 0x49, 0x8e, 0xc3, 0xbe, 0x6a, 0x3c,
                   0xc7, 0x7e, 0x4b, 0xe9, 0x5a, 0x26, 0xcf, 0x56,
                   0xcf, 0xe9, 0x11, 0x8f, 0xe2, 0xe6, 0xf4, 0x6e,
                   0x16, 0x58, 0xd2, 0x8b, 0xd7, 0xee, 0x75, 0xc1 } },
        {   256, { 0x10, 0xfc, 0x3c, 0x51, 0xa1, 0x52, 0xe9, 0x0e,
                   0x5b, 0x90, 0x31, 0x9b, 0x60, 0x1d, 0x92, 0xcc,
                   0xf3, 0x72, 0x90, 0xef, 0x53, 0xc3, 0x5f, 0xf9,
                   0x25, 0x07, 0x68, 0x7d, 0x8a, 0x91, 0x1a, 0x08 } },
        {   384, { 0x12, 0xad, 0xc9, 0xdf, 0xf8, 0x06, 0x88, 0x80,
                   0x0f, 0x2f, 0x59, 0x1f, 0x0d, 0xa6, 0xab, 0x2f,
                   0x81, 0x09, 0xd6, 0x1d, 0x91, 0x06, 0x97, 0x80,
                   0x1f, 0x57, 0x66, 0x9e, 0xc0, 0xd7, 0x19, 0xd3 } },
        {   512, { 0xc8, 0xf5, 0xd0, 0x34, 0x1d, 0x54, 0xd9, 0x51,
                   0xa7, 0x1b, 0x13, 0x6e, 0x6e, 0x2a, 0xfc, 0xb1,
                   0x4d, 0x11, 0xed, 0x84, 0x89, 0xa7, 0xae, 0x12,
                   0x6a, 0x8f, 0xee, 0x0d, 0xf6, 0xec, 0xf1, 0x93 } },
        {   879, { 0xcd, 0xe3, 0xf7, 0xc5, 0x4a, 0x8e, 0xbe, 0x0e,   // Keccakf cm2
                   0xed, 0xa0, 0x7b, 0x3b, 0x45, 0x1d, 0x1f, 0xa1,
                   0x89, 0x26, 0x9c, 0x11, 0x58, 0xea, 0x21, 0xcc,
                   0xc2, 0xfe, 0x94, 0xd3, 0x98, 0xa2, 0x38, 0xfd } },
        {  1024, { 0xdc, 0x40, 0x4a, 0x61, 0x3f, 0xed, 0xae, 0xb5,
                   0x40, 0x34, 0x51, 0x4b, 0xc6, 0x50, 0x5f, 0x56,
                   0xb9, 0x33, 0xca, 0xa5, 0x25, 0x02, 0x99, 0xba,
                   0x7d, 0x09, 0x43, 0x77, 0xa5, 0x1c, 0xaa, 0x46 } },
        {  2137, { 0x34, 0x7b, 0x9b, 0x86, 0xc9, 0x6d, 0x31, 0xff,   // Keccakf cm1
                   0x02, 0x1c, 0xd4, 0x41, 0xd1, 0x24, 0x85, 0xd5,
                   0xbb, 0x98, 0xba, 0xbd, 0x5b, 0xf9, 0x8f, 0xf7,
                   0x0a, 0xbd, 0xe2, 0xf4, 0x64, 0xcd, 0xb8, 0x61 } },
    };

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    for (const Case &c : cases)
    {
        const uint32_t nbytes = c.n_u64 * 8;
        std::vector<uint8_t> hin(nbytes);
        for (uint32_t i = 0; i < nbytes; ++i) hin[i] = (uint8_t)(i & 0xFF);

        const uint64_t alloc_u64 = c.n_u64 ? c.n_u64 : 1;  // avoid 0-byte cudaMalloc
        uint64_t *d_in = nullptr, *d_out = nullptr;
        CUDA_OK(cudaMalloc((void **)&d_in,  alloc_u64 * sizeof(uint64_t)));
        CUDA_OK(cudaMalloc((void **)&d_out, 4 * sizeof(uint64_t)));
        if (nbytes)
            CUDA_OK(cudaMemcpy(d_in, hin.data(), nbytes, cudaMemcpyHostToDevice));

        Sha256GoldilocksGPU::linearHash(d_out, d_in, c.n_u64, 1, Layout::RowMajor, stream);
        CUDA_OK(cudaStreamSynchronize(stream));

        uint64_t hout[4];
        CUDA_OK(cudaMemcpy(hout, d_out, sizeof(hout), cudaMemcpyDeviceToHost));

        if (std::memcmp(hout, c.expect, 32) != 0)
        {
            const uint8_t *got = (const uint8_t *)hout;
            fprintf(stderr, "[SHA-256 self-check] FAIL on %u u64 (%u bytes)\n  expected ",
                    c.n_u64, nbytes);
            for (int i = 0; i < 32; ++i) fprintf(stderr, "%02x", c.expect[i]);
            fprintf(stderr, "\n  got      ");
            for (int i = 0; i < 32; ++i) fprintf(stderr, "%02x", got[i]);
            fprintf(stderr, "\n");
            CUDA_OK(cudaFree(d_in));
            CUDA_OK(cudaFree(d_out));
            std::abort();
        }
        CUDA_OK(cudaFree(d_in));
        CUDA_OK(cudaFree(d_out));
    }

    CUDA_OK(cudaStreamDestroy(stream));
    fprintf(stderr,
            "[SHA-256 self-check] passed: %zu lengths (0..2137 u64, all three "
            "padding cases) match hashlib.sha256 vectors\n",
            sizeof(cases) / sizeof(cases[0]));
}


// Real ZisK shapes: the full per-stage width list at 2^23 extended rows plus a
// subset at 2^24 (a 2^23-row air at blowup 2). range(0) = nCols, range(1) = log2N.
static void ZiskShapes(benchmark::internal::Benchmark *b)
{
    static const int widths23[] = {1,3,6,8,9,10,12,13,14,15,16,18,21,24,29,32,34,35,36,38,39,40,43,44,45,66};
    for (int w : widths23) b->Args({w, 23});
    static const int widths24[] = {12, 24, 38, 66};
    for (int w : widths24) b->Args({w, 24});
}

// ---------------------------------------------------------------------------
// Trace init (same pattern as initTraceKernel_blake3).
// ---------------------------------------------------------------------------
static __global__ void initTraceKernel_sha256(uint64_t *d_trace, uint64_t nRows, uint64_t nCols)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows)
        for (uint64_t j = 0; j < nCols; ++j)
            d_trace[idx * nCols + j] = idx + j;
}

// ---------------------------------------------------------------------------
// linearHash -- one thread per row, RowMajor input.
// ---------------------------------------------------------------------------
static void LINEAR_HASH_ROWMAJOR_GPU_SHA256_BENCH(benchmark::State &state)
{
    static const bool checked = (sha256_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t nRows = 1ULL << state.range(1);
    uint64_t *d_trace, *d_hash;
    CUDA_OK(cudaMalloc((void **)&d_trace, nRows * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_hash,  nRows * 4 * sizeof(uint64_t)));

    dim3 thr(128), blk((nRows + 127) / 128);
    initTraceKernel_sha256<<<blk, thr, 0, stream>>>(d_trace, nRows, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    // Warm up
    Sha256GoldilocksGPU::linearHash(d_hash, d_trace, nCols, nRows, Layout::RowMajor, stream);
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        Sha256GoldilocksGPU::linearHash(d_hash, d_trace, nCols, nRows, Layout::RowMajor, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_hash));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(LINEAR_HASH_ROWMAJOR_GPU_SHA256_BENCH)
    ->Name("LINEAR_HASH_ROWMAJOR_GPU_SHA256_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Apply(ZiskShapes)
    ->UseRealTime();

// Wide linear hash -- fewer rows so the 2137-col trace (~9 GB) fits in GPU memory.
static void LINEAR_HASH_ROWMAJOR_WIDE_GPU_SHA256_BENCH(benchmark::State &state)
{
    static const bool checked = (sha256_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t *d_trace, *d_hash;
    CUDA_OK(cudaMalloc((void **)&d_trace, BENCH_NROWS_WIDE * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_hash,  BENCH_NROWS_WIDE * 4 * sizeof(uint64_t)));

    dim3 thr(128), blk((BENCH_NROWS_WIDE + 127) / 128);
    initTraceKernel_sha256<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS_WIDE, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    Sha256GoldilocksGPU::linearHash(d_hash, d_trace, nCols, BENCH_NROWS_WIDE, Layout::RowMajor, stream);  // warm up
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        Sha256GoldilocksGPU::linearHash(d_hash, d_trace, nCols, BENCH_NROWS_WIDE, Layout::RowMajor, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_hash));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(LINEAR_HASH_ROWMAJOR_WIDE_GPU_SHA256_BENCH)
    ->Name("LINEAR_HASH_ROWMAJOR_WIDE_GPU_SHA256_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Arg(129)->Arg(256)->Arg(512)->Arg(879)->Arg(2137)
    ->UseRealTime();

// ===========================================================================
// Full merklization: leaf linear hash + arity-2 tree reduction -- the geometry
// ZisK actually proves with (hash_family::merkle_tree_arity("blake3") == 2).
// ===========================================================================
static void MERKLETREE_AR2_ROWMAJOR_GPU_SHA256_BENCH(benchmark::State &state)
{
    static const bool checked = (sha256_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t nRows = 1ULL << state.range(1);
    uint64_t tree_size = getTreeNumElements(nRows, MT_ARITY);
    uint64_t *d_trace, *d_tree;
    CUDA_OK(cudaMalloc((void **)&d_trace, nRows * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_tree,  tree_size * sizeof(uint64_t)));

    dim3 thr(128), blk((nRows + 127) / 128);
    initTraceKernel_sha256<<<blk, thr, 0, stream>>>(d_trace, nRows, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    Sha256GoldilocksGPU::merkletree(MT_ARITY, d_tree, d_trace, nCols, nRows, Layout::RowMajor, stream);  // warm
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        Sha256GoldilocksGPU::merkletree(MT_ARITY, d_tree, d_trace, nCols, nRows, Layout::RowMajor, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_tree));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(MERKLETREE_AR2_ROWMAJOR_GPU_SHA256_BENCH)
    ->Name("MERKLETREE_AR2_ROWMAJOR_GPU_SHA256_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Apply(ZiskShapes)
    ->UseRealTime();

// Wide (long leaves) merklization, smaller row count to fit in memory.
static void MERKLETREE_AR2_WIDE_GPU_SHA256_BENCH(benchmark::State &state)
{
    static const bool checked = (sha256_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t tree_size = getTreeNumElements(BENCH_NROWS_WIDE, MT_ARITY);
    uint64_t *d_trace, *d_tree;
    CUDA_OK(cudaMalloc((void **)&d_trace, BENCH_NROWS_WIDE * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_tree,  tree_size * sizeof(uint64_t)));

    dim3 thr(128), blk((BENCH_NROWS_WIDE + 127) / 128);
    initTraceKernel_sha256<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS_WIDE, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    Sha256GoldilocksGPU::merkletree(MT_ARITY, d_tree, d_trace, nCols, BENCH_NROWS_WIDE, Layout::RowMajor, stream);
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        Sha256GoldilocksGPU::merkletree(MT_ARITY, d_tree, d_trace, nCols, BENCH_NROWS_WIDE, Layout::RowMajor, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_tree));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(MERKLETREE_AR2_WIDE_GPU_SHA256_BENCH)
    ->Name("MERKLETREE_AR2_WIDE_GPU_SHA256_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Arg(879)->Arg(2137)
    ->UseRealTime();

// Tree-reduction cost is MERKLETREE minus LINEAR_HASH: timing merkletreeReduce
// directly would fold in its cudaMalloc, copy and sync.

// Pure compression throughput, no memory traffic: a register-resident dependency
// chain of block compressions. The LINEAR_HASH / MERKLETREE benches hide this cost
// behind DRAM; grinding and the recursion circuit pay it directly.
// COMPRESS_CHAIN_GPU_BLAKE3_BENCH in bench_blake3_gpu.cu is the counterpart, so
// the ratio of the two is the compression-cost ratio.

static constexpr uint64_t CT_THREADS = 1ULL << 20;
static constexpr uint32_t CT_ITERS   = 64;   // compressions per thread

__global__ void sha256CompressChainKernel(uint64_t *__restrict__ out, uint32_t iters)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h[8], W[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = sha256core::sha_iv(i) ^ (uint32_t)tid;
#pragma unroll
    for (int i = 0; i < 16; ++i) W[i] = (uint32_t)(tid * 2654435761u) + i;

    for (uint32_t k = 0; k < iters; ++k)
    {
        sha256core::compress_in_place(h, W);
        W[0] ^= h[0];   // serialize the chain so nothing is hoisted or elided
    }
    if (tid < CT_THREADS) out[tid] = (uint64_t)h[0] | ((uint64_t)h[7] << 32);
}

static void COMPRESS_CHAIN_SHA256(benchmark::State &state)
{
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t *d_out;
    CUDA_OK(cudaMalloc((void **)&d_out, CT_THREADS * sizeof(uint64_t)));

    const uint32_t tpb  = 128;
    const uint32_t blks = (uint32_t)(CT_THREADS / tpb);

    auto launch = [&]() {
        sha256CompressChainKernel<<<blks, tpb, 0, stream>>>(d_out, CT_ITERS);
    };

    launch();  // warm up
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        launch();
        CUDA_OK(cudaStreamSynchronize(stream));
    }
    // Compressions per second, so the two rows are directly comparable.
    state.counters["Gcompress/s"] = benchmark::Counter(
        (double)CT_THREADS * CT_ITERS / 1e9, benchmark::Counter::kIsIterationInvariantRate);

    CUDA_OK(cudaFree(d_out));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(COMPRESS_CHAIN_SHA256)
    ->Name("COMPRESS_CHAIN_GPU_SHA256_BENCH")->Unit(benchmark::kMillisecond)->UseRealTime();

// ONE LEVEL of node hashing, isolated. The full MERKLETREE bench walks 23 arity-2
// levels as 23 launches, and per-node cost gets swamped: going from FIPS nodes
// (2 compressions) to node_hash (1) did not move it at all. These launch one level
// over N nodes so the per-node cost is what is measured.
//
// blake3_core.hpp is include-safe here: inline functions only, no kernel symbols.
#include "../src/blake3_core.hpp"

static constexpr uint64_t NODE_N = 1ULL << 23;   // nodes in the level

// What the node hash WAS: FIPS over 8 words, 2 compressions.
__global__ void nodeLevelFipsKernel(uint64_t *__restrict__ out, const uint64_t *__restrict__ in, uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha256core::hash_le64(in + i * 8, 8, out + i * 4);
}

// What it IS now: 1 compression.
__global__ void nodeLevelNodeHashKernel(uint64_t *__restrict__ out, const uint64_t *__restrict__ in, uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha256core::node_hash(in + i * 8, 8, out + i * 4);
}

// BLAKE3's, the baseline (also 1 compression).
__global__ void nodeLevelBlake3Kernel(uint64_t *__restrict__ out, const uint64_t *__restrict__ in, uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t cv[8];
    blake3core::compress_chunk(in + i * 8, 8, 0ull, true, cv);
    blake3core::pack4(cv, out + i * 4);
}

enum class NodeKind { Fips, NodeHash, Blake3 };

template<NodeKind K>
static void NODE_LEVEL(benchmark::State &state)
{
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t *d_in, *d_out;
    CUDA_OK(cudaMalloc((void **)&d_in, NODE_N * 8 * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_out, NODE_N * 4 * sizeof(uint64_t)));

    dim3 thr(128), blk((unsigned)((NODE_N + 127) / 128));
    initTraceKernel_sha256<<<blk, thr, 0, stream>>>(d_in, NODE_N, 8);
    CUDA_OK(cudaStreamSynchronize(stream));

    auto launch = [&]() {
        if (K == NodeKind::Fips)
            nodeLevelFipsKernel<<<blk, thr, 0, stream>>>(d_out, d_in, NODE_N);
        else if (K == NodeKind::NodeHash)
            nodeLevelNodeHashKernel<<<blk, thr, 0, stream>>>(d_out, d_in, NODE_N);
        else
            nodeLevelBlake3Kernel<<<blk, thr, 0, stream>>>(d_out, d_in, NODE_N);
    };

    launch();
    CUDA_OK(cudaStreamSynchronize(stream));
    for (auto _ : state)
    {
        launch();
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_in));
    CUDA_OK(cudaFree(d_out));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK_TEMPLATE(NODE_LEVEL, NodeKind::Blake3)
    ->Name("NODE_LEVEL_GPU_BLAKE3")->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(NODE_LEVEL, NodeKind::Fips)
    ->Name("NODE_LEVEL_GPU_SHA256_FIPS")->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(NODE_LEVEL, NodeKind::NodeHash)
    ->Name("NODE_LEVEL_GPU_SHA256_NODEHASH")->Unit(benchmark::kMillisecond)->UseRealTime();
