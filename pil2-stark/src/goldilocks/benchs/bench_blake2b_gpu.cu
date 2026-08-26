// ---------------------------------------------------------------------------
// bench_blake2b_gpu.cu -- GPU BLAKE2b-256 leaf + merklization benchmark.
//
// Same shapes as the Poseidon / BLAKE3 GPU benches (BENCH_NROWS = 1<<23, the
// ZisK Air per-stage column counts, one thread per hash, RowMajor, RealTime,
// kMillisecond) so Blake2b-vs-Poseidon ratios are apples-to-apples.
//
// A self-check validates the device hash against ground-truth digests from
// Python's hashlib.blake2b (digest_size=32) before timing.
//
// No BENCHMARK_MAIN() here -- bench_ntt_gpu.cu defines it for the whole binary.
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "blake2b_device.cuh"
#include "../src/goldilocks_tooling.hpp"   // getTreeNumElements

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error '%s' at %s:%d -> %s\n", #call,         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

static constexpr uint64_t BENCH_NROWS      = 1ULL << 23;  // 8M rows
static constexpr uint64_t BENCH_NROWS_WIDE = 1ULL << 19;  // 512K rows (wide cols)
static constexpr uint32_t MT_ARITY         = 4;

// ---------------------------------------------------------------------------
// Correctness self-check vs hashlib.blake2b(data, digest_size=32) over the byte
// pattern in[i] = i & 0xFF. Covers 1..134 blocks (incl. Keccakf widths).
// ---------------------------------------------------------------------------
static void blake2b_self_check()
{
    struct Case { uint32_t n_u64; uint8_t expect[32]; };
    static const Case cases[] = {
        {    0, { 0x0e, 0x57, 0x51, 0xc0, 0x26, 0xe5, 0x43, 0xb2,
                  0xe8, 0xab, 0x2e, 0xb0, 0x60, 0x99, 0xda, 0xa1,
                  0xd1, 0xe5, 0xdf, 0x47, 0x77, 0x8f, 0x77, 0x87,
                  0xfa, 0xab, 0x45, 0xcd, 0xf1, 0x2f, 0xe3, 0xa8 } },
        {    8, { 0x10, 0xd8, 0xe6, 0xd5, 0x34, 0xb0, 0x09, 0x39,
                  0x84, 0x3f, 0xe9, 0xdc, 0xc4, 0xda, 0xe4, 0x8c,
                  0xdf, 0x00, 0x8f, 0x6b, 0x8b, 0x2b, 0x82, 0xb1,
                  0x56, 0xf5, 0x40, 0x4d, 0x87, 0x48, 0x87, 0xf5 } },
        {    9, { 0xad, 0xb3, 0x9b, 0x3e, 0x53, 0xa7, 0x91, 0xdf,
                  0xd3, 0x2e, 0xca, 0x83, 0x19, 0x1e, 0xb8, 0x2f,
                  0xde, 0xf4, 0x32, 0x1a, 0xc5, 0x04, 0x53, 0x9b,
                  0x25, 0x7c, 0x33, 0x76, 0xda, 0x24, 0x0a, 0xca } },
        {   16, { 0xc3, 0x58, 0x2f, 0x71, 0xeb, 0xb2, 0xbe, 0x66,
                  0xfa, 0x5d, 0xd7, 0x50, 0xf8, 0x0b, 0xaa, 0xe9,
                  0x75, 0x54, 0xf3, 0xb0, 0x15, 0x66, 0x3c, 0x8b,
                  0xe3, 0x77, 0xcf, 0xcb, 0x24, 0x88, 0xc1, 0xd1 } },
        {   17, { 0x6a, 0x35, 0xd3, 0xda, 0xdc, 0x62, 0xdf, 0xe7,
                  0x81, 0x95, 0x19, 0xf9, 0x21, 0x81, 0xb2, 0xf8,
                  0xd3, 0x8f, 0x5e, 0x0e, 0xd3, 0xd5, 0x1a, 0x22,
                  0xcf, 0x8a, 0x13, 0x3a, 0xb6, 0x28, 0xd6, 0xf4 } },
        {   18, { 0xe1, 0x5d, 0xc6, 0x23, 0x8e, 0x2e, 0x58, 0xe9,
                  0xea, 0x21, 0x2b, 0x0d, 0x7a, 0xbf, 0xd7, 0x00,
                  0xda, 0x3a, 0xe5, 0x12, 0x0d, 0x4d, 0x60, 0x13,
                  0x41, 0xce, 0x9e, 0x42, 0x4a, 0x7c, 0x58, 0x28 } },
        {   24, { 0x7b, 0x8f, 0x54, 0x24, 0x74, 0x22, 0xc4, 0x3a,
                  0x6d, 0x36, 0x97, 0x72, 0x60, 0xe1, 0x95, 0xd0,
                  0x6e, 0x1d, 0xbb, 0xa4, 0x4c, 0x39, 0x2b, 0x3f,
                  0xe7, 0x6d, 0xcf, 0x4a, 0x96, 0xc4, 0x33, 0xd5 } },
        {   36, { 0x43, 0x97, 0x5b, 0xf5, 0xdf, 0xb7, 0x46, 0x7f,
                  0x45, 0xba, 0x74, 0x5d, 0x2b, 0x98, 0x44, 0x1a,
                  0x08, 0x78, 0xdb, 0x65, 0x66, 0x50, 0x82, 0x21,
                  0x77, 0x00, 0x94, 0x04, 0x81, 0x74, 0xd2, 0x30 } },
        {   56, { 0x17, 0x0b, 0x3e, 0x6c, 0x35, 0x7c, 0x15, 0xb2,
                  0x91, 0xa0, 0xab, 0x25, 0x2a, 0x16, 0x12, 0x35,
                  0xd7, 0xdb, 0x7b, 0xc6, 0x1c, 0xf3, 0x12, 0xf7,
                  0x1b, 0x56, 0xe7, 0x91, 0x92, 0x70, 0xea, 0x81 } },
        {  128, { 0xf1, 0x55, 0x1f, 0xee, 0xb2, 0x52, 0xc7, 0xe6,
                  0x0b, 0xb3, 0x62, 0x20, 0x5b, 0xd1, 0xac, 0x2f,
                  0x70, 0xb1, 0x45, 0x26, 0x0a, 0x91, 0xd4, 0x1e,
                  0x8c, 0x5d, 0x0a, 0x18, 0x75, 0x49, 0xa5, 0xf2 } },
        {  256, { 0x6e, 0xd9, 0xbf, 0x54, 0x57, 0x05, 0xdb, 0xa5,
                  0x97, 0x1e, 0x83, 0xa1, 0xf2, 0xa4, 0x6a, 0x9d,
                  0xd5, 0xac, 0x2f, 0xe8, 0xa9, 0x34, 0xf1, 0x3c,
                  0xee, 0x8d, 0x35, 0x30, 0x03, 0xea, 0xf9, 0x08 } },
        {  879, { 0x0d, 0xd9, 0x0a, 0x0f, 0x99, 0xa9, 0x8d, 0x02,
                  0xa2, 0x18, 0x1c, 0xd0, 0x3f, 0x9e, 0x8f, 0x9c,
                  0x98, 0x5c, 0x90, 0xfd, 0xac, 0xf0, 0xa5, 0x84,
                  0x54, 0x14, 0x13, 0xf7, 0x51, 0xc7, 0x59, 0xd6 } },
        { 1024, { 0x5f, 0xac, 0xd7, 0xce, 0x6f, 0x94, 0xc7, 0x93,
                  0xe0, 0xc9, 0xdf, 0x34, 0x5c, 0xcc, 0xc2, 0x72,
                  0xad, 0xa6, 0xe3, 0x2e, 0xa1, 0x7b, 0xe9, 0x96,
                  0xe5, 0x12, 0xf3, 0x46, 0xee, 0xf3, 0x06, 0x52 } },
        { 2137, { 0x0a, 0xae, 0x9c, 0x18, 0x31, 0xca, 0xa7, 0x4e,
                  0x7a, 0x60, 0x6b, 0x3c, 0xa0, 0x9d, 0xfb, 0x6c,
                  0xec, 0x2d, 0x43, 0x39, 0x33, 0x54, 0x69, 0xa1,
                  0xf7, 0xb6, 0x87, 0xb9, 0xda, 0xfc, 0x17, 0xc3 } },
    };

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    for (const Case &c : cases)
    {
        const uint32_t nbytes = c.n_u64 * 8;
        std::vector<uint8_t> hin(nbytes);
        for (uint32_t i = 0; i < nbytes; ++i) hin[i] = (uint8_t)(i & 0xFF);

        const uint64_t alloc_u64 = c.n_u64 ? c.n_u64 : 1;
        uint64_t *d_in = nullptr, *d_out = nullptr;
        CUDA_OK(cudaMalloc((void **)&d_in,  alloc_u64 * sizeof(uint64_t)));
        CUDA_OK(cudaMalloc((void **)&d_out, 4 * sizeof(uint64_t)));
        if (nbytes)
            CUDA_OK(cudaMemcpy(d_in, hin.data(), nbytes, cudaMemcpyHostToDevice));

        blake2bgpu::blake2b_linear_hash(d_out, d_in, c.n_u64, 1, stream);
        CUDA_OK(cudaStreamSynchronize(stream));

        uint64_t hout[4];
        CUDA_OK(cudaMemcpy(hout, d_out, sizeof(hout), cudaMemcpyDeviceToHost));

        if (std::memcmp(hout, c.expect, 32) != 0)
        {
            const uint8_t *got = (const uint8_t *)hout;
            fprintf(stderr, "[BLAKE2b self-check] FAIL on %u u64 (%u bytes)\n  expected ",
                    c.n_u64, nbytes);
            for (int i = 0; i < 32; ++i) fprintf(stderr, "%02x", c.expect[i]);
            fprintf(stderr, "\n  got      ");
            for (int i = 0; i < 32; ++i) fprintf(stderr, "%02x", got[i]);
            fprintf(stderr, "\n");
            std::abort();
        }
        CUDA_OK(cudaFree(d_in));
        CUDA_OK(cudaFree(d_out));
    }

    CUDA_OK(cudaStreamDestroy(stream));
    fprintf(stderr, "[BLAKE2b self-check] passed: %zu lengths (up to 2137 u64 = 134 "
                    "blocks) match hashlib.blake2b vectors\n",
            sizeof(cases) / sizeof(cases[0]));
}

static __global__ void initTraceKernel_b2(uint64_t *d_trace, uint64_t nRows, uint64_t nCols)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows)
        for (uint64_t j = 0; j < nCols; ++j)
            d_trace[idx * nCols + j] = idx + j;
}

#define NCOLS_ARGS                                                             \
    ->Arg(1)->Arg(3)->Arg(6)->Arg(8)->Arg(9)->Arg(10)->Arg(12)->Arg(13)        \
    ->Arg(14)->Arg(15)->Arg(16)->Arg(18)->Arg(21)->Arg(24)->Arg(29)->Arg(32)   \
    ->Arg(34)->Arg(35)->Arg(36)->Arg(38)->Arg(39)->Arg(40)->Arg(43)->Arg(44)   \
    ->Arg(45)->Arg(66)

// ===================================================================
// Leaf / linear hash
// ===================================================================
template<uint64_t NROWS>
static void LINEAR_HASH_BLAKE2B(benchmark::State &state)
{
    static const bool checked = (blake2b_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t *d_trace, *d_hash;
    CUDA_OK(cudaMalloc((void **)&d_trace, NROWS * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_hash,  NROWS * 4 * sizeof(uint64_t)));

    dim3 thr(128), blk((NROWS + 127) / 128);
    initTraceKernel_b2<<<blk, thr, 0, stream>>>(d_trace, NROWS, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    blake2bgpu::blake2b_linear_hash(d_hash, d_trace, nCols, NROWS, stream);  // warm
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        blake2bgpu::blake2b_linear_hash(d_hash, d_trace, nCols, NROWS, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }
    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_hash));
    CUDA_OK(cudaStreamDestroy(stream));
}

// ===================================================================
// Full merklization (leaf + arity-4 tree) and pure reduce
// ===================================================================
template<uint64_t NROWS>
static void MERKLETREE_BLAKE2B(benchmark::State &state)
{
    static const bool checked = (blake2b_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t tree_size = getTreeNumElements(NROWS, MT_ARITY);
    uint64_t *d_trace, *d_tree;
    CUDA_OK(cudaMalloc((void **)&d_trace, NROWS * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_tree,  tree_size * sizeof(uint64_t)));

    dim3 thr(128), blk((NROWS + 127) / 128);
    initTraceKernel_b2<<<blk, thr, 0, stream>>>(d_trace, NROWS, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    blake2bgpu::blake2b_merkletree(MT_ARITY, d_tree, d_trace, nCols, NROWS, stream);  // warm
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        blake2bgpu::blake2b_merkletree(MT_ARITY, d_tree, d_trace, nCols, NROWS, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }
    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_tree));
    CUDA_OK(cudaStreamDestroy(stream));
}

static void MERKLE_REDUCE_BLAKE2B(benchmark::State &state)
{
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t N = state.range(0);
    uint64_t tree_size = getTreeNumElements(N, MT_ARITY);
    uint64_t *d_tree;
    CUDA_OK(cudaMalloc((void **)&d_tree, tree_size * sizeof(uint64_t)));

    dim3 thr(128), blk((N + 127) / 128);
    initTraceKernel_b2<<<blk, thr, 0, stream>>>(d_tree, N, 4);
    CUDA_OK(cudaStreamSynchronize(stream));

    blake2bgpu::blake2b_merkletree_reduce(MT_ARITY, d_tree, N, stream);
    CUDA_OK(cudaStreamSynchronize(stream));
    for (auto _ : state)
    {
        blake2bgpu::blake2b_merkletree_reduce(MT_ARITY, d_tree, N, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }
    CUDA_OK(cudaFree(d_tree));
    CUDA_OK(cudaStreamDestroy(stream));
}

// ===================================================================
// Registration
// ===================================================================
BENCHMARK_TEMPLATE(LINEAR_HASH_BLAKE2B, 1ULL << 23)
    ->Name("LINEAR_HASH_ROWMAJOR_GPU_BLAKE2B_BENCH")
    ->Unit(benchmark::kMillisecond) NCOLS_ARGS ->UseRealTime();
BENCHMARK_TEMPLATE(LINEAR_HASH_BLAKE2B, 1ULL << 19)
    ->Name("LINEAR_HASH_ROWMAJOR_WIDE_GPU_BLAKE2B_BENCH")
    ->Unit(benchmark::kMillisecond)->Arg(129)->Arg(256)->Arg(512)->Arg(879)->Arg(2137)->UseRealTime();

BENCHMARK_TEMPLATE(MERKLETREE_BLAKE2B, 1ULL << 23)
    ->Name("MERKLETREE_AR4_ROWMAJOR_GPU_BLAKE2B_BENCH")
    ->Unit(benchmark::kMillisecond) NCOLS_ARGS ->UseRealTime();
BENCHMARK_TEMPLATE(MERKLETREE_BLAKE2B, 1ULL << 19)
    ->Name("MERKLETREE_AR4_WIDE_GPU_BLAKE2B_BENCH")
    ->Unit(benchmark::kMillisecond)->Arg(879)->Arg(2137)->UseRealTime();

BENCHMARK(MERKLE_REDUCE_BLAKE2B)
    ->Name("MERKLE_REDUCE_AR4_GPU_BLAKE2B_BENCH")
    ->Unit(benchmark::kMillisecond)->Arg(1ULL << 23)->UseRealTime();

#undef NCOLS_ARGS
