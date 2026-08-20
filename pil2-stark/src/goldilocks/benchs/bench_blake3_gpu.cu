// ---------------------------------------------------------------------------
// bench_blake3_gpu.cu -- GPU BLAKE3 linear/leaf-hash benchmark.
//
// Measures BLAKE3 throughput for the leaf/linear-hash workload (one row of
// num_cols Goldilocks u64 elements -> one 4-word/256-bit digest), under the
// exact same shapes as the Poseidon v1 / Poseidon2 LINEAR_HASH GPU benches
// (BENCH_NROWS = 1<<23, NCOLS in {9,18,24,36,56}, one thread per row,
// per-iteration stream sync, RealTime, kMillisecond) so the Blake3-vs-Poseidon
// ratio is apples-to-apples in the same benchsgpu binary.
//
// A one-shot correctness self-check runs before timing and aborts the process
// if the device hash disagrees with official BLAKE3 vectors -- a fast-but-wrong
// port must not masquerade as a result.
//
// No BENCHMARK_MAIN() here -- bench_ntt_gpu.cu defines it for the whole binary.
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "blake3_device.cuh"
#include "../src/goldilocks_tooling.hpp"   // getTreeNumElements (tree buffer sizing)
#include "../src/blake3_goldilocks.cuh"    // grinding: the entry point the prover calls

// Self-contained CUDA error check (keeps this bench independent of the prover
// utility headers; BLAKE3 stays confined to the benchmark binary).
#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error '%s' at %s:%d -> %s\n", #call,         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

static constexpr uint64_t BENCH_NROWS = 1ULL << 23;  // 8M rows (matches Poseidon)

// ---------------------------------------------------------------------------
// Correctness self-check.
//
// Ground-truth digests produced by the official `blake3` Rust crate (v1.8.5)
// over the input byte pattern in[i] = i & 0xFF, for lengths chosen to exercise
// the empty path, a single full block (validates the message schedule), and
// 16-block chunk chaining. Output bytes are the little-endian serialization of
// the chaining value, so the uint8_t view of the 4 u64 output words equals the
// 32 reference bytes directly.
// ---------------------------------------------------------------------------
static void blake3_self_check()
{
    // n_u64 covers: empty; 1 full block; every benchmark width 9/18/24/36/56
    // (9/18/36 have a PARTIAL last block, 24/56 are exact multiples); and a full
    // single chunk (16 blocks). The partial-block widths are the ones the timing
    // benchmark actually hashes.
    struct Case { uint32_t n_u64; uint8_t expect[32]; };
    static const Case cases[] = {
        {   0, { 0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6,
                 0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9, 0x49,
                 0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1, 0x12, 0xb7,
                 0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f, 0x32, 0x62 } },
        {   8, { 0x4e, 0xed, 0x71, 0x41, 0xea, 0x4a, 0x5c, 0xd4,
                 0xb7, 0x88, 0x60, 0x6b, 0xd2, 0x3f, 0x46, 0xe2,
                 0x12, 0xaf, 0x9c, 0xac, 0xeb, 0xac, 0xdc, 0x7d,
                 0x1f, 0x4c, 0x6d, 0xc7, 0xf2, 0x51, 0x1b, 0x98 } },
        {   9, { 0x02, 0x8e, 0xb9, 0x7d, 0x80, 0x29, 0x1f, 0xc1,
                 0xf4, 0xab, 0x84, 0x66, 0x57, 0xfb, 0x22, 0x77,
                 0xca, 0xe9, 0xd7, 0xed, 0xa6, 0x39, 0xc0, 0x9b,
                 0xd2, 0x20, 0xa9, 0xc8, 0x69, 0xf0, 0xe9, 0xe6 } },
        {  18, { 0xc3, 0x16, 0x9e, 0xba, 0x2c, 0xa8, 0x0e, 0x6e,
                 0x7c, 0x90, 0x6e, 0xbb, 0x7f, 0xec, 0xc2, 0x09,
                 0x44, 0xc1, 0x2f, 0x2a, 0x01, 0x87, 0x42, 0x37,
                 0xe6, 0x36, 0x32, 0x20, 0xf4, 0xb9, 0x31, 0xce } },
        {  24, { 0x4a, 0xbc, 0x5d, 0x23, 0x28, 0xfb, 0x4a, 0xcc,
                 0x54, 0x9a, 0xff, 0x4b, 0x87, 0x7d, 0xf0, 0x0b,
                 0xa5, 0x2d, 0x46, 0x97, 0x57, 0x74, 0x9d, 0x6b,
                 0x8c, 0x33, 0xd4, 0x58, 0x70, 0xb8, 0xfc, 0xff } },
        {  36, { 0xfb, 0x26, 0xba, 0xa3, 0x11, 0x35, 0x05, 0x24,
                 0x9c, 0x56, 0x65, 0xe6, 0x53, 0xd0, 0xd8, 0x33,
                 0xbe, 0x72, 0x78, 0x8d, 0x87, 0x8c, 0xa7, 0x76,
                 0x25, 0xf1, 0x49, 0xa3, 0xd1, 0xb3, 0xcd, 0xe4 } },
        {  56, { 0x91, 0xcb, 0x45, 0x34, 0x6a, 0xd5, 0xdb, 0xc6,
                 0xae, 0xfe, 0x89, 0x87, 0x53, 0x5e, 0x19, 0xf7,
                 0x11, 0x54, 0x12, 0x61, 0xf7, 0x82, 0x00, 0x17,
                 0x2b, 0x04, 0x37, 0x35, 0x16, 0x46, 0x48, 0xc3 } },
        { 128, { 0x88, 0x21, 0x79, 0xb8, 0xdb, 0xcc, 0xd2, 0x85,
                 0xcd, 0xa2, 0x41, 0xd9, 0x68, 0xcf, 0xcc, 0xcb,
                 0x31, 0x56, 0xc5, 0xed, 0xac, 0x2f, 0xa3, 0x76,
                 0x1b, 0xb6, 0xed, 0xa7, 0xff, 0x8c, 0xb1, 0x72 } },
        // --- multi-chunk (n_u64 > 128): exercises the CV-stack tree merge ---
        {  129, { 0x40, 0x4d, 0x05, 0x42, 0xfe, 0xb6, 0x66, 0x6c,   // 2 chunks
                  0x46, 0x2f, 0x46, 0x0d, 0x0e, 0x09, 0x5d, 0x66,
                  0x2a, 0xec, 0x32, 0xc8, 0x40, 0xa0, 0xa6, 0x65,
                  0x12, 0x47, 0x0f, 0x8b, 0xb6, 0xd0, 0x49, 0x77 } },
        {  200, { 0x53, 0x2d, 0x7e, 0x65, 0xb6, 0x8b, 0xf7, 0xd4,   // 2 chunks (partial 2nd)
                  0xeb, 0x26, 0x07, 0x08, 0x14, 0x19, 0x5f, 0x15,
                  0x5f, 0xc4, 0x14, 0x60, 0x55, 0x74, 0x66, 0xbb,
                  0x96, 0xce, 0xfc, 0xe9, 0xbf, 0x06, 0x61, 0x31 } },
        {  256, { 0x1b, 0xdc, 0xcf, 0xde, 0x02, 0x10, 0xa8, 0xca,   // 2 chunks (exact)
                  0x17, 0x8b, 0xe1, 0x9c, 0x67, 0x77, 0xcd, 0xb4,
                  0xb9, 0xa8, 0xfd, 0x24, 0xe7, 0xfe, 0x2b, 0x6b,
                  0x25, 0x9b, 0x98, 0xe7, 0xaa, 0xaa, 0x0b, 0xb6 } },
        {  384, { 0xa1, 0x09, 0x98, 0xbe, 0xb5, 0x19, 0x3c, 0x47,   // 3 chunks (odd tree)
                  0xa0, 0xc1, 0xcf, 0x19, 0xaa, 0x8d, 0xaa, 0xa8,
                  0xde, 0xde, 0x3d, 0x9e, 0x5c, 0x53, 0xf7, 0x8e,
                  0xcf, 0xe5, 0xd2, 0x2e, 0x20, 0xf7, 0xf9, 0xbd } },
        {  512, { 0x0b, 0x3d, 0xda, 0x6f, 0xbf, 0xe0, 0x1c, 0x93,   // 4 chunks
                  0xd7, 0x93, 0x88, 0x63, 0x2f, 0x66, 0xc5, 0xc1,
                  0xfa, 0x78, 0x13, 0x82, 0x8c, 0xa8, 0xf6, 0x2e,
                  0xf8, 0x63, 0x04, 0xee, 0x31, 0x03, 0x68, 0x97 } },
        {  879, { 0x5c, 0x1d, 0x91, 0xb1, 0x19, 0xf0, 0x8a, 0x4a,   // 7 chunks (Keccakf cm2)
                  0xfa, 0x65, 0x4c, 0xdc, 0xc1, 0xb8, 0xa7, 0x21,
                  0xfa, 0x32, 0x7c, 0x6d, 0x55, 0x5c, 0x7f, 0xf6,
                  0x78, 0x1a, 0xe7, 0x38, 0x3d, 0xaa, 0x7d, 0x19 } },
        { 1024, { 0x4c, 0x5a, 0x52, 0x32, 0xab, 0x61, 0x63, 0x30,   // 8 chunks (balanced)
                  0x92, 0x32, 0x4d, 0x9e, 0x26, 0x01, 0xae, 0x66,
                  0x43, 0xa9, 0xa6, 0x05, 0x1e, 0xf7, 0x7b, 0xce,
                  0x5d, 0x38, 0xc2, 0x9f, 0x43, 0x8f, 0x7f, 0x54 } },
        { 2137, { 0xc6, 0xa8, 0x87, 0x23, 0x9c, 0x7f, 0x87, 0x0a,   // 17 chunks (Keccakf cm1)
                  0xac, 0x46, 0x0c, 0x78, 0xc7, 0x97, 0x0b, 0xc9,
                  0x5c, 0xa1, 0x2e, 0xf0, 0x72, 0x2c, 0x84, 0x82,
                  0xde, 0x56, 0xd8, 0x63, 0x7c, 0xcc, 0xe5, 0x36 } },
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

        blake3gpu::blake3_linear_hash(d_out, d_in, c.n_u64, 1, stream);
        CUDA_OK(cudaStreamSynchronize(stream));

        uint64_t hout[4];
        CUDA_OK(cudaMemcpy(hout, d_out, sizeof(hout), cudaMemcpyDeviceToHost));

        if (std::memcmp(hout, c.expect, 32) != 0)
        {
            const uint8_t *got = (const uint8_t *)hout;
            fprintf(stderr, "[BLAKE3 self-check] FAIL on %u u64 (%u bytes)\n  expected ",
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
            "[BLAKE3 self-check] passed: %zu lengths (single- and multi-chunk, "
            "up to 2137 u64 = 17 chunks) match official blake3 crate vectors\n",
            sizeof(cases) / sizeof(cases[0]));
}

// ---------------------------------------------------------------------------
// Trace init (mirrors initTraceKernel_pos1 in bench_poseidon_gpu.cu).
// ---------------------------------------------------------------------------
static __global__ void initTraceKernel_blake3(uint64_t *d_trace, uint64_t nRows, uint64_t nCols)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows)
        for (uint64_t j = 0; j < nCols; ++j)
            d_trace[idx * nCols + j] = idx + j;
}

// ---------------------------------------------------------------------------
// linearHash -- one thread per row, RowMajor input (matches the Poseidon
// LINEAR_HASH_*_ROWMAJOR benches).
// ---------------------------------------------------------------------------
static void LINEAR_HASH_ROWMAJOR_GPU_BLAKE3_BENCH(benchmark::State &state)
{
    static const bool checked = (blake3_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t *d_trace, *d_hash;
    CUDA_OK(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_hash,  BENCH_NROWS * 4 * sizeof(uint64_t)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel_blake3<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    // Warm up
    blake3gpu::blake3_linear_hash(d_hash, d_trace, nCols, BENCH_NROWS, stream);
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        blake3gpu::blake3_linear_hash(d_hash, d_trace, nCols, BENCH_NROWS, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_hash));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(LINEAR_HASH_ROWMAJOR_GPU_BLAKE3_BENCH)
    ->Name("LINEAR_HASH_ROWMAJOR_GPU_BLAKE3_BENCH")
    ->Unit(benchmark::kMillisecond)
    // Distinct per-stage column counts (cm1/cm2/cm3) from the ZisK Air table,
    // all <= 128 (single BLAKE3 chunk).
    ->Arg(1)->Arg(3)->Arg(6)->Arg(8)->Arg(9)->Arg(10)->Arg(12)->Arg(13)
    ->Arg(14)->Arg(15)->Arg(16)->Arg(18)->Arg(21)->Arg(24)->Arg(29)->Arg(32)
    ->Arg(34)->Arg(35)->Arg(36)->Arg(38)->Arg(39)->Arg(40)->Arg(43)->Arg(44)
    ->Arg(45)->Arg(66)
    ->UseRealTime();

// ---------------------------------------------------------------------------
// Wide (multi-chunk) linear hash. num_cols > 128 dispatches to the chunk-tree
// kernel. Uses a smaller row count so the 2137-col trace (~9 GB) fits in GPU
// memory; throughput/row is N-independent in the saturated regime.
// ---------------------------------------------------------------------------
static constexpr uint64_t BENCH_NROWS_WIDE = 1ULL << 19;  // 512K rows

static void LINEAR_HASH_ROWMAJOR_WIDE_GPU_BLAKE3_BENCH(benchmark::State &state)
{
    static const bool checked = (blake3_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t *d_trace, *d_hash;
    CUDA_OK(cudaMalloc((void **)&d_trace, BENCH_NROWS_WIDE * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_hash,  BENCH_NROWS_WIDE * 4 * sizeof(uint64_t)));

    dim3 thr(128), blk((BENCH_NROWS_WIDE + 127) / 128);
    initTraceKernel_blake3<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS_WIDE, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    blake3gpu::blake3_linear_hash(d_hash, d_trace, nCols, BENCH_NROWS_WIDE, stream);  // warm up
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        blake3gpu::blake3_linear_hash(d_hash, d_trace, nCols, BENCH_NROWS_WIDE, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_hash));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(LINEAR_HASH_ROWMAJOR_WIDE_GPU_BLAKE3_BENCH)
    ->Name("LINEAR_HASH_ROWMAJOR_WIDE_GPU_BLAKE3_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Arg(129)->Arg(256)->Arg(512)->Arg(879)->Arg(2137)  // 879/2137 = Keccakf cm2/cm1
    ->UseRealTime();

// ===========================================================================
// Full merklization: leaf linear hash + arity-4 tree reduction (matches the
// Poseidon W=16 arity-4 MERKLETREE benches). The tree-reduction part is
// column-count-independent (it always compresses 4-element digests).
// ===========================================================================
static constexpr uint32_t MT_ARITY = 4;

static void MERKLETREE_AR4_ROWMAJOR_GPU_BLAKE3_BENCH(benchmark::State &state)
{
    static const bool checked = (blake3_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t tree_size = getTreeNumElements(BENCH_NROWS, MT_ARITY);
    uint64_t *d_trace, *d_tree;
    CUDA_OK(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_tree,  tree_size * sizeof(uint64_t)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel_blake3<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    blake3gpu::blake3_merkletree(MT_ARITY, d_tree, d_trace, nCols, BENCH_NROWS, stream);  // warm
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        blake3gpu::blake3_merkletree(MT_ARITY, d_tree, d_trace, nCols, BENCH_NROWS, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_tree));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(MERKLETREE_AR4_ROWMAJOR_GPU_BLAKE3_BENCH)
    ->Name("MERKLETREE_AR4_ROWMAJOR_GPU_BLAKE3_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)->Arg(3)->Arg(6)->Arg(8)->Arg(9)->Arg(10)->Arg(12)->Arg(13)
    ->Arg(14)->Arg(15)->Arg(16)->Arg(18)->Arg(21)->Arg(24)->Arg(29)->Arg(32)
    ->Arg(34)->Arg(35)->Arg(36)->Arg(38)->Arg(39)->Arg(40)->Arg(43)->Arg(44)
    ->Arg(45)->Arg(66)
    ->UseRealTime();

// Wide (multi-chunk leaves) merklization, smaller row count to fit in memory.
static void MERKLETREE_AR4_WIDE_GPU_BLAKE3_BENCH(benchmark::State &state)
{
    static const bool checked = (blake3_self_check(), true);
    (void)checked;

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t tree_size = getTreeNumElements(BENCH_NROWS_WIDE, MT_ARITY);
    uint64_t *d_trace, *d_tree;
    CUDA_OK(cudaMalloc((void **)&d_trace, BENCH_NROWS_WIDE * nCols * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_tree,  tree_size * sizeof(uint64_t)));

    dim3 thr(128), blk((BENCH_NROWS_WIDE + 127) / 128);
    initTraceKernel_blake3<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS_WIDE, nCols);
    CUDA_OK(cudaStreamSynchronize(stream));

    blake3gpu::blake3_merkletree(MT_ARITY, d_tree, d_trace, nCols, BENCH_NROWS_WIDE, stream);
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        blake3gpu::blake3_merkletree(MT_ARITY, d_tree, d_trace, nCols, BENCH_NROWS_WIDE, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_trace));
    CUDA_OK(cudaFree(d_tree));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(MERKLETREE_AR4_WIDE_GPU_BLAKE3_BENCH)
    ->Name("MERKLETREE_AR4_WIDE_GPU_BLAKE3_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Arg(879)->Arg(2137)
    ->UseRealTime();

// Pure tree reduction (no leaf hash) over N pre-filled leaf digests -- isolates
// the column-count-independent per-leaf tree cost.
static void MERKLE_REDUCE_AR4_GPU_BLAKE3_BENCH(benchmark::State &state)
{
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    uint64_t N = state.range(0);
    uint64_t tree_size = getTreeNumElements(N, MT_ARITY);
    uint64_t *d_tree;
    CUDA_OK(cudaMalloc((void **)&d_tree, tree_size * sizeof(uint64_t)));

    dim3 thr(128), blk((N + 127) / 128);
    initTraceKernel_blake3<<<blk, thr, 0, stream>>>(d_tree, N, 4);  // fill N leaf digests
    CUDA_OK(cudaStreamSynchronize(stream));

    blake3gpu::blake3_merkletree_reduce(MT_ARITY, d_tree, N, stream);
    CUDA_OK(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        blake3gpu::blake3_merkletree_reduce(MT_ARITY, d_tree, N, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    CUDA_OK(cudaFree(d_tree));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(MERKLE_REDUCE_AR4_GPU_BLAKE3_BENCH)
    ->Name("MERKLE_REDUCE_AR4_GPU_BLAKE3_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Arg(1ULL << 23)
    ->UseRealTime();

// grinding -- GPU, timed on the production launcher (n_bits, not nCols) so it is comparable to
// GRINDING_GPU_POS1_BENCH. The host re-verifies each nonce, so a fast-but-wrong kernel cannot win.
static void GRINDING_GPU_BLAKE3_BENCH(benchmark::State &state)
{
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    const uint32_t n_bits = (uint32_t)state.range(0);

    // The kernel reads only in[0..2] (the FIELD_EXTENSION challenge) and writes
    // the nonce itself into slot 3.
    constexpr int GRIND_CHALLENGE_ELEMS = 3;

    uint64_t *d_in, *d_nonce, *d_nonceBlock;
    CUDA_OK(cudaMalloc((void **)&d_in,         4 * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_nonce,      sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void **)&d_nonceBlock, BLAKE3_GRIND_GRID * sizeof(uint64_t)));

    uint64_t h_in[GRIND_CHALLENGE_ELEMS];
    uint64_t iteration = 0;

    for (auto _ : state)
    {
        iteration++;
        for (int i = 0; i < GRIND_CHALLENGE_ELEMS; ++i)
            h_in[i] = (iteration * 1000 + i) * 123456789ULL;
        CUDA_OK(cudaMemcpy(d_in, h_in, GRIND_CHALLENGE_ELEMS * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));

        Blake3GoldilocksGPU::grinding(d_nonce, d_nonceBlock, d_in, n_bits, stream);
        CUDA_OK(cudaStreamSynchronize(stream));

        uint64_t h_nonce;
        CUDA_OK(cudaMemcpy(&h_nonce, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (h_nonce == UINT64_MAX)
        {
            state.SkipWithError("BLAKE3 grinding found no nonce");
            break;
        }

        const uint64_t v_in[8] = {h_in[0], h_in[1], h_in[2], h_nonce, 0, 0, 0, 0};
        uint64_t v_out[8];
        blake3core::permute8(v_in, v_out);
        if (v_out[0] >= (1ULL << (64 - n_bits)))
        {
            state.SkipWithError("BLAKE3 grinding nonce does not satisfy the bit requirement");
            break;
        }
    }

    CUDA_OK(cudaFree(d_in));
    CUDA_OK(cudaFree(d_nonce));
    CUDA_OK(cudaFree(d_nonceBlock));
    CUDA_OK(cudaStreamDestroy(stream));
}

BENCHMARK(GRINDING_GPU_BLAKE3_BENCH)
    ->Name("GRINDING_GPU_BLAKE3_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Arg(16)->Arg(20)->Arg(23)->Arg(24)->Arg(25)->Arg(26)->Arg(27)->Arg(28)
    ->UseRealTime();
