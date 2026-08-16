// ---------------------------------------------------------------------------
// bench_poseidon2_gpu.cu -- GPU Poseidon2 benchmarks
//
// Unified parameters for apples-to-apples comparison with CPU benchmarks:
//   BENCH_NROWS     = 1 << 23  (8M rows)
//   BENCH_NCOLS     = {24, 36, 56}  (via ->Arg())
//
// Naming: OPERATION_DETAIL_PLATFORM_BENCH
// All benches use ->UseRealTime().
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cstdint>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/goldilocks_tooling.hpp"
#include "../src/poseidon2_goldilocks.cuh"
#include "../src/goldilocks_tooling.cuh"
#include "../utils/cuda_utils.hpp"

// ---------------------------------------------------------------------------
// Unified parameters
// ---------------------------------------------------------------------------
static constexpr uint64_t BENCH_NROWS = 1ULL << 23;   // 8M rows

// ---------------------------------------------------------------------------
// GPU trace initialization kernel
// ---------------------------------------------------------------------------
static __global__ void initTraceKernel(gl64_t *d_trace, uint64_t nRows, uint64_t nCols)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows)
        for (uint64_t j = 0; j < nCols; j++)
            d_trace[idx * nCols + j] = uint64_t(idx + j);
}

// ===================================================================
// permute — single-sponge public API (<<<1,1>>> kernel). Mirrors the
// CPU PERMUTE_W_*_CPU_BENCH pattern: one permute per iteration, with
// per-iteration stream sync.
// ===================================================================

template<uint32_t W>
static void PERMUTE_W_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    Poseidon2GoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  Poseidon2GoldilocksGPU<W>::SPONGE_WIDTH * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, Poseidon2GoldilocksGPU<W>::SPONGE_WIDTH * sizeof(gl64_t)));

    Goldilocks::Element h_in[Poseidon2GoldilocksGPU<W>::SPONGE_WIDTH];
    for (uint32_t i = 0; i < Poseidon2GoldilocksGPU<W>::SPONGE_WIDTH; i++)
        h_in[i] = Goldilocks::fromU64(i + 1);
    CHECKCUDAERR(cudaMemcpy(d_in, h_in,
                            Poseidon2GoldilocksGPU<W>::SPONGE_WIDTH * sizeof(gl64_t),
                            cudaMemcpyHostToDevice));

    // Warm up
    Poseidon2GoldilocksGPU<W>::permute((uint64_t *)d_out, (uint64_t *)d_in, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        Poseidon2GoldilocksGPU<W>::permute((uint64_t *)d_out, (uint64_t *)d_in, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_in));
    CHECKCUDAERR(cudaFree(d_out));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===================================================================
// linearHash -- parameterized by nCols, one bench per (W, layout)
// ===================================================================

template<uint32_t W>
static void LINEAR_HASH_W_TILES_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    Poseidon2GoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    gl64_t *d_trace, *d_hash;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_hash,  BENCH_NROWS * Poseidon2GoldilocksGPU<W>::CAPACITY * sizeof(gl64_t)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        Poseidon2GoldilocksGPU<W>::linearHash((uint64_t *)d_hash, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::ColMajor, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_hash));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

template<uint32_t W>
static void LINEAR_HASH_W_ROWMAJOR_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    Poseidon2GoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    gl64_t *d_trace, *d_hash;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_hash,  BENCH_NROWS * Poseidon2GoldilocksGPU<W>::CAPACITY * sizeof(gl64_t)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        Poseidon2GoldilocksGPU<W>::linearHash((uint64_t *)d_hash, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::RowMajor, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_hash));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===================================================================
// merkletree -- parameterized by nCols, one bench per (W/arity, layout)
// ===================================================================

template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_TILES_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    Poseidon2GoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t tree_size = getTreeNumElements(BENCH_NROWS, ARITY);

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>((gl64_t *)d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    Poseidon2GoldilocksGPU<W>::merkletree(ARITY, (uint64_t *)d_tree, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::ColMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        Poseidon2GoldilocksGPU<W>::merkletree(ARITY, (uint64_t *)d_tree, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::ColMajor, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_ROWMAJOR_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    Poseidon2GoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t tree_size = getTreeNumElements(BENCH_NROWS, ARITY);

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>((gl64_t *)d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    Poseidon2GoldilocksGPU<W>::merkletree(ARITY, (uint64_t *)d_tree, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::RowMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        Poseidon2GoldilocksGPU<W>::merkletree(ARITY, (uint64_t *)d_tree, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::RowMajor, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===================================================================
// grinding -- GPU (not parameterized by nCols)
// ===================================================================

static void GRINDING_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    Poseidon2GoldilocksGPUGrinding::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint32_t n_bits = state.range(0);

    gl64_t *d_in, *d_nonce, *d_nonceBlock;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,         4 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonce,       sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonceBlock, NONCES_LAUNCH_GRID_SIZE * sizeof(gl64_t)));

    Goldilocks::Element h_in[Poseidon2GoldilocksGPUGrinding::SPONGE_WIDTH];
    uint64_t iteration = 0;

    for (auto _ : state) {
        iteration++;
        for (int i = 0; i < (Poseidon2GoldilocksGPUGrinding::SPONGE_WIDTH - 1); i++)
            h_in[i] = Goldilocks::fromU64((iteration * 1000 + i) * 123456789ULL);
        CHECKCUDAERR(cudaMemcpy(d_in, h_in, (Poseidon2GoldilocksGPUGrinding::SPONGE_WIDTH - 1) * sizeof(gl64_t), cudaMemcpyHostToDevice));

        Poseidon2GoldilocksGPUGrinding::grinding((uint64_t *)d_nonce, (uint64_t *)d_nonceBlock, (uint64_t *)d_in, n_bits, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));

        uint64_t h_nonce;
        CHECKCUDAERR(cudaMemcpy(&h_nonce, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        assert(h_nonce != UINT64_MAX);
        iteration++;
    }

    CHECKCUDAERR(cudaFree(d_in));
    CHECKCUDAERR(cudaFree(d_nonce));
    CHECKCUDAERR(cudaFree(d_nonceBlock));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===================================================================
// Registration
// ===================================================================

#define NCOLS_ARGS ->Arg(9)->Arg(18)->Arg(24)->Arg(36)->Arg(56)

#define REG_NCOLS(FUNC, W, LABEL)                                       \
    BENCHMARK_TEMPLATE(FUNC, W)                                         \
        ->Name(LABEL)                                                        \
        ->Unit(benchmark::kMillisecond)                                      \
        NCOLS_ARGS                                                           \
        ->UseRealTime();

#define REG_NCOLS_AR(FUNC, W, AR, LABEL)                                \
    BENCHMARK_TEMPLATE(FUNC, W, AR)                                     \
        ->Name(LABEL)                                                        \
        ->Unit(benchmark::kMillisecond)                                      \
        NCOLS_ARGS                                                           \
        ->UseRealTime();

// ---------------------------------------------------------------------------
// permute — W in {8, 12, 16}
// ---------------------------------------------------------------------------

BENCHMARK_TEMPLATE(PERMUTE_W_GPU_BENCH, 8)
    ->Name("PERMUTE_W8_GPU_BENCH")
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(PERMUTE_W_GPU_BENCH, 12)
    ->Name("PERMUTE_W12_GPU_BENCH")
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(PERMUTE_W_GPU_BENCH, 16)
    ->Name("PERMUTE_W16_GPU_BENCH")
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// ---------------------------------------------------------------------------
// linearHash — W in {12, 16}, both ColMajor and RowMajor layouts
// ---------------------------------------------------------------------------

REG_NCOLS(LINEAR_HASH_W_TILES_GPU_BENCH,    12, "LINEAR_HASH_W12_TILES_GPU_BENCH")
REG_NCOLS(LINEAR_HASH_W_TILES_GPU_BENCH,    16, "LINEAR_HASH_W16_TILES_GPU_BENCH")
REG_NCOLS(LINEAR_HASH_W_ROWMAJOR_GPU_BENCH, 12, "LINEAR_HASH_W12_ROWMAJOR_GPU_BENCH")
REG_NCOLS(LINEAR_HASH_W_ROWMAJOR_GPU_BENCH, 16, "LINEAR_HASH_W16_ROWMAJOR_GPU_BENCH")

// ---------------------------------------------------------------------------
// merkletree — (W,arity) in {(12,3),(16,4)}, both layouts
// ---------------------------------------------------------------------------

REG_NCOLS_AR(MERKLETREE_W_AR_TILES_GPU_BENCH,    12, 3, "MERKLETREE_W12_AR3_TILES_GPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_TILES_GPU_BENCH,    16, 4, "MERKLETREE_W16_AR4_TILES_GPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_ROWMAJOR_GPU_BENCH, 12, 3, "MERKLETREE_W12_AR3_ROWMAJOR_GPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_ROWMAJOR_GPU_BENCH, 16, 4, "MERKLETREE_W16_AR4_ROWMAJOR_GPU_BENCH")

// ---------------------------------------------------------------------------
// grinding
// ---------------------------------------------------------------------------

BENCHMARK(GRINDING_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->Arg(16)->Arg(20)->Arg(23)->Arg(24)->Arg(25)
    ->UseRealTime();

#undef REG_NCOLS
#undef REG_NCOLS_AR
#undef NCOLS_ARGS
