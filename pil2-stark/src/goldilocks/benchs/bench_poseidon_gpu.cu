// ---------------------------------------------------------------------------
// bench_poseidon_gpu.cu -- GPU Poseidon v1 benchmarks.
//
// Mirrors bench_poseidon2_gpu.cu (same shapes, same unified parameters) so
// side-by-side comparison with Poseidon2 is straightforward.
//
// Parameters:
//   BENCH_NROWS = 1 << 23  (8M rows)
//   BENCH_NCOLS = {24, 36, 56}  (via ->Arg())
// Naming: OPERATION_DETAIL_PLATFORM_BENCH. All benches use ->UseRealTime().
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cstdint>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/goldilocks_tooling.hpp"
#include "../src/poseidon_goldilocks.cuh"
#include "../src/poseidon2_goldilocks.cuh"      // Layout enum
#include "../src/goldilocks_tooling.cuh"
#include "../utils/cuda_utils.hpp"

static constexpr uint64_t BENCH_NROWS = 1ULL << 23;    // 8M rows

// ---------------------------------------------------------------------------
// GPU trace-initialization kernel (mirrors bench_poseidon2_gpu.cu).
// ---------------------------------------------------------------------------
static __global__ void initTraceKernel_pos1(gl64_t *d_trace, uint64_t nRows, uint64_t nCols)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows)
        for (uint64_t j = 0; j < nCols; ++j)
            d_trace[idx * nCols + j] = uint64_t(idx + j);
}

// ===================================================================
// permute — single-sponge public API (<<<1,1>>> kernel). Mirrors the
// CPU PERMUTE_W12_*_CPU_BENCH pattern: one permute per iteration, with
// per-iteration stream sync.
// ===================================================================

template<uint32_t W>
static void PERMUTE_W_GPU_POS1_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_in, *d_out;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,  PoseidonGoldilocksGPU<W>::SPONGE_WIDTH * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_out, PoseidonGoldilocksGPU<W>::SPONGE_WIDTH * sizeof(gl64_t)));

    Goldilocks::Element h_in[PoseidonGoldilocksGPU<W>::SPONGE_WIDTH];
    for (uint32_t i = 0; i < PoseidonGoldilocksGPU<W>::SPONGE_WIDTH; ++i)
        h_in[i] = Goldilocks::fromU64(i + 1);
    CHECKCUDAERR(cudaMemcpy(d_in, h_in,
                            PoseidonGoldilocksGPU<W>::SPONGE_WIDTH * sizeof(gl64_t),
                            cudaMemcpyHostToDevice));

    // Warm up
    PoseidonGoldilocksGPU<W>::permute((uint64_t *)d_out, (uint64_t *)d_in, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        PoseidonGoldilocksGPU<W>::permute((uint64_t *)d_out, (uint64_t *)d_in, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_in));
    CHECKCUDAERR(cudaFree(d_out));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===================================================================
// linearHash — parameterised by nCols, two layouts.
// ===================================================================

template<uint32_t W>
static void LINEAR_HASH_W_TILES_GPU_POS1_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    gl64_t *d_trace, *d_hash;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_hash,  BENCH_NROWS * PoseidonGoldilocksGPU<W>::CAPACITY * sizeof(gl64_t)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel_pos1<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        PoseidonGoldilocksGPU<W>::linearHash(
            (uint64_t *)d_hash, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::ColMajor, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_hash));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

template<uint32_t W>
static void LINEAR_HASH_W_ROWMAJOR_GPU_POS1_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    gl64_t *d_trace, *d_hash;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_hash,  BENCH_NROWS * PoseidonGoldilocksGPU<W>::CAPACITY * sizeof(gl64_t)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel_pos1<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        PoseidonGoldilocksGPU<W>::linearHash(
            (uint64_t *)d_hash, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::RowMajor, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_hash));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===================================================================
// merkletree — parameterised by nCols, two layouts.
// ===================================================================

template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_TILES_GPU_POS1_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t tree_size = getTreeNumElements(BENCH_NROWS, ARITY);

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel_pos1<<<blk, thr, 0, stream>>>((gl64_t *)d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    PoseidonGoldilocksGPU<W>::merkletree(
        ARITY, (uint64_t *)d_tree, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::ColMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        PoseidonGoldilocksGPU<W>::merkletree(
            ARITY, (uint64_t *)d_tree, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::ColMajor, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_ROWMAJOR_GPU_POS1_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);
    uint64_t tree_size = getTreeNumElements(BENCH_NROWS, ARITY);

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel_pos1<<<blk, thr, 0, stream>>>((gl64_t *)d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    PoseidonGoldilocksGPU<W>::merkletree(
        ARITY, (uint64_t *)d_tree, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::RowMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        PoseidonGoldilocksGPU<W>::merkletree(
            ARITY, (uint64_t *)d_tree, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::RowMajor, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}


static void MERKLETREE_HYBRID_W12POS1_W16POS2_AR4_TILES_GPU_BENCH(benchmark::State &state)
{
    constexpr uint32_t W_LEAF = 12;   // Poseidon v1 sponge width
    constexpr uint32_t W_NODE = 16;   // Poseidon2 sponge width (arity 4 * CAPACITY 4)
    constexpr uint32_t ARITY  = 4;
    constexpr uint32_t CAP    = PoseidonGoldilocksGPU<W_LEAF>::CAPACITY;  // 4

    uint32_t gpu_id = 0;
    cudaGetDevice((int *)&gpu_id);
    PoseidonGoldilocksGPU<W_LEAF>::initConstants(&gpu_id, 1);
    Poseidon2GoldilocksGPU<W_NODE>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);

    gl64_t *d_trace, *d_leaves, *d_root;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace,  BENCH_NROWS * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_leaves, BENCH_NROWS * CAP  * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_root,   CAP * sizeof(gl64_t)));

    dim3 thr(128), blk((BENCH_NROWS + 127) / 128);
    initTraceKernel_pos1<<<blk, thr, 0, stream>>>(d_trace, BENCH_NROWS, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    auto build = [&]() {
        // Leaf layer: Poseidon v1 linear hash over the trace rows.
        PoseidonGoldilocksGPU<W_LEAF>::linearHash(
            (uint64_t *)d_leaves, (uint64_t *)d_trace, nCols, BENCH_NROWS, Layout::ColMajor, stream);
        // Internal nodes: Poseidon2 arity-4 reduction over the leaf digests.
        Poseidon2GoldilocksGPU<W_NODE>::merkletreeReduce(
            (uint64_t *)d_root, (uint64_t *)d_leaves, BENCH_NROWS, ARITY, stream);
    };

    // Warm up
    build();
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        build();
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_leaves));
    CHECKCUDAERR(cudaFree(d_root));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===================================================================
// grinding — GPU (not parameterised by nCols; parameterised by n_bits).
// ===================================================================

static void GRINDING_GPU_POS1_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    PoseidonGoldilocksGPUGrinding::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint32_t n_bits = state.range(0);

    // The kernel reads only in[0..2] (the FIELD_EXTENSION challenge) and writes
    // the nonce itself into slot 3.
    constexpr int GRIND_CHALLENGE_ELEMS = 3;

    gl64_t *d_in, *d_nonce, *d_nonceBlock;
    CHECKCUDAERR(cudaMalloc((void **)&d_in,          PoseidonGoldilocksGPUGrinding::SPONGE_WIDTH * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonce,       sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonceBlock,  POSEIDON1_GRIND_GRID * sizeof(gl64_t)));

    Goldilocks::Element h_in[GRIND_CHALLENGE_ELEMS];
    uint64_t iteration = 0;

    for (auto _ : state) {
        iteration++;
        for (int i = 0; i < GRIND_CHALLENGE_ELEMS; ++i)
            h_in[i] = Goldilocks::fromU64((iteration * 1000 + i) * 123456789ULL);
        CHECKCUDAERR(cudaMemcpy(d_in, h_in,
                                GRIND_CHALLENGE_ELEMS * sizeof(gl64_t),
                                cudaMemcpyHostToDevice));

        PoseidonGoldilocksGPUGrinding::grinding(
            (uint64_t *)d_nonce, (uint64_t *)d_nonceBlock, (uint64_t *)d_in, n_bits, stream);
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

#define REG_NCOLS(FUNC, W, LABEL)                                           \
    BENCHMARK_TEMPLATE(FUNC, W)                                             \
        ->Name(LABEL)                                                            \
        ->Unit(benchmark::kMillisecond)                                          \
        NCOLS_ARGS                                                               \
        ->UseRealTime();

#define REG_NCOLS_AR(FUNC, W, AR, LABEL)                                    \
    BENCHMARK_TEMPLATE(FUNC, W, AR)                                         \
        ->Name(LABEL)                                                            \
        ->Unit(benchmark::kMillisecond)                                          \
        NCOLS_ARGS                                                               \
        ->UseRealTime();

// permute — W=12 (Poseidon v1 production single-sponge AVX path).
BENCHMARK_TEMPLATE(PERMUTE_W_GPU_POS1_BENCH, 12)
    ->Name("PERMUTE_W12_GPU_POS1_BENCH")
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// linearHash — W=12, both layouts.
REG_NCOLS(LINEAR_HASH_W_TILES_GPU_POS1_BENCH,    12, "LINEAR_HASH_W12_TILES_GPU_POS1_BENCH")
REG_NCOLS(LINEAR_HASH_W_ROWMAJOR_GPU_POS1_BENCH, 12, "LINEAR_HASH_W12_ROWMAJOR_GPU_POS1_BENCH")
// linearHash — W=16 (rate 12): the production STARK commit sponge.
REG_NCOLS(LINEAR_HASH_W_TILES_GPU_POS1_BENCH,    16, "LINEAR_HASH_W16_TILES_GPU_POS1_BENCH")

// merkletree — Poseidon1, (W,arity) in {(8,2),(12,3),(16,4)}; both layouts.
REG_NCOLS_AR(MERKLETREE_W_AR_TILES_GPU_POS1_BENCH,    12, 2, "MERKLETREE_W12_AR2_TILES_GPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_TILES_GPU_POS1_BENCH,    12, 3, "MERKLETREE_W12_AR3_TILES_GPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_TILES_GPU_POS1_BENCH,    16, 4, "MERKLETREE_W16_AR4_TILES_GPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_ROWMAJOR_GPU_POS1_BENCH, 12, 2, "MERKLETREE_W12_AR2_ROWMAJOR_GPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_ROWMAJOR_GPU_POS1_BENCH, 12, 3, "MERKLETREE_W12_AR3_ROWMAJOR_GPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_ROWMAJOR_GPU_POS1_BENCH, 16, 4, "MERKLETREE_W16_AR4_ROWMAJOR_GPU_POS1_BENCH")

// HYBRID merkletree — v1 (W=12) leaves + Poseidon2 (W=16, arity 4) tree.
BENCHMARK(MERKLETREE_HYBRID_W12POS1_W16POS2_AR4_TILES_GPU_BENCH)
    ->Name("MERKLETREE_HYBRID_W12POS1_W16POS2_AR4_TILES_GPU_BENCH")
    ->Unit(benchmark::kMillisecond)
    ->Arg(9)->Arg(18)->Arg(24)->Arg(36)->Arg(56)
    ->UseRealTime();

// grinding
BENCHMARK(GRINDING_GPU_POS1_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->Arg(16)->Arg(20)->Arg(23)->Arg(24)->Arg(25)->Arg(26)->Arg(27)->Arg(28)
    ->UseRealTime();

#undef REG_NCOLS
#undef REG_NCOLS_AR
#undef NCOLS_ARGS
