// ---------------------------------------------------------------------------
// bench_poseidon2_cpu.cpp -- CPU Poseidon2 benchmarks
//
// Unified parameters for apples-to-apples comparison with GPU benchmarks:
//   BENCH_NROWS     = 1 << 23  (8M rows)
//   BENCH_NCOLS     = {24, 36, 56}  (via ->Arg())
//
// Naming: OPERATION_DETAIL_PLATFORM_BENCH
// All benches use max threads and ->UseRealTime().
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/goldilocks_tooling.hpp"
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "omp.h"

// ---------------------------------------------------------------------------
// Unified parameters
// ---------------------------------------------------------------------------
static constexpr uint64_t BENCH_NROWS = 1ULL << 23;   // 8M rows
static constexpr uint64_t NUM_HASHES  = 1ULL << 21;   // 2M hashes for per-element ops

// ---------------------------------------------------------------------------
// Helper: fill buffer with simple deterministic data
// ---------------------------------------------------------------------------
static void fillData(Goldilocks::Element *buf, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
        buf[i] = Goldilocks::fromU64(i + 1);
}

// ===================================================================
// permute / permuteTrunc -- per-element throughput (thread-count sweep)
// No GPU counterpart; kept for micro-level CPU profiling.
// ===================================================================

template<uint32_t W>
static void PERMUTE_W_SCALAR_CPU_BENCH(benchmark::State &state)
{
    uint64_t total = (uint64_t)NUM_HASHES * Poseidon2Goldilocks<W>::SPONGE_WIDTH;
    Goldilocks::Element *x = new Goldilocks::Element[total];
    Goldilocks::Element *r = new Goldilocks::Element[total];
    fillData(x, total);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
            Poseidon2Goldilocks<W>::permute(
                (Goldilocks::Element(&)[Poseidon2Goldilocks<W>::SPONGE_WIDTH])r[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                (const Goldilocks::Element(&)[Poseidon2Goldilocks<W>::SPONGE_WIDTH])x[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                Poseidon2Mode::Scalar);
    }
    state.SetItemsProcessed(state.iterations() * NUM_HASHES);
    state.counters["hashes"] = NUM_HASHES;
    delete[] x; delete[] r;
}

#ifdef __AVX2__
template<uint32_t W>
static void PERMUTE_W_AVX_CPU_BENCH(benchmark::State &state)
{
    uint64_t total = (uint64_t)NUM_HASHES * Poseidon2Goldilocks<W>::SPONGE_WIDTH;
    Goldilocks::Element *x = new Goldilocks::Element[total];
    Goldilocks::Element *r = new Goldilocks::Element[total];
    fillData(x, total);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
            Poseidon2Goldilocks<W>::permute(
                (Goldilocks::Element(&)[Poseidon2Goldilocks<W>::SPONGE_WIDTH])r[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                (const Goldilocks::Element(&)[Poseidon2Goldilocks<W>::SPONGE_WIDTH])x[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                Poseidon2Mode::Avx);
    }
    state.SetItemsProcessed(state.iterations() * NUM_HASHES);
    state.counters["hashes"] = NUM_HASHES;
    delete[] x; delete[] r;
}

#endif

template<uint32_t W>
static void PERMUTE_TRUNC_W_SCALAR_CPU_BENCH(benchmark::State &state)
{
    uint64_t in_total  = (uint64_t)NUM_HASHES * Poseidon2Goldilocks<W>::SPONGE_WIDTH;
    uint64_t out_total = (uint64_t)NUM_HASHES * Poseidon2Goldilocks<W>::CAPACITY;
    Goldilocks::Element *x = new Goldilocks::Element[in_total];
    Goldilocks::Element *r = new Goldilocks::Element[out_total];
    fillData(x, in_total);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
            Poseidon2Goldilocks<W>::permuteTrunc(
                (Goldilocks::Element(&)[Poseidon2Goldilocks<W>::CAPACITY])r[i * Poseidon2Goldilocks<W>::CAPACITY],
                (Goldilocks::Element(&)[Poseidon2Goldilocks<W>::SPONGE_WIDTH])x[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                Poseidon2Mode::Scalar);
    }
    delete[] x; delete[] r;
}

#ifdef __AVX2__
template<uint32_t W>
static void PERMUTE_TRUNC_W_AVX_CPU_BENCH(benchmark::State &state)
{
    uint64_t in_total  = (uint64_t)NUM_HASHES * Poseidon2Goldilocks<W>::SPONGE_WIDTH;
    uint64_t out_total = (uint64_t)NUM_HASHES * Poseidon2Goldilocks<W>::CAPACITY;
    Goldilocks::Element *x = new Goldilocks::Element[in_total];
    Goldilocks::Element *r = new Goldilocks::Element[out_total];
    fillData(x, in_total);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
            Poseidon2Goldilocks<W>::permuteTrunc(
                (Goldilocks::Element(&)[Poseidon2Goldilocks<W>::CAPACITY])r[i * Poseidon2Goldilocks<W>::CAPACITY],
                (Goldilocks::Element(&)[Poseidon2Goldilocks<W>::SPONGE_WIDTH])x[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                Poseidon2Mode::Avx);
    }
    delete[] x; delete[] r;
}
#endif

// ===================================================================
// linearHash -- parameterized by nCols via ->Arg()
// ===================================================================

template<uint32_t W>
static void LINEAR_HASH_W_SCALAR_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    Goldilocks::Element *cols   = new Goldilocks::Element[nCols * BENCH_NROWS];
    Goldilocks::Element *result = new Goldilocks::Element[HASH_SIZE * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < BENCH_NROWS; i++)
            Poseidon2Goldilocks<W>::linearHash(&result[i * HASH_SIZE], &cols[i * nCols], nCols, Poseidon2Mode::Scalar);
    }
    delete[] cols; delete[] result;
}

#ifdef __AVX2__
template<uint32_t W>
static void LINEAR_HASH_W_AVX_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    Goldilocks::Element *cols   = new Goldilocks::Element[nCols * BENCH_NROWS];
    Goldilocks::Element *result = new Goldilocks::Element[HASH_SIZE * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < BENCH_NROWS; i++)
            Poseidon2Goldilocks<W>::linearHash(&result[i * HASH_SIZE], &cols[i * nCols], nCols, Poseidon2Mode::Avx);
    }
    delete[] cols; delete[] result;
}
#endif

// ===================================================================
// merkletree -- parameterized by nCols via ->Arg()
// ===================================================================

template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_SCALAR_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);

    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];

    for (auto _ : state)
        Poseidon2Goldilocks<W>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, Poseidon2Mode::Scalar);

    delete[] cols; delete[] tree;
}

#ifdef __AVX2__
template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_AVX_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);

    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];

    for (auto _ : state)
        Poseidon2Goldilocks<W>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, Poseidon2Mode::Avx);

    delete[] cols; delete[] tree;
}

template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_AVXBATCH_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);

    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];

    for (auto _ : state)
        Poseidon2Goldilocks<W>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, Poseidon2Mode::AvxBatch);

    delete[] cols; delete[] tree;
}
#endif

#ifdef __AVX512__
template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_AVX512BATCH_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);

    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];

    for (auto _ : state)
        Poseidon2Goldilocks<W>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, Poseidon2Mode::Avx512Batch);

    delete[] cols; delete[] tree;
}
#endif

// ===================================================================
// grinding -- CPU (not parameterized by nCols)
// ===================================================================

static void GRINDING_CPU_BENCH(benchmark::State &state)
{
    uint32_t n_bits = state.range(0);
    uint64_t iteration = 0;

    for (auto _ : state) {
        iteration++;
        uint64_t in[3];
        for (int i = 0; i < 3; i++)
            in[i] = (iteration * 1000 + i) * 123456789ULL;

        uint64_t nonce = UINT64_MAX;
        Poseidon2Goldilocks<4>::grinding(nonce, in, n_bits);
        iteration++;
    }
}

// ===================================================================
// Registration macros
// ===================================================================

// nCols-parameterized benchmarks (linearHash, merkletree)
#define NCOLS_ARGS ->Arg(24)->Arg(36)->Arg(56)

#define REG_NCOLS(FUNC, W, LABEL)                                            \
    BENCHMARK_TEMPLATE(FUNC, W)                                              \
        ->Name(LABEL)                                                        \
        ->Unit(benchmark::kMillisecond)                                      \
        NCOLS_ARGS                                                           \
        ->UseRealTime();

#define REG_NCOLS_AR(FUNC, W, AR, LABEL)                                     \
    BENCHMARK_TEMPLATE(FUNC, W, AR)                                          \
        ->Name(LABEL)                                                        \
        ->Unit(benchmark::kMillisecond)                                      \
        NCOLS_ARGS                                                           \
        ->UseRealTime();

#define REG_ELEM(FUNC, W, LABEL)                                             \
    BENCHMARK_TEMPLATE(FUNC, W)                                              \
        ->Name(LABEL)                                                        \
        ->Unit(benchmark::kMillisecond)                                      \
        ->UseRealTime();

// ---------------------------------------------------------------------------
// permute / permuteTrunc registrations (per-element throughput, no nCols param)
// ---------------------------------------------------------------------------

REG_ELEM(PERMUTE_W_SCALAR_CPU_BENCH, 8,  "PERMUTE_W8_SCALAR_CPU_BENCH")
REG_ELEM(PERMUTE_W_SCALAR_CPU_BENCH, 12, "PERMUTE_W12_SCALAR_CPU_BENCH")
REG_ELEM(PERMUTE_W_SCALAR_CPU_BENCH, 16, "PERMUTE_W16_SCALAR_CPU_BENCH")

#ifdef __AVX2__
REG_ELEM(PERMUTE_W_AVX_CPU_BENCH, 8,  "PERMUTE_W8_AVX_CPU_BENCH")
REG_ELEM(PERMUTE_W_AVX_CPU_BENCH, 12, "PERMUTE_W12_AVX_CPU_BENCH")
REG_ELEM(PERMUTE_W_AVX_CPU_BENCH, 16, "PERMUTE_W16_AVX_CPU_BENCH")
#endif

REG_ELEM(PERMUTE_TRUNC_W_SCALAR_CPU_BENCH, 4,  "PERMUTE_TRUNC_W4_SCALAR_CPU_BENCH")
REG_ELEM(PERMUTE_TRUNC_W_SCALAR_CPU_BENCH, 8,  "PERMUTE_TRUNC_W8_SCALAR_CPU_BENCH")
REG_ELEM(PERMUTE_TRUNC_W_SCALAR_CPU_BENCH, 12, "PERMUTE_TRUNC_W12_SCALAR_CPU_BENCH")
REG_ELEM(PERMUTE_TRUNC_W_SCALAR_CPU_BENCH, 16, "PERMUTE_TRUNC_W16_SCALAR_CPU_BENCH")

#ifdef __AVX2__
REG_ELEM(PERMUTE_TRUNC_W_AVX_CPU_BENCH, 4,  "PERMUTE_TRUNC_W4_AVX_CPU_BENCH")
REG_ELEM(PERMUTE_TRUNC_W_AVX_CPU_BENCH, 8,  "PERMUTE_TRUNC_W8_AVX_CPU_BENCH")
REG_ELEM(PERMUTE_TRUNC_W_AVX_CPU_BENCH, 12, "PERMUTE_TRUNC_W12_AVX_CPU_BENCH")
REG_ELEM(PERMUTE_TRUNC_W_AVX_CPU_BENCH, 16, "PERMUTE_TRUNC_W16_AVX_CPU_BENCH")
#endif

// ---------------------------------------------------------------------------
// linearHash registrations -- W in {8,12,16}, modes: Scalar, Avx
// ---------------------------------------------------------------------------

REG_NCOLS(LINEAR_HASH_W_SCALAR_CPU_BENCH, 8,  "LINEAR_HASH_W8_SCALAR_CPU_BENCH")
REG_NCOLS(LINEAR_HASH_W_SCALAR_CPU_BENCH, 12, "LINEAR_HASH_W12_SCALAR_CPU_BENCH")
REG_NCOLS(LINEAR_HASH_W_SCALAR_CPU_BENCH, 16, "LINEAR_HASH_W16_SCALAR_CPU_BENCH")

#ifdef __AVX2__
REG_NCOLS(LINEAR_HASH_W_AVX_CPU_BENCH, 8,  "LINEAR_HASH_W8_AVX_CPU_BENCH")
REG_NCOLS(LINEAR_HASH_W_AVX_CPU_BENCH, 12, "LINEAR_HASH_W12_AVX_CPU_BENCH")
REG_NCOLS(LINEAR_HASH_W_AVX_CPU_BENCH, 16, "LINEAR_HASH_W16_AVX_CPU_BENCH")
#endif

// ---------------------------------------------------------------------------
// merkletree registrations -- (W,arity) in {(8,2),(12,3),(16,4)}
// modes: Scalar, Avx, AvxBatch, Avx512Batch
// ---------------------------------------------------------------------------

REG_NCOLS_AR(MERKLETREE_W_AR_SCALAR_CPU_BENCH, 8,  2, "MERKLETREE_W8_AR2_SCALAR_CPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_SCALAR_CPU_BENCH, 12, 3, "MERKLETREE_W12_AR3_SCALAR_CPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_SCALAR_CPU_BENCH, 16, 4, "MERKLETREE_W16_AR4_SCALAR_CPU_BENCH")

#ifdef __AVX2__
REG_NCOLS_AR(MERKLETREE_W_AR_AVX_CPU_BENCH, 8,  2, "MERKLETREE_W8_AR2_AVX_CPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVX_CPU_BENCH, 12, 3, "MERKLETREE_W12_AR3_AVX_CPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVX_CPU_BENCH, 16, 4, "MERKLETREE_W16_AR4_AVX_CPU_BENCH")

REG_NCOLS_AR(MERKLETREE_W_AR_AVXBATCH_CPU_BENCH, 8,  2, "MERKLETREE_W8_AR2_AVXBATCH_CPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVXBATCH_CPU_BENCH, 12, 3, "MERKLETREE_W12_AR3_AVXBATCH_CPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVXBATCH_CPU_BENCH, 16, 4, "MERKLETREE_W16_AR4_AVXBATCH_CPU_BENCH")
#endif

#ifdef __AVX512__
REG_NCOLS_AR(MERKLETREE_W_AR_AVX512BATCH_CPU_BENCH, 8,  2, "MERKLETREE_W8_AR2_AVX512BATCH_CPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVX512BATCH_CPU_BENCH, 12, 3, "MERKLETREE_W12_AR3_AVX512BATCH_CPU_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVX512BATCH_CPU_BENCH, 16, 4, "MERKLETREE_W16_AR4_AVX512BATCH_CPU_BENCH")
#endif

// ---------------------------------------------------------------------------
// grinding registration
// ---------------------------------------------------------------------------

BENCHMARK(GRINDING_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->Arg(20)->Arg(21)->Arg(22)->Arg(24)->Arg(25)
    ->UseRealTime();

#undef REG_NCOLS
#undef REG_NCOLS_AR
#undef REG_ELEM
#undef NCOLS_ARGS
