// ---------------------------------------------------------------------------
// bench_poseidon_cpu.cpp -- CPU Poseidon v1 benchmarks (W ∈ {8, 12, 16}).
//
// Parameters mirror bench_poseidon2_cpu.cpp for side-by-side comparison:
//   BENCH_NROWS = 1 << 23 (8M rows)
//   NUM_HASHES  = 1 << 21 (2M hashes) for per-element ops
//   (W, arity) pairs benched: (8,2), (12,3), (16,4) — matches the production
//   STARK_POSEIDON1 dispatch (W = arity * CAPACITY).
//
// Single-sponge AVX (`Avx` mode) is W=12-only — registrations are gated by
// the `W == 12` constexpr inside the bench bodies; the W != 12 cases assert
// out and aren't registered.
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cstdint>
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/goldilocks_tooling.hpp"
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "omp.h"

static constexpr uint64_t BENCH_NROWS = 1ULL << 23;
static constexpr uint64_t NUM_HASHES  = 1ULL << 21;

static void fillData(Goldilocks::Element *buf, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
        buf[i] = Goldilocks::fromU64(i + 1);
}

// ===========================================================================
// permute / permuteTrunc — per-element throughput
// ===========================================================================

template<uint32_t W>
static void PERMUTE_W_SCALAR_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t total = (uint64_t)NUM_HASHES * W;
    Goldilocks::Element *x = new Goldilocks::Element[total];
    Goldilocks::Element *r = new Goldilocks::Element[total];
    fillData(x, total);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
            PoseidonGoldilocks<W>::permute(
                (Goldilocks::Element(&)[W])r[i * W],
                (const Goldilocks::Element(&)[W])x[i * W],
                PoseidonMode::Scalar);
    }
    state.SetItemsProcessed(state.iterations() * NUM_HASHES);
    state.counters["hashes"] = NUM_HASHES;
    delete[] x; delete[] r;
}

#ifdef __AVX2__
template<uint32_t W>
static void PERMUTE_W_AVX_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t total = (uint64_t)NUM_HASHES * W;
    Goldilocks::Element *x = new Goldilocks::Element[total];
    Goldilocks::Element *r = new Goldilocks::Element[total];
    fillData(x, total);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
            PoseidonGoldilocks<W>::permute(
                (Goldilocks::Element(&)[W])r[i * W],
                (const Goldilocks::Element(&)[W])x[i * W],
                PoseidonMode::Avx);
    }
    state.SetItemsProcessed(state.iterations() * NUM_HASHES);
    state.counters["hashes"] = NUM_HASHES;
    delete[] x; delete[] r;
}
#endif

// ===========================================================================
// linearHash — size sweep over nCols
// ===========================================================================

template<uint32_t W>
static void LINEAR_HASH_W_SCALAR_CPU_POS1_BENCH(benchmark::State &state)
{
    constexpr uint32_t CAP = PoseidonGoldilocks<W>::CAPACITY;
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    Goldilocks::Element *res  = new Goldilocks::Element[CAP * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    for (auto _ : state) {
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (uint64_t i = 0; i < BENCH_NROWS; i++)
            PoseidonGoldilocks<W>::linearHash(&res[i * CAP], &cols[i * nCols], nCols, PoseidonMode::Scalar);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] res;
}

#ifdef __AVX2__
template<uint32_t W>
static void LINEAR_HASH_W_AVX_CPU_POS1_BENCH(benchmark::State &state)
{
    constexpr uint32_t CAP = PoseidonGoldilocks<W>::CAPACITY;
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    Goldilocks::Element *res  = new Goldilocks::Element[CAP * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    for (auto _ : state) {
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (uint64_t i = 0; i < BENCH_NROWS; i++)
            PoseidonGoldilocks<W>::linearHash(&res[i * CAP], &cols[i * nCols], nCols, PoseidonMode::Avx);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] res;
}
#endif

// ===========================================================================
// merkletree — (W, arity) ∈ {(8,2),(12,3),(16,4)}
// ===========================================================================

template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_SCALAR_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];
    for (auto _ : state) {
        PoseidonGoldilocks<W>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, PoseidonMode::Scalar);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] tree;
}

#ifdef __AVX2__
template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_AVX_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];
    for (auto _ : state) {
        PoseidonGoldilocks<W>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, PoseidonMode::Avx);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] tree;
}

template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_AVXBATCH_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];
    for (auto _ : state) {
        PoseidonGoldilocks<W>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, PoseidonMode::AvxBatch);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] tree;
}
#endif

#ifdef __AVX512__
template<uint32_t W, uint32_t ARITY>
static void MERKLETREE_W_AR_AVX512BATCH_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];
    for (auto _ : state) {
        PoseidonGoldilocks<W>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, PoseidonMode::Avx512Batch);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] tree;
}
#endif

// ===========================================================================
// Registration
// ===========================================================================

#define NCOLS_ARGS ->Arg(24)->Arg(36)->Arg(56)

#define REG_PERMUTE(FN, W, LABEL)                                            \
    BENCHMARK_TEMPLATE(FN, W)                                                \
        ->Name(LABEL)->Unit(benchmark::kMillisecond)->UseRealTime();

#define REG_NCOLS_AR(FN, W, AR, LABEL)                                       \
    BENCHMARK_TEMPLATE(FN, W, AR)                                            \
        ->Name(LABEL)->Unit(benchmark::kMillisecond) NCOLS_ARGS              \
        ->UseRealTime();

REG_PERMUTE(PERMUTE_W_SCALAR_CPU_POS1_BENCH,  8, "PERMUTE_W8_SCALAR_CPU_POS1_BENCH")
REG_PERMUTE(PERMUTE_W_SCALAR_CPU_POS1_BENCH, 12, "PERMUTE_W12_SCALAR_CPU_POS1_BENCH")
REG_PERMUTE(PERMUTE_W_SCALAR_CPU_POS1_BENCH, 16, "PERMUTE_W16_SCALAR_CPU_POS1_BENCH")

#ifdef __AVX2__
// single-sponge AVX is W=12-only (hand-tuned 12×12 kernels)
REG_PERMUTE(PERMUTE_W_AVX_CPU_POS1_BENCH,    12, "PERMUTE_W12_AVX_CPU_POS1_BENCH")
#endif

// linearHash — Scalar across all widths, AVX only at W=12.
#define REG_LH_NCOLS(FN, W, LABEL)                                           \
    BENCHMARK_TEMPLATE(FN, W)                                                \
        ->Name(LABEL)->Unit(benchmark::kMillisecond) NCOLS_ARGS              \
        ->UseRealTime();

REG_LH_NCOLS(LINEAR_HASH_W_SCALAR_CPU_POS1_BENCH,  8, "LINEAR_HASH_W8_SCALAR_CPU_POS1_BENCH")
REG_LH_NCOLS(LINEAR_HASH_W_SCALAR_CPU_POS1_BENCH, 12, "LINEAR_HASH_W12_SCALAR_CPU_POS1_BENCH")
REG_LH_NCOLS(LINEAR_HASH_W_SCALAR_CPU_POS1_BENCH, 16, "LINEAR_HASH_W16_SCALAR_CPU_POS1_BENCH")

#ifdef __AVX2__
REG_LH_NCOLS(LINEAR_HASH_W_AVX_CPU_POS1_BENCH,    12, "LINEAR_HASH_W12_AVX_CPU_POS1_BENCH")
#endif

// merkletree — (W,arity) ∈ {(8,2),(12,3),(16,4)} matches production dispatch.
REG_NCOLS_AR(MERKLETREE_W_AR_SCALAR_CPU_POS1_BENCH,  8, 2, "MERKLETREE_W8_AR2_SCALAR_CPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_SCALAR_CPU_POS1_BENCH, 12, 3, "MERKLETREE_W12_AR3_SCALAR_CPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_SCALAR_CPU_POS1_BENCH, 16, 4, "MERKLETREE_W16_AR4_SCALAR_CPU_POS1_BENCH")

#ifdef __AVX2__
// Single-sponge AVX is W=12 only.
REG_NCOLS_AR(MERKLETREE_W_AR_AVX_CPU_POS1_BENCH, 12, 3, "MERKLETREE_W12_AR3_AVX_CPU_POS1_BENCH")

// AVX batch path is W-generic for {8,12,16}.
REG_NCOLS_AR(MERKLETREE_W_AR_AVXBATCH_CPU_POS1_BENCH,  8, 2, "MERKLETREE_W8_AR2_AVXBATCH_CPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVXBATCH_CPU_POS1_BENCH, 12, 3, "MERKLETREE_W12_AR3_AVXBATCH_CPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVXBATCH_CPU_POS1_BENCH, 16, 4, "MERKLETREE_W16_AR4_AVXBATCH_CPU_POS1_BENCH")
#endif

#ifdef __AVX512__
REG_NCOLS_AR(MERKLETREE_W_AR_AVX512BATCH_CPU_POS1_BENCH,  8, 2, "MERKLETREE_W8_AR2_AVX512BATCH_CPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVX512BATCH_CPU_POS1_BENCH, 12, 3, "MERKLETREE_W12_AR3_AVX512BATCH_CPU_POS1_BENCH")
REG_NCOLS_AR(MERKLETREE_W_AR_AVX512BATCH_CPU_POS1_BENCH, 16, 4, "MERKLETREE_W16_AR4_AVX512BATCH_CPU_POS1_BENCH")
#endif
