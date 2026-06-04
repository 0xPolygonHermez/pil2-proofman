// ---------------------------------------------------------------------------
// bench_poseidon_cpu.cpp -- CPU Poseidon v1 benchmarks (W=12 only).
//
// Parameters mirror bench_poseidon2_cpu.cpp so numbers can be compared
// side-by-side with the Poseidon2 suite:
//   BENCH_NROWS = 1 << 23 (8M rows)
//   NUM_HASHES  = 1 << 21 (2M hashes) for per-element ops
//   ARITY=2 used for merkletree (Poseidon v1's canonical binary tree).
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
static constexpr uint32_t W     = PoseidonGoldilocks<12>::SPONGE_WIDTH;
static constexpr uint32_t CAP   = PoseidonGoldilocks<12>::CAPACITY;
static constexpr uint64_t ARITY = 2;

static void fillData(Goldilocks::Element *buf, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
        buf[i] = Goldilocks::fromU64(i + 1);
}

// ===========================================================================
// permute / permuteTrunc — per-element throughput
// ===========================================================================

static void PERMUTE_W12_SCALAR_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t total = (uint64_t)NUM_HASHES * W;
    Goldilocks::Element *x = new Goldilocks::Element[total];
    Goldilocks::Element *r = new Goldilocks::Element[total];
    fillData(x, total);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
            PoseidonGoldilocks<12>::permute(
                (Goldilocks::Element(&)[W])r[i * W],
                (const Goldilocks::Element(&)[W])x[i * W],
                PoseidonMode::Scalar);
    }
    state.SetItemsProcessed(state.iterations() * NUM_HASHES);
    state.counters["hashes"] = NUM_HASHES;
    delete[] x; delete[] r;
}
BENCHMARK(PERMUTE_W12_SCALAR_CPU_POS1_BENCH)->Unit(benchmark::kMillisecond)->UseRealTime();

#ifdef __AVX2__
static void PERMUTE_W12_AVX_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t total = (uint64_t)NUM_HASHES * W;
    Goldilocks::Element *x = new Goldilocks::Element[total];
    Goldilocks::Element *r = new Goldilocks::Element[total];
    fillData(x, total);

    int nT = omp_get_max_threads();
    for (auto _ : state) {
#pragma omp parallel for num_threads(nT) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
            PoseidonGoldilocks<12>::permute(
                (Goldilocks::Element(&)[W])r[i * W],
                (const Goldilocks::Element(&)[W])x[i * W],
                PoseidonMode::Avx);
    }
    state.SetItemsProcessed(state.iterations() * NUM_HASHES);
    state.counters["hashes"] = NUM_HASHES;
    delete[] x; delete[] r;
}
BENCHMARK(PERMUTE_W12_AVX_CPU_POS1_BENCH)->Unit(benchmark::kMillisecond)->UseRealTime();
#endif

// ===========================================================================
// linearHash — size sweep over BENCH_NCOLS
// ===========================================================================

static void LINEAR_HASH_W12_SCALAR_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    Goldilocks::Element *res  = new Goldilocks::Element[CAP * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    for (auto _ : state) {
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (uint64_t i = 0; i < BENCH_NROWS; i++)
            PoseidonGoldilocks<12>::linearHash(&res[i * CAP], &cols[i * nCols], nCols, PoseidonMode::Scalar);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] res;
}
BENCHMARK(LINEAR_HASH_W12_SCALAR_CPU_POS1_BENCH)->Unit(benchmark::kMillisecond)->Arg(24)->Arg(36)->Arg(56)->UseRealTime();

#ifdef __AVX2__
static void LINEAR_HASH_W12_AVX_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    Goldilocks::Element *res  = new Goldilocks::Element[CAP * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    for (auto _ : state) {
        #pragma omp parallel for num_threads(omp_get_max_threads())
        for (uint64_t i = 0; i < BENCH_NROWS; i++)
            PoseidonGoldilocks<12>::linearHash(&res[i * CAP], &cols[i * nCols], nCols, PoseidonMode::Avx);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] res;
}
BENCHMARK(LINEAR_HASH_W12_AVX_CPU_POS1_BENCH)->Unit(benchmark::kMillisecond)->Arg(24)->Arg(36)->Arg(56)->UseRealTime();
#endif

// ===========================================================================
// merkletree — binary (arity=2), full tree over BENCH_NROWS leaves.
// ===========================================================================

static void MERKLETREE_W12_SCALAR_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];
    for (auto _ : state) {
        PoseidonGoldilocks<12>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, PoseidonMode::Scalar);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] tree;
}
BENCHMARK(MERKLETREE_W12_SCALAR_CPU_POS1_BENCH)->Unit(benchmark::kMillisecond)->Arg(24)->Arg(36)->Arg(56)->UseRealTime();

#ifdef __AVX2__
static void MERKLETREE_W12_AVX_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];
    for (auto _ : state) {
        PoseidonGoldilocks<12>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, PoseidonMode::Avx);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] tree;
}
BENCHMARK(MERKLETREE_W12_AVX_CPU_POS1_BENCH)->Unit(benchmark::kMillisecond)->Arg(24)->Arg(36)->Arg(56)->UseRealTime();

static void MERKLETREE_W12_AVXBATCH_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];
    for (auto _ : state) {
        PoseidonGoldilocks<12>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, PoseidonMode::AvxBatch);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] tree;
}
BENCHMARK(MERKLETREE_W12_AVXBATCH_CPU_POS1_BENCH)->Unit(benchmark::kMillisecond)->Arg(24)->Arg(36)->Arg(56)->UseRealTime();
#endif

#ifdef __AVX512__
static void MERKLETREE_W12_AVX512BATCH_CPU_POS1_BENCH(benchmark::State &state)
{
    const uint64_t nCols = state.range(0);
    Goldilocks::Element *cols = new Goldilocks::Element[nCols * BENCH_NROWS];
    fillData(cols, nCols * BENCH_NROWS);
    uint64_t numElems = getTreeNumElements(BENCH_NROWS, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];
    for (auto _ : state) {
        PoseidonGoldilocks<12>::merkletree(tree, cols, nCols, BENCH_NROWS, ARITY, PoseidonMode::Avx512Batch);
    }
    state.SetItemsProcessed(state.iterations() * BENCH_NROWS);
    delete[] cols; delete[] tree;
}
BENCHMARK(MERKLETREE_W12_AVX512BATCH_CPU_POS1_BENCH)->Unit(benchmark::kMillisecond)->Arg(24)->Arg(36)->Arg(56)->UseRealTime();
#endif
