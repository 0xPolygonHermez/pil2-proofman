// ---------------------------------------------------------------------------
// bench_ntt_cpu.cpp -- CPU NTT/INTT/LDE benchmarks
//
// Unified parameters for apples-to-apples comparison with GPU benchmarks:
//   BENCH_NTT_NBITS     = 22 (fixed production size for NTT/INTT)
//   BENCH_LDE_NBITS     = 20
//   BENCH_LDE_BLOWUP    = 2  (nBitsExt = nBits + 2 = 22, 4x blowup)
//   BENCH_NCOLS          = {24, 36, 56}  (via ->Arg())
//
// Naming: OPERATION_CPU_BENCH
// All benches use max threads and ->UseRealTime().
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cstdint>
#include "../src/goldilocks_base_field.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "omp.h"

// ---------------------------------------------------------------------------
// Unified parameters
// ---------------------------------------------------------------------------
static constexpr uint64_t BENCH_NTT_NBITS      = 22;
static constexpr uint64_t BENCH_NTT_SIZE        = 1ULL << BENCH_NTT_NBITS;

static constexpr uint64_t BENCH_LDE_NBITS       = 20;
static constexpr uint64_t BENCH_LDE_SIZE         = 1ULL << BENCH_LDE_NBITS;
static constexpr uint64_t BENCH_LDE_BLOWUP_BITS  = 2;
static constexpr uint64_t BENCH_LDE_NBITS_EXT    = BENCH_LDE_NBITS + BENCH_LDE_BLOWUP_BITS;
static constexpr uint64_t BENCH_LDE_SIZE_EXT     = 1ULL << BENCH_LDE_NBITS_EXT;

// ---------------------------------------------------------------------------
// Helper: fill block-interleaved buffer with deterministic data
// Block layout: nRows x nCols, row-by-row
// ---------------------------------------------------------------------------
static void fillBlock(Goldilocks::Element *buf, uint64_t nRows, uint64_t nCols)
{
    for (uint64_t i = 0; i < 2 && i < nRows; i++)
        for (uint64_t j = 0; j < nCols; j++)
            buf[i * nCols + j] = Goldilocks::fromU64(j + 1);

    for (uint64_t i = 2; i < nRows; i++)
        for (uint64_t j = 0; j < nCols; j++)
            buf[i * nCols + j] = buf[(i - 1) * nCols + j] + buf[(i - 2) * nCols + j];
}

// ===================================================================
// NTT -- nBits=22, parameterized by nCols
// Uses the multi-column block NTT API (row-interleaved layout).
// ===================================================================

static void NTT_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    NTT_Goldilocks ntt(BENCH_NTT_SIZE);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(BENCH_NTT_SIZE * nCols * sizeof(Goldilocks::Element));
    fillBlock(a, BENCH_NTT_SIZE, nCols);

    for (auto _ : state)
        ntt.NTT(a, a, BENCH_NTT_SIZE, nCols);

    free(a);
}

// ===================================================================
// INTT -- nBits=22, parameterized by nCols
// ===================================================================

static void INTT_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    NTT_Goldilocks ntt(BENCH_NTT_SIZE);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(BENCH_NTT_SIZE * nCols * sizeof(Goldilocks::Element));
    fillBlock(a, BENCH_NTT_SIZE, nCols);

    for (auto _ : state)
        ntt.INTT(a, a, BENCH_NTT_SIZE, nCols);

    free(a);
}

// ===================================================================
// LDE -- nBits=20, nBitsExt=22, parameterized by nCols
// Uses the public LDE API (INTT + shift + zero-pad + NTT).
// ===================================================================

static void LDE_CPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);
    NTT_Goldilocks ntt(BENCH_LDE_SIZE);

    Goldilocks::Element *input  = (Goldilocks::Element *)malloc(BENCH_LDE_SIZE_EXT * nCols * sizeof(Goldilocks::Element));
    Goldilocks::Element *output = (Goldilocks::Element *)malloc(BENCH_LDE_SIZE_EXT * nCols * sizeof(Goldilocks::Element));
    Goldilocks::Element *buffer = (Goldilocks::Element *)malloc(BENCH_LDE_SIZE_EXT * nCols * sizeof(Goldilocks::Element));
    fillBlock(input, BENCH_LDE_SIZE, nCols);

    for (auto _ : state)
        ntt.LDE(output, input, BENCH_LDE_SIZE_EXT, BENCH_LDE_SIZE, nCols, buffer);

    free(input);
    free(output);
    free(buffer);
}

// ===================================================================
// Registration
// ===================================================================

#define NCOLS_ARGS ->Arg(24)->Arg(36)->Arg(56)

BENCHMARK(NTT_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    NCOLS_ARGS
    ->UseRealTime();

BENCHMARK(INTT_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    NCOLS_ARGS
    ->UseRealTime();

BENCHMARK(LDE_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    NCOLS_ARGS
    ->UseRealTime();

#undef NCOLS_ARGS
