#include <benchmark/benchmark.h>
#include "../src/goldilocks_base_field.hpp"
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "omp.h"

static void ADD_OP_BENCH(benchmark::State &state)
{
    Goldilocks::Element term0 = Goldilocks::zero(), term1 = Goldilocks::zero(), term2 = Goldilocks::zero();
    // Benchmark
    for (auto _ : state)
    {
        term0 = Goldilocks::one() + Goldilocks::one();
        term1 = Goldilocks::one() + Goldilocks::one() + Goldilocks::one();
        ;
        for (uint64_t i = 0; i < 1000000; i++)
        {
            Goldilocks::add(term2, term0, term1);
            term0 = term1;
            term1 = term2;
        }
    }
    assert(Goldilocks::toU64(term2) == 9315693631868018522ULL);
}
static void SUB_OP_BENCH(benchmark::State &state)
{
    Goldilocks::Element term0 = Goldilocks::zero(), term1 = Goldilocks::zero(), term2 = Goldilocks::zero();
    // Benchmark
    for (auto _ : state)
    {
        term0 = Goldilocks::one() + Goldilocks::one();
        term1 = Goldilocks::one() + Goldilocks::one() + Goldilocks::one();
        ;
        for (uint64_t i = 0; i < 1000000; i++)
        {
            Goldilocks::sub(term2, term0, term1);
            term0 = term1;
            term1 = term2;
        }
    }
    assert(Goldilocks::toU64(term2) == 17916187359919173389ULL);
}
static void MUL_OP_BENCH(benchmark::State &state)
{
    Goldilocks::Element term0 = Goldilocks::zero(), term1 = Goldilocks::zero(), term2 = Goldilocks::zero();
    // Benchmark
    for (auto _ : state)
    {
        term0 = Goldilocks::one() + Goldilocks::one();
        term1 = Goldilocks::one() + Goldilocks::one() + Goldilocks::one();
        ;
        for (uint64_t i = 0; i < 1000000; i++)
        {
            Goldilocks::mul(term2, term0, term1);
            term0 = term1;
            term1 = term2;
        }
    }
    assert(Goldilocks::toU64(term2) == 1922281271747280077ULL);
}
static void INV_OP_BENCH(benchmark::State &state)
{
    Goldilocks::Element term0 = Goldilocks::zero(), term1 = Goldilocks::zero();
    // Benchmark
    for (auto _ : state)
    {
        term0 = Goldilocks::one() + Goldilocks::one();
        term1 = Goldilocks::one() + Goldilocks::one() + Goldilocks::one();

        for (uint64_t i = 0; i < 1000000; i++)
        {
            Goldilocks::inv(term0, term1);
            term1 = term0;
        }
    }
    assert(Goldilocks::toU64(term0) == 3ULL);
}

#ifdef __AVX2__
static void ADD_OP_AVX_BENCH(benchmark::State &state)
{
    Goldilocks::Element term0, term1;
    __m256i term0_, term1_, term2_ = _mm256_setzero_si256();

    // Benchmark
    for (auto _ : state)
    {
        term0 = Goldilocks::one() + Goldilocks::one();
        term1 = Goldilocks::one() + Goldilocks::one() + Goldilocks::one();
        Goldilocks::set_avx(term0_, term0, term0, term0, term0);
        Goldilocks::set_avx(term1_, term1, term1, term1, term1);
        for (uint64_t i = 0; i < 1000000; i++)
        {
            Goldilocks::add_avx(term2_, term0_, term1_);
            term0_ = term1_;
            term1_ = term2_;
        }
    }
    Goldilocks::Element res[4];
    Goldilocks::store_avx(res, term2_);
    assert(Goldilocks::toU64(res[0]) == 9315693631868018522ULL);
}
static void SUB_OP_AVX_BENCH(benchmark::State &state)
{
    Goldilocks::Element term0, term1;
    __m256i term0_, term1_, term2_ = _mm256_setzero_si256();

    // Benchmark
    for (auto _ : state)
    {
        term0 = Goldilocks::one() + Goldilocks::one();
        term1 = Goldilocks::one() + Goldilocks::one() + Goldilocks::one();
        Goldilocks::set_avx(term0_, term0, term0, term0, term0);
        Goldilocks::set_avx(term1_, term1, term1, term1, term1);
        for (uint64_t i = 0; i < 1000000; i++)
        {
            Goldilocks::sub_avx(term2_, term0_, term1_);
            term0_ = term1_;
            term1_ = term2_;
        }
    }
    Goldilocks::Element res[4];
    Goldilocks::store_avx(res, term2_);
    assert(Goldilocks::toU64(res[0]) == 17916187359919173389ULL);

}
static void MUL_OP_AVX_BENCH(benchmark::State &state)
{
    Goldilocks::Element term0, term1;
    __m256i term0_, term1_, term2_ = _mm256_setzero_si256();

    // Benchmark
    for (auto _ : state)
    {
        term0 = Goldilocks::one() + Goldilocks::one();
        term1 = Goldilocks::one() + Goldilocks::one() + Goldilocks::one();
        Goldilocks::set_avx(term0_, term0, term0, term0, term0);
        Goldilocks::set_avx(term1_, term1, term1, term1, term1);
        for (uint64_t i = 0; i < 1000000; i++)
        {
            Goldilocks::mult_avx(term2_, term0_, term1_);
            term0_ = term1_;
            term1_ = term2_;
        }
    }
    Goldilocks::Element res[4];
    Goldilocks::store_avx(res, term2_);
    assert(Goldilocks::toU64(res[0]) == 1922281271747280077ULL);
}
#endif

#ifdef __AVX2__
BENCHMARK(ADD_OP_AVX_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK(SUB_OP_AVX_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK(MUL_OP_AVX_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
#endif
   
BENCHMARK(ADD_OP_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK(SUB_OP_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK(MUL_OP_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK(INV_OP_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
