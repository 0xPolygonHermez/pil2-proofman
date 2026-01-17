#include <benchmark/benchmark.h>
#include <gmp.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <omp.h>
#include <random>
#include <chrono>
#include "fr.hpp"
#include "fq.hpp"
#include "alt_bn128.hpp"
#include "multiexp.hpp"
#if defined(__BLST__)
#include <blst.h>
#endif


// Unsigned Benchmarks (as reference)

static void ADD_U64_BENCH(benchmark::State &state)
{
    uint64_t a = 123456789;
    uint64_t b = 987654321;
    uint64_t c = 0;
    // Benchmark
    for (auto _ : state)
    {
        c = a + b;  
        a = b;
        b = c;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(ADD_U64_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void SUB_U64_BENCH(benchmark::State &state)
{
    uint64_t a = 987654321;
    uint64_t b = 123456789;
    uint64_t c = 0;
    // Benchmark
    for (auto _ : state)
    {
        c = a - b;  
        a = b;
        b = c;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(SUB_U64_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void MUL_U64_BENCH(benchmark::State &state)
{
    uint64_t a = 123456789;
    uint64_t b = 987654321;
    uint64_t c = 0;
    // Benchmark
    for (auto _ : state)
    {
        c = a * b;  
        a = b;
        b = c;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(MUL_U64_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// FR Benchmarks

static void ADD_FR_BENCH(benchmark::State &state)
{
    RawFrP field;
    RawFrP::Element a, b, c;
    
    // Use 253-bit values
    mpz_t a_mpz, b_mpz;
    mpz_init_set_str(a_mpz, "14474011154666747474405997541838961898253990025393074346253298847191858934464", 10);
    mpz_init_set_str(b_mpz, "7237005577333373737202998770919480949126995012696537173126649423595929467232", 10);
    field.fromMpz(a, a_mpz);
    field.fromMpz(b, b_mpz);
    mpz_clear(a_mpz);
    mpz_clear(b_mpz);
    
    // Benchmark
    for (auto _ : state)
    {
        field.add(c, a, b);  
        field.copy(a, b);
        field.copy(b, c);
        benchmark::DoNotOptimize(c);    
    }
}

BENCHMARK(ADD_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void SUB_FR_BENCH(benchmark::State &state)
{
    RawFrP field;
    RawFrP::Element a, b, c;
    
    mpz_t a_mpz, b_mpz;
    mpz_init_set_str(a_mpz, "7237005577333373737202998770919480949126995012696537173126649423595929467232", 10);
    mpz_init_set_str(b_mpz, "14474011154666747474405997541838961898253990025393074346253298847191858934464", 10);
    field.fromMpz(a, a_mpz);
    field.fromMpz(b, b_mpz);
    mpz_clear(a_mpz);
    mpz_clear(b_mpz);
    
    // Benchmark
    for (auto _ : state)
    {
        field.sub(c, a, b);  
        field.copy(a, b);
        field.copy(b, c);
        benchmark::DoNotOptimize(c);    
    }
}

BENCHMARK(SUB_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void MUL_FR_BENCH(benchmark::State &state)
{
    RawFrP field;
    RawFrP::Element a, b, c;
    
    mpz_t a_mpz, b_mpz;
    mpz_init_set_str(a_mpz, "14474011154666747474405997541838961898253990025393074346253298847191858934464", 10);
    mpz_init_set_str(b_mpz, "7237005577333373737202998770919480949126995012696537173126649423595929467232", 10);
    field.fromMpz(a, a_mpz);
    field.fromMpz(b, b_mpz);
    mpz_clear(a_mpz);
    mpz_clear(b_mpz);
    
    // Benchmark
    for (auto _ : state)
    {
        field.mul(c, a, b);  
        field.copy(a, b);
        field.copy(b, c);
        benchmark::DoNotOptimize(c);    
    }
}

BENCHMARK(MUL_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void SQUARE_FR_BENCH(benchmark::State &state)
{
    RawFrP field;
    RawFrP::Element a, c;
    
    mpz_t a_mpz;
    mpz_init_set_str(a_mpz, "14474011154666747474405997541838961898253990025393074346253298847191858934464", 10);
    field.fromMpz(a, a_mpz);
    mpz_clear(a_mpz);
    
    // Benchmark
    for (auto _ : state)
    {
        field.square(c, a);  
        field.copy(a, c);
        benchmark::DoNotOptimize(c);    
    }
}

BENCHMARK(SQUARE_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void DIV_FR_BENCH(benchmark::State &state)
{
    RawFrP field;
    RawFrP::Element a, b, c;
    
    mpz_t a_mpz, b_mpz;
    mpz_init_set_str(a_mpz, "7237005577333373737202998770919480949126995012696537173126649423595929467232", 10);
    mpz_init_set_str(b_mpz, "14474011154666747474405997541838961898253990025393074346253298847191858934464", 10);
    field.fromMpz(a, a_mpz);
    field.fromMpz(b, b_mpz);
    mpz_clear(a_mpz);
    mpz_clear(b_mpz);
    
    // Benchmark
    for (auto _ : state)
    {
        field.div(c, a, b);  
        field.copy(a, b);
        field.copy(b, a);
        benchmark::DoNotOptimize(c);    
    }
}

BENCHMARK(DIV_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void INV_FR_BENCH(benchmark::State &state)
{
    RawFrP field;
    RawFrP::Element a, c;
    
    mpz_t a_mpz;
    mpz_init_set_str(a_mpz, "14474011154666747474405997541838961898253990025393074346253298847191858934464", 10);
    field.fromMpz(a, a_mpz);
    mpz_clear(a_mpz);
    
    // Benchmark
    for (auto _ : state)
    {
        field.inv(c, a);  
        field.copy(a, c);
        benchmark::DoNotOptimize(c);    
    }
}

BENCHMARK(INV_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

// Add BLS FR benchmarks if BLST is enabled
// We can compare BLS's FR implementation performance with our own RawFrP implementation
// the scalar fields are not the same but similar: 255-bit prime for BLS vs 254-bit prime for RawFrP
#if defined(__BLST__)

static void BLST_ADD_FR_BENCH(benchmark::State &state)
{
    blst_fr a, b, c;
    
    // Use 253-bit values (same as FR benchmarks)
    uint64_t a_arr[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF};
    uint64_t b_arr[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x07FFFFFFFFFFFFFF};
    blst_fr_from_uint64(&a, a_arr);
    blst_fr_from_uint64(&b, b_arr);
    
    // Benchmark
    for (auto _ : state)
    {
        blst_fr_add(&c, &a, &b);  
        a = b;
        b = c;
        benchmark::DoNotOptimize(c);    
    }
}   
BENCHMARK(BLST_ADD_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void BLST_SUB_FR_BENCH(benchmark::State &state)
{
    blst_fr a, b, c;
    
    // Use 253-bit values (same as FR benchmarks)
    uint64_t a_arr[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x07FFFFFFFFFFFFFF};
    uint64_t b_arr[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF};
    blst_fr_from_uint64(&a, a_arr);
    blst_fr_from_uint64(&b, b_arr);
    
    // Benchmark
    for (auto _ : state)
    {
        blst_fr_sub(&c, &a, &b);  
        a = b;
        b = c;
        benchmark::DoNotOptimize(c);    
    }
}

BENCHMARK(BLST_SUB_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void BLST_MUL_FR_BENCH(benchmark::State &state)
{
    blst_fr a, b, c;    
    // Use 253-bit values (same as FR benchmarks)
    uint64_t a_arr[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF};
    uint64_t b_arr[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x07FFFFFFFFFFFFFF};
    blst_fr_from_uint64(&a, a_arr);
    blst_fr_from_uint64(&b, b_arr);
    // Benchmark
    for (auto _ : state)
    {
        blst_fr_mul(&c, &a, &b);  
        a = b;
        b = c;
        benchmark::DoNotOptimize(c);    
    }
}   

BENCHMARK(BLST_MUL_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

static void BLST_INV_FR_BENCH(benchmark::State &state)
{
    blst_fr a, c;    
    // Use 253-bit value (same as FR benchmarks)
    uint64_t a_arr[4] = {0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF};
    blst_fr_from_uint64(&a, a_arr);
    // Benchmark
    for (auto _ : state)
    {
        blst_fr_inverse(&c, &a);  
        a = c;
        benchmark::DoNotOptimize(c);    
    }
}   
BENCHMARK(BLST_INV_FR_BENCH)
    ->Unit(benchmark::kNanosecond)
    ->UseRealTime();

#endif

// =====================
// MSM CPU Benchmark
// =====================

// Global storage for precomputed test data (allocated once)
static AltBn128::G1PointAffine* g_msm_bases = nullptr;
static uint8_t* g_msm_scalars = nullptr;
static uint64_t g_msm_n = 0;
static const uint64_t MSM_SCALAR_SIZE = 32;  // 256-bit scalars

static void setup_msm_data(uint64_t n) {
    if (g_msm_bases != nullptr && g_msm_n == n) {
        return;  // Already initialized with same size
    }
    
    // Cleanup previous allocation
    if (g_msm_bases != nullptr) {
        delete[] g_msm_bases;
        delete[] g_msm_scalars;
    }
    
    g_msm_n = n;
    g_msm_bases = new AltBn128::G1PointAffine[n];
    g_msm_scalars = new uint8_t[n * MSM_SCALAR_SIZE];
        
    // Generate base points: g_msm_bases[i] = 2^i * G
    
    // Pre-compute chunk starting points for parallel generation
    const int nChunks = omp_get_max_threads();
    std::vector<AltBn128::G1Point> chunkStarts(nChunks);
    
    // Compute chunk starting points sequentially
    uint64_t chunkSize = (n + nChunks - 1) / nChunks;
    AltBn128::G1Point acc;
    AltBn128::G1.copy(acc, AltBn128::G1.oneAffine());
    
    for (int c = 0; c < nChunks; c++) {
        AltBn128::G1.copy(chunkStarts[c], acc);
        // Advance acc by chunkSize doublings
        for (uint64_t j = 0; j < chunkSize && (c * chunkSize + j) < n; j++) {
            AltBn128::G1.dbl(acc, acc);
        }
    }
    
    // Generate points in parallel: base[i] = 2^i * G
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t start_idx = tid * chunkSize;
        uint64_t end_idx = std::min(start_idx + chunkSize, n);
        
        AltBn128::G1Point localPoint;
        AltBn128::G1.copy(localPoint, chunkStarts[tid]);
        
        for (uint64_t i = start_idx; i < end_idx; i++) {
            AltBn128::G1.copy(g_msm_bases[i], localPoint);
            AltBn128::G1.dbl(localPoint, localPoint);
        }
    }
    
    // Generate random scalars in parallel
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // Use thread-local RNG with unique seed per thread
        std::mt19937_64 rng(42 + tid);  // Deterministic seed for reproducibility
        
        uint64_t start_idx = (tid * n) / omp_get_num_threads();
        uint64_t end_idx = ((tid + 1) * n) / omp_get_num_threads();
        
        for (uint64_t i = start_idx; i < end_idx; i++) {
            uint8_t* scalar = &g_msm_scalars[i * MSM_SCALAR_SIZE];
            // Generate 253-bit scalar (BN254 scalar field is ~253 bits)
            for (size_t j = 0; j < MSM_SCALAR_SIZE; j++) {
                scalar[j] = rng() & 0xFF;
            }
            // Ensure scalar < field order by clearing top bits
            scalar[MSM_SCALAR_SIZE - 1] &= 0x1F;  // ~253 bits
        }
    }
}

static void cleanup_msm_data() {
    if (g_msm_bases != nullptr) {
        delete[] g_msm_bases;
        delete[] g_msm_scalars;
        g_msm_bases = nullptr;
        g_msm_scalars = nullptr;
        g_msm_n = 0;
    }
}

static void MSM_CPU_BENCH(benchmark::State &state) {
    // state.range(0) is the power of 2, so n = 2^range(0)
    uint64_t power = state.range(0);
    uint64_t n = 1ULL << power;
    
    // Setup test data (done once, cached for subsequent iterations)
    setup_msm_data(n);
    
    // AltBn128::G1 is Curve<RawFqP>, so use that as the template type
    ParallelMultiexp<Curve<RawFqP>> pme(AltBn128::G1);
    AltBn128::G1Point result;
    
    for (auto _ : state) {
        pme.multiexp(result, g_msm_bases, g_msm_scalars, MSM_SCALAR_SIZE, n);
        benchmark::DoNotOptimize(result);
    }
    
    // Report throughput
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["points"] = n;
    state.counters["log2(n)"] = power;
}

// Register MSM benchmarks: argument is the power of 2 (n = 2^arg)
// Default: 22, 23, 24, 25 -> 4M, 8M, 16M, 32M points
BENCHMARK(MSM_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->DenseRange(22, 25);  // 2^22 to 2^25

int main(int argc, char** argv) {
    // Run benchmarks
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    
    // Cleanup MSM test data
    cleanup_msm_data();
    
    return 0;
}