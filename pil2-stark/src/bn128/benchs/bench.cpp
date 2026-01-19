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
#include "fft.hpp"
#include "poseidon2_bn128.hpp"
#include "poseidon_bn128.hpp"
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
    RawFr field;
    RawFr::Element a, b, c;
    
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
    RawFr field;
    RawFr::Element a, b, c;
    
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
    RawFr field;
    RawFr::Element a, b, c;
    
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
    RawFr field;
    RawFr::Element a, c;
    
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
    RawFr field;
    RawFr::Element a, b, c;
    
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
    RawFr field;
    RawFr::Element a, c;
    
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
// We can compare BLS's FR implementation performance with our own RawFr implementation
// the scalar fields are not the same but similar: 255-bit prime for BLS vs 254-bit prime for RawFr
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
// Utilities
// =====================

// Generate random scalars in parallel
static void generate_random_scalars(uint8_t* scalars, uint64_t n, uint64_t scalar_size = 32, int seed = 42) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(seed + tid);
        
        uint64_t start_idx = (tid * n) / omp_get_num_threads();
        uint64_t end_idx = ((tid + 1) * n) / omp_get_num_threads();
        
        for (uint64_t i = start_idx; i < end_idx; i++) {
            uint8_t* scalar = &scalars[i * scalar_size];
            for (size_t j = 0; j < scalar_size; j++) {
                scalar[j] = rng() & 0xFF;
            }
            // Ensure scalar < field order by clearing top bits (~253 bits)
            scalar[scalar_size - 1] &= 0x1F;
        }
    }
}

// Generate curve points in parallel: points[i] = 2^i * G
static void generate_curve_points(AltBn128::G1PointAffine* points, uint64_t n) {
    const int nChunks = omp_get_max_threads();
    std::vector<AltBn128::G1Point> chunkStarts(nChunks);
    
    uint64_t chunkSize = (n + nChunks - 1) / nChunks;
    AltBn128::G1Point acc;
    AltBn128::G1.copy(acc, AltBn128::G1.oneAffine());
    
    for (int c = 0; c < nChunks; c++) {
        AltBn128::G1.copy(chunkStarts[c], acc);
        for (uint64_t j = 0; j < chunkSize && (c * chunkSize + j) < n; j++) {
            AltBn128::G1.dbl(acc, acc);
        }
    }
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t start_idx = tid * chunkSize;
        uint64_t end_idx = std::min(start_idx + chunkSize, n);
        
        AltBn128::G1Point localPoint;
        AltBn128::G1.copy(localPoint, chunkStarts[tid]);
        
        for (uint64_t i = start_idx; i < end_idx; i++) {
            AltBn128::G1.copy(points[i], localPoint);
            AltBn128::G1.dbl(localPoint, localPoint);
        }
    }
}

// =====================
// MSM CPU Benchmark
// =====================

static const uint64_t MSM_SCALAR_SIZE = 32;

static void MSM_CPU_BENCH(benchmark::State &state) {
    uint64_t power = state.range(0);
    uint64_t n = 1ULL << power;
    
    // Allocate data
    AltBn128::G1PointAffine* bases = new AltBn128::G1PointAffine[n];
    uint8_t* scalars = new uint8_t[n * MSM_SCALAR_SIZE];
    
    // Generate test data
    generate_curve_points(bases, n);
    generate_random_scalars(scalars, n, MSM_SCALAR_SIZE, 42);
    
    ParallelMultiexp<Curve<RawFqP>> pme(AltBn128::G1);
    AltBn128::G1Point result;
    
    for (auto _ : state) {
        pme.multiexp(result, bases, scalars, MSM_SCALAR_SIZE, n);
        benchmark::DoNotOptimize(result);
    }
    
    // Cleanup
    delete[] bases;
    delete[] scalars;
    
    // Report throughput
    state.counters["log2(n)"] = power;
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(MSM_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->DenseRange(22, 25);

// =====================
// NTT CPU Benchmark
// =====================

static void NTT_CPU_BENCH(benchmark::State &state) {
    uint64_t power = state.range(0);
    uint64_t n = 1ULL << power;
    
    // Allocate data
    RawFr::Element* data = new RawFr::Element[n];
    
    // Generate random field elements (reuse generate_random_scalars)
    generate_random_scalars(reinterpret_cast<uint8_t*>(data), n, sizeof(RawFr::Element));
    
    FFT<RawFr> fft(n);
    
    for (auto _ : state) {
        fft.fft(data, n);
        benchmark::DoNotOptimize(data);
    }
    
    // Cleanup
    delete[] data;
    
    // Report throughput
    state.counters["log2(n)"] = power;
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(NTT_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->DenseRange(22, 25);

// =====================
// Poseidon2 CPU Benchmark
// =====================

static const int POSEIDON2_NUM_HASHES = 10000;

static void POSEIDON2_SEQ_CPU_BENCH(benchmark::State &state) {
    int t = state.range(0);
    
    RawFr field;
    Poseidon2BN128 poseidon2;
    
    // Initialize state with sequential values
    std::vector<RawFr::Element> hash_state(t);
    for (int i = 0; i < t; i++) {
        field.fromUI(hash_state[i], i);
    }
    
    for (auto _ : state) {
        std::vector<RawFr::Element> state_copy = hash_state;
        for(int i = 0; i < POSEIDON2_NUM_HASHES; i++){
            poseidon2.hash(state_copy);
        }
        benchmark::DoNotOptimize(state_copy);
    }
    
    state.counters["t"] = t;
    state.counters["hashes"] = POSEIDON2_NUM_HASHES;
    state.SetItemsProcessed(state.iterations() * POSEIDON2_NUM_HASHES);
}

BENCHMARK(POSEIDON2_SEQ_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Args({2})
    ->Args({3})
    ->Args({4})
    ->Args({8})
    ->Args({12})
    ->Args({16});

// =====================
// Poseidon CPU Benchmark
// =====================

static const int POSEIDON_NUM_HASHES = 10000;

static void POSEIDON_SEQ_CPU_BENCH(benchmark::State &state) {
    int t = state.range(0);
    
    RawFr field;
    PoseidonBN128 poseidon;
    
    std::vector<RawFr::Element> hash_state(t);
    for (int i = 0; i < t; i++) {
        field.fromUI(hash_state[i], i);
    }
    
    for (auto _ : state) {
        std::vector<RawFr::Element> state_copy = hash_state;
        for(int i = 0; i < POSEIDON_NUM_HASHES; i++){
            poseidon.hash(state_copy);
        }
        benchmark::DoNotOptimize(state_copy);
    }
    
    state.counters["t"] = t;
    state.counters["hashes"] = POSEIDON_NUM_HASHES;
    state.SetItemsProcessed(state.iterations() * POSEIDON_NUM_HASHES);
}

BENCHMARK(POSEIDON_SEQ_CPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Args({2})
    ->Args({3})
    ->Args({4})
    ->Args({5})
    ->Args({6})
    ->Args({7})
    ->Args({8})
    ->Args({9})
    ->Args({10})
    ->Args({11})
    ->Args({12})
    ->Args({13})
    ->Args({14})
    ->Args({15})
    ->Args({16})
    ->Args({17});

BENCHMARK_MAIN();