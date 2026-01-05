#include <benchmark/benchmark.h>
#include <gmp.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "../fr.hpp"
#include "../fq.hpp"
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

int main(int argc, char** argv) {
    // Run benchmarks
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}