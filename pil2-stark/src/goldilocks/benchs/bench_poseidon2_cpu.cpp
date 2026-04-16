#include <benchmark/benchmark.h>
#include <iostream>
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/merklehash_goldilocks.hpp"
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <math.h>
#include "omp.h"

#define NUM_HASHES 2097152
#define NCOLS_HASH 128
#define NROWS_HASH (1 << 23)

static void POSEIDON2_BENCH_FULL(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)Poseidon2Goldilocks<16>::SPONGE_WIDTH;
    Goldilocks::Element *x = new Goldilocks::Element[input_size];
    Goldilocks::Element *result = new Goldilocks::Element[input_size];

    for (uint64_t i = 0; i < input_size; i++)
    {
        x[i] = Goldilocks::fromU64(i);
    }

    // Benchmark
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            Poseidon2Goldilocks<16>::hashFullResult(&result[i * Poseidon2Goldilocks<16>::SPONGE_WIDTH], &x[i * Poseidon2Goldilocks<16>::SPONGE_WIDTH], Poseidon2Mode::Scalar);
        }
    }
    // Check poseidon results poseidon ( 0 1 2 3 4 5 6 7 8 9 10 11 )
    delete[] x;
    delete[] result;
    // Rate = time to process 1 posseidon per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

#ifdef __AVX2__
static void POSEIDON2_BENCH_FULL_AVX(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)Poseidon2Goldilocks<16>::SPONGE_WIDTH;
    Goldilocks::Element *x = new Goldilocks::Element[input_size];
    Goldilocks::Element *result = new Goldilocks::Element[input_size];

    for (uint64_t i = 0; i < input_size; i++)
    {
        x[i] = Goldilocks::fromU64(i);
    }

    // Benchmark
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            Poseidon2Goldilocks<16>::hashFullResult(&result[i * Poseidon2Goldilocks<16>::SPONGE_WIDTH], &x[i * Poseidon2Goldilocks<16>::SPONGE_WIDTH], Poseidon2Mode::Avx);
        }
    }
    delete[] x;
    delete[] result;
    // Rate = time to process 1 posseidon per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

// NOTE: POSEIDON2_BENCH_FULL_AVX_BATCH was removed — it benched the private
// primitive hash_full_result_batch_avx, which has no public Mode. The AVX
// batch hash path is still exercised by the merkletree benches running in
// AvxBatch mode (MERKLETREE_BATCH_BENCH_AVX and the _AR variants).
#endif

// NOTE: POSEIDON_BENCH_FULL_AVX512 / POSEIDON_BENCH_AVX512 / LINEAR_HASH_BENCH_AVX512
// / MERKLETREE_BENCH_AVX512 were removed. Single-sponge AVX512 is intentionally
// unimplemented (see Poseidon2Mode enum comment): at state sizes 4..16 the
// 8-lane register offers no meaningful gain over the 4-lane AVX2 path. The real
// AVX512 win-case — 8 parallel sponges — is exercised by MERKLETREE_BATCH_BENCH_AVX512.

static void POSEIDON2_BENCH(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)Poseidon2Goldilocks<16>::SPONGE_WIDTH;
    uint64_t output_size = (uint64_t)NUM_HASHES * (uint64_t)Poseidon2Goldilocks<16>::CAPACITY;
    Goldilocks::Element *x = new Goldilocks::Element[input_size];
    Goldilocks::Element *result = new Goldilocks::Element[output_size];

    for (uint64_t i = 0; i < input_size; i++)
    {
        x[i] = Goldilocks::fromU64(i);
    }

    // Benchmark
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            Poseidon2Goldilocks<16>::hash((Goldilocks::Element(&)[Poseidon2Goldilocks<16>::CAPACITY])result[i * Poseidon2Goldilocks<16>::CAPACITY], (Goldilocks::Element(&)[Poseidon2Goldilocks<16>::SPONGE_WIDTH])x[i * Poseidon2Goldilocks<16>::SPONGE_WIDTH], Poseidon2Mode::Scalar);
        }
    }

    delete[] x;
    delete[] result;
    // Rate = time to process 1 posseidon per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}

#ifdef __AVX2__
static void POSEIDON2_BENCH_AVX(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)Poseidon2Goldilocks<16>::SPONGE_WIDTH;
    uint64_t output_size = (uint64_t)NUM_HASHES * (uint64_t)Poseidon2Goldilocks<16>::CAPACITY;
    Goldilocks::Element *x = new Goldilocks::Element[input_size];
    Goldilocks::Element *result = new Goldilocks::Element[output_size];

    for (uint64_t i = 0; i < input_size; i++)
    {
        x[i] = Goldilocks::fromU64(i);
    }

    // Benchmark
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            Poseidon2Goldilocks<16>::hash((Goldilocks::Element(&)[Poseidon2Goldilocks<16>::CAPACITY])result[i * Poseidon2Goldilocks<16>::CAPACITY], (Goldilocks::Element(&)[Poseidon2Goldilocks<16>::SPONGE_WIDTH])x[i * Poseidon2Goldilocks<16>::SPONGE_WIDTH], Poseidon2Mode::Avx);
        }
    }

    delete[] x;
    delete[] result;
    // Rate = time to process 1 posseidon per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (double)NUM_HASHES / (double)state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter(input_size * sizeof(uint64_t), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
}
#endif

static void LINEAR_HASH_BENCH(benchmark::State &state)
{
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH];
    Goldilocks::Element *result = new Goldilocks::Element[(uint64_t)HASH_SIZE * (uint64_t)NROWS_HASH];

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS
    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    // Benchmark
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NROWS_HASH; i++)
        {
            Poseidon2Goldilocks<16>::linearHash(&result[i * HASH_SIZE], &cols[i * NCOLS_HASH], NCOLS_HASH, Poseidon2Mode::Scalar);
        }
    }

    // Rate = time to process 1 linear hash per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (double)NROWS_HASH * (double)ceil((double)NCOLS_HASH / (double)Poseidon2Goldilocks<16>::RATE) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

    delete[] cols;
    delete[] result;
}

#ifdef __AVX2__
static void LINEAR_HASH_BENCH_AVX(benchmark::State &state)
{
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH];
    Goldilocks::Element *result = new Goldilocks::Element[(uint64_t)HASH_SIZE * (uint64_t)NROWS_HASH];

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS
    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    // Benchmark
    for (auto _ : state)
    {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NROWS_HASH; i++)
        {
            Poseidon2Goldilocks<16>::linearHash(&result[i * HASH_SIZE], &cols[i * NCOLS_HASH], NCOLS_HASH, Poseidon2Mode::Avx);
        }
    }

    // Rate = time to process 1 linear hash per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (double)NROWS_HASH * (double)ceil((double)NCOLS_HASH / (double)Poseidon2Goldilocks<16>::RATE) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

    delete[] cols;
    delete[] result;
}
#endif

static void MERKLETREE_BENCH(benchmark::State &state)
{
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH];

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS
    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(NROWS_HASH);
    Goldilocks::Element *tree = new Goldilocks::Element[numElementsTree];

    // Benchmark
    for (auto _ : state)
    {
        Poseidon2Goldilocks<16>::merkletree(tree, cols, NCOLS_HASH, NROWS_HASH, state.range(0), /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::Scalar);
    }
    Goldilocks::Element root[4];
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    // Rate = time to process 1 linear hash per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (((double)NROWS_HASH * (double)ceil((double)NCOLS_HASH / (double)Poseidon2Goldilocks<16>::RATE)) + log2(NROWS_HASH)) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
    delete[] cols;
    delete[] tree;
}

#ifdef __AVX2__
static void MERKLETREE_BENCH_AVX(benchmark::State &state)
{
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH];

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS
    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(NROWS_HASH);
    Goldilocks::Element *tree = new Goldilocks::Element[numElementsTree];

    // Benchmark
    for (auto _ : state)
    {
        Poseidon2Goldilocks<16>::merkletree(tree, cols, NCOLS_HASH, NROWS_HASH, /*arity=*/3, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::Avx);
    }
    Goldilocks::Element root[4];
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    // check results
    // assert(Goldilocks::toU64(root[0]) == 0Xc935fb33cd86c0b8);
    // assert(Goldilocks::toU64(root[1]) == 0X906753f66aa2791d);
    // assert(Goldilocks::toU64(root[2]) == 0X3f6163b1b58a6ed7);
    // assert(Goldilocks::toU64(root[3]) == 0Xbd575d9ed19d18c2);

    // Rate = time to process 1 linear hash per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (((double)NROWS_HASH * (double)ceil((double)NCOLS_HASH / (double)Poseidon2Goldilocks<16>::RATE)) + log2(NROWS_HASH)) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
    delete[] cols;
    delete[] tree;
}
#endif

#ifdef __AVX2__
static void MERKLETREE_BATCH_BENCH_AVX(benchmark::State &state)
{
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH];

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS
    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(NROWS_HASH);
    Goldilocks::Element *tree = new Goldilocks::Element[numElementsTree];

    // Benchmark
    for (auto _ : state)
    {
        Poseidon2Goldilocks<16>::merkletree(tree, cols, NCOLS_HASH, NROWS_HASH, /*arity=*/3, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::AvxBatch);
    }
    Goldilocks::Element root[4];
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    // check results
    // assert(Goldilocks::toU64(root[0]) == 0X9ce696d26651e066);
    // assert(Goldilocks::toU64(root[1]) == 0Xc7f662974b960728);
    // assert(Goldilocks::toU64(root[2]) == 0Xad8a489fec5811a1);
    // assert(Goldilocks::toU64(root[3]) == 0Xd34d83367c86e333);

    // Rate = time to process 1 linear hash per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (((double)NROWS_HASH * (double)ceil((double)NCOLS_HASH / (double)Poseidon2Goldilocks<16>::RATE)) + log2(NROWS_HASH)) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
    delete[] cols;
    delete[] tree;
}
#endif
#ifdef __AVX512__
static void MERKLETREE_BATCH_BENCH_AVX512(benchmark::State &state)
{
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH];

    // Test vector: Fibonacci series on the columns and increase the initial values to the right,
    // 1 2 3 4  5  6  ... NUM_COLS
    // 1 2 3 4  5  6  ... NUM_COLS
    // 2 4 6 8  10 12 ... NUM_COLS + NUM_COLS
    // 3 6 9 12 15 18 ... NUM_COLS + NUM_COLS + NUM_COLS
    for (uint64_t i = 0; i < NCOLS_HASH; i++)
    {
        cols[i] = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + NCOLS_HASH] = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < NROWS_HASH; j++)
    {
        for (uint64_t i = 0; i < NCOLS_HASH; i++)
        {
            cols[j * NCOLS_HASH + i] = cols[(j - 2) * NCOLS_HASH + i] + cols[(j - 1) * NCOLS_HASH + i];
        }
    }

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(NROWS_HASH);
    Goldilocks::Element *tree = new Goldilocks::Element[numElementsTree];

    // Benchmark
    for (auto _ : state)
    {
        Poseidon2Goldilocks<16>::merkletree(tree, cols, NCOLS_HASH, NROWS_HASH, /*arity=*/3, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::Avx512Batch);
    }
    Goldilocks::Element root[4];
    MerklehashGoldilocks::root(&(root[0]), tree, numElementsTree);

    // check results
    // assert(Goldilocks::toU64(root[0]) == 0X9ce696d26651e066);
    // assert(Goldilocks::toU64(root[1]) == 0Xc7f662974b960728);
    // assert(Goldilocks::toU64(root[2]) == 0Xad8a489fec5811a1);
    // assert(Goldilocks::toU64(root[3]) == 0Xd34d83367c86e333);

    // Rate = time to process 1 linear hash per core
    // BytesProcessed = total bytes processed per second on every iteration
    int threads_core = 2 * state.range(0) / omp_get_max_threads(); // we assume hyperthreading
    state.counters["Rate"] = benchmark::Counter(threads_core * (((double)NROWS_HASH * (double)ceil((double)NCOLS_HASH / (double)Poseidon2Goldilocks<16>::RATE)) + log2(NROWS_HASH)) / state.range(0), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
    state.counters["BytesProcessed"] = benchmark::Counter((uint64_t)NROWS_HASH * (uint64_t)NCOLS_HASH * sizeof(Goldilocks::Element), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
    delete[] cols;
    delete[] tree;
}
#endif


static void GRINDING_BENCH_CPU(benchmark::State &state)
{
    uint32_t n_bits = state.range(0);
    
    // Create different input for each iteration
    uint64_t iteration = 0;
        
    for (auto _ : state)
    {
        // Generate different input for each iteration based on iteration counter
        iteration++;
        uint64_t in[3];
        for (int i = 0; i < 3; i++)
        {
            in[i] = (iteration * 1000 + i) * 123456789ULL;
        }
        
        uint64_t nonce = UINT64_MAX;
        Poseidon2Goldilocks<4>::grinding(nonce, in, n_bits);

        iteration++;
    }
}

// ---------------------------------------------------------------------------
// Benchmark coverage for widths {4,8,12}, arities {3,4}, and standalone INTT.
// ---------------------------------------------------------------------------

// Smaller row count for scalar merkletree at non-W16 widths — keeps baseline
// runtime tolerable. AVX batch benches at the same size so seq↔avx comparisons
// are meaningful.

#define NROWS_MERKLETREE_SMALL (1 << 18)

template<uint32_t W>
static void POSEIDON2_BENCH_FULL_W(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)Poseidon2Goldilocks<W>::SPONGE_WIDTH;
    Goldilocks::Element *x = new Goldilocks::Element[input_size];
    Goldilocks::Element *result = new Goldilocks::Element[input_size];
    for (uint64_t i = 0; i < input_size; i++) x[i] = Goldilocks::fromU64(i);

    for (auto _ : state) {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++) {
            Poseidon2Goldilocks<W>::hashFullResult(
                &result[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                &x[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                Poseidon2Mode::Scalar);
        }
    }
    delete[] x; delete[] result;
}

#ifdef __AVX2__
template<uint32_t W>
static void POSEIDON2_BENCH_FULL_AVX_W(benchmark::State &state)
{
    uint64_t input_size = (uint64_t)NUM_HASHES * (uint64_t)Poseidon2Goldilocks<W>::SPONGE_WIDTH;
    Goldilocks::Element *x = new Goldilocks::Element[input_size];
    Goldilocks::Element *result = new Goldilocks::Element[input_size];
    for (uint64_t i = 0; i < input_size; i++) x[i] = Goldilocks::fromU64(i);

    for (auto _ : state) {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NUM_HASHES; i++) {
            Poseidon2Goldilocks<W>::hashFullResult(
                &result[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                &x[i * Poseidon2Goldilocks<W>::SPONGE_WIDTH],
                Poseidon2Mode::Avx);
        }
    }
    delete[] x; delete[] result;
}
#endif

// NOTE: W=4 has RATE = 0, so linear_hash never terminates. Only W∈{8,12} are
// exercised here; W=16 is covered by LINEAR_HASH_BENCH / LINEAR_HASH_BENCH_AVX.
template<uint32_t W>
static void LINEAR_HASH_BENCH_W(benchmark::State &state)
{
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH];
    Goldilocks::Element *result = new Goldilocks::Element[(uint64_t)HASH_SIZE * (uint64_t)NROWS_HASH];
    for (uint64_t i = 0; i < (uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH; i++)
        cols[i] = Goldilocks::fromU64(i + 1);

    for (auto _ : state) {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NROWS_HASH; i++) {
            Poseidon2Goldilocks<W>::linearHash(&result[i * HASH_SIZE], &cols[i * NCOLS_HASH], NCOLS_HASH, Poseidon2Mode::Scalar);
        }
    }
    delete[] cols; delete[] result;
}

#ifdef __AVX2__
template<uint32_t W>
static void LINEAR_HASH_BENCH_AVX_W(benchmark::State &state)
{
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH];
    Goldilocks::Element *result = new Goldilocks::Element[(uint64_t)HASH_SIZE * (uint64_t)NROWS_HASH];
    for (uint64_t i = 0; i < (uint64_t)NCOLS_HASH * (uint64_t)NROWS_HASH; i++)
        cols[i] = Goldilocks::fromU64(i + 1);

    for (auto _ : state) {
#pragma omp parallel for num_threads(state.range(0)) schedule(static)
        for (uint64_t i = 0; i < NROWS_HASH; i++) {
            Poseidon2Goldilocks<W>::linearHash(&result[i * HASH_SIZE], &cols[i * NCOLS_HASH], NCOLS_HASH, Poseidon2Mode::Avx);
        }
    }
    delete[] cols; delete[] result;
}
#endif

template<uint32_t W, uint64_t ARITY>
static void MERKLETREE_BENCH_AR(benchmark::State &state)
{
    uint64_t nrows = NROWS_MERKLETREE_SMALL;
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * nrows];
    for (uint64_t i = 0; i < (uint64_t)NCOLS_HASH * nrows; i++)
        cols[i] = Goldilocks::fromU64(i + 1);
    uint64_t numElems = MerklehashGoldilocks::getTreeNumElements(nrows, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];

    for (auto _ : state) {
        Poseidon2Goldilocks<W>::merkletree(tree, cols, NCOLS_HASH, nrows, ARITY, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::Scalar);
    }
    delete[] cols; delete[] tree;
}

#ifdef __AVX2__
template<uint32_t W, uint64_t ARITY>
static void MERKLETREE_BATCH_BENCH_AVX_AR(benchmark::State &state)
{
    uint64_t nrows = NROWS_MERKLETREE_SMALL;
    Goldilocks::Element *cols = new Goldilocks::Element[(uint64_t)NCOLS_HASH * nrows];
    for (uint64_t i = 0; i < (uint64_t)NCOLS_HASH * nrows; i++)
        cols[i] = Goldilocks::fromU64(i + 1);
    uint64_t numElems = MerklehashGoldilocks::getTreeNumElements(nrows, ARITY);
    Goldilocks::Element *tree = new Goldilocks::Element[numElems];

    for (auto _ : state) {
        Poseidon2Goldilocks<W>::merkletree(tree, cols, NCOLS_HASH, nrows, ARITY, /*nThreads=*/0, /*dim=*/1, Poseidon2Mode::AvxBatch);
    }
    delete[] cols; delete[] tree;
}
#endif

BENCHMARK(POSEIDON2_BENCH_FULL)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();

#ifdef __AVX2__
BENCHMARK(POSEIDON2_BENCH_FULL_AVX)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();
#endif

BENCHMARK(POSEIDON2_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();

#ifdef __AVX2__
BENCHMARK(POSEIDON2_BENCH_AVX)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();
#endif
BENCHMARK(LINEAR_HASH_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();
#ifdef __AVX2__
BENCHMARK(LINEAR_HASH_BENCH_AVX)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();
#endif
BENCHMARK(MERKLETREE_BENCH)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();

#ifdef __AVX2__
BENCHMARK(MERKLETREE_BENCH_AVX)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();
#endif
#ifdef __AVX2__
BENCHMARK(MERKLETREE_BATCH_BENCH_AVX)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();
#endif

#ifdef __AVX512__
BENCHMARK(MERKLETREE_BATCH_BENCH_AVX512)
    ->Unit(benchmark::kMicrosecond)
    ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2)
    ->UseRealTime();
#endif


BENCHMARK(GRINDING_BENCH_CPU)
    ->Unit(benchmark::kMillisecond)
    ->Arg(20)
    ->Arg(21)
    ->Arg(22)
    ->Arg(24)
    ->Arg(25)
    ->UseRealTime();

// ---- Step 0.6 registrations ----
// Use ->Name(...) with explicit suffixes so bench names contain no spaces
// (default BENCHMARK_TEMPLATE formatting "<12, 3>" breaks bench_compare.sh's
// whitespace-splitting awk parser).

#define REG_W(NAME, W)                                                             \
    BENCHMARK_TEMPLATE(NAME, W)->Name(#NAME "_" #W)                                \
        ->Unit(benchmark::kMicrosecond)                                            \
        ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2) \
        ->UseRealTime();

#define REG_AR(NAME, W, AR)                                                        \
    BENCHMARK_TEMPLATE(NAME, W, AR)->Name(#NAME "_" #W "_" #AR)                    \
        ->Unit(benchmark::kMicrosecond)                                            \
        ->DenseRange(omp_get_max_threads() / 2, omp_get_max_threads(), omp_get_max_threads() / 2) \
        ->UseRealTime();

REG_W(POSEIDON2_BENCH_FULL_W, 4)
REG_W(POSEIDON2_BENCH_FULL_W, 8)
REG_W(POSEIDON2_BENCH_FULL_W, 12)

#ifdef __AVX2__
REG_W(POSEIDON2_BENCH_FULL_AVX_W, 4)
REG_W(POSEIDON2_BENCH_FULL_AVX_W, 8)
REG_W(POSEIDON2_BENCH_FULL_AVX_W, 12)
#endif

REG_W(LINEAR_HASH_BENCH_W, 8)
REG_W(LINEAR_HASH_BENCH_W, 12)

#ifdef __AVX2__
REG_W(LINEAR_HASH_BENCH_AVX_W, 8)
REG_W(LINEAR_HASH_BENCH_AVX_W, 12)
#endif

REG_AR(MERKLETREE_BENCH_AR, 12, 3)
REG_AR(MERKLETREE_BENCH_AR, 16, 4)

#ifdef __AVX2__
REG_AR(MERKLETREE_BATCH_BENCH_AVX_AR, 12, 3)
REG_AR(MERKLETREE_BATCH_BENCH_AVX_AR, 16, 4)
#endif

#undef REG_W
#undef REG_AR

