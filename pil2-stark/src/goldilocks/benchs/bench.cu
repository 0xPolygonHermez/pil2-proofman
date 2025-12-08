#include <benchmark/benchmark.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/gl64_t.cuh"
#include "../src/poseidon2_goldilocks.cuh"
#include "../utils/cuda_utils.hpp"
#include "../src/merklehash_goldilocks.hpp"

#include <math.h> /* ceil */
#include "omp.h"


#define TRACE_NROWS  (1 << 23)

__global__ void initTrace(gl64_t *d_trace, uint64_t nRows, uint64_t nCols)
{
   uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < nRows)
   {
       for(int j = 0; j < nCols; j++)
           d_trace[idx * nCols + j]=  uint64_t(idx + j);
   }
}


static void LINEAR_HASH12_BENCH_GPU(benchmark::State &state)
{
    // Initialize GPU constants
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initPoseidon2GPUConstants(&gpu_id, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //performance test
    gl64_t *d_trace, *d_hash_output;
    uint64_t trace_cols = state.range(0);
    uint64_t trace_size = TRACE_NROWS * trace_cols;
    cudaMalloc((void **)&d_trace, trace_size * sizeof(gl64_t));
    cudaMalloc((void **)&d_hash_output, TRACE_NROWS * Poseidon2GoldilocksGPU<12>::CAPACITY * sizeof(gl64_t));
    dim3 threads(128);
    dim3 blocks((TRACE_NROWS + threads.x - 1) / threads.x);

    //initialize trace
    initTrace<<<blocks, threads, 0, stream>>>(d_trace, TRACE_NROWS, trace_cols);
    cudaStreamSynchronize(stream);

    for (auto _ : state)
    {
        Poseidon2GoldilocksGPU<12>::linearHashCoalescedBlocks((uint64_t *)d_hash_output, (uint64_t *)d_trace, trace_cols, TRACE_NROWS,stream);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_trace);
    cudaFree(d_hash_output);
    cudaStreamDestroy(stream);

}

static void LINEAR_HASH16_BENCH_GPU(benchmark::State &state)
{
    // Initialize GPU constants
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initPoseidon2GPUConstants(&gpu_id, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //performance test
    gl64_t *d_trace, *d_hash_output;
    uint64_t trace_cols = state.range(0);
    uint64_t trace_size = TRACE_NROWS * trace_cols;
    cudaMalloc((void **)&d_trace, trace_size * sizeof(gl64_t));
    cudaMalloc((void **)&d_hash_output, TRACE_NROWS * Poseidon2GoldilocksGPU<16>::CAPACITY * sizeof(gl64_t));
    dim3 threads(128);
    dim3 blocks((TRACE_NROWS + threads.x - 1) / threads.x);

    //initialize trace
    initTrace<<<blocks, threads, 0, stream>>>(d_trace, TRACE_NROWS, trace_cols);
    cudaStreamSynchronize(stream);

    for (auto _ : state)
    {      
        Poseidon2GoldilocksGPU<16>::linearHashCoalescedBlocks((uint64_t *)d_hash_output, (uint64_t *)d_trace, trace_cols, TRACE_NROWS,stream);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_trace);
    cudaFree(d_hash_output);
    cudaStreamDestroy(stream);

}

//merkletree
static void MERKLETREE12_BENCH_GPU(benchmark::State &state)
{
    // Initialize GPU constants
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initPoseidon2GPUConstants(&gpu_id, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //performance test
    gl64_t *d_trace, *d_tree;
    uint64_t trace_cols = state.range(0);
    uint64_t trace_size = TRACE_NROWS * trace_cols;
    uint32_t arity = 3;
    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(TRACE_NROWS, arity);
    cudaMalloc((void **)&d_trace, trace_size * sizeof(gl64_t));
    cudaMalloc((void **)&d_tree, tree_size * sizeof(gl64_t));
    dim3 threads(128);
    dim3 blocks((TRACE_NROWS + threads.x - 1) / threads.x);

    //initialize trace
    initTrace<<<blocks, threads, 0, stream>>>(d_trace, TRACE_NROWS, trace_cols);
    cudaStreamSynchronize(stream);

    for (auto _ : state)
    {
        Poseidon2GoldilocksGPU<12>::merkletreeCoalescedBlocks(arity, (uint64_t*) d_tree, (uint64_t *)d_trace, trace_cols, TRACE_NROWS, stream);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_trace);
    cudaFree(d_tree);
    cudaStreamDestroy(stream);
}

static void MERKLETREE16_BENCH_GPU(benchmark::State &state)
{
    // Initialize GPU constants
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<16>::initPoseidon2GPUConstants(&gpu_id, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //performance test
    gl64_t *d_trace, *d_tree;
    uint64_t trace_cols = state.range(0);
    uint64_t trace_size = TRACE_NROWS * trace_cols;
    uint32_t arity = 4;
    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(TRACE_NROWS, arity);
    cudaMalloc((void **)&d_trace, trace_size * sizeof(gl64_t));
    cudaMalloc((void **)&d_tree, tree_size * sizeof(gl64_t));
    dim3 threads(128);
    dim3 blocks((TRACE_NROWS + threads.x - 1) / threads.x);

    //initialize trace
    initTrace<<<blocks, threads, 0, stream>>>(d_trace, TRACE_NROWS, trace_cols);
    cudaStreamSynchronize(stream);

    for (auto _ : state)
    {
        Poseidon2GoldilocksGPU<16>::merkletreeCoalescedBlocks(arity, (uint64_t*) d_tree, (uint64_t *)d_trace, trace_cols, TRACE_NROWS, stream);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_trace);
    cudaFree(d_tree);
    cudaStreamDestroy(stream);
}

BENCHMARK(LINEAR_HASH12_BENCH_GPU)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK(LINEAR_HASH16_BENCH_GPU)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK(MERKLETREE12_BENCH_GPU)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK(MERKLETREE16_BENCH_GPU)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK_MAIN();


//  RUN:
// ./bench --benchmark_filter=POSEIDON