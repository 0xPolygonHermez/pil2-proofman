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

__global__ void init_array_gl64(gl64_t *arr, uint64_t nRows, uint64_t nCols)
{
   uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < nRows)
   {
       for(int j = 0; j < nCols; j++)
           arr[idx * nCols + j]=  uint64_t(idx + j);
   }
}


static void LINEAR_HASH_BENCH_GPU(benchmark::State &state)
{
    // Initialize GPU constants
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU::init_gpu_const_2(&gpu_id, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //performance test
    gl64_t *d_trace, *d_hash_output;
    uint64_t trace_cols = state.range(0);
    uint64_t trace_size = TRACE_NROWS * trace_cols;
    cudaMalloc((void **)&d_trace, trace_size * sizeof(gl64_t));
    cudaMalloc((void **)&d_hash_output, TRACE_NROWS * CAPACITY * sizeof(gl64_t));
    dim3 threads(128);
    dim3 blocks((TRACE_NROWS + threads.x - 1) / threads.x);

    //initialize trace
    init_array_gl64<<<blocks, threads, 0, stream>>>(d_trace, TRACE_NROWS, trace_cols);
    cudaStreamSynchronize(stream);

    for (auto _ : state)
    {
        linear_hash_gpu_coalesced_2_blocks<<<blocks, threads, threads.x * SPONGE_WIDTH * sizeof(gl64_t), stream>>>((uint64_t *)d_hash_output, (uint64_t *)d_trace, trace_cols, TRACE_NROWS);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_trace);
    cudaFree(d_hash_output);
    cudaStreamDestroy(stream);

}
//merkletree
static void MERKLETREE_BENCH_GPU(benchmark::State &state)
{
    // Initialize GPU constants
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU::init_gpu_const_2(&gpu_id, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //performance test
    gl64_t *d_trace, *d_tree;
    uint64_t trace_cols = state.range(0);
    uint64_t trace_size = TRACE_NROWS * trace_cols;
    uint32_t arity = SPONGE_WIDTH >> 2;  // arity is 4 for SPONGE_WIDTH=16
    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(TRACE_NROWS, arity);
    cudaMalloc((void **)&d_trace, trace_size * sizeof(gl64_t));
    cudaMalloc((void **)&d_tree, tree_size * sizeof(gl64_t));
    dim3 threads(128);
    dim3 blocks((TRACE_NROWS + threads.x - 1) / threads.x);

    //initialize trace
    init_array_gl64<<<blocks, threads, 0, stream>>>(d_trace, TRACE_NROWS, trace_cols);
    cudaStreamSynchronize(stream);

    for (auto _ : state)
    {
        Poseidon2GoldilocksGPU::merkletree_cuda_coalesced_blocks(arity, (uint64_t*) d_tree, (uint64_t *)d_trace, trace_cols, TRACE_NROWS, stream);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_trace);
    cudaFree(d_tree);
    cudaStreamDestroy(stream);

}

BENCHMARK(LINEAR_HASH_BENCH_GPU)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->UseRealTime();

BENCHMARK(MERKLETREE_BENCH_GPU)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->UseRealTime();

BENCHMARK_MAIN();


//  RUN:
// ./bench --benchmark_filter=POSEIDON