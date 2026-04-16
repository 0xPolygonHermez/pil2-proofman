#include <benchmark/benchmark.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/poseidon2_goldilocks.cuh"
#include "../src/ntt_goldilocks.cuh"
#include "../src/gl64_tooling.cuh"
#include "../utils/cuda_utils.hpp"
#include "../src/merklehash_goldilocks.hpp"

#include <math.h> /* ceil */
#include "omp.h"


#define TRACE_NROWS  (1 << 23)

static __global__ void initTrace(gl64_t *d_trace, uint64_t nRows, uint64_t nCols)
{
   uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < nRows)
   {
       for(int j = 0; j < nCols; j++)
           d_trace[idx * nCols + j]=  uint64_t(idx + j);
   }
}

static void INTT_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int*)&gpu_id));

    uint64_t n_bits = state.range(0);
    uint64_t domain_size = 1ULL << n_bits;
    uint64_t nCols = 1;

    NTTGoldilocksGPU gpu_ntt(24, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_data;
    CHECKCUDAERR(cudaMalloc((void **)&d_data, domain_size * sizeof(gl64_t)));

    dim3 threads(128);
    dim3 blocks((domain_size + threads.x - 1) / threads.x);
    initTrace<<<blocks, threads, 0, stream>>>(d_data, domain_size, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.INTT(d_data, n_bits, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        gpu_ntt.INTT(d_data, n_bits, nCols, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_data));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

BENCHMARK(INTT_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->Arg(20)
    ->Arg(22)
    ->Arg(24)
    ->UseRealTime();

// LDE benchmark — measures only the NTT+coset-extension step (no merkle build).
// d_src is block-tiled (as produced by prepare_blocks_trace); allocated once, never modified.
static void LDE_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int*)&gpu_id));
    NTTGoldilocksGPU gpu_ntt(24, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    constexpr uint64_t n_bits     = 20;
    constexpr uint64_t n_bits_ext = 22; // 4× blowup
    const uint64_t nRows          = 1ULL << n_bits;
    const uint64_t nRows_ext      = 1ULL << n_bits_ext;
    const uint64_t nCols          = state.range(0);

    gl64_t *d_flat, *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat, nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_src,  nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_dst,  nRows_ext * nCols * sizeof(gl64_t)));

    dim3 thr(128), blk((nRows + 127) / 128);
    initTrace<<<blk, thr, 0, stream>>>(d_flat, nRows, nCols);
    fromRowMajorToTiled(nRows, nCols, d_flat, d_src, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.LDE(d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        gpu_ntt.LDE(d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Full pipeline benchmark — LDE + merkle tree build (the prover's hot path).
static void LDE_MERKLETREE_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int*)&gpu_id));
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(24, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    constexpr uint64_t n_bits     = 20;
    constexpr uint64_t n_bits_ext = 22;
    const uint64_t nRows          = 1ULL << n_bits;
    const uint64_t nRows_ext      = 1ULL << n_bits_ext;
    const uint64_t nCols          = state.range(0);
    constexpr uint32_t arity      = 3;

    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(nRows_ext, arity);

    gl64_t *d_flat, *d_src, *d_dst;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat, nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_src,  nRows * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_dst,  nRows_ext * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree, tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((nRows + 127) / 128);
    initTrace<<<blk, thr, 0, stream>>>(d_flat, nRows, nCols);
    fromRowMajorToTiled(nRows, nCols, d_flat, d_src, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.LDE(d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
    buildMerkleTreeGPU(arity, (uint64_t*)d_tree, (uint64_t*)d_dst, nCols, nRows_ext, Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        gpu_ntt.LDE(d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
        buildMerkleTreeGPU(arity, (uint64_t*)d_tree, (uint64_t*)d_dst, nCols, nRows_ext, Layout::Tiles, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

BENCHMARK(LDE_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK(LDE_MERKLETREE_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

// merkletree benchmark — flat row-major input (used by prover for FRI trees).

BENCHMARK_MAIN();
