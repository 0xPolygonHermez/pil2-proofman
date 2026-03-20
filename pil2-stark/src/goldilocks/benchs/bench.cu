#include <benchmark/benchmark.h>
#include <iostream>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/poseidon2_goldilocks.cuh"
#include "../src/ntt_goldilocks.cuh"
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
    Poseidon2GoldilocksGPU<16>::initPoseidon2GPUConstants(&gpu_id, 1);

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

    //print the root of the tree
    uint64_t root[HASH_SIZE];
    cudaMemcpy(&root[0], d_tree + (tree_size - HASH_SIZE), HASH_SIZE * sizeof(gl64_t), cudaMemcpyDeviceToHost);
    //for(int i = 0; i < HASH_SIZE; i++)
    //    std::cout << "Root[" << i << "]: " << root[i] << std::endl;
    /*if(state.range(0) == 56){
        assert(root[0] == uint64_t(0x9e1bd81a45f7dedb));
        assert(root[1] == uint64_t(0x27268bc3f7feb493));
        assert(root[2] == uint64_t(0x41618b1ff42048d1));
        assert(root[3] == uint64_t(0x6e093bed170bcb8f));
    }*/
    cudaFree(d_trace);
    cudaFree(d_tree);
    cudaStreamDestroy(stream);
}

static void GRINDING_BENCH_GPU(benchmark::State &state)
{
    // Initialize GPU constants
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int*)&gpu_id));
    Poseidon2GoldilocksGPUGrinding::initPoseidon2GPUConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint32_t n_bits = state.range(0);
    
    // Allocate device memory
    gl64_t *d_in, *d_nonce, *d_nonceBlock;
    CHECKCUDAERR(cudaMalloc((void **)&d_in, 4 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonce, sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_nonceBlock, NONCES_LAUNCH_GRID_SIZE * sizeof(gl64_t)));
    
    // Create different input for each iteration
    Goldilocks::Element h_in[Poseidon2GoldilocksGPUGrinding::SPONGE_WIDTH];
    uint64_t iteration = 0;
        
    for (auto _ : state)
    {
        // Generate different input for each iteration based on iteration counter
        iteration++;
        for (int i = 0; i < (Poseidon2GoldilocksGPUGrinding::SPONGE_WIDTH-1); i++)
        {
            h_in[i] = Goldilocks::fromU64((iteration * 1000 + i) * 123456789ULL);
        }
        CHECKCUDAERR(cudaMemcpy(d_in, h_in, (Poseidon2GoldilocksGPUGrinding::SPONGE_WIDTH-1) * sizeof(gl64_t), cudaMemcpyHostToDevice));
        
        Poseidon2GoldilocksGPUGrinding::grinding((uint64_t *)d_nonce, (uint64_t *)d_nonceBlock, (uint64_t *)d_in, n_bits, stream);
        cudaStreamSynchronize(stream);
        
        //check if d_nonce is valid
        uint64_t h_nonce;
        CHECKCUDAERR(cudaMemcpy(&h_nonce, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        assert(h_nonce != UINT64_MAX);
        iteration++;
    }

    cudaFree(d_in);
    cudaFree(d_nonce);
    cudaFree(d_nonceBlock);
    cudaStreamDestroy(stream);
}

BENCHMARK(LINEAR_HASH12_BENCH_GPU)
    ->Unit(benchmark::kMillisecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK(LINEAR_HASH16_BENCH_GPU)
    ->Unit(benchmark::kMillisecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK(MERKLETREE12_BENCH_GPU)
    ->Unit(benchmark::kMillisecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK(MERKLETREE16_BENCH_GPU)
    ->Unit(benchmark::kMillisecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK(GRINDING_BENCH_GPU)
    ->Unit(benchmark::kMillisecond)
    ->Arg(16)   
    ->Arg(20)   
    ->Arg(23)   
    ->Arg(24)
    ->Arg(25)    
    ->UseRealTime();

static void INTT_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int*)&gpu_id));

    uint64_t n_bits = state.range(0);
    uint64_t domain_size = 1ULL << n_bits;
    uint64_t nCols = 1;

    NTT_Goldilocks_GPU gpu_ntt(24, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_data;
    CHECKCUDAERR(cudaMalloc((void **)&d_data, domain_size * sizeof(gl64_t)));

    dim3 threads(128);
    dim3 blocks((domain_size + threads.x - 1) / threads.x);
    initTrace<<<blocks, threads, 0, stream>>>(d_data, domain_size, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.INTT_inplace(d_data, n_bits, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        gpu_ntt.INTT_inplace(d_data, n_bits, nCols, stream);
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
static void LDE_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int*)&gpu_id));
    NTT_Goldilocks_GPU gpu_ntt(24, 1, &gpu_id);

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
    gpu_ntt.prepare_blocks_trace(d_src, d_flat, nCols, nRows, stream, timer);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.LDE_GPU(d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        gpu_ntt.LDE_GPU(d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTT_Goldilocks_GPU::free_twiddle_factors_and_r(&gpu_id);
}

// Full pipeline benchmark — LDE + merkle tree build (the prover's hot path).
static void LDE_MERKLETREE_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int*)&gpu_id));
    Poseidon2GoldilocksGPU<12>::initPoseidon2GPUConstants(&gpu_id, 1);
    NTT_Goldilocks_GPU gpu_ntt(24, 1, &gpu_id);

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
    gpu_ntt.prepare_blocks_trace(d_src, d_flat, nCols, nRows, stream, timer);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.LDE_MerkleTree_GPU(d_tree, d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, arity, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        gpu_ntt.LDE_MerkleTree_GPU(d_tree, d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, arity, timer, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTT_Goldilocks_GPU::free_twiddle_factors_and_r(&gpu_id);
}

BENCHMARK(LDE_GPU_BENCH)
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

// merkletreeCoalesced benchmark — flat row-major input (used by prover for FRI trees).
// Complements MERKLETREE12_BENCH_GPU which uses block-tiled input (CoalescedBlocks).
static void MERKLETREE_COALESCED12_BENCH_GPU(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int*)&gpu_id));
    Poseidon2GoldilocksGPU<12>::initPoseidon2GPUConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    const uint64_t trace_cols = state.range(0);
    constexpr uint32_t arity = 3;
    const uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(TRACE_NROWS, arity);

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void**)&d_trace, TRACE_NROWS * trace_cols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((TRACE_NROWS + 127) / 128);
    initTrace<<<blk, thr, 0, stream>>>(d_trace, TRACE_NROWS, trace_cols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    Poseidon2GoldilocksGPU<12>::merkletreeCoalesced(
        arity, (uint64_t*)d_tree, (uint64_t*)d_trace, trace_cols, TRACE_NROWS, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state)
    {
        Poseidon2GoldilocksGPU<12>::merkletreeCoalesced(
            arity, (uint64_t*)d_tree, (uint64_t*)d_trace, trace_cols, TRACE_NROWS, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

BENCHMARK(MERKLETREE_COALESCED12_BENCH_GPU)
    ->Unit(benchmark::kMillisecond)
    ->Arg(24)
    ->Arg(36)
    ->Arg(38)
    ->Arg(56)
    ->UseRealTime();

BENCHMARK_MAIN();


//  RUN:
// ./bench --benchmark_filter=POSEIDON