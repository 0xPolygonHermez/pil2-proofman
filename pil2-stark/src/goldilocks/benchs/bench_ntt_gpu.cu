// ---------------------------------------------------------------------------
// bench_ntt_gpu.cu -- GPU NTT/INTT/LDE benchmarks
//
// Unified parameters for apples-to-apples comparison with CPU benchmarks:
//   BENCH_NTT_NBITS      = 22 (fixed production size for NTT/INTT)
//   BENCH_LDE_NBITS      = 20
//   BENCH_LDE_BLOWUP_BITS = 2  (nBitsExt = 22, 4x blowup)
//   BENCH_NCOLS           = {24, 36, 56}  (via ->Arg())
//
// Naming: OPERATION_GPU_BENCH
// All benches use ->UseRealTime().
// This file contains BENCHMARK_MAIN() -- the sole entry point for benchsgpu.
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cstdint>

#include "../src/goldilocks_base_field.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/ntt_goldilocks.cuh"
#include "../src/goldilocks_tooling.cuh"
#include "../utils/cuda_utils.hpp"

// ---------------------------------------------------------------------------
// Unified parameters
// ---------------------------------------------------------------------------
static constexpr uint64_t BENCH_NTT_NBITS        = 22;
static constexpr uint64_t BENCH_NTT_SIZE          = 1ULL << BENCH_NTT_NBITS;

static constexpr uint64_t BENCH_LDE_NBITS         = 20;
static constexpr uint64_t BENCH_LDE_SIZE           = 1ULL << BENCH_LDE_NBITS;
static constexpr uint64_t BENCH_LDE_BLOWUP_BITS   = 2;
static constexpr uint64_t BENCH_LDE_NBITS_EXT     = BENCH_LDE_NBITS + BENCH_LDE_BLOWUP_BITS;
static constexpr uint64_t BENCH_LDE_SIZE_EXT       = 1ULL << BENCH_LDE_NBITS_EXT;

// Max domain size for NTTGoldilocksGPU init (must cover both NTT=22 and LDE_EXT=22)
static constexpr uint64_t MAX_LOG_DOMAIN           = 24;

// ---------------------------------------------------------------------------
// GPU trace initialization kernel
// ---------------------------------------------------------------------------
static __global__ void initTraceKernel(gl64_t *d_data, uint64_t nElems, uint64_t nCols)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElems)
        for (uint64_t j = 0; j < nCols; j++)
            d_data[idx * nCols + j] = uint64_t(idx + j);
}

// ===================================================================
// NTT -- nBits=22, parameterized by nCols
// ===================================================================

static void NTT_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    NTTGoldilocksGPU gpu_ntt(MAX_LOG_DOMAIN, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);

    gl64_t *d_data;
    CHECKCUDAERR(cudaMalloc((void **)&d_data, BENCH_NTT_SIZE * nCols * sizeof(gl64_t)));

    dim3 thr(128), blk((BENCH_NTT_SIZE + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_data, BENCH_NTT_SIZE, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.NTT(d_data, BENCH_NTT_NBITS, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        gpu_ntt.NTT(d_data, BENCH_NTT_NBITS, nCols, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_data));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// ===================================================================
// INTT -- nBits=22, parameterized by nCols
// ===================================================================

static void INTT_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    NTTGoldilocksGPU gpu_ntt(MAX_LOG_DOMAIN, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t nCols = state.range(0);

    gl64_t *d_data;
    CHECKCUDAERR(cudaMalloc((void **)&d_data, BENCH_NTT_SIZE * nCols * sizeof(gl64_t)));

    dim3 thr(128), blk((BENCH_NTT_SIZE + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_data, BENCH_NTT_SIZE, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.INTT(d_data, BENCH_NTT_NBITS, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        gpu_ntt.INTT(d_data, BENCH_NTT_NBITS, nCols, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_data));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// ===================================================================
// LDE -- nBits=20, nBitsExt=22, parameterized by nCols
// Input is in flat column-major layout (fromRowMajorToColMajor with Layout::ColMajor).
// ===================================================================

static void LDE_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    NTTGoldilocksGPU gpu_ntt(MAX_LOG_DOMAIN, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    uint64_t nCols = state.range(0);

    gl64_t *d_flat, *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void **)&d_flat, BENCH_LDE_SIZE * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_src,  BENCH_LDE_SIZE * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_dst,  BENCH_LDE_SIZE_EXT * nCols * sizeof(gl64_t)));

    dim3 thr(128), blk((BENCH_LDE_SIZE + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_flat, BENCH_LDE_SIZE, nCols);
    fromRowMajorToColMajor(BENCH_LDE_SIZE, nCols, d_flat, d_src, Layout::ColMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.LDE(d_dst, 0, d_src, 0, BENCH_LDE_NBITS, BENCH_LDE_NBITS_EXT, nCols, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        gpu_ntt.LDE(d_dst, 0, d_src, 0, BENCH_LDE_NBITS, BENCH_LDE_NBITS_EXT, nCols, timer, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// ===================================================================
// Registration
// ===================================================================

#define NCOLS_ARGS ->Arg(24)->Arg(36)->Arg(56)

BENCHMARK(NTT_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    NCOLS_ARGS
    ->UseRealTime();

BENCHMARK(INTT_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    NCOLS_ARGS
    ->UseRealTime();

BENCHMARK(LDE_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    NCOLS_ARGS
    ->UseRealTime();

#undef NCOLS_ARGS

// ---------------------------------------------------------------------------
// BENCHMARK_MAIN() -- sole entry point for the benchsgpu binary.
// GPU bench files do NOT link -lbenchmark_main, so exactly one .cu file
// must provide main().
// ---------------------------------------------------------------------------
BENCHMARK_MAIN();
