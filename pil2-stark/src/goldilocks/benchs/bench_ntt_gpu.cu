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
// LDE backend comparison -- a SINGLE benchmark that runs the same (nBits, nCols) through BOTH backends
// directly (bypassing the resolveLayout dispatch) and reports them head-to-head, so each row shows
// sppark vs native + the speedup. This is how we validate the resolveLayout threshold. Args: {nBits, nCols}.
// Per-row counters: sppark_ms, native_ms, native_speedup (sppark/native; >1 means native is faster).
// ===================================================================

// Time one backend over `iters` launches with CUDA events; returns total ms. Lays `d_src` out in the
// backend's own storage layout first. Caller owns the buffers.
static double timeLdeBackend(int native, NTTGoldilocksGPU &gpu_ntt,
                             gl64_t *d_flat, gl64_t *d_src, gl64_t *d_dst,
                             uint64_t nBits, uint64_t nBitsExt, uint64_t nCols,
                             cudaStream_t stream, int iters)
{
    const uint64_t N = 1ULL << nBits;
    const Layout layout = native ? Layout::ColMajorTiled : Layout::ColMajor;
    dim3 thr(128), blk((N + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_flat, N, nCols);
    fromRowMajorToColMajor(N, nCols, d_flat, d_src, layout, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    auto run = [&]() {
        if (native == 1)      gpu_ntt.ldeTiled(d_dst, d_src, nBits, nBitsExt, nCols, stream);
        else if (native == 2) gpu_ntt.ldeColMajor(d_dst, d_src, nBits, nBitsExt, nCols, stream);
        else                  gpu_ntt.ldeSppark(d_dst, d_src, nBits, nBitsExt, nCols, stream);
    };
    run();  // warm up
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    cudaEvent_t beg, end;
    CHECKCUDAERR(cudaEventCreate(&beg));
    CHECKCUDAERR(cudaEventCreate(&end));
    CHECKCUDAERR(cudaEventRecord(beg, stream));
    for (int i = 0; i < iters; i++) run();
    CHECKCUDAERR(cudaEventRecord(end, stream));
    CHECKCUDAERR(cudaEventSynchronize(end));
    float ms = 0.0f;
    CHECKCUDAERR(cudaEventElapsedTime(&ms, beg, end));
    CHECKCUDAERR(cudaEventDestroy(beg));
    CHECKCUDAERR(cudaEventDestroy(end));
    return (double)ms / iters;
}

static void LDE_COMPARE_BENCH(benchmark::State &state)
{
    const uint64_t nBits    = (uint64_t)state.range(0);
    const uint64_t nBitsExt = nBits + BENCH_LDE_BLOWUP_BITS;
    const uint64_t nCols    = (uint64_t)state.range(1);
    const uint64_t N        = 1ULL << nBits;
    const uint64_t NExt     = 1ULL << nBitsExt;

    // Footprint = d_flat(N) + d_src(N) + d_dst(NExt) columns. Skip points that won't fit (extended
    // buffer dominates). Leave headroom for sppark's twiddle tables.
    size_t freeB = 0, totalB = 0;
    CHECKCUDAERR(cudaMemGetInfo(&freeB, &totalB));
    size_t needB = (2 * N + NExt) * nCols * sizeof(gl64_t);
    if (needB > (size_t)(0.85 * (double)freeB)) {
        state.SkipWithError("skip: does not fit in GPU memory");
        return;
    }

    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    NTTGoldilocksGPU gpu_ntt(MAX_LOG_DOMAIN, 1, &gpu_id);
    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    gl64_t *d_flat, *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void **)&d_flat, N    * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_src,  N    * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_dst,  NExt * nCols * sizeof(gl64_t)));

    double sppark_ms = 0.0, native_ms = 0.0;
    for (auto _ : state) {
        sppark_ms = timeLdeBackend(/*native=*/false, gpu_ntt, d_flat, d_src, d_dst, nBits, nBitsExt, nCols, stream, 3);
        native_ms = timeLdeBackend(/*native=*/true,  gpu_ntt, d_flat, d_src, d_dst, nBits, nBitsExt, nCols, stream, 3);
        state.SetIterationTime((sppark_ms + native_ms) / 1e3);
    }
    // How many times faster sppark is than native (e.g. 2.2 => sppark 2.2x faster;
    // 0.85 => sppark slower). Google Benchmark renders counter columns alphabetically,
    // so the leading "1_/2_/3_" prefixes force the order native, sppark, ratio.
    const double sppark_speedup_x = (sppark_ms > 0.0) ? native_ms / sppark_ms : 0.0;
    state.counters["1_native_ms"]      = native_ms;
    state.counters["2_sppark_ms"]      = sppark_ms;
    state.counters["3_sppark_speedup"] = sppark_speedup_x;

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Per-nBits column grids (small domains can afford wide nCols; large domains only narrow ones fit/matter).
static void LdeCompareArgs(benchmark::internal::Benchmark *b) {
    for (int64_t nCols : {64, 512, 1024, 2137}) b->Args({17, nCols});
    for (int64_t nCols : {64, 128, 256})        b->Args({19, nCols});
    for (int64_t nCols : {64, 128})             b->Args({21, nCols});
    for (int64_t nCols : {64, 128})             b->Args({22, nCols});
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

// LDE backend comparison: each row runs BOTH backends at one (nBits, nCols) and reports
// sppark_ms / native_ms / native_speedup counters side by side.
BENCHMARK(LDE_COMPARE_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->ArgNames({"nBits", "nCols"})
    ->Apply(LdeCompareArgs)
    ->UseManualTime();

// ---------------------------------------------------------------------------
// BENCHMARK_MAIN() -- sole entry point for the benchsgpu binary.
// GPU bench files do NOT link -lbenchmark_main, so exactly one .cu file
// must provide main().
// ---------------------------------------------------------------------------
BENCHMARK_MAIN();
