// ---------------------------------------------------------------------------
// bench_main_proof_gpu.cu -- Reproduces the heaviest GPU kernels of a "Main"
// air STARK proof instance at its real production dimensions, without pulling
// in the pil2-proofman prover machinery.
//
// Main air dimensions (for the zisk mainnet block we targeted):
//   N       = 2^22 = 4 194 304
//   N_ext   = 2^23 = 8 388 608
//   cm1     = 38 (committed stage-1 columns)
//   cm2     = 24 (committed stage-2 columns after challenge)
//   cm3     =  6 (committed stage-3 columns, dim-3 extension)
//
// Benches registered:
//   MAIN_LDE_CM1_GPU_BENCH     -- LDE 38 cols,  N -> N_ext  (identical kernel)
//   MAIN_LDE_CM2_GPU_BENCH     -- LDE 24 cols,  N -> N_ext  (identical kernel)
//   MAIN_MERKLE_STAGE1_GPU_BENCH -- Poseidon2 Merkle 8M x 38, W=16 arity=4
//   MAIN_MERKLE_STAGE2_GPU_BENCH -- Poseidon2 Merkle 8M x 24, W=16 arity=4
//   MAIN_EXPR_PATTERN_GPU_BENCH -- heavy-arithmetic kernel mimicking the
//                                  computeExpression_ compute density
//                                  (not the interpreter; see below)
//
// Production uses W=16 arity=4 for basic-air Merkle trees (confirmed via
// nsys kernel template args <RATE=12, CAPACITY=4, SPONGE_WIDTH=16, ...>).
//
// This file does NOT define BENCHMARK_MAIN() -- bench_ntt_gpu.cu already has
// it; the Makefile links both into the same benchsgpu binary.
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <cstdint>

#include "../src/goldilocks_base_field.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/ntt_goldilocks.cuh"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/poseidon2_goldilocks.cuh"
#include "../src/goldilocks_tooling.hpp"
#include "../src/goldilocks_tooling.cuh"
#include "../src/goldilocks_trace_layout.cuh"    // tiled layout (column-major within tiles)
#include "../src/goldilocks_cubic_extension.cuh"
#include "../utils/cuda_utils.hpp"

// ---------------------------------------------------------------------------
// Main air dimensions
// ---------------------------------------------------------------------------
static constexpr uint64_t MAIN_NBITS     = 22;
static constexpr uint64_t MAIN_NBITS_EXT = 23;
static constexpr uint64_t MAIN_N         = 1ULL << MAIN_NBITS;        
static constexpr uint64_t MAIN_N_EXT     = 1ULL << MAIN_NBITS_EXT;    
static constexpr uint64_t MAIN_CM1       = 38;
static constexpr uint64_t MAIN_CM2       = 24;
static constexpr uint64_t MAIN_CM3       =  6;

// Max domain we'll ever feed to NTT constants (covers N_ext = 2^23).
static constexpr uint64_t MAX_LOG_DOMAIN = 24;

// Poseidon2 sponge width / arity used by the basic-air Merkle trees in
// production (nsys template: <12,4,16,8,22>, i.e. RATE=12 CAPACITY=4 W=16).
static constexpr uint32_t MAIN_MERKLE_W     = 16;
static constexpr uint32_t MAIN_MERKLE_ARITY = 4;

// Expression-mimic knobs. Tuned so the bench runtime lands near the solo-
// stream Main Q-kernel median (~135 ms -- the 41-launch cluster that
// corresponds to Main in the zisk solo-stream nsys trace). Scaling vs ops is
// sub-linear (~0.5 exponent), so values are dialed empirically, not by rule.
// At SAMPLE=16 / 8M rows / (512 blocks * 256 threads) = 64 rows/thread.
static constexpr uint32_t EXPR_BASE_FMA_OPS = 7000;  // base-field FMAs per row
static constexpr uint32_t EXPR_FP3_OPS      = 900;   // Fp3 mul-adds per row

// ---------------------------------------------------------------------------
// Trace-init kernel shared with existing bench files' style.
// ---------------------------------------------------------------------------
static __global__ void initTraceKernel(gl64_t *d_data, uint64_t nRows, uint64_t nCols)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows) {
        for (uint64_t j = 0; j < nCols; j++)
            d_data[idx * nCols + j] = uint64_t(idx + j);
    }
}

// ===========================================================================
// MAIN_LDE_CM1_GPU_BENCH -- LDE 38 cols, N -> N_ext
// Identical kernel path to production stage-1 commit LDE for Main.
// ===========================================================================
static void MAIN_LDE_CM1_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    NTTGoldilocksGPU gpu_ntt(MAX_LOG_DOMAIN, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    gl64_t *d_flat, *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void **)&d_flat, MAIN_N     * MAIN_CM1 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_src,  MAIN_N     * MAIN_CM1 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_dst,  MAIN_N_EXT * MAIN_CM1 * sizeof(gl64_t)));

    dim3 thr(128), blk((MAIN_N + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_flat, MAIN_N, MAIN_CM1);
    fromRowMajorToTiled(MAIN_N, MAIN_CM1, d_flat, d_src, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.LDE(d_dst, 0, d_src, 0, MAIN_NBITS, MAIN_NBITS_EXT, MAIN_CM1, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        gpu_ntt.LDE(d_dst, 0, d_src, 0, MAIN_NBITS, MAIN_NBITS_EXT, MAIN_CM1, timer, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// ===========================================================================
// MAIN_LDE_CM2_GPU_BENCH -- LDE 24 cols, N -> N_ext (stage-2 shape)
// ===========================================================================
static void MAIN_LDE_CM2_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    NTTGoldilocksGPU gpu_ntt(MAX_LOG_DOMAIN, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    gl64_t *d_flat, *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void **)&d_flat, MAIN_N     * MAIN_CM2 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_src,  MAIN_N     * MAIN_CM2 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_dst,  MAIN_N_EXT * MAIN_CM2 * sizeof(gl64_t)));

    dim3 thr(128), blk((MAIN_N + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_flat, MAIN_N, MAIN_CM2);
    fromRowMajorToTiled(MAIN_N, MAIN_CM2, d_flat, d_src, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.LDE(d_dst, 0, d_src, 0, MAIN_NBITS, MAIN_NBITS_EXT, MAIN_CM2, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        gpu_ntt.LDE(d_dst, 0, d_src, 0, MAIN_NBITS, MAIN_NBITS_EXT, MAIN_CM2, timer, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// ===========================================================================
// MAIN_MERKLE_STAGE1_GPU_BENCH -- Poseidon2 Merkle tree over 8M x 38
// W=16 arity=4, Tiles layout (identical to production stage-1 commit).
// ===========================================================================
static void MAIN_MERKLE_STAGE1_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t tree_size = getTreeNumElements(MAIN_N_EXT, MAIN_MERKLE_ARITY);

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, MAIN_N_EXT * MAIN_CM1 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((MAIN_N_EXT + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_trace, MAIN_N_EXT, MAIN_CM1);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::merkletree(
        MAIN_MERKLE_ARITY,
        (uint64_t *)d_tree,
        (uint64_t *)d_trace,
        MAIN_CM1, MAIN_N_EXT,
        Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::merkletree(
            MAIN_MERKLE_ARITY,
            (uint64_t *)d_tree,
            (uint64_t *)d_trace,
            MAIN_CM1, MAIN_N_EXT,
            Layout::Tiles, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===========================================================================
// MAIN_MERKLE_STAGE2_GPU_BENCH -- Poseidon2 Merkle tree over 8M x 24
// ===========================================================================
static void MAIN_MERKLE_STAGE2_GPU_BENCH(benchmark::State &state)
{
    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t tree_size = getTreeNumElements(MAIN_N_EXT, MAIN_MERKLE_ARITY);

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, MAIN_N_EXT * MAIN_CM2 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((MAIN_N_EXT + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_trace, MAIN_N_EXT, MAIN_CM2);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::merkletree(
        MAIN_MERKLE_ARITY,
        (uint64_t *)d_tree,
        (uint64_t *)d_trace,
        MAIN_CM2, MAIN_N_EXT,
        Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::merkletree(
            MAIN_MERKLE_ARITY,
            (uint64_t *)d_tree,
            (uint64_t *)d_trace,
            MAIN_CM2, MAIN_N_EXT,
            Layout::Tiles, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===========================================================================
// MAIN_LDE_SWEEP_GPU_BENCH -- LDE N->N_ext, nCols swept via ->Arg().
// Same kernel path as the production stage-1/2 commit LDE.
// ===========================================================================
static void MAIN_LDE_SWEEP_GPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);

    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    NTTGoldilocksGPU gpu_ntt(MAX_LOG_DOMAIN, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    gl64_t *d_flat, *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void **)&d_flat, MAIN_N     * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_src,  MAIN_N     * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_dst,  MAIN_N_EXT * nCols * sizeof(gl64_t)));

    dim3 thr(128), blk((MAIN_N + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_flat, MAIN_N, nCols);
    fromRowMajorToTiled(MAIN_N, nCols, d_flat, d_src, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    gpu_ntt.LDE(d_dst, 0, d_src, 0, MAIN_NBITS, MAIN_NBITS_EXT, nCols, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        gpu_ntt.LDE(d_dst, 0, d_src, 0, MAIN_NBITS, MAIN_NBITS_EXT, nCols, timer, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// ===========================================================================
// MAIN_MERKLE_SWEEP_GPU_BENCH -- Poseidon2 Merkle over N_ext rows, W=16
// arity=4, Tiles layout; nCols swept via ->Arg(). Tree is built over the
// extended (LDE'd) domain, matching production stage commits.
// ===========================================================================
static void MAIN_MERKLE_SWEEP_GPU_BENCH(benchmark::State &state)
{
    uint64_t nCols = state.range(0);

    uint32_t gpu_id = 0;
    CHECKCUDAERR(cudaGetDevice((int *)&gpu_id));
    Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::initConstants(&gpu_id, 1);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    uint64_t tree_size = getTreeNumElements(MAIN_N_EXT, MAIN_MERKLE_ARITY);

    gl64_t *d_trace;
    Goldilocks::Element *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_trace, MAIN_N_EXT * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_tree,  tree_size * sizeof(Goldilocks::Element)));

    dim3 thr(128), blk((MAIN_N_EXT + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_trace, MAIN_N_EXT, nCols);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Warm up
    Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::merkletree(
        MAIN_MERKLE_ARITY,
        (uint64_t *)d_tree,
        (uint64_t *)d_trace,
        nCols, MAIN_N_EXT,
        Layout::Tiles, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        Poseidon2GoldilocksGPU<MAIN_MERKLE_W>::merkletree(
            MAIN_MERKLE_ARITY,
            (uint64_t *)d_tree,
            (uint64_t *)d_trace,
            nCols, MAIN_N_EXT,
            Layout::Tiles, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_trace));
    CHECKCUDAERR(cudaFree(d_tree));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===========================================================================
// expressionMimicKernel -- heavy-arithmetic pattern over TILED trace data.
//
// Not the production bytecode interpreter (that would require copying
// ParserArgs/ParserParams from pil2-stark). Instead we reproduce:
//   - The compute density (FMA chain on committed cols + Fp3 ops).
//   - The MEMORY ACCESS PATTERN: data lives in column-major-within-tiles
//     layout (TILE_HEIGHT=256, TILE_WIDTH=4), and rows within a warp form
//     a coalesced 256 B load per column. This is the same layout used by
//     production's `computeExpression_` via getBufferOffset_pack256.
//
// Shape matches production:
//   Grid  = 512 blocks       (matches the prover's capped nBlocks)
//   Block = 256 threads      (one tile-row per thread)
//   Each block processes nchunks / gridDim.x tiles serially.
//   Static smem = 256 + 2048*2 = 4352 B   (ops_staged + args_staged)
//   Dynamic smem = ~40 KB                 (baseline Path-A budget)
//
// launch_bounds caps occupancy so the compiler can't blow register count.
// ===========================================================================
__global__ __launch_bounds__(256, 2)
static void expressionMimicKernel(
    const gl64_t * __restrict__ cm1_data,   // tiled [nRows x MAIN_CM1]
    const gl64_t * __restrict__ cm2_data,   // tiled [nRows x MAIN_CM2]
    const gl64_t * __restrict__ cm3_data,   // tiled [nRows x MAIN_CM3 * 3] (Fp3 split into 3 gl64 cols)
    gl64_t       * __restrict__ dest,       // row-major [nRows] scratch
    uint32_t nRows,
    uint32_t baseOps,
    uint32_t fp3Ops)
{
    // Reproduce production's static shared memory footprint.
    __shared__ uint8_t  ops_staged[256];
    __shared__ uint16_t args_staged[2048];
    extern __shared__ gl64_t smem[];   // dynamic smem placeholder (matches launch cfg)

    // Cooperative stage to prevent DCE of the smem arrays.
    for (uint32_t t = threadIdx.x; t < 256; t += blockDim.x)
        ops_staged[t] = uint8_t(t ^ blockIdx.x);
    for (uint32_t t = threadIdx.x; t < 2048; t += blockDim.x)
        args_staged[t] = uint16_t(t * 0x9E37u);
    __syncthreads();

    // 16 sample lanes per side keeps register pressure bounded while still
    // representing enough column diversity to match production's pattern.
    constexpr uint32_t SAMPLE = 16;

    // Iterate tiles: chunk_idx = which 256-row tile, stepping by gridDim.x.
    uint64_t nchunks = nRows / TILE_HEIGHT;   // 8M / 256 = 32768
    uint32_t chunk_idx = blockIdx.x;
    while (chunk_idx < nchunks) {
        uint64_t chunkBase = (uint64_t)chunk_idx * TILE_HEIGHT;

        // Tiled reads: each thread (row = chunkBase + threadIdx.x) reads its
        // own row across all sampled columns. Within a warp of 32 threads the
        // 32 consecutive rows map to 32 consecutive offsets -> coalesced load.
        gl64_t a[SAMPLE], b[SAMPLE];
        #pragma unroll
        for (uint32_t k = 0; k < SAMPLE; ++k) {
            a[k] = cm1_data[getBufferOffset_pack256(chunkBase, k, nRows, MAIN_CM1)];
            b[k] = cm2_data[getBufferOffset_pack256(chunkBase, k, nRows, MAIN_CM2)];
        }

        Goldilocks3GPU::Element c3[MAIN_CM3];
        #pragma unroll
        for (uint32_t k = 0; k < MAIN_CM3; ++k) {
            c3[k][0] = cm3_data[getBufferOffset_pack256(chunkBase, k * 3 + 0, nRows, MAIN_CM3 * 3)];
            c3[k][1] = cm3_data[getBufferOffset_pack256(chunkBase, k * 3 + 1, nRows, MAIN_CM3 * 3)];
            c3[k][2] = cm3_data[getBufferOffset_pack256(chunkBase, k * 3 + 2, nRows, MAIN_CM3 * 3)];
        }

        // Base-field FMA chain (mimics the prover's hot per-row ops).
        gl64_t acc = a[0];
        for (uint32_t k = 0; k < baseOps; ++k) {
            acc = acc * a[k & (SAMPLE - 1)] + b[k & (SAMPLE - 1)];
        }

        // Fp3 chain (mimics cubic-extension ops; dim=3 structure in production).
        Goldilocks3GPU::Element acc3;
        acc3[0] = c3[0][0]; acc3[1] = c3[0][1]; acc3[2] = c3[0][2];
        for (uint32_t k = 0; k < fp3Ops; ++k) {
            Goldilocks3GPU::Element t;
            Goldilocks3GPU::mul(t, acc3, c3[k % MAIN_CM3]);
            Goldilocks3GPU::add(acc3, t, c3[(k + 1) % MAIN_CM3]);
        }

        // Fuse both accumulators into a per-row store; prevents DCE.
        // Also read from staged arrays to suppress unused-variable warnings.
        uint64_t row = chunkBase + threadIdx.x;
        gl64_t staged_contrib = gl64_t(ops_staged[threadIdx.x & 255]) & gl64_t(args_staged[threadIdx.x & 2047] == 0xFFFF);
        dest[row] = acc + acc3[0] + acc3[1] + acc3[2] + staged_contrib;

        chunk_idx += gridDim.x;
    }
}

// ===========================================================================
// MAIN_EXPR_PATTERN_GPU_BENCH -- compute-density mimic of computeExpression_
// ===========================================================================
static void MAIN_EXPR_PATTERN_GPU_BENCH(benchmark::State &state)
{
    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    // Allocate BOTH row-major (for init) and tiled (kernel input) buffers.
    gl64_t *d_cm1_flat, *d_cm2_flat, *d_cm3_flat;
    gl64_t *d_cm1, *d_cm2, *d_cm3, *d_dest;
    CHECKCUDAERR(cudaMalloc((void **)&d_cm1_flat, MAIN_N_EXT * MAIN_CM1 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_cm2_flat, MAIN_N_EXT * MAIN_CM2 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_cm3_flat, MAIN_N_EXT * MAIN_CM3 * 3 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_cm1,      MAIN_N_EXT * MAIN_CM1 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_cm2,      MAIN_N_EXT * MAIN_CM2 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_cm3,      MAIN_N_EXT * MAIN_CM3 * 3 * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void **)&d_dest,     MAIN_N_EXT * sizeof(gl64_t)));

    dim3 thr(128), blk((MAIN_N_EXT + 127) / 128);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_cm1_flat, MAIN_N_EXT, MAIN_CM1);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_cm2_flat, MAIN_N_EXT, MAIN_CM2);
    initTraceKernel<<<blk, thr, 0, stream>>>(d_cm3_flat, MAIN_N_EXT, MAIN_CM3 * 3);

    // Convert to tiled layout (column-major within 256x4 tiles). This is what
    // production's `computeExpression_` reads from.
    fromRowMajorToTiled(MAIN_N_EXT, MAIN_CM1,     d_cm1_flat, d_cm1, stream);
    fromRowMajorToTiled(MAIN_N_EXT, MAIN_CM2,     d_cm2_flat, d_cm2, stream);
    fromRowMajorToTiled(MAIN_N_EXT, MAIN_CM3 * 3, d_cm3_flat, d_cm3, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Match production: 512 blocks x 256 threads, ~40 KB dynamic smem.
    constexpr uint32_t NBLOCKS  = 512;
    constexpr uint32_t NTHREADS = 256;
    constexpr size_t   DYN_SMEM = 40960;   // 5120 gl64_t elements

    // Warm up
    expressionMimicKernel<<<NBLOCKS, NTHREADS, DYN_SMEM, stream>>>(
        d_cm1, d_cm2, d_cm3, d_dest,
        (uint32_t)MAIN_N_EXT, EXPR_BASE_FMA_OPS, EXPR_FP3_OPS);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    for (auto _ : state) {
        expressionMimicKernel<<<NBLOCKS, NTHREADS, DYN_SMEM, stream>>>(
            d_cm1, d_cm2, d_cm3, d_dest,
            (uint32_t)MAIN_N_EXT, EXPR_BASE_FMA_OPS, EXPR_FP3_OPS);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }

    CHECKCUDAERR(cudaFree(d_cm1_flat));
    CHECKCUDAERR(cudaFree(d_cm2_flat));
    CHECKCUDAERR(cudaFree(d_cm3_flat));
    CHECKCUDAERR(cudaFree(d_cm1));
    CHECKCUDAERR(cudaFree(d_cm2));
    CHECKCUDAERR(cudaFree(d_cm3));
    CHECKCUDAERR(cudaFree(d_dest));
    CHECKCUDAERR(cudaStreamDestroy(stream));
}

// ===========================================================================
// Registration
// ===========================================================================

BENCHMARK(MAIN_LDE_CM1_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(MAIN_LDE_CM2_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(MAIN_MERKLE_STAGE1_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(MAIN_MERKLE_STAGE2_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(MAIN_EXPR_PATTERN_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// ---------------------------------------------------------------------------
// Column sweep: nCols in {5,10,22,38,44} at N=2^22 -> N_ext=2^23 (blowup x2).
// LDE timed over N->N_ext; merkle timed over the extended N_ext domain.
// ---------------------------------------------------------------------------
#define MAIN_NCOLS_ARGS ->Arg(5)->Arg(10)->Arg(22)->Arg(38)->Arg(44)

BENCHMARK(MAIN_LDE_SWEEP_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    MAIN_NCOLS_ARGS
    ->UseRealTime();

BENCHMARK(MAIN_MERKLE_SWEEP_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    MAIN_NCOLS_ARGS
    ->UseRealTime();

#undef MAIN_NCOLS_ARGS

// No BENCHMARK_MAIN() here -- bench_ntt_gpu.cu defines it for the whole binary.