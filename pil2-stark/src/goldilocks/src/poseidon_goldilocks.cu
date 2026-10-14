// ---------------------------------------------------------------------------
// Poseidon v1
// ---------------------------------------------------------------------------

#include "goldilocks_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>
#include <math.h>
#include <cassert>

#include "poseidon_goldilocks.hpp"
#include "poseidon_goldilocks.cuh"
#include "grinding_gpu.cuh"
#include "poseidon_goldilocks_constants.hpp"

// Pull in the STARK_POSEIDON1 toggle from the API-internal header when reachable
// (full pil2-stark build).
#if __has_include("starks_api_internal.hpp")
#include "starks_api_internal.hpp"
#endif

typedef uint32_t u32;
typedef uint64_t u64;

// The immediate-MDS path (pos1_dot_mimm_ and the pos1_mvp_*_mimm_ helpers)
// accumulates raw 64-bit-state x constant products in a u128 with no carry
// limb, which is only sound while every M entry is < 2^32. Check that
// assumption at compile time.
template<uint32_t W>
static constexpr bool pos1MdsEntriesBelow32(const Goldilocks::Element (&m)[W][W])
{
    for (uint32_t i = 0; i < W; ++i)
        for (uint32_t j = 0; j < W; ++j)
            if (m[i][j].fe >= (1ULL << 32)) return false;
    return true;
}
static_assert(pos1MdsEntriesBelow32<8>(PoseidonGoldilocksConstants::M8),  "Poseidon1 M8 entries must be < 2^32 for the lazy MDS path");
static_assert(pos1MdsEntriesBelow32<12>(PoseidonGoldilocksConstants::M12), "Poseidon1 M12 entries must be < 2^32 for the lazy MDS path");
static_assert(pos1MdsEntriesBelow32<16>(PoseidonGoldilocksConstants::M16), "Poseidon1 M16 entries must be < 2^32 for the lazy MDS path");


static constexpr size_t CUDA_SMEM_DEFAULT_BLOCK_LIMIT = 48 * 1024;

// CUDA threads per block
template<uint32_t W>
static constexpr u32 pos1Tpb()
{
    return (W == 16) ? 256 : 512;
}
static_assert(pos1Tpb<8>()  * 8  * sizeof(uint64_t) <= CUDA_SMEM_DEFAULT_BLOCK_LIMIT, "Poseidon1 W8 smem over launch limit");
static_assert(pos1Tpb<12>() * 12 * sizeof(uint64_t) <= CUDA_SMEM_DEFAULT_BLOCK_LIMIT, "Poseidon1 W12 smem over launch limit");
static_assert(pos1Tpb<16>() * 16 * sizeof(uint64_t) <= CUDA_SMEM_DEFAULT_BLOCK_LIMIT, "Poseidon1 W16 smem over launch limit");

// Min resident blocks/SM the tiled kernel's register budget is compiled for
// (launch_bounds). 2 pins the measured optimum: with 1 the compiler balloons
// registers and loses ~20%; forcing 3 spills and loses 4-6%.
#define POS1_TILED_MINB 2

// Merkle levels with at most this many nodes use the warp-per-node kernel; larger levels keep the thread-per-node kernel
#define POS1_MERKLE_WARP_MAX_NODES 16384

// ---------------------------------------------------------------------------
// __constant__ memory — distinct symbols from Poseidon2.
// ---------------------------------------------------------------------------
__device__ __constant__ uint64_t GPU_POS1_C12[118];
__device__ __constant__ uint64_t GPU_POS1_S12[507];
__device__ __constant__ uint64_t GPU_POS1_M12[144];
__device__ __constant__ uint64_t GPU_POS1_P12[144];

__device__ __constant__ uint64_t GPU_POS1_C8[86];
__device__ __constant__ uint64_t GPU_POS1_S8[330];
__device__ __constant__ uint64_t GPU_POS1_M8[64];
__device__ __constant__ uint64_t GPU_POS1_P8[64];

__device__ __constant__ uint64_t GPU_POS1_C16[150];
__device__ __constant__ uint64_t GPU_POS1_S16[683];
__device__ __constant__ uint64_t GPU_POS1_M16[256];
__device__ __constant__ uint64_t GPU_POS1_P16[256];

// Per-W constant accessor. Specialized below.
template<uint32_t W> struct Pos1ConstGPU;
template<> struct Pos1ConstGPU<8> {
    static __device__ __forceinline__ const gl64_t* C() { return (const gl64_t*)GPU_POS1_C8; }
    static __device__ __forceinline__ const gl64_t* S() { return (const gl64_t*)GPU_POS1_S8; }
    static __device__ __forceinline__ const gl64_t* M() { return (const gl64_t*)GPU_POS1_M8; }
    static __device__ __forceinline__ const gl64_t* P() { return (const gl64_t*)GPU_POS1_P8; }
};
template<> struct Pos1ConstGPU<12> {
    static __device__ __forceinline__ const gl64_t* C() { return (const gl64_t*)GPU_POS1_C12; }
    static __device__ __forceinline__ const gl64_t* S() { return (const gl64_t*)GPU_POS1_S12; }
    static __device__ __forceinline__ const gl64_t* M() { return (const gl64_t*)GPU_POS1_M12; }
    static __device__ __forceinline__ const gl64_t* P() { return (const gl64_t*)GPU_POS1_P12; }
};
template<> struct Pos1ConstGPU<16> {
    static __device__ __forceinline__ const gl64_t* C() { return (const gl64_t*)GPU_POS1_C16; }
    static __device__ __forceinline__ const gl64_t* S() { return (const gl64_t*)GPU_POS1_S16; }
    static __device__ __forceinline__ const gl64_t* M() { return (const gl64_t*)GPU_POS1_M16; }
    static __device__ __forceinline__ const gl64_t* P() { return (const gl64_t*)GPU_POS1_P16; }
};

// ---------------------------------------------------------------------------
// Kernel definitions.
// ---------------------------------------------------------------------------

// Single-element permute — launched <<<1, 1>>>.
template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void permuteKernel_pos1(uint64_t *output, const uint64_t *input)
{
    assert(blockDim.x == 1 && gridDim.x == 1);

    gl64_t state[W];
    gl64_t in_reg[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        in_reg[i] = input[i];

    poseidon1PermuteReg<W, HALF_F, N_PART>(
        state, in_reg,
        Pos1ConstGPU<W>::C(),
        Pos1ConstGPU<W>::S(),
        Pos1ConstGPU<W>::M(),
        Pos1ConstGPU<W>::P());

#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        output[i] = state[i];
}

// Single-element permuteTrunc — first CAPACITY out of permute.
template<uint32_t W, uint32_t HALF_F, uint32_t N_PART, uint32_t CAPACITY_T>
__global__ void permuteTruncKernel_pos1(uint64_t *output, const uint64_t *input)
{
    assert(blockDim.x == 1 && gridDim.x == 1);

    gl64_t state[W];
    gl64_t in_reg[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        in_reg[i] = input[i];

    poseidon1PermuteReg<W, HALF_F, N_PART>(
        state, in_reg,
        Pos1ConstGPU<W>::C(),
        Pos1ConstGPU<W>::S(),
        Pos1ConstGPU<W>::M(),
        Pos1ConstGPU<W>::P());

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY_T; ++i)
        output[i] = state[i];
}

// --- linear-hash helpers (shared-memory path) ---
// Launched with dynamic shared mem = blockDim.x * W * sizeof(uint64_t).
template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void linearHashKernel_pos1(uint64_t *__restrict__ output,
                                      uint64_t *__restrict__ input,
                                      uint32_t num_cols, uint32_t num_rows)
{
    const uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    gl64_t *row_in  = (gl64_t *)(input  + tid * (uint64_t)num_cols);
    gl64_t *row_out = (gl64_t *)(output + tid * (uint64_t)CAPACITY_T);

    if (num_cols == 0)
    {
#pragma unroll
        for (uint32_t i = 0; i < CAPACITY_T; ++i)
            row_out[i] = gl64_t(uint64_t(0));
        return;
    }

    uint32_t remaining = num_cols;
    bool first = true;

    while (remaining)
    {
        if (first)
        {
#pragma unroll
            for (uint32_t i = 0; i < CAPACITY_T; ++i)
                scratchpad[(RATE_T + i) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));
            first = false;
        }
        else
        {
            // Carry previous output's capacity block up (scratchpad-local).
#pragma unroll
            for (uint32_t i = 0; i < CAPACITY_T; ++i)
                scratchpad[(RATE_T + i) * blockDim.x + threadIdx.x] =
                    scratchpad[i * blockDim.x + threadIdx.x];
        }

        uint32_t n = (remaining < RATE_T) ? remaining : RATE_T;

        // Fill rate region + zero-pad tail.
        for (uint32_t i = 0; i < n; ++i)
            scratchpad[i * blockDim.x + threadIdx.x] = row_in[(num_cols - remaining) + i];
        for (uint32_t i = n; i < RATE_T; ++i)
            scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));

        poseidon1PermuteSmem<W, HALF_F, N_PART>(
            Pos1ConstGPU<W>::C(),
            Pos1ConstGPU<W>::S(),
            Pos1ConstGPU<W>::M(),
            Pos1ConstGPU<W>::P());

        remaining -= n;
    }

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY_T; ++i)
        row_out[i] = scratchpad[i * blockDim.x + threadIdx.x];
}

// linearHashTiledKernel_pos1: block-tiled input layout + shared-memory state.
// TPB_V/MINB bound the launch so the compiler budgets registers for MINB
// resident blocks (W=16 is register-limited to 2 blocks/SM at natural
// allocation; forcing more trades spills for occupancy — measured choice).
template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART,
         uint32_t TPB_V, uint32_t MINB>
__global__ void __launch_bounds__(TPB_V, MINB) linearHashTiledKernel_pos1(uint64_t *__restrict__ output,
                                           uint64_t *__restrict__ input,
                                           uint32_t num_cols, uint32_t num_rows, Layout layout, uint32_t ld)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    gl64_t *row_out = (gl64_t *)(output + (uint64_t)row * (uint64_t)CAPACITY_T);

    if (num_cols == 0)
    {
#pragma unroll
        for (uint32_t i = 0; i < CAPACITY_T; ++i)
            row_out[i] = gl64_t(uint64_t(0));
        return;
    }

    uint32_t remaining = num_cols;
    bool first = true;

    while (remaining)
    {
        if (first)
        {
#pragma unroll
            for (uint32_t i = 0; i < CAPACITY_T; ++i)
                scratchpad[(RATE_T + i) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));
            first = false;
        }
        else
        {
#pragma unroll
            for (uint32_t i = 0; i < CAPACITY_T; ++i)
                scratchpad[(RATE_T + i) * blockDim.x + threadIdx.x] =
                    scratchpad[i * blockDim.x + threadIdx.x];
        }

        uint32_t n = (remaining < RATE_T) ? remaining : RATE_T;
        uint32_t col0 = num_cols - remaining;
        for (uint32_t i = 0; i < n; ++i)
        {
            uint64_t idx = getBufferOffset(row, col0 + i, ld, num_cols, layout);
            scratchpad[i * blockDim.x + threadIdx.x] = ((gl64_t *)input)[idx];
        }
        for (uint32_t i = n; i < RATE_T; ++i)
            scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));

        poseidon1PermuteSmem<W, HALF_F, N_PART>(
            Pos1ConstGPU<W>::C(),
            Pos1ConstGPU<W>::S(),
            Pos1ConstGPU<W>::M(),
            Pos1ConstGPU<W>::P());

        remaining -= n;
    }

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY_T; ++i)
        row_out[i] = scratchpad[i * blockDim.x + threadIdx.x];
}

// Shared-memory merkle reduction kernel: reads arity*CAPACITY child digests
// per parent, zero-pads to SPONGE_WIDTH, permuteTrunces.
// Launched with dynamic shared mem = blockDim.x * W * sizeof(uint64_t).
template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void merkleNodeKernel_pos1(uint64_t nextN, uint64_t nextIndex,
                                      uint64_t pending, uint32_t arity,
                                      uint64_t *cursor)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN) return;

    // Load input: arity*CAPACITY child bytes into [0..copy_n), zero-pad [copy_n..W).
    const uint32_t stride = arity * CAPACITY_T;
    const uint64_t base = (uint64_t)nextIndex + (uint64_t)tid * (uint64_t)stride;
    const uint32_t n = (stride < W) ? stride : W;

    for (uint32_t i = 0; i < n; ++i)
        scratchpad[i * blockDim.x + threadIdx.x] = ((gl64_t *)cursor)[base + i];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        if (i >= n)
            scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));

    poseidon1PermuteSmem<W, HALF_F, N_PART>(
        Pos1ConstGPU<W>::C(),
        Pos1ConstGPU<W>::S(),
        Pos1ConstGPU<W>::M(),
        Pos1ConstGPU<W>::P());

    gl64_t *out = (gl64_t *)(&cursor[(uint64_t)nextIndex + ((uint64_t)pending + tid) * CAPACITY_T]);
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY_T; ++i)
        out[i] = scratchpad[i * blockDim.x + threadIdx.x];
}

// Warp-per-node merkle reduction for SMALL levels. One warp cooperates on each
// node's permutation (lane i owns state element i, MVP via shuffles), cutting
// the per-node serial dependency chain ~W-fold vs. the one-thread-per-node
// kernel above, whose ~100us single-permute latency dominates tree-top levels
// that are far too small to fill the GPU. Digests are stored canonicalized
template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void merkleNodeWarpKernel_pos1(uint64_t nextN, uint64_t nextIndex,
                                          uint64_t pending, uint32_t arity,
                                          uint64_t *cursor)
{
    const uint32_t lane = threadIdx.x & 31;
    const uint64_t node = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (node >= nextN || lane >= W) return;
    const uint32_t mask = (uint32_t)((1ull << W) - 1);

    const uint32_t stride = arity * CAPACITY_T;
    const uint64_t base = (uint64_t)nextIndex + node * (uint64_t)stride;
    const uint32_t n = (stride < W) ? stride : W;

    gl64_t v = (lane < n) ? ((gl64_t *)cursor)[base + lane] : gl64_t(uint64_t(0));
    v = poseidon1PermuteWarpReg<W, HALF_F, N_PART>(v, lane, mask,
        Pos1ConstGPU<W>::C(), Pos1ConstGPU<W>::S(),
        Pos1ConstGPU<W>::M(), Pos1ConstGPU<W>::P());

    if (lane < CAPACITY_T)
    {
        uint64_t u = v[0];
        if (u >= GOLDILOCKS_PRIME) u -= GOLDILOCKS_PRIME;
        cursor[nextIndex + (pending + node) * CAPACITY_T + lane] = u;
    }
}

// Grinding hash: [c0, c1, c2, nonce, 0, ...] -> first output limb, on the register-path permute.
template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
struct Pos1GrindHash
{
    static constexpr uint32_t ROUNDS = 1;
    gl64_t in_reg[W];
    __device__ explicit Pos1GrindHash(const uint64_t *in)
    {
#pragma unroll
        for (uint32_t i = 0; i < W; ++i)
            in_reg[i] = (gl64_t)(i < 3 ? in[i] : (uint64_t)0);
    }
    __device__ uint64_t operator()(uint64_t n)
    {
        gl64_t state[W];
        in_reg[3] = (gl64_t)n;
        poseidon1PermuteReg<W, HALF_F, N_PART>(state, in_reg, Pos1ConstGPU<W>::C(), Pos1ConstGPU<W>::S(),
                                               Pos1ConstGPU<W>::M(), Pos1ConstGPU<W>::P());
        return (uint64_t)state[0];
    }
};

// ---------------------------------------------------------------------------
// Public API: initConstants — upload __constant__ arrays to each GPU.
// ---------------------------------------------------------------------------
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::initConstants(uint32_t *gpu_ids, uint32_t num_gpu_ids)
{
    int saved;
    CHECKCUDAERR(cudaGetDevice(&saved));
    static int initialized[17] = {0}; // indexed by SPONGE_WIDTH
    static_assert(W < 17, "initialized[] needs to cover W");
    if (initialized[W] == 0)
    {
        for (uint32_t i = 0; i < num_gpu_ids; ++i)
        {
            CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
            if constexpr (W == 12) {
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_C12, PoseidonGoldilocksConstants::C12, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_M12, PoseidonGoldilocksConstants::M12, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_P12, PoseidonGoldilocksConstants::P12, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_S12, PoseidonGoldilocksConstants::S12, 507 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else if constexpr (W == 8) {
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_C8, PoseidonGoldilocksConstants::C8, 86 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_M8, PoseidonGoldilocksConstants::M8, 64 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_P8, PoseidonGoldilocksConstants::P8, 64 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_S8, PoseidonGoldilocksConstants::S8, 330 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else if constexpr (W == 16) {
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_C16, PoseidonGoldilocksConstants::C16, 150 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_M16, PoseidonGoldilocksConstants::M16, 256 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_P16, PoseidonGoldilocksConstants::P16, 256 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_POS1_S16, PoseidonGoldilocksConstants::S16, 683 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            }
        }
        initialized[W] = 1;
    }
    cudaSetDevice(saved);
}

// ---------------------------------------------------------------------------
// Public API: permute / permuteTrunc — <<<1, 1>>>, no shared mem.
// ---------------------------------------------------------------------------
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::permute(uint64_t *output, const uint64_t *input, cudaStream_t stream)
{
    permuteKernel_pos1<SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
        <<<1, 1, 0, stream>>>(output, input);
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t W>
void PoseidonGoldilocksGPU<W>::permuteTrunc(uint64_t *output, const uint64_t *input, cudaStream_t stream)
{
    permuteTruncKernel_pos1<SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS, CAPACITY>
        <<<1, 1, 0, stream>>>(output, input);
    CHECKCUDAERR(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Public API: linearHash — one thread per row, register-path permute.
// ---------------------------------------------------------------------------
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::linearHash(uint64_t *d_hash_output, uint64_t *d_trace,
                                          uint64_t num_cols, uint64_t num_rows,
                                          Layout layout, cudaStream_t stream, uint64_t ld)
{
    if (num_rows == 0) return;

    constexpr u32 tpbW = pos1Tpb<SPONGE_WIDTH>();
    u32 tpb = tpbW;
    u32 blks = (num_rows + tpbW - 1) / tpbW;
    if (num_rows < tpbW)
    {
        tpb = (u32)num_rows;
        blks = 1;
    }

    const size_t smem_bytes = (size_t)tpb * SPONGE_WIDTH * sizeof(uint64_t);

    // RowMajor reads contiguous columns per row (flat kernel); ColMajor and ColMajorTiled both go
    // through the getBufferOffset-based kernel, which honors the exact layout passed in.
    if (layout == Layout::RowMajor)
    {
        linearHashKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<blks, tpb, smem_bytes, stream>>>(d_hash_output, d_trace, (uint32_t)num_cols, (uint32_t)num_rows);
    }
    else
    {
        linearHashTiledKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS,
                                   pos1Tpb<SPONGE_WIDTH>(), POS1_TILED_MINB>
            <<<blks, tpb, smem_bytes, stream>>>(d_hash_output, d_trace, (uint32_t)num_cols, (uint32_t)num_rows, layout,
                                                (uint32_t)(ld ? ld : num_rows));
    }
    CHECKCUDAERR(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Public API: merkletree — linearHash leaves then arity-generic reduction.
// Uses arity*CAPACITY as the child stride (matches CPU merkletree_seq fix).
// ---------------------------------------------------------------------------
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                                          uint64_t num_cols, uint64_t num_rows,
                                          Layout layout, cudaStream_t stream)
{
    // arity*CAPACITY children must fit in SPONGE_WIDTH (the permuteTrunc input).
    assert(arity * 4 <= W &&
           "PoseidonGoldilocksGPU::merkletree: arity*CAPACITY exceeds SPONGE_WIDTH");
    if (num_rows == 0) return;
    linearHash(d_tree, d_input, num_cols, num_rows, layout, stream);
    reduceLevels(arity, d_tree, num_rows, stream);
}

// `pending` digests (a multiple of arity) at d_nodes -> their parents, written right after them.
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::reduceOneLevel(uint32_t arity, uint64_t *d_nodes, uint64_t pending, cudaStream_t stream)
{
    const uint64_t nextN = pending / arity;
    if (nextN <= POS1_MERKLE_WARP_MAX_NODES)
    {
        u32 tpb_w = 256;
        u32 blks_w = (u32)((nextN * 32 + tpb_w - 1) / tpb_w);
        merkleNodeWarpKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<blks_w, tpb_w, 0, stream>>>(nextN, 0, pending, arity, d_nodes);
    }
    else
    {
        u32 tpb = pos1Tpb<SPONGE_WIDTH>();
        u32 blks = (u32)((nextN + tpb - 1) / tpb);
        size_t smem_bytes = (size_t)tpb * SPONGE_WIDTH * sizeof(uint64_t);
        merkleNodeKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<blks, tpb, smem_bytes, stream>>>(nextN, 0, pending, arity, d_nodes);
    }
    CHECKCUDAERR(cudaGetLastError());
}

// Every level above the `pending` digests at the start of d_tree.
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::reduceLevels(uint32_t arity, uint64_t *d_tree, uint64_t pending, cudaStream_t stream)
{
    uint64_t nextIndex = 0;
    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync((uint64_t *)(d_tree + nextIndex + pending * CAPACITY),
                                         0, extraZeros * CAPACITY * sizeof(uint64_t), stream));
        reduceOneLevel(arity, d_tree + nextIndex, pending + extraZeros, stream);
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
    }
}

// ---------------------------------------------------------------------------
// Public API: merkletreeReduce — takes pre-hashed digests and reduces to root.
// ---------------------------------------------------------------------------
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::merkletreeReduce(uint64_t *d_root, uint64_t *d_input,
                                                uint64_t num_elements, uint64_t arity,
                                                cudaStream_t stream)
{
    assert(arity * 4 <= W &&
           "PoseidonGoldilocksGPU::merkletreeReduce: arity*CAPACITY exceeds SPONGE_WIDTH");
    // Compute total tree buffer size (matches the CPU merkletreeReduce layout).
    uint64_t numNodes = num_elements;
    uint64_t nodesLevel = num_elements;
    while (nodesLevel > 1) {
        uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
        numNodes += extraZeros;
        numNodes += (nodesLevel + (arity - 1)) / arity;
        nodesLevel = (nodesLevel + (arity - 1)) / arity;
    }

    uint64_t *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_tree, numNodes * CAPACITY * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyAsync(d_tree, d_input,
                                 num_elements * CAPACITY * sizeof(uint64_t),
                                 cudaMemcpyDeviceToDevice, stream));

    uint64_t pending = num_elements;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync(d_tree + nextIndex + pending * CAPACITY, 0,
                                         extraZeros * CAPACITY * sizeof(uint64_t), stream));

        if (nextN <= POS1_MERKLE_WARP_MAX_NODES)
        {
            u32 tpb_w = 256;
            u32 blks_w = (u32)((nextN * 32 + tpb_w - 1) / tpb_w);
            merkleNodeWarpKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
                <<<blks_w, tpb_w, 0, stream>>>(
                    nextN, nextIndex, pending + extraZeros, (uint32_t)arity, d_tree);
        }
        else
        {
            u32 tpb = pos1Tpb<SPONGE_WIDTH>();
            u32 blks = (u32)(nextN / tpb + 1);
            size_t smem_bytes = (size_t)tpb * SPONGE_WIDTH * sizeof(uint64_t);

            merkleNodeKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
                <<<blks, tpb, smem_bytes, stream>>>(
                    nextN, nextIndex, pending + extraZeros, (uint32_t)arity, d_tree);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());

    CHECKCUDAERR(cudaMemcpyAsync(d_root, d_tree + nextIndex,
                                 CAPACITY * sizeof(uint64_t),
                                 cudaMemcpyDeviceToDevice, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    CHECKCUDAERR(cudaFree(d_tree));
}

// ---------------------------------------------------------------------------
// Public API: grinding (see grinding_gpu.cuh).
// ---------------------------------------------------------------------------
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock,
                                        const uint64_t *d_in, const uint32_t n_bits,
                                        cudaStream_t stream)
{
    grindMin<Pos1GrindHash<SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>>(d_nonce, d_nonceBlock, d_in, n_bits, stream);
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// W=12: full (compatibility with existing default Poseidon1 path).---------------------------------------------------------------------------
template class PoseidonGoldilocksGPU<8>;
template class PoseidonGoldilocksGPU<12>;
template class PoseidonGoldilocksGPU<16>;

