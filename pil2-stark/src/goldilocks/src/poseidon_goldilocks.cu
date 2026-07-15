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
#include "poseidon_goldilocks_constants.hpp"
#include "poseidon2_goldilocks.hpp"  // NONCES_LAUNCH_* grinding launch params (shared with Poseidon2)

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

// CUDA threads per block.
#define TPB_POS1 128

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
template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void linearHashTiledKernel_pos1(uint64_t *__restrict__ output,
                                           uint64_t *__restrict__ input,
                                           uint32_t num_cols, uint32_t num_rows, Layout layout)
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
            uint64_t idx = getBufferOffset(row, col0 + i, num_rows, num_cols, layout);
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

// Grinding kernel. Uses an in-register state and the register-path permute.
template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void grindingKernel_pos1(uint64_t *nonce,
                                    uint64_t *__restrict__ nonceBlock,
                                    uint64_t *__restrict__ input,
                                    uint64_t n_bits,
                                    uint64_t hashes_per_thread,
                                    uint64_t nonces_offset)
{
    if (nonces_offset != 0 && nonce[0] != UINT64_MAX)
        return;

    // Dynamic shared-mem scratchpad (declared by poseidon2_goldilocks.cuh).
    // We use it only for the per-block reduction of found nonces.
    uint64_t *shared_nonces = (uint64_t *)&scratchpad[0];

    if (threadIdx.x == 0)
    {
        shared_nonces[0] = UINT64_MAX;
        if (blockIdx.x == 0)
            nonce[0] = UINT64_MAX;
        if (nonces_offset != 0)
        {
            for (int i = 0; i < (int)gridDim.x; ++i)
            {
                if (nonceBlock[i] != UINT64_MAX)
                {
                    shared_nonces[0] = nonceBlock[i];
                    if (blockIdx.x == 0) nonce[0] = nonceBlock[i];
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (shared_nonces[0] != UINT64_MAX)
        return;

    nonceBlock[blockIdx.x] = UINT64_MAX;

    const uint64_t idx   = nonces_offset + ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * hashes_per_thread;
    const uint64_t level = 1ULL << (64 - n_bits);
    uint64_t locId = UINT64_MAX;

    gl64_t state[W];
    gl64_t in_reg[W];

#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        in_reg[i] = (gl64_t)(uint64_t)0;

    for (uint64_t k = 0; k < hashes_per_thread; ++k)
    {
        uint64_t idx_k = idx + k;
        // STARK grinding contract:
        //   in_reg[0..2] = FIELD_EXTENSION challenge
        //   in_reg[3]    = nonce
        //   in_reg[4..W-1] = 0 (already zero from init above)
#pragma unroll
        for (uint32_t i = 0; i < 3; ++i)
            in_reg[i] = input[i];
        in_reg[3] = (gl64_t)idx_k;

        poseidon1PermuteReg<W, HALF_F, N_PART>(
            state, in_reg,
            Pos1ConstGPU<W>::C(),
            Pos1ConstGPU<W>::S(),
            Pos1ConstGPU<W>::M(),
            Pos1ConstGPU<W>::P());

        uint64_t hash_val = (uint64_t)state[0];
        if (hash_val < level)
        {
            locId = idx_k;
            break;
        }
    }

    shared_nonces[threadIdx.x] = locId;
    __syncthreads();

    uint32_t alive = blockDim.x >> 1;
    while (alive > 0)
    {
        if (threadIdx.x < alive && shared_nonces[threadIdx.x + alive] < shared_nonces[threadIdx.x])
            shared_nonces[threadIdx.x] = shared_nonces[threadIdx.x + alive];
        __syncthreads();
        alive >>= 1;
    }
    if (threadIdx.x == 0)
        nonceBlock[blockIdx.x] = shared_nonces[0];
}

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
                                          Layout layout, cudaStream_t stream)
{
    if (num_rows == 0) return;

    u32 tpb = TPB_POS1;
    u32 blks = (num_rows + TPB_POS1 - 1) / TPB_POS1;
    if (num_rows < TPB_POS1)
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
        linearHashTiledKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<blks, tpb, smem_bytes, stream>>>(d_hash_output, d_trace, (uint32_t)num_cols, (uint32_t)num_rows, layout);
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

    u32 tpb = TPB_POS1;
    u32 blks = (num_rows + TPB_POS1 - 1) / TPB_POS1;
    if (num_rows < TPB_POS1)
    {
        tpb = (u32)num_rows;
        blks = 1;
    }

    size_t smem_bytes = (size_t)tpb * SPONGE_WIDTH * sizeof(uint64_t);

    // RowMajor reads contiguous columns per row (flat kernel); ColMajor and ColMajorTiled both go
    // through the getBufferOffset-based kernel, which honors the exact layout passed in.
    if (layout == Layout::RowMajor)
    {
        linearHashKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<blks, tpb, smem_bytes, stream>>>(d_tree, d_input, (uint32_t)num_cols, (uint32_t)num_rows);
    }
    else
    {
        linearHashTiledKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<blks, tpb, smem_bytes, stream>>>(d_tree, d_input, (uint32_t)num_cols, (uint32_t)num_rows, layout);
    }
    CHECKCUDAERR(cudaGetLastError());

    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync(
                (uint64_t *)(d_tree + nextIndex + pending * CAPACITY),
                0, extraZeros * CAPACITY * sizeof(uint64_t), stream));

        if (nextN < TPB_POS1) { tpb = (u32)nextN; blks = 1; }
        else                  { tpb = TPB_POS1;   blks = (u32)(nextN / TPB_POS1 + 1); }
        smem_bytes = (size_t)tpb * SPONGE_WIDTH * sizeof(uint64_t);

        merkleNodeKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<blks, tpb, smem_bytes, stream>>>(nextN, nextIndex,
                                                pending + extraZeros, arity, d_tree);

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
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

        u32 tpb = (nextN < TPB_POS1) ? (u32)nextN : TPB_POS1;
        u32 blks = (nextN < TPB_POS1) ? 1 : (u32)(nextN / TPB_POS1 + 1);
        size_t smem_bytes = (size_t)tpb * SPONGE_WIDTH * sizeof(uint64_t);

        merkleNodeKernel_pos1<RATE, CAPACITY, SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<blks, tpb, smem_bytes, stream>>>(
                nextN, nextIndex, pending + extraZeros, (uint32_t)arity, d_tree);

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
// Public API: grinding — same launch strategy as Poseidon2 (launch_iters
// rounds × NONCES_LAUNCH_GRID_SIZE blocks × NONCES_LAUNCH_BLOCKS threads).
// Shared mem: one uint64 per thread for the block-local nonce reduction.
// ---------------------------------------------------------------------------
template<uint32_t W>
void PoseidonGoldilocksGPU<W>::grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock,
                                        const uint64_t *d_in, const uint32_t n_bits,
                                        cudaStream_t stream)
{
    uint64_t log_launch_iters = 7;
    uint64_t launch_iters = 1ULL << log_launch_iters;
    uint64_t log_N = NONCES_LAUNCH_BITS;
    uint64_t N = 1ULL << log_N;
    uint64_t security = 128;

    // Numerical notes:
    //   eps    = 2^-n_bits is built with ldexp (exact; no integer shift).
    //   log1p  = ln(1 + x) without the cancellation that hits `log(1.0 - eps)`
    double eps                 = ldexp(1.0, -int(n_bits));
    double totalHashesRequired = -double(security) * log(2.0) / log1p(-eps);
    uint64_t log_totalHashesRequired = (uint64_t)ceil(log2(totalHashesRequired));
    uint64_t log_hashesPerThread = (log_totalHashesRequired > log_launch_iters + log_N)
                                   ? log_totalHashesRequired - log_launch_iters - log_N : 0;
    uint64_t hashesPerThread = 1ULL << log_hashesPerThread;

    dim3 blockSize(NONCES_LAUNCH_BLOCKS);
    dim3 gridSize(NONCES_LAUNCH_GRID_SIZE);

    // Only need blockDim.x uint64_t slots for the block-local reduction.
    size_t shared_mem_size = blockSize.x * sizeof(uint64_t);

    uint64_t nonces_offset = 0;
    uint64_t nonces_per_iteration = N * hashesPerThread;

    for (uint64_t i = 0; i < launch_iters; ++i)
    {
        grindingKernel_pos1<SPONGE_WIDTH, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS>
            <<<gridSize, blockSize, shared_mem_size, stream>>>(
                (uint64_t *)d_nonce, (uint64_t *)d_nonceBlock,
                (uint64_t *)d_in, n_bits, hashesPerThread, nonces_offset);
        nonces_offset += nonces_per_iteration;
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// W=12: full (compatibility with existing default Poseidon1 path).---------------------------------------------------------------------------
template class PoseidonGoldilocksGPU<8>;
template class PoseidonGoldilocksGPU<12>;
template class PoseidonGoldilocksGPU<16>;

