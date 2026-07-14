#ifndef POSEIDON_GOLDILOCKS_CUH
#define POSEIDON_GOLDILOCKS_CUH

// ---------------------------------------------------------------------------
// Poseidon v1 — GPU implementation.
// ---------------------------------------------------------------------------

#include "gl64_t.cuh"
#include "goldilocks_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "poseidon_goldilocks.hpp"
#include "goldilocks_trace_layout.cuh"
#include "poseidon_gpu_common.cuh"  // Layout, pow7(gl64_t&), scratchpad

// ---------------------------------------------------------------------------
// Public class (W=12).
// ---------------------------------------------------------------------------


template<uint32_t SPONGE_WIDTH_T>
class PoseidonGoldilocksGPU
{
public:
    static_assert(SPONGE_WIDTH_T == 8 || SPONGE_WIDTH_T == 12 || SPONGE_WIDTH_T == 16,
                  "PoseidonGoldilocksGPU: SPONGE_WIDTH_T must be 8, 12, or 16");

    static constexpr uint32_t CAPACITY            = 4;
    static constexpr uint32_t RATE                = SPONGE_WIDTH_T - CAPACITY;
    static constexpr uint32_t SPONGE_WIDTH        = SPONGE_WIDTH_T;
    static constexpr uint32_t N_FULL_ROUNDS_TOTAL = 8;
    static constexpr uint32_t HALF_N_FULL_ROUNDS  = 4;
    static constexpr uint32_t N_PARTIAL_ROUNDS    = PoseidonGoldilocksConstants::Poseidon1Tables<SPONGE_WIDTH_T>::N_PARTIAL_ROUNDS;
    static constexpr uint32_t N_ROUNDS            = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;

    static void initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);

    static void permute(uint64_t *output, const uint64_t *input, cudaStream_t stream = 0);
    static void permuteTrunc(uint64_t *output, const uint64_t *input, cudaStream_t stream = 0);

    static void linearHash(uint64_t *d_hash_output, uint64_t *d_trace,
                           uint64_t num_cols, uint64_t num_rows,
                           Layout layout, cudaStream_t stream);

    static void merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                           uint64_t num_cols, uint64_t num_rows,
                           Layout layout, cudaStream_t stream);

    static void merkletreeReduce(uint64_t *d_root, uint64_t *d_input,
                                 uint64_t num_elements, uint64_t arity,
                                 cudaStream_t stream);

    static void grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock,
                         const uint64_t *d_in, const uint32_t n_bits,
                         cudaStream_t stream);
};

using PoseidonGoldilocksGPUGrinding = PoseidonGoldilocksGPU<8>;

// ---------------------------------------------------------------------------
// Lazy-reduction arithmetic
//
// The MDS matrices M_8/M_12/M_16 have tiny entries (<= 7 bits), yet the naive
// path pays a full 64x64 modular multiply + reduction per term. Instead we
// accumulate the raw products wide and do a single modular reduction per
// dot product:
//   - M (tiny constants): compile-time immediates, sum fits u128 -> reduce128
//     (see the pos1_*_mimm_ helpers below).
//   - full-width constants (P, S): 16 x 128-bit products need a carry limb
//     (sum < 2^132) -> reduce160 (handles up to c*2^128 + 128 bits, c < 2^32),
//     using 2^128 == -2^32 (mod p).
// The reductions produce canonical results; inputs may be any u64.
// ---------------------------------------------------------------------------

// 2^64 mod p = 2^32 - 1 (the "epsilon" of the Goldilocks field): the
// correction to apply whenever a value crosses the 2^64 boundary (borrow or
// carry) during reduction.
__device__ constexpr uint64_t POS1_EPSILON = 0xFFFFFFFFULL;

// (hi*2^64 + lo) mod p, canonical. hi = hh*2^32 + hl:
// v == lo - hh + hl*(2^32 - 1)  (mod p)
__device__ __forceinline__ uint64_t pos1_reduce128_(uint64_t hi, uint64_t lo)
{
    uint32_t hh = (uint32_t)(hi >> 32);
    uint32_t hl = (uint32_t)hi;
    uint64_t t0 = lo - hh;
    if (lo < hh) t0 -= POS1_EPSILON;          // borrow: the wrap added 2^64 == epsilon, take it back
    uint64_t t1 = ((uint64_t)hl << 32) - hl;  // hl * (2^32 - 1), fits u64
    uint64_t r = t0 + t1;
    if (r < t1) r += POS1_EPSILON;            // carry: the wrap dropped 2^64 == epsilon, restore it
    if (r >= GOLDILOCKS_PRIME) r -= GOLDILOCKS_PRIME;
    return r;
}

// (c*2^128 + hi*2^64 + lo) mod p, canonical, using 2^128 == -2^32 (mod p).
__device__ __forceinline__ uint64_t pos1_reduce160_(uint32_t c, uint64_t hi, uint64_t lo)
{
    uint64_t r = pos1_reduce128_(hi, lo);
    uint64_t sub = ((uint64_t)c) << 32;
    r = (r >= sub) ? (r - sub) : (r + GOLDILOCKS_PRIME - sub);
    if (r >= GOLDILOCKS_PRIME) r -= GOLDILOCKS_PRIME;
    return r;
}

// Dot product with full-width constants: u128 + carry-limb accumulation.
template<uint32_t W>
__device__ __forceinline__ uint64_t pos1_dot_wide_raw_(const uint64_t *s, const gl64_t *col, uint32_t stride)
{
    unsigned __int128 acc = (unsigned __int128)s[0] * col[0][0];
    uint32_t cy = 0;
#pragma unroll
    for (uint32_t j = 1; j < W; ++j)
    {
        unsigned __int128 pr = (unsigned __int128)s[j] * col[j * stride][0];
        acc += pr;
        cy += (acc < pr);
    }
    return pos1_reduce160_(cy, (uint64_t)(acc >> 64), (uint64_t)acc);
}

// x + a*b with a single reduction. Bound: (2^64-1)^2 + (2^64-1) < 2^128.
__device__ __forceinline__ uint64_t pos1_muladd_raw_(uint64_t x, uint64_t a, uint64_t b)
{
    unsigned __int128 v = (unsigned __int128)a * b + x;
    return pos1_reduce128_((uint64_t)(v >> 64), (uint64_t)v);
}

// MDS dot product with the M matrix as COMPILE-TIME immediates: the M
// entries are tiny (<= 7 bits) constexpr values, so with the row loop
// expanded via template recursion every term is a multiply by a small
// literal, which the compiler strength-reduces far below a runtime 64x64
// product (entries equal to 1 become plain adds). The entry is extracted
// through a constexpr function so it is frontend-evaluated -- no device-side
// access to the host constexpr array.
template<uint32_t W>
__host__ __device__ constexpr uint64_t pos1_m_entry_(uint32_t j, uint32_t i)
{
    return PoseidonGoldilocksConstants::Poseidon1Tables<W>::M[j][i].fe;
}

template<uint32_t W, uint32_t I, uint32_t J = 0>
__device__ __forceinline__ void pos1_dot_mimm_rows_(unsigned __int128 &acc, const uint64_t *s)
{
    if constexpr (J < W)
    {
        constexpr uint64_t m = pos1_m_entry_<W>(J, I);
        if constexpr (m == 1)
            acc += s[J];
        else if constexpr (m != 0)
            acc += (unsigned __int128)s[J] * m;
        pos1_dot_mimm_rows_<W, I, J + 1>(acc, s);
    }
}

template<uint32_t W, uint32_t I>
__device__ __forceinline__ uint64_t pos1_dot_mimm_(const uint64_t *s)
{
    unsigned __int128 acc = 0;
    pos1_dot_mimm_rows_<W, I>(acc, s);
    return pos1_reduce128_((uint64_t)(acc >> 64), (uint64_t)acc);
}

// Immediate-MDS x state, shared-memory state. out[I] = sum_j M[j][I]*s[j]:
// pos1_dot_mimm_rows_ recurses over the matrix rows J of one column, and
// the _cols_ helpers recurse over the columns I (one per output word) -- both
// compile-time so M[J][I] stays a literal.
template<uint32_t W, uint32_t I = 0>
__device__ __forceinline__ void pos1_mvp_smem_mimm_cols_(const uint64_t *s)
{
    if constexpr (I < W)
    {
        gl64_t r;
        r[0] = pos1_dot_mimm_<W, I>(s);
        scratchpad[I * blockDim.x + threadIdx.x] = r;
        pos1_mvp_smem_mimm_cols_<W, I + 1>(s);
    }
}

template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_smem_mimm_()
{
    uint64_t s[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        s[i] = scratchpad[i * blockDim.x + threadIdx.x][0];
    pos1_mvp_smem_mimm_cols_<W>(s);
}

// Immediate-MDS x state, register state.
template<uint32_t W, uint32_t I = 0>
__device__ __forceinline__ void pos1_mvp_mimm_cols_(gl64_t *state, const uint64_t *olds)
{
    if constexpr (I < W)
    {
        state[I][0] = pos1_dot_mimm_<W, I>(olds);
        pos1_mvp_mimm_cols_<W, I + 1>(state, olds);
    }
}

template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_mimm_(gl64_t *state)
{
    uint64_t olds[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        olds[i] = state[i][0];
    pos1_mvp_mimm_cols_<W>(state, olds);
}

template<uint32_t W>
__device__ __forceinline__ void pos1_pow7_(gl64_t *x)
{
    gl64_t x2[W], x3[W], x4[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i]  = x3[i] * x4[i];
    }
}

template<uint32_t W>
__device__ __forceinline__ void pos1_add_(gl64_t *x, const gl64_t *C)
{
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        x[i] = x[i] + C[i];
}

template<uint32_t W>
__device__ __forceinline__ void pos1_prod_(gl64_t *x, const gl64_t alpha, const gl64_t *C)
{
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        x[i] = alpha * C[i];
}

template<uint32_t W>
__device__ __forceinline__ void pos1_pow7add_(gl64_t *x, const gl64_t *C)
{
    gl64_t x2[W], x3[W], x4[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i]  = x3[i] * x4[i];
        x[i]  = x[i] + C[i];
    }
}

template<uint32_t W>
__device__ __forceinline__ gl64_t pos1_dot_(const gl64_t *x, const gl64_t *C)
{
    uint64_t s[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        s[i] = x[i][0];
    gl64_t r;
    r[0] = pos1_dot_wide_raw_<W>(s, C, 1);
    return r;
}

template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_(gl64_t *state, const gl64_t *mat)
{
    uint64_t old_state[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        old_state[i] = state[i][0];

#pragma unroll 1
    for (int i = 0; i < (int)W; ++i)
        state[i][0] = pos1_dot_wide_raw_<W>(old_state, &mat[i], W);
}

// ---------------------------------------------------------------------------
// Full register-path permutation.
// ---------------------------------------------------------------------------

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__device__ void poseidon1PermuteReg(gl64_t *state,
                                    const gl64_t *input,
                                    const gl64_t *GPU_C_GL,
                                    const gl64_t *GPU_S_GL,
                                    const gl64_t *GPU_M_GL,
                                    const gl64_t *GPU_P_GL)
{

    mymemcpy((uint64_t *)state, (uint64_t *)input, W);

    // First half: ARK -> (pow7+ARK, MVP(M)) × (HALF_F - 1).
    pos1_add_<W>(state, GPU_C_GL);
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        pos1_pow7add_<W>(state, &GPU_C_GL[(r + 1) * W]);
        pos1_mvp_mimm_<W>(state);
    }

    // Transition into partial rounds: pow7+ARK, MVP(P).
    pos1_pow7add_<W>(state, &GPU_C_GL[HALF_F * W]);
    pos1_mvp_<W>(state, GPU_P_GL);

    // Partial rounds: dot + rank-1 update, both with lazy reduction.
    for (uint32_t r = 0; r < N_PART; ++r)
    {
        pow7(state[0]);
        state[0] = state[0] + GPU_C_GL[(HALF_F + 1) * W + r];

        const gl64_t *S_row = &GPU_S_GL[(W * 2 - 1) * r];
        gl64_t s0 = pos1_dot_<W>(state, S_row);

        uint64_t a = state[0][0];
#pragma unroll
        for (uint32_t j = 1; j < W; ++j)
            state[j][0] = pos1_muladd_raw_(state[j][0], a, S_row[W - 1 + j][0]);
        state[0] = s0;
    }

    // Second half: (pow7+ARK, MVP(M)) × (HALF_F - 1), then pow7 + MVP(M).
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        pos1_pow7add_<W>(state, &GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W]);
        pos1_mvp_mimm_<W>(state);
    }
    pos1_pow7_<W>(state);
    pos1_mvp_mimm_<W>(state);

}

// ---------------------------------------------------------------------------
// Shared-memory variant of the permutation.
// State layout: scratchpad[i * blockDim.x + threadIdx.x] = state element i of
// this thread (one thread per sponge). Avoids the register path's stack spill,
// and fuses dot + rank-1 in the partial rounds.
// ---------------------------------------------------------------------------

template<uint32_t W>
__device__ __forceinline__ void pos1_pow7_smem_()
{
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
    {
        gl64_t x  = scratchpad[i * blockDim.x + threadIdx.x];
        gl64_t x2 = x * x;
        gl64_t x3 = x * x2;
        gl64_t x4 = x2 * x2;
        scratchpad[i * blockDim.x + threadIdx.x] = x3 * x4;
    }
}

template<uint32_t W>
__device__ __forceinline__ void pos1_add_smem_(const gl64_t *C)
{
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        scratchpad[i * blockDim.x + threadIdx.x] =
            scratchpad[i * blockDim.x + threadIdx.x] + C[i];
}

// Fused pow7 + ARK: state[i] = pow7(state[i]) + C[i], single read / single
// write per element (vs. pow7_smem_ + add_smem_ doing two passes).
template<uint32_t W>
__device__ __forceinline__ void pos1_pow7add_smem_(const gl64_t *C)
{
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
    {
        gl64_t x  = scratchpad[i * blockDim.x + threadIdx.x];
        gl64_t x2 = x * x;
        gl64_t x3 = x * x2;
        gl64_t x4 = x2 * x2;
        scratchpad[i * blockDim.x + threadIdx.x] = (x3 * x4) + C[i];
    }
}

// P x state in shared memory. Indexing matches pos1_mvp_ (mat[W*j + i],
// transposed). Outer loop kept at unroll-1 on purpose: fully unrolling it
// spikes register use and drops occupancy.
template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_smem_(const gl64_t *mat)
{
    uint64_t s[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        s[i] = scratchpad[i * blockDim.x + threadIdx.x][0];

#pragma unroll 1
    for (uint32_t i = 0; i < W; ++i)
    {
        gl64_t r;
        r[0] = pos1_dot_wide_raw_<W>(s, &mat[i], W);
        scratchpad[i * blockDim.x + threadIdx.x] = r;
    }
}

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__device__ void poseidon1PermuteSmem(const gl64_t *GPU_C_GL,
                                     const gl64_t *GPU_S_GL,
                                     const gl64_t *GPU_M_GL,
                                     const gl64_t *GPU_P_GL)
{


    // First half: initial ARK, then (HALF_F - 1) × (pow7+ARK fused + MVP(M)).
    pos1_add_smem_<W>(GPU_C_GL);
#pragma unroll 1
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        pos1_pow7add_smem_<W>(&GPU_C_GL[(r + 1) * W]);
        pos1_mvp_smem_mimm_<W>();
    }

    // Transition full round → MVP(P).
    pos1_pow7add_smem_<W>(&GPU_C_GL[HALF_F * W]);
    pos1_mvp_smem_<W>(GPU_P_GL);

    // Partial rounds — fused dot + rank-1 loop, both lazily reduced. state[0]
    // is rewritten every round, so keep it in a register across all rounds
    // (read/write scratchpad once). Caching the full state in registers
    // instead spikes register use and drops occupancy.
    {
        gl64_t s0_reg = scratchpad[threadIdx.x];
#pragma unroll 1
        for (uint32_t r = 0; r < N_PART; ++r)
        {
            gl64_t x2 = s0_reg * s0_reg;
            gl64_t x3 = s0_reg * x2;
            gl64_t x4 = x2 * x2;
            s0_reg = (x3 * x4) + GPU_C_GL[(HALF_F + 1) * W + r];

            const gl64_t *S_row = &GPU_S_GL[(W * 2 - 1) * r];
            uint64_t a = s0_reg[0];
            unsigned __int128 acc = (unsigned __int128)a * S_row[0][0];
            uint32_t cy = 0;
#pragma unroll
            for (uint32_t j = 1; j < W; ++j)
            {
                uint64_t sj = scratchpad[j * blockDim.x + threadIdx.x][0];
                unsigned __int128 pr = (unsigned __int128)sj * S_row[j][0];
                acc += pr;
                cy += (acc < pr);
                gl64_t nsj;
                nsj[0] = pos1_muladd_raw_(sj, a, S_row[W - 1 + j][0]);
                scratchpad[j * blockDim.x + threadIdx.x] = nsj;
            }
            s0_reg[0] = pos1_reduce160_(cy, (uint64_t)(acc >> 64), (uint64_t)acc);
        }
        scratchpad[threadIdx.x] = s0_reg;
    }

    // Second half: (HALF_F - 1) × (pow7+ARK fused + MVP(M)), then pow7 + MVP(M).
#pragma unroll 1
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        pos1_pow7add_smem_<W>(&GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W]);
        pos1_mvp_smem_mimm_<W>();
    }
    pos1_pow7_smem_<W>();
    pos1_mvp_smem_mimm_<W>();

}

// ---------------------------------------------------------------------------
// Register-resident warp-cooperative permutation (one thread per state lane).
//
// A single sponge state is spread across lanes 0..W-1 of one warp: each lane
// keeps its state element in a register and pulls peer lanes' values with
// __shfl_sync. No shared memory, no __syncwarp (the shuffle's mask provides the
// ordering). `mask` must name exactly the active lanes (0..W-1). Entered only
// by active lanes; returns this lane's output.
// ---------------------------------------------------------------------------

// MVP: out_lane = sum_j mat[W*j + lane] * v_j, gathering v_j over the warp.
template<uint32_t W>
__device__ __forceinline__ gl64_t pos1_mvp_warp(gl64_t v, const gl64_t *mat, uint32_t lane, uint32_t mask)
{
    gl64_t acc = mat[lane] * shfl_gl(mask, v, 0);
#pragma unroll
    for (uint32_t j = 1; j < W; ++j)
        acc = acc + mat[W * j + lane] * shfl_gl(mask, v, j);
    return acc;
}

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__device__ gl64_t poseidon1PermuteWarpReg(gl64_t v,
                                          uint32_t mask,
                                          const gl64_t *GPU_C_GL,
                                          const gl64_t *GPU_S_GL,
                                          const gl64_t *GPU_M_GL,
                                          const gl64_t *GPU_P_GL)
{
    const uint32_t lane = threadIdx.x;

    // ARK.
    v = v + GPU_C_GL[lane];

    // First half full rounds.
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        gl64_t x2 = v * v, x3 = v * x2, x4 = x2 * x2;
        v = (x3 * x4) + GPU_C_GL[(r + 1) * W + lane];
        v = pos1_mvp_warp<W>(v, GPU_M_GL, lane, mask);
    }

    // Transition full round → MVP(P).
    {
        gl64_t x2 = v * v, x3 = v * x2, x4 = x2 * x2;
        v = (x3 * x4) + GPU_C_GL[HALF_F * W + lane];
        v = pos1_mvp_warp<W>(v, GPU_P_GL, lane, mask);
    }

    // Partial rounds. Lane 0 owns the S-box; `a` (post-pow7 state[0]) drives the
    // sparse rank-1 update. The dot product s0 = Σ_j v_j·S_row[j] is formed as a
    // per-lane term then butterfly-reduced over the warp (log2(W) shuffles).
    for (uint32_t r = 0; r < N_PART; ++r)
    {
        if (lane == 0)
        {
            pow7(v);
            v = v + GPU_C_GL[(HALF_F + 1) * W + r];
        }
        gl64_t a = shfl_gl(mask, v, 0);   // post-pow7 state[0], broadcast to all

        const gl64_t *S_row = &GPU_S_GL[(W * 2 - 1) * r];
        gl64_t term = v * S_row[lane];    // this lane's contribution (lane 0: a·S_row[0])
        gl64_t s0 = warp_allreduce_add<W>(term, mask);

        v = (lane == 0) ? s0 : (v + a * S_row[W - 1 + lane]);
    }

    // Second half full rounds.
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        gl64_t x2 = v * v, x3 = v * x2, x4 = x2 * x2;
        v = (x3 * x4) + GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W + lane];
        v = pos1_mvp_warp<W>(v, GPU_M_GL, lane, mask);
    }
    // Final pow7 (no ARK) + MVP(M).
    {
        gl64_t x2 = v * v, x3 = v * x2, x4 = x2 * x2;
        v = x3 * x4;
        v = pos1_mvp_warp<W>(v, GPU_M_GL, lane, mask);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void permuteKernel_pos1(uint64_t *output, const uint64_t *input);

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART, uint32_t CAPACITY_T>
__global__ void permuteTruncKernel_pos1(uint64_t *output, const uint64_t *input);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void linearHashKernel_pos1(uint64_t *__restrict__ output,
                                      uint64_t *__restrict__ input,
                                      uint32_t num_cols, uint32_t num_rows);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void linearHashTiledKernel_pos1(uint64_t *__restrict__ output,
                                           uint64_t *__restrict__ input,
                                           uint32_t num_cols, uint32_t num_rows, Layout layout);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void merkleNodeKernel_pos1(uint64_t nextN, uint64_t nextIndex,
                                      uint64_t pending, uint32_t arity,
                                      uint64_t *cursor);

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void grindingKernel_pos1(uint64_t *nonce, uint64_t *__restrict__ nonceBlock,
                                    uint64_t *__restrict__ input, uint64_t n_bits,
                                    uint64_t hashes_per_thread, uint64_t nonces_offset);

#endif  // POSEIDON_GOLDILOCKS_CUH
