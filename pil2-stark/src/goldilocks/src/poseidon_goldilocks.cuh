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
    gl64_t s0 = x[0] * C[0];
#pragma unroll
    for (int i = 1; i < (int)W; ++i)
        s0 = s0 + x[i] * C[i];
    return s0;
}

template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_(gl64_t *state, const gl64_t *mat)
{
    gl64_t old_state[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        old_state[i] = state[i];

#pragma unroll 1
    for (int i = 0; i < (int)W; ++i)
    {
        state[i] = mat[i] * old_state[0];
        for (int j = 1; j < (int)W; ++j)
            state[i] = state[i] + (mat[(int)W * j + i] * old_state[j]);
    }
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
        pos1_mvp_<W>(state, GPU_M_GL);
    }

    // Transition into partial rounds: pow7+ARK, MVP(P).
    pos1_pow7add_<W>(state, &GPU_C_GL[HALF_F * W]);
    pos1_mvp_<W>(state, GPU_P_GL);

    // Partial rounds.
    for (uint32_t r = 0; r < N_PART; ++r)
    {
        pow7(state[0]);
        state[0] = state[0] + GPU_C_GL[(HALF_F + 1) * W + r];

        gl64_t s0 = pos1_dot_<W>(state, &GPU_S_GL[(W * 2 - 1) * r]);

        gl64_t W_[W];
        pos1_prod_<W>(W_, state[0], &GPU_S_GL[(W * 2 - 1) * r + W - 1]);
        pos1_add_<W>(state, W_);
        state[0] = s0;
    }

    // Second half: (pow7+ARK, MVP(M)) × (HALF_F - 1), then pow7 + MVP(M).
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        pos1_pow7add_<W>(state, &GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W]);
        pos1_mvp_<W>(state, GPU_M_GL);
    }
    pos1_pow7_<W>(state);
    pos1_mvp_<W>(state, GPU_M_GL);

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

// MDS × state in shared memory. Indexing matches pos1_mvp_ (mat[W*j + i],
// transposed). Outer loop kept at unroll-1 on purpose: fully unrolling it
// spikes register use and drops occupancy.
template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_smem_(const gl64_t *mat)
{
    gl64_t s[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        s[i] = scratchpad[i * blockDim.x + threadIdx.x];

#pragma unroll 1
    for (uint32_t i = 0; i < W; ++i)
    {
        gl64_t acc = mat[i] * s[0];
#pragma unroll
        for (uint32_t j = 1; j < W; ++j)
            acc = acc + (mat[W * j + i] * s[j]);
        scratchpad[i * blockDim.x + threadIdx.x] = acc;
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
        pos1_mvp_smem_<W>(GPU_M_GL);
    }

    // Transition full round → MVP(P).
    pos1_pow7add_smem_<W>(&GPU_C_GL[HALF_F * W]);
    pos1_mvp_smem_<W>(GPU_P_GL);

    // Partial rounds — fused dot + rank-1 loop. state[0] is rewritten every
    // round, so keep it in a register across all rounds (read/write scratchpad
    // once). Caching the full state in registers instead spikes register use
    // and drops occupancy.
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
            gl64_t acc = s0_reg * S_row[0];
#pragma unroll
            for (uint32_t j = 1; j < W; ++j)
            {
                gl64_t sj = scratchpad[j * blockDim.x + threadIdx.x];
                acc = acc + sj * S_row[j];
                scratchpad[j * blockDim.x + threadIdx.x] = sj + s0_reg * S_row[W - 1 + j];
            }
            s0_reg = acc;
        }
        scratchpad[threadIdx.x] = s0_reg;
    }

    // Second half: (HALF_F - 1) × (pow7+ARK fused + MVP(M)), then pow7 + MVP(M).
#pragma unroll 1
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        pos1_pow7add_smem_<W>(&GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W]);
        pos1_mvp_smem_<W>(GPU_M_GL);
    }
    pos1_pow7_smem_<W>();
    pos1_mvp_smem_<W>(GPU_M_GL);

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
                                           uint32_t num_cols, uint32_t num_rows);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void merkleNodeKernel_pos1(uint64_t nextN, uint64_t nextIndex,
                                      uint64_t pending, uint32_t arity,
                                      uint64_t *cursor);

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void grindingKernel_pos1(uint64_t *nonce, uint64_t *__restrict__ nonceBlock,
                                    uint64_t *__restrict__ input, uint64_t n_bits,
                                    uint64_t hashes_per_thread, uint64_t nonces_offset);

#endif  // POSEIDON_GOLDILOCKS_CUH
