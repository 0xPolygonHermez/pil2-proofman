#ifndef POSEIDON_GOLDILOCKS_CUH
#define POSEIDON_GOLDILOCKS_CUH

// ---------------------------------------------------------------------------
// Poseidon v1 (Goldilocks, W=12) — GPU port.
//
// Shape mirrors poseidon2_goldilocks.cuh so both hash families share the same
// public surface (initConstants / permute / compress / linearHash / merkletree
// / merkletreeReduce / grinding).
//
// Algorithm source: pil2-stark 0.14.0 poseidon_goldilocks.cu (hash_full_result_seq)
// and the CPU reference in poseidon_goldilocks.cpp (permute_seq) — this file
// preserves that math byte-for-byte.
//
// We intentionally include poseidon2_goldilocks.cuh to reuse:
//   - the Layout enum (RowMajor / Tiles),
//   - the pow7(gl64_t&) device leaf,
//   - the extern __shared__ scratchpad declaration.
// No Poseidon2 kernel is called from here, but sharing the header avoids
// duplicate symbols and keeps the two paths co-installable.
// ---------------------------------------------------------------------------

#include "gl64_t.cuh"
#include "goldilocks_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "poseidon_goldilocks.hpp"
#include "goldilocks_trace_layout.cuh"
#include "poseidon2_goldilocks.cuh"  // brings: Layout, pow7(gl64_t&), scratchpad

// ---------------------------------------------------------------------------
// Public class — parameterised only at W=12 this iteration (matches CPU).
// ---------------------------------------------------------------------------

template<uint32_t SPONGE_WIDTH_T>
class PoseidonGoldilocksGPU
{
public:
    static_assert(SPONGE_WIDTH_T == 12,
                  "PoseidonGoldilocksGPU: only W=12 instantiated in this iteration");

    static constexpr uint32_t RATE                = SPONGE_WIDTH_T - 4;  // 8
    static constexpr uint32_t CAPACITY            = 4;
    static constexpr uint32_t SPONGE_WIDTH        = SPONGE_WIDTH_T;      // 12
    static constexpr uint32_t N_FULL_ROUNDS_TOTAL = 8;
    static constexpr uint32_t HALF_N_FULL_ROUNDS  = 4;
    static constexpr uint32_t N_PARTIAL_ROUNDS    = 22;
    static constexpr uint32_t N_ROUNDS            = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;

    static void initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);

    static void permute(uint64_t *output, const uint64_t *input, cudaStream_t stream = 0);
    static void compress(uint64_t *output, const uint64_t *input, cudaStream_t stream = 0);

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

using PoseidonGoldilocksGPUGrinding = PoseidonGoldilocksGPU<12>;

// ---------------------------------------------------------------------------
// Constant memory arrays GPU_POS1_{C,S,M,P} are defined in poseidon_goldilocks.cu.
// They are NOT forward-declared in this header: nvcc treats any declaration of a
// __constant__ symbol (even `extern`) as a definition, so a forward decl would
// cause a "redefinition" error at link time. Device helpers in this header take
// the constants through pointer parameters (see poseidon1PermuteReg), mirroring
// how poseidon2_goldilocks.cuh passes its GPU_C_GL / GPU_D_GL.

// ---------------------------------------------------------------------------
// Device leaf primitives — register-path Poseidon v1.
// Array-wide variants (over SPONGE_WIDTH_T = 12) used by the permutation.
// ---------------------------------------------------------------------------

// Note: pow7(gl64_t&) is already defined by poseidon2_goldilocks.cuh (same sig).

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

// Matrix-vector product on W=12. 0.14.0 indexes the flat matrix as
// mat[12*j + i] (i.e. transposed layout), so M_12/P_12 can be applied in
// this linearised form. DO NOT change this indexing — it's the load-bearing
// piece that reproduces the committed goldens.
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
// Full register-path permutation (matches 0.14.0 hash_full_result_seq).
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
//
// State layout: scratchpad[i * blockDim.x + threadIdx.x] = state element i of
// this thread. Matches the layout used by poseidon2_goldilocks.cuh and 0.14.0's
// linear_hash_gpu_coalesced.
//
// Win over the register path (poseidon1PermuteReg):
//   - No stack spill (the register path spills the state[12] array to local
//     memory — 96 bytes/thread visible in ptxas output).
//   - Partial rounds fuse dot+prod+add into a single inner loop (half the
//     shared-mem traffic of the split version).
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

// MDS * state_in_smem. Reads state once into registers, writes products back.
// Indexing matches register-path mvp_ (mat[W*j + i] = M[j][i], transposed-style).
// Outer loop is NOT fully unrolled: unrolling it (12 live accumulators +
// full expansion of inner × outer = 132 mul-adds) pushed register use to 255/
// thread and crashed occupancy. Keeping outer as unroll-1 but inner as full-
// unroll keeps the kernel at 48 regs/thread.
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

    // Partial rounds — fused dot + prod+add loop.
    //
    // Optimisation: state[0] is rewritten on every iteration anyway, so keep
    // it in a register across all N_PART rounds. Read from scratchpad once at
    // the start, write back once at the end. Saves (N_PART*2 - 1) = 43 smem
    // ops per permutation vs. the pure-smem version.
    //
    // Holding the *full* W-element state in registers blew up register count
    // to 255/thread and dropped occupancy by ~40% — see earlier note. Keeping
    // only state[0] in a register (1 extra register) is safe: compiler stays
    // at 48 regs/thread.
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
// Kernels — all declared here, defined in poseidon_goldilocks.cu.
// ---------------------------------------------------------------------------

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void permuteKernel_pos1(uint64_t *output, const uint64_t *input);

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART, uint32_t CAPACITY_T>
__global__ void compressKernel_pos1(uint64_t *output, const uint64_t *input);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void linearHashKernel_pos1(uint64_t *__restrict__ output,
                                      uint64_t *__restrict__ input,
                                      uint32_t num_cols, uint32_t num_rows);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void linearHashTiledKernel_pos1(uint64_t *__restrict__ output,
                                           uint64_t *__restrict__ input,
                                           uint32_t num_cols, uint32_t num_rows);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void merkleNodeKernel_pos1(uint32_t nextN, uint32_t nextIndex,
                                      uint32_t pending, uint32_t arity,
                                      uint64_t *cursor);

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void grindingKernel_pos1(uint64_t *nonce, uint64_t *__restrict__ nonceBlock,
                                    uint64_t *__restrict__ input, uint64_t n_bits,
                                    uint64_t hashes_per_thread, uint64_t nonces_offset);

#endif  // POSEIDON_GOLDILOCKS_CUH
