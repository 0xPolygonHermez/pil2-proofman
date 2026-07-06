#ifndef POSEIDON2_GOLDILOCKS_CUH
#define POSEIDON2_GOLDILOCKS_CUH

#include "gl64_t.cuh"
#include "goldilocks_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "poseidon2_goldilocks.hpp"
#include "goldilocks_trace_layout.cuh"  // Layout enum, getBufferOffset
#include "poseidon_gpu_common.cuh"      // pow7(gl64_t&), scratchpad



template<uint32_t SPONGE_WIDTH_T>
class Poseidon2GoldilocksGPU  {
public:
    
    static constexpr uint32_t RATE = SPONGE_WIDTH_T-4;
    static constexpr uint32_t CAPACITY = 4;
    static constexpr uint32_t SPONGE_WIDTH = SPONGE_WIDTH_T;
    static constexpr uint32_t N_FULL_ROUNDS_TOTAL = 8;
    static constexpr uint32_t HALF_N_FULL_ROUNDS = N_FULL_ROUNDS_TOTAL / 2;
    static constexpr uint32_t N_PARTIAL_ROUNDS = SPONGE_WIDTH_T == 4 ? 21 : 22;
    static constexpr uint32_t N_ROUNDS = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;

    static_assert(SPONGE_WIDTH_T == 4 || SPONGE_WIDTH_T == 8 || SPONGE_WIDTH_T == 12 || SPONGE_WIDTH_T == 16, "SPONGE_WIDTH_T must be 4, 8, 12, or 16");

    void static initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);

    void static permute(uint64_t * output, const uint64_t * input, cudaStream_t stream = 0);

    void static permuteTrunc(uint64_t * output, const uint64_t * input, cudaStream_t stream = 0);

    void static linearHash(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, Layout layout, cudaStream_t stream);

    void static merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, Layout layout, cudaStream_t stream);

    void static merkletreeReduce(uint64_t * d_root, uint64_t * d_input, uint64_t num_elements, uint64_t arity, cudaStream_t stream);

    void static grinding(uint64_t * d_nonce, uint64_t *d_nonceBlock, const uint64_t * d_in, const uint32_t n_bits, cudaStream_t stream);

};

using Poseidon2GoldilocksGPUGrinding = Poseidon2GoldilocksGPU<8>;  // SPONGE_WIDTH = 8



// --- Forward declarations (defined in .cu) ---

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ void poseidon2PermuteSmem();

// --- Leaf device helpers ---

__device__ __forceinline__ void mds4x4(gl64_t *x)
{
    gl64_t t0 = x[0] + x[1];
    gl64_t t1 = x[2] + x[3];
    gl64_t t2 = x[1] + x[1] + t1;
    gl64_t t3 = x[3] + x[3] + t0;
    gl64_t t1_2 = t1 + t1;
    gl64_t t0_2 = t0 + t0;
    gl64_t t4 = t1_2 + t1_2 + t3;
    gl64_t t5 = t0_2 + t0_2 + t2;
    gl64_t t6 = t3 + t5;
    gl64_t t7 = t2 + t4;

    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

__device__ __forceinline__ void mds4x4Smem(uint32_t offset)
{
    gl64_t t0 = scratchpad[(offset + 0) * blockDim.x + threadIdx.x] + scratchpad[(offset + 1) * blockDim.x + threadIdx.x];
    gl64_t t1 = scratchpad[(offset + 2) * blockDim.x + threadIdx.x] + scratchpad[(offset + 3) * blockDim.x + threadIdx.x];
    gl64_t t2 = scratchpad[(offset + 1) * blockDim.x + threadIdx.x] + scratchpad[(offset + 1) * blockDim.x + threadIdx.x] + t1;
    gl64_t t3 = scratchpad[(offset + 3) * blockDim.x + threadIdx.x] + scratchpad[(offset + 3) * blockDim.x + threadIdx.x] + t0;
    gl64_t t1_2 = t1 + t1;
    gl64_t t0_2 = t0 + t0;
    gl64_t t4 = t1_2 + t1_2 + t3;
    gl64_t t5 = t0_2 + t0_2 + t2;
    gl64_t t6 = t3 + t5;
    gl64_t t7 = t2 + t4;

    scratchpad[(offset + 0) * blockDim.x + threadIdx.x] = t6;
    scratchpad[(offset + 1) * blockDim.x + threadIdx.x] = t5;
    scratchpad[(offset + 2) * blockDim.x + threadIdx.x] = t7;
    scratchpad[(offset + 3) * blockDim.x + threadIdx.x] = t4;
}

// --- Register-based Poseidon2 helpers ---

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void sumAll(gl64_t *x, const gl64_t C[SPONGE_WIDTH_T])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH_T; ++i)
    {
        x[0] = x[0] + C[i];
    }
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void sboxFull(gl64_t *x, const gl64_t C[SPONGE_WIDTH_T])
{

#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH_T; ++i)
    {
        gl64_t xi = x[i] + C[i];
        gl64_t x2 = xi * xi;
        gl64_t x3 = xi * x2;
        gl64_t x4 = x2 * x2;
        x[i] = x3 * x4;
    }
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void mdsExternal(gl64_t *x)
{
#pragma unroll
    for(int i=0; i<SPONGE_WIDTH_T; i+=4)
        mds4x4(&x[i]);

    if( SPONGE_WIDTH_T > 4 ){
        gl64_t stored[4]={gl64_t(uint64_t(0)), gl64_t(uint64_t(0)), gl64_t(uint64_t(0)), gl64_t(uint64_t(0))};
        for(int i=0; i<SPONGE_WIDTH_T; i+=4){
            stored[0] = stored[0] + x[i + 0];
            stored[1] = stored[1] + x[i + 1];
            stored[2] = stored[2] + x[i + 2];
            stored[3] = stored[3] + x[i + 3];
        }
    #pragma unroll
        for (int i = 0; i < SPONGE_WIDTH_T; ++i)
        {
            x[i] = x[i] + stored[i & 3];
        }
    }
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void partialRoundMul(gl64_t *x, const gl64_t D[SPONGE_WIDTH_T], const gl64_t &sum)
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH_T; ++i)
    {
        x[i] = x[i] * D[i] + sum;
    }
}

// --- Shared-memory Poseidon2 helpers ---

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void sumAllSmem(gl64_t *x)
{
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH_T; i++)
       x[0]= x[0] + scratchpad[i * blockDim.x + threadIdx.x];
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void mdsExternalSmem()
{
#pragma unroll
    for(int i=0; i<SPONGE_WIDTH_T; i+=4){
        mds4x4Smem(i);
    }
    if(SPONGE_WIDTH_T > 4){
        gl64_t stored[4]={gl64_t(uint64_t(0)), gl64_t(uint64_t(0)), gl64_t(uint64_t(0)), gl64_t(uint64_t(0))};
        for(int i=0; i<SPONGE_WIDTH_T; i+=4){
            stored[0] = stored[0] + scratchpad[(i + 0) * blockDim.x + threadIdx.x];
            stored[1] = stored[1] + scratchpad[(i + 1) * blockDim.x + threadIdx.x];
            stored[2] = stored[2] + scratchpad[(i + 2) * blockDim.x + threadIdx.x];
            stored[3] = stored[3] + scratchpad[(i + 3) * blockDim.x + threadIdx.x];
        }
    #pragma unroll
        for (int i = 0; i < SPONGE_WIDTH_T; ++i)
        {
            scratchpad[i * blockDim.x + threadIdx.x] = scratchpad[i * blockDim.x + threadIdx.x] + stored[i & 3];
        }
    }
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void sboxFullSmem(const gl64_t C[SPONGE_WIDTH_T])
{

#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH_T; ++i)
    {
        gl64_t xi = scratchpad[i * blockDim.x + threadIdx.x] + C[i];
        gl64_t x2 = xi * xi;
        gl64_t x3 = xi * x2;
        gl64_t x4 = x2 * x2;
        scratchpad[i * blockDim.x + threadIdx.x] = x3 * x4;
    }
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void partialRoundMulSmem(const gl64_t D[SPONGE_WIDTH_T], const gl64_t &sum)
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH_T; ++i)
    {
        scratchpad[i * blockDim.x + threadIdx.x] = scratchpad[i * blockDim.x + threadIdx.x] * D[i] + sum;
    }
}

// --- Poseidon2 permutation ---

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void poseidon2PermuteReg(gl64_t *state, const gl64_t *input, const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL)
{

    mymemcpy((uint64_t *)state, (uint64_t *)input, SPONGE_WIDTH_T);

    mdsExternal<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(state);

    for (int r = 0; r < (N_FULL_ROUNDS_TOTAL_T>>1); r++)
    {
        sboxFull<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(state, &(GPU_C_GL[r * SPONGE_WIDTH_T]));
        mdsExternal<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T> (state);
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS_T; r++)
    {
        state[0] = state[0] + GPU_C_GL[(N_FULL_ROUNDS_TOTAL_T>>1) * SPONGE_WIDTH_T + r];
        pow7(state[0]);
        gl64_t sum_;
        sum_ = gl64_t(uint64_t(0));
        sumAll<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(&sum_, state);
        partialRoundMul<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(state, GPU_D_GL, sum_);
    }

    for (int r = 0; r < (N_FULL_ROUNDS_TOTAL_T>>1); r++)
    {
        sboxFull<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(state, &(GPU_C_GL[(N_FULL_ROUNDS_TOTAL_T>>1) * SPONGE_WIDTH_T + N_PARTIAL_ROUNDS_T + r * SPONGE_WIDTH_T]));
        mdsExternal<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(state);
    }

}

// Register-resident warp-cooperative permutation (one thread per state lane).
// Each lane keeps its state element in a register; peer lanes' values are
// pulled with __shfl_sync. No shared memory, no __syncwarp. `mask` must name
// exactly the active lanes (0..W-1). Entered only by active lanes; returns this
// lane's output.
// ---------------------------------------------------------------------------

// External MDS on register-resident state: 4x4 mix per block, then (W>4) the
// cross-block fold. All reads gathered via shuffle.
template<uint32_t W>
__device__ __forceinline__ gl64_t mdsExternalWarpReg(gl64_t v, uint32_t lane, uint32_t mask)
{
    uint32_t b = lane & ~3u, k = lane & 3u;
    gl64_t x0 = shfl_gl(mask, v, b);
    gl64_t x1 = shfl_gl(mask, v, b + 1);
    gl64_t x2 = shfl_gl(mask, v, b + 2);
    gl64_t x3 = shfl_gl(mask, v, b + 3);
    gl64_t t0 = x0 + x1;
    gl64_t t1 = x2 + x3;
    gl64_t t2 = x1 + x1 + t1;
    gl64_t t3 = x3 + x3 + t0;
    gl64_t t1_2 = t1 + t1;
    gl64_t t0_2 = t0 + t0;
    gl64_t t4 = t1_2 + t1_2 + t3;
    gl64_t t5 = t0_2 + t0_2 + t2;
    gl64_t t6 = t3 + t5;
    gl64_t t7 = t2 + t4;
    gl64_t r = (k == 0) ? t6 : (k == 1) ? t5 : (k == 2) ? t7 : t4;

    if (W > 4)
    {
        gl64_t acc = gl64_t(uint64_t(0));
#pragma unroll
        for (uint32_t blk = 0; blk < W; blk += 4)
            acc = acc + shfl_gl(mask, r, blk + k);
        r = r + acc;
    }
    return r;
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ gl64_t poseidon2PermuteWarpReg(gl64_t v, uint32_t mask, const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL)
{
    constexpr uint32_t W = SPONGE_WIDTH_T;
    constexpr uint32_t HALF_F = N_FULL_ROUNDS_TOTAL_T >> 1;
    const uint32_t lane = threadIdx.x;

    v = mdsExternalWarpReg<W>(v, lane, mask);

    for (uint32_t r = 0; r < HALF_F; ++r)
    {
        gl64_t xi = v + GPU_C_GL[r * W + lane];
        gl64_t x2 = xi * xi, x3 = xi * x2, x4 = x2 * x2;
        v = x3 * x4;
        v = mdsExternalWarpReg<W>(v, lane, mask);
    }

    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS_T; ++r)
    {
        if (lane == 0)
        {
            v = v + GPU_C_GL[HALF_F * W + r];
            pow7(v);
        }
        gl64_t sum = warp_allreduce_add<W>(v, mask);   // Σ over all lanes, to all
        v = v * GPU_D_GL[lane] + sum;
    }

    for (uint32_t r = 0; r < HALF_F; ++r)
    {
        gl64_t xi = v + GPU_C_GL[HALF_F * W + N_PARTIAL_ROUNDS_T + r * W + lane];
        gl64_t x2 = xi * xi, x3 = xi * x2, x4 = x2 * x2;
        v = x3 * x4;
        v = mdsExternalWarpReg<W>(v, lane, mask);
    }
    return v;
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void poseidon2PermuteSmem(const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL)
{

    mdsExternalSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    for (int r = 0; r < (N_FULL_ROUNDS_TOTAL_T>>1); r++)
    {
        sboxFullSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(&(GPU_C_GL[r * SPONGE_WIDTH_T]));
        mdsExternalSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    }

    for(int r = 0; r < N_PARTIAL_ROUNDS_T; r++)
    {
        scratchpad[threadIdx.x] = scratchpad[threadIdx.x] + GPU_C_GL[(N_FULL_ROUNDS_TOTAL_T>>1) * SPONGE_WIDTH_T + r];
        pow7(scratchpad[threadIdx.x]);
        gl64_t sum_;
        sum_ = gl64_t(uint64_t(0));
        sumAllSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(&sum_);
        partialRoundMulSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(GPU_D_GL, sum_);
    }

    for (int r = 0; r < (N_FULL_ROUNDS_TOTAL_T>>1); r++)
    {
        sboxFullSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(&(GPU_C_GL[(N_FULL_ROUNDS_TOTAL_T>>1) * SPONGE_WIDTH_T + N_PARTIAL_ROUNDS_T + r * SPONGE_WIDTH_T]));
        mdsExternalSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    }

}

// --- Sponge I/O ---

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void spongeLoad(const uint64_t *in, uint32_t initial_col, uint32_t ncols,
                                               uint32_t col_stride, size_t row_stride = 1)
{
    gl64_t r[RATE_T];

    const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    in += tid * col_stride + initial_col * row_stride;

#pragma unroll
    for (uint32_t i = 0; i < RATE_T; i++, in += row_stride)
        if (i < ncols){
            r[i] = __ldcv((uint64_t *)in);
        }

    __syncwarp();

    for (uint32_t i = 0; i < RATE_T; i++)
        scratchpad[i * blockDim.x + threadIdx.x] = r[i];

    __syncwarp();
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ void spongeLoadTiled(const uint64_t *in, uint64_t num_rows, uint64_t num_cols, uint32_t initial_col, uint32_t ncols, Layout layout)
{
    gl64_t r[RATE_T];

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll
    for (uint32_t i = 0; i < RATE_T; i++) {
        if (i < ncols){
            uint32_t col = initial_col + i;
            uint64_t idx = getBufferOffset(row, col, num_rows, num_cols, layout);
            r[i] = in[idx];
        }
    }

    __syncwarp();

    for (uint32_t i = 0; i < RATE_T; i++)
        scratchpad[i * blockDim.x + threadIdx.x] = r[i];

    __syncwarp();
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void spongeStore(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride)
{
    gl64_t r[CAPACITY_T];

    __syncwarp();

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY_T; i++)
        r[i] = scratchpad[i * blockDim.x + threadIdx.x];

    __syncwarp();

    const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    out += tid * col_stride;

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY_T; i++, out++)
        *(uint64_t *)out = r[i];
}

// --- Sponge absorb ---

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void spongeAbsorb(const uint64_t *__restrict__ in, uint32_t ncols)
{
    for (uint32_t col = 0;;)
    {
        uint32_t delta = min(ncols - col, RATE_T);
        spongeLoad<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(in, col, delta, ncols);
        if (delta < RATE_T)
        {
            for (uint32_t i = delta; i < RATE_T; i++)
            {
                scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));
            }
        }

        poseidon2PermuteSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();

        if ((col += RATE_T) >= ncols)
            break;

        gl64_t tmp[CAPACITY_T];

#pragma unroll
        for (uint32_t i = 0; i < CAPACITY_T; i++)
            tmp[i] = scratchpad[i * blockDim.x + threadIdx.x];
        __syncwarp();

#pragma unroll

        for (uint32_t i = 0; i < CAPACITY_T; i++)
            scratchpad[(i + RATE_T) * blockDim.x + threadIdx.x] = tmp[i];
        __syncwarp();
    }
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ __forceinline__ void spongeAbsorbTiled(const uint64_t *__restrict__ in, uint32_t num_cols, uint32_t num_rows, Layout layout)
{
    for (uint32_t col = 0;;)
    {
        uint32_t delta = min(num_cols - col, RATE_T);
        spongeLoadTiled<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(in, num_rows, num_cols, col, delta, layout);
        if (delta < RATE_T)
        {
            for (uint32_t i = delta; i < RATE_T; i++)
            {
                scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));
            }
        }
        poseidon2PermuteSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();

        if ((col += RATE_T) >= num_cols)
            break;

        gl64_t tmp[CAPACITY_T];

#pragma unroll
        for (uint32_t i = 0; i < CAPACITY_T; i++)
            tmp[i] = scratchpad[i * blockDim.x + threadIdx.x];
        __syncwarp();

#pragma unroll

        for (uint32_t i = 0; i < CAPACITY_T; i++)
            scratchpad[(i + RATE_T) * blockDim.x + threadIdx.x] = tmp[i];
        __syncwarp();
    }
}

// --- Kernel definitions ---

// permuteKernel / permuteTruncKernel: single-element permutation on contiguous state.
// MUST be launched with <<<1, 1>>> 

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void permuteKernel(uint64_t * output, const uint64_t * input){
    assert(blockDim.x == 1 && gridDim.x == 1);
    for (uint32_t i = 0; i < SPONGE_WIDTH_T; i++)
        scratchpad[i] = input[i];
    poseidon2PermuteSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    for (uint32_t i = 0; i < SPONGE_WIDTH_T; i++)
        output[i] = scratchpad[i];
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void permuteTruncKernel(uint64_t * output, const uint64_t * input){
    assert(blockDim.x == 1 && gridDim.x == 1);
    for (uint32_t i = 0; i < SPONGE_WIDTH_T; i++)
        scratchpad[i] = input[i];
    poseidon2PermuteSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    for (uint32_t i = 0; i < CAPACITY_T; i++)
        output[i] = scratchpad[i];
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void linearHashKernel(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t num_cols, uint32_t num_rows)
{
    if (num_cols == 0)
    {
        const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
        uint64_t *out = output + tid * CAPACITY_T;
#pragma unroll
        for (uint32_t i = 0; i < CAPACITY_T; i++)
            out[i] = 0;
        return;
    }

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY_T; i++)
        scratchpad[(i + RATE_T) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));

    spongeAbsorb<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(input, num_cols);
    spongeStore<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(output, CAPACITY_T, 1);
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void linearHashTiledKernel(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t num_cols, uint32_t num_rows, Layout layout)
{
    if (num_cols == 0)
    {
        const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
        uint64_t *out = output + tid * CAPACITY_T;
#pragma unroll
        for (uint32_t i = 0; i < CAPACITY_T; i++)
            out[i] = 0;
        return;
    }

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY_T; i++)
        scratchpad[(i + RATE_T) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0));

    spongeAbsorbTiled<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(input, num_cols, num_rows, layout);
    spongeStore<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(output, CAPACITY_T, 1);
}

#endif
