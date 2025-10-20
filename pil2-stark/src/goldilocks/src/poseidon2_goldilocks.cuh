#ifndef POSEIDON2_GOLDILOCKS_CUH
#define POSEIDON2_GOLDILOCKS_CUH

#include "gl64_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "poseidon2_goldilocks.hpp"
#include "data_layout.cuh"

__global__ void linear_hash_gpu_2_2_tree_2(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows);
__global__ void linear_hash_gpu_coalesced_2(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t size, uint32_t num_rows);
__global__ void linear_hash_gpu_coalesced_2_blocks(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t num_cols, uint32_t num_rows);
__device__ __forceinline__ void poseidon2_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride);
__device__ __forceinline__ void poseidon2_store(gl64_t *out, uint32_t col_stride, size_t row_stride);
__device__ __forceinline__ void poseidon2_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols);
__device__ __forceinline__ void poseidon2_hash_loop_blocks(const uint64_t *__restrict__ in, uint32_t num_cols, uint32_t num_rows);
__device__ __forceinline__ void poseidon2_hash();
__device__ __forceinline__ void pow7_2(gl64_t &x);
__device__ __forceinline__ void matmul_m4_(gl64_t *x);

__device__ __forceinline__ void pow7_2(gl64_t &x)
{
    gl64_t x2 = x * x;
    gl64_t x3 = x * x2;
    gl64_t x4 = x2 * x2;
    x = x3 * x4;
}

__device__ __forceinline__ void add_2(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[0] = x[0] + C[i];
    }
}

__device__ __forceinline__ void prod_2(gl64_t *x, const gl64_t alpha, const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = alpha * C[i];
    }
}

__device__ __forceinline__ void pow7add_2(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
    gl64_t x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        gl64_t xi = x[i] + C[i];
        x2[i] = xi * xi;
        x3[i] = xi * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
}

__device__ __forceinline__ void matmul_external_(gl64_t *x)
{
    matmul_m4_(&x[0]);
    matmul_m4_(&x[4]);
    matmul_m4_(&x[8]);

    gl64_t stored[4] = {
        x[0] + x[4] + x[8],
        x[1] + x[5] + x[9],
        x[2] + x[6] + x[10],
        x[3] + x[7] + x[11],
    };
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i] + stored[i % 4];
    }
}

__device__ __forceinline__ void matmul_m4_(gl64_t *x)
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

__device__ __forceinline__ void prodadd_(gl64_t *x, const gl64_t D[SPONGE_WIDTH], const gl64_t &sum)
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i] * D[i] + sum;
    }
}
__device__ __forceinline__ void hash_full_result_seq_2(gl64_t *state, const gl64_t *input, const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL)
{
    mymemcpy((uint64_t *)state, (uint64_t *)input, SPONGE_WIDTH);
    
    matmul_external_(state);

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_2(state, &(GPU_C_GL[r * SPONGE_WIDTH]));
        matmul_external_(state);
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        state[0] = state[0] + GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7_2(state[0]);
        gl64_t sum_;
        sum_ = gl64_t(uint64_t(0));
        add_2(&sum_, state);
        prodadd_(state, GPU_D_GL, sum_);
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_2(state, &(GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_(state);
    }
}


class Poseidon2GoldilocksGPU : public Poseidon2Goldilocks {
public:
    using Poseidon2Goldilocks::Poseidon2Goldilocks;

   
    void static merkletree_cuda_coalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads = 0, uint64_t dim = 1);

    void static merkletree_cuda_coalesced_blocks(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads = 0, uint64_t dim = 1);
};


#endif