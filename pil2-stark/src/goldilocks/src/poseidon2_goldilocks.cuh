#ifndef POSEIDON2_GOLDILOCKS_CUH
#define POSEIDON2_GOLDILOCKS_CUH

#include "gl64_t.cuh"
#include "gl64_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "poseidon2_goldilocks.hpp"
#include "data_layout.cuh"

extern __shared__ gl64_t scratchpad[];

// Constants defined in "poseidon2_goldilocks_constants.hpp"
__device__ __constant__ uint64_t GPU_C[SPONGE_WIDTH * N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS];
__device__ __constant__ uint64_t GPU_D[SPONGE_WIDTH];

__global__ void linear_hash_gpu_coalesced_2(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t size, uint32_t num_rows);
__global__ void linear_hash_gpu_coalesced_2_blocks(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t num_cols, uint32_t num_rows);
__device__ __forceinline__ void poseidon2_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride);
__device__ __forceinline__ void poseidon2_store(gl64_t *out, uint32_t col_stride, size_t row_stride);
__device__ __forceinline__ void poseidon2_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols);
__device__ __forceinline__ void poseidon2_hash_loop_blocks(const uint64_t *__restrict__ in, uint32_t num_cols, uint32_t num_rows);
__device__ __forceinline__ void poseidon2_hash();
__device__ __forceinline__ void poseidon2_hash_with_constants(const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL);
__device__ __forceinline__ void poseidon2_hash_shared(gl64_t *out, const gl64_t *in, const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL);
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
#pragma unroll
    for(int i=0; i<SPONGE_WIDTH; i+=4)
        matmul_m4_(&x[i]);

    if( SPONGE_WIDTH > 4 ){
        gl64_t stored[4]={gl64_t(uint64_t(0)), gl64_t(uint64_t(0)), gl64_t(uint64_t(0)), gl64_t(uint64_t(0))};
        for(int i=0; i<SPONGE_WIDTH; i+=4){
            stored[0] = stored[0] + x[i + 0];
            stored[1] = stored[1] + x[i + 1];
            stored[2] = stored[2] + x[i + 2];
            stored[3] = stored[3] + x[i + 3];
        }
    #pragma unroll
        for (int i = 0; i < SPONGE_WIDTH; ++i)
        {
            x[i] = x[i] + stored[i & 3];
        }
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

__device__ __forceinline__ void add_state_2(gl64_t *x)
{
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
       x[0]= x[0] + scratchpad[i * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void matmul_m4_state_(uint32_t offset)
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

__device__ __forceinline__ void matmul_external_state_()
{

#pragma unroll
    for(int i=0; i<SPONGE_WIDTH; i+=4){
        matmul_m4_state_(i);
    }
    gl64_t stored[4]={gl64_t(uint64_t(0)), gl64_t(uint64_t(0)), gl64_t(uint64_t(0)), gl64_t(uint64_t(0))};
    for(int i=0; i<SPONGE_WIDTH; i+=4){
        stored[0] = stored[0] + scratchpad[(i + 0) * blockDim.x + threadIdx.x];
        stored[1] = stored[1] + scratchpad[(i + 1) * blockDim.x + threadIdx.x];
        stored[2] = stored[2] + scratchpad[(i + 2) * blockDim.x + threadIdx.x];
        stored[3] = stored[3] + scratchpad[(i + 3) * blockDim.x + threadIdx.x];
    }
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        scratchpad[i * blockDim.x + threadIdx.x] = scratchpad[i * blockDim.x + threadIdx.x] + stored[i & 3];
    }
}

__device__ __forceinline__ void pow7add_state_(const gl64_t C[SPONGE_WIDTH])
{
    gl64_t x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        gl64_t xi = scratchpad[i * blockDim.x + threadIdx.x] + C[i];
        x2[i] = xi * xi;
        x3[i] = xi * x2[i];
        x4[i] = x2[i] * x2[i];
        scratchpad[i * blockDim.x + threadIdx.x] = x3[i] * x4[i];
    }
}


__device__ __forceinline__ void prodadd_state_(const gl64_t D[SPONGE_WIDTH], const gl64_t &sum)
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        scratchpad[i * blockDim.x + threadIdx.x] = scratchpad[i * blockDim.x + threadIdx.x] * D[i] + sum;
    }
}

__device__ __forceinline__ void poseidon2_hash()
{
    const gl64_t *GPU_C_GL = (gl64_t *)GPU_C;
    const gl64_t *GPU_D_GL = (gl64_t *)GPU_D;

    matmul_external_state_();
    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_state_(&(GPU_C_GL[r * SPONGE_WIDTH]));
        matmul_external_state_();
    }

    for(int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        scratchpad[threadIdx.x] = scratchpad[threadIdx.x] + GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7_2(scratchpad[threadIdx.x]);
        gl64_t sum_;
        sum_ = gl64_t(uint64_t(0));
        add_state_2(&sum_);
        prodadd_state_(GPU_D_GL, sum_);
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_state_(&(GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_state_();
    }
}

__device__ __forceinline__ void poseidon2_hash_with_constants(const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL)
{
    matmul_external_state_();
    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_state_(&(GPU_C_GL[r * SPONGE_WIDTH]));
        matmul_external_state_();
    }

    for(int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        scratchpad[threadIdx.x] = scratchpad[threadIdx.x] + GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7_2(scratchpad[threadIdx.x]);
        gl64_t sum_;
        sum_ = gl64_t(uint64_t(0));
        add_state_2(&sum_);
        prodadd_state_(GPU_D_GL, sum_);
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_state_(&(GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_state_();
    }
}

__device__ __forceinline__ void poseidon2_hash_shared(gl64_t *out, const gl64_t *in, const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL)
{
    for (int i = 0; i < SPONGE_WIDTH; i++) {
        scratchpad[i * blockDim.x + threadIdx.x] = in[i];
    }

    poseidon2_hash_with_constants(GPU_C_GL, GPU_D_GL);
    
    for (int i = 0; i < SPONGE_WIDTH; i++) {
        out[i] = scratchpad[i * blockDim.x + threadIdx.x];
    }
}


class Poseidon2GoldilocksGPU : public Poseidon2Goldilocks {
public:
    using Poseidon2Goldilocks::Poseidon2Goldilocks;

   
    void static merkletree_cuda_coalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads = 0, uint64_t dim = 1);

    void static merkletree_cuda_coalesced_blocks(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads = 0, uint64_t dim = 1);

    void static init_gpu_const_2(uint32_t* gpu_ids, uint32_t num_gpu_ids);
};


#endif