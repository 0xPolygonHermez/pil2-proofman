#include "gl64_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>

#include "poseidon2_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"
#include "poseidon2_goldilocks.cuh"

// #ifdef GPU_TIMING
#include "timer_gl.hpp"
// #endif

typedef uint32_t u32;
typedef uint64_t u64;

// CUDA Threads per Block
#define TPB 128

#define MAX_WIDTH 12

extern __shared__ gl64_t scratchpad[];

__device__ __forceinline__ void matmul_m4_state_(uint32_t offset);


// Constants defined in "poseidon2_goldilocks_constants.hpp"
__device__ __constant__ uint64_t GPU_C[118];
__device__ __constant__ uint64_t GPU_D[12];

void init_gpu_const_2(uint32_t* gpu_ids, uint32_t num_gpu_ids)
{
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    static int initialized = 0;
    if (initialized == 0)
    {
        for(int i = 0; i < num_gpu_ids; i++)
        {
           CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
           CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C, Poseidon2GoldilocksConstants::C, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
           CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D, Poseidon2GoldilocksConstants::D, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                
        }
        initialized = 1;        
    }
    cudaSetDevice(deviceId);

}




/* --- integration --- */

__device__ void linear_hash_one_2(gl64_t *output, gl64_t *input, uint32_t size, int tid)
{
    u32 remaining = size;
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_D_SM[12];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_D_SM, GPU_D, 12);
    }
    __syncthreads();

    gl64_t state[SPONGE_WIDTH];

    if (size <= CAPACITY)
    {
        mymemcpy((uint64_t *)output, (uint64_t *)input, size);
        mymemset((uint64_t *)&output[size], 0, (CAPACITY - size));
        return; // no need to hash
    }
    while (remaining)
    {
        if (remaining == size)
        {
            mymemset((uint64_t *)(state + RATE), 0, CAPACITY);
        }
        else
        {
            mymemcpy((uint64_t *)(state + RATE), (uint64_t *)state, CAPACITY);
        }

        u32 n = (remaining < RATE) ? remaining : RATE;
        mymemset((uint64_t *)&state[n], 0, (RATE - n));
        mymemcpy((uint64_t *)state, (uint64_t *)(input + (size - remaining)), n);
        hash_full_result_seq_2(state, state, GPU_C_SM, GPU_D_SM);
        remaining -= n;
    }
    mymemcpy((uint64_t *)output, (uint64_t *)state, CAPACITY);
}

__device__ void linear_partial_hash_one_2(gl64_t *input, uint32_t size, gl64_t *state, int tid)
{
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_D_SM[12];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_D_SM, GPU_D, 12);
    }
    __syncthreads();

    u32 remaining = size;

    while (remaining)
    {
        mymemcpy((uint64_t *)(state + RATE), (uint64_t *)state, CAPACITY);
        u32 n = (remaining < RATE) ? remaining : RATE;
        mymemset((uint64_t *)&state[n], 0, (RATE - n));
        mymemcpy((uint64_t *)state, (uint64_t *)(input + (size - remaining)), n);
        hash_full_result_seq_2(state, state, GPU_C_SM, GPU_D_SM);
        remaining -= n;
    }
}

__global__ void linear_hash_gpu_2_2(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *inp = (gl64_t *)(input + tid * size);
    gl64_t *out = (gl64_t *)(output + tid * CAPACITY);
    linear_hash_one_2(out, inp, size, threadIdx.x);
}

__global__ void linear_partial_init_hash_gpu_2_2(uint64_t *gstate, int32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *state = (gl64_t *)(gstate + tid * SPONGE_WIDTH);
    memset(state, 0, SPONGE_WIDTH * sizeof(gl64_t));
}

__global__ void linear_partial_hash_gpu_2(uint64_t *input, uint32_t num_cols, uint32_t num_rows, uint64_t *gstate, uint32_t hash_per_thread = 1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    for (uint32_t i = 0; i < hash_per_thread; i++)
    {
        gl64_t *inp = (gl64_t *)(input + (tid * hash_per_thread + i) * num_cols);
        gl64_t *state = (gl64_t *)(gstate + (tid * hash_per_thread + i) * SPONGE_WIDTH);
        linear_partial_hash_one_2(inp, num_cols, state, threadIdx.x);
    }
}

__device__ void hash_one_2(gl64_t *state, gl64_t *const input, int tid)
{
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_D_SM[12];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_D_SM, GPU_D, 12);
    }
    __syncthreads();

    gl64_t aux[SPONGE_WIDTH];
    hash_full_result_seq_2(aux, input, GPU_C_SM, GPU_D_SM);
    mymemcpy((uint64_t *)state, (uint64_t *)aux, CAPACITY);
}

__global__ void hash_gpu_2(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input, 0, SPONGE_WIDTH);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor[nextIndex + tid * RATE], RATE);
    hash_one_2((gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY]), pol_input, threadIdx.x);
}

__global__ void hash_gpu_3(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input, 0, SPONGE_WIDTH);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor[nextIndex + tid * SPONGE_WIDTH], SPONGE_WIDTH);
    hash_one_2((gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY]), pol_input, threadIdx.x);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////

void Poseidon2GoldilocksGPU::merkletree_cuda_coalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    // init_gpu_const_2(); // this needs to be done only once !!
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;


    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu_coalesced_2<<<actual_blks, actual_tpb, actual_tpb * 12 * 8, stream>>>(d_tree, d_input, num_cols * dim, num_rows); // rick: el 12 aqeust harcoded no please!!
    CHECKCUDAERR(cudaGetLastError());

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0){

            //std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
            CHECKCUDAERR(cudaMemsetAsync((uint64_t *)(d_tree + nextIndex + pending * CAPACITY), 0, extraZeros * CAPACITY * sizeof(uint64_t), stream));
        }
        if (nextN < TPB)
        {
            actual_tpb = nextN;
            actual_blks = 1;
        }
        else
        {
            actual_tpb = TPB;
            actual_blks = nextN / TPB + 1;
        }
        hash_gpu_3<<<actual_blks, actual_tpb, 0, stream>>>(nextN, nextIndex, pending + extraZeros, d_tree);       
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

void Poseidon2GoldilocksGPU::merkletree_cuda_coalesced_blocks(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    // init_gpu_const_2(); // this needs to be done only once !!
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;


    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu_coalesced_2_blocks<<<actual_blks, actual_tpb, actual_tpb * 12 * 8, stream>>>(d_tree, d_input, num_cols * dim, num_rows); // rick: el 12 aqeust harcoded no please!!
    CHECKCUDAERR(cudaGetLastError());

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0){

            //std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
            CHECKCUDAERR(cudaMemsetAsync((uint64_t *)(d_tree + nextIndex + pending * CAPACITY), 0, extraZeros * CAPACITY * sizeof(uint64_t), stream));
        }
        if (nextN < TPB)
        {
            actual_tpb = nextN;
            actual_blks = 1;
        }
        else
        {
            actual_tpb = TPB;
            actual_blks = nextN / TPB + 1;
        }
        hash_gpu_3<<<actual_blks, actual_tpb, 0, stream>>>(nextN, nextIndex, pending + extraZeros, d_tree);       
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

__global__ void hash_gpu_2(uint32_t nextN, uint64_t *cursor_in, uint64_t *cursor_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input + RATE, 0, CAPACITY);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor_in[tid * RATE], RATE);
    hash_one_2((gl64_t *)(&cursor_out[tid * CAPACITY]), pol_input, threadIdx.x);
}

__global__ void hash_gpu_3(uint32_t nextN, uint64_t *cursor_in, uint64_t *cursor_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    //mymemset((uint64_t *)pol_input + RATE, 0, CAPACITY);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor_in[tid * SPONGE_WIDTH], SPONGE_WIDTH);
    hash_one_2((gl64_t *)(&cursor_out[tid * CAPACITY]), pol_input, threadIdx.x);
}

__device__ __noinline__ void add_state_2(gl64_t *x)
{
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
       x[0]= x[0] + scratchpad[i * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void poseidon2_load(const uint64_t *in, uint32_t initial_col, uint32_t ncols,
                                               uint32_t col_stride, size_t row_stride = 1)
{
    gl64_t r[RATE];

    const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    in += tid * col_stride + initial_col * row_stride;

#pragma unroll
    for (uint32_t i = 0; i < RATE; i++, in += row_stride)
        if (i < ncols){
            r[i] = __ldcv((uint64_t *)in);
        }

    __syncwarp();

    for (uint32_t i = 0; i < RATE; i++)
        scratchpad[i * blockDim.x + threadIdx.x] = r[i];

    __syncwarp();
}

__device__ void poseidon2_load_blocks(const uint64_t *in, uint64_t num_rows, uint64_t num_cols, uint32_t initial_col, uint32_t ncols)
{
    gl64_t r[RATE];

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll
    for (uint32_t i = 0; i < RATE; i++) {
        if (i < ncols){
            uint32_t col = initial_col + i;
            uint64_t idx = getBufferOffset(row, col, num_rows, num_cols);
            r[i] = in[idx];
        }
    }

    __syncwarp();

    for (uint32_t i = 0; i < RATE; i++)
        scratchpad[i * blockDim.x + threadIdx.x] = r[i];

    __syncwarp();
}

__device__ __forceinline__ void poseidon2_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride = 1)
{
    gl64_t r[CAPACITY];

    __syncwarp();

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        r[i] = scratchpad[i * blockDim.x + threadIdx.x];

    __syncwarp();

    const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    out += tid * col_stride;

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++, out++)
        *(uint64_t *)out = r[i];
}

__device__ __forceinline__ void poseidon2_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols)
{
    if (ncols <= CAPACITY)
    {
        poseidon2_load(in, 0, ncols, ncols);
        for (uint32_t i = ncols; i < CAPACITY; i++)
        {
            scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 
        }
    }
    else
    {
        for (uint32_t col = 0;;)
        {
            uint32_t delta = min(ncols - col, RATE);
            poseidon2_load(in, col, delta, ncols);
            if (delta < RATE)
            {
                for (uint32_t i = delta; i < RATE; i++)
                {
                    scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 
                }
            }
            /*if(blockIdx.x == 0 && threadIdx.x == 0){
                for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
                    printf("tmp abans[%d] = %lu col=%d ncols=%d\n", i, scratchpad[i * blockDim.x + threadIdx.x][0], col, ncols);
            }*/
            poseidon2_hash();
            
            /*if(blockIdx.x == 0 && threadIdx.x == 0){
                for (uint32_t i = 0; i < CAPACITY; i++)
                    printf("tmp[%d] = %lu col=%d ncols=%d\n", i, scratchpad[i * blockDim.x + threadIdx.x][0], col, ncols);
            }*/

            if ((col += RATE) >= ncols)
                break;

            gl64_t tmp[CAPACITY];

#pragma unroll
            for (uint32_t i = 0; i < CAPACITY; i++)
                tmp[i] = scratchpad[i * blockDim.x + threadIdx.x];
            __syncwarp();

#pragma unroll

            for (uint32_t i = 0; i < CAPACITY; i++)
                scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = tmp[i];
            __syncwarp();
        }
    }
}

__device__ __forceinline__ void poseidon2_hash_loop_blocks(const uint64_t *__restrict__ in, uint32_t num_cols, uint32_t num_rows)
{
    if (num_cols <= CAPACITY)
    {
        poseidon2_load_blocks(in, num_rows, num_cols, 0, num_cols);
        for (uint32_t i = num_cols; i < CAPACITY; i++)
        {
            scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 
        }
    }
    else
    {
        for (uint32_t col = 0;;)
        {
            uint32_t delta = min(num_cols - col, RATE);
            poseidon2_load_blocks(in, num_rows, num_cols, col, delta);
            if (delta < RATE)
            {
                for (uint32_t i = delta; i < RATE; i++)
                {
                    scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 
                }
            }
            /*if(blockIdx.x == 0 && threadIdx.x == 0){
                for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
                    printf("tmp abans[%d] = %lu col=%d ncols=%d\n", i, scratchpad[i * blockDim.x + threadIdx.x][0], col, ncols);
            }*/
            poseidon2_hash();
            
            /*if(blockIdx.x == 0 && threadIdx.x == 0){
                for (uint32_t i = 0; i < CAPACITY; i++)
                    printf("tmp[%d] = %lu col=%d ncols=%d\n", i, scratchpad[i * blockDim.x + threadIdx.x][0], col, ncols);
            }*/

            if ((col += RATE) >= num_cols)
                break;

            gl64_t tmp[CAPACITY];

#pragma unroll
            for (uint32_t i = 0; i < CAPACITY; i++)
                tmp[i] = scratchpad[i * blockDim.x + threadIdx.x];
            __syncwarp();

#pragma unroll

            for (uint32_t i = 0; i < CAPACITY; i++)
                scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = tmp[i];
            __syncwarp();
        }
    }
}

__device__ __forceinline__ void matmul_external_state_()
{
    matmul_m4_state_(0);
    matmul_m4_state_(4);
    matmul_m4_state_(8);

    gl64_t stored[4] = {
       scratchpad[0 * blockDim.x + threadIdx.x] + scratchpad[4 * blockDim.x + threadIdx.x] + scratchpad[8 * blockDim.x + threadIdx.x],
       scratchpad[1 * blockDim.x + threadIdx.x] + scratchpad[5 * blockDim.x + threadIdx.x] + scratchpad[9 * blockDim.x + threadIdx.x],
       scratchpad[2 * blockDim.x + threadIdx.x] + scratchpad[6 * blockDim.x + threadIdx.x] + scratchpad[10 * blockDim.x + threadIdx.x],
       scratchpad[3 * blockDim.x + threadIdx.x] + scratchpad[7 * blockDim.x + threadIdx.x] + scratchpad[11 * blockDim.x + threadIdx.x],
    };
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        scratchpad[i * blockDim.x + threadIdx.x] = scratchpad[i * blockDim.x + threadIdx.x] + stored[i % 4];
    }
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

__global__ void linear_hash_gpu_coalesced_2(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t num_cols, uint32_t num_rows)
{
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 

    poseidon2_hash_loop(input, num_cols);
    poseidon2_store(output, CAPACITY);
}

__global__ void linear_hash_gpu_coalesced_2_blocks(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t num_cols, uint32_t num_rows)
{
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 

    poseidon2_hash_loop_blocks(input, num_cols, num_rows);
    poseidon2_store(output, CAPACITY);
}