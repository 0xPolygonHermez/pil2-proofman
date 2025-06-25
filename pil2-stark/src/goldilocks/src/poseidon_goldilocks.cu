#include "gl64_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>

#include "poseidon_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"

// #ifdef GPU_TIMING
#include "timer_gl.hpp"
// #endif

typedef uint32_t u32;
typedef uint64_t u64;

// CUDA Threads per Block
#define TPB 128

#define MAX_WIDTH 12

extern __shared__ gl64_t scratchpad[];

/* new functions */
__global__ void linear_hash_gpu_tree(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows);
__global__ void linear_hash_gpu_coalesced(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t size, uint32_t num_rows);
__device__ __forceinline__ void poseidon_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride);
__device__ __forceinline__ void poseidon_store(gl64_t *out, uint32_t col_stride, size_t row_stride);
__device__ __forceinline__ void poseidon_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols);
__device__ __forceinline__ void poseidon_hash();
__device__ __noinline__ void pow7(gl64_t &x);

/* --- Based on seq code --- */

__device__ __forceinline__ void pow7_(gl64_t *x)
{
    gl64_t x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
}

__device__ __forceinline__ void add_(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i] + C[i];
    }
}

__device__ __forceinline__ void prod_(gl64_t *x, const gl64_t alpha, const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = alpha * C[i];
    }
}

__device__ __forceinline__ void pow7add_(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
    gl64_t x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
        x[i] = x[i] + C[i];
    }
}

__device__ __forceinline__ gl64_t dot_(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
    gl64_t s0 = x[0] * C[0];
#pragma unroll
    for (int i = 1; i < SPONGE_WIDTH; i++)
    {
        s0 = s0 + x[i] * C[i];
    }
    return s0;
}

__device__ __forceinline__ void mvp_(gl64_t *state, const gl64_t *__restrict__ mat)
{
    gl64_t old_state[SPONGE_WIDTH];
    mymemcpy((uint64_t *)old_state, (uint64_t *)state, SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = mat[i] * old_state[0];
        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (mat[12 * j + i] * old_state[j]);
        }
    }
}

// Constants defined in "poseidon_goldilocks_constants.hpp"
__device__ __constant__ uint64_t GPU_C[118];
__device__ __constant__ uint64_t GPU_S[507];
__device__ __constant__ uint64_t GPU_M[144];
__device__ __constant__ uint64_t GPU_P[144];

void init_gpu_const(int nDevices = 0)
{
    static int initialized = 0;

    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    if (initialized == 0)
    {
        initialized = 1;
        if (nDevices == 0)
        {
            CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
        }
        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_M, PoseidonGoldilocksConstants::M, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_P, PoseidonGoldilocksConstants::P, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C, PoseidonGoldilocksConstants::C, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_S, PoseidonGoldilocksConstants::S, 507 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        }
    }
    cudaSetDevice(deviceId);

}

__device__ void hash_full_result_seq(gl64_t *state, const gl64_t *input, const gl64_t *GPU_C_GL, const gl64_t *GPU_S_GL, const gl64_t *GPU_M_GL, const gl64_t *GPU_P_GL)
{
    mymemcpy((uint64_t *)state, (uint64_t *)input, SPONGE_WIDTH);

    add_(state, GPU_C_GL);
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(GPU_C_GL[(r + 1) * SPONGE_WIDTH]));
        mvp_(state, GPU_M_GL);
    }

    pow7add_(state, &(GPU_C_GL[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    mvp_(state, GPU_P_GL);

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(state[0]);
        state[0] = state[0] + GPU_C_GL[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        gl64_t s0 = dot_(state, &(GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r]));
        gl64_t W_[SPONGE_WIDTH];
        prod_(W_, state[0], &(GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        add_(state, W_);
        state[0] = s0;
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(GPU_C_GL[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        mvp_(state, GPU_M_GL);
    }
    pow7_(&(state[0]));
    mvp_(state, GPU_M_GL);
}

/* --- integration --- */

__device__ void linear_hash_one(gl64_t *output, gl64_t *input, uint32_t size, int tid)
{
    u32 remaining = size;
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_S_SM[507];
    __shared__ gl64_t GPU_M_SM[144];
    __shared__ gl64_t GPU_P_SM[144];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_S_SM, GPU_S, 507);
        mymemcpy((uint64_t *)GPU_M_SM, GPU_M, 144);
        mymemcpy((uint64_t *)GPU_P_SM, GPU_P, 144);
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
        hash_full_result_seq(state, state, GPU_C_SM, GPU_S_SM, GPU_M_SM, GPU_P_SM);
        remaining -= n;
    }
    mymemcpy((uint64_t *)output, (uint64_t *)state, CAPACITY);
}

__device__ void linear_partial_hash_one(gl64_t *input, uint32_t size, gl64_t *state, int tid)
{
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_S_SM[507];
    __shared__ gl64_t GPU_M_SM[144];
    __shared__ gl64_t GPU_P_SM[144];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_S_SM, GPU_S, 507);
        mymemcpy((uint64_t *)GPU_M_SM, GPU_M, 144);
        mymemcpy((uint64_t *)GPU_P_SM, GPU_P, 144);
    }
    __syncthreads();

    u32 remaining = size;

    while (remaining)
    {
        mymemcpy((uint64_t *)(state + RATE), (uint64_t *)state, CAPACITY);
        u32 n = (remaining < RATE) ? remaining : RATE;
        mymemset((uint64_t *)&state[n], 0, (RATE - n));
        mymemcpy((uint64_t *)state, (uint64_t *)(input + (size - remaining)), n);
        hash_full_result_seq(state, state, GPU_C_SM, GPU_S_SM, GPU_M_SM, GPU_P_SM);
        remaining -= n;
    }
}

__global__ void linear_hash_gpu(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *inp = (gl64_t *)(input + tid * size);
    gl64_t *out = (gl64_t *)(output + tid * CAPACITY);
    linear_hash_one(out, inp, size, threadIdx.x);
}

__global__ void linear_partial_init_hash_gpu(uint64_t *gstate, int32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *state = (gl64_t *)(gstate + tid * SPONGE_WIDTH);
    memset(state, 0, SPONGE_WIDTH * sizeof(gl64_t));
}

__device__ void hash_one(gl64_t *state, gl64_t *const input, int tid)
{
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_S_SM[507];
    __shared__ gl64_t GPU_M_SM[144];
    __shared__ gl64_t GPU_P_SM[144];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_S_SM, GPU_S, 507);
        mymemcpy((uint64_t *)GPU_M_SM, GPU_M, 144);
        mymemcpy((uint64_t *)GPU_P_SM, GPU_P, 144);
    }
    __syncthreads();

    gl64_t aux[SPONGE_WIDTH];
    hash_full_result_seq(aux, input, GPU_C_SM, GPU_S_SM, GPU_M_SM, GPU_P_SM);
    mymemcpy((uint64_t *)state, (uint64_t *)aux, CAPACITY);
}

__global__ void hash_gpu(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input, 0, SPONGE_WIDTH);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor[nextIndex + tid * RATE], RATE);
    hash_one((gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY]), pol_input, threadIdx.x);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////

void PoseidonGoldilocks::merkletree_cuda_coalesced(uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    init_gpu_const(); // this needs to be done only once !!
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows); // includes CAPACITY
    CHECKCUDAERR(cudaMalloc(d_tree, numElementsTree * sizeof(uint64_t)));

    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu_coalesced<<<actual_blks, actual_tpb, actual_tpb * 12 * 8>>>(*d_tree, d_input, num_cols * dim, num_rows); // rick: el 12 aqeust harcoded no please!!

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;
    while (pending > 1)
    {
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
        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, *d_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
}

__global__ void hash_gpu(uint32_t nextN, uint64_t *cursor_in, uint64_t *cursor_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input + RATE, 0, CAPACITY);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor_in[tid * RATE], RATE);
    hash_one((gl64_t *)(&cursor_out[tid * CAPACITY]), pol_input, threadIdx.x);
}


__device__ __noinline__ void pow7(gl64_t &x)
{
    gl64_t x2 = x * x;
    gl64_t x3 = x * x2;
    gl64_t x4 = x2 * x2;
    x = x3 * x4;
}

__device__ __noinline__ static void pow7_state_()
{
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
        pow7(scratchpad[i * blockDim.x + threadIdx.x]);
}

__device__ __forceinline__ void mvp_state_(const gl64_t *mat)
{
    gl64_t state[SPONGE_WIDTH];
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
        state[i] = scratchpad[i * blockDim.x + threadIdx.x];

#pragma unroll 1
    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        scratchpad[i * blockDim.x + threadIdx.x] = mat[i] * state[0];
        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            scratchpad[i * blockDim.x + threadIdx.x] += (mat[12 * j + i] * state[j]); // rick: aquest access no mola gents
        }
    }
}

__device__ __noinline__ void add_state_(const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
        scratchpad[i * blockDim.x + threadIdx.x] += C[i];
}

__device__ __forceinline__ void poseidon_load(const uint64_t *in, uint32_t col, uint32_t ncols,
                                              uint32_t col_stride, size_t row_stride = 1)
{
    gl64_t r[RATE];

    const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    in += tid * col_stride + col * row_stride;

#pragma unroll
    for (uint32_t i = 0; i < RATE; i++, in += row_stride)
        if (i < ncols)
            r[i] = __ldcv((uint64_t *)in);

    __syncwarp();

    for (uint32_t i = 0; i < RATE; i++)
        scratchpad[i * blockDim.x + threadIdx.x] = r[i];

    __syncwarp();
}

__device__ __forceinline__ void poseidon_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride = 1)
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
        *(uint64_t *)out = r[i][0];
}

__device__ __forceinline__ void poseidon_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols)
{
    if (ncols <= CAPACITY)
    {
        poseidon_load(in, 0, ncols, ncols);
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
            poseidon_load(in, col, delta, ncols);
            if (delta < RATE)
            {
                for (uint32_t i = delta; i < RATE; i++)
                {
                    scratchpad[i * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 
                }
            }
            poseidon_hash();
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

__device__ __forceinline__ void poseidon_hash()
{
    const gl64_t *GPU_C_GL = (gl64_t *)GPU_C;
    const gl64_t *GPU_M_GL = (gl64_t *)GPU_M;
    const gl64_t *GPU_S_GL = (gl64_t *)GPU_S;
    const gl64_t *GPU_P_GL = (gl64_t *)GPU_P;

    add_state_(GPU_C_GL);
#pragma unroll 1
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_state_();
        add_state_(&(GPU_C_GL[(r + 1) * SPONGE_WIDTH]));
        mvp_state_(GPU_M_GL);
    }
    pow7_state_();
    add_state_(&(GPU_C_GL[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    mvp_state_(GPU_P_GL);

#pragma unroll 1
    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        // rick: millor tot aixo dins una funcio
        pow7(scratchpad[threadIdx.x]);
        gl64_t p0 = scratchpad[threadIdx.x] + GPU_C_GL[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        gl64_t s0 = p0 * GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r];

        for (uint32_t j = 1; j < SPONGE_WIDTH; j++)
        {
            s0 += scratchpad[j * blockDim.x + threadIdx.x] * GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r + j];
            scratchpad[j * blockDim.x + threadIdx.x] += p0 * GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + j];
        }
        scratchpad[threadIdx.x] = s0;
    }
#pragma unroll 1
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_state_();
        add_state_(&(GPU_C_GL[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        mvp_state_(GPU_M_GL);
    }
    pow7_state_();
    mvp_state_(GPU_M_GL);
}

__global__ void linear_hash_gpu_coalesced(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t size, uint32_t num_rows)
{
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 

    poseidon_hash_loop(input, size);
    poseidon_store(output, CAPACITY);
}

// funciont not tested
__device__ void generate_tree()
{
    uint32_t numThreads = blockDim.x;
    uint32_t offset_ant = 0;
    uint32_t offset = 0;
    while (numThreads > 1)
    {
        numThreads >>= 1;
        if (threadIdx.x >= offset && threadIdx.x < offset + numThreads)
        {
            uint32_t thread_relative = threadIdx.x - offset;
            uint32_t pos_previous = offset_ant + thread_relative << 1;

#pragma unroll
            for (uint32_t i = 0; i < CAPACITY; i++)
            {
                uint32_t idx = i * blockDim.x + threadIdx.x;
                uint32_t prev_idx = i * blockDim.x + pos_previous;

                gl64_t val1 = scratchpad[prev_idx];
                gl64_t val2 = scratchpad[prev_idx + 1];

                scratchpad[idx] = val1;
                scratchpad[(i + CAPACITY) * blockDim.x + threadIdx.x] = val2;
            }
            poseidon_hash();
        }
        __syncthreads();

        offset_ant = offset;
        offset += numThreads;
    }
}

// function not tested
__device__ void generate_1level_tree()
{

    if (threadIdx.x < blockDim.x >> 1)
    {
#pragma unroll
        for (uint32_t i = 0; i < CAPACITY; i++)
        {
            uint32_t idx = i * blockDim.x + threadIdx.x;
            uint32_t idx2 = idx >> 1;

            gl64_t val1 = scratchpad[idx2];
            gl64_t val2 = scratchpad[idx2 + 1];

            scratchpad[idx] = val1;
            scratchpad[(i + CAPACITY) * blockDim.x + threadIdx.x] = val2;
        }
        poseidon_hash();
    }
}

// function not tested
// rick: check limits on load and store
__global__ void linear_hash_gpu_tree(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows)
{
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 

    poseidon_hash_loop(input, size);
    poseidon_store(output, CAPACITY);
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = gl64_t(uint64_t(0)); 
    generate_1level_tree();
}