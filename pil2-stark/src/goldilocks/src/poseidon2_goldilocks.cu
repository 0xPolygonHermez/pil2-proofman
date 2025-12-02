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

/* --- integration --- */
__device__ void hash_one_2(gl64_t *state, gl64_t *const input, int tid)
{
    __shared__ gl64_t GPU_C_SM[SPONGE_WIDTH * N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS];
    __shared__ gl64_t GPU_D_SM[SPONGE_WIDTH];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, SPONGE_WIDTH * N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS); //rick_pos: why mymemcpy?
        mymemcpy((uint64_t *)GPU_D_SM, GPU_D, SPONGE_WIDTH);
    }
    __syncthreads();

    gl64_t aux[SPONGE_WIDTH];
    hash_full_result_seq_2(aux, input, GPU_C_SM, GPU_D_SM);
    mymemcpy((uint64_t *)state, (uint64_t *)aux, CAPACITY);
}


__global__ void hash_gpu_3(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input, 0, SPONGE_WIDTH); //rick_pos: ineficient
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor[nextIndex + tid * SPONGE_WIDTH], SPONGE_WIDTH);
    hash_one_2((gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY]), pol_input, threadIdx.x);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////
void Poseidon2GoldilocksGPU::init_gpu_const_2(uint32_t* gpu_ids, uint32_t num_gpu_ids)
{
    static_assert(SPONGE_WIDTH == 12 || SPONGE_WIDTH==16, "Error: Unsupported SPONGE_WIDTH.");
    
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    static int initialized = 0;
    if (initialized == 0)
    {
        for(int i = 0; i < num_gpu_ids; i++)
        {
            if( SPONGE_WIDTH == 12){
                CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C, Poseidon2GoldilocksConstants::C12, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D, Poseidon2GoldilocksConstants::D12, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            }else if( SPONGE_WIDTH == 16 ){
                CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C, Poseidon2GoldilocksConstants::C16, 150 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D, Poseidon2GoldilocksConstants::D16, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            }
                
        }
        initialized = 1;        
    }
    cudaSetDevice(deviceId);
}

void Poseidon2GoldilocksGPU::merkletree_cuda_coalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;


    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu_coalesced_2<<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * 8, stream>>>(d_tree, d_input, num_cols * dim, num_rows);
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

    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;


    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu_coalesced_2_blocks<<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_tree, d_input, num_cols * dim, num_rows);
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