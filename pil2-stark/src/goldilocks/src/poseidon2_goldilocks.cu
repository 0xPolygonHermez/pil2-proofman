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


__device__ __constant__ uint64_t GPU_C_4[53]; 
__device__ __constant__ uint64_t GPU_D_4[4];
__device__ __constant__ uint64_t GPU_C_12[118]; 
__device__ __constant__ uint64_t GPU_D_12[12];
__device__ __constant__ uint64_t GPU_C_16[150]; 
__device__ __constant__ uint64_t GPU_D_16[16];

/* --- integration --- */
template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__ void hash_one_2(gl64_t *state, gl64_t *const input, int tid)
{
    __shared__ gl64_t GPU_C_SM[SPONGE_WIDTH_T * N_FULL_ROUNDS_TOTAL_T + N_PARTIAL_ROUNDS_T];
    __shared__ gl64_t GPU_D_SM[SPONGE_WIDTH_T];

    if (tid == 0)
    {
        if (SPONGE_WIDTH_T == 4) {
            mymemcpy((uint64_t *)GPU_C_SM, GPU_C_4, 53);
            mymemcpy((uint64_t *)GPU_D_SM, GPU_D_4, 4);
        }
        else if (SPONGE_WIDTH_T == 12) {
            mymemcpy((uint64_t *)GPU_C_SM, GPU_C_12, 118);
            mymemcpy((uint64_t *)GPU_D_SM, GPU_D_12, 12);
        }
        else if (SPONGE_WIDTH_T == 16) {
            mymemcpy((uint64_t *)GPU_C_SM, GPU_C_16, 150);
            mymemcpy((uint64_t *)GPU_D_SM, GPU_D_16, 16);
        }        
    }
    __syncthreads();


    gl64_t aux[SPONGE_WIDTH_T];
    hash_full_result_seq_2<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(aux, input, GPU_C_SM, GPU_D_SM);
    mymemcpy((uint64_t *)state, (uint64_t *)aux, CAPACITY_T);
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::initPoseidon2GPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids)
{    
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    static int initialized = 0;
    if (initialized == 0)
    {
        for(int i = 0; i < num_gpu_ids; i++)
        {
            CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C_4, Poseidon2GoldilocksConstants::C4, 53 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D_4, Poseidon2GoldilocksConstants::D4, 4 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C_12, Poseidon2GoldilocksConstants::C12, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D_12, Poseidon2GoldilocksConstants::D12, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C_16, Poseidon2GoldilocksConstants::C16, 150 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D_16, Poseidon2GoldilocksConstants::D16, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        }
        initialized = 1;        
    }
    cudaSetDevice(deviceId);
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::merkletreeCoalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim)
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
    linear_hash_gpu_coalesced_2<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * 8, stream>>>(d_tree, d_input, num_cols * dim, num_rows);
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
        hash_gpu_3<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, 0, stream>>>(nextN, nextIndex, pending + extraZeros, d_tree);       
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::merkletreeCoalescedBlocks(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim)
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
    linear_hash_gpu_coalesced_2_blocks<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_tree, d_input, num_cols * dim, num_rows);
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
        hash_gpu_3<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, 0, stream>>>(nextN, nextIndex, pending + extraZeros, d_tree);       
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::hashFullResult(uint64_t * output, const uint64_t * input){
    hash_full_result_2<RATE,CAPACITY,SPONGE_WIDTH,N_FULL_ROUNDS_TOTAL,N_PARTIAL_ROUNDS ><<<1, 1, SPONGE_WIDTH*sizeof(gl64_t)>>>(output, input);
    CHECKCUDAERR(cudaGetLastError());
}
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::linearHashCoalescedBlocks(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream){
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu_coalesced_2_blocks<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows);
    CHECKCUDAERR(cudaGetLastError());
}



template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__  void poseidon2_hash()
{
    const gl64_t *GPU_C_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_C_4 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_C_12 : (gl64_t *)GPU_C_16);
    const gl64_t *GPU_D_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_D_4 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_D_12 : (gl64_t *)GPU_D_16);

    matmul_external_state_<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    for (int r = 0; r < (N_FULL_ROUNDS_TOTAL_T>>1); r++)
    {
        pow7add_state_<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(&(GPU_C_GL[r * SPONGE_WIDTH_T]));
        matmul_external_state_<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    }

    for(int r = 0; r < N_PARTIAL_ROUNDS_T; r++)
    {
        scratchpad[threadIdx.x] = scratchpad[threadIdx.x] + GPU_C_GL[(N_FULL_ROUNDS_TOTAL_T>>1) * SPONGE_WIDTH_T + r];
        pow7_2(scratchpad[threadIdx.x]);
        gl64_t sum_;
        sum_ = gl64_t(uint64_t(0));
        add_state_2<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(&sum_);
        prodadd_state_<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(GPU_D_GL, sum_);
    }

    for (int r = 0; r < (N_FULL_ROUNDS_TOTAL_T>>1); r++)
    {
        pow7add_state_<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(&(GPU_C_GL[(N_FULL_ROUNDS_TOTAL_T>>1) * SPONGE_WIDTH_T + N_PARTIAL_ROUNDS_T + r * SPONGE_WIDTH_T]));
        matmul_external_state_<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    }
}

// Explicit instantiation for class methods
template void Poseidon2GoldilocksGPUGrinding::initPoseidon2GPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPUCommit::initPoseidon2GPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPUCommit::merkletreeCoalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim);
template void Poseidon2GoldilocksGPUCommit::merkletreeCoalescedBlocks(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim);
template void Poseidon2GoldilocksGPUGrinding::hashFullResult(uint64_t * output, const uint64_t * input);
template void Poseidon2GoldilocksGPUCommit::hashFullResult(uint64_t * output, const uint64_t * input);

#if __GOLDILOCKS_ENV__
template void Poseidon2GoldilocksGPU<16>::initPoseidon2GPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPU<12>::linearHashCoalescedBlocks(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<16>::linearHashCoalescedBlocks(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<16>::merkletreeCoalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim);
template void Poseidon2GoldilocksGPU<16>::merkletreeCoalescedBlocks(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim);
template void Poseidon2GoldilocksGPU<16>::hashFullResult(uint64_t * output, const uint64_t * input);
#endif


