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
__global__ void grinding_calc_(uint64_t *__restrict__ indxBlock, uint64_t *__restrict__ input, uint32_t n_bits, uint32_t hashes_per_thread, uint64_t offset)
{
    // scratchpad is declared globally, shared_nonces is allocated right after it
    uint64_t* shared_nonces = (uint64_t*)&scratchpad[SPONGE_WIDTH_T * blockDim.x];
    
    indxBlock[blockIdx.x] = UINT64_MAX;
    uint64_t idx = offset + (blockIdx.x * blockDim.x + threadIdx.x) * hashes_per_thread;
    uint64_t level = 1ULL << (64 - n_bits);
    uint64_t locId = UINT64_MAX;

    for(uint32_t k=0; k<hashes_per_thread; k++){
        uint64_t idx_k = idx + k;        
        #pragma unroll
        for (uint32_t i = 0; i < SPONGE_WIDTH_T-1; i++)
            scratchpad[i * blockDim.x + threadIdx.x] = input[i];
        scratchpad[(SPONGE_WIDTH_T-1) * blockDim.x + threadIdx.x] = idx_k;
        poseidon2_hash<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
        // Compare the raw uint64 value, not the field element
        uint64_t hash_val = (uint64_t)scratchpad[threadIdx.x];
        if(hash_val < level){
            locId = idx_k;
            break;
        }
    } 
    shared_nonces[threadIdx.x] = locId;
    __syncthreads();
    //reduce to find the minimum nonce value
    uint32_t alive = blockDim.x >> 1;
    while(alive > 0){
        if(threadIdx.x < alive && shared_nonces[threadIdx.x + alive] < shared_nonces[threadIdx.x]){
            shared_nonces[threadIdx.x] = shared_nonces[threadIdx.x + alive];
        }
        __syncthreads();
        alive >>= 1;
    }
    if(threadIdx.x == 0){
        indxBlock[blockIdx.x] = shared_nonces[0];
    }
    
}

__global__ void grinding_check_(uint64_t* indx, uint64_t *__restrict__ indxBlock, uint32_t n_blocks)
{
    if(threadIdx.x > 31 || blockIdx.x > 0){
        return;
    }
    uint32_t stride = min(blockDim.x, 32);
    __shared__  uint64_t local_indxBlock[32];
    local_indxBlock[threadIdx.x] = UINT64_MAX;
    for(uint32_t i=threadIdx.x; i<n_blocks; i+=stride){
        if(indxBlock[i] != UINT64_MAX && local_indxBlock[threadIdx.x] == UINT64_MAX){
            local_indxBlock[threadIdx.x] = indxBlock[i];
        }
    }
    __syncthreads();
    indx[0] = UINT64_MAX;
    if(threadIdx.x == 0){
        for(uint32_t i=0; i<stride; i++){
            if(local_indxBlock[i] < indx[0]){
                indx[0] = local_indxBlock[i];
            }
       }
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::grinding(uint64_t * d_out, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream){
    uint32_t hashesPerThread = 2;
    uint64_t N = 1 << 21; // Search 2M x 2 nonces per iteration
    dim3 blockSize( 128 );
    dim3 gridSize( (N/hashesPerThread + blockSize.x - 1) / blockSize.x );

    uint64_t* d_indxBlock;
    cudaMalloc((void**)&d_indxBlock, sizeof(uint64_t)*gridSize.x);
    bool found = false;
    uint64_t offset = 0;
    uint64_t nonces_per_iteration = gridSize.x * blockSize.x * hashesPerThread;
    uint64_t iteration_count = 0;
    uint64_t max_iterations = 10000; // Prevent infinite loop
    
    while (found == false && iteration_count < max_iterations)
    {
        size_t shared_mem_size = blockSize.x * SPONGE_WIDTH * sizeof(gl64_t) + blockSize.x * sizeof(uint64_t);
        grinding_calc_<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<gridSize, blockSize, shared_mem_size, stream>>>((uint64_t *)d_indxBlock, (uint64_t *)d_in, n_bits, hashesPerThread, offset);
        CHECKCUDAERR(cudaGetLastError());

        grinding_check_<<<1, 32, 0, stream>>>((uint64_t *)d_out, (uint64_t *)d_indxBlock, gridSize.x);
        CHECKCUDAERR(cudaGetLastError());

        uint64_t h_indx;
        CHECKCUDAERR(cudaMemcpyAsync(&h_indx, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        CHECKCUDAERR(cudaStreamSynchronize(stream));
        iteration_count++;
        if (h_indx != UINT64_MAX)
        {
            found = true;
        }
        else
        {
            offset += nonces_per_iteration;
        }
    }
    CHECKCUDAERR(cudaGetLastError());
    cudaFree(d_indxBlock);
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
template void Poseidon2GoldilocksGPUGrinding::grinding(uint64_t * d_out, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream);

template void Poseidon2GoldilocksGPUCommit::initPoseidon2GPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPUCommit::merkletreeCoalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim);
template void Poseidon2GoldilocksGPUCommit::merkletreeCoalescedBlocks(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim);

#if __GOLDILOCKS_ENV__
template void Poseidon2GoldilocksGPUCommit::grinding(uint64_t * d_out, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream);
template void Poseidon2GoldilocksGPUCommit::linearHashCoalescedBlocks(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream);
template void Poseidon2GoldilocksGPUCommit::hashFullResult(uint64_t * output, const uint64_t * input);

template void Poseidon2GoldilocksGPU<12>::initPoseidon2GPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPU<12>::linearHashCoalescedBlocks(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<12>::merkletreeCoalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim);
template void Poseidon2GoldilocksGPU<12>::merkletreeCoalescedBlocks(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream, int nThreads, uint64_t dim);
template void Poseidon2GoldilocksGPU<12>::hashFullResult(uint64_t * output, const uint64_t * input);
template void Poseidon2GoldilocksGPU<12>::grinding(uint64_t * d_out, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream);

template void Poseidon2GoldilocksGPUGrinding::hashFullResult(uint64_t * output, const uint64_t * input);

#endif


