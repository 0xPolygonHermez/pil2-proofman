#include "gl64_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>

#include "poseidon2_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"
#include "poseidon2_goldilocks.cuh"

// #ifdef GPU_TIMING
#include "timer_gl.hpp"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
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
__device__ void hash_one_2(gl64_t *out, gl64_t *const input, int tid)
{
    
    const gl64_t *GPU_C_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_C_4 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_C_12 : (gl64_t *)GPU_C_16);
    const gl64_t *GPU_D_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_D_4 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_D_12 : (gl64_t *)GPU_D_16);
    
    gl64_t aux[SPONGE_WIDTH_T];
    hash_full_result_seq_2<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(aux, input, GPU_C_GL, GPU_D_GL);
    mymemcpy((uint64_t *)out, (uint64_t *)aux, CAPACITY_T);
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
    linearHashGPU<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * 8, stream>>>(d_tree, d_input, num_cols * dim, num_rows);
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


//repassar assumtions que faig, sponge width es 16
template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void hash_gpu_16(uint64_t* data, int N)
{
 
    for(int i=threadIdx.x; i <150; i+= blockDim.x){
        scratchpad[16*blockDim.x + i] = GPU_C_16[i];
    }
    for(int i=threadIdx.x; i <16; i+= blockDim.x){
        scratchpad[16*blockDim.x + 150 + i] = GPU_D_16[i];
    }
    gl64_t* C = scratchpad + 16*blockDim.x;
    gl64_t* D = scratchpad + 16*blockDim.x + 150;

    __syncthreads();
    uint64_t* blockData = data + (blockIdx.x * blockDim.x) * 16;

    //coalesced read of hasshes which are "romajor" sthore in scratchpad in colmajor
    for(uint32_t i=0; i<16; i++){
        uint32_t offset = i * blockDim.x + threadIdx.x;
        uint32_t row = offset >> 4;
        uint32_t col = offset & 0xF;
        scratchpad[col * blockDim.x + row] = gl64_t(blockData[offset]);        
    }
    __syncthreads();
    //poseidon2_hash<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
    poseidon2_hash_with_constants<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>((gl64_t*) C,(gl64_t*) D);
    uint64_t* outData = data + 16 * N + (blockIdx.x * blockDim.x) * 4;
    //coalesced write of hashes that are in "colummajor" in scratchpad and write them in rowmajor in d_tree
    __syncthreads();
    for(uint32_t i=0; i<4; i++){
        uint32_t offset = i * blockDim.x + threadIdx.x;
        uint32_t row = offset >> 2;
        uint32_t col = offset & 0x3;
        outData[(row << 2) + col] = scratchpad[col * blockDim.x + row];
    }
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
    //linearHashGPUTiles_<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_tree, d_input, num_cols * dim, num_rows);
    linearHashGPUTiles<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_tree, d_input, num_cols * dim, num_rows);
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
        if(actual_tpb == TPB && SPONGE_WIDTH == 16 && nextN >=128 && 0){
            hash_gpu_16<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS> <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t)+ (150+16) * sizeof(gl64_t), stream>>>(d_tree + nextIndex, nextN);  
        }else{
            hash_gpu_3<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, 0, stream>>>(nextN, nextIndex, pending + extraZeros, d_tree);
        }
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void linearHashGPUTiles_(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t num_cols, uint32_t num_rows)
{

    //const gl64_t *GPU_C_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_C_4 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_C_12 : (gl64_t *)GPU_C_16);
    //const gl64_t *GPU_D_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_D_4 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_D_12 : (gl64_t *)GPU_D_16);
    for(int i=0; i<150; i+= blockDim.x){
        scratchpad[i] = GPU_C_16[i];
    }
    for(int i=0; i<16; i+= blockDim.x){
        scratchpad[150 + i] = GPU_D_16[i];
    }
    __syncthreads();
    const gl64_t *GPU_C_GL = scratchpad;
    const gl64_t *GPU_D_GL = scratchpad + 150;

    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if( row >= num_rows)
        return;

    if (num_cols <= CAPACITY_T)
    {
        uint64_t out_idx = row*CAPACITY_T;
        #pragma unroll
        for (uint32_t i = 0; i < CAPACITY_T; i++) {
            if (i < num_cols){
                output[out_idx + i] = input[getBufferOffset(row, i, num_rows, num_cols)];
            } else {
                output[out_idx + i] = gl64_t(uint64_t(0));
            }
        }
    }
    else
    {
        gl64_t state[SPONGE_WIDTH_T];
        #pragma unroll
        for (uint32_t i = 0; i < CAPACITY_T; i++){
            state[RATE_T+i] =  gl64_t(uint64_t(0));
        }
        for (uint32_t col = 0;;)
        {
            __syncwarp();
            uint32_t delta = min(num_cols - col, RATE_T);
            for (uint32_t i = 0; i < delta; i++) {
                state[i] = input[getBufferOffset(row, col + i, num_rows, num_cols)];
            }
            for (uint32_t i = delta; i < RATE_T; i++) {
                state[i] = gl64_t(uint64_t(0));
            }
            __syncwarp();
            hash_full_result_seq_2_<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(state, GPU_C_GL, GPU_D_GL);

            if ((col += RATE_T) >= num_cols)
                break;
            __syncwarp();
            for (uint32_t i = 0; i < CAPACITY_T; i++)
                state[i + RATE_T] = state[i];
        }
        uint64_t out_idx = row*CAPACITY_T;
        for (uint32_t i = 0; i < CAPACITY_T; i++) {
            output[out_idx+i] = state[i];            
        }
    }
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
    linearHashGPUTiles<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows);
    //linearHashGPUTiles_<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows);
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void grinding_calc_(uint64_t* nonce, uint64_t *__restrict__ nonceBlock, uint64_t *__restrict__ input, uint64_t n_bits, uint64_t hashes_per_thread, uint64_t nonces_offset)
{

    if(nonces_offset != 0 && nonce[0] != UINT64_MAX)
        return;

    uint64_t* shared_nonces = (uint64_t*)&scratchpad[SPONGE_WIDTH_T * blockDim.x];

    //check if was found in previous launch
    if(threadIdx.x ==0){
        shared_nonces[0] = UINT64_MAX;
        if(blockIdx.x ==0){
            nonce[0] = UINT64_MAX;
        }
        if(nonces_offset != 0) { //not first iteration
            for(int i=0; i<gridDim.x; ++i){
                if(nonceBlock[i] != UINT64_MAX){
                    shared_nonces[0] = nonceBlock[i];
                    if(blockIdx.x ==0){
                        nonce[0] = nonceBlock[i];
                    }
                    break;
                }
            }
        }
    }
    __syncthreads();
    if(shared_nonces[0] != UINT64_MAX){
        return;
    }
    // scratchpad is declared globally, shared_nonces is allocated right after it
    
    nonceBlock[blockIdx.x] = UINT64_MAX;
    uint64_t idx = nonces_offset + (blockIdx.x * blockDim.x + threadIdx.x) * hashes_per_thread;
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
        nonceBlock[blockIdx.x] = shared_nonces[0];
    }   
}


__global__ void grinding_check_(uint64_t* nonce, uint64_t *__restrict__ nonceBlock, uint32_t n_blocks)
{
    
    if(nonce[0] != UINT64_MAX)
        return;
    if(threadIdx.x > 31 || blockIdx.x > 0){
        return;
    }
    uint32_t stride = min(blockDim.x, 32);
    __shared__  uint64_t local_nonceBlock[32];
    local_nonceBlock[threadIdx.x] = UINT64_MAX;
    for(uint32_t i=threadIdx.x; i<n_blocks; i+=stride){
        if(nonceBlock[i] != UINT64_MAX && local_nonceBlock[threadIdx.x] == UINT64_MAX){
            local_nonceBlock[threadIdx.x] = nonceBlock[i];
        }
    }
    __syncthreads();
    nonce[0] = UINT64_MAX;
    if(threadIdx.x == 0){
        for(uint32_t i=0; i<stride; i++){
            if(local_nonceBlock[i] < nonce[0]){
                nonce[0] = local_nonceBlock[i];
            }
       }
    }
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::grinding(uint64_t * d_nonce, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream){

    uint64_t log_launch_iters = 7; //64 launch iterations
    uint64_t launch_iters = 1ULL << log_launch_iters;
    uint64_t log_N = 19; //512K nonces per launch
    uint64_t N = 1 << log_N;
    uint64_t security = 128;
    // we need to determine log_hashesPerThread such that, suthat probabilty of not finding a valid nonce is lower
    // than 2^(-security)
    // (1-1/2^n_bits)^(totalHashesRequired) = 2^(-security)
    // totalHashesRequired = log(2^(-security)) / log(1-1/2^n_bits)
    double totalHashesRequired =(double(-double(security))) * log(double(2.0))/log(double(1.0)-double(1.0)/double(1ULL << (n_bits)));
    uint64_t log_totalHashesRequired = ceil(log2(totalHashesRequired));
    uint64_t log_hashesPerThread;
    if(log_totalHashesRequired > log_launch_iters + log_N){
        log_hashesPerThread = log_totalHashesRequired - log_launch_iters - log_N;
    }else{
        log_hashesPerThread = 0;
    }
    uint64_t hashesPerThread = 1ULL << log_hashesPerThread;
    
    dim3 blockSize( 512 );
    dim3 gridSize( (N + blockSize.x - 1) / blockSize.x );
    uint64_t* d_nonceBlock;
    cudaMalloc((void**)&d_nonceBlock, sizeof(uint64_t)*gridSize.x);

    size_t shared_mem_size = blockSize.x * SPONGE_WIDTH * sizeof(gl64_t) + blockSize.x * sizeof(uint64_t);
    uint64_t nonces_offset = 0;
    uint64_t nonces_per_iteration = N * hashesPerThread;

    for(int i=0; i<launch_iters; ++i){
        grinding_calc_<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<gridSize, blockSize, shared_mem_size, stream>>>((uint64_t *)d_nonce, (uint64_t *)d_nonceBlock, (uint64_t *)d_in, n_bits, hashesPerThread, nonces_offset);
        nonces_offset += nonces_per_iteration;
    }
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

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void grinding_persistent_kernel(
    uint64_t* __restrict__ d_nonce,           // global result (single uint64_t)
    uint64_t* __restrict__ d_nonceBlock,      // per-block minima written here
    const uint64_t* __restrict__ d_input,     
    uint32_t n_bits,
    uint32_t hashes_per_thread,
    uint64_t max_iterations)
{
    cg::grid_group grid = cg::this_grid();

    uint64_t* shared_nonces = reinterpret_cast<uint64_t*>(scratchpad + (size_t)SPONGE_WIDTH_T * (size_t)blockDim.x);
    const uint64_t level = 1ULL << (64 - n_bits);
    const uint64_t nonces_per_iteration = gridDim.x * blockDim.x * (uint64_t)hashes_per_thread;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *d_nonce = UINT64_MAX;
    }
    grid.sync(); // ensure d_nonce initialized

    uint64_t iteration = 0;
    while (true) {
        
        uint64_t offset = iteration * nonces_per_iteration;

        uint64_t nonce_base = offset + ( (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x ) * (uint64_t)hashes_per_thread;
        uint64_t loc_nonce = UINT64_MAX;

        for (uint32_t k = 0; k < hashes_per_thread; ++k) {
            uint64_t nonce_k = nonce_base + k;

            #pragma unroll
            for (uint32_t i = 0; i < SPONGE_WIDTH_T - 1; ++i) {
                scratchpad[(size_t)i * blockDim.x + threadIdx.x] = d_input[i];
            }
            scratchpad[(size_t)(SPONGE_WIDTH_T - 1) * blockDim.x + threadIdx.x] = nonce_k;
            poseidon2_hash<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
            uint64_t hash_val = (uint64_t)scratchpad[threadIdx.x];
            if (hash_val < level) {
                loc_nonce = nonce_k;
                break;
            }
        }
        shared_nonces[threadIdx.x] = loc_nonce;
        __syncthreads();

        // intra-block reduction to find minimum (same as original)
        uint32_t alive = blockDim.x >> 1;
        while (alive > 0) {
            if (threadIdx.x < alive) {
                uint64_t a = shared_nonces[threadIdx.x];
                uint64_t b = shared_nonces[threadIdx.x + alive];
                if (b < a) shared_nonces[threadIdx.x] = b;
            }
            __syncthreads();
            alive >>= 1;
        }

        // thread 0 writes per-block minimum into global d_nonceBlock
        if (threadIdx.x == 0) {
            d_nonceBlock[blockIdx.x] = shared_nonces[0];
        }

        grid.sync(); // ensure all blocks have written their minima

        // block 0 scans per-block minima to find global minimum and writes into d_nonce
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            uint64_t found = UINT64_MAX;
            for (uint32_t i = 0; i < gridDim.x; ++i) {
                uint64_t v = d_nonceBlock[i];
                if (v < found){
                    found = v;
                    break; // stop at first found (will be the minimum)
                }
            }
            *d_nonce = found;
        }

        grid.sync(); // ensure all blocks see d_nonce

        // if found or reached max iterations, exit
        if (*d_nonce != UINT64_MAX) break;
        ++iteration;
        if (iteration >= max_iterations) break;

    } 
    grid.sync();
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


