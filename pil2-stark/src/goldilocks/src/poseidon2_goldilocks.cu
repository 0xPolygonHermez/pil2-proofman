#include "gl64_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>

#include "poseidon2_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"
#include "poseidon2_goldilocks.cuh"

#ifdef GPU_TIMING
#include "timer_gl.hpp"
#endif

typedef uint32_t u32;
typedef uint64_t u64;

// CUDA Threads per Block
#define TPB 128

__device__ __constant__ uint64_t GPU_C_4[53];
__device__ __constant__ uint64_t GPU_D_4[4];
__device__ __constant__ uint64_t GPU_C_8[86];
__device__ __constant__ uint64_t GPU_D_8[8];
__device__ __constant__ uint64_t GPU_C_12[118];
__device__ __constant__ uint64_t GPU_D_12[12];
__device__ __constant__ uint64_t GPU_C_16[150];
__device__ __constant__ uint64_t GPU_D_16[16];

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void merkleNodeKernel(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t* pol_input = (gl64_t *)(&cursor[nextIndex + tid * SPONGE_WIDTH_T]);
    gl64_t* pol_output = (gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY_T]);

    const gl64_t *GPU_C_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_C_4 : (SPONGE_WIDTH_T==8 ? (gl64_t *)GPU_C_8 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_C_12 : (gl64_t *)GPU_C_16));
    const uint64_t N_VALS_C = SPONGE_WIDTH_T==4 ? 53 : (SPONGE_WIDTH_T==8 ? 86 : (SPONGE_WIDTH_T==12 ? 118 : 150));
    const gl64_t *GPU_D_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_D_4 : (SPONGE_WIDTH_T==8 ? (gl64_t *)GPU_D_8 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_D_12 : (gl64_t *)GPU_D_16));

    __shared__ gl64_t GPU_C_SM[150];
    __shared__ gl64_t GPU_D_SM[16];

    if (threadIdx.x == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, (uint64_t *)GPU_C_GL, N_VALS_C);
        mymemcpy((uint64_t *)GPU_D_SM, (uint64_t *)GPU_D_GL, SPONGE_WIDTH_T);
    }
    __syncthreads();

    gl64_t aux[SPONGE_WIDTH_T];
    poseidon2PermuteReg<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(aux, pol_input, GPU_C_SM, GPU_D_SM);
    mymemcpy((uint64_t *)pol_output, (uint64_t *)aux, CAPACITY_T);
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids)
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
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C_8, Poseidon2GoldilocksConstants::C8, 86 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D_8, Poseidon2GoldilocksConstants::D8, 8 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
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
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::hash(uint64_t * output, const uint64_t * input, cudaStream_t stream){
    hashKernel<RATE,CAPACITY,SPONGE_WIDTH,N_FULL_ROUNDS_TOTAL,N_PARTIAL_ROUNDS ><<<1, 1, SPONGE_WIDTH*sizeof(gl64_t), stream>>>(output, input);
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::linearHashTiled(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream){
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linearHashTiledKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows);
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::linearHash(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream){
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linearHashKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows);
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void grindingKernel(uint64_t* nonce, uint64_t *__restrict__ nonceBlock, uint64_t *__restrict__ input, uint64_t n_bits, uint64_t hashes_per_thread, uint64_t nonces_offset)
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
        poseidon2PermuteSmem<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>();
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

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::grinding(uint64_t * d_nonce, uint64_t *d_nonceBlock, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream){

    uint64_t log_launch_iters = 7; //128 launch iterations
    uint64_t launch_iters = 1ULL << log_launch_iters;
    uint64_t log_N = NONCES_LAUNCH_BITS; //~512K nonces per launch
    uint64_t N = 1 << log_N;
    uint64_t security = 128;
    // we need to determine log_hashesPerThread such that, the probabilty of not finding a valid nonce is lower
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

    dim3 blockSize( NONCES_LAUNCH_BLOCKS );
    dim3 gridSize( NONCES_LAUNCH_GRID_SIZE );

    size_t shared_mem_size = blockSize.x * SPONGE_WIDTH * sizeof(gl64_t) + blockSize.x * sizeof(uint64_t);
    uint64_t nonces_offset = 0;
    uint64_t nonces_per_iteration = N * hashesPerThread;

    for(int i=0; i<launch_iters; ++i){
        grindingKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS><<<gridSize, blockSize, shared_mem_size, stream>>>((uint64_t *)d_nonce, (uint64_t *)d_nonceBlock, (uint64_t *)d_in, n_bits, hashesPerThread, nonces_offset);
        nonces_offset += nonces_per_iteration;
    }
}

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__device__  void poseidon2PermuteSmem()
{
    const gl64_t *GPU_C_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_C_4 : (SPONGE_WIDTH_T==8 ? (gl64_t *)GPU_C_8 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_C_12 : (gl64_t *)GPU_C_16));
    const gl64_t *GPU_D_GL = SPONGE_WIDTH_T==4 ? (gl64_t *)GPU_D_4 : (SPONGE_WIDTH_T==8 ? (gl64_t *)GPU_D_8 : (SPONGE_WIDTH_T==12 ? (gl64_t *)GPU_D_12 : (gl64_t *)GPU_D_16));

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

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::merkletree(
    uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
    uint64_t num_cols, uint64_t num_rows, cudaStream_t stream)
{
    if (num_rows == 0) return;

    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linearHashKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
        <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_tree, d_input, num_cols, num_rows);
    CHECKCUDAERR(cudaGetLastError());

    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync((uint64_t *)(d_tree + nextIndex + pending * CAPACITY), 0, extraZeros * CAPACITY * sizeof(uint64_t), stream));

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
        merkleNodeKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, 0, stream>>>(nextN, nextIndex, pending + extraZeros, d_tree);
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::merkletreeTiled(
    uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
    uint64_t num_cols, uint64_t num_rows, cudaStream_t stream)
{
    if (num_rows == 0) return;

    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linearHashTiledKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
        <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_tree, d_input, num_cols, num_rows);
    CHECKCUDAERR(cudaGetLastError());

    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync((uint64_t *)(d_tree + nextIndex + pending * CAPACITY), 0, extraZeros * CAPACITY * sizeof(uint64_t), stream));

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
        merkleNodeKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, 0, stream>>>(nextN, nextIndex, pending + extraZeros, d_tree);
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

void buildMerkleTreeTilesGPU(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                              uint64_t nCols, uint64_t nRows, cudaStream_t stream)
{
    switch (arity)
    {
    case 2:
        Poseidon2GoldilocksGPU<8>::merkletreeTiled(arity, d_tree, d_input, nCols, nRows, stream);
        break;
    case 3:
        Poseidon2GoldilocksGPU<12>::merkletreeTiled(arity, d_tree, d_input, nCols, nRows, stream);
        break;
    case 4:
        Poseidon2GoldilocksGPU<16>::merkletreeTiled(arity, d_tree, d_input, nCols, nRows, stream);
        break;
    default:
#ifndef __GOLDILOCKS_ENV__
        zklog.error("buildMerkleTreeTilesGPU: Unsupported arity");
        exitProcess();
#endif
        exit(-1);
    }
}

void buildMerkleTreeGPU(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                         uint64_t nCols, uint64_t nRows, cudaStream_t stream)
{
    switch (arity)
    {
    case 2:
        Poseidon2GoldilocksGPU<8>::merkletree(arity, d_tree, d_input, nCols, nRows, stream);
        break;
    case 3:
        Poseidon2GoldilocksGPU<12>::merkletree(arity, d_tree, d_input, nCols, nRows, stream);
        break;
    case 4:
        Poseidon2GoldilocksGPU<16>::merkletree(arity, d_tree, d_input, nCols, nRows, stream);
        break;
    default:
#ifndef __GOLDILOCKS_ENV__
        zklog.error("buildMerkleTreeGPU: Unsupported arity");
        exitProcess();
#endif
        exit(-1);
    }
}

// Explicit instantiation — Poseidon2
template void Poseidon2GoldilocksGPUGrinding::initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPUGrinding::grinding(uint64_t * d_nonce, uint64_t *d_nonceBlock, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream);

template void Poseidon2GoldilocksGPU<8>::initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPU<8>::hash(uint64_t * output, const uint64_t * input, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<12>::initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPU<12>::hash(uint64_t * output, const uint64_t * input, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<16>::initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
template void Poseidon2GoldilocksGPU<16>::hash(uint64_t * output, const uint64_t * input, cudaStream_t stream);

template void Poseidon2GoldilocksGPU<8>::merkletree(uint32_t, uint64_t*, uint64_t*, uint64_t, uint64_t, cudaStream_t);
template void Poseidon2GoldilocksGPU<8>::merkletreeTiled(uint32_t, uint64_t*, uint64_t*, uint64_t, uint64_t, cudaStream_t);
template void Poseidon2GoldilocksGPU<12>::merkletree(uint32_t, uint64_t*, uint64_t*, uint64_t, uint64_t, cudaStream_t);
template void Poseidon2GoldilocksGPU<12>::merkletreeTiled(uint32_t, uint64_t*, uint64_t*, uint64_t, uint64_t, cudaStream_t);
template void Poseidon2GoldilocksGPU<16>::merkletree(uint32_t, uint64_t*, uint64_t*, uint64_t, uint64_t, cudaStream_t);
template void Poseidon2GoldilocksGPU<16>::merkletreeTiled(uint32_t, uint64_t*, uint64_t*, uint64_t, uint64_t, cudaStream_t);

#if __GOLDILOCKS_ENV__
template void Poseidon2GoldilocksGPU<4>::hash(uint64_t * output, const uint64_t * input, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<16>::grinding(uint64_t * d_nonce, uint64_t *d_nonceBlock, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<16>::linearHash(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<16>::linearHashTiled(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<12>::linearHash(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<12>::linearHashTiled(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, cudaStream_t stream);
template void Poseidon2GoldilocksGPU<12>::grinding(uint64_t * d_nonce, uint64_t *d_nonceBlock, const uint64_t * d_in, uint32_t n_bits, cudaStream_t stream);
#endif
