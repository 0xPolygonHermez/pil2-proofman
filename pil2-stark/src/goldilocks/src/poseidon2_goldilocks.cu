#include "goldilocks_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>

#include "poseidon2_goldilocks.hpp"
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
template<uint32_t W> struct Pos2ConstGPU;
template<> struct Pos2ConstGPU<4> {
    static __device__ __forceinline__ const gl64_t* C() { return (const gl64_t*)GPU_C_4; }
    static __device__ __forceinline__ const gl64_t* D() { return (const gl64_t*)GPU_D_4; }
    static constexpr uint64_t N_VALS_C = 53;
};
template<> struct Pos2ConstGPU<8> {
    static __device__ __forceinline__ const gl64_t* C() { return (const gl64_t*)GPU_C_8; }
    static __device__ __forceinline__ const gl64_t* D() { return (const gl64_t*)GPU_D_8; }
    static constexpr uint64_t N_VALS_C = 86;
};
template<> struct Pos2ConstGPU<12> {
    static __device__ __forceinline__ const gl64_t* C() { return (const gl64_t*)GPU_C_12; }
    static __device__ __forceinline__ const gl64_t* D() { return (const gl64_t*)GPU_D_12; }
    static constexpr uint64_t N_VALS_C = 118;
};
template<> struct Pos2ConstGPU<16> {
    static __device__ __forceinline__ const gl64_t* C() { return (const gl64_t*)GPU_C_16; }
    static __device__ __forceinline__ const gl64_t* D() { return (const gl64_t*)GPU_D_16; }
    static constexpr uint64_t N_VALS_C = 150;
};

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
__global__ void merkleNodeKernel(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t* pol_input = (gl64_t *)(&cursor[nextIndex + tid * SPONGE_WIDTH_T]);
    gl64_t* pol_output = (gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY_T]);

    // Read round constants straight from __constant__ memory: it's already cached
    // and broadcast-friendly for this single-input-per-thread kernel, so the
    // previous per-block __shared__ staging (+ __syncthreads) was redundant.
    const gl64_t *GPU_C_GL = Pos2ConstGPU<SPONGE_WIDTH_T>::C();
    const gl64_t *GPU_D_GL = Pos2ConstGPU<SPONGE_WIDTH_T>::D();

    gl64_t aux[SPONGE_WIDTH_T];
    poseidon2PermuteReg<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(aux, pol_input, GPU_C_GL, GPU_D_GL);
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
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::permute(uint64_t * output, const uint64_t * input, cudaStream_t stream){
    permuteKernel<RATE,CAPACITY,SPONGE_WIDTH,N_FULL_ROUNDS_TOTAL,N_PARTIAL_ROUNDS><<<1, 1, SPONGE_WIDTH*sizeof(gl64_t), stream>>>(output, input);
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::permuteTrunc(uint64_t * output, const uint64_t * input, cudaStream_t stream){
    permuteTruncKernel<RATE,CAPACITY,SPONGE_WIDTH,N_FULL_ROUNDS_TOTAL,N_PARTIAL_ROUNDS><<<1, 1, SPONGE_WIDTH*sizeof(gl64_t), stream>>>(output, input);
    CHECKCUDAERR(cudaGetLastError());
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::merkletreeReduce(uint64_t * d_root, uint64_t * d_input, uint64_t num_elements, uint64_t arity, cudaStream_t stream){
    // Compute total tree buffer size (same logic as CPU merkletreeReduce).
    uint64_t numNodes = num_elements;
    uint64_t nodesLevel = num_elements;
    while (nodesLevel > 1) {
        uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
        numNodes += extraZeros;
        numNodes += (nodesLevel + (arity - 1)) / arity;
        nodesLevel = (nodesLevel + (arity - 1)) / arity;
    }

    // Allocate tree buffer on device and copy input digests.
    uint64_t *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_tree, numNodes * CAPACITY * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyAsync(d_tree, d_input, num_elements * CAPACITY * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));

    // Tree reduction: hash groups of `arity` digests using merkleNodeKernel.
    uint64_t pending = num_elements;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1) {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync(d_tree + nextIndex + pending * CAPACITY, 0, extraZeros * CAPACITY * sizeof(uint64_t), stream));

        u32 actual_tpb = (nextN < TPB) ? nextN : TPB;
        u32 actual_blks = (nextN < TPB) ? 1 : (nextN / TPB + 1);

        merkleNodeKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, 0, stream>>>(nextN, nextIndex, pending + extraZeros, d_tree);

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());

    // Copy root (last CAPACITY elements) to output.
    CHECKCUDAERR(cudaMemcpyAsync(d_root, d_tree + nextIndex, CAPACITY * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    CHECKCUDAERR(cudaFree(d_tree));
}

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::linearHash(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, Layout layout, cudaStream_t stream){
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    // RowMajor reads contiguous columns per row (flat kernel); ColMajor and ColMajorTiled both go
    // through the getBufferOffset-based kernel, which honors the exact layout passed in.
    if (layout == Layout::RowMajor) {
        linearHashKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows);
    } else {
        linearHashTiledKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows, layout);
    }
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

    nonceBlock[blockIdx.x] = UINT64_MAX;
    uint64_t idx = nonces_offset + (blockIdx.x * blockDim.x + threadIdx.x) * hashes_per_thread;
    uint64_t level = 1ULL << (64 - n_bits);
    uint64_t locId = UINT64_MAX;

    // STARK grinding contract: 3 challenge field elements + nonce, zero-padded
    // to SPONGE_WIDTH. Middle slots must be zero so prover and verifier match.
    const gl64_t *GPU_C_GL = Pos2ConstGPU<SPONGE_WIDTH_T>::C();
    const gl64_t *GPU_D_GL = Pos2ConstGPU<SPONGE_WIDTH_T>::D();

    gl64_t state[SPONGE_WIDTH_T];
    gl64_t in_reg[SPONGE_WIDTH_T];
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH_T; ++i)
        in_reg[i] = (gl64_t)(uint64_t)0;

    for(uint32_t k=0; k<hashes_per_thread; k++){
        uint64_t idx_k = idx + k;
        // STARK grinding contract:
        //   in_reg[0..2] = FIELD_EXTENSION challenge
        //   in_reg[3]    = nonce
        //   in_reg[4..W-1] = 0 (already zero from init above)
#pragma unroll
        for (uint32_t i = 0; i < 3; ++i)
            in_reg[i] = input[i];
        in_reg[3] = (gl64_t)idx_k;

        poseidon2PermuteReg<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(state, in_reg, GPU_C_GL, GPU_D_GL);

        uint64_t hash_val = (uint64_t)state[0];
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
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::grinding(uint64_t * d_nonce, uint64_t *d_nonceBlock, const uint64_t * d_in, const uint32_t n_bits, cudaStream_t stream){

    uint64_t log_launch_iters = 7; //128 launch iterations
    uint64_t launch_iters = 1ULL << log_launch_iters;
    uint64_t log_N = POSEIDON2_GRIND_BITS; //~512K nonces per launch
    uint64_t N = 1 << log_N;
    uint64_t security = 128;
    // we need to determine log_hashesPerThread such that, the probabilty of not finding a valid nonce is lower
    // than 2^(-security)
    // (1-1/2^n_bits)^(totalHashesRequired) = 2^(-security)
    // totalHashesRequired = log(2^(-security)) / log(1-1/2^n_bits)
    //
    // Numerical notes:
    //   eps    = 2^-n_bits is built with ldexp 
    //   log1p  = ln(1 + x) without the cancellation that hits `log(1.0 - eps)
    double eps                 = ldexp(1.0, -int(n_bits));
    double totalHashesRequired = -double(security) * log(2.0) / log1p(-eps);
    uint64_t log_totalHashesRequired = (uint64_t)ceil(log2(totalHashesRequired));
    uint64_t log_hashesPerThread;
    if(log_totalHashesRequired > log_launch_iters + log_N){
        log_hashesPerThread = log_totalHashesRequired - log_launch_iters - log_N;
    }else{
        log_hashesPerThread = 0;
    }
    uint64_t hashesPerThread = 1ULL << log_hashesPerThread;

    dim3 blockSize( POSEIDON2_GRIND_BLOCKS );
    dim3 gridSize( POSEIDON2_GRIND_GRID );

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
    const gl64_t *GPU_C_GL = Pos2ConstGPU<SPONGE_WIDTH_T>::C();
    const gl64_t *GPU_D_GL = Pos2ConstGPU<SPONGE_WIDTH_T>::D();



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
    uint64_t num_cols, uint64_t num_rows, Layout layout, cudaStream_t stream)
{
    if (num_rows == 0) return;

    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    // RowMajor reads contiguous columns per row (flat kernel); ColMajor and ColMajorTiled both go
    // through the getBufferOffset-based kernel, which honors the exact layout passed in.
    if (layout == Layout::RowMajor) {
        linearHashKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_tree, d_input, num_cols, num_rows);
    } else {
        linearHashTiledKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_tree, d_input, num_cols, num_rows, layout);
    }
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


// Explicit instantiations
template void Poseidon2GoldilocksGPU<4>::initConstants(uint32_t*, uint32_t);
template void Poseidon2GoldilocksGPU<4>::grinding(uint64_t*, uint64_t*, const uint64_t*, uint32_t, cudaStream_t);
template class Poseidon2GoldilocksGPU<8>;
template class Poseidon2GoldilocksGPU<12>;
template class Poseidon2GoldilocksGPU<16>;

#if __GOLDILOCKS_ENV__
// Standalone-build extras: W=4 permute/permuteTrunc are exercised by the
// goldilocks tests/benches but not by the production prover.
template void Poseidon2GoldilocksGPU<4>::permute(uint64_t*, const uint64_t*, cudaStream_t);
template void Poseidon2GoldilocksGPU<4>::permuteTrunc(uint64_t*, const uint64_t*, cudaStream_t);
#endif
