#include "goldilocks_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>

#include "poseidon2_goldilocks.hpp"
#include "poseidon2_goldilocks.cuh"
#include "grinding_gpu.cuh"

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
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::linearHash(uint64_t * d_hash_output, uint64_t * d_trace, uint64_t num_cols, uint64_t num_rows, Layout layout, cudaStream_t stream, uint64_t ld){
    if (num_rows == 0) return;
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    // The flat RowMajor kernel has no row bound, so only whole blocks take it.
    if (layout == Layout::RowMajor && (num_rows < TPB || num_rows % TPB == 0)) {
        linearHashKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows);
    } else {
        linearHashTiledKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
            <<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), stream>>>(d_hash_output, d_trace, num_cols, num_rows, layout, ld ? ld : num_rows);
    }
    CHECKCUDAERR(cudaGetLastError());
}

// Grinding hash: [c0, c1, c2, nonce, 0, ...] -> first output limb (the STARK grinding contract).
template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t SPONGE_WIDTH_T, uint32_t N_FULL_ROUNDS_TOTAL_T, uint32_t N_PARTIAL_ROUNDS_T>
struct Pos2GrindHash
{
    static constexpr uint32_t ROUNDS = 1;
    gl64_t in_reg[SPONGE_WIDTH_T];
    __device__ explicit Pos2GrindHash(const uint64_t *in)
    {
#pragma unroll
        for (uint32_t i = 0; i < SPONGE_WIDTH_T; ++i)
            in_reg[i] = (gl64_t)(i < 3 ? in[i] : (uint64_t)0);
    }
    __device__ uint64_t operator()(uint64_t n)
    {
        gl64_t state[SPONGE_WIDTH_T];
        in_reg[3] = (gl64_t)n;
        poseidon2PermuteReg<RATE_T, CAPACITY_T, SPONGE_WIDTH_T, N_FULL_ROUNDS_TOTAL_T, N_PARTIAL_ROUNDS_T>(
            state, in_reg, Pos2ConstGPU<SPONGE_WIDTH_T>::C(), Pos2ConstGPU<SPONGE_WIDTH_T>::D());
        return (uint64_t)state[0];
    }
};

template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::grinding(uint64_t * d_nonce, uint64_t *d_nonceBlock, const uint64_t * d_in, const uint32_t n_bits, cudaStream_t stream){
    grindMin<Pos2GrindHash<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>>(d_nonce, d_nonceBlock, d_in, n_bits, stream);
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
    linearHash(d_tree, d_input, num_cols, num_rows, layout, stream);
    reduceLevels(arity, d_tree, num_rows, stream);
}

// `pending` digests (a multiple of arity) at d_nodes -> their parents, written right after them.
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::reduceOneLevel(uint32_t arity, uint64_t *d_nodes, uint64_t pending, cudaStream_t stream)
{
    uint64_t nextN = pending / arity;
    u32 actual_tpb = (nextN < TPB) ? nextN : TPB;
    u32 actual_blks = (nextN < TPB) ? 1 : (nextN / TPB + 1);
    merkleNodeKernel<RATE, CAPACITY, SPONGE_WIDTH, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS>
        <<<actual_blks, actual_tpb, 0, stream>>>(nextN, 0, pending, d_nodes);
    CHECKCUDAERR(cudaGetLastError());
}

// Every level above the `pending` digests at the start of d_tree.
template<uint32_t SPONGE_WIDTH_T>
void Poseidon2GoldilocksGPU<SPONGE_WIDTH_T>::reduceLevels(uint32_t arity, uint64_t *d_tree, uint64_t pending, cudaStream_t stream)
{
    uint64_t nextIndex = 0;
    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync((uint64_t *)(d_tree + nextIndex + pending * CAPACITY), 0, extraZeros * CAPACITY * sizeof(uint64_t), stream));
        reduceOneLevel(arity, d_tree + nextIndex, pending + extraZeros, stream);
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
    }
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
