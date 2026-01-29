#include "cuda_utils.cuh"
#include "poseidon_bn128.cuh"
#include "poseidon_bn128_constants.hpp"  // Shared CPU/GPU constants (binary compatible)
#include <cuda_runtime.h>
#include "data_layout.cuh"

typedef PoseidonBN128GPU::FrElement FrElementGPU;

// Device-side pointers arrays indexed by (t-2)
__device__ FrElementGPU *GPU_C_ptr[16] = {nullptr};
__device__ FrElementGPU *GPU_M_ptr[16] = {nullptr};
__device__ FrElementGPU *GPU_P_ptr[16] = {nullptr};
__device__ FrElementGPU *GPU_S_ptr[16] = {nullptr};

// Host-side pointers to device memory for cleanup
static FrElementGPU *h_GPU_C[16] = {nullptr};
static FrElementGPU *h_GPU_M[16] = {nullptr};
static FrElementGPU *h_GPU_P[16] = {nullptr};
static FrElementGPU *h_GPU_S[16] = {nullptr};

// Track if constants have been initialized
static bool constants_initialized = false;

// Partial round counts
__device__ __constant__ int N_ROUNDS_P_POSEIDON[16] = {56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68};

__global__ void poseidon_hash_kernel(FrElementGPU *state, int t) {
    PoseidonBN128GPU poseidon;
    
    // Get constants from device global pointers
    const FrElementGPU *C = GPU_C_ptr[t - 2];
    const FrElementGPU *M = GPU_M_ptr[t - 2];
    const FrElementGPU *P = GPU_P_ptr[t - 2];
    const FrElementGPU *S = GPU_S_ptr[t - 2];
    const int nRoundsP = N_ROUNDS_P_POSEIDON[t - 2];
    
    poseidon.hash_(state, t, C, M, P, S, nRoundsP);
}

void PoseidonBN128GPU::hash(FrElement* d_state, int t) {
    poseidon_hash_kernel<<<1, 1>>>(d_state, t);
}

// Helper macro for initializing a single t value
#define INIT_T_CONSTANTS(t_val) do { \
    int idx = t_val - 2; \
    CHECKCUDAERR(cudaMalloc(&h_GPU_C[idx], sizeof(PoseidonBN128Constants::C##t_val))); \
    CHECKCUDAERR(cudaMemcpy(h_GPU_C[idx], PoseidonBN128Constants::C##t_val, sizeof(PoseidonBN128Constants::C##t_val), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C_ptr, &h_GPU_C[idx], sizeof(FrElementGPU*), idx * sizeof(FrElementGPU*))); \
    CHECKCUDAERR(cudaMalloc(&h_GPU_M[idx], sizeof(PoseidonBN128Constants::M##t_val))); \
    CHECKCUDAERR(cudaMemcpy(h_GPU_M[idx], PoseidonBN128Constants::M##t_val, sizeof(PoseidonBN128Constants::M##t_val), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(GPU_M_ptr, &h_GPU_M[idx], sizeof(FrElementGPU*), idx * sizeof(FrElementGPU*))); \
    CHECKCUDAERR(cudaMalloc(&h_GPU_P[idx], sizeof(PoseidonBN128Constants::P##t_val))); \
    CHECKCUDAERR(cudaMemcpy(h_GPU_P[idx], PoseidonBN128Constants::P##t_val, sizeof(PoseidonBN128Constants::P##t_val), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(GPU_P_ptr, &h_GPU_P[idx], sizeof(FrElementGPU*), idx * sizeof(FrElementGPU*))); \
    CHECKCUDAERR(cudaMalloc(&h_GPU_S[idx], sizeof(PoseidonBN128Constants::S##t_val))); \
    CHECKCUDAERR(cudaMemcpy(h_GPU_S[idx], PoseidonBN128Constants::S##t_val, sizeof(PoseidonBN128Constants::S##t_val), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(GPU_S_ptr, &h_GPU_S[idx], sizeof(FrElementGPU*), idx * sizeof(FrElementGPU*))); \
} while(0)

// Initialize GPU constants - uploads all t values (2-17)
void PoseidonBN128GPU::initGPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids) {
    if (constants_initialized) return;

    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));

    for(uint32_t i = 0; i < num_gpu_ids; i++)
    {
        CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));

        INIT_T_CONSTANTS(2);
        INIT_T_CONSTANTS(3);
        INIT_T_CONSTANTS(4);
        INIT_T_CONSTANTS(5);
        INIT_T_CONSTANTS(6);
        INIT_T_CONSTANTS(7);
        INIT_T_CONSTANTS(8);
        INIT_T_CONSTANTS(9);
        INIT_T_CONSTANTS(10);
        INIT_T_CONSTANTS(11);
        INIT_T_CONSTANTS(12);
        INIT_T_CONSTANTS(13);
        INIT_T_CONSTANTS(14);
        INIT_T_CONSTANTS(15);
        INIT_T_CONSTANTS(16);
        INIT_T_CONSTANTS(17);
    }
    
    CHECKCUDAERR(cudaSetDevice(deviceId));
    constants_initialized = true;
}

// Free GPU memory for all constants
void PoseidonBN128GPU::freeGPUConstants() {
    if (!constants_initialized) return;

    FrElementGPU* null_ptr = nullptr;
    for (int idx = 0; idx < 16; idx++) {
        if (h_GPU_C[idx]) { cudaFree(h_GPU_C[idx]); h_GPU_C[idx] = nullptr; }
        if (h_GPU_M[idx]) { cudaFree(h_GPU_M[idx]); h_GPU_M[idx] = nullptr; }
        if (h_GPU_P[idx]) { cudaFree(h_GPU_P[idx]); h_GPU_P[idx] = nullptr; }
        if (h_GPU_S[idx]) { cudaFree(h_GPU_S[idx]); h_GPU_S[idx] = nullptr; }
        
        cudaMemcpyToSymbol(GPU_C_ptr, &null_ptr, sizeof(FrElementGPU*), idx * sizeof(FrElementGPU*));
        cudaMemcpyToSymbol(GPU_M_ptr, &null_ptr, sizeof(FrElementGPU*), idx * sizeof(FrElementGPU*));
        cudaMemcpyToSymbol(GPU_P_ptr, &null_ptr, sizeof(FrElementGPU*), idx * sizeof(FrElementGPU*));
        cudaMemcpyToSymbol(GPU_S_ptr, &null_ptr, sizeof(FrElementGPU*), idx * sizeof(FrElementGPU*));
    }
    constants_initialized = false;
}

// =============================================================================
// Linear Hash GPU
// =============================================================================

// Goldilocks prime: p = 2^64 - 2^32 + 1
__device__ __constant__ uint64_t GOLDILOCKS_MOD = 0xFFFFFFFF00000001ULL;

// Reduce a Goldilocks value from partially reduced form to canonical form
// In partially reduced form, values can be in range [0, 2*MOD)
// This reduces them to canonical form [0, MOD)
__device__ __forceinline__ uint64_t reduce_gl64(uint64_t val) {
    return (val >= GOLDILOCKS_MOD) ? (val - GOLDILOCKS_MOD) : val;
}

// Load 3 Goldilocks elements into a BN128 Fr element (in registers)
// Supports both row-major and tiled layouts via template parameter
template<bool TILED>
__device__ __forceinline__ void poseidon_bn128_load_fr_from_gl(
    BN128GPUScalarField::Element &elem,
    const uint64_t *input,
    uint32_t row,
    uint64_t base_col,
    uint64_t num_rows,
    uint64_t num_cols
) {
    uint32_t* limbs = (uint32_t*)&elem.v;
    limbs[0] = limbs[1] = limbs[2] = limbs[3] = 0;
    limbs[4] = limbs[5] = limbs[6] = limbs[7] = 0;
    
    #pragma unroll
    for (uint32_t k = 0; k < 3; k++) {
        uint64_t col = base_col + k;
        if (col < num_cols) {
            uint64_t idx;
            if constexpr (TILED) {
                idx = getBufferOffset(row, col, num_rows, num_cols);
            } else {
                idx = row * num_cols + col;
            }
            // Load and reduce from partially reduced form to canonical form
            uint64_t gl_val = reduce_gl64(input[idx]);
            limbs[k * 2] = (uint32_t)gl_val;
            limbs[k * 2 + 1] = (uint32_t)(gl_val >> 32);
        }
    }
}

// Perform the hash loop for linear hash
// Supports both row-major (TILED=false) and tiled (TILED=true) layouts
template<bool TILED>
__device__ void poseidon_bn128_hash_loop(
    BN128GPUScalarField::Element &result,
    const uint64_t *__restrict__ input,
    uint64_t num_cols,
    uint64_t num_rows,
    int t,
    bool custom
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Number of Fr elements needed to pack all Goldilocks columns
    // Each Fr element holds 3 Goldilocks field elements
    uint64_t nElementsFr = (num_cols + 2) / 3;
    
    // Rate is t-1 (capacity is 1, stored in position 0)
    int rate = t - 1;
    
    // State array in registers - capacity (position 0) starts as zero
    BN128GPUScalarField::Element state[18];
    state[0] = BN128GPUScalarField::zero();
    
    // Process input in chunks of (t-1) Fr elements
    uint64_t pending = nElementsFr;
    uint64_t fr_offset = 0;
    
    while (pending > 0) {
        uint32_t batch = (pending >= rate) ? rate : pending;
        
        // Determine actual_t for this iteration
        int actual_t = t;
        if (pending < rate && !custom) {
            actual_t = batch + 1;
        }
        
        // Load batch Fr elements directly into state registers (positions 1 to batch)
        // and convert to Montgomery form
        for (uint32_t fr_idx = 0; fr_idx < batch; fr_idx++) {
            uint64_t base_col = (fr_offset + fr_idx) * 3;
            poseidon_bn128_load_fr_from_gl<TILED>(state[fr_idx + 1], input, row, base_col, num_rows, num_cols);
            BN128GPUScalarField::toMontgomery(state[fr_idx + 1]);
        }
        
        for (uint32_t i = batch + 1; i < actual_t; i++) {
            state[i] = BN128GPUScalarField::zero();
        }
        
        // Perform hash
        PoseidonBN128GPU poseidon;
        const FrElementGPU *C_actual = GPU_C_ptr[actual_t - 2];
        const FrElementGPU *M_actual = GPU_M_ptr[actual_t - 2];
        const FrElementGPU *P_actual = GPU_P_ptr[actual_t - 2];
        const FrElementGPU *S_actual = GPU_S_ptr[actual_t - 2];
        const int nRoundsP_actual = N_ROUNDS_P_POSEIDON[actual_t - 2];
        
        poseidon.hash_(state, actual_t, C_actual, M_actual, P_actual, S_actual, nRoundsP_actual);
        
        // state[0] now contains the hash result, which becomes capacity for next iteration
        pending -= batch;
        fr_offset += batch;
    }
    
    // Return the final result
    result = state[0];
}

template<bool TILED>
__global__ void linearHashGPUBN128(
    FrElementGPU *__restrict__ output,
    uint64_t *__restrict__ input,
    uint64_t num_cols,
    uint64_t num_rows,
    int t,
    bool custom
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    
    BN128GPUScalarField::Element result;
    poseidon_bn128_hash_loop<TILED>(result, input, num_cols, num_rows, t, custom);
    output[row] = result;
}

// Linear hash for traces stored in row-major layout
void PoseidonBN128GPU::linearHash(FrElement *d_output, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int t, bool custom, cudaStream_t stream) {

    int threadsPerBlock = 64;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    
    linearHashGPUBN128<false><<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_output, d_input, num_cols, num_rows, t, custom
    );
}

// Linear hash for traces stored in tiled layout
void PoseidonBN128GPU::linearHashTiles(FrElement *d_output, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int t, bool custom, cudaStream_t stream) {
    
    int threadsPerBlock = 64;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    
    linearHashGPUBN128<true><<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_output, d_input, num_cols, num_rows, t, custom
    );
}

// =============================================================================
// Merkle Tree GPU - Hash kernel for tree building
// =============================================================================

// Kernel to hash groups of Fr elements for Merkle tree building
// Each thread hashes (t-1) inputs into 1 output
// Input layout: cursor[nextIndex + tid * (t-1)] for tid in [0, nextN)
// Output layout: cursor[nextIndex + pending + tid] for tid in [0, nextN)
__global__ void hashTreeKernelBN128(
    FrElementGPU *cursor,
    uint64_t nextN,
    uint64_t nextIndex,
    uint64_t pending,
    int t
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN) return;
    
    int rate = t - 1;
    
    // Build state: state[0] = 0 (capacity), state[1..t-1] = inputs
    BN128GPUScalarField::Element state[18];
    state[0] = BN128GPUScalarField::zero();
    
    // Load input elements
    FrElementGPU *input_ptr = &cursor[nextIndex + tid * rate];
    for (int i = 0; i < rate; i++) {
        state[i + 1] = input_ptr[i];
    }
    
    // Hash
    PoseidonBN128GPU poseidon;
    const FrElementGPU *C = GPU_C_ptr[t - 2];
    const FrElementGPU *M = GPU_M_ptr[t - 2];
    const FrElementGPU *P = GPU_P_ptr[t - 2];
    const FrElementGPU *S = GPU_S_ptr[t - 2];
    const int nRoundsP = N_ROUNDS_P_POSEIDON[t - 2];
    
    poseidon.hash_(state, t, C, M, P, S, nRoundsP);
    
    // Write output
    cursor[nextIndex + pending + tid] = state[0];
}

template<bool TILED>
void merkletreeGPUBN128(
    FrElementGPU *d_tree,
    uint64_t *d_input,
    uint64_t num_cols,
    uint64_t num_rows,
    uint64_t arity,
    bool custom,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    
    int t = arity + 1;  // arity inputs + 1 capacity
    int threadsPerBlock = 64;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    
    // Step 1: Compute leaf hashes
    linearHashGPUBN128<TILED><<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_tree, d_input, num_cols, num_rows, t, custom
    );
    CHECKCUDAERR(cudaGetLastError());
    
    // Step 2: Build the Merkle tree
    uint64_t pending = num_rows;  
    uint64_t nextN = (pending + arity - 1) / arity;
    uint64_t nextIndex = 0;  // Start index of current level in d_tree
    
    while (pending > 1) {
        // Pad with zeros if needed
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0) {
            CHECKCUDAERR(cudaMemsetAsync(
                (uint8_t*)(d_tree + nextIndex + pending), 
                0, 
                extraZeros * sizeof(FrElementGPU), 
                stream
            ));
        }
        
        // Hash this level
        numBlocks = (nextN + threadsPerBlock - 1) / threadsPerBlock;
        hashTreeKernelBN128<<<numBlocks, threadsPerBlock, 0, stream>>>(
            d_tree, nextN, nextIndex, pending + extraZeros, t
        );
        CHECKCUDAERR(cudaGetLastError());
        
        // Move to next level
        nextIndex += pending + extraZeros;
        pending = nextN;
        nextN = (pending + arity - 1) / arity;
    }
}

// Merkle tree for row-major layout
void PoseidonBN128GPU::merkletree(FrElement *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, bool custom, cudaStream_t stream) {
    merkletreeGPUBN128<false>(d_tree, d_input, num_cols, num_rows, arity, custom, stream);
}

// Merkle tree for tiled layout
void PoseidonBN128GPU::merkletreeTiles(FrElement *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, bool custom, cudaStream_t stream) {
    merkletreeGPUBN128<true>(d_tree, d_input, num_cols, num_rows, arity, custom, stream);
}

#undef INIT_T_CONSTANTS
