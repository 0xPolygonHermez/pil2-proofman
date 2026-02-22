#include "cuda_utils.cuh"
#include "poseidon2_bn128.cuh"
#include "poseidon2_bn128_constants.hpp"  // Shared CPU/GPU constants (binary compatible)
#include "grinding_constants.hpp"
#include <cuda_runtime.h>

typedef Poseidon2BN128GPU::FrElement FrElementGPU;

// Device constant memory - uninitialized, will be copied at runtime
__device__ __constant__ FrElementGPU GPU_C2[72];
__device__ __constant__ FrElementGPU GPU_D2[2];
__device__ __constant__ FrElementGPU GPU_C3[80];
__device__ __constant__ FrElementGPU GPU_D3[3];
__device__ __constant__ FrElementGPU GPU_C4[88];
__device__ __constant__ FrElementGPU GPU_D4[4];
__device__ __constant__ FrElementGPU GPU_C8[121];
__device__ __constant__ FrElementGPU GPU_D8[8];
__device__ __constant__ FrElementGPU GPU_C12[153];
__device__ __constant__ FrElementGPU GPU_D12[12];
__device__ __constant__ FrElementGPU GPU_C16[185];
__device__ __constant__ FrElementGPU GPU_D16[16];

// Round counts
__device__ __constant__ int N_ROUNDS_F = 8;
__device__ __constant__ int N_ROUNDS_P[6] = {56, 56, 56, 57, 57, 57};

// Track if constants have been initialized
static bool constants_initialized = false;

__device__ __forceinline__ const FrElementGPU* get_C(int t) {
    switch(t) {
        case 2:  return GPU_C2;
        case 3:  return GPU_C3;
        case 4:  return GPU_C4;
        case 8:  return GPU_C8;
        case 12: return GPU_C12;
        case 16: return GPU_C16;
        default: return nullptr;
    }
}

__device__ __forceinline__ const FrElementGPU* get_D(int t) {
    switch(t) {
        case 2:  return GPU_D2;
        case 3:  return GPU_D3;
        case 4:  return GPU_D4;
        case 8:  return GPU_D8;
        case 12: return GPU_D12;
        case 16: return GPU_D16;
        default: return nullptr;
    }
}


__device__ __forceinline__ int get_nRoundsP_pos(int t) {
    return t <= 4 ? t - 2 : t / 4 + 1;
}

__global__ void poseidon2_hash_kernel(FrElementGPU *state, int t) {
    Poseidon2BN128GPU poseidon;
    const FrElementGPU *C = get_C(t);
    const FrElementGPU *D = get_D(t);
    const int nRoundsP = N_ROUNDS_P[get_nRoundsP_pos(t)];

    poseidon.hash_(state, t, C, D, nRoundsP);
}

void Poseidon2BN128GPU::hash(FrElement* d_state, int t) {
    poseidon2_hash_kernel<<<1, 1>>>(d_state, t);
}

// Initialize GPU constants - copies all constants to constant memory
void Poseidon2BN128GPU::initGPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids) {
    if (constants_initialized) return;

    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));

    for(uint32_t i = 0; i < num_gpu_ids; i++)
    {
        CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));

        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C2, Poseidon2BN128Constants::C2, sizeof(Poseidon2BN128Constants::C2), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D2, Poseidon2BN128Constants::D2, sizeof(Poseidon2BN128Constants::D2), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C3, Poseidon2BN128Constants::C3, sizeof(Poseidon2BN128Constants::C3), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D3, Poseidon2BN128Constants::D3, sizeof(Poseidon2BN128Constants::D3), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C4, Poseidon2BN128Constants::C4, sizeof(Poseidon2BN128Constants::C4), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D4, Poseidon2BN128Constants::D4, sizeof(Poseidon2BN128Constants::D4), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C8, Poseidon2BN128Constants::C8, sizeof(Poseidon2BN128Constants::C8), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D8, Poseidon2BN128Constants::D8, sizeof(Poseidon2BN128Constants::D8), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C12, Poseidon2BN128Constants::C12, sizeof(Poseidon2BN128Constants::C12), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D12, Poseidon2BN128Constants::D12, sizeof(Poseidon2BN128Constants::D12), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C16, Poseidon2BN128Constants::C16, sizeof(Poseidon2BN128Constants::C16), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D16, Poseidon2BN128Constants::D16, sizeof(Poseidon2BN128Constants::D16), 0, cudaMemcpyHostToDevice));
    }
    
    CHECKCUDAERR(cudaSetDevice(deviceId));
    constants_initialized = true;
}

// Free GPU constants (constant memory doesn't need deallocation, just reset flag)
void Poseidon2BN128GPU::freeGPUConstants() {
    constants_initialized = false;
}

// =============================================================================
// Parallel Hash
// =============================================================================

// Parallel hash kernel - uses 32 threads (one warp) for parallel operations
__global__ void poseidon2_hash_parallel_kernel(FrElementGPU *d_state, int t) {

    __shared__ uint32_t shared_state_raw[32 * 8];  // 32 elements * 8 limbs
    __shared__ uint32_t tmp_raw[32 * 8];
    FrElementGPU* shared_state = reinterpret_cast<FrElementGPU*>(shared_state_raw);
    FrElementGPU* tmp = reinterpret_cast<FrElementGPU*>(tmp_raw);

    int tid = threadIdx.x;

    if (tid < t) {
        shared_state[tid] = d_state[tid];
    } else if (tid < 32) {
        shared_state[tid] = BN128GPUScalarField::zero();
    }
    __syncthreads();

    // Get constants
    const FrElementGPU *C = get_C(t);
    const FrElementGPU *D = get_D(t);
    const int nRoundsP = N_ROUNDS_P[get_nRoundsP_pos(t)];

    Poseidon2BN128GPU poseidon;
    poseidon.hash_parallel_(shared_state, tmp, t, C, D, nRoundsP);

    if (tid < t) {
        d_state[tid] = shared_state[tid];
    }
}

void Poseidon2BN128GPU::hashParallel(FrElement* d_state, int t) {
    poseidon2_hash_parallel_kernel<<<1, 32>>>(d_state, t);
}

// =============================================================================
// Linear Hash GPU
// =============================================================================


// Perform the hash loop for linear hash
// Supports both row-major (TILED=false) and tiled (TILED=true) layouts
template<bool TILED>
__device__ void poseidon2_bn128_hash_loop(
    BN128GPUScalarField::Element &result,
    const uint64_t *__restrict__ input,
    uint64_t num_cols,
    uint64_t num_rows,
    int t,
    bool custom
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // Each Fr element holds 3 Goldilocks field elements
    uint64_t nElementsFr = (num_cols + 2) / 3;

    int rate = t - 1;

    // State array in registers - capacity (position 0) starts as zero
    BN128GPUScalarField::Element state[16];
    state[0] = BN128GPUScalarField::zero();

    const FrElementGPU *C = get_C(t);
    const FrElementGPU *D = get_D(t);
    const int nRoundsP = N_ROUNDS_P[get_nRoundsP_pos(t)];

    Poseidon2BN128GPU poseidon;

    uint64_t pending = nElementsFr;
    uint64_t fr_offset = 0;

    while (pending > 0) {
        uint32_t batch = (pending >= (uint64_t)rate) ? rate : pending;

        // Load batch Fr elements directly into state registers (positions 1 to batch)
        // and convert to Montgomery form
        for (uint32_t fr_idx = 0; fr_idx < batch; fr_idx++) {
            uint64_t col = (fr_offset + fr_idx) * 3;
            bn128_load_fr_from_gl<TILED>(state[fr_idx + 1], input, row, col, num_rows, num_cols);
            BN128GPUScalarField::toMontgomery(state[fr_idx + 1]);
        }

        // Zero remaining positions
        for (uint32_t i = batch + 1; i < (uint32_t)t; i++) {
            state[i] = BN128GPUScalarField::zero();
        }

        poseidon.hash_(state, t, C, D, nRoundsP);
        pending -= batch;
        fr_offset += batch;
    }

    result = state[0];
}

template<bool TILED>
__global__ void linearHashGPUBN128P2(
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
    poseidon2_bn128_hash_loop<TILED>(result, input, num_cols, num_rows, t, custom);
    output[row] = result;
}

// Linear hash for traces stored in row-major layout
void Poseidon2BN128GPU::linearHash(FrElement *d_output, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int t, bool custom, cudaStream_t stream) {
    int threadsPerBlock = 64;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    linearHashGPUBN128P2<false><<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_output, d_input, num_cols, num_rows, t, custom
    );
}

// Linear hash for traces stored in tiled layout
void Poseidon2BN128GPU::linearHashTiles(FrElement *d_output, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int t, bool custom, cudaStream_t stream) {
    int threadsPerBlock = 64;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    linearHashGPUBN128P2<true><<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_output, d_input, num_cols, num_rows, t, custom
    );
}

// =============================================================================
// Merkle Tree GPU
// =============================================================================

// Hash kernel for tree building (compressor mode)
__global__ void hashTreeKernelBN128P2(
    FrElementGPU *cursor,
    uint64_t nextN,
    uint64_t nextIndex,
    uint64_t inputsN,
    int t
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN) return;

    BN128GPUScalarField::Element state[16]; // we should use template to be caurate here

    // Load input elements directly (no capacity offset)
    FrElementGPU *input_ptr = &cursor[nextIndex + tid * t];
    for (int i = 0; i < t; i++) {
        state[i] = input_ptr[i];
    }

    // Hash
    Poseidon2BN128GPU poseidon;
    const FrElementGPU *C = get_C(t);
    const FrElementGPU *D = get_D(t);
    const int nRoundsP = N_ROUNDS_P[get_nRoundsP_pos(t)];

    poseidon.hash_(state, t, C, D, nRoundsP);

    // Write output
    cursor[nextIndex + inputsN + tid] = state[0];
}

template<bool TILED>
void merkletreeGPUBN128P2(
    FrElementGPU *d_tree,
    uint64_t *d_input,
    uint64_t num_cols,
    uint64_t num_rows,
    uint64_t arity,
    bool custom,
    cudaStream_t stream
) {
    if (num_rows == 0) return;

    int t = arity;  
    int threadsPerBlock = 64;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    linearHashGPUBN128P2<TILED><<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_tree, d_input, num_cols, num_rows, t, custom
    );
    CHECKCUDAERR(cudaGetLastError());

    // Build the Merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + arity - 1) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1) {

        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0) {
            CHECKCUDAERR(cudaMemsetAsync(
                (uint8_t*)(d_tree + nextIndex + pending),
                0,
                extraZeros * sizeof(FrElementGPU),
                stream
            ));
        }

        numBlocks = (nextN + threadsPerBlock - 1) / threadsPerBlock;
        hashTreeKernelBN128P2<<<numBlocks, threadsPerBlock, 0, stream>>>(
            d_tree, nextN, nextIndex, pending + extraZeros, t
        );
        CHECKCUDAERR(cudaGetLastError());

        nextIndex += pending + extraZeros;
        pending = nextN;
        nextN = (pending + arity - 1) / arity;
    }
}

// Merkle tree for row-major layout
void Poseidon2BN128GPU::merkletree(FrElement *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, bool custom, cudaStream_t stream) {
    merkletreeGPUBN128P2<false>(d_tree, d_input, num_cols, num_rows, arity, custom, stream);
}

// Merkle tree for tiled layout
void Poseidon2BN128GPU::merkletreeTiles(FrElement *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, bool custom, cudaStream_t stream) {
    merkletreeGPUBN128P2<true>(d_tree, d_input, num_cols, num_rows, arity, custom, stream);
}

// =============================================================================
// Grinding GPU Implementation
// =============================================================================

// Using common constants from grinding_constants.hpp:
// NONCES_LAUNCH_BITS, NONCES_LAUNCH_BLOCKS, NONCES_LAUNCH_GRID_SIZE

// Shared memory for block-level reduction
extern __shared__ uint64_t grinding_shared_p2[];

// Grinding kernel: searches for nonce where hash(state || nonce).v[0] < level
// Uses t=4 state: [state[0], state[1], state[2], nonce] — no capacity
__global__ void grinding_kernel_bn128_p2(
    uint64_t *d_nonce,
    uint64_t *d_nonceBlock,
    const FrElementGPU *d_state,
    uint32_t n_bits,
    uint64_t hashes_per_thread,
    uint64_t nonces_offset
) {
    // Early exit if nonce already found in previous launch
    if (nonces_offset != 0 && d_nonce[0] != UINT64_MAX) {
        return;
    }

    uint64_t *shared_nonces = grinding_shared_p2;

    // Initialize shared memory and check for previously found nonce
    if (threadIdx.x == 0) {
        shared_nonces[0] = UINT64_MAX;
        if (blockIdx.x == 0) {
            d_nonce[0] = UINT64_MAX;
        }
        if (nonces_offset != 0) {
            for (int i = 0; i < gridDim.x; ++i) {
                if (d_nonceBlock[i] != UINT64_MAX) {
                    shared_nonces[0] = d_nonceBlock[i];
                    if (blockIdx.x == 0) {
                        d_nonce[0] = d_nonceBlock[i];
                    }
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (shared_nonces[0] != UINT64_MAX) {
        return;
    }

    // Calculate starting nonce for this thread
    uint64_t idx = nonces_offset + (blockIdx.x * blockDim.x + threadIdx.x) * hashes_per_thread;
    uint64_t level = 1ULL << (64 - n_bits);
    uint64_t locId = UINT64_MAX;

    // Get constants for t=4
    const int t = 4;
    const FrElementGPU *C = get_C(t);
    const FrElementGPU *D = get_D(t);
    const int nRoundsP = N_ROUNDS_P[get_nRoundsP_pos(t)];

    Poseidon2BN128GPU poseidon;

    for (uint64_t k = 0; k < hashes_per_thread && locId == UINT64_MAX; k++) {
        uint64_t nonce_k = idx + k;

        // Build state: [state[0], state[1], state[2], nonce] — no capacity
        FrElementGPU state[4];
        state[0] = d_state[0];  // Already in Montgomery form
        state[1] = d_state[1];
        state[2] = d_state[2];

        // Create nonce element and convert to Montgomery form
        state[3] = BN128GPUScalarField::zero();
        state[3][0] = (uint32_t)nonce_k;
        state[3][1] = (uint32_t)(nonce_k >> 32);
        BN128GPUScalarField::toMontgomery(state[3]);

        // Perform hash
        poseidon.hash_(state, t, C, D, nRoundsP);

        // Convert result from Montgomery and check
        FrElementGPU result = state[0];
        BN128GPUScalarField::fromMontgomery(result);

        // Check if hash satisfies grinding requirement
        uint64_t hash_low = ((uint64_t)result[1] << 32) | result[0];
        if (hash_low < level) {
            locId = nonce_k;
        }
    }

    // Store result in shared memory for block-level reduction
    shared_nonces[threadIdx.x] = locId;
    __syncthreads();

    // Parallel reduction to find minimum nonce in block
    uint32_t alive = blockDim.x >> 1;
    while (alive > 0) {
        if (threadIdx.x < alive && shared_nonces[threadIdx.x + alive] < shared_nonces[threadIdx.x]) {
            shared_nonces[threadIdx.x] = shared_nonces[threadIdx.x + alive];
        }
        __syncthreads();
        alive >>= 1;
    }

    if (threadIdx.x == 0) {
        d_nonceBlock[blockIdx.x] = shared_nonces[0];
    }
}

void Poseidon2BN128GPU::grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock, const FrElement *d_state, uint32_t n_bits, cudaStream_t stream) {
    // Calculate number of iterations needed for 128-bit security
    // Probability of not finding nonce in totalHashes: (1 - 1/2^n_bits)^totalHashes = 2^(-128)
    const uint64_t security = 128;
    double totalHashesRequired = (double(-double(security))) * log(2.0) / log(1.0 - 1.0 / double(1ULL << n_bits));
    uint64_t log_totalHashesRequired = (uint64_t)ceil(log2(totalHashesRequired));

    uint64_t log_N = NONCES_LAUNCH_BITS;  // 1<<NONCES_LAUNCH_BITS nonces tried per launch
    uint64_t log_launch_iters = 7;  // 128 launch iterations

    uint64_t log_hashesPerThread;
    if (log_totalHashesRequired > log_launch_iters + log_N) {
        log_hashesPerThread = log_totalHashesRequired - log_launch_iters - log_N;
    } else {
        log_hashesPerThread = 0;
    }
    uint64_t hashesPerThread = 1ULL << log_hashesPerThread;

    dim3 blockSize(NONCES_LAUNCH_BLOCKS);
    dim3 gridSize(NONCES_LAUNCH_GRID_SIZE);

    size_t shared_mem_size = blockSize.x * sizeof(uint64_t);
    uint64_t nonces_offset = 0;
    uint64_t nonces_per_iteration = blockSize.x * gridSize.x * hashesPerThread;
    uint64_t launch_iters = 1ULL << log_launch_iters;

    for (uint64_t i = 0; i < launch_iters; ++i) {
        grinding_kernel_bn128_p2<<<gridSize, blockSize, shared_mem_size, stream>>>(
            d_nonce, d_nonceBlock, d_state, n_bits, hashesPerThread, nonces_offset
        );
        nonces_offset += nonces_per_iteration;
    }
}



