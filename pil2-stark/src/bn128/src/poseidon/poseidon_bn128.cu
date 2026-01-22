#include "cuda_utils.cuh"
#include "poseidon_bn128.cuh"
#include "poseidon_bn128_constants.hpp"  // Shared CPU/GPU constants (binary compatible)
#include <cuda_runtime.h>

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

#undef INIT_T_CONSTANTS
