#include "cuda_utils.cuh"
#include "poseidon2_bn128.cuh"
#include "poseidon2_bn128_constants.hpp"  // Shared CPU/GPU constants (binary compatible)
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


__global__ void poseidon2_hash_kernel(FrElementGPU *state, int t) {
    Poseidon2BN128GPU poseidon;
    const FrElementGPU *C = get_C(t);
    const FrElementGPU *D = get_D(t);
	uint32_t pos = t<=4 ? t-2 : t/4 + 1;
    const int nRoundsP = N_ROUNDS_P[pos];

    poseidon.matmul_external(&state[0], t);

    for (int r = 0; r < N_ROUNDS_F / 2; r++) {
        poseidon.pow5add(&state[0], &C[r * t], t);
        poseidon.matmul_external(&state[0], t);
    }
    for (int r = 0; r < nRoundsP; r++) {
        BN128GPUScalarField::add(state[0], state[0], C[(N_ROUNDS_F / 2) * t + r]);
        poseidon.pow5(state[0]);
        FrElementGPU sum = BN128GPUScalarField::zero();
        poseidon.add(sum, &state[0], t);
        poseidon.prodadd(&state[0], &D[0], sum, t);
    }
    for (int r = 0; r < N_ROUNDS_F / 2; r++) {
        poseidon.pow5add(&state[0], &C[(N_ROUNDS_F / 2) * t + nRoundsP + r * t], t);
        poseidon.matmul_external(&state[0], t);
    }
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



