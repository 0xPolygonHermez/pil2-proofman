#include "cuda_utils.cuh"
#include "poseidon_bn128.cuh"
#include "poseidon_bn128_constants.hpp"  // Shared CPU/GPU constants (binary compatible)
#include <cuda_runtime.h>

typedef PoseidonBN128GPU::FrElement FrElementGPU;

// All constants in global memory (constant memory is limited to 64KB)
// Device-side pointers
__device__ FrElementGPU *GPU_C2_ptr = nullptr;
__device__ FrElementGPU *GPU_C3_ptr = nullptr;
__device__ FrElementGPU *GPU_C4_ptr = nullptr;
__device__ FrElementGPU *GPU_C5_ptr = nullptr;
__device__ FrElementGPU *GPU_C6_ptr = nullptr;
__device__ FrElementGPU *GPU_C7_ptr = nullptr;
__device__ FrElementGPU *GPU_C8_ptr = nullptr;
__device__ FrElementGPU *GPU_C9_ptr = nullptr;
__device__ FrElementGPU *GPU_C10_ptr = nullptr;
__device__ FrElementGPU *GPU_C11_ptr = nullptr;
__device__ FrElementGPU *GPU_C12_ptr = nullptr;
__device__ FrElementGPU *GPU_C13_ptr = nullptr;
__device__ FrElementGPU *GPU_C14_ptr = nullptr;
__device__ FrElementGPU *GPU_C15_ptr = nullptr;
__device__ FrElementGPU *GPU_C16_ptr = nullptr;
__device__ FrElementGPU *GPU_C17_ptr = nullptr;

__device__ FrElementGPU *GPU_M2_ptr = nullptr;
__device__ FrElementGPU *GPU_M3_ptr = nullptr;
__device__ FrElementGPU *GPU_M4_ptr = nullptr;
__device__ FrElementGPU *GPU_M5_ptr = nullptr;
__device__ FrElementGPU *GPU_M6_ptr = nullptr;
__device__ FrElementGPU *GPU_M7_ptr = nullptr;
__device__ FrElementGPU *GPU_M8_ptr = nullptr;
__device__ FrElementGPU *GPU_M9_ptr = nullptr;
__device__ FrElementGPU *GPU_M10_ptr = nullptr;
__device__ FrElementGPU *GPU_M11_ptr = nullptr;
__device__ FrElementGPU *GPU_M12_ptr = nullptr;
__device__ FrElementGPU *GPU_M13_ptr = nullptr;
__device__ FrElementGPU *GPU_M14_ptr = nullptr;
__device__ FrElementGPU *GPU_M15_ptr = nullptr;
__device__ FrElementGPU *GPU_M16_ptr = nullptr;
__device__ FrElementGPU *GPU_M17_ptr = nullptr;

__device__ FrElementGPU *GPU_P2_ptr = nullptr;
__device__ FrElementGPU *GPU_P3_ptr = nullptr;
__device__ FrElementGPU *GPU_P4_ptr = nullptr;
__device__ FrElementGPU *GPU_P5_ptr = nullptr;
__device__ FrElementGPU *GPU_P6_ptr = nullptr;
__device__ FrElementGPU *GPU_P7_ptr = nullptr;
__device__ FrElementGPU *GPU_P8_ptr = nullptr;
__device__ FrElementGPU *GPU_P9_ptr = nullptr;
__device__ FrElementGPU *GPU_P10_ptr = nullptr;
__device__ FrElementGPU *GPU_P11_ptr = nullptr;
__device__ FrElementGPU *GPU_P12_ptr = nullptr;
__device__ FrElementGPU *GPU_P13_ptr = nullptr;
__device__ FrElementGPU *GPU_P14_ptr = nullptr;
__device__ FrElementGPU *GPU_P15_ptr = nullptr;
__device__ FrElementGPU *GPU_P16_ptr = nullptr;
__device__ FrElementGPU *GPU_P17_ptr = nullptr;

__device__ FrElementGPU *GPU_S2_ptr = nullptr;
__device__ FrElementGPU *GPU_S3_ptr = nullptr;
__device__ FrElementGPU *GPU_S4_ptr = nullptr;
__device__ FrElementGPU *GPU_S5_ptr = nullptr;
__device__ FrElementGPU *GPU_S6_ptr = nullptr;
__device__ FrElementGPU *GPU_S7_ptr = nullptr;
__device__ FrElementGPU *GPU_S8_ptr = nullptr;
__device__ FrElementGPU *GPU_S9_ptr = nullptr;
__device__ FrElementGPU *GPU_S10_ptr = nullptr;
__device__ FrElementGPU *GPU_S11_ptr = nullptr;
__device__ FrElementGPU *GPU_S12_ptr = nullptr;
__device__ FrElementGPU *GPU_S13_ptr = nullptr;
__device__ FrElementGPU *GPU_S14_ptr = nullptr;
__device__ FrElementGPU *GPU_S15_ptr = nullptr;
__device__ FrElementGPU *GPU_S16_ptr = nullptr;
__device__ FrElementGPU *GPU_S17_ptr = nullptr;

// Host-side pointers to device memory
static FrElementGPU *h_GPU_C2 = nullptr, *h_GPU_C3 = nullptr, *h_GPU_C4 = nullptr, *h_GPU_C5 = nullptr;
static FrElementGPU *h_GPU_C6 = nullptr, *h_GPU_C7 = nullptr, *h_GPU_C8 = nullptr, *h_GPU_C9 = nullptr;
static FrElementGPU *h_GPU_C10 = nullptr, *h_GPU_C11 = nullptr, *h_GPU_C12 = nullptr, *h_GPU_C13 = nullptr;
static FrElementGPU *h_GPU_C14 = nullptr, *h_GPU_C15 = nullptr, *h_GPU_C16 = nullptr, *h_GPU_C17 = nullptr;

static FrElementGPU *h_GPU_M2 = nullptr, *h_GPU_M3 = nullptr, *h_GPU_M4 = nullptr, *h_GPU_M5 = nullptr;
static FrElementGPU *h_GPU_M6 = nullptr, *h_GPU_M7 = nullptr, *h_GPU_M8 = nullptr, *h_GPU_M9 = nullptr;
static FrElementGPU *h_GPU_M10 = nullptr, *h_GPU_M11 = nullptr, *h_GPU_M12 = nullptr, *h_GPU_M13 = nullptr;
static FrElementGPU *h_GPU_M14 = nullptr, *h_GPU_M15 = nullptr, *h_GPU_M16 = nullptr, *h_GPU_M17 = nullptr;

static FrElementGPU *h_GPU_P2 = nullptr, *h_GPU_P3 = nullptr, *h_GPU_P4 = nullptr, *h_GPU_P5 = nullptr;
static FrElementGPU *h_GPU_P6 = nullptr, *h_GPU_P7 = nullptr, *h_GPU_P8 = nullptr, *h_GPU_P9 = nullptr;
static FrElementGPU *h_GPU_P10 = nullptr, *h_GPU_P11 = nullptr, *h_GPU_P12 = nullptr, *h_GPU_P13 = nullptr;
static FrElementGPU *h_GPU_P14 = nullptr, *h_GPU_P15 = nullptr, *h_GPU_P16 = nullptr, *h_GPU_P17 = nullptr;

static FrElementGPU *h_GPU_S2 = nullptr, *h_GPU_S3 = nullptr, *h_GPU_S4 = nullptr, *h_GPU_S5 = nullptr;
static FrElementGPU *h_GPU_S6 = nullptr, *h_GPU_S7 = nullptr, *h_GPU_S8 = nullptr, *h_GPU_S9 = nullptr;
static FrElementGPU *h_GPU_S10 = nullptr, *h_GPU_S11 = nullptr, *h_GPU_S12 = nullptr, *h_GPU_S13 = nullptr;
static FrElementGPU *h_GPU_S14 = nullptr, *h_GPU_S15 = nullptr, *h_GPU_S16 = nullptr, *h_GPU_S17 = nullptr;

// Round counts in constant memory (small enough)
#define N_ROUNDS_F 8
__device__ __constant__ int N_ROUNDS_P_POSEIDON[16] = {56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68};

// Helper to get C constants pointer based on t
__device__ __forceinline__ const FrElementGPU* get_C_poseidon(int t) {
    switch(t) {
        case 2:  return GPU_C2_ptr;
        case 3:  return GPU_C3_ptr;
        case 4:  return GPU_C4_ptr;
        case 5:  return GPU_C5_ptr;
        case 6:  return GPU_C6_ptr;
        case 7:  return GPU_C7_ptr;
        case 8:  return GPU_C8_ptr;
        case 9:  return GPU_C9_ptr;
        case 10: return GPU_C10_ptr;
        case 11: return GPU_C11_ptr;
        case 12: return GPU_C12_ptr;
        case 13: return GPU_C13_ptr;
        case 14: return GPU_C14_ptr;
        case 15: return GPU_C15_ptr;
        case 16: return GPU_C16_ptr;
        case 17: return GPU_C17_ptr;
        default: return nullptr;
    }
}

// Helper to get M constants pointer based on t
__device__ __forceinline__ const FrElementGPU* get_M_poseidon(int t) {
    switch(t) {
        case 2:  return GPU_M2_ptr;
        case 3:  return GPU_M3_ptr;
        case 4:  return GPU_M4_ptr;
        case 5:  return GPU_M5_ptr;
        case 6:  return GPU_M6_ptr;
        case 7:  return GPU_M7_ptr;
        case 8:  return GPU_M8_ptr;
        case 9:  return GPU_M9_ptr;
        case 10: return GPU_M10_ptr;
        case 11: return GPU_M11_ptr;
        case 12: return GPU_M12_ptr;
        case 13: return GPU_M13_ptr;
        case 14: return GPU_M14_ptr;
        case 15: return GPU_M15_ptr;
        case 16: return GPU_M16_ptr;
        case 17: return GPU_M17_ptr;
        default: return nullptr;
    }
}

// Helper to get P constants pointer based on t
__device__ __forceinline__ const FrElementGPU* get_P_poseidon(int t) {
    switch(t) {
        case 2:  return GPU_P2_ptr;
        case 3:  return GPU_P3_ptr;
        case 4:  return GPU_P4_ptr;
        case 5:  return GPU_P5_ptr;
        case 6:  return GPU_P6_ptr;
        case 7:  return GPU_P7_ptr;
        case 8:  return GPU_P8_ptr;
        case 9:  return GPU_P9_ptr;
        case 10: return GPU_P10_ptr;
        case 11: return GPU_P11_ptr;
        case 12: return GPU_P12_ptr;
        case 13: return GPU_P13_ptr;
        case 14: return GPU_P14_ptr;
        case 15: return GPU_P15_ptr;
        case 16: return GPU_P16_ptr;
        case 17: return GPU_P17_ptr;
        default: return nullptr;
    }
}

// Helper to get S constants pointer based on t (from global memory)
__device__ __forceinline__ const FrElementGPU* get_S_poseidon(int t) {
    switch(t) {
        case 2:  return GPU_S2_ptr;
        case 3:  return GPU_S3_ptr;
        case 4:  return GPU_S4_ptr;
        case 5:  return GPU_S5_ptr;
        case 6:  return GPU_S6_ptr;
        case 7:  return GPU_S7_ptr;
        case 8:  return GPU_S8_ptr;
        case 9:  return GPU_S9_ptr;
        case 10: return GPU_S10_ptr;
        case 11: return GPU_S11_ptr;
        case 12: return GPU_S12_ptr;
        case 13: return GPU_S13_ptr;
        case 14: return GPU_S14_ptr;
        case 15: return GPU_S15_ptr;
        case 16: return GPU_S16_ptr;
        case 17: return GPU_S17_ptr;
        default: return nullptr;
    }
}

__global__ void poseidon_hash_kernel(FrElementGPU *state, int t) {
    PoseidonBN128GPU poseidon;
    
    const FrElementGPU *C = get_C_poseidon(t);
    const FrElementGPU *S = get_S_poseidon(t);
    const FrElementGPU *M = get_M_poseidon(t);
    const FrElementGPU *P = get_P_poseidon(t);
    const int nRoundsP = N_ROUNDS_P_POSEIDON[t - 2];
    
    // Temporary buffer for mix operation
    FrElementGPU tmp[17];
    
    poseidon.ark(state, C, t, 0);
    
    for (int r = 0; r < N_ROUNDS_F / 2 - 1; r++)
    {
        poseidon.sbox(state, C, t, (r + 1) * t);
        poseidon.mix(state, tmp, M, t);
    }
    
    poseidon.sbox(state, C, t, (N_ROUNDS_F / 2) * t);
    poseidon.mix(state, tmp, P, t);
    
    for (int r = 0; r < nRoundsP; r++)
    {
        poseidon.exp5(state[0]);
        BN128GPUScalarField::add(state[0], state[0], C[(N_ROUNDS_F / 2 + 1) * t + r]);

        FrElementGPU s0 = BN128GPUScalarField::zero();
        FrElementGPU accumulator1;
        FrElementGPU accumulator2;
        
        for (int j = 0; j < t; j++)
        {
            accumulator1 = S[(t * 2 - 1) * r + j];
            BN128GPUScalarField::mul(accumulator1, accumulator1, state[j]);
            BN128GPUScalarField::add(s0, s0, accumulator1);
            if (j > 0)
            {
                accumulator2 = S[(t * 2 - 1) * r + t + j - 1];
                BN128GPUScalarField::mul(accumulator2, state[0], accumulator2);
                BN128GPUScalarField::add(state[j], state[j], accumulator2);
            }
        }
        state[0] = s0;
    }
    
    for (int r = 0; r < N_ROUNDS_F / 2 - 1; r++)
    {
        poseidon.sbox(state, C, t, (N_ROUNDS_F / 2 + 1) * t + nRoundsP + r * t);
        poseidon.mix(state, tmp, M, t);
    }
    
    for (int i = 0; i < t; i++)
    {
        poseidon.exp5(state[i]);
    }
    poseidon.mix(state, tmp, M, t);
}

void PoseidonBN128GPU::hash(FrElement* d_state, int t) {
    poseidon_hash_kernel<<<1, 1>>>(d_state, t);
}

// Macro to allocate, copy, and set device pointer
#define INIT_GPU_ARRAY(name, hostPtr) do { \
    CHECKCUDAERR(cudaMalloc(&h_GPU_##name, sizeof(PoseidonBN128Constants::name))); \
    CHECKCUDAERR(cudaMemcpy(h_GPU_##name, PoseidonBN128Constants::name, sizeof(PoseidonBN128Constants::name), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(GPU_##name##_ptr, &h_GPU_##name, sizeof(FrElementGPU*))); \
} while(0)

// Initialize GPU constants by copying from host
void PoseidonBN128GPU::initGPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids) {
    static bool initialized = false;
    if (initialized) return;
    
    initialized = true;
    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));

    for(uint32_t i = 0; i < num_gpu_ids; i++)
    {
        CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));

        // Copy C constants to global memory
        INIT_GPU_ARRAY(C2, PoseidonBN128Constants::C2);
        INIT_GPU_ARRAY(C3, PoseidonBN128Constants::C3);
        INIT_GPU_ARRAY(C4, PoseidonBN128Constants::C4);
        INIT_GPU_ARRAY(C5, PoseidonBN128Constants::C5);
        INIT_GPU_ARRAY(C6, PoseidonBN128Constants::C6);
        INIT_GPU_ARRAY(C7, PoseidonBN128Constants::C7);
        INIT_GPU_ARRAY(C8, PoseidonBN128Constants::C8);
        INIT_GPU_ARRAY(C9, PoseidonBN128Constants::C9);
        INIT_GPU_ARRAY(C10, PoseidonBN128Constants::C10);
        INIT_GPU_ARRAY(C11, PoseidonBN128Constants::C11);
        INIT_GPU_ARRAY(C12, PoseidonBN128Constants::C12);
        INIT_GPU_ARRAY(C13, PoseidonBN128Constants::C13);
        INIT_GPU_ARRAY(C14, PoseidonBN128Constants::C14);
        INIT_GPU_ARRAY(C15, PoseidonBN128Constants::C15);
        INIT_GPU_ARRAY(C16, PoseidonBN128Constants::C16);
        INIT_GPU_ARRAY(C17, PoseidonBN128Constants::C17);

        // Copy M constants to global memory
        INIT_GPU_ARRAY(M2, PoseidonBN128Constants::M2);
        INIT_GPU_ARRAY(M3, PoseidonBN128Constants::M3);
        INIT_GPU_ARRAY(M4, PoseidonBN128Constants::M4);
        INIT_GPU_ARRAY(M5, PoseidonBN128Constants::M5);
        INIT_GPU_ARRAY(M6, PoseidonBN128Constants::M6);
        INIT_GPU_ARRAY(M7, PoseidonBN128Constants::M7);
        INIT_GPU_ARRAY(M8, PoseidonBN128Constants::M8);
        INIT_GPU_ARRAY(M9, PoseidonBN128Constants::M9);
        INIT_GPU_ARRAY(M10, PoseidonBN128Constants::M10);
        INIT_GPU_ARRAY(M11, PoseidonBN128Constants::M11);
        INIT_GPU_ARRAY(M12, PoseidonBN128Constants::M12);
        INIT_GPU_ARRAY(M13, PoseidonBN128Constants::M13);
        INIT_GPU_ARRAY(M14, PoseidonBN128Constants::M14);
        INIT_GPU_ARRAY(M15, PoseidonBN128Constants::M15);
        INIT_GPU_ARRAY(M16, PoseidonBN128Constants::M16);
        INIT_GPU_ARRAY(M17, PoseidonBN128Constants::M17);

        // Copy P constants to global memory
        INIT_GPU_ARRAY(P2, PoseidonBN128Constants::P2);
        INIT_GPU_ARRAY(P3, PoseidonBN128Constants::P3);
        INIT_GPU_ARRAY(P4, PoseidonBN128Constants::P4);
        INIT_GPU_ARRAY(P5, PoseidonBN128Constants::P5);
        INIT_GPU_ARRAY(P6, PoseidonBN128Constants::P6);
        INIT_GPU_ARRAY(P7, PoseidonBN128Constants::P7);
        INIT_GPU_ARRAY(P8, PoseidonBN128Constants::P8);
        INIT_GPU_ARRAY(P9, PoseidonBN128Constants::P9);
        INIT_GPU_ARRAY(P10, PoseidonBN128Constants::P10);
        INIT_GPU_ARRAY(P11, PoseidonBN128Constants::P11);
        INIT_GPU_ARRAY(P12, PoseidonBN128Constants::P12);
        INIT_GPU_ARRAY(P13, PoseidonBN128Constants::P13);
        INIT_GPU_ARRAY(P14, PoseidonBN128Constants::P14);
        INIT_GPU_ARRAY(P15, PoseidonBN128Constants::P15);
        INIT_GPU_ARRAY(P16, PoseidonBN128Constants::P16);
        INIT_GPU_ARRAY(P17, PoseidonBN128Constants::P17);

        // Copy S constants to global memory
        INIT_GPU_ARRAY(S2, PoseidonBN128Constants::S2);
        INIT_GPU_ARRAY(S3, PoseidonBN128Constants::S3);
        INIT_GPU_ARRAY(S4, PoseidonBN128Constants::S4);
        INIT_GPU_ARRAY(S5, PoseidonBN128Constants::S5);
        INIT_GPU_ARRAY(S6, PoseidonBN128Constants::S6);
        INIT_GPU_ARRAY(S7, PoseidonBN128Constants::S7);
        INIT_GPU_ARRAY(S8, PoseidonBN128Constants::S8);
        INIT_GPU_ARRAY(S9, PoseidonBN128Constants::S9);
        INIT_GPU_ARRAY(S10, PoseidonBN128Constants::S10);
        INIT_GPU_ARRAY(S11, PoseidonBN128Constants::S11);
        INIT_GPU_ARRAY(S12, PoseidonBN128Constants::S12);
        INIT_GPU_ARRAY(S13, PoseidonBN128Constants::S13);
        INIT_GPU_ARRAY(S14, PoseidonBN128Constants::S14);
        INIT_GPU_ARRAY(S15, PoseidonBN128Constants::S15);
        INIT_GPU_ARRAY(S16, PoseidonBN128Constants::S16);
        INIT_GPU_ARRAY(S17, PoseidonBN128Constants::S17);
    }
    
    cudaSetDevice(deviceId);
}

#undef INIT_GPU_ARRAY
