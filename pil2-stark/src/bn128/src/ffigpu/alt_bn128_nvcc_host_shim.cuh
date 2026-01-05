#pragma once
#include <cstdint>

// Compatibility shim for NVCC to access device constants from host code. Just for compilation purposes.

namespace device {
extern __device__ __constant__ __align__(16) uint32_t ALT_BN128_P[8];
extern __device__ __constant__ __align__(16) uint32_t ALT_BN128_RR[8];
extern __device__ __constant__ __align__(16) uint32_t ALT_BN128_one[8];
extern __device__ __constant__ __align__(16) uint32_t ALT_BN128_Px4[8];
extern __device__ __constant__ uint32_t ALT_BN128_M0;

extern __device__ __constant__ __align__(16) uint32_t ALT_BN128_r[8];
extern __device__ __constant__ __align__(16) uint32_t ALT_BN128_rRR[8];
extern __device__ __constant__ __align__(16) uint32_t ALT_BN128_rone[8];
extern __device__ __constant__ __align__(16) uint32_t ALT_BN128_rx4[8];
extern __device__ __constant__ uint32_t ALT_BN128_m0;
}

namespace alt_bn128 {
struct fr_t {
    uint32_t limbs[8];
    __host__ __device__ inline fr_t() {}
};
}
