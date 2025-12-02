#include "poseidon2_goldilocks.hpp"

// Define GPU constant memory for Poseidon2
// These are declared as extern in poseidon2_goldilocks.cuh
__device__ __constant__ uint64_t GPU_C[SPONGE_WIDTH * N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS];
__device__ __constant__ uint64_t GPU_D[SPONGE_WIDTH];
