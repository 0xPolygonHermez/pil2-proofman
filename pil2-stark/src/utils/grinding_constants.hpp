#ifndef GRINDING_CONSTANTS_HPP
#define GRINDING_CONSTANTS_HPP

// Common grinding constants used by GPU implementations
// These are consistent across all grinding implementations:
// - Goldilocks Poseidon2 GPU
// - BN128 Poseidon GPU
// - BN128 Poseidon2 GPU

#define NONCES_LAUNCH_BITS 19
#define NONCES_LAUNCH_BLOCKS 512
#define NONCES_LAUNCH_GRID_SIZE \
    (((1ULL << NONCES_LAUNCH_BITS) + NONCES_LAUNCH_BLOCKS - 1) / NONCES_LAUNCH_BLOCKS)

#endif // 
