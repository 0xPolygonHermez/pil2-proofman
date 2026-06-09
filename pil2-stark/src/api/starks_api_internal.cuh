#ifndef STARKS_API_INTERNAL_CUH
#define STARKS_API_INTERNAL_CUH


#include <cstdint>
#include <cuda_runtime.h>

enum class Layout : uint8_t;  // full definition in poseidon_gpu_common.cuh

void buildMerkleTreeGPU(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                         uint64_t nCols, uint64_t nRows, Layout layout, cudaStream_t stream);
void runGrindingGPU(uint64_t *d_nonce, uint64_t *d_nonceBlock, const uint64_t *d_in,
                    uint32_t n_bits, cudaStream_t stream);

#endif
