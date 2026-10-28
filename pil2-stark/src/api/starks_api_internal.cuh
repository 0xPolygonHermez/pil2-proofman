#ifndef STARKS_API_INTERNAL_CUH
#define STARKS_API_INTERNAL_CUH


#include <cstdint>
#include <cuda_runtime.h>

enum class Layout : uint8_t;  // full definition in poseidon_gpu_common.cuh

void buildMerkleTreeGPU(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                         uint64_t nCols, uint64_t nRows, Layout layout, cudaStream_t stream);
// Same tree without its leaf level (see merkle_noleaves.cuh).
void buildMerkleTreeNoLeavesGPU(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                                uint64_t nCols, uint64_t nRows, Layout layout, cudaStream_t stream);
// Level-0 query siblings of such a tree, rehashed from the trace into the query proof buffers.
void leafSiblingsNoLeavesGPU(uint32_t arity, const uint64_t *d_trace, uint64_t nCols, uint64_t nRows, Layout layout,
                             const uint64_t *d_queries, uint64_t nQueries, uint64_t *d_scratch, uint64_t *d_proofBuf,
                             uint64_t bufferWidth, uint64_t maxTreeWidth, cudaStream_t stream);
void runGrindingGPU(uint64_t *d_nonce, uint64_t *d_nonceBlock, const uint64_t *d_in,
                    uint32_t n_bits, cudaStream_t stream);

#endif
