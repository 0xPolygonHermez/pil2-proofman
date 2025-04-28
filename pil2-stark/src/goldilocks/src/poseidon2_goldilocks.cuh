#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"

__global__ void linear_hash_gpu_2_2_tree_2(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows);
__global__ void linear_hash_gpu_coalesced_2(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t size, uint32_t num_rows);
__device__ __forceinline__ void poseidon2_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride);
__device__ __forceinline__ void poseidon2_store(gl64_t *out, uint32_t col_stride, size_t row_stride);
__device__ __forceinline__ void poseidon2_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols);
__device__ __forceinline__ void poseidon2_hash();
__device__ __noinline__ void pow7_2(gl64_t &x);
__device__ __forceinline__ void pow7add_2(gl64_t *x, const gl64_t C[SPONGE_WIDTH]);
__device__ __forceinline__ void matmul_m4_(gl64_t *x);
__device__ __forceinline__ void matmul_external_(gl64_t *x);
__device__ __forceinline__ void prodadd_(gl64_t *x, const gl64_t D[SPONGE_WIDTH], const gl64_t &sum);
__device__ __forceinline__ void matmul_m4_state_(uint32_t offset);
__device__ __forceinline__ void matmul_external_state_();
__device__ void hash_full_result_seq_2(gl64_t *state, const gl64_t *input, const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL);

