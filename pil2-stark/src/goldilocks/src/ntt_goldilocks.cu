#include "ntt_goldilocks.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_tooling.cuh"
#include "poseidon2_goldilocks.cuh"
#include "ntt_goldilocks.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "omp.h"
#include "poseidon2_goldilocks.hpp"
#include <atomic>
#include <mutex>

// CUDA Threads per Block
#define TPB_V1 64

// CUDA Threads Per Block
#define TPB_NTT 16
#define TPB_NTT_x 32
#define TPB_NTT_y 16
#define SHIFT 7


// #ifdef GPU_TIMING
#include "timer_gl.hpp"
// #endif

__global__ void br_ntt_8_steps(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size, uint32_t log_domain_size, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t col_min, uint32_t col_max);
__global__ void br_ntt_8_steps_blocks(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void br_ntt_batch_steps_blocks_par(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void br_ntt_batch_steps_blocks_par_dif( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void br_ntt_batch_steps_blocks_par2(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void br_ntt_batch_steps_blocks_par_noBR(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor);
__global__ void br_ntt_batch_steps_blocks_par_dif_noBR( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor);
__global__ void br_ntt_batch_steps_blocks_par_dif_noBR_compact( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor);

__global__ void br_ntt_group(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t nCols, bool inverse, bool extend, uint64_t maxLogDomainSize);

__global__ void reverse_permutation_new(gl64_t *data, uint32_t log_domain_size, uint32_t nCols);
__global__ void eval_twiddle_factors_small_size(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void eval_twiddle_factors_first_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void eval_twiddle_factors_second_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
void eval_twiddle_factors(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size, cudaStream_t stream);
__global__ void eval_r_small_size(gl64_t *r, uint32_t log_domain_size);
__global__ void eval_r_first_step(gl64_t *r, uint32_t log_domain_size);
__global__ void eval_r_second_step(gl64_t *r, uint32_t log_domain_size);
void eval_r(gl64_t *r, uint32_t log_domain_size, cudaStream_t stream);
void ntt_cuda( gl64_t *data, gl64_t **d_r, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
void ntt_cuda_blocks_par( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
void ntt_cuda_blocks_par_dif( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
void ntt_cuda_blocks_par1_noBR( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
void ntt_cuda_blocks_par1_noBR_compact( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
void ntt_cuda_blocks_par2_noBR( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);

__global__ void applyS(gl64_t *d_cmQ, gl64_t *d_q, gl64_t *d_S, Goldilocks::Element shiftIn, uint64_t N, uint64_t NExtended, uint64_t extendBits, uint64_t qDeg, uint64_t qDim)
{
    d_S[0] = gl64_t(uint64_t(1));
    for(uint64_t i = 1; i < qDeg; ++i) {
        d_S[i] = gl64_t(shiftIn.fe) * d_S[i - 1];
    }
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;

    for (uint64_t p = 0; p < qDeg; p++)
    {      
        Goldilocks3GPU::Element src;
        src[0] = d_q[getBufferOffsetRowMajor(row + p * N, 0, NExtended, qDim)];
        src[1] = d_q[getBufferOffsetRowMajor(row + p * N, 1, NExtended, qDim)];
        src[2] = d_q[getBufferOffsetRowMajor(row + p * N, 2, NExtended, qDim)];

        Goldilocks3GPU::Element dst;
        
        Goldilocks3GPU::mul((Goldilocks3GPU::Element &)dst,
                            (Goldilocks3GPU::Element &)src,
                            d_S[p]);
        d_cmQ[getBufferOffsetRowMajor(row, p * qDim, NExtended, qDeg * qDim)] = dst[0];
        d_cmQ[getBufferOffsetRowMajor(row, p * qDim + 1, NExtended, qDeg * qDim)] = dst[1];
        d_cmQ[getBufferOffsetRowMajor(row, p * qDim + 2, NExtended, qDeg * qDim)] = dst[2];
        for (uint64_t j = 1; j < (1 << extendBits); j++) {
            d_cmQ[getBufferOffsetRowMajor(row + j * N, p * qDim, NExtended, qDeg * qDim)] = gl64_t(uint64_t(0));
            d_cmQ[getBufferOffsetRowMajor(row + j * N, p * qDim + 1, NExtended, qDeg * qDim)] = gl64_t(uint64_t(0));
            d_cmQ[getBufferOffsetRowMajor(row + j * N, p * qDim + 2, NExtended, qDeg * qDim)] = gl64_t(uint64_t(0));
        }
    }
}

__global__ void prepareBlockFromRowMajor(gl64_t * dst, gl64_t * src, uint64_t nRows, uint64_t nCols)
{
    extern __shared__ gl64_t shared[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nRows || col >= nCols)
        return;
    shared[threadIdx.y * blockDim.x + threadIdx.x] = src[row * nCols + col];
    __syncthreads();
    int out_idx = getBufferOffset(row, col, nRows, nCols);
    dst[out_idx] = shared[threadIdx.y * blockDim.x + threadIdx.x];
}

void NTT_Goldilocks_GPU::prepare_blocks_trace(
    gl64_t* dst,
    gl64_t* src,
    uint64_t nCols,
    uint64_t nRows,
    cudaStream_t stream,
    TimerGPU &timer
) {
    if (nCols == 0 || nRows == 0) return;
    dim3 block(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid((nRows + block.x - 1) / block.x,
             (nCols + block.y - 1) / block.y);
    int sharedMemSize = block.x * block.y * sizeof(gl64_t);
    prepareBlockFromRowMajor<<<grid, block, sharedMemSize, stream>>>(dst, src, nRows, nCols);
    CHECKCUDAERR(cudaGetLastError());
}

__global__ void transposeSubBlocksInPlace(gl64_t * data, uint64_t nRows, uint64_t nCols)
{ 
    extern __shared__ gl64_t shared[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nRows || col >= nCols)
        return;

    uint64_t offset_src = getBufferOffsetRowMajor(row, col, nRows, nCols);
    shared[threadIdx.y * blockDim.x + threadIdx.x] = data[offset_src];
    __syncthreads();
    uint64_t offset_dst = getBufferOffset(row, col, nRows, nCols);
    data[offset_dst] = shared[threadIdx.y * blockDim.x + threadIdx.x];
}

__global__ void transposeSubBlocksBackInPlace(gl64_t *data, uint64_t nRows, uint64_t nCols)
{
    extern __shared__ gl64_t shared[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nRows || col >= nCols)
        return;

    uint64_t offset_src = getBufferOffset(row, col, nRows, nCols);
    shared[threadIdx.y * blockDim.x + threadIdx.x] = data[offset_src];
    __syncthreads();
    uint64_t offset_dst = getBufferOffsetRowMajor(row, col, nRows, nCols);
    data[offset_dst] = shared[threadIdx.y * blockDim.x + threadIdx.x];
}

//Assumes src and dst buffers are disjoint
__global__ void transposeSubBlocksBack(gl64_t *src, uint64_t n_bits_src, gl64_t *dst, uint64_t n_bits_dst, uint64_t nCols)
{
    extern __shared__ gl64_t shared[];   
    int n_src = 1 << n_bits_src;
    int n_dst = 1 << n_bits_dst;

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= n_src || col >= nCols)
        return;

    uint64_t offset_src = getBufferOffset(row, col, n_src, nCols);
    shared[threadIdx.y * blockDim.x + threadIdx.x] = src[offset_src];
    __syncthreads();
    uint64_t offset_dst = getBufferOffsetRowMajor(row, col, n_dst, nCols);
    dst[offset_dst] = shared[threadIdx.y * blockDim.x + threadIdx.x];

    for (uint64_t j = 1; j < (1 << (n_bits_dst - n_bits_src)); j++) {
        int offset_dst2 = getBufferOffset(row + j * n_src, col, n_dst, nCols);
        dst[offset_dst2] = gl64_t(uint64_t(0));
    }
}

//Assumes src and dst buffers are disjoint
//not bit-reversal version
__global__ void transposeSubBlocksBack_noBR(gl64_t *src, uint64_t n_bits_src, gl64_t *dst, uint64_t n_bits_dst, uint64_t nCols)
{
    extern __shared__ gl64_t shared[];   
    uint32_t n_src = 1 << n_bits_src;
    uint32_t n_dst = 1 << n_bits_dst;
    uint32_t blowup = 1 << (n_bits_dst - n_bits_src);

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= n_src || col >= nCols)
        return;

    uint64_t offset_src = getBufferOffset(row, col, n_src, nCols);
    shared[threadIdx.y * blockDim.x + threadIdx.x] = src[offset_src];
    __syncthreads();
    uint64_t target_row = row * blowup;
    uint64_t offset_dst = getBufferOffsetRowMajor(target_row, col, n_dst, nCols);
    dst[offset_dst] = shared[threadIdx.y * blockDim.x + threadIdx.x];
    for (uint64_t j = 1; j < (1 << (n_bits_dst - n_bits_src)); j++) {
        uint64_t offset_dst2 = getBufferOffsetRowMajor(target_row + j, col, n_dst, nCols);
        dst[offset_dst2] = gl64_t(uint64_t(0));
    }
}

//Assumes src and dst buffers are disjoint
//not bit-reversal version
//compact format, fills only the first TILE_HEIGHT/(blowup factor) rows of each block
__global__ void transposeSubBlocksBack_noBR_compact(gl64_t *src, uint64_t n_bits_src, gl64_t *dst, uint64_t n_bits_dst, uint64_t nCols)
{
    extern __shared__ gl64_t shared[];   
    uint32_t n_src = 1 << n_bits_src;
    uint32_t n_dst = 1 << n_bits_dst;

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t blowupFactor = 1 << (n_bits_dst - n_bits_src);

    if(row >= n_src || col >= nCols)
        return;

    uint64_t offset_src = getBufferOffset(row, col, n_src, nCols);
    shared[threadIdx.y * blockDim.x + threadIdx.x] = src[offset_src];
    __syncthreads();
    uint64_t offset_dst = getBufferOffsetRowMajor_compact(row, col, n_dst, nCols, blowupFactor);
    dst[offset_dst] = shared[threadIdx.y * blockDim.x + threadIdx.x];
}

void NTT_Goldilocks_GPU::computeQ_inplace(Goldilocks::Element *d_tree, uint64_t offset_cmQ, uint64_t offset_q, uint64_t qDeg, uint64_t qDim, Goldilocks::Element shiftIn, uint64_t n_bits, uint64_t n_bits_ext, uint64_t nCols, gl64_t *d_aux_trace, uint64_t offset_helper, TimerGPU &timer, cudaStream_t stream)
{
   
    if (nCols == 0 || n_bits_ext == 0)
    {
        return;
    }

    TimerStartCategoryGPU(timer, NTT);

    if(n_bits_ext > maxLogDomainSize)
    {
        printf("[NTT] ERROR: n_bits_ext %lu exceeds maxLogDomainSize %lu\n", n_bits_ext, maxLogDomainSize);
        abort();
    }

    uint64_t N = 1 << n_bits;
    uint64_t NExtended = 1 << n_bits_ext;
    gl64_t* d_S = d_aux_trace + offset_helper;
    gl64_t *d_q = d_aux_trace + offset_q;
    gl64_t *d_cmQ = d_aux_trace + offset_cmQ;

    // Intt
    dim3 block(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid0((NExtended + block.x - 1) / block.x,
             (qDim + block.y - 1) / block.y);
    int sharedMemSize = block.x * block.y * sizeof(gl64_t);
    transposeSubBlocksBackInPlace<<<grid0, block, sharedMemSize, stream>>>(d_q, NExtended, qDim);
    ntt_cuda_blocks_par(d_q, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, n_bits_ext, qDim, true, false, stream, maxLogDomainSize);

    dim3 threads(TILE_HEIGHT, 1, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, 1, 1);
    applyS<<<blocks, threads, 0, stream>>>(d_cmQ, d_q, d_S, shiftIn, N, NExtended, n_bits_ext - n_bits, qDeg, qDim);

    dim3 grid1((NExtended + block.x - 1) / block.x,
             (nCols + block.y - 1) / block.y);
    ntt_cuda_blocks_par(d_cmQ, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, n_bits_ext, nCols, false, false, stream, maxLogDomainSize);
    transposeSubBlocksInPlace<<<grid1, block, sharedMemSize, stream>>>(d_cmQ, NExtended, nCols);

    TimerStopCategoryGPU(timer, NTT);
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    Poseidon2GoldilocksGPUCommit::merkletreeCoalescedBlocks(3, (uint64_t*) d_tree, (uint64_t *)d_cmQ, nCols, NExtended, stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);
}

void NTT_Goldilocks_GPU::LDE_MerkleTree_GPU(Goldilocks::Element *d_tree, gl64_t *d_dst_ntt, uint64_t offset_dst_ntt, gl64_t *d_src_ntt, uint64_t offset_src_ntt, u_int64_t n_bits, u_int64_t n_bits_ext, u_int64_t nCols, TimerGPU &timer, cudaStream_t stream)
{
    if (nCols == 0 || n_bits == 0)
    {
        return;
    }
    TimerStartCategoryGPU(timer, NTT);
    if (n_bits_ext > maxLogDomainSize)
    {
        printf("[NTT] ERROR: n_bits_ext %lu exceeds maxLogDomainSize %lu\n", n_bits_ext, maxLogDomainSize);
        abort();
    }

    uint64_t size = 1 << n_bits;
    uint64_t ext_size = 1 << n_bits_ext;    
    gl64_t *d_dst_ntt_ = &d_dst_ntt[offset_dst_ntt];
    gl64_t *d_src_ntt_ = &d_src_ntt[offset_src_ntt];
    dim3 block(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid0((size + block.x - 1) / block.x,
             (nCols + block.y - 1) / block.y);
    int sharedMemSize = block.x * block.y * sizeof(gl64_t);

#if 1
    transposeSubBlocksBack_noBR_compact<<<grid0, block, sharedMemSize, stream>>>(d_src_ntt_, n_bits, d_dst_ntt_, n_bits_ext, nCols);
    ntt_cuda_blocks_par1_noBR_compact(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, n_bits_ext, nCols, true, true, stream, maxLogDomainSize); 
    ntt_cuda_blocks_par2_noBR(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, n_bits_ext, nCols, false, false, stream, maxLogDomainSize);
#endif

#if 0
    transposeSubBlocksBack_noBR<<<grid0, block, sharedMemSize, stream>>>(d_src_ntt_, n_bits, d_dst_ntt_, n_bits_ext, nCols);
    ntt_cuda_blocks_par1_noBR(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, n_bits_ext, nCols, true, true, stream, maxLogDomainSize); 
    ntt_cuda_blocks_par2_noBR(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, n_bits_ext, nCols, false, false, stream, maxLogDomainSize);
#endif
#if 0
    transposeSubBlocksBack<<<grid0, block, sharedMemSize, stream>>>(d_src_ntt_, n_bits, d_dst_ntt_, n_bits_ext, nCols);
    ntt_cuda_blocks_par(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, n_bits_ext, nCols, true, true, stream, maxLogDomainSize); 
    ntt_cuda_blocks_par(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, n_bits_ext, nCols, false, false, stream, maxLogDomainSize);
#endif
    dim3 grid1((ext_size + block.x - 1) / block.x,
             (nCols + block.y - 1) / block.y);
    transposeSubBlocksInPlace<<<grid1, block, sharedMemSize, stream>>>(d_dst_ntt_, ext_size, nCols);
    TimerStopCategoryGPU(timer, NTT);
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    Poseidon2GoldilocksGPUCommit::merkletreeCoalescedBlocks(3, (uint64_t*) d_tree, (uint64_t *)d_dst_ntt_, nCols, ext_size, stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);
}

void NTT_Goldilocks_GPU::INTT_inplace(gl64_t *dst, u_int64_t n_bits, u_int64_t nCols, cudaStream_t stream)
{
    if (nCols == 0 || n_bits == 0)
    {
        return;
    }
    if (n_bits > maxLogDomainSize)
    {
        printf("[NTT] ERROR: n_bits %lu exceeds maxLogDomainSize %lu\n", n_bits, maxLogDomainSize);
        abort();
    }

    uint64_t N = 1 << n_bits;

    dim3 block_0(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid_0((N + block_0.x - 1) / block_0.x,
             (nCols + block_0.y - 1) / block_0.y);
    int sharedMemSize_0 = block_0.x * block_0.y * sizeof(gl64_t);
    transposeSubBlocksBackInPlace<<<grid_0, block_0, sharedMemSize_0, stream>>>(dst, N, nCols);
    ntt_cuda_blocks_par(dst, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, n_bits, nCols, true, false, stream, maxLogDomainSize);
    transposeSubBlocksInPlace<<<grid_0, block_0, sharedMemSize_0, stream>>>(dst, N, nCols);
}

// Static member definitions
gl64_t **NTT_Goldilocks_GPU::d_fwd_twiddle_factors = nullptr;
gl64_t **NTT_Goldilocks_GPU::d_inv_twiddle_factors = nullptr;
gl64_t **NTT_Goldilocks_GPU::d_r = nullptr;
uint64_t NTT_Goldilocks_GPU::maxLogDomainSize = 0;
uint32_t NTT_Goldilocks_GPU::nGPUs_available = 0;

void NTT_Goldilocks_GPU::init_twiddle_factors_and_r(uint64_t maxLogDomainSize_, uint32_t nGPUs_input, uint32_t* gpu_ids_) {
    static std::mutex init_mutex;
    std::lock_guard<std::mutex> lock(init_mutex);

    
    int nGPUs_available_;
    cudaGetDeviceCount(&nGPUs_available_);
    assert(maxLogDomainSize_ <= 32);

    if(maxLogDomainSize_ > maxLogDomainSize || nGPUs_available_ != nGPUs_available) {
        free_twiddle_factors_and_r(); 
        maxLogDomainSize = maxLogDomainSize_;
        nGPUs_available = nGPUs_available_;
        d_fwd_twiddle_factors = new gl64_t*[nGPUs_available];
        d_inv_twiddle_factors = new gl64_t*[nGPUs_available];
        d_r = new gl64_t*[nGPUs_available];
        for(int i=0; i < nGPUs_available; i++) {
            d_fwd_twiddle_factors[i] = nullptr;
            d_inv_twiddle_factors[i] = nullptr;
            d_r[i] = nullptr;
        }
    }
    uint32_t nGPUs;
    uint32_t* gpu_ids = nullptr;
    bool free_inputs = false;
    if( nGPUs_input == 0 || gpu_ids_ == nullptr) {
        nGPUs = nGPUs_available;
        gpu_ids = new uint32_t[nGPUs_available];
        for(int i = 0; i < nGPUs_available; i++) {
            gpu_ids[i] = i;
        }
        free_inputs = true;
    }else{
        nGPUs = nGPUs_input;
        gpu_ids = gpu_ids_;
    }

    cudaStream_t stream[nGPUs];
    
    for (int i = 0; i < nGPUs; i++) {
        if (d_fwd_twiddle_factors[gpu_ids[i]] != nullptr && d_inv_twiddle_factors[gpu_ids[i]] != nullptr && d_r[gpu_ids[i]] != nullptr) {
            continue; // Already initialized
        } else {
            assert(d_fwd_twiddle_factors[gpu_ids[i]] == nullptr && d_inv_twiddle_factors[gpu_ids[i]] == nullptr && d_r[gpu_ids[i]] == nullptr);
            cudaSetDevice(gpu_ids[i]);
            cudaStreamCreate(&stream[i]);
            cudaMalloc(&d_fwd_twiddle_factors[gpu_ids[i]], (1 << (maxLogDomainSize - 1)) * sizeof(gl64_t));
            cudaMalloc(&d_inv_twiddle_factors[gpu_ids[i]], (1 << (maxLogDomainSize - 1)) * sizeof(gl64_t));
            cudaMalloc(&d_r[gpu_ids[i]], (1 << maxLogDomainSize) * sizeof(gl64_t));
            eval_twiddle_factors(d_fwd_twiddle_factors[gpu_ids[i]], d_inv_twiddle_factors[gpu_ids[i]], maxLogDomainSize, stream[i]);
            eval_r(d_r[gpu_ids[i]], maxLogDomainSize, stream[i]);
        }
    }
    for (int i = 0; i < nGPUs; i++) {
        cudaSetDevice(gpu_ids[i]);
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    if(free_inputs) {
        delete[] gpu_ids;
    }
    CHECKCUDAERR(cudaGetLastError());
}

void NTT_Goldilocks_GPU::free_twiddle_factors_and_r() {
    static std::mutex free_mutex;
    std::lock_guard<std::mutex> lock(free_mutex);

    if (d_fwd_twiddle_factors == nullptr) {
        assert(d_inv_twiddle_factors == nullptr);
        assert(d_r == nullptr);
        return; // Already freed or never allocated
    }

    for(int i = 0; i < nGPUs_available; i++) {
        if(d_fwd_twiddle_factors[i] != nullptr && d_inv_twiddle_factors[i] != nullptr && d_r[i] != nullptr) {
            cudaSetDevice(i);
            cudaFree(d_fwd_twiddle_factors[i]);
            cudaFree(d_inv_twiddle_factors[i]);
            cudaFree(d_r[i]);
        } else {
            assert(d_fwd_twiddle_factors[i] == nullptr && d_inv_twiddle_factors[i] == nullptr && d_r[i] == nullptr);
        }
    }
    delete[] d_fwd_twiddle_factors;
    delete[] d_inv_twiddle_factors;
    delete[] d_r;

    // Reset pointers to nullptr
    d_fwd_twiddle_factors = nullptr;
    d_inv_twiddle_factors = nullptr;
    d_r = nullptr;
}

__global__ void br_ntt_group(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t nCols, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    uint32_t i = blockIdx.x;
    uint32_t col = threadIdx.x;

    if (i < domain_size / 2 && col < nCols)
    {
        uint32_t group_size = 1 << stage;
        uint32_t group = i >> stage;                          // i/(group_size)
        uint32_t group_pos = i & (group_size - 1);            // i%(group_size)
        uint32_t index1 = (group << (stage + 1)) + group_pos; // stage + 1 is sizeof of group
        uint32_t index2 = index1 + group_size;
        gl64_t factor = twiddles[group_pos * ((1 << maxLogDomainSize) >> (stage + 1))];  // Use actual domain size
        gl64_t odd_sub = gl64_t((uint64_t)data[index2 * nCols + col]) * factor;
        gl64_t result1 = gl64_t((uint64_t)data[index1 * nCols + col]) + odd_sub;
        gl64_t result2 = gl64_t((uint64_t)data[index1 * nCols + col]) - odd_sub;
        
        // Apply scaling only on the last stage for inverse NTT
        if(inverse && stage == log_domain_size - 1){
            gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size]);
            if(extend) {
                result1 *= inv_factor * d_r[index1];
                result2 *= inv_factor * d_r[index2];
            } else {
                result1 *= inv_factor;
                result2 *= inv_factor;
            }
        }
        
        data[index1 * nCols + col] = result1;
        data[index2 * nCols + col] = result2;
    }
}

__global__ void br_ntt_group_blocks(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t domain_size_out, uint32_t nCols, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    uint32_t i = blockIdx.x;
    uint32_t col = threadIdx.x;
    uint32_t block_idx = col / BATCH_WIDTH;
    uint32_t block_col = col % BATCH_WIDTH;
    uint32_t ncols_block = (nCols - block_idx * BATCH_WIDTH) < BATCH_WIDTH ? (nCols - block_idx * BATCH_WIDTH) : BATCH_WIDTH;
    gl64_t* data_block = data + block_idx * BATCH_WIDTH * domain_size_out;

    if (i < domain_size / 2 && col < nCols)
    {
        uint32_t group_size = 1 << stage;
        uint32_t group = i >> stage;                          // i/(group_size)
        uint32_t group_pos = i & (group_size - 1);       // i%(group_size)
        uint32_t index1 = (group << (stage + 1)) + group_pos; // stage + 1 is sizeof of group
        uint32_t index2 = index1 + group_size;
        gl64_t factor = twiddles[group_pos * ((1 << maxLogDomainSize) >> (stage + 1))];  // Use actual domain size
        gl64_t odd_sub = gl64_t((uint64_t)data_block[index2 * ncols_block + block_col]) * factor;
        gl64_t result1 = gl64_t((uint64_t)data_block[index1 * ncols_block + block_col]) + odd_sub;
        gl64_t result2 = gl64_t((uint64_t)data_block[index1 * ncols_block + block_col]) - odd_sub;
        
        // Apply scaling only on the last stage for inverse NTT
        if(inverse && stage == log_domain_size - 1){
            gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size]);
            if(extend) {
                result1 *= inv_factor * d_r[index1];
                result2 *= inv_factor * d_r[index2];
            } else {
                result1 *= inv_factor;
                result2 *= inv_factor;
            }
        }

        data_block[index1 * ncols_block + block_col] = result1;
        data_block[index2 * ncols_block + block_col] = result2;
    }
}

__global__ void br_ntt_group_blocks_dif(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t domain_size_out, uint32_t nCols, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    uint32_t i = blockIdx.x;
    uint32_t col = threadIdx.x;
    uint32_t block_idx = col / BATCH_WIDTH;
    uint32_t block_col = col % BATCH_WIDTH;
    uint32_t ncols_block = (nCols - block_idx * BATCH_WIDTH) < BATCH_WIDTH ? (nCols - block_idx * BATCH_WIDTH) : BATCH_WIDTH;
    gl64_t* data_block = data + block_idx * BATCH_WIDTH * domain_size_out;

    if (i < domain_size / 2 && col < nCols)
    {
        uint32_t group_size = 1 << stage;
        uint32_t group = i >> stage;                          // i/(group_size)
        uint32_t group_pos = i & (group_size - 1);            // i%(group_size)
        uint32_t index1 = (group << (stage + 1)) + group_pos; // group * 2*group_size + group_pos
        uint32_t index2 = index1 + group_size;
        gl64_t factor = twiddles[group_pos * ((1 << maxLogDomainSize) >> (stage + 1))];  // Use actual domain size

        gl64_t a = gl64_t((uint64_t)data_block[index1 * ncols_block + block_col]);
        gl64_t b = gl64_t((uint64_t)data_block[index2 * ncols_block + block_col]);
        gl64_t result1 = a + b;
        gl64_t result2 = (a - b) * factor;
        
        // Apply scaling only on the last stage for inverse NTT
        if(inverse && stage == 0){
            gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size]);
            if(extend) {
                uint64_t index1_r = (__brev(index1) >> (32 - log_domain_size));
                result1 *= inv_factor * d_r[index1_r];
                uint64_t index2_r = (__brev(index2) >> (32 - log_domain_size));
                result2 *= inv_factor * d_r[index2_r];
            } else {
                result1 *= inv_factor;
                result2 *= inv_factor;
            }
        }

        data_block[index1 * ncols_block + block_col] = result1;
        data_block[index2 * ncols_block + block_col] = result2;
    }
}

__global__ void br_ntt_8_steps(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size, uint32_t log_domain_size, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t col_min, uint32_t col_max)
{
    __shared__ gl64_t tile[1024];

    //assume domain_size is multiple of 256
    uint32_t n_loc_steps = min(log_domain_size - base_step, 8);    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Reorganize row indices for batched processing based on base_step (shift base_step bits after each batch)
    uint32_t groupSize = 1 << base_step;
    uint32_t nGroups = domain_size / groupSize;
    uint32_t low_bits = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    //remaining steps
    uint32_t remaining_high_bits = log_domain_size - (base_step+1); //log(nGroups / 2)
    uint32_t high_mask = (1 << remaining_high_bits) - 1; 
    
    for(int col_base = col_min; col_base <= col_max; col_base +=4){
        
        //copy data to tile 
        tile[threadIdx.x*4] = data[row*nCols + col_base];
        if(col_base + 3 < nCols){
            tile[threadIdx.x*4+1] = data[row*nCols + col_base+1];
            tile[threadIdx.x*4+2] = data[row*nCols + col_base+2];
            tile[threadIdx.x*4+3] = data[row*nCols + col_base+3];
        } else if(col_base + 2 < nCols){
            tile[threadIdx.x*4+1] = data[row*nCols + col_base+1];
            tile[threadIdx.x*4+2] = data[row*nCols + col_base+2];
        } else if(col_base + 1 < nCols){
            tile[threadIdx.x*4+1] = data[row*nCols + col_base+1];
        }
        
        __syncthreads();

        for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
            uint32_t i = threadIdx.x;
            if (threadIdx.x < 128){ // Only process first 128 threads (half of them)
                uint32_t group_size = 1 << loc_step;   
                uint32_t group = i >> loc_step;                          // i/(group_size)    
                uint32_t group_pos = i & (group_size - 1);               // i%(group_size)  
                uint32_t index1 = (group << (loc_step + 1)) + group_pos; // 2*group*group_size + group_pos
                uint32_t index2 = index1 + group_size;
                gl64_t factor;
                {
                    //global_step
                    uint32_t gs = base_step + loc_step;
                    //global_group_size
                    uint32_t ggs = 1 << gs;
                    //global_group_pos
                    uint32_t ggp =(blockIdx.x << 7) + i; //blockIdx.x* blockDim.x/2 + i;
                    ggp = ((ggp & high_mask)<< base_step) + (ggp >> remaining_high_bits);
                    ggp = ggp & (ggs - 1);
                    factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];  // Use actual domain size
                }
                
                index1 = index1 << 2; //4 rows at once
                index2 = index2 << 2;
                gl64_t odd_sub = tile[index2] * factor;
                tile[index2] = tile[index1] - odd_sub;               
                tile[index1] = tile[index1] + odd_sub;
                
                index1 = index1 + 1;
                index2 = index2 + 1;
                odd_sub = tile[index2] * factor;
                tile[index2] = tile[index1] - odd_sub;               
                tile[index1] = tile[index1] + odd_sub;

                index1 = index1 + 1;
                index2 = index2 + 1;
                odd_sub = tile[index2] * factor;
                tile[index2] = tile[index1] - odd_sub;               
                tile[index1] = tile[index1] + odd_sub;

                index1 = index1 + 1;
                index2 = index2 + 1;
                odd_sub = tile[index2] * factor;
                tile[index2] = tile[index1] - odd_sub;               
                tile[index1] = tile[index1] + odd_sub;                
            }
            __syncthreads();
        }
        // copy values to data with scaling applied when this iteration includes the final stage
        if(inverse && (base_step + n_loc_steps) >= log_domain_size){
            gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size]);
            if(extend) inv_factor = inv_factor * d_r[row];
            data[row*nCols + col_base] = tile[threadIdx.x*4] * inv_factor;
            if(col_base + 3 < nCols){
                data[row*nCols + col_base+1] = tile[threadIdx.x*4+1] * inv_factor;
                data[row*nCols + col_base+2] = tile[threadIdx.x*4+2] * inv_factor;
                data[row*nCols + col_base+3] = tile[threadIdx.x*4+3] * inv_factor;
            } else if(col_base + 2 < nCols){
                data[row*nCols + col_base+1] = tile[threadIdx.x*4+1] * inv_factor;
                data[row*nCols + col_base+2] = tile[threadIdx.x*4+2] * inv_factor;
            } else if(col_base + 1 < nCols){
                data[row*nCols + col_base+1] = tile[threadIdx.x*4+1] * inv_factor;
            }
        }else{
            data[row*nCols + col_base] = tile[threadIdx.x*4];
            if(col_base + 3 < nCols){
                data[row*nCols + col_base+1] = tile[threadIdx.x*4+1];
                data[row*nCols + col_base+2] = tile[threadIdx.x*4+2];
                data[row*nCols + col_base+3] = tile[threadIdx.x*4+3];
            } else if(col_base + 2 < nCols){
                data[row*nCols + col_base+1] = tile[threadIdx.x*4+1];
                data[row*nCols + col_base+2] = tile[threadIdx.x*4+2];
            } else if(col_base + 1 < nCols){
                data[row*nCols + col_base+1] = tile[threadIdx.x*4+1];
            }
        }
    }   
}

__global__ void br_ntt_batch_steps_blocks_par(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(log_domain_size_in - base_step, BATCH_HEIGHT_LOG2);    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    //evaluate row as if I shifted BATCH_WIDTH bits after each batch
    uint32_t groupSize = 1 << base_step;
    uint32_t nGroups = domain_size_in / groupSize;
    uint32_t low_bits = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    //define my block
    uint32_t block_stride = domain_size_out * BATCH_WIDTH;
    gl64_t *data_block = data + blockIdx.y*block_stride;
    uint32_t col_base = blockIdx.y * BATCH_WIDTH;
    uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? nCols - col_base : BATCH_WIDTH;
    
    //copy data to tile 
    for(int i=0; i<ncols_block; i++){
        tile[BATCH_HEIGHT*i+threadIdx.x] = data_block[row*ncols_block+i];
    }
    
    __syncthreads();

     //remaining steps
    uint32_t remaining_high_bits = log_domain_size_in - (base_step+1); //log(nGroups / 2) i.e log(butterfly groups)
    uint32_t high_mask = (1 << remaining_high_bits) - 1; 

    for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
        uint32_t i = threadIdx.x;
        if (threadIdx.x < BATCH_HEIGHT_DIV2){ // Only process first (half of them)
            uint32_t group_size = 1 << loc_step;   
            uint32_t group = i >> loc_step;                          // i/(group_size)    
            uint32_t group_pos = i & (group_size - 1);               // i%(group_size)  
            uint32_t index1 = (group << (loc_step + 1)) + group_pos; // 2*group*group_size + group_pos
            uint32_t index2 = index1 + group_size;
            gl64_t factor;
            {
                //global_step
                uint32_t gs = base_step + loc_step;
                //global_group_size
                uint32_t ggs = 1 << gs;
                //batched butterfly index
                uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                //global_group_pos
                uint32_t gbi = (((bbi & high_mask)<< base_step) + (bbi >> remaining_high_bits)); //undo the batching
                uint32_t ggp = gbi & (ggs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];  
            }
            for(int j=0; j<ncols_block; j++){
                gl64_t odd_sub = tile[ j*BATCH_HEIGHT + index2] * factor;
                tile[j*BATCH_HEIGHT +index2] = tile[j*BATCH_HEIGHT + index1] - odd_sub;               
                tile[j*BATCH_HEIGHT +index1] = tile[j*BATCH_HEIGHT + index1] + odd_sub;                
            }                             
        }
        __syncthreads();
    }
    // copy values to data with scaling applied when this iteration includes the final stage
    if(inverse && (base_step + n_loc_steps) >= log_domain_size_in){
        gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size_in]);
        if(extend) inv_factor = inv_factor * d_r[row];
        for(int i=0; i<ncols_block; i++){
            data_block[row*ncols_block+i] = tile[i*BATCH_HEIGHT+threadIdx.x] * inv_factor;
        }
    }else{
        for(int i=0; i<ncols_block; i++){
            data_block[row*ncols_block+i] = tile[i*BATCH_HEIGHT+threadIdx.x];
        }
    } 
}

__global__ void br_ntt_batch_steps_blocks_par_noBR(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(log_domain_size_in - base_step, BATCH_HEIGHT_LOG2);    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    //evaluate row as if I shifted BATCH_WIDTH bits after each batch
    uint32_t groupSize = 1 << base_step;
    uint32_t nGroups = domain_size_in / groupSize;
    uint32_t low_bits = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;
    uint32_t rowbf = row *blowupFactor; 

    //remaining steps
    uint32_t remaining_high_bits = log_domain_size_in - (base_step+1); //log(nGroups / 2)
    uint32_t high_mask = (1 << remaining_high_bits) - 1; 
    uint32_t offset = domain_size_out * BATCH_WIDTH;

    uint32_t block=blockIdx.y;
    gl64_t *data_block = data + block*offset;
    uint32_t col_base = block * BATCH_WIDTH;
    uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? nCols - col_base : BATCH_WIDTH;
    //copy data to tile 
    for(int i=0; i<ncols_block; i++){
        tile[BATCH_HEIGHT*i+threadIdx.x] = data_block[rowbf*ncols_block+i];
    }
    
    __syncthreads();

    for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
        uint32_t i = threadIdx.x;
        if (threadIdx.x < BATCH_HEIGHT_DIV2){ // Only process first (half of them)
            uint32_t group_size = 1 << loc_step;   
            uint32_t group = i >> loc_step;                          // i/(group_size)    
            uint32_t group_pos = i & (group_size - 1);               // i%(group_size)  
            uint32_t index1 = (group << (loc_step + 1)) + group_pos; // 2*group*group_size + group_pos
            uint32_t index2 = index1 + group_size;
            gl64_t factor; 
            {
                //global_step
                uint32_t gs = base_step + loc_step;
                //global_group_size
                uint32_t ggs = 1 << gs;
                //batched butterfly index
                uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                //global_group_pos
                uint32_t gbi = (((bbi & high_mask)<< base_step) + (bbi >> remaining_high_bits)); //undo the batching
                uint32_t ggp = gbi & (ggs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];
            }
            for(int j=0; j<ncols_block; j++){
                gl64_t odd_sub = tile[ j*BATCH_HEIGHT + index2] * factor;
                tile[j*BATCH_HEIGHT +index2] = tile[j*BATCH_HEIGHT + index1] - odd_sub;               
                tile[j*BATCH_HEIGHT +index1] = tile[j*BATCH_HEIGHT + index1] + odd_sub;                
            }                             
        }
        __syncthreads();
    }
    // copy values to data with scaling applied when this iteration includes the final stage
    if(inverse && (base_step + n_loc_steps) >= log_domain_size_in){
        gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size_in]);
        if(extend) inv_factor = inv_factor * d_r[row];
        for(int i=0; i<ncols_block; i++){
            data_block[rowbf*ncols_block+i] = tile[i*BATCH_HEIGHT+threadIdx.x] * inv_factor;
        }
    }else{
        for(int i=0; i<ncols_block; i++){
            data_block[rowbf*ncols_block+i] = tile[i*BATCH_HEIGHT+threadIdx.x];
        }
    } 
}

__global__ void br_ntt_batch_steps_blocks_par_dif( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(base_step+1, BATCH_HEIGHT_LOG2);
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // Evaluate row as if we shifted BATCH_WIDTH bits after each batch (same packing as DIT version)
    uint32_t groupSize = 1 << (base_step + 1 - n_loc_steps);
    uint32_t nGroups   = domain_size_in / groupSize;
    uint32_t low_bits  = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    // Define my block
    uint32_t block_stride = domain_size_out * BATCH_WIDTH;
    gl64_t *data_block = data + blockIdx.y * block_stride;
    uint32_t col_base = blockIdx.y * BATCH_WIDTH;
    uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? (nCols - col_base) : BATCH_WIDTH;

    // Copy data to tile
    for (int i = 0; i < ncols_block; i++){
        tile[BATCH_HEIGHT * i + threadIdx.x] = data_block[row * ncols_block + i];
    }
    __syncthreads();


    // remaining_high_bits: number of high bits for butterfly indexing (same meaning as your DIT code)
    uint32_t remaining_high_bits = log_domain_size_in -1 - (base_step + 1 - n_loc_steps); // log(nGroups/2)
    uint32_t high_mask = (1u << remaining_high_bits) - 1u;
    for (int loc_step = n_loc_steps-1; loc_step >= 0; loc_step--){
        uint32_t i = threadIdx.x;
        if (i < BATCH_HEIGHT_DIV2) { // Only process first half (each handles one butterfly)
            uint32_t group_size = 1u << loc_step;
            uint32_t group = i >> loc_step;                       // i / group_size
            uint32_t group_pos = i & (group_size - 1u);           // i % group_size
            uint32_t index1 = (group << (loc_step + 1)) + group_pos; // 2*group*group_size + group_pos
            uint32_t index2 = index1 + group_size;

            // compute twiddle using the batched->global mapping (same as DIT)
            gl64_t factor; 
            {
                //global_step
                uint32_t gs = base_step -(n_loc_steps -1 - loc_step);
                //global_group_size
                uint32_t ggs = 1 << gs;
                //batched butterfly index
                uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                //global_group_pos
                uint32_t gbi = (((bbi & high_mask)<< (base_step + 1 - n_loc_steps)) + (bbi >> remaining_high_bits)); //undo the batching
                uint32_t ggp = gbi & (ggs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];                
            }

            // DIF: t1 = a + b           -> stored at index1
            //      t2 = (a - b) * W^k   -> stored at index2
            for (int j = 0; j < ncols_block; j++) {
                gl64_t a = tile[j * BATCH_HEIGHT + index1];
                gl64_t b = tile[j * BATCH_HEIGHT + index2];

                gl64_t t1 = a + b;
                gl64_t t2 = a - b;
                t2 = t2 * factor;

                tile[j * BATCH_HEIGHT + index1] = t1;
                tile[j * BATCH_HEIGHT + index2] = t2;
            }
            
        }
        __syncthreads();
    }

    // copy values back to data with scaling applied when this iteration includes the final stage
    if (inverse && (base_step + 1  - n_loc_steps) <= 0) {
        gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size_in]);
        uint64_t row_r = (__brev(row) >> (32 - log_domain_size_in));
        if (extend) inv_factor = inv_factor * d_r[row_r];
        for (int i = 0; i < ncols_block; i++){
            data_block[row * ncols_block + i] = tile[i * BATCH_HEIGHT + threadIdx.x] * inv_factor;
        }
    } else {
        for (int i = 0; i < ncols_block; i++){
            data_block[row * ncols_block + i] = tile[i * BATCH_HEIGHT + threadIdx.x];
        }
    }
}

__global__ void br_ntt_batch_steps_blocks_par_dif_noBR( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(base_step+1, BATCH_HEIGHT_LOG2);
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // Evaluate row as if we shifted BATCH_WIDTH bits after each batch (same packing as DIT version)
    uint32_t groupSize = 1 << (base_step + 1 - n_loc_steps);
    uint32_t nGroups   = domain_size_in / groupSize;
    uint32_t low_bits  = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;
    uint32_t rowbf = row *blowupFactor; 


    // Define my block
    uint32_t block_stride = domain_size_out * BATCH_WIDTH;
    gl64_t *data_block = data + blockIdx.y * block_stride;
    uint32_t col_base = blockIdx.y * BATCH_WIDTH;
    uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? (nCols - col_base) : BATCH_WIDTH;

    // Copy data to tile
    for (int i = 0; i < ncols_block; i++){
        tile[BATCH_HEIGHT * i + threadIdx.x] = data_block[rowbf * ncols_block + i];
    }
    __syncthreads();


    // remaining_high_bits: number of high bits for butterfly indexing (same meaning as your DIT code)
    uint32_t remaining_high_bits = log_domain_size_in -1 - (base_step + 1 - n_loc_steps); // log(nGroups/2)
    uint32_t high_mask = (1u << remaining_high_bits) - 1u;
    for (int loc_step = n_loc_steps-1; loc_step >= 0; loc_step--){
        uint32_t i = threadIdx.x;
        if (i < BATCH_HEIGHT_DIV2) { // Only process first half (each handles one butterfly)
            uint32_t group_size = 1u << loc_step;
            uint32_t group = i >> loc_step;                       // i / group_size
            uint32_t group_pos = i & (group_size - 1u);           // i % group_size
            uint32_t index1 = (group << (loc_step + 1)) + group_pos; // 2*group*group_size + group_pos
            uint32_t index2 = index1 + group_size;

            // compute twiddle using the batched->global mapping (same as DIT)
            gl64_t factor; 
            {
                //global_step
                uint32_t gs = base_step -(n_loc_steps -1 - loc_step);
                //global_group_size
                uint32_t ggs = 1 << gs;
                //batched butterfly index
                uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                //global_group_pos
                uint32_t gbi = (((bbi & high_mask)<< (base_step + 1 - n_loc_steps)) + (bbi >> remaining_high_bits)); //undo the batching
                uint32_t ggp = gbi & (ggs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];                
            }

            // DIF: t1 = a + b           -> stored at index1
            //      t2 = (a - b) * W^k   -> stored at index2
            for (int j = 0; j < ncols_block; j++) {
                gl64_t a = tile[j * BATCH_HEIGHT + index1];
                gl64_t b = tile[j * BATCH_HEIGHT + index2];

                gl64_t t1 = a + b;
                gl64_t t2 = a - b;
                t2 = t2 * factor;

                tile[j * BATCH_HEIGHT + index1] = t1;
                tile[j * BATCH_HEIGHT + index2] = t2;
            }
            
        }
        __syncthreads();
    }

    // copy values back to data with scaling applied when this iteration includes the final stage
    if (inverse && (base_step + 1  - n_loc_steps) <= 0) {
        gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size_in]);
        uint64_t row_r = (__brev(row) >> (32 - log_domain_size_in));
        if (extend) inv_factor = inv_factor * d_r[row_r];
        for (int i = 0; i < ncols_block; i++){
            data_block[rowbf * ncols_block + i] = tile[i * BATCH_HEIGHT + threadIdx.x] * inv_factor;
        }
    } else {
        for (int i = 0; i < ncols_block; i++){
            data_block[rowbf * ncols_block + i] = tile[i * BATCH_HEIGHT + threadIdx.x];
        }
    }
}

__global__ void br_ntt_batch_steps_blocks_par_dif_noBR_compact( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(base_step+1, BATCH_HEIGHT_LOG2);
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // Evaluate row as if we shifted BATCH_WIDTH bits after each batch (same packing as DIT version)
    uint32_t groupSize = 1 << (base_step + 1 - n_loc_steps);
    uint32_t nGroups   = domain_size_in / groupSize;
    uint32_t low_bits  = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    // Define my block
    uint32_t block_stride = domain_size_out * BATCH_WIDTH;
    gl64_t *data_block = data + blockIdx.y * block_stride;
    uint32_t col_base = blockIdx.y * BATCH_WIDTH;
    uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? (nCols - col_base) : BATCH_WIDTH;
    uint64_t batch_height_blown = BATCH_HEIGHT / blowupFactor;
    uint64_t blockX = (row / batch_height_blown);
    uint64_t row_block = row % batch_height_blown;
    uint32_t row_comp = blockX * BATCH_HEIGHT + row_block;

    // Copy data to tile
    for (int i = 0; i < ncols_block; i++){
        tile[BATCH_HEIGHT * i + threadIdx.x] = data_block[row_comp * ncols_block + i];
    }
    __syncthreads();


    // remaining_high_bits: number of high bits for butterfly indexing (same meaning as your DIT code)
    uint32_t remaining_high_bits = log_domain_size_in -1 - (base_step + 1 - n_loc_steps); // log(nGroups/2)
    uint32_t high_mask = (1u << remaining_high_bits) - 1u;
    for (int loc_step = n_loc_steps-1; loc_step >= 0; loc_step--){
        uint32_t i = threadIdx.x;
        if (i < BATCH_HEIGHT_DIV2) { // Only process first half (each handles one butterfly)
            uint32_t group_size = 1u << loc_step;
            uint32_t group = i >> loc_step;                       // i / group_size
            uint32_t group_pos = i & (group_size - 1u);           // i % group_size
            uint32_t index1 = (group << (loc_step + 1)) + group_pos; // 2*group*group_size + group_pos
            uint32_t index2 = index1 + group_size;

            // compute twiddle using the batched->global mapping (same as DIT)
            gl64_t factor; 
            {
                //global_step
                uint32_t gs = base_step -(n_loc_steps -1 - loc_step);
                //global_group_size
                uint32_t ggs = 1 << gs;
                //batched butterfly index
                uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                //global_group_pos
                uint32_t gbi = (((bbi & high_mask)<< (base_step + 1 - n_loc_steps)) + (bbi >> remaining_high_bits)); //undo the batching
                uint32_t ggp = gbi & (ggs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];                
            }

            // DIF: t1 = a + b           -> stored at index1
            //      t2 = (a - b) * W^k   -> stored at index2
            for (int j = 0; j < ncols_block; j++) {
                gl64_t a = tile[j * BATCH_HEIGHT + index1];
                gl64_t b = tile[j * BATCH_HEIGHT + index2];

                gl64_t t1 = a + b;
                gl64_t t2 = a - b;
                t2 = t2 * factor;

                tile[j * BATCH_HEIGHT + index1] = t1;
                tile[j * BATCH_HEIGHT + index2] = t2;
            }
            
        }
        __syncthreads();
    }

    // copy values back to data with scaling applied when this iteration includes the final stage
    if (inverse && (base_step + 1  - n_loc_steps) <= 0) {
        gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size_in]);
        uint64_t row_r = (__brev(row) >> (32 - log_domain_size_in));
        if (extend) inv_factor = inv_factor * d_r[row_r];
        uint32_t rowbf = row * blowupFactor;
        for (int i = 0; i < ncols_block; i++){
            data_block[rowbf * ncols_block + i] = tile[i * BATCH_HEIGHT + threadIdx.x] * inv_factor;
            for(uint32_t b=1; b<blowupFactor; b++){
                data_block[(rowbf + b) * ncols_block + i] = gl64_t(uint64_t(0));
            }
        }
    } else {
        for (int i = 0; i < ncols_block; i++){
            data_block[row_comp * ncols_block + i] = tile[i * BATCH_HEIGHT + threadIdx.x];
        }
    }
}


__global__ void br_ntt_batch_steps_blocks_par2(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(log_domain_size_in - base_step, BATCH_HEIGHT_LOG2);    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    //evaluate row as if I shifted BATCH_WIDTH bits after each batch
    uint32_t groupSize = 1 << base_step;
    uint32_t nGroups = domain_size_in / groupSize;
    uint32_t low_bits = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    //remaining steps
    uint32_t remaining_high_bits = log_domain_size_in - (base_step+1); //log(nGroups / 2)
    uint32_t high_mask = (1 << remaining_high_bits) - 1; 
    uint32_t offset = domain_size_out * BATCH_WIDTH;

    uint32_t block=blockIdx.y;
    gl64_t *data_block = data + block*offset;
    uint32_t col_base = block * BATCH_WIDTH;
    uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? nCols - col_base : BATCH_WIDTH;
    if(threadIdx.y >= ncols_block) return;
    //copy data to tile 
    tile[BATCH_HEIGHT*threadIdx.y+threadIdx.x] = data_block[row*ncols_block+threadIdx.y];
    
    
    __syncthreads();

    for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
        uint32_t i = threadIdx.x;
        if (threadIdx.x < BATCH_HEIGHT_DIV2){ // Only process first (half of them)
            uint32_t group_size = 1 << loc_step;   
            uint32_t group = i >> loc_step;                          // i/(group_size)    
            uint32_t group_pos = i & (group_size - 1);               // i%(group_size)  
            uint32_t index1 = (group << (loc_step + 1)) + group_pos; // 2*group*group_size + group_pos
            uint32_t index2 = index1 + group_size;
            gl64_t factor;
            {
                //global_step
                uint32_t gs = base_step + loc_step;
                //global_group_size
                uint32_t ggs = 1 << gs;
                //batched butterfly index
                uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                //global_group_pos
                uint32_t gbi = (((bbi & high_mask)<< base_step) + (bbi >> remaining_high_bits)); //undo the batching
                uint32_t ggp = gbi & (ggs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];
            }
            gl64_t odd_sub = tile[ threadIdx.y*BATCH_HEIGHT + index2] * factor;
            tile[threadIdx.y*BATCH_HEIGHT +index2] = tile[threadIdx.y*BATCH_HEIGHT + index1] - odd_sub;               
            tile[threadIdx.y*BATCH_HEIGHT +index1] = tile[threadIdx.y*BATCH_HEIGHT + index1] + odd_sub;                
                                         
        }
        __syncthreads();
    }
    // copy values to data with scaling applied when this iteration includes the final stage
    if(inverse && (base_step + n_loc_steps) >= log_domain_size_in){
        gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size_in]);
        if(extend) inv_factor = inv_factor * d_r[row];
        data_block[row*ncols_block+threadIdx.y] = tile[threadIdx.y*BATCH_HEIGHT+threadIdx.x] * inv_factor;
    }else{
        data_block[row*ncols_block+threadIdx.y] = tile[threadIdx.y*BATCH_HEIGHT+threadIdx.x]; 
    } 
}

__global__ void br_ntt_8_steps_blocks(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(log_domain_size_in - base_step, BATCH_HEIGHT_LOG2);    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    //evaluate row as if I shifted BATCH_WIDTH bits after each batch
    uint32_t groupSize = 1 << base_step;
    uint32_t nGroups = domain_size_in / groupSize;
    uint32_t low_bits = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    //remaining steps
    uint32_t remaining_high_bits = log_domain_size_in - (base_step+1); //log(nGroups / 2)
    uint32_t high_mask = (1 << remaining_high_bits) - 1; 
    uint32_t nblocks = (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH;
    uint32_t offset = domain_size_out * BATCH_WIDTH; 

    for(int block = 0; block < nblocks; block++){
        gl64_t *data_block = data + block*offset;
        uint32_t col_base = block * BATCH_WIDTH;
        uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? nCols - col_base : BATCH_WIDTH;
        //copy data to tile 
       for(int i=0; i<ncols_block; i++){
            tile[threadIdx.x*BATCH_WIDTH+i] = data_block[row*ncols_block+i];
       }
        
        __syncthreads();

        for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
            uint32_t i = threadIdx.x;
            if (threadIdx.x < BATCH_HEIGHT_DIV2){ // Only process first(half of them)
                uint32_t group_size = 1 << loc_step;   
                uint32_t group = i >> loc_step;                          // i/(group_size)    
                uint32_t group_pos = i & (group_size - 1);               // i%(group_size)  
                uint32_t index1 = (group << (loc_step + 1)) + group_pos; // 2*group*group_size + group_pos
                uint32_t index2 = index1 + group_size;
                gl64_t factor;
                {
                    //global_step
                    uint32_t gs = base_step + loc_step;
                    //global_group_size
                    uint32_t ggs = 1 << gs;
                    //batched butterfly index
                    uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                    //global_group_pos
                    uint32_t gbi = (((bbi & high_mask)<< base_step) + (bbi >> remaining_high_bits)); //undo the batching
                    uint32_t ggp = gbi & (ggs - 1);
                    factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];
                }
                index1 = index1 * BATCH_WIDTH;
                index2 = index2 * BATCH_WIDTH;
                for(int j=0; j<ncols_block; j++){
                    gl64_t odd_sub = tile[index2 + j] * factor;
                    tile[index2 + j] = tile[index1 + j] - odd_sub;               
                    tile[index1 + j] = tile[index1 + j] + odd_sub;                
                }                             
            }
            __syncthreads();
        }
        // copy values to data with scaling applied when this iteration includes the final stage
        if(inverse && (base_step + n_loc_steps) >= log_domain_size_in){
            gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size_in]);
            if(extend) inv_factor = inv_factor * d_r[row];
           for(int i=0; i<ncols_block; i++){
                data_block[row*ncols_block+i] = tile[threadIdx.x*BATCH_WIDTH+i] * inv_factor;
           }
        }else{
           for(int i=0; i<ncols_block; i++){
                data_block[row*ncols_block+i] = tile[threadIdx.x*BATCH_WIDTH+i];
           }
        }
    }   
}

__global__ void reverse_permutation_new(gl64_t *data, uint32_t log_domain_size, uint32_t nCols)
{
    uint64_t row = blockIdx.x;
    uint64_t col = threadIdx.x;
    uint64_t domain_size = 1 << log_domain_size;

    for (uint64_t r = row; r < domain_size; r += gridDim.x)
    {
        uint64_t rowr = __brev(r) >> (32 - log_domain_size);
        if (rowr > r)
        {
            for (uint64_t c = col; c < nCols; c += blockDim.x)
            {
                gl64_t tmp = data[r * nCols + c];
                data[r * nCols + c] = data[rowr * nCols + c];
                data[rowr * nCols + c] = tmp;
            }
        }
    }
}

__global__ void reverse_permutation_blocks(gl64_t *data, uint32_t log2_domain_size_in, uint64_t domain_size_out, uint32_t nCols)
{
    uint64_t row = blockIdx.x;
    uint64_t ncols_block = (nCols - BATCH_WIDTH*blockIdx.y) < BATCH_WIDTH ? nCols - blockIdx.y * BATCH_WIDTH : BATCH_WIDTH;
    uint64_t domain_size_in = 1 << log2_domain_size_in;
    uint64_t offset = blockIdx.y * BATCH_WIDTH * domain_size_out;
    gl64_t *data_block = data + offset;

    if (threadIdx.x >= ncols_block) return;

    for (uint64_t r = row; r < domain_size_in; r += gridDim.x)
    {
        uint64_t rowr = (__brev(r) >> (32 - log2_domain_size_in));
        if (rowr > r) 
        {
            gl64_t tmp = data_block[r * ncols_block + threadIdx.x];
            data_block[r * ncols_block + threadIdx.x] = data_block[rowr * ncols_block + threadIdx.x];
            data_block[rowr * ncols_block + threadIdx.x] = tmp;   
        }
    }
}

__global__ void reverse_permutation_blocks_noBR(gl64_t *data, uint32_t log2_domain_size_in, uint64_t domain_size_out, uint32_t nCols, uint32_t blowupFactor)
{
    uint64_t row = blockIdx.x;
    uint64_t ncols_block = (nCols - BATCH_WIDTH*blockIdx.y) < BATCH_WIDTH ? nCols - blockIdx.y * BATCH_WIDTH : BATCH_WIDTH;
    uint64_t domain_size_in = 1 << log2_domain_size_in;
    uint64_t offset = blockIdx.y * BATCH_WIDTH * domain_size_out;
    gl64_t *data_block = data + offset;

    if (threadIdx.x >= ncols_block) return;

    for (uint64_t r = row; r < domain_size_in; r += gridDim.x)
    {
        uint64_t rowr = (__brev(r) >> (32 - log2_domain_size_in));
        if (rowr > r) 
        {
            uint32_t rbf = r * blowupFactor;
            uint32_t rrbf = rowr * blowupFactor;
            gl64_t tmp = data_block[rbf * ncols_block + threadIdx.x];
            data_block[rbf * ncols_block + threadIdx.x] = data_block[rrbf * ncols_block + threadIdx.x];
            data_block[rrbf * ncols_block + threadIdx.x] = tmp;   
        }
    }
}

__global__ void eval_twiddle_factors_small_size(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    gl64_t omega = gl64_t(omegas[log_domain_size]);
    gl64_t omega_inv = gl64_t(omegas_inv[log_domain_size]);

    fwd_twiddles[0] = gl64_t(uint64_t(1));
    inv_twiddles[0] = gl64_t(uint64_t(1));

    for (uint32_t i = 1; i < 1 << (log_domain_size - 1); i++)
    {
        fwd_twiddles[i] = fwd_twiddles[i - 1] * omega;
        inv_twiddles[i] = inv_twiddles[i - 1] * omega_inv;
    }
}

__global__ void eval_twiddle_factors_first_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    gl64_t omega = gl64_t(omegas[log_domain_size]);
    gl64_t omega_inv = gl64_t(omegas_inv[log_domain_size]);

    fwd_twiddles[0] = gl64_t(uint64_t(1));
    inv_twiddles[0] = gl64_t(uint64_t(1));

    for (uint32_t i = 1; i <= (1 << 12); i++)
    {
        fwd_twiddles[i] = fwd_twiddles[i - 1] * omega;
        inv_twiddles[i] = inv_twiddles[i - 1] * omega_inv;
    }
}

__global__ void eval_twiddle_factors_second_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = 1; i < 1 << log_domain_size - 13; i++)
    {
        fwd_twiddles[i * 4096 + idx] = fwd_twiddles[(i - 1) * 4096 + idx] * fwd_twiddles[4096];
        inv_twiddles[i * 4096 + idx] = inv_twiddles[(i - 1) * 4096 + idx] * inv_twiddles[4096];
    }
}

void eval_twiddle_factors(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size, cudaStream_t stream)
{
    if (log_domain_size <= 13)
    {
        eval_twiddle_factors_small_size<<<1, 1, 0, stream>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
    else
    {
        eval_twiddle_factors_first_step<<<1, 1, 0, stream>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
        eval_twiddle_factors_second_step<<<(1 << 12), 1, 0, stream>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
}

__global__ void eval_r_small_size(gl64_t *r, uint32_t log_domain_size)
{
    r[0] = gl64_t(uint64_t(1));
    for (uint32_t i = 1; i < 1 << log_domain_size; i++)
    {
        r[i] = r[i - 1] * gl64_t(SHIFT);
    }
}

__global__ void eval_r_first_step(gl64_t *r, uint32_t log_domain_size)
{
    r[0] = gl64_t(uint64_t(1));
    // init first 4097 elements and then init others in parallel
    for (uint32_t i = 1; i <= 1 << 12; i++)
    {
        r[i] = r[i - 1] * gl64_t(SHIFT);
    }
}

__global__ void eval_r_second_step(gl64_t *r, uint32_t log_domain_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = 1; i < 1 << log_domain_size - 12; i++)
    {
        r[i * 4096 + idx] = r[(i - 1) * 4096 + idx] * r[4096];
    }
}

void eval_r(gl64_t *r, uint32_t log_domain_size, cudaStream_t stream)
{
    if (log_domain_size <= 12)
    {
        eval_r_small_size<<<1, 1, 0, stream>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
    else
    {
        eval_r_first_step<<<1, 1, 0, stream>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
        eval_r_second_step<<<(1 << 12), 1, 0, stream>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
}

void ntt_cuda( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{   
    uint32_t domain_size = 1 << log_domain_size;

    dim3 blockDim;
    dim3 gridDim;
    
    blockDim = dim3(TPB_NTT);
    gridDim = dim3(8192);
    reverse_permutation_new<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size, nCols);
    CHECKCUDAERR(cudaGetLastError());

    // Get device ID and twiddle factors once
    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call init_twiddle_factors()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];

    if(log_domain_size >= 8) {
         for(uint32_t step = 0; step < log_domain_size; step+=8){
                br_ntt_8_steps<<<domain_size / 256, 256, 0, stream>>>(data, d_twiddles, d_r, domain_size, log_domain_size, nCols, step, true, inverse, extend, maxLogDomainSize, 0, nCols-1);
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        for (uint32_t stage = 0; stage < log_domain_size; stage++)
        {
            br_ntt_group<<<domain_size / 2, nCols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size, log_domain_size, nCols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

}

void ntt_cuda_blocks_par( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{   

    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    dim3 blockDim;
    dim3 gridDim;
    blockDim = dim3(BATCH_WIDTH);
    gridDim = dim3(4096,(nCols + BATCH_WIDTH - 1) / BATCH_WIDTH);
    reverse_permutation_blocks<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size_in, domain_size_out, nCols);
    CHECKCUDAERR(cudaGetLastError());

    // Get device ID and twiddle factors once
    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call init_twiddle_factors()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];
    

    if(log_domain_size_in >= BATCH_HEIGHT_LOG2 ) {
         for(uint32_t step = 0; step < log_domain_size_in; step+=BATCH_HEIGHT_LOG2){
                dim3 blocks = dim3(domain_size_in / BATCH_HEIGHT, (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH, 1);
                dim3 threads = dim3(BATCH_HEIGHT,1,1);
                br_ntt_batch_steps_blocks_par<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, nCols, step, true, inverse, extend, maxLogDomainSize);
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        assert(0);
        for (uint32_t stage = 0; stage < log_domain_size_in; stage++)
        {
            br_ntt_group_blocks<<<domain_size_in / 2, nCols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size_in, log_domain_size_in, domain_size_out, nCols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

}

void ntt_cuda_blocks_par_dif( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{   

    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    // Get device ID and twiddle factors once
    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call init_twiddle_factors()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];
    

    if(log_domain_size_in >= BATCH_HEIGHT_LOG2) {
        for(int step = log_domain_size_in-1; step >= 0; step-=BATCH_HEIGHT_LOG2){
                dim3 blocks = dim3(domain_size_in / BATCH_HEIGHT, (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH, 1);
                dim3 threads = dim3(BATCH_HEIGHT,1,1);
                br_ntt_batch_steps_blocks_par_dif<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, nCols, step, true, inverse, extend, maxLogDomainSize);
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        for (int step = log_domain_size_in - 1; step >= 0; step--)
        {
            br_ntt_group_blocks_dif<<<domain_size_in / 2, nCols, 0, stream>>>(data, d_twiddles, d_r, step, domain_size_in, log_domain_size_in, domain_size_out, nCols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

    dim3 blockDim;
    dim3 gridDim;
    blockDim = dim3(BATCH_WIDTH);
    gridDim = dim3(4096,(nCols + BATCH_WIDTH - 1) / BATCH_WIDTH);
    reverse_permutation_blocks<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size_in, domain_size_out, nCols);
    CHECKCUDAERR(cudaGetLastError());


}

void ntt_cuda_blocks_par1_noBR( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{   

    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    // Get device ID and twiddle factors once
    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call init_twiddle_factors()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];
    

    if(log_domain_size_in >= BATCH_HEIGHT_LOG2 ) {
         for(int step = log_domain_size_in-1; step >= 0; step-=BATCH_HEIGHT_LOG2){
                dim3 blocks = dim3(domain_size_in / BATCH_HEIGHT, (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH, 1);
                dim3 threads = dim3(BATCH_HEIGHT,1,1);
                br_ntt_batch_steps_blocks_par_dif_noBR<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, nCols, step, true, inverse, extend, maxLogDomainSize, (1 << (log_domain_size_out - log_domain_size_in)));
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        assert(0);
        for (uint32_t stage = 0; stage < log_domain_size_in; stage++)
        {
            br_ntt_group_blocks<<<domain_size_in / 2, nCols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size_in, log_domain_size_in, domain_size_out, nCols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

}

void ntt_cuda_blocks_par1_noBR_compact( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{   

    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    // Get device ID and twiddle factors once
    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call init_twiddle_factors()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];
    

    if(log_domain_size_in >= BATCH_HEIGHT_LOG2 ) {
         for(int step = log_domain_size_in-1; step >= 0; step-=BATCH_HEIGHT_LOG2){
                dim3 blocks = dim3(domain_size_in / BATCH_HEIGHT, (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH, 1);
                dim3 threads = dim3(BATCH_HEIGHT,1,1);
                br_ntt_batch_steps_blocks_par_dif_noBR_compact<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, nCols, step, true, inverse, extend, maxLogDomainSize, (1 << (log_domain_size_out - log_domain_size_in)));
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        assert(0);
        for (uint32_t stage = 0; stage < log_domain_size_in; stage++)
        {
            br_ntt_group_blocks<<<domain_size_in / 2, nCols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size_in, log_domain_size_in, domain_size_out, nCols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

}

void ntt_cuda_blocks_par2_noBR( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{   

    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    // Get device ID and twiddle factors once
    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call init_twiddle_factors()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];
    

    if(log_domain_size_in >= BATCH_HEIGHT_LOG2 ) {
         for(uint32_t step = 0; step < log_domain_size_in; step+=BATCH_HEIGHT_LOG2){
                dim3 blocks = dim3(domain_size_in / BATCH_HEIGHT, (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH, 1);
                dim3 threads = dim3(BATCH_HEIGHT,1,1);
                br_ntt_batch_steps_blocks_par<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, nCols, step, true, inverse, extend, maxLogDomainSize);
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        assert(0);
        for (uint32_t stage = 0; stage < log_domain_size_in; stage++)
        {
            br_ntt_group_blocks<<<domain_size_in / 2, nCols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size_in, log_domain_size_in, domain_size_out, nCols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

}