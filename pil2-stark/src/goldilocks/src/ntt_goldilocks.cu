#include "ntt_goldilocks.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_tooling.cuh"
#include "poseidon2_goldilocks.cuh"
#include "ntt_goldilocks.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "omp.h"
#include "poseidon_goldilocks.hpp"
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

__global__ void br_ntt_8_steps(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t col_min, uint32_t col_max);
__global__ void br_ntt_8_steps_blocks(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t ncols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void br_ntt_8_steps_blocks_par(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t ncols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize);

__global__ void br_ntt_group(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void br_ntt_group_new(gl64_t *data, gl64_t *twiddles, uint32_t i, uint32_t domain_size, uint32_t ncols, uint64_t maxLogDomainSize);
__global__ void intt_scale(gl64_t *data, gl64_t *r, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, bool extend);
__global__ void reverse_permutation_new(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void reverse_permutation_column(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void reverse_permutation(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void reverse_permutation_1d(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void reverse_permutation_2d(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void eval_twiddle_factors_small_size(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void eval_twiddle_factors_first_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void eval_twiddle_factors_second_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
void eval_twiddle_factors(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size, cudaStream_t stream);
__global__ void eval_r_small_size(gl64_t *r, uint32_t log_domain_size);
__global__ void eval_r_first_step(gl64_t *r, uint32_t log_domain_size);
__global__ void eval_r_second_step(gl64_t *r, uint32_t log_domain_size);
void eval_r(gl64_t *r, uint32_t log_domain_size, cudaStream_t stream);
void ntt_cuda( gl64_t *data, gl64_t **d_r, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size, uint32_t ncols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
void ntt_cuda_blocks( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t ncols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize, gl64_t *aux_data);
void ntt_cuda_blocks_par( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t ncols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize, gl64_t *aux_data);

__global__ void printDataKernel_(gl64_t *data, uint32_t domain_size, uint32_t ncols){
    Goldilocks::Element * data_print = (Goldilocks::Element *) data;

    printf("Abans Domain size %u, ncols %u\n", domain_size, ncols);
    for( int i = 0; i < domain_size*ncols; ++i){
        printf("%lu, ", data_print[i].fe);
        
    }
    printf("\n");
} 

__global__ void printDataKernel(gl64_t *data, uint32_t domain_size, uint32_t ncols){
    Goldilocks::Element * data_print = (Goldilocks::Element *) data;

    printf("Domain size %u, ncols %u\n", domain_size, ncols);
    for( int i = 0; i < domain_size*ncols; ++i){
        printf("%lu, ", data_print[i].fe);
        
    }
    printf("\n");
} 

__global__ void applyS(gl64_t *d_cmQ, gl64_t *d_q, gl64_t *d_S, Goldilocks::Element shiftIn, uint64_t N, uint64_t qDeg, uint64_t qDim)
{
    d_S[0] = gl64_t(uint64_t(1));
    for(uint64_t i = 1; i < qDeg; ++i) {
        d_S[i] = gl64_t(shiftIn.fe) * d_S[i - 1];
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    for (uint64_t p = 0; p < qDeg; p++)
    {        
        Goldilocks3GPU::mul((Goldilocks3GPU::Element &)d_cmQ[(i * qDeg + p) * FIELD_EXTENSION],
                            (Goldilocks3GPU::Element &)d_q[(p * N + i) * FIELD_EXTENSION],
                            d_S[p]);
    }
}

void NTT_Goldilocks_GPU::computeQ_inplace(Goldilocks::Element *d_tree, uint64_t offset_cmQ, uint64_t offset_q, uint64_t qDeg, uint64_t qDim, Goldilocks::Element shiftIn, uint64_t N, uint64_t n_bits_ext, uint64_t ncols, gl64_t *d_aux_trace, uint64_t offset_helper, TimerGPU &timer, cudaStream_t stream)
{
   
    if (ncols == 0 || n_bits_ext == 0)
    {
        return;
    }

    TimerStartCategoryGPU(timer, NTT);

    if(n_bits_ext > maxLogDomainSize)
    {
        printf("[NTT] ERROR: n_bits_ext %lu exceeds maxLogDomainSize %lu\n", n_bits_ext, maxLogDomainSize);
        abort();
    }

    uint64_t NExtended = 1 << n_bits_ext;
    gl64_t* d_S = d_aux_trace + offset_helper;
    gl64_t *d_q = d_aux_trace + offset_q;
    gl64_t *d_cmQ = d_aux_trace + offset_cmQ;


    // Intt
    ntt_cuda(d_q, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, qDim, true, false, stream, maxLogDomainSize);


    dim3 threads(128, 1, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, 1, 1);
    applyS<<<blocks, threads, 0, stream>>>(d_cmQ, d_q, d_S, shiftIn, N, qDeg, qDim);
    CHECKCUDAERR(cudaMemsetAsync(d_cmQ + N * qDeg * qDim, 0, (NExtended - N) * qDeg * qDim * sizeof(gl64_t), stream));


    ntt_cuda(d_cmQ, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, ncols, false, false, stream, maxLogDomainSize);
    TimerStopCategoryGPU(timer, NTT);
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    Poseidon2GoldilocksGPU::merkletree_cuda_coalesced(3, (uint64_t*) d_tree, (uint64_t *)d_cmQ, ncols, NExtended, stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);
}

//assume inputs blocks are of size 32x4 by now
__global__ void prepareBlocksInput(gl64_t * dst, gl64_t * src, uint64_t n, uint64_t ncols)
{
    extern __shared__ gl64_t shared[];
    
    int row_read = blockIdx.x * blockDim.x + threadIdx.x; 
    int col_read= blockIdx.y * blockDim.y + threadIdx.y;
    if (row_read >= n || col_read >= ncols)
        return;
    int row_base = blockIdx.x * blockDim.x;
    int col_base = blockIdx.y * blockDim.y;
    int ndest = n << 1;
    int sharedIdx = threadIdx.y * blockDim.x + threadIdx.x;
    shared[sharedIdx] = src[col_read * n + row_read];
    __syncthreads();

    int block_ncols = (ncols - col_base) < 4 ? ncols - col_base : 4;
    int shared_row_write = sharedIdx / block_ncols;
    int shared_col_write = sharedIdx % block_ncols;
    int sharedIdx_write = shared_col_write * blockDim.x + shared_row_write;
    int row_write = row_base + shared_row_write;
    int col_write = col_base + shared_col_write;

    dst[blockIdx.y * 4 * ndest + row_write * block_ncols + col_write] = shared[sharedIdx_write];
    dst[blockIdx.y * 4 * ndest + row_write * block_ncols + col_write + n * block_ncols] = gl64_t(uint64_t(0));
}

__global__ void prepareBlocksInputRowMajor(gl64_t * dst, gl64_t * src, uint64_t n, uint64_t ncols)
{
    extern __shared__ gl64_t shared[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= ncols)
        return;
    int row_base = blockIdx.x * blockDim.x;
    int col_base = blockIdx.y * blockDim.y;
    int ndest = n << 1;
    int block_ncols = (ncols - col_base) < 4 ? ncols - col_base : 4;
    int sharedIdx = threadIdx.y * blockDim.x + threadIdx.x;
    shared[sharedIdx] = src[row * ncols + col];
    __syncthreads();
    int out_idx = blockIdx.y * 4 * ndest + row * block_ncols + threadIdx.y;
    dst[out_idx] = shared[sharedIdx];
    dst[out_idx + n * block_ncols] = gl64_t(uint64_t(0));
    
}

//assume blocks are in 32 x 4
__global__ void fromBlocksToRowMajor(gl64_t * dst, gl64_t * src, uint64_t n, uint64_t ncols)
{    
    uint32_t block_id = blockIdx.y;
    uint32_t offset = block_id * 4 * (n * 2); 
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= ncols)
        return;
    uint32_t block_ncols = (ncols - block_id * 4) < 4 ? (ncols - block_id * 4) : 4;
    dst[row*ncols + col] = src[offset + row * block_ncols + threadIdx.y];
    return;
}

__global__ void compareResults(gl64_t* res1, gl64_t* res2, uint64_t n, uint64_t ncols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int block_id = col >> 2;
    int block_col = col & 3;
    int block_ncols = (ncols - block_id * 4) < 4 ? (ncols - block_id * 4) : 4;
    if (row >= n || col >= ncols)
        return;
    Goldilocks::Element * res1_data = (Goldilocks::Element *) res1;
    Goldilocks::Element * res2_data = (Goldilocks::Element *) res2;
    Goldilocks::Element val1 = res1_data[row * ncols + col];
    Goldilocks::Element val2 = res2_data[block_id * 4 * n + row * block_ncols + block_col];

    if (val1.fe != val2.fe)
    {
        printf("Difference at row %d, col %d, block_id %d, block_col %d: %lu != %lu\n", row, col, block_id, block_col, val1.fe, val2.fe);
        assert(0);
    }
}

__global__ void compareResults_col(gl64_t* res1, gl64_t* res2, uint64_t n, uint64_t ncols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int block_id = col >> 2;
    int block_col = col & 3;
    int sub_block_id = row >> 8;
    int sub_block_row = row & 255;
    int block_stride = n * 4;
    int block_ncols = (ncols - block_id * 4) < 4 ? (ncols - block_id * 4) : 4;
    int sub_block_stride = 256 * block_ncols;

    if (row >= n || col >= ncols)
        return;
    Goldilocks::Element * res1_data = (Goldilocks::Element *) res1;
    Goldilocks::Element * res2_data = (Goldilocks::Element *) res2;
    Goldilocks::Element val1 = res1_data[row * ncols + col];
    Goldilocks::Element val2 = res2_data[block_id * block_stride + sub_block_id * sub_block_stride + block_col * 256 + sub_block_row];

    if (val1.fe != val2.fe)
    {
        printf("Difference at row %d, col %d, block_id %d, block_col %d: %lu != %lu\n", row, col, block_id, block_col, val1.fe, val2.fe);
        assert(0);
    }
}

//assume is launched in blocks of 256x4 that cover the whole matrix
__global__ void transposeSubBlocksInPlace(gl64_t * data, uint64_t n, uint64_t n_ext, uint64_t ncols)
{
    int sublock_ncols = (ncols - blockIdx.y * 4) < 4 ? (ncols - blockIdx.y * 4) : 4;
    int offset = blockIdx.y * 4 * n_ext + blockIdx.x * 256 * sublock_ncols;
    if( threadIdx.y >= sublock_ncols)
        return;
    extern __shared__ gl64_t shared[];   
    shared[threadIdx.x * 4 + threadIdx.y] = data[offset + threadIdx.x * sublock_ncols + threadIdx.y];
    __syncthreads();
    data[offset + threadIdx.y*256 + threadIdx.x] = shared[threadIdx.x * 4 + threadIdx.y];
    if( n != n_ext){
        int offset2 = offset + gridDim.x * 256 * sublock_ncols;
        data[offset2 + threadIdx.y*256 + threadIdx.x] = gl64_t(uint64_t(0));
    }

}
__global__ void transposeSubBlocksInPlaceBack(gl64_t * data, uint64_t n, uint64_t n_ext, uint64_t ncols)
{
    int sublock_ncols = (ncols - blockIdx.y * 4) < 4 ? (ncols - blockIdx.y * 4) : 4;
    int offset = blockIdx.y * 4 * n_ext + blockIdx.x * 256 * sublock_ncols;
    if( threadIdx.y >= sublock_ncols)
        return;
    extern __shared__ gl64_t shared[];   
    shared[threadIdx.x * 4 + threadIdx.y] = data[offset + threadIdx.y*256 + threadIdx.x];
    __syncthreads();
    data[offset + threadIdx.x * sublock_ncols + threadIdx.y] = shared[threadIdx.x * 4 + threadIdx.y];
    
    if( n != n_ext){
        int offset2 = offset + gridDim.x * 256 * sublock_ncols;
        data[offset2 + threadIdx.y*256 + threadIdx.x] = gl64_t(uint64_t(0));
    }

}

void NTT_Goldilocks_GPU::LDE_MerkleTree_GPU_inplace(Goldilocks::Element *d_tree, gl64_t *d_dst_ntt, uint64_t offset_dst_ntt, gl64_t *d_src_ntt, uint64_t offset_src_ntt, u_int64_t n_bits, u_int64_t n_bits_ext, u_int64_t ncols, TimerGPU &timer, cudaStream_t stream)
{
    if (ncols == 0 || n_bits == 0)
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

#if 0
    gl64_t *d_aux;
    cudaMalloc(&d_aux, ext_size * ncols * sizeof(gl64_t));
    cudaMemset(d_aux, 0, ext_size * ncols * sizeof(gl64_t));
    dim3 block_0(256, 4);
    dim3 grid_0((size + block_0.x - 1) / block_0.x,
             (ncols + block_0.y - 1) / block_0.y);
    int sharedMemSize_0 = block_0.x * block_0.y * sizeof(gl64_t);

    dim3 block(32, 4);
    dim3 grid((size + block.x - 1) / block.x,
             (ncols + block.y - 1) / block.y);
    size_t sharedMemSize = block.x * block.y * sizeof(gl64_t);
    prepareBlocksInputRowMajor<<<grid, block, sharedMemSize, stream>>>(d_aux, d_src_ntt_, size, ncols);
    transposeSubBlocksInPlace<<<grid_0, block_0, sharedMemSize_0, stream>>>(d_aux, size, ext_size, ncols);
    transposeSubBlocksInPlaceBack<<<grid_0, block_0, sharedMemSize_0, stream>>>(d_aux, size, ext_size, ncols);

    CHECKCUDAERR(cudaMemcpyAsync(d_dst_ntt_, d_src_ntt_, size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToDevice, stream));
    CHECKCUDAERR(cudaMemsetAsync(d_dst_ntt_ + size * ncols, 0, (ext_size - size) * ncols * sizeof(gl64_t), stream));

    ntt_cuda(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, ncols, true, true, stream, maxLogDomainSize);
    ntt_cuda_blocks_par(d_aux, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, n_bits_ext, ncols, true, true, stream, maxLogDomainSize, d_src_ntt);

    
    dim3 block_(16, 4);
    dim3 grid_((ext_size + block_.x - 1) / block_.x,
             (ncols + block_.y - 1) / block_.y);
    compareResults<<<grid_, block_, 0, stream>>>(d_dst_ntt_, d_aux, ext_size, ncols);

    ntt_cuda(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, ncols, false, false, stream, maxLogDomainSize);
    ntt_cuda_blocks_par(d_aux, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, n_bits_ext, ncols, false, false, stream, maxLogDomainSize, d_src_ntt);
    dim3 block_1(256, 4);
    dim3 grid_1((ext_size + block_1.x - 1) / block_1.x,
             (ncols + block_1.y - 1) / block_1.y);
    int sharedMemSize_1 = block_1.x * block_1.y * sizeof(gl64_t);
    transposeSubBlocksInPlace<<<grid_1, block_1, sharedMemSize_1, stream>>>(d_aux, ext_size, ext_size, ncols);
    compareResults_col<<<grid_, block_, 0, stream>>>(d_dst_ntt_, d_aux, ext_size, ncols);

    cudaStreamSynchronize(stream);
    cudaFree(d_aux);
#endif
#if 1
    dim3 block_0(256, 4);
    dim3 grid_0((size + block_0.x - 1) / block_0.x,
             (ncols + block_0.y - 1) / block_0.y);
    int sharedMemSize_0 = block_0.x * block_0.y * sizeof(gl64_t);
    transposeSubBlocksInPlaceBack<<<grid_0, block_0, sharedMemSize_0, stream>>>(d_dst_ntt, size, ext_size, ncols);
    ntt_cuda_blocks_par(d_dst_ntt, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, n_bits_ext, ncols, true, true, stream, maxLogDomainSize, d_src_ntt); 
    ntt_cuda_blocks_par(d_dst_ntt, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, n_bits_ext, ncols, false, false, stream, maxLogDomainSize, d_src_ntt);
    dim3 block_1(256, 4);
    dim3 grid_1((ext_size + block_1.x - 1) / block_1.x,
             (ncols + block_1.y - 1) / block_1.y);
    int sharedMemSize_ = block_1.x * block_1.y * sizeof(gl64_t);
    transposeSubBlocksInPlace<<<grid_1, block_1, sharedMemSize_, stream>>>(d_src_ntt, ext_size, ext_size, ncols);

#endif
    
#if 0
    CHECKCUDAERR(cudaMemcpyAsync(d_dst_ntt_, d_src_ntt_, size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToDevice, stream));
    CHECKCUDAERR(cudaMemsetAsync(d_dst_ntt_ + size * ncols, 0, (ext_size - size) * ncols * sizeof(gl64_t), stream));
    ntt_cuda(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, ncols, true, true, stream, maxLogDomainSize);
    ntt_cuda(d_dst_ntt_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits_ext, ncols, false, false, stream, maxLogDomainSize);
#endif

    TimerStopCategoryGPU(timer, NTT);
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    Poseidon2GoldilocksGPU::merkletree_cuda_coalesced(3, (uint64_t*) d_tree, (uint64_t *)d_dst_ntt_, ncols, ext_size, stream);
    TimerStopCategoryGPU(timer, MERKLE_TREE);
}

void NTT_Goldilocks_GPU::INTT_inplace(uint64_t data_offset, u_int64_t n_bits, u_int64_t ncols, gl64_t *d_aux_trace, uint64_t offset_helper, gl64_t* d_data, cudaStream_t stream)
{
    if (ncols == 0 || n_bits == 0)
    {
        return;
    }
    if (n_bits > maxLogDomainSize)
    {
        printf("[NTT] ERROR: n_bits %lu exceeds maxLogDomainSize %lu\n", n_bits, maxLogDomainSize);
        abort();
    }

    gl64_t *dst_src = d_data == nullptr ? d_aux_trace + data_offset : d_data;
    ntt_cuda(dst_src, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, n_bits, ncols, true, false, stream, maxLogDomainSize);
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

__global__ void br_ntt_group(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    uint32_t i = blockIdx.x;
    uint32_t col = threadIdx.x;

    if (i < domain_size / 2 && col < ncols)
    {
        uint32_t half_group_size = 1 << stage;
        uint32_t group = i >> stage;                          // i/(half_group_size)
        uint32_t group_pos = i & (half_group_size - 1);       // i%(half_group_size)
        uint32_t index1 = (group << (stage + 1)) + group_pos; // stage + 1 is sizeof of group
        uint32_t index2 = index1 + half_group_size;
        gl64_t factor = twiddles[group_pos * ((1 << maxLogDomainSize) >> (stage + 1))];  // Use actual domain size
        gl64_t odd_sub = gl64_t((uint64_t)data[index2 * ncols + col]) * factor;
        gl64_t result1 = gl64_t((uint64_t)data[index1 * ncols + col]) + odd_sub;
        gl64_t result2 = gl64_t((uint64_t)data[index1 * ncols + col]) - odd_sub;
        
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
        
        data[index1 * ncols + col] = result1;
        data[index2 * ncols + col] = result2;
    }
}

__global__ void br_ntt_group_blocks(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t domain_size_out, uint32_t ncols, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    uint32_t i = blockIdx.x;
    uint32_t col = threadIdx.x;
    uint32_t block_idx = col >> 2;
    uint32_t block_col = col & 3;
    uint32_t ncols_block = (ncols - block_idx * 4) < 4 ? (ncols - block_idx * 4) : 4;
    gl64_t* data_block = data + block_idx * 4 * domain_size_out;

    if (i < domain_size / 2 && col < ncols)
    {
        uint32_t half_group_size = 1 << stage;
        uint32_t group = i >> stage;                          // i/(half_group_size)
        uint32_t group_pos = i & (half_group_size - 1);       // i%(half_group_size)
        uint32_t index1 = (group << (stage + 1)) + group_pos; // stage + 1 is sizeof of group
        uint32_t index2 = index1 + half_group_size;
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

__global__ void br_ntt_8_steps(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t col_min, uint32_t col_max)
{
    __shared__ gl64_t tile[1024];

    //assume domain_size is multiple of 256
    uint32_t n_loc_steps = min(log_domain_size - base_step, 8);    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    //evaluate row as if I shited 8 bits after each batch
    uint32_t accumBatchSize = 1 << base_step;
    uint32_t nBatches = domain_size / accumBatchSize;
    uint32_t low_bits = row / nBatches;
    uint32_t up_bits = row % nBatches;
    row = up_bits * accumBatchSize + low_bits;

    //remaining steps
    uint32_t remaining_steps = log_domain_size - (base_step+1); 
    uint32_t remaining_msk = (1 << remaining_steps) - 1; 
    
    for(int col_base = col_min; col_base <= col_max; col_base +=4){
        
        //copy data to tile 
        tile[threadIdx.x*4] = data[row*ncols + col_base];
        if(col_base + 3 < ncols){
            tile[threadIdx.x*4+1] = data[row*ncols + col_base+1];
            tile[threadIdx.x*4+2] = data[row*ncols + col_base+2];
            tile[threadIdx.x*4+3] = data[row*ncols + col_base+3];
        } else if(col_base + 2 < ncols){
            tile[threadIdx.x*4+1] = data[row*ncols + col_base+1];
            tile[threadIdx.x*4+2] = data[row*ncols + col_base+2];
        } else if(col_base + 1 < ncols){
            tile[threadIdx.x*4+1] = data[row*ncols + col_base+1];
        }
        
        __syncthreads();

        for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
            uint32_t i = threadIdx.x;
            if (threadIdx.x < 128){ // Only process first 128 threads (half of them)
                uint32_t half_group_size = 1 << loc_step;   
                uint32_t group = i >> loc_step;                           // i/(half_group_size)    
                uint32_t group_pos = i & (half_group_size - 1);   // i%(half_group_size)  
                uint32_t index1 = (group << (loc_step + 1)) + group_pos; // stage + 1 is size of group
                uint32_t index2 = index1 + half_group_size;
                gl64_t factor;
                {
                    //global_step
                    uint32_t gs = base_step + loc_step;
                    //global_half_group_size
                    uint32_t ghgs = 1 << gs; //group half
                    //global_group_pos
                    uint32_t ggp =(blockIdx.x << 7) + i; //blockIdx.x* blockDim.x/2 + i;
                    ggp = ((ggp & remaining_msk)<< base_step) + (ggp >> remaining_steps);
                    ggp = ggp & (ghgs - 1);
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
            data[row*ncols + col_base] = tile[threadIdx.x*4] * inv_factor;
            if(col_base + 3 < ncols){
                data[row*ncols + col_base+1] = tile[threadIdx.x*4+1] * inv_factor;
                data[row*ncols + col_base+2] = tile[threadIdx.x*4+2] * inv_factor;
                data[row*ncols + col_base+3] = tile[threadIdx.x*4+3] * inv_factor;
            } else if(col_base + 2 < ncols){
                data[row*ncols + col_base+1] = tile[threadIdx.x*4+1] * inv_factor;
                data[row*ncols + col_base+2] = tile[threadIdx.x*4+2] * inv_factor;
            } else if(col_base + 1 < ncols){
                data[row*ncols + col_base+1] = tile[threadIdx.x*4+1] * inv_factor;
            }
        }else{
            data[row*ncols + col_base] = tile[threadIdx.x*4];
            if(col_base + 3 < ncols){
                data[row*ncols + col_base+1] = tile[threadIdx.x*4+1];
                data[row*ncols + col_base+2] = tile[threadIdx.x*4+2];
                data[row*ncols + col_base+3] = tile[threadIdx.x*4+3];
            } else if(col_base + 2 < ncols){
                data[row*ncols + col_base+1] = tile[threadIdx.x*4+1];
                data[row*ncols + col_base+2] = tile[threadIdx.x*4+2];
            } else if(col_base + 1 < ncols){
                data[row*ncols + col_base+1] = tile[threadIdx.x*4+1];
            }
        }
    }   
}

__global__ void br_ntt_8_steps_blocks_par(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t ncols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    __shared__ gl64_t tile[1024];

    //assume domain_size is multiple of 256
    uint32_t n_loc_steps = min(log_domain_size_in - base_step, 8);    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    //evaluate row as if I shited 8 bits after each batch
    uint32_t accumBatchSize = 1 << base_step;
    uint32_t nBatches = domain_size_in / accumBatchSize;
    uint32_t low_bits = row / nBatches;
    uint32_t up_bits = row % nBatches;
    row = up_bits * accumBatchSize + low_bits;

    //remaining steps
    uint32_t remaining_steps = log_domain_size_in - (base_step+1); 
    uint32_t remaining_msk = (1 << remaining_steps) - 1; 
    uint32_t offset = domain_size_out << 2; //4 cols per block and is embede in the Nextended size

    uint32_t block=blockIdx.y;
    gl64_t *data_block = data + block*offset;
    uint32_t col_base = block * 4;
    uint32_t ncols_block = (ncols - col_base) < 4 ? ncols - col_base : 4;
    //copy data to tile 
    for(int i=0; i<ncols_block; i++){
        //tile[threadIdx.x*4+i] = data_block[row*ncols_block+i];
        tile[256*i+threadIdx.x] = data_block[row*ncols_block+i];
    }
    
    __syncthreads();

    for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
        uint32_t i = threadIdx.x;
        if (threadIdx.x < 128){ // Only process first 128 threads (half of them)
            uint32_t half_group_size = 1 << loc_step;   
            uint32_t group = i >> loc_step;                           // i/(half_group_size)    
            uint32_t group_pos = i & (half_group_size - 1);   // i%(half_group_size)  
            uint32_t index1 = (group << (loc_step + 1)) + group_pos; // stage + 1 is size of group
            uint32_t index2 = index1 + half_group_size;
            gl64_t factor;
            {
                //global_step
                uint32_t gs = base_step + loc_step;
                //global_half_group_size
                uint32_t ghgs = 1 << gs; //group half
                //global_group_pos
                uint32_t ggp =(blockIdx.x << 7) + i; //blockIdx.x* blockDim.x/2 + i;
                ggp = ((ggp & remaining_msk)<< base_step) + (ggp >> remaining_steps);
                ggp = ggp & (ghgs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];  // Use actual domain size
            }
            for(int j=0; j<ncols_block; j++){
                gl64_t odd_sub = tile[ j*256 + index2] * factor;
                tile[j*256 +index2] = tile[j*256 + index1] - odd_sub;               
                tile[j*256 +index1] = tile[j*256 + index1] + odd_sub;                
            }                             
        }
        __syncthreads();
    }
    // copy values to data with scaling applied when this iteration includes the final stage
    if(inverse && (base_step + n_loc_steps) >= log_domain_size_in){
        gl64_t inv_factor = gl64_t(domain_size_inverse[log_domain_size_in]);
        if(extend) inv_factor = inv_factor * d_r[row];
        for(int i=0; i<ncols_block; i++){
            data_block[row*ncols_block+i] = tile[i*256+threadIdx.x] * inv_factor;
        }
    }else{
        for(int i=0; i<ncols_block; i++){
            data_block[row*ncols_block+i] = tile[i*256+threadIdx.x];
        }
    } 
}

__global__ void br_ntt_8_steps_blocks(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t ncols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    __shared__ gl64_t tile[1024];

    //assume domain_size is multiple of 256
    uint32_t n_loc_steps = min(log_domain_size_in - base_step, 8);    
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    //evaluate row as if I shited 8 bits after each batch
    uint32_t accumBatchSize = 1 << base_step;
    uint32_t nBatches = domain_size_in / accumBatchSize;
    uint32_t low_bits = row / nBatches;
    uint32_t up_bits = row % nBatches;
    row = up_bits * accumBatchSize + low_bits;

    //remaining steps
    uint32_t remaining_steps = log_domain_size_in - (base_step+1); 
    uint32_t remaining_msk = (1 << remaining_steps) - 1; 
    uint32_t nblocks = (ncols + 3) / 4;
    uint32_t offset = domain_size_out << 2; //4 cols per block and is embede in the Nextended size

    for(int block = 0; block < nblocks; block++){
        gl64_t *data_block = data + block*offset;
        uint32_t col_base = block * 4;
        uint32_t ncols_block = (ncols - col_base) < 4 ? ncols - col_base : 4;
        //copy data to tile 
       for(int i=0; i<ncols_block; i++){
            tile[threadIdx.x*4+i] = data_block[row*ncols_block+i];
       }
        
        __syncthreads();

        for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
            uint32_t i = threadIdx.x;
            if (threadIdx.x < 128){ // Only process first 128 threads (half of them)
                uint32_t half_group_size = 1 << loc_step;   
                uint32_t group = i >> loc_step;                           // i/(half_group_size)    
                uint32_t group_pos = i & (half_group_size - 1);   // i%(half_group_size)  
                uint32_t index1 = (group << (loc_step + 1)) + group_pos; // stage + 1 is size of group
                uint32_t index2 = index1 + half_group_size;
                gl64_t factor;
                {
                    //global_step
                    uint32_t gs = base_step + loc_step;
                    //global_half_group_size
                    uint32_t ghgs = 1 << gs; //group half
                    //global_group_pos
                    uint32_t ggp =(blockIdx.x << 7) + i; //blockIdx.x* blockDim.x/2 + i;
                    ggp = ((ggp & remaining_msk)<< base_step) + (ggp >> remaining_steps);
                    ggp = ggp & (ghgs - 1);
                    factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];  // Use actual domain size
                }
                index1 = index1 << 2;
                index2 = index2 << 2;
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
                data_block[row*ncols_block+i] = tile[threadIdx.x*4+i] * inv_factor;
           }
        }else{
           for(int i=0; i<ncols_block; i++){
                data_block[row*ncols_block+i] = tile[threadIdx.x*4+i];
           }
        }
    }   
}

__global__ void br_ntt_group_new(gl64_t *data, gl64_t *twiddles, uint32_t i, uint32_t domain_size, uint32_t ncols)
{
    uint32_t start = domain_size >> 1;
    twiddles = twiddles + start;

    for (uint32_t j = blockIdx.x; j < domain_size / 2; j += gridDim.x)
    {
        for (uint32_t col = threadIdx.x; col < ncols; col += blockDim.x)
        {
            uint32_t half_group_size = 1 << i;
            uint32_t group = j >> i;                     // j/(group_size/2);
            uint32_t offset = j & (half_group_size - 1); // j%(half_group_size);
            uint32_t index1 = (group << (i + 1)) + offset;
            uint32_t index2 = index1 + half_group_size;
            gl64_t factor = twiddles[offset * (domain_size >> (i + 1))];
            gl64_t odd_sub = gl64_t((uint64_t)data[index2 * ncols + col]) * factor;
            data[index2 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) - odd_sub;
            data[index1 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) + odd_sub;
        }
    }
}

__global__ void intt_scale(gl64_t *data, gl64_t *d_r, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, bool extend)
{
    uint32_t j = blockIdx.x;    // domain_size
    uint32_t col = threadIdx.x; // cols
    uint32_t index = j * ncols + col;
    gl64_t factor = gl64_t(domain_size_inverse[log_domain_size]);
    if (extend)
    {
        factor = factor * d_r[j];
    } 
    if (index < domain_size * ncols)
    {
        data[index] = gl64_t((uint64_t)data[index]) * factor;
    }
}

__global__ void reverse_permutation_column(gl64_t *data, uint32_t log_domain_size, uint32_t ncols)
{
    uint64_t r = blockIdx.x * blockDim.x + threadIdx.x; 
    uint64_t c = blockIdx.y * blockDim.y + threadIdx.y;   
    uint64_t domain_size = 1 << log_domain_size;
    gl64_t *column = &data[c * domain_size];

    if(r < domain_size && c < ncols)
    {
        uint64_t rr = __brev(r) >> (32 - log_domain_size);
        if (r < rr)
        {
            gl64_t tmp = column[r];
            column[r] = data[rr];
            column[rr] = tmp;
        }
    }
}

__global__ void reverse_permutation_new(gl64_t *data, uint32_t log_domain_size, uint32_t ncols)
{
    uint64_t row = blockIdx.x;
    uint64_t col = threadIdx.x;
    uint64_t domain_size = 1 << log_domain_size;

    for (uint64_t r = row; r < domain_size; r += gridDim.x)
    {
        uint64_t rowr = __brev(r) >> (32 - log_domain_size);
        if (rowr > r)
        {
            for (uint64_t c = col; c < ncols; c += blockDim.x)
            {
                gl64_t tmp = data[r * ncols + c];
                data[r * ncols + c] = data[rowr * ncols + c];
                data[rowr * ncols + c] = tmp;
            }
        }
    }
}

__global__ void reverse_permutation_blocks(gl64_t *data, uint32_t log2_domain_size_in, uint64_t domain_size_out, uint32_t ncols)
{
    uint64_t row = blockIdx.x;
    uint64_t ncols_block = (ncols - 4*blockIdx.y) < 4 ? ncols - blockIdx.y * 4 : 4;
    uint64_t domain_size_in = 1 << log2_domain_size_in;
    uint64_t offset = blockIdx.y * 4 * domain_size_out;
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

__global__ void reverse_permutation(gl64_t *data, uint32_t log_domain_size, uint32_t ncols)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ibr = __brev(idx) >> (32 - log_domain_size);
    if (ibr > idx)
    {
        gl64_t tmp;
        for (uint32_t i = 0; i < ncols; i++)
        {
            tmp = data[idx * ncols + i];
            data[idx * ncols + i] = data[ibr * ncols + i];
            data[ibr * ncols + i] = tmp;
        }
    }
}

__global__ void reverse_permutation_1d(gl64_t *data, uint32_t log_domain_size, uint32_t ncols)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = idx / ncols;
    uint32_t col = idx % ncols;

    if (row < (1 << log_domain_size) && col < ncols)
    {
        uint32_t ibr = __brev(row) >> (32 - log_domain_size);
        if (ibr > row)
        {
            gl64_t tmp = data[row * ncols + col];
            data[row * ncols + col] = data[ibr * ncols + col];
            data[ibr * ncols + col] = tmp;
        }
    }
}

__global__ void reverse_permutation_2d(gl64_t *data, uint32_t log_domain_size, uint32_t ncols)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < (1 << log_domain_size) && col < ncols)
    {
        uint32_t ibr = __brev(row) >> (32 - log_domain_size);
        if (ibr > row)
        {
            gl64_t tmp = data[row * ncols + col];
            data[row * ncols + col] = data[ibr * ncols + col];
            data[ibr * ncols + col] = tmp;
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

void ntt_cuda( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size, uint32_t ncols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{   

    uint32_t domain_size = 1 << log_domain_size;

    dim3 blockDim;
    dim3 gridDim;
    
    blockDim = dim3(TPB_NTT);
    gridDim = dim3(8192);
    reverse_permutation_new<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size, ncols);
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
                br_ntt_8_steps<<<domain_size / 256, 256, 0, stream>>>(data, d_twiddles, d_r, domain_size, log_domain_size, ncols, step, true, inverse, extend, maxLogDomainSize, 0, ncols-1);
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        for (uint32_t stage = 0; stage < log_domain_size; stage++)
        {
            br_ntt_group<<<domain_size / 2, ncols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size, log_domain_size, ncols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

}

void ntt_cuda_blocks( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t ncols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize, gl64_t* aux_data)
{   

    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    dim3 blockDim;
    dim3 gridDim;
    blockDim = dim3(4);
    gridDim = dim3(1024,(ncols + 3) / 4);

    reverse_permutation_blocks<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size_in, domain_size_out, ncols);
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
    

    if(log_domain_size_in >= 8 ) {
         for(uint32_t step = 0; step < log_domain_size_in; step+=8){
                br_ntt_8_steps_blocks<<<domain_size_in / 256, 256, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, ncols, step, true, inverse, extend, maxLogDomainSize);
        }
    } else {
        for (uint32_t stage = 0; stage < log_domain_size_in; stage++)
        {
            br_ntt_group_blocks<<<domain_size_in / 2, ncols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size_in, log_domain_size_in, domain_size_out, ncols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

}

void ntt_cuda_blocks_par( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t ncols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize, gl64_t* aux_data)
{   

    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    dim3 blockDim;
    dim3 gridDim;
    blockDim = dim3(4);
    gridDim = dim3(1024,(ncols + 3) / 4);

    reverse_permutation_blocks<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size_in, domain_size_out, ncols);
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
    

    if(log_domain_size_in >= 8 ) {
         for(uint32_t step = 0; step < log_domain_size_in; step+=8){
                dim3 blocks = dim3(domain_size_in / 256, (ncols+3)/4, 1);
                dim3 threads = dim3(256,1,1);
                br_ntt_8_steps_blocks_par<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, ncols, step, true, inverse, extend, maxLogDomainSize);
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        for (uint32_t stage = 0; stage < log_domain_size_in; stage++)
        {
            br_ntt_group_blocks<<<domain_size_in / 2, ncols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size_in, log_domain_size_in, domain_size_out, ncols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }

}