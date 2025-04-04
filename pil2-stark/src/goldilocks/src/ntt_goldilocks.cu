#include "ntt_goldilocks.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "gl64_t.cuh"
#include "poseidon2_goldilocks.hpp"
#include "ntt_goldilocks.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "omp.h"
#include "poseidon_goldilocks.hpp"

// CUDA Threads per Block
#define TPB_V1 64

// CUDA Threads Per Block
#define TPB_NTT 16
#define TPB_NTT_x 32
#define TPB_NTT_y 16
#define SHIFT 7

#define TRANSPOSE_TILE_DIM 32  // Tile size for shared memory
#define TRANSPOSE_BLOCK_ROWS 8

// #ifdef GPU_TIMING
#include "timer_gl.hpp"
// #endif

__device__ __forceinline__ uint32_t root_idx(uint32_t log_domain_size, uint32_t step, uint32_t idx);
__global__ void br_ntt_8_steps(gl64_t *data, gl64_t *twiddles, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, uint32_t base_step, bool suffle);
__global__ void br_ntt_group(gl64_t *data, gl64_t *twiddles, uint32_t i, uint32_t domain_size, uint32_t ncols);
__global__ void br_ntt_group_new(gl64_t *data, gl64_t *twiddles, uint32_t i, uint32_t domain_size, uint32_t ncols);
__global__ void intt_scale(gl64_t *data, gl64_t *r, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, bool extend);
__global__ void reverse_permutation_new(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void reverse_permutation_column(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void reverse_permutation(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void reverse_permutation_1d(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void reverse_permutation_2d(gl64_t *data, uint32_t log_domain_size, uint32_t ncols);
__global__ void init_twiddle_factors_small_size(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void init_twiddle_factors_first_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void init_twiddle_factors_second_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
void init_twiddle_factors(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void init_r_small_size(gl64_t *r, uint32_t log_domain_size);
__global__ void init_r_first_step(gl64_t *r, uint32_t log_domain_size);
__global__ void init_r_second_step(gl64_t *r, uint32_t log_domain_size);
void init_r(gl64_t *r, uint32_t log_domain_size);
void ntt_cuda( gl64_t *data, gl64_t *r, gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size, uint32_t ncols, bool inverse, bool extend);
__global__ void transpose_section(gl64_t *out, const gl64_t *in, uint64_t nCols, uint64_t domainSize);


__global__ void transpose(uint64_t *dst, uint64_t *src, uint32_t nblocks, uint32_t nrows, uint32_t ncols, uint32_t ncols_last_block)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid
    if (i >= nrows)
        return;

    uint64_t *ldst = dst + i * ((nblocks - 1) * ncols + ncols_last_block);

    for (uint32_t k = 0; k < nblocks - 1; k++)
    {
        for (uint32_t j = 0; j < ncols; j++)
        {
            *ldst = src[k * nrows * ncols + i * ncols + j];
            ldst++;
        }
    }
    // last block
    for (uint32_t j = 0; j < ncols_last_block; j++)
    {
        *ldst = src[(nblocks - 1) * nrows * ncols + i * ncols_last_block + j];
        ldst++;
    }
}

__global__ void transpose_opt(uint64_t *dst, uint64_t *src, uint32_t nblocks, uint32_t nrows, uint32_t ncols, uint32_t ncols_last_block, uint32_t nrb)
{
    __shared__ uint64_t row[1056];

    int ncols_total = (nblocks - 1) * ncols + ncols_last_block;
    // tid is the destination column
    int tid = threadIdx.x;
    if (tid >= ncols_total)
        return;

    // bid is the destination/source row
    int bid = blockIdx.x * nrb;
    if (bid >= nrows)
        return;

    int k = tid / ncols;
    int nc = ncols;
    if (k == nblocks - 1)
    {
        nc = ncols_last_block;
    }
    int j = tid % ncols;

    for (int r = bid; r < bid + nrb; r++)
    {
        uint64_t *pdst = dst + r * ncols_total + tid;
        uint64_t *psrc = src + (k * ncols * nrows) + r * nc + j;
        row[tid] = *psrc;
        __syncthreads();
        *pdst = row[tid];
        __syncthreads();
    }
}

__global__ void applyS(gl64_t *d_cmQ, gl64_t *d_q, gl64_t *d_S, uint64_t N, uint64_t qDeg, uint64_t qDim)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    for (uint64_t p = 0; p < qDeg; p++)
    {
        for (uint64_t j = 0; j < qDim; j++)
        {
            Goldilocks3GPU::mul((Goldilocks3GPU::Element &)d_cmQ[(i * qDeg + p) * FIELD_EXTENSION],
                                (Goldilocks3GPU::Element &)d_q[(p * N + i) * FIELD_EXTENSION],
                                d_S[p]);
        }
    }
}

void NTT_Goldilocks::computeQ_inplace(Goldilocks::Element *d_tree, uint64_t offset_cmQ, uint64_t offset_q, uint64_t qDeg, uint64_t qDim, Goldilocks::Element *S, uint64_t N, uint64_t NExtended, uint64_t ncols, DeviceCommitBuffers *d_buffers, uint64_t offset_helper, double *nttTime, double *merkleTime)
{
    cudaEvent_t point1, point2, point3;
    cudaEventCreate(&point1);
    cudaEventCreate(&point2);
    cudaEventCreate(&point3);

    cudaEventRecord(point1);

    gl64_t* d_r = d_buffers->d_aux_trace + offset_helper;
    gl64_t* d_forwardTwiddleFactors = d_buffers->d_aux_trace + offset_helper + NExtended;
    gl64_t* d_inverseTwiddleFactors = d_buffers->d_aux_trace + offset_helper + 2*NExtended;

    gl64_t *d_q = d_buffers->d_aux_trace + offset_q;
    gl64_t *d_cmQ = d_buffers->d_aux_trace + offset_cmQ;
    gl64_t *d_S;
    CHECKCUDAERR(cudaMalloc(&d_S, qDeg * sizeof(gl64_t))); //rick
    CHECKCUDAERR(cudaMemcpy(d_S, S, qDeg * sizeof(gl64_t), cudaMemcpyHostToDevice));
    if (ncols == 0 || NExtended == 0)
    {
        return;
    }

    CHECKCUDAERR(cudaMemset(d_forwardTwiddleFactors, 0, NExtended * sizeof(uint64_t)))
    CHECKCUDAERR(cudaMemset(d_inverseTwiddleFactors, 0, NExtended * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemset(d_r, 0, NExtended * sizeof(uint64_t)));

    // Init twiddle factors
    int lg2ext = log2(NExtended);
    init_twiddle_factors(d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2ext);

    // Intt
    ntt_cuda(d_q, d_r, d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2ext, qDim, true, false);


    dim3 threads(128, 1, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, 1, 1);
    applyS<<<blocks, threads>>>(d_cmQ, d_q, d_S, N, qDeg, qDim);
    CHECKCUDAERR(cudaMemset(d_cmQ + N * qDeg * qDim, 0, (NExtended - N) * qDeg * qDim * sizeof(gl64_t)));


    ntt_cuda(d_cmQ, d_r, d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2ext, ncols, false, false);

    cudaEventRecord(point2);

    Poseidon2Goldilocks::merkletree_cuda_coalesced(3, (uint64_t*) d_tree, (uint64_t *)d_cmQ, ncols, NExtended);

    cudaEventRecord(point3);
    
    if(nttTime!= nullptr){
        cudaEventSynchronize(point2);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, point1, point2);
        *nttTime = elapsedTime/1000;
    }
    if(merkleTime!= nullptr){ 
        cudaEventSynchronize(point3);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, point2, point3);
        *merkleTime = elapsedTime/1000;
    }
    cudaEventDestroy(point1);
    cudaEventDestroy(point2);
    cudaEventDestroy(point3);
}

void NTT_Goldilocks::LDE_MerkleTree_GPU_inplace(Goldilocks::Element *d_tree, gl64_t *d_dst_ntt, uint64_t offset_dst_ntt, gl64_t *d_src_ntt, uint64_t offset_src_ntt, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, DeviceCommitBuffers *d_buffers, uint64_t offset_helper, double *nttTime, double *merkleTime)
{
    cudaEvent_t point1, point2, point3;
    cudaEventCreate(&point1);
    cudaEventCreate(&point2);
    cudaEventCreate(&point3);
    cudaEventRecord(point1);


    gl64_t *d_dst_ntt_ = &d_dst_ntt[offset_dst_ntt];
    gl64_t *d_src_ntt_ = &d_src_ntt[offset_src_ntt];

    if (ncols == 0 || size == 0)
    {
        return;
    }

    gl64_t* d_r = d_buffers->d_aux_trace + offset_helper;
    gl64_t* d_forwardTwiddleFactors = d_buffers->d_aux_trace + offset_helper + ext_size;
    gl64_t* d_inverseTwiddleFactors = d_buffers->d_aux_trace + offset_helper + 2*ext_size;


    CHECKCUDAERR(cudaMemset(d_forwardTwiddleFactors, 0, ext_size * sizeof(uint64_t)))
    CHECKCUDAERR(cudaMemset(d_inverseTwiddleFactors, 0, ext_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemset(d_r, 0, ext_size * sizeof(uint64_t)));

    int lg2 = log2(size);
    int lg2ext = log2(ext_size);
    init_twiddle_factors(d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2);
    init_twiddle_factors(d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2ext);
    init_r(d_r, lg2);

    CHECKCUDAERR(cudaMemcpy(d_dst_ntt_, d_src_ntt_, size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToDevice));
    CHECKCUDAERR(cudaMemset(d_dst_ntt_ + size * ncols, 0, (ext_size - size) * ncols * sizeof(gl64_t)));
    ntt_cuda(d_dst_ntt_, d_r, d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2, ncols, true, true);

    ntt_cuda(d_dst_ntt_, d_r, d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2ext, ncols, false, false);

    cudaEventRecord(point2);

    Poseidon2Goldilocks::merkletree_cuda_coalesced(3, (uint64_t*) d_tree, (uint64_t *)d_dst_ntt_, ncols, ext_size);
    
    cudaEventRecord(point3);
    
    if(nttTime!= nullptr){
        cudaEventSynchronize(point2);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, point1, point2);
        *nttTime = elapsedTime/1000;
    }
    if(merkleTime!= nullptr){ 
        cudaEventSynchronize(point3);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, point2, point3);
        *merkleTime = elapsedTime/1000;
    }
    cudaEventDestroy(point1);
    cudaEventDestroy(point2);
    cudaEventDestroy(point3);
}

void NTT_Goldilocks::INTT_inplace(uint64_t data_offset, u_int64_t size, u_int64_t ncols, DeviceCommitBuffers *d_buffers, uint64_t offset_helper, gl64_t* d_data)
{

    gl64_t* d_r = d_buffers->d_aux_trace + offset_helper;
    gl64_t* d_forwardTwiddleFactors = d_buffers->d_aux_trace + offset_helper + size;
    gl64_t* d_inverseTwiddleFactors = d_buffers->d_aux_trace + offset_helper + 2*size;

    gl64_t *dst_src = d_data == nullptr ? d_buffers->d_aux_trace + data_offset : d_data;
    if (ncols == 0 || size == 0)
    {
        return;
    }

    CHECKCUDAERR(cudaMemset(d_forwardTwiddleFactors, 0, size * sizeof(uint64_t)))
    CHECKCUDAERR(cudaMemset(d_inverseTwiddleFactors, 0, size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemset(d_r, 0, size * sizeof(uint64_t)));

    int lg2 = log2(size);
    init_twiddle_factors(d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2);
    ntt_cuda(dst_src, d_r, d_forwardTwiddleFactors, d_inverseTwiddleFactors, lg2, ncols, true, false);
}


__global__ void br_ntt_group(gl64_t *data, gl64_t *twiddles, uint32_t i, uint32_t domain_size, uint32_t ncols)
{
    uint32_t j = blockIdx.x;
    uint32_t col = threadIdx.x;
    uint32_t start = domain_size >> 1; 
    twiddles = twiddles + start; 
    if (j < domain_size / 2 && col < ncols)
    {
        uint32_t half_group_size = 1 << i;
        uint32_t group = j >> i;                     // j/(group_size/2);
        uint32_t offset = j & (half_group_size - 1); // j%(half_group_size);
        uint32_t index1 = (group << i + 1) + offset;
        uint32_t index2 = index1 + half_group_size;
        gl64_t factor = twiddles[offset * (domain_size >> i + 1)];
        gl64_t odd_sub = gl64_t((uint64_t)data[index2 * ncols + col]) * factor;
        data[index2 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) - odd_sub;
        data[index1 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) + odd_sub;
        // DEGUG: assert(data[index2 * ncols + col] < 18446744069414584321ULL);
        // DEBUG: assert(data[index1 * ncols + col] < 18446744069414584321ULL);
    }
}

__device__ __forceinline__ uint32_t root_idx(uint32_t log_domain_size, uint32_t step, uint32_t idx)
{
    return idx << (log_domain_size - step);
}

__global__ void br_ntt_8_steps(gl64_t *data, gl64_t *twiddles, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, uint32_t base_step, bool suffle)
{
    __shared__ gl64_t tile[1024];

    //assume domain_size is multiple of 256
    uint32_t start = domain_size >> 1;
    twiddles = twiddles + start;
    uint32_t steps = min(log_domain_size - base_step, 8);

    
    uint32_t row;
    row = blockIdx.x * blockDim.x + threadIdx.x;        
    uint32_t bachSize = 1 << base_step;
    uint32_t nBatches = domain_size / bachSize;
    uint32_t thid = row / nBatches;
    uint32_t b = row % nBatches;
    row = b * bachSize + thid;
    

    uint32_t remaining_steps = log_domain_size - (base_step+1); //remaining steps
    uint32_t remaining_msk = (1 << remaining_steps) - 1; 
    
    for(int col_base = 0; col_base < ncols; col_base +=4){
        
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

        for(int i=0; i< steps; i++){
            uint32_t j = threadIdx.x;
            if (threadIdx.x < 128){ 
                uint32_t subgroup_size = 1 << i;    //subgroup := groups half
                uint32_t group = j >> i;                     
                uint32_t subgroup_offset = j & (subgroup_size - 1);     
                uint32_t index1 = (group << (i + 1)) + subgroup_offset;
                uint32_t index2 = index1 + subgroup_size;
                gl64_t factor;
                {
                    //global_step
                    uint32_t gs = base_step + i;
                    //global_subgroup_size
                    uint32_t gss = 1 << gs; //group half

                    //global_subgroup_offset
                    uint32_t gso =(blockIdx.x << 7) + j; //blockIdx.x* blockDim.x/2 + j;
                    gso = ((gso & remaining_msk)<< base_step) + (gso >> remaining_steps);
                    gso = gso & (gss - 1);
                    factor = twiddles[root_idx(log_domain_size, gs + 1, gso)];
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
        // copy values to data
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

__global__ void intt_scale(gl64_t *data, gl64_t *r, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, bool extend)
{
    uint32_t j = blockIdx.x;    // domain_size
    uint32_t col = threadIdx.x; // cols
    uint32_t index = j * ncols + col;
    gl64_t factor = gl64_t(domain_size_inverse[log_domain_size]);
    if (extend)
    {
        factor = factor * r[domain_size + j];
    }
    if (index < domain_size * ncols)
    {
        data[index] = gl64_t((uint64_t)data[index]) * factor;
        // DEBUG: assert(data[index] < 18446744069414584321ULL);
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

__global__ void init_twiddle_factors_small_size(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    gl64_t omega = gl64_t(omegas[log_domain_size]);
    gl64_t omega_inv = gl64_t(omegas_inv[log_domain_size]);

    uint32_t start = 1 << log_domain_size - 1;

    fwd_twiddles[start] = gl64_t::one();
    inv_twiddles[start] = gl64_t::one();

    for (uint32_t i = start + 1; i < start + (1 << log_domain_size - 1); i++)
    {
        fwd_twiddles[i] = fwd_twiddles[i - 1] * omega;
        inv_twiddles[i] = inv_twiddles[i - 1] * omega_inv;
    }
}

__global__ void init_twiddle_factors_first_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    gl64_t omega = gl64_t(omegas[log_domain_size]);
    gl64_t omega_inv = gl64_t(omegas_inv[log_domain_size]);

    uint32_t start = 1 << log_domain_size - 1;

    fwd_twiddles[start] = gl64_t::one();
    inv_twiddles[start] = gl64_t::one();

    for (uint32_t i = start + 1; i <= start + (1 << 12); i++)
    {
        fwd_twiddles[i] = fwd_twiddles[i - 1] * omega;
        inv_twiddles[i] = inv_twiddles[i - 1] * omega_inv;
    }
}

__global__ void init_twiddle_factors_second_step(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 1 << log_domain_size - 1;
    for (uint32_t i = 1; i < 1 << log_domain_size - 13; i++)
    {
        fwd_twiddles[start + i * 4096 + idx] = fwd_twiddles[start + (i - 1) * 4096 + idx] * fwd_twiddles[start + 4096];
        inv_twiddles[start + i * 4096 + idx] = inv_twiddles[start + (i - 1) * 4096 + idx] * inv_twiddles[start + 4096];
    }
}

void init_twiddle_factors(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    if (log_domain_size <= 13)
    {
        init_twiddle_factors_small_size<<<1, 1>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
    else
    {
        init_twiddle_factors_first_step<<<1, 1>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
        init_twiddle_factors_second_step<<<1 << 12, 1>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
}

__global__ void init_r_small_size(gl64_t *r, uint32_t log_domain_size)
{
    uint32_t start = 1 << log_domain_size;
    r[start] = gl64_t::one();
    for (uint32_t i = start + 1; i < start + (1 << log_domain_size); i++)
    {
        r[i] = r[i - 1] * gl64_t(SHIFT);
    }
}

__global__ void init_r_first_step(gl64_t *r, uint32_t log_domain_size)
{
    uint32_t start = 1 << log_domain_size;
    r[start] = gl64_t::one();
    // init first 4097 elements and then init others in parallel
    for (uint32_t i = start + 1; i <= start + (1 << 12); i++)
    {
        r[i] = r[i - 1] * gl64_t(SHIFT);
    }
}

__global__ void init_r_second_step(gl64_t *r, uint32_t log_domain_size)
{
    uint32_t start = 1 << log_domain_size;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = 1; i < 1 << log_domain_size - 12; i++)
    {
        r[start + i * 4096 + idx] = r[start + (i - 1) * 4096 + idx] * r[start + 4096];
    }
}

void init_r(gl64_t *r, uint32_t log_domain_size)
{
    if (log_domain_size <= 12)
    {
        init_r_small_size<<<1, 1>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
    else
    {
        init_r_first_step<<<1, 1>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
        init_r_second_step<<<1 << 12, 1>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
}

void ntt_cuda( gl64_t *data, gl64_t *r, gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size, uint32_t ncols, bool inverse, bool extend)
{   

    uint32_t domain_size = 1 << log_domain_size;

    dim3 blockDim;
    dim3 gridDim;
    
    blockDim = dim3(TPB_NTT);
    gridDim = dim3(8192);
    reverse_permutation_new<<<gridDim, blockDim, 0>>>(data, log_domain_size, ncols);
    CHECKCUDAERR(cudaGetLastError());

    gl64_t *ptr_twiddles = fwd_twiddles;
    if (inverse)
    {
        ptr_twiddles = inv_twiddles;
    }

    if(log_domain_size >= 8) {
         for(uint32_t step = 0; step < log_domain_size; step+=8){
            br_ntt_8_steps<<<domain_size / 256, 256>>>(data, ptr_twiddles, domain_size, log_domain_size, ncols, step, true);
            CHECKCUDAERR(cudaGetLastError());               
        }
    } else {
        for (uint32_t i = 0; i < log_domain_size; i++)
        {
            br_ntt_group<<<domain_size / 2, ncols, 0>>>(data, ptr_twiddles, i, domain_size, ncols);
            CHECKCUDAERR(cudaGetLastError());
        }
    }
   
    
    if (inverse)
    {
        intt_scale<<<domain_size, ncols, 0>>>(data, r, domain_size, log_domain_size, ncols, extend);
        
    }
}

__global__ void transpose_section(gl64_t *out, const gl64_t *in, uint64_t nCols, uint64_t domainSize) {
    
    __shared__ gl64_t tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1]; // Avoid bank conflicts

    int row = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x; 
    int col = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y; 

    // Load from global memory to shared memory
    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS) {
        if ((row + i) < domainSize && col < nCols) {
            tile[threadIdx.y + i][threadIdx.x] = in[(row + i) * nCols + col];
        }
    }
    
    __syncthreads();

    // Transpose indices
    row = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    col = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    // Store back to global memory
    for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS) {
        if ((row + i) < nCols && col < domainSize) {
            out[(row + i) * domainSize + col] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}