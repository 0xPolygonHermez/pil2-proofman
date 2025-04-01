#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>

#include "poseidon2_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"

// #ifdef GPU_TIMING
#include "timer_gl.hpp"
// #endif

typedef uint32_t u32;
typedef uint64_t u64;

// CUDA Threads per Block
#define TPB 128

#define MAX_WIDTH 12

extern __shared__ gl64_t scratchpad[];

/* new functions */
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

/* --- Based on seq code --- */

__device__ __forceinline__ void pow7_2(gl64_t *x)
{
    gl64_t x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
}

__device__ __forceinline__ void add_2(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[0] = x[0] + C[i];
    }
}

__device__ __forceinline__ void prod_2(gl64_t *x, const gl64_t alpha, const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = alpha * C[i];
    }
}

__device__ __forceinline__ void pow7add_2(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
    gl64_t x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        gl64_t xi = x[i] + C[i];
        x2[i] = xi * xi;
        x3[i] = xi * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
}

__device__ __forceinline__ void matmul_external_(gl64_t *x)
{
    matmul_m4_(&x[0]);
    matmul_m4_(&x[4]);
    matmul_m4_(&x[8]);

    gl64_t stored[4] = {
        x[0] + x[4] + x[8],
        x[1] + x[5] + x[9],
        x[2] + x[6] + x[10],
        x[3] + x[7] + x[11],
    };
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i] + stored[i % 4];
    }
}

__device__ __forceinline__ void matmul_m4_(gl64_t *x)
{
    gl64_t t0 = x[0] + x[1];
    gl64_t t1 = x[2] + x[3];
    gl64_t t2 = x[1] + x[1] + t1;
    gl64_t t3 = x[3] + x[3] + t0;
    gl64_t t1_2 = t1 + t1;
    gl64_t t0_2 = t0 + t0;
    gl64_t t4 = t1_2 + t1_2 + t3;
    gl64_t t5 = t0_2 + t0_2 + t2;
    gl64_t t6 = t3 + t5;
    gl64_t t7 = t2 + t4;

    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

__device__ __forceinline__ void prodadd_(gl64_t *x, const gl64_t D[SPONGE_WIDTH], const gl64_t &sum)
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i] * D[i] + sum;
    }
}

// Constants defined in "poseidon2_goldilocks_constants.hpp"
__device__ __constant__ uint64_t GPU_C[118];
__device__ __constant__ uint64_t GPU_D[12];

void init_gpu_const_2(int nDevices = 0)
{
    static int initialized = 0;

    if (initialized == 0)
    {
        initialized = 1;
        /*if (nDevices == 0)
        {
            CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
        }*/
        if (nDevices == 0)
        {
            nDevices = 1;
        }
        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C, Poseidon2GoldilocksConstants::C, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D, Poseidon2GoldilocksConstants::D, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        }
        CHECKCUDAERR(cudaSetDevice(0));
    }
}

// rick: reescriure
__device__ void hash_full_result_seq_2(gl64_t *state, const gl64_t *input, const gl64_t *GPU_C_GL, const gl64_t *GPU_D_GL)
{
    mymemcpy((uint64_t *)state, (uint64_t *)input, SPONGE_WIDTH);
    
    matmul_external_(state);

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_2(state, &(GPU_C_GL[r * SPONGE_WIDTH]));
        matmul_external_(state);
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        state[0] = state[0] + GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7_2(state[0]);
        gl64_t sum_;
        sum_.val = 0;
        add_2(&sum_, state);
        prodadd_(state, GPU_D_GL, sum_);
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_2(state, &(GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_(state);
    }
   
}

/* --- integration --- */

__device__ void linear_hash_one_2(gl64_t *output, gl64_t *input, uint32_t size, int tid)
{
    u32 remaining = size;
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_D_SM[12];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_D_SM, GPU_D, 12);
    }
    __syncthreads();

    gl64_t state[SPONGE_WIDTH];

    if (size <= CAPACITY)
    {
        mymemcpy((uint64_t *)output, (uint64_t *)input, size);
        mymemset((uint64_t *)&output[size], 0, (CAPACITY - size));
        return; // no need to hash
    }
    while (remaining)
    {
        if (remaining == size)
        {
            mymemset((uint64_t *)(state + RATE), 0, CAPACITY);
        }
        else
        {
            mymemcpy((uint64_t *)(state + RATE), (uint64_t *)state, CAPACITY);
        }

        u32 n = (remaining < RATE) ? remaining : RATE;
        mymemset((uint64_t *)&state[n], 0, (RATE - n));
        mymemcpy((uint64_t *)state, (uint64_t *)(input + (size - remaining)), n);
        hash_full_result_seq_2(state, state, GPU_C_SM, GPU_D_SM);
        remaining -= n;
    }
    mymemcpy((uint64_t *)output, (uint64_t *)state, CAPACITY);
}

__device__ void linear_partial_hash_one_2(gl64_t *input, uint32_t size, gl64_t *state, int tid)
{
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_D_SM[12];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_D_SM, GPU_D, 12);
    }
    __syncthreads();

    u32 remaining = size;

    while (remaining)
    {
        mymemcpy((uint64_t *)(state + RATE), (uint64_t *)state, CAPACITY);
        u32 n = (remaining < RATE) ? remaining : RATE;
        mymemset((uint64_t *)&state[n], 0, (RATE - n));
        mymemcpy((uint64_t *)state, (uint64_t *)(input + (size - remaining)), n);
        hash_full_result_seq_2(state, state, GPU_C_SM, GPU_D_SM);
        remaining -= n;
    }
}

__global__ void linear_hash_gpu_2_2(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *inp = (gl64_t *)(input + tid * size);
    gl64_t *out = (gl64_t *)(output + tid * CAPACITY);
    linear_hash_one_2(out, inp, size, threadIdx.x);
}

__global__ void linear_partial_init_hash_gpu_2_2(uint64_t *gstate, int32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *state = (gl64_t *)(gstate + tid * SPONGE_WIDTH);
    memset(state, 0, SPONGE_WIDTH * sizeof(gl64_t));
}

__global__ void linear_partial_hash_gpu_2(uint64_t *input, uint32_t num_cols, uint32_t num_rows, uint64_t *gstate, uint32_t hash_per_thread = 1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    for (uint32_t i = 0; i < hash_per_thread; i++)
    {
        gl64_t *inp = (gl64_t *)(input + (tid * hash_per_thread + i) * num_cols);
        gl64_t *state = (gl64_t *)(gstate + (tid * hash_per_thread + i) * SPONGE_WIDTH);
        linear_partial_hash_one_2(inp, num_cols, state, threadIdx.x);
    }
}

__device__ void hash_one_2(gl64_t *state, gl64_t *const input, int tid)
{
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_D_SM[12];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_D_SM, GPU_D, 12);
    }
    __syncthreads();

    gl64_t aux[SPONGE_WIDTH];
    hash_full_result_seq_2(aux, input, GPU_C_SM, GPU_D_SM);
    mymemcpy((uint64_t *)state, (uint64_t *)aux, CAPACITY);
}

__global__ void hash_gpu_2(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input, 0, SPONGE_WIDTH);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor[nextIndex + tid * RATE], RATE);
    hash_one_2((gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY]), pol_input, threadIdx.x);
}

__global__ void hash_gpu_3(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input, 0, SPONGE_WIDTH);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor[nextIndex + tid * SPONGE_WIDTH], SPONGE_WIDTH);
    hash_one_2((gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY]), pol_input, threadIdx.x);
}

void merkletree_cuda_batch_2(Goldilocks::Element *tree, uint64_t *dst_gpu_tree, uint64_t *gpu_tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t dim, uint32_t const gpu_id)
{
    cudaStream_t gpu_stream;
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamCreate(&gpu_stream));
    cudaDeviceProp prop;
    CHECKCUDAERR(cudaGetDeviceProperties(&prop, gpu_id));
    size_t numElementsTree = num_rows * CAPACITY;
    size_t totalMemNeeded = num_rows * num_cols * dim * sizeof(uint64_t) + numElementsTree * sizeof(uint64_t);
    size_t maxMem = prop.totalGlobalMem * 8 / 10;
    size_t batches = (size_t)ceil(totalMemNeeded / (1.0 * maxMem));
    size_t rowsBatch = (size_t)ceil(num_rows / (1.0 * batches));
    size_t rowsLastBatch = num_rows % rowsBatch;
    if (rowsLastBatch > 0)
    {
        batches--;
    }

#ifdef FDEBUG
    printf("GPU max mem: %lu\n", prop.totalGlobalMem);
    printf("GPU max usable mem: %lu\n", maxMem);
    printf("Total needed mem: %lu\n", totalMemNeeded);
    printf("Batches %lu\n", batches);
    printf("Rows per batch %lu\n", rowsBatch);
    printf("Rows last batch %lu\n", rowsLastBatch);
#endif

    uint64_t *gpu_input;
    CHECKCUDAERR(cudaMalloc(&gpu_input, rowsBatch * num_cols * dim * sizeof(uint64_t)));

    Goldilocks::Element *iptr = input;
    uint64_t *gtree_ptr = gpu_tree;
    for (uint32_t b = 0; b < batches; b++)
    {
        CHECKCUDAERR(cudaMemcpyAsync(gpu_input, (uint64_t *)iptr, rowsBatch * num_cols * dim * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream));
        iptr += (rowsBatch * num_cols * dim);
        linear_hash_gpu_2_2<<<ceil(rowsBatch / (1.0 * TPB)), TPB, 0, gpu_stream>>>(gtree_ptr, gpu_input, num_cols * dim, rowsBatch);
        gtree_ptr += (rowsBatch * CAPACITY);
    }
    if (rowsLastBatch > 0)
    {
        CHECKCUDAERR(cudaMemcpyAsync(gpu_input, (uint64_t *)iptr, rowsLastBatch * num_cols * dim * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream));
        linear_hash_gpu_2_2<<<ceil(rowsLastBatch / (1.0 * TPB)), TPB, 0, gpu_stream>>>(gtree_ptr, gpu_input, num_cols * dim, rowsLastBatch);
    }
    if (dst_gpu_tree != NULL)
    {
        CHECKCUDAERR(cudaMemcpyPeerAsync(dst_gpu_tree, 0, gpu_tree, gpu_id, numElementsTree * sizeof(uint64_t), gpu_stream));
    }
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream));
    CHECKCUDAERR(cudaFree(gpu_input));
    CHECKCUDAERR(cudaStreamDestroy(gpu_stream));
}

void merkletree_cuda_multi_gpu_2(Goldilocks::Element *tree, uint64_t *dst_gpu_tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim, uint32_t const ngpu)
{
    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // size_t totalMemNeeded = num_rows * num_cols * dim * sizeof(uint64_t) + numElementsTree * sizeof(uint64_t);
    // size_t maxMem = prop.totalGlobalMem * 8 / 10 * ngpu;
    // bool use_batch = (totalMemNeeded >= maxMem);
    bool use_batch = false;
    size_t rowsDevice = num_rows / ngpu;
    uint64_t numElementsTreeDevice = rowsDevice * CAPACITY;
    uint64_t **gpu_input = (uint64_t **)malloc(ngpu * sizeof(uint64_t *));
    uint64_t **gpu_tree = (uint64_t **)malloc(ngpu * sizeof(uint64_t *));
    cudaStream_t *gpu_stream = (cudaStream_t *)malloc(ngpu * sizeof(cudaStream_t));
    assert(gpu_input != NULL);
    assert(gpu_tree != NULL);
    assert(gpu_stream != NULL);

#ifdef FDEBUG
    if (use_batch)
    {
        printf("Doing multi batch on multi gpu (%d GPUs)\n", ngpu);
    }
    else
    {
        printf("Doing multi gpu single batch (%d GPUs)\n", ngpu);
    }
    printf("Total rows: %lu\nRows per GPU: %lu\n", num_rows, rowsDevice);
#endif

    if (use_batch)
    {
#pragma omp parallel for num_threads(ngpu)
        for (uint32_t d = 0; d < ngpu; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaMalloc(&gpu_tree[d], numElementsTreeDevice * sizeof(uint64_t)));
            merkletree_cuda_batch_2(tree + (d * numElementsTreeDevice), dst_gpu_tree + (d * numElementsTreeDevice), gpu_tree[d], input + (d * rowsDevice * num_cols * dim), num_cols, rowsDevice, dim, d);
        }

#pragma omp parallel for num_threads(ngpu)
        for (uint32_t d = 0; d < ngpu; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaFree(gpu_tree[d]));
        }
    }
    else
    {
#ifdef GPU_TIMING
        TimerStart(merkletree_cuda_multi_gpu_copyToGPU);
#endif
#pragma omp parallel for num_threads(ngpu)
        for (uint32_t d = 0; d < ngpu; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaMalloc(&gpu_tree[d], numElementsTreeDevice * sizeof(uint64_t)));
            CHECKCUDAERR(cudaMalloc(&gpu_input[d], rowsDevice * num_cols * dim * sizeof(uint64_t)));
            CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
            CHECKCUDAERR(cudaMemcpyAsync(gpu_input[d], (uint64_t *)(input + d * rowsDevice * num_cols * dim), rowsDevice * num_cols * dim * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        }
#ifdef GPU_TIMING
        for (uint32_t d = 0; d < ngpu; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
        TimerStopAndLog(merkletree_cuda_multi_gpu_copyToGPU);
        TimerStart(merkletree_cuda_multi_gpu_kernel);
#endif
#pragma omp parallel for num_threads(ngpu)
        for (uint32_t d = 0; d < ngpu; d++)
        {
            linear_hash_gpu_2_2<<<ceil(rowsDevice / (1.0 * TPB)), TPB, 0, gpu_stream[d]>>>(gpu_tree[d], gpu_input[d], num_cols * dim, rowsDevice);
        }
#ifdef GPU_TIMING
        for (uint32_t d = 0; d < ngpu; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
        TimerStopAndLog(merkletree_cuda_multi_gpu_kernel);
        TimerStart(merkletree_cuda_multi_gpu_copyPeer2Peer);
#endif
#pragma omp parallel for num_threads(ngpu)
        for (uint32_t d = 0; d < ngpu; d++)
        {
            CHECKCUDAERR(cudaMemcpyPeer(dst_gpu_tree + (d * numElementsTreeDevice), 0, gpu_tree[d], d, numElementsTreeDevice * sizeof(uint64_t)));
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(merkletree_cuda_multi_gpu_copyPeer2Peer);
        TimerStart(merkletree_cuda_multi_gpu_cleanup);
#endif
#pragma omp parallel for num_threads(ngpu)
        for (uint32_t d = 0; d < ngpu; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
            CHECKCUDAERR(cudaFree(gpu_input[d]));
            CHECKCUDAERR(cudaFree(gpu_tree[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(merkletree_cuda_multi_gpu_cleanup);
#endif
    }

    free(gpu_input);
    free(gpu_tree);
    free(gpu_stream);
}


void Poseidon2Goldilocks::merkletree_cuda_gpudata_inplace(uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    init_gpu_const_2();
    u32 actual_tpb = TPB;
    u32 actual_blks = num_rows / TPB + 1;

    CHECKCUDAERR(cudaSetDevice(0));
    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows); // includes CAPACITY
    CHECKCUDAERR(cudaMalloc(d_tree, numElementsTree * sizeof(uint64_t)));

    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu_2_2<<<actual_blks, actual_tpb>>>(*d_tree, d_input, num_cols * dim, num_rows);

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;
    while (pending > 1)
    {
        if (nextN < TPB)
        {
            actual_tpb = nextN;
            actual_blks = 1;
        }
        else
        {
            actual_tpb = TPB;
            actual_blks = nextN / TPB + 1;
        }
        hash_gpu_2<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, *d_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////

void Poseidon2Goldilocks::merkletree_cuda_coalesced(uint32_t arity, uint64_t *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    init_gpu_const_2(); // this needs to be done only once !!
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;


    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu_coalesced_2<<<actual_blks, actual_tpb, actual_tpb * 12 * 8>>>(d_tree, d_input, num_cols * dim, num_rows); // rick: el 12 aqeust harcoded no please!!
    CHECKCUDAERR(cudaGetLastError());

    // Build the merkle tree
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0){

            //std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
            CHECKCUDAERR(cudaMemset((uint64_t *)(d_tree + nextIndex + pending * CAPACITY), 0, extraZeros * CAPACITY * sizeof(uint64_t)));
        }
        if (nextN < TPB)
        {
            actual_tpb = nextN;
            actual_blks = 1;
        }
        else
        {
            actual_tpb = TPB;
            actual_blks = nextN / TPB + 1;
        }
        hash_gpu_3<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending + extraZeros, d_tree);       
        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

__global__ void hash_gpu_2(uint32_t nextN, uint64_t *cursor_in, uint64_t *cursor_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input + RATE, 0, CAPACITY);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor_in[tid * RATE], RATE);
    hash_one_2((gl64_t *)(&cursor_out[tid * CAPACITY]), pol_input, threadIdx.x);
}

__global__ void hash_gpu_3(uint32_t nextN, uint64_t *cursor_in, uint64_t *cursor_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    //mymemset((uint64_t *)pol_input + RATE, 0, CAPACITY);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor_in[tid * SPONGE_WIDTH], SPONGE_WIDTH);
    hash_one_2((gl64_t *)(&cursor_out[tid * CAPACITY]), pol_input, threadIdx.x);
}

void Poseidon2Goldilocks::merkletree_cuda_streams(uint32_t arity, uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    assert(arity == 2);

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows, arity);
    CHECKCUDAERR(cudaMalloc(d_tree, numElementsTree * sizeof(uint64_t)));

    init_gpu_const_2(); // this needs to be done only once !!
    // rick: we are assuming here that TPB is a power of 2
    int num_streams = 8; // note: must be a power of two
    uint64_t rows_per_stream = num_rows / num_streams;
    uint32_t actual_tpb = TPB;
    if (rows_per_stream < TPB)
    {
        num_streams = 1;
        rows_per_stream = num_rows;
        if (num_rows < TPB)
        {
            actual_tpb = num_rows;
        }
    }
    else
    {
        // we are considering simple case of perfect division
        assert(rows_per_stream % TPB == 0);
    }
    uint32_t actual_blks;


    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++)
    {
        CHECKCUDAERR(cudaStreamCreate(&streams[i]));
    }

    // Divide workload across streams
    uint64_t *cursor_final_in;
    uint64_t *cursor_final_out;

    for (int i = 0; i < num_streams; i++)
    {
        uint64_t row_offset = i * rows_per_stream;
        uint64_t rows_to_process = rows_per_stream; // min(rows_per_stream, num_rows - row_offset);

        if (rows_to_process > 0)
        {
            actual_tpb = TPB;
            if (rows_to_process < TPB)
            {
                actual_tpb = rows_to_process;
            }
            actual_blks = (rows_to_process + actual_tpb - 1) / actual_tpb;
            linear_hash_gpu_coalesced_2<<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), streams[i]>>>(
                *d_tree + row_offset * CAPACITY,
                d_input + row_offset * num_cols * dim,
                num_cols * dim,
                rows_to_process);

            uint64_t nextN = rows_to_process >> 1;
            uint64_t *cursor_in = *d_tree + row_offset * CAPACITY;
            uint64_t *cursor_out = *d_tree + (num_rows + nextN * i) * CAPACITY;
            while (nextN >= 1)
            {
                if (nextN <= 64)
                {
                    actual_tpb = nextN;
                    actual_blks = 1;
                }
                else
                {
                    actual_tpb = 64;
                    actual_blks = nextN / 64; // assuming perfect division
                }

                hash_gpu_2<<<actual_blks, actual_tpb, 0, streams[i]>>>(nextN, cursor_in, cursor_out);
                cursor_in = cursor_out;
                cursor_out = cursor_out + (num_streams - i) * nextN * CAPACITY + i * (nextN >> 1) * CAPACITY;
                nextN >>= 1;
            }
            if (i == 0)
            {
                cursor_final_in = cursor_in;
                cursor_final_out = cursor_out;
            }
        }
    }
    // Wait for all streams to finish
    for (int i = 0; i < num_streams; i++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(streams[i]));
        CHECKCUDAERR(cudaStreamDestroy(streams[i]));
    }

    uint64_t nextN = num_streams >> 1;
    uint64_t *cursor_in = cursor_final_in;
    uint64_t *cursor_out = cursor_final_out;
    // rick: this should be solved in a single block call
    while (nextN >= 1)
    {
        if (nextN <= 32)
        {
            actual_tpb = nextN;
            actual_blks = 1;
        }
        else
        {
            actual_tpb = 32;
            actual_blks = nextN / 32; // assuming perfect division
        }

        hash_gpu_2<<<actual_blks, actual_tpb>>>(nextN, cursor_in, cursor_out);
        cursor_in = cursor_out;
        cursor_out = cursor_out + nextN * CAPACITY;
        nextN >>= 1;
    }

}

__device__ __noinline__ void pow7_2(gl64_t &x)
{
    gl64_t x2 = x * x;
    gl64_t x3 = x * x2;
    gl64_t x4 = x2 * x2;
    x = x3 * x4;
}

__device__ __noinline__ void add_state_2(gl64_t *x)
{
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
       x[0]= x[0] + scratchpad[i * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void poseidon2_load(const uint64_t *in, uint32_t col, uint32_t ncols,
                                               uint32_t col_stride, size_t row_stride = 1)
{
    gl64_t r[RATE];

    const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    in += tid * col_stride + col * row_stride;

#pragma unroll
    for (uint32_t i = 0; i < RATE; i++, in += row_stride)
        if (i < ncols){
            r[i] = __ldcv((uint64_t *)in);
        }

    __syncwarp();

    for (uint32_t i = 0; i < RATE; i++)
        scratchpad[i * blockDim.x + threadIdx.x] = r[i];

    __syncwarp();
}

__device__ __forceinline__ void poseidon2_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride = 1)
{
    gl64_t r[CAPACITY];

    __syncwarp();

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        r[i] = scratchpad[i * blockDim.x + threadIdx.x];

    __syncwarp();

    const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    out += tid * col_stride;

#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++, out++)
        *(uint64_t *)out = r[i].get_val();
}

__device__ __forceinline__ void poseidon2_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols)
{
    if (ncols <= CAPACITY)
    {
        poseidon2_load(in, 0, ncols, ncols);
        for (uint32_t i = ncols; i < CAPACITY; i++)
        {
            scratchpad[i * blockDim.x + threadIdx.x].set_val(0);
        }
    }
    else
    {
        for (uint32_t col = 0;;)
        {
            uint32_t delta = min(ncols - col, RATE);
            poseidon2_load(in, col, delta, ncols);
            if (delta < RATE)
            {
                for (uint32_t i = delta; i < RATE; i++)
                {
                    scratchpad[i * blockDim.x + threadIdx.x].set_val(0);
                }
            }
            /*if(blockIdx.x == 0 && threadIdx.x == 0){
                for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
                    printf("tmp abans[%d] = %lu col=%d ncols=%d\n", i, scratchpad[i * blockDim.x + threadIdx.x].get_val(), col, ncols);
            }*/
            poseidon2_hash();
            
            /*if(blockIdx.x == 0 && threadIdx.x == 0){
                for (uint32_t i = 0; i < CAPACITY; i++)
                    printf("tmp[%d] = %lu col=%d ncols=%d\n", i, scratchpad[i * blockDim.x + threadIdx.x].get_val(), col, ncols);
            }*/

            if ((col += RATE) >= ncols)
                break;

            gl64_t tmp[CAPACITY];

#pragma unroll
            for (uint32_t i = 0; i < CAPACITY; i++)
                tmp[i] = scratchpad[i * blockDim.x + threadIdx.x];
            __syncwarp();

#pragma unroll

            for (uint32_t i = 0; i < CAPACITY; i++)
                scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = tmp[i];
            __syncwarp();
        }
    }
}

__device__ __forceinline__ void matmul_external_state_()
{
    matmul_m4_state_(0);
    matmul_m4_state_(4);
    matmul_m4_state_(8);

    gl64_t stored[4] = {
       scratchpad[0 * blockDim.x + threadIdx.x] + scratchpad[4 * blockDim.x + threadIdx.x] + scratchpad[8 * blockDim.x + threadIdx.x],
       scratchpad[1 * blockDim.x + threadIdx.x] + scratchpad[5 * blockDim.x + threadIdx.x] + scratchpad[9 * blockDim.x + threadIdx.x],
       scratchpad[2 * blockDim.x + threadIdx.x] + scratchpad[6 * blockDim.x + threadIdx.x] + scratchpad[10 * blockDim.x + threadIdx.x],
       scratchpad[3 * blockDim.x + threadIdx.x] + scratchpad[7 * blockDim.x + threadIdx.x] + scratchpad[11 * blockDim.x + threadIdx.x],
    };
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        scratchpad[i * blockDim.x + threadIdx.x] = scratchpad[i * blockDim.x + threadIdx.x] + stored[i % 4];
    }
}

__device__ __forceinline__ void matmul_m4_state_(uint32_t offset)
{
    
    gl64_t t0 = scratchpad[(offset + 0) * blockDim.x + threadIdx.x] + scratchpad[(offset + 1) * blockDim.x + threadIdx.x];
    gl64_t t1 = scratchpad[(offset + 2) * blockDim.x + threadIdx.x] + scratchpad[(offset + 3) * blockDim.x + threadIdx.x];
    gl64_t t2 = scratchpad[(offset + 1) * blockDim.x + threadIdx.x] + scratchpad[(offset + 1) * blockDim.x + threadIdx.x] + t1;
    gl64_t t3 = scratchpad[(offset + 3) * blockDim.x + threadIdx.x] + scratchpad[(offset + 3) * blockDim.x + threadIdx.x] + t0;
    gl64_t t1_2 = t1 + t1;
    gl64_t t0_2 = t0 + t0;
    gl64_t t4 = t1_2 + t1_2 + t3;
    gl64_t t5 = t0_2 + t0_2 + t2;
    gl64_t t6 = t3 + t5;
    gl64_t t7 = t2 + t4;

    scratchpad[(offset + 0) * blockDim.x + threadIdx.x] = t6;
    scratchpad[(offset + 1) * blockDim.x + threadIdx.x] = t5;
    scratchpad[(offset + 2) * blockDim.x + threadIdx.x] = t7;
    scratchpad[(offset + 3) * blockDim.x + threadIdx.x] = t4;

}


__device__ __forceinline__ void pow7add_state_(const gl64_t C[SPONGE_WIDTH])
{
    gl64_t x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        gl64_t xi = scratchpad[i * blockDim.x + threadIdx.x] + C[i];
        x2[i] = xi * xi;
        x3[i] = xi * x2[i];
        x4[i] = x2[i] * x2[i];
        scratchpad[i * blockDim.x + threadIdx.x] = x3[i] * x4[i];
    }
}


__device__ __forceinline__ void prodadd_state_(const gl64_t D[SPONGE_WIDTH], const gl64_t &sum)
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        scratchpad[i * blockDim.x + threadIdx.x] = scratchpad[i * blockDim.x + threadIdx.x] * D[i] + sum;
    }
}

__device__ __forceinline__ void poseidon2_hash()
{
    const gl64_t *GPU_C_GL = (gl64_t *)GPU_C;
    const gl64_t *GPU_D_GL = (gl64_t *)GPU_D;

    matmul_external_state_();
    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_state_(&(GPU_C_GL[r * SPONGE_WIDTH]));
        matmul_external_state_();
    }

    for(int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        scratchpad[threadIdx.x] = scratchpad[threadIdx.x] + GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7_2(scratchpad[threadIdx.x]);
        gl64_t sum_;
        sum_.val = 0;
        add_state_2(&sum_);
        prodadd_state_(GPU_D_GL, sum_);
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS; r++)
    {
        pow7add_state_(&(GPU_C_GL[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        matmul_external_state_();
    }
}

__global__ void linear_hash_gpu_coalesced_2(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t size, uint32_t num_rows)
{
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x].set_val(0);

    poseidon2_hash_loop(input, size);
    poseidon2_store(output, CAPACITY);
}