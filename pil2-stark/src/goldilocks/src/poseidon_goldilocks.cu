#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <omp.h>

#include "poseidon_goldilocks.hpp"
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
__global__ void linear_hash_gpu_tree(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows);
__global__ void linear_hash_gpu_coalesced(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t size, uint32_t num_rows);
__device__ __noinline__ void poseidon_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride);
__device__ __noinline__ void poseidon_store(gl64_t *out, uint32_t col_stride, size_t row_stride);
__device__ __noinline__ void poseidon_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols);
__device__ __noinline__ void poseidon_hash();
__device__ __noinline__ void pow7(gl64_t &x);

/* --- Based on seq code --- */

__device__ __forceinline__ void pow7_(gl64_t *x)
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

__device__ __forceinline__ void add_(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i] + C[i];
    }
}

__device__ __forceinline__ void prod_(gl64_t *x, const gl64_t alpha, const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = alpha * C[i];
    }
}

__device__ __forceinline__ void pow7add_(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
    gl64_t x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
#pragma unroll
    for (int i = 0; i < SPONGE_WIDTH; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
        x[i] = x[i] + C[i];
    }
}

__device__ __forceinline__ gl64_t dot_(gl64_t *x, const gl64_t C[SPONGE_WIDTH])
{
    gl64_t s0 = x[0] * C[0];
#pragma unroll
    for (int i = 1; i < SPONGE_WIDTH; i++)
    {
        s0 = s0 + x[i] * C[i];
    }
    return s0;
}

__device__ __forceinline__ void mvp_(gl64_t *state, const gl64_t *__restrict__ mat)
{
    gl64_t old_state[SPONGE_WIDTH];
    mymemcpy((uint64_t *)old_state, (uint64_t *)state, SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = mat[i] * old_state[0];
        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (mat[12 * j + i] * old_state[j]);
        }
    }
}

// Constants defined in "poseidon_goldilocks_constants.hpp"
__device__ __constant__ uint64_t GPU_C[118];
__device__ __constant__ uint64_t GPU_S[507];
__device__ __constant__ uint64_t GPU_M[144];
__device__ __constant__ uint64_t GPU_P[144];

void init_gpu_const(int nDevices = 0)
{
    static int initialized = 0;

    if (initialized == 0)
    {
        initialized = 1;
        if (nDevices == 0)
        {
            CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
        }
        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_M, PoseidonGoldilocksConstants::M, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_P, PoseidonGoldilocksConstants::P, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C, PoseidonGoldilocksConstants::C, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_S, PoseidonGoldilocksConstants::S, 507 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        }
        CHECKCUDAERR(cudaSetDevice(0));
    }
}

__device__ void hash_full_result_seq(gl64_t *state, const gl64_t *input, const gl64_t *GPU_C_GL, const gl64_t *GPU_S_GL, const gl64_t *GPU_M_GL, const gl64_t *GPU_P_GL)
{
    mymemcpy((uint64_t *)state, (uint64_t *)input, SPONGE_WIDTH);

    add_(state, GPU_C_GL);
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(GPU_C_GL[(r + 1) * SPONGE_WIDTH]));
        mvp_(state, GPU_M_GL);
    }

    pow7add_(state, &(GPU_C_GL[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    mvp_(state, GPU_P_GL);

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(state[0]);
        state[0] = state[0] + GPU_C_GL[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        gl64_t s0 = dot_(state, &(GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r]));
        gl64_t W_[SPONGE_WIDTH];
        prod_(W_, state[0], &(GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        add_(state, W_);
        state[0] = s0;
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(GPU_C_GL[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        mvp_(state, GPU_M_GL);
    }
    pow7_(&(state[0]));
    mvp_(state, GPU_M_GL);
}

/* --- integration --- */

__device__ void linear_hash_one(gl64_t *output, gl64_t *input, uint32_t size, int tid)
{
    u32 remaining = size;
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_S_SM[507];
    __shared__ gl64_t GPU_M_SM[144];
    __shared__ gl64_t GPU_P_SM[144];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_S_SM, GPU_S, 507);
        mymemcpy((uint64_t *)GPU_M_SM, GPU_M, 144);
        mymemcpy((uint64_t *)GPU_P_SM, GPU_P, 144);
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
        hash_full_result_seq(state, state, GPU_C_SM, GPU_S_SM, GPU_M_SM, GPU_P_SM);
        remaining -= n;
    }
    mymemcpy((uint64_t *)output, (uint64_t *)state, CAPACITY);
}

__device__ void linear_partial_hash_one(gl64_t *input, uint32_t size, gl64_t *state, int tid)
{
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_S_SM[507];
    __shared__ gl64_t GPU_M_SM[144];
    __shared__ gl64_t GPU_P_SM[144];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_S_SM, GPU_S, 507);
        mymemcpy((uint64_t *)GPU_M_SM, GPU_M, 144);
        mymemcpy((uint64_t *)GPU_P_SM, GPU_P, 144);
    }
    __syncthreads();

    u32 remaining = size;

    while (remaining)
    {
        mymemcpy((uint64_t *)(state + RATE), (uint64_t *)state, CAPACITY);
        u32 n = (remaining < RATE) ? remaining : RATE;
        mymemset((uint64_t *)&state[n], 0, (RATE - n));
        mymemcpy((uint64_t *)state, (uint64_t *)(input + (size - remaining)), n);
        hash_full_result_seq(state, state, GPU_C_SM, GPU_S_SM, GPU_M_SM, GPU_P_SM);
        remaining -= n;
    }
}

__global__ void linear_hash_gpu(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *inp = (gl64_t *)(input + tid * size);
    gl64_t *out = (gl64_t *)(output + tid * CAPACITY);
    linear_hash_one(out, inp, size, threadIdx.x);
}

__global__ void linear_partial_init_hash_gpu(uint64_t *gstate, int32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *state = (gl64_t *)(gstate + tid * SPONGE_WIDTH);
    memset(state, 0, SPONGE_WIDTH * sizeof(gl64_t));
}

__global__ void linear_partial_hash_gpu(uint64_t *input, uint32_t num_cols, uint32_t num_rows, uint64_t *gstate, uint32_t hash_per_thread = 1)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    for (uint32_t i = 0; i < hash_per_thread; i++)
    {
        gl64_t *inp = (gl64_t *)(input + (tid * hash_per_thread + i) * num_cols);
        gl64_t *state = (gl64_t *)(gstate + (tid * hash_per_thread + i) * SPONGE_WIDTH);
        linear_partial_hash_one(inp, num_cols, state, threadIdx.x);
    }
}

__global__ void linear_partial_copy_hash_gpu(uint64_t *output, uint64_t *gstate, uint32_t num_cols, uint32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows)
        return;

    gl64_t *state = (gl64_t *)(gstate + tid * SPONGE_WIDTH);
    gl64_t *out = (gl64_t *)(output + tid * CAPACITY);
    mymemcpy((uint64_t *)out, (uint64_t *)state, CAPACITY);
    /*
    if (num_cols > 0)
    {
        mymemcpy((uint64_t*)out, (uint64_t*)state, CAPACITY);
    }
    else
    {
        mymemset((uint64_t*)out, 0, CAPACITY);
    }
    */
}

__device__ void hash_one(gl64_t *state, gl64_t *const input, int tid)
{
    __shared__ gl64_t GPU_C_SM[118];
    __shared__ gl64_t GPU_S_SM[507];
    __shared__ gl64_t GPU_M_SM[144];
    __shared__ gl64_t GPU_P_SM[144];

    if (tid == 0)
    {
        mymemcpy((uint64_t *)GPU_C_SM, GPU_C, 118);
        mymemcpy((uint64_t *)GPU_S_SM, GPU_S, 507);
        mymemcpy((uint64_t *)GPU_M_SM, GPU_M, 144);
        mymemcpy((uint64_t *)GPU_P_SM, GPU_P, 144);
    }
    __syncthreads();

    gl64_t aux[SPONGE_WIDTH];
    hash_full_result_seq(aux, input, GPU_C_SM, GPU_S_SM, GPU_M_SM, GPU_P_SM);
    mymemcpy((uint64_t *)state, (uint64_t *)aux, CAPACITY);
}

__global__ void hash_gpu(uint32_t nextN, uint32_t nextIndex, uint32_t pending, uint64_t *cursor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input, 0, SPONGE_WIDTH);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor[nextIndex + tid * RATE], RATE);
    hash_one((gl64_t *)(&cursor[nextIndex + (pending + tid) * CAPACITY]), pol_input, threadIdx.x);
}

void merkletree_cuda_batch(Goldilocks::Element *tree, uint64_t *dst_gpu_tree, uint64_t *gpu_tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, uint64_t dim, uint32_t const gpu_id)
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
        linear_hash_gpu<<<ceil(rowsBatch / (1.0 * TPB)), TPB, 0, gpu_stream>>>(gtree_ptr, gpu_input, num_cols * dim, rowsBatch);
        gtree_ptr += (rowsBatch * CAPACITY);
    }
    if (rowsLastBatch > 0)
    {
        CHECKCUDAERR(cudaMemcpyAsync(gpu_input, (uint64_t *)iptr, rowsLastBatch * num_cols * dim * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream));
        linear_hash_gpu<<<ceil(rowsLastBatch / (1.0 * TPB)), TPB, 0, gpu_stream>>>(gtree_ptr, gpu_input, num_cols * dim, rowsLastBatch);
    }
    if (dst_gpu_tree != NULL)
    {
        CHECKCUDAERR(cudaMemcpyPeerAsync(dst_gpu_tree, 0, gpu_tree, gpu_id, numElementsTree * sizeof(uint64_t), gpu_stream));
    }
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream));
    CHECKCUDAERR(cudaFree(gpu_input));
    CHECKCUDAERR(cudaStreamDestroy(gpu_stream));
}

void merkletree_cuda_multi_gpu(Goldilocks::Element *tree, uint64_t *dst_gpu_tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim, uint32_t const ngpu)
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
            merkletree_cuda_batch(tree + (d * numElementsTreeDevice), dst_gpu_tree + (d * numElementsTreeDevice), gpu_tree[d], input + (d * rowsDevice * num_cols * dim), num_cols, rowsDevice, dim, d);
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
            linear_hash_gpu<<<ceil(rowsDevice / (1.0 * TPB)), TPB, 0, gpu_stream[d]>>>(gpu_tree[d], gpu_input[d], num_cols * dim, rowsDevice);
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

void PoseidonGoldilocks::merkletree_cuda_multi_gpu_full(Goldilocks::Element *tree, uint64_t **gpu_inputs, uint64_t **gpu_trees, void *v_gpu_streams, uint64_t num_cols, uint64_t num_rows, uint64_t num_rows_device, uint32_t const ngpu, uint64_t dim)
{
    cudaStream_t *gpu_streams = (cudaStream_t *)v_gpu_streams;
    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows);
    uint64_t numElementsTreeDevice = num_rows_device * CAPACITY;

    uint64_t *gpu_final_tree;
    CHECKCUDAERR(cudaSetDevice(0));
    CHECKCUDAERR(cudaMalloc(&gpu_final_tree, numElementsTree * sizeof(uint64_t)));

    init_gpu_const(ngpu);

#ifdef GPU_TIMING
    TimerStart(merkletree_cuda_multi_gpu_kernel);
#endif
#pragma omp parallel for num_threads(ngpu)
    for (uint32_t d = 0; d < ngpu; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        linear_hash_gpu<<<ceil(num_rows_device / (1.0 * TPB)), TPB, 0, gpu_streams[d]>>>(gpu_trees[d], gpu_inputs[d], num_cols * dim, num_rows_device);
    }
#ifdef GPU_TIMING
    for (uint32_t d = 0; d < ngpu; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_streams[d]));
    }
    TimerStopAndLog(merkletree_cuda_multi_gpu_kernel);
    TimerStart(merkletree_cuda_multi_gpu_copyPeer2Peer);
#endif
#pragma omp parallel for num_threads(ngpu)
    for (uint32_t d = 0; d < ngpu; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_final_tree + (d * numElementsTreeDevice), 0, gpu_trees[d], d, numElementsTreeDevice * sizeof(uint64_t), gpu_streams[d]));
        // CHECKCUDAERR(cudaStreamSynchronize(gpu_streams[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(merkletree_cuda_multi_gpu_copyPeer2Peer);
    TimerStart(merkletree_cuda_multi_gpu_cleanup);
#endif
#pragma omp parallel for num_threads(ngpu)
    for (uint32_t d = 0; d < ngpu; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamSynchronize(gpu_streams[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(merkletree_cuda_multi_gpu_cleanup);
#endif

    // Build the merkle tree
    CHECKCUDAERR(cudaSetDevice(0));
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;
    int actual_tpb, actual_blks;
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
        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, gpu_final_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
    uint64_t *buffer = get_pinned_mem();
    CHECKCUDAERR(cudaMemcpy(buffer, gpu_final_tree, numElementsTree * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t nthreads = omp_get_max_threads() / 2;
    uint64_t piece = numElementsTree / nthreads;
    uint64_t last_piece = numElementsTree - (nthreads - 1) * piece;
#pragma omp parallel for num_threads(nthreads)
    for (uint64_t d = 0; d < nthreads; d++)
    {
        uint64_t cur_piece = d == nthreads - 1 ? last_piece : piece;
        memcpy(tree + d * piece, buffer + d * piece, cur_piece * sizeof(uint64_t));
    }

    CHECKCUDAERR(cudaFree(gpu_final_tree));
}

void PoseidonGoldilocks::merkletree_cuda_multi_gpu_steps(uint64_t **gpu_inputs, uint64_t **gpu_trees, void *v_gpu_streams, uint64_t num_cols, uint64_t num_rows_device, uint32_t const ngpu, uint64_t dim)
{
    cudaStream_t *gpu_streams = (cudaStream_t *)v_gpu_streams;

    init_gpu_const(ngpu);

#ifdef GPU_TIMING
    TimerStart(merkletree_cuda_multi_gpu_kernel);
#endif
#pragma omp parallel for num_threads(ngpu)
    for (uint32_t d = 0; d < ngpu; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        linear_hash_gpu<<<ceil(num_rows_device / (1.0 * TPB)), TPB, 0, gpu_streams[d]>>>(gpu_trees[d], gpu_inputs[d], num_cols * dim, num_rows_device);
    }
#ifdef GPU_TIMING
    for (uint32_t d = 0; d < ngpu; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_streams[d]));
    }
    TimerStopAndLog(merkletree_cuda_multi_gpu_kernel);
#endif
}

void PoseidonGoldilocks::merkletree_cuda_multi_gpu_final(Goldilocks::Element *tree, uint64_t *final_tree, void *v_gpu_streams, uint64_t num_rows)
{
    cudaStream_t *gpu_streams = (cudaStream_t *)v_gpu_streams;
    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows);

    uint64_t *gpu_final_tree;
    CHECKCUDAERR(cudaSetDevice(0));
    CHECKCUDAERR(cudaMalloc(&gpu_final_tree, numElementsTree * sizeof(uint64_t)));

    init_gpu_const(0);

#ifdef GPU_TIMING
    TimerStart(merkletree_cuda_multi_gpu_final_copy2gpu);
#endif

    CHECKCUDAERR(cudaMemcpyAsync(gpu_final_tree, final_tree, num_rows * CAPACITY * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_streams[0]));

#ifdef GPU_TIMING
    TimerStopAndLog(merkletree_cuda_multi_gpu_final_copy2gpu);
#endif

    CHECKCUDAERR(cudaStreamSynchronize(gpu_streams[0]));

    // Build the merkle tree
    CHECKCUDAERR(cudaSetDevice(0));
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;
    int actual_tpb, actual_blks;
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
        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, gpu_final_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
    uint64_t *buffer = get_pinned_mem();
    CHECKCUDAERR(cudaMemcpy(buffer, gpu_final_tree, numElementsTree * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t nthreads = omp_get_max_threads() / 2;
    uint64_t piece = numElementsTree / nthreads;
    uint64_t last_piece = numElementsTree - (nthreads - 1) * piece;
#pragma omp parallel for num_threads(nthreads)
    for (uint64_t d = 0; d < nthreads; d++)
    {
        uint64_t cur_piece = d == nthreads - 1 ? last_piece : piece;
        memcpy(tree + d * piece, buffer + d * piece, cur_piece * sizeof(uint64_t));
    }

    CHECKCUDAERR(cudaFree(gpu_final_tree));
}

void PoseidonGoldilocks::merkletree_cuda_async(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows)
{

    printf("merkletree_cuda_async, num_rows:%lu, num_cols:%lu\n", num_rows, num_cols);
    uint64_t num_rows_device = 1 << 14;
    if (num_rows < num_rows_device)
    {
#ifdef __AVX512__
        PoseidonGoldilocks::merkletree_avx512(tree, input, num_cols, num_rows);
#else
        PoseidonGoldilocks::merkletree_avx(tree, input, num_cols, num_rows);
#endif
        return;
    }
    const int nStreams = 2;
    const int MAX_GPUS = 8;
    cudaStream_t cuda_streams[nStreams * MAX_GPUS];
    cudaEvent_t events[nStreams * MAX_GPUS];
    uint64_t *gpu_input[nStreams * MAX_GPUS];
    uint64_t *gpu_subtree[nStreams * MAX_GPUS];
    uint64_t *gpu_final_tree;

    int nDevices;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));

    init_gpu_const(nDevices);

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows);
    CHECKCUDAERR(cudaSetDevice(0));
    CHECKCUDAERR(cudaMalloc(&gpu_final_tree, numElementsTree * sizeof(uint64_t)));

    uint64_t numElementsTreeDevice = MerklehashGoldilocks::getTreeNumElements(num_rows_device);

    for (int d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        for (int s = 0; s < nStreams; s++)
        {
            int idx = s * nDevices + d;
            CHECKCUDAERR(cudaStreamCreate(cuda_streams + idx));
            CHECKCUDAERR(cudaEventCreate(events + idx));
            CHECKCUDAERR(cudaMalloc(&gpu_input[idx], num_rows_device * num_cols * sizeof(uint64_t)));
            CHECKCUDAERR(cudaMalloc(&gpu_subtree[idx], num_rows_device * CAPACITY * sizeof(uint64_t)));
        }
    }

    assert(num_rows % num_rows_device == 0);
    int pack_count = num_rows / num_rows_device;

    printf("pack_count = %d\n", pack_count);

    for (int i = 0; i < pack_count; i++)
    {
        int d = i % nDevices;
        CHECKCUDAERR(cudaSetDevice(d));
        int idx = i % (nDevices * nStreams);
        cudaStream_t stream = cuda_streams[idx];
        CHECKCUDAERR(cudaEventSynchronize(events[idx]));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_input[idx], input + i * num_rows_device * num_cols, num_rows_device * num_cols * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        linear_hash_gpu<<<num_rows_device / TPB, TPB, 0, stream>>>(gpu_subtree[idx], gpu_input[idx], num_cols, num_rows_device);
        CHECKCUDAERR(cudaGetLastError());
        CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_final_tree + i * num_rows_device * CAPACITY, 0, gpu_subtree[idx], d, num_rows_device * CAPACITY * sizeof(uint64_t), stream));
        CHECKCUDAERR(cudaEventRecord(events[idx], stream));
    }

    for (int idx = 0; idx < nStreams * nDevices; idx++)
    {
        CHECKCUDAERR(cudaEventSynchronize(events[idx]));
    }

    // Build the merkle tree
    CHECKCUDAERR(cudaSetDevice(0));
    uint64_t pending = num_rows;
    uint64_t nextN = floor((pending - 1) / 2) + 1;
    uint64_t nextIndex = 0;
    int actual_tpb, actual_blks;
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
        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, gpu_final_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }

    CHECKCUDAERR(cudaMemcpy(tree, gpu_final_tree, numElementsTree * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    /*
    uint64_t *buffer = get_pinned_mem();
    CHECKCUDAERR(cudaMemcpy(buffer, gpu_final_tree, numElementsTree * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t nthreads = omp_get_max_threads()/2;
    uint64_t piece = numElementsTree / nthreads;
    uint64_t last_piece = numElementsTree - (nthreads -1) * piece;
#pragma omp parallel for num_threads(nthreads)
    for (uint64_t d = 0; d < nthreads; d++) {
        uint64_t cur_piece = d == nthreads -1 ? last_piece: piece;
        memcpy(tree+d*piece, buffer+d*piece, cur_piece * sizeof(uint64_t));
    }
    */

    cudaFree(gpu_final_tree);
    for (int s = 0; s < nStreams * nDevices; s++)
    {
        cudaFree(gpu_input[s]);
        cudaFree(gpu_subtree[s]);
        cudaEventDestroy(events[s]);
        cudaStreamDestroy(cuda_streams[s]);
    }
}

void PoseidonGoldilocks::merkletree_cuda(Goldilocks::Element *tree, Goldilocks::Element *input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    // is the input < 2 GB -> run on CPU
    if (num_rows * num_cols * dim <= (1ul << 32))
    {
#ifdef __AVX512__
        PoseidonGoldilocks::merkletree_avx512(tree, input, num_cols, num_rows, nThreads, dim);
#else
        PoseidonGoldilocks::merkletree_avx(tree, input, num_cols, num_rows, nThreads, dim);
#endif
        return;
    }

    uint64_t *gpu_tree = NULL;
    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows); // includes CAPACITY
    init_gpu_const();
    u32 actual_tpb = TPB;
    u32 actual_blks = num_rows / TPB + 1;

    // is the input > 1 GB?
    if (num_rows * num_cols * dim > 134217728)
    {
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        if (nDevices > 1)
        {
            CHECKCUDAERR(cudaSetDevice(0));
            CHECKCUDAERR(cudaMalloc(&gpu_tree, numElementsTree * sizeof(uint64_t)));
            merkletree_cuda_multi_gpu(tree, gpu_tree, input, num_cols, num_rows, nThreads, dim, nDevices);
        }
        else
        {
            CHECKCUDAERR(cudaSetDevice(0));
            CHECKCUDAERR(cudaMalloc(&gpu_tree, numElementsTree * sizeof(uint64_t)));
            merkletree_cuda_batch(tree, NULL, gpu_tree, input, num_cols, num_rows, dim, 0);
        }
    }
    else
    {
#ifdef FDEBUG
        printf("On GPU, 1 batch\n");
#endif
        CHECKCUDAERR(cudaSetDevice(0));
        uint64_t *gpu_input;
        CHECKCUDAERR(cudaMalloc(&gpu_tree, numElementsTree * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_input, num_rows * num_cols * dim * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_input, (uint64_t *)input, num_rows * num_cols * dim * sizeof(uint64_t), cudaMemcpyHostToDevice));
        if (num_rows < TPB)
        {
            actual_tpb = num_rows;
            actual_blks = 1;
        }
        linear_hash_gpu<<<actual_blks, actual_tpb>>>(gpu_tree, gpu_input, num_cols * dim, num_rows);
        CHECKCUDAERR(cudaFree(gpu_input));
    }

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
        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, gpu_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
    CHECKCUDAERR(cudaMemcpy(tree, gpu_tree, numElementsTree * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(gpu_tree));
}

void PoseidonGoldilocks::partial_hash_init_gpu(uint64_t **state, uint32_t num_rows, uint32_t ngpus)
{
    init_gpu_const();
    int nDevices;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    nDevices = (ngpus < nDevices) ? ngpus : nDevices;
    for (int i = 0; i < nDevices; i++)
    {
        CHECKCUDAERR(cudaSetDevice(i));
        linear_partial_init_hash_gpu<<<ceil(num_rows / (1.0 * TPB)), TPB>>>(state[i], num_rows);
    }
    CHECKCUDAERR(cudaSetDevice(0));
}

void PoseidonGoldilocks::merkletree_cuda_gpudata(Goldilocks::Element *tree, uint64_t *gpu_input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    uint64_t *gpu_tree = NULL;
    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows); // includes CAPACITY
    init_gpu_const();
    u32 actual_tpb = TPB;
    u32 actual_blks = num_rows / TPB + 1;

    CHECKCUDAERR(cudaSetDevice(0));
    CHECKCUDAERR(cudaMalloc(&gpu_tree, numElementsTree * sizeof(uint64_t)));
    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    linear_hash_gpu<<<actual_blks, actual_tpb>>>(gpu_tree, gpu_input, num_cols * dim, num_rows);

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
        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, gpu_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
    CHECKCUDAERR(cudaMemcpy(tree, gpu_tree, numElementsTree * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(gpu_tree));
}

void PoseidonGoldilocks::merkletree_cuda_gpudata_inplace(uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    CHECKCUDAERR(cudaSetDevice(0));
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time0 = omp_get_wtime();
    if (num_rows == 0)
    {
        return;
    }

    init_gpu_const();
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
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time1 = omp_get_wtime();
    std::cout << "          check dins 1: " << time1 - time0 << std::endl;
    linear_hash_gpu<<<actual_blks, actual_tpb>>>(*d_tree, d_input, num_cols * dim, num_rows);
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time2 = omp_get_wtime();
    std::cout << "          check dins 2: " << time2 - time1 << std::endl;

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
        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, *d_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time3 = omp_get_wtime();
    std::cout << "          check dins 3: " << time3 - time2 << std::endl;
}

void PoseidonGoldilocks::partial_hash_gpu(uint64_t *input, uint32_t num_cols, uint32_t num_rows, uint64_t *state)
{
    linear_partial_hash_gpu<<<ceil((num_rows / 2048) / (1.0 * TPB)), TPB>>>(input, num_cols, num_rows, state, 2048);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////

void PoseidonGoldilocks::merkletree_cuda_coalesced(uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    CHECKCUDAERR(cudaSetDevice(0));
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time0 = omp_get_wtime();
    if (num_rows == 0)
    {
        return;
    }

    init_gpu_const(); // this needs to be done only once !!
    u32 actual_tpb = TPB;
    u32 actual_blks = (num_rows + TPB - 1) / TPB;

    CHECKCUDAERR(cudaSetDevice(0));
    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows); // includes CAPACITY
    CHECKCUDAERR(cudaMalloc(d_tree, numElementsTree * sizeof(uint64_t)));

    if (num_rows < TPB)
    {
        actual_tpb = num_rows;
        actual_blks = 1;
    }
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time1 = omp_get_wtime();
    std::cout << "          check dins 1: " << time1 - time0 << std::endl;
    linear_hash_gpu_coalesced<<<actual_blks, actual_tpb, actual_tpb * 12 * 8>>>(*d_tree, d_input, num_cols * dim, num_rows); // rick: el 12 aqeust harcoded no please!!
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time2 = omp_get_wtime();
    std::cout << "          check dins 2: " << time2 - time1 << std::endl;

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
        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, nextIndex, pending, *d_tree);
        nextIndex += pending * CAPACITY;
        pending = pending / 2;
        nextN = floor((pending - 1) / 2) + 1;
    }
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time3 = omp_get_wtime();
    std::cout << "          check dins 3: " << time3 - time2 << std::endl;
}

__global__ void hash_gpu(uint32_t nextN, uint64_t *cursor_in, uint64_t *cursor_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN)
        return;

    gl64_t pol_input[SPONGE_WIDTH];
    mymemset((uint64_t *)pol_input + RATE, 0, CAPACITY);
    mymemcpy((uint64_t *)pol_input, (uint64_t *)&cursor_in[tid * RATE], RATE);
    hash_one((gl64_t *)(&cursor_out[tid * CAPACITY]), pol_input, threadIdx.x);
}

void PoseidonGoldilocks::merkletree_cuda_streams(uint64_t **d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int nThreads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    CHECKCUDAERR(cudaSetDevice(0));
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time0 = omp_get_wtime();

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(num_rows);
    CHECKCUDAERR(cudaMalloc(d_tree, numElementsTree * sizeof(uint64_t)));

    init_gpu_const(); // this needs to be done only once !!
    // rick: we are assuming here that TPB is a power of 2
    int num_streams = 8; // note: must be a power of two
    uint64_t rows_per_stream = num_rows / num_streams;
    uint32_t actual_tpb = TPB;
    std::cout << "actual_tpb: " << actual_tpb << std::endl;
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

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time1 = omp_get_wtime();
    std::cout << "          check dins 1: " << time1 - time0 << std::endl;

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
            linear_hash_gpu_coalesced<<<actual_blks, actual_tpb, actual_tpb * SPONGE_WIDTH * sizeof(gl64_t), streams[i]>>>(
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

                hash_gpu<<<actual_blks, actual_tpb, 0, streams[i]>>>(nextN, cursor_in, cursor_out);
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
    CHECKCUDAERR(cudaDeviceSynchronize());
    double time2 = omp_get_wtime();
    std::cout << "          check dins 2: " << time2 - time1 << std::endl;

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

        hash_gpu<<<actual_blks, actual_tpb>>>(nextN, cursor_in, cursor_out);
        cursor_in = cursor_out;
        cursor_out = cursor_out + nextN * CAPACITY;
        nextN >>= 1;
    }

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time3 = omp_get_wtime();
    std::cout << "          check dins 3: " << time3 - time2 << std::endl;
}

__device__ __noinline__ void pow7(gl64_t &x)
{
    gl64_t x2 = x * x;
    gl64_t x3 = x * x2;
    gl64_t x4 = x2 * x2;
    x = x3 * x4;
}

__device__ __noinline__ static void pow7_state_()
{
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
        pow7(scratchpad[i * blockDim.x + threadIdx.x]);
}

__device__ __noinline__ void dotp(const gl64_t *__restrict__ mat, const gl64_t *__restrict__ state, uint32_t i)
{
    scratchpad[i * blockDim.x + threadIdx.x] = mat[i] * state[0];
#pragma unroll 1
    for (int j = 1; j < SPONGE_WIDTH; j++)
    {
        scratchpad[i * blockDim.x + threadIdx.x] += (mat[12 * j + i] * state[j]);
    }
}
__device__ __noinline__ void mvp_state_(const gl64_t *__restrict__ mat)
{
    gl64_t state[SPONGE_WIDTH];
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
        state[i] = scratchpad[i * blockDim.x + threadIdx.x];

#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
    {
        dotp(mat, state, i);
    }
}

__device__ __noinline__ void add_state_(const gl64_t C[SPONGE_WIDTH])
{
#pragma unroll
    for (uint32_t i = 0; i < SPONGE_WIDTH; i++)
        scratchpad[i * blockDim.x + threadIdx.x] += C[i];
}

__device__ __noinline__ void poseidon_load(const uint64_t *in, uint32_t col, uint32_t ncols,
                                           uint32_t col_stride, size_t row_stride = 1)
{
    const size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    in += tid * col_stride + col * row_stride;

    for (uint32_t i = 0; i < RATE; i++, in += row_stride)
        if (i < ncols)
        {
            scratchpad[i * blockDim.x + threadIdx.x] = __ldcv((uint64_t *)in);
        }
        else
        {
            scratchpad[i * blockDim.x + threadIdx.x].set_val(0);
        }
}

__device__ __noinline__ void poseidon_store(uint64_t *__restrict__ out, uint32_t col_stride, size_t row_stride = 1)
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

__device__ __noinline__ void poseidon_hash_loop(const uint64_t *__restrict__ in, uint32_t ncols)
{
    if (ncols <= CAPACITY)
    {
        poseidon_load(in, 0, ncols, ncols);
    }
    else
    {
        for (uint32_t col = 0;;)
        {
            uint32_t delta = min(ncols - col, RATE);
            poseidon_load(in, col, delta, ncols);
            poseidon_hash();
            if ((col += RATE) >= ncols)
                break;
#pragma unroll
            for (uint32_t i = 0; i < CAPACITY; i++)
                scratchpad[(i + RATE) * blockDim.x + threadIdx.x] = scratchpad[i * blockDim.x + threadIdx.x];
        }
    }
}
__device__ __noinline__ void parcial_rounds()
{
    const gl64_t *GPU_C_GL = (gl64_t *)GPU_C;
    const gl64_t *GPU_S_GL = (gl64_t *)GPU_S;
#pragma unroll 1
    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(scratchpad[threadIdx.x]);
        gl64_t p0 = scratchpad[threadIdx.x] + GPU_C_GL[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        scratchpad[threadIdx.x] = p0 * GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r];
#pragma unroll 1
        for (uint32_t j = 1; j < SPONGE_WIDTH; j++)
        {
            scratchpad[threadIdx.x] += scratchpad[j * blockDim.x + threadIdx.x] * GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r + j];
            scratchpad[j * blockDim.x + threadIdx.x] += p0 * GPU_S_GL[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + j];
        }
    }
}

__device__ __noinline__ void poseidon_hash()
{
    const gl64_t *GPU_C_GL = (gl64_t *)GPU_C;
    const gl64_t *GPU_M_GL = (gl64_t *)GPU_M;
    const gl64_t *GPU_S_GL = (gl64_t *)GPU_S;
    const gl64_t *GPU_P_GL = (gl64_t *)GPU_P;

    add_state_(GPU_C_GL);
#pragma unroll 1
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_state_();
        add_state_(&(GPU_C_GL[(r + 1) * SPONGE_WIDTH]));
        mvp_state_(GPU_M_GL);
    }
    pow7_state_();
    add_state_(&(GPU_C_GL[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    mvp_state_(GPU_P_GL);

    parcial_rounds();

#pragma unroll 1
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_state_();
        add_state_(&(GPU_C_GL[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        mvp_state_(GPU_M_GL);
    }
    pow7_state_();
    mvp_state_(GPU_M_GL);
}

__global__ void linear_hash_gpu_coalesced(uint64_t *__restrict__ output, uint64_t *__restrict__ input, uint32_t size, uint32_t num_rows)
{
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x].set_val(0);

    poseidon_hash_loop(input, size);
    poseidon_store(output, CAPACITY);
}

// funciont not tested
__device__ void generate_tree()
{
    uint32_t numThreads = blockDim.x;
    uint32_t offset_ant = 0;
    uint32_t offset = 0;
    while (numThreads > 1)
    {
        numThreads >>= 1;
        if (threadIdx.x >= offset && threadIdx.x < offset + numThreads)
        {
            uint32_t thread_relative = threadIdx.x - offset;
            uint32_t pos_previous = offset_ant + thread_relative << 1;

#pragma unroll
            for (uint32_t i = 0; i < CAPACITY; i++)
            {
                uint32_t idx = i * blockDim.x + threadIdx.x;
                uint32_t prev_idx = i * blockDim.x + pos_previous;

                gl64_t val1 = scratchpad[prev_idx];
                gl64_t val2 = scratchpad[prev_idx + 1];

                scratchpad[idx] = val1;
                scratchpad[(i + CAPACITY) * blockDim.x + threadIdx.x] = val2;
            }
            poseidon_hash();
        }
        __syncthreads();

        offset_ant = offset;
        offset += numThreads;
    }
}

// function not tested
__device__ void generate_1level_tree()
{

    if (threadIdx.x < blockDim.x >> 1)
    {
#pragma unroll
        for (uint32_t i = 0; i < CAPACITY; i++)
        {
            uint32_t idx = i * blockDim.x + threadIdx.x;
            uint32_t idx2 = idx >> 1;

            gl64_t val1 = scratchpad[idx2];
            gl64_t val2 = scratchpad[idx2 + 1];

            scratchpad[idx] = val1;
            scratchpad[(i + CAPACITY) * blockDim.x + threadIdx.x] = val2;
        }
        poseidon_hash();
    }
}

// function not tested
// rick: check limits on load and store
__global__ void linear_hash_gpu_tree(uint64_t *output, uint64_t *input, uint32_t size, uint32_t num_rows)
{
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x].set_val(0);

    poseidon_hash_loop(input, size);
    poseidon_store(output, CAPACITY);
#pragma unroll
    for (uint32_t i = 0; i < CAPACITY; i++)
        scratchpad[(i + RATE) * blockDim.x + threadIdx.x].set_val(0);
    generate_1level_tree();
}