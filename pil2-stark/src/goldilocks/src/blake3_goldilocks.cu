// ---------------------------------------------------------------------------
// Blake3GoldilocksGPU -- GPU BLAKE3 (arity 2)
// ---------------------------------------------------------------------------

#include "blake3_goldilocks.cuh"
#include "goldilocks_trace_layout.cuh"   // getBufferOffset (layout-aware), Layout enum
#include "cuda_utils.cuh"                // CHECKCUDAERR

#define TPB_B3 128

// ---------------------------------------------------------------------------
// Device leaf hashing: read one trace row through the requested layout and hash
// it (single chunk for <=128 cols, else the standard BLAKE3 chunk tree).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void b3_gather_block(const uint64_t *in, uint32_t base_col, uint32_t in_block,
                                                 uint64_t row, uint64_t nRows, uint32_t nCols, Layout layout,
                                                 uint32_t block[16])
{
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        uint64_t v = 0;
        if ((uint32_t)k < in_block)
        {
            uint32_t col = base_col + (uint32_t)k;
            // getBufferOffset handles RowMajor / ColMajor / ColMajorTiled per the
            // runtime layout (resolveLayout / fixedLayout), same as the Poseidon path.
            uint64_t off = getBufferOffset(row, col, nRows, nCols, layout);
            // Canonicalize: the LDE trace cells live in [0, 2^64); the verifier
            // re-hashes the mod-p value from the proof, so BLAKE3 must too.
            v = blake3core::to_canonical(in[off]);
        }
        block[2 * k]     = (uint32_t)v;
        block[2 * k + 1] = (uint32_t)(v >> 32);
    }
}

__device__ __forceinline__ void b3_chunk_gathered(const uint64_t *in, uint32_t base_col, uint32_t cu,
                                                   uint64_t counter, bool root, uint64_t row, uint64_t nRows,
                                                   uint32_t nCols, Layout layout, uint32_t cv[8])
{
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = blake3core::b3_iv(i);
    uint32_t nblocks = (cu + 7u) / 8u;
    if (nblocks == 0) nblocks = 1;
    uint32_t idx = 0, rem = cu;
    for (uint32_t b = 0; b < nblocks; ++b)
    {
        uint32_t in_block = (rem >= 8u) ? 8u : rem;
        uint32_t block[16];
        b3_gather_block(in, base_col + idx, in_block, row, nRows, nCols, layout, block);
        uint8_t flags = 0;
        if (b == 0)           flags |= blake3core::FLAG_CHUNK_START;
        if (b == nblocks - 1) { flags |= blake3core::FLAG_CHUNK_END; if (root) flags |= blake3core::FLAG_ROOT; }
        blake3core::compress_in_place(cv, block, (uint8_t)(in_block * 8u), counter, flags);
        idx += in_block; rem -= in_block;
    }
}

__device__ void b3_hash_row(const uint64_t *in, uint32_t nCols, uint64_t row, uint64_t nRows,
                            Layout layout, uint64_t out[4])
{
    uint32_t nchunks = (nCols + blake3core::CHUNK_U64 - 1) / blake3core::CHUNK_U64;
    if (nchunks <= 1)
    {
        uint32_t cv[8];
        b3_chunk_gathered(in, 0, nCols, 0ull, true, row, nRows, nCols, layout, cv);
        blake3core::pack4(cv, out);
        return;
    }
    uint32_t stack[blake3core::CV_STACK * 8];
    int slen = 0;
    uint32_t node[8];
    uint32_t base = 0, rem = nCols;
    for (uint32_t ci = 0; ci < nchunks; ++ci)
    {
        uint32_t cu = (rem >= blake3core::CHUNK_U64) ? blake3core::CHUNK_U64 : rem;
        b3_chunk_gathered(in, base, cu, (uint64_t)ci, false, row, nRows, nCols, layout, node);
        base += cu; rem -= cu;
        if (ci != nchunks - 1)
        {
            uint64_t total = (uint64_t)ci + 1;
            while ((total & 1ull) == 0)
            {
                uint32_t m[8];
                blake3core::parent_cv(&stack[(slen - 1) * 8], node, false, m);
                for (int i = 0; i < 8; ++i) node[i] = m[i];
                --slen; total >>= 1;
            }
            for (int i = 0; i < 8; ++i) stack[slen * 8 + i] = node[i];
            ++slen;
        }
        else
        {
            while (slen > 0)
            {
                uint32_t m[8];
                blake3core::parent_cv(&stack[(slen - 1) * 8], node, slen == 1, m);
                for (int i = 0; i < 8; ++i) node[i] = m[i];
                --slen;
            }
        }
    }
    blake3core::pack4(node, out);
}

__global__ void b3_linearHashKernel(uint64_t *__restrict__ out, const uint64_t *__restrict__ in,
                                    uint32_t num_cols, uint32_t num_rows, Layout layout)
{
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t dig[4];
    b3_hash_row(in, num_cols, row, num_rows, layout, dig);
    uint64_t *o = out + row * 4ull;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = dig[i];
}

// Internal node: arity child digests are contiguous in `cursor`.
__global__ void b3_merkleNodeKernel(uint64_t *cursor, uint64_t nextN, uint64_t nextIndex,
                                    uint64_t pending, uint32_t arity)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN) return;
    uint64_t base = nextIndex + tid * (uint64_t)arity * 4ull;
    uint64_t dig[4];
    blake3core::hash_le64(&cursor[base], arity * 4u, dig);
    uint64_t *o = &cursor[nextIndex + (pending + tid) * 4ull];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = dig[i];
}

__global__ void b3_grindingKernel(uint64_t *nonce, uint64_t *__restrict__ nonceBlock,
                                  uint64_t *__restrict__ input, uint64_t n_bits,
                                  uint64_t hashes_per_thread, uint64_t nonces_offset)
{
    uint64_t *shared_nonces = (uint64_t *)&scratchpad[0];

    if (nonces_offset != 0 && nonce[0] != UINT64_MAX) return;

    if (threadIdx.x == 0)
    {
        shared_nonces[0] = UINT64_MAX;
        if (blockIdx.x == 0) nonce[0] = UINT64_MAX;
        if (nonces_offset != 0)
        {
            for (int i = 0; i < (int)gridDim.x; ++i)
                if (nonceBlock[i] != UINT64_MAX)
                {
                    shared_nonces[0] = nonceBlock[i];
                    if (blockIdx.x == 0) nonce[0] = nonceBlock[i];
                    break;
                }
        }
    }
    __syncthreads();
    if (shared_nonces[0] != UINT64_MAX) return;

    nonceBlock[blockIdx.x] = UINT64_MAX;

    const uint64_t idx   = nonces_offset + ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * hashes_per_thread;
    const uint64_t level = 1ULL << (64 - n_bits);
    uint64_t locId = UINT64_MAX;

    uint64_t in8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t out8[8];
    for (uint64_t k = 0; k < hashes_per_thread; ++k)
    {
        uint64_t idx_k = idx + k;
        in8[0] = input[0]; in8[1] = input[1]; in8[2] = input[2]; in8[3] = idx_k;
        blake3core::permute8(in8, out8);
        if (out8[0] < level) { locId = idx_k; break; }
    }

    shared_nonces[threadIdx.x] = locId;
    __syncthreads();
    uint32_t alive = blockDim.x >> 1;
    while (alive > 0)
    {
        if (threadIdx.x < alive && shared_nonces[threadIdx.x + alive] < shared_nonces[threadIdx.x])
            shared_nonces[threadIdx.x] = shared_nonces[threadIdx.x + alive];
        __syncthreads();
        alive >>= 1;
    }
    if (threadIdx.x == 0) nonceBlock[blockIdx.x] = shared_nonces[0];
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------
void Blake3GoldilocksGPU::linearHash(uint64_t *d_hash_output, uint64_t *d_trace,
                                     uint64_t num_cols, uint64_t num_rows,
                                     Layout layout, cudaStream_t stream)
{
    if (num_rows == 0) return;
    uint32_t tpb = (num_rows < TPB_B3) ? (uint32_t)num_rows : TPB_B3;
    uint32_t blks = (uint32_t)((num_rows + TPB_B3 - 1) / TPB_B3);
    b3_linearHashKernel<<<blks, tpb, 0, stream>>>(
        d_hash_output, d_trace, (uint32_t)num_cols, (uint32_t)num_rows, layout);
    CHECKCUDAERR(cudaGetLastError());
}

void Blake3GoldilocksGPU::merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                                     uint64_t num_cols, uint64_t num_rows,
                                     Layout layout, cudaStream_t stream)
{
    if (num_rows == 0) return;
    linearHash(d_tree, d_input, num_cols, num_rows, layout, stream);

    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;
    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync(d_tree + nextIndex + pending * CAPACITY, 0,
                                         extraZeros * CAPACITY * sizeof(uint64_t), stream));

        uint32_t tpb = (nextN < TPB_B3) ? (uint32_t)nextN : TPB_B3;
        uint32_t blks = (nextN < TPB_B3) ? 1u : (uint32_t)(nextN / TPB_B3 + 1);
        b3_merkleNodeKernel<<<blks, tpb, 0, stream>>>(d_tree, nextN, nextIndex,
                                                      pending + extraZeros, arity);

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

void Blake3GoldilocksGPU::merkletreeReduce(uint64_t *d_root, uint64_t *d_input,
                                           uint64_t num_elements, uint64_t arity,
                                           cudaStream_t stream)
{
    uint64_t numNodes = num_elements;
    uint64_t nodesLevel = num_elements;
    while (nodesLevel > 1)
    {
        uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
        numNodes += extraZeros;
        numNodes += (nodesLevel + (arity - 1)) / arity;
        nodesLevel = (nodesLevel + (arity - 1)) / arity;
    }

    uint64_t *d_tree;
    CHECKCUDAERR(cudaMalloc((void **)&d_tree, numNodes * CAPACITY * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyAsync(d_tree, d_input, num_elements * CAPACITY * sizeof(uint64_t),
                                 cudaMemcpyDeviceToDevice, stream));

    uint64_t pending = num_elements;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;
    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            CHECKCUDAERR(cudaMemsetAsync(d_tree + nextIndex + pending * CAPACITY, 0,
                                         extraZeros * CAPACITY * sizeof(uint64_t), stream));

        uint32_t tpb = (nextN < TPB_B3) ? (uint32_t)nextN : TPB_B3;
        uint32_t blks = (nextN < TPB_B3) ? 1u : (uint32_t)(nextN / TPB_B3 + 1);
        b3_merkleNodeKernel<<<blks, tpb, 0, stream>>>(d_tree, nextN, nextIndex,
                                                      pending + extraZeros, (uint32_t)arity);

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());

    CHECKCUDAERR(cudaMemcpyAsync(d_root, d_tree + nextIndex, CAPACITY * sizeof(uint64_t),
                                 cudaMemcpyDeviceToDevice, stream));
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    CHECKCUDAERR(cudaFree(d_tree));
}

void Blake3GoldilocksGPU::grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock,
                                   const uint64_t *d_in, const uint32_t n_bits,
                                   cudaStream_t stream)
{
    uint64_t log_launch_iters = 7;
    uint64_t launch_iters = 1ULL << log_launch_iters;
    uint64_t log_N = BLAKE3_GRIND_BITS;
    uint64_t N = 1ULL << log_N;
    uint64_t security = 128;

    double eps                 = ldexp(1.0, -int(n_bits));
    double totalHashesRequired = -double(security) * log(2.0) / log1p(-eps);
    uint64_t log_totalHashesRequired = (uint64_t)ceil(log2(totalHashesRequired));
    uint64_t log_hashesPerThread = (log_totalHashesRequired > log_launch_iters + log_N)
                                   ? log_totalHashesRequired - log_launch_iters - log_N : 0;
    uint64_t hashesPerThread = 1ULL << log_hashesPerThread;

    dim3 blockSize(BLAKE3_GRIND_BLOCKS);
    dim3 gridSize(BLAKE3_GRIND_GRID);
    size_t shared_mem_size = blockSize.x * sizeof(uint64_t);

    uint64_t nonces_offset = 0;
    uint64_t nonces_per_iteration = N * hashesPerThread;
    for (uint64_t i = 0; i < launch_iters; ++i)
    {
        b3_grindingKernel<<<gridSize, blockSize, shared_mem_size, stream>>>(
            d_nonce, d_nonceBlock, (uint64_t *)d_in, n_bits, hashesPerThread, nonces_offset);
        nonces_offset += nonces_per_iteration;
    }
}
