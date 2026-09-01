// ---------------------------------------------------------------------------
// Sha256GoldilocksGPU -- GPU SHA-256 (arity 2)
// ---------------------------------------------------------------------------

#include "sha256_goldilocks.cuh"
#include "goldilocks_trace_layout.cuh"   // getBufferOffset (layout-aware), Layout enum
#include "cuda_utils.cuh"                // CHECKCUDAERR

#define TPB_SHA 128

// Layout-aware block gather: the core's fill_block_le64 reads contiguous words,
// but a trace row is contiguous only under RowMajor, so resolve each column
// through getBufferOffset as the BLAKE3 and Poseidon leaf kernels do.
__device__ __forceinline__ void sha_gather_block(uint32_t W[16], const uint64_t *in,
                                                 uint32_t base_col, uint32_t in_block,
                                                 uint64_t row, uint64_t nRows, uint32_t nCols,
                                                 Layout layout, bool pad_start,
                                                 bool with_len, uint64_t total_bits)
{
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        uint32_t lo = 0, hi = 0;
        if ((uint32_t)k < in_block)
        {
            // CLAMP, not cosmetic: ptxas speculates this load ABOVE the
            // `k < in_block` predicate, so the guard does not keep the ADDRESS in
            // bounds when a partial final block names columns past nCols. Faults on
            // a tightly sized buffer (caught by LeafHashIsLayoutInvariant); the same
            // pattern in b3_gather_block survives only because the prover's trace
            // allocation is large enough to absorb the over-read.
            const uint32_t col_raw = base_col + (uint32_t)k;
            const uint32_t col = (col_raw < nCols) ? col_raw : 0u;
            const uint64_t off = getBufferOffset(row, col, nRows, nCols, layout);
            const uint64_t v = sha256core::to_canonical(in[off]);
            lo = sha256core::bswap32((uint32_t)v);
            hi = sha256core::bswap32((uint32_t)(v >> 32));
        }
        else if ((uint32_t)k == in_block && pad_start)
        {
            lo = 0x80000000u;
        }
        W[2 * k]     = lo;
        W[2 * k + 1] = hi;
    }
    if (with_len)
    {
        W[14] = (uint32_t)(total_bits >> 32);
        W[15] = (uint32_t)total_bits;
    }
}

// Leaf hash of one trace row: literal FIPS SHA-256, read through `layout`.
__device__ void sha_hash_row(const uint64_t *in, uint32_t nCols, uint64_t row, uint64_t nRows,
                             Layout layout, uint64_t out[4])
{
    uint32_t h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = sha256core::sha_iv(i);

    const uint64_t total_bits = (uint64_t)nCols * 64ull;
    const uint32_t full = nCols / 8u;
    const uint32_t tail = nCols % 8u;
    const bool tail_fits = (tail <= 6u);

    uint32_t W[16];
    for (uint32_t b = 0; b < full; ++b)
    {
        sha_gather_block(W, in, b * 8u, 8u, row, nRows, nCols, layout, false, false, 0ull);
        sha256core::compress_in_place(h, W);
    }
    // tail == 0 has nothing to gather (base_col would be nCols), so build the
    // padding block from the core: no out-of-range column can even be named.
    if (tail == 0u)
        sha256core::fill_block_le64(W, nullptr, 0u, true, tail_fits, total_bits);
    else
        sha_gather_block(W, in, full * 8u, tail, row, nRows, nCols, layout, true, tail_fits, total_bits);
    sha256core::compress_in_place(h, W);
    if (!tail_fits)
    {
        sha256core::fill_block_le64(W, nullptr, 0u, false, true, total_bits);
        sha256core::compress_in_place(h, W);
    }
    sha256core::pack4(h, out);
}

__global__ void sha_linearHashKernel(uint64_t *__restrict__ out, const uint64_t *__restrict__ in,
                                     uint32_t num_cols, uint32_t num_rows, Layout layout)
{
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t dig[4];
    sha_hash_row(in, num_cols, row, num_rows, layout, dig);
    uint64_t *o = out + row * 4ull;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = dig[i];
}

// Internal node: children are contiguous, so node_hash applies directly. At arity
// 2 skipping the padding block is the difference between 1 and 2 compressions.
__global__ void sha_merkleNodeKernel(uint64_t *cursor, uint64_t nextN, uint64_t nextIndex,
                                     uint64_t pending, uint32_t arity)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nextN) return;
    uint64_t base = nextIndex + tid * (uint64_t)arity * 4ull;
    uint64_t dig[4];
    sha256core::node_hash(&cursor[base], arity * 4u, dig);
    uint64_t *o = &cursor[nextIndex + (pending + tid) * 4ull];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = dig[i];
}

__global__ void sha_grindingKernel(uint64_t *nonce, uint64_t *__restrict__ nonceBlock,
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
    uint64_t out4[4];
    for (uint64_t k = 0; k < hashes_per_thread; ++k)
    {
        uint64_t idx_k = idx + k;
        in8[0] = input[0]; in8[1] = input[1]; in8[2] = input[2]; in8[3] = idx_k;
        sha256core::grind_hash(in8, out4);
        if (out4[0] < level) { locId = idx_k; break; }
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
void Sha256GoldilocksGPU::linearHash(uint64_t *d_hash_output, uint64_t *d_trace,
                                     uint64_t num_cols, uint64_t num_rows,
                                     Layout layout, cudaStream_t stream)
{
    if (num_rows == 0) return;
    uint32_t tpb = (num_rows < TPB_SHA) ? (uint32_t)num_rows : TPB_SHA;
    uint32_t blks = (uint32_t)((num_rows + TPB_SHA - 1) / TPB_SHA);
    sha_linearHashKernel<<<blks, tpb, 0, stream>>>(
        d_hash_output, d_trace, (uint32_t)num_cols, (uint32_t)num_rows, layout);
    CHECKCUDAERR(cudaGetLastError());
}

void Sha256GoldilocksGPU::merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
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

        uint32_t tpb = (nextN < TPB_SHA) ? (uint32_t)nextN : TPB_SHA;
        uint32_t blks = (nextN < TPB_SHA) ? 1u : (uint32_t)(nextN / TPB_SHA + 1);
        sha_merkleNodeKernel<<<blks, tpb, 0, stream>>>(d_tree, nextN, nextIndex,
                                                       pending + extraZeros, arity);

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    CHECKCUDAERR(cudaGetLastError());
}

void Sha256GoldilocksGPU::merkletreeReduce(uint64_t *d_root, uint64_t *d_input,
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

        uint32_t tpb = (nextN < TPB_SHA) ? (uint32_t)nextN : TPB_SHA;
        uint32_t blks = (nextN < TPB_SHA) ? 1u : (uint32_t)(nextN / TPB_SHA + 1);
        sha_merkleNodeKernel<<<blks, tpb, 0, stream>>>(d_tree, nextN, nextIndex,
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

void Sha256GoldilocksGPU::grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock,
                                   const uint64_t *d_in, const uint32_t n_bits,
                                   cudaStream_t stream)
{
    uint64_t log_launch_iters = 7;
    uint64_t launch_iters = 1ULL << log_launch_iters;
    uint64_t log_N = SHA256_GRIND_BITS;
    uint64_t N = 1ULL << log_N;
    uint64_t security = 128;

    double eps                 = ldexp(1.0, -int(n_bits));
    double totalHashesRequired = -double(security) * log(2.0) / log1p(-eps);
    uint64_t log_totalHashesRequired = (uint64_t)ceil(log2(totalHashesRequired));
    uint64_t log_hashesPerThread = (log_totalHashesRequired > log_launch_iters + log_N)
                                   ? log_totalHashesRequired - log_launch_iters - log_N : 0;
    uint64_t hashesPerThread = 1ULL << log_hashesPerThread;

    dim3 blockSize(SHA256_GRIND_BLOCKS);
    dim3 gridSize(SHA256_GRIND_GRID);
    size_t shared_mem_size = blockSize.x * sizeof(uint64_t);

    uint64_t nonces_offset = 0;
    uint64_t nonces_per_iteration = N * hashesPerThread;
    for (uint64_t i = 0; i < launch_iters; ++i)
    {
        sha_grindingKernel<<<gridSize, blockSize, shared_mem_size, stream>>>(
            d_nonce, d_nonceBlock, (uint64_t *)d_in, n_bits, hashesPerThread, nonces_offset);
        nonces_offset += nonces_per_iteration;
    }
}
