/*
 * NTT GPU Implementation for Goldilocks Field
 *
 * This file implements the Number Theoretic Transform (NTT) on GPU for the
 * Goldilocks prime field (p = 2^64 - 2^32 + 1).
 *
 * Two backends live here: the in-house ColMajor engine (ldeColMajor / inttColMajor /
 * computeQColMajor -- the production path, resolveLayout is always Layout::ColMajor) and the legacy
 * TILED backend (ldeTiled / inttTiled / computeQTiled, Layout::ColMajorTiled), kept as a directly
 * callable reference for tests/benches but no longer selected by any dispatch.
 *
 * === Data Layouts (tiled path) ===
 *
 * Tiled data is organized in tiles of TILE_HEIGHT=256 rows x TILE_WIDTH=4 cols
 * (defined in goldilocks_trace_layout.cuh). Three orderings within tiles are used:
 *
 * - Column-major tiles (getBufferOffset): Elements stored column-by-column
 *   within each tile. This is the prover's storage format — consecutive rows
 *   of the same column are contiguous for coalesced evaluation access.
 *
 * - Row-major tiles (getBufferOffsetRowMajor): Elements stored row-by-row
 *   within each tile. NTT butterfly kernels operate on this layout because
 *   each thread processes one row across all columns in a tile.
 *
 * - Packed row-major tiles (getBufferOffsetRowMajorPacked): Row-major within
 *   tiles, but for LDE only. When extending domain N to N*B (blowup factor B),
 *   each tile has only 256/B rows of actual data, packed at the start. Used on the
 *   LDE to obtain result of INTT on the extended domain in bit reversed order.
 *
 * === NTT Pipelines ===
 *
 * NTT/INTT (standalone):
 *   columnMajorToRowMajor -> bitReversal + DIT butterfly loop -> rowMajorToColumnMajor
 *
 * LDE (INTT small domain + NTT extended domain):
 *   columnMajorToPackedRowMajor -> DIF butterfly (INTT + zero-pad) -> DIT butterfly -> rowMajorToColumnMajor (no bit reverals)
 *
 * computeQ (INTT + coset shift + NTT):
 *   columnMajorToRowMajor -> bitReversal + DIT butterfly loop (INTT) -> applyCosetShift -> bitReversal + DIT butterfly loop (NTT) -> rowMajorToColumnMajor
 *
 * === Butterfly Strategies ===
 *
 * - DIT (Decimation-In-Time): Bit-reversed input, natural output.
 * - DIF (Decimation-In-Frequency): Natural input, bit-reversed output. 
 *
 * Each butterfly kernel launch processes up to BATCH_HEIGHT_LOG2=8 stages in shared memory.
 */

#include "ntt_goldilocks.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "goldilocks_tooling.cuh"
#include "poseidon2_goldilocks.cuh"
#include "ntt_goldilocks.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "omp.h"
#include "poseidon2_goldilocks.hpp"
#include <atomic>
#include <mutex>

#include "timer_gl.hpp"

#define COSET_SHIFT 7

// --- Forward declarations ---

__global__ void nttDitButterflyKernel(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void nttDifButterflyPackedKernel( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor);
__global__ void evalTwiddleSmallKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void evalTwiddleFirstKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void evalTwiddleSecondKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
void evalTwiddleFactors(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size, cudaStream_t stream);
__global__ void evalCosetShiftSmallKernel(gl64_t *r, uint32_t log_domain_size);
__global__ void evalCosetShiftFirstKernel(gl64_t *r, uint32_t log_domain_size);
__global__ void evalCosetShiftSecondKernel(gl64_t *r, uint32_t log_domain_size);
void evalCosetShifts(gl64_t *r, uint32_t log_domain_size, cudaStream_t stream);
void nttDit( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
void nttDifLde( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
void nttDitLde( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);

// =============================================================================
// Q-polynomial kernel
// =============================================================================

// Computes coset shift powers S[p] = shiftIn^p, then multiplies each q-polynomial
// segment by S[p] using Goldilocks extension field (Goldilocks3GPU::mul), writes
// to cmQ in row-major tile layout, and zero-fills extended-domain rows.
// Launch: <<<ceil(N/256), 256>>>
__global__ void applyCosetShiftKernel(gl64_t *d_cmQ, gl64_t *d_q, gl64_t *d_S, Goldilocks::Element shiftIn, uint64_t N, uint64_t NExtended, uint64_t extendBits, uint64_t qDeg, uint64_t qDim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;

    // Compute S values inline per-thread in registers instead of via global memory
    gl64_t s_val = gl64_t(uint64_t(1));
    for (uint64_t p = 0; p < qDeg; p++)
    {
        Goldilocks3GPU::Element src;
        src[0] = d_q[getBufferOffsetRowMajor(row + p * N, 0, NExtended, qDim)];
        src[1] = d_q[getBufferOffsetRowMajor(row + p * N, 1, NExtended, qDim)];
        src[2] = d_q[getBufferOffsetRowMajor(row + p * N, 2, NExtended, qDim)];

        Goldilocks3GPU::Element dst;

        Goldilocks3GPU::mul((Goldilocks3GPU::Element &)dst,
                            (Goldilocks3GPU::Element &)src,
                            s_val);
        d_cmQ[getBufferOffsetRowMajor(row, p * qDim, NExtended, qDeg * qDim)] = dst[0];
        d_cmQ[getBufferOffsetRowMajor(row, p * qDim + 1, NExtended, qDeg * qDim)] = dst[1];
        d_cmQ[getBufferOffsetRowMajor(row, p * qDim + 2, NExtended, qDeg * qDim)] = dst[2];
        for (uint64_t j = 1; j < (1 << extendBits); j++) {
            d_cmQ[getBufferOffsetRowMajor(row + j * N, p * qDim, NExtended, qDeg * qDim)] = gl64_t(uint64_t(0));
            d_cmQ[getBufferOffsetRowMajor(row + j * N, p * qDim + 1, NExtended, qDeg * qDim)] = gl64_t(uint64_t(0));
            d_cmQ[getBufferOffsetRowMajor(row + j * N, p * qDim + 2, NExtended, qDeg * qDim)] = gl64_t(uint64_t(0));
        }
        s_val = gl64_t(shiftIn.fe) * s_val;
    }
}

// =============================================================================
// Layout conversion kernels
// =============================================================================

// Row-major tiles -> storage layout `layout`, in-place via shared memory.
// Restores the prover's column-major storage (ColMajor flat or ColMajorTiled) after NTT operations.
// Launch: <<<(ceil(nRows/256), ceil(nCols/4)), (256, 4), 256*4*sizeof(gl64_t)>>>
__global__ void rowMajorToColumnMajorKernel(gl64_t * data, uint64_t nRows, uint64_t nCols, Layout layout)
{
    extern __shared__ gl64_t shared[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nRows || col >= nCols)
        return;

    uint64_t offset_src = getBufferOffsetRowMajor(row, col, nRows, nCols);
    shared[threadIdx.y * blockDim.x + threadIdx.x] = data[offset_src];
    __syncthreads();
    uint64_t offset_dst = getBufferOffset(row, col, nRows, nCols, layout);
    data[offset_dst] = shared[threadIdx.y * blockDim.x + threadIdx.x];
}

// Storage layout `layout` -> row-major tiles, in-place via shared memory.
// Prepares data for NTT butterfly kernels which operate on row-major tiles.
// Launch: <<<(ceil(nRows/256), ceil(nCols/4)), (256, 4), 256*4*sizeof(gl64_t)>>>
__global__ void columnMajorToRowMajorKernel(gl64_t *data, uint64_t nRows, uint64_t nCols, Layout layout)
{
    extern __shared__ gl64_t shared[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nRows || col >= nCols)
        return;

    uint64_t offset_src = getBufferOffset(row, col, nRows, nCols, layout);
    shared[threadIdx.y * blockDim.x + threadIdx.x] = data[offset_src];
    __syncthreads();
    uint64_t offset_dst = getBufferOffsetRowMajor(row, col, nRows, nCols);
    data[offset_dst] = shared[threadIdx.y * blockDim.x + threadIdx.x];
}

// Storage layout `layout` -> packed row-major tiles, disjoint src/dst.
// Used by LDE to pack small-domain data into extended-domain tiles:
// only the first TILE_HEIGHT/blowup rows per tile get data.
// Launch: <<<(ceil(nSrc/256), ceil(nCols/4)), (256, 4), 256*4*sizeof(gl64_t)>>>
__global__ void columnMajorToPackedRowMajorKernel(gl64_t *src, uint64_t n_bits_src, gl64_t *dst, uint64_t n_bits_dst, uint64_t nCols, Layout layout)
{
    extern __shared__ gl64_t shared[];
    uint32_t n_src = 1 << n_bits_src;
    uint32_t n_dst = 1 << n_bits_dst;

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t blowupFactor = 1 << (n_bits_dst - n_bits_src);

    if(row >= n_src || col >= nCols)
        return;

    uint64_t offset_src = getBufferOffset(row, col, n_src, nCols, layout);
    shared[threadIdx.y * blockDim.x + threadIdx.x] = src[offset_src];
    __syncthreads();
    uint64_t offset_dst = getBufferOffsetRowMajorPacked(row, col, n_dst, nCols, blowupFactor);
    dst[offset_dst] = shared[threadIdx.y * blockDim.x + threadIdx.x];
}

// =============================================================================
// Twiddle factor and coset shift initialization kernels
// =============================================================================

// Note: Overall, implementtions are not optimal but is not critical part, done only once.

// First TWIDDLE_SPLIT entries are computed sequentially (single thread),
// then the rest are filled in parallel using those entries as seeds.
#define TWIDDLE_SPLIT_LOG2 12
#define TWIDDLE_SPLIT (1 << TWIDDLE_SPLIT_LOG2)

// Sequential twiddle factor computation for small domains (log <= TWIDDLE_SPLIT_LOG2+1).
// factor[i] = omega^i where omega is the primitive 2^logDomain-th root of unity.
// Launch: <<<1, 1>>>
__global__ void evalTwiddleSmallKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
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

// First TWIDDLE_SPLIT+1 twiddle entries, computed sequentially (single thread).
// Launch: <<<1, 1>>>
__global__ void evalTwiddleFirstKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    gl64_t omega = gl64_t(omegas[log_domain_size]);
    gl64_t omega_inv = gl64_t(omegas_inv[log_domain_size]);

    fwd_twiddles[0] = gl64_t(uint64_t(1));
    inv_twiddles[0] = gl64_t(uint64_t(1));

    for (uint32_t i = 1; i <= TWIDDLE_SPLIT; i++)
    {
        fwd_twiddles[i] = fwd_twiddles[i - 1] * omega;
        inv_twiddles[i] = inv_twiddles[i - 1] * omega_inv;
    }
}

// Remaining twiddle entries filled in parallel.
// Thread idx computes: factor[i*TWIDDLE_SPLIT + idx] = factor[(i-1)*TWIDDLE_SPLIT + idx] * factor[TWIDDLE_SPLIT]
// Launch: <<<TWIDDLE_SPLIT, 1>>>
__global__ void evalTwiddleSecondKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = 1; i < 1 << (log_domain_size - TWIDDLE_SPLIT_LOG2 - 1); i++)
    {
        fwd_twiddles[i * TWIDDLE_SPLIT + idx] = fwd_twiddles[(i - 1) * TWIDDLE_SPLIT + idx] * fwd_twiddles[TWIDDLE_SPLIT];
        inv_twiddles[i * TWIDDLE_SPLIT + idx] = inv_twiddles[(i - 1) * TWIDDLE_SPLIT + idx] * inv_twiddles[TWIDDLE_SPLIT];
    }
}

// Dispatches twiddle factor computation: single-kernel for small domains, two-step for large.
void evalTwiddleFactors(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size, cudaStream_t stream)
{
    if (log_domain_size <= TWIDDLE_SPLIT_LOG2 + 1)
    {
        evalTwiddleSmallKernel<<<1, 1, 0, stream>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
    else
    {
        evalTwiddleFirstKernel<<<1, 1, 0, stream>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
        evalTwiddleSecondKernel<<<TWIDDLE_SPLIT, 1, 0, stream>>>(fwd_twiddles, inv_twiddles, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
}

// Sequential coset shift computation for small domains (log <= TWIDDLE_SPLIT_LOG2).
// r[i] = COSET_SHIFT^i where COSET_SHIFT=7 is the Goldilocks multiplicative generator.
// Launch: <<<1, 1>>>
__global__ void evalCosetShiftSmallKernel(gl64_t *r, uint32_t log_domain_size)
{
    r[0] = gl64_t(uint64_t(1));
    for (uint32_t i = 1; i < 1 << log_domain_size; i++)
    {
        r[i] = r[i - 1] * gl64_t(COSET_SHIFT);
    }
}

// First TWIDDLE_SPLIT+1 coset shift entries, sequential.
// Launch: <<<1, 1>>>
__global__ void evalCosetShiftFirstKernel(gl64_t *r, uint32_t log_domain_size)
{
    r[0] = gl64_t(uint64_t(1));
    for (uint32_t i = 1; i <= TWIDDLE_SPLIT; i++)
    {
        r[i] = r[i - 1] * gl64_t(COSET_SHIFT);
    }
}

// Remaining coset shift entries filled in parallel (same pattern as twiddles).
// Launch: <<<TWIDDLE_SPLIT, 1>>>
__global__ void evalCosetShiftSecondKernel(gl64_t *r, uint32_t log_domain_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = 1; i < 1 << (log_domain_size - TWIDDLE_SPLIT_LOG2); i++)
    {
        r[i * TWIDDLE_SPLIT + idx] = r[(i - 1) * TWIDDLE_SPLIT + idx] * r[TWIDDLE_SPLIT];
    }
}

// Dispatches coset shift computation: single-kernel for small domains, two-step for large.
void evalCosetShifts(gl64_t *r, uint32_t log_domain_size, cudaStream_t stream)
{
    if (log_domain_size <= TWIDDLE_SPLIT_LOG2)
    {
        evalCosetShiftSmallKernel<<<1, 1, 0, stream>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
    else
    {
        evalCosetShiftFirstKernel<<<1, 1, 0, stream>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
        evalCosetShiftSecondKernel<<<TWIDDLE_SPLIT, 1, 0, stream>>>(r, log_domain_size);
        CHECKCUDAERR(cudaGetLastError());
    }
}

// =============================================================================
// Bit-reversal kernel (tiled, production)
// =============================================================================

// Bit-reversal permutation on row-major tiled data.
// Each block handles BATCH_WIDTH=4 columns of one tile column-group.
// Iterates over rows with stride gridDim.x, swaps elements at (i, bitrev(i)).
// Launch: <<<(min(domainSize, 4096), ceil(nCols/BATCH_WIDTH)), BATCH_WIDTH>>>
__global__ void bitReversalKernel(gl64_t *data, uint32_t log2_domain_size_in, uint64_t domain_size_out, uint32_t nCols)
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

// =============================================================================
// NTT butterfly kernels (tiled, production)
// =============================================================================

// DIT radix-2 butterfly kernel. Processes up to BATCH_HEIGHT_LOG2=8 stages per launch.
// Loads a 256x4 tile into shared memory, performs butterflies with twiddle lookup,
// writes back. On final-stage launch with inverse=true: multiplies by N^-1 and
// optionally by coset shift r[row].
// Launch: <<<(domainSize/256, ceil(nCols/4)), 256>>>
__global__ void nttDitButterflyKernel(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(log_domain_size_in - base_step, BATCH_HEIGHT_LOG2);
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // Reorganize row indices for batched processing
    uint32_t groupSize = 1 << base_step;
    uint32_t nGroups = domain_size_in / groupSize;
    uint32_t low_bits = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    // Define column block
    uint32_t block_stride = domain_size_out * BATCH_WIDTH;
    gl64_t *data_block = data + blockIdx.y*block_stride;
    uint32_t col_base = blockIdx.y * BATCH_WIDTH;
    uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? nCols - col_base : BATCH_WIDTH;

    // Load tile from global memory
    for(int i=0; i<ncols_block; i++){
        tile[BATCH_HEIGHT*i+threadIdx.x] = data_block[row*ncols_block+i];
    }
    __syncthreads();

    // Butterfly stages
    uint32_t remaining_high_bits = log_domain_size_in - (base_step+1);
    uint32_t high_mask = (1 << remaining_high_bits) - 1;

    for(int loc_step=0; loc_step<n_loc_steps; loc_step++){
        uint32_t i = threadIdx.x;
        if (threadIdx.x < BATCH_HEIGHT_DIV2){
            uint32_t group_size = 1 << loc_step;
            uint32_t group = i >> loc_step;
            uint32_t group_pos = i & (group_size - 1);
            uint32_t index1 = (group << (loc_step + 1)) + group_pos;
            uint32_t index2 = index1 + group_size;
            gl64_t factor;
            {
                uint32_t gs = base_step + loc_step;
                uint32_t ggs = 1 << gs;
                uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                uint32_t gbi = (((bbi & high_mask)<< base_step) + (bbi >> remaining_high_bits));
                uint32_t ggp = gbi & (ggs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];
            }
            if (ncols_block == BATCH_WIDTH) {
                #pragma unroll
                for(int j=0; j<BATCH_WIDTH; j++){
                    gl64_t odd_sub = tile[ j*BATCH_HEIGHT + index2] * factor;
                    tile[j*BATCH_HEIGHT +index2] = tile[j*BATCH_HEIGHT + index1] - odd_sub;
                    tile[j*BATCH_HEIGHT +index1] = tile[j*BATCH_HEIGHT + index1] + odd_sub;
                }
            } else {
                for(int j=0; j<ncols_block; j++){
                    gl64_t odd_sub = tile[ j*BATCH_HEIGHT + index2] * factor;
                    tile[j*BATCH_HEIGHT +index2] = tile[j*BATCH_HEIGHT + index1] - odd_sub;
                    tile[j*BATCH_HEIGHT +index1] = tile[j*BATCH_HEIGHT + index1] + odd_sub;
                }
            }
        }
        __syncthreads();
    }

    // Store tile back, with INTT scaling on final stage
    // Store tile back, with INTT scaling on final stage
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

// DIF radix-2 butterfly kernel with packed storage for LDE.
// Processes up to BATCH_HEIGHT_LOG2=8 stages per launch in reverse order.
// Reads from packed tile positions. On final-stage launch with
// inverse=true: applies N^-1 scaling, coset shift via bit-reversed row index,
// writes to blowup-strided positions (row*blowupFactor), and zero-fills gaps.
// Launch: <<<(domainSize/256, ceil(nCols/4)), 256>>>
__global__ void nttDifButterflyPackedKernel( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor)
{
    __shared__ gl64_t tile[BATCH_HEIGHT * BATCH_WIDTH];

    uint32_t n_loc_steps = min(base_step+1, BATCH_HEIGHT_LOG2);
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // Reorganize row indices for batched processing (same packing as DIT)
    uint32_t groupSize = 1 << (base_step + 1 - n_loc_steps);
    uint32_t nGroups   = domain_size_in / groupSize;
    uint32_t low_bits  = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    // Define column block
    uint32_t block_stride = domain_size_out * BATCH_WIDTH;
    gl64_t *data_block = data + blockIdx.y * block_stride;
    uint32_t col_base = blockIdx.y * BATCH_WIDTH;
    uint32_t ncols_block = (nCols - col_base) < BATCH_WIDTH ? (nCols - col_base) : BATCH_WIDTH;

    // Compute packed tile position
    uint64_t batch_height_blown = BATCH_HEIGHT / blowupFactor;
    uint64_t blockX = (row / batch_height_blown);
    uint64_t row_block = row % batch_height_blown;
    uint32_t row_comp = blockX * BATCH_HEIGHT + row_block;

    // Load tile from packed positions
    for (int i = 0; i < ncols_block; i++){
        tile[BATCH_HEIGHT * i + threadIdx.x] = data_block[row_comp * ncols_block + i];
    }
    __syncthreads();

    // DIF butterfly stages (reverse order)
    uint32_t remaining_high_bits = log_domain_size_in -1 - (base_step + 1 - n_loc_steps);
    uint32_t high_mask = (1u << remaining_high_bits) - 1u;
    for (int loc_step = n_loc_steps-1; loc_step >= 0; loc_step--){
        uint32_t i = threadIdx.x;
        if (i < BATCH_HEIGHT_DIV2) {
            uint32_t group_size = 1u << loc_step;
            uint32_t group = i >> loc_step;
            uint32_t group_pos = i & (group_size - 1u);
            uint32_t index1 = (group << (loc_step + 1)) + group_pos;
            uint32_t index2 = index1 + group_size;

            gl64_t factor;
            {
                uint32_t gs = base_step -(n_loc_steps -1 - loc_step);
                uint32_t ggs = 1 << gs;
                uint32_t bbi = blockIdx.x* BATCH_HEIGHT_DIV2 + i;
                uint32_t gbi = (((bbi & high_mask)<< (base_step + 1 - n_loc_steps)) + (bbi >> remaining_high_bits));
                uint32_t ggp = gbi & (ggs - 1);
                factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];
            }

            // DIF: t1 = a + b, t2 = (a - b) * W^k
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

    // Store: on final stage, apply INTT scaling + coset shift + zero-fill extended rows
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

// =============================================================================
// Host NTT 
// =============================================================================

// DIT NTT driver (tiled): bit-reversal + butterfly loop in BATCH_HEIGHT_LOG2-stage chunks.
// Used by NTT(), INTT(), and computeQ().
void nttDit( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{
    assert(log_domain_size_in >= BATCH_HEIGHT_LOG2 && "Domain size must be >= BATCH_HEIGHT for tiled NTT");
    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    dim3 blockDim;
    dim3 gridDim;
    blockDim = dim3(BATCH_WIDTH);
    gridDim = dim3(min(domain_size_in, (uint32_t)4096), (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH);
    bitReversalKernel<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size_in, domain_size_out, nCols);
    CHECKCUDAERR(cudaGetLastError());

    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call initConstants()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];

    for(uint32_t step = 0; step < log_domain_size_in; step+=BATCH_HEIGHT_LOG2){
        dim3 blocks = dim3(domain_size_in / BATCH_HEIGHT, (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH, 1);
        dim3 threads = dim3(BATCH_HEIGHT,1,1);
        nttDitButterflyKernel<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, nCols, step, true, inverse, extend, maxLogDomainSize);
        CHECKCUDAERR(cudaGetLastError());
    }
}

// DIF INTT driver for LDE phase 1 (packed storage, no separate bit-reversal).
// DIF naturally produces bit-reversed output needed by phase 2.
void nttDifLde( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{
    assert(log_domain_size_in >= BATCH_HEIGHT_LOG2 && "Domain size must be >= BATCH_HEIGHT for tiled NTT");
    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call initConstants()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];

    for(int step = log_domain_size_in-1; step >= 0; step-=BATCH_HEIGHT_LOG2){
        dim3 blocks = dim3(domain_size_in / BATCH_HEIGHT, (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH, 1);
        dim3 threads = dim3(BATCH_HEIGHT,1,1);
        nttDifButterflyPackedKernel<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, nCols, step, true, inverse, extend, maxLogDomainSize, (1 << (log_domain_size_out - log_domain_size_in)));
        CHECKCUDAERR(cudaGetLastError());
    }
}

// DIT NTT driver for LDE phase 2 (no separate bit-reversal — input already
// bit-reversed from DIF phase 1 output).
void nttDitLde( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size_in, uint32_t log_domain_size_out, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{
    assert(log_domain_size_in >= BATCH_HEIGHT_LOG2 && "Domain size must be >= BATCH_HEIGHT for tiled NTT");
    uint32_t domain_size_in = 1 << log_domain_size_in;
    uint32_t domain_size_out = 1 << log_domain_size_out;

    int device_id;
    cudaGetDevice(&device_id);
    if (d_fwd_twiddle_factors[device_id] == nullptr || d_inv_twiddle_factors[device_id] == nullptr)
    {
        fprintf(stderr, "[NTT] ERROR: Twiddle factors not initialized for device %d. Did you call initConstants()?\n", device_id);
        abort();
    }

    gl64_t *d_twiddles = inverse ? d_inv_twiddle_factors[device_id] : d_fwd_twiddle_factors[device_id];
    gl64_t *d_r = d_r_[device_id];

    for(uint32_t step = 0; step < log_domain_size_in; step+=BATCH_HEIGHT_LOG2){
        dim3 blocks = dim3(domain_size_in / BATCH_HEIGHT, (nCols + BATCH_WIDTH - 1) / BATCH_WIDTH, 1);
        dim3 threads = dim3(BATCH_HEIGHT,1,1);
        nttDitButterflyKernel<<<blocks, threads, 0, stream>>>(data, d_twiddles, d_r, domain_size_in, log_domain_size_in, domain_size_out, nCols, step, true, inverse, extend, maxLogDomainSize);
        CHECKCUDAERR(cudaGetLastError());
    }
}

// =============================================================================
// ColMajor NTT engine
//
// Mixed-radix NTT/LDE on column-major storage. The butterflies are the classic
// register-resident mixed-radix design: the working set lives in registers,
// stages 1-5 exchange partners with warp shuffles, wider stages go through
// shared memory, and per-thread positioning roots come from windowed tables,
// derived incrementally. Kernels are column-batched (gridDim.y) and the driver
// chunks columns to L2 so a transform's steps stay cache-resident between
// launches.
// =============================================================================

#define NTT_MAX_LG   28                    // max supported log2 domain (mirrors table depth)
#define NTT_LG_WIN   7                     // log2 of a partial-twiddle window
#define NTT_WIN_SIZE 128                   // 1 << NTT_LG_WIN
#define NTT_WIN_NUM  4                     // ceil(NTT_MAX_LG / NTT_LG_WIN)
#define NTT_PO_WINS  (NTT_MAX_LG - 15)    // z-step-root windows (one per sub-transform size)
#define NTT_RADIX_BLOB (512 + 256 + 128 + 64 + 32)
#define NTT_MAX_DEVS 64

__device__ __forceinline__ uint32_t nttBitRev(uint32_t i, uint32_t nbits)
{
    return __brev(i) >> (32 - nbits);
}

// Positioning roots for a butterfly pair: product of window-table entries so a
// single table of NTT_WIN_NUM x NTT_WIN_SIZE covers exponents < 2^NTT_MAX_LG.
__device__ __forceinline__ void nttIntermediateRoots(gl64_t &root0, gl64_t &root1,
                                                     uint32_t idx0, uint32_t idx1,
                                                     const gl64_t (*roots)[NTT_WIN_SIZE])
{
    int win = (NTT_WIN_NUM - 1) * NTT_LG_WIN;
    int off = (NTT_WIN_NUM - 1);
    uint32_t idxo = idx0 | idx1;
    uint32_t mask = ((uint32_t)1 << win) - 1;

    root0 = roots[off][idx0 >> win];
    root1 = roots[off][idx1 >> win];
    #pragma unroll 1
    while (off-- && (idxo & mask)) {
        win -= NTT_LG_WIN;
        mask >>= NTT_LG_WIN;
        root0 *= roots[off][(idx0 >> win) % NTT_WIN_SIZE];
        root1 *= roots[off][(idx1 >> win) % NTT_WIN_SIZE];
    }
}

// Coalesced z_count-element load/store: consecutive lanes read consecutive
// addresses, then a warp-level shared transpose hands each lane its strided set.
template<int z_count>
__device__ __forceinline__ void nttCoalescedLoad(gl64_t r[z_count], const gl64_t *inout,
                                                 uint32_t idx, uint32_t stage)
{
    const uint32_t x = threadIdx.x & (z_count - 1);
    idx &= ~((uint32_t)(z_count - 1) << stage);
    idx += x;
    #pragma unroll
    for (int z = 0; z < z_count; z++, idx += (uint32_t)1 << stage)
        r[z] = inout[idx];
}

template<int z_count>
__device__ __forceinline__ void nttTranspose(gl64_t r[z_count])
{
    extern __shared__ int nttTransposeShm[];
    gl64_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(nttTransposeShm);

    const uint32_t x = threadIdx.x & (z_count - 1);
    const uint32_t y = threadIdx.x & ~(z_count - 1);

    #pragma unroll
    for (int z = 0; z < z_count; z++)
        xchg[y + z][x] = r[z];
    __syncwarp();
    #pragma unroll
    for (int z = 0; z < z_count; z++)
        r[z] = xchg[y + x][z];
}

template<int z_count>
__device__ __forceinline__ void nttCoalescedStore(gl64_t *inout, uint32_t idx,
                                                  const gl64_t r[z_count], uint32_t stage)
{
    const uint32_t x = threadIdx.x & (z_count - 1);
    idx &= ~((uint32_t)(z_count - 1) << stage);
    idx += x;
    #pragma unroll
    for (int z = 0; z < z_count; z++, idx += (uint32_t)1 << stage)
        inout[idx] = r[z];
}

// DIT (decimation-in-time) step: bit-reversed input order, natural output -- the
// forward NTT of the LDE. gridDim.y selects the column (stride col_stride elements).
template<int z_count, bool coalesced = false>
__launch_bounds__(768, 1) __global__
void nttDitColMajorKernel(const uint32_t radix, const uint32_t lg_domain_size,
                          const uint32_t stage, const uint32_t iterations,
                          gl64_t *d_base, const size_t col_stride,
                          const gl64_t (*d_partialRoots)[NTT_WIN_SIZE],
                          const gl64_t (*d_zStepRoots)[1024],
                          const gl64_t *d_radix6, const gl64_t *d_radixX,
                          bool is_intt, const uint64_t domain_inv)
{
    gl64_t *d_inout = d_base + (size_t)blockIdx.y * col_stride;
    extern __shared__ int nttShm[];
    gl64_t *shared_exchange = reinterpret_cast<gl64_t *>(nttShm);

    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t diff_mask = (1 << (iterations - 1)) - 1;
    const uint32_t inp_mask = ((uint32_t)1 << stage) - 1;
    const uint32_t out_mask = ((uint32_t)1 << (stage + iterations - 1)) - 1;

    const uint32_t tiz = (tid & ~diff_mask) * z_count + (tid & diff_mask);
    const uint32_t thread_ntt_pos = (tiz >> (iterations - 1)) & inp_mask;

    uint32_t idx0 = (tiz & ~out_mask) | ((tiz << stage) & out_mask);
    idx0 = idx0 * 2 + thread_ntt_pos;
    uint32_t idx1 = idx0 + ((uint32_t)1 << stage);

    gl64_t r[2][z_count];

    if (coalesced) {
        nttCoalescedLoad<z_count>(r[0], d_inout, idx0, stage + 1);
        nttCoalescedLoad<z_count>(r[1], d_inout, idx1, stage + 1);
        nttTranspose<z_count>(r[0]);
        __syncwarp();
        nttTranspose<z_count>(r[1]);
    } else {
        uint32_t z_shift = inp_mask == 0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = d_inout[idx0 + (z << z_shift)];
            r[1][z] = d_inout[idx1 + (z << z_shift)];
        }
    }

    if (stage != 0) {
        uint32_t thread_ntt_idx = (tiz & diff_mask) * 2;
        uint32_t nbits = NTT_MAX_LG - stage;
        uint32_t idx0r = nttBitRev(thread_ntt_idx, nbits);
        uint32_t root_idx0 = idx0r * thread_ntt_pos;
        uint32_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        gl64_t first_root, second_root;
        nttIntermediateRoots(first_root, second_root, root_idx0, root_idx1, d_partialRoots);
        r[0][0] = r[0][0] * first_root;
        r[1][0] = r[1][0] * second_root;

        if (z_count > 1) {
            uint32_t off = nbits >= 10 ? (nbits - 10) : 0;
            uint32_t scale = nbits >= 10 ? 0 : (10 - nbits);

            thread_ntt_idx <<= scale;
            gl64_t first_root_z = d_zStepRoots[off][thread_ntt_idx];
            gl64_t second_root_z = d_zStepRoots[off][thread_ntt_idx + (1 << scale)];

            #pragma unroll
            for (int z = 1; z < z_count; z++) {
                first_root *= first_root_z;
                second_root *= second_root_z;
                r[0][z] = r[0][z] * first_root;
                r[1][z] = r[1][z] * second_root;
            }
        }
    }

    #pragma unroll
    for (int z = 0; z < z_count; z++) {
        gl64_t t = r[1][z];
        r[1][z] = r[0][z] - t;
        r[0][z] = r[0][z] + t;
    }

    #pragma unroll 1
    for (uint32_t s = 1; s < min(iterations, 6u); s++) {
        uint32_t laneMask = 1 << (s - 1);
        uint32_t thrdMask = (1 << s) - 1;
        uint32_t rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        gl64_t root = d_radix6[rank << (6 - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            gl64_t t = gl64_t::csel(r[1][z], r[0][z], pos);
            t.shfl_bfly(laneMask);
            r[0][z] = gl64_t::csel(r[0][z], t, pos);
            r[1][z] = gl64_t::csel(t, r[1][z], pos);

            t = root * r[1][z];
            r[1][z] = r[0][z] - t;
            r[0][z] = r[0][z] + t;
        }
    }

    #pragma unroll 1
    for (uint32_t s = 6; s < iterations; s++) {
        uint32_t laneMask = 1 << (s - 1);
        uint32_t thrdMask = (1 << s) - 1;
        uint32_t rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        gl64_t root = d_radixX[rank << (radix - (s + 1))];

        gl64_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            gl64_t t = gl64_t::csel(r[1][z], r[0][z], pos);
            xchg[threadIdx.x][z] = t;
        }
        __syncthreads();
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            gl64_t t = xchg[threadIdx.x ^ laneMask][z];
            r[0][z] = gl64_t::csel(r[0][z], t, pos);
            r[1][z] = gl64_t::csel(t, r[1][z], pos);

            t = root * r[1][z];
            r[1][z] = r[0][z] - t;
            r[0][z] = t + r[0][z];
        }
        __syncthreads();
    }

    if (is_intt && (stage + iterations) == lg_domain_size) {
        gl64_t dsi = gl64_t(domain_inv);
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = r[0][z] * dsi;
            r[1][z] = r[1][z] * dsi;
        }
    }

    // rotate "iterations" bits of the output indices
    uint32_t mask = (uint32_t)((1 << iterations) - 1) << stage;
    uint32_t rotw = idx0 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

    if (coalesced) {
        nttTranspose<z_count>(r[0]);
        __syncwarp();
        nttTranspose<z_count>(r[1]);
        nttCoalescedStore<z_count>(d_inout, idx0, r[0], stage);
        nttCoalescedStore<z_count>(d_inout, idx1, r[1], stage);
    } else {
        uint32_t z_shift = inp_mask == 0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            d_inout[idx0 + (z << z_shift)] = r[0][z];
            d_inout[idx1 + (z << z_shift)] = r[1][z];
        }
    }
}

// DIF (decimation-in-frequency) step: natural input order, bit-reversed output --
// the iNTT of the LDE. gridDim.y selects the column.
template<int z_count, bool coalesced = false>
__launch_bounds__(768, 1) __global__
void nttDifColMajorKernel(const uint32_t radix, const uint32_t lg_domain_size,
                          const uint32_t stage, const uint32_t iterations,
                          gl64_t *d_base, const size_t col_stride,
                          const gl64_t (*d_partialRoots)[NTT_WIN_SIZE],
                          const gl64_t (*d_zStepRoots)[1024],
                          const gl64_t *d_radix6, const gl64_t *d_radixX,
                          bool is_intt, const uint64_t domain_inv)
{
    gl64_t *d_inout = d_base + (size_t)blockIdx.y * col_stride;
    extern __shared__ int nttShm[];
    gl64_t *shared_exchange = reinterpret_cast<gl64_t *>(nttShm);

    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint32_t diff_mask = (1 << (iterations - 1)) - 1;
    const uint32_t inp_mask = ((uint32_t)1 << (stage - 1)) - 1;
    const uint32_t out_mask = ((uint32_t)1 << (stage - iterations)) - 1;

    const uint32_t tiz = (tid & ~diff_mask) * z_count + (tid & diff_mask);

    uint32_t idx0 = (tiz & ~inp_mask) * 2;
    idx0 += (tiz << (stage - iterations)) & inp_mask;
    idx0 += (tiz >> (iterations - 1)) & out_mask;
    uint32_t idx1 = idx0 + ((uint32_t)1 << (stage - 1));

    gl64_t r[2][z_count];

    if (coalesced) {
        nttCoalescedLoad<z_count>(r[0], d_inout, idx0, stage - iterations);
        nttCoalescedLoad<z_count>(r[1], d_inout, idx1, stage - iterations);
        nttTranspose<z_count>(r[0]);
        __syncwarp();
        nttTranspose<z_count>(r[1]);
    } else {
        uint32_t z_shift = out_mask == 0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = d_inout[idx0 + (z << z_shift)];
            r[1][z] = d_inout[idx1 + (z << z_shift)];
        }
    }

    #pragma unroll 1
    for (uint32_t s = iterations; --s >= 6;) {
        uint32_t laneMask = 1 << (s - 1);
        uint32_t thrdMask = (1 << s) - 1;
        uint32_t rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        gl64_t root = d_radixX[rank << (radix - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            gl64_t t = root * (r[0][z] - r[1][z]);
            r[0][z] = r[0][z] + r[1][z];
            r[1][z] = t;
        }
        __syncthreads();

        gl64_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            gl64_t t = gl64_t::csel(r[1][z], r[0][z], pos);
            xchg[threadIdx.x][z] = t;
        }
        __syncthreads();
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            gl64_t t = xchg[threadIdx.x ^ laneMask][z];
            r[0][z] = gl64_t::csel(r[0][z], t, pos);
            r[1][z] = gl64_t::csel(t, r[1][z], pos);
        }
    }

    #pragma unroll 1
    for (uint32_t s = min(iterations, 6u); --s >= 1;) {
        uint32_t laneMask = 1 << (s - 1);
        uint32_t thrdMask = (1 << s) - 1;
        uint32_t rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        gl64_t root = d_radix6[rank << (6 - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            gl64_t t = root * (r[0][z] - r[1][z]);
            r[0][z] = r[0][z] + r[1][z];
            r[1][z] = t;

            t = gl64_t::csel(r[1][z], r[0][z], pos);
            t.shfl_bfly(laneMask);
            r[0][z] = gl64_t::csel(r[0][z], t, pos);
            r[1][z] = gl64_t::csel(t, r[1][z], pos);
        }
    }

    #pragma unroll
    for (int z = 0; z < z_count; z++) {
        gl64_t t = r[0][z] - r[1][z];
        r[0][z] = r[0][z] + r[1][z];
        r[1][z] = t;
    }

    if (stage - iterations != 0) {
        uint32_t thread_ntt_pos = (tiz & inp_mask) >> (iterations - 1);
        uint32_t thread_ntt_idx = (tiz & diff_mask) * 2;
        uint32_t nbits = NTT_MAX_LG - (stage - iterations);
        uint32_t idx0r = nttBitRev(thread_ntt_idx, nbits);
        uint32_t root_idx0 = idx0r * thread_ntt_pos;
        uint32_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        gl64_t first_root, second_root;
        nttIntermediateRoots(first_root, second_root, root_idx0, root_idx1, d_partialRoots);
        r[0][0] = r[0][0] * first_root;
        r[1][0] = r[1][0] * second_root;

        if (z_count > 1) {
            uint32_t off = nbits >= 10 ? (nbits - 10) : 0;
            uint32_t scale = nbits >= 10 ? 0 : (10 - nbits);

            thread_ntt_idx <<= scale;
            gl64_t first_root_z = d_zStepRoots[off][thread_ntt_idx];
            gl64_t second_root_z = d_zStepRoots[off][thread_ntt_idx + (1 << scale)];

            #pragma unroll
            for (int z = 1; z < z_count; z++) {
                first_root *= first_root_z;
                second_root *= second_root_z;
                r[0][z] = r[0][z] * first_root;
                r[1][z] = r[1][z] * second_root;
            }
        }
    }

    if (is_intt && stage == iterations) {
        gl64_t dsi = gl64_t(domain_inv);
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = r[0][z] * dsi;
            r[1][z] = r[1][z] * dsi;
        }
    }

    uint32_t mask = (uint32_t)((1 << iterations) - 1) << (stage - iterations);
    uint32_t rotw = idx0 & mask;
    rotw = (rotw << 1) | (rotw >> (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw << 1) | (rotw >> (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

    if (coalesced) {
        nttTranspose<z_count>(r[0]);
        __syncwarp();
        nttTranspose<z_count>(r[1]);
        nttCoalescedStore<z_count>(d_inout, idx0, r[0], stage - iterations + 1);
        nttCoalescedStore<z_count>(d_inout, idx1, r[1], stage - iterations + 1);
    } else {
        uint32_t z_shift = out_mask == 0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            d_inout[idx0 + (z << z_shift)] = r[0][z];
            d_inout[idx1 + (z << z_shift)] = r[1][z];
        }
    }
}

// Coset spread: reads the iNTT result (bit-reversed order), multiplies by
// gen^bit_rev(idx) from the windowed coset-power table, writes to out[idx<<blowup]
// and zero-fills the blowup gaps (out = the destination column). in/out must be disjoint.
__global__ void nttCosetSpreadKernel(gl64_t *out, const gl64_t *in,
                                     const gl64_t (*cosetPows)[NTT_WIN_SIZE],
                                     uint32_t lg_domain_size, uint32_t lg_blowup)
{
    uint32_t domain_size = 1u << lg_domain_size;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size)
        return;

    uint32_t blowup = 1u << lg_blowup;
    gl64_t r = in[idx];

    uint32_t pow = nttBitRev(idx, lg_domain_size);
    // coset root = gen^pow, assembled from the windowed powers
    gl64_t root = cosetPows[0][pow % NTT_WIN_SIZE];
    #pragma unroll
    for (int o = 1; o < NTT_WIN_NUM; o++) {
        uint32_t w = (pow >> (o * NTT_LG_WIN)) % NTT_WIN_SIZE;
        if (w)
            root *= cosetPows[o][w];
    }
    r = r * root;

    size_t base = (size_t)idx << lg_blowup;
    out[base] = r;
    gl64_t zero = gl64_t(uint64_t(0));
    for (uint32_t j = 1; j < blowup; j++)
        out[base + j] = zero;
}

// Copy a chunk's source columns into the TAIL N elements of each destination
// column (the iNTT then runs in place on the tails; src is never written).
__global__ void nttCopyToDestinationTailsKernel(gl64_t *dst_base, const gl64_t *src_base,
                                    uint32_t lg_n, size_t dst_stride)
{
    const size_t N = (size_t)1 << lg_n;
    const gl64_t *src = src_base + (size_t)blockIdx.y * N;
    gl64_t *dst = dst_base + (size_t)blockIdx.y * dst_stride + (dst_stride - N);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < N;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// In-place bit-reversal permutation of each column (gridDim.y selects the column).
// Each (i, rev(i)) pair is swapped once, by its lower-indexed side. Combined with a
// DIT run this forms the NN-order (natural in -> natural out) transform.
__global__ void nttBitRevColMajorKernel(gl64_t *base, size_t col_stride, uint32_t lg)
{
    gl64_t *col = base + (size_t)blockIdx.y * col_stride;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (1u << lg))
        return;
    uint32_t j = nttBitRev(i, lg);
    if (i < j) {
        gl64_t t = col[i];
        col[i] = col[j];
        col[j] = t;
    }
}

// Coset shift + zero-pad for computeQ: reads qDim columns of q coefficients (length NExt,
// only the first N meaningful after the iNTT of a degree-<N polynomial) and writes
// qDeg*qDim columns of cmQ:
//   cmQ[p*qDim + k][row] = q[k][row + p*N] * shiftIn^p   for row < N
//   cmQ[p*qDim + k][row] = 0                             for row >= N
// shiftIn is a BASE-FIELD scalar, so the cubic-extension element scales component-wise.
__global__ void nttCosetShiftQKernel(const gl64_t *q, gl64_t *cmQ, uint32_t N, uint32_t Next,
                                     uint32_t qDeg, uint32_t qDim, uint64_t shiftIn)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Next)
        return;
    gl64_t shift = gl64_t(shiftIn);
    gl64_t s = gl64_t(uint64_t(1));
    for (uint32_t p = 0; p < qDeg; p++) {
        for (uint32_t k = 0; k < qDim; k++) {
            uint32_t outcol = p * qDim + k;
            gl64_t v = (row < N) ? q[(uint64_t)k * Next + (row + (uint64_t)p * N)] * s
                                 : gl64_t(uint64_t(0));
            cmQ[(uint64_t)outcol * Next + row] = v;
        }
        s = shift * s;
    }
}

// ---- per-device twiddle tables (host-generated once, lazily) ----

namespace {

struct NttTables {
    gl64_t *base = nullptr;      // one allocation: partialRoots | cosetPows | zStepRoots | radix blob
    gl64_t *partialRoots;        // [NTT_WIN_NUM][NTT_WIN_SIZE] windowed powers of the master root
    gl64_t *cosetPows;           // [NTT_WIN_NUM][NTT_WIN_SIZE] windowed powers of the coset generator
    gl64_t *zStepRoots;          // [NTT_PO_WINS][1024] step roots between consecutive z-slots
                                 // (slot z+1's positioning exponent = slot z's + a fixed delta;
                                 //  one multiply by this root advances a slot -- see the
                                 //  first_root *= first_root_z chains in the kernels)
    gl64_t *radixTwiddles[5];    // per-radix stage twiddles, radix6..radix10: table r-6 holds
                                 // the first 2^(r-1) powers of w_{2^r}; radixTwiddles[0] also
                                 // serves the shuffle stages (1-5) of every launch
};

NttTables nttTbl[2][NTT_MAX_DEVS];   // [inverse][device]
uint64_t    nttDomainSizeInv[NTT_MAX_LG + 1];    // host copy of domain-size inverses
bool        nttDomainSizeInvReady = false;
std::mutex  nttTablesMutex;

uint64_t nttHostMul(uint64_t a, uint64_t b)
{
    return Goldilocks::toU64(Goldilocks::mul(Goldilocks::fromU64(a), Goldilocks::fromU64(b)));
}

uint64_t nttHostPow(uint64_t base, uint64_t e)
{
    uint64_t acc = 1;
    while (e) {
        if (e & 1) acc = nttHostMul(acc, base);
        base = nttHostMul(base, base);
        e >>= 1;
    }
    return acc;
}

// Build one direction's tables for the CURRENT device. Roots come from the
// omegas[] / omegas_inv[] constants of this TU (single source of truth).
void nttBuildTables(NttTables &t, bool inverse)
{
    const size_t partial_n = (size_t)NTT_WIN_NUM * NTT_WIN_SIZE;
    const size_t plus_n = (size_t)NTT_PO_WINS * 1024;
    const size_t total = 2 * partial_n + plus_n + NTT_RADIX_BLOB;

    uint64_t om[NTT_MAX_LG + 1];
    CHECKCUDAERR(cudaMemcpyFromSymbol(om, inverse ? omegas_inv : omegas, sizeof(om)));

    std::vector<uint64_t> h(total);
    uint64_t *hp = h.data();

    // partial windows of the max-domain root: partial[o][k] = root^(k << (o*LG_WIN))
    uint64_t root = om[NTT_MAX_LG];
    for (int o = 0; o < NTT_WIN_NUM; o++)
        for (uint64_t k = 0; k < NTT_WIN_SIZE; k++)
            hp[o * NTT_WIN_SIZE + k] = nttHostPow(root, k << (o * NTT_LG_WIN));
    hp += partial_n;

    // coset-generator windows (SHIFT = 7 / its inverse), same window scheme
    uint64_t gen = COSET_SHIFT;
    if (inverse)
        gen = nttHostPow(gen, 0xFFFFFFFF00000001ULL - 2);   // Fermat inverse
    for (int o = 0; o < NTT_WIN_NUM; o++)
        for (uint64_t k = 0; k < NTT_WIN_SIZE; k++)
            hp[o * NTT_WIN_SIZE + k] = nttHostPow(gen, k << (o * NTT_LG_WIN));
    hp += partial_n;

    // z-step roots: zStepRoots[o][k] = root^(bit_rev(k,10) << o) -- the exponent DELTA
    // between consecutive z-slots of one thread, per sub-transform size o
    for (int o = 0; o < NTT_PO_WINS; o++)
        for (uint32_t k = 0; k < 1024; k++) {
            uint32_t kr = 0;
            for (int b = 0; b < 10; b++) kr |= ((k >> b) & 1u) << (9 - b);
            hp[o * 1024 + k] = nttHostPow(root, (uint64_t)kr << o);
        }
    hp += plus_n;

    // radix twiddle blob: [radix10(512) | radix9(256) | radix8(128) | radix7(64) | radix6(32)]
    size_t off = 0;
    for (int rad = 10; rad >= 6; rad--) {
        uint64_t w = om[rad];
        size_t n = (size_t)1 << (rad - 1);
        for (size_t k = 0; k < n; k++)
            hp[off + k] = nttHostPow(w, k);
        off += n;
    }

    CHECKCUDAERR(cudaMalloc(&t.base, total * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(t.base, h.data(), total * sizeof(gl64_t), cudaMemcpyHostToDevice));

    t.partialRoots = t.base;
    t.cosetPows = t.partialRoots + partial_n;
    t.zStepRoots = t.cosetPows + partial_n;
    gl64_t *radix = t.zStepRoots + plus_n;
    t.radixTwiddles[4] = radix;              // radix10
    t.radixTwiddles[3] = t.radixTwiddles[4] + 512;      // radix9
    t.radixTwiddles[2] = t.radixTwiddles[3] + 256;      // radix8
    t.radixTwiddles[1] = t.radixTwiddles[2] + 128;      // radix7
    t.radixTwiddles[0] = t.radixTwiddles[1] + 64;       // radix6
}

const NttTables &nttEnsureTables(bool inverse)
{
    int dev = 0;
    CHECKCUDAERR(cudaGetDevice(&dev));
    assert(dev < NTT_MAX_DEVS);
    NttTables &t = nttTbl[inverse ? 1 : 0][dev];

    std::lock_guard<std::mutex> lk(nttTablesMutex);
    if (!t.base) {
        if (!nttDomainSizeInvReady) {
            uint64_t inv2 = 0x7FFFFFFF80000001ULL;   // 2^-1 mod p
            nttDomainSizeInv[0] = 1;
            for (int k = 1; k <= NTT_MAX_LG; k++)
                nttDomainSizeInv[k] = nttHostMul(nttDomainSizeInv[k - 1], inv2);
            nttDomainSizeInvReady = true;
        }
        NttTables fresh;
        nttBuildTables(fresh, inverse);
        t = fresh;
    }
    return t;
}

// Step launcher: same radix/config policy for DIT and DIF; gridDim.y = ncols.
struct NttRun {
    gl64_t *base;          // column 0 of the run
    size_t col_stride;     // ELEMENTS between consecutive columns' bases
    unsigned ncols;        // columns batched per launch (gridDim.y)
    int lg;                // log2 domain size of the transform
    bool is_intt;          // apply 1/N scaling on the final step
    const NttTables &t;    // twiddle tables (forward or inverse direction)
    cudaStream_t s;
    bool dit;              // true = DIT (rev->nat, the NTT), false = DIF (nat->rev, the iNTT)
    int stage = 0;         // internal cursor, set by run() -- callers omit it

    void step(int iterations)
    {
        assert(iterations <= 10);
        const int radix = iterations < 6 ? 6 : iterations;

        uint32_t num_threads = (uint32_t)1 << (lg - 1);
        uint32_t block_size = 1 << (radix - 1);
        block_size = (num_threads <= block_size) ? num_threads : block_size;
        uint32_t num_blocks = (num_threads + block_size - 1) / block_size;

        // Z (z-count): independent butterfly pairs each thread carries in registers
        // (r[2][Z] = 8 field elements). Chosen so 4 consecutive 8-byte elements = 32 B
        // = exactly one memory sector: the coalesced load moves full sectors with no
        // waste, and the positioning roots are derived once per thread then extended
        // to the other Z-1 slots with one multiply each (the zStepRoots tables).
        // Grid and shared memory scale with it: num_blocks/Z blocks, Z*shared_sz bytes.
        const int Z = 4;
        size_t shared_sz = sizeof(gl64_t) << (radix - 1);

#ifdef NTT_ARGS
#error "NTT_ARGS is already defined -- rename one of the two definitions"
#endif
        #define NTT_ARGS (uint32_t)radix, (uint32_t)lg, (uint32_t)stage, (uint32_t)iterations, \
                base, col_stride, (const gl64_t(*)[NTT_WIN_SIZE])t.partialRoots, \
                (const gl64_t(*)[1024])t.zStepRoots, t.radixTwiddles[0], t.radixTwiddles[radix-6], is_intt, nttDomainSizeInv[lg]
        if (dit) {
            if (num_blocks < (uint32_t)Z)
                nttDitColMajorKernel<1><<<dim3(num_blocks, ncols), block_size, shared_sz, s>>>(NTT_ARGS);
            else if (stage == 0 || lg < 12)
                nttDitColMajorKernel<4><<<dim3(num_blocks/Z, ncols), block_size, Z*shared_sz, s>>>(NTT_ARGS);
            else
                nttDitColMajorKernel<4, true><<<dim3(num_blocks/Z, ncols), block_size, Z*shared_sz, s>>>(NTT_ARGS);
            stage += iterations;
        } else {
            if (num_blocks < (uint32_t)Z)
                nttDifColMajorKernel<1><<<dim3(num_blocks, ncols), block_size, shared_sz, s>>>(NTT_ARGS);
            else if (stage == iterations || lg < 12)
                nttDifColMajorKernel<4><<<dim3(num_blocks/Z, ncols), block_size, Z*shared_sz, s>>>(NTT_ARGS);
            else
                nttDifColMajorKernel<4, true><<<dim3(num_blocks/Z, ncols), block_size, Z*shared_sz, s>>>(NTT_ARGS);
            stage -= iterations;
        }
        #undef NTT_ARGS
        CHECKCUDAERR(cudaGetLastError());
    }

    void run()
    {
        stage = dit ? 0 : lg;
        if (lg <= 10) { step(lg); }
        else if (lg <= 18) {
            int st = lg / 2;
            if (dit) { step(st + lg % 2); step(st); }
            else     { step(st); step(st + lg % 2); }
        } else if (lg <= 30) {
            // lg == 29 treated to avoid 9,9,11 split
            int st = lg / 3, rem = lg % 3;
            if (dit) { step(st); step(st + (lg == 29)); step(st + (lg == 29 ? 1 : rem)); }
            else     { step(st + (lg == 29 ? 1 : rem)); step(st + (lg == 29)); step(st); }
        } else { assert(false); }
    }
};


// NN-order (natural in -> natural out) transform: bit-reversal permutation + DIT run.
// inverse=true uses the inverse-direction tables and applies the 1/N scaling.
void nttTransformNN(gl64_t *base, size_t col_stride, unsigned ncols, int lg,
                    bool inverse, const NttTables &t, cudaStream_t s)
{
    uint32_t N = 1u << lg;
    uint32_t thr = 256;
    uint32_t blk = (N + thr - 1) / thr;
    nttBitRevColMajorKernel<<<dim3(blk, ncols), thr, 0, s>>>(base, col_stride, (uint32_t)lg);
    CHECKCUDAERR(cudaGetLastError());
    NttRun{base, col_stride, ncols, lg, inverse, t, s, true}.run();
}

// Column-chunk size that keeps a transform's steps L2-resident between launches
// (same policy as ldeColMajor; ~60% budget leaves room for tables and other traffic).
uint32_t nttL2ChunkCols(size_t colBytes, uint64_t nCols)
{
    int dev = 0, l2Bytes = 0;
    CHECKCUDAERR(cudaGetDevice(&dev));
    if (cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, dev) != cudaSuccess || l2Bytes <= 0)
        l2Bytes = 32 << 20;
    uint32_t chunk = (uint32_t)(((size_t)l2Bytes * 3 / 5) / colBytes);
    if (chunk < 1) chunk = 1;
    if (chunk > nCols) chunk = (uint32_t)nCols;
    return chunk;
}

} // anonymous namespace

// ColMajor LDE: iNTT(N) -> coset spread -> NTT(NExt) per column. Columns are
// processed in L2-sized chunks with column-batched kernel launches; when the
// chunk degenerates to a single column (large domains, already occupancy-
// saturated) the driver switches to the cheaper scratch-staged serial flow,
// which avoids the batched flow's extra tail->scratch staging copy.
void NTTGoldilocksGPU::ldeColMajor(gl64_t *d_dst_, gl64_t *d_src_,
                                   uint64_t nBits, uint64_t nBitsExt, uint64_t nCols,
                                   cudaStream_t stream, bool preserve_src, gl64_t *preserve_scratch)
{
    if (nCols == 0 || nBits == 0)
        return;
    assert(nBitsExt > nBits && nBitsExt <= NTT_MAX_LG);

    const NttTables &tf = nttEnsureTables(false);
    const NttTables &ti = nttEnsureTables(true);

    size_t N = (size_t)1 << nBits;
    size_t Next = (size_t)1 << nBitsExt;
    uint32_t lg_blowup = (uint32_t)(nBitsExt - nBits);

    // Queried per call: cudaGetDevice is a thread-local read and the attribute lookup
    // costs ~1us -- noise next to a millisecond-scale LDE, and it keeps this free of
    // shared mutable state.
    int dev = 0, l2Bytes = 0;
    CHECKCUDAERR(cudaGetDevice(&dev));
    if (cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, dev) != cudaSuccess || l2Bytes <= 0)
        l2Bytes = 32 << 20;
    uint32_t chunk = (uint32_t)(((size_t)l2Bytes * 3 / 5) / (Next * sizeof(gl64_t)));
    if (chunk > nCols) chunk = (uint32_t)nCols;

    // The batched flow stages each column's iNTT result (in the destination tail)
    // through scratch for the spread. The serial flow needs it only to preserve src.
    gl64_t *scratch = preserve_scratch;
    bool own_scratch = false;
    if (scratch == nullptr && (preserve_src || chunk >= 2)) {
        CHECKCUDAERR(cudaMallocAsync(&scratch, N * sizeof(gl64_t), stream));
        own_scratch = true;
    }

    uint32_t sthr = 256;
    uint32_t sblk = (uint32_t)((N + sthr - 1) / sthr);

    if (chunk >= 2) {
        // batched: copy chunk cols to the destination-column tails -> batched DIF-iNTT
        // -> per-column spread (tail -> scratch -> dest column) -> batched DIT-NTT
        for (uint32_t c0 = 0; c0 < nCols; c0 += chunk) {
            uint32_t nc = (c0 + chunk <= nCols) ? chunk : (uint32_t)(nCols - c0);
            gl64_t *dchunk = d_dst_ + (size_t)c0 * Next;

            uint32_t cblk = (uint32_t)std::min<size_t>((N + 255) / 256, 4096);
            nttCopyToDestinationTailsKernel<<<dim3(cblk, nc), 256, 0, stream>>>(dchunk, d_src_ + (size_t)c0 * N,
                                                                   (uint32_t)nBits, Next);
            CHECKCUDAERR(cudaGetLastError());

            NttRun{dchunk + (Next - N), Next, nc, (int)nBits, true, ti, stream, false}.run();

            for (uint32_t c = 0; c < nc; c++) {
                gl64_t *dstCol = dchunk + (size_t)c * Next;
                CHECKCUDAERR(cudaMemcpyAsync(scratch, dstCol + (Next - N), N * sizeof(gl64_t),
                                             cudaMemcpyDeviceToDevice, stream));
                nttCosetSpreadKernel<<<sblk, sthr, 0, stream>>>(dstCol, scratch,
                    (const gl64_t(*)[NTT_WIN_SIZE])tf.cosetPows, (uint32_t)nBits, lg_blowup);
                CHECKCUDAERR(cudaGetLastError());
            }

            NttRun{dchunk, Next, nc, (int)nBitsExt, false, tf, stream, true}.run();
        }
    } else {
        // serial per column: iNTT on scratch (or in place on src when the caller
        // allows trashing it) -> spread -> DIT-NTT on the destination column
        for (uint32_t c = 0; c < nCols; c++) {
            gl64_t *srcCol = d_src_ + (size_t)c * N;
            gl64_t *dstCol = d_dst_ + (size_t)c * Next;
            gl64_t *inttCol = srcCol;            
            if (preserve_src) {
                CHECKCUDAERR(cudaMemcpyAsync(scratch, srcCol, N * sizeof(gl64_t),
                                             cudaMemcpyDeviceToDevice, stream));
                inttCol = scratch;
            }
            NttRun{inttCol, N, 1, (int)nBits, true, ti, stream, false}.run();
            nttCosetSpreadKernel<<<sblk, sthr, 0, stream>>>(dstCol, inttCol,
                (const gl64_t(*)[NTT_WIN_SIZE])tf.cosetPows, (uint32_t)nBits, lg_blowup);
            CHECKCUDAERR(cudaGetLastError());
            NttRun{dstCol, Next, 1, (int)nBitsExt, false, tf, stream, true}.run();
        }
    }

    if (own_scratch)
        CHECKCUDAERR(cudaFreeAsync(scratch, stream));
}

// ColMajor in-place INTT of nCols flat columns of 2^nBits rows (column c at dst + c*N).
// NN order (natural in/out), matching the reference flow. Used for the LEv vector.
void NTTGoldilocksGPU::inttColMajor(gl64_t *dst, uint64_t nBits, uint64_t nCols, cudaStream_t stream)
{
    if (nCols == 0 || nBits == 0)
        return;
    assert(nBits <= NTT_MAX_LG);
    const NttTables &ti = nttEnsureTables(true);
    size_t N = (size_t)1 << nBits;

    uint32_t chunk = nttL2ChunkCols(N * sizeof(gl64_t), nCols);
    for (uint32_t c0 = 0; c0 < nCols; c0 += chunk) {
        uint32_t nc = (c0 + chunk <= nCols) ? chunk : (uint32_t)(nCols - c0);
        nttTransformNN(dst + (size_t)c0 * N, N, nc, (int)nBits, true, ti, stream);
    }
}

// ColMajor computeQ: iNTT(NN) each q column over the extended domain -> coset shift +
// zero-pad into cmQ (disjoint region) -> NTT(NN) each cmQ column.
void NTTGoldilocksGPU::computeQColMajor(uint64_t offset_cmQ, uint64_t offset_q, uint64_t qDeg, uint64_t qDim,
                                        Goldilocks::Element shiftIn, uint64_t nBits, uint64_t nBitsExt,
                                        uint64_t nCols, gl64_t *d_aux_trace, cudaStream_t stream)
{
    assert(nBitsExt <= NTT_MAX_LG);
    const NttTables &tf = nttEnsureTables(false);
    const NttTables &ti = nttEnsureTables(true);

    gl64_t *d_q = d_aux_trace + offset_q;       // flat, qDim cols, NExt rows
    gl64_t *d_cmQ = d_aux_trace + offset_cmQ;   // flat, nCols cols, NExt rows
    uint32_t N = 1u << nBits;
    uint32_t Next = 1u << nBitsExt;

    uint32_t chunk = nttL2ChunkCols((size_t)Next * sizeof(gl64_t), qDim);
    for (uint32_t c0 = 0; c0 < qDim; c0 += chunk) {
        uint32_t nc = (c0 + chunk <= qDim) ? chunk : (uint32_t)(qDim - c0);
        nttTransformNN(d_q + (size_t)c0 * Next, Next, nc, (int)nBitsExt, true, ti, stream);
    }

    uint32_t thr = 256;
    uint32_t blk = (Next + thr - 1) / thr;
    nttCosetShiftQKernel<<<blk, thr, 0, stream>>>(d_q, d_cmQ, N, Next, (uint32_t)qDeg, (uint32_t)qDim, shiftIn.fe);
    CHECKCUDAERR(cudaGetLastError());

    chunk = nttL2ChunkCols((size_t)Next * sizeof(gl64_t), nCols);
    for (uint32_t c0 = 0; c0 < nCols; c0 += chunk) {
        uint32_t nc = (c0 + chunk <= nCols) ? chunk : (uint32_t)(nCols - c0);
        nttTransformNN(d_cmQ + (size_t)c0 * Next, Next, nc, (int)nBitsExt, false, tf, stream);
    }
}


// =============================================================================
// Class methods
// =============================================================================

// Tiled computeQ backend (ColMajorTiled): iNTT(ext) -> coset shift -> NTT(ext). Pure kernels.
void NTTGoldilocksGPU::computeQTiled(uint64_t offset_cmQ, uint64_t offset_q, uint64_t qDeg, uint64_t qDim,
                                           Goldilocks::Element shiftIn, uint64_t nBits, uint64_t nBitsExt,
                                           uint64_t nCols, gl64_t *d_aux_trace, uint64_t offset_helper, cudaStream_t stream)
{
    uint64_t N = 1 << nBits;
    uint64_t NExtended = 1 << nBitsExt;

    gl64_t* d_S = d_aux_trace + offset_helper;
    gl64_t *d_q = d_aux_trace + offset_q;
    gl64_t *d_cmQ = d_aux_trace + offset_cmQ;

    dim3 block(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid0((NExtended + block.x - 1) / block.x,
             (qDim + block.y - 1) / block.y);
    int sharedMemSize = block.x * block.y * sizeof(gl64_t);
    // q-input and cmQ-output storage are ColMajorTiled; the internal butterfly working format is row-major-in-tile.
    columnMajorToRowMajorKernel<<<grid0, block, sharedMemSize, stream>>>(d_q, NExtended, qDim, Layout::ColMajorTiled);
    nttDit(d_q, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, nBitsExt, nBitsExt, qDim, true, false, stream, maxLogDomainSize);

    dim3 threads(TILE_HEIGHT, 1, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, 1, 1);
    applyCosetShiftKernel<<<blocks, threads, 0, stream>>>(d_cmQ, d_q, d_S, shiftIn, N, NExtended, nBitsExt - nBits, qDeg, qDim);

    dim3 grid1((NExtended + block.x - 1) / block.x,
             (nCols + block.y - 1) / block.y);
    nttDit(d_cmQ, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, nBitsExt, nBitsExt, nCols, false, false, stream, maxLogDomainSize);
    rowMajorToColumnMajorKernel<<<grid1, block, sharedMemSize, stream>>>(d_cmQ, NExtended, nCols, Layout::ColMajorTiled);
}

// computeQ: INTT on extended domain -> coset shift -> NTT on extended domain. ColMajor engine
// everywhere (resolveLayout is always ColMajor; the tiled branch is kept only for the legacy
// reference backend and is currently unreachable).
void NTTGoldilocksGPU::computeQ(uint64_t offset_cmQ, uint64_t offset_q, uint64_t qDeg, uint64_t qDim,
                                Goldilocks::Element shiftIn, uint64_t nBits, uint64_t nBitsExt,
                                uint64_t nCols, gl64_t *d_aux_trace, uint64_t offset_helper,
                                TimerGPU &timer, cudaStream_t stream)
{
    if (nCols == 0 || nBitsExt == 0)
    {
        return;
    }

    TimerStartCategoryGPU(timer, NTT);

    if(nBitsExt > maxLogDomainSize)
    {
        printf("[NTT] ERROR: nBitsExt %lu exceeds maxLogDomainSize %lu\n", nBitsExt, maxLogDomainSize);
        abort();
    }

    if (!isGraphCapturableLayout(nBits, nCols)) {
        computeQColMajor(offset_cmQ, offset_q, qDeg, qDim, shiftIn, nBits, nBitsExt, nCols, d_aux_trace, stream);
    } else {
        computeQTiled(offset_cmQ, offset_q, qDeg, qDim, shiftIn, nBits, nBitsExt, nCols, d_aux_trace, offset_helper, stream);
    }

    TimerStopCategoryGPU(timer, NTT);
}

// Tiled LDE backend (ColMajorTiled in/out; d_src_/d_dst_ disjoint -> out-of-place, src intact).
// columnMajorToPacked -> DIF(INTT+zero-pad) -> DIT(NTT) -> rowMajorToColumnMajor. Pure kernels (capturable).
void NTTGoldilocksGPU::ldeTiled(gl64_t* d_dst_, gl64_t* d_src_,
                                      uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, cudaStream_t stream)
{
    uint64_t size = 1 << nBits;
    uint64_t ext_size = 1 << nBitsExt;

    dim3 block(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid0((size + block.x - 1) / block.x,
             (nCols + block.y - 1) / block.y);
    int sharedMemSize = block.x * block.y * sizeof(gl64_t);

    columnMajorToPackedRowMajorKernel<<<grid0, block, sharedMemSize, stream>>>(d_src_, nBits, d_dst_, nBitsExt, nCols, Layout::ColMajorTiled);
    nttDifLde(d_dst_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, nBits, nBitsExt, nCols, true, true, stream, maxLogDomainSize);
    nttDitLde(d_dst_, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, nBitsExt, nBitsExt, nCols, false, false, stream, maxLogDomainSize);
    dim3 grid1((ext_size + block.x - 1) / block.x,
             (nCols + block.y - 1) / block.y);
    rowMajorToColumnMajorKernel<<<grid1, block, sharedMemSize, stream>>>(d_dst_, ext_size, nCols, Layout::ColMajorTiled);
}

// LDE: ColMajor engine everywhere (resolveLayout is always ColMajor; the tiled branch is kept only
// for the legacy reference backend and is currently unreachable).
void NTTGoldilocksGPU::LDE(gl64_t* d_dst, uint64_t offset_dst,
                           gl64_t* d_src, uint64_t offset_src,
                           uint64_t nBits, uint64_t nBitsExt, uint64_t nCols,
                           TimerGPU &timer, cudaStream_t stream, bool preserve_src,
                           gl64_t* preserve_scratch){

    if (nCols == 0 || nBits == 0)
    {
        return;
    }
    TimerStartCategoryGPU(timer, NTT);
    if (nBitsExt > maxLogDomainSize)
    {
        printf("[NTT] ERROR: nBitsExt %lu exceeds maxLogDomainSize %lu\n", nBitsExt, maxLogDomainSize);
        abort();
    }

    gl64_t *d_dst_ = &d_dst[offset_dst];
    gl64_t *d_src_ = &d_src[offset_src];

    if (!isGraphCapturableLayout(nBits, nCols)) {
        ldeColMajor(d_dst_, d_src_, nBits, nBitsExt, nCols, stream, preserve_src, preserve_scratch);
    } else {
        ldeTiled(d_dst_, d_src_, nBits, nBitsExt, nCols, stream);
    }
    TimerStopCategoryGPU(timer, NTT);
}

// Forward NTT: columnMajorToRowMajor -> bitReversal + DIT -> rowMajorToColumnMajor
void NTTGoldilocksGPU::NTT(gl64_t *dst, uint64_t nBits, uint64_t nCols, cudaStream_t stream)
{
    if (nCols == 0 || nBits == 0)
    {
        return;
    }
    if (nBits > maxLogDomainSize)
    {
        printf("[NTT] ERROR: nBits %lu exceeds maxLogDomainSize %lu\n", nBits, maxLogDomainSize);
        abort();
    }

    uint64_t N = 1 << nBits;
    Layout layout = resolveLayout(nBits, nCols);

    dim3 block_0(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid_0((N + block_0.x - 1) / block_0.x,
             (nCols + block_0.y - 1) / block_0.y);
    int sharedMemSize_0 = block_0.x * block_0.y * sizeof(gl64_t);
    columnMajorToRowMajorKernel<<<grid_0, block_0, sharedMemSize_0, stream>>>(dst, N, nCols, layout);
    nttDit(dst, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, nBits, nBits, nCols, false, false, stream, maxLogDomainSize);
    rowMajorToColumnMajorKernel<<<grid_0, block_0, sharedMemSize_0, stream>>>(dst, N, nCols, layout);
}

// Tiled INTT backend (ColMajorTiled in-place): transpose tiled-storage -> row-major, nttDit
// (inverse), transpose back. Pure kernels (capturable).
void NTTGoldilocksGPU::inttTiled(gl64_t *dst, uint64_t nBits, uint64_t nCols, cudaStream_t stream)
{
    uint64_t N = 1 << nBits;
    dim3 block_0(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid_0((N + block_0.x - 1) / block_0.x,
             (nCols + block_0.y - 1) / block_0.y);
    int sharedMemSize_0 = block_0.x * block_0.y * sizeof(gl64_t);
    columnMajorToRowMajorKernel<<<grid_0, block_0, sharedMemSize_0, stream>>>(dst, N, nCols, Layout::ColMajorTiled);
    nttDit(dst, d_r, d_fwd_twiddle_factors, d_inv_twiddle_factors, nBits, nBits, nCols, true, false, stream, maxLogDomainSize);
    rowMajorToColumnMajorKernel<<<grid_0, block_0, sharedMemSize_0, stream>>>(dst, N, nCols, Layout::ColMajorTiled);
}

// Inverse NTT: ColMajor engine everywhere (resolveLayout is always ColMajor; the tiled branch is
// kept only for the legacy reference backend and is currently unreachable).
void NTTGoldilocksGPU::INTT(gl64_t *dst, uint64_t nBits, uint64_t nCols, cudaStream_t stream)
{
    if (nCols == 0 || nBits == 0)
    {
        return;
    }
    if (nBits > maxLogDomainSize)
    {
        printf("[NTT] ERROR: nBits %lu exceeds maxLogDomainSize %lu\n", nBits, maxLogDomainSize);
        abort();
    }

    if (resolveLayout(nBits, nCols) == Layout::ColMajor) {
        inttColMajor(dst, nBits, nCols, stream);
    } else {
        inttTiled(dst, nBits, nCols, stream);
    }
}

// =============================================================================
// Static member definitions
// =============================================================================

gl64_t **NTTGoldilocksGPU::d_fwd_twiddle_factors = nullptr;
gl64_t **NTTGoldilocksGPU::d_inv_twiddle_factors = nullptr;
gl64_t **NTTGoldilocksGPU::d_r = nullptr;
uint64_t NTTGoldilocksGPU::maxLogDomainSize = 0;
uint32_t NTTGoldilocksGPU::nGPUs_available = 0;


// Allocates and computes per-GPU precomputed tables:
//   - d_fwd_twiddle_factors: forward NTT twiddle factors (omega^k), size 2^(maxLogDomainSize-1)
//   - d_inv_twiddle_factors: inverse NTT twiddle factors (omega_inv^k), size 2^(maxLogDomainSize-1)
//   - d_r: coset shift powers (COSET_SHIFT^k), size 2^maxLogDomainSize
// Thread-safe; skips GPUs already initialized; re-initializes all if maxLogDomainSize grows.
void NTTGoldilocksGPU::initConstants(uint64_t maxLogDomainSize_, uint32_t nGPUs_input, uint32_t* gpu_ids_) {
    static std::mutex init_mutex;
    std::lock_guard<std::mutex> lock(init_mutex);


    int nGPUs_available_;
    cudaGetDeviceCount(&nGPUs_available_);
    assert(maxLogDomainSize_ <= 32);

    if(maxLogDomainSize_ > maxLogDomainSize || nGPUs_available_ != nGPUs_available) {
        freeConstants();
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
    bool stream_created[nGPUs];
    for (int i = 0; i < nGPUs; i++) {
        stream_created[i] = false;
    }
    for (int i = 0; i < nGPUs; i++) {
        if (d_fwd_twiddle_factors[gpu_ids[i]] != nullptr && d_inv_twiddle_factors[gpu_ids[i]] != nullptr && d_r[gpu_ids[i]] != nullptr) {
            continue; // Already initialized
        } else {
            assert(d_fwd_twiddle_factors[gpu_ids[i]] == nullptr && d_inv_twiddle_factors[gpu_ids[i]] == nullptr && d_r[gpu_ids[i]] == nullptr);
            cudaSetDevice(gpu_ids[i]);
            cudaStreamCreate(&stream[i]);
            stream_created[i] = true;
            cudaMalloc(&d_fwd_twiddle_factors[gpu_ids[i]], (1 << (maxLogDomainSize - 1)) * sizeof(gl64_t));
            cudaMalloc(&d_inv_twiddle_factors[gpu_ids[i]], (1 << (maxLogDomainSize - 1)) * sizeof(gl64_t));
            cudaMalloc(&d_r[gpu_ids[i]], (1 << maxLogDomainSize) * sizeof(gl64_t));
            evalTwiddleFactors(d_fwd_twiddle_factors[gpu_ids[i]], d_inv_twiddle_factors[gpu_ids[i]], maxLogDomainSize, stream[i]);
            evalCosetShifts(d_r[gpu_ids[i]], maxLogDomainSize, stream[i]);
        }
    }
    for (int i = 0; i < nGPUs; i++) {
        if (stream_created[i]) {
            cudaSetDevice(gpu_ids[i]);
            cudaStreamSynchronize(stream[i]);
            cudaStreamDestroy(stream[i]);
        }
    }

    if(free_inputs) {
        delete[] gpu_ids;
    }
    CHECKCUDAERR(cudaGetLastError());
}

void NTTGoldilocksGPU::freeConstants() {
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

    d_fwd_twiddle_factors = nullptr;
    d_inv_twiddle_factors = nullptr;
    d_r = nullptr;
    maxLogDomainSize = 0;
    nGPUs_available = 0;
}
