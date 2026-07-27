/*
 * NTT GPU Implementation for Goldilocks Field
 *
 * This file implements the Number Theoretic Transform (NTT) on GPU for the
 * Goldilocks prime field (p = 2^64 - 2^32 + 1).
 *
 * This is the NATIVE, TILED backend, selected by resolveLayout (goldilocks_trace_layout.cuh) for
 * Layout::ColMajorTiled sections (small domain, many columns). The default Layout::ColMajor (flat
 * column-major) path is delegated to sppark instead -- see sppark_lde.cu; the LDE/computeQ/INTT methods
 * below branch on resolveLayout and only run the tiled kernels here when it returns ColMajorTiled.
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

// sppark-backed flat-layout NTT primitives (defined in the isolated sppark_lde.cu TU).
#include "sppark_lde.cuh"

#define COSET_SHIFT 7

// --- Forward declarations ---

__global__ void nttDitButterflyFlat8Kernel(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size, uint32_t log_domain_size, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t col_min, uint32_t col_max);
__global__ void nttDitButterflyKernel(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void nttDifButterflyPackedKernel( gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size_in, uint32_t log_domain_size_in, uint32_t domain_size_out, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t blowupFactor);
__global__ void nttDitButterflyFlatKernel(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t nCols, bool inverse, bool extend, uint64_t maxLogDomainSize);
__global__ void bitReversalFlatKernel(gl64_t *data, uint32_t log_domain_size, uint32_t nCols);
__global__ void evalTwiddleSmallKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void evalTwiddleFirstKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
__global__ void evalTwiddleSecondKernel(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size);
void evalTwiddleFactors(gl64_t *fwd_twiddles, gl64_t *inv_twiddles, uint32_t log_domain_size, cudaStream_t stream);
__global__ void evalCosetShiftSmallKernel(gl64_t *r, uint32_t log_domain_size);
__global__ void evalCosetShiftFirstKernel(gl64_t *r, uint32_t log_domain_size);
__global__ void evalCosetShiftSecondKernel(gl64_t *r, uint32_t log_domain_size);
void evalCosetShifts(gl64_t *r, uint32_t log_domain_size, cudaStream_t stream);
void nttFlat( gl64_t *data, gl64_t **d_r, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize);
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
// Class methods
// =============================================================================

// sppark flat-layout computeQ backend (ColMajor). Host-syncs -> not graph-capturable.
void NTTGoldilocksGPU::computeQSppark(uint64_t offset_cmQ, uint64_t offset_q, uint64_t qDeg, uint64_t qDim,
                                      Goldilocks::Element shiftIn, uint64_t nBits, uint64_t nBitsExt,
                                      uint64_t nCols, gl64_t *d_aux_trace, cudaStream_t stream)
{
    sppark_computeq_flat((void *)d_aux_trace, offset_cmQ, offset_q,
                         (uint32_t)qDeg, (uint32_t)qDim, Goldilocks::toU64(shiftIn),
                         (uint32_t)nBits, (uint32_t)nBitsExt, (uint32_t)nCols, (void *)stream);
}

// Native tiled computeQ backend (ColMajorTiled): iNTT(ext) -> coset shift -> NTT(ext). Pure kernels.
void NTTGoldilocksGPU::computeQNativeTiled(uint64_t offset_cmQ, uint64_t offset_q, uint64_t qDeg, uint64_t qDim,
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

// computeQ: INTT on extended domain -> coset shift -> NTT on extended domain.
// Dispatch on resolveLayout(nBits, nCols) to the sppark (flat) or native (tiled) backend.
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
        computeQSppark(offset_cmQ, offset_q, qDeg, qDim, shiftIn, nBits, nBitsExt, nCols, d_aux_trace, stream);
    } else {
        computeQNativeTiled(offset_cmQ, offset_q, qDeg, qDim, shiftIn, nBits, nBitsExt, nCols, d_aux_trace, offset_helper, stream);
    }

    TimerStopCategoryGPU(timer, NTT);
}

// sppark flat-layout LDE backend (ColMajor in/out). Host-syncs -> not graph-capturable.
void NTTGoldilocksGPU::ldeSppark(gl64_t* d_dst_, gl64_t* d_src_,
                                 uint64_t nBits, uint64_t nBitsExt, uint64_t nCols,
                                 cudaStream_t stream, bool preserve_src, gl64_t* preserve_scratch)
{
    sppark_lde_flat((void *)d_dst_, (void *)d_src_, (uint32_t)nBits, (uint32_t)nBitsExt, (uint32_t)nCols, preserve_src, (void *)preserve_scratch, (void *)stream);
}

// Native tiled LDE backend (ColMajorTiled in/out; d_src_/d_dst_ disjoint -> out-of-place, src intact).
// columnMajorToPacked -> DIF(INTT+zero-pad) -> DIT(NTT) -> rowMajorToColumnMajor. Pure kernels (capturable).
void NTTGoldilocksGPU::ldeNativeTiled(gl64_t* d_dst_, gl64_t* d_src_,
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

// LDE: dispatch on resolveLayout(nBits, nCols) to the sppark (flat) or native (tiled) backend.
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
        ldeSppark(d_dst_, d_src_, nBits, nBitsExt, nCols, stream, preserve_src, preserve_scratch);
    } else {
        ldeNativeTiled(d_dst_, d_src_, nBits, nBitsExt, nCols, stream);
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

// sppark flat-layout INTT backend (ColMajor in-place). Host-syncs -> not graph-capturable.
void NTTGoldilocksGPU::inttSppark(gl64_t *dst, uint64_t nBits, uint64_t nCols, cudaStream_t stream)
{
    sppark_intt_flat((void*)dst, (uint32_t)nBits, (uint32_t)nCols, (void *)stream);
}

// Native tiled INTT backend (ColMajorTiled in-place): transpose tiled-storage -> row-major, nttDit
// (inverse), transpose back. Pure kernels (capturable).
void NTTGoldilocksGPU::inttNativeTiled(gl64_t *dst, uint64_t nBits, uint64_t nCols, cudaStream_t stream)
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

// Inverse NTT: dispatch on resolveLayout(nBits, nCols) to the sppark (flat) or native (tiled) backend.
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
        inttSppark(dst, nBits, nCols, stream);
    } else {
        inttNativeTiled(dst, nBits, nCols, stream);
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

// =============================================================================
// Reference/debug NTT (flat layout, not used in production)
// =============================================================================

#define TPB_NTT 16

// Single-stage DIT butterfly, flat layout, one thread per (butterfly, col).
// Naive reference implementation for debugging.
// Launch: <<<domainSize/2, nCols>>>
__global__ void nttDitButterflyFlatKernel(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t stage, uint32_t domain_size, uint32_t log_domain_size, uint32_t nCols, bool inverse, bool extend, uint64_t maxLogDomainSize)
{
    uint32_t i = blockIdx.x;
    uint32_t col = threadIdx.x;

    if (i < domain_size / 2 && col < nCols)
    {
        uint32_t group_size = 1 << stage;
        uint32_t group = i >> stage;
        uint32_t group_pos = i & (group_size - 1);
        uint32_t index1 = (group << (stage + 1)) + group_pos;
        uint32_t index2 = index1 + group_size;
        gl64_t factor = twiddles[group_pos * ((1 << maxLogDomainSize) >> (stage + 1))];
        gl64_t odd_sub = gl64_t((uint64_t)data[index2 * nCols + col]) * factor;
        gl64_t result1 = gl64_t((uint64_t)data[index1 * nCols + col]) + odd_sub;
        gl64_t result2 = gl64_t((uint64_t)data[index1 * nCols + col]) - odd_sub;

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

// 8-step DIT butterfly, flat layout, 1KB shared memory tile.
// Processes 4 columns at a time in an outer loop.
// Launch: <<<domainSize/256, 256>>>
__global__ void nttDitButterflyFlat8Kernel(gl64_t *data, gl64_t *twiddles, gl64_t* d_r, uint32_t domain_size, uint32_t log_domain_size, uint32_t nCols, uint32_t base_step, bool suffle, bool inverse, bool extend, uint64_t maxLogDomainSize, uint32_t col_min, uint32_t col_max)
{
    __shared__ gl64_t tile[1024];

    uint32_t n_loc_steps = min(log_domain_size - base_step, 8);
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t groupSize = 1 << base_step;
    uint32_t nGroups = domain_size / groupSize;
    uint32_t low_bits = row / nGroups;
    uint32_t high_bits = row % nGroups;
    row = high_bits * groupSize + low_bits;

    uint32_t remaining_high_bits = log_domain_size - (base_step+1);
    uint32_t high_mask = (1 << remaining_high_bits) - 1;

    for(int col_base = col_min; col_base <= col_max; col_base +=4){

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
            if (threadIdx.x < 128){
                uint32_t group_size = 1 << loc_step;
                uint32_t group = i >> loc_step;
                uint32_t group_pos = i & (group_size - 1);
                uint32_t index1 = (group << (loc_step + 1)) + group_pos;
                uint32_t index2 = index1 + group_size;
                gl64_t factor;
                {
                    uint32_t gs = base_step + loc_step;
                    uint32_t ggs = 1 << gs;
                    uint32_t ggp =(blockIdx.x << 7) + i;
                    ggp = ((ggp & high_mask)<< base_step) + (ggp >> remaining_high_bits);
                    ggp = ggp & (ggs - 1);
                    factor = twiddles[ggp*((1 << maxLogDomainSize) >> (gs + 1))];
                }

                index1 = index1 << 2;
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

// Bit-reversal permutation on flat (non-tiled) data.
// Launch: <<<8192, TPB_NTT>>>
__global__ void bitReversalFlatKernel(gl64_t *data, uint32_t log_domain_size, uint32_t nCols)
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

// Flat NTT driver (reference/debug). Uses flat bit-reversal + flat butterfly kernels.
// Not used in production — kept for debugging and testing.
void nttFlat( gl64_t *data, gl64_t **d_r_, gl64_t **d_fwd_twiddle_factors, gl64_t **d_inv_twiddle_factors, uint32_t log_domain_size, uint32_t nCols, bool inverse, bool extend, cudaStream_t stream, uint64_t maxLogDomainSize)
{
    assert(log_domain_size >= 1 && "Domain size must be >= 2 for NTT");
    uint32_t domain_size = 1 << log_domain_size;

    dim3 blockDim;
    dim3 gridDim;

    blockDim = dim3(TPB_NTT);
    gridDim = dim3(8192);
    bitReversalFlatKernel<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size, nCols);
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

    if(log_domain_size >= 8) {
         for(uint32_t step = 0; step < log_domain_size; step+=8){
                nttDitButterflyFlat8Kernel<<<domain_size / 256, 256, 0, stream>>>(data, d_twiddles, d_r, domain_size, log_domain_size, nCols, step, true, inverse, extend, maxLogDomainSize, 0, nCols-1);
                CHECKCUDAERR(cudaGetLastError());
        }
    } else {
        for (uint32_t stage = 0; stage < log_domain_size; stage++)
        {
            nttDitButterflyFlatKernel<<<domain_size / 2, nCols, 0, stream>>>(data, d_twiddles, d_r, stage, domain_size, log_domain_size, nCols, inverse, extend, maxLogDomainSize);
            CHECKCUDAERR(cudaGetLastError());
        }
    }
}
