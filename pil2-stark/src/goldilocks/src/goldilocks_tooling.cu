#include "goldilocks_tooling.cuh"
#include "cuda_utils.cuh"
#include "goldilocks_trace_layout.cuh"
#include <cstdio>

// When the source is already host-pinned (e.g. via register_host_memory) and
// large, skip the pinned-staging double-buffer and issue a single direct H2D.
// Self-selecting: a non-pinned source returns false and falls through to the
// staged path, so this is always safe to attempt.
static bool copy_direct_registered_h2d_if_enabled(const void *src, void *dst, uint64_t total_size, cudaStream_t stream)
{
    if (total_size < 16 * 1024 * 1024) return false;

    cudaPointerAttributes attrs;
    cudaError_t attrErr = cudaPointerGetAttributes(&attrs, src);
    if (attrErr != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    // Only take the fast path when src is host memory the driver can DMA directly.
    if (attrs.type != cudaMemoryTypeHost) return false;

    // TEMP-DIAG: the direct DMA reads [src, src+total_size). Only the FIRST byte was checked
    // above; if the tail is not registered host memory the copy engine reads unmapped pages ->
    // cudaErrorIllegalAddress, surfacing later at the trace_copy_event sync. Probe the last byte.
    {
        static int diagShown = 0;
        if (diagShown < 3) {
            diagShown++;
            fprintf(stderr, "[H2D-DIAG] direct path ACTIVE (probe live): src=%p size=%llu\n",
                    src, (unsigned long long)total_size);
            fflush(stderr);
        }
        cudaPointerAttributes endAttrs;
        const void *last = (const void *)((const uint8_t *)src + total_size - 1);
        cudaError_t e2 = cudaPointerGetAttributes(&endAttrs, last);
        if (e2 != cudaSuccess || endAttrs.type != cudaMemoryTypeHost) {
            fprintf(stderr, "[H2D-DIAG] SOURCE RANGE NOT FULLY REGISTERED: src=%p size=%llu last=%p err=%d type=%d\n",
                    src, (unsigned long long)total_size, last, (int)e2,
                    (e2 == cudaSuccess) ? (int)endAttrs.type : -1);
            fflush(stderr);
            cudaGetLastError();
        }
    }

    CHECKCUDAERR(cudaMemcpyAsync(dst, src, total_size, cudaMemcpyHostToDevice, stream));
    return true;
}

// Kernel: row-major input -> destination layout `layout` (see getBufferOffset).
// Uses blockIdx.x for rows (which can be very large) and blockIdx.y for cols.
__global__ void fromRowMajorToColMajor(
    const uint64_t nRows,
    const uint64_t nCols,
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    Layout layout
) {
    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < nRows && col < nCols) {
        uint64_t inputOffset = row * nCols + col;
        uint64_t outputOffset = getBufferOffset(row, col, nRows, nCols, layout);
        output[outputOffset] = input[inputOffset];
    }
}

void fromRowMajorToColMajor(
    uint64_t nRows,
    uint64_t nCols,
    gl64_t* src,
    gl64_t* dst,
    Layout layout,
    cudaStream_t stream
) {
    if (nCols == 0 || nRows == 0) return;
    dim3 block(TILE_HEIGHT, TILE_WIDTH);
    dim3 grid((nRows + block.x - 1) / block.x,
              (nCols + block.y - 1) / block.y);
    fromRowMajorToColMajor<<<grid, block, 0, stream>>>(nRows, nCols, (uint64_t*)src, (uint64_t*)dst, layout);
    CHECKCUDAERR(cudaGetLastError());
}

#ifndef __GOLDILOCKS_ENV__
void copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const void* src,
    void* dst,
    uint64_t total_size,
    uint64_t streamId,
    TimerGPU &timer
    ){
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;

    cudaSetDevice(gpuId);

    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;

    // Fast paths that do NOT need the shared pinned-staging buffer, so they skip
    // the per-GPU mutex_pinned lock (avoids serializing behind large stagings):
    //  - direct: large + already host-pinned source
    //  - small:  sub-threshold copy, one shot
    TimerStartCategoryGPU(timer, H2D_COPY);
    if (copy_direct_registered_h2d_if_enabled(src, dst, total_size, stream)) {
        // The direct path leaves a DMA reading `src`, so mark where it finishes: this is what
        // gates recycling the host trace buffer (see wait_trace_h2d_done).
        cudaEventRecord(d_buffers->streamsData[streamId].trace_copy_event, stream);
        TimerStopCategoryGPU(timer, H2D_COPY);
        return;
    }

    std::lock_guard<std::mutex> lock(d_buffers->mutex_pinned[gpuLocalId]);
    uint64_t block_size = d_buffers->pinned_size;
    Goldilocks::Element *pinned_buffer = d_buffers->pinned_buffer[gpuLocalId];
    Goldilocks::Element *pinned_buffer_extra = d_buffers->pinned_buffer_extra[gpuLocalId];

    uint64_t nBlocks = (total_size + block_size - 1) / block_size;

    Goldilocks::Element *pinned_buffer_temp;
    
    uint64_t copySizeBlock = std::min(block_size, total_size);
    std::memcpy(pinned_buffer_extra, (const uint8_t*)src, copySizeBlock);

    for (uint64_t i = 1; i < nBlocks; ++i) {
        CHECKCUDAERR(cudaStreamSynchronize(stream));

        pinned_buffer_temp = pinned_buffer;
        pinned_buffer = pinned_buffer_extra;
        pinned_buffer_extra = pinned_buffer_temp;

        uint64_t copySizeBlockPrev = std::min(block_size, total_size - (i - 1) * block_size);

        CHECKCUDAERR(cudaMemcpyAsync(
            (uint8_t*)dst + (i - 1) * block_size,
            pinned_buffer,
            copySizeBlockPrev,
            cudaMemcpyHostToDevice,
            stream));

        uint64_t copySizeBlock = std::min(block_size, total_size - i * block_size);
        std::memcpy(pinned_buffer_extra, (const uint8_t*)src + i * block_size, copySizeBlock);
    }

    CHECKCUDAERR(cudaStreamSynchronize(stream));
    
    uint64_t copySizeBlockFinal = std::min(block_size, total_size - (nBlocks - 1) * block_size);
    
    CHECKCUDAERR(cudaMemcpyAsync(
        (uint8_t*)dst + (nBlocks - 1) * block_size,
        pinned_buffer_extra,
        copySizeBlockFinal,
        cudaMemcpyHostToDevice,
        stream
    ));

    CHECKCUDAERR(cudaStreamSynchronize(stream));
    // Staged path: `src` was memcpy'd into the pinned staging buffer and the copies are already
    // synced, so the host trace buffer is free here. Record anyway so the event always marks this
    // commit's release point rather than leaving a stale record from an earlier one.
    cudaEventRecord(d_buffers->streamsData[streamId].trace_copy_event, stream);
    TimerStopCategoryGPU(timer, H2D_COPY);
}

void copy_to_device_in_chunks(
    const uint8_t* src,
    uint8_t* dst,
    uint64_t total_size_bytes,
    uint8_t* pinnedBuffer,
    uint64_t pinnedBufferSize,
    cudaStream_t stream
){
    if (copy_direct_registered_h2d_if_enabled(src, dst, total_size_bytes, stream)) return;

     uint64_t block_size = pinnedBufferSize/2;
    
    uint8_t *pinned_buffer = pinnedBuffer;
    uint8_t *pinned_buffer_extra = (uint8_t *)pinnedBuffer + block_size;
    uint64_t nBlocks = (total_size_bytes + block_size - 1) / block_size;

    uint8_t *pinned_buffer_temp;
    
    uint64_t copySizeBlock = std::min(block_size, total_size_bytes);
    std::memcpy(pinned_buffer_extra, (const uint8_t*)src, copySizeBlock);

    for (uint64_t i = 1; i < nBlocks; ++i) {
        CHECKCUDAERR(cudaStreamSynchronize(stream));

        pinned_buffer_temp = pinned_buffer;
        pinned_buffer = pinned_buffer_extra;
        pinned_buffer_extra = pinned_buffer_temp;

        uint64_t copySizeBlockPrev = std::min(block_size, total_size_bytes - (i - 1) * block_size);

        CHECKCUDAERR(cudaMemcpyAsync(
            (uint8_t*)dst + (i - 1) * block_size,
            pinned_buffer,
            copySizeBlockPrev,
            cudaMemcpyHostToDevice,
            stream));

        uint64_t copySizeBlock = std::min(block_size, total_size_bytes - i * block_size);
        std::memcpy(pinned_buffer_extra, (const uint8_t*)src + i * block_size, copySizeBlock);
    }

    CHECKCUDAERR(cudaStreamSynchronize(stream));
    
    uint64_t copySizeBlockFinal = std::min(block_size, total_size_bytes - (nBlocks - 1) * block_size);
    
    CHECKCUDAERR(cudaMemcpyAsync(
        (uint8_t*)dst + (nBlocks - 1) * block_size,
        pinned_buffer_extra,
        copySizeBlockFinal,
        cudaMemcpyHostToDevice,
        stream
    ));

    CHECKCUDAERR(cudaStreamSynchronize(stream));

}

void load_and_copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const char* bufferPath,
    void* dst,
    uint64_t total_size,
    uint64_t streamId,
    uint64_t header_skip_bytes
    ){

    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;

    cudaSetDevice(gpuId);

    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    std::lock_guard<std::mutex> lock(d_buffers->mutex_pinned[gpuLocalId]);

    uint64_t block_size = d_buffers->pinned_size;

    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    Goldilocks::Element *pinned_buffer = d_buffers->pinned_buffer[gpuLocalId];
    Goldilocks::Element *pinned_buffer_extra = d_buffers->pinned_buffer_extra[gpuLocalId];

    uint64_t nBlocks = (total_size + block_size - 1) / block_size;

    Goldilocks::Element *pinned_buffer_temp;

    loadFileParallel_block(pinned_buffer_extra, bufferPath, block_size, true, 0, header_skip_bytes);

    for (uint64_t i = 1; i < nBlocks; ++i) {
        CHECKCUDAERR(cudaStreamSynchronize(stream));

        pinned_buffer_temp = pinned_buffer;
        pinned_buffer = pinned_buffer_extra;
        pinned_buffer_extra = pinned_buffer_temp;

        uint64_t copySizeBlockPrev = std::min(block_size, total_size - (i - 1) * block_size);
        CHECKCUDAERR(cudaMemcpyAsync(
            (uint8_t*)dst + (i - 1) * block_size,
            pinned_buffer,
            copySizeBlockPrev,
            cudaMemcpyHostToDevice,
            stream));

        loadFileParallel_block(pinned_buffer_extra, bufferPath, block_size, true, i, header_skip_bytes);
    }

    CHECKCUDAERR(cudaStreamSynchronize(stream));

    uint64_t copySizeBlockFinal = std::min(block_size, total_size - (nBlocks - 1) * block_size);

    CHECKCUDAERR(cudaMemcpyAsync(
        (uint8_t*)dst + (nBlocks - 1) * block_size,
        pinned_buffer_extra,
        copySizeBlockFinal,
        cudaMemcpyHostToDevice,
        stream
    ));

    CHECKCUDAERR(cudaStreamSynchronize(stream));
}
#endif // __GOLDILOCKS_ENV__