#include "gl64_tooling.cuh"

__global__ void unpack_rows_kernel_pinned(
    const uint64_t* src,
    uint64_t* dst,
    uint64_t nRows,
    uint64_t nCols,
    uint64_t words_per_row,
    const uint64_t* __restrict__ d_unpack_info
) {
    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) return;

    const uint64_t* packed_row = src + row * words_per_row;
    uint64_t* unpacked_row = dst + row * nCols;

    uint64_t word = packed_row[0];
    uint64_t word_idx = 0;
    uint64_t bit_offset = 0;

    #pragma unroll
    for (uint64_t c = 0; c < nCols; c++) {
        uint64_t nbits = __ldg(&d_unpack_info[c]);
        uint64_t val;

        uint64_t bits_left = 64 - bit_offset;

        if (nbits <= bits_left) {
            uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
            val = (word >> bit_offset) & mask;
            bit_offset += nbits;

            if (bit_offset == 64 && word_idx + 1 < words_per_row) {
                word = packed_row[++word_idx];
                bit_offset = 0;
            }
        } else {
            uint64_t low = word >> bit_offset;
            word = packed_row[++word_idx];
            uint64_t high = word & ((1ULL << (nbits - bits_left)) - 1ULL);
            val = (high << bits_left) | low;
            bit_offset = nbits - bits_left;
        }

        unpacked_row[c] = val;
    }
}

void copy_to_device_in_chunks_packed(
    DeviceCommitBuffers* d_buffers,
    const void* src,
    void* dst,
    uint64_t N,
    uint64_t nCols,
    PackedInfo *packed_info,
    uint64_t streamId
    ){

    uint64_t* d_unpack_info;
    cudaMalloc(&d_unpack_info, nCols * sizeof(uint64_t));
    cudaMemcpy(d_unpack_info, packed_info->unpack_info, nCols * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;

    cudaSetDevice(gpuId);

    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    std::lock_guard<std::mutex> lock(d_buffers->mutex_pinned[gpuLocalId]);

    uint64_t block_size = d_buffers->pinned_size;
    
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    uint64_t *pinned_buffer = (uint64_t *)d_buffers->pinned_buffer[gpuLocalId];
    uint64_t *pinned_buffer_extra = (uint64_t *)d_buffers->pinned_buffer_extra[gpuLocalId];

    uint64_t total_size = packed_info->num_packed_words * N * sizeof(Goldilocks::Element);

    uint64_t nBlocks = (total_size + block_size - 1) / block_size;

    uint64_t *pinned_buffer_temp;
    
    uint64_t copySizeBlock = std::min(block_size, total_size);
    std::memcpy(pinned_buffer_extra, (const uint8_t*)src, copySizeBlock);

    for (uint64_t i = 1; i < nBlocks; ++i) {
        CHECKCUDAERR(cudaStreamSynchronize(stream));

        pinned_buffer_temp = pinned_buffer;
        pinned_buffer = pinned_buffer_extra;
        pinned_buffer_extra = pinned_buffer_temp;

        uint64_t copySizeBlockPrev = std::min(block_size, total_size - (i - 1) * block_size);
        uint64_t nRowsPerBlock = copySizeBlockPrev / (packed_info->num_packed_words * sizeof(Goldilocks::Element));
        
        dim3 threads(256);
        dim3 blocks((nRowsPerBlock + threads.x - 1) / threads.x);
        unpack_rows_kernel_pinned<<<blocks, threads, 0, stream>>>(
            pinned_buffer,
            (uint64_t*)dst + (i - 1) * (block_size / sizeof(Goldilocks::Element)),
            nRowsPerBlock,
            nCols,
            packed_info->num_packed_words,
            d_unpack_info
        );
        CHECKCUDAERR(cudaGetLastError());

        uint64_t copySizeBlock = std::min(block_size, total_size - i * block_size);
        std::memcpy(pinned_buffer_extra, (const uint8_t*)src + i * block_size, copySizeBlock);
    }

    CHECKCUDAERR(cudaStreamSynchronize(stream));
    
    uint64_t copySizeBlockFinal = std::min(block_size, total_size - (nBlocks - 1) * block_size);
    uint64_t nRowsPerBlockFinal = copySizeBlockFinal / (packed_info->num_packed_words * sizeof(Goldilocks::Element));
    dim3 threads(256);
    dim3 blocks((nRowsPerBlockFinal + threads.x - 1) / threads.x);
    unpack_rows_kernel_pinned<<<blocks, threads, 0, stream>>>(
        pinned_buffer,
        (uint64_t*)dst + (nBlocks - 1) * (block_size / sizeof(Goldilocks::Element)),
        nRowsPerBlockFinal,
        nCols,
        packed_info->num_packed_words,
        d_unpack_info
    );
    CHECKCUDAERR(cudaGetLastError());

    CHECKCUDAERR(cudaStreamSynchronize(stream));
    
    cudaFree(d_unpack_info);
}

void copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const void* src,
    void* dst,
    uint64_t total_size,
    uint64_t streamId
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
}

void load_and_copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const char* bufferPath,
    void* dst,
    uint64_t total_size,
    uint64_t streamId
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

    loadFileParallel_block(pinned_buffer_extra, bufferPath, block_size, true, 0);

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
        
        loadFileParallel_block(pinned_buffer_extra, bufferPath, block_size, true, i);
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
