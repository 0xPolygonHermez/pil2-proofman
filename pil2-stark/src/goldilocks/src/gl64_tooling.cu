#include "gl64_tooling.cuh"

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
