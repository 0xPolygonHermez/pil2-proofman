#include "gl64_tooling.cuh"


void copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const void* src,
    void* dst,
    uint64_t total_size,
    uint64_t streamId
    ){

    uint64_t block_size = d_buffers->streamsData[streamId].pinned_size;
    Goldilocks::Element *pinned_buffer = d_buffers->streamsData[streamId].pinned_buffer;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;

    uint64_t nBlocks = (total_size + block_size - 1) / block_size;

    for (uint64_t i = 0; i < nBlocks; ++i) {
        uint64_t copySizeBlock = std::min(block_size, total_size - i * block_size);

        std::memcpy(pinned_buffer, (const uint8_t*)src + i * block_size, copySizeBlock);

        CHECKCUDAERR(cudaMemcpyAsync(
            (uint8_t*)dst + i * block_size,
            pinned_buffer,
            copySizeBlock,
            cudaMemcpyHostToDevice,
            stream));

        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }
}

void load_and_copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const char* bufferPath,
    void* dst,
    uint64_t total_size,
    uint64_t streamId
    ){

    uint64_t block_size = d_buffers->streamsData[streamId].pinned_size;
    Goldilocks::Element *pinned_buffer = d_buffers->streamsData[streamId].pinned_buffer;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;

    uint64_t nBlocks = (total_size + block_size - 1) / block_size;

    for (uint64_t i = 0; i < nBlocks; ++i) {
        uint64_t copySizeBlock = std::min(block_size, total_size - i * block_size);

        loadFileParallel_block(pinned_buffer, bufferPath, block_size, true, i);

        CHECKCUDAERR(cudaMemcpyAsync(
            (uint8_t*)dst + i * block_size,
            pinned_buffer,
            copySizeBlock,
            cudaMemcpyHostToDevice,
            stream));

        CHECKCUDAERR(cudaStreamSynchronize(stream));
    }
}
