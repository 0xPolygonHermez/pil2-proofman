// NTT GPU implementation for BN128/BN254 scalar field
// Uses supranational/sppark NTT algorithm

#include <cuda.h>

// Enable BN254 curve
#ifndef FEATURE_BN254
#define FEATURE_BN254
#endif

#include "ntt_bn128.cuh"
#include <ff/alt_bn128.hpp>
#include <ntt/ntt.cuh>

#ifndef __CUDA_ARCH__

void NTT_BN128_GPU::ntt(BN128GPUScalarField::Element* data, uint32_t lg_n) {
    // BN128GPUScalarField::Element and fr_t should have the same memory layout
    // Both use 256-bit Montgomery representation
    fr_t* gpu_data = reinterpret_cast<fr_t*>(data);
    
    // Select current GPU
    auto& gpu = select_gpu(-1);
    
    // Call sppark's NTT::Base with forward direction
    // InputOutputOrder::NN = Natural order in, Natural order out
    RustError err = NTT::Base(gpu, gpu_data, lg_n,
                              NTT::InputOutputOrder::NN,
                              NTT::Direction::forward,
                              NTT::Type::standard);
    
    if (err.code != 0) {
        // Handle error - for now just print
        fprintf(stderr, "NTT error: %d\n", err.code);
    }
}

void NTT_BN128_GPU::intt(BN128GPUScalarField::Element* data, uint32_t lg_n) {
    // BN128GPUScalarField::Element and fr_t should have the same memory layout
    // Both use 256-bit Montgomery representation
    fr_t* gpu_data = reinterpret_cast<fr_t*>(data);
    
    // Select current GPU
    auto& gpu = select_gpu(-1);
    
    // Call sppark's NTT::Base with inverse direction
    // InputOutputOrder::NN = Natural order in, Natural order out
    RustError err = NTT::Base(gpu, gpu_data, lg_n,
                              NTT::InputOutputOrder::NN,
                              NTT::Direction::inverse,
                              NTT::Type::standard);
    
    if (err.code != 0) {
        // Handle error - for now just print
        fprintf(stderr, "INTT error: %d\n", err.code);
    }
}

// C-linkage wrapper functions for calling from g++ code
extern "C" void ntt_bn128_gpu(void* data, uint32_t lg_n) {
    fr_t* gpu_data = reinterpret_cast<fr_t*>(data);
    
    auto& gpu = select_gpu(-1);
    
    RustError err = NTT::Base(gpu, gpu_data, lg_n,
                              NTT::InputOutputOrder::NN,
                              NTT::Direction::forward,
                              NTT::Type::standard);
    
    if (err.code != 0) {
        fprintf(stderr, "NTT GPU error: %d\n", err.code);
    }
}

extern "C" void intt_bn128_gpu(void* data, uint32_t lg_n) {
    fr_t* gpu_data = reinterpret_cast<fr_t*>(data);
    
    auto& gpu = select_gpu(-1);
    
    RustError err = NTT::Base(gpu, gpu_data, lg_n,
                              NTT::InputOutputOrder::NN,
                              NTT::Direction::inverse,
                              NTT::Type::standard);
    
    if (err.code != 0) {
        fprintf(stderr, "INTT GPU error: %d\n", err.code);
    }
}

extern "C" void ntt_bn128_gpu_dev_ptr(void* d_data, uint32_t lg_n) {
    fr_t* d_fr = reinterpret_cast<fr_t*>(d_data);
    
    auto& gpu = select_gpu(-1);
    stream_t& stream = gpu;
    
    NTT::Base_dev_ptr(stream, d_fr, lg_n,
                      NTT::InputOutputOrder::NN,
                      NTT::Direction::forward,
                      NTT::Type::standard);
    
    CUDA_OK(cudaStreamSynchronize(stream));
}

extern "C" void intt_bn128_gpu_dev_ptr(void* d_data, uint32_t lg_n) {
    fr_t* d_fr = reinterpret_cast<fr_t*>(d_data);
    
    auto& gpu = select_gpu(-1);
    stream_t& stream = gpu;
    
    NTT::Base_dev_ptr(stream, d_fr, lg_n,
                      NTT::InputOutputOrder::NN,
                      NTT::Direction::inverse,
                      NTT::Type::standard);
    
    CUDA_OK(cudaStreamSynchronize(stream));
}

#endif // !__CUDA_ARCH__
