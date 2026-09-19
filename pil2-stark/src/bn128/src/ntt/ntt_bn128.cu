// NTT GPU implementation for BN128/BN254 scalar field
// Uses supranational/sppark NTT algorithm

#include <cuda.h>
#include <cstring>
#include <vector>

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

// Low degree extension matching NTT_AltBn128::extendPol: interpolate, zero-pad
// the coefficients, evaluate over the larger domain.
//
// Built from NTT::Base rather than NTT::LDE on purpose. NTT::LDE distributes
// generator powers over the coefficients (LDE_spread_distribute_powers with
// perform_shift = true), which evaluates over a coset; extendPol does not
// shift, so the two disagree. Composing two Base calls with an explicit zero
// fill reproduces the CPU transform exactly.
void NTT_BN128_GPU::lde(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t lg_blowup) {
    fr_t* gpu_data = reinterpret_cast<fr_t*>(data);
    auto& gpu = select_gpu(-1);

    size_t n = (size_t)1 << lg_n;
    size_t ext_n = n << lg_blowup;

    // Interpolate over the small domain: evaluations -> coefficients.
    RustError err = NTT::Base(gpu, gpu_data, lg_n,
                              NTT::InputOutputOrder::NN,
                              NTT::Direction::inverse,
                              NTT::Type::standard);
    if (err.code != 0) {
        fprintf(stderr, "LDE interpolate error: %d\n", err.code);
        return;
    }

    // Zero-pad the coefficients up to the extended domain.
    memset(&data[n], 0, (ext_n - n) * sizeof(BN128GPUScalarField::Element));

    // Evaluate over the extended domain.
    err = NTT::Base(gpu, gpu_data, lg_n + lg_blowup,
                    NTT::InputOutputOrder::NN,
                    NTT::Direction::forward,
                    NTT::Type::standard);
    if (err.code != 0) {
        fprintf(stderr, "LDE evaluate error: %d\n", err.code);
    }
}

// sppark's own LDE: shifts onto a coset. Kept distinct from lde() so the
// difference is a choice at the call site rather than a silent mismatch.
void NTT_BN128_GPU::lde_coset(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t lg_blowup) {
    fr_t* gpu_data = reinterpret_cast<fr_t*>(data);
    auto& gpu = select_gpu(-1);

    RustError err = NTT::LDE(gpu, gpu_data, lg_n, lg_blowup);
    if (err.code != 0) {
        fprintf(stderr, "coset LDE error: %d\n", err.code);
    }
}

namespace {

// The CPU interleaves columns row-major; sppark needs one contiguous column.
// These move a single column in and out of a scratch buffer.
inline void gather_column(BN128GPUScalarField::Element* dst,
                          const BN128GPUScalarField::Element* src,
                          size_t rows, uint32_t ncols, uint32_t col) {
    for (size_t r = 0; r < rows; r++) dst[r] = src[r * ncols + col];
}

inline void scatter_column(BN128GPUScalarField::Element* dst,
                           const BN128GPUScalarField::Element* src,
                           size_t rows, uint32_t ncols, uint32_t col) {
    for (size_t r = 0; r < rows; r++) dst[r * ncols + col] = src[r];
}

} // namespace

void NTT_BN128_GPU::ntt_multicol(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t ncols) {
    size_t n = (size_t)1 << lg_n;
    std::vector<BN128GPUScalarField::Element> col(n);
    for (uint32_t c = 0; c < ncols; c++) {
        gather_column(col.data(), data, n, ncols, c);
        ntt(col.data(), lg_n);
        scatter_column(data, col.data(), n, ncols, c);
    }
}

void NTT_BN128_GPU::intt_multicol(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t ncols) {
    size_t n = (size_t)1 << lg_n;
    std::vector<BN128GPUScalarField::Element> col(n);
    for (uint32_t c = 0; c < ncols; c++) {
        gather_column(col.data(), data, n, ncols, c);
        intt(col.data(), lg_n);
        scatter_column(data, col.data(), n, ncols, c);
    }
}

void NTT_BN128_GPU::lde_multicol(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t lg_blowup,
                                 uint32_t ncols) {
    size_t n = (size_t)1 << lg_n;
    size_t ext_n = n << lg_blowup;

    // One column at a time: gather its 2^lg_n input rows, extend, scatter the
    // 2^(lg_n + lg_blowup) results back. sppark has no batched NTT, so the
    // per-column launch overhead is unavoidable here.
    std::vector<BN128GPUScalarField::Element> col(ext_n);
    for (uint32_t c = 0; c < ncols; c++) {
        gather_column(col.data(), data, n, ncols, c);
        lde(col.data(), lg_n, lg_blowup);
        scatter_column(data, col.data(), ext_n, ncols, c);
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
