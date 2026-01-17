#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <omp.h>
#include <random>

#include "bn128.cuh"
#include "fq.cuh"
#include "msm_bn128.cuh"
#include "point.cuh"
#include "alt_bn128.hpp"

// =====================
// MSM GPU Benchmark
// =====================

// Global storage for precomputed test data (allocated once)
static PointAffineGPU* g_msm_points_gpu = nullptr;
static BN128GPUScalarField::Element* g_msm_scalars_gpu = nullptr;
static uint64_t g_msm_n = 0;
static const uint64_t MSM_SCALAR_SIZE = 32;  // 256-bit scalars

static void setup_msm_data_gpu(uint64_t n) {
    if (g_msm_points_gpu != nullptr && g_msm_n == n) {
        return;  // Already initialized with same size
    }
    
    // Cleanup previous allocation
    if (g_msm_points_gpu != nullptr) {
        delete[] g_msm_points_gpu;
        delete[] g_msm_scalars_gpu;
    }
    
    g_msm_n = n;
    g_msm_points_gpu = new PointAffineGPU[n];
    g_msm_scalars_gpu = new BN128GPUScalarField::Element[n];
    
    // Generate base points: g_msm_points_gpu[i] = 2^i * G
    
    // Pre-compute chunk starting points for parallel generation
    const int nChunks = omp_get_max_threads();
    std::vector<AltBn128::G1Point> chunkStarts(nChunks);
    
    // Compute chunk starting points sequentially
    uint64_t chunkSize = (n + nChunks - 1) / nChunks;
    AltBn128::G1Point acc;
    AltBn128::G1.copy(acc, AltBn128::G1.oneAffine());
    
    for (int c = 0; c < nChunks; c++) {
        AltBn128::G1.copy(chunkStarts[c], acc);
        for (uint64_t j = 0; j < chunkSize && (c * chunkSize + j) < n; j++) {
            AltBn128::G1.dbl(acc, acc);
        }
    }
    
    // Generate points in parallel: base[i] = 2^i * G
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t start_idx = tid * chunkSize;
        uint64_t end_idx = std::min(start_idx + chunkSize, n);
        
        AltBn128::G1Point localPoint;
        AltBn128::G1.copy(localPoint, chunkStarts[tid]);
        
        AltBn128::G1PointAffine P_affine;
        for (uint64_t i = start_idx; i < end_idx; i++) {
            AltBn128::G1.copy(P_affine, localPoint);
            memcpy(&g_msm_points_gpu[i].x, &P_affine.x, sizeof(AltBn128::F1Element));
            memcpy(&g_msm_points_gpu[i].y, &P_affine.y, sizeof(AltBn128::F1Element));
            AltBn128::G1.dbl(localPoint, localPoint);
        }
    }
    
    // Generate random scalars in parallel
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // Use thread-local RNG with unique seed per thread
        std::mt19937_64 rng(42 + tid);  // Deterministic seed for reproducibility
        
        uint64_t start_idx = (tid * n) / omp_get_num_threads();
        uint64_t end_idx = ((tid + 1) * n) / omp_get_num_threads();
        
        for (uint64_t i = start_idx; i < end_idx; i++) {
            uint8_t scalar[MSM_SCALAR_SIZE];
            // Generate 253-bit scalar (BN254 scalar field is ~253 bits)
            for (size_t j = 0; j < MSM_SCALAR_SIZE; j++) {
                scalar[j] = rng() & 0xFF;
            }
            // Ensure scalar < field order by clearing top bits
            scalar[MSM_SCALAR_SIZE - 1] &= 0x1F;  // ~253 bits
            memcpy(&g_msm_scalars_gpu[i], scalar, MSM_SCALAR_SIZE);
        }
    }
}

static void cleanup_msm_data_gpu() {
    if (g_msm_points_gpu != nullptr) {
        delete[] g_msm_points_gpu;
        delete[] g_msm_scalars_gpu;
        g_msm_points_gpu = nullptr;
        g_msm_scalars_gpu = nullptr;
        g_msm_n = 0;
    }
}

static void MSM_GPU_BENCH(benchmark::State &state) {
    // state.range(0) is the power of 2, so n = 2^range(0)
    uint64_t power = state.range(0);
    uint64_t n = 1ULL << power;
    
    setup_msm_data_gpu(n);
    
    PointJacobianGPU gpu_result;
    
    // Warm-up GPU
    MSM_BN128_GPU::msm(gpu_result, g_msm_points_gpu, g_msm_scalars_gpu, n, false);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        MSM_BN128_GPU::msm(gpu_result, g_msm_points_gpu, g_msm_scalars_gpu, n, false);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(gpu_result);
    }
    
    // Report throughput
    state.SetItemsProcessed(state.iterations() * n);
    state.counters["points"] = n;
    state.counters["log2(n)"] = power;
}

// Register MSM GPU benchmarks: argument is the power of 2 (n = 2^arg)
// Default: 22, 23, 24, 25 -> 4M, 8M, 16M, 32M points
BENCHMARK(MSM_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->DenseRange(22, 25);

int main(int argc, char** argv) {
    // Print GPU info
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << " (" << prop.totalGlobalMem / (1024*1024*1024) << " GB)" << std::endl;
    }
    
    // Run benchmarks
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    
    // Cleanup MSM test data
    cleanup_msm_data_gpu();
    
    return 0;
}
