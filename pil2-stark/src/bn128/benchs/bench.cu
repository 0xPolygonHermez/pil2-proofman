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
// Utilities
// =====================

// Generate random scalars in parallel
static void generate_random_scalars(uint8_t* scalars, uint64_t n, uint64_t scalar_size = 32, int seed = 42) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(seed + tid);
        
        uint64_t start_idx = (tid * n) / omp_get_num_threads();
        uint64_t end_idx = ((tid + 1) * n) / omp_get_num_threads();
        
        for (uint64_t i = start_idx; i < end_idx; i++) {
            uint8_t* scalar = &scalars[i * scalar_size];
            for (size_t j = 0; j < scalar_size; j++) {
                scalar[j] = rng() & 0xFF;
            }
            // Ensure scalar < field order by clearing top bits (~253 bits)
            scalar[scalar_size - 1] &= 0x1F;
        }
    }
}

// Generate curve points in parallel: points[i] = 2^i * G (GPU format)
static void generate_curve_points_gpu(PointAffineGPU* points, uint64_t n) {
    const int nChunks = omp_get_max_threads();
    std::vector<AltBn128::G1Point> chunkStarts(nChunks);
    
    uint64_t chunkSize = (n + nChunks - 1) / nChunks;
    AltBn128::G1Point acc;
    AltBn128::G1.copy(acc, AltBn128::G1.oneAffine());
    
    for (int c = 0; c < nChunks; c++) {
        AltBn128::G1.copy(chunkStarts[c], acc);
        for (uint64_t j = 0; j < chunkSize && (c * chunkSize + j) < n; j++) {
            AltBn128::G1.dbl(acc, acc);
        }
    }
    
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
            memcpy(&points[i].x, &P_affine.x, sizeof(AltBn128::F1Element));
            memcpy(&points[i].y, &P_affine.y, sizeof(AltBn128::F1Element));
            AltBn128::G1.dbl(localPoint, localPoint);
        }
    }
}

// =====================
// MSM GPU Benchmark
// =====================

static const uint64_t MSM_SCALAR_SIZE = 32;

static void MSM_GPU_BENCH(benchmark::State &state) {
    uint64_t power = state.range(0);
    uint64_t n = 1ULL << power;
    
    // Allocate data
    PointAffineGPU* points = new PointAffineGPU[n];
    BN128GPUScalarField::Element* scalars = new BN128GPUScalarField::Element[n];
    
    // Generate test data
    generate_curve_points_gpu(points, n);
    generate_random_scalars(reinterpret_cast<uint8_t*>(scalars), n, MSM_SCALAR_SIZE, 42);
    
    PointJacobianGPU result;
    
    // Warm-up GPU
    MSM_BN128_GPU::msm(result, points, scalars, n, false);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        MSM_BN128_GPU::msm(result, points, scalars, n, false);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(result);
    }
    
    // Cleanup
    delete[] points;
    delete[] scalars;
    
    // Report throughput
    state.counters["log2(n)"] = power;
    state.SetItemsProcessed(state.iterations() * n);
}

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
    
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    
    return 0;
}
