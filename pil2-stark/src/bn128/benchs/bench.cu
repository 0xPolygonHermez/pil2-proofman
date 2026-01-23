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
#include "ntt_bn128.cuh"
#include "poseidon_bn128.cuh"
#include "poseidon2_bn128.cuh"
#include "point.cuh"
#include "alt_bn128.hpp"
#include "fft.hpp"

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

// =====================
// NTT GPU Benchmark
// =====================

static void NTT_GPU_BENCH(benchmark::State &state) {
    uint64_t power = state.range(0);
    uint64_t n = 1ULL << power;
    
    // Allocate data (use RawFr::Element which has same layout as GPU element)
    RawFr::Element* data = new RawFr::Element[n];
    generate_random_scalars(reinterpret_cast<uint8_t*>(data), n, sizeof(RawFr::Element));
    BN128GPUScalarField::Element* gpu_data = reinterpret_cast<BN128GPUScalarField::Element*>(data);
    
    // Warm-up GPU
    NTT_BN128_GPU::ntt(gpu_data, power);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {        
        NTT_BN128_GPU::ntt(gpu_data, power);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(data);
    }
    
    delete[] data;
    
    // Report throughput
    state.counters["log2(n)"] = power;
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(NTT_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->DenseRange(22, 25);

// =====================
// Poseidon GPU Benchmark
// =====================

static const int POSEIDON_NUM_HASHES = 10000;

static void POSEIDON_SEQ_GPU_BENCH(benchmark::State &state) {
    int t = state.range(0);
    
    // Initialize GPU constants
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    BN128GPUScalarField::Element* d_state = nullptr;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    
    // Initialize host state
    RawFr field;
    RawFr::Element* h_state = new RawFr::Element[t];
    for (int i = 0; i < t; i++) {
        field.fromUI(h_state[i], i);
    }
    
    // Warm-up
    cudaMemcpy(d_state, h_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyHostToDevice);
    PoseidonBN128GPU poseidon;
    poseidon.hash(d_state, t);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        // Copy fresh state to device
        cudaMemcpy(d_state, h_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyHostToDevice);
        for(int i = 0; i < POSEIDON_NUM_HASHES; i++){
            poseidon.hash(d_state, t);
        }
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(d_state);
    }
    
    // Cleanup
    cudaFree(d_state);
    delete[] h_state;
    
    state.counters["t"] = t;
    state.counters["hashes"] = POSEIDON_NUM_HASHES;
    state.SetItemsProcessed(state.iterations() * POSEIDON_NUM_HASHES);
}

BENCHMARK(POSEIDON_SEQ_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Args({2})
    ->Args({3})
    ->Args({4})
    ->Args({5})
    ->Args({6})
    ->Args({7})
    ->Args({8})
    ->Args({9})
    ->Args({10})
    ->Args({11})
    ->Args({12})
    ->Args({13})
    ->Args({14})
    ->Args({15})
    ->Args({16})
    ->Args({17});

// =====================
// Poseidon2 GPU Benchmark
// =====================

static const int POSEIDON2_NUM_HASHES = 10000;

static void POSEIDON2_SEQ_GPU_BENCH(benchmark::State &state) {
    int t = state.range(0);
    
    // Initialize GPU constants
    uint32_t gpu_idxs[] = {0};
    Poseidon2BN128GPU::initGPUConstants(gpu_idxs, 1);
    
    BN128GPUScalarField::Element* d_state = nullptr;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    
    // Initialize host state
    RawFr field;
    RawFr::Element* h_state = new RawFr::Element[t];
    for (int i = 0; i < t; i++) {
        field.fromUI(h_state[i], i);
    }
    
    // Warm-up
    cudaMemcpy(d_state, h_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyHostToDevice);
    Poseidon2BN128GPU poseidon2;
    poseidon2.hash(d_state, t);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        // Copy fresh state to device
        cudaMemcpy(d_state, h_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyHostToDevice);
        for(int i = 0; i < POSEIDON2_NUM_HASHES; i++){
            poseidon2.hash(d_state, t);
        }
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(d_state);
    }
    
    // Cleanup
    cudaFree(d_state);
    delete[] h_state;
    
    state.counters["t"] = t;
    state.counters["hashes"] = POSEIDON2_NUM_HASHES;
    state.SetItemsProcessed(state.iterations() * POSEIDON2_NUM_HASHES);
}

BENCHMARK(POSEIDON2_SEQ_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Args({2})
    ->Args({3})
    ->Args({4})
    ->Args({8})
    ->Args({12})
    ->Args({16});

// =====================
// LinearHash GPU Benchmark
// =====================

static void LINEARHASH_GPU_BENCH(benchmark::State &state) {
    uint64_t rows = state.range(0);
    uint64_t cols = state.range(1);
    int t = state.range(2);
    
    // Initialize GPU constants
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    uint64_t* h_input = new uint64_t[rows * cols];
    for (uint64_t i = 0; i < rows * cols; i++) {
        h_input[i] = i;
    }
    
    uint64_t* d_input = nullptr;
    BN128GPUScalarField::Element* d_output = nullptr;
    cudaMalloc(&d_input, rows * cols * sizeof(uint64_t));
    cudaMalloc(&d_output, rows * sizeof(BN128GPUScalarField::Element));
    
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Warm-up
    PoseidonBN128GPU::linearHash(d_output, d_input, cols, rows, t, false, 0);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        PoseidonBN128GPU::linearHash(d_output, d_input, cols, rows, t, false, 0);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(d_output);
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    state.counters["rows"] = rows;
    state.counters["cols"] = cols;
    state.counters["t"] = t;
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

BENCHMARK(LINEARHASH_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Args({1024, 100, 17})      // 1K rows × 100 cols, t=17
    ->Args({1024, 1000, 17})     // 1K rows × 1000 cols, t=17
    ->Args({1 << 16, 100, 17})   // 64K rows × 100 cols, t=17
    ->Args({1 << 16, 1000, 17})  // 64K rows × 1000 cols, t=17
    ->Args({1 << 20, 100, 17})   // 1M rows × 100 cols, t=17
    ->Args({1024, 100, 9})       // 1K rows × 100 cols, t=9
    ->Args({1 << 16, 100, 9});   // 64K rows × 100 cols, t=9

// =====================
// LinearHashTiles GPU Benchmark (Tiled layout)
// =====================

static void LINEARHASHTILES_GPU_BENCH(benchmark::State &state) {
    uint64_t rows = state.range(0);
    uint64_t cols = state.range(1);
    int t = state.range(2);
    
    // Initialize GPU constants
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    uint64_t* h_input = new uint64_t[rows * cols];
    for (uint64_t i = 0; i < rows * cols; i++) {
        h_input[i] = i;
    }
    
    uint64_t* d_input = nullptr;
    BN128GPUScalarField::Element* d_output = nullptr;
    cudaMalloc(&d_input, rows * cols * sizeof(uint64_t));
    cudaMalloc(&d_output, rows * sizeof(BN128GPUScalarField::Element));
    
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Warm-up
    PoseidonBN128GPU::linearHashTiles(d_output, d_input, cols, rows, t, false, 0);
    cudaDeviceSynchronize();
    
    for (auto _ : state) {
        PoseidonBN128GPU::linearHashTiles(d_output, d_input, cols, rows, t, false, 0);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(d_output);
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    state.counters["rows"] = rows;
    state.counters["cols"] = cols;
    state.counters["t"] = t;
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

BENCHMARK(LINEARHASHTILES_GPU_BENCH)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Args({1024, 100, 17})      // 1K rows × 100 cols, t=17
    ->Args({1024, 1000, 17})     // 1K rows × 1000 cols, t=17
    ->Args({1 << 16, 100, 17})   // 64K rows × 100 cols, t=17
    ->Args({1 << 16, 1000, 17})  // 64K rows × 1000 cols, t=17
    ->Args({1 << 20, 100, 17})   // 1M rows × 100 cols, t=17
    ->Args({1024, 100, 9})       // 1K rows × 100 cols, t=9
    ->Args({1 << 16, 100, 9});   // 64K rows × 100 cols, t=9

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
