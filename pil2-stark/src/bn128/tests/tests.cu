
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "bn128.cuh"
#include "fq.cuh"
#include "poseidon_bn128.cuh"
#include "poseidon2_bn128.cuh"
#include "msm_bn128.cuh"
#include "ntt_bn128.cuh"
#include "point.cuh"
#include "alt_bn128.hpp"
#include "fft.hpp"
#include "cuda_utils.cuh"
#include "fr.hpp"

__global__ void kernel_fr_add_one_one(int* ok);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__global__ void kernel_fr_add_one_one(int* ok)
{
    BN128GPUScalarField::Element a;
    BN128GPUScalarField::Element b;
    BN128GPUScalarField::Element r;

    a[0] = 1;
    b[0] = 1;
    for(int i = 1; i < 8; ++i) {
        a[i] = 0;
        b[i] = 0;
    }    
    a.v.to(); // Convert to montgomery form
    b.v.to(); // Convert to montgomery form

    BN128GPUScalarField::add(r, a, b);

    r.v.from(); // Convert back from montgomery form
    
    bool same = true;    
    for(int i = 0; i < 8; ++i) {
        uint32_t expected = (i == 0) ? 2 : 0;
        if(r[i] != expected) {
            same = false;
        }
    }
    *ok = same;
}
#endif

TEST(BN128_FR, add)
{
    int *d_ok = nullptr;
    int h_ok = 0;
    cudaMalloc(&d_ok, sizeof(int));
    cudaMemset(d_ok, 0, sizeof(int));

    kernel_fr_add_one_one<<<1,1>>>(d_ok);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_ok);

    EXPECT_EQ(h_ok, 1);
}

// =====================
// Fq (Base Field) Tests
// =====================
__global__ void kernel_fq_add_one_one(int* ok);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__global__ void kernel_fq_add_one_one(int* ok)
{
    BN128GPUBaseField::Element a;
    BN128GPUBaseField::Element b;
    BN128GPUBaseField::Element r;

    a[0] = 1;
    b[0] = 1;
    for(int i = 1; i < 8; ++i) {
        a[i] = 0;
        b[i] = 0;
    }    
    a.v.to(); // Convert to montgomery form
    b.v.to(); // Convert to montgomery form

    BN128GPUBaseField::add(r, a, b);

    r.v.from(); // Convert back from montgomery form
    
    bool same = true;    
    for(int i = 0; i < 8; ++i) {
        uint32_t expected = (i == 0) ? 2 : 0;
        if(r[i] != expected) {
            same = false;
        }
    }
    *ok = same;
}
#endif

TEST(BN128_FQ, add)
{
    int *d_ok = nullptr;
    int h_ok = 0;
    cudaMalloc(&d_ok, sizeof(int));
    cudaMemset(d_ok, 0, sizeof(int));

    kernel_fq_add_one_one<<<1,1>>>(d_ok);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_ok);

    EXPECT_EQ(h_ok, 1);
}

// Forward declarations for GPU kernels
__global__ void init_state_kernel(BN128GPUScalarField::Element* state, int t);
__global__ void from_montgomery_kernel(BN128GPUScalarField::Element* state, int t);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
// GPU kernel to initialize state values: state[i] = i (in Montgomery form)
__global__ void init_state_kernel(BN128GPUScalarField::Element* state, int t) {
    for (int i = 0; i < t; i++) {
        // Initialize to i
        state[i][0] = i;
        for (int j = 1; j < 8; j++) {
            state[i][j] = 0;
        }
        state[i].v.to(); // Convert to Montgomery form
    }
}

// GPU kernel to convert state from Montgomery form and copy to result buffer
__global__ void from_montgomery_kernel(BN128GPUScalarField::Element* state, int t) {
    for (int i = 0; i < t; i++) {
        state[i].v.from();
    }
}

#endif

TEST(BN128_POSEIDON2_TEST, hash_gpu_t2) {
    
    Poseidon2BN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 2;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    Poseidon2BN128GPU::initGPUConstants(gpu_idxs, 1); // Initialize GPU constants
    
    // Initialize state: state[i] = i (in Montgomery form)
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    // Run hash kernel
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    // Convert from Montgomery form
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    // Copy result to host
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    // Format as hex strings for comparison
    char hex0[65], hex1[65]; //64 hex chars + 1 null
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex1, sizeof(hex1), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[1][7], h_state[1][6], h_state[1][5], h_state[1][4], h_state[1][3], h_state[1][2], h_state[1][1], h_state[1][0]);
    delete[] h_state;
    EXPECT_STREQ(hex0, "1d01e56f49579cec72319e145f06f6177f6c5253206e78c2689781452a31878b");
    EXPECT_STREQ(hex1, "0d189ec589c41b8cffa88cfc523618a055abe8192c70f75aa72fc514560f6c61");
}

TEST(BN128_POSEIDON2_TEST, hash_gpu_t3) {
    
    Poseidon2BN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 3;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    Poseidon2BN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex2[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex2, sizeof(hex2), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[2][7], h_state[2][6], h_state[2][5], h_state[2][4], h_state[2][3], h_state[2][2], h_state[2][1], h_state[2][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0bb61d24daca55eebcb1929a82650f328134334da98ea4f847f760054f4a3033");
    EXPECT_STREQ(hex2, "1ed25194542b12eef8617361c3ba7c52e660b145994427cc86296242cf766ec8");
}

TEST(BN128_POSEIDON2_TEST, hash_gpu_t4) {
    
    Poseidon2BN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 4;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    Poseidon2BN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex3[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex3, sizeof(hex3), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[3][7], h_state[3][6], h_state[3][5], h_state[3][4], h_state[3][3], h_state[3][2], h_state[3][1], h_state[3][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "01bd538c2ee014ed5141b29e9ae240bf8db3fe5b9a38629a9647cf8d76c01737");
    EXPECT_STREQ(hex3, "2e11c5cff2a22c64d01304b778d78f6998eff1ab73163a35603f54794c30847a");
}

TEST(BN128_POSEIDON2_TEST, hash_gpu_t8) {
    
    Poseidon2BN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 8;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    Poseidon2BN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex7[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex7, sizeof(hex7), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[7][7], h_state[7][6], h_state[7][5], h_state[7][4], h_state[7][3], h_state[7][2], h_state[7][1], h_state[7][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "1d1a50bcde871247856df135d56a4ca61af575f1140ed9b1503c77528cf345df");
    EXPECT_STREQ(hex7, "0b19bfa00c8f1d505074130e7f8b49a8624b1905e280ceca5ba11099b081b265");
}

TEST(BN128_POSEIDON2_TEST, hash_gpu_t12) {
    
    Poseidon2BN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 12;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    Poseidon2BN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex11[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex11, sizeof(hex11), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[11][7], h_state[11][6], h_state[11][5], h_state[11][4], h_state[11][3], h_state[11][2], h_state[11][1], h_state[11][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "3014e0ec17029f7e4f5cfe8c7c54fc3df6a5f7539f6aa304b2f3c747a9105618");
    EXPECT_STREQ(hex11, "0905469a776b7d5a3f18841edb90fa0d8c6de479c2789c042dafefb367ad1a2b");
}

TEST(BN128_POSEIDON2_TEST, hash_gpu_t16) {
    
    Poseidon2BN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 16;    
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    Poseidon2BN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex15[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex15, sizeof(hex15), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[15][7], h_state[15][6], h_state[15][5], h_state[15][4], h_state[15][3], h_state[15][2], h_state[15][1], h_state[15][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0fc2e6b758f493969e1d860f9a44ee3bdffdf796f382aa4ffb16fa4e9bcc333f");
    EXPECT_STREQ(hex15, "0e2ceb1f8fde5f80be1f41bd239fabdc2f6133a6a98920a55c42891c3a925152");
}

// =====================
// Poseidon GPU Tests (t=2 to t=17)
// =====================

TEST(BN128_POSEIDON_TEST, hash_gpu_t2) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 2;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex1[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex1, sizeof(hex1), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[1][7], h_state[1][6], h_state[1][5], h_state[1][4], h_state[1][3], h_state[1][2], h_state[1][1], h_state[1][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "29176100eaa962bdc1fe6c654d6a3c130e96a4d1168b33848b897dc502820133");
    EXPECT_STREQ(hex1, "112a4f9241e384b0ede4655e6d2bbf7ebd9595775de9e7536df87cd487852fc4");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t3) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 3;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex2[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex2, sizeof(hex2), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[2][7], h_state[2][6], h_state[2][5], h_state[2][4], h_state[2][3], h_state[2][2], h_state[2][1], h_state[2][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "115cc0f5e7d690413df64c6b9662e9cf2a3617f2743245519e19607a4417189a");
    EXPECT_STREQ(hex2, "0e7ae82e40091e63cbd4f16a6d16310b3729d4b6e138fcf54110e2867045a30c");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t4) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 4;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex3[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex3, sizeof(hex3), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[3][7], h_state[3][6], h_state[3][5], h_state[3][4], h_state[3][3], h_state[3][2], h_state[3][1], h_state[3][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0e7732d89e6939c0ff03d5e58dab6302f3230e269dc5b968f725df34ab36d732");
    EXPECT_STREQ(hex3, "1a779bd9781d3a8354eae5ed74e7fa44fa0e458e45a1407524bddf3b9f2bf2d7");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t5) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 5;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex4[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex4, sizeof(hex4), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[4][7], h_state[4][6], h_state[4][5], h_state[4][4], h_state[4][3], h_state[4][2], h_state[4][1], h_state[4][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "299c867db6c1fdd79dcefa40e4510b9837e60ebb1ce0663dbaa525df65250465");
    EXPECT_STREQ(hex4, "07748bc6877c9b82c8b98666ee9d0626ec7f5be4205f79ee8528ef1c4a376fc7");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t6) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 6;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex5[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex5, sizeof(hex5), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[5][7], h_state[5][6], h_state[5][5], h_state[5][4], h_state[5][3], h_state[5][2], h_state[5][1], h_state[5][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0dab9449e4a1398a15224c0b15a49d598b2174d305a316c918125f8feeb123c0");
    EXPECT_STREQ(hex5, "208adf8d7f4ac061f00db710aef42f3b2f13176de26674b0a5f4436b883db6bc");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t7) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 7;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex6[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex6, sizeof(hex6), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[6][7], h_state[6][6], h_state[6][5], h_state[6][4], h_state[6][3], h_state[6][2], h_state[6][1], h_state[6][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "2d1a03850084442813c8ebf094dea47538490a68b05f2239134a4cca2f6302e1");
    EXPECT_STREQ(hex6, "2ac1d41181b675cbbfe7801457f882bfcd0d9994a37a6a105452b48a71f3c810");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t8) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 8;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex7[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex7, sizeof(hex7), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[7][7], h_state[7][6], h_state[7][5], h_state[7][4], h_state[7][3], h_state[7][2], h_state[7][1], h_state[7][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "1c2f3482dbb140c4ebb9ada49abdbc374a9a85fcfc6533ec2e9df45b4921c318");
    EXPECT_STREQ(hex7, "073534f0cedf2b30a870814eee062903ce751e545270c3cbfc5e4732c450ba9c");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t9) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 9;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex8[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex8, sizeof(hex8), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[8][7], h_state[8][6], h_state[8][5], h_state[8][4], h_state[8][3], h_state[8][2], h_state[8][1], h_state[8][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "2921ab9bd0140cbc98e40395c0fefb40337a4d54fbbecd9a4d43b3d8d0c4d8d1");
    EXPECT_STREQ(hex8, "2c8e23a3569963447e55619f1d1462f63ea2e40d3d405c18bbf394f13c253749");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t10) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 10;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex9[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex9, sizeof(hex9), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[9][7], h_state[9][6], h_state[9][5], h_state[9][4], h_state[9][3], h_state[9][2], h_state[9][1], h_state[9][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "1e0b893aa2ad802275e749d260330b7675b22bb3aaa4461d204af32e60cd9078");
    EXPECT_STREQ(hex9, "0315afa225921ebb807ba0f33feef2bb5b74c51b740b58faa205dc127e8aa7ac");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t11) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 11;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex10[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex10, sizeof(hex10), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[10][7], h_state[10][6], h_state[10][5], h_state[10][4], h_state[10][3], h_state[10][2], h_state[10][1], h_state[10][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0816126a09c29ecfcc0628461dacfb9459816fc60d6738b78db9ad07206fdc21");
    EXPECT_STREQ(hex10, "10f779eb86c66f6e316473976ca0b6b81e8c0c2cadf917ce84bf9cce1b72c45e");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t12) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 12;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex11[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex11, sizeof(hex11), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[11][7], h_state[11][6], h_state[11][5], h_state[11][4], h_state[11][3], h_state[11][2], h_state[11][1], h_state[11][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "07e5b070aa2dba008f30a6b785b6c5ae2429e211f71cacdbdae0e07fc05b47a8");
    EXPECT_STREQ(hex11, "1941a33364c6d1904c0e540b5170c73567d31cb038d5d6b83cd769412139321a");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t13) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 13;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex12[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex12, sizeof(hex12), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[12][7], h_state[12][6], h_state[12][5], h_state[12][4], h_state[12][3], h_state[12][2], h_state[12][1], h_state[12][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "058814945232937db248a01e7cc55b3d681cc08702c8168494e856c1ef7693b5");
    EXPECT_STREQ(hex12, "1a6df4eadbafbed2a14f78606ca1326f4bef58a348cffc2a0e8c050dab9cff94");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t14) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 14;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex13[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex13, sizeof(hex13), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[13][7], h_state[13][6], h_state[13][5], h_state[13][4], h_state[13][3], h_state[13][2], h_state[13][1], h_state[13][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0f918939632fadca6456a2fe6e65a124828d4c3920d379cc744e90a666887806");
    EXPECT_STREQ(hex13, "05a2ad96bd0cec0ed170ae830c1800d3e83a72d3fb84673213aab431fc578cb7");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t15) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 15;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex14[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex14, sizeof(hex14), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[14][7], h_state[14][6], h_state[14][5], h_state[14][4], h_state[14][3], h_state[14][2], h_state[14][1], h_state[14][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "1278779aaafc5ca58bf573151005830cdb4683fb26591c85a7464d4f0e527776");
    EXPECT_STREQ(hex14, "2c24786e78a255df1c1f11c09c5bea75c4ac1f96ad7978e6867f033363ed6bda");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t16) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 16;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex15[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex15, sizeof(hex15), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[15][7], h_state[15][6], h_state[15][5], h_state[15][4], h_state[15][3], h_state[15][2], h_state[15][1], h_state[15][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "094ae33b67a845998abb55e917642d4022d078d96f7c36ea11da4273ecf20f50");
    EXPECT_STREQ(hex15, "254e179b1f643318769c2480e0bdbc9f8e0aaeda3bb50be1284c184c0ce9d2a4");
}

TEST(BN128_POSEIDON_TEST, hash_gpu_t17) {
    PoseidonBN128GPU p;
    BN128GPUScalarField::Element* d_state = nullptr;
    BN128GPUScalarField::Element* h_state = nullptr;
    
    int t = 17;
    cudaMalloc(&d_state, t * sizeof(BN128GPUScalarField::Element));
    h_state = new BN128GPUScalarField::Element[t];
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    init_state_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    p.hash(d_state, t);
    cudaDeviceSynchronize();
    
    from_montgomery_kernel<<<1, 1>>>(d_state, t);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_state, d_state, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    
    char hex0[65], hex16[65];
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[0][7], h_state[0][6], h_state[0][5], h_state[0][4], h_state[0][3], h_state[0][2], h_state[0][1], h_state[0][0]);
    snprintf(hex16, sizeof(hex16), "%08x%08x%08x%08x%08x%08x%08x%08x",
             h_state[16][7], h_state[16][6], h_state[16][5], h_state[16][4], h_state[16][3], h_state[16][2], h_state[16][1], h_state[16][0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "16159a551cbb66108281a48099fff949ae08afd7f1f2ec06de2ffb96b919b765");
    EXPECT_STREQ(hex16, "0ffa1bd9b53dbedee9ab5742283c8968d0435c3b3a566fcb66ca61ce04a5b5bf");
}

// =====================
// Poseidon Parallel Hash GPU Tests
// =====================

// Test parallel hash for various t values (2-17)
TEST(BN128_POSEIDON_TEST, hash_parallel_gpu_all_t) {
    PoseidonBN128GPU p;
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    for (int t = 2; t <= 17; t++) {
        BN128GPUScalarField::Element* d_state_seq = nullptr;
        BN128GPUScalarField::Element* d_state_par = nullptr;
        BN128GPUScalarField::Element* h_state_seq = nullptr;
        BN128GPUScalarField::Element* h_state_par = nullptr;
        
        cudaMalloc(&d_state_seq, t * sizeof(BN128GPUScalarField::Element));
        cudaMalloc(&d_state_par, t * sizeof(BN128GPUScalarField::Element));
        h_state_seq = new BN128GPUScalarField::Element[t];
        h_state_par = new BN128GPUScalarField::Element[t];
        
        // Initialize both states with same values
        init_state_kernel<<<1, 1>>>(d_state_seq, t);
        init_state_kernel<<<1, 1>>>(d_state_par, t);
        cudaDeviceSynchronize();
        
        // Run sequential hash
        p.hash(d_state_seq, t);
        cudaDeviceSynchronize();
        
        // Run parallel hash
        p.hashParallel(d_state_par, t);
        cudaDeviceSynchronize();
        
        // Convert from Montgomery form
        from_montgomery_kernel<<<1, 1>>>(d_state_seq, t);
        from_montgomery_kernel<<<1, 1>>>(d_state_par, t);
        cudaDeviceSynchronize();
        
        // Copy results to host
        cudaMemcpy(h_state_seq, d_state_seq, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_state_par, d_state_par, t * sizeof(BN128GPUScalarField::Element), cudaMemcpyDeviceToHost);
        cudaFree(d_state_seq);
        cudaFree(d_state_par);
        
        // Compare all elements
        bool all_match = true;
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < 8; j++) {
                if (h_state_seq[i][j] != h_state_par[i][j]) {
                    all_match = false;
                    printf("t=%d: Mismatch at state[%d][%d]: seq=%08x, par=%08x\n", t, i, j, h_state_seq[i][j], h_state_par[i][j]);
                }
            }
        }
        
        delete[] h_state_seq;
        delete[] h_state_par;
        
        EXPECT_TRUE(all_match) << "Parallel hash mismatch for t=" << t;
    }
}

// =====================
// Poseidon linearHash GPU Test
// =====================

TEST(BN128_POSEIDON_TEST, linearHash_gpu_4rows_100cols) {
    const size_t rows = 4;
    const size_t cols = 100;
    int t = 17;
    
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    // Create trace: 4 rows × 100 cols Goldilocks elements (uint64_t)
    std::vector<uint64_t> trace(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        trace[i] = i;
    }
    
    uint64_t* d_input = nullptr;
    BN128GPUScalarField::Element* d_output = nullptr;
    CHECKCUDAERR(cudaMalloc(&d_input, rows * cols * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&d_output, rows * sizeof(BN128GPUScalarField::Element)));
    
    CHECKCUDAERR(cudaMemset(d_output, 0, rows * sizeof(BN128GPUScalarField::Element)));
    
    CHECKCUDAERR(cudaMemcpy(d_input, trace.data(), rows * cols * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    PoseidonBN128GPU::linearHash(d_output, d_input, cols, rows, t, false, 0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    
    RawFr::Element* h_output = new RawFr::Element[rows];
    CHECKCUDAERR(cudaMemcpy(h_output, d_output, rows * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < rows; i++) {
        std::string hex = RawFr::field.toString(h_output[i], 16);
        
        if (i == 0) EXPECT_EQ(hex, "f51f3d0104201ef2bf7424be924330f937e4504111c6b26f7c195afbcb9d6cd");
        if (i == 1) EXPECT_EQ(hex, "12b276d381cf64df6b1732ec6e91ae30f1f8c60cc08959aa9f81ae3b00cae371");
        if (i == 2) EXPECT_EQ(hex, "160dcdd70c78a86409e231df7f4add4e32c9a92fe74ac92dff516d3d8fa728");
        if (i == 3) EXPECT_EQ(hex, "2393e4f4bbaadaee5acaf60590547ef52c7dfd553f856283726497304ad47b5b");
    }
    
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

// Helper function to convert row-major trace to tiled layout (CPU version of getBufferOffset)
// Tile layout: TILE_HEIGHT=256 rows × TILE_WIDTH=4 cols per tile
// Within each tile, data is stored column-major with TILE_HEIGHT stride
static inline uint64_t getBufferOffsetCPU(uint64_t row, uint64_t col, uint64_t nRows, uint64_t nCols) {
    const uint64_t TILE_HEIGHT = 256;
    const uint64_t TILE_WIDTH = 4;
    
    uint64_t blockY = col / TILE_WIDTH;
    uint64_t blockX = row / TILE_HEIGHT;
    uint64_t nCols_block = (nCols - TILE_WIDTH * blockY < TILE_WIDTH) 
                           ? (nCols - TILE_WIDTH * blockY) : TILE_WIDTH;
    uint64_t col_block = col % TILE_WIDTH;
    uint64_t row_block = row % TILE_HEIGHT;

    return blockY * TILE_WIDTH * nRows + blockX * nCols_block * TILE_HEIGHT
           + col_block * TILE_HEIGHT + row_block;
}

// Test linearHashTiles with tiled layout - should produce same results as linearHash with row-major
TEST(BN128_POSEIDON_TEST, linearHashTiles_gpu_256rows_100cols) {
    const size_t rows = 256;  // Use TILE_HEIGHT to properly test tiled layout
    const size_t cols = 100;
    int t = 17;
    
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    std::vector<uint64_t> trace_rowmajor(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        trace_rowmajor[i] = i;
    }
    
    std::vector<uint64_t> trace_tiled(rows * cols);
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            uint64_t tiled_idx = getBufferOffsetCPU(row, col, rows, cols);
            trace_tiled[tiled_idx] = trace_rowmajor[row * cols + col];
        }
    }
    
    uint64_t *d_input_rowmajor = nullptr, *d_input_tiled = nullptr;
    BN128GPUScalarField::Element *d_output_rowmajor = nullptr, *d_output_tiled = nullptr;
    
    CHECKCUDAERR(cudaMalloc(&d_input_rowmajor, rows * cols * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&d_input_tiled, rows * cols * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&d_output_rowmajor, rows * sizeof(BN128GPUScalarField::Element)));
    CHECKCUDAERR(cudaMalloc(&d_output_tiled, rows * sizeof(BN128GPUScalarField::Element)));
    
    CHECKCUDAERR(cudaMemcpy(d_input_rowmajor, trace_rowmajor.data(), rows * cols * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(d_input_tiled, trace_tiled.data(), rows * cols * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    PoseidonBN128GPU::linearHash(d_output_rowmajor, d_input_rowmajor, cols, rows, t, false, 0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    
    PoseidonBN128GPU::linearHashTiles(d_output_tiled, d_input_tiled, cols, rows, t, false, 0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    
    std::vector<RawFr::Element> h_output_rowmajor(rows);
    std::vector<RawFr::Element> h_output_tiled(rows);
    CHECKCUDAERR(cudaMemcpy(h_output_rowmajor.data(), d_output_rowmajor, rows * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(h_output_tiled.data(), d_output_tiled, rows * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    
    int mismatches = 0;
    for (size_t i = 0; i < rows; i++) {
        const uint32_t* p_rowmajor = reinterpret_cast<const uint32_t*>(&h_output_rowmajor[i]);
        const uint32_t* p_tiled = reinterpret_cast<const uint32_t*>(&h_output_tiled[i]);
        
        bool match = true;
        for (int j = 0; j < 8; j++) {
            if (p_rowmajor[j] != p_tiled[j]) {
                match = false;
                break;
            }
        }
        
        if (!match) {
            mismatches++;
            if (mismatches <= 5) {  // Print first 5 mismatches
                std::string hex_rowmajor = RawFr::field.toString(h_output_rowmajor[i], 16);
                std::string hex_tiled = RawFr::field.toString(h_output_tiled[i], 16);
                printf("Row %zu MISMATCH:\n", i);
                printf("  rowmajor: %s\n", hex_rowmajor.c_str());
                printf("  tiled:    %s\n", hex_tiled.c_str());
            }
        }
    }
    
    EXPECT_EQ(mismatches, 0) << "Found " << mismatches << " mismatches between row-major and tiled results";
    
    cudaFree(d_input_rowmajor);
    cudaFree(d_input_tiled);
    cudaFree(d_output_rowmajor);
    cudaFree(d_output_tiled);
}

// =====================
// Poseidon merkletree GPU Test
// =====================

TEST(BN128_POSEIDON_TEST, merkletree_gpu_8rows_100cols) {
    const size_t rows = 8;
    const size_t cols = 100;
    const size_t arity = 16;
    const size_t numNodes = 17;
    
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    // Create trace: 8 rows × 100 cols Goldilocks elements (row-major)
    std::vector<uint64_t> trace(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        trace[i] = i;
    }
    
    uint64_t* d_input = nullptr;
    BN128GPUScalarField::Element* d_tree = nullptr;
    CHECKCUDAERR(cudaMalloc(&d_input, rows * cols * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&d_tree, numNodes * sizeof(BN128GPUScalarField::Element)));
    
    CHECKCUDAERR(cudaMemcpy(d_input, trace.data(), rows * cols * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemset(d_tree, 0, numNodes * sizeof(BN128GPUScalarField::Element)));
    
    // Run GPU merkletree
    PoseidonBN128GPU::merkletree(d_tree, d_input, cols, rows, arity, false, 0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    
    std::vector<RawFr::Element> h_tree(numNodes);
    CHECKCUDAERR(cudaMemcpy(h_tree.data(), d_tree, numNodes * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    
    // Verify leaves
    EXPECT_EQ(RawFr::field.toString(h_tree[0], 16), "f51f3d0104201ef2bf7424be924330f937e4504111c6b26f7c195afbcb9d6cd");
    EXPECT_EQ(RawFr::field.toString(h_tree[1], 16), "12b276d381cf64df6b1732ec6e91ae30f1f8c60cc08959aa9f81ae3b00cae371");
    EXPECT_EQ(RawFr::field.toString(h_tree[2], 16), "160dcdd70c78a86409e231df7f4add4e32c9a92fe74ac92dff516d3d8fa728");
    EXPECT_EQ(RawFr::field.toString(h_tree[3], 16), "2393e4f4bbaadaee5acaf60590547ef52c7dfd553f856283726497304ad47b5b");
    EXPECT_EQ(RawFr::field.toString(h_tree[4], 16), "69539124331a9758e99c6f6d9f4f86088a4e48aa51d643a87c85f2536820c91");
    EXPECT_EQ(RawFr::field.toString(h_tree[5], 16), "3e5fef46738269d441b69cb79ed05afbc4cb319e81cb8500da2be30800fd478");
    EXPECT_EQ(RawFr::field.toString(h_tree[6], 16), "26a631438c157a61dbfcf090f4da49ef1d7e05b3a0f1ace53df8aa5966334987");
    EXPECT_EQ(RawFr::field.toString(h_tree[7], 16), "17bcde7b057013734be16c97b6397967fb42c57f10f9a85dbe92436bf1446f60");
    
    // Verify root
    EXPECT_EQ(RawFr::field.toString(h_tree[numNodes - 1], 16), "b6d02c9e5ea48185580c371837a1ecc09d7cf40774e1ac6f8491819deb3824d");
    
    cudaFree(d_input);
    cudaFree(d_tree);
}

TEST(BN128_POSEIDON_TEST, merkletreeTiles_gpu_256rows_100cols) {
    const size_t rows = 256;  // Use TILE_HEIGHT to properly test tiled layout
    const size_t cols = 100;
    const size_t arity = 16;
    
    // numNodes for merkletree with arity 16: 
    // level 0: 256 leaves
    // level 1: ceil(256/16) = 16 nodes
    // level 2: ceil(16/16) = 1 node (root)
    // total: 256 + 16 + 1 = 273
    const size_t numNodes = 273;
    
    uint32_t gpu_idxs[] = {0};
    PoseidonBN128GPU::initGPUConstants(gpu_idxs, 1);
    
    // Create trace in row-major format: 256 rows × 100 cols
    std::vector<uint64_t> trace_rowmajor(rows * cols);
    for (size_t i = 0; i < rows * cols; i++) {
        trace_rowmajor[i] = i;
    }
    
    // Convert to tiled layout using getBufferOffsetCPU
    std::vector<uint64_t> trace_tiled(rows * cols);
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            uint64_t tiled_idx = getBufferOffsetCPU(row, col, rows, cols);
            trace_tiled[tiled_idx] = trace_rowmajor[row * cols + col];
        }
    }
    
    // Allocate GPU memory
    uint64_t *d_input_rowmajor = nullptr, *d_input_tiled = nullptr;
    BN128GPUScalarField::Element *d_tree_rowmajor = nullptr, *d_tree_tiled = nullptr;
    
    CHECKCUDAERR(cudaMalloc(&d_input_rowmajor, rows * cols * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&d_input_tiled, rows * cols * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&d_tree_rowmajor, numNodes * sizeof(BN128GPUScalarField::Element)));
    CHECKCUDAERR(cudaMalloc(&d_tree_tiled, numNodes * sizeof(BN128GPUScalarField::Element)));
    
    CHECKCUDAERR(cudaMemcpy(d_input_rowmajor, trace_rowmajor.data(), rows * cols * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(d_input_tiled, trace_tiled.data(), rows * cols * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemset(d_tree_rowmajor, 0, numNodes * sizeof(BN128GPUScalarField::Element)));
    CHECKCUDAERR(cudaMemset(d_tree_tiled, 0, numNodes * sizeof(BN128GPUScalarField::Element)));
    
    // Run GPU merkletree with row-major layout
    PoseidonBN128GPU::merkletree(d_tree_rowmajor, d_input_rowmajor, cols, rows, arity, false, 0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    
    // Run GPU merkletreeTiles with tiled layout
    PoseidonBN128GPU::merkletreeTiles(d_tree_tiled, d_input_tiled, cols, rows, arity, false, 0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    
    // Copy results back to host
    std::vector<RawFr::Element> h_tree_rowmajor(numNodes);
    std::vector<RawFr::Element> h_tree_tiled(numNodes);
    CHECKCUDAERR(cudaMemcpy(h_tree_rowmajor.data(), d_tree_rowmajor, numNodes * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(h_tree_tiled.data(), d_tree_tiled, numNodes * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    
    // Compare results
    int mismatches = 0;
    for (size_t i = 0; i < numNodes; i++) {
        const uint32_t* p_rowmajor = reinterpret_cast<const uint32_t*>(&h_tree_rowmajor[i]);
        const uint32_t* p_tiled = reinterpret_cast<const uint32_t*>(&h_tree_tiled[i]);
        
        bool match = true;
        for (int j = 0; j < 8; j++) {
            if (p_rowmajor[j] != p_tiled[j]) {
                match = false;
                break;
            }
        }
        
        if (!match) {
            mismatches++;
            if (mismatches <= 5) {  // Print first 5 mismatches
                std::string hex_rowmajor = RawFr::field.toString(h_tree_rowmajor[i], 16);
                std::string hex_tiled = RawFr::field.toString(h_tree_tiled[i], 16);
                printf("Node %zu MISMATCH:\n", i);
                printf("  rowmajor: %s\n", hex_rowmajor.c_str());
                printf("  tiled:    %s\n", hex_tiled.c_str());
            }
        }
    }
    
    EXPECT_EQ(mismatches, 0) << "Found " << mismatches << " mismatches between row-major and tiled merkletree results";
    
    
    cudaFree(d_input_rowmajor);
    cudaFree(d_input_tiled);
    cudaFree(d_tree_rowmajor);
    cudaFree(d_tree_tiled);
}

// =====================
// MSM (Multi-Scalar Multiplication) GPU Test
// =====================

TEST(BN128_MSM, msm) {
    // Use CPU curve for computing expected result
    AltBn128::G1PointAffine& G = AltBn128::G1.oneAffine();
    
    // Create points: [G, 2G, 4G, 8G]
    // With large 253-bit scalars (same as CPU test)
    // MSM result: s0*G + s1*(2G) + s2*(4G) + s3*(8G) = (s0 + 2*s1 + 4*s2 + 8*s3)*G
    const size_t npoints = 4;
    const size_t scalarSize = 32;  // 256-bit scalars
    PointAffineGPU* h_points = new PointAffineGPU[npoints];
    BN128GPUScalarField::Element* h_scalars = new BN128GPUScalarField::Element[npoints];
    
    // Compute points: G, 2G, 4G, 8G using CPU
    AltBn128::G1Point P;
    AltBn128::G1PointAffine P_affine;
    AltBn128::G1.copy(P, G);  // P = G
    
    for (size_t i = 0; i < npoints; i++) {
        AltBn128::G1.copy(P_affine, P);
        memcpy(&h_points[i].x, &P_affine.x, sizeof(AltBn128::F1Element));
        memcpy(&h_points[i].y, &P_affine.y, sizeof(AltBn128::F1Element));
        AltBn128::G1.dbl(P, P);
    }
    
    // Large 253-bit scalars (same as CPU multiexp test)
    const char* scalarStrs[4] = {
        "5708990770823839524233143877797980545530985996",
        "8563486156235759286349715816696970818296478975",
        "9234567890123456789012345678901234567890123456",
        "10876543210987654321098765432109876543210987654"
    };
    
    // Parse scalars and convert to little-endian format for GPU
    AltBn128::FrElement rawScalars[4];
    for (size_t i = 0; i < npoints; i++) {
        AltBn128::Fr.fromString(rawScalars[i], scalarStrs[i], 10);
        // Fr stores in Montgomery form internally, but we need raw LE for GPU
        // Convert to big-endian bytes, then reverse to little-endian
        uint8_t beBytes[scalarSize];
        AltBn128::Fr.toRprBE(rawScalars[i], beBytes, scalarSize);
        // Reverse to little-endian (GPU MSM expects LE)
        uint8_t leBytes[scalarSize];
        for (size_t j = 0; j < scalarSize; j++) {
            leBytes[j] = beBytes[scalarSize - 1 - j];
        }
        memcpy(&h_scalars[i], leBytes, scalarSize);
    }
    
    // ========== GPU MSM ==========
    PointJacobianGPU gpu_result;
    memset(&gpu_result, 0, sizeof(gpu_result));
    
    MSM_BN128_GPU::msm(gpu_result, h_points, h_scalars, npoints, false);
    
    // ========== CPU verification ==========
    // Compute combined scalar: s0 + 2*s1 + 4*s2 + 8*s3
    AltBn128::FrElement combinedScalar;
    AltBn128::Fr.fromUI(combinedScalar, 0);
    
    for (size_t i = 0; i < npoints; i++) {
        AltBn128::FrElement powerOfTwo;
        AltBn128::Fr.fromUI(powerOfTwo, 1ULL << i);
        AltBn128::FrElement term;
        AltBn128::Fr.mul(term, rawScalars[i], powerOfTwo);
        AltBn128::Fr.add(combinedScalar, combinedScalar, term);
    }
    
    // Convert combined scalar to little-endian bytes
    uint8_t combinedBE[scalarSize];
    AltBn128::Fr.toRprBE(combinedScalar, combinedBE, scalarSize);
    uint8_t combinedLE[scalarSize];
    for (size_t j = 0; j < scalarSize; j++) {
        combinedLE[j] = combinedBE[scalarSize - 1 - j];
    }
    
    // Compute expected: (s0 + 2*s1 + 4*s2 + 8*s3) * G
    AltBn128::G1Point cpu_result;
    AltBn128::G1.mulByScalar(cpu_result, G, combinedLE, scalarSize);
    
    // Convert CPU result to affine
    AltBn128::G1PointAffine cpu_affine;
    AltBn128::G1.copy(cpu_affine, cpu_result);
    
    // Convert GPU result (Jacobian) to affine using CPU field operations
    // GPU Jacobian: affine_x = X/Z^2, affine_y = Y/Z^3
    
    AltBn128::F1Element gpu_X, gpu_Y, gpu_Z;
    memcpy(&gpu_X, &gpu_result.X, sizeof(AltBn128::F1Element));
    memcpy(&gpu_Y, &gpu_result.Y, sizeof(AltBn128::F1Element));
    memcpy(&gpu_Z, &gpu_result.Z, sizeof(AltBn128::F1Element));
    
    // Compute Z^2 and Z^3
    AltBn128::F1Element z2, z3, z2_inv, z3_inv;
    AltBn128::F1.square(z2, gpu_Z);
    AltBn128::F1.mul(z3, z2, gpu_Z);
    
    // Compute inverses
    AltBn128::F1.inv(z2_inv, z2);
    AltBn128::F1.inv(z3_inv, z3);
    
    // Compute affine coordinates
    AltBn128::F1Element gpu_affine_x, gpu_affine_y;
    AltBn128::F1.mul(gpu_affine_x, gpu_X, z2_inv);
    AltBn128::F1.mul(gpu_affine_y, gpu_Y, z3_inv);
    
    // Compare with CPU affine result
    bool x_eq = AltBn128::F1.eq(gpu_affine_x, cpu_affine.x);
    bool y_eq = AltBn128::F1.eq(gpu_affine_y, cpu_affine.y);
    
    EXPECT_TRUE(x_eq) << "GPU X coordinate does not match CPU";
    EXPECT_TRUE(y_eq) << "GPU Y coordinate does not match CPU";
    
    if (!x_eq || !y_eq) {
        // Print both results for debugging
        const uint32_t* gx = reinterpret_cast<const uint32_t*>(&gpu_affine_x);
        const uint32_t* gy = reinterpret_cast<const uint32_t*>(&gpu_affine_y);
        const uint32_t* cx = reinterpret_cast<const uint32_t*>(&cpu_affine.x);
        const uint32_t* cy = reinterpret_cast<const uint32_t*>(&cpu_affine.y);
        
        printf("GPU affine X = %08x%08x%08x%08x%08x%08x%08x%08x\n",
               gx[7], gx[6], gx[5], gx[4], gx[3], gx[2], gx[1], gx[0]);
        printf("CPU affine X = %08x%08x%08x%08x%08x%08x%08x%08x\n",
               cx[7], cx[6], cx[5], cx[4], cx[3], cx[2], cx[1], cx[0]);
        printf("GPU affine Y = %08x%08x%08x%08x%08x%08x%08x%08x\n",
               gy[7], gy[6], gy[5], gy[4], gy[3], gy[2], gy[1], gy[0]);
        printf("CPU affine Y = %08x%08x%08x%08x%08x%08x%08x%08x\n",
               cy[7], cy[6], cy[5], cy[4], cy[3], cy[2], cy[1], cy[0]);
    }
    
    delete[] h_points;
    delete[] h_scalars;
}

// =====================
// NTT GPU Tests
// =====================

TEST(BN128_NTT_GPU_TEST, ntt_then_intt_roundtrip) {
    // Test: NTT followed by INTT should recover the original data
    const uint32_t lg_n = 4;
    const uint64_t n = 1ULL << lg_n;
    
    // Use CPU field to initialize data properly
    RawFr field;
    
    // Allocate host memory (RawFr::Element has same layout as GPU element)
    RawFr::Element* h_data = new RawFr::Element[n];
    RawFr::Element* h_original = new RawFr::Element[n];
    
    // Initialize data: data[i] = i
    for (uint64_t i = 0; i < n; i++) {
        field.fromUI(h_data[i], i);
        field.copy(h_original[i], h_data[i]);
    }
    
    // Cast to GPU type (same memory layout)
    BN128GPUScalarField::Element* gpu_data = reinterpret_cast<BN128GPUScalarField::Element*>(h_data);
    
    // Apply NTT then INTT
    NTT_BN128_GPU::ntt(gpu_data, lg_n);
    NTT_BN128_GPU::intt(gpu_data, lg_n);
        
    // Verify result matches original
    bool all_match = true;
    for (uint64_t i = 0; i < n; i++) {
        if (!field.eq(h_data[i], h_original[i])) {
            all_match = false;
            printf("Mismatch at index %lu: expected %s, got %s\n", i,
                   field.toString(h_original[i], 10).c_str(),
                   field.toString(h_data[i], 10).c_str());
        }
    }
    
    EXPECT_TRUE(all_match) << "NTT->INTT roundtrip failed";
    
    delete[] h_data;
    delete[] h_original;
}

TEST(BN128_NTT_GPU_TEST, intt_then_ntt_roundtrip) {
    // Test: INTT followed by NTT should recover the original data
    const uint32_t lg_n = 4;
    const uint64_t n = 1ULL << lg_n;
    
    RawFr field;
    
    RawFr::Element* h_data = new RawFr::Element[n];
    RawFr::Element* h_original = new RawFr::Element[n];
    
    // Initialize data: data[i] = i
    for (uint64_t i = 0; i < n; i++) {
        field.fromUI(h_data[i], i);
        field.copy(h_original[i], h_data[i]);
    }
    
    BN128GPUScalarField::Element* gpu_data = reinterpret_cast<BN128GPUScalarField::Element*>(h_data);
    
    // Apply INTT then NTT
    NTT_BN128_GPU::intt(gpu_data, lg_n);
    NTT_BN128_GPU::ntt(gpu_data, lg_n);
    
    // Verify result matches original
    bool all_match = true;
    for (uint64_t i = 0; i < n; i++) {
        if (!field.eq(h_data[i], h_original[i])) {
            all_match = false;
            printf("Mismatch at index %lu: expected %s, got %s\n", i,
                   field.toString(h_original[i], 10).c_str(),
                   field.toString(h_data[i], 10).c_str());
        }
    }
    
    EXPECT_TRUE(all_match) << "INTT->NTT roundtrip failed";
    
    delete[] h_data;
    delete[] h_original;
}

TEST(BN128_NTT_GPU_TEST, ntt_linearity) {
    // Test: NTT(a + b) == NTT(a) + NTT(b)  (NTT is a linear operation)
    const uint32_t lg_n = 4;
    const uint64_t n = 1ULL << lg_n;
    
    RawFr field;
    
    RawFr::Element* h_a = new RawFr::Element[n];
    RawFr::Element* h_b = new RawFr::Element[n];
    RawFr::Element* h_a_plus_b = new RawFr::Element[n];
    
    // Initialize vectors a and b
    for (uint64_t i = 0; i < n; i++) {
        field.fromUI(h_a[i], i + 1);           // a = [1, 2, 3, ..., 16]
        field.fromUI(h_b[i], (i * 7) % 13);    // b = different pattern
        field.add(h_a_plus_b[i], h_a[i], h_b[i]);
    }
    
    // Cast to GPU type
    BN128GPUScalarField::Element* gpu_a = reinterpret_cast<BN128GPUScalarField::Element*>(h_a);
    BN128GPUScalarField::Element* gpu_b = reinterpret_cast<BN128GPUScalarField::Element*>(h_b);
    BN128GPUScalarField::Element* gpu_a_plus_b = reinterpret_cast<BN128GPUScalarField::Element*>(h_a_plus_b);
    
    // Compute NTT(a), NTT(b), NTT(a+b)
    NTT_BN128_GPU::ntt(gpu_a, lg_n);
    NTT_BN128_GPU::ntt(gpu_b, lg_n);
    NTT_BN128_GPU::ntt(gpu_a_plus_b, lg_n);
    
    // Verify: NTT(a+b) == NTT(a) + NTT(b)
    bool all_match = true;
    for (uint64_t i = 0; i < n; i++) {
        RawFr::Element expected_sum;
        field.add(expected_sum, h_a[i], h_b[i]);
        
        if (!field.eq(h_a_plus_b[i], expected_sum)) {
            all_match = false;
            printf("Linearity failed at index %lu\n", i);
        }
    }
    
    EXPECT_TRUE(all_match) << "NTT linearity test failed";
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_a_plus_b;
}

TEST(BN128_NTT_GPU_TEST, ntt_gpu_vs_cpu) {
    // Test: GPU NTT result should match CPU FFT result
    RawFr field;
    const uint32_t lg_n = 4;
    const uint64_t n = 1ULL << lg_n;
    
    // Allocate memory
    RawFr::Element* h_gpu_data = new RawFr::Element[n];
    std::vector<RawFr::Element> cpu_data(n);
    
    // Initialize both with same data
    for (uint64_t i = 0; i < n; i++) {
        field.fromUI(cpu_data[i], i);
        field.copy(h_gpu_data[i], cpu_data[i]);
    }
    
    // Run GPU NTT
    BN128GPUScalarField::Element* gpu_data = reinterpret_cast<BN128GPUScalarField::Element*>(h_gpu_data);
    NTT_BN128_GPU::ntt(gpu_data, lg_n);
    
    // Run CPU FFT
    FFT<RawFr> fft(n);
    fft.fft(cpu_data.data(), n);
    
    // Compare results
    bool all_match = true;
    for (uint64_t i = 0; i < n; i++) {
        if (!field.eq(h_gpu_data[i], cpu_data[i])) {
            all_match = false;
            printf("Index %lu mismatch:\n", i);
            printf("  GPU: %s\n", field.toString(h_gpu_data[i], 16).c_str());
            printf("  CPU: %s\n", field.toString(cpu_data[i], 16).c_str());
        }
    }
    
    EXPECT_TRUE(all_match) << "GPU NTT does not match CPU FFT";
    
    delete[] h_gpu_data;
}

// =====================
// Grinding GPU Tests
// =====================

// Note: CPU verification of GPU grinding results is done in the CPU test suite
// (tests.cpp grinding_cpu test). These GPU tests verify the kernel execution
// completes successfully and finds the same nonce as the CPU test.

TEST(BN128_POSEIDON_GPU_TEST, grinding_gpu) {
    // Initialize GPU constants
    uint32_t gpu_id = 0;
    PoseidonBN128GPU::initGPUConstants(&gpu_id, 1);
    
    const uint8_t n_bits = 8;
    
    // Create input state with 3 elements
    // Note: fromUI already converts to Montgomery form
    RawFr field;
    RawFr::Element h_state[3];
    field.fromUI(h_state[0], 0x1234567890abcdefULL);
    field.fromUI(h_state[1], 0xfedcba0987654321ULL);
    field.fromUI(h_state[2], 0x0123456789abcdefULL);
    
    // Allocate device memory
    PoseidonBN128GPU::FrElement *d_state;
    uint64_t *d_nonce;
    uint64_t *d_nonceBlock;
    
    CHECKCUDAERR(cudaMalloc(&d_state, 3 * sizeof(PoseidonBN128GPU::FrElement)));
    CHECKCUDAERR(cudaMalloc(&d_nonce, sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&d_nonceBlock, 256 * sizeof(uint64_t)));  // GRINDING_GRID_SIZE
    
    // Copy state to device
    CHECKCUDAERR(cudaMemcpy(d_state, h_state, 3 * sizeof(PoseidonBN128GPU::FrElement), cudaMemcpyHostToDevice));
    
    // Initialize nonce to not found
    uint64_t init_nonce = UINT64_MAX;
    CHECKCUDAERR(cudaMemcpy(d_nonce, &init_nonce, sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // Run GPU grinding
    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    
    PoseidonBN128GPU::grinding(d_nonce, d_nonceBlock, d_state, n_bits, stream);
    
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    
    // Copy result back
    uint64_t h_nonce;
    CHECKCUDAERR(cudaMemcpy(&h_nonce, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    // Verify we found a valid nonce
    ASSERT_NE(h_nonce, UINT64_MAX) << "GPU grinding did not find a nonce";
    
    // Verify the nonce matches the CPU test expectation (CPU/GPU parity)
    EXPECT_EQ(h_nonce, 1530ULL) << "GPU nonce does not match CPU expected nonce";
    
    // Cleanup
    CHECKCUDAERR(cudaStreamDestroy(stream));
    CHECKCUDAERR(cudaFree(d_state));
    CHECKCUDAERR(cudaFree(d_nonce));
    CHECKCUDAERR(cudaFree(d_nonceBlock));
}

// =====================
// Device Pointer NTT/MSM Tests
// =====================

// Extern declarations for device pointer functions
extern "C" void ntt_bn128_gpu_dev_ptr(void* d_data, uint32_t lg_n);
extern "C" void intt_bn128_gpu_dev_ptr(void* d_data, uint32_t lg_n);
extern "C" void msm_bn128_gpu_dev_ptr(void* out, const void* d_points, const void* d_scalars, size_t npoints, bool mont);

// Also declare the host-pointer versions for comparison
extern "C" void ntt_bn128_gpu(void* data, uint32_t lg_n);
extern "C" void intt_bn128_gpu(void* data, uint32_t lg_n);
extern "C" void msm_bn128_gpu(void* out, const void* points, const void* scalars, size_t npoints, bool mont);

TEST(BN128_NTT_DEV_PTR, ntt_dev_ptr_vs_host_ptr) {
    // Test: NTT with device pointer should produce same result as host pointer version
    const uint32_t lg_n = 10;
    const uint64_t n = 1ULL << lg_n;
    
    RawFr field;
    
    RawFr::Element* h_data_host = new RawFr::Element[n];
    RawFr::Element* h_data_dev = new RawFr::Element[n];
    
    for (uint64_t i = 0; i < n; i++) {
        field.fromUI(h_data_host[i], i + 1);
        field.copy(h_data_dev[i], h_data_host[i]);
    }
    
    ntt_bn128_gpu(h_data_host, lg_n);
    
    void* d_data;
    CHECKCUDAERR(cudaMalloc(&d_data, n * sizeof(RawFr::Element)));
    CHECKCUDAERR(cudaMemcpy(d_data, h_data_dev, n * sizeof(RawFr::Element), cudaMemcpyHostToDevice));
    
    ntt_bn128_gpu_dev_ptr(d_data, lg_n);
    
    CHECKCUDAERR(cudaMemcpy(h_data_dev, d_data, n * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(d_data));
    
    // Compare
    bool all_match = true;
    for (uint64_t i = 0; i < n; i++) {
        if (!field.eq(h_data_host[i], h_data_dev[i])) {
            all_match = false;
            printf("NTT dev_ptr mismatch at index %lu: host=%s, dev=%s\n", i,
                   field.toString(h_data_host[i], 10).c_str(),
                   field.toString(h_data_dev[i], 10).c_str());
            if (i > 5) break;  // Limit output
        }
    }
    
    EXPECT_TRUE(all_match) << "NTT device pointer version differs from host pointer version";
    
    delete[] h_data_host;
    delete[] h_data_dev;
}

TEST(BN128_NTT_DEV_PTR, intt_dev_ptr_vs_host_ptr) {
    const uint32_t lg_n = 10;
    const uint64_t n = 1ULL << lg_n;
    
    RawFr field;
    
    RawFr::Element* h_data_host = new RawFr::Element[n];
    RawFr::Element* h_data_dev = new RawFr::Element[n];
    
    for (uint64_t i = 0; i < n; i++) {
        field.fromUI(h_data_host[i], i + 1);
        field.copy(h_data_dev[i], h_data_host[i]);
    }
    
    intt_bn128_gpu(h_data_host, lg_n);
    
    void* d_data;
    CHECKCUDAERR(cudaMalloc(&d_data, n * sizeof(RawFr::Element)));
    CHECKCUDAERR(cudaMemcpy(d_data, h_data_dev, n * sizeof(RawFr::Element), cudaMemcpyHostToDevice));
    
    intt_bn128_gpu_dev_ptr(d_data, lg_n);
    
    CHECKCUDAERR(cudaMemcpy(h_data_dev, d_data, n * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(d_data));
    
    // Compare
    bool all_match = true;
    for (uint64_t i = 0; i < n; i++) {
        if (!field.eq(h_data_host[i], h_data_dev[i])) {
            all_match = false;
            printf("INTT dev_ptr mismatch at index %lu\n", i);
            if (i > 5) break;
        }
    }
    
    EXPECT_TRUE(all_match) << "INTT device pointer version differs from host pointer version";
    
    delete[] h_data_host;
    delete[] h_data_dev;
}

TEST(BN128_NTT_DEV_PTR, ntt_intt_roundtrip_dev_ptr) {
    // Test: NTT followed by INTT using device pointers should recover original data
    const uint32_t lg_n = 12;
    const uint64_t n = 1ULL << lg_n;
    
    RawFr field;
    
    RawFr::Element* h_original = new RawFr::Element[n];
    RawFr::Element* h_result = new RawFr::Element[n];
    
    for (uint64_t i = 0; i < n; i++) {
        field.fromUI(h_original[i], (i * 17 + 5) % 1000000);
    }
    
    void* d_data;
    CHECKCUDAERR(cudaMalloc(&d_data, n * sizeof(RawFr::Element)));
    CHECKCUDAERR(cudaMemcpy(d_data, h_original, n * sizeof(RawFr::Element), cudaMemcpyHostToDevice));
    
    ntt_bn128_gpu_dev_ptr(d_data, lg_n);
    intt_bn128_gpu_dev_ptr(d_data, lg_n);
    
    CHECKCUDAERR(cudaMemcpy(h_result, d_data, n * sizeof(RawFr::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(d_data));
    
    // Verify roundtrip
    bool all_match = true;
    for (uint64_t i = 0; i < n; i++) {
        if (!field.eq(h_original[i], h_result[i])) {
            all_match = false;
            printf("Roundtrip mismatch at index %lu: expected=%s, got=%s\n", i,
                   field.toString(h_original[i], 10).c_str(),
                   field.toString(h_result[i], 10).c_str());
            if (i > 5) break;
        }
    }
    
    EXPECT_TRUE(all_match) << "NTT->INTT roundtrip with device pointers failed";
    
    delete[] h_original;
    delete[] h_result;
}

TEST(BN128_MSM_DEV_PTR, msm_dev_ptr_vs_host_ptr) {
    // Test: MSM with device pointers should produce same result as host pointer version
    AltBn128::G1PointAffine& G = AltBn128::G1.oneAffine();
    
    const size_t npoints = 64;
    const size_t scalarSize = 32;
    
    PointAffineGPU* h_points = new PointAffineGPU[npoints];
    BN128GPUScalarField::Element* h_scalars = new BN128GPUScalarField::Element[npoints];
    
    // Generate points: G, 2G, 3G, ..., nG
    AltBn128::G1Point P;
    AltBn128::G1PointAffine P_affine;
    AltBn128::G1.copy(P, G);
    
    for (size_t i = 0; i < npoints; i++) {
        AltBn128::G1.copy(P_affine, P);
        memcpy(&h_points[i].x, &P_affine.x, sizeof(AltBn128::F1Element));
        memcpy(&h_points[i].y, &P_affine.y, sizeof(AltBn128::F1Element));
        AltBn128::G1Point temp;
        AltBn128::G1.add(temp, P, G);
        AltBn128::G1.copy(P, temp);
    }
    
    // Generate scalars: small values for simplicity (in little-endian)
    for (size_t i = 0; i < npoints; i++) {
        AltBn128::FrElement scalar;
        AltBn128::Fr.fromUI(scalar, (i + 1) * 1000);
        
        uint8_t beBytes[scalarSize];
        AltBn128::Fr.toRprBE(scalar, beBytes, scalarSize);
        uint8_t leBytes[scalarSize];
        for (size_t j = 0; j < scalarSize; j++) {
            leBytes[j] = beBytes[scalarSize - 1 - j];
        }
        memcpy(&h_scalars[i], leBytes, scalarSize);
    }
    
    // Run host-pointer version
    PointJacobianGPU host_result;
    memset(&host_result, 0, sizeof(host_result));
    msm_bn128_gpu(&host_result, h_points, h_scalars, npoints, false);
    
    // Allocate device memory
    void* d_points;
    void* d_scalars;
    CHECKCUDAERR(cudaMalloc(&d_points, npoints * sizeof(PointAffineGPU)));
    CHECKCUDAERR(cudaMalloc(&d_scalars, npoints * sizeof(BN128GPUScalarField::Element)));
    CHECKCUDAERR(cudaMemcpy(d_points, h_points, npoints * sizeof(PointAffineGPU), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(d_scalars, h_scalars, npoints * sizeof(BN128GPUScalarField::Element), cudaMemcpyHostToDevice));
    
    // Run device-pointer version
    PointJacobianGPU dev_result;
    memset(&dev_result, 0, sizeof(dev_result));
    msm_bn128_gpu_dev_ptr(&dev_result, d_points, d_scalars, npoints, false);
    
    CHECKCUDAERR(cudaFree(d_points));
    CHECKCUDAERR(cudaFree(d_scalars));
    
    // Compare Jacobian coordinates
    bool match = true;
    
    // Compare X, Y, Z fields
    AltBn128::F1Element host_X, host_Y, host_Z;
    AltBn128::F1Element dev_X, dev_Y, dev_Z;
    memcpy(&host_X, &host_result.X, sizeof(AltBn128::F1Element));
    memcpy(&host_Y, &host_result.Y, sizeof(AltBn128::F1Element));
    memcpy(&host_Z, &host_result.Z, sizeof(AltBn128::F1Element));
    memcpy(&dev_X, &dev_result.X, sizeof(AltBn128::F1Element));
    memcpy(&dev_Y, &dev_result.Y, sizeof(AltBn128::F1Element));
    memcpy(&dev_Z, &dev_result.Z, sizeof(AltBn128::F1Element));
    
    // Convert both to affine and compare
    AltBn128::F1Element host_z2, host_z3, dev_z2, dev_z3;
    AltBn128::F1.square(host_z2, host_Z);
    AltBn128::F1.mul(host_z3, host_z2, host_Z);
    AltBn128::F1.square(dev_z2, dev_Z);
    AltBn128::F1.mul(dev_z3, dev_z2, dev_Z);
    
    AltBn128::F1Element host_z2_inv, host_z3_inv, dev_z2_inv, dev_z3_inv;
    AltBn128::F1.inv(host_z2_inv, host_z2);
    AltBn128::F1.inv(host_z3_inv, host_z3);
    AltBn128::F1.inv(dev_z2_inv, dev_z2);
    AltBn128::F1.inv(dev_z3_inv, dev_z3);
    
    AltBn128::F1Element host_affine_x, host_affine_y, dev_affine_x, dev_affine_y;
    AltBn128::F1.mul(host_affine_x, host_X, host_z2_inv);
    AltBn128::F1.mul(host_affine_y, host_Y, host_z3_inv);
    AltBn128::F1.mul(dev_affine_x, dev_X, dev_z2_inv);
    AltBn128::F1.mul(dev_affine_y, dev_Y, dev_z3_inv);
    
    if (!AltBn128::F1.eq(host_affine_x, dev_affine_x)) {
        match = false;
        printf("MSM dev_ptr X mismatch\n");
    }
    if (!AltBn128::F1.eq(host_affine_y, dev_affine_y)) {
        match = false;
        printf("MSM dev_ptr Y mismatch\n");
    }
    
    EXPECT_TRUE(match) << "MSM device pointer version differs from host pointer version";
    
    delete[] h_points;
    delete[] h_scalars;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
