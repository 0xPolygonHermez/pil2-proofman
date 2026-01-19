
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "bn128.cuh"
#include "fq.cuh"
#include "poseidon2_bn128.cuh"
#include "msm_bn128.cuh"
#include "ntt_bn128.cuh"
#include "point.cuh"
#include "alt_bn128.hpp"
#include "fft.hpp"

__global__ void kernel_fr_add_one_one(int* ok);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__global__ void kernel_fr_add_one_one(int* ok)
{
    BN128GPUScalarField::Element a;
    BN128GPUScalarField::Element b;
    BN128GPUScalarField::Element r;

    a.v[0] = 1;
    b.v[0] = 1;
    for(int i = 1; i < 8; ++i) {
        a.v[i] = 0;
        b.v[i] = 0;
    }    
    a.v.to(); // Convert to montgomery form
    b.v.to(); // Convert to montgomery form

    BN128GPUScalarField::add(r, a, b);

    r.v.from(); // Convert back from montgomery form
    
    bool same = true;    
    for(int i = 0; i < 8; ++i) {
        uint32_t expected = (i == 0) ? 2 : 0;
        if(r.v[i] != expected) {
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

    a.v[0] = 1;
    b.v[0] = 1;
    for(int i = 1; i < 8; ++i) {
        a.v[i] = 0;
        b.v[i] = 0;
    }    
    a.v.to(); // Convert to montgomery form
    b.v.to(); // Convert to montgomery form

    BN128GPUBaseField::add(r, a, b);

    r.v.from(); // Convert back from montgomery form
    
    bool same = true;    
    for(int i = 0; i < 8; ++i) {
        uint32_t expected = (i == 0) ? 2 : 0;
        if(r.v[i] != expected) {
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
        state[i].v[0] = i;
        for (int j = 1; j < 8; j++) {
            state[i].v[j] = 0;
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
    
    // Use pointer cast to access the raw uint32_t values 
    char hex0[65], hex1[65]; //64 hex chars + 1 null
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&h_state[0].v);
    const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&h_state[1].v);
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p0[7], p0[6], p0[5], p0[4], p0[3], p0[2], p0[1], p0[0]);
    snprintf(hex1, sizeof(hex1), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p1[7], p1[6], p1[5], p1[4], p1[3], p1[2], p1[1], p1[0]);
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
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&h_state[0].v);
    const uint32_t* p2 = reinterpret_cast<const uint32_t*>(&h_state[2].v);
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p0[7], p0[6], p0[5], p0[4], p0[3], p0[2], p0[1], p0[0]);
    snprintf(hex2, sizeof(hex2), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p2[7], p2[6], p2[5], p2[4], p2[3], p2[2], p2[1], p2[0]);
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
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&h_state[0].v);
    const uint32_t* p3 = reinterpret_cast<const uint32_t*>(&h_state[3].v);
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p0[7], p0[6], p0[5], p0[4], p0[3], p0[2], p0[1], p0[0]);
    snprintf(hex3, sizeof(hex3), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p3[7], p3[6], p3[5], p3[4], p3[3], p3[2], p3[1], p3[0]);
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
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&h_state[0].v);
    const uint32_t* p7 = reinterpret_cast<const uint32_t*>(&h_state[7].v);
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p0[7], p0[6], p0[5], p0[4], p0[3], p0[2], p0[1], p0[0]);
    snprintf(hex7, sizeof(hex7), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p7[7], p7[6], p7[5], p7[4], p7[3], p7[2], p7[1], p7[0]);
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
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&h_state[0].v);
    const uint32_t* p11 = reinterpret_cast<const uint32_t*>(&h_state[11].v);
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p0[7], p0[6], p0[5], p0[4], p0[3], p0[2], p0[1], p0[0]);
    snprintf(hex11, sizeof(hex11), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p11[7], p11[6], p11[5], p11[4], p11[3], p11[2], p11[1], p11[0]);
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
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&h_state[0].v);
    const uint32_t* p15 = reinterpret_cast<const uint32_t*>(&h_state[15].v);
    snprintf(hex0, sizeof(hex0), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p0[7], p0[6], p0[5], p0[4], p0[3], p0[2], p0[1], p0[0]);
    snprintf(hex15, sizeof(hex15), "%08x%08x%08x%08x%08x%08x%08x%08x",
             p15[7], p15[6], p15[5], p15[4], p15[3], p15[2], p15[1], p15[0]);
    delete[] h_state;
    
    EXPECT_STREQ(hex0, "0fc2e6b758f493969e1d860f9a44ee3bdffdf796f382aa4ffb16fa4e9bcc333f");
    EXPECT_STREQ(hex15, "0e2ceb1f8fde5f80be1f41bd239fabdc2f6133a6a98920a55c42891c3a925152");
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
    AltBn128::F1Element z2, z3, z_inv, z2_inv, z3_inv;
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
    RawFrP field;
    
    // Allocate host memory (RawFrP::Element has same layout as GPU element)
    RawFrP::Element* h_data = new RawFrP::Element[n];
    RawFrP::Element* h_original = new RawFrP::Element[n];
    
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
    
    RawFrP field;
    
    RawFrP::Element* h_data = new RawFrP::Element[n];
    RawFrP::Element* h_original = new RawFrP::Element[n];
    
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
    
    RawFrP field;
    
    RawFrP::Element* h_a = new RawFrP::Element[n];
    RawFrP::Element* h_b = new RawFrP::Element[n];
    RawFrP::Element* h_a_plus_b = new RawFrP::Element[n];
    
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
        RawFrP::Element expected_sum;
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
    RawFrP field;
    const uint32_t lg_n = 4;
    const uint64_t n = 1ULL << lg_n;
    
    // Allocate memory
    RawFrP::Element* h_gpu_data = new RawFrP::Element[n];
    std::vector<RawFrP::Element> cpu_data(n);
    
    // Initialize both with same data
    for (uint64_t i = 0; i < n; i++) {
        field.fromUI(cpu_data[i], i);
        field.copy(h_gpu_data[i], cpu_data[i]);
    }
    
    // Run GPU NTT
    BN128GPUScalarField::Element* gpu_data = reinterpret_cast<BN128GPUScalarField::Element*>(h_gpu_data);
    NTT_BN128_GPU::ntt(gpu_data, lg_n);
    
    // Run CPU FFT
    FFT<RawFrP> fft(n);
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



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
